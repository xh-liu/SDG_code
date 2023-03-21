import copy
import functools
import os
import time
import glob

import blobfile as bf
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW

from .distributed import get_world_size, is_master
from .distributed import master_only_print as print
from .fp16_util import MixedPrecisionTrainer
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler
from .logging import write_tb
from .misc import to_cuda
from . import logger

# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0


class TrainLoop:
    def __init__(
        self,
        cfg,
        model,
        diffusion,
        data_train,
        tb_log,
        schedule_sampler=None,
    ):
        self.tb_log = tb_log
        self.model = model
        self.diffusion = diffusion
        self.data_train = data_train
        self.batch_size = cfg.batch_size
        self.microbatch = cfg.microbatch if cfg.microbatch > 0 else cfg.batch_size
        self.lr = cfg.lr
        self.ema_rate = (
            [cfg.ema_rate]
            if isinstance(cfg.ema_rate, float)
            else [float(x) for x in cfg.ema_rate.split(",")]
        )
        self.log_interval = cfg.log_interval
        self.save_interval = cfg.save_interval
        self.resume_checkpoint = cfg.resume_checkpoint
        self.use_fp16 = getattr(cfg, 'use_fp16', False)
        self.fp16_scale_growth = getattr(cfg, 'fp16_scale_growth', 1e-3)
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = getattr(cfg, 'weight_decay', 0.0)
        self.lr_anneal_steps = getattr(cfg, 'lr_anneal_steps', 0)
        self.logdir = getattr(cfg, 'logdir', 'logs/debug')
        fp16_hyperparams = getattr(cfg, 'fp16_hyperparams', 'openai')
        if self.use_fp16:
            if fp16_hyperparams == 'pytorch':
                self.scaler = th.cuda.amp.GradScaler(init_scale=65536.0, growth_factor=2.0, backoff_factor=0.5, growth_interval=2000, enabled=True)
            elif fp16_hyperparams == 'openai':
                self.scaler = th.cuda.amp.GradScaler(init_scale=2**20 * 1.0, growth_factor=2**0.001, backoff_factor=0.5, growth_interval=1, enabled=True)

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * get_world_size()

        self.sync_cuda = True  # th.cuda.is_available()

        self._load_and_sync_parameters()

        self.params = list(self.model.parameters())
        self.opt = AdamW(self.params, lr=self.lr, weight_decay=self.weight_decay)
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.params)
                for _ in range(len(self.ema_rate))
            ]

        self.use_ddp = th.cuda.is_available() and th.distributed.is_available() and dist.is_initialized()
        if self.use_ddp:
            self.ddp_model = DDP(
                self.model,
                device_ids=[cfg.local_rank],
                output_device=cfg.local_rank,
                bucket_cap_mb=128,
                broadcast_buffers=False,
                find_unused_parameters=False,
            )
        else:
            print(
                "Single GPU Training without DistributedDataParallel. "
            )
            self.ddp_model = self.model

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint(self.logdir) or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            print(f"loading model from checkpoint: {resume_checkpoint}...")
            checkpoint = th.load(resume_checkpoint, map_location=lambda storage, loc: storage)
            self.model.load_state_dict(checkpoint)

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.params)

        main_checkpoint = find_resume_checkpoint(self.logdir) or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            print(f"loading EMA from checkpoint: {ema_checkpoint}...")
            state_dict = th.load(ema_checkpoint, map_location=lambda storage, loc: storage)
            ema_params = to_cuda([state_dict[name] for name, _ in self.model.named_parameters()])

        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint(self.logdir) or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            print(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            checkpoint = th.load(opt_checkpoint, map_location=lambda storage, loc: storage)
            self.opt.load_state_dict(checkpoint)

    def _compute_norms(self, grad_scale=1.0):
        grad_norm = 0.0
        param_norm = 0.0
        for p in self.params:
            with th.no_grad():
                param_norm += th.norm(p, p=2, dtype=th.float32) ** 2
                if p.grad is not None:
                    grad_norm += th.norm(p.grad, p=2, dtype=th.float32) ** 2
        return th.sqrt(grad_norm) / grad_scale, th.sqrt(param_norm)

    def run_loop(self):
        time0 = time.time()
        while (
            not self.lr_anneal_steps
            or self.step + self.resume_step < self.lr_anneal_steps
        ):
            print('***step %d  ' % (self.step + self.resume_step), end='')
            batch, cond = next(self.data_train)
            time2 = time.time()
            if self.use_fp16:
                self.run_step_amp(batch, cond)
            else:
                self.run_step(batch, cond)
            if self.step % self.log_interval == 0 and is_master():
                self.tb_log.flush()
            if self.step % self.save_interval == 0:
                self.save()
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            num_samples = (self.step + self.resume_step + 1) * self.global_batch
            time1 = time.time()
            if is_master():
                self.tb_log.add_scalar('status/step', self.step+self.resume_step, self.step + self.resume_step)
                self.tb_log.add_scalar('status/samples', num_samples, self.step + self.resume_step)
                self.tb_log.add_scalar('time/time_per_iter', time1-time0, self.step + self.resume_step)
                self.tb_log.add_scalar('time/data_time_per_iter', time2-time0, self.step + self.resume_step)
                self.tb_log.add_scalar('time/model_time_per_iter', time1-time2, self.step + self.resume_step)
                self.tb_log.add_scalar('status/lr', self.lr, self.step + self.resume_step)
            print('lr: %f  ' % self.lr, end='')
            print('samples: %d  ' % num_samples, end='')
            print('data time: %f  ' % (time2-time0), end='')
            print('model time: %f  ' % (time1-time2), end='')
            print('')
            self.step += 1
            time0 = time1
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        self.opt.step()
        self._update_ema()
        self._anneal_lr()

    def forward_backward(self, batch, cond):
        self.opt.zero_grad()
        for i in range(0, batch.shape[0], self.microbatch):
            micro = to_cuda(batch[i : i + self.microbatch])
            micro_cond = {
                k: to_cuda(v[i : i + self.microbatch])
                for k, v in cond.items()
            }
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], 'cuda')

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,
                t,
                model_kwargs=micro_cond,
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()
            if self.step % 10 == 0:
                log_loss_dict(
                    self.diffusion, t, {k: v * weights for k, v in losses.items()},
                    self.tb_log, self.step + self.resume_step
                )
            loss.backward()

    def run_step_amp(self, batch, cond):
        self.forward_backward_amp(batch, cond)
        self.scaler.step(self.opt)
        self.scaler.update()
        self._update_ema()
        self._anneal_lr()

    def forward_backward_amp(self, batch, cond):
        self.opt.zero_grad()
        for i in range(0, batch.shape[0], self.microbatch):
            micro = to_cuda(batch[i : i + self.microbatch])
            micro_cond = {
                k: to_cuda(v[i : i + self.microbatch])
                for k, v in cond.items()
            }
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], 'cuda')

            with th.cuda.amp.autocast(True):
                compute_losses = functools.partial(
                    self.diffusion.training_losses,
                    self.ddp_model,
                    micro,
                    t,
                    model_kwargs=micro_cond,
                )

                if last_batch or not self.use_ddp:
                    losses = compute_losses()
                else:
                    with self.ddp_model.no_sync():
                        losses = compute_losses()

                if isinstance(self.schedule_sampler, LossAwareSampler):
                    self.schedule_sampler.update_with_local_losses(
                        t, losses["loss"].detach()
                    )

                loss = (losses["loss"] * weights).mean()
                if self.step % 10 == 0:
                    log_loss_dict(
                        self.diffusion, t, {k: v * weights for k, v in losses.items()},
                        self.tb_log, self.step + self.resume_step
                    )
                self.scaler.scale(loss).backward()



    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.params, rate=rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def save(self):
        def save_checkpoint(rate):
            print(f"saving model {rate}...")
            if is_master():
                if not rate:
                    filename = f"model{(self.step+self.resume_step):06d}.pt"
                else:
                    filename = f"ema_{rate}_{(self.step+self.resume_step):06d}.pt"
                with bf.BlobFile(bf.join(self.logdir, filename), "wb") as f:
                    th.save(self.model.state_dict(), f)

        save_checkpoint(0)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate)

        if is_master():
            with bf.BlobFile(
                bf.join(self.logdir, f"opt{(self.step+self.resume_step):06d}.pt"),
                "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)

        if is_master() and self.use_fp16:
            with bf.BlobFile(
                bf.join(self.logdir, f"scaler{(self.step+self.resume_step):06d}.pt"),
                "wb",
            ) as f:
                th.save(self.scaler.state_dict(), f)


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint(logdir):
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    models = sorted(glob.glob(os.path.join(logdir, 'model*.pt')))
    if len(models) >= 1:
        return models[-1]
    else:
        return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses, tb_log, step, prefix='loss'):
    for key, values in losses.items():
        write_tb(tb_log, f"{prefix}/{key}", values, step)
        quartile_list = [[] for cnt in range(4)]
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            quartile_list[quartile].append(sub_loss)
        for cnt in range(4):
            if len(quartile_list[cnt]) != 0:
                write_tb(tb_log, f"{prefix}/{key}_q{cnt}", sum(quartile_list[cnt])/len(quartile_list[cnt]), step)
            else:
                write_tb(tb_log, f"{prefix}/{key}_q{cnt}", 0.0, step)
