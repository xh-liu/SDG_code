"""
Finetune a noised CLIP image encoder on the target dataset without text annotations.
"""

import argparse
import os
import time

import blobfile as bf
import numpy as np
import torch as th
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils import tensorboard

from sgd.parser import create_argparser
from sgd.logging import init_logging, make_logging_dir
from sgd.distributed import init_dist, is_master, get_world_size
from sgd.distributed import master_only_print as print
from sgd.distributed import dist_all_gather_tensor, all_gather_with_gradient
from sgd.gpu_affinity import set_affinity
from sgd.image_datasets import load_data
from sgd.resample import create_named_schedule_sampler
from sgd.script_util import (
    add_dict_to_argparser,
    args_to_dict,
    classifier_and_diffusion_defaults,
    create_clip_and_diffusion,
)
from sgd.train_util import parse_resume_step_from_filename, log_loss_dict
from sgd.misc import set_random_seed
from sgd.misc import to_cuda


def main():
    args = create_argparser().parse_args()

    set_affinity(args.local_rank)
    if args.randomized_seed:
        args.seed = random.randint(0, 10000)
    set_random_seed(args.seed, by_rank=True)
    if not args.single_gpu:
        init_dist(args.local_rank)
    tb_log = None
    args.logdir = init_logging(args.exp_name)
    if is_master():
        tb_log = make_logging_dir(args.logdir)
    world_size = get_world_size()

    print("creating model and diffusion...")
    model, diffusion = create_clip_and_diffusion(
        args, **args_to_dict(args, classifier_and_diffusion_defaults().keys())
    )
    model.to('cuda')
    if args.noised:
        schedule_sampler = create_named_schedule_sampler(
            args.schedule_sampler, diffusion
        )

    resume_step = 0
    if args.resume_checkpoint:
        resume_step = parse_resume_step_from_filename(args.resume_checkpoint)
        print(f"loading model from checkpoint: {args.resume_checkpoint}... at {resume_step} step")
        model.load_state_dict(th.load(args.resume_checkpoint, map_location=lambda storage, loc: storage))
    model.to('cuda')

    if args.use_fp16:
        if args.fp16_hyperparams == 'pytorch':
            scaler = th.cuda.amp.GradScaler(init_scale=65536.0, growth_factor=2.0, backoff_factor=0.5, growth_interval=2000, enabled=True)
        elif args.fp16_hyperparams == 'openai':
            scaler = th.cuda.amp.GradScaler(init_scale=65536.0, growth_factor=2**0.001, backoff_factor=0.5, growth_interval=1, enabled=True)


    use_ddp = th.cuda.is_available() and th.distributed.is_available() and dist.is_initialized()
    if use_ddp:
        ddp_model = DDP(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
            bucket_cap_mb=128,
            find_unused_parameters=False,
        )
    else:
        print("Single GPU Training without DistributedDataParallel. ")
        ddp_model = model

    print("creating data loader...")
    args.return_text = False
    args.return_class = False
    args.return_yall = False
    train_dataloader = load_data(args)

    print(f"creating optimizer...")
    opt = AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
    if args.resume_checkpoint:
        opt_checkpoint = bf.join(
            bf.dirname(args.resume_checkpoint), f"opt{resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            print(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            opt.load_state_dict(th.load(opt_checkpoint, map_location=lambda storage, loc: storage))
        else:
            print('Warning: opt checkpoint %s not found' % opt_checkpoint)
        sca_checkpoint = bf.join(
            bf.dirname(args.resume_checkpoint), f"scaler{resume_step:06}.pt"
        )
        if bf.exists(sca_checkpoint):
            print(f"loading optimizer state from checkpoint: {sca_checkpoint}")
            scaler.load_state_dict(th.load(sca_checkpoint, map_location=lambda storage, loc: storage))
        else:
            print('Warning: opt checkpoint %s not found' % opt_checkpoint)



    print("training classifier model...")

    import clip
    clip_pretrained, _ = clip.load('RN50x16', jit=False)
    clip_pretrained = clip_pretrained.float()
    clip_pretrained.eval()
    clip_pretrained = clip_pretrained.cuda()

    def forward_backward_log(data_loader, prefix="train", step=0):
        batch, batch2 = next(data_loader)

        batch = to_cuda(batch)
        batch2 = to_cuda(batch2)
        # Noisy images
        if args.noised:
            t, _ = schedule_sampler.sample(batch.shape[0], 'cuda')
            batch = diffusion.q_sample(batch, t)
        else:
            t = th.zeros(batch.shape[0], dtype=th.long, device='cuda')

        ground_truth = th.arange(batch.shape[0] * world_size, dtype=th.long, device='cuda')
        for i, (sub_batch, sub_batch2, sub_t) in enumerate(
            split_microbatches(args.microbatch, batch, batch2, t)
        ):
            with th.cuda.amp.autocast(args.use_fp16):
                with th.no_grad():
                    sub_labels = clip_pretrained.encode_image(sub_batch2)
                if args.structure == 'classifier':
                    image_features = ddp_model(sub_batch, timesteps=sub_t)
                    text_features = sub_labels
                else:
                    image_features, text_features = ddp_model(sub_batch, sub_labels, sub_t)

                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                losses = {}

                image_features = all_gather_with_gradient(image_features)
                text_features = all_gather_with_gradient(text_features)
                logits_per_image = 100.0 * image_features @ text_features.t()
                logits_per_text = logits_per_image.t()

                loss_i2t = F.cross_entropy(logits_per_image, ground_truth, reduction='none')
                loss_t2i = F.cross_entropy(logits_per_text, ground_truth, reduction='none')
                loss = loss_i2t + loss_t2i
                losses[f"{prefix}_loss_i2t"] = loss_i2t.detach()
                losses[f"{prefix}_loss_t2i"] = loss_t2i.detach()
            

            losses[f"{prefix}_loss"] = loss.detach()
 
            log_loss_dict(diffusion, sub_t, losses, tb_log, step)
            del losses
            loss = loss.mean()
            if loss.requires_grad:
                if i == 0:
                    opt.zero_grad()
            if args.use_fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()

    global_batch = args.batch_size * world_size
    for step in range(args.iterations - resume_step):
        print("***step %d  " % (step + resume_step), end='')
        num_samples = (step + resume_step + 1) * global_batch
        print('samples: %d  ' % num_samples, end='')
        if args.anneal_lr:
            set_annealed_lr(opt, args.lr, args.lr_anneal_steps, step + resume_step)
        if is_master():
            tb_log.add_scalar('status/step', step + resume_step, step)
            tb_log.add_scalar('status/samples', num_samples, step)
            tb_log.add_scalar('status/lr', args.lr, step)
        forward_backward_log(train_dataloader, step=step)
        if args.use_fp16:
            scaler.step(opt)
            scaler.update()
        else:
            opt.step()
        if not step % args.log_interval and is_master():
            tb_log.flush()
        if not (step + resume_step) % args.save_interval:
            print("saving model...")
            save_model(args.logdir, model, opt, scaler, step + resume_step)

    print("saving model...")
    if is_master():
        save_model(args.logdir, model, opt, step + resume_step)
        tb_log.close()


def set_annealed_lr(opt, base_lr, anneal_steps, current_steps):
    lr_decay_cnt = current_steps // anneal_steps
    lr = base_lr * 0.1**lr_decay_cnt
    for param_group in opt.param_groups:
        param_group["lr"] = lr


def save_model(logdir, model, opt, scaler, step):
    if is_master():
        th.save(model.state_dict(), os.path.join(logdir, f"model{step:06d}.pt"))
        th.save(opt.state_dict(), os.path.join(logdir, f"opt{step:06d}.pt"))
        th.save(scaler.state_dict(), os.path.join(logdir, f"scaler{step:06d}.pt"))


def compute_top_k(logits, labels, k, reduction="mean"):
    _, top_ks = th.topk(logits, k, dim=-1)
    if reduction == "mean":
        return (top_ks == labels[:, None]).float().sum(dim=-1).mean().item()
    elif reduction == "none":
        return (top_ks == labels[:, None]).float().sum(dim=-1)


def split_microbatches(microbatch, *args):
    bs = len(args[0])
    if microbatch == -1 or microbatch >= bs:
        yield tuple(args)
    else:
        for i in range(0, bs, microbatch):
            yield tuple(x[i : i + microbatch] if x is not None else None for x in args)


if __name__ == "__main__":
    main()
