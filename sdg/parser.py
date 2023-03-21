"""
Train a diffusion model on images.
"""
import os
import argparse

from sdg.resample import create_named_schedule_sampler
from sdg.script_util import (
    model_and_diffusion_defaults,
    args_to_dict,
    add_dict_to_argparser,
)


def create_argparser():
    parser = argparse.ArgumentParser()
    # basic
    parser.add_argument('--exp_name', required=True)
    parser.add_argument('--resume_checkpoint', default=None)
    parser.add_argument('--local_rank', type=int, default=os.getenv('LOCAL_RANK', 0))
    parser.add_argument('--single_gpu', action='store_true')
    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--randomized_seed', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--logdir', type=str)

    # data
    parser.add_argument('--data_dir', type=str, default='')
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--deterministic', type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--random_crop', type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--random_flip', type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--clip_model', type=str, default='ViT-B-16')

    # model
    parser.add_argument('--class_cond', type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--text_cond', type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--num_channels', type=int, default=128)
    parser.add_argument('--num_res_blocks', type=int, default=2)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--num_heads_upsample', type=int, default=-1)
    parser.add_argument('--num_head_channels', type=int, default=-1)
    parser.add_argument('--attention_resolutions', type=str, default="16,8")
    parser.add_argument('--channel_mult', default="")
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--use_checkpoint', type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--use_scale_shift_norm', type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument('--resblock_updown', type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--use_new_attention_order', type=lambda x: (str(x).lower() == 'true'), default=False)

    # diffusion
    parser.add_argument('--learn_sigma', type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--diffusion_steps', type=int, default=1000)
    parser.add_argument('--noise_schedule', type=str, default='linear')
    parser.add_argument('--timestep_respacing', default='')
    parser.add_argument('--use_kl', type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--predict_xstart', type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--rescale_timesteps',type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--rescale_learned_sigmas', type=lambda x: (str(x).lower() == 'true'), default=False)

    # classifier
    parser.add_argument('--classifier_use_fp16', type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--classifier_width', type=int, default=128)
    parser.add_argument('--classifier_depth', type=int, default=2)
    parser.add_argument('--classifier_attention_resolutions', type=str, default="32,16,8")
    parser.add_argument('--classifier_use_scale_shift_norm', type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument('--classifier_resblock_updown', type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument('--classifier_pool', type=str, default="attention")
    parser.add_argument('--num_classes', type=int, default=1000)

    # sr
    parser.add_argument('--large_size', type=int, default=256)
    parser.add_argument('--small_size', type=int, default=64)

    # train
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--microbatch', type=int, default=-1)
    parser.add_argument('--schedule_sampler', type=str, default='uniform')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--lr_anneal_steps', type=int, default=0)
    parser.add_argument('--use_fp16', type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--fp16_scale_growth', type=float, default=1e-3)
    parser.add_argument('--fp16_hyperparams', type=str, default='openai')
    parser.add_argument('--anneal_lr', type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--iterations', type=int, default=500000)

    # save
    parser.add_argument('--ema_rate', default='0.9999')
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--save_interval', type=int, default=10000)
    parser.add_argument('--eval_interval', type=int, default=10)

    # inference
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--use_ddim', type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--clip_denoised', type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument('--text_weight', type=float, default=1.0)
    parser.add_argument('--image_weight', type=float, default=1.0)
    parser.add_argument('--image_loss', type=str, default='semantic')
    parser.add_argument('--text_instruction_file', type=str, default='ref/ffhq_instructions.txt')
    parser.add_argument('--clip_path', type=str, default='')


    # train classifier/clip
    parser.add_argument('--noised', type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument('--finetune_clip_layer', type=str, default='all')

    # superres
    parser.add_argument('--base_name', type=str, default='')


    return parser

