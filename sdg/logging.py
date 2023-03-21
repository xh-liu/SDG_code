# Copyright (C) 2021 NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
import datetime
import os
import torch

from torch.utils.tensorboard import SummaryWriter

from .distributed import master_only, is_master
from .distributed import master_only_print as print
from .distributed import dist_all_reduce_tensor
from .misc import to_cuda
import pdb


def get_date_uid():
    """Generate a unique id based on date.
    Returns:
        str: Return uid string, e.g. '20171122171307111552'.
    """
    return str(datetime.datetime.now().strftime("%Y_%m%d_%H%M_%S"))


def init_logging(exp_name, root_dir='logs', timestamp=False):
    r"""Create log directory for storing checkpoints and output images.

    Args:
        config_path (str): Path to the configuration file.
        logdir (str): Log directory name
    Returns:
        str: Return log dir
    """
    #config_file = os.path.basename(config_path)
    if timestamp:
        date_uid = get_date_uid()
        exp_name = '_'.join([exp_name, date_uid])
    # example: logs/2019_0125_1047_58_spade_cocostuff
    #log_file = '_'.join([date_uid, os.path.splitext(config_file)[0]])
    #log_file = os.path.splitext(config_file)[0]
    logdir = os.path.join(root_dir, exp_name)
    return logdir


@master_only
def make_logging_dir(logdir, no_tb=False):
    r"""Create the logging directory

    Args:
        logdir (str): Log directory name
    """
    print('Make folder {}'.format(logdir))
    os.makedirs(logdir, exist_ok=True)
    if no_tb:
        return None
    tensorboard_dir = os.path.join(logdir, 'tensorboard')
    os.makedirs(tensorboard_dir, exist_ok=True)
    tb_log = SummaryWriter(log_dir=tensorboard_dir)
    return tb_log

def write_tb(tb_log, key, value, step):
    if not torch.is_tensor(value):
        value = torch.tensor(value)
    if not value.is_cuda:
        value = to_cuda(value)
    value = dist_all_reduce_tensor(value.mean()).item()
    if is_master():
        tb_log.add_scalar(key, value, step)
    print('%s: %f  ' % (key, value), end='')
