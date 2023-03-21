import math
import random
import os
import pickle

from PIL import Image
import blobfile as bf
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
import clip
from sdg.distributed import master_only_print as print
import torchvision.transforms as transforms
import json
import io


def load_ref_data(args, ref_img_path=None):
    if ref_img_path is None:
        ref_img_path = args.ref_img_path
    with bf.BlobFile(ref_img_path, "rb") as f:
        pil_image = Image.open(f)
        pil_image.load()

    pil_image = pil_image.convert("RGB")
    arr = center_crop_arr(pil_image, args.image_size)
    arr = arr.astype(np.float32) / 127.5 - 1
    arr = np.repeat(np.expand_dims(np.transpose(arr, [2, 0, 1]), axis=0), args.batch_size, axis=0)
    kwargs = {}
    kwargs["ref_img"] = torch.tensor(arr)
    return kwargs

def load_data(args, is_train=True, swap=False):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """
    if not args.data_dir:
        raise ValueError("unspecified data directory")
    data_dir = getattr(args, 'data_dir')
    batch_size = getattr(args, 'batch_size')
    image_size = getattr(args, 'image_size')
    class_cond = getattr(args, 'class_cond', False)
    deterministic = getattr(args, 'deterministic', False)
    random_crop = getattr(args, 'random_crop', False)
    random_flip = getattr(args, 'random_flip', True)
    if not is_train:
        deterministic = True
        random_crop = False
        random_flip = False
    num_workers = getattr(args, 'num_workers', 4)
    all_files = _list_image_files_recursively(data_dir)
    classes = None
    if class_cond:
        class_names = [bf.basename(path).split("_")[0] for path in all_files]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        classes = [sorted_classes[x] for x in class_names]
    dataset = ImageDataset(
        image_size,
        all_files,
        classes=classes,
        random_crop=random_crop,
        random_flip=random_flip,
    )
    not_distributed = args.single_gpu or args.debug or not dist.is_initialized()
    if not_distributed:
        sampler = None
    else:
        if deterministic:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False)
        else:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True)
    if deterministic or not not_distributed:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, sampler=sampler, num_workers=num_workers, drop_last=False
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, sampler=sampler, num_workers=num_workers, drop_last=False
        )
    while True:
        yield from loader

   

def load_ref_data(args, ref_img_path=None):
    if ref_img_path is None:
        ref_img_path = args.ref_img_path
    with bf.BlobFile(ref_img_path, "rb") as f:
        pil_image = Image.open(f)
        pil_image.load()

    pil_image = pil_image.convert("RGB")

    arr = center_crop_arr(pil_image, args.image_size)

    arr = arr.astype(np.float32) / 127.5 - 1

    arr = np.repeat(np.expand_dims(np.transpose(arr, [2, 0, 1]), axis=0), args.batch_size, axis=0)
    
    kwargs = {}

    kwargs["ref_img"] = torch.tensor(arr)

    return kwargs

def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


class ImageDataset(Dataset):
    def __init__(
        self,
        resolution,
        image_paths,
        classes=None,
        random_crop=False,
        random_flip=True,
    ):
        super().__init__()
        self.resolution = resolution
        self.images = image_paths
        self.classes = None if classes is None else classes
        self.random_crop = random_crop
        self.random_flip = random_flip

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        path = self.images[idx]
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("RGB")

        if self.random_crop:
            arr = random_crop_arr(pil_image, self.resolution)
        else:
            arr = center_crop_arr(pil_image, self.resolution)

        if self.random_flip and random.random() < 0.5:
            arr = arr[:, ::-1]

        arr = arr.astype(np.float32) / 127.5 - 1

        out_dict = {}
        if self.classes is not None:
            out_dict["y"] = np.array(self.classes[idx], dtype=np.int64)
        return np.transpose(arr, [2, 0, 1]), out_dict

def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def random_crop_arr(pil_image, image_size, min_crop_frac=0.85, max_crop_frac=0.95):
    if min(*pil_image.size) != max(*pil_image.size):
        min_crop_frac = 1.0
        max_crop_frac = 1.0
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)

    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)

    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]
