from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from torch import rot90
from torchvision.transforms import RandomVerticalFlip
from torchvision.transforms.v2 import (
    Compose,
    ToImage,
    ToDtype,
    RandomVerticalFlip,
    RandomCrop,
    CenterCrop,
    RandomRotation,
    RandomApply,
    Resize,
)
from torchvision.transforms.v2 import RandomHorizontalFlip
from torchvision.transforms.v2.functional import hflip

from frdc.load.dataset import FRDCDataset

THIS_DIR = Path(__file__).parent

BANDS = ["NB", "NG", "NR", "RE", "NIR"]


class FRDCDatasetStaticEval(FRDCDataset):
    def __len__(self):
        """Assume that the dataset is 8x larger than it actually is.

        There are 8 possible orientations for each image.
        1.       As-is
        2, 3, 4. Rotated 90, 180, 270 degrees
        5.       Horizontally flipped
        6, 7, 8. Horizontally flipped and rotated 90, 180, 270 degrees
        """
        return super().__len__() * 8

    def __getitem__(self, idx):
        """Alter the getitem method to implement the logic above."""
        x, y = super().__getitem__(int(idx // 8))
        assert x.ndim == 3, "x must be a 3D tensor"
        x_ = None
        if idx % 8 == 0:
            x_ = x
        elif idx % 8 == 1:
            x_ = rot90(x, 1, (1, 2))
        elif idx % 8 == 2:
            x_ = rot90(x, 2, (1, 2))
        elif idx % 8 == 3:
            x_ = rot90(x, 3, (1, 2))
        elif idx % 8 == 4:
            x_ = hflip(x)
        elif idx % 8 == 5:
            x_ = hflip(rot90(x, 1, (1, 2)))
        elif idx % 8 == 6:
            x_ = hflip(rot90(x, 2, (1, 2)))
        elif idx % 8 == 7:
            x_ = hflip(rot90(x, 3, (1, 2)))

        return x_, y


def n_times(f, n: int):
    return lambda x: [f(x) for _ in range(n)]


def n_rand_weak_aug(size, n_aug: int = 2):
    return n_times(rand_weak_aug(size), n_aug)


def n_rand_strong_aug(size, n_aug: int = 2):
    return n_times(rand_strong_aug(size), n_aug)


def n_rand_weak_strong_aug(size, n_aug: int = 2):
    def f(x):
        # x_weak = [weak_0, weak_1, ..., weak_n]
        x_weak = n_rand_weak_aug(size, n_aug)(x)
        # x_strong = [strong_0, strong_1, ..., strong_n]
        x_strong = n_rand_strong_aug(size, n_aug)(x)
        # x_paired = [(weak_0, strong_0), (weak_1, strong_1),
        #             ..., (weak_n, strong_n)]
        x_paired = list(zip(*[x_weak, x_strong]))
        return x_paired

    return f


def rand_weak_aug(size: int):
    return Compose(
        [
            ToImage(),
            ToDtype(torch.float32, scale=True),
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
            RandomApply([RandomRotation((90, 90))], p=0.5),
            Resize(size, antialias=True),
            CenterCrop(size),
        ]
    )


def const_weak_aug(size: int):
    return Compose(
        [
            ToImage(),
            ToDtype(torch.float32, scale=True),
            Resize(size, antialias=True),
            CenterCrop(size),
        ]
    )


def rand_strong_aug(size: int):
    return Compose(
        [
            ToImage(),
            ToDtype(torch.float32, scale=True),
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
            RandomApply([RandomRotation((90, 90))], p=0.5),
            Resize(size, antialias=True),
            RandomCrop(size, pad_if_needed=False),  # Strong
        ]
    )
