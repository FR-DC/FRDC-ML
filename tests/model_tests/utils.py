from __future__ import annotations

from pathlib import Path

import torch
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

THIS_DIR = Path(__file__).parent

BANDS = ["NB", "NG", "NR", "RE", "NIR"]


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
            Resize(size, antialias=True),
            CenterCrop(size),
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
            RandomApply([RandomRotation((90, 90))], p=0.5),
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
            Resize(size, antialias=True),
            RandomCrop(size, pad_if_needed=False),
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
            RandomApply([RandomRotation((90, 90))], p=0.5),
        ]
    )
