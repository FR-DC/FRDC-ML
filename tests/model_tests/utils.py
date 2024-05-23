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

from frdc.load.dataset import FRDCDataset

THIS_DIR = Path(__file__).parent

BANDS = ["NB", "NG", "NR", "RE", "NIR"]


class FRDCDatasetFlipped(FRDCDataset):
    def __len__(self):
        """Assume that the dataset is 4x larger than it actually is.

        For example, for index 0, we return the original image. For index 1, we
        return the horizontally flipped image and so on, until index 3.
        Then, return the next image for index 4, and so on.
        """
        return super().__len__() * 4

    def __getitem__(self, idx):
        """Alter the getitem method to implement the logic above."""
        x, y = super().__getitem__(int(idx // 4))
        if idx % 4 == 0:
            return x, y
        elif idx % 4 == 1:
            return RandomHorizontalFlip(p=1)(x), y
        elif idx % 4 == 2:
            return RandomVerticalFlip(p=1)(x), y
        elif idx % 4 == 3:
            return RandomHorizontalFlip(p=1)(RandomVerticalFlip(p=1)(x)), y


def val_preprocess(size: int):
    return lambda x: Compose(
        [
            ToImage(),
            ToDtype(torch.float32, scale=True),
            Resize(size, antialias=True),
            CenterCrop(size),
        ]
    )(x)


def n_weak_aug(size, n_aug: int = 2):
    return lambda x: (
        [weak_aug(size)(x) for _ in range(n_aug)] if n_aug > 0 else None
    )


def weak_aug(size: int):
    return lambda x: Compose(
        [
            ToImage(),
            ToDtype(torch.float32, scale=True),
            Resize(size, antialias=True),
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
            RandomApply([RandomRotation((90, 90))], p=0.5),
        ]
    )(x)


def strong_aug(size: int):
    return lambda x: Compose(
        [
            ToImage(),
            ToDtype(torch.float32, scale=True),
            Resize(size, antialias=True),
            RandomCrop(size, pad_if_needed=False),  # Strong
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
            RandomApply([RandomRotation((90, 90))], p=0.5),
        ]
    )(x)
