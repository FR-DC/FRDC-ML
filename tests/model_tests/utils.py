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


def n_strong_aug(size, n_aug: int = 2):
    return lambda x: (
        [strong_aug(size)(x) for _ in range(n_aug)] if n_aug > 0 else None
    )


def n_weak_strong_aug(size, n_aug: int = 2):
    def f(x):
        x_weak = n_weak_aug(size, n_aug)(x)
        x_strong = n_strong_aug(size, n_aug)(x)
        return list(zip(*[x_weak, x_strong])) if n_aug > 0 else None

    return f


def weak_aug(size: int):
    return lambda x: Compose(
        [
            ToImage(),
            ToDtype(torch.float32, scale=True),
            Resize(size, antialias=True),
            CenterCrop(size),
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


def get_y_encoder(targets):
    oe = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=np.nan,
    )
    oe.fit(np.array(targets).reshape(-1, 1))
    return oe


def get_x_scaler(segments):
    ss = StandardScaler()
    ss.fit(
        np.concatenate([segm.reshape(-1, segm.shape[-1]) for segm in segments])
    )
    return ss
