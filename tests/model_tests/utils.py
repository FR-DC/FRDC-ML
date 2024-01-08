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
)
from torchvision.transforms.v2 import RandomHorizontalFlip

from frdc.load.dataset import FRDCDataset
from frdc.models.inceptionv3 import InceptionV3MixMatchModule

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


def preprocess(x):
    return Compose(
        [
            ToImage(),
            ToDtype(torch.float32, scale=True),
            CenterCrop(
                [
                    InceptionV3MixMatchModule.MIN_SIZE,
                    InceptionV3MixMatchModule.MIN_SIZE,
                ],
            ),
        ]
    )(x)


def train_preprocess(x):
    return Compose(
        [
            ToImage(),
            ToDtype(torch.float32, scale=True),
            RandomCrop(
                [
                    InceptionV3MixMatchModule.MIN_SIZE,
                    InceptionV3MixMatchModule.MIN_SIZE,
                ],
                pad_if_needed=True,
                padding_mode="constant",
                fill=0,
            ),
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
        ]
    )(x)


def train_unl_preprocess(n_aug: int = 2):
    def f(x):
        # This simulates the n_aug of MixMatch
        return (
            [train_preprocess(x) for _ in range(n_aug)] if n_aug > 0 else None
        )

    return f
