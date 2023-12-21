from pathlib import Path

import lightning as pl
import numpy as np
import torch
from matplotlib import pyplot as plt
from seaborn import heatmap
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
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

from frdc.load import FRDCDataset
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


def evaluate(
    ds: FRDCDataset, ckpt_pth: Path | str | None = None
) -> tuple[plt.Figure, float]:
    if ckpt_pth is None:
        # This fetches all possible checkpoints and gets the latest one
        ckpt_pth = sorted(
            THIS_DIR.glob("**/*.ckpt"), key=lambda x: x.stat().st_mtime_ns
        )[-1]

    m = InceptionV3MixMatchModule.load_from_checkpoint(ckpt_pth)
    # Make predictions
    trainer = pl.Trainer(logger=False)
    pred = trainer.predict(m, dataloaders=DataLoader(ds, batch_size=32))

    y_trues = []
    y_preds = []
    for y_true, y_pred in pred:
        y_trues.append(y_true)
        y_preds.append(y_pred)
    y_trues = np.concatenate(y_trues)
    y_preds = np.concatenate(y_preds)
    acc = (y_trues == y_preds).mean()

    # Plot the confusion matrix
    cm = confusion_matrix(y_trues, y_preds)

    plt.figure(figsize=(10, 10))

    heatmap(
        cm,
        annot=True,
        xticklabels=m.y_encoder.categories_[0],
        yticklabels=m.y_encoder.categories_[0],
        cbar=False,
    )
    plt.title(f"Accuracy: {acc:.2%}")
    plt.tight_layout(pad=3)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    return plt.gcf(), acc


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
