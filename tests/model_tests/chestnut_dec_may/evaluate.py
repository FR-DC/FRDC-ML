import os
from pathlib import Path

import lightning as pl
import numpy as np
from matplotlib import pyplot as plt
from seaborn import heatmap
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from torchvision.transforms import RandomVerticalFlip
from torchvision.transforms.v2 import RandomHorizontalFlip

from frdc.load import FRDCDataset
from model_tests.chestnut_dec_may.train import InceptionV3Module, preprocess


# TODO: Ideally, we should have a separate dataset for testing.


# TODO: This is pretty hacky, I'm not sure if there's a better way to do this.
#       Note that initializing datasets separately then concatenating them
#       together is 4x slower than initializing a dataset then hacking into
#       the __getitem__ method.
class FRDCDatasetFlipped(FRDCDataset):
    def __len__(self):
        return super().__len__() * 4

    def __getitem__(self, idx):
        x, y = super().__getitem__(int(idx // 4))
        if idx % 4 == 0:
            return x, y
        elif idx % 4 == 1:
            return RandomHorizontalFlip(p=1)(x), y
        elif idx % 4 == 2:
            return RandomVerticalFlip(p=1)(x), y
        elif idx % 4 == 3:
            return RandomHorizontalFlip(p=1)(RandomVerticalFlip(p=1)(x)), y


def main():
    ds = FRDCDatasetFlipped(
        "chestnut_nature_park",
        "20210510",
        "90deg43m85pct255deg",
        transform=preprocess,
    )

    m = InceptionV3Module.load_from_checkpoint(
        Path("frdc/jz0devw6/checkpoints/epoch=5-step=300.ckpt")
    )
    # Make predictions
    trainer = pl.Trainer(logger=False)
    pred = trainer.predict(m, dataloaders=DataLoader(ds, batch_size=32))
    y_trues = []
    y_preds = []
    for y_true, y_pred in pred:
        y_trues.append(y_true)
        y_preds.append(y_pred.argmax(dim=1))
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
    plt.savefig("confusion_matrix.png")


if __name__ == "__main__":
    os.environ["GOOGLE_CLOUD_PROJECT"] = "frmodel"
    main()
