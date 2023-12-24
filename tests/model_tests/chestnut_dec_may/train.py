""" Tests for the InceptionV3 model on the Chestnut Nature Park dataset.


This test is done by training a model on the 20201218 dataset, then testing on
the 20210510 dataset.
"""

import os
from pathlib import Path

import lightning as pl
import numpy as np
import wandb
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    EarlyStopping,
)
from lightning.pytorch.loggers import WandbLogger
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
import matplotlib.pyplot as plt

from frdc.load import FRDCDataset
from frdc.load.dataset import FRDCUnlabelledDataset
from frdc.models.inceptionv3_facenet import InceptionV3FacenetModule
from frdc.train.frdc_datamodule import FRDCDataModule

from src.frdc.utils.triplet_loss import adapted_triplet_loss

from torchvision.transforms.v2 import Compose, RandomHorizontalFlip, RandomVerticalFlip, RandomCrop, ToImage, ToDtype, CenterCrop

import torch

THIS_DIR = Path(__file__).parent


# TODO: Ideally, we should have a separate dataset for testing.


# TODO: This is pretty hacky, I'm not sure if there's a better way to do this.
#       Note that initializing datasets separately then concatenating them
#       together is 4x slower than initializing a dataset then hacking into
#       the __getitem__ method.
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


def evaluate_old(ckpt_pth: Path | str | None = None) -> tuple[plt.Figure, float]:
    ds = FRDCDatasetFlipped(
        "chestnut_nature_park",
        "20210510",
        "90deg43m85pct255deg/map",
        transform=preprocess,
    )

    if ckpt_pth is None:
        # This fetches all possible checkpoints and gets the latest one
        ckpt_pth = sorted(
            THIS_DIR.glob("**/*.ckpt"), key=lambda x: x.stat().st_mtime_ns
        )[-1]

    m = InceptionV3FacenetModule.load_from_checkpoint(ckpt_pth)
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

def evaluate_old(ckpt_pth: Path | str | None = None) -> tuple[plt.Figure, float]:
    ds = FRDCDatasetFlipped(
        "chestnut_nature_park",
        "20210510",
        "90deg43m85pct255deg/map",
        transform=preprocess,
    )

    if ckpt_pth is None:
        # This fetches all possible checkpoints and gets the latest one
        ckpt_pth = sorted(
            THIS_DIR.glob("**/*.ckpt"), key=lambda x: x.stat().st_mtime_ns
        )[-1]

    m = InceptionV3FacenetModule.load_from_checkpoint(ckpt_pth)

    # Judge triplet loss on test data
    trainer = pl.Trainer(logger=False)
    acc = trainer.validate(m, dataloaders=DataLoader(ds, batch_size=32))

    return acc

def preprocess(x):
    return Compose(
        [
            ToImage(),
            ToDtype(torch.float32, scale=True),
            CenterCrop(
                [
                    InceptionV3FacenetModule.MIN_SIZE,
                    InceptionV3FacenetModule.MIN_SIZE,
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
                    InceptionV3FacenetModule.MIN_SIZE,
                    InceptionV3FacenetModule.MIN_SIZE,
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


def main(
    embedding_size=128,
    margin=0.5,
    loss=adapted_triplet_loss,
    squared=False,
    lr_gamma=0.96,
    batch_size=24,
    epochs=10,
    train_iters=25,
    val_iters=15,
    lr=5e-4,
):
    print("Wandb init")
    run = wandb.init()
    logger = WandbLogger(name="chestnut_dec_may", project="frdc")

    print("Train DS init")
    # Prepare the dataset
    train_lab_ds = FRDCDataset(
        "chestnut_nature_park",
        "20201218",
        None,
        transform=train_preprocess,
    )

    """
    print("Train UL DS init")
    # TODO: This is a hacky impl of the unlabelled dataset, see the docstring
    #       for future work.
    train_unl_ds = FRDCUnlabelledDataset(
        "chestnut_nature_park",
        "20201218",
        None,
        transform=train_unl_preprocess(2),
    )
    """

    print("Val DS init")
    # Subset(train_ds, np.argwhere(train_ds.targets == 0).reshape(-1))
    val_ds = FRDCDataset(
        "chestnut_nature_park",
        "20210510",
        "90deg43m85pct255deg",
        transform=preprocess,
    )

    print("OE init")
    oe = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=np.nan,
    )
    oe.fit(np.array(train_lab_ds.targets).reshape(-1, 1))
    n_classes = len(oe.categories_[0])

    print("SS init")
    ss = StandardScaler()
    ss.fit(train_lab_ds.ar.reshape(-1, train_lab_ds.ar.shape[-1]))

    print("DM init")
    # Prepare the datamodule and trainer
    dm = FRDCDataModule(
        train_lab_ds=train_lab_ds,
        # Pass in None to use the default supervised DM
        train_unl_ds=None,#train_unl_ds,
        val_ds=val_ds,
        batch_size=batch_size,
        train_iters=train_iters,
        val_iters=val_iters,
    )

    print("Trainer init")
    trainer = pl.Trainer(
        max_epochs=epochs,
        deterministic=True,
        accelerator="gpu",
        log_every_n_steps=4,
        callbacks=[
            # Stop training if the validation loss doesn't improve for 4 epochs
            EarlyStopping(monitor="val_loss", patience=4, mode="min"),
            # Log the learning rate on TensorBoard
            LearningRateMonitor(logging_interval="epoch"),
            # Save the best model
            ckpt := ModelCheckpoint(
                monitor="val_loss", mode="min", save_top_k=1
            ),
        ],
        logger=logger,
    )


    print("Model init")
    m = InceptionV3FacenetModule(
        embedding_size=embedding_size,
        margin=margin,
        squared=squared,
        lr=lr,
        lr_gamma=lr_gamma,
        loss=loss,
        #x_scaler=ss,
        y_encoder=oe,
    )

    print("Trainer.fit(m, datamodule=dm)")
    trainer.fit(m, datamodule=dm)

    print("Writing report.md")
    with open(Path(__file__).parent / "report.md", "w") as f:
        f.write(
            f"# Chestnut Nature Park (Dec 2020 vs May 2021)\n"
            f"- Results: [WandB Report]({run.get_url()})"
        )

    print("fig, acc = evaluate(...)")
    """
    fig, acc = evaluate(
        ds=FRDCDatasetFlipped(
            "chestnut_nature_park",
            "20210510",
            "90deg43m85pct255deg",
            transform=preprocess,
        ),
        ckpt_pth=Path(ckpt.best_model_path),
    )
    """
    acc = evaluate(
        ds=FRDCDatasetFlipped(
            "chestnut_nature_park",
            "20210510",
            "90deg43m85pct255deg",
            transform=preprocess,
        ),
        ckpt_pth=Path(ckpt.best_model_path),
    )

    print("Log to wandb")
    #wandb.log({"confusion_matrix": wandb.Image(fig)})
    #wandb.log({"eval_accuracy": acc})
    wandb.log({"eval_triplet_loss": acc})

    wandb.finish()


if __name__ == "__main__":
    BATCH_SIZE = 32
    EPOCHS = 10
    TRAIN_ITERS = 25
    VAL_ITERS = 15
    LR = 1e-3

    assert wandb.run is None
    wandb.setup(wandb.Settings(program=__name__, program_relpath=__name__))
    main(
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        train_iters=TRAIN_ITERS,
        val_iters=VAL_ITERS,
        lr=LR,
    )
