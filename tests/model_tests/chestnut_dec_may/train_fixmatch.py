""" Tests for the model on the Chestnut Nature Park dataset.

This test is done by training a model on the 20201218 dataset, then testing on
the 20210510 dataset.
"""
from pathlib import Path

import lightning as pl
import numpy as np
import torch
import wandb
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    EarlyStopping,
)
from lightning.pytorch.loggers import WandbLogger

from frdc.load.preset import FRDCDatasetPreset as ds
from frdc.models.efficientnetb1 import (
    EfficientNetB1FixMatchModule,
)
from frdc.train.frdc_datamodule import FRDCDataModule
from frdc.utils.training import predict, plot_confusion_matrix
from model_tests.utils import (
    const_weak_aug,
    FRDCDatasetStaticEval,
    n_rand_weak_strong_aug,
    rand_weak_aug,
)


# Uncomment this to run the W&B monitoring locally
# import os
#
# os.environ["WANDB_MODE"] = "offline"


def main(
        batch_size=32,
        epochs=10,
        train_iters=25,
        unlabelled_factor=2,
        lr=1e-3,
        accelerator="gpu",
        wandb_active: bool = True,
        wandb_name="chestnut_dec_may",
        wandb_project="frdc",
):
    if not wandb_active:
        import os

        os.environ["WANDB_MODE"] = "offline"

    # Prepare the dataset
    im_size = 255
    train_lab_ds = ds.chestnut_20201218(transform=rand_weak_aug(im_size))
    train_unl_ds = ds.chestnut_20201218.unlabelled(
        transform=n_rand_weak_strong_aug(im_size, unlabelled_factor),
    )
    val_ds = ds.chestnut_20210510_43m(
        transform=const_weak_aug(im_size),
        transform_scale=train_lab_ds.x_scaler,
    )

    # Prepare the datamodule and trainer
    dm = FRDCDataModule(
        train_lab_ds=train_lab_ds,
        train_unl_ds=train_unl_ds,  # None to use supervised DM
        val_ds=val_ds,
        batch_size=batch_size,
        train_iters=train_iters,
        sampling_strategy="random",
    )

    trainer = pl.Trainer(
        max_epochs=epochs,
        deterministic=True,
        accelerator=accelerator,
        log_every_n_steps=4,
        callbacks=[
            # Stop training if the validation loss doesn't improve for 4 epochs
            EarlyStopping(monitor="val/ce_loss", patience=4, mode="min"),
            # Log the learning rate on TensorBoard
            LearningRateMonitor(logging_interval="epoch"),
            # Save the best model
            ckpt := ModelCheckpoint(
                monitor="val/ce_loss", mode="min", save_top_k=1
            ),
        ],
        logger=(
            logger := WandbLogger(
                name=wandb_name,
                project=wandb_project,
            )
        ),
    )

    m = EfficientNetB1FixMatchModule(
        in_channels=train_lab_ds.ar.shape[-1],
        out_targets=train_lab_ds.targets,
        lr=lr,
        frozen=True,
    )

    trainer.fit(m, datamodule=dm)

    with open(Path(__file__).parent / "report.md", "a+") as f:
        f.write(
            f"# Chestnut Nature Park (Dec 2020 vs May 2021) FixMatch\n"
            f"- Results: [WandB Report]({wandb.run.get_url()})\n"
        )

    y_true, y_pred = predict(
        ds=FRDCDatasetStaticEval(
            "chestnut_nature_park",
            "20210510",
            "90deg43m85pct255deg",
            transform=const_weak_aug(im_size),
            transform_scale=train_lab_ds.x_scaler,
        ),
        model=m,
    )
    fig, ax = plot_confusion_matrix(y_true, y_pred, m.y_encoder.categories_[0])
    acc = np.sum(y_true == y_pred) / len(y_true)
    ax.set_title(f"Accuracy: {acc:.2%}")

    wandb.log({"eval/confusion_matrix": wandb.Image(fig)})
    wandb.log({"eval/eval_accuracy": acc})

    wandb.finish()


if __name__ == "__main__":
    BATCH_SIZE = 32
    EPOCHS = 10
    TRAIN_ITERS = 25
    LR = 3e-3

    torch.set_float32_matmul_precision("high")
    main(
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        train_iters=TRAIN_ITERS,
        lr=LR,
        unlabelled_factor=5,
        wandb_name="FixMatch EfficientNet 255x255",
        wandb_project="frdc-dev",
    )
