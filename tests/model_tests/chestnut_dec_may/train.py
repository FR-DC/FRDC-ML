""" Tests for the InceptionV3 model on the Chestnut Nature Park dataset.

This test is done by training a model on the 20201218 dataset, then testing on
the 20210510 dataset.
"""
import os

# Uncomment this to run the W&B monitoring locally
# import os
# from frdc.utils.training import predict, plot_confusion_matrix
# os.environ["WANDB_MODE"] = "offline"

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

from frdc.load.preset import FRDCDatasetPreset as ds
from frdc.models.inceptionv3 import InceptionV3MixMatchModule
from frdc.train.frdc_datamodule import FRDCDataModule
from frdc.utils.training import predict, plot_confusion_matrix
from model_tests.utils import (
    train_preprocess,
    train_unl_preprocess,
    preprocess,
    FRDCDatasetFlipped,
)


def main(
    batch_size=32,
    epochs=10,
    train_iters=25,
    val_iters=15,
    lr=1e-3,
):
    # Prepare the dataset
    train_lab_ds = ds.chestnut_20201218(transform=train_preprocess)
    train_unl_ds = ds.chestnut_20201218.unlabelled(
        transform=train_unl_preprocess(2)
    )
    val_ds = ds.chestnut_20210510_43m(transform=preprocess)

    oe = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=np.nan,
    )
    oe.fit(np.array(train_lab_ds.targets).reshape(-1, 1))
    n_classes = len(oe.categories_[0])

    ss = StandardScaler()
    ss.fit(train_lab_ds.ar.reshape(-1, train_lab_ds.ar.shape[-1]))

    # Prepare the datamodule and trainer
    dm = FRDCDataModule(
        train_lab_ds=train_lab_ds,
        train_unl_ds=train_unl_ds,  # None to use supervised DM
        val_ds=val_ds,
        batch_size=batch_size,
        train_iters=train_iters,
        val_iters=val_iters,
        sampling_strategy="stratified",
    )

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
        logger=(
            logger := WandbLogger(name="chestnut_dec_may", project="frdc")
        ),
    )

    m = InceptionV3MixMatchModule(
        n_classes=n_classes,
        lr=lr,
        x_scaler=ss,
        y_encoder=oe,
    )
    logger.watch(m)

    trainer.fit(m, datamodule=dm)

    with open(Path(__file__).parent / "report.md", "w") as f:
        f.write(
            f"# Chestnut Nature Park (Dec 2020 vs May 2021)\n"
            f"- Results: [WandB Report]({wandb.run.get_url()})"
        )

    y_true, y_pred = predict(
        ds=FRDCDatasetFlipped(
            "chestnut_nature_park",
            "20210510",
            "90deg43m85pct255deg",
            transform=preprocess,
        ),
        model_cls=InceptionV3MixMatchModule,
        ckpt_pth=Path(ckpt.best_model_path),
    )
    fig, ax = plot_confusion_matrix(y_true, y_pred, oe.categories_[0])
    acc = np.sum(y_true == y_pred) / len(y_true)
    ax.set_title(f"Accuracy: {acc:.2%}")

    wandb.log({"confusion_matrix": wandb.Image(fig)})
    wandb.log({"eval_accuracy": acc})

    wandb.finish()


if __name__ == "__main__":
    BATCH_SIZE = 32
    EPOCHS = 10
    TRAIN_ITERS = 25
    VAL_ITERS = 15
    LR = 1e-3

    wandb.login(key=os.environ["WANDB_API_KEY"])

    main(
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        train_iters=TRAIN_ITERS,
        val_iters=VAL_ITERS,
        lr=LR,
    )
