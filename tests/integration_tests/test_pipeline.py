import logging

import lightning as pl
import numpy as np
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

from frdc.models.efficientnetb1 import EfficientNetB1MixMatchModule
from frdc.train.frdc_datamodule import FRDCDataModule

BATCH_SIZE = 3


def test_manual_segmentation_pipeline(ds):
    """Manually segment the image according to bounds.csv,
    then train a model on it."""

    dm = FRDCDataModule(
        train_lab_ds=ds,
        train_unl_ds=None,
        val_ds=ds,
        batch_size=BATCH_SIZE,
    )

    oe = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=np.nan,
    )
    oe.fit(np.array(ds.targets).reshape(-1, 1))
    n_classes = len(oe.categories_[0])

    ss = StandardScaler()
    ss.fit(ds.ar.reshape(-1, ds.ar.shape[-1]))

    m = EfficientNetB1MixMatchModule(
        in_channels=ds.ar.shape[-1],
        n_classes=n_classes,
        lr=1e-3,
        x_scaler=ss,
        y_encoder=oe,
    )

    trainer = pl.Trainer(fast_dev_run=True)
    trainer.fit(m, datamodule=dm)

    val_loss = trainer.validate(m, datamodule=dm)[0]["val/ce_loss"]
    logging.debug(f"Validation score: {val_loss:.2%}")
