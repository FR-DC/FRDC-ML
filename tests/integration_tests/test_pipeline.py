import logging
from pathlib import Path

import lightning as pl
import numpy as np
import pytest
import torch
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

from frdc.models.efficientnetb1 import (
    EfficientNetB1MixMatchModule,
    EfficientNetB1FixMatchModule,
)
from frdc.train.frdc_datamodule import FRDCDataModule

BATCH_SIZE = 3


@pytest.mark.parametrize(
    "model_fn",
    [
        EfficientNetB1FixMatchModule,
        EfficientNetB1MixMatchModule,
    ],
)
def test_manual_segmentation_pipeline(model_fn, ds):
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

    m = model_fn(
        in_channels=ds.ar.shape[-1],
        n_classes=n_classes,
        lr=1e-3,
        x_scaler=ss,
        y_encoder=oe,
        frozen=True,
    )

    trainer = pl.Trainer(fast_dev_run=True, accelerator="cpu")
    trainer.fit(m, datamodule=dm)

    val_loss = trainer.validate(m, datamodule=dm)[0]["val/ce_loss"]
    logging.debug(f"Validation score: {val_loss:.2%}")

    # Save the model
    model_fp = Path(__file__).parent / f"{model_fn.__name__}_model.ckpt"
    trainer.save_checkpoint(model_fp)
    logging.debug(f"Model saved to {model_fp}")

    # Attempt to load the model
    m_load = model_fn.load_from_checkpoint(model_fp)
    m_load.eval()
    logging.debug("Model loaded successfully")

    any_diff = False
    # Check which modules differ
    for (m_p_name, m_p), (m_l_name, m_l) in zip(
        m.named_parameters(), m_load.named_parameters()
    ):
        if not torch.allclose(m_p, m_l):
            logging.warning(f"Parameter {m_p_name}")
            logging.warning(f"Original: {m_p}")
            logging.warning(f"Loaded: {m_l}")
            any_diff = True

    assert not any_diff, "Loaded model parameters differ from original"

    # If this step fails, it's likely something "hidden", like the BatchNorm
    # running statistics that differs.
    val_load_loss = trainer.validate(m_load, datamodule=dm)[0]["val/ce_loss"]
    assert val_loss == val_load_loss, "Validation loss differs after loading"

    # Note:
    #   This test doesn't check for all modules to be the same.
    #   E.g. achieved via hash comparison.
    #   This is because BatchNorm usually keeps running statistics
    #   and reloading the model will reset them.
    #   We don't necessarily need to
