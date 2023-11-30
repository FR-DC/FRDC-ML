from abc import abstractmethod
from typing import Any

import numpy as np
import torch
from lightning import LightningModule
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from torch import nn


class FRDCModule(LightningModule):
    def __init__(self, x_scaler: StandardScaler, y_encoder: OrdinalEncoder):
        super().__init__()
        self.x_scaler = x_scaler
        self.y_encoder = y_encoder
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.CrossEntropyLoss()(y_hat, y.squeeze().long())
        self.log("loss", loss, prog_bar=True)
        self.log(
            "acc", (y_hat.argmax(dim=1) == y).float().mean(), prog_bar=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.CrossEntropyLoss()(y_hat, y.squeeze().long())
        self.log("val_loss", loss)
        self.log(
            "val_acc", (y_hat.argmax(dim=1) == y).float().mean(), prog_bar=True
        )
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        self.log("test_loss", loss)
        self.log("test_acc", (y_hat.argmax(dim=1) == y).float().mean())
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x, y = batch
        y_hat = self(x)
        return y, y_hat

    # TODO: PyTorch Lightning is complaining that I'm setting this in the
    #       Module instead of DataModule, we can likely migrate this to
    #       the ___step() functions.
    @torch.no_grad()
    def on_before_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        x, y = batch

        # Standard Scaler only accepts (n_samples, n_features), so we need to
        # do some fancy reshaping.
        # Note that moving dimensions then reshaping is different from just
        # reshaping!
        x: torch.Tensor
        b, c, h, w = x.shape

        # Move Channel to the last dimension then transform
        x_ss: np.ndarray = self.x_scaler.transform(
            x.permute(0, 2, 3, 1).reshape(-1, c)
        )

        # Move Channel back to the second dimension
        x_: torch.Tensor = (
            torch.from_numpy(x_ss.reshape(b, h, w, c))
            .permute(0, 3, 1, 2)
            .float()
        )

        # Ordinal Encoder only accepts (n_samples, 1), so we need to do some
        y: tuple[str]
        y_: torch.Tensor = torch.from_numpy(
            self.y_encoder.transform(np.array(y).reshape(-1, 1)).squeeze()
        )

        # Ordinal Encoders can return a np.nan if the value is not in the
        # categories. We will remove that from the batch.
        x_ = x_[~torch.isnan(y_)]
        y_ = y_[~torch.isnan(y_)]

        return x_, y_.long()

    @abstractmethod
    def forward(self, x):
        ...

    @abstractmethod
    def configure_optimizers(self):
        ...
