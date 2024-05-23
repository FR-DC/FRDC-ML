from __future__ import annotations

from abc import abstractmethod
from typing import Any

import torch
import torch.nn.functional as F
import wandb
from lightning import LightningModule
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from torchmetrics.functional import accuracy

from frdc.train.utils import (
    wandb_hist,
    preprocess,
)


class FixMatchModule(LightningModule):
    def __init__(
        self,
        *,
        x_scaler: StandardScaler,
        y_encoder: OrdinalEncoder,
        n_classes: int = 10,
    ):
        """PyTorch Lightning Module for MixMatch

        Notes:
            This performs MixMatch as described in the paper.
            https://arxiv.org/abs/1905.02249

            This module is designed to be used with any model, not only
            the WideResNet model.

            Furthermore, while it's possible to switch datasets, take a look
            at how we implement the CIFAR10DataModule's DataLoaders to see
            how to implement a new dataset.

        Args:
            n_classes: The number of classes in the dataset.
        """

        super().__init__()

        self.x_scaler = x_scaler
        self.y_encoder = y_encoder
        self.n_classes = n_classes
        self.save_hyperparameters()
        self.automatic_optimization = False

    @abstractmethod
    def forward(self, x):
        ...

    @staticmethod
    def loss_lbl(lbl_pred: torch.Tensor, lbl: torch.Tensor):
        return F.cross_entropy(lbl_pred, lbl)

    @staticmethod
    def loss_unl(unl_pred: torch.Tensor, unl: torch.Tensor):
        return F.cross_entropy(unl_pred, unl)

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        opt.zero_grad()
        (x_lbl, y_lbl), x_unls = batch

        self.log("train/x_lbl_mean", x_lbl.mean())
        self.log("train/x_lbl_stdev", x_lbl.std())

        wandb.log({"train/x_lbl": wandb_hist(y_lbl, self.n_classes)})
        loss_lbl = F.cross_entropy((y_lbl_pred := self(x_lbl)), y_lbl.long())
        self.manual_backward(loss_lbl)
        opt.step()

        wandb.log(
            {
                "train/y_lbl_pred": wandb_hist(
                    torch.argmax(y_lbl_pred, dim=1), self.n_classes
                )
            }
        )
        loss_unl = 0

        for x_weak, x_strong in x_unls:
            opt.zero_grad()
            self.log("train/x0_unl_mean", x_weak[0].mean())
            self.log("train/x0_unl_stdev", x_weak[0].std())
            thres = 0.95
            with torch.no_grad():
                y_pred_weak = self(x_weak)
                y_pred_weak_max, y_pred_weak_max_ix = torch.max(
                    y_pred_weak, dim=1
                )
            y_pred_strong = self(x_strong)

            loss_unl_i = F.cross_entropy(
                y_pred_strong[y_pred_weak_max >= thres],
                y_pred_weak_max_ix[y_pred_weak_max >= thres],
                reduction="sum",
            ) / (len(x_unls) * x_lbl.shape[0])

            self.manual_backward(loss_unl_i)
            opt.step()

            loss_unl += loss_unl_i.detach().item()

            wandb.log(
                {
                    "train/y_unl_pred": wandb_hist(
                        torch.argmax(y_pred_strong, dim=1), self.n_classes
                    )
                }
            )

        self.log("train/ce_loss_lbl", loss_lbl)
        self.log("train/ce_loss_unl", loss_unl)
        self.log("train/loss", loss_lbl + loss_unl)

        # Evaluate train accuracy
        with torch.no_grad():
            y_pred = self(x_lbl)
            acc = accuracy(
                y_pred, y_lbl, task="multiclass", num_classes=y_pred.shape[1]
            )
            self.log("train/acc", acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        wandb.log({"val/y_lbl": wandb_hist(y, self.n_classes)})
        y_pred = self(x)
        wandb.log(
            {
                "val/y_lbl_pred": wandb_hist(
                    torch.argmax(y_pred, dim=1), self.n_classes
                )
            }
        )
        loss = F.cross_entropy(y_pred, y.long())

        acc = accuracy(
            y_pred, y, task="multiclass", num_classes=y_pred.shape[1]
        )
        self.log("val/ce_loss", loss)
        self.log("val/acc", acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = F.cross_entropy(y_pred, y.long())

        acc = accuracy(
            y_pred, y, task="multiclass", num_classes=y_pred.shape[1]
        )
        self.log("test/ce_loss", loss)
        self.log("test/acc", acc, prog_bar=True)
        return loss

    def predict_step(self, batch, *args, **kwargs) -> Any:
        x, y = batch
        y_pred = self(x)
        y_true_str = self.y_encoder.inverse_transform(
            y.cpu().numpy().reshape(-1, 1)
        )
        y_pred_str = self.y_encoder.inverse_transform(
            y_pred.argmax(dim=1).cpu().numpy().reshape(-1, 1)
        )
        return y_true_str, y_pred_str

    @torch.no_grad()
    def on_before_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        """This method is called before any data transfer to the device.

        We leverage this to do some preprocessing on the data.
        Namely, we use the StandardScaler and OrdinalEncoder to transform the
        data.

        Notes:
            PyTorch Lightning may complain about this being on the Module
            instead of the DataModule. However, this is intentional as we
            want to export the model alongside the transformations.
        """

        # We need to handle the train and val dataloaders differently.
        # For training, the unlabelled data is returned while for validation,
        # the unlabelled data is just omitted.
        if self.training:
            (x_lab, y), x_unl = batch
        else:
            x_lab, y = batch
            x_unl = []

        (x_lab_trans, y_trans), x_unl_trans = preprocess(
            x_lab=x_lab,
            y_lab=y,
            x_unl=x_unl,
            x_scaler=self.x_scaler,
            y_encoder=self.y_encoder,
        )
        if self.training:
            return (x_lab_trans, y_trans), x_unl_trans
        else:
            return x_lab_trans, y_trans
