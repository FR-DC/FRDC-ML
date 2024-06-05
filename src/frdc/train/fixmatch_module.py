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
        unl_conf_threshold: float = 0.95,
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
            x_scaler: The StandardScaler to use for the data.
            y_encoder: The OrdinalEncoder to use for the labels.
            unl_conf_threshold: The confidence threshold for unlabelled data
                to be considered correctly labelled.
        """

        super().__init__()

        self.x_scaler = x_scaler
        self.y_encoder = y_encoder
        self.n_classes = n_classes
        self.unl_conf_threshold = unl_conf_threshold
        self.save_hyperparameters()

        # We disable's PyTorch Lightning's auto backward during training.
        # See why in `self.training_step` docstring
        self.automatic_optimization = False

    @abstractmethod
    def forward(self, x):
        ...

    def training_step(self, batch, batch_idx):
        """A single training step for a batch

        Notes:
            As mentioned in __init__, we manually back propagate for
            performance reasons. When we loop through `x_unls` (unlabelled
            batches), the gradient tape accumulates unnecessarily.
            We actually just need to back propagate every batch in `x_unls`,
            thus cutting the tape shorter.

            Every tape is bounded by the code:

            >>> opt.zero_grad()
            >>> # Steps that requires grad
            >>> loss = ...
            >>> self.manual_backward(loss)
            >>> opt.step()

            The losses are defined as follows:

            ℓ_lbl           1
            Labelled loss: ---    Σ CE(y_true, y_weak)
                            n  ∀ lbl

            ℓ_unl             1         if  : y_weak > t,
            Unlabelled loss: ---    Σ { then: CE(y_strong, y_weak) }
                              n  ∀ unl  else: 0
            ℓ
            Loss: ℓ_lbl + ℓ_unl
        """
    def training_step(self, batch, batch_idx):
        (x_lbl, y_lbl), x_unls = batch
        opt = self.optimizers()

        # Backprop for labelled data
        opt.zero_grad()
        loss_lbl = F.cross_entropy((y_lbl_pred := self(x_lbl)), y_lbl.long())
        self.manual_backward(loss_lbl)
        opt.step()

        # This is only for logging purposes
        loss_unl = 0

        # Backprop for unlabelled data
        for x_weak, x_strong in x_unls:
            opt.zero_grad()

            # Test if     y_weak is over the threshold
            #      if so, include into the loss
            #      else,  we simply mask it out
            with torch.no_grad():
                y_weak = self(x_weak)
                y_weak_max, y_weak_max_ix = torch.max(y_weak, dim=1)
                is_confident = y_weak_max >= self.unl_conf_threshold

            y_strong = self(x_strong[is_confident])

            # CE only on the masked out samples
            # We perform `reduction="sum"` so that we "include" the masked out
            # samples by fixing the denominator.
            # E.g.
            # y_weak > t        = [T, F, T, F]
            # Losses            = [1, 2, 3, 4]
            # Masked Losses     = [1,    3,  ]
            # Incorrect CE Mean = (1 + 3) / 2
            # Correct CE Mean   = (1 + 3) / 4
            batch_size = x_lbl.shape[0]
            loss_unl_i = F.cross_entropy(
                y_strong,
                y_weak_max_ix[is_confident],
                reduction="sum",
            ) / (len(x_unls) * batch_size)

            self.manual_backward(loss_unl_i)
            opt.step()

            loss_unl += loss_unl_i.detach().item()

            self.log("train/x0_unl_mean", x_weak[0].mean())
            self.log("train/x0_unl_stdev", x_weak[0].std())
            wandb.log(
                {
                    "train/y_unl_pred": wandb_hist(
                        torch.argmax(y_strong, dim=1), self.n_classes
                    )
                }
            )

        # Evaluate train accuracy
        with torch.no_grad():
            y_pred = self(x_lbl)
            acc = accuracy(
                y_pred, y_lbl, task="multiclass", num_classes=y_pred.shape[1]
            )

        self.log("train/x_lbl_mean", x_lbl.mean())
        self.log("train/x_lbl_stdev", x_lbl.std())
        wandb.log({"train/x_lbl": wandb_hist(y_lbl, self.n_classes)})
        self.log("train/ce_loss_lbl", loss_lbl)
        self.log("train/ce_loss_unl", loss_unl)
        self.log("train/loss", loss_lbl + loss_unl)
        self.log("train/acc", acc, prog_bar=True)

        wandb.log(
            {
                "train/y_lbl_pred": wandb_hist(
                    torch.argmax(y_lbl_pred, dim=1), self.n_classes
                )
            }
        )

    def validation_step(self, batch, batch_idx):
        # The batch outputs x_unls due to our on_before_batch_transfer
        (x, y), _x_unls = batch
        wandb.log({"val/y_lbl": wandb_hist(y, self.n_classes)})
        y_pred = self(x)
        loss = F.cross_entropy(y_pred, y.long())
        acc = accuracy(
            y_pred, y, task="multiclass", num_classes=self.n_classes
        )

        wandb.log({"val/y_lbl": wandb_hist(y, self.n_classes)})
        wandb.log(
            {
                "val/y_lbl_pred": wandb_hist(
                    torch.argmax(y_pred, dim=1), self.n_classes
                )
            }
        )
        self.log("val/ce_loss", loss)
        self.log("val/acc", acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        # The batch outputs x_unls due to our on_before_batch_transfer
        (x, y), _x_unls = batch
        y_pred = self(x)
        loss = F.cross_entropy(y_pred, y.long())

        acc = accuracy(
            y_pred, y, task="multiclass", num_classes=self.n_classes
        )
        self.log("test/ce_loss", loss)
        self.log("test/acc", acc, prog_bar=True)
        return loss

    def predict_step(self, batch, *args, **kwargs) -> Any:
        (x, y), _x_unls = batch
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

        if self.training:
            (x_lbl, y_lbl), x_unl = batch
        else:
            x_lbl, y_lbl = batch
            x_unl = None

        return preprocess(
            x_lbl=x_lbl,
            y_lbl=y_lbl,
            x_scaler=self.x_scaler,
            y_encoder=self.y_encoder,
            x_unl=x_unl,
        )
