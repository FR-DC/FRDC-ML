from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, Sequence

import torch
import torch.nn.functional as F
import wandb
from torch.nn.functional import one_hot
from torchmetrics.functional import accuracy

from frdc.models.utils import save_unfrozen
from frdc.train.frdc_module import FRDCModule
from frdc.train.utils import (
    mix_up,
    sharpen,
    wandb_hist,
)


class MixMatchModule(FRDCModule):
    def __init__(
        self,
        *,
        out_targets: Sequence[str],
        sharpen_temp: float = 0.5,
        mix_beta_alpha: float = 0.75,
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
            out_targets: The output targets for the model.
            sharpen_temp: The temperature to use for sharpening.
            mix_beta_alpha: The alpha to use for the beta distribution
                when mixing.
        """

        super().__init__(out_targets=out_targets)

        self.sharpen_temp = sharpen_temp
        self.mix_beta_alpha = mix_beta_alpha
        self.save_hyperparameters()

    @property
    @abstractmethod
    def ema_model(self):
        """The inherited class should return the EMA model, which it should
        retroactively create through `deepcopy(self)`. Furthermore, the
        training loop will automatically call `update_ema` after each batch.
        Thus, the inherited class should implement `update_ema` to update the
        EMA model.
        """
        ...

    @abstractmethod
    def update_ema(self):
        """This method should update the EMA model, which is handled by the
        inherited class.
        """
        ...

    @abstractmethod
    def forward(self, x):
        ...

    @staticmethod
    def loss_unl_scaler(progress: float) -> float:
        return progress * 75

    @staticmethod
    def loss_lbl(lbl_pred: torch.Tensor, lbl: torch.Tensor):
        return F.cross_entropy(lbl_pred, lbl)

    @staticmethod
    def loss_unl(unl_pred: torch.Tensor, unl: torch.Tensor):
        return torch.mean((torch.softmax(unl_pred, dim=1) - unl) ** 2)

    def guess_labels(
        self,
        x_unls: list[torch.Tensor],
    ) -> torch.Tensor:
        """Guess labels from the unlabelled data"""
        y_unls: list[torch.Tensor] = [
            torch.softmax(self.ema_model(u), dim=1) for u in x_unls
        ]
        # The sum will sum the tensors in the list,
        # it doesn't reduce the tensors
        y_unl = sum(y_unls) / len(y_unls)
        # noinspection PyTypeChecker
        return y_unl

    @property
    def progress(self):
        # Progress is a linear ramp from 0 to 1 over the course of training.
        return (
            self.global_step / self.trainer.num_training_batches
        ) / self.trainer.max_epochs

    def training_step(
        self,
        batch: tuple[tuple[torch.Tensor, torch.Tensor], list[torch.Tensor]],
        batch_idx: int,
    ):
        (x_lbl, y_lbl), x_unls = batch

        self.log("train/x_lbl_mean", x_lbl.mean())
        self.log("train/x_lbl_stdev", x_lbl.std())

        wandb.log({"train/x_lbl": wandb_hist(y_lbl, self.n_classes)})
        y_lbl_ohe = one_hot(y_lbl.long(), num_classes=self.n_classes)

        # If x_unls is Truthy, then we are using MixMatch.
        # Otherwise, we are just using supervised learning.
        if x_unls:
            # This route implies that we are using SSL
            self.log("train/x0_unl_mean", x_unls[0].mean())
            self.log("train/x0_unl_stdev", x_unls[0].std())
            with torch.no_grad():
                y_unl = self.guess_labels(x_unls=x_unls)
                y_unl = sharpen(y_unl, self.sharpen_temp)

            x = torch.cat([x_lbl, *x_unls], dim=0)
            y = torch.cat([y_lbl_ohe, *(y_unl,) * len(x_unls)], dim=0)
            x_mix, y_mix = mix_up(x, y, self.mix_beta_alpha)

            # This had interleaving, but it was removed as it's not
            # significantly better
            batch_size = x_lbl.shape[0]
            y_mix_pred = self(x_mix)
            y_mix_lbl_pred = y_mix_pred[:batch_size]
            y_mix_unl_pred = y_mix_pred[batch_size:]
            y_mix_lbl = y_mix[:batch_size]
            y_mix_unl = y_mix[batch_size:]

            loss_lbl = self.loss_lbl(y_mix_lbl_pred, y_mix_lbl)
            loss_unl = self.loss_unl(y_mix_unl_pred, y_mix_unl)
            wandb.log(
                {
                    "train/y_lbl_pred": wandb_hist(
                        torch.argmax(y_mix_lbl_pred, dim=1), self.n_classes
                    )
                }
            )
            wandb.log(
                {
                    "train/y_unl_pred": wandb_hist(
                        torch.argmax(y_mix_unl_pred, dim=1), self.n_classes
                    )
                }
            )
            loss_unl_scale = self.loss_unl_scaler(progress=self.progress)

            loss = loss_lbl + loss_unl * loss_unl_scale

            self.log("train/loss_unl_scale", loss_unl_scale, prog_bar=True)
            self.log("train/ce_loss_lbl", loss_lbl)
            self.log("train/mse_loss_unl", loss_unl)
        else:
            # This route implies that we are just using supervised learning
            y_pred = self(x_lbl)
            loss = self.loss_lbl(y_pred, y_lbl_ohe.float())

        self.log("train/loss", loss)

        # Evaluate train accuracy
        with torch.no_grad():
            y_pred = self.ema_model(x_lbl)
            acc = accuracy(
                y_pred, y_lbl, task="multiclass", num_classes=y_pred.shape[1]
            )
            self.log("train/acc", acc, prog_bar=True)
        return loss

    # PyTorch Lightning doesn't automatically no_grads the EMA step.
    # It's important to keep this to avoid a memory leak.
    @torch.no_grad()
    def on_after_backward(self) -> None:
        self.update_ema()

    def validation_step(
        self,
        batch: tuple[tuple[torch.Tensor, torch.Tensor], list[torch.Tensor]],
        batch_idx: int,
    ):
        (x, y), _x_unls = batch
        wandb.log({"val/y_lbl": wandb_hist(y, self.n_classes)})
        y_pred = self.ema_model(x)
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

    def test_step(
        self,
        batch: tuple[tuple[torch.Tensor, torch.Tensor], list[torch.Tensor]],
        batch_idx: int,
    ):
        (x, y), _x_unls = batch
        y_pred = self.ema_model(x)
        loss = F.cross_entropy(y_pred, y.long())

        acc = accuracy(
            y_pred, y, task="multiclass", num_classes=y_pred.shape[1]
        )
        self.log("test/ce_loss", loss)
        self.log("test/acc", acc, prog_bar=True)
        return loss

    def predict_step(
        self,
        batch: tuple[tuple[torch.Tensor, torch.Tensor], list[torch.Tensor]],
    ) -> Any:
        (x, y), _x_unls = batch
        y_pred = self.ema_model(x)
        y_true_str = self.y_encoder.inverse_transform(
            y.cpu().numpy().reshape(-1, 1)
        )
        y_pred_str = self.y_encoder.inverse_transform(
            y_pred.argmax(dim=1).cpu().numpy().reshape(-1, 1)
        )
        return y_true_str, y_pred_str

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """This override the original method to save the EMAs as well."""
        save_unfrozen(
            self,
            checkpoint,
            include_also=lambda k: k.startswith("_ema_model.fc."),
        )
