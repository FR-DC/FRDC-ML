from copy import deepcopy
from typing import Dict, Any

import torch
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from torch import nn
from torchvision.models import (
    EfficientNet,
    efficientnet_b1,
    EfficientNet_B1_Weights,
)

from frdc.models.utils import on_save_checkpoint, on_load_checkpoint
from frdc.train.fixmatch_module import FixMatchModule
from frdc.train.mixmatch_module import MixMatchModule
from frdc.utils.ema import EMA


class PartialConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        return super().forward(x)


def efficientnet_b1_backbone(in_channels: int, frozen: bool):
    """Get the N Channel adapted EfficientNet B1 model without classifier.

    Args:
        in_channels: The number of input channels.
        frozen: Whether to freeze the base model.

    Returns:
        The EfficientNet B1 model.
    """
    eff = efficientnet_b1(weights=EfficientNet_B1_Weights.IMAGENET1K_V2)

    # Adapt the first layer to accept the number of channels
    eff = fork_n_in_channel(eff, in_channels)

    # Remove the final layer
    eff.classifier = nn.Identity()

    if frozen:
        for param in eff.parameters():
            param.requires_grad = False
        for param in eff.features[0][0].conv_other.parameters():
            param.requires_grad = True

    return eff


def adapt_n_in_channel(eff: EfficientNet, in_channels: int) -> EfficientNet:
    """Adapt the EfficientNet model to accept a different number of
    input channels.

    Notes:
        This operation is in-place, however will still return the model

    Args:
        eff: The EfficientNet model
        in_channels: The number of input channels

    Returns:
        The adapted EfficientNet model.
    """
    old_conv = eff.features[0][0]
    new_conv = nn.Conv2d(
        in_channels=in_channels,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=old_conv.bias,
    )
    new_conv.weight.data[:, :3] = old_conv.weight.data
    new_conv.weight.data[:, 3:] = old_conv.weight.data[:, 1:2].repeat(
        1, in_channels - 3, 1, 1
    )
    eff.features[0][0] = new_conv

    return eff


def fork_n_in_channel(eff: EfficientNet, in_channels: int) -> EfficientNet:
    """Adapt the EfficientNet model to accept a different number of
    input channels.

    Notes:
        This operation is in-place, however will still return the model

    Args:
        eff: The EfficientNet model
        in_channels: The number of input channels

    Returns:
        The adapted EfficientNet model.
    """

    class ForkConv2d(nn.Module):
        def __init__(
            self, in_channels, out_channels, kernel_size, stride, padding, bias
        ):
            super().__init__()
            self.conv_rgb = nn.Conv2d(
                in_channels=3,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
            )
            self.conv_other = nn.Conv2d(
                in_channels=in_channels - 3,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
            )
            self.rgb_weight = 3 / 8
            self.other_weight = 5 / 8

        def forward(self, x):
            x_rgb = self.conv_rgb(x[:, :3])
            x_other = self.conv_other(x[:, 3:])

            return self.rgb_weight * x_rgb + self.other_weight * x_other

    old_conv = eff.features[0][0]
    new_conv = ForkConv2d(
        in_channels,
        old_conv.out_channels,
        old_conv.kernel_size,
        old_conv.stride,
        old_conv.padding,
        old_conv.bias,
    )

    new_conv.conv_rgb.weight.data[:, :3] = old_conv.weight.data
    eff.features[0][0] = new_conv

    return eff


class EfficientNetB1MixMatchModule(MixMatchModule):
    MIN_SIZE = 255
    EFF_OUT_DIMS = 1280

    def __init__(
        self,
        *,
        in_channels: int,
        n_classes: int,
        lr: float,
        x_scaler: StandardScaler,
        y_encoder: OrdinalEncoder,
        ema_lr: float = 0.001,
        weight_decay: float = 1e-5,
        frozen: bool = True,
    ):
        """Initialize the EfficientNet model.

        Args:
            in_channels: The number of input channels.
            n_classes: The number of classes.
            lr: The learning rate.
            x_scaler: The X input StandardScaler.
            y_encoder: The Y input OrdinalEncoder.
            ema_lr: The learning rate for the EMA model.
            weight_decay: The weight decay.
            frozen: Whether to freeze the base model.

        Notes:
            - Min input size: 255 x 255
        """
        self.lr = lr
        self.weight_decay = weight_decay

        super().__init__(
            n_classes=n_classes,
            x_scaler=x_scaler,
            y_encoder=y_encoder,
            sharpen_temp=0.5,
            mix_beta_alpha=0.75,
        )

        self.eff = efficientnet_b1_backbone(in_channels, frozen)
        self.fc = nn.Sequential(
            nn.Linear(self.EFF_OUT_DIMS, n_classes),
            nn.Softmax(dim=1),
        )

        # The problem is that the deep copy runs even before the module is
        # initialized, which means ema_model is empty.
        ema_model = deepcopy(self)
        for param in ema_model.parameters():
            param.detach_()

        self._ema_model = ema_model
        self.ema_updater = EMA(model=self, ema_model=self.ema_model)
        self.ema_lr = ema_lr

    @property
    def ema_model(self):
        return self._ema_model

    def update_ema(self):
        self.ema_updater.update(self.ema_lr)

    def forward(self, x: torch.Tensor):
        """Forward pass."""
        return self.fc(self.eff(x))

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        # TODO: MixMatch's saving is a bit complicated due to the dependency
        #       on the EMA model. This only saves the FC for both the
        #       main model and the EMA model.
        #       This may be the reason certain things break when loading
        if checkpoint["hyper_parameters"]["frozen"]:
            on_save_checkpoint(
                self,
                checkpoint,
                saved_module_prefixes=("_ema_model.fc.", "fc."),
            )

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        on_load_checkpoint(self, checkpoint)


class EfficientNetB1FixMatchModule(FixMatchModule):
    MIN_SIZE = 255
    EFF_OUT_DIMS = 1280

    def __init__(
        self,
        *,
        in_channels: int,
        n_classes: int,
        lr: float,
        x_scaler: StandardScaler,
        y_encoder: OrdinalEncoder,
        weight_decay: float = 1e-5,
        frozen: bool = True,
    ):
        """Initialize the EfficientNet model.

        Args:
            in_channels: The number of input channels.
            n_classes: The number of classes.
            lr: The learning rate.
            x_scaler: The X input StandardScaler.
            y_encoder: The Y input OrdinalEncoder.
            weight_decay: The weight decay.
            frozen: Whether to freeze the base model.

        Notes:
            - Min input size: 255 x 255
        """
        self.lr = lr
        self.weight_decay = weight_decay

        super().__init__(
            n_classes=n_classes,
            x_scaler=x_scaler,
            y_encoder=y_encoder,
        )

        self.eff = efficientnet_b1_backbone(in_channels, frozen)

        self.fc = nn.Sequential(
            nn.Linear(self.EFF_OUT_DIMS, n_classes),
            nn.Softmax(dim=1),
        )

    def forward(self, x: torch.Tensor):
        """Forward pass."""
        return self.fc(self.eff(x))

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        if checkpoint["hyper_parameters"]["frozen"]:
            on_save_checkpoint(
                self,
                checkpoint,
                saved_module_prefixes=("fc.",),
            )

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        on_load_checkpoint(self, checkpoint)
