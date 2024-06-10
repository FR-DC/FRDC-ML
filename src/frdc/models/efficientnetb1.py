from copy import deepcopy
from typing import Sequence

import torch
from torch import nn
from torchvision.models import (
    EfficientNet,
    efficientnet_b1,
    EfficientNet_B1_Weights,
)

from frdc.train.fixmatch_module import FixMatchModule
from frdc.train.mixmatch_module import MixMatchModule
from frdc.utils.ema import EMA


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
    eff = adapt_n_in_channel(eff, in_channels)

    # Remove the final layer
    eff.classifier = nn.Identity()

    if frozen:
        for param in eff.parameters():
            param.requires_grad = False

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


class EfficientNetB1MixMatchModule(MixMatchModule):
    MIN_SIZE = 255
    EFF_OUT_DIMS = 1280

    def __init__(
        self,
        *,
        in_channels: int,
        out_targets: Sequence[str],
        lr: float,
        ema_lr: float = 0.001,
        weight_decay: float = 1e-5,
        frozen: bool = True,
    ):
        """Initialize the EfficientNet model.

        Args:
            in_channels: The number of input channels.
            out_targets: The output targets.
            lr: The learning rate.
            ema_lr: The learning rate for the EMA model.
            weight_decay: The weight decay.
            frozen: Whether to freeze the base model.

        Notes:
            - Min input size: 255 x 255
        """
        self.lr = lr
        self.weight_decay = weight_decay

        super().__init__(
            out_targets=out_targets,
            sharpen_temp=0.5,
            mix_beta_alpha=0.75,
        )

        self.eff = efficientnet_b1_backbone(in_channels, frozen)
        self.fc = nn.Sequential(
            nn.Linear(self.EFF_OUT_DIMS, self.n_classes),
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
        return self.fc(self.eff(x))

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )


class EfficientNetB1FixMatchModule(FixMatchModule):
    MIN_SIZE = 255
    EFF_OUT_DIMS = 1280

    def __init__(
        self,
        *,
        in_channels: int,
        out_targets: Sequence[str],
        lr: float,
        weight_decay: float = 1e-5,
        frozen: bool = True,
    ):
        """Initialize the EfficientNet model.

        Args:
            in_channels: The number of input channels.
            out_targets: The output targets.
            lr: The learning rate.
            weight_decay: The weight decay.
            frozen: Whether to freeze the base model.

        Notes:
            - Min input size: 255 x 255
        """
        self.lr = lr
        self.weight_decay = weight_decay

        super().__init__(out_targets=out_targets)

        self.eff = efficientnet_b1_backbone(in_channels, frozen)

        self.fc = nn.Sequential(
            nn.Linear(self.EFF_OUT_DIMS, self.n_classes),
            nn.Softmax(dim=1),
        )

    def forward(self, x: torch.Tensor):
        return self.fc(self.eff(x))

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
