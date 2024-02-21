from copy import deepcopy

import torch
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from torch import nn
from torchvision.models import (
    EfficientNet,
    efficientnet_b1,
    EfficientNet_B1_Weights,
)

from frdc.train.mixmatch_module import MixMatchModule
from frdc.utils.ema import EMA


class EfficientNetB1MixMatchModule(MixMatchModule):
    MIN_SIZE = 320
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
            - Min input size: 320 x 320
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

        self.eff = efficientnet_b1(
            weights=EfficientNet_B1_Weights.IMAGENET1K_V2
        )

        # Remove the final layer
        self.eff.classifier = nn.Identity()

        if frozen:
            for param in self.eff.parameters():
                param.requires_grad = False

        # Adapt the first layer to accept the number of channels
        self.eff = self.adapt_efficient_multi_channel(self.eff, in_channels)

        self.fc = nn.Sequential(
            nn.Linear(self.EFF_OUT_DIMS, self.EFF_OUT_DIMS // 2),
            nn.BatchNorm1d(self.EFF_OUT_DIMS // 2),
            nn.Linear(self.EFF_OUT_DIMS // 2, n_classes),
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

    @staticmethod
    def adapt_efficient_multi_channel(
        eff: EfficientNet,
        in_channels: int,
    ) -> EfficientNet:
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
            1, 5, 1, 1
        )
        eff.features[0][0] = new_conv

        return eff

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
