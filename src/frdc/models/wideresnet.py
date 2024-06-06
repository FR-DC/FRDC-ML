from __future__ import annotations

from typing import Sequence

import torch
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from torch import nn, softmax
from torch.nn import (
    Conv2d,
    Sequential,
    Identity,
    Linear,
    AdaptiveAvgPool2d,
    Flatten,
    ReLU,
    Softmax,
)

from frdc.models.efficientnetb1 import (
    efficientnet_b1_backbone,
    EfficientNetB1FixMatchModule,
)
from frdc.train.fixmatch_module import FixMatchModule


class ForkAndSum(nn.Module):
    def __init__(self, *mods: nn.Module):
        super().__init__()
        self.mods = mods

    def forward(self, x):
        return sum((mod(x) for mod in self.mods))


class BNReLU(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(x))


class WRNBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        main_strides: Sequence[int] = (1, 1),
        main_paddings: Sequence[int] = (1, 1),
        res_stride: int | None = None,
        res_padding: int | None = None,
    ):
        """Construct a Wide ResNet block.

        Args:
            in_channels: The number of input channels.
            out_channels: The number of output channels.
            main_strides: The strides for the main branch.
            main_paddings: The paddings for the main branch.
            res_stride: The stride for the residual branch. If None, we
                use the Identity.
            res_padding: The padding for the residual branch. If None, we
                  use the Identity.
        """
        super().__init__()
        self.bn_relu = BNReLU(in_channels)

        main_modules = []
        # We construct Conv2d layers with BNReLUs in between
        # E.g. [Conv2d -> BNReLU -> Conv2d -> BNReLU -> ... -> Conv2D]
        #      With 1 element main_... [Conv2d]
        #      With 2              ... [Conv2D -> BNReLU -> Conv2D]
        for e, (stride, padding) in enumerate(
            zip(main_strides, main_paddings)
        ):
            if e == 0:
                main_modules.append(
                    Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=3,
                        stride=stride,
                        padding=padding,
                    )
                )
            else:
                main_modules.append(BNReLU(out_channels))
                main_modules.append(
                    Conv2d(
                        out_channels,
                        out_channels,
                        kernel_size=3,
                        stride=stride,
                        padding=padding,
                    ),
                )

        self.main_branch = Sequential(*main_modules)

        if ((res_stride is not None) and (res_padding is None)) or (
            (res_stride is None) and (res_padding is not None)
        ):
            raise ValueError(
                "If you specify one of residual_stride or residual_padding, "
                "you must specify both. If both are None, we use the Identity."
            )

        if (res_stride is None) and (res_padding is None):
            self.residual_branch = Identity()
        else:
            self.residual_branch = Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=res_stride,
                padding=res_padding,
            )

    def forward(self, x):
        x = self.bn_relu(x)
        x_main = self.main_branch(x)
        x_res = self.residual_branch(x)
        return x_main + x_res


def wideresnet(
    channels: Sequence[int] = (3, 16, 32, 64, 128),
    n_repeats: int = 2,
):
    assert len(channels) == 5, "Channels must be of length 5"
    c = channels

    m = Sequential(
        # B x c[0] x H x W -> B x c[1] x H x W
        Conv2d(c[0], c[1], 3, 1, 1, bias=False),
        # Projects c[1] -> c[2]
        # B x c[1] x H x W -> B x c[2] x H x W
        WRNBlock(c[1], c[2], main_strides=(1, 1), res_stride=1, res_padding=0),
        # Repeats c[2] -> c[2]
        # B x c[2] x H x W -> B x c[2] x H x W
        *[WRNBlock(c[2], c[2], main_strides=(1, 1)) for _ in range(n_repeats)],
        # Projects c[2] -> c[3]
        # B x c[2] x H x W -> B x c[3] x H x W
        WRNBlock(c[2], c[3], main_strides=(2, 1), res_stride=2, res_padding=0),
        # Repeats c[3] -> c[3]
        # B x c[3] x H x W -> B x c[3] x H x W
        *[WRNBlock(c[3], c[3], main_strides=(1, 1)) for _ in range(n_repeats)],
        # Projects c[3] -> c[4]
        # B x c[3] x H x W -> B x c[4] x H x W
        WRNBlock(c[3], c[4], main_strides=(2, 1), res_stride=2, res_padding=0),
        # Repeats c[4] -> c[4]
        # B x c[4] x H x W -> B x c[4] x H x W
        *[WRNBlock(c[4], c[4], main_strides=(1, 1)) for _ in range(n_repeats)],
        ReLU(inplace=True),
        # B x C x H x W -> B x C x 1 x 1
        AdaptiveAvgPool2d(1),
        # B x C x 1 x 1 -> B x C
        Flatten(1, 3),
    )

    return m


class WideResNetB1FixMatchModule(FixMatchModule):
    def __init__(
        self,
        *,
        in_channels: int,
        n_classes: int,
        lr: float,
        x_scaler: StandardScaler,
        y_encoder: OrdinalEncoder,
        weight_decay: float = 1e-5,
    ):
        """ """
        self.lr = lr
        self.weight_decay = weight_decay

        super().__init__(
            n_classes=n_classes,
            x_scaler=x_scaler,
            y_encoder=y_encoder,
        )

        channels = (16, 32, 64, 128)
        self.wrn = wideresnet((in_channels, *channels))
        self.eff = efficientnet_b1_backbone(in_channels, frozen=True)

        self.fc = Sequential(
            Linear(
                channels[-1] + EfficientNetB1FixMatchModule.EFF_OUT_DIMS,
                n_classes,
            ),
            Softmax(dim=1),
        )

    def forward(self, x: torch.Tensor):
        return self.fc(
            torch.cat(
                (self.wrn(x), self.eff(x)),
                dim=1,
            )
        )

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
