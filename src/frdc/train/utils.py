import numpy as np
import torch
import wandb
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

from frdc.utils.utils import fn_recursive


def mix_up(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mix up the data

    Notes:
        This performs mixup as described in: https://arxiv.org/abs/1710.09412
        However, this is adapted by taking the maximum of the ratio as
        described in MixMatch: https://arxiv.org/abs/1905.02249

    Args:
        x: The data to mix up.
        y: The labels to mix up.
        alpha: The alpha to use for the beta distribution.

    Returns:
        The mixed up data and labels.
    """
    ratio = np.random.beta(alpha, alpha)
    ratio = max(ratio, 1 - ratio)

    shuf_idx = torch.randperm(x.size(0))

    x_mix = ratio * x + (1 - ratio) * x[shuf_idx]
    y_mix = ratio * y + (1 - ratio) * y[shuf_idx]
    return x_mix, y_mix


def sharpen(y: torch.Tensor, temp: float) -> torch.Tensor:
    """Sharpen the predictions by raising them to the power of 1 / temp

    Args:
        y: The predictions to sharpen.
        temp: The temperature to use.

    Returns:
        The probability-normalized sharpened predictions
    """
    y_sharp = y ** (1 / temp)
    # Sharpening will change the sum of the predictions.
    y_sharp /= y_sharp.sum(dim=1, keepdim=True)
    return y_sharp


def wandb_hist(x: torch.Tensor, num_bins: int) -> wandb.Histogram:
    """Records a W&B Histogram"""
    return wandb.Histogram(
        torch.flatten(x).detach().cpu().tolist(),
        num_bins=num_bins,
    )
