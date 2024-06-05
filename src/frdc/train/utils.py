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


def preprocess(
    x_lbl: torch.Tensor,
    y_lbl: torch.Tensor,
    x_scaler: StandardScaler,
    y_encoder: OrdinalEncoder,
    x_unl: list[torch.Tensor] = None,
    nan_mask: bool = True,
) -> tuple[tuple[torch.Tensor, torch.Tensor], list[torch.Tensor]]:
    """Preprocesses the data

    Notes:
        The reason why x and y's preprocessing is coupled is due to the NaN
        elimination step. The NaN elimination step is due to unseen labels by y

        fn_recursive is to recursively apply some function to a nested list.
        This happens due to unlabelled being a list of tensors.

    Args:
        x_lbl: The data to preprocess.
        y_lbl: The labels to preprocess.
        x_scaler: The StandardScaler to use.
        y_encoder: The OrdinalEncoder to use.
        x_unl: The unlabelled data to preprocess.
        nan_mask: Whether to remove nan values from the batch.

    Returns:
        The preprocessed data and labels.
    """

    x_unl = [] if x_unl is None else x_unl

    x_lbl_trans = x_standard_scale(x_scaler, x_lbl)
    y_trans = y_encode(y_encoder, y_lbl)
    x_unl_trans = fn_recursive(
        x_unl,
        fn=lambda x: x_standard_scale(x_scaler, x),
        type_atom=torch.Tensor,
        type_list=list,
    )

    # Remove nan values from the batch
    #   Ordinal Encoders can return a np.nan if the value is not in the
    #   categories. We will remove that from the batch.
    nan = (
        ~torch.isnan(y_trans) if nan_mask else torch.ones_like(y_trans).bool()
    )
    x_lbl_trans = x_lbl_trans[nan]
    x_lbl_trans = torch.nan_to_num(x_lbl_trans)
    x_unl_trans = fn_recursive(
        x_unl_trans,
        fn=lambda x: torch.nan_to_num(x[nan]),
        type_atom=torch.Tensor,
        type_list=list,
    )
    y_trans = y_trans[nan]

    return (x_lbl_trans, y_trans.long()), x_unl_trans


def x_standard_scale(
    x_scaler: StandardScaler, x: torch.Tensor
) -> torch.Tensor:
    """Standard scales the data

    Notes:
        This is a wrapper around the StandardScaler to handle PyTorch tensors.

    Args:
        x_scaler: The StandardScaler to use.
        x: The data to standard scale, of shape (B, C, H, W).
    """
    # Standard Scaler only accepts (n_samples, n_features),
    # so we need to do some fancy reshaping.
    # Note that moving dimensions then reshaping is different from just
    # reshaping!

    # Move Channel to the last dimension then transform
    # B x C x H x W -> B x H x W x C
    b, c, h, w = x.shape
    x_ss = x_scaler.transform(x.permute(0, 2, 3, 1).reshape(-1, c))

    # Move Channel back to the second dimension
    # B x H x W x C -> B x C x H x W
    return torch.nan_to_num(
        torch.from_numpy(x_ss.reshape(b, h, w, c)).permute(0, 3, 1, 2).float()
    )


def y_encode(y_encoder: OrdinalEncoder, y: torch.Tensor) -> torch.Tensor:
    """Encodes the labels

    Notes:
        This is a wrapper around the OrdinalEncoder to handle PyTorch tensors.

    Args:
        y_encoder: The OrdinalEncoder to use.
        y: The labels to encode.
    """
    return torch.from_numpy(
        y_encoder.transform(np.array(y).reshape(-1, 1))[..., 0]
    )


def wandb_hist(x: torch.Tensor, num_bins: int) -> wandb.Histogram:
    """Records a W&B Histogram"""
    return wandb.Histogram(
        torch.flatten(x).detach().cpu().tolist(),
        num_bins=num_bins,
    )
