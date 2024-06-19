import logging
import warnings
from typing import Dict, Any, Sequence, Callable


def save_unfrozen(
    self,
    checkpoint: Dict[str, Any],
    include_also: Callable[[str], bool] = lambda k: False,
) -> None:
    """Saving only the classifier if frozen.

    Notes:
        This is used for the on_save_checkpoint method of the LightningModule.
        This only reduces the size of the checkpoint, however, loading
        will still require the full model.

        This usually reduces the model size by 99.9%, so it's worth it.

        By default, this will save any parameter that requires grad
        and the BatchNorm running statistics.

    Args:
        self: Not used, but kept for consistency with on_load_checkpoint.
        checkpoint: The checkpoint to save.
        include_also: A function that returns whether to include a parameter,
            on top of any parameter that requires grad and BatchNorm running
            statistics.
    """

    # Keep only the classifier
    new_state_dict = {}

    for k, v in checkpoint["state_dict"].items():
        # We keep 2 things,
        # 1. The BatchNorm running statistics
        # 2. Anything that requires grad

        # BatchNorm running statistics should be kept
        # for closer reconstruction of the model
        is_bn_var = k.endswith(
            ("running_mean", "running_var", "num_batches_tracked")
        )
        try:
            # We need to retrieve it from the original model
            # as the state dict already freezes the model
            is_required_grad = self.get_parameter(k).requires_grad
        except AttributeError:
            if not is_bn_var:
                warnings.warn(
                    f"Unknown non-parameter key in state_dict. {k}."
                    f"This is an edge case where it's not a parameter nor "
                    f"BatchNorm running statistics. This will still be saved."
                )
            is_required_grad = True

        # These are additional parameters to keep
        is_include = include_also(k)

        if is_required_grad or is_bn_var or is_include:
            logging.debug(f"Keeping {k}")
            new_state_dict[k] = v

    checkpoint["state_dict"] = new_state_dict


def load_checkpoint_lenient(self, checkpoint: Dict[str, Any]) -> None:
    """Loading only the classifier if frozen

    Notes:
        This is used for the on_load_checkpoint method of the LightningModule.
        This still supports unfrozen models.
    """

    self.load_state_dict(checkpoint["state_dict"], strict=False)
    checkpoint["state_dict"] = self.state_dict()
