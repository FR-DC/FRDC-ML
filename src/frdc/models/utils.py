from typing import Dict, Any, Sequence


def on_save_checkpoint(
    self,
    checkpoint: Dict[str, Any],
    saved_module_prefixes: Sequence[str] = ("fc.",),
    saved_module_suffixes: Sequence[str] = (
        "running_mean",
        "running_var",
        "num_batches_tracked",
    ),
) -> None:
    """Saving only the classifier if frozen.

    Notes:
        This is used for the on_save_checkpoint method of the LightningModule.
        This only reduces the size of the checkpoint, however, loading
        will still require the full model.

        This usually reduces the model size by 99.9%, so it's worth it.

        By default, this will save the classifier and the BatchNorm running
        statistics.

    Args:
        self: Not used, but kept for consistency with on_load_checkpoint.
        checkpoint: The checkpoint to save.
        saved_module_prefixes: The prefixes of the modules to save.
        saved_module_suffixes: The suffixes of the modules to save.
    """

    if checkpoint["hyper_parameters"]["frozen"]:
        # Keep only the classifier
        checkpoint["state_dict"] = {
            k: v
            for k, v in checkpoint["state_dict"].items()
            if (
                k.startswith(saved_module_prefixes)
                or k.endswith(saved_module_suffixes)
            )
        }


def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
    """Loading only the classifier if frozen

    Notes:
        This is used for the on_load_checkpoint method of the LightningModule.
        This still supports unfrozen models.
    """

    self.load_state_dict(checkpoint["state_dict"], strict=False)
    checkpoint["state_dict"] = self.state_dict()
