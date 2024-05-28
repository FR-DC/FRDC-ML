from typing import Dict, Any, Sequence


def on_save_checkpoint(
    self,
    checkpoint: Dict[str, Any],
    saved_module_prefixes: Sequence[str] = ("fc.",),
) -> None:
    """Saving only the classifier if frozen.

    Notes:
        This only reduces the size of the checkpoint, however, loading
        will still require the full model.

        This reduces the model size from 200MB to 300kB.

    Args:
        self: Not used, but kept for consistency with on_load_checkpoint.
        checkpoint: The checkpoint to save.
        saved_module_prefixes: The prefixes to save.
    """

    if checkpoint["hyper_parameters"]["frozen"]:
        # Keep only the classifier
        checkpoint["state_dict"] = {
            k: v
            for k, v in checkpoint["state_dict"].items()
            if k.startswith(saved_module_prefixes)
        }


def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
    """Loading only the classifier if frozen

    Notes:
        This still supports unfrozen models.
    """

    self.load_state_dict(checkpoint["state_dict"], strict=False)
    checkpoint["state_dict"] = self.state_dict()
