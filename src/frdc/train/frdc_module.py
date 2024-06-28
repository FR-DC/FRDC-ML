from typing import Any, Dict, Sequence

import numpy as np
import torch
from lightning import LightningModule
from sklearn.preprocessing import OrdinalEncoder

from frdc.models.utils import save_unfrozen, load_checkpoint_lenient
from frdc.utils.utils import map_nested


class FRDCModule(LightningModule):
    def __init__(
        self,
        *,
        out_targets: Sequence[str],
        nan_mask_missing_y_labels: bool = True,
    ):
        """Base Lightning Module for MixMatch

        Notes:
            This is the base class for MixMatch and FixMatch.
            This implements the Y-Encoder logic so that all modules can
            encode and decode the tree string labels.

            Generally the hierarchy is:
                <Model><Architecture>Module
                -> <Architecture>Module
                -> FRDCModule

                E.g.
                EfficientNetB1MixMatchModule
                -> MixMatchModule
                -> FRDCModule

                WideResNetFixMatchModule
                -> FixMatchModule
                -> FRDCModule

        Args:
            out_targets: The output targets for the model.
            nan_mask_missing_y_labels: Whether to mask away x values that
                have missing y labels. This happens when the y label is not
                present in the OrdinalEncoder's categories, which happens
                during non-training steps. E.g. A new unseen tree is inferred.
        """

        super().__init__()

        self.y_encoder: OrdinalEncoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=np.nan,
        )
        self.y_encoder.fit(np.array(out_targets).reshape(-1, 1))
        self.nan_mask_missing_y_labels = nan_mask_missing_y_labels
        self.save_hyperparameters()

    @property
    def n_classes(self):
        return len(self.y_encoder.categories_[0])

    @torch.no_grad()
    def on_before_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        """This method is called before any data transfer to the device.

        Notes:
            This method wraps OrdinalEncoder to convert labels from str to int
            before transferring the data to the device.

            Note that this step must happen before the transfer as tensors
            don't support str types.

            PyTorch Lightning may complain about this being on the Module
            instead of the DataModule. However, this is intentional as we
            want to export the model alongside the transformations.
        """

        if self.training:
            (x_lbl, y_lbl), x_unl = batch
        else:
            x_lbl, y_lbl = batch
            x_unl = []

        y_trans = torch.from_numpy(
            self.y_encoder.transform(np.array(y_lbl).reshape(-1, 1))[..., 0]
        )

        # Remove nan values from the batch
        #   Ordinal Encoders can return a np.nan if the value is not in the
        #   categories. We will remove that from the batch.
        nan = (
            ~torch.isnan(y_trans)  # Keeps all non-nan values
            if self.nan_mask_missing_y_labels
            else torch.ones_like(y_trans).bool()  # Keeps all values
        )

        x_lbl_trans = torch.nan_to_num(x_lbl[nan])

        # This function applies nan_to_num to all tensors in the list,
        # regardless of how deeply nested they are.
        x_unl_trans = map_nested(
            x_unl,
            fn=lambda x: torch.nan_to_num(x[nan]),
            type_atom=torch.Tensor,
            type_list=list,
        )
        y_trans = y_trans[nan].long()

        return (x_lbl_trans, y_trans), x_unl_trans

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        save_unfrozen(self, checkpoint)

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        load_checkpoint_lenient(self, checkpoint)

    # The following methods are to enforce the batch schema typing.
    def training_step(
        self,
        batch: tuple[tuple[torch.Tensor, torch.Tensor], list[torch.Tensor]],
        batch_idx: int,
    ):
        ...

    def validation_step(
        self,
        batch: tuple[tuple[torch.Tensor, torch.Tensor], list[torch.Tensor]],
        batch_idx: int,
    ):
        ...

    def test_step(
        self,
        batch: tuple[tuple[torch.Tensor, torch.Tensor], list[torch.Tensor]],
        batch_idx: int,
    ):
        ...

    def predict_step(
        self,
        batch: tuple[tuple[torch.Tensor, torch.Tensor], list[torch.Tensor]],
    ) -> Any:
        ...
