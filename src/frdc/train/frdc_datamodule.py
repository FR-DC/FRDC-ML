from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from lightning import LightningDataModule
from torch.utils.data import DataLoader, RandomSampler

from frdc.load.dataset import FRDCDataset, FRDCUnlabelledDataset
from frdc.train.stratified_sampling import RandomStratifiedSampler


@dataclass
class FRDCDataModule(LightningDataModule):
    """Lightning DataModule for FRDC Dataset.

    Notes:
        This is a special datamodule for semi-supervised learning, which
        can accept an optional dataloaders for an unlabelled dataset.

        Without an unsupervised dataset it can be used for supervised learning,
        by passing in None for the unlabelled dataset.

        If you're using our MixMatch Module, using None for the unlabelled
        dataset will skip the MixMatch. However, note that this is not
        equivalent to passing the Labelled set as unlabelled as well.

        For example::

            FRDCDataModule(
                train_lab_ds=train_lab_ds,
                train_unl_ds=train_lab_ds,
                ...
            )

        Does not have the same performance as::

            FRDCDataModule(
                train_lab_ds=train_lab_ds,
                train_unl_ds=None,
                ...
            )

        As partially, some samples in MixMatch uses the unlabelled loss.

    Args:
        train_lab_ds: The labelled training dataset.
        train_unl_ds: The unlabelled training dataset. Can be None, which will
            default to a DataModule suitable for supervised learning. If
            train_unl_ds is a FRDCDataset, it will be converted to a
            FRDCUnlabelledDataset, which simply masks away the labels.
        val_ds: The validation dataset.
        batch_size: The batch size to use for the dataloaders.
        train_iters: The number of iterations to run for the labelled training
            dataset.

    """

    train_lab_ds: FRDCDataset
    val_ds: FRDCDataset
    train_unl_ds: FRDCDataset | FRDCUnlabelledDataset | None = None
    batch_size: int = 4
    train_iters: int = 100
    sampling_strategy: Literal["stratified", "random"] = "stratified"

    def __post_init__(self):
        super().__init__()

        if isinstance(self.train_unl_ds, FRDCDataset):
            self.train_unl_ds.__class__ = FRDCUnlabelledDataset

    def train_dataloader(self):
        n_samples = self.batch_size * self.train_iters
        if self.sampling_strategy == "stratified":
            sampler_fn = lambda ds: RandomStratifiedSampler(
                ds.targets,
                num_samples=n_samples,
            )
        elif self.sampling_strategy == "random":
            sampler_fn = lambda ds: RandomSampler(
                ds,
                num_samples=n_samples,
            )
        else:
            raise ValueError(f"Invalid strategy: {self.sampling_strategy}")

        lab_dl = DataLoader(
            self.train_lab_ds,
            batch_size=self.batch_size,
            sampler=sampler_fn(self.train_lab_ds),
        )
        unl_dl = (
            DataLoader(
                self.train_unl_ds,
                batch_size=self.batch_size,
                sampler=sampler_fn(self.train_unl_ds),
            )
            if self.train_unl_ds is not None
            # This is a hacky way to create an empty dataloader.
            # The size should be the same or larger than the
            # labelled dataloader so the iterator doesn't prematurely stop.
            else DataLoader([[] for _ in range(len(lab_dl))])
        )

        return [lab_dl, unl_dl]

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size)
