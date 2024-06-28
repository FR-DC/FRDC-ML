from __future__ import annotations

import logging
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Callable, Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.preprocessing import StandardScaler
from torch import rot90
from torch.utils.data import Dataset, ConcatDataset
from torchvision.transforms.v2 import Compose
from torchvision.transforms.v2.functional import hflip
from torchvision.tv_tensors import Image as ImageTensor

from frdc.conf import BAND_CONFIG, LABEL_STUDIO_CLIENT
from frdc.load import gcs
from frdc.load.label_studio import get_task
from frdc.preprocess.extract_segments import (
    extract_segments_from_bounds,
    extract_segments_from_polybounds,
)
from frdc.utils.utils import Rect, flatten_nested, map_nested

logger = logging.getLogger(__name__)


class ImageStandardScaler(StandardScaler):
    def fit(self, X, y=None, sample_weight=None):
        X = X.reshape(X.shape[0], -1)
        return super().fit(X, y, sample_weight)

    def transform(self, X, copy=None):
        shape = X.shape
        X = X.reshape(shape[0], -1)
        X = torch.nan_to_num(X, nan=0)
        X = super().transform(X, copy).reshape(*shape)
        X = torch.tensor(X)
        return X

    def transform_one(self, X, copy=None):
        shape = X.shape
        X = X.reshape(1, -1)
        return self.transform(X, copy).reshape(shape)

    def inverse_transform(self, X, y=None, **fit_params):
        shape = X.shape
        X = X.reshape(shape[0], -1)
        return (
            super()
            .inverse_transform(X, y, **fit_params)
            .reshape(shape[0], *shape[1:])
        )

    def fit_nested(self, X):
        # Adapted method of `fit` to handle nested lists/tuples
        X = torch.stack(flatten_nested(X, type_list=(list, tuple)))
        self.fit(X)
        return self

    def transform_nested(self, X):
        # Adapted method of `transform` to handle nested lists/tuples
        # This preserves the nested structure of the input by treating every
        # atom as a single entity and transforming as-is.
        return map_nested(X, self.transform_one, ImageTensor, (list, tuple))


@dataclass
class FRDCDataset(Dataset):
    def __init__(
        self,
        site: str,
        date: str,
        version: str | None,
        transform: Compose = lambda x: x,
        target_transform: Callable[[str], str] = lambda x: x,
        use_legacy_bounds: bool = False,
        polycrop: bool = False,
        polycrop_value: Any = np.nan,
    ):
        """Initializes the FRDC Dataset.

        Notes:
            We recommend to check FRDCDatasetPreset if you want to use a
            pre-defined dataset.

            You can concatenate datasets using the addition operator, e.g.::

                ds = FRDCDataset(...) + FRDCDataset(...)

            This will return a FRDCConcatDataset, see FRDCConcatDataset for
            more information.

        Args:
            site: The site of the dataset, e.g. "chestnut_nature_park".
            date: The date of the dataset, e.g. "20201218".
            version: The version of the dataset, e.g. "183deg".
            transform: The transform to apply to each segment.
            target_transform: The transform to apply to each label.
            use_legacy_bounds: Whether to use the legacy bounds.csv file.
                This will automatically be set to True if LABEL_STUDIO_CLIENT
                is None, which happens when Label Studio cannot be connected
                to.
            polycrop: Whether to further crop the segments via its polygon
                bounds. The cropped area will be padded with np.nan.
            polycrop_value: The value to pad the cropped area with.
        """
        self.site = site
        self.date = date
        self.version = version

        self.ar, self.band_order = self._get_ar_bands()
        self.targets = None

        if use_legacy_bounds:
            bounds, self.targets = self._get_legacy_bounds_and_labels()
            self.ar_segments = extract_segments_from_bounds(self.ar, bounds)
        else:
            if LABEL_STUDIO_CLIENT:
                bounds, self.targets = self._get_polybounds_and_labels()
                self.ar_segments = extract_segments_from_polybounds(
                    self.ar,
                    bounds,
                    cropped=True,
                    polycrop=polycrop,
                    polycrop_value=polycrop_value,
                )
            else:
                raise ConnectionError(
                    "Cannot connect to Label Studio, cannot use live bounds. "
                    "Retry with use_legacy_bounds=True to attempt to use the "
                    "legacy bounds.csv file."
                )

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.ar_segments)

    def __getitem__(self, idx):
        return (
            self.transform(self.ar_segments[idx]),
            self.target_transform(self.targets[idx]),
        )

    @property
    def dataset_dir(self):
        """Returns the path format of the dataset."""
        return Path(
            f"{self.site}/{self.date}/"
            f"{self.version + '/' if self.version else ''}"
        )

    def _get_ar_bands_as_dict(
        self,
        bands: Iterable[str] = BAND_CONFIG.keys(),
    ) -> dict[str, np.ndarray]:
        """Gets the bands from the dataset as a dictionary of (name, image)

        Notes:
            Use get_ar_bands to get the bands as a concatenated numpy array.
            This is used to preserve the bands separately as keys and values.

        Args:
            bands: The bands to get, e.g. ['WB', 'WG', 'WR']. By default, this
                get all bands in BAND_CONFIG.

        Examples:
            >>> self._get_ar_bands_as_dict(['WB', 'WG', 'WR']])

            Returns

            >>> {'WB': np.ndarray, 'WG': np.ndarray, 'WR': np.ndarray}

        Returns:
            A dictionary of (KeyName, image) pairs.
        """
        d = {}
        fp_cache = {}

        try:
            config = OrderedDict({k: BAND_CONFIG[k] for k in bands})
        except KeyError:
            raise KeyError(
                f"Invalid band name. Valid band names are {BAND_CONFIG.keys()}"
            )

        for band_name, (glob, band_transform) in config.items():
            fp = gcs.download(fp=self.dataset_dir / glob)

            # We may use the same file multiple times, so we cache it
            if fp in fp_cache:
                logging.debug(f"Cache hit for {fp}, using cached image...")
                im_band = fp_cache[fp]
            else:
                logging.debug(f"Cache miss for {fp}, loading...")
                im_band = self._load_image(fp)
                fp_cache[fp] = im_band

            d[band_name] = band_transform(im_band)

        return d

    def _get_ar_bands(
        self,
        bands: Iterable[str] = BAND_CONFIG.keys(),
    ) -> tuple[np.ndarray, list[str]]:
        """Gets the bands as a numpy array, and the band order as a list.

        Notes:
            This is a wrapper around get_bands, concatenating the bands.

        Args:
            bands: The bands to get, e.g. ['WB', 'WG', 'WR']. By default, this
                get all bands in BAND_CONFIG.

        Examples
            >>> self._get_ar_bands(['WB', 'WG', 'WR'])

            Returns

            >>> (np.ndarray, ['WB', 'WG', 'WR'])

        Returns:
            A tuple of (ar, band_order), where ar is a numpy array of shape
            (H, W, C) and band_order is a list of band names.
        """

        d: dict[str, np.ndarray] = self._get_ar_bands_as_dict(bands)
        return np.concatenate(list(d.values()), axis=-1), list(d.keys())

    def _get_legacy_bounds_and_labels(
        self,
        file_name="bounds.csv",
    ) -> tuple[list[Rect], list[str]]:
        """Gets the bounds and labels from the bounds.csv file.

        Notes:
            In the context of np.ndarray, to slice with x, y coordinates,
            you need to slice with [y0:y1, x0:x1]. Which is different from the
            bounds.csv file.

        Args:
            file_name: The name of the bounds.csv file.

        Returns:
            A tuple of (bounds, labels), where bounds is a list of
            (x0, y0, x1, y1) and labels is a list of labels.
        """
        logger.warning(
            "Using legacy bounds.csv file for dataset."
            "This is pending to be deprecated in favour of pulling "
            "annotations from Label Studio."
        )
        try:
            fp = gcs.download(fp=self.dataset_dir / file_name)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"bounds.csv not found in {self.dataset_dir}. "
                f"Please check the file exists."
            )
        df = pd.read_csv(fp)
        return (
            [Rect(i.x0, i.y0, i.x1, i.y1) for i in df.itertuples()],
            df["name"].tolist(),
        )

    def _get_polybounds_and_labels(self):
        """Gets the bounds and labels from Label Studio."""
        return get_task(
            Path(f"{self.dataset_dir}/result.jpg")
        ).get_bounds_and_labels()

    @staticmethod
    def _load_image(path: Path | str) -> np.ndarray:
        """Loads an Image from a path into a 3D numpy array. (H, W, C)

        Notes:
            If the image has only 1 channel, then it will be (H, W, 1) instead

        Args:
            path: Path to image. pathlib.Path is preferred, but str is also
                accepted.

        Returns:
            3D Image as numpy array.
        """

        im = Image.open(Path(path).as_posix())
        ar = np.asarray(im)
        return np.expand_dims(ar, axis=-1) if ar.ndim == 2 else ar

    def __add__(self, other) -> FRDCConcatDataset:
        return FRDCConcatDataset([self, other])


class FRDCUnlabelledDataset(FRDCDataset):
    """An implementation of FRDCDataset that masks away the labels.

    Notes:
        If you already have a FRDCDataset, you can simply set __class__ to
        FRDCUnlabelledDataset to achieve the same behaviour::

            ds.__class__ = FRDCUnlabelledDataset

        This will replace the __getitem__ method with the one below.

        However, it's also perfectly fine to initialize this directly::

            ds_unl = FRDCUnlabelledDataset(...)
    """

    def __getitem__(self, item):
        return (
            self.transform(self.ar_segments[item])
            if self.transform
            else self.ar_segments[item]
        )


class FRDCConstRotatedDataset(FRDCDataset):
    def __len__(self):
        """Assume that the dataset is 8x larger than it actually is.

        There are 8 possible orientations for each image.
        1.       As-is
        2, 3, 4. Rotated 90, 180, 270 degrees
        5.       Horizontally flipped
        6, 7, 8. Horizontally flipped and rotated 90, 180, 270 degrees
        """
        return super().__len__() * 8

    def __getitem__(self, idx):
        """Alter the getitem method to implement the logic above."""
        x, y = super().__getitem__(int(idx // 8))
        assert x.ndim == 3, "x must be a 3D tensor"
        x_ = None
        if idx % 8 == 0:
            x_ = x
        elif idx % 8 == 1:
            x_ = rot90(x, 1, (1, 2))
        elif idx % 8 == 2:
            x_ = rot90(x, 2, (1, 2))
        elif idx % 8 == 3:
            x_ = rot90(x, 3, (1, 2))
        elif idx % 8 == 4:
            x_ = hflip(x)
        elif idx % 8 == 5:
            x_ = hflip(rot90(x, 1, (1, 2)))
        elif idx % 8 == 6:
            x_ = hflip(rot90(x, 2, (1, 2)))
        elif idx % 8 == 7:
            x_ = hflip(rot90(x, 3, (1, 2)))

        return x_, y


class FRDCConcatDataset(ConcatDataset):
    """ConcatDataset for FRDCDataset.

    Notes:
        This handles concatenating the targets when you add two datasets
        together, furthermore, implements the addition operator to
        simplify the syntax.

    Examples:
        If you have two datasets, ds1 and ds2, you can concatenate them::

            ds = ds1 + ds2

        `ds` will be a FRDCConcatDataset, which is a subclass of ConcatDataset.

        You can further add to a concatenated dataset::

            ds = ds1 + ds2
            ds = ds + ds3

        Finallu, all concatenated datasets have the `targets` property, which
        is a list of all the targets in the datasets::

            (ds1 + ds2).targets == ds1.targets + ds2.targets
    """

    def __init__(self, datasets: list[FRDCDataset]):
        super().__init__(datasets)
        self.datasets: list[FRDCDataset] = datasets

    @property
    def targets(self):
        return [t for ds in self.datasets for t in ds.targets]

    def __add__(self, other: FRDCDataset) -> FRDCConcatDataset:
        return FRDCConcatDataset([*self.datasets, other])
