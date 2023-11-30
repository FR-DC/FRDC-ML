from __future__ import annotations

import io
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Callable

import dvc
import dvc.api
import numpy as np
import pandas as pd
import torch
from PIL import Image
from PIL.Image import Image
from torch.utils.data import Dataset, ConcatDataset
from torchvision.transforms.v2 import (
    Compose,
    ToImage,
    ToDtype,
    Resize,
)

from frdc.conf import BAND_CONFIG
from frdc.preprocess.extract_segments import extract_segments_from_bounds
from frdc.utils import Rect


@dataclass
class FRDCDataset(Dataset):
    def __init__(
        self,
        site: str,
        date: str,
        version: str | None,
        transform: Callable[[list[np.ndarray]], list[np.ndarray]] = None,
        target_transform: Callable[[list[str]], list[str]] = None,
    ):
        """Initializes the FRDC Dataset.

        Args:
            site: The site of the dataset, e.g. "chestnut_nature_park".
            date: The date of the dataset, e.g. "20201218".
            version: The version of the dataset, e.g. "183deg".
        """
        self.site = site
        self.date = date
        self.version = version

        self.ar, self.order = self.get_ar_bands()
        bounds, self.targets = self.get_bounds_and_labels()
        self.ar_segments = extract_segments_from_bounds(self.ar, bounds)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.ar_segments)

    def __getitem__(self, idx):
        return (
            self.transform(self.ar_segments[idx])
            if self.transform
            else self.ar_segments[idx],
            self.target_transform(self.targets[idx])
            if self.target_transform
            else self.targets[idx],
        )

    @staticmethod
    def _load_debug_dataset(resize: int = 299) -> FRDCDataset:
        """Loads a debug dataset from Google Cloud Storage.

        Returns:
            A dictionary of the dataset, with keys as the filenames and values
            as the images.
        """
        return FRDCDataset(
            site="DEBUG",
            date="0",
            version=None,
            transform=Compose(
                [
                    ToImage(),
                    ToDtype(torch.float32),
                    Resize((resize, resize)),
                ]
            ),
            target_transform=None,
        )

    @property
    def dataset_dir(self):
        return Path(
            f"{self.site}/{self.date}/"
            f"{self.version + '/' if self.version else ''}"
        )

    def get_ar_bands_as_dict(
        self, bands: Iterable[str] = BAND_CONFIG.keys()
    ) -> dict[str, np.ndarray]:
        """Gets the bands from the dataset as a dictionary of (name, image)

        Notes:
            Use get_ar_bands to get the bands as a concatenated numpy array.
            This is used to preserve the bands separately as keys and values.

        Args:
            bands: The bands to get, e.g. ['WB', 'WG', 'WR']. By default, this
                get all bands in BAND_CONFIG.

        Examples:
            >>> get_ar_bands_as_dict(['WB', 'WG', 'WR']])

            Returns

            >>> {'WB': np.ndarray, 'WG': np.ndarray, 'WR': np.ndarray}

        Returns:
            A dictionary of (KeyName, image) pairs.
        """
        try:
            config = OrderedDict({k: BAND_CONFIG[k] for k in bands})
        except KeyError:
            raise KeyError(
                f"Invalid band name. Valid band names are {BAND_CONFIG.keys()}"
            )

        return {
            name: transform(self.imread(path=self.dataset_dir / path))
            for name, (path, transform) in config.items()
        }

    @staticmethod
    def imread(path: Path) -> np.ndarray:
        """Reads an image from a path into a 3D numpy array. (H, W, C)"""
        b = dvc.api.read(path=path.as_posix(), mode="rb")
        ar = np.asarray(Image.open(io.BytesIO(b)))
        return np.expand_dims(ar, axis=-1) if ar.ndim == 2 else ar

    def get_ar_bands(
        self, bands: Iterable[str] = BAND_CONFIG.keys()
    ) -> tuple[np.ndarray, list[str]]:
        """Gets the bands as a numpy array, and the band order as a list.

        Notes:
            This is a wrapper around get_bands, concatenating the bands.

        Args:
            bands: The bands to get, e.g. ['WB', 'WG', 'WR']. By default, this
                get all bands in BAND_CONFIG.

        Examples
            >>> get_ar_bands(['WB', 'WG', 'WR'])

            Returns

            >>> (np.ndarray, ['WB', 'WG', 'WR'])

        Returns:
            A tuple of (ar, band_order), where ar is a numpy array of shape
            (H, W, C) and band_order is a list of band names.
        """

        d: dict[str, np.ndarray] = self.get_ar_bands_as_dict(bands)
        return np.concatenate(list(d.values()), axis=-1), list(d.keys())

    def get_bounds_and_labels(
        self, file_name="bounds.csv"
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
        fp = dvc.api.read(path=self.dataset_dir / file_name)
        df = pd.read_csv(fp)
        return (
            [Rect(i.x0, i.y0, i.x1, i.y1) for i in df.itertuples()],
            df["name"].tolist(),
        )

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


class FRDCConcatDataset(ConcatDataset):
    def __init__(self, datasets: list[FRDCDataset]):
        super().__init__(datasets)
        self.datasets = datasets
        #

    def __getitem__(self, idx):
        x, y = super().__getitem__(idx)
        return x, y

    @property
    def targets(self):
        return [t for ds in self.datasets for t in ds.targets]
