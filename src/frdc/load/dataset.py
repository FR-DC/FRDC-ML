from __future__ import annotations

import io
from collections import OrderedDict
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Iterable, Callable, Any

import dvc.api
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, ConcatDataset
from torchvision.transforms.v2 import (
    Compose,
    ToImage,
    ToDtype,
    Resize,
)

from frdc.conf import BAND_CONFIG, ROOT_DIR, LOCAL_DATASET_ROOT_DIR
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
        import sys

        # sys.argv[0] is the path to the running script, not this file.
        # This is important as DVC references that instead of this file.
        # See Also: https://github.com/iterative/dvc/issues/8038
        #           https://github.com/iterative/dvc/issues/8682
        script_path = Path(sys.argv[0])

        # This gets the number of "back" directories to go to get to the root
        # E.g. ../../ is 2 "backs".
        # DVC doesn't seem to support absolute paths,
        # so we need to use relative
        n_backs = len(script_path.relative_to(ROOT_DIR).parts) - 1

        assert n_backs >= 0, "The running script must be in the root directory"

        return Path(
            (f"../" * n_backs) + f"rsc/{self.site}/{self.date}/"
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
            name: transform(self.imread(relpath=relpath))
            for name, (relpath, transform) in config.items()
        }

    def imread(self, relpath: Path) -> np.ndarray:
        """Reads an image from a path into a 3D numpy array. (H, W, C)"""
        b = self.read(relpath=relpath, mode="rb")
        ar = np.asarray(Image.open(io.BytesIO(b)))
        return np.expand_dims(ar, axis=-1) if ar.ndim == 2 else ar

    def read(self, relpath: Path | str, mode: str = "r") -> Any:
        """Reads an image from a path into a PIL Image

        Args:
            relpath: The relative path to the image. This is scoped to the
                dataset directory.
            mode: The mode to open the file in, this is the same as the mode
                in open().
        """
        return dvc.api.read(
            path=(self.dataset_dir / relpath).as_posix(), mode=mode
        )

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
        d = self.read(relpath=file_name)
        df = pd.read_csv(StringIO(d))
        return (
            [Rect(i.x0, i.y0, i.x1, i.y1) for i in df.itertuples()],
            df["name"].tolist(),
        )


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
