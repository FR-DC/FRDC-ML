import numpy as np
import pytest
import wandb

from frdc.load.dataset import FRDCDataset
from frdc.load.preset import FRDCDatasetPreset

wandb.init(mode="disabled")


@pytest.fixture(scope="session")
def ds() -> FRDCDataset:
    return FRDCDatasetPreset._debug(use_legacy_bounds=True)


@pytest.fixture(scope="session")
def ar_and_order(ds) -> tuple[np.ndarray, list[str]]:
    return ds._get_ar_bands()


@pytest.fixture(scope="session")
def debug_file_path():
    return "DEBUG/0/result_Red.tif"


@pytest.fixture(scope="session")
def ar(ar_and_order):
    return ar_and_order[0]


@pytest.fixture(scope="session")
def order(ar_and_order):
    return ar_and_order[1]
