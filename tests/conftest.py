import numpy as np
import pytest

from frdc.load import FRDCDataset


@pytest.fixture(scope="session")
def ds() -> FRDCDataset:
    return FRDCDataset._load_debug_dataset()


@pytest.fixture(scope="session")
def ar_and_order(ds) -> tuple[np.ndarray, list[str]]:
    return ds.get_ar_bands()


@pytest.fixture(scope="session")
def debug_file_path():
    return "DEBUG/0/result_Red.tif"


@pytest.fixture(scope="session")
def ar(ar_and_order):
    return ar_and_order[0]


@pytest.fixture(scope="session")
def order(ar_and_order):
    return ar_and_order[1]
