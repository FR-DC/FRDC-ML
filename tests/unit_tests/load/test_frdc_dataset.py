from frdc.conf import BAND_CONFIG
from frdc.load.dataset import FRDCConcatDataset
from frdc.utils import Rect


def test_get_ar_bands_as_dict(ds):
    d = ds.get_ar_bands_as_dict(BAND_CONFIG)
    assert set(d.keys()) == set(d.keys())


def test_get_ar_bands(ds):
    ar, order = ds.get_ar_bands()
    assert ar.shape[-1] == len(BAND_CONFIG)
    assert order == list(BAND_CONFIG.keys())


def test_get_ar_bands_ordering(ds):
    ar, order = ds.get_ar_bands(["WB", "WG"])
    assert ar.shape[-1] == 2
    assert order == ["WB", "WG"]


def test_get_bounds(ds):
    bounds, labels = ds.get_bounds_and_labels()
    assert all([isinstance(b, Rect) for b in bounds])
    assert len(bounds) == len(labels)


def test_ds_add_ds_creates_concat_ds(ds):
    assert isinstance(ds + ds, FRDCConcatDataset)
    assert len(ds + ds) == len(ds) * 2


def test_concat_ds_add_ds_creates_concat_ds(ds):
    cds = ds + ds
    assert isinstance(cds + ds, FRDCConcatDataset)
    assert len(cds + ds) == len(ds) * 3
