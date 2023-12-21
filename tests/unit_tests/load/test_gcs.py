import pytest

from frdc.load.gcs import download, list_gcs_datasets


def test_download_multiple_files():
    """Test that download_file shouldn't support multiple files."""
    with pytest.raises(ValueError):
        download(fp="**/*.tif")


def test_list_datasets():
    df = list_gcs_datasets()
    assert len(df) > 0
