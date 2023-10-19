from pathlib import Path

import numpy as np
import pytest

from frdc.utils.file_cache import file_cache


@pytest.mark.cupy
def test_caching():
    cache_dir = Path(__file__).parent / 'cache'
    cache_dir.mkdir(exist_ok=True, parents=True)
    for f in cache_dir.glob("*.npy"):
        f.unlink()

    calls = 0

    @file_cache(
        fn_cache_fp=lambda x: cache_dir / f"{x}.npy",
        fn_save_object=np.save,
        fn_load_object=np.load,
    )
    def my_fn(x):
        nonlocal calls
        calls += 1
        np.random.seed(x)
        return np.random.randint(0, 10, [5, 5])

    my_fn(1)
    assert calls == 1
    my_fn(1)
    assert calls == 1
    my_fn(2)
    assert calls == 2

    for f in cache_dir.glob("*.npy"):
        f.unlink()