from collections import OrderedDict
from pathlib import Path

ROOT_DIR = Path(__file__).parents[2]
LOCAL_DATASET_ROOT_DIR = ROOT_DIR / "rsc"
SECRETS_DIR = ROOT_DIR / ".secrets"
GCS_PROJECT_ID = "frmodel"
GCS_BUCKET_NAME = "frdc-scan"

BAND_CONFIG = OrderedDict(
    {
        "WB": ("result.tif", lambda x: x[..., 2:3]),
        "WG": ("result.tif", lambda x: x[..., 1:2]),
        "WR": ("result.tif", lambda x: x[..., 0:1]),
        "NB": ("result_Blue.tif", lambda x: x),
        "NG": ("result_Green.tif", lambda x: x),
        "NR": ("result_Red.tif", lambda x: x),
        "RE": ("result_RedEdge.tif", lambda x: x),
        "NIR": ("result_NIR.tif", lambda x: x),
    }
)

BAND_MAX_CONFIG: dict[str, tuple[int, int]] = {
    "WR": (0, 2**8),
    "WG": (0, 2**8),
    "WB": (0, 2**8),
    "NR": (0, 2**14),
    "NG": (0, 2**14),
    "NB": (0, 2**14),
    "RE": (0, 2**14),
    "NIR": (0, 2**14),
}
