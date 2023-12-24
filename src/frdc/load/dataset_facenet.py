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
from torch.utils.data import Dataset, ConcatDataset

from frdc.conf import (
    BAND_CONFIG,
    LABEL_STUDIO_CLIENT,
)
from frdc.load.gcs import download
from frdc.load.label_studio import get_task
from frdc.preprocess.extract_segments import (
    extract_segments_from_bounds,
    extract_segments_from_polybounds,
)
from frdc.utils import Rect

from .dataset import FRDCDataset

logger = logging.getLogger(__name__)

class FRDCDataset_Facenet(FRDCDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx):
        gotten = super().__getitem__(idx)
        #Before: WB, WG, WR, NB, NG, NR, RE, NIR
        #After: RedEdge, Blue, NIR, Red, Green (Narrowbands)
        #TODO: Last 5 bands should be after GLCM mean filter
        #TODO: Pick the bands in non-hardcoded manner.
        return (np.concatenate((gotten[0][[6,3,7,5,4],:,:],)*2), gotten[1])
