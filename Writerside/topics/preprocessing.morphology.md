# preprocessing.morphology

<tldr>
Performs morphological operations on the NDArray bands.
</tldr>

> This is currently only used for auto-segmentation. If you want to use
> predefined segmentation see
> [preprocessing.extract_segments](preprocessing.extract_segments.md).

## Functions

<warning>
Assumes shape H x W x C, where C is the number of bands.
</warning>

<deflist type="medium">
<def title="threshold_binary_mask">
<b>Thresholds a selected NDArray bands to yield a binary mask.</b>
</def>
<def title="binary_watershed">
<b>Performs watershed on a binary mask to yield a mapped label
classification</b>
</def>
</deflist>

## Usage

Perform auto-segmentation on a dataset to yield a label classification.

```python
from frdc.load.preset import FRDCDatasetPreset
from frdc.preprocess.morphology import (
    threshold_binary_mask, binary_watershed
)

ds = FRDCDatasetPreset.chestnut_20201218()
ar, order = ds.get_ar_bands()
mask = threshold_binary_mask(ar, order.index('NIR'), 90 / 256)
ar_label = binary_watershed(mask)
```

## API

<deflist>
<def title="threshold_binary_mask(ar, band_idx, threshold_value)">
<b>Thresholds a selected NDArray bands to yield a binary mask as 
<code>np.ndarray</code></b><br/>
This is equivalent to 
<code-block lang="python">
ar[..., band_idx] > threshold_value
</code-block>
</def>
<def title="binary_watershed(ar_mask, peaks_footprint, watershed_compactness)">
<b>Performs watershed on a binary mask to yield a mapped label
classification as a <code>np.ndarray</code></b><br/>
<list>
<li> <code>peaks_footprint</code> is the footprint of 
<code>skimage.feature.peak_local_max</code> </li>
<li> <code>watershed_compactness</code> is the compactness of
<code>skimage.morphology.watershed</code> </li>
</list>
</def>
</deflist>