# Retrieve our Datasets

<tldr>A tutorial to retrieve our datasets</tldr>

In this tutorial, we'll learn how to :

- Retrieve FRDC's Hyperspectral Image Data as `np.ndarray`
- Retrieve FRDC's Ground Truth bounds and labels
- Slice/segment the image data by the bounds

## Prerequisites

- New here? [Get Started](Getting-Started.md).
- Setup the Google Cloud Authorization to download the data.

## Retrieve the Data

To retrieve the data, use [FRDCDataset](load.dataset.md)

Here, we'll download and load our

- `ar`: Hyperspectral Image Data
- `order`: The order of the bands
- `bounds`: The bounds of the trees (segments)
- `labels`: The labels of the trees (segments)

```python
from frdc.load.dataset import FRDCDataset

ds = FRDCDataset(site="chestnut_nature_park", date="20201218", version=None)
ar, order = ds.get_ar_bands()
bounds, labels = ds.get_bounds_and_labels()
```

### What Datasets are there? {collapsible="true"}

> To know what datasets are available, you can run
> [load.gcs](load.gcs.md)'s `list_gcs_datasets()`
> method

> Note that some datasets do not have `bounds` and `labels` available as they
> have not been annotated yet.
> {style='warning'}

```python
from frdc.load.gcs import list_gcs_datasets 
print(list_gcs_datasets())
# 0  DEBUG/0
# 1  casuarina/20220418/183deg
# 2  casuarina/20220418/93deg
# 3  chestnut_nature_park/20201218
# ...
```

- The first part of the path is the `site`, and the second part is the `date`.
- The `version` is the rest of the path, if there isn't any, use `None`.

<tabs>
<tab title="ds/date/ver/">
<list>
<li><code>site=&quot;ds&quot;</code></li>
<li><code>date=&quot;date&quot;</code></li>
<li><code>version=&quot;ver&quot;</code></li>
</list>
</tab>
<tab title="ds/date/ver/01/data/">
<list>
<li><code>site=&quot;ds&quot;</code></li>
<li><code>date=&quot;date&quot;</code></li>
<li><code>version=&quot;ver/01/data&quot;</code></li>
</list>
</tab>
<tab title="ds/date/">
<list>
<li><code>site=&quot;ds&quot;</code></li>
<li><code>date=&quot;date&quot;</code></li>
<li><code>version=None</code></li>
</list>
</tab>
</tabs>

## Segment the Data

To segment the data, use [Extract Segments](preprocessing.extract_segments.md).

Here, we'll segment the data by the bounds.

```python
from frdc.load.dataset import FRDCDataset
from frdc.preprocess.extract_segments import extract_segments_from_bounds

ds = FRDCDataset(site="chestnut_nature_park", date="20201218", version=None)
ar, order = ds.get_ar_bands()
bounds, labels = ds.get_bounds_and_labels()
segments = extract_segments_from_bounds(ar, bounds)
```

`segments` is a list of `np.ndarray` of shape H, W, C, representing a tree.
The order of `segments` is the same as `labels`, so you can use `labels` to
identify the tree.

> While we have not used `order` in our example, it's useful to determine the
> order of the bands in `ar` in other applications.

## Plot the Data (Optional) {collapsible="true"}

We can then use these data to plot out the first tree segment.

```python
import matplotlib.pyplot as plt

from frdc.load.dataset import FRDCDataset
from frdc.preprocess.extract_segments import extract_segments_from_bounds
from frdc.preprocess.scale import scale_0_1_per_band

ds = FRDCDataset(site="chestnut_nature_park", date="20201218", version=None)
ar, order = ds.get_ar_bands()
bounds, labels = ds.get_bounds_and_labels()
segments = extract_segments_from_bounds(ar, bounds)
segment_0_bgr = segments[0]
segment_0_rgb = segment_0_bgr[..., [2, 1, 0]]
segment_0_rgb_scaled = scale_0_1_per_band(segment_0_rgb)

plt.imshow(segment_0_rgb_scaled)
plt.title(f"Tree {labels[0]}")
plt.show()
```
See also: [preprocessing.scale.scale_0_1_per_band](preprocessing.scale.md)

MatPlotLib cannot show the data correctly as-is, so we need to
- Convert the data from BGR to RGB
- Scale the data to 0-1 per band

> Remember that the library returns the band `order`? This is useful in 
> debugging the data. If we had shown it in BGR, it'll look off!
{style='note'}
