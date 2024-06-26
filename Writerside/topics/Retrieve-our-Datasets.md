# Retrieve our Datasets

<tldr>A tutorial to retrieve our datasets</tldr>

In this tutorial, we'll learn how to :

- Retrieve FRDC's Datasets
- How to inspect the data
- How to integrate it with PyTorch's DataLoader
- How to visualize the data

## Prerequisites

- New here? [Get Started](Getting-Started.md).
- Setup the Google Cloud Authorization to download the data.

## Retrieve the Data

To retrieve the data, use [FRDCDatasetPreset](load.dataset.md).
This module presets to load explicitly known datasets.

For example:
```python
from frdc.load.preset import FRDCDatasetPreset

ds = FRDCDatasetPreset.chestnut_20201218()
for x, y in ds:
    print(x.shape, y)
```

You should get something like this:
```
(831, 700, 8) Falcataria Moluccana
(540, 536, 8) Ficus Variegata
(457, 660, 8) Bridelia Sp.
...
```

- `x` is a `torch.Tensor`
- `y` is a `str`.

> [What if I can't find a preset dataset?](load.dataset.md#i-can-t-find-a-dataset)
{style='warning'}

## Iterate through the Data

The dataset, when you load it, will be automatically segmented by bounds.
Therefore, if you want to simply loop through the segments and labels,
you can treat the dataset as an iterable.

```python
from frdc.load.preset import FRDCDatasetPreset

ds = FRDCDatasetPreset.chestnut_20201218()
for x, y in ds:
    print(x.shape, y)
```

If you just want the segments or targets separately, use `.ar_segments` and 
`.targets` respectively.

```python
from frdc.load.preset import FRDCDatasetPreset

ds = FRDCDatasetPreset.chestnut_20201218()
for x in ds.ar_segments:
    print(x.shape)

for y in ds.targets:
    print(y)
```

If you want the entire image, use `.ar`.

```python
from frdc.load.preset import FRDCDatasetPreset

ds = FRDCDatasetPreset.chestnut_20201218()
ar = ds.ar
```

Finally, inspect the order of the bands through the `band_order` attribute.

```python
from frdc.load.preset import FRDCDatasetPreset

ds = FRDCDatasetPreset.chestnut_20201218()
ds.band_order
```

```Console
> ['WB', 'WG', 'WR', 'NB', 'NG', 'NR', 'RE', 'NIR']
```

## Using with PyTorch's DataLoader

Every `FRDCDataset` is a `Dataset` object, so you can use it with PyTorch's
`DataLoader`. This allows you to retrieve by batches!

```python
from torch.utils.data import DataLoader
from torchvision.transforms.v2 import CenterCrop, Compose, Resize, ToImage

from frdc.load.preset import FRDCDatasetPreset

ds = FRDCDatasetPreset.chestnut_20201218(
    use_legacy_bounds=True,
    transform=Compose([ToImage(), Resize(100), CenterCrop(100)]),
)
dl = DataLoader(ds, batch_size=4, shuffle=True)

for x, y in dl:
    print(x.shape, y)
```

Which should output

```Console
torch.Size([4, 8, 100, 100]) ('Falcataria Moluccana', ...)
torch.Size([4, 8, 100, 100]) ('Clausena Excavata', ...)
torch.Size([4, 8, 100, 100]) ('Clausena Excavata', ...)
...
```

> **RuntimeError: stack expects each tensor to be equal size**:
> The reason for this error is that `DataLoader` expects equal dimensions 
> (image height and width) for all images. To fix this, you can use 
> `torchvision.transforms.v2.Resize` to resize the images to a fixed size in
> the above example.
{style='warning'}

## Plot the Data (Optional) {collapsible="true"}

We can then use these data to plot out the first tree segment.

```python
import matplotlib.pyplot as plt

from frdc.load.preset import FRDCDatasetPreset
from frdc.preprocess.scale import scale_0_1_per_band

ds = FRDCDatasetPreset.chestnut_20201218()
segment_0_bgr = ds.ar_segments[0]
segment_0_rgb = segment_0_bgr[..., [2, 1, 0]]
segment_0_rgb_scaled = scale_0_1_per_band(segment_0_rgb)

plt.imshow(segment_0_rgb_scaled)
plt.title(f"Tree {ds.targets[0]}")
plt.show()
```
See also: [preprocessing.scale.scale_0_1_per_band](preprocessing.scale.md)

MatPlotLib cannot show the data correctly as-is, so we need to
- Convert the data from BGR to RGB
- Scale the data to 0-1 per band

> Remember that the library returns the band `order`? This is useful in 
> debugging the data. If we had shown it in BGR, it'll look off!
{style='note'}
