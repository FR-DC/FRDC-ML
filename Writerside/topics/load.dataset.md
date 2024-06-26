# load.dataset

> You need to Set-Up [Google Cloud](Getting-Started.md#gcloud) with the
> appropriate permissions to use this library.
> {style='warning'}

<tldr>
Load dataset objects from our GCS bucket.
</tldr>

## Usage

Firstly, to load a dataset instance, you need to initiliaze a `FRDCDataset`
object, providing the site, date, and version.
 
We recommend using the `FRDCDatasetPreset` module to load explicitly known
datasets.

```python
from frdc.load.preset import FRDCDatasetPreset

ds = FRDCDatasetPreset.chestnut_20201218()
```

Then, we can use the `ds` object to load objects of the dataset:

```python
ar, order = ds._get_ar_bands()
d = ds._get_ar_bands_as_dict()
bounds, labels = ds._get_legacy_bounds_and_labels()
```

- `ar` is a stacked NDArray of the hyperspectral bands of shape (H x W x C)
- `order` is a list of strings, containing the names of the bands, ordered
  according to the channels of `ar`
- `d` is a dictionary of the hyperspectral bands of shape (H x W), keyed by
  the band names
- `bounds` is a list of bounding boxes, in the format of `Rect`, a
  `namedtuple` of x0, y0, x1, y1
- `labels` is a list of strings, containing the labels of the bounding boxes,
  ordered according to `bounds`

> `get_ar_bands()` and `get_ar_bands_as_dict()` retrieves the same data, but
> `get_ar_bands()` is a convenience function that stacks the bands into a single
> NDArray, and returns the channel order as well.
{style='note'}


### I can't find a dataset!

Some datasets, especially new ones may be unregistered and you must specify
the **exact** site / date / version of it.

```python
from frdc.load.dataset import FRDCDataset

ds = FRDCDataset(site="mysite", date="mydate", version="myversion")
```

> `version` can be `None` if there isn't one.
{style='note'}

See below for examples on how to format this.

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

