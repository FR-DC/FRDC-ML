# load.dataset

> You need to Set-Up [Google Cloud](Getting-Started.md#gcloud) with the
> appropriate permissions to use this library.
> {style='warning'}

<tldr>
Load dataset objects from our GCS bucket.
</tldr>

## Usage

Firstly, to load a dataset instance, you need to
initiliaze a `FRDCDataset` object, providing the site, date, and
version.

For example, to load our Chestnut Nature Park dataset. 

```python
from frdc.load import FRDCDataset

ds = FRDCDataset(site='chestnut_nature_park',
                 date='20201218',
                 version=None)
```

Then, we can use the `ds` object to load objects of the dataset:

```python
ar, order = ds.get_ar_bands()
d = ds.get_ar_bands_as_dict()
bounds, labels = ds.get_bounds_and_labels()
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

## Filters

You can also selectively get the channels for both `get_ar_bands()` and
`get_ar_bands_as_dict()` by providing a list of strings to the `bands`
argument.

For example, to get the Wideband RGB bands, you can do:

```python
ar, order = ds.get_ar_bands(bands=['WR', 'WG', 'WB'])
d = ds.get_ar_bands_as_dict(bands=['WR', 'WG', 'WB'])
```

This will also alter the channel order to the order of the bands provided.

See [load.gcs](load.gcs.md#configuration) for configuration options.
