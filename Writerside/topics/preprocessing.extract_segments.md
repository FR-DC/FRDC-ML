# preprocessing.extract_segments

<tldr>
Extracts segments from a label classification or bounds and labels.
</tldr>

## Functions

<deflist type="medium">
<def title="extract_segments_from_labels">
<b>Extracts segments from a label classification.</b>
</def>
<def title="extract_segments_from_bounds">
<b>Extracts segments from <code>Rect</code> bounds.</b>
</def>
<def title="remove_small_segments_from_labels">
<b>Removes small segments from a label classification.</b>
</def>
</deflist>

### Extract with Boundaries

A boundary is a `Rect` object that represents the minimum bounding box of a
segment, with x0, y0, x1, y1 coordinates.

It simply slices the original image to the bounding box. The origin is
the top left corner of the image.

<tabs>
<tab title="Cropped = True">
<code-block>
+-----------------+                +-----------+      
| Original        |                | Segmented |      
| Image           |                | Image     |      
+-----+-----+-----+                +-----+-----+      
|  1  |  2  |  3  |                |  2  |  3  |
+-----+-----+-----+                +-----+-----+
|  4  |  5  |  6  |  -----------&gt;  |  5  |  6  |      
+-----+-----+-----+   1, 2, 0, 2   +-----+-----+
|  7  |  8  |  9  |   x0 y0 x1 y1  |  8  |  9  |
+-----+-----+-----+                +-----+-----+
</code-block>
</tab>
<tab title="Cropped = False">
<code-block>
+-----------------+                +-----------------+      
| Original        |                | Segmented       |      
| Image           |                | Image           |      
+-----+-----+-----+                +-----+-----+-----+      
|  1  |  2  |  3  |                |  0  |  2  |  3  |      
+-----+-----+-----+                +-----+-----+-----+      
|  4  |  5  |  6  |  -----------&gt;  |  0  |  5  |  6  |             
+-----+-----+-----+   1, 2, 0, 2   +-----+-----+-----+      
|  7  |  8  |  9  |   x0 y0 x1 y1  |  0  |  8  |  9  |      
+-----+-----+-----+                +-----+-----+-----+      
</code-block>
</tab>
</tabs>

<warning>
The shape of an NDArray is usually H x W x C. Thus, if you're manually slicing
with the bounds, make sure that you're slicing the correct axis.

The correct syntax should be <code>ar[y0:y1,x0:x1]</code>.
</warning>

### Extract with Labels {collapsible="true"}

A label classification is a `np.ndarray` where each pixel is mapped to a
segment. The segments are mapped to a unique integer.
In our project, the 0th label is the background.

For example, a label classification of 3 segments will look like this:

```
+-----------------+  +-----------------+
| Label           |  | Original        |
| Classification  |  | Image           |
+-----+-----+-----+  +-----+-----+-----+
|  1  |  2  |  0  |  |  1  |  2  |  3  |
+-----+-----+-----+  +-----+-----+-----+
|  1  |  2  |  2  |  |  4  |  5  |  6  |
+-----+-----+-----+  +-----+-----+-----+
|  1  |  1  |  0  |  |  7  |  8  |  9  |
+-----+-----+-----+  +-----+-----+-----+
```

The extraction will take the **minimum bounding box** of each segment and
return a list of segments.

For example, the label 1 and 2 extracted images will be

<tabs>
<tab title="Cropped = True">
<code-block>
+-----------+  +-----------+
| Extracted |  | Extracted |
| Segment 1 |  | Segment 2 |
+-----+-----+  +-----+-----+
|  1  |  0  |  |  2  |  0  |
+-----+-----+  +-----+-----+
|  4  |  0  |  |  5  |  6  |
+-----+-----+  +-----+-----+
|  7  |  8  |  
+-----+-----+  
</code-block>
</tab>
<tab title="Cropped = False">
<code-block>
+-----------------+  +-----------------+
| Extracted       |  | Extracted       |
| Segment 1       |  | Segment 2       |
+-----+-----+-----+  +-----+-----+-----+
|  1  |  0  |  0  |  |  0  |  2  |  0  |
+-----+-----+-----+  +-----+-----+-----+
|  4  |  0  |  0  |  |  0  |  5  |  6  |
+-----+-----+-----+  +-----+-----+-----+
|  7  |  8  |  0  |  |  0  |  0  |  0  |
+-----+-----+-----+  +-----+-----+-----+
</code-block>
</tab>
</tabs>

- If **cropped is False**, the segments are padded with 0s to the
  original image size. While this can ensure shape consistency, it can consume
  more memory for large images.
- If **cropped is True**, the segments are cropped to the minimum bounding box.
  This can save memory, but the shape of the segments will be inconsistent.

## Usage

### Extract from Bounds and Labels

Extract segments from bounds and labels.

```python
import numpy as np
from frdc.load.preset import FRDCDatasetPreset
from frdc.preprocess.extract_segments import extract_segments_from_bounds

ds = FRDCDatasetPreset.chestnut_20201218()
ar, order = ds.get_ar_bands()
bounds, labels = ds.get_bounds_and_labels()

segments: list[np.ndarray] = extract_segments_from_bounds(ar, bounds)
```

### Extract from Auto-Segmentation {collapsible="true"}

Extract segments from a label classification.

```python
from skimage.morphology import remove_small_objects, remove_small_holes
import numpy as np

from frdc.load.preset import FRDCDatasetPreset
from frdc.preprocess.morphology import (
    threshold_binary_mask, binary_watershed
)
from frdc.preprocess.scale import scale_0_1_per_band
from frdc.preprocess.extract_segments import (
    extract_segments_from_labels, remove_small_segments_from_labels
)

ds = FRDCDatasetPreset.chestnut_20201218()
ar, order = ds.get_ar_bands()
ar = scale_0_1_per_band(ar)
ar_mask = threshold_binary_mask(ar, -1, 90 / 256)
ar_mask = remove_small_objects(ar_mask, min_size=100, connectivity=2)
ar_mask = remove_small_holes(ar_mask, area_threshold=100, connectivity=2)
ar_labels = binary_watershed(ar_mask)
ar_labels = remove_small_segments_from_labels(ar_labels,
                                              min_height=10, min_width=10)

segments: list[np.ndarray] = extract_segments_from_labels(ar, ar_labels)
```

> The `remove_small_objects` and `remove_small_holes` are used to remove
> small noise from the binary mask. This is recommended and used in the
> original paper.
> {style='note'}

## API

<deflist>
<def title="extract_segments_from_labels(ar, ar_labels, cropped)">
<b>Extracts segments from a label classification.</b><br/>
<code>ar_labels</code> is a label classification as a <code>np.ndarray</code>
</def>
<def title="extract_segments_from_bounds(ar, bounds, cropped)">
<b>Extracts segments from <code>Rect</code> bounds.</b><br/>
<code>bounds</code> is a list of <code>Rect</code> bounds.
</def>
<def title="remove_small_segments_from_labels(ar_labels, min_height, min_width)">
<b>Removes small segments from a label classification.</b><br/>

</def>

</deflist>