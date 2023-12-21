import numpy as np

from frdc.preprocess.extract_segments import (
    remove_small_segments_from_labels,
    extract_segments_from_bounds,
    extract_segments_from_labels,
    extract_segments_from_polybounds,
)
from utils import get_labels


def test_remove_small_segments_from_labels():
    """We'll test that it correctly removes the correct segments.

    The test case:
        1 1 2 2 2
        1 1 2 2 2
        3 3 4 4 4
        3 3 4 4 4
        3 3 4 4 4

    For example, if we removed anything smaller than width 3 or height 3,
        then we expect:
        0 0 0 0 0
        0 0 0 0 0
        0 0 4 4 4
        0 0 4 4 4
        0 0 4 4 4

    Then the unique labels should be {0, 4}

    """
    ar = np.zeros((5, 5), dtype=np.uint8)
    ar[0:2, 0:2] = 1
    ar[0:2, 2:5] = 2
    ar[2:5, 0:2] = 3
    ar[2:5, 2:5] = 4

    def test_unique_labels(
        expected_labels: set, min_height: int = 2, min_width: int = 2
    ):
        """Tests the unique labels are as expected."""
        assert (
            set(
                np.unique(
                    remove_small_segments_from_labels(
                        ar, min_height=min_height, min_width=min_width
                    )
                )
            )
            == expected_labels
        )

    # 0 represents "removed" labels are relabelled to the background 0.
    test_unique_labels({1, 2, 3, 4}, min_height=2, min_width=2)
    test_unique_labels({0, 3, 4}, min_height=3, min_width=2)
    test_unique_labels({0, 2, 4}, min_height=2, min_width=3)
    test_unique_labels({0, 4}, min_height=3, min_width=3)
    test_unique_labels({0}, min_height=4, min_width=4)


"""
We check if the cropping works:
If we crop, then >= 1 shapes in the output must be different from the input.
Else,       then ALL  shapes in the output must be the same as the input.

For example, given that our input image 10 x 10:
- If we crop, then we expect at least one of the output images to be < 10 x 10,
    for example 7 x 5
- If we don't crop, then we expect all the output images to be 10 x 10

Thus, our test is:
- If we crop,       any(segment.shape != ar.shape for segment in segments)
- If we don't crop, all(segment.shape == ar.shape for segment in segments)
"""


def test_extract_segments_from_bounds_cropped(ds, ar):
    bounds, labels = ds.get_bounds_and_labels()
    segments = extract_segments_from_bounds(ar, bounds, cropped=True)
    assert any(segment.shape != ar.shape for segment in segments)


def test_extract_segments_from_bounds_no_crop(ds, ar):
    bounds, labels = ds.get_bounds_and_labels()
    segments = extract_segments_from_bounds(ar, bounds, cropped=False)
    assert all(segment.shape == ar.shape for segment in segments)


def test_extract_segments_from_labels_cropped(ar, order):
    ar_labels = get_labels(ar, order)
    segments = extract_segments_from_labels(ar, ar_labels, cropped=True)
    assert any(segment.shape != ar.shape for segment in segments)


def test_extract_segments_from_labels_no_crop(ar, order):
    ar_labels = get_labels(ar, order)
    segments = extract_segments_from_labels(ar, ar_labels, cropped=False)
    assert all(segment.shape == ar.shape for segment in segments)


def test_extract_segments_from_labels():
    """Test our segment extraction function.

    To do this, we have a source array and a label array.
    Our function will loop through each unique label in the label array,
    then mask the source array with that label.

    The following thus should return
    ar_segments = [
        [[[10],[nan]],[[nan],[nan]]],
        [[[nan],[20]],[[nan],[nan]]],
        [[[nan],[nan]],[[30],[nan]]],
        [[[nan],[nan]],[[nan],[40]]],
    ]
    """
    ar_source = np.array([[[10], [20]], [[30], [40]]])
    ar_label = np.array([[0, 1], [2, 3]])
    ar_segments = extract_segments_from_labels(
        ar_source, ar_label, cropped=False
    )
    assert np.isclose(
        ar_segments[0],
        np.array([[[10], [np.nan]], [[np.nan], [np.nan]]]),
        equal_nan=True,
    ).all()
    assert np.isclose(
        ar_segments[1],
        np.array([[[np.nan], [20]], [[np.nan], [np.nan]]]),
        equal_nan=True,
    ).all()
    assert np.isclose(
        ar_segments[2],
        np.array([[[np.nan], [np.nan]], [[30], [np.nan]]]),
        equal_nan=True,
    ).all()
    assert np.isclose(
        ar_segments[3],
        np.array([[[np.nan], [np.nan]], [[np.nan], [40]]]),
        equal_nan=True,
    ).all()


def test_extract_segments_from_polybounds():
    """Test polybound segment extraction."""

    # 0   1   2   3
    # 4   5   6   7
    # 8   9   10  11
    # 12  13  14  15
    ar = np.arange(16).reshape(4, 4, 1)

    # 1   1   1   0
    # 1   1   1   0
    # 1   1   0   0
    # 0   0   0   0
    poly = [[(0, 0), (0, 3), (3, 0), (0, 0)]]

    # 0   1   2
    # 4   5   6
    # 8   9   nan
    s = extract_segments_from_polybounds(
        ar,
        poly,
        cropped=True,
        polycropped=True,
    )[0]
    assert s.shape == (3, 3, 1)
    assert np.isnan(s[-1, -1, 0])

    # 0   1   2   nan
    # 4   5   6   nan
    # 8   9   nan nan
    # nan nan nan nan
    s = extract_segments_from_polybounds(
        ar,
        poly,
        cropped=False,
    )[0]
    assert s.shape == (4, 4, 1)
    assert np.isnan(s).any()

    # 0   1   2
    # 4   5   6
    # 8   9   10
    s = extract_segments_from_polybounds(
        ar,
        poly,
        cropped=True,
        polycropped=False,
    )[0]
    assert s.shape == (3, 3, 1)
    assert not np.isnan(s).any()
