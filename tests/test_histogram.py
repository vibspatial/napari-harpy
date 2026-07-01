from __future__ import annotations

import dask
import dask.array as da
import numpy as np
import pytest
from spatialdata import SpatialData

from napari_harpy.core.histogram import HistogramSettings, HistogramTarget, calculate_histogram


def test_calculate_histogram_from_sdata_blobs_matches_dask_histogram(sdata_blobs: SpatialData) -> None:
    settings = HistogramSettings(bins=8)
    target = HistogramTarget(coordinate_system="global", image_name="blobs_image", channel_name="0")

    result = calculate_histogram(sdata_blobs, target, settings)

    array = da.asarray(sdata_blobs.images["blobs_image"].sel(c=0).data).ravel()
    expected_counts, expected_bin_edges = da.histogram(array, bins=settings.bins, range=result.data_range)
    expected_counts, expected_bin_edges = dask.compute(expected_counts, expected_bin_edges)

    assert result.target == target
    assert result.settings == settings
    assert result.resolved_scale == "scale0"
    assert result.data_range == (0.0, 1.0)
    np.testing.assert_array_equal(result.counts, expected_counts)
    np.testing.assert_allclose(result.bin_edges, expected_bin_edges)


def test_calculate_histogram_uses_value_range_for_counts_not_percentiles(sdata_blobs: SpatialData) -> None:
    settings = HistogramSettings(bins=4, value_range=(0.25, 0.75), percentiles=(0, 100))

    result = calculate_histogram(
        sdata_blobs,
        HistogramTarget(coordinate_system="global", image_name="blobs_image", channel_name="0"),
        settings,
    )

    array = da.asarray(sdata_blobs.images["blobs_image"].sel(c=0).data).ravel()
    expected_counts, expected_bin_edges = da.histogram(array, bins=settings.bins, range=settings.value_range)
    expected_counts, expected_bin_edges = dask.compute(expected_counts, expected_bin_edges)

    assert result.data_range == settings.value_range
    np.testing.assert_array_equal(result.counts, expected_counts)
    np.testing.assert_allclose(result.bin_edges, expected_bin_edges)
    assert result.percentile_values[0.0] == pytest.approx(0.0)
    assert result.percentile_values[100.0] == pytest.approx(1.0)


def test_calculate_histogram_uses_requested_multiscale_scale(sdata_blobs: SpatialData) -> None:
    result = calculate_histogram(
        sdata_blobs,
        HistogramTarget(coordinate_system="global", image_name="blobs_multiscale_image", channel_name="0"),
        HistogramSettings(bins=4, scale="scale1"),
    )

    array = next(iter(sdata_blobs.images["blobs_multiscale_image"]["scale1"].values()))

    assert result.resolved_scale == "scale1"
    assert int(result.counts.sum()) == array.sizes["y"] * array.sizes["x"]


def test_histogram_dataclasses_validate_basic_inputs() -> None:
    assert HistogramSettings(percentiles=(50, 50)).percentiles == (50.0,)

    with pytest.raises(ValueError, match="positive integer"):
        HistogramSettings(bins=0)

    with pytest.raises(ValueError, match="low < high"):
        HistogramSettings(value_range=(2, 2))

    with pytest.raises(ValueError, match="explicit channel name"):
        HistogramTarget(coordinate_system="global", image_name="blobs_image", channel_name=None)
