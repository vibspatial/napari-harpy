from __future__ import annotations

from types import SimpleNamespace

import dask.array as da
import numpy as np
import pytest
from spatialdata import SpatialData
from spatialdata.models import Image2DModel
from spatialdata.transformations import Identity
from xarray import DataArray

from napari_harpy.core import histogram as histogram_module
from napari_harpy.core.histogram import HistogramSettings, HistogramTarget, calculate_histogram


def _make_image_sdata(
    values: np.ndarray,
    *,
    channel_names: tuple[str, ...] = ("DAPI",),
    chunks: tuple[int, ...] | int | None = None,
    scale_factors: list[int] | None = None,
) -> SpatialData:
    if chunks is None:
        chunks = values.shape
    image = Image2DModel.parse(
        da.from_array(values.astype(float), chunks=chunks),
        dims=("c", "y", "x"),
        c_coords=list(channel_names),
        transformations={"global": Identity()},
        scale_factors=scale_factors,
    )
    return SpatialData(images={"image": image})


def test_calculate_histogram_defaults_compute_counts_and_edges() -> None:
    sdata = _make_image_sdata(np.array([[[0, 1], [2, 3]]], dtype=float))

    result = calculate_histogram(
        sdata,
        HistogramTarget(coordinate_system="global", image_name="image", channel_name="DAPI"),
        HistogramSettings(bins=3),
    )

    np.testing.assert_array_equal(result.counts, np.array([1, 1, 2]))
    np.testing.assert_allclose(result.bin_edges, np.array([0, 1, 2, 3]))
    assert result.data_range == (0.0, 3.0)
    assert result.resolved_scale == "scale0"
    assert result.percentile_values == {}


def test_calculate_histogram_selects_named_channel() -> None:
    values = np.array(
        [
            [[0, 0], [0, 0]],
            [[10, 11], [12, 13]],
        ],
        dtype=float,
    )
    sdata = _make_image_sdata(values, channel_names=("DAPI", "CD3"))

    result = calculate_histogram(
        sdata,
        HistogramTarget(coordinate_system="global", image_name="image", channel_name="CD3"),
        HistogramSettings(bins=3),
    )

    np.testing.assert_array_equal(result.counts, np.array([1, 1, 2]))
    np.testing.assert_allclose(result.bin_edges, np.array([10, 11, 12, 13]))


def test_calculate_histogram_selects_integer_coordinate_channel_from_string_name() -> None:
    values = np.array(
        [
            [[0, 0], [0, 0]],
            [[10, 11], [12, 13]],
        ],
        dtype=float,
    )
    image = Image2DModel.parse(
        da.from_array(values, chunks=values.shape),
        dims=("c", "y", "x"),
        transformations={"global": Identity()},
    )
    sdata = SpatialData(images={"image": image})

    result = calculate_histogram(
        sdata,
        HistogramTarget(coordinate_system="global", image_name="image", channel_name="1"),
        HistogramSettings(bins=3),
    )

    np.testing.assert_array_equal(result.counts, np.array([1, 1, 2]))
    np.testing.assert_allclose(result.bin_edges, np.array([10, 11, 12, 13]))


def test_histogram_target_requires_explicit_channel_name() -> None:
    with pytest.raises(ValueError, match="explicit channel name"):
        HistogramTarget(coordinate_system="global", image_name="image", channel_name=None)


def test_calculate_histogram_rejects_image_without_channel_axis() -> None:
    image = DataArray(
        da.from_array(np.array([[1, 2], [3, 4]], dtype=float), chunks=(2, 2)),
        dims=("y", "x"),
        attrs={"transform": {"global": Identity()}},
    )
    sdata = SimpleNamespace(images={"image": image})

    with pytest.raises(ValueError, match="does not have a channel axis"):
        calculate_histogram(
            sdata,
            HistogramTarget(coordinate_system="global", image_name="image", channel_name="DAPI"),
            HistogramSettings(),
        )


def test_calculate_histogram_filters_nan_and_zero_values() -> None:
    sdata = _make_image_sdata(np.array([[[0, np.nan, 2], [3, 0, 4]]], dtype=float), chunks=(1, 1, 3))

    result = calculate_histogram(
        sdata,
        HistogramTarget(coordinate_system="global", image_name="image", channel_name="DAPI"),
        HistogramSettings(bins=2, exclude_nan=True, exclude_zeros=True),
    )

    np.testing.assert_array_equal(result.counts, np.array([1, 2]))
    np.testing.assert_allclose(result.bin_edges, np.array([2, 3, 4]))
    assert result.data_range == (2.0, 4.0)


def test_calculate_histogram_value_range_limits_counts_but_not_percentiles() -> None:
    sdata = _make_image_sdata(np.array([[[0, 1, 2], [3, 4, 100]]], dtype=float), chunks=(1, 2, 3))

    result = calculate_histogram(
        sdata,
        HistogramTarget(coordinate_system="global", image_name="image", channel_name="DAPI"),
        HistogramSettings(
            bins=4,
            value_range=(1, 5),
            percentiles=(0, 100),
        ),
    )

    np.testing.assert_array_equal(result.counts, np.array([1, 1, 1, 1]))
    np.testing.assert_allclose(result.bin_edges, np.array([1, 2, 3, 4, 5]))
    assert result.data_range == (1.0, 5.0)
    assert result.percentile_values[0.0] == pytest.approx(0.0)
    assert result.percentile_values[100.0] == pytest.approx(100.0)


def test_calculate_histogram_percentiles_use_tdigest(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}
    original_percentile = histogram_module.da.percentile

    def _capturing_percentile(*args: object, **kwargs: object) -> da.Array:
        captured["internal_method"] = kwargs.get("internal_method")
        return original_percentile(*args, **kwargs)

    monkeypatch.setattr(histogram_module.da, "percentile", _capturing_percentile)
    sdata = _make_image_sdata(np.array([[[1, 2], [3, 4]]], dtype=float))

    result = calculate_histogram(
        sdata,
        HistogramTarget(coordinate_system="global", image_name="image", channel_name="DAPI"),
        HistogramSettings(percentiles=(50,)),
    )

    assert captured["internal_method"] == "tdigest"
    assert result.percentile_values[50.0] == pytest.approx(2.5)


def test_calculate_histogram_selects_requested_multiscale_scale() -> None:
    values = np.arange(16, dtype=float).reshape(1, 4, 4)
    sdata = _make_image_sdata(values, channel_names=("DAPI",), chunks=(1, 4, 4), scale_factors=[2])

    result = calculate_histogram(
        sdata,
        HistogramTarget(coordinate_system="global", image_name="image", channel_name="DAPI"),
        HistogramSettings(bins=4, scale="scale1"),
    )

    assert result.resolved_scale == "scale1"
    assert result.counts.sum() == 4


@pytest.mark.parametrize(
    ("settings_kwargs", "match"),
    [
        ({"bins": 0}, "positive integer"),
        ({"value_range": (2, 2)}, "low < high"),
        ({"value_range": (np.nan, 2)}, "finite"),
        ({"percentiles": (-1,)}, r"\[0, 100\]"),
        ({"percentiles": (101,)}, r"\[0, 100\]"),
    ],
)
def test_histogram_settings_rejects_invalid_values(settings_kwargs: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        HistogramSettings(**settings_kwargs)


def test_histogram_settings_exposes_requested_percentiles() -> None:
    assert HistogramSettings().percentiles == ()
    assert HistogramSettings(percentiles=(1, 99.5)).percentiles == (1.0, 99.5)
    assert HistogramSettings(percentiles=(50, 50)).percentiles == (50.0,)


@pytest.mark.parametrize(
    ("target", "match"),
    [
        (HistogramTarget("global", "missing", "DAPI"), "Image element"),
        (HistogramTarget("missing", "image", "DAPI"), "coordinate system"),
        (HistogramTarget("global", "image", "missing"), "Channel"),
    ],
)
def test_calculate_histogram_rejects_invalid_target(target: HistogramTarget, match: str) -> None:
    sdata = _make_image_sdata(np.array([[[1, 2], [3, 4]]], dtype=float))

    with pytest.raises(ValueError, match=match):
        calculate_histogram(sdata, target, HistogramSettings())
