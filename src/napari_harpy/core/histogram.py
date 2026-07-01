from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import dask
import dask.array as da
import numpy as np
from spatialdata.transformations import get_transformation
from xarray import DataArray, DataTree

if TYPE_CHECKING:
    from spatialdata import SpatialData


@dataclass(frozen=True)
class HistogramTarget:
    coordinate_system: str
    image_name: str
    channel_name: str

    def __post_init__(self) -> None:
        if not isinstance(self.channel_name, str) or not self.channel_name.strip():
            raise ValueError("Histogram target requires an explicit channel name.")


@dataclass(frozen=True)
class HistogramSettings:
    bins: int = 256
    value_range: tuple[float, float] | None = None
    density: bool = False
    exclude_nan: bool = True
    exclude_zeros: bool = False
    log_y: bool = False
    scale: str | None = None
    percentiles: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.bins, bool) or not isinstance(self.bins, int) or self.bins <= 0:
            raise ValueError("Histogram settings require `bins` to be a positive integer.")

        if self.value_range is not None:
            low, high = self.value_range
            if not np.isfinite(low) or not np.isfinite(high):
                raise ValueError("Histogram settings require `value_range` bounds to be finite.")
            if low >= high:
                raise ValueError("Histogram settings require `value_range` to satisfy low < high.")

        percentiles: list[float] = []
        for percentile in self.percentiles:
            percentile_value = float(percentile)
            if not np.isfinite(percentile_value) or percentile_value < 0 or percentile_value > 100:
                raise ValueError("Histogram percentile values must be finite values in [0, 100].")
            if percentile_value not in percentiles:
                percentiles.append(percentile_value)
        object.__setattr__(self, "percentiles", tuple(percentiles))


@dataclass(frozen=True)
class HistogramResult:
    target: HistogramTarget
    settings: HistogramSettings
    counts: np.ndarray
    bin_edges: np.ndarray
    data_range: tuple[float, float]
    percentile_values: Mapping[float, float]
    resolved_scale: str | None


def calculate_histogram(
    sdata: SpatialData,
    target: HistogramTarget,
    settings: HistogramSettings,
) -> HistogramResult:
    """Calculate histogram data for one explicit SpatialData image target.

    `settings.value_range` is passed to `dask.array.histogram(..., range=...)`
    and therefore controls which values contribute to histogram counts and bin
    edges. Percentiles are computed after NaN/zero filtering but before applying
    `settings.value_range`, matching Harpy's histogram semantics.
    """
    if target.image_name not in sdata.images:
        raise ValueError(f"Image element `{target.image_name}` is not available in the selected SpatialData object.")

    image_element = sdata.images[target.image_name]
    coordinate_systems = tuple(sorted(get_transformation(image_element, get_all=True).keys()))
    if target.coordinate_system not in coordinate_systems:
        available = ", ".join(f"`{name}`" for name in coordinate_systems) or "none"
        raise ValueError(
            f"Image element `{target.image_name}` is not available in coordinate system "
            f"`{target.coordinate_system}`. Available coordinate systems: {available}."
        )

    array, resolved_scale = _resolve_array(image_element, settings.scale, target.image_name)
    array = _select_channel_values(array, target.channel_name, target.image_name)
    array = _filter_image_values(
        array.ravel(),
        exclude_nan=settings.exclude_nan,
        exclude_zeros=settings.exclude_zeros,
    )

    percentile_values: dict[float, float] = {}
    if settings.percentiles:
        values = da.percentile(array, q=list(settings.percentiles), internal_method="tdigest").compute()
        values = np.atleast_1d(values)
        percentile_values = {
            percentile: float(value) for percentile, value in zip(settings.percentiles, values, strict=True)
        }
    data_range = _resolve_histogram_range(array, settings.value_range)
    counts, bin_edges = da.histogram(
        array,
        bins=settings.bins,
        range=data_range,
        density=settings.density,
    )
    counts, bin_edges = dask.compute(counts, bin_edges)

    return HistogramResult(
        target=target,
        settings=settings,
        counts=np.asarray(counts),
        bin_edges=np.asarray(bin_edges),
        data_range=data_range,
        percentile_values=percentile_values,
        resolved_scale=resolved_scale,
)


def _resolve_array(
    image_element: DataArray | DataTree,
    requested_scale: str | None,
    image_name: str,
) -> tuple[DataArray, str]:
    if isinstance(image_element, DataArray):
        if requested_scale not in {None, "scale0"}:
            raise ValueError(f"Scale `{requested_scale}` is not available for single-scale image element `{image_name}`.")
        return image_element, "scale0"

    resolved_scale = requested_scale or "scale0"
    if resolved_scale not in image_element:
        available = ", ".join(f"`{scale}`" for scale in image_element.keys()) or "none"
        raise ValueError(
            f"Scale `{resolved_scale}` is not available for image element `{image_name}`. Available scales: {available}."
        )

    scale_values = list(image_element[resolved_scale].values())
    if len(scale_values) != 1 or not isinstance(scale_values[0], DataArray):
        raise ValueError(f"Scale `{resolved_scale}` in image element `{image_name}` is not a supported image scale.")

    return scale_values[0], resolved_scale


def _select_channel_values(array: DataArray, channel_name: str, image_name: str) -> da.Array:
    if "c" not in array.dims:
        raise ValueError(f"Image element `{image_name}` does not have a channel axis.")

    channel_values = list(array.c.data)
    channel_names = [str(channel_value) for channel_value in channel_values]
    duplicate_names = sorted({name for name in channel_names if channel_names.count(name) > 1})
    if duplicate_names:
        duplicates = ", ".join(f"`{name}`" for name in duplicate_names)
        raise ValueError(
            f"Image element `{image_name}` exposes duplicate channel names ({duplicates}), "
            "which napari-harpy does not support."
        )
    if channel_name not in channel_names:
        available = ", ".join(f"`{name}`" for name in channel_names) or "none"
        raise ValueError(
            f"Channel `{channel_name}` is not available in image element `{image_name}`. Available channels: {available}."
        )

    channel_value = channel_values[channel_names.index(channel_name)]
    return da.asarray(array.sel(c=channel_value).data)


def _filter_image_values(array: da.Array, *, exclude_nan: bool, exclude_zeros: bool) -> da.Array:
    if not exclude_nan and not exclude_zeros:
        return array

    mask = da.ones(array.shape, dtype=bool, chunks=array.chunks)
    if exclude_nan:
        mask &= ~da.isnan(array)
    if exclude_zeros:
        mask &= array != 0

    array = da.compress(mask, array)
    if _has_unknown_chunks(array):
        array = array.compute_chunk_sizes()
    return array


def _has_unknown_chunks(array: da.Array) -> bool:
    return any(np.isnan(chunk_size) for chunks in array.chunks for chunk_size in chunks)


def _resolve_histogram_range(array: da.Array, value_range: tuple[float, float] | None) -> tuple[float, float]:
    if value_range is not None:
        return (float(value_range[0]), float(value_range[1]))

    low, high = dask.compute(da.nanmin(array), da.nanmax(array))
    data_range = (float(low), float(high))
    if not np.isfinite(data_range[0]) or not np.isfinite(data_range[1]):
        raise ValueError("Histogram calculation requires at least one finite image value.")
    if data_range[0] == data_range[1]:
        return (data_range[0], data_range[1] + 1.0)
    return data_range
