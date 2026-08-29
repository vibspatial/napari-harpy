"""Application-facing contracts for one cache-backed tiled-points layer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
from matplotlib.colors import to_rgba

from napari_harpy.core.class_palette import default_categorical_colors
from napari_harpy.core.multi_scale_cache_points_zarr.reader import _CacheDatasetInfo
from napari_harpy.viewer.tiled_points.contracts import TiledPointsDatasetReference
from napari_harpy.viewer.tiled_points.runtime.cache_session import _CacheSessionSettings

DEFAULT_MAX_CPU_TILE_BYTES = 1_073_741_824
DEFAULT_MAX_GPU_TILE_BYTES = 536_870_912


@dataclass(frozen=True)
class TiledPointsApplicationSettings:
    """Define napari-harpy's injectable points-cache resource policy.

    ``max_gpu_tile_bytes`` retains its compatibility name during the
    constant-resource renderer transition. It bounds one complete packed
    candidate vertex payload, not a residency cache of per-tile GPU objects.
    """

    max_bucket_lookup_bytes: int | None = None
    max_selected_value_index_bytes: int | None = None
    max_cpu_tile_bytes: int = DEFAULT_MAX_CPU_TILE_BYTES
    max_gpu_tile_bytes: int = DEFAULT_MAX_GPU_TILE_BYTES

    def __post_init__(self) -> None:
        # Reuse the session contract for the three worker-side limits.
        _CacheSessionSettings(
            max_bucket_lookup_bytes=self.max_bucket_lookup_bytes,
            max_selected_value_index_bytes=self.max_selected_value_index_bytes,
            max_cpu_tile_bytes=self.max_cpu_tile_bytes,
        )
        if (
            not isinstance(self.max_gpu_tile_bytes, int)
            or isinstance(self.max_gpu_tile_bytes, bool)
            or self.max_gpu_tile_bytes <= 0
        ):
            raise ValueError("`max_gpu_tile_bytes` must be a positive integer.")

    @property
    def cache_session_settings(self) -> _CacheSessionSettings:
        """Return the worker-owned cache-session settings."""
        return _CacheSessionSettings(
            max_bucket_lookup_bytes=self.max_bucket_lookup_bytes,
            max_selected_value_index_bytes=self.max_selected_value_index_bytes,
            max_cpu_tile_bytes=self.max_cpu_tile_bytes,
        )


@dataclass(frozen=True)
class TiledPointsCacheDescriptor:
    """Describe a completed nested points cache without retaining cache IO."""

    cache_root: Path
    dataset_info: _CacheDatasetInfo

    def __post_init__(self) -> None:
        if not isinstance(self.cache_root, Path):
            raise ValueError("`cache_root` must be pathlib.Path.")

    @property
    def value_names(self) -> tuple[str, ...]:
        """Return canonical value names in implicit uint32 ID order."""
        return self.dataset_info.value_names

    def requested_value_ids(self, values: tuple[str, ...] | str) -> tuple[int, ...] | None:
        """Map canonical value labels to stable sorted IDs; ``all`` maps to None."""
        if values == "all":
            return None
        if not isinstance(values, tuple) or not values:
            raise ValueError("Select at least one canonical value or use 'all'.")
        if len(set(values)) != len(values):
            raise ValueError("Requested value names must be unique.")
        value_id_by_name = {name: value_id for value_id, name in enumerate(self.value_names)}
        unknown = tuple(value for value in values if value not in value_id_by_name)
        if unknown:
            raise ValueError(f"Requested values are not present in the cache vocabulary: {unknown!r}.")
        return tuple(sorted(value_id_by_name[value] for value in values))

    @property
    def dataset_reference(self) -> TiledPointsDatasetReference:
        """Return the logical napari-layer data reference."""
        info = self.dataset_info
        return TiledPointsDatasetReference(
            cache_generation_id=info.cache_generation_id,
            points_name=info.points_name,
            value_column=info.value_column,
            value_count=len(info.value_names),
            x_origin=info.x_origin,
            y_origin=info.y_origin,
            x_min=info.x_min,
            x_max=info.x_max,
            y_min=info.y_min,
            y_max=info.y_max,
        )


def canonical_value_palette(value_count: int) -> npt.NDArray[np.uint8]:
    """Return a value-ID-aligned RGBA palette repeating Harpy's 102 colours."""
    if not isinstance(value_count, int) or isinstance(value_count, bool) or value_count <= 0:
        raise ValueError("`value_count` must be a positive integer.")
    colors = default_categorical_colors(value_count)
    rgba = np.asarray([to_rgba(color, alpha=1.0) for color in colors], dtype=np.float64)
    return np.ascontiguousarray(np.rint(rgba * 255.0), dtype=np.uint8)
