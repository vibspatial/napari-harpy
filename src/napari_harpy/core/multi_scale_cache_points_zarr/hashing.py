from __future__ import annotations

import numpy as np

from napari_harpy.core.multi_scale_cache_points_zarr.build_plan import _LevelBuildPlan

BUCKET_HASH_METHOD = "harpy-zarr-tile-splitmix64-v1"
TARGET_POINTS_PER_BUCKET = 2_000_000

_MAX_BUCKET_COUNT = 2**32
_UINT64_27 = np.uint64(27)
_UINT64_30 = np.uint64(30)
_UINT64_31 = np.uint64(31)
_UINT64_32 = np.uint64(32)
_SPLITMIX64_INCREMENT = np.uint64(0x9E3779B97F4A7C15)
_SPLITMIX64_MULTIPLIER_1 = np.uint64(0xBF58476D1CE4E5B9)
_SPLITMIX64_MULTIPLIER_2 = np.uint64(0x94D049BB133111EB)


def _bucket_count_for_level(level: _LevelBuildPlan) -> int:
    """Return the provisional deterministic bucket count for one level."""
    if not isinstance(level, _LevelBuildPlan):
        raise ValueError("`level` must be a _LevelBuildPlan.")
    bucket_count = max(
        1,
        (level.point_count_upper_bound + TARGET_POINTS_PER_BUCKET - 1) // TARGET_POINTS_PER_BUCKET,
    )
    if bucket_count > _MAX_BUCKET_COUNT:
        raise ValueError("The planned bucket count exceeds the supported uint32 space.")
    return bucket_count


def _tile_bucket_ids(
    tile_x: np.ndarray,
    tile_y: np.ndarray,
    *,
    bucket_count: int,
) -> np.ndarray:
    """Map exact uint32 tile arrays through the versioned SplitMix64 policy."""
    _require_tile_coordinate_array(tile_x, "tile_x")
    _require_tile_coordinate_array(tile_y, "tile_y")
    if tile_x.shape != tile_y.shape:
        raise ValueError("`tile_x` and `tile_y` must have identical shapes.")
    _require_positive_bucket_count(bucket_count)
    tile_key = (tile_y.astype(np.uint64) << _UINT64_32) | tile_x.astype(np.uint64)
    return _splitmix64(tile_key) % np.uint64(bucket_count)


def _splitmix64(values: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):
        mixed = values + _SPLITMIX64_INCREMENT
        mixed = (mixed ^ (mixed >> _UINT64_30)) * _SPLITMIX64_MULTIPLIER_1
        mixed = (mixed ^ (mixed >> _UINT64_27)) * _SPLITMIX64_MULTIPLIER_2
    return mixed ^ (mixed >> _UINT64_31)


def _require_tile_coordinate_array(value: object, name: str) -> None:
    if not isinstance(value, np.ndarray):
        raise ValueError(f"`{name}` must be a NumPy array.")
    if value.ndim != 1 or value.dtype != np.dtype(np.uint32) or not value.flags.c_contiguous:
        raise ValueError(f"`{name}` must be a one-dimensional C-contiguous uint32 array.")


def _require_positive_bucket_count(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= _MAX_BUCKET_COUNT:
        raise ValueError(f"`bucket_count` must be an integer in the range [1, {_MAX_BUCKET_COUNT}].")
    return value
