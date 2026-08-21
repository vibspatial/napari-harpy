from __future__ import annotations

import numpy as np
import pytest

from napari_harpy.core.multi_scale_cache_points_zarr.build_plan import _LevelBuildPlan, _LevelKind
from napari_harpy.core.multi_scale_cache_points_zarr.hashing import (
    BUCKET_HASH_METHOD,
    _bucket_count_for_level,
    _tile_bucket_ids,
)


def _level(point_count: int) -> _LevelBuildPlan:
    return _LevelBuildPlan(0, _LevelKind.EXACT, 512, 1, 1, None, point_count)


def test_splitmix64_bucket_mapping_has_independent_fixed_vectors() -> None:
    tile_x = np.array([0, 1, 0, 1, 2**32 - 1], dtype=np.uint32)
    tile_y = np.array([0, 0, 1, 1, 2**32 - 1], dtype=np.uint32)

    assert BUCKET_HASH_METHOD == "harpy-zarr-tile-splitmix64-v1"
    assert _tile_bucket_ids(tile_x, tile_y, bucket_count=69).tolist() == [16, 26, 43, 1, 2]


def test_bucket_count_uses_the_provisional_two_million_point_policy() -> None:
    assert _bucket_count_for_level(_level(136_578_750)) == 69


def test_bucket_count_rejects_a_count_that_cannot_fit_uint32() -> None:
    with pytest.raises(ValueError, match="bucket count"):
        _bucket_count_for_level(_level(2**63 - 1))


@pytest.mark.parametrize(
    ("tile_x", "tile_y", "bucket_count", "message"),
    [
        ([0], np.array([0], dtype=np.uint32), 1, "NumPy"),
        (np.array([0], dtype=np.int64), np.array([0], dtype=np.uint32), 1, "uint32"),
        (np.array([[0]], dtype=np.uint32), np.array([0], dtype=np.uint32), 1, "one-dimensional"),
        (np.arange(4, dtype=np.uint32)[::2], np.array([0, 1], dtype=np.uint32), 1, "C-contiguous"),
        (np.array([0], dtype=np.uint32), np.array([0, 1], dtype=np.uint32), 1, "identical shapes"),
        (np.array([0], dtype=np.uint32), np.array([0], dtype=np.uint32), 0, "bucket_count"),
        (np.array([0], dtype=np.uint32), np.array([0], dtype=np.uint32), 2**32 + 1, "bucket_count"),
    ],
)
def test_bucket_mapping_rejects_invalid_inputs(
    tile_x: object,
    tile_y: object,
    bucket_count: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _tile_bucket_ids(tile_x, tile_y, bucket_count=bucket_count)  # type: ignore[arg-type]
