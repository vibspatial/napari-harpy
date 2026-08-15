from __future__ import annotations

from dataclasses import replace

import pytest

from napari_harpy.core.multi_scale_cache_points_zarr.models import _bucket_path, _TileDescriptor
from napari_harpy.core.multi_scale_cache_points_zarr.storage.models import (
    _BucketPlan,
    _BucketWriteResult,
    _LevelWriteResult,
    _PlannedTile,
    _ZarrWriteSettings,
)


def _settings(**overrides: object) -> _ZarrWriteSettings:
    values: dict[str, object] = {
        "point_chunk_rows": 4_096,
        "point_shard_rows": 131_072,
        "range_chunk_rows": 8_192,
        "range_shard_rows": 131_072,
        "codec_id": "zstd-v1",
    }
    values.update(overrides)
    return _ZarrWriteSettings(**values)  # type: ignore[arg-type]


def _tile(**overrides: object) -> _TileDescriptor:
    values: dict[str, object] = {
        "level": 0,
        "bucket_id": 0,
        "bucket_tile_index": 0,
        "tile_x": 0,
        "tile_y": 0,
        "n_points": 3,
    }
    values.update(overrides)
    return _TileDescriptor(**values)  # type: ignore[arg-type]


def test_bucket_path_is_canonical_and_derived_from_identity() -> None:
    assert _bucket_path(level=0, bucket_id=3) == "levels/level_0/bucket-003.zarr"
    assert _bucket_path(level=2, bucket_id=999) == "levels/level_2/bucket-999.zarr"
    assert _bucket_path(level=2, bucket_id=1_000) == "levels/level_2/bucket-1000.zarr"
    assert _tile(level=2, bucket_id=3).bucket_path == "levels/level_2/bucket-003.zarr"


@pytest.mark.parametrize(
    ("level", "bucket_id", "message"),
    [
        (-1, 0, "level"),
        (2**15, 0, "level"),
        (0, True, "bucket_id"),
        (0, 2**32, "bucket_id"),
    ],
)
def test_bucket_path_rejects_invalid_identity(
    level: object,
    bucket_id: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _bucket_path(level=level, bucket_id=bucket_id)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("level", 2**15),
        ("bucket_id", True),
        ("bucket_tile_index", -1),
        ("tile_x", 2**32),
        ("tile_y", -1),
        ("n_points", 0),
        ("n_points", 2**63),
    ],
)
def test_tile_descriptor_enforces_serialized_ranges(field: str, value: object) -> None:
    with pytest.raises(ValueError, match=field):
        _tile(**{field: value})


def test_write_settings_require_positive_aligned_rows_and_codec() -> None:
    assert _settings().point_shard_rows == 32 * _settings().point_chunk_rows
    assert _settings().range_shard_rows == 16 * _settings().range_chunk_rows
    with pytest.raises(ValueError, match="point_chunk_rows"):
        _settings(point_chunk_rows=0)
    with pytest.raises(ValueError, match="multiple"):
        _settings(point_shard_rows=5_000)
    with pytest.raises(ValueError, match="range_chunk_rows"):
        _settings(range_chunk_rows=True)
    with pytest.raises(ValueError, match="multiple"):
        _settings(range_shard_rows=10_000)
    with pytest.raises(ValueError, match="codec_id"):
        _settings(codec_id="")


def test_bucket_plan_exposes_exact_read_only_prefix_sums() -> None:
    plan = _BucketPlan(
        level=0,
        bucket_id=0,
        tiles=(_PlannedTile(0, 0, 3), _PlannedTile(1, 0, 2), _PlannedTile(0, 1, 5)),
        settings=_settings(),
    )

    assert plan.tile_count == 3
    assert plan.bucket_path == "levels/level_0/bucket-000.zarr"
    assert plan.point_count == 10
    assert plan.tile_offset.tolist() == [0, 3, 5, 10]
    assert not plan.tile_offset.flags.writeable


def test_bucket_plan_rejects_empty_duplicate_unordered_or_overflowing_tiles() -> None:
    values = {
        "level": 0,
        "bucket_id": 0,
        "settings": _settings(),
    }
    with pytest.raises(ValueError, match="at least one"):
        _BucketPlan(tiles=(), **values)
    with pytest.raises(ValueError, match="unique"):
        _BucketPlan(tiles=(_PlannedTile(0, 0, 1), _PlannedTile(0, 0, 1)), **values)
    with pytest.raises(ValueError, match="ordered"):
        _BucketPlan(tiles=(_PlannedTile(0, 1, 1), _PlannedTile(1, 0, 1)), **values)
    with pytest.raises(ValueError, match="point count"):
        _BucketPlan(tiles=(_PlannedTile(0, 0, 2**63 - 1), _PlannedTile(1, 0, 1)), **values)


def test_bucket_and_level_results_reconcile_and_order_logical_tiles() -> None:
    bucket_0_tiles = (_tile(n_points=3), _tile(bucket_tile_index=1, tile_x=0, tile_y=1, n_points=2))
    bucket_0 = _BucketWriteResult(bucket_0_tiles, 5, 3)
    bucket_1_tile = _tile(
        bucket_id=1,
        tile_x=1,
        tile_y=0,
        n_points=4,
    )
    bucket_1 = _BucketWriteResult((bucket_1_tile,), 4, 2)

    result = _LevelWriteResult((bucket_0, bucket_1))

    assert (bucket_1.level, bucket_1.bucket_id, bucket_1.bucket_path) == (
        0,
        1,
        "levels/level_0/bucket-001.zarr",
    )
    assert result.level == 0
    assert [(tile.tile_x, tile.tile_y) for tile in result.tile_descriptors] == [(0, 0), (1, 0), (0, 1)]
    assert (result.bucket_count, result.tile_count, result.point_count, result.range_count) == (2, 3, 9, 5)


def test_bucket_result_rejects_wrong_ownership_indexes_counts_and_ranges() -> None:
    tile = _tile()
    other_bucket_tile = _tile(
        bucket_id=1,
        bucket_tile_index=1,
        tile_x=1,
    )
    with pytest.raises(ValueError, match="same bucket identity"):
        _BucketWriteResult((tile, other_bucket_tile), 6, 2)
    with pytest.raises(ValueError, match="contiguous"):
        _BucketWriteResult((replace(tile, bucket_tile_index=1),), 3, 1)
    with pytest.raises(ValueError, match="point_count"):
        _BucketWriteResult((tile,), 2, 1)
    with pytest.raises(ValueError, match="range_count"):
        _BucketWriteResult((tile,), 3, 4)


def test_level_result_rejects_duplicate_bucket_and_tile_ownership() -> None:
    tile = _tile()
    bucket = _BucketWriteResult((tile,), 3, 1)
    with pytest.raises(ValueError, match="bucket IDs"):
        _LevelWriteResult((bucket, bucket))

    other_tile = _tile(bucket_id=1)
    other_bucket = _BucketWriteResult((other_tile,), 3, 1)
    with pytest.raises(ValueError, match="tile coordinates"):
        _LevelWriteResult((bucket, other_bucket))

    other_level_tile = _tile(level=1, tile_x=1)
    other_level_bucket = _BucketWriteResult((other_level_tile,), 3, 1)
    with pytest.raises(ValueError, match="same level"):
        _LevelWriteResult((bucket, other_level_bucket))
