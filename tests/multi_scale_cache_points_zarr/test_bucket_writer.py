from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import zarr
from zarr.codecs import BytesCodec, Crc32cCodec, ShardingCodec, ZstdCodec
from zarr.storage import LocalStore

from napari_harpy.core.multi_scale_cache_points_zarr.payload import _PointPayload
from napari_harpy.core.multi_scale_cache_points_zarr.storage.bucket_writer import _BucketWriter
from napari_harpy.core.multi_scale_cache_points_zarr.storage.models import (
    _BucketPlan,
    _PlannedTile,
    _ZarrWriteSettings,
)


def _settings(*, codec_id: str = "zstd-v1") -> _ZarrWriteSettings:
    return _ZarrWriteSettings(
        point_chunk_rows=2,
        point_shard_rows=4,
        range_chunk_rows=2,
        range_shard_rows=4,
        codec_id=codec_id,
    )


def _payload(values: list[int], *, first_point_id: int = 0) -> _PointPayload:
    size = len(values)
    return _PointPayload(
        x_rel=np.arange(size, dtype=np.float32)[::-1].copy(),
        y_rel=np.arange(size, dtype=np.float32),
        value_id=np.asarray(values, dtype=np.uint32),
        point_id=np.arange(first_point_id, first_point_id + size, dtype=np.uint64)[::-1].copy(),
    )


def test_writer_persists_exact_sharded_layout_ranges_and_attributes(tmp_path: Path) -> None:
    plan = _BucketPlan(
        level=0,
        bucket_id=2,
        tiles=(_PlannedTile(0, 0, 5), _PlannedTile(1, 0, 3)),
        settings=_settings(),
    )
    with _BucketWriter(tmp_path, plan) as writer:
        writer.write_tile(0, 0, _payload([2, 0, 1, 2, 0]))
        writer.write_tile(1, 0, _payload([3, 1, 1], first_point_id=5))
        result = writer.finalize()

    assert result.point_count == 8
    assert result.range_count == 5
    assert [tile.n_points for tile in result.tile_descriptors] == [5, 3]

    with LocalStore(tmp_path / plan.bucket_path, read_only=True) as store:
        root = zarr.open_group(store=store, mode="r", zarr_format=3, use_consolidated=False)
        assert root["location"].shape == (8, 2)
        assert root["location"].chunks == (2, 2)
        assert root["location"].shards == (4, 2)
        assert root["ranges/value_id"].shape == (5,)
        assert root["ranges/value_id"].chunks == (2,)
        assert root["ranges/value_id"].shards == (4,)
        assert root["tile_x"].shards is None
        assert root["ranges/tile_indptr"].shards is None
        assert root["tile_offset"][:].tolist() == [0, 5, 8]
        assert root["ranges/tile_indptr"][:].tolist() == [0, 3, 5]
        assert root["ranges/value_id"][:].tolist() == [0, 1, 2, 1, 3]
        assert root["ranges/row_start"][:].tolist() == [0, 2, 3, 5, 7]
        assert root["ranges/row_count"][:].tolist() == [2, 1, 2, 2, 1]
        assert dict(root.attrs) == {
            "payload_schema_version": 1,
            "level": 0,
            "bucket_id": 2,
            "tile_count": 2,
            "point_count": 8,
            "range_count": 5,
            "point_row_order": ["tile_y", "tile_x", "value_id", "point_id"],
            "coordinate_encoding": "tile-relative-xy-float32-v1",
            "codec_id": "zstd-v1",
        }

        sharding = root["value_id"].metadata.codecs
        assert sharding == (
            ShardingCodec(
                chunk_shape=(2,),
                codecs=(BytesCodec(endian="little"), ZstdCodec(level=3, checksum=True)),
                index_codecs=(BytesCodec(endian="little"), Crc32cCodec()),
                index_location="end",
            ),
        )


def test_writer_grows_and_trims_range_arrays_at_shard_boundaries(tmp_path: Path) -> None:
    plan = _BucketPlan(
        level=0,
        bucket_id=0,
        tiles=(_PlannedTile(0, 0, 6),),
        settings=_settings(),
    )
    with _BucketWriter(tmp_path, plan) as writer:
        writer.write_tile(0, 0, _payload([5, 4, 3, 2, 1, 0]))
        result = writer.finalize()

    assert result.range_count == 6
    with LocalStore(tmp_path / plan.bucket_path, read_only=True) as store:
        root = zarr.open_group(store=store, mode="r", zarr_format=3, use_consolidated=False)
        assert root["ranges/value_id"].shape == (6,)
        assert root["ranges/value_id"][:].tolist() == list(range(6))


@pytest.mark.parametrize("failure", ["wrong_tile", "wrong_count", "negative", "early_finalize"])
def test_writer_failure_closes_and_removes_exact_partial_target(tmp_path: Path, failure: str) -> None:
    plan = _BucketPlan(
        level=0,
        bucket_id=0,
        tiles=(_PlannedTile(0, 0, 2),),
        settings=_settings(),
    )
    writer = _BucketWriter(tmp_path, plan)
    with pytest.raises((ValueError, RuntimeError)):
        with writer:
            if failure == "wrong_tile":
                writer.write_tile(1, 0, _payload([0, 1]))
            elif failure == "wrong_count":
                writer.write_tile(0, 0, _payload([0]))
            elif failure == "negative":
                payload = _payload([0, 1])
                negative = _PointPayload(
                    x_rel=np.array([-1, 0], dtype=np.float32),
                    y_rel=payload.y_rel.copy(),
                    value_id=payload.value_id.copy(),
                    point_id=payload.point_id.copy(),
                )
                writer.write_tile(0, 0, negative)
            else:
                writer.finalize()
    assert not writer.target.exists()
    with pytest.raises(RuntimeError, match="not open"):
        writer.write_tile(0, 0, _payload([0, 1]))


def test_writer_context_without_finalize_removes_partial_target(tmp_path: Path) -> None:
    plan = _BucketPlan(0, 0, (_PlannedTile(0, 0, 2),), _settings())
    writer = _BucketWriter(tmp_path, plan)
    with writer:
        writer.write_tile(0, 0, _payload([0, 1]))
    assert not writer.target.exists()


def test_writer_refuses_existing_target_and_preserves_finalized_bucket(tmp_path: Path) -> None:
    plan = _BucketPlan(0, 0, (_PlannedTile(0, 0, 2),), _settings())
    writer = _BucketWriter(tmp_path, plan)
    with writer:
        writer.write_tile(0, 0, _payload([0, 1]))
        writer.finalize()
    with pytest.raises(RuntimeError, match="not open"):
        writer.finalize()
    assert writer.target.exists()
    with pytest.raises(FileExistsError):
        with _BucketWriter(tmp_path, plan):
            pass
    assert writer.target.exists()


def test_writer_rejects_unknown_codec_before_leaving_a_target(tmp_path: Path) -> None:
    plan = _BucketPlan(0, 0, (_PlannedTile(0, 0, 1),), _settings(codec_id="unknown-v1"))
    writer = _BucketWriter(tmp_path, plan)
    with pytest.raises(ValueError, match="Unsupported"):
        with writer:
            pass
    assert not writer.target.exists()
