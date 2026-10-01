from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import zarr
from zarr.storage import LocalStore

from napari_harpy.core.multi_scale_cache_points_zarr.payload import _PointPayload
from napari_harpy.core.multi_scale_cache_points_zarr.storage.bucket_reader import _BucketReader
from napari_harpy.core.multi_scale_cache_points_zarr.storage.bucket_validation import _validate_bucket
from napari_harpy.core.multi_scale_cache_points_zarr.storage.bucket_writer import _BucketWriter
from napari_harpy.core.multi_scale_cache_points_zarr.storage.models import (
    _BucketPlan,
    _BucketWriteResult,
    _PlannedTile,
    _ZarrWriteSettings,
)


def _build_bucket(root: Path, *, all_zero: bool = False) -> tuple[_BucketPlan, _BucketWriteResult]:
    plan = _BucketPlan(
        level=0,
        bucket_id=0,
        tiles=(_PlannedTile(0, 0, 5), _PlannedTile(1, 0, 3)),
        settings=_ZarrWriteSettings(2, 4, 2, 4, "zstd-v1"),
    )
    if all_zero:
        first = _PointPayload(
            x_rel=np.zeros(5, dtype=np.float32),
            y_rel=np.zeros(5, dtype=np.float32),
            value_id=np.zeros(5, dtype=np.uint32),
            point_id=np.zeros(5, dtype=np.uint64),
        )
    else:
        first = _PointPayload(
            x_rel=np.arange(5, dtype=np.float32),
            y_rel=np.arange(5, dtype=np.float32),
            value_id=np.array([2, 0, 1, 2, 0], dtype=np.uint32),
            point_id=np.array([5, 4, 3, 2, 1], dtype=np.uint64),
        )
    second = _PointPayload(
        x_rel=np.arange(3, dtype=np.float32),
        y_rel=np.arange(3, dtype=np.float32),
        value_id=np.array([3, 1, 1], dtype=np.uint32),
        point_id=np.array([8, 7, 6], dtype=np.uint64),
    )
    with _BucketWriter(root, plan) as writer:
        writer.write_tile(0, 0, first)
        writer.write_tile(1, 0, second)
        return plan, writer.finalize()


def _open_writable(root: Path, plan: _BucketPlan) -> tuple[LocalStore, zarr.Group]:
    store = LocalStore(root / plan.bucket_path, read_only=False)
    group = zarr.open_group(store=store, mode="a", zarr_format=3, use_consolidated=False)
    return store, group


def test_validator_reconstructs_result_without_writer_state(tmp_path: Path) -> None:
    _, expected = _build_bucket(tmp_path)
    actual = _validate_bucket(tmp_path, level=0, bucket_id=0)
    assert actual == expected


def test_all_zero_inner_chunks_are_physical_and_validate_strictly(tmp_path: Path) -> None:
    _, expected = _build_bucket(tmp_path, all_zero=True)
    actual = _validate_bucket(tmp_path, level=0, bucket_id=0)
    assert actual == expected
    with _BucketReader(tmp_path, level=0, bucket_id=0) as reader:
        payload = reader.read_construction_payload(expected.tile_descriptors[0])
    assert not payload.x_rel.any()
    assert not payload.value_id.any()
    assert not payload.point_id.any()


@pytest.mark.parametrize(
    ("attribute", "value", "message"),
    [
        ("payload_schema_version", 2, "schema version"),
        ("point_row_order", ["point_id"], "row ordering"),
        ("point_order", ["tile_y", "tile_x", "value_id", "point_id"], "root attributes"),
        ("coordinate_encoding", "unknown", "coordinate encoding"),
        ("codec_id", "unknown-v1", "Unsupported"),
        ("point_count", 7, "shape"),
        ("point_chunk_rows", 2, "root attributes"),
    ],
)
def test_validator_rejects_corrupt_root_attributes(
    tmp_path: Path,
    attribute: str,
    value: object,
    message: str,
) -> None:
    plan, _ = _build_bucket(tmp_path)
    store, root = _open_writable(tmp_path, plan)
    root.attrs[attribute] = value
    store.close()
    with pytest.raises(ValueError, match=message):
        _validate_bucket(tmp_path, level=0, bucket_id=0)


@pytest.mark.parametrize("corruption", ["offset", "indptr", "range_count", "range_value", "point_rows"])
def test_validator_rejects_corrupt_pointer_range_and_point_content(tmp_path: Path, corruption: str) -> None:
    plan, _ = _build_bucket(tmp_path)
    store, root = _open_writable(tmp_path, plan)
    if corruption == "offset":
        root["tile_offset"][1] = 0
    elif corruption == "indptr":
        root["ranges/tile_indptr"][1] = 0
    elif corruption == "range_count":
        root["ranges/row_count"][0] = 1
    elif corruption == "range_value":
        root["ranges/value_id"][0] = 9
    else:
        root["value_id"][0] = 9
    store.close()
    with pytest.raises(ValueError):
        _validate_bucket(tmp_path, level=0, bucket_id=0)


def test_validator_rejects_unexpected_logical_node(tmp_path: Path) -> None:
    plan, _ = _build_bucket(tmp_path)
    store, root = _open_writable(tmp_path, plan)
    root.create_group("unexpected")
    store.close()
    with pytest.raises(ValueError, match="unexpected"):
        _validate_bucket(tmp_path, level=0, bucket_id=0)


@pytest.mark.parametrize("array_path", ["point_id", "ranges/row_start"])
def test_validator_rejects_parallel_array_with_misaligned_chunk_rows(tmp_path: Path, array_path: str) -> None:
    plan, _ = _build_bucket(tmp_path)
    store, root = _open_writable(tmp_path, plan)
    parent, name = (root, array_path) if "/" not in array_path else (root["ranges"], array_path.split("/")[1])
    values = np.asarray(parent[name][:])
    del parent[name]
    parent.create_array(name, data=values, chunks=(1,), shards=(4,))
    store.close()

    with pytest.raises(ValueError, match="shape, chunks, or shards"):
        _validate_bucket(tmp_path, level=0, bucket_id=0)


def test_missing_shard_fails_strict_reader_and_validator(tmp_path: Path) -> None:
    plan, result = _build_bucket(tmp_path, all_zero=True)
    shard_files = [path for path in (tmp_path / plan.bucket_path / "location" / "c").rglob("*") if path.is_file()]
    assert shard_files
    shard_files[0].unlink()

    with pytest.raises(Exception, match="chunk|Chunk|shard|Shard"):
        _validate_bucket(tmp_path, level=0, bucket_id=0)
    with _BucketReader(tmp_path, level=0, bucket_id=0) as reader:
        with pytest.raises(Exception, match="chunk|Chunk|shard|Shard"):
            reader.read_construction_payload(result.tile_descriptors[0])


def test_validator_rejects_wrong_requested_identity_and_missing_bucket(tmp_path: Path) -> None:
    _build_bucket(tmp_path)
    with pytest.raises(FileNotFoundError):
        _validate_bucket(tmp_path, level=0, bucket_id=1)
