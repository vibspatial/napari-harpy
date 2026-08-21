from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from napari_harpy.core.multi_scale_cache_points_zarr.models import _TileDescriptor
from napari_harpy.core.multi_scale_cache_points_zarr.payload import _PointPayload
from napari_harpy.core.multi_scale_cache_points_zarr.storage.bucket_reader import _BucketReader
from napari_harpy.core.multi_scale_cache_points_zarr.storage.bucket_writer import _BucketWriter
from napari_harpy.core.multi_scale_cache_points_zarr.storage.models import (
    _BucketPlan,
    _PlannedTile,
    _ZarrWriteSettings,
)
from napari_harpy.core.multi_scale_cache_points_zarr.storage.reader_cache import _BucketReaderCache


def _build_bucket(root: Path, *, level: int, bucket_id: int, tile_x: int) -> _TileDescriptor:
    plan = _BucketPlan(
        level=level,
        bucket_id=bucket_id,
        tiles=(_PlannedTile(tile_x, 0, 1),),
        settings=_ZarrWriteSettings(2, 4, 2, 4, "zstd-v1"),
    )
    payload = _PointPayload(
        x_rel=np.array([tile_x], dtype=np.float32),
        y_rel=np.array([level], dtype=np.float32),
        value_id=np.array([bucket_id], dtype=np.uint32),
        point_id=np.array([level * 100 + bucket_id], dtype=np.uint64),
    )
    with _BucketWriter(root, plan) as writer:
        writer.write_tile(tile_x, 0, payload)
        return writer.finalize().tile_descriptors[0]


def _assert_open(reader: _BucketReader, descriptor: _TileDescriptor) -> None:
    assert reader.read_construction_payload(descriptor).n_points == descriptor.n_points


def _assert_closed(reader: _BucketReader, descriptor: _TileDescriptor) -> None:
    with pytest.raises(RuntimeError, match="not open"):
        reader.read_construction_payload(descriptor)


def test_reader_cache_reuses_hits_and_evicts_least_recently_used(tmp_path: Path) -> None:
    first_descriptor = _build_bucket(tmp_path, level=0, bucket_id=1, tile_x=1)
    second_descriptor = _build_bucket(tmp_path, level=0, bucket_id=2, tile_x=2)
    third_descriptor = _build_bucket(tmp_path, level=0, bucket_id=3, tile_x=3)

    with _BucketReaderCache(tmp_path, max_open_readers=2) as cache:
        first = cache.get(level=0, bucket_id=1)
        second = cache.get(level=0, bucket_id=2)
        assert cache.get(level=0, bucket_id=1) is first

        third = cache.get(level=0, bucket_id=3)

        assert cache.open_reader_count == 2
        _assert_open(first, first_descriptor)
        _assert_closed(second, second_descriptor)
        _assert_open(third, third_descriptor)

    _assert_closed(first, first_descriptor)
    _assert_closed(third, third_descriptor)


def test_reader_cache_bound_one_closes_evicted_reader(tmp_path: Path) -> None:
    first_descriptor = _build_bucket(tmp_path, level=0, bucket_id=4, tile_x=4)
    second_descriptor = _build_bucket(tmp_path, level=0, bucket_id=5, tile_x=5)

    with _BucketReaderCache(tmp_path, max_open_readers=1) as cache:
        first = cache.get(level=0, bucket_id=4)
        _assert_open(first, first_descriptor)

        second = cache.get(level=0, bucket_id=5)

        _assert_closed(first, first_descriptor)
        _assert_open(second, second_descriptor)

    _assert_closed(second, second_descriptor)


def test_failed_open_is_not_cached(tmp_path: Path) -> None:
    with _BucketReaderCache(tmp_path, max_open_readers=2) as cache:
        with pytest.raises(FileNotFoundError):
            cache.get(level=0, bucket_id=7)
        assert cache.open_reader_count == 0

        with pytest.raises(FileNotFoundError):
            cache.get(level=0, bucket_id=7)
        assert cache.open_reader_count == 0


def test_exceptional_unwind_closes_every_cached_reader(tmp_path: Path) -> None:
    first_descriptor = _build_bucket(tmp_path, level=0, bucket_id=1, tile_x=1)
    second_descriptor = _build_bucket(tmp_path, level=1, bucket_id=2, tile_x=2)
    entered_readers: list[tuple[_BucketReader, _TileDescriptor]] = []

    with pytest.raises(RuntimeError, match="injected body failure"):
        with _BucketReaderCache(tmp_path, max_open_readers=3) as cache:
            entered_readers.append((cache.get(level=0, bucket_id=1), first_descriptor))
            entered_readers.append((cache.get(level=1, bucket_id=2), second_descriptor))
            raise RuntimeError("injected body failure")

    for reader, descriptor in entered_readers:
        _assert_closed(reader, descriptor)


def test_reader_cache_rejects_invalid_lifecycle_and_bound(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="max_open_readers"):
        _BucketReaderCache(tmp_path, max_open_readers=0)

    cache = _BucketReaderCache(tmp_path, max_open_readers=1)
    with pytest.raises(RuntimeError, match="not open"):
        cache.get(level=0, bucket_id=0)
    with cache:
        with pytest.raises(RuntimeError, match="entered only once"):
            cache.__enter__()
    with pytest.raises(RuntimeError, match="not open"):
        cache.get(level=0, bucket_id=0)
