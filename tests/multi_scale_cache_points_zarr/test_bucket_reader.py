from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from napari_harpy.core.multi_scale_cache_points_zarr.models import _TileDescriptor
from napari_harpy.core.multi_scale_cache_points_zarr.payload import _PointPayload
from napari_harpy.core.multi_scale_cache_points_zarr.storage.bucket_reader import (
    _BucketReader,
    _coalesced_read_blocks_for_intervals,
)
from napari_harpy.core.multi_scale_cache_points_zarr.storage.bucket_writer import _BucketWriter
from napari_harpy.core.multi_scale_cache_points_zarr.storage.models import (
    _BucketPlan,
    _BucketWriteResult,
    _PlannedTile,
    _ZarrWriteSettings,
)


def _build_bucket(root: Path) -> _BucketWriteResult:
    settings = _ZarrWriteSettings(2, 4, 2, 4, "zstd-v1")
    plan = _BucketPlan(
        level=1,
        bucket_id=3,
        tiles=(_PlannedTile(0, 0, 5), _PlannedTile(1, 0, 3)),
        settings=settings,
    )
    first = _PointPayload(
        x_rel=np.array([4, 3, 2, 1, 0], dtype=np.float32),
        y_rel=np.arange(5, dtype=np.float32),
        value_id=np.array([2, 0, 1, 2, 0], dtype=np.uint32),
        point_id=np.array([5, 4, 3, 2, 1], dtype=np.uint64),
    )
    second = _PointPayload(
        x_rel=np.array([2, 1, 0], dtype=np.float32),
        y_rel=np.array([5, 6, 7], dtype=np.float32),
        value_id=np.array([3, 1, 1], dtype=np.uint32),
        point_id=np.array([8, 7, 6], dtype=np.uint64),
    )
    with _BucketWriter(root, plan) as writer:
        writer.write_tile(0, 0, first)
        writer.write_tile(1, 0, second)
        return writer.finalize()


def test_reader_roundtrips_complete_and_selected_payloads(tmp_path: Path) -> None:
    result = _build_bucket(tmp_path)
    first, second = result.tile_descriptors
    with _BucketReader(tmp_path, level=1, bucket_id=3) as reader:
        complete = reader.read_complete(first)
        assert complete.value_id.tolist() == [0, 0, 1, 2, 2]
        assert complete.point_id.tolist() == [1, 4, 3, 2, 5]
        assert complete.x_rel.tolist() == [0, 3, 2, 1, 4]
        assert complete.y_rel.tolist() == [4, 1, 2, 3, 0]

        selected = reader.read_selected(first, np.array([0, 2], dtype=np.uint32))
        assert selected is not None
        assert selected.value_id.tolist() == [0, 0, 2, 2]
        assert selected.point_id.tolist() == [1, 4, 2, 5]
        assert reader.read_selected(second, np.array([2], dtype=np.uint32)) is None


@pytest.mark.parametrize(
    ("intervals", "expected"),
    [
        (((1, 2),), ((1, 2),)),
        (((1, 2), (3, 4), (8, 9)), ((1, 4), (8, 9))),
        (((1, 2), (5, 6)), ((1, 6),)),  # Consecutive touched chunks share one read envelope.
        (((3, 6),), ((3, 6),)),
    ],
)
def test_selected_read_planner_coalesces_by_chunk_without_expanding_outer_bounds(
    intervals: tuple[tuple[int, int], ...],
    expected: tuple[tuple[int, int], ...],
) -> None:
    assert _coalesced_read_blocks_for_intervals(
        intervals,
        chunk_rows=4,
    ) == expected


@pytest.mark.parametrize(
    "selected",
    [
        np.array([], dtype=np.uint32),
        np.array([1, 1], dtype=np.uint32),
        np.array([2, 1], dtype=np.uint32),
        np.array([1], dtype=np.uint64),
        np.array([[1]], dtype=np.uint32),
    ],
)
def test_reader_rejects_invalid_selected_value_ids(tmp_path: Path, selected: np.ndarray) -> None:
    descriptor = _build_bucket(tmp_path).tile_descriptors[0]
    with _BucketReader(tmp_path, level=1, bucket_id=3) as reader:
        with pytest.raises(ValueError, match="selected_value_ids"):
            reader.read_selected(descriptor, selected)  # type: ignore[arg-type]


def test_reader_rejects_unknown_descriptor_and_calls_after_close(tmp_path: Path) -> None:
    descriptor = _build_bucket(tmp_path).tile_descriptors[0]
    reader = _BucketReader(tmp_path, level=1, bucket_id=3)
    with pytest.raises(RuntimeError, match="not open"):
        reader.read_complete(descriptor)
    with reader:
        wrong_bucket = _TileDescriptor(1, 4, 0, 0, 0, 5)
        with pytest.raises(ValueError, match="different bucket"):
            reader.read_complete(wrong_bucket)
        wrong_coordinate = _TileDescriptor(1, 3, 0, 2, 0, 5)
        with pytest.raises(ValueError, match="coordinates"):
            reader.read_complete(wrong_coordinate)
        wrong_count = _TileDescriptor(1, 3, 0, 0, 0, 4)
        with pytest.raises(ValueError, match="count"):
            reader.read_complete(wrong_count)
    with pytest.raises(RuntimeError, match="not open"):
        reader.read_complete(descriptor)
    with pytest.raises(RuntimeError, match="entered only once"):
        with reader:
            pass


def test_reader_rejects_missing_bucket(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        with _BucketReader(tmp_path, level=0, bucket_id=0):
            pass
