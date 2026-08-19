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
        complete = reader.read_construction_payload(first)
        assert complete.value_id.tolist() == [0, 0, 1, 2, 2]
        assert complete.point_id.tolist() == [1, 4, 3, 2, 5]
        assert complete.x_rel.tolist() == [0, 3, 2, 1, 4]
        assert complete.y_rel.tolist() == [4, 1, 2, 3, 0]

        selected = reader.read_display_payload(first, np.array([0, 2], dtype=np.uint32))
        assert selected is not None
        assert selected.value_id.tolist() == [0, 0, 2, 2]
        assert selected.location.tolist() == [[0, 4], [3, 1], [1, 3], [4, 0]]
        assert not selected.location.flags.writeable
        assert not selected.value_id.flags.writeable
        assert reader.read_display_payload(second, np.array([2], dtype=np.uint32)) is None


def test_visualization_reader_never_requires_point_id_payload_chunks(tmp_path: Path) -> None:
    result = _build_bucket(tmp_path)
    first = result.tile_descriptors[0]
    point_id_objects = [path for path in (tmp_path / first.bucket_path / "point_id" / "c").rglob("*") if path.is_file()]
    assert point_id_objects
    point_id_objects[0].unlink()

    with _BucketReader(tmp_path, level=1, bucket_id=3) as reader:
        complete = reader.read_display_payload(first)
        assert complete is not None
        assert complete.value_id.tolist() == [0, 0, 1, 2, 2]
        assert complete.location.tolist() == [[0, 4], [3, 1], [2, 2], [1, 3], [4, 0]]

        selected = reader.read_display_payload(first, np.array([0, 2], dtype=np.uint32))
        assert selected is not None
        assert selected.value_id.tolist() == [0, 0, 2, 2]
        assert len(selected.location) == len(selected.value_id) == 4

        with pytest.raises(Exception, match="chunk|Chunk|shard|Shard"):
            reader.read_construction_payload(first)


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
    assert (
        _coalesced_read_blocks_for_intervals(
            intervals,
            chunk_rows=4,
        )
        == expected
    )


def test_point_read_plan_keeps_exact_intervals_and_coalesced_blocks(tmp_path: Path) -> None:
    _build_bucket(tmp_path)
    with _BucketReader(tmp_path, level=1, bucket_id=3) as reader:
        plan = reader._point_read_plan(((1, 2), (3, 4), (6, 7)))

    assert plan.intervals == ((1, 2), (3, 4), (6, 7))
    assert plan.blocks == ((1, 4), (6, 7))


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
            reader.read_display_payload(descriptor, selected)  # type: ignore[arg-type]


def test_reader_rejects_unknown_descriptor_and_calls_after_close(tmp_path: Path) -> None:
    descriptor = _build_bucket(tmp_path).tile_descriptors[0]
    reader = _BucketReader(tmp_path, level=1, bucket_id=3)
    with pytest.raises(RuntimeError, match="not open"):
        reader.read_construction_payload(descriptor)
    with reader:
        wrong_bucket = _TileDescriptor(1, 4, 0, 0, 0, 5)
        with pytest.raises(ValueError, match="different bucket"):
            reader.read_construction_payload(wrong_bucket)
        wrong_coordinate = _TileDescriptor(1, 3, 0, 2, 0, 5)
        with pytest.raises(ValueError, match="coordinates"):
            reader.read_construction_payload(wrong_coordinate)
        wrong_count = _TileDescriptor(1, 3, 0, 0, 0, 4)
        with pytest.raises(ValueError, match="count"):
            reader.read_construction_payload(wrong_count)
    with pytest.raises(RuntimeError, match="not open"):
        reader.read_construction_payload(descriptor)
    with pytest.raises(RuntimeError, match="entered only once"):
        with reader:
            pass


def test_reader_rejects_missing_bucket(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        with _BucketReader(tmp_path, level=0, bucket_id=0):
            pass
