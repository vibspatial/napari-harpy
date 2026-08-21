from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from napari_harpy.core.multi_scale_cache_points_zarr.models import _TileDescriptor
from napari_harpy.core.multi_scale_cache_points_zarr.payload import _PointPayload
from napari_harpy.core.multi_scale_cache_points_zarr.storage.bucket_reader import (
    _BucketReader,
    _exact_row_selection,
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


def _build_partial_bucket(root: Path) -> _BucketWriteResult:
    settings = _ZarrWriteSettings(2, 4, 2, 4, "zstd-v1")
    plan = _BucketPlan(
        level=2,
        bucket_id=4,
        tiles=(_PlannedTile(0, 0, 2), _PlannedTile(1, 0, 3)),
        settings=settings,
    )
    first = _PointPayload(
        x_rel=np.array([0, 1], dtype=np.float32),
        y_rel=np.array([0, 1], dtype=np.float32),
        value_id=np.array([0, 1], dtype=np.uint32),
        point_id=np.array([0, 1], dtype=np.uint64),
    )
    second = _PointPayload(
        x_rel=np.array([2, 3, 4], dtype=np.float32),
        y_rel=np.array([2, 3, 4], dtype=np.float32),
        value_id=np.array([0, 1, 2], dtype=np.uint32),
        point_id=np.array([2, 3, 4], dtype=np.uint64),
    )
    with _BucketWriter(root, plan) as writer:
        writer.write_tile(0, 0, first)
        writer.write_tile(1, 0, second)
        return writer.finalize()


def _load_lookup(reader: _BucketReader) -> None:
    reader.load_lookup_index()


def test_reader_roundtrips_complete_and_selected_payloads(tmp_path: Path) -> None:
    result = _build_bucket(tmp_path)
    first, second = result.tile_descriptors
    with _BucketReader(tmp_path, level=1, bucket_id=3) as reader:
        complete = reader.read_construction_payload(first)
        assert complete.value_id.tolist() == [0, 0, 1, 2, 2]
        assert complete.point_id.tolist() == [1, 4, 3, 2, 5]
        assert complete.x_rel.tolist() == [0, 3, 2, 1, 4]
        assert complete.y_rel.tolist() == [4, 1, 2, 3, 0]

        _load_lookup(reader)
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
        _load_lookup(reader)
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


def test_exact_row_selection_uses_slice_only_for_touching_intervals() -> None:
    contiguous = _exact_row_selection(
        ((1, 2), (2, 5), (5, 7)),
        point_count=10,
        expected_row_count=6,
    )
    assert contiguous == slice(1, 7)

    disjoint = _exact_row_selection(
        ((1, 2), (3, 5), (8, 9)),
        point_count=10,
        expected_row_count=4,
    )
    assert isinstance(disjoint, np.ndarray)
    assert disjoint.dtype == np.dtype(np.int64)
    assert disjoint.flags.c_contiguous
    assert disjoint.tolist() == [1, 3, 4, 8]


@pytest.mark.parametrize(
    ("intervals", "point_count", "expected_row_count", "match"),
    [
        ((), 10, 0, "nonempty"),
        (((3, 5), (2, 3)), 10, 3, "ordered"),
        (((1, 4), (3, 5)), 10, 5, "ordered"),
        (((1, 11),), 10, 10, "inside"),
        (((1, 3),), 10, 1, "reconcile"),
    ],
)
def test_exact_row_selection_rejects_invalid_batch_intervals(
    intervals: tuple[tuple[int, int], ...],
    point_count: int,
    expected_row_count: int,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        _exact_row_selection(
            intervals,
            point_count=point_count,
            expected_row_count=expected_row_count,
        )


def test_display_batch_reads_each_point_array_once_and_splits_payloads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _build_bucket(tmp_path)
    first, second = result.tile_descriptors
    calls: list[tuple[str, tuple[object, ...]]] = []

    with _BucketReader(tmp_path, level=1, bucket_id=3) as reader:
        _load_lookup(reader)
        original_array = reader._array

        class _TrackedArray:
            def __init__(self, name: str) -> None:
                self._name = name
                self._array = original_array(name)

            def get_orthogonal_selection(self, selection: tuple[object, ...]) -> np.ndarray:
                calls.append((self._name, selection))
                return self._array.get_orthogonal_selection(selection)

        def tracked_array(name: str) -> object:
            if name in {"location", "value_id"}:
                return _TrackedArray(name)
            return original_array(name)

        monkeypatch.setattr(reader, "_array", tracked_array)
        complete = reader.read_display_payloads(((first, None), (second, None)))
        assert [name for name, _ in calls] == ["location", "value_id"]
        assert all(selection[0] == slice(0, 8) for _, selection in calls)
        assert complete[0] is not None and complete[0].value_id.tolist() == [0, 0, 1, 2, 2]
        assert complete[1] is not None and complete[1].value_id.tolist() == [1, 1, 3]

        calls.clear()
        selected = reader.read_display_payloads(
            (
                (first, np.array([0, 2], dtype=np.uint32)),
                (second, np.array([1], dtype=np.uint32)),
            )
        )
        assert [name for name, _ in calls] == ["location", "value_id"]
        assert all(isinstance(selection[0], np.ndarray) for _, selection in calls)
        selected_rows = calls[0][1][0]
        assert isinstance(selected_rows, np.ndarray)
        assert selected_rows.tolist() == [0, 1, 3, 4, 5, 6]
        assert selected[0] is not None and selected[0].value_id.tolist() == [0, 0, 2, 2]
        assert selected[1] is not None and selected[1].value_id.tolist() == [1, 1]
        assert all(
            payload is not None
            and payload.location.flags.c_contiguous
            and not payload.location.flags.writeable
            and not payload.value_id.flags.writeable
            for payload in selected
        )


def test_display_batch_omits_unrequested_row_gaps_and_preserves_empty_results(tmp_path: Path) -> None:
    result = _build_bucket(tmp_path)
    first, second = result.tile_descriptors
    with _BucketReader(tmp_path, level=1, bucket_id=3) as reader:
        _load_lookup(reader)
        payloads = reader.read_display_payloads(
            (
                (first, np.array([0], dtype=np.uint32)),
                (second, np.array([3], dtype=np.uint32)),
            )
        )
        assert payloads[0] is not None and payloads[0].value_id.tolist() == [0, 0]
        assert payloads[1] is not None and payloads[1].value_id.tolist() == [3]

        assert reader.read_display_payloads(((second, np.array([2], dtype=np.uint32)),)) == (None,)

        with pytest.raises(ValueError, match="increasing bucket-local"):
            reader.read_display_payloads(((second, None), (first, None)))


def test_direct_construction_and_display_batch_reach_the_final_partial_chunk(tmp_path: Path) -> None:
    result = _build_partial_bucket(tmp_path)
    first, second = result.tile_descriptors
    with _BucketReader(tmp_path, level=2, bucket_id=4) as reader:
        constructed = reader.read_construction_payload(second)
        assert constructed.point_id.tolist() == [2, 3, 4]
        assert constructed.value_id.tolist() == [0, 1, 2]

        _load_lookup(reader)
        displayed = reader.read_display_payloads(((first, None), (second, None)))
        assert displayed[0] is not None and displayed[0].value_id.tolist() == [0, 1]
        assert displayed[1] is not None and displayed[1].value_id.tolist() == [0, 1, 2]


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
    result = _build_bucket(tmp_path)
    descriptor = result.tile_descriptors[0]
    with _BucketReader(tmp_path, level=1, bucket_id=3) as reader:
        _load_lookup(reader)
        with pytest.raises(ValueError, match="selected_value_ids"):
            reader.read_display_payload(descriptor, selected)  # type: ignore[arg-type]


def test_closing_reader_releases_resident_lookup_index(tmp_path: Path) -> None:
    _build_bucket(tmp_path)
    reader = _BucketReader(tmp_path, level=1, bucket_id=3)
    with reader:
        _load_lookup(reader)
        assert reader.lookup_index_loaded
        assert reader.resident_lookup_bytes == reader.projected_lookup_bytes

    assert not reader.lookup_index_loaded
    assert reader.resident_lookup_bytes == 0


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
