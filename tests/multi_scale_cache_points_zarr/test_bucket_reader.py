from __future__ import annotations

from dataclasses import replace
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


def test_reader_roundtrips_construction_and_complete_display_payloads(tmp_path: Path) -> None:
    result = _build_bucket(tmp_path)
    first = result.tile_descriptors[0]
    with _BucketReader(tmp_path, level=1, bucket_id=3) as reader:
        complete = reader.read_construction_payload(first)
        assert complete.value_id.tolist() == [0, 0, 1, 2, 2]
        assert complete.point_id.tolist() == [1, 4, 3, 2, 5]
        assert complete.x_rel.tolist() == [0, 3, 2, 1, 4]
        assert complete.y_rel.tolist() == [4, 1, 2, 3, 0]

        reader.set_tile_descriptors(result.tile_descriptors)
        displayed = reader.read_complete_display_payload(first)
        np.testing.assert_array_equal(displayed.value_id, complete.value_id)
        np.testing.assert_array_equal(displayed.location, np.column_stack((complete.x_rel, complete.y_rel)))
        assert not displayed.location.flags.writeable
        assert not displayed.value_id.flags.writeable


def test_visualization_reader_never_requires_point_id_payload_chunks(tmp_path: Path) -> None:
    result = _build_bucket(tmp_path)
    first = result.tile_descriptors[0]
    point_id_objects = [path for path in (tmp_path / first.bucket_path / "point_id" / "c").rglob("*") if path.is_file()]
    assert point_id_objects
    point_id_objects[0].unlink()

    with _BucketReader(tmp_path, level=1, bucket_id=3) as reader:
        reader.set_tile_descriptors(result.tile_descriptors)
        complete = reader.read_complete_display_payload(first)
        assert complete is not None
        assert complete.value_id.tolist() == [0, 0, 1, 2, 2]
        assert complete.location.tolist() == [[0, 4], [3, 1], [2, 2], [1, 3], [4, 0]]

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
        (((1, 3),), 0, 2, "point_count"),
        (((1, 3),), 10, 0, "expected_row_count"),
        (((1.0, 3),), 10, 2, "inside"),
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
        reader.set_tile_descriptors(result.tile_descriptors)
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
        complete = reader.read_complete_display_payloads((first, second))
        assert [name for name, _ in calls] == ["location", "value_id"]
        assert all(selection[0] == slice(0, 8) for _, selection in calls)
        assert complete[0] is not None and complete[0].value_id.tolist() == [0, 0, 1, 2, 2]
        assert complete[1] is not None and complete[1].value_id.tolist() == [1, 1, 3]

        assert complete[0].location.base is complete[1].location.base
        assert complete[0].value_id.base is complete[1].value_id.base
        assert all(
            payload.location.flags.c_contiguous
            and payload.value_id.flags.c_contiguous
            and not payload.location.flags.writeable
            and not payload.value_id.flags.writeable
            for payload in complete
        )


@pytest.mark.parametrize("malformed", ["empty", "list", "pair", "invalid-second"])
def test_complete_display_batch_validates_requests_before_resolution_or_io(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    malformed: str,
) -> None:
    first = _build_bucket(tmp_path).tile_descriptors[0]
    requests = {
        "empty": (),
        "list": [first],
        "pair": ((first, None),),
        "invalid-second": (first, None),
    }[malformed]
    with _BucketReader(tmp_path, level=1, bucket_id=3) as reader:

        def reject_side_effect(*args: object, **kwargs: object) -> None:
            raise AssertionError("Malformed request reached resolution or IO.")

        monkeypatch.setattr(reader, "resolve_complete_tile_interval", reject_side_effect)
        monkeypatch.setattr(reader, "_array", reject_side_effect)
        with pytest.raises(ValueError, match="nonempty tuple|_TileDescriptor"):
            reader.read_complete_display_payloads(requests)


def test_complete_display_requires_installed_descriptors_before_payload_io(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _build_bucket(tmp_path)
    with _BucketReader(tmp_path, level=1, bucket_id=3) as reader:
        with monkeypatch.context() as patches:

            def reject_io(*args: object, **kwargs: object) -> None:
                raise AssertionError("Uninitialized addressing reached physical IO.")

            patches.setattr(reader, "_array", reject_io)
            with pytest.raises(RuntimeError, match="set_tile_descriptors"):
                reader.read_complete_display_payload(result.tile_descriptors[0])
        reader.set_tile_descriptors(result.tile_descriptors)
        assert len(reader.read_complete_display_payload(result.tile_descriptors[0]).value_id) == 5


def test_complete_display_batch_omits_skipped_tiles_and_rejects_out_of_order_requests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Read nonadjacent complete tiles without selecting intervening tile rows.

    Check that location and value-ID arrays use the same exact row selection
    and return aligned per-tile payloads, without accessing point IDs or sparse
    ranges. Reversed or duplicate tile requests must fail before payload IO.
    """
    plan = _BucketPlan(
        level=0,
        bucket_id=0,
        tiles=tuple(_PlannedTile(x, 0, 2) for x in range(3)),
        settings=_ZarrWriteSettings(2, 4, 2, 4, "zstd-v1"),
    )
    with _BucketWriter(tmp_path, plan) as writer:
        for x in range(3):
            writer.write_tile(
                x,
                0,
                _PointPayload(
                    x_rel=np.array([x, x + 0.5], dtype=np.float32),
                    y_rel=np.array([0, 1], dtype=np.float32),
                    value_id=np.array([0, 1], dtype=np.uint32),
                    point_id=np.array([2 * x, 2 * x + 1], dtype=np.uint64),
                ),
            )
        result = writer.finalize()
    with _BucketReader(tmp_path, level=0, bucket_id=0) as reader:
        reader.set_tile_descriptors(result.tile_descriptors)
        first, _, last = result.tile_descriptors
        original_array = reader._array
        selections = []

        class TrackedArray:
            def __init__(self, name):
                self.array = original_array(name)

            def get_orthogonal_selection(self, selection):
                selections.append(selection[0])
                return self.array.get_orthogonal_selection(selection)

        def tracked_array(name):
            if name.startswith("ranges/") or name == "point_id":
                raise AssertionError("Display accessed sparse ranges or point IDs.")
            return TrackedArray(name)

        monkeypatch.setattr(reader, "_array", tracked_array)
        payloads = reader.read_complete_display_payloads((first, last))
        assert len(selections) == 2
        for selection in selections:
            np.testing.assert_array_equal(selection, [0, 1, 4, 5])
        assert [payload.location[:, 0].tolist() for payload in payloads] == [[0, 0.5], [2, 2.5]]
        assert [payload.value_id.tolist() for payload in payloads] == [[0, 1], [0, 1]]
        selections.clear()
        for invalid in ((last, first), (first, first)):
            with pytest.raises(ValueError, match="increasing bucket-local"):
                reader.read_complete_display_payloads(invalid)
        assert selections == []


def test_direct_construction_and_display_batch_reach_the_final_partial_chunk(tmp_path: Path) -> None:
    result = _build_partial_bucket(tmp_path)
    first, second = result.tile_descriptors
    with _BucketReader(tmp_path, level=2, bucket_id=4) as reader:
        constructed = reader.read_construction_payload(second)
        assert constructed.point_id.tolist() == [2, 3, 4]
        assert constructed.value_id.tolist() == [0, 1, 2]

        reader.set_tile_descriptors(result.tile_descriptors)
        displayed = reader.read_complete_display_payloads((first, second))
        assert displayed[0] is not None and displayed[0].value_id.tolist() == [0, 1]
        assert displayed[1] is not None and displayed[1].value_id.tolist() == [0, 1, 2]


def test_reader_rejects_unknown_descriptor_and_calls_after_close(tmp_path: Path) -> None:
    descriptor = _build_bucket(tmp_path).tile_descriptors[0]
    reader = _BucketReader(tmp_path, level=1, bucket_id=3)
    with pytest.raises(RuntimeError, match="not open"):
        reader.read_construction_payload(descriptor)
    with reader:
        wrong_bucket = _TileDescriptor(1, 4, 0, 0, 0, 0, 5)
        with pytest.raises(ValueError, match="different bucket"):
            reader.read_construction_payload(wrong_bucket)
        wrong_coordinate = _TileDescriptor(1, 3, 0, 0, 2, 0, 5)
        with pytest.raises(ValueError, match="coordinates"):
            reader.read_construction_payload(wrong_coordinate)
        wrong_count = _TileDescriptor(1, 3, 0, 0, 0, 0, 4)
        with pytest.raises(ValueError, match="count"):
            reader.read_construction_payload(wrong_count)
        with pytest.raises(ValueError, match="row start"):
            reader.read_construction_payload(replace(descriptor, bucket_row_start=1))
    with pytest.raises(RuntimeError, match="not open"):
        reader.read_construction_payload(descriptor)
    with pytest.raises(RuntimeError, match="entered only once"):
        with reader:
            pass


def test_reader_rejects_missing_bucket(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        with _BucketReader(tmp_path, level=0, bucket_id=0):
            pass
