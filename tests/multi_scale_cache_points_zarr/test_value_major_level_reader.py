"""Test level-local value-major layouts, pointer loading, and bounded reads."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import zarr
from zarr.storage import LocalStore

from napari_harpy.core.multi_scale_cache_points_zarr.cache_format import _ValueMajorMetadata
from napari_harpy.core.multi_scale_cache_points_zarr.storage._schema import _array_creation_options
from napari_harpy.core.multi_scale_cache_points_zarr.storage.value_major_reader import (
    _ValueMajorLevelReader,
)


@pytest.fixture
def value_major_root(tmp_path: Path):
    values = np.arange(20, dtype=np.float32).reshape(10, 2)
    with LocalStore(tmp_path, read_only=False) as store:
        root = zarr.open_group(store=store, mode="w", zarr_format=3)
        group = root.create_group("value_major/level_0")
        group.create_array(
            "location",
            data=values,
            chunks=(2, 2),
            shards=(4, 2),
            **_array_creation_options("zstd-v1"),
        )
        group.create_array(
            "value_point_indptr",
            data=np.array([0, 4, 10], dtype=np.uint64),
            chunks=(3,),
            **_array_creation_options("zstd-v1"),
        )
        yield root


def _open_level(root: zarr.Group) -> _ValueMajorLevelReader:
    return _ValueMajorLevelReader(
        root,
        level=0,
        point_count=10,
        value_count=2,
        metadata=_ValueMajorMetadata(2, 4),
        codec_id="zstd-v1",
    )


@pytest.fixture
def value_major_reader(value_major_root: zarr.Group):
    reader = _open_level(value_major_root)
    try:
        yield reader
    finally:
        reader.close()


def test_value_major_reader_reads_adjacent_and_disjoint_intervals(value_major_reader: _ValueMajorLevelReader) -> None:
    reader = value_major_reader

    adjacent = reader.read_intervals(((1, 3), (3, 4)), expected_row_count=3)
    disjoint = reader.read_intervals(((0, 2), (5, 7)), expected_row_count=4)

    assert adjacent.tolist() == [[2.0, 3.0], [4.0, 5.0], [6.0, 7.0]]
    assert disjoint.tolist() == [[0.0, 1.0], [2.0, 3.0], [10.0, 11.0], [12.0, 13.0]]
    assert adjacent.flags.c_contiguous
    assert disjoint.flags.c_contiguous


def test_value_major_reader_splits_at_shard_row_bound_and_checks_cancellation(
    value_major_reader: _ValueMajorLevelReader,
) -> None:
    reader = value_major_reader
    cancellation_checks = 0

    def check() -> None:
        nonlocal cancellation_checks
        cancellation_checks += 1

    result = reader.read_intervals(
        ((0, 2), (4, 7)),
        expected_row_count=5,
        raise_if_cancelled=check,
    )

    assert result.tolist() == [[0.0, 1.0], [2.0, 3.0], [8.0, 9.0], [10.0, 11.0], [12.0, 13.0]]
    assert cancellation_checks == 4


def test_value_major_reader_propagates_cancellation_between_bounded_reads(
    value_major_reader: _ValueMajorLevelReader,
) -> None:
    reader = value_major_reader
    cancellation_checks = 0

    def raise_on_second_batch() -> None:
        nonlocal cancellation_checks
        cancellation_checks += 1
        if cancellation_checks == 3:
            raise RuntimeError("cancelled")

    with pytest.raises(RuntimeError, match="cancelled"):
        reader.read_intervals(
            ((0, 2), (4, 7)),
            expected_row_count=5,
            raise_if_cancelled=raise_on_second_batch,
        )

    assert cancellation_checks == 3


@pytest.mark.parametrize(
    ("intervals", "expected_row_count", "match"),
    [
        (((0, 2),), 1, "expected_row_count"),
        (((2, 4), (3, 5)), 4, "ordered and nonoverlapping"),
        (((-1, 1),), 2, "inside the location array"),
        (((9, 11),), 2, "inside the location array"),
    ],
)
def test_value_major_reader_rejects_invalid_interval_contracts(
    value_major_reader: _ValueMajorLevelReader,
    intervals: tuple[tuple[int, int], ...],
    expected_row_count: int,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        value_major_reader.read_intervals(
            intervals,
            expected_row_count=expected_row_count,
        )


def test_level_open_and_empty_selection_do_not_read_arrays(
    value_major_root: zarr.Group, monkeypatch: pytest.MonkeyPatch
) -> None:
    def reject_read(*args: object, **kwargs: object) -> object:
        raise AssertionError("Opening a level or reading an empty selection decoded an array.")

    monkeypatch.setattr(zarr.Array, "__getitem__", reject_read)
    monkeypatch.setattr(zarr.Array, "get_orthogonal_selection", reject_read)
    reader = _open_level(value_major_root)
    try:
        result = reader.read_intervals((), expected_row_count=0)
        assert result.shape == (0, 2)
        assert result.dtype == np.float32
    finally:
        reader.close()


def test_pointer_loading_is_explicit_read_only_and_caller_owned(
    value_major_reader: _ValueMajorLevelReader,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_read = zarr.Array.__getitem__
    reads = []

    def tracked_read(array, selection):
        reads.append(array.name)
        return original_read(array, selection)

    monkeypatch.setattr(zarr.Array, "__getitem__", tracked_read)
    pointer = value_major_reader.load_point_indptr()
    assert reads == ["/value_major/level_0/value_point_indptr"]
    assert pointer.tolist() == [0, 4, 10]
    assert pointer.dtype == np.uint64
    assert pointer.flags.c_contiguous and not pointer.flags.writeable
    value_major_reader.close()
    assert pointer.tolist() == [0, 4, 10]
    with pytest.raises(RuntimeError, match="closed"):
        value_major_reader.load_point_indptr()
    with pytest.raises(RuntimeError, match="closed"):
        value_major_reader.read_intervals((), expected_row_count=0)


def test_interval_reads_preserve_exact_selectors_and_batch_bound(
    value_major_reader: _ValueMajorLevelReader, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_read = zarr.Array.get_orthogonal_selection
    selected_rows = []

    def tracked_read(array, selection, **kwargs):
        selected_rows.append((array.name, selection[0]))
        return original_read(array, selection, **kwargs)

    monkeypatch.setattr(zarr.Array, "get_orthogonal_selection", tracked_read)
    value_major_reader.read_intervals(((1, 3), (3, 4)), expected_row_count=3)
    value_major_reader.read_intervals(((0, 2), (5, 7)), expected_row_count=4)
    value_major_reader.read_intervals(((0, 2), (4, 7)), expected_row_count=5)

    assert all(name == "/value_major/level_0/location" for name, _ in selected_rows)
    assert len(selected_rows) == 4
    assert selected_rows[0][1] == slice(1, 4)
    np.testing.assert_array_equal(selected_rows[1][1], [0, 1, 5, 6])
    np.testing.assert_array_equal(selected_rows[2][1], [0, 1, 4, 5])
    assert selected_rows[3][1] == slice(6, 7)


@pytest.mark.parametrize("name", ["location", "value_point_indptr"])
@pytest.mark.parametrize("corruption", ["dtype", "shape", "chunks", "shards", "codec", "attributes", "missing"])
def test_level_reader_rejects_malformed_arrays(value_major_root: zarr.Group, name: str, corruption: str) -> None:
    group = value_major_root["value_major/level_0"]
    array = group[name]
    if corruption == "attributes":
        array.attrs["unexpected"] = True
    else:
        data = array[:]
        chunks, shards = array.chunks, array.shards
        del group[name]
        if corruption != "missing":
            options = _array_creation_options("zstd-v1")
            if corruption == "dtype":
                data = data.astype(np.float64)
            elif corruption == "shape":
                data = data[:-1]
            elif corruption == "chunks":
                chunks = (1, 2) if name == "location" else (1,)
            elif corruption == "shards":
                shards = (8, 2) if name == "location" else (3,)
            elif corruption == "codec":
                options["compressors"] = []
            group.create_array(name, data=data, chunks=chunks, shards=shards, **options)

    with pytest.raises((ValueError, KeyError)):
        _open_level(value_major_root)


@pytest.mark.parametrize("name", ["location", "value_point_indptr"])
def test_level_reader_rejects_missing_payload_chunks(value_major_root: zarr.Group, name: str) -> None:
    # Metadata is intact: opening succeeds, but the explicit read must not
    # silently substitute fill values for a missing data chunk or shard.
    store_root = value_major_root.store.root
    chunks = tuple(path for path in (store_root / "value_major/level_0" / name / "c").rglob("*") if path.is_file())
    assert chunks
    chunks[0].unlink()
    reader = _open_level(value_major_root)
    try:
        with pytest.raises(Exception, match="chunk|Chunk|shard|Shard"):
            if name == "location":
                reader.read_intervals(((0, 10),), expected_row_count=10)
            else:
                reader.load_point_indptr()
    finally:
        reader.close()
