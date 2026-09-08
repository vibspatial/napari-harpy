"""Test shared-store ownership and runtime residency of value-major readers."""

from collections import Counter
from typing import Any

import numpy as np
import pytest
import zarr
from zarr.storage import LocalStore

import napari_harpy.core.multi_scale_cache_points_zarr.storage.value_major_reader as level_module
from napari_harpy.core.multi_scale_cache_points_zarr.reader import _IntrinsicViewport, _PointsCacheReader
from napari_harpy.core.multi_scale_cache_points_zarr.storage.catalog_reader import _CacheRootReader
from napari_harpy.core.multi_scale_cache_points_zarr.storage.value_major_reader import _ValueMajorLevelReader


def _assert_closed(reader: _ValueMajorLevelReader) -> None:
    assert reader._location is None and reader._point_indptr is None
    with pytest.raises(RuntimeError, match="closed"):
        reader.load_point_indptr()
    with pytest.raises(RuntimeError, match="closed"):
        reader.read_intervals(((0, 1),), expected_row_count=1)


def test_root_shares_one_store_without_decoding_and_invalidates_borrowed_levels(
    reader_fixture: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    stores = []
    borrowed = []
    original_init, original_close = LocalStore.__init__, LocalStore.close

    def tracked_init(store, *args, **kwargs):
        original_init(store, *args, **kwargs)
        stores.append(store)

    def tracked_close(store):
        # Invalidate the borrowed readers before the owner closes their store.
        for level_reader in borrowed:
            _assert_closed(level_reader)
        original_close(store)

    def reject_decode(*args, **kwargs):
        raise AssertionError("Root opening decoded an array.")

    monkeypatch.setattr(LocalStore, "__init__", tracked_init)
    monkeypatch.setattr(LocalStore, "close", tracked_close)
    monkeypatch.setattr(zarr.Array, "__getitem__", reject_decode)
    monkeypatch.setattr(zarr.Array, "get_orthogonal_selection", reject_decode)
    with _CacheRootReader(reader_fixture.cache_root) as root_reader:
        for level in range(len(root_reader.attributes.levels)):
            reader = root_reader.value_major_level(level)
            borrowed.append(reader)
            assert root_reader.value_major_level(level) is reader
            assert reader._location.store is root_reader._store
            assert reader._point_indptr.store is root_reader._store
        assert len(stores) == 1
        assert not any(name.startswith("value_major/") for name in root_reader._arrays)
        with pytest.raises(ValueError, match="Unknown"):
            root_reader.array("value_major/level_0/location")
    assert borrowed
    for reader in borrowed:
        _assert_closed(reader)
    with pytest.raises(RuntimeError, match="not open"):
        root_reader.value_major_level(0)


def test_runtime_loads_each_pointer_once_and_reuses_it_across_viewports(
    reader_fixture: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    pointer_reads = Counter()
    loaded_pointers = []
    original_read = zarr.Array.__getitem__
    original_load = _ValueMajorLevelReader.load_point_indptr

    def tracked_read(array, selection):
        if array.name.endswith("/value_point_indptr"):
            pointer_reads[array.name] += 1
        return original_read(array, selection)

    def tracked_load(reader):
        pointer = original_load(reader)
        loaded_pointers.append(pointer)
        return pointer

    def reject_reconciliation(*args, **kwargs):
        raise AssertionError("Viewer startup ran publication reconciliation.")

    monkeypatch.setattr(zarr.Array, "__getitem__", tracked_read)
    monkeypatch.setattr(_ValueMajorLevelReader, "load_point_indptr", tracked_load)
    monkeypatch.setattr(_CacheRootReader, "validate_contents", reject_reconciliation)
    with _PointsCacheReader(reader_fixture.cache_root) as reader:
        expected_reads = {f"/value_major/level_{level}/value_point_indptr": 1 for level in range(reader.level_count)}
        assert pointer_reads == expected_reads
        assert len(loaded_pointers) == reader.level_count
        for level, pointer in enumerate(loaded_pointers):
            assert reader._value_major_point_indptr[level] is pointer
            assert reader._value_major_readers[level] is reader._cache_root_reader.value_major_level(level)
        assert reader.resident_value_major_pointer_bytes == sum(pointer.nbytes for pointer in loaded_pointers)
        initial_bytes = reader.resident_index_bytes
        index = reader.load_selected_value_index(np.array([0], dtype=np.uint32), max_resident_bytes=None)
        for viewport in (_IntrinsicViewport(0, 0, 12, 10), _IntrinsicViewport(0, 0, 10, 10)):
            for value_index in (None, index):
                reader.read_viewport(0, viewport, value_index=value_index)
        assert pointer_reads == expected_reads
        assert reader.resident_index_bytes == initial_bytes
        borrowed = reader._value_major_readers
    for level_reader in borrowed:
        _assert_closed(level_reader)
    assert reader._value_major_readers == () and reader._value_major_point_indptr == ()


@pytest.mark.parametrize("failure_stage", ["layout", "pointer_load"])
def test_failed_open_closes_shared_store_and_invalidates_constructed_levels(
    reader_fixture: Any, monkeypatch: pytest.MonkeyPatch, failure_stage: str
) -> None:
    constructed = []
    closed_stores = []
    original_init = _ValueMajorLevelReader.__init__
    original_layout = level_module._validate_array_layout
    original_load = _ValueMajorLevelReader.load_point_indptr
    original_close = LocalStore.close

    def tracked_init(reader, *args, **kwargs):
        original_init(reader, *args, **kwargs)
        constructed.append(reader)

    def fail_layout(array, **kwargs):
        if kwargs["name"] == "value_major/level_1/location":
            raise ValueError("injected layout failure")
        return original_layout(array, **kwargs)

    def fail_pointer(reader):
        if reader is constructed[1] and reader._point_indptr is not None:
            raise ValueError("injected pointer failure")
        return original_load(reader)

    def tracked_close(store):
        for reader in constructed:
            _assert_closed(reader)
        closed_stores.append(store)
        original_close(store)

    with monkeypatch.context() as patches:
        patches.setattr(_ValueMajorLevelReader, "__init__", tracked_init)
        patches.setattr(LocalStore, "close", tracked_close)
        if failure_stage == "layout":
            patches.setattr(level_module, "_validate_array_layout", fail_layout)
        else:
            patches.setattr(_ValueMajorLevelReader, "load_point_indptr", fail_pointer)
        runtime = _PointsCacheReader(reader_fixture.cache_root)
        with pytest.raises(ValueError, match="injected"):
            with runtime:
                pytest.fail("Invalid cache was accepted.")
        assert constructed
        assert len(closed_stores) == 1
        assert runtime._cache_root_reader is None
        assert runtime._value_major_readers == () and runtime._value_major_point_indptr == ()
    # A failed reader does not damage the on-disk cache or poison a new owner.
    with _PointsCacheReader(reader_fixture.cache_root) as reader:
        assert reader.level_count > 1
