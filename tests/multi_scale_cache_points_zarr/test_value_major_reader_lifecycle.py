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
    with pytest.raises(RuntimeError, match="closed"):
        reader.load_point_indptr()
    with pytest.raises(RuntimeError, match="closed"):
        reader.read_intervals(((0, 1),), expected_row_count=1)


def test_root_opens_shared_level_readers_without_extra_stores_or_array_decoding(
    reader_fixture: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    stores = []
    original_init = LocalStore.__init__

    def tracked_init(store, *args, **kwargs):
        original_init(store, *args, **kwargs)
        stores.append(store)

    def reject_decode(*args, **kwargs):
        raise AssertionError("Root opening decoded an array.")

    monkeypatch.setattr(LocalStore, "__init__", tracked_init)
    monkeypatch.setattr(zarr.Array, "__getitem__", reject_decode)
    monkeypatch.setattr(zarr.Array, "get_orthogonal_selection", reject_decode)
    with _CacheRootReader(reader_fixture.cache_root) as root_reader:
        for level in range(len(root_reader.attributes.levels)):
            reader = root_reader.value_major_level(level)
            assert root_reader.value_major_level(level) is reader
        assert len(stores) == 1
        with pytest.raises(ValueError, match="Unknown"):
            root_reader.array("value_major/level_0/location")


def test_root_invalidates_borrowed_level_readers_before_closing_shared_store(
    reader_fixture: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    borrowed = []
    closed_stores = []
    original_close = LocalStore.close

    def tracked_close(store):
        # Borrowed readers must reject use before their owner's store closes.
        for level_reader in borrowed:
            _assert_closed(level_reader)
        closed_stores.append(store)
        original_close(store)

    monkeypatch.setattr(LocalStore, "close", tracked_close)
    with _CacheRootReader(reader_fixture.cache_root) as root_reader:
        borrowed = [root_reader.value_major_level(level) for level in range(len(root_reader.attributes.levels))]
    assert borrowed
    assert len(closed_stores) == 1
    for reader in borrowed:
        _assert_closed(reader)
    with pytest.raises(RuntimeError, match="not open"):
        root_reader.value_major_level(0)


def test_runtime_loads_each_pointer_once_and_reuses_it_across_viewports(
    reader_fixture: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    pointer_loads = Counter()
    loaded_pointers = []
    original_load = _ValueMajorLevelReader.load_point_indptr

    def tracked_load(reader):
        pointer = original_load(reader)
        pointer_loads[reader] += 1
        loaded_pointers.append(pointer)
        return pointer

    def reject_reconciliation(*args, **kwargs):
        raise AssertionError("Viewer startup ran publication reconciliation.")

    # Observe our pointer-loading boundary; level-reader tests cover its Zarr IO.
    monkeypatch.setattr(_ValueMajorLevelReader, "load_point_indptr", tracked_load)
    monkeypatch.setattr(_CacheRootReader, "validate_contents", reject_reconciliation)
    with _PointsCacheReader(reader_fixture.cache_root) as reader:
        assert len(pointer_loads) == reader.level_count
        assert all(count == 1 for count in pointer_loads.values())
        startup_loads = pointer_loads.copy()
        # Retention must not duplicate the loaded allocations; NumPy views are fine.
        for level, pointer in enumerate(loaded_pointers):
            assert np.shares_memory(reader._value_major_point_indptr[level], pointer)
        index = reader.load_selected_value_index(np.array([0], dtype=np.uint32), max_resident_bytes=None)
        for viewport in (_IntrinsicViewport(0, 0, 12, 10), _IntrinsicViewport(0, 0, 10, 10)):
            for value_index in (None, index):
                reader.read_viewport(0, viewport, value_index=value_index)
        assert pointer_loads == startup_loads
    for level_reader in pointer_loads:
        _assert_closed(level_reader)


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
        pointer = original_load(reader)
        if reader is constructed[1]:
            raise ValueError("injected pointer failure")
        return pointer

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
        with pytest.raises(RuntimeError, match="not open"):
            runtime.read_viewport(0, _IntrinsicViewport(0, 0, 12, 10))
    # A failed reader does not damage the on-disk cache or poison a new owner.
    with _PointsCacheReader(reader_fixture.cache_root) as reader:
        assert reader.level_count > 1
