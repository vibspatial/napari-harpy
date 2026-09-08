"""Verify complete-tile addressing without resident bucket sparse ranges."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pytest

import napari_harpy.core.multi_scale_cache_points_zarr.reader as reader_module
from napari_harpy.core.multi_scale_cache_points_zarr.reader import _IntrinsicViewport, _PointsCacheReader
from napari_harpy.core.multi_scale_cache_points_zarr.storage._schema import (
    MANIFEST_BUCKET_ID,
    MANIFEST_BUCKET_TILE_INDEX,
    MANIFEST_N_POINTS,
    TILE_MAJOR_TILE_OFFSET,
    TILE_MAJOR_TILE_X,
    TILE_MAJOR_TILE_Y,
)
from napari_harpy.core.multi_scale_cache_points_zarr.storage.bucket_reader import _BucketReader
from napari_harpy.core.multi_scale_cache_points_zarr.storage.bucket_validation import _validate_bucket
from napari_harpy.core.multi_scale_cache_points_zarr.storage.catalog_reader import _CatalogReader
from napari_harpy.core.multi_scale_cache_points_zarr.writer.staging_validation import _read_manifest_inventory


@pytest.fixture
def forbid_sparse_ranges(monkeypatch: pytest.MonkeyPatch) -> None:
    def reject(*args: object, **kwargs: object) -> object:
        raise AssertionError("A viewer operation accessed sparse lookup metadata.")

    original_array = _BucketReader._array

    def guarded_array(self: _BucketReader, name: str) -> Any:
        if name.startswith("ranges/"):
            return reject()
        return original_array(self, name)

    monkeypatch.setattr(_PointsCacheReader, "project_bucket_lookup_index_bytes", reject)
    monkeypatch.setattr(_PointsCacheReader, "load_bucket_lookup_indexes", reject)
    monkeypatch.setattr(_BucketReader, "load_lookup_index", reject)
    monkeypatch.setattr(_BucketReader, "resolve_selected_tile_intervals", reject)
    monkeypatch.setattr(_BucketReader, "_array", guarded_array)


@pytest.mark.usefixtures("forbid_sparse_ranges")
def test_repeated_level_and_selection_changes_never_load_sparse_ranges(reader_fixture: Any) -> None:
    full = _IntrinsicViewport(0, 0, 12, 10)
    partial = _IntrinsicViewport(10, 0, 12, 10)
    with _PointsCacheReader(reader_fixture.cache_root) as reader:
        assert reader.open_bucket_reader_count == 0
        selected = reader.load_selected_value_index(np.array([0, 1], dtype=np.uint32), max_resident_bytes=None)
        assert reader.open_bucket_reader_count == 0
        for level in (*range(reader.level_count), *reversed(range(reader.level_count))):
            for viewport in (full, partial):
                for value_index in (selected, None):
                    plan = reader.plan_viewport(level, viewport, value_index=value_index)
                    # Skip earlier tiles to exercise nonzero bucket row offsets.
                    keys = plan.tile_keys[-1:]
                    result = reader.read_planned_tiles(plan, keys)
                    assert tuple((tile.level, tile.tile_x, tile.tile_y) for tile in result.tiles) == keys
                    for tile in result.tiles:
                        descriptor = reader._descriptors[
                            reader._manifest_row_by_tile[(level, tile.tile_x, tile.tile_y)]
                        ]
                        bucket = reader._bucket_cache_or_raise().get(level=level, bucket_id=descriptor.bucket_id)
                        reference = bucket.read_construction_payload(descriptor)
                        mask = (
                            np.ones(len(reference.value_id), dtype=bool)
                            if value_index is None
                            else np.isin(reference.value_id, value_index.value_ids)
                        )
                        np.testing.assert_array_equal(
                            tile.location, np.column_stack((reference.x_rel[mask], reference.y_rel[mask]))
                        )
                        np.testing.assert_array_equal(tile.value_id, reference.value_id[mask])
                    assert reader.loaded_bucket_lookup_index_count == reader.resident_bucket_lookup_bytes == 0


def test_complete_tile_descriptors_are_validated_once_reused_and_released(reader_fixture: Any, monkeypatch) -> None:
    with _PointsCacheReader(reader_fixture.cache_root) as reader:
        descriptors = reader._descriptors_by_bucket[(0, 0)]
        assert [tile.bucket_row_start for tile in descriptors] == [0, 5_000]
        original_array = _BucketReader._array
        address_reads = []

        def tracked_array(self: _BucketReader, name: str) -> Any:
            if name in {TILE_MAJOR_TILE_OFFSET, TILE_MAJOR_TILE_X, TILE_MAJOR_TILE_Y}:
                address_reads.append(name)
            return original_array(self, name)

        monkeypatch.setattr(_BucketReader, "_array", tracked_array)
        for _ in range(3):
            result = reader.read_tile(0, 1, 0)
            assert result is not None
            np.testing.assert_array_equal(result.location, [[1, 1], [1.5, 1.5]])
        assert address_reads == [TILE_MAJOR_TILE_OFFSET, TILE_MAJOR_TILE_X, TILE_MAJOR_TILE_Y]
        bucket = reader._bucket_cache_or_raise().get(level=0, bucket_id=0)
        assert bucket._tile_descriptors is descriptors
        assert reader.loaded_bucket_lookup_index_count == reader.resident_bucket_lookup_bytes == 0
    assert bucket._tile_descriptors is None


@pytest.mark.parametrize("array_name", [TILE_MAJOR_TILE_OFFSET, TILE_MAJOR_TILE_X, TILE_MAJOR_TILE_Y])
def test_bucket_address_mismatch_fails_before_payload_io_without_accepting_descriptors(
    reader_fixture: Any, monkeypatch: pytest.MonkeyPatch, array_name: str
) -> None:
    original_array = _BucketReader._array

    def corrupted_array(self: _BucketReader, name: str) -> Any:
        array = original_array(self, name)
        if name == array_name:
            values = array[:].copy()
            values[1] += 1
            return values
        return array

    def reject_payload(*args: object, **kwargs: object) -> object:
        raise AssertionError("Invalid tile addressing reached payload IO.")

    with _PointsCacheReader(reader_fixture.cache_root) as reader:
        with monkeypatch.context() as patches:
            patches.setattr(_BucketReader, "_array", corrupted_array)
            patches.setattr(_BucketReader, "read_display_payloads", reject_payload)
            with pytest.raises(ValueError, match="disagree with the bucket"):
                reader.read_tile(0, 1, 0)
            bucket = reader._bucket_cache_or_raise().get(level=0, bucket_id=0)
            assert bucket._tile_descriptors is None
        # A failed installation did not publish invalid addressing.
        assert reader.read_tile(0, 1, 0) is not None


@pytest.mark.parametrize("corruption", ["duplicate-tile-index", "out-of-order-tile-index", "row-overflow"])
def test_invalid_manifest_addressing_is_rejected_without_opening_buckets(reader_fixture: Any, monkeypatch, corruption):
    original_read = reader_module._read_only_array

    def corrupted_read(catalog: Any, name: str, **kwargs: Any) -> np.ndarray:
        values = original_read(catalog, name, **kwargs)
        if (
            corruption in {"duplicate-tile-index", "out-of-order-tile-index"} and name == MANIFEST_BUCKET_TILE_INDEX
        ) or (corruption == "row-overflow" and name == MANIFEST_N_POINTS):
            values = values.copy()
            if corruption == "duplicate-tile-index":
                values[1] = values[0]
            elif corruption == "out-of-order-tile-index":
                # Unique but reversed indexes must be rejected, not sorted into place.
                values[:2] = values[[1, 0]]
            else:
                values[0] = np.iinfo(np.int64).max
        return values

    def reject_bucket(*args: object, **kwargs: object) -> object:
        raise AssertionError("Invalid manifest addressing opened a bucket.")

    monkeypatch.setattr(reader_module, "_read_only_array", corrupted_read)
    monkeypatch.setattr(_BucketReader, "__enter__", reject_bucket)
    with pytest.raises(ValueError, match="bucket-local tile indexes|supported row domain"):
        with _PointsCacheReader(reader_fixture.cache_root):
            pass


@pytest.mark.parametrize(
    ("position", "changes"),
    [
        (0, {"bucket_row_start": 1}),
        (1, {"bucket_row_start": 4_999}),
        (1, {"bucket_row_start": 5_001}),
        (1, {"bucket_tile_index": 0}),
        (1, {"bucket_tile_index": 2}),
        (1, {"n_points": 3}),
        (1, {"tile_x": 0}),
        (1, {"tile_x": 2}),
        (1, {"bucket_id": 1}),
        (1, {"level": 1}),
    ],
)
def test_invalid_descriptor_tuple_is_not_accepted(reader_fixture: Any, position: int, changes: dict) -> None:
    with _PointsCacheReader(reader_fixture.cache_root) as reader:
        descriptors = reader._descriptors_by_bucket[(0, 0)]
        candidate = tuple(replace(tile, **changes) if i == position else tile for i, tile in enumerate(descriptors))
        bucket = reader._bucket_cache_or_raise().get(level=0, bucket_id=0)
        with pytest.raises(ValueError):
            bucket.set_tile_descriptors(candidate)
        assert bucket._tile_descriptors is None
        # A rejected candidate leaves installation retryable with valid facts.
        bucket.set_tile_descriptors(descriptors)
        assert bucket.resolve_complete_tile_interval(descriptors[1]) == (5_000, 5_002)


def test_descriptor_installation_rejects_bad_tuple_shape_and_replacement(reader_fixture: Any) -> None:
    with _PointsCacheReader(reader_fixture.cache_root) as reader:
        descriptors = reader._descriptors_by_bucket[(0, 0)]
        bucket = reader._bucket_cache_or_raise().get(level=0, bucket_id=0)
        for candidate in ((), list(descriptors), descriptors[:1], descriptors[::-1], (None, descriptors[1])):
            with pytest.raises(ValueError):
                bucket.set_tile_descriptors(candidate)
            assert bucket._tile_descriptors is None
        bucket.set_tile_descriptors(descriptors)
        with pytest.raises(ValueError, match="cannot be replaced"):
            bucket.set_tile_descriptors(tuple(replace(tile) for tile in descriptors))
        assert bucket._tile_descriptors is descriptors


@pytest.mark.parametrize("changes", [{"bucket_row_start": 4_999}, {"n_points": 1}, {"tile_x": 2}, {"bucket_id": 1}])
def test_mismatched_request_cannot_borrow_accepted_descriptor_validation(reader_fixture: Any, monkeypatch, changes):
    with _PointsCacheReader(reader_fixture.cache_root) as reader:
        bucket = reader._complete_tile_reader(level=0, bucket_id=0)
        descriptor = reader._descriptors_by_bucket[(0, 0)][1]

        def reject_io(*args: object, **kwargs: object) -> None:
            raise AssertionError("An accepted complete-tile address performed pointer IO.")

        monkeypatch.setattr(bucket, "_array", reject_io)
        with pytest.raises(ValueError, match="disagrees|different bucket"):
            bucket.resolve_complete_tile_interval(replace(descriptor, **changes))
        assert bucket.resolve_complete_tile_interval(descriptor) == (5_000, 5_002)
        assert bucket.resolve_complete_tile_interval(replace(descriptor)) == (5_000, 5_002)


def test_manifest_and_independent_bucket_validation_produce_identical_addresses(reader_fixture: Any) -> None:
    with _PointsCacheReader(reader_fixture.cache_root) as reader, _CatalogReader(reader_fixture.cache_root) as catalog:
        inventory = _read_manifest_inventory(catalog)
        assert reader.open_bucket_reader_count == 0
        for level in inventory.levels:
            for bucket in level.buckets:
                independent = _validate_bucket(
                    reader_fixture.cache_root, level=bucket.level, bucket_id=bucket.bucket_id
                )
                descriptors = reader._descriptors_by_bucket[(bucket.level, bucket.bucket_id)]
                assert descriptors == bucket.descriptors == independent.tile_descriptors
                assert descriptors[0].bucket_row_start == 0
                for manifest_row, descriptor in zip(bucket.manifest_indexes, descriptors, strict=True):
                    assert reader._descriptors[int(manifest_row)] is descriptor
        assert reader.open_bucket_reader_count == 0


def test_manifest_row_starts_reset_across_levels_with_non_dense_bucket_ids(reader_fixture: Any, monkeypatch) -> None:
    original_read = reader_module._read_only_array

    def remapped_read(catalog: Any, name: str, **kwargs: Any) -> np.ndarray:
        values = original_read(catalog, name, **kwargs)
        if name == MANIFEST_BUCKET_ID:
            values = values * np.uint32(7) + np.uint32(11)
            values.flags.writeable = False
        return values

    def reject_bucket(*args: object, **kwargs: object) -> None:
        raise AssertionError("Manifest row-start derivation opened a bucket.")

    monkeypatch.setattr(reader_module, "_read_only_array", remapped_read)
    monkeypatch.setattr(_BucketReader, "__enter__", reject_bucket)
    with _PointsCacheReader(reader_fixture.cache_root) as reader:
        assert reader.level_count > 1
        for (level, bucket_id), descriptors in reader._descriptors_by_bucket.items():
            assert bucket_id >= 11
            row_start = 0
            for bucket_tile_index, descriptor in enumerate(descriptors):
                assert (descriptor.level, descriptor.bucket_id) == (level, bucket_id)
                assert descriptor.bucket_tile_index == bucket_tile_index
                assert descriptor.bucket_row_start == row_start
                row_start += descriptor.n_points
        assert reader.open_bucket_reader_count == 0
