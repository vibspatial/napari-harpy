"""Test viewport routing and logical tile reconstruction from value-major storage."""

from __future__ import annotations

from dataclasses import fields, replace
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import zarr

from napari_harpy.core.multi_scale_cache_points_zarr.builder import (
    _build_points_cache_zarr,
    _PointsCacheBuilderConfig,
)
from napari_harpy.core.multi_scale_cache_points_zarr.cache_format import (
    _CatalogWriteSettings,
    _ValueMajorWriteSettings,
)
from napari_harpy.core.multi_scale_cache_points_zarr.models import _TileDescriptor
from napari_harpy.core.multi_scale_cache_points_zarr.reader import (
    _IntrinsicViewport,
    _PointsCacheReader,
    _TileReadResult,
    _ViewportReadPlan,
)
from napari_harpy.core.multi_scale_cache_points_zarr.source import (
    ParquetPointsSource,
    PointColumnSelection,
    validate_parquet_points_source,
)
from napari_harpy.core.multi_scale_cache_points_zarr.storage.bucket_reader import (
    _BucketReader,
    _PointDisplayPayload,
)
from napari_harpy.core.multi_scale_cache_points_zarr.storage.models import _ZarrWriteSettings
from napari_harpy.core.multi_scale_cache_points_zarr.storage.value_major_reader import (
    _ValueMajorLocationReader,
)


def _read_filtered_tile_major_reference(
    reader: _PointsCacheReader,
    level: int,
    manifest_rows: tuple[int, ...],
    selected: npt.NDArray[np.uint32],
) -> tuple[_TileReadResult, ...]:
    """Read actual tile-major point arrays in bucket batches, then filter in memory."""
    complete = reader._read_complete_tile_major_requests(level, manifest_rows, raise_if_cancelled=None)
    filtered = []
    for tile in complete.tiles:
        matches = np.isin(tile.value_id, selected)
        if matches.any():
            filtered.append(replace(tile, location=tile.location[matches], value_id=tile.value_id[matches]))
    return tuple(filtered)


def _assert_value_major_read_matches_tile_major(
    reader: _PointsCacheReader,
    plan: _ViewportReadPlan,
    *,
    tile_keys_to_read: tuple[tuple[int, int, int], ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Compare the requested value-major tiles against tile-major reference reads."""
    assert plan.route == "value_major_subset"
    assert plan.requested_value_ids is not None
    selected = np.asarray(plan.requested_value_ids, dtype=np.uint32)
    requested_keys = set(tile_keys_to_read)
    requests = tuple(request for request in plan.requests if request.tile_key in requested_keys)
    expected_keys = tuple(request.tile_key for request in requests)
    # Read independent point-level IDs, never reconstructed selected-value IDs.
    # Keep the reference batched by bucket rather than looping over read_tile().
    with _PointsCacheReader(reader._cache_root) as reference_reader:
        result_tile_major = _read_filtered_tile_major_reference(
            reference_reader,
            plan.level,
            tuple(request.manifest_row for request in requests),
            selected,
        )
    assert all(tile is not None for tile in result_tile_major)
    expected_point_count = sum(len(tile.location) for tile in result_tile_major if tile is not None)

    def reject_bucket_payload(*args: object, **kwargs: object) -> object:
        raise AssertionError("Value-major comparison fell back to tile-major payload reads.")

    original_read = _ValueMajorLocationReader.read_intervals
    selected_row_counts: list[int] = []

    # Count requested rows while preserving the real read: matching output
    # alone would not catch reading extra tiles and filtering them afterward.
    # This does not count additional rows decoded from shared Zarr chunks.
    def tracked_read(
        self: _ValueMajorLocationReader,
        intervals: tuple[tuple[int, int], ...],
        **kwargs: Any,
    ) -> npt.NDArray[np.float32]:
        selected_row_counts.append(sum(stop - start for start, stop in intervals))
        return original_read(self, intervals, **kwargs)

    with monkeypatch.context() as patches:
        patches.setattr(_BucketReader, "read_complete_display_payloads", reject_bucket_payload)
        patches.setattr(_ValueMajorLocationReader, "read_intervals", tracked_read)
        # Input order must not change the logical output's original plan order.
        result_value_major = reader.read_planned_tiles(plan, tuple(reversed(tile_keys_to_read)))

    assert result_value_major.level == plan.level
    assert tuple((tile.level, tile.tile_x, tile.tile_y) for tile in result_value_major.tiles) == expected_keys
    assert selected_row_counts == [expected_point_count]
    assert len(result_value_major.tiles) == len(result_tile_major)
    for value_major_tile, tile_major_tile in zip(result_value_major.tiles, result_tile_major, strict=True):
        assert tile_major_tile is not None
        assert value_major_tile.tile_size == tile_major_tile.tile_size
        assert value_major_tile.location.dtype == tile_major_tile.location.dtype == np.dtype(np.float32)
        assert value_major_tile.value_id.dtype == tile_major_tile.value_id.dtype == np.dtype(np.uint32)
        np.testing.assert_array_equal(value_major_tile.location, tile_major_tile.location)
        np.testing.assert_array_equal(value_major_tile.value_id, tile_major_tile.value_id)
        assert tuple(np.unique(value_major_tile.value_id)) == plan.requested_value_ids


@pytest.fixture(scope="module")
def multi_tile_reader_cache(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("multi-tile-reader")
    source = ParquetPointsSource(
        spatialdata_path=root / "source.zarr",
        points_name="transcripts",
        columns=PointColumnSelection(x="x", y="y", value="gene"),
    )
    source.parquet_path.mkdir(parents=True)

    # These eight leaf tiles merge into four tiles at both Spatial levels 2
    # and 3. Every value occurs in every tile, with different counts and local
    # locations, so skipping an earlier tile must still advance its value's
    # sidecar offset. The dense first tile also exercises Bridge sampling.
    x_parts = []
    y_parts = []
    value_parts = []
    for tile_index, (tile_y, tile_x) in enumerate((y, x) for y in (0, 4) for x in (0, 1, 4, 5)):
        count = 5_000 if tile_index == 0 else 9 + 3 * tile_index
        row = np.arange(count)
        x_parts.append(100.0 + tile_x * 10 + 0.25 + tile_index * 0.03 + row * 0.001)
        y_parts.append(-60.0 + tile_y * 10 + 0.25 + tile_index * 0.03 + (row % 17) * 0.01)
        value_parts.append(np.array(["A", "B", "C"])[row % 3])
    pq.write_table(
        pa.table(
            {
                "x": pa.array(np.concatenate(x_parts), type=pa.float64()),
                "y": pa.array(np.concatenate(y_parts), type=pa.float64()),
                "gene": pa.array(np.concatenate(value_parts), type=pa.string()),
            }
        ),
        source.parquet_path / "part.0.parquet",
    )
    validated = validate_parquet_points_source(source, max_batch_rows=1_000)
    temporary_root = root / "temporary"
    temporary_root.mkdir()
    return _build_points_cache_zarr(
        validated,
        output_path=root / "transcripts_vis_zarr",
        temporary_directory_root=temporary_root,
        config=_PointsCacheBuilderConfig(
            leaf_tile_size=10,
            overview_point_budget=100,
            # Spread logical tiles over several buckets to exercise physical
            # grouping and cooperative cancellation between bucket batches.
            target_points_per_bucket=1_000,
            dask_worker_count=2,
            zarr_settings=_ZarrWriteSettings(256, 1_024, 64, 256, "zstd-v1"),
            catalog_settings=_CatalogWriteSettings(4, 8, 4, 8),
            value_major_settings=_ValueMajorWriteSettings(256, 1_024, 1_024),
        ),
    )


def test_diagnostic_missing_tile_returns_none_without_opening_a_bucket(
    multi_tile_reader_cache: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Leaf column 2 is inside the grid but absent from this cache's manifest.
    with _PointsCacheReader(multi_tile_reader_cache) as reader:

        def reject_open(*args: object, **kwargs: object) -> None:
            raise AssertionError("A missing logical tile opened a physical bucket.")

        monkeypatch.setattr(_BucketReader, "__enter__", reject_open)
        assert reader.read_tile(0, 2, 0) is None
        assert reader.read_tile(0, 2, 0, value_ids=np.array([0], dtype=np.uint32)) is None
        assert reader.open_bucket_reader_count == 0


def test_selected_viewport_plan_retains_only_tile_identities_and_shared_level_index(
    reader_fixture: Any,
) -> None:
    selected_a_and_c = np.array([0, 2], dtype=np.uint32)
    full = _IntrinsicViewport(0, 0, 12, 10)

    with _PointsCacheReader(reader_fixture.cache_root) as reader:
        value_index = reader.load_selected_value_index(selected_a_and_c, max_resident_bytes=10_000_000)
        plan = reader.plan_viewport(0, full, value_index=value_index)
        assert plan.requested_value_ids == (0, 2)
        assert plan.route == "value_major_subset"
        assert plan.selected_value_level_index is value_index.levels[0]
        # These tiles contain different selected values (A and C respectively),
        # but the plan stores only their identities, not that per-tile mapping.
        assert plan.tile_keys == ((0, 0, 0), (0, 1, 0))
        for request in plan.requests:
            assert {field.name for field in fields(request)} == {
                "level",
                "tile_x",
                "tile_y",
                "manifest_row",
                "bucket_id",
            }
            assert all(isinstance(getattr(request, field.name), int) for field in fields(request))
        with pytest.raises(ValueError, match="all-values route"):
            replace(plan, route="tile_major_all_values")
        with pytest.raises(ValueError, match="selected-value level index"):
            replace(plan, selected_value_level_index=None)

        unknown_tile = (0, 99, 0)
        with pytest.raises(ValueError, match="absent from the viewport plan"):
            reader.read_planned_tiles(plan, (unknown_tile,))
        with pytest.raises(ValueError, match="duplicates"):
            reader.read_planned_tiles(plan, (plan.tile_keys[0], plan.tile_keys[0]))
        with pytest.raises(ValueError, match=r"\(level, tile_x, tile_y\)"):
            reader.read_planned_tiles(plan, ((0, 0),))  # type: ignore[arg-type]

        foreign_generation = "12345678-1234-5678-9234-567812345678"
        foreign_plan = replace(
            plan,
            cache_generation_id=foreign_generation,
        )
        with pytest.raises(ValueError, match="another cache generation"):
            reader.read_planned_tiles(foreign_plan, foreign_plan.tile_keys)


def test_nonempty_value_major_plan_without_missing_tiles_skips_physical_addressing(
    reader_fixture: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject_addressing(*args: object, **kwargs: object) -> object:
        raise AssertionError("A fully resident viewport entered physical addressing.")

    with _PointsCacheReader(reader_fixture.cache_root) as reader:
        value_index = reader.load_selected_value_index(np.array([0, 1], dtype=np.uint32), max_resident_bytes=10_000_000)
        plan = reader.plan_viewport(0, _IntrinsicViewport(0, 0, 12, 10), value_index=value_index)
        assert plan.requests
        assert plan.route == "value_major_subset"

        # The plan contains visible tiles, but the caller has retained all of
        # them: no missing keys means no block resolution or payload reads.
        monkeypatch.setattr(reader, "_read_value_major_requests", reject_addressing)
        result = reader.read_planned_tiles(plan, ())
        assert result.level == plan.level
        assert result.tiles == ()


def test_value_major_viewport_cancellation_prevents_payload_io(
    reader_fixture: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_if_cancelled() -> None:
        raise RuntimeError("cancelled viewport read")

    def reject_payload_read(*args: object, **kwargs: object) -> object:
        raise AssertionError("A cancelled viewport read accessed Zarr payload rows.")

    with _PointsCacheReader(reader_fixture.cache_root) as reader:
        value_index = reader.load_selected_value_index(np.array([0, 1], dtype=np.uint32), max_resident_bytes=10_000_000)
        plan = reader.plan_viewport(0, _IntrinsicViewport(0, 0, 12, 10), value_index=value_index)
        assert plan.requests
        assert plan.route == "value_major_subset"

        # Keep the real dispatch and physical reader; guard the Zarr selection
        # so dropping or delaying cancellation until after IO fails this test.
        monkeypatch.setattr(zarr.Array, "get_orthogonal_selection", reject_payload_read)
        with pytest.raises(RuntimeError, match="cancelled viewport read"):
            reader.read_planned_tiles(plan, plan.tile_keys, raise_if_cancelled=raise_if_cancelled)


@pytest.mark.parametrize(
    ("level", "viewport"),
    [(0, _IntrinsicViewport(10, 0, 12, 10)), (1, _IntrinsicViewport(0, 0, 12, 10))],
    ids=["no-visible-match", "value-absent-from-level"],
)
def test_selected_viewport_without_positive_tiles_skips_physical_addressing(
    reader_fixture: Any,
    monkeypatch: pytest.MonkeyPatch,
    level: int,
    viewport: _IntrinsicViewport,
) -> None:
    def reject_addressing(*args: object, **kwargs: object) -> object:
        raise AssertionError("An empty positive-tile union entered physical addressing.")

    with _PointsCacheReader(reader_fixture.cache_root) as reader:
        value_index = reader.load_selected_value_index(np.array([0], dtype=np.uint32), max_resident_bytes=10_000_000)
        monkeypatch.setattr(reader, "_read_value_major_requests", reject_addressing)
        plan = reader.plan_viewport(level, viewport, value_index=value_index)
        assert plan.requests == ()
        assert plan.selected_value_level_index is value_index.levels[level]
        assert reader.read_planned_tiles(plan, plan.tile_keys).tiles == ()


def test_selected_viewport_reads_value_major_sidecar_without_bucket_payload_access(
    reader_fixture: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected_a_and_c = np.array([0, 2], dtype=np.uint32)
    full = _IntrinsicViewport(0, 0, 12, 10)

    def reject_bucket_payload(*args: object, **kwargs: object) -> object:
        raise AssertionError("Proper-subset viewport read accessed a tile-major bucket payload.")

    monkeypatch.setattr(_BucketReader, "read_complete_display_payloads", reject_bucket_payload)
    with _PointsCacheReader(reader_fixture.cache_root) as reader:
        value_index = reader.load_selected_value_index(selected_a_and_c, max_resident_bytes=10_000_000)

        def reject_catalog_array(*args: object, **kwargs: object) -> object:
            raise AssertionError("Viewport read reopened a catalog or sidecar array.")

        monkeypatch.setattr(reader._catalog_or_raise(), "array", reject_catalog_array)
        plan = reader.plan_viewport(0, full, value_index=value_index)

        # Read only the second logical tile. Its value-major address follows
        # rows belonging to earlier values and tiles, so this also proves that
        # sidecar offsets use the complete selected-value level records rather
        # than a prefix computed from the requested tile subset.
        result = reader.read_planned_tiles(plan, (plan.tile_keys[1],))

        assert [(tile.tile_x, tile.tile_y) for tile in result.tiles] == [(1, 0)]
        assert result.tiles[0].value_id.tolist() == [2]
        assert result.tiles[0].location.tolist() == [[1.5, 1.5]]


def test_selected_viewport_sidecar_preserves_manifest_and_value_order_at_every_level(
    reader_fixture: Any,
) -> None:
    selected_b = np.array([1], dtype=np.uint32)
    full = _IntrinsicViewport(0, 0, 12, 10)

    with _PointsCacheReader(reader_fixture.cache_root) as reader:
        value_index = reader.load_selected_value_index(selected_b, max_resident_bytes=10_000_000)
        for level in range(reader.level_count):
            plan = reader.plan_viewport(level, full, value_index=value_index)
            result = reader.read_planned_tiles(plan, tuple(reversed(plan.tile_keys)))

            assert plan.route == "value_major_subset"
            assert plan.selected_value_level_index is value_index.levels[level]
            assert [(tile.tile_y, tile.tile_x) for tile in result.tiles] == sorted(
                (tile.tile_y, tile.tile_x) for tile in result.tiles
            )
            assert all(bool((tile.value_id == np.uint32(1)).all()) for tile in result.tiles)


def test_all_values_viewport_retains_tile_major_route_at_every_level(
    reader_fixture: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    full = _IntrinsicViewport(0, 0, 12, 10)

    def reject_sidecar_read(*args: object, **kwargs: object) -> object:
        raise AssertionError("An all-values viewport read accessed a value-major sidecar.")

    monkeypatch.setattr(_ValueMajorLocationReader, "read_intervals", reject_sidecar_read)
    with _PointsCacheReader(reader_fixture.cache_root) as reader:
        for level in range(reader.level_count):
            plan = reader.plan_viewport(level, full)
            result = reader.read_planned_tiles(plan, plan.tile_keys)

            assert plan.route == "tile_major_all_values"
            assert plan.selected_value_level_index is None
            assert tuple((tile.level, tile.tile_x, tile.tile_y) for tile in result.tiles) == plan.tile_keys
            assert sum(len(tile.location) for tile in result.tiles) == sum(
                reader._descriptors[request.manifest_row].n_points for request in plan.requests
            )


def test_complete_tile_major_offsets_cover_disjoint_tiles_across_levels_and_buckets(
    multi_tile_reader_cache: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Check complete tile-major reads retain correct offsets when tiles are skipped.

    Read all values for alternating tiles across levels and buckets. Skipped
    tiles' counts must still contribute to later row starts in the same bucket.
    Check requested tile order and compare locations and value IDs against
    construction reads that independently use persisted bucket offsets.
    Reads of persisted sparse-range arrays are forbidden.
    """

    original_array = _BucketReader._array

    def guarded_array(self: _BucketReader, name: str):
        if name.startswith("ranges/"):
            raise AssertionError("Complete-tile reads accessed sparse ranges.")
        return original_array(self, name)

    monkeypatch.setattr(_BucketReader, "_array", guarded_array)
    with _PointsCacheReader(multi_tile_reader_cache) as reader:
        for level in range(reader.level_count):
            plan = reader.plan_viewport(level, _IntrinsicViewport(100, -60, 160, -10))
            # Skip alternating tiles, including the first one when possible.
            # Their counts must still contribute to bucket-local row offsets.
            keys = plan.tile_keys[1::2] if len(plan.tile_keys) > 1 else plan.tile_keys
            result = reader.read_planned_tiles(plan, keys)
            assert tuple((tile.level, tile.tile_x, tile.tile_y) for tile in result.tiles) == keys
            for tile in result.tiles:
                descriptor = reader._descriptors[reader._manifest_row_by_tile[(level, tile.tile_x, tile.tile_y)]]
                bucket = reader._bucket_cache_or_raise().get(level=level, bucket_id=descriptor.bucket_id)
                # Construction reads use the persisted offsets independently
                # of the viewer's manifest-derived addressing.
                reference = bucket.read_construction_payload(descriptor)
                np.testing.assert_array_equal(tile.location, np.column_stack((reference.x_rel, reference.y_rel)))
                np.testing.assert_array_equal(tile.value_id, reference.value_id)


def test_complete_tile_major_reads_batch_by_bucket_and_check_cancellation_on_both_sides(
    multi_tile_reader_cache: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    original_read = _BucketReader.read_complete_display_payloads

    def tracked_read(
        self: _BucketReader,
        requests: tuple[_TileDescriptor, ...],
    ) -> tuple[_PointDisplayPayload, ...]:
        assert len({descriptor.bucket_id for descriptor in requests}) == 1
        events.append("read")
        return original_read(self, requests)

    def check_cancelled() -> None:
        events.append("check")

    with _PointsCacheReader(multi_tile_reader_cache) as reader:
        plan = reader.plan_viewport(0, _IntrinsicViewport(100, -60, 160, -10))
        bucket_count = len(plan.required_bucket_keys)
        assert bucket_count > 1
        # Empty hash buckets are omitted: serialized IDs need not be dense.
        assert any(bucket_id >= bucket_count for _, bucket_id in plan.required_bucket_keys)
        monkeypatch.setattr(_BucketReader, "read_complete_display_payloads", tracked_read)
        result = reader.read_planned_tiles(plan, plan.tile_keys, raise_if_cancelled=check_cancelled)
        assert events == ["check", "read", "check"] * bucket_count
        assert tuple((tile.level, tile.tile_x, tile.tile_y) for tile in result.tiles) == plan.tile_keys


@pytest.mark.parametrize("after_batches", [0, 1], ids=["before-first-batch", "after-first-batch"])
def test_complete_tile_major_cancellation_prevents_later_bucket_reads(
    multi_tile_reader_cache: Path,
    monkeypatch: pytest.MonkeyPatch,
    after_batches: int,
) -> None:
    calls = []
    original_read = _BucketReader.read_complete_display_payloads

    def tracked_read(self: _BucketReader, requests: Any) -> Any:
        result = original_read(self, requests)
        calls.append(requests)
        return result

    def raise_if_cancelled() -> None:
        if len(calls) == after_batches:
            raise RuntimeError("cancelled bucket read")

    with _PointsCacheReader(multi_tile_reader_cache) as reader:
        plan = reader.plan_viewport(0, _IntrinsicViewport(100, -60, 160, -10))
        assert len(plan.required_bucket_keys) > 1
        monkeypatch.setattr(_BucketReader, "read_complete_display_payloads", tracked_read)
        with pytest.raises(RuntimeError, match="cancelled bucket read"):
            reader.read_planned_tiles(plan, plan.tile_keys, raise_if_cancelled=raise_if_cancelled)
        assert len(calls) == after_batches


def test_complete_tile_major_truncated_payload_fails_before_returning_tiles(
    reader_fixture: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_array = _BucketReader._array

    class TruncatedArray:
        def __init__(self, array):
            self.array = array

        def get_orthogonal_selection(self, selection):
            return self.array.get_orthogonal_selection(selection)[:-1]

    def truncated_array(self: _BucketReader, name: str):
        array = original_array(self, name)
        return TruncatedArray(array) if name == "location" else array

    with _PointsCacheReader(reader_fixture.cache_root) as reader:
        plan = reader.plan_viewport(1, _IntrinsicViewport(0, 0, 12, 10))
        monkeypatch.setattr(_BucketReader, "_array", truncated_array)
        with pytest.raises(RuntimeError, match="aligned array shapes"):
            reader.read_planned_tiles(plan, ((1, 1, 0),))


@pytest.mark.parametrize("use_subset", [False, True], ids=["tile-major", "value-major"])
def test_viewport_payload_failure_propagates_without_returning_partial_tiles(
    reader_fixture: Any,
    monkeypatch: pytest.MonkeyPatch,
    use_subset: bool,
) -> None:
    def fail_read(*args: object, **kwargs: object) -> object:
        raise OSError("injected payload failure")

    with _PointsCacheReader(reader_fixture.cache_root) as reader:
        value_index = (
            reader.load_selected_value_index(np.array([0, 1], dtype=np.uint32), max_resident_bytes=10_000_000)
            if use_subset
            else None
        )
        plan = reader.plan_viewport(0, _IntrinsicViewport(0, 0, 12, 10), value_index=value_index)
        monkeypatch.setattr(_BucketReader, "read_complete_display_payloads", fail_read)
        monkeypatch.setattr(_ValueMajorLocationReader, "read_intervals", fail_read)
        with pytest.raises(OSError, match="injected payload failure"):
            reader.read_planned_tiles(plan, plan.tile_keys)


def test_value_major_and_tile_major_subset_paths_return_identical_logical_tiles(
    reader_fixture: Any,
) -> None:
    selected_a_and_b = np.array([0, 1], dtype=np.uint32)
    full = _IntrinsicViewport(0, 0, 12, 10)

    with _PointsCacheReader(reader_fixture.cache_root) as reader:
        value_index = reader.load_selected_value_index(selected_a_and_b, max_resident_bytes=10_000_000)
        plan = reader.plan_viewport(0, full, value_index=value_index)
        with _PointsCacheReader(reader_fixture.cache_root) as reference_reader:
            result_tile_major = _read_filtered_tile_major_reference(
                reference_reader,
                plan.level,
                tuple(request.manifest_row for request in plan.requests),
                selected_a_and_b,
            )
        # Selecting A and B but not C gives this plan the `value_major_subset`
        # route, so this call reads value-major locations rather than the
        # tile-major buckets used to build `result_tile_major` above.
        result_value_major = reader.read_planned_tiles(plan, plan.tile_keys).tiles

    assert len(result_value_major) == len(result_tile_major)
    for value_major_tile, tile_major_tile in zip(result_value_major, result_tile_major, strict=True):
        assert tile_major_tile is not None
        assert (
            value_major_tile.level,
            value_major_tile.tile_x,
            value_major_tile.tile_y,
            value_major_tile.tile_size,
        ) == (
            tile_major_tile.level,
            tile_major_tile.tile_x,
            tile_major_tile.tile_y,
            tile_major_tile.tile_size,
        )
        assert np.array_equal(value_major_tile.location, tile_major_tile.location)
        assert np.array_equal(value_major_tile.value_id, tile_major_tile.value_id)


@pytest.mark.parametrize(
    ("level", "level_kind"),
    [(1, "bridge"), (2, "spatial"), (3, "spatial")],
    ids=["bridge", "spatial-2", "spatial-3"],
)
@pytest.mark.parametrize("selected_ids", [(1,), (0, 2)], ids=["one-value", "multiple-values"])
def test_coarser_value_major_full_viewport_matches_tile_major(
    multi_tile_reader_cache: Path,
    monkeypatch: pytest.MonkeyPatch,
    level: int,
    level_kind: str,
    selected_ids: tuple[int, ...],
) -> None:
    selected = np.asarray(selected_ids, dtype=np.uint32)
    full_viewport = _IntrinsicViewport(100, -60, 160, 0)

    with _PointsCacheReader(multi_tile_reader_cache) as reader:
        info = reader.dataset_info
        assert (info.x_origin, info.y_origin) == (100.0, -60.0)
        assert info.levels[level].kind == level_kind
        value_index = reader.load_selected_value_index(selected, max_resident_bytes=10_000_000)
        plan = reader.plan_viewport(level, full_viewport, value_index=value_index)
        assert len(plan.requests) == (8 if level == 1 else 4)

        _assert_value_major_read_matches_tile_major(
            reader, plan, tile_keys_to_read=plan.tile_keys, monkeypatch=monkeypatch
        )


@pytest.mark.parametrize(
    ("level", "level_kind"),
    [(1, "bridge"), (2, "spatial"), (3, "spatial")],
    ids=["bridge", "spatial-2", "spatial-3"],
)
@pytest.mark.parametrize("selected_ids", [(1,), (0, 2)], ids=["one-value", "multiple-values"])
def test_coarser_value_major_partial_viewport_preserves_offsets(
    multi_tile_reader_cache: Path,
    monkeypatch: pytest.MonkeyPatch,
    level: int,
    level_kind: str,
    selected_ids: tuple[int, ...],
) -> None:
    selected = np.asarray(selected_ids, dtype=np.uint32)
    full_viewport = _IntrinsicViewport(100, -60, 160, 0)
    partial_viewport = _IntrinsicViewport(140, -60, 160, 0)

    with _PointsCacheReader(multi_tile_reader_cache) as reader:
        info = reader.dataset_info
        assert (info.x_origin, info.y_origin) == (100.0, -60.0)
        assert info.levels[level].kind == level_kind
        value_index = reader.load_selected_value_index(selected, max_resident_bytes=10_000_000)
        full_plan = reader.plan_viewport(level, full_viewport, value_index=value_index)
        plan = reader.plan_viewport(level, partial_viewport, value_index=value_index)
        assert len(full_plan.requests) == (8 if level == 1 else 4)
        assert len(plan.requests) == len(full_plan.requests) // 2
        assert plan.requests[0].manifest_row > full_plan.requests[0].manifest_row
        assert all(request.tile_x * info.levels[level].tile_size >= 40 for request in plan.requests)

        # Read every visible tile. Earlier tiles outside the viewport must
        # still contribute their point counts to these value-major offsets.
        _assert_value_major_read_matches_tile_major(
            reader, plan, tile_keys_to_read=plan.tile_keys, monkeypatch=monkeypatch
        )


@pytest.mark.parametrize(
    ("level", "level_kind"),
    [(1, "bridge"), (2, "spatial"), (3, "spatial")],
    ids=["bridge", "spatial-2", "spatial-3"],
)
@pytest.mark.parametrize("selected_ids", [(1,), (0, 2)], ids=["one-value", "multiple-values"])
@pytest.mark.parametrize(
    "viewport",
    [_IntrinsicViewport(100, -60, 160, 0), _IntrinsicViewport(140, -60, 160, 0)],
    ids=["full-viewport", "partial-viewport"],
)
def test_coarser_value_major_missing_tile_reads_preserve_offsets(
    multi_tile_reader_cache: Path,
    monkeypatch: pytest.MonkeyPatch,
    level: int,
    level_kind: str,
    selected_ids: tuple[int, ...],
    viewport: _IntrinsicViewport,
) -> None:
    selected = np.asarray(selected_ids, dtype=np.uint32)

    with _PointsCacheReader(multi_tile_reader_cache) as reader:
        info = reader.dataset_info
        assert (info.x_origin, info.y_origin) == (100.0, -60.0)
        assert info.levels[level].kind == level_kind
        value_index = reader.load_selected_value_index(selected, max_resident_bytes=10_000_000)
        plan = reader.plan_viewport(level, viewport, value_index=value_index)

        # Simulate CPU residency by omitting the first and every other planned
        # tile from physical reads. Their counts must still advance later
        # blocks of the same value, including within a partial viewport.
        missing_keys = plan.tile_keys[1::2]
        assert 0 < len(missing_keys) < len(plan.tile_keys)
        assert missing_keys[0] != plan.tile_keys[0]

        _assert_value_major_read_matches_tile_major(
            reader, plan, tile_keys_to_read=missing_keys, monkeypatch=monkeypatch
        )
