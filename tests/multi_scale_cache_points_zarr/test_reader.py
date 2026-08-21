from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from napari_harpy.core.multi_scale_cache_points_zarr.builder import (
    _build_points_cache_zarr,
    _PointsCacheBuilderConfig,
)
from napari_harpy.core.multi_scale_cache_points_zarr.cache_format import (
    VALUE_TILES_MANIFEST_INDEX,
    VALUE_TILES_N_POINTS,
    _CatalogWriteSettings,
)
from napari_harpy.core.multi_scale_cache_points_zarr.models import _TileDescriptor
from napari_harpy.core.multi_scale_cache_points_zarr.reader import (
    _exact_value_tile_row_selection,
    _IntrinsicViewport,
    _PointsCacheReader,
    _SelectedValueIndex,
    _ValueTileInterval,
)
from napari_harpy.core.multi_scale_cache_points_zarr.sampling import _select_sampled_tile_indices
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
from napari_harpy.core.multi_scale_cache_points_zarr.writer.catalog import _write_staged_cache_catalog

CatalogExactFixture = Any


@dataclass(frozen=True)
class _ReaderFixture:
    cache_root: Path
    dropped_point_ids: tuple[int, int]


class _TrackingPointsCacheReader(_PointsCacheReader):
    def __init__(self, cache_root: Path) -> None:
        super().__init__(cache_root)
        self.value_filtered_levels: list[int] = []

    def _selected_value_manifest_summary(
        self,
        level: int,
        visible_rows: npt.NDArray[np.int64],
        value_index: _SelectedValueIndex,
    ) -> tuple[npt.NDArray[np.uint64], int]:
        self.value_filtered_levels.append(level)
        return super()._selected_value_manifest_summary(level, visible_rows, value_index)


def _load_selected_value_index(
    reader: _PointsCacheReader,
    value_ids: npt.NDArray[np.uint32],
) -> _SelectedValueIndex:
    value_index = reader.load_selected_value_index(value_ids, max_resident_bytes=10_000_000)
    assert value_index is not None
    return value_index


def _load_bucket_lookup_indexes(
    reader: _PointsCacheReader,
    *,
    levels: tuple[int, ...] | None = None,
) -> int:
    return reader.load_bucket_lookup_indexes(
        levels=levels,
        max_resident_bytes=10_000_000,
    )


@pytest.fixture(scope="module")
def reader_fixture(tmp_path_factory: pytest.TempPathFactory) -> _ReaderFixture:
    root = tmp_path_factory.mktemp("acceptance-reader")
    source = ParquetPointsSource(
        spatialdata_path=root / "source.zarr",
        points_name="transcripts",
        columns=PointColumnSelection(x="x", y="y", value="gene"),
    )
    source.parquet_path.mkdir(parents=True)

    dense_count = 5_000
    point_id = np.arange(dense_count, dtype=np.uint64)
    x_dense = np.ascontiguousarray((point_id % 9).astype(np.float32) + np.float32(0.5))
    y_dense = np.ascontiguousarray(((point_id // 9) % 9).astype(np.float32) + np.float32(0.5))
    retained = _select_sampled_tile_indices(
        x_dense,
        y_dense,
        point_id,
        level=1,
        tile_x=0,
        tile_y=0,
        tile_size=10,
        target=4_096,
    )
    dropped = np.setdiff1d(np.arange(dense_count, dtype=np.int64), retained, assume_unique=True)
    assert len(dropped) >= 2
    dropped_point_ids = (int(dropped[0]), int(dropped[1]))
    genes = np.full(dense_count + 2, "B", dtype=object)
    genes[list(dropped_point_ids)] = "A"
    genes[-1] = "C"

    pq.write_table(
        pa.table(
            {
                "x": pa.array(np.concatenate((x_dense.astype(np.float64), [11.0, 11.5])), type=pa.float64()),
                "y": pa.array(np.concatenate((y_dense.astype(np.float64), [1.0, 1.5])), type=pa.float64()),
                "gene": pa.array(genes.tolist(), type=pa.string()),
            }
        ),
        source.parquet_path / "part.0.parquet",
        row_group_size=1_000,
    )
    validated = validate_parquet_points_source(source, max_batch_rows=1_000)
    temporary_root = root / "temporary"
    temporary_root.mkdir()
    cache_root = _build_points_cache_zarr(
        validated,
        output_path=root / "transcripts_vis_zarr",
        temporary_directory_root=temporary_root,
        config=_PointsCacheBuilderConfig(
            leaf_tile_size=10,
            overview_point_budget=100,
            dask_worker_count=2,
            zarr_settings=_ZarrWriteSettings(
                point_chunk_rows=256,
                point_shard_rows=1_024,
                range_chunk_rows=64,
                range_shard_rows=256,
                codec_id="zstd-v1",
            ),
            catalog_settings=_CatalogWriteSettings(
                manifest_chunk_rows=4,
                manifest_shard_rows=8,
                value_tile_chunk_rows=4,
                value_tile_shard_rows=8,
            ),
        ),
    )
    return _ReaderFixture(cache_root=cache_root, dropped_point_ids=dropped_point_ids)


def test_reader_rejects_unpublished_staging_catalog(catalog_exact_fixture: CatalogExactFixture) -> None:
    _write_staged_cache_catalog(
        catalog_exact_fixture.validated,
        catalog_exact_fixture.plan,
        (catalog_exact_fixture.result,),
        staging_root=catalog_exact_fixture.staging_root,
        cache_generation_id="12345678-1234-5678-9234-567812345678",
        settings=_CatalogWriteSettings(2, 4, 2, 4),
    )

    with pytest.raises(ValueError, match="publication_state"):
        with _PointsCacheReader(catalog_exact_fixture.staging_root):
            pass


def test_reader_reads_tiles_and_viewports_in_manifest_order(reader_fixture: _ReaderFixture) -> None:
    full = _IntrinsicViewport(0, 0, 12, 10)
    first_tile = _IntrinsicViewport(0, 0, 10, 10)
    selected_a = np.array([0], dtype=np.uint32)

    with _PointsCacheReader(reader_fixture.cache_root) as reader:
        assert _load_bucket_lookup_indexes(reader) == reader.resident_bucket_lookup_bytes
        assert reader.loaded_bucket_lookup_index_count == reader.open_bucket_reader_count > 0
        assert reader.value_names == ("A", "B", "C")
        exact = reader.read_tile(0, 0, 0)
        assert exact is not None
        assert len(exact.value_id) == 5_000
        assert exact.value_id[:2].tolist() == [0, 0]
        assert not exact.location.flags.writeable
        assert not exact.value_id.flags.writeable

        selected = reader.read_tile(0, 0, 0, value_ids=selected_a)
        assert selected is not None
        assert selected.value_id.tolist() == [0, 0]

        first_view = reader.read_viewport(0, first_tile)
        assert [(tile.tile_x, tile.tile_y) for tile in first_view.tiles] == [(0, 0)]
        assert sum(len(tile.value_id) for tile in first_view.tiles) == 5_000

        full_view = reader.read_viewport(0, full)
        assert [(tile.tile_x, tile.tile_y) for tile in full_view.tiles] == [(0, 0), (1, 0)]
        assert sum(len(tile.value_id) for tile in full_view.tiles) == 5_002

        value_index_a = _load_selected_value_index(reader, selected_a)
        filtered_view = reader.read_viewport(0, full, value_index=value_index_a)
        assert [(tile.tile_x, tile.tile_y) for tile in filtered_view.tiles] == [(0, 0)]
        assert filtered_view.tiles[0].value_id.tolist() == [0, 0]

        origin_x = 0.0 + filtered_view.tiles[0].tile_x * filtered_view.tiles[0].tile_size
        intrinsic_x = origin_x + filtered_view.tiles[0].location[:, 0]
        expected_x = np.array(
            [reader_fixture.dropped_point_ids[0] % 9 + 0.5, reader_fixture.dropped_point_ids[1] % 9 + 0.5],
            dtype=np.float32,
        )
        assert intrinsic_x.tolist() == expected_x.tolist()


def test_singleton_and_viewport_reads_share_the_plural_bucket_path(
    reader_fixture: _ReaderFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    full = _IntrinsicViewport(0, 0, 12, 10)
    with _PointsCacheReader(reader_fixture.cache_root) as reader:
        _load_bucket_lookup_indexes(reader, levels=(0,))
        bucket_reader = reader._bucket_cache_or_raise().get(level=0, bucket_id=0)
        original = bucket_reader.read_display_payloads
        calls: list[tuple[int, ...]] = []

        def tracked_batch(
            requests: tuple[tuple[_TileDescriptor, npt.NDArray[np.uint32] | None], ...],
        ) -> tuple[_PointDisplayPayload | None, ...]:
            calls.append(tuple(descriptor.bucket_tile_index for descriptor, _ in requests))
            return original(requests)

        monkeypatch.setattr(bucket_reader, "read_display_payloads", tracked_batch)
        assert reader.read_tile(0, 0, 0) is not None
        assert calls == [(0,)]

        calls.clear()
        result = reader.read_viewport(0, full)
        assert [(tile.tile_x, tile.tile_y) for tile in result.tiles] == [(0, 0), (1, 0)]
        assert calls == [(0, 1)]


def test_value_tile_index_prunes_gene_lost_during_sampling(reader_fixture: _ReaderFixture) -> None:
    selected_a = np.array([0], dtype=np.uint32)
    viewport = _IntrinsicViewport(0, 0, 12, 10)

    with _PointsCacheReader(reader_fixture.cache_root) as reader:
        value_index_a = _load_selected_value_index(reader, selected_a)
        assert reader.open_bucket_reader_count == 0
        result = reader.read_viewport(1, viewport, value_index=value_index_a)
        assert result.tiles == ()
        assert reader.open_bucket_reader_count == 0


def test_reader_cache_retains_bucket_metadata_across_levels(reader_fixture: _ReaderFixture) -> None:
    with _PointsCacheReader(reader_fixture.cache_root) as reader:
        assert reader.resident_index_bytes > 0
        assert reader.resident_bucket_lookup_bytes == 0
        _load_bucket_lookup_indexes(reader, levels=(0,))
        assert reader.loaded_bucket_lookup_index_count == 1
        assert reader.read_tile(0, 0, 0) is not None
        assert reader.open_bucket_reader_count == 1

        assert reader.read_tile(0, 0, 0) is not None
        assert reader.open_bucket_reader_count == 1

        _load_bucket_lookup_indexes(reader, levels=(1, 2))
        assert reader.read_tile(1, 0, 0) is not None
        assert reader.read_tile(2, 0, 0) is not None
        assert reader.open_bucket_reader_count == 3
        assert reader.loaded_bucket_lookup_index_count == 3


def test_bucket_lookup_priming_is_explicit_immutable_and_byte_accounted(
    reader_fixture: _ReaderFixture,
) -> None:
    progress: list[tuple[int, int]] = []
    with _PointsCacheReader(reader_fixture.cache_root) as reader:
        projected = reader.project_bucket_lookup_index_bytes(bucket_keys=((0, 0),))
        assert projected > 0
        assert reader.open_bucket_reader_count == 1
        assert reader.loaded_bucket_lookup_index_count == 0
        with pytest.raises(RuntimeError, match="prime it before display reads"):
            reader.read_tile(0, 0, 0)

        resident = reader.load_bucket_lookup_indexes(
            bucket_keys=((0, 0),),
            max_resident_bytes=projected,
            progress=lambda completed, total: progress.append((completed, total)),
        )
        assert resident == projected == reader.resident_bucket_lookup_bytes
        assert reader.loaded_bucket_lookup_index_count == 1
        assert progress == [(1, 1)]

        bucket_reader = reader._bucket_cache_or_raise().get(level=0, bucket_id=0)
        lookup = bucket_reader._lookup_index_or_raise()
        assert all(
            array.flags.c_contiguous and not array.flags.writeable
            for array in (
                lookup.tile_offset,
                lookup.tile_indptr,
                lookup.range_value_id,
                lookup.range_row_start,
                lookup.range_row_count,
            )
        )
        assert lookup.resident_bytes == projected


def test_bucket_lookup_budget_fails_before_lookup_arrays_are_loaded(
    reader_fixture: _ReaderFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    load_calls = 0
    original = _BucketReader.load_lookup_index

    def counted_load(self: _BucketReader, *args: object, **kwargs: object) -> object:
        nonlocal load_calls
        load_calls += 1
        return original(self, *args, **kwargs)  # type: ignore[arg-type]

    with _PointsCacheReader(reader_fixture.cache_root) as reader:
        projected = reader.project_bucket_lookup_index_bytes(levels=(0,))
        monkeypatch.setattr(_BucketReader, "load_lookup_index", counted_load)
        with pytest.raises(ValueError, match="resident bytes"):
            reader.load_bucket_lookup_indexes(
                levels=(0,),
                max_resident_bytes=projected - 1,
            )
        assert load_calls == 0
        assert reader.loaded_bucket_lookup_index_count == 0
        assert reader.resident_bucket_lookup_bytes == 0


def test_bucket_lookup_priming_rolls_back_new_indexes_after_failure(
    reader_fixture: _ReaderFixture,
) -> None:
    with _PointsCacheReader(reader_fixture.cache_root) as reader:
        projected = reader.project_bucket_lookup_index_bytes(levels=(0,))

        def fail_progress(completed: int, total: int) -> None:
            assert (completed, total) == (1, 1)
            raise RuntimeError("injected progress failure")

        with pytest.raises(RuntimeError, match="injected progress failure"):
            reader.load_bucket_lookup_indexes(
                levels=(0,),
                max_resident_bytes=projected,
                progress=fail_progress,
            )
        assert reader.loaded_bucket_lookup_index_count == 0
        assert reader.resident_bucket_lookup_bytes == 0


def test_primed_display_reads_do_not_reread_bucket_lookup_arrays(
    reader_fixture: _ReaderFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lookup_names = {
        "tile_x",
        "tile_y",
        "tile_offset",
        "ranges/tile_indptr",
        "ranges/value_id",
        "ranges/row_start",
        "ranges/row_count",
    }
    with _PointsCacheReader(reader_fixture.cache_root) as reader:
        _load_bucket_lookup_indexes(reader, levels=(0,))
        bucket_reader = reader._bucket_cache_or_raise().get(level=0, bucket_id=0)
        original_array = bucket_reader._array

        def reject_lookup_array(name: str) -> object:
            if name in lookup_names:
                raise AssertionError(f"Display request reread resident lookup array: {name}.")
            return original_array(name)

        monkeypatch.setattr(bucket_reader, "_array", reject_lookup_array)
        complete = reader.read_tile(0, 0, 0)
        selected = reader.read_tile(0, 0, 0, value_ids=np.array([0], dtype=np.uint32))

    assert complete is not None and len(complete.value_id) == 5_000
    assert selected is not None and selected.value_id.tolist() == [0, 0]


def test_bucket_lookup_priming_reads_only_resident_lookup_arrays(
    reader_fixture: _ReaderFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_names = (
        "tile_offset",
        "ranges/tile_indptr",
        "ranges/value_id",
        "ranges/row_start",
        "ranges/row_count",
    )
    with _PointsCacheReader(reader_fixture.cache_root) as reader:
        projected = reader.project_bucket_lookup_index_bytes(levels=(0,))
        bucket_reader = reader._bucket_cache_or_raise().get(level=0, bucket_id=0)
        original_array = bucket_reader._array
        observed_names: list[str] = []

        def record_lookup_array(name: str) -> object:
            observed_names.append(name)
            if name not in expected_names:
                raise AssertionError(f"Lookup priming read an unrelated array: {name}.")
            return original_array(name)

        monkeypatch.setattr(bucket_reader, "_array", record_lookup_array)
        assert (
            reader.load_bucket_lookup_indexes(
                levels=(0,),
                max_resident_bytes=projected,
            )
            == projected
        )
        assert tuple(observed_names) == expected_names


def test_level_selection_uses_budget_even_when_values_disappear(reader_fixture: _ReaderFixture) -> None:
    full = _IntrinsicViewport(0, 0, 12, 10)
    second_exact_tile = _IntrinsicViewport(10, 0, 12, 10)
    selected_a = np.array([0], dtype=np.uint32)
    selected_b = np.array([1], dtype=np.uint32)
    selected_both = np.array([0, 1], dtype=np.uint32)

    with _PointsCacheReader(reader_fixture.cache_root) as reader:
        value_index_a = _load_selected_value_index(reader, selected_a)
        value_index_b = _load_selected_value_index(reader, selected_b)
        value_index_both = _load_selected_value_index(reader, selected_both)
        exact = reader.select_level(full, 6_000)
        assert (exact.level, exact.estimated_point_count, exact.within_budget) == (0, 5_002, True)
        assert exact.omitted_value_ids is None

        overview = reader.select_level(full, 100)
        assert (overview.level, overview.estimated_point_count, overview.within_budget) == (2, 100, True)
        over_budget_overview = reader.select_level(full, 50)
        assert (over_budget_overview.level, over_budget_overview.estimated_point_count) == (2, 100)
        assert not over_budget_overview.within_budget

        exact_a = reader.select_level(full, 2, value_index=value_index_a)
        assert (exact_a.level, exact_a.estimated_point_count, exact_a.within_budget) == (0, 2, True)
        assert exact_a.omitted_value_ids is not None
        assert exact_a.omitted_value_ids.tolist() == []
        assert not exact_a.omitted_value_ids.flags.writeable
        empty_sampled_a = reader.select_level(full, 1, value_index=value_index_a)
        assert (empty_sampled_a.level, empty_sampled_a.estimated_point_count) == (1, 0)
        assert empty_sampled_a.within_budget
        assert empty_sampled_a.omitted_value_ids is not None
        assert empty_sampled_a.omitted_value_ids.tolist() == [0]

        lost_one_of_two = reader.select_level(full, 4_100, value_index=value_index_both)
        assert (lost_one_of_two.level, lost_one_of_two.estimated_point_count) == (1, 4_097)
        assert lost_one_of_two.within_budget
        assert lost_one_of_two.omitted_value_ids is not None
        assert lost_one_of_two.omitted_value_ids.tolist() == [0]

        sampled_b = reader.select_level(full, 100, value_index=value_index_b)
        assert (sampled_b.level, sampled_b.estimated_point_count, sampled_b.within_budget) == (2, 100, True)
        sampled_b_over_budget = reader.select_level(full, 50, value_index=value_index_b)
        assert (sampled_b_over_budget.level, sampled_b_over_budget.estimated_point_count) == (2, 100)
        assert not sampled_b_over_budget.within_budget

        absent_at_exact = reader.select_level(second_exact_tile, 1, value_index=value_index_a)
        assert (absent_at_exact.level, absent_at_exact.estimated_point_count) == (0, 0)
        assert absent_at_exact.within_budget
        assert absent_at_exact.omitted_value_ids is not None
        assert absent_at_exact.omitted_value_ids.tolist() == []


def test_selected_level_selection_stops_after_first_valid_fit(reader_fixture: _ReaderFixture) -> None:
    full = _IntrinsicViewport(0, 0, 12, 10)
    selected_a = np.array([0], dtype=np.uint32)
    selected_b = np.array([1], dtype=np.uint32)

    with _TrackingPointsCacheReader(reader_fixture.cache_root) as reader:
        value_index_a = _load_selected_value_index(reader, selected_a)
        value_index_b = _load_selected_value_index(reader, selected_b)
        assert reader.select_level(full, 2, value_index=value_index_a).level == 0
        assert reader.value_filtered_levels == [0]

        reader.value_filtered_levels.clear()
        assert reader.select_level(full, 100, value_index=value_index_b).level == 2
        assert reader.value_filtered_levels == [0, 1, 2]

        reader.value_filtered_levels.clear()
        empty_sampled = reader.select_level(full, 1, value_index=value_index_a)
        assert (empty_sampled.level, empty_sampled.within_budget) == (1, True)
        assert reader.value_filtered_levels == [0, 1]


def test_exact_value_tile_row_selection_uses_slice_only_for_touching_intervals() -> None:
    contiguous = _exact_value_tile_row_selection(
        (
            _ValueTileInterval(0, 0, 1, 3),
            _ValueTileInterval(1, 1, 3, 5),
        ),
        catalog_row_count=12,
        expected_row_count=4,
    )
    assert contiguous == slice(1, 5)

    disjoint = _exact_value_tile_row_selection(
        (
            _ValueTileInterval(0, 0, 1, 3),
            _ValueTileInterval(2, 2, 7, 10),
        ),
        catalog_row_count=12,
        expected_row_count=5,
    )
    assert isinstance(disjoint, np.ndarray)
    assert disjoint.dtype == np.dtype(np.int64)
    assert disjoint.flags.c_contiguous
    assert disjoint.tolist() == [1, 2, 7, 8, 9]


def test_value_tile_interval_rejects_invalid_fields() -> None:
    with pytest.raises(ValueError, match="selected_value_position"):
        _ValueTileInterval(True, 0, 1, 2)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="value_id"):
        _ValueTileInterval(0, 2**32, 1, 2)
    with pytest.raises(ValueError, match="start"):
        _ValueTileInterval(0, 0, 1.5, 2)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="smaller"):
        _ValueTileInterval(0, 0, 2, 2)


@pytest.mark.parametrize(
    ("intervals", "catalog_row_count", "expected_row_count", "match"),
    [
        ((), 12, 1, "nonempty"),
        ((_ValueTileInterval(0, 0, 1, 3),), 2, 2, "inside"),
        (
            (
                _ValueTileInterval(1, 1, 3, 5),
                _ValueTileInterval(0, 0, 1, 2),
            ),
            12,
            3,
            "selected-value and nonoverlapping",
        ),
        (
            (
                _ValueTileInterval(0, 0, 1, 4),
                _ValueTileInterval(1, 1, 3, 5),
            ),
            12,
            5,
            "selected-value and nonoverlapping",
        ),
        ((_ValueTileInterval(0, 0, 1, 3),), 12, 3, "reconcile"),
    ],
)
def test_exact_value_tile_row_selection_rejects_invalid_intervals(
    intervals: tuple[_ValueTileInterval, ...],
    catalog_row_count: int,
    expected_row_count: int,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        _exact_value_tile_row_selection(
            intervals,
            catalog_row_count=catalog_row_count,
            expected_row_count=expected_row_count,
        )


def test_complete_value_index_load_normalizes_without_catalog_payload_reads(
    reader_fixture: _ReaderFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected_all = np.array([0, 1, 2], dtype=np.uint32)
    calls = {VALUE_TILES_MANIFEST_INDEX: 0, VALUE_TILES_N_POINTS: 0}

    with _PointsCacheReader(reader_fixture.cache_root) as reader:
        catalog = reader._catalog_or_raise()
        original_array = catalog.array

        class _CountingArray:
            def __init__(self, name: str) -> None:
                self._name = name
                self._array = original_array(name)

            def __getitem__(self, selection: object) -> object:
                calls[self._name] += 1
                return self._array[selection]

        def counted_array(name: str) -> object:
            if name in calls:
                return _CountingArray(name)
            return original_array(name)

        monkeypatch.setattr(catalog, "array", counted_array)
        value_index = reader.load_selected_value_index(
            selected_all,
            max_resident_bytes=1,
        )
        assert value_index is None

    # Selecting the complete vocabulary normalizes to the all-values path and
    # deliberately reads no value-tile payload arrays.
    assert calls == {VALUE_TILES_MANIFEST_INDEX: 0, VALUE_TILES_N_POINTS: 0}


def test_selected_value_index_is_immutable_bounded_and_catalog_io_free(
    reader_fixture: _ReaderFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected_a = np.array([0], dtype=np.uint32)
    full = _IntrinsicViewport(0, 0, 12, 10)

    with _PointsCacheReader(reader_fixture.cache_root) as reader:
        _load_bucket_lookup_indexes(reader, levels=(0,))
        value_index = _load_selected_value_index(reader, selected_a)
        selected_a[0] = np.uint32(2)
        assert value_index.value_ids.tolist() == [0]
        assert value_index.resident_bytes == value_index.value_ids.nbytes + sum(
            level.resident_bytes for level in value_index.levels
        )
        assert not value_index.value_ids.flags.writeable
        assert all(
            array.flags.c_contiguous and not array.flags.writeable
            for level in value_index.levels
            for array in (level.value_indptr, level.manifest_index, level.n_points)
        )
        assert np.diff(value_index.levels[1].value_indptr).tolist() == [0]

        catalog = reader._catalog_or_raise()
        original_array = catalog.array

        def reject_value_tile_payload(name: str) -> object:
            if name in (VALUE_TILES_MANIFEST_INDEX, VALUE_TILES_N_POINTS):
                raise AssertionError("Viewport planning reread selected-value catalog payloads.")
            return original_array(name)

        monkeypatch.setattr(catalog, "array", reject_value_tile_payload)
        exact = reader.select_level(full, 2, value_index=value_index)
        assert (exact.level, exact.estimated_point_count) == (0, 2)
        sampled = reader.select_level(full, 1, value_index=value_index)
        assert (sampled.level, sampled.estimated_point_count) == (1, 0)
        result = reader.read_viewport(0, full, value_index=value_index)
        assert sum(len(tile.value_id) for tile in result.tiles) == 2


def test_selected_value_index_preserves_separated_values_and_empty_level_intervals(
    reader_fixture: _ReaderFixture,
) -> None:
    selected_a_and_c = np.array([0, 2], dtype=np.uint32)
    full = _IntrinsicViewport(0, 0, 12, 10)

    with _PointsCacheReader(reader_fixture.cache_root) as reader:
        _load_bucket_lookup_indexes(reader, levels=(0,))
        value_index = _load_selected_value_index(reader, selected_a_and_c)
        assert value_index.levels[0].value_indptr.tolist() == [0, 1, 2]
        assert value_index.levels[0].n_points.tolist() == [2, 1]
        assert np.diff(value_index.levels[1].value_indptr).tolist() == [0, 1]
        result = reader.read_viewport(0, full, value_index=value_index)

    assert sum(len(tile.value_id) for tile in result.tiles) == 3
    assert sorted(np.concatenate(tuple(tile.value_id for tile in result.tiles)).tolist()) == [0, 0, 2]


def test_selected_value_index_uses_one_exact_selection_per_nonempty_level(
    reader_fixture: _ReaderFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected_a_and_c = np.array([0, 2], dtype=np.uint32)
    selections: dict[str, list[slice | npt.NDArray[np.int64]]] = {
        VALUE_TILES_MANIFEST_INDEX: [],
        VALUE_TILES_N_POINTS: [],
    }

    with _PointsCacheReader(reader_fixture.cache_root) as reader:
        pointers = reader._value_tiles_indptr_or_raise()
        selected_indexes = selected_a_and_c.astype(np.int64, copy=False)
        record_counts = pointers[:, selected_indexes + 1] - pointers[:, selected_indexes]
        expected_nonempty_levels = int((record_counts.sum(axis=1, dtype=np.uint64) > 0).sum())
        catalog = reader._catalog_or_raise()
        original_array = catalog.array

        class _TrackingArray:
            def __init__(self, name: str) -> None:
                self._name = name
                self._array = original_array(name)

            def get_orthogonal_selection(self, selection: tuple[object, ...]) -> object:
                assert len(selection) == 1
                row_selection = selection[0]
                assert isinstance(row_selection, (slice, np.ndarray))
                selections[self._name].append(row_selection)
                return self._array.get_orthogonal_selection(selection)

        def tracked_array(name: str) -> object:
            if name in selections:
                return _TrackingArray(name)
            return original_array(name)

        monkeypatch.setattr(catalog, "array", tracked_array)
        value_index = _load_selected_value_index(reader, selected_a_and_c)

    assert len(selections[VALUE_TILES_MANIFEST_INDEX]) == expected_nonempty_levels
    assert len(selections[VALUE_TILES_N_POINTS]) == expected_nonempty_levels
    for manifest_selection, count_selection in zip(
        selections[VALUE_TILES_MANIFEST_INDEX],
        selections[VALUE_TILES_N_POINTS],
        strict=True,
    ):
        if isinstance(manifest_selection, slice):
            assert manifest_selection == count_selection
        else:
            assert isinstance(count_selection, np.ndarray)
            assert np.array_equal(manifest_selection, count_selection)
    assert any(isinstance(selection, np.ndarray) for selection in selections[VALUE_TILES_MANIFEST_INDEX])
    assert value_index.levels[0].value_indptr.tolist() == [0, 1, 2]
    assert value_index.levels[0].n_points.tolist() == [2, 1]


def test_value_index_load_rejects_budget_before_catalog_payload_reads(
    reader_fixture: _ReaderFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected_a = np.array([0], dtype=np.uint32)
    calls = {VALUE_TILES_MANIFEST_INDEX: 0, VALUE_TILES_N_POINTS: 0}

    with _PointsCacheReader(reader_fixture.cache_root) as reader:
        catalog = reader._catalog_or_raise()
        original_array = catalog.array

        def counted_array(name: str) -> object:
            if name in calls:
                calls[name] += 1
            return original_array(name)

        monkeypatch.setattr(catalog, "array", counted_array)
        with pytest.raises(ValueError, match="resident bytes"):
            reader.load_selected_value_index(selected_a, max_resident_bytes=1)

    assert calls == {VALUE_TILES_MANIFEST_INDEX: 0, VALUE_TILES_N_POINTS: 0}


def test_reader_rejects_selected_value_index_from_another_generation(reader_fixture: _ReaderFixture) -> None:
    selected_a = np.array([0], dtype=np.uint32)
    full = _IntrinsicViewport(0, 0, 12, 10)

    with _PointsCacheReader(reader_fixture.cache_root) as reader:
        value_index = _load_selected_value_index(reader, selected_a)
        foreign = replace(value_index, cache_generation_id="12345678-1234-5678-9234-567812345678")
        with pytest.raises(ValueError, match="another cache generation"):
            reader.select_level(full, 100, value_index=foreign)


def test_reader_rejects_invalid_inputs_and_closed_use(reader_fixture: _ReaderFixture) -> None:
    with pytest.raises(ValueError, match="positive width"):
        _IntrinsicViewport(0, 0, 0, 1)

    reader = _PointsCacheReader(reader_fixture.cache_root)
    with pytest.raises(RuntimeError, match="not open"):
        reader.read_tile(0, 0, 0)
    with reader:
        viewport = _IntrinsicViewport(0, 0, 12, 10)
        with pytest.raises(ValueError, match="level"):
            reader.read_viewport(99, viewport)
        with pytest.raises(ValueError, match="tile_x"):
            reader.read_tile(0, 2, 0)
        with pytest.raises(ValueError, match="point_budget"):
            reader.select_level(viewport, 0)
        for invalid in (
            np.array([], dtype=np.uint32),
            np.array([0, 0], dtype=np.uint32),
            np.array([3], dtype=np.uint32),
            np.array([0], dtype=np.uint64),
        ):
            with pytest.raises(ValueError, match="value_ids"):
                reader.load_selected_value_index(invalid, max_resident_bytes=1_000_000)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="max_resident_bytes"):
            reader.load_bucket_lookup_indexes(max_resident_bytes=0)
        with pytest.raises(ValueError, match="mutually exclusive"):
            reader.load_bucket_lookup_indexes(
                levels=(0,),
                bucket_keys=((0, 0),),
                max_resident_bytes=1_000_000,
            )
        with pytest.raises(ValueError, match="sorted unique"):
            reader.load_bucket_lookup_indexes(
                levels=(1, 0),
                max_resident_bytes=1_000_000,
            )
        with pytest.raises(ValueError, match="unknown bucket"):
            reader.load_bucket_lookup_indexes(
                bucket_keys=((0, 99),),
                max_resident_bytes=1_000_000,
            )

    with pytest.raises(RuntimeError, match="not open"):
        reader.read_tile(0, 0, 0)
    with pytest.raises(RuntimeError, match="entered only once"):
        with reader:
            pass
