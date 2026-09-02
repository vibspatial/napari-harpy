from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import napari_harpy.core.multi_scale_cache_points_zarr.writer.value_major as value_major_module
from napari_harpy.core.multi_scale_cache_points_zarr.build_plan import _plan_points_cache
from napari_harpy.core.multi_scale_cache_points_zarr.cache_format import (
    _CatalogWriteSettings,
    _ValueMajorWriteSettings,
)
from napari_harpy.core.multi_scale_cache_points_zarr.source import (
    ParquetPointsSource,
    PointColumnSelection,
    validate_parquet_points_source,
)
from napari_harpy.core.multi_scale_cache_points_zarr.storage.bucket_reader import _BucketReader
from napari_harpy.core.multi_scale_cache_points_zarr.storage.catalog_reader import _CatalogReader
from napari_harpy.core.multi_scale_cache_points_zarr.storage.models import _ZarrWriteSettings
from napari_harpy.core.multi_scale_cache_points_zarr.storage.reader_cache import _BucketReaderCache
from napari_harpy.core.multi_scale_cache_points_zarr.writer.bridge import (
    _BridgeWriterConfig,
    _write_bridge_level,
)
from napari_harpy.core.multi_scale_cache_points_zarr.writer.catalog import _write_staged_cache_catalog
from napari_harpy.core.multi_scale_cache_points_zarr.writer.exact import (
    _ExactWriterConfig,
    _write_exact_level,
)
from napari_harpy.core.multi_scale_cache_points_zarr.writer.spatial import (
    _SpatialWriterConfig,
    _write_spatial_levels,
)
from napari_harpy.core.multi_scale_cache_points_zarr.writer.value_major import (
    _expand_ranges,
    _RangeFragmentBatch,
    _read_fragment_locations,
    _split_range_records_by_points,
)

CatalogExactFixture = Any
_GENERATION_ID = "12345678-1234-5678-9234-567812345678"


def test_fragment_locations_group_source_reads_and_restore_record_order() -> None:
    read_calls: list[tuple[int, list[int]]] = []

    class _FakeLocationReader:
        def __init__(self, bucket_id: int) -> None:
            self._bucket_id = bucket_id

        def read_location_rows(self, rows: np.ndarray) -> np.ndarray:
            read_calls.append((self._bucket_id, rows.tolist()))
            return np.column_stack(
                (
                    np.full(len(rows), self._bucket_id, dtype=np.float32),
                    rows.astype(np.float32),
                )
            )

    class _FakeReaderCache:
        def get(self, *, level: int, bucket_id: int) -> _FakeLocationReader:
            assert level == 2
            return _FakeLocationReader(bucket_id)

    fragments = _RangeFragmentBatch(
        manifest_index=np.array([0, 1, 2, 3], dtype=np.uint64),
        row_start=np.array([5, 7, 2, 1], dtype=np.uint64),
        row_count=np.array([2, 1, 1, 2], dtype=np.uint64),
    )

    locations = _read_fragment_locations(
        fragments,
        level=2,
        manifest_bucket_id=np.array([1, 0, 1, 0], dtype=np.uint32),
        readers=_FakeReaderCache(),  # type: ignore[arg-type]
    )

    # Each bucket is read once in increasing source-row order even though its
    # records are interleaved and reversed in the incoming value-major batch.
    assert len(read_calls) == 2
    assert dict(read_calls) == {0: [1, 2, 7], 1: [2, 5, 6]}
    np.testing.assert_array_equal(
        locations,
        np.array(
            [
                [1, 5],
                [1, 6],
                [0, 7],
                [1, 2],
                [0, 1],
                [0, 2],
            ],
            dtype=np.float32,
        ),
    )
    assert locations.flags.c_contiguous


def test_point_bounded_range_batches_split_inside_oversized_range() -> None:
    batches = tuple(
        _split_range_records_by_points(
            np.array([4, 9], dtype=np.uint64),
            np.array([10, 20], dtype=np.uint64),
            np.array([5, 2], dtype=np.uint64),
            max_points=3,
        )
    )

    assert [batch.manifest_index.tolist() for batch in batches] == [[4], [4, 9], [9]]
    assert [batch.row_start.tolist() for batch in batches] == [[10], [13, 20], [21]]
    assert [batch.row_count.tolist() for batch in batches] == [[3], [2, 1], [1]]
    assert [batch.point_count for batch in batches] == [3, 3, 1]


def test_expand_ranges_preserves_interval_order_and_gaps() -> None:
    rows = _expand_ranges(
        np.array([3, 9, 20], dtype=np.int64),
        np.array([2, 3, 1], dtype=np.int64),
    )

    assert rows.tolist() == [3, 4, 9, 10, 11, 20]
    assert rows.flags.c_contiguous


def test_writer_persists_complete_multilevel_locations_and_empty_value_interval(
    catalog_exact_fixture: CatalogExactFixture,
) -> None:
    fixture = catalog_exact_fixture
    plan = _plan_points_cache(
        fixture.validated,
        leaf_tile_size=10,
        overview_point_budget=2,
    )
    bridge = _write_bridge_level(
        fixture.result,
        plan,
        staging_root=fixture.staging_root,
        config=_BridgeWriterConfig(fixture.zarr_settings),
    )
    spatial = _write_spatial_levels(
        bridge,
        plan,
        staging_root=fixture.staging_root,
        config=_SpatialWriterConfig(fixture.zarr_settings),
    )
    level_results = (fixture.result, bridge, *spatial)
    _write_staged_cache_catalog(
        fixture.validated,
        plan,
        level_results,
        staging_root=fixture.staging_root,
        cache_generation_id=_GENERATION_ID,
        settings=_CatalogWriteSettings(2, 4, 2, 4),
        value_major_settings=_ValueMajorWriteSettings(2, 4, 4),
        temporary_directory_root=fixture.temporary_root,
    )

    expected = (
        (
            [0, 3, 6],
            [[1, 1], [2, 3], [4, 4], [3, 2], [1, 1], [2, 2]],
        ),
        (
            [0, 3, 6],
            [[1, 1], [2, 3], [4, 4], [3, 2], [1, 1], [2, 2]],
        ),
        (
            [0, 0, 2],
            [[3, 2], [12, 2]],
        ),
    )
    assert len(level_results) == len(expected)
    with _CatalogReader(fixture.staging_root) as reader:
        for level, (expected_pointer, expected_locations) in enumerate(expected):
            assert reader.array(f"value_major/level_{level}/value_point_indptr")[:].tolist() == expected_pointer
            assert reader.array(f"value_major/level_{level}/location")[:].tolist() == expected_locations


def test_writer_preserves_output_while_bounded_bucket_readers_are_reopened(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = ParquetPointsSource(
        spatialdata_path=tmp_path / "source.zarr",
        points_name="transcripts",
        columns=PointColumnSelection(x="x", y="y", value="gene"),
    )
    source.parquet_path.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "x": pa.array([1.0, 2.0, 11.0, 12.0], type=pa.float64()),
                "y": pa.array([1.0, 1.0, 1.0, 1.0], type=pa.float64()),
                "gene": pa.array(["A", "B", "A", "B"]),
            }
        ),
        source.parquet_path / "part.0.parquet",
        row_group_size=2,
    )
    validated = validate_parquet_points_source(source, max_batch_rows=2)
    plan = _plan_points_cache(validated, leaf_tile_size=10, overview_point_budget=4)
    staging_root = tmp_path / "staging"
    temporary_root = tmp_path / "temporary"
    staging_root.mkdir()
    temporary_root.mkdir()
    zarr_settings = _ZarrWriteSettings(1, 2, 1, 2, "zstd-v1")
    exact_result = _write_exact_level(
        validated,
        plan,
        staging_root=staging_root,
        temporary_directory_root=temporary_root,
        config=_ExactWriterConfig(
            zarr_settings=zarr_settings,
            dask_worker_count=2,
            target_points_per_bucket=1,
        ),
    )
    buckets_by_tile = tuple(descriptor.bucket_id for descriptor in exact_result.tile_descriptors)
    assert len(buckets_by_tile) == 2
    assert len(set(buckets_by_tile)) == 2

    reader_capacities: list[int] = []
    reader_misses: list[tuple[int, int]] = []

    class _RecordingReaderCache(_BucketReaderCache):
        def __init__(self, cache_root: str | Path, *, max_open_readers: int) -> None:
            super().__init__(cache_root, max_open_readers=max_open_readers)
            reader_capacities.append(max_open_readers)
            self._last_key: tuple[int, int] | None = None

        def get(self, *, level: int, bucket_id: int) -> _BucketReader:
            key = (level, bucket_id)
            # Capacity is fixed at one below, so every key change is a cache
            # miss that evicts the preceding reader. A repeated key later in
            # this sequence therefore proves that its reader was reopened.
            if key != self._last_key:
                reader_misses.append(key)
            self._last_key = key
            return super().get(level=level, bucket_id=bucket_id)

    monkeypatch.setattr(value_major_module, "_BucketReaderCache", _RecordingReaderCache)

    _write_staged_cache_catalog(
        validated,
        plan,
        (exact_result,),
        staging_root=staging_root,
        cache_generation_id=_GENERATION_ID,
        settings=_CatalogWriteSettings(1, 2, 1, 2),
        value_major_settings=_ValueMajorWriteSettings(1, 2, 1),
        max_open_value_major_readers=1,
        temporary_directory_root=temporary_root,
        target_points_per_bucket=1,
    )

    first_bucket, second_bucket = buckets_by_tile
    assert reader_capacities == [1]
    assert reader_misses == [
        (0, first_bucket),
        (0, second_bucket),
        (0, first_bucket),
        (0, second_bucket),
    ]
    with _CatalogReader(staging_root) as reader:
        assert reader.array("value_major/level_0/value_point_indptr")[:].tolist() == [0, 2, 4]
        assert reader.array("value_major/level_0/location")[:].tolist() == [
            [1, 1],
            [1, 1],
            [2, 1],
            [2, 1],
        ]
