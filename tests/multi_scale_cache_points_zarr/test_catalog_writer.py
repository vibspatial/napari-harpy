from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import zarr
from zarr.storage import LocalStore

import napari_harpy.core.multi_scale_cache_points_zarr.writer.catalog as catalog_module
import napari_harpy.core.multi_scale_cache_points_zarr.writer.value_major as value_major_module
from napari_harpy.core.multi_scale_cache_points_zarr.cache_format import (
    PUBLICATION_STATE_STAGING,
    _CatalogWriteSettings,
    _ValueMajorWriteSettings,
)
from napari_harpy.core.multi_scale_cache_points_zarr.storage.catalog_reader import (
    _CatalogReader,
    _iter_bucket_range_batches,
    _RangeRecordBatch,
)
from napari_harpy.core.multi_scale_cache_points_zarr.storage.catalog_writer import _CatalogWriter
from napari_harpy.core.multi_scale_cache_points_zarr.storage.models import _ZarrWriteSettings
from napari_harpy.core.multi_scale_cache_points_zarr.storage.reader_cache import _BucketReaderCache
from napari_harpy.core.multi_scale_cache_points_zarr.writer.catalog import _write_staged_cache_catalog

_GENERATION_ID = "12345678-1234-5678-9234-567812345678"
CatalogExactFixture = Any


def _small_settings(**overrides: int) -> _CatalogWriteSettings:
    values = {
        "manifest_chunk_rows": 2,
        "manifest_shard_rows": 4,
        "value_tile_chunk_rows": 2,
        "value_tile_shard_rows": 4,
    }
    values.update(overrides)
    return _CatalogWriteSettings(**values)


def _zarr_settings() -> _ZarrWriteSettings:
    return _ZarrWriteSettings(2, 4, 2, 4, "zstd-v1")


def test_catalog_coordinator_writes_exact_zarr_hierarchy_and_inverted_index(
    catalog_exact_fixture: CatalogExactFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = catalog_exact_fixture
    reader_capacities: list[int] = []
    reader_cache_type = value_major_module._BucketReaderCache

    def tracked_reader_cache(cache_root: str | Path, *, max_open_readers: int) -> _BucketReaderCache:
        reader_capacities.append(max_open_readers)
        return reader_cache_type(cache_root, max_open_readers=max_open_readers)

    monkeypatch.setattr(value_major_module, "_BucketReaderCache", tracked_reader_cache)

    _write_staged_cache_catalog(
        fixture.validated,
        fixture.plan,
        (fixture.result,),
        staging_root=fixture.staging_root,
        cache_generation_id=_GENERATION_ID,
        settings=_small_settings(),
        value_major_settings=_ValueMajorWriteSettings(2, 4, 4),
        temporary_directory_root=fixture.temporary_root,
    )

    with _CatalogReader(fixture.staging_root) as reader:
        assert reader.attributes.cache_generation_id == _GENERATION_ID
        assert reader.attributes.publication_state == PUBLICATION_STATE_STAGING
        assert reader.attributes.value_names == ("A", "B")
        assert reader.array("values/n_points")[:].tolist() == [3, 3]
        assert reader.array("manifest/level_indptr")[:].tolist() == [0, 2]
        assert reader.array("manifest/bucket_id")[:].tolist() == [0, 0]
        assert reader.array("manifest/bucket_tile_index")[:].tolist() == [0, 1]
        assert reader.array("manifest/tile_x")[:].tolist() == [0, 1]
        assert reader.array("manifest/tile_y")[:].tolist() == [0, 0]
        assert reader.array("manifest/n_points")[:].tolist() == [4, 2]
        assert reader.array("value_tiles/indptr")[:].tolist() == [[0, 1, 3]]
        assert reader.array("value_tiles/manifest_index")[:].tolist() == [0, 0, 1]
        assert reader.array("value_tiles/n_points")[:].tolist() == [3, 1, 2]
        assert reader.array("value_major/level_0/value_point_indptr")[:].tolist() == [0, 3, 6]
        assert reader.array("value_major/level_0/location")[:].tolist() == [
            [1.0, 1.0],
            [2.0, 3.0],
            [4.0, 4.0],
            [3.0, 2.0],
            [1.0, 1.0],
            [2.0, 2.0],
        ]

    assert (fixture.staging_root / "tile_major/level_0/zarr.json").is_file()
    assert (fixture.staging_root / "tile_major/level_0/bucket-000.zarr/zarr.json").is_file()
    assert not list(fixture.staging_root.rglob("*.parquet"))
    assert not list(fixture.temporary_root.iterdir())
    assert reader_capacities == [fixture.result.bucket_count]

    with pytest.raises(FileExistsError):
        _write_staged_cache_catalog(
            fixture.validated,
            fixture.plan,
            (fixture.result,),
            staging_root=fixture.staging_root,
            cache_generation_id=_GENERATION_ID,
            settings=_small_settings(),
            value_major_settings=_ValueMajorWriteSettings(2, 4, 4),
            temporary_directory_root=fixture.temporary_root,
        )


def test_compact_range_iterator_does_not_read_missing_point_payload_shard(
    catalog_exact_fixture: CatalogExactFixture,
) -> None:
    fixture = catalog_exact_fixture
    bucket = fixture.result.buckets[0]
    location_objects = [
        path for path in (fixture.staging_root / bucket.bucket_path / "location" / "c").rglob("*") if path.is_file()
    ]
    assert location_objects
    location_objects[0].unlink()

    batches = list(
        _iter_bucket_range_batches(
            fixture.staging_root,
            bucket,
            np.array([0, 1], dtype=np.uint64),
            batch_rows=1,
            expected_settings=fixture.zarr_settings,
        )
    )

    assert [batch.value_id.tolist() for batch in batches] == [[0], [1], [1]]
    assert [batch.manifest_index.tolist() for batch in batches] == [[0], [0], [1]]
    assert [batch.row_start.tolist() for batch in batches] == [[0], [3], [4]]
    assert [batch.n_points.tolist() for batch in batches] == [[3], [1], [2]]


def test_catalog_coordinator_cleans_construction_index_after_sidecar_failure(
    catalog_exact_fixture: CatalogExactFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = catalog_exact_fixture

    def fail_sidecar(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise RuntimeError("injected value-major failure")

    monkeypatch.setattr(catalog_module, "_write_value_major_sidecars", fail_sidecar)

    with pytest.raises(RuntimeError, match="injected value-major failure"):
        _write_staged_cache_catalog(
            fixture.validated,
            fixture.plan,
            (fixture.result,),
            staging_root=fixture.staging_root,
            cache_generation_id=_GENERATION_ID,
            settings=_small_settings(),
            value_major_settings=_ValueMajorWriteSettings(2, 4, 4),
            temporary_directory_root=fixture.temporary_root,
        )

    assert not list(fixture.temporary_root.iterdir())


def test_levelwise_sort_writes_globally_ordered_value_tiles_in_batches(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    settings = _small_settings()
    records_by_level = (
        ((1, 1, 2), (0, 2, 3), (0, 0, 1)),
        ((2, 4, 5), (0, 3, 4)),
    )
    batches_by_level = tuple(
        tuple(
            _RangeRecordBatch(
                value_id=np.array([value], dtype=np.uint32),
                manifest_index=np.array([manifest], dtype=np.uint64),
                row_start=np.array([0], dtype=np.uint64),
                n_points=np.array([count], dtype=np.uint64),
            )
            for value, manifest, count in records
        )
        for records in records_by_level
    )
    with _CatalogWriter(
        staging,
        level_count=2,
        value_count=3,
        manifest_row_count=5,
        value_tile_row_count=5,
        zarr_settings=_zarr_settings(),
        catalog_settings=settings,
    ) as writer:
        ordered_row_start = np.empty(5, dtype=np.uint64)
        summary = writer.write_value_tiles_by_level(
            batches_by_level,
            level_indptr=np.array([0, 3, 5], dtype=np.uint64),
            expected_level_row_counts=(3, 2),
            value_count=3,
            output_batch_rows=2,
            ordered_row_start=ordered_row_start,
        )

    assert summary.indptr.tolist() == [[0, 2, 3, 3], [3, 4, 4, 5]]
    assert summary.manifest_n_points.tolist() == [1, 2, 3, 4, 5]
    assert summary.level_n_points.tolist() == [6, 9]
    assert summary.level_value_n_points.tolist() == [[4, 2, 0], [4, 0, 5]]
    assert ordered_row_start.tolist() == [0, 0, 0, 0, 0]
    with LocalStore(staging, read_only=True) as store:
        root = zarr.open_group(store=store, mode="r", zarr_format=3, use_consolidated=False)
        assert root["value_tiles/manifest_index"][:].tolist() == [0, 2, 1, 3, 4]
        assert root["value_tiles/n_points"][:].tolist() == [1, 3, 2, 4, 5]
        assert root["value_tiles/indptr"][:].tolist() == [[0, 2, 3, 3], [3, 4, 4, 5]]


def test_levelwise_sort_rejects_duplicate_across_output_batches(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    settings = _small_settings()
    duplicate = _RangeRecordBatch(
        value_id=np.array([0], dtype=np.uint32),
        manifest_index=np.array([0], dtype=np.uint64),
        row_start=np.array([0], dtype=np.uint64),
        n_points=np.array([1], dtype=np.uint64),
    )

    with _CatalogWriter(
        staging,
        level_count=1,
        value_count=1,
        manifest_row_count=1,
        value_tile_row_count=2,
        zarr_settings=_zarr_settings(),
        catalog_settings=settings,
    ) as writer:
        ordered_row_start = np.empty(2, dtype=np.uint64)
        with pytest.raises(ValueError, match="Duplicate"):
            writer.write_value_tiles_by_level(
                ((duplicate, duplicate),),
                level_indptr=np.array([0, 1], dtype=np.uint64),
                expected_level_row_counts=(2,),
                value_count=1,
                output_batch_rows=1,
                ordered_row_start=ordered_row_start,
            )
