from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from napari_harpy.core.multi_scale_cache_points_zarr.build_plan import (
    _plan_points_cache,
    _PointsCacheBuildPlan,
)
from napari_harpy.core.multi_scale_cache_points_zarr.builder import (
    _build_points_cache_zarr,
    _PointsCacheBuilderConfig,
)
from napari_harpy.core.multi_scale_cache_points_zarr.cache_format import _CatalogWriteSettings
from napari_harpy.core.multi_scale_cache_points_zarr.sampling import _select_sampled_tile_indices
from napari_harpy.core.multi_scale_cache_points_zarr.source import (
    ParquetPointsSource,
    PointColumnSelection,
    validate_parquet_points_source,
)
from napari_harpy.core.multi_scale_cache_points_zarr.source.models import ValidatedPointsSource
from napari_harpy.core.multi_scale_cache_points_zarr.storage.models import (
    _LevelWriteResult,
    _ZarrWriteSettings,
)
from napari_harpy.core.multi_scale_cache_points_zarr.writer.exact import (
    _ExactWriterConfig,
    _write_exact_level,
)


@dataclass(frozen=True)
class CatalogExactFixture:
    validated: ValidatedPointsSource
    plan: _PointsCacheBuildPlan
    result: _LevelWriteResult
    staging_root: Path
    temporary_root: Path
    zarr_settings: _ZarrWriteSettings


@pytest.fixture
def catalog_exact_fixture(tmp_path: Path) -> CatalogExactFixture:
    source = ParquetPointsSource(
        spatialdata_path=tmp_path / "source.zarr",
        points_name="transcripts",
        columns=PointColumnSelection(x="x", y="y", value="gene"),
    )
    source.parquet_path.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "x": pa.array([1.0, 3.0, 2.0, 4.0, 11.0, 12.0], type=pa.float64()),
                "y": pa.array([1.0, 2.0, 3.0, 4.0, 1.0, 2.0], type=pa.float64()),
                "gene": pa.array(["A", "B", "A", "A", "B", "B"]),
            }
        ),
        source.parquet_path / "part.0.parquet",
        row_group_size=2,
    )
    validated = validate_parquet_points_source(source, max_batch_rows=2)
    plan = _plan_points_cache(validated, leaf_tile_size=10, overview_point_budget=10)
    staging_root = tmp_path / "staging"
    temporary_root = tmp_path / "temporary"
    staging_root.mkdir()
    temporary_root.mkdir()
    zarr_settings = _ZarrWriteSettings(
        point_chunk_rows=2,
        point_shard_rows=4,
        range_chunk_rows=2,
        range_shard_rows=4,
        codec_id="zstd-v1",
    )
    result = _write_exact_level(
        validated,
        plan,
        staging_root=staging_root,
        temporary_directory_root=temporary_root,
        config=_ExactWriterConfig(zarr_settings=zarr_settings, dask_worker_count=2),
    )
    return CatalogExactFixture(
        validated=validated,
        plan=plan,
        result=result,
        staging_root=staging_root,
        temporary_root=temporary_root,
        zarr_settings=zarr_settings,
    )


@dataclass(frozen=True)
class _ReaderFixture:
    cache_root: Path
    dropped_point_ids: tuple[int, int]


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
