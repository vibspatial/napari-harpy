from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from napari_harpy.core.multi_scale_cache_points_zarr.build_plan import (
    _plan_points_cache,
    _PointsCacheBuildPlan,
)
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
