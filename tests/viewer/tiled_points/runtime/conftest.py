from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from napari_harpy.core.multi_scale_cache_points_zarr.builder import (
    _build_points_cache_zarr,
    _PointsCacheBuilderConfig,
)
from napari_harpy.core.multi_scale_cache_points_zarr.cache_format import _CatalogWriteSettings
from napari_harpy.core.multi_scale_cache_points_zarr.source import (
    ParquetPointsSource,
    PointColumnSelection,
    validate_parquet_points_source,
)
from napari_harpy.core.multi_scale_cache_points_zarr.storage.models import _ZarrWriteSettings


@pytest.fixture(scope="session")
def real_cache_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build the shared tiny cache used by real runtime integration tests."""
    root = tmp_path_factory.mktemp("tiled-points-runtime")
    source = ParquetPointsSource(
        spatialdata_path=root / "source.zarr",
        points_name="transcripts",
        columns=PointColumnSelection(x="x", y="y", value="gene"),
    )
    source.parquet_path.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "x": pa.array([1.0, 3.0, 2.0, 11.0], type=pa.float64()),
                "y": pa.array([1.0, 2.0, 3.0, 1.0], type=pa.float64()),
                "gene": pa.array(["A", "B", "A", "B"]),
            }
        ),
        source.parquet_path / "part.0.parquet",
        row_group_size=2,
    )
    validated = validate_parquet_points_source(source, max_batch_rows=2)
    temporary_root = root / "temporary"
    temporary_root.mkdir()
    return _build_points_cache_zarr(
        validated,
        output_path=root / "transcripts_vis_zarr",
        temporary_directory_root=temporary_root,
        config=_PointsCacheBuilderConfig(
            leaf_tile_size=10,
            overview_point_budget=10,
            dask_worker_count=2,
            zarr_settings=_ZarrWriteSettings(2, 4, 2, 4, "zstd-v1"),
            catalog_settings=_CatalogWriteSettings(
                manifest_chunk_rows=2,
                manifest_shard_rows=4,
                value_tile_chunk_rows=2,
                value_tile_shard_rows=4,
            ),
        ),
    )
