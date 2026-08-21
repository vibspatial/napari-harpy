from pathlib import Path

import pytest
from spatialdata import SpatialData

from napari_harpy.core.multi_scale_cache_points_zarr.source import (
    PointsSourceResolutionError,
    resolve_spatialdata_points_source,
)


def test_resolve_backed_spatialdata_points_source(backed_sdata_blobs: SpatialData) -> None:
    source = resolve_spatialdata_points_source(
        backed_sdata_blobs,
        "blobs_points",
        value="genes",
    )

    assert source.spatialdata_path == backed_sdata_blobs.path
    assert source.points_name == "blobs_points"
    assert (source.columns.x, source.columns.y, source.columns.value) == ("x", "y", "genes")
    assert source.parquet_path == Path(backed_sdata_blobs.path) / "points/blobs_points/points.parquet"
    assert source.parquet_path.is_dir()


def test_resolve_rejects_unbacked_spatialdata(sdata_blobs: SpatialData) -> None:
    with pytest.raises(PointsSourceResolutionError, match="must be backed") as error:
        resolve_spatialdata_points_source(sdata_blobs, "blobs_points", value="genes")

    assert error.value.code == "spatialdata_not_backed"


def test_resolve_rejects_missing_points_element(backed_sdata_blobs: SpatialData) -> None:
    with pytest.raises(PointsSourceResolutionError, match="`missing` is not available") as error:
        resolve_spatialdata_points_source(backed_sdata_blobs, "missing", value="genes")

    assert error.value.code == "points_element_not_found"


def test_resolve_rejects_missing_selected_column(backed_sdata_blobs: SpatialData) -> None:
    with pytest.raises(PointsSourceResolutionError, match="`missing_value`") as error:
        resolve_spatialdata_points_source(backed_sdata_blobs, "blobs_points", value="missing_value")

    assert error.value.code == "missing_point_columns"
