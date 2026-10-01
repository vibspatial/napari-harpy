from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import dask.dataframe as dd

from napari_harpy.core.multi_scale_cache_points_zarr.source.errors import PointsSourceResolutionError
from napari_harpy.core.multi_scale_cache_points_zarr.source.models import ParquetPointsSource, PointColumnSelection

if TYPE_CHECKING:
    from spatialdata import SpatialData


def resolve_spatialdata_points_source(
    sdata: SpatialData,
    points_name: str,
    *,
    x: str = "x",
    y: str = "y",
    value: str = "gene",
) -> ParquetPointsSource:
    """Resolve a backed local SpatialData points element to its Parquet source."""
    columns = PointColumnSelection(x=x, y=y, value=value)

    if not sdata.is_backed() or sdata.path is None:
        raise PointsSourceResolutionError(
            "SpatialData must be backed by a local zarr store.",
            code="spatialdata_not_backed",
        )
    if not isinstance(sdata.path, Path):
        raise PointsSourceResolutionError(
            "Only local SpatialData zarr stores are supported.",
            code="unsupported_spatialdata_path",
        )

    source = ParquetPointsSource(
        spatialdata_path=sdata.path,
        points_name=points_name,
        columns=columns,
    )

    if points_name not in sdata.points:
        raise PointsSourceResolutionError(
            f"Points element `{points_name}` is not available in the SpatialData object.",
            code="points_element_not_found",
        )

    points = sdata.points[points_name]
    if not isinstance(points, dd.DataFrame):
        raise PointsSourceResolutionError(
            f"Points element `{points_name}` is not represented as a Dask dataframe.",
            code="unsupported_points_element",
        )

    missing_columns = [column for column in (columns.x, columns.y, columns.value) if column not in points.columns]
    if missing_columns:
        missing = ", ".join(f"`{column}`" for column in missing_columns)
        raise PointsSourceResolutionError(
            f"Points element `{points_name}` is missing selected column(s): {missing}.",
            code="missing_point_columns",
        )

    return source
