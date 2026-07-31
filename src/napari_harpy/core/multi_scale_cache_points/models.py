from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from napari_harpy.core.multi_scale_cache_points.errors import PointsSourceValidationError


@dataclass(frozen=True)
class PointColumnSelection:
    """Exact physical column names selected for point-cache construction."""

    x: str
    y: str
    value: str

    def __post_init__(self) -> None:
        for role, column in (("x", self.x), ("y", self.y), ("value", self.value)):
            if not isinstance(column, str) or column == "":
                raise PointsSourceValidationError(
                    f"Point column `{role}` must be a non-empty string.",
                    code="invalid_point_column",
                )

        if len({self.x, self.y, self.value}) != 3:
            raise PointsSourceValidationError(
                "Point columns `x`, `y`, and `value` must be distinct.",
                code="duplicate_point_column",
            )


@dataclass(frozen=True)
class ParquetPointsSource:
    """Immutable physical description of one resolved local Parquet points source."""

    spatialdata_path: Path
    points_name: str
    columns: PointColumnSelection

    @property
    def element_path(self) -> str:
        """Return the canonical SpatialData-relative path for the points element."""
        return f"points/{self.points_name}"

    @property
    def parquet_path(self) -> Path:
        """Return the canonical physical Parquet dataset path for the points element."""
        return self.spatialdata_path / self.element_path / "points.parquet"

    def __post_init__(self) -> None:
        if not isinstance(self.spatialdata_path, Path):
            raise PointsSourceValidationError(
                "`spatialdata_path` must be a pathlib.Path.",
                code="invalid_spatialdata_path",
            )
        if not isinstance(self.points_name, str) or self.points_name == "" or "/" in self.points_name:
            raise PointsSourceValidationError(
                "`points_name` must be a non-empty string without `/`.",
                code="invalid_points_name",
            )
        if not isinstance(self.columns, PointColumnSelection):
            raise PointsSourceValidationError(
                "`columns` must be a PointColumnSelection.",
                code="invalid_point_column_selection",
            )
