from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from napari_harpy.core.multi_scale_cache_points.errors import (
    ParquetMetadataValidationError,
    PointsSourceValidationError,
)


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
    """Describe which resolved local Parquet points source should be validated.

    This stores stable source identity and the requested point columns. It does
    not contain metadata observed from the Parquet files; that belongs to the
    private source inventory constructed during validation.
    """

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


@dataclass(frozen=True)
class ParquetSourceRowGroup:
    """File-metadata-derived row count and compressed size for one row group."""

    row_count: int
    compressed_size_bytes: int

    def __post_init__(self) -> None:
        _require_non_negative_integer(self.row_count, "row-group row count")
        _require_non_negative_integer(self.compressed_size_bytes, "row-group compressed size")


@dataclass(frozen=True)
class ParquetSourceFile:
    """Deterministic metadata description of one physical source file."""

    relative_path: str
    size_bytes: int
    modified_time_ns: int | None
    row_count: int
    row_offset: int
    row_groups: tuple[ParquetSourceRowGroup, ...]

    @property
    def row_group_count(self) -> int:
        """Return the number of physical row groups in the source file."""
        return len(self.row_groups)

    def __post_init__(self) -> None:
        if not isinstance(self.relative_path, str) or self.relative_path == "":
            raise ParquetMetadataValidationError(
                "Source-file relative path must be a non-empty POSIX path.",
                code="invalid_source_file_path",
            )
        relative_path = PurePosixPath(self.relative_path)
        if (
            relative_path.is_absolute()
            or ".." in relative_path.parts
            or relative_path.as_posix() != self.relative_path
        ):
            raise ParquetMetadataValidationError(
                f"Source-file path `{self.relative_path}` is not a normalized dataset-relative POSIX path.",
                code="invalid_source_file_path",
            )

        _require_non_negative_integer(self.size_bytes, "source-file size")
        _require_non_negative_integer(self.row_count, "source-file row count")
        _require_non_negative_integer(self.row_offset, "source-file row offset")
        if self.modified_time_ns is not None and (
            not isinstance(self.modified_time_ns, int) or isinstance(self.modified_time_ns, bool)
        ):
            raise ParquetMetadataValidationError(
                "Source-file modification time must be an integer or None.",
                code="invalid_source_file_modified_time",
            )
        if not isinstance(self.row_groups, tuple) or not self.row_groups:
            raise ParquetMetadataValidationError(
                "A Parquet source file must contain at least one row group.",
                code="source_file_has_no_row_groups",
            )
        if not all(isinstance(row_group, ParquetSourceRowGroup) for row_group in self.row_groups):
            raise ParquetMetadataValidationError(
                "Source-file row groups must be ParquetSourceRowGroup records.",
                code="invalid_source_file_row_groups",
            )
        if sum(row_group.row_count for row_group in self.row_groups) != self.row_count:
            raise ParquetMetadataValidationError(
                "Source-file row-group counts do not match its file-metadata row count.",
                code="source_file_row_count_mismatch",
            )


def _require_non_negative_integer(value: object, label: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ParquetMetadataValidationError(
            f"{label.capitalize()} must be a non-negative integer.",
            code="invalid_parquet_metadata_count",
        )
