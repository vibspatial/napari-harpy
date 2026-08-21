from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import pyarrow as pa

from napari_harpy.core.multi_scale_cache_points_zarr.source.errors import (
    ParquetMetadataValidationError,
    PointContentValidationError,
    PointsSourceValidationError,
)

_VALUE_TABLE_SCHEMA = pa.schema(
    [
        pa.field("value_id", pa.uint32(), nullable=False),
        pa.field("value", pa.string(), nullable=False),
        pa.field("n_points", pa.uint64(), nullable=False),
    ]
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
class PointsBounds:
    """Exact finite coordinate bounds observed during point-content scanning."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float

    def __post_init__(self) -> None:
        bounds = (self.x_min, self.x_max, self.y_min, self.y_max)
        if not all(isinstance(value, float) and math.isfinite(value) for value in bounds):
            raise PointContentValidationError(
                "Point bounds must contain finite floats.",
                code="invalid_coordinate_content",
            )
        if self.x_min > self.x_max or self.y_min > self.y_max:
            raise PointContentValidationError(
                "Point bounds must not contain an inverted coordinate range.",
                code="invalid_coordinate_content",
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
        if relative_path.is_absolute() or ".." in relative_path.parts or relative_path.as_posix() != self.relative_path:
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


@dataclass(frozen=True)
class ValidatedPointsSource:
    """Immutable, build-ready description of a validated Parquet points source.

    Parameters
    ----------
    source
        Resolved canonical Parquet source and selected physical column names.
    files
        Deterministically ordered source files with dataset-relative paths, row
        offsets, and physical row-group metadata.
    selected_schema
        Arrow schema containing only the selected source ``x``, ``y``, and
        ``value`` fields in semantic order.
    row_count
        Total row count from Parquet metadata, reconciled with the content scan.
    bounds
        Finite coordinate bounds observed during the content scan.
    value_table
        Canonical Arrow table of normalized ``value_id``, ``value``, and exact
        source-wide ``n_points`` counts.
    source_signature
        Digest of the versioned source inventory verified across the scan.
    source_signature_method
        Versioned method used to calculate ``source_signature``.
    value_normalization_method
        Versioned method used to normalize and order source values.
    point_id_policy
        Versioned rule for deriving internal ``uint64`` point IDs during cache
        construction.

    Notes
    -----
    Direct construction checks the structural consistency of ``value_table``.
    Only instances returned by ``validate_parquet_points_source()`` additionally
    establish that ``bounds`` and ``value_table`` came from the same successful
    scan of aligned source rows; that provenance cannot be reconstructed from
    these aggregate objects independently.
    """

    source: ParquetPointsSource
    files: tuple[ParquetSourceFile, ...]
    selected_schema: pa.Schema
    row_count: int
    bounds: PointsBounds
    value_table: pa.Table
    source_signature: str
    source_signature_method: str
    value_normalization_method: str
    point_id_policy: str

    def __post_init__(self) -> None:
        if not isinstance(self.value_table, pa.Table) or not self.value_table.schema.equals(
            _VALUE_TABLE_SCHEMA,
            check_metadata=False,
        ):
            raise PointContentValidationError(
                "The canonical value table has an incompatible schema.",
                code="invalid_value_content",
            )
        if self.value_table.num_rows == 0:
            raise PointContentValidationError(
                "The canonical value table must contain at least one value.",
                code="invalid_value_content",
            )

        value_ids = self.value_table["value_id"].combine_chunks()
        values = self.value_table["value"].combine_chunks()
        counts = self.value_table["n_points"].combine_chunks()
        if value_ids.null_count or values.null_count or counts.null_count:
            raise PointContentValidationError(
                "The canonical value table must not contain nulls.",
                code="invalid_value_content",
            )

        expected_ids = pa.array(range(self.value_table.num_rows), type=pa.uint32())
        if not value_ids.equals(expected_ids):
            raise PointContentValidationError(
                "Canonical value IDs must be contiguous and match table row order.",
                code="invalid_value_content",
            )

        labels = values.to_pylist()
        if (
            any(label == "" for label in labels)
            or len(set(labels)) != len(labels)
            or labels != sorted(labels, key=lambda label: label.encode("utf-8"))
        ):
            raise PointContentValidationError(
                "Canonical values must be unique, non-empty, and sorted by UTF-8 bytes.",
                code="invalid_value_content",
            )

        point_counts = counts.to_pylist()
        if any(count <= 0 for count in point_counts):
            raise PointContentValidationError(
                "Canonical value counts must be positive.",
                code="invalid_value_content",
            )
        if sum(point_counts) != self.row_count:
            raise PointContentValidationError(
                "Canonical value counts must sum to the validated source row count.",
                code="invalid_value_content",
            )


def _require_non_negative_integer(value: object, label: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ParquetMetadataValidationError(
            f"{label.capitalize()} must be a non-negative integer.",
            code="invalid_parquet_metadata_count",
        )
