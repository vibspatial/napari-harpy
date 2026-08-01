from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from napari_harpy.core.multi_scale_cache_points.errors import ParquetMetadataValidationError
from napari_harpy.core.multi_scale_cache_points.models import (
    ParquetPointsSource,
    ParquetSourceFile,
    ParquetSourceRowGroup,
)

_MAX_UINT64 = 2**64 - 1
_PARQUET_METADATA_FILES = {"_metadata", "_common_metadata"}


@dataclass(frozen=True)
class _ParquetSourceInventory:
    """Record the Parquet metadata observed for a resolved points source.

    This private validation intermediate combines the stable source description
    with the deterministic file inventory, selected schema, and metadata row
    count. It is a metadata snapshot, not the final content-validated build
    input.
    """

    source: ParquetPointsSource
    files: tuple[ParquetSourceFile, ...]
    selected_schema: pa.Schema
    row_count: int

    def __post_init__(self) -> None:
        if not self.files:
            raise ParquetMetadataValidationError(
                "The Parquet dataset does not contain any source files.",
                code="parquet_dataset_empty",
            )
        if self.row_count <= 0:
            raise ParquetMetadataValidationError(
                "The Parquet points source does not contain any rows.",
                code="parquet_source_has_no_rows",
            )
        if sum(source_file.row_count for source_file in self.files) != self.row_count:
            raise ParquetMetadataValidationError(
                "Source-file row counts do not match the source row count.",
                code="source_row_count_mismatch",
            )

        expected_offset = 0
        for source_file in self.files:
            if source_file.row_offset != expected_offset:
                raise ParquetMetadataValidationError(
                    "Source-file row offsets are not contiguous.",
                    code="invalid_source_file_row_offset",
                )
            expected_offset += source_file.row_count


def _read_parquet_source_inventory(source: ParquetPointsSource) -> _ParquetSourceInventory:
    """Build a deterministic source inventory without decoding Parquet data pages."""
    source_files = _discover_parquet_files(source.parquet_path)
    files: list[ParquetSourceFile] = []
    selected_schema: pa.Schema | None = None
    row_offset = 0

    for relative_path, path in source_files:
        parquet_file = _open_parquet_file(path, relative_path)
        file_schema = _selected_file_schema(
            parquet_file.schema_arrow,
            source,
            relative_path,
        )
        if selected_schema is None:
            selected_schema = file_schema
        else:
            _validate_selected_schema_compatibility(
                selected_schema,
                file_schema,
                source,
                relative_path,
            )

        metadata = parquet_file.metadata
        if metadata is None:
            raise ParquetMetadataValidationError(
                f"Parquet source file `{relative_path}` does not contain file metadata.",
                code="missing_parquet_file_metadata",
            )
        if metadata.num_row_groups == 0:
            raise ParquetMetadataValidationError(
                f"Parquet source file `{relative_path}` does not contain a row group.",
                code="source_file_has_no_row_groups",
            )

        row_groups = tuple(
            _read_row_group(metadata.row_group(index), relative_path, index)
            for index in range(metadata.num_row_groups)
        )
        row_count = metadata.num_rows
        _check_row_total(row_offset, row_count, relative_path)

        try:
            stat = path.stat()
        except OSError as error:
            raise ParquetMetadataValidationError(
                f"Could not inspect Parquet source file `{relative_path}`: {error}.",
                code="parquet_source_file_stat_failed",
            ) from error

        files.append(
            ParquetSourceFile(
                relative_path=relative_path,
                size_bytes=stat.st_size,
                modified_time_ns=stat.st_mtime_ns,
                row_count=row_count,
                row_offset=row_offset,
                row_groups=row_groups,
            )
        )
        row_offset += row_count

    if selected_schema is None:
        raise ParquetMetadataValidationError(
            "The Parquet dataset does not contain any source files.",
            code="parquet_dataset_empty",
        )

    return _ParquetSourceInventory(
        source=source,
        files=tuple(files),
        selected_schema=selected_schema,
        row_count=row_offset,
    )


def _discover_parquet_files(root: Path) -> tuple[tuple[str, Path], ...]:
    if not root.exists():
        raise ParquetMetadataValidationError(
            f"Parquet dataset `{root}` does not exist.",
            code="parquet_dataset_not_found",
        )
    if not root.is_dir():
        raise ParquetMetadataValidationError(
            f"Parquet dataset `{root}` must be a directory.",
            code="parquet_dataset_not_directory",
        )

    try:
        source_files = [
            (path.relative_to(root).as_posix(), path)
            for path in root.rglob("*")
            if path.is_file() and path.name not in _PARQUET_METADATA_FILES and path.name.endswith(".parquet")
        ]
    except OSError as error:
        raise ParquetMetadataValidationError(
            f"Could not enumerate Parquet dataset `{root}`: {error}.",
            code="parquet_dataset_enumeration_failed",
        ) from error

    source_files.sort(key=lambda item: item[0])
    if not source_files:
        raise ParquetMetadataValidationError(
            f"Parquet dataset `{root}` does not contain any source files.",
            code="parquet_dataset_empty",
        )
    return tuple(source_files)


def _open_parquet_file(path: Path, relative_path: str) -> pq.ParquetFile:
    try:
        return pq.ParquetFile(path)
    except Exception as error:
        raise ParquetMetadataValidationError(
            f"Could not read file metadata for Parquet source file `{relative_path}`: {error}.",
            code="unreadable_parquet_file_metadata",
        ) from error


def _selected_file_schema(
    schema: pa.Schema,
    source: ParquetPointsSource,
    relative_path: str,
) -> pa.Schema:
    fields: list[pa.Field] = []
    for role, name in (
        ("x", source.columns.x),
        ("y", source.columns.y),
        ("value", source.columns.value),
    ):
        indices = schema.get_all_field_indices(name)
        if not indices:
            raise ParquetMetadataValidationError(
                f"Parquet source file `{relative_path}` is missing selected {role} column `{name}`.",
                code="missing_selected_column",
            )
        if len(indices) != 1:
            raise ParquetMetadataValidationError(
                f"Parquet source file `{relative_path}` contains duplicate selected column `{name}`.",
                code="duplicate_selected_column",
            )

        field = schema.field(indices[0])
        _validate_selected_field_type(field, role, relative_path)
        fields.append(field)

    return pa.schema(fields)


def _validate_selected_field_type(field: pa.Field, role: str, relative_path: str) -> None:
    if role in {"x", "y"}:
        if pa.types.is_integer(field.type) or pa.types.is_floating(field.type):
            return
        expected = "an integer or floating-point Arrow type"
    elif _is_supported_value_type(field.type):
        return
    else:
        expected = "string, large_string, or an integer-indexed dictionary of strings"

    raise ParquetMetadataValidationError(
        f"Selected {role} column `{field.name}` in source file `{relative_path}` has unsupported type "
        f"`{field.type}`; expected {expected}.",
        code="unsupported_selected_column_type",
    )


def _is_supported_value_type(data_type: pa.DataType) -> bool:
    if pa.types.is_string(data_type) or pa.types.is_large_string(data_type):
        return True
    if not pa.types.is_dictionary(data_type):
        return False
    return pa.types.is_integer(data_type.index_type) and (
        pa.types.is_string(data_type.value_type) or pa.types.is_large_string(data_type.value_type)
    )


def _validate_selected_schema_compatibility(
    expected: pa.Schema,
    actual: pa.Schema,
    source: ParquetPointsSource,
    relative_path: str,
) -> None:
    for role, name in (
        ("x", source.columns.x),
        ("y", source.columns.y),
        ("value", source.columns.value),
    ):
        expected_field = expected.field(name)
        actual_field = actual.field(name)
        if expected_field.type != actual_field.type or expected_field.nullable != actual_field.nullable:
            raise ParquetMetadataValidationError(
                f"Selected {role} column `{name}` in source file `{relative_path}` is incompatible with the "
                "first source file.",
                code="incompatible_selected_schema",
            )


def _read_row_group(
    row_group: pq.RowGroupMetaData,
    relative_path: str,
    row_group_index: int,
) -> ParquetSourceRowGroup:
    compressed_size_bytes = 0
    for column_index in range(row_group.num_columns):
        compressed_size = row_group.column(column_index).total_compressed_size
        if compressed_size < 0:
            raise ParquetMetadataValidationError(
                f"Row group {row_group_index} in source file `{relative_path}` has an invalid compressed size.",
                code="invalid_row_group_compressed_size",
            )
        compressed_size_bytes += compressed_size

    return ParquetSourceRowGroup(
        row_count=row_group.num_rows,
        compressed_size_bytes=compressed_size_bytes,
    )


def _check_row_total(row_offset: int, row_count: int, relative_path: str) -> None:
    if row_count < 0:
        raise ParquetMetadataValidationError(
            f"Parquet source file `{relative_path}` has a negative row count.",
            code="invalid_parquet_metadata_count",
        )
    if row_count > _MAX_UINT64 - row_offset:
        raise ParquetMetadataValidationError(
            f"Parquet row count exceeds the uint64 limit at source file `{relative_path}`.",
            code="parquet_row_count_overflow",
        )
