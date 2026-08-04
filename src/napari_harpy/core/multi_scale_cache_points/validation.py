from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from napari_harpy.core.multi_scale_cache_points.errors import (
    ParquetMetadataValidationError,
    PointContentValidationError,
)
from napari_harpy.core.multi_scale_cache_points.models import (
    ParquetPointsSource,
    ParquetSourceFile,
    ParquetSourceRowGroup,
    PointsBounds,
)

_MAX_UINT64 = 2**64 - 1
_MAX_VALUE_CARDINALITY = 2**32
_PARQUET_METADATA_FILES = {"_metadata", "_common_metadata"}
VALUE_NORMALIZATION_METHOD = "harpy-string-trim-unicode-white-space-case-sensitive-v1"
_UNICODE_WHITE_SPACE = "".join(
    chr(code_point)
    for start, stop in (
        (0x0009, 0x000D),
        (0x0020, 0x0020),
        (0x0085, 0x0085),
        (0x00A0, 0x00A0),
        (0x1680, 0x1680),
        (0x2000, 0x200A),
        (0x2028, 0x2029),
        (0x202F, 0x202F),
        (0x205F, 0x205F),
        (0x3000, 0x3000),
    )
    for code_point in range(start, stop + 1)
)
_VALUE_TABLE_SCHEMA = pa.schema(
    [
        pa.field("value_id", pa.uint32(), nullable=False),
        pa.field("value", pa.string(), nullable=False),
        pa.field("n_points", pa.uint64(), nullable=False),
    ]
)


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


@dataclass(frozen=True)
class _ScannedPointsContent:
    """Compact private result of one successful point-content scan."""

    row_count: int
    bounds: PointsBounds
    value_table: pa.Table


def _scan_points_content(
    inventory: _ParquetSourceInventory,
    *,
    max_batch_rows: int = 1_048_576,
) -> _ScannedPointsContent:
    """Validate and summarize selected point content in one bounded pass."""
    if not isinstance(max_batch_rows, int) or isinstance(max_batch_rows, bool) or max_batch_rows <= 0:
        raise PointContentValidationError(
            "`max_batch_rows` must be a positive integer.",
            code="invalid_max_batch_rows",
        )

    columns = [
        inventory.source.columns.x,
        inventory.source.columns.y,
        inventory.source.columns.value,
    ]
    total_rows = 0
    global_value_counts: dict[str, int] = {}
    x_min: float | None = None
    x_max: float | None = None
    y_min: float | None = None
    y_max: float | None = None

    for source_file in inventory.files:
        path = inventory.source.parquet_path / source_file.relative_path
        parquet_file = _open_parquet_content_file(path, source_file.relative_path)
        file_rows = 0

        for row_group_index, expected_row_group in enumerate(source_file.row_groups):
            row_group_rows = 0
            try:
                batches = parquet_file.iter_batches(
                    row_groups=[row_group_index],
                    batch_size=max_batch_rows,
                    columns=columns,
                )
                for batch in batches:
                    batch_rows = batch.num_rows
                    if batch_rows == 0:
                        continue

                    batch_x_bounds = _coordinate_batch_bounds(
                        batch.column(0),
                        role="x",
                        column_name=columns[0],
                        relative_path=source_file.relative_path,
                        row_group_index=row_group_index,
                    )
                    batch_y_bounds = _coordinate_batch_bounds(
                        batch.column(1),
                        role="y",
                        column_name=columns[1],
                        relative_path=source_file.relative_path,
                        row_group_index=row_group_index,
                    )
                    batch_value_counts = _normalized_value_counts(
                        batch.column(2),
                        column_name=columns[2],
                        relative_path=source_file.relative_path,
                        row_group_index=row_group_index,
                    )

                    if sum(batch_value_counts.values()) != batch_rows:
                        raise _content_error(
                            "Normalized value counts do not match the decoded batch row count.",
                            code="batch_value_count_mismatch",
                            relative_path=source_file.relative_path,
                            row_group_index=row_group_index,
                            role="value",
                            column_name=columns[2],
                        )

                    observed_row_group_rows = row_group_rows + batch_rows
                    if observed_row_group_rows > expected_row_group.row_count:
                        raise _content_error(
                            "Decoded rows exceed the Parquet row-group metadata count "
                            f"({observed_row_group_rows} observed, {expected_row_group.row_count} expected).",
                            code="row_group_row_count_mismatch",
                            relative_path=source_file.relative_path,
                            row_group_index=row_group_index,
                        )

                    row_group_rows = observed_row_group_rows
                    x_min = batch_x_bounds[0] if x_min is None else min(x_min, batch_x_bounds[0])
                    x_max = batch_x_bounds[1] if x_max is None else max(x_max, batch_x_bounds[1])
                    y_min = batch_y_bounds[0] if y_min is None else min(y_min, batch_y_bounds[0])
                    y_max = batch_y_bounds[1] if y_max is None else max(y_max, batch_y_bounds[1])
                    _merge_value_counts(
                        global_value_counts,
                        batch_value_counts,
                    )
            except PointContentValidationError:
                raise
            except Exception as error:
                raise _content_error(
                    f"Could not decode selected point content: {error}.",
                    code="parquet_content_read_failed",
                    relative_path=source_file.relative_path,
                    row_group_index=row_group_index,
                ) from error

            if row_group_rows != expected_row_group.row_count:
                raise _content_error(
                    "Decoded rows do not match the Parquet row-group metadata count "
                    f"({row_group_rows} observed, {expected_row_group.row_count} expected).",
                    code="row_group_row_count_mismatch",
                    relative_path=source_file.relative_path,
                    row_group_index=row_group_index,
                )
            file_rows += row_group_rows

        if file_rows != source_file.row_count:
            raise _content_error(
                "Decoded row-group rows do not match the Parquet source-file metadata count "
                f"({file_rows} observed, {source_file.row_count} expected).",
                code="source_file_row_count_mismatch",
                relative_path=source_file.relative_path,
            )
        total_rows += file_rows

    if total_rows != inventory.row_count:
        raise PointContentValidationError(
            "Decoded source-file rows do not match the Parquet source inventory count "
            f"({total_rows} observed, {inventory.row_count} expected).",
            code="source_row_count_mismatch",
        )

    value_count_total = sum(global_value_counts.values())
    if value_count_total != total_rows:
        raise PointContentValidationError(
            "Normalized value counts do not match the decoded source row count "
            f"({value_count_total} counted, {total_rows} observed).",
            code="value_count_mismatch",
        )

    if x_min is None or x_max is None or y_min is None or y_max is None:
        raise PointContentValidationError(
            "The non-empty Parquet points source produced no coordinate bounds.",
            code="source_row_count_mismatch",
        )

    return _ScannedPointsContent(
        row_count=total_rows,
        bounds=PointsBounds(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max),
        value_table=_build_value_table(global_value_counts),
    )


def _open_parquet_content_file(path: Path, relative_path: str) -> pq.ParquetFile:
    try:
        return pq.ParquetFile(path)
    except Exception as error:
        raise _content_error(
            f"Could not open selected point content: {error}.",
            code="parquet_content_read_failed",
            relative_path=relative_path,
        ) from error


def _coordinate_batch_bounds(
    values: pa.Array,
    *,
    role: str,
    column_name: str,
    relative_path: str,
    row_group_index: int,
) -> tuple[float, float]:
    if values.null_count:
        raise _content_error(
            f"Selected coordinate contains {values.null_count} null value(s) in the current batch.",
            code="invalid_coordinate_content",
            relative_path=relative_path,
            row_group_index=row_group_index,
            role=role,
            column_name=column_name,
        )

    coordinates = pc.cast(values, pa.float64()).to_numpy(zero_copy_only=False)
    finite = np.isfinite(coordinates)
    if not bool(finite.all()):
        invalid_count = int(np.count_nonzero(~finite))
        raise _content_error(
            f"Selected coordinate contains {invalid_count} non-finite value(s) in the current batch.",
            code="invalid_coordinate_content",
            relative_path=relative_path,
            row_group_index=row_group_index,
            role=role,
            column_name=column_name,
        )
    return float(coordinates.min()), float(coordinates.max())


def _normalized_value_counts(
    values: pa.Array,
    *,
    column_name: str,
    relative_path: str,
    row_group_index: int,
) -> dict[str, int]:
    if pa.types.is_dictionary(values.type):
        return _normalized_dictionary_value_counts(
            values,
            column_name=column_name,
            relative_path=relative_path,
            row_group_index=row_group_index,
        )
    return _normalized_plain_value_counts(
        values,
        column_name=column_name,
        relative_path=relative_path,
        row_group_index=row_group_index,
    )


def _normalized_plain_value_counts(
    values: pa.Array,
    *,
    column_name: str,
    relative_path: str,
    row_group_index: int,
) -> dict[str, int]:
    if values.null_count:
        raise _content_error(
            f"Selected value contains {values.null_count} null logical value(s) in the current batch.",
            code="invalid_value_content",
            relative_path=relative_path,
            row_group_index=row_group_index,
            role="value",
            column_name=column_name,
        )

    normalized = pc.utf8_trim(values, characters=_UNICODE_WHITE_SPACE)
    empty_count = int(pc.sum(pc.cast(pc.equal(normalized, ""), pa.int64())).as_py() or 0)
    if empty_count:
        raise _content_error(
            f"Selected value contains {empty_count} value(s) that normalize to an empty string in the current batch.",
            code="invalid_value_content",
            relative_path=relative_path,
            row_group_index=row_group_index,
            role="value",
            column_name=column_name,
        )
    return _value_counts_result_to_mapping(pc.value_counts(normalized))


def _normalized_dictionary_value_counts(
    values: pa.DictionaryArray,
    *,
    column_name: str,
    relative_path: str,
    row_group_index: int,
) -> dict[str, int]:
    if values.indices.null_count:
        raise _content_error(
            f"Selected value contains {values.indices.null_count} null dictionary index value(s) in the current batch.",
            code="invalid_value_content",
            relative_path=relative_path,
            row_group_index=row_group_index,
            role="value",
            column_name=column_name,
        )

    normalized_dictionary = pc.utf8_trim(values.dictionary, characters=_UNICODE_WHITE_SPACE)
    index_counts = pc.value_counts(values.indices)
    local_counts: dict[str, int] = {}
    for index_scalar, count_scalar in zip(index_counts.field("values"), index_counts.field("counts"), strict=True):
        index = int(index_scalar.as_py())
        normalized_value = normalized_dictionary[index]
        if not normalized_value.is_valid:
            raise _content_error(
                "Selected value contains a dictionary index that references a null value.",
                code="invalid_value_content",
                relative_path=relative_path,
                row_group_index=row_group_index,
                role="value",
                column_name=column_name,
            )
        label = normalized_value.as_py()
        if label == "":
            raise _content_error(
                "Selected value contains a referenced dictionary value that normalizes to an empty string.",
                code="invalid_value_content",
                relative_path=relative_path,
                row_group_index=row_group_index,
                role="value",
                column_name=column_name,
            )
        local_counts[label] = local_counts.get(label, 0) + int(count_scalar.as_py())
    return local_counts


def _value_counts_result_to_mapping(value_counts: pa.StructArray) -> dict[str, int]:
    return {
        value_scalar.as_py(): int(count_scalar.as_py())
        for value_scalar, count_scalar in zip(
            value_counts.field("values"),
            value_counts.field("counts"),
            strict=True,
        )
    }


def _merge_value_counts(
    global_counts: dict[str, int],
    batch_counts: dict[str, int],
) -> None:
    for label, count in batch_counts.items():
        if label not in global_counts:
            if len(global_counts) == _MAX_VALUE_CARDINALITY:
                raise PointContentValidationError(
                    "Selected value cardinality exceeds the uint32 value-id capacity.",
                    code="value_cardinality_overflow",
                )
            global_counts[label] = count
        else:
            global_counts[label] += count


def _build_value_table(value_counts: dict[str, int]) -> pa.Table:
    labels = sorted(value_counts, key=lambda label: label.encode("utf-8"))
    try:
        return pa.Table.from_arrays(
            [
                pa.array(range(len(labels)), type=pa.uint32()),
                pa.array(labels, type=pa.string()),
                pa.array((value_counts[label] for label in labels), type=pa.uint64()),
            ],
            schema=_VALUE_TABLE_SCHEMA,
        )
    except (pa.ArrowCapacityError, pa.ArrowInvalid, pa.ArrowTypeError, OverflowError) as error:
        raise PointContentValidationError(
            f"Normalized values cannot be represented by the exact value-table schema: {error}.",
            code="invalid_value_content",
        ) from error


def _content_error(
    message: str,
    *,
    code: str,
    relative_path: str,
    row_group_index: int | None = None,
    role: str | None = None,
    column_name: str | None = None,
) -> PointContentValidationError:
    context = [f"source file `{relative_path}`"]
    if row_group_index is not None:
        context.append(f"row group {row_group_index}")
    if role is not None and column_name is not None:
        context.append(f"selected {role} column `{column_name}`")
    return PointContentValidationError(f"{message} ({', '.join(context)})", code=code)


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
