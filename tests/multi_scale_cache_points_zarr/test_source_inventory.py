from collections.abc import Callable
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from napari_harpy.core.multi_scale_cache_points_zarr.source import ParquetPointsSource, PointColumnSelection
from napari_harpy.core.multi_scale_cache_points_zarr.source.errors import ParquetMetadataValidationError
from napari_harpy.core.multi_scale_cache_points_zarr.source.validation import _read_parquet_source_inventory


def _source(tmp_path: Path) -> ParquetPointsSource:
    source = ParquetPointsSource(
        spatialdata_path=tmp_path / "example.zarr",
        points_name="transcripts",
        columns=PointColumnSelection(x="x", y="y", value="gene"),
    )
    source.parquet_path.mkdir(parents=True)
    return source


def _table(
    row_count: int,
    *,
    x_type: pa.DataType = pa.float64(),
    include_gene: bool = True,
    extra: pa.Array | None = None,
) -> pa.Table:
    fields = {
        "x": pa.array(range(row_count), type=x_type),
        "y": pa.array(range(row_count), type=pa.float64()),
    }
    if include_gene:
        fields["gene"] = pa.array(
            [f"gene_{index % 2}" for index in range(row_count)],
            type=pa.string(),
        ).dictionary_encode()
    if extra is not None:
        fields["extra"] = extra
    return pa.table(fields)


def test_read_metadata_builds_deterministic_inventory(tmp_path: Path) -> None:
    source = _source(tmp_path)
    pq.write_table(
        _table(3, extra=pa.array([1, 2, 3], type=pa.int64())),
        source.parquet_path / "part.1.parquet",
        row_group_size=2,
    )
    pq.write_table(
        _table(2, extra=pa.array(["a", "b"])),
        source.parquet_path / "part.10.parquet",
        row_group_size=2,
    )
    pq.write_table(
        _table(4),
        source.parquet_path / "part.2.parquet",
        row_group_size=2,
    )

    inventory = _read_parquet_source_inventory(source)

    assert inventory.row_count == 9
    assert tuple(source_file.relative_path for source_file in inventory.files) == (
        "part.1.parquet",
        "part.10.parquet",
        "part.2.parquet",
    )
    assert tuple(source_file.row_offset for source_file in inventory.files) == (0, 3, 5)
    assert tuple(source_file.row_count for source_file in inventory.files) == (3, 2, 4)
    assert tuple(source_file.row_group_count for source_file in inventory.files) == (2, 1, 2)
    assert tuple(row_group.row_count for row_group in inventory.files[0].row_groups) == (2, 1)
    assert inventory.selected_schema.names == ["x", "y", "gene"]
    assert inventory.selected_schema.field("x").type == pa.float64()
    assert pa.types.is_dictionary(inventory.selected_schema.field("gene").type)
    assert all(source_file.size_bytes > 0 for source_file in inventory.files)
    assert all(source_file.modified_time_ns is not None for source_file in inventory.files)

    first_metadata = pq.ParquetFile(source.parquet_path / "part.1.parquet").metadata
    first_row_group = first_metadata.row_group(0)
    expected_compressed_size = sum(
        first_row_group.column(index).total_compressed_size for index in range(first_row_group.num_columns)
    )
    assert inventory.files[0].row_groups[0].compressed_size_bytes == expected_compressed_size


@pytest.mark.parametrize(
    ("prepare", "code"),
    [
        (lambda _path: None, "parquet_dataset_empty"),
        (
            lambda path: (path / "corrupt.parquet").write_bytes(b"not parquet"),
            "unreadable_parquet_file_metadata",
        ),
    ],
)
def test_read_metadata_rejects_empty_or_corrupt_dataset(
    tmp_path: Path,
    prepare: Callable[[Path], object],
    code: str,
) -> None:
    source = _source(tmp_path)
    prepare(source.parquet_path)

    with pytest.raises(ParquetMetadataValidationError) as error:
        _read_parquet_source_inventory(source)

    assert error.value.code == code


@pytest.mark.parametrize(
    ("variant", "code"),
    [
        ("missing", "missing_selected_column"),
        ("incompatible", "incompatible_selected_schema"),
    ],
)
def test_read_metadata_rejects_invalid_selected_schema(tmp_path: Path, variant: str, code: str) -> None:
    source = _source(tmp_path)
    pq.write_table(_table(2), source.parquet_path / "part.0.parquet")
    if variant == "missing":
        second = _table(2, include_gene=False)
    else:
        second = _table(2, x_type=pa.int64())
    pq.write_table(second, source.parquet_path / "part.1.parquet")

    with pytest.raises(ParquetMetadataValidationError) as error:
        _read_parquet_source_inventory(source)

    assert error.value.code == code


def test_read_metadata_rejects_zero_row_source(tmp_path: Path) -> None:
    source = _source(tmp_path)
    pq.write_table(_table(0), source.parquet_path / "part.0.parquet")

    with pytest.raises(ParquetMetadataValidationError) as error:
        _read_parquet_source_inventory(source)

    assert error.value.code == "parquet_source_has_no_rows"
