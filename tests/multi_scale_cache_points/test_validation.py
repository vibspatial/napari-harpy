from dataclasses import replace
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import napari_harpy.core.multi_scale_cache_points.validation as validation_module
from napari_harpy.core.multi_scale_cache_points import (
    ParquetPointsSource,
    PointColumnSelection,
    PointsSourceValidationError,
    ValidatedPointsSource,
    validate_parquet_points_source,
)
from napari_harpy.core.multi_scale_cache_points.errors import PointContentValidationError
from napari_harpy.core.multi_scale_cache_points.signature import POINT_ID_POLICY, SOURCE_SIGNATURE_METHOD
from napari_harpy.core.multi_scale_cache_points.validation import VALUE_NORMALIZATION_METHOD


def _source(tmp_path: Path) -> ParquetPointsSource:
    source = ParquetPointsSource(
        spatialdata_path=tmp_path / "example.zarr",
        points_name="transcripts",
        columns=PointColumnSelection(x="x", y="y", value="gene"),
    )
    source.parquet_path.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "x": pa.array([-2.0, 0.0, 8.0]),
                "y": pa.array([5.0, 3.0, 1.0]),
                "gene": pa.array([" B ", "A", "B"]),
            }
        ),
        source.parquet_path / "part.0.parquet",
        row_group_size=2,
    )
    return source


def test_validate_returns_deterministic_build_ready_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source(tmp_path)
    inventory_calls = 0
    scan_calls = 0
    read_inventory = validation_module._read_parquet_source_inventory
    scan_content = validation_module._scan_points_content

    def recording_inventory(source: ParquetPointsSource) -> validation_module._ParquetSourceInventory:
        nonlocal inventory_calls
        inventory_calls += 1
        return read_inventory(source)

    def recording_scan(
        inventory: validation_module._ParquetSourceInventory,
        *,
        max_batch_rows: int,
    ) -> validation_module._ScannedPointsContent:
        nonlocal scan_calls
        scan_calls += 1
        return scan_content(inventory, max_batch_rows=max_batch_rows)

    monkeypatch.setattr(validation_module, "_read_parquet_source_inventory", recording_inventory)
    monkeypatch.setattr(validation_module, "_scan_points_content", recording_scan)

    validated = validate_parquet_points_source(source, max_batch_rows=2)

    assert isinstance(validated, ValidatedPointsSource)
    assert validated.source is source
    assert validated.row_count == 3
    assert tuple(source_file.relative_path for source_file in validated.files) == ("part.0.parquet",)
    assert tuple(source_file.row_offset for source_file in validated.files) == (0,)
    assert validated.selected_schema.names == ["x", "y", "gene"]
    assert (validated.bounds.x_min, validated.bounds.x_max) == (-2.0, 8.0)
    assert (validated.bounds.y_min, validated.bounds.y_max) == (1.0, 5.0)
    assert validated.value_table.to_pylist() == [
        {"value_id": 0, "value": "A", "n_points": 1},
        {"value_id": 1, "value": "B", "n_points": 2},
    ]
    assert len(validated.source_signature) == 64
    assert validated.source_signature_method == SOURCE_SIGNATURE_METHOD
    assert validated.value_normalization_method == VALUE_NORMALIZATION_METHOD
    assert validated.point_id_policy == POINT_ID_POLICY
    assert inventory_calls == 2
    assert scan_calls == 1

    repeated = validate_parquet_points_source(source, max_batch_rows=2)
    assert repeated.source == validated.source
    assert repeated.files == validated.files
    assert repeated.selected_schema.equals(validated.selected_schema, check_metadata=True)
    assert repeated.row_count == validated.row_count
    assert repeated.bounds == validated.bounds
    assert repeated.value_table.equals(validated.value_table, check_metadata=True)
    assert repeated.source_signature == validated.source_signature
    assert repeated.source_signature_method == validated.source_signature_method
    assert repeated.value_normalization_method == validated.value_normalization_method
    assert repeated.point_id_policy == validated.point_id_policy


def test_validated_source_rejects_value_ids_that_do_not_match_row_order(tmp_path: Path) -> None:
    validated = validate_parquet_points_source(_source(tmp_path), max_batch_rows=2)
    reordered_ids = validated.value_table.set_column(
        0,
        pa.field("value_id", pa.uint32(), nullable=False),
        pa.array([1, 0], type=pa.uint32()),
    )

    with pytest.raises(PointContentValidationError, match="match table row order") as error:
        replace(validated, value_table=reordered_ids)

    assert error.value.code == "invalid_value_content"


def test_validate_rejects_source_signature_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source(tmp_path)
    inventory_before_scan = validation_module._read_parquet_source_inventory(source)
    source_file = inventory_before_scan.files[0]
    inventory_after_scan = replace(
        inventory_before_scan,
        files=(replace(source_file, modified_time_ns=source_file.modified_time_ns + 1),),
    )
    inventories = iter((inventory_before_scan, inventory_after_scan))
    monkeypatch.setattr(validation_module, "_read_parquet_source_inventory", lambda _source: next(inventories))

    with pytest.raises(PointsSourceValidationError, match="changed during validation") as error:
        validate_parquet_points_source(source, max_batch_rows=2)

    assert error.value.code == "source_changed_during_validation"


def test_validate_propagates_content_error_without_second_inventory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source(tmp_path)
    inventory_calls = 0
    read_inventory = validation_module._read_parquet_source_inventory
    expected_error = PointContentValidationError("invalid content", code="invalid_value_content")

    def recording_inventory(source: ParquetPointsSource) -> validation_module._ParquetSourceInventory:
        nonlocal inventory_calls
        inventory_calls += 1
        return read_inventory(source)

    def fail_scan(
        _inventory: validation_module._ParquetSourceInventory,
        *,
        max_batch_rows: int,
    ) -> validation_module._ScannedPointsContent:
        del max_batch_rows
        raise expected_error

    monkeypatch.setattr(validation_module, "_read_parquet_source_inventory", recording_inventory)
    monkeypatch.setattr(validation_module, "_scan_points_content", fail_scan)

    with pytest.raises(PointContentValidationError) as error:
        validate_parquet_points_source(source)

    assert error.value is expected_error
    assert inventory_calls == 1
