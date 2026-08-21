from __future__ import annotations

from pathlib import Path

import pytest

from napari_harpy.core.multi_scale_cache_points_zarr.source import (
    ParquetPointsSource,
    PointColumnSelection,
    PointsSourceResolutionError,
    PointsSourceValidationError,
)


def _source(**overrides: object) -> ParquetPointsSource:
    values: dict[str, object] = {
        "spatialdata_path": Path("/source/example.zarr"),
        "points_name": "transcripts",
        "columns": PointColumnSelection(x="x", y="y", value="gene"),
    }
    values.update(overrides)
    return ParquetPointsSource(**values)  # type: ignore[arg-type]


def test_point_column_selection_preserves_exact_names() -> None:
    columns = PointColumnSelection(x=" x ", y="Y", value="gene label")

    assert (columns.x, columns.y, columns.value) == (" x ", "Y", "gene label")


@pytest.mark.parametrize(
    ("role", "column"),
    [("x", ""), ("y", None), ("value", 1)],
)
def test_point_column_selection_rejects_invalid_names(role: str, column: object) -> None:
    values: dict[str, object] = {"x": "x", "y": "y", "value": "gene"}
    values[role] = column

    with pytest.raises(PointsSourceValidationError, match=rf"`{role}`.*non-empty string") as error:
        PointColumnSelection(**values)  # type: ignore[arg-type]

    assert error.value.code == "invalid_point_column"


def test_point_column_selection_requires_distinct_names() -> None:
    with pytest.raises(PointsSourceValidationError, match="must be distinct") as error:
        PointColumnSelection(x="same", y="same", value="value")

    assert error.value.code == "duplicate_point_column"


@pytest.mark.parametrize(
    ("field_name", "value", "code"),
    [
        ("spatialdata_path", "/source/example.zarr", "invalid_spatialdata_path"),
        ("points_name", "", "invalid_points_name"),
        ("points_name", "nested/transcripts", "invalid_points_name"),
        ("columns", None, "invalid_point_column_selection"),
    ],
)
def test_parquet_points_source_rejects_invalid_contract_fields(field_name: str, value: object, code: str) -> None:
    with pytest.raises(PointsSourceValidationError) as error:
        _source(**{field_name: value})

    assert error.value.code == code


def test_validation_error_hierarchy_carries_stable_codes() -> None:
    validation_error = PointsSourceValidationError("invalid source")
    resolution_error = PointsSourceResolutionError("cannot resolve source")
    specific_resolution_error = PointsSourceResolutionError("not backed", code="spatialdata_not_backed")

    assert isinstance(validation_error, ValueError)
    assert validation_error.code == "points_source_validation"
    assert isinstance(resolution_error, PointsSourceValidationError)
    assert resolution_error.code == "points_source_resolution"
    assert specific_resolution_error.code == "spatialdata_not_backed"
