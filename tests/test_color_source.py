from __future__ import annotations

import pytest

from napari_harpy.core._color_source import (
    ShapeColumnColorSourceSpec,
    TableColorSourceSpec,
)
from napari_harpy.viewer.labels_styling import LabelsStyleResult
from napari_harpy.viewer.shapes_styling import ShapesStyleResult


def test_table_color_source_spec_rejects_invalid_source_kind() -> None:
    with pytest.raises(ValueError, match="Invalid table color source kind"):
        TableColorSourceSpec(
            table_name="table",
            source_kind="layer",
            value_key="cell_type",
            value_kind="categorical",
        )


def test_table_color_source_spec_rejects_invalid_value_kind() -> None:
    with pytest.raises(ValueError, match="Invalid table color value kind"):
        TableColorSourceSpec(
            table_name="table",
            source_kind="obs_column",
            value_key="cell_type",
            value_kind="ordinal",
        )


def test_shape_color_source_spec_rejects_invalid_source_kind() -> None:
    with pytest.raises(ValueError, match="Invalid shape color source kind"):
        ShapeColumnColorSourceSpec(
            source_kind="obs_column",
            value_key="cell_type",
            value_kind="categorical",
        )


def test_shape_color_source_spec_rejects_invalid_value_kind() -> None:
    with pytest.raises(ValueError, match="Invalid shape color value kind"):
        ShapeColumnColorSourceSpec(
            source_kind="shape_column",
            value_key="cell_type",
            value_kind="instance",
        )


def test_explicit_shape_source_union_covers_shape_columns_and_table_sources() -> None:
    sources: list[ShapeColumnColorSourceSpec | TableColorSourceSpec] = [
        ShapeColumnColorSourceSpec(
            source_kind="shape_column",
            value_key="cell_type",
            value_kind="categorical",
        ),
        TableColorSourceSpec(
            table_name="table",
            source_kind="obs_column",
            value_key="cell_type",
            value_kind="categorical",
        ),
    ]

    assert [source.value_key for source in sources] == ["cell_type", "cell_type"]


def test_labels_style_result_accepts_no_style_metadata() -> None:
    result = LabelsStyleResult(
        value_kind=None,
        palette_source=None,
        coercion_applied=False,
    )

    assert result.value_kind is None


def test_labels_style_result_rejects_invalid_value_kind() -> None:
    with pytest.raises(ValueError, match="Invalid table color value kind"):
        LabelsStyleResult(
            value_kind="ordinal",
            palette_source=None,
            coercion_applied=False,
        )


def test_labels_style_result_rejects_invalid_palette_source() -> None:
    with pytest.raises(ValueError, match="Invalid styled palette source"):
        LabelsStyleResult(
            value_kind="categorical",
            palette_source="generated",
            coercion_applied=False,
        )


def test_shapes_style_result_rejects_invalid_palette_source() -> None:
    with pytest.raises(ValueError, match="Invalid styled palette source"):
        ShapesStyleResult(
            value_kind="categorical",
            palette_source="generated",
            coercion_applied=False,
        )


def test_shapes_style_result_accepts_no_style_metadata() -> None:
    result = ShapesStyleResult(
        value_kind=None,
        palette_source=None,
        coercion_applied=False,
    )

    assert result.value_kind is None


def test_shapes_style_result_accepts_table_backed_instance_coloring_metadata() -> None:
    result = ShapesStyleResult(
        value_kind="instance",
        palette_source=None,
        coercion_applied=False,
        unannotated_source_shape_count=2,
        unannotated_rendered_shape_count=3,
    )

    assert result.value_kind == "instance"
    assert result.unannotated_source_shape_count == 2
    assert result.unannotated_rendered_shape_count == 3


def test_shapes_style_result_rejects_invalid_value_kind() -> None:
    with pytest.raises(ValueError, match="Invalid shape color value kind"):
        ShapesStyleResult(
            value_kind="ordinal",
            palette_source=None,
            coercion_applied=False,
        )
