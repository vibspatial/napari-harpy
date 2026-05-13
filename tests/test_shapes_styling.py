from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_rgba
from napari.layers import Shapes
from shapely.geometry import Polygon

from napari_harpy.core._color_source import ShapeColorSourceSpec
from napari_harpy.viewer._styling import continuous_colors_for_values
from napari_harpy.viewer.shapes_styling import (
    SHAPES_EDGE_ALPHA,
    SHAPES_FACE_ALPHA,
    SHAPES_MISSING_BASE_COLOR,
    _apply_rendered_row_colors_to_shapes_layer,
    apply_shape_color_source_to_shapes_layer,
    build_styled_shapes_layer_name,
    disambiguate_shape_style_feature_name,
)


def _polygon(offset: float = 0.0) -> Polygon:
    return Polygon([(offset, 0), (offset + 1, 0), (offset + 1, 1), (offset, 1)])


def _shape_vertices(offset: float) -> np.ndarray:
    return np.asarray([(0, offset), (0, offset + 1), (1, offset + 1), (1, offset)], dtype=float)


def _make_shapes_layer(
    source_indices: tuple[object, ...],
    *,
    source_index_feature_name: str = "index",
) -> Shapes:
    return Shapes(
        [_shape_vertices(float(index)) for index in range(len(source_indices))],
        shape_type=["polygon"] * len(source_indices),
        features=pd.DataFrame({source_index_feature_name: list(source_indices)}),
    )


def _rgba(color: object, alpha: float) -> tuple[float, float, float, float]:
    red, green, blue, _alpha = to_rgba(color)
    return (red, green, blue, alpha)


def test_apply_shape_color_source_to_shapes_layer_uses_stored_categorical_companion_palette() -> None:
    geodataframe = gpd.GeoDataFrame(
        {
            "cell_type": pd.Categorical(["T", "B", None]),
            "cell_type_colors": ["red", "blue", None],
        },
        geometry=[_polygon(0), _polygon(2), _polygon(4)],
        index=["cell_1", "cell_2", "cell_3"],
    )
    layer = _make_shapes_layer(("cell_1", "cell_1", "cell_2", "cell_3"))
    style_spec = ShapeColorSourceSpec(
        source_kind="shape_column",
        value_key="cell_type",
        value_kind="categorical",
    )

    result = apply_shape_color_source_to_shapes_layer(
        layer,
        shapes_element=geodataframe,
        style_spec=style_spec,
        source_shapes_index_by_row=("cell_1", "cell_1", "cell_2", "cell_3"),
        source_shapes_index_feature_name="index",
    )

    assert result.value_kind == "categorical"
    assert result.palette_source == "stored"
    assert result.coercion_applied is False
    np.testing.assert_allclose(
        layer.face_color,
        np.asarray(
            [
                _rgba("red", SHAPES_FACE_ALPHA),
                _rgba("red", SHAPES_FACE_ALPHA),
                _rgba("blue", SHAPES_FACE_ALPHA),
                _rgba(SHAPES_MISSING_BASE_COLOR, SHAPES_FACE_ALPHA),
            ]
        ),
    )
    np.testing.assert_allclose(
        layer.edge_color,
        np.asarray(
            [
                _rgba("red", SHAPES_EDGE_ALPHA),
                _rgba("red", SHAPES_EDGE_ALPHA),
                _rgba("blue", SHAPES_EDGE_ALPHA),
                _rgba(SHAPES_MISSING_BASE_COLOR, SHAPES_EDGE_ALPHA),
            ]
        ),
    )
    assert layer.features["cell_type"].iloc[:3].to_list() == ["T", "T", "B"]
    assert pd.isna(layer.features["cell_type"].iloc[3])


def test_apply_shape_color_source_to_shapes_layer_uses_continuous_colormap_and_missing_gray() -> None:
    geodataframe = gpd.GeoDataFrame(
        {"score": [0.0, 10.0, None]},
        geometry=[_polygon(0), _polygon(2), _polygon(4)],
        index=["cell_1", "cell_2", "cell_3"],
    )
    layer = _make_shapes_layer(("cell_1", "cell_2", "cell_3"))
    style_spec = ShapeColorSourceSpec(
        source_kind="shape_column",
        value_key="score",
        value_kind="continuous",
    )

    result = apply_shape_color_source_to_shapes_layer(
        layer,
        shapes_element=geodataframe,
        style_spec=style_spec,
        source_shapes_index_by_row=("cell_1", "cell_2", "cell_3"),
        source_shapes_index_feature_name="index",
    )

    rendered_row_colors = continuous_colors_for_values(
        pd.Series([0.0, 10.0, np.nan]),
        missing_color=SHAPES_MISSING_BASE_COLOR,
    )
    expected_face = np.asarray([_rgba(color, SHAPES_FACE_ALPHA) for color in rendered_row_colors])
    expected_edge = np.asarray([_rgba(color, SHAPES_EDGE_ALPHA) for color in rendered_row_colors])

    assert result.value_kind == "continuous"
    assert result.palette_source is None
    assert result.coercion_applied is False
    np.testing.assert_allclose(layer.face_color, expected_face)
    np.testing.assert_allclose(layer.edge_color, expected_edge)
    assert layer.features["score"].to_list()[:2] == [0.0, 10.0]
    assert pd.isna(layer.features["score"].iloc[2])


@pytest.mark.parametrize(
    "colors",
    [
        ["red", "not-a-color", "green"],
        ["red", "blue", "green"],
        ["red", "red", None],
    ],
)
def test_apply_shape_color_source_to_shapes_layer_rejects_invalid_companion_palettes(colors: list[object]) -> None:
    geodataframe = gpd.GeoDataFrame(
        {
            "group": pd.Categorical(["a", "a", "b"]),
            "group_colors": colors,
        },
        geometry=[_polygon(0), _polygon(2), _polygon(4)],
        index=["cell_1", "cell_2", "cell_3"],
    )
    layer = _make_shapes_layer(("cell_1", "cell_2", "cell_3"))
    style_spec = ShapeColorSourceSpec(
        source_kind="shape_column",
        value_key="group",
        value_kind="categorical",
    )

    result = apply_shape_color_source_to_shapes_layer(
        layer,
        shapes_element=geodataframe,
        style_spec=style_spec,
        source_shapes_index_by_row=("cell_1", "cell_2", "cell_3"),
        source_shapes_index_feature_name="index",
    )

    assert result.value_kind == "categorical"
    assert result.palette_source == "default_invalid"
    assert result.coercion_applied is False


@pytest.mark.parametrize(
    ("values", "colors"),
    [
        ([True, False, True], ["red", "blue", "red"]),
        ([0, 1, 0], ["red", "blue", "red"]),
    ],
)
def test_apply_shape_color_source_to_shapes_layer_uses_stored_palettes_for_bool_and_binary_integer_columns(
    values: list[object],
    colors: list[str],
) -> None:
    geodataframe = gpd.GeoDataFrame(
        {
            "group": values,
            "group_colors": colors,
        },
        geometry=[_polygon(0), _polygon(2), _polygon(4)],
        index=["cell_1", "cell_2", "cell_3"],
    )
    layer = _make_shapes_layer(("cell_1", "cell_2", "cell_3"))
    style_spec = ShapeColorSourceSpec(
        source_kind="shape_column",
        value_key="group",
        value_kind="categorical",
    )

    result = apply_shape_color_source_to_shapes_layer(
        layer,
        shapes_element=geodataframe,
        style_spec=style_spec,
        source_shapes_index_by_row=("cell_1", "cell_2", "cell_3"),
        source_shapes_index_feature_name="index",
    )

    assert result.palette_source == "stored"
    np.testing.assert_allclose(
        layer.edge_color,
        np.asarray(
            [
                _rgba("red", SHAPES_EDGE_ALPHA),
                _rgba("blue", SHAPES_EDGE_ALPHA),
                _rgba("red", SHAPES_EDGE_ALPHA),
            ]
        ),
    )


def test_apply_shape_color_source_to_shapes_layer_reports_missing_companion_palette() -> None:
    geodataframe = gpd.GeoDataFrame(
        {"group": pd.Categorical(["a", "b"])},
        geometry=[_polygon(0), _polygon(2)],
        index=["cell_1", "cell_2"],
    )
    layer = _make_shapes_layer(("cell_1", "cell_2"))
    style_spec = ShapeColorSourceSpec(
        source_kind="shape_column",
        value_key="group",
        value_kind="categorical",
    )

    result = apply_shape_color_source_to_shapes_layer(
        layer,
        shapes_element=geodataframe,
        style_spec=style_spec,
        source_shapes_index_by_row=("cell_1", "cell_2"),
        source_shapes_index_feature_name="index",
    )

    assert result.palette_source == "default_missing"
    assert result.coercion_applied is False


def test_apply_shape_color_source_to_shapes_layer_treats_string_object_values_as_temporary_categorical() -> None:
    geodataframe = gpd.GeoDataFrame(
        {
            "free_text": ["alpha", "beta"],
            "free_text_colors": ["red", "blue"],
        },
        geometry=[_polygon(0), _polygon(2)],
        index=["cell_1", "cell_2"],
    )
    original_dtype = geodataframe["free_text"].dtype
    layer = _make_shapes_layer(("cell_1", "cell_2"))
    style_spec = ShapeColorSourceSpec(
        source_kind="shape_column",
        value_key="free_text",
        value_kind="categorical",
    )

    result = apply_shape_color_source_to_shapes_layer(
        layer,
        shapes_element=geodataframe,
        style_spec=style_spec,
        source_shapes_index_by_row=("cell_1", "cell_2"),
        source_shapes_index_feature_name="index",
    )

    assert result.value_kind == "categorical"
    assert result.palette_source == "default_missing"
    assert result.coercion_applied is True
    assert geodataframe["free_text"].dtype == original_dtype
    assert layer.features["free_text"].to_list() == ["alpha", "beta"]


def test_apply_shape_color_source_to_shapes_layer_disambiguates_style_feature_from_source_index_feature() -> None:
    geodataframe = gpd.GeoDataFrame(
        {"cell_id": pd.Categorical(["A"])},
        geometry=[_polygon(0)],
        index=["cell_1"],
    )
    geodataframe.index.name = "cell_id"
    layer = _make_shapes_layer(("cell_1",), source_index_feature_name="cell_id")
    style_spec = ShapeColorSourceSpec(
        source_kind="shape_column",
        value_key="cell_id",
        value_kind="categorical",
    )

    apply_shape_color_source_to_shapes_layer(
        layer,
        shapes_element=geodataframe,
        style_spec=style_spec,
        source_shapes_index_by_row=("cell_1",),
        source_shapes_index_feature_name="cell_id",
    )

    assert layer.features["cell_id"].to_list() == ["cell_1"]
    assert layer.features["cell_id__value"].to_list() == ["A"]
    assert disambiguate_shape_style_feature_name("cell_id", "cell_id") == "cell_id__value"


def test_apply_shape_color_source_to_shapes_layer_rejects_duplicate_source_indices() -> None:
    geodataframe = gpd.GeoDataFrame(
        {"group": pd.Categorical(["a", "b"])},
        geometry=[_polygon(0), _polygon(2)],
        index=["cell_1", "cell_1"],
    )
    layer = _make_shapes_layer(("cell_1",))
    style_spec = ShapeColorSourceSpec(
        source_kind="shape_column",
        value_key="group",
        value_kind="categorical",
    )

    with pytest.raises(ValueError, match="requires unique source GeoDataFrame index"):
        apply_shape_color_source_to_shapes_layer(
            layer,
            shapes_element=geodataframe,
            style_spec=style_spec,
            source_shapes_index_by_row=("cell_1",),
            source_shapes_index_feature_name="index",
        )


def test_apply_rendered_row_colors_to_shapes_layer_requires_one_color_per_rendered_row() -> None:
    layer = _make_shapes_layer(("cell_1", "cell_2"))

    with pytest.raises(ValueError, match="one color for each rendered napari shape row"):
        _apply_rendered_row_colors_to_shapes_layer(layer, pd.Series(["red"]))


def test_build_styled_shapes_layer_name_returns_stable_shape_column_variant_name() -> None:
    style_spec = ShapeColorSourceSpec(
        source_kind="shape_column",
        value_key="cell_type",
        value_kind="categorical",
    )

    assert build_styled_shapes_layer_name("cell_boundaries", style_spec) == "cell_boundaries[shape:cell_type]"
