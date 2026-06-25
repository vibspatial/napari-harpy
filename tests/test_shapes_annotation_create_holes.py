from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_rgba
from napari.layers import Shapes
from napari.layers.shapes._shapes_constants import Mode
from shapely.geometry import Polygon

import napari_harpy.widgets.shapes_annotation._create_holes as create_holes_module
from napari_harpy.core.shapes_geometry import (
    napari_polygon_vertices_to_shapely_polygon,
    shapely_polygon_to_napari_polygon_vertices,
)
from napari_harpy.viewer.shapes_styling import apply_primary_shapes_layer_style


def _copy_layer_data(layer: object) -> list[np.ndarray]:
    return [np.asarray(vertices, dtype=float).copy() for vertices in layer.data]


def _assert_layer_data_unchanged(layer: object, expected_data: list[np.ndarray]) -> None:
    assert len(layer.data) == len(expected_data)
    for actual_vertices, expected_vertices in zip(layer.data, expected_data, strict=True):
        np.testing.assert_allclose(np.asarray(actual_vertices, dtype=float), expected_vertices)


def _rgba(color: str) -> np.ndarray:
    return np.asarray(to_rgba(color), dtype=float)


def test_create_holes_plan_from_selection_selects_largest_shell_and_encodes_child_holes() -> None:
    shell = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    child_1 = Polygon([(2, 2), (2, 4), (4, 4), (4, 2)])
    child_2 = Polygon([(6, 6), (6, 8), (8, 8), (8, 6)])
    layer = Shapes(
        [
            shapely_polygon_to_napari_polygon_vertices(child_2),
            shapely_polygon_to_napari_polygon_vertices(shell),
            shapely_polygon_to_napari_polygon_vertices(child_1),
        ],
        shape_type=["polygon", "polygon", "polygon"],
    )
    layer.selected_data = {2, 0, 1}
    original_data = _copy_layer_data(layer)

    plan = create_holes_module._create_holes_plan_from_selection(layer)

    assert plan.shell_row_index == 1
    assert plan.hole_row_indices == (0, 2)
    planned_polygon = napari_polygon_vertices_to_shapely_polygon(plan.vertices)
    assert planned_polygon.equals(Polygon(shell.exterior.coords, holes=[child_2.exterior.coords, child_1.exterior.coords]))
    assert len(planned_polygon.interiors) == 2
    _assert_layer_data_unchanged(layer, original_data)
    assert set(layer.selected_data) == {0, 1, 2}


def test_create_holes_plan_from_selection_preserves_existing_shell_holes() -> None:
    existing_hole = [(2, 2), (2, 4), (4, 4), (4, 2)]
    shell = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)], holes=[existing_hole])
    child = Polygon([(6, 6), (6, 8), (8, 8), (8, 6)])
    layer = Shapes(
        [
            shapely_polygon_to_napari_polygon_vertices(shell),
            shapely_polygon_to_napari_polygon_vertices(child),
        ],
        shape_type=["polygon", "polygon"],
    )
    layer.selected_data = {0, 1}

    plan = create_holes_module._create_holes_plan_from_selection(layer)

    assert plan.shell_row_index == 0
    assert plan.hole_row_indices == (1,)
    planned_polygon = napari_polygon_vertices_to_shapely_polygon(plan.vertices)
    assert planned_polygon.equals(Polygon(shell.exterior.coords, holes=[existing_hole, child.exterior.coords]))
    assert len(planned_polygon.interiors) == 2


def test_create_holes_plan_from_selection_fails_for_ambiguous_largest_area_without_mutation() -> None:
    shell_1 = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
    shell_2 = Polygon([(10, 10), (14, 10), (14, 14), (10, 14)])
    layer = Shapes(
        [
            shapely_polygon_to_napari_polygon_vertices(shell_1),
            shapely_polygon_to_napari_polygon_vertices(shell_2),
        ],
        shape_type=["polygon", "polygon"],
    )
    layer.selected_data = {0, 1}
    original_data = _copy_layer_data(layer)

    with pytest.raises(ValueError, match="largest selected polygons have equal area"):
        create_holes_module._create_holes_plan_from_selection(layer)

    _assert_layer_data_unchanged(layer, original_data)
    assert set(layer.selected_data) == {0, 1}


@pytest.mark.parametrize("selected_data", [set(), {0}])
def test_create_holes_plan_from_selection_requires_at_least_two_selected_rows(selected_data: set[int]) -> None:
    shell = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
    layer = Shapes([shapely_polygon_to_napari_polygon_vertices(shell)], shape_type="polygon")
    layer.selected_data = selected_data

    with pytest.raises(ValueError, match="Select one shell polygon"):
        create_holes_module._create_holes_plan_from_selection(layer)


def test_create_holes_plan_from_selection_rejects_non_polygon_selected_row_without_mutation() -> None:
    shell = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
    line_vertices = np.asarray([[1.0, 1.0], [2.0, 2.0]], dtype=float)
    layer = Shapes(
        [shapely_polygon_to_napari_polygon_vertices(shell), line_vertices],
        shape_type=["polygon", "line"],
    )
    layer.selected_data = {0, 1}
    original_data = _copy_layer_data(layer)

    with pytest.raises(ValueError, match="requires selected polygon rows"):
        create_holes_module._create_holes_plan_from_selection(layer)

    _assert_layer_data_unchanged(layer, original_data)
    assert set(layer.selected_data) == {0, 1}


def test_apply_create_holes_plan_replaces_shell_removes_children_and_preserves_layer_state() -> None:
    child_before = Polygon([(2, 2), (2, 4), (4, 4), (4, 2)])
    shell = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    child_after = Polygon([(6, 6), (6, 8), (8, 8), (8, 6)])
    unselected = Polygon([(20, 20), (20, 22), (22, 22), (22, 20)])
    layer = Shapes(
        [
            shapely_polygon_to_napari_polygon_vertices(child_before),
            shapely_polygon_to_napari_polygon_vertices(shell),
            shapely_polygon_to_napari_polygon_vertices(child_after),
            shapely_polygon_to_napari_polygon_vertices(unselected),
        ],
        shape_type=["polygon", "polygon", "polygon", "polygon"],
        features=pd.DataFrame(
            {
                "label": ["child_before", "shell", "child_after", "unselected"],
                "source_id": ["row-0", "row-1", "row-2", "row-3"],
            }
        ),
    )
    layer.mode = Mode.DIRECT
    layer.selected_data = {0, 1, 2}
    plan = create_holes_module._create_holes_plan_from_selection(layer)

    create_holes_module._apply_create_holes_plan(layer, plan)

    assert len(layer.data) == 2
    assert list(layer.shape_type) == ["polygon", "polygon"]
    assert layer.features["label"].tolist() == ["shell", "unselected"]
    assert layer.features["source_id"].tolist() == ["row-1", "row-3"]
    assert layer.mode == Mode.DIRECT
    assert set(layer.selected_data) == {0}

    expected_shell = Polygon(
        shell.exterior.coords,
        holes=[child_before.exterior.coords, child_after.exterior.coords],
    )
    planned_shell = napari_polygon_vertices_to_shapely_polygon(layer.data[0])
    assert planned_shell.equals(expected_shell)
    assert napari_polygon_vertices_to_shapely_polygon(layer.data[1]).equals(unselected)


def test_apply_create_holes_plan_preserves_styles_with_stale_napari_color_arrays() -> None:
    child_before = Polygon([(2, 2), (2, 4), (4, 4), (4, 2)])
    shell = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    child_after = Polygon([(6, 6), (6, 8), (8, 8), (8, 6)])
    unselected = Polygon([(20, 20), (20, 22), (22, 22), (22, 20)])
    layer = Shapes(
        [
            shapely_polygon_to_napari_polygon_vertices(child_before),
            shapely_polygon_to_napari_polygon_vertices(shell),
            shapely_polygon_to_napari_polygon_vertices(child_after),
            shapely_polygon_to_napari_polygon_vertices(unselected),
        ],
        shape_type=["polygon", "polygon", "polygon", "polygon"],
    )
    apply_primary_shapes_layer_style(layer)

    current_edge_color = "#aa00aa"
    current_face_color = "#bbccdd44"
    current_edge_width = 9
    layer.current_edge_color = current_edge_color
    layer.current_face_color = current_face_color
    layer.current_edge_width = current_edge_width

    edge_color = np.asarray(
        [
            _rgba("#ff0000"),
            _rgba("#00ffff"),
            _rgba("#00ff00"),
            _rgba("#123456"),
        ],
        dtype=float,
    )
    face_color = np.asarray(
        [
            _rgba("#111111ff"),
            _rgba("#00000000"),
            _rgba("#222222ff"),
            _rgba("#65432188"),
        ],
        dtype=float,
    )
    edge_width = [2, 3, 4, 5]
    z_index = [10, 11, 12, 13]
    layer.edge_color = edge_color
    layer.face_color = face_color
    layer.edge_width = edge_width
    layer.z_index = z_index
    layer.opacity = 0.37

    # Simulate the intermittent napari state behind the UI bug: the logical
    # data rows are correct, but private color arrays have stale extra rows.
    layer._data_view._edge_color = np.vstack([layer._data_view._edge_color, _rgba("#ffff00")])
    layer._data_view._face_color = np.vstack([layer._data_view._face_color, _rgba("#ffff00")])

    layer.mode = Mode.DIRECT
    layer.selected_data = {0, 1, 2}
    plan = create_holes_module._create_holes_plan_from_selection(layer)

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        create_holes_module._apply_create_holes_plan(layer, plan)

    style_warnings = [
        str(warning.message)
        for warning in caught_warnings
        if "edge_color" in str(warning.message) or "face_color" in str(warning.message)
    ]
    assert style_warnings == []
    assert set(layer.selected_data) == {0}
    assert layer.mode == Mode.DIRECT
    assert layer.opacity == 0.37
    assert layer.edge_width == [3, 5]
    assert layer.z_index == [11, 13]
    np.testing.assert_allclose(layer.edge_color, edge_color[[1, 3]])
    np.testing.assert_allclose(layer.face_color, face_color[[1, 3]])
    np.testing.assert_allclose(to_rgba(layer.current_edge_color), to_rgba(current_edge_color))
    np.testing.assert_allclose(to_rgba(layer.current_face_color), to_rgba(current_face_color))
    assert layer.current_edge_width == current_edge_width

    expected_shell = Polygon(
        shell.exterior.coords,
        holes=[child_before.exterior.coords, child_after.exterior.coords],
    )
    assert napari_polygon_vertices_to_shapely_polygon(layer.data[0]).equals(expected_shell)
    assert napari_polygon_vertices_to_shapely_polygon(layer.data[1]).equals(unselected)


def test_create_holes_plan_rejects_self_inconsistent_indices() -> None:
    shell = Polygon([(0, 0), (8, 0), (8, 8), (0, 8)])

    with pytest.raises(ValueError, match="cannot remove the shell row"):
        create_holes_module._CreateHolesShapesLayerPlan(
            shell_row_index=0,
            hole_row_indices=(0,),
            vertices=shapely_polygon_to_napari_polygon_vertices(shell),
        )


def test_apply_create_holes_plan_rejects_invalid_plan_before_mutation() -> None:
    shell = Polygon([(0, 0), (8, 0), (8, 8), (0, 8)])
    child = Polygon([(2, 2), (2, 4), (4, 4), (4, 2)])
    layer = Shapes(
        [
            shapely_polygon_to_napari_polygon_vertices(shell),
            shapely_polygon_to_napari_polygon_vertices(child),
        ],
        shape_type=["polygon", "polygon"],
        features=pd.DataFrame({"label": ["shell", "child"]}),
    )
    layer.selected_data = {0, 1}
    original_data = _copy_layer_data(layer)
    original_features = layer.features.copy()
    invalid_plan = create_holes_module._CreateHolesShapesLayerPlan(
        shell_row_index=0,
        hole_row_indices=(2,),
        vertices=shapely_polygon_to_napari_polygon_vertices(shell),
    )

    with pytest.raises(ValueError, match="hole row is no longer present"):
        create_holes_module._apply_create_holes_plan(layer, invalid_plan)

    _assert_layer_data_unchanged(layer, original_data)
    pd.testing.assert_frame_equal(layer.features, original_features)
    assert set(layer.selected_data) == {0, 1}
