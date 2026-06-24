from __future__ import annotations

import numpy as np
import pytest
from napari.layers import Shapes
from shapely.geometry import Polygon

import napari_harpy.widgets.shapes_annotation._create_holes as create_holes_module
from napari_harpy.core.shapes_geometry import (
    napari_polygon_vertices_to_shapely_polygon,
    shapely_polygon_to_napari_polygon_vertices,
)


def _copy_layer_data(layer: object) -> list[np.ndarray]:
    return [np.asarray(vertices, dtype=float).copy() for vertices in layer.data]


def _assert_layer_data_unchanged(layer: object, expected_data: list[np.ndarray]) -> None:
    assert len(layer.data) == len(expected_data)
    for actual_vertices, expected_vertices in zip(layer.data, expected_data, strict=True):
        np.testing.assert_allclose(np.asarray(actual_vertices, dtype=float), expected_vertices)


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
