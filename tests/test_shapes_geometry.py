from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import Polygon

from napari_harpy.core.shapes_geometry import (
    napari_polygon_vertices_to_shapely_polygon,
    shapely_polygon_to_napari_polygon_vertices,
)


def test_napari_polygon_vertices_to_shapely_polygon_accepts_simple_polygon() -> None:
    vertices_yx = np.asarray(
        [
            [1.0, 2.0],
            [1.0, 5.0],
            [4.0, 5.0],
            [4.0, 2.0],
        ]
    )

    polygon = napari_polygon_vertices_to_shapely_polygon(vertices_yx)

    assert polygon.equals(Polygon([(2, 1), (5, 1), (5, 4), (2, 4)]))
    assert len(polygon.interiors) == 0


def test_napari_polygon_vertices_to_shapely_polygon_accepts_closed_simple_polygon() -> None:
    source = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])

    polygon = napari_polygon_vertices_to_shapely_polygon(shapely_polygon_to_napari_polygon_vertices(source))

    assert polygon.equals(source)
    assert len(polygon.interiors) == 0


def test_napari_polygon_vertices_to_shapely_polygon_preserves_one_hole() -> None:
    source = Polygon(
        [(0, 0), (8, 0), (8, 8), (0, 8)],
        holes=[[(2, 2), (2, 4), (4, 4), (4, 2)]],
    )

    polygon = napari_polygon_vertices_to_shapely_polygon(shapely_polygon_to_napari_polygon_vertices(source))

    assert polygon.equals(source)
    assert len(polygon.interiors) == 1
    assert polygon.area == source.area
    assert polygon.bounds == source.bounds


def test_napari_polygon_vertices_to_shapely_polygon_preserves_multiple_holes() -> None:
    source = Polygon(
        [(0, 0), (10, 0), (10, 10), (0, 10)],
        holes=[
            [(2, 2), (2, 4), (4, 4), (4, 2)],
            [(6, 6), (6, 8), (8, 8), (8, 6)],
        ],
    )

    polygon = napari_polygon_vertices_to_shapely_polygon(shapely_polygon_to_napari_polygon_vertices(source))

    assert polygon.equals(source)
    assert len(polygon.interiors) == 2
    assert polygon.area == source.area
    assert polygon.bounds == source.bounds


def test_napari_polygon_vertices_to_shapely_polygon_rejects_missing_hole_separator() -> None:
    source = Polygon(
        [(0, 0), (8, 0), (8, 8), (0, 8)],
        holes=[[(2, 2), (2, 4), (4, 4), (4, 2)]],
    )
    path = shapely_polygon_to_napari_polygon_vertices(source)[:-1]

    with pytest.raises(ValueError, match="path with holes must end on the exterior anchor"):
        napari_polygon_vertices_to_shapely_polygon(path)


def test_napari_polygon_vertices_to_shapely_polygon_rejects_open_hole_ring() -> None:
    path_yx = np.asarray(
        [
            [0.0, 0.0],
            [0.0, 8.0],
            [8.0, 8.0],
            [8.0, 0.0],
            [0.0, 0.0],
            [2.0, 2.0],
            [2.0, 4.0],
            [4.0, 4.0],
            [4.0, 2.0],
            [0.0, 0.0],
        ]
    )

    with pytest.raises(ValueError, match="each hole ring must be closed"):
        napari_polygon_vertices_to_shapely_polygon(path_yx)


def test_napari_polygon_vertices_to_shapely_polygon_rejects_nested_holes() -> None:
    path = shapely_polygon_to_napari_polygon_vertices(
        Polygon(
            [(0, 0), (10, 0), (10, 10), (0, 10)],
            holes=[
                [(2, 2), (2, 8), (8, 8), (8, 2)],
                [(4, 4), (4, 5), (5, 5), (5, 4)],
            ],
        )
    )

    with pytest.raises(ValueError, match="Nested polygon holes are not supported"):
        napari_polygon_vertices_to_shapely_polygon(path)


def test_napari_polygon_vertices_to_shapely_polygon_rejects_overlapping_holes() -> None:
    path = shapely_polygon_to_napari_polygon_vertices(
        Polygon(
            [(0, 0), (10, 0), (10, 10), (0, 10)],
            holes=[
                [(2, 2), (2, 6), (6, 6), (6, 2)],
                [(4, 4), (4, 8), (8, 8), (8, 4)],
            ],
        )
    )

    with pytest.raises(ValueError, match="Polygon holes must not overlap or share edges"):
        napari_polygon_vertices_to_shapely_polygon(path)


def test_napari_polygon_vertices_to_shapely_polygon_rejects_edge_touching_holes() -> None:
    path = shapely_polygon_to_napari_polygon_vertices(
        Polygon(
            [(0, 0), (10, 0), (10, 10), (0, 10)],
            holes=[
                [(2, 2), (2, 4), (4, 4), (4, 2)],
                [(4, 2), (4, 4), (6, 4), (6, 2)],
            ],
        )
    )

    with pytest.raises(ValueError, match="Polygon holes must not overlap or share edges"):
        napari_polygon_vertices_to_shapely_polygon(path)


def test_napari_polygon_vertices_to_shapely_polygon_allows_point_touching_holes() -> None:
    source = Polygon(
        [(0, 0), (10, 0), (10, 10), (0, 10)],
        holes=[
            [(2, 2), (2, 4), (4, 4), (4, 2)],
            [(4, 4), (4, 6), (6, 6), (6, 4)],
        ],
    )

    polygon = napari_polygon_vertices_to_shapely_polygon(shapely_polygon_to_napari_polygon_vertices(source))

    assert polygon.is_valid
    assert polygon.equals(source)
    assert len(polygon.interiors) == 2


def test_napari_polygon_vertices_to_shapely_polygon_rejects_holes_outside_shell() -> None:
    path = shapely_polygon_to_napari_polygon_vertices(
        Polygon(
            [(0, 0), (4, 0), (4, 4), (0, 4)],
            holes=[[(6, 6), (6, 8), (8, 8), (8, 6)]],
        )
    )

    with pytest.raises(ValueError, match="Polygon holes must be contained by the exterior ring"):
        napari_polygon_vertices_to_shapely_polygon(path)


def test_napari_polygon_vertices_to_shapely_polygon_rejects_hole_sharing_edge_with_shell() -> None:
    path = shapely_polygon_to_napari_polygon_vertices(
        Polygon(
            [(0, 0), (8, 0), (8, 8), (0, 8)],
            holes=[[(0, 2), (0, 4), (2, 4), (2, 2)]],
        )
    )

    with pytest.raises(ValueError, match="Polygon holes must not touch the exterior ring"):
        napari_polygon_vertices_to_shapely_polygon(path)


def test_napari_polygon_vertices_to_shapely_polygon_rejects_hole_touching_shell_at_point() -> None:
    path = shapely_polygon_to_napari_polygon_vertices(
        Polygon(
            [(0, 0), (8, 0), (8, 8), (0, 8)],
            holes=[[(0, 4), (2, 3), (2, 5)]],
        )
    )

    with pytest.raises(ValueError, match="Polygon holes must not touch the exterior ring"):
        napari_polygon_vertices_to_shapely_polygon(path)


@pytest.mark.parametrize(
    ("vertices", "message"),
    [
        ([[0.0, 0.0], [1.0, 1.0]], "too few vertices"),
        ([[0.0, 0.0], [1.0, np.nan], [1.0, 0.0]], "non-finite"),
        ([0.0, 1.0], "2D coordinates"),
    ],
)
def test_napari_polygon_vertices_to_shapely_polygon_rejects_malformed_vertices(vertices: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        napari_polygon_vertices_to_shapely_polygon(vertices)
