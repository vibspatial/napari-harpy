from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest
from napari.layers import Shapes
from napari.settings import get_settings
from napari.utils.triangulation_backend import TriangulationBackend, get_backend, set_backend
from shapely.geometry import Polygon
from shapely.ops import unary_union

from napari_harpy._shapes_triangulation import ensure_shapes_triangulation_backend
from napari_harpy.core.shapes_geometry import (
    napari_polygon_vertices_to_shapely_polygon,
    shapely_polygon_to_napari_polygon_vertices,
)

POLYGON_WITH_HOLES_TRIANGULATION_FIXTURE = Polygon(
    shell=[
        (1883.543213, 2352.524414),
        (2182.454590, 2429.829102),
        (2208.222656, 2496.826416),
        (2208.222656, 2615.360107),
        (2177.300781, 2713.279297),
        (2120.610840, 2780.276855),
        (2063.920654, 2821.505859),
        (1966.001465, 2852.427734),
        (1899.004150, 2852.427734),
        (1821.699463, 2831.813232),
        (1775.316772, 2800.891357),
        (1713.473022, 2697.818359),
        (1698.012085, 2620.513916),
        (1698.012085, 2543.209229),
        (1734.087524, 2460.750977),
        (1832.006836, 2373.138916),
        (1883.543213, 2352.524414),
    ],
    holes=[
        [
            (1841.824951, 2505.260742),
            (1814.584351, 2521.605225),
            (1790.067871, 2548.845703),
            (1803.688110, 2581.534424),
            (1847.273071, 2589.706543),
            (1877.237793, 2581.534424),
            (1888.134033, 2543.397705),
            (1841.824951, 2505.260742),
        ],
        [
            (2037.957397, 2584.258545),
            (2010.716797, 2600.602783),
            (1988.924316, 2630.567627),
            (2002.544556, 2660.532227),
            (2037.957397, 2663.256348),
            (2067.922119, 2636.015625),
            (2037.957397, 2584.258545),
        ],
        [
            (2007.992676, 2502.536621),
            (2016.164917, 2540.673584),
            (2054.301758, 2540.673584),
            (2081.542236, 2513.432861),
            (2086.990479, 2486.192383),
            (2007.992676, 2502.536621),
        ],
    ],
)


@pytest.fixture
def restore_triangulation_backend() -> Iterator[None]:
    settings = get_settings()
    previous_settings_backend = settings.experimental.triangulation_backend
    previous_runtime_backend = get_backend()

    try:
        yield
    finally:
        settings.experimental.triangulation_backend = previous_settings_backend
        if get_backend() != previous_runtime_backend:
            set_backend(previous_runtime_backend)


def _shape_mesh_metrics(layer: Shapes, expected: Polygon) -> tuple[str | None, float, float]:
    shape = layer._data_view.shapes[0]
    face_vertices = np.asarray(shape._face_vertices, dtype=float)
    face_triangles = np.asarray(shape._face_triangles, dtype=int)
    triangle_polygons = [Polygon(face_vertices[triangle][:, ::-1]) for triangle in face_triangles if len(triangle) == 3]
    triangle_polygons = [polygon for polygon in triangle_polygons if polygon.area > 0]
    triangle_union = unary_union(triangle_polygons)
    triangle_area_sum = sum(polygon.area for polygon in triangle_polygons)
    overdraw = triangle_area_sum - triangle_union.area
    symmetric_difference = triangle_union.symmetric_difference(expected).area
    mesh_path = getattr(shape._set_meshes, "__name__", None)
    return mesh_path, overdraw, symmetric_difference


def test_shapes_triangulation_backend_helper_makes_annotation_1_fixture_use_numba_mesh(
    restore_triangulation_backend: None,
) -> None:
    settings = get_settings()
    settings.experimental.triangulation_backend = TriangulationBackend.fastest_available
    set_backend(TriangulationBackend.fastest_available)

    ensure_shapes_triangulation_backend()
    vertices = shapely_polygon_to_napari_polygon_vertices(POLYGON_WITH_HOLES_TRIANGULATION_FIXTURE)
    layer = Shapes([vertices], shape_type="polygon")
    expected = napari_polygon_vertices_to_shapely_polygon(layer.data[0])
    mesh_path, overdraw, symmetric_difference = _shape_mesh_metrics(layer, expected)

    assert get_backend() == TriangulationBackend.numba
    assert mesh_path == "_set_meshes_py"
    assert overdraw == 0
    assert symmetric_difference == 0


def test_bermuda_triangulation_backend_overdraws_annotation_1_fixture(
    restore_triangulation_backend: None,
) -> None:
    settings = get_settings()
    settings.experimental.triangulation_backend = TriangulationBackend.bermuda
    set_backend(TriangulationBackend.bermuda)

    vertices = shapely_polygon_to_napari_polygon_vertices(POLYGON_WITH_HOLES_TRIANGULATION_FIXTURE)
    layer = Shapes([vertices], shape_type="polygon")
    expected = napari_polygon_vertices_to_shapely_polygon(layer.data[0])
    mesh_path, overdraw, symmetric_difference = _shape_mesh_metrics(layer, expected)
    tolerance = expected.area * 1e-6

    assert get_backend() == TriangulationBackend.bermuda
    assert mesh_path == "_set_meshes_compiled_bermuda"
    assert overdraw > tolerance
    assert symmetric_difference > tolerance
