from __future__ import annotations

import numpy as np
import pytest
from _shapes_regression_fixtures import (
    POLYGON_WITH_HOLES_TRIANGULATION_FIXTURE_1,
    POLYGON_WITH_HOLES_TRIANGULATION_FIXTURE_2,
)
from napari.layers import Shapes
from napari.settings import get_settings
from napari.utils.triangulation_backend import TriangulationBackend, get_backend, set_backend
from shapely.geometry import Polygon
from shapely.ops import unary_union

from napari_harpy._shapes_triangulation import (
    configure_shapes_triangulation_backend,
    ensure_shapes_triangulation_backend,
)
from napari_harpy.core.shapes_geometry import (
    napari_polygon_vertices_to_shapely_polygon,
    shapely_polygon_to_napari_polygon_vertices,
)
from napari_harpy.viewer.adapter import _build_shapes_layer

TRIANGULATION_REGRESSION_POLYGONS = (
    pytest.param(POLYGON_WITH_HOLES_TRIANGULATION_FIXTURE_1, id="annotation_1"),
    pytest.param(POLYGON_WITH_HOLES_TRIANGULATION_FIXTURE_2, id="annotation_2"),
)


def _shape_mesh_metrics(layer: Shapes, expected: Polygon) -> tuple[str | None, float, float]:
    """Return napari's mesh path plus triangle overlap and coverage error.

    Napari renders a filled polygon by triangulating it into face vertices and
    face triangles. ``overdraw`` compares the summed area of those triangles
    with the area of their union; a positive value means generated triangles
    overlap each other. ``symmetric_difference`` compares that triangle union
    with the expected polygon, catching holes or filled regions that render in
    the wrong place.
    """
    shape = layer._data_view.shapes[0]
    face_vertices = np.asarray(shape._face_vertices, dtype=float)
    face_triangles = np.asarray(shape._face_triangles, dtype=int)
    # Reconstruct each rendered mesh triangle as a Shapely polygon. Napari
    # stores coordinates in (y, x) order, so flip them back to Shapely's (x, y).
    triangle_polygons = [Polygon(face_vertices[triangle][:, ::-1]) for triangle in face_triangles if len(triangle) == 3]
    # Drop degenerate zero-area triangles before computing rendered coverage.
    triangle_polygons = [polygon for polygon in triangle_polygons if polygon.area > 0]
    # Compare the summed triangle area with the union area (where overlapping
    # regions count only once) to detect overdraw from intersecting mesh triangles.
    triangle_union = unary_union(triangle_polygons)
    triangle_area_sum = sum(polygon.area for polygon in triangle_polygons)
    overdraw = triangle_area_sum - triangle_union.area
    symmetric_difference = triangle_union.symmetric_difference(expected).area
    mesh_path = getattr(shape._set_meshes, "__name__", None)
    return mesh_path, overdraw, symmetric_difference


def test_harpy_shapes_construction_without_interactive_uses_default_bermuda(
    restore_triangulation_backend: None,
    sdata_blobs,
) -> None:
    """Cover the viewer-adapter path used after napari's CLI reader launch."""
    settings = get_settings()
    settings.experimental.triangulation_backend = TriangulationBackend.numba
    set_backend(TriangulationBackend.numba)

    built_layer = _build_shapes_layer(
        sdata_blobs,
        "blobs_polygons",
        "global",
        name="blobs_polygons",
    )

    shape = built_layer.layer._data_view.shapes[0]
    assert settings.experimental.triangulation_backend == TriangulationBackend.bermuda
    assert get_backend() == TriangulationBackend.bermuda
    assert shape._set_meshes.__name__ == "_set_meshes_compiled_bermuda"


@pytest.mark.parametrize("polygon", TRIANGULATION_REGRESSION_POLYGONS)
def test_shapes_triangulation_backend_helper_preserves_configured_numba_backend(
    restore_triangulation_backend: None,
    polygon: Polygon,
) -> None:
    configure_shapes_triangulation_backend("numba")
    settings = get_settings()
    settings.experimental.triangulation_backend = TriangulationBackend.fastest_available
    set_backend(TriangulationBackend.fastest_available)

    ensure_shapes_triangulation_backend()
    vertices = shapely_polygon_to_napari_polygon_vertices(polygon)
    layer = Shapes([vertices], shape_type="polygon")
    expected = napari_polygon_vertices_to_shapely_polygon(layer.data[0])
    mesh_path, overdraw, symmetric_difference = _shape_mesh_metrics(layer, expected)

    assert get_backend() == TriangulationBackend.numba
    assert mesh_path == "_set_meshes_py"
    assert overdraw == 0
    assert symmetric_difference == 0


@pytest.mark.parametrize("polygon", TRIANGULATION_REGRESSION_POLYGONS)
def test_shapes_triangulation_backend_helper_preserves_configured_bermuda_backend(
    restore_triangulation_backend: None,
    polygon: Polygon,
) -> None:
    configure_shapes_triangulation_backend("bermuda")
    settings = get_settings()
    settings.experimental.triangulation_backend = TriangulationBackend.fastest_available
    set_backend(TriangulationBackend.fastest_available)

    ensure_shapes_triangulation_backend()
    vertices = shapely_polygon_to_napari_polygon_vertices(polygon)
    layer = Shapes([vertices], shape_type="polygon")
    expected = napari_polygon_vertices_to_shapely_polygon(layer.data[0])
    mesh_path, overdraw, symmetric_difference = _shape_mesh_metrics(layer, expected)
    tolerance = expected.area * 1e-6

    assert get_backend() == TriangulationBackend.bermuda
    assert mesh_path == "_set_meshes_compiled_bermuda"
    assert overdraw > tolerance
    assert symmetric_difference > tolerance


def test_shapes_triangulation_backend_rejects_unsupported_backend(
    restore_triangulation_backend: None,
) -> None:
    configure_shapes_triangulation_backend("bermuda")

    with pytest.raises(ValueError, match="Unknown Shapes triangulation backend.*bermuda, numba"):
        configure_shapes_triangulation_backend("fastest_available")  # type: ignore[arg-type]

    assert get_settings().experimental.triangulation_backend == TriangulationBackend.bermuda
    assert get_backend() == TriangulationBackend.bermuda
