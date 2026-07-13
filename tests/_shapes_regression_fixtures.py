from __future__ import annotations

import numpy as np
from shapely.geometry import Polygon

from napari_harpy.core.shapes_geometry import (
    napari_polygon_vertices_to_shapely_polygon,
    napari_polygon_vertices_to_topology,
    shapely_polygon_to_napari_polygon_vertices,
)

POLYGON_WITH_HOLES_TRIANGULATION_FIXTURE_1 = Polygon(
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

POLYGON_WITH_HOLES_TRIANGULATION_FIXTURE_2 = Polygon(
    shell=[
        (1.855311, 0.000000),
        (5.102106, 1.443020),
        (2.679894, 4.999033),
        (0.000000, 2.679895),
        (1.855311, 0.000000),
    ],
    holes=[
        [
            (3.699100, 2.834912),
            (2.909122, 2.780432),
            (3.045325, 3.080078),
            (3.699100, 2.834912),
        ],
        [
            (3.099806, 1.500122),
            (3.562897, 1.881492),
            (3.889784, 1.336680),
            (3.099806, 1.500122),
        ],
    ],
)

TRIANGULATION_REGRESSION_SHELL_ANCHOR_INDICES = (0, 16, 25, 33, 40)
TRIANGULATION_REGRESSION_HOLE_ANCHOR_INDICES = (34, 39)
TRIANGULATION_REGRESSION_MOVED_VERTEX_INDEX = 39
TRIANGULATION_REGRESSION_NEARBY_SHELL_VERTEX_INDEX = 40

# Recorded coordinates use napari's raw (y, x) row order.
# P: shared coordinate of every shell-anchor copy before the failing move.
TRIANGULATION_REGRESSION_SHELL_ANCHOR_COORDINATE = (
    2350.921630859375,
    1882.07861328125,
)
# Q: shared coordinate of third-hole-anchor copies 34 and 39 before the move.
TRIANGULATION_REGRESSION_HOLE_ANCHOR_COORDINATE = (
    2350.9228515625,
    1882.0780029296875,
)
# Q': coordinate written only to third-hole-anchor copy 39 by the failing move.
TRIANGULATION_REGRESSION_FAILED_MOVE_COORDINATE = (
    2350.921630859375,
    1882.0780029296875,
)

CONCAVE_SIMPLE_POLYGON_DELETION_REGRESSION_VERTICES = (
    (0.0, 0.0),
    (4.0, 0.0),
    (4.0, 4.0),
    (3.0, 4.0),
    (3.0, 1.0),
    (1.0, 1.0),
    (1.0, 4.0),
    (0.0, 4.0),
)
CONCAVE_SIMPLE_POLYGON_DELETION_REGRESSION_VERTEX_INDEX = 0


def make_triangulation_regression_pre_drag_vertices() -> np.ndarray:
    """Build the valid napari row used as the failing drag's starting state.

    ``POLYGON_WITH_HOLES_TRIANGULATION_FIXTURE_1`` is the hardcoded source
    Shapely polygon. This function first encodes that polygon as one 41-row
    napari vertex array. It does not add or remove vertices. It then changes
    only the coordinates of these existing duplicated anchors:

    - shell-anchor copies ``(0, 16, 25, 33, 40)`` all receive
      ``P = (2350.921630859375, 1882.07861328125)``;
    - third-hole-anchor copies ``(34, 39)`` both receive
      ``Q = (2350.9228515625, 1882.0780029296875)``.

    Coordinates are in napari ``(y, x)`` order. Because every alias is changed
    together, the returned pre-drag row remains synchronized and geometrically
    valid.

    The failing move is not performed here. The characterization test later
    changes only vertex ``39`` from ``Q`` to
    ``Q' = TRIANGULATION_REGRESSION_FAILED_MOVE_COORDINATE =
    (2350.921630859375, 1882.0780029296875)``. Vertex ``34`` remains at ``Q``;
    that temporary mismatch is the native triangulation regression.
    """
    vertices = shapely_polygon_to_napari_polygon_vertices(POLYGON_WITH_HOLES_TRIANGULATION_FIXTURE_1)
    topology = napari_polygon_vertices_to_topology(vertices)
    if topology.shell_anchor_group != TRIANGULATION_REGRESSION_SHELL_ANCHOR_INDICES:
        raise AssertionError("Unexpected shell-anchor topology in the triangulation regression fixture.")
    if topology.hole_anchor_groups[-1] != TRIANGULATION_REGRESSION_HOLE_ANCHOR_INDICES:
        raise AssertionError("Unexpected hole-anchor topology in the triangulation regression fixture.")

    vertices[list(TRIANGULATION_REGRESSION_SHELL_ANCHOR_INDICES)] = TRIANGULATION_REGRESSION_SHELL_ANCHOR_COORDINATE
    vertices[list(TRIANGULATION_REGRESSION_HOLE_ANCHOR_INDICES)] = TRIANGULATION_REGRESSION_HOLE_ANCHOR_COORDINATE
    if len(vertices) != 41:
        raise AssertionError("Unexpected vertex count in the triangulation regression fixture.")
    _ = napari_polygon_vertices_to_shapely_polygon(vertices)
    return vertices


def make_concave_simple_polygon_deletion_regression_vertices() -> np.ndarray:
    """Return a valid row whose native index-0 deletion is invalid."""
    vertices = np.asarray(CONCAVE_SIMPLE_POLYGON_DELETION_REGRESSION_VERTICES, dtype=float)
    _ = napari_polygon_vertices_to_shapely_polygon(vertices)
    return vertices
