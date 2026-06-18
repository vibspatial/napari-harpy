from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike
from shapely.errors import GEOSException
from shapely.geometry import Polygon
from shapely.geometry.polygon import orient


def polygon_to_napari_path(polygon: Polygon) -> np.ndarray:
    """Encode a Shapely polygon as one napari path, preserving direct holes."""
    oriented = orient(polygon, sign=1.0)
    path = [_xy_coordinate(coord) for coord in oriented.exterior.coords]
    anchor = path[0]
    for interior in oriented.interiors:
        path.extend(_xy_coordinate(coord) for coord in interior.coords)
        path.append(anchor)
    return np.asarray([(y, x) for x, y in path], dtype=float)


def napari_path_to_polygon(vertices: ArrayLike) -> Polygon:
    """Decode one napari polygon path into a Shapely polygon.

    The adapter encodes holes by closing the exterior ring, appending each
    closed interior ring, and repeating the exterior anchor after every hole.
    Paths without that separator pattern are interpreted as simple polygons.
    """
    coordinates_yx = _coerce_vertices(vertices)
    if len(coordinates_yx) < 3:
        raise ValueError("Polygon path has too few vertices for a valid polygon.")

    coordinates_xy = coordinates_yx[:, [1, 0]]
    anchor = coordinates_xy[0]
    anchor_indices = _matching_coordinate_indices(coordinates_xy, anchor, start=1)
    if not anchor_indices:
        return _make_valid_polygon(coordinates_xy)

    shell_end = anchor_indices[0]
    shell = coordinates_xy[: shell_end + 1]
    if shell_end == len(coordinates_xy) - 1:
        return _make_valid_polygon(shell)

    if shell_end < 3:
        raise ValueError("Malformed polygon hole encoding: exterior ring must contain at least four coordinates.")
    if not _same_coordinate(coordinates_xy[-1], anchor):
        raise ValueError("Malformed polygon hole encoding: path with holes must end on the exterior anchor.")

    holes: list[np.ndarray] = []
    chunk_start = shell_end + 1
    for separator_index in anchor_indices[1:]:
        hole = coordinates_xy[chunk_start:separator_index]
        _validate_hole_ring(hole)
        holes.append(hole)
        chunk_start = separator_index + 1

    if chunk_start != len(coordinates_xy):
        raise ValueError("Malformed polygon hole encoding: missing exterior-anchor separator after a hole ring.")

    return _make_valid_polygon_with_holes(shell, holes)


def _coerce_vertices(vertices: ArrayLike) -> np.ndarray:
    try:
        coordinates = np.asarray(vertices, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("Polygon path must contain numeric 2D coordinates.") from error

    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("Polygon path must contain 2D coordinates.")
    if coordinates.size == 0:
        raise ValueError("Polygon path is empty.")
    if not np.isfinite(coordinates).all():
        raise ValueError("Polygon path contains non-finite coordinates.")
    return coordinates


def _matching_coordinate_indices(coordinates: np.ndarray, target: np.ndarray, *, start: int) -> list[int]:
    return [index for index in range(start, len(coordinates)) if _same_coordinate(coordinates[index], target)]


def _same_coordinate(left: np.ndarray, right: np.ndarray) -> bool:
    return bool(np.array_equal(left, right))


def _validate_hole_ring(hole: np.ndarray) -> None:
    if len(hole) < 4:
        raise ValueError("Malformed polygon hole encoding: each hole ring must contain at least four coordinates.")
    if not _same_coordinate(hole[0], hole[-1]):
        raise ValueError("Malformed polygon hole encoding: each hole ring must be closed before the next separator.")


def _make_valid_polygon(coordinates_xy: np.ndarray) -> Polygon:
    try:
        polygon = Polygon(coordinates_xy)
    except (GEOSException, ValueError) as error:
        raise ValueError("Polygon path cannot be converted to a valid polygon.") from error

    if polygon.is_empty or not polygon.is_valid or polygon.area <= 0:
        raise ValueError("Polygon path cannot be converted to a valid polygon.")
    return polygon


def _make_valid_polygon_with_holes(shell: np.ndarray, holes: list[np.ndarray]) -> Polygon:
    shell_polygon = _make_valid_polygon(shell)
    hole_polygons = [_make_valid_hole_polygon(hole) for hole in holes]
    _validate_direct_holes(shell_polygon, hole_polygons)

    try:
        polygon = Polygon(shell, holes=holes)
    except (GEOSException, ValueError) as error:
        raise ValueError("Polygon path cannot be converted to a valid polygon with holes.") from error

    if polygon.is_empty or not polygon.is_valid or polygon.area <= 0:
        raise ValueError("Polygon path cannot be converted to a valid polygon with holes.")
    return polygon


def _make_valid_hole_polygon(hole: np.ndarray) -> Polygon:
    try:
        polygon = Polygon(hole)
    except (GEOSException, ValueError) as error:
        raise ValueError("Polygon hole ring cannot be converted to a valid polygon.") from error

    if polygon.is_empty or not polygon.is_valid or polygon.area <= 0:
        raise ValueError("Polygon hole ring cannot be converted to a valid polygon.")
    return polygon


def _validate_direct_holes(shell: Polygon, holes: list[Polygon]) -> None:
    """Validate one-level interior rings for ``Polygon(shell, holes=...)``.

    Supported topology is one exterior shell with direct sibling holes:

        +-------------+
        |  +---+      |
        |  | A | +--+ |
        |  +---+ |B | |
        |        +--+ |
        +-------------+

    Point-only contact between sibling holes is allowed because Shapely treats
    it as a valid polygon topology. In that case the pairwise intersection is a
    ``Point`` or ``MultiPoint``:

        +---+
        | A |
        +---+---+
            | B |
            +---+

    We reject topology that would need interpretation beyond one direct Shapely
    polygon with holes:

    - holes outside, crossing the shell, or touching the shell
    - nested holes / islands-in-holes
    - overlapping holes
    - holes that share an edge, whose intersection is line-like

      Outside/crossing/touching shell:

        +-------------+        +-------------+        +-------------+
        |             |        |         +---+---+    *--A         |
        |             |        |         | A     |    |            |
        +-------------+        +---------+---+---+    +----+-------+
             +---+
             | A |
             +---+

      Nested holes / islands-in-holes:

        +-----------+
        | A         |
        |  +---+    |
        |  | B |    |
        |  +---+    |
        +-----------+

      Overlapping holes:

        +------+
        | A    |
        |   +--+---+
        +---+ B    |
            +------+

      Edge-sharing holes:

        +---++---+
        | A || B |
        +---++---+
    """
    for hole in holes:
        if not shell.contains(hole):
            raise ValueError("Polygon holes must be contained by the exterior ring.")
        # The napari path encoding reserves the exterior anchor as a ring separator.
        # Shell-touching holes can put hole vertices on the exterior boundary; if that
        # point is the anchor, the path grammar becomes ambiguous. Keep the contract
        # stricter by requiring holes to be fully inside the shell.
        shell_intersection = shell.boundary.intersection(hole.boundary)
        if not shell_intersection.is_empty:
            raise ValueError("Polygon holes must not touch the exterior ring.")

    for index, left in enumerate(holes):
        for right in holes[index + 1 :]:
            if left.contains(right) or right.contains(left):
                raise ValueError("Nested polygon holes are not supported.")
            intersection = left.intersection(right)
            if not intersection.is_empty and intersection.geom_type not in {"Point", "MultiPoint"}:
                raise ValueError("Polygon holes must not overlap or share edges.")


def _xy_coordinate(coordinate: Sequence[float]) -> tuple[float, float]:
    return float(coordinate[0]), float(coordinate[1])
