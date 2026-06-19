from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from shapely.errors import GEOSException
from shapely.geometry import Polygon
from shapely.geometry.polygon import orient


@dataclass(frozen=True)
class NapariPolygonTopology:
    shell_anchor_group: tuple[int, ...]
    hole_anchor_groups: tuple[tuple[int, ...], ...]

    @property
    def synchronized_anchor_groups(self) -> tuple[tuple[int, ...], ...]:
        if not self.shell_anchor_group:
            return self.hole_anchor_groups
        return (self.shell_anchor_group, *self.hole_anchor_groups)


@dataclass(frozen=True)
class _ParsedNapariPolygonVertices:
    shell: np.ndarray
    holes: tuple[np.ndarray, ...]
    topology: NapariPolygonTopology


def shapely_polygon_to_napari_polygon_vertices(polygon: Polygon) -> np.ndarray:
    """Encode a Shapely polygon as one napari polygon vertex row."""
    oriented = orient(polygon, sign=1.0)
    coordinates_xy = [_xy_coordinate(coord) for coord in oriented.exterior.coords]
    anchor = coordinates_xy[0]
    for interior in oriented.interiors:
        coordinates_xy.extend(_xy_coordinate(coord) for coord in interior.coords)
        coordinates_xy.append(anchor)
    return np.asarray([(y, x) for x, y in coordinates_xy], dtype=float)


def napari_polygon_vertices_to_topology(vertices: ArrayLike) -> NapariPolygonTopology:
    """Return synchronized anchor groups for one napari polygon vertex row."""
    parsed = _parse_napari_polygon_vertices(vertices)
    if parsed.holes:
        _make_valid_polygon_with_holes(parsed.shell, list(parsed.holes))
    return parsed.topology


def sync_napari_polygon_anchor_vertex(
    vertices: ArrayLike,
    topology: NapariPolygonTopology,
    moved_vertex_index: int,
    moved_coordinate: ArrayLike,
) -> np.ndarray:
    """Return vertices with the moved anchor copied to its synchronized group.

    ``NapariPolygonTopology`` stores raw indices into the napari vertex row.
    For a one-hole row encoded as ``A B C D A E F G H E A``, the synchronized
    groups are ``(0, 4, 10)`` for the exterior anchor copies and ``(5, 9)`` for
    the hole-anchor copies.

    If napari moves one member of such a group, this helper writes the moved
    coordinate to every index in that group. Moving exterior index ``4`` to
    ``A'`` therefore turns ``A B C D A' E F G H E A`` into
    ``A' B C D A' E F G H E A'``. Moving an ordinary non-anchor vertex, such as
    ``G`` at index ``7``, returns an unchanged copy.

    Topology groups are validated before synchronization so a vertex index
    cannot belong to multiple groups or point outside the vertex row.
    """
    vertices = _coerce_vertices(vertices)
    moved_vertex_index = _coerce_vertex_index(moved_vertex_index, vertex_count=len(vertices))
    moved_coordinate = _coerce_moved_coordinate(moved_coordinate)

    matched_group: tuple[int, ...] | None = None
    seen_indices: set[int] = set()
    for group in topology.synchronized_anchor_groups:
        normalized_group = _validated_anchor_group(group, vertex_count=len(vertices))
        overlap = seen_indices.intersection(normalized_group)
        if overlap:
            raise ValueError("Polygon topology anchor groups must not overlap.")
        seen_indices.update(normalized_group)
        if moved_vertex_index in normalized_group:
            matched_group = normalized_group

    synchronized = vertices.copy()
    if matched_group is not None:
        synchronized[list(matched_group)] = moved_coordinate
    return synchronized


def insert_napari_polygon_vertex(
    vertices: ArrayLike,
    topology: NapariPolygonTopology,
    insert_index: int,
    inserted_coordinate: ArrayLike,
) -> tuple[np.ndarray, NapariPolygonTopology]:
    """Return vertices and topology after inserting one ordinary ring vertex."""
    vertices = _coerce_vertices(vertices)
    insert_index = _coerce_insert_index(insert_index, vertex_count=len(vertices))
    inserted_coordinate = _coerce_inserted_coordinate(inserted_coordinate)
    shell_anchor_group, hole_anchor_groups = _validated_hole_topology(topology, vertex_count=len(vertices))

    shell_start, shell_end = shell_anchor_group[0], shell_anchor_group[1]
    is_real_ring_insert_index = shell_start < insert_index <= shell_end or any(
        hole_start < insert_index <= hole_end for hole_start, hole_end in hole_anchor_groups
    )
    if not is_real_ring_insert_index:
        raise ValueError("Inserted polygon vertex must split a shell or hole ring edge.")

    inserted_vertices = np.insert(vertices, insert_index, [inserted_coordinate], axis=0)
    # Compute the topology expected after insertion by shifting every existing
    # anchor/separator index at or after the insertion point.
    shifted_topology = NapariPolygonTopology(
        shell_anchor_group=tuple(
            _shift_index_after_insert(index, insert_index=insert_index) for index in shell_anchor_group
        ),
        hole_anchor_groups=tuple(
            tuple(_shift_index_after_insert(index, insert_index=insert_index) for index in group)
            for group in hole_anchor_groups
        ),
    )
    # Re-parse the updated row as a grammar check: parsed_topology is what the
    # new vertices actually encode.
    parsed_topology = napari_polygon_vertices_to_topology(inserted_vertices)
    if parsed_topology != shifted_topology:
        raise ValueError("Inserted polygon vertex produced ambiguous polygon topology.")
    return inserted_vertices, parsed_topology


def napari_polygon_vertices_to_shapely_polygon(vertices: ArrayLike) -> Polygon:
    """Decode one napari polygon vertex row into a Shapely polygon.

    The adapter encodes holes by closing the exterior ring, appending each
    closed interior ring, and repeating the exterior anchor after every hole.
    Vertex rows without that separator pattern are interpreted as simple
    polygons.
    """
    parsed = _parse_napari_polygon_vertices(vertices)
    if not parsed.holes:
        return _make_valid_polygon(parsed.shell)
    return _make_valid_polygon_with_holes(parsed.shell, list(parsed.holes))


def _parse_napari_polygon_vertices(vertices: ArrayLike) -> _ParsedNapariPolygonVertices:
    coordinates_yx = _coerce_vertices(vertices)
    if len(coordinates_yx) < 3:
        raise ValueError("Polygon path has too few vertices for a valid polygon.")

    coordinates_xy = coordinates_yx[:, [1, 0]]
    anchor = coordinates_xy[0]
    anchor_indices = _matching_coordinate_indices(coordinates_xy, anchor, start=1)
    simple_topology = NapariPolygonTopology(shell_anchor_group=(), hole_anchor_groups=())
    if not anchor_indices:
        return _ParsedNapariPolygonVertices(shell=coordinates_xy, holes=(), topology=simple_topology)

    shell_end = anchor_indices[0]
    shell = coordinates_xy[: shell_end + 1]
    if shell_end == len(coordinates_xy) - 1:
        return _ParsedNapariPolygonVertices(shell=shell, holes=(), topology=simple_topology)

    if shell_end < 3:
        raise ValueError("Malformed polygon hole encoding: exterior ring must contain at least four coordinates.")
    if not _same_coordinate(coordinates_xy[-1], anchor):
        raise ValueError("Malformed polygon hole encoding: path with holes must end on the exterior anchor.")

    holes: list[np.ndarray] = []
    hole_anchor_groups: list[tuple[int, ...]] = []
    chunk_start = shell_end + 1
    for separator_index in anchor_indices[1:]:
        hole = coordinates_xy[chunk_start:separator_index]
        _validate_hole_ring(hole)
        holes.append(hole)
        hole_anchor_groups.append((chunk_start, separator_index - 1))
        chunk_start = separator_index + 1

    if chunk_start != len(coordinates_xy):
        raise ValueError("Malformed polygon hole encoding: missing exterior-anchor separator after a hole ring.")

    topology = NapariPolygonTopology(
        shell_anchor_group=(0, *anchor_indices),
        hole_anchor_groups=tuple(hole_anchor_groups),
    )
    return _ParsedNapariPolygonVertices(shell=shell, holes=tuple(holes), topology=topology)


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


def _coerce_vertex_index(index: int, *, vertex_count: int) -> int:
    if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
        raise ValueError("Moved polygon vertex index must be an integer.")

    vertex_index = int(index)
    if vertex_index < 0 or vertex_index >= vertex_count:
        raise ValueError("Moved polygon vertex index is outside the vertex row.")
    return vertex_index


def _coerce_insert_index(index: int, *, vertex_count: int) -> int:
    if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
        raise ValueError("Inserted polygon vertex index must be an integer.")

    insert_index = int(index)
    if insert_index < 0 or insert_index >= vertex_count:
        raise ValueError("Inserted polygon vertex index is outside the vertex row.")
    return insert_index


def _coerce_moved_coordinate(coordinate: ArrayLike) -> np.ndarray:
    try:
        moved_coordinate = np.asarray(coordinate, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("Moved polygon vertex coordinate must contain numeric 2D coordinates.") from error

    if moved_coordinate.shape != (2,):
        raise ValueError("Moved polygon vertex coordinate must contain one 2D coordinate.")
    if not np.isfinite(moved_coordinate).all():
        raise ValueError("Moved polygon vertex coordinate contains non-finite coordinates.")
    return moved_coordinate


def _coerce_inserted_coordinate(coordinate: ArrayLike) -> np.ndarray:
    try:
        inserted_coordinate = np.asarray(coordinate, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("Inserted polygon vertex coordinate must contain numeric 2D coordinates.") from error

    if inserted_coordinate.shape != (2,):
        raise ValueError("Inserted polygon vertex coordinate must contain one 2D coordinate.")
    if not np.isfinite(inserted_coordinate).all():
        raise ValueError("Inserted polygon vertex coordinate contains non-finite coordinates.")
    return inserted_coordinate


def _validated_hole_topology(
    topology: NapariPolygonTopology,
    *,
    vertex_count: int,
) -> tuple[tuple[int, ...], tuple[tuple[int, int], ...]]:
    if len(topology.shell_anchor_group) < 2 or not topology.hole_anchor_groups:
        raise ValueError("Polygon topology must describe a hole-bearing vertex row.")

    shell_anchor_group = _validated_anchor_group(topology.shell_anchor_group, vertex_count=vertex_count)
    if shell_anchor_group[0] != 0:
        raise ValueError("Polygon topology shell anchor group must start at the first vertex.")
    if not _is_strictly_increasing(shell_anchor_group):
        raise ValueError("Polygon topology shell anchor group indices must be strictly increasing.")

    seen_indices = set(shell_anchor_group)
    hole_anchor_groups: list[tuple[int, int]] = []
    for group in topology.hole_anchor_groups:
        normalized_group = _validated_anchor_group(group, vertex_count=vertex_count)
        if len(normalized_group) != 2:
            raise ValueError("Polygon topology hole anchor groups must contain start and end indices.")
        if normalized_group[0] >= normalized_group[1]:
            raise ValueError("Polygon topology hole anchor group indices must be strictly increasing.")
        if seen_indices.intersection(normalized_group):
            raise ValueError("Polygon topology anchor groups must not overlap.")
        seen_indices.update(normalized_group)
        hole_start, hole_end = normalized_group
        hole_anchor_groups.append((hole_start, hole_end))
    return shell_anchor_group, tuple(hole_anchor_groups)


def _is_strictly_increasing(indices: Sequence[int]) -> bool:
    return all(left < right for left, right in zip(indices, indices[1:], strict=False))


def _shift_index_after_insert(index: int, *, insert_index: int) -> int:
    if index >= insert_index:
        return index + 1
    return index


def _validated_anchor_group(group: Sequence[int], *, vertex_count: int) -> tuple[int, ...]:
    if not group:
        raise ValueError("Polygon topology anchor groups must not be empty.")

    normalized_group: list[int] = []
    for index in group:
        if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
            raise ValueError("Polygon topology anchor group indices must be integers.")
        vertex_index = int(index)
        if vertex_index < 0 or vertex_index >= vertex_count:
            raise ValueError("Polygon topology anchor group references a vertex outside the row.")
        normalized_group.append(vertex_index)
    return tuple(normalized_group)


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
