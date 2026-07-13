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
    """Raw vertex-index topology for one napari polygon row.

    The napari row stores polygon holes by repeating anchor coordinates inside a
    single vertex array. For ``A B C D A E F G H E A``, the shell anchor
    copies are indices ``(0, 4, 10)`` and the hole-anchor copies are
    ``((5, 9),)``. These synchronized groups identify which raw vertex indices
    must move together when napari edits one anchor copy.

    The topology stores indices only, not coordinates. Simple polygons without
    encoded holes use empty shell and hole groups.
    """

    shell_anchor_group: tuple[int, ...]
    hole_anchor_groups: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        shell_anchor_group = _normalized_anchor_group(self.shell_anchor_group, allow_empty=True)
        try:
            hole_anchor_groups = tuple(self.hole_anchor_groups)
        except TypeError as error:
            raise ValueError("Polygon topology hole anchor groups must be sequences of anchor groups.") from error
        hole_anchor_groups = tuple(_normalized_anchor_group(group, allow_empty=False) for group in hole_anchor_groups)

        if not shell_anchor_group:
            if hole_anchor_groups:
                raise ValueError("Polygon topology cannot define hole anchors without shell anchors.")
            object.__setattr__(self, "shell_anchor_group", shell_anchor_group)
            object.__setattr__(self, "hole_anchor_groups", hole_anchor_groups)
            return

        if len(shell_anchor_group) < 2:
            raise ValueError("Polygon topology shell anchor group must contain at least two indices.")
        if shell_anchor_group[0] != 0:
            raise ValueError("Polygon topology shell anchor group must start at the first vertex.")
        if not _is_strictly_increasing(shell_anchor_group):
            raise ValueError("Polygon topology shell anchor group indices must be strictly increasing.")

        seen_indices = set(shell_anchor_group)
        for group in hole_anchor_groups:
            if len(group) != 2:
                raise ValueError("Polygon topology hole anchor groups must contain start and end indices.")
            if not _is_strictly_increasing(group):
                raise ValueError("Polygon topology hole anchor group indices must be strictly increasing.")
            if seen_indices.intersection(group):
                raise ValueError("Polygon topology anchor groups must not overlap.")
            seen_indices.update(group)

        object.__setattr__(self, "shell_anchor_group", shell_anchor_group)
        object.__setattr__(self, "hole_anchor_groups", hole_anchor_groups)

    @property
    def synchronized_anchor_groups(self) -> tuple[tuple[int, ...], ...]:
        if not self.shell_anchor_group:
            return self.hole_anchor_groups
        return (self.shell_anchor_group, *self.hole_anchor_groups)


@dataclass(frozen=True)
class NapariPolygonVertexDeletion:
    """Validated outcome of deleting one semantic polygon vertex.

    ``vertices`` and ``topology`` contain a shortened polygon candidate. Both
    are ``None`` when deleting from a semantic triangle must remove the whole
    polygon instead of constructing an invalid two-vertex row.
    """

    vertices: np.ndarray | None
    topology: NapariPolygonTopology | None

    def __post_init__(self) -> None:
        if (self.vertices is None) != (self.topology is None):
            raise ValueError("Polygon deletion vertices and topology must either both be present or both be absent.")

    @property
    def removes_shape(self) -> bool:
        return self.vertices is None


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


def create_polygon_with_direct_holes(shell: Polygon, holes: Sequence[Polygon]) -> Polygon:
    """Return ``shell`` with the child polygons added as direct holes.

    ``shell`` is the Shapely polygon that survives the operation. If it already
    has holes, those existing interiors are read from ``shell.interiors`` and
    preserved.

    ``holes`` contains only the new child polygons selected to become
    additional holes. The child polygons must be simple polygons without their
    own interiors; each child exterior ring is appended after any existing
    ``shell.interiors``.
    """
    if not isinstance(shell, Polygon) or shell.is_empty or not shell.is_valid or shell.area <= 0:
        raise ValueError("Shell polygon must be a non-empty valid polygon.")

    try:
        child_holes = tuple(holes)
    except TypeError as error:
        raise ValueError("Polygon holes must be a sequence of polygons.") from error

    hole_rings = [np.asarray(interior.coords, dtype=float) for interior in shell.interiors]
    for hole in child_holes:
        if not isinstance(hole, Polygon) or hole.is_empty or not hole.is_valid or hole.area <= 0:
            raise ValueError("Polygon holes must be non-empty valid polygons.")
        if len(hole.interiors):
            raise ValueError("Polygon holes must be simple polygons without interiors.")
        hole_rings.append(np.asarray(hole.exterior.coords, dtype=float))

    exterior_shell = np.asarray(shell.exterior.coords, dtype=float)
    return _make_valid_polygon_with_holes(exterior_shell, hole_rings)


def napari_polygon_vertices_to_topology(vertices: ArrayLike) -> NapariPolygonTopology:
    """Return synchronized anchor groups for one napari polygon vertex row."""
    return _parse_napari_polygon_vertices(vertices).topology


# Topology-preserving edit helpers for one napari polygon vertex row.
def sync_napari_polygon_anchor_vertex(
    vertices: ArrayLike,
    topology: NapariPolygonTopology,
    moved_vertex_index: int,
    moved_coordinate: ArrayLike,
) -> np.ndarray:
    """Return vertices with the moved anchor copied to its synchronized group.

    ``NapariPolygonTopology`` stores raw indices into the napari vertex row.
    For a one-hole row encoded as ``A B C D A E F G H E A``, the synchronized
    groups are ``(0, 4, 10)`` for the shell anchor copies and ``(5, 9)`` for
    the hole-anchor copies.

    If napari moves one member of such a group, this helper writes the moved
    coordinate to every index in that group. Moving shell-anchor index ``4`` to
    ``A'`` therefore turns ``A B C D A' E F G H E A`` into
    ``A' B C D A' E F G H E A'``. Moving an ordinary non-anchor vertex, such as
    ``G`` at index ``7``, returns an unchanged copy.

    The topology is not returned because it stores indices, not coordinates.
    Synchronizing a moved coordinate does not insert or remove vertices, so the
    anchor-group indices remain unchanged.

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
    # Validate explicitly because topology parsing only describes the encoded
    # row; it does not prove the row can be saved as a Shapely polygon.
    _ = napari_polygon_vertices_to_shapely_polygon(inserted_vertices)
    parsed_topology = napari_polygon_vertices_to_topology(inserted_vertices)
    if parsed_topology != shifted_topology:
        raise ValueError("Inserted polygon vertex produced ambiguous polygon topology.")
    return inserted_vertices, parsed_topology


def delete_napari_polygon_vertex(
    vertices: ArrayLike,
    topology: NapariPolygonTopology,
    deleted_vertex_index: int,
) -> NapariPolygonVertexDeletion:
    """Return the validated outcome of deleting one polygon vertex.

    Simple polygons preserve their implicit or explicit closure form. The
    first and last coordinates of an explicitly closed row are treated as
    aliases of one semantic vertex. Deleting from a semantic triangle reports
    whole-shape removal.

    Hole-bearing rows additionally synchronize encoded shell and hole anchors,
    shift separator indices, and remove a complete minimal hole when needed.
    """
    vertices = _coerce_vertices(vertices)
    deleted_vertex_index = _coerce_deleted_vertex_index(deleted_vertex_index, vertex_count=len(vertices))
    parsed_topology = napari_polygon_vertices_to_topology(vertices)
    if parsed_topology != topology:
        raise ValueError("Polygon topology does not match the encoded vertex row.")
    _ = napari_polygon_vertices_to_shapely_polygon(vertices)

    if not topology.hole_anchor_groups:
        return _delete_simple_napari_polygon_vertex(
            vertices,
            topology=topology,
            deleted_vertex_index=deleted_vertex_index,
        )

    # Parsing above has already proved the hole-bearing row grammar: the shell
    # and every hole are explicitly closed, every hole has a following shell
    # separator, and the complete row ends on the shell anchor. The private
    # branch can therefore operate on the validated anchor-group indices.
    deleted_vertices, deleted_topology = _delete_hole_bearing_napari_polygon_vertex(
        vertices,
        topology=topology,
        deleted_vertex_index=deleted_vertex_index,
    )
    return NapariPolygonVertexDeletion(vertices=deleted_vertices, topology=deleted_topology)


def _delete_simple_napari_polygon_vertex(
    vertices: np.ndarray,
    *,
    topology: NapariPolygonTopology,
    deleted_vertex_index: int,
) -> NapariPolygonVertexDeletion:
    explicitly_closed = bool(np.array_equal(vertices[0], vertices[-1]))
    semantic_vertex_count = len(vertices) - int(explicitly_closed)
    if semantic_vertex_count == 3:
        return NapariPolygonVertexDeletion(vertices=None, topology=None)
    if semantic_vertex_count < 3:
        raise ValueError("Polygon deletion requires at least three semantic vertices.")

    if explicitly_closed and deleted_vertex_index in {0, len(vertices) - 1}:
        semantic_vertices = np.delete(vertices[:-1], 0, axis=0)
        deleted_vertices = np.vstack([semantic_vertices, semantic_vertices[0]])
    else:
        deleted_vertices = np.delete(vertices, deleted_vertex_index, axis=0)

    deleted_topology = napari_polygon_vertices_to_topology(deleted_vertices)
    if deleted_topology != topology:
        raise ValueError("Polygon deletion changed the encoded topology.")
    _ = napari_polygon_vertices_to_shapely_polygon(deleted_vertices)
    return NapariPolygonVertexDeletion(vertices=deleted_vertices, topology=deleted_topology)


def _delete_hole_bearing_napari_polygon_vertex(
    vertices: np.ndarray,
    *,
    topology: NapariPolygonTopology,
    deleted_vertex_index: int,
) -> tuple[np.ndarray, NapariPolygonTopology]:
    shell_anchor_group, hole_anchor_groups = _validated_hole_topology(topology, vertex_count=len(vertices))

    if deleted_vertex_index in shell_anchor_group:
        return _delete_napari_polygon_shell_anchor(
            vertices,
            shell_anchor_group=shell_anchor_group,
            hole_anchor_groups=hole_anchor_groups,
        )

    for hole_index, (hole_start, hole_end) in enumerate(hole_anchor_groups):
        if deleted_vertex_index in (hole_start, hole_end):
            return _delete_napari_polygon_hole_anchor(
                vertices,
                shell_anchor_group=shell_anchor_group,
                hole_anchor_groups=hole_anchor_groups,
                hole_index=hole_index,
            )

    for hole_start, hole_end in hole_anchor_groups:
        if hole_start < deleted_vertex_index < hole_end and hole_end - hole_start + 1 == 4:
            return _delete_napari_polygon_hole(vertices, hole_start=hole_start, hole_end=hole_end)

    deleted_vertices = np.delete(vertices, deleted_vertex_index, axis=0)
    # Compute the topology expected after deletion by shifting every existing
    # anchor/separator index after the deleted vertex one slot to the left.
    shifted_topology = NapariPolygonTopology(
        shell_anchor_group=tuple(
            _shift_index_after_delete(index, deleted_vertex_index=deleted_vertex_index) for index in shell_anchor_group
        ),
        hole_anchor_groups=tuple(
            tuple(_shift_index_after_delete(index, deleted_vertex_index=deleted_vertex_index) for index in group)
            for group in hole_anchor_groups
        ),
    )
    # Re-parse the updated row as a grammar check: parsed_topology is what the
    # new vertices actually encode.
    # Validate explicitly because topology parsing only describes the encoded
    # row; it does not prove the row can be saved as a Shapely polygon.
    _ = napari_polygon_vertices_to_shapely_polygon(deleted_vertices)
    parsed_topology = napari_polygon_vertices_to_topology(deleted_vertices)
    if parsed_topology != shifted_topology:
        raise ValueError("Deleted polygon vertex produced ambiguous polygon topology.")
    return deleted_vertices, parsed_topology


def napari_polygon_vertices_to_shapely_polygon(vertices: ArrayLike) -> Polygon:
    """Decode one napari polygon vertex row into a Shapely polygon.

    The adapter encodes holes by closing the exterior ring, appending each
    closed interior ring, and repeating the shell anchor after every hole.
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
        raise ValueError("Malformed polygon hole encoding: path with holes must end on the shell anchor.")

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
        raise ValueError("Malformed polygon hole encoding: missing shell-anchor separator after a hole ring.")

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


def _coerce_deleted_vertex_index(index: int, *, vertex_count: int) -> int:
    if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
        raise ValueError("Deleted polygon vertex index must be an integer.")

    deleted_vertex_index = int(index)
    if deleted_vertex_index < 0 or deleted_vertex_index >= vertex_count:
        raise ValueError("Deleted polygon vertex index is outside the vertex row.")
    return deleted_vertex_index


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
    hole_anchor_groups: list[tuple[int, int]] = []
    for group in topology.hole_anchor_groups:
        normalized_group = _validated_anchor_group(group, vertex_count=vertex_count)
        hole_start, hole_end = normalized_group
        hole_anchor_groups.append((hole_start, hole_end))
    return shell_anchor_group, tuple(hole_anchor_groups)


def _is_strictly_increasing(indices: Sequence[int]) -> bool:
    return all(left < right for left, right in zip(indices, indices[1:], strict=False))


def _shift_index_after_insert(index: int, *, insert_index: int) -> int:
    if index >= insert_index:
        return index + 1
    return index


def _shift_index_after_delete(index: int, *, deleted_vertex_index: int) -> int:
    if index > deleted_vertex_index:
        return index - 1
    return index


def _delete_napari_polygon_hole(
    vertices: np.ndarray,
    *,
    hole_start: int,
    hole_end: int,
) -> tuple[np.ndarray, NapariPolygonTopology]:
    separator_index = hole_end + 1
    if separator_index >= len(vertices):
        raise ValueError("Polygon hole ring must be followed by a shell separator.")

    deleted_vertices = np.delete(vertices, np.arange(hole_start, separator_index + 1), axis=0)
    # Deliberately call the Shapely decoder as an early-failure geometry gate.
    # We discard the polygon because the caller needs napari vertices plus
    # freshly parsed topology, but invalid rebuilt rows should fail at the edit
    # operation instead of being accepted until save time.
    _ = napari_polygon_vertices_to_shapely_polygon(deleted_vertices)
    return deleted_vertices, napari_polygon_vertices_to_topology(deleted_vertices)


def _delete_napari_polygon_shell_anchor(
    vertices: np.ndarray,
    *,
    shell_anchor_group: tuple[int, ...],
    hole_anchor_groups: tuple[tuple[int, int], ...],
) -> tuple[np.ndarray, NapariPolygonTopology]:
    """Delete the logical shell anchor and rebuild with the next shell vertex.

    For ``A B C D A E F G H E A``, deleting any shell anchor/separator copy
    means deleting logical shell vertex ``A``. The rebuilt row uses ``B`` as
    the replacement shell anchor:

    ``A B C D A E F G H E A`` -> ``B C D B E F G H E B``.
    """
    shell_end = shell_anchor_group[1]
    shell_yx = vertices[:shell_end]
    shell_yx = shell_yx[1:]
    holes_yx = tuple(vertices[hole_start:hole_end] for hole_start, hole_end in hole_anchor_groups)
    rebuilt_vertices = _encode_napari_polygon_vertices_from_rings(shell_yx, holes_yx)
    # Deliberately call the Shapely decoder as an early-failure geometry gate.
    # We discard the polygon because the caller needs napari vertices plus
    # freshly parsed topology, but invalid rebuilt rows should fail at the edit
    # operation instead of being accepted until save time.
    _ = napari_polygon_vertices_to_shapely_polygon(rebuilt_vertices)
    return rebuilt_vertices, napari_polygon_vertices_to_topology(rebuilt_vertices)


def _delete_napari_polygon_hole_anchor(
    vertices: np.ndarray,
    *,
    shell_anchor_group: tuple[int, ...],
    hole_anchor_groups: tuple[tuple[int, int], ...],
    hole_index: int,
) -> tuple[np.ndarray, NapariPolygonTopology]:
    """Delete the logical hole anchor and rebuild that hole.

    For ``A B C D A E F G H E A``, deleting either hole-anchor copy means
    deleting logical hole vertex ``E``. The rebuilt row uses ``F`` as the
    replacement hole anchor:

    ``A B C D A E F G H E A`` -> ``A B C D A F G H F A``.

    For a minimal hole ``E F G E``, deleting ``E`` removes the entire hole
    instead of returning an invalid two-vertex ring.
    """
    hole_start, hole_end = hole_anchor_groups[hole_index]
    if hole_end - hole_start + 1 == 4:
        return _delete_napari_polygon_hole(vertices, hole_start=hole_start, hole_end=hole_end)

    shell_end = shell_anchor_group[1]
    shell_yx = vertices[:shell_end]
    holes_yx = [vertices[start:end] for start, end in hole_anchor_groups]
    holes_yx[hole_index] = holes_yx[hole_index][1:]
    rebuilt_vertices = _encode_napari_polygon_vertices_from_rings(shell_yx, tuple(holes_yx))
    # Deliberately call the Shapely decoder as an early-failure geometry gate.
    # We discard the polygon because the caller needs napari vertices plus
    # freshly parsed topology, but invalid rebuilt rows should fail at the edit
    # operation instead of being accepted until save time.
    _ = napari_polygon_vertices_to_shapely_polygon(rebuilt_vertices)
    return rebuilt_vertices, napari_polygon_vertices_to_topology(rebuilt_vertices)


def _encode_napari_polygon_vertices_from_rings(
    shell_yx: ArrayLike,
    holes_yx: tuple[ArrayLike, ...],
) -> np.ndarray:
    """Encode unclosed napari rings as one polygon vertex row.

    Inputs are unclosed logical rings in napari ``(y, x)`` order. The shell
    input is ``A B C D``, not the already-closed ring ``A B C D A``. The holes
    input contains rings such as ``E F G H``, not already-closed rings such as
    ``E F G H E``. For shell ``A B C D`` and hole ``E F G H``, this helper returns
    ``A B C D A E F G H E A``: close the shell, close each hole, and append the
    shell anchor after each hole as the shell separator.

    With multiple holes, each hole is appended the same way. Shell ``A B C D``
    with holes ``E F G H`` and ``I J K L`` becomes
    ``A B C D A E F G H E A I J K L I A``.
    """
    shell_yx = _coerce_vertices(shell_yx)
    if len(shell_yx) < 3:
        raise ValueError("Polygon shell must contain at least three vertices.")

    encoded_parts = [shell_yx, shell_yx[:1]]
    for hole_yx in holes_yx:
        hole_yx = _coerce_vertices(hole_yx)
        if len(hole_yx) < 3:
            raise ValueError("Polygon holes must contain at least three vertices.")
        encoded_parts.extend([hole_yx, hole_yx[:1], shell_yx[:1]])
    return np.concatenate(encoded_parts, axis=0)


def _normalized_anchor_group(group: Sequence[int], *, allow_empty: bool) -> tuple[int, ...]:
    try:
        indices = tuple(group)
    except TypeError as error:
        raise ValueError("Polygon topology anchor groups must be sequences of indices.") from error

    if not indices and not allow_empty:
        raise ValueError("Polygon topology anchor groups must not be empty.")

    normalized_indices: list[int] = []
    for index in indices:
        if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
            raise ValueError("Polygon topology anchor group indices must be integers.")
        vertex_index = int(index)
        if vertex_index < 0:
            raise ValueError("Polygon topology anchor group indices must be non-negative.")
        normalized_indices.append(vertex_index)
    return tuple(normalized_indices)


def _validated_anchor_group(group: Sequence[int], *, vertex_count: int) -> tuple[int, ...]:
    validated_group: list[int] = []
    for index in group:
        if index >= vertex_count:
            raise ValueError("Polygon topology anchor group references a vertex outside the row.")
        validated_group.append(index)
    return tuple(validated_group)


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
        # The napari path encoding reserves the shell anchor as a ring separator.
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
