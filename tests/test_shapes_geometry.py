from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import Polygon

from napari_harpy.core.shapes_geometry import (
    NapariPolygonTopology,
    delete_napari_polygon_vertex,
    insert_napari_polygon_vertex,
    napari_polygon_vertices_to_shapely_polygon,
    napari_polygon_vertices_to_topology,
    shapely_polygon_to_napari_polygon_vertices,
    sync_napari_polygon_anchor_vertex,
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


def test_napari_polygon_vertices_to_topology_returns_no_groups_for_simple_polygon() -> None:
    vertices_yx = np.asarray(
        [
            [1.0, 2.0],
            [1.0, 5.0],
            [4.0, 5.0],
            [4.0, 2.0],
        ]
    )

    topology = napari_polygon_vertices_to_topology(vertices_yx)

    assert topology == NapariPolygonTopology(shell_anchor_group=(), hole_anchor_groups=())
    assert topology.synchronized_anchor_groups == ()


def test_napari_polygon_vertices_to_topology_returns_no_groups_for_closed_simple_polygon() -> None:
    source = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])

    topology = napari_polygon_vertices_to_topology(shapely_polygon_to_napari_polygon_vertices(source))

    assert topology == NapariPolygonTopology(shell_anchor_group=(), hole_anchor_groups=())
    assert topology.synchronized_anchor_groups == ()


def test_napari_polygon_vertices_to_topology_distinguishes_shell_and_one_hole_anchors() -> None:
    source = Polygon(
        [(0, 0), (8, 0), (8, 8), (0, 8)],
        holes=[[(2, 2), (2, 4), (4, 4), (4, 2)]],
    )

    topology = napari_polygon_vertices_to_topology(shapely_polygon_to_napari_polygon_vertices(source))

    assert topology == NapariPolygonTopology(
        shell_anchor_group=(0, 4, 10),
        hole_anchor_groups=((5, 9),),
    )
    assert topology.synchronized_anchor_groups == ((0, 4, 10), (5, 9))


def test_napari_polygon_vertices_to_topology_distinguishes_multiple_hole_anchors() -> None:
    source = Polygon(
        [(0, 0), (10, 0), (10, 10), (0, 10)],
        holes=[
            [(2, 2), (2, 4), (4, 4), (4, 2)],
            [(6, 6), (6, 8), (8, 8), (8, 6)],
        ],
    )

    topology = napari_polygon_vertices_to_topology(shapely_polygon_to_napari_polygon_vertices(source))

    assert topology == NapariPolygonTopology(
        shell_anchor_group=(0, 4, 10, 16),
        hole_anchor_groups=((5, 9), (11, 15)),
    )
    assert topology.synchronized_anchor_groups == ((0, 4, 10, 16), (5, 9), (11, 15))


def test_napari_polygon_topology_normalizes_anchor_groups() -> None:
    topology = NapariPolygonTopology(
        shell_anchor_group=[0, np.int64(4), 10],
        hole_anchor_groups=[[5, 9]],
    )

    assert topology.shell_anchor_group == (0, 4, 10)
    assert topology.hole_anchor_groups == ((5, 9),)
    assert topology.synchronized_anchor_groups == ((0, 4, 10), (5, 9))


@pytest.mark.parametrize(
    ("shell_anchor_group", "hole_anchor_groups", "match"),
    [
        ((0,), (), "at least two"),
        ((1, 4), (), "first vertex"),
        ((0, 4, 4), (), "strictly increasing"),
        ((), ((5, 9),), "without shell anchors"),
        ((0, 4), ((5,),), "start and end"),
        ((0, 4), ((9, 5),), "strictly increasing"),
        ((0, 4), ((4, 5),), "must not overlap"),
        ((0, -4), (), "non-negative"),
        ((0, True), (), "must be integers"),
    ],
)
def test_napari_polygon_topology_rejects_structurally_invalid_groups(
    shell_anchor_group: tuple[object, ...],
    hole_anchor_groups: tuple[tuple[object, ...], ...],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        NapariPolygonTopology(
            shell_anchor_group=shell_anchor_group,
            hole_anchor_groups=hole_anchor_groups,
        )


def test_sync_napari_polygon_anchor_vertex_synchronizes_exterior_anchor_group() -> None:
    source = Polygon(
        [(0, 0), (8, 0), (8, 8), (0, 8)],
        holes=[[(2, 2), (2, 4), (4, 4), (4, 2)]],
    )
    vertices = shapely_polygon_to_napari_polygon_vertices(source)
    original_vertices = vertices.copy()
    topology = napari_polygon_vertices_to_topology(vertices)
    moved_coordinate = np.asarray([25.0, -10.0])

    synchronized = sync_napari_polygon_anchor_vertex(
        vertices,
        topology,
        moved_vertex_index=4,
        moved_coordinate=moved_coordinate,
    )

    np.testing.assert_allclose(synchronized[[0, 4, 10]], np.asarray([moved_coordinate] * 3))
    np.testing.assert_allclose(synchronized[[1, 2, 3, 5, 6, 7, 8, 9]], original_vertices[[1, 2, 3, 5, 6, 7, 8, 9]])
    np.testing.assert_allclose(vertices, original_vertices)
    assert synchronized is not vertices


def test_sync_napari_polygon_anchor_vertex_synchronizes_hole_anchor_group() -> None:
    source = Polygon(
        [(0, 0), (8, 0), (8, 8), (0, 8)],
        holes=[[(2, 2), (2, 4), (4, 4), (4, 2)]],
    )
    vertices = shapely_polygon_to_napari_polygon_vertices(source)
    original_vertices = vertices.copy()
    topology = napari_polygon_vertices_to_topology(vertices)
    moved_coordinate = np.asarray([12.0, 7.5])

    synchronized = sync_napari_polygon_anchor_vertex(
        vertices,
        topology,
        moved_vertex_index=9,
        moved_coordinate=moved_coordinate,
    )

    np.testing.assert_allclose(synchronized[[5, 9]], np.asarray([moved_coordinate] * 2))
    np.testing.assert_allclose(synchronized[[0, 1, 2, 3, 4, 6, 7, 8, 10]], original_vertices[[0, 1, 2, 3, 4, 6, 7, 8, 10]])
    np.testing.assert_allclose(vertices, original_vertices)


def test_sync_napari_polygon_anchor_vertex_leaves_non_anchor_vertices_unchanged() -> None:
    source = Polygon(
        [(0, 0), (8, 0), (8, 8), (0, 8)],
        holes=[[(2, 2), (2, 4), (4, 4), (4, 2)]],
    )
    vertices = shapely_polygon_to_napari_polygon_vertices(source)
    topology = napari_polygon_vertices_to_topology(vertices)

    synchronized = sync_napari_polygon_anchor_vertex(
        vertices,
        topology,
        moved_vertex_index=7,
        moved_coordinate=np.asarray([50.0, 50.0]),
    )

    np.testing.assert_allclose(synchronized, vertices)
    assert synchronized is not vertices


def test_sync_napari_polygon_anchor_vertex_synchronizes_only_affected_hole_group() -> None:
    source = Polygon(
        [(0, 0), (10, 0), (10, 10), (0, 10)],
        holes=[
            [(2, 2), (2, 4), (4, 4), (4, 2)],
            [(6, 6), (6, 8), (8, 8), (8, 6)],
        ],
    )
    vertices = shapely_polygon_to_napari_polygon_vertices(source)
    original_vertices = vertices.copy()
    topology = napari_polygon_vertices_to_topology(vertices)
    moved_coordinate = np.asarray([-3.0, 14.0])

    synchronized = sync_napari_polygon_anchor_vertex(
        vertices,
        topology,
        moved_vertex_index=11,
        moved_coordinate=moved_coordinate,
    )

    np.testing.assert_allclose(synchronized[[11, 15]], np.asarray([moved_coordinate] * 2))
    np.testing.assert_allclose(synchronized[[0, 4, 10, 16]], original_vertices[[0, 4, 10, 16]])
    np.testing.assert_allclose(synchronized[[5, 9]], original_vertices[[5, 9]])


@pytest.mark.parametrize("moved_vertex_index", [-1, 11])
def test_sync_napari_polygon_anchor_vertex_rejects_out_of_range_moved_index(moved_vertex_index: int) -> None:
    source = Polygon(
        [(0, 0), (8, 0), (8, 8), (0, 8)],
        holes=[[(2, 2), (2, 4), (4, 4), (4, 2)]],
    )
    vertices = shapely_polygon_to_napari_polygon_vertices(source)
    topology = napari_polygon_vertices_to_topology(vertices)

    with pytest.raises(ValueError, match="outside the vertex row"):
        sync_napari_polygon_anchor_vertex(
            vertices,
            topology,
            moved_vertex_index=moved_vertex_index,
            moved_coordinate=np.asarray([1.0, 2.0]),
        )


def test_sync_napari_polygon_anchor_vertex_rejects_invalid_moved_coordinate() -> None:
    vertices = np.asarray(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
    )

    with pytest.raises(ValueError, match="one 2D coordinate"):
        sync_napari_polygon_anchor_vertex(
            vertices,
            NapariPolygonTopology(shell_anchor_group=(), hole_anchor_groups=()),
            moved_vertex_index=0,
            moved_coordinate=np.asarray([[1.0, 2.0]]),
        )


def test_sync_napari_polygon_anchor_vertex_rejects_topology_outside_vertex_row() -> None:
    vertices = np.asarray(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
    )

    with pytest.raises(ValueError, match="outside the row"):
        sync_napari_polygon_anchor_vertex(
            vertices,
            NapariPolygonTopology(shell_anchor_group=(0, 3), hole_anchor_groups=()),
            moved_vertex_index=0,
            moved_coordinate=np.asarray([1.0, 2.0]),
        )


def test_sync_napari_polygon_anchor_vertex_rejects_overlapping_topology_groups() -> None:
    vertices = np.asarray(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
    )

    with pytest.raises(ValueError, match="must not overlap"):
        sync_napari_polygon_anchor_vertex(
            vertices,
            NapariPolygonTopology(shell_anchor_group=(0, 1), hole_anchor_groups=((1, 2),)),
            moved_vertex_index=0,
            moved_coordinate=np.asarray([1.0, 2.0]),
        )


def test_insert_napari_polygon_vertex_inserts_shell_vertex_and_updates_topology() -> None:
    source = Polygon(
        [(0, 0), (8, 0), (8, 8), (0, 8)],
        holes=[[(2, 2), (2, 4), (4, 4), (4, 2)]],
    )
    vertices = shapely_polygon_to_napari_polygon_vertices(source)
    original_vertices = vertices.copy()
    topology = napari_polygon_vertices_to_topology(vertices)
    insert_index = 3
    inserted_coordinate = np.mean(vertices[[insert_index - 1, insert_index]], axis=0)

    inserted_vertices, inserted_topology = insert_napari_polygon_vertex(
        vertices,
        topology,
        insert_index=insert_index,
        inserted_coordinate=inserted_coordinate,
    )

    expected_vertices = np.insert(vertices, insert_index, [inserted_coordinate], axis=0)
    np.testing.assert_allclose(inserted_vertices, expected_vertices)
    assert inserted_topology == NapariPolygonTopology(
        shell_anchor_group=(0, 5, 11),
        hole_anchor_groups=((6, 10),),
    )
    assert inserted_topology == napari_polygon_vertices_to_topology(inserted_vertices)
    np.testing.assert_allclose(vertices, original_vertices)


def test_insert_napari_polygon_vertex_inserts_hole_vertex_and_updates_topology() -> None:
    source = Polygon(
        [(0, 0), (8, 0), (8, 8), (0, 8)],
        holes=[[(2, 2), (2, 4), (4, 4), (4, 2)]],
    )
    vertices = shapely_polygon_to_napari_polygon_vertices(source)
    original_vertices = vertices.copy()
    topology = napari_polygon_vertices_to_topology(vertices)
    insert_index = 8
    inserted_coordinate = np.mean(vertices[[insert_index - 1, insert_index]], axis=0)

    inserted_vertices, inserted_topology = insert_napari_polygon_vertex(
        vertices,
        topology,
        insert_index=insert_index,
        inserted_coordinate=inserted_coordinate,
    )

    expected_vertices = np.insert(vertices, insert_index, [inserted_coordinate], axis=0)
    np.testing.assert_allclose(inserted_vertices, expected_vertices)
    assert inserted_topology == NapariPolygonTopology(
        shell_anchor_group=(0, 4, 11),
        hole_anchor_groups=((5, 10),),
    )
    assert inserted_topology == napari_polygon_vertices_to_topology(inserted_vertices)
    np.testing.assert_allclose(vertices, original_vertices)


def test_insert_napari_polygon_vertex_in_one_hole_keeps_earlier_hole_group_stable() -> None:
    source = Polygon(
        [(0, 0), (10, 0), (10, 10), (0, 10)],
        holes=[
            [(2, 2), (2, 4), (4, 4), (4, 2)],
            [(6, 6), (6, 8), (8, 8), (8, 6)],
        ],
    )
    vertices = shapely_polygon_to_napari_polygon_vertices(source)
    topology = napari_polygon_vertices_to_topology(vertices)
    insert_index = 13
    inserted_coordinate = np.mean(vertices[[insert_index - 1, insert_index]], axis=0)

    inserted_vertices, inserted_topology = insert_napari_polygon_vertex(
        vertices,
        topology,
        insert_index=insert_index,
        inserted_coordinate=inserted_coordinate,
    )

    assert inserted_topology == NapariPolygonTopology(
        shell_anchor_group=(0, 4, 10, 17),
        hole_anchor_groups=((5, 9), (11, 16)),
    )
    assert inserted_topology == napari_polygon_vertices_to_topology(inserted_vertices)


@pytest.mark.parametrize("insert_index", [0, 5, 10])
def test_insert_napari_polygon_vertex_rejects_bridge_or_separator_edges(insert_index: int) -> None:
    source = Polygon(
        [(0, 0), (8, 0), (8, 8), (0, 8)],
        holes=[[(2, 2), (2, 4), (4, 4), (4, 2)]],
    )
    vertices = shapely_polygon_to_napari_polygon_vertices(source)
    topology = napari_polygon_vertices_to_topology(vertices)

    with pytest.raises(ValueError, match="must split a shell or hole ring edge"):
        insert_napari_polygon_vertex(
            vertices,
            topology,
            insert_index=insert_index,
            inserted_coordinate=np.asarray([1.0, 1.0]),
        )


@pytest.mark.parametrize("insert_index", [-1, 11])
def test_insert_napari_polygon_vertex_rejects_out_of_range_insert_index(insert_index: int) -> None:
    source = Polygon(
        [(0, 0), (8, 0), (8, 8), (0, 8)],
        holes=[[(2, 2), (2, 4), (4, 4), (4, 2)]],
    )
    vertices = shapely_polygon_to_napari_polygon_vertices(source)
    topology = napari_polygon_vertices_to_topology(vertices)

    with pytest.raises(ValueError, match="outside the vertex row"):
        insert_napari_polygon_vertex(
            vertices,
            topology,
            insert_index=insert_index,
            inserted_coordinate=np.asarray([1.0, 1.0]),
        )


def test_insert_napari_polygon_vertex_rejects_invalid_inserted_coordinate() -> None:
    source = Polygon(
        [(0, 0), (8, 0), (8, 8), (0, 8)],
        holes=[[(2, 2), (2, 4), (4, 4), (4, 2)]],
    )
    vertices = shapely_polygon_to_napari_polygon_vertices(source)
    topology = napari_polygon_vertices_to_topology(vertices)

    with pytest.raises(ValueError, match="one 2D coordinate"):
        insert_napari_polygon_vertex(
            vertices,
            topology,
            insert_index=3,
            inserted_coordinate=np.asarray([[1.0, 1.0]]),
        )


def test_insert_napari_polygon_vertex_rejects_simple_polygon_topology() -> None:
    source = Polygon([(0, 0), (8, 0), (8, 8), (0, 8)])
    vertices = shapely_polygon_to_napari_polygon_vertices(source)
    topology = napari_polygon_vertices_to_topology(vertices)

    with pytest.raises(ValueError, match="hole-bearing vertex row"):
        insert_napari_polygon_vertex(
            vertices,
            topology,
            insert_index=2,
            inserted_coordinate=np.asarray([8.0, 4.0]),
        )


def test_insert_napari_polygon_vertex_rejects_ambiguous_inserted_coordinate() -> None:
    source = Polygon(
        [(0, 0), (8, 0), (8, 8), (0, 8)],
        holes=[[(2, 2), (2, 4), (4, 4), (4, 2)]],
    )
    vertices = shapely_polygon_to_napari_polygon_vertices(source)
    topology = napari_polygon_vertices_to_topology(vertices)

    with pytest.raises(ValueError):
        insert_napari_polygon_vertex(
            vertices,
            topology,
            insert_index=3,
            inserted_coordinate=vertices[0],
        )


def test_delete_napari_polygon_vertex_deletes_shell_vertex_and_updates_topology() -> None:
    source = Polygon(
        [(0, 0), (8, 0), (8, 8), (4, 8), (0, 8)],
        holes=[[(2, 2), (2, 4), (4, 4), (4, 2)]],
    )
    vertices = shapely_polygon_to_napari_polygon_vertices(source)
    original_vertices = vertices.copy()
    topology = napari_polygon_vertices_to_topology(vertices)

    deleted_vertices, deleted_topology = delete_napari_polygon_vertex(
        vertices,
        topology,
        deleted_vertex_index=3,
    )

    expected_vertices = np.delete(vertices, 3, axis=0)
    np.testing.assert_allclose(deleted_vertices, expected_vertices)
    assert deleted_topology == NapariPolygonTopology(
        shell_anchor_group=(0, 4, 10),
        hole_anchor_groups=((5, 9),),
    )
    assert deleted_topology == napari_polygon_vertices_to_topology(deleted_vertices)
    assert len(napari_polygon_vertices_to_shapely_polygon(deleted_vertices).interiors) == 1
    np.testing.assert_allclose(vertices, original_vertices)


def test_delete_napari_polygon_vertex_deletes_hole_vertex_and_updates_topology() -> None:
    source = Polygon(
        [(0, 0), (8, 0), (8, 8), (0, 8)],
        holes=[[(2, 2), (2, 4), (4, 4), (4, 2)]],
    )
    vertices = shapely_polygon_to_napari_polygon_vertices(source)
    original_vertices = vertices.copy()
    topology = napari_polygon_vertices_to_topology(vertices)

    deleted_vertices, deleted_topology = delete_napari_polygon_vertex(
        vertices,
        topology,
        deleted_vertex_index=7,
    )

    expected_vertices = np.delete(vertices, 7, axis=0)
    np.testing.assert_allclose(deleted_vertices, expected_vertices)
    assert deleted_topology == NapariPolygonTopology(
        shell_anchor_group=(0, 4, 9),
        hole_anchor_groups=((5, 8),),
    )
    assert deleted_topology == napari_polygon_vertices_to_topology(deleted_vertices)
    assert len(napari_polygon_vertices_to_shapely_polygon(deleted_vertices).interiors) == 1
    np.testing.assert_allclose(vertices, original_vertices)


def test_delete_napari_polygon_vertex_in_one_hole_updates_later_topology_groups() -> None:
    source = Polygon(
        [(0, 0), (10, 0), (10, 10), (0, 10)],
        holes=[
            [(2, 2), (2, 4), (4, 4), (4, 2)],
            [(6, 6), (6, 8), (8, 8), (8, 6)],
        ],
    )
    vertices = shapely_polygon_to_napari_polygon_vertices(source)
    topology = napari_polygon_vertices_to_topology(vertices)

    deleted_vertices, deleted_topology = delete_napari_polygon_vertex(
        vertices,
        topology,
        deleted_vertex_index=7,
    )

    assert deleted_topology == NapariPolygonTopology(
        shell_anchor_group=(0, 4, 9, 15),
        hole_anchor_groups=((5, 8), (10, 14)),
    )
    assert deleted_topology == napari_polygon_vertices_to_topology(deleted_vertices)


@pytest.mark.parametrize("deleted_vertex_index", [0, 4, 5, 9, 10])
def test_delete_napari_polygon_vertex_rejects_anchor_or_separator_deletion(deleted_vertex_index: int) -> None:
    source = Polygon(
        [(0, 0), (8, 0), (8, 8), (0, 8)],
        holes=[[(2, 2), (2, 4), (4, 4), (4, 2)]],
    )
    vertices = shapely_polygon_to_napari_polygon_vertices(source)
    topology = napari_polygon_vertices_to_topology(vertices)

    with pytest.raises(ValueError, match="anchor/separator"):
        delete_napari_polygon_vertex(
            vertices,
            topology,
            deleted_vertex_index=deleted_vertex_index,
        )


@pytest.mark.parametrize("deleted_vertex_index", [-1, 11])
def test_delete_napari_polygon_vertex_rejects_out_of_range_deleted_index(deleted_vertex_index: int) -> None:
    source = Polygon(
        [(0, 0), (8, 0), (8, 8), (0, 8)],
        holes=[[(2, 2), (2, 4), (4, 4), (4, 2)]],
    )
    vertices = shapely_polygon_to_napari_polygon_vertices(source)
    topology = napari_polygon_vertices_to_topology(vertices)

    with pytest.raises(ValueError, match="outside the vertex row"):
        delete_napari_polygon_vertex(
            vertices,
            topology,
            deleted_vertex_index=deleted_vertex_index,
        )


def test_delete_napari_polygon_vertex_rejects_simple_polygon_topology() -> None:
    source = Polygon([(0, 0), (8, 0), (8, 8), (0, 8)])
    vertices = shapely_polygon_to_napari_polygon_vertices(source)
    topology = napari_polygon_vertices_to_topology(vertices)

    with pytest.raises(ValueError, match="hole-bearing vertex row"):
        delete_napari_polygon_vertex(
            vertices,
            topology,
            deleted_vertex_index=2,
        )


def test_delete_napari_polygon_vertex_rejects_too_short_shell_ring() -> None:
    source = Polygon(
        [(0, 0), (10, 0), (0, 10)],
        holes=[[(1, 1), (2, 1), (1, 2)]],
    )
    vertices = shapely_polygon_to_napari_polygon_vertices(source)
    topology = napari_polygon_vertices_to_topology(vertices)

    with pytest.raises(ValueError, match="at least four coordinates"):
        delete_napari_polygon_vertex(
            vertices,
            topology,
            deleted_vertex_index=1,
        )


def test_delete_napari_polygon_vertex_rejects_too_short_hole_ring() -> None:
    source = Polygon(
        [(0, 0), (8, 0), (8, 8), (0, 8)],
        holes=[[(2, 2), (4, 2), (2, 4)]],
    )
    vertices = shapely_polygon_to_napari_polygon_vertices(source)
    topology = napari_polygon_vertices_to_topology(vertices)

    with pytest.raises(ValueError, match="at least four coordinates"):
        delete_napari_polygon_vertex(
            vertices,
            topology,
            deleted_vertex_index=6,
        )


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


def test_napari_polygon_vertices_to_topology_rejects_missing_hole_separator() -> None:
    source = Polygon(
        [(0, 0), (8, 0), (8, 8), (0, 8)],
        holes=[[(2, 2), (2, 4), (4, 4), (4, 2)]],
    )
    path = shapely_polygon_to_napari_polygon_vertices(source)[:-1]

    with pytest.raises(ValueError, match="path with holes must end on the exterior anchor"):
        napari_polygon_vertices_to_topology(path)


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


def test_napari_polygon_vertices_to_topology_rejects_open_hole_ring() -> None:
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
        napari_polygon_vertices_to_topology(path_yx)


def test_napari_polygon_vertices_to_topology_rejects_too_short_hole_ring() -> None:
    path_yx = np.asarray(
        [
            [0.0, 0.0],
            [0.0, 8.0],
            [8.0, 8.0],
            [8.0, 0.0],
            [0.0, 0.0],
            [2.0, 2.0],
            [2.0, 2.0],
            [0.0, 0.0],
        ]
    )

    with pytest.raises(ValueError, match="each hole ring must contain at least four coordinates"):
        napari_polygon_vertices_to_topology(path_yx)


def test_napari_polygon_vertices_to_topology_rejects_too_short_shell_ring() -> None:
    path_yx = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [0.0, 0.0],
            [0.5, 0.5],
            [0.5, 0.5],
            [0.0, 0.0],
        ]
    )

    with pytest.raises(ValueError, match="exterior ring must contain at least four coordinates"):
        napari_polygon_vertices_to_topology(path_yx)


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
