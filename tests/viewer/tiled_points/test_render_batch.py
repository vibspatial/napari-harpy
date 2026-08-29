from __future__ import annotations

import numpy as np
import pytest

from napari_harpy.viewer.tiled_points.contracts import (
    TiledPointsRenderSnapshot,
    TiledPointsRenderTile,
    TileResidencyKey,
)
from napari_harpy.viewer.tiled_points.render_batch import (
    TILED_POINTS_VERTEX_DTYPE,
    pack_snapshot_vertices,
)

_GENERATION_ID = "12345678-1234-5678-9234-567812345678"


def _tile(
    tile_x: int,
    tile_y: int,
    location: tuple[tuple[float, float], ...],
    value_id: tuple[int, ...],
) -> TiledPointsRenderTile:
    return TiledPointsRenderTile(
        key=TileResidencyKey(
            cache_generation_id=_GENERATION_ID,
            requested_value_ids=None,
            level=0,
            tile_x=tile_x,
            tile_y=tile_y,
        ),
        tile_size=10,
        location=np.asarray(location, dtype=np.float32),
        value_id=np.asarray(value_id, dtype=np.uint32),
    )


def _snapshot(tiles: tuple[TiledPointsRenderTile, ...]) -> TiledPointsRenderSnapshot:
    return TiledPointsRenderSnapshot(
        cache_generation_id=_GENERATION_ID,
        request_generation=1,
        selection_generation=0,
        requested_value_ids=None,
        level=0,
        level_kind="exact",
        within_budget=True,
        estimated_point_count=sum(tile.point_count for tile in tiles),
        omitted_value_ids=(),
        tiles=tiles,
    )


def test_pack_snapshot_vertices_folds_offsets_into_one_canonical_array() -> None:
    first = _tile(0, 0, ((1.0, 2.0), (3.0, 4.0)), (0, 1))
    second = _tile(2, 1, ((0.5, 1.5),), (2,))

    vertices = pack_snapshot_vertices(_snapshot((first, second)), value_count=3)

    assert vertices.dtype == TILED_POINTS_VERTEX_DTYPE
    assert vertices.flags.c_contiguous
    assert vertices.flags.owndata
    assert np.array_equal(
        vertices["a_position"],
        np.asarray(((1.0, 2.0), (3.0, 4.0), (20.5, 11.5)), dtype=np.float32),
    )
    assert np.array_equal(vertices["a_value_id"], np.asarray((0.0, 1.0, 2.0), dtype=np.float32))


def test_pack_snapshot_vertices_returns_an_owning_empty_array() -> None:
    vertices = pack_snapshot_vertices(_snapshot(()), value_count=3)

    assert vertices.shape == (0,)
    assert vertices.dtype == TILED_POINTS_VERTEX_DTYPE
    assert vertices.flags.c_contiguous
    assert vertices.flags.owndata


def test_pack_snapshot_vertices_rejects_value_outside_palette() -> None:
    tile = _tile(0, 0, ((1.0, 2.0),), (3,))

    with pytest.raises(ValueError, match="exceeds the complete value palette"):
        pack_snapshot_vertices(_snapshot((tile,)), value_count=3)


def test_pack_snapshot_vertices_rejects_nonfinite_cache_relative_position() -> None:
    tile = _tile(0, 0, ((np.nan, 2.0),), (0,))

    with pytest.raises(ValueError, match="must be finite"):
        pack_snapshot_vertices(_snapshot((tile,)), value_count=3)
