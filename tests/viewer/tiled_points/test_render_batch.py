from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from napari_harpy.viewer.tiled_points.contracts import (
    TILED_POINTS_VERTEX_DTYPE,
    TiledPointsRenderBatch,
    TiledPointsRenderTile,
    TileResidencyKey,
)
from napari_harpy.viewer.tiled_points.render_batch import pack_render_tiles

_GENERATION_ID = "12345678-1234-5678-9234-567812345678"
_MAX_PAYLOAD_BYTES = 1_000_000


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


def _pack(
    tiles: tuple[TiledPointsRenderTile, ...],
    *,
    point_count: int | None = None,
    value_count: int = 3,
    max_vertex_payload_bytes: int = _MAX_PAYLOAD_BYTES,
    check_cancelled: Callable[[], None] | None = None,
) -> TiledPointsRenderBatch:
    return pack_render_tiles(
        tiles,
        point_count=sum(tile.point_count for tile in tiles) if point_count is None else point_count,
        value_count=value_count,
        max_vertex_payload_bytes=max_vertex_payload_bytes,
        check_cancelled=check_cancelled,
    )


def test_pack_render_tiles_folds_offsets_into_one_immutable_canonical_array() -> None:
    first = _tile(0, 0, ((1.0, 2.0), (3.0, 4.0)), (0, 1))
    second = _tile(2, 1, ((0.5, 1.5),), (2,))

    batch = _pack((first, second))
    vertices = batch.vertices

    assert batch.point_count == 3
    assert batch.nbytes == 36
    assert vertices.dtype == TILED_POINTS_VERTEX_DTYPE
    assert vertices.flags.c_contiguous
    assert vertices.flags.owndata
    assert not vertices.flags.writeable
    assert np.array_equal(
        vertices["a_position"],
        np.asarray(((1.0, 2.0), (3.0, 4.0), (20.5, 11.5)), dtype=np.float32),
    )
    assert np.array_equal(vertices["a_value_id"], np.asarray((0.0, 1.0, 2.0), dtype=np.float32))


def test_pack_render_tiles_returns_an_owning_immutable_empty_batch() -> None:
    batch = _pack(())

    assert batch.vertices.shape == (0,)
    assert batch.vertices.dtype == TILED_POINTS_VERTEX_DTYPE
    assert batch.vertices.flags.c_contiguous
    assert batch.vertices.flags.owndata
    assert not batch.vertices.flags.writeable


@pytest.mark.parametrize("point_count", [0, 2])
def test_pack_render_tiles_rejects_incorrect_declared_point_count(point_count: int) -> None:
    tile = _tile(0, 0, ((1.0, 2.0),), (0,))

    with pytest.raises(RuntimeError, match="declared render-batch point count"):
        _pack((tile,), point_count=point_count)


def test_pack_render_tiles_preflights_capacity_before_allocation(monkeypatch: pytest.MonkeyPatch) -> None:
    tile = _tile(0, 0, ((1.0, 2.0),), (0,))
    allocation_attempted = False

    def _unexpected_empty(*args, **kwargs):
        nonlocal allocation_attempted
        del args, kwargs
        allocation_attempted = True
        raise AssertionError("allocation should not be attempted")

    monkeypatch.setattr("napari_harpy.viewer.tiled_points.render_batch.np.empty", _unexpected_empty)
    with pytest.raises(ValueError, match="max_vertex_payload_bytes=11"):
        _pack((tile,), max_vertex_payload_bytes=11)

    assert not allocation_attempted


def test_pack_render_tiles_checks_cancellation_before_allocation(monkeypatch: pytest.MonkeyPatch) -> None:
    tile = _tile(0, 0, ((1.0, 2.0),), (0,))
    allocation_attempted = False

    def _unexpected_empty(*args, **kwargs):
        nonlocal allocation_attempted
        del args, kwargs
        allocation_attempted = True
        raise AssertionError("allocation should not be attempted")

    def _cancel() -> None:
        raise RuntimeError("cancelled")

    monkeypatch.setattr("napari_harpy.viewer.tiled_points.render_batch.np.empty", _unexpected_empty)
    with pytest.raises(RuntimeError, match="cancelled"):
        _pack((tile,), check_cancelled=_cancel)

    assert not allocation_attempted


def test_pack_render_tiles_checks_cancellation_between_fragmented_tile_groups() -> None:
    tiles = tuple(_tile(tile_x, 0, ((1.0, 2.0),), (0,)) for tile_x in range(129))
    checks = 0

    def _cancel_during_pack() -> None:
        nonlocal checks
        checks += 1
        if checks == 2:
            raise RuntimeError("cancelled during pack")

    with pytest.raises(RuntimeError, match="cancelled during pack"):
        _pack(tiles, check_cancelled=_cancel_during_pack)

    assert checks == 2


def test_pack_render_tiles_rejects_value_outside_palette() -> None:
    tile = _tile(0, 0, ((1.0, 2.0),), (3,))

    with pytest.raises(ValueError, match="exceeds the complete value palette"):
        _pack((tile,))


def test_pack_render_tiles_rejects_nonfinite_cache_relative_position() -> None:
    tile = _tile(0, 0, ((np.nan, 2.0),), (0,))

    with pytest.raises(ValueError, match="must be finite"):
        _pack((tile,))


def test_pack_render_tiles_preserves_large_finite_cache_relative_position() -> None:
    tile = _tile(1_000_000, 2_000_000, ((1.0, 2.0),), (0,))

    position = _pack((tile,)).vertices["a_position"][0]

    np.testing.assert_array_equal(position, np.asarray((10_000_001.0, 20_000_002.0), dtype=np.float32))
