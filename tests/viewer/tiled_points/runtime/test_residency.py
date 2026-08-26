from __future__ import annotations

import numpy as np

from napari_harpy.viewer.tiled_points.contracts import TiledPointsRenderTile, TileResidencyKey
from napari_harpy.viewer.tiled_points.runtime.residency import _CpuTileResidency

_GENERATION_ID = "12345678-1234-5678-9234-567812345678"


def _tile(tile_x: int, *, point_count: int = 1) -> TiledPointsRenderTile:
    return TiledPointsRenderTile(
        key=TileResidencyKey(_GENERATION_ID, None, 0, tile_x, 0),
        tile_size=10,
        location=np.full((point_count, 2), tile_x, dtype=np.float32),
        value_id=np.zeros(point_count, dtype=np.uint32),
    )


def test_residency_evicts_least_recent_unprotected_tile() -> None:
    residency = _CpuTileResidency(max_resident_bytes=24)
    first, second, third = (_tile(tile_x) for tile_x in range(3))

    assert residency.retain((first, second)) == (first.key, second.key)
    assert residency.get(first.key) is first
    assert residency.retain((third,)) == (third.key,)

    assert residency.keys == (first.key, third.key)
    assert residency.get(second.key) is None
    assert residency.resident_bytes == 24


def test_residency_keeps_active_tiles_and_leaves_new_tile_transient() -> None:
    residency = _CpuTileResidency(max_resident_bytes=24)
    first, second, third = (_tile(tile_x) for tile_x in range(3))
    residency.retain((first, second))

    retained = residency.retain((third,), protected_keys=(first.key, second.key))

    assert retained == ()
    assert residency.keys == (first.key, second.key)
    assert residency.resident_bytes == 24


def test_residency_does_not_retain_oversized_payload() -> None:
    residency = _CpuTileResidency(max_resident_bytes=24)
    oversized = _tile(0, point_count=3)

    assert residency.retain((oversized,)) == ()
    assert residency.tile_count == 0
    assert residency.resident_bytes == 0
