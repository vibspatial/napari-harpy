from __future__ import annotations

from collections import OrderedDict

import numpy as np
from loguru import logger

from napari_harpy.viewer.tiled_points.contracts import TiledPointsRenderTile, TileResidencyKey
from napari_harpy.viewer.tiled_points.runtime.residency import _CpuTileResidency

_GENERATION_ID = "12345678-1234-5678-9234-567812345678"


class _TraversalCountingOrderedDict(OrderedDict[TileResidencyKey, TiledPointsRenderTile]):
    """Count complete entry traversal without counting direct lookups."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.key_yields = 0
        self.item_yields = 0
        self.value_yields = 0

    def __iter__(self):
        for key in super().__iter__():
            self.key_yields += 1
            yield key

    def items(self):
        for item in super().items():
            self.item_yields += 1
            yield item

    def values(self):
        for value in super().values():
            self.value_yields += 1
            yield value

    @property
    def traversal_count(self) -> int:
        return self.key_yields + self.item_yields + self.value_yields

    def reset_counts(self) -> None:
        self.key_yields = 0
        self.item_yields = 0
        self.value_yields = 0


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


def test_residency_warns_once_and_does_not_retain_oversized_payload() -> None:
    residency = _CpuTileResidency(max_resident_bytes=24)
    oversized = _tile(0, point_count=3)
    messages: list[str] = []
    sink_id = logger.add(lambda message: messages.append(str(message)), format="{message}", level="WARNING")

    try:
        assert residency.retain((oversized,)) == ()
        residency.clear()
        assert residency.retain((_tile(1, point_count=3),)) == ()
    finally:
        logger.remove(sink_id)

    assert residency.tile_count == 0
    assert residency.resident_bytes == 0
    oversized_warnings = [message.strip() for message in messages if "Decoded tiled-points tile" in message]
    assert oversized_warnings == [
        "Decoded tiled-points tile (0, 0, 0) requires 36 bytes, exceeding "
        "max_cpu_tile_bytes=24; it will remain transient and may be read again "
        "for later viewport requests."
    ]


def test_residency_fitting_insertion_does_not_plan_evictions() -> None:
    residency = _CpuTileResidency(max_resident_bytes=36)
    first, second, third = (_tile(tile_x) for tile_x in range(3))
    residency.retain((first, second))
    entries = _TraversalCountingOrderedDict(residency._entries)
    residency._entries = entries
    entries.reset_counts()

    assert residency.retain((third,)) == (third.key,)

    assert entries.key_yields == 0
    assert entries.item_yields == 0
    # The one final byte-accounting reconciliation remains intentionally O(N).
    assert entries.value_yields == 3
    assert residency.keys == (first.key, second.key, third.key)
    assert residency.resident_bytes == 36


def test_residency_fitting_bulk_retention_scales_linearly() -> None:
    def traversal_count(tile_count: int) -> int:
        residency = _CpuTileResidency(max_resident_bytes=12 * tile_count)
        entries = _TraversalCountingOrderedDict()
        residency._entries = entries
        tiles = tuple(_tile(tile_x) for tile_x in range(tile_count))

        assert residency.retain(tiles) == tuple(tile.key for tile in tiles)
        assert residency.tile_count == tile_count
        assert residency.resident_bytes == 12 * tile_count
        return entries.traversal_count

    small_count = traversal_count(128)
    large_count = traversal_count(256)

    assert small_count == 128
    assert large_count == 2 * small_count


def test_residency_evicts_oldest_unprotected_entry_after_protected_prefix() -> None:
    residency = _CpuTileResidency(max_resident_bytes=36)
    first, second, third, fourth = (_tile(tile_x) for tile_x in range(4))
    residency.retain((first, second, third))

    assert residency.retain((fourth,), protected_keys=(first.key,)) == (fourth.key,)

    assert residency.keys == (first.key, third.key, fourth.key)
    assert residency.get(second.key) is None
    assert residency.resident_bytes == 36


def test_residency_evicts_available_entry_when_protected_capacity_is_insufficient() -> None:
    residency = _CpuTileResidency(max_resident_bytes=24)
    first, second = (_tile(tile_x) for tile_x in range(2))
    transient = _tile(2, point_count=2)
    residency.retain((first, second))

    assert residency.retain((transient,), protected_keys=(first.key,)) == ()

    assert residency.keys == (first.key,)
    assert residency.get(second.key) is None
    assert residency.resident_bytes == 12


def test_residency_successful_replacement_updates_bytes_and_mru_order() -> None:
    residency = _CpuTileResidency(max_resident_bytes=36)
    first, second = (_tile(tile_x) for tile_x in range(2))
    replacement = _tile(0, point_count=2)
    residency.retain((first, second))

    assert residency.retain((replacement,)) == (replacement.key,)

    assert residency.keys == (second.key, replacement.key)
    assert residency.get(replacement.key) is replacement
    assert residency.resident_bytes == 36


def test_residency_failed_replacement_restores_previous_entry() -> None:
    residency = _CpuTileResidency(max_resident_bytes=24)
    first, second = (_tile(tile_x) for tile_x in range(2))
    replacement = _tile(0, point_count=2)
    residency.retain((first, second))

    assert residency.retain((replacement,), protected_keys=(second.key,)) == ()

    assert residency.keys == (second.key, first.key)
    assert residency.get(first.key) is first
    assert residency.resident_bytes == 24
