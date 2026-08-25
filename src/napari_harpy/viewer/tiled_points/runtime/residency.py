"""Retain decoded tiled-points payloads under an explicit byte budget."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable

from napari_harpy.viewer.tiled_points.contracts import TiledPointsRenderTile, TileResidencyKey


class _CpuTileResidency:
    """Own one worker-thread LRU of immutable decoded point tiles.

    The byte budget covers only entries retained by this LRU. A caller may
    still hold transient immutable references returned for snapshot assembly.
    Payloads larger than the complete budget are therefore usable for one
    result but are never reported as resident.
    """

    def __init__(self, max_resident_bytes: int) -> None:
        if not isinstance(max_resident_bytes, int) or isinstance(max_resident_bytes, bool) or max_resident_bytes <= 0:
            raise ValueError("`max_resident_bytes` must be a positive integer.")
        self._max_resident_bytes = max_resident_bytes
        self._resident_bytes = 0
        self._entries: OrderedDict[TileResidencyKey, TiledPointsRenderTile] = OrderedDict()

    @property
    def max_resident_bytes(self) -> int:
        """Return the configured retained-payload byte limit."""
        return self._max_resident_bytes

    @property
    def resident_bytes(self) -> int:
        """Return bytes currently accounted to retained tile arrays."""
        return self._resident_bytes

    @property
    def tile_count(self) -> int:
        """Return the number of currently retained logical tiles."""
        return len(self._entries)

    @property
    def keys(self) -> tuple[TileResidencyKey, ...]:
        """Return residency keys from least to most recently used."""
        return tuple(self._entries)

    def get(self, key: TileResidencyKey) -> TiledPointsRenderTile | None:
        """Return and mark one resident tile as most recently used."""
        if not isinstance(key, TileResidencyKey):
            raise ValueError("`key` must be TileResidencyKey.")
        tile = self._entries.get(key)
        if tile is not None:
            self._entries.move_to_end(key)
        return tile

    def retain(
        self,
        tiles: tuple[TiledPointsRenderTile, ...],
        *,
        protected_keys: Iterable[TileResidencyKey] = (),
    ) -> tuple[TileResidencyKey, ...]:
        """Retain eligible tiles without evicting protected active entries.

        Tiles are considered in caller order. Before each insertion, the least
        recently used unprotected entries are evicted until the tile fits. If
        protected entries occupy too much of the budget, that tile remains a
        transient caller-owned payload rather than making accounting exceed the
        configured limit.
        """
        if not isinstance(tiles, tuple) or not all(isinstance(tile, TiledPointsRenderTile) for tile in tiles):
            raise ValueError("`tiles` must be a tuple of TiledPointsRenderTile values.")
        if len({tile.key for tile in tiles}) != len(tiles):
            raise ValueError("`tiles` must not contain duplicate residency keys.")
        protected = frozenset(protected_keys)
        if not all(isinstance(key, TileResidencyKey) for key in protected):
            raise ValueError("`protected_keys` must contain TileResidencyKey values.")

        retained: list[TileResidencyKey] = []
        for tile in tiles:
            if tile.resident_bytes > self._max_resident_bytes:
                continue
            existing = self._entries.pop(tile.key, None)
            if existing is not None:
                self._resident_bytes -= existing.resident_bytes
            self._evict_until_fits(tile.resident_bytes, protected)
            if self._resident_bytes + tile.resident_bytes > self._max_resident_bytes:
                if existing is not None:
                    self._entries[tile.key] = existing
                    self._resident_bytes += existing.resident_bytes
                continue
            self._entries[tile.key] = tile
            self._resident_bytes += tile.resident_bytes
            retained.append(tile.key)
        self._require_consistent_bytes()
        return tuple(retained)

    def clear(self) -> None:
        """Drop all retained tile references and reset byte accounting."""
        self._entries.clear()
        self._resident_bytes = 0

    def _evict_until_fits(self, required_bytes: int, protected: frozenset[TileResidencyKey]) -> None:
        for key in tuple(self._entries):
            if self._resident_bytes + required_bytes <= self._max_resident_bytes:
                return
            if key in protected:
                continue
            tile = self._entries.pop(key)
            self._resident_bytes -= tile.resident_bytes

    def _require_consistent_bytes(self) -> None:
        observed = sum(tile.resident_bytes for tile in self._entries.values())
        if observed != self._resident_bytes or observed > self._max_resident_bytes:
            raise RuntimeError("CPU tile residency byte accounting is inconsistent.")
