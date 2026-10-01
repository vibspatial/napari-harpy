"""Retain decoded tiled-points payloads under an explicit byte budget."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable

from loguru import logger

from napari_harpy.viewer.tiled_points.contracts import TiledPointsRenderTile, TileResidencyKey


class _CpuTileResidency:
    """Own one worker-thread LRU of immutable decoded point tiles.

    The retention policy is::

        byte-bounded LRU
                |
                v
        evict oldest unprotected tiles first
                |
                v
        protect resident tiles used by the active snapshot
                |
                v
        if no space remains, return the new payload transiently
        without caching it

    Entries are ordered from least to most recently used. When one decoded
    batch exceeds the available budget, later tiles may evict earlier new tiles
    from the same batch; the viewport assembly retains all tile references until
    their renderer batch has been packed.

    Newly decoded entries have independently owned point-array allocations at
    the viewer-residency boundary. Consequently, each tile's
    ``resident_bytes`` is the allocation released when that tile is evicted;
    it does not merely describe a view into a larger retained reader batch.

    The byte budget covers only entries retained by this LRU. Viewport assembly
    may temporarily hold immutable decoded payloads outside that budget while
    packing them. Nonresident payloads are released before the completed snapshot
    crosses to the GUI thread.
    """

    def __init__(self, max_resident_bytes: int) -> None:
        if not isinstance(max_resident_bytes, int) or isinstance(max_resident_bytes, bool) or max_resident_bytes <= 0:
            raise ValueError("`max_resident_bytes` must be a positive integer.")
        self._max_resident_bytes = max_resident_bytes
        self._resident_bytes = 0
        self._entries: OrderedDict[TileResidencyKey, TiledPointsRenderTile] = OrderedDict()
        self._oversized_tile_warning_emitted = False

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
            # A tile larger than the complete residency budget can never be
            # admitted, even after every other entry is evicted. Keep it
            # caller-owned and transient so it can still be rendered now.
            if tile.resident_bytes > self._max_resident_bytes:
                if not self._oversized_tile_warning_emitted:
                    logger.warning(
                        "Decoded tiled-points tile {} requires {:,} bytes, exceeding "
                        "max_cpu_tile_bytes={:,}; it will remain transient and may be "
                        "read again for later viewport requests.",
                        tile.key.logical_tile_key,
                        tile.resident_bytes,
                        self._max_resident_bytes,
                    )
                    self._oversized_tile_warning_emitted = True
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
        # Clear payload state only. The one-shot warning is session-lifetime
        # diagnostic state, so selection changes must not make it noisy.
        self._entries.clear()
        self._resident_bytes = 0

    def _evict_until_fits(self, required_bytes: int, protected: frozenset[TileResidencyKey]) -> None:
        """Evict least-recently-used unprotected tiles until a payload fits."""
        if self._resident_bytes + required_bytes <= self._max_resident_bytes:
            return

        # The candidate is not resident yet. This is the number of currently
        # resident bytes that must be evicted before it can be admitted.
        bytes_to_reclaim = self._resident_bytes + required_bytes - self._max_resident_bytes
        victims: list[TileResidencyKey] = []
        selected_victim_bytes = 0
        for key, tile in self._entries.items():
            if key in protected:
                continue
            victims.append(key)
            selected_victim_bytes += tile.resident_bytes
            if selected_victim_bytes >= bytes_to_reclaim:
                break

        # OrderedDict cannot be mutated during direct iteration. Removing the
        # collected candidates afterwards also avoids materializing every
        # resident key when only a small number of LRU victims is needed.
        for key in victims:
            tile = self._entries.pop(key)
            self._resident_bytes -= tile.resident_bytes

    def _require_consistent_bytes(self) -> None:
        observed = sum(tile.resident_bytes for tile in self._entries.values())
        if observed != self._resident_bytes or observed > self._max_resident_bytes:
            raise RuntimeError("CPU tile residency byte accounting is inconsistent.")
