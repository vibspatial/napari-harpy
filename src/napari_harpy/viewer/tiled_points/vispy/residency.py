"""Byte-bounded GPU tile residency for the tiled-points renderer."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable
from typing import Protocol

from napari_harpy.viewer.tiled_points.contracts import TileResidencyKey


class _GpuTileCapacityError(RuntimeError):
    """Report that an atomic active-plus-pending transition cannot fit."""


class _GpuTileResource(Protocol):
    """Describe resources accepted by GPU tile residency.

    This protocol is a structural typing contract, not an instantiated resource
    or required base class. ``_VispyTileResource`` is the production object that
    satisfies it by providing ``key``, ``point_count``, ``resident_bytes``, and
    ``close()``. Residency tests may use lightweight objects implementing the
    same fields without constructing VisPy resources.
    """

    key: TileResidencyKey
    point_count: int
    resident_bytes: int

    def close(self) -> None:
        """Release the resource on the GUI/OpenGL thread."""


class _GpuTileResidency:
    """Own a byte-bounded LRU of renderer tile resources.

    Entries follow least-to-most-recent use order. Capacity preparation pins
    the complete active and pending resource union, evicts only inactive LRU
    entries, and fails before upload when the pinned transition cannot fit.
    """

    def __init__(self, max_resident_bytes: int) -> None:
        if not isinstance(max_resident_bytes, int) or isinstance(max_resident_bytes, bool) or max_resident_bytes <= 0:
            raise ValueError("`max_resident_bytes` must be a positive integer.")
        self._max_resident_bytes = max_resident_bytes
        self._resident_bytes = 0
        self._eviction_count = 0
        self._entries: OrderedDict[TileResidencyKey, _GpuTileResource] = OrderedDict()

    @property
    def max_resident_bytes(self) -> int:
        """Return the configured logical GPU tile-byte limit."""
        return self._max_resident_bytes

    @property
    def resident_bytes(self) -> int:
        """Return currently retained logical tile-buffer bytes."""
        return self._resident_bytes

    @property
    def tile_count(self) -> int:
        """Return the number of retained tile resources."""
        return len(self._entries)

    @property
    def point_count(self) -> int:
        """Return points represented by retained tile resources."""
        return sum(resource.point_count for resource in self._entries.values())

    @property
    def eviction_count(self) -> int:
        """Return the number of resources released for capacity."""
        return self._eviction_count

    @property
    def keys(self) -> tuple[TileResidencyKey, ...]:
        """Return retained keys from least to most recently used."""
        return tuple(self._entries)

    def get(self, key: TileResidencyKey) -> _GpuTileResource | None:
        """Return and mark one retained resource as most recently used."""
        resource = self._entries.get(key)
        if resource is not None:
            self._entries.move_to_end(key)
        return resource

    def prepare_capacity(
        self,
        *,
        required_new_bytes: int,
        protected_keys: Iterable[TileResidencyKey],
    ) -> tuple[TileResidencyKey, ...]:
        """Evict inactive LRU resources so a protected transition can fit."""
        if not isinstance(required_new_bytes, int) or isinstance(required_new_bytes, bool) or required_new_bytes < 0:
            raise ValueError("`required_new_bytes` must be a nonnegative integer.")
        protected = frozenset(protected_keys)
        protected_bytes = sum(resource.resident_bytes for key, resource in self._entries.items() if key in protected)
        if protected_bytes + required_new_bytes > self._max_resident_bytes:
            raise _GpuTileCapacityError(
                "Active and pending tile resources require "
                f"{protected_bytes + required_new_bytes} bytes, exceeding "
                f"max_gpu_tile_bytes={self._max_resident_bytes}."
            )

        evicted: list[TileResidencyKey] = []
        for key in tuple(self._entries):
            if self._resident_bytes + required_new_bytes <= self._max_resident_bytes:
                break
            if key in protected:
                continue
            self._discard(key, count_eviction=True)
            evicted.append(key)
        if self._resident_bytes + required_new_bytes > self._max_resident_bytes:
            raise RuntimeError("GPU tile residency could not reconcile its capacity preflight.")
        return tuple(evicted)

    def retain(self, resource: _GpuTileResource) -> None:
        """Retain one newly created resource after successful preflight."""
        if resource.key in self._entries:
            raise ValueError("A GPU tile resource with this key is already retained.")
        if self._resident_bytes + resource.resident_bytes > self._max_resident_bytes:
            raise RuntimeError("GPU tile resource was retained without sufficient prepared capacity.")
        self._entries[resource.key] = resource
        self._resident_bytes += resource.resident_bytes
        self._require_consistent_bytes()

    def discard(self, key: TileResidencyKey) -> None:
        """Release one retained resource without recording an LRU eviction."""
        self._discard(key, count_eviction=False)
        self._require_consistent_bytes()

    def clear(self) -> None:
        """Release every retained resource and reset byte accounting."""
        for key in tuple(self._entries):
            self._discard(key, count_eviction=False)
        self._require_consistent_bytes()

    def _discard(self, key: TileResidencyKey, *, count_eviction: bool) -> None:
        resource = self._entries.pop(key)
        self._resident_bytes -= resource.resident_bytes
        resource.close()
        if count_eviction:
            self._eviction_count += 1

    def _require_consistent_bytes(self) -> None:
        observed = sum(resource.resident_bytes for resource in self._entries.values())
        if observed != self._resident_bytes or observed > self._max_resident_bytes:
            raise RuntimeError("GPU tile residency byte accounting is inconsistent.")
