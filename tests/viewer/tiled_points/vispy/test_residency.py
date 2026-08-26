from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

import pytest

from napari_harpy.viewer.tiled_points import TileResidencyKey
from napari_harpy.viewer.tiled_points.vispy.residency import (
    _GpuTileCapacityError,
    _GpuTileResidency,
)

_GENERATION = str(uuid4())


@dataclass
class _TrackingGpuTileResource:
    key: TileResidencyKey
    point_count: int = 1
    resident_bytes: int = 12
    closed: bool = False

    def close(self) -> None:
        self.closed = True


def _resource(tile_x: int, *, resident_bytes: int = 12) -> _TrackingGpuTileResource:
    return _TrackingGpuTileResource(
        TileResidencyKey(
            cache_generation_id=_GENERATION,
            requested_value_ids=(1,),
            level=0,
            tile_x=tile_x,
            tile_y=0,
        ),
        resident_bytes=resident_bytes,
    )


def test_gpu_residency_evicts_oldest_inactive_resource_for_pending_capacity() -> None:
    residency = _GpuTileResidency(max_resident_bytes=36)
    first, second, third = (_resource(tile_x) for tile_x in range(3))
    residency.retain(first)
    residency.retain(second)
    residency.retain(third)
    assert residency.get(second.key) is second

    evicted = residency.prepare_capacity(required_new_bytes=12, protected_keys=(second.key, third.key))

    assert evicted == (first.key,)
    assert first.closed
    assert residency.keys == (third.key, second.key)
    assert residency.eviction_count == 1
    assert residency.resident_bytes == 24


def test_gpu_residency_rejects_active_pending_union_before_eviction() -> None:
    residency = _GpuTileResidency(max_resident_bytes=24)
    active, inactive = _resource(0), _resource(1)
    residency.retain(active)
    residency.retain(inactive)

    with pytest.raises(_GpuTileCapacityError, match="Active and pending"):
        residency.prepare_capacity(required_new_bytes=24, protected_keys=(active.key,))

    assert residency.keys == (active.key, inactive.key)
    assert not active.closed
    assert not inactive.closed


def test_gpu_residency_clear_releases_every_resource_idempotently() -> None:
    residency = _GpuTileResidency(max_resident_bytes=24)
    first, second = _resource(0), _resource(1)
    residency.retain(first)
    residency.retain(second)

    residency.clear()
    residency.clear()

    assert first.closed and second.closed
    assert residency.tile_count == 0
    assert residency.point_count == 0
    assert residency.resident_bytes == 0
