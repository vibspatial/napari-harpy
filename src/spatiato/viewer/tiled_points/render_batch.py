"""Pack ordered logical tiles into one renderer vertex payload."""

from __future__ import annotations

from collections.abc import Callable
from numbers import Integral
from typing import Final

import numpy as np

from napari_harpy.viewer.tiled_points.contracts import (
    TILED_POINTS_VERTEX_DTYPE,
    TiledPointsRenderBatch,
    TiledPointsRenderTile,
)

MAX_EXACT_FLOAT32_INTEGER: Final = 2**24
_FLOAT32_MAX: Final = float(np.finfo(np.float32).max)
_CANCELLATION_CHECK_TILE_INTERVAL: Final = 64


def pack_render_tiles(
    tiles: tuple[TiledPointsRenderTile, ...],
    *,
    point_count: int,
    value_count: int,
    max_vertex_payload_bytes: int,
    raise_if_cancelled: Callable[[], None] | None = None,
) -> TiledPointsRenderBatch:
    """Pack one complete ordered tile tuple into an immutable render batch.

    Parameters
    ----------
    tiles
        Complete logical tile payload in final spatial plan order. It may
        contain only a selected, spatially restricted, or sampled subset of the
        complete cache vocabulary and therefore does not define the complete
        valid value-ID range.
    point_count
        Catalog-declared row count for the complete tile tuple. The packer uses
        it for byte preflight and allocation, then reconciles the final cursor.
    value_count
        Number of canonical values in the complete cache vocabulary, not the
        number selected or present in ``tiles``. Valid value IDs lie in
        ``[0, value_count)``.

        This stable dataset-level value is passed separately rather than stored
        on every tile. It cannot be inferred from a selected, spatially
        restricted, sampled, or empty tile tuple. The packer uses it to reject
        value IDs that cannot address the layer's complete palette.
    max_vertex_payload_bytes
        Maximum logical byte size permitted for the one packed allocation.
        Capacity is checked before allocation.
    raise_if_cancelled
        Optional cooperative cancellation checkpoint called before allocation
        and between bounded groups of logical tiles. It returns normally to
        continue or raises to abort packing; this function does not catch the
        exception.

    Returns
    -------
    TiledPointsRenderBatch
        One owning, read-only, C-contiguous canonical vertex allocation in the
        supplied tile order.

    Notes
    -----
    Each input ``tile.location`` contains tile-local ``(x, y)`` coordinates:
    the cache writer has already removed both the shared cache origin and the
    logical tile-grid offset. This function folds only the tile-grid offset
    back into each row::

        packed_x = location_x + tile_x * tile_size
        packed_y = location_y + tile_y * tile_size

    The resulting positions therefore equal the intrinsic coordinates minus
    the shared cache origin. Do not add ``x_origin`` or ``y_origin`` here.
    ``VispyTiledPointsLayer._on_matrix_change()`` restores that origin through
    the float64 root transform before the normal napari affine is applied.
    Keeping the VBO coordinates cache-relative limits float32 precision loss
    while allowing the complete snapshot to use one visual and vertex buffer.
    """
    if not isinstance(tiles, tuple) or not all(isinstance(tile, TiledPointsRenderTile) for tile in tiles):
        raise ValueError("`tiles` must be a tuple of TiledPointsRenderTile values.")
    if not isinstance(point_count, Integral) or isinstance(point_count, bool) or point_count < 0:
        raise ValueError("`point_count` must be a nonnegative integer.")
    if not isinstance(value_count, Integral) or isinstance(value_count, bool) or value_count <= 0:
        raise ValueError("`value_count` must be a positive integer.")
    if (
        not isinstance(max_vertex_payload_bytes, Integral)
        or isinstance(max_vertex_payload_bytes, bool)
        or max_vertex_payload_bytes <= 0
    ):
        raise ValueError("`max_vertex_payload_bytes` must be a positive integer.")
    if raise_if_cancelled is not None and not callable(raise_if_cancelled):
        raise ValueError("`raise_if_cancelled` must be callable or None.")
    point_count = int(point_count)
    value_count = int(value_count)
    max_vertex_payload_bytes = int(max_vertex_payload_bytes)
    required_bytes = point_count * TILED_POINTS_VERTEX_DTYPE.itemsize
    if required_bytes > max_vertex_payload_bytes:
        raise ValueError(
            f"Render batch requires {required_bytes} bytes, exceeding "
            f"max_vertex_payload_bytes={max_vertex_payload_bytes}."
        )
    if raise_if_cancelled is not None:
        raise_if_cancelled()

    vertices = np.empty(point_count, dtype=TILED_POINTS_VERTEX_DTYPE)
    cursor = 0
    for tile_index, tile in enumerate(tiles):
        if (
            raise_if_cancelled is not None
            and tile_index > 0
            and tile_index % _CANCELLATION_CHECK_TILE_INTERVAL == 0
        ):
            raise_if_cancelled()
        stop = cursor + tile.point_count
        if stop > point_count:
            raise RuntimeError("Packed tile rows exceed the declared render-batch point count.")
        maximum_value_id = int(tile.value_id.max())
        if maximum_value_id >= value_count:
            raise ValueError("A snapshot value ID exceeds the complete value palette.")
        if maximum_value_id > MAX_EXACT_FLOAT32_INTEGER:
            raise ValueError("Snapshot value IDs exceed exact float32 integer representation.")

        # Fold in only the tile-grid offset. The shared cache origin remains
        # excluded and is restored by the renderer's float64 root transform.
        x_offset = tile.key.tile_x * tile.tile_size
        y_offset = tile.key.tile_y * tile.tile_size
        if x_offset > _FLOAT32_MAX or y_offset > _FLOAT32_MAX:
            raise ValueError("A logical tile offset exceeds finite float32 representation.")

        positions = vertices["a_position"][cursor:stop]
        positions[...] = tile.location
        positions[:, 0] += np.float32(x_offset)
        positions[:, 1] += np.float32(y_offset)
        if not bool(np.isfinite(positions).all()):
            raise ValueError("Packed cache-relative positions must be finite float32 values.")
        vertices["a_value_id"][cursor:stop] = tile.value_id
        cursor = stop

    if cursor != len(vertices):
        raise RuntimeError("Packed tile rows do not match the declared render-batch point count.")
    vertices.flags.writeable = False
    return TiledPointsRenderBatch(vertices)
