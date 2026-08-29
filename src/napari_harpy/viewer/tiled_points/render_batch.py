"""Pack logical tiled-points snapshots into one renderer vertex payload."""

from __future__ import annotations

from numbers import Integral
from typing import Final

import numpy as np
import numpy.typing as npt

from napari_harpy.viewer.tiled_points.contracts import TiledPointsRenderSnapshot

MAX_EXACT_FLOAT32_INTEGER: Final = 2**24
TILED_POINTS_VERTEX_DTYPE: Final = np.dtype([("a_position", np.float32, (2,)), ("a_value_id", np.float32)])
_FLOAT32_MAX: Final = float(np.finfo(np.float32).max)


def pack_snapshot_vertices(
    snapshot: TiledPointsRenderSnapshot,
    *,
    value_count: int,
) -> npt.NDArray[np.void]:
    """Pack one tiled render snapshot into an owning vertex array.

    Parameters
    ----------
    snapshot
        Complete ordered logical tile payload for one viewport render
        candidate. A snapshot may contain only a selected, spatially restricted,
        or sampled subset of the complete cache vocabulary and therefore does
        not define the complete valid value-ID range.
    value_count
        Number of canonical values in the complete cache vocabulary, not the
        number selected or present in ``snapshot``. Valid value IDs lie in
        ``[0, value_count)``.

        This stable dataset-level value is passed separately rather than stored
        on every viewport snapshot. It cannot be inferred from a selected,
        spatially restricted, sampled, or empty snapshot. The packer uses it to
        reject value IDs that cannot address the layer's complete palette.

    Returns
    -------
    numpy.ndarray
        One owning, C-contiguous ``TILED_POINTS_VERTEX_DTYPE`` array in snapshot
        tile order.

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
    if not isinstance(snapshot, TiledPointsRenderSnapshot):
        raise ValueError("`snapshot` must be TiledPointsRenderSnapshot.")
    if not isinstance(value_count, Integral) or isinstance(value_count, bool) or value_count <= 0:
        raise ValueError("`value_count` must be a positive integer.")
    value_count = int(value_count)

    vertices = np.empty(snapshot.rendered_point_count, dtype=TILED_POINTS_VERTEX_DTYPE)
    cursor = 0
    for tile in snapshot.tiles:
        stop = cursor + tile.point_count
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
        raise RuntimeError("Packed snapshot point count does not match its logical tiles.")
    return vertices
