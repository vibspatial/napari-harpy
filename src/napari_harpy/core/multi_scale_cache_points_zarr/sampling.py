from __future__ import annotations

import numpy as np
import numpy.typing as npt

from napari_harpy.core.multi_scale_cache_points_zarr.hashing import _splitmix64
from napari_harpy.core.multi_scale_cache_points_zarr.models import (
    _INT16_MAX,
    _INT64_MAX,
    _UINT32_MAX,
    _require_integer_in_range,
)

SAMPLING_METHOD = "harpy-value-neutral-stratified-splitmix64-v1"
SAMPLING_SEED = 0
SAMPLED_TILE_MICROGRID_EDGE = 16

_POINT_PRIORITY_DOMAIN = np.uint64(0x48504F494E543031)  # "HPOINT01"
_CELL_PRIORITY_DOMAIN = np.uint64(0x4843454C4C303031)  # "HCELL001"
_UINT64_32 = np.uint64(32)
_MICROGRID_CELL_COUNT = SAMPLED_TILE_MICROGRID_EDGE**2


def _select_sampled_tile_indices(
    x_rel: npt.NDArray[np.float32],
    y_rel: npt.NDArray[np.float32],
    point_id: npt.NDArray[np.uint64],
    *,
    level: int,
    tile_x: int,
    tile_y: int,
    tile_size: int,
    target: int,
) -> npt.NDArray[np.int64]:
    """Select deterministic value-neutral representatives from one tile.

    Parameters
    ----------
    x_rel, y_rel
        Aligned one-dimensional C-contiguous ``float32`` coordinates relative
        to the current logical tile. Values must be finite and lie in the
        closed interval ``[0, tile_size]``. An exactly represented upper edge
        is assigned to the last microgrid cell.
    point_id
        Aligned one-dimensional C-contiguous ``uint64`` internal point
        identities. Uniqueness is an accepted immediate-finer-level invariant
        and is not rediscovered by sorting every candidate tile here.
    level
        Non-negative serialized output level in the supported int16 range.
    tile_x, tile_y
        Non-negative uint32 coordinates of the logical output tile.
    tile_size
        Positive edge length of the logical tile in intrinsic source units.
    target
        Positive maximum number of representatives retained from this tile.

    Returns
    -------
    numpy.ndarray
        Exactly ``min(N, target)`` unique original candidate-row positions as a
        C-contiguous ``int64`` array, ordered by ascending retained
        ``point_id``.

    Notes
    -----
    Cache tiles are persistent storage, indexing, and loading units. The
    16-by-16 microgrid used here is transient sampling state inside one current
    tile; its cells are not stored as cache tiles::

        Level    tile edge    microgrid    cell edge
        Bridge         512      16 x 16           32
        L1           1,024      16 x 16           64
        L2           2,048      16 x 16          128

    At coarser levels, four rebased immediate-finer tiles occupy four 8-by-8
    quadrants of the current 16-by-16 sampling microgrid::

        coarser 16 x 16 microgrid
        +---------+---------+
        | finer   | finer   |
        | 8 x 8   | 8 x 8   |
        +---------+---------+
        | finer   | finer   |
        | 8 x 8   | 8 x 8   |
        +---------+---------+

    Sampling first assigns candidates to microgrid cells, then allocates the
    target proportionally with integer largest remainders. Versioned SplitMix64
    cell priorities and numeric cell IDs resolve allocation ties. Within each
    cell, versioned point priorities followed by ``point_id`` rank candidates.
    The retained original positions are finally ordered by ``point_id``.

    ``value_id`` is absent from the API by design: candidate values, their
    frequencies, and the input's value-major storage order cannot influence
    membership.
    """
    _require_array(x_rel, "x_rel", dtype=np.dtype(np.float32))
    _require_array(y_rel, "y_rel", dtype=np.dtype(np.float32))
    _require_array(point_id, "point_id", dtype=np.dtype(np.uint64))
    if not (len(x_rel) == len(y_rel) == len(point_id)):
        raise ValueError("`x_rel`, `y_rel`, and `point_id` must have matching lengths.")

    _require_integer_in_range(level, "level", maximum=_INT16_MAX)
    _require_integer_in_range(tile_x, "tile_x", maximum=_UINT32_MAX)
    _require_integer_in_range(tile_y, "tile_y", maximum=_UINT32_MAX)
    _require_integer_in_range(tile_size, "tile_size", minimum=1, maximum=_INT64_MAX)
    _require_integer_in_range(target, "target", minimum=1, maximum=_INT64_MAX)
    if not bool(np.isfinite(x_rel).all()) or not bool(np.isfinite(y_rel).all()):
        raise ValueError("Relative coordinates must be finite.")
    if (
        bool((x_rel < 0).any())
        or bool((x_rel > tile_size).any())
        or bool((y_rel < 0).any())
        or bool((y_rel > tile_size).any())
    ):
        raise ValueError("Relative coordinates must lie in the closed interval [0, tile_size].")

    candidate_count = len(point_id)
    if candidate_count == 0:
        return np.empty(0, dtype=np.int64)
    # `target` is a maximum capacity, not a requested sample size. Sparse tiles
    # retain every candidate and only restore deterministic point-ID order.
    if candidate_count <= target:
        return np.ascontiguousarray(np.argsort(point_id, kind="stable"), dtype=np.int64)

    # Map every candidate to one cell, then turn those point-level IDs into the
    # fixed 256-entry occupancy histogram used to allocate the tile budget.
    candidate_cell_id = _microgrid_cell_ids(x_rel, y_rel, tile_size=tile_size)
    cell_counts = np.bincount(candidate_cell_id, minlength=_MICROGRID_CELL_COUNT)
    # `cell_targets[cell_id]` is the representative quota for that microgrid cell.
    cell_targets = _allocate_cell_targets(
        cell_counts,
        target=target,
        level=level,
        tile_x=tile_x,
        tile_y=tile_y,
    )
    # Point IDs and cell IDs are parallel candidate arrays. Their fixed domain,
    # seed, level, tile key, cell ID, and point ID produce one priority per row.
    candidate_priority = _point_priorities(
        point_id,
        candidate_cell_id,
        level=level,
        tile_x=tile_x,
        tile_y=tile_y,
    )

    # `np.lexsort` treats its final key as primary: group by cell, rank by
    # priority within that cell, and use point ID for an exact priority tie.
    ordered = np.lexsort((point_id, candidate_priority, candidate_cell_id))
    # Sorting primarily by cell makes each cell contiguous, for example
    # [0, 0, 2, 2, 2]. Adjacent comparisons then produce group starts [0, 2].
    ordered_cell_id = candidate_cell_id[ordered]
    group_starts = np.flatnonzero(np.concatenate((np.array([True]), ordered_cell_id[1:] != ordered_cell_id[:-1])))
    selected_parts: list[npt.NDArray[np.intp]] = []
    for start in group_starts:
        target_for_cell = int(cell_targets[ordered_cell_id[start]])
        if target_for_cell > 0:
            selected_parts.append(ordered[start : start + target_for_cell])

    # These are original candidate-row positions, currently grouped by cell.
    selected = np.concatenate(selected_parts).astype(np.int64, copy=False)
    # Persisted membership has one canonical order independent of cell traversal.
    point_id_order = np.argsort(point_id[selected], kind="stable")
    return np.ascontiguousarray(selected[point_id_order], dtype=np.int64)


def _microgrid_cell_ids(
    x_rel: npt.NDArray[np.float32],
    y_rel: npt.NDArray[np.float32],
    *,
    tile_size: int,
) -> npt.NDArray[np.int64]:
    """Return one row-major 16-by-16 microgrid cell ID per candidate."""
    cell_x = np.floor(x_rel * SAMPLED_TILE_MICROGRID_EDGE / tile_size).astype(np.int64)
    cell_y = np.floor(y_rel * SAMPLED_TILE_MICROGRID_EDGE / tile_size).astype(np.int64)
    # Coordinates equal to the inclusive upper tile edge initially map to 16.
    np.minimum(cell_x, SAMPLED_TILE_MICROGRID_EDGE - 1, out=cell_x)
    np.minimum(cell_y, SAMPLED_TILE_MICROGRID_EDGE - 1, out=cell_y)
    return np.ascontiguousarray(cell_y * SAMPLED_TILE_MICROGRID_EDGE + cell_x)


def _allocate_cell_targets(
    cell_counts: npt.NDArray[np.int64],
    *,
    target: int,
    level: int,
    tile_x: int,
    tile_y: int,
) -> npt.NDArray[np.int64]:
    """Allocate a tile target proportionally across the fixed microgrid.

    Integer division and remainder implement proportional allocation without
    floating-point rounding. For counts ``[5, 4, 2, 1]`` and target ``7``, the
    base allocations are ``[2, 2, 1, 0]`` and the two remaining slots go to the
    largest remainders, producing ``[3, 2, 1, 1]``.
    """
    total = int(cell_counts.sum())
    base = np.fromiter(
        ((target * int(count)) // total for count in cell_counts),
        dtype=np.int64,
        count=len(cell_counts),
    )
    remainders = np.fromiter(
        ((target * int(count)) % total for count in cell_counts),
        dtype=np.int64,
        count=len(cell_counts),
    )
    remaining = target - int(base.sum())
    if remaining == 0:
        return base

    occupied_cell_id = np.flatnonzero(cell_counts).astype(np.uint64)
    # Leftover slots rank by remainder, versioned pseudo-random cell priority,
    # then numeric cell ID so even an engineered priority collision is stable.
    cell_priority = _cell_tie_break_priorities(
        occupied_cell_id,
        level=level,
        tile_x=tile_x,
        tile_y=tile_y,
    )
    order = np.lexsort(
        (
            occupied_cell_id,
            cell_priority,
            -remainders[occupied_cell_id.astype(np.intp)],
        )
    )
    base[occupied_cell_id[order[:remaining]].astype(np.intp)] += 1
    return base


def _point_priorities(
    point_id: npt.NDArray[np.uint64],
    candidate_cell_id: npt.NDArray[np.int64],
    *,
    level: int,
    tile_x: int,
    tile_y: int,
) -> npt.NDArray[np.uint64]:
    state = _splitmix64(_POINT_PRIORITY_DOMAIN ^ np.uint64(SAMPLING_SEED))
    state = _splitmix64(state ^ np.uint64(level))
    state = _splitmix64(state ^ _tile_key(tile_x, tile_y))
    state = _splitmix64(state ^ candidate_cell_id.astype(np.uint64, copy=False))
    return _splitmix64(state ^ point_id)


def _cell_tie_break_priorities(
    cell_id: npt.NDArray[np.uint64],
    *,
    level: int,
    tile_x: int,
    tile_y: int,
) -> npt.NDArray[np.uint64]:
    """Return versioned priorities for equal-remainder allocation ties."""
    state = _splitmix64(_CELL_PRIORITY_DOMAIN ^ np.uint64(SAMPLING_SEED))
    state = _splitmix64(state ^ np.uint64(level))
    state = _splitmix64(state ^ _tile_key(tile_x, tile_y))
    return _splitmix64(state ^ cell_id)


def _tile_key(tile_x: int, tile_y: int) -> np.uint64:
    return (np.uint64(tile_y) << _UINT64_32) | np.uint64(tile_x)


def _require_array(value: object, name: str, *, dtype: np.dtype[object]) -> None:
    if not isinstance(value, np.ndarray):
        raise ValueError(f"`{name}` must be a NumPy array.")
    if value.ndim != 1 or value.dtype != dtype or not value.flags.c_contiguous:
        raise ValueError(f"`{name}` must be a one-dimensional C-contiguous {dtype.name} array.")
