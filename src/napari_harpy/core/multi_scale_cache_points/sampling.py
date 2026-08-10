from __future__ import annotations

import numpy as np

from napari_harpy.core.multi_scale_cache_points.hashing import _splitmix64

SAMPLING_METHOD = "harpy-value-neutral-stratified-splitmix64-v1"
SAMPLING_SEED = 0
SAMPLED_TILE_MICROGRID_EDGE = 16

_POINT_PRIORITY_DOMAIN = np.uint64(0x48504F494E543031)  # "HPOINT01"
_CELL_PRIORITY_DOMAIN = np.uint64(0x4843454C4C303031)  # "HCELL001"
_UINT64_32 = np.uint64(32)
_MAX_SERIALIZED_LEVEL = 2**15 - 1
_MAX_TILE_COORDINATE = 2**32 - 1
_MICROGRID_CELL_COUNT = SAMPLED_TILE_MICROGRID_EDGE**2


def _select_sampled_tile_indices(
    x_rel: np.ndarray,
    y_rel: np.ndarray,
    point_ids: np.ndarray,
    *,
    level: int,
    tile_x: int,
    tile_y: int,
    tile_size: int,
    target: int,
) -> np.ndarray:
    """Select deterministic spatial representatives from one logical tile.

    Parameters
    ----------
    x_rel, y_rel
        One-dimensional numeric coordinate arrays relative to the current
        logical output tile. Values must be finite and lie in the closed
        interval ``[0, tile_size]``. The upper edge accommodates float32
        rounding and is assigned to the last microgrid cell.
    point_ids
        One-dimensional ``uint64`` identities corresponding to the coordinate
        rows. Their uniqueness is guaranteed by upstream cache construction.
    level
        Non-negative serialized cache level in the supported int16 range.
    tile_x, tile_y
        Non-negative uint32 coordinates of the current logical output tile.
    tile_size
        Positive edge length of the current logical tile in intrinsic source
        coordinates.
    target
        Positive maximum number of representatives to retain.

    Returns
    -------
    numpy.ndarray
        Original row indices with dtype ``np.intp``, ordered by ascending
        ``point_id``. Exactly ``min(candidate_count, target)`` rows are
        returned.

    Notes
    -----
    Logical tiles are persistent storage, manifest, and loading units. The
    16-by-16 microgrid is only a transient sampling structure inside the
    current tile; microgrid cells are not cache tiles::

        Level    tile edge    microgrid    cell edge
        Bridge         512      16 x 16           32
        L1           1,024      16 x 16           64
        L2           2,048      16 x 16          128

    Four bridge tiles form one L1 parent. After child coordinates are rebased
    into that parent, each child covers one 8-by-8 quadrant of the L1
    microgrid::

        L1 microgrid: 16 x 16
        +---------+---------+
        | 8 x 8   | 8 x 8   |
        +---------+---------+
        | 8 x 8   | 8 x 8   |
        +---------+---------+

    Candidate counts determine proportional cell allocations. ``value_id`` is
    deliberately absent, so categorical values cannot influence membership.
    """
    x = np.asarray(x_rel)
    y = np.asarray(y_rel)
    ids = np.asarray(point_ids)
    if x.ndim != 1 or y.ndim != 1 or ids.ndim != 1:
        raise ValueError("`x_rel`, `y_rel`, and `point_ids` must be one-dimensional arrays.")
    if not (len(x) == len(y) == len(ids)):
        raise ValueError("`x_rel`, `y_rel`, and `point_ids` must have matching lengths.")
    if not _is_real_numeric_dtype(x.dtype) or not _is_real_numeric_dtype(y.dtype):
        raise ValueError("`x_rel` and `y_rel` must be numeric arrays with real-valued dtypes.")
    if ids.dtype != np.dtype(np.uint64):
        raise ValueError("`point_ids` must have dtype uint64.")

    _require_integer_in_range(level, "level", maximum=_MAX_SERIALIZED_LEVEL)
    _require_integer_in_range(tile_x, "tile_x", maximum=_MAX_TILE_COORDINATE)
    _require_integer_in_range(tile_y, "tile_y", maximum=_MAX_TILE_COORDINATE)
    _require_positive_integer(tile_size, "tile_size")
    _require_positive_integer(target, "target")

    x_float64 = x.astype(np.float64, copy=False)
    y_float64 = y.astype(np.float64, copy=False)
    if not bool(np.isfinite(x_float64).all()) or not bool(np.isfinite(y_float64).all()):
        raise ValueError("Relative coordinates must be finite.")
    if (
        bool((x_float64 < 0).any())
        or bool((x_float64 > tile_size).any())
        or bool((y_float64 < 0).any())
        or bool((y_float64 > tile_size).any())
    ):
        raise ValueError("Relative coordinates must lie in the closed interval [0, tile_size].")

    candidate_count = len(ids)
    if candidate_count == 0:
        return np.empty(0, dtype=np.intp)
    # `target` is a maximum tile capacity, not a requested sample size. When
    # every candidate already fits, retain them all and only restore the
    # deterministic point-ID order.
    if candidate_count <= target:
        return np.argsort(ids, kind="stable").astype(np.intp, copy=False)

    # Map every candidate to one cell, then turn those point-level IDs into the
    # fixed 256-entry histogram used to allocate the tile's sampling budget.
    # Empty microgrid cells remain present with count zero.
    candidate_cell_ids = _microgrid_cell_ids(x_float64, y_float64, tile_size=tile_size)
    cell_counts = np.bincount(candidate_cell_ids, minlength=_MICROGRID_CELL_COUNT)
    # `cell_targets[cell_id]` is the representative quota allocated to that
    # microgrid cell.
    cell_targets = _allocate_cell_targets(
        cell_counts,
        target=target,
        level=level,
        tile_x=tile_x,
        tile_y=tile_y,
    )
    # `ids` and `candidate_cell_ids` are parallel point-level arrays, so hash
    # one deterministic priority per candidate for ranking within its assigned
    # cell.
    candidate_priorities = _point_priorities(
        ids,
        candidate_cell_ids,
        level=level,
        tile_x=tile_x,
        tile_y=tile_y,
    )

    # `np.lexsort` treats the last key as primary.
    ordered = np.lexsort(
        (
            ids,  # tertiary: deterministic tie-break for priority collisions
            candidate_priorities,  # secondary: rank candidates within each cell
            candidate_cell_ids,  # primary: make every cell's candidates contiguous
        )
    )
    # Because cell ID was the primary sort key, applying `ordered` groups all
    # candidates from the same cell together, for example [0, 0, 2, 2, 2].
    ordered_candidate_cell_ids = candidate_cell_ids[ordered]
    # Find the first position of each cell group by comparing adjacent IDs.
    # For [0, 0, 2, 2, 2], the group-start mask is
    # [True, False, True, False, False], which gives positions [0, 2].
    group_starts = np.flatnonzero(
        np.concatenate((np.array([True]), ordered_candidate_cell_ids[1:] != ordered_candidate_cell_ids[:-1]))
    )
    selected_parts: list[np.ndarray] = []
    for start in group_starts:
        target_for_cell = int(cell_targets[ordered_candidate_cell_ids[start]])
        if target_for_cell > 0:
            selected_parts.append(ordered[start : start + target_for_cell])
    # `selected` contains positions into the original candidate arrays for all
    # retained points. At this stage they are grouped by microgrid cell and
    # ranked within each cell by their deterministic sampling priority.
    selected = np.concatenate(selected_parts).astype(np.intp, copy=False)
    # Order the selected original row positions by point_id before returning them.
    selected_point_ids = ids[selected]
    point_id_order = np.argsort(selected_point_ids, kind="stable")
    return selected[point_id_order]


def _microgrid_cell_ids(x_rel: np.ndarray, y_rel: np.ndarray, *, tile_size: int) -> np.ndarray:
    """Return one flattened 16-by-16 microgrid cell ID per candidate.

    Cell IDs range from 0 through 255 and are laid out row by row as
    ``cell_y * 16 + cell_x``. The returned array has the same length as the
    coordinate arrays and may contain duplicate IDs. For example, five
    candidates assigned to cells ``[0, 18, 63, 18, 0]`` produce counts of two
    in cell 0, two in cell 18, and one in cell 63.
    """
    cell_x = np.floor(x_rel * SAMPLED_TILE_MICROGRID_EDGE / tile_size).astype(np.int64)
    cell_y = np.floor(y_rel * SAMPLED_TILE_MICROGRID_EDGE / tile_size).astype(np.int64)
    np.minimum(cell_x, SAMPLED_TILE_MICROGRID_EDGE - 1, out=cell_x)
    np.minimum(cell_y, SAMPLED_TILE_MICROGRID_EDGE - 1, out=cell_y)
    return cell_y * SAMPLED_TILE_MICROGRID_EDGE + cell_x


def _allocate_cell_targets(
    cell_counts: np.ndarray,
    *,
    target: int,
    level: int,
    tile_x: int,
    tile_y: int,
) -> np.ndarray:
    """Allocate a tile's representative target proportionally across cells.

    Parameters
    ----------
    cell_counts
        Candidate counts indexed by flattened microgrid cell ID. The expected
        16-by-16 grid therefore has 256 entries, including zeros for empty
        cells.
    target
        Total number of representatives to allocate across the tile. The
        caller guarantees that it is positive and no greater than the sum of
        ``cell_counts``.
    level, tile_x, tile_y
        Current logical tile identity used only for deterministic ordering when
        cells have equal proportional remainders.

    Returns
    -------
    numpy.ndarray
        One integer allocation per cell. The allocations sum to ``target`` and
        never exceed the corresponding candidate counts.

    Notes
    -----
    Allocation uses integer largest remainders. For candidate counts
    ``[5, 4, 2, 1]`` (total 12) and target 7, applying
    ``divmod(target * count, total)`` gives base allocations ``[2, 2, 1, 0]``
    and remainders ``[11, 4, 2, 7]``. The two unallocated slots go to the
    largest remainders, producing ``[3, 2, 1, 1]``.
    """
    total = int(cell_counts.sum())
    # Split each ideal allocation (`target * count / total`) into its integer
    # floor and remainder without introducing floating-point rounding.
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

    # Give leftover slots to occupied cells with the largest remainders. Stable
    # tie-break priorities, then cell IDs, resolve exact ties without repeatedly
    # preferring the lowest-numbered cells.
    occupied_cell_ids = np.flatnonzero(cell_counts).astype(np.uint64)
    cell_tie_break_priorities = _cell_tie_break_priorities(
        occupied_cell_ids,
        level=level,
        tile_x=tile_x,
        tile_y=tile_y,
    )
    # `np.lexsort` treats the last key as primary.
    order = np.lexsort(
        (
            occupied_cell_ids,  # third and final tie-breaker
            cell_tie_break_priorities,  # second tie-breaker
            -remainders[occupied_cell_ids.astype(np.intp)],  # primary: largest remainder first
        )
    )
    base[occupied_cell_ids[order[:remaining]].astype(np.intp)] += 1
    return base


def _point_priorities(
    point_ids: np.ndarray,
    candidate_cell_ids: np.ndarray,
    *,
    level: int,
    tile_x: int,
    tile_y: int,
) -> np.ndarray:
    state = _splitmix64(_POINT_PRIORITY_DOMAIN ^ np.uint64(SAMPLING_SEED))
    state = _splitmix64(state ^ np.uint64(level))
    state = _splitmix64(state ^ _tile_key(tile_x, tile_y))
    state = _splitmix64(state ^ candidate_cell_ids.astype(np.uint64, copy=False))
    return _splitmix64(state ^ point_ids)


def _cell_tie_break_priorities(
    cell_ids: np.ndarray,
    *,
    level: int,
    tile_x: int,
    tile_y: int,
) -> np.ndarray:
    """Return deterministic priorities for cells with equal remainders.

    Each occupied microgrid cell receives one random-looking ``uint64`` derived
    from the fixed cell-priority domain, sampling seed, current level, logical
    tile key, and cell ID. The allocation step consults these values only when
    two cells have the same proportional remainder; lower priorities rank
    first. Numeric cell ID remains the final tie-breaker for a priority
    collision.

    Including the current level and tile key varies tie resolution across the
    cache instead of repeatedly favoring the same low-numbered spatial cells.
    The calculation is deterministic and does not use point values.
    """
    state = _splitmix64(_CELL_PRIORITY_DOMAIN ^ np.uint64(SAMPLING_SEED))
    state = _splitmix64(state ^ np.uint64(level))
    state = _splitmix64(state ^ _tile_key(tile_x, tile_y))
    return _splitmix64(state ^ cell_ids.astype(np.uint64, copy=False))


def _tile_key(tile_x: int, tile_y: int) -> np.uint64:
    return (np.uint64(tile_y) << _UINT64_32) | np.uint64(tile_x)


def _is_real_numeric_dtype(dtype: np.dtype[object]) -> bool:
    return bool(np.issubdtype(dtype, np.number) and not np.issubdtype(dtype, np.complexfloating))


def _require_positive_integer(value: object, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"`{name}` must be a positive integer.")


def _require_integer_in_range(value: object, name: str, *, maximum: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= maximum:
        raise ValueError(f"`{name}` must be an integer in the range [0, {maximum}].")
