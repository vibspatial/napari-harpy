"""Choose a viewport LOD without owning reader, renderer, or history state."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

from napari_harpy.core.multi_scale_cache_points_zarr.reader import _LevelSelection

_LodDecisionReason = Literal[
    "ordinary",
    "hysteresis_refine",
    "hysteresis_stay",
    "hysteresis_coarsen",
    "coarsest_density_fallback",
    "hard_limit",
]


def _lod_thresholds(preferred_point_budget: int, hard_point_capacity: int) -> tuple[int, int]:
    """Convert two point budgets into inclusive LOD switching thresholds.

    Inputs and derived thresholds (all measured in points)::

        P = preferred_point_budget: desired count, capped at H
        H = hard_point_capacity: maximum allowed by both the point limit
                                 and the vertex-byte limit

        upper      = min(floor(1.5 * P), H)
        refinement = min(P, floor(0.8 * upper))

    ``_select_lod()`` compares fresh estimates for the new viewport with these
    thresholds, not the size of the previously retained batch:

    - ``upper``: the accepted level may stay when its estimate is at or below
      this count; hysteresis coarsening candidates use the same threshold.
    - ``refinement``: a candidate finer level must fit this count to trigger
      hysteresis refinement, for example from Bridge to Exact.

    Example thresholds with and without hard-capacity headroom::

        P           H           upper       refinement
        500,000     2,000,000   750,000     500,000
        500,000       500,000   500,000     400,000

    Refinement normally uses the preferred budget P. When hard capacity clips
    the upper threshold, the 0.8 guard can lower refinement below P to preserve
    a switching gap. These policy coefficients never relax hard capacity.
    This helper only calculates thresholds; it does not choose a level.
    """
    upper = min(3 * preferred_point_budget // 2, hard_point_capacity)
    return upper, min(preferred_point_budget, 4 * upper // 5)


def _select_lod(
    candidates: Iterable[_LevelSelection],
    *,
    preferred_point_budget: int,
    hard_point_capacity: int,
    previous_level: int | None,
) -> tuple[_LevelSelection, _LodDecisionReason]:
    """Consume fresh level estimates until the viewport's LOD is determined.

    ``candidates`` visits every serialized level in finest-to-coarsest order.
    The positive preferred budget is at most hard capacity; candidates retain
    their fit against that preference, independently of this policy's choice.
    ``previous_level`` is the last accepted level for the same cache/selection,
    or ``None`` for ordinary first-fit selection. It is not pending work or a
    claim that the previous packed batch is reusable.

    Refine to the finest finer candidate meeting the lower threshold; otherwise
    stay if the current level fits the upper threshold, then try coarser levels.
    If none succeeds, use ordinary selection or the hard-valid coarsest fallback.
    Remember the first preferred fit during the same pass so that this escape
    does not repeat metadata work. Non-monotonic counts are valid.

    Only the worker's later activation acknowledgement advances history. The
    returned reason distinguishes tolerated density from a coarsest fallback;
    a ``hard_limit`` decision describes a rejection, not a renderable payload.
    """
    upper, refinement = _lod_thresholds(preferred_point_budget, hard_point_capacity)
    # A one-point upper threshold has no useful positive-integer switching gap.
    if upper == 1:
        previous_level = None
    first_preferred = None
    coarsest = None
    for candidate in candidates:
        coarsest = candidate
        count = candidate.estimated_point_count
        if count <= preferred_point_budget:
            if previous_level is None:
                return candidate, "ordinary"
            if first_preferred is None:
                first_preferred = candidate
        if previous_level is None:
            continue
        if candidate.level < previous_level:
            if count <= refinement:
                return candidate, "hysteresis_refine"
        elif candidate.level == previous_level:
            if count <= upper:
                return candidate, "hysteresis_stay"
        elif count <= upper:
            return candidate, "hysteresis_coarsen"

    if first_preferred is not None:
        return first_preferred, "ordinary"
    if coarsest is None:
        raise RuntimeError("Cache has no serialized levels.")
    if coarsest.estimated_point_count <= hard_point_capacity:
        return coarsest, "coarsest_density_fallback"
    return coarsest, "hard_limit"
