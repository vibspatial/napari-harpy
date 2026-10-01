"""Exercise state-dependent LOD decisions independently of storage and Qt."""

from itertools import product

import pytest

from spatiato.core.multi_scale_cache_points_zarr.reader import _LevelSelection
from spatiato.viewer.tiled_points.runtime.lod import _lod_thresholds, _select_lod


def _candidates(counts, preferred):
    return tuple(
        _LevelSelection(level, count, int(count > 0), count <= preferred, None) for level, count in enumerate(counts)
    )


@pytest.mark.parametrize(
    ("counts", "preferred", "hard", "previous", "level", "reason"),
    [
        ((40, 5), 50, 100, None, 0, "ordinary"),
        ((60, 8), 50, 100, 0, 0, "hysteresis_stay"),
        ((75, 9), 50, 100, 0, 0, "hysteresis_stay"),
        ((76, 10), 50, 100, 0, 1, "hysteresis_coarsen"),
        ((70, 9), 50, 100, 1, 1, "hysteresis_stay"),
        ((51, 6), 50, 100, 1, 1, "hysteresis_stay"),
        ((50, 6), 50, 100, 1, 0, "hysteresis_refine"),
        ((80, 70, 5), 50, 100, 0, 1, "hysteresis_coarsen"),
        ((90, 80, 70, 5), 50, 100, 0, 2, "hysteresis_coarsen"),
        ((40, 30, 20, 5), 50, 100, 3, 0, "hysteresis_refine"),
        # Counts can rise at coarser levels: refinement still takes precedence.
        ((40, 90, 100), 50, 100, 1, 0, "hysteresis_refine"),
        # Hard capacity clips the upper bound: the refinement boundary is 40.
        ((41, 5), 50, 50, 1, 1, "hysteresis_stay"),
        ((40, 5), 50, 50, 1, 0, "hysteresis_refine"),
        ((51, 5), 50, 50, 0, 1, "hysteresis_coarsen"),
        # No stay/coarser fit, but an earlier candidate fits the preference.
        ((45, 60, 70), 50, 50, 1, 0, "ordinary"),
        ((120, 90), 50, 100, None, 1, "coarsest_density_fallback"),
        ((120, 100), 50, 100, 0, 1, "coarsest_density_fallback"),
        ((120, 101), 50, 100, 0, 1, "hard_limit"),
        ((120, 101), 50, 100, None, 1, "hard_limit"),
        ((0, 3), 50, 100, 1, 0, "hysteresis_refine"),
        ((1, 0), 1, 1, 1, 0, "ordinary"),
        ((2, 1), 1, 100, 1, 1, "ordinary"),
    ],
)
def test_lod_transition_boundaries_and_fallbacks(counts, preferred, hard, previous, level, reason):
    candidates = _candidates(counts, preferred)
    chosen, actual_reason = _select_lod(
        iter(candidates), preferred_point_budget=preferred, hard_point_capacity=hard, previous_level=previous
    )
    assert chosen is candidates[level]  # Keep all of the chosen level's evidence.
    assert actual_reason == reason
    assert chosen.fits_point_budget == (counts[level] <= preferred)
    assert (chosen.estimated_point_count > hard) == (reason == "hard_limit")


@pytest.mark.parametrize(
    ("preferred", "hard", "expected"),
    [(50, 100, (75, 50)), (50, 50, (50, 40)), (3, 10, (4, 3)), (2, 2, (2, 1)), (1, 1, (1, 0))],
)
def test_thresholds_use_floor_rounding_and_hard_capacity(preferred, hard, expected):
    assert _lod_thresholds(preferred, hard) == expected


@pytest.mark.parametrize(
    ("counts", "previous", "expected", "consumed"),
    [((40, 20, 10), None, 0, [0]), ((60, 40, 20), 2, 1, [0, 1]), ((60, 40, 20), 0, 0, [0])],
)
def test_lod_decision_stops_consuming_candidates_as_soon_as_known(counts, previous, expected, consumed):
    evaluated = []

    def candidates():
        for candidate in _candidates(counts, 50):
            evaluated.append(candidate.level)
            yield candidate

    chosen, _ = _select_lod(candidates(), preferred_point_budget=50, hard_point_capacity=100, previous_level=previous)
    assert chosen.level == expected
    assert evaluated == consumed


def test_identical_estimates_do_not_oscillate_after_acceptance():
    """Include non-monotonic counts, zero estimates, and clipped integer bands."""
    for hard in range(1, 6):
        for preferred in range(1, hard + 1):
            for counts in product(range(7), repeat=3):
                candidates = _candidates(counts, preferred)
                ordinary, ordinary_reason = _select_lod(
                    candidates, preferred_point_budget=preferred, hard_point_capacity=hard, previous_level=None
                )
                for previous in (None, 0, 1, 2):
                    chosen, reason = _select_lod(
                        candidates, preferred_point_budget=preferred, hard_point_capacity=hard, previous_level=previous
                    )
                    if ordinary_reason != "hard_limit":
                        assert reason != "hard_limit", (counts, preferred, hard, previous, ordinary)
                    if reason == "hard_limit":
                        assert chosen.estimated_point_count > hard
                        continue
                    assert chosen.estimated_point_count <= hard
                    repeated, _ = _select_lod(
                        candidates,
                        preferred_point_budget=preferred,
                        hard_point_capacity=hard,
                        previous_level=chosen.level,
                    )
                    assert repeated is chosen, (counts, preferred, hard, previous)
