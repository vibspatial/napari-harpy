from __future__ import annotations

import numpy as np
import pytest

import napari_harpy.core.multi_scale_cache_points.sampling as sampling_module
from napari_harpy.core.multi_scale_cache_points.sampling import (
    SAMPLED_TILE_MICROGRID_EDGE,
    SAMPLING_METHOD,
    SAMPLING_SEED,
    _allocate_cell_targets,
    _cell_tie_break_priorities,
    _microgrid_cell_ids,
    _point_priorities,
    _select_sampled_tile_indices,
)


def test_sampling_priorities_have_fixed_vectors() -> None:
    point_ids = np.array([0, 1, 2, 2**64 - 1], dtype=np.uint64)
    candidate_cell_ids = np.array([0, 1, 255, 17], dtype=np.int64)
    cells = np.array([0, 1, 17, 255], dtype=np.uint64)

    assert SAMPLING_METHOD == "harpy-value-neutral-stratified-splitmix64-v1"
    assert SAMPLING_SEED == 0
    assert SAMPLED_TILE_MICROGRID_EDGE == 16
    assert _point_priorities(point_ids, candidate_cell_ids, level=1, tile_x=2, tile_y=3).tolist() == [
        6_201_991_828_904_615_279,
        15_120_207_244_828_415_500,
        11_288_417_978_785_451_448,
        1_846_923_911_944_463_736,
    ]
    assert _cell_tie_break_priorities(cells, level=1, tile_x=2, tile_y=3).tolist() == [
        12_091_186_754_204_247_450,
        11_829_258_099_368_956_577,
        7_792_663_632_242_143_724,
        15_939_108_990_728_393_407,
    ]


def test_sampler_passes_through_sparse_and_empty_tiles() -> None:
    selected = _select_sampled_tile_indices(
        np.array([4.0, 2.0, 3.0], dtype=np.float32),
        np.array([1.0, 1.0, 1.0], dtype=np.float32),
        np.array([9, 2, 7], dtype=np.uint64),
        level=1,
        tile_x=0,
        tile_y=0,
        tile_size=512,
        target=3,
    )
    empty = _select_sampled_tile_indices(
        np.array([], dtype=np.float32),
        np.array([], dtype=np.float32),
        np.array([], dtype=np.uint64),
        level=1,
        tile_x=0,
        tile_y=0,
        tile_size=512,
        target=4_096,
    )

    assert selected.tolist() == [1, 2, 0]
    assert selected.dtype == np.intp
    assert empty.tolist() == []
    assert empty.dtype == np.intp


def test_sampler_allocates_proportionally_across_microgrid_cells() -> None:
    cell_counts = (5, 4, 2, 1)
    x = np.concatenate(
        [
            np.full(count, coordinate, dtype=np.float32)
            for count, coordinate in zip(cell_counts, (16.0, 48.0, 80.0, 112.0), strict=True)
        ]
    )
    y = np.full(len(x), 16.0, dtype=np.float32)
    point_ids = np.arange(len(x), dtype=np.uint64)

    selected = _select_sampled_tile_indices(
        x,
        y,
        point_ids,
        level=1,
        tile_x=0,
        tile_y=0,
        tile_size=512,
        target=7,
    )
    selected_cells = _microgrid_cell_ids(x[selected].astype(np.float64), y[selected].astype(np.float64), tile_size=512)

    assert len(selected) == 7
    assert np.bincount(selected_cells, minlength=4)[:4].tolist() == [3, 2, 1, 1]


def test_microgrid_scales_with_tile_and_clamps_only_the_upper_edge() -> None:
    bridge_cells = _microgrid_cell_ids(
        np.array([0.0, 31.999, 32.0, 511.999, 512.0]),
        np.zeros(5),
        tile_size=512,
    )
    l1_cells = _microgrid_cell_ids(
        np.array([0.0, 63.999, 64.0, 1_023.999, 1_024.0]),
        np.zeros(5),
        tile_size=1_024,
    )

    assert bridge_cells.tolist() == [0, 0, 1, 15, 15]
    assert l1_cells.tolist() == [0, 0, 1, 15, 15]


def test_equal_remainders_use_deterministic_cell_priority(monkeypatch: pytest.MonkeyPatch) -> None:
    counts = np.zeros(256, dtype=np.int64)
    counts[[0, 1, 17, 255]] = 1

    targets = _allocate_cell_targets(counts, target=2, level=1, tile_x=2, tile_y=3)

    assert np.flatnonzero(targets).tolist() == [1, 17]

    monkeypatch.setattr(
        sampling_module,
        "_cell_tie_break_priorities",
        lambda cell_ids, *_args, **_kwargs: np.zeros(len(cell_ids), dtype=np.uint64),
    )
    collision_targets = _allocate_cell_targets(counts, target=2, level=1, tile_x=2, tile_y=3)

    assert np.flatnonzero(collision_targets).tolist() == [0, 1]


def test_membership_is_invariant_to_input_order() -> None:
    point_ids = np.arange(300, dtype=np.uint64)
    x = ((point_ids * np.uint64(37)) % np.uint64(512)).astype(np.float32)
    y = ((point_ids * np.uint64(83)) % np.uint64(512)).astype(np.float32)
    permutation = np.arange(len(point_ids) - 1, -1, -1)

    selected = _select_sampled_tile_indices(
        x,
        y,
        point_ids,
        level=1,
        tile_x=4,
        tile_y=7,
        tile_size=512,
        target=75,
    )
    permuted_selected = _select_sampled_tile_indices(
        x[permutation],
        y[permutation],
        point_ids[permutation],
        level=1,
        tile_x=4,
        tile_y=7,
        tile_size=512,
        target=75,
    )

    assert point_ids[selected].tolist() == point_ids[permutation][permuted_selected].tolist()


def test_point_id_breaks_controlled_priority_collision(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sampling_module,
        "_point_priorities",
        lambda point_ids, *_args, **_kwargs: np.zeros(len(point_ids), dtype=np.uint64),
    )
    point_ids = np.array([8, 2, 5, 1, 9], dtype=np.uint64)

    selected = _select_sampled_tile_indices(
        np.ones(5, dtype=np.float32),
        np.ones(5, dtype=np.float32),
        point_ids,
        level=1,
        tile_x=0,
        tile_y=0,
        tile_size=512,
        target=3,
    )

    assert point_ids[selected].tolist() == [1, 2, 5]


def test_sampler_rejects_invalid_core_inputs() -> None:
    valid_ids = np.array([0, 1], dtype=np.uint64)

    with pytest.raises(ValueError, match="matching lengths"):
        _select_sampled_tile_indices(
            np.array([1.0]),
            np.array([1.0, 2.0]),
            valid_ids,
            level=1,
            tile_x=0,
            tile_y=0,
            tile_size=512,
            target=1,
        )
    with pytest.raises(ValueError, match="dtype uint64"):
        _select_sampled_tile_indices(
            np.array([1.0]),
            np.array([1.0]),
            np.array([0]),
            level=1,
            tile_x=0,
            tile_y=0,
            tile_size=512,
            target=1,
        )
    with pytest.raises(ValueError, match="closed interval"):
        _select_sampled_tile_indices(
            np.array([513.0]),
            np.array([1.0]),
            np.array([0], dtype=np.uint64),
            level=1,
            tile_x=0,
            tile_y=0,
            tile_size=512,
            target=1,
        )
