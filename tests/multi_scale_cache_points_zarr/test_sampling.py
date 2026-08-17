from __future__ import annotations

import numpy as np
import pytest

import napari_harpy.core.multi_scale_cache_points_zarr.sampling as sampling_module
from napari_harpy.core.multi_scale_cache_points_zarr.sampling import (
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
    point_id = np.array([0, 1, 2, 2**64 - 1], dtype=np.uint64)
    candidate_cell_id = np.array([0, 1, 255, 17], dtype=np.int64)
    cell_id = np.array([0, 1, 17, 255], dtype=np.uint64)

    assert SAMPLING_METHOD == "harpy-value-neutral-stratified-splitmix64-v1"
    assert SAMPLING_SEED == 0
    assert SAMPLED_TILE_MICROGRID_EDGE == 16
    assert _point_priorities(point_id, candidate_cell_id, level=1, tile_x=2, tile_y=3).tolist() == [
        6_201_991_828_904_615_279,
        15_120_207_244_828_415_500,
        11_288_417_978_785_451_448,
        1_846_923_911_944_463_736,
    ]
    assert _cell_tie_break_priorities(cell_id, level=1, tile_x=2, tile_y=3).tolist() == [
        12_091_186_754_204_247_450,
        11_829_258_099_368_956_577,
        7_792_663_632_242_143_724,
        15_939_108_990_728_393_407,
    ]


def test_sampler_passes_through_sparse_and_empty_tiles_in_point_id_order() -> None:
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
        np.empty(0, dtype=np.float32),
        np.empty(0, dtype=np.float32),
        np.empty(0, dtype=np.uint64),
        level=1,
        tile_x=0,
        tile_y=0,
        tile_size=512,
        target=4_096,
    )

    assert selected.tolist() == [1, 2, 0]
    assert selected.dtype == np.dtype(np.int64)
    assert selected.flags.c_contiguous
    assert empty.tolist() == []
    assert empty.dtype == np.dtype(np.int64)


def test_sampler_allocates_proportionally_and_clamps_upper_edge() -> None:
    cell_counts = (5, 4, 2, 1)
    x = np.concatenate(
        [
            np.full(count, coordinate, dtype=np.float32)
            for count, coordinate in zip(cell_counts, (16.0, 48.0, 80.0, 112.0), strict=True)
        ]
    )
    y = np.full(len(x), 16.0, dtype=np.float32)
    point_id = np.arange(len(x), dtype=np.uint64)

    selected = _select_sampled_tile_indices(
        x,
        y,
        point_id,
        level=1,
        tile_x=0,
        tile_y=0,
        tile_size=512,
        target=7,
    )
    selected_cells = _microgrid_cell_ids(x[selected], y[selected], tile_size=512)
    edge_cells = _microgrid_cell_ids(
        np.array([0.0, 31.999, 32.0, 511.999, 512.0], dtype=np.float32),
        np.zeros(5, dtype=np.float32),
        tile_size=512,
    )

    assert len(selected) == 7
    assert np.bincount(selected_cells, minlength=4)[:4].tolist() == [3, 2, 1, 1]
    assert edge_cells.tolist() == [0, 0, 1, 15, 15]


def test_equal_remainders_and_priority_collisions_have_fixed_ties(monkeypatch: pytest.MonkeyPatch) -> None:
    counts = np.zeros(256, dtype=np.int64)
    counts[[0, 1, 17, 255]] = 1

    targets = _allocate_cell_targets(counts, target=2, level=1, tile_x=2, tile_y=3)
    assert np.flatnonzero(targets).tolist() == [1, 17]

    monkeypatch.setattr(
        sampling_module,
        "_cell_tie_break_priorities",
        lambda cell_id, **_kwargs: np.zeros(len(cell_id), dtype=np.uint64),
    )
    collision_targets = _allocate_cell_targets(counts, target=2, level=1, tile_x=2, tile_y=3)
    assert np.flatnonzero(collision_targets).tolist() == [0, 1]


def test_membership_is_input_order_and_value_independent_by_api() -> None:
    point_id = np.arange(300, dtype=np.uint64)
    x = ((point_id * np.uint64(37)) % np.uint64(512)).astype(np.float32)
    y = ((point_id * np.uint64(83)) % np.uint64(512)).astype(np.float32)
    permutation = np.arange(len(point_id) - 1, -1, -1)

    selected = _select_sampled_tile_indices(
        x,
        y,
        point_id,
        level=1,
        tile_x=4,
        tile_y=7,
        tile_size=512,
        target=75,
    )
    permuted = _select_sampled_tile_indices(
        np.ascontiguousarray(x[permutation]),
        np.ascontiguousarray(y[permutation]),
        np.ascontiguousarray(point_id[permutation]),
        level=1,
        tile_x=4,
        tile_y=7,
        tile_size=512,
        target=75,
    )

    assert point_id[selected].tolist() == point_id[permutation][permuted].tolist()


def test_point_id_breaks_controlled_point_priority_collision(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sampling_module,
        "_point_priorities",
        lambda point_id, *_args, **_kwargs: np.zeros(len(point_id), dtype=np.uint64),
    )
    point_id = np.array([8, 2, 5, 1, 9], dtype=np.uint64)
    selected = _select_sampled_tile_indices(
        np.ones(5, dtype=np.float32),
        np.ones(5, dtype=np.float32),
        point_id,
        level=1,
        tile_x=0,
        tile_y=0,
        tile_size=512,
        target=3,
    )

    assert point_id[selected].tolist() == [1, 2, 5]


@pytest.mark.parametrize(
    ("x", "y", "point_id", "match"),
    [
        (np.array([1.0]), np.array([1.0], dtype=np.float32), np.array([0], dtype=np.uint64), "x_rel"),
        (
            np.array([1.0], dtype=np.float32),
            np.array([1.0]),
            np.array([0], dtype=np.uint64),
            "y_rel",
        ),
        (
            np.array([1.0], dtype=np.float32),
            np.array([1.0], dtype=np.float32),
            np.array([0]),
            "point_id",
        ),
    ],
)
def test_sampler_rejects_noncanonical_dtypes(
    x: np.ndarray,
    y: np.ndarray,
    point_id: np.ndarray,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        _select_sampled_tile_indices(
            x,
            y,
            point_id,
            level=1,
            tile_x=0,
            tile_y=0,
            tile_size=512,
            target=1,
        )


def test_sampler_rejects_invalid_shape_coordinates_and_identity() -> None:
    x = np.array([1.0], dtype=np.float32)
    y = np.array([1.0], dtype=np.float32)
    point_id = np.array([0], dtype=np.uint64)

    with pytest.raises(ValueError, match="matching lengths"):
        _select_sampled_tile_indices(
            x,
            np.array([1.0, 2.0], dtype=np.float32),
            point_id,
            level=1,
            tile_x=0,
            tile_y=0,
            tile_size=512,
            target=1,
        )
    with pytest.raises(ValueError, match="closed interval"):
        _select_sampled_tile_indices(
            np.array([513.0], dtype=np.float32),
            y,
            point_id,
            level=1,
            tile_x=0,
            tile_y=0,
            tile_size=512,
            target=1,
        )
    with pytest.raises(ValueError, match="finite"):
        _select_sampled_tile_indices(
            np.array([np.nan], dtype=np.float32),
            y,
            point_id,
            level=1,
            tile_x=0,
            tile_y=0,
            tile_size=512,
            target=1,
        )
    with pytest.raises(ValueError, match="level"):
        _select_sampled_tile_indices(
            x,
            y,
            point_id,
            level=-1,
            tile_x=0,
            tile_y=0,
            tile_size=512,
            target=1,
        )
    with pytest.raises(ValueError, match="target"):
        _select_sampled_tile_indices(
            x,
            y,
            point_id,
            level=1,
            tile_x=0,
            tile_y=0,
            tile_size=512,
            target=0,
        )
