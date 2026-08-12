from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

from napari_harpy.core.multi_scale_cache_points.build_plan import _LevelBuildPlan, _LevelKind
from napari_harpy.core.multi_scale_cache_points.writer.spatial import (
    _assemble_and_sample_coarser_tile,
    _FinerLevelTile,
)
from napari_harpy.core.multi_scale_cache_points.writer.support import _POINT_PAYLOAD_SCHEMA


def _level(
    *,
    level: int,
    kind: _LevelKind,
    tile_size: int,
    grid_width: int,
    grid_height: int,
    capacity: int,
) -> _LevelBuildPlan:
    return _LevelBuildPlan(
        level=level,
        kind=kind,
        tile_size=tile_size,
        grid_width=grid_width,
        grid_height=grid_height,
        max_points_per_tile=capacity,
        point_count_upper_bound=grid_width * grid_height * capacity,
    )


def _points(
    *,
    x_rel: list[float],
    y_rel: list[float],
    value_ids: list[int],
    point_ids: list[int],
) -> pa.Table:
    return pa.Table.from_arrays(
        [
            pa.array(x_rel, type=pa.float32()),
            pa.array(y_rel, type=pa.float32()),
            pa.array(value_ids, type=pa.uint32()),
            pa.array(point_ids, type=pa.uint64()),
        ],
        schema=_POINT_PAYLOAD_SCHEMA,
    )


def _levels(*, capacity: int = 8_192) -> tuple[_LevelBuildPlan, _LevelBuildPlan]:
    return (
        _level(
            level=1,
            kind=_LevelKind.BRIDGE,
            tile_size=512,
            grid_width=4,
            grid_height=4,
            capacity=4_096,
        ),
        _level(
            level=2,
            kind=_LevelKind.SPATIAL,
            tile_size=1_024,
            grid_width=2,
            grid_height=2,
            capacity=capacity,
        ),
    )


def test_assembly_rebases_four_finer_quadrants_and_orders_by_point_id() -> None:
    finer_level, coarser_level = _levels()
    finer_tiles = (
        _FinerLevelTile(3, 3, _points(x_rel=[40], y_rel=[41], value_ids=[4], point_ids=[40])),
        _FinerLevelTile(2, 2, _points(x_rel=[10], y_rel=[11], value_ids=[1], point_ids=[10])),
        _FinerLevelTile(2, 3, _points(x_rel=[30], y_rel=[31], value_ids=[3], point_ids=[30])),
        _FinerLevelTile(3, 2, _points(x_rel=[20], y_rel=[21], value_ids=[2], point_ids=[20])),
    )

    sampled = _assemble_and_sample_coarser_tile(
        finer_tiles,
        finer_level=finer_level,
        coarser_level=coarser_level,
        coarser_tile_x=1,
        coarser_tile_y=1,
    )

    assert sampled.schema.equals(_POINT_PAYLOAD_SCHEMA, check_metadata=False)
    assert sampled["point_id"].to_pylist() == [10, 20, 30, 40]
    assert sampled["value_id"].to_pylist() == [1, 2, 3, 4]
    assert sampled["x_rel"].to_pylist() == [10.0, 532.0, 30.0, 552.0]
    assert sampled["y_rel"].to_pylist() == [11.0, 21.0, 543.0, 553.0]


def test_dense_assembly_is_deterministic_nested_and_value_neutral() -> None:
    finer_level, coarser_level = _levels(capacity=7)
    finer_tiles = tuple(
        _FinerLevelTile(
            tile_x,
            tile_y,
            _points(
                x_rel=[float((point_id * 37) % 512) for point_id in point_ids],
                y_rel=[float((point_id * 83) % 512) for point_id in point_ids],
                value_ids=[point_id % 3 for point_id in point_ids],
                point_ids=point_ids,
            ),
        )
        for (tile_x, tile_y), point_ids in zip(
            ((0, 0), (1, 0), (0, 1), (1, 1)),
            (list(range(0, 5)), list(range(5, 10)), list(range(10, 15)), list(range(15, 20))),
            strict=True,
        )
    )
    relabeled_tiles = tuple(
        _FinerLevelTile(
            tile.tile_x,
            tile.tile_y,
            tile.points.set_column(
                2,
                _POINT_PAYLOAD_SCHEMA.field("value_id"),
                pa.array(np.full(tile.points.num_rows, 99, dtype=np.uint32), type=pa.uint32()),
            ),
        )
        for tile in reversed(finer_tiles)
    )

    sampled = _assemble_and_sample_coarser_tile(
        finer_tiles,
        finer_level=finer_level,
        coarser_level=coarser_level,
        coarser_tile_x=0,
        coarser_tile_y=0,
    )
    relabeled = _assemble_and_sample_coarser_tile(
        relabeled_tiles,
        finer_level=finer_level,
        coarser_level=coarser_level,
        coarser_tile_x=0,
        coarser_tile_y=0,
    )

    selected_point_ids = sampled["point_id"].to_pylist()
    assert len(selected_point_ids) == 7
    assert selected_point_ids == sorted(selected_point_ids)
    assert set(selected_point_ids) <= set(range(20))
    assert relabeled["point_id"].to_pylist() == selected_point_ids


def test_single_edge_tile_allows_coarser_upper_edge() -> None:
    finer_level, coarser_level = _levels()
    finer_tile = _FinerLevelTile(
        1,
        1,
        _points(x_rel=[512.0], y_rel=[512.0], value_ids=[1], point_ids=[1]),
    )

    sampled = _assemble_and_sample_coarser_tile(
        (finer_tile,),
        finer_level=finer_level,
        coarser_level=coarser_level,
        coarser_tile_x=0,
        coarser_tile_y=0,
    )

    assert sampled["x_rel"].to_pylist() == [1_024.0]
    assert sampled["y_rel"].to_pylist() == [1_024.0]


def test_assembly_rejects_invalid_level_or_finer_tile_geometry() -> None:
    finer_level, coarser_level = _levels()
    tile = _FinerLevelTile(0, 0, _points(x_rel=[1], y_rel=[1], value_ids=[1], point_ids=[1]))

    with pytest.raises(ValueError, match="immediately follow"):
        _assemble_and_sample_coarser_tile(
            (tile,),
            finer_level=finer_level,
            coarser_level=_level(
                level=3,
                kind=_LevelKind.SPATIAL,
                tile_size=1_024,
                grid_width=2,
                grid_height=2,
                capacity=8_192,
            ),
            coarser_tile_x=0,
            coarser_tile_y=0,
        )

    unrelated = _FinerLevelTile(2, 0, tile.points)
    with pytest.raises(ValueError, match="does not contribute"):
        _assemble_and_sample_coarser_tile(
            (unrelated,),
            finer_level=finer_level,
            coarser_level=coarser_level,
            coarser_tile_x=0,
            coarser_tile_y=0,
        )

    with pytest.raises(ValueError, match="unique"):
        _assemble_and_sample_coarser_tile(
            (tile, tile),
            finer_level=finer_level,
            coarser_level=coarser_level,
            coarser_tile_x=0,
            coarser_tile_y=0,
        )

    out_of_grid = _FinerLevelTile(4, 0, tile.points)
    with pytest.raises(ValueError, match="planned grid"):
        _assemble_and_sample_coarser_tile(
            (out_of_grid,),
            finer_level=finer_level,
            coarser_level=coarser_level,
            coarser_tile_x=0,
            coarser_tile_y=0,
        )
