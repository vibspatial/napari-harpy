from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import napari_harpy.core.multi_scale_cache_points.writer.spatial as spatial_writer_module
from napari_harpy.core.multi_scale_cache_points.build_plan import (
    _LevelBuildPlan,
    _LevelKind,
    _PointsCacheBuildPlan,
)
from napari_harpy.core.multi_scale_cache_points.writer.models import _LevelWriteResult, _ManifestRow
from napari_harpy.core.multi_scale_cache_points.writer.spatial import (
    _assemble_and_sample_coarser_tile,
    _FinerLevelTile,
    _write_spatial_level,
    _write_spatial_levels,
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


def _persistent_plan(*, spatial: bool = True) -> _PointsCacheBuildPlan:
    exact = _LevelBuildPlan(
        level=0,
        kind=_LevelKind.EXACT,
        tile_size=512,
        grid_width=4,
        grid_height=4,
        max_points_per_tile=None,
        point_count_upper_bound=80,
    )
    bridge = _level(
        level=1,
        kind=_LevelKind.BRIDGE,
        tile_size=512,
        grid_width=4,
        grid_height=4,
        capacity=5,
    )
    levels = [exact, bridge]
    if spatial:
        levels.extend(
            [
                _level(
                    level=2,
                    kind=_LevelKind.SPATIAL,
                    tile_size=1_024,
                    grid_width=2,
                    grid_height=2,
                    capacity=7,
                ),
                _level(
                    level=3,
                    kind=_LevelKind.SPATIAL,
                    tile_size=2_048,
                    grid_width=1,
                    grid_height=1,
                    capacity=9,
                ),
            ]
        )
    return _PointsCacheBuildPlan(
        x_origin=0.0,
        y_origin=0.0,
        leaf_tile_size=512,
        overview_point_budget=9 if spatial else 80,
        levels=tuple(levels),
    )


def _write_bridge_fixture(staging: Path, *, value_variant: int = 0) -> _LevelWriteResult:
    staging.mkdir()
    level_directory = staging / "levels/level_1"
    level_directory.mkdir(parents=True)
    relative_path = "levels/level_1/bucket-000.parquet"
    manifest_rows: list[_ManifestRow] = []
    with pq.ParquetWriter(
        staging / relative_path,
        _POINT_PAYLOAD_SCHEMA,
        compression="snappy",
        use_dictionary=["value_id"],
    ) as writer:
        row_group = 0
        for tile_y in range(4):
            for tile_x in range(4):
                first_point_id = (tile_y * 4 + tile_x) * 5
                point_ids = list(range(first_point_id, first_point_id + 5))
                points = _points(
                    x_rel=[10.0, 100.0, 200.0, 300.0, 400.0],
                    y_rel=[20.0, 120.0, 220.0, 320.0, 420.0],
                    value_ids=[(point_id + value_variant) % 4 for point_id in point_ids],
                    point_ids=point_ids,
                )
                writer.write_table(points, row_group_size=points.num_rows)
                manifest_rows.append(
                    _ManifestRow(
                        level=1,
                        level_file=relative_path,
                        tile_x=tile_x,
                        tile_y=tile_y,
                        n_points=points.num_rows,
                        row_group=row_group,
                        tile_shard=0,
                    )
                )
                row_group += 1
    return _LevelWriteResult(
        manifest_rows=tuple(manifest_rows),
        intermediate_tile_value_count_files=(),
    )


def _point_ids_by_tile(result: _LevelWriteResult, staging: Path) -> dict[tuple[int, int], list[int]]:
    point_ids: dict[tuple[int, int], list[int]] = {}
    for row in result.manifest_rows:
        decoded = pq.ParquetFile(staging / row.level_file).read_row_group(
            row.row_group,
            columns=["point_id"],
        )
        point_ids.setdefault((row.tile_y, row.tile_x), []).extend(decoded["point_id"].to_pylist())
    return point_ids


def _intermediate_rows(result: _LevelWriteResult, staging: Path) -> list[dict[str, int]]:
    tables = [pq.read_table(staging / file.relative_path) for file in result.intermediate_tile_value_count_files]
    return sorted(
        pa.concat_tables(tables).to_pylist(),
        key=lambda row: (row["level"], row["tile_y"], row["tile_x"], row["value_id"]),
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


def test_spatial_writer_builds_deterministic_nested_value_neutral_levels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _persistent_plan()
    monkeypatch.setattr(spatial_writer_module, "DEFAULT_MAX_ROWS_PER_ROW_GROUP", 4)
    first_staging = tmp_path / "first"
    second_staging = tmp_path / "second"
    changed_values_staging = tmp_path / "changed-values"

    first = _write_spatial_levels(
        _write_bridge_fixture(first_staging),
        plan,
        staging_directory=first_staging,
    )
    second = _write_spatial_levels(
        _write_bridge_fixture(second_staging),
        plan,
        staging_directory=second_staging,
    )
    changed_values = _write_spatial_levels(
        _write_bridge_fixture(changed_values_staging, value_variant=2),
        plan,
        staging_directory=changed_values_staging,
    )

    assert first == second
    assert len(first) == 2
    assert [sum(row.n_points for row in result.manifest_rows) for result in first] == [28, 9]
    assert [row.tile_shard for row in first[0].manifest_rows] == [0, 1] * 4
    assert [row.tile_shard for row in first[1].manifest_rows] == [0, 1, 2]

    bridge_point_ids = set(range(80))
    l1_by_tile = _point_ids_by_tile(first[0], first_staging)
    l2_by_tile = _point_ids_by_tile(first[1], first_staging)
    l1_point_ids = {point_id for tile_ids in l1_by_tile.values() for point_id in tile_ids}
    l2_point_ids = {point_id for tile_ids in l2_by_tile.values() for point_id in tile_ids}
    assert all(tile_ids == sorted(tile_ids) for tile_ids in l1_by_tile.values())
    assert all(tile_ids == sorted(tile_ids) for tile_ids in l2_by_tile.values())
    assert l2_point_ids <= l1_point_ids <= bridge_point_ids

    assert [_point_ids_by_tile(result, first_staging) for result in first] == [
        _point_ids_by_tile(result, second_staging) for result in second
    ]
    assert [_point_ids_by_tile(result, first_staging) for result in first] == [
        _point_ids_by_tile(result, changed_values_staging) for result in changed_values
    ]
    assert [_intermediate_rows(result, first_staging) for result in first] == [
        _intermediate_rows(result, second_staging) for result in second
    ]
    assert [sum(row["n_points"] for row in _intermediate_rows(result, first_staging)) for result in first] == [
        28,
        9,
    ]


def test_spatial_writer_returns_no_results_for_bridge_terminal_plan(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()

    assert (
        _write_spatial_levels(
            _LevelWriteResult(manifest_rows=(), intermediate_tile_value_count_files=()),
            _persistent_plan(spatial=False),
            staging_directory=staging,
        )
        == ()
    )


def test_spatial_writer_rejects_invalid_finer_shards_and_row_counts(tmp_path: Path) -> None:
    plan = _persistent_plan()
    finer_level = plan.levels[1]
    coarser_level = plan.levels[2]

    shards_staging = tmp_path / "invalid-shards"
    shards_result = _write_bridge_fixture(shards_staging)
    invalid_shard_row = replace(shards_result.manifest_rows[0], tile_shard=1)
    with pytest.raises(ValueError, match="non-contiguous shards"):
        _write_spatial_level(
            replace(shards_result, manifest_rows=(invalid_shard_row, *shards_result.manifest_rows[1:])),
            finer_level=finer_level,
            coarser_level=coarser_level,
            staging_directory=shards_staging,
        )

    rows_staging = tmp_path / "invalid-rows"
    rows_result = _write_bridge_fixture(rows_staging)
    invalid_count_row = replace(rows_result.manifest_rows[0], n_points=6)
    with pytest.raises(ValueError, match="does not match its manifest row count"):
        _write_spatial_level(
            replace(rows_result, manifest_rows=(invalid_count_row, *rows_result.manifest_rows[1:])),
            finer_level=finer_level,
            coarser_level=coarser_level,
            staging_directory=rows_staging,
        )
