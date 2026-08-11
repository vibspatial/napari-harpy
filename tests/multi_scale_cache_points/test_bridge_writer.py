from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from napari_harpy.core.multi_scale_cache_points.build_plan import _plan_points_cache, _PointsCacheBuildPlan
from napari_harpy.core.multi_scale_cache_points.models import PointsBounds, ValidatedPointsSource
from napari_harpy.core.multi_scale_cache_points.writer.bridge import _write_bridge_level
from napari_harpy.core.multi_scale_cache_points.writer.models import _LevelWriteResult, _ManifestRow
from napari_harpy.core.multi_scale_cache_points.writer.support import _POINT_PAYLOAD_SCHEMA


def _plan(*, row_count: int, overview_point_budget: int) -> _PointsCacheBuildPlan:
    validated = cast(
        ValidatedPointsSource,
        SimpleNamespace(
            row_count=row_count,
            bounds=PointsBounds(x_min=0.0, x_max=1_023.0, y_min=0.0, y_max=511.0),
        ),
    )
    return _plan_points_cache(
        validated,
        leaf_tile_size=512,
        overview_point_budget=overview_point_budget,
    )


def _point_table(*, point_ids: np.ndarray, value_variant: int) -> pa.Table:
    offsets = point_ids - point_ids.min()
    return pa.Table.from_arrays(
        [
            pa.array((offsets % 512).astype(np.float32), type=pa.float32()),
            pa.array(((offsets // 512) % 512).astype(np.float32), type=pa.float32()),
            pa.array(((point_ids * 7 + value_variant) % 5).astype(np.uint32), type=pa.uint32()),
            pa.array(point_ids, type=pa.uint64()),
        ],
        schema=_POINT_PAYLOAD_SCHEMA,
    )


def _write_exact_fixture(staging: Path, *, value_variant: int = 0) -> _LevelWriteResult:
    staging.mkdir()
    level_directory = staging / "levels/level_0"
    level_directory.mkdir(parents=True)
    relative_path = "levels/level_0/bucket-000.parquet"
    manifest_rows: list[_ManifestRow] = []
    physical_row_group = 0

    tile_point_ids = {
        (0, 0): np.arange(0, 3, dtype=np.uint64),
        (0, 1): np.arange(3, 4_103, dtype=np.uint64),
    }
    shard_sizes = {(0, 0): 2, (0, 1): 1_500}
    with pq.ParquetWriter(
        staging / relative_path,
        _POINT_PAYLOAD_SCHEMA,
        compression="snappy",
        use_dictionary=["value_id"],
    ) as writer:
        for (tile_y, tile_x), point_ids in tile_point_ids.items():
            table = _point_table(
                point_ids=point_ids,
                value_variant=value_variant,
            )
            shard_size = shard_sizes[(tile_y, tile_x)]
            for tile_shard, start in enumerate(range(0, table.num_rows, shard_size)):
                shard = table.slice(start, shard_size)
                writer.write_table(shard, row_group_size=shard.num_rows)
                manifest_rows.append(
                    _ManifestRow(
                        level=0,
                        level_file=relative_path,
                        tile_x=tile_x,
                        tile_y=tile_y,
                        n_points=shard.num_rows,
                        row_group=physical_row_group,
                        tile_shard=tile_shard,
                    )
                )
                physical_row_group += 1
    return _LevelWriteResult(
        manifest_rows=tuple(manifest_rows),
        intermediate_tile_value_count_files=(),
    )


def _point_ids_by_tile(result: _LevelWriteResult, staging: Path) -> dict[tuple[int, int], list[int]]:
    point_ids: dict[tuple[int, int], list[int]] = {}
    for row in result.manifest_rows:
        table = pq.ParquetFile(staging / row.level_file).read_row_group(
            row.row_group,
            columns=["point_id"],
        )
        point_ids[(row.tile_y, row.tile_x)] = table["point_id"].to_pylist()
    return point_ids


def _intermediate_rows(result: _LevelWriteResult, staging: Path) -> list[dict[str, int]]:
    tables = [pq.read_table(staging / file.relative_path) for file in result.intermediate_tile_value_count_files]
    return pa.concat_tables(tables).to_pylist()


def test_bridge_writer_reconstructs_shards_and_writes_deterministic_value_neutral_samples(
    tmp_path: Path,
) -> None:
    plan = _plan(row_count=4_103, overview_point_budget=1_000)
    first_staging = tmp_path / "first"
    second_staging = tmp_path / "second"
    changed_values_staging = tmp_path / "changed-values"

    first = _write_bridge_level(
        _write_exact_fixture(first_staging),
        plan,
        staging_directory=first_staging,
    )
    second = _write_bridge_level(
        _write_exact_fixture(second_staging),
        plan,
        staging_directory=second_staging,
    )
    changed_values = _write_bridge_level(
        _write_exact_fixture(changed_values_staging, value_variant=3),
        plan,
        staging_directory=changed_values_staging,
    )

    assert first == second
    assert tuple(
        (row.level_file, row.row_group, row.tile_y, row.tile_x, row.tile_shard, row.n_points)
        for row in first.manifest_rows
    ) == (
        ("levels/level_1/bucket-000.parquet", 0, 0, 0, 0, 3),
        ("levels/level_1/bucket-000.parquet", 1, 0, 1, 0, 4_096),
    )
    first_point_ids = _point_ids_by_tile(first, first_staging)
    assert first_point_ids[(0, 0)] == [0, 1, 2]
    assert len(first_point_ids[(0, 1)]) == 4_096
    assert first_point_ids[(0, 1)] == sorted(first_point_ids[(0, 1)])
    assert first_point_ids == _point_ids_by_tile(second, second_staging)
    assert first_point_ids == _point_ids_by_tile(changed_values, changed_values_staging)
    assert _intermediate_rows(first, first_staging) == _intermediate_rows(second, second_staging)
    assert sum(row["n_points"] for row in _intermediate_rows(first, first_staging)) == 4_099

    point_file = pq.ParquetFile(first_staging / "levels/level_1/bucket-000.parquet")
    assert point_file.schema_arrow.equals(_POINT_PAYLOAD_SCHEMA, check_metadata=False)
    assert point_file.num_row_groups == 2
    assert (first_staging / "levels/level_0/bucket-000.parquet").is_file()


def test_bridge_writer_rejects_an_exact_only_plan(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    plan = _plan(row_count=10, overview_point_budget=10)

    with pytest.raises(ValueError, match="no Bridge level"):
        _write_bridge_level(
            _LevelWriteResult(manifest_rows=(), intermediate_tile_value_count_files=()),
            plan,
            staging_directory=staging,
        )
