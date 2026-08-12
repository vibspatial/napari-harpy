from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import napari_harpy.core.multi_scale_cache_points.writer.exact as exact_writer_module
import napari_harpy.core.multi_scale_cache_points.writer.support as writer_support_module
from napari_harpy.core.multi_scale_cache_points import (
    ParquetPointsSource,
    PointColumnSelection,
    ValidatedPointsSource,
    validate_parquet_points_source,
)
from napari_harpy.core.multi_scale_cache_points.build_plan import _plan_points_cache
from napari_harpy.core.multi_scale_cache_points.models import ParquetSourceFile, ParquetSourceRowGroup
from napari_harpy.core.multi_scale_cache_points.writer.exact import (
    _annotate_source_partition,
    _write_exact_level,
)
from napari_harpy.core.multi_scale_cache_points.writer.models import (
    _ExactLevelWriterConfig,
    _LevelWriteResult,
)
from napari_harpy.core.multi_scale_cache_points.writer.support import _bucket_count_for_level


def _dictionary_array(dictionary: list[str], indices: list[int]) -> pa.DictionaryArray:
    return pa.DictionaryArray.from_arrays(
        pa.array(indices, type=pa.int8()),
        pa.array(dictionary, type=pa.string()),
    )


def _source(tmp_path: Path) -> ParquetPointsSource:
    source = ParquetPointsSource(
        spatialdata_path=tmp_path / "example.zarr",
        points_name="transcripts",
        columns=PointColumnSelection(x="x", y="y", value="gene"),
    )
    source.parquet_path.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "x": pa.array([1.0, 11.0, 3.0]),
                "y": pa.array([1.0, 1.0, 2.0]),
                "gene": _dictionary_array([" B ", "A"], [1, 0, 0]),
            }
        ),
        source.parquet_path / "part.0.parquet",
        row_group_size=2,
    )
    pq.write_table(
        pa.table(
            {
                "x": pa.array([2.0, 12.0, 4.0]),
                "y": pa.array([3.0, 2.0, 4.0]),
                "gene": _dictionary_array(["A", " B "], [0, 1, 0]),
            }
        ),
        source.parquet_path / "part.1.parquet",
        row_group_size=2,
    )
    return source


def _build_exact(
    source: ParquetPointsSource,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    staging_name: str,
) -> tuple[_LevelWriteResult, ValidatedPointsSource, Path, Path]:
    validated = validate_parquet_points_source(source, max_batch_rows=2)
    plan = _plan_points_cache(validated, leaf_tile_size=10, overview_point_budget=10)
    monkeypatch.setattr(writer_support_module, "TARGET_ROWS_PER_OUTPUT_BUCKET", 2)
    exact = plan.levels[0]
    config = _ExactLevelWriterConfig(
        bucket_count=_bucket_count_for_level(exact),
        max_rows_per_row_group=2,
        dask_worker_count=1,
    )
    staging = tmp_path / staging_name
    temporary = tmp_path / f"{staging_name}-temporary"
    staging.mkdir()
    temporary.mkdir()
    result = _write_exact_level(
        validated,
        plan,
        staging_directory=staging,
        temporary_directory_root=temporary,
        config=config,
    )
    return result, validated, staging, temporary


def _read_written_points(result: _LevelWriteResult, staging: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for manifest_row in result.manifest_rows:
        parquet_file = pq.ParquetFile(staging / manifest_row.level_file)
        table = parquet_file.read_row_group(manifest_row.row_group)
        for point in table.to_pylist():
            rows.append(
                {
                    "point_id": point["point_id"],
                    "value_id": point["value_id"],
                    "x": manifest_row.tile_x * 10 + point["x_rel"],
                    "y": manifest_row.tile_y * 10 + point["y_rel"],
                    "tile_x": manifest_row.tile_x,
                    "tile_y": manifest_row.tile_y,
                }
            )
    return sorted(rows, key=lambda row: row["point_id"])


def test_annotation_reconstructs_fractional_coordinates_within_float32_tolerance() -> None:
    x = np.array([-511.99997, -0.00003, 0.0, 511.99997], dtype=np.float64)
    y = np.array([-1023.75, -512.00003, -512.0, -0.00003], dtype=np.float64)
    source_file = ParquetSourceFile(
        relative_path="part.0.parquet",
        size_bytes=0,
        modified_time_ns=None,
        row_count=len(x),
        row_offset=10,
        row_groups=(ParquetSourceRowGroup(row_count=len(x), compressed_size_bytes=0),),
    )

    annotated = _annotate_source_partition(
        pd.DataFrame({"x": x, "y": y, "value": ["A"] * len(x)}),
        source_file=source_file,
        x_column="x",
        y_column="y",
        value_column="value",
        x_origin=-512.0,
        y_origin=-1024.0,
        tile_size=512,
        grid_width=2,
        grid_height=2,
        bucket_count=3,
        value_labels_by_id=("A",),
    )

    reconstructed_x = -512.0 + annotated["tile_x"].to_numpy(dtype=np.float64) * 512 + annotated["x_rel"]
    reconstructed_y = -1024.0 + annotated["tile_y"].to_numpy(dtype=np.float64) * 512 + annotated["y_rel"]
    tolerance = float(np.spacing(np.float32(512)))
    np.testing.assert_allclose(reconstructed_x, x, rtol=0.0, atol=tolerance)
    np.testing.assert_allclose(reconstructed_y, y, rtol=0.0, atol=tolerance)


def test_exact_writer_co_locates_tiles_and_writes_deterministic_shards(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source(tmp_path)
    first, validated, first_staging, first_temporary = _build_exact(
        source,
        tmp_path,
        monkeypatch,
        staging_name="first",
    )
    second, _, second_staging, second_temporary = _build_exact(
        source,
        tmp_path,
        monkeypatch,
        staging_name="second",
    )

    assert first == second
    assert list(first_temporary.iterdir()) == []
    assert list(second_temporary.iterdir()) == []
    assert sum(row.n_points for row in first.manifest_rows) == validated.row_count == 6
    assert tuple((row.tile_x, row.tile_y, row.tile_shard, row.n_points) for row in first.manifest_rows) == (
        (0, 0, 0, 2),
        (0, 0, 1, 2),
        (1, 0, 0, 2),
    )
    tile_files: dict[tuple[int, int], set[str]] = {}
    for row in first.manifest_rows:
        tile_files.setdefault((row.tile_x, row.tile_y), set()).add(row.level_file)
    assert all(len(paths) == 1 for paths in tile_files.values())

    expected_values = {row["value"]: row["value_id"] for row in validated.value_table.to_pylist()}
    expected_points = [
        {"point_id": 0, "value_id": expected_values["A"], "x": 1.0, "y": 1.0, "tile_x": 0, "tile_y": 0},
        {"point_id": 1, "value_id": expected_values["B"], "x": 11.0, "y": 1.0, "tile_x": 1, "tile_y": 0},
        {"point_id": 2, "value_id": expected_values["B"], "x": 3.0, "y": 2.0, "tile_x": 0, "tile_y": 0},
        {"point_id": 3, "value_id": expected_values["A"], "x": 2.0, "y": 3.0, "tile_x": 0, "tile_y": 0},
        {"point_id": 4, "value_id": expected_values["B"], "x": 12.0, "y": 2.0, "tile_x": 1, "tile_y": 0},
        {"point_id": 5, "value_id": expected_values["A"], "x": 4.0, "y": 4.0, "tile_x": 0, "tile_y": 0},
    ]
    assert _read_written_points(first, first_staging) == expected_points
    assert _read_written_points(second, second_staging) == expected_points

    count_tables = [
        pq.read_table(first_staging / file.relative_path) for file in first.intermediate_tile_value_count_files
    ]
    count_rows = sorted(
        pa.concat_tables(count_tables).to_pylist(),
        key=lambda row: (row["tile_y"], row["tile_x"], row["value_id"]),
    )
    assert count_rows == [
        {"level": 0, "value_id": expected_values["A"], "tile_x": 0, "tile_y": 0, "n_points": 3},
        {"level": 0, "value_id": expected_values["B"], "tile_x": 0, "tile_y": 0, "n_points": 1},
        {"level": 0, "value_id": expected_values["B"], "tile_x": 1, "tile_y": 0, "n_points": 2},
    ]


def test_exact_writer_propagates_finalizer_failure_and_cleans_shuffle_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source(tmp_path)
    validated = validate_parquet_points_source(source, max_batch_rows=2)
    plan = _plan_points_cache(validated, leaf_tile_size=10, overview_point_budget=10)
    monkeypatch.setattr(writer_support_module, "TARGET_ROWS_PER_OUTPUT_BUCKET", 2)
    config = _ExactLevelWriterConfig(
        bucket_count=_bucket_count_for_level(plan.levels[0]),
        max_rows_per_row_group=2,
        dask_worker_count=1,
    )
    staging = tmp_path / "staging"
    temporary = tmp_path / "temporary"
    staging.mkdir()
    temporary.mkdir()

    def fail_finalizer(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("injected finalizer failure")

    monkeypatch.setattr(exact_writer_module, "_finalize_bucket", fail_finalizer)

    with pytest.raises(RuntimeError, match="injected finalizer failure"):
        _write_exact_level(
            validated,
            plan,
            staging_directory=staging,
            temporary_directory_root=temporary,
            config=config,
        )

    assert list(temporary.iterdir()) == []
