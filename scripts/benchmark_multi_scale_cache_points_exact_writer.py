from __future__ import annotations

import argparse
import json
import os
import platform
import tempfile
import threading
from collections import defaultdict
from pathlib import Path
from time import perf_counter

import numpy as np
import psutil
import pyarrow.parquet as pq

from napari_harpy.core.multi_scale_cache_points import (
    ParquetPointsSource,
    PointColumnSelection,
    validate_parquet_points_source,
)
from napari_harpy.core.multi_scale_cache_points.build_plan import _plan_points_cache
from napari_harpy.core.multi_scale_cache_points.writer.exact import (
    DEFAULT_DASK_WORKER_COUNT,
    _write_exact_level,
)
from napari_harpy.core.multi_scale_cache_points.writer.models import _ExactLevelWriterConfig, _LevelWriteResult
from napari_harpy.core.multi_scale_cache_points.writer.support import (
    BUCKET_HASH_METHOD,
    DEFAULT_MAX_ROWS_PER_ROW_GROUP,
    TARGET_ROWS_PER_OUTPUT_BUCKET,
    _bucket_count_for_level,
)

_SAMPLE_INTERVAL_SECONDS = 0.5


class _ResourceSampler:
    """Sample process RSS and benchmark-workspace bytes during one build."""

    def __init__(self, workspace: Path) -> None:
        self._workspace = workspace
        self._process = psutil.Process()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample_until_stopped, daemon=True)
        self.baseline_rss_bytes = self._process.memory_info().rss
        self.peak_rss_bytes = self.baseline_rss_bytes
        self.peak_workspace_bytes = 0

    def __enter__(self) -> _ResourceSampler:
        self._thread.start()
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self._sample()
        self._stop.set()
        self._thread.join()

    def _sample_until_stopped(self) -> None:
        while not self._stop.wait(_SAMPLE_INTERVAL_SECONDS):
            self._sample()

    def _sample(self) -> None:
        self.peak_rss_bytes = max(self.peak_rss_bytes, self._process.memory_info().rss)
        self.peak_workspace_bytes = max(self.peak_workspace_bytes, _directory_size(self._workspace))


def _directory_size(root: Path) -> int:
    total = 0
    for directory, _, filenames in os.walk(root):
        for filename in filenames:
            try:
                total += (Path(directory) / filename).stat().st_size
            except FileNotFoundError:
                continue
    return total


def _verify_point_ids(result: _LevelWriteResult, staging: Path, *, expected_count: int) -> dict[str, object]:
    start = perf_counter()
    seen = np.zeros((expected_count + 7) // 8, dtype=np.uint8)
    observed_count = 0
    files: dict[str, list[int]] = defaultdict(list)
    for row in result.manifest_rows:
        files[row.level_file].append(row.row_group)

    for relative_path, row_groups in files.items():
        parquet_file = pq.ParquetFile(staging / relative_path)
        for row_group in row_groups:
            point_ids = parquet_file.read_row_group(row_group, columns=["point_id"])["point_id"].to_numpy()
            if len(point_ids) != len(np.unique(point_ids)):
                raise RuntimeError("An Exact row group contains duplicate point IDs.")
            if bool((point_ids >= expected_count).any()):
                raise RuntimeError("An Exact point ID lies outside the validated source range.")
            byte_indices = point_ids >> np.uint64(3)
            masks = np.left_shift(np.uint8(1), (point_ids & np.uint64(7)).astype(np.uint8))
            if bool((seen[byte_indices] & masks).any()):
                raise RuntimeError("An Exact point ID occurs in more than one row group.")
            np.bitwise_or.at(seen, byte_indices, masks)
            observed_count += len(point_ids)

    if observed_count != expected_count:
        raise RuntimeError("Exact point IDs do not cover the validated source row count.")
    return {
        "seconds": perf_counter() - start,
        "observed_count": observed_count,
        "unique_and_complete": True,
    }


def _measure_representative_tile_read(result: _LevelWriteResult, staging: Path) -> dict[str, object]:
    tile_rows: dict[tuple[int, int], list[object]] = defaultdict(list)
    for row in result.manifest_rows:
        tile_rows[(row.tile_x, row.tile_y)].append(row)
    tile, rows = max(tile_rows.items(), key=lambda item: sum(row.n_points for row in item[1]))

    start = perf_counter()
    observed_rows = 0
    files: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        files[row.level_file].append(row.row_group)
    for relative_path, row_groups in files.items():
        parquet_file = pq.ParquetFile(staging / relative_path)
        observed_rows += parquet_file.read_row_groups(row_groups).num_rows
    seconds = perf_counter() - start

    expected_rows = sum(row.n_points for row in rows)
    if observed_rows != expected_rows:
        raise RuntimeError("Representative tile read does not match its manifest rows.")
    return {
        "tile_x": tile[0],
        "tile_y": tile[1],
        "shard_count": len(rows),
        "n_points": observed_rows,
        "seconds": seconds,
    }


def _output_summary(result: _LevelWriteResult, staging: Path) -> dict[str, object]:
    bucket_rows: dict[str, int] = defaultdict(int)
    for row in result.manifest_rows:
        bucket_rows[row.level_file] += row.n_points
    bucket_bytes = {path: (staging / path).stat().st_size for path in bucket_rows}
    intermediate_count_bytes = sum(
        (staging / file.relative_path).stat().st_size for file in result.intermediate_tile_value_count_files
    )
    rows = list(bucket_rows.values())
    sizes = list(bucket_bytes.values())
    return {
        "bucket_file_count": len(bucket_rows),
        "row_group_count": len(result.manifest_rows),
        "manifest_row_count": len(result.manifest_rows),
        "point_rows": sum(rows),
        "point_file_bytes": sum(sizes),
        "average_bucket_rows": sum(rows) / len(rows),
        "maximum_bucket_rows": max(rows),
        "average_bucket_bytes": sum(sizes) / len(sizes),
        "maximum_bucket_bytes": max(sizes),
        "intermediate_count_file_count": len(result.intermediate_tile_value_count_files),
        "intermediate_count_rows": sum(file.row_count for file in result.intermediate_tile_value_count_files),
        "intermediate_count_bytes": intermediate_count_bytes,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one Exact point-cache writer acceptance benchmark.")
    parser.add_argument("spatialdata_path", type=Path)
    parser.add_argument("--points-name", required=True)
    parser.add_argument("--x", default="x")
    parser.add_argument("--y", default="y")
    parser.add_argument("--value", default="gene")
    parser.add_argument("--dask-worker-count", type=int, default=DEFAULT_DASK_WORKER_COUNT)
    parser.add_argument("--work-directory", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Run one benchmark and remove its staged cache after reporting."""
    args = _parse_args()
    source = ParquetPointsSource(
        spatialdata_path=args.spatialdata_path,
        points_name=args.points_name,
        columns=PointColumnSelection(x=args.x, y=args.y, value=args.value),
    )

    validation_start = perf_counter()
    validated = validate_parquet_points_source(source)
    validation_seconds = perf_counter() - validation_start
    plan = _plan_points_cache(validated, leaf_tile_size=512, overview_point_budget=100_000)
    exact = plan.levels[0]
    config = _ExactLevelWriterConfig(
        bucket_count=_bucket_count_for_level(exact),
        max_rows_per_row_group=DEFAULT_MAX_ROWS_PER_ROW_GROUP,
        dask_worker_count=args.dask_worker_count,
    )

    args.work_directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="napari-harpy-exact-benchmark-", dir=args.work_directory) as workspace_text:
        workspace = Path(workspace_text)
        staging = workspace / "staging"
        shuffle = workspace / "shuffle"
        staging.mkdir()
        shuffle.mkdir()

        with _ResourceSampler(workspace) as resources:
            build_start = perf_counter()
            result = _write_exact_level(
                validated,
                plan,
                staging_directory=staging,
                temporary_directory_root=shuffle,
                config=config,
            )
            build_seconds = perf_counter() - build_start

        output = _output_summary(result, staging)
        point_ids = _verify_point_ids(result, staging, expected_count=validated.row_count)
        representative_tile = _measure_representative_tile_read(result, staging)
        report = {
            "schema_version": "harpy-exact-writer-acceptance-v1",
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "logical_cpu_count": os.cpu_count(),
                "physical_memory_bytes": psutil.virtual_memory().total,
            },
            "source": {
                "spatialdata_path": str(args.spatialdata_path),
                "points_name": args.points_name,
                "row_count": validated.row_count,
                "file_count": len(validated.files),
                "row_group_count": sum(source_file.row_group_count for source_file in validated.files),
                "value_count": validated.value_table.num_rows,
                "source_signature": validated.source_signature,
            },
            "configuration": {
                "leaf_tile_size": exact.tile_size,
                "bucket_hash_method": BUCKET_HASH_METHOD,
                "target_rows_per_output_bucket": TARGET_ROWS_PER_OUTPUT_BUCKET,
                "bucket_count": config.bucket_count,
                "max_rows_per_row_group": config.max_rows_per_row_group,
                "dask_worker_count": config.dask_worker_count,
                "float32_coordinate_tolerance": float(np.spacing(np.float32(exact.tile_size))),
            },
            "timing_seconds": {
                "validation": validation_seconds,
                "exact_build": build_seconds,
            },
            "resources": {
                "rss_sample_interval_seconds": _SAMPLE_INTERVAL_SECONDS,
                "baseline_rss_bytes": resources.baseline_rss_bytes,
                "peak_rss_bytes": resources.peak_rss_bytes,
                "incremental_peak_rss_bytes": resources.peak_rss_bytes - resources.baseline_rss_bytes,
                "peak_workspace_bytes": resources.peak_workspace_bytes,
            },
            "output": output,
            "point_id_verification": point_ids,
            "representative_tile_read": representative_tile,
        }
        args.json_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
