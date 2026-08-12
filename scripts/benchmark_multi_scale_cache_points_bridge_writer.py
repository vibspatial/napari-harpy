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

import psutil
import pyarrow as pa
import pyarrow.parquet as pq

from napari_harpy.core.multi_scale_cache_points import (
    ParquetPointsSource,
    PointColumnSelection,
    validate_parquet_points_source,
)
from napari_harpy.core.multi_scale_cache_points.build_plan import _plan_points_cache
from napari_harpy.core.multi_scale_cache_points.writer.bridge import _write_bridge_level
from napari_harpy.core.multi_scale_cache_points.writer.exact import (
    DEFAULT_DASK_WORKER_COUNT,
    _write_exact_level,
)
from napari_harpy.core.multi_scale_cache_points.writer.models import (
    _ExactLevelWriterConfig,
    _LevelWriteResult,
    _ManifestRow,
)
from napari_harpy.core.multi_scale_cache_points.writer.support import (
    BUCKET_HASH_METHOD,
    DEFAULT_MAX_ROWS_PER_ROW_GROUP,
    TARGET_ROWS_PER_OUTPUT_BUCKET,
    _bucket_count_for_level,
)

_SAMPLE_INTERVAL_SECONDS = 0.1


class _ResourceSampler:
    """Sample process RSS and workspace growth during Bridge construction."""

    def __init__(self, workspace: Path) -> None:
        self._workspace = workspace
        self._process = psutil.Process()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample_until_stopped, daemon=True)
        self.baseline_rss_bytes = self._process.memory_info().rss
        self.peak_rss_bytes = self.baseline_rss_bytes
        self.baseline_workspace_bytes = _directory_size(workspace)
        self.peak_workspace_bytes = self.baseline_workspace_bytes

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


def _level_output_summary(result: _LevelWriteResult, staging: Path) -> dict[str, object]:
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
        "point_rows": sum(rows),
        "point_file_bytes": sum(sizes),
        "average_bucket_rows": sum(rows) / len(rows),
        "maximum_bucket_rows": max(rows),
        "minimum_bucket_rows": min(rows),
        "average_bucket_bytes": sum(sizes) / len(sizes),
        "maximum_bucket_bytes": max(sizes),
        "intermediate_count_file_count": len(result.intermediate_tile_value_count_files),
        "intermediate_count_rows": sum(file.row_count for file in result.intermediate_tile_value_count_files),
        "intermediate_count_bytes": intermediate_count_bytes,
    }


def _largest_exact_tile(result: _LevelWriteResult, staging: Path) -> dict[str, object]:
    tiles: dict[tuple[int, int], list[_ManifestRow]] = defaultdict(list)
    for row in result.manifest_rows:
        tiles[(row.tile_y, row.tile_x)].append(row)
    tile, rows = max(tiles.items(), key=lambda item: sum(row.n_points for row in item[1]))

    tables: list[pa.Table] = []
    files: dict[str, list[int]] = defaultdict(list)
    for row in sorted(rows, key=lambda item: item.tile_shard):
        files[row.level_file].append(row.row_group)
    for relative_path, row_groups in files.items():
        tables.append(pq.ParquetFile(staging / relative_path).read_row_groups(row_groups))
    table = pa.concat_tables(tables)
    return {
        "tile_y": tile[0],
        "tile_x": tile[1],
        "shard_count": len(rows),
        "n_points": table.num_rows,
        "decoded_table_bytes": table.nbytes,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one Bridge point-cache writer acceptance benchmark.")
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
    """Prepare Exact once, measure Bridge once, and remove staged artifacts."""
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
    if len(plan.levels) < 2:
        raise RuntimeError("The benchmark source does not require a Bridge level.")
    exact, bridge = plan.levels[:2]
    exact_config = _ExactLevelWriterConfig(
        bucket_count=_bucket_count_for_level(exact),
        max_rows_per_row_group=DEFAULT_MAX_ROWS_PER_ROW_GROUP,
        dask_worker_count=args.dask_worker_count,
    )

    args.work_directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="napari-harpy-bridge-benchmark-", dir=args.work_directory
    ) as workspace_text:
        workspace = Path(workspace_text)
        staging = workspace / "staging"
        shuffle = workspace / "shuffle"
        staging.mkdir()
        shuffle.mkdir()

        exact_start = perf_counter()
        exact_result = _write_exact_level(
            validated,
            plan,
            staging_directory=staging,
            temporary_directory_root=shuffle,
            config=exact_config,
        )
        exact_seconds = perf_counter() - exact_start

        with _ResourceSampler(workspace) as resources:
            bridge_start = perf_counter()
            bridge_result = _write_bridge_level(
                exact_result,
                plan,
                staging_directory=staging,
            )
            bridge_seconds = perf_counter() - bridge_start

        report = {
            "schema_version": "harpy-bridge-writer-acceptance-v1",
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
                "source_signature": validated.source_signature,
            },
            "configuration": {
                "leaf_tile_size": exact.tile_size,
                "bridge_level": bridge.level,
                "bridge_max_points_per_tile": bridge.max_points_per_tile,
                "bucket_hash_method": BUCKET_HASH_METHOD,
                "target_rows_per_output_bucket": TARGET_ROWS_PER_OUTPUT_BUCKET,
                "exact_bucket_count": exact_config.bucket_count,
                "bridge_bucket_count": _bucket_count_for_level(bridge),
                "dask_worker_count_for_exact_preparation": exact_config.dask_worker_count,
            },
            "timing_seconds": {
                "validation": validation_seconds,
                "exact_preparation": exact_seconds,
                "bridge_build": bridge_seconds,
            },
            "bridge_resources": {
                "rss_sample_interval_seconds": _SAMPLE_INTERVAL_SECONDS,
                "baseline_rss_bytes": resources.baseline_rss_bytes,
                "peak_rss_bytes": resources.peak_rss_bytes,
                "incremental_peak_rss_bytes": resources.peak_rss_bytes - resources.baseline_rss_bytes,
                "baseline_workspace_bytes": resources.baseline_workspace_bytes,
                "peak_workspace_bytes": resources.peak_workspace_bytes,
                "incremental_peak_workspace_bytes": (
                    resources.peak_workspace_bytes - resources.baseline_workspace_bytes
                ),
            },
            "largest_exact_tile": _largest_exact_tile(exact_result, staging),
            "bridge_output": _level_output_summary(bridge_result, staging),
        }
        args.json_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
