from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import tempfile
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from time import perf_counter, perf_counter_ns, process_time_ns
from types import ModuleType
from typing import Any

import psutil
from benchmark_multi_scale_cache_points_zarr_exact import _ResourceSampler

import napari_harpy.core.multi_scale_cache_points_zarr.sampling as sampling_module
import napari_harpy.core.multi_scale_cache_points_zarr.storage.bucket_writer as bucket_writer_module
import napari_harpy.core.multi_scale_cache_points_zarr.writer.bridge as bridge_module
from napari_harpy.core.multi_scale_cache_points_zarr.build_plan import _plan_points_cache
from napari_harpy.core.multi_scale_cache_points_zarr.payload import _PointPayload
from napari_harpy.core.multi_scale_cache_points_zarr.source import (
    ParquetPointsSource,
    PointColumnSelection,
    validate_parquet_points_source,
)
from napari_harpy.core.multi_scale_cache_points_zarr.storage.bucket_reader import _BucketReader
from napari_harpy.core.multi_scale_cache_points_zarr.storage.bucket_writer import _BucketWriter
from napari_harpy.core.multi_scale_cache_points_zarr.storage.models import _ZarrWriteSettings
from napari_harpy.core.multi_scale_cache_points_zarr.storage.reader_cache import _BucketReaderCache
from napari_harpy.core.multi_scale_cache_points_zarr.writer.bridge import _BridgeWriterConfig
from napari_harpy.core.multi_scale_cache_points_zarr.writer.exact import (
    _ExactWriterConfig,
    _write_exact_level,
)

_EXPECTED_XENIUM_POINT_COUNT = 136_578_750


@dataclass
class _Timing:
    calls: int = 0
    wall_ns: int = 0
    cpu_ns: int = 0

    def add(self, wall_start: int, cpu_start: int) -> None:
        self.calls += 1
        self.wall_ns += perf_counter_ns() - wall_start
        self.cpu_ns += process_time_ns() - cpu_start

    def serialized(self, *, bridge_seconds: float) -> dict[str, int | float]:
        wall_seconds = self.wall_ns / 1_000_000_000
        return {
            "calls": self.calls,
            "wall_seconds": wall_seconds,
            "cpu_seconds": self.cpu_ns / 1_000_000_000,
            "percent_of_bridge_wall": 100 * wall_seconds / bridge_seconds,
        }


class _BridgeStageProfiler:
    """Install temporary aggregate timers around the existing Bridge code."""

    def __init__(self) -> None:
        self.timings: dict[str, _Timing] = {}
        self.reader_cache_hits = 0
        self.reader_cache_misses = 0
        self.reader_cache_evictions = 0
        self.peak_open_readers = 0
        self._ordering_depth = 0
        self._originals: list[tuple[object, str, object]] = []

    def __enter__(self) -> _BridgeStageProfiler:
        self._patch_timed(_BucketReader, "_tile_interval", "reader_tile_interval")
        self._patch_timed(_BucketReader, "read_construction_payload", "reader_construction_payload")
        self._patch_reader_cache_get()

        self._patch_timed(sampling_module, "_microgrid_cell_ids", "sample_microgrid")
        self._patch_timed(sampling_module, "_allocate_cell_targets", "sample_allocate")
        self._patch_timed(sampling_module, "_cell_tie_break_priorities", "sample_cell_priority")
        self._patch_timed(sampling_module, "_point_priorities", "sample_point_priority")
        self._patch_bridge_sampler()

        self._patch_payload_take()
        self._patch_payload_ordering()
        self._patch_timed(bucket_writer_module, "_ranges_for_payload", "writer_ranges")
        self._patch_timed(_BucketWriter, "_create_arrays", "writer_create_arrays")
        self._patch_timed(_BucketWriter, "__enter__", "writer_enter")
        self._patch_timed(_BucketWriter, "_append_points", "writer_append_points")
        self._patch_timed(_BucketWriter, "_append_ranges", "writer_append_ranges")
        self._patch_timed(_BucketWriter, "_flush_point_buffer", "writer_flush_points")
        self._patch_timed(_BucketWriter, "_flush_range_buffer", "writer_flush_ranges")
        self._patch_timed(_BucketWriter, "write_tile", "writer_write_tile")
        self._patch_timed(_BucketWriter, "finalize", "writer_finalize")
        return self

    def __exit__(self, *_exc_info: object) -> None:
        for owner, name, original in reversed(self._originals):
            setattr(owner, name, original)

    def report(self, *, bridge_seconds: float) -> dict[str, object]:
        serialized = {
            name: timing.serialized(bridge_seconds=bridge_seconds) for name, timing in sorted(self.timings.items())
        }
        top_level_names = (
            "writer_enter",
            "reader_cache_get",
            "reader_complete",
            "sample_total",
            "payload_take_selected",
            "writer_write_tile",
            "writer_finalize",
        )
        accounted = sum(self.timings.get(name, _Timing()).wall_ns for name in top_level_names) / 1_000_000_000
        derived = {
            "top_level_accounted_wall_seconds": accounted,
            "top_level_unattributed_wall_seconds": bridge_seconds - accounted,
            "reader_payload_after_interval_seconds": self._difference("reader_complete", "reader_tile_interval"),
            "sample_other_seconds": self._difference(
                "sample_total",
                "sample_microgrid",
                "sample_allocate",
                "sample_point_priority",
            ),
            "payload_order_without_take_seconds": self._difference(
                "payload_order_value_point",
                "payload_take_ordering",
            ),
            "writer_write_other_seconds": self._difference(
                "writer_write_tile",
                "payload_order_value_point",
                "writer_ranges",
                "writer_append_points",
                "writer_append_ranges",
            ),
        }
        return {
            "timings": serialized,
            "derived_nonoverlapping_or_exclusive_seconds": derived,
            "reader_cache": {
                "hits": self.reader_cache_hits,
                "misses": self.reader_cache_misses,
                "evictions": self.reader_cache_evictions,
                "peak_open_readers": self.peak_open_readers,
            },
        }

    def _difference(self, total: str, *children: str) -> float:
        total_ns = self.timings.get(total, _Timing()).wall_ns
        child_ns = sum(self.timings.get(name, _Timing()).wall_ns for name in children)
        return (total_ns - child_ns) / 1_000_000_000

    def _timing(self, name: str) -> _Timing:
        return self.timings.setdefault(name, _Timing())

    def _remember(self, owner: object, name: str) -> object:
        original = getattr(owner, name)
        self._originals.append((owner, name, original))
        return original

    def _patch_timed(self, owner: type[Any] | ModuleType, name: str, timing_name: str) -> None:
        original = self._remember(owner, name)

        @wraps(original)
        def timed(*args: object, **kwargs: object) -> object:
            wall_start = perf_counter_ns()
            cpu_start = process_time_ns()
            try:
                return original(*args, **kwargs)  # type: ignore[operator]
            finally:
                self._timing(timing_name).add(wall_start, cpu_start)

        setattr(owner, name, timed)

    def _patch_reader_cache_get(self) -> None:
        original = self._remember(_BucketReaderCache, "get")

        @wraps(original)
        def timed_get(cache: _BucketReaderCache, *, level: int, bucket_id: int) -> _BucketReader:
            key = (level, bucket_id)
            is_hit = key in cache._readers
            if is_hit:
                self.reader_cache_hits += 1
            else:
                self.reader_cache_misses += 1
                if len(cache._readers) == cache._max_open_readers:
                    self.reader_cache_evictions += 1
            wall_start = perf_counter_ns()
            cpu_start = process_time_ns()
            try:
                return original(cache, level=level, bucket_id=bucket_id)  # type: ignore[operator]
            finally:
                self._timing("reader_cache_get").add(wall_start, cpu_start)
                self.peak_open_readers = max(self.peak_open_readers, cache.open_reader_count)

        _BucketReaderCache.get = timed_get

    def _patch_bridge_sampler(self) -> None:
        original = self._remember(bridge_module, "_select_sampled_tile_indices")

        @wraps(original)
        def timed(*args: object, **kwargs: object) -> object:
            wall_start = perf_counter_ns()
            cpu_start = process_time_ns()
            try:
                return original(*args, **kwargs)  # type: ignore[operator]
            finally:
                self._timing("sample_total").add(wall_start, cpu_start)

        bridge_module._select_sampled_tile_indices = timed

    def _patch_payload_take(self) -> None:
        original = self._remember(_PointPayload, "take")

        @wraps(original)
        def timed_take(payload: _PointPayload, *args: object, **kwargs: object) -> _PointPayload:
            timing_name = "payload_take_ordering" if self._ordering_depth else "payload_take_selected"
            wall_start = perf_counter_ns()
            cpu_start = process_time_ns()
            try:
                return original(payload, *args, **kwargs)  # type: ignore[operator]
            finally:
                self._timing(timing_name).add(wall_start, cpu_start)

        _PointPayload.take = timed_take

    def _patch_payload_ordering(self) -> None:
        original = self._remember(_PointPayload, "ordered_by_value_and_point_id")

        @wraps(original)
        def timed_order(payload: _PointPayload) -> _PointPayload:
            wall_start = perf_counter_ns()
            cpu_start = process_time_ns()
            self._ordering_depth += 1
            try:
                return original(payload)  # type: ignore[operator]
            finally:
                self._ordering_depth -= 1
                self._timing("payload_order_value_point").add(wall_start, cpu_start)

        _PointPayload.ordered_by_value_and_point_id = timed_order


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage-profile one sequential full-Xenium Zarr Bridge build.")
    parser.add_argument("spatialdata_path", type=Path)
    parser.add_argument("--points-name", required=True)
    parser.add_argument("--x", default="x")
    parser.add_argument("--y", default="y")
    parser.add_argument("--value", default="gene")
    parser.add_argument("--work-directory", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument(
        "--max-open-exact-readers",
        type=int,
        default=None,
        help="Maximum entered Exact readers; defaults to all nonempty Exact buckets.",
    )
    return parser.parse_args()


def main() -> None:
    """Prepare Exact once, profile one sequential Bridge build, and report stages."""
    args = _parse_args()
    source = ParquetPointsSource(
        spatialdata_path=args.spatialdata_path,
        points_name=args.points_name,
        columns=PointColumnSelection(x=args.x, y=args.y, value=args.value),
    )
    validation_start = perf_counter()
    validated = validate_parquet_points_source(source)
    validation_seconds = perf_counter() - validation_start
    if validated.row_count != _EXPECTED_XENIUM_POINT_COUNT:
        raise RuntimeError(
            f"Validated source contains {validated.row_count} rows; expected {_EXPECTED_XENIUM_POINT_COUNT}."
        )
    plan = _plan_points_cache(validated, leaf_tile_size=512, overview_point_budget=100_000)
    settings = _ZarrWriteSettings(4_096, 131_072, 8_192, 131_072, "zstd-v1")

    args.work_directory.mkdir(parents=True, exist_ok=True)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="napari-harpy-zarr-bridge-profile-", dir=args.work_directory) as text:
        workspace = Path(text)
        staging = workspace / "staging"
        shuffle = workspace / "shuffle"
        staging.mkdir()
        shuffle.mkdir()

        exact_start = perf_counter()
        exact_result = _write_exact_level(
            validated,
            plan,
            staging_root=staging,
            temporary_directory_root=shuffle,
            config=_ExactWriterConfig(settings, dask_worker_count=2),
        )
        exact_seconds = perf_counter() - exact_start
        effective_reader_bound = (
            exact_result.bucket_count
            if args.max_open_exact_readers is None
            else min(args.max_open_exact_readers, exact_result.bucket_count)
        )

        gc.collect()
        bridge_directory = staging / "levels/level_1"
        with _BridgeStageProfiler() as profiler, _ResourceSampler(bridge_directory) as resources:
            bridge_start = perf_counter()
            bridge_result = bridge_module._write_bridge_level(
                exact_result,
                plan,
                staging_root=staging,
                config=_BridgeWriterConfig(
                    settings,
                    max_open_exact_readers=args.max_open_exact_readers,
                ),
            )
            bridge_seconds = perf_counter() - bridge_start

        report = {
            "schema_version": "harpy-zarr-bridge-stage-profile-v1",
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "logical_cpu_count": os.cpu_count(),
                "physical_memory_bytes": psutil.virtual_memory().total,
            },
            "source": {
                "row_count": validated.row_count,
                "points_name": args.points_name,
                "source_signature": validated.source_signature,
            },
            "configuration": {
                "sequential_output_buckets": True,
                "bucket_threading": False,
                "requested_max_open_exact_readers": args.max_open_exact_readers,
                "effective_max_open_exact_readers": effective_reader_bound,
                "point_chunk_rows": settings.point_chunk_rows,
                "point_shard_rows": settings.point_shard_rows,
                "range_chunk_rows": settings.range_chunk_rows,
                "range_shard_rows": settings.range_shard_rows,
                "codec_id": settings.codec_id,
            },
            "timing_seconds": {
                "validation": validation_seconds,
                "prerequisite_exact_build": exact_seconds,
                "profiled_bridge_build": bridge_seconds,
            },
            "bridge_resources": {
                "baseline_rss_bytes": resources.baseline_rss_bytes,
                "peak_rss_bytes": resources.peak_rss_bytes,
                "incremental_peak_rss_bytes": resources.peak_rss_bytes - resources.baseline_rss_bytes,
                "peak_bridge_output_bytes": resources.peak_workspace_bytes,
            },
            "bridge_output": {
                "bucket_count": bridge_result.bucket_count,
                "tile_count": bridge_result.tile_count,
                "point_count": bridge_result.point_count,
                "range_count": bridge_result.range_count,
            },
            "profile": profiler.report(bridge_seconds=bridge_seconds),
        }
        serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
        args.json_output.write_text(serialized, encoding="utf-8")
        print(serialized, end="")


if __name__ == "__main__":
    main()
