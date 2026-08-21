from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import psutil
import pyarrow as pa

import napari_harpy.core.multi_scale_cache_points_zarr.source.validation as validation_module
from napari_harpy.core.multi_scale_cache_points_zarr.source import (
    ParquetPointsSource,
    PointColumnSelection,
    ValidatedPointsSource,
    validate_parquet_points_source,
)

_BENCHMARK_SCHEMA_VERSION = "harpy-points-validation-benchmark-v1"
_RSS_SAMPLE_INTERVAL_SECONDS = 0.01


@dataclass
class _RunMeasurements:
    inventory_seconds: list[float]
    scan_seconds: list[float]
    batch_count: int = 0
    largest_batch_rows: int = 0
    largest_batch_nbytes: int = 0


class _MeasuredParquetFile:
    def __init__(self, parquet_file: Any, measurements: _RunMeasurements) -> None:
        self._parquet_file = parquet_file
        self._measurements = measurements

    def iter_batches(self, *args: object, **kwargs: object) -> Iterator[pa.RecordBatch]:
        for batch in self._parquet_file.iter_batches(*args, **kwargs):
            self._measurements.batch_count += 1
            self._measurements.largest_batch_rows = max(
                self._measurements.largest_batch_rows,
                batch.num_rows,
            )
            self._measurements.largest_batch_nbytes = max(
                self._measurements.largest_batch_nbytes,
                batch.nbytes,
            )
            yield batch


class _RssSampler:
    def __init__(self, interval_seconds: float) -> None:
        self._interval_seconds = interval_seconds
        self._process = psutil.Process()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample_until_stopped, daemon=True)
        self.baseline_bytes = self._process.memory_info().rss
        self.peak_bytes = self.baseline_bytes

    def __enter__(self) -> _RssSampler:
        self._thread.start()
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self._sample()
        self._stop.set()
        self._thread.join()

    def _sample_until_stopped(self) -> None:
        while not self._stop.wait(self._interval_seconds):
            self._sample()

    def _sample(self) -> None:
        self.peak_bytes = max(self.peak_bytes, self._process.memory_info().rss)


@contextmanager
def _instrument_validation(measurements: _RunMeasurements) -> Iterator[None]:
    read_inventory = validation_module._read_parquet_source_inventory
    scan_content = validation_module._scan_points_content
    open_content_file = validation_module._open_parquet_content_file

    def measured_inventory(source: ParquetPointsSource) -> Any:
        start = perf_counter()
        try:
            return read_inventory(source)
        finally:
            measurements.inventory_seconds.append(perf_counter() - start)

    def measured_scan(*args: object, **kwargs: object) -> Any:
        start = perf_counter()
        try:
            return scan_content(*args, **kwargs)
        finally:
            measurements.scan_seconds.append(perf_counter() - start)

    def measured_content_file(*args: object, **kwargs: object) -> _MeasuredParquetFile:
        return _MeasuredParquetFile(
            open_content_file(*args, **kwargs),
            measurements,
        )

    validation_module._read_parquet_source_inventory = measured_inventory
    validation_module._scan_points_content = measured_scan
    validation_module._open_parquet_content_file = measured_content_file
    try:
        yield
    finally:
        validation_module._read_parquet_source_inventory = read_inventory
        validation_module._scan_points_content = scan_content
        validation_module._open_parquet_content_file = open_content_file


def _run_once(
    source: ParquetPointsSource,
    *,
    run_index: int,
    run_label: str,
    max_batch_rows: int,
) -> dict[str, object]:
    measurements = _RunMeasurements(inventory_seconds=[], scan_seconds=[])
    sampler = _RssSampler(_RSS_SAMPLE_INTERVAL_SECONDS)

    with _instrument_validation(measurements), sampler:
        start = perf_counter()
        validated = validate_parquet_points_source(
            source,
            max_batch_rows=max_batch_rows,
        )
        total_seconds = perf_counter() - start

    if len(measurements.inventory_seconds) != 2 or len(measurements.scan_seconds) != 1:
        raise RuntimeError(
            "Expected one initial inventory, one scan, and one final inventory from validate_parquet_points_source()."
        )

    return {
        "run_index": run_index,
        "run_label": run_label,
        "timings_seconds": {
            "initial_inventory": measurements.inventory_seconds[0],
            "scan": measurements.scan_seconds[0],
            "final_inventory": measurements.inventory_seconds[1],
            "total": total_seconds,
        },
        "batches": {
            "count": measurements.batch_count,
            "largest_rows": measurements.largest_batch_rows,
            "largest_nbytes": measurements.largest_batch_nbytes,
        },
        "memory": {
            "sample_interval_seconds": _RSS_SAMPLE_INTERVAL_SECONDS,
            "baseline_rss_bytes": sampler.baseline_bytes,
            "peak_rss_bytes": sampler.peak_bytes,
            "incremental_peak_rss_bytes": max(0, sampler.peak_bytes - sampler.baseline_bytes),
        },
        "validated_source": _validated_source_summary(validated),
    }


def _validated_source_summary(validated: ValidatedPointsSource) -> dict[str, object]:
    return {
        "row_count": validated.row_count,
        "file_count": len(validated.files),
        "row_group_count": sum(source_file.row_group_count for source_file in validated.files),
        "value_count": validated.value_table.num_rows,
        "bounds": {
            "x_min": validated.bounds.x_min,
            "x_max": validated.bounds.x_max,
            "y_min": validated.bounds.y_min,
            "y_max": validated.bounds.y_max,
        },
        "source_signature": validated.source_signature,
        "source_signature_method": validated.source_signature_method,
        "value_normalization_method": validated.value_normalization_method,
        "point_id_policy": validated.point_id_policy,
    }


def _version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _context(source_path: Path) -> dict[str, object]:
    disk = shutil.disk_usage(source_path)
    return {
        "versions": {
            "python": platform.python_version(),
            "napari_harpy": _version("napari-harpy"),
            "pyarrow": _version("pyarrow"),
            "numpy": _version("numpy"),
            "dask": _version("dask"),
            "spatialdata": _version("spatialdata"),
            "psutil": _version("psutil"),
        },
        "machine": {
            "system": platform.system(),
            "release": platform.release(),
            "architecture": platform.machine(),
            "processor": platform.processor() or None,
            "logical_cpu_count": os.cpu_count(),
            "total_physical_memory_bytes": psutil.virtual_memory().total,
        },
        "storage": {
            "capacity_bytes": disk.total,
            "free_bytes": disk.free,
        },
    }


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark multiscale point-source validation.")
    parser.add_argument("spatialdata_path", type=Path, help="Path to the SpatialData zarr root.")
    parser.add_argument("--points-name", required=True, help="Name of the SpatialData points element.")
    parser.add_argument("--x", default="x", help="Physical x-coordinate column name.")
    parser.add_argument("--y", default="y", help="Physical y-coordinate column name.")
    parser.add_argument("--value", default="gene", help="Physical value column name.")
    parser.add_argument("--runs", type=_positive_integer, default=1, help="Number of measured runs.")
    parser.add_argument("--run-label", default="warm", help="Caller-supplied cache-state label.")
    parser.add_argument(
        "--max-batch-rows",
        type=_positive_integer,
        default=1_048_576,
        help="Maximum rows requested from each Arrow record batch.",
    )
    parser.add_argument("--json-output", type=Path, required=True, help="Destination for versioned JSON output.")
    return parser.parse_args()


def _print_summary(run: dict[str, object]) -> None:
    timings = run["timings_seconds"]
    batches = run["batches"]
    memory = run["memory"]
    validated = run["validated_source"]
    assert isinstance(timings, dict)
    assert isinstance(batches, dict)
    assert isinstance(memory, dict)
    assert isinstance(validated, dict)
    print(
        f"run {run['run_index']} ({run['run_label']}): "
        f"total={timings['total']:.3f}s, scan={timings['scan']:.3f}s, "
        f"rows={validated['row_count']:,}, values={validated['value_count']:,}, "
        f"batches={batches['count']}, incremental_peak_rss={memory['incremental_peak_rss_bytes'] / 2**20:.1f} MiB"
    )


def main() -> None:
    """Run the requested validation benchmark and write its JSON result."""
    args = _parse_args()
    source_path = args.spatialdata_path.resolve()
    source = ParquetPointsSource(
        spatialdata_path=source_path,
        points_name=args.points_name,
        columns=PointColumnSelection(x=args.x, y=args.y, value=args.value),
    )

    runs = [
        _run_once(
            source,
            run_index=index,
            run_label=args.run_label,
            max_batch_rows=args.max_batch_rows,
        )
        for index in range(1, args.runs + 1)
    ]
    result = {
        "schema_version": _BENCHMARK_SCHEMA_VERSION,
        "parameters": {
            "spatialdata_path": str(source_path),
            "points_name": args.points_name,
            "x": args.x,
            "y": args.y,
            "value": args.value,
            "runs": args.runs,
            "run_label": args.run_label,
            "max_batch_rows": args.max_batch_rows,
        },
        "context": _context(source_path),
        "runs": runs,
    }
    args.json_output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for run in runs:
        _print_summary(run)
    print(f"JSON: {args.json_output}")


if __name__ == "__main__":
    main()
