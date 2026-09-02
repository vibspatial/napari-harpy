"""Build and retain one explicitly configured tiled-points cache variant."""

from __future__ import annotations

import argparse
import json
import os
import platform
import threading
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import psutil

from napari_harpy.core.multi_scale_cache_points_zarr.builder import (
    _build_points_cache_zarr,
    _PointsCacheBuilderConfig,
)
from napari_harpy.core.multi_scale_cache_points_zarr.cache_format import (
    _ValueMajorMetadata,
    _ValueMajorWriteSettings,
)
from napari_harpy.core.multi_scale_cache_points_zarr.source import (
    ParquetPointsSource,
    PointColumnSelection,
    validate_parquet_points_source,
)
from napari_harpy.core.multi_scale_cache_points_zarr.storage.catalog_reader import _CatalogReader
from napari_harpy.core.multi_scale_cache_points_zarr.storage.models import _ZarrWriteSettings

_RSS_SAMPLE_INTERVAL_SECONDS = 0.25


class _RssSampler:
    def __init__(self) -> None:
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
        while not self._stop.wait(_RSS_SAMPLE_INTERVAL_SECONDS):
            self._sample()

    def _sample(self) -> None:
        self.peak_bytes = max(self.peak_bytes, self._process.memory_info().rss)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build one retained tiled-points cache experiment variant.")
    parser.add_argument("spatialdata_path", type=Path)
    parser.add_argument("--points-name", required=True)
    parser.add_argument("--x", default="x")
    parser.add_argument("--y", default="y")
    parser.add_argument("--value", default="gene")
    parser.add_argument("--leaf-tile-size", type=int, default=512)
    parser.add_argument("--overview-point-budget", type=int, default=100_000)
    parser.add_argument("--dask-worker-count", type=int, default=2)
    parser.add_argument("--expected-row-count", type=int, default=0)
    parser.add_argument("--target-points-per-bucket", type=int, default=2_000_000)
    parser.add_argument("--point-chunk-rows", type=int, default=4_096)
    parser.add_argument("--point-shard-rows", type=int, default=131_072)
    parser.add_argument("--range-chunk-rows", type=int, default=8_192)
    parser.add_argument("--range-shard-rows", type=int, default=131_072)
    parser.add_argument("--codec-id", default="zstd-v1")
    parser.add_argument("--value-major-point-chunk-rows", type=int, default=4_096)
    parser.add_argument("--value-major-point-shard-rows", type=int, default=131_072)
    parser.add_argument("--value-major-construction-batch-points", type=int, default=1_048_576)
    parser.add_argument("--max-open-value-major-readers", type=int)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--temporary-directory-root", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    return parser.parse_args()


def _directory_summary(root: Path) -> tuple[int, int]:
    byte_count = 0
    file_count = 0
    for directory, _, filenames in os.walk(root):
        file_count += len(filenames)
        for filename in filenames:
            try:
                byte_count += (Path(directory) / filename).stat().st_size
            except FileNotFoundError:
                continue
    return byte_count, file_count


def _value_major_level_summary(
    cache_root: Path,
    *,
    level: int,
    point_count: int,
    value_count: int,
) -> dict[str, int | float]:
    stored_bytes, file_count = _directory_summary(cache_root / f"value_major/level_{level}")
    logical_bytes = point_count * 2 * 4 + (value_count + 1) * 8
    return {
        "logical_bytes": logical_bytes,
        "stored_bytes": stored_bytes,
        "filesystem_file_count": file_count,
        "logical_to_stored_ratio": logical_bytes / stored_bytes,
    }


def main() -> None:
    """Validate the source, build one cache variant, and persist provenance."""
    args = _parse_args()
    if not args.spatialdata_path.is_dir():
        raise ValueError("`spatialdata_path` must be an existing directory.")
    if not args.temporary_directory_root.is_dir():
        raise ValueError("`temporary_directory_root` must be an existing directory.")
    if args.output_path.exists():
        raise FileExistsError(f"Variant output already exists: {args.output_path}")
    if args.json_output.exists():
        raise FileExistsError(f"Variant report already exists: {args.json_output}")
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)

    source = ParquetPointsSource(
        spatialdata_path=args.spatialdata_path,
        points_name=args.points_name,
        columns=PointColumnSelection(x=args.x, y=args.y, value=args.value),
    )
    print("Validating canonical points source...", flush=True)
    started = perf_counter()
    validated = validate_parquet_points_source(source)
    validation_seconds = perf_counter() - started
    if args.expected_row_count > 0 and validated.row_count != args.expected_row_count:
        raise RuntimeError(f"Validated {validated.row_count} rows; expected {args.expected_row_count}.")

    zarr_settings = _ZarrWriteSettings(
        point_chunk_rows=args.point_chunk_rows,
        point_shard_rows=args.point_shard_rows,
        range_chunk_rows=args.range_chunk_rows,
        range_shard_rows=args.range_shard_rows,
        codec_id=args.codec_id,
    )
    config = _PointsCacheBuilderConfig(
        leaf_tile_size=args.leaf_tile_size,
        overview_point_budget=args.overview_point_budget,
        dask_worker_count=args.dask_worker_count,
        target_points_per_bucket=args.target_points_per_bucket,
        zarr_settings=zarr_settings,
        value_major_settings=_ValueMajorWriteSettings(
            point_chunk_rows=args.value_major_point_chunk_rows,
            point_shard_rows=args.value_major_point_shard_rows,
            construction_batch_points=args.value_major_construction_batch_points,
        ),
        max_open_value_major_readers=args.max_open_value_major_readers,
    )
    print(
        f"Building variant: point_chunk_rows={args.point_chunk_rows}, "
        f"target_points_per_bucket={args.target_points_per_bucket}...",
        flush=True,
    )
    with _RssSampler() as resources:
        started = perf_counter()
        output = _build_points_cache_zarr(
            validated,
            output_path=args.output_path,
            temporary_directory_root=args.temporary_directory_root,
            config=config,
        )
        build_seconds = perf_counter() - started
    stored_bytes, file_count = _directory_summary(output)
    with _CatalogReader(output) as reader:
        attributes = reader.attributes
        value_count = attributes.catalog.value_count
        levels = tuple(
            {
                "level": level.level,
                "kind": level.kind,
                "bucket_count": level.bucket_count,
                "tile_count": level.tile_count,
                "point_count": level.point_count,
                "range_count": level.range_count,
                "value_major": _value_major_level_summary(
                    output,
                    level=level.level,
                    point_count=level.point_count,
                    value_count=value_count,
                ),
            }
            for level in attributes.levels
        )
        generation_id = attributes.cache_generation_id
        if attributes.build.target_points_per_bucket != args.target_points_per_bucket:
            raise RuntimeError("Published bucket target differs from the requested experiment configuration.")
        if attributes.zarr_settings != zarr_settings:
            raise RuntimeError("Published Zarr settings differ from the requested experiment configuration.")
        if attributes.value_major != _ValueMajorMetadata.from_write_settings(config.value_major_settings):
            raise RuntimeError("Published value-major settings differ from the requested experiment configuration.")
    value_major_logical_bytes = sum(level["value_major"]["logical_bytes"] for level in levels)  # type: ignore[index]
    value_major_stored_bytes = sum(level["value_major"]["stored_bytes"] for level in levels)  # type: ignore[index]

    report = {
        "schema_version": "harpy-tiled-points-cache-variant-v1",
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "logical_cpu_count": os.cpu_count(),
            "physical_memory_bytes": psutil.virtual_memory().total,
        },
        "source": {
            "spatialdata_path": str(args.spatialdata_path.resolve()),
            "points_name": args.points_name,
            "row_count": validated.row_count,
            "value_count": validated.value_table.num_rows,
            "source_signature": validated.source_signature,
        },
        "configuration": {
            "leaf_tile_size": args.leaf_tile_size,
            "overview_point_budget": args.overview_point_budget,
            "dask_worker_count": args.dask_worker_count,
            "target_points_per_bucket": args.target_points_per_bucket,
            **asdict(zarr_settings),
            **asdict(config.catalog_settings),
            "max_open_value_major_readers": config.max_open_value_major_readers,
            "value_major": asdict(config.value_major_settings),
        },
        "publication": {
            "output_path": str(output.resolve()),
            "cache_generation_id": generation_id,
            "stored_bytes": stored_bytes,
            "filesystem_file_count": file_count,
            "levels": levels,
            "value_major": {
                "logical_bytes": value_major_logical_bytes,
                "stored_bytes": value_major_stored_bytes,
                "logical_to_stored_ratio": value_major_logical_bytes / value_major_stored_bytes,
            },
        },
        "timing_seconds": {
            "source_validation": validation_seconds,
            "complete_builder": build_seconds,
        },
        "resources": {
            "rss_sample_interval_seconds": _RSS_SAMPLE_INTERVAL_SECONDS,
            "baseline_rss_bytes": resources.baseline_bytes,
            "peak_rss_bytes": resources.peak_bytes,
            "incremental_peak_rss_bytes": resources.peak_bytes - resources.baseline_bytes,
        },
    }
    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.json_output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")


if __name__ == "__main__":
    main()
