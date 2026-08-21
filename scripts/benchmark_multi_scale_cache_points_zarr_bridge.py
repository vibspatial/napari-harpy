from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import tempfile
from pathlib import Path
from time import perf_counter

import numpy as np
import psutil
from benchmark_multi_scale_cache_points_zarr_exact import (
    _ARRAY_PATHS,
    _array_storage_summary,
    _directory_file_count,
    _directory_size,
    _ResourceSampler,
)

from napari_harpy.core.multi_scale_cache_points_zarr.build_plan import (
    _plan_points_cache,
    _PointsCacheBuildPlan,
)
from napari_harpy.core.multi_scale_cache_points_zarr.hashing import (
    BUCKET_HASH_METHOD,
    TARGET_POINTS_PER_BUCKET,
    _bucket_count_for_level,
)
from napari_harpy.core.multi_scale_cache_points_zarr.sampling import SAMPLING_METHOD, _select_sampled_tile_indices
from napari_harpy.core.multi_scale_cache_points_zarr.source import (
    ParquetPointsSource,
    PointColumnSelection,
    validate_parquet_points_source,
)
from napari_harpy.core.multi_scale_cache_points_zarr.storage.bucket_reader import _BucketReader
from napari_harpy.core.multi_scale_cache_points_zarr.storage.bucket_validation import _validate_bucket
from napari_harpy.core.multi_scale_cache_points_zarr.storage.models import (
    _LevelWriteResult,
    _ZarrWriteSettings,
)
from napari_harpy.core.multi_scale_cache_points_zarr.storage.reader_cache import _BucketReaderCache
from napari_harpy.core.multi_scale_cache_points_zarr.writer.bridge import (
    _BridgeWriterConfig,
    _write_bridge_level,
)
from napari_harpy.core.multi_scale_cache_points_zarr.writer.exact import (
    _ExactWriterConfig,
    _write_exact_level,
)

_EXPECTED_XENIUM_POINT_COUNT = 136_578_750


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and independently evaluate the full Zarr Bridge level.")
    parser.add_argument("spatialdata_path", type=Path)
    parser.add_argument("--points-name", required=True)
    parser.add_argument("--x", default="x")
    parser.add_argument("--y", default="y")
    parser.add_argument("--value", default="gene")
    parser.add_argument("--leaf-tile-size", type=int, default=512)
    parser.add_argument("--overview-point-budget", type=int, default=100_000)
    parser.add_argument("--point-chunk-rows", type=int, default=4_096)
    parser.add_argument("--point-shard-rows", type=int, default=131_072)
    parser.add_argument("--range-chunk-rows", type=int, default=8_192)
    parser.add_argument("--range-shard-rows", type=int, default=131_072)
    parser.add_argument("--codec-id", default="zstd-v1")
    parser.add_argument("--dask-worker-count", type=int, default=2)
    parser.add_argument(
        "--max-open-exact-readers",
        type=int,
        default=None,
        help="Maximum entered Exact readers; defaults to all nonempty Exact buckets.",
    )
    parser.add_argument("--expected-row-count", type=int, default=_EXPECTED_XENIUM_POINT_COUNT)
    parser.add_argument("--work-directory", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    return parser.parse_args()


def _storage_summary(result: _LevelWriteResult, staging: Path) -> dict[str, object]:
    per_array_bytes = dict.fromkeys(_ARRAY_PATHS, 0)
    logical_chunks = 0
    physical_shards = 0
    filesystem_objects = 0
    total_stored_bytes = 0
    for bucket in result.buckets:
        with _BucketReader(staging, level=result.level, bucket_id=bucket.bucket_id) as reader:
            root = reader._root
            if root is None:
                raise RuntimeError("Bridge storage-summary reader did not open its Zarr group.")
            storage = _array_storage_summary(staging / bucket.bucket_path, root)
        for name, byte_count in storage["array_stored_bytes"].items():
            per_array_bytes[name] += byte_count
        logical_chunks += storage["logical_chunk_count"]
        physical_shards += storage["physical_chunk_or_shard_count"]
        filesystem_objects += storage["filesystem_object_count"]
        total_stored_bytes += storage["total_stored_bytes"]
    return {
        "per_array_stored_bytes": per_array_bytes,
        "logical_chunk_count": logical_chunks,
        "physical_chunk_or_shard_count": physical_shards,
        "filesystem_object_count": filesystem_objects,
        "total_stored_bytes": total_stored_bytes,
    }


def _verify_bridge(
    exact_result: _LevelWriteResult,
    bridge_result: _LevelWriteResult,
    *,
    plan: _PointsCacheBuildPlan,
    staging: Path,
    reader_bound: int,
) -> dict[str, object]:
    bridge = plan.levels[1]
    capacity = bridge.max_points_per_tile
    if capacity is None:
        raise RuntimeError("Gate plan does not contain a capped Bridge level.")
    exact_by_coordinate = {
        (descriptor.tile_x, descriptor.tile_y): descriptor for descriptor in exact_result.tile_descriptors
    }
    bridge_by_coordinate = {
        (descriptor.tile_x, descriptor.tile_y): descriptor for descriptor in bridge_result.tile_descriptors
    }
    if exact_by_coordinate.keys() != bridge_by_coordinate.keys():
        raise RuntimeError("Bridge and Exact tile-coordinate sets differ.")

    maximum_exact_tile_rows = 0
    maximum_bridge_tile_rows = 0
    verified_rows = 0
    for bucket in bridge_result.buckets:
        independently_validated = _validate_bucket(staging, level=1, bucket_id=bucket.bucket_id)
        if independently_validated != bucket:
            raise RuntimeError("Independent Bridge bucket validation reconstructed a different result.")

    with (
        _BucketReaderCache(staging, max_open_readers=reader_bound) as exact_readers,
        _BucketReaderCache(staging, max_open_readers=reader_bound) as bridge_readers,
    ):
        for coordinate, exact_descriptor in exact_by_coordinate.items():
            bridge_descriptor = bridge_by_coordinate[coordinate]
            expected_count = min(exact_descriptor.n_points, capacity)
            if bridge_descriptor.n_points != expected_count:
                raise RuntimeError("Bridge descriptor count differs from the Exact-derived capacity.")
            exact_payload = exact_readers.get(
                level=exact_descriptor.level,
                bucket_id=exact_descriptor.bucket_id,
            ).read_construction_payload(exact_descriptor)
            bridge_payload = bridge_readers.get(
                level=bridge_descriptor.level,
                bucket_id=bridge_descriptor.bucket_id,
            ).read_construction_payload(bridge_descriptor)
            selected = _select_sampled_tile_indices(
                exact_payload.x_rel,
                exact_payload.y_rel,
                exact_payload.point_id,
                level=bridge.level,
                tile_x=coordinate[0],
                tile_y=coordinate[1],
                tile_size=bridge.tile_size,
                target=capacity,
            )
            bridge_order = np.argsort(bridge_payload.point_id, kind="stable")
            expected_point_id = exact_payload.point_id[selected]
            if not np.array_equal(bridge_payload.point_id[bridge_order], expected_point_id):
                raise RuntimeError("Persisted Bridge membership differs from the fresh sampler.")
            if (
                not np.array_equal(bridge_payload.x_rel[bridge_order], exact_payload.x_rel[selected])
                or not np.array_equal(bridge_payload.y_rel[bridge_order], exact_payload.y_rel[selected])
                or not np.array_equal(bridge_payload.value_id[bridge_order], exact_payload.value_id[selected])
            ):
                raise RuntimeError("A retained Bridge field differs from its Exact source row.")
            maximum_exact_tile_rows = max(maximum_exact_tile_rows, exact_payload.n_points)
            maximum_bridge_tile_rows = max(maximum_bridge_tile_rows, bridge_payload.n_points)
            verified_rows += bridge_payload.n_points
    if verified_rows != bridge_result.point_count:
        raise RuntimeError("Exhaustively verified Bridge rows do not match the result total.")
    return {
        "every_bucket_validated": True,
        "every_tile_membership_recomputed": True,
        "retained_fields_match_exact": True,
        "maximum_exact_candidate_tile_rows": maximum_exact_tile_rows,
        "maximum_bridge_output_tile_rows": maximum_bridge_tile_rows,
        "verified_rows": verified_rows,
    }


def main() -> None:
    """Run one full Exact prerequisite and independently evaluated Bridge Gate."""
    args = _parse_args()
    source = ParquetPointsSource(
        spatialdata_path=args.spatialdata_path,
        points_name=args.points_name,
        columns=PointColumnSelection(x=args.x, y=args.y, value=args.value),
    )
    validation_start = perf_counter()
    validated = validate_parquet_points_source(source)
    validation_seconds = perf_counter() - validation_start
    if validated.row_count != args.expected_row_count:
        raise RuntimeError(f"Validated source contains {validated.row_count} rows; expected {args.expected_row_count}.")
    plan = _plan_points_cache(
        validated,
        leaf_tile_size=args.leaf_tile_size,
        overview_point_budget=args.overview_point_budget,
    )
    if len(plan.levels) < 2:
        raise RuntimeError("The full-Xenium plan unexpectedly has no Bridge level.")
    settings = _ZarrWriteSettings(
        point_chunk_rows=args.point_chunk_rows,
        point_shard_rows=args.point_shard_rows,
        range_chunk_rows=args.range_chunk_rows,
        range_shard_rows=args.range_shard_rows,
        codec_id=args.codec_id,
    )

    args.work_directory.mkdir(parents=True, exist_ok=True)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="napari-harpy-zarr-bridge-evaluation-", dir=args.work_directory) as text:
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
            config=_ExactWriterConfig(settings, dask_worker_count=args.dask_worker_count),
        )
        exact_build_seconds = perf_counter() - exact_start
        effective_reader_bound = (
            exact_result.bucket_count
            if args.max_open_exact_readers is None
            else min(args.max_open_exact_readers, exact_result.bucket_count)
        )
        exact_directory = staging / "levels/level_0"
        exact_bytes_before = _directory_size(exact_directory)
        exact_objects_before = _directory_file_count(exact_directory)

        gc.collect()
        bridge_directory = staging / "levels/level_1"
        with _ResourceSampler(bridge_directory) as resources:
            bridge_start = perf_counter()
            bridge_result = _write_bridge_level(
                exact_result,
                plan,
                staging_root=staging,
                config=_BridgeWriterConfig(
                    settings,
                    max_open_exact_readers=args.max_open_exact_readers,
                ),
            )
            bridge_build_seconds = perf_counter() - bridge_start

        verification_start = perf_counter()
        verification = _verify_bridge(
            exact_result,
            bridge_result,
            plan=plan,
            staging=staging,
            reader_bound=effective_reader_bound,
        )
        verification_seconds = perf_counter() - verification_start
        storage = _storage_summary(bridge_result, staging)
        if list(shuffle.iterdir()) or list(bridge_directory.rglob("*.parquet")):
            raise RuntimeError("Bridge Gate found retained shuffle data or derived point Parquet.")
        if (
            _directory_size(exact_directory) != exact_bytes_before
            or _directory_file_count(exact_directory) != exact_objects_before
        ):
            raise RuntimeError("Bridge construction modified its Exact input level.")

        bridge = plan.levels[1]
        expected_point_count = sum(
            min(descriptor.n_points, bridge.max_points_per_tile) for descriptor in exact_result.tile_descriptors
        )
        report = {
            "schema_version": "harpy-zarr-bridge-evaluation-v1",
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
                "leaf_tile_size": bridge.tile_size,
                "bridge_capacity": bridge.max_points_per_tile,
                "bridge_point_count_upper_bound": bridge.point_count_upper_bound,
                "planned_bridge_bucket_count": _bucket_count_for_level(bridge),
                "bucket_hash_method": BUCKET_HASH_METHOD,
                "target_points_per_bucket": TARGET_POINTS_PER_BUCKET,
                "sampling_method": SAMPLING_METHOD,
                "point_chunk_rows": settings.point_chunk_rows,
                "point_shard_rows": settings.point_shard_rows,
                "range_chunk_rows": settings.range_chunk_rows,
                "range_shard_rows": settings.range_shard_rows,
                "codec_id": settings.codec_id,
                "requested_max_open_exact_readers": args.max_open_exact_readers,
                "effective_max_open_exact_readers": effective_reader_bound,
                "reader_bound_enforced_by_cache_contract": True,
            },
            "timing_seconds": {
                "validation": validation_seconds,
                "prerequisite_exact_build": exact_build_seconds,
                "bridge_build": bridge_build_seconds,
                "independent_bridge_verification": verification_seconds,
            },
            "bridge_resources": {
                "rss_sample_interval_seconds": 0.5,
                "baseline_rss_bytes": resources.baseline_rss_bytes,
                "peak_rss_bytes": resources.peak_rss_bytes,
                "incremental_peak_rss_bytes": resources.peak_rss_bytes - resources.baseline_rss_bytes,
                "peak_bridge_output_bytes": resources.peak_workspace_bytes,
            },
            "output": {
                "expected_point_count": expected_point_count,
                "bucket_count": bridge_result.bucket_count,
                "tile_count": bridge_result.tile_count,
                "point_count": bridge_result.point_count,
                "range_count": bridge_result.range_count,
                "maximum_bucket_rows": max(bucket.point_count for bucket in bridge_result.buckets),
                **storage,
            },
            "verification": {
                **verification,
                "exact_input_unchanged": True,
                "derived_point_parquet_absent": True,
                "bridge_did_not_receive_source_object": True,
            },
        }
        serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
        args.json_output.write_text(serialized, encoding="utf-8")
        print(serialized, end="")


if __name__ == "__main__":
    main()
