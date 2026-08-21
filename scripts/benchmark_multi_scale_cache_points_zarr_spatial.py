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

import napari_harpy.core.multi_scale_cache_points_zarr.writer.spatial as spatial_module
from napari_harpy.core.multi_scale_cache_points_zarr.build_plan import (
    _LevelBuildPlan,
    _plan_points_cache,
)
from napari_harpy.core.multi_scale_cache_points_zarr.hashing import (
    BUCKET_HASH_METHOD,
    TARGET_POINTS_PER_BUCKET,
    _bucket_count_for_level,
)
from napari_harpy.core.multi_scale_cache_points_zarr.models import _TileDescriptor
from napari_harpy.core.multi_scale_cache_points_zarr.payload import _PointPayload
from napari_harpy.core.multi_scale_cache_points_zarr.sampling import (
    SAMPLING_METHOD,
    _select_sampled_tile_indices,
)
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
from napari_harpy.core.multi_scale_cache_points_zarr.writer.spatial import _SpatialWriterConfig

_EXPECTED_XENIUM_POINT_COUNT = 136_578_750


class _TrackingReaderCache(_BucketReaderCache):
    """Record peak entered-reader count without changing cache behavior."""

    instances: list[_TrackingReaderCache] = []

    def __init__(self, cache_root: Path, *, max_open_readers: int) -> None:
        super().__init__(cache_root, max_open_readers=max_open_readers)
        self.peak_open_reader_count = 0
        self.instances.append(self)

    def get(self, *, level: int, bucket_id: int) -> _BucketReader:
        reader = super().get(level=level, bucket_id=bucket_id)
        self.peak_open_reader_count = max(self.peak_open_reader_count, self.open_reader_count)
        return reader


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and evaluate the full Zarr Spatial pyramid once.")
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
                raise RuntimeError("Storage-summary reader did not open its Zarr group.")
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


def _candidate_counts_by_coordinate(
    finer_result: _LevelWriteResult,
) -> dict[tuple[int, int], int]:
    counts: dict[tuple[int, int], int] = {}
    for descriptor in finer_result.tile_descriptors:
        coordinate = (descriptor.tile_x // 2, descriptor.tile_y // 2)
        counts[coordinate] = counts.get(coordinate, 0) + descriptor.n_points
    return counts


def _representative_coordinates(candidate_counts: dict[tuple[int, int], int]) -> set[tuple[int, int]]:
    coordinates = tuple(candidate_counts)
    return {
        min(coordinates, key=lambda coordinate: (candidate_counts[coordinate], coordinate[1], coordinate[0])),
        max(coordinates, key=lambda coordinate: (candidate_counts[coordinate], -coordinate[1], -coordinate[0])),
        max(coordinates, key=lambda coordinate: (coordinate[1], coordinate[0])),
    }


def _assemble_independent_candidates(
    finer_descriptors: tuple[_TileDescriptor, ...],
    *,
    finer_level: _LevelBuildPlan,
    coarser_tile_x: int,
    coarser_tile_y: int,
    readers: _BucketReaderCache,
) -> _PointPayload:
    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    value_parts: list[np.ndarray] = []
    point_parts: list[np.ndarray] = []
    for descriptor in finer_descriptors:
        payload = readers.get(level=descriptor.level, bucket_id=descriptor.bucket_id).read_construction_payload(
            descriptor
        )
        quadrant_x = descriptor.tile_x - 2 * coarser_tile_x
        quadrant_y = descriptor.tile_y - 2 * coarser_tile_y
        if quadrant_x not in (0, 1) or quadrant_y not in (0, 1):
            raise RuntimeError("Independent verification derived an invalid finer quadrant.")
        x_parts.append(
            np.ascontiguousarray(payload.x_rel + np.float32(quadrant_x * finer_level.tile_size), dtype=np.float32)
        )
        y_parts.append(
            np.ascontiguousarray(payload.y_rel + np.float32(quadrant_y * finer_level.tile_size), dtype=np.float32)
        )
        value_parts.append(payload.value_id)
        point_parts.append(payload.point_id)
    return _PointPayload(
        x_rel=np.ascontiguousarray(np.concatenate(x_parts), dtype=np.float32),
        y_rel=np.ascontiguousarray(np.concatenate(y_parts), dtype=np.float32),
        value_id=np.ascontiguousarray(np.concatenate(value_parts), dtype=np.uint32),
        point_id=np.ascontiguousarray(np.concatenate(point_parts), dtype=np.uint64),
    )


def _verify_spatial_level(
    finer_result: _LevelWriteResult,
    result: _LevelWriteResult,
    *,
    finer_level: _LevelBuildPlan,
    coarser_level: _LevelBuildPlan,
    staging: Path,
    verify_sampler_for_all_tiles: bool,
) -> dict[str, object]:
    capacity = coarser_level.max_points_per_tile
    if capacity is None:
        raise RuntimeError("Spatial Gate encountered an uncapped coarser level.")
    finer_by_coarser: dict[tuple[int, int], list[_TileDescriptor]] = {}
    for descriptor in finer_result.tile_descriptors:
        coordinate = (descriptor.tile_x // 2, descriptor.tile_y // 2)
        finer_by_coarser.setdefault(coordinate, []).append(descriptor)
    finer_by_coarser = {
        coordinate: sorted(descriptors, key=lambda descriptor: (descriptor.tile_y, descriptor.tile_x))
        for coordinate, descriptors in finer_by_coarser.items()
    }
    output_by_coordinate = {
        (descriptor.tile_x, descriptor.tile_y): descriptor for descriptor in result.tile_descriptors
    }
    if output_by_coordinate.keys() != finer_by_coarser.keys():
        raise RuntimeError("Spatial output coordinate set differs from descriptor-derived coarser tiles.")

    for bucket in result.buckets:
        if _validate_bucket(staging, level=result.level, bucket_id=bucket.bucket_id) != bucket:
            raise RuntimeError("Independent Spatial bucket validation reconstructed a different result.")

    candidate_counts = {
        coordinate: sum(descriptor.n_points for descriptor in descriptors)
        for coordinate, descriptors in finer_by_coarser.items()
    }
    representative_coordinates = (
        set(candidate_counts) if verify_sampler_for_all_tiles else _representative_coordinates(candidate_counts)
    )
    verified_rows = 0
    sampler_tiles_recomputed = 0
    maximum_candidate_rows = 0
    maximum_output_rows = 0
    with (
        _BucketReaderCache(staging, max_open_readers=finer_result.bucket_count) as finer_readers,
        _BucketReaderCache(staging, max_open_readers=result.bucket_count) as output_readers,
    ):
        for coordinate, descriptors in finer_by_coarser.items():
            output_descriptor = output_by_coordinate[coordinate]
            expected_count = min(candidate_counts[coordinate], capacity)
            if output_descriptor.n_points != expected_count:
                raise RuntimeError("Spatial descriptor count differs from its finer-derived capacity.")
            candidates = _assemble_independent_candidates(
                tuple(descriptors),
                finer_level=finer_level,
                coarser_tile_x=coordinate[0],
                coarser_tile_y=coordinate[1],
                readers=finer_readers,
            )
            output = output_readers.get(
                level=output_descriptor.level,
                bucket_id=output_descriptor.bucket_id,
            ).read_construction_payload(output_descriptor)

            candidate_order = np.argsort(candidates.point_id, kind="stable")
            sorted_candidate_ids = candidates.point_id[candidate_order]
            positions = np.searchsorted(sorted_candidate_ids, output.point_id)
            if bool((positions >= candidates.n_points).any()) or not np.array_equal(
                sorted_candidate_ids[positions], output.point_id
            ):
                raise RuntimeError("A Spatial point ID is absent from its immediate-finer contributors.")
            source_rows = candidate_order[positions]
            if (
                not np.array_equal(output.x_rel, candidates.x_rel[source_rows])
                or not np.array_equal(output.y_rel, candidates.y_rel[source_rows])
                or not np.array_equal(output.value_id, candidates.value_id[source_rows])
            ):
                raise RuntimeError("A retained Spatial field differs after independent rebasing.")

            if coordinate in representative_coordinates:
                selected = _select_sampled_tile_indices(
                    candidates.x_rel,
                    candidates.y_rel,
                    candidates.point_id,
                    level=coarser_level.level,
                    tile_x=coordinate[0],
                    tile_y=coordinate[1],
                    tile_size=coarser_level.tile_size,
                    target=capacity,
                )
                if not np.array_equal(np.sort(output.point_id), candidates.point_id[selected]):
                    raise RuntimeError("Representative Spatial membership differs from a fresh sampler run.")
                sampler_tiles_recomputed += 1

            verified_rows += output.n_points
            maximum_candidate_rows = max(maximum_candidate_rows, candidates.n_points)
            maximum_output_rows = max(maximum_output_rows, output.n_points)
    if verified_rows != result.point_count:
        raise RuntimeError("Verified Spatial rows do not match the finalized level total.")
    return {
        "every_bucket_validated": True,
        "every_output_point_nested_in_immediate_predecessor": True,
        "every_retained_field_matches_after_rebasing": True,
        "sampler_tiles_recomputed": sampler_tiles_recomputed,
        "sampler_scope": "all" if verify_sampler_for_all_tiles else "sparse_dense_edge_representatives",
        "verified_rows": verified_rows,
        "maximum_candidate_tile_rows": maximum_candidate_rows,
        "maximum_output_tile_rows": maximum_output_rows,
    }


def main() -> None:
    """Run one full current-tree Spatial-pyramid engineering Gate."""
    args = _parse_args()
    source = ParquetPointsSource(
        spatialdata_path=args.spatialdata_path,
        points_name=args.points_name,
        columns=PointColumnSelection(x=args.x, y=args.y, value=args.value),
    )
    print("Validating canonical Xenium source...", flush=True)
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
    if len(plan.levels) < 3:
        raise RuntimeError("The full-Xenium plan unexpectedly has no Spatial levels.")
    settings = _ZarrWriteSettings(
        point_chunk_rows=args.point_chunk_rows,
        point_shard_rows=args.point_shard_rows,
        range_chunk_rows=args.range_chunk_rows,
        range_shard_rows=args.range_shard_rows,
        codec_id=args.codec_id,
    )

    args.work_directory.mkdir(parents=True, exist_ok=True)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="napari-harpy-zarr-spatial-evaluation-", dir=args.work_directory) as text:
        workspace = Path(text)
        staging = workspace / "staging"
        shuffle = workspace / "shuffle"
        staging.mkdir()
        shuffle.mkdir()

        print("Building Exact prerequisite...", flush=True)
        exact_start = perf_counter()
        exact_result = _write_exact_level(
            validated,
            plan,
            staging_root=staging,
            temporary_directory_root=shuffle,
            config=_ExactWriterConfig(settings, dask_worker_count=args.dask_worker_count),
        )
        exact_seconds = perf_counter() - exact_start

        print("Building Bridge prerequisite...", flush=True)
        bridge_start = perf_counter()
        bridge_result = _write_bridge_level(
            exact_result,
            plan,
            staging_root=staging,
            config=_BridgeWriterConfig(settings),
        )
        bridge_seconds = perf_counter() - bridge_start

        prerequisite_snapshots = {
            level.level: (
                _directory_size(staging / level.relative_directory),
                _directory_file_count(staging / level.relative_directory),
            )
            for level in plan.levels[:2]
        }
        source_snapshot = (_directory_size(source.parquet_path), _directory_file_count(source.parquet_path))

        spatial_results: list[_LevelWriteResult] = []
        construction_reports: list[dict[str, object]] = []
        finer_result = bridge_result
        finer_level = plan.levels[1]
        original_reader_cache = spatial_module._BucketReaderCache
        try:
            spatial_module._BucketReaderCache = _TrackingReaderCache
            for coarser_level in plan.levels[2:]:
                gc.collect()
                _TrackingReaderCache.instances.clear()
                output_directory = staging / coarser_level.relative_directory
                print(
                    f"Building Spatial level {coarser_level.level}: "
                    f"tile={coarser_level.tile_size}, grid={coarser_level.grid_width}x{coarser_level.grid_height}, "
                    f"capacity={coarser_level.max_points_per_tile}...",
                    flush=True,
                )
                with _ResourceSampler(output_directory) as resources:
                    started = perf_counter()
                    result = spatial_module._write_spatial_level(
                        finer_result,
                        finer_level=finer_level,
                        coarser_level=coarser_level,
                        staging_root=staging,
                        config=_SpatialWriterConfig(settings),
                    )
                    elapsed = perf_counter() - started
                if len(_TrackingReaderCache.instances) != 1:
                    raise RuntimeError("Spatial level did not create exactly one reader cache.")
                tracked_cache = _TrackingReaderCache.instances[0]
                candidate_counts = _candidate_counts_by_coordinate(finer_result)
                storage = _storage_summary(result, staging)
                construction_reports.append(
                    {
                        "level": coarser_level.level,
                        "tile_size": coarser_level.tile_size,
                        "grid_width": coarser_level.grid_width,
                        "grid_height": coarser_level.grid_height,
                        "capacity": coarser_level.max_points_per_tile,
                        "point_count_upper_bound": coarser_level.point_count_upper_bound,
                        "planned_bucket_count": _bucket_count_for_level(coarser_level),
                        "input_point_count": finer_result.point_count,
                        "input_tile_count": finer_result.tile_count,
                        "maximum_descriptor_candidate_count": max(candidate_counts.values()),
                        "construction_seconds": elapsed,
                        "reader_capacity": finer_result.bucket_count,
                        "peak_open_readers": tracked_cache.peak_open_reader_count,
                        "readers_closed_after_level": tracked_cache.open_reader_count == 0,
                        "baseline_rss_bytes": resources.baseline_rss_bytes,
                        "peak_rss_bytes": resources.peak_rss_bytes,
                        "incremental_peak_rss_bytes": resources.peak_rss_bytes - resources.baseline_rss_bytes,
                        "peak_output_bytes": resources.peak_workspace_bytes,
                        "bucket_count": result.bucket_count,
                        "tile_count": result.tile_count,
                        "point_count": result.point_count,
                        "range_count": result.range_count,
                        "maximum_bucket_rows": max(bucket.point_count for bucket in result.buckets),
                        "maximum_output_tile_rows": max(descriptor.n_points for descriptor in result.tile_descriptors),
                        **storage,
                    }
                )
                print(
                    f"Completed level {coarser_level.level}: {result.point_count:,} points in {elapsed:.2f}s.",
                    flush=True,
                )
                spatial_results.append(result)
                finer_result = result
                finer_level = coarser_level
        finally:
            spatial_module._BucketReaderCache = original_reader_cache

        validation_reports: list[dict[str, object]] = []
        finer_result = bridge_result
        finer_level = plan.levels[1]
        for index, (coarser_level, result) in enumerate(zip(plan.levels[2:], spatial_results, strict=True)):
            print(f"Validating Spatial level {coarser_level.level}...", flush=True)
            started = perf_counter()
            verification = _verify_spatial_level(
                finer_result,
                result,
                finer_level=finer_level,
                coarser_level=coarser_level,
                staging=staging,
                verify_sampler_for_all_tiles=index == len(spatial_results) - 1,
            )
            validation_reports.append(
                {
                    "level": coarser_level.level,
                    "validation_seconds": perf_counter() - started,
                    **verification,
                }
            )
            finer_result = result
            finer_level = coarser_level

        if spatial_results[-1].point_count > plan.overview_point_budget:
            raise RuntimeError("Terminal Spatial level exceeds the overview point budget.")
        prerequisites_unchanged = all(
            prerequisite_snapshots[level.level]
            == (
                _directory_size(staging / level.relative_directory),
                _directory_file_count(staging / level.relative_directory),
            )
            for level in plan.levels[:2]
        )
        source_unchanged = source_snapshot == (
            _directory_size(source.parquet_path),
            _directory_file_count(source.parquet_path),
        )
        if not prerequisites_unchanged or not source_unchanged:
            raise RuntimeError("Spatial construction or validation modified an input artifact.")
        if list(shuffle.iterdir()) or list((staging / "levels").rglob("*.parquet")):
            raise RuntimeError("Spatial Gate found retained shuffle data or derived point Parquet.")

        report = {
            "schema_version": "harpy-zarr-spatial-evaluation-v1",
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
                "leaf_tile_size": args.leaf_tile_size,
                "overview_point_budget": args.overview_point_budget,
                "bucket_hash_method": BUCKET_HASH_METHOD,
                "target_points_per_bucket": TARGET_POINTS_PER_BUCKET,
                "sampling_method": SAMPLING_METHOD,
                "point_chunk_rows": settings.point_chunk_rows,
                "point_shard_rows": settings.point_shard_rows,
                "range_chunk_rows": settings.range_chunk_rows,
                "range_shard_rows": settings.range_shard_rows,
                "codec_id": settings.codec_id,
                "max_open_finer_readers": None,
                "sequential_output_buckets": True,
            },
            "timing_seconds": {
                "source_validation": validation_seconds,
                "prerequisite_exact_build": exact_seconds,
                "prerequisite_bridge_build": bridge_seconds,
                "spatial_build_total": sum(report["construction_seconds"] for report in construction_reports),
                "spatial_validation_total": sum(report["validation_seconds"] for report in validation_reports),
            },
            "planned_levels": [
                {
                    "level": level.level,
                    "kind": level.kind.value,
                    "tile_size": level.tile_size,
                    "grid_width": level.grid_width,
                    "grid_height": level.grid_height,
                    "capacity": level.max_points_per_tile,
                    "point_count_upper_bound": level.point_count_upper_bound,
                }
                for level in plan.levels
            ],
            "spatial_construction": construction_reports,
            "spatial_validation": validation_reports,
            "verification": {
                "terminal_level": spatial_results[-1].level,
                "terminal_point_count": spatial_results[-1].point_count,
                "overview_budget_respected": True,
                "exact_and_bridge_inputs_unchanged": prerequisites_unchanged,
                "canonical_source_unchanged": source_unchanged,
                "spatial_writer_did_not_receive_source_object": True,
                "derived_point_parquet_absent": True,
            },
        }
        serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
        args.json_output.write_text(serialized, encoding="utf-8")
        print(serialized, end="", flush=True)


if __name__ == "__main__":
    main()
