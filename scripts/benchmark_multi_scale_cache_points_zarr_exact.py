from __future__ import annotations

import argparse
import json
import math
import os
import platform
import tempfile
import threading
from collections.abc import Callable
from pathlib import Path
from time import perf_counter

import numpy as np
import psutil
import pyarrow.compute as pc
import pyarrow.parquet as pq
import zarr
from zarr.storage import LocalStore

from napari_harpy.core.multi_scale_cache_points import (
    ParquetPointsSource,
    PointColumnSelection,
    validate_parquet_points_source,
)
from napari_harpy.core.multi_scale_cache_points.models import ValidatedPointsSource
from napari_harpy.core.multi_scale_cache_points.value_normalization import _normalized_row_values
from napari_harpy.core.multi_scale_cache_points_zarr.build_plan import (
    _plan_points_cache,
    _PointsCacheBuildPlan,
)
from napari_harpy.core.multi_scale_cache_points_zarr.hashing import (
    BUCKET_HASH_METHOD,
    TARGET_POINTS_PER_BUCKET,
    _bucket_count_for_level,
)
from napari_harpy.core.multi_scale_cache_points_zarr.models import _TileDescriptor
from napari_harpy.core.multi_scale_cache_points_zarr.storage.bucket_reader import _BucketReader
from napari_harpy.core.multi_scale_cache_points_zarr.storage.bucket_validation import _validate_bucket
from napari_harpy.core.multi_scale_cache_points_zarr.storage.models import (
    _LevelWriteResult,
    _ZarrWriteSettings,
)
from napari_harpy.core.multi_scale_cache_points_zarr.writer.exact import (
    _ExactWriterConfig,
    _write_exact_level,
)

_ARRAY_PATHS = (
    "location",
    "point_id",
    "value_id",
    "tile_x",
    "tile_y",
    "tile_offset",
    "ranges/tile_indptr",
    "ranges/value_id",
    "ranges/row_start",
    "ranges/row_count",
)
_EXPECTED_XENIUM_POINT_COUNT = 136_578_750
_RSS_SAMPLE_INTERVAL_SECONDS = 0.5


class _ResourceSampler:
    """Sample process RSS and complete benchmark-workspace bytes."""

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
        while not self._stop.wait(_RSS_SAMPLE_INTERVAL_SECONDS):
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


def _directory_file_count(root: Path) -> int:
    return sum(len(filenames) for _, _, filenames in os.walk(root))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and independently evaluate the complete Zarr-backed Exact level.",
    )
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


def _array_storage_summary(bucket_path: Path, root: zarr.Group) -> dict[str, object]:
    stored_bytes: dict[str, int] = {}
    logical_chunks = 0
    physical_shards = 0
    for name in _ARRAY_PATHS:
        array = root[name]
        if not isinstance(array, zarr.Array):
            raise RuntimeError(f"Required bucket node is not an array: {name}.")
        array_path = bucket_path.joinpath(*name.split("/"))
        stored_bytes[name] = _directory_size(array_path)
        logical_chunks += math.prod(
            math.ceil(size / chunk) for size, chunk in zip(array.shape, array.chunks, strict=True)
        )
        outer = array.shards if array.shards is not None else array.chunks
        physical_shards += math.prod(math.ceil(size / shard) for size, shard in zip(array.shape, outer, strict=True))
    return {
        "array_stored_bytes": stored_bytes,
        "logical_chunk_count": logical_chunks,
        "physical_chunk_or_shard_count": physical_shards,
        "filesystem_object_count": _directory_file_count(bucket_path),
        "total_stored_bytes": _directory_size(bucket_path),
    }


def _verify_and_summarize(
    result: _LevelWriteResult,
    *,
    staging: Path,
    validated: ValidatedPointsSource,
    plan: _PointsCacheBuildPlan,
    workspace: Path,
) -> tuple[dict[str, object], dict[int, _TileDescriptor]]:
    """Validate every store and verify global IDs, values, and coordinates."""
    row_count = validated.row_count
    exact = plan.levels[0]
    seen = np.memmap(workspace / "seen-point-ids.bin", mode="w+", dtype=np.uint8, shape=((row_count + 7) // 8,))
    reconstructed = np.memmap(
        workspace / "reconstructed-location.bin",
        mode="w+",
        dtype=np.float64,
        shape=(row_count, 2),
    )
    output_value_id = np.memmap(
        workspace / "output-value-id.bin",
        mode="w+",
        dtype=np.uint32,
        shape=(row_count,),
    )
    observed_value_counts = np.zeros(validated.value_table.num_rows, dtype=np.uint64)
    value_tile_counts = np.zeros(validated.value_table.num_rows, dtype=np.uint64)
    first_descriptor_by_value: dict[int, _TileDescriptor] = {}
    per_array_bytes = dict.fromkeys(_ARRAY_PATHS, 0)
    observed_rows = 0
    maximum_bucket_rows = 0
    maximum_tile_rows = 0
    logical_chunks = 0
    physical_shards = 0
    filesystem_objects = 0
    total_stored_bytes = 0

    for bucket in result.buckets:
        independently_validated = _validate_bucket(staging, level=0, bucket_id=bucket.bucket_id)
        if independently_validated != bucket:
            raise RuntimeError("Independent bucket validation reconstructed a different result.")
        maximum_bucket_rows = max(maximum_bucket_rows, bucket.point_count)
        bucket_path = staging / bucket.bucket_path
        store = LocalStore(bucket_path, read_only=True)
        try:
            root = zarr.open_group(store=store, mode="r", zarr_format=3, use_consolidated=False)
            storage = _array_storage_summary(bucket_path, root)
        finally:
            store.close()
        for name, byte_count in storage["array_stored_bytes"].items():
            per_array_bytes[name] += byte_count
        logical_chunks += storage["logical_chunk_count"]
        physical_shards += storage["physical_chunk_or_shard_count"]
        filesystem_objects += storage["filesystem_object_count"]
        total_stored_bytes += storage["total_stored_bytes"]

        with _BucketReader(staging, level=0, bucket_id=bucket.bucket_id) as reader:
            for descriptor in bucket.tile_descriptors:
                payload = reader.read_complete(descriptor)
                maximum_tile_rows = max(maximum_tile_rows, payload.n_points)
                point_ids = payload.point_id
                if len(np.unique(point_ids)) != len(point_ids) or bool((point_ids >= row_count).any()):
                    raise RuntimeError("A finalized tile contains duplicate or out-of-range point IDs.")
                byte_indices = point_ids >> np.uint64(3)
                masks = np.left_shift(np.uint8(1), (point_ids & np.uint64(7)).astype(np.uint8))
                if bool((seen[byte_indices] & masks).any()):
                    raise RuntimeError("A finalized point ID occurs in more than one tile or bucket.")
                np.bitwise_or.at(seen, byte_indices, masks)
                observed_rows += payload.n_points

                unique_values, counts = np.unique(payload.value_id, return_counts=True)
                np.add.at(observed_value_counts, unique_values.astype(np.intp), counts.astype(np.uint64))
                np.add.at(value_tile_counts, unique_values.astype(np.intp), np.uint64(1))
                for value_id in unique_values.tolist():
                    first_descriptor_by_value.setdefault(value_id, descriptor)

                reconstructed[point_ids, 0] = (
                    plan.x_origin + descriptor.tile_x * exact.tile_size + payload.x_rel.astype(np.float64)
                )
                reconstructed[point_ids, 1] = (
                    plan.y_origin + descriptor.tile_y * exact.tile_size + payload.y_rel.astype(np.float64)
                )
                output_value_id[point_ids] = payload.value_id

    expected_value_counts = np.asarray(validated.value_table["n_points"].combine_chunks(), dtype=np.uint64)
    if observed_rows != row_count or not np.array_equal(observed_value_counts, expected_value_counts):
        raise RuntimeError("Finalized Exact totals do not match the validated source.")
    reconstructed.flush()
    output_value_id.flush()
    seen.flush()

    tolerance = float(np.spacing(np.float32(exact.tile_size)))
    histogram_edges = np.array([0.0, 0.01, 0.1, 0.25, 0.5, 1.0, np.inf])
    histogram_counts = np.zeros(len(histogram_edges) - 1, dtype=np.uint64)
    maximum_error = 0.0
    labels = validated.value_table["value"].combine_chunks()
    point_id_start = 0
    for source_file in validated.files:
        parquet_file = pq.ParquetFile(validated.source.parquet_path / source_file.relative_path)
        try:
            for row_group_index, row_group in enumerate(source_file.row_groups):
                table = parquet_file.read_row_group(
                    row_group_index,
                    columns=[validated.source.columns.x, validated.source.columns.y, validated.source.columns.value],
                )
                if table.num_rows != row_group.row_count:
                    raise RuntimeError("Source row-group size changed after validation.")
                stop = point_id_start + table.num_rows
                x = np.asarray(table[validated.source.columns.x], dtype=np.float64)
                y = np.asarray(table[validated.source.columns.y], dtype=np.float64)
                errors = np.maximum(
                    np.abs(reconstructed[point_id_start:stop, 0] - x),
                    np.abs(reconstructed[point_id_start:stop, 1] - y),
                )
                if len(errors):
                    maximum_error = max(maximum_error, float(errors.max()))
                    histogram_counts += np.histogram(errors / tolerance, bins=histogram_edges)[0].astype(np.uint64)
                # Parquet yields a ChunkedArray even when this explicit row-group
                # read contains one physical chunk. The canonical normalizer
                # handles Arrow arrays (including DictionaryArray), so combine
                # the wrapper before comparing source rows with finalized IDs.
                normalized = _normalized_row_values(table[validated.source.columns.value].combine_chunks())
                expected_ids = pc.index_in(normalized, value_set=labels)
                if expected_ids.null_count or not np.array_equal(
                    np.asarray(expected_ids, dtype=np.uint32),
                    output_value_id[point_id_start:stop],
                ):
                    raise RuntimeError("Finalized Exact value IDs do not match canonical source rows.")
                point_id_start = stop
        finally:
            parquet_file.close()
    if point_id_start != row_count or maximum_error > tolerance:
        raise RuntimeError("Finalized Exact coordinates do not reconstruct the complete validated source.")

    del seen, reconstructed, output_value_id
    output = {
        "bucket_count": result.bucket_count,
        "tile_count": result.tile_count,
        "point_count": result.point_count,
        "range_count": result.range_count,
        "maximum_bucket_rows": maximum_bucket_rows,
        "maximum_tile_rows": maximum_tile_rows,
        "per_array_stored_bytes": per_array_bytes,
        "total_stored_bytes": total_stored_bytes,
        "logical_chunk_count": logical_chunks,
        "physical_chunk_or_shard_count": physical_shards,
        "filesystem_object_count": filesystem_objects,
        "point_ids_unique_complete_and_in_range": True,
        "per_value_totals_match": True,
        "coordinate_error": {
            "absolute_tolerance": tolerance,
            "maximum": maximum_error,
            "relative_to_tolerance_histogram_edges": histogram_edges.tolist(),
            "counts": histogram_counts.tolist(),
        },
        "value_tile_counts": value_tile_counts.tolist(),
    }
    return output, first_descriptor_by_value


def _time(operation: Callable[[], object], *, repeats: int = 3) -> list[float]:
    timings: list[float] = []
    for _ in range(repeats):
        start = perf_counter()
        operation()
        timings.append(perf_counter() - start)
    return timings


def _read_measurements(
    result: _LevelWriteResult,
    *,
    staging: Path,
    validated: ValidatedPointsSource,
    first_descriptor_by_value: dict[int, _TileDescriptor],
    value_tile_counts: list[int],
) -> dict[str, object]:
    counts = np.asarray(validated.value_table["n_points"].combine_chunks(), dtype=np.uint64)
    ordered = np.argsort(counts, kind="stable")
    rare_pool = ordered[: max(1, len(ordered) // 4)]
    categories = {
        "common": int(np.argmax(counts)),
        "median": int(ordered[len(ordered) // 2]),
        "rare_localized": int(min(rare_pool, key=lambda index: (value_tile_counts[index], counts[index]))),
        "rare_distributed": int(
            max(
                rare_pool,
                key=lambda index: (value_tile_counts[index], -int(counts[index])),
            )
        ),
    }
    largest = max(result.tile_descriptors, key=lambda descriptor: descriptor.n_points)
    measurements: dict[str, object] = {}
    with _BucketReader(staging, level=0, bucket_id=largest.bucket_id) as reader:
        measurements["largest_complete_tile"] = {
            "tile_x": largest.tile_x,
            "tile_y": largest.tile_y,
            "logical_rows": largest.n_points,
            "seconds": _time(lambda: reader.read_complete(largest)),
        }
    labels = validated.value_table["value"].to_pylist()
    for category, value_id in categories.items():
        descriptor = first_descriptor_by_value[value_id]
        selected = np.array([value_id], dtype=np.uint32)
        with _BucketReader(staging, level=0, bucket_id=descriptor.bucket_id) as reader:
            payload = reader.read_selected(descriptor, selected)
            if payload is None:
                raise RuntimeError("A selected-read measurement value is absent from its recorded tile.")
            timings = _time(
                lambda descriptor=descriptor, selected=selected: reader.read_selected(
                    descriptor,
                    selected,
                )
            )
        physical = _selected_read_physical_stats(
            staging,
            descriptor=descriptor,
            value_id=value_id,
        )
        measurements[category] = {
            "value_id": value_id,
            "value": labels[value_id],
            "source_point_count": int(counts[value_id]),
            "source_tile_count": value_tile_counts[value_id],
            "tile_x": descriptor.tile_x,
            "tile_y": descriptor.tile_y,
            "logical_rows": payload.n_points,
            **physical,
            "seconds": timings,
        }
    return measurements


def _selected_read_physical_stats(
    staging: Path,
    *,
    descriptor: _TileDescriptor,
    value_id: int,
) -> dict[str, int | float]:
    """Calculate inner-chunk decode amplification for one selected value run."""
    bucket_path = staging / descriptor.bucket_path
    store = LocalStore(bucket_path, read_only=True)
    try:
        root = zarr.open_group(store=store, mode="r", zarr_format=3, use_consolidated=False)
        tile_index = descriptor.bucket_tile_index
        indptr = np.asarray(root["ranges/tile_indptr"][tile_index : tile_index + 2], dtype=np.uint64)
        range_start, range_stop = (int(value) for value in indptr)
        values = np.asarray(root["ranges/value_id"][range_start:range_stop], dtype=np.uint32)
        position = int(np.searchsorted(values, np.uint32(value_id)))
        if position >= len(values) or int(values[position]) != value_id:
            raise RuntimeError("Selected-read range disappeared during evaluation.")
        row_start = int(root["ranges/row_start"][range_start + position])
        row_count = int(root["ranges/row_count"][range_start + position])
        point_array = root["value_id"]
        chunk_rows = point_array.chunks[0]
        first_chunk = row_start // chunk_rows
        last_chunk = (row_start + row_count - 1) // chunk_rows
        decoded_rows = sum(
            min((chunk_id + 1) * chunk_rows, point_array.shape[0]) - chunk_id * chunk_rows
            for chunk_id in range(first_chunk, last_chunk + 1)
        )
    finally:
        store.close()
    return {
        "point_chunks_touched": last_chunk - first_chunk + 1,
        "decoded_point_rows": decoded_rows,
        "decoded_row_amplification": decoded_rows / row_count,
    }


def main() -> None:
    """Run the opt-in full-Exact build, validation, and read characterization."""
    args = _parse_args()
    source = ParquetPointsSource(
        spatialdata_path=args.spatialdata_path,
        points_name=args.points_name,
        columns=PointColumnSelection(x=args.x, y=args.y, value=args.value),
    )
    validation_start = perf_counter()
    validated = validate_parquet_points_source(source)
    validation_seconds = perf_counter() - validation_start
    if args.expected_row_count > 0 and validated.row_count != args.expected_row_count:
        raise RuntimeError(f"Validated source contains {validated.row_count} rows; expected {args.expected_row_count}.")
    plan = _plan_points_cache(
        validated,
        leaf_tile_size=args.leaf_tile_size,
        overview_point_budget=args.overview_point_budget,
    )
    settings = _ZarrWriteSettings(
        point_chunk_rows=args.point_chunk_rows,
        point_shard_rows=args.point_shard_rows,
        range_chunk_rows=args.range_chunk_rows,
        range_shard_rows=args.range_shard_rows,
        codec_id=args.codec_id,
    )
    config = _ExactWriterConfig(settings, dask_worker_count=args.dask_worker_count)

    args.work_directory.mkdir(parents=True, exist_ok=True)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="napari-harpy-zarr-exact-evaluation-",
        dir=args.work_directory,
    ) as workspace_text:
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
                staging_root=staging,
                temporary_directory_root=shuffle,
                config=config,
            )
            build_seconds = perf_counter() - build_start
        verification_start = perf_counter()
        output, first_descriptor_by_value = _verify_and_summarize(
            result,
            staging=staging,
            validated=validated,
            plan=plan,
            workspace=workspace,
        )
        verification_seconds = perf_counter() - verification_start
        if list(shuffle.iterdir()) or list(staging.rglob("*.parquet")):
            raise RuntimeError("Exact evaluation found retained shuffle data or derived point Parquet.")
        reads = _read_measurements(
            result,
            staging=staging,
            validated=validated,
            first_descriptor_by_value=first_descriptor_by_value,
            value_tile_counts=output.pop("value_tile_counts"),
        )
        exact = plan.levels[0]
        report = {
            "schema_version": "harpy-zarr-exact-evaluation-v1",
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
                "largest_row_group_rows": max(
                    row_group.row_count for source_file in validated.files for row_group in source_file.row_groups
                ),
                "value_count": validated.value_table.num_rows,
                "source_signature": validated.source_signature,
            },
            "configuration": {
                "leaf_tile_size": exact.tile_size,
                "bucket_hash_method": BUCKET_HASH_METHOD,
                "target_points_per_bucket": TARGET_POINTS_PER_BUCKET,
                "planned_bucket_count": _bucket_count_for_level(exact),
                "point_chunk_rows": settings.point_chunk_rows,
                "point_shard_rows": settings.point_shard_rows,
                "range_chunk_rows": settings.range_chunk_rows,
                "range_shard_rows": settings.range_shard_rows,
                "codec_id": settings.codec_id,
                "dask_worker_count": config.dask_worker_count,
            },
            "timing_seconds": {
                "validation": validation_seconds,
                "exact_build": build_seconds,
                "independent_verification": verification_seconds,
            },
            "resources": {
                "rss_sample_interval_seconds": _RSS_SAMPLE_INTERVAL_SECONDS,
                "baseline_rss_bytes": resources.baseline_rss_bytes,
                "peak_rss_bytes": resources.peak_rss_bytes,
                "incremental_peak_rss_bytes": resources.peak_rss_bytes - resources.baseline_rss_bytes,
                "peak_workspace_bytes": resources.peak_workspace_bytes,
            },
            "output": output,
            "reads": reads,
        }
        serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
        args.json_output.write_text(serialized, encoding="utf-8")
        print(serialized, end="")


if __name__ == "__main__":
    main()
