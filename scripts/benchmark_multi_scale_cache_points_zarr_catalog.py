from __future__ import annotations

import argparse
import json
import math
import os
import platform
import shutil
import tempfile
import uuid
from pathlib import Path
from time import perf_counter

import numpy as np
import psutil
from benchmark_multi_scale_cache_points_zarr_exact import (
    _directory_file_count,
    _directory_size,
    _ResourceSampler,
)

from napari_harpy.core.multi_scale_cache_points_zarr.build_plan import (
    _plan_points_cache,
    _PointsCacheBuildPlan,
)
from napari_harpy.core.multi_scale_cache_points_zarr.cache_format import (
    _CatalogWriteSettings,
    _ValueMajorWriteSettings,
)
from napari_harpy.core.multi_scale_cache_points_zarr.models import _TileDescriptor
from napari_harpy.core.multi_scale_cache_points_zarr.source import (
    ParquetPointsSource,
    PointColumnSelection,
    ValidatedPointsSource,
    validate_parquet_points_source,
)
from napari_harpy.core.multi_scale_cache_points_zarr.storage._schema import (
    MANIFEST_BUCKET_ID,
    MANIFEST_BUCKET_TILE_INDEX,
    MANIFEST_LEVEL_INDPTR,
    VALUE_TILES_MANIFEST_INDEX,
    VALUE_TILES_N_POINTS,
)
from napari_harpy.core.multi_scale_cache_points_zarr.storage.catalog_reader import (
    _CatalogReader,
    _iter_bucket_range_batches,
)
from napari_harpy.core.multi_scale_cache_points_zarr.storage.models import (
    _BucketWriteResult,
    _LevelWriteResult,
    _ZarrWriteSettings,
)
from napari_harpy.core.multi_scale_cache_points_zarr.writer.bridge import (
    _BridgeWriterConfig,
    _write_bridge_level,
)
from napari_harpy.core.multi_scale_cache_points_zarr.writer.catalog import _write_staged_cache_catalog
from napari_harpy.core.multi_scale_cache_points_zarr.writer.exact import (
    _ExactWriterConfig,
    _write_exact_level,
)
from napari_harpy.core.multi_scale_cache_points_zarr.writer.spatial import (
    _SpatialWriterConfig,
    _write_spatial_levels,
)

_EXPECTED_XENIUM_POINT_COUNT = 136_578_750
_PYRAMID_INVENTORY_SCHEMA_VERSION = "harpy-zarr-benchmark-pyramid-inventory-v1"
_PYRAMID_INVENTORY_NAME = "_benchmark_pyramid_inventory.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and evaluate the full Zarr catalog once.")
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
    parser.add_argument("--evaluation-name", default="z6-evaluated")
    parser.add_argument("--json-output", type=Path, required=True)
    return parser.parse_args()


def _plan_inventory(plan: _PointsCacheBuildPlan) -> dict[str, object]:
    return {
        "x_origin": plan.x_origin,
        "y_origin": plan.y_origin,
        "leaf_tile_size": plan.leaf_tile_size,
        "overview_point_budget": plan.overview_point_budget,
        "levels": [
            {
                "level": level.level,
                "kind": level.kind.value,
                "tile_size": level.tile_size,
                "grid_width": level.grid_width,
                "grid_height": level.grid_height,
                "max_points_per_tile": level.max_points_per_tile,
                "point_count_upper_bound": level.point_count_upper_bound,
            }
            for level in plan.levels
        ],
    }


def _settings_inventory(settings: _ZarrWriteSettings) -> dict[str, object]:
    return {
        "point_chunk_rows": settings.point_chunk_rows,
        "point_shard_rows": settings.point_shard_rows,
        "range_chunk_rows": settings.range_chunk_rows,
        "range_shard_rows": settings.range_shard_rows,
        "codec_id": settings.codec_id,
    }


def _level_results_inventory(level_results: tuple[_LevelWriteResult, ...]) -> list[dict[str, object]]:
    return [
        {
            "level": result.level,
            "buckets": [
                {
                    "bucket_id": bucket.bucket_id,
                    "point_count": bucket.point_count,
                    "range_count": bucket.range_count,
                    "tiles": [
                        {
                            "bucket_tile_index": tile.bucket_tile_index,
                            "tile_x": tile.tile_x,
                            "tile_y": tile.tile_y,
                            "n_points": tile.n_points,
                        }
                        for tile in bucket.tile_descriptors
                    ],
                }
                for bucket in result.buckets
            ],
        }
        for result in level_results
    ]


def _write_pyramid_inventory(
    pyramid_root: Path,
    *,
    source_signature: str,
    plan: _PointsCacheBuildPlan,
    zarr_settings: _ZarrWriteSettings,
    level_results: tuple[_LevelWriteResult, ...],
) -> None:
    payload = {
        "schema_version": _PYRAMID_INVENTORY_SCHEMA_VERSION,
        "source_signature": source_signature,
        "plan": _plan_inventory(plan),
        "zarr_settings": _settings_inventory(zarr_settings),
        "level_results": _level_results_inventory(level_results),
    }
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    (pyramid_root / _PYRAMID_INVENTORY_NAME).write_text(serialized, encoding="utf-8")


def _read_pyramid_inventory(
    pyramid_root: Path,
    *,
    source_signature: str,
    plan: _PointsCacheBuildPlan,
    zarr_settings: _ZarrWriteSettings,
) -> tuple[_LevelWriteResult, ...]:
    inventory_path = pyramid_root / _PYRAMID_INVENTORY_NAME
    try:
        payload = json.loads(inventory_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Cannot read reusable pyramid inventory: {inventory_path}") from error
    if not isinstance(payload, dict) or payload.get("schema_version") != _PYRAMID_INVENTORY_SCHEMA_VERSION:
        raise RuntimeError("Reusable pyramid inventory has an unsupported schema version.")
    if payload.get("source_signature") != source_signature:
        raise RuntimeError("Reusable pyramid belongs to a different validated source signature.")
    if payload.get("plan") != _plan_inventory(plan):
        raise RuntimeError("Reusable pyramid does not match the current logical build plan.")
    if payload.get("zarr_settings") != _settings_inventory(zarr_settings):
        raise RuntimeError("Reusable pyramid does not match the requested Zarr storage settings.")

    raw_levels = payload.get("level_results")
    if not isinstance(raw_levels, list):
        raise RuntimeError("Reusable pyramid inventory has invalid level results.")
    try:
        level_results = tuple(
            _LevelWriteResult(
                tuple(
                    _BucketWriteResult(
                        tuple(
                            _TileDescriptor(
                                level=raw_level["level"],
                                bucket_id=raw_bucket["bucket_id"],
                                bucket_tile_index=raw_tile["bucket_tile_index"],
                                tile_x=raw_tile["tile_x"],
                                tile_y=raw_tile["tile_y"],
                                n_points=raw_tile["n_points"],
                            )
                            for raw_tile in raw_bucket["tiles"]
                        ),
                        point_count=raw_bucket["point_count"],
                        range_count=raw_bucket["range_count"],
                    )
                    for raw_bucket in raw_level["buckets"]
                )
            )
            for raw_level in raw_levels
        )
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("Reusable pyramid inventory contains invalid bucket results.") from error

    expected_paths = {bucket.bucket_path for result in level_results for bucket in result.buckets}
    observed_paths = {
        str(path.relative_to(pyramid_root))
        for path in (pyramid_root / "tile_major").glob("level_*/bucket-*.zarr")
        if path.is_dir()
    }
    if observed_paths != expected_paths:
        raise RuntimeError("Reusable pyramid bucket paths do not match its inventory.")
    return level_results


def _build_or_reuse_pyramid(
    validated: ValidatedPointsSource,
    plan: _PointsCacheBuildPlan,
    *,
    workspace: Path,
    temporary: Path,
    zarr_settings: _ZarrWriteSettings,
    dask_worker_count: int,
) -> tuple[Path, tuple[_LevelWriteResult, ...], dict[str, float | None], bool]:
    pyramid_root = workspace / "pyramid-base"
    if pyramid_root.exists():
        print(f"Reusing persistent prerequisite pyramid: {pyramid_root}", flush=True)
        level_results = _read_pyramid_inventory(
            pyramid_root,
            source_signature=validated.source_signature,
            plan=plan,
            zarr_settings=zarr_settings,
        )
        return pyramid_root, level_results, {"exact": None, "bridge": None, "spatial": None}, True

    with tempfile.TemporaryDirectory(prefix=".pyramid-build-", dir=workspace) as build_text:
        staging = Path(build_text) / "pyramid-base"
        staging.mkdir()

        print("Building Exact prerequisite...", flush=True)
        started = perf_counter()
        exact = _write_exact_level(
            validated,
            plan,
            staging_root=staging,
            temporary_directory_root=temporary,
            config=_ExactWriterConfig(zarr_settings, dask_worker_count=dask_worker_count),
        )
        exact_seconds = perf_counter() - started

        print("Building Bridge prerequisite...", flush=True)
        started = perf_counter()
        bridge = _write_bridge_level(
            exact,
            plan,
            staging_root=staging,
            config=_BridgeWriterConfig(zarr_settings),
        )
        bridge_seconds = perf_counter() - started

        print("Building Spatial prerequisites...", flush=True)
        started = perf_counter()
        spatial = _write_spatial_levels(
            bridge,
            plan,
            staging_root=staging,
            config=_SpatialWriterConfig(zarr_settings),
        )
        spatial_seconds = perf_counter() - started
        level_results = (exact, bridge, *spatial)
        if len(level_results) != len(plan.levels):
            raise RuntimeError("Completed level results do not match the plan.")
        _write_pyramid_inventory(
            staging,
            source_signature=validated.source_signature,
            plan=plan,
            zarr_settings=zarr_settings,
            level_results=level_results,
        )
        staging.replace(pyramid_root)

    return (
        pyramid_root,
        level_results,
        {"exact": exact_seconds, "bridge": bridge_seconds, "spatial": spatial_seconds},
        False,
    )


def _hardlink_pyramid_tile_major(pyramid_root: Path, evaluation_root: Path) -> None:
    """Clone immutable bucket files cheaply while keeping catalog metadata private."""
    evaluation_root.mkdir()
    try:
        shutil.copytree(pyramid_root / "tile_major", evaluation_root / "tile_major", copy_function=os.link)
    except Exception:
        shutil.rmtree(evaluation_root)
        raise


def _bucket_snapshots(staging: Path, results: tuple[_LevelWriteResult, ...]) -> dict[str, tuple[int, int]]:
    return {
        bucket.bucket_path: (
            _directory_size(staging / bucket.bucket_path),
            _directory_file_count(staging / bucket.bucket_path),
        )
        for result in results
        for bucket in result.buckets
    }


def _catalog_storage(staging: Path, reader: _CatalogReader) -> dict[str, object]:
    groups = ("values", "manifest", "value_tiles", "value_major")
    per_group_bytes = {group: _directory_size(staging / group) for group in groups}
    per_group_objects = {group: _directory_file_count(staging / group) for group in groups}
    array_paths = (
        "values/n_points",
        "manifest/level_indptr",
        "manifest/bucket_id",
        "manifest/bucket_tile_index",
        "manifest/tile_x",
        "manifest/tile_y",
        "manifest/n_points",
        "value_tiles/indptr",
        "value_tiles/manifest_index",
        "value_tiles/n_points",
        *(
            path
            for level in range(len(reader.attributes.levels))
            for path in (
                f"value_major/level_{level}/location",
                f"value_major/level_{level}/value_point_indptr",
            )
        ),
    )
    array_layouts: dict[str, object] = {}
    for name in array_paths:
        array = reader.array(name)
        path = staging.joinpath(*name.split("/"))
        outer = array.shards if array.shards is not None else array.chunks
        array_layouts[name] = {
            "shape": array.shape,
            "chunks": array.chunks,
            "shards": array.shards,
            "logical_chunk_count": math.prod(
                math.ceil(size / chunk) for size, chunk in zip(array.shape, array.chunks, strict=True)
            ),
            "physical_object_count": math.prod(
                math.ceil(size / shard) for size, shard in zip(array.shape, outer, strict=True)
            ),
            "stored_bytes": _directory_size(path),
            "filesystem_objects": _directory_file_count(path),
        }
    return {
        "per_group_stored_bytes": per_group_bytes,
        "per_group_filesystem_objects": per_group_objects,
        "arrays": array_layouts,
        "catalog_stored_bytes": sum(per_group_bytes.values()),
        "catalog_filesystem_objects": sum(per_group_objects.values()),
        "root_zarr_json_bytes": (staging / "zarr.json").stat().st_size,
    }


def _verify_representative_bucket_indexes(
    staging: Path,
    results: tuple[_LevelWriteResult, ...],
    reader: _CatalogReader,
    *,
    zarr_settings: _ZarrWriteSettings,
    batch_rows: int,
) -> dict[str, int]:
    level_indptr = np.asarray(reader.array(MANIFEST_LEVEL_INDPTR)[:], dtype=np.uint64)
    bucket_ids = np.asarray(reader.array(MANIFEST_BUCKET_ID)[:], dtype=np.uint32)
    bucket_indexes = np.asarray(reader.array(MANIFEST_BUCKET_TILE_INDEX)[:], dtype=np.uint32)
    address_to_manifest = {
        (level, int(bucket_ids[row]), int(bucket_indexes[row])): row
        for level in range(len(results))
        for row in range(int(level_indptr[level]), int(level_indptr[level + 1]))
    }
    sampled_buckets = tuple(max(result.buckets, key=lambda bucket: bucket.range_count) for result in results)
    selected_manifest = np.zeros(len(bucket_ids), dtype=np.bool_)
    expected_parts: dict[str, list[np.ndarray]] = {
        "level": [],
        "value_id": [],
        "manifest_index": [],
        "n_points": [],
    }
    for bucket in sampled_buckets:
        manifest_indexes = np.fromiter(
            (
                address_to_manifest[(bucket.level, bucket.bucket_id, descriptor.bucket_tile_index)]
                for descriptor in bucket.tile_descriptors
            ),
            dtype=np.uint64,
            count=len(bucket.tile_descriptors),
        )
        selected_manifest[manifest_indexes] = True
        for batch in _iter_bucket_range_batches(
            staging,
            bucket,
            np.ascontiguousarray(manifest_indexes),
            batch_rows=batch_rows,
            expected_settings=zarr_settings,
        ):
            # Level identity belongs to the sampled bucket stream and is no
            # longer duplicated in every production range-record batch.
            expected_parts["level"].append(np.full(len(batch.value_id), bucket.level, dtype=np.uint16))
            for name in ("value_id", "manifest_index", "n_points"):
                expected_parts[name].append(getattr(batch, name))

    indptr = np.asarray(reader.array("value_tiles/indptr")[:], dtype=np.uint64)
    value_count = reader.attributes.catalog.value_count
    flat_indptr = np.empty(len(results) * value_count + 1, dtype=np.uint64)
    for level in range(len(results)):
        start = level * value_count
        flat_indptr[start : start + value_count + 1] = indptr[level]
    observed_parts: dict[str, list[np.ndarray]] = {
        "level": [],
        "value_id": [],
        "manifest_index": [],
        "n_points": [],
    }
    row_count = reader.attributes.catalog.value_tile_row_count
    for start in range(0, row_count, batch_rows):
        stop = min(start + batch_rows, row_count)
        manifests = np.asarray(reader.array(VALUE_TILES_MANIFEST_INDEX)[start:stop], dtype=np.uint64)
        selected = selected_manifest[manifests]
        if not bool(selected.any()):
            continue
        positions = np.arange(start, stop, dtype=np.uint64)
        flat_keys = np.searchsorted(flat_indptr, positions, side="right") - 1
        counts = np.asarray(reader.array(VALUE_TILES_N_POINTS)[start:stop], dtype=np.uint64)
        observed_parts["level"].append(np.ascontiguousarray(flat_keys[selected] // value_count, dtype=np.uint16))
        observed_parts["value_id"].append(np.ascontiguousarray(flat_keys[selected] % value_count, dtype=np.uint32))
        observed_parts["manifest_index"].append(np.ascontiguousarray(manifests[selected]))
        observed_parts["n_points"].append(np.ascontiguousarray(counts[selected]))

    expected = {name: np.ascontiguousarray(np.concatenate(parts)) for name, parts in expected_parts.items()}
    observed = {name: np.ascontiguousarray(np.concatenate(parts)) for name, parts in observed_parts.items()}
    expected_order = np.lexsort((expected["manifest_index"], expected["value_id"], expected["level"]))
    observed_order = np.lexsort((observed["manifest_index"], observed["value_id"], observed["level"]))
    if any(not np.array_equal(expected[name][expected_order], observed[name][observed_order]) for name in expected):
        raise RuntimeError("Representative bucket compact indexes differ from value_tiles.")
    return {
        "sampled_bucket_count": len(sampled_buckets),
        "sampled_range_count": len(expected["level"]),
        "sampled_manifest_tile_count": int(selected_manifest.sum()),
    }


def main() -> None:
    """Run one full current-tree Z6 engineering gate."""
    args = _parse_args()
    source = ParquetPointsSource(
        spatialdata_path=args.spatialdata_path,
        points_name=args.points_name,
        columns=PointColumnSelection(x=args.x, y=args.y, value=args.value),
    )
    print("Validating canonical Xenium source...", flush=True)
    started = perf_counter()
    validated = validate_parquet_points_source(source)
    validation_seconds = perf_counter() - started
    if validated.row_count != args.expected_row_count:
        raise RuntimeError(f"Validated {validated.row_count} rows; expected {args.expected_row_count}.")
    plan = _plan_points_cache(
        validated,
        leaf_tile_size=args.leaf_tile_size,
        overview_point_budget=args.overview_point_budget,
    )
    zarr_settings = _ZarrWriteSettings(
        point_chunk_rows=args.point_chunk_rows,
        point_shard_rows=args.point_shard_rows,
        range_chunk_rows=args.range_chunk_rows,
        range_shard_rows=args.range_shard_rows,
        codec_id=args.codec_id,
    )
    catalog_settings = _CatalogWriteSettings()
    value_major_settings = _ValueMajorWriteSettings()
    max_open_value_major_readers = None

    if args.evaluation_name in {"", ".", ".."} or Path(args.evaluation_name).name != args.evaluation_name:
        raise ValueError("`evaluation-name` must be one nonempty directory name.")
    workspace = args.work_directory
    evaluation_root = workspace / args.evaluation_name
    if evaluation_root.exists():
        raise FileExistsError(f"Gate Z6 evaluation already exists: {evaluation_root}")
    if args.json_output.exists():
        raise FileExistsError(f"Gate Z6 report already exists: {args.json_output}")

    workspace.mkdir(parents=True, exist_ok=True)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".z6-scratch-", dir=workspace) as temporary_text:
        temporary = Path(temporary_text)
        pyramid_root, level_results, prerequisite_seconds, prerequisites_reused = _build_or_reuse_pyramid(
            validated,
            plan,
            workspace=workspace,
            temporary=temporary,
            zarr_settings=zarr_settings,
            dask_worker_count=args.dask_worker_count,
        )
        pyramid_snapshot = _bucket_snapshots(pyramid_root, level_results)

        print(f"Creating reusable evaluation generation: {evaluation_root}", flush=True)
        started = perf_counter()
        _hardlink_pyramid_tile_major(pyramid_root, evaluation_root)
        prerequisite_clone_seconds = perf_counter() - started

        bucket_snapshot = _bucket_snapshots(evaluation_root, level_results)
        source_snapshot = (_directory_size(source.parquet_path), _directory_file_count(source.parquet_path))
        generation_id = str(uuid.uuid4())
        value_tile_rows = sum(result.range_count for result in level_results)
        largest_level_sort_rows = max(result.range_count for result in level_results)

        print(f"Building Z6 catalog for {value_tile_rows:,} value-tile rows...", flush=True)
        with _ResourceSampler(evaluation_root) as resources:
            started = perf_counter()
            _write_staged_cache_catalog(
                validated,
                plan,
                level_results,
                staging_root=evaluation_root,
                cache_generation_id=generation_id,
                settings=catalog_settings,
                value_major_settings=value_major_settings,
                max_open_value_major_readers=max_open_value_major_readers,
                temporary_directory_root=temporary,
            )
            catalog_seconds = perf_counter() - started

        print("Reopening and streaming strict catalog validation...", flush=True)
        started = perf_counter()
        with _CatalogReader(evaluation_root) as reader:
            reader.validate_contents()
            representative_verification = _verify_representative_bucket_indexes(
                evaluation_root,
                level_results,
                reader,
                zarr_settings=zarr_settings,
                batch_rows=catalog_settings.value_tile_chunk_rows,
            )
            observed_manifest_rows = reader.attributes.catalog.manifest_row_count
            observed_value_tile_rows = reader.attributes.catalog.value_tile_row_count
            observed_value_count = reader.attributes.catalog.value_count
            manifest_index_shape = reader.array(VALUE_TILES_MANIFEST_INDEX).shape
            value_tile_count_shape = reader.array(VALUE_TILES_N_POINTS).shape
            catalog_storage = _catalog_storage(evaluation_root, reader)
        catalog_validation_seconds = perf_counter() - started

        if bucket_snapshot != _bucket_snapshots(evaluation_root, level_results):
            raise RuntimeError("Catalog construction modified an existing bucket store.")
        if pyramid_snapshot != _bucket_snapshots(pyramid_root, level_results):
            raise RuntimeError("Catalog construction modified the reusable prerequisite pyramid.")
        if source_snapshot != (_directory_size(source.parquet_path), _directory_file_count(source.parquet_path)):
            raise RuntimeError("Catalog construction modified the canonical source.")
        if list(temporary.iterdir()):
            raise RuntimeError("Catalog construction retained unexpected scratch data.")
        standalone_json = [path for path in evaluation_root.rglob("*.json") if path.name != "zarr.json"]
        if list(evaluation_root.rglob("*.parquet")) or standalone_json or (evaluation_root / "COMPLETED").exists():
            raise RuntimeError("Z6 wrote forbidden Parquet, JSON-sidecar, or completion artifacts.")

        report = {
            "schema_version": "harpy-zarr-catalog-evaluation-v1",
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
                "value_count": validated.value_table.num_rows,
                "source_signature": validated.source_signature,
            },
            "configuration": {
                **catalog_settings.__dict__,
                "max_open_value_major_readers": max_open_value_major_readers,
                "value_major": value_major_settings.__dict__,
                "point_chunk_rows": zarr_settings.point_chunk_rows,
                "point_shard_rows": zarr_settings.point_shard_rows,
                "range_chunk_rows": zarr_settings.range_chunk_rows,
                "range_shard_rows": zarr_settings.range_shard_rows,
                "codec_id": zarr_settings.codec_id,
            },
            "artifacts": {
                "workspace": str(workspace),
                "pyramid_root": str(pyramid_root),
                "evaluation_root": str(evaluation_root),
                "pyramid_inventory": str(pyramid_root / _PYRAMID_INVENTORY_NAME),
                "prerequisites_reused": prerequisites_reused,
                "evaluation_level_copy_method": "hardlink",
            },
            "timing_seconds": {
                "source_validation": validation_seconds,
                "prerequisite_exact": prerequisite_seconds["exact"],
                "prerequisite_bridge": prerequisite_seconds["bridge"],
                "prerequisite_spatial": prerequisite_seconds["spatial"],
                "prerequisite_level_hardlinking": prerequisite_clone_seconds,
                "catalog_construction": catalog_seconds,
                "catalog_strict_validation": catalog_validation_seconds,
            },
            "resources": {
                "baseline_rss_bytes": resources.baseline_rss_bytes,
                "peak_rss_bytes": resources.peak_rss_bytes,
                "incremental_peak_rss_bytes": resources.peak_rss_bytes - resources.baseline_rss_bytes,
                "peak_workspace_bytes": resources.peak_workspace_bytes,
            },
            "catalog": {
                "level_count": len(level_results),
                "value_count": observed_value_count,
                "manifest_row_count": observed_manifest_rows,
                "value_tile_row_count": observed_value_tile_rows,
                "value_tile_manifest_index_shape": manifest_index_shape,
                "value_tile_n_points_shape": value_tile_count_shape,
                "largest_level_sort_rows": largest_level_sort_rows,
                "estimated_largest_level_input_bytes": largest_level_sort_rows * (4 + 8 + 8 + 8),
                "estimated_largest_level_order_bytes": largest_level_sort_rows * 8,
                **representative_verification,
                **catalog_storage,
            },
            "levels": [
                {
                    "level": result.level,
                    "bucket_count": result.bucket_count,
                    "tile_count": result.tile_count,
                    "point_count": result.point_count,
                    "range_count": result.range_count,
                }
                for result in level_results
            ],
            "verification": {
                "strict_catalog_reader_passed": True,
                "representative_bucket_indexes_match": True,
                "point_payload_arrays_not_read": True,
                "canonical_source_data_pages_not_read": True,
                "bucket_stores_unchanged": True,
                "reusable_pyramid_unchanged": True,
                "canonical_source_unchanged": True,
                "catalog_sort_scratch_absent": True,
                "derived_parquet_absent": True,
                "standalone_json_sidecar_absent": True,
                "completion_marker_absent": True,
            },
        }
        serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
        args.json_output.write_text(serialized, encoding="utf-8")
        print(serialized, end="", flush=True)


if __name__ == "__main__":
    main()
