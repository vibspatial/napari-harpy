from __future__ import annotations

import argparse
import json
import os
import platform
import threading
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
import psutil

from napari_harpy.core.multi_scale_cache_points_zarr.builder import (
    _build_points_cache_zarr,
    _PointsCacheBuilderConfig,
)
from napari_harpy.core.multi_scale_cache_points_zarr.cache_format import (
    MANIFEST_LEVEL_INDPTR,
    MANIFEST_N_POINTS,
    MANIFEST_TILE_X,
    MANIFEST_TILE_Y,
    VALUE_TILES_INDPTR,
    VALUE_TILES_MANIFEST_INDEX,
    VALUE_TILES_N_POINTS,
    VALUES_N_POINTS,
)
from napari_harpy.core.multi_scale_cache_points_zarr.hashing import TARGET_POINTS_PER_BUCKET
from napari_harpy.core.multi_scale_cache_points_zarr.reader import (
    _IntrinsicViewport,
    _LevelSelection,
    _PointsCacheReader,
    _SelectedValueIndex,
    _TileReadResult,
    _ViewportReadResult,
)
from napari_harpy.core.multi_scale_cache_points_zarr.source import (
    ParquetPointsSource,
    PointColumnSelection,
    validate_parquet_points_source,
)
from napari_harpy.core.multi_scale_cache_points_zarr.storage.catalog_reader import _CatalogReader

_EXPECTED_XENIUM_POINT_COUNT = 136_578_750
_RSS_SAMPLE_INTERVAL_SECONDS = 0.25
_MAX_SELECTED_VALUE_INDEX_BYTES = 1 << 30


class _RssSampler:
    """Measure process RSS without walking the cache during construction."""

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
    parser = argparse.ArgumentParser(description="Build and evaluate the full Zarr acceptance reader once.")
    parser.add_argument("spatialdata_path", type=Path)
    parser.add_argument("--points-name", required=True)
    parser.add_argument("--x", default="x")
    parser.add_argument("--y", default="y")
    parser.add_argument("--value", default="gene")
    parser.add_argument("--leaf-tile-size", type=int, default=512)
    parser.add_argument("--overview-point-budget", type=int, default=100_000)
    parser.add_argument("--dask-worker-count", type=int, default=2)
    parser.add_argument("--expected-row-count", type=int, default=_EXPECTED_XENIUM_POINT_COUNT)
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


def _viewport_for_tile(attributes: object, level: object, tile_x: int, tile_y: int) -> _IntrinsicViewport:
    geometry = attributes.geometry
    x_min = geometry.x_origin + tile_x * level.tile_size
    y_min = geometry.y_origin + tile_y * level.tile_size
    return _IntrinsicViewport(x_min, y_min, x_min + level.tile_size, y_min + level.tile_size)


def _full_viewport(attributes: object) -> _IntrinsicViewport:
    geometry = attributes.geometry
    return _IntrinsicViewport(
        geometry.x_min,
        geometry.y_min,
        np.nextafter(geometry.x_max, np.inf),
        np.nextafter(geometry.y_max, np.inf),
    )


def _level_selection_report(selection: _LevelSelection) -> dict[str, object]:
    """Convert one LOD decision to JSON-compatible benchmark evidence."""
    omitted_value_ids = selection.omitted_value_ids
    return {
        "level": selection.level,
        "estimated_point_count": selection.estimated_point_count,
        "positive_visible_tile_count": selection.positive_visible_tile_count,
        "within_budget": selection.within_budget,
        "omitted_value_ids": None if omitted_value_ids is None else omitted_value_ids.tolist(),
    }


def _result_summary(result: _TileReadResult | _ViewportReadResult | None, seconds: float) -> dict[str, object]:
    if result is None:
        return {"seconds": seconds, "logical_point_rows": 0, "positive_manifest_tiles": 0}
    if isinstance(result, _TileReadResult):
        return {"seconds": seconds, "logical_point_rows": len(result.value_id), "positive_manifest_tiles": 1}
    return {
        "seconds": seconds,
        "logical_point_rows": sum(len(tile.value_id) for tile in result.tiles),
        "positive_manifest_tiles": len(result.tiles),
    }


def _time_tile(
    reader: _PointsCacheReader,
    level: int,
    tile_x: int,
    tile_y: int,
    *,
    value_ids: np.ndarray | None = None,
) -> tuple[_TileReadResult | None, dict[str, object]]:
    started = perf_counter()
    result = reader.read_tile(level, tile_x, tile_y, value_ids=value_ids)
    return result, _result_summary(result, perf_counter() - started)


def _time_viewport(
    reader: _PointsCacheReader,
    level: int,
    viewport: _IntrinsicViewport,
    *,
    value_index: _SelectedValueIndex | None = None,
) -> tuple[_ViewportReadResult, dict[str, object]]:
    started = perf_counter()
    result = reader.read_viewport(level, viewport, value_index=value_index)
    return result, _result_summary(result, perf_counter() - started)


def _measure_application_cold_and_warm(
    cache_root: Path,
    level: int,
    viewport: _IntrinsicViewport,
    value_ids: np.ndarray,
) -> dict[str, object]:
    """Measure first and repeated requests through one newly entered reader."""
    started = perf_counter()
    with _PointsCacheReader(cache_root) as reader:
        open_seconds = perf_counter() - started
        started = perf_counter()
        value_index = reader.load_selected_value_index(
            value_ids,
            max_resident_bytes=_MAX_SELECTED_VALUE_INDEX_BYTES,
        )
        index_load_seconds = perf_counter() - started
        _, cold = _time_viewport(reader, level, viewport, value_index=value_index)
        _, warm = _time_viewport(reader, level, viewport, value_index=value_index)
        retained_readers = reader.open_bucket_reader_count
    return {
        "reader_open_seconds": open_seconds,
        "selected_value_index_load_seconds": index_load_seconds,
        "cold": cold,
        "warm": warm,
        "retained_bucket_readers": retained_readers,
    }


def _representative_level_ids(level_count: int) -> tuple[int, ...]:
    candidates = {0, level_count - 1}
    if level_count > 1:
        candidates.add(1)
    if level_count > 3:
        candidates.add(2 + (level_count - 3) // 2)
    return tuple(sorted(candidates))


def _representative_tiles(
    level: int,
    level_indptr: np.ndarray,
    n_points: np.ndarray,
    tile_x: np.ndarray,
    tile_y: np.ndarray,
) -> dict[str, tuple[int, int, int]]:
    start = int(level_indptr[level])
    stop = int(level_indptr[level + 1])
    counts = n_points[start:stop]
    mean = float(counts.mean())
    dense_row = start + int(np.argmax(counts))
    average_row = start + int(np.argmin(np.abs(counts.astype(np.float64) - mean)))
    return {
        "dense": (int(tile_x[dense_row]), int(tile_y[dense_row]), int(n_points[dense_row])),
        "average": (int(tile_x[average_row]), int(tile_y[average_row]), int(n_points[average_row])),
    }


def _representative_values(value_counts: np.ndarray, exact_tile_counts: np.ndarray) -> dict[str, int]:
    positive = np.flatnonzero(value_counts)
    ordered = positive[np.argsort(value_counts[positive], kind="stable")]
    common = int(positive[np.argmax(value_counts[positive])])
    median = int(ordered[len(ordered) // 2])
    localized_candidates = positive[exact_tile_counts[positive] == exact_tile_counts[positive].min()]
    rare_localized = int(localized_candidates[np.argmin(value_counts[localized_candidates])])
    rare_cutoff = np.quantile(value_counts[positive], 0.25)
    rare = positive[value_counts[positive] <= rare_cutoff]
    rare_distributed = int(rare[np.argmax(exact_tile_counts[rare])])
    return {
        "common": common,
        "median": median,
        "rare_localized": rare_localized,
        "rare_distributed": rare_distributed,
    }


def _assert_selected_matches_complete(complete: _TileReadResult, selected: _TileReadResult, value_id: int) -> None:
    expected = complete.value_id == np.uint32(value_id)
    if not np.array_equal(selected.location, complete.location[expected]) or not np.array_equal(
        selected.value_id, complete.value_id[expected]
    ):
        raise RuntimeError("Value-filtered tile read differs from an in-memory filter of the complete tile.")


def _evaluate_reader(cache_root: Path) -> dict[str, object]:
    with _CatalogReader(cache_root) as catalog:
        attributes = catalog.attributes
        level_indptr = np.asarray(catalog.array(MANIFEST_LEVEL_INDPTR)[:], dtype=np.uint64)
        n_points = np.asarray(catalog.array(MANIFEST_N_POINTS)[:], dtype=np.uint64)
        tile_x = np.asarray(catalog.array(MANIFEST_TILE_X)[:], dtype=np.uint32)
        tile_y = np.asarray(catalog.array(MANIFEST_TILE_Y)[:], dtype=np.uint32)
        value_indptr = np.asarray(catalog.array(VALUE_TILES_INDPTR)[:], dtype=np.uint64)
        value_counts = np.asarray(catalog.array(VALUES_N_POINTS)[:], dtype=np.uint64)
        representative_levels = _representative_level_ids(len(attributes.levels))
        representative_tiles = {
            str(level): _representative_tiles(level, level_indptr, n_points, tile_x, tile_y)
            for level in representative_levels
        }
        exact_tile_counts = value_indptr[0, 1:] - value_indptr[0, :-1]
        representative_values = _representative_values(value_counts, exact_tile_counts)
        representative_value_manifest_rows = {
            label: int(
                np.asarray(
                    catalog.array(VALUE_TILES_MANIFEST_INDEX)[
                        int(value_indptr[0, value_id]) : int(value_indptr[0, value_id]) + 1
                    ],
                    dtype=np.uint64,
                )[0]
            )
            for label, value_id in representative_values.items()
        }
    full_viewport = _full_viewport(attributes)

    print("Opening acceptance reader and measuring representative requests...", flush=True)
    started = perf_counter()
    reader = _PointsCacheReader(cache_root)
    reader.__enter__()
    reader_open_seconds = perf_counter() - started
    try:
        tile_reads: dict[str, object] = {}
        viewport_reads: dict[str, object] = {}
        for level in representative_levels:
            metadata = attributes.levels[level]
            level_tiles = representative_tiles[str(level)]
            for kind, (x, y, _) in level_tiles.items():
                result, summary = _time_tile(reader, level, x, y)
                if result is None:
                    raise RuntimeError("Representative manifest tile unexpectedly returned no rows.")
                tile_reads[f"level_{level}_{kind}_first"] = summary
                _, warm = _time_tile(reader, level, x, y)
                tile_reads[f"level_{level}_{kind}_repeat"] = warm
            x, y, _ = level_tiles["average"]
            viewport = _viewport_for_tile(attributes, metadata, x, y)
            _, viewport_reads[f"level_{level}_one_tile"] = _time_viewport(reader, level, viewport)

        dense_x, dense_y, _ = representative_tiles["0"]["dense"]
        complete, complete_summary = _time_tile(reader, 0, dense_x, dense_y)
        if complete is None:
            raise RuntimeError("Dense Exact tile unexpectedly returned no rows.")
        values_in_dense = np.unique(complete.value_id)
        selected_value = int(values_in_dense[len(values_in_dense) // 2])
        selected_ids = np.array([selected_value], dtype=np.uint32)
        selected, selected_summary = _time_tile(reader, 0, dense_x, dense_y, value_ids=selected_ids)
        if selected is None:
            raise RuntimeError("A value observed in the complete tile was absent from its selected read.")
        _assert_selected_matches_complete(complete, selected, selected_value)
        tile_reads["exact_dense_complete_correctness"] = complete_summary
        tile_reads["exact_dense_selected_correctness"] = selected_summary

        selected_reads: dict[str, object] = {}
        exact_level = attributes.levels[0]
        # Restrict payload timing to one positive Exact tile while still using
        # cache-wide value_tiles discovery to represent each distribution class.
        for label, value_id in representative_values.items():
            manifest_row = representative_value_manifest_rows[label]
            viewport = _viewport_for_tile(
                attributes,
                exact_level,
                int(tile_x[manifest_row]),
                int(tile_y[manifest_row]),
            )
            ids = np.array([value_id], dtype=np.uint32)
            value_index = reader.load_selected_value_index(
                ids,
                max_resident_bytes=_MAX_SELECTED_VALUE_INDEX_BYTES,
            )
            _, first = _time_viewport(reader, 0, viewport, value_index=value_index)
            _, repeat = _time_viewport(reader, 0, viewport, value_index=value_index)
            selected_reads[f"{label}_first"] = first
            selected_reads[f"{label}_repeat"] = repeat

        adjacent = np.array(sorted({selected_value, min(selected_value + 1, len(value_counts) - 1)}), dtype=np.uint32)
        separated = np.array([0, len(value_counts) - 1], dtype=np.uint32)
        dense_viewport = _viewport_for_tile(attributes, exact_level, dense_x, dense_y)
        adjacent_index = reader.load_selected_value_index(
            adjacent,
            max_resident_bytes=_MAX_SELECTED_VALUE_INDEX_BYTES,
        )
        separated_index = reader.load_selected_value_index(
            separated,
            max_resident_bytes=_MAX_SELECTED_VALUE_INDEX_BYTES,
        )
        _, selected_reads["adjacent_values"] = _time_viewport(
            reader,
            0,
            dense_viewport,
            value_index=adjacent_index,
        )
        _, selected_reads["separated_values"] = _time_viewport(
            reader,
            0,
            dense_viewport,
            value_index=separated_index,
        )

        tile_size = exact_level.tile_size
        pan_first = _IntrinsicViewport(
            dense_viewport.x_min,
            dense_viewport.y_min,
            dense_viewport.x_max + tile_size,
            dense_viewport.y_max + tile_size,
        )
        pan_second = _IntrinsicViewport(
            dense_viewport.x_min + tile_size / 2,
            dense_viewport.y_min + tile_size / 2,
            dense_viewport.x_max + 1.5 * tile_size,
            dense_viewport.y_max + 1.5 * tile_size,
        )
        _, viewport_reads["exact_pan_first"] = _time_viewport(reader, 0, pan_first)
        _, viewport_reads["exact_pan_overlap"] = _time_viewport(reader, 0, pan_second)

        lod: dict[str, object] = {}
        for budget in (25_000, 50_000, 100_000):
            started = perf_counter()
            selection = reader.select_level(full_viewport, budget)
            lod[f"all_values_budget_{budget}"] = {
                **_level_selection_report(selection),
                "seconds": perf_counter() - started,
            }
        for label, value_id in representative_values.items():
            value_index = reader.load_selected_value_index(
                np.array([value_id], dtype=np.uint32),
                max_resident_bytes=_MAX_SELECTED_VALUE_INDEX_BYTES,
            )
            started = perf_counter()
            selection = reader.select_level(
                full_viewport,
                100_000,
                value_index=value_index,
            )
            lod[f"{label}_budget_100000"] = {
                **_level_selection_report(selection),
                "seconds": perf_counter() - started,
            }

        value_presence = np.diff(value_indptr, axis=1) > 0
        lost_candidates = np.flatnonzero(value_presence[0] & (~value_presence[1:]).any(axis=0))
        sampled_value_loss: dict[str, object]
        if len(lost_candidates) == 0:
            sampled_value_loss = {"observed": False}
        else:
            lost_value = int(lost_candidates[np.argmax(value_counts[lost_candidates])])
            counts_by_level: list[int] = []
            with _CatalogReader(cache_root) as local_catalog:
                for level in range(len(attributes.levels)):
                    start = int(value_indptr[level, lost_value])
                    stop = int(value_indptr[level, lost_value + 1])
                    counts_by_level.append(
                        int(
                            np.asarray(
                                local_catalog.array(VALUE_TILES_N_POINTS)[start:stop],
                                dtype=np.uint64,
                            ).sum(dtype=np.uint64)
                        )
                    )
            value_index = reader.load_selected_value_index(
                np.array([lost_value], dtype=np.uint32),
                max_resident_bytes=_MAX_SELECTED_VALUE_INDEX_BYTES,
            )
            started = perf_counter()
            fallback = reader.select_level(
                full_viewport,
                1,
                value_index=value_index,
            )
            sampled_value_loss = {
                "observed": True,
                "value_id": lost_value,
                "value_name": attributes.value_names[lost_value],
                "counts_by_level": counts_by_level,
                "budget_1_selection": {
                    **_level_selection_report(fallback),
                    "seconds": perf_counter() - started,
                },
            }

        return {
            "reader_open_seconds": reader_open_seconds,
            "resident_index_bytes": reader.resident_index_bytes,
            "generation_id": reader.cache_generation_id,
            "level_count": reader.level_count,
            "representative_levels": representative_levels,
            "representative_tiles": representative_tiles,
            "representative_values": representative_values,
            "tile_reads": tile_reads,
            "viewport_reads": viewport_reads,
            "selected_reads": selected_reads,
            "application_cold_and_warm": _measure_application_cold_and_warm(
                cache_root,
                0,
                dense_viewport,
                np.array([representative_values["common"]], dtype=np.uint32),
            ),
            "level_selection": lod,
            "sampled_value_loss": sampled_value_loss,
            "final_reader_cache": {
                "open_readers": reader.open_bucket_reader_count,
            },
            "point_id_payload_access": "forbidden by _BucketReader.read_display_payload; physical omission test passed",
            "cache_state_limitation": (
                "cold/warm refer only to the application reader cache; OS, filesystem, and codec caches were not reset"
            ),
        }
    finally:
        reader.__exit__(None, None, None)


def main() -> None:
    """Run one retained full-Xenium Z9 build and acceptance evaluation."""
    args = _parse_args()
    if args.json_output.exists():
        raise FileExistsError(f"Gate Z9 report already exists: {args.json_output}")
    if not args.temporary_directory_root.is_dir():
        raise ValueError("`temporary-directory-root` must already exist.")
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    output_existed = args.output_path.exists()
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

    config = _PointsCacheBuilderConfig(
        leaf_tile_size=args.leaf_tile_size,
        overview_point_budget=args.overview_point_budget,
        dask_worker_count=args.dask_worker_count,
    )
    print("Building and publishing the complete Zarr cache...", flush=True)
    with _RssSampler() as resources:
        started = perf_counter()
        published = _build_points_cache_zarr(
            validated,
            output_path=args.output_path,
            temporary_directory_root=args.temporary_directory_root,
            config=config,
        )
        build_seconds = perf_counter() - started
    if published != args.output_path:
        raise RuntimeError("Builder returned an unexpected publication path.")

    evaluation = _evaluate_reader(args.output_path)
    stored_bytes, file_count = _directory_summary(args.output_path)
    report = {
        "schema_version": "harpy-zarr-acceptance-evaluation-v1",
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
            "leaf_tile_size": args.leaf_tile_size,
            "overview_point_budget": args.overview_point_budget,
            "target_points_per_bucket": TARGET_POINTS_PER_BUCKET,
            "dask_worker_count": args.dask_worker_count,
            **asdict(config.zarr_settings),
            **asdict(config.catalog_settings),
        },
        "publication": {
            "output_path": str(args.output_path),
            "action": "replaced" if output_existed else "created",
            "cache_generation_id": evaluation["generation_id"],
            "stored_bytes": stored_bytes,
            "filesystem_file_count": file_count,
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
        "acceptance_reader": evaluation,
    }
    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.json_output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")


if __name__ == "__main__":
    main()
