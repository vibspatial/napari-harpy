from __future__ import annotations

import argparse
import json
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import psutil

from napari_harpy.core.multi_scale_cache_points_zarr.reader import (
    _IntrinsicViewport,
    _PointsCacheReader,
    _SelectedValueIndex,
    _ViewportReadResult,
)
from napari_harpy.core.multi_scale_cache_points_zarr.storage._schema import (
    VALUE_TILES_MANIFEST_INDEX,
    VALUE_TILES_N_POINTS,
)

_TARGET_ARRAYS = (VALUE_TILES_MANIFEST_INDEX, VALUE_TILES_N_POINTS)
_RSS_SAMPLE_INTERVAL_SECONDS = 0.01


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


@dataclass(frozen=True)
class _CatalogSelectionSummary:
    selector_identity: int
    kind: str
    row_count: int
    selector_bytes: int
    first_row: int
    last_row: int


class _TrackedArray:
    def __init__(
        self,
        name: str,
        array: object,
        selections: dict[str, list[_CatalogSelectionSummary]],
    ) -> None:
        self._name = name
        self._array = array
        self._selections = selections

    def get_orthogonal_selection(self, selection: tuple[object, ...]) -> object:
        if not isinstance(selection, tuple) or len(selection) != 1:
            raise RuntimeError("Selected-value-index evaluation expected one row selector.")
        row_selection = selection[0]
        if isinstance(row_selection, slice):
            if row_selection.step not in (None, 1) or row_selection.start is None or row_selection.stop is None:
                raise RuntimeError("Catalog slice selectors require explicit unit-step terminals.")
            start = int(row_selection.start)
            stop = int(row_selection.stop)
            summary = _CatalogSelectionSummary(id(row_selection), "slice", stop - start, 0, start, stop - 1)
        elif (
            isinstance(row_selection, np.ndarray)
            and row_selection.dtype == np.dtype(np.int64)
            and row_selection.ndim == 1
            and row_selection.flags.c_contiguous
            and len(row_selection) > 0
        ):
            summary = _CatalogSelectionSummary(
                id(row_selection),
                "int64",
                len(row_selection),
                row_selection.nbytes,
                int(row_selection[0]),
                int(row_selection[-1]),
            )
        else:
            raise RuntimeError("Catalog row selector must be an explicit slice or nonempty C-contiguous int64 array.")
        self._selections[self._name].append(summary)
        return self._array.get_orthogonal_selection(selection)  # type: ignore[union-attr]


class _CatalogSelectionTracker:
    def __init__(self, reader: _PointsCacheReader) -> None:
        catalog = reader._catalog_or_raise()
        self._original_array = catalog.array
        self.selections = {name: [] for name in _TARGET_ARRAYS}

        def tracked_array(name: str) -> object:
            array = self._original_array(name)
            if name in self.selections:
                return _TrackedArray(name, array, self.selections)
            return array

        catalog.array = tracked_array  # type: ignore[method-assign]

    def reset(self) -> None:
        for selections in self.selections.values():
            selections.clear()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the resident selected-value runtime index.")
    parser.add_argument("cache_root", type=Path)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--max-resident-bytes", type=int, default=1 << 30)
    parser.add_argument("--point-budget", type=int, default=100_000)
    return parser.parse_args()


def _full_viewport(reader: _PointsCacheReader) -> _IntrinsicViewport:
    geometry = reader._attributes_or_raise().geometry
    return _IntrinsicViewport(
        geometry.x_min,
        geometry.y_min,
        np.nextafter(geometry.x_max, np.inf),
        np.nextafter(geometry.y_max, np.inf),
    )


def _evaluation_viewports(reader: _PointsCacheReader) -> tuple[_IntrinsicViewport, ...]:
    full = _full_viewport(reader)
    width = full.x_max - full.x_min
    height = full.y_max - full.y_min
    return (
        full,
        _IntrinsicViewport(full.x_min, full.y_min, full.x_min + width * 0.55, full.y_min + height * 0.55),
        _IntrinsicViewport(
            full.x_min + width * 0.2,
            full.y_min + height * 0.2,
            full.x_min + width * 0.75,
            full.y_min + height * 0.75,
        ),
    )


def _abundant_value_sets(reader: _PointsCacheReader) -> dict[int, np.ndarray]:
    counts = reader._value_n_points
    if counts is None:
        raise RuntimeError("Value totals are not loaded.")
    order = np.argsort(counts, kind="stable")[::-1]
    if len(order) < 100:
        raise RuntimeError("The retained evaluation cache has fewer than 100 values.")
    return {size: np.sort(order[:size]).astype(np.uint32) for size in (1, 10, 100)}


def _viewport_value_sets(reader: _PointsCacheReader) -> dict[str, np.ndarray]:
    """Return selections that exercise common, rare, and many-value reads."""
    counts = reader._value_n_points
    if counts is None:
        raise RuntimeError("Value totals are not loaded.")
    positive = np.flatnonzero(counts)
    exact_tile_counts = np.diff(reader._value_tiles_indptr_or_raise()[0])
    rare_cutoff = np.quantile(counts[positive], 0.25)
    rare = positive[counts[positive] <= rare_cutoff]
    rare_distributed = int(rare[np.argmax(exact_tile_counts[rare])])
    abundant = _abundant_value_sets(reader)
    return {
        "common_1": abundant[1],
        "rare_distributed_1": np.array([rare_distributed], dtype=np.uint32),
        "abundant_10": abundant[10],
        "abundant_100": abundant[100],
    }


def _exact_nine_tile_viewport(reader: _PointsCacheReader) -> tuple[_IntrinsicViewport, dict[str, int]]:
    """Construct a half-tile-offset viewport intersecting nine dense Exact tiles."""
    metadata = reader._attributes_or_raise().levels[0]
    pointers = reader._manifest_level_indptr_or_raise()
    start = int(pointers[0])
    stop = int(pointers[1])
    tile_x = reader._manifest_tile_x_or_raise()[start:stop]
    tile_y = reader._manifest_tile_y_or_raise()[start:stop]
    n_points = reader._manifest_n_points_or_raise()[start:stop]
    eligible = (tile_x + 2 < metadata.grid_width) & (tile_y + 2 < metadata.grid_height)
    eligible_rows = np.flatnonzero(eligible)
    if len(eligible_rows) == 0:
        raise RuntimeError("Exact grid has no tile from which a nine-tile viewport can be constructed.")
    dense_position = int(eligible_rows[np.argmax(n_points[eligible_rows])])
    dense_x = int(tile_x[dense_position])
    dense_y = int(tile_y[dense_position])
    geometry = reader._attributes_or_raise().geometry
    tile_size = metadata.tile_size
    x_start = geometry.x_origin + dense_x * tile_size
    y_start = geometry.y_origin + dense_y * tile_size
    viewport = _IntrinsicViewport(
        x_start + tile_size / 2,
        y_start + tile_size / 2,
        x_start + 2.5 * tile_size,
        y_start + 2.5 * tile_size,
    )
    visible = reader._visible_manifest_rows(0, viewport)
    if len(visible) != 9:
        raise RuntimeError(f"Expected nine nonempty Exact tiles, observed {len(visible)}.")
    return viewport, {
        "anchor_tile_x": dense_x,
        "anchor_tile_y": dense_y,
        "visible_exact_tiles": len(visible),
        "complete_exact_points": int(reader._manifest_n_points_or_raise()[visible].sum(dtype=np.uint64)),
    }


def _projected_bytes(reader: _PointsCacheReader, value_ids: np.ndarray) -> int:
    pointers = reader._value_tiles_indptr_or_raise()
    indexes = value_ids.astype(np.int64, copy=False)
    records = pointers[:, indexes + 1] - pointers[:, indexes]
    return int(
        value_ids.nbytes
        + pointers.shape[0] * (len(value_ids) + 1) * np.dtype(np.uint64).itemsize
        + int(records.sum(dtype=np.uint64)) * 2 * np.dtype(np.uint64).itemsize
    )


def _index_load_io_summary(
    reader: _PointsCacheReader,
    value_ids: np.ndarray,
    selections: Sequence[_CatalogSelectionSummary],
    *,
    chunk_rows: int,
    shard_rows: int,
) -> dict[str, int]:
    pointers = reader._value_tiles_indptr_or_raise()
    indexes = value_ids.astype(np.int64, copy=False)
    touched_chunks = 0
    touched_shards = 0
    for level in range(reader.level_count):
        starts = pointers[level, indexes]
        stops = pointers[level, indexes + 1]
        chunk_ids = {
            chunk_id
            for start, stop in zip(starts.tolist(), stops.tolist(), strict=True)
            if start < stop
            for chunk_id in range(start // chunk_rows, (stop - 1) // chunk_rows + 1)
        }
        shard_ids = {
            shard_id
            for start, stop in zip(starts.tolist(), stops.tolist(), strict=True)
            if start < stop
            for shard_id in range(start // shard_rows, (stop - 1) // shard_rows + 1)
        }
        touched_chunks += len(chunk_ids)
        touched_shards += len(shard_ids)
    return {
        "selections_per_catalog_array": len(selections),
        "total_parallel_array_selections": 2 * len(selections),
        "slice_selections": sum(selection.kind == "slice" for selection in selections),
        "int64_selections": sum(selection.kind == "int64" for selection in selections),
        "exact_selected_rows": sum(selection.row_count for selection in selections),
        "total_selector_bytes": sum(selection.selector_bytes for selection in selections),
        "maximum_level_selector_bytes": max(
            (selection.selector_bytes for selection in selections),
            default=0,
        ),
        "touched_inner_chunks": touched_chunks,
        "touched_shards": touched_shards,
    }


def _time_runtime_planning(
    reader: _PointsCacheReader,
    value_index: _SelectedValueIndex,
    viewports: tuple[_IntrinsicViewport, ...],
    *,
    point_budget: int,
) -> list[dict[str, object]]:
    reports: list[dict[str, object]] = []
    for viewport_index, viewport in enumerate(viewports):
        started = perf_counter()
        selection = reader.select_level(viewport, point_budget, value_index=value_index)
        lod_seconds = perf_counter() - started
        visible_rows = reader._visible_manifest_rows(selection.level, viewport)
        started = perf_counter()
        positive = reader._selected_value_manifest(selection.level, visible_rows, value_index)
        discovery_seconds = perf_counter() - started
        reports.append(
            {
                "viewport_index": viewport_index,
                "level": selection.level,
                "estimated_point_count": selection.estimated_point_count,
                "positive_visible_tile_count": selection.positive_visible_tile_count,
                "within_budget": selection.within_budget,
                "lod_seconds": lod_seconds,
                "positive_tile_discovery_seconds": discovery_seconds,
                "discovered_positive_tiles": len(positive),
            }
        )
    return reports


def _viewport_result_summary(
    result: _ViewportReadResult,
    seconds: float,
    *,
    readers_opened: int,
) -> dict[str, object]:
    tiles = result.tiles
    return {
        "seconds": seconds,
        "returned_tiles": len(tiles),
        "returned_points": sum(len(tile.value_id) for tile in tiles),
        "bucket_readers_opened": readers_opened,
    }


def _measure_selected_viewport(
    cache_root: Path,
    viewport: _IntrinsicViewport,
    value_ids: np.ndarray,
    *,
    fixed_level: int | None,
    max_resident_bytes: int,
    point_budget: int,
) -> dict[str, object]:
    """Measure first and repeated payload reads through one fresh reader."""
    with _PointsCacheReader(cache_root) as reader:
        started = perf_counter()
        value_index = reader.load_selected_value_index(value_ids, max_resident_bytes=max_resident_bytes)
        index_load_seconds = perf_counter() - started
        if value_index is None:
            raise RuntimeError("A proper subset unexpectedly normalized to the all-values path.")

        started = perf_counter()
        if fixed_level is None:
            selection = reader.select_level(viewport, point_budget, value_index=value_index)
            if not selection.within_budget:
                raise RuntimeError("The realistic selected viewport did not fit the supplied point budget.")
            level = selection.level
            estimated_points = selection.estimated_point_count
            estimated_positive_tiles = selection.positive_visible_tile_count
        else:
            level = fixed_level
            visible = reader._visible_manifest_rows(level, viewport)
            counts_by_value, estimated_positive_tiles = reader._selected_value_manifest_summary(
                level,
                visible,
                value_index,
            )
            estimated_points = int(counts_by_value.sum(dtype=np.uint64))
        planning_seconds = perf_counter() - started
        visible_rows = reader._visible_manifest_rows(level, viewport)
        visible_tile_count = len(visible_rows)
        positive_rows = reader._selected_value_manifest(level, visible_rows, value_index)
        bucket_keys = tuple(
            sorted(
                {
                    (reader._descriptors[manifest_row].level, reader._descriptors[manifest_row].bucket_id)
                    for manifest_row in positive_rows
                }
            )
        )
        started = perf_counter()
        if bucket_keys:
            projected_lookup_bytes = reader.project_bucket_lookup_index_bytes(bucket_keys=bucket_keys)
            resident_lookup_bytes = reader.load_bucket_lookup_indexes(
                bucket_keys=bucket_keys,
                max_resident_bytes=projected_lookup_bytes,
            )
        else:
            resident_lookup_bytes = 0
        bucket_lookup_prime_seconds = perf_counter() - started

        open_before = reader.open_bucket_reader_count
        started = perf_counter()
        first = reader.read_viewport(level, viewport, value_index=value_index)
        first_seconds = perf_counter() - started
        open_after_first = reader.open_bucket_reader_count

        started = perf_counter()
        repeated = reader.read_viewport(level, viewport, value_index=value_index)
        repeated_seconds = perf_counter() - started
        open_after_repeated = reader.open_bucket_reader_count

        first_summary = _viewport_result_summary(
            first,
            first_seconds,
            readers_opened=open_after_first - open_before,
        )
        repeated_summary = _viewport_result_summary(
            repeated,
            repeated_seconds,
            readers_opened=open_after_repeated - open_after_first,
        )
        if first_summary["returned_tiles"] != estimated_positive_tiles:
            raise RuntimeError("Selected viewport returned an unexpected positive-tile count.")
        if first_summary["returned_points"] != estimated_points:
            raise RuntimeError("Selected viewport returned an unexpected point count.")
        if (
            repeated_summary["returned_tiles"] != first_summary["returned_tiles"]
            or repeated_summary["returned_points"] != first_summary["returned_points"]
        ):
            raise RuntimeError("Repeated selected viewport read returned different logical content.")

        return {
            "level": level,
            "visible_tiles": visible_tile_count,
            "estimated_positive_tiles": estimated_positive_tiles,
            "estimated_points": estimated_points,
            "index_load_seconds": index_load_seconds,
            "planning_seconds": planning_seconds,
            "bucket_lookup_prime_seconds": bucket_lookup_prime_seconds,
            "resident_bucket_lookup_bytes": resident_lookup_bytes,
            "first": first_summary,
            "repeated": repeated_summary,
        }


def _measure_selected_viewports(
    cache_root: Path,
    viewport: _IntrinsicViewport,
    value_sets: dict[str, np.ndarray],
    *,
    max_resident_bytes: int,
    point_budget: int,
) -> dict[str, object]:
    reports: dict[str, object] = {}
    for label, value_ids in value_sets.items():
        reports[label] = {
            "value_ids": value_ids.tolist(),
            "forced_exact": _measure_selected_viewport(
                cache_root,
                viewport,
                value_ids,
                fixed_level=0,
                max_resident_bytes=max_resident_bytes,
                point_budget=point_budget,
            ),
            "budget_selected": _measure_selected_viewport(
                cache_root,
                viewport,
                value_ids,
                fixed_level=None,
                max_resident_bytes=max_resident_bytes,
                point_budget=point_budget,
            ),
        }
    return reports


def _evaluate_selection(
    reader: _PointsCacheReader,
    tracker: _CatalogSelectionTracker,
    value_ids: np.ndarray,
    viewports: tuple[_IntrinsicViewport, ...],
    *,
    max_resident_bytes: int,
    point_budget: int,
) -> dict[str, object]:
    projected_bytes = _projected_bytes(reader, value_ids)
    tracker.reset()
    with _RssSampler() as resources:
        started = perf_counter()
        value_index = reader.load_selected_value_index(value_ids, max_resident_bytes=max_resident_bytes)
        index_load_seconds = perf_counter() - started
    if value_index is None:
        raise RuntimeError("A proper subset unexpectedly normalized to the all-values path.")
    if tracker.selections[VALUE_TILES_MANIFEST_INDEX] != tracker.selections[VALUE_TILES_N_POINTS]:
        raise RuntimeError("Parallel value-tile arrays were not read through identical exact selectors.")
    settings = reader._attributes_or_raise().catalog.settings
    index_load_io = _index_load_io_summary(
        reader,
        value_ids,
        tracker.selections[VALUE_TILES_MANIFEST_INDEX],
        chunk_rows=settings.value_tile_chunk_rows,
        shard_rows=settings.value_tile_shard_rows,
    )
    index_load_io["exact_retained_records"] = sum(len(level.manifest_index) for level in value_index.levels)

    tracker.reset()
    runtime = _time_runtime_planning(
        reader,
        value_index,
        viewports,
        point_budget=point_budget,
    )
    runtime_catalog_selections = sum(len(items) for items in tracker.selections.values())
    if runtime_catalog_selections:
        raise RuntimeError("Indexed viewport planning performed catalog Zarr selections.")
    return {
        "value_ids": value_ids.tolist(),
        "index_load_seconds": index_load_seconds,
        "projected_resident_bytes": projected_bytes,
        "actual_resident_bytes": value_index.resident_bytes,
        "index_load_incremental_peak_rss_bytes": resources.peak_bytes - resources.baseline_bytes,
        "index_load_io": index_load_io,
        "runtime": runtime,
        "runtime_catalog_zarr_selections": runtime_catalog_selections,
    }


def main() -> None:
    """Evaluate one retained full-Xenium cache without rebuilding it."""
    args = _parse_args()
    if not args.cache_root.is_dir():
        raise ValueError("`cache_root` must be an existing directory.")
    if args.json_output.exists():
        raise FileExistsError(f"Z11 report already exists: {args.json_output}")
    args.json_output.parent.mkdir(parents=True, exist_ok=True)

    with _PointsCacheReader(args.cache_root) as reader:
        tracker = _CatalogSelectionTracker(reader)
        viewports = _evaluation_viewports(reader)
        viewport_value_sets = _viewport_value_sets(reader)
        selected_viewport, selected_viewport_metadata = _exact_nine_tile_viewport(reader)
        cache_generation_id = reader.cache_generation_id
        level_count = reader.level_count
        reports = {
            str(size): _evaluate_selection(
                reader,
                tracker,
                value_ids,
                viewports,
                max_resident_bytes=args.max_resident_bytes,
                point_budget=args.point_budget,
            )
            for size, value_ids in _abundant_value_sets(reader).items()
        }

    selected_viewport_reads = _measure_selected_viewports(
        args.cache_root,
        selected_viewport,
        viewport_value_sets,
        max_resident_bytes=args.max_resident_bytes,
        point_budget=args.point_budget,
    )
    report = {
        "schema_version": "harpy-zarr-selected-value-index-evaluation-v3",
        "cache_root": str(args.cache_root),
        "cache_generation_id": cache_generation_id,
        "level_count": level_count,
        "point_budget": args.point_budget,
        "max_resident_bytes": args.max_resident_bytes,
        "rss_sample_interval_seconds": _RSS_SAMPLE_INTERVAL_SECONDS,
        "cache_state_limitation": (
            "First and repeated refer to application bucket-reader state; operating-system, filesystem, and codec "
            "caches were not reset between measurements."
        ),
        "selections": reports,
        "selected_viewport": {
            **selected_viewport_metadata,
            "x_min": selected_viewport.x_min,
            "y_min": selected_viewport.y_min,
            "x_max": selected_viewport.x_max,
            "y_max": selected_viewport.y_max,
            "reads": selected_viewport_reads,
        },
    }

    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.json_output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")


if __name__ == "__main__":
    main()
