"""Compare worker planning and value-major reads under different CPU residency.

Example::

    .venv/bin/python scripts/benchmark_tiled_points_viewport_planning.py \
        /path/to/transcripts_vis_zarr --selection value_a \
        --selection value_a value_b --json-output /tmp/viewport-planning.json

Cold means no CPU-resident tiles, not a flushed filesystem cache. Startup and
index loading are measured separately. No GUI or physical draw is performed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import sys
import time
from dataclasses import fields, is_dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from benchmark_tiled_points_cache_to_canvas import (
    _centered_viewport,
    _elapsed_ms,
    _git_state,
    _install_reader_timers,
    _TemporaryPatches,
    _TimingLog,
)

from napari_harpy.core.multi_scale_cache_points_zarr.reader import (
    _PointsCacheReader,
    _SelectedValueIndex,
    _ViewportReadPlan,
)
from napari_harpy.core.multi_scale_cache_points_zarr.storage.value_major_reader import _ValueMajorLocationReader
from napari_harpy.viewer.tiled_points.contracts import _ViewportRequest
from napari_harpy.viewer.tiled_points.runtime.cache_session import _read_viewport_snapshot
from napari_harpy.viewer.tiled_points.runtime.residency import _CpuTileResidency


def _plan_owned_arrays(plan: _ViewportReadPlan) -> dict[str, int]:
    """Count retained plan arrays, excluding its borrowed selected-level index.

    Inspect dataclass fields rather than a particular per-tile field name, so
    allocation evidence remains comparable across changes to the plan layout.
    These are retained arrays, not all transient allocations during planning.
    """
    arrays: dict[int, np.ndarray] = {}
    borrowed = plan.selected_value_level_index

    def visit(value: Any) -> None:
        if value is borrowed:
            return
        if isinstance(value, np.ndarray):
            arrays[id(value)] = value
        elif is_dataclass(value):
            for field in fields(value):
                visit(getattr(value, field.name))
        elif isinstance(value, (tuple, list)):
            for item in value:
                visit(item)

    visit(plan)
    return {"count": len(arrays), "bytes": sum(array.nbytes for array in arrays.values())}


def _measure_worker(
    reader: _PointsCacheReader,
    index: _SelectedValueIndex,
    residency: _CpuTileResidency,
    request: _ViewportRequest,
) -> dict[str, Any]:
    timings = _TimingLog()
    plans = []
    missing_counts = []
    block_resolution_ms = []
    physical_started: float | None = None
    with _TemporaryPatches() as patches:
        _install_reader_timers(timings, patches)
        original_plan = _PointsCacheReader.plan_viewport
        original_read = _PointsCacheReader.read_planned_tiles
        original_value_read = _PointsCacheReader._read_value_major_requests
        original_intervals = _ValueMajorLocationReader.read_intervals

        def plan_viewport(*args: Any, **kwargs: Any) -> Any:
            plan = original_plan(*args, **kwargs)
            plans.append(plan)
            return plan

        def read_planned_tiles(self: Any, plan: Any, tile_keys: Any, **kwargs: Any) -> Any:
            missing_counts.append(len(tile_keys))
            return original_read(self, plan, tile_keys, **kwargs)

        def value_read(*args: Any, **kwargs: Any) -> Any:
            nonlocal physical_started
            physical_started = time.perf_counter()
            try:
                return original_value_read(*args, **kwargs)
            finally:
                physical_started = None

        def read_intervals(*args: Any, **kwargs: Any) -> Any:
            # Everything before the location-reader call is block resolution;
            # payload IO and scattering are excluded from this measurement.
            if physical_started is not None:
                block_resolution_ms.append(_elapsed_ms(physical_started))
            return original_intervals(*args, **kwargs)

        patches.patch(_PointsCacheReader, "plan_viewport", plan_viewport)
        patches.patch(_PointsCacheReader, "read_planned_tiles", read_planned_tiles)
        patches.patch(_PointsCacheReader, "_read_value_major_requests", value_read)
        patches.patch(_ValueMajorLocationReader, "read_intervals", read_intervals)
        started = time.perf_counter()
        snapshot = _read_viewport_snapshot(
            reader,
            index,
            residency,
            request,
            max_vertex_payload_bytes=512 << 20,
            raise_if_cancelled=lambda: None,
        )
        worker_ms = _elapsed_ms(started)

    if not snapshot.within_budget or len(plans) != 1:
        raise RuntimeError("Benchmark requires one accepted viewport plan.")
    plan = plans[0]
    if plan.selected_value_level_index is None:
        raise RuntimeError("Benchmark requires a proper-subset viewport plan.")
    # Introspection and hashing run after timing, particularly because counting
    # per-tile arrays would otherwise inflate the older plan's worker time.
    return {
        "worker_ms": worker_ms,
        "block_resolution_ms": sum(block_resolution_ms),
        "block_resolution_calls": len(block_resolution_ms),
        "positive_tiles": len(plan.requests),
        "missing_tiles": sum(missing_counts),
        "selected_value_tile_records": len(plan.selected_value_level_index.manifest_index),
        "plan_owned_arrays": _plan_owned_arrays(plan),
        "level": snapshot.level,
        "level_kind": snapshot.level_kind,
        "point_count": snapshot.estimated_point_count,
        "omitted_value_ids": snapshot.omitted_value_ids,
        "render_batch_sha256": hashlib.sha256(snapshot.render_batch.vertices.tobytes()).hexdigest(),
        "breakdown": timings.summary(),
    }


def main() -> None:
    """Measure repeatable viewport requests and write the comparison evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache_root", type=Path)
    parser.add_argument("--selection", nargs="+", action="append", required=True)
    parser.add_argument("--viewport-fractions", nargs="+", type=float, default=[1.0, 0.2])
    parser.add_argument("--point-budget", type=int, default=100_000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--json-output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1 or args.point_budget < 1 or any(not 0 < f <= 1 for f in args.viewport_fractions):
        parser.error("Repeats and budget must be positive; viewport fractions must be in (0, 1].")
    report: dict[str, Any] = {
        "cache_root": str(args.cache_root.resolve()),
        "git": _git_state(),
        "point_budget": args.point_budget,
        "notes": "Cold is empty CPU residency; OS caches are not flushed. Hooks add some overhead.",
        "cases": [],
    }
    started = time.perf_counter()
    with _PointsCacheReader(args.cache_root) as reader:
        report["reader_enter_ms"] = _elapsed_ms(started)
        report["cache_generation_id"] = reader.cache_generation_id
        report["resident_compact_index_bytes"] = reader.resident_index_bytes
        report["tile_descriptor_count"] = reader.tile_descriptor_count
        report["index_memory_scope"] = "NumPy arrays only; Python descriptors and containers are excluded."
        report["resident_value_major_pointer_bytes"] = reader.resident_value_major_pointer_bytes
        report["open_bucket_readers"] = reader.open_bucket_reader_count
        for names in args.selection:
            ids = tuple(sorted({reader.value_names.index(name) for name in names}))
            started = time.perf_counter()
            index = reader.load_selected_value_index(np.asarray(ids, dtype=np.uint32), max_resident_bytes=None)
            index_ms = _elapsed_ms(started)
            for fraction in args.viewport_fractions:
                request = _ViewportRequest(
                    request_generation=1,
                    selection_generation=1,
                    requested_value_ids=ids,
                    viewport=_centered_viewport(reader.dataset_info, fraction, args.point_budget, 1200, 900),
                )
                case: dict[str, Any] = {
                    "values": names,
                    "value_ids": ids,
                    "viewport_fraction": fraction,
                    "selected_index_ms": index_ms,
                    "states": {state: [] for state in ("cold", "partial", "full")},
                }
                seed_tiles = ()
                for repeat in range(args.repeats):
                    for state, stride in (("cold", None), ("partial", 2), ("full", 1)):
                        residency = _CpuTileResidency(1 << 30)
                        if stride is not None:
                            residency.retain(seed_tiles[::stride], protected_keys=())
                        measurement = _measure_worker(reader, index, residency, request)
                        case["states"][state].append(measurement)
                        if state == "cold" and repeat == 0:
                            seed_tiles = tuple(residency._entries.values())
                        request = replace(request, request_generation=request.request_generation + 1)
                hashes = {
                    measurement["render_batch_sha256"]
                    for measurements in case["states"].values()
                    for measurement in measurements
                }
                if len(hashes) != 1:
                    raise RuntimeError("Residency changed the rendered payload.")
                report["cases"].append(case)
                print(f"Measured {names}, viewport fraction {fraction}", flush=True)
        report["after_viewport_trace"] = {
            "open_bucket_readers": reader.open_bucket_reader_count,
        }
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report["peak_rss_mib"] = peak_rss / ((1 << 20) if sys.platform == "darwin" else 1024)
    args.json_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Report: {args.json_output}")


if __name__ == "__main__":
    main()
