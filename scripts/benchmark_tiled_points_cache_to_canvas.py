"""Benchmark one selected-value request from a completed cache to the canvas.

The default run measures cache/catalog startup, resident bucket-index loading,
selected-value-index loading, cold and warm worker snapshots, Zarr selection
amplification, CPU residency, and steady Qt delivery. Add ``--real-canvas`` to
also measure VisPy resource creation/residency and synchronous physical draws.

For example, on macOS::

    QT_QPA_PLATFORM=cocoa .venv/bin/python \
        scripts/benchmark_tiled_points_cache_to_canvas.py \
        /path/to/transcripts_vis_zarr \
        --value AAMP \
        --point-budget 100000 \
        --real-canvas \
        --json-output /tmp/aamp-cache-to-canvas.json

The JSON report is descriptive evidence intended for comparison across code
changes. This script deliberately defines no numerical pass/fail thresholds.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import statistics
import subprocess
import time
from collections import defaultdict
from contextlib import AbstractContextManager
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import psutil
from napari._vispy.utils.qt_font import FontInfo
from qtpy.QtCore import QCoreApplication, QObject, QThread, Signal, Slot
from qtpy.QtWidgets import QApplication
from vispy.scene import SceneCanvas
from zarr.core.array import Array

import napari_harpy.core.multi_scale_cache_points_zarr.storage.bucket_reader as bucket_reader_module
import napari_harpy.viewer.tiled_points.runtime.cache_session as cache_session_module
from napari_harpy.core.multi_scale_cache_points_zarr.reader import _PointsCacheReader
from napari_harpy.viewer.tiled_points.application import canonical_value_palette
from napari_harpy.viewer.tiled_points.contracts import (
    TILED_POINTS_VERTEX_DTYPE,
    TiledPointsDatasetReference,
    TiledPointsRenderSnapshot,
    TiledPointsRenderTile,
    TiledPointsViewportState,
    TileResidencyKey,
    _ViewportRequest,
)
from napari_harpy.viewer.tiled_points.napari.layer import TiledPointsLayerModel
from napari_harpy.viewer.tiled_points.render_batch import pack_render_tiles
from napari_harpy.viewer.tiled_points.runtime.cache_session import _read_viewport_snapshot
from napari_harpy.viewer.tiled_points.runtime.residency import _CpuTileResidency
from napari_harpy.viewer.tiled_points.vispy.layer import VispyTiledPointsLayer

_MIB = 1 << 20
_DEFAULT_CPU_TILE_BYTES = 1 << 30
_DEFAULT_MAX_VERTEX_PAYLOAD_BYTES = 512 << 20
_DEFAULT_POINT_BUDGET = 100_000


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Profile one selected-value request from a completed tiled-points cache through CPU and, optionally, "
            "a real VisPy canvas. Results are descriptive benchmark evidence, not pass/fail thresholds."
        )
    )
    parser.add_argument("cache_root", type=Path, help="Completed transcripts_vis_zarr cache root.")
    parser.add_argument("--value", default="AAMP", help="Canonical value name to display (default: AAMP).")
    parser.add_argument("--point-budget", type=int, default=_DEFAULT_POINT_BUDGET)
    parser.add_argument(
        "--viewport-fraction",
        type=float,
        default=1.0,
        help="Centered fraction of the intrinsic data width and height to request (default: 1.0).",
    )
    parser.add_argument("--canvas-width", type=int, default=1_200)
    parser.add_argument("--canvas-height", type=int, default=900)
    parser.add_argument("--cpu-tile-cache-bytes", type=int, default=_DEFAULT_CPU_TILE_BYTES)
    parser.add_argument(
        "--max-vertex-payload-bytes",
        type=int,
        default=_DEFAULT_MAX_VERTEX_PAYLOAD_BYTES,
    )
    parser.add_argument("--qt-delivery-repeats", type=int, default=25)
    parser.add_argument("--warm-draw-repeats", type=int, default=7)
    parser.add_argument(
        "--synthetic-million-point-packing",
        action="store_true",
        help="Also benchmark worker packing for 1,000,000 points across representative tile counts.",
    )
    parser.add_argument("--synthetic-packing-repeats", type=int, default=5)
    parser.add_argument(
        "--real-canvas",
        action="store_true",
        help=(
            "Also construct the real VisPy renderer and synchronously render frames. Run with an appropriate "
            "QT_QPA_PLATFORM for the host, for example QT_QPA_PLATFORM=cocoa on macOS."
        ),
    )
    parser.add_argument("--json-output", type=Path, required=True)
    return parser.parse_args()


def _require_args(args: argparse.Namespace) -> None:
    if not args.cache_root.is_dir():
        raise ValueError(f"Cache root does not exist or is not a directory: {args.cache_root}")
    if args.point_budget <= 0:
        raise ValueError("--point-budget must be positive.")
    if not 0.0 < args.viewport_fraction <= 1.0:
        raise ValueError("--viewport-fraction must be in (0, 1].")
    if args.canvas_width <= 0 or args.canvas_height <= 0:
        raise ValueError("Canvas dimensions must be positive.")
    if args.cpu_tile_cache_bytes <= 0 or args.max_vertex_payload_bytes <= 0:
        raise ValueError("CPU tile-cache and vertex-payload byte limits must be positive.")
    if args.qt_delivery_repeats < 0 or args.warm_draw_repeats < 0:
        raise ValueError("Repeat counts must be nonnegative.")
    if args.synthetic_packing_repeats <= 0:
        raise ValueError("--synthetic-packing-repeats must be positive.")


def _elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1_000.0


def _rss_mib() -> float:
    return psutil.Process().memory_info().rss / _MIB


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _git_state() -> dict[str, object]:
    try:
        commit = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ("git", "status", "--porcelain"),
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        )
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": dirty}


class _TimingLog:
    def __init__(self) -> None:
        self.calls: dict[str, list[float]] = defaultdict(list)
        self.zarr_calls: list[dict[str, object]] = []

    def clear(self) -> None:
        self.calls.clear()
        self.zarr_calls.clear()

    def summary(self) -> dict[str, object]:
        result: dict[str, object] = {
            name: {
                "calls": len(values),
                "total_ms": sum(values),
                "median_ms": statistics.median(values),
                "max_ms": max(values),
            }
            for name, values in self.calls.items()
        }
        by_array: dict[str, list[dict[str, object]]] = defaultdict(list)
        for item in self.zarr_calls:
            by_array[str(item["name"])].append(item)
        result["zarr_arrays"] = {
            name: {
                "calls": len(items),
                "total_ms": sum(float(item["elapsed_ms"]) for item in items),
                "median_ms": statistics.median(float(item["elapsed_ms"]) for item in items),
                "max_ms": max(float(item["elapsed_ms"]) for item in items),
                "returned_mib": sum(int(item["returned_bytes"]) for item in items) / _MIB,
                "selected_rows": sum(int(item["selected_rows"]) for item in items),
                "touched_chunks": sum(int(item["touched_chunks"]) for item in items),
                "touched_shards": sum(int(item["touched_shards"]) for item in items),
                "estimated_decoded_rows": sum(int(item["estimated_decoded_rows"]) for item in items),
                "estimated_decoded_mib": sum(int(item["estimated_decoded_bytes"]) for item in items) / _MIB,
                "row_amplification": (
                    sum(int(item["estimated_decoded_rows"]) for item in items)
                    / sum(int(item["selected_rows"]) for item in items)
                ),
                "chunk_rows": sorted({int(item["chunk_rows"]) for item in items}),
                "shard_rows": sorted({int(item["shard_rows"]) for item in items}),
            }
            for name, items in by_array.items()
        }
        return result


class _TemporaryPatches(AbstractContextManager["_TemporaryPatches"]):
    """Restore benchmark timing hooks even when the profiled operation fails."""

    def __init__(self) -> None:
        self._patches: list[tuple[object, str, object]] = []

    def patch(self, owner: object, name: str, replacement: object) -> None:
        self._patches.append((owner, name, getattr(owner, name)))
        setattr(owner, name, replacement)

    def __exit__(self, *_exc_info: object) -> None:
        for owner, name, original in reversed(self._patches):
            setattr(owner, name, original)


def _row_selection_statistics(array: Array, selection: object) -> dict[str, int]:
    row_selection = selection[0] if isinstance(selection, tuple) else selection
    if isinstance(row_selection, slice):
        start, stop, step = row_selection.indices(array.shape[0])
        if step != 1:
            return {
                "selected_rows": len(range(start, stop, step)),
                "touched_chunks": 0,
                "touched_shards": 0,
                "estimated_decoded_rows": 0,
            }
        selected_rows = max(stop - start, 0)
        rows = None
    elif isinstance(row_selection, np.ndarray) and row_selection.ndim == 1:
        selected_rows = len(row_selection)
        rows = row_selection.astype(np.int64, copy=False)
        start = int(rows[0]) if selected_rows else 0
        stop = int(rows[-1]) + 1 if selected_rows else 0
    else:
        return {"selected_rows": 0, "touched_chunks": 0, "touched_shards": 0, "estimated_decoded_rows": 0}

    if selected_rows == 0:
        return {"selected_rows": 0, "touched_chunks": 0, "touched_shards": 0, "estimated_decoded_rows": 0}
    chunk_rows = int(array.chunks[0])
    if rows is None:
        chunk_ids = np.arange(start // chunk_rows, (stop - 1) // chunk_rows + 1, dtype=np.int64)
    else:
        chunk_ids = np.unique(rows // chunk_rows)
    shard_shape = array.shards
    shard_rows = int(shard_shape[0]) if shard_shape is not None else chunk_rows
    shard_ids = np.unique((chunk_ids * chunk_rows) // shard_rows)
    point_count = int(array.shape[0])
    decoded_rows = sum(
        min((int(chunk_id) + 1) * chunk_rows, point_count) - int(chunk_id) * chunk_rows for chunk_id in chunk_ids
    )
    return {
        "selected_rows": selected_rows,
        "touched_chunks": len(chunk_ids),
        "touched_shards": len(shard_ids),
        "estimated_decoded_rows": decoded_rows,
    }


def _install_reader_timers(timings: _TimingLog, patches: _TemporaryPatches) -> None:
    def timed_method(owner: object, name: str, label: str) -> None:
        original = getattr(owner, name)

        def measured(*args: object, **kwargs: object) -> object:
            started = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                timings.calls[label].append(_elapsed_ms(started))

        patches.patch(owner, name, measured)

    timed_method(_PointsCacheReader, "select_level", "level_selection")
    timed_method(_PointsCacheReader, "plan_viewport", "viewport_plan")
    timed_method(_PointsCacheReader, "read_planned_tiles", "read_planned_tiles")
    timed_method(bucket_reader_module._BucketReader, "read_display_payloads", "bucket_batch")
    timed_method(
        bucket_reader_module._BucketReader,
        "resolve_selected_tile_intervals",
        "sparse_interval_resolution",
    )
    timed_method(bucket_reader_module, "_exact_row_selection", "exact_row_selector_construction")
    timed_method(_CpuTileResidency, "get", "cpu_residency_get")
    timed_method(_CpuTileResidency, "retain", "cpu_residency_retain")
    timed_method(cache_session_module, "_require_ordered_render_tiles", "render_tile_validation")
    timed_method(cache_session_module, "pack_render_tiles", "render_batch_packing")
    timed_method(
        cache_session_module,
        "TiledPointsRenderSnapshot",
        "logical_snapshot_construction",
    )

    original_zarr_selection = Array.get_orthogonal_selection

    def get_orthogonal_selection(self: Array, *args: object, **kwargs: object) -> object:
        selection = args[0] if args else kwargs.get("selection")
        started = time.perf_counter()
        result = original_zarr_selection(self, *args, **kwargs)
        name = self.name.rsplit("/", 1)[-1]
        if name in {"location", "value_id"} and selection is not None:
            row_statistics = _row_selection_statistics(self, selection)
            row_width = int(np.prod(self.shape[1:], dtype=np.int64)) if self.ndim > 1 else 1
            timings.zarr_calls.append(
                {
                    "name": name,
                    "elapsed_ms": _elapsed_ms(started),
                    "returned_bytes": int(getattr(result, "nbytes", 0)),
                    "estimated_decoded_bytes": row_statistics["estimated_decoded_rows"]
                    * row_width
                    * self.dtype.itemsize,
                    "chunk_rows": int(self.chunks[0]),
                    "shard_rows": int(self.shards[0]) if self.shards is not None else int(self.chunks[0]),
                    **row_statistics,
                }
            )
        return result

    patches.patch(Array, "get_orthogonal_selection", get_orthogonal_selection)


class _DeliveryWorker(QObject):
    delivered = Signal(object, float)

    @Slot(object)
    def forward(self, value: object) -> None:
        self.delivered.emit(value, time.perf_counter())


class _DeliveryRequester(QObject):
    requested = Signal(object)


class _DeliveryReceiver(QObject):
    def __init__(self) -> None:
        super().__init__()
        self.elapsed_ms: float | None = None
        self.vertex_allocation_id: int | None = None

    @Slot(object, float)
    def receive(self, value: object, emitted_at: float) -> None:
        if not isinstance(value, TiledPointsRenderSnapshot):
            raise ValueError("Qt delivery benchmark expected TiledPointsRenderSnapshot.")
        self.vertex_allocation_id = id(value.render_batch.vertices)
        self.elapsed_ms = _elapsed_ms(emitted_at)


def _measure_qt_delivery(
    application: QCoreApplication,
    snapshot: TiledPointsRenderSnapshot,
    *,
    repeats: int,
) -> dict[str, object] | None:
    if repeats == 0:
        return None
    thread = QThread()
    worker = _DeliveryWorker()
    requester = _DeliveryRequester()
    receiver = _DeliveryReceiver()
    worker.moveToThread(thread)
    requester.requested.connect(worker.forward)
    worker.delivered.connect(receiver.receive)
    thread.start()
    measurements: list[float] = []
    expected_vertex_allocation_id = id(snapshot.render_batch.vertices)
    preserved_allocation_identity = True
    try:
        # Discard the first delivery because it includes QThread startup scheduling.
        for _ in range(repeats + 1):
            receiver.elapsed_ms = None
            receiver.vertex_allocation_id = None
            requester.requested.emit(snapshot)
            deadline = time.monotonic() + 10.0
            while receiver.elapsed_ms is None and time.monotonic() < deadline:
                application.processEvents()
                time.sleep(0.0001)
            if receiver.elapsed_ms is None:
                raise TimeoutError("Qt delivery profiling timed out.")
            preserved_allocation_identity &= receiver.vertex_allocation_id == expected_vertex_allocation_id
            measurements.append(receiver.elapsed_ms)
    finally:
        thread.quit()
        thread.wait()
    steady = measurements[1:]
    return {
        "repeats": len(steady),
        "median_ms": statistics.median(steady),
        "minimum_ms": min(steady),
        "maximum_ms": max(steady),
        "measurements_ms": steady,
        "vertex_allocation_identity_preserved": preserved_allocation_identity,
    }


def _centered_viewport(
    info: object, fraction: float, point_budget: int, width: int, height: int
) -> TiledPointsViewportState:
    center_x = (info.x_min + info.x_max) / 2.0
    center_y = (info.y_min + info.y_max) / 2.0
    half_width = (math.nextafter(info.x_max, math.inf) - info.x_min) * fraction / 2.0
    half_height = (math.nextafter(info.y_max, math.inf) - info.y_min) * fraction / 2.0
    return TiledPointsViewportState(
        displayed_axes=(0, 1),
        x_min=center_x - half_width,
        y_min=center_y - half_height,
        x_max=center_x + half_width,
        y_max=center_y + half_height,
        canvas_width=width,
        canvas_height=height,
        hard_render_point_budget=point_budget,
        screen_density_budget=point_budget,
    )


def _snapshot_with_generation(
    snapshot: TiledPointsRenderSnapshot, request_generation: int
) -> TiledPointsRenderSnapshot:
    return TiledPointsRenderSnapshot(
        cache_generation_id=snapshot.cache_generation_id,
        request_generation=request_generation,
        selection_generation=snapshot.selection_generation,
        requested_value_ids=snapshot.requested_value_ids,
        level=snapshot.level,
        level_kind=snapshot.level_kind,
        within_budget=snapshot.within_budget,
        estimated_point_count=snapshot.estimated_point_count,
        omitted_value_ids=snapshot.omitted_value_ids,
        rendered_tile_count=snapshot.rendered_tile_count,
        render_batch=snapshot.render_batch,
    )


def _renderer_report(
    snapshot: TiledPointsRenderSnapshot,
    subset_snapshot: TiledPointsRenderSnapshot,
    info: object,
    viewport: TiledPointsViewportState,
    *,
    max_vertex_payload_bytes: int,
    warm_draw_repeats: int,
) -> dict[str, object]:
    layer = TiledPointsLayerModel(
        data=TiledPointsDatasetReference(
            cache_generation_id=info.cache_generation_id,
            points_name=info.points_name,
            value_column=info.value_column,
            value_count=len(info.value_names),
            x_origin=info.x_origin,
            y_origin=info.y_origin,
            x_min=info.x_min,
            x_max=info.x_max,
            y_min=info.y_min,
            y_max=info.y_max,
        ),
        value_palette=canonical_value_palette(len(info.value_names)),
        max_vertex_payload_bytes=max_vertex_payload_bytes,
        point_diameter=2.0,
        hard_render_point_budget=viewport.hard_render_point_budget,
    )
    report: dict[str, object] = {}
    started = time.perf_counter()
    visual = VispyTiledPointsLayer(layer, FontInfo())
    report["empty_layer_construction_ms"] = _elapsed_ms(started)
    report["visual_count"] = visual.visual_count
    report["vbo_count"] = visual.vbo_count
    canvas = SceneCanvas(show=True, size=(viewport.canvas_width, viewport.canvas_height))
    view = canvas.central_widget.add_view()
    view.camera = "panzoom"
    view.camera.rect = (
        viewport.x_min,
        viewport.y_min,
        viewport.x_max - viewport.x_min,
        viewport.y_max - viewport.y_min,
    )
    view.add(visual.node)
    try:
        started = time.perf_counter()
        canvas.render()
        report["empty_canvas_render_ms"] = _elapsed_ms(started)
        report["rss_before_snapshot_mib"] = _rss_mib()

        started = time.perf_counter()
        report["cold_apply_applied"] = visual.apply_snapshot(snapshot)
        report["cold_apply_ms"] = _elapsed_ms(started)
        if not report["cold_apply_applied"]:
            raise RuntimeError("The cold render snapshot was rejected by the renderer.")
        report["cold_vertex_staging_ms"] = visual.last_vertex_staging_ms
        report["cold_packed_vertex_bytes"] = visual.active_vertex_bytes
        report["cold_active_point_count"] = visual.active_point_count
        report["cold_payload_replacement_count"] = visual.payload_replacement_count
        report["point_draw_submissions_per_frame"] = visual.point_draw_submission_count
        report["cold_apply_breakdown"] = {
            "vertex_buffer_staging": {
                "calls": 1,
                "total_ms": visual.last_vertex_staging_ms,
                "median_ms": visual.last_vertex_staging_ms,
                "max_ms": visual.last_vertex_staging_ms,
            },
        }
        report["rss_after_apply_mib"] = _rss_mib()

        started = time.perf_counter()
        canvas.render()
        report["cold_first_draw_ms"] = _elapsed_ms(started)
        report["rss_after_first_draw_mib"] = _rss_mib()

        warm_draws: list[float] = []
        for _ in range(warm_draw_repeats):
            started = time.perf_counter()
            canvas.render()
            warm_draws.append(_elapsed_ms(started))
        report["warm_draw_measurements_ms"] = warm_draws
        report["warm_draw_median_ms"] = statistics.median(warm_draws) if warm_draws else None

        warm_snapshot = _snapshot_with_generation(snapshot, snapshot.request_generation + 1)
        started = time.perf_counter()
        report["warm_full_apply_applied"] = visual.apply_snapshot(warm_snapshot)
        report["warm_full_apply_ms"] = _elapsed_ms(started)
        report["warm_full_vertex_staging_ms"] = visual.last_vertex_staging_ms
        report["warm_full_payload_replacement_count"] = visual.payload_replacement_count
        started = time.perf_counter()
        canvas.render()
        report["warm_full_draw_ms"] = _elapsed_ms(started)

        report["subset_tile_count"] = subset_snapshot.rendered_tile_count
        report["subset_point_count"] = subset_snapshot.rendered_point_count
        started = time.perf_counter()
        report["warm_full_to_subset_applied"] = visual.apply_snapshot(subset_snapshot)
        report["warm_full_to_subset_apply_ms"] = _elapsed_ms(started)
        report["warm_full_to_subset_vertex_staging_ms"] = visual.last_vertex_staging_ms
        report["warm_full_to_subset_packed_vertex_bytes"] = visual.active_vertex_bytes
        report["final_payload_replacement_count"] = visual.payload_replacement_count
        started = time.perf_counter()
        canvas.render()
        report["warm_subset_draw_ms"] = _elapsed_ms(started)
    finally:
        visual.close()
        canvas.close()
        QApplication.processEvents()
    return report


def _dense_exact_tile_report(reader: _PointsCacheReader) -> dict[str, object]:
    """Measure one complete read of the densest serialized Exact tile twice."""
    level_indptr = reader._manifest_level_indptr_or_raise()
    level_start = int(level_indptr[0])
    level_stop = int(level_indptr[1])
    counts = reader._manifest_n_points_or_raise()[level_start:level_stop]
    manifest_row = level_start + int(np.argmax(counts))
    descriptor = reader._descriptors[manifest_row]

    measurements: dict[str, object] = {
        "tile_x": descriptor.tile_x,
        "tile_y": descriptor.tile_y,
        "manifest_point_count": descriptor.n_points,
    }
    timings = _TimingLog()
    with _TemporaryPatches() as patches:
        _install_reader_timers(timings, patches)
        started = time.perf_counter()
        first = reader.read_tile(0, descriptor.tile_x, descriptor.tile_y)
        measurements["first_ms"] = _elapsed_ms(started)
        measurements["first_breakdown"] = timings.summary()
        if first is None or len(first.value_id) != descriptor.n_points:
            raise RuntimeError("Densest Exact tile read did not return its complete manifest payload.")

        timings.clear()
        started = time.perf_counter()
        repeated = reader.read_tile(0, descriptor.tile_x, descriptor.tile_y)
        measurements["repeat_ms"] = _elapsed_ms(started)
        measurements["repeat_breakdown"] = timings.summary()
        if repeated is None or len(repeated.value_id) != descriptor.n_points:
            raise RuntimeError("Repeated densest Exact tile read did not return its complete manifest payload.")
    return measurements


def _synthetic_million_point_packing_report(*, repeats: int) -> dict[str, object]:
    """Measure one-million-point packing across representative fragmentation."""
    point_count = 1_000_000
    generation_id = "12345678-1234-5678-9234-567812345678"
    required_bytes = point_count * TILED_POINTS_VERTEX_DTYPE.itemsize
    cases: list[dict[str, object]] = []
    for tile_count in (1, 4_453, 7_294, 100_000):
        quotient, remainder = divmod(point_count, tile_count)
        started = time.perf_counter()
        tiles = tuple(
            TiledPointsRenderTile(
                key=TileResidencyKey(
                    cache_generation_id=generation_id,
                    requested_value_ids=None,
                    level=0,
                    tile_x=tile_x,
                    tile_y=0,
                ),
                tile_size=1,
                location=np.zeros((quotient + int(tile_x < remainder), 2), dtype=np.float32),
                value_id=np.zeros(quotient + int(tile_x < remainder), dtype=np.uint32),
            )
            for tile_x in range(tile_count)
        )
        input_construction_ms = _elapsed_ms(started)
        measurements: list[float] = []
        batch_bytes = 0
        for _ in range(repeats):
            started = time.perf_counter()
            batch = pack_render_tiles(
                tiles,
                point_count=point_count,
                value_count=1,
                max_vertex_payload_bytes=required_bytes,
            )
            measurements.append(_elapsed_ms(started))
            batch_bytes = batch.nbytes
        cases.append(
            {
                "tile_count": tile_count,
                "input_construction_ms": input_construction_ms,
                "measurements_ms": measurements,
                "median_ms": statistics.median(measurements),
                "maximum_ms": max(measurements),
                "batch_bytes": batch_bytes,
            }
        )
        del batch, tiles
        gc.collect()
    return {
        "point_count": point_count,
        "vertex_bytes_per_point": TILED_POINTS_VERTEX_DTYPE.itemsize,
        "repeats": repeats,
        "cases": cases,
    }


def _print_summary(report: dict[str, object]) -> None:
    startup = report["startup"]
    worker = report["worker"]
    snapshot = report["snapshot"]
    print(f"Value: {report['value']} (value_id={report['value_id']})")
    print(
        f"Snapshot: level={snapshot['level']} ({snapshot['level_kind']}), "
        f"tiles={snapshot['tile_count']:,}, points={snapshot['point_count']:,}"
    )
    print(
        f"Startup: reader={startup['reader_enter_ms']:.1f} ms, "
        f"bucket-index projection={startup['bucket_index_projection_ms']:.1f} ms, "
        f"bucket-index load={startup['bucket_index_loading_ms']:.1f} ms, "
        f"value-index load={startup['selected_value_index_ms']:.1f} ms"
    )
    print(f"Worker snapshot: cold={worker['cold_snapshot_ms']:.1f} ms, warm={worker['warm_snapshot_ms']:.1f} ms")
    cold_pack = worker["cold_breakdown"]["render_batch_packing"]["total_ms"]
    warm_pack = worker["warm_breakdown"]["render_batch_packing"]["total_ms"]
    print(f"Worker render-batch packing: cold={cold_pack:.1f} ms, warm={warm_pack:.1f} ms")
    dense = report["dense_exact_tile"]
    print(
        f"Dense Exact tile: points={dense['manifest_point_count']:,}, "
        f"first={dense['first_ms']:.1f} ms, repeat={dense['repeat_ms']:.1f} ms"
    )
    renderer = report.get("renderer")
    if isinstance(renderer, dict):
        print(
            f"Renderer: visuals={renderer['visual_count']}, VBOs={renderer['vbo_count']}, "
            f"cold apply={renderer['cold_apply_ms']:.1f} ms, "
            f"first draw={renderer['cold_first_draw_ms']:.1f} ms, "
            f"warm draw median={renderer['warm_draw_median_ms']} ms"
        )
    synthetic = report.get("synthetic_million_point_packing")
    if isinstance(synthetic, dict):
        summary = ", ".join(f"{case['tile_count']:,} tiles={case['median_ms']:.1f} ms" for case in synthetic["cases"])
        print(f"Synthetic 1M packing: {summary}")
    print(f"JSON: {report['json_output']}")


def main() -> None:
    """Run the requested cache-to-canvas benchmark and persist its JSON report."""
    args = _parse_args()
    _require_args(args)
    report: dict[str, object] = {
        "schema_version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cache_root": str(args.cache_root.resolve()),
        "value": args.value,
        "point_budget": args.point_budget,
        "viewport_fraction": args.viewport_fraction,
        "json_output": str(args.json_output.resolve()),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "qt_qpa_platform": os.environ.get("QT_QPA_PLATFORM"),
            "packages": {
                name: _package_version(name)
                for name in ("napari-harpy", "napari", "vispy", "zarr", "numpy", "numcodecs")
            },
            "git": _git_state(),
        },
        "notes": (
            "Cold means the first request in this process after indexes were loaded; it does not flush operating-system "
            "filesystem caches. Detailed method hooks add small instrumentation overhead. Canvas.render() is synchronous "
            "and includes framebuffer readback."
        ),
        "rss_start_mib": _rss_mib(),
    }

    started = time.perf_counter()
    reader_context = _PointsCacheReader(args.cache_root)
    reader = reader_context.__enter__()
    startup: dict[str, object] = {"reader_enter_ms": _elapsed_ms(started), "rss_after_reader_enter_mib": _rss_mib()}
    try:
        started = time.perf_counter()
        projected_lookup_bytes = reader.project_bucket_lookup_index_bytes()
        startup["bucket_index_projection_ms"] = _elapsed_ms(started)
        startup["bucket_index_projected_mib"] = projected_lookup_bytes / _MIB
        startup["rss_after_bucket_index_projection_mib"] = _rss_mib()

        started = time.perf_counter()
        resident_lookup_bytes = reader.load_bucket_lookup_indexes(max_resident_bytes=None)
        startup["bucket_index_loading_ms"] = _elapsed_ms(started)
        startup["bucket_index_resident_mib"] = resident_lookup_bytes / _MIB
        startup["bucket_index_count"] = reader.loaded_bucket_lookup_index_count
        startup["rss_after_bucket_index_loading_mib"] = _rss_mib()

        try:
            value_id = reader.value_names.index(args.value)
        except ValueError as exc:
            raise ValueError(f"Value {args.value!r} is not present in the cache vocabulary.") from exc
        value_ids = np.asarray((value_id,), dtype=np.uint32)
        started = time.perf_counter()
        selected_value_index = reader.load_selected_value_index(value_ids, max_resident_bytes=None)
        startup["selected_value_index_ms"] = _elapsed_ms(started)
        startup["selected_value_index_kib"] = (
            0.0 if selected_value_index is None else selected_value_index.resident_bytes / 1024
        )
        startup["rss_after_selected_value_index_mib"] = _rss_mib()
        report["startup"] = startup
        report["value_id"] = value_id

        info = reader.dataset_info
        report["cache_generation_id"] = info.cache_generation_id
        viewport = _centered_viewport(
            info,
            args.viewport_fraction,
            args.point_budget,
            args.canvas_width,
            args.canvas_height,
        )
        report["viewport"] = {
            "x_min": viewport.x_min,
            "y_min": viewport.y_min,
            "x_max": viewport.x_max,
            "y_max": viewport.y_max,
            "canvas_width": viewport.canvas_width,
            "canvas_height": viewport.canvas_height,
            "effective_point_budget": viewport.effective_point_budget,
        }
        requested_value_ids = None if selected_value_index is None else (value_id,)
        request = _ViewportRequest(
            request_generation=1,
            selection_generation=1,
            requested_value_ids=requested_value_ids,
            viewport=viewport,
        )
        residency = _CpuTileResidency(args.cpu_tile_cache_bytes)
        timings = _TimingLog()
        with _TemporaryPatches() as patches:
            _install_reader_timers(timings, patches)
            started = time.perf_counter()
            snapshot = _read_viewport_snapshot(
                reader,
                selected_value_index,
                residency,
                request,
                max_vertex_payload_bytes=args.max_vertex_payload_bytes,
                check_cancelled=lambda: None,
            )
            cold_snapshot_ms = _elapsed_ms(started)
            cold_breakdown = timings.summary()
            rss_after_cold_snapshot_mib = _rss_mib()

            timings.clear()
            warm_request = _ViewportRequest(
                request_generation=2,
                selection_generation=1,
                requested_value_ids=requested_value_ids,
                viewport=viewport,
            )
            started = time.perf_counter()
            warm_snapshot = _read_viewport_snapshot(
                reader,
                selected_value_index,
                residency,
                warm_request,
                max_vertex_payload_bytes=args.max_vertex_payload_bytes,
                check_cancelled=lambda: None,
            )
            warm_snapshot_ms = _elapsed_ms(started)
            warm_breakdown = timings.summary()

        report["worker"] = {
            "cold_snapshot_ms": cold_snapshot_ms,
            "cold_breakdown": cold_breakdown,
            "rss_after_cold_snapshot_mib": rss_after_cold_snapshot_mib,
            "warm_snapshot_ms": warm_snapshot_ms,
            "warm_breakdown": warm_breakdown,
            "cpu_resident_mib": residency.resident_bytes / _MIB,
            "cpu_resident_tiles": residency.tile_count,
        }
        report["snapshot"] = {
            "level": snapshot.level,
            "level_kind": snapshot.level_kind,
            "within_budget": snapshot.within_budget,
            "estimated_point_count": snapshot.estimated_point_count,
            "tile_count": snapshot.rendered_tile_count,
            "point_count": snapshot.rendered_point_count,
            "render_batch_point_count": snapshot.render_batch.point_count,
            "render_batch_bytes": snapshot.render_batch.nbytes,
            "omitted_value_ids": snapshot.omitted_value_ids,
        }
        report["dense_exact_tile"] = _dense_exact_tile_report(reader)
        if args.synthetic_million_point_packing:
            report["synthetic_million_point_packing"] = _synthetic_million_point_packing_report(
                repeats=args.synthetic_packing_repeats
            )

        if args.real_canvas:
            application = QApplication.instance() or QApplication([])
        else:
            application = QCoreApplication.instance() or QCoreApplication([])
        report["qt_delivery"] = _measure_qt_delivery(
            application,
            warm_snapshot,
            repeats=args.qt_delivery_repeats,
        )
        if args.real_canvas:
            if not isinstance(application, QApplication):
                raise RuntimeError("A QCoreApplication already exists; a QApplication is required for --real-canvas.")
            subset_viewport = _centered_viewport(
                info,
                args.viewport_fraction / 4.0,
                args.point_budget,
                args.canvas_width,
                args.canvas_height,
            )
            subset_snapshot = _read_viewport_snapshot(
                reader,
                selected_value_index,
                residency,
                _ViewportRequest(
                    request_generation=3,
                    selection_generation=1,
                    requested_value_ids=requested_value_ids,
                    viewport=subset_viewport,
                ),
                max_vertex_payload_bytes=args.max_vertex_payload_bytes,
                check_cancelled=lambda: None,
            )
            if not subset_snapshot.within_budget:
                raise RuntimeError("The centered renderer subset unexpectedly exceeds the point budget.")
            report["renderer"] = _renderer_report(
                snapshot,
                subset_snapshot,
                info,
                viewport,
                max_vertex_payload_bytes=args.max_vertex_payload_bytes,
                warm_draw_repeats=args.warm_draw_repeats,
            )
    finally:
        reader_context.__exit__(None, None, None)

    report["rss_end_mib"] = _rss_mib()
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    _print_summary(report)


if __name__ == "__main__":
    main()
