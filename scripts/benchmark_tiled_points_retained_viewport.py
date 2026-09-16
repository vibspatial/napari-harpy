"""Compare normal replacement with retained-batch navigation through the Qt runtime.

Example::

    .venv/bin/python scripts/benchmark_tiled_points_retained_viewport.py CACHE \
        --selection value_a --selection value_a value_b --all-values \
        --real-canvas --json-output /tmp/retained-viewports.json

The baseline disables only the worker's retained entry at the snapshot helper
boundary. Both modes use real LOD selection, CPU tile residency, storage routes,
queued delivery, generation handling, and activation feedback. With --real-canvas
they also use the real single-VBO renderer; otherwise activation is simulated
and no GPU timings are reported. Filesystem caches are not flushed. Timing hooks
add overhead, particularly on tile-heavy replacements. Frame-gap measurements
are Qt timer intervals (including explicit draw/readback work in the harness),
not hardware presentation timestamps.
"""

from __future__ import annotations

import argparse
import gc
import json
import resource
import statistics
import sys
import threading
import time
from dataclasses import asdict, replace
from pathlib import Path

from benchmark_tiled_points_cache_to_canvas import (
    _centered_viewport,
    _elapsed_ms,
    _git_state,
    _install_reader_timers,
    _rss_mib,
    _TemporaryPatches,
    _TimingLog,
)
from napari._vispy.utils.qt_font import FontInfo
from qtpy.QtCore import Qt, QTimer, Slot
from qtpy.QtWidgets import QApplication
from vispy.scene import SceneCanvas

import napari_harpy.viewer.tiled_points.runtime.cache_session as session_module
from napari_harpy.core.multi_scale_cache_points_zarr.reader import _PointsCacheReader
from napari_harpy.viewer.tiled_points.application import canonical_value_palette
from napari_harpy.viewer.tiled_points.contracts import TiledPointsDatasetReference, TiledPointsRenderResult
from napari_harpy.viewer.tiled_points.napari.layer import TiledPointsLayerModel
from napari_harpy.viewer.tiled_points.runtime.layer_runtime import _TiledPointsLayerRuntime
from napari_harpy.viewer.tiled_points.vispy.layer import VispyTiledPointsLayer


def _wait(app, predicate, *, timeout=120.0):
    deadline = time.perf_counter() + timeout
    while not predicate():
        app.processEvents()
        if time.perf_counter() > deadline:
            raise TimeoutError("Viewport benchmark timed out waiting for the runtime")
        time.sleep(0.001)


def _bounds(view):
    return [view.x_min, view.y_min, view.x_max, view.y_max]


def _trace(info, point_budget):
    full = _centered_viewport(info, 1.0, point_budget, 1200, 900)
    inner = _centered_viewport(info, 0.25, point_budget, 1200, 900)
    width = full.x_max - full.x_min
    left = replace(inner, x_min=full.x_min, x_max=full.x_min + width * 0.25)
    right = replace(inner, x_min=full.x_max - width * 0.25, x_max=full.x_max)
    almost_full = _centered_viewport(info, 0.999999, point_budget, 1200, 900)
    return [
        ("full", full),
        ("almost_full", almost_full),
        ("almost_full_return", full),
        ("inner", inner),
        ("pan_inside", left),
        ("return_full", full),
        ("equal_full", full),
        ("disjoint_left", left),
        ("disjoint_right", right),
        ("return_left", left),
    ]


def _run_case(app, cache_root, info, selection, *, retain, real_canvas, point_budget):
    records = {}
    timings = _TimingLog()
    visual = canvas = view = runtime = None
    submitted = {}
    frame_gaps = []
    last_tick = time.perf_counter()
    gc_intervals = []
    gc_started = {}

    def track_gc(phase, details):
        key = (threading.get_ident(), details["generation"])
        now = time.perf_counter()
        if phase == "start":
            gc_started[key] = now
        elif key in gc_started:
            gc_intervals.append((gc_started.pop(key), now))

    def tick():
        nonlocal last_tick
        now = time.perf_counter()
        frame_gaps.append((now - last_tick) * 1000)
        last_tick = now

    class _MeasuredCacheWorker(session_module._TiledPointsCacheWorker):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._measurement = None
            # Connect before the session installs queued GUI delivery, so the
            # timing record exists before the GUI can receive this snapshot.
            self.viewport_ready.connect(self._record_snapshot, Qt.DirectConnection)

        @Slot(object)
        def read_viewport_snapshot(self, request):
            timings.clear()
            self._measurement = (request, self._retained_viewport, time.perf_counter())
            try:
                super().read_viewport_snapshot(request)
            finally:
                self._measurement = None

        def _record_snapshot(self, snapshot):
            request, accepted, started = self._measurement
            retained = accepted if retain else None
            finished = time.perf_counter()
            reused = retained is not None and snapshot.render_batch is retained.snapshot.render_batch
            records[request.request_generation] = {
                "request_generation": request.request_generation,
                "viewport": _bounds(request.viewport),
                "original_bounds": _bounds(retained.bounds) if reused else _bounds(request.viewport),
                "previous_bounds": None if accepted is None else _bounds(accepted.bounds),
                "reuse_rejection": "disabled"
                if not retain
                else "no_accepted_batch"
                if retained is None
                else retained.rejection_reason(
                    request,
                    cache_generation_id=snapshot.cache_generation_id,
                    level=snapshot.level,
                    max_vertex_payload_bytes=self._settings.max_vertex_payload_bytes,
                ),
                "worker_reused_batch": reused,
                "worker_ms": (finished - started) * 1000,
                "gc_overlap_ms": sum(
                    max(0.0, min(finished, stop) - max(started, start)) * 1000 for start, stop in gc_intervals
                ),
                "worker_finished": finished,
                "batch_identity": id(snapshot.render_batch),
                "vertices_identity": id(snapshot.render_batch.vertices),
                "level": snapshot.level,
                "level_kind": snapshot.level_kind,
                "within_budget": snapshot.within_budget,
                "visible_estimate": snapshot.estimated_point_count,
                "payload_points": snapshot.rendered_point_count,
                "payload_tiles": snapshot.rendered_tile_count,
                "payload_bytes": snapshot.render_batch.nbytes,
                "omitted_value_ids": snapshot.omitted_value_ids,
                "previous_retained_bytes": 0 if accepted is None else accepted.snapshot.render_batch.nbytes,
                "transient_candidate_bytes": 0 if reused else snapshot.render_batch.nbytes,
                "breakdown": timings.summary(),
                "activated": False,
            }
            if not snapshot.within_budget:
                records[request.request_generation]["outcome"] = "metadata_only_over_budget"
                records[request.request_generation]["reuse_rejection"] = "visible_budget"
            # Exclude building the benchmark record itself from queued-delivery time.
            records[request.request_generation]["worker_finished"] = time.perf_counter()

    def delivered(event):
        snapshot = event.value
        record = records[snapshot.request_generation]
        record["qt_delivery_ms"] = _elapsed_ms(record["worker_finished"])
        record["same_allocation_after_qt"] = id(snapshot.render_batch.vertices) == record["vertices_identity"]
        record["activation_started"] = time.perf_counter()
        record["replacements_before"] = 0 if visual is None else visual.payload_replacement_count
        if visual is None:
            layer.events.render_snapshot_result(
                value=TiledPointsRenderResult(snapshot.request_generation, snapshot.selection_generation, True)
            )

    def activated(event):
        result = event.value
        record = records[result.request_generation]
        record["activated"] = result.applied
        record["activation_ms"] = _elapsed_ms(record["activation_started"])
        record["end_to_end_activation_ms"] = _elapsed_ms(submitted[result.request_generation][1])
        if visual is not None:
            uploads = visual.payload_replacement_count - record["replacements_before"]
            record["vbo_uploads"] = uploads
            record["vbo_bytes"] = record["payload_bytes"] if uploads else 0
            record["vbo_staging_ms"] = visual.last_vertex_staging_ms if uploads else 0.0
            record["point_draws_per_frame"] = visual.point_draw_submission_count
        record["outcome"] = (
            "active_batch_reuse"
            if record["worker_reused_batch"] and record.get("vbo_uploads", 0) == 0
            else "viewport_replacement"
        )

    layer = TiledPointsLayerModel(
        TiledPointsDatasetReference(
            **{
                name: getattr(info, name)
                for name in (
                    "cache_generation_id",
                    "points_name",
                    "value_column",
                    "x_origin",
                    "y_origin",
                    "x_min",
                    "x_max",
                    "y_min",
                    "y_max",
                )
            },
            value_count=len(info.value_names),
        ),
        value_palette=canonical_value_palette(len(info.value_names)),
        max_vertex_payload_bytes=512 << 20,
        hard_render_point_budget=point_budget,
    )
    timer = QTimer()
    timer.setInterval(16)
    timer.timeout.connect(tick)
    with _TemporaryPatches() as patches:
        _install_reader_timers(timings, patches)
        patches.patch(session_module, "_TiledPointsCacheWorker", _MeasuredCacheWorker)
        if not retain:
            # Disable only the reuse decision. The worker still evaluates LOD,
            # retains decoded tiles and processes activation acknowledgements.
            patches.patch(session_module._RetainedViewport, "rejection_reason", lambda *args, **kwargs: "disabled")
        runtime = _TiledPointsLayerRuntime(
            layer,
            cache_root,
            session_module._CacheSessionSettings(None, 1 << 30, 512 << 20),
            initial_requested_value_ids=selection,
        )
        try:
            layer.events.render_snapshot.connect(delivered)
            if real_canvas:
                canvas = SceneCanvas(show=True, size=(1200, 900))
                view = canvas.central_widget.add_view()
                view.camera = "panzoom"
                visual = VispyTiledPointsLayer(layer, FontInfo())
                view.add(visual.node)
                canvas.render()
            layer.events.render_snapshot_result.connect(activated)
            _wait(
                app,
                lambda: (
                    runtime.state is session_module._CacheSessionState.READY
                    and not runtime._viewport_scheduler.selection_update_pending
                ),
            )
            # Reclaim prior cases outside timed navigation. Automatic collection
            # stays enabled; report any pauses overlapping worker measurements.
            gc.collect()
            gc.callbacks.append(track_gc)
            last_tick = time.perf_counter()
            timer.start()

            def submit(label, viewport):
                generation = runtime._viewport_scheduler.request_generation + 1
                submitted[generation] = (label, time.perf_counter())
                if view is not None:
                    # Viewport bounds are already intrinsic source coordinates;
                    # the visual adds the cache origin to its relative vertices.
                    view.camera.rect = (
                        viewport.x_min,
                        viewport.y_min,
                        viewport.x_max - viewport.x_min,
                        viewport.y_max - viewport.y_min,
                    )
                runtime._viewport_scheduler.submit_viewport(viewport)
                return generation

            def idle():
                return (
                    runtime._viewport_scheduler.active_request_generation is None
                    and runtime._viewport_scheduler.pending_request_generation is None
                )

            for label, viewport in _trace(info, point_budget):
                # Start a genuinely disjoint sequence with a fresh packed entry.
                # A full-extent active entry would otherwise cover both regions.
                if label == "disjoint_left":
                    outside = replace(
                        viewport,
                        x_min=viewport.x_min - (info.x_max - info.x_min) * 2,
                        x_max=viewport.x_max - (info.x_max - info.x_min) * 2,
                    )
                    submit("leave_full_extent", outside)
                    _wait(app, idle)
                generation = submit(label, viewport)
                _wait(app, idle)
                record = records.get(generation)
                if record is None:
                    raise RuntimeError(f"Request failed: {layer.display_status.message}")
                if record["within_budget"] and not record["activated"]:
                    raise RuntimeError(f"Activation failed: {layer.display_status.message}")
                if canvas is not None:
                    started = time.perf_counter()
                    canvas.render()
                    # Qt may already have painted during queued delivery. This
                    # is the first explicit readback, not necessarily first GL draw.
                    record["first_explicit_draw_ms"] = _elapsed_ms(started)
                    draws = []
                    for _ in range(3):
                        started = time.perf_counter()
                        canvas.render()
                        draws.append(_elapsed_ms(started))
                    record["warm_draw_median_ms"] = statistics.median(draws)
                record["rss_mib"] = _rss_mib()

            # Rapid requests intentionally outpace the worker; distinguish skipped
            # submissions and obsolete dispatched work from accepted-view timings.
            full = _trace(info, point_budget)[0][1]
            width = (full.x_max - full.x_min) * 0.15
            for i in range(24):
                start = full.x_min + (full.x_max - full.x_min) * i / 24
                submit(f"fast_pan_{i}", replace(full, x_min=start, x_max=start + width))
                app.processEvents()
                time.sleep(0.003)
            _wait(app, idle)
        finally:
            timer.stop()
            if track_gc in gc.callbacks:
                gc.callbacks.remove(track_gc)
            runtime.close()
            _wait(app, lambda: runtime.state is session_module._CacheSessionState.CLOSED)
            if visual is not None:
                visual.close()
            if canvas is not None:
                canvas.close()

    for generation, record in records.items():
        record["label"] = submitted[generation][0]
        record["obsolete"] = record["within_budget"] and not record["activated"]
        for name in (
            "worker_finished",
            "activation_started",
            "replacements_before",
            "batch_identity",
            "vertices_identity",
        ):
            record.pop(name, None)
    return {
        "retention_enabled": retain,
        "requests": list(records.values()),
        "coalesced_submissions": len(submitted) - len(records),
        "qt_timer_gap_max_ms": max(frame_gaps, default=0),
        "qt_timer_gap_median_ms": statistics.median(frame_gaps) if frame_gaps else None,
    }


def main():
    """Run paired Qt navigation traces and persist timing and allocation evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache_root", type=Path)
    parser.add_argument("--selection", nargs="+", action="append", default=[])
    parser.add_argument("--all-values", action="store_true")
    parser.add_argument("--point-budget", type=int, default=100_000)
    parser.add_argument("--real-canvas", action="store_true")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--json-output", type=Path, required=True)
    args = parser.parse_args()
    if args.point_budget <= 0 or args.repeats <= 0 or not (args.selection or args.all_values):
        parser.error("Supply a selection or --all-values and a positive point budget.")
    app = QApplication.instance() or QApplication([])
    with _PointsCacheReader(args.cache_root) as reader:
        info = reader.dataset_info
    selections = [(names, tuple(sorted({info.value_names.index(name) for name in names}))) for names in args.selection]
    if args.all_values:
        selections.append((None, None))
    report = {
        "cache_root": str(args.cache_root),
        "git": _git_state(),
        "dataset": asdict(info),
        "real_canvas": args.real_canvas,
        "cases": [],
    }
    for names, ids in selections:
        case = {"values": names, "modes": []}
        for repeat in range(args.repeats):
            # Alternate mode order so cold filesystem/allocator effects do not
            # systematically favor either branch of the comparison.
            for retain in (False, True) if repeat % 2 == 0 else (True, False):
                result = _run_case(
                    app,
                    args.cache_root,
                    info,
                    ids,
                    retain=retain,
                    real_canvas=args.real_canvas,
                    point_budget=args.point_budget,
                )
                result["repeat"] = repeat
                case["modes"].append(result)
                print(f"Completed selection={names}, retention={retain}, repeat={repeat}", flush=True)
        report["cases"].append(case)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report["peak_rss_mib"] = peak / ((1 << 20) if sys.platform == "darwin" else 1024)
    args.json_output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
