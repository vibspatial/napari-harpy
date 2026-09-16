"""Test worker LOD decisions using small real Zarr caches.

Screen density is a soft preference; point and vertex-byte limits are
hard constraints. Cover coarsest-level fallback, budget-driven LOD choice,
and hard-limit rejection before point-payload work.

Also verify that changing the preferred density updates the status message
without repacking when the retained batch remains reusable, and that byte
limits smaller than one vertex are rejected.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import napari_harpy.viewer.tiled_points.runtime.cache_session as cache_session_module
from napari_harpy.core.multi_scale_cache_points_zarr.reader import _PointsCacheReader
from napari_harpy.viewer.tiled_points.contracts import (
    TILED_POINTS_VERTEX_DTYPE,
    TiledPointsRenderResult,
    TiledPointsViewportState,
    _ViewportRequest,
)
from napari_harpy.viewer.tiled_points.runtime.cache_session import _CacheSessionSettings, _TiledPointsCacheWorker

_VERTEX_BYTES = TILED_POINTS_VERTEX_DTYPE.itemsize


@contextmanager
def _worker(cache_root: Path, *, max_bytes: int, selection=None):
    """Exercise LOD policy at its worker boundary, with a real cache reader."""
    worker = _TiledPointsCacheWorker(
        cache_root, _CacheSessionSettings(None, 1000, max_bytes), threading.Event(), _PointsCacheReader
    )
    failures = []
    worker.failed.connect(failures.append)
    try:
        worker.start()
        worker.update_selected_value_index(selection)
        assert not failures
        assert worker._reader is not None
        yield worker
    finally:
        worker.close()


def _read_snapshot(worker, request):
    snapshots = []
    failures = []

    def failed(_generation, failure):
        failures.append(failure)

    worker.viewport_ready.connect(snapshots.append)
    worker.viewport_failed.connect(failed)
    try:
        worker.read_viewport_snapshot(request)
        assert not failures
        assert len(snapshots) == 1
        return snapshots[0]
    finally:
        worker.viewport_ready.disconnect(snapshots.append)
        worker.viewport_failed.disconnect(failed)


def _request(*, hard_points: int = 4, density: int = 1) -> _ViewportRequest:
    return _ViewportRequest(
        request_generation=1,
        selection_generation=0,
        requested_value_ids=None,
        viewport=TiledPointsViewportState(
            displayed_axes=(0, 1),
            x_min=0.0,
            y_min=0.0,
            x_max=20.0,
            y_max=10.0,
            canvas_width=100,
            canvas_height=100,
            hard_render_point_budget=hard_points,
            screen_density_budget=density,
        ),
    )


@pytest.mark.parametrize("selection", [None, (0,)], ids=["all-values", "subset"])
def test_coarsest_density_fallback_accepts_payload_at_both_hard_limits(real_cache_root: Path, selection) -> None:
    """A density target below every level must not reject an otherwise permitted payload."""
    point_count = 4 if selection is None else 2
    request = replace(_request(hard_points=point_count), requested_value_ids=selection)
    with _worker(real_cache_root, max_bytes=point_count * _VERTEX_BYTES, selection=selection) as worker:
        snapshot = _read_snapshot(worker, request)
        assert snapshot.level == worker._reader.level_count - 1
    assert snapshot.within_budget
    assert snapshot.rendered_point_count == snapshot.estimated_point_count == point_count
    assert snapshot.render_batch.nbytes == point_count * _VERTEX_BYTES
    assert "above preferred screen density" in snapshot.budget_message
    assert f"{point_count} points; target 1" in snapshot.budget_message
    if selection is not None:
        np.testing.assert_array_equal(snapshot.render_batch.vertices["a_value_id"], np.zeros(point_count))


@pytest.mark.parametrize("selection", [None, (0,)], ids=["all-values", "subset"])
@pytest.mark.parametrize("limiting_budget", ["points", "bytes", "both"])
def test_hard_budget_rejection_reports_actual_limits_before_tile_work(
    real_cache_root: Path, selection, limiting_budget: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    point_count = 4 if selection is None else 2
    hard_points = point_count - 1 if limiting_budget in ("points", "both") else point_count
    byte_limit = point_count * _VERTEX_BYTES
    if limiting_budget in ("bytes", "both"):
        byte_limit -= 1
    request = replace(_request(hard_points=hard_points), requested_value_ids=selection)

    def forbidden(*args, **kwargs):
        raise AssertionError("A hard-budget rejection must not plan, read, or pack tiles.")

    with _worker(real_cache_root, max_bytes=byte_limit, selection=selection) as worker:
        monkeypatch.setattr(worker._reader, "plan_viewport", forbidden)
        monkeypatch.setattr(worker._reader, "read_planned_tiles", forbidden)
        monkeypatch.setattr(cache_session_module, "_read_viewport_snapshot", forbidden)
        monkeypatch.setattr(cache_session_module, "pack_render_tiles", forbidden)
        snapshot = _read_snapshot(worker, request)
        assert worker._pending_viewport is None
    assert not snapshot.within_budget
    assert snapshot.rendered_point_count == snapshot.rendered_tile_count == 0
    assert snapshot.estimated_point_count == point_count
    if point_count > hard_points:
        assert f"{point_count} points required, render limit {hard_points} points" in snapshot.budget_message
    if point_count * _VERTEX_BYTES > byte_limit:
        assert (
            f"{point_count * _VERTEX_BYTES} vertex bytes required, limit {byte_limit} bytes" in snapshot.budget_message
        )


@pytest.mark.parametrize(
    ("density", "max_bytes", "expected_level"),
    [(4, 4 * _VERTEX_BYTES, 0), (2, 4 * _VERTEX_BYTES, -1), (4, 2 * _VERTEX_BYTES, -1)],
    ids=["finest-fit", "density-prefers-coarser", "byte-limit-requires-coarser"],
)
def test_lod_choice_obeys_density_preference_and_vertex_capacity(
    sampled_cache_root: Path, density: int, max_bytes: int, expected_level: int
) -> None:
    with _worker(sampled_cache_root, max_bytes=max_bytes) as worker:
        snapshot = _read_snapshot(worker, _request(density=density))
        assert snapshot.level == (worker._reader.level_count - 1 if expected_level == -1 else expected_level)
    assert snapshot.within_budget
    assert snapshot.rendered_point_count == (4 if expected_level == 0 else 2)
    assert snapshot.render_batch.nbytes <= max_bytes
    assert snapshot.budget_message is None


def test_retained_density_fallback_refreshes_status_without_repacking(
    sampled_cache_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _request(hard_points=2)
    with _worker(sampled_cache_root, max_bytes=24) as worker:
        first = _read_snapshot(worker, request)
        assert "above preferred screen density" in first.budget_message
        worker.acknowledge_render_result(TiledPointsRenderResult(1, 0, True))

        def forbidden(*args, **kwargs):
            raise AssertionError("A retained density fallback must not repeat tile work.")

        monkeypatch.setattr(worker._reader, "plan_viewport", forbidden)
        monkeypatch.setattr(worker._reader, "read_planned_tiles", forbidden)
        monkeypatch.setattr(worker._cpu_tile_residency, "get", forbidden)
        monkeypatch.setattr(cache_session_module, "_read_viewport_snapshot", forbidden)
        monkeypatch.setattr(cache_session_module, "pack_render_tiles", forbidden)
        for generation, density in enumerate((2, 1), start=2):
            current = replace(
                request,
                request_generation=generation,
                viewport=replace(request.viewport, screen_density_budget=density),
            )
            snapshot = _read_snapshot(worker, current)
            assert snapshot.render_batch is first.render_batch
            assert snapshot.request_generation == generation
            assert (snapshot.budget_message is None) == (density == 2)
            worker.acknowledge_render_result(TiledPointsRenderResult(generation, 0, True))


@pytest.mark.parametrize("byte_limit", [1, _VERTEX_BYTES - 1])
def test_worker_rejects_byte_limit_below_one_vertex_before_opening_reader(
    real_cache_root: Path, byte_limit: int
) -> None:
    with pytest.raises(ValueError, match=rf"max_vertex_payload_bytes.*at least {_VERTEX_BYTES} bytes"):
        with _worker(real_cache_root, max_bytes=byte_limit):
            pytest.fail("An invalid byte limit must not allow the worker to start.")
