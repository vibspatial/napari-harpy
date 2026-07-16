from __future__ import annotations

import threading

import numpy as np
import pytest
from qtpy.QtCore import QObject, Signal
from spatialdata import SpatialData

import napari_harpy.widgets.spatial_query.controller as controller_module
from napari_harpy._app_state import HarpyAppState, TableStateChangedEvent
from napari_harpy.core.spatial_query import (
    CANONICAL_CACHE_PATHS,
    CANONICAL_OBSM_KEY,
    SPATIAL_COORDINATES_KEY,
    CanonicalCacheReport,
    CanonicalCacheUpdateAction,
    CanonicalCacheUpdatePayload,
    CanonicalCentersResult,
    apply_canonical_cache_update,
    build_canonical_cache_update_payload,
    inspect_canonical_cache,
)
from napari_harpy.widgets.spatial_query.cache_state import (
    CANONICAL_CACHE_UPDATE_SOURCE,
    record_canonical_cache_update,
)
from napari_harpy.widgets.spatial_query.controller import (
    SpatialQueryController,
    _run_canonical_centers_calculation,
)


class _DeferredWorker(QObject):
    returned = Signal(object)
    errored = Signal(object)
    finished = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.started = False
        self.quit_called = False

    def start(self) -> None:
        self.started = True

    def quit(self) -> None:
        self.quit_called = True

    def emit_returned(self, payload: CanonicalCacheUpdatePayload) -> None:
        self.returned.emit(payload)
        self.finished.emit()

    def emit_errored(self, error: Exception) -> None:
        self.errored.emit(error)
        self.finished.emit()


def _report(sdata: SpatialData) -> CanonicalCacheReport:
    return inspect_canonical_cache(sdata, table_name="table", labels_name="blobs_labels")


def _payload(report: CanonicalCacheReport) -> CanonicalCacheUpdatePayload:
    centers = np.zeros((report.binding.n_obs, 3), dtype=np.float64)
    centers[:, 1] = np.arange(report.binding.n_obs, dtype=np.float64) + 0.25
    centers[:, 2] = np.arange(report.binding.n_obs, dtype=np.float64) + 0.75
    return build_canonical_cache_update_payload(
        binding=report.binding,
        centers=centers,
        source_signature=report.source_signature,
    )


def test_canonical_centers_worker_returns_payload_without_mutating_cache(
    sdata_blobs: SpatialData,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _report(sdata_blobs)
    payload = _payload(report)
    table = sdata_blobs.tables["table"]
    monkeypatch.setattr(controller_module, "calculate_canonical_centers", lambda sdata, current: payload)

    result = _run_canonical_centers_calculation.__wrapped__(sdata_blobs, report)

    assert result is payload
    assert CANONICAL_OBSM_KEY not in table.obsm
    assert SPATIAL_COORDINATES_KEY not in table.uns


def test_controller_calculates_off_thread_and_applies_current_payload_on_main_thread(
    qtbot,
    sdata_blobs: SpatialData,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _report(sdata_blobs)
    payload = _payload(report)
    table = sdata_blobs.tables["table"]
    main_thread = threading.get_ident()
    calculation_threads: list[int] = []
    application_threads: list[int] = []
    accepted_results = []
    app_state = HarpyAppState()
    emitted_events: list[TableStateChangedEvent] = []
    app_state.table_state_changed.connect(emitted_events.append)
    real_apply = apply_canonical_cache_update

    def calculate(sdata: SpatialData, current_report: CanonicalCacheReport) -> CanonicalCacheUpdatePayload:
        assert sdata is sdata_blobs
        assert current_report is report
        calculation_threads.append(threading.get_ident())
        return payload

    def apply(sdata: SpatialData, current_payload: CanonicalCacheUpdatePayload):
        application_threads.append(threading.get_ident())
        return real_apply(sdata, current_payload)

    monkeypatch.setattr(controller_module, "calculate_canonical_centers", calculate)
    monkeypatch.setattr(controller_module, "apply_canonical_cache_update", apply)

    def accept(result) -> None:
        accepted_results.append(result)
        record_canonical_cache_update(app_state, sdata_blobs, result)

    controller = SpatialQueryController(on_centers_ready=accept)

    assert controller.start_canonical_centers_calculation(sdata_blobs, report)
    assert controller.is_running
    assert controller.active_phase == "canonical_centers"
    assert controller.status_message == "Spatial query: calculating canonical centers."

    qtbot.waitUntil(lambda: not controller.is_running, timeout=5000)

    assert calculation_threads and calculation_threads[0] != main_thread
    assert application_threads == [main_thread]
    assert len(accepted_results) == 1
    assert accepted_results[0] is controller.centers_result
    assert accepted_results[0].cache_update.action is CanonicalCacheUpdateAction.CREATE
    assert controller.status_kind == "success"
    assert CANONICAL_OBSM_KEY in table.obsm
    assert SPATIAL_COORDINATES_KEY in table.uns
    assert len(emitted_events) == 1
    assert emitted_events[0].paths == CANONICAL_CACHE_PATHS
    assert emitted_events[0].regions == ("blobs_labels",)
    assert emitted_events[0].change_kind == "created"
    assert emitted_events[0].source == CANONICAL_CACHE_UPDATE_SOURCE
    assert app_state.snapshot_table_dirty_state(sdata_blobs, "table").paths == CANONICAL_CACHE_PATHS

    reused_result = CanonicalCentersResult(
        source_signature=payload.source_signature,
        binding=payload.binding,
        centers=payload.centers,
        cache_update=None,
    )
    assert record_canonical_cache_update(app_state, sdata_blobs, reused_result) is None
    assert len(emitted_events) == 1


def test_controller_ignores_cancelled_worker_signals_and_accepts_new_operation(
    sdata_blobs: SpatialData,
) -> None:
    report = _report(sdata_blobs)
    payload = _payload(report)
    table = sdata_blobs.tables["table"]
    first_worker = _DeferredWorker()
    second_worker = _DeferredWorker()
    workers = [first_worker, second_worker]
    accepted_results = []
    controller = SpatialQueryController(on_centers_ready=accepted_results.append)
    controller._create_canonical_centers_worker = (  # type: ignore[method-assign]
        lambda sdata, current_report: workers.pop(0)
    )

    assert controller.start_canonical_centers_calculation(sdata_blobs, report)
    assert first_worker.started
    assert controller.operation_id == 1
    assert controller.cancel_active_operation()
    assert first_worker.quit_called
    assert controller.operation_id == 1

    assert controller.start_canonical_centers_calculation(sdata_blobs, report)
    assert second_worker.started
    assert controller.operation_id == 2

    first_worker.emit_returned(payload)
    assert CANONICAL_OBSM_KEY not in table.obsm
    assert accepted_results == []
    assert controller.is_running

    second_worker.emit_returned(payload)
    assert CANONICAL_OBSM_KEY in table.obsm
    assert len(accepted_results) == 1
    assert not controller.is_running


def test_controller_routes_current_worker_error_without_mutation(sdata_blobs: SpatialData) -> None:
    report = _report(sdata_blobs)
    table = sdata_blobs.tables["table"]
    worker = _DeferredWorker()
    controller = SpatialQueryController()
    controller._create_canonical_centers_worker = (  # type: ignore[method-assign]
        lambda sdata, current_report: worker
    )

    assert controller.start_canonical_centers_calculation(sdata_blobs, report)
    worker.emit_errored(RuntimeError("aggregation failed"))

    assert not controller.is_running
    assert controller.centers_result is None
    assert controller.status_kind == "error"
    assert "aggregation failed" in controller.status_message
    assert CANONICAL_OBSM_KEY not in table.obsm
    assert SPATIAL_COORDINATES_KEY not in table.uns
