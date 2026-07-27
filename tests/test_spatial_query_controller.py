from __future__ import annotations

import threading

import numpy as np
import pytest
from qtpy.QtCore import QObject, Signal
from spatialdata import SpatialData

import napari_harpy.widgets.spatial_query.controller as controller_module
from napari_harpy.core.spatial_query import (
    CANONICAL_OBSM_KEY,
    SPATIAL_COORDINATES_KEY,
    CanonicalCacheReport,
    CanonicalCacheUpdatePayload,
    CanonicalCenterQueryResult,
    CanonicalCentersResult,
    apply_canonical_cache_update,
    build_canonical_cache_update_payload,
    ensure_canonical_centers,
    inspect_canonical_cache,
)
from napari_harpy.widgets.spatial_query.controller import (
    SpatialQueryController,
    _run_canonical_center_query,
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

    def emit_returned(self, value: object) -> None:
        self.returned.emit(value)
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


def _query_result(centers: CanonicalCentersResult, *, count: int) -> CanonicalCenterQueryResult:
    return CanonicalCenterQueryResult(
        canonical_centers=centers,
        matched_instance_ids=centers.binding.instance_ids[:count],
    )


def test_worker_wrappers_return_domain_results_without_mutating_table(
    sdata_blobs: SpatialData,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _report(sdata_blobs)
    payload = _payload(report)
    table = sdata_blobs.tables["table"]
    centers = CanonicalCentersResult(
        source_signature=payload.source_signature,
        binding=payload.binding,
        centers=payload.centers,
        cache_update=None,
    )
    query_result = _query_result(centers, count=2)
    request = object()
    monkeypatch.setattr(controller_module, "calculate_canonical_centers", lambda sdata, current: payload)
    monkeypatch.setattr(controller_module, "evaluate_canonical_center_query", lambda current: query_result)

    calculated = _run_canonical_centers_calculation.__wrapped__(sdata_blobs, report)
    queried = _run_canonical_center_query.__wrapped__(request)

    assert calculated is payload
    assert queried is query_result
    assert CANONICAL_OBSM_KEY not in table.obsm
    assert SPATIAL_COORDINATES_KEY not in table.uns


def test_controller_runs_sequential_workers_and_applies_cache_on_main_thread(
    qtbot,
    sdata_blobs: SpatialData,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _report(sdata_blobs)
    payload = _payload(report)
    main_thread = threading.get_ident()
    calculation_threads: list[int] = []
    query_threads: list[int] = []
    application_threads: list[int] = []
    accepted_centers: list[CanonicalCentersResult] = []
    accepted_queries: list[CanonicalCenterQueryResult] = []
    real_apply = apply_canonical_cache_update

    def calculate(sdata: SpatialData, current_report: CanonicalCacheReport) -> CanonicalCacheUpdatePayload:
        assert sdata is sdata_blobs
        assert current_report is report
        calculation_threads.append(threading.get_ident())
        return payload

    def apply(sdata: SpatialData, current_payload: CanonicalCacheUpdatePayload):
        application_threads.append(threading.get_ident())
        return real_apply(sdata, current_payload)

    def evaluate(request: object) -> CanonicalCenterQueryResult:
        del request
        query_threads.append(threading.get_ident())
        return _query_result(accepted_centers[0], count=2)

    monkeypatch.setattr(controller_module, "calculate_canonical_centers", calculate)
    monkeypatch.setattr(controller_module, "apply_canonical_cache_update", apply)
    monkeypatch.setattr(controller_module, "build_canonical_center_query_request", lambda *args, **kwargs: object())
    monkeypatch.setattr(controller_module, "evaluate_canonical_center_query", evaluate)

    controller = SpatialQueryController(
        on_centers_ready=accepted_centers.append,
        on_query_ready=accepted_queries.append,
    )

    assert controller.start_spatial_query(
        sdata_blobs,
        report,
        shapes_name="blobs_circles",
        coordinate_system="global",
    )
    assert controller.active_phase == "canonical_centers"
    assert controller.operation_id == 1
    assert not controller.start_spatial_query(
        sdata_blobs,
        report,
        shapes_name="blobs_circles",
        coordinate_system="global",
    )
    assert controller.operation_id == 1

    qtbot.waitUntil(lambda: not controller.is_running, timeout=5000)

    assert calculation_threads and calculation_threads[0] != main_thread
    assert query_threads and query_threads[0] != main_thread
    assert application_threads == [main_thread]
    assert len(accepted_centers) == 1
    assert accepted_centers[0].cache_update is not None
    assert len(accepted_queries) == 1
    assert accepted_queries[0].matched_instance_count == 2
    assert controller.status_kind == "success"
    assert controller.status_message == "2 instance centroids found."
    assert CANONICAL_OBSM_KEY in sdata_blobs.tables["table"].obsm
    assert SPATIAL_COORDINATES_KEY in sdata_blobs.tables["table"].uns


def test_valid_cache_skips_center_worker_and_starts_query_phase(
    sdata_blobs: SpatialData,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ensure_canonical_centers(
        sdata_blobs,
        table_name="table",
        labels_name="blobs_labels",
    )
    report = _report(sdata_blobs)
    query_worker = _DeferredWorker()
    accepted_centers: list[CanonicalCentersResult] = []
    accepted_queries: list[CanonicalCenterQueryResult] = []
    controller = SpatialQueryController(
        on_centers_ready=accepted_centers.append,
        on_query_ready=accepted_queries.append,
    )
    controller._create_canonical_centers_worker = (  # type: ignore[method-assign]
        lambda sdata, current_report: pytest.fail("A valid cache must not start center calculation.")
    )
    controller._create_containment_query_worker = (  # type: ignore[method-assign]
        lambda request: query_worker
    )
    monkeypatch.setattr(controller_module, "build_canonical_center_query_request", lambda *args, **kwargs: object())

    assert controller.start_spatial_query(
        sdata_blobs,
        report,
        shapes_name="blobs_circles",
        coordinate_system="global",
    )

    assert query_worker.started
    assert controller.active_phase == "containment_query"
    assert len(accepted_centers) == 1
    assert accepted_centers[0].reused
    query_worker.emit_returned(_query_result(accepted_centers[0], count=1))

    assert not controller.is_running
    assert len(accepted_queries) == 1
    assert accepted_queries[0].matched_instance_count == 1
    assert controller.status_message == "1 instance centroid found."


def test_center_phase_finished_signal_cannot_clear_installed_query_phase(
    sdata_blobs: SpatialData,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _report(sdata_blobs)
    payload = _payload(report)
    center_worker = _DeferredWorker()
    query_worker = _DeferredWorker()
    controller = SpatialQueryController()
    controller._create_canonical_centers_worker = (  # type: ignore[method-assign]
        lambda sdata, current_report: center_worker
    )
    controller._create_containment_query_worker = (  # type: ignore[method-assign]
        lambda request: query_worker
    )
    monkeypatch.setattr(controller_module, "build_canonical_center_query_request", lambda *args, **kwargs: object())

    assert controller.start_spatial_query(
        sdata_blobs,
        report,
        shapes_name="blobs_circles",
        coordinate_system="global",
    )

    # emit_returned() emits the center worker's `finished` signal after its
    # returned callback has installed the query phase.
    center_worker.emit_returned(payload)

    assert query_worker.started
    assert controller.is_running
    assert controller.active_phase == "containment_query"


def test_controller_ignores_cancelled_center_result_and_accepts_new_operation(
    sdata_blobs: SpatialData,
) -> None:
    report = _report(sdata_blobs)
    payload = _payload(report)
    first_worker = _DeferredWorker()
    second_worker = _DeferredWorker()
    workers = [first_worker, second_worker]
    controller = SpatialQueryController()
    controller._create_canonical_centers_worker = (  # type: ignore[method-assign]
        lambda sdata, current_report: workers.pop(0)
    )

    assert controller.start_spatial_query(
        sdata_blobs,
        report,
        shapes_name="blobs_circles",
        coordinate_system="global",
    )
    assert controller.cancel_active_operation()
    assert first_worker.quit_called

    assert controller.start_spatial_query(
        sdata_blobs,
        report,
        shapes_name="blobs_circles",
        coordinate_system="global",
    )
    assert controller.operation_id == 2

    first_worker.emit_returned(payload)
    assert CANONICAL_OBSM_KEY not in sdata_blobs.tables["table"].obsm
    assert controller.active_phase == "canonical_centers"
    assert second_worker.started


def test_zero_match_query_does_not_publish_query_ready(
    sdata_blobs: SpatialData,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ensure_canonical_centers(
        sdata_blobs,
        table_name="table",
        labels_name="blobs_labels",
    )
    report = _report(sdata_blobs)
    query_worker = _DeferredWorker()
    accepted_centers: list[CanonicalCentersResult] = []
    accepted_queries: list[CanonicalCenterQueryResult] = []
    controller = SpatialQueryController(
        on_centers_ready=accepted_centers.append,
        on_query_ready=accepted_queries.append,
    )
    controller._create_containment_query_worker = (  # type: ignore[method-assign]
        lambda request: query_worker
    )
    monkeypatch.setattr(controller_module, "build_canonical_center_query_request", lambda *args, **kwargs: object())
    controller.start_spatial_query(
        sdata_blobs,
        report,
        shapes_name="blobs_circles",
        coordinate_system="global",
    )

    query_worker.emit_returned(_query_result(accepted_centers[0], count=0))

    assert accepted_queries == []
    assert controller.status_kind == "info"
    assert (
        controller.status_message
        == "No instances from the selected Labels element have their center inside the selected Shapes."
    )


def test_active_query_worker_error_stops_operation_without_delivering_result(
    sdata_blobs: SpatialData,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ensure_canonical_centers(
        sdata_blobs,
        table_name="table",
        labels_name="blobs_labels",
    )
    report = _report(sdata_blobs)
    query_worker = _DeferredWorker()
    accepted_queries: list[CanonicalCenterQueryResult] = []
    controller = SpatialQueryController(on_query_ready=accepted_queries.append)
    controller._create_containment_query_worker = (  # type: ignore[method-assign]
        lambda request: query_worker
    )
    monkeypatch.setattr(controller_module, "build_canonical_center_query_request", lambda *args, **kwargs: object())
    controller.start_spatial_query(
        sdata_blobs,
        report,
        shapes_name="blobs_circles",
        coordinate_system="global",
    )

    query_worker.emit_errored(RuntimeError("predicate failed"))

    assert not controller.is_running
    assert accepted_queries == []
    assert controller.status_kind == "error"
    assert "predicate failed" in controller.status_message
