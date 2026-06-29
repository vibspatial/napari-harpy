from __future__ import annotations

import numpy as np
from qtpy.QtCore import QObject, Signal
from spatialdata import SpatialData

import napari_harpy.widgets.histogram.controller as histogram_controller_module
from napari_harpy.core.histogram import HistogramResult, HistogramSettings, HistogramTarget
from napari_harpy.widgets.histogram.controller import (
    HISTOGRAM_IDLE_STATUS,
    HistogramController,
    HistogramJob,
    HistogramJobResult,
    _run_histogram_job,
)


class _DeferredWorker(QObject):
    returned = Signal(object)
    errored = Signal(object)
    finished = Signal()

    def __init__(self, result: HistogramJobResult | None = None) -> None:
        super().__init__()
        self._result = result
        self.started = False
        self.quit_called = False

    def start(self) -> None:
        self.started = True

    def quit(self) -> None:
        self.quit_called = True

    def emit_returned(self) -> None:
        assert self._result is not None
        self.returned.emit(self._result)
        self.finished.emit()

    def emit_errored(self, error: Exception) -> None:
        self.errored.emit(error)
        self.finished.emit()


def make_target() -> HistogramTarget:
    return HistogramTarget(coordinate_system="global", image_name="blobs_image", channel_name="0")


def make_settings(*, bins: int = 8) -> HistogramSettings:
    return HistogramSettings(bins=bins, scale="scale0")


def make_job_result(job: HistogramJob) -> HistogramJobResult:
    return HistogramJobResult(
        card_id=job.card_id,
        job_id=job.job_id,
        target=job.target,
        settings=job.settings,
        result=HistogramResult(
            target=job.target,
            settings=job.settings,
            counts=np.array([1, 2]),
            bin_edges=np.array([0.0, 0.5, 1.0]),
            data_range=(0.0, 1.0),
            percentile_values={},
            resolved_scale=job.settings.scale,
        ),
    )


def test_histogram_controller_starts_idle() -> None:
    controller = HistogramController()

    assert controller.status_message("missing") == HISTOGRAM_IDLE_STATUS
    assert controller.status_kind("missing") == "warning"
    assert controller.can_calculate("missing") is False
    assert controller.is_running("missing") is False


def test_histogram_controller_bind_invalid_state_blocks_calculate() -> None:
    state_changes: list[str] = []
    controller = HistogramController(on_state_changed=state_changes.append)

    changed = controller.bind(
        "card",
        None,
        None,
        None,
        validation_error="No SpatialData loaded.",
    )

    assert changed is True
    assert state_changes == ["card"]
    assert controller.can_calculate("card") is False
    assert controller.status_kind("card") == "warning"
    assert controller.status_message("card") == "No SpatialData loaded."
    assert controller.calculate("card") is False


def test_histogram_controller_calculate_launches_job_for_bound_card(sdata_blobs: SpatialData) -> None:
    captured_jobs: list[HistogramJob] = []
    workers: list[_DeferredWorker] = []
    controller = HistogramController()
    target = make_target()
    settings = make_settings()

    def capture_worker(job: HistogramJob) -> _DeferredWorker:
        captured_jobs.append(job)
        worker = _DeferredWorker(make_job_result(job))
        workers.append(worker)
        return worker

    controller._create_histogram_worker = capture_worker  # type: ignore[method-assign]
    controller.bind("card", sdata_blobs, target, settings)

    launched = controller.calculate("card")

    assert launched is True
    assert len(captured_jobs) == 1
    assert captured_jobs[0].card_id == "card"
    assert captured_jobs[0].job_id == "1"
    assert captured_jobs[0].sdata is sdata_blobs
    assert captured_jobs[0].target == target
    assert captured_jobs[0].settings == settings
    assert workers[0].started is True
    assert controller.is_running("card") is True
    assert controller.can_calculate("card") is False
    assert controller.status_message("card") == "Calculating histogram."

    workers[0].emit_returned()

    assert controller.is_running("card") is False
    assert controller.status_kind("card") == "success"
    assert controller.status_message("card") == "Histogram calculated."
    result = controller.result_for_card("card")
    assert result is not None
    np.testing.assert_array_equal(result.counts, np.array([1, 2]))
    np.testing.assert_array_equal(result.bin_edges, np.array([0.0, 0.5, 1.0]))
    assert result.target == target
    assert result.settings == settings


def test_histogram_controller_rebinding_card_ignores_stale_worker_result(sdata_blobs: SpatialData) -> None:
    workers: list[_DeferredWorker] = []
    controller = HistogramController()
    target = make_target()

    def capture_worker(job: HistogramJob) -> _DeferredWorker:
        worker = _DeferredWorker(make_job_result(job))
        workers.append(worker)
        return worker

    controller._create_histogram_worker = capture_worker  # type: ignore[method-assign]
    controller.bind("card", sdata_blobs, target, make_settings(bins=8))
    assert controller.calculate("card") is True

    controller.bind("card", sdata_blobs, target, make_settings(bins=16))
    workers[0].emit_returned()

    assert workers[0].quit_called is True
    assert controller.result_for_card("card") is None
    assert controller.is_running("card") is False
    assert controller.status_kind("card") == "info"
    assert controller.status_message("card") == "Target or settings changed. Calculate again."


def test_histogram_controller_worker_errors_surface_as_card_errors(sdata_blobs: SpatialData) -> None:
    workers: list[_DeferredWorker] = []
    controller = HistogramController()

    def capture_worker(job: HistogramJob) -> _DeferredWorker:
        worker = _DeferredWorker(make_job_result(job))
        workers.append(worker)
        return worker

    controller._create_histogram_worker = capture_worker  # type: ignore[method-assign]
    controller.bind("card", sdata_blobs, make_target(), make_settings())
    assert controller.calculate("card") is True

    workers[0].emit_errored(RuntimeError("boom"))

    assert controller.is_running("card") is False
    assert controller.result_for_card("card") is None
    assert controller.status_kind("card") == "error"
    assert controller.status_message("card") == "Histogram calculation failed: boom"


def test_run_histogram_job_calls_core_calculator(monkeypatch, sdata_blobs: SpatialData) -> None:
    target = make_target()
    settings = make_settings()
    job = HistogramJob(
        card_id="card",
        job_id="42",
        sdata=sdata_blobs,
        target=target,
        settings=settings,
    )
    expected_result = make_job_result(job).result
    calls: list[tuple[SpatialData, HistogramTarget, HistogramSettings]] = []

    def fake_calculate_histogram(
        sdata: SpatialData,
        request_target: HistogramTarget,
        request_settings: HistogramSettings,
    ) -> HistogramResult:
        calls.append((sdata, request_target, request_settings))
        return expected_result

    monkeypatch.setattr(histogram_controller_module.histogram_core, "calculate_histogram", fake_calculate_histogram)

    result = _run_histogram_job.__wrapped__(job)

    assert calls == [(sdata_blobs, target, settings)]
    assert result == HistogramJobResult(
        card_id="card",
        job_id="42",
        target=target,
        settings=settings,
        result=expected_result,
    )
