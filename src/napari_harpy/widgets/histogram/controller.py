from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, Literal

import napari_harpy.core.histogram as histogram_core
from napari_harpy.core.histogram import HistogramResult, HistogramSettings, HistogramTarget

if TYPE_CHECKING:
    from spatialdata import SpatialData


def _resolve_thread_worker() -> Any:
    try:
        from napari.qt.threading import thread_worker
    except Exception:  # pragma: no cover - fallback for sandboxed test imports  # noqa: BLE001
        from superqt.utils import thread_worker

    return thread_worker


HistogramStatusKind = Literal["info", "warning", "success", "error"]
HISTOGRAM_IDLE_STATUS = "Histogram: choose a complete image target."

thread_worker = _resolve_thread_worker()


@dataclass(frozen=True)
class HistogramJob:
    """Immutable histogram payload copied on the main thread and consumed by a worker."""

    card_id: str
    job_id: str
    sdata: SpatialData
    target: HistogramTarget
    settings: HistogramSettings


@dataclass(frozen=True)
class HistogramJobResult:
    """Histogram data produced by a completed worker."""

    card_id: str
    job_id: str
    target: HistogramTarget
    settings: HistogramSettings
    result: HistogramResult


@dataclass(frozen=True)
class HistogramCardBindingState:
    """Structured card state bound into the controller."""

    sdata: SpatialData | None
    target: HistogramTarget | None
    settings: HistogramSettings | None
    validation_error: str | None = None

    @property
    def can_calculate(self) -> bool:
        return (
            self.sdata is not None
            and self.target is not None
            and self.settings is not None
            and self.validation_error is None
        )


@thread_worker(start_thread=False, ignore_errors=True)
def _run_histogram_job(job: HistogramJob) -> HistogramJobResult:
    result = histogram_core.calculate_histogram(job.sdata, job.target, job.settings)
    return HistogramJobResult(
        card_id=job.card_id,
        job_id=job.job_id,
        target=job.target,
        settings=job.settings,
        result=result,
    )


class HistogramController:
    """Manage per-card background histogram calculation state."""

    def __init__(self, *, on_state_changed: Callable[[str], None] | None = None) -> None:
        self._on_state_changed = on_state_changed

        self._bindings: dict[str, HistogramCardBindingState] = {}
        self._latest_requested_job_ids: dict[str, str] = {}
        self._active_worker_job_ids: dict[str, str] = {}
        self._active_workers: dict[str, Any] = {}
        self._results: dict[str, HistogramResult] = {}
        self._status_messages: dict[str, str] = {}
        self._status_kinds: dict[str, HistogramStatusKind] = {}
        self._next_job_number = 0

    def bind(
        self,
        card_id: str,
        sdata: SpatialData | None,
        target: HistogramTarget | None,
        settings: HistogramSettings | None,
        validation_error: str | None = None,
    ) -> bool:
        """Bind a card to its latest structured UI state."""
        normalized_error = _normalize_validation_error(validation_error)
        next_state = HistogramCardBindingState(
            sdata=sdata,
            target=target,
            settings=settings,
            validation_error=normalized_error,
        )
        previous_state = self._bindings.get(card_id)
        changed = previous_state is None or not _binding_states_match(previous_state, next_state)
        if not changed:
            return False

        was_running = self.is_running(card_id)
        had_result = card_id in self._results
        self._bindings[card_id] = next_state
        self._results.pop(card_id, None)
        self._cancel_active_worker(card_id)
        self._update_idle_status(card_id, was_running=was_running, had_result=had_result)
        return True

    def calculate(self, card_id: str) -> bool:
        """Launch histogram calculation for the current bound state of one card."""
        if self.is_running(card_id):
            self._set_status(card_id, "Histogram calculation is already running.", kind="info")
            return False

        job = self._prepare_histogram_job(card_id)
        if job is None:
            return False

        worker = self._create_histogram_worker(job)
        self._active_workers[card_id] = worker
        self._active_worker_job_ids[card_id] = job.job_id
        worker.returned.connect(partial(self._on_worker_returned, card_id, job.job_id))
        worker.errored.connect(partial(self._on_worker_errored, card_id, job.job_id))
        worker.finished.connect(partial(self._on_worker_finished, card_id, job.job_id))
        self._set_status(card_id, "Calculating histogram.", kind="info")
        worker.start()
        return True

    def invalidate_card(self, card_id: str) -> None:
        """Invalidate in-flight work for a card without changing its bound state."""
        if card_id not in self._bindings:
            return

        had_result = card_id in self._results
        was_running = self.is_running(card_id)
        self._results.pop(card_id, None)
        self._cancel_active_worker(card_id)
        self._update_idle_status(card_id, was_running=was_running, had_result=had_result)

    def remove_card(self, card_id: str) -> None:
        """Forget a card and ignore any late worker signal for it."""
        self._cancel_active_worker(card_id)
        self._bindings.pop(card_id, None)
        self._latest_requested_job_ids.pop(card_id, None)
        self._results.pop(card_id, None)
        self._status_messages.pop(card_id, None)
        self._status_kinds.pop(card_id, None)

    def can_calculate(self, card_id: str) -> bool:
        state = self._bindings.get(card_id)
        return state is not None and state.can_calculate and not self.is_running(card_id)

    def is_running(self, card_id: str) -> bool:
        return card_id in self._active_workers

    def binding_state(self, card_id: str) -> HistogramCardBindingState | None:
        return self._bindings.get(card_id)

    def result_for_card(self, card_id: str) -> HistogramResult | None:
        return self._results.get(card_id)

    def status_message(self, card_id: str) -> str:
        return self._status_messages.get(card_id, HISTOGRAM_IDLE_STATUS)

    def status_kind(self, card_id: str) -> HistogramStatusKind:
        return self._status_kinds.get(card_id, "warning")

    def _prepare_histogram_job(self, card_id: str) -> HistogramJob | None:
        state = self._bindings.get(card_id)
        if state is None:
            self._set_status(card_id, HISTOGRAM_IDLE_STATUS, kind="warning")
            return None

        if state.validation_error is not None:
            self._set_status(card_id, state.validation_error, kind="warning")
            return None

        if state.sdata is None:
            self._set_status(card_id, "No SpatialData loaded.", kind="warning")
            return None
        if state.target is None:
            self._set_status(card_id, "Choose a histogram target.", kind="warning")
            return None
        if state.settings is None:
            self._set_status(card_id, "Choose valid histogram settings.", kind="warning")
            return None

        job_id = self._next_job_id()
        self._latest_requested_job_ids[card_id] = job_id
        return HistogramJob(
            card_id=card_id,
            job_id=job_id,
            sdata=state.sdata,
            target=state.target,
            settings=state.settings,
        )

    def _next_job_id(self) -> str:
        self._next_job_number += 1
        return str(self._next_job_number)

    def _update_idle_status(self, card_id: str, *, was_running: bool = False, had_result: bool = False) -> None:
        state = self._bindings.get(card_id)
        if state is None:
            self._set_status(card_id, HISTOGRAM_IDLE_STATUS, kind="warning")
            return

        if state.validation_error is not None:
            self._set_status(card_id, state.validation_error, kind="warning")
            return

        if state.sdata is None:
            self._set_status(card_id, "No SpatialData loaded.", kind="warning")
            return
        if state.target is None:
            self._set_status(card_id, "Choose a histogram target.", kind="warning")
            return
        if state.settings is None:
            self._set_status(card_id, "Choose valid histogram settings.", kind="warning")
            return

        if was_running or had_result:
            self._set_status(card_id, "Target or settings changed. Calculate again.", kind="info")
            return

        self._set_status(card_id, "Ready to calculate.", kind="info")

    def _set_status(self, card_id: str, message: str, *, kind: HistogramStatusKind) -> None:
        self._status_messages[card_id] = message
        self._status_kinds[card_id] = kind
        if self._on_state_changed is not None:
            self._on_state_changed(card_id)

    def _create_histogram_worker(self, job: HistogramJob) -> Any:
        return _run_histogram_job(job)

    def _on_worker_returned(self, card_id: str, job_id: str, result: HistogramJobResult) -> None:
        if not self._is_current_job(card_id, job_id):
            return

        self._results[card_id] = result.result
        self._set_status(card_id, "Histogram calculated.", kind="success")

    def _on_worker_errored(self, card_id: str, job_id: str, error: Exception) -> None:
        if not self._is_current_job(card_id, job_id):
            return

        self._results.pop(card_id, None)
        self._set_status(card_id, f"Histogram calculation failed: {error}", kind="error")

    def _on_worker_finished(self, card_id: str, job_id: str) -> None:
        if self._active_worker_job_ids.get(card_id) != job_id:
            return

        self._active_workers.pop(card_id, None)
        self._active_worker_job_ids.pop(card_id, None)
        if self._on_state_changed is not None:
            self._on_state_changed(card_id)

    def _is_current_job(self, card_id: str, job_id: str) -> bool:
        return (
            self._latest_requested_job_ids.get(card_id) == job_id and self._active_worker_job_ids.get(card_id) == job_id
        )

    def _cancel_active_worker(self, card_id: str) -> None:
        worker = self._active_workers.pop(card_id, None)
        self._active_worker_job_ids.pop(card_id, None)
        if worker is None:
            return

        quit_worker = getattr(worker, "quit", None)
        if callable(quit_worker):
            quit_worker()


def _normalize_validation_error(validation_error: str | None) -> str | None:
    if validation_error is None:
        return None

    normalized = str(validation_error).strip()
    return normalized or None


def _binding_states_match(left: HistogramCardBindingState, right: HistogramCardBindingState) -> bool:
    return (
        left.sdata is right.sdata
        and left.target == right.target
        and left.settings == right.settings
        and left.validation_error == right.validation_error
    )
