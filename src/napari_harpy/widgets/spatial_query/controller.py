from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, Literal

from napari_harpy.core.spatial_query import (
    CanonicalCacheReport,
    CanonicalCacheUpdatePayload,
    CanonicalCentersResult,
    apply_canonical_cache_update,
    calculate_canonical_centers,
)

if TYPE_CHECKING:
    from spatialdata import SpatialData


def _resolve_thread_worker() -> Any:
    try:
        from napari.qt.threading import thread_worker
    except Exception:  # pragma: no cover - fallback for sandboxed test imports  # noqa: BLE001
        from superqt.utils import thread_worker

    return thread_worker


thread_worker = _resolve_thread_worker()

type SpatialQueryStatusKind = Literal["info", "warning", "success", "error"]
type SpatialQueryWorkerPhase = Literal["canonical_centers"]

SPATIAL_QUERY_IDLE_STATUS = "Spatial query: idle."


@dataclass(frozen=True)
class _ActiveOperationPhase:
    """Worker and domain context for the currently active operation phase."""

    operation_id: int
    phase: SpatialQueryWorkerPhase
    worker: Any
    sdata: SpatialData


@thread_worker(start_thread=False, ignore_errors=True)
def _run_canonical_centers_calculation(
    sdata: SpatialData,
    report: CanonicalCacheReport,
) -> CanonicalCacheUpdatePayload:
    """Calculate canonical centers without mutating the cache."""
    return calculate_canonical_centers(sdata, report)


class SpatialQueryController:
    """Manage background Spatial Query work and main-thread result acceptance.

    Canonical centers follow this calculation and mutation boundary:

        worker thread
            calculate_canonical_centers()
            → read labels
            → calculate centers
            → build payload
            → no table mutation

        main thread
            apply_canonical_cache_update()
            → re-inspect current source and table binding
            → reject outdated payload if necessary
            → update .obsm and .uns atomically

    Cancellation clears the active operation phase. Any later worker signal
    carrying an operation ID that is no longer active is ignored.
    """

    def __init__(
        self,
        *,
        on_state_changed: Callable[[], None] | None = None,
        on_centers_ready: Callable[[CanonicalCentersResult], None] | None = None,
    ) -> None:
        self._on_state_changed = on_state_changed
        self._on_centers_ready = on_centers_ready
        self._is_shutdown = False

        self._last_operation_id = 0
        self._active: _ActiveOperationPhase | None = None

        self._centers_result: CanonicalCentersResult | None = None
        self._status_message = SPATIAL_QUERY_IDLE_STATUS
        self._status_kind: SpatialQueryStatusKind = "info"

    @property
    def operation_id(self) -> int:
        """Return the most recently allocated operation ID."""
        return self._last_operation_id

    @property
    def active_phase(self) -> SpatialQueryWorkerPhase | None:
        """Return the phase owning the active worker, if any."""
        return None if self._active is None else self._active.phase

    @property
    def is_running(self) -> bool:
        """Return whether a Spatial Query worker is active."""
        return self._active is not None

    @property
    def status_message(self) -> str:
        """Return the current user-facing status message."""
        return self._status_message

    @property
    def status_kind(self) -> SpatialQueryStatusKind:
        """Return the current status level."""
        return self._status_kind

    @property
    def centers_result(self) -> CanonicalCentersResult | None:
        """Return the latest accepted canonical-centers result."""
        return self._centers_result

    def start_canonical_centers_calculation(
        self,
        sdata: SpatialData,
        report: CanonicalCacheReport,
    ) -> bool:
        """Start mutation-free canonical-center calculation for a captured report."""
        if self._is_shutdown:
            return False
        if self.is_running:
            self._set_status("Spatial query: canonical-center calculation is already running.", kind="info")
            return False

        self._last_operation_id += 1
        operation_id = self._last_operation_id
        worker = self._create_canonical_centers_worker(sdata, report)
        self._active = _ActiveOperationPhase(
            operation_id=operation_id,
            phase="canonical_centers",
            worker=worker,
            sdata=sdata,
        )
        self._centers_result = None

        worker.returned.connect(partial(self._on_canonical_centers_returned, operation_id))
        worker.errored.connect(partial(self._on_worker_errored, operation_id, "canonical_centers"))
        worker.finished.connect(partial(self._on_worker_finished, operation_id, "canonical_centers"))
        self._set_status("Spatial query: calculating canonical centers.", kind="info")
        worker.start()
        return True

    def cancel_active_operation(self) -> bool:
        """Invalidate active work and ignore every later signal from it."""
        if not self.is_running:
            return False

        self._cancel_active_worker()
        self._centers_result = None
        self._set_status("Spatial query: canonical-center calculation cancelled.", kind="info")
        return True

    def shutdown(self, *args: object) -> None:
        """Invalidate active work and permanently detach callbacks."""
        del args
        if self._is_shutdown:
            return

        self._is_shutdown = True
        self._cancel_active_worker()
        self._centers_result = None
        self._on_state_changed = None
        self._on_centers_ready = None

    def _create_canonical_centers_worker(
        self,
        sdata: SpatialData,
        report: CanonicalCacheReport,
    ) -> Any:
        return _run_canonical_centers_calculation(sdata, report)

    def _on_canonical_centers_returned(
        self,
        operation_id: int,
        payload: CanonicalCacheUpdatePayload,
    ) -> None:
        active = self._get_current_operation_phase(operation_id, "canonical_centers")
        if active is None:
            return

        try:
            cache_update = apply_canonical_cache_update(active.sdata, payload)
            result = CanonicalCentersResult(
                source_signature=payload.source_signature,
                binding=payload.binding,
                centers=payload.centers,
                cache_update=cache_update,
            )
        except Exception as exc:  # noqa: BLE001
            self._centers_result = None
            self._set_status(f"Spatial query: canonical-center update failed: {exc}", kind="error")
            return

        self._centers_result = result
        if self._on_centers_ready is not None:
            self._on_centers_ready(result)
        self._set_status("Spatial query: canonical centers are ready.", kind="success")

    def _on_worker_errored(
        self,
        operation_id: int,
        phase: SpatialQueryWorkerPhase,
        error: Exception,
    ) -> None:
        if self._get_current_operation_phase(operation_id, phase) is None:
            return

        self._centers_result = None
        self._set_status(f"Spatial query: canonical-center calculation failed: {error}", kind="error")

    def _on_worker_finished(
        self,
        operation_id: int,
        phase: SpatialQueryWorkerPhase,
    ) -> None:
        if self._get_current_operation_phase(operation_id, phase) is None:
            return

        self._active = None
        self._notify_state_changed()

    def _get_current_operation_phase(
        self,
        operation_id: int,
        phase: SpatialQueryWorkerPhase,
    ) -> _ActiveOperationPhase | None:
        active = self._active
        if (
            self._is_shutdown
            or active is None
            or active.operation_id != operation_id
            or active.phase != phase
        ):
            return None
        return active

    def _cancel_active_worker(self) -> None:
        active = self._active
        self._active = None
        if active is None:
            return

        quit_worker = getattr(active.worker, "quit", None)
        if callable(quit_worker):
            quit_worker()

    def _set_status(self, message: str, *, kind: SpatialQueryStatusKind) -> None:
        self._status_message = message
        self._status_kind = kind
        self._notify_state_changed()

    def _notify_state_changed(self) -> None:
        if self._on_state_changed is not None:
            self._on_state_changed()
