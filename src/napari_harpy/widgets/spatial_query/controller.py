from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, Literal

from napari_harpy.core.spatial_query import (
    CanonicalCacheReport,
    CanonicalCacheState,
    CanonicalCacheUpdatePayload,
    CanonicalCenterQueryRequest,
    CanonicalCenterQueryResult,
    CanonicalCentersResult,
    apply_canonical_cache_update,
    build_canonical_center_query_request,
    calculate_canonical_centers,
    evaluate_canonical_center_query,
    read_canonical_centers_from_cache,
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
type SpatialQueryWorkerPhase = Literal["canonical_centers", "containment_query"]

SPATIAL_QUERY_IDLE_STATUS = "Spatial query: idle."


@dataclass(frozen=True)
class _ActiveWorkerPhase:
    """Worker and domain context for the currently active operation phase."""

    operation_id: int
    phase: SpatialQueryWorkerPhase
    worker: Any
    sdata: SpatialData
    shapes_name: str
    coordinate_system: str


@thread_worker(start_thread=False, ignore_errors=True)
def _run_canonical_centers_calculation(
    sdata: SpatialData,
    report: CanonicalCacheReport,
) -> CanonicalCacheUpdatePayload:
    """Calculate canonical centers without mutating the cache."""
    return calculate_canonical_centers(sdata, report)


@thread_worker(start_thread=False, ignore_errors=True)
def _run_canonical_center_query(
    request: CanonicalCenterQueryRequest,
) -> CanonicalCenterQueryResult:
    """Evaluate one immutable containment request without mutation."""
    return evaluate_canonical_center_query(request)


class SpatialQueryController:
    """Manage background Spatial Query work and main-thread result acceptance.

    A valid-cache operation follows this read-only boundary:

        main thread
            read_canonical_centers_from_cache()
            → read already-validated centers from .obsm
            → no SpatialData or AnnData mutation

        main thread
            build_canonical_center_query_request()
            → snapshot validated polygons and their labels-frame affine

        worker thread
            evaluate_canonical_center_query()
            → return matching instance IDs
            → no SpatialData or AnnData mutation

    An operation without a valid cache follows this calculation and mutation
    boundary:

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

        main thread
            build_canonical_center_query_request()
            → snapshot validated polygons and their labels-frame affine

        worker thread
            evaluate_canonical_center_query()
            → return matching instance IDs
            → no SpatialData or AnnData mutation

    Both sequential workers share one operation ID. Cancellation clears the
    active phase, and every late signal must match both the current operation
    ID and phase before it can affect state.
    """

    def __init__(
        self,
        *,
        on_state_changed: Callable[[], None] | None = None,
        on_centers_ready: Callable[[CanonicalCentersResult], None] | None = None,
        on_query_ready: Callable[[CanonicalCenterQueryResult], None] | None = None,
    ) -> None:
        self._on_state_changed = on_state_changed
        self._on_centers_ready = on_centers_ready
        self._on_query_ready = on_query_ready
        self._is_shutdown = False

        self._last_operation_id = 0
        self._active: _ActiveWorkerPhase | None = None

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

    def start_spatial_query(
        self,
        sdata: SpatialData,
        report: CanonicalCacheReport,
        *,
        shapes_name: str,
        coordinate_system: str,
    ) -> bool:
        """Start one optional-center-calculation and containment operation."""
        if not isinstance(report, CanonicalCacheReport):
            raise TypeError("Spatial Query requires a CanonicalCacheReport.")
        if not isinstance(shapes_name, str) or not shapes_name:
            raise ValueError("Spatial Query Shapes name must be a non-empty string.")
        if not isinstance(coordinate_system, str) or not coordinate_system:
            raise ValueError("Spatial Query coordinate system must be a non-empty string.")
        if self._is_shutdown:
            return False
        if self.is_running:
            return False

        self._last_operation_id += 1
        operation_id = self._last_operation_id

        if report.state is CanonicalCacheState.VALID:
            try:
                centers_result = read_canonical_centers_from_cache(sdata, report)
                query_phase = self._build_containment_query_phase(
                    operation_id,
                    sdata,
                    centers_result,
                    shapes_name=shapes_name,
                    coordinate_system=coordinate_system,
                )
            except Exception as exc:  # noqa: BLE001
                self._set_status(f"Spatial query preflight failed: {exc}", kind="error")
                return True

            self._active = query_phase
            self._notify_centers_ready(centers_result)
            if self._get_matching_active_phase(operation_id, "containment_query") is not None:
                self._start_containment_query_worker(query_phase)
            return True

        worker = self._create_canonical_centers_worker(sdata, report)
        self._active = _ActiveWorkerPhase(
            operation_id=operation_id,
            phase="canonical_centers",
            worker=worker,
            sdata=sdata,
            shapes_name=shapes_name,
            coordinate_system=coordinate_system,
        )

        worker.returned.connect(partial(self._on_canonical_centers_returned, operation_id))
        worker.errored.connect(partial(self._on_worker_errored, operation_id, "canonical_centers"))
        worker.finished.connect(partial(self._on_worker_finished, operation_id, "canonical_centers"))
        self._set_status(f'Calculating centers for "{report.labels_name}".', kind="info")
        worker.start()
        return True

    def cancel_active_operation(self) -> bool:
        """Invalidate active work and ignore every later signal from it."""
        active = self._active
        if active is None:
            return False

        self._cancel_active_worker()
        phase_name = "center calculation" if active.phase == "canonical_centers" else "containment query"
        self._set_status(f"Spatial query {phase_name} cancelled.", kind="info")
        return True

    def shutdown(self, *args: object) -> None:
        """Invalidate active work and permanently detach callbacks."""
        del args
        if self._is_shutdown:
            return

        self._is_shutdown = True
        self._cancel_active_worker()
        self._on_state_changed = None
        self._on_centers_ready = None
        self._on_query_ready = None

    def _create_canonical_centers_worker(
        self,
        sdata: SpatialData,
        report: CanonicalCacheReport,
    ) -> Any:
        return _run_canonical_centers_calculation(sdata, report)

    def _create_containment_query_worker(
        self,
        request: CanonicalCenterQueryRequest,
    ) -> Any:
        return _run_canonical_center_query(request)

    def _on_canonical_centers_returned(
        self,
        operation_id: int,
        payload: CanonicalCacheUpdatePayload,
    ) -> None:
        active = self._get_matching_active_phase(operation_id, "canonical_centers")
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
            self._set_status(f"Spatial query: canonical-center update failed: {exc}", kind="error")
            return

        self._notify_centers_ready(result)
        if self._get_matching_active_phase(operation_id, "canonical_centers") is None:
            return

        try:
            query_phase = self._build_containment_query_phase(
                operation_id,
                active.sdata,
                result,
                shapes_name=active.shapes_name,
                coordinate_system=active.coordinate_system,
            )
        except Exception as exc:  # noqa: BLE001
            self._set_status(f"Spatial query preflight failed: {exc}", kind="error")
            return

        # Replace the active phase before starting the second worker. The
        # canonical worker emits `finished` afterward; its phase-tagged handler
        # must then ignore that old signal instead of clearing this query phase.
        self._active = query_phase
        self._start_containment_query_worker(query_phase)

    def _build_containment_query_phase(
        self,
        operation_id: int,
        sdata: SpatialData,
        centers_result: CanonicalCentersResult,
        *,
        shapes_name: str,
        coordinate_system: str,
    ) -> _ActiveWorkerPhase:
        request = build_canonical_center_query_request(
            sdata,
            shapes_name=shapes_name,
            coordinate_system=coordinate_system,
            canonical_centers=centers_result,
        )
        worker = self._create_containment_query_worker(request)
        worker.returned.connect(partial(self._on_containment_query_returned, operation_id))
        worker.errored.connect(partial(self._on_worker_errored, operation_id, "containment_query"))
        worker.finished.connect(partial(self._on_worker_finished, operation_id, "containment_query"))
        return _ActiveWorkerPhase(
            operation_id=operation_id,
            phase="containment_query",
            worker=worker,
            sdata=sdata,
            shapes_name=shapes_name,
            coordinate_system=coordinate_system,
        )

    def _start_containment_query_worker(self, phase: _ActiveWorkerPhase) -> None:
        self._set_status(f'Querying centers inside "{phase.shapes_name}".', kind="info")
        phase.worker.start()

    def _on_containment_query_returned(
        self,
        operation_id: int,
        result: CanonicalCenterQueryResult,
    ) -> None:
        if self._get_matching_active_phase(operation_id, "containment_query") is None:
            return
        if not isinstance(result, CanonicalCenterQueryResult):
            self._set_status("Spatial query returned an invalid containment result.", kind="error")
            return

        count = result.matched_instance_count
        if count == 0:
            self._set_status("No instance centroids found in the annotation.", kind="info")
            return

        noun = "centroid" if count == 1 else "centroids"
        self._set_status(f"{count} instance {noun} found.", kind="success")
        if self._on_query_ready is not None:
            self._on_query_ready(result)

    def _notify_centers_ready(self, result: CanonicalCentersResult) -> None:
        if self._on_centers_ready is not None:
            self._on_centers_ready(result)

    def _on_worker_errored(
        self,
        operation_id: int,
        phase: SpatialQueryWorkerPhase,
        error: Exception,
    ) -> None:
        if self._get_matching_active_phase(operation_id, phase) is None:
            return

        if phase == "canonical_centers":
            message = f"Canonical-center calculation failed: {error}"
        else:
            message = f"Containment query failed: {error}"
        self._set_status(message, kind="error")

    def _on_worker_finished(
        self,
        operation_id: int,
        phase: SpatialQueryWorkerPhase,
    ) -> None:
        if self._get_matching_active_phase(operation_id, phase) is None:
            return

        self._active = None
        self._notify_state_changed()

    def _get_matching_active_phase(
        self,
        operation_id: int,
        phase: SpatialQueryWorkerPhase,
    ) -> _ActiveWorkerPhase | None:
        """Return the active phase when both operation ID and phase match."""
        active = self._active
        if self._is_shutdown or active is None or active.operation_id != operation_id or active.phase != phase:
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
