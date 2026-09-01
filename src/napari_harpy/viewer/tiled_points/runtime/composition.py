"""Compose one tiled-points layer with its cache session and renderer events.

The composition object in this module lives on the napari GUI thread and owns
the signal wiring around one already-created layer::

    TiledPointsLayerModel.events.viewport
            |
            v
    _TiledPointsLayerRuntime
            |
            v
    _TiledPointsViewportCoordinator.submit_viewport()
            |
            v
    _TiledPointsCacheSession.request_viewport()
            |
            | Qt queued signal
            v
    _TiledPointsCacheWorker.read_viewport_snapshot()
            |
            v
    _PointsCacheReader / Zarr
            |
            | immutable TiledPointsRenderSnapshot
            v
    _TiledPointsViewportCoordinator.snapshot_ready
            |
            v
    _TiledPointsLayerRuntime
            |
            v
    TiledPointsLayerModel.events.render_snapshot
            |
            v
    VispyTiledPointsLayer
            |
            | TiledPointsRenderResult
            v
    TiledPointsLayerModel.events.render_snapshot_result
            |
            v
    _TiledPointsLayerRuntime commits display status

Renderer failures follow a parallel acknowledgement path::

    VispyTiledPointsLayer encounters a specific exception
            |
            v
    TiledPointsLayerModel.events.render_error(error)
            |
            v
    _TiledPointsLayerRuntime._on_render_error()
            |-- records the error on _PendingRenderActivation
            `-- publishes the specific failure status
            |
            v
    TiledPointsRenderResult(applied=False)
            |
            v
    _TiledPointsLayerRuntime clears the pending activation without replacing
    the specific error with a generic renderer-declined error

The runtime performs no cache IO and does not address the VisPy layer directly.
Its viewport callback only submits to the coordinator's latest-request mailbox.
Worker results return through queued Qt signals, and the runtime emits only
model events on the GUI thread.

Closure uses the inverse ownership order: disconnect model and result listeners,
close the coordinator so no pending request can be dispatched, then ask the
session to close its reader on the worker thread. This explicit close boundary
is what prevents a late result from mutating a removed layer.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

from napari.utils.events import Event
from qtpy.QtCore import QObject, Slot

from napari_harpy.core.multi_scale_cache_points_zarr.reader import _CacheDatasetInfo
from napari_harpy.viewer.tiled_points.contracts import (
    TiledPointsDatasetReference,
    TiledPointsLayerStatus,
    TiledPointsRenderResult,
    TiledPointsRenderSnapshot,
    TiledPointsViewportState,
)
from napari_harpy.viewer.tiled_points.napari.layer import TiledPointsLayerModel
from napari_harpy.viewer.tiled_points.runtime.cache_session import (
    _CacheSessionFailure,
    _CacheSessionSettings,
    _CacheSessionState,
    _TiledPointsCacheSession,
)
from napari_harpy.viewer.tiled_points.runtime.coordinator import _TiledPointsViewportCoordinator

_SessionFactory = Callable[[Path, _CacheSessionSettings], _TiledPointsCacheSession]


@dataclass
class _PendingRenderActivation:
    """Retain one candidate status until the renderer acknowledges its snapshot."""

    request_generation: int
    selection_generation: int
    status: TiledPointsLayerStatus
    error: object | None = None


class _TiledPointsLayerRuntime(QObject):
    """Own the GUI-side runtime wiring for one cache-backed points layer.

    Parameters
    ----------
    layer
        Persistent logical napari layer that emits normalized viewport states
        and accepts complete render snapshots.
    cache_root
        Published Zarr points-cache root opened by the worker-owned reader.
    settings
        Explicit metadata-index and decoded CPU-tile memory policy for the
        cache session. Renderer residency remains configured on ``layer``.
    initial_requested_value_ids
        Initial proper subset to make resident before the first viewport is
        dispatched. ``None`` selects all canonical values.

    Notes
    -----
    Construction installs every listener before starting the cache session.
    The layer's dataset reference is immutable: a different cache generation
    requires a new layer and runtime rather than rebinding this ownership graph.
    :meth:`close` is terminal and idempotent. The application binding that owns
    this runtime must call it when the corresponding layer is removed.
    """

    def __init__(
        self,
        layer: TiledPointsLayerModel,
        cache_root: str | Path,
        settings: _CacheSessionSettings,
        *,
        initial_requested_value_ids: tuple[int, ...] | None = None,
        session_factory: _SessionFactory = _TiledPointsCacheSession,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._layer = layer
        self._session = session_factory(Path(cache_root), settings)
        self._coordinator = _TiledPointsViewportCoordinator(
            self._session,
            initial_requested_value_ids=initial_requested_value_ids,
            parent=self,
        )
        self._dataset_verified = False
        self._closed = False
        self._active_status = layer.display_status
        self._pending_render_activation: _PendingRenderActivation | None = None

        layer.events.viewport.connect(self._on_viewport)
        layer.events.render_error.connect(self._on_render_error)
        layer.events.render_snapshot_result.connect(self._on_render_snapshot_result)
        self._coordinator.snapshot_ready.connect(self._on_snapshot_ready)
        self._coordinator.viewport_failed.connect(self._on_viewport_failed)
        self._session.dataset_available.connect(self._on_dataset_available)
        self._session.bucket_index_progress.connect(self._on_bucket_index_progress)
        self._session.ready.connect(self._on_ready)
        self._session.value_selection_ready.connect(self._on_value_selection_ready)
        self._session.failed.connect(self._on_session_failed)
        self._session.closed.connect(self._on_session_closed)

        self._set_transient_status("Opening cache")
        try:
            self._session.start()
        except Exception:
            self.close()
            raise

    @property
    def state(self) -> _CacheSessionState:
        """Return the current cache-session lifecycle state."""
        return self._session.state

    @property
    def selected_value_ids(self) -> tuple[int, ...] | None:
        """Return the successfully committed value selection."""
        return self._session.selected_value_ids

    @property
    def closed(self) -> bool:
        """Return whether this composition owner has stopped accepting work."""
        return self._closed

    def set_selected_value_ids(self, requested_value_ids: tuple[int, ...] | None) -> bool:
        """Request one worker-resident selected-value-index replacement.

        The preceding accepted snapshot remains visible while the worker loads
        a changed proper-subset index. ``None`` selects all canonical values.
        """
        self._require_open()
        accepted = self._coordinator.set_selected_value_ids(requested_value_ids)
        if accepted:
            self._set_transient_status("Updating selected-value index")
        return accepted

    def close(self) -> bool:
        """Disconnect the layer first, then stop scheduling and cache IO."""
        if self._closed:
            return False
        self._closed = True
        self._pending_render_activation = None
        self._disconnect_runtime_listeners()
        self._coordinator.close()
        self._session.close()
        return True

    @Slot(object)
    def _on_viewport(self, event: Event) -> None:
        if self._closed:
            return
        viewport = event.value
        if not isinstance(viewport, TiledPointsViewportState):
            raise ValueError("The layer viewport event must carry TiledPointsViewportState.")
        self._set_transient_status("Loading view")
        self._coordinator.submit_viewport(viewport)

    @Slot(object)
    def _on_dataset_available(self, dataset_info: _CacheDatasetInfo) -> None:
        if self._closed:
            return
        try:
            _require_dataset_matches_layer(dataset_info, self._layer.data)
        except Exception as error:  # noqa: BLE001
            self._publish_error(error)
            self.close()
            return
        self._dataset_verified = True

    @Slot(int, int)
    def _on_bucket_index_progress(self, completed_buckets: int, total_buckets: int) -> None:
        if self._closed:
            return
        self._set_transient_status(f"Loading bucket indexes ({completed_buckets:,}/{total_buckets:,})")

    @Slot()
    def _on_ready(self) -> None:
        if self._closed:
            return
        if not self._dataset_verified:
            error = RuntimeError("Cache session became ready before its dataset identity was verified.")
            self._publish_error(error)
            self.close()
            return
        if self._coordinator.selection_update_pending:
            message = "Updating selected-value index"
        else:
            message = "Loading view" if self._coordinator.active_request_generation is not None else "Ready"
        self._set_transient_status(message)

    @Slot(object, int)
    def _on_value_selection_ready(
        self,
        requested_value_ids: tuple[int, ...] | None,
        resident_bytes: int,
    ) -> None:
        del requested_value_ids, resident_bytes
        if self._closed:
            return
        message = "Loading view" if self._coordinator.active_request_generation is not None else "Ready"
        self._set_transient_status(message)

    @Slot(object)
    def _on_snapshot_ready(self, snapshot: TiledPointsRenderSnapshot) -> None:
        if self._closed:
            return
        if not snapshot.within_budget:
            # This is a metadata-only viewport result, not renderer input. Keep
            # the active visual untouched and report why it was not replaced.
            self._set_transient_status(
                f"View exceeds the point budget ({snapshot.estimated_point_count:,} estimated points); "
                "retaining the previous view"
            )
            return

        status = _status_from_snapshot(snapshot)
        if self._pending_render_activation is not None:
            raise RuntimeError("A render snapshot was submitted while another activation was pending.")
        pending = _PendingRenderActivation(
            request_generation=snapshot.request_generation,
            selection_generation=snapshot.selection_generation,
            status=status,
        )
        self._pending_render_activation = pending
        try:
            self._layer.events.render_snapshot(value=snapshot)
        except Exception:
            if self._pending_render_activation is pending:
                self._pending_render_activation = None
            raise
        if self._pending_render_activation is pending:
            # Model events are synchronous. Reaching this branch means no
            # renderer listener acknowledged the candidate snapshot.
            self._pending_render_activation = None
            self._publish_error(RuntimeError("Renderer did not acknowledge the submitted snapshot."))

    @Slot(object)
    def _on_render_snapshot_result(self, event: Event) -> None:
        if self._closed:
            return
        result = event.value
        if not isinstance(result, TiledPointsRenderResult):
            raise ValueError("The render-result event must carry TiledPointsRenderResult.")
        pending = self._pending_render_activation
        if pending is None:
            raise RuntimeError("Renderer acknowledged a snapshot when no activation was pending.")
        if (
            result.request_generation != pending.request_generation
            or result.selection_generation != pending.selection_generation
        ):
            raise ValueError("Renderer result does not match the pending snapshot generation.")
        self._pending_render_activation = None
        if result.applied:
            self._active_status = pending.status
            self._layer.display_status = pending.status
        elif pending.error is None:
            self._publish_error(RuntimeError("Renderer declined the submitted snapshot without an error."))

    @Slot(int, object)
    def _on_viewport_failed(self, request_generation: int, failure: _CacheSessionFailure) -> None:
        if self._closed or request_generation != self._coordinator.request_generation:
            return
        self._publish_error(failure)

    @Slot(object)
    def _on_session_failed(self, failure: _CacheSessionFailure) -> None:
        if self._closed or failure.phase == "viewport":
            return
        self._publish_error(failure)

    @Slot(object)
    def _on_render_error(self, event: Event) -> None:
        if self._closed:
            return
        pending = self._pending_render_activation
        if pending is not None:
            # Preserve the renderer's specific failure so the subsequent
            # ``applied=False`` acknowledgement does not publish a second,
            # generic renderer-declined error.
            pending.error = event.value
        self._set_transient_status(f"Display failed: {_error_message(event.value)}")

    @Slot()
    def _on_session_closed(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._pending_render_activation = None
        self._disconnect_runtime_listeners()
        self._coordinator.close()

    def _publish_error(self, error: object) -> None:
        self._layer.events.render_error(value=error)

    def _set_transient_status(self, message: str) -> None:
        active = self._active_status
        self._layer.display_status = TiledPointsLayerStatus(
            level=active.level,
            level_kind=active.level_kind,
            rendered_point_count=active.rendered_point_count,
            rendered_tile_count=active.rendered_tile_count,
            message=message,
            sampled=active.sampled,
            omitted_value_ids=active.omitted_value_ids,
        )

    def _disconnect_runtime_listeners(self) -> None:
        with suppress(TypeError, RuntimeError):
            self._layer.events.viewport.disconnect(self._on_viewport)
        with suppress(TypeError, RuntimeError):
            self._layer.events.render_error.disconnect(self._on_render_error)
        with suppress(TypeError, RuntimeError):
            self._layer.events.render_snapshot_result.disconnect(self._on_render_snapshot_result)
        for signal, callback in (
            (self._coordinator.snapshot_ready, self._on_snapshot_ready),
            (self._coordinator.viewport_failed, self._on_viewport_failed),
            (self._session.dataset_available, self._on_dataset_available),
            (self._session.bucket_index_progress, self._on_bucket_index_progress),
            (self._session.ready, self._on_ready),
            (self._session.value_selection_ready, self._on_value_selection_ready),
            (self._session.failed, self._on_session_failed),
            (self._session.closed, self._on_session_closed),
        ):
            with suppress(TypeError, RuntimeError):
                signal.disconnect(callback)

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("The tiled-points layer runtime is closed.")


def _require_dataset_matches_layer(
    dataset_info: _CacheDatasetInfo,
    reference: TiledPointsDatasetReference,
) -> None:
    """Reject a cache root that does not describe the layer's logical data."""
    observed = {
        "cache_generation_id": dataset_info.cache_generation_id,
        "points_name": dataset_info.points_name,
        "value_column": dataset_info.value_column,
        "value_count": len(dataset_info.value_names),
        "x_origin": dataset_info.x_origin,
        "y_origin": dataset_info.y_origin,
        "x_min": dataset_info.x_min,
        "x_max": dataset_info.x_max,
        "y_min": dataset_info.y_min,
        "y_max": dataset_info.y_max,
    }
    mismatches = tuple(name for name, value in observed.items() if value != getattr(reference, name))
    if mismatches:
        names = ", ".join(mismatches)
        raise ValueError(f"Cache dataset does not match the tiled-points layer reference: {names}.")


def _status_from_snapshot(snapshot: TiledPointsRenderSnapshot) -> TiledPointsLayerStatus:
    sampled = snapshot.level_kind != "exact"
    if snapshot.all_exact_present_values_omitted:
        message = "Selected values are not represented at the sampled LOD"
    elif snapshot.rendered_point_count == 0:
        message = "No points in view"
    elif snapshot.omitted_value_ids:
        message = "Ready; some selected values are not represented at this sampled LOD"
    else:
        message = "Ready"
    return TiledPointsLayerStatus(
        level=snapshot.level,
        level_kind=snapshot.level_kind,
        rendered_point_count=snapshot.rendered_point_count,
        rendered_tile_count=snapshot.rendered_tile_count,
        message=message,
        sampled=sampled,
        omitted_value_ids=snapshot.omitted_value_ids,
    )


def _error_message(error: object) -> str:
    if isinstance(error, _CacheSessionFailure):
        return f"{error.phase}: {error.message}"
    return str(error)
