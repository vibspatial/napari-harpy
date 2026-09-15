"""Own one tiled-points layer's cache session and renderer integration.

Participants and responsibilities
--------------------------------
The diagrams use short class names for these implementations:

``_TiledPointsLayerRuntime``
    GUI-side wiring between layer events, viewport submission, and renderer replies.

    Module: ``napari_harpy.viewer.tiled_points.runtime.layer_runtime``

``_TiledPointsViewportScheduler``
    GUI-side request scheduling, stale-result rejection, and activation feedback.

    Module: ``napari_harpy.viewer.tiled_points.runtime.viewport_scheduler``

``_TiledPointsCacheSession``
    GUI-side interface that transports requests and results across the worker-thread
    boundary and manages the worker lifecycle.

    Module: ``napari_harpy.viewer.tiled_points.runtime.cache_session``

``_TiledPointsCacheWorker``
    Worker-thread owner of the reader and cached data; selects LOD, decides whether
    the accepted batch can be reused, and prepares replacement snapshots.

    Module: ``napari_harpy.viewer.tiled_points.runtime.cache_session``

``_PointsCacheReader``
    Cache metadata, viewport tile planning, and physical Zarr payload reads.

    Module: ``napari_harpy.core.multi_scale_cache_points_zarr.reader``

``TiledPointsLayerModel``
    Logical layer and events connecting viewport changes to snapshot rendering.

    Module: ``napari_harpy.viewer.tiled_points.napari.layer``

``VispyTiledPointsLayer``
    Renderer that consumes snapshots and emits acceptance or failure results.

    Module: ``napari_harpy.viewer.tiled_points.vispy.layer``

A render batch (``TiledPointsRenderBatch``) holds one immutable NumPy point array
prepared for rendering from the required tiles. Each row contains a cache-relative
position and a value ID. This is CPU-side data, not the renderer's GPU VBO.
A snapshot (``TiledPointsRenderSnapshot``) references this batch alongside request
identity, LOD, and status metadata. These types and ``TiledPointsRenderResult``
live in ``napari_harpy.viewer.tiled_points.contracts``.

Worker-owned cached data
-----------------------
``_RetainedViewport``
    Original requested viewport bounds paired with a snapshot containing an
    already-packed render batch. The worker's ``_retained_viewport`` is the last
    accepted entry; ``_pending_viewport`` awaits GUI acceptance. These are not a
    viewport-history cache.

    Module: ``napari_harpy.viewer.tiled_points.runtime.cache_session``

``_CpuTileResidency``
    Decoded tile locations and value IDs, retained within a byte-bounded LRU.
    These tiles can be reused when preparing a new render batch, independently
    of whether the previous packed batch can be reused.

    Module: ``napari_harpy.viewer.tiled_points.runtime.residency``

Both are CPU-side state, separate from the renderer's VBO. A retained-batch hit
avoids tile planning, payload reads, and packing. A replacement can also avoid
payload reads when all required tiles are CPU-resident, but still packs a new
batch. LOD evaluation precedes either path.

Request and activation flow
---------------------------
The layer runtime in this module lives on the napari GUI thread and owns
the signal wiring around one already-created layer. The rendering portion below
shows a current, renderable result; stale or over-budget snapshots do not reach
the renderer::

    TiledPointsLayerModel.events.viewport
            |
            v
    _TiledPointsLayerRuntime
            |
            v
    _TiledPointsViewportScheduler.submit_viewport()
            |
            v
    _TiledPointsCacheSession.request_viewport()
            |
            | Qt queued signal
            v
    _TiledPointsCacheWorker.read_viewport_snapshot()
            |-- hard-limit rejection: metadata-only snapshot
            |-- retained viewport reusable:
            |     create a snapshot with updated request metadata
            |     reuse the same TiledPointsRenderBatch and its point array
            |     (no point-array copying or repacking)
            `-- replacement: reuse CPU tiles; read only missing payloads
                             via _PointsCacheReader / Zarr; pack a new batch
            |
            | immutable TiledPointsRenderSnapshot, queued through the session
            v
    _TiledPointsViewportScheduler calls activate_snapshot(snapshot)
            |
            v
    _TiledPointsLayerRuntime._activate_snapshot()
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
    _TiledPointsLayerRuntime._on_render_snapshot_result() records the result
            |
            | snapshot event emission returns
            v
    _activate_snapshot() commits display status and clears activation state
            |
            | _activate_snapshot() returns TiledPointsRenderResult
            v
    scheduler forwards the result through session.acknowledge_render_result()
            |
            | Qt queued signal
            v
    _TiledPointsCacheWorker.acknowledge_render_result()
        applied=True: promote the matching pending viewport to retained
        applied=False: discard it and preserve the previous retained viewport

Renderer failures follow a parallel acknowledgement path::

    VispyTiledPointsLayer encounters a specific exception
            |
            v
    TiledPointsLayerModel.events.render_error(error)
            |
            v
    _TiledPointsLayerRuntime._on_render_error()
            |-- records the error on _RenderActivationState
            `-- publishes the specific failure status
            |
            v
    TiledPointsRenderResult(applied=False)
            |
            v
    _activate_snapshot() clears activation state without replacing
    the specific error with a generic renderer-declined error

The runtime performs no cache IO and does not address the VisPy layer directly.
Its viewport callback only submits to the scheduler's latest-request mailbox.
Worker results return through queued Qt signals, and the runtime emits only
model events on the GUI thread.

Closure uses the inverse ownership order: disconnect model and result listeners,
close the scheduler so no pending request can be dispatched, then ask the
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
from napari_harpy.viewer.tiled_points.runtime.viewport_scheduler import _TiledPointsViewportScheduler

_SessionFactory = Callable[[Path, _CacheSessionSettings], _TiledPointsCacheSession]


@dataclass
class _RenderActivationState:
    """Share mutable state for one synchronous render activation attempt.

    ``_activate_snapshot()`` creates this object and exposes the same object
    through ``_render_activation_state`` for the renderer-result and error
    handlers to update. The result handler only validates and records
    ``renderer_result``. The caller reads that result and commits display status
    while the state remains installed, then clears the runtime attribute in its
    ``finally`` block. No handler creates, replaces, or clears activation state.
    """

    request_generation: int
    selection_generation: int
    error: object | None = None
    renderer_result: TiledPointsRenderResult | None = None


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
        self._viewport_scheduler = _TiledPointsViewportScheduler(
            self._session,
            activate_snapshot=self._activate_snapshot,
            initial_requested_value_ids=initial_requested_value_ids,
            parent=self,
        )
        self._dataset_verified = False
        self._closed = False
        self._active_status = layer.display_status
        self._render_activation_state: _RenderActivationState | None = None

        layer.events.viewport.connect(self._on_viewport)
        layer.events.render_error.connect(self._on_render_error)
        layer.events.render_snapshot_result.connect(self._on_render_snapshot_result)
        self._viewport_scheduler.viewport_failed.connect(self._on_viewport_failed)
        self._session.dataset_available.connect(self._on_dataset_available)
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
        """Return whether this layer runtime has stopped accepting work."""
        return self._closed

    def set_selected_value_ids(self, requested_value_ids: tuple[int, ...] | None) -> bool:
        """Request one worker-resident selected-value-index replacement.

        The preceding accepted snapshot remains visible while the worker loads
        a changed proper-subset index. ``None`` selects all canonical values.
        """
        self._require_open()
        accepted = self._viewport_scheduler.set_selected_value_ids(requested_value_ids)
        if accepted:
            self._set_transient_status("Updating selected-value index")
        return accepted

    def close(self) -> bool:
        """Disconnect the layer first, then stop scheduling and cache IO.

        If closure occurs during activation, that call observes ``_closed`` and
        releases its own activation state in ``finally``.
        """
        if self._closed:
            return False
        self._closed = True
        self._disconnect_runtime_listeners()
        self._viewport_scheduler.close()
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
        self._viewport_scheduler.submit_viewport(viewport)

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

    @Slot()
    def _on_ready(self) -> None:
        if self._closed:
            return
        if not self._dataset_verified:
            error = RuntimeError("Cache session became ready before its dataset identity was verified.")
            self._publish_error(error)
            self.close()
            return
        if self._viewport_scheduler.selection_update_pending:
            message = "Updating selected-value index"
        else:
            message = "Loading view" if self._viewport_scheduler.active_request_generation is not None else "Ready"
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
        message = "Loading view" if self._viewport_scheduler.active_request_generation is not None else "Ready"
        self._set_transient_status(message)

    def _activate_snapshot(self, snapshot: TiledPointsRenderSnapshot) -> TiledPointsRenderResult:
        """Submit a snapshot through model events and return synchronous acceptance.

        Two separate model events connect this runtime to the renderer.
        ``VispyTiledPointsLayer.__init__()`` connects ``render_snapshot`` to
        its ``_on_render_snapshot()`` handler. This runtime's constructor
        connects ``render_snapshot_result`` to ``_on_render_snapshot_result()``.
        For a snapshot submitted to the renderer, the call sequence is::

            Runtime: _activate_snapshot()
                |
                | emit layer.events.render_snapshot(value=snapshot)
                v
            Renderer: VispyTiledPointsLayer._on_render_snapshot()
                |
                | apply_snapshot(snapshot)
                | emit layer.events.render_snapshot_result(value=result)
                v
            Runtime: _on_render_snapshot_result()
                |
                | activation.renderer_result = result
                v
            Handlers finish;
            self._layer.events.render_snapshot(value=snapshot) returns
                |
                v
            Runtime: _activate_snapshot() reads the result and updates status
                |
                v
            Runtime: finally clears self._render_activation_state
                |
                v
            Runtime: _activate_snapshot() returns the result

        Both events belong to the same layer model, and their connected
        handlers run synchronously on the GUI thread. The result handler
        updates the same activation object held locally by this method, so
        its result is available when the original event emission returns.
        This acknowledges synchronous activation, not completion of a GPU draw.

        This method alone sets and clears ``self._render_activation_state``
        for each activation attempt. It assigns the activation object,
        emits ``self._layer.events.render_snapshot(value=snapshot)``, and
        processes the reply before its ``finally`` block clears the attribute.
        The result handler (``_on_render_snapshot_result()``) only validates
        and records feedback; it neither clears state nor commits display status.
        ``finally`` also runs if the event call or reply processing raises.
        It does not wait for a background task; all these handlers execute
        synchronously on the GUI thread.

        Closure disables callbacks and further dispatch but leaves cleanup to
        this method. A reentrant activation is rejected until its owner exits.

        Missing feedback, an over-budget snapshot, or closure returns ``applied=False``;
        exceptions clear the runtime's state reference and propagate to the
        scheduler, which still sends rejection feedback before dispatching
        another viewport.
        """
        rejected = TiledPointsRenderResult(snapshot.request_generation, snapshot.selection_generation, applied=False)
        if self._closed:
            return rejected
        if not snapshot.within_budget:
            # This is a metadata-only viewport result, not renderer input. Keep
            # the active visual untouched and report why it was not replaced.
            self._set_transient_status(f"{snapshot.budget_message}; retaining the previous view")
            return rejected

        status = _status_from_snapshot(snapshot)
        if self._render_activation_state is not None:
            raise RuntimeError("A render snapshot was submitted while another activation was pending.")
        activation = _RenderActivationState(
            request_generation=snapshot.request_generation,
            selection_generation=snapshot.selection_generation,
        )
        # Share this same object with the renderer-result and error handlers.
        self._render_activation_state = activation
        try:
            self._layer.events.render_snapshot(value=snapshot)
            if self._closed:
                return rejected
            result = activation.renderer_result
            if result is None:
                # Model events are synchronous: the renderer must have replied
                # before the snapshot event emission returns.
                self._publish_error(RuntimeError("Renderer did not acknowledge the submitted snapshot."))
                return rejected
            if result.applied:
                self._active_status = status
                self._layer.display_status = status
            elif activation.error is None:
                self._publish_error(RuntimeError("Renderer declined the submitted snapshot without an error."))
            # Status listeners may close the runtime while processing the reply.
            if self._closed:
                return rejected
            return result
        finally:
            self._render_activation_state = None

    @Slot(object)
    def _on_render_snapshot_result(self, event: Event) -> None:
        """Validate and record one reply; the activation caller owns finalization."""
        if self._closed:
            return
        result = event.value
        if not isinstance(result, TiledPointsRenderResult):
            raise ValueError("The render-result event must carry TiledPointsRenderResult.")
        activation = self._render_activation_state
        if activation is None:
            raise RuntimeError("Renderer acknowledged a snapshot when no activation was pending.")
        if (
            result.request_generation != activation.request_generation
            or result.selection_generation != activation.selection_generation
        ):
            raise ValueError("Renderer result does not match the pending snapshot generation.")
        if activation.renderer_result is not None:
            raise RuntimeError("Renderer acknowledged the same snapshot more than once.")
        # Leave the state installed: _activate_snapshot() reads this reply,
        # updates display status, and clears state in its own finally block.
        activation.renderer_result = result

    @Slot(int, object)
    def _on_viewport_failed(self, request_generation: int, failure: _CacheSessionFailure) -> None:
        if self._closed or request_generation != self._viewport_scheduler.request_generation:
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
        activation = self._render_activation_state
        if activation is not None:
            # Preserve the renderer's specific failure so the subsequent
            # ``applied=False`` acknowledgement does not publish a second,
            # generic renderer-declined error.
            activation.error = event.value
        self._set_transient_status(f"Display failed: {_error_message(event.value)}")

    @Slot()
    def _on_session_closed(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._disconnect_runtime_listeners()
        self._viewport_scheduler.close()

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
            (self._viewport_scheduler.viewport_failed, self._on_viewport_failed),
            (self._session.dataset_available, self._on_dataset_available),
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
    elif snapshot.estimated_point_count == 0:
        message = "No points in view"
    elif snapshot.omitted_value_ids:
        message = "Ready; some selected values are not represented at this sampled LOD"
    else:
        message = "Ready"
    if snapshot.budget_message is not None:
        message = f"{message}; {snapshot.budget_message}"
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
