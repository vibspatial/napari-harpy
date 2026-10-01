"""Schedule latest-only tiled-points viewport work on the GUI thread.

Participants and responsibilities
--------------------------------
The diagram uses short class names for these implementations:

``_TiledPointsLayerRuntime``
    GUI-side integration that submits viewports and supplies the synchronous
    ``activate_snapshot`` callback, including renderer events and status updates.

    Module: ``napari_harpy.viewer.tiled_points.runtime.layer_runtime``

``_TiledPointsViewportScheduler``
    GUI-side scheduling with one active request and one latest pending submission;
    rejects stale results and forwards activation feedback before the next request.

    Module: ``napari_harpy.viewer.tiled_points.runtime.viewport_scheduler``

``_TiledPointsCacheSession``
    GUI-side interface that transports requests and results across the worker-thread
    boundary and manages the worker lifecycle.

    Module: ``napari_harpy.viewer.tiled_points.runtime.cache_session``

``_TiledPointsCacheWorker``
    Worker-thread owner of the reader and cached data; evaluates LOD and rendering
    limits, reuses the accepted batch when possible, or prepares a replacement.

    Module: ``napari_harpy.viewer.tiled_points.runtime.cache_session``

``_PointsCacheReader``
    Cache metadata, viewport tile planning, and physical Zarr payload reads.

    Module: ``napari_harpy.core.multi_scale_cache_points_zarr.reader``

A render batch (``TiledPointsRenderBatch``) holds one immutable NumPy point array
prepared for rendering from the required tiles. Each row contains a cache-relative
position and a value ID. This is CPU-side data, not the renderer's GPU VBO.
A snapshot (``TiledPointsRenderSnapshot``) references this batch alongside request
identity, LOD, and status metadata. These types and ``TiledPointsRenderResult``
live in ``napari_harpy.viewer.tiled_points.contracts``.

Worker-owned cached data
-----------------------
``_RetainedViewport``
    Original viewport bounds paired with a snapshot and its packed render batch.
    The worker keeps one accepted ``_retained_viewport`` and may also hold a
    ``_pending_viewport`` awaiting GUI acceptance, not a history of past viewports.

    Module: ``napari_harpy.viewer.tiled_points.runtime.cache_session``

``_CpuTileResidency``
    Byte-bounded LRU of decoded tile locations and value IDs, reusable when a
    replacement batch must be packed. This is separate from the retained batch.

    Module: ``napari_harpy.viewer.tiled_points.runtime.residency``

Neither cache is owned by the viewport_scheduler. Its ``_pending_submission`` is a
viewport request waiting to be dispatched, not the worker's prepared candidate
in ``_pending_viewport``.

Request flow and thread boundary
--------------------------------
The worker evaluates LOD before deciding whether a batch can be reused::

    _TiledPointsLayerRuntime
            |
            v
    _TiledPointsViewportScheduler
            |
            v
    _TiledPointsCacheSession
            |
            | Qt queued signals cross the thread boundary
            v
    _TiledPointsCacheWorker
        |-- hard-limit rejection: metadata-only snapshot
        |-- retained viewport reusable:
        |     create a snapshot with updated request metadata
        |     reuse the same TiledPointsRenderBatch and its point array
        |     (no point-array copying or repacking)
        `-- replacement: reuse decoded CPU tiles
                         read only missing payloads via _PointsCacheReader / Zarr
                         pack a new batch

Napari-facing code submits work to the viewport_scheduler. The scheduler and
session remain on the GUI thread; only the worker owns and accesses the cache
reader and its Zarr resources. A retained-batch hit performs no tile planning,
payload reads, or packing. A replacement whose tiles are all CPU-resident still
requires packing, but no payload reads.

Results return through the session to ``_on_viewport_ready()``. The scheduler
activates only current results and sends acceptance back through
``session.acknowledge_render_result()``. The worker promotes a matching pending
candidate only for ``applied=True``; rejection preserves its previous retained
viewport. This feedback reports renderer acceptance, not GPU draw completion.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from qtpy.QtCore import QObject, Signal, Slot

from napari_harpy.viewer.tiled_points.contracts import (
    TiledPointsRenderResult,
    TiledPointsRenderSnapshot,
    TiledPointsViewportState,
    _ViewportRequest,
)
from napari_harpy.viewer.tiled_points.runtime.cache_session import (
    _CacheSessionFailure,
    _CacheSessionState,
    _TiledPointsCacheSession,
)


@dataclass(frozen=True)
class _ViewportSubmission:
    """Retain the latest GUI-stamped viewport until it can be dispatched."""

    request_generation: int
    selection_generation: int
    viewport: TiledPointsViewportState


class _TiledPointsViewportScheduler(QObject):
    """Own request generations and a one-active/one-latest-pending mailbox.

    The scheduler runs on the GUI thread and never performs cache IO. A new
    viewport synchronously advances ``request_generation`` and replaces the
    pending submission. At most one request is dispatched to the serial cache
    worker. A completion may warm worker-owned CPU residency, but it is passed
    to ``activate_snapshot`` only when both its request and selection
    generations remain current.

    The required ``activate_snapshot`` callback runs synchronously on the GUI
    thread and returns a matching ``TiledPointsRenderResult``. The integration
    owns renderer events and status updates; this scheduler only validates
    and queues the returned acceptance before the next viewport dispatch.
    Only an accepted candidate can replace the worker's retained batch and
    original viewport bounds; rejected completions may still warm the decoded
    CPU tile cache. No vertex arrays are inspected on this viewport_scheduler.

    Selection changes use the same stale-result boundary. Accepting a change
    advances ``selection_generation`` immediately, invalidates old viewport
    activation, and schedules the latest viewport after the selected-value
    index succeeds or the previous committed selection is retained on failure.

    Viewport scheduling coalesces requests according to worker occupancy; it
    is not a time-based debounce. An idle worker receives a viewport request
    immediately. While that request is active, newer viewports only replace
    the one pending submission::

        viewport 1 -> active worker request
        viewport 2 -> pending
        viewport 3 -> replaces pending viewport 2
        viewport 1 finishes -> reject stale activation; dispatch viewport 3

    A synchronous Zarr read that has already started is allowed to finish and
    may warm CPU tile residency. Its snapshot is never activated after a newer
    request generation has been submitted.

    If integration profiling shows that reads frequently finish between rapid
    camera events, a short GUI-side debounce before submission may be evaluated
    as an additional optimization. It is deliberately not part of this
    scheduler policy: any debounce should be justified by measured dispatch
    churn and must not delay ordinary isolated viewport updates.
    """

    viewport_failed = Signal(int, object)

    def __init__(
        self,
        session: _TiledPointsCacheSession,
        *,
        activate_snapshot: Callable[[TiledPointsRenderSnapshot], TiledPointsRenderResult],
        initial_requested_value_ids: tuple[int, ...] | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        _require_requested_value_ids(initial_requested_value_ids)
        self._session = session
        self._activate_snapshot = activate_snapshot
        self._request_generation = 0
        self._selection_generation = int(initial_requested_value_ids is not None)
        self._desired_value_ids = initial_requested_value_ids
        self._latest_submission: _ViewportSubmission | None = None
        self._active_request: _ViewportRequest | None = None
        self._pending_submission: _ViewportSubmission | None = None
        # A proper initial subset must become resident before the first viewport
        # is dispatched; otherwise startup could briefly request all values.
        self._selection_update_pending = initial_requested_value_ids is not None
        self._selection_failure_pending = False
        self._initial_subset_uncommitted = initial_requested_value_ids is not None
        self._block_viewport_after_initial_selection_failure = False
        self._closed = False

        session.ready.connect(self._on_session_ready)
        session.value_selection_ready.connect(self._on_value_selection_ready)
        session.viewport_ready.connect(self._on_viewport_ready)
        session.viewport_failed.connect(self._on_viewport_failed)
        session.failed.connect(self._on_session_failed)
        session.state_changed.connect(self._on_session_state_changed)
        session.closed.connect(self.close)

    @property
    def request_generation(self) -> int:
        """Return the latest assigned viewport request generation."""
        return self._request_generation

    @property
    def selection_generation(self) -> int:
        """Return the latest accepted value-selection generation."""
        return self._selection_generation

    @property
    def active_request_generation(self) -> int | None:
        """Return the one currently dispatched request generation, if any."""
        return None if self._active_request is None else self._active_request.request_generation

    @property
    def pending_request_generation(self) -> int | None:
        """Return the replaceable pending request generation, if any."""
        return None if self._pending_submission is None else self._pending_submission.request_generation

    @property
    def selection_update_pending(self) -> bool:
        """Return whether viewport dispatch waits for a value-index update."""
        return self._selection_update_pending

    def submit_viewport(self, viewport: TiledPointsViewportState) -> int:
        """Stamp and submit or retain the newest immutable viewport state.

        The request crosses into the cache worker only when the latest-request
        mailbox permits dispatch::

            napari GUI thread
                    |
                    v
            viewport_scheduler.submit_viewport()
                    |
                    v
            one-active/one-latest-pending mailbox
                    |
                    | only when dispatch is permitted
                    v
            viewport_scheduler._dispatch_pending()
                    |
                    v
            session.request_viewport()
                    |
                    | Qt queued signal
                    v
            worker.read_viewport_snapshot()
                    |
                    v
            cache reader and Zarr access
        """
        self._require_open()
        if not isinstance(viewport, TiledPointsViewportState):
            raise ValueError("`viewport` must be TiledPointsViewportState.")
        self._request_generation += 1
        submission = _ViewportSubmission(
            request_generation=self._request_generation,
            selection_generation=self._selection_generation,
            viewport=viewport,
        )
        self._latest_submission = submission
        self._pending_submission = submission
        self._dispatch_pending()
        return submission.request_generation

    def set_selected_value_ids(self, requested_value_ids: tuple[int, ...] | None) -> bool:
        """Request a selected-value index change and invalidate old viewport work.

        Value-index loading runs on the reader worker. Once the worker commits
        the selection, the scheduler may dispatch its latest viewport::

            napari GUI thread
                    |
                    v
            viewport_scheduler.set_selected_value_ids()
                    |
                    v
            session.set_selected_value_ids()
                    |
                    | Qt queued signal
                    v
            worker.update_selected_value_index()
                    |
                    v
            cache_reader.load_selected_value_index()
                    |
                    v
            worker reports committed selection
                    |
                    v
            scheduler dispatches latest retained viewport
        """
        self._require_open()
        _require_requested_value_ids(requested_value_ids)
        if requested_value_ids == self._desired_value_ids and not self._block_viewport_after_initial_selection_failure:
            return False
        startup = self._session.state in {
            _CacheSessionState.NEW,
            _CacheSessionState.STARTING,
        }
        session_change_pending = False
        if not startup:
            session_change_pending = self._session.set_selected_value_ids(requested_value_ids)
            if not session_change_pending and requested_value_ids != self._session.selected_value_ids:
                return False
        self._desired_value_ids = requested_value_ids
        self._selection_generation += 1
        self._selection_update_pending = (startup and requested_value_ids is not None) or session_change_pending
        self._selection_failure_pending = False
        self._block_viewport_after_initial_selection_failure = False
        if requested_value_ids is None:
            self._initial_subset_uncommitted = False
        self._pending_submission = None
        if self._latest_submission is not None:
            # A selection change needs a fresh request generation even when the
            # camera stayed fixed, because the previous payload identity is stale.
            self._request_generation += 1
            submission = _ViewportSubmission(
                request_generation=self._request_generation,
                selection_generation=self._selection_generation,
                viewport=self._latest_submission.viewport,
            )
            self._latest_submission = submission
            self._pending_submission = submission
        self._dispatch_pending()
        return True

    @Slot()
    def close(self) -> None:
        """Stop scheduling and discard active/pending GUI-side references."""
        if self._closed:
            return
        self._closed = True
        self._active_request = None
        self._pending_submission = None
        self._latest_submission = None

    @Slot()
    def _on_session_ready(self) -> None:
        if self._selection_update_pending and self._session.state is _CacheSessionState.READY:
            if self._desired_value_ids is None:
                self._selection_update_pending = False
                self._initial_subset_uncommitted = False
            elif self._session.set_selected_value_ids(self._desired_value_ids):
                return
            else:
                self._selection_update_pending = False
                self._initial_subset_uncommitted = False
        self._dispatch_pending()

    @Slot(object, int)
    def _on_value_selection_ready(self, selected_value_ids: tuple[int, ...] | None, resident_bytes: int) -> None:
        del selected_value_ids, resident_bytes
        if self._closed:
            return
        self._selection_update_pending = False
        self._selection_failure_pending = False
        self._initial_subset_uncommitted = False
        self._block_viewport_after_initial_selection_failure = False
        self._dispatch_pending()

    @Slot(object)
    def _on_viewport_ready(self, snapshot: TiledPointsRenderSnapshot) -> None:
        if self._closed:
            return
        active = self._active_request
        if active is None or snapshot.request_generation != active.request_generation:
            return
        latest = self._latest_submission
        # Stale snapshots and failed activation calls must release the worker's
        # pending candidate without replacing its last accepted retained batch.
        result = TiledPointsRenderResult(snapshot.request_generation, snapshot.selection_generation, applied=False)
        try:
            if (
                latest is not None
                and snapshot.request_generation == latest.request_generation
                and snapshot.selection_generation == self._selection_generation
            ):
                # The runtime callback (self._activate_snapshot) submits the snapshot
                # through layer events and collects the renderer's synchronous reply
                # before returning. The result reports acceptance, not completion
                # of the GPU draw.
                activation_result = self._activate_snapshot(snapshot)
                if self._closed:
                    return
                if not isinstance(activation_result, TiledPointsRenderResult):
                    raise ValueError("Snapshot activation must return TiledPointsRenderResult.")
                if (
                    activation_result.request_generation != active.request_generation
                    or activation_result.selection_generation != active.selection_generation
                ):
                    raise ValueError("Snapshot activation result does not match the active viewport request.")
                result = activation_result
        finally:
            # Keep _active_request set until activation feedback has been queued
            # (via self._session.acknowledge_render_result()).
            # Any viewport submitted during activation waits in _pending_submission.
            # This lets the worker accept or discard the current pending candidate
            # before processing the next queued viewport request.
            self._session.acknowledge_render_result(result)
            self._active_request = None
            self._dispatch_pending()

    @Slot(int, object)
    def _on_viewport_failed(self, request_generation: int, failure: _CacheSessionFailure) -> None:
        if self._closed:
            return
        active = self._active_request
        if active is not None and request_generation == active.request_generation:
            self._active_request = None
        self.viewport_failed.emit(request_generation, failure)
        self._dispatch_pending()

    @Slot(object)
    def _on_session_failed(self, failure: _CacheSessionFailure) -> None:
        if self._closed:
            return
        if failure.phase == "selection" and self._selection_update_pending:
            if self._initial_subset_uncommitted:
                # There is no preceding user-approved selection to fall back to:
                # dispatching here would issue the forbidden startup all-values read.
                self._selection_update_pending = False
                self._block_viewport_after_initial_selection_failure = True
            else:
                self._selection_failure_pending = True

    @Slot(object)
    def _on_session_state_changed(self, state: _CacheSessionState) -> None:
        if self._closed:
            return
        if state is _CacheSessionState.READY and self._selection_failure_pending:
            self._selection_update_pending = False
            self._selection_failure_pending = False
            self._dispatch_pending()

    def _dispatch_pending(self) -> None:
        """Dispatch the latest pending viewport when scheduling permits.

        Wait while another request is active or the session/selection is not ready.
        On dispatch, clear ``_pending_submission``, set ``_active_request``, and
        queue its worker read. ``_latest_submission`` remains available for
        freshness checks when the result arrives.
        """
        if (
            self._closed
            or self._active_request is not None
            or self._pending_submission is None
            or self._selection_update_pending
            or self._block_viewport_after_initial_selection_failure
            or self._session.state is not _CacheSessionState.READY
        ):
            return
        submission = self._pending_submission
        request = _ViewportRequest(
            request_generation=submission.request_generation,
            selection_generation=submission.selection_generation,
            requested_value_ids=self._session.selected_value_ids,
            viewport=submission.viewport,
        )
        self._pending_submission = None
        self._active_request = request
        self._session.request_viewport(request)

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("The tiled-points viewport scheduler is closed.")


def _require_requested_value_ids(value: object) -> None:
    if value is None:
        return
    if (
        not isinstance(value, tuple)
        or not value
        or any(not isinstance(value_id, int) or isinstance(value_id, bool) or value_id < 0 for value_id in value)
        or tuple(sorted(set(value))) != value
    ):
        raise ValueError("`requested_value_ids` must be None or sorted unique nonnegative integers.")
