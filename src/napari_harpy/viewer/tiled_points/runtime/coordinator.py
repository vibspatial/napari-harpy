"""Coordinate latest-only tiled-points viewport work on the GUI thread.

The runtime ownership and thread boundary are::

    napari integration
            |
            v
    _TiledPointsViewportCoordinator
            |
            v
    _TiledPointsCacheSession
            |
            | Qt queued signals cross the thread boundary
            v
    _TiledPointsCacheWorker
            |
            v
    _PointsCacheReader / Zarr

Napari-facing code submits work to the coordinator. The coordinator and
session remain on the GUI thread; only the worker owns and accesses the cache
reader and its Zarr resources.
"""

from __future__ import annotations

from dataclasses import dataclass

from qtpy.QtCore import QObject, Signal, Slot

from napari_harpy.viewer.tiled_points.contracts import (
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


class _TiledPointsViewportCoordinator(QObject):
    """Own request generations and a one-active/one-latest-pending mailbox.

    The coordinator runs on the GUI thread and never performs cache IO. A new
    viewport synchronously advances ``request_generation`` and replaces the
    pending submission. At most one request is dispatched to the serial cache
    worker. A completion may warm worker-owned CPU residency, but it is emitted
    as an active snapshot only when both its request and selection generations
    remain current.

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
    coordinator policy: any debounce should be justified by measured dispatch
    churn and must not delay ordinary isolated viewport updates.
    """

    snapshot_ready = Signal(object)
    viewport_failed = Signal(int, object)

    def __init__(
        self,
        session: _TiledPointsCacheSession,
        *,
        initial_requested_value_ids: tuple[int, ...] | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        _require_requested_value_ids(initial_requested_value_ids)
        self._session = session
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
            coordinator.submit_viewport()
                    |
                    v
            one-active/one-latest-pending mailbox
                    |
                    | only when dispatch is permitted
                    v
            coordinator._dispatch_pending()
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
        the selection, the coordinator may dispatch its latest viewport::

            napari GUI thread
                    |
                    v
            coordinator.set_selected_value_ids()
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
            coordinator dispatches latest retained viewport
        """
        self._require_open()
        _require_requested_value_ids(requested_value_ids)
        if requested_value_ids == self._desired_value_ids and not self._block_viewport_after_initial_selection_failure:
            return False
        startup = self._session.state in {
            _CacheSessionState.NEW,
            _CacheSessionState.STARTING,
            _CacheSessionState.LOADING_BUCKET_INDEXES,
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
        self._active_request = None
        latest = self._latest_submission
        if (
            latest is not None
            and snapshot.request_generation == latest.request_generation
            and snapshot.selection_generation == self._selection_generation
        ):
            self.snapshot_ready.emit(snapshot)
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
            raise RuntimeError("The tiled-points viewport coordinator is closed.")


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
