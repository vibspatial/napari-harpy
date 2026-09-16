from __future__ import annotations

import pytest
from qtpy.QtCore import QObject, Signal

from napari_harpy.viewer.tiled_points.contracts import (
    TiledPointsRenderBatch,
    TiledPointsRenderResult,
    TiledPointsRenderSnapshot,
    TiledPointsViewportState,
    _ViewportRequest,
)
from napari_harpy.viewer.tiled_points.runtime.cache_session import (
    _CacheSessionFailure,
    _CacheSessionState,
)
from napari_harpy.viewer.tiled_points.runtime.viewport_scheduler import _TiledPointsViewportScheduler

_GENERATION_ID = "12345678-1234-5678-9234-567812345678"


class _ControllableSession(QObject):
    ready = Signal()
    value_selection_ready = Signal(object, int)
    viewport_ready = Signal(object)
    viewport_failed = Signal(int, object)
    failed = Signal(object)
    state_changed = Signal(object)
    closed = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.state = _CacheSessionState.READY
        self.selected_value_ids: tuple[int, ...] | None = None
        self.viewport_requests: list[_ViewportRequest] = []
        self.requested_selection: tuple[int, ...] | None = None
        self.render_results: list[TiledPointsRenderResult] = []

    def acknowledge_render_result(self, result: TiledPointsRenderResult) -> None:
        self.render_results.append(result)

    def request_viewport(self, request: _ViewportRequest) -> None:
        self.viewport_requests.append(request)

    def set_selected_value_ids(self, requested_value_ids: tuple[int, ...] | None) -> bool:
        if requested_value_ids == self.selected_value_ids:
            return False
        self.requested_selection = requested_value_ids
        self.state = _CacheSessionState.UPDATING_SELECTED_VALUE_INDEX
        self.state_changed.emit(self.state)
        return True

    def complete_viewport(self, request: _ViewportRequest, *, level: int = 0) -> None:
        self.viewport_ready.emit(_snapshot(request, level=level))

    def complete_selection(self) -> None:
        self.selected_value_ids = self.requested_selection
        self.state = _CacheSessionState.READY
        self.state_changed.emit(self.state)
        self.value_selection_ready.emit(self.selected_value_ids, 24)

    def fail_selection(self) -> None:
        self.failed.emit(_CacheSessionFailure("selection", "builtins.ValueError", "selection failed"))
        self.state = _CacheSessionState.READY
        self.state_changed.emit(self.state)


def _snapshot(request: _ViewportRequest, *, level: int = 0) -> TiledPointsRenderSnapshot:
    return TiledPointsRenderSnapshot(
        cache_generation_id=_GENERATION_ID,
        request_generation=request.request_generation,
        selection_generation=request.selection_generation,
        requested_value_ids=request.requested_value_ids,
        level=level,
        level_kind="exact" if level == 0 else "bridge" if level == 1 else "spatial",
        within_budget=True,
        estimated_point_count=0,
        omitted_value_ids=(),
        rendered_tile_count=0,
        render_batch=TiledPointsRenderBatch.empty(),
    )


def _accept_snapshot(snapshot: TiledPointsRenderSnapshot) -> TiledPointsRenderResult:
    return TiledPointsRenderResult(snapshot.request_generation, snapshot.selection_generation, applied=True)


def _viewport(x_min: float) -> TiledPointsViewportState:
    return TiledPointsViewportState(
        displayed_axes=(0, 1),
        x_min=x_min,
        y_min=0.0,
        x_max=x_min + 10.0,
        y_max=10.0,
        canvas_width=100,
        canvas_height=100,
        hard_render_point_budget=100,
        screen_density_budget=100,
    )


def test_viewport_scheduler_keeps_one_active_and_only_the_latest_pending_request() -> None:
    session = _ControllableSession()
    published: list[TiledPointsRenderSnapshot] = []

    def activate(snapshot: TiledPointsRenderSnapshot) -> TiledPointsRenderResult:
        published.append(snapshot)
        return _accept_snapshot(snapshot)

    viewport_scheduler = _TiledPointsViewportScheduler(session, activate_snapshot=activate)  # type: ignore[arg-type]

    assert viewport_scheduler.submit_viewport(_viewport(0.0)) == 1
    assert viewport_scheduler.submit_viewport(_viewport(10.0)) == 2
    assert viewport_scheduler.submit_viewport(_viewport(20.0)) == 3
    assert [request.request_generation for request in session.viewport_requests] == [1]
    assert viewport_scheduler.active_request_generation == 1
    assert viewport_scheduler.pending_request_generation == 3

    session.complete_viewport(session.viewport_requests[0], level=2)

    assert published == []
    assert session.render_results[-1] == TiledPointsRenderResult(1, 0, False)
    assert [request.request_generation for request in session.viewport_requests] == [1, 3]
    assert viewport_scheduler.active_request_generation == 3
    assert viewport_scheduler.pending_request_generation is None

    session.complete_viewport(session.viewport_requests[1], level=3)

    assert [snapshot.request_generation for snapshot in published] == [3]
    assert published[0].level == 3
    assert session.render_results[-1] == TiledPointsRenderResult(3, 0, True)
    assert viewport_scheduler.active_request_generation is None


@pytest.mark.parametrize("applied", [True, False], ids=["accepted", "rejected"])
def test_renderer_feedback_precedes_reentrant_viewport_dispatch(applied: bool) -> None:
    """Report A's activation result before dispatching viewport B.

    In ``_on_viewport_ready()``, the scheduler synchronously calls
    ``activation_result = self._activate_snapshot(snapshot)``.
    This test supplies an activation callback that submits B while handling
    A's snapshot, before returning A's acceptance or rejection.

    B must remain pending until the callback returns and the scheduler reports
    A's result to the session. Only then may B be dispatched.

    Exercise this ordering for both accepted and rejected activation results.
    """
    session = _ControllableSession()

    def activate(snapshot: TiledPointsRenderSnapshot) -> TiledPointsRenderResult:
        viewport_scheduler.submit_viewport(_viewport(1.0))
        assert len(session.viewport_requests) == 1
        return TiledPointsRenderResult(snapshot.request_generation, snapshot.selection_generation, applied=applied)

    viewport_scheduler = _TiledPointsViewportScheduler(session, activate_snapshot=activate)  # type: ignore[arg-type]
    original_request = session.request_viewport

    def dispatch(request: _ViewportRequest) -> None:
        if request.request_generation == 2:
            assert session.render_results == [TiledPointsRenderResult(1, 0, applied)]
        original_request(request)

    session.request_viewport = dispatch
    viewport_scheduler.submit_viewport(_viewport(0.0))
    session.complete_viewport(session.viewport_requests[0])
    assert viewport_scheduler.active_request_generation == 2


def test_activation_exception_rejects_candidate_before_dispatching_next_viewport() -> None:
    """Reject request 1 before dispatching request 2 when activation raises.

    In ``_on_viewport_ready()``, the scheduler calls
    ``activation_result = self._activate_snapshot(snapshot)`` for request 1
    (viewport A). The test callback submits request 2 (viewport B), then
    raises an exception instead of returning an activation result.

    Verify that the scheduler's finally block reports request 1 as rejected
    before dispatching request 2, while allowing the exception to propagate.
    Request 2 must subsequently complete successfully, leaving no active request.
    """
    session = _ControllableSession()

    def activate(snapshot: TiledPointsRenderSnapshot) -> TiledPointsRenderResult:
        if snapshot.request_generation == 1:
            viewport_scheduler.submit_viewport(_viewport(1.0))
            raise RuntimeError("synthetic activation failure")
        return _accept_snapshot(snapshot)

    viewport_scheduler = _TiledPointsViewportScheduler(session, activate_snapshot=activate)  # type: ignore[arg-type]
    original_request = session.request_viewport

    def dispatch(request: _ViewportRequest) -> None:
        if request.request_generation == 2:
            assert session.render_results == [TiledPointsRenderResult(1, 0, False)]
        original_request(request)

    session.request_viewport = dispatch
    viewport_scheduler.submit_viewport(_viewport(0.0))
    # Call the completion handler directly so pytest can catch the callback's
    # exception without routing it through Qt's unhandled-slot exception hook.
    with pytest.raises(RuntimeError, match="synthetic activation failure"):
        viewport_scheduler._on_viewport_ready(_snapshot(session.viewport_requests[0]))

    assert viewport_scheduler.active_request_generation == 2
    session.complete_viewport(session.viewport_requests[-1])
    assert session.render_results[-1] == TiledPointsRenderResult(2, 0, True)
    assert viewport_scheduler.active_request_generation is None


@pytest.mark.parametrize(
    "invalid_result",
    [None, TiledPointsRenderResult(2, 0, True), TiledPointsRenderResult(1, 1, True)],
    ids=["missing-result", "wrong-request", "wrong-selection"],
)
def test_invalid_activation_result_cannot_promote_a_retained_candidate(invalid_result: object) -> None:
    """Reject activation replies that violate the callback contract.

    Normal activation returns a TiledPointsRenderResult matching the active
    request and selection generations, whether accepted or rejected.
    This test deliberately returns None or mismatched generations: these are
    contract violations, not normal renderer rejection.

    Verify that the scheduler raises ValueError, reports the active candidate
    as rejected, releases its request slot, and can process a subsequent request.
    """
    session = _ControllableSession()

    def activate(snapshot: TiledPointsRenderSnapshot) -> TiledPointsRenderResult:
        if snapshot.request_generation == 1:
            return invalid_result  # type: ignore[return-value]
        return _accept_snapshot(snapshot)

    viewport_scheduler = _TiledPointsViewportScheduler(session, activate_snapshot=activate)  # type: ignore[arg-type]
    viewport_scheduler.submit_viewport(_viewport(0.0))
    with pytest.raises(ValueError, match="Snapshot activation"):
        viewport_scheduler._on_viewport_ready(_snapshot(session.viewport_requests[0]))

    assert session.render_results == [TiledPointsRenderResult(1, 0, False)]
    assert viewport_scheduler.active_request_generation is None
    viewport_scheduler.submit_viewport(_viewport(1.0))
    session.complete_viewport(session.viewport_requests[-1])
    assert session.render_results[-1] == TiledPointsRenderResult(2, 0, True)


def test_close_during_activation_rejects_result_and_does_not_dispatch_pending_viewport() -> None:
    session = _ControllableSession()
    activated: list[int] = []

    def activate(snapshot: TiledPointsRenderSnapshot) -> TiledPointsRenderResult:
        activated.append(snapshot.request_generation)
        viewport_scheduler.submit_viewport(_viewport(1.0))
        viewport_scheduler.close()
        return _accept_snapshot(snapshot)

    viewport_scheduler = _TiledPointsViewportScheduler(session, activate_snapshot=activate)  # type: ignore[arg-type]
    viewport_scheduler.submit_viewport(_viewport(0.0))
    request = session.viewport_requests[0]
    session.complete_viewport(request)

    assert session.render_results == [TiledPointsRenderResult(1, 0, False)]
    assert len(session.viewport_requests) == 1
    assert viewport_scheduler.active_request_generation is None
    assert viewport_scheduler.pending_request_generation is None
    session.complete_viewport(request)
    assert activated == [1]


def test_selection_change_invalidates_active_viewport_and_replans_latest() -> None:
    session = _ControllableSession()
    published: list[TiledPointsRenderSnapshot] = []

    def activate(snapshot: TiledPointsRenderSnapshot) -> TiledPointsRenderResult:
        published.append(snapshot)
        return _accept_snapshot(snapshot)

    viewport_scheduler = _TiledPointsViewportScheduler(session, activate_snapshot=activate)  # type: ignore[arg-type]

    viewport_scheduler.submit_viewport(_viewport(0.0))
    active_s1 = session.viewport_requests[0]
    assert viewport_scheduler.set_selected_value_ids((1,))
    assert viewport_scheduler.selection_generation == 1
    assert viewport_scheduler.request_generation == 2

    session.complete_viewport(active_s1)
    assert published == []
    assert len(session.viewport_requests) == 1

    session.complete_selection()
    active_s2 = session.viewport_requests[1]
    assert active_s2.request_generation == 2
    assert active_s2.selection_generation == 1
    assert active_s2.requested_value_ids == (1,)

    session.complete_viewport(active_s2)
    assert [snapshot.request_generation for snapshot in published] == [2]


def test_selection_failure_replans_latest_with_previous_committed_values() -> None:
    session = _ControllableSession()
    viewport_scheduler = _TiledPointsViewportScheduler(session, activate_snapshot=_accept_snapshot)  # type: ignore[arg-type]

    viewport_scheduler.submit_viewport(_viewport(0.0))
    first = session.viewport_requests[0]
    assert viewport_scheduler.set_selected_value_ids((1,))
    session.complete_viewport(first)
    session.fail_selection()

    retry = session.viewport_requests[1]
    assert retry.selection_generation == 1
    assert retry.requested_value_ids is None


def test_initial_subset_is_committed_before_the_first_viewport_dispatch() -> None:
    session = _ControllableSession()
    session.state = _CacheSessionState.NEW
    viewport_scheduler = _TiledPointsViewportScheduler(  # type: ignore[arg-type]
        session,
        activate_snapshot=_accept_snapshot,
        initial_requested_value_ids=(1,),
    )

    viewport_scheduler.submit_viewport(_viewport(0.0))
    assert session.viewport_requests == []

    session.state = _CacheSessionState.READY
    session.ready.emit()
    assert session.requested_selection == (1,)
    assert session.viewport_requests == []

    session.complete_selection()
    assert len(session.viewport_requests) == 1
    assert session.viewport_requests[0].requested_value_ids == (1,)


def test_startup_selection_replacement_retains_only_latest_subset() -> None:
    session = _ControllableSession()
    session.state = _CacheSessionState.NEW
    viewport_scheduler = _TiledPointsViewportScheduler(  # type: ignore[arg-type]
        session,
        activate_snapshot=_accept_snapshot,
        initial_requested_value_ids=(0,),
    )

    assert viewport_scheduler.set_selected_value_ids((1,))
    viewport_scheduler.submit_viewport(_viewport(0.0))
    session.state = _CacheSessionState.READY
    session.ready.emit()

    assert session.requested_selection == (1,)
    assert session.viewport_requests == []


def test_initial_subset_failure_never_falls_back_to_an_all_values_viewport() -> None:
    session = _ControllableSession()
    session.state = _CacheSessionState.NEW
    viewport_scheduler = _TiledPointsViewportScheduler(  # type: ignore[arg-type]
        session,
        activate_snapshot=_accept_snapshot,
        initial_requested_value_ids=(1,),
    )
    viewport_scheduler.submit_viewport(_viewport(0.0))
    session.state = _CacheSessionState.READY
    session.ready.emit()

    session.fail_selection()
    assert session.viewport_requests == []

    # An explicit later choice of all values reconciles with the session's
    # already-committed default and releases the retained latest viewport.
    assert viewport_scheduler.set_selected_value_ids(None)
    assert len(session.viewport_requests) == 1
    assert session.viewport_requests[0].requested_value_ids is None
