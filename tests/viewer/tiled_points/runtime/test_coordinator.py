from __future__ import annotations

from qtpy.QtCore import QObject, Signal

from napari_harpy.viewer.tiled_points.contracts import (
    TiledPointsRenderSnapshot,
    TiledPointsViewportState,
    _ViewportRequest,
)
from napari_harpy.viewer.tiled_points.runtime.cache_session import (
    _CacheSessionFailure,
    _CacheSessionState,
)
from napari_harpy.viewer.tiled_points.runtime.coordinator import _TiledPointsViewportCoordinator

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
        self.viewport_ready.emit(
            TiledPointsRenderSnapshot(
                cache_generation_id=_GENERATION_ID,
                request_generation=request.request_generation,
                selection_generation=request.selection_generation,
                requested_value_ids=request.requested_value_ids,
                level=level,
                level_kind="exact" if level == 0 else "bridge" if level == 1 else "spatial",
                within_budget=True,
                estimated_point_count=0,
                omitted_value_ids=(),
                tiles=(),
            )
        )

    def complete_selection(self) -> None:
        self.selected_value_ids = self.requested_selection
        self.state = _CacheSessionState.READY
        self.state_changed.emit(self.state)
        self.value_selection_ready.emit(self.selected_value_ids, 24)

    def fail_selection(self) -> None:
        self.failed.emit(_CacheSessionFailure("selection", "builtins.ValueError", "selection failed"))
        self.state = _CacheSessionState.READY
        self.state_changed.emit(self.state)


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


def test_coordinator_keeps_one_active_and_only_the_latest_pending_request() -> None:
    session = _ControllableSession()
    coordinator = _TiledPointsViewportCoordinator(session)  # type: ignore[arg-type]
    published: list[TiledPointsRenderSnapshot] = []
    coordinator.snapshot_ready.connect(published.append)

    assert coordinator.submit_viewport(_viewport(0.0)) == 1
    assert coordinator.submit_viewport(_viewport(10.0)) == 2
    assert coordinator.submit_viewport(_viewport(20.0)) == 3
    assert [request.request_generation for request in session.viewport_requests] == [1]
    assert coordinator.active_request_generation == 1
    assert coordinator.pending_request_generation == 3

    session.complete_viewport(session.viewport_requests[0], level=2)

    assert published == []
    assert [request.request_generation for request in session.viewport_requests] == [1, 3]
    assert coordinator.active_request_generation == 3
    assert coordinator.pending_request_generation is None

    session.complete_viewport(session.viewport_requests[1], level=3)

    assert [snapshot.request_generation for snapshot in published] == [3]
    assert published[0].level == 3
    assert coordinator.active_request_generation is None


def test_selection_change_invalidates_active_viewport_and_replans_latest() -> None:
    session = _ControllableSession()
    coordinator = _TiledPointsViewportCoordinator(session)  # type: ignore[arg-type]
    published: list[TiledPointsRenderSnapshot] = []
    coordinator.snapshot_ready.connect(published.append)

    coordinator.submit_viewport(_viewport(0.0))
    active_s1 = session.viewport_requests[0]
    assert coordinator.set_selected_value_ids((1,))
    assert coordinator.selection_generation == 1
    assert coordinator.request_generation == 2

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
    coordinator = _TiledPointsViewportCoordinator(session)  # type: ignore[arg-type]

    coordinator.submit_viewport(_viewport(0.0))
    first = session.viewport_requests[0]
    assert coordinator.set_selected_value_ids((1,))
    session.complete_viewport(first)
    session.fail_selection()

    retry = session.viewport_requests[1]
    assert retry.selection_generation == 1
    assert retry.requested_value_ids is None
