from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from napari._vispy.utils.qt_font import FontInfo
from napari.utils.events import Event
from qtpy.QtCore import QObject, Signal

import napari_harpy.viewer.tiled_points.runtime.cache_session as cache_session_module
from napari_harpy.core.multi_scale_cache_points_zarr.reader import (
    _CacheDatasetInfo,
    _CacheLevelInfo,
    _PointsCacheReader,
)
from napari_harpy.viewer.tiled_points.contracts import (
    TiledPointsDatasetReference,
    TiledPointsRenderResult,
    TiledPointsRenderSnapshot,
    TiledPointsRenderTile,
    TiledPointsViewportState,
    TileResidencyKey,
    _ViewportRequest,
)
from napari_harpy.viewer.tiled_points.napari.layer import TiledPointsLayerModel
from napari_harpy.viewer.tiled_points.render_batch import pack_render_tiles
from napari_harpy.viewer.tiled_points.runtime.cache_session import (
    _CacheSessionFailure,
    _CacheSessionSettings,
    _CacheSessionState,
)
from napari_harpy.viewer.tiled_points.runtime.layer_runtime import _TiledPointsLayerRuntime
from napari_harpy.viewer.tiled_points.runtime.residency import _CpuTileResidency
from napari_harpy.viewer.tiled_points.vispy.layer import VispyTiledPointsLayer

_GENERATION_ID = "12345678-1234-5678-9234-567812345678"


class _ControllableSession(QObject):
    state_changed = Signal(object)
    dataset_available = Signal(object)
    ready = Signal()
    value_selection_ready = Signal(object, int)
    viewport_ready = Signal(object)
    viewport_failed = Signal(int, object)
    failed = Signal(object)
    closed = Signal()

    def __init__(self, dataset_info: _CacheDatasetInfo) -> None:
        super().__init__()
        self.dataset_info = dataset_info
        self.state = _CacheSessionState.NEW
        self.selected_value_ids: tuple[int, ...] | None = None
        self.viewport_requests: list[_ViewportRequest] = []
        self.requested_selection: tuple[int, ...] | None = None
        self.close_count = 0
        self.render_results: list[TiledPointsRenderResult] = []

    def acknowledge_render_result(self, result: TiledPointsRenderResult) -> None:
        self.render_results.append(result)

    def start(self) -> None:
        self._set_state(_CacheSessionState.STARTING)
        self.dataset_available.emit(self.dataset_info)
        if self.state is _CacheSessionState.CLOSED:
            return
        self._set_state(_CacheSessionState.READY)
        self.ready.emit()

    def request_viewport(self, request: _ViewportRequest) -> None:
        self.viewport_requests.append(request)

    def set_selected_value_ids(self, requested_value_ids: tuple[int, ...] | None) -> bool:
        if requested_value_ids == self.selected_value_ids:
            return False
        self.requested_selection = requested_value_ids
        self._set_state(_CacheSessionState.UPDATING_SELECTED_VALUE_INDEX)
        return True

    def complete_selection(self) -> None:
        self.selected_value_ids = self.requested_selection
        self._set_state(_CacheSessionState.READY)
        self.value_selection_ready.emit(self.selected_value_ids, 24)

    def complete_viewport(self, snapshot: TiledPointsRenderSnapshot) -> None:
        self.viewport_ready.emit(snapshot)

    def fail_viewport(self, request_generation: int) -> None:
        failure = _CacheSessionFailure("viewport", "builtins.RuntimeError", "synthetic viewport failure")
        self.failed.emit(failure)
        self.viewport_failed.emit(request_generation, failure)

    def close(self) -> bool:
        if self.state in (_CacheSessionState.CLOSING, _CacheSessionState.CLOSED):
            return False
        self.close_count += 1
        self._set_state(_CacheSessionState.CLOSING)
        self._set_state(_CacheSessionState.CLOSED)
        self.closed.emit()
        return True

    def _set_state(self, state: _CacheSessionState) -> None:
        self.state = state
        self.state_changed.emit(state)


def _dataset_info(**overrides: object) -> _CacheDatasetInfo:
    values = {
        "cache_generation_id": _GENERATION_ID,
        "points_name": "transcripts",
        "value_column": "gene",
        "value_names": ("A", "B"),
        "x_origin": 0.0,
        "y_origin": 0.0,
        "x_min": 1.0,
        "x_max": 11.0,
        "y_min": 1.0,
        "y_max": 3.0,
        "levels": (
            _CacheLevelInfo(
                level=0,
                kind="exact",
                tile_size=10,
                grid_width=2,
                grid_height=1,
                max_points_per_tile=None,
                bucket_count=1,
                tile_count=2,
                point_count=4,
            ),
        ),
        "overview_point_budget": 10,
    }
    values.update(overrides)
    return _CacheDatasetInfo(**values)


def _reference(info: _CacheDatasetInfo) -> TiledPointsDatasetReference:
    return TiledPointsDatasetReference(
        cache_generation_id=info.cache_generation_id,
        points_name=info.points_name,
        value_column=info.value_column,
        value_count=len(info.value_names),
        x_origin=info.x_origin,
        y_origin=info.y_origin,
        x_min=info.x_min,
        x_max=info.x_max,
        y_min=info.y_min,
        y_max=info.y_max,
    )


def _layer(info: _CacheDatasetInfo, *, max_vertex_payload_bytes: int = 1_000_000) -> TiledPointsLayerModel:
    return TiledPointsLayerModel(
        _reference(info),
        value_palette=np.asarray(((255, 0, 0, 255), (0, 255, 0, 255)), dtype=np.uint8),
        max_vertex_payload_bytes=max_vertex_payload_bytes,
    )


def _settings() -> _CacheSessionSettings:
    return _CacheSessionSettings(
        max_selected_value_index_bytes=None,
        max_cpu_tile_bytes=1_000_000,
        max_vertex_payload_bytes=1_000_000,
    )


def _viewport(x_min: float = 0.0, *, width: float = 20.0) -> TiledPointsViewportState:
    return TiledPointsViewportState(
        displayed_axes=(0, 1),
        x_min=x_min,
        y_min=0.0,
        x_max=x_min + width,
        y_max=10.0,
        canvas_width=100,
        canvas_height=100,
        hard_render_point_budget=100,
        screen_density_budget=100,
    )


def _tile(
    layer: TiledPointsLayerModel,
    request: _ViewportRequest,
    *,
    tile_x: int = 0,
) -> TiledPointsRenderTile:
    return TiledPointsRenderTile(
        key=TileResidencyKey(
            cache_generation_id=layer.data.cache_generation_id,
            requested_value_ids=request.requested_value_ids,
            level=0,
            tile_x=tile_x,
            tile_y=0,
        ),
        tile_size=10,
        location=np.asarray(((1.0, 2.0),), dtype=np.float32),
        value_id=np.asarray((0,), dtype=np.uint32),
    )


def _snapshot(
    layer: TiledPointsLayerModel,
    request: _ViewportRequest,
    tiles: tuple[TiledPointsRenderTile, ...],
    *,
    within_budget: bool = True,
    estimated_point_count: int | None = None,
    omitted_value_ids: tuple[int, ...] = (),
    level: int = 0,
    budget_message: str | None = None,
) -> TiledPointsRenderSnapshot:
    point_count = sum(tile.point_count for tile in tiles)
    render_batch = pack_render_tiles(
        tiles,
        point_count=point_count,
        value_count=layer.data.value_count,
        max_vertex_payload_bytes=1_000_000,
    )
    return TiledPointsRenderSnapshot(
        cache_generation_id=layer.data.cache_generation_id,
        request_generation=request.request_generation,
        selection_generation=request.selection_generation,
        requested_value_ids=request.requested_value_ids,
        level=level,
        level_kind="exact" if level == 0 else "bridge" if level == 1 else "spatial",
        within_budget=within_budget,
        estimated_point_count=(point_count if estimated_point_count is None else estimated_point_count),
        omitted_value_ids=omitted_value_ids,
        rendered_tile_count=len(tiles),
        render_batch=render_batch,
        budget_message=budget_message,
    )


def _runtime(
    layer: TiledPointsLayerModel,
    session: _ControllableSession,
) -> _TiledPointsLayerRuntime:
    factory: Callable[[Path, _CacheSessionSettings], _ControllableSession] = lambda _root, _settings: session
    return _TiledPointsLayerRuntime(
        layer,
        Path("unused.zarr"),
        _settings(),
        session_factory=factory,  # type: ignore[arg-type]
    )


@pytest.fixture
def maximum_texture_size(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("napari._vispy.layers.base.get_max_texture_sizes", lambda: (8192, 2048))


def test_runtime_connects_layer_viewports_to_complete_renderer_snapshots(maximum_texture_size: None) -> None:
    info = _dataset_info()
    layer = _layer(info)
    session = _ControllableSession(info)
    runtime = _runtime(layer, session)
    visual = VispyTiledPointsLayer(layer, FontInfo())
    try:
        layer.events.viewport(value=_viewport())
        request = session.viewport_requests[-1]
        tile = _tile(layer, request)
        session.complete_viewport(_snapshot(layer, request, (tile,)))

        assert visual.active_point_count == tile.point_count
        assert layer.display_status.level == 0
        assert layer.display_status.rendered_point_count == 1
        assert layer.display_status.rendered_tile_count == 1
        assert layer.display_status.message == "Ready"
        assert session.render_results == [
            TiledPointsRenderResult(request.request_generation, request.selection_generation, applied=True)
        ]
    finally:
        runtime.close()
        visual.close()


def test_gui_status_and_renderer_activation_do_not_iterate_logical_tiles(
    maximum_texture_size: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    info = _dataset_info()
    layer = _layer(info)
    session = _ControllableSession(info)
    runtime = _runtime(layer, session)
    visual = VispyTiledPointsLayer(layer, FontInfo())
    try:
        layer.events.viewport(value=_viewport())
        request = session.viewport_requests[-1]
        tile = _tile(layer, request)
        snapshot = _snapshot(layer, request, (tile,))

        def _unexpected_tile_point_count(_tile: TiledPointsRenderTile) -> int:
            raise AssertionError("GUI activation must not inspect logical tile point counts")

        monkeypatch.setattr(TiledPointsRenderTile, "point_count", property(_unexpected_tile_point_count))
        session.complete_viewport(snapshot)

        assert visual.active_point_count == 1
        assert layer.display_status.rendered_point_count == 1
        assert layer.display_status.rendered_tile_count == 1
    finally:
        runtime.close()
        visual.close()


def test_runtime_never_submits_a_stale_snapshot_to_vispy(maximum_texture_size: None) -> None:
    info = _dataset_info()
    layer = _layer(info)
    session = _ControllableSession(info)
    runtime = _runtime(layer, session)
    visual = VispyTiledPointsLayer(layer, FontInfo())
    try:
        layer.events.viewport(value=_viewport())
        stale_request = session.viewport_requests[-1]
        layer.events.viewport(value=_viewport(10.0))

        session.complete_viewport(_snapshot(layer, stale_request, (_tile(layer, stale_request),)))

        assert visual.payload_replacement_count == 0
        assert visual.active_point_count == 0
        latest_request = session.viewport_requests[-1]
        assert latest_request.request_generation > stale_request.request_generation

        latest_tile = _tile(layer, latest_request, tile_x=1)
        session.complete_viewport(_snapshot(layer, latest_request, (latest_tile,)))

        assert visual.payload_replacement_count == 1
        assert visual.active_point_count == latest_tile.point_count
    finally:
        runtime.close()
        visual.close()


@pytest.mark.parametrize(
    "budget_message",
    [
        "View exceeds hard rendering limits: 101 points required, render limit 100 points",
        "View exceeds hard rendering limits: 1,212 vertex bytes required, limit 1,200 bytes",
    ],
    ids=["point-limit", "vertex-byte-limit"],
)
def test_runtime_retains_active_visual_for_over_budget_and_failure_then_clears_sampled_omission(
    maximum_texture_size: None,
    budget_message: str,
) -> None:
    info = _dataset_info()
    layer = _layer(info)
    session = _ControllableSession(info)
    runtime = _runtime(layer, session)
    visual = VispyTiledPointsLayer(layer, FontInfo())
    rendered_snapshots: list[TiledPointsRenderSnapshot] = []
    layer.events.render_snapshot.connect(lambda event: rendered_snapshots.append(event.value))
    try:
        layer.events.viewport(value=_viewport())
        first_request = session.viewport_requests[-1]
        tile = _tile(layer, first_request)
        session.complete_viewport(_snapshot(layer, first_request, (tile,)))
        assert len(rendered_snapshots) == 1
        assert rendered_snapshots[0].request_generation == first_request.request_generation

        layer.events.viewport(value=_viewport(10.0))
        over_budget_request = session.viewport_requests[-1]
        session.complete_viewport(
            _snapshot(
                layer,
                over_budget_request,
                (),
                within_budget=False,
                estimated_point_count=101,
                budget_message=budget_message,
            )
        )
        assert visual.active_point_count == tile.point_count
        assert len(rendered_snapshots) == 1
        assert layer.display_status.rendered_point_count == 1
        assert "retaining the previous view" in layer.display_status.message
        assert layer.display_status.message == f"{budget_message}; retaining the previous view"

        layer.events.viewport(value=_viewport(20.0))
        failed_request = session.viewport_requests[-1]
        session.fail_viewport(failed_request.request_generation)
        assert visual.active_point_count == tile.point_count
        assert layer.display_status.rendered_point_count == 1
        assert "synthetic viewport failure" in layer.display_status.message

        assert runtime.set_selected_value_ids((0,))
        session.complete_selection()
        omitted_request = session.viewport_requests[-1]
        session.complete_viewport(
            _snapshot(
                layer,
                omitted_request,
                (),
                omitted_value_ids=(0,),
                level=2,
            )
        )
        assert visual.active_point_count == 0
        assert len(rendered_snapshots) == 2
        assert layer.display_status.level == 2
        assert layer.display_status.rendered_point_count == 0
        assert layer.display_status.omitted_value_ids == (0,)
        assert layer.display_status.message == "Selected values are not represented at the sampled LOD"
    finally:
        runtime.close()
        visual.close()


def test_runtime_activates_density_fallback_and_refreshes_notice_on_batch_reuse(maximum_texture_size: None) -> None:
    info = _dataset_info()
    layer = _layer(info)
    session = _ControllableSession(info)
    runtime = _runtime(layer, session)
    visual = VispyTiledPointsLayer(layer, FontInfo())
    try:
        layer.events.viewport(value=_viewport())
        request = session.viewport_requests[-1]
        tile = _tile(layer, request)
        snapshot = _snapshot(
            layer,
            request,
            (tile,),
            budget_message="Coarsest level; above preferred screen density",
        )
        session.complete_viewport(snapshot)
        assert visual.active_point_count == tile.point_count
        assert layer.display_status.message == "Ready; Coarsest level; above preferred screen density"
        assert visual.payload_replacement_count == 1

        layer.events.viewport(value=_viewport(1.0))
        next_request = session.viewport_requests[-1]
        session.complete_viewport(
            replace(snapshot, request_generation=next_request.request_generation, budget_message=None)
        )
        assert layer.display_status.message == "Ready"
        assert visual.payload_replacement_count == 1
    finally:
        runtime.close()
        visual.close()


def test_runtime_applies_ordinary_empty_snapshot_and_clears_active_visual(maximum_texture_size: None) -> None:
    info = _dataset_info()
    layer = _layer(info)
    session = _ControllableSession(info)
    runtime = _runtime(layer, session)
    visual = VispyTiledPointsLayer(layer, FontInfo())
    try:
        layer.events.viewport(value=_viewport())
        populated_request = session.viewport_requests[-1]
        populated_tile = _tile(layer, populated_request)
        session.complete_viewport(_snapshot(layer, populated_request, (populated_tile,)))
        assert visual.active_point_count == populated_tile.point_count

        layer.events.viewport(value=_viewport(10.0))
        empty_request = session.viewport_requests[-1]
        session.complete_viewport(_snapshot(layer, empty_request, ()))

        assert visual.active_point_count == 0
        assert layer.display_status.rendered_point_count == 0
        assert layer.display_status.rendered_tile_count == 0
        assert layer.display_status.omitted_value_ids == ()
        assert layer.display_status.message == "No points in view"
    finally:
        runtime.close()
        visual.close()


def test_runtime_reports_visible_sampled_omission_without_discarding_offscreen_batch(
    maximum_texture_size: None,
) -> None:
    info = _dataset_info()
    layer = _layer(info)
    session = _ControllableSession(info)
    runtime = _runtime(layer, session)
    visual = VispyTiledPointsLayer(layer, FontInfo())
    try:
        runtime.set_selected_value_ids((0,))
        session.complete_selection()
        layer.events.viewport(value=_viewport())
        request = session.viewport_requests[-1]
        first = _snapshot(layer, request, (_tile(layer, request),), level=1)
        session.complete_viewport(first)
        layer.events.viewport(value=_viewport(width=10.0))
        request = session.viewport_requests[-1]
        session.complete_viewport(
            replace(
                first, request_generation=request.request_generation, estimated_point_count=0, omitted_value_ids=(0,)
            )
        )
        assert visual.active_point_count == 1
        assert visual.payload_replacement_count == 1
        assert layer.display_status.rendered_point_count == 1
        assert layer.display_status.message == "Selected values are not represented at the sampled LOD"
        assert session.render_results[-1].applied
    finally:
        runtime.close()
        visual.close()


def test_activation_owns_state_and_defers_status_until_snapshot_handlers_finish(maximum_texture_size: None) -> None:
    info = _dataset_info()
    layer = _layer(info)
    session = _ControllableSession(info)
    runtime = _runtime(layer, session)
    visual = VispyTiledPointsLayer(layer, FontInfo())
    observed_statuses = []

    def after_renderer(event: Event) -> None:
        # The visual has already replied, but the outer snapshot event has not
        # returned. State must remain owned and candidate status uncommitted.
        observed_statuses.append(layer.display_status)
        with pytest.raises(RuntimeError, match="another activation was pending"):
            runtime._activate_snapshot(event.value)

    layer.events.render_snapshot.connect(after_renderer)
    layer.events.render_snapshot.ignore_callback_errors = False
    try:
        for x_min in (0.0, 1.0):
            layer.events.viewport(value=_viewport(x_min))
            previous_status = layer.display_status
            request = session.viewport_requests[-1]
            session.complete_viewport(_snapshot(layer, request, (_tile(layer, request),)))

            assert observed_statuses[-1] == previous_status
            assert session.render_results[-1].applied
            assert layer.display_status.message == "Ready"
            assert layer.display_status.rendered_point_count == 1
        assert len(observed_statuses) == 2
    finally:
        runtime.close()
        visual.close()


@pytest.mark.parametrize("applied", [True, False], ids=["accepted", "rejected"])
def test_duplicate_renderer_acknowledgement_cannot_overwrite_first_result(applied: bool) -> None:
    info = _dataset_info()
    layer = _layer(info)
    session = _ControllableSession(info)
    runtime = _runtime(layer, session)

    def acknowledge_twice(event: Event) -> None:
        snapshot = event.value
        result = TiledPointsRenderResult(snapshot.request_generation, snapshot.selection_generation, applied)
        layer.events.render_snapshot_result(value=result)
        # The state remains installed until _activate_snapshot() finishes, so
        # explicitly reject a second reply instead of replacing the first one.
        with pytest.raises(RuntimeError, match="same snapshot more than once"):
            layer.events.render_snapshot_result(value=replace(result, applied=not applied))

    layer.events.render_snapshot.connect(acknowledge_twice)
    layer.events.render_snapshot.ignore_callback_errors = False
    layer.events.render_snapshot_result.ignore_callback_errors = False
    try:
        layer.events.viewport(value=_viewport())
        request = session.viewport_requests[-1]
        session.complete_viewport(_snapshot(layer, request, (_tile(layer, request),)))
        assert session.render_results == [
            TiledPointsRenderResult(request.request_generation, request.selection_generation, applied)
        ]
        assert layer.display_status.rendered_point_count == (1 if applied else 0)
    finally:
        runtime.close()


@pytest.mark.parametrize("close_source", ["runtime", "session"])
def test_close_during_snapshot_event_rejects_recorded_result_and_cleans_state(close_source: str) -> None:
    info = _dataset_info()
    layer = _layer(info)
    session = _ControllableSession(info)
    runtime = _runtime(layer, session)

    def acknowledge_then_close(event: Event) -> None:
        snapshot = event.value
        layer.events.render_snapshot_result(
            value=TiledPointsRenderResult(snapshot.request_generation, snapshot.selection_generation, applied=True)
        )
        layer.events.viewport(value=_viewport(1.0))
        if close_source == "runtime":
            runtime.close()
        else:
            session.close()

    layer.events.render_snapshot.connect(acknowledge_then_close)
    try:
        layer.events.viewport(value=_viewport())
        request = session.viewport_requests[-1]
        session.complete_viewport(_snapshot(layer, request, (_tile(layer, request),)))

        assert runtime.closed
        assert runtime._render_activation_state is None
        assert not session.render_results[-1].applied
        assert layer.display_status.rendered_point_count == 0
        assert len(session.viewport_requests) == 1
        assert session.close_count == 1
    finally:
        runtime.close()


def test_close_from_status_listener_rejects_activation_and_cleans_state(maximum_texture_size: None) -> None:
    info = _dataset_info()
    layer = _layer(info)
    session = _ControllableSession(info)
    runtime = _runtime(layer, session)
    visual = VispyTiledPointsLayer(layer, FontInfo())

    def close_on_ready(event: Event) -> None:
        if event.value.message == "Ready":
            runtime.close()

    layer.events.display_status.connect(close_on_ready)
    try:
        layer.events.viewport(value=_viewport())
        request = session.viewport_requests[-1]
        session.complete_viewport(_snapshot(layer, request, (_tile(layer, request),)))

        assert runtime.closed
        assert runtime._render_activation_state is None
        assert not session.render_results[-1].applied
        assert session.close_count == 1
    finally:
        runtime.close()
        visual.close()


def test_status_processing_exception_cleans_activation_and_allows_next_request(maximum_texture_size: None) -> None:
    info = _dataset_info()
    layer = _layer(info)
    session = _ControllableSession(info)
    runtime = _runtime(layer, session)
    visual = VispyTiledPointsLayer(layer, FontInfo())

    def fail_on_ready(event: Event) -> None:
        if event.value.message == "Ready":
            raise RuntimeError("synthetic status failure")

    layer.events.display_status.connect(fail_on_ready)
    layer.events.display_status.ignore_callback_errors = False
    try:
        layer.events.viewport(value=_viewport())
        request = session.viewport_requests[-1]
        # Catch the exception directly, outside Qt's unhandled-slot hook.
        with pytest.raises(RuntimeError, match="synthetic status failure"):
            runtime._viewport_scheduler._on_viewport_ready(_snapshot(layer, request, (_tile(layer, request),)))
        assert runtime._render_activation_state is None
        assert not session.render_results[-1].applied

        layer.events.display_status.disconnect(fail_on_ready)
        layer.events.viewport(value=_viewport(1.0))
        request = session.viewport_requests[-1]
        session.complete_viewport(_snapshot(layer, request, (_tile(layer, request),)))
        assert session.render_results[-1].applied
        assert layer.display_status.message == "Ready"
    finally:
        runtime.close()
        visual.close()


def test_missing_renderer_acknowledgement_rejects_candidate_and_allows_next_activation(
    maximum_texture_size: None,
) -> None:
    info = _dataset_info()
    layer = _layer(info)
    session = _ControllableSession(info)
    runtime = _runtime(layer, session)
    visual = None
    try:
        # No visual has connected to render_snapshot yet. Emitting the model
        # event alone must not be treated as successful renderer acceptance.
        layer.events.viewport(value=_viewport())
        request = session.viewport_requests[-1]
        session.complete_viewport(_snapshot(layer, request, (_tile(layer, request),)))
        assert session.render_results[-1] == TiledPointsRenderResult(
            request.request_generation, request.selection_generation, applied=False
        )
        assert "Renderer did not acknowledge" in layer.display_status.message
        assert layer.display_status.rendered_point_count == 0

        visual = VispyTiledPointsLayer(layer, FontInfo())
        layer.events.viewport(value=_viewport(1.0))
        request = session.viewport_requests[-1]
        session.complete_viewport(_snapshot(layer, request, (_tile(layer, request),)))
        assert session.render_results[-1].applied
        assert visual.active_point_count == 1
        assert layer.display_status.message == "Ready"
    finally:
        runtime.close()
        if visual is not None:
            visual.close()


def test_render_event_exception_clears_pending_activation_and_reports_rejection(maximum_texture_size: None) -> None:
    info = _dataset_info()
    layer = _layer(info)
    session = _ControllableSession(info)
    runtime = _runtime(layer, session)
    visual = None

    def fail_activation(event) -> None:
        raise RuntimeError("synthetic render event failure")

    layer.events.render_snapshot.connect(fail_activation)
    layer.events.render_snapshot.ignore_callback_errors = False
    try:
        layer.events.viewport(value=_viewport())
        request = session.viewport_requests[-1]
        snapshot = _snapshot(layer, request, (_tile(layer, request),))
        # Exercise the completion handler directly to catch the propagated
        # event exception without Qt's unhandled-slot exception hook.
        with pytest.raises(RuntimeError, match="synthetic render event failure"):
            runtime._viewport_scheduler._on_viewport_ready(snapshot)
        assert session.render_results[-1] == TiledPointsRenderResult(
            request.request_generation, request.selection_generation, applied=False
        )
        assert layer.display_status.rendered_point_count == 0

        layer.events.render_snapshot.disconnect(fail_activation)
        visual = VispyTiledPointsLayer(layer, FontInfo())
        layer.events.viewport(value=_viewport(1.0))
        request = session.viewport_requests[-1]
        session.complete_viewport(_snapshot(layer, request, (_tile(layer, request),)))
        assert session.render_results[-1].applied
        assert visual.active_point_count == 1
    finally:
        runtime.close()
        if visual is not None:
            visual.close()


def test_runtime_reports_renderer_failure_without_committing_candidate_status(maximum_texture_size: None) -> None:
    info = _dataset_info()
    layer = _layer(info, max_vertex_payload_bytes=12)
    session = _ControllableSession(info)
    runtime = _runtime(layer, session)
    visual = VispyTiledPointsLayer(layer, FontInfo())
    try:
        layer.events.viewport(value=_viewport())
        first_request = session.viewport_requests[-1]
        first_tile = _tile(layer, first_request)
        session.complete_viewport(_snapshot(layer, first_request, (first_tile,)))

        layer.events.viewport(value=_viewport(10.0))
        second_request = session.viewport_requests[-1]
        second_tiles = (
            _tile(layer, second_request),
            _tile(layer, second_request, tile_x=1),
        )
        session.complete_viewport(_snapshot(layer, second_request, second_tiles))

        assert visual.active_point_count == first_tile.point_count
        assert visual.payload_replacement_count == 1
        assert layer.display_status.rendered_point_count == 1
        assert layer.display_status.rendered_tile_count == 1
        assert "max_vertex_payload_bytes=12" in layer.display_status.message
        assert session.render_results[-1] == TiledPointsRenderResult(
            second_request.request_generation, second_request.selection_generation, applied=False
        )
    finally:
        runtime.close()
        visual.close()


def test_runtime_rejects_mismatched_cache_before_accepting_viewports() -> None:
    layer = _layer(_dataset_info())
    session = _ControllableSession(_dataset_info(points_name="other-points"))
    errors: list[object] = []
    layer.events.render_error.connect(lambda event: errors.append(event.value))

    runtime = _runtime(layer, session)

    assert runtime.closed
    assert session.close_count == 1
    assert len(errors) == 1
    assert "points_name" in str(errors[0])
    assert "does not match" in layer.display_status.message


def test_runtime_close_disconnects_layer_and_rejects_late_worker_results() -> None:
    info = _dataset_info()
    layer = _layer(info)
    session = _ControllableSession(info)
    runtime = _runtime(layer, session)
    snapshots: list[object] = []
    layer.events.render_snapshot.connect(lambda event: snapshots.append(event.value))
    layer.events.viewport(value=_viewport())
    request = session.viewport_requests[-1]

    assert runtime.close()
    assert not runtime.close()
    layer.events.viewport(value=_viewport(10.0))
    session.complete_viewport(_snapshot(layer, request, ()))

    assert len(session.viewport_requests) == 1
    assert snapshots == []
    assert session.close_count == 1


def test_runtime_remains_open_when_layer_rejects_data_replacement() -> None:
    info = _dataset_info()
    layer = _layer(info)
    session = _ControllableSession(info)
    runtime = _runtime(layer, session)
    try:
        with pytest.raises(ValueError, match="cannot be replaced; construct a new layer and cache runtime"):
            layer.data = _reference(_dataset_info(cache_generation_id="87654321-4321-6789-a234-678943216789"))

        assert not runtime.closed
        assert session.close_count == 0
        assert layer.data.cache_generation_id == info.cache_generation_id
    finally:
        runtime.close()


def test_real_cache_flows_from_layer_viewport_to_renderer_and_selected_values(
    real_cache_root: Path,
    maximum_texture_size: None,
    qtbot,
) -> None:
    with _PointsCacheReader(real_cache_root) as reader:
        info = reader.dataset_info
    layer = _layer(info)
    runtime = _TiledPointsLayerRuntime(layer, real_cache_root, _settings())
    visual = VispyTiledPointsLayer(layer, FontInfo())
    observed: list[TiledPointsRenderSnapshot] = []
    callback_thread_ids: list[int] = []
    layer.events.render_snapshot.connect(lambda event: observed.append(event.value))
    layer.events.render_snapshot.connect(lambda _event: callback_thread_ids.append(threading.get_ident()))
    try:
        qtbot.waitUntil(lambda: runtime.state is _CacheSessionState.READY, timeout=5_000)
        layer._emit_viewport(_viewport(width=10.0))
        qtbot.waitUntil(lambda: len(observed) == 1, timeout=5_000)

        assert observed[-1].rendered_point_count == 3
        assert observed[-1].rendered_tile_count == 1
        assert visual.active_point_count == observed[-1].rendered_point_count
        assert visual.payload_replacement_count == 1
        assert visual.visual_count == 1
        assert visual.vbo_count == 1

        # Expanding by one logical tile reuses the existing CPU tile, while the
        # renderer deliberately replaces its one complete VBO payload.
        layer._emit_viewport(_viewport(width=20.0))
        qtbot.waitUntil(lambda: len(observed) == 2, timeout=5_000)
        assert observed[-1].rendered_point_count == 4
        assert observed[-1].rendered_tile_count == 2
        assert visual.payload_replacement_count == 2

        # The model suppresses an unchanged normalized viewport before it can
        # reach the scheduler, worker, or renderer.
        layer._emit_viewport(_viewport(width=20.0))
        qtbot.wait(50)
        assert len(observed) == 2
        assert visual.payload_replacement_count == 2

        assert runtime.set_selected_value_ids((0,))
        qtbot.waitUntil(lambda: len(observed) == 3, timeout=5_000)
        assert observed[-1].requested_value_ids == (0,)
        assert observed[-1].rendered_point_count == 2
        assert bool((observed[-1].render_batch.vertices["a_value_id"] == 0).all())
        assert visual.payload_replacement_count == 3
        assert callback_thread_ids and set(callback_thread_ids) == {threading.get_ident()}
    finally:
        runtime.close()
        qtbot.waitUntil(lambda: runtime.state is _CacheSessionState.CLOSED, timeout=5_000)
        visual.close()


@pytest.mark.parametrize("selection", [None, (0,)], ids=["all-values", "subset"])
def test_real_contained_viewports_retain_worker_batch_and_vbo_with_fresh_status(
    real_cache_root: Path, maximum_texture_size: None, qtbot, monkeypatch, selection
) -> None:
    """Exercise real IO, queued Qt identity, renderer feedback, and empty inner views."""
    with _PointsCacheReader(real_cache_root) as reader:
        info = reader.dataset_info
    layer = _layer(info)
    runtime = _TiledPointsLayerRuntime(layer, real_cache_root, _settings(), initial_requested_value_ids=selection)
    visual = VispyTiledPointsLayer(layer, FontInfo())
    observed = []
    layer.events.render_snapshot.connect(lambda event: observed.append(event.value))
    outer = replace(_viewport(width=40.0), x_min=-10.0, y_min=-10.0, y_max=20.0)
    try:
        qtbot.waitUntil(lambda: runtime.state is _CacheSessionState.READY, timeout=5_000)
        layer._emit_viewport(outer)
        qtbot.waitUntil(lambda: len(observed) == 1, timeout=5_000)
        first = observed[0]
        assert first.rendered_point_count == (4 if selection is None else 2)
        assert visual.payload_replacement_count == 1

        def forbidden(*args, **kwargs):
            raise AssertionError("An active-batch hit entered the tile preparation/upload pipeline")

        with monkeypatch.context() as guarded:
            guarded.setattr(_PointsCacheReader, "plan_viewport", forbidden)
            guarded.setattr(_PointsCacheReader, "read_planned_tiles", forbidden)
            guarded.setattr(_CpuTileResidency, "get", forbidden)
            guarded.setattr(cache_session_module, "pack_render_tiles", forbidden)
            guarded.setattr(visual._snapshot_visual, "replace_vertices", forbidden)
            for generation, viewport in enumerate((_viewport(width=10.0), _viewport(20.0, width=10.0), outer), start=2):
                layer._emit_viewport(viewport)
                qtbot.waitUntil(lambda generation=generation: len(observed) == generation, timeout=5_000)
                current = observed[-1]
                assert current.render_batch is first.render_batch
                assert current.request_generation == generation
                assert current.rendered_tile_count == first.rendered_tile_count
                assert visual.active_point_count == first.rendered_point_count
                if generation == 3:
                    assert current.estimated_point_count == 0
                    assert layer.display_status.message == "No points in view"
                else:
                    assert layer.display_status.message == "Ready"

        # A disjoint view replaces the retained entry. Returning cannot restore
        # historical packed data, although decoded tile residency can avoid IO.
        layer._emit_viewport(_viewport(50.0))
        qtbot.waitUntil(lambda: len(observed) == 5, timeout=5_000)
        assert observed[-1].rendered_point_count == 0
        layer._emit_viewport(outer)
        qtbot.waitUntil(lambda: len(observed) == 6, timeout=5_000)
        assert observed[-1].render_batch is not first.render_batch
        np.testing.assert_array_equal(observed[-1].render_batch.vertices, first.render_batch.vertices)
        assert visual.payload_replacement_count == 2
        assert visual.visual_count == visual.vbo_count == 1
    finally:
        runtime.close()
        qtbot.waitUntil(lambda: runtime.state is _CacheSessionState.CLOSED, timeout=5_000)
        visual.close()
