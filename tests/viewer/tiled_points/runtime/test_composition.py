from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from napari._vispy.utils.qt_font import FontInfo
from qtpy.QtCore import QObject, Signal

from napari_harpy.core.multi_scale_cache_points_zarr.builder import (
    _build_points_cache_zarr,
    _PointsCacheBuilderConfig,
)
from napari_harpy.core.multi_scale_cache_points_zarr.cache_format import _CatalogWriteSettings
from napari_harpy.core.multi_scale_cache_points_zarr.reader import (
    _CacheDatasetInfo,
    _CacheLevelInfo,
    _PointsCacheReader,
)
from napari_harpy.core.multi_scale_cache_points_zarr.source import (
    ParquetPointsSource,
    PointColumnSelection,
    validate_parquet_points_source,
)
from napari_harpy.core.multi_scale_cache_points_zarr.storage.models import _ZarrWriteSettings
from napari_harpy.viewer.tiled_points.contracts import (
    TiledPointsDatasetReference,
    TiledPointsRenderSnapshot,
    TiledPointsRenderTile,
    TiledPointsViewportState,
    TileResidencyKey,
    _ViewportRequest,
)
from napari_harpy.viewer.tiled_points.napari.layer import TiledPointsLayerModel
from napari_harpy.viewer.tiled_points.runtime.cache_session import (
    _CacheSessionFailure,
    _CacheSessionSettings,
    _CacheSessionState,
)
from napari_harpy.viewer.tiled_points.runtime.composition import _TiledPointsLayerRuntime
from napari_harpy.viewer.tiled_points.vispy.layer import VispyTiledPointsLayer

_GENERATION_ID = "12345678-1234-5678-9234-567812345678"


class _ControllableSession(QObject):
    state_changed = Signal(object)
    dataset_available = Signal(object)
    bucket_index_progress = Signal(int, int)
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

    def start(self) -> None:
        self._set_state(_CacheSessionState.STARTING)
        self.dataset_available.emit(self.dataset_info)
        if self.state is _CacheSessionState.CLOSED:
            return
        self._set_state(_CacheSessionState.LOADING_BUCKET_INDEXES)
        self.bucket_index_progress.emit(1, 1)
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


def _layer(info: _CacheDatasetInfo, *, max_gpu_tile_bytes: int = 1_000_000) -> TiledPointsLayerModel:
    return TiledPointsLayerModel(
        _reference(info),
        value_palette=np.asarray(((255, 0, 0, 255), (0, 255, 0, 255)), dtype=np.uint8),
        max_gpu_tile_bytes=max_gpu_tile_bytes,
    )


def _settings() -> _CacheSessionSettings:
    return _CacheSessionSettings(
        max_bucket_lookup_bytes=None,
        max_selected_value_index_bytes=None,
        max_cpu_tile_bytes=1_000_000,
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
) -> TiledPointsRenderSnapshot:
    return TiledPointsRenderSnapshot(
        cache_generation_id=layer.data.cache_generation_id,
        request_generation=request.request_generation,
        selection_generation=request.selection_generation,
        requested_value_ids=request.requested_value_ids,
        level=level,
        level_kind="exact" if level == 0 else "bridge" if level == 1 else "spatial",
        within_budget=within_budget,
        estimated_point_count=(
            sum(tile.point_count for tile in tiles) if estimated_point_count is None else estimated_point_count
        ),
        omitted_value_ids=omitted_value_ids,
        tiles=tiles,
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


def _track_tile_resource_creation(
    visual: VispyTiledPointsLayer,
    monkeypatch: pytest.MonkeyPatch,
) -> list[TileResidencyKey]:
    """Record test-local tile uploads without production history bookkeeping."""
    created_keys: list[TileResidencyKey] = []
    create_tile_resource = visual._create_tile_resource

    def _record(tile: TiledPointsRenderTile):
        resource = create_tile_resource(tile)
        created_keys.append(tile.key)
        return resource

    monkeypatch.setattr(visual, "_create_tile_resource", _record)
    return created_keys


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

        assert visual.active_keys == (tile.key,)
        assert layer.display_status.level == 0
        assert layer.display_status.rendered_point_count == 1
        assert layer.display_status.rendered_tile_count == 1
        assert layer.display_status.message == "Ready"
    finally:
        runtime.close()
        visual.close()


def test_runtime_retains_active_visual_for_over_budget_and_failure_then_clears_sampled_omission(
    maximum_texture_size: None,
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
            )
        )
        assert visual.active_keys == (tile.key,)
        assert len(rendered_snapshots) == 1
        assert layer.display_status.rendered_point_count == 1
        assert "retaining the previous view" in layer.display_status.message

        layer.events.viewport(value=_viewport(20.0))
        failed_request = session.viewport_requests[-1]
        session.fail_viewport(failed_request.request_generation)
        assert visual.active_keys == (tile.key,)
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
        assert visual.active_keys == ()
        assert len(rendered_snapshots) == 2
        assert layer.display_status.level == 2
        assert layer.display_status.rendered_point_count == 0
        assert layer.display_status.omitted_value_ids == (0,)
        assert layer.display_status.message == "Selected values are not represented at the sampled LOD"
    finally:
        runtime.close()
        visual.close()


def test_runtime_reports_renderer_failure_without_committing_candidate_status(maximum_texture_size: None) -> None:
    info = _dataset_info()
    layer = _layer(info, max_gpu_tile_bytes=12)
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
        second_tile = _tile(layer, second_request, tile_x=1)
        session.complete_viewport(_snapshot(layer, second_request, (second_tile,)))

        assert visual.active_keys == (first_tile.key,)
        assert layer.display_status.rendered_point_count == 1
        assert layer.display_status.rendered_tile_count == 1
        assert "max_gpu_tile_bytes=12" in layer.display_status.message
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


def test_runtime_closes_when_layer_data_changes_to_another_generation() -> None:
    info = _dataset_info()
    layer = _layer(info)
    session = _ControllableSession(info)
    runtime = _runtime(layer, session)

    layer.data = _reference(_dataset_info(cache_generation_id="87654321-4321-6789-a234-678943216789"))

    assert runtime.closed
    assert session.close_count == 1
    assert "construct a new cache runtime" in layer.display_status.message


@pytest.fixture(scope="module")
def composed_real_cache_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("tiled-points-composition")
    source = ParquetPointsSource(
        spatialdata_path=root / "source.zarr",
        points_name="transcripts",
        columns=PointColumnSelection(x="x", y="y", value="gene"),
    )
    source.parquet_path.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "x": pa.array([1.0, 3.0, 2.0, 11.0], type=pa.float64()),
                "y": pa.array([1.0, 2.0, 3.0, 1.0], type=pa.float64()),
                "gene": pa.array(["A", "B", "A", "B"]),
            }
        ),
        source.parquet_path / "part.0.parquet",
        row_group_size=2,
    )
    validated = validate_parquet_points_source(source, max_batch_rows=2)
    temporary_root = root / "temporary"
    temporary_root.mkdir()
    return _build_points_cache_zarr(
        validated,
        output_path=root / "transcripts_vis_zarr",
        temporary_directory_root=temporary_root,
        config=_PointsCacheBuilderConfig(
            leaf_tile_size=10,
            overview_point_budget=10,
            dask_worker_count=2,
            zarr_settings=_ZarrWriteSettings(2, 4, 2, 4, "zstd-v1"),
            catalog_settings=_CatalogWriteSettings(
                manifest_chunk_rows=2,
                manifest_shard_rows=4,
                value_tile_chunk_rows=2,
                value_tile_shard_rows=4,
            ),
        ),
    )


def test_real_cache_flows_from_layer_viewport_to_renderer_and_selected_values(
    composed_real_cache_root: Path,
    maximum_texture_size: None,
    monkeypatch: pytest.MonkeyPatch,
    qtbot,
) -> None:
    with _PointsCacheReader(composed_real_cache_root) as reader:
        info = reader.dataset_info
    layer = _layer(info)
    runtime = _TiledPointsLayerRuntime(layer, composed_real_cache_root, _settings())
    visual = VispyTiledPointsLayer(layer, FontInfo())
    created_keys = _track_tile_resource_creation(visual, monkeypatch)
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
        assert visual.active_keys == tuple(tile.key for tile in observed[-1].tiles)
        assert created_keys == [observed[-1].tiles[0].key]

        # Expanding by one logical tile reuses the existing CPU/GPU tile and
        # reads/uploads only the entering tile.
        layer._emit_viewport(_viewport(width=20.0))
        qtbot.waitUntil(lambda: len(observed) == 2, timeout=5_000)
        assert observed[-1].rendered_point_count == 4
        assert observed[-1].rendered_tile_count == 2
        assert created_keys == [tile.key for tile in observed[-1].tiles]

        # The model suppresses an unchanged normalized viewport before it can
        # reach the coordinator, worker, or renderer.
        layer._emit_viewport(_viewport(width=20.0))
        qtbot.wait(50)
        assert len(observed) == 2
        assert created_keys == [tile.key for tile in observed[-1].tiles]

        assert runtime.set_selected_value_ids((0,))
        qtbot.waitUntil(lambda: len(observed) == 3, timeout=5_000)
        assert observed[-1].requested_value_ids == (0,)
        assert observed[-1].rendered_point_count == 2
        assert all(bool((tile.value_id == 0).all()) for tile in observed[-1].tiles)
        assert callback_thread_ids and set(callback_thread_ids) == {threading.get_ident()}
    finally:
        runtime.close()
        qtbot.waitUntil(lambda: runtime.state is _CacheSessionState.CLOSED, timeout=5_000)
        visual.close()
