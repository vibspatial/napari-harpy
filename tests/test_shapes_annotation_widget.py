from __future__ import annotations

import warnings
from collections.abc import Callable
from html import unescape
from types import SimpleNamespace

import anndata as ad
import geopandas as gpd
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba
from napari.layers import Shapes
from napari.layers.base._base_constants import ActionType
from napari.layers.shapes._shapes_constants import Mode
from napari_builtins.io import csv_to_layer_data, napari_write_shapes
from qtpy.QtWidgets import QComboBox, QLabel
from shapely.geometry import Polygon
from spatialdata import SpatialData, read_zarr
from spatialdata.models import ShapesModel, TableModel
from spatialdata.transformations import Identity

import napari_harpy._app_state as app_state_module
import napari_harpy.widgets.shapes_annotation.widget as shapes_annotation_widget_module
from napari_harpy._app_state import ShapesElementWrittenEvent, get_or_create_app_state
from napari_harpy.core.shapes_annotation import AnnotateShapesElementResult
from napari_harpy.core.shapes_geometry import (
    delete_napari_polygon_vertex,
    napari_polygon_vertices_to_shapely_polygon,
    napari_polygon_vertices_to_topology,
    shapely_polygon_to_napari_polygon_vertices,
)
from napari_harpy.viewer.adapter import ShapesLayerBinding
from napari_harpy.viewer.shapes_styling import (
    _SHAPES_EDGE_COLOR_SYNC_CALLBACK_ATTR,
    _SHAPES_EDGE_WIDTH_SYNC_CALLBACK_ATTR,
    apply_primary_shapes_layer_style,
)
from napari_harpy.widgets import ShapesAnnotation as LazyShapesAnnotation
from napari_harpy.widgets.shapes_annotation.widget import ShapesAnnotation


def _rgba(color: str) -> np.ndarray:
    return np.asarray(to_rgba(color), dtype=float)


class DummyEventEmitter:
    def __init__(self) -> None:
        self._callbacks: list[Callable[[object], None]] = []

    def connect(self, callback: Callable[[object], None]) -> None:
        self._callbacks.append(callback)

    def emit(self, value: object | None = None) -> None:
        event = SimpleNamespace(value=value)
        for callback in list(self._callbacks):
            callback(event)


class DummyLayers(list):
    def __init__(self) -> None:
        super().__init__()
        self.selection = SimpleNamespace(active=None, select_only=self._select_only)
        self.events = SimpleNamespace(
            inserted=DummyEventEmitter(),
            removed=DummyEventEmitter(),
            reordered=DummyEventEmitter(),
        )

    def _select_only(self, layer: object) -> None:
        self.selection.active = layer

    def remove(self, layer: object) -> None:
        super().remove(layer)
        self.events.removed.emit(layer)


class DummyViewer:
    def __init__(self) -> None:
        self.layers = DummyLayers()

    def add_layer(self, layer: object) -> object:
        self.layers.append(layer)
        self.layers.events.inserted.emit(layer)
        return layer


def _combo_texts(combo: QComboBox) -> list[str]:
    return [combo.itemText(index) for index in range(combo.count())]


def _combo_data(combo: QComboBox) -> list[object]:
    return [combo.itemData(index) for index in range(combo.count())]


def _combo_index_for_text(combo: QComboBox, text: str) -> int:
    for index in range(combo.count()):
        if combo.itemText(index) == text:
            return index
    return -1


def _status_text(widget: ShapesAnnotation) -> str:
    return unescape(widget.status_label.text())


def _tooltip_text(widget: ShapesAnnotation) -> str:
    return unescape(widget.status_label.toolTip()).replace("&#8203;", "").replace("\u200b", "")


def _assert_layer_data_unchanged(layer: Shapes, expected_data: list[np.ndarray]) -> None:
    assert len(layer.data) == len(expected_data)
    for actual_vertices, expected_vertices in zip(layer.data, expected_data, strict=True):
        np.testing.assert_allclose(np.asarray(actual_vertices, dtype=float), expected_vertices)


def _patch_coordinate_system_names(monkeypatch, coordinate_systems: list[str]) -> None:
    monkeypatch.setattr(
        shapes_annotation_widget_module,
        "get_coordinate_system_names_from_sdata",
        lambda sdata: coordinate_systems,
    )
    monkeypatch.setattr(
        app_state_module,
        "get_coordinate_system_names_from_sdata",
        lambda sdata: coordinate_systems,
    )


def _create_ready_annotation_widget(qtbot, viewer: DummyViewer, sdata: SpatialData) -> ShapesAnnotation:
    app_state = get_or_create_app_state(viewer)
    app_state.set_sdata(sdata)
    widget = ShapesAnnotation(viewer)
    qtbot.addWidget(widget)
    widget.name_edit.setText("new_regions")
    return widget


def _add_polygon(layer: Shapes, offset: float = 0.0) -> None:
    layer.add_polygons(
        np.asarray(
            [
                [offset + 0.0, 0.0],
                [offset + 0.0, 2.0],
                [offset + 2.0, 2.0],
                [offset + 2.0, 0.0],
            ],
            dtype=float,
        )
    )


def _native_polygon_layer(name: str, *, affine: np.ndarray | None = None) -> Shapes:
    return Shapes(
        [
            np.asarray(
                [
                    [0.0, 0.0],
                    [0.0, 2.0],
                    [2.0, 2.0],
                    [2.0, 0.0],
                ],
                dtype=float,
            )
        ],
        shape_type="polygon",
        affine=affine,
        name=name,
    )


def _yx_to_xy(coordinates_yx: np.ndarray) -> list[tuple[float, float]]:
    return [(float(x), float(y)) for y, x in np.asarray(coordinates_yx, dtype=float)]


def _polygon_hole_roundtrip_fixture() -> tuple[Polygon, Polygon]:
    center_y = 1000.0
    center_x = 2000.0

    polygon_1_shell_yx = np.array(
        [
            [center_y - 350, center_x - 280],
            [center_y - 420, center_x + 120],
            [center_y - 120, center_x + 320],
            [center_y + 180, center_x + 140],
            [center_y + 120, center_x - 240],
        ],
        dtype=float,
    )
    polygon_1_hole_yx = np.array(
        [
            [center_y - 150, center_x - 40],
            [center_y - 170, center_x + 70],
            [center_y - 80, center_x + 130],
            [center_y - 10, center_x + 40],
            [center_y - 50, center_x - 70],
        ],
        dtype=float,
    )
    polygon_2_yx = np.array(
        [
            [center_y + 260, center_x - 40],
            [center_y + 180, center_x + 260],
            [center_y + 420, center_x + 340],
            [center_y + 520, center_x + 40],
            [center_y + 360, center_x - 180],
        ],
        dtype=float,
    )

    polygon_1 = Polygon(
        _yx_to_xy(polygon_1_shell_yx),
        holes=[_yx_to_xy(polygon_1_hole_yx)],
    )
    polygon_2 = Polygon(_yx_to_xy(polygon_2_yx))
    assert polygon_1.is_valid
    assert polygon_2.is_valid
    assert len(polygon_1.interiors) == 1
    # When polygon_1 is encoded for napari, the flat path is:
    # shell[0:5] + shell[0] + hole[0:5] + hole[0] + shell[0].
    # The shell anchor/separator copies are shell[0]; the hole anchor
    # copies are hole[0].
    return polygon_1, polygon_2


def _direct_drag_callback_moving_vertex(
    *,
    moved_vertex_index: int,
    moved_coordinate: np.ndarray,
) -> Callable[[Shapes, object], object]:
    def direct_drag_callback(layer: Shapes, event: object) -> object:
        layer._moving_value = (0, moved_vertex_index)
        yield "press"

        while getattr(event, "type", None) == "mouse_move":
            vertices = np.asarray(layer.data[0], dtype=float).copy()
            vertices[moved_vertex_index] = moved_coordinate
            layer._data_view.edit(0, vertices)
            layer.refresh()
            yield "move"

    return direct_drag_callback


def _install_direct_drag_callback_for_annotation_guard(
    widget: ShapesAnnotation,
    layer: Shapes,
    *,
    moved_vertex_index: int,
    moved_coordinate: np.ndarray,
) -> None:
    widget._annotation_edit_guard.disconnect()
    layer._drag_modes = dict(layer._drag_modes)
    layer._drag_modes[Mode.DIRECT] = _direct_drag_callback_moving_vertex(
        moved_vertex_index=moved_vertex_index,
        moved_coordinate=moved_coordinate,
    )
    widget._annotation_edit_guard.attach(layer)
    assert layer._drag_modes[Mode.DIRECT] is widget._annotation_edit_guard._wrapped_direct_callback


def _drag_annotation_vertex(
    layer: Shapes,
    *,
    vertex_index: int,
    moved_coordinate: np.ndarray,
) -> np.ndarray:
    event = SimpleNamespace(type="mouse_press")
    drag = layer._drag_modes[Mode.DIRECT](layer, event)
    assert next(drag) == "press"
    event.type = "mouse_move"
    assert next(drag) == "move"
    event.type = "mouse_release"
    try:
        next(drag)
    except StopIteration:
        pass
    edited_vertices = np.asarray(layer.data[0], dtype=float)
    np.testing.assert_allclose(edited_vertices[vertex_index], moved_coordinate)
    return edited_vertices


def _make_polygon_hole_roundtrip_sdata(*, shapes_name: str = "hole_regions") -> SpatialData:
    polygon_1, polygon_2 = _polygon_hole_roundtrip_fixture()
    geodataframe = gpd.GeoDataFrame(
        {"label": ["polygon_with_hole", "simple_polygon"], "score": [1.25, 2.5]},
        geometry=[polygon_1, polygon_2],
        index=pd.Index(["hole_row", "simple_row"], name="region_id"),
    )
    shapes = ShapesModel.parse(geodataframe, transformations={"global": Identity()})
    return SpatialData(shapes={shapes_name: shapes})


def _make_create_holes_sdata(
    *,
    shapes_name: str = "create_holes_regions",
) -> tuple[SpatialData, Polygon, Polygon, Polygon]:
    shell = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    child = Polygon([(2, 2), (2, 4), (4, 4), (4, 2)])
    unselected = Polygon([(20, 20), (20, 24), (24, 24), (24, 20)])
    geodataframe = gpd.GeoDataFrame(
        {"label": ["shell", "child", "unselected"], "score": [1.0, 2.0, 3.0]},
        geometry=[shell, child, unselected],
        index=pd.Index(["shell_row", "child_row", "unselected_row"], name="region_id"),
    )
    shapes = ShapesModel.parse(geodataframe, transformations={"global": Identity()})
    return SpatialData(shapes={shapes_name: shapes}), shell, child, unselected


def _assert_polygon_hole_geometries_preserved(saved: gpd.GeoDataFrame, polygon_1: Polygon, polygon_2: Polygon) -> None:
    saved_polygon_1 = saved.geometry.iloc[0]
    saved_polygon_2 = saved.geometry.iloc[1]

    assert saved_polygon_1.equals(polygon_1)
    assert len(saved_polygon_1.interiors) == 1
    assert saved_polygon_1.area == polygon_1.area
    assert saved_polygon_1.bounds == polygon_1.bounds
    assert saved_polygon_2.equals(polygon_2)
    assert len(saved_polygon_2.interiors) == 0


def _add_dummy_table_annotating_shapes(sdata: SpatialData, *, shapes_name: str, table_name: str) -> ad.AnnData:
    shapes = sdata.shapes[shapes_name]
    index_values = shapes.index.to_list()
    cell_types = np.resize(np.asarray(["T", "B"], dtype=object), len(index_values))
    table = TableModel.parse(
        ad.AnnData(
            obs=pd.DataFrame(
                {
                    "region": [shapes_name] * len(index_values),
                    "index": index_values,
                    "cell_type": pd.Categorical(cell_types, categories=["T", "B"]),
                },
                index=[f"obs_{index}" for index in index_values],
            )
        ),
        region=shapes_name,
        region_key="region",
        instance_key="index",
    )
    sdata.tables[table_name] = table
    return table


def test_shapes_annotation_widget_can_be_instantiated(qtbot) -> None:
    widget = ShapesAnnotation()

    qtbot.addWidget(widget)

    assert widget.app_state.sdata is None
    assert widget.selected_spatialdata is None
    assert widget.selected_coordinate_system is None
    assert widget.selected_shapes_name is None
    assert widget._logo_path.is_file()
    header_logo = widget.findChild(QLabel, "shapes_annotation_header_logo")
    assert header_logo is not None
    pixmap = header_logo.pixmap()
    assert (pixmap is not None and not pixmap.isNull()) or header_logo.text() == "napari-harpy"
    assert widget.coordinate_system_combo.minimumWidth() == widget.name_edit.minimumWidth()
    assert widget.coordinate_system_combo.count() == 0
    assert widget.coordinate_system_combo.isEnabled() is False
    assert widget.shapes_combo.count() == 0
    assert widget.shapes_combo.isEnabled() is False
    assert widget.create_layer_button.isEnabled() is False
    assert widget.create_holes_button.isEnabled() is False
    assert widget.save_shapes_button.isEnabled() is False
    assert "No SpatialData Loaded" in _status_text(widget)


def test_shapes_annotation_widget_lazy_export() -> None:
    assert LazyShapesAnnotation is ShapesAnnotation


def test_annotation_layer_edit_guard_delegates_direct_mode_and_restores_instance_mapping() -> None:
    layer = Shapes([], ndim=2)
    layer._drag_modes = dict(layer._drag_modes)
    original_drag_modes = layer._drag_modes
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def original_direct_callback(*args: object, **kwargs: object) -> str:
        calls.append((args, kwargs))
        return "delegated"

    original_drag_modes[Mode.DIRECT] = original_direct_callback
    original_vertex_remove_callback = original_drag_modes[Mode.VERTEX_REMOVE]
    guard = shapes_annotation_widget_module._AnnotationLayerEditGuard()

    guard.attach(layer)
    wrapped_direct_callback = layer._drag_modes[Mode.DIRECT]
    wrapped_vertex_remove_callback = layer._drag_modes[Mode.VERTEX_REMOVE]

    assert guard.layer is layer
    assert layer._drag_modes is not original_drag_modes
    assert wrapped_direct_callback is not original_direct_callback
    assert wrapped_vertex_remove_callback is not original_vertex_remove_callback
    assert wrapped_direct_callback("event", value=3) == "delegated"
    assert calls == [(("event",), {"value": 3})]

    guard.disconnect()

    assert guard.layer is None
    assert layer._drag_modes is original_drag_modes
    assert layer._drag_modes[Mode.DIRECT] is original_direct_callback
    assert layer._drag_modes[Mode.VERTEX_REMOVE] is original_vertex_remove_callback


def test_annotation_layer_edit_guard_attach_is_idempotent_and_restores_class_mapping() -> None:
    layer = Shapes([], ndim=2)
    original_direct_callback = layer._drag_modes[Mode.DIRECT]
    original_vertex_remove_callback = layer._drag_modes[Mode.VERTEX_REMOVE]
    guard = shapes_annotation_widget_module._AnnotationLayerEditGuard()

    guard.attach(layer)
    first_wrapped_direct_callback = layer._drag_modes[Mode.DIRECT]
    first_wrapped_vertex_remove_callback = layer._drag_modes[Mode.VERTEX_REMOVE]
    guard.attach(layer)

    assert layer._drag_modes[Mode.DIRECT] is first_wrapped_direct_callback
    assert layer._drag_modes[Mode.VERTEX_REMOVE] is first_wrapped_vertex_remove_callback
    assert "_drag_modes" in vars(layer)

    guard.disconnect()

    assert guard.layer is None
    assert "_drag_modes" not in vars(layer)
    assert layer._drag_modes[Mode.DIRECT] is original_direct_callback
    assert layer._drag_modes[Mode.VERTEX_REMOVE] is original_vertex_remove_callback


def test_annotation_layer_edit_guard_replacing_layer_disconnects_previous_layer() -> None:
    first_layer = Shapes([], ndim=2)
    second_layer = Shapes([], ndim=2)
    first_direct_callback = first_layer._drag_modes[Mode.DIRECT]
    first_vertex_remove_callback = first_layer._drag_modes[Mode.VERTEX_REMOVE]
    guard = shapes_annotation_widget_module._AnnotationLayerEditGuard()

    guard.attach(first_layer)
    first_wrapped_direct_callback = first_layer._drag_modes[Mode.DIRECT]
    first_wrapped_vertex_remove_callback = first_layer._drag_modes[Mode.VERTEX_REMOVE]
    # `attach(...)` first calls `disconnect(...)`, so moving the guard to a new
    # layer must restore the previous layer before patching the new one.
    guard.attach(second_layer)

    assert guard.layer is second_layer
    assert first_layer._drag_modes[Mode.DIRECT] is first_direct_callback
    assert first_layer._drag_modes[Mode.VERTEX_REMOVE] is first_vertex_remove_callback
    assert "_drag_modes" not in vars(first_layer)
    assert "_drag_modes" in vars(second_layer)
    assert second_layer._drag_modes[Mode.DIRECT] is not first_wrapped_direct_callback
    assert second_layer._drag_modes[Mode.VERTEX_REMOVE] is not first_wrapped_vertex_remove_callback


def test_annotation_layer_edit_guard_direct_drag_syncs_shell_anchor_group() -> None:
    polygon, _ = _polygon_hole_roundtrip_fixture()
    moved_coordinate = np.asarray([1234.0, 2345.0])
    layer = Shapes([shapely_polygon_to_napari_polygon_vertices(polygon)], shape_type="polygon")
    layer._drag_modes = dict(layer._drag_modes)
    layer._drag_modes[Mode.DIRECT] = _direct_drag_callback_moving_vertex(
        moved_vertex_index=0,
        moved_coordinate=moved_coordinate,
    )
    guard = shapes_annotation_widget_module._AnnotationLayerEditGuard()
    guard.attach(layer)
    event = SimpleNamespace(type="mouse_press")

    drag = layer._drag_modes[Mode.DIRECT](layer, event)
    assert next(drag) == "press"
    event.type = "mouse_move"
    assert next(drag) == "move"

    edited_vertices = np.asarray(layer.data[0], dtype=float)
    np.testing.assert_allclose(edited_vertices[[0, 5, 12]], np.repeat(moved_coordinate[None, :], 3, axis=0))


def test_annotation_layer_edit_guard_direct_drag_syncs_shell_separator_group() -> None:
    polygon, _ = _polygon_hole_roundtrip_fixture()
    moved_coordinate = np.asarray([1234.0, 2345.0])
    layer = Shapes([shapely_polygon_to_napari_polygon_vertices(polygon)], shape_type="polygon")
    layer._drag_modes = dict(layer._drag_modes)
    layer._drag_modes[Mode.DIRECT] = _direct_drag_callback_moving_vertex(
        moved_vertex_index=12,
        moved_coordinate=moved_coordinate,
    )
    guard = shapes_annotation_widget_module._AnnotationLayerEditGuard()
    guard.attach(layer)
    event = SimpleNamespace(type="mouse_press")

    drag = layer._drag_modes[Mode.DIRECT](layer, event)
    assert next(drag) == "press"
    event.type = "mouse_move"
    assert next(drag) == "move"

    edited_vertices = np.asarray(layer.data[0], dtype=float)
    np.testing.assert_allclose(edited_vertices[[0, 5, 12]], np.repeat(moved_coordinate[None, :], 3, axis=0))


def test_annotation_layer_edit_guard_direct_drag_syncs_hole_anchor_group() -> None:
    polygon, _ = _polygon_hole_roundtrip_fixture()
    moved_coordinate = np.asarray([1234.0, 2345.0])
    layer = Shapes([shapely_polygon_to_napari_polygon_vertices(polygon)], shape_type="polygon")
    layer._drag_modes = dict(layer._drag_modes)
    layer._drag_modes[Mode.DIRECT] = _direct_drag_callback_moving_vertex(
        moved_vertex_index=6,
        moved_coordinate=moved_coordinate,
    )
    guard = shapes_annotation_widget_module._AnnotationLayerEditGuard()
    guard.attach(layer)
    event = SimpleNamespace(type="mouse_press")

    drag = layer._drag_modes[Mode.DIRECT](layer, event)
    assert next(drag) == "press"
    event.type = "mouse_move"
    assert next(drag) == "move"

    edited_vertices = np.asarray(layer.data[0], dtype=float)
    np.testing.assert_allclose(edited_vertices[[6, 11]], np.repeat(moved_coordinate[None, :], 2, axis=0))


def test_annotation_layer_edit_guard_direct_drag_leaves_non_anchor_vertex_local() -> None:
    polygon, _ = _polygon_hole_roundtrip_fixture()
    original_vertices = shapely_polygon_to_napari_polygon_vertices(polygon)
    moved_coordinate = np.asarray([1234.0, 2345.0])
    layer = Shapes([original_vertices], shape_type="polygon")
    layer._drag_modes = dict(layer._drag_modes)
    layer._drag_modes[Mode.DIRECT] = _direct_drag_callback_moving_vertex(
        moved_vertex_index=8,
        moved_coordinate=moved_coordinate,
    )
    guard = shapes_annotation_widget_module._AnnotationLayerEditGuard()
    guard.attach(layer)
    event = SimpleNamespace(type="mouse_press")

    drag = layer._drag_modes[Mode.DIRECT](layer, event)
    assert next(drag) == "press"
    event.type = "mouse_move"
    assert next(drag) == "move"

    edited_vertices = np.asarray(layer.data[0], dtype=float)
    np.testing.assert_allclose(edited_vertices[8], moved_coordinate)
    np.testing.assert_allclose(edited_vertices[[0, 5, 12]], original_vertices[[0, 5, 12]])
    np.testing.assert_allclose(edited_vertices[[6, 11]], original_vertices[[6, 11]])


def test_annotation_layer_edit_guard_direct_drag_does_not_guess_malformed_topology() -> None:
    polygon, _ = _polygon_hole_roundtrip_fixture()
    original_vertices = shapely_polygon_to_napari_polygon_vertices(polygon)[:-1]
    moved_coordinate = np.asarray([1234.0, 2345.0])
    layer = Shapes([original_vertices], shape_type="polygon")
    layer._drag_modes = dict(layer._drag_modes)
    layer._drag_modes[Mode.DIRECT] = _direct_drag_callback_moving_vertex(
        moved_vertex_index=0,
        moved_coordinate=moved_coordinate,
    )
    guard = shapes_annotation_widget_module._AnnotationLayerEditGuard()
    guard.attach(layer)
    event = SimpleNamespace(type="mouse_press")

    drag = layer._drag_modes[Mode.DIRECT](layer, event)
    assert next(drag) == "press"
    event.type = "mouse_move"
    assert next(drag) == "move"

    edited_vertices = np.asarray(layer.data[0], dtype=float)
    np.testing.assert_allclose(edited_vertices[0], moved_coordinate)
    np.testing.assert_allclose(edited_vertices[5], original_vertices[5])


def test_annotation_layer_edit_guard_vertex_remove_delegates_when_no_vertex(monkeypatch) -> None:
    polygon, _ = _polygon_hole_roundtrip_fixture()
    layer = Shapes([shapely_polygon_to_napari_polygon_vertices(polygon)], shape_type="polygon")
    layer._drag_modes = dict(layer._drag_modes)
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def original_vertex_remove_callback(*args: object, **kwargs: object) -> str:
        calls.append((args, kwargs))
        return "delegated"

    monkeypatch.setattr(layer, "get_value", lambda position, world=True: (0, None))
    layer._drag_modes[Mode.VERTEX_REMOVE] = original_vertex_remove_callback
    guard = shapes_annotation_widget_module._AnnotationLayerEditGuard()
    guard.attach(layer)
    event = SimpleNamespace(position=(0.0, 0.0))

    result = layer._drag_modes[Mode.VERTEX_REMOVE](layer, event)

    assert result == "delegated"
    assert calls == [((layer, event), {})]


def test_annotation_layer_edit_guard_vertex_remove_delegates_simple_polygon(monkeypatch) -> None:
    _, polygon = _polygon_hole_roundtrip_fixture()
    layer = Shapes([shapely_polygon_to_napari_polygon_vertices(polygon)], shape_type="polygon")
    layer._drag_modes = dict(layer._drag_modes)
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def original_vertex_remove_callback(*args: object, **kwargs: object) -> str:
        calls.append((args, kwargs))
        return "delegated"

    monkeypatch.setattr(layer, "get_value", lambda position, world=True: (0, 1))
    layer._drag_modes[Mode.VERTEX_REMOVE] = original_vertex_remove_callback
    guard = shapes_annotation_widget_module._AnnotationLayerEditGuard()
    guard.attach(layer)
    event = SimpleNamespace(position=(0.0, 0.0))

    result = layer._drag_modes[Mode.VERTEX_REMOVE](layer, event)

    assert result == "delegated"
    assert calls == [((layer, event), {})]


def test_annotation_layer_edit_guard_vertex_remove_uses_helper_for_hole_bearing_polygon(monkeypatch) -> None:
    polygon, _ = _polygon_hole_roundtrip_fixture()
    deleted_vertex_indices = [2, 8, 0, 12, 6]

    for deleted_vertex_index in deleted_vertex_indices:
        original_vertices = shapely_polygon_to_napari_polygon_vertices(polygon)
        topology = napari_polygon_vertices_to_topology(original_vertices)
        expected_vertices, _ = delete_napari_polygon_vertex(original_vertices, topology, deleted_vertex_index)
        layer = Shapes([original_vertices], shape_type="polygon")
        layer._drag_modes = dict(layer._drag_modes)
        event = SimpleNamespace(position=(0.0, 0.0))
        events: list[tuple[ActionType, tuple[int, ...], tuple[tuple[int, ...], ...]]] = []

        def original_vertex_remove_callback(*args: object, **kwargs: object) -> None:
            raise AssertionError("hole-bearing polygon deletion should use the topology helper")

        def record_data_event(
            event: object,
            event_log: list[tuple[ActionType, tuple[int, ...], tuple[tuple[int, ...], ...]]] = events,
        ) -> None:
            event_log.append((event.action, event.data_indices, event.vertex_indices))

        monkeypatch.setattr(
            layer,
            "get_value",
            lambda position, world=True, index=deleted_vertex_index: (0, index),
        )
        layer._drag_modes[Mode.VERTEX_REMOVE] = original_vertex_remove_callback
        layer.events.data.connect(record_data_event)
        guard = shapes_annotation_widget_module._AnnotationLayerEditGuard()
        guard.attach(layer)

        layer._drag_modes[Mode.VERTEX_REMOVE](layer, event)

        np.testing.assert_allclose(np.asarray(layer.data[0], dtype=float), expected_vertices)
        assert events == [
            (ActionType.CHANGING, (0,), ((deleted_vertex_index,),)),
            (ActionType.CHANGED, (0,), ((deleted_vertex_index,),)),
        ]


def test_annotation_layer_edit_guard_vertex_remove_removes_minimal_hole(monkeypatch) -> None:
    shell_yx = np.asarray([[0.0, 0.0], [0.0, 10.0], [10.0, 10.0], [10.0, 0.0]], dtype=float)
    hole_yx = np.asarray([[3.0, 3.0], [3.0, 5.0], [5.0, 4.0]], dtype=float)
    polygon = Polygon(_yx_to_xy(shell_yx), holes=[_yx_to_xy(hole_yx)])
    original_vertices = shapely_polygon_to_napari_polygon_vertices(polygon)
    topology = napari_polygon_vertices_to_topology(original_vertices)
    expected_vertices, expected_topology = delete_napari_polygon_vertex(original_vertices, topology, 5)
    layer = Shapes([original_vertices], shape_type="polygon")
    layer._drag_modes = dict(layer._drag_modes)

    def original_vertex_remove_callback(*args: object, **kwargs: object) -> None:
        raise AssertionError("minimal-hole deletion should use the topology helper")

    monkeypatch.setattr(layer, "get_value", lambda position, world=True: (0, 5))
    layer._drag_modes[Mode.VERTEX_REMOVE] = original_vertex_remove_callback
    guard = shapes_annotation_widget_module._AnnotationLayerEditGuard()
    guard.attach(layer)

    layer._drag_modes[Mode.VERTEX_REMOVE](layer, SimpleNamespace(position=(0.0, 0.0)))

    np.testing.assert_allclose(np.asarray(layer.data[0], dtype=float), expected_vertices)
    assert not expected_topology.synchronized_anchor_groups


def test_annotation_layer_edit_guard_vertex_remove_rebuilds_cache_after_shortening_hole_row() -> None:
    polygon, simple_polygon = _polygon_hole_roundtrip_fixture()
    original_hole_vertices = shapely_polygon_to_napari_polygon_vertices(polygon)
    original_simple_vertices = shapely_polygon_to_napari_polygon_vertices(simple_polygon)
    layer = Shapes(
        [original_hole_vertices, original_simple_vertices],
        shape_type=["polygon", "polygon"],
        features=pd.DataFrame({"instance_id": ["hole_row", "simple_row"]}),
    )
    layer.mode = Mode.VERTEX_REMOVE
    layer.selected_data = {0}
    layer._drag_modes = dict(layer._drag_modes)

    def original_vertex_remove_callback(*args: object, **kwargs: object) -> None:
        raise AssertionError("hole-bearing polygon deletion should use the topology helper")

    layer._drag_modes[Mode.VERTEX_REMOVE] = original_vertex_remove_callback
    original_features = layer.features.copy()
    original_shape_types = list(layer.shape_type)
    events: list[tuple[ActionType, tuple[int, ...], tuple[tuple[int, ...], ...]]] = []

    def record_data_event(event: object) -> None:
        events.append((event.action, event.data_indices, event.vertex_indices))

    layer.events.data.connect(record_data_event)
    guard = shapes_annotation_widget_module._AnnotationLayerEditGuard()
    guard.attach(layer)

    layer._drag_modes[Mode.VERTEX_REMOVE](layer, SimpleNamespace(position=original_hole_vertices[7]))

    shortened_vertices = np.asarray(layer.data[0], dtype=float)
    assert len(shortened_vertices) == len(original_hole_vertices) - 1
    assert layer._data_view._vertices_index.tolist() == [
        0,
        len(shortened_vertices),
        len(shortened_vertices) + len(original_simple_vertices),
    ]
    shell_hit = layer.get_value(shortened_vertices[0], world=True)
    assert shell_hit is not None
    assert tuple(int(index) for index in shell_hit) == (0, len(shortened_vertices) - 1)
    pd.testing.assert_frame_equal(layer.features, original_features)
    assert list(layer.shape_type) == original_shape_types
    assert layer.mode == Mode.VERTEX_REMOVE
    assert set(layer.selected_data) == {0}

    layer._drag_modes[Mode.VERTEX_REMOVE](layer, SimpleNamespace(position=shortened_vertices[0]))

    shell_deleted_vertices = np.asarray(layer.data[0], dtype=float)
    assert len(shell_deleted_vertices) == len(shortened_vertices) - 1
    assert len(napari_polygon_vertices_to_shapely_polygon(shell_deleted_vertices).interiors) == 1
    pd.testing.assert_frame_equal(layer.features, original_features)
    assert list(layer.shape_type) == original_shape_types
    assert layer.mode == Mode.VERTEX_REMOVE
    assert set(layer.selected_data) == {0}
    assert events == [
        (ActionType.CHANGING, (0,), ((7,),)),
        (ActionType.CHANGED, (0,), ((7,),)),
        (ActionType.CHANGING, (0,), ((len(shortened_vertices) - 1,),)),
        (ActionType.CHANGED, (0,), ((len(shortened_vertices) - 1,),)),
    ]


def test_annotation_layer_edit_guard_vertex_remove_preserves_styles_with_stale_napari_color_arrays(monkeypatch) -> None:
    polygon, simple_polygon = _polygon_hole_roundtrip_fixture()
    original_hole_vertices = shapely_polygon_to_napari_polygon_vertices(polygon)
    original_simple_vertices = shapely_polygon_to_napari_polygon_vertices(simple_polygon)
    deleted_vertex_index = 7
    topology = napari_polygon_vertices_to_topology(original_hole_vertices)
    expected_vertices, _ = delete_napari_polygon_vertex(
        original_hole_vertices,
        topology,
        deleted_vertex_index,
    )
    layer = Shapes(
        [original_hole_vertices, original_simple_vertices],
        shape_type=["polygon", "polygon"],
    )
    apply_primary_shapes_layer_style(layer)
    layer.mode = Mode.VERTEX_REMOVE
    layer.selected_data = {0}
    layer._drag_modes = dict(layer._drag_modes)

    current_edge_color = "#aa00aa"
    current_face_color = "#bbccdd44"
    current_edge_width = 9
    layer.current_edge_color = current_edge_color
    layer.current_face_color = current_face_color
    layer.current_edge_width = current_edge_width

    edge_color = np.asarray([_rgba("#00ffff"), _rgba("#123456")], dtype=float)
    face_color = np.asarray([_rgba("#00000000"), _rgba("#65432188")], dtype=float)
    edge_width = [3, 5]
    z_index = [11, 13]
    layer.edge_color = edge_color
    layer.face_color = face_color
    layer.edge_width = edge_width
    layer.z_index = z_index
    layer.opacity = 0.37

    # Simulate the intermittent napari state behind the UI bug: the logical
    # data rows are correct, but the private color arrays have stale extra rows.
    layer._data_view._edge_color = np.vstack([layer._data_view._edge_color, _rgba("#ffff00")])
    layer._data_view._face_color = np.vstack([layer._data_view._face_color, _rgba("#ffff00")])

    def original_vertex_remove_callback(*args: object, **kwargs: object) -> None:
        raise AssertionError("hole-bearing polygon deletion should use the topology helper")

    monkeypatch.setattr(layer, "get_value", lambda position, world=True: (0, deleted_vertex_index))
    layer._drag_modes[Mode.VERTEX_REMOVE] = original_vertex_remove_callback
    guard = shapes_annotation_widget_module._AnnotationLayerEditGuard()
    guard.attach(layer)

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        layer._drag_modes[Mode.VERTEX_REMOVE](layer, SimpleNamespace(position=(0.0, 0.0)))

    style_warnings = [
        str(warning.message)
        for warning in caught_warnings
        if "edge_color" in str(warning.message) or "face_color" in str(warning.message)
    ]
    assert style_warnings == []
    np.testing.assert_allclose(np.asarray(layer.data[0], dtype=float), expected_vertices)
    assert set(layer.selected_data) == {0}
    assert layer.mode == Mode.VERTEX_REMOVE
    assert layer.opacity == 0.37
    assert layer.edge_width == edge_width
    assert layer.z_index == z_index
    np.testing.assert_allclose(layer.edge_color, edge_color)
    np.testing.assert_allclose(layer.face_color, face_color)
    np.testing.assert_allclose(to_rgba(layer.current_edge_color), to_rgba(current_edge_color))
    np.testing.assert_allclose(to_rgba(layer.current_face_color), to_rgba(current_face_color))
    assert layer.current_edge_width == current_edge_width


def test_annotation_layer_edit_guard_vertex_remove_delegates_malformed_topology(monkeypatch) -> None:
    polygon, _ = _polygon_hole_roundtrip_fixture()
    malformed_vertices = shapely_polygon_to_napari_polygon_vertices(polygon)[:-1]
    layer = Shapes([malformed_vertices], shape_type="polygon")
    layer._drag_modes = dict(layer._drag_modes)
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def original_vertex_remove_callback(*args: object, **kwargs: object) -> str:
        calls.append((args, kwargs))
        return "delegated"

    monkeypatch.setattr(layer, "get_value", lambda position, world=True: (0, 0))
    layer._drag_modes[Mode.VERTEX_REMOVE] = original_vertex_remove_callback
    guard = shapes_annotation_widget_module._AnnotationLayerEditGuard()
    guard.attach(layer)
    event = SimpleNamespace(position=(0.0, 0.0))

    result = layer._drag_modes[Mode.VERTEX_REMOVE](layer, event)

    assert result == "delegated"
    assert calls == [((layer, event), {})]


def test_annotation_layer_edit_guard_vertex_remove_helper_error_warns_without_mutating_layer(monkeypatch) -> None:
    polygon, _ = _polygon_hole_roundtrip_fixture()
    original_vertices = shapely_polygon_to_napari_polygon_vertices(polygon)
    layer = Shapes([original_vertices], shape_type="polygon")
    layer._drag_modes = dict(layer._drag_modes)
    warnings: list[str] = []
    events: list[object] = []

    def original_vertex_remove_callback(*args: object, **kwargs: object) -> None:
        raise AssertionError("rejected hole deletion should not fall back to napari deletion")

    def reject_delete(*args: object, **kwargs: object) -> tuple[np.ndarray, object]:
        raise ValueError("Deletion would make the polygon invalid.")

    monkeypatch.setattr(layer, "get_value", lambda position, world=True: (0, 0))
    monkeypatch.setattr(shapes_annotation_widget_module, "delete_napari_polygon_vertex", reject_delete)
    layer._drag_modes[Mode.VERTEX_REMOVE] = original_vertex_remove_callback
    layer.events.data.connect(events.append)
    guard = shapes_annotation_widget_module._AnnotationLayerEditGuard(warning_callback=warnings.append)
    guard.attach(layer)

    layer._drag_modes[Mode.VERTEX_REMOVE](layer, SimpleNamespace(position=(0.0, 0.0)))

    np.testing.assert_allclose(np.asarray(layer.data[0], dtype=float), original_vertices)
    assert warnings == ["Deletion would make the polygon invalid."]
    assert events == []


def test_shapes_annotation_widget_shares_app_state(qtbot) -> None:
    viewer = DummyViewer()

    widget = ShapesAnnotation(viewer)

    qtbot.addWidget(widget)

    assert widget.app_state is get_or_create_app_state(viewer)


def test_shapes_annotation_widget_refreshes_when_shared_sdata_changes(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    viewer = DummyViewer()
    app_state = get_or_create_app_state(viewer)
    widget = ShapesAnnotation(viewer)

    qtbot.addWidget(widget)
    app_state.set_sdata(sdata_blobs)

    assert widget.selected_spatialdata is sdata_blobs
    assert _combo_texts(widget.coordinate_system_combo) == ["global"]
    assert _combo_data(widget.coordinate_system_combo) == ["global"]
    assert widget.coordinate_system_combo.currentText() == "global"
    assert widget.selected_coordinate_system == "global"
    assert widget.shapes_combo.currentText() == "Create shapes..."
    assert widget.name_edit.isEnabled() is True
    assert widget.create_layer_button.isEnabled() is False
    assert widget.save_shapes_button.isEnabled() is False
    assert "Shapes element name must not be empty" in _status_text(widget)


def test_shapes_annotation_widget_shapes_selector_auto_opens_existing_target(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    viewer = DummyViewer()
    app_state = get_or_create_app_state(viewer)
    app_state.set_sdata(sdata_blobs)
    widget = ShapesAnnotation(viewer)
    qtbot.addWidget(widget)
    existing_shapes_name = "blobs_polygons"

    index = _combo_index_for_text(widget.shapes_combo, existing_shapes_name)
    assert index >= 0
    widget.shapes_combo.setCurrentIndex(index)

    assert len(viewer.layers) == 1
    assert widget._selected_shapes_target == shapes_annotation_widget_module._ShapesAnnotationTarget.edit_existing(
        existing_shapes_name
    )
    assert widget.selected_shapes_name == existing_shapes_name
    assert widget.name_edit.isHidden() is True
    assert widget.name_edit.isEnabled() is False
    assert widget.create_layer_button.text() == "Create layer"
    assert widget.create_layer_button.isEnabled() is False
    assert widget.save_shapes_button.isEnabled() is True
    assert "Existing Shapes Opened" in _status_text(widget)


def test_shapes_annotation_widget_shapes_selector_defaults_back_to_create_when_existing_disappears(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    viewer = DummyViewer()
    app_state = get_or_create_app_state(viewer)
    app_state.set_sdata(sdata_blobs)
    widget = ShapesAnnotation(viewer)
    qtbot.addWidget(widget)
    existing_shapes_name = next(iter(sdata_blobs.shapes))
    widget.shapes_combo.setCurrentIndex(_combo_index_for_text(widget.shapes_combo, existing_shapes_name))

    sdata_blobs.shapes.pop(existing_shapes_name)
    widget.refresh_from_sdata(sdata_blobs)

    assert widget._selected_shapes_target == shapes_annotation_widget_module._ShapesAnnotationTarget.create_new()
    assert widget.shapes_combo.currentText() == "Create shapes..."
    assert widget.name_edit.isHidden() is False


def test_shapes_annotation_widget_user_coordinate_system_selection_updates_app_state(
    qtbot,
    monkeypatch,
    sdata_blobs: SpatialData,
) -> None:
    _patch_coordinate_system_names(monkeypatch, ["global", "local"])
    viewer = DummyViewer()
    app_state = get_or_create_app_state(viewer)
    app_state.set_sdata(sdata_blobs)
    widget = ShapesAnnotation(viewer)

    qtbot.addWidget(widget)
    widget.coordinate_system_combo.setCurrentIndex(1)

    assert app_state.coordinate_system == "local"
    assert widget.selected_coordinate_system == "local"


def test_shapes_annotation_widget_external_coordinate_system_change_updates_selector(
    qtbot,
    monkeypatch,
    sdata_blobs: SpatialData,
) -> None:
    _patch_coordinate_system_names(monkeypatch, ["global", "local"])
    viewer = DummyViewer()
    app_state = get_or_create_app_state(viewer)
    app_state.set_sdata(sdata_blobs)
    widget = ShapesAnnotation(viewer)

    qtbot.addWidget(widget)
    app_state.set_coordinate_system("local", source="viewer_widget")

    assert widget.coordinate_system_combo.currentText() == "local"
    assert widget.selected_coordinate_system == "local"


def test_shapes_annotation_widget_disables_create_when_coordinate_system_is_cleared(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    viewer = DummyViewer()
    app_state = get_or_create_app_state(viewer)
    app_state.set_sdata(sdata_blobs)
    widget = ShapesAnnotation(viewer)

    qtbot.addWidget(widget)
    widget.name_edit.setText("new_regions")
    app_state.clear_coordinate_system(source="test")

    assert widget.coordinate_system_combo.currentIndex() == -1
    assert widget.selected_coordinate_system is None
    assert widget.create_layer_button.isEnabled() is False
    assert "Choose Coordinate System" in _status_text(widget)


def test_shapes_annotation_widget_validates_empty_invalid_and_duplicate_names(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    viewer = DummyViewer()
    app_state = get_or_create_app_state(viewer)
    app_state.set_sdata(sdata_blobs)
    widget = ShapesAnnotation(viewer)

    qtbot.addWidget(widget)
    assert "Shapes element name must not be empty" in _status_text(widget)

    widget.name_edit.setText("bad/name")
    assert widget.create_layer_button.isEnabled() is False
    assert "must be a valid SpatialData name" in _status_text(widget)

    existing_shapes_name = next(iter(sdata_blobs.shapes))
    widget.name_edit.setText(existing_shapes_name)
    assert widget.create_layer_button.isEnabled() is False
    assert "Name Already Exists" in _status_text(widget)

    widget.name_edit.setText(existing_shapes_name.upper())
    assert widget.create_layer_button.isEnabled() is False
    assert "Name Already Exists" in _status_text(widget)


def test_shapes_annotation_widget_status_cards_shorten_long_identifiers(
    qtbot,
    monkeypatch,
    sdata_blobs: SpatialData,
) -> None:
    coordinate_system = "global_long_name_" + "x" * 80
    shapes_name = "annotation_shapes_long_name_" + "y" * 80
    _patch_coordinate_system_names(monkeypatch, [coordinate_system])
    viewer = DummyViewer()
    app_state = get_or_create_app_state(viewer)
    app_state.set_sdata(sdata_blobs)
    widget = ShapesAnnotation(viewer)
    qtbot.addWidget(widget)

    widget.name_edit.setText(shapes_name)

    status = _status_text(widget)
    assert "Ready" in status
    assert shapes_name not in status
    assert coordinate_system not in status
    assert "…" in status
    tooltip = _tooltip_text(widget)
    assert shapes_name in tooltip
    assert coordinate_system in tooltip

    def fake_create_shapes_element(request, napari_layer):
        del napari_layer
        request.sdata.shapes[request.shapes_name] = request.sdata.shapes["blobs_polygons"].copy()
        return AnnotateShapesElementResult(
            shapes_name=request.shapes_name,
            coordinate_system=request.coordinate_system,
            row_count=2,
        )

    monkeypatch.setattr(
        shapes_annotation_widget_module,
        "create_shapes_element_from_napari_shapes_layer",
        fake_create_shapes_element,
    )
    widget.create_layer_button.click()
    widget.save_shapes_button.click()

    status = _status_text(widget)
    assert "Shapes Saved" in status
    assert shapes_name not in status
    assert coordinate_system not in status
    assert "…" in status
    tooltip = _tooltip_text(widget)
    assert shapes_name in tooltip
    assert coordinate_system in tooltip


def test_shapes_annotation_widget_create_layer_adds_registered_active_empty_shapes_layer(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    viewer = DummyViewer()
    widget = _create_ready_annotation_widget(qtbot, viewer, sdata_blobs)

    assert widget.create_layer_button.isEnabled() is True
    widget.create_layer_button.click()

    assert len(viewer.layers) == 1
    layer = viewer.layers[0]
    assert isinstance(layer, Shapes)
    assert layer.name == "new_regions"
    assert len(layer.data) == 0
    assert layer.ndim == 2
    assert layer.current_edge_width == 1
    np.testing.assert_allclose(to_rgba(layer.current_edge_color), to_rgba("#00FFFF"))
    np.testing.assert_allclose(to_rgba(layer.current_face_color), to_rgba("#00000000"))
    assert layer.opacity == 0.8
    assert hasattr(layer, _SHAPES_EDGE_WIDTH_SYNC_CALLBACK_ATTR)
    assert hasattr(layer, _SHAPES_EDGE_COLOR_SYNC_CALLBACK_ATTR)

    binding = widget.app_state.viewer_adapter.layer_bindings.get_binding(layer)
    assert isinstance(binding, ShapesLayerBinding)
    assert binding.element_name == "new_regions"
    assert binding.coordinate_system == "global"
    assert binding.shapes_role == "primary"
    assert binding.shapes_rendering_mode == "shapes"
    assert binding.style_spec is None
    assert binding.source_shapes_index_feature_name == "instance_id"
    assert viewer.layers.selection.active is layer

    assert widget.selected_shapes_name == "new_regions"
    assert widget._annotation_layer is layer
    assert widget._annotation_edit_guard.layer is layer
    assert "_drag_modes" in vars(layer)
    assert layer._drag_modes[Mode.DIRECT] is widget._annotation_edit_guard._wrapped_direct_callback
    assert widget._annotation_shapes_name == "new_regions"
    assert widget._annotation_coordinate_system == "global"
    assert widget._annotation_has_been_saved is False
    assert widget._annotation_session is not None
    assert widget._annotation_session.reload_on_discard is False
    assert widget.name_edit.isEnabled() is False
    assert widget.create_layer_button.isEnabled() is False
    assert widget.save_shapes_button.isEnabled() is True
    assert "Annotation Layer Ready" in _status_text(widget)


def test_shapes_annotation_widget_cancelling_coordinate_change_preserves_annotation_layer(
    qtbot,
    monkeypatch,
    sdata_blobs: SpatialData,
) -> None:
    _patch_coordinate_system_names(monkeypatch, ["global", "local"])
    viewer = DummyViewer()
    widget = _create_ready_annotation_widget(qtbot, viewer, sdata_blobs)
    widget.create_layer_button.click()
    layer = viewer.layers[0]
    _add_polygon(layer)
    discard_contexts: list[str] = []

    def cancel_discard(*, context: str) -> bool:
        discard_contexts.append(context)
        return False

    monkeypatch.setattr(widget, "_confirm_discard_annotation_layer", cancel_discard)

    widget.coordinate_system_combo.setCurrentIndex(1)

    assert discard_contexts == ["coordinate_system"]
    assert widget.app_state.coordinate_system == "global"
    assert widget.coordinate_system_combo.currentText() == "global"
    assert widget.selected_coordinate_system == "global"
    assert list(viewer.layers) == [layer]
    assert widget.app_state.viewer_adapter.layer_bindings.get_binding(layer) is not None
    assert widget._annotation_layer is layer
    assert widget.name_edit.isEnabled() is False
    assert widget.create_layer_button.isEnabled() is False
    assert widget.save_shapes_button.isEnabled() is True


def test_shapes_annotation_widget_clean_coordinate_change_closes_empty_create_layer_without_warning(
    qtbot,
    monkeypatch,
    sdata_blobs: SpatialData,
) -> None:
    _patch_coordinate_system_names(monkeypatch, ["global", "local"])
    viewer = DummyViewer()
    widget = _create_ready_annotation_widget(qtbot, viewer, sdata_blobs)
    widget.create_layer_button.click()
    layer = viewer.layers[0]

    def fail_if_confirmed(*, context: str) -> bool:
        raise AssertionError(f"Clean coordinate-system switch should not warn: {context}")

    monkeypatch.setattr(widget, "_confirm_discard_annotation_layer", fail_if_confirmed)

    widget.coordinate_system_combo.setCurrentIndex(1)

    assert widget.app_state.coordinate_system == "local"
    assert widget.coordinate_system_combo.currentText() == "local"
    assert widget.selected_coordinate_system == "local"
    assert list(viewer.layers) == []
    assert widget.app_state.viewer_adapter.layer_bindings.get_binding(layer) is None
    assert widget._annotation_layer is None
    assert widget._annotation_edit_guard.layer is None
    assert "_drag_modes" not in vars(layer)
    assert widget._annotation_shapes_name is None
    assert widget._annotation_coordinate_system is None
    assert widget._annotation_has_been_saved is False
    assert widget._annotation_session is None
    assert widget.name_edit.isEnabled() is True
    assert widget.create_layer_button.isEnabled() is True
    assert widget.save_shapes_button.isEnabled() is False
    assert "Ready" in _status_text(widget)


def test_shapes_annotation_widget_cancelling_target_change_preserves_annotation_layer(
    qtbot,
    monkeypatch,
    sdata_blobs: SpatialData,
) -> None:
    viewer = DummyViewer()
    widget = _create_ready_annotation_widget(qtbot, viewer, sdata_blobs)
    widget.create_layer_button.click()
    layer = viewer.layers[0]
    existing_shapes_name = "blobs_polygons"
    _add_polygon(layer)
    discard_contexts: list[str] = []

    def cancel_discard(*, context: str) -> bool:
        discard_contexts.append(context)
        return False

    monkeypatch.setattr(widget, "_confirm_discard_annotation_layer", cancel_discard)

    widget.shapes_combo.setCurrentIndex(_combo_index_for_text(widget.shapes_combo, existing_shapes_name))

    assert discard_contexts == ["target"]
    assert widget.shapes_combo.currentText() == "Create shapes..."
    assert widget._selected_shapes_target == shapes_annotation_widget_module._ShapesAnnotationTarget.create_new()
    assert list(viewer.layers) == [layer]
    assert widget.app_state.viewer_adapter.layer_bindings.get_binding(layer) is not None
    assert widget._annotation_layer is layer
    assert widget.name_edit.isHidden() is False
    assert widget.name_edit.isEnabled() is False
    assert widget.create_layer_button.isEnabled() is False
    assert widget.save_shapes_button.isEnabled() is True


def test_shapes_annotation_widget_clean_target_change_closes_empty_create_layer_without_warning(
    qtbot,
    monkeypatch,
    sdata_blobs: SpatialData,
) -> None:
    viewer = DummyViewer()
    widget = _create_ready_annotation_widget(qtbot, viewer, sdata_blobs)
    widget.create_layer_button.click()
    layer = viewer.layers[0]
    existing_shapes_name = "blobs_polygons"

    def fail_if_confirmed(*, context: str) -> bool:
        raise AssertionError(f"Clean target switch should not warn: {context}")

    monkeypatch.setattr(widget, "_confirm_discard_annotation_layer", fail_if_confirmed)

    widget.shapes_combo.setCurrentIndex(_combo_index_for_text(widget.shapes_combo, existing_shapes_name))

    assert len(viewer.layers) == 1
    opened_layer = viewer.layers[0]
    assert opened_layer is not layer
    assert isinstance(opened_layer, Shapes)
    assert widget.app_state.viewer_adapter.layer_bindings.get_binding(layer) is None
    assert widget._annotation_layer is opened_layer
    assert widget._annotation_shapes_name == existing_shapes_name
    assert widget._annotation_coordinate_system == "global"
    assert widget._annotation_has_been_saved is True
    assert widget._selected_shapes_target == shapes_annotation_widget_module._ShapesAnnotationTarget.edit_existing(
        existing_shapes_name
    )
    assert widget.name_edit.isHidden() is True
    assert widget.create_layer_button.text() == "Create layer"
    assert widget.create_layer_button.isEnabled() is False
    assert widget.save_shapes_button.isEnabled() is True
    assert "Existing Shapes Opened" in _status_text(widget)


def test_shapes_annotation_widget_clean_saved_target_change_keeps_saved_layer_without_warning(
    qtbot,
    monkeypatch,
    sdata_blobs: SpatialData,
) -> None:
    viewer = DummyViewer()
    widget = _create_ready_annotation_widget(qtbot, viewer, sdata_blobs)
    widget.create_layer_button.click()
    saved_layer = viewer.layers[0]
    _add_polygon(saved_layer)
    widget.save_shapes_button.click()

    def fail_if_confirmed(*, context: str) -> bool:
        raise AssertionError(f"Clean saved target switch should not warn: {context}")

    monkeypatch.setattr(widget, "_confirm_discard_annotation_layer", fail_if_confirmed)

    widget.shapes_combo.setCurrentIndex(_combo_index_for_text(widget.shapes_combo, "blobs_polygons"))

    assert saved_layer in viewer.layers
    assert widget.app_state.viewer_adapter.layer_bindings.get_binding(saved_layer) is not None
    assert widget._annotation_shapes_name == "blobs_polygons"
    assert widget._annotation_layer is not saved_layer
    assert len(viewer.layers) == 2


def test_shapes_annotation_widget_clean_existing_target_switch_preserves_layer_order(
    qtbot,
    monkeypatch,
    sdata_blobs: SpatialData,
) -> None:
    sdata_blobs.shapes["other_polygons"] = sdata_blobs.shapes["blobs_polygons"].copy()
    viewer = DummyViewer()
    app_state = get_or_create_app_state(viewer)
    app_state.set_sdata(sdata_blobs)
    widget = ShapesAnnotation(viewer)
    qtbot.addWidget(widget)

    def fail_if_confirmed(*, context: str) -> bool:
        raise AssertionError(f"Clean existing target switch should not warn: {context}")

    monkeypatch.setattr(widget, "_confirm_discard_annotation_layer", fail_if_confirmed)

    widget.shapes_combo.setCurrentIndex(_combo_index_for_text(widget.shapes_combo, "blobs_polygons"))
    blobs_layer = widget._annotation_layer
    assert blobs_layer is not None
    assert widget._annotation_edit_guard.layer is blobs_layer
    assert "_drag_modes" in vars(blobs_layer)

    widget.shapes_combo.setCurrentIndex(_combo_index_for_text(widget.shapes_combo, "other_polygons"))
    other_layer = widget._annotation_layer
    assert other_layer is not None
    assert other_layer is not blobs_layer
    assert list(viewer.layers) == [blobs_layer, other_layer]
    assert widget._annotation_edit_guard.layer is other_layer
    assert "_drag_modes" not in vars(blobs_layer)
    assert "_drag_modes" in vars(other_layer)

    widget.shapes_combo.setCurrentIndex(_combo_index_for_text(widget.shapes_combo, "blobs_polygons"))

    assert widget._annotation_layer is blobs_layer
    assert list(viewer.layers) == [blobs_layer, other_layer]
    assert widget._annotation_edit_guard.layer is blobs_layer
    assert "_drag_modes" in vars(blobs_layer)
    assert "_drag_modes" not in vars(other_layer)


def test_shapes_annotation_widget_open_existing_target_loads_edit_session_layer(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    viewer = DummyViewer()
    app_state = get_or_create_app_state(viewer)
    app_state.set_sdata(sdata_blobs)
    widget = ShapesAnnotation(viewer)
    qtbot.addWidget(widget)
    widget.shapes_combo.setCurrentIndex(_combo_index_for_text(widget.shapes_combo, "blobs_polygons"))

    assert len(viewer.layers) == 1
    layer = viewer.layers[0]
    assert isinstance(layer, Shapes)
    assert widget._annotation_layer is layer
    assert widget._annotation_edit_guard.layer is layer
    assert "_drag_modes" in vars(layer)
    assert layer._drag_modes[Mode.DIRECT] is widget._annotation_edit_guard._wrapped_direct_callback
    assert widget._annotation_shapes_name == "blobs_polygons"
    assert widget._annotation_coordinate_system == "global"
    assert widget._annotation_session is not None
    assert widget._annotation_session.mode == "edit_existing"
    assert widget._annotation_session.layer_origin == "loaded_by_annotation"
    assert widget._annotation_session.shapes_name == "blobs_polygons"
    assert widget._annotation_session.coordinate_system == "global"
    assert widget._annotation_session.source_shapes_index_feature_name == "index"
    assert widget._annotation_session.source_geodataframe is not sdata_blobs.shapes["blobs_polygons"]
    assert widget._annotation_session.source_geodataframe is not None
    assert widget._annotation_session.source_geodataframe.index.equals(sdata_blobs.shapes["blobs_polygons"].index)
    assert widget._annotation_session.source_geodataframe_index_name == sdata_blobs.shapes["blobs_polygons"].index.name
    assert widget._annotation_session.table_linked is False
    assert widget._annotation_session.reload_on_discard is True
    assert viewer.layers.selection.active is layer
    assert widget.create_layer_button.isEnabled() is False
    assert widget.save_shapes_button.isEnabled() is True
    assert "Existing Shapes Opened" in _status_text(widget)


def test_shapes_annotation_widget_open_existing_target_adopts_loaded_primary_layer(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    viewer = DummyViewer()
    app_state = get_or_create_app_state(viewer)
    app_state.set_sdata(sdata_blobs)
    load_result = app_state.viewer_adapter.ensure_shapes_loaded(sdata_blobs, "blobs_polygons", "global")
    widget = ShapesAnnotation(viewer)
    qtbot.addWidget(widget)
    widget.shapes_combo.setCurrentIndex(_combo_index_for_text(widget.shapes_combo, "blobs_polygons"))

    assert len(viewer.layers) == 1
    assert widget._annotation_layer is load_result.layer
    assert widget._annotation_session is not None
    assert widget._annotation_session.layer_origin == "adopted_primary"
    assert viewer.layers.selection.active is load_result.layer


def test_shapes_annotation_widget_adopts_selected_target_loaded_from_viewer(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    viewer = DummyViewer()
    app_state = get_or_create_app_state(viewer)
    app_state.set_sdata(sdata_blobs)
    widget = ShapesAnnotation(viewer)
    qtbot.addWidget(widget)
    target = shapes_annotation_widget_module._ShapesAnnotationTarget.edit_existing("blobs_polygons")
    widget._refresh_shapes_targets(preferred_target=target)
    widget._refresh_create_layer_state()

    assert widget._selected_shapes_target == target
    assert widget._annotation_layer is None
    assert widget._annotation_session is None
    assert widget.save_shapes_button.isEnabled() is False

    load_result = app_state.viewer_adapter.ensure_shapes_loaded(sdata_blobs, "blobs_polygons", "global")

    assert load_result.created is True
    assert widget._annotation_layer is load_result.layer
    assert widget._annotation_session is not None
    assert widget._annotation_session.layer_origin == "adopted_primary"
    assert widget._annotation_session.shapes_name == "blobs_polygons"
    assert widget._annotation_session.coordinate_system == "global"
    assert widget.save_shapes_button.isEnabled() is True
    assert viewer.layers.selection.active is load_result.layer


def test_shapes_annotation_widget_ignores_viewer_loaded_nonmatching_target(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    viewer = DummyViewer()
    app_state = get_or_create_app_state(viewer)
    sdata_blobs.shapes["other_polygons"] = sdata_blobs.shapes["blobs_polygons"].copy()
    app_state.set_sdata(sdata_blobs)
    widget = ShapesAnnotation(viewer)
    qtbot.addWidget(widget)
    target = shapes_annotation_widget_module._ShapesAnnotationTarget.edit_existing("blobs_polygons")
    widget._refresh_shapes_targets(preferred_target=target)
    widget._refresh_create_layer_state()

    load_result = app_state.viewer_adapter.ensure_shapes_loaded(sdata_blobs, "other_polygons", "global")

    assert load_result.created is True
    assert widget._annotation_layer is None
    assert widget._annotation_session is None
    assert widget.save_shapes_button.isEnabled() is False


def test_shapes_annotation_widget_viewer_load_does_not_steal_active_session(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    viewer = DummyViewer()
    app_state = get_or_create_app_state(viewer)
    sdata_blobs.shapes["other_polygons"] = sdata_blobs.shapes["blobs_polygons"].copy()
    app_state.set_sdata(sdata_blobs)
    widget = ShapesAnnotation(viewer)
    qtbot.addWidget(widget)
    widget.shapes_combo.setCurrentIndex(_combo_index_for_text(widget.shapes_combo, "blobs_polygons"))
    annotation_layer = widget._annotation_layer

    load_result = app_state.viewer_adapter.ensure_shapes_loaded(sdata_blobs, "other_polygons", "global")

    assert load_result.created is True
    assert annotation_layer is not None
    assert widget._annotation_layer is annotation_layer
    assert widget._annotation_session is not None
    assert widget._annotation_session.shapes_name == "blobs_polygons"
    assert widget._annotation_session.layer_origin == "loaded_by_annotation"


def test_shapes_annotation_widget_open_existing_target_rejects_multipolygon_source(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    viewer = DummyViewer()
    app_state = get_or_create_app_state(viewer)
    app_state.set_sdata(sdata_blobs)
    widget = ShapesAnnotation(viewer)
    qtbot.addWidget(widget)
    widget.shapes_combo.setCurrentIndex(_combo_index_for_text(widget.shapes_combo, "blobs_multipolygons"))

    assert list(viewer.layers) == []
    assert widget._annotation_layer is None
    assert widget._annotation_session is None
    assert widget.create_layer_button.isEnabled() is False
    assert "Could Not Open Shapes" in _status_text(widget)
    assert "Polygon geometries only" in _status_text(widget)


def test_shapes_annotation_widget_edit_existing_save_updates_shapes_element_and_session_snapshot(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    viewer = DummyViewer()
    app_state = get_or_create_app_state(viewer)
    app_state.set_sdata(sdata_blobs)
    widget = ShapesAnnotation(viewer)
    qtbot.addWidget(widget)
    shapes_name = "blobs_polygons"
    original_index = sdata_blobs.shapes[shapes_name].index.to_list()
    emitted_events: list[object] = []
    widget.app_state.shapes_element_written.connect(emitted_events.append)

    widget.shapes_combo.setCurrentIndex(_combo_index_for_text(widget.shapes_combo, shapes_name))
    layer = widget._annotation_layer
    assert isinstance(layer, Shapes)
    assert widget.save_shapes_button.isEnabled() is True

    _add_polygon(layer, offset=100)
    layer.features.loc[len(layer.features) - 1, "index"] = None
    widget.save_shapes_button.click()

    saved_geodataframe = sdata_blobs.shapes[shapes_name]
    assert saved_geodataframe.index.to_list() == [*original_index, "__annotation_0"]
    assert widget._annotation_session is not None
    assert widget._annotation_session.mode == "edit_existing"
    assert widget._annotation_session.source_geodataframe is not saved_geodataframe
    assert widget._annotation_session.source_geodataframe is not None
    assert widget._annotation_session.source_geodataframe.index.equals(saved_geodataframe.index)
    assert widget._annotation_session.source_shapes_index_feature_name == "index"
    assert widget._annotation_session.reload_on_discard is True
    assert emitted_events == [
        ShapesElementWrittenEvent(
            sdata=sdata_blobs,
            shapes_name=shapes_name,
            coordinate_system="global",
            source="shapes_annotation_widget",
        )
    ]
    assert "Shapes Saved" in _status_text(widget)


def test_shapes_annotation_widget_edit_existing_preserves_polygon_holes_on_save(qtbot) -> None:
    shapes_name = "hole_regions"
    polygon_1, polygon_2 = _polygon_hole_roundtrip_fixture()
    sdata = _make_polygon_hole_roundtrip_sdata(shapes_name=shapes_name)
    viewer = DummyViewer()
    app_state = get_or_create_app_state(viewer)
    app_state.set_sdata(sdata)
    widget = ShapesAnnotation(viewer)
    qtbot.addWidget(widget)

    widget.shapes_combo.setCurrentIndex(_combo_index_for_text(widget.shapes_combo, shapes_name))
    layer = widget._annotation_layer
    assert isinstance(layer, Shapes)
    assert widget.save_shapes_button.isEnabled() is True

    widget.save_shapes_button.click()

    saved = sdata.shapes[shapes_name]
    assert saved.index.name == "region_id"
    assert saved.index.tolist() == ["hole_row", "simple_row"]
    assert saved["label"].tolist() == ["polygon_with_hole", "simple_polygon"]
    np.testing.assert_allclose(saved["score"].to_numpy(), np.asarray([1.25, 2.5]))
    _assert_polygon_hole_geometries_preserved(saved, polygon_1, polygon_2)
    assert widget._annotation_session is not None
    assert widget._annotation_session.source_geodataframe is not saved
    assert widget._annotation_session.source_geodataframe is not None
    assert widget._annotation_session.source_geodataframe.index.equals(saved.index)
    assert "Shapes Saved" in _status_text(widget)


def test_shapes_annotation_widget_edit_existing_preserves_non_anchor_hole_vertex_edits(qtbot) -> None:
    shapes_name = "hole_regions"
    original_polygon_1, polygon_2 = _polygon_hole_roundtrip_fixture()
    sdata = _make_polygon_hole_roundtrip_sdata(shapes_name=shapes_name)
    viewer = DummyViewer()
    app_state = get_or_create_app_state(viewer)
    app_state.set_sdata(sdata)
    widget = ShapesAnnotation(viewer)
    qtbot.addWidget(widget)

    widget.shapes_combo.setCurrentIndex(_combo_index_for_text(widget.shapes_combo, shapes_name))
    layer = widget._annotation_layer
    assert isinstance(layer, Shapes)

    edited_vertices = np.asarray(layer.data[0], dtype=float).copy()
    np.testing.assert_allclose(edited_vertices[[0, 5, 12]], edited_vertices[[0, 0, 0]])
    np.testing.assert_allclose(edited_vertices[[6, 11]], edited_vertices[[6, 6]])
    edited_vertices[2] += np.asarray([35.0, -45.0])
    edited_vertices[8] += np.asarray([-25.0, 30.0])
    layer.data = [edited_vertices, np.asarray(layer.data[1], dtype=float)]
    expected_polygon_1 = Polygon(
        _yx_to_xy(edited_vertices[:6]),
        holes=[_yx_to_xy(edited_vertices[6:12])],
    )
    assert expected_polygon_1.is_valid
    assert not expected_polygon_1.equals(original_polygon_1)

    widget.save_shapes_button.click()

    saved = sdata.shapes[shapes_name]
    assert saved.index.name == "region_id"
    assert saved.index.tolist() == ["hole_row", "simple_row"]
    assert saved["label"].tolist() == ["polygon_with_hole", "simple_polygon"]
    np.testing.assert_allclose(saved["score"].to_numpy(), np.asarray([1.25, 2.5]))
    _assert_polygon_hole_geometries_preserved(saved, expected_polygon_1, polygon_2)

    reloaded_viewer = DummyViewer()
    reloaded_layer = get_or_create_app_state(reloaded_viewer).viewer_adapter.ensure_shapes_loaded(
        sdata,
        shapes_name,
        "global",
    ).layer
    reloaded_polygon_1 = napari_polygon_vertices_to_shapely_polygon(reloaded_layer.data[0])

    assert reloaded_polygon_1.equals(saved.geometry.iloc[0])
    assert len(reloaded_polygon_1.interiors) == 1


def test_shapes_annotation_widget_create_holes_invalid_selection_warns_without_mutation(qtbot) -> None:
    shapes_name = "create_holes_regions"
    sdata, _shell, _child, _unselected = _make_create_holes_sdata(shapes_name=shapes_name)
    viewer = DummyViewer()
    app_state = get_or_create_app_state(viewer)
    app_state.set_sdata(sdata)
    widget = ShapesAnnotation(viewer)
    qtbot.addWidget(widget)

    widget.shapes_combo.setCurrentIndex(_combo_index_for_text(widget.shapes_combo, shapes_name))
    layer = widget._annotation_layer
    assert isinstance(layer, Shapes)
    assert widget.create_holes_button.isEnabled() is True
    original_data = [np.asarray(vertices, dtype=float).copy() for vertices in layer.data]
    original_features = layer.features.copy()
    layer.selected_data = {0}

    widget.create_holes_button.click()

    assert "Could Not Create Holes" in _status_text(widget)
    assert "Select one shell polygon" in _status_text(widget)
    _assert_layer_data_unchanged(layer, original_data)
    pd.testing.assert_frame_equal(layer.features, original_features)
    assert widget.save_shapes_button.isEnabled() is True
    assert widget.create_holes_button.isEnabled() is True


def test_shapes_annotation_widget_create_holes_mutates_layer_and_marks_dirty(qtbot) -> None:
    shapes_name = "create_holes_regions"
    sdata, shell, child, unselected = _make_create_holes_sdata(shapes_name=shapes_name)
    viewer = DummyViewer()
    app_state = get_or_create_app_state(viewer)
    app_state.set_sdata(sdata)
    widget = ShapesAnnotation(viewer)
    qtbot.addWidget(widget)

    widget.shapes_combo.setCurrentIndex(_combo_index_for_text(widget.shapes_combo, shapes_name))
    layer = widget._annotation_layer
    assert isinstance(layer, Shapes)
    assert widget._annotation_layer_has_unsaved_changes() is False
    layer.selected_data = {0, 1}

    widget.create_holes_button.click()

    assert len(layer.data) == 2
    assert list(layer.shape_type) == ["polygon", "polygon"]
    assert layer.features["region_id"].tolist() == ["shell_row", "unselected_row"]
    assert set(layer.selected_data) == {0}
    expected_shell = Polygon(shell.exterior.coords, holes=[child.exterior.coords])
    assert napari_polygon_vertices_to_shapely_polygon(layer.data[0]).equals(expected_shell)
    assert napari_polygon_vertices_to_shapely_polygon(layer.data[1]).equals(unselected)
    assert widget._annotation_layer_has_unsaved_changes() is True
    assert widget.save_shapes_button.isEnabled() is True
    assert widget.create_holes_button.isEnabled() is True
    status = _status_text(widget)
    assert "Created Holes" in status
    assert "Converted 1 selected polygon(s) into hole(s) and removed their shape row(s)." in status


def test_shapes_annotation_widget_create_holes_table_linked_warning_is_explicit(qtbot) -> None:
    shapes_name = "create_holes_regions"
    table_name = "create_holes_table"
    sdata, _shell, _child, _unselected = _make_create_holes_sdata(shapes_name=shapes_name)
    table = _add_dummy_table_annotating_shapes(sdata, shapes_name=shapes_name, table_name=table_name)
    original_obs = table.obs.copy(deep=True)
    viewer = DummyViewer()
    app_state = get_or_create_app_state(viewer)
    app_state.set_sdata(sdata)
    widget = ShapesAnnotation(viewer)
    qtbot.addWidget(widget)

    widget.shapes_combo.setCurrentIndex(_combo_index_for_text(widget.shapes_combo, shapes_name))
    layer = widget._annotation_layer
    assert isinstance(layer, Shapes)
    assert widget._annotation_session is not None
    assert widget._annotation_session.table_linked is True
    layer.selected_data = {0, 1}

    widget.create_holes_button.click()

    assert sdata.tables[table_name] is table
    pd.testing.assert_frame_equal(sdata.tables[table_name].obs, original_obs)
    status = _status_text(widget)
    assert "Created Holes" in status
    assert "Linked tables are not updated automatically" in status
    assert "table annotations may no longer match the shapes rows" in status


# Keep end-to-end anchor-edit coverage focused: one shell-anchor edit on an
# edit-existing layer and one hole-anchor edit on an adopted native layer.
def test_shapes_annotation_widget_edit_existing_shell_anchor_edit_saves_and_reloads_with_hole(qtbot) -> None:
    """Exercise guarded shell-anchor drag through edit-existing save and reload."""
    shapes_name = "hole_regions"
    original_polygon_1, polygon_2 = _polygon_hole_roundtrip_fixture()
    sdata = _make_polygon_hole_roundtrip_sdata(shapes_name=shapes_name)
    viewer = DummyViewer()
    app_state = get_or_create_app_state(viewer)
    app_state.set_sdata(sdata)
    widget = ShapesAnnotation(viewer)
    qtbot.addWidget(widget)

    widget.shapes_combo.setCurrentIndex(_combo_index_for_text(widget.shapes_combo, shapes_name))
    layer = widget._annotation_layer
    assert isinstance(layer, Shapes)
    original_vertices = np.asarray(layer.data[0], dtype=float)
    moved_coordinate = original_vertices[0] + np.asarray([-25.0, 35.0])
    _install_direct_drag_callback_for_annotation_guard(
        widget,
        layer,
        moved_vertex_index=0,
        moved_coordinate=moved_coordinate,
    )

    edited_vertices = _drag_annotation_vertex(layer, vertex_index=0, moved_coordinate=moved_coordinate)

    np.testing.assert_allclose(edited_vertices[[0, 5, 12]], np.repeat(moved_coordinate[None, :], 3, axis=0))
    expected_polygon_1 = napari_polygon_vertices_to_shapely_polygon(edited_vertices)
    assert expected_polygon_1.is_valid
    assert len(expected_polygon_1.interiors) == 1
    assert not expected_polygon_1.equals(original_polygon_1)

    widget.save_shapes_button.click()

    saved = sdata.shapes[shapes_name]
    assert saved.index.name == "region_id"
    assert saved.index.tolist() == ["hole_row", "simple_row"]
    assert saved["label"].tolist() == ["polygon_with_hole", "simple_polygon"]
    np.testing.assert_allclose(saved["score"].to_numpy(), np.asarray([1.25, 2.5]))
    _assert_polygon_hole_geometries_preserved(saved, expected_polygon_1, polygon_2)

    reloaded_viewer = DummyViewer()
    reloaded_layer = get_or_create_app_state(reloaded_viewer).viewer_adapter.ensure_shapes_loaded(
        sdata,
        shapes_name,
        "global",
    ).layer
    reloaded_polygon_1 = napari_polygon_vertices_to_shapely_polygon(reloaded_layer.data[0])

    assert reloaded_polygon_1.equals(saved.geometry.iloc[0])
    assert len(reloaded_polygon_1.interiors) == 1
    assert "Shapes Saved" in _status_text(widget)


def test_shapes_annotation_widget_adopted_native_hole_anchor_edit_saves_and_reloads_with_hole(qtbot) -> None:
    """Exercise guarded hole-anchor drag through native adoption, save, and reload."""
    shapes_name = "native_hole_anchor_roundtrip"
    original_polygon_1, polygon_2 = _polygon_hole_roundtrip_fixture()
    sdata = _make_polygon_hole_roundtrip_sdata(shapes_name="reference_regions")
    native_layer = Shapes(
        [
            shapely_polygon_to_napari_polygon_vertices(original_polygon_1),
            shapely_polygon_to_napari_polygon_vertices(polygon_2),
        ],
        shape_type=["polygon", "polygon"],
        name=shapes_name,
    )
    viewer = DummyViewer()
    app_state = get_or_create_app_state(viewer)
    app_state.set_sdata(sdata)
    widget = ShapesAnnotation(viewer)
    qtbot.addWidget(widget)

    viewer.add_layer(native_layer)
    qtbot.waitUntil(lambda: widget._annotation_layer is native_layer)
    original_vertices = np.asarray(native_layer.data[0], dtype=float)
    moved_coordinate = original_vertices[6] + np.asarray([15.0, -20.0])
    _install_direct_drag_callback_for_annotation_guard(
        widget,
        native_layer,
        moved_vertex_index=6,
        moved_coordinate=moved_coordinate,
    )

    edited_vertices = _drag_annotation_vertex(native_layer, vertex_index=6, moved_coordinate=moved_coordinate)

    np.testing.assert_allclose(edited_vertices[[6, 11]], np.repeat(moved_coordinate[None, :], 2, axis=0))
    expected_polygon_1 = napari_polygon_vertices_to_shapely_polygon(edited_vertices)
    assert expected_polygon_1.is_valid
    assert len(expected_polygon_1.interiors) == 1
    assert not expected_polygon_1.equals(original_polygon_1)

    widget.save_shapes_button.click()

    assert shapes_name in sdata.shapes
    saved = sdata.shapes[shapes_name]
    assert saved.index.name == "instance_id"
    assert saved.index.tolist() == ["__annotation_0", "__annotation_1"]
    assert native_layer.features["instance_id"].tolist() == ["__annotation_0", "__annotation_1"]
    _assert_polygon_hole_geometries_preserved(saved, expected_polygon_1, polygon_2)
    assert widget._annotation_session is not None
    assert widget._annotation_session.mode == "edit_existing"
    assert widget._annotation_session.shapes_name == shapes_name
    assert widget.shapes_combo.currentText() == shapes_name

    reloaded_viewer = DummyViewer()
    reloaded_layer = get_or_create_app_state(reloaded_viewer).viewer_adapter.ensure_shapes_loaded(
        sdata,
        shapes_name,
        "global",
    ).layer
    reloaded_polygon_1 = napari_polygon_vertices_to_shapely_polygon(reloaded_layer.data[0])

    assert reloaded_polygon_1.equals(saved.geometry.iloc[0])
    assert len(reloaded_polygon_1.interiors) == 1
    assert "Shapes Saved" in _status_text(widget)


def test_shapes_annotation_widget_table_linked_edit_warns_without_mutating_table(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    shapes_name = "blobs_polygons"
    table_name = "shapes_annotation_table"
    table = _add_dummy_table_annotating_shapes(sdata_blobs, shapes_name=shapes_name, table_name=table_name)
    original_obs = table.obs.copy(deep=True)
    viewer = DummyViewer()
    app_state = get_or_create_app_state(viewer)
    app_state.set_sdata(sdata_blobs)
    widget = ShapesAnnotation(viewer)
    qtbot.addWidget(widget)

    widget.shapes_combo.setCurrentIndex(_combo_index_for_text(widget.shapes_combo, shapes_name))

    assert widget._annotation_session is not None
    assert widget._annotation_session.table_linked is True
    assert "Linked tables are not updated by Annotation and may go out of sync if rows are added or removed." in _status_text(widget)

    layer = widget._annotation_layer
    assert isinstance(layer, Shapes)
    _add_polygon(layer, offset=100)
    layer.features.loc[len(layer.features) - 1, "index"] = None
    widget.save_shapes_button.click()

    assert sdata_blobs.tables[table_name] is table
    pd.testing.assert_frame_equal(sdata_blobs.tables[table_name].obs, original_obs)
    assert "Linked tables are not updated by Annotation and may go out of sync if rows are added or removed." in _status_text(widget)


def test_shapes_annotation_widget_clears_annotation_state_when_sdata_is_cleared(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    viewer = DummyViewer()
    widget = _create_ready_annotation_widget(qtbot, viewer, sdata_blobs)
    widget.create_layer_button.click()
    layer = viewer.layers[0]

    widget.app_state.clear_sdata()

    assert list(viewer.layers) == []
    assert widget.app_state.viewer_adapter.layer_bindings.get_binding(layer) is None
    assert widget._annotation_layer is None
    assert widget._annotation_shapes_name is None
    assert widget._annotation_coordinate_system is None
    assert widget._annotation_has_been_saved is False
    assert widget.name_edit.isHidden() is True
    assert widget.name_edit.isEnabled() is False
    assert widget.create_layer_button.isEnabled() is False
    assert widget.save_shapes_button.isEnabled() is False
    assert "No SpatialData Loaded" in _status_text(widget)


def test_shapes_annotation_widget_manual_annotation_layer_deletion_clears_state(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    viewer = DummyViewer()
    widget = _create_ready_annotation_widget(qtbot, viewer, sdata_blobs)
    widget.create_layer_button.click()
    layer = viewer.layers[0]

    viewer.layers.remove(layer)

    assert list(viewer.layers) == []
    assert widget.app_state.viewer_adapter.layer_bindings.get_binding(layer) is None
    assert widget._annotation_layer is None
    assert widget._annotation_shapes_name is None
    assert widget._annotation_coordinate_system is None
    assert widget._annotation_has_been_saved is False
    assert widget._selected_shapes_target == shapes_annotation_widget_module._ShapesAnnotationTarget.create_new()
    assert widget.shapes_combo.currentText() == "Create shapes..."
    assert widget.name_edit.isEnabled() is True
    assert widget.create_layer_button.isEnabled() is True
    assert widget.save_shapes_button.isEnabled() is False
    assert "Ready" in _status_text(widget)


def test_shapes_annotation_widget_manual_existing_layer_deletion_resets_selector_and_can_reopen(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    viewer = DummyViewer()
    app_state = get_or_create_app_state(viewer)
    app_state.set_sdata(sdata_blobs)
    widget = ShapesAnnotation(viewer)
    qtbot.addWidget(widget)
    existing_shapes_name = "blobs_polygons"
    widget.shapes_combo.setCurrentIndex(_combo_index_for_text(widget.shapes_combo, existing_shapes_name))
    removed_layer = widget._annotation_layer

    assert removed_layer is not None
    assert widget._annotation_session is not None
    assert widget._annotation_session.shapes_name == existing_shapes_name

    viewer.layers.remove(removed_layer)

    assert widget.app_state.viewer_adapter.layer_bindings.get_binding(removed_layer) is None
    assert widget._annotation_layer is None
    assert widget._annotation_session is None
    assert widget._selected_shapes_target == shapes_annotation_widget_module._ShapesAnnotationTarget.create_new()
    assert widget.shapes_combo.currentText() == "Create shapes..."
    assert _combo_index_for_text(widget.shapes_combo, existing_shapes_name) >= 0
    assert widget.save_shapes_button.isEnabled() is False

    widget.shapes_combo.setCurrentIndex(_combo_index_for_text(widget.shapes_combo, existing_shapes_name))

    assert widget._annotation_layer is not None
    assert widget._annotation_layer is not removed_layer
    assert widget._annotation_session is not None
    assert widget._annotation_session.shapes_name == existing_shapes_name
    assert widget.save_shapes_button.isEnabled() is True


def test_shapes_annotation_widget_removal_listener_defensively_unregisters_annotation_layer(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    viewer = DummyViewer()
    widget = _create_ready_annotation_widget(qtbot, viewer, sdata_blobs)
    widget.create_layer_button.click()
    layer = viewer.layers[0]

    widget._on_viewer_layer_removed(SimpleNamespace(value=layer))

    assert widget.app_state.viewer_adapter.layer_bindings.get_binding(layer) is None
    assert widget._annotation_layer is None
    assert widget._annotation_shapes_name is None
    assert widget._annotation_coordinate_system is None
    assert widget._annotation_has_been_saved is False
    assert widget.name_edit.isEnabled() is True
    assert widget.create_layer_button.isEnabled() is True
    assert widget.save_shapes_button.isEnabled() is False


def test_shapes_annotation_widget_ignores_unrelated_layer_removal(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    viewer = DummyViewer()
    widget = _create_ready_annotation_widget(qtbot, viewer, sdata_blobs)
    widget.create_layer_button.click()
    annotation_layer = viewer.layers[0]
    unrelated_layer = Shapes(
        [np.asarray([(0, 0), (0, 1), (1, 1)], dtype=float)],
        shape_type="polygon",
        name="unrelated",
    )
    viewer.add_layer(unrelated_layer)

    viewer.layers.remove(unrelated_layer)

    assert list(viewer.layers) == [annotation_layer]
    assert widget.app_state.viewer_adapter.layer_bindings.get_binding(annotation_layer) is not None
    assert widget._annotation_layer is annotation_layer
    assert widget._annotation_shapes_name == "new_regions"
    assert widget._annotation_coordinate_system == "global"
    assert widget.name_edit.isEnabled() is False
    assert widget.create_layer_button.isEnabled() is False
    assert widget.save_shapes_button.isEnabled() is True


def test_shapes_annotation_widget_adopts_native_empty_shapes_layer(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    viewer = DummyViewer()
    app_state = get_or_create_app_state(viewer)
    app_state.set_sdata(sdata_blobs)
    widget = ShapesAnnotation(viewer)
    qtbot.addWidget(widget)
    native_layer = Shapes(
        [],
        shape_type="polygon",
        name="native_shapes",
        affine=np.asarray([[1.0, 0.0, 5.0], [0.0, 1.0, 7.0], [0.0, 0.0, 1.0]]),
    )

    viewer.add_layer(native_layer)

    qtbot.waitUntil(lambda: widget._annotation_layer is native_layer)
    binding = widget.app_state.viewer_adapter.layer_bindings.get_binding(native_layer)
    assert isinstance(binding, ShapesLayerBinding)
    assert binding.element_name == "native_shapes"
    assert binding.coordinate_system == "global"
    assert binding.shapes_role == "primary"
    assert binding.shapes_rendering_mode == "shapes"
    assert binding.source_shapes_index_feature_name == "instance_id"
    assert widget._annotation_session is not None
    assert widget._annotation_session.mode == "create_new"
    assert widget._annotation_session.shapes_name == "native_shapes"
    assert widget._annotation_edit_guard.layer is native_layer
    assert "_drag_modes" in vars(native_layer)
    assert native_layer._drag_modes[Mode.DIRECT] is widget._annotation_edit_guard._wrapped_direct_callback
    assert widget._selected_shapes_target == shapes_annotation_widget_module._ShapesAnnotationTarget.create_new()
    assert widget.shapes_combo.currentText() == "Create shapes..."
    assert widget.name_edit.text() == "native_shapes"
    assert native_layer.name == "native_shapes"
    np.testing.assert_allclose(native_layer.affine.affine_matrix, np.eye(3))
    assert native_layer.current_edge_width == 1
    np.testing.assert_allclose(to_rgba(native_layer.current_edge_color), to_rgba("#00FFFF"))
    np.testing.assert_allclose(to_rgba(native_layer.current_face_color), to_rgba("#00000000"))
    assert native_layer.opacity == 0.8
    assert hasattr(native_layer, _SHAPES_EDGE_WIDTH_SYNC_CALLBACK_ATTR)
    assert hasattr(native_layer, _SHAPES_EDGE_COLOR_SYNC_CALLBACK_ATTR)
    assert widget.create_layer_button.isEnabled() is False
    assert widget.save_shapes_button.isEnabled() is True


def test_shapes_annotation_widget_saves_adopted_native_nonempty_shapes_layer(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    viewer = DummyViewer()
    app_state = get_or_create_app_state(viewer)
    app_state.set_sdata(sdata_blobs)
    widget = ShapesAnnotation(viewer)
    qtbot.addWidget(widget)
    native_layer = _native_polygon_layer("native_import")

    viewer.add_layer(native_layer)
    qtbot.waitUntil(lambda: widget._annotation_layer is native_layer)
    np.testing.assert_allclose(native_layer.edge_color[0], to_rgba("#00FFFF"))
    np.testing.assert_allclose(native_layer.face_color[0], to_rgba("#00000000"))
    np.testing.assert_allclose(to_rgba(native_layer.current_edge_color), to_rgba("#00FFFF"))
    np.testing.assert_allclose(to_rgba(native_layer.current_face_color), to_rgba("#00000000"))
    widget.save_shapes_button.click()

    assert "native_import" in sdata_blobs.shapes
    assert sdata_blobs.shapes["native_import"].index.tolist() == ["__annotation_0"]
    assert native_layer.features["instance_id"].tolist() == ["__annotation_0"]
    assert widget._annotation_session is not None
    assert widget._annotation_session.mode == "edit_existing"
    assert widget._annotation_session.shapes_name == "native_import"
    assert widget.shapes_combo.currentText() == "native_import"
    assert widget.name_edit.text() == ""


def test_shapes_annotation_widget_saves_native_csv_layer_with_polygon_hole(
    qtbot,
    tmp_path,
) -> None:
    """Save a napari-native CSV Shapes import with a hole through the widget."""
    polygon_1, polygon_2 = _polygon_hole_roundtrip_fixture()
    sdata = _make_polygon_hole_roundtrip_sdata(shapes_name="reference_regions")
    output_path = tmp_path / "native_hole_import.csv"
    napari_write_shapes(
        str(output_path),
        [
            shapely_polygon_to_napari_polygon_vertices(polygon_1),
            shapely_polygon_to_napari_polygon_vertices(polygon_2),
        ],
        {"shape_type": ["polygon", "polygon"]},
    )
    loaded = csv_to_layer_data(str(output_path), require_type="shapes")
    assert loaded is not None
    data, meta, _layer_type = loaded
    native_layer = Shapes(data, **meta, name="native_hole_import")
    viewer = DummyViewer()
    app_state = get_or_create_app_state(viewer)
    app_state.set_sdata(sdata)
    widget = ShapesAnnotation(viewer)
    qtbot.addWidget(widget)

    viewer.add_layer(native_layer)
    qtbot.waitUntil(lambda: widget._annotation_layer is native_layer)
    widget.save_shapes_button.click()

    assert "native_hole_import" in sdata.shapes
    saved = sdata.shapes["native_hole_import"]
    assert saved.index.name == "instance_id"
    assert saved.index.tolist() == ["__annotation_0", "__annotation_1"]
    assert native_layer.features["instance_id"].tolist() == ["__annotation_0", "__annotation_1"]
    _assert_polygon_hole_geometries_preserved(saved, polygon_1, polygon_2)
    assert widget._annotation_session is not None
    assert widget._annotation_session.mode == "edit_existing"
    assert widget._annotation_session.shapes_name == "native_hole_import"
    assert widget.shapes_combo.currentText() == "native_hole_import"
    assert "Shapes Saved" in _status_text(widget)


def test_shapes_annotation_widget_saves_reloads_adopted_translated_native_shapes_layer(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    viewer = DummyViewer()
    app_state = get_or_create_app_state(viewer)
    app_state.set_sdata(sdata_blobs)
    widget = ShapesAnnotation(viewer)
    qtbot.addWidget(widget)
    native_layer = _native_polygon_layer(
        "native_translated",
        affine=np.asarray([[1.0, 0.0, 5.0], [0.0, 1.0, 7.0], [0.0, 0.0, 1.0]]),
    )
    expected_layer_vertices = np.asarray(
        [
            [5.0, 7.0],
            [5.0, 9.0],
            [7.0, 9.0],
            [7.0, 7.0],
        ]
    )
    expected_geometry_coords = np.asarray(
        [
            [7.0, 5.0],
            [9.0, 5.0],
            [9.0, 7.0],
            [7.0, 7.0],
            [7.0, 5.0],
        ]
    )
    # Reloading goes through the saved Shapely polygon exterior, which is a
    # closed ring and therefore repeats the first vertex at the end.
    expected_reloaded_vertices = np.asarray(
        [
            [5.0, 7.0],
            [5.0, 9.0],
            [7.0, 9.0],
            [7.0, 7.0],
            [5.0, 7.0],
        ]
    )

    viewer.add_layer(native_layer)

    qtbot.waitUntil(lambda: widget._annotation_layer is native_layer)
    np.testing.assert_allclose(native_layer.affine.affine_matrix, np.eye(3))
    np.testing.assert_allclose(native_layer.data[0], expected_layer_vertices)

    widget.save_shapes_button.click()

    assert "native_translated" in sdata_blobs.shapes
    np.testing.assert_allclose(
        np.asarray(sdata_blobs.shapes["native_translated"].geometry.iloc[0].exterior.coords),
        expected_geometry_coords,
    )

    viewer.layers.remove(native_layer)

    reloaded = widget.app_state.viewer_adapter.ensure_shapes_loaded(sdata_blobs, "native_translated", "global").layer

    assert isinstance(reloaded, Shapes)
    assert reloaded is not native_layer
    np.testing.assert_allclose(reloaded.data[0], expected_reloaded_vertices)


def test_shapes_annotation_widget_native_name_falls_back_and_suffixes_collision(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    sdata_blobs.shapes["new_shapes"] = sdata_blobs.shapes["blobs_polygons"].copy()
    sdata_blobs.shapes["New_Shapes_1"] = sdata_blobs.shapes["blobs_polygons"].copy()
    viewer = DummyViewer()
    app_state = get_or_create_app_state(viewer)
    app_state.set_sdata(sdata_blobs)
    widget = ShapesAnnotation(viewer)
    qtbot.addWidget(widget)
    native_layer = Shapes([], shape_type="polygon", name="bad/name")

    viewer.add_layer(native_layer)

    qtbot.waitUntil(lambda: widget._annotation_layer is native_layer)
    binding = widget.app_state.viewer_adapter.layer_bindings.get_binding(native_layer)
    assert isinstance(binding, ShapesLayerBinding)
    assert binding.element_name == "new_shapes_2"
    assert widget.name_edit.text() == "new_shapes_2"
    assert native_layer.name == "new_shapes_2"


def test_shapes_annotation_widget_deferred_native_adoption_ignores_harpy_loaded_shapes(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    viewer = DummyViewer()
    app_state = get_or_create_app_state(viewer)
    app_state.set_sdata(sdata_blobs)
    widget = ShapesAnnotation(viewer)
    qtbot.addWidget(widget)

    result = widget.app_state.viewer_adapter.ensure_shapes_loaded(sdata_blobs, "blobs_polygons", "global")
    qtbot.wait(10)

    assert result.created is True
    assert widget.app_state.viewer_adapter.layer_bindings.get_binding(result.layer) is not None
    assert widget._annotation_layer is None
    assert widget._annotation_session is None
    assert widget._selected_shapes_target == shapes_annotation_widget_module._ShapesAnnotationTarget.create_new()


def test_shapes_annotation_widget_native_adoption_cancel_keeps_dirty_session_unbound(
    qtbot,
    monkeypatch,
    sdata_blobs: SpatialData,
) -> None:
    viewer = DummyViewer()
    widget = _create_ready_annotation_widget(qtbot, viewer, sdata_blobs)
    widget.create_layer_button.click()
    annotation_layer = viewer.layers[0]
    _add_polygon(annotation_layer)
    native_layer = Shapes([], shape_type="polygon", name="native_shapes")
    confirm_calls: list[str] = []

    def cancel_discard(*, context: str) -> bool:
        confirm_calls.append(context)
        return False

    monkeypatch.setattr(widget, "_confirm_discard_annotation_layer", cancel_discard)

    viewer.add_layer(native_layer)

    qtbot.waitUntil(lambda: bool(confirm_calls))
    assert confirm_calls == ["target"]
    assert widget._annotation_layer is annotation_layer
    assert widget._annotation_session is not None
    assert widget._annotation_session.shapes_name == "new_regions"
    assert widget.app_state.viewer_adapter.layer_bindings.get_binding(native_layer) is None
    assert native_layer in viewer.layers


def test_shapes_annotation_widget_native_adoption_confirm_discards_dirty_session(
    qtbot,
    monkeypatch,
    sdata_blobs: SpatialData,
) -> None:
    viewer = DummyViewer()
    widget = _create_ready_annotation_widget(qtbot, viewer, sdata_blobs)
    widget.create_layer_button.click()
    annotation_layer = viewer.layers[0]
    _add_polygon(annotation_layer)
    native_layer = Shapes([], shape_type="polygon", name="native_shapes")
    confirm_calls: list[str] = []

    def confirm_discard(*, context: str) -> bool:
        confirm_calls.append(context)
        return True

    monkeypatch.setattr(widget, "_confirm_discard_annotation_layer", confirm_discard)

    viewer.add_layer(native_layer)

    qtbot.waitUntil(lambda: widget._annotation_layer is native_layer)
    assert confirm_calls == ["target"]
    assert annotation_layer not in viewer.layers
    assert widget.app_state.viewer_adapter.layer_bindings.get_binding(annotation_layer) is None
    assert widget.app_state.viewer_adapter.layer_bindings.get_binding(native_layer) is not None
    assert widget._annotation_session is not None
    assert widget._annotation_session.mode == "create_new"
    assert widget._annotation_session.shapes_name == "native_shapes"
    assert widget.name_edit.text() == "native_shapes"


def test_shapes_annotation_widget_coordinate_discard_guard_avoids_duplicate_cleanup(
    qtbot,
    monkeypatch,
    sdata_blobs: SpatialData,
) -> None:
    _patch_coordinate_system_names(monkeypatch, ["global", "local"])
    viewer = DummyViewer()
    widget = _create_ready_annotation_widget(qtbot, viewer, sdata_blobs)
    widget.create_layer_button.click()
    _add_polygon(viewer.layers[0])
    monkeypatch.setattr(widget, "_confirm_discard_annotation_layer", lambda *, context: True)
    remove_guard_values: list[bool] = []
    clear_call_count = 0
    original_remove_annotation_layer = widget._remove_annotation_layer
    original_clear_annotation_state = widget._clear_annotation_state

    def remove_annotation_layer() -> None:
        remove_guard_values.append(widget._is_handling_annotation_layer_removal)
        original_remove_annotation_layer()

    def clear_annotation_state() -> None:
        nonlocal clear_call_count
        clear_call_count += 1
        original_clear_annotation_state()

    monkeypatch.setattr(widget, "_remove_annotation_layer", remove_annotation_layer)
    monkeypatch.setattr(widget, "_clear_annotation_state", clear_annotation_state)

    widget.coordinate_system_combo.setCurrentIndex(1)

    assert remove_guard_values == [True]
    assert clear_call_count == 1
    assert widget._is_handling_annotation_layer_removal is False
    assert widget.app_state.coordinate_system == "local"
    assert list(viewer.layers) == []


def test_shapes_annotation_widget_discard_saved_annotation_layer_reloads_clean_primary_layer(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    viewer = DummyViewer()
    widget = _create_ready_annotation_widget(qtbot, viewer, sdata_blobs)
    widget.create_layer_button.click()
    dirty_layer = viewer.layers[0]
    _add_polygon(dirty_layer)
    widget.save_shapes_button.click()
    assert widget._annotation_session is not None
    assert widget._annotation_session.reload_on_discard is True

    _add_polygon(dirty_layer, offset=10)
    assert len(dirty_layer.data) == 2

    widget._discard_annotation_layer()

    assert dirty_layer not in viewer.layers
    assert widget.app_state.viewer_adapter.layer_bindings.get_binding(dirty_layer) is None
    assert len(viewer.layers) == 1
    clean_layer = viewer.layers[0]
    assert clean_layer is not dirty_layer
    assert isinstance(clean_layer, Shapes)
    assert clean_layer.name == "new_regions"
    assert len(clean_layer.data) == 1
    binding = widget.app_state.viewer_adapter.layer_bindings.get_binding(clean_layer)
    assert isinstance(binding, ShapesLayerBinding)
    assert binding.element_name == "new_regions"
    assert binding.coordinate_system == "global"
    assert binding.shapes_role == "primary"
    assert widget._annotation_layer is None
    assert widget._annotation_shapes_name is None
    assert widget._annotation_coordinate_system is None
    assert widget._annotation_has_been_saved is False
    assert widget._annotation_session is None
    assert widget.name_edit.isHidden() is True
    assert widget.name_edit.isEnabled() is False


def test_shapes_annotation_widget_backed_edit_existing_discard_reloads_clean_primary_layer(
    qtbot,
    backed_sdata_blobs: SpatialData,
) -> None:
    viewer = DummyViewer()
    app_state = get_or_create_app_state(viewer)
    app_state.set_sdata(backed_sdata_blobs)
    widget = ShapesAnnotation(viewer)
    qtbot.addWidget(widget)
    shapes_name = "blobs_polygons"

    widget.shapes_combo.setCurrentIndex(_combo_index_for_text(widget.shapes_combo, shapes_name))
    dirty_layer = widget._annotation_layer
    assert isinstance(dirty_layer, Shapes)
    initial_row_count = len(dirty_layer.data)

    _add_polygon(dirty_layer, offset=100)
    assert len(dirty_layer.data) == initial_row_count + 1

    widget._discard_annotation_layer()

    assert dirty_layer not in viewer.layers
    assert app_state.viewer_adapter.layer_bindings.get_binding(dirty_layer) is None
    assert len(viewer.layers) == 1
    clean_layer = viewer.layers[0]
    assert clean_layer is not dirty_layer
    assert isinstance(clean_layer, Shapes)
    assert clean_layer.name == shapes_name
    assert len(clean_layer.data) == initial_row_count
    binding = app_state.viewer_adapter.layer_bindings.get_binding(clean_layer)
    assert isinstance(binding, ShapesLayerBinding)
    assert binding.element_name == shapes_name
    assert binding.coordinate_system == "global"
    assert binding.shapes_role == "primary"
    assert widget._annotation_layer is None
    assert widget._annotation_session is None

    reread = read_zarr(backed_sdata_blobs.path)
    assert len(reread.shapes[shapes_name]) == initial_row_count


def test_shapes_annotation_widget_save_calls_core_with_locked_request_and_reports_success(
    qtbot,
    monkeypatch,
    sdata_blobs: SpatialData,
) -> None:
    viewer = DummyViewer()
    widget = _create_ready_annotation_widget(qtbot, viewer, sdata_blobs)
    widget.create_layer_button.click()
    layer = viewer.layers[0]
    captured_requests = []
    captured_layers = []
    emitted_events: list[object] = []
    widget.app_state.shapes_element_written.connect(emitted_events.append)

    def fake_create_shapes_element(request, napari_layer):
        captured_requests.append(request)
        captured_layers.append(napari_layer)
        request.sdata.shapes[request.shapes_name] = request.sdata.shapes["blobs_polygons"].copy()
        return AnnotateShapesElementResult(
            shapes_name=request.shapes_name,
            coordinate_system=request.coordinate_system,
            row_count=3,
        )

    monkeypatch.setattr(
        shapes_annotation_widget_module,
        "create_shapes_element_from_napari_shapes_layer",
        fake_create_shapes_element,
    )
    widget._selected_coordinate_system = "local"

    widget.save_shapes_button.click()

    assert captured_layers == [layer]
    assert len(captured_requests) == 1
    request = captured_requests[0]
    assert request.sdata is sdata_blobs
    assert request.shapes_name == "new_regions"
    assert request.coordinate_system == "global"
    assert request.overwrite is False
    assert request.index_name == "instance_id"
    assert request.index_prefix == "__annotation"
    assert widget._annotation_has_been_saved is True
    assert widget._annotation_session is not None
    assert widget._annotation_session.mode == "edit_existing"
    assert widget._annotation_session.reload_on_discard is True
    assert widget.save_shapes_button.isEnabled() is True
    assert emitted_events == [
        ShapesElementWrittenEvent(
            sdata=sdata_blobs,
            shapes_name="new_regions",
            coordinate_system="global",
            source="shapes_annotation_widget",
        )
    ]
    status = _status_text(widget)
    assert "Shapes Saved" in status
    assert 'Saved "new_regions" with 3 shape(s) in coordinate system "global".' in status


def test_shapes_annotation_widget_repeated_save_uses_edit_helper_after_create_success(
    qtbot,
    monkeypatch,
    sdata_blobs: SpatialData,
) -> None:
    viewer = DummyViewer()
    widget = _create_ready_annotation_widget(qtbot, viewer, sdata_blobs)
    widget.create_layer_button.click()
    _add_polygon(viewer.layers[0])
    original_create_shapes_element = shapes_annotation_widget_module.create_shapes_element_from_napari_shapes_layer
    original_edit_shapes_element = shapes_annotation_widget_module.edit_shapes_element_from_napari_shapes_layer
    create_overwrites: list[bool] = []
    edit_requests = []
    emitted_events: list[object] = []
    widget.app_state.shapes_element_written.connect(emitted_events.append)

    def fake_create_shapes_element(request, napari_layer):
        create_overwrites.append(request.overwrite)
        return original_create_shapes_element(request, napari_layer)

    def fake_edit_shapes_element(request, napari_layer):
        edit_requests.append(request)
        return original_edit_shapes_element(request, napari_layer)

    monkeypatch.setattr(
        shapes_annotation_widget_module,
        "create_shapes_element_from_napari_shapes_layer",
        fake_create_shapes_element,
    )
    monkeypatch.setattr(
        shapes_annotation_widget_module,
        "edit_shapes_element_from_napari_shapes_layer",
        fake_edit_shapes_element,
    )

    widget.save_shapes_button.click()
    widget.save_shapes_button.click()

    assert create_overwrites == [False]
    assert len(edit_requests) == 1
    assert edit_requests[0].sdata is sdata_blobs
    assert edit_requests[0].shapes_name == "new_regions"
    assert edit_requests[0].coordinate_system == "global"
    assert edit_requests[0].source_shapes_index_feature_name == "instance_id"
    assert widget._annotation_has_been_saved is True
    assert widget._annotation_session is not None
    assert widget._annotation_session.mode == "edit_existing"
    assert widget._annotation_session.reload_on_discard is True
    assert emitted_events == [
        ShapesElementWrittenEvent(
            sdata=sdata_blobs,
            shapes_name="new_regions",
            coordinate_system="global",
            source="shapes_annotation_widget",
        ),
        ShapesElementWrittenEvent(
            sdata=sdata_blobs,
            shapes_name="new_regions",
            coordinate_system="global",
            source="shapes_annotation_widget",
        ),
    ]


def test_shapes_annotation_widget_failed_first_save_keeps_later_overwrite_false(
    qtbot,
    monkeypatch,
    sdata_blobs: SpatialData,
) -> None:
    viewer = DummyViewer()
    widget = _create_ready_annotation_widget(qtbot, viewer, sdata_blobs)
    widget.create_layer_button.click()
    overwrites: list[bool] = []

    def fake_create_shapes_element(request, napari_layer):
        del napari_layer
        overwrites.append(request.overwrite)
        if len(overwrites) == 1:
            raise ValueError("same-name element appeared externally")
        request.sdata.shapes[request.shapes_name] = request.sdata.shapes["blobs_polygons"].copy()
        return AnnotateShapesElementResult(
            shapes_name=request.shapes_name,
            coordinate_system=request.coordinate_system,
            row_count=1,
        )

    monkeypatch.setattr(
        shapes_annotation_widget_module,
        "create_shapes_element_from_napari_shapes_layer",
        fake_create_shapes_element,
    )

    widget.save_shapes_button.click()

    assert overwrites == [False]
    assert widget._annotation_has_been_saved is False
    assert widget.save_shapes_button.isEnabled() is True
    assert "same-name element appeared externally" in _status_text(widget)

    widget.save_shapes_button.click()

    assert overwrites == [False, False]
    assert widget._annotation_has_been_saved is True


def test_shapes_annotation_widget_empty_layer_save_error_is_feedback(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    viewer = DummyViewer()
    widget = _create_ready_annotation_widget(qtbot, viewer, sdata_blobs)
    widget.create_layer_button.click()

    widget.save_shapes_button.click()

    assert "new_regions" not in sdata_blobs.shapes
    assert widget._annotation_has_been_saved is False
    assert widget.save_shapes_button.isEnabled() is True
    assert "Draw at least one supported shape before saving" in _status_text(widget)


def test_shapes_annotation_widget_save_writes_real_shapes_element(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    viewer = DummyViewer()
    widget = _create_ready_annotation_widget(qtbot, viewer, sdata_blobs)
    widget.create_layer_button.click()
    layer = viewer.layers[0]
    _add_polygon(layer)

    widget.save_shapes_button.click()

    assert "new_regions" in sdata_blobs.shapes
    assert sdata_blobs.shapes["new_regions"].index.name == "instance_id"
    assert sdata_blobs.shapes["new_regions"].index.tolist() == ["__annotation_0"]
    assert layer.features["instance_id"].tolist() == ["__annotation_0"]
    binding = widget.app_state.viewer_adapter.layer_bindings.get_binding(layer)
    assert isinstance(binding, ShapesLayerBinding)
    assert list(binding.source_row_id_by_rendered_row) == [0]
    assert binding.source_shapes_index_feature_name == "instance_id"
    assert list(viewer.layers) == [layer]
    assert widget._annotation_layer is layer
    assert widget._annotation_has_been_saved is True
    assert widget._annotation_session is not None
    assert widget._annotation_session.mode == "edit_existing"
    assert widget._annotation_session.reload_on_discard is True
    assert widget._annotation_session.source_geodataframe is not sdata_blobs.shapes["new_regions"]
    assert widget._annotation_session.source_geodataframe is not None
    assert widget._annotation_session.source_geodataframe.index.tolist() == ["__annotation_0"]
    assert widget._selected_shapes_target == shapes_annotation_widget_module._ShapesAnnotationTarget.edit_existing(
        "new_regions"
    )
    assert widget.shapes_combo.currentText() == "new_regions"
    assert widget.name_edit.isHidden() is True
    assert widget.name_edit.text() == ""
    assert "Shapes Saved" in _status_text(widget)

    widget.shapes_combo.setCurrentIndex(_combo_index_for_text(widget.shapes_combo, "Create shapes..."))

    assert widget.name_edit.isHidden() is False
    assert widget.name_edit.text() == ""


def test_shapes_annotation_widget_save_syncs_binding_without_primary_registration_event(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    viewer = DummyViewer()
    widget = _create_ready_annotation_widget(qtbot, viewer, sdata_blobs)
    emitted_bindings: list[object] = []
    widget.app_state.viewer_adapter.primary_shapes_layer_registered.connect(emitted_bindings.append)
    widget.create_layer_button.click()
    layer = viewer.layers[0]
    emitted_bindings.clear()
    _add_polygon(layer)

    widget.save_shapes_button.click()

    assert emitted_bindings == []
    binding = widget.app_state.viewer_adapter.layer_bindings.get_binding(layer)
    assert isinstance(binding, ShapesLayerBinding)
    assert list(binding.source_row_id_by_rendered_row) == [0]


def test_shapes_annotation_widget_saved_create_new_layer_can_be_reopened_after_target_switch(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    viewer = DummyViewer()
    widget = _create_ready_annotation_widget(qtbot, viewer, sdata_blobs)
    widget.create_layer_button.click()
    layer = viewer.layers[0]
    _add_polygon(layer)
    widget.save_shapes_button.click()
    _add_polygon(layer, offset=10)
    widget.save_shapes_button.click()

    binding = widget.app_state.viewer_adapter.layer_bindings.get_binding(layer)
    assert isinstance(binding, ShapesLayerBinding)
    assert list(binding.source_row_id_by_rendered_row) == [0, 1]

    widget.shapes_combo.setCurrentIndex(_combo_index_for_text(widget.shapes_combo, "blobs_polygons"))
    assert widget._annotation_session is not None
    assert widget._annotation_session.shapes_name == "blobs_polygons"

    widget.shapes_combo.setCurrentIndex(_combo_index_for_text(widget.shapes_combo, "new_regions"))

    assert widget._annotation_layer is layer
    assert widget._annotation_session is not None
    assert widget._annotation_session.shapes_name == "new_regions"
    assert widget.save_shapes_button.isEnabled() is True
    assert "Could Not Open Shapes" not in _status_text(widget)


def test_shapes_annotation_widget_keeps_ownership_when_viewer_adds_saved_primary_shapes(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    viewer = DummyViewer()
    widget = _create_ready_annotation_widget(qtbot, viewer, sdata_blobs)
    widget.create_layer_button.click()
    layer = viewer.layers[0]
    _add_polygon(layer)
    widget.save_shapes_button.click()

    result = widget.app_state.viewer_adapter.ensure_shapes_loaded(sdata_blobs, "new_regions", "global")

    assert result.layer is layer
    assert result.created is False
    assert list(viewer.layers) == [layer]
    assert widget._annotation_layer is layer
    assert widget._annotation_shapes_name == "new_regions"
    assert widget._annotation_coordinate_system == "global"
    assert widget._annotation_has_been_saved is True
    assert widget._annotation_layer_binding_matches()
    assert widget.save_shapes_button.isEnabled() is True


def test_shapes_annotation_widget_binding_mismatch_disables_save_without_calling_core(
    qtbot,
    monkeypatch,
    sdata_blobs: SpatialData,
) -> None:
    viewer = DummyViewer()
    widget = _create_ready_annotation_widget(qtbot, viewer, sdata_blobs)
    widget.create_layer_button.click()
    layer = viewer.layers[0]
    widget.app_state.viewer_adapter.unregister_layer(layer)

    def fail_if_called(request, napari_layer):
        del request, napari_layer
        raise AssertionError("binding mismatch should not call the core save helper")

    monkeypatch.setattr(
        shapes_annotation_widget_module,
        "create_shapes_element_from_napari_shapes_layer",
        fail_if_called,
    )

    widget.save_shapes_button.click()

    assert widget.save_shapes_button.isEnabled() is False
    assert "no longer registered as the widget-owned primary shapes layer" in _status_text(widget)
