from __future__ import annotations

from collections.abc import Callable
from html import unescape
from types import SimpleNamespace

import numpy as np
from matplotlib.colors import to_rgba
from napari.layers import Shapes
from qtpy.QtWidgets import QComboBox, QLabel
from spatialdata import SpatialData

import napari_harpy._app_state as app_state_module
import napari_harpy.widgets.shapes_annotation.widget as shapes_annotation_widget_module
from napari_harpy._app_state import get_or_create_app_state
from napari_harpy.viewer.adapter import ShapesLayerBinding
from napari_harpy.viewer.shapes_styling import (
    _SHAPES_EDGE_COLOR_SYNC_CALLBACK_ATTR,
    _SHAPES_EDGE_WIDTH_SYNC_CALLBACK_ATTR,
)
from napari_harpy.widgets import ShapesAnnotation as LazyShapesAnnotation
from napari_harpy.widgets.shapes_annotation.widget import ShapesAnnotation


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


def _status_text(widget: ShapesAnnotation) -> str:
    return unescape(widget.status_label.text())


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
    assert widget.coordinate_system_combo.count() == 0
    assert widget.coordinate_system_combo.isEnabled() is False
    assert widget.create_layer_button.isEnabled() is False
    assert widget.save_shapes_button.isEnabled() is False
    assert "No SpatialData Loaded" in _status_text(widget)


def test_shapes_annotation_widget_lazy_export() -> None:
    assert LazyShapesAnnotation is ShapesAnnotation


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
    assert widget.create_layer_button.isEnabled() is False
    assert widget.save_shapes_button.isEnabled() is False
    assert "Shapes element name must not be empty" in _status_text(widget)


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
    assert viewer.layers.selection.active is layer

    assert widget.selected_shapes_name == "new_regions"
    assert widget._annotation_layer is layer
    assert widget._annotation_shapes_name == "new_regions"
    assert widget._annotation_coordinate_system == "global"
    assert widget.name_edit.isEnabled() is False
    assert widget.create_layer_button.isEnabled() is False
    assert widget.save_shapes_button.isEnabled() is False
    assert "Annotation Layer Created" in _status_text(widget)


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
    monkeypatch.setattr(widget, "_confirm_discard_annotation_layer", lambda: False)

    widget.coordinate_system_combo.setCurrentIndex(1)

    assert widget.app_state.coordinate_system == "global"
    assert widget.coordinate_system_combo.currentText() == "global"
    assert widget.selected_coordinate_system == "global"
    assert list(viewer.layers) == [layer]
    assert widget.app_state.viewer_adapter.layer_bindings.get_binding(layer) is not None
    assert widget._annotation_layer is layer
    assert widget.name_edit.isEnabled() is False
    assert widget.create_layer_button.isEnabled() is False
    assert widget.save_shapes_button.isEnabled() is False


def test_shapes_annotation_widget_confirming_coordinate_change_discards_annotation_layer(
    qtbot,
    monkeypatch,
    sdata_blobs: SpatialData,
) -> None:
    _patch_coordinate_system_names(monkeypatch, ["global", "local"])
    viewer = DummyViewer()
    widget = _create_ready_annotation_widget(qtbot, viewer, sdata_blobs)
    widget.create_layer_button.click()
    layer = viewer.layers[0]
    monkeypatch.setattr(widget, "_confirm_discard_annotation_layer", lambda: True)

    widget.coordinate_system_combo.setCurrentIndex(1)

    assert widget.app_state.coordinate_system == "local"
    assert widget.coordinate_system_combo.currentText() == "local"
    assert widget.selected_coordinate_system == "local"
    assert list(viewer.layers) == []
    assert widget.app_state.viewer_adapter.layer_bindings.get_binding(layer) is None
    assert widget._annotation_layer is None
    assert widget._annotation_shapes_name is None
    assert widget._annotation_coordinate_system is None
    assert widget.name_edit.isEnabled() is True
    assert widget.create_layer_button.isEnabled() is True
    assert widget.save_shapes_button.isEnabled() is False
    assert "Ready" in _status_text(widget)


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
    assert widget.name_edit.isEnabled() is True
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
    assert widget.name_edit.isEnabled() is True
    assert widget.create_layer_button.isEnabled() is True
    assert widget.save_shapes_button.isEnabled() is False
    assert "Ready" in _status_text(widget)


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
    assert widget.save_shapes_button.isEnabled() is False


def test_shapes_annotation_widget_coordinate_discard_guard_avoids_duplicate_cleanup(
    qtbot,
    monkeypatch,
    sdata_blobs: SpatialData,
) -> None:
    _patch_coordinate_system_names(monkeypatch, ["global", "local"])
    viewer = DummyViewer()
    widget = _create_ready_annotation_widget(qtbot, viewer, sdata_blobs)
    widget.create_layer_button.click()
    monkeypatch.setattr(widget, "_confirm_discard_annotation_layer", lambda: True)
    remove_guard_values: list[bool] = []
    clear_call_count = 0
    original_remove_annotation_layer = widget._remove_annotation_layer
    original_clear_annotation_state = widget._clear_annotation_state

    def remove_annotation_layer() -> None:
        remove_guard_values.append(widget._is_handling_coordinate_system_change)
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
    assert widget._is_handling_coordinate_system_change is False
    assert widget.app_state.coordinate_system == "local"
    assert list(viewer.layers) == []
