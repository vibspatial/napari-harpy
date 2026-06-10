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
from napari_harpy._app_state import ShapesElementWrittenEvent, get_or_create_app_state
from napari_harpy.core.shapes_annotation import CreateShapesElementResult
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


def _tooltip_text(widget: ShapesAnnotation) -> str:
    return unescape(widget.status_label.toolTip()).replace("&#8203;", "").replace("\u200b", "")


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
        return CreateShapesElementResult(
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
    assert widget._annotation_shapes_name == "new_regions"
    assert widget._annotation_coordinate_system == "global"
    assert widget._annotation_has_been_saved is False
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
    assert widget.save_shapes_button.isEnabled() is True


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
    assert widget._annotation_has_been_saved is False
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
    assert widget._annotation_has_been_saved is False
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
    assert widget._annotation_has_been_saved is False
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
        return CreateShapesElementResult(
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


def test_shapes_annotation_widget_repeated_save_uses_overwrite_after_success(
    qtbot,
    monkeypatch,
    sdata_blobs: SpatialData,
) -> None:
    viewer = DummyViewer()
    widget = _create_ready_annotation_widget(qtbot, viewer, sdata_blobs)
    widget.create_layer_button.click()
    overwrites: list[bool] = []
    emitted_events: list[object] = []
    widget.app_state.shapes_element_written.connect(emitted_events.append)

    def fake_create_shapes_element(request, napari_layer):
        del napari_layer
        overwrites.append(request.overwrite)
        return CreateShapesElementResult(
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
    widget.save_shapes_button.click()

    assert overwrites == [False, True]
    assert widget._annotation_has_been_saved is True
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
        return CreateShapesElementResult(
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
    assert widget._annotation_has_been_saved is True
    assert "Shapes Saved" in _status_text(widget)


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
