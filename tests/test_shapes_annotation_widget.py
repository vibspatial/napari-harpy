from __future__ import annotations

from collections.abc import Callable
from html import unescape
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba
from napari.layers import Shapes
from qtpy.QtWidgets import QComboBox, QLabel
from spatialdata import SpatialData, read_zarr
from spatialdata.models import TableModel

import napari_harpy._app_state as app_state_module
import napari_harpy.widgets.shapes_annotation.widget as shapes_annotation_widget_module
from napari_harpy._app_state import ShapesElementWrittenEvent, get_or_create_app_state
from napari_harpy.core.shapes_annotation import AnnotateShapesElementResult
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


def _combo_index_for_text(combo: QComboBox, text: str) -> int:
    for index in range(combo.count()):
        if combo.itemText(index) == text:
            return index
    return -1


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


def _native_polygon_layer(name: str) -> Shapes:
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
        name=name,
    )


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

    widget.shapes_combo.setCurrentIndex(_combo_index_for_text(widget.shapes_combo, "other_polygons"))
    other_layer = widget._annotation_layer
    assert other_layer is not None
    assert other_layer is not blobs_layer
    assert list(viewer.layers) == [blobs_layer, other_layer]

    widget.shapes_combo.setCurrentIndex(_combo_index_for_text(widget.shapes_combo, "blobs_polygons"))

    assert widget._annotation_layer is blobs_layer
    assert list(viewer.layers) == [blobs_layer, other_layer]


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
    native_layer = Shapes([], shape_type="polygon", name="native_shapes")

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
    assert widget._selected_shapes_target == shapes_annotation_widget_module._ShapesAnnotationTarget.create_new()
    assert widget.shapes_combo.currentText() == "Create shapes..."
    assert widget.name_edit.text() == "native_shapes"
    assert native_layer.name == "native_shapes"
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
