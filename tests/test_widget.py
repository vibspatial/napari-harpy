from __future__ import annotations

from collections.abc import Callable
from html import unescape
from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import zarr
from matplotlib.colors import to_rgba
from napari.layers import Image, Labels
from napari.utils.colormaps import DirectLabelColormap
from qtpy.QtCore import QObject, Signal
from qtpy.QtWidgets import QComboBox, QScrollArea
from spatialdata import SpatialData, read_zarr
from spatialdata.models import TableModel
from spatialdata.transformations import get_transformation

import napari_harpy._annotation as annotation_module
import napari_harpy._app_state as app_state_module
import napari_harpy._class_palette as class_palette_module
import napari_harpy._classifier as classifier_module
import napari_harpy.widgets._object_classification_widget as widget_module
import napari_harpy.widgets._viewer_widget as viewer_widget_module
from napari_harpy._annotation import USER_CLASS_COLORS_KEY, USER_CLASS_COLUMN
from napari_harpy._app_state import FeatureMatrixWrittenEvent, get_or_create_app_state
from napari_harpy._class_palette import default_class_colors
from napari_harpy._classifier import (
    CLASSIFIER_CONFIG_KEY,
    PRED_CLASS_COLORS_KEY,
    PRED_CLASS_COLUMN,
    PRED_CONFIDENCE_COLUMN,
)
from napari_harpy._spatialdata import SpatialDataLabelsOption
from napari_harpy.widgets._object_classification_widget import (
    ObjectClassificationWidget as HarpyWidget,
)
from napari_harpy.widgets._viewer_widget import ViewerWidget


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
    def __init__(self, layers: list[Labels] | None = None) -> None:
        super().__init__(layers or [])
        self.selection = SimpleNamespace(active=None, select_only=self._select_only)
        self.events = SimpleNamespace(
            inserted=DummyEventEmitter(),
            removed=DummyEventEmitter(),
            reordered=DummyEventEmitter(),
        )

    def _select_only(self, layer: Labels) -> None:
        self.selection.active = layer


class DummyViewer:
    def __init__(self, layers: list[Labels] | None = None, *, seed_shared_sdata: bool = True) -> None:
        self.layers = DummyLayers(layers)
        if not seed_shared_sdata:
            return

        sdata_values: list[SpatialData] = []
        for layer in self.layers:
            metadata = getattr(layer, "metadata", None)
            if not isinstance(metadata, dict):
                continue
            sdata = metadata.get("sdata")
            if isinstance(sdata, SpatialData):
                sdata_values.append(sdata)

        if sdata_values and len({id(sdata) for sdata in sdata_values}) == 1:
            app_state = get_or_create_app_state(self)
            app_state.set_sdata(sdata_values[0])
            for layer in self.layers:
                if not isinstance(layer, Labels):
                    continue
                metadata = getattr(layer, "metadata", None)
                if not isinstance(metadata, dict):
                    continue
                sdata = metadata.get("sdata")
                if sdata is not sdata_values[0]:
                    continue
                element_name = metadata.get("name", getattr(layer, "name", None))
                if not isinstance(element_name, str):
                    continue
                coordinate_system = metadata.get("coordinate_system", metadata.get("_current_cs"))
                if not isinstance(coordinate_system, str) and element_name in sdata.labels:
                    available_coordinate_systems = tuple(
                        get_transformation(sdata.labels[element_name], get_all=True).keys()
                    )
                    if len(available_coordinate_systems) == 1:
                        coordinate_system = available_coordinate_systems[0]
                    elif "global" in available_coordinate_systems:
                        coordinate_system = "global"
                app_state.viewer_adapter.register_layer(
                    layer,
                    sdata=sdata,
                    element_name=element_name,
                    element_type="labels",
                    coordinate_system=coordinate_system if isinstance(coordinate_system, str) else None,
                )


def make_viewer_with_shared_sdata(sdata: SpatialData, layers: list[Labels] | None = None) -> DummyViewer:
    viewer = DummyViewer(layers=layers, seed_shared_sdata=False)
    get_or_create_app_state(viewer).set_sdata(sdata)
    return viewer


def select_segmentation(widget: HarpyWidget, index: int = 0) -> None:
    widget.segmentation_combo.setCurrentIndex(index)


_SUCCESS_FEEDBACK_STYLE = {
    "text": "#047857",
    "border": "#a7f3d0",
    "background": "#ecfdf5",
}


def _assert_persistence_success_feedback(widget: HarpyWidget, expected_message: str) -> None:
    assert "Persistence Updated" in widget.persistence_feedback.text()
    assert expected_message in widget.persistence_feedback.text()

    stylesheet = widget.persistence_feedback.styleSheet()
    assert f"color: {_SUCCESS_FEEDBACK_STYLE['text']}" in stylesheet
    assert f"background-color: {_SUCCESS_FEEDBACK_STYLE['background']}" in stylesheet
    assert f"border: 1px solid {_SUCCESS_FEEDBACK_STYLE['border']}" in stylesheet


def _patch_coordinate_system_names(monkeypatch, coordinate_systems: list[str]) -> None:
    monkeypatch.setattr(
        widget_module,
        "get_coordinate_system_names_from_sdata",
        lambda sdata: list(coordinate_systems),
    )
    monkeypatch.setattr(
        app_state_module,
        "get_coordinate_system_names_from_sdata",
        lambda sdata: list(coordinate_systems),
    )
    monkeypatch.setattr(
        viewer_widget_module,
        "get_coordinate_system_names_from_sdata",
        lambda sdata: list(coordinate_systems),
    )


class _DeferredWorker(QObject):
    returned = Signal(object)
    errored = Signal(object)
    finished = Signal()

    def __init__(self, result: classifier_module.ClassifierJobResult) -> None:
        super().__init__()
        self._result = result
        self.started = False
        self.quit_called = False

    def start(self) -> None:
        self.started = True

    def quit(self) -> None:
        self.quit_called = True

    def emit_returned(self) -> None:
        self.returned.emit(self._result)
        self.finished.emit()


def make_blobs_labels_layer(sdata: SpatialData, label_name: str = "blobs_labels") -> Labels:
    layer = Labels(
        sdata.labels[label_name],
        name=label_name,
        metadata={"sdata": sdata, "name": label_name},
    )
    return layer


def make_multiscale_blobs_labels_layer(sdata: SpatialData, label_name: str = "blobs_labels") -> Labels:
    base_data = np.asarray(sdata.labels[label_name])
    multiscale_data = [base_data, base_data[::2, ::2]]
    indices = [int(value) for value in np.unique(base_data).tolist() if int(value) > 0]
    layer = Labels(
        multiscale_data,
        name=label_name,
        metadata={"sdata": sdata, "name": label_name, "indices": indices},
    )
    return layer


def _write_disk_table_state(
    backed_sdata_blobs: SpatialData,
    *,
    obs: pd.DataFrame,
    obsm: dict[str, object],
    uns: dict[str, object],
) -> None:
    root = zarr.open_group(backed_sdata_blobs.path, mode="a", use_consolidated=False)
    table_group = root["tables/table"]
    ad.io.write_elem(table_group, "obs", obs)
    ad.io.write_elem(table_group, "obsm", obsm)
    ad.io.write_elem(table_group, "uns", uns)


def rename_table_instance_key(sdata: SpatialData, *, table_name: str = "table", instance_key: str) -> None:
    table = sdata[table_name]
    table.obs = table.obs.rename(columns={"instance_id": instance_key})
    table.uns[TableModel.ATTRS_KEY][TableModel.INSTANCE_KEY] = instance_key


def test_widget_can_be_instantiated(qtbot) -> None:
    widget = HarpyWidget()

    qtbot.addWidget(widget)

    scroll_area = widget.findChild(QScrollArea, "object_classification_scroll_area")
    assert scroll_area is not None
    assert scroll_area.widgetResizable()
    assert widget is not None
    assert widget.selected_segmentation_name is None
    assert widget.selected_table_name is None
    assert widget.selected_feature_key is None
    assert widget.selected_training_scope == classifier_module.DEFAULT_TRAINING_SCOPE
    assert widget.selected_prediction_scope == classifier_module.DEFAULT_PREDICTION_SCOPE
    assert widget.selected_coordinate_system is None
    assert widget.selected_color_by == "user_class"
    assert all(button.text() != "Rescan Viewer" for button in widget.findChildren(type(widget.retrain_button)))
    assert "No SpatialData Loaded" in widget.selection_status.text()
    assert widget.coordinate_system_combo.sizeAdjustPolicy() == (
        QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
    )
    assert (
        widget.segmentation_combo.sizeAdjustPolicy() == QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
    )
    assert widget.table_combo.sizeAdjustPolicy() == QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
    assert widget.feature_matrix_combo.sizeAdjustPolicy() == (
        QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
    )
    assert widget.training_scope_combo.currentData() == classifier_module.DEFAULT_TRAINING_SCOPE
    assert widget.prediction_scope_combo.currentData() == classifier_module.DEFAULT_PREDICTION_SCOPE


def test_widget_refreshes_when_shared_sdata_changes(qtbot, sdata_blobs: SpatialData) -> None:
    layer = make_blobs_labels_layer(sdata_blobs)
    viewer = DummyViewer(layers=[layer], seed_shared_sdata=False)
    app_state = get_or_create_app_state(viewer)
    widget = HarpyWidget(viewer)

    qtbot.addWidget(widget)

    assert widget.segmentation_combo.count() == 0
    assert "No SpatialData Loaded" in widget.selection_status.text()

    app_state.set_sdata(sdata_blobs)

    assert widget.coordinate_system_combo.count() == 1
    assert widget.coordinate_system_combo.itemText(0) == "global"
    assert widget.selected_coordinate_system == "global"
    assert widget.segmentation_combo.count() == 2
    assert widget.selected_segmentation_name is None
    assert widget.selected_spatialdata is sdata_blobs
    assert widget.selected_table_name is None
    assert widget.selected_feature_key is None
    assert "Choose a segmentation mask" in widget.selection_status.text()


def test_widget_clears_when_shared_sdata_is_cleared(qtbot, sdata_blobs: SpatialData) -> None:
    layer = make_blobs_labels_layer(sdata_blobs)
    viewer = make_viewer_with_shared_sdata(sdata_blobs, layers=[layer])
    app_state = get_or_create_app_state(viewer)
    widget = HarpyWidget(viewer)

    qtbot.addWidget(widget)

    assert widget.segmentation_combo.count() == 2

    app_state.clear_sdata()

    assert widget.selected_coordinate_system is None
    assert widget.selected_segmentation_name is None
    assert widget.selected_spatialdata is None
    assert widget.selected_table_name is None
    assert widget.selected_feature_key is None
    assert widget.coordinate_system_combo.count() == 0
    assert not widget.coordinate_system_combo.isEnabled()
    assert widget.segmentation_combo.count() == 0
    assert not widget.segmentation_combo.isEnabled()
    assert widget.table_combo.count() == 0
    assert not widget.table_combo.isEnabled()
    assert widget.feature_matrix_combo.count() == 0
    assert not widget.feature_matrix_combo.isEnabled()
    assert "No SpatialData Loaded" in widget.selection_status.text()


def test_widget_populates_segmentation_dropdown_from_spatialdata(qtbot, sdata_blobs: SpatialData) -> None:
    layer = make_blobs_labels_layer(sdata_blobs)
    viewer = DummyViewer(layers=[layer])

    widget = HarpyWidget(viewer)
    qtbot.addWidget(widget)

    assert widget.coordinate_system_combo.count() == 1
    assert widget.coordinate_system_combo.itemText(0) == "global"
    assert widget.selected_coordinate_system == "global"
    assert widget.segmentation_combo.count() == 2
    assert [widget.segmentation_combo.itemText(index) for index in range(widget.segmentation_combo.count())] == [
        "blobs_labels",
        "blobs_multiscale_labels",
    ]
    assert widget.table_combo.count() == 0
    assert widget.feature_matrix_combo.count() == 0
    assert widget.color_by_combo.count() == 3
    assert [widget.color_by_combo.itemText(index) for index in range(widget.color_by_combo.count())] == [
        "user_class",
        "pred_class",
        "pred_confidence",
    ]
    assert widget.selected_segmentation_name is None
    assert widget.selected_spatialdata is sdata_blobs
    assert widget.selected_table_name is None
    assert widget.selected_feature_key is None
    assert widget.selected_color_by == "user_class"
    assert widget.selected_table_metadata is None
    assert "adata" not in layer.metadata
    assert widget.selected_instance_id is None
    assert all(button.text() != "Rescan Viewer" for button in widget.findChildren(type(widget.retrain_button)))
    assert widget.retrain_button.text() == "Train Classifier"
    assert widget.sync_button.text() == "Write"
    assert widget.reload_button.text() == "Reload"
    assert not widget.sync_button.isEnabled()
    assert not widget.reload_button.isEnabled()
    assert not widget.retrain_button.isEnabled()
    assert len(viewer.layers) == 1
    assert viewer.layers.selection.active is None
    assert "Choose a segmentation mask" in widget.selection_status.text()
    assert widget.validation_status.isHidden()
    assert widget.validation_status.text() == ""
    assert widget.classifier_feedback.isHidden()
    assert widget.classifier_preparation_status.isHidden()
    assert widget.classifier_preparation_status.objectName() == "classifier_preparation_status"


def test_widget_populates_segmentation_choices_from_shared_sdata_without_loaded_layer(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    viewer = make_viewer_with_shared_sdata(sdata_blobs)

    widget = HarpyWidget(viewer)
    qtbot.addWidget(widget)

    assert widget.coordinate_system_combo.count() == 1
    assert widget.coordinate_system_combo.itemText(0) == "global"
    assert widget.selected_coordinate_system == "global"
    assert widget.segmentation_combo.count() == 2
    assert [widget.segmentation_combo.itemText(index) for index in range(widget.segmentation_combo.count())] == [
        "blobs_labels",
        "blobs_multiscale_labels",
    ]
    assert widget.selected_segmentation_name is None
    assert widget.selected_spatialdata is sdata_blobs
    assert widget.selected_table_name is None
    assert widget.selected_feature_key is None
    assert len(viewer.layers) == 0
    assert widget._annotation_controller.labels_layer is None
    assert widget._viewer_styling_controller.labels_layer is None
    assert "Choose a segmentation mask" in widget.selection_status.text()
    assert not widget.apply_class_button.isEnabled()


def test_widget_filters_segmentation_choices_by_selected_coordinate_system(
    qtbot, monkeypatch, sdata_blobs: SpatialData
) -> None:
    _patch_coordinate_system_names(monkeypatch, ["cells", "global"])
    viewer = make_viewer_with_shared_sdata(sdata_blobs)

    global_option = SpatialDataLabelsOption(
        label_name="blobs_labels",
        display_name="blobs_labels",
        sdata=sdata_blobs,
        coordinate_systems=("global",),
    )
    cells_option = SpatialDataLabelsOption(
        label_name="blobs_multiscale_labels",
        display_name="blobs_multiscale_labels",
        sdata=sdata_blobs,
        coordinate_systems=("cells",),
    )
    monkeypatch.setattr(
        widget_module,
        "get_spatialdata_label_options_for_coordinate_system_from_sdata",
        lambda *, sdata, coordinate_system: [global_option] if coordinate_system == "global" else [cells_option],
    )

    widget = HarpyWidget(viewer)
    qtbot.addWidget(widget)

    assert widget.coordinate_system_combo.count() == 2
    assert [
        widget.coordinate_system_combo.itemText(index) for index in range(widget.coordinate_system_combo.count())
    ] == [
        "cells",
        "global",
    ]
    assert widget.selected_coordinate_system == "cells"
    assert widget.segmentation_combo.count() == 1
    assert widget.segmentation_combo.itemText(0) == "blobs_multiscale_labels"
    assert widget.selected_segmentation_name is None
    assert widget.selected_table_name is None
    assert widget.selected_feature_key is None

    with qtbot.waitSignal(widget.app_state.coordinate_system_changed) as blocker:
        widget.coordinate_system_combo.setCurrentIndex(1)

    assert blocker.args[0].previous_coordinate_system == "cells"
    assert blocker.args[0].coordinate_system == "global"
    assert blocker.args[0].source == "object_classification_widget"
    assert widget.selected_coordinate_system == "global"
    assert widget.segmentation_combo.count() == 1
    assert widget.segmentation_combo.itemText(0) == "blobs_labels"
    assert widget.selected_segmentation_name is None
    assert widget.selected_table_name is None
    assert widget.selected_feature_key is None

    select_segmentation(widget)

    assert widget.selected_segmentation_name == "blobs_labels"
    assert widget.selected_table_name == "table"
    assert widget.selected_feature_key == "features_1"


def test_widget_coordinate_system_change_updates_viewer_widget(qtbot, monkeypatch) -> None:
    _patch_coordinate_system_names(monkeypatch, ["global", "local"])
    fake_sdata = object()
    shared_option = SpatialDataLabelsOption(
        label_name="shared_labels",
        display_name="shared_labels",
        sdata=fake_sdata,
        coordinate_systems=("global", "local"),
    )

    monkeypatch.setattr(
        widget_module,
        "get_spatialdata_label_options_for_coordinate_system_from_sdata",
        lambda *, sdata, coordinate_system: [shared_option],
    )
    monkeypatch.setattr(widget_module, "get_annotating_table_names", lambda sdata, label_name: [])
    monkeypatch.setattr(viewer_widget_module, "_get_labels_in_coordinate_system", lambda sdata, coordinate_system: [])
    monkeypatch.setattr(viewer_widget_module, "_get_images_in_coordinate_system", lambda sdata, coordinate_system: [])

    viewer = DummyViewer(seed_shared_sdata=False)
    app_state = get_or_create_app_state(viewer)
    app_state.set_sdata(fake_sdata)
    viewer_widget = ViewerWidget(viewer)
    object_widget = HarpyWidget(viewer)

    qtbot.addWidget(viewer_widget)
    qtbot.addWidget(object_widget)

    assert viewer_widget.coordinate_system_combo.currentText() == "global"
    assert object_widget.coordinate_system_combo.currentText() == "global"

    with qtbot.waitSignal(app_state.coordinate_system_changed) as blocker:
        object_widget.coordinate_system_combo.setCurrentIndex(1)

    assert blocker.args[0].previous_coordinate_system == "global"
    assert blocker.args[0].coordinate_system == "local"
    assert blocker.args[0].source == "object_classification_widget"
    assert viewer_widget.coordinate_system_combo.currentText() == "local"
    assert object_widget.coordinate_system_combo.currentText() == "local"


def test_shared_coordinate_system_switch_prunes_registered_layers_and_keeps_external_layers(qtbot, monkeypatch) -> None:
    _patch_coordinate_system_names(monkeypatch, ["global", "local"])
    fake_sdata = object()
    global_image = Image(np.zeros((4, 4), dtype=np.float32), name="global_image")
    local_image = Image(np.zeros((4, 4), dtype=np.float32), name="local_image")
    external_image = Image(np.zeros((4, 4), dtype=np.float32), name="external_image")
    global_labels = Labels(np.ones((4, 4), dtype=np.int32), name="global_labels")
    local_labels = Labels(np.ones((4, 4), dtype=np.int32), name="local_labels")
    external_labels = Labels(np.ones((4, 4), dtype=np.int32), name="external_labels")

    monkeypatch.setattr(viewer_widget_module, "_get_labels_in_coordinate_system", lambda sdata, coordinate_system: [])
    monkeypatch.setattr(viewer_widget_module, "_get_images_in_coordinate_system", lambda sdata, coordinate_system: [])
    monkeypatch.setattr(
        widget_module,
        "get_spatialdata_label_options_for_coordinate_system_from_sdata",
        lambda *, sdata, coordinate_system: [],
    )

    viewer = DummyViewer(seed_shared_sdata=False)
    viewer.layers.extend([global_image, local_image, external_image, global_labels, local_labels, external_labels])
    app_state = get_or_create_app_state(viewer)
    app_state.set_sdata(fake_sdata)
    viewer_widget = ViewerWidget(viewer)
    object_widget = HarpyWidget(viewer)

    qtbot.addWidget(viewer_widget)
    qtbot.addWidget(object_widget)

    app_state.viewer_adapter.register_layer(
        global_image,
        sdata=fake_sdata,
        element_name="global_image",
        element_type="image",
        coordinate_system="global",
    )
    app_state.viewer_adapter.register_layer(
        local_image,
        sdata=fake_sdata,
        element_name="local_image",
        element_type="image",
        coordinate_system="local",
    )
    app_state.viewer_adapter.register_layer(
        global_labels,
        sdata=fake_sdata,
        element_name="global_labels",
        element_type="labels",
        coordinate_system="global",
    )
    app_state.viewer_adapter.register_layer(
        local_labels,
        sdata=fake_sdata,
        element_name="local_labels",
        element_type="labels",
        coordinate_system="local",
    )

    with qtbot.waitSignal(app_state.coordinate_system_changed):
        viewer_widget.coordinate_system_combo.setCurrentIndex(1)

    assert app_state.coordinate_system == "local"
    assert viewer_widget.coordinate_system_combo.currentText() == "local"
    assert object_widget.coordinate_system_combo.currentText() == "local"
    assert list(viewer.layers) == [local_image, external_image, local_labels, external_labels]
    assert app_state.viewer_adapter.layer_bindings.get_binding(global_image) is None
    assert app_state.viewer_adapter.layer_bindings.get_binding(global_labels) is None
    assert app_state.viewer_adapter.layer_bindings.get_binding(local_image) is not None
    assert app_state.viewer_adapter.layer_bindings.get_binding(local_labels) is not None
    assert app_state.viewer_adapter.layer_bindings.get_binding(external_image) is None
    assert app_state.viewer_adapter.layer_bindings.get_binding(external_labels) is None
    assert [binding.element_name for binding in app_state.viewer_adapter.layer_bindings.iter_bindings()] == [
        "local_image",
        "local_labels",
    ]


def test_widget_clears_selected_segmentation_on_coordinate_system_change_even_when_it_is_valid(
    qtbot, monkeypatch
) -> None:
    _patch_coordinate_system_names(monkeypatch, ["global", "local"])
    fake_sdata = object()
    shared_option = SpatialDataLabelsOption(
        label_name="shared_labels",
        display_name="shared_labels",
        sdata=fake_sdata,
        coordinate_systems=("global", "local"),
    )
    global_layer = Labels(np.ones((4, 4), dtype=np.int32), name="shared_labels")

    monkeypatch.setattr(
        widget_module,
        "get_spatialdata_label_options_for_coordinate_system_from_sdata",
        lambda *, sdata, coordinate_system: [shared_option],
    )
    monkeypatch.setattr(widget_module, "get_annotating_table_names", lambda sdata, label_name: [])

    viewer = DummyViewer(seed_shared_sdata=False)
    app_state = get_or_create_app_state(viewer)
    app_state.set_sdata(fake_sdata)
    viewer.layers.append(global_layer)
    app_state.viewer_adapter.register_layer(
        global_layer,
        sdata=fake_sdata,
        element_name="shared_labels",
        element_type="labels",
        coordinate_system="global",
    )

    monkeypatch.setattr(
        app_state.viewer_adapter,
        "ensure_labels_loaded",
        lambda sdata, label_name, coordinate_system: (_ for _ in ()).throw(
            AssertionError("Coordinate-system switching should not auto-load a replacement segmentation layer.")
        ),
    )

    widget = HarpyWidget(viewer)
    qtbot.addWidget(widget)
    select_segmentation(widget)

    assert widget.selected_segmentation_name == "shared_labels"
    assert widget._annotation_controller.labels_layer is global_layer
    assert viewer.layers.selection.active is global_layer

    with qtbot.waitSignal(app_state.coordinate_system_changed):
        widget.coordinate_system_combo.setCurrentIndex(1)

    assert widget.selected_coordinate_system == "local"
    assert widget.selected_segmentation_name is None
    assert widget.selected_table_name is None
    assert widget.selected_feature_key is None
    assert widget._annotation_controller.labels_layer is None
    assert widget._viewer_styling_controller.labels_layer is None
    assert list(viewer.layers) == []
    assert app_state.viewer_adapter.layer_bindings.get_binding(global_layer) is None
    assert "Choose a segmentation mask" in widget.selection_status.text()


def test_widget_unbinds_when_selected_segmentation_is_not_valid_in_new_coordinate_system(qtbot, monkeypatch) -> None:
    _patch_coordinate_system_names(monkeypatch, ["global", "local"])
    fake_sdata = object()
    global_option = SpatialDataLabelsOption(
        label_name="global_labels",
        display_name="global_labels",
        sdata=fake_sdata,
        coordinate_systems=("global",),
    )
    local_option = SpatialDataLabelsOption(
        label_name="local_labels",
        display_name="local_labels",
        sdata=fake_sdata,
        coordinate_systems=("local",),
    )
    global_layer = Labels(np.ones((4, 4), dtype=np.int32), name="global_labels")

    monkeypatch.setattr(
        widget_module,
        "get_spatialdata_label_options_for_coordinate_system_from_sdata",
        lambda *, sdata, coordinate_system: [global_option] if coordinate_system == "global" else [local_option],
    )
    monkeypatch.setattr(widget_module, "get_annotating_table_names", lambda sdata, label_name: [])

    viewer = DummyViewer(seed_shared_sdata=False)
    app_state = get_or_create_app_state(viewer)
    app_state.set_sdata(fake_sdata)
    viewer.layers.append(global_layer)
    app_state.viewer_adapter.register_layer(
        global_layer,
        sdata=fake_sdata,
        element_name="global_labels",
        element_type="labels",
        coordinate_system="global",
    )

    widget = HarpyWidget(viewer)
    qtbot.addWidget(widget)
    select_segmentation(widget)

    assert widget.selected_segmentation_name == "global_labels"
    assert widget._annotation_controller.labels_layer is global_layer

    with qtbot.waitSignal(app_state.coordinate_system_changed):
        widget.coordinate_system_combo.setCurrentIndex(1)

    assert widget.selected_coordinate_system == "local"
    assert widget.selected_segmentation_name is None
    assert widget.selected_table_name is None
    assert widget.selected_feature_key is None
    assert widget._annotation_controller.labels_layer is None
    assert widget._viewer_styling_controller.labels_layer is None
    assert list(viewer.layers) == []
    assert "Choose a segmentation mask" in widget.selection_status.text()


def test_widget_surfaces_invalid_table_binding_for_duplicate_instance_ids(qtbot, sdata_blobs: SpatialData) -> None:
    table = sdata_blobs["table"]
    first_index, second_index = table.obs.index[:2]
    table.obs.loc[second_index, "instance_id"] = table.obs.loc[first_index, "instance_id"]
    layer = make_blobs_labels_layer(sdata_blobs)
    viewer = DummyViewer(layers=[layer])

    widget = HarpyWidget(viewer)
    qtbot.addWidget(widget)
    select_segmentation(widget)

    assert widget.selected_table_name == "table"
    assert widget.validation_status.isHidden()
    assert widget.validation_status.text() == ""
    assert "contains duplicate values within that region" in widget.selection_status.text()
    assert not widget.color_by_combo.isEnabled()
    assert not widget.class_spinbox.isEnabled()
    assert not widget.retrain_button.isEnabled()
    assert not widget.sync_button.isEnabled()
    assert not widget.reload_button.isEnabled()


def test_widget_auto_loads_selected_segmentation_when_shared_sdata_is_set(qtbot, sdata_blobs: SpatialData) -> None:
    viewer = DummyViewer()
    app_state = get_or_create_app_state(viewer)
    widget = HarpyWidget(viewer)
    qtbot.addWidget(widget)

    assert widget.segmentation_combo.count() == 0

    app_state.set_sdata(sdata_blobs)
    assert widget.coordinate_system_combo.count() == 1
    assert widget.coordinate_system_combo.itemText(0) == "global"
    assert widget.selected_coordinate_system == "global"
    assert widget.segmentation_combo.count() == 2
    assert len(viewer.layers) == 0
    assert widget.table_combo.count() == 0
    assert widget.feature_matrix_combo.count() == 0
    assert widget.selected_segmentation_name is None
    assert widget.selected_table_name is None
    assert widget.selected_feature_key is None
    assert "Choose a segmentation mask" in widget.selection_status.text()


def test_widget_updates_table_dropdown_when_segmentation_changes(qtbot, sdata_blobs: SpatialData) -> None:
    layer = make_blobs_labels_layer(sdata_blobs)
    viewer = DummyViewer(layers=[layer])

    widget = HarpyWidget(viewer)
    qtbot.addWidget(widget)

    widget.segmentation_combo.setCurrentIndex(1)

    assert widget.selected_segmentation_name == "blobs_multiscale_labels"
    assert widget.table_combo.count() == 0
    assert not widget.table_combo.isEnabled()
    assert widget.selected_table_name is None
    assert widget.feature_matrix_combo.count() == 0
    assert not widget.feature_matrix_combo.isEnabled()
    assert widget.selected_feature_key is None


def test_widget_warns_when_loaded_segmentation_has_no_annotation_table(qtbot, sdata_blobs: SpatialData) -> None:
    primary_layer = make_blobs_labels_layer(sdata_blobs)
    base_data = np.asarray(sdata_blobs.labels["blobs_labels"])
    multiscale_layer = Labels(
        [base_data, base_data[::2, ::2]],
        name="blobs_multiscale_labels",
        metadata={
            "sdata": sdata_blobs,
            "name": "blobs_multiscale_labels",
            "indices": [int(value) for value in np.unique(base_data).tolist() if int(value) > 0],
        },
    )
    viewer = DummyViewer(layers=[primary_layer, multiscale_layer])

    widget = HarpyWidget(viewer)
    qtbot.addWidget(widget)

    widget.segmentation_combo.setCurrentIndex(1)

    assert widget.selected_segmentation_name == "blobs_multiscale_labels"
    assert widget.selected_table_name is None
    assert viewer.layers.selection.active is multiscale_layer
    assert "This segmentation is loaded, but no annotation table is linked to it." in widget.selection_status.text()
    assert not widget.class_spinbox.isEnabled()
    assert not widget.apply_class_button.isEnabled()


def test_widget_updates_selected_feature_key_when_feature_matrix_changes(qtbot, sdata_blobs: SpatialData) -> None:
    layer = make_blobs_labels_layer(sdata_blobs)
    viewer = DummyViewer(layers=[layer])

    widget = HarpyWidget(viewer)
    qtbot.addWidget(widget)
    select_segmentation(widget)
    widget.training_scope_combo.setCurrentIndex(widget.training_scope_combo.findData("selected_segmentation_only"))

    bind_calls: list[tuple[str, str]] = []

    def record_bind(
        sdata,
        label_name,
        table_name,
        feature_key,
        *,
        training_scope=classifier_module.DEFAULT_TRAINING_SCOPE,
        prediction_scope=classifier_module.DEFAULT_PREDICTION_SCOPE,
    ) -> bool:
        del sdata, label_name, table_name, feature_key
        bind_calls.append((training_scope, prediction_scope))
        return True

    widget._classifier_controller.bind = record_bind  # type: ignore[method-assign]

    widget.feature_matrix_combo.setCurrentIndex(1)

    assert widget.selected_feature_key == "features_2"
    assert bind_calls == [("selected_segmentation_only", classifier_module.DEFAULT_PREDICTION_SCOPE)]
    assert "feature matrix changed" in widget.classifier_feedback.text()


def test_widget_marks_classifier_dirty_when_training_scope_changes(qtbot, monkeypatch, sdata_blobs: SpatialData) -> None:
    layer = make_blobs_labels_layer(sdata_blobs)
    viewer = DummyViewer(layers=[layer])

    widget = HarpyWidget(viewer)
    qtbot.addWidget(widget)
    select_segmentation(widget)

    bind_calls: list[tuple[str, str]] = []
    mark_dirty_reasons: list[str | None] = []

    def record_bind(
        sdata,
        label_name,
        table_name,
        feature_key,
        *,
        training_scope=classifier_module.DEFAULT_TRAINING_SCOPE,
        prediction_scope=classifier_module.DEFAULT_PREDICTION_SCOPE,
    ) -> bool:
        del sdata, label_name, table_name, feature_key
        bind_calls.append((training_scope, prediction_scope))
        return True

    def record_mark_dirty(*, reason: str | None = None) -> None:
        mark_dirty_reasons.append(reason)

    monkeypatch.setattr(widget._classifier_controller, "bind", record_bind)
    monkeypatch.setattr(widget._classifier_controller, "mark_dirty", record_mark_dirty)

    widget.training_scope_combo.setCurrentIndex(widget.training_scope_combo.findData("selected_segmentation_only"))

    assert widget.selected_training_scope == "selected_segmentation_only"
    assert bind_calls == [("selected_segmentation_only", classifier_module.DEFAULT_PREDICTION_SCOPE)]
    assert mark_dirty_reasons == ["the training scope changed"]


def test_widget_marks_classifier_dirty_when_prediction_scope_changes(
    qtbot, monkeypatch, sdata_blobs: SpatialData
) -> None:
    layer = make_blobs_labels_layer(sdata_blobs)
    viewer = DummyViewer(layers=[layer])

    widget = HarpyWidget(viewer)
    qtbot.addWidget(widget)
    select_segmentation(widget)

    bind_calls: list[tuple[str, str]] = []
    mark_dirty_reasons: list[str | None] = []

    def record_bind(
        sdata,
        label_name,
        table_name,
        feature_key,
        *,
        training_scope=classifier_module.DEFAULT_TRAINING_SCOPE,
        prediction_scope=classifier_module.DEFAULT_PREDICTION_SCOPE,
    ) -> bool:
        del sdata, label_name, table_name, feature_key
        bind_calls.append((training_scope, prediction_scope))
        return True

    def record_mark_dirty(*, reason: str | None = None) -> None:
        mark_dirty_reasons.append(reason)

    monkeypatch.setattr(widget._classifier_controller, "bind", record_bind)
    monkeypatch.setattr(widget._classifier_controller, "mark_dirty", record_mark_dirty)

    widget.prediction_scope_combo.setCurrentIndex(widget.prediction_scope_combo.findData("all"))

    assert widget.selected_prediction_scope == "all"
    assert bind_calls == [(classifier_module.DEFAULT_TRAINING_SCOPE, "all")]
    assert mark_dirty_reasons == ["the prediction scope changed"]


def test_widget_shows_classifier_preparation_hidden_write_warning_for_table_wide_prediction_scope(
    qtbot, sdata_blobs_multi_region: SpatialData
) -> None:
    layer = make_blobs_labels_layer(sdata_blobs_multi_region)
    viewer = DummyViewer(layers=[layer])

    widget = HarpyWidget(viewer)
    qtbot.addWidget(widget)
    select_segmentation(widget)
    table_index = widget.table_combo.findData("table_multi")
    assert table_index >= 0
    widget.table_combo.setCurrentIndex(table_index)

    widget.prediction_scope_combo.setCurrentIndex(widget.prediction_scope_combo.findData("all"))

    table = sdata_blobs_multi_region["table_multi"]
    assert not widget.classifier_preparation_status.isHidden()
    assert f"Prediction: {table.n_obs} eligible rows across 2 regions." in widget.classifier_preparation_status.text()
    assert "Some prediction updates may not be visible in the current selection." in (
        widget.classifier_preparation_status.text()
    )


def test_widget_omits_hidden_write_line_for_effectively_selected_prediction_scope(
    qtbot, sdata_blobs: SpatialData
) -> None:
    layer = make_blobs_labels_layer(sdata_blobs)
    viewer = DummyViewer(layers=[layer])

    widget = HarpyWidget(viewer)
    qtbot.addWidget(widget)
    select_segmentation(widget)

    widget.prediction_scope_combo.setCurrentIndex(widget.prediction_scope_combo.findData("all"))

    assert widget.selected_prediction_scope == "all"
    assert not widget.classifier_preparation_status.isHidden()
    assert "Prediction:" in widget.classifier_preparation_status.text()
    assert "Some prediction updates may not be visible" not in widget.classifier_preparation_status.text()


def test_widget_shows_eligible_classifier_preparation_summary(qtbot, sdata_blobs: SpatialData) -> None:
    table = sdata_blobs["table"]
    instance_ids = table.obs["instance_id"].to_numpy(dtype=np.int64)
    table.obs[USER_CLASS_COLUMN] = pd.Categorical(
        [1 if int(instance_id) in {1, 2} else 2 if int(instance_id) in {24, 25} else 0 for instance_id in instance_ids],
        categories=[0, 1, 2],
    )
    layer = make_blobs_labels_layer(sdata_blobs)
    viewer = DummyViewer(layers=[layer])

    widget = HarpyWidget(viewer)
    qtbot.addWidget(widget)
    select_segmentation(widget)

    assert not widget.classifier_preparation_status.isHidden()
    preparation_text = widget.classifier_preparation_status.text()
    assert "Training: 4 labeled rows across 1 region." in preparation_text
    assert f"Prediction: {table.n_obs} eligible rows in selected region." in preparation_text
    assert "Feature matrix: `features_1`, 4 features." in preparation_text
    assert "Need at least" not in preparation_text


def test_widget_refreshes_feature_matrix_selector_when_first_key_is_written(qtbot, sdata_blobs: SpatialData) -> None:
    table = sdata_blobs["table"]
    for key in list(table.obsm.keys()):
        del table.obsm[key]

    layer = make_blobs_labels_layer(sdata_blobs)
    viewer = DummyViewer(layers=[layer])
    app_state = get_or_create_app_state(viewer)
    widget = HarpyWidget(viewer)
    qtbot.addWidget(widget)
    select_segmentation(widget)
    mark_dirty_reasons: list[str | None] = []

    def record_mark_dirty(*, reason: str | None = None) -> None:
        mark_dirty_reasons.append(reason)

    widget._classifier_controller.mark_dirty = record_mark_dirty  # type: ignore[method-assign]

    assert widget.selected_feature_key is None
    assert widget._persistence_controller.is_dirty is False

    table.obsm["features_new"] = np.arange(table.n_obs, dtype=np.float64).reshape(table.n_obs, 1)
    app_state.emit_feature_matrix_written(
        FeatureMatrixWrittenEvent(
            sdata=sdata_blobs,
            table_name="table",
            feature_key="features_new",
            change_kind="created",
        )
    )

    assert widget.feature_matrix_combo.count() == 1
    assert widget.selected_feature_key == "features_new"
    assert widget._persistence_controller.is_dirty is True
    assert mark_dirty_reasons == []


def test_widget_invalidates_classifier_when_selected_feature_matrix_is_overwritten(
    qtbot, sdata_blobs: SpatialData
) -> None:
    layer = make_blobs_labels_layer(sdata_blobs)
    viewer = DummyViewer(layers=[layer])
    app_state = get_or_create_app_state(viewer)
    widget = HarpyWidget(viewer)
    qtbot.addWidget(widget)
    select_segmentation(widget)
    table = sdata_blobs["table"]
    instance_ids = table.obs["instance_id"].to_numpy(dtype=np.int64)
    table.obs[USER_CLASS_COLUMN] = pd.Categorical(
        [1 if int(instance_id) in {1, 2} else 2 if int(instance_id) in {24, 25} else 0 for instance_id in instance_ids],
        categories=[0, 1, 2],
    )
    training_scope = classifier_module.ResolvedClassifierScope(
        mode="selected_segmentation_only",
        regions=("blobs_labels",),
        table_row_positions=np.array([0, 1], dtype=np.int64),
        n_rows_in_regions=2,
    )
    prediction_scope = classifier_module.ResolvedClassifierScope(
        mode="selected_segmentation_only",
        regions=("blobs_labels",),
        table_row_positions=np.array([0, 1], dtype=np.int64),
        n_rows_in_regions=2,
    )
    worker = _DeferredWorker(
        classifier_module.ClassifierJobResult(
            job_id=1,
            feature_key="features_1",
            label_name="blobs_labels",
            table_name="table",
            pred_classes=np.array([1, 2], dtype=np.int64),
            pred_confidences=np.array([0.9, 0.8], dtype=np.float64),
            trained_at="2026-04-23T09:00:00+00:00",
            model_params=dict(classifier_module.RANDOM_FOREST_PARAMS),
            summary=classifier_module.ClassifierPreparationSummary(
                training_scope=training_scope,
                prediction_scope=prediction_scope,
                eligible=True,
                reason="Ready to train.",
                labeled_count=2,
                class_labels=(1, 2),
                n_features=2,
            ),
        )
    )

    widget._classifier_controller._create_training_worker = lambda job: worker  # type: ignore[method-assign]

    assert widget._classifier_controller.schedule_retrain(immediate=True) is True
    assert widget._classifier_controller.is_training is True

    table.obsm["features_1"] = np.arange(table.n_obs * 2, dtype=np.float64).reshape(table.n_obs, 2)
    app_state.emit_feature_matrix_written(
        FeatureMatrixWrittenEvent(
            sdata=sdata_blobs,
            table_name="table",
            feature_key="features_1",
            change_kind="updated",
        )
    )

    assert worker.quit_called is True
    assert widget._classifier_controller.is_training is False
    assert widget._classifier_controller.is_dirty is True
    assert widget.selected_feature_key == "features_1"
    assert widget._persistence_controller.is_dirty is True
    assert "overwritten" in widget.classifier_feedback.text()


def test_widget_ignores_feature_matrix_writes_for_other_tables(qtbot, sdata_blobs: SpatialData) -> None:
    layer = make_blobs_labels_layer(sdata_blobs)
    viewer = DummyViewer(layers=[layer])
    app_state = get_or_create_app_state(viewer)
    widget = HarpyWidget(viewer)
    qtbot.addWidget(widget)
    select_segmentation(widget)
    previous_items = [
        widget.feature_matrix_combo.itemText(index) for index in range(widget.feature_matrix_combo.count())
    ]

    app_state.emit_feature_matrix_written(
        FeatureMatrixWrittenEvent(
            sdata=sdata_blobs,
            table_name="other_table",
            feature_key="features_new",
            change_kind="created",
        )
    )

    assert [
        widget.feature_matrix_combo.itemText(index) for index in range(widget.feature_matrix_combo.count())
    ] == previous_items
    assert widget.selected_feature_key == "features_1"
    assert widget._persistence_controller.is_dirty is False


def test_widget_updates_color_by_mode_when_selection_changes(qtbot, sdata_blobs: SpatialData) -> None:
    layer = make_blobs_labels_layer(sdata_blobs)
    viewer = DummyViewer(layers=[layer])

    widget = HarpyWidget(viewer)
    qtbot.addWidget(widget)

    widget.color_by_combo.setCurrentIndex(1)

    assert widget.selected_color_by == "pred_class"


def test_widget_tracks_picked_instance_id_from_labels_layer(qtbot, sdata_blobs: SpatialData) -> None:
    layer = make_blobs_labels_layer(sdata_blobs)
    viewer = DummyViewer(layers=[layer])

    widget = HarpyWidget(viewer)
    qtbot.addWidget(widget)
    select_segmentation(widget)

    layer.selected_label = 5

    assert widget.selected_instance_id == 5
    assert widget.apply_class_button.isEnabled()
    assert "Current instance_id: 5." in widget.selection_status.text()
    assert "Current class: unlabeled." in widget.selection_status.text()


def test_widget_accepts_first_pick_when_instance_id_is_one(qtbot, sdata_blobs: SpatialData) -> None:
    layer = make_blobs_labels_layer(sdata_blobs)
    viewer = DummyViewer(layers=[layer])

    widget = HarpyWidget(viewer)
    qtbot.addWidget(widget)
    select_segmentation(widget)

    layer.selected_label = 1

    assert widget.selected_instance_id == 1
    assert widget.apply_class_button.isEnabled()
    assert "Current instance_id: 1." in widget.selection_status.text()


def test_widget_automatically_enables_pick_mode_for_bound_labels_layer(qtbot, sdata_blobs: SpatialData) -> None:
    layer = make_blobs_labels_layer(sdata_blobs)
    viewer = DummyViewer(layers=[layer])

    widget = HarpyWidget(viewer)
    qtbot.addWidget(widget)
    select_segmentation(widget)

    assert str(layer.mode) == "pick"
    assert viewer.layers.selection.active is layer


def test_widget_picks_multiscale_labels_layers_without_napari_pick_mode(qtbot, sdata_blobs: SpatialData) -> None:
    layer = make_multiscale_blobs_labels_layer(sdata_blobs)
    viewer = DummyViewer(layers=[layer])

    widget = HarpyWidget(viewer)
    qtbot.addWidget(widget)
    select_segmentation(widget)

    assert str(layer.mode) == "pan_zoom"
    assert viewer.layers.selection.active is layer

    coords = tuple(float(value) for value in np.argwhere(np.asarray(sdata_blobs.labels["blobs_labels"]) == 5)[0])
    event = SimpleNamespace(position=coords, view_direction=None, dims_displayed=[0, 1])
    layer.mouse_drag_callbacks[-1](layer, event)

    assert widget.selected_instance_id == 5
    assert widget.apply_class_button.isEnabled()

    widget.class_spinbox.setValue(7)
    widget.apply_class_button.click()

    table = sdata_blobs["table"]
    mask = (table.obs["region"] == "blobs_labels") & (table.obs["instance_id"] == 5)

    assert table.obs.loc[mask, USER_CLASS_COLUMN].tolist() == [7]
    assert "Assigned class 7" in widget.annotation_feedback.text()


def test_widget_auto_loads_selected_segmentation_when_it_is_not_yet_loaded(qtbot, sdata_blobs: SpatialData) -> None:
    layer = make_blobs_labels_layer(sdata_blobs)
    viewer = DummyViewer(layers=[layer])

    widget = HarpyWidget(viewer)
    qtbot.addWidget(widget)

    widget.segmentation_combo.setCurrentIndex(1)
    layer.selected_label = 9

    assert widget.selected_segmentation_name == "blobs_multiscale_labels"
    assert widget.selected_instance_id is None
    assert len(viewer.layers) == 2
    assert viewer.layers[-1].name == "blobs_multiscale_labels"
    assert viewer.layers.selection.active is viewer.layers[-1]
    assert widget._annotation_controller.labels_layer is viewer.layers[-1]
    assert widget._viewer_styling_controller.labels_layer is viewer.layers[-1]
    assert "Loaded segmentation `blobs_multiscale_labels` in coordinate system `global`." in (
        widget.selection_status.text()
    )
    assert "This segmentation is loaded, but no annotation table is linked to it." in widget.selection_status.text()


def test_widget_clears_selected_segmentation_after_manual_layer_removal(qtbot, sdata_blobs: SpatialData) -> None:
    layer = make_blobs_labels_layer(sdata_blobs)
    viewer = DummyViewer(layers=[layer])

    widget = HarpyWidget(viewer)
    qtbot.addWidget(widget)
    select_segmentation(widget)

    assert widget._annotation_controller.labels_layer is layer
    assert widget._viewer_styling_controller.labels_layer is layer

    viewer.layers.remove(layer)
    viewer.layers.selection.active = None
    viewer.layers.events.removed.emit(layer)

    assert widget.segmentation_combo.currentIndex() == -1
    assert widget.selected_segmentation_name is None
    assert len(viewer.layers) == 0
    assert widget._annotation_controller.labels_layer is None
    assert widget._viewer_styling_controller.labels_layer is None
    assert widget.selected_table_name is None
    assert widget.selected_feature_key is None
    assert "Choose a segmentation mask" in widget.selection_status.text()


def test_widget_ignores_unrelated_labels_layer_removal(qtbot, monkeypatch, sdata_blobs: SpatialData) -> None:
    primary_layer = make_blobs_labels_layer(sdata_blobs)
    viewer = DummyViewer(layers=[primary_layer])

    widget = HarpyWidget(viewer)
    qtbot.addWidget(widget)

    widget.segmentation_combo.setCurrentIndex(1)
    selected_layer = viewer.layers[-1]

    bind_calls: list[str | None] = []

    def _record_bind(*, classifier_dirty_reason: str | None = None) -> None:
        bind_calls.append(classifier_dirty_reason)

    monkeypatch.setattr(widget, "_bind_current_selection", _record_bind)

    viewer.layers.remove(primary_layer)
    viewer.layers.selection.active = selected_layer
    viewer.layers.events.removed.emit(primary_layer)

    assert widget.selected_segmentation_name == "blobs_multiscale_labels"
    assert widget._annotation_controller.labels_layer is selected_layer
    assert widget._viewer_styling_controller.labels_layer is selected_layer
    assert bind_calls == []


def test_widget_handles_tables_without_obsm_entries(qtbot, sdata_blobs: SpatialData) -> None:
    table = sdata_blobs["table"]
    for key in list(table.obsm.keys()):
        del table.obsm[key]

    layer = make_blobs_labels_layer(sdata_blobs)
    viewer = DummyViewer(layers=[layer])

    widget = HarpyWidget(viewer)
    qtbot.addWidget(widget)
    select_segmentation(widget)

    assert widget.table_combo.count() == 1
    assert widget.table_combo.itemText(0) == "table"
    assert widget.feature_matrix_combo.count() == 0
    assert not widget.feature_matrix_combo.isEnabled()
    assert widget.selected_table_name == "table"
    assert widget.selected_feature_key is None
    assert not widget.validation_status.isHidden()
    assert "does not contain any feature matrices in `.obsm`" in widget.validation_status.text()


def test_widget_applies_user_class_to_picked_instance(qtbot, sdata_blobs: SpatialData) -> None:
    layer = make_blobs_labels_layer(sdata_blobs)
    viewer = DummyViewer(layers=[layer])

    widget = HarpyWidget(viewer)
    qtbot.addWidget(widget)
    select_segmentation(widget)

    layer.selected_label = 5
    widget.class_spinbox.setValue(3)
    widget.apply_class_button.click()

    table = sdata_blobs["table"]
    mask = (table.obs["region"] == "blobs_labels") & (table.obs["instance_id"] == 5)

    assert USER_CLASS_COLUMN in table.obs
    assert isinstance(table.obs[USER_CLASS_COLUMN].dtype, pd.CategoricalDtype)
    assert list(table.obs[USER_CLASS_COLUMN].cat.categories) == [0, 3]
    assert table.obs.loc[mask, USER_CLASS_COLUMN].tolist() == [3]
    assert int(table.obs.loc[table.obs["instance_id"] == 6, USER_CLASS_COLUMN].iloc[0]) == 0
    assert table.uns[USER_CLASS_COLORS_KEY] == default_class_colors([0, 3])
    assert "adata" not in layer.metadata
    assert "Current class: 3." in widget.selection_status.text()
    assert "Assigned class 3" in widget.annotation_feedback.text()


def test_widget_apply_shortcut_applies_user_class_to_picked_instance(qtbot, sdata_blobs: SpatialData) -> None:
    layer = make_blobs_labels_layer(sdata_blobs)
    viewer = DummyViewer(layers=[layer])

    widget = HarpyWidget(viewer)
    qtbot.addWidget(widget)
    select_segmentation(widget)

    layer.selected_label = 5
    widget.class_spinbox.setValue(4)

    assert "Shortcut: A." in widget.apply_class_button.toolTip()
    assert widget._annotation_shortcuts[0].key().toString() == "A"

    widget._annotation_shortcuts[0].activated.emit()

    table = sdata_blobs["table"]
    mask = (table.obs["region"] == "blobs_labels") & (table.obs["instance_id"] == 5)

    assert table.obs.loc[mask, USER_CLASS_COLUMN].tolist() == [4]
    assert "Assigned class 4" in widget.annotation_feedback.text()


def test_widget_uses_table_instance_key_name_in_status_and_annotation_feedback(qtbot, sdata_blobs: SpatialData) -> None:
    rename_table_instance_key(sdata_blobs, instance_key="cell_id")

    layer = make_blobs_labels_layer(sdata_blobs)
    viewer = DummyViewer(layers=[layer])

    widget = HarpyWidget(viewer)
    qtbot.addWidget(widget)
    select_segmentation(widget)

    layer.selected_label = 5
    widget.class_spinbox.setValue(3)
    widget.apply_class_button.click()

    assert "Current cell_id: 5." in widget.selection_status.text()
    assert "Assigned class 3 to cell_id 5." in widget.annotation_feedback.text()


def test_widget_can_clear_user_class_for_picked_instance(qtbot, sdata_blobs: SpatialData) -> None:
    layer = make_blobs_labels_layer(sdata_blobs)
    viewer = DummyViewer(layers=[layer])

    widget = HarpyWidget(viewer)
    qtbot.addWidget(widget)
    select_segmentation(widget)

    layer.selected_label = 5
    widget.class_spinbox.setValue(2)
    widget.apply_class_button.click()
    widget.clear_class_button.click()

    table = sdata_blobs["table"]
    mask = (table.obs["region"] == "blobs_labels") & (table.obs["instance_id"] == 5)

    assert isinstance(table.obs[USER_CLASS_COLUMN].dtype, pd.CategoricalDtype)
    assert list(table.obs[USER_CLASS_COLUMN].cat.categories) == [0]
    assert table.obs.loc[mask, USER_CLASS_COLUMN].tolist() == [0]
    assert table.uns[USER_CLASS_COLORS_KEY] == default_class_colors([0])
    assert "Current class: unlabeled." in widget.selection_status.text()
    assert "Cleared the user class" in widget.annotation_feedback.text()


def test_widget_clear_shortcut_clears_user_class_for_picked_instance(qtbot, sdata_blobs: SpatialData) -> None:
    layer = make_blobs_labels_layer(sdata_blobs)
    viewer = DummyViewer(layers=[layer])

    widget = HarpyWidget(viewer)
    qtbot.addWidget(widget)
    select_segmentation(widget)

    layer.selected_label = 5
    widget.class_spinbox.setValue(2)
    widget.apply_class_button.click()

    assert "Shortcut: R." in widget.clear_class_button.toolTip()
    assert widget._annotation_shortcuts[1].key().toString() == "R"

    widget._annotation_shortcuts[1].activated.emit()

    table = sdata_blobs["table"]
    mask = (table.obs["region"] == "blobs_labels") & (table.obs["instance_id"] == 5)

    assert table.obs.loc[mask, USER_CLASS_COLUMN].tolist() == [0]
    assert "Cleared the user class" in widget.annotation_feedback.text()


def test_widget_warns_when_selected_label_is_missing_from_annotation_table(
    qtbot, monkeypatch, sdata_blobs: SpatialData
) -> None:
    rename_table_instance_key(sdata_blobs, instance_key="cell_id")
    table = sdata_blobs["table"]
    keep_mask = ~((table.obs["region"] == "blobs_labels") & (table.obs["cell_id"] == 5))
    table._inplace_subset_obs(keep_mask.to_numpy())

    warnings: list[str] = []

    class DummyLogger:
        def warning(self, message: str) -> None:
            warnings.append(message)

    monkeypatch.setattr(annotation_module, "logger", DummyLogger())

    layer = make_blobs_labels_layer(sdata_blobs)
    viewer = DummyViewer(layers=[layer])
    widget = HarpyWidget(viewer)
    qtbot.addWidget(widget)
    select_segmentation(widget)

    layer.selected_label = 5

    assert not widget.apply_class_button.isEnabled()
    assert "Selected cell_id 5 is not present in annotation table" in widget.selection_status.text()
    assert "cannot receive a user class" in widget.selection_status.text()

    widget.class_spinbox.setValue(3)
    widget._apply_current_class()

    assert "Selected cell_id 5 is not present in annotation table" in widget.annotation_feedback.text()
    assert "#b45309" in widget.annotation_feedback.styleSheet()
    assert USER_CLASS_COLUMN not in table.obs
    assert warnings == [widget._annotation_controller.missing_table_row_message]
    assert warnings[0] in widget.annotation_feedback.text()


def test_widget_recolors_layer_from_user_class_annotations(qtbot, sdata_blobs: SpatialData) -> None:
    layer = make_blobs_labels_layer(sdata_blobs)
    viewer = DummyViewer(layers=[layer])

    widget = HarpyWidget(viewer)
    qtbot.addWidget(widget)
    select_segmentation(widget)

    layer.selected_label = 5
    widget.class_spinbox.setValue(4)
    widget.apply_class_button.click()

    assert isinstance(layer.colormap, DirectLabelColormap)
    assert np.allclose(layer.colormap.color_dict[0], np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32))
    assert layer.colormap.color_dict[5][3] > 0
    assert layer.colormap.color_dict[6][3] > 0
    assert not np.allclose(layer.colormap.color_dict[5], layer.colormap.color_dict[6])
    assert "instance_id" in layer.features.columns
    assert USER_CLASS_COLUMN in layer.features.columns
    assert layer.features.set_index("index").loc[5, "instance_id"] == 5
    assert layer.features.set_index("index").loc[5, USER_CLASS_COLUMN] == 4


def test_widget_does_not_log_warning_when_existing_user_class_colors_are_overwritten(
    qtbot, monkeypatch, sdata_blobs: SpatialData
) -> None:
    table = sdata_blobs["table"]
    table.obs[USER_CLASS_COLUMN] = pd.Categorical([0] * table.n_obs, categories=[0])
    table.uns[USER_CLASS_COLORS_KEY] = ["#ffffffff"]

    layer = make_blobs_labels_layer(sdata_blobs)
    viewer = DummyViewer(layers=[layer])
    warnings: list[str] = []

    class DummyLogger:
        def warning(self, message: str) -> None:
            warnings.append(message)

    monkeypatch.setattr(class_palette_module, "logger", DummyLogger())
    widget = HarpyWidget(viewer)
    qtbot.addWidget(widget)
    select_segmentation(widget)

    assert warnings == []


def test_widget_enables_sync_for_backed_spatialdata(qtbot, backed_sdata_blobs: SpatialData) -> None:
    layer = make_blobs_labels_layer(backed_sdata_blobs)
    viewer = DummyViewer(layers=[layer])

    widget = HarpyWidget(viewer)
    qtbot.addWidget(widget)
    select_segmentation(widget)
    expected_table_path = Path(backed_sdata_blobs.path) / "tables" / "table"
    sync_tooltip = unescape(widget.sync_button.toolTip()).replace("&#8203;", "").replace("\u200b", "")
    reload_tooltip = unescape(widget.reload_button.toolTip()).replace("&#8203;", "").replace("\u200b", "")

    assert widget.sync_button.isEnabled()
    assert widget.reload_button.isEnabled()
    assert f"Write the current in-memory `table` table state to `{expected_table_path}`." in sync_tooltip
    assert f"Discard the current in-memory `table` table state and reload it from `{expected_table_path}`." in (
        reload_tooltip
    )


def test_widget_marks_persistence_dirty_on_annotation_change_and_clears_it_on_sync(
    qtbot, monkeypatch, backed_sdata_blobs: SpatialData
) -> None:
    layer = make_blobs_labels_layer(backed_sdata_blobs)
    viewer = DummyViewer(layers=[layer])

    widget = HarpyWidget(viewer)
    qtbot.addWidget(widget)
    select_segmentation(widget)
    monkeypatch.setattr(widget._classifier_controller, "schedule_retrain", lambda *args, **kwargs: False)

    layer.selected_label = 5
    widget.class_spinbox.setValue(3)
    widget.apply_class_button.click()
    sync_tooltip = unescape(widget.sync_button.toolTip()).replace("&#8203;", "").replace("\u200b", "")
    reload_tooltip = unescape(widget.reload_button.toolTip()).replace("&#8203;", "").replace("\u200b", "")

    assert widget._persistence_controller.is_dirty is True
    assert "Unsynced local in-memory table changes are present." in sync_tooltip
    assert "Unsynced local in-memory table changes would be discarded." in reload_tooltip

    widget.sync_button.click()
    sync_tooltip = unescape(widget.sync_button.toolTip()).replace("&#8203;", "").replace("\u200b", "")
    reload_tooltip = unescape(widget.reload_button.toolTip()).replace("&#8203;", "").replace("\u200b", "")

    assert widget._persistence_controller.is_dirty is False
    assert "Unsynced local in-memory table changes are present." not in sync_tooltip
    assert "Unsynced local in-memory table changes would be discarded." not in reload_tooltip


def test_widget_syncs_user_class_to_backed_zarr(qtbot, backed_sdata_blobs: SpatialData) -> None:
    layer = make_blobs_labels_layer(backed_sdata_blobs)
    viewer = DummyViewer(layers=[layer])

    widget = HarpyWidget(viewer)
    qtbot.addWidget(widget)
    select_segmentation(widget)
    expected_table_path = Path(backed_sdata_blobs.path) / "tables" / "table"

    layer.selected_label = 5
    widget.class_spinbox.setValue(3)
    widget.apply_class_button.click()
    widget.sync_button.click()

    reread = read_zarr(backed_sdata_blobs.path)
    mask = (reread["table"].obs["region"] == "blobs_labels") & (reread["table"].obs["instance_id"] == 5)

    assert widget.sync_button.isEnabled()
    assert widget.reload_button.isEnabled()
    _assert_persistence_success_feedback(widget, f"Wrote `table` table state to `{expected_table_path}`.")
    assert isinstance(reread["table"].obs[USER_CLASS_COLUMN].dtype, pd.CategoricalDtype)
    assert list(reread["table"].obs[USER_CLASS_COLUMN].cat.categories) == [0, 3]
    assert reread["table"].obs.loc[mask, USER_CLASS_COLUMN].tolist() == [3]
    assert list(reread["table"].uns[USER_CLASS_COLORS_KEY]) == default_class_colors([0, 3])


def test_widget_marks_persistence_dirty_after_classifier_writes_results(qtbot, backed_sdata_blobs: SpatialData) -> None:
    table = backed_sdata_blobs["table"]
    instance_ids = table.obs["instance_id"].to_numpy(dtype=np.int64)
    table.obsm["features_1"] = np.column_stack(
        [
            (instance_ids > 13).astype(np.float64),
            instance_ids.astype(np.float64) / instance_ids.max(),
        ]
    )
    table.obs[USER_CLASS_COLUMN] = pd.Categorical(
        [1 if int(instance_id) in {1, 2} else 2 if int(instance_id) in {24, 25} else 0 for instance_id in instance_ids],
        categories=[0, 1, 2],
    )

    layer = make_blobs_labels_layer(backed_sdata_blobs)
    viewer = DummyViewer(layers=[layer])
    widget = HarpyWidget(viewer)
    qtbot.addWidget(widget)
    select_segmentation(widget)

    assert widget._persistence_controller.is_dirty is False

    widget.retrain_button.click()
    qtbot.waitUntil(
        lambda: widget._persistence_controller.is_dirty and table.obs[PRED_CLASS_COLUMN].astype("string").ne("0").any(),
        timeout=5000,
    )
    sync_tooltip = unescape(widget.sync_button.toolTip()).replace("&#8203;", "").replace("\u200b", "")
    reload_tooltip = unescape(widget.reload_button.toolTip()).replace("&#8203;", "").replace("\u200b", "")

    assert widget._persistence_controller.is_dirty is True
    assert "Unsynced local in-memory table changes are present." in sync_tooltip
    assert "Unsynced local in-memory table changes would be discarded." in reload_tooltip


def test_widget_cancels_dirty_reload_when_user_chooses_cancel(
    qtbot, monkeypatch, backed_sdata_blobs: SpatialData
) -> None:
    layer = make_blobs_labels_layer(backed_sdata_blobs)
    viewer = DummyViewer(layers=[layer])
    widget = HarpyWidget(viewer)
    qtbot.addWidget(widget)
    select_segmentation(widget)
    monkeypatch.setattr(widget._classifier_controller, "schedule_retrain", lambda *args, **kwargs: False)

    table = backed_sdata_blobs["table"]
    disk_obs = table.obs.copy()
    disk_obs[USER_CLASS_COLUMN] = pd.Categorical([0] * table.n_obs, categories=[0])
    _write_disk_table_state(backed_sdata_blobs, obs=disk_obs, obsm=dict(table.obsm), uns=dict(table.uns))

    layer.selected_label = 5
    widget.class_spinbox.setValue(3)
    widget.apply_class_button.click()
    monkeypatch.setattr(
        widget,
        "_prompt_dirty_reload_decision",
        lambda: widget_module._DirtyReloadDecision.CANCEL,
    )

    widget.reload_button.click()

    mask = (table.obs["region"] == "blobs_labels") & (table.obs["instance_id"] == 5)
    reread = read_zarr(backed_sdata_blobs.path)
    disk_mask = (reread["table"].obs["region"] == "blobs_labels") & (reread["table"].obs["instance_id"] == 5)

    assert widget._persistence_controller.is_dirty is True
    assert table.obs.loc[mask, USER_CLASS_COLUMN].tolist() == [3]
    assert reread["table"].obs.loc[disk_mask, USER_CLASS_COLUMN].tolist() == [0]


def test_widget_dirty_reload_can_write_then_reload(qtbot, monkeypatch, backed_sdata_blobs: SpatialData) -> None:
    layer = make_blobs_labels_layer(backed_sdata_blobs)
    viewer = DummyViewer(layers=[layer])
    widget = HarpyWidget(viewer)
    qtbot.addWidget(widget)
    select_segmentation(widget)
    monkeypatch.setattr(widget._classifier_controller, "schedule_retrain", lambda *args, **kwargs: False)

    expected_table_path = Path(backed_sdata_blobs.path) / "tables" / "table"
    table = backed_sdata_blobs["table"]

    layer.selected_label = 5
    widget.class_spinbox.setValue(3)
    widget.apply_class_button.click()
    monkeypatch.setattr(
        widget,
        "_prompt_dirty_reload_decision",
        lambda: widget_module._DirtyReloadDecision.WRITE,
    )

    widget.reload_button.click()

    reread = read_zarr(backed_sdata_blobs.path)
    mask = (table.obs["region"] == "blobs_labels") & (table.obs["instance_id"] == 5)
    disk_mask = (reread["table"].obs["region"] == "blobs_labels") & (reread["table"].obs["instance_id"] == 5)

    assert widget._persistence_controller.is_dirty is False
    assert table.obs.loc[mask, USER_CLASS_COLUMN].tolist() == [3]
    assert reread["table"].obs.loc[disk_mask, USER_CLASS_COLUMN].tolist() == [3]
    _assert_persistence_success_feedback(
        widget,
        f"Wrote local changes and reloaded `table` table state from `{expected_table_path}`.",
    )


def test_widget_dirty_reload_can_discard_local_edits(qtbot, monkeypatch, backed_sdata_blobs: SpatialData) -> None:
    layer = make_blobs_labels_layer(backed_sdata_blobs)
    viewer = DummyViewer(layers=[layer])
    widget = HarpyWidget(viewer)
    qtbot.addWidget(widget)
    select_segmentation(widget)
    monkeypatch.setattr(widget._classifier_controller, "schedule_retrain", lambda *args, **kwargs: False)

    expected_table_path = Path(backed_sdata_blobs.path) / "tables" / "table"
    table = backed_sdata_blobs["table"]
    disk_obs = table.obs.copy()
    disk_obs[USER_CLASS_COLUMN] = pd.Categorical([0] * table.n_obs, categories=[0])
    _write_disk_table_state(backed_sdata_blobs, obs=disk_obs, obsm=dict(table.obsm), uns=dict(table.uns))

    layer.selected_label = 5
    widget.class_spinbox.setValue(3)
    widget.apply_class_button.click()
    monkeypatch.setattr(
        widget,
        "_prompt_dirty_reload_decision",
        lambda: widget_module._DirtyReloadDecision.RELOAD_DISCARD,
    )

    widget.reload_button.click()

    mask = (table.obs["region"] == "blobs_labels") & (table.obs["instance_id"] == 5)
    reread = read_zarr(backed_sdata_blobs.path)
    disk_mask = (reread["table"].obs["region"] == "blobs_labels") & (reread["table"].obs["instance_id"] == 5)

    assert widget._persistence_controller.is_dirty is False
    assert table.obs.loc[mask, USER_CLASS_COLUMN].tolist() == [0]
    assert reread["table"].obs.loc[disk_mask, USER_CLASS_COLUMN].tolist() == [0]
    _assert_persistence_success_feedback(widget, f"Reloaded `table` table state from `{expected_table_path}`.")


def test_widget_reloads_table_state_from_backed_zarr(qtbot, backed_sdata_blobs: SpatialData) -> None:
    layer = make_blobs_labels_layer(backed_sdata_blobs)
    viewer = DummyViewer(layers=[layer])

    widget = HarpyWidget(viewer)
    qtbot.addWidget(widget)
    select_segmentation(widget)
    expected_table_path = Path(backed_sdata_blobs.path) / "tables" / "table"
    table = backed_sdata_blobs["table"]

    obs = table.obs.copy()
    obs[USER_CLASS_COLUMN] = pd.Categorical(
        [0] * (table.n_obs - 1) + [7],
        categories=[0, 7],
    )
    obsm = dict(table.obsm)
    obsm["disk_features"] = np.arange(table.n_obs, dtype=np.float64).reshape(table.n_obs, 1)
    uns = dict(table.uns)
    _write_disk_table_state(backed_sdata_blobs, obs=obs, obsm=obsm, uns=uns)

    layer.selected_label = int(table.obs["instance_id"].iloc[-1])
    widget.reload_button.click()

    mask = (table.obs["region"] == "blobs_labels") & (
        table.obs["instance_id"] == int(table.obs["instance_id"].iloc[-1])
    )

    _assert_persistence_success_feedback(widget, f"Reloaded `table` table state from `{expected_table_path}`.")
    assert isinstance(table.obs[USER_CLASS_COLUMN].dtype, pd.CategoricalDtype)
    assert list(table.obs[USER_CLASS_COLUMN].cat.categories) == [0, 7]
    assert table.obs.loc[mask, USER_CLASS_COLUMN].tolist() == [7]
    assert "disk_features" in table.obsm
    feature_matrix_items = [
        widget.feature_matrix_combo.itemText(index) for index in range(widget.feature_matrix_combo.count())
    ]
    assert feature_matrix_items == ["disk_features", "features_1", "features_2"]
    assert widget.selected_feature_key == "features_1"
    assert "Current class: 7." in widget.selection_status.text()


def test_widget_reload_falls_back_when_selected_feature_key_disappears(qtbot, backed_sdata_blobs: SpatialData) -> None:
    layer = make_blobs_labels_layer(backed_sdata_blobs)
    viewer = DummyViewer(layers=[layer])

    widget = HarpyWidget(viewer)
    qtbot.addWidget(widget)
    select_segmentation(widget)
    expected_table_path = Path(backed_sdata_blobs.path) / "tables" / "table"
    table = backed_sdata_blobs["table"]

    widget.feature_matrix_combo.setCurrentIndex(1)

    assert widget.selected_feature_key == "features_2"

    obs = table.obs.copy()
    obsm = {"features_1": table.obsm["features_1"]}
    uns = dict(table.uns)
    _write_disk_table_state(backed_sdata_blobs, obs=obs, obsm=obsm, uns=uns)

    widget.reload_button.click()

    feature_matrix_items = [
        widget.feature_matrix_combo.itemText(index) for index in range(widget.feature_matrix_combo.count())
    ]

    _assert_persistence_success_feedback(widget, f"Reloaded `table` table state from `{expected_table_path}`.")
    assert feature_matrix_items == ["features_1"]
    assert widget.selected_feature_key == "features_1"
    assert "features_2" not in table.obsm


def test_widget_reload_freezes_classifier_worker_and_ignores_late_results(
    qtbot, monkeypatch, backed_sdata_blobs: SpatialData
) -> None:
    table = backed_sdata_blobs["table"]
    instance_ids = table.obs["instance_id"].to_numpy(dtype=np.int64)
    table.obsm["features_1"] = np.column_stack(
        [
            (instance_ids > 13).astype(np.float64),
            instance_ids.astype(np.float64) / instance_ids.max(),
        ]
    )
    table.obs[USER_CLASS_COLUMN] = pd.Categorical(
        [1 if int(instance_id) in {1, 2} else 2 if int(instance_id) in {24, 25} else 0 for instance_id in instance_ids],
        categories=[0, 1, 2],
    )

    layer = make_blobs_labels_layer(backed_sdata_blobs)
    viewer = DummyViewer(layers=[layer])
    widget = HarpyWidget(viewer)
    qtbot.addWidget(widget)
    select_segmentation(widget)
    workers: list[_DeferredWorker] = []

    def fake_create_training_worker(job):
        result = classifier_module.ClassifierJobResult(
            job_id=job.job_id,
            feature_key=job.feature_key,
            label_name=job.label_name,
            table_name=job.table_name,
            pred_classes=np.full(job.prediction_scope.table_row_positions.shape, 1, dtype=np.int64),
            pred_confidences=np.full(job.prediction_scope.table_row_positions.shape, 0.91, dtype=np.float64),
            trained_at="2026-04-13T09:00:00+00:00",
            model_params=dict(classifier_module.RANDOM_FOREST_PARAMS),
            summary=job.summary,
        )
        worker = _DeferredWorker(result)
        workers.append(worker)
        return worker

    monkeypatch.setattr(widget._classifier_controller, "_create_training_worker", fake_create_training_worker)

    widget.retrain_button.click()

    assert len(workers) == 1
    assert workers[0].started is True
    assert widget._classifier_controller.is_training is True

    expected_table_path = Path(backed_sdata_blobs.path) / "tables" / "table"
    obs = table.obs.copy()
    obs[PRED_CLASS_COLUMN] = pd.Categorical(np.full(table.n_obs, 7, dtype=np.int64), categories=[0, 7])
    obs[PRED_CONFIDENCE_COLUMN] = pd.Series(np.full(table.n_obs, 0.77), index=obs.index, dtype="float64")
    obsm = dict(table.obsm)
    uns = dict(table.uns)
    uns[CLASSIFIER_CONFIG_KEY] = {
        "model_type": "RandomForestClassifier",
        "feature_key": "features_1",
        "table_name": "table",
        "roi_mode": "none",
        "trained": True,
        "eligible": True,
        "reason": "Ready to train.",
        "training_timestamp": "2026-04-13T09:00:00+00:00",
        "n_labeled_objects": 4,
        "n_features": 2,
        "class_labels_seen": [1, 2],
        "rf_params": dict(classifier_module.RANDOM_FOREST_PARAMS),
        "training_scope": "all",
        "training_regions": ["blobs_labels"],
        "n_training_rows": int(table.n_obs),
        "prediction_scope": "selected_segmentation_only",
        "prediction_regions": ["blobs_labels"],
        "n_predicted_rows": int(table.n_obs),
    }
    _write_disk_table_state(backed_sdata_blobs, obs=obs, obsm=obsm, uns=uns)

    widget.reload_button.click()

    assert workers[0].quit_called is True
    assert widget._classifier_controller.is_training is False
    assert widget._classifier_controller.is_dirty is False
    _assert_persistence_success_feedback(widget, f"Reloaded `table` table state from `{expected_table_path}`.")
    assert table.obs[PRED_CLASS_COLUMN].eq(7).all()
    assert table.obs[PRED_CONFIDENCE_COLUMN].eq(0.77).all()
    assert "Loaded predictions for" in widget.classifier_feedback.text()
    assert len(workers) == 1

    workers[0].emit_returned()

    assert table.obs[PRED_CLASS_COLUMN].eq(7).all()
    assert table.obs[PRED_CONFIDENCE_COLUMN].eq(0.77).all()
    assert "Loaded predictions for" in widget.classifier_feedback.text()


def test_widget_retrain_button_recovers_after_worker_finishes(qtbot, monkeypatch, sdata_blobs: SpatialData) -> None:
    table = sdata_blobs["table"]
    instance_ids = table.obs["instance_id"].to_numpy(dtype=np.int64)
    table.obsm["features_1"] = np.column_stack(
        [
            (instance_ids > 13).astype(np.float64),
            instance_ids.astype(np.float64) / instance_ids.max(),
        ]
    )
    table.obs[USER_CLASS_COLUMN] = pd.Categorical(
        [1 if int(instance_id) in {1, 2} else 2 if int(instance_id) in {24, 25} else 0 for instance_id in instance_ids],
        categories=[0, 1, 2],
    )

    layer = make_blobs_labels_layer(sdata_blobs)
    viewer = DummyViewer(layers=[layer])
    widget = HarpyWidget(viewer)
    qtbot.addWidget(widget)
    select_segmentation(widget)
    workers: list[_DeferredWorker] = []

    def fake_create_training_worker(job):
        result = classifier_module.ClassifierJobResult(
            job_id=job.job_id,
            feature_key=job.feature_key,
            label_name=job.label_name,
            table_name=job.table_name,
            pred_classes=np.full(job.prediction_scope.table_row_positions.shape, 1, dtype=np.int64),
            pred_confidences=np.full(job.prediction_scope.table_row_positions.shape, 0.91, dtype=np.float64),
            trained_at="2026-04-13T09:00:00+00:00",
            model_params=dict(classifier_module.RANDOM_FOREST_PARAMS),
            summary=job.summary,
        )
        worker = _DeferredWorker(result)
        workers.append(worker)
        return worker

    monkeypatch.setattr(widget._classifier_controller, "_create_training_worker", fake_create_training_worker)

    widget.retrain_button.click()

    assert len(workers) == 1
    assert widget._classifier_controller.is_training is True
    assert widget.retrain_button.isEnabled() is False
    assert "currently running" in widget.retrain_button.toolTip()

    workers[0].emit_returned()

    qtbot.waitUntil(lambda: widget._classifier_controller.is_training is False, timeout=1000)
    qtbot.waitUntil(lambda: widget.retrain_button.isEnabled(), timeout=1000)

    assert "currently running" not in widget.retrain_button.toolTip()
    assert "write predictions for the selected prediction scope" in widget.retrain_button.toolTip()
    assert "model is up to date" in widget.classifier_feedback.text()


def test_widget_retrains_classifier_after_annotation_changes(qtbot, sdata_blobs: SpatialData) -> None:
    table = sdata_blobs["table"]
    instance_ids = table.obs["instance_id"].to_numpy(dtype=np.int64)
    table.obsm["features_1"] = np.column_stack(
        [
            (instance_ids > 13).astype(np.float64),
            instance_ids.astype(np.float64) / instance_ids.max(),
        ]
    )

    layer = make_blobs_labels_layer(sdata_blobs)
    viewer = DummyViewer(layers=[layer])
    widget = HarpyWidget(viewer)
    qtbot.addWidget(widget)
    select_segmentation(widget)

    layer.selected_label = 1
    widget.class_spinbox.setValue(1)
    widget.apply_class_button.click()

    layer.selected_label = 24
    widget.class_spinbox.setValue(2)
    widget.apply_class_button.click()

    qtbot.waitUntil(lambda: table.obs[PRED_CLASS_COLUMN].astype("string").ne("0").any(), timeout=5000)

    pred_class = table.obs.set_index("instance_id")[PRED_CLASS_COLUMN]
    assert isinstance(table.obs[PRED_CLASS_COLUMN].dtype, pd.CategoricalDtype)
    assert list(table.obs[PRED_CLASS_COLUMN].cat.categories) == [0, 1, 2]
    assert table.uns[PRED_CLASS_COLORS_KEY] == default_class_colors([0, 1, 2])
    assert pred_class.loc[1] == 1
    assert pred_class.loc[24] == 2
    assert "adata" not in layer.metadata
    assert "model is up to date" in widget.classifier_feedback.text()
    assert table.uns[CLASSIFIER_CONFIG_KEY]["trained"] is True


def test_widget_colors_predictions_using_pred_class_palette_in_pred_class_mode(qtbot, sdata_blobs: SpatialData) -> None:
    table = sdata_blobs["table"]
    instance_ids = table.obs["instance_id"].to_numpy(dtype=np.int64)
    table.obsm["features_1"] = np.column_stack(
        [
            (instance_ids > 13).astype(np.float64),
            instance_ids.astype(np.float64) / instance_ids.max(),
        ]
    )

    layer = make_blobs_labels_layer(sdata_blobs)
    viewer = DummyViewer(layers=[layer])
    widget = HarpyWidget(viewer)
    qtbot.addWidget(widget)
    select_segmentation(widget)

    layer.selected_label = 1
    widget.class_spinbox.setValue(1)
    widget.apply_class_button.click()

    layer.selected_label = 24
    widget.class_spinbox.setValue(2)
    widget.apply_class_button.click()

    qtbot.waitUntil(lambda: table.obs[PRED_CLASS_COLUMN].astype("string").ne("0").any(), timeout=5000)

    table.uns[USER_CLASS_COLORS_KEY] = ["#80808099", "#ff0000", "#00ff00"]
    table.uns[PRED_CLASS_COLORS_KEY] = ["#80808099", "#0000ff", "#ffff00"]

    assert not np.allclose(layer.colormap.color_dict[1], layer.colormap.color_dict[5])

    widget.color_by_combo.setCurrentIndex(widget.color_by_combo.findData("pred_class"))

    assert np.allclose(layer.colormap.color_dict[1], layer.colormap.color_dict[5])
    assert np.allclose(layer.colormap.color_dict[24], layer.colormap.color_dict[26])
    assert np.allclose(layer.colormap.color_dict[1], np.asarray(to_rgba("#0000ff"), dtype=np.float32))
    assert np.allclose(layer.colormap.color_dict[24], np.asarray(to_rgba("#ffff00"), dtype=np.float32))
    assert PRED_CLASS_COLUMN in layer.features.columns


def test_widget_colors_confidence_continuously_in_pred_confidence_mode(qtbot, sdata_blobs: SpatialData) -> None:
    table = sdata_blobs["table"]
    table.obs[PRED_CONFIDENCE_COLUMN] = pd.Series(
        np.linspace(0.0, 1.0, table.n_obs),
        index=table.obs.index,
        dtype="float64",
    )

    layer = make_blobs_labels_layer(sdata_blobs)
    viewer = DummyViewer(layers=[layer])
    widget = HarpyWidget(viewer)
    qtbot.addWidget(widget)
    select_segmentation(widget)

    widget.color_by_combo.setCurrentIndex(widget.color_by_combo.findData("pred_confidence"))

    assert isinstance(layer.colormap, DirectLabelColormap)
    assert not np.allclose(layer.colormap.color_dict[1], layer.colormap.color_dict[24])
    assert PRED_CONFIDENCE_COLUMN in layer.features.columns


def test_widget_exposes_label_metadata_in_napari_status_bar(qtbot, sdata_blobs: SpatialData) -> None:
    table = sdata_blobs["table"]
    mask = (table.obs["region"] == "blobs_labels") & (table.obs["instance_id"] == 5)
    table.obs.loc[mask, USER_CLASS_COLUMN] = pd.Categorical([4], categories=[0, 4])[0]
    table.obs.loc[mask, PRED_CLASS_COLUMN] = 2
    table.obs.loc[mask, PRED_CONFIDENCE_COLUMN] = 0.95

    layer = make_blobs_labels_layer(sdata_blobs)
    viewer = DummyViewer(layers=[layer])
    widget = HarpyWidget(viewer)
    qtbot.addWidget(widget)
    select_segmentation(widget)

    coords = tuple(float(value) for value in np.argwhere(np.asarray(sdata_blobs.labels["blobs_labels"]) == 5)[0])
    status = layer.get_status(position=coords, view_direction=np.array([1.0, 0.0]), dims_displayed=[0, 1])

    assert "instance_id: 5" in status["value"]
    assert "user_class: 4" in status["value"]
    assert "pred_class: 2" in status["value"]
    assert "pred_confidence: 0.95" in status["value"]


def test_widget_retrain_button_triggers_manual_retraining(qtbot, monkeypatch, sdata_blobs: SpatialData) -> None:
    layer = make_blobs_labels_layer(sdata_blobs)
    viewer = DummyViewer(layers=[layer])
    widget = HarpyWidget(viewer)
    qtbot.addWidget(widget)
    select_segmentation(widget)

    retrain_calls: list[bool] = []

    def fake_retrain_now() -> bool:
        retrain_calls.append(True)
        return True

    monkeypatch.setattr(widget._classifier_controller, "retrain_now", fake_retrain_now)

    widget.retrain_button.click()

    assert retrain_calls == [True]
