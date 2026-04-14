from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import zarr
from matplotlib.colors import to_rgba
from napari.layers import Labels
from napari.utils.colormaps import DirectLabelColormap
from qtpy.QtCore import QObject, Signal
from spatialdata import SpatialData, read_zarr
from spatialdata.models import TableModel

import napari_harpy._annotation as annotation_module
import napari_harpy._class_palette as class_palette_module
import napari_harpy._classifier as classifier_module
import napari_harpy._widget as widget_module
from napari_harpy._annotation import USER_CLASS_COLORS_KEY, USER_CLASS_COLUMN
from napari_harpy._class_palette import default_class_colors
from napari_harpy._classifier import (
    CLASSIFIER_CONFIG_KEY,
    PRED_CLASS_COLORS_KEY,
    PRED_CLASS_COLUMN,
    PRED_CONFIDENCE_COLUMN,
)
from napari_harpy._spatialdata import get_spatialdata_label_options
from napari_harpy._widget import HarpyWidget


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
    def __init__(self, layers: list[Labels] | None = None) -> None:
        self.layers = DummyLayers(layers)


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

    assert widget is not None
    assert widget.selected_segmentation_name is None
    assert widget.selected_table_name is None
    assert widget.selected_feature_key is None
    assert widget.selected_color_by == "user_class"


def test_spatialdata_label_options_are_deduplicated_per_dataset(sdata_blobs: SpatialData) -> None:
    """Avoid duplicate dropdown entries when multiple layers share one SpatialData object.

    In a real napari-spatialdata session, the viewer can contain multiple layers that all
    reference the same `SpatialData` dataset through `layer.metadata["sdata"]`. If we expanded
    `sdata.labels` once per layer, the segmentation dropdown would repeat the same label names.

    This test uses two layers pointing to the same `sdata` to verify that we deduplicate at the
    dataset level and expose each labels element only once.
    """
    first_layer = make_blobs_labels_layer(sdata_blobs, "blobs_labels")
    second_layer = Labels(
        sdata_blobs.labels["blobs_labels"],
        name="blobs_labels_duplicate",
        metadata={"sdata": sdata_blobs, "name": "blobs_labels"},
    )
    viewer = DummyViewer(layers=[first_layer, second_layer])

    options = get_spatialdata_label_options(viewer)

    assert [option.label_name for option in options] == [
        "blobs_labels",
        "blobs_multiscale_labels",
    ]


def test_widget_populates_segmentation_dropdown_from_spatialdata(qtbot, sdata_blobs: SpatialData) -> None:
    layer = make_blobs_labels_layer(sdata_blobs)
    viewer = DummyViewer(layers=[layer])

    widget = HarpyWidget(viewer)
    qtbot.addWidget(widget)

    assert widget.segmentation_combo.count() == 2
    assert [widget.segmentation_combo.itemText(index) for index in range(widget.segmentation_combo.count())] == [
        "blobs_labels",
        "blobs_multiscale_labels",
    ]
    assert widget.table_combo.count() == 1
    assert widget.table_combo.itemText(0) == "table"
    assert widget.feature_matrix_combo.count() == 2
    assert [widget.feature_matrix_combo.itemText(index) for index in range(widget.feature_matrix_combo.count())] == [
        "features_1",
        "features_2",
    ]
    assert widget.color_by_combo.count() == 3
    assert [widget.color_by_combo.itemText(index) for index in range(widget.color_by_combo.count())] == [
        "user_class",
        "pred_class",
        "pred_confidence",
    ]
    assert widget.selected_segmentation_name == "blobs_labels"
    assert widget.selected_spatialdata is sdata_blobs
    assert widget.selected_table_name == "table"
    assert widget.selected_feature_key == "features_1"
    assert widget.selected_color_by == "user_class"
    assert widget.selected_table_metadata is not None
    assert widget.selected_table_metadata.region_key == "region"
    assert widget.selected_table_metadata.instance_key == "instance_id"
    assert widget.selected_table_metadata.regions == ("blobs_labels",)
    assert "adata" not in layer.metadata
    assert widget.selected_instance_id is None
    assert widget.refresh_button.text() == "Rescan Viewer"
    assert widget.retrain_button.text() == "Retrain"
    assert widget.sync_button.text() == "Write Table to zarr"
    assert widget.reload_button.text() == "Reload Table from zarr"
    assert not widget.sync_button.isEnabled()
    assert not widget.reload_button.isEnabled()
    assert widget.retrain_button.isEnabled()
    assert str(layer.mode) == "pick"
    assert viewer.layers.selection.active is layer
    assert "Click an object in the viewer." in widget.selection_status.text()
    assert widget.validation_status.isHidden()
    assert widget.validation_status.text() == ""
    assert "model is stale" in widget.classifier_feedback.text()


def test_widget_surfaces_invalid_table_binding_for_duplicate_instance_ids(qtbot, sdata_blobs: SpatialData) -> None:
    table = sdata_blobs["table"]
    first_index, second_index = table.obs.index[:2]
    table.obs.loc[second_index, "instance_id"] = table.obs.loc[first_index, "instance_id"]
    layer = make_blobs_labels_layer(sdata_blobs)
    viewer = DummyViewer(layers=[layer])

    widget = HarpyWidget(viewer)
    qtbot.addWidget(widget)

    assert widget.selected_table_name == "table"
    assert widget.validation_status.isHidden()
    assert widget.validation_status.text() == ""
    assert "contains duplicate values within that region" in widget.selection_status.text()
    assert not widget.color_by_combo.isEnabled()
    assert not widget.class_spinbox.isEnabled()
    assert not widget.retrain_button.isEnabled()
    assert not widget.sync_button.isEnabled()
    assert not widget.reload_button.isEnabled()


def test_widget_refreshes_when_a_spatialdata_layer_is_added(qtbot, sdata_blobs: SpatialData) -> None:
    viewer = DummyViewer()
    widget = HarpyWidget(viewer)
    qtbot.addWidget(widget)

    assert widget.segmentation_combo.count() == 0

    layer = make_blobs_labels_layer(sdata_blobs)
    viewer.layers.append(layer)
    viewer.layers.events.inserted.emit(layer)

    assert widget.segmentation_combo.count() == 2
    assert widget.segmentation_combo.itemText(0) == "blobs_labels"
    assert widget.table_combo.count() == 1
    assert widget.table_combo.itemText(0) == "table"
    assert widget.feature_matrix_combo.count() == 2
    assert widget.feature_matrix_combo.itemText(0) == "features_1"
    assert widget.selected_segmentation_name == "blobs_labels"
    assert widget.selected_table_name == "table"
    assert widget.selected_feature_key == "features_1"


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

    widget.feature_matrix_combo.setCurrentIndex(1)

    assert widget.selected_feature_key == "features_2"
    assert "feature matrix changed" in widget.classifier_feedback.text()


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

    layer.selected_label = 1

    assert widget.selected_instance_id == 1
    assert widget.apply_class_button.isEnabled()
    assert "Current instance_id: 1." in widget.selection_status.text()


def test_widget_automatically_enables_pick_mode_for_bound_labels_layer(qtbot, sdata_blobs: SpatialData) -> None:
    layer = make_blobs_labels_layer(sdata_blobs)
    viewer = DummyViewer(layers=[layer])

    widget = HarpyWidget(viewer)
    qtbot.addWidget(widget)

    assert str(layer.mode) == "pick"
    assert viewer.layers.selection.active is layer


def test_widget_picks_multiscale_labels_layers_without_napari_pick_mode(qtbot, sdata_blobs: SpatialData) -> None:
    layer = make_multiscale_blobs_labels_layer(sdata_blobs)
    viewer = DummyViewer(layers=[layer])

    widget = HarpyWidget(viewer)
    qtbot.addWidget(widget)

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


def test_widget_disables_pick_mode_when_selected_segmentation_layer_is_not_loaded(
    qtbot, sdata_blobs: SpatialData
) -> None:
    layer = make_blobs_labels_layer(sdata_blobs)
    viewer = DummyViewer(layers=[layer])

    widget = HarpyWidget(viewer)
    qtbot.addWidget(widget)

    widget.segmentation_combo.setCurrentIndex(1)
    layer.selected_label = 9

    assert widget.selected_segmentation_name == "blobs_multiscale_labels"
    assert widget.selected_instance_id is None
    assert "not currently loaded as a napari Labels layer" in widget.selection_status.text()


def test_widget_handles_tables_without_obsm_entries(qtbot, sdata_blobs: SpatialData) -> None:
    table = sdata_blobs["table"]
    for key in list(table.obsm.keys()):
        del table.obsm[key]

    layer = make_blobs_labels_layer(sdata_blobs)
    viewer = DummyViewer(layers=[layer])

    widget = HarpyWidget(viewer)
    qtbot.addWidget(widget)

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

    layer.selected_label = 5
    widget.class_spinbox.setValue(4)

    assert widget.apply_class_button.toolTip().endswith("Shortcut: A.")
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

    layer.selected_label = 5
    widget.class_spinbox.setValue(2)
    widget.apply_class_button.click()

    assert widget.clear_class_button.toolTip().endswith("Shortcut: R.")
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

    layer.selected_label = 5
    widget.class_spinbox.setValue(4)
    widget.apply_class_button.click()

    assert isinstance(layer.colormap, DirectLabelColormap)
    assert np.allclose(layer.colormap.color_dict[0], np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32))
    assert layer.colormap.color_dict[5][3] > 0
    assert layer.colormap.color_dict[6][3] > 0
    assert not np.allclose(layer.colormap.color_dict[5], layer.colormap.color_dict[6])
    assert USER_CLASS_COLUMN in layer.features.columns
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

    assert warnings == []


def test_widget_enables_sync_for_backed_spatialdata(qtbot, backed_sdata_blobs: SpatialData) -> None:
    layer = make_blobs_labels_layer(backed_sdata_blobs)
    viewer = DummyViewer(layers=[layer])

    widget = HarpyWidget(viewer)
    qtbot.addWidget(widget)
    expected_table_path = Path(backed_sdata_blobs.path) / "tables" / "table"

    assert widget.sync_button.isEnabled()
    assert widget.reload_button.isEnabled()
    assert widget.sync_button.toolTip() == f"Write `table` table state to `{expected_table_path}`."
    assert widget.reload_button.toolTip() == f"Reload `table` table state from `{expected_table_path}`."


def test_widget_marks_persistence_dirty_on_annotation_change_and_clears_it_on_sync(
    qtbot, monkeypatch, backed_sdata_blobs: SpatialData
) -> None:
    layer = make_blobs_labels_layer(backed_sdata_blobs)
    viewer = DummyViewer(layers=[layer])

    widget = HarpyWidget(viewer)
    qtbot.addWidget(widget)
    monkeypatch.setattr(widget._classifier_controller, "schedule_retrain", lambda *args, **kwargs: False)

    layer.selected_label = 5
    widget.class_spinbox.setValue(3)
    widget.apply_class_button.click()

    assert widget._persistence_controller.is_dirty is True
    assert "Unsynced local table changes are present." in widget.sync_button.toolTip()
    assert "Unsynced local table changes are present." in widget.reload_button.toolTip()

    widget.sync_button.click()

    assert widget._persistence_controller.is_dirty is False
    assert "Unsynced local table changes are present." not in widget.sync_button.toolTip()
    assert "Unsynced local table changes are present." not in widget.reload_button.toolTip()


def test_widget_syncs_user_class_to_backed_zarr(qtbot, backed_sdata_blobs: SpatialData) -> None:
    layer = make_blobs_labels_layer(backed_sdata_blobs)
    viewer = DummyViewer(layers=[layer])

    widget = HarpyWidget(viewer)
    qtbot.addWidget(widget)
    expected_table_path = Path(backed_sdata_blobs.path) / "tables" / "table"

    layer.selected_label = 5
    widget.class_spinbox.setValue(3)
    widget.apply_class_button.click()
    widget.sync_button.click()

    reread = read_zarr(backed_sdata_blobs.path)
    mask = (reread["table"].obs["region"] == "blobs_labels") & (reread["table"].obs["instance_id"] == 5)

    assert widget.sync_button.isEnabled()
    assert widget.reload_button.isEnabled()
    assert "Persistence Updated" in widget.persistence_feedback.text()
    assert f"Wrote `table` table state to `{expected_table_path}`." in widget.persistence_feedback.text()
    assert "#166534" in widget.persistence_feedback.styleSheet()
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

    assert widget._persistence_controller.is_dirty is False

    widget.retrain_button.click()
    qtbot.waitUntil(
        lambda: widget._persistence_controller.is_dirty and table.obs[PRED_CLASS_COLUMN].astype("string").ne("0").any(),
        timeout=5000,
    )

    assert widget._persistence_controller.is_dirty is True
    assert "Unsynced local table changes are present." in widget.sync_button.toolTip()
    assert "Unsynced local table changes are present." in widget.reload_button.toolTip()


def test_widget_cancels_dirty_reload_when_user_chooses_cancel(
    qtbot, monkeypatch, backed_sdata_blobs: SpatialData
) -> None:
    layer = make_blobs_labels_layer(backed_sdata_blobs)
    viewer = DummyViewer(layers=[layer])
    widget = HarpyWidget(viewer)
    qtbot.addWidget(widget)
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
    assert "Persistence Updated" in widget.persistence_feedback.text()
    assert (
        f"Wrote local changes and reloaded `table` table state from `{expected_table_path}`."
        in widget.persistence_feedback.text()
    )
    assert "#166534" in widget.persistence_feedback.styleSheet()


def test_widget_dirty_reload_can_discard_local_edits(qtbot, monkeypatch, backed_sdata_blobs: SpatialData) -> None:
    layer = make_blobs_labels_layer(backed_sdata_blobs)
    viewer = DummyViewer(layers=[layer])
    widget = HarpyWidget(viewer)
    qtbot.addWidget(widget)
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
    assert "Persistence Updated" in widget.persistence_feedback.text()
    assert f"Reloaded `table` table state from `{expected_table_path}`." in widget.persistence_feedback.text()
    assert "#166534" in widget.persistence_feedback.styleSheet()


def test_widget_reloads_table_state_from_backed_zarr(qtbot, backed_sdata_blobs: SpatialData) -> None:
    layer = make_blobs_labels_layer(backed_sdata_blobs)
    viewer = DummyViewer(layers=[layer])

    widget = HarpyWidget(viewer)
    qtbot.addWidget(widget)
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

    assert "Persistence Updated" in widget.persistence_feedback.text()
    assert f"Reloaded `table` table state from `{expected_table_path}`." in widget.persistence_feedback.text()
    assert "#166534" in widget.persistence_feedback.styleSheet()
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

    assert "Persistence Updated" in widget.persistence_feedback.text()
    assert f"Reloaded `table` table state from `{expected_table_path}`." in widget.persistence_feedback.text()
    assert "#166534" in widget.persistence_feedback.styleSheet()
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
    workers: list[_DeferredWorker] = []

    def fake_create_training_worker(job):
        result = classifier_module.ClassifierJobResult(
            job_id=job.job_id,
            feature_key=job.feature_key,
            label_name=job.label_name,
            table_name=job.table_name,
            active_positions=job.active_positions,
            pred_classes=np.full(job.active_positions.shape, 1, dtype=np.int64),
            pred_confidences=np.full(job.active_positions.shape, 0.91, dtype=np.float64),
            trained_at="2026-04-13T09:00:00+00:00",
            model_params=dict(classifier_module.RANDOM_FOREST_PARAMS),
            eligibility=job.eligibility,
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
        "label_name": "blobs_labels",
        "roi_mode": "none",
        "trained": True,
        "eligible": True,
        "reason": "Ready to train.",
        "training_timestamp": "2026-04-13T09:00:00+00:00",
        "n_labeled_objects": 4,
        "n_active_objects": int(table.n_obs),
        "n_features": 2,
        "class_labels_seen": [1, 2],
        "rf_params": dict(classifier_module.RANDOM_FOREST_PARAMS),
    }
    _write_disk_table_state(backed_sdata_blobs, obs=obs, obsm=obsm, uns=uns)

    widget.reload_button.click()

    assert workers[0].quit_called is True
    assert widget._classifier_controller.is_training is False
    assert widget._classifier_controller.is_dirty is False
    assert "Persistence Updated" in widget.persistence_feedback.text()
    assert f"Reloaded `table` table state from `{expected_table_path}`." in widget.persistence_feedback.text()
    assert "#166534" in widget.persistence_feedback.styleSheet()
    assert table.obs[PRED_CLASS_COLUMN].eq(7).all()
    assert table.obs[PRED_CONFIDENCE_COLUMN].eq(0.77).all()
    assert "Loaded predictions for" in widget.classifier_feedback.text()
    assert len(workers) == 1

    workers[0].emit_returned()

    assert table.obs[PRED_CLASS_COLUMN].eq(7).all()
    assert table.obs[PRED_CONFIDENCE_COLUMN].eq(0.77).all()
    assert "Loaded predictions for" in widget.classifier_feedback.text()


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

    coords = tuple(float(value) for value in np.argwhere(np.asarray(sdata_blobs.labels["blobs_labels"]) == 5)[0])
    status = layer.get_status(position=coords, view_direction=np.array([1.0, 0.0]), dims_displayed=[0, 1])

    assert "user_class: 4" in status["value"]
    assert "pred_class: 2" in status["value"]
    assert "pred_confidence: 0.95" in status["value"]


def test_widget_rescans_viewer_without_retraining_same_classifier_context(
    qtbot, monkeypatch, sdata_blobs: SpatialData
) -> None:
    layer = make_blobs_labels_layer(sdata_blobs)
    viewer = DummyViewer(layers=[layer])
    widget = HarpyWidget(viewer)
    qtbot.addWidget(widget)

    retrain_calls: list[bool] = []

    def fake_retrain(*, immediate: bool = False) -> bool:
        retrain_calls.append(immediate)
        return True

    monkeypatch.setattr(widget._classifier_controller, "schedule_retrain", fake_retrain)

    widget.refresh_button.click()

    assert retrain_calls == []


def test_widget_retrain_button_triggers_manual_retraining(qtbot, monkeypatch, sdata_blobs: SpatialData) -> None:
    layer = make_blobs_labels_layer(sdata_blobs)
    viewer = DummyViewer(layers=[layer])
    widget = HarpyWidget(viewer)
    qtbot.addWidget(widget)

    retrain_calls: list[bool] = []

    def fake_retrain_now() -> bool:
        retrain_calls.append(True)
        return True

    monkeypatch.setattr(widget._classifier_controller, "retrain_now", fake_retrain_now)

    widget.retrain_button.click()

    assert retrain_calls == [True]
