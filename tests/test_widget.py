from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba
from napari.layers import Labels
from napari.utils.colormaps import DirectLabelColormap
from spatialdata import SpatialData, read_zarr
from spatialdata.models import TableModel

import napari_harpy._annotation as annotation_module
from napari_harpy._annotation import USER_CLASS_COLORS_KEY, USER_CLASS_COLUMN, _default_user_class_colors
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
    assert layer.metadata["adata"] is not None
    assert PRED_CLASS_COLUMN in layer.metadata["adata"].obs
    assert PRED_CONFIDENCE_COLUMN in layer.metadata["adata"].obs
    assert isinstance(layer.metadata["adata"].obs[PRED_CLASS_COLUMN].dtype, pd.CategoricalDtype)
    assert USER_CLASS_COLORS_KEY not in layer.metadata["adata"].uns
    assert PRED_CLASS_COLORS_KEY not in layer.metadata["adata"].uns
    assert widget.selected_instance_id is None
    assert widget.refresh_button.text() == "Rescan Viewer"
    assert widget.retrain_button.text() == "Retrain"
    assert widget.sync_button.text() == "Sync to zarr"
    assert not widget.sync_button.isEnabled()
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
    assert not widget.validation_status.isHidden()
    assert "contains duplicate values within that region" in widget.validation_status.text()
    assert "contains duplicate values within that region" in widget.selection_status.text()
    assert not widget.color_by_combo.isEnabled()
    assert not widget.class_spinbox.isEnabled()
    assert not widget.retrain_button.isEnabled()
    assert not widget.sync_button.isEnabled()


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
    assert table.uns[USER_CLASS_COLORS_KEY] == _default_user_class_colors([0, 3])
    metadata_adata = layer.metadata["adata"]
    metadata_mask = metadata_adata.obs["instance_id"] == 5
    assert metadata_adata.obs.loc[metadata_mask, USER_CLASS_COLUMN].tolist() == [3]
    assert list(metadata_adata.obs[USER_CLASS_COLUMN].cat.categories) == [0, 3]
    assert USER_CLASS_COLORS_KEY not in metadata_adata.uns
    assert "Current class: 3." in widget.selection_status.text()
    assert "Assigned class 3" in widget.annotation_feedback.text()


def test_widget_uses_table_instance_key_name_in_status_and_annotation_feedback(
    qtbot, sdata_blobs: SpatialData
) -> None:
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
    assert table.uns[USER_CLASS_COLORS_KEY] == _default_user_class_colors([0])
    assert "Current class: unlabeled." in widget.selection_status.text()
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


def test_widget_logs_warning_when_existing_user_class_colors_are_overwritten(
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

    monkeypatch.setattr(annotation_module, "logger", DummyLogger())
    widget = HarpyWidget(viewer)
    qtbot.addWidget(widget)

    assert any(f"Overwriting existing `{USER_CLASS_COLORS_KEY}` palette" in message for message in warnings)


def test_widget_enables_sync_for_backed_spatialdata(qtbot, backed_sdata_blobs: SpatialData) -> None:
    layer = make_blobs_labels_layer(backed_sdata_blobs)
    viewer = DummyViewer(layers=[layer])

    widget = HarpyWidget(viewer)
    qtbot.addWidget(widget)
    expected_table_path = Path(backed_sdata_blobs.path) / "tables" / "table"

    assert widget.sync_button.isEnabled()
    assert widget.sync_button.toolTip() == f"Write `table` table state to `{expected_table_path}`."


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
    assert widget.sync_feedback.text() == f"Synced `table` table state to `{expected_table_path}`."
    assert isinstance(reread["table"].obs[USER_CLASS_COLUMN].dtype, pd.CategoricalDtype)
    assert list(reread["table"].obs[USER_CLASS_COLUMN].cat.categories) == [0, 3]
    assert reread["table"].obs.loc[mask, USER_CLASS_COLUMN].tolist() == [3]
    assert list(reread["table"].uns[USER_CLASS_COLORS_KEY]) == _default_user_class_colors([0, 3])


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
    assert table.uns[PRED_CLASS_COLORS_KEY] == _default_user_class_colors([0, 1, 2])
    assert pred_class.loc[1] == 1
    assert pred_class.loc[24] == 2
    metadata_adata = layer.metadata["adata"]
    metadata_pred_class = metadata_adata.obs.set_index("instance_id")[PRED_CLASS_COLUMN]
    assert isinstance(metadata_adata.obs[PRED_CLASS_COLUMN].dtype, pd.CategoricalDtype)
    assert PRED_CLASS_COLORS_KEY not in metadata_adata.uns
    assert metadata_pred_class.loc[1] == 1
    assert metadata_pred_class.loc[24] == 2
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


def test_widget_rescans_viewer_without_retraining_same_classifier_context(qtbot, monkeypatch, sdata_blobs: SpatialData) -> None:
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
