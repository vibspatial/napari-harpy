from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace

import numpy as np
from napari.layers import Labels
from napari.utils.colormaps import DirectLabelColormap
from spatialdata import SpatialData

from napari_harpy._annotation import USER_CLASS_COLUMN
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


def test_widget_can_be_instantiated(qtbot) -> None:
    widget = HarpyWidget()

    qtbot.addWidget(widget)

    assert widget is not None
    assert widget.selected_segmentation_name is None
    assert widget.selected_table_name is None
    assert widget.selected_feature_key is None


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
    assert [
        widget.feature_matrix_combo.itemText(index) for index in range(widget.feature_matrix_combo.count())
    ] == ["features_1", "features_2"]
    assert widget.selected_segmentation_name == "blobs_labels"
    assert widget.selected_spatialdata is sdata_blobs
    assert widget.selected_table_name == "table"
    assert widget.selected_feature_key == "features_1"
    assert widget.selected_table_metadata is not None
    assert widget.selected_table_metadata.region_key == "region"
    assert widget.selected_table_metadata.instance_key == "instance_id"
    assert widget.selected_table_metadata.regions == ("blobs_labels",)
    assert widget.selected_instance_id is None
    assert widget.refresh_button.text() == "Rescan Viewer"
    assert str(layer.mode) == "pick"
    assert viewer.layers.selection.active is layer
    assert "Click an object in the viewer." in widget.selection_status.text()
    assert widget.validation_status.isHidden()
    assert widget.validation_status.text() == ""


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


def test_widget_tracks_picked_instance_id_from_labels_layer(qtbot, sdata_blobs: SpatialData) -> None:
    layer = make_blobs_labels_layer(sdata_blobs)
    viewer = DummyViewer(layers=[layer])

    widget = HarpyWidget(viewer)
    qtbot.addWidget(widget)

    layer.selected_label = 5

    assert widget.selected_instance_id == 5
    assert widget.apply_class_button.isEnabled()
    assert "Current instance id: 5." in widget.selection_status.text()
    assert "Current class: unlabeled." in widget.selection_status.text()


def test_widget_accepts_first_pick_when_instance_id_is_one(qtbot, sdata_blobs: SpatialData) -> None:
    layer = make_blobs_labels_layer(sdata_blobs)
    viewer = DummyViewer(layers=[layer])

    widget = HarpyWidget(viewer)
    qtbot.addWidget(widget)

    layer.selected_label = 1

    assert widget.selected_instance_id == 1
    assert widget.apply_class_button.isEnabled()
    assert "Current instance id: 1." in widget.selection_status.text()


def test_widget_automatically_enables_pick_mode_for_bound_labels_layer(qtbot, sdata_blobs: SpatialData) -> None:
    layer = make_blobs_labels_layer(sdata_blobs)
    viewer = DummyViewer(layers=[layer])

    widget = HarpyWidget(viewer)
    qtbot.addWidget(widget)

    assert str(layer.mode) == "pick"
    assert viewer.layers.selection.active is layer


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
    assert table.obs.loc[mask, USER_CLASS_COLUMN].tolist() == [3]
    assert int(table.obs.loc[table.obs["instance_id"] == 6, USER_CLASS_COLUMN].iloc[0]) == 0
    assert "Current class: 3." in widget.selection_status.text()
    assert "Assigned class 3" in widget.annotation_feedback.text()


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

    assert table.obs.loc[mask, USER_CLASS_COLUMN].tolist() == [0]
    assert "Current class: unlabeled." in widget.selection_status.text()
    assert "Cleared the user class" in widget.annotation_feedback.text()


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
