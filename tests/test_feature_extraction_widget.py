from __future__ import annotations

from types import SimpleNamespace

from napari.layers import Labels
from qtpy.QtWidgets import QCheckBox, QScrollArea
from spatialdata import SpatialData

from napari_harpy.widgets._feature_extraction_widget import FeatureExtractionWidget


class DummyEventEmitter:
    def __init__(self) -> None:
        self._callbacks = []

    def connect(self, callback) -> None:
        self._callbacks.append(callback)

    def emit(self, value: object | None = None) -> None:
        event = SimpleNamespace(value=value)
        for callback in list(self._callbacks):
            callback(event)


class DummyLayers(list):
    def __init__(self, layers: list[Labels] | None = None) -> None:
        super().__init__(layers or [])
        self.events = SimpleNamespace(
            inserted=DummyEventEmitter(),
            removed=DummyEventEmitter(),
            reordered=DummyEventEmitter(),
        )


class DummyViewer:
    def __init__(self, layers: list[Labels] | None = None) -> None:
        self.layers = DummyLayers(layers)


def make_blobs_labels_layer(sdata: SpatialData, label_name: str = "blobs_labels") -> Labels:
    return Labels(
        sdata.labels[label_name],
        name=label_name,
        metadata={"sdata": sdata, "name": label_name},
    )


def test_feature_extraction_widget_can_be_instantiated(qtbot) -> None:
    widget = FeatureExtractionWidget()

    qtbot.addWidget(widget)

    assert widget is not None
    assert widget.selected_segmentation_name is None
    assert widget.selected_spatialdata is None
    assert widget.selected_image_name is None
    assert widget.selected_table_name is None
    assert widget.selected_coordinate_system is None
    assert widget.selected_feature_names == ()
    assert widget.selected_feature_key is None
    assert widget.overwrite_feature_key is False
    assert widget.segmentation_combo.count() == 0
    assert widget.image_combo.count() == 1
    assert widget.image_combo.itemText(0) == "No image"
    assert widget.table_combo.count() == 0
    assert widget.coordinate_system_combo.count() == 0
    assert widget.calculate_button.isEnabled() is False


def test_feature_extraction_widget_populates_selector_flow_from_spatialdata(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    viewer = DummyViewer([make_blobs_labels_layer(sdata_blobs)])
    widget = FeatureExtractionWidget(viewer)

    qtbot.addWidget(widget)

    assert widget.segmentation_combo.count() == 2
    assert [widget.segmentation_combo.itemText(index) for index in range(widget.segmentation_combo.count())] == [
        "blobs_labels",
        "blobs_multiscale_labels",
    ]
    assert widget.image_combo.count() == 3
    assert [widget.image_combo.itemText(index) for index in range(widget.image_combo.count())] == [
        "No image",
        "blobs_image",
        "blobs_multiscale_image",
    ]
    assert widget.table_combo.count() == 1
    assert widget.table_combo.itemText(0) == "table"
    assert widget.coordinate_system_combo.count() == 1
    assert widget.coordinate_system_combo.itemText(0) == "global"
    assert widget.selected_segmentation_name == "blobs_labels"
    assert widget.selected_spatialdata is sdata_blobs
    assert widget.selected_image_name is None
    assert widget.selected_table_name == "table"
    assert widget.selected_coordinate_system == "global"
    assert "Selection Ready" in widget.selection_status.text()
    assert "Segmentation: blobs_labels" in widget.selection_status.text()
    assert "Table: table" in widget.selection_status.text()
    assert "Coordinate system: global" in widget.selection_status.text()
    assert widget.validation_status.isHidden()


def test_feature_extraction_widget_blocks_when_selected_segmentation_has_no_linked_table(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    viewer = DummyViewer([make_blobs_labels_layer(sdata_blobs)])
    widget = FeatureExtractionWidget(viewer)

    qtbot.addWidget(widget)
    widget.segmentation_combo.setCurrentIndex(1)

    assert widget.selected_segmentation_name == "blobs_multiscale_labels"
    assert widget.image_combo.count() == 3
    assert widget.selected_image_name is None
    assert widget.table_combo.count() == 0
    assert widget.selected_table_name is None
    assert widget.coordinate_system_combo.count() == 1
    assert widget.selected_coordinate_system == "global"
    assert "No Table Linked" in widget.selection_status.text()
    assert "creating a new linked table" in widget.selection_status.text()
    assert widget.validation_status.isHidden()


def test_feature_extraction_widget_exposes_grouped_feature_checkboxes(qtbot) -> None:
    widget = FeatureExtractionWidget()

    qtbot.addWidget(widget)

    scroll_area = widget.findChild(QScrollArea, "feature_extraction_scroll_area")
    assert scroll_area is not None
    assert scroll_area.widgetResizable()
    assert widget.intensity_features_group.title() == "Intensity Features"
    assert widget.morphology_features_group.title() == "Morphology Features"
    assert widget.findChild(QCheckBox, "feature_checkbox_mean") is not None
    assert widget.findChild(QCheckBox, "feature_checkbox_var") is not None
    assert widget.findChild(QCheckBox, "feature_checkbox_area") is not None
    assert widget.findChild(QCheckBox, "feature_checkbox_perimeter") is not None
    assert widget.findChild(QCheckBox, "feature_extraction_overwrite_feature_key_checkbox") is None
    assert widget.intensity_features_hint.isHidden()


def test_feature_extraction_widget_reads_back_feature_selection_and_output_key(qtbot) -> None:
    widget = FeatureExtractionWidget()

    qtbot.addWidget(widget)

    widget.findChild(QCheckBox, "feature_checkbox_mean").setChecked(True)
    widget.findChild(QCheckBox, "feature_checkbox_var").setChecked(True)
    widget.findChild(QCheckBox, "feature_checkbox_area").setChecked(True)
    widget.output_key_line_edit.setText("object_features")

    assert widget.selected_feature_names == ("mean", "var", "area")
    assert widget.selected_feature_key == "object_features"
    assert widget.overwrite_feature_key is False
    assert not widget.intensity_features_hint.isHidden()
    assert "choose an image" in widget.intensity_features_hint.text()


def test_feature_extraction_widget_hides_intensity_warning_when_image_is_selected(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    viewer = DummyViewer([make_blobs_labels_layer(sdata_blobs)])
    widget = FeatureExtractionWidget(viewer)

    qtbot.addWidget(widget)
    widget.image_combo.setCurrentIndex(1)
    widget.findChild(QCheckBox, "feature_checkbox_mean").setChecked(True)

    assert widget.selected_image_name == "blobs_image"
    assert widget.intensity_features_hint.isHidden()
