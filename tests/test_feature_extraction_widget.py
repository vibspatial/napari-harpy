from __future__ import annotations

from types import SimpleNamespace

from napari.layers import Labels
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
