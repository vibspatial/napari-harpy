from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace

from napari.layers import Labels
from spatialdata import SpatialData

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
        self.events = SimpleNamespace(
            inserted=DummyEventEmitter(),
            removed=DummyEventEmitter(),
            reordered=DummyEventEmitter(),
        )


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
    assert widget.selected_segmentation_name == "blobs_labels"
    assert widget.selected_spatialdata is sdata_blobs
    assert widget.selected_table_name == "table"


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
    assert widget.selected_segmentation_name == "blobs_labels"
    assert widget.selected_table_name == "table"


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
