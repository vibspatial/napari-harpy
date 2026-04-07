from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace

from napari.layers import Labels
from spatialdata.datasets import blobs

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


def make_blobs_labels_layer(label_name: str = "blobs_labels") -> tuple[object, Labels]:
    sdata = blobs()
    layer = Labels(
        sdata.labels[label_name],
        name=label_name,
        metadata={"sdata": sdata, "name": label_name},
    )
    return sdata, layer


def test_widget_can_be_instantiated(qtbot) -> None:
    widget = HarpyWidget()

    qtbot.addWidget(widget)

    assert widget is not None
    assert widget.selected_segmentation_name is None


def test_spatialdata_label_options_are_deduplicated_per_dataset() -> None:
    sdata, first_layer = make_blobs_labels_layer("blobs_labels")
    second_layer = Labels(
        sdata.labels["blobs_labels"],
        name="blobs_labels_duplicate",
        metadata={"sdata": sdata, "name": "blobs_labels"},
    )
    viewer = DummyViewer(layers=[first_layer, second_layer])

    options = get_spatialdata_label_options(viewer)

    assert [option.label_name for option in options] == [
        "blobs_labels",
        "blobs_multiscale_labels",
    ]


def test_widget_populates_segmentation_dropdown_from_spatialdata(qtbot) -> None:
    sdata, layer = make_blobs_labels_layer()
    viewer = DummyViewer(layers=[layer])

    widget = HarpyWidget(viewer)
    qtbot.addWidget(widget)

    assert widget.segmentation_combo.count() == 2
    assert [widget.segmentation_combo.itemText(index) for index in range(widget.segmentation_combo.count())] == [
        "blobs_labels",
        "blobs_multiscale_labels",
    ]
    assert widget.selected_segmentation_name == "blobs_labels"
    assert widget.selected_spatialdata is sdata


def test_widget_refreshes_when_a_spatialdata_layer_is_added(qtbot) -> None:
    viewer = DummyViewer()
    widget = HarpyWidget(viewer)
    qtbot.addWidget(widget)

    assert widget.segmentation_combo.count() == 0

    _, layer = make_blobs_labels_layer()
    viewer.layers.append(layer)
    viewer.layers.events.inserted.emit(layer)

    assert widget.segmentation_combo.count() == 2
    assert widget.segmentation_combo.itemText(0) == "blobs_labels"
    assert widget.selected_segmentation_name == "blobs_labels"
