from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from types import SimpleNamespace

from napari_harpy._spatialdata import get_spatialdata_label_options
from napari_harpy._widget import HarpyWidget


@dataclass
class DummySpatialData:
    labels: dict[str, object]
    path: str | None = None


@dataclass
class DummyLayer:
    metadata: dict[str, object]


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
    def __init__(self, layers: list[DummyLayer] | None = None) -> None:
        super().__init__(layers or [])
        self.events = SimpleNamespace(
            inserted=DummyEventEmitter(),
            removed=DummyEventEmitter(),
            reordered=DummyEventEmitter(),
        )


class DummyViewer:
    def __init__(self, layers: list[DummyLayer] | None = None) -> None:
        self.layers = DummyLayers(layers)


def test_widget_can_be_instantiated(qtbot) -> None:
    widget = HarpyWidget()

    qtbot.addWidget(widget)

    assert widget is not None
    assert widget.selected_segmentation_name is None


def test_spatialdata_label_options_are_deduplicated_per_dataset() -> None:
    sdata = DummySpatialData(
        labels={
            "cell_labels_global": object(),
            "nucleus_labels_global": object(),
        }
    )
    viewer = DummyViewer(
        layers=[
            DummyLayer(metadata={"sdata": sdata}),
            DummyLayer(metadata={"sdata": sdata}),
        ]
    )

    options = get_spatialdata_label_options(viewer)

    assert [option.label_name for option in options] == [
        "cell_labels_global",
        "nucleus_labels_global",
    ]


def test_widget_populates_segmentation_dropdown_from_spatialdata(qtbot) -> None:
    sdata = DummySpatialData(
        path="/Users/arne.defauw/VIB/DATA/test_data/sdata_xenium.zarr",
        labels={
            "cell_labels_global": object(),
            "nucleus_labels_global": object(),
        },
    )
    viewer = DummyViewer(layers=[DummyLayer(metadata={"sdata": sdata})])

    widget = HarpyWidget(viewer)
    qtbot.addWidget(widget)

    assert widget.segmentation_combo.count() == 2
    assert [widget.segmentation_combo.itemText(index) for index in range(widget.segmentation_combo.count())] == [
        "cell_labels_global",
        "nucleus_labels_global",
    ]
    assert widget.selected_segmentation_name == "cell_labels_global"
    assert widget.selected_spatialdata is sdata


def test_widget_refreshes_when_a_spatialdata_layer_is_added(qtbot) -> None:
    viewer = DummyViewer()
    widget = HarpyWidget(viewer)
    qtbot.addWidget(widget)

    assert widget.segmentation_combo.count() == 0

    sdata = DummySpatialData(labels={"cell_labels_global": object()})
    layer = DummyLayer(metadata={"sdata": sdata})
    viewer.layers.append(layer)
    viewer.layers.events.inserted.emit(layer)

    assert widget.segmentation_combo.count() == 1
    assert widget.segmentation_combo.itemText(0) == "cell_labels_global"
    assert widget.selected_segmentation_name == "cell_labels_global"
