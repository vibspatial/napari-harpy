from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace

import napari_harpy.widgets._viewer_widget as viewer_widget_module
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


class DummyViewer:
    def __init__(self) -> None:
        self.layers = DummyLayers()


def test_viewer_widget_can_be_instantiated(qtbot) -> None:
    widget = ViewerWidget()

    qtbot.addWidget(widget)

    assert widget is not None
    assert widget.app_state.sdata is None
    assert not widget.empty_state_label.isHidden()
    assert widget.summary_label.text() == "No SpatialData loaded."
    assert widget.coordinate_system_combo.count() == 0
    assert not widget.coordinate_system_combo.isEnabled()
    assert widget.image_cards == []
    assert widget.labels_cards == []


def test_viewer_widget_refreshes_cards_when_shared_sdata_changes(qtbot, sdata_blobs) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)

    qtbot.addWidget(widget)

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(sdata_blobs)

    assert widget.app_state.sdata is sdata_blobs
    assert widget.empty_state_label.isHidden()
    assert widget.coordinate_system_combo.count() == 1
    assert widget.coordinate_system_combo.itemText(0) == "global"
    assert len(widget.image_cards) == 2
    assert len(widget.labels_cards) == 2
    assert [card.image_name for card in widget.image_cards] == ["blobs_image", "blobs_multiscale_image"]
    assert [card.label_name for card in widget.labels_cards] == ["blobs_labels", "blobs_multiscale_labels"]
    assert widget.image_cards[0].channel_names == ["0", "1", "2"]
    assert widget.image_cards[0].show_stack_button.text() == "Show stack"
    assert not widget.image_cards[0].show_stack_button.isEnabled()
    assert widget.labels_cards[0].linked_table_combo.count() == 1
    assert widget.labels_cards[0].linked_table_combo.itemText(0) == "table"
    assert widget.labels_cards[1].linked_table_combo.count() == 1
    assert widget.labels_cards[1].linked_table_combo.itemText(0) == "No linked tables"
    assert not widget.labels_cards[1].linked_table_combo.isEnabled()
    assert "In coordinate system `global`" in widget.summary_label.text()


def test_viewer_widget_filters_cards_by_selected_coordinate_system(qtbot, monkeypatch) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)
    fake_sdata = object()

    qtbot.addWidget(widget)

    monkeypatch.setattr(viewer_widget_module, "_get_coordinate_systems_from_sdata", lambda sdata: ["global", "local"])
    monkeypatch.setattr(
        viewer_widget_module,
        "_get_labels_in_coordinate_system",
        lambda sdata, coordinate_system: ["labels_global"] if coordinate_system == "global" else ["labels_local"],
    )
    monkeypatch.setattr(
        viewer_widget_module,
        "_get_images_in_coordinate_system",
        lambda sdata, coordinate_system: ["image_global"] if coordinate_system == "global" else ["image_local"],
    )
    monkeypatch.setattr(viewer_widget_module, "_get_image_channel_names", lambda sdata, image_name: ["c0", "c1"])
    monkeypatch.setattr(
        viewer_widget_module,
        "get_annotating_table_names",
        lambda sdata, label_name: ["table_global"] if label_name == "labels_global" else ["table_local"],
    )

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(fake_sdata)

    assert widget.coordinate_system_combo.count() == 2
    assert [card.image_name for card in widget.image_cards] == ["image_global"]
    assert [card.label_name for card in widget.labels_cards] == ["labels_global"]

    widget.coordinate_system_combo.setCurrentIndex(1)

    assert [card.image_name for card in widget.image_cards] == ["image_local"]
    assert [card.label_name for card in widget.labels_cards] == ["labels_local"]


def test_viewer_widget_add_update_labels_loads_and_activates_layer(qtbot, sdata_blobs) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)

    qtbot.addWidget(widget)

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(sdata_blobs)

    first_card = widget.labels_cards[0]

    first_card.add_update_button.click()

    assert len(viewer.layers) == 1
    layer = viewer.layers[0]
    assert layer.name == "blobs_labels"
    assert widget.app_state.viewer_adapter.layer_bindings.get_binding(layer) is not None
    assert viewer.layers.selection.active is layer
    assert "Loaded segmentation `blobs_labels`" in widget.action_feedback_label.text()
    assert not widget.action_feedback_label.isHidden()

    first_card.add_update_button.click()

    assert len(viewer.layers) == 1


def test_viewer_widget_shares_app_state_for_same_viewer(qtbot) -> None:
    viewer = DummyViewer()
    first = ViewerWidget(viewer)
    second = ViewerWidget(viewer)

    qtbot.addWidget(first)
    qtbot.addWidget(second)

    assert first.app_state is second.app_state
