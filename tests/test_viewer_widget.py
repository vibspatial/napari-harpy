from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace

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


def test_viewer_widget_refreshes_when_shared_sdata_changes(qtbot, sdata_blobs) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)

    qtbot.addWidget(widget)

    assert widget.app_state.sdata is None

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(sdata_blobs)

    assert widget.app_state.sdata is sdata_blobs
    assert widget.empty_state_label.isHidden()
    assert widget.coordinate_system_combo.count() == 1
    assert widget.coordinate_system_combo.itemText(0) == "global"
    assert widget.labels_combo.count() == 2
    assert widget.image_combo.count() == 2
    assert widget.display_mode_combo.count() == 2
    assert widget.display_mode_combo.itemText(0) == "stack"
    assert widget.display_mode_combo.itemText(1) == "overlay"
    assert widget.coordinate_system_combo.isEnabled()
    assert "Loaded SpatialData with 1 coordinate system(s)" in widget.summary_label.text()


def test_viewer_widget_shares_app_state_for_same_viewer(qtbot) -> None:
    viewer = DummyViewer()
    first = ViewerWidget(viewer)
    second = ViewerWidget(viewer)

    qtbot.addWidget(first)
    qtbot.addWidget(second)

    assert first.app_state is second.app_state
