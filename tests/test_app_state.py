from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace

import napari_harpy._interactive as interactive_module
from napari_harpy._app_state import HarpyAppState, get_or_create_app_state
from napari_harpy.widgets._feature_extraction_widget import FeatureExtractionWidget
from napari_harpy.widgets._object_classification_widget import ObjectClassificationWidget


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


class DummyWindow:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str | None, bool]] = []
        self._dock_widgets: dict[tuple[str, str | None], tuple[object, object]] = {}

    def add_plugin_dock_widget(
        self,
        plugin_name: str,
        widget_name: str | None = None,
        tabify: bool = False,
    ) -> tuple[object, object]:
        key = (plugin_name, widget_name)
        if key in self._dock_widgets:
            return self._dock_widgets[key]

        dock_widget = SimpleNamespace(plugin_name=plugin_name, widget_name=widget_name, tabify=tabify)
        inner_widget = SimpleNamespace(plugin_name=plugin_name, widget_name=widget_name)
        self._dock_widgets[key] = (dock_widget, inner_widget)
        self.calls.append((plugin_name, widget_name, tabify))
        return self._dock_widgets[key]


class DummyViewer:
    def __init__(self) -> None:
        self.layers = DummyLayers()
        self.window = DummyWindow()


def test_get_or_create_app_state_returns_same_state_for_same_viewer() -> None:
    viewer = DummyViewer()
    first = get_or_create_app_state(viewer)
    second = get_or_create_app_state(viewer)
    other = get_or_create_app_state(DummyViewer())

    assert first is second
    assert first.viewer is viewer
    assert other is not first


def test_harpy_app_state_emits_sdata_changed(qtbot, sdata_blobs) -> None:
    state = HarpyAppState()

    with qtbot.waitSignal(state.sdata_changed) as blocker:
        state.set_sdata(sdata_blobs)

    assert blocker.args == [sdata_blobs]
    assert state.sdata is sdata_blobs

    with qtbot.waitSignal(state.sdata_changed) as blocker:
        state.clear_sdata()

    assert blocker.args == [None]
    assert state.sdata is None


def test_widgets_share_app_state_for_same_viewer(qtbot) -> None:
    viewer = DummyViewer()
    feature_widget = FeatureExtractionWidget(viewer)
    object_widget = ObjectClassificationWidget(viewer)

    qtbot.addWidget(feature_widget)
    qtbot.addWidget(object_widget)

    assert feature_widget.app_state is object_widget.app_state
    assert feature_widget.app_state is get_or_create_app_state(viewer)


def test_interactive_headless_sets_sdata_without_running_event_loop(monkeypatch, sdata_blobs) -> None:
    viewer = DummyViewer()
    run_calls: list[str] = []

    monkeypatch.setattr(interactive_module.napari, "run", lambda: run_calls.append("run"))

    interactive = interactive_module.Interactive(sdata_blobs, viewer=viewer, headless=True)

    assert interactive.viewer is viewer
    assert interactive.app_state is get_or_create_app_state(viewer)
    assert interactive.app_state.sdata is sdata_blobs
    assert run_calls == []
    assert viewer.window.calls == [
        ("napari-harpy", "Viewer", True),
        ("napari-harpy", "Feature Extraction", True),
        ("napari-harpy", "Object Classification", True),
    ]


def test_interactive_auto_runs_and_reuses_existing_plugin_widgets(monkeypatch, sdata_blobs) -> None:
    viewer = DummyViewer()
    run_calls: list[str] = []

    monkeypatch.setattr(interactive_module.napari, "run", lambda: run_calls.append("run"))

    interactive_module.Interactive(sdata_blobs, viewer=viewer, headless=False)
    interactive_module.Interactive(sdata_blobs, viewer=viewer, headless=False)

    assert run_calls == ["run", "run"]
    assert viewer.window.calls == [
        ("napari-harpy", "Viewer", True),
        ("napari-harpy", "Feature Extraction", True),
        ("napari-harpy", "Object Classification", True),
    ]
