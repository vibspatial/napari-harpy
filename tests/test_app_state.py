from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace

import napari_harpy._app_state as app_state_module
import napari_harpy._interactive as interactive_module
from napari_harpy._app_state import (
    CoordinateSystemChangedEvent,
    FeatureMatrixWrittenEvent,
    HarpyAppState,
    get_or_create_app_state,
)
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


def test_harpy_app_state_emits_feature_matrix_written_and_marks_table_dirty(qtbot, sdata_blobs) -> None:
    state = HarpyAppState()
    event = FeatureMatrixWrittenEvent(
        sdata=sdata_blobs,
        table_name="table",
        feature_key="features_new",
        change_kind="created",
    )

    assert state.is_table_dirty(sdata_blobs, "table") is False

    with qtbot.waitSignal(state.feature_matrix_written) as blocker:
        state.emit_feature_matrix_written(event)

    assert blocker.args == [event]
    assert state.is_table_dirty(sdata_blobs, "table") is True

    state.clear_table_dirty(sdata_blobs, "table")

    assert state.is_table_dirty(sdata_blobs, "table") is False


def test_harpy_app_state_set_coordinate_system_emits_event_and_prunes_layers(qtbot, monkeypatch, sdata_blobs) -> None:
    state = HarpyAppState()
    state.sdata = sdata_blobs
    removed_calls: list[dict[str, object | None]] = []

    monkeypatch.setattr(
        state.viewer_adapter,
        "remove_layers_outside_coordinate_system",
        lambda *, sdata, coordinate_system: removed_calls.append(
            {"sdata": sdata, "coordinate_system": coordinate_system}
        )
        or [],
    )

    with qtbot.waitSignal(state.coordinate_system_changed) as blocker:
        changed = state.set_coordinate_system("global", source="test")

    assert changed is True
    assert state.coordinate_system == "global"
    assert removed_calls == [{"sdata": sdata_blobs, "coordinate_system": "global"}]
    assert isinstance(blocker.args[0], CoordinateSystemChangedEvent)
    assert blocker.args[0] == CoordinateSystemChangedEvent(
        sdata=sdata_blobs,
        previous_coordinate_system=None,
        coordinate_system="global",
        source="test",
    )

    changed = state.set_coordinate_system("global", source="test")

    assert changed is False
    assert removed_calls == [{"sdata": sdata_blobs, "coordinate_system": "global"}]


def test_harpy_app_state_set_sdata_keeps_previous_coordinate_system_when_still_valid(monkeypatch) -> None:
    first_sdata = object()
    second_sdata = object()
    state = HarpyAppState()
    removed_sdata_calls: list[object] = []
    coordinate_events: list[CoordinateSystemChangedEvent] = []

    coordinate_systems_by_sdata_id = {
        id(first_sdata): ["global", "local"],
        id(second_sdata): ["local", "zeta"],
    }
    monkeypatch.setattr(
        app_state_module,
        "get_coordinate_system_names_from_sdata",
        lambda sdata: coordinate_systems_by_sdata_id[id(sdata)],
    )
    monkeypatch.setattr(
        state.viewer_adapter,
        "remove_layers_for_sdata",
        lambda sdata: removed_sdata_calls.append(sdata) or [],
    )
    state.coordinate_system_changed.connect(coordinate_events.append)

    state.set_sdata(first_sdata)
    assert state.coordinate_system == "global"

    coordinate_events.clear()
    state.set_coordinate_system("local", source="test")
    assert state.coordinate_system == "local"

    coordinate_events.clear()
    removed_sdata_calls.clear()
    state.set_sdata(second_sdata)

    assert state.sdata is second_sdata
    assert state.coordinate_system == "local"
    assert removed_sdata_calls == [first_sdata]
    assert coordinate_events == []


def test_harpy_app_state_set_sdata_selects_first_sorted_coordinate_system_when_previous_is_invalid(monkeypatch) -> None:
    first_sdata = object()
    second_sdata = object()
    state = HarpyAppState()
    removed_sdata_calls: list[object] = []
    coordinate_events: list[CoordinateSystemChangedEvent] = []

    coordinate_systems_by_sdata_id = {
        id(first_sdata): ["local"],
        id(second_sdata): ["alpha", "beta"],
    }
    monkeypatch.setattr(
        app_state_module,
        "get_coordinate_system_names_from_sdata",
        lambda sdata: coordinate_systems_by_sdata_id[id(sdata)],
    )
    monkeypatch.setattr(
        state.viewer_adapter,
        "remove_layers_for_sdata",
        lambda sdata: removed_sdata_calls.append(sdata) or [],
    )
    state.coordinate_system_changed.connect(coordinate_events.append)

    state.set_sdata(first_sdata)
    assert state.coordinate_system == "local"

    coordinate_events.clear()
    removed_sdata_calls.clear()
    state.set_sdata(second_sdata)

    assert state.sdata is second_sdata
    assert state.coordinate_system == "alpha"
    assert removed_sdata_calls == [first_sdata]
    assert coordinate_events == [
        CoordinateSystemChangedEvent(
            sdata=second_sdata,
            previous_coordinate_system="local",
            coordinate_system="alpha",
            source="set_sdata",
        )
    ]

    coordinate_events.clear()
    removed_sdata_calls.clear()
    state.clear_sdata()

    assert state.sdata is None
    assert state.coordinate_system is None
    assert removed_sdata_calls == [second_sdata]
    assert coordinate_events == [
        CoordinateSystemChangedEvent(
            sdata=None,
            previous_coordinate_system="alpha",
            coordinate_system=None,
            source="set_sdata",
        )
    ]


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
