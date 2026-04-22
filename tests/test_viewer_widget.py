from __future__ import annotations

from collections.abc import Callable
from html import unescape
from types import SimpleNamespace

from qtpy.QtCore import Qt
from qtpy.QtWidgets import QComboBox

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


def test_elided_label_only_shows_tooltip_when_text_is_truncated(qtbot, monkeypatch) -> None:
    label = viewer_widget_module._ElidedLabel("blobs_multiscale_image")

    qtbot.addWidget(label)

    class _FakeRect:
        def __init__(self, width: int) -> None:
            self._width = width

        def width(self) -> int:
            return self._width

    class _FakeFontMetrics:
        def elidedText(self, text: str, mode: object, width: int) -> str:
            del mode
            return text if width >= len(text) else "blobs_multiscale…"

    monkeypatch.setattr(label, "fontMetrics", lambda: _FakeFontMetrics())
    monkeypatch.setattr(label, "contentsRect", lambda: _FakeRect(400))
    label._update_elided_text()

    assert label.toolTip() == ""

    monkeypatch.setattr(label, "contentsRect", lambda: _FakeRect(10))
    label._update_elided_text()

    tooltip = unescape(label.toolTip()).replace("&#8203;", "").replace("\u200b", "")
    assert "blobs_multiscale_image" in tooltip
    assert "..." in label.text() or "\u2026" in label.text()


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
    assert widget.image_cards[0].stack_toggle.text() == "stack"
    assert widget.image_cards[0].stack_toggle.isChecked()
    assert widget.image_cards[0].overlay_toggle.text() == "overlay"
    assert not widget.image_cards[0].overlay_toggle.isChecked()
    assert widget.labels_cards[0].linked_table_combo.count() == 1
    assert widget.labels_cards[0].linked_table_combo.itemText(0) == "table"
    assert widget.labels_cards[0].linked_table_combo.sizeAdjustPolicy() == (
        QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
    )
    assert widget.labels_cards[1].linked_table_combo.count() == 1
    assert widget.labels_cards[1].linked_table_combo.itemText(0) == "No linked tables"
    assert not widget.labels_cards[1].linked_table_combo.isEnabled()
    assert "In coordinate system `global`" in widget.summary_label.text()


def test_viewer_widget_image_mode_toggles_are_mutually_exclusive(qtbot, sdata_blobs) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)

    qtbot.addWidget(widget)

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(sdata_blobs)

    image_card = widget.image_cards[0]

    assert image_card.stack_toggle.isChecked()
    assert not image_card.overlay_toggle.isChecked()
    assert image_card.channel_panel.isHidden()
    assert image_card.add_update_button.isEnabled()

    image_card.overlay_toggle.setChecked(True)

    assert not image_card.stack_toggle.isChecked()
    assert image_card.overlay_toggle.isChecked()
    assert not image_card.channel_panel.isHidden()
    assert image_card.add_update_button.isEnabled()

    image_card.stack_toggle.setChecked(True)

    assert image_card.stack_toggle.isChecked()
    assert not image_card.overlay_toggle.isChecked()
    assert image_card.channel_panel.isHidden()
    assert image_card.add_update_button.isEnabled()


def test_viewer_widget_overlay_channel_panel_scrolls_when_many_channels(qtbot, monkeypatch) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)
    fake_sdata = object()
    many_channels = [f"c{i}" for i in range(12)]

    qtbot.addWidget(widget)

    monkeypatch.setattr(viewer_widget_module, "_get_coordinate_systems_from_sdata", lambda sdata: ["global"])
    monkeypatch.setattr(viewer_widget_module, "_get_labels_in_coordinate_system", lambda sdata, coordinate_system: [])
    monkeypatch.setattr(viewer_widget_module, "_get_images_in_coordinate_system", lambda sdata, coordinate_system: ["image"])
    monkeypatch.setattr(viewer_widget_module, "get_image_channel_names_from_sdata", lambda sdata, image_name: many_channels)

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(fake_sdata)

    image_card = widget.image_cards[0]

    assert len(image_card.channel_checkboxes) == len(many_channels)
    assert image_card.channel_scroll_area.verticalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAsNeeded
    assert image_card.channel_scroll_area.maximumHeight() > 0
    assert image_card.channel_scroll_area.maximumHeight() < image_card.channel_list_widget.sizeHint().height()


def test_viewer_widget_surfaces_duplicate_channel_names_and_disables_overlay(qtbot, monkeypatch) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)
    fake_sdata = object()

    qtbot.addWidget(widget)

    monkeypatch.setattr(viewer_widget_module, "_get_coordinate_systems_from_sdata", lambda sdata: ["global"])
    monkeypatch.setattr(viewer_widget_module, "_get_labels_in_coordinate_system", lambda sdata, coordinate_system: [])
    monkeypatch.setattr(viewer_widget_module, "_get_images_in_coordinate_system", lambda sdata, coordinate_system: ["image"])
    monkeypatch.setattr(
        viewer_widget_module,
        "get_image_channel_names_from_sdata",
        lambda sdata, image_name: (_ for _ in ()).throw(
            ValueError(
                "Image element `image` exposes duplicate channel names (`dup`), "
                "which napari-harpy does not support. "
                "Update the channel names in the SpatialData object with "
                "`sdata.set_channel_names(...)`."
            )
        ),
    )

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(fake_sdata)

    image_card = widget.image_cards[0]

    assert image_card.channel_names == []
    assert image_card.channel_error is not None
    assert not image_card.overlay_toggle.isEnabled()
    assert not image_card.channel_warning_label.isHidden()
    assert "sdata.set_channel_names(...)" in image_card.channel_warning_label.text()
    assert "duplicate channel names" in image_card.channel_warning_label.toolTip()


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
    monkeypatch.setattr(viewer_widget_module, "get_image_channel_names_from_sdata", lambda sdata, image_name: ["c0", "c1"])
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


def test_viewer_widget_open_spatialdata_loads_selected_store(qtbot, monkeypatch, sdata_blobs) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)
    recorded_paths: list[str] = []
    recorded_sdata: list[object] = []
    original_set_sdata = widget.app_state.set_sdata

    qtbot.addWidget(widget)

    monkeypatch.setattr(
        viewer_widget_module.QFileDialog,
        "getExistingDirectory",
        lambda *args, **kwargs: "/tmp/example.zarr",
    )
    monkeypatch.setattr(
        viewer_widget_module,
        "read_zarr",
        lambda path: recorded_paths.append(path) or sdata_blobs,
    )

    def wrapped_set_sdata(sdata: object) -> None:
        recorded_sdata.append(sdata)
        original_set_sdata(sdata)

    monkeypatch.setattr(widget.app_state, "set_sdata", wrapped_set_sdata)
    widget._set_action_feedback("Old error", is_error=True)

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.open_sdata_button.click()

    assert recorded_paths == ["/tmp/example.zarr"]
    assert recorded_sdata == [sdata_blobs]
    assert widget.app_state.sdata is sdata_blobs
    assert widget.coordinate_system_combo.count() == 1
    assert widget.coordinate_system_combo.itemText(0) == "global"
    assert widget.action_feedback_label.text() == ""
    assert widget.action_feedback_label.isHidden()


def test_viewer_widget_open_spatialdata_shows_error_when_loading_fails(qtbot, monkeypatch) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)

    qtbot.addWidget(widget)

    monkeypatch.setattr(
        viewer_widget_module.QFileDialog,
        "getExistingDirectory",
        lambda *args, **kwargs: "/tmp/example.zarr",
    )

    def raise_read_error(path: str) -> object:
        raise ValueError(f"bad store at {path}")

    monkeypatch.setattr(viewer_widget_module, "read_zarr", raise_read_error)

    widget.open_sdata_button.click()

    assert widget.app_state.sdata is None
    assert "Could not load SpatialData store" in widget.action_feedback_label.text()
    assert "bad store at /tmp/example.zarr" in widget.action_feedback_label.text()
    assert not widget.action_feedback_label.isHidden()


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


def test_viewer_widget_add_update_image_loads_stack_layer(qtbot, sdata_blobs) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)

    qtbot.addWidget(widget)

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(sdata_blobs)

    first_card = widget.image_cards[0]

    first_card.add_update_button.click()

    assert len(viewer.layers) == 1
    layer = viewer.layers[0]
    assert layer.name == "blobs_image"
    binding = widget.app_state.viewer_adapter.layer_bindings.get_binding(layer)
    assert binding is not None
    assert binding.image_display_mode == "stack"
    assert viewer.layers.selection.active is layer
    assert "Loaded image `blobs_image` in stack mode" in widget.action_feedback_label.text()
    assert not widget.action_feedback_label.isHidden()


def test_viewer_widget_add_update_image_reuses_existing_stack_layer(qtbot, sdata_blobs) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)

    qtbot.addWidget(widget)

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(sdata_blobs)

    first_card = widget.image_cards[0]

    first_card.add_update_button.click()
    first_layer = viewer.layers[0]

    first_card.add_update_button.click()

    assert len(viewer.layers) == 1
    assert viewer.layers[0] is first_layer


def test_viewer_widget_add_update_image_overlay_passes_selected_channels_and_colors(qtbot, monkeypatch) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)
    fake_sdata = object()
    fake_layers = [object(), object()]
    recorded_calls: list[tuple[object, str, str, str, list[int] | None, list[str] | None]] = []
    activated_layers: list[object] = []

    qtbot.addWidget(widget)

    monkeypatch.setattr(viewer_widget_module, "_get_coordinate_systems_from_sdata", lambda sdata: ["global"])
    monkeypatch.setattr(viewer_widget_module, "_get_labels_in_coordinate_system", lambda sdata, coordinate_system: [])
    monkeypatch.setattr(viewer_widget_module, "_get_images_in_coordinate_system", lambda sdata, coordinate_system: ["image"])
    monkeypatch.setattr(
        viewer_widget_module,
        "get_image_channel_names_from_sdata",
        lambda sdata, image_name: ["c0", "c1", "c2"],
    )
    monkeypatch.setattr(
        widget.app_state.viewer_adapter,
        "ensure_image_loaded",
        lambda sdata, image_name, coordinate_system, *, mode, channels=None, channel_colors=None: (
            recorded_calls.append((sdata, image_name, coordinate_system, mode, channels, channel_colors)) or fake_layers
        ),
    )
    monkeypatch.setattr(
        widget.app_state.viewer_adapter,
        "activate_layer",
        lambda layer: activated_layers.append(layer) or True,
    )

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(fake_sdata)

    image_card = widget.image_cards[0]
    image_card.overlay_toggle.setChecked(True)
    image_card.channel_checkboxes[0].setChecked(True)
    image_card.channel_checkboxes[2].setChecked(True)
    image_card.channel_color_combos[0].setCurrentText("#00FFFF")
    image_card.channel_color_combos[2].setCurrentText("#FFA500")

    image_card.add_update_button.click()

    assert recorded_calls == [(fake_sdata, "image", "global", "overlay", [0, 2], ["#00FFFF", "#FFA500"])]
    assert activated_layers == [fake_layers[0]]


def test_viewer_widget_add_update_image_overlay_loads_reuses_and_replaces_layers(qtbot, sdata_blobs) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)

    qtbot.addWidget(widget)

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(sdata_blobs)

    image_card = widget.image_cards[0]

    image_card.overlay_toggle.setChecked(True)
    image_card.channel_checkboxes[0].setChecked(True)
    image_card.channel_checkboxes[2].setChecked(True)
    image_card.add_update_button.click()

    assert len(viewer.layers) == 2
    first_layers = list(viewer.layers)
    assert [layer.name for layer in first_layers] == ["blobs_image[0]", "blobs_image[2]"]
    assert viewer.layers.selection.active is first_layers[0]
    assert "Loaded image `blobs_image` in overlay mode" in widget.action_feedback_label.text()

    image_card.add_update_button.click()

    assert len(viewer.layers) == 2
    assert list(viewer.layers) == first_layers

    image_card.channel_checkboxes[0].setChecked(False)
    image_card.add_update_button.click()

    assert len(viewer.layers) == 1
    assert viewer.layers[0].name == "blobs_image[2]"


def test_viewer_widget_empty_overlay_selection_removes_existing_image_layers(qtbot, sdata_blobs) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)

    qtbot.addWidget(widget)

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(sdata_blobs)

    image_card = widget.image_cards[0]

    image_card.add_update_button.click()

    assert len(viewer.layers) == 1
    assert viewer.layers[0].name == "blobs_image"

    image_card.overlay_toggle.setChecked(True)
    image_card.add_update_button.click()

    assert list(viewer.layers) == []
    assert "Overlay mode requires at least one selected channel." in widget.action_feedback_label.text()
    assert not widget.action_feedback_label.isHidden()


def test_viewer_widget_add_update_image_uses_selected_coordinate_system(qtbot, monkeypatch) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)
    fake_sdata = object()
    fake_layer = object()
    recorded_calls: list[tuple[object, str, str, str]] = []
    activated_layers: list[object] = []

    qtbot.addWidget(widget)

    monkeypatch.setattr(viewer_widget_module, "_get_coordinate_systems_from_sdata", lambda sdata: ["global", "local"])
    monkeypatch.setattr(viewer_widget_module, "_get_labels_in_coordinate_system", lambda sdata, coordinate_system: [])
    monkeypatch.setattr(
        viewer_widget_module,
        "_get_images_in_coordinate_system",
        lambda sdata, coordinate_system: ["image_global"] if coordinate_system == "global" else ["image_local"],
    )
    monkeypatch.setattr(viewer_widget_module, "get_image_channel_names_from_sdata", lambda sdata, image_name: ["c0", "c1"])
    monkeypatch.setattr(
        widget.app_state.viewer_adapter,
        "ensure_image_loaded",
        lambda sdata, image_name, coordinate_system, *, mode, channels=None, channel_colors=None: (
            recorded_calls.append((sdata, image_name, coordinate_system, mode)) or fake_layer
        ),
    )
    monkeypatch.setattr(
        widget.app_state.viewer_adapter,
        "activate_layer",
        lambda layer: activated_layers.append(layer) or True,
    )

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(fake_sdata)

    widget.coordinate_system_combo.setCurrentIndex(1)
    image_card = widget.image_cards[0]

    image_card.add_update_button.click()

    assert recorded_calls == [(fake_sdata, "image_local", "local", "stack")]
    assert activated_layers == [fake_layer]


def test_viewer_widget_shares_app_state_for_same_viewer(qtbot) -> None:
    viewer = DummyViewer()
    first = ViewerWidget(viewer)
    second = ViewerWidget(viewer)

    qtbot.addWidget(first)
    qtbot.addWidget(second)

    assert first.app_state is second.app_state
