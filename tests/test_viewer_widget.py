from __future__ import annotations

from collections.abc import Callable
from html import unescape
from types import SimpleNamespace

from qtpy.QtCore import Qt
from qtpy.QtGui import QColor
from qtpy.QtWidgets import QComboBox

import napari_harpy.widgets._viewer_widget as viewer_widget_module
from napari_harpy._table_color_source import TableColorSourceSpec
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


_FEEDBACK_BACKGROUND_BY_KIND = {
    "info": "#eef6ff",
    "warning": "#fffbeb",
    "success": "#ecfdf5",
    "error": "#fef2f2",
}


def _assert_action_feedback_card(widget: ViewerWidget, *, title: str, kind: str) -> None:
    assert title in widget.action_feedback_label.text()
    assert f"background-color: {_FEEDBACK_BACKGROUND_BY_KIND[kind]}" in widget.action_feedback_label.styleSheet()
    assert not widget.action_feedback_label.isHidden()


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


def test_elided_tool_button_only_shows_tooltip_when_text_is_truncated(qtbot, monkeypatch) -> None:
    button = viewer_widget_module._ElidedToolButton("blobs_image_long_name_blobs_image_long_name")

    qtbot.addWidget(button)

    class _FakeRect:
        def __init__(self, width: int) -> None:
            self._width = width

        def width(self) -> int:
            return self._width

    class _FakeFontMetrics:
        def elidedText(self, text: str, mode: object, width: int) -> str:
            del mode
            return text if width >= len(text) else "blobs_image..."

    monkeypatch.setattr(button, "fontMetrics", lambda: _FakeFontMetrics())
    monkeypatch.setattr(button, "contentsRect", lambda: _FakeRect(400))
    button.refresh_elision()

    assert button.toolTip() == ""

    monkeypatch.setattr(button, "contentsRect", lambda: _FakeRect(20))
    button.refresh_elision()

    tooltip = unescape(button.toolTip()).replace("&#8203;", "").replace("\u200b", "")
    assert "blobs_image_long_name_blobs_image_long_name" in tooltip
    assert "collapsed" not in tooltip
    assert "..." in button.text() or "\u2026" in button.text()


def test_overlay_color_button_uses_color_dialog_selection(qtbot, monkeypatch) -> None:
    button = viewer_widget_module._OverlayColorButton("#00FFFF")

    qtbot.addWidget(button)

    monkeypatch.setattr(
        viewer_widget_module.QColorDialog,
        "getColor",
        lambda *args, **kwargs: QColor("#123456"),
    )

    button.choose_color()

    assert button.current_color == "#123456"
    assert "background-color: #123456" in button.styleSheet()
    assert "Current color" in button.toolTip()


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
    assert widget.image_cards[0].channel_color_buttons[0].current_color == "#00FFFF"
    assert "background-color: #00FFFF" in widget.image_cards[0].channel_color_buttons[0].styleSheet()
    assert "Cyan" in widget.image_cards[0].channel_color_buttons[0].toolTip()
    assert len(widget.image_rows) == 2
    assert len(widget.labels_rows) == 2
    assert widget.images_section_toggle.text() == "Images (2)"
    assert widget.labels_section_toggle.text() == "Segmentations (2)"
    assert not widget.images_group.is_expanded()
    assert not widget.labels_group.is_expanded()
    assert widget.image_rows[0].detail_widget.isHidden()
    assert widget.labels_rows[0].detail_widget.isHidden()
    assert widget.labels_cards[0].linked_table_combo.count() == 1
    assert widget.labels_cards[0].linked_table_combo.itemText(0) == "table"
    assert widget.labels_cards[0].linked_table_combo.sizeAdjustPolicy() == (
        QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
    )
    assert widget.labels_cards[1].linked_table_combo.count() == 1
    assert widget.labels_cards[1].linked_table_combo.itemText(0) == "No linked tables"
    assert not widget.labels_cards[1].linked_table_combo.isEnabled()
    assert "In coordinate system `global`" in widget.summary_label.text()


def test_viewer_widget_progressive_disclosure_expands_sections_and_elements(qtbot, sdata_blobs) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)

    qtbot.addWidget(widget)

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(sdata_blobs)

    first_image_row = widget.image_rows[0]
    second_image_row = widget.image_rows[1]
    first_labels_row = widget.labels_rows[0]

    assert widget.images_group.content_widget.isHidden()
    assert widget.labels_group.content_widget.isHidden()
    assert first_image_row.detail_widget.isHidden()
    assert first_labels_row.detail_widget.isHidden()
    assert widget.images_section_toggle.arrowType() == Qt.ArrowType.NoArrow
    assert not widget.images_section_toggle.icon().isNull()

    widget.images_section_toggle.click()

    assert widget.images_group.is_expanded()
    assert not widget.images_group.content_widget.isHidden()
    assert first_image_row.detail_widget.isHidden()

    first_image_row.toggle_button.click()

    assert first_image_row.is_expanded()
    assert not first_image_row.detail_widget.isHidden()
    assert widget.image_cards[0].stack_toggle.isChecked()

    second_image_row.toggle_button.click()

    assert first_image_row.is_expanded()
    assert not first_image_row.detail_widget.isHidden()
    assert second_image_row.is_expanded()
    assert not second_image_row.detail_widget.isHidden()

    widget.labels_section_toggle.click()
    first_labels_row.toggle_button.click()

    assert widget.labels_group.is_expanded()
    assert first_labels_row.is_expanded()
    assert not first_labels_row.detail_widget.isHidden()
    assert widget.labels_cards[0].linked_table_combo.currentText() == "table"


def test_viewer_widget_progressive_disclosure_actions_still_load_layers(qtbot, sdata_blobs) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)

    qtbot.addWidget(widget)

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(sdata_blobs)

    widget.images_section_toggle.click()
    widget.image_rows[0].toggle_button.click()
    widget.image_cards[0].add_update_button.click()

    assert len(viewer.layers) == 1
    assert viewer.layers[0].name == "blobs_image"

    widget.labels_section_toggle.click()
    widget.labels_rows[0].toggle_button.click()
    widget.labels_cards[0].add_update_button.click()

    assert len(viewer.layers) == 2
    assert viewer.layers[1].name == "blobs_labels"


def test_viewer_widget_labels_cards_expose_table_driven_coloring_controls(qtbot, sdata_blobs) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)

    qtbot.addWidget(widget)

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(sdata_blobs)

    first_card = widget.labels_cards[0]
    second_card = widget.labels_cards[1]

    assert first_card.color_source_kind_combo.count() == 3
    assert [first_card.color_source_kind_combo.itemText(index) for index in range(3)] == [
        "None",
        "Observations",
        "Vars",
    ]
    assert first_card.color_source_value_input.completer() is not None
    assert not first_card.color_source_value_input.isEnabled()
    assert first_card.action_status_label.text() == "Action: add/update primary labels layer"

    first_card.color_source_kind_combo.setCurrentIndex(1)
    assert not first_card.color_source_value_input.isEnabled()
    assert first_card.action_status_label.text() == "Action: no colorable observation columns available"

    first_card.color_source_kind_combo.setCurrentIndex(2)
    assert first_card.color_source_value_input.isEnabled()
    assert first_card.color_source_value_input.completer().model().stringList() == [
        "channel_0_sum",
        "channel_1_sum",
        "channel_2_sum",
    ]
    assert first_card.action_status_label.text() == 'Action: add/update colored overlay for X[:, "channel_0_sum"]'

    second_card.color_source_kind_combo.setCurrentIndex(2)
    assert second_card.action_status_label.text() == "Action: colored overlays require a linked table"


def test_viewer_widget_labels_card_repopulates_color_sources_when_linked_table_changes(qtbot, monkeypatch) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)
    fake_sdata = object()

    qtbot.addWidget(widget)

    monkeypatch.setattr(viewer_widget_module, "_get_coordinate_systems_from_sdata", lambda sdata: ["global"])
    monkeypatch.setattr(viewer_widget_module, "_get_images_in_coordinate_system", lambda sdata, coordinate_system: [])
    monkeypatch.setattr(viewer_widget_module, "_get_labels_in_coordinate_system", lambda sdata, coordinate_system: ["labels"])
    monkeypatch.setattr(viewer_widget_module, "get_annotating_table_names", lambda sdata, label_name: ["table_a", "table_b"])
    monkeypatch.setattr(
        viewer_widget_module,
        "get_table_color_source_options",
        lambda sdata, table_name: (
            [TableColorSourceSpec(table_name=table_name, source_kind="obs_column", value_key="cell_type", value_kind="categorical")]
            if table_name == "table_a"
            else [TableColorSourceSpec(table_name=table_name, source_kind="x_var", value_key="GeneA", value_kind="continuous")]
        ),
    )

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(fake_sdata)

    card = widget.labels_cards[0]

    card.color_source_kind_combo.setCurrentIndex(1)
    assert card.color_source_value_input.isEnabled()
    assert card.color_source_value_input.completer().model().stringList() == [
        "cell_type"
    ]
    assert card.action_status_label.text() == 'Action: add/update colored overlay for obs["cell_type"]'

    card.linked_table_combo.setCurrentIndex(1)
    assert not card.color_source_value_input.isEnabled()
    assert card.action_status_label.text() == "Action: no colorable observation columns available"

    card.color_source_kind_combo.setCurrentIndex(2)
    assert card.color_source_value_input.isEnabled()
    assert card.color_source_value_input.completer().model().stringList() == [
        "GeneA"
    ]
    assert card.action_status_label.text() == 'Action: add/update colored overlay for X[:, "GeneA"]'


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
    assert image_card.channel_section_label.text() == "Channels"
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
    monkeypatch.setattr(
        viewer_widget_module,
        "get_table_color_source_options",
        lambda sdata, table_name: [],
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
    _assert_action_feedback_card(widget, title="Segmentation Loaded", kind="success")
    assert "Loaded segmentation `blobs_labels`" in widget.action_feedback_label.text()


def test_viewer_widget_add_update_labels_dispatches_to_styled_overlay_path(qtbot, monkeypatch, sdata_blobs) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)
    recorded_requests: list[object] = []

    qtbot.addWidget(widget)

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(sdata_blobs)

    monkeypatch.setattr(widget, "_add_or_update_styled_labels_layer", lambda request: recorded_requests.append(request))

    first_card = widget.labels_cards[0]
    first_card.color_source_kind_combo.setCurrentIndex(2)
    first_card.color_source_value_input.setText("channel_1_sum")
    first_card.add_update_button.click()

    assert len(recorded_requests) == 1
    request = recorded_requests[0]
    assert request.label_name == "blobs_labels"
    assert request.table_name == "table"
    assert request.selected_source_kind == "x_var"
    assert request.selected_color_source is not None
    assert request.selected_color_source.value_key == "channel_1_sum"


def test_viewer_widget_add_update_labels_creates_and_updates_styled_overlay(qtbot, sdata_blobs) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)
    table = sdata_blobs["table"]
    table.obs["cell_type"] = ["odd" if instance_id % 2 else "even" for instance_id in table.obs["instance_id"]]
    table.obs["cell_type"] = table.obs["cell_type"].astype("category")
    table.uns["cell_type_colors"] = ["#ff0000", "#00ff00"]

    qtbot.addWidget(widget)

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(sdata_blobs)

    first_card = widget.labels_cards[0]
    first_card.color_source_kind_combo.setCurrentIndex(1)
    first_card.color_source_value_input.setText("cell_type")

    first_card.add_update_button.click()

    assert len(viewer.layers) == 1
    layer = viewer.layers[0]
    binding = widget.app_state.viewer_adapter.layer_bindings.get_binding(layer)
    assert binding is not None
    assert binding.labels_role == "styled"
    _assert_action_feedback_card(widget, title="Colored Overlay Created", kind="success")
    assert "Created colored overlay for obs[\"cell_type\"]" in widget.action_feedback_label.text()
    assert "stored categorical palette" in widget.action_feedback_label.text()

    first_card.add_update_button.click()

    assert len(viewer.layers) == 1
    assert viewer.layers[0] is layer
    _assert_action_feedback_card(widget, title="Colored Overlay Updated", kind="success")
    assert "Updated colored overlay for obs[\"cell_type\"]" in widget.action_feedback_label.text()


def test_viewer_widget_styled_overlay_missing_palette_uses_info_card(qtbot, sdata_blobs) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)
    table = sdata_blobs["table"]
    table.obs["cell_type"] = ["odd" if instance_id % 2 else "even" for instance_id in table.obs["instance_id"]]
    table.obs["cell_type"] = table.obs["cell_type"].astype("category")

    qtbot.addWidget(widget)

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(sdata_blobs)

    first_card = widget.labels_cards[0]
    first_card.color_source_kind_combo.setCurrentIndex(1)
    first_card.color_source_value_input.setText("cell_type")

    first_card.add_update_button.click()

    _assert_action_feedback_card(widget, title="Colored Overlay Created", kind="info")
    assert "no stored palette was present" in widget.action_feedback_label.text()


def test_viewer_widget_styled_overlay_invalid_palette_uses_warning_card(qtbot, sdata_blobs) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)
    table = sdata_blobs["table"]
    table.obs["cell_type"] = ["odd"] * table.n_obs
    table.obs["cell_type"] = table.obs["cell_type"].astype("category")
    table.uns["cell_type_colors"] = ["#ff0000", "#00ff00"]

    qtbot.addWidget(widget)

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(sdata_blobs)

    first_card = widget.labels_cards[0]
    first_card.color_source_kind_combo.setCurrentIndex(1)
    first_card.color_source_value_input.setText("cell_type")

    first_card.add_update_button.click()

    _assert_action_feedback_card(widget, title="Colored Overlay Created With Warning", kind="warning")
    assert "stored categorical palette was invalid" in widget.action_feedback_label.text()


def test_viewer_widget_styled_overlay_string_coercion_uses_warning_card(qtbot, sdata_blobs) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)
    table = sdata_blobs["table"]
    table.obs["sample_type"] = [
        "odd" if instance_id % 2 else "even" for instance_id in table.obs["instance_id"]
    ]

    qtbot.addWidget(widget)

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(sdata_blobs)

    first_card = widget.labels_cards[0]
    first_card.color_source_kind_combo.setCurrentIndex(1)
    first_card.color_source_value_input.setText("sample_type")

    first_card.add_update_button.click()

    _assert_action_feedback_card(widget, title="Colored Overlay Created With Warning", kind="warning")
    assert "Coerced string values to categorical" in widget.action_feedback_label.text()


def test_viewer_widget_styled_overlay_precondition_error_uses_error_card(qtbot, sdata_blobs) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)

    qtbot.addWidget(widget)

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(sdata_blobs)

    first_card = widget.labels_cards[0]
    first_card.color_source_kind_combo.setCurrentIndex(1)

    first_card.add_update_button.click()

    _assert_action_feedback_card(widget, title="Colored Overlay Error", kind="error")
    assert "Select an observation column" in widget.action_feedback_label.text()


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
    _assert_action_feedback_card(widget, title="Image Loaded", kind="success")
    assert "Loaded image `blobs_image` in stack mode" in widget.action_feedback_label.text()


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
    image_card.channel_color_buttons[0].set_color("#00FFFF")
    image_card.channel_color_buttons[2].set_color("#FFA500")

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
