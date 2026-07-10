from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QCompleter

from napari_harpy.widgets.viewer.points_widget import PointsValueWidget


def _fake_controller(
    *,
    can_load_values: bool = True,
    can_visualize: bool = True,
    is_loading: bool = False,
):
    return SimpleNamespace(
        can_load_values=can_load_values,
        can_visualize=can_visualize,
        is_loading=is_loading,
        is_loading_values=is_loading,
        status_message="Points: values are ready.",
        status_kind="success",
    )


def _fake_value_source(values: list[str]):
    return SimpleNamespace(
        value_table=SimpleNamespace(
            values=pd.DataFrame({"value": values}),
        )
    )


def test_points_value_widget_adds_selected_values_in_order(qtbot) -> None:
    widget = PointsValueWidget()
    recorded_requests: list[tuple[object, int]] = []

    qtbot.addWidget(widget)
    widget.add_update_requested.connect(lambda values, budget: recorded_requests.append((values, budget)))

    widget.set_points_names(["transcripts"])
    widget.set_index_columns(["gene"])
    widget.set_value_source(_fake_value_source(["AAMP", "AXL", "MALAT1"]))
    widget.render_controller_state(_fake_controller())
    widget.value_input.setText("AAMP")
    qtbot.mouseClick(widget.add_value_button, Qt.MouseButton.LeftButton)
    widget.value_input.setText("AXL")
    qtbot.mouseClick(widget.add_value_button, Qt.MouseButton.LeftButton)
    widget.value_input.setText("AAMP")
    qtbot.mouseClick(widget.add_value_button, Qt.MouseButton.LeftButton)
    widget.render_point_budget_input.setText("25_000")

    assert widget.add_update_button.text() == "Add / Update in viewer"
    assert widget.selected_values() == ("AAMP", "AXL")
    assert widget.selected_values_summary_label.text() == "AAMP\nAXL"

    qtbot.mouseClick(widget.add_update_button, Qt.MouseButton.LeftButton)

    assert recorded_requests == [(("AAMP", "AXL"), 25_000)]


def test_points_value_widget_value_input_browses_and_filters_values(qtbot) -> None:
    widget = PointsValueWidget()
    values = [f"GENE{index:02d}" for index in range(20)]

    qtbot.addWidget(widget)

    widget.set_points_names(["transcripts"])
    widget.set_index_columns(["gene"])
    widget.set_value_source(_fake_value_source(values))
    widget.render_controller_state(_fake_controller())

    completer = widget.value_input.completer()

    assert widget.value_input.isEnabled()
    assert widget.value_input.text() == ""
    assert widget.value_input.placeholderText() == "Select value"
    assert completer is not None
    assert completer.completionMode() == QCompleter.CompletionMode.PopupCompletion
    assert completer.maxVisibleItems() == 10

    widget.value_input.show_completion_popup()

    assert completer.completionPrefix() == ""
    assert completer.completionModel().rowCount() == len(values)

    widget.value_input.setText("GENE1")
    widget.value_input.show_completion_popup()

    assert completer.completionPrefix() == "GENE1"
    assert completer.completionModel().rowCount() == 10

    completer.popup().hide()


def test_points_value_widget_all_values_disables_value_input(qtbot) -> None:
    widget = PointsValueWidget()
    recorded_requests: list[tuple[object, int]] = []

    qtbot.addWidget(widget)
    widget.add_update_requested.connect(lambda values, budget: recorded_requests.append((values, budget)))

    widget.set_points_names(["transcripts"])
    widget.set_index_columns(["gene"])
    widget.set_value_source(_fake_value_source(["AAMP", "AXL"]))
    widget.render_controller_state(_fake_controller())

    widget.all_values_checkbox.setChecked(True)

    assert not widget.value_input.isEnabled()
    assert not widget.add_value_button.isEnabled()
    assert not widget.clear_selection_button.isEnabled()

    qtbot.mouseClick(widget.add_update_button, Qt.MouseButton.LeftButton)

    assert recorded_requests == [("all", 100_000)]


def test_points_value_widget_clear_removes_selected_values(qtbot) -> None:
    widget = PointsValueWidget()

    qtbot.addWidget(widget)

    widget.set_points_names(["transcripts"])
    widget.set_index_columns(["gene"])
    widget.set_value_source(_fake_value_source(["AAMP", "AXL"]))
    widget.render_controller_state(_fake_controller())
    widget.value_input.setText("AAMP")
    qtbot.mouseClick(widget.add_value_button, Qt.MouseButton.LeftButton)
    widget.value_input.setText("AXL")
    qtbot.mouseClick(widget.add_value_button, Qt.MouseButton.LeftButton)

    assert widget.selected_values() == ("AAMP", "AXL")

    qtbot.mouseClick(widget.clear_selection_button, Qt.MouseButton.LeftButton)

    assert widget.selected_values() == ()
    assert widget.selected_values_summary_label.text() == "None"


def test_points_value_widget_reloading_values_preserves_valid_and_drops_invalid(qtbot) -> None:
    widget = PointsValueWidget()

    qtbot.addWidget(widget)

    widget.set_points_names(["transcripts"])
    widget.set_index_columns(["gene"])
    widget.set_value_source(_fake_value_source(["AAMP", "AXL"]))
    widget.render_controller_state(_fake_controller())
    widget.value_input.setText("AAMP")
    qtbot.mouseClick(widget.add_value_button, Qt.MouseButton.LeftButton)
    widget.value_input.setText("AXL")
    qtbot.mouseClick(widget.add_value_button, Qt.MouseButton.LeftButton)

    widget.set_value_source(_fake_value_source(["AXL", "MALAT1"]))

    assert widget.selected_values() == ("AXL",)
    assert widget.selected_values_summary_label.text() == "AXL"
    assert "Dropped 1 selected value" in widget.status_label.text()


def test_points_value_widget_unknown_value_is_not_added(qtbot) -> None:
    widget = PointsValueWidget()

    qtbot.addWidget(widget)

    widget.set_points_names(["transcripts"])
    widget.set_index_columns(["gene"])
    widget.set_value_source(_fake_value_source(["AAMP"]))
    widget.render_controller_state(_fake_controller())
    widget.value_input.setText("NOT_A_GENE")

    assert not widget.add_value_button.isEnabled()

    widget.value_input.returnPressed.emit()

    assert widget.selected_values() == ()
    assert "NOT_A_GENE" in widget.status_label.text()


def test_points_value_widget_invalid_render_budget_disables_add_update(qtbot) -> None:
    widget = PointsValueWidget()

    qtbot.addWidget(widget)

    widget.set_points_names(["transcripts"])
    widget.set_index_columns(["gene"])
    widget.set_value_source(_fake_value_source(["AAMP"]))
    widget.render_controller_state(_fake_controller())
    widget.value_input.setText("AAMP")
    qtbot.mouseClick(widget.add_value_button, Qt.MouseButton.LeftButton)

    assert widget.add_update_button.isEnabled()

    widget.render_point_budget_input.setText("10")

    assert not widget.add_update_button.isEnabled()
    assert "Render point budget" in widget.status_label.text()
