from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
from qtpy.QtCore import Qt

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


def test_points_value_widget_parses_comma_separated_values(qtbot) -> None:
    widget = PointsValueWidget()
    recorded_requests: list[tuple[object, int]] = []

    qtbot.addWidget(widget)
    widget.visualize_requested.connect(lambda values, budget: recorded_requests.append((values, budget)))

    widget.set_points_names(["transcripts"])
    widget.set_index_columns(["gene"])
    widget.set_value_source(_fake_value_source(["AAMP", "AXL", "MALAT1"]))
    widget.render_controller_state(_fake_controller())
    widget.value_input.setText(" AAMP, AXL, AAMP ")
    widget.render_point_budget_input.setText("25_000")

    qtbot.mouseClick(widget.visualize_button, Qt.MouseButton.LeftButton)

    assert recorded_requests == [(("AAMP", "AXL"), 25_000)]


def test_points_value_widget_all_values_disables_value_input(qtbot) -> None:
    widget = PointsValueWidget()
    recorded_requests: list[tuple[object, int]] = []

    qtbot.addWidget(widget)
    widget.visualize_requested.connect(lambda values, budget: recorded_requests.append((values, budget)))

    widget.set_points_names(["transcripts"])
    widget.set_index_columns(["gene"])
    widget.set_value_source(_fake_value_source(["AAMP", "AXL"]))
    widget.render_controller_state(_fake_controller())

    widget.all_values_checkbox.setChecked(True)

    assert not widget.value_input.isEnabled()

    qtbot.mouseClick(widget.visualize_button, Qt.MouseButton.LeftButton)

    assert recorded_requests == [("all", 100_000)]


def test_points_value_widget_invalid_render_budget_disables_visualize(qtbot) -> None:
    widget = PointsValueWidget()

    qtbot.addWidget(widget)

    widget.set_points_names(["transcripts"])
    widget.set_index_columns(["gene"])
    widget.set_value_source(_fake_value_source(["AAMP"]))
    widget.render_controller_state(_fake_controller())
    widget.value_input.setText("AAMP")

    assert widget.visualize_button.isEnabled()

    widget.render_point_budget_input.setText("10")

    assert not widget.visualize_button.isEnabled()
    assert "Render point budget" in widget.status_label.text()
