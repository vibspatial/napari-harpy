from __future__ import annotations

from html import unescape

from qtpy.QtCore import Qt
from qtpy.QtWidgets import QComboBox, QLabel, QWidget
from spatialdata import SpatialData

import napari_harpy.core.histogram as histogram_core
import napari_harpy.widgets.histogram.widget as histogram_widget_module
from napari_harpy._app_state import get_or_create_app_state
from napari_harpy.core.spatialdata import get_spatialdata_image_options_for_coordinate_system_from_sdata
from napari_harpy.widgets.histogram.widget import HistogramCalculationRequest, HistogramWidget


class DummyViewer:
    pass


def combo_texts(combo: QComboBox) -> list[str]:
    return [combo.itemText(index) for index in range(combo.count())]


def set_combo_data(combo: QComboBox, data: str) -> None:
    for index in range(combo.count()):
        if combo.itemData(index) == data:
            combo.setCurrentIndex(index)
            return
    raise AssertionError(f"Combo data {data!r} is not available.")


def tooltip_text(widget) -> str:
    return unescape(widget.toolTip()).replace("&#8203;", "").replace("\u200b", "")


def make_widget_with_sdata(qtbot, sdata: SpatialData) -> HistogramWidget:
    viewer = DummyViewer()
    get_or_create_app_state(viewer).set_sdata(sdata)
    widget = HistogramWidget(viewer)
    qtbot.addWidget(widget)
    return widget


def add_valid_histogram_card(widget: HistogramWidget) -> tuple[str, object]:
    card_id = widget.add_histogram_card()
    card = widget._cards[card_id]
    set_combo_data(card.image_combo, "blobs_image")
    set_combo_data(card.channel_combo, "0")
    return card_id, card


def test_histogram_widget_instantiates_without_viewer(qtbot) -> None:
    widget = HistogramWidget()

    qtbot.addWidget(widget)

    assert widget.app_state.sdata is None
    assert widget.card_count == 0
    assert not widget.empty_state_label.isHidden()
    assert widget.add_button.text() == "Add histogram"
    assert widget.findChild(QLabel, "histogram_header_logo") is not None
    assert widget.findChild(QLabel, "histogram_title") is None
    action_row = widget.findChild(QWidget, "histogram_header_action_row")
    assert action_row is not None
    assert action_row.layout().itemAt(0).widget() is widget.add_button
    assert action_row.layout().itemAt(1).spacerItem() is not None


def test_histogram_widget_attaches_to_shared_app_state(qtbot) -> None:
    viewer = DummyViewer()
    app_state = get_or_create_app_state(viewer)

    widget = HistogramWidget(viewer)
    qtbot.addWidget(widget)

    assert widget.app_state is app_state


def test_histogram_cards_can_be_added_and_removed_without_mutating_sdata(qtbot, sdata_blobs: SpatialData) -> None:
    widget = make_widget_with_sdata(qtbot, sdata_blobs)
    image_names_before = tuple(sdata_blobs.images)

    card_id = widget.add_histogram_card()
    card = widget._cards[card_id]

    assert widget.card_count == 1
    assert widget.empty_state_label.isHidden()
    assert "Remove histogram" in tooltip_text(card.remove_button)
    assert card.remove_button.accessibleName() == "Remove histogram"

    card.remove_button.click()

    assert widget.card_count == 0
    assert not widget.empty_state_label.isHidden()
    assert tuple(sdata_blobs.images) == image_names_before


def test_histogram_widget_populates_target_selectors_and_emits_request(
    qtbot,
    monkeypatch,
    sdata_blobs: SpatialData,
) -> None:
    def fail_calculate(*args, **kwargs):
        raise AssertionError("Slice 2 must not call calculate_histogram(...).")

    monkeypatch.setattr(histogram_core, "calculate_histogram", fail_calculate)
    widget = make_widget_with_sdata(qtbot, sdata_blobs)

    card_id, card = add_valid_histogram_card(widget)

    assert card.coordinate_system_combo.currentText() == "global"
    assert combo_texts(card.image_combo) == ["blobs_image", "blobs_multiscale_image"]
    assert combo_texts(card.channel_combo) == ["0", "1", "2"]
    assert card.calculate_button.isEnabled()

    with qtbot.waitSignal(widget.calculation_requested) as blocker:
        qtbot.mouseClick(card.calculate_button, Qt.MouseButton.LeftButton)

    request = blocker.args[0]
    assert isinstance(request, HistogramCalculationRequest)
    assert request.card_id == card_id
    assert request.target.coordinate_system == "global"
    assert request.target.image_name == "blobs_image"
    assert request.target.channel_name == "0"
    assert request.settings.bins == 256
    assert request.settings.scale == "scale0"
    assert request.settings.percentiles == ()
    assert "Calculation request emitted." in card.status_label.text()


def test_histogram_widget_refresh_preserves_valid_target_and_clears_invalid_downstream_selection(
    qtbot,
    monkeypatch,
    sdata_blobs: SpatialData,
) -> None:
    widget = make_widget_with_sdata(qtbot, sdata_blobs)
    _card_id, card = add_valid_histogram_card(widget)

    widget._on_sdata_changed(sdata_blobs)

    assert card.image_combo.currentData() == "blobs_image"
    assert card.channel_combo.currentData() == "0"

    monkeypatch.setattr(
        histogram_widget_module,
        "get_spatialdata_image_options_for_coordinate_system_from_sdata",
        lambda *, sdata, coordinate_system: [
            option
            for option in get_spatialdata_image_options_for_coordinate_system_from_sdata(
                sdata=sdata,
                coordinate_system=coordinate_system,
            )
            if option.image_name == "blobs_multiscale_image"
        ],
    )

    widget._on_sdata_changed(sdata_blobs)

    assert card.image_combo.currentIndex() == -1
    assert card.channel_combo.currentIndex() == -1
    assert not card.calculate_button.isEnabled()


def test_histogram_widget_rejects_images_without_channel_names(
    qtbot,
    monkeypatch,
    sdata_blobs: SpatialData,
) -> None:
    monkeypatch.setattr(histogram_widget_module, "get_image_channel_names_from_sdata", lambda sdata, image_name: [])
    widget = make_widget_with_sdata(qtbot, sdata_blobs)

    card_id = widget.add_histogram_card()
    card = widget._cards[card_id]
    set_combo_data(card.image_combo, "blobs_image")

    assert card.channel_combo.count() == 0
    assert not card.calculate_button.isEnabled()
    assert "does not expose channel names" in card.status_label.text()


def test_histogram_widget_settings_are_collapsed_optional_and_card_local(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    widget = make_widget_with_sdata(qtbot, sdata_blobs)
    first_id, first_card = add_valid_histogram_card(widget)
    second_id, second_card = add_valid_histogram_card(widget)

    assert first_card.settings_panel.isHidden()
    assert first_card.settings_toggle.text() == "Settings"
    assert "scale: scale0" in tooltip_text(first_card.settings_toggle)
    assert "bins: 256" in tooltip_text(first_card.settings_toggle)

    first_card.bins_spin.setValue(512)
    first_card.value_range_low_edit.setText("0.1")
    first_card.value_range_high_edit.setText("0.9")
    first_card.percentile_min_edit.setText("1")
    first_card.percentile_max_edit.setText("99")
    first_card.density_checkbox.setChecked(True)
    first_card.exclude_zeros_checkbox.setChecked(True)
    first_card.log_y_checkbox.setChecked(True)
    second_card.bins_spin.setValue(128)

    first_request = widget.build_request_for_card(first_id)
    second_request = widget.build_request_for_card(second_id)

    assert first_request.settings.bins == 512
    assert first_request.settings.value_range == (0.1, 0.9)
    assert first_request.settings.percentiles == (1.0, 99.0)
    assert first_request.settings.density is True
    assert first_request.settings.exclude_zeros is True
    assert first_request.settings.log_y is True
    assert second_request.settings.bins == 128
    assert second_request.settings.value_range is None
    assert second_request.settings.percentiles == ()
    first_settings_tooltip = tooltip_text(first_card.settings_toggle)
    assert first_card.settings_toggle.text() == "Settings"
    assert "value_range: (0.1, 0.9)" in first_settings_tooltip
    assert "density: True" in first_settings_tooltip
    assert "exclude_zeros: True" in first_settings_tooltip
    assert "log_y: True" in first_settings_tooltip
    assert "percentiles: 1, 99" in first_settings_tooltip

    first_card.reset_settings_button.click()

    reset_settings = widget.build_request_for_card(first_id).settings
    assert reset_settings.bins == 256
    assert reset_settings.value_range is None
    assert reset_settings.percentiles == ()
    assert reset_settings.density is False
    assert reset_settings.exclude_zeros is False
    assert reset_settings.log_y is False
    assert reset_settings.scale == "scale0"
    assert widget.build_request_for_card(second_id).settings.bins == 128


def test_histogram_widget_settings_panel_stays_compact_at_dock_width(qtbot, sdata_blobs: SpatialData) -> None:
    widget = make_widget_with_sdata(qtbot, sdata_blobs)
    _card_id, card = add_valid_histogram_card(widget)

    closed_icon_key = card.settings_toggle.icon().cacheKey()
    assert card.settings_toggle.arrowType() == Qt.ArrowType.NoArrow
    assert not card.settings_toggle.icon().isNull()

    card.settings_toggle.setChecked(True)
    card.settings_panel.layout().activate()

    assert card.settings_toggle.icon().cacheKey() != closed_icon_key
    assert "QToolButton:checked" in card.settings_toggle.styleSheet()
    assert "min-height: 26px" in card.settings_toggle.styleSheet()
    assert "padding: 3px 10px" in card.settings_toggle.styleSheet()
    assert "font-size: 13px" in card.settings_toggle.styleSheet()
    assert card.settings_panel.sizeHint().width() <= 340
    assert card.scale_combo.maximumWidth() == 150
    assert card.value_range_low_edit.maximumWidth() == 112
    assert card.value_range_high_edit.maximumWidth() == 112


def test_histogram_widget_invalid_optional_settings_disable_calculate(qtbot, sdata_blobs: SpatialData) -> None:
    widget = make_widget_with_sdata(qtbot, sdata_blobs)
    _card_id, card = add_valid_histogram_card(widget)

    card.value_range_low_edit.setText("0.1")

    assert not card.calculate_button.isEnabled()
    assert "requires both low and high values" in card.status_label.text()
