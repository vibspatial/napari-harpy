import pytest
from qtpy.QtWidgets import QSizePolicy

from spatiato._resources import get_logo_path
from spatiato.widgets.shared_styles import (
    HEADER_LOGO_WIDTH,
    LOGO_NAVY_COLOR,
    CompactComboBox,
    CompleterPopupLineEdit,
    build_input_control_stylesheet,
    create_header_logo,
    format_feedback_identifier,
    format_tooltip,
)


def test_build_input_control_stylesheet_suffixes_each_selector_individually() -> None:
    stylesheet = build_input_control_stylesheet("QComboBox, QLineEdit")

    assert "QComboBox:disabled, QLineEdit:disabled" in stylesheet
    assert "QComboBox:focus, QLineEdit:focus" in stylesheet
    assert "QComboBox, QLineEdit:disabled" not in stylesheet
    assert "QComboBox, QLineEdit:focus" not in stylesheet


def test_compact_combo_box_uses_compact_width_policy(qtbot) -> None:
    combo = CompactComboBox(minimum_contents_length=12)
    qtbot.addWidget(combo)

    assert combo.sizeAdjustPolicy() == CompactComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
    assert combo.minimumContentsLength() == 12
    assert combo.sizePolicy().horizontalPolicy() == QSizePolicy.Policy.Expanding


def test_create_header_logo_shows_packaged_logo_at_header_width(qtbot) -> None:
    assert get_logo_path().is_file()

    logo_label = create_header_logo("test_header_logo")
    qtbot.addWidget(logo_label)

    pixmap = logo_label.pixmap()
    assert logo_label.objectName() == "test_header_logo"
    assert pixmap is not None and not pixmap.isNull()
    assert round(pixmap.width() / pixmap.devicePixelRatio()) == HEADER_LOGO_WIDTH


def test_logo_svg_uses_navy_that_header_logo_recolors() -> None:
    # If the logo's navy changes, the header would silently draw it unreadable on the dark widget surface.
    assert LOGO_NAVY_COLOR in get_logo_path().read_text(encoding="utf-8")


def test_compact_combo_box_elides_long_current_text_and_sets_tooltip(qtbot) -> None:
    combo = CompactComboBox(minimum_contents_length=4)
    combo.addItems(["short", "very_long_item_name_" * 5])
    combo.resize(120, 36)
    qtbot.addWidget(combo)

    combo.setCurrentIndex(0)
    combo._update_current_text_tooltip()

    assert combo.toolTip() == ""
    assert combo._elided_current_text() == "short"

    combo.setCurrentIndex(1)
    combo._update_current_text_tooltip()

    assert combo.toolTip() != ""
    assert combo._elided_current_text() != combo.currentText()


def test_compact_combo_box_uses_placeholder_text_when_current_index_is_unbound(qtbot) -> None:
    combo = CompactComboBox(minimum_contents_length=6)
    combo.setPlaceholderText("Choose segmentation mask")
    combo.addItems(["first"])
    combo.setCurrentIndex(-1)
    combo.resize(180, 36)
    qtbot.addWidget(combo)

    assert combo.currentText() == ""
    assert combo._elided_current_text().startswith("Choose segmentation")
    assert combo.toolTip() == ""


def test_completer_line_edit_clears_text_reinserted_after_accepted_completion(qtbot) -> None:
    line_edit = CompleterPopupLineEdit()
    qtbot.addWidget(line_edit)
    line_edit.setText("AP")

    line_edit.clear_after_accepted_completion("DAPI")

    assert line_edit.text() == ""

    # Match Cocoa Qt's final write after activated callbacks return.
    line_edit.setText("DAPI")
    line_edit._completion_clear_timer.timeout.emit()

    assert line_edit.text() == ""


def test_completer_line_edit_preserves_unrelated_text_after_accepted_completion(qtbot) -> None:
    line_edit = CompleterPopupLineEdit()
    qtbot.addWidget(line_edit)
    line_edit.setText("AP")

    line_edit.clear_after_accepted_completion("DAPI")
    line_edit.setText("CD3")
    line_edit._completion_clear_timer.timeout.emit()

    assert line_edit.text() == "CD3"


def test_format_tooltip_preserves_line_breaks_and_adds_soft_wrap_points() -> None:
    tooltip = format_tooltip("Image: very_long_identifier_name\nCoordinate system: global/test")

    assert "<br>" in tooltip
    assert "max-width: 360px" in tooltip
    assert "_&#8203;" in tooltip
    assert "/&#8203;" in tooltip


def test_format_feedback_identifier_respects_compact_max_length() -> None:
    display_name, was_shortened = format_feedback_identifier("identifier_" + "x" * 80, max_length=32)

    assert was_shortened is True
    assert len(display_name) == 32
    assert "…" in display_name


def test_format_feedback_identifier_rejects_too_small_max_length() -> None:
    with pytest.raises(ValueError, match="at least 4"):
        format_feedback_identifier("identifier", max_length=3)
