from qtpy.QtWidgets import QSizePolicy

from napari_harpy.widgets._shared_styles import CompactComboBox, build_input_control_stylesheet, format_tooltip


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


def test_format_tooltip_preserves_line_breaks_and_adds_soft_wrap_points() -> None:
    tooltip = format_tooltip("Image: very_long_identifier_name\nCoordinate system: global/test")

    assert "<br>" in tooltip
    assert "max-width: 360px" in tooltip
    assert "_&#8203;" in tooltip
    assert "/&#8203;" in tooltip
