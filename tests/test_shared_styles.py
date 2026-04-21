from qtpy.QtWidgets import QSizePolicy

from napari_harpy.widgets._shared_styles import CompactComboBox, build_input_control_stylesheet


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
