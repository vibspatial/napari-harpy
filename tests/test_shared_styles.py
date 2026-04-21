from napari_harpy.widgets._shared_styles import build_input_control_stylesheet


def test_build_input_control_stylesheet_suffixes_each_selector_individually() -> None:
    stylesheet = build_input_control_stylesheet("QComboBox, QLineEdit")

    assert "QComboBox:disabled, QLineEdit:disabled" in stylesheet
    assert "QComboBox:focus, QLineEdit:focus" in stylesheet
    assert "QComboBox, QLineEdit:disabled" not in stylesheet
    assert "QComboBox, QLineEdit:focus" not in stylesheet
