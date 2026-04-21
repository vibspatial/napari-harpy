from __future__ import annotations

from html import escape

from qtpy.QtCore import QSize, Qt
from qtpy.QtGui import QColor, QPalette
from qtpy.QtWidgets import (
    QComboBox,
    QLabel,
    QSizePolicy,
    QStyle,
    QStyleOptionComboBox,
    QStylePainter,
    QWidget,
)

WIDGET_SURFACE_COLOR = "#fcf6f3"
WIDGET_SURFACE_STYLESHEET = f"background-color: {WIDGET_SURFACE_COLOR};"
WIDGET_MIN_WIDTH = 370
TOOLTIP_TEXT_COLOR = "#111827"
FORM_LABEL_STYLESHEET = "color: #374151; font-weight: 600; padding-top: 6px;"
ACTION_BUTTON_STYLESHEET = (
    "QPushButton {"
    "background-color: #f7ede8; "
    "border: 1px solid #ddcfc7; "
    "border-radius: 8px; "
    "color: #111827; "
    "font-weight: 600; "
    "padding: 4px 10px; "
    "min-height: 30px;}"
    "QPushButton:hover { background-color: #f3e5de; border-color: #c9b6ac; }"
    "QPushButton:pressed { background-color: #ebd7cf; border-color: #b59a8e; }"
    "QPushButton:disabled { background-color: #faf4f1; border-color: #ede3dd; color: #a8a29e; }"
)
CHECKBOX_STYLESHEET = (
    "QCheckBox {"
    "color: #111827; "
    "font-weight: 500; "
    "spacing: 8px; "
    "background: transparent;}"
    "QCheckBox:disabled { color: #9ca3af; }"
    "QCheckBox::indicator {"
    "width: 16px; "
    "height: 16px; "
    "border-radius: 4px; "
    "border: 1px solid #d8c8bf; "
    "background-color: #fffdfb;}"
    "QCheckBox::indicator:hover { border-color: #c7b2a7; }"
    "QCheckBox::indicator:checked {"
    "border-color: #7aa7bd; "
    "background-color: #8fb6c9;}"
    "QCheckBox::indicator:disabled {"
    "border-color: #e9ddd7; "
    "background-color: #f7efea;}"
)


def build_input_control_stylesheet(control_selector: str) -> str:
    selectors = [selector.strip() for selector in control_selector.split(",")]
    disabled_selector = ", ".join(f"{selector}:disabled" for selector in selectors)
    focus_selector = ", ".join(f"{selector}:focus" for selector in selectors)

    return (
        f"{control_selector} {{"
        "background-color: #fffdfb; "
        "border: 1px solid #ddcfc7; "
        "border-radius: 8px; "
        "color: #111827; "
        "padding: 4px 10px; "
        "min-height: 30px;}"
        f"{disabled_selector} {{"
        "background-color: #f7efea; "
        "border-color: #e9ddd7; "
        "color: #9ca3af;}"
        f"{focus_selector} {{"
        "border-color: #8fb6c9; "
        "background-color: #ffffff;}"
        "QComboBox { padding-right: 24px; }"
        "QComboBox::drop-down {"
        "subcontrol-origin: padding; "
        "subcontrol-position: top right; "
        "width: 24px; "
        "border: 0px; "
        "background: transparent;}"
    )


class CompactComboBox(QComboBox):
    """Combo box with a capped width hint for long item names."""

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        minimum_contents_length: int = 12,
    ) -> None:
        super().__init__(parent)
        self.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.setMinimumContentsLength(minimum_contents_length)
        self.setMinimumWidth(0)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.currentIndexChanged.connect(self._update_current_text_tooltip)
        self._update_current_text_tooltip()

    def sizeHint(self) -> QSize:
        return self._capped_size_hint(super().sizeHint())

    def minimumSizeHint(self) -> QSize:
        return self._capped_size_hint(super().minimumSizeHint())

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._update_current_text_tooltip()

    def paintEvent(self, event) -> None:
        del event
        painter = QStylePainter(self)
        option = QStyleOptionComboBox()
        self.initStyleOption(option)
        option.currentText = self._elided_current_text(option)
        painter.drawComplexControl(QStyle.ComplexControl.CC_ComboBox, option)
        painter.drawControl(QStyle.ControlElement.CE_ComboBoxLabel, option)

    def _capped_size_hint(self, hint: QSize) -> QSize:
        cap_width = self.fontMetrics().horizontalAdvance("M" * max(1, self.minimumContentsLength())) + 48
        return QSize(min(hint.width(), cap_width), hint.height())

    def _elided_current_text(self, option: QStyleOptionComboBox | None = None) -> str:
        if option is None:
            option = QStyleOptionComboBox()
            self.initStyleOption(option)

        text_rect = self.style().subControlRect(
            QStyle.ComplexControl.CC_ComboBox,
            option,
            QStyle.SubControl.SC_ComboBoxEditField,
            self,
        )
        available_width = max(0, text_rect.width())
        return self.fontMetrics().elidedText(option.currentText, Qt.TextElideMode.ElideRight, available_width)

    def _update_current_text_tooltip(self, _index: int | None = None) -> None:
        current_text = self.currentText()
        if not current_text:
            self.setToolTip("")
            return

        elided_text = self._elided_current_text()
        self.setToolTip(format_tooltip(current_text) if elided_text != current_text else "")


def apply_widget_surface(widget: QWidget) -> None:
    widget.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
    widget.setAutoFillBackground(True)
    palette = widget.palette()
    palette.setColor(QPalette.ColorRole.Window, QColor(WIDGET_SURFACE_COLOR))
    widget.setPalette(palette)
    widget.setStyleSheet(WIDGET_SURFACE_STYLESHEET)


def apply_scroll_content_surface(widget: QWidget) -> None:
    widget.setAutoFillBackground(True)
    palette = widget.palette()
    palette.setColor(QPalette.ColorRole.Window, QColor(WIDGET_SURFACE_COLOR))
    widget.setPalette(palette)
    widget.setStyleSheet(WIDGET_SURFACE_STYLESHEET)


def create_form_label(text: str) -> QLabel:
    label = QLabel(text)
    label.setStyleSheet(FORM_LABEL_STYLESHEET)
    label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
    return label


def format_tooltip(message: str) -> str:
    return f"<qt><span style='color: {TOOLTIP_TEXT_COLOR};'>{escape(message)}</span></qt>"


def format_feedback_identifier(name: str, *, max_length: int = 56) -> tuple[str, bool]:
    """Return a visible identifier and whether it was shortened."""
    if len(name) <= max_length:
        return name, False

    head_length = 32
    tail_length = max_length - head_length - 1
    return f"{name[:head_length]}…{name[-tail_length:]}", True
