from __future__ import annotations

import re
from html import escape
from typing import Literal, cast

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

WIDGET_SURFACE_COLOR = "#25272c"
WIDGET_SURFACE_STYLESHEET = f"background-color: {WIDGET_SURFACE_COLOR};"
WIDGET_MIN_WIDTH = 390
WIDGET_PANEL_COLOR = "#30333a"
WIDGET_PANEL_MUTED_COLOR = "#3a3e47"
WIDGET_PANEL_SUBTLE_COLOR = "#2b2e34"
WIDGET_BORDER_COLOR = "#4b515d"
WIDGET_BORDER_STRONG_COLOR = "#67707f"
WIDGET_TEXT_COLOR = "#f2f4f8"
WIDGET_TEXT_MUTED_COLOR = "#aeb6c2"
WIDGET_TEXT_SECONDARY_COLOR = "#d4d9e2"
WIDGET_DISABLED_TEXT_COLOR = "#7f8794"
WIDGET_DISABLED_BORDER_COLOR = "#414651"
WIDGET_ACCENT_COLOR = "#9b7bea"
WIDGET_ACCENT_SOFT_COLOR = "#3c3450"
WIDGET_ACCENT_BORDER_COLOR = "#b196f3"
WIDGET_SELECTION_COLOR = "#4b3f63"
WIDGET_PRESSED_COLOR = "#454a55"
WIDGET_SUCCESS_COLOR = "#70c98a"
WIDGET_SUCCESS_HOVER_COLOR = "#83d99b"
WIDGET_PRIMARY_BUTTON_TEXT_COLOR = "#111827"
WIDGET_WARNING_TEXT_COLOR = "#f0c36a"
WIDGET_WARNING_BACKGROUND_COLOR = "#3f3325"
WIDGET_WARNING_BORDER_COLOR = "#c99845"
WIDGET_WARNING_HOVER_COLOR = "#4f402c"
TOOLTIP_TEXT_COLOR = WIDGET_TEXT_COLOR
FORM_LABEL_STYLESHEET = (
    f"color: {WIDGET_TEXT_SECONDARY_COLOR}; "
    "font-weight: 600; "
    "padding-top: 6px; "
    "background: transparent;"
)
ACTION_BUTTON_STYLESHEET = (
    "QPushButton {"
    f"background-color: {WIDGET_PANEL_COLOR}; "
    f"border: 1px solid {WIDGET_BORDER_COLOR}; "
    "border-radius: 8px; "
    f"color: {WIDGET_TEXT_COLOR}; "
    "font-weight: 600; "
    "padding: 4px 10px; "
    "min-height: 30px;}"
    f"QPushButton:hover {{ background-color: {WIDGET_PANEL_MUTED_COLOR}; border-color: {WIDGET_BORDER_STRONG_COLOR}; }}"
    f"QPushButton:pressed {{ background-color: {WIDGET_PRESSED_COLOR}; border-color: {WIDGET_BORDER_STRONG_COLOR}; }}"
    f"QPushButton:disabled {{ background-color: {WIDGET_PANEL_SUBTLE_COLOR}; "
    f"border-color: {WIDGET_DISABLED_BORDER_COLOR}; color: {WIDGET_DISABLED_TEXT_COLOR}; }}"
)
CHECKBOX_STYLESHEET = (
    "QCheckBox {"
    f"color: {WIDGET_TEXT_COLOR}; "
    "font-weight: 500; "
    "spacing: 8px; "
    "background: transparent;}"
    f"QCheckBox:disabled {{ color: {WIDGET_DISABLED_TEXT_COLOR}; }}"
    "QCheckBox::indicator {"
    "width: 16px; "
    "height: 16px; "
    "border-radius: 4px; "
    f"border: 1px solid {WIDGET_BORDER_COLOR}; "
    f"background-color: {WIDGET_PANEL_COLOR};}}"
    f"QCheckBox::indicator:hover {{ border-color: {WIDGET_ACCENT_BORDER_COLOR}; }}"
    "QCheckBox::indicator:checked {"
    f"border-color: {WIDGET_ACCENT_COLOR}; "
    f"background-color: {WIDGET_ACCENT_COLOR};}}"
    "QCheckBox::indicator:disabled {"
    f"border-color: {WIDGET_DISABLED_BORDER_COLOR}; "
    f"background-color: {WIDGET_PANEL_MUTED_COLOR};}}"
)
COMPLETER_POPUP_STYLESHEET = (
    "QAbstractItemView {"
    f"background-color: {WIDGET_PANEL_COLOR}; "
    f"border: 1px solid {WIDGET_BORDER_COLOR}; "
    f"color: {WIDGET_TEXT_COLOR}; "
    f"selection-background-color: {WIDGET_SELECTION_COLOR}; "
    f"selection-color: {WIDGET_TEXT_COLOR}; "
    "outline: 0px;}"
    "QAbstractItemView::item { padding: 4px 8px; }"
)
StatusCardKind = Literal["info", "warning", "success", "error"]
STATUS_CARD_KINDS: tuple[StatusCardKind, ...] = ("info", "warning", "success", "error")
STATUS_CARD_PALETTE: dict[StatusCardKind, dict[str, str]] = {
    "info": {"text": "#9ecbff", "border": "#5f8fd9", "background": "#28384f"},
    "warning": {
        "text": WIDGET_WARNING_TEXT_COLOR,
        "border": WIDGET_WARNING_BORDER_COLOR,
        "background": WIDGET_WARNING_BACKGROUND_COLOR,
    },
    "success": {"text": "#8ee6a6", "border": WIDGET_SUCCESS_COLOR, "background": "#253b30"},
    "error": {"text": "#ff9aa2", "border": "#e06c75", "background": "#4a2b31"},
}
PRIMARY_BUTTON_STYLESHEET = (
    "QPushButton {"
    f"background-color: {WIDGET_SUCCESS_COLOR}; "
    f"color: {WIDGET_PRIMARY_BUTTON_TEXT_COLOR}; "
    f"border: 1px solid {WIDGET_SUCCESS_COLOR}; "
    "border-radius: 6px; "
    "padding: 7px 14px; "
    "font-weight: 600;}"
    f"QPushButton:hover {{ background-color: {WIDGET_SUCCESS_HOVER_COLOR}; border-color: {WIDGET_SUCCESS_HOVER_COLOR}; }}"
)
WARNING_BUTTON_STYLESHEET = (
    "QPushButton {"
    f"background-color: {WIDGET_WARNING_BACKGROUND_COLOR}; "
    f"color: {WIDGET_WARNING_TEXT_COLOR}; "
    f"border: 1px solid {WIDGET_WARNING_BORDER_COLOR}; "
    "border-radius: 6px; "
    "padding: 7px 14px; "
    "font-weight: 600;}"
    f"QPushButton:hover {{ background-color: {WIDGET_WARNING_HOVER_COLOR}; }}"
)
SECONDARY_BUTTON_STYLESHEET = (
    "QPushButton {"
    f"background-color: {WIDGET_PANEL_COLOR}; "
    f"color: {WIDGET_TEXT_COLOR}; "
    f"border: 1px solid {WIDGET_BORDER_COLOR}; "
    "border-radius: 6px; "
    "padding: 7px 14px;}"
    f"QPushButton:hover {{ background-color: {WIDGET_PANEL_MUTED_COLOR}; }}"
)
_INLINE_BACKTICK_RE = re.compile(r"`([^`]+)`")


def validate_status_card_kind(kind: str) -> StatusCardKind:
    """Return a validated status-card kind, or fail loudly for invalid input."""
    if kind not in STATUS_CARD_KINDS:
        allowed_kinds = ", ".join(repr(allowed_kind) for allowed_kind in STATUS_CARD_KINDS)
        raise ValueError(f"Invalid status card kind {kind!r}. Expected one of: {allowed_kinds}.")

    return cast(StatusCardKind, kind)


def build_input_control_stylesheet(control_selector: str) -> str:
    selectors = [selector.strip() for selector in control_selector.split(",")]
    disabled_selector = ", ".join(f"{selector}:disabled" for selector in selectors)
    focus_selector = ", ".join(f"{selector}:focus" for selector in selectors)

    return (
        f"{control_selector} {{"
        f"background-color: {WIDGET_PANEL_COLOR}; "
        f"border: 1px solid {WIDGET_BORDER_COLOR}; "
        "border-radius: 8px; "
        f"color: {WIDGET_TEXT_COLOR}; "
        "padding: 4px 10px; "
        "min-height: 20px;}"
        f"{disabled_selector} {{"
        f"background-color: {WIDGET_PANEL_MUTED_COLOR}; "
        f"border-color: {WIDGET_DISABLED_BORDER_COLOR}; "
        f"color: {WIDGET_DISABLED_TEXT_COLOR};}}"
        f"{focus_selector} {{"
        f"border-color: {WIDGET_ACCENT_COLOR}; "
        f"background-color: {WIDGET_PANEL_COLOR};}}"
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
        if self.currentIndex() < 0 and self.placeholderText():
            placeholder_color = option.palette.color(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text)
            option.palette.setColor(QPalette.ColorRole.ButtonText, placeholder_color)
            option.palette.setColor(QPalette.ColorRole.Text, placeholder_color)
        painter.drawComplexControl(QStyle.ComplexControl.CC_ComboBox, option)
        painter.drawControl(QStyle.ControlElement.CE_ComboBoxLabel, option)

    def _capped_size_hint(self, hint: QSize) -> QSize:
        cap_width = self.fontMetrics().horizontalAdvance("M" * max(1, self.minimumContentsLength())) + 48
        return QSize(min(hint.width(), cap_width), hint.height())

    def _elided_current_text(self, option: QStyleOptionComboBox | None = None) -> str:
        if option is None:
            option = QStyleOptionComboBox()
            self.initStyleOption(option)

        current_text = option.currentText
        if not current_text and self.currentIndex() < 0:
            current_text = self.placeholderText()

        text_rect = self.style().subControlRect(
            QStyle.ComplexControl.CC_ComboBox,
            option,
            QStyle.SubControl.SC_ComboBoxEditField,
            self,
        )
        available_width = max(0, text_rect.width())
        return self.fontMetrics().elidedText(current_text, Qt.TextElideMode.ElideRight, available_width)

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
    message = format_feedback_text(message)
    escaped_message = (
        escape(message).replace("_", "_&#8203;").replace("/", "/&#8203;").replace("-", "-&#8203;").replace("\n", "<br>")
    )
    return f"<qt><div style='color: {TOOLTIP_TEXT_COLOR}; max-width: 360px;'>{escaped_message}</div></qt>"


def format_feedback_text(message: str) -> str:
    """Return text with napari-friendly inline identifier styling."""
    return _INLINE_BACKTICK_RE.sub(r'"\1"', message)


def format_feedback_identifier(name: str, *, max_length: int = 56) -> tuple[str, bool]:
    """Return a visible identifier and whether it was shortened."""
    if max_length < 4:
        raise ValueError("`max_length` must be at least 4.")
    if len(name) <= max_length:
        return name, False

    head_length = min(32, max_length - 4)
    tail_length = max_length - head_length - 1
    return f"{name[:head_length]}…{name[-tail_length:]}", True


def set_status_card(
    label: QLabel,
    *,
    title: str,
    lines: list[str],
    kind: StatusCardKind,
    tooltip_message: str | None = None,
) -> None:
    """Render a compact titled status card into a QLabel."""
    palette = STATUS_CARD_PALETTE.get(kind, STATUS_CARD_PALETTE["info"])
    formatted_lines = "<br>".join(f"<span>{escape(format_feedback_text(line), quote=False)}</span>" for line in lines)
    label.setText(
        "<div>"
        f"<span style='font-size: 11px; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase;'>"
        f"{escape(title, quote=False)}</span><br>"
        f"{formatted_lines}"
        "</div>"
    )
    label.setStyleSheet(
        "font-weight: 500; "
        f"color: {palette['text']}; "
        f"background-color: {palette['background']}; "
        f"border: 1px solid {palette['border']}; "
        "border-radius: 8px; "
        "padding: 10px 12px;"
    )
    label.setToolTip(format_tooltip(tooltip_message) if tooltip_message else "")
    label.setVisible(bool(lines))
