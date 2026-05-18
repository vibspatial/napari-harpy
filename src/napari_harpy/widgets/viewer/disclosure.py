from __future__ import annotations

from qtpy.QtCore import QPointF, QSignalBlocker, QSize, Qt, Signal
from qtpy.QtGui import QColor, QIcon, QPainter, QPen, QPixmap
from qtpy.QtWidgets import QFrame, QLabel, QSizePolicy, QToolButton, QVBoxLayout, QWidget

from napari_harpy.widgets.shared_styles import (
    WIDGET_ACCENT_BORDER_COLOR,
    WIDGET_ACCENT_SOFT_COLOR,
    WIDGET_BORDER_COLOR,
    WIDGET_BORDER_STRONG_COLOR,
    WIDGET_PANEL_COLOR,
    WIDGET_PANEL_MUTED_COLOR,
    WIDGET_PANEL_SUBTLE_COLOR,
    WIDGET_TEXT_COLOR,
    format_tooltip,
)

_DISCLOSURE_CHEVRON_SIZE = 14
_SECTION_GROUP_STYLESHEET = (
    "QFrame[harpyViewerDisclosureSection='true'] {"
    f"background-color: {WIDGET_PANEL_MUTED_COLOR}; "
    f"border: 1px solid {WIDGET_BORDER_COLOR}; "
    "border-radius: 12px;}"
)
_DISCLOSURE_BUTTON_STYLESHEET = (
    "QToolButton {"
    f"background-color: {WIDGET_PANEL_COLOR}; "
    f"border: 1px solid {WIDGET_BORDER_COLOR}; "
    "border-radius: 8px; "
    f"color: {WIDGET_TEXT_COLOR}; "
    "font-size: 13px; "
    "font-weight: 600; "
    "padding: 4px 10px; "
    "min-height: 30px; "
    "text-align: left;}"
    f"QToolButton:hover {{ background-color: {WIDGET_PANEL_MUTED_COLOR}; border-color: {WIDGET_BORDER_STRONG_COLOR}; }}"
    f"QToolButton:checked {{ background-color: {WIDGET_ACCENT_SOFT_COLOR}; border-color: {WIDGET_ACCENT_BORDER_COLOR}; }}"
)
_ELEMENT_DISCLOSURE_STYLESHEET = (
    "QFrame[harpyViewerDisclosureRow='true'] {"
    f"background-color: {WIDGET_PANEL_SUBTLE_COLOR}; "
    f"border: 1px solid {WIDGET_BORDER_COLOR}; "
    "border-radius: 10px;}"
)


class _ElidedLabel(QLabel):
    """Single-line label that shows a tooltip only when the text is elided."""

    def __init__(self, text: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._full_text = text
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.setMinimumWidth(0)
        self.setMinimumHeight(36)
        self.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self._update_elided_text()

    def set_full_text(self, text: str) -> None:
        self._full_text = text
        self._update_elided_text()

    def resizeEvent(self, event: object) -> None:
        super().resizeEvent(event)
        self._update_elided_text()

    def _update_elided_text(self) -> None:
        available_width = max(0, self.contentsRect().width())
        elided_text = self.fontMetrics().elidedText(self._full_text, Qt.TextElideMode.ElideRight, available_width)
        super().setText(elided_text)
        self.setToolTip(format_tooltip(self._full_text) if elided_text != self._full_text else "")


def _create_disclosure_chevron_icon(*, expanded: bool, color: str = WIDGET_TEXT_COLOR) -> QIcon:
    pixmap = QPixmap(_DISCLOSURE_CHEVRON_SIZE, _DISCLOSURE_CHEVRON_SIZE)
    pixmap.fill(Qt.GlobalColor.transparent)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    pen = QPen(QColor(color))
    pen.setWidthF(2.0)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    painter.setPen(pen)

    if expanded:
        painter.drawLine(QPointF(3.5, 5.0), QPointF(7.0, 8.5))
        painter.drawLine(QPointF(7.0, 8.5), QPointF(10.5, 5.0))
    else:
        painter.drawLine(QPointF(5.0, 3.5), QPointF(8.5, 7.0))
        painter.drawLine(QPointF(8.5, 7.0), QPointF(5.0, 10.5))

    painter.end()
    return QIcon(pixmap)


class _ElidedToolButton(QToolButton):
    """Tool button that elides visible text and only shows a tooltip when shortened."""

    def __init__(
        self,
        text: str = "",
        parent: QWidget | None = None,
        *,
        max_size_hint_width: int = 320,
    ) -> None:
        super().__init__(parent)
        self._full_text = text
        self._max_size_hint_width = max_size_hint_width
        self.setMinimumWidth(0)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setIconSize(QSize(_DISCLOSURE_CHEVRON_SIZE, _DISCLOSURE_CHEVRON_SIZE))
        self.set_chevron_expanded(False)
        self._update_elided_text()

    def full_text(self) -> str:
        return self._full_text

    def set_full_text(self, text: str) -> None:
        self._full_text = text
        self._update_elided_text()

    def sizeHint(self) -> QSize:
        return self._capped_size(super().sizeHint())

    def minimumSizeHint(self) -> QSize:
        hint = super().minimumSizeHint()
        return QSize(min(hint.width(), 48), hint.height())

    def resizeEvent(self, event: object) -> None:
        super().resizeEvent(event)
        self._update_elided_text()

    def refresh_elision(self) -> None:
        self._update_elided_text()

    def set_chevron_expanded(self, expanded: bool) -> None:
        self.setIcon(_create_disclosure_chevron_icon(expanded=expanded))

    def _capped_size(self, hint: QSize) -> QSize:
        return QSize(min(hint.width(), self._max_size_hint_width), hint.height())

    def _update_elided_text(self) -> None:
        available_width = self.contentsRect().width()
        if available_width <= 0:
            available_width = self._max_size_hint_width
        available_width = min(available_width, self._max_size_hint_width)
        text_width = max(0, available_width - 42)
        elided_text = self.fontMetrics().elidedText(
            self._full_text,
            Qt.TextElideMode.ElideRight,
            text_width,
        )
        if self.text() != elided_text:
            super().setText(elided_text)
        self.setToolTip(format_tooltip(self._full_text) if elided_text != self._full_text else "")


class _CollapsibleSectionWidget(QFrame):
    """Top-level collapsible section for viewer element categories."""

    def __init__(
        self,
        *,
        title: str,
        object_name: str,
        toggle_object_name: str,
        expanded: bool = False,
    ) -> None:
        super().__init__()
        self._title = title
        self.setObjectName(object_name)
        self.setProperty("harpyViewerDisclosureSection", True)
        self.setStyleSheet(_SECTION_GROUP_STYLESHEET)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        self.toggle_button = _ElidedToolButton()
        self.toggle_button.setObjectName(toggle_object_name)
        self.toggle_button.setCheckable(True)
        self.toggle_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.toggle_button.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.toggle_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.toggle_button.setStyleSheet(_DISCLOSURE_BUTTON_STYLESHEET)
        self.toggle_button.toggled.connect(self._on_toggled)

        self.content_widget = QWidget()
        self.content_widget.setObjectName(f"{object_name}_content")
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setSpacing(8)

        layout.addWidget(self.toggle_button)
        layout.addWidget(self.content_widget)
        self.set_expanded(expanded)

    def is_expanded(self) -> bool:
        return self.toggle_button.isChecked()

    def set_count(self, count: int) -> None:
        self.toggle_button.set_full_text(f"{self._title} ({count})")
        self._update_accessible_text()

    def set_expanded(self, expanded: bool) -> None:
        with QSignalBlocker(self.toggle_button):
            self.toggle_button.setChecked(expanded)
        self._sync_expanded_state(expanded)

    def _on_toggled(self, expanded: bool) -> None:
        self._sync_expanded_state(expanded)

    def _sync_expanded_state(self, expanded: bool) -> None:
        self.content_widget.setVisible(expanded)
        self.toggle_button.set_chevron_expanded(expanded)
        self.toggle_button.refresh_elision()
        self._update_accessible_text()

    def _update_accessible_text(self) -> None:
        state = "expanded" if self.is_expanded() else "collapsed"
        self.toggle_button.setAccessibleName(f"{self.toggle_button.full_text()} section, {state}")


class _DisclosureElementWidget(QFrame):
    """Compact element row with an expandable detail widget."""

    expanded_changed = Signal(bool)

    def __init__(
        self,
        *,
        title: str,
        object_name: str,
        toggle_object_name: str,
        detail_widget: QWidget,
        expanded: bool = False,
    ) -> None:
        super().__init__()
        self.title = title
        self.detail_widget = detail_widget
        self.setObjectName(object_name)
        self.setProperty("harpyViewerDisclosureRow", True)
        self.setStyleSheet(_ELEMENT_DISCLOSURE_STYLESHEET)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        self.toggle_button = _ElidedToolButton(title)
        self.toggle_button.setObjectName(toggle_object_name)
        self.toggle_button.setCheckable(True)
        self.toggle_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.toggle_button.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.toggle_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.toggle_button.setStyleSheet(_DISCLOSURE_BUTTON_STYLESHEET)
        self.toggle_button.toggled.connect(self._on_toggled)

        layout.addWidget(self.toggle_button)
        layout.addWidget(self.detail_widget)
        self.set_expanded(expanded)

    def is_expanded(self) -> bool:
        return self.toggle_button.isChecked()

    def set_expanded(self, expanded: bool) -> None:
        with QSignalBlocker(self.toggle_button):
            self.toggle_button.setChecked(expanded)
        self._sync_expanded_state(expanded)

    def _on_toggled(self, expanded: bool) -> None:
        self._sync_expanded_state(expanded)
        self.expanded_changed.emit(expanded)

    def _sync_expanded_state(self, expanded: bool) -> None:
        self.detail_widget.setVisible(expanded)
        self.toggle_button.set_chevron_expanded(expanded)
        self.toggle_button.refresh_elision()
        state = "expanded" if expanded else "collapsed"
        self.toggle_button.setAccessibleName(f"{self.title} element, {state}")
