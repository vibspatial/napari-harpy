from __future__ import annotations

from qtpy.QtCore import Qt
from qtpy.QtGui import QColor
from qtpy.QtWidgets import QColorDialog, QPushButton, QWidget

from napari_harpy.widgets.shared_styles import (
    WIDGET_ACCENT_BORDER_COLOR,
    WIDGET_BORDER_STRONG_COLOR,
    format_tooltip,
)

_OVERLAY_COLOR_BUTTON_WIDTH = 34
_OVERLAY_COLOR_BUTTON_HEIGHT = 22
_OVERLAY_COLOR_BUTTON_RADIUS = 6
_OVERLAY_COLOR_NAMES_BY_HEX = {
    "#00FFFF": "Cyan",
    "#FF00FF": "Magenta",
    "#FFFF00": "Yellow",
    "#00FF7F": "Green",
    "#FF5050": "Red",
    "#1E90FF": "Blue",
    "#FFA500": "Orange",
    "#9370DB": "Purple",
    "#ADFF2F": "Green-yellow",
    "#7B68EE": "Slate blue",
    "#FF1493": "Deep pink",
    "#20B2AA": "Teal",
    "#FFD700": "Gold",
    "#FF7F50": "Coral",
    "#87CEFA": "Sky blue",
    "#32CD32": "Lime green",
    "#FF69B4": "Hot pink",
    "#DDA0DD": "Plum",
}


def _overlay_color_label(color: str) -> str:
    return _OVERLAY_COLOR_NAMES_BY_HEX.get(color.upper(), color)


def _normalize_hex_color(color: str) -> str:
    normalized_color = QColor(color)
    if not normalized_color.isValid():
        return color.upper()
    return normalized_color.name(QColor.NameFormat.HexRgb).upper()


class OverlayColorButton(QPushButton):
    """Button that shows the current overlay color and opens a color picker."""

    def __init__(self, color: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._color = ""
        self.setText("")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedSize(_OVERLAY_COLOR_BUTTON_WIDTH, _OVERLAY_COLOR_BUTTON_HEIGHT)
        self.clicked.connect(self.choose_color)
        self.set_color(color)

    @property
    def current_color(self) -> str:
        return self._color

    def set_color(self, color: str) -> None:
        self._color = _normalize_hex_color(color)
        label = _overlay_color_label(self._color)
        self.setAccessibleName(f"Channel color {label} {self._color}")
        self.setToolTip(format_tooltip(f"Click to choose channel color. Current color: {label} ({self._color})."))
        self.setStyleSheet(
            "QPushButton {"
            f"background-color: {self._color}; "
            f"border: 1px solid {WIDGET_BORDER_STRONG_COLOR}; "
            f"border-radius: {_OVERLAY_COLOR_BUTTON_RADIUS}px; "
            f"min-height: {_OVERLAY_COLOR_BUTTON_HEIGHT}px; "
            f"max-height: {_OVERLAY_COLOR_BUTTON_HEIGHT}px; "
            f"min-width: {_OVERLAY_COLOR_BUTTON_WIDTH}px; "
            f"max-width: {_OVERLAY_COLOR_BUTTON_WIDTH}px; "
            "padding: 0px;}"
            f"QPushButton:hover {{ border: 2px solid {WIDGET_ACCENT_BORDER_COLOR}; }}"
            f"QPushButton:focus {{ border: 2px solid {WIDGET_ACCENT_BORDER_COLOR}; }}"
        )

    def choose_color(self) -> None:
        selected_color = QColorDialog.getColor(QColor(self._color), self, "Select channel color")
        if selected_color.isValid():
            self.set_color(selected_color.name(QColor.NameFormat.HexRgb))
