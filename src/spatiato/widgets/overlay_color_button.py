from __future__ import annotations

from collections.abc import Sequence

from qtpy.QtCore import Qt, Signal
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
_MAX_GRADIENT_PREVIEW_STOPS = 7
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
    """Render an overlay colormap and emit accepted solid-color user intent."""

    color_selected = Signal(str)

    def __init__(self, color: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._color = ""
        self._gradient_name: str | None = None
        self.setText("")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedSize(_OVERLAY_COLOR_BUTTON_WIDTH, _OVERLAY_COLOR_BUTTON_HEIGHT)
        self.clicked.connect(self.choose_color)
        self.set_color(color)

    @property
    def current_color(self) -> str:
        """Return the last valid solid color used by the picker."""
        return self._color

    @property
    def gradient_name(self) -> str | None:
        """Return the displayed gradient name, or ``None`` for a solid color."""
        return self._gradient_name

    def set_color(self, color: str) -> None:
        """Render one solid color without emitting user intent."""
        self._color = _normalize_hex_color(color)
        self._gradient_name = None
        label = _overlay_color_label(self._color)
        self.setAccessibleName(f"Channel color {label} {self._color}")
        self.setToolTip(format_tooltip(f"Click to choose channel color. Current color: {label} ({self._color})."))
        self._set_background_stylesheet(self._color)

    def set_colormap_preview(self, name: str, colors: Sequence[str]) -> None:
        """Render a named gradient without changing the solid picker seed."""
        normalized_colors = tuple(_normalize_hex_color(color) for color in colors)
        if len(normalized_colors) < 2:
            raise ValueError("A colormap preview requires at least two color stops.")

        sampled_colors = _sample_gradient_colors(normalized_colors)
        stop_count = len(sampled_colors)
        stops = ", ".join(f"stop: {index / (stop_count - 1):.4f} {color}" for index, color in enumerate(sampled_colors))
        self._gradient_name = name
        self.setAccessibleName(f"Channel colormap {name}")
        self.setToolTip(format_tooltip(f"Click to choose a solid channel color. Current colormap: {name}."))
        self._set_background_stylesheet(f"qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0, {stops})")

    def _set_background_stylesheet(self, background: str) -> None:
        self.setStyleSheet(
            "QPushButton {"
            f"background: {background}; "
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
            color = selected_color.name(QColor.NameFormat.HexRgb)
            self.set_color(color)
            self.color_selected.emit(self._color)


def _sample_gradient_colors(colors: Sequence[str]) -> tuple[str, ...]:
    if len(colors) <= _MAX_GRADIENT_PREVIEW_STOPS:
        return tuple(colors)

    last_index = len(colors) - 1
    return tuple(
        colors[round(sample_index * last_index / (_MAX_GRADIENT_PREVIEW_STOPS - 1))]
        for sample_index in range(_MAX_GRADIENT_PREVIEW_STOPS)
    )
