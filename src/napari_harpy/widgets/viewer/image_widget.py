from __future__ import annotations

from dataclasses import dataclass

from qtpy.QtCore import QSignalBlocker, Qt, Signal
from qtpy.QtGui import QColor
from qtpy.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from napari_harpy.viewer.image_styling import DEFAULT_OVERLAY_COLORS, ImageDisplayMode
from napari_harpy.widgets.shared_styles import (
    ACTION_BUTTON_STYLESHEET,
    CHECKBOX_STYLESHEET,
    WIDGET_ACCENT_BORDER_COLOR,
    WIDGET_BORDER_STRONG_COLOR,
    WIDGET_TEXT_MUTED_COLOR,
    WIDGET_WARNING_TEXT_COLOR,
    format_tooltip,
)
from napari_harpy.widgets.viewer.disclosure import _ElidedLabel
from napari_harpy.widgets.viewer.styles import CARD_TITLE_STYLESHEET, DETAIL_PANEL_STYLESHEET, EMPTY_STATE_STYLESHEET

_CHANNEL_WARNING_STYLESHEET = f"color: {WIDGET_WARNING_TEXT_COLOR}; font-weight: 600;"
_CHANNEL_PANEL_STYLESHEET = "QWidget { background: transparent; }"
_SUBSECTION_LABEL_STYLESHEET = f"color: {WIDGET_TEXT_MUTED_COLOR}; font-size: 11px; font-weight: 600;"
_MAX_VISIBLE_OVERLAY_CHANNELS = 5
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


@dataclass(frozen=True)
class ImageLoadRequest:
    image_name: str
    mode: ImageDisplayMode
    channels: list[int]
    channel_colors: list[str]


def _overlay_color_label(color: str) -> str:
    return _OVERLAY_COLOR_NAMES_BY_HEX.get(color.upper(), color)


def _normalize_hex_color(color: str) -> str:
    normalized_color = QColor(color)
    if not normalized_color.isValid():
        return color.upper()
    return normalized_color.name(QColor.NameFormat.HexRgb).upper()


class _OverlayColorButton(QPushButton):
    """Button that shows the current channel color and opens a color picker on click."""

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


class _ImageCardWidget(QFrame):
    """Card UI for one image element in the selected coordinate system."""

    add_update_requested = Signal(object)

    def __init__(
        self,
        *,
        image_name: str,
        channel_names: list[str],
        channel_error: str | None = None,
    ) -> None:
        super().__init__()
        self.image_name = image_name
        self.channel_names = channel_names
        self.channel_error = channel_error
        self.setObjectName(f"viewer_widget_image_card_{image_name}")
        self.setProperty("harpyViewerDetailPanel", True)
        self.setStyleSheet(DETAIL_PANEL_STYLESHEET)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        self.title_label = _ElidedLabel(image_name, self)
        self.title_label.setObjectName(f"viewer_widget_image_card_title_{image_name}")
        self.title_label.setStyleSheet(CARD_TITLE_STYLESHEET)
        self.title_label.hide()

        mode_layout = QHBoxLayout()
        mode_layout.setContentsMargins(0, 0, 0, 0)
        mode_layout.setSpacing(16)

        self.stack_toggle = QCheckBox("stack")
        self.stack_toggle.setObjectName(f"viewer_widget_stack_toggle_{image_name}")
        self.stack_toggle.setStyleSheet(CHECKBOX_STYLESHEET)
        self.stack_toggle.setChecked(True)

        self.overlay_toggle = QCheckBox("overlay")
        self.overlay_toggle.setObjectName(f"viewer_widget_overlay_toggle_{image_name}")
        self.overlay_toggle.setStyleSheet(CHECKBOX_STYLESHEET)

        mode_layout.addWidget(self.stack_toggle)
        mode_layout.addWidget(self.overlay_toggle)
        mode_layout.addStretch(1)

        self.channel_warning_label = QLabel()
        self.channel_warning_label.setObjectName(f"viewer_widget_channel_warning_{image_name}")
        self.channel_warning_label.setWordWrap(True)
        self.channel_warning_label.setStyleSheet(_CHANNEL_WARNING_STYLESHEET)
        self.channel_warning_label.hide()

        self.channel_panel = QWidget()
        self.channel_panel.setObjectName(f"viewer_widget_channel_panel_{image_name}")
        self.channel_panel.setStyleSheet(_CHANNEL_PANEL_STYLESHEET)
        self.channel_panel.setVisible(False)
        channel_layout = QVBoxLayout(self.channel_panel)
        channel_layout.setContentsMargins(24, 10, 0, 0)
        channel_layout.setSpacing(8)

        self.channel_section_label = QLabel("Channels")
        self.channel_section_label.setObjectName(f"viewer_widget_channel_section_label_{image_name}")
        self.channel_section_label.setStyleSheet(_SUBSECTION_LABEL_STYLESHEET)
        channel_layout.addWidget(self.channel_section_label)

        self.channel_scroll_area = QScrollArea()
        self.channel_scroll_area.setObjectName(f"viewer_widget_channel_scroll_area_{image_name}")
        self.channel_scroll_area.setWidgetResizable(True)
        self.channel_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.channel_scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.channel_scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.channel_scroll_area.setStyleSheet("QScrollArea { border: 0px; background: transparent; }")

        self.channel_list_widget = QWidget()
        self.channel_list_widget.setObjectName(f"viewer_widget_channel_list_{image_name}")
        self.channel_list_widget.setStyleSheet(_CHANNEL_PANEL_STYLESHEET)
        self.channel_list_layout = QVBoxLayout(self.channel_list_widget)
        self.channel_list_layout.setContentsMargins(0, 0, 0, 0)
        self.channel_list_layout.setSpacing(6)
        self.channel_scroll_area.setWidget(self.channel_list_widget)
        channel_layout.addWidget(self.channel_scroll_area)

        self.channel_checkboxes: list[QCheckBox] = []
        self.channel_color_buttons: list[_OverlayColorButton] = []
        channel_rows: list[QWidget] = []

        if channel_error is not None:
            self.overlay_toggle.setEnabled(False)
            self.overlay_toggle.setToolTip(format_tooltip(channel_error))
            self.channel_warning_label.setText(
                "Overlay is unavailable because this image has duplicate channel names. "
                'Use "sdata.set_channel_names(...)" to rename them.'
            )
            self.channel_warning_label.setToolTip(format_tooltip(channel_error))
        elif channel_names:
            for index, channel_name in enumerate(channel_names):
                row = QWidget()
                row_layout = QHBoxLayout(row)
                row_layout.setContentsMargins(0, 0, 0, 0)
                row_layout.setSpacing(8)

                checkbox = QCheckBox(channel_name)
                checkbox.setObjectName(f"viewer_widget_channel_checkbox_{image_name}_{channel_name}")
                checkbox.setStyleSheet(CHECKBOX_STYLESHEET)

                color_button = _OverlayColorButton(DEFAULT_OVERLAY_COLORS[index % len(DEFAULT_OVERLAY_COLORS)])
                color_button.setObjectName(f"viewer_widget_channel_color_button_{image_name}_{channel_name}")

                row_layout.addWidget(checkbox, 1)
                row_layout.addWidget(color_button)

                self.channel_list_layout.addWidget(row)
                self.channel_checkboxes.append(checkbox)
                self.channel_color_buttons.append(color_button)
                channel_rows.append(row)
        else:
            no_channels_label = QLabel("No channel axis available for this image.")
            no_channels_label.setObjectName(f"viewer_widget_no_channels_label_{image_name}")
            no_channels_label.setWordWrap(True)
            no_channels_label.setStyleSheet(EMPTY_STATE_STYLESHEET)
            self.channel_list_layout.addWidget(no_channels_label)
            channel_rows.append(no_channels_label)

        self._set_channel_scroll_height(channel_rows)

        self.add_update_button = QPushButton("Add / Update in viewer")
        self.add_update_button.setObjectName(f"viewer_widget_add_update_image_button_{image_name}")
        self.add_update_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.add_update_button.setMinimumHeight(28)
        self.add_update_button.setStyleSheet(ACTION_BUTTON_STYLESHEET)
        self.add_update_button.clicked.connect(self._emit_add_update_request)
        self.add_update_button.setToolTip("")

        self.stack_toggle.toggled.connect(self._on_stack_toggled)
        self.overlay_toggle.toggled.connect(self._on_overlay_toggled)

        layout.addLayout(mode_layout)
        layout.addWidget(self.channel_warning_label)
        layout.addWidget(self.channel_panel)
        layout.addWidget(self.add_update_button)
        self.channel_warning_label.setVisible(channel_error is not None)

    def _on_stack_toggled(self, checked: bool) -> None:
        if checked:
            with QSignalBlocker(self.overlay_toggle):
                self.overlay_toggle.setChecked(False)
            self.channel_panel.setVisible(False)
            return

        if not self.overlay_toggle.isChecked():
            with QSignalBlocker(self.stack_toggle):
                self.stack_toggle.setChecked(True)

    def _on_overlay_toggled(self, checked: bool) -> None:
        if checked:
            with QSignalBlocker(self.stack_toggle):
                self.stack_toggle.setChecked(False)
            self.channel_panel.setVisible(True)
            return

        self.channel_panel.setVisible(False)
        if not self.stack_toggle.isChecked():
            with QSignalBlocker(self.stack_toggle):
                self.stack_toggle.setChecked(True)

    def display_mode(self) -> str:
        return "overlay" if self.overlay_toggle.isChecked() else "stack"

    def get_selected_overlay_channels(self) -> list[int]:
        return [index for index, checkbox in enumerate(self.channel_checkboxes) if checkbox.isChecked()]

    def get_selected_overlay_channel_names(self) -> list[str]:
        return [checkbox.text() for checkbox in self.channel_checkboxes if checkbox.isChecked()]

    def get_selected_overlay_colors(self) -> list[str]:
        return [
            color_button.current_color
            for checkbox, color_button in zip(self.channel_checkboxes, self.channel_color_buttons, strict=False)
            if checkbox.isChecked()
        ]

    def _emit_add_update_request(self, _checked: bool = False) -> None:
        self.add_update_requested.emit(
            ImageLoadRequest(
                image_name=self.image_name,
                mode=self.display_mode(),
                channels=self.get_selected_overlay_channels(),
                channel_colors=self.get_selected_overlay_colors(),
            )
        )

    def _set_channel_scroll_height(self, channel_rows: list[QWidget]) -> None:
        visible_rows = channel_rows[:_MAX_VISIBLE_OVERLAY_CHANNELS]
        if not visible_rows:
            return

        visible_height = sum(row.sizeHint().height() for row in visible_rows)
        visible_height += self.channel_list_layout.spacing() * max(0, len(visible_rows) - 1)
        margins = self.channel_list_layout.contentsMargins()
        visible_height += margins.top() + margins.bottom()
        visible_height += self.channel_scroll_area.frameWidth() * 2
        self.channel_scroll_area.setMaximumHeight(visible_height)
