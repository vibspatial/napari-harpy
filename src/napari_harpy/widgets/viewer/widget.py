from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype, is_object_dtype, is_string_dtype
from qtpy.QtCore import QPointF, QSignalBlocker, QSize, QStringListModel, Qt, Signal
from qtpy.QtGui import QColor, QIcon, QPainter, QPen, QPixmap
from qtpy.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QCompleter,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLayout,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)
from spatialdata import read_zarr

from napari_harpy._app_state import CoordinateSystemChangedEvent, HarpyAppState, get_or_create_app_state
from napari_harpy.core._color_source import (
    ShapeColorSourceKind,
    ShapeColorSourceSpec,
    TableColorSourceKind,
    TableColorSourceSpec,
)
from napari_harpy.core.spatialdata import (
    get_annotating_table_names,
    get_coordinate_system_names_from_sdata,
    get_image_channel_names_from_sdata,
    get_shape_column_color_source_options,
    get_spatialdata_image_options_for_coordinate_system_from_sdata,
    get_spatialdata_labels_options_for_coordinate_system_from_sdata,
    get_spatialdata_points_options_for_coordinate_system_from_sdata,
    get_spatialdata_shapes_options_for_coordinate_system_from_sdata,
    get_table_color_source_options,
)
from napari_harpy.viewer.adapter import DEFAULT_OVERLAY_COLORS, ShapesLayerBinding, ViewerAdapter
from napari_harpy.viewer.points_styling import PointsLayerResult
from napari_harpy.widgets.shared_styles import (
    ACTION_BUTTON_STYLESHEET as _ACTION_BUTTON_STYLESHEET,
)
from napari_harpy.widgets.shared_styles import (
    CHECKBOX_STYLESHEET as _CHECKBOX_STYLESHEET,
)
from napari_harpy.widgets.shared_styles import (
    WIDGET_ACCENT_BORDER_COLOR,
    WIDGET_ACCENT_SOFT_COLOR,
    WIDGET_BORDER_COLOR,
    WIDGET_BORDER_STRONG_COLOR,
    WIDGET_PANEL_COLOR,
    WIDGET_PANEL_MUTED_COLOR,
    WIDGET_PANEL_SUBTLE_COLOR,
    WIDGET_TEXT_COLOR,
    WIDGET_TEXT_SECONDARY_COLOR,
    CompactComboBox,
    StatusCardKind,
    apply_scroll_content_surface,
    apply_widget_surface,
    build_input_control_stylesheet,
    create_form_label,
    format_feedback_identifier,
    format_tooltip,
    set_status_card,
)
from napari_harpy.widgets.shared_styles import (
    WIDGET_MIN_WIDTH as _WIDGET_MIN_WIDTH,
)
from napari_harpy.widgets.viewer.points_controller import PointsController, PointsLoadResult, PointsValueSource
from napari_harpy.widgets.viewer.points_widget import PointsValueWidget

if TYPE_CHECKING:
    import napari
    from spatialdata import SpatialData


_INPUT_CONTROL_STYLESHEET = build_input_control_stylesheet("QComboBox")
_DETAIL_PANEL_STYLESHEET = (
    "QFrame[harpyViewerDetailPanel='true'] {"
    f"background-color: {WIDGET_PANEL_COLOR}; "
    f"border: 1px solid {WIDGET_BORDER_COLOR}; "
    "border-radius: 8px;}"
)
_CARD_TITLE_STYLESHEET = (
    "QLabel {"
    f"background-color: {WIDGET_ACCENT_SOFT_COLOR}; "
    f"border: 1px solid {WIDGET_ACCENT_BORDER_COLOR}; "
    "border-radius: 8px; "
    f"color: {WIDGET_TEXT_SECONDARY_COLOR}; "
    "font-weight: 700; "
    "padding: 6px 10px;}"
)
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
_SUMMARY_LABEL_STYLESHEET = f"color: {WIDGET_TEXT_SECONDARY_COLOR}; font-weight: 500;"
_EMPTY_STATE_STYLESHEET = "color: #64748b; font-weight: 500;"
_CHANNEL_WARNING_STYLESHEET = "color: #b45309; font-weight: 600;"
_CHANNEL_PANEL_STYLESHEET = "QWidget { background: transparent; }"
_SUBSECTION_LABEL_STYLESHEET = "color: #64748b; font-size: 11px; font-weight: 600;"
_MAX_VISIBLE_OVERLAY_CHANNELS = 5
_DISCLOSURE_CHEVRON_SIZE = 14
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
    mode: str
    channels: list[int]
    channel_colors: list[str]


@dataclass(frozen=True)
class LabelsLoadRequest:
    labels_name: str
    table_name: str | None
    selected_source_kind: TableColorSourceKind | None
    selected_color_source: TableColorSourceSpec | None


@dataclass(frozen=True)
class ShapesLoadRequest:
    shapes_name: str
    selected_source_kind: ShapeColorSourceKind | None
    selected_color_source: ShapeColorSourceSpec | None
    fill_shapes: bool


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


class _LabelsCardWidget(QFrame):
    """Card UI for one labels element in the selected coordinate system."""

    add_update_requested = Signal(object)

    def __init__(
        self,
        *,
        labels_name: str,
        table_names: list[str],
        table_color_sources_by_table: dict[str, list[TableColorSourceSpec]],
    ) -> None:
        super().__init__()
        self.labels_name = labels_name
        self._table_color_sources_by_table = table_color_sources_by_table
        self._filtered_color_sources: list[TableColorSourceSpec] = []
        self.setObjectName(f"viewer_widget_labels_card_{labels_name}")
        self.setProperty("harpyViewerDetailPanel", True)
        self.setStyleSheet(_DETAIL_PANEL_STYLESHEET)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        self.title_label = _ElidedLabel(labels_name, self)
        self.title_label.setObjectName(f"viewer_widget_labels_card_title_{labels_name}")
        self.title_label.setStyleSheet(_CARD_TITLE_STYLESHEET)
        self.title_label.hide()

        form_layout = QFormLayout()
        form_layout.setContentsMargins(0, 0, 0, 0)
        form_layout.setHorizontalSpacing(8)
        form_layout.setVerticalSpacing(6)

        linked_table_label = _create_form_label("Linked table")
        self.linked_table_combo = CompactComboBox()
        self.linked_table_combo.setObjectName(f"viewer_widget_linked_table_combo_{labels_name}")
        self.linked_table_combo.setStyleSheet(_INPUT_CONTROL_STYLESHEET)
        if table_names:
            self.linked_table_combo.addItems(table_names)
        else:
            self.linked_table_combo.addItem("No linked tables")
            self.linked_table_combo.setEnabled(False)

        color_source_kind_label = _create_form_label("Color source")
        self.color_source_kind_combo = CompactComboBox()
        self.color_source_kind_combo.setObjectName(f"viewer_widget_color_source_kind_combo_{labels_name}")
        self.color_source_kind_combo.setStyleSheet(_INPUT_CONTROL_STYLESHEET)
        self.color_source_kind_combo.addItem("None", None)
        self.color_source_kind_combo.addItem("Observations", "obs_column")
        self.color_source_kind_combo.addItem("Vars", "x_var")

        self.color_source_value_label = _create_form_label("Value source")
        self.color_source_value_input = QLineEdit()
        self.color_source_value_input.setObjectName(f"viewer_widget_color_source_value_input_{labels_name}")
        self.color_source_value_input.setStyleSheet(build_input_control_stylesheet("QLineEdit"))
        self.color_source_value_input.setMinimumWidth(0)
        self.color_source_value_input.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.color_source_value_input.setEnabled(False)

        self._color_source_completer_model = QStringListModel(self.color_source_value_input)
        self._color_source_completer = QCompleter(self._color_source_completer_model, self.color_source_value_input)
        self._color_source_completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self._color_source_completer.setFilterMode(Qt.MatchFlag.MatchContains)
        self.color_source_value_input.setCompleter(self._color_source_completer)

        self.action_status_label = QLabel()
        self.action_status_label.setObjectName(f"viewer_widget_action_status_{labels_name}")
        self.action_status_label.setWordWrap(True)
        self.action_status_label.setStyleSheet(_SUMMARY_LABEL_STYLESHEET)

        self.add_update_button = QPushButton("Add / Update in viewer")
        self.add_update_button.setObjectName(f"viewer_widget_add_update_labels_button_{labels_name}")
        self.add_update_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.add_update_button.setMinimumHeight(28)
        self.add_update_button.setStyleSheet(_ACTION_BUTTON_STYLESHEET)
        self.add_update_button.clicked.connect(self._emit_add_update_request)

        form_layout.addRow(linked_table_label, self.linked_table_combo)
        form_layout.addRow(color_source_kind_label, self.color_source_kind_combo)
        form_layout.addRow(self.color_source_value_label, self.color_source_value_input)

        layout.addLayout(form_layout)
        layout.addWidget(self.action_status_label)
        layout.addWidget(self.add_update_button)

        self.linked_table_combo.currentIndexChanged.connect(self._refresh_color_source_controls)
        self.color_source_kind_combo.currentIndexChanged.connect(self._refresh_color_source_controls)
        self.color_source_value_input.textChanged.connect(self._update_action_status)
        self.color_source_value_input.editingFinished.connect(self._sync_current_source_selection)

        self._refresh_color_source_controls()

    @property
    def selected_table_name(self) -> str | None:
        if not self.linked_table_combo.isEnabled():
            return None

        table_name = self.linked_table_combo.currentText()
        return table_name if table_name in self._table_color_sources_by_table else None

    @property
    def selected_source_kind(self) -> TableColorSourceKind | None:
        value = self.color_source_kind_combo.currentData()
        return value if value in {"obs_column", "x_var"} else None

    @property
    def selected_color_source(self) -> TableColorSourceSpec | None:
        current_text = self.color_source_value_input.text().strip()
        for source in self._filtered_color_sources:
            if source.display_name == current_text:
                return source
        return None

    def _refresh_color_source_controls(self, _index: int | None = None) -> None:
        selected_source_identity = (
            self.selected_color_source.identity if self.selected_color_source is not None else None
        )
        source_kind = self.selected_source_kind
        table_name = self.selected_table_name

        if source_kind == "obs_column":
            self.color_source_value_label.setText("Observation")
        elif source_kind == "x_var":
            self.color_source_value_label.setText("Var")
        else:
            self.color_source_value_label.setText("Value source")

        available_sources = (
            list(self._table_color_sources_by_table.get(table_name, ())) if table_name is not None else []
        )
        self._filtered_color_sources = [
            source for source in available_sources if source_kind is None or source.source_kind == source_kind
        ]

        with QSignalBlocker(self.color_source_value_input):
            if source_kind is None:
                self.color_source_value_input.setEnabled(False)
                self.color_source_value_input.clear()
                self.color_source_value_input.setPlaceholderText("Select a color source kind first")
            else:
                self.color_source_value_input.setEnabled(bool(self._filtered_color_sources))
                if selected_source_identity is not None:
                    matching_source = next(
                        (
                            source
                            for source in self._filtered_color_sources
                            if source.identity == selected_source_identity
                        ),
                        None,
                    )
                    if matching_source is not None:
                        self.color_source_value_input.setText(matching_source.display_name)
                    elif self._filtered_color_sources:
                        self.color_source_value_input.setText(self._filtered_color_sources[0].display_name)
                    else:
                        self.color_source_value_input.clear()
                elif self._filtered_color_sources:
                    self.color_source_value_input.setText(self._filtered_color_sources[0].display_name)
                else:
                    self.color_source_value_input.clear()

                if source_kind == "obs_column":
                    self.color_source_value_input.setPlaceholderText("Search observations")
                else:
                    self.color_source_value_input.setPlaceholderText("Search vars")

            self._color_source_completer_model.setStringList(
                [source.display_name for source in self._filtered_color_sources]
            )

        self._update_action_status()

    def _sync_current_source_selection(self) -> None:
        if not self.color_source_value_input.isEnabled():
            return

        current_text = self.color_source_value_input.text().strip()
        for source in self._filtered_color_sources:
            if source.display_name == current_text:
                self.color_source_value_input.setText(source.display_name)
                break
        self._update_action_status()

    def _update_action_status(self) -> None:
        source_kind = self.selected_source_kind
        table_name = self.selected_table_name
        selected_source = self.selected_color_source

        if source_kind is None:
            self.action_status_label.setText("Action: add/update primary labels layer")
            return

        if table_name is None:
            self.action_status_label.setText("Action: colored overlays require a linked table")
            return

        if source_kind == "obs_column":
            if selected_source is None:
                if self._filtered_color_sources:
                    self.action_status_label.setText("Action: select an observation column for a colored overlay")
                else:
                    self.action_status_label.setText("Action: no colorable observation columns available")
                return
            self.action_status_label.setText(
                f'Action: add/update colored overlay for obs["{selected_source.value_key}"]'
            )
            return

        if selected_source is None:
            if self._filtered_color_sources:
                self.action_status_label.setText("Action: select a var for a colored overlay")
            else:
                self.action_status_label.setText("Action: no vars available for a colored overlay")
            return

        self.action_status_label.setText(f'Action: add/update colored overlay for X[:, "{selected_source.value_key}"]')

    def _emit_add_update_request(self, _checked: bool = False) -> None:
        self.add_update_requested.emit(
            LabelsLoadRequest(
                labels_name=self.labels_name,
                table_name=self.selected_table_name,
                selected_source_kind=self.selected_source_kind,
                selected_color_source=self.selected_color_source,
            )
        )


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
        self.setStyleSheet(_DETAIL_PANEL_STYLESHEET)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        self.title_label = _ElidedLabel(image_name, self)
        self.title_label.setObjectName(f"viewer_widget_image_card_title_{image_name}")
        self.title_label.setStyleSheet(_CARD_TITLE_STYLESHEET)
        self.title_label.hide()

        mode_layout = QHBoxLayout()
        mode_layout.setContentsMargins(0, 0, 0, 0)
        mode_layout.setSpacing(16)

        self.stack_toggle = QCheckBox("stack")
        self.stack_toggle.setObjectName(f"viewer_widget_stack_toggle_{image_name}")
        self.stack_toggle.setStyleSheet(_CHECKBOX_STYLESHEET)
        self.stack_toggle.setChecked(True)

        self.overlay_toggle = QCheckBox("overlay")
        self.overlay_toggle.setObjectName(f"viewer_widget_overlay_toggle_{image_name}")
        self.overlay_toggle.setStyleSheet(_CHECKBOX_STYLESHEET)

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
                "Use `sdata.set_channel_names(...)` to rename them."
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
                checkbox.setStyleSheet(_CHECKBOX_STYLESHEET)

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
            no_channels_label.setStyleSheet(_EMPTY_STATE_STYLESHEET)
            self.channel_list_layout.addWidget(no_channels_label)
            channel_rows.append(no_channels_label)

        self._set_channel_scroll_height(channel_rows)

        self.add_update_button = QPushButton("Add / Update in viewer")
        self.add_update_button.setObjectName(f"viewer_widget_add_update_image_button_{image_name}")
        self.add_update_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.add_update_button.setMinimumHeight(28)
        self.add_update_button.setStyleSheet(_ACTION_BUTTON_STYLESHEET)
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


class _ShapesCardWidget(QFrame):
    """Card UI shell for one shapes element in the selected coordinate system."""

    add_update_requested = Signal(object)

    def __init__(
        self,
        *,
        shapes_name: str,
        shape_color_sources: list[ShapeColorSourceSpec],
    ) -> None:
        super().__init__()
        self.shapes_name = shapes_name
        self._shape_color_sources = shape_color_sources
        self._filtered_color_sources: list[ShapeColorSourceSpec] = []
        self.setObjectName(f"viewer_widget_shapes_card_{shapes_name}")
        self.setProperty("harpyViewerDetailPanel", True)
        self.setStyleSheet(_DETAIL_PANEL_STYLESHEET)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        self.title_label = _ElidedLabel(shapes_name, self)
        self.title_label.setObjectName(f"viewer_widget_shapes_card_title_{shapes_name}")
        self.title_label.setStyleSheet(_CARD_TITLE_STYLESHEET)
        self.title_label.hide()

        form_layout = QFormLayout()
        form_layout.setContentsMargins(0, 0, 0, 0)
        form_layout.setHorizontalSpacing(8)
        form_layout.setVerticalSpacing(6)

        color_source_kind_label = _create_form_label("Color source")
        self.color_source_kind_combo = CompactComboBox()
        self.color_source_kind_combo.setObjectName(f"viewer_widget_shapes_color_source_kind_combo_{shapes_name}")
        self.color_source_kind_combo.setStyleSheet(_INPUT_CONTROL_STYLESHEET)
        self.color_source_kind_combo.addItem("None", None)
        self.color_source_kind_combo.addItem("Shapes column", "shape_column")

        self.color_source_value_label = _create_form_label("Shapes column")
        self.color_source_value_input = QLineEdit()
        self.color_source_value_input.setObjectName(f"viewer_widget_shapes_color_source_value_input_{shapes_name}")
        self.color_source_value_input.setStyleSheet(build_input_control_stylesheet("QLineEdit"))
        self.color_source_value_input.setMinimumWidth(0)
        self.color_source_value_input.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.color_source_value_input.setEnabled(False)

        self.fill_toggle = QCheckBox("Fill")
        self.fill_toggle.setObjectName(f"viewer_widget_shapes_fill_toggle_{shapes_name}")
        self.fill_toggle.setStyleSheet(_CHECKBOX_STYLESHEET)
        self.fill_toggle.setChecked(False)

        self._color_source_completer_model = QStringListModel(self.color_source_value_input)
        self._color_source_completer = QCompleter(self._color_source_completer_model, self.color_source_value_input)
        self._color_source_completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self._color_source_completer.setFilterMode(Qt.MatchFlag.MatchContains)
        self.color_source_value_input.setCompleter(self._color_source_completer)

        self.action_status_label = QLabel()
        self.action_status_label.setObjectName(f"viewer_widget_shapes_action_status_{shapes_name}")
        self.action_status_label.setWordWrap(True)
        self.action_status_label.setStyleSheet(_SUMMARY_LABEL_STYLESHEET)

        self.add_update_button = QPushButton("Add / Update in viewer")
        self.add_update_button.setObjectName(f"viewer_widget_add_update_shapes_button_{shapes_name}")
        self.add_update_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.add_update_button.setMinimumHeight(28)
        self.add_update_button.setStyleSheet(_ACTION_BUTTON_STYLESHEET)
        self.add_update_button.clicked.connect(self._emit_add_update_request)
        self.add_update_button.setToolTip("")

        form_layout.addRow(color_source_kind_label, self.color_source_kind_combo)
        form_layout.addRow(self.color_source_value_label, self.color_source_value_input)
        form_layout.addRow(_create_form_label("Display"), self.fill_toggle)

        layout.addLayout(form_layout)
        layout.addWidget(self.action_status_label)
        layout.addWidget(self.add_update_button)

        self.color_source_kind_combo.currentIndexChanged.connect(self._refresh_color_source_controls)
        self.color_source_value_input.textChanged.connect(self._update_action_status)
        self.color_source_value_input.editingFinished.connect(self._sync_current_source_selection)

        self._refresh_color_source_controls()

    @property
    def selected_source_kind(self) -> ShapeColorSourceKind | None:
        value = self.color_source_kind_combo.currentData()
        return value if value == "shape_column" else None

    @property
    def selected_color_source(self) -> ShapeColorSourceSpec | None:
        current_text = self.color_source_value_input.text().strip()
        for source in self._filtered_color_sources:
            if source.display_name == current_text:
                return source
        return None

    @property
    def fill_shapes(self) -> bool:
        return self.fill_toggle.isChecked()

    def _refresh_color_source_controls(self, _index: int | None = None) -> None:
        selected_source_identity = (
            self.selected_color_source.identity if self.selected_color_source is not None else None
        )
        source_kind = self.selected_source_kind
        self._filtered_color_sources = [
            source for source in self._shape_color_sources if source_kind is None or source.source_kind == source_kind
        ]

        with QSignalBlocker(self.color_source_value_input):
            if source_kind is None:
                self.color_source_value_input.setEnabled(False)
                self.color_source_value_input.clear()
                self.color_source_value_input.setPlaceholderText("Select a color source kind first")
            else:
                self.color_source_value_input.setEnabled(bool(self._filtered_color_sources))
                if selected_source_identity is not None:
                    matching_source = next(
                        (
                            source
                            for source in self._filtered_color_sources
                            if source.identity == selected_source_identity
                        ),
                        None,
                    )
                    if matching_source is not None:
                        self.color_source_value_input.setText(matching_source.display_name)
                    elif self._filtered_color_sources:
                        self.color_source_value_input.setText(self._filtered_color_sources[0].display_name)
                    else:
                        self.color_source_value_input.clear()
                elif self._filtered_color_sources:
                    self.color_source_value_input.setText(self._filtered_color_sources[0].display_name)
                else:
                    self.color_source_value_input.clear()

                self.color_source_value_input.setPlaceholderText("Search shapes columns")

            self._color_source_completer_model.setStringList(
                [source.display_name for source in self._filtered_color_sources]
            )

        self._update_action_status()

    def _sync_current_source_selection(self) -> None:
        if not self.color_source_value_input.isEnabled():
            return

        current_text = self.color_source_value_input.text().strip()
        for source in self._filtered_color_sources:
            if source.display_name == current_text:
                self.color_source_value_input.setText(source.display_name)
                break
        self._update_action_status()

    def _update_action_status(self) -> None:
        source_kind = self.selected_source_kind
        selected_source = self.selected_color_source
        self._update_fill_toggle_enabled(selected_source is not None)

        if source_kind is None:
            self.action_status_label.setText("Action: add/update primary shapes layer")
            return

        if selected_source is None:
            if self._filtered_color_sources:
                self.action_status_label.setText("Action: select a shapes column for a styled shapes layer")
            else:
                self.action_status_label.setText("Action: no colorable shapes columns available")
            return

        self.action_status_label.setText(
            f'Action: add/update styled shapes layer for column "{selected_source.value_key}"'
        )

    def _update_fill_toggle_enabled(self, enabled: bool) -> None:
        self.fill_toggle.setEnabled(enabled)
        if not enabled:
            self.fill_toggle.setChecked(False)

    def _emit_add_update_request(self, _checked: bool = False) -> None:
        self.add_update_requested.emit(
            ShapesLoadRequest(
                shapes_name=self.shapes_name,
                selected_source_kind=self.selected_source_kind,
                selected_color_source=self.selected_color_source,
                fill_shapes=self.fill_shapes,
            )
        )


class ViewerWidget(QWidget):
    """Shared viewer widget backed by `HarpyAppState` and `ViewerAdapter`."""

    def __init__(self, napari_viewer: napari.Viewer | None = None) -> None:
        super().__init__()
        self.setObjectName("viewer_widget")
        apply_widget_surface(self)
        self.setMinimumWidth(_WIDGET_MIN_WIDTH)
        self._viewer = napari_viewer
        self._app_state = get_or_create_app_state(napari_viewer)
        self._points_controller = PointsController(
            on_state_changed=self._on_points_controller_state_changed,
            on_value_source_loaded=self._on_points_value_source_loaded,
            on_points_loaded=self._on_points_loaded,
        )
        self._last_points_load_result: PointsLoadResult | None = None
        self._last_points_layer_result: PointsLayerResult | None = None
        self._labels_cards: list[_LabelsCardWidget] = []
        self._image_cards: list[_ImageCardWidget] = []
        self._shape_cards: list[_ShapesCardWidget] = []
        self._labels_rows: list[_DisclosureElementWidget] = []
        self._image_rows: list[_DisclosureElementWidget] = []
        self._shape_rows: list[_DisclosureElementWidget] = []
        self._expanded_labels_names: set[str] = set()
        self._expanded_image_names: set[str] = set()
        self._expanded_shapes_names: set[str] = set()
        self._logo_path = Path(__file__).resolve().parents[4] / "docs" / "_static" / "logo.png"

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        self.scroll_area = QScrollArea()
        self.scroll_area.setObjectName("viewer_widget_scroll_area")
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll_area.setStyleSheet("QScrollArea { border: 0px; background: transparent; }")

        self.scroll_content = QWidget()
        self.scroll_content.setObjectName("viewer_widget_scroll_content")
        apply_scroll_content_surface(self.scroll_content)
        self.content_layout = QVBoxLayout(self.scroll_content)
        self.content_layout.setContentsMargins(12, 12, 12, 12)
        self.content_layout.setSpacing(10)

        title = QLabel("Viewer")
        title.setObjectName("viewer_widget_title")
        title.setStyleSheet("color: #111827; font-size: 18px; font-weight: 700;")
        title.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        header_logo = self._create_header_logo()

        self.open_sdata_button = QPushButton("Load SpatialData")
        self.open_sdata_button.setObjectName("viewer_widget_open_sdata_button")
        self.open_sdata_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.open_sdata_button.setMinimumHeight(28)
        self.open_sdata_button.setStyleSheet(_ACTION_BUTTON_STYLESHEET)
        self.open_sdata_button.clicked.connect(self._open_spatialdata)

        self.empty_state_label = QLabel(
            "No SpatialData loaded. Use `Interactive(sdata)` for now; an in-widget open action will follow later."
        )
        self.empty_state_label.setObjectName("viewer_widget_empty_state")
        self.empty_state_label.setWordWrap(True)
        self.empty_state_label.setStyleSheet(_EMPTY_STATE_STYLESHEET)

        self.summary_label = QLabel("No SpatialData loaded.")
        self.summary_label.setObjectName("viewer_widget_summary")
        self.summary_label.setWordWrap(True)
        self.summary_label.setStyleSheet(_SUMMARY_LABEL_STYLESHEET)

        selector_layout = QFormLayout()
        selector_layout.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        selector_layout.setHorizontalSpacing(12)
        selector_layout.setVerticalSpacing(10)

        self.coordinate_system_combo = QComboBox()
        self.coordinate_system_combo.setObjectName("viewer_widget_coordinate_system_combo")
        self.coordinate_system_combo.currentIndexChanged.connect(self._on_coordinate_system_changed)
        self.coordinate_system_combo.setStyleSheet(_INPUT_CONTROL_STYLESHEET)
        selector_layout.addRow(_create_form_label("Coordinate system"), self.coordinate_system_combo)

        self.action_feedback_label = QLabel("")
        self.action_feedback_label.setObjectName("viewer_widget_action_feedback")
        self.action_feedback_label.setWordWrap(True)
        self.action_feedback_label.hide()

        self.images_empty_label = QLabel("No images available in the selected coordinate system.")
        self.images_empty_label.setObjectName("viewer_widget_images_empty_state")
        self.images_empty_label.setWordWrap(True)
        self.images_empty_label.setStyleSheet(_EMPTY_STATE_STYLESHEET)

        self.images_section = QWidget()
        self.images_section.setObjectName("viewer_widget_images_section")
        self.images_section_layout = QVBoxLayout(self.images_section)
        self.images_section_layout.setContentsMargins(0, 0, 0, 0)
        self.images_section_layout.setSpacing(8)
        self.images_group = _CollapsibleSectionWidget(
            title="Images",
            object_name="viewer_widget_images_group",
            toggle_object_name="viewer_widget_images_section_toggle",
            expanded=False,
        )
        self.images_group.content_layout.addWidget(self.images_empty_label)
        self.images_group.content_layout.addWidget(self.images_section)
        self.images_section_toggle = self.images_group.toggle_button
        self.images_section_title = self.images_section_toggle

        self.labels_empty_label = QLabel("No labels available in the selected coordinate system.")
        self.labels_empty_label.setObjectName("viewer_widget_labels_empty_state")
        self.labels_empty_label.setWordWrap(True)
        self.labels_empty_label.setStyleSheet(_EMPTY_STATE_STYLESHEET)

        self.labels_section = QWidget()
        self.labels_section.setObjectName("viewer_widget_labels_section")
        self.labels_section_layout = QVBoxLayout(self.labels_section)
        self.labels_section_layout.setContentsMargins(0, 0, 0, 0)
        self.labels_section_layout.setSpacing(8)
        self.labels_group = _CollapsibleSectionWidget(
            title="Labels",
            object_name="viewer_widget_labels_group",
            toggle_object_name="viewer_widget_labels_section_toggle",
            expanded=False,
        )
        self.labels_group.content_layout.addWidget(self.labels_empty_label)
        self.labels_group.content_layout.addWidget(self.labels_section)
        self.labels_section_toggle = self.labels_group.toggle_button
        self.labels_section_title = self.labels_section_toggle

        self.shapes_empty_label = QLabel("No shapes available in the selected coordinate system.")
        self.shapes_empty_label.setObjectName("viewer_widget_shapes_empty_state")
        self.shapes_empty_label.setWordWrap(True)
        self.shapes_empty_label.setStyleSheet(_EMPTY_STATE_STYLESHEET)

        self.shapes_section = QWidget()
        self.shapes_section.setObjectName("viewer_widget_shapes_section")
        self.shapes_section_layout = QVBoxLayout(self.shapes_section)
        self.shapes_section_layout.setContentsMargins(0, 0, 0, 0)
        self.shapes_section_layout.setSpacing(8)
        self.shapes_group = _CollapsibleSectionWidget(
            title="Shapes",
            object_name="viewer_widget_shapes_group",
            toggle_object_name="viewer_widget_shapes_section_toggle",
            expanded=False,
        )
        self.shapes_group.content_layout.addWidget(self.shapes_empty_label)
        self.shapes_group.content_layout.addWidget(self.shapes_section)
        self.shapes_section_toggle = self.shapes_group.toggle_button
        self.shapes_section_title = self.shapes_section_toggle

        self.points_empty_label = QLabel("No points available in the selected coordinate system.")
        self.points_empty_label.setObjectName("viewer_widget_points_empty_state")
        self.points_empty_label.setWordWrap(True)
        self.points_empty_label.setStyleSheet(_EMPTY_STATE_STYLESHEET)

        self.points_widget = PointsValueWidget()
        self.points_widget.source_changed.connect(self._on_points_source_changed)
        self.points_widget.add_update_requested.connect(self._add_or_update_points_selection)
        self.points_group = _CollapsibleSectionWidget(
            title="Points",
            object_name="viewer_widget_points_group",
            toggle_object_name="viewer_widget_points_section_toggle",
            expanded=False,
        )
        self.points_group.content_layout.addWidget(self.points_empty_label)
        self.points_group.content_layout.addWidget(self.points_widget)
        self.points_section_toggle = self.points_group.toggle_button
        self.points_section_title = self.points_section_toggle

        self.content_layout.addWidget(header_logo)
        self.content_layout.addWidget(title)
        self.content_layout.addWidget(self.open_sdata_button)
        self.content_layout.addWidget(self.empty_state_label)
        self.content_layout.addWidget(self.summary_label)
        self.content_layout.addLayout(selector_layout)
        self.content_layout.addWidget(self.action_feedback_label)
        self.content_layout.addWidget(self.images_group)
        self.content_layout.addWidget(self.labels_group)
        self.content_layout.addWidget(self.shapes_group)
        self.content_layout.addWidget(self.points_group)
        self.content_layout.addStretch(1)

        self.scroll_area.setWidget(self.scroll_content)
        root_layout.addWidget(self.scroll_area)

        self._app_state.sdata_changed.connect(self._on_sdata_changed)
        self._app_state.coordinate_system_changed.connect(self._on_app_state_coordinate_system_changed)
        self.refresh_from_sdata(self._app_state.sdata)

    @property
    def app_state(self) -> HarpyAppState:
        """Return the shared Harpy app state for this widget."""
        return self._app_state

    @property
    def labels_cards(self) -> list[_LabelsCardWidget]:
        """Return the currently visible labels cards."""
        return list(self._labels_cards)

    @property
    def image_cards(self) -> list[_ImageCardWidget]:
        """Return the currently visible image cards."""
        return list(self._image_cards)

    @property
    def shape_cards(self) -> list[_ShapesCardWidget]:
        """Return the currently visible shapes cards."""
        return list(self._shape_cards)

    @property
    def image_rows(self) -> list[_DisclosureElementWidget]:
        """Return the currently visible compact image rows."""
        return list(self._image_rows)

    @property
    def labels_rows(self) -> list[_DisclosureElementWidget]:
        """Return the currently visible compact labels rows."""
        return list(self._labels_rows)

    @property
    def shape_rows(self) -> list[_DisclosureElementWidget]:
        """Return the currently visible compact shapes rows."""
        return list(self._shape_rows)

    def _on_sdata_changed(self, sdata: SpatialData | None) -> None:
        """Refresh the widget when the shared loaded `SpatialData` changes."""
        self.refresh_from_sdata(sdata)

    def _on_coordinate_system_changed(self, index: int) -> None:
        """Publish explicit user coordinate-system changes to shared app state."""
        coordinate_system = self.coordinate_system_combo.itemData(index)
        self._app_state.set_coordinate_system(
            coordinate_system if isinstance(coordinate_system, str) else None,
            source="viewer_widget",
        )

    def _on_app_state_coordinate_system_changed(self, event: CoordinateSystemChangedEvent) -> None:
        """Refresh the combo and cards when the shared coordinate system changes."""
        del event
        self._clear_action_feedback()
        self._sync_coordinate_system_combo_selection(self._app_state.coordinate_system)
        self._refresh_coordinate_system_content()

    def _open_spatialdata(self, _checked: bool = False) -> None:
        selected_path = QFileDialog.getExistingDirectory(
            self,
            "Load SpatialData",
            "",
            QFileDialog.Option.ShowDirsOnly | QFileDialog.Option.DontResolveSymlinks,
        )
        if not selected_path:
            return

        try:
            sdata = read_zarr(selected_path)
        except (OSError, ValueError) as error:
            self._set_action_feedback(
                title="SpatialData Load Error",
                lines=[f"Could not load SpatialData store: {error}"],
                kind="error",
            )
            return

        self._app_state.set_sdata(sdata)
        self._clear_action_feedback()

    def refresh_from_sdata(self, sdata: SpatialData | None) -> None:
        """Refresh the viewer widget from the currently loaded `SpatialData`."""
        with QSignalBlocker(self.coordinate_system_combo):
            self.coordinate_system_combo.clear()

            if sdata is None:
                self.empty_state_label.show()
                self.summary_label.setText("No SpatialData loaded.")
                self.coordinate_system_combo.setEnabled(False)
                self._clear_cards()
                self._update_section_empty_states([], [], [], [])
                return

            coordinate_systems = get_coordinate_system_names_from_sdata(sdata)
            for coordinate_system in coordinate_systems:
                self.coordinate_system_combo.addItem(coordinate_system, coordinate_system)

            self.empty_state_label.hide()
            self.coordinate_system_combo.setEnabled(bool(coordinate_systems))

        self._sync_coordinate_system_combo_selection(self._app_state.coordinate_system)
        self._refresh_coordinate_system_content()

    def _refresh_coordinate_system_content(self) -> None:
        sdata = self._app_state.sdata
        coordinate_system = self._app_state.coordinate_system

        if sdata is None or not coordinate_system:
            self._clear_cards()
            self._update_section_empty_states([], [], [], [])
            if sdata is None:
                self.summary_label.setText("No SpatialData loaded.")
            else:
                self.summary_label.setText("No coordinate system selected.")
            return

        labels_names = _get_labels_in_coordinate_system(sdata, coordinate_system)
        image_names = _get_images_in_coordinate_system(sdata, coordinate_system)
        shapes_names = _get_shapes_in_coordinate_system(sdata, coordinate_system)
        points_names = _get_points_in_coordinate_system(sdata, coordinate_system)

        self.summary_label.setText(
            f"In coordinate system `{coordinate_system}`: "
            f"{len(image_names)} image element(s), {len(labels_names)} labels element(s), "
            f"{len(shapes_names)} shapes element(s), and {len(points_names)} points element(s)."
        )
        self._rebuild_image_cards(sdata, image_names)
        self._rebuild_labels_cards(sdata, labels_names)
        self._rebuild_shapes_cards(sdata, shapes_names)
        self._refresh_points_section(sdata, points_names)
        self._update_section_empty_states(image_names, labels_names, shapes_names, points_names)

    def _rebuild_image_cards(self, sdata: SpatialData, image_names: list[str]) -> None:
        _clear_layout(self.images_section_layout)
        self._image_cards = []
        self._image_rows = []
        self._expanded_image_names.intersection_update(image_names)

        for image_name in image_names:
            channel_error = None
            channel_names: list[str] = []
            try:
                channel_names = get_image_channel_names_from_sdata(sdata, image_name)
            except ValueError as error:
                channel_error = str(error)
            card = _ImageCardWidget(
                image_name=image_name,
                channel_names=channel_names,
                channel_error=channel_error,
            )
            card.add_update_requested.connect(self._add_or_update_image_layer)
            row = _DisclosureElementWidget(
                title=image_name,
                object_name=f"viewer_widget_image_row_{image_name}",
                toggle_object_name=f"viewer_widget_image_row_toggle_{image_name}",
                detail_widget=card,
                expanded=image_name in self._expanded_image_names,
            )
            row.expanded_changed.connect(
                lambda expanded, *, name=image_name: self._on_image_row_expanded(
                    name,
                    expanded,
                )
            )
            self.images_section_layout.addWidget(row)
            self._image_cards.append(card)
            self._image_rows.append(row)

    def _rebuild_labels_cards(self, sdata: SpatialData, labels_names: list[str]) -> None:
        _clear_layout(self.labels_section_layout)
        self._labels_cards = []
        self._labels_rows = []
        self._expanded_labels_names.intersection_update(labels_names)

        for labels_name in labels_names:
            table_names = get_annotating_table_names(sdata, labels_name)
            table_color_sources_by_table = {
                table_name: get_table_color_source_options(sdata, table_name) for table_name in table_names
            }
            card = _LabelsCardWidget(
                labels_name=labels_name,
                table_names=table_names,
                table_color_sources_by_table=table_color_sources_by_table,
            )
            card.add_update_requested.connect(self._add_or_update_labels_layer)
            row = _DisclosureElementWidget(
                title=labels_name,
                object_name=f"viewer_widget_labels_row_{labels_name}",
                toggle_object_name=f"viewer_widget_labels_row_toggle_{labels_name}",
                detail_widget=card,
                expanded=labels_name in self._expanded_labels_names,
            )
            row.expanded_changed.connect(
                lambda expanded, *, name=labels_name: self._on_labels_row_expanded(
                    name,
                    expanded,
                )
            )
            self.labels_section_layout.addWidget(row)
            self._labels_cards.append(card)
            self._labels_rows.append(row)

    def _rebuild_shapes_cards(self, sdata: SpatialData, shapes_names: list[str]) -> None:
        _clear_layout(self.shapes_section_layout)
        self._shape_cards = []
        self._shape_rows = []
        self._expanded_shapes_names.intersection_update(shapes_names)

        for shapes_name in shapes_names:
            card = _ShapesCardWidget(
                shapes_name=shapes_name,
                shape_color_sources=get_shape_column_color_source_options(sdata, shapes_name),
            )
            card.add_update_requested.connect(self._add_or_update_shapes_layer)
            row = _DisclosureElementWidget(
                title=shapes_name,
                object_name=f"viewer_widget_shapes_row_{shapes_name}",
                toggle_object_name=f"viewer_widget_shapes_row_toggle_{shapes_name}",
                detail_widget=card,
                expanded=shapes_name in self._expanded_shapes_names,
            )
            row.expanded_changed.connect(
                lambda expanded, *, name=shapes_name: self._on_shapes_row_expanded(
                    name,
                    expanded,
                )
            )
            self.shapes_section_layout.addWidget(row)
            self._shape_cards.append(card)
            self._shape_rows.append(row)

    def _refresh_points_section(self, sdata: SpatialData, points_names: list[str]) -> None:
        self.points_widget.set_points_names(points_names)
        self._refresh_points_index_columns(sdata)
        self._bind_points_source()

    def _refresh_points_index_columns(self, sdata: SpatialData | None) -> None:
        points_name = self.points_widget.selected_points_name()
        index_columns = [] if sdata is None or points_name is None else _get_points_index_columns(sdata, points_name)
        self.points_widget.set_index_columns(index_columns)

    def _on_points_source_changed(self) -> None:
        sdata = self._app_state.sdata
        self._refresh_points_index_columns(sdata)
        self._bind_points_source()

    def _bind_points_source(self) -> None:
        sdata = self._app_state.sdata
        coordinate_system = self._app_state.coordinate_system
        points_name = self.points_widget.selected_points_name()
        index_column = self.points_widget.selected_index_column()

        changed = self._points_controller.bind_source(
            sdata,
            points_name,
            coordinate_system,
            index_column,
        )
        if changed:
            self._last_points_load_result = None
            self._last_points_layer_result = None
        if changed and self._points_controller.can_load_values:
            self._points_controller.load_value_source()
        else:
            self.points_widget.render_controller_state(self._points_controller)

    def _add_or_update_points_selection(self, values: Sequence[str] | Literal["all"], render_point_budget: int) -> None:
        self._last_points_load_result = None
        self._last_points_layer_result = None
        self._points_controller.load_selection(
            values,
            render_point_budget=render_point_budget,
        )

    def _on_points_controller_state_changed(self) -> None:
        self.points_widget.render_controller_state(self._points_controller)
        if self._last_points_load_result is not None and self._last_points_layer_result is not None:
            self._render_points_loaded_status(self._last_points_load_result, self._last_points_layer_result)

    def _on_points_value_source_loaded(self, value_source: PointsValueSource) -> None:
        self.points_widget.set_value_source(value_source)

    def _on_points_loaded(self, load_result: PointsLoadResult) -> None:
        try:
            layer_result = self._app_state.viewer_adapter._ensure_points_layer_from_selection(
                load_result.identity,
                selection=load_result.selection,
            )
        except ValueError as error:
            self._set_action_feedback(
                title="Points Layer Error",
                lines=[str(error)],
                kind="error",
            )
            return

        self._last_points_load_result = load_result
        self._last_points_layer_result = layer_result
        self._app_state.viewer_adapter.activate_layer(layer_result.layer)
        self._render_points_loaded_status(load_result, layer_result)

    def _render_points_loaded_status(
        self,
        load_result: PointsLoadResult,
        layer_result: PointsLayerResult,
    ) -> None:
        selection = load_result.selection
        action = "Created" if layer_result.created else "Updated"
        lines = [
            (
                f"{action} points layer for `{load_result.identity.points_name}` "
                f"by `{selection.index_column}` with {selection.loaded_count:,} point(s)."
            )
        ]
        kind: StatusCardKind = "success"
        title = f"Points Layer {action}"
        if selection.warning:
            kind = "warning"
            title = f"{title} With Warning"
            lines.append(selection.warning)
        if layer_result.categorical_coloring_disabled:
            kind = "warning"
            if not title.endswith(" With Warning"):
                title = f"{title} With Warning"
            lines.append(
                f"Categorical coloring is disabled for {layer_result.selected_value_count:,} selected values; "
                f"using one solid color because the categorical limit is {layer_result.categorical_limit:,}."
            )
        self._set_action_feedback(title=title, lines=lines, kind=kind)

    def _on_image_row_expanded(
        self,
        image_name: str,
        expanded: bool,
    ) -> None:
        if expanded:
            self._expanded_image_names.add(image_name)
            return

        self._expanded_image_names.discard(image_name)

    def _on_labels_row_expanded(
        self,
        labels_name: str,
        expanded: bool,
    ) -> None:
        if expanded:
            self._expanded_labels_names.add(labels_name)
            return

        self._expanded_labels_names.discard(labels_name)

    def _on_shapes_row_expanded(
        self,
        shapes_name: str,
        expanded: bool,
    ) -> None:
        if expanded:
            self._expanded_shapes_names.add(shapes_name)
            return

        self._expanded_shapes_names.discard(shapes_name)

    def _add_or_update_labels_layer(self, request: LabelsLoadRequest) -> None:
        if request.selected_source_kind is None:
            self._add_or_update_primary_labels_layer(request.labels_name)
            return

        self._add_or_update_styled_labels_layer(request)

    def _add_or_update_primary_labels_layer(self, labels_name: str) -> None:
        sdata = self._app_state.sdata
        coordinate_system = self._app_state.coordinate_system

        if sdata is None or not coordinate_system:
            self._set_action_feedback(
                title="Labels Load Error",
                lines=["Load a SpatialData object and select a coordinate system first."],
                kind="error",
            )
            return

        try:
            layer = self._app_state.viewer_adapter.ensure_labels_loaded(sdata, labels_name, coordinate_system)
        except ValueError as error:
            self._set_action_feedback(title="Labels Load Error", lines=[str(error)], kind="error")
            return

        self._app_state.viewer_adapter.activate_layer(layer)
        display_name, was_shortened = format_feedback_identifier(labels_name)
        self._set_action_feedback(
            title="Labels Loaded",
            lines=[f"Loaded labels `{display_name}` in coordinate system `{coordinate_system}`."],
            kind="success",
            tooltip_message=(
                f"Loaded labels `{labels_name}` in coordinate system `{coordinate_system}`." if was_shortened else None
            ),
        )

    def _add_or_update_styled_labels_layer(self, request: LabelsLoadRequest) -> None:
        sdata = self._app_state.sdata
        coordinate_system = self._app_state.coordinate_system

        if sdata is None or not coordinate_system:
            self._set_action_feedback(
                title="Colored Overlay Error",
                lines=["Load a SpatialData object and select a coordinate system first."],
                kind="error",
            )
            return

        if request.table_name is None:
            self._set_action_feedback(
                title="Colored Overlay Error",
                lines=[f"Labels element `{request.labels_name}` has no linked table for table-driven coloring."],
                kind="error",
            )
            return

        if request.selected_color_source is None:
            missing_source_label = "observation column" if request.selected_source_kind == "obs_column" else "var"
            missing_source_article = "an" if request.selected_source_kind == "obs_column" else "a"
            self._set_action_feedback(
                title="Colored Overlay Error",
                lines=[
                    f"Select {missing_source_article} {missing_source_label} "
                    f"to create a colored overlay for `{request.labels_name}`."
                ],
                kind="error",
            )
            return

        try:
            result = self._app_state.viewer_adapter.ensure_styled_labels_loaded(
                sdata,
                request.labels_name,
                coordinate_system,
                request.selected_color_source,
            )
        except ValueError as error:
            self._set_action_feedback(title="Colored Overlay Error", lines=[str(error)], kind="error")
            return

        self._app_state.viewer_adapter.activate_layer(result.layer)
        action = "Created" if result.created else "Updated"
        if request.selected_color_source.source_kind == "obs_column":
            source_text = f'obs["{request.selected_color_source.value_key}"]'
        else:
            source_text = f'X[:, "{request.selected_color_source.value_key}"]'

        action_line = (
            f"{action} colored overlay for {source_text} on labels element `{request.labels_name}` "
            f"in coordinate system `{coordinate_system}`."
        )
        feedback_kind: StatusCardKind = "success"
        title = f"Colored Overlay {action}"
        lines = [action_line]
        if result.value_kind == "instance":
            lines.append("Used instance label colors.")
        elif result.coercion_applied:
            feedback_kind = "warning"
            title = f"{title} With Warning"
            lines.append("Coerced string values to categorical and used the default categorical palette.")
        elif result.palette_source == "stored":
            lines.append("Used the stored categorical palette.")
        elif result.palette_source == "default_invalid":
            feedback_kind = "warning"
            title = f"{title} With Warning"
            lines.append("The stored categorical palette was invalid, so Harpy used the default categorical palette.")
        elif result.palette_source == "default_missing":
            feedback_kind = "info"
            lines.append("Used the default categorical palette because no stored palette was present.")

        self._set_action_feedback(
            title=title,
            lines=lines,
            kind=feedback_kind,
        )

    def _add_or_update_shapes_layer(self, request: ShapesLoadRequest) -> None:
        if request.selected_source_kind is None:
            self._add_or_update_primary_shapes_layer(request.shapes_name)
            return

        self._add_or_update_styled_shapes_layer(request)

    def _add_or_update_primary_shapes_layer(self, shapes_name: str) -> None:
        sdata = self._app_state.sdata
        coordinate_system = self._app_state.coordinate_system

        if sdata is None or not coordinate_system:
            self._set_action_feedback(
                title="Shapes Load Error",
                lines=["Load a SpatialData object and select a coordinate system first."],
                kind="error",
            )
            return

        try:
            layer = self._app_state.viewer_adapter.ensure_shapes_loaded(sdata, shapes_name, coordinate_system)
        except ValueError as error:
            self._set_action_feedback(title="Shapes Load Error", lines=[str(error)], kind="error")
            return

        self._app_state.viewer_adapter.activate_layer(layer)
        display_name, was_shortened = format_feedback_identifier(shapes_name)
        skipped_geometry_count = _get_layer_skipped_geometry_count(self._app_state.viewer_adapter, layer)
        feedback_kind: StatusCardKind = "success"
        title = "Shapes Loaded"
        lines = [f"Loaded shapes `{display_name}` in coordinate system `{coordinate_system}`."]
        if skipped_geometry_count:
            feedback_kind = "warning"
            title = "Shapes Loaded With Warning"
            lines.append(
                f"Skipped {skipped_geometry_count} empty, invalid, or unsupported "
                "geometries while loading renderable shapes."
            )

        self._set_action_feedback(
            title=title,
            lines=lines,
            kind=feedback_kind,
            tooltip_message=(
                f"Loaded shapes `{shapes_name}` in coordinate system `{coordinate_system}`." if was_shortened else None
            ),
        )

    def _add_or_update_styled_shapes_layer(self, request: ShapesLoadRequest) -> None:
        sdata = self._app_state.sdata
        coordinate_system = self._app_state.coordinate_system

        if sdata is None or not coordinate_system:
            self._set_action_feedback(
                title="Styled Shapes Error",
                lines=["Load a SpatialData object and select a coordinate system first."],
                kind="error",
            )
            return

        if request.selected_color_source is None:
            self._set_action_feedback(
                title="Styled Shapes Error",
                lines=[f"Select a shapes column to create a styled shapes layer for `{request.shapes_name}`."],
                kind="error",
            )
            return

        try:
            result = self._app_state.viewer_adapter.ensure_styled_shapes_loaded(
                sdata,
                request.shapes_name,
                coordinate_system,
                request.selected_color_source,
                fill=request.fill_shapes,
            )
        except ValueError as error:
            self._set_action_feedback(title="Styled Shapes Error", lines=[str(error)], kind="error")
            return

        self._app_state.viewer_adapter.activate_layer(result.layer)
        action = "Created" if result.created else "Updated"
        source_text = f'column "{request.selected_color_source.value_key}"'
        action_line = (
            f"{action} styled shapes layer for {source_text} on shapes element `{request.shapes_name}` "
            f"in coordinate system `{coordinate_system}`."
        )
        feedback_kind: StatusCardKind = "success"
        title = f"Styled Shapes {action}"
        lines = [action_line]
        if result.coercion_applied:
            feedback_kind = "warning"
            title = f"{title} With Warning"
            lines.append("Coerced string values to categorical and used the default categorical palette.")
        elif result.palette_source == "stored":
            lines.append("Used the stored categorical palette.")
        elif result.palette_source == "default_invalid":
            feedback_kind = "warning"
            title = f"{title} With Warning"
            lines.append("The stored categorical palette was invalid, so Harpy used the default categorical palette.")
        elif result.palette_source == "default_missing":
            feedback_kind = "info"
            lines.append("Used the default categorical palette because no stored palette was present.")

        skipped_geometry_count = _get_layer_skipped_geometry_count(self._app_state.viewer_adapter, result.layer)
        if skipped_geometry_count:
            feedback_kind = "warning"
            if not title.endswith(" With Warning"):
                title = f"{title} With Warning"
            lines.append(
                f"Skipped {skipped_geometry_count} empty, invalid, or unsupported "
                "geometries while loading renderable shapes."
            )

        self._set_action_feedback(
            title=title,
            lines=lines,
            kind=feedback_kind,
        )

    def _add_or_update_image_layer(self, request: ImageLoadRequest) -> None:
        sdata = self._app_state.sdata
        coordinate_system = self._app_state.coordinate_system
        image_name = request.image_name

        if sdata is None or not coordinate_system:
            self._set_action_feedback(
                title="Image Load Error",
                lines=["Load a SpatialData object and select a coordinate system first."],
                kind="error",
            )
            return

        mode = request.mode
        if mode == "overlay" and not request.channels:
            self._app_state.viewer_adapter.remove_image_layers(sdata, image_name, coordinate_system)
            self._set_action_feedback(
                title="Image Load Error",
                lines=["Overlay mode requires at least one selected channel."],
                kind="error",
            )
            return

        try:
            layer_or_layers = self._app_state.viewer_adapter.ensure_image_loaded(
                sdata,
                image_name,
                coordinate_system,
                mode=mode,
                channels=request.channels if mode == "overlay" else None,
                channel_colors=request.channel_colors if mode == "overlay" else None,
            )
        except ValueError as error:
            self._set_action_feedback(title="Image Load Error", lines=[str(error)], kind="error")
            return

        if mode == "stack":
            if isinstance(layer_or_layers, list):
                self._set_action_feedback(
                    title="Image Load Error",
                    lines=[f"Expected one stack image layer for `{image_name}`, but received multiple layers."],
                    kind="error",
                )
                return

            self._app_state.viewer_adapter.activate_layer(layer_or_layers)
            display_name, was_shortened = format_feedback_identifier(image_name)
            self._set_action_feedback(
                title="Image Loaded",
                lines=[f"Loaded image `{display_name}` in stack mode for coordinate system `{coordinate_system}`."],
                kind="success",
                tooltip_message=(
                    f"Loaded image `{image_name}` in stack mode for coordinate system `{coordinate_system}`."
                    if was_shortened
                    else None
                ),
            )
            return

        if not isinstance(layer_or_layers, list):
            self._set_action_feedback(
                title="Image Load Error",
                lines=[f"Expected overlay image layers for `{image_name}`, but received a single layer."],
                kind="error",
            )
            return
        if not layer_or_layers:
            self._set_action_feedback(
                title="Image Load Error",
                lines=[f"No overlay layers were returned for image `{image_name}`."],
                kind="error",
            )
            return

        self._app_state.viewer_adapter.activate_layer(layer_or_layers[0])
        display_name, was_shortened = format_feedback_identifier(image_name)
        self._set_action_feedback(
            title="Image Loaded",
            lines=[
                f"Loaded image `{display_name}` in overlay mode for channels {request.channels} "
                f"in coordinate system `{coordinate_system}`."
            ],
            kind="success",
            tooltip_message=(
                f"Loaded image `{image_name}` in overlay mode for channels {request.channels} "
                f"in coordinate system `{coordinate_system}`."
                if was_shortened
                else None
            ),
        )

    def _update_section_empty_states(
        self,
        image_names: list[str],
        labels_names: list[str],
        shapes_names: list[str],
        points_names: list[str],
    ) -> None:
        self.images_group.set_count(len(image_names))
        self.labels_group.set_count(len(labels_names))
        self.shapes_group.set_count(len(shapes_names))
        self.points_group.set_count(len(points_names))
        self.images_empty_label.setVisible(not image_names)
        self.labels_empty_label.setVisible(not labels_names)
        self.shapes_empty_label.setVisible(not shapes_names)
        self.points_empty_label.setVisible(not points_names)
        self.images_section.setVisible(bool(image_names))
        self.labels_section.setVisible(bool(labels_names))
        self.shapes_section.setVisible(bool(shapes_names))
        self.points_widget.setVisible(bool(points_names))

    def _clear_cards(self) -> None:
        _clear_layout(self.images_section_layout)
        _clear_layout(self.labels_section_layout)
        _clear_layout(self.shapes_section_layout)
        self._image_cards = []
        self._labels_cards = []
        self._shape_cards = []
        self._image_rows = []
        self._labels_rows = []
        self._shape_rows = []
        self._expanded_image_names.clear()
        self._expanded_labels_names.clear()
        self._expanded_shapes_names.clear()
        self.points_widget.set_points_names([])
        self.points_widget.set_index_columns([])
        self.points_widget.set_value_source(None)
        self._last_points_load_result = None
        self._last_points_layer_result = None
        self._points_controller.bind_source(None, None, None, None)

    def _set_action_feedback(
        self,
        message: str | None = None,
        *,
        title: str | None = None,
        lines: list[str] | None = None,
        kind: StatusCardKind | None = None,
        is_error: bool | None = None,
        tooltip_message: str | None = None,
    ) -> None:
        """Render viewer action feedback as the shared status-card pattern."""
        if lines is None:
            if message is None:
                lines = []
            else:
                lines = [message]
        if kind is None:
            kind = "error" if is_error else "success"
        if title is None:
            title = "Viewer Error" if kind == "error" else "Viewer Updated"

        set_status_card(
            self.action_feedback_label,
            title=title,
            lines=lines,
            kind=kind,
            tooltip_message=tooltip_message,
        )

    def _clear_action_feedback(self) -> None:
        self.action_feedback_label.clear()
        self.action_feedback_label.setToolTip("")
        self.action_feedback_label.setStyleSheet("")
        self.action_feedback_label.hide()

    def _sync_coordinate_system_combo_selection(self, coordinate_system: str | None) -> None:
        with QSignalBlocker(self.coordinate_system_combo):
            if coordinate_system is None:
                self.coordinate_system_combo.setCurrentIndex(-1)
                return

            index = self.coordinate_system_combo.findData(coordinate_system)
            self.coordinate_system_combo.setCurrentIndex(index)

    def _create_header_logo(self) -> QLabel:
        logo_label = QLabel()
        logo_label.setObjectName("viewer_widget_header_logo")
        logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        logo_pixmap = QPixmap(str(self._logo_path))
        if not logo_pixmap.isNull():
            logo_label.setPixmap(logo_pixmap.scaledToWidth(120, Qt.TransformationMode.SmoothTransformation))
            return logo_label

        logo_label.setText("napari-harpy")
        logo_label.setStyleSheet("color: #111827; font-size: 18px; font-weight: 600;")
        return logo_label


def _clear_layout(layout: QLayout) -> None:
    while layout.count():
        item = layout.takeAt(0)
        widget = item.widget()
        child_layout = item.layout()
        if widget is not None:
            widget.deleteLater()
        elif child_layout is not None:
            _clear_layout(child_layout)


def _create_form_label(text: str) -> QLabel:
    return create_form_label(text)


def _get_labels_in_coordinate_system(sdata: SpatialData, coordinate_system: str) -> list[str]:
    return [
        option.labels_name
        for option in get_spatialdata_labels_options_for_coordinate_system_from_sdata(
            sdata=sdata,
            coordinate_system=coordinate_system,
        )
    ]


def _get_images_in_coordinate_system(sdata: SpatialData, coordinate_system: str) -> list[str]:
    return [
        option.image_name
        for option in get_spatialdata_image_options_for_coordinate_system_from_sdata(
            sdata=sdata,
            coordinate_system=coordinate_system,
        )
    ]


def _get_shapes_in_coordinate_system(sdata: SpatialData, coordinate_system: str) -> list[str]:
    return [
        option.shapes_name
        for option in get_spatialdata_shapes_options_for_coordinate_system_from_sdata(
            sdata=sdata,
            coordinate_system=coordinate_system,
        )
    ]


def _get_points_in_coordinate_system(sdata: SpatialData, coordinate_system: str) -> list[str]:
    return [
        option.points_name
        for option in get_spatialdata_points_options_for_coordinate_system_from_sdata(
            sdata=sdata,
            coordinate_system=coordinate_system,
        )
    ]


def _get_points_index_columns(sdata: SpatialData, points_name: str) -> list[str]:
    points = getattr(sdata, "points", {}).get(points_name)
    if points is None:
        return []

    meta = getattr(points, "_meta", points)
    columns = list(getattr(meta, "columns", ()))
    index_columns: list[str] = []
    for column in columns:
        column_name = str(column)
        if column_name in {"x", "y"}:
            continue

        dtype = getattr(meta[column], "dtype", None)
        if dtype is None or is_bool_dtype(dtype) or is_numeric_dtype(dtype):
            continue
        if isinstance(dtype, pd.CategoricalDtype) or is_string_dtype(dtype) or is_object_dtype(dtype):
            index_columns.append(column_name)

    return index_columns


def _get_layer_skipped_geometry_count(viewer_adapter: ViewerAdapter, layer: object) -> int:
    binding = viewer_adapter.layer_bindings.get_binding(layer)
    if not isinstance(binding, ShapesLayerBinding):
        return 0

    try:
        return int(binding.skipped_geometry_count)
    except (TypeError, ValueError):
        return 0
