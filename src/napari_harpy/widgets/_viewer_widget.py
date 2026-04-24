from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from qtpy.QtCore import QSignalBlocker, QStringListModel, Qt, Signal
from qtpy.QtGui import QPixmap
from qtpy.QtWidgets import (
    QCheckBox,
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
from spatialdata.transformations import get_transformation

from napari_harpy._app_state import HarpyAppState, get_or_create_app_state
from napari_harpy._spatialdata import (
    get_annotating_table_names,
    get_image_channel_names_from_sdata,
    get_table_color_source_options,
)
from napari_harpy._table_color_source import ColorSourceKind, TableColorSourceSpec
from napari_harpy._viewer_adapter import DEFAULT_OVERLAY_COLORS
from napari_harpy.widgets._shared_styles import (
    ACTION_BUTTON_STYLESHEET as _ACTION_BUTTON_STYLESHEET,
)
from napari_harpy.widgets._shared_styles import (
    CHECKBOX_STYLESHEET as _CHECKBOX_STYLESHEET,
)
from napari_harpy.widgets._shared_styles import (
    WIDGET_MIN_WIDTH as _WIDGET_MIN_WIDTH,
)
from napari_harpy.widgets._shared_styles import (
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

if TYPE_CHECKING:
    import napari
    from spatialdata import SpatialData


_INPUT_CONTROL_STYLESHEET = build_input_control_stylesheet("QComboBox")
_DETAIL_PANEL_STYLESHEET = (
    "QFrame[harpyViewerDetailPanel='true'] {"
    "background-color: #fff7f3; "
    "border: 1px solid #eadfd8; "
    "border-radius: 8px;}"
)
_CARD_TITLE_STYLESHEET = (
    "QLabel {"
    "background-color: #EFDCCF; "
    "border: 1px solid #D3B19E; "
    "border-radius: 8px; "
    "color: #374151; "
    "font-weight: 700; "
    "padding: 6px 10px;}"
)
_SECTION_GROUP_STYLESHEET = (
    "QFrame[harpyViewerDisclosureSection='true'] {"
    "background-color: #f7ece7; "
    "border: 1px solid #e3d2c8; "
    "border-radius: 12px;}"
)
_DISCLOSURE_BUTTON_STYLESHEET = (
    "QToolButton {"
    "background-color: #fff8f5; "
    "border: 1px solid #eadfd8; "
    "border-radius: 8px; "
    "color: #374151; "
    "font-weight: 700; "
    "padding: 7px 10px; "
    "text-align: left;}"
    "QToolButton:hover { background-color: #f6e8e0; border-color: #d9c5ba; }"
    "QToolButton:checked { background-color: #efdccf; border-color: #d3b19e; }"
)
_ELEMENT_DISCLOSURE_STYLESHEET = (
    "QFrame[harpyViewerDisclosureRow='true'] {"
    "background-color: #fffaf7; "
    "border: 1px solid #eadfd8; "
    "border-radius: 10px;}"
)
_SUMMARY_LABEL_STYLESHEET = "color: #374151; font-weight: 500;"
_EMPTY_STATE_STYLESHEET = "color: #6b7280; font-weight: 500;"
_CHANNEL_WARNING_STYLESHEET = "color: #b45309; font-weight: 600;"
_CHANNEL_PANEL_STYLESHEET = "QWidget { background: transparent; }"
_MAX_VISIBLE_OVERLAY_CHANNELS = 5


@dataclass(frozen=True)
class ImageLoadRequest:
    image_name: str
    mode: str
    channels: list[int]
    channel_colors: list[str]


@dataclass(frozen=True)
class LabelsLoadRequest:
    label_name: str
    table_name: str | None
    selected_source_kind: ColorSourceKind | None
    selected_color_source: TableColorSourceSpec | None


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

        self.toggle_button = QToolButton()
        self.toggle_button.setObjectName(toggle_object_name)
        self.toggle_button.setCheckable(True)
        self.toggle_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.toggle_button.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.toggle_button.setMinimumWidth(0)
        self.toggle_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
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
        self.toggle_button.setText(f"{self._title} ({count})")
        self._update_accessible_text()

    def set_expanded(self, expanded: bool) -> None:
        with QSignalBlocker(self.toggle_button):
            self.toggle_button.setChecked(expanded)
        self._sync_expanded_state(expanded)

    def _on_toggled(self, expanded: bool) -> None:
        self._sync_expanded_state(expanded)

    def _sync_expanded_state(self, expanded: bool) -> None:
        self.content_widget.setVisible(expanded)
        self.toggle_button.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)
        self._update_accessible_text()

    def _update_accessible_text(self) -> None:
        state = "expanded" if self.is_expanded() else "collapsed"
        self.toggle_button.setAccessibleName(f"{self.toggle_button.text()} section, {state}")
        self.toggle_button.setToolTip(f"{self.toggle_button.text()} section is {state}.")


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

        self.toggle_button = QToolButton()
        self.toggle_button.setObjectName(toggle_object_name)
        self.toggle_button.setText(title)
        self.toggle_button.setToolTip(format_tooltip(title))
        self.toggle_button.setCheckable(True)
        self.toggle_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.toggle_button.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.toggle_button.setMinimumWidth(0)
        self.toggle_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
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
        self.toggle_button.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)
        state = "expanded" if expanded else "collapsed"
        self.toggle_button.setAccessibleName(f"{self.title} element, {state}")
        self.toggle_button.setToolTip(format_tooltip(f"{self.title} element is {state}."))


class _LabelsCardWidget(QFrame):
    """Card UI for one labels element in the selected coordinate system."""

    add_update_requested = Signal(object)

    def __init__(
        self,
        *,
        label_name: str,
        table_names: list[str],
        table_color_sources_by_table: dict[str, list[TableColorSourceSpec]],
    ) -> None:
        super().__init__()
        self.label_name = label_name
        self._table_color_sources_by_table = table_color_sources_by_table
        self._filtered_color_sources: list[TableColorSourceSpec] = []
        self.setObjectName(f"viewer_widget_labels_card_{label_name}")
        self.setProperty("harpyViewerDetailPanel", True)
        self.setStyleSheet(_DETAIL_PANEL_STYLESHEET)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        self.title_label = _ElidedLabel(label_name, self)
        self.title_label.setObjectName(f"viewer_widget_labels_card_title_{label_name}")
        self.title_label.setStyleSheet(_CARD_TITLE_STYLESHEET)
        self.title_label.hide()

        form_layout = QFormLayout()
        form_layout.setContentsMargins(0, 0, 0, 0)
        form_layout.setHorizontalSpacing(8)
        form_layout.setVerticalSpacing(6)

        linked_table_label = _create_form_label("Linked table")
        self.linked_table_combo = CompactComboBox()
        self.linked_table_combo.setObjectName(f"viewer_widget_linked_table_combo_{label_name}")
        self.linked_table_combo.setStyleSheet(_INPUT_CONTROL_STYLESHEET)
        if table_names:
            self.linked_table_combo.addItems(table_names)
        else:
            self.linked_table_combo.addItem("No linked tables")
            self.linked_table_combo.setEnabled(False)

        color_source_kind_label = _create_form_label("Color source")
        self.color_source_kind_combo = CompactComboBox()
        self.color_source_kind_combo.setObjectName(f"viewer_widget_color_source_kind_combo_{label_name}")
        self.color_source_kind_combo.setStyleSheet(_INPUT_CONTROL_STYLESHEET)
        self.color_source_kind_combo.addItem("None", None)
        self.color_source_kind_combo.addItem("Observations", "obs_column")
        self.color_source_kind_combo.addItem("Vars", "x_var")

        self.color_source_value_label = _create_form_label("Value source")
        self.color_source_value_input = QLineEdit()
        self.color_source_value_input.setObjectName(f"viewer_widget_color_source_value_input_{label_name}")
        self.color_source_value_input.setStyleSheet(
            build_input_control_stylesheet("QLineEdit")
        )
        self.color_source_value_input.setMinimumWidth(0)
        self.color_source_value_input.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.color_source_value_input.setEnabled(False)

        self._color_source_completer_model = QStringListModel(self.color_source_value_input)
        self._color_source_completer = QCompleter(self._color_source_completer_model, self.color_source_value_input)
        self._color_source_completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self._color_source_completer.setFilterMode(Qt.MatchFlag.MatchContains)
        self.color_source_value_input.setCompleter(self._color_source_completer)

        self.action_status_label = QLabel()
        self.action_status_label.setObjectName(f"viewer_widget_action_status_{label_name}")
        self.action_status_label.setWordWrap(True)
        self.action_status_label.setStyleSheet(_SUMMARY_LABEL_STYLESHEET)

        self.add_update_button = QPushButton("Add / Update in viewer")
        self.add_update_button.setObjectName(f"viewer_widget_add_update_labels_button_{label_name}")
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
    def selected_source_kind(self) -> ColorSourceKind | None:
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
        selected_source_identity = self.selected_color_source.identity if self.selected_color_source is not None else None
        source_kind = self.selected_source_kind
        table_name = self.selected_table_name

        if source_kind == "obs_column":
            self.color_source_value_label.setText("Observation")
        elif source_kind == "x_var":
            self.color_source_value_label.setText("Var")
        else:
            self.color_source_value_label.setText("Value source")

        available_sources = list(self._table_color_sources_by_table.get(table_name, ())) if table_name is not None else []
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
                        (source for source in self._filtered_color_sources if source.identity == selected_source_identity),
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

        self.action_status_label.setText(
            f'Action: add/update colored overlay for X[:, "{selected_source.value_key}"]'
        )

    def _emit_add_update_request(self, _checked: bool = False) -> None:
        self.add_update_requested.emit(
            LabelsLoadRequest(
                label_name=self.label_name,
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
        channel_layout.setContentsMargins(0, 0, 0, 0)
        channel_layout.setSpacing(6)

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
        self.channel_color_combos: list[QComboBox] = []
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

                color_combo = QComboBox()
                color_combo.setObjectName(f"viewer_widget_channel_color_combo_{image_name}_{channel_name}")
                color_combo.setStyleSheet(_INPUT_CONTROL_STYLESHEET)
                for color in DEFAULT_OVERLAY_COLORS:
                    color_combo.addItem(color, color)
                color_combo.setCurrentIndex(index % color_combo.count())

                row_layout.addWidget(checkbox, 1)
                row_layout.addWidget(color_combo)

                self.channel_list_layout.addWidget(row)
                self.channel_checkboxes.append(checkbox)
                self.channel_color_combos.append(color_combo)
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
            str(color_combo.currentData() or color_combo.currentText())
            for checkbox, color_combo in zip(self.channel_checkboxes, self.channel_color_combos, strict=False)
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


class ViewerWidget(QWidget):
    """Shared viewer widget backed by `HarpyAppState` and `ViewerAdapter`."""

    def __init__(self, napari_viewer: napari.Viewer | None = None) -> None:
        super().__init__()
        self.setObjectName("viewer_widget")
        apply_widget_surface(self)
        self.setMinimumWidth(_WIDGET_MIN_WIDTH)
        self._viewer = napari_viewer
        self._app_state = get_or_create_app_state(napari_viewer)
        self._labels_cards: list[_LabelsCardWidget] = []
        self._image_cards: list[_ImageCardWidget] = []
        self._labels_rows: list[_DisclosureElementWidget] = []
        self._image_rows: list[_DisclosureElementWidget] = []
        self._expanded_label_names: set[str] = set()
        self._expanded_image_names: set[str] = set()
        self._logo_path = Path(__file__).resolve().parents[3] / "docs" / "_static" / "logo.png"

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

        self.labels_empty_label = QLabel("No segmentation masks available in the selected coordinate system.")
        self.labels_empty_label.setObjectName("viewer_widget_labels_empty_state")
        self.labels_empty_label.setWordWrap(True)
        self.labels_empty_label.setStyleSheet(_EMPTY_STATE_STYLESHEET)

        self.labels_section = QWidget()
        self.labels_section.setObjectName("viewer_widget_labels_section")
        self.labels_section_layout = QVBoxLayout(self.labels_section)
        self.labels_section_layout.setContentsMargins(0, 0, 0, 0)
        self.labels_section_layout.setSpacing(8)
        self.labels_group = _CollapsibleSectionWidget(
            title="Segmentations",
            object_name="viewer_widget_labels_group",
            toggle_object_name="viewer_widget_labels_section_toggle",
            expanded=False,
        )
        self.labels_group.content_layout.addWidget(self.labels_empty_label)
        self.labels_group.content_layout.addWidget(self.labels_section)
        self.labels_section_toggle = self.labels_group.toggle_button
        self.labels_section_title = self.labels_section_toggle

        self.content_layout.addWidget(header_logo)
        self.content_layout.addWidget(title)
        self.content_layout.addWidget(self.open_sdata_button)
        self.content_layout.addWidget(self.empty_state_label)
        self.content_layout.addWidget(self.summary_label)
        self.content_layout.addLayout(selector_layout)
        self.content_layout.addWidget(self.action_feedback_label)
        self.content_layout.addWidget(self.images_group)
        self.content_layout.addWidget(self.labels_group)
        self.content_layout.addStretch(1)

        self.scroll_area.setWidget(self.scroll_content)
        root_layout.addWidget(self.scroll_area)

        self._app_state.sdata_changed.connect(self._on_sdata_changed)
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
    def image_rows(self) -> list[_DisclosureElementWidget]:
        """Return the currently visible compact image rows."""
        return list(self._image_rows)

    @property
    def labels_rows(self) -> list[_DisclosureElementWidget]:
        """Return the currently visible compact labels rows."""
        return list(self._labels_rows)

    def _on_sdata_changed(self, sdata: SpatialData | None) -> None:
        """Refresh the widget when the shared loaded `SpatialData` changes."""
        self.refresh_from_sdata(sdata)

    def _on_coordinate_system_changed(self) -> None:
        """Refresh the filtered image and labels cards when the coordinate system changes."""
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
            previous_coordinate_system = self.coordinate_system_combo.currentText()
            self.coordinate_system_combo.clear()

            if sdata is None:
                self.empty_state_label.show()
                self.summary_label.setText("No SpatialData loaded.")
                self.coordinate_system_combo.setEnabled(False)
                self._clear_cards()
                self._update_section_empty_states([], [])
                return

            coordinate_systems = _get_coordinate_systems_from_sdata(sdata)
            self.coordinate_system_combo.addItems(coordinate_systems)

            if previous_coordinate_system in coordinate_systems:
                self.coordinate_system_combo.setCurrentIndex(coordinate_systems.index(previous_coordinate_system))
            elif coordinate_systems:
                self.coordinate_system_combo.setCurrentIndex(0)

            self.empty_state_label.hide()
            self.coordinate_system_combo.setEnabled(bool(coordinate_systems))

        self._refresh_coordinate_system_content()

    def _refresh_coordinate_system_content(self) -> None:
        sdata = self._app_state.sdata
        coordinate_system = self.coordinate_system_combo.currentText()

        if sdata is None or not coordinate_system:
            self._clear_cards()
            self._update_section_empty_states([], [])
            if sdata is None:
                self.summary_label.setText("No SpatialData loaded.")
            else:
                self.summary_label.setText("No coordinate system selected.")
            return

        label_names = _get_labels_in_coordinate_system(sdata, coordinate_system)
        image_names = _get_images_in_coordinate_system(sdata, coordinate_system)

        self.summary_label.setText(
            f"In coordinate system `{coordinate_system}`: "
            f"{len(image_names)} image element(s) and {len(label_names)} segmentation mask(s)."
        )
        self._rebuild_image_cards(sdata, image_names)
        self._rebuild_labels_cards(sdata, label_names)
        self._update_section_empty_states(image_names, label_names)

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

    def _rebuild_labels_cards(self, sdata: SpatialData, label_names: list[str]) -> None:
        _clear_layout(self.labels_section_layout)
        self._labels_cards = []
        self._labels_rows = []
        self._expanded_label_names.intersection_update(label_names)

        for label_name in label_names:
            table_names = get_annotating_table_names(sdata, label_name)
            table_color_sources_by_table = {
                table_name: get_table_color_source_options(sdata, table_name) for table_name in table_names
            }
            card = _LabelsCardWidget(
                label_name=label_name,
                table_names=table_names,
                table_color_sources_by_table=table_color_sources_by_table,
            )
            card.add_update_requested.connect(self._add_or_update_labels_layer)
            row = _DisclosureElementWidget(
                title=label_name,
                object_name=f"viewer_widget_labels_row_{label_name}",
                toggle_object_name=f"viewer_widget_labels_row_toggle_{label_name}",
                detail_widget=card,
                expanded=label_name in self._expanded_label_names,
            )
            row.expanded_changed.connect(
                lambda expanded, *, name=label_name: self._on_labels_row_expanded(
                    name,
                    expanded,
                )
            )
            self.labels_section_layout.addWidget(row)
            self._labels_cards.append(card)
            self._labels_rows.append(row)

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
        label_name: str,
        expanded: bool,
    ) -> None:
        if expanded:
            self._expanded_label_names.add(label_name)
            return

        self._expanded_label_names.discard(label_name)

    def _add_or_update_labels_layer(self, request: LabelsLoadRequest) -> None:
        if request.selected_source_kind is None:
            self._add_or_update_primary_labels_layer(request.label_name)
            return

        self._add_or_update_styled_labels_layer(request)

    def _add_or_update_primary_labels_layer(self, label_name: str) -> None:
        sdata = self._app_state.sdata
        coordinate_system = self.coordinate_system_combo.currentText()

        if sdata is None or not coordinate_system:
            self._set_action_feedback(
                title="Segmentation Load Error",
                lines=["Load a SpatialData object and select a coordinate system first."],
                kind="error",
            )
            return

        try:
            layer = self._app_state.viewer_adapter.ensure_labels_loaded(sdata, label_name, coordinate_system)
        except ValueError as error:
            self._set_action_feedback(title="Segmentation Load Error", lines=[str(error)], kind="error")
            return

        self._app_state.viewer_adapter.activate_layer(layer)
        display_name, was_shortened = format_feedback_identifier(label_name)
        self._set_action_feedback(
            title="Segmentation Loaded",
            lines=[f"Loaded segmentation `{display_name}` in coordinate system `{coordinate_system}`."],
            kind="success",
            tooltip_message=(
                f"Loaded segmentation `{label_name}` in coordinate system `{coordinate_system}`."
                if was_shortened
                else None
            ),
        )

    def _add_or_update_styled_labels_layer(self, request: LabelsLoadRequest) -> None:
        sdata = self._app_state.sdata
        coordinate_system = self.coordinate_system_combo.currentText()

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
                lines=[f"Segmentation `{request.label_name}` has no linked table for table-driven coloring."],
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
                    f"to create a colored overlay for `{request.label_name}`."
                ],
                kind="error",
            )
            return

        try:
            result = self._app_state.viewer_adapter.ensure_styled_labels_loaded(
                sdata,
                request.label_name,
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
            f"{action} colored overlay for {source_text} on segmentation `{request.label_name}` "
            f"in coordinate system `{coordinate_system}`."
        )
        feedback_kind: StatusCardKind = "success"
        title = f"Colored Overlay {action}"
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

        self._set_action_feedback(
            title=title,
            lines=lines,
            kind=feedback_kind,
        )

    def _add_or_update_image_layer(self, request: ImageLoadRequest) -> None:
        sdata = self._app_state.sdata
        coordinate_system = self.coordinate_system_combo.currentText()
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

    def _update_section_empty_states(self, image_names: list[str], label_names: list[str]) -> None:
        self.images_group.set_count(len(image_names))
        self.labels_group.set_count(len(label_names))
        self.images_empty_label.setVisible(not image_names)
        self.labels_empty_label.setVisible(not label_names)
        self.images_section.setVisible(bool(image_names))
        self.labels_section.setVisible(bool(label_names))

    def _clear_cards(self) -> None:
        _clear_layout(self.images_section_layout)
        _clear_layout(self.labels_section_layout)
        self._image_cards = []
        self._labels_cards = []
        self._image_rows = []
        self._labels_rows = []
        self._expanded_image_names.clear()
        self._expanded_label_names.clear()

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


def _get_coordinate_systems_from_sdata(sdata: SpatialData) -> list[str]:
    coordinate_systems: set[str] = set()

    for collection_name in ("labels", "images"):
        collection = getattr(sdata, collection_name, {})
        for element in collection.values():
            coordinate_systems.update(get_transformation(element, get_all=True).keys())

    return sorted(coordinate_systems)


def _get_labels_in_coordinate_system(sdata: SpatialData, coordinate_system: str) -> list[str]:
    labels = getattr(sdata, "labels", {})
    return sorted(
        label_name
        for label_name, element in labels.items()
        if coordinate_system in get_transformation(element, get_all=True).keys()
    )


def _get_images_in_coordinate_system(sdata: SpatialData, coordinate_system: str) -> list[str]:
    images = getattr(sdata, "images", {})
    return sorted(
        image_name
        for image_name, element in images.items()
        if coordinate_system in get_transformation(element, get_all=True).keys()
    )
