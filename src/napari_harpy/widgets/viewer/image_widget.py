from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from qtpy.QtCore import QSignalBlocker, QStringListModel, Qt, Signal
from qtpy.QtGui import QColor
from qtpy.QtWidgets import (
    QCheckBox,
    QCompleter,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from napari_harpy.viewer.adapter import ImageLayerBinding
from napari_harpy.viewer.image_styling import DEFAULT_OVERLAY_COLORS, ImageDisplayMode
from napari_harpy.widgets.shared_styles import (
    ACTION_BUTTON_STYLESHEET,
    CHECKBOX_STYLESHEET,
    COMPLETER_POPUP_STYLESHEET,
    WIDGET_TEXT_MUTED_COLOR,
    WIDGET_WARNING_TEXT_COLOR,
    CompleterPopupLineEdit,
    build_input_control_stylesheet,
    format_tooltip,
)
from napari_harpy.widgets.viewer.disclosure import _ElidedLabel
from napari_harpy.widgets.viewer.styles import CARD_TITLE_STYLESHEET, DETAIL_PANEL_STYLESHEET, EMPTY_STATE_STYLESHEET

_CHANNEL_WARNING_STYLESHEET = f"color: {WIDGET_WARNING_TEXT_COLOR}; font-weight: 600;"
_CHANNEL_PANEL_STYLESHEET = "QWidget { background: transparent; }"
_SUBSECTION_LABEL_STYLESHEET = f"color: {WIDGET_TEXT_MUTED_COLOR}; font-size: 11px; font-weight: 600;"
_MAX_VISIBLE_OVERLAY_CHANNELS = 5


@dataclass(frozen=True)
class ImageLoadRequest:
    image_name: str
    mode: ImageDisplayMode
    channels: list[int]
    channel_colors: list[str]


class _OverlayChannelRow(QWidget):
    """Selected-only row for one live overlay channel binding."""

    remove_requested = Signal(int)

    def __init__(self, binding: ImageLayerBinding, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.binding = binding
        channel_index = _binding_channel_index(binding)
        channel_name = _binding_channel_name(binding)
        self.setObjectName(f"viewer_widget_selected_channel_row_{binding.element_name}_{channel_index}")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.channel_label = _ElidedLabel(channel_name, self)
        self.channel_label.setObjectName(
            f"viewer_widget_selected_channel_label_{binding.element_name}_{channel_index}"
        )

        self.remove_button = QPushButton("×")
        self.remove_button.setObjectName(
            f"viewer_widget_remove_channel_button_{binding.element_name}_{channel_index}"
        )
        self.remove_button.setAccessibleName(f"Remove channel {channel_name} from viewer")
        self.remove_button.setToolTip(format_tooltip(f"Remove channel {channel_name} from viewer"))
        self.remove_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.remove_button.setFixedWidth(28)
        self.remove_button.clicked.connect(
            lambda _checked=False, current_channel_index=channel_index: self.remove_requested.emit(
                current_channel_index
            )
        )

        layout.addWidget(self.channel_label, 1)
        layout.addWidget(self.remove_button)

    @property
    def channel_index(self) -> int:
        return _binding_channel_index(self.binding)

    @property
    def channel_name(self) -> str:
        return _binding_channel_name(self.binding)

    def set_binding(self, binding: ImageLayerBinding) -> None:
        """Refresh the live binding without recreating the row."""
        if _binding_channel_index(binding) != self.channel_index:
            raise ValueError("Cannot rebind an overlay row to a different channel index.")
        self.binding = binding
        channel_name = _binding_channel_name(binding)
        self.channel_label.setText(channel_name)
        self.remove_button.setAccessibleName(f"Remove channel {channel_name} from viewer")
        self.remove_button.setToolTip(format_tooltip(f"Remove channel {channel_name} from viewer"))

    def dispose(self) -> None:
        """Release row-owned resources before removal.

        Slice 3a extends this boundary with napari property-event cleanup.
        """


class _ImageCardWidget(QFrame):
    """Render one image card and emit image-layer user intent.

    The card owns available-channel search, selected-row presentation, and
    card-local color preferences. It does not call ``ViewerAdapter`` or mutate
    napari layers. Instead, its signals carry stack and focused overlay intent
    to ``ViewerWidget``:

    - ``add_update_requested`` carries an ``ImageLoadRequest`` for the
      transitional stack-only Add/Update action;
    - ``overlay_channel_add_requested`` carries image name, channel index, and
      requested initial color;
    - ``overlay_channel_remove_requested`` carries image name and channel
      index;
    - ``overlay_channels_remove_all_requested`` carries image name.

    ``ViewerWidget`` validates the active context and performs each adapter
    mutation. Completed membership changes return through
    ``ViewerAdapter.image_overlay_layers_changed``; the Viewer re-queries live
    bindings and calls ``set_loaded_overlay_bindings`` to reconcile this card.
    """

    add_update_requested = Signal(object)
    overlay_channel_add_requested = Signal(str, int, str)
    overlay_channel_remove_requested = Signal(str, int)
    overlay_channels_remove_all_requested = Signal(str)

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
        self._selected_rows_by_channel_index: dict[int, _OverlayChannelRow] = {}
        self._selected_channel_order: list[int] = []
        self._last_used_overlay_colors: dict[int, str] = {}
        self._membership_error: str | None = None
        self._channel_names_by_casefold: dict[str, list[tuple[int, str]]] = {}
        for channel_index, channel_name in enumerate(channel_names):
            self._channel_names_by_casefold.setdefault(channel_name.casefold(), []).append(
                (channel_index, channel_name)
            )

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

        self.channel_search_input = CompleterPopupLineEdit()
        self.channel_search_input.setObjectName(f"viewer_widget_channel_search_input_{image_name}")
        self.channel_search_input.setPlaceholderText("Search or add channels")
        self.channel_search_input.setStyleSheet(build_input_control_stylesheet("QLineEdit"))
        self.channel_search_input.setMinimumWidth(0)
        self.channel_search_input.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.channel_search_input.set_completion_popup_on_entry_enabled(True)

        self._channel_completer_model = QStringListModel(self.channel_search_input)
        self._channel_completer = QCompleter(self._channel_completer_model, self.channel_search_input)
        self._channel_completer.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
        self._channel_completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self._channel_completer.setFilterMode(Qt.MatchFlag.MatchContains)
        self._channel_completer.setMaxVisibleItems(10)
        self._channel_completer.popup().setStyleSheet(COMPLETER_POPUP_STYLESHEET)
        self.channel_search_input.setCompleter(self._channel_completer)
        channel_layout.addWidget(self.channel_search_input)

        selected_summary_layout = QHBoxLayout()
        selected_summary_layout.setContentsMargins(0, 0, 0, 0)
        selected_summary_layout.setSpacing(8)

        self.selected_count_label = QLabel("0 channels")
        self.selected_count_label.setObjectName(f"viewer_widget_selected_channel_count_{image_name}")
        self.selected_count_label.setStyleSheet(_SUBSECTION_LABEL_STYLESHEET)

        self.remove_all_channels_button = QPushButton("Remove all")
        self.remove_all_channels_button.setObjectName(
            f"viewer_widget_remove_all_channels_button_{image_name}"
        )
        self.remove_all_channels_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.remove_all_channels_button.setToolTip(format_tooltip("Remove all overlay channels for this image"))
        self.remove_all_channels_button.setVisible(False)

        selected_summary_layout.addWidget(self.selected_count_label)
        selected_summary_layout.addStretch(1)
        selected_summary_layout.addWidget(self.remove_all_channels_button)
        channel_layout.addLayout(selected_summary_layout)

        self.no_selected_channels_label = QLabel("No channels in viewer")
        self.no_selected_channels_label.setObjectName(
            f"viewer_widget_no_selected_channels_label_{image_name}"
        )
        self.no_selected_channels_label.setStyleSheet(EMPTY_STATE_STYLESHEET)
        channel_layout.addWidget(self.no_selected_channels_label)

        self.channel_scroll_area = QScrollArea()
        self.channel_scroll_area.setObjectName(f"viewer_widget_channel_scroll_area_{image_name}")
        self.channel_scroll_area.setWidgetResizable(True)
        self.channel_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.channel_scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.channel_scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.channel_scroll_area.setStyleSheet("QScrollArea { border: 0px; background: transparent; }")
        self.channel_scroll_area.hide()

        self.channel_list_widget = QWidget()
        self.channel_list_widget.setObjectName(f"viewer_widget_channel_list_{image_name}")
        self.channel_list_widget.setStyleSheet(_CHANNEL_PANEL_STYLESHEET)
        self.channel_list_layout = QVBoxLayout(self.channel_list_widget)
        self.channel_list_layout.setContentsMargins(0, 0, 0, 0)
        self.channel_list_layout.setSpacing(6)
        self.channel_scroll_area.setWidget(self.channel_list_widget)
        channel_layout.addWidget(self.channel_scroll_area)

        self.add_update_button = QPushButton("Add / Update in viewer")
        self.add_update_button.setObjectName(f"viewer_widget_add_update_image_button_{image_name}")
        self.add_update_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.add_update_button.setMinimumHeight(28)
        self.add_update_button.setStyleSheet(ACTION_BUTTON_STYLESHEET)
        self.add_update_button.setToolTip("")

        self.add_update_button.clicked.connect(self._emit_add_update_request)
        self.stack_toggle.toggled.connect(self._on_stack_toggled)
        self.overlay_toggle.toggled.connect(self._on_overlay_toggled)
        self._channel_completer.activated[str].connect(self._request_channel_from_text)
        self.channel_search_input.returnPressed.connect(self._request_channel_from_input)
        self.remove_all_channels_button.clicked.connect(self._emit_remove_all_requested)

        layout.addLayout(mode_layout)
        layout.addWidget(self.channel_warning_label)
        layout.addWidget(self.channel_panel)
        layout.addWidget(self.add_update_button)

        self._refresh_overlay_availability()
        self._refresh_search_model()
        self._refresh_membership_presentation()

    @property
    def selected_overlay_rows(self) -> list[_OverlayChannelRow]:
        """Return selected rows in their rendered napari order."""
        return [
            self._selected_rows_by_channel_index[channel_index]
            for channel_index in self._selected_channel_order
        ]

    @property
    def loaded_overlay_channel_indices(self) -> tuple[int, ...]:
        return tuple(self._selected_channel_order)

    @property
    def loaded_overlay_channel_names(self) -> tuple[str, ...]:
        return tuple(row.channel_name for row in self.selected_overlay_rows)

    @property
    def available_channel_names(self) -> tuple[str, ...]:
        return tuple(self._channel_completer_model.stringList())

    def set_loaded_overlay_bindings(self, bindings: Sequence[ImageLayerBinding]) -> None:
        """Reconcile selected-only rows against complete live overlay membership."""
        ordered_bindings: list[ImageLayerBinding] = []
        seen_channel_indices: set[int] = set()
        for binding in bindings:
            channel_index = _binding_channel_index(binding)
            _binding_channel_name(binding)
            if channel_index in seen_channel_indices:
                raise ValueError(
                    f"Found multiple live overlay bindings for image `{self.image_name}` "
                    f"and channel index {channel_index}."
                )
            seen_channel_indices.add(channel_index)
            ordered_bindings.append(binding)

        self.set_overlay_membership_error(None)
        next_channel_indices = [_binding_channel_index(binding) for binding in ordered_bindings]
        next_channel_index_set = set(next_channel_indices)

        for channel_index in tuple(self._selected_channel_order):
            if channel_index in next_channel_index_set:
                continue
            row = self._selected_rows_by_channel_index.pop(channel_index)
            self._remember_row_solid_color(row)
            row.dispose()
            self.channel_list_layout.removeWidget(row)
            row.deleteLater()

        for binding in ordered_bindings:
            channel_index = _binding_channel_index(binding)
            row = self._selected_rows_by_channel_index.get(channel_index)
            if row is None:
                row = _OverlayChannelRow(binding, self.channel_list_widget)
                row.remove_requested.connect(self._emit_overlay_channel_remove_requested)
                self._selected_rows_by_channel_index[channel_index] = row
            else:
                row.set_binding(binding)

        for row in self._selected_rows_by_channel_index.values():
            self.channel_list_layout.removeWidget(row)
        for channel_index in next_channel_indices:
            self.channel_list_layout.addWidget(self._selected_rows_by_channel_index[channel_index])

        self._selected_channel_order = next_channel_indices
        self._refresh_search_model()
        self._refresh_membership_presentation()

    def set_overlay_membership_error(self, message: str | None) -> None:
        """Set or clear a card-scoped binding invariant error."""
        self._membership_error = message
        self._refresh_overlay_availability()

    def finish_overlay_channel_add(self, channel_index: int, *, succeeded: bool) -> None:
        """Finish one synchronous add intent without clearing unrelated input."""
        if not succeeded:
            return

        self.channel_search_input.clear_after_accepted_completion(
            self.channel_names[channel_index]
        )

    def finish_overlay_channel_remove(self, channel_index: int, *, succeeded: bool) -> None:
        """Clear stale input after one channel was successfully removed."""
        if succeeded:
            self._clear_channel_input_if_matches(channel_index)

    def cache_overlay_channel_color(self, channel_index: int) -> None:
        """Cache the current live solid color for one selected row."""
        row = self._selected_rows_by_channel_index.get(channel_index)
        if row is not None:
            self._remember_row_solid_color(row)

    def cache_all_overlay_channel_colors(self) -> None:
        for row in self.selected_overlay_rows:
            self._remember_row_solid_color(row)

    def _refresh_overlay_availability(self) -> None:
        if self.channel_error is not None:
            warning = (
                "Overlay is unavailable because this image has duplicate channel names. "
                'Use "sdata.set_channel_names(...)" to rename them.'
            )
        elif not self.channel_names:
            warning = "No channel axis available for this image."
        else:
            warning = self._membership_error

        is_available = warning is None
        self.overlay_toggle.setEnabled(self.channel_error is None and bool(self.channel_names))
        self.channel_search_input.setEnabled(is_available)
        self.remove_all_channels_button.setEnabled(is_available and bool(self._selected_channel_order))
        for row in self._selected_rows_by_channel_index.values():
            row.setEnabled(is_available)

        self.channel_warning_label.setText(warning or "")
        self.channel_warning_label.setToolTip(format_tooltip(warning) if warning else "")
        self.channel_warning_label.setVisible(warning is not None)

    def _on_stack_toggled(self, checked: bool) -> None:
        if checked:
            with QSignalBlocker(self.overlay_toggle):
                self.overlay_toggle.setChecked(False)
            self.channel_panel.setVisible(False)
            self.add_update_button.setVisible(True)
            return

        if not self.overlay_toggle.isChecked():
            with QSignalBlocker(self.stack_toggle):
                self.stack_toggle.setChecked(True)

    def _on_overlay_toggled(self, checked: bool) -> None:
        if checked:
            with QSignalBlocker(self.stack_toggle):
                self.stack_toggle.setChecked(False)
            self.channel_panel.setVisible(True)
            self.add_update_button.setVisible(False)
            return

        self.channel_panel.setVisible(False)
        self.add_update_button.setVisible(True)
        if not self.stack_toggle.isChecked():
            with QSignalBlocker(self.stack_toggle):
                self.stack_toggle.setChecked(True)

    def display_mode(self) -> str:
        return "overlay" if self.overlay_toggle.isChecked() else "stack"

    def _emit_add_update_request(self, _checked: bool = False) -> None:
        if self.display_mode() != "stack":
            return
        self.add_update_requested.emit(
            ImageLoadRequest(
                image_name=self.image_name,
                mode="stack",
                channels=[],
                channel_colors=[],
            )
        )

    def _request_channel_from_input(self) -> None:
        self._request_channel_from_text(self.channel_search_input.text())

    def _request_channel_from_text(self, text: str) -> None:
        if not self.channel_search_input.isEnabled():
            return
        resolved = self._resolve_channel_text(text)
        if resolved is None:
            return
        channel_index, _channel_name = resolved
        if channel_index in self._selected_rows_by_channel_index:
            return
        self.overlay_channel_add_requested.emit(
            self.image_name,
            channel_index,
            self._initial_color_for_channel(channel_index),
        )

    def _resolve_channel_text(self, text: str) -> tuple[int, str] | None:
        normalized_text = text.strip()
        if not normalized_text:
            return None

        for channel_index, channel_name in enumerate(self.channel_names):
            if channel_name == normalized_text:
                return channel_index, channel_name

        matches = self._channel_names_by_casefold.get(normalized_text.casefold(), [])
        if len(matches) == 1:
            return matches[0]
        return None

    def _clear_channel_input_if_matches(self, channel_index: int) -> None:
        resolved = self._resolve_channel_text(self.channel_search_input.text())
        if resolved is not None and resolved[0] == channel_index:
            self.channel_search_input.clear()

    def _emit_overlay_channel_remove_requested(self, channel_index: int) -> None:
        self.overlay_channel_remove_requested.emit(self.image_name, channel_index)

    def _emit_remove_all_requested(self, _checked: bool = False) -> None:
        if not self._selected_channel_order:
            return
        self.overlay_channels_remove_all_requested.emit(self.image_name)

    def _refresh_search_model(self) -> None:
        selected_channel_indices = set(self._selected_channel_order)
        available_names = [
            channel_name
            for channel_index, channel_name in enumerate(self.channel_names)
            if channel_index not in selected_channel_indices
        ]
        self._channel_completer_model.setStringList(available_names)

    def _refresh_membership_presentation(self) -> None:
        selected_count = len(self._selected_channel_order)
        count_text = f"{selected_count} channel" if selected_count == 1 else f"{selected_count} channels"
        self.selected_count_label.setText(count_text)
        self.no_selected_channels_label.setVisible(selected_count == 0)
        self.channel_scroll_area.setVisible(selected_count > 0)
        self.remove_all_channels_button.setVisible(selected_count > 0)
        self._set_channel_scroll_height(self.selected_overlay_rows)
        self._refresh_overlay_availability()

    def _set_channel_scroll_height(self, channel_rows: list[QWidget]) -> None:
        visible_rows = channel_rows[:_MAX_VISIBLE_OVERLAY_CHANNELS]
        if not visible_rows:
            self.channel_scroll_area.setMaximumHeight(0)
            return

        visible_height = sum(row.sizeHint().height() for row in visible_rows)
        visible_height += self.channel_list_layout.spacing() * max(0, len(visible_rows) - 1)
        margins = self.channel_list_layout.contentsMargins()
        visible_height += margins.top() + margins.bottom()
        visible_height += self.channel_scroll_area.frameWidth() * 2
        self.channel_scroll_area.setMaximumHeight(visible_height)

    def _initial_color_for_channel(self, channel_index: int) -> str:
        cached_color = self._last_used_overlay_colors.get(channel_index)
        if cached_color is not None:
            return cached_color

        used_colors = {
            color
            for row in self.selected_overlay_rows
            if (color := _solid_color_from_layer(row.binding.layer)) is not None
        }
        for color in DEFAULT_OVERLAY_COLORS:
            normalized_color = _normalized_color_or_none(color)
            if normalized_color is not None and normalized_color not in used_colors:
                return normalized_color

        return DEFAULT_OVERLAY_COLORS[len(used_colors) % len(DEFAULT_OVERLAY_COLORS)]

    def _remember_row_solid_color(self, row: _OverlayChannelRow) -> None:
        color = _solid_color_from_layer(row.binding.layer)
        if color is not None:
            self._last_used_overlay_colors[row.channel_index] = color


def _binding_channel_index(binding: ImageLayerBinding) -> int:
    channel_index = binding.channel_index
    if not isinstance(channel_index, int) or channel_index < 0:
        raise ValueError("Overlay bindings require a non-negative channel index.")
    return channel_index


def _binding_channel_name(binding: ImageLayerBinding) -> str:
    channel_name = binding.channel_name
    if not isinstance(channel_name, str) or not channel_name:
        raise ValueError("Overlay bindings require a non-empty channel name.")
    return channel_name


def _solid_color_from_layer(layer: object) -> str | None:
    colormap = getattr(layer, "colormap", None)
    name = getattr(colormap, "name", None)
    if isinstance(name, str):
        return _normalized_color_or_none(name)
    if isinstance(colormap, str):
        return _normalized_color_or_none(colormap)
    return None


def _normalized_color_or_none(color: str) -> str | None:
    qcolor = QColor(color)
    if not qcolor.isValid():
        return None
    return qcolor.name(QColor.NameFormat.HexRgb).upper()
