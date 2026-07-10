from __future__ import annotations

from dataclasses import dataclass

from qtpy.QtCore import QSignalBlocker, QStringListModel, Qt, Signal
from qtpy.QtWidgets import (
    QCompleter,
    QFormLayout,
    QFrame,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
)

from napari_harpy.core._color_source import TableColorSourceKind, TableColorSourceSpec
from napari_harpy.widgets.shared_styles import (
    ACTION_BUTTON_STYLESHEET,
    COMPLETER_POPUP_STYLESHEET,
    CompactComboBox,
    CompleterPopupLineEdit,
    build_input_control_stylesheet,
    create_form_label,
)
from napari_harpy.widgets.viewer.disclosure import _ElidedLabel
from napari_harpy.widgets.viewer.styles import (
    CARD_TITLE_STYLESHEET,
    DETAIL_PANEL_STYLESHEET,
    INPUT_CONTROL_STYLESHEET,
    SUMMARY_LABEL_STYLESHEET,
)

_OBS_SOURCE_PLACEHOLDER = "Select obs column"
_VAR_SOURCE_PLACEHOLDER = "Select var"
_NO_COLOR_SOURCE_LABEL = "No color source"
_TABLE_SOURCE_KIND_ITEMS: tuple[tuple[str, TableColorSourceKind], ...] = (
    ("Observations", "obs_column"),
    ("Vars", "x_var"),
)


@dataclass(frozen=True)
class LabelsLoadRequest:
    labels_name: str
    table_name: str | None
    selected_source_kind: TableColorSourceKind | None
    selected_color_source: TableColorSourceSpec | None


class _LabelsCardWidget(QFrame):
    """Card UI for one labels element in the selected coordinate system."""

    add_update_requested = Signal(object)

    def __init__(
        self,
        *,
        labels_name: str,
        table_color_sources_by_table: dict[str, list[TableColorSourceSpec]],
    ) -> None:
        super().__init__()
        self.labels_name = labels_name
        self._table_color_sources_by_table = table_color_sources_by_table
        self._filtered_color_sources: list[TableColorSourceSpec] = []
        self._active_source_kind: TableColorSourceKind | None = None
        self.setObjectName(f"viewer_widget_labels_card_{labels_name}")
        self.setProperty("harpyViewerDetailPanel", True)
        self.setStyleSheet(DETAIL_PANEL_STYLESHEET)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        self.title_label = _ElidedLabel(labels_name, self)
        self.title_label.setObjectName(f"viewer_widget_labels_card_title_{labels_name}")
        self.title_label.setStyleSheet(CARD_TITLE_STYLESHEET)
        self.title_label.hide()

        form_layout = QFormLayout()
        form_layout.setContentsMargins(0, 0, 0, 0)
        form_layout.setHorizontalSpacing(8)
        form_layout.setVerticalSpacing(6)

        linked_table_label = create_form_label("Linked table")
        self.linked_table_combo = CompactComboBox(minimum_contents_length=8)
        self.linked_table_combo.setObjectName(f"viewer_widget_linked_table_combo_{labels_name}")
        self.linked_table_combo.setStyleSheet(INPUT_CONTROL_STYLESHEET)
        table_names = list(table_color_sources_by_table)
        if table_names:
            self.linked_table_combo.addItems(table_names)
        else:
            self.linked_table_combo.addItem("No linked tables")
            self.linked_table_combo.setEnabled(False)

        color_source_kind_label = create_form_label("Color source")
        self.color_source_kind_combo = CompactComboBox(minimum_contents_length=8)
        self.color_source_kind_combo.setObjectName(f"viewer_widget_color_source_kind_combo_{labels_name}")
        self.color_source_kind_combo.setStyleSheet(INPUT_CONTROL_STYLESHEET)
        self.color_source_kind_combo.addItem(_NO_COLOR_SOURCE_LABEL, None)

        self.color_source_value_label = create_form_label("Value source")
        self.color_source_value_input = CompleterPopupLineEdit()
        self.color_source_value_input.setObjectName(f"viewer_widget_color_source_value_input_{labels_name}")
        self.color_source_value_input.setStyleSheet(build_input_control_stylesheet("QLineEdit"))
        self.color_source_value_input.setMinimumWidth(0)
        self.color_source_value_input.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.color_source_value_input.setEnabled(False)

        self._color_source_completer_model = QStringListModel(self.color_source_value_input)
        self._color_source_completer = QCompleter(self._color_source_completer_model, self.color_source_value_input)
        self._color_source_completer.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
        self._color_source_completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self._color_source_completer.setFilterMode(Qt.MatchFlag.MatchContains)
        self._color_source_completer.setMaxVisibleItems(10)
        self._color_source_completer.popup().setStyleSheet(COMPLETER_POPUP_STYLESHEET)
        self.color_source_value_input.setCompleter(self._color_source_completer)

        self.action_hint_label = QLabel()
        self.action_hint_label.setObjectName(f"viewer_widget_action_hint_{labels_name}")
        self.action_hint_label.setWordWrap(True)
        self.action_hint_label.setStyleSheet(SUMMARY_LABEL_STYLESHEET)

        self.add_update_button = QPushButton("Add / Update in viewer")
        self.add_update_button.setObjectName(f"viewer_widget_add_update_labels_button_{labels_name}")
        self.add_update_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.add_update_button.setMinimumHeight(28)
        self.add_update_button.setStyleSheet(ACTION_BUTTON_STYLESHEET)
        self.add_update_button.clicked.connect(self._emit_add_update_request)

        form_layout.addRow(linked_table_label, self.linked_table_combo)
        form_layout.addRow(color_source_kind_label, self.color_source_kind_combo)
        form_layout.addRow(self.color_source_value_label, self.color_source_value_input)

        layout.addLayout(form_layout)
        layout.addWidget(self.action_hint_label)
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
        return value if value in self._available_source_kinds() else None

    @property
    def selected_color_source(self) -> TableColorSourceSpec | None:
        current_text = self.color_source_value_input.text().strip()
        for source in self._filtered_color_sources:
            if source.display_name == current_text:
                return source
        return None

    def set_linked_tables(
        self,
        table_color_sources_by_table: dict[str, list[TableColorSourceSpec]],
    ) -> None:
        previous_table_name = self.selected_table_name
        previous_source_kind = self.selected_source_kind
        previous_source_identity = (
            self.selected_color_source.identity if self.selected_color_source is not None else None
        )

        self._table_color_sources_by_table = table_color_sources_by_table
        table_names = list(table_color_sources_by_table)
        with QSignalBlocker(self.linked_table_combo):
            # Rebuild linked-table choices after the sdata table set changed,
            # then restore the previous selection when it is still valid.
            self.linked_table_combo.clear()
            if table_names:
                self.linked_table_combo.addItems(table_names)
                self.linked_table_combo.setEnabled(True)
                next_index = table_names.index(previous_table_name) if previous_table_name in table_names else 0
                self.linked_table_combo.setCurrentIndex(next_index)
            else:
                self.linked_table_combo.addItem("No linked tables")
                self.linked_table_combo.setEnabled(False)
                self.linked_table_combo.setCurrentIndex(0)

        self._refresh_color_source_controls(
            preferred_source_kind=previous_source_kind,
            preferred_source_identity=previous_source_identity,
        )

    def _refresh_color_source_controls(
        self,
        _index: int | None = None,
        *,
        preferred_source_kind: TableColorSourceKind | None = None,
        preferred_source_identity: tuple[str, TableColorSourceKind, str] | None = None,
    ) -> None:
        current_source_kind = preferred_source_kind or self.selected_source_kind
        if preferred_source_identity is not None:
            selected_source_identity = preferred_source_identity
        elif self._active_source_kind == current_source_kind:
            selected_source = self.selected_color_source
            selected_source_identity = selected_source.identity if selected_source is not None else None
        else:
            selected_source_identity = None

        self._rebuild_color_source_kind_combo(preferred_source_kind=current_source_kind)
        source_kind = self.selected_source_kind
        table_name = self.selected_table_name

        if source_kind == "obs_column":
            self.color_source_value_label.setText("Observation")
            placeholder_text = _OBS_SOURCE_PLACEHOLDER
        elif source_kind == "x_var":
            self.color_source_value_label.setText("Var")
            placeholder_text = _VAR_SOURCE_PLACEHOLDER
        else:
            self.color_source_value_label.setText("Value source")
            placeholder_text = "Select a color source kind first"

        available_sources = (
            list(self._table_color_sources_by_table.get(table_name, ())) if table_name is not None else []
        )
        self._filtered_color_sources = (
            [] if source_kind is None else [source for source in available_sources if source.source_kind == source_kind]
        )

        with QSignalBlocker(self.color_source_value_input):
            if source_kind is None:
                self.color_source_value_input.setEnabled(False)
                self.color_source_value_input.clear()
                self.color_source_value_input.setPlaceholderText(placeholder_text)
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
                    else:
                        self.color_source_value_input.clear()
                else:
                    self.color_source_value_input.clear()

                self.color_source_value_input.setPlaceholderText(placeholder_text)

            self._color_source_completer_model.setStringList(
                [source.display_name for source in self._filtered_color_sources]
            )

        self.color_source_value_input.set_completion_popup_on_entry_enabled(source_kind is not None)
        self._active_source_kind = source_kind
        self._update_action_status()

    def _available_source_kinds(self) -> tuple[TableColorSourceKind, ...]:
        table_name = self.selected_table_name
        if table_name is None:
            return ()

        source_kinds = {source.source_kind for source in self._table_color_sources_by_table.get(table_name, ())}
        return tuple(source_kind for _, source_kind in _TABLE_SOURCE_KIND_ITEMS if source_kind in source_kinds)

    def _rebuild_color_source_kind_combo(
        self,
        *,
        preferred_source_kind: TableColorSourceKind | None = None,
    ) -> None:
        available_source_kinds = self._available_source_kinds()
        selected_source_kind = preferred_source_kind if preferred_source_kind in available_source_kinds else None

        with QSignalBlocker(self.color_source_kind_combo):
            self.color_source_kind_combo.clear()
            self.color_source_kind_combo.addItem(_NO_COLOR_SOURCE_LABEL, None)
            for label, source_kind in _TABLE_SOURCE_KIND_ITEMS:
                if source_kind in available_source_kinds:
                    self.color_source_kind_combo.addItem(label, source_kind)

            next_index = self.color_source_kind_combo.findData(selected_source_kind)
            self.color_source_kind_combo.setCurrentIndex(next_index if next_index >= 0 else 0)
            self.color_source_kind_combo.setEnabled(bool(available_source_kinds))

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
            self.action_hint_label.setText("Action: add/update primary labels layer")
            return

        if table_name is None:
            self.action_hint_label.setText("Action: colored overlays require a linked table")
            return

        if source_kind == "obs_column":
            if selected_source is None:
                if self._filtered_color_sources:
                    self.action_hint_label.setText("Action: select an observation column for a colored overlay")
                else:
                    self.action_hint_label.setText("Action: no colorable observation columns available")
                return
            self.action_hint_label.setText(f'Action: add/update colored overlay for obs["{selected_source.value_key}"]')
            return

        if selected_source is None:
            if self._filtered_color_sources:
                self.action_hint_label.setText("Action: select a var for a colored overlay")
            else:
                self.action_hint_label.setText("Action: no vars available for a colored overlay")
            return

        self.action_hint_label.setText(f'Action: add/update colored overlay for X[:, "{selected_source.value_key}"]')

    def _emit_add_update_request(self, _checked: bool = False) -> None:
        self.add_update_requested.emit(
            LabelsLoadRequest(
                labels_name=self.labels_name,
                table_name=self.selected_table_name,
                selected_source_kind=self.selected_source_kind,
                selected_color_source=self.selected_color_source,
            )
        )
