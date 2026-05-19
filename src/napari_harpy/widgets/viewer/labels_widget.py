from __future__ import annotations

from dataclasses import dataclass

from qtpy.QtCore import QSignalBlocker, QStringListModel, Qt, Signal
from qtpy.QtWidgets import (
    QCompleter,
    QFormLayout,
    QFrame,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
)

from napari_harpy.core._color_source import TableColorSourceKind, TableColorSourceSpec
from napari_harpy.widgets.shared_styles import (
    ACTION_BUTTON_STYLESHEET,
    CompactComboBox,
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
        table_names: list[str],
        table_color_sources_by_table: dict[str, list[TableColorSourceSpec]],
    ) -> None:
        super().__init__()
        self.labels_name = labels_name
        self._table_color_sources_by_table = table_color_sources_by_table
        self._filtered_color_sources: list[TableColorSourceSpec] = []
        self.setObjectName(f"viewer_widget_labels_card_{labels_name}")
        self.setProperty("harpyViewerDetailPanel", True)
        self.setStyleSheet(DETAIL_PANEL_STYLESHEET)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
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
        self.linked_table_combo = CompactComboBox()
        self.linked_table_combo.setObjectName(f"viewer_widget_linked_table_combo_{labels_name}")
        self.linked_table_combo.setStyleSheet(INPUT_CONTROL_STYLESHEET)
        if table_names:
            self.linked_table_combo.addItems(table_names)
        else:
            self.linked_table_combo.addItem("No linked tables")
            self.linked_table_combo.setEnabled(False)

        color_source_kind_label = create_form_label("Color source")
        self.color_source_kind_combo = CompactComboBox()
        self.color_source_kind_combo.setObjectName(f"viewer_widget_color_source_kind_combo_{labels_name}")
        self.color_source_kind_combo.setStyleSheet(INPUT_CONTROL_STYLESHEET)
        self.color_source_kind_combo.addItem("None", None)
        self.color_source_kind_combo.addItem("Observations", "obs_column")
        self.color_source_kind_combo.addItem("Vars", "x_var")

        self.color_source_value_label = create_form_label("Value source")
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
