from __future__ import annotations

from dataclasses import dataclass

from qtpy.QtCore import QSignalBlocker, QStringListModel, Qt, Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QCompleter,
    QFormLayout,
    QFrame,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
)

from napari_harpy.core._color_source import (
    ShapeColorSourceKind,
    ShapeColumnColorSourceSpec,
    TableColorSourceKind,
    TableColorSourceSpec,
)
from napari_harpy.widgets.shared_styles import (
    ACTION_BUTTON_STYLESHEET,
    CHECKBOX_STYLESHEET,
    COMPLETER_POPUP_STYLESHEET,
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
class ShapesLoadRequest:
    shapes_name: str
    table_name: str | None
    selected_source_kind: ShapeColorSourceKind | TableColorSourceKind | None
    selected_color_source: ShapeColumnColorSourceSpec | TableColorSourceSpec | None
    fill_shapes: bool


class _ShapesCardWidget(QFrame):
    """Card UI shell for one shapes element in the selected coordinate system."""

    add_update_requested = Signal(object)

    def __init__(
        self,
        *,
        shapes_name: str,
        shape_column_color_sources: list[ShapeColumnColorSourceSpec],
        table_color_sources_by_table: dict[str, list[TableColorSourceSpec]],
    ) -> None:
        super().__init__()
        self.shapes_name = shapes_name
        self._shape_column_color_sources = shape_column_color_sources
        self._table_color_sources_by_table = table_color_sources_by_table
        self._filtered_color_sources: list[ShapeColumnColorSourceSpec | TableColorSourceSpec] = []
        self.setObjectName(f"viewer_widget_shapes_card_{shapes_name}")
        self.setProperty("harpyViewerDetailPanel", True)
        self.setStyleSheet(DETAIL_PANEL_STYLESHEET)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        self.title_label = _ElidedLabel(shapes_name, self)
        self.title_label.setObjectName(f"viewer_widget_shapes_card_title_{shapes_name}")
        self.title_label.setStyleSheet(CARD_TITLE_STYLESHEET)
        self.title_label.hide()

        form_layout = QFormLayout()
        form_layout.setContentsMargins(0, 0, 0, 0)
        form_layout.setHorizontalSpacing(8)
        form_layout.setVerticalSpacing(6)

        linked_table_label = create_form_label("Linked table")
        self.linked_table_combo = CompactComboBox(minimum_contents_length=8)
        self.linked_table_combo.setObjectName(f"viewer_widget_shapes_linked_table_combo_{shapes_name}")
        self.linked_table_combo.setStyleSheet(INPUT_CONTROL_STYLESHEET)
        table_names = list(table_color_sources_by_table)
        if table_names:
            self.linked_table_combo.addItems(table_names)
        else:
            self.linked_table_combo.addItem("No linked tables")
            self.linked_table_combo.setEnabled(False)

        color_source_kind_label = create_form_label("Color source")
        self.color_source_kind_combo = CompactComboBox(minimum_contents_length=8)
        self.color_source_kind_combo.setObjectName(f"viewer_widget_shapes_color_source_kind_combo_{shapes_name}")
        self.color_source_kind_combo.setStyleSheet(INPUT_CONTROL_STYLESHEET)
        self.color_source_kind_combo.addItem("None", None)
        self.color_source_kind_combo.addItem("Shapes column", "shape_column")
        self.color_source_kind_combo.addItem("Observations", "obs_column")
        self.color_source_kind_combo.addItem("Vars", "x_var")

        self.color_source_value_label = create_form_label("Shapes column")
        self.color_source_value_input = QLineEdit()
        self.color_source_value_input.setObjectName(f"viewer_widget_shapes_color_source_value_input_{shapes_name}")
        self.color_source_value_input.setStyleSheet(build_input_control_stylesheet("QLineEdit"))
        self.color_source_value_input.setMinimumWidth(0)
        self.color_source_value_input.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.color_source_value_input.setEnabled(False)

        self.fill_toggle = QCheckBox("Fill")
        self.fill_toggle.setObjectName(f"viewer_widget_shapes_fill_toggle_{shapes_name}")
        self.fill_toggle.setStyleSheet(CHECKBOX_STYLESHEET)
        self.fill_toggle.setChecked(False)

        self._color_source_completer_model = QStringListModel(self.color_source_value_input)
        self._color_source_completer = QCompleter(self._color_source_completer_model, self.color_source_value_input)
        self._color_source_completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self._color_source_completer.setFilterMode(Qt.MatchFlag.MatchContains)
        self._color_source_completer.popup().setStyleSheet(COMPLETER_POPUP_STYLESHEET)
        self.color_source_value_input.setCompleter(self._color_source_completer)

        self.action_hint_label = QLabel()
        self.action_hint_label.setObjectName(f"viewer_widget_shapes_action_hint_{shapes_name}")
        self.action_hint_label.setWordWrap(True)
        self.action_hint_label.setStyleSheet(SUMMARY_LABEL_STYLESHEET)

        self.add_update_button = QPushButton("Add / Update in viewer")
        self.add_update_button.setObjectName(f"viewer_widget_add_update_shapes_button_{shapes_name}")
        self.add_update_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.add_update_button.setMinimumHeight(28)
        self.add_update_button.setStyleSheet(ACTION_BUTTON_STYLESHEET)
        self.add_update_button.clicked.connect(self._emit_add_update_request)
        self.add_update_button.setToolTip("")

        form_layout.addRow(linked_table_label, self.linked_table_combo)
        form_layout.addRow(color_source_kind_label, self.color_source_kind_combo)
        form_layout.addRow(self.color_source_value_label, self.color_source_value_input)
        form_layout.addRow(create_form_label("Display"), self.fill_toggle)

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
    def selected_source_kind(self) -> ShapeColorSourceKind | TableColorSourceKind | None:
        value = self.color_source_kind_combo.currentData()
        return value if value in {"shape_column", "obs_column", "x_var"} else None

    @property
    def selected_color_source(self) -> ShapeColumnColorSourceSpec | TableColorSourceSpec | None:
        current_text = self.color_source_value_input.text().strip()
        for source in self._filtered_color_sources:
            if source.display_name == current_text:
                return source
        return None

    @property
    def fill_shapes(self) -> bool:
        return self.fill_toggle.isChecked()

    def set_linked_tables(
        self,
        table_color_sources_by_table: dict[str, list[TableColorSourceSpec]],
    ) -> None:
        previous_table_name = self.selected_table_name
        previous_source_identity = (
            self.selected_color_source.identity if self.selected_color_source is not None else None
        )

        self._table_color_sources_by_table = table_color_sources_by_table
        table_names = list(table_color_sources_by_table)
        with QSignalBlocker(self.linked_table_combo):
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

        self._refresh_color_source_controls(preferred_source_identity=previous_source_identity)

    def _refresh_color_source_controls(
        self,
        _index: int | None = None,
        *,
        preferred_source_identity: tuple[object, ...] | None = None,
    ) -> None:
        if preferred_source_identity is not None:
            selected_source_identity = preferred_source_identity
        else:
            selected_source_identity = (
                self.selected_color_source.identity if self.selected_color_source is not None else None
            )
        source_kind = self.selected_source_kind

        if source_kind == "shape_column":
            self.color_source_value_label.setText("Shapes column")
            available_sources: list[ShapeColumnColorSourceSpec | TableColorSourceSpec] = list(
                self._shape_column_color_sources
            )
            placeholder_text = "Search shapes columns"
        elif source_kind == "obs_column":
            self.color_source_value_label.setText("Observation")
            table_name = self.selected_table_name
            available_sources = (
                list(self._table_color_sources_by_table.get(table_name, ())) if table_name is not None else []
            )
            placeholder_text = "Search observations"
        elif source_kind == "x_var":
            self.color_source_value_label.setText("Var")
            table_name = self.selected_table_name
            available_sources = (
                list(self._table_color_sources_by_table.get(table_name, ())) if table_name is not None else []
            )
            placeholder_text = "Search vars"
        else:
            self.color_source_value_label.setText("Value source")
            available_sources = []
            placeholder_text = "Select a color source kind first"

        self._filtered_color_sources = [
            source for source in available_sources if source_kind is None or source.source_kind == source_kind
        ]

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
                    elif self._filtered_color_sources:
                        self.color_source_value_input.setText(self._filtered_color_sources[0].display_name)
                    else:
                        self.color_source_value_input.clear()
                elif self._filtered_color_sources:
                    self.color_source_value_input.setText(self._filtered_color_sources[0].display_name)
                else:
                    self.color_source_value_input.clear()

                self.color_source_value_input.setPlaceholderText(placeholder_text)

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
        self._update_fill_toggle_enabled(selected_source is not None)

        if source_kind is None:
            self.action_hint_label.setText("Action: add/update primary shapes layer")
            return

        if source_kind in {"obs_column", "x_var"} and table_name is None:
            self.action_hint_label.setText("Action: table-backed shapes coloring requires a linked table")
            return

        if selected_source is None:
            if source_kind == "shape_column" and self._filtered_color_sources:
                self.action_hint_label.setText("Action: select a shapes column for a styled shapes layer")
            elif source_kind == "shape_column":
                self.action_hint_label.setText("Action: no colorable shapes columns available")
            elif source_kind == "obs_column" and self._filtered_color_sources:
                self.action_hint_label.setText("Action: select an observation column for a styled shapes layer")
            elif source_kind == "obs_column":
                self.action_hint_label.setText("Action: no colorable observation columns available")
            elif self._filtered_color_sources:
                self.action_hint_label.setText("Action: select a var for a styled shapes layer")
            else:
                self.action_hint_label.setText("Action: no vars available for a styled shapes layer")
            return

        if selected_source.source_kind == "shape_column":
            self.action_hint_label.setText(
                f'Action: add/update styled shapes layer for column "{selected_source.value_key}"'
            )
        elif selected_source.source_kind == "obs_column":
            self.action_hint_label.setText(
                f'Action: add/update styled shapes layer for obs["{selected_source.value_key}"]'
            )
        else:
            self.action_hint_label.setText(
                f'Action: add/update styled shapes layer for X[:, "{selected_source.value_key}"]'
            )

    def _update_fill_toggle_enabled(self, enabled: bool) -> None:
        self.fill_toggle.setEnabled(enabled)
        if not enabled:
            self.fill_toggle.setChecked(False)

    def _emit_add_update_request(self, _checked: bool = False) -> None:
        self.add_update_requested.emit(
            ShapesLoadRequest(
                shapes_name=self.shapes_name,
                table_name=self.selected_table_name,
                selected_source_kind=self.selected_source_kind,
                selected_color_source=self.selected_color_source,
                fill_shapes=self.fill_shapes,
            )
        )
