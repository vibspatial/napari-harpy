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

from napari_harpy.core._color_source import ShapeColorSourceKind, ShapeColorSourceSpec
from napari_harpy.widgets.shared_styles import (
    ACTION_BUTTON_STYLESHEET,
    CHECKBOX_STYLESHEET,
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
    selected_source_kind: ShapeColorSourceKind | None
    selected_color_source: ShapeColorSourceSpec | None
    fill_shapes: bool


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

        color_source_kind_label = create_form_label("Color source")
        self.color_source_kind_combo = CompactComboBox(minimum_contents_length=8)
        self.color_source_kind_combo.setObjectName(f"viewer_widget_shapes_color_source_kind_combo_{shapes_name}")
        self.color_source_kind_combo.setStyleSheet(INPUT_CONTROL_STYLESHEET)
        self.color_source_kind_combo.addItem("None", None)
        self.color_source_kind_combo.addItem("Shapes column", "shape_column")

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

        form_layout.addRow(color_source_kind_label, self.color_source_kind_combo)
        form_layout.addRow(self.color_source_value_label, self.color_source_value_input)
        form_layout.addRow(create_form_label("Display"), self.fill_toggle)

        layout.addLayout(form_layout)
        layout.addWidget(self.action_hint_label)
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
            self.action_hint_label.setText("Action: add/update primary shapes layer")
            return

        if selected_source is None:
            if self._filtered_color_sources:
                self.action_hint_label.setText("Action: select a shapes column for a styled shapes layer")
            else:
                self.action_hint_label.setText("Action: no colorable shapes columns available")
            return

        self.action_hint_label.setText(
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
