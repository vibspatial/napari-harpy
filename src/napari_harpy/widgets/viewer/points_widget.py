from __future__ import annotations

from typing import TYPE_CHECKING

from qtpy.QtCore import QSignalBlocker, QStringListModel, Qt, Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QCompleter,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from napari_harpy._points_value_index import DEFAULT_RENDER_POINT_BUDGET
from napari_harpy.widgets.shared_styles import (
    ACTION_BUTTON_STYLESHEET,
    CHECKBOX_STYLESHEET,
    COMPLETER_POPUP_STYLESHEET,
    WIDGET_BORDER_COLOR,
    WIDGET_PANEL_COLOR,
    WIDGET_PANEL_SUBTLE_COLOR,
    WIDGET_TEXT_COLOR,
    WIDGET_TEXT_MUTED_COLOR,
    CompactComboBox,
    CompleterPopupLineEdit,
    StatusCardKind,
    build_input_control_stylesheet,
    create_form_label,
    set_status_card,
)

if TYPE_CHECKING:
    from napari_harpy.widgets.viewer.points_controller import PointsController, PointsValueSource

POINTS_RENDER_BUDGET_MIN = 1_000
POINTS_RENDER_BUDGET_MAX = 1_000_000
_DETAIL_PANEL_STYLESHEET = (
    "QFrame[harpyViewerDetailPanel='true'] {"
    f"background-color: {WIDGET_PANEL_COLOR}; "
    f"border: 1px solid {WIDGET_BORDER_COLOR}; "
    "border-radius: 8px;}"
)
_SELECTED_VALUES_SUMMARY_STYLESHEET = (
    "QLabel {"
    f"background-color: {WIDGET_PANEL_SUBTLE_COLOR}; "
    "border: 0px; "
    f"color: {WIDGET_TEXT_COLOR}; "
    "font-weight: 500; "
    "padding: 4px 6px;}"
)
_SELECTED_VALUES_EMPTY_STYLESHEET = (
    "QLabel {"
    f"background-color: {WIDGET_PANEL_SUBTLE_COLOR}; "
    "border: 0px; "
    f"color: {WIDGET_TEXT_MUTED_COLOR}; "
    "font-weight: 500; "
    "padding: 4px 6px;}"
)
_INLINE_ROW_STYLESHEET = "QWidget { background: transparent; }"


class PointsValueWidget(QFrame):
    """Qt controls for selecting points values to visualize.

    This widget intentionally owns only the points section UI: selectors,
    value-entry parsing, completer contents, render-budget validation, and
    status-card rendering. SpatialData validation, Dask computation, and napari
    layer application stay in the controller and viewer adapter.
    """

    source_changed = Signal()
    add_update_requested = Signal(object, int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._can_visualize = False
        self._available_values: tuple[str, ...] = ()
        self._available_values_by_casefold: dict[str, str] = {}
        self._selected_values: list[str] = []
        self._value_selection_warning: str | None = None
        self.setObjectName("viewer_widget_points_value_widget")
        self.setProperty("harpyViewerDetailPanel", True)
        self.setStyleSheet(_DETAIL_PANEL_STYLESHEET)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        form_layout = QFormLayout()
        form_layout.setContentsMargins(0, 0, 0, 0)
        form_layout.setHorizontalSpacing(8)
        form_layout.setVerticalSpacing(6)

        self.points_combo = CompactComboBox(minimum_contents_length=8)
        self.points_combo.setObjectName("viewer_widget_points_combo")
        self.points_combo.setStyleSheet(build_input_control_stylesheet("QComboBox"))

        self.index_column_combo = CompactComboBox(minimum_contents_length=8)
        self.index_column_combo.setObjectName("viewer_widget_points_index_column_combo")
        self.index_column_combo.setStyleSheet(build_input_control_stylesheet("QComboBox"))

        self.value_input = CompleterPopupLineEdit()
        self.value_input.setObjectName("viewer_widget_points_value_input")
        self.value_input.setStyleSheet(build_input_control_stylesheet("QLineEdit"))
        self.value_input.setPlaceholderText("Select value")

        self._value_completer_model = QStringListModel(self.value_input)
        self._value_completer = QCompleter(self._value_completer_model, self.value_input)
        self._value_completer.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
        self._value_completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self._value_completer.setFilterMode(Qt.MatchFlag.MatchContains)
        self._value_completer.setMaxVisibleItems(10)
        self._value_completer.popup().setStyleSheet(COMPLETER_POPUP_STYLESHEET)
        self.value_input.setCompleter(self._value_completer)

        self.add_value_button = QPushButton("Add")
        self.add_value_button.setObjectName("viewer_widget_points_add_value_button")
        self.add_value_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.add_value_button.setMinimumHeight(28)
        self.add_value_button.setStyleSheet(ACTION_BUTTON_STYLESHEET)

        value_search_widget = QWidget()
        value_search_widget.setStyleSheet(_INLINE_ROW_STYLESHEET)
        value_search_layout = QHBoxLayout(value_search_widget)
        value_search_layout.setContentsMargins(0, 0, 0, 0)
        value_search_layout.setSpacing(6)
        value_search_layout.addWidget(self.value_input, 1)
        value_search_layout.addWidget(self.add_value_button)

        self.selected_values_summary_label = QLabel("None")
        self.selected_values_summary_label.setObjectName("viewer_widget_points_selected_values_summary")
        self.selected_values_summary_label.setWordWrap(True)
        self.selected_values_summary_label.setStyleSheet(_SELECTED_VALUES_EMPTY_STYLESHEET)

        self.clear_selection_button = QPushButton("Clear")
        self.clear_selection_button.setObjectName("viewer_widget_points_clear_selection_button")
        self.clear_selection_button.setAccessibleName("Clear selected point values")
        self.clear_selection_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.clear_selection_button.setMinimumHeight(28)
        self.clear_selection_button.setStyleSheet(ACTION_BUTTON_STYLESHEET)
        self.clear_selection_button.setToolTip("Clear selected point values")

        selected_values_widget = QWidget()
        selected_values_widget.setStyleSheet(_INLINE_ROW_STYLESHEET)
        selected_values_layout = QHBoxLayout(selected_values_widget)
        selected_values_layout.setContentsMargins(0, 0, 0, 0)
        selected_values_layout.setSpacing(6)
        selected_values_layout.addWidget(self.selected_values_summary_label, 1)
        selected_values_layout.addWidget(self.clear_selection_button)

        self.all_values_checkbox = QCheckBox("All values")
        self.all_values_checkbox.setObjectName("viewer_widget_points_all_values_checkbox")
        self.all_values_checkbox.setStyleSheet(CHECKBOX_STYLESHEET)

        self.render_point_budget_input = QLineEdit(str(DEFAULT_RENDER_POINT_BUDGET))
        self.render_point_budget_input.setObjectName("viewer_widget_points_render_point_budget_input")
        self.render_point_budget_input.setStyleSheet(build_input_control_stylesheet("QLineEdit"))

        form_layout.addRow(create_form_label("Points"), self.points_combo)
        form_layout.addRow(create_form_label("Index column"), self.index_column_combo)
        form_layout.addRow(create_form_label("Values"), value_search_widget)
        form_layout.addRow(create_form_label("Selected"), selected_values_widget)
        form_layout.addRow(create_form_label("Render budget"), self.render_point_budget_input)
        form_layout.addRow("", self.all_values_checkbox)

        self.status_label = QLabel()
        self.status_label.setObjectName("viewer_widget_points_status")
        self.status_label.setWordWrap(True)

        self.add_update_button = QPushButton("Add / Update in viewer")
        self.add_update_button.setObjectName("viewer_widget_points_add_update_button")
        self.add_update_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.add_update_button.setMinimumHeight(28)
        self.add_update_button.setStyleSheet(ACTION_BUTTON_STYLESHEET)

        layout.addLayout(form_layout)
        layout.addWidget(self.status_label)
        layout.addWidget(self.add_update_button)

        self.points_combo.currentIndexChanged.connect(self._emit_source_changed)
        self.index_column_combo.currentIndexChanged.connect(self._emit_source_changed)
        self.add_value_button.clicked.connect(self._add_value_from_input)
        self.add_update_button.clicked.connect(self._emit_add_update_requested)
        self.all_values_checkbox.toggled.connect(self._refresh_value_input_state)
        self.value_input.textChanged.connect(self._refresh_add_update_state)
        self.value_input.returnPressed.connect(self._add_value_from_input)
        self.clear_selection_button.clicked.connect(self._clear_selected_values)
        self.render_point_budget_input.textChanged.connect(self._refresh_add_update_state)

        self.show_status(
            title="Points",
            lines=["Choose a points element and index column."],
            kind="warning",
        )
        self._refresh_enabled_state(can_visualize=False)

    def set_points_names(self, points_names: list[str]) -> None:
        """Replace available points element names."""
        current = self.selected_points_name()
        with QSignalBlocker(self.points_combo):
            self.points_combo.clear()
            self.points_combo.addItems(points_names)
            if current in points_names:
                self.points_combo.setCurrentIndex(points_names.index(current))
            elif points_names:
                self.points_combo.setCurrentIndex(0)
            else:
                self.points_combo.setCurrentIndex(-1)
        self.points_combo.setEnabled(bool(points_names))
        self._refresh_value_input_state()

    def set_index_columns(self, index_columns: list[str], *, preferred: str | None = "gene") -> None:
        """Replace available index columns for the selected points element."""
        current = self.selected_index_column()
        with QSignalBlocker(self.index_column_combo):
            self.index_column_combo.clear()
            self.index_column_combo.addItems(index_columns)
            if current in index_columns:
                self.index_column_combo.setCurrentIndex(index_columns.index(current))
            elif preferred in index_columns:
                self.index_column_combo.setCurrentIndex(index_columns.index(str(preferred)))
            elif index_columns:
                self.index_column_combo.setCurrentIndex(0)
            else:
                self.index_column_combo.setCurrentIndex(-1)
        self.index_column_combo.setEnabled(bool(index_columns))
        self._refresh_value_input_state()

    def set_value_source(self, value_source: PointsValueSource | None) -> None:
        """Update the value completer from a prepared value source."""
        if value_source is None:
            values: list[str] = []
            self._selected_values.clear()
            self._clear_value_selection_warning()
        else:
            values = list(dict.fromkeys(str(value) for value in value_source.value_table.values["value"]))
        self._available_values = tuple(values)
        self._available_values_by_casefold = {value.casefold(): value for value in self._available_values}
        self._value_completer_model.setStringList(values)
        self._drop_selected_values_not_in_available_values()
        self._render_selected_values_summary()
        self._refresh_value_input_state()

    def render_controller_state(self, controller: PointsController) -> None:
        """Render controller state into controls and the points status card."""
        self._refresh_enabled_state(
            can_visualize=controller.can_visualize,
            is_loading=controller.is_loading or controller.is_loading_values,
        )
        if self._value_selection_warning and controller.can_visualize and not (controller.is_loading or controller.is_loading_values):
            self.show_status(
                title="Points Warning",
                lines=[self._value_selection_warning],
                kind="warning",
            )
            return
        self.show_status(
            title="Points",
            lines=[controller.status_message],
            kind=controller.status_kind,
        )

    def selected_points_name(self) -> str | None:
        """Return the selected points element name, if any."""
        text = self.points_combo.currentText().strip()
        return text or None

    def selected_index_column(self) -> str | None:
        """Return the selected index column, if any."""
        text = self.index_column_combo.currentText().strip()
        return text or None

    def selected_values(self) -> tuple[str, ...] | str:
        """Return parsed value selection for the next add/update request."""
        if self.all_values_checkbox.isChecked():
            return "all"
        return tuple(self._selected_values)

    def render_point_budget(self) -> int | None:
        """Return the validated render point budget, or `None` if invalid."""
        text = self.render_point_budget_input.text().strip().replace(",", "").replace("_", "")
        try:
            value = int(text)
        except ValueError:
            return None
        if POINTS_RENDER_BUDGET_MIN <= value <= POINTS_RENDER_BUDGET_MAX:
            return value
        return None

    def show_status(
        self,
        *,
        title: str,
        lines: list[str],
        kind: StatusCardKind,
        tooltip_message: str | None = None,
    ) -> None:
        """Render the points section status card."""
        set_status_card(
            self.status_label,
            title=title,
            lines=lines,
            kind=kind,
            tooltip_message=tooltip_message,
        )

    def _emit_source_changed(self, _index: int | None = None) -> None:
        self.source_changed.emit()

    def _emit_add_update_requested(self, _checked: bool = False) -> None:
        budget = self.render_point_budget()
        if budget is None:
            self.show_status(
                title="Points Warning",
                lines=[
                    "Render point budget must be an integer between "
                    f"{POINTS_RENDER_BUDGET_MIN:,} and {POINTS_RENDER_BUDGET_MAX:,}."
                ],
                kind="warning",
            )
            return

        values = self.selected_values()
        if values != "all" and not values:
            self.show_status(
                title="Points Warning",
                lines=["Enter at least one value or enable All values."],
                kind="warning",
            )
            return

        self.add_update_requested.emit(values, budget)

    def _add_value_from_input(self, _checked: bool = False) -> None:
        if not self.value_input.isEnabled():
            return

        value = self._resolve_available_value(self.value_input.text())
        if value is None:
            typed_value = self.value_input.text().strip()
            if typed_value:
                self._set_value_selection_warning(f"`{typed_value}` is not in the loaded value table.")
            return

        self._add_selected_value(value)

    def _add_selected_value(self, value: str) -> None:
        if value in self._selected_values:
            self.value_input.clear()
            return

        self._selected_values.append(value)
        self.value_input.clear()
        self._clear_value_selection_warning()
        self._render_selected_values_summary()
        self._refresh_add_update_state()

    def _clear_selected_values(self) -> None:
        if not self._selected_values:
            return

        self._selected_values.clear()
        self._clear_value_selection_warning()
        self._render_selected_values_summary()
        self._refresh_add_update_state()

    def _drop_selected_values_not_in_available_values(self) -> None:
        if not self._selected_values:
            self._clear_value_selection_warning()
            return

        available = set(self._available_values)
        kept_values = [value for value in self._selected_values if value in available]
        dropped_count = len(self._selected_values) - len(kept_values)
        self._selected_values = kept_values
        if dropped_count:
            self._set_value_selection_warning(
                f"Dropped {dropped_count} selected value(s) that are no longer available."
            )
        else:
            self._clear_value_selection_warning()

    def _render_selected_values_summary(self) -> None:
        if self._selected_values:
            self.selected_values_summary_label.setText("\n".join(self._selected_values))
            self.selected_values_summary_label.setStyleSheet(_SELECTED_VALUES_SUMMARY_STYLESHEET)
        else:
            self.selected_values_summary_label.setText("None")
            self.selected_values_summary_label.setStyleSheet(_SELECTED_VALUES_EMPTY_STYLESHEET)
        self.clear_selection_button.setEnabled(bool(self._selected_values))

    def _resolve_available_value(self, value: str) -> str | None:
        normalized_value = value.strip()
        if not normalized_value:
            return None
        return self._available_values_by_casefold.get(normalized_value.casefold())

    def _set_value_selection_warning(self, message: str) -> None:
        self._value_selection_warning = message
        self.show_status(title="Points Warning", lines=[message], kind="warning")

    def _clear_value_selection_warning(self) -> None:
        self._value_selection_warning = None

    def _refresh_value_input_state(self, _checked: bool | None = None) -> None:
        value_selection_enabled = (
            self._can_visualize
            and self.points_combo.isEnabled()
            and self.index_column_combo.isEnabled()
            and bool(self._value_completer_model.stringList())
            and not self.all_values_checkbox.isChecked()
        )
        self.value_input.setEnabled(value_selection_enabled)
        self.value_input.set_completion_popup_on_entry_enabled(value_selection_enabled)
        self.add_value_button.setEnabled(
            value_selection_enabled and self._resolve_available_value(self.value_input.text()) is not None
        )
        self.clear_selection_button.setEnabled(
            value_selection_enabled and bool(self._selected_values)
        )
        self._refresh_add_update_state()

    def _refresh_add_update_state(self, _text: str | None = None) -> None:
        budget_is_valid = self.render_point_budget() is not None
        if self._can_visualize and not budget_is_valid:
            self.show_status(
                title="Points Warning",
                lines=[
                    "Render point budget must be an integer between "
                    f"{POINTS_RENDER_BUDGET_MIN:,} and {POINTS_RENDER_BUDGET_MAX:,}."
                ],
                kind="warning",
            )
        value_can_be_added = self._resolve_available_value(self.value_input.text()) is not None
        self.add_value_button.setEnabled(self.value_input.isEnabled() and value_can_be_added)
        self.add_update_button.setEnabled(
            self._can_visualize
            and budget_is_valid
            and (self.all_values_checkbox.isChecked() or bool(self.selected_values()))
        )

    def _refresh_enabled_state(
        self,
        *,
        can_visualize: bool,
        is_loading: bool = False,
    ) -> None:
        self.points_combo.setEnabled(not is_loading and self.points_combo.count() > 0)
        self.index_column_combo.setEnabled(not is_loading and self.index_column_combo.count() > 0)
        self.all_values_checkbox.setEnabled(can_visualize and not is_loading)
        self.render_point_budget_input.setEnabled(can_visualize and not is_loading)
        self._can_visualize = can_visualize and not is_loading
        self._refresh_value_input_state()
