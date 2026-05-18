from __future__ import annotations

from typing import TYPE_CHECKING

from qtpy.QtCore import QSignalBlocker, QStringListModel, Qt, Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QCompleter,
    QFormLayout,
    QFrame,
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
    WIDGET_BORDER_COLOR,
    WIDGET_PANEL_COLOR,
    CompactComboBox,
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


class PointsValueWidget(QFrame):
    """Qt controls for selecting points values to visualize.

    This widget intentionally owns only the points section UI: selectors,
    value-entry parsing, completer contents, render-budget validation, and
    status-card rendering. SpatialData validation, Dask computation, and napari
    layer application stay in the controller and viewer adapter.
    """

    source_changed = Signal()
    visualize_requested = Signal(object, int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._can_visualize = False
        self.setObjectName("viewer_widget_points_value_widget")
        self.setProperty("harpyViewerDetailPanel", True)
        self.setStyleSheet(_DETAIL_PANEL_STYLESHEET)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        form_layout = QFormLayout()
        form_layout.setContentsMargins(0, 0, 0, 0)
        form_layout.setHorizontalSpacing(8)
        form_layout.setVerticalSpacing(6)

        self.points_combo = CompactComboBox()
        self.points_combo.setObjectName("viewer_widget_points_combo")
        self.points_combo.setStyleSheet(build_input_control_stylesheet("QComboBox"))

        self.index_column_combo = CompactComboBox()
        self.index_column_combo.setObjectName("viewer_widget_points_index_column_combo")
        self.index_column_combo.setStyleSheet(build_input_control_stylesheet("QComboBox"))

        self.value_input = QLineEdit()
        self.value_input.setObjectName("viewer_widget_points_value_input")
        self.value_input.setStyleSheet(build_input_control_stylesheet("QLineEdit"))
        self.value_input.setPlaceholderText("AAMP, AXL, MALAT1")

        self._value_completer_model = QStringListModel(self.value_input)
        self._value_completer = QCompleter(self._value_completer_model, self.value_input)
        self._value_completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self._value_completer.setFilterMode(Qt.MatchFlag.MatchContains)
        self.value_input.setCompleter(self._value_completer)

        self.all_values_checkbox = QCheckBox("All values")
        self.all_values_checkbox.setObjectName("viewer_widget_points_all_values_checkbox")
        self.all_values_checkbox.setStyleSheet(CHECKBOX_STYLESHEET)

        self.render_point_budget_input = QLineEdit(str(DEFAULT_RENDER_POINT_BUDGET))
        self.render_point_budget_input.setObjectName("viewer_widget_points_render_point_budget_input")
        self.render_point_budget_input.setStyleSheet(build_input_control_stylesheet("QLineEdit"))

        form_layout.addRow(create_form_label("Points"), self.points_combo)
        form_layout.addRow(create_form_label("Index column"), self.index_column_combo)
        form_layout.addRow(create_form_label("Values"), self.value_input)
        form_layout.addRow(create_form_label("Render budget"), self.render_point_budget_input)
        form_layout.addRow("", self.all_values_checkbox)

        self.status_label = QLabel()
        self.status_label.setObjectName("viewer_widget_points_status")
        self.status_label.setWordWrap(True)

        self.visualize_button = QPushButton("Visualize")
        self.visualize_button.setObjectName("viewer_widget_points_visualize_button")
        self.visualize_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.visualize_button.setMinimumHeight(28)
        self.visualize_button.setStyleSheet(ACTION_BUTTON_STYLESHEET)

        layout.addLayout(form_layout)
        layout.addWidget(self.status_label)
        layout.addWidget(self.visualize_button)

        self.points_combo.currentIndexChanged.connect(self._emit_source_changed)
        self.index_column_combo.currentIndexChanged.connect(self._emit_source_changed)
        self.visualize_button.clicked.connect(self._emit_visualize_requested)
        self.all_values_checkbox.toggled.connect(self._refresh_value_input_state)
        self.value_input.textChanged.connect(self._refresh_visualize_state)
        self.render_point_budget_input.textChanged.connect(self._refresh_visualize_state)

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
        values = [] if value_source is None else [str(value) for value in value_source.value_table.values["value"]]
        self._value_completer_model.setStringList(values)
        self._refresh_value_input_state()

    def render_controller_state(self, controller: PointsController) -> None:
        """Render controller state into controls and the points status card."""
        self._refresh_enabled_state(
            can_visualize=controller.can_visualize,
            is_loading=controller.is_loading or controller.is_loading_values,
        )
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
        """Return parsed value selection for the next visualize request."""
        if self.all_values_checkbox.isChecked():
            return "all"
        values = tuple(dict.fromkeys(value.strip() for value in self.value_input.text().split(",") if value.strip()))
        return values

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

    def _emit_visualize_requested(self, _checked: bool = False) -> None:
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

        self.visualize_requested.emit(values, budget)

    def _refresh_value_input_state(self, _checked: bool | None = None) -> None:
        self.value_input.setEnabled(
            self._can_visualize
            and self.points_combo.isEnabled()
            and self.index_column_combo.isEnabled()
            and bool(self._value_completer_model.stringList())
            and not self.all_values_checkbox.isChecked()
        )
        self._refresh_visualize_state()

    def _refresh_visualize_state(self, _text: str | None = None) -> None:
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
        self.visualize_button.setEnabled(
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
