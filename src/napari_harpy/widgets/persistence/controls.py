"""Reusable controls for writing and requesting reload of shared table state."""

from __future__ import annotations

from typing import TYPE_CHECKING

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from napari_harpy._app_state import HarpyAppState, TableDirtyStateChangedEvent
from napari_harpy.widgets.persistence.controller import PersistenceController
from napari_harpy.widgets.shared_styles import (
    ACTION_BUTTON_STYLESHEET,
    format_tooltip,
    set_status_card,
)

if TYPE_CHECKING:
    from spatialdata import SpatialData


class TablePersistenceControls(QWidget):
    """Present reusable controls for one table's shared persistence state.

    Writing is generic: the control can call ``PersistenceController``
    directly, acknowledge the persisted mutation tokens, and let the shared
    dirty-state event refresh every bound control.

    Reload cannot be executed directly by this component. Replacing live table
    components requires coordinated pre-reload preparation and post-reload
    recovery by every affected workflow. The Reload button therefore does not
    call ``PersistenceController`` directly. It emits ``reload_requested`` and
    requires its current host—or the future shared reload coordinator—to
    coordinate the accepted transition:

        Reload button clicked
            ↓
        TablePersistenceControls.reload_requested
            ↓
        host resolves clean versus dirty-table behavior
            ↓
        host performs its pre-reload preparation
            ↓
        PersistenceController reloads the table
            ↓
        host rebinds and refreshes from the restored state
    """

    reload_requested = Signal()

    def __init__(
        self,
        app_state: HarpyAppState,
        *,
        write_content_description: str = "table state",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        if not isinstance(write_content_description, str) or not write_content_description:
            raise ValueError("Persistence controls require a non-empty write-content description.")

        self._app_state = app_state
        self._controller = PersistenceController(app_state)
        self._write_content_description = write_content_description
        self._selected_spatialdata: SpatialData | None = None
        self._selected_table_name: str | None = None
        self._binding_error: str | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.action_row = QWidget()
        self.action_row.setObjectName("persistence_action_row")
        action_layout = QHBoxLayout(self.action_row)
        action_layout.setContentsMargins(0, 0, 0, 0)
        action_layout.setSpacing(8)

        self.write_button = QPushButton("Write Table State")
        self.write_button.setObjectName("sync_to_zarr_button")
        self.write_button.setEnabled(False)
        self.write_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.write_button.setMinimumHeight(28)
        self.write_button.setStyleSheet(ACTION_BUTTON_STYLESHEET)
        self.write_button.clicked.connect(self.write_table_state)

        self.reload_button = QPushButton("Reload Table State")
        self.reload_button.setObjectName("reload_from_zarr_button")
        self.reload_button.setEnabled(False)
        self.reload_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.reload_button.setMinimumHeight(28)
        self.reload_button.setStyleSheet(ACTION_BUTTON_STYLESHEET)
        self.reload_button.clicked.connect(self.reload_requested.emit)

        action_layout.addWidget(self.write_button, 1)
        action_layout.addWidget(self.reload_button, 1)

        self.feedback_label = QLabel()
        self.feedback_label.setObjectName("persistence_feedback")
        self.feedback_label.setWordWrap(True)
        self.feedback_label.hide()

        layout.addWidget(self.action_row)
        layout.addWidget(self.feedback_label)

        self._app_state.table_dirty_state_changed.connect(self._on_table_dirty_state_changed)
        self.refresh()

    @property
    def controller(self) -> PersistenceController:
        """Return the component-level persistence service."""
        return self._controller

    def bind(
        self,
        sdata: SpatialData | None,
        table_name: str | None,
        labels_name: str | None = None,
        *,
        binding_error: str | None = None,
    ) -> None:
        """Bind the controls to one selected table and refresh presentation."""
        if binding_error is not None and (not isinstance(binding_error, str) or not binding_error):
            raise ValueError("Persistence binding errors must be non-empty strings.")

        self._selected_spatialdata = sdata
        self._selected_table_name = table_name
        self._binding_error = binding_error
        effective_table_name = None if binding_error is not None else table_name
        self._controller.bind(sdata, effective_table_name, labels_name)
        self.clear_feedback()
        self.refresh()

    def refresh(self) -> None:
        """Refresh button readiness and tooltips from authoritative shared state."""
        can_sync = self._controller.can_sync
        can_write = self._controller.can_write_table_state
        can_reload = self._controller.can_reload
        self.write_button.setEnabled(can_write)
        self.reload_button.setEnabled(can_reload)

        if self._selected_spatialdata is None or self._selected_table_name is None:
            write_tooltip = (
                f"Choose a backed SpatialData annotation table to enable writing "
                f"{self._write_content_description} to disk."
            )
            reload_tooltip = (
                "Choose a backed SpatialData annotation table to enable discarding the current in-memory table state "
                "and reloading the table from disk."
            )
        elif self._binding_error is not None:
            write_tooltip = self._binding_error
            reload_tooltip = self._binding_error
        elif not can_sync or not can_reload:
            write_tooltip = (
                "The selected SpatialData dataset is not backed by zarr, so the in-memory table state cannot be "
                "written to disk."
            )
            reload_tooltip = (
                "The selected SpatialData dataset is not backed by zarr, so the table state cannot be reloaded "
                "from disk."
            )
        else:
            table_store_path = self._controller.selected_table_store_path
            destination = (
                self._selected_spatialdata.path
                if table_store_path is None
                else table_store_path
            )
            has_unsynced_changes = self._controller.has_unsynced_table_changes
            if has_unsynced_changes:
                write_tooltip = (
                    f'Write {self._write_content_description} for "{self._selected_table_name}" '
                    f'to "{destination}".'
                )
            else:
                write_tooltip = (
                    f'The selected "{self._selected_table_name}" table has no unsynced local in-memory changes '
                    "to write."
                )
            reload_tooltip = (
                f'Discard the current in-memory "{self._selected_table_name}" table state and reload the table '
                f'from "{destination}".'
            )
            if has_unsynced_changes:
                write_tooltip += " Unsynced local in-memory table changes are present."
                reload_tooltip += " Unsynced local in-memory table changes would be discarded."

        self.write_button.setToolTip(format_tooltip(write_tooltip))
        self.reload_button.setToolTip(format_tooltip(reload_tooltip))

    def write_table_state(
        self,
        *,
        show_feedback: bool = True,
        feedback_message: str | None = None,
    ) -> bool:
        """Write captured dirty components and optionally present the outcome."""
        try:
            self._controller.write_table_state()
        except ValueError as error:
            self.set_feedback(str(error), error=True)
            return False

        if show_feedback:
            destination = self.selected_table_store_destination()
            message = feedback_message or (
                f'Wrote "{self._selected_table_name}" {self._write_content_description} to "{destination}".'
            )
            self.set_feedback(message)
        return True

    def clear_feedback(self) -> None:
        """Hide any previous persistence outcome."""
        self.set_feedback("")

    def set_feedback(self, message: str, *, error: bool = False) -> None:
        """Present one persistence success or error outcome."""
        if not message:
            self.feedback_label.setText("")
            self.feedback_label.setVisible(False)
            return

        set_status_card(
            self.feedback_label,
            title="Persistence Error" if error else "Persistence Updated",
            lines=[message],
            kind="error" if error else "success",
        )

    def selected_table_store_destination(self) -> str:
        """Return the user-facing destination for the current table."""
        table_store_path = self._controller.selected_table_store_path
        if table_store_path is not None:
            return str(table_store_path)
        store_path = self._controller.selected_store_path
        return "" if store_path is None else str(store_path)

    def _on_table_dirty_state_changed(self, event: object) -> None:
        if not isinstance(event, TableDirtyStateChangedEvent):
            return
        if event.sdata is not self._selected_spatialdata or event.table_name != self._selected_table_name:
            return
        self.refresh()
