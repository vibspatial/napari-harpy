"""User confirmation for destructive SpatialData replacement."""

from __future__ import annotations

from qtpy.QtWidgets import QDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from napari_harpy.widgets.shared_styles import (
    SECONDARY_BUTTON_STYLESHEET,
    WARNING_BUTTON_STYLESHEET,
    set_status_card,
)


def confirm_spatialdata_replacement(parent: QWidget | None = None) -> bool:
    """Ask whether the currently loaded SpatialData session may be discarded."""
    dialog = QDialog(parent)
    dialog.setWindowTitle("Replace SpatialData")
    dialog.setModal(True)
    dialog.setMinimumWidth(520)

    layout = QVBoxLayout(dialog)
    layout.setContentsMargins(14, 14, 14, 14)
    layout.setSpacing(10)

    warning_card = QLabel()
    warning_card.setWordWrap(True)
    set_status_card(
        warning_card,
        title="Replace SpatialData?",
        lines=["Any unsaved changes will be lost."],
        kind="warning",
    )
    layout.addWidget(warning_card)

    button_row = QHBoxLayout()
    button_row.setSpacing(10)
    button_row.addStretch(1)

    proceed_button = QPushButton("Proceed")
    cancel_button = QPushButton("Cancel")
    proceed_button.setStyleSheet(WARNING_BUTTON_STYLESHEET)
    cancel_button.setStyleSheet(SECONDARY_BUTTON_STYLESHEET)

    button_row.addWidget(proceed_button)
    button_row.addWidget(cancel_button)
    layout.addLayout(button_row)

    proceed_button.clicked.connect(dialog.accept)
    cancel_button.clicked.connect(dialog.reject)
    cancel_button.setDefault(True)

    return dialog.exec() == QDialog.DialogCode.Accepted
