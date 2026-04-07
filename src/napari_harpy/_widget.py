from __future__ import annotations

from typing import TYPE_CHECKING

from qtpy.QtWidgets import QLabel, QVBoxLayout, QWidget

if TYPE_CHECKING:
    import napari


class HarpyWidget(QWidget):
    """Minimal dock widget for the Phase 0 plugin skeleton."""

    def __init__(self, napari_viewer: napari.Viewer | None = None) -> None:
        super().__init__()
        self._viewer = napari_viewer

        layout = QVBoxLayout(self)

        title = QLabel("napari-harpy")
        title.setStyleSheet("font-size: 18px; font-weight: 600;")

        subtitle = QLabel(
            "Plugin skeleton ready.\n"
            "Next step: connect SpatialData discovery and object classification."
        )
        subtitle.setWordWrap(True)

        viewer_status = QLabel(
            "Viewer connected." if napari_viewer is not None else "Viewer not connected."
        )
        viewer_status.setWordWrap(True)

        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addWidget(viewer_status)
        layout.addStretch(1)
