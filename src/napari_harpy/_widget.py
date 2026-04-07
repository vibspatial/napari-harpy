from __future__ import annotations

from typing import TYPE_CHECKING

from qtpy.QtCore import QSignalBlocker
from qtpy.QtWidgets import QComboBox, QFormLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from napari_harpy._spatialdata import SpatialDataLabelsOption, get_spatialdata_label_options

if TYPE_CHECKING:
    import napari


class HarpyWidget(QWidget):
    """Phase 1 widget for selecting a segmentation mask from SpatialData labels."""

    def __init__(self, napari_viewer: napari.Viewer | None = None) -> None:
        super().__init__()
        self._viewer = napari_viewer
        self._label_options: list[SpatialDataLabelsOption] = []
        self._selected_option: SpatialDataLabelsOption | None = None

        layout = QVBoxLayout(self)

        title = QLabel("napari-harpy")
        title.setStyleSheet("font-size: 18px; font-weight: 600;")

        subtitle = QLabel(
            "Phase 1 setup.\n"
            "Select the segmentation mask to classify from the active SpatialData object."
        )
        subtitle.setWordWrap(True)

        self.viewer_status = QLabel(
            "Viewer connected." if napari_viewer is not None else "Viewer not connected."
        )
        self.viewer_status.setWordWrap(True)

        selector_layout = QFormLayout()
        self.segmentation_combo = QComboBox()
        self.segmentation_combo.setObjectName("segmentation_mask_combo")
        self.segmentation_combo.currentIndexChanged.connect(self._on_segmentation_changed)

        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.refresh_segmentation_masks)
        self.refresh_button.setEnabled(napari_viewer is not None)

        self.selection_status = QLabel()
        self.selection_status.setWordWrap(True)

        selector_layout.addRow("Segmentation mask", self.segmentation_combo)

        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addWidget(self.viewer_status)
        layout.addLayout(selector_layout)
        layout.addWidget(self.refresh_button)
        layout.addWidget(self.selection_status)
        layout.addStretch(1)

        self._connect_viewer_events()
        self.refresh_segmentation_masks()

    @property
    def selected_segmentation_name(self) -> str | None:
        """Return the currently selected labels element name."""
        return None if self._selected_option is None else self._selected_option.label_name

    @property
    def selected_spatialdata(self) -> object | None:
        """Return the SpatialData object that owns the current labels selection."""
        return None if self._selected_option is None else self._selected_option.sdata

    def refresh_segmentation_masks(self) -> None:
        """Refresh the segmentation mask choices from viewer-linked SpatialData layers."""
        previous_identity = None if self._selected_option is None else self._selected_option.identity
        self._label_options = get_spatialdata_label_options(self._viewer)

        with QSignalBlocker(self.segmentation_combo):
            self.segmentation_combo.clear()
            for option in self._label_options:
                self.segmentation_combo.addItem(option.display_name)

            has_options = bool(self._label_options)
            self.segmentation_combo.setEnabled(has_options)

            next_index = self._find_option_index(previous_identity)
            if has_options:
                self.segmentation_combo.setCurrentIndex(0 if next_index is None else next_index)
            else:
                self.segmentation_combo.setCurrentIndex(-1)

        if self.segmentation_combo.currentIndex() >= 0:
            self._set_selected_option(self.segmentation_combo.currentIndex())
        else:
            self._selected_option = None
            self._update_selection_status()

    def _connect_viewer_events(self) -> None:
        layers = getattr(self._viewer, "layers", None)
        events = getattr(layers, "events", None)
        if events is None:
            return

        for event_name in ("inserted", "removed", "reordered"):
            event_emitter = getattr(events, event_name, None)
            if event_emitter is not None:
                event_emitter.connect(self._on_viewer_layers_changed)

    def _on_viewer_layers_changed(self, event: object | None = None) -> None:
        del event
        self.refresh_segmentation_masks()

    def _on_segmentation_changed(self, index: int) -> None:
        self._set_selected_option(index)

    def _set_selected_option(self, index: int) -> None:
        if index < 0 or index >= len(self._label_options):
            self._selected_option = None
        else:
            self._selected_option = self._label_options[index]

        self._update_selection_status()

    def _find_option_index(self, identity: tuple[int, str] | None) -> int | None:
        if identity is None:
            return None

        for index, option in enumerate(self._label_options):
            if option.identity == identity:
                return index

        return None

    def _update_selection_status(self) -> None:
        if self._viewer is None:
            self.selection_status.setText("Connect the widget to a napari viewer to discover segmentation masks.")
            return

        if not self._label_options:
            self.selection_status.setText(
                "Load a SpatialData labels layer with napari-spatialdata to populate the segmentation menu."
            )
            return

        count = len(self._label_options)
        plural = "s" if count != 1 else ""

        if self._selected_option is None:
            self.selection_status.setText(f"Found {count} segmentation mask{plural}. Select one to continue.")
            return

        self.selection_status.setText(
            f"Found {count} segmentation mask{plural}. Selected: {self._selected_option.display_name}."
        )
