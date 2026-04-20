from __future__ import annotations

from typing import TYPE_CHECKING

from qtpy.QtCore import QSignalBlocker, Qt
from qtpy.QtWidgets import (
    QComboBox,
    QFormLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from spatialdata.transformations import get_transformation

from napari_harpy._app_state import HarpyAppState, get_or_create_app_state

if TYPE_CHECKING:
    import napari
    from spatialdata import SpatialData


class ViewerWidget(QWidget):
    """Minimal shared viewer widget shell backed by `HarpyAppState`."""

    def __init__(self, napari_viewer: napari.Viewer | None = None) -> None:
        super().__init__()
        self.setObjectName("viewer_widget")
        self._viewer = napari_viewer
        self._app_state = get_or_create_app_state(napari_viewer)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        title = QLabel("Viewer")
        title.setObjectName("viewer_widget_title")
        title.setStyleSheet("font-size: 18px; font-weight: 700;")

        self.empty_state_label = QLabel(
            "No SpatialData loaded. Use `Interactive(sdata)` for now; an in-widget open action will follow later."
        )
        self.empty_state_label.setObjectName("viewer_widget_empty_state")
        self.empty_state_label.setWordWrap(True)

        self.summary_label = QLabel("No SpatialData loaded.")
        self.summary_label.setObjectName("viewer_widget_summary")
        self.summary_label.setWordWrap(True)

        form_layout = QFormLayout()
        form_layout.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        form_layout.setHorizontalSpacing(12)
        form_layout.setVerticalSpacing(10)

        self.coordinate_system_combo = QComboBox()
        self.coordinate_system_combo.setObjectName("viewer_widget_coordinate_system_combo")

        self.labels_combo = QComboBox()
        self.labels_combo.setObjectName("viewer_widget_labels_combo")

        self.linked_table_combo = QComboBox()
        self.linked_table_combo.setObjectName("viewer_widget_linked_table_combo")

        self.image_combo = QComboBox()
        self.image_combo.setObjectName("viewer_widget_image_combo")

        self.display_mode_combo = QComboBox()
        self.display_mode_combo.setObjectName("viewer_widget_display_mode_combo")
        self.display_mode_combo.addItem("stack", "stack")
        self.display_mode_combo.addItem("overlay", "overlay")

        self.show_segmentation_button = QPushButton("Show segmentation")
        self.show_segmentation_button.setObjectName("viewer_widget_show_segmentation_button")
        self.show_segmentation_button.setEnabled(False)

        self.show_image_button = QPushButton("Show image")
        self.show_image_button.setObjectName("viewer_widget_show_image_button")
        self.show_image_button.setEnabled(False)

        self.show_selected_channels_button = QPushButton("Show selected channels")
        self.show_selected_channels_button.setObjectName("viewer_widget_show_selected_channels_button")
        self.show_selected_channels_button.setEnabled(False)

        form_layout.addRow("Coordinate system", self.coordinate_system_combo)
        form_layout.addRow("Labels", self.labels_combo)
        form_layout.addRow("Linked table", self.linked_table_combo)
        form_layout.addRow("Image", self.image_combo)
        form_layout.addRow("Display mode", self.display_mode_combo)

        layout.addWidget(title)
        layout.addWidget(self.empty_state_label)
        layout.addWidget(self.summary_label)
        layout.addLayout(form_layout)
        layout.addWidget(self.show_segmentation_button)
        layout.addWidget(self.show_image_button)
        layout.addWidget(self.show_selected_channels_button)
        layout.addStretch(1)

        self._app_state.sdata_changed.connect(self._on_sdata_changed)
        self.refresh_from_sdata(self._app_state.sdata)

    @property
    def app_state(self) -> HarpyAppState:
        """Return the shared Harpy app state for this widget."""
        return self._app_state

    def _on_sdata_changed(self, sdata: SpatialData | None) -> None:
        """Refresh the widget when the shared loaded `SpatialData` changes."""
        self.refresh_from_sdata(sdata)

    def refresh_from_sdata(self, sdata: SpatialData | None) -> None:
        """Refresh the minimal widget shell from the currently loaded `SpatialData`."""
        with (
            QSignalBlocker(self.coordinate_system_combo),
            QSignalBlocker(self.labels_combo),
            QSignalBlocker(self.linked_table_combo),
            QSignalBlocker(self.image_combo),
            QSignalBlocker(self.display_mode_combo),
        ):
            self.coordinate_system_combo.clear()
            self.labels_combo.clear()
            self.linked_table_combo.clear()
            self.image_combo.clear()

            if sdata is None:
                self.empty_state_label.show()
                self.summary_label.setText("No SpatialData loaded.")
                self._set_controls_enabled(False)
                return

            coordinate_systems = _get_coordinate_systems_from_sdata(sdata)
            labels = sorted(getattr(sdata, "labels", {}).keys())
            images = sorted(getattr(sdata, "images", {}).keys())

            self.coordinate_system_combo.addItems(coordinate_systems)
            self.labels_combo.addItems(labels)
            self.image_combo.addItems(images)
            self.linked_table_combo.addItem("Available after label selection")

            self.empty_state_label.hide()
            self.summary_label.setText(
                "Loaded SpatialData with "
                f"{len(coordinate_systems)} coordinate system(s), "
                f"{len(labels)} labels element(s), and "
                f"{len(images)} image element(s)."
            )
            self._set_controls_enabled(True)
            self.linked_table_combo.setEnabled(False)
            self.show_segmentation_button.setEnabled(bool(labels))
            self.show_image_button.setEnabled(bool(images))
            self.show_selected_channels_button.setEnabled(bool(images))

    def _set_controls_enabled(self, enabled: bool) -> None:
        self.coordinate_system_combo.setEnabled(enabled)
        self.labels_combo.setEnabled(enabled)
        self.image_combo.setEnabled(enabled)
        self.display_mode_combo.setEnabled(enabled)
        self.linked_table_combo.setEnabled(enabled)
        self.show_segmentation_button.setEnabled(enabled)
        self.show_image_button.setEnabled(enabled)
        self.show_selected_channels_button.setEnabled(enabled)


def _get_coordinate_systems_from_sdata(sdata: SpatialData) -> list[str]:
    coordinate_systems: set[str] = set()

    for collection_name in ("labels", "images"):
        collection = getattr(sdata, collection_name, {})
        for element in collection.values():
            coordinate_systems.update(get_transformation(element, get_all=True).keys())

    return sorted(coordinate_systems)
