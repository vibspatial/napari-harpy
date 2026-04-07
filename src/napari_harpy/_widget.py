from __future__ import annotations

from typing import TYPE_CHECKING

from qtpy.QtCore import QSignalBlocker
from qtpy.QtWidgets import QComboBox, QFormLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from napari_harpy._spatialdata import (
    SpatialDataAdapter,
    SpatialDataLabelsOption,
    SpatialDataTableMetadata,
)

if TYPE_CHECKING:
    import napari
    from spatialdata import SpatialData


class HarpyWidget(QWidget):
    """Phase 1 widget for selecting segmentation, table, and feature inputs.

    The widget does not retrieve a `SpatialData` object directly from the napari
    viewer itself. Instead, it inspects the current viewer layers and looks for
    `napari-spatialdata` metadata stored as `layer.metadata["sdata"]`.

    From those viewer-linked `SpatialData` objects, the widget exposes:

    - segmentation masks from `sdata.labels`
    - annotating tables for the selected segmentation
    - feature matrix keys from `table.obsm`
    """

    def __init__(self, napari_viewer: napari.Viewer | None = None) -> None:
        super().__init__()
        self._viewer = napari_viewer
        self._spatialdata_adapter = SpatialDataAdapter(napari_viewer)
        self._label_options: list[SpatialDataLabelsOption] = []
        self._selected_label_option: SpatialDataLabelsOption | None = None
        self._table_names: list[str] = []
        self._selected_table_name: str | None = None
        self._feature_matrix_keys: list[str] = []
        self._selected_feature_key: str | None = None

        layout = QVBoxLayout(self)

        title = QLabel("napari-harpy")
        title.setStyleSheet("font-size: 18px; font-weight: 600;")

        subtitle = QLabel(
            "Phase 1 setup.\nSelect the segmentation mask to classify from the active SpatialData object."
        )
        subtitle.setWordWrap(True)

        self.viewer_status = QLabel("Viewer connected." if napari_viewer is not None else "Viewer not connected.")
        self.viewer_status.setWordWrap(True)

        selector_layout = QFormLayout()
        self.segmentation_combo = QComboBox()
        self.segmentation_combo.setObjectName("segmentation_mask_combo")
        self.segmentation_combo.currentIndexChanged.connect(self._on_segmentation_changed)

        self.table_combo = QComboBox()
        self.table_combo.setObjectName("annotation_table_combo")
        self.table_combo.currentIndexChanged.connect(self._on_table_changed)

        self.feature_matrix_combo = QComboBox()
        self.feature_matrix_combo.setObjectName("feature_matrix_combo")
        self.feature_matrix_combo.currentIndexChanged.connect(self._on_feature_matrix_changed)

        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.refresh_segmentation_masks)
        self.refresh_button.setEnabled(napari_viewer is not None)

        self.validation_status = QLabel()
        self.validation_status.setObjectName("validation_status")
        self.validation_status.setWordWrap(True)
        self.validation_status.setStyleSheet("color: #b45309; font-weight: 600;")
        self.validation_status.hide()

        selector_layout.addRow("Segmentation mask", self.segmentation_combo)
        selector_layout.addRow("Table", self.table_combo)
        selector_layout.addRow("Feature matrix", self.feature_matrix_combo)

        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addWidget(self.viewer_status)
        layout.addLayout(selector_layout)
        layout.addWidget(self.refresh_button)
        layout.addWidget(self.validation_status)
        layout.addStretch(1)

        self._connect_viewer_events()
        self.refresh_segmentation_masks()

    @property
    def selected_segmentation_name(self) -> str | None:
        """Return the currently selected labels element name."""
        return None if self._selected_label_option is None else self._selected_label_option.label_name

    @property
    def selected_spatialdata(self) -> SpatialData | None:
        """Return the SpatialData object that owns the current labels selection."""
        return None if self._selected_label_option is None else self._selected_label_option.sdata

    @property
    def selected_table_name(self) -> str | None:
        """Return the currently selected annotation table name."""
        return self._selected_table_name

    @property
    def selected_feature_key(self) -> str | None:
        """Return the currently selected feature matrix key from `adata.obsm`."""
        return self._selected_feature_key

    @property
    def selected_table_metadata(self) -> SpatialDataTableMetadata | None:
        """Return the linkage metadata for the current table selection."""
        if self.selected_spatialdata is None or self.selected_table_name is None:
            return None

        return self._spatialdata_adapter.get_table_metadata(self.selected_spatialdata, self.selected_table_name)

    def refresh_segmentation_masks(self) -> None:
        """Refresh the segmentation mask choices from viewer-linked SpatialData layers."""
        previous_identity = None if self._selected_label_option is None else self._selected_label_option.identity
        self._label_options = self._spatialdata_adapter.get_label_options()

        with QSignalBlocker(self.segmentation_combo):
            self.segmentation_combo.clear()
            for option in self._label_options:
                self.segmentation_combo.addItem(option.display_name)

            has_options = bool(self._label_options)
            self.segmentation_combo.setEnabled(has_options)

            # If the previously selected label is still available after a refresh,
            # keep it selected instead of resetting the user back to the first item.
            next_index = self._find_option_index(previous_identity)
            if has_options:
                self.segmentation_combo.setCurrentIndex(0 if next_index is None else next_index)
            else:
                self.segmentation_combo.setCurrentIndex(-1)

        if self.segmentation_combo.currentIndex() >= 0:
            self._set_selected_label_option(self.segmentation_combo.currentIndex())
        else:
            self._selected_label_option = None
            self._refresh_table_names()
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
        self._set_selected_label_option(index)

    def _set_selected_label_option(self, index: int) -> None:
        if index < 0 or index >= len(self._label_options):
            self._selected_label_option = None
        else:
            self._selected_label_option = self._label_options[index]

        self._refresh_table_names()
        self._update_selection_status()

    def _find_option_index(self, identity: tuple[int, str] | None) -> int | None:
        if identity is None:
            return None

        for index, option in enumerate(self._label_options):
            if option.identity == identity:
                return index

        return None

    def _refresh_table_names(self) -> None:
        previous_table_name = self._selected_table_name

        if self.selected_spatialdata is None or self.selected_segmentation_name is None:
            self._table_names = []
        else:
            self._table_names = self._spatialdata_adapter.get_annotating_table_names(
                self.selected_spatialdata, self.selected_segmentation_name
            )

        with QSignalBlocker(self.table_combo):
            self.table_combo.clear()
            for table_name in self._table_names:
                self.table_combo.addItem(table_name)

            has_tables = bool(self._table_names)
            self.table_combo.setEnabled(has_tables)

            next_index = self._find_table_index(previous_table_name)
            if has_tables:
                self.table_combo.setCurrentIndex(0 if next_index is None else next_index)
            else:
                self.table_combo.setCurrentIndex(-1)

        if self.table_combo.currentIndex() >= 0:
            self._set_selected_table_name(self.table_combo.currentIndex())
        else:
            self._selected_table_name = None
            self._refresh_feature_matrix_keys()

    def _on_table_changed(self, index: int) -> None:
        self._set_selected_table_name(index)
        self._update_selection_status()

    def _set_selected_table_name(self, index: int) -> None:
        if index < 0 or index >= len(self._table_names):
            self._selected_table_name = None
        else:
            self._selected_table_name = self._table_names[index]

        self._refresh_feature_matrix_keys()

    def _find_table_index(self, table_name: str | None) -> int | None:
        if table_name is None:
            return None

        for index, candidate in enumerate(self._table_names):
            if candidate == table_name:
                return index

        return None

    def _refresh_feature_matrix_keys(self) -> None:
        previous_feature_key = self._selected_feature_key

        if self.selected_spatialdata is None or self.selected_table_name is None:
            self._feature_matrix_keys = []
        else:
            self._feature_matrix_keys = self._spatialdata_adapter.get_table_obsm_keys(
                self.selected_spatialdata, self.selected_table_name
            )

        with QSignalBlocker(self.feature_matrix_combo):
            self.feature_matrix_combo.clear()
            for feature_key in self._feature_matrix_keys:
                self.feature_matrix_combo.addItem(feature_key)

            has_feature_matrices = bool(self._feature_matrix_keys)
            self.feature_matrix_combo.setEnabled(has_feature_matrices)

            next_index = self._find_feature_matrix_index(previous_feature_key)
            if has_feature_matrices:
                self.feature_matrix_combo.setCurrentIndex(0 if next_index is None else next_index)
            else:
                self.feature_matrix_combo.setCurrentIndex(-1)

        if self.feature_matrix_combo.currentIndex() >= 0:
            self._set_selected_feature_key(self.feature_matrix_combo.currentIndex())
        else:
            self._selected_feature_key = None

    def _on_feature_matrix_changed(self, index: int) -> None:
        self._set_selected_feature_key(index)
        self._update_selection_status()

    def _set_selected_feature_key(self, index: int) -> None:
        if index < 0 or index >= len(self._feature_matrix_keys):
            self._selected_feature_key = None
        else:
            self._selected_feature_key = self._feature_matrix_keys[index]

    def _find_feature_matrix_index(self, feature_key: str | None) -> int | None:
        if feature_key is None:
            return None

        for index, candidate in enumerate(self._feature_matrix_keys):
            if candidate == feature_key:
                return index

        return None

    def _update_selection_status(self) -> None:
        self._update_validation_status()

    def _update_validation_status(self) -> None:
        message = None

        if self.selected_table_name is not None and not self._feature_matrix_keys:
            message = (
                "Warning: the selected table does not contain any feature matrices in `.obsm`. "
                "Add one before continuing."
            )

        self.validation_status.setText("" if message is None else message)
        self.validation_status.setVisible(message is not None)
