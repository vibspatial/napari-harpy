from __future__ import annotations

from html import escape
from pathlib import Path
from typing import TYPE_CHECKING

from qtpy.QtCore import QSignalBlocker, Qt
from qtpy.QtGui import QPixmap
from qtpy.QtWidgets import QComboBox, QFormLayout, QLabel, QPushButton, QSpinBox, QVBoxLayout, QWidget

from napari_harpy._annotation import UNLABELED_CLASS, AnnotationController
from napari_harpy._classifier import ClassifierController
from napari_harpy._persistence import PersistenceController
from napari_harpy._spatialdata import (
    SpatialDataAdapter,
    SpatialDataLabelsOption,
    SpatialDataTableMetadata,
    SpatialDataViewerBinding,
)
from napari_harpy._viewer_styling import (
    COLOR_BY_OPTIONS,
    COLOR_BY_PRED_CLASS,
    COLOR_BY_PRED_CONFIDENCE,
    COLOR_BY_USER_CLASS,
    ViewerStylingController,
)

if TYPE_CHECKING:
    import napari
    from spatialdata import SpatialData


class HarpyWidget(QWidget):
    """Phase 2 widget for selecting inputs and picking segmentation objects.

    The widget does not retrieve a `SpatialData` object directly from the napari
    viewer itself. Instead, it inspects the current viewer layers and looks for
    `napari-spatialdata` metadata stored as `layer.metadata["sdata"]`.

    From those viewer-linked `SpatialData` objects, the widget exposes:

    - segmentation masks from `sdata.labels`
    - annotating tables for the selected segmentation
    - feature matrix keys from `table.obsm`
    - the currently picked segmentation instance id from the active `Labels` layer
    """

    def __init__(self, napari_viewer: napari.Viewer | None = None) -> None:
        super().__init__()
        self._viewer = napari_viewer
        self._spatialdata_adapter = SpatialDataAdapter()
        self._viewer_binding = SpatialDataViewerBinding(napari_viewer, self._spatialdata_adapter)
        self._annotation_controller = AnnotationController(
            self._spatialdata_adapter,
            self._viewer_binding,
            on_selected_instance_changed=self._on_selected_instance_changed,
            on_annotation_changed=self._on_annotation_changed,
        )
        self._classifier_controller = ClassifierController(
            self._spatialdata_adapter,
            on_state_changed=self._on_classifier_state_changed,
            on_table_state_changed=self._on_classifier_table_state_changed,
        )
        self._viewer_styling_controller = ViewerStylingController(
            self._spatialdata_adapter,
            self._viewer_binding,
        )
        self._persistence_controller = PersistenceController(self._spatialdata_adapter)
        self._label_options: list[SpatialDataLabelsOption] = []
        self._selected_label_option: SpatialDataLabelsOption | None = None
        self._table_names: list[str] = []
        self._selected_table_name: str | None = None
        self._table_binding_error: str | None = None
        self._feature_matrix_keys: list[str] = []
        self._selected_feature_key: str | None = None
        self._logo_path = Path(__file__).resolve().parents[2] / "docs" / "_static" / "logo.png"

        layout = QVBoxLayout(self)

        title = self._create_header_logo()

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

        self.color_by_combo = QComboBox()
        self.color_by_combo.setObjectName("color_by_combo")
        for color_by in COLOR_BY_OPTIONS:
            self.color_by_combo.addItem(color_by, color_by)
        self.color_by_combo.setCurrentIndex(self.color_by_combo.findData(COLOR_BY_USER_CLASS))
        self.color_by_combo.currentIndexChanged.connect(self._on_color_by_changed)

        self.class_spinbox = QSpinBox()
        self.class_spinbox.setObjectName("user_class_spinbox")
        self.class_spinbox.setRange(1, 999)
        self.class_spinbox.setValue(1)

        self.refresh_button = QPushButton("Rescan Viewer")
        self.refresh_button.clicked.connect(self.refresh_segmentation_masks)
        self.refresh_button.setEnabled(napari_viewer is not None)

        self.retrain_button = QPushButton("Retrain")
        self.retrain_button.setObjectName("retrain_button")
        self.retrain_button.clicked.connect(self._retrain_classifier)
        self.retrain_button.setEnabled(False)

        self.sync_button = QPushButton("Sync to zarr")
        self.sync_button.setObjectName("sync_to_zarr_button")
        self.sync_button.clicked.connect(self._sync_to_zarr)
        self.sync_button.setEnabled(False)

        self.reload_button = QPushButton("Reload from zarr")
        self.reload_button.setObjectName("reload_from_zarr_button")
        self.reload_button.clicked.connect(self._reload_from_zarr)
        self.reload_button.setEnabled(False)

        self.validation_status = QLabel()
        self.validation_status.setObjectName("validation_status")
        self.validation_status.setWordWrap(True)
        self.validation_status.setStyleSheet("color: #b45309; font-weight: 600;")
        self.validation_status.hide()

        self.selection_status = QLabel()
        self.selection_status.setObjectName("selection_status")
        self.selection_status.setWordWrap(True)

        self.apply_class_button = QPushButton("Apply Class")
        self.apply_class_button.setObjectName("apply_class_button")
        self.apply_class_button.clicked.connect(self._apply_current_class)
        self.apply_class_button.setEnabled(False)

        self.clear_class_button = QPushButton("Clear Class")
        self.clear_class_button.setObjectName("clear_class_button")
        self.clear_class_button.clicked.connect(self._clear_current_class)
        self.clear_class_button.setEnabled(False)

        self.annotation_feedback = QLabel()
        self.annotation_feedback.setObjectName("annotation_feedback")
        self.annotation_feedback.setWordWrap(True)
        self.annotation_feedback.hide()

        self.classifier_feedback = QLabel()
        self.classifier_feedback.setObjectName("classifier_feedback")
        self.classifier_feedback.setWordWrap(True)
        self.classifier_feedback.hide()

        self.persistence_feedback = QLabel()
        self.persistence_feedback.setObjectName("persistence_feedback")
        self.persistence_feedback.setWordWrap(True)
        self.persistence_feedback.hide()

        selector_layout.addRow("Segmentation mask", self.segmentation_combo)
        selector_layout.addRow("Table", self.table_combo)
        selector_layout.addRow("Feature matrix", self.feature_matrix_combo)
        selector_layout.addRow("Color by", self.color_by_combo)
        selector_layout.addRow("User class", self.class_spinbox)

        layout.addWidget(title)
        layout.addLayout(selector_layout)
        layout.addWidget(self.refresh_button)
        layout.addWidget(self.retrain_button)
        layout.addWidget(self.sync_button)
        layout.addWidget(self.reload_button)
        layout.addWidget(self.apply_class_button)
        layout.addWidget(self.clear_class_button)
        layout.addWidget(self.selection_status)
        layout.addWidget(self.annotation_feedback)
        layout.addWidget(self.classifier_feedback)
        layout.addWidget(self.persistence_feedback)
        layout.addWidget(self.validation_status)
        layout.addStretch(1)

        self._connect_viewer_events()
        self.refresh_segmentation_masks()

    def _create_header_logo(self) -> QLabel:
        logo_label = QLabel()
        logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        logo_pixmap = QPixmap(str(self._logo_path))
        if not logo_pixmap.isNull():
            logo_label.setPixmap(
                logo_pixmap.scaledToWidth(120, Qt.TransformationMode.SmoothTransformation)
            )
            return logo_label

        logo_label.setText("napari-harpy")
        logo_label.setStyleSheet("font-size: 18px; font-weight: 600;")
        return logo_label

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
    def selected_instance_id(self) -> int | None:
        """Return the currently picked segmentation instance id."""
        return self._annotation_controller.selected_instance_id

    @property
    def selected_color_by(self) -> str:
        """Return the current labels-layer coloring mode."""
        return self._viewer_styling_controller.color_by

    @property
    def selected_table_metadata(self) -> SpatialDataTableMetadata | None:
        """Return the linkage metadata for the current table selection."""
        if self.selected_spatialdata is None or self.selected_table_name is None:
            return None

        return self._spatialdata_adapter.get_table_metadata(self.selected_spatialdata, self.selected_table_name)

    def refresh_segmentation_masks(self) -> None:
        """Refresh the segmentation mask choices from viewer-linked SpatialData layers."""
        previous_identity = None if self._selected_label_option is None else self._selected_label_option.identity
        self._label_options = self._viewer_binding.get_label_options()

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
            # No valid segmentation remains after the refresh, so clear every
            # controller and dependent UI element back to the unbound state.
            self._selected_label_option = None
            self._table_binding_error = None
            self._refresh_table_names()
            self._annotation_controller.bind(None, None, None)
            self._classifier_controller.bind(None, None, None, None)
            self._viewer_styling_controller.bind(None, None, None)
            self._persistence_controller.bind(None, None, None)
            self._refresh_layer_styling()
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
        self._bind_current_selection(classifier_dirty_reason="the segmentation selection changed")

    def _find_option_index(self, identity: tuple[int, str] | None) -> int | None:
        if identity is None:
            return None

        for index, option in enumerate(self._label_options):
            if option.identity == identity:
                return index

        return None

    def _refresh_table_names(self) -> None:
        previous_table_name = self.selected_table_name

        if self.selected_spatialdata is None or self.selected_segmentation_name is None:
            self._table_names = []
        else:
            self._table_names = self._spatialdata_adapter.get_annotating_table_names(
                self.selected_spatialdata, self.selected_segmentation_name
            )

        with QSignalBlocker(self.table_combo):
            self.table_combo.clear()
            for table_name in self._table_names:
                self.table_combo.addItem(table_name, table_name)

            has_tables = bool(self._table_names)
            self.table_combo.setEnabled(has_tables)

            next_index = -1 if previous_table_name is None else self.table_combo.findData(previous_table_name)
            if has_tables:
                self.table_combo.setCurrentIndex(0 if next_index < 0 else next_index)
            else:
                self.table_combo.setCurrentIndex(-1)

        self._set_selected_table_name(self.table_combo.currentIndex())
        self._refresh_feature_matrix_keys()

    def _on_table_changed(self, index: int) -> None:
        self._set_selected_table_name(index)
        self._refresh_feature_matrix_keys()
        self._bind_current_selection(classifier_dirty_reason="the annotation table changed")

    def _refresh_feature_matrix_keys(self) -> None:
        previous_feature_key = self.selected_feature_key

        if self.selected_spatialdata is None or self.selected_table_name is None:
            self._feature_matrix_keys = []
        else:
            self._feature_matrix_keys = self._spatialdata_adapter.get_table_obsm_keys(
                self.selected_spatialdata, self.selected_table_name
            )

        with QSignalBlocker(self.feature_matrix_combo):
            self.feature_matrix_combo.clear()
            for feature_key in self._feature_matrix_keys:
                self.feature_matrix_combo.addItem(feature_key, feature_key)

            has_feature_matrices = bool(self._feature_matrix_keys)
            self.feature_matrix_combo.setEnabled(has_feature_matrices)

            next_index = (
                -1 if previous_feature_key is None else self.feature_matrix_combo.findData(previous_feature_key)
            )
            if has_feature_matrices:
                self.feature_matrix_combo.setCurrentIndex(0 if next_index < 0 else next_index)
            else:
                self.feature_matrix_combo.setCurrentIndex(-1)

        self._set_selected_feature_key(self.feature_matrix_combo.currentIndex())

    def _on_feature_matrix_changed(self, index: int) -> None:
        self._set_selected_feature_key(index)
        classifier_context_changed = self._classifier_controller.bind(
            self.selected_spatialdata,
            self.selected_segmentation_name,
            self._effective_table_name(),
            self.selected_feature_key,
        )
        if classifier_context_changed and self._effective_table_name() is not None:
            self._classifier_controller.mark_dirty(reason="the feature matrix changed")
        self._refresh_layer_styling()
        self._update_selection_status()

    def _on_color_by_changed(self, index: int) -> None:
        color_by = self.color_by_combo.itemData(index)
        if not isinstance(color_by, str):
            return

        self._viewer_styling_controller.set_color_by(color_by)
        self._refresh_layer_styling()

    def _set_selected_table_name(self, index: int) -> None:
        if index < 0 or index >= len(self._table_names):
            self._selected_table_name = None
        else:
            self._selected_table_name = self._table_names[index]

    def _set_selected_feature_key(self, index: int) -> None:
        if index < 0 or index >= len(self._feature_matrix_keys):
            self._selected_feature_key = None
        else:
            self._selected_feature_key = self._feature_matrix_keys[index]

    def _effective_table_name(self) -> str | None:
        if self._table_binding_error is not None:
            return None

        return self.selected_table_name

    def _validate_selected_table_binding(self) -> str | None:
        if (
            self.selected_spatialdata is None
            or self.selected_segmentation_name is None
            or self.selected_table_name is None
        ):
            return None

        try:
            self._spatialdata_adapter.validate_table_binding(
                self.selected_spatialdata,
                self.selected_segmentation_name,
                self.selected_table_name,
            )
        except ValueError as error:
            return str(error)

        return None

    def _bind_current_selection(self, *, classifier_dirty_reason: str | None = None) -> None:
        """Rebind every controller to the current widget selection.

        This is the central handoff point from widget-level selection state to
        the controllers. Whenever the selected segmentation, table, or feature
        matrix context changes, we call this helper so each controller updates
        which in-memory ``SpatialData`` object and table it should operate on.

        Importantly, the controllers do not own independent copies of
        ``SpatialData`` or ``AnnData``. They receive references to the current
        in-memory objects and resolve the selected table from that authoritative
        state. Rebinding here therefore "refreshes" controller state by
        updating those references, not by materializing new table objects.

        Before rebinding, the selected table is validated against the selected
        labels layer. If the linkage is invalid, the table is intentionally
        downgraded to an ``effective_table_name`` of ``None`` so the
        controllers can stay bound to the current ``SpatialData``/labels
        context without attempting table-backed operations that would be
        inconsistent.

        The method also:

        - validates whether the selected table can annotate the selected labels layer
        - propagates that effective binding to annotation, classifier, styling,
          and persistence controllers
        - marks classifier outputs dirty when the classifier selection context
          changed in a way that invalidates them
        - re-applies layer styling and refreshes the user-facing status cards
        """
        self._table_binding_error = self._validate_selected_table_binding()
        effective_table_name = self._effective_table_name()

        self._annotation_controller.bind(
            self.selected_spatialdata,
            self.selected_segmentation_name,
            effective_table_name,
        )
        classifier_context_changed = self._classifier_controller.bind(
            self.selected_spatialdata,
            self.selected_segmentation_name,
            effective_table_name,
            self.selected_feature_key,
        )
        if (
            classifier_dirty_reason is not None
            and classifier_context_changed
            and effective_table_name is not None
        ):
            self._classifier_controller.mark_dirty(reason=classifier_dirty_reason)
        self._viewer_styling_controller.bind(
            self.selected_spatialdata,
            self.selected_segmentation_name,
            effective_table_name,
        )
        self._persistence_controller.bind(
            self.selected_spatialdata,
            effective_table_name,
            self.selected_segmentation_name,
        )
        self._annotation_controller.activate_layer()
        self._refresh_layer_styling()
        self._set_annotation_feedback("")
        self._set_persistence_feedback("")
        self._update_selection_status()

    def _update_selection_status(self) -> None:
        self._update_validation_status()
        self._update_annotation_status()
        self._update_annotation_controls()
        self._update_color_by_controls()
        self._update_classifier_controls()
        self._update_persistence_controls()

    def _update_validation_status(self) -> None:
        message = None

        if self._table_binding_error is not None:
            message = self._table_binding_error
        elif self.selected_table_name is not None and self.feature_matrix_combo.count() == 0:
            message = (
                "Warning: the selected table does not contain any feature matrices in `.obsm`. "
                "Add one before continuing."
            )

        self.validation_status.setText("" if message is None else message)
        self.validation_status.setVisible(message is not None)

    def _update_annotation_status(self) -> None:
        labels_layer = self._annotation_controller.labels_layer
        missing_table_row_message = self._annotation_controller.missing_table_row_message

        if self.selected_segmentation_name is None:
            self._set_selection_status(
                title="Selection",
                lines=["Choose a segmentation mask to enable object picking."],
                kind="info",
            )
        elif labels_layer is None:
            self._set_selection_status(
                title="Selection",
                lines=[
                    "The chosen segmentation is known in SpatialData but is not currently loaded as a napari Labels layer."
                ],
                kind="warning",
            )
        elif self._table_binding_error is not None:
            self._set_selection_status(
                title="Selection Warning",
                lines=[
                    f"Bound to {self.selected_segmentation_name}.",
                    self._table_binding_error,
                ],
                kind="warning",
            )
        elif self.selected_instance_id is None:
            self._set_selection_status(
                title="Selection",
                lines=[
                    f"Bound to {self.selected_segmentation_name}.",
                    "Click an object in the viewer.",
                ],
                kind="info",
            )
        elif missing_table_row_message is not None:
            self._set_selection_status(
                title="Selection Warning",
                lines=[
                    f"Bound to {self.selected_segmentation_name}.",
                    missing_table_row_message,
                ],
                kind="warning",
            )
        else:
            current_user_class = self._annotation_controller.current_user_class
            current_class_label = (
                "unlabeled" if current_user_class in (None, UNLABELED_CLASS) else str(current_user_class)
            )
            instance_key_name = self._selected_instance_key_name()
            self._set_selection_status(
                title="Selection Ready",
                lines=[
                    f"Bound to {self.selected_segmentation_name}.",
                    f"Current {instance_key_name}: {self.selected_instance_id}.",
                    f"Current class: {current_class_label}.",
                ],
                kind="success",
            )

    def _update_annotation_controls(self) -> None:
        has_table = self._effective_table_name() is not None
        current_user_class = self._annotation_controller.current_user_class

        self.class_spinbox.setEnabled(has_table)
        self.apply_class_button.setEnabled(self._annotation_controller.can_annotate)
        self.clear_class_button.setEnabled(
            self._annotation_controller.can_annotate and current_user_class not in (None, UNLABELED_CLASS)
        )

    def _apply_current_class(self) -> None:
        class_id = self.class_spinbox.value()
        try:
            warning_message = self._annotation_controller.apply_class(class_id)
        except ValueError as error:
            self._set_annotation_feedback(str(error), kind="error")
            return

        if warning_message is not None:
            self._set_annotation_feedback(warning_message, kind="warning")
            self._update_selection_status()
            return

        self._set_annotation_feedback(
            f"Assigned class {class_id} to {self._selected_instance_key_name()} {self.selected_instance_id}.",
            kind="success",
        )
        self._update_selection_status()

    def _clear_current_class(self) -> None:
        try:
            warning_message = self._annotation_controller.clear_current_class()
        except ValueError as error:
            self._set_annotation_feedback(str(error), kind="error")
            return

        if warning_message is not None:
            self._set_annotation_feedback(warning_message, kind="warning")
            self._update_selection_status()
            return

        self._set_annotation_feedback(
            f"Cleared the user class for {self._selected_instance_key_name()} {self.selected_instance_id}.",
            kind="success",
        )
        self._update_selection_status()

    def _set_annotation_feedback(self, message: str, *, kind: str = "success") -> None:
        if not message:
            self.annotation_feedback.setText("")
            self.annotation_feedback.setVisible(False)
            return

        title_by_kind = {
            "error": "Annotation Error",
            "warning": "Annotation Warning",
            "success": "Annotation Updated",
        }
        self._set_status_card(
            self.annotation_feedback,
            title=title_by_kind.get(kind, "Annotation"),
            lines=[message],
            kind=kind,
        )

    def _set_selection_status(self, title: str, lines: list[str], *, kind: str) -> None:
        self._set_status_card(self.selection_status, title=title, lines=lines, kind=kind)

    def _set_classifier_feedback(self, message: str, *, kind: str = "info") -> None:
        if not message:
            self.classifier_feedback.setText("")
            self.classifier_feedback.setVisible(False)
            return

        title_by_kind = {
            "error": "Classifier Error",
            "warning": "Classifier Warning",
            "success": "Classifier Ready",
            "info": "Classifier",
        }
        body = message.removeprefix("Classifier: ").strip()
        self._set_status_card(
            self.classifier_feedback,
            title=title_by_kind.get(kind, "Classifier"),
            lines=[body],
            kind=kind,
        )

    def _set_status_card(self, label: QLabel, *, title: str, lines: list[str], kind: str) -> None:
        palette_by_kind = {
            "info": {"text": "#1d4ed8", "border": "#93c5fd", "background": "#eff6ff"},
            "warning": {"text": "#b45309", "border": "#fdba74", "background": "#fff7ed"},
            "success": {"text": "#166534", "border": "#86efac", "background": "#f0fdf4"},
            "error": {"text": "#b91c1c", "border": "#fca5a5", "background": "#fef2f2"},
        }
        palette = palette_by_kind.get(kind, palette_by_kind["info"])
        formatted_lines = "<br>".join(f"<span>{escape(line)}</span>" for line in lines)
        label.setText(
            "<div>"
            f"<span style='font-size: 11px; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase;'>"
            f"{escape(title)}</span><br>"
            f"{formatted_lines}"
            "</div>"
        )
        label.setStyleSheet(
            "font-weight: 500; "
            f"color: {palette['text']}; "
            f"background-color: {palette['background']}; "
            f"border: 1px solid {palette['border']}; "
            "border-radius: 8px; "
            "padding: 10px 12px;"
        )
        label.setVisible(bool(lines))

    def _selected_instance_key_name(self) -> str:
        instance_key_name = self._annotation_controller.selected_instance_key_name
        if instance_key_name is not None:
            return instance_key_name

        return "label value"

    def _update_persistence_controls(self) -> None:
        can_sync = self._persistence_controller.can_sync
        can_reload = self._persistence_controller.can_reload
        self.sync_button.setEnabled(can_sync)
        self.reload_button.setEnabled(can_reload)

        if self.selected_spatialdata is None or self.selected_table_name is None:
            sync_tooltip = "Choose a backed SpatialData annotation table to enable sync."
            reload_tooltip = "Choose a backed SpatialData annotation table to enable reload."
        elif self._table_binding_error is not None:
            sync_tooltip = self._table_binding_error
            reload_tooltip = self._table_binding_error
        elif not can_sync or not can_reload:
            sync_tooltip = "The selected SpatialData dataset is not backed by zarr."
            reload_tooltip = "The selected SpatialData dataset is not backed by zarr."
        else:
            table_store_path = self._persistence_controller.selected_table_store_path
            destination = (
                self.selected_spatialdata.path
                if table_store_path is None
                else table_store_path
            )
            sync_tooltip = (
                f"Write `{self.selected_table_name}` table state "
                f"to `{destination}`."
            )
            reload_tooltip = (
                f"Reload `{self.selected_table_name}` table state "
                f"from `{destination}`."
            )
            if self._persistence_controller.is_dirty:
                sync_tooltip += " Unsynced local table changes are present."
                reload_tooltip += " Unsynced local table changes are present."

        self.sync_button.setToolTip(sync_tooltip)
        self.reload_button.setToolTip(reload_tooltip)

    def _update_color_by_controls(self) -> None:
        has_table = self._effective_table_name() is not None
        self.color_by_combo.setEnabled(has_table)

        if not has_table:
            tooltip = (
                self._table_binding_error
                if self._table_binding_error is not None
                else "Choose an annotation table before changing the labels-layer coloring mode."
            )
        elif self.selected_color_by == COLOR_BY_USER_CLASS:
            tooltip = "Color the labels layer by `user_class`."
        elif self.selected_color_by == COLOR_BY_PRED_CLASS:
            tooltip = "Color the labels layer by `pred_class` using the stable user-class palette."
        elif self.selected_color_by == COLOR_BY_PRED_CONFIDENCE:
            tooltip = "Color the labels layer by continuous `pred_confidence` values."
        else:
            tooltip = "Choose how to color the labels layer."

        self.color_by_combo.setToolTip(tooltip)

    def _update_classifier_controls(self) -> None:
        can_retrain = self._classifier_controller.can_retrain
        self.retrain_button.setEnabled(can_retrain)

        if self.selected_spatialdata is None or self.selected_table_name is None:
            tooltip = "Choose a segmentation and annotation table to enable retraining."
        elif self._table_binding_error is not None:
            tooltip = self._table_binding_error
        elif self.selected_feature_key is None:
            tooltip = "Choose a feature matrix before retraining the classifier."
        elif self._classifier_controller.is_training:
            tooltip = "A classifier retraining job is currently running."
        elif self._classifier_controller.is_dirty:
            tooltip = "The classifier model is stale. Click to retrain and refresh predictions."
        else:
            tooltip = "Retrain the classifier using the current annotations and feature matrix."

        self.retrain_button.setToolTip(tooltip)

    def _sync_to_zarr(self) -> None:
        try:
            self._persistence_controller.sync_table_state()
        except ValueError as error:
            self._set_persistence_feedback(str(error), error=True)
            return

        table_store_path = self._persistence_controller.selected_table_store_path
        destination = (
            self.selected_spatialdata.path
            if table_store_path is None or self.selected_spatialdata is None
            else table_store_path
        )
        self._set_persistence_feedback(
            f"Synced `{self.selected_table_name}` table state to `{destination}`.",
            error=False,
        )
        self._update_selection_status()

    def _reload_from_zarr(self) -> None:
        try:
            self._persistence_controller.reload_table_state()
        except ValueError as error:
            self._set_persistence_feedback(str(error), error=True)
            return

        self._refresh_feature_matrix_keys()
        self._bind_current_selection()
        table_store_path = self._persistence_controller.selected_table_store_path
        source = (
            self.selected_spatialdata.path
            if table_store_path is None or self.selected_spatialdata is None
            else table_store_path
        )
        self._set_persistence_feedback(
            f"Reloaded `{self.selected_table_name}` table state from `{source}`.",
            error=False,
        )

    def _set_persistence_feedback(self, message: str, *, error: bool = False) -> None:
        self.persistence_feedback.setText(message)
        self.persistence_feedback.setStyleSheet(
            "color: #b91c1c; font-weight: 600;" if error else "color: #166534; font-weight: 600;"
        )
        self.persistence_feedback.setVisible(bool(message))

    def _on_selected_instance_changed(self, instance_id: int | None) -> None:
        del instance_id
        self._set_annotation_feedback("")
        self._update_annotation_status()
        self._update_annotation_controls()

    def _on_annotation_changed(self) -> None:
        self._mark_persistence_dirty()
        self._classifier_controller.mark_dirty(reason="the annotations changed")
        self._refresh_layer_styling()
        self._classifier_controller.schedule_retrain()
        self._update_selection_status()

    def _on_classifier_table_state_changed(self) -> None:
        self._mark_persistence_dirty()
        self._update_persistence_controls()

    def _on_classifier_state_changed(self) -> None:
        self._set_classifier_feedback(
            self._classifier_controller.status_message,
            kind=self._classifier_controller.status_kind,
        )
        self._refresh_layer_styling()
        self._update_classifier_controls()

    def _retrain_classifier(self) -> None:
        self._classifier_controller.mark_dirty(reason="the user requested a retrain")
        self._classifier_controller.retrain_now()
        self._update_selection_status()

    def _refresh_layer_styling(self) -> None:
        self._viewer_styling_controller.refresh()

    def _mark_persistence_dirty(self) -> None:
        self._persistence_controller.mark_dirty()
        self._set_persistence_feedback("")
