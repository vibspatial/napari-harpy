from __future__ import annotations

from enum import Enum
from html import escape
from pathlib import Path
from typing import TYPE_CHECKING

from qtpy.QtCore import QSignalBlocker, Qt
from qtpy.QtGui import QKeySequence, QPixmap, QShortcut
from qtpy.QtWidgets import (
    QComboBox,
    QDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from napari_harpy._annotation import UNLABELED_CLASS, AnnotationController
from napari_harpy._app_state import HarpyAppState, get_or_create_app_state
from napari_harpy._classifier import ClassifierController
from napari_harpy._persistence import PersistenceController
from napari_harpy._spatialdata import (
    SpatialDataLabelsOption,
    SpatialDataTableMetadata,
    get_annotating_table_names,
    get_coordinate_system_names_from_sdata,
    get_spatialdata_label_options_for_coordinate_system_from_sdata,
    get_table_metadata,
    get_table_obsm_keys,
    validate_table_binding,
)
from napari_harpy._viewer_styling import (
    COLOR_BY_OPTIONS,
    COLOR_BY_PRED_CLASS,
    COLOR_BY_PRED_CONFIDENCE,
    COLOR_BY_USER_CLASS,
    ViewerStylingController,
)
from napari_harpy.widgets._shared_styles import (
    ACTION_BUTTON_STYLESHEET as _ACTION_BUTTON_STYLESHEET,
)
from napari_harpy.widgets._shared_styles import (
    WIDGET_MIN_WIDTH as _WIDGET_MIN_WIDTH,
)
from napari_harpy.widgets._shared_styles import (
    CompactComboBox,
    apply_scroll_content_surface,
    apply_widget_surface,
    build_input_control_stylesheet,
    create_form_label,
    format_tooltip,
)

if TYPE_CHECKING:
    import napari
    from spatialdata import SpatialData


class _DirtyReloadDecision(Enum):
    WRITE = "write"
    RELOAD_DISCARD = "reload_discard"
    CANCEL = "cancel"


_APPLY_CLASS_SHORTCUT = "A"
_REMOVE_CLASS_SHORTCUT = "R"
_INPUT_CONTROL_STYLESHEET = build_input_control_stylesheet("QComboBox, QSpinBox")
_CLASS_EDITOR_STYLESHEET = (
    "QWidget#class_editor {background-color: #f8eeea; border: 1px solid #eadfd8; border-radius: 10px;}"
)


class ObjectClassificationWidget(QWidget):
    """
    Widget for object classification.

    The widget is migrating toward the shared Harpy app-state architecture.
    It already receives the loaded `SpatialData` object through
    `self._app_state.sdata` / `sdata_changed`, and now resolves live labels
    layers through the shared `ViewerAdapter` plus layer bindings.

    In the current transition state, the widget exposes:

    - coordinate systems from the shared loaded `SpatialData`
    - segmentation masks filtered by the selected coordinate system
    - annotating tables for the selected segmentation
    - feature matrix keys from `table.obsm`
    - the currently picked segmentation instance id from the active `Labels` layer
    """

    def __init__(self, napari_viewer: napari.Viewer | None = None) -> None:
        super().__init__()
        self.setObjectName("object_classification_widget")
        apply_widget_surface(self)
        self.setMinimumWidth(_WIDGET_MIN_WIDTH)
        self._viewer = napari_viewer
        # The napari viewer identifies which shared Harpy session this widget
        # belongs to. We use it to attach to the per-viewer HarpyAppState even
        # though some legacy refresh UX is still being cleaned up in VW-05.
        self._app_state = get_or_create_app_state(napari_viewer)
        self._annotation_controller = AnnotationController(
            self._app_state.viewer_adapter,
            on_selected_instance_changed=self._on_selected_instance_changed,
            on_annotation_changed=self._on_annotation_changed,
        )
        self._classifier_controller = ClassifierController(
            on_state_changed=self._on_classifier_state_changed,
            on_table_state_changed=self._on_classifier_table_state_changed,
        )
        self._viewer_styling_controller = ViewerStylingController(
            self._app_state.viewer_adapter,
        )
        self._persistence_controller = PersistenceController()
        self._coordinate_systems: list[str] = []
        self._selected_coordinate_system: str | None = None
        self._label_options: list[SpatialDataLabelsOption] = []
        self._selected_label_option: SpatialDataLabelsOption | None = None
        self._is_preparing_labels_layer = False
        self._labels_layer_preparation_message: str | None = None
        self._labels_layer_preparation_error: str | None = None
        self._table_names: list[str] = []
        self._selected_table_name: str | None = None
        self._table_binding_error: str | None = None
        self._feature_matrix_keys: list[str] = []
        self._selected_feature_key: str | None = None
        self._logo_path = Path(__file__).resolve().parents[3] / "docs" / "_static" / "logo.png"

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.scroll_area = QScrollArea()
        self.scroll_area.setObjectName("object_classification_scroll_area")
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll_area.setStyleSheet("QScrollArea { border: 0px; background: transparent; }")

        self.scroll_content = QWidget()
        self.scroll_content.setObjectName("object_classification_scroll_content")
        apply_scroll_content_surface(self.scroll_content)

        content_layout = QVBoxLayout(self.scroll_content)
        content_layout.setContentsMargins(12, 12, 12, 12)
        content_layout.setSpacing(10)
        self.scroll_area.setWidget(self.scroll_content)
        layout.addWidget(self.scroll_area)

        title = self._create_header_logo()

        selector_layout = QFormLayout()
        selector_layout.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        selector_layout.setHorizontalSpacing(12)
        selector_layout.setVerticalSpacing(10)
        self.coordinate_system_combo = CompactComboBox()
        self.coordinate_system_combo.setObjectName("object_classification_coordinate_system_combo")
        self.coordinate_system_combo.currentIndexChanged.connect(self._on_coordinate_system_changed)
        self.coordinate_system_combo.setStyleSheet(_INPUT_CONTROL_STYLESHEET)

        self.segmentation_combo = CompactComboBox()
        self.segmentation_combo.setObjectName("segmentation_mask_combo")
        self.segmentation_combo.setPlaceholderText("Choose segmentation mask")
        self.segmentation_combo.currentIndexChanged.connect(self._on_segmentation_changed)
        self.segmentation_combo.setStyleSheet(_INPUT_CONTROL_STYLESHEET)

        self.table_combo = CompactComboBox()
        self.table_combo.setObjectName("annotation_table_combo")
        self.table_combo.currentIndexChanged.connect(self._on_table_changed)
        self.table_combo.setStyleSheet(_INPUT_CONTROL_STYLESHEET)

        self.feature_matrix_combo = CompactComboBox()
        self.feature_matrix_combo.setObjectName("feature_matrix_combo")
        self.feature_matrix_combo.currentIndexChanged.connect(self._on_feature_matrix_changed)
        self.feature_matrix_combo.setStyleSheet(_INPUT_CONTROL_STYLESHEET)

        self.color_by_combo = QComboBox()
        self.color_by_combo.setObjectName("color_by_combo")
        for color_by in COLOR_BY_OPTIONS:
            self.color_by_combo.addItem(color_by, color_by)
        self.color_by_combo.setCurrentIndex(self.color_by_combo.findData(COLOR_BY_USER_CLASS))
        self.color_by_combo.currentIndexChanged.connect(self._on_color_by_changed)
        self.color_by_combo.setStyleSheet(_INPUT_CONTROL_STYLESHEET)

        self.class_spinbox = QSpinBox()
        self.class_spinbox.setObjectName("user_class_spinbox")
        self.class_spinbox.setRange(1, 999)
        self.class_spinbox.setValue(1)
        self.class_spinbox.setStyleSheet(_INPUT_CONTROL_STYLESHEET)
        self.class_editor = QWidget()
        self.class_editor.setObjectName("class_editor")
        self.class_editor.setStyleSheet(_CLASS_EDITOR_STYLESHEET)
        class_editor_layout = QVBoxLayout(self.class_editor)
        class_editor_layout.setContentsMargins(8, 8, 8, 8)
        class_editor_layout.setSpacing(8)
        self.class_action_row = QWidget()
        self.class_action_row.setObjectName("class_action_row")
        class_action_layout = QHBoxLayout(self.class_action_row)
        class_action_layout.setContentsMargins(0, 0, 0, 0)
        class_action_layout.setSpacing(8)
        self.retrain_action_row = QWidget()
        self.retrain_action_row.setObjectName("retrain_action_row")
        retrain_action_layout = QHBoxLayout(self.retrain_action_row)
        retrain_action_layout.setContentsMargins(0, 0, 0, 0)
        retrain_action_layout.setSpacing(8)
        self.persistence_action_row = QWidget()
        self.persistence_action_row.setObjectName("persistence_action_row")
        persistence_action_layout = QHBoxLayout(self.persistence_action_row)
        persistence_action_layout.setContentsMargins(0, 0, 0, 0)
        persistence_action_layout.setSpacing(8)
        self.refresh_action_row = QWidget()
        self.refresh_action_row.setObjectName("refresh_action_row")
        refresh_action_layout = QHBoxLayout(self.refresh_action_row)
        refresh_action_layout.setContentsMargins(0, 0, 0, 0)
        refresh_action_layout.setSpacing(8)

        self.refresh_button = QPushButton("Rescan Viewer")
        self.refresh_button.clicked.connect(self._refresh_from_current_app_state)
        self.refresh_button.setEnabled(False)
        self.refresh_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.refresh_button.setMinimumHeight(28)
        self.refresh_button.setStyleSheet(_ACTION_BUTTON_STYLESHEET)

        self.retrain_button = QPushButton("Retrain")
        self.retrain_button.setObjectName("retrain_button")
        self.retrain_button.clicked.connect(self._retrain_classifier)
        self.retrain_button.setEnabled(False)
        self.retrain_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.retrain_button.setMinimumHeight(28)
        self.retrain_button.setStyleSheet(_ACTION_BUTTON_STYLESHEET)

        self.sync_button = QPushButton("Write")
        self.sync_button.setObjectName("sync_to_zarr_button")
        self.sync_button.clicked.connect(self._write_to_zarr)
        self.sync_button.setEnabled(False)
        self.sync_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.sync_button.setMinimumHeight(28)
        self.sync_button.setStyleSheet(_ACTION_BUTTON_STYLESHEET)

        self.reload_button = QPushButton("Reload")
        self.reload_button.setObjectName("reload_from_zarr_button")
        self.reload_button.clicked.connect(self._reload_from_zarr)
        self.reload_button.setEnabled(False)
        self.reload_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.reload_button.setMinimumHeight(28)
        self.reload_button.setStyleSheet(_ACTION_BUTTON_STYLESHEET)

        self.validation_status = QLabel()
        self.validation_status.setObjectName("validation_status")
        self.validation_status.setWordWrap(True)
        self.validation_status.setStyleSheet("color: #b45309; font-weight: 600;")
        self.validation_status.hide()

        self.selection_status = QLabel()
        self.selection_status.setObjectName("selection_status")
        self.selection_status.setWordWrap(True)

        self.apply_class_button = QPushButton("Add (A)")
        self.apply_class_button.setObjectName("apply_class_button")
        self.apply_class_button.clicked.connect(self._apply_current_class)
        self.apply_class_button.setEnabled(False)
        self.apply_class_button.setAccessibleName("Add")
        self.apply_class_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.apply_class_button.setMinimumHeight(28)
        self.apply_class_button.setStyleSheet(_ACTION_BUTTON_STYLESHEET)

        self.clear_class_button = QPushButton("Remove (R)")
        self.clear_class_button.setObjectName("clear_class_button")
        self.clear_class_button.clicked.connect(self._clear_current_class)
        self.clear_class_button.setEnabled(False)
        self.clear_class_button.setAccessibleName("Remove")
        self.clear_class_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.clear_class_button.setMinimumHeight(28)
        self.clear_class_button.setStyleSheet(_ACTION_BUTTON_STYLESHEET)
        class_editor_layout.addWidget(self.class_spinbox)
        class_action_layout.addWidget(self.apply_class_button, 1)
        class_action_layout.addWidget(self.clear_class_button, 1)
        class_editor_layout.addWidget(self.class_action_row)
        retrain_action_layout.addWidget(self.retrain_button, 1)
        persistence_action_layout.addWidget(self.sync_button, 1)
        persistence_action_layout.addWidget(self.reload_button, 1)
        refresh_action_layout.addWidget(self.refresh_button, 1)
        self._annotation_shortcuts = self._create_annotation_shortcuts()

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

        selector_layout.addRow(self._create_form_label("Coordinate system"), self.coordinate_system_combo)
        selector_layout.addRow(self._create_form_label("Segmentation mask"), self.segmentation_combo)
        selector_layout.addRow(self._create_form_label("Table"), self.table_combo)
        selector_layout.addRow(self._create_form_label("Feature matrix"), self.feature_matrix_combo)
        selector_layout.addRow(self._create_form_label("Color by"), self.color_by_combo)
        selector_layout.addRow(self._create_form_label("User class"), self.class_editor)

        content_layout.addWidget(title)
        content_layout.addLayout(selector_layout)
        content_layout.addWidget(self.retrain_action_row)
        content_layout.addWidget(self.persistence_action_row)
        content_layout.addWidget(self.refresh_action_row)
        content_layout.addWidget(self.selection_status)
        content_layout.addWidget(self.annotation_feedback)
        content_layout.addWidget(self.classifier_feedback)
        content_layout.addWidget(self.persistence_feedback)
        content_layout.addWidget(self.validation_status)
        content_layout.addStretch(1)

        self._app_state.sdata_changed.connect(self._on_sdata_changed)
        self._app_state.viewer_adapter.labels_layers_changed.connect(self._on_labels_layers_changed)
        self.refresh_from_sdata(self._app_state.sdata)

    @property
    def app_state(self) -> HarpyAppState:
        """Return the shared Harpy app state for this widget."""
        return self._app_state

    def _create_annotation_shortcuts(self) -> list[QShortcut]:
        apply_shortcut = QShortcut(QKeySequence(_APPLY_CLASS_SHORTCUT), self)
        apply_shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
        apply_shortcut.activated.connect(self._trigger_apply_class_shortcut)

        remove_shortcut = QShortcut(QKeySequence(_REMOVE_CLASS_SHORTCUT), self)
        remove_shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
        remove_shortcut.activated.connect(self._trigger_clear_class_shortcut)

        return [apply_shortcut, remove_shortcut]

    def _create_header_logo(self) -> QLabel:
        logo_label = QLabel()
        logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        logo_pixmap = QPixmap(str(self._logo_path))
        if not logo_pixmap.isNull():
            logo_label.setPixmap(logo_pixmap.scaledToWidth(120, Qt.TransformationMode.SmoothTransformation))
            return logo_label

        logo_label.setText("napari-harpy")
        logo_label.setStyleSheet("font-size: 18px; font-weight: 600;")
        return logo_label

    def _create_form_label(self, text: str) -> QLabel:
        return create_form_label(text)

    def _set_tooltip(self, widget: QWidget, message: str) -> None:
        widget.setToolTip(format_tooltip(message))

    @property
    def selected_segmentation_name(self) -> str | None:
        """Return the currently selected labels element name."""
        return None if self._selected_label_option is None else self._selected_label_option.label_name

    @property
    def selected_spatialdata(self) -> SpatialData | None:
        """Return the loaded SpatialData object backing the current widget state."""
        return self._app_state.sdata

    @property
    def selected_coordinate_system(self) -> str | None:
        """Return the currently selected coordinate system."""
        return self._selected_coordinate_system

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

        return get_table_metadata(self.selected_spatialdata, self.selected_table_name)

    def refresh_from_sdata(self, sdata: SpatialData | None) -> None:
        """Refresh the widget from the shared Harpy SpatialData state."""
        self._update_refresh_button_state(sdata)
        if sdata is None:
            self._clear_selection_inputs()
            self._bind_current_selection()
            return

        self._refresh_coordinate_systems()
        self._refresh_label_options()
        self._refresh_table_names()
        self._prepare_selected_labels_layer()
        self._bind_current_selection(classifier_dirty_reason="the segmentation selection changed")

    def _on_sdata_changed(self, sdata: SpatialData | None) -> None:
        self.refresh_from_sdata(sdata)

    def _refresh_label_options(self) -> None:
        """Refresh segmentation choices from the selected coordinate system."""
        previous_identity = None if self._selected_label_option is None else self._selected_label_option.identity
        if self.selected_spatialdata is None or self.selected_coordinate_system is None:
            self._label_options = []
        else:
            self._label_options = get_spatialdata_label_options_for_coordinate_system_from_sdata(
                sdata=self.selected_spatialdata,
                coordinate_system=self.selected_coordinate_system,
            )

        with QSignalBlocker(self.segmentation_combo):
            self.segmentation_combo.clear()
            for option in self._label_options:
                self.segmentation_combo.addItem(option.display_name)

            has_options = bool(self._label_options)
            self.segmentation_combo.setEnabled(has_options)

            # If the previously selected label is still available after a refresh,
            # keep it selected instead of resetting the user back to the first item.
            # When nothing was selected yet, stay explicitly unbound so opening the
            # widget does not auto-load or auto-bind the first segmentation.
            next_index = self._find_label_option_index(previous_identity)
            if has_options:
                self.segmentation_combo.setCurrentIndex(-1 if next_index is None else next_index)
            else:
                self.segmentation_combo.setCurrentIndex(-1)

        self._set_selected_label_option(self.segmentation_combo.currentIndex())

    def _refresh_from_current_app_state(self) -> None:
        self.refresh_from_sdata(self._app_state.sdata)

    def _update_refresh_button_state(self, sdata: SpatialData | None) -> None:
        self.refresh_button.setEnabled(self._viewer is not None and sdata is not None)

    def _clear_selection_inputs(self) -> None:
        self._coordinate_systems = []
        self._selected_coordinate_system = None
        self._label_options = []
        self._selected_label_option = None
        self._labels_layer_preparation_message = None
        self._labels_layer_preparation_error = None
        self._table_names = []
        self._selected_table_name = None
        self._table_binding_error = None
        self._feature_matrix_keys = []
        self._selected_feature_key = None

        with QSignalBlocker(self.coordinate_system_combo):
            self.coordinate_system_combo.clear()
            self.coordinate_system_combo.setEnabled(False)
            self.coordinate_system_combo.setCurrentIndex(-1)

        with QSignalBlocker(self.segmentation_combo):
            self.segmentation_combo.clear()
            self.segmentation_combo.setEnabled(False)
            self.segmentation_combo.setCurrentIndex(-1)

        with QSignalBlocker(self.table_combo):
            self.table_combo.clear()
            self.table_combo.setEnabled(False)
            self.table_combo.setCurrentIndex(-1)

        with QSignalBlocker(self.feature_matrix_combo):
            self.feature_matrix_combo.clear()
            self.feature_matrix_combo.setEnabled(False)
            self.feature_matrix_combo.setCurrentIndex(-1)

        self._set_annotation_feedback("")
        self._set_classifier_feedback("")
        self._set_persistence_feedback("")

    def _on_labels_layers_changed(self) -> None:
        if self._is_preparing_labels_layer:
            return
        # A labels-layer insert/remove only changes live viewer availability,
        # not the shared SpatialData selection model. React narrow here:
        # clear the current segmentation if *its* live layer disappeared, or
        # rebind only if the selected segmentation was previously missing and
        # has now become available.
        self._labels_layer_preparation_message = None
        self._labels_layer_preparation_error = None
        if self._selected_segmentation_layer_was_removed():
            self._clear_selected_segmentation()
            self._bind_current_selection()
            return
        # If the form still points at a selected segmentation but the
        # controllers are currently unbound (for example because no live labels
        # layer was available a moment ago), then a newly inserted matching
        # labels layer should rebind the controllers. This lets the widget
        # recover when the selected segmentation becomes available again
        # without forcing a full auto-load pass on every labels-layer change.
        if self._selected_segmentation_layer_became_available():
            self._bind_current_selection()

    def _selected_segmentation_layer_was_removed(self) -> bool:
        if (
            self.selected_spatialdata is None
            or self.selected_segmentation_name is None
            or self.selected_coordinate_system is None
        ):
            return False

        loaded_layer = self._app_state.viewer_adapter.get_loaded_labels_layer(
            self.selected_spatialdata,
            self.selected_segmentation_name,
            self.selected_coordinate_system,
        )
        return loaded_layer is None

    def _selected_segmentation_layer_became_available(self) -> bool:
        if (
            self.selected_spatialdata is None
            or self.selected_segmentation_name is None
            or self.selected_coordinate_system is None
            # This helper is only for the "selected in the form, but not
            # currently bound to a live labels layer" case. If annotation is
            # already bound to some labels layer, then nothing has "become
            # available" from the controller's perspective, so we can return early
            # without rebinding.
            or self._annotation_controller.labels_layer is not None
        ):
            return False

        loaded_layer = self._app_state.viewer_adapter.get_loaded_labels_layer(
            self.selected_spatialdata,
            self.selected_segmentation_name,
            self.selected_coordinate_system,
        )
        return loaded_layer is not None

    def _clear_selected_segmentation(self) -> None:
        with QSignalBlocker(self.segmentation_combo):
            self.segmentation_combo.setCurrentIndex(-1)

        self._set_selected_label_option(-1)
        self._refresh_table_names()

    def _refresh_coordinate_systems(self) -> None:
        previous_coordinate_system = self.selected_coordinate_system

        if self.selected_spatialdata is None:
            self._coordinate_systems = []
        else:
            self._coordinate_systems = get_coordinate_system_names_from_sdata(self.selected_spatialdata)

        with QSignalBlocker(self.coordinate_system_combo):
            self.coordinate_system_combo.clear()
            for coordinate_system in self._coordinate_systems:
                self.coordinate_system_combo.addItem(coordinate_system, coordinate_system)

            has_coordinate_systems = bool(self._coordinate_systems)
            self.coordinate_system_combo.setEnabled(has_coordinate_systems)

            next_index = (
                -1
                if previous_coordinate_system is None
                else self.coordinate_system_combo.findData(previous_coordinate_system)
            )
            if has_coordinate_systems:
                self.coordinate_system_combo.setCurrentIndex(0 if next_index < 0 else next_index)
            else:
                self.coordinate_system_combo.setCurrentIndex(-1)

        self._set_selected_coordinate_system(self.coordinate_system_combo.currentIndex())

    def _on_coordinate_system_changed(self, index: int) -> None:
        self._set_selected_coordinate_system(index)
        self._refresh_label_options()
        self._refresh_table_names()
        self._prepare_selected_labels_layer()
        self._bind_current_selection(classifier_dirty_reason="the segmentation selection changed")

    def _set_selected_coordinate_system(self, index: int) -> None:
        coordinate_system = self.coordinate_system_combo.itemData(index)
        self._selected_coordinate_system = coordinate_system if isinstance(coordinate_system, str) else None

    def _on_segmentation_changed(self, index: int) -> None:
        self._set_selected_label_option(index)
        self._refresh_table_names()
        self._prepare_selected_labels_layer()
        self._bind_current_selection(classifier_dirty_reason="the segmentation selection changed")

    def _set_selected_label_option(self, index: int) -> None:
        if index < 0 or index >= len(self._label_options):
            self._selected_label_option = None
        else:
            self._selected_label_option = self._label_options[index]

    def _find_label_option_index(self, identity: tuple[int, str] | None) -> int | None:
        if identity is None:
            return None

        for index, option in enumerate(self._label_options):
            if option.identity == identity:
                return index

        return None

    def _prepare_selected_labels_layer(self) -> None:
        self._labels_layer_preparation_message = None
        self._labels_layer_preparation_error = None

        if (
            self.selected_spatialdata is None
            or self.selected_segmentation_name is None
            or self.selected_coordinate_system is None
        ):
            return

        self._is_preparing_labels_layer = True
        try:
            existing_layer = self._app_state.viewer_adapter.get_loaded_labels_layer(
                self.selected_spatialdata,
                self.selected_segmentation_name,
                self.selected_coordinate_system,
            )
            if existing_layer is not None:
                if self._app_state.viewer_adapter.layer_bindings.get_binding(existing_layer) is None:
                    self._app_state.viewer_adapter.register_layer(
                        existing_layer,
                        sdata=self.selected_spatialdata,
                        element_name=self.selected_segmentation_name,
                        element_type="labels",
                        coordinate_system=self.selected_coordinate_system,
                    )
                activated = self._app_state.viewer_adapter.activate_layer(existing_layer)
                if activated:
                    self._labels_layer_preparation_message = (
                        f"Activated segmentation `{self.selected_segmentation_name}` in coordinate system "
                        f"`{self.selected_coordinate_system}`."
                    )
                return

            try:
                layer = self._app_state.viewer_adapter.ensure_labels_loaded(
                    self.selected_spatialdata,
                    self.selected_segmentation_name,
                    self.selected_coordinate_system,
                )
            except ValueError as error:
                self._labels_layer_preparation_error = str(error)
                return

            self._app_state.viewer_adapter.activate_layer(layer)
            self._labels_layer_preparation_message = (
                f"Loaded segmentation `{self.selected_segmentation_name}` in coordinate system "
                f"`{self.selected_coordinate_system}`."
            )
        finally:
            self._is_preparing_labels_layer = False

    def _refresh_table_names(self) -> None:
        previous_table_name = self.selected_table_name

        if self.selected_spatialdata is None or self.selected_segmentation_name is None:
            self._table_names = []
        else:
            self._table_names = get_annotating_table_names(self.selected_spatialdata, self.selected_segmentation_name)

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
            self._feature_matrix_keys = get_table_obsm_keys(self.selected_spatialdata, self.selected_table_name)

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
        effective_table_name = None if self._table_binding_error is not None else self.selected_table_name
        classifier_context_changed = self._classifier_controller.bind(
            self.selected_spatialdata,
            self.selected_segmentation_name,
            effective_table_name,
            self.selected_feature_key,
        )
        if classifier_context_changed and effective_table_name is not None:
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

    def _validate_selected_table_binding(self) -> str | None:
        if (
            self.selected_spatialdata is None
            or self.selected_segmentation_name is None
            or self.selected_table_name is None
        ):
            return None

        try:
            validate_table_binding(self.selected_spatialdata, self.selected_segmentation_name, self.selected_table_name)
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
        effective_table_name = None if self._table_binding_error is not None else self.selected_table_name

        self._annotation_controller.bind(
            self.selected_spatialdata,
            self.selected_segmentation_name,
            effective_table_name,
            self.selected_coordinate_system,
        )
        classifier_context_changed = self._classifier_controller.bind(
            self.selected_spatialdata,
            self.selected_segmentation_name,
            effective_table_name,
            self.selected_feature_key,
        )
        if classifier_dirty_reason is not None and classifier_context_changed and effective_table_name is not None:
            self._classifier_controller.mark_dirty(reason=classifier_dirty_reason)
        self._viewer_styling_controller.bind(
            self.selected_spatialdata,
            self.selected_segmentation_name,
            effective_table_name,
            self.selected_coordinate_system,
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
        self._update_primary_status_card()
        self._update_annotation_controls()
        self._update_color_by_controls()
        self._update_classifier_controls()
        self._update_persistence_controls()

    def _update_validation_status(self) -> None:
        message = None

        if self.selected_table_name is not None and self.feature_matrix_combo.count() == 0:
            message = (
                "Warning: the selected table does not contain any feature matrices in `.obsm`. "
                "Add one before continuing."
            )

        self.validation_status.setText("" if message is None else message)
        self.validation_status.setVisible(message is not None)

    def _update_primary_status_card(self) -> None:
        labels_layer = self._annotation_controller.labels_layer
        missing_table_row_message = self._annotation_controller.missing_table_row_message
        layer_preparation_lines = (
            [] if self._labels_layer_preparation_message is None else [self._labels_layer_preparation_message]
        )

        if self._app_state.sdata is None:
            self._set_selection_status(
                title="No SpatialData Loaded",
                lines=[
                    "Load a SpatialData object through the Harpy Viewer widget, reader, or `Interactive(sdata)`.",
                    "This form updates automatically from the shared Harpy state.",
                ],
                kind="warning",
            )
        elif self.selected_coordinate_system is None:
            self._set_selection_status(
                title="Selection",
                lines=["Choose a coordinate system to continue configuring object classification."],
                kind="info",
            )
        elif self.selected_segmentation_name is None:
            self._set_selection_status(
                title="Selection",
                lines=["Choose a segmentation mask in the selected coordinate system to enable object picking."],
                kind="info",
            )
        elif self._labels_layer_preparation_error is not None:
            self._set_selection_status(
                title="Layer Load Issue",
                lines=[self._labels_layer_preparation_error],
                kind="warning",
            )
        elif labels_layer is None:
            self._set_selection_status(
                title="Selection",
                lines=[
                    "The chosen segmentation is known in SpatialData for the selected coordinate system, "
                    "but is not currently loaded as a napari Labels layer."
                ],
                kind="warning",
            )
        elif self.selected_table_name is None:
            self._set_selection_status(
                title="Selection Warning",
                lines=[
                    *layer_preparation_lines,
                    f"Bound to {self.selected_segmentation_name}.",
                    "This segmentation is loaded, but no annotation table is linked to it.",
                ],
                kind="warning",
            )
        elif self._table_binding_error is not None:
            self._set_selection_status(
                title="Selection Warning",
                lines=[
                    *layer_preparation_lines,
                    f"Bound to {self.selected_segmentation_name}.",
                    self._table_binding_error,
                ],
                kind="warning",
            )
        elif self.selected_instance_id is None:
            self._set_selection_status(
                title="Selection",
                lines=[
                    *layer_preparation_lines,
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
        has_table = self.selected_table_name is not None and self._table_binding_error is None
        current_user_class = self._annotation_controller.current_user_class
        can_apply = self._annotation_controller.can_annotate
        can_clear = can_apply and current_user_class not in (None, UNLABELED_CLASS)

        self.class_spinbox.setEnabled(has_table)
        self.apply_class_button.setEnabled(can_apply)
        self.clear_class_button.setEnabled(can_clear)

        self._set_tooltip(
            self.apply_class_button,
            self._annotation_action_tooltip(
                enabled=can_apply,
                ready_message="Assign the selected user class to the picked object.",
                unavailable_message="Pick an annotated object in the viewer before applying a class.",
                shortcut_hint=_APPLY_CLASS_SHORTCUT,
            ),
        )
        self._set_tooltip(
            self.clear_class_button,
            self._annotation_action_tooltip(
                enabled=can_clear,
                ready_message="Clear the current user class for the picked object.",
                unavailable_message="Pick a labeled object before clearing its user class.",
                shortcut_hint=_REMOVE_CLASS_SHORTCUT,
            ),
        )

    def _annotation_action_tooltip(
        self,
        *,
        enabled: bool,
        ready_message: str,
        unavailable_message: str,
        shortcut_hint: str,
    ) -> str:
        message = ready_message if enabled else unavailable_message
        return f"{message} Shortcut: {shortcut_hint}."

    def _trigger_apply_class_shortcut(self) -> None:
        if not self.apply_class_button.isEnabled():
            return

        self._apply_current_class()

    def _trigger_clear_class_shortcut(self) -> None:
        if not self.clear_class_button.isEnabled():
            return

        self._clear_current_class()

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
            sync_tooltip = "Choose a backed SpatialData annotation table to enable writing the in-memory table state to disk."
            reload_tooltip = (
                "Choose a backed SpatialData annotation table to enable discarding the current in-memory table state "
                "and reloading it from disk."
            )
        elif self._table_binding_error is not None:
            sync_tooltip = self._table_binding_error
            reload_tooltip = self._table_binding_error
        elif not can_sync or not can_reload:
            sync_tooltip = "The selected SpatialData dataset is not backed by zarr, so the in-memory table state cannot be written to disk."
            reload_tooltip = "The selected SpatialData dataset is not backed by zarr, so the table state cannot be reloaded from disk."
        else:
            table_store_path = self._persistence_controller.selected_table_store_path
            destination = self.selected_spatialdata.path if table_store_path is None else table_store_path
            sync_tooltip = f"Write the current in-memory `{self.selected_table_name}` table state to `{destination}`."
            reload_tooltip = (
                f"Discard the current in-memory `{self.selected_table_name}` table state and reload it from "
                f"`{destination}`."
            )
            if self._persistence_controller.is_dirty:
                sync_tooltip += " Unsynced local in-memory table changes are present."
                reload_tooltip += " Unsynced local in-memory table changes would be discarded."

        self._set_tooltip(self.sync_button, sync_tooltip)
        self._set_tooltip(self.reload_button, reload_tooltip)

    def _update_color_by_controls(self) -> None:
        has_table = self.selected_table_name is not None and self._table_binding_error is None
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

        self._set_tooltip(self.color_by_combo, tooltip)

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

        self._set_tooltip(self.retrain_button, tooltip)

    def _write_to_zarr(self) -> None:
        # TODO: consider disabling write while classifier retraining is pending
        # so "Write Table to zarr" always snapshots a settled table state.
        self._write_selected_table_to_zarr()

    def _write_selected_table_to_zarr(
        self,
        *,
        show_feedback: bool = True,
        feedback_message: str | None = None,
    ) -> bool:
        try:
            self._persistence_controller.write_table_state()
        except ValueError as error:
            self._set_persistence_feedback(str(error), error=True)
            return False

        if show_feedback:
            destination = self._selected_table_store_destination()
            message = feedback_message or f"Wrote `{self.selected_table_name}` table state to `{destination}`."
            self._set_persistence_feedback(message, error=False)
        self._update_selection_status()
        return True

    def _reload_from_zarr(self) -> None:
        if not self._persistence_controller.is_dirty:
            self._reload_selected_table_from_zarr()
            return

        decision = self._prompt_dirty_reload_decision()
        if decision is _DirtyReloadDecision.CANCEL:
            return

        if decision is _DirtyReloadDecision.WRITE:
            if not self._write_selected_table_to_zarr(show_feedback=False):
                return

            source = self._selected_table_store_destination()
            self._reload_selected_table_from_zarr(
                feedback_message=f"Wrote local changes and reloaded `{self.selected_table_name}` table state from `{source}`.",
            )
            return

        if decision is _DirtyReloadDecision.RELOAD_DISCARD:
            self._reload_selected_table_from_zarr()
            return

        raise RuntimeError(f"Unhandled dirty reload decision: {decision!r}")

    def _reload_selected_table_from_zarr(self, *, feedback_message: str | None = None) -> bool:
        self._classifier_controller.freeze_for_reload()
        try:
            # For now the persistence layer reports expected reload failures as
            # `ValueError` (selection/precondition issues, reload validation
            # failures, and similar user-facing problems). A future cleanup may
            # replace this broad catch with a dedicated reload error type once
            # the persistence-layer error boundary is formalized.
            self._persistence_controller.reload_table_state()
        except ValueError as error:
            self._set_persistence_feedback(str(error), error=True)
            return False

        self._refresh_feature_matrix_keys()
        self._bind_current_selection()
        self._classifier_controller.reset_after_reload()
        source = self._selected_table_store_destination()
        message = feedback_message or f"Reloaded `{self.selected_table_name}` table state from `{source}`."
        self._set_persistence_feedback(message, error=False)
        return True

    def _prompt_dirty_reload_decision(self) -> _DirtyReloadDecision:
        table_name = self.selected_table_name
        dialog = QDialog(self)
        dialog.setWindowTitle("Unsynced Table Changes")
        dialog.setModal(True)
        dialog.setMinimumWidth(560)

        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(14)

        warning_message = (
            f"Table `{escape(table_name)}` has in-memory changes that have not been written to zarr."
            if table_name is not None
            else "The selected table has in-memory changes that have not been written to zarr."
        )
        warning_card = QLabel(
            "<div>"
            "<span style='font-size: 11px; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase;'>"
            "Unsynced Changes</span><br>"
            f"<span>{warning_message}</span>"
            "</div>"
        )
        warning_card.setWordWrap(True)
        warning_card.setStyleSheet(
            "font-weight: 500; "
            "color: #b45309; "
            "background-color: #fff7ed; "
            "border: 1px solid #fdba74; "
            "border-radius: 10px; "
            "padding: 12px 14px;"
        )
        layout.addWidget(warning_card)

        button_row = QHBoxLayout()
        button_row.setSpacing(10)
        button_row.addStretch(1)
        write_button = QPushButton("Write local edits and reload")
        discard_button = QPushButton("Reload and discard local edits")
        cancel_button = QPushButton("Cancel")

        write_button.setStyleSheet(
            "QPushButton {"
            "background-color: #166534; "
            "color: white; "
            "border: 1px solid #166534; "
            "border-radius: 6px; "
            "padding: 7px 14px; "
            "font-weight: 600;}"
            "QPushButton:hover { background-color: #15803d; border-color: #15803d; }"
        )
        discard_button.setStyleSheet(
            "QPushButton {"
            "background-color: #fff7ed; "
            "color: #9a3412; "
            "border: 1px solid #fdba74; "
            "border-radius: 6px; "
            "padding: 7px 14px; "
            "font-weight: 600;}"
            "QPushButton:hover { background-color: #ffedd5; }"
        )
        cancel_button.setStyleSheet(
            "QPushButton {"
            "background-color: #f9fafb; "
            "color: #111827; "
            "border: 1px solid #d1d5db; "
            "border-radius: 6px; "
            "padding: 7px 14px;}"
            "QPushButton:hover { background-color: #f3f4f6; }"
        )

        button_row.addWidget(write_button)
        button_row.addWidget(discard_button)
        button_row.addWidget(cancel_button)
        layout.addLayout(button_row)

        write_button.clicked.connect(lambda: dialog.done(1))
        discard_button.clicked.connect(lambda: dialog.done(2))
        cancel_button.clicked.connect(dialog.reject)
        cancel_button.setDefault(True)

        result = dialog.exec()
        if result == 1:
            return _DirtyReloadDecision.WRITE
        if result == 2:
            return _DirtyReloadDecision.RELOAD_DISCARD
        return _DirtyReloadDecision.CANCEL

    def _set_persistence_feedback(self, message: str, *, error: bool = False) -> None:
        if not message:
            self.persistence_feedback.setText("")
            self.persistence_feedback.setVisible(False)
            return

        kind = "error" if error else "success"
        title = "Persistence Error" if error else "Persistence Updated"
        self._set_status_card(
            self.persistence_feedback,
            title=title,
            lines=[message],
            kind=kind,
        )

    def _on_selected_instance_changed(self, instance_id: int | None) -> None:
        del instance_id
        self._set_annotation_feedback("")
        self._update_primary_status_card()
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

    def _selected_table_store_destination(self) -> Path | str | None:
        table_store_path = self._persistence_controller.selected_table_store_path
        if table_store_path is None or self.selected_spatialdata is None:
            return None if self.selected_spatialdata is None else self.selected_spatialdata.path

        return table_store_path
