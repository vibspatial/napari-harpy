from __future__ import annotations

from html import escape
from pathlib import Path
from typing import TYPE_CHECKING

from qtpy.QtCore import QSignalBlocker, Qt
from qtpy.QtGui import QColor, QPalette, QPixmap
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from napari_harpy._feature_extraction import FeatureExtractionController
from napari_harpy._spatialdata import (
    SpatialDataImageOption,
    SpatialDataLabelsOption,
    SpatialDataViewerBinding,
    get_annotating_table_names,
    validate_table_binding,
)

if TYPE_CHECKING:
    import napari
    from spatialdata import SpatialData


_WIDGET_SURFACE_COLOR = "#fcf6f3"
_WIDGET_SURFACE_STYLESHEET = f"background-color: {_WIDGET_SURFACE_COLOR};"
_TOOLTIP_TEXT_COLOR = "#111827"
_FORM_LABEL_STYLESHEET = "color: #374151; font-weight: 600; padding-top: 6px;"
_INPUT_CONTROL_STYLESHEET = (
    "QComboBox, QLineEdit {"
    "background-color: #fffdfb; "
    "border: 1px solid #ddcfc7; "
    "border-radius: 8px; "
    "color: #111827; "
    "padding: 4px 10px; "
    "min-height: 30px;}"
    "QComboBox:disabled, QLineEdit:disabled {"
    "background-color: #f7efea; "
    "border-color: #e9ddd7; "
    "color: #9ca3af;}"
    "QComboBox:focus, QLineEdit:focus {"
    "border-color: #8fb6c9; "
    "background-color: #ffffff;}"
    "QComboBox { padding-right: 24px; }"
    "QComboBox::drop-down {"
    "subcontrol-origin: padding; "
    "subcontrol-position: top right; "
    "width: 24px; "
    "border: 0px; "
    "background: transparent;}"
)
_ACTION_BUTTON_STYLESHEET = (
    "QPushButton {"
    "background-color: #f7ede8; "
    "border: 1px solid #ddcfc7; "
    "border-radius: 8px; "
    "color: #111827; "
    "font-weight: 600; "
    "padding: 4px 10px; "
    "min-height: 30px;}"
    "QPushButton:hover { background-color: #f3e5de; border-color: #c9b6ac; }"
    "QPushButton:pressed { background-color: #ebd7cf; border-color: #b59a8e; }"
    "QPushButton:disabled { background-color: #faf4f1; border-color: #ede3dd; color: #a8a29e; }"
)
_FEATURE_GROUP_STYLESHEET = (
    "QGroupBox {"
    "background-color: #f8eeea; "
    "border: 1px solid #eadfd8; "
    "border-radius: 10px; "
    "color: #374151; "
    "font-weight: 600; "
    "margin-top: 10px; "
    "padding: 12px 12px 10px 12px;}"
    "QGroupBox::title {"
    "subcontrol-origin: margin; "
    "left: 12px; "
    f"padding: 0 6px; color: #374151; background-color: {_WIDGET_SURFACE_COLOR};"
    "}"
)
_FEATURE_CHECKBOX_STYLESHEET = (
    "QCheckBox {"
    "color: #111827; "
    "font-weight: 500; "
    "spacing: 8px; "
    "background: transparent;}"
    "QCheckBox:disabled { color: #9ca3af; }"
    "QCheckBox::indicator {"
    "width: 16px; "
    "height: 16px; "
    "border-radius: 4px; "
    "border: 1px solid #d8c8bf; "
    "background-color: #fffdfb;}"
    "QCheckBox::indicator:hover { border-color: #c7b2a7; }"
    "QCheckBox::indicator:checked {"
    "border-color: #7aa7bd; "
    "background-color: #8fb6c9;}"
    "QCheckBox::indicator:disabled {"
    "border-color: #e9ddd7; "
    "background-color: #f7efea;}"
)
_FEATURE_HINT_INFO_STYLESHEET = "color: #6b7280; font-size: 12px; font-weight: 500;"
_FEATURE_HINT_WARNING_STYLESHEET = "color: #b45309; font-size: 12px; font-weight: 600;"
_NO_IMAGE_TEXT = "No image"
_INTENSITY_FEATURES = ("sum", "mean", "var", "min", "max", "kurtosis", "skew")
_MORPHOLOGY_FEATURES = (
    "area",
    "eccentricity",
    "major_axis_length",
    "minor_axis_length",
    "perimeter",
    "convex_area",
    "equivalent_diameter",
    "major_minor_axis_ratio",
    "perim_square_over_area",
    "major_axis_equiv_diam_ratio",
    "convex_hull_resid",
    "centroid_dif",
)
_WIDGET_MIN_WIDTH = 370


class FeatureExtractionWidget(QWidget):
    """
    Widget for feature extraction.

    The widget discovers selectable labels and images from viewer-linked
    `SpatialData` objects and keeps the selection flow dataset-centric:

    - labels come from viewer-linked `sdata.labels`
    - images come from the selected dataset's `sdata.images`
    - tables are restricted to annotators of the selected labels element
    - coordinate systems come from the selected labels/image context
    """

    def __init__(self, napari_viewer: napari.Viewer | None = None) -> None:
        super().__init__()
        self.setObjectName("feature_extraction_widget")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor(_WIDGET_SURFACE_COLOR))
        self.setPalette(palette)
        self.setStyleSheet(_WIDGET_SURFACE_STYLESHEET)
        self.setMinimumWidth(_WIDGET_MIN_WIDTH)

        self._viewer = napari_viewer
        self._viewer_binding = SpatialDataViewerBinding(napari_viewer)
        self._feature_extraction_controller = FeatureExtractionController(
            on_state_changed=self._on_controller_state_changed,
            on_table_state_changed=self._on_controller_table_state_changed,
        )

        self._label_options: list[SpatialDataLabelsOption] = []
        self._selected_label_option: SpatialDataLabelsOption | None = None
        self._image_options: list[SpatialDataImageOption] = []
        self._selected_image_option: SpatialDataImageOption | None = None
        self._table_names: list[str] = []
        self._selected_table_name: str | None = None
        self._coordinate_systems: list[str] = []
        self._selected_coordinate_system: str | None = None
        self._table_binding_error: str | None = None
        self._feature_checkboxes: dict[str, QCheckBox] = {}
        self._logo_path = Path(__file__).resolve().parents[3] / "docs" / "_static" / "logo.png"

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.scroll_area = QScrollArea()
        self.scroll_area.setObjectName("feature_extraction_scroll_area")
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll_area.setStyleSheet("QScrollArea { border: 0px; background: transparent; }")

        self.scroll_content = QWidget()
        self.scroll_content.setObjectName("feature_extraction_scroll_content")
        self.scroll_content.setAutoFillBackground(True)
        scroll_palette = self.scroll_content.palette()
        scroll_palette.setColor(QPalette.ColorRole.Window, QColor(_WIDGET_SURFACE_COLOR))
        self.scroll_content.setPalette(scroll_palette)
        self.scroll_content.setStyleSheet(_WIDGET_SURFACE_STYLESHEET)

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

        self.segmentation_combo = QComboBox()
        self.segmentation_combo.setObjectName("feature_extraction_segmentation_combo")
        self.segmentation_combo.currentIndexChanged.connect(self._on_segmentation_changed)
        self.segmentation_combo.setStyleSheet(_INPUT_CONTROL_STYLESHEET)

        self.image_combo = QComboBox()
        self.image_combo.setObjectName("feature_extraction_image_combo")
        self.image_combo.currentIndexChanged.connect(self._on_image_changed)
        self.image_combo.setStyleSheet(_INPUT_CONTROL_STYLESHEET)

        self.table_combo = QComboBox()
        self.table_combo.setObjectName("feature_extraction_table_combo")
        self.table_combo.currentIndexChanged.connect(self._on_table_changed)
        self.table_combo.setStyleSheet(_INPUT_CONTROL_STYLESHEET)

        self.coordinate_system_combo = QComboBox()
        self.coordinate_system_combo.setObjectName("feature_extraction_coordinate_system_combo")
        self.coordinate_system_combo.currentIndexChanged.connect(self._on_coordinate_system_changed)
        self.coordinate_system_combo.setStyleSheet(_INPUT_CONTROL_STYLESHEET)
        self._set_tooltip(
            self.coordinate_system_combo,
            "Choose a coordinate system in which the segmentation mask and image are registered. In the selected coordinate system they should have the same shape, and only an identity transform, a translation, or a sequence of translations may be defined relative to it.",
        )

        self.output_key_line_edit = QLineEdit()
        self.output_key_line_edit.setObjectName("feature_extraction_output_key_line_edit")
        self.output_key_line_edit.setPlaceholderText("e.g. features")
        self.output_key_line_edit.textChanged.connect(self._on_feature_key_changed)
        self.output_key_line_edit.setStyleSheet(_INPUT_CONTROL_STYLESHEET)
        self._set_tooltip(
            self.output_key_line_edit,
            'This key names the feature matrix created by the Feature Extraction widget. It will appear in the Object Classification widget under "Feature matrix". The matrix is stored in the selected table as `.obsm[output_key]`, with companion metadata in `.uns["feature_matrices"][output_key]`.',
        )

        self.intensity_features_group = self._create_feature_group(
            "Intensity Features",
            _INTENSITY_FEATURES,
            object_name="feature_extraction_intensity_features_group",
        )
        self.intensity_features_hint = QLabel()
        self.intensity_features_hint.setObjectName("feature_extraction_intensity_features_hint")
        self.intensity_features_hint.setWordWrap(True)
        self.intensity_features_group.layout().addWidget(self.intensity_features_hint)

        self.morphology_features_group = self._create_feature_group(
            "Morphology Features",
            _MORPHOLOGY_FEATURES,
            object_name="feature_extraction_morphology_features_group",
        )

        self.refresh_action_row = QWidget()
        self.refresh_action_row.setObjectName("feature_extraction_refresh_action_row")
        refresh_action_layout = QHBoxLayout(self.refresh_action_row)
        refresh_action_layout.setContentsMargins(0, 0, 0, 0)
        refresh_action_layout.setSpacing(8)

        self.calculate_action_row = QWidget()
        self.calculate_action_row.setObjectName("feature_extraction_calculate_action_row")
        calculate_action_layout = QHBoxLayout(self.calculate_action_row)
        calculate_action_layout.setContentsMargins(0, 0, 0, 0)
        calculate_action_layout.setSpacing(8)

        self.refresh_button = QPushButton("Rescan Viewer")
        self.refresh_button.clicked.connect(self.refresh_segmentation_masks)
        self.refresh_button.setEnabled(napari_viewer is not None)
        self.refresh_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.refresh_button.setMinimumHeight(28)
        self.refresh_button.setStyleSheet(_ACTION_BUTTON_STYLESHEET)

        self.calculate_button = QPushButton("Calculate")
        self.calculate_button.setObjectName("feature_extraction_calculate_button")
        self.calculate_button.setEnabled(False)
        self.calculate_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.calculate_button.setMinimumHeight(28)
        self.calculate_button.setStyleSheet(_ACTION_BUTTON_STYLESHEET)
        self._set_tooltip(
            self.calculate_button,
            "Calculation will be enabled once the feature-extraction controller is wired to these inputs.",
        )

        self.validation_status = QLabel()
        self.validation_status.setObjectName("feature_extraction_validation_status")
        self.validation_status.setWordWrap(True)
        self.validation_status.setStyleSheet("color: #b45309; font-weight: 600;")
        self.validation_status.hide()

        self.selection_status = QLabel()
        self.selection_status.setObjectName("feature_extraction_selection_status")
        self.selection_status.setWordWrap(True)
        self.selection_status.hide()

        self.controller_feedback = QLabel()
        self.controller_feedback.setObjectName("feature_extraction_controller_feedback")
        self.controller_feedback.setWordWrap(True)
        self.controller_feedback.hide()

        selector_layout.addRow(self._create_form_label("Segmentation mask"), self.segmentation_combo)
        selector_layout.addRow(self._create_form_label("Image"), self.image_combo)
        selector_layout.addRow(self._create_form_label("Table"), self.table_combo)
        selector_layout.addRow(self._create_form_label("Coordinate system"), self.coordinate_system_combo)
        selector_layout.addRow(self._create_form_label("Feature matrix key"), self.output_key_line_edit)

        refresh_action_layout.addWidget(self.refresh_button, 1)
        calculate_action_layout.addWidget(self.calculate_button, 1)

        content_layout.addWidget(title)
        content_layout.addLayout(selector_layout)
        content_layout.addWidget(self.intensity_features_group)
        content_layout.addWidget(self.morphology_features_group)
        content_layout.addWidget(self.calculate_action_row)
        content_layout.addWidget(self.refresh_action_row)
        content_layout.addWidget(self.selection_status)
        content_layout.addWidget(self.controller_feedback)
        content_layout.addWidget(self.validation_status)
        content_layout.addStretch(1)

        self._connect_viewer_events()
        self._update_intensity_features_hint()
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
    def selected_image_name(self) -> str | None:
        """Return the currently selected image element name, if any."""
        return None if self._selected_image_option is None else self._selected_image_option.image_name

    @property
    def selected_table_name(self) -> str | None:
        """Return the currently selected annotation table name."""
        return self._selected_table_name

    @property
    def selected_coordinate_system(self) -> str | None:
        """Return the currently selected coordinate system."""
        return self._selected_coordinate_system

    @property
    def selected_feature_names(self) -> tuple[str, ...]:
        """Return selected feature names in stable UI order."""
        ordered_features = (*_INTENSITY_FEATURES, *_MORPHOLOGY_FEATURES)
        return tuple(feature for feature in ordered_features if self._feature_checkboxes[feature].isChecked())

    @property
    def selected_feature_key(self) -> str | None:
        """Return the trimmed feature-matrix output key, if any."""
        value = self.output_key_line_edit.text().strip()
        return value or None

    @property
    def overwrite_feature_key(self) -> bool:
        """Return whether an existing feature key should be replaced."""
        return False

    def refresh_segmentation_masks(self) -> None:
        """Refresh the segmentation choices from viewer-linked SpatialData layers."""
        previous_identity = None if self._selected_label_option is None else self._selected_label_option.identity
        self._label_options = self._viewer_binding.get_label_options()

        with QSignalBlocker(self.segmentation_combo):
            self.segmentation_combo.clear()
            for option in self._label_options:
                self.segmentation_combo.addItem(option.display_name)

            has_options = bool(self._label_options)
            self.segmentation_combo.setEnabled(has_options)

            next_index = self._find_label_option_index(previous_identity)
            if has_options:
                self.segmentation_combo.setCurrentIndex(0 if next_index is None else next_index)
            else:
                self.segmentation_combo.setCurrentIndex(-1)

        if self.segmentation_combo.currentIndex() >= 0:
            self._set_selected_label_option(self.segmentation_combo.currentIndex())
        else:
            self._selected_label_option = None
            self._refresh_image_options()
            self._refresh_table_names()
            self._refresh_coordinate_systems()
            self._bind_current_selection()

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
        label = QLabel(text)
        label.setStyleSheet(_FORM_LABEL_STYLESHEET)
        label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        return label

    def _create_feature_group(self, title: str, feature_names: tuple[str, ...], *, object_name: str) -> QGroupBox:
        group = QGroupBox(title)
        group.setObjectName(object_name)
        group.setStyleSheet(_FEATURE_GROUP_STYLESHEET)
        layout = QVBoxLayout(group)
        layout.setContentsMargins(12, 18, 12, 8)
        layout.setSpacing(8)

        feature_layout = QVBoxLayout()
        feature_layout.setContentsMargins(0, 0, 0, 0)
        feature_layout.setSpacing(6)
        for feature_name in feature_names:
            checkbox = QCheckBox(feature_name)
            checkbox.setObjectName(f"feature_checkbox_{feature_name}")
            checkbox.setStyleSheet(_FEATURE_CHECKBOX_STYLESHEET)
            checkbox.toggled.connect(self._on_feature_selection_changed)
            self._feature_checkboxes[feature_name] = checkbox
            feature_layout.addWidget(checkbox)

        layout.addLayout(feature_layout)
        return group

    def _set_tooltip(self, widget: QWidget, message: str) -> None:
        widget.setToolTip(f"<qt><span style='color: {_TOOLTIP_TEXT_COLOR};'>{escape(message)}</span></qt>")

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

        self._refresh_image_options()
        self._refresh_table_names()
        self._refresh_coordinate_systems()
        self._bind_current_selection()

    def _find_label_option_index(self, identity: tuple[int, str] | None) -> int | None:
        if identity is None:
            return None

        for index, option in enumerate(self._label_options):
            if option.identity == identity:
                return index

        return None

    def _refresh_image_options(self) -> None:
        previous_identity = None if self._selected_image_option is None else self._selected_image_option.identity

        if self.selected_spatialdata is None or self.selected_segmentation_name is None:
            self._image_options = []
        else:
            self._image_options = self._viewer_binding.get_image_options(
                self.selected_spatialdata,
                self.selected_segmentation_name,
            )

        with QSignalBlocker(self.image_combo):
            self.image_combo.clear()
            self.image_combo.addItem(_NO_IMAGE_TEXT, None)
            for option in self._image_options:
                self.image_combo.addItem(option.display_name)

            self.image_combo.setEnabled(
                self.selected_spatialdata is not None and self.selected_segmentation_name is not None
            )

            next_index = self._find_image_option_index(previous_identity)
            if self.image_combo.count() == 1:
                self.image_combo.setCurrentIndex(0)
            elif next_index is None:
                self.image_combo.setCurrentIndex(0)
            else:
                self.image_combo.setCurrentIndex(next_index + 1)

        self._set_selected_image_option(self.image_combo.currentIndex())

    def _on_image_changed(self, index: int) -> None:
        self._set_selected_image_option(index)
        self._refresh_coordinate_systems()
        self._update_intensity_features_hint()
        self._bind_current_selection()

    def _set_selected_image_option(self, index: int) -> None:
        if index <= 0 or index > len(self._image_options):
            self._selected_image_option = None
        else:
            self._selected_image_option = self._image_options[index - 1]

    def _find_image_option_index(self, identity: tuple[int, str] | None) -> int | None:
        if identity is None:
            return None

        for index, option in enumerate(self._image_options):
            if option.identity == identity:
                return index

        return None

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

    def _on_table_changed(self, index: int) -> None:
        self._set_selected_table_name(index)
        self._bind_current_selection()

    def _set_selected_table_name(self, index: int) -> None:
        if index < 0 or index >= len(self._table_names):
            self._selected_table_name = None
        else:
            self._selected_table_name = self._table_names[index]

    def _refresh_coordinate_systems(self) -> None:
        previous_coordinate_system = self.selected_coordinate_system

        if self.selected_spatialdata is None or self.selected_segmentation_name is None:
            self._coordinate_systems = []
        elif self.selected_image_name is None:
            self._coordinate_systems = (
                [] if self._selected_label_option is None else list(self._selected_label_option.coordinate_systems)
            )
        else:
            self._coordinate_systems = (
                [] if self._selected_image_option is None else list(self._selected_image_option.coordinate_systems)
            )

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
        self._bind_current_selection()

    def _set_selected_coordinate_system(self, index: int) -> None:
        coordinate_system = self.coordinate_system_combo.itemData(index)
        self._selected_coordinate_system = coordinate_system if isinstance(coordinate_system, str) else None

    def _on_feature_selection_changed(self, _checked: bool) -> None:
        self._update_intensity_features_hint()
        self._bind_current_selection()

    def _on_feature_key_changed(self, _text: str) -> None:
        self._bind_current_selection()

    def _update_intensity_features_hint(self) -> None:
        intensity_selected = any(self._feature_checkboxes[name].isChecked() for name in _INTENSITY_FEATURES)
        if intensity_selected and self.selected_image_name is None:
            self.intensity_features_hint.setText(
                "Intensity features are selected, so choose an image before calculating."
            )
            self.intensity_features_hint.setStyleSheet(_FEATURE_HINT_WARNING_STYLESHEET)
            self.intensity_features_hint.setVisible(True)
            return

        self.intensity_features_hint.setText("")
        self.intensity_features_hint.setStyleSheet(_FEATURE_HINT_INFO_STYLESHEET)
        self.intensity_features_hint.setVisible(False)

    def _validate_selected_table_binding(self) -> str | None:
        if (
            self.selected_spatialdata is None
            or self.selected_segmentation_name is None
            or self.selected_table_name is None
        ):
            return None

        try:
            validate_table_binding(
                self.selected_spatialdata,
                self.selected_segmentation_name,
                self.selected_table_name,
            )
        except ValueError as error:
            return str(error)

        return None

    def _bind_current_selection(self) -> None:
        self._table_binding_error = self._validate_selected_table_binding()
        effective_table_name = None if self._table_binding_error is not None else self.selected_table_name
        self._feature_extraction_controller.bind(
            self.selected_spatialdata,
            self.selected_segmentation_name,
            self.selected_image_name,
            effective_table_name,
            self.selected_coordinate_system,
            self.selected_feature_names,
            self.selected_feature_key,
            overwrite_feature_key=False,
        )
        self._update_selection_status()

    def _update_selection_status(self) -> None:
        self._update_validation_status()
        self._update_primary_status_card()
        self._update_calculate_controls()

    def _update_validation_status(self) -> None:
        message = self._table_binding_error
        self.validation_status.setText("" if message is None else message)
        self.validation_status.setVisible(message is not None)

    def _update_primary_status_card(self) -> None:
        if self.selected_spatialdata is None or self.selected_segmentation_name is None:
            self._set_selection_status(
                "Selection Needed",
                ["Choose a segmentation linked from the current viewer."],
                kind="warning",
            )
            return

        if self.selected_table_name is None:
            self._set_selection_status(
                "No Table Linked",
                [
                    f"Segmentation `{self.selected_segmentation_name}` is not linked to an annotation table.",
                    "Support for creating a new linked table from this widget is coming soon.",
                ],
                kind="warning",
            )
            return

        if self._table_binding_error is not None:
            self._set_selection_status(
                "Table Binding Issue",
                [
                    f"Table `{self.selected_table_name}` cannot currently be used for segmentation `{self.selected_segmentation_name}`.",
                    "Choose a different table or segmentation.",
                ],
                kind="warning",
            )
            return

        if self.selected_coordinate_system is None:
            self._set_selection_status(
                "Choose Coordinate System",
                ["Choose a coordinate system to continue configuring feature extraction."],
                kind="warning",
            )
            return

        image_line = (
            "Image: none selected yet" if self.selected_image_name is None else f"Image: {self.selected_image_name}"
        )
        self._set_selection_status(
            "Selection Ready",
            [
                f"Segmentation: {self.selected_segmentation_name}",
                f"Table: {self.selected_table_name}",
                image_line,
                f"Coordinate system: {self.selected_coordinate_system}",
            ],
            kind="success",
        )

    def _set_selection_status(self, title: str, lines: list[str], *, kind: str) -> None:
        self._set_status_card(self.selection_status, title=title, lines=lines, kind=kind)

    def _set_controller_feedback(self, message: str, *, kind: str = "info") -> None:
        if not message:
            self.controller_feedback.setText("")
            self.controller_feedback.setVisible(False)
            return

        title_by_kind = {
            "error": "Feature Extraction Error",
            "warning": "Feature Extraction Warning",
            "success": "Feature Extraction Ready",
            "info": "Feature Extraction",
        }
        body = message.removeprefix("Feature extraction: ").strip()
        self._set_status_card(
            self.controller_feedback,
            title=title_by_kind.get(kind, "Feature Extraction"),
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

    def _update_calculate_controls(self) -> None:
        can_calculate = self._feature_extraction_controller.can_calculate
        is_running = self._feature_extraction_controller.is_running
        self.calculate_button.setEnabled(can_calculate)

        if is_running:
            tooltip = "A feature-extraction job is currently running."
        elif can_calculate:
            tooltip = "Calculate the selected features and store them in the selected table."
        else:
            tooltip = self._feature_extraction_controller.status_message

        self._set_tooltip(self.calculate_button, tooltip)

    def _on_controller_state_changed(self) -> None:
        self._set_controller_feedback(
            self._feature_extraction_controller.status_message,
            kind=self._feature_extraction_controller.status_kind,
        )
        self._update_calculate_controls()

    def _on_controller_table_state_changed(self) -> None:
        self._refresh_table_names()
        self._bind_current_selection()
