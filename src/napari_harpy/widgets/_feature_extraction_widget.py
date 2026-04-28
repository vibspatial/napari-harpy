from __future__ import annotations

from dataclasses import dataclass, replace
from html import escape
from pathlib import Path
from typing import TYPE_CHECKING

from qtpy.QtCore import QSignalBlocker, Qt
from qtpy.QtGui import QPixmap
from qtpy.QtWidgets import (
    QCheckBox,
    QDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from napari_harpy._app_state import FeatureMatrixWrittenEvent, HarpyAppState, get_or_create_app_state
from napari_harpy._feature_extraction import FeatureExtractionController
from napari_harpy._spatialdata import (
    SpatialDataImageOption,
    SpatialDataLabelsOption,
    get_annotating_table_names,
    get_coordinate_system_names_from_sdata,
    get_image_channel_names_from_sdata,
    get_spatialdata_feature_extraction_label_options_for_coordinate_system_from_sdata,
    get_spatialdata_matching_image_options_for_coordinate_system_and_label_from_sdata,
    get_table,
    validate_table_binding,
)
from napari_harpy.widgets._shared_styles import (
    ACTION_BUTTON_STYLESHEET as _ACTION_BUTTON_STYLESHEET,
)
from napari_harpy.widgets._shared_styles import (
    CHECKBOX_STYLESHEET as _FEATURE_CHECKBOX_STYLESHEET,
)
from napari_harpy.widgets._shared_styles import (
    WIDGET_BORDER_COLOR,
    WIDGET_PANEL_COLOR,
    WIDGET_TEXT_COLOR,
    CompactComboBox,
    StatusCardKind,
    apply_scroll_content_surface,
    apply_widget_surface,
    build_input_control_stylesheet,
    create_form_label,
    format_feedback_identifier,
    format_tooltip,
    set_status_card,
)
from napari_harpy.widgets._shared_styles import (
    WIDGET_MIN_WIDTH as _WIDGET_MIN_WIDTH,
)
from napari_harpy.widgets._shared_styles import (
    WIDGET_SURFACE_COLOR as _WIDGET_SURFACE_COLOR,
)

if TYPE_CHECKING:
    import napari
    from spatialdata import SpatialData


_INPUT_CONTROL_STYLESHEET = build_input_control_stylesheet("QComboBox, QLineEdit")
_FEATURE_GROUP_STYLESHEET = (
    "QGroupBox {"
    f"background-color: {WIDGET_PANEL_COLOR}; "
    f"border: 1px solid {WIDGET_BORDER_COLOR}; "
    "border-radius: 10px; "
    f"color: {WIDGET_TEXT_COLOR}; "
    "font-weight: 700; "
    "margin-top: 10px; "
    "padding: 12px 12px 10px 12px;}"
    "QGroupBox QComboBox, QGroupBox QLineEdit {"
    "font-weight: 400;"
    "}"
    "QGroupBox::title {"
    "subcontrol-origin: margin; "
    "left: 12px; "
    f"padding: 0 6px; color: {WIDGET_TEXT_COLOR}; background-color: {_WIDGET_SURFACE_COLOR};"
    "}"
)
_FEATURE_HINT_INFO_STYLESHEET = "color: #6b7280; font-size: 12px; font-weight: 500;"
_FEATURE_HINT_WARNING_STYLESHEET = "color: #b45309; font-size: 12px; font-weight: 600;"
_CHANNEL_SELECTION_PANEL_STYLESHEET = "QWidget { background: transparent; }"
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
_DEFAULT_FEATURE_MATRIX_KEY = "features"
_MAX_VISIBLE_EXTRACTION_CHANNELS = 5
_FEATURE_GROUPS_TOP_SPACING = 12
ElementIdentity = tuple[int, str]


@dataclass(frozen=True)
class _FeatureExtractionTripletCardWidgets:
    coordinate_system: str
    container: QGroupBox
    segmentation_combo: CompactComboBox
    image_combo: CompactComboBox


@dataclass(frozen=True)
class _FeatureExtractionTripletCardState:
    coordinate_system: str | None
    label_options: list[SpatialDataLabelsOption]
    selected_label_option: SpatialDataLabelsOption | None
    image_options: list[SpatialDataImageOption]
    selected_image_option: SpatialDataImageOption | None


class FeatureExtractionWidget(QWidget):
    """
    Widget for feature extraction.

    The widget discovers selectable labels and images from the shared loaded
    `SpatialData` object and keeps the selection flow coordinate-system-first:

    - coordinate systems come from the loaded `sdata`
    - labels come from the selected coordinate system in `sdata.labels`
    - images come from the selected coordinate system in `sdata.images`
    - tables are restricted to annotators of the selected labels element
    """

    def __init__(self, napari_viewer: napari.Viewer | None = None) -> None:
        super().__init__()
        self.setObjectName("feature_extraction_widget")
        apply_widget_surface(self)
        self.setMinimumWidth(_WIDGET_MIN_WIDTH)

        # The napari viewer identifies which shared Harpy session this widget
        # belongs to. We use it to attach to the per-viewer HarpyAppState
        # instead of scanning viewer layers directly.
        self._app_state = get_or_create_app_state(napari_viewer)
        self._feature_extraction_controller = FeatureExtractionController(
            on_state_changed=self._on_controller_state_changed,
            on_table_state_changed=self._on_controller_table_state_changed,
            on_feature_matrix_written=self._on_controller_feature_matrix_written,
        )

        self._label_options: list[SpatialDataLabelsOption] = []
        self._selected_label_option: SpatialDataLabelsOption | None = None
        self._image_options: list[SpatialDataImageOption] = []
        self._selected_image_option: SpatialDataImageOption | None = None
        self._image_channel_names: list[str] = []
        self._image_channel_checkboxes: list[QCheckBox] = []
        self._selected_channel_names_by_image_identity: dict[ElementIdentity, tuple[str, ...]] = {}
        # Persistent per-coordinate-system memory of the last explicit user
        # triplet choice. During one rebuild we pass one remembered identity
        # through as a one-shot "candidate to preserve", but these dicts are
        # the longer-lived source of truth across coordinate-system switches
        # while the current SpatialData object remains loaded.
        self._remembered_label_identity_by_coordinate_system: dict[str, ElementIdentity | None] = {}
        self._remembered_image_identity_by_coordinate_system: dict[str, ElementIdentity | None] = {}
        self._table_names: list[str] = []
        self._selected_table_name: str | None = None
        self._coordinate_systems: list[str] = []
        self._checked_coordinate_systems: list[str] = []
        self._selected_coordinate_system: str | None = None
        self._table_binding_error: str | None = None
        self._image_channel_error: str | None = None
        self._feature_checkboxes: dict[str, QCheckBox] = {}
        self._coordinate_system_checkboxes: dict[str, QCheckBox] = {}
        # Multi-card state is split into:
        # - one widget map for the rendered card controls per coordinate system
        # - one state map for the resolved logical card state per coordinate system
        # - one active-card snapshot used as a compatibility bridge while the
        #   lower part of the widget is still largely single-card oriented
        self._triplet_card_widgets_by_coordinate_system: dict[str, _FeatureExtractionTripletCardWidgets] = {}
        self._triplet_card_states_by_coordinate_system: dict[str, _FeatureExtractionTripletCardState] = {}
        self._triplet_card_state = _FeatureExtractionTripletCardState(
            coordinate_system=None,
            label_options=[],
            selected_label_option=None,
            image_options=[],
            selected_image_option=None,
        )
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

        shared_controls_layout = QFormLayout()
        shared_controls_layout.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        shared_controls_layout.setHorizontalSpacing(12)
        shared_controls_layout.setVerticalSpacing(10)

        self._triplet_card_widgets = self._create_placeholder_triplet_card_widgets()
        self.segmentation_combo = self._triplet_card_widgets.segmentation_combo
        self.image_combo = self._triplet_card_widgets.image_combo

        self.coordinate_system_selection_container = QWidget()
        self.coordinate_system_selection_container.setObjectName(
            "feature_extraction_coordinate_system_selection_container"
        )
        coordinate_system_selection_layout = QVBoxLayout(self.coordinate_system_selection_container)
        coordinate_system_selection_layout.setContentsMargins(0, 0, 0, 0)
        coordinate_system_selection_layout.setSpacing(6)
        self.coordinate_system_selection_layout = coordinate_system_selection_layout

        self.triplet_cards_container = QWidget()
        self.triplet_cards_container.setObjectName("feature_extraction_triplet_cards_container")
        triplet_cards_layout = QVBoxLayout(self.triplet_cards_container)
        triplet_cards_layout.setContentsMargins(0, 0, 0, 0)
        triplet_cards_layout.setSpacing(10)
        self.triplet_cards_layout = triplet_cards_layout

        self.channel_selection_label = self._create_form_label("Channels")
        self.channel_selection_container = QWidget()
        self.channel_selection_container.setObjectName("feature_extraction_channel_selection_container")
        self.channel_selection_container.setStyleSheet(_CHANNEL_SELECTION_PANEL_STYLESHEET)
        self.channel_selection_container.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        channel_selection_layout = QVBoxLayout(self.channel_selection_container)
        channel_selection_layout.setContentsMargins(0, 0, 0, 0)
        channel_selection_layout.setSpacing(0)

        self.channel_selection_scroll_area = QScrollArea()
        self.channel_selection_scroll_area.setObjectName("feature_extraction_channel_selection_scroll_area")
        self.channel_selection_scroll_area.setWidgetResizable(True)
        self.channel_selection_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.channel_selection_scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.channel_selection_scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.channel_selection_scroll_area.setStyleSheet("QScrollArea { border: 0px; background: transparent; }")
        self.channel_selection_scroll_area.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.channel_selection_scroll_area.setMinimumWidth(self.image_combo.sizeHint().width())

        self.channel_selection_list_widget = QWidget()
        self.channel_selection_list_widget.setObjectName("feature_extraction_channel_selection_list")
        self.channel_selection_list_widget.setStyleSheet(_CHANNEL_SELECTION_PANEL_STYLESHEET)
        self.channel_selection_list_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.channel_selection_list_layout = QVBoxLayout(self.channel_selection_list_widget)
        self.channel_selection_list_layout.setContentsMargins(0, 0, 0, 0)
        self.channel_selection_list_layout.setSpacing(6)
        self.channel_selection_scroll_area.setWidget(self.channel_selection_list_widget)
        channel_selection_layout.addWidget(self.channel_selection_scroll_area)

        self.channel_selection_label.hide()
        self.channel_selection_container.hide()

        self.table_combo = CompactComboBox()
        self.table_combo.setObjectName("feature_extraction_table_combo")
        self.table_combo.currentIndexChanged.connect(self._on_table_changed)
        self.table_combo.setStyleSheet(_INPUT_CONTROL_STYLESHEET)

        self.coordinate_system_combo = CompactComboBox()
        self.coordinate_system_combo.setObjectName("feature_extraction_coordinate_system_combo")
        self.coordinate_system_combo.currentIndexChanged.connect(self._on_coordinate_system_changed)
        self.coordinate_system_combo.setStyleSheet(_INPUT_CONTROL_STYLESHEET)
        self.coordinate_system_combo.hide()
        self._set_tooltip(
            self.coordinate_system_combo,
            "Choose a coordinate system in which the segmentation mask and image are registered. In the selected coordinate system they should have the same shape, and only an identity transform, a translation, or a sequence of translations may be defined relative to it.",
        )

        self.output_key_line_edit = QLineEdit()
        self.output_key_line_edit.setObjectName("feature_extraction_output_key_line_edit")
        self.output_key_line_edit.setText(_DEFAULT_FEATURE_MATRIX_KEY)
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

        self.calculate_action_row = QWidget()
        self.calculate_action_row.setObjectName("feature_extraction_calculate_action_row")
        calculate_action_layout = QHBoxLayout(self.calculate_action_row)
        calculate_action_layout.setContentsMargins(0, 0, 0, 0)
        calculate_action_layout.setSpacing(8)

        self.calculate_button = QPushButton("Calculate")
        self.calculate_button.setObjectName("feature_extraction_calculate_button")
        self.calculate_button.clicked.connect(self._calculate_feature_matrix)
        self.calculate_button.setEnabled(False)
        self.calculate_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.calculate_button.setMinimumHeight(28)
        self.calculate_button.setStyleSheet(_ACTION_BUTTON_STYLESHEET)
        self._set_tooltip(
            self.calculate_button,
            "Calculation is enabled once the shared SpatialData selection and feature choices form a valid extraction request.",
        )

        self.selection_status = QLabel()
        self.selection_status.setObjectName("feature_extraction_selection_status")
        self.selection_status.setWordWrap(True)
        self.selection_status.hide()

        self.controller_feedback = QLabel()
        self.controller_feedback.setObjectName("feature_extraction_controller_feedback")
        self.controller_feedback.setWordWrap(True)
        self.controller_feedback.hide()

        selector_layout.addRow(
            self._create_form_label("Coordinate systems"), self.coordinate_system_selection_container
        )
        shared_controls_layout.addRow(self.channel_selection_label, self.channel_selection_container)
        shared_controls_layout.addRow(self._create_form_label("Table"), self.table_combo)
        shared_controls_layout.addRow(self._create_form_label("Feature matrix key"), self.output_key_line_edit)

        calculate_action_layout.addWidget(self.calculate_button, 1)

        content_layout.addWidget(title)
        content_layout.addLayout(selector_layout)
        content_layout.addWidget(self.triplet_cards_container)
        content_layout.addLayout(shared_controls_layout)
        content_layout.addSpacing(_FEATURE_GROUPS_TOP_SPACING)
        content_layout.addWidget(self.intensity_features_group)
        content_layout.addWidget(self.morphology_features_group)
        content_layout.addWidget(self.calculate_action_row)
        content_layout.addWidget(self.selection_status)
        content_layout.addWidget(self.controller_feedback)
        content_layout.addStretch(1)

        self._app_state.sdata_changed.connect(self._on_sdata_changed)
        self._update_intensity_features_hint()
        self.refresh_from_sdata(self._app_state.sdata)

    @property
    def app_state(self) -> HarpyAppState:
        """Return the shared Harpy app state for this widget."""
        return self._app_state

    @property
    def selected_segmentation_name(self) -> str | None:
        """Return the currently selected labels element name."""
        return None if self._selected_label_option is None else self._selected_label_option.label_name

    @property
    def selected_spatialdata(self) -> SpatialData | None:
        """Return the loaded SpatialData object backing the current widget state."""
        return self._app_state.sdata

    @property
    def selected_image_name(self) -> str | None:
        """Return the currently selected image element name, if any."""
        return None if self._selected_image_option is None else self._selected_image_option.image_name

    @property
    def selected_extraction_channel_names(self) -> tuple[str, ...] | None:
        """Return the locally selected extraction-channel names for the current image."""
        if self.selected_image_name is None or not self._image_channel_names:
            return None

        return tuple(checkbox.text() for checkbox in self._image_channel_checkboxes if checkbox.isChecked())

    @property
    def selected_extraction_channel_indices(self) -> tuple[int, ...] | None:
        """Return the locally selected extraction-channel indices for the current image."""
        if self.selected_image_name is None or not self._image_channel_names:
            return None

        return tuple(index for index, checkbox in enumerate(self._image_channel_checkboxes) if checkbox.isChecked())

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

    def refresh_from_sdata(self, sdata: SpatialData | None) -> None:
        """Refresh the widget from the shared Harpy SpatialData state."""
        if sdata is None:
            self._clear_selection_inputs()
            self._bind_current_selection()
            return

        self._reset_staged_triplet_state()
        self._refresh_coordinate_systems()
        self._refresh_table_names()
        self._update_intensity_features_hint()
        self._bind_current_selection()

    def _on_sdata_changed(self, sdata: SpatialData | None) -> None:
        self.refresh_from_sdata(sdata)

    def _clear_selection_inputs(self) -> None:
        self._reset_staged_triplet_state()
        self._set_active_card_widgets(None)
        self._clear_triplet_cards()
        self._clear_coordinate_system_checkboxes()

        with QSignalBlocker(self.segmentation_combo):
            self.segmentation_combo.clear()
            self.segmentation_combo.setEnabled(False)
            self.segmentation_combo.setCurrentIndex(-1)

        with QSignalBlocker(self.image_combo):
            self.image_combo.clear()
            self.image_combo.addItem(_NO_IMAGE_TEXT, None)
            self.image_combo.setEnabled(False)
            self.image_combo.setCurrentIndex(0)

        with QSignalBlocker(self.table_combo):
            self.table_combo.clear()
            self.table_combo.setEnabled(False)
            self.table_combo.setCurrentIndex(-1)

        with QSignalBlocker(self.coordinate_system_combo):
            self.coordinate_system_combo.clear()
            self.coordinate_system_combo.setEnabled(False)
            self.coordinate_system_combo.setCurrentIndex(-1)

    def _reset_staged_triplet_state(self) -> None:
        self._label_options = []
        self._selected_label_option = None
        self._image_options = []
        self._selected_image_option = None
        self._selected_channel_names_by_image_identity = {}
        self._remembered_label_identity_by_coordinate_system = {}
        self._remembered_image_identity_by_coordinate_system = {}
        self._clear_image_channel_options()
        self._table_names = []
        self._selected_table_name = None
        self._coordinate_systems = []
        self._checked_coordinate_systems = []
        self._selected_coordinate_system = None
        self._table_binding_error = None
        self._image_channel_error = None
        self._coordinate_system_checkboxes = {}
        self._triplet_card_widgets_by_coordinate_system = {}
        self._triplet_card_states_by_coordinate_system = {}
        self._triplet_card_state = _FeatureExtractionTripletCardState(
            coordinate_system=None,
            label_options=[],
            selected_label_option=None,
            image_options=[],
            selected_image_option=None,
        )

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

    def _create_placeholder_triplet_card_widgets(self) -> _FeatureExtractionTripletCardWidgets:
        """Create hidden placeholder controls used when no visible card is active."""
        segmentation_combo = CompactComboBox()
        segmentation_combo.setObjectName("feature_extraction_segmentation_combo")
        segmentation_combo.setStyleSheet(_INPUT_CONTROL_STYLESHEET)
        segmentation_combo.hide()

        image_combo = CompactComboBox()
        image_combo.setObjectName("feature_extraction_image_combo")
        image_combo.setStyleSheet(_INPUT_CONTROL_STYLESHEET)
        image_combo.hide()

        container = QGroupBox()
        container.hide()

        return _FeatureExtractionTripletCardWidgets(
            coordinate_system="",
            container=container,
            segmentation_combo=segmentation_combo,
            image_combo=image_combo,
        )

    def _create_triplet_card_widgets(self, coordinate_system: str) -> _FeatureExtractionTripletCardWidgets:
        """Create one visible triplet-card widget for a coordinate system."""
        container = QGroupBox(coordinate_system)
        container.setObjectName(f"feature_extraction_triplet_card_{coordinate_system}")
        container.setStyleSheet(_FEATURE_GROUP_STYLESHEET)

        layout = QVBoxLayout(container)
        layout.setContentsMargins(12, 18, 12, 10)
        layout.setSpacing(10)

        segmentation_combo = CompactComboBox()
        segmentation_combo.setObjectName(f"feature_extraction_segmentation_combo_{coordinate_system}")
        segmentation_combo.currentIndexChanged.connect(
            lambda index, current_coordinate_system=coordinate_system: self._on_triplet_card_segmentation_changed(
                current_coordinate_system, index
            )
        )
        segmentation_combo.setStyleSheet(_INPUT_CONTROL_STYLESHEET)

        image_combo = CompactComboBox()
        image_combo.setObjectName(f"feature_extraction_image_combo_{coordinate_system}")
        image_combo.currentIndexChanged.connect(
            lambda index, current_coordinate_system=coordinate_system: self._on_triplet_card_image_changed(
                current_coordinate_system, index
            )
        )
        image_combo.setStyleSheet(_INPUT_CONTROL_STYLESHEET)

        layout.addWidget(self._create_form_label("Segmentation mask"))
        layout.addWidget(segmentation_combo)
        layout.addWidget(self._create_form_label("Image"))
        layout.addWidget(image_combo)

        return _FeatureExtractionTripletCardWidgets(
            coordinate_system=coordinate_system,
            container=container,
            segmentation_combo=segmentation_combo,
            image_combo=image_combo,
        )

    def _get_remembered_triplet_identities_for_coordinate_system(
        self,
        coordinate_system: str | None,
    ) -> tuple[ElementIdentity | None, ElementIdentity | None]:
        if coordinate_system is None:
            return None, None

        return (
            self._remembered_label_identity_by_coordinate_system.get(coordinate_system),
            self._remembered_image_identity_by_coordinate_system.get(coordinate_system),
        )

    def _clear_coordinate_system_checkboxes(self) -> None:
        while self.coordinate_system_selection_layout.count():
            item = self.coordinate_system_selection_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._coordinate_system_checkboxes = {}

    def _clear_triplet_cards(self) -> None:
        while self.triplet_cards_layout.count():
            item = self.triplet_cards_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._triplet_card_widgets_by_coordinate_system = {}
        self._triplet_card_states_by_coordinate_system = {}

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
        widget.setToolTip(format_tooltip(message))

    def _build_triplet_card_state(
        self,
        coordinate_system: str | None,
        *,
        previous_label_identity: ElementIdentity | None,
        previous_image_identity: ElementIdentity | None,
    ) -> _FeatureExtractionTripletCardState:
        """Resolve one coordinate-system card state from shared `SpatialData`."""
        if self.selected_spatialdata is None or coordinate_system is None:
            label_options: list[SpatialDataLabelsOption] = []
        else:
            label_options = get_spatialdata_feature_extraction_label_options_for_coordinate_system_from_sdata(
                sdata=self.selected_spatialdata,
                coordinate_system=coordinate_system,
            )

        next_label_index = None
        for index, option in enumerate(label_options):
            if option.identity == previous_label_identity:
                next_label_index = index
                break
        selected_label_option = None
        if label_options:
            selected_label_option = label_options[0 if next_label_index is None else next_label_index]

        if self.selected_spatialdata is None or coordinate_system is None or selected_label_option is None:
            image_options: list[SpatialDataImageOption] = []
        else:
            image_options = get_spatialdata_matching_image_options_for_coordinate_system_and_label_from_sdata(
                sdata=self.selected_spatialdata,
                coordinate_system=coordinate_system,
                label_name=selected_label_option.label_name,
            )

        next_image_index = None
        for index, option in enumerate(image_options):
            if option.identity == previous_image_identity:
                next_image_index = index
                break
        if next_image_index is not None:
            selected_image_option = image_options[next_image_index]
        else:
            selected_image_option = None

        return _FeatureExtractionTripletCardState(
            coordinate_system=coordinate_system,
            label_options=label_options,
            selected_label_option=selected_label_option,
            image_options=image_options,
            selected_image_option=selected_image_option,
        )

    def _apply_triplet_card_state(
        self,
        widgets: _FeatureExtractionTripletCardWidgets,
        state: _FeatureExtractionTripletCardState,
    ) -> None:
        """Apply a resolved state to one visible triplet-card widget."""
        with QSignalBlocker(widgets.segmentation_combo):
            widgets.segmentation_combo.clear()
            for option in state.label_options:
                widgets.segmentation_combo.addItem(option.display_name)

            has_label_options = bool(state.label_options)
            widgets.segmentation_combo.setEnabled(has_label_options)

            next_segmentation_index = self._find_label_option_index_in_options(
                state.label_options,
                None if state.selected_label_option is None else state.selected_label_option.identity,
            )
            if has_label_options:
                widgets.segmentation_combo.setCurrentIndex(
                    0 if next_segmentation_index is None else next_segmentation_index
                )
            else:
                widgets.segmentation_combo.setCurrentIndex(-1)

        with QSignalBlocker(widgets.image_combo):
            widgets.image_combo.clear()
            widgets.image_combo.addItem(_NO_IMAGE_TEXT, None)
            for option in state.image_options:
                widgets.image_combo.addItem(option.display_name)

            widgets.image_combo.setEnabled(
                self.selected_spatialdata is not None
                and state.coordinate_system is not None
                and state.selected_label_option is not None
            )

            next_image_index = self._find_image_option_index_in_options(
                state.image_options,
                None if state.selected_image_option is None else state.selected_image_option.identity,
            )
            if next_image_index is None:
                widgets.image_combo.setCurrentIndex(0)
            else:
                widgets.image_combo.setCurrentIndex(next_image_index + 1)

    def _set_active_card_widgets(self, coordinate_system: str | None) -> None:
        if coordinate_system is None:
            self.segmentation_combo = self._triplet_card_widgets.segmentation_combo
            self.image_combo = self._triplet_card_widgets.image_combo
            return

        widgets = self._triplet_card_widgets_by_coordinate_system.get(coordinate_system)
        if widgets is None:
            self.segmentation_combo = self._triplet_card_widgets.segmentation_combo
            self.image_combo = self._triplet_card_widgets.image_combo
            return

        self.segmentation_combo = widgets.segmentation_combo
        self.image_combo = widgets.image_combo

    def _set_selected_coordinate_system_by_name(self, coordinate_system: str | None) -> None:
        self._selected_coordinate_system = coordinate_system
        self._set_active_card_widgets(coordinate_system)
        with QSignalBlocker(self.coordinate_system_combo):
            if coordinate_system is None:
                self.coordinate_system_combo.setCurrentIndex(-1)
            else:
                self.coordinate_system_combo.setCurrentIndex(self.coordinate_system_combo.findData(coordinate_system))

    def _sync_coordinate_system_combo_items(self) -> None:
        with QSignalBlocker(self.coordinate_system_combo):
            self.coordinate_system_combo.clear()
            for coordinate_system in self._coordinate_systems:
                self.coordinate_system_combo.addItem(coordinate_system, coordinate_system)
            self.coordinate_system_combo.setEnabled(bool(self._coordinate_systems))
            if self.selected_coordinate_system is None:
                self.coordinate_system_combo.setCurrentIndex(-1)
            else:
                self.coordinate_system_combo.setCurrentIndex(
                    self.coordinate_system_combo.findData(self.selected_coordinate_system)
                )

    def _sync_coordinate_system_checkboxes(self) -> None:
        self._clear_coordinate_system_checkboxes()
        checked_coordinate_systems = set(self._checked_coordinate_systems)
        for coordinate_system in self._coordinate_systems:
            checkbox = QCheckBox(coordinate_system)
            checkbox.setObjectName(f"feature_extraction_coordinate_system_checkbox_{coordinate_system}")
            checkbox.setStyleSheet(_FEATURE_CHECKBOX_STYLESHEET)
            checkbox.setChecked(coordinate_system in checked_coordinate_systems)
            checkbox.toggled.connect(
                lambda checked, current_coordinate_system=coordinate_system: (
                    self._on_coordinate_system_checkbox_toggled(current_coordinate_system, checked)
                )
            )
            self.coordinate_system_selection_layout.addWidget(checkbox)
            self._coordinate_system_checkboxes[coordinate_system] = checkbox

    def _checked_coordinate_system_names(self) -> list[str]:
        checked_coordinate_systems = set(self._checked_coordinate_systems)
        return [
            coordinate_system
            for coordinate_system in self._coordinate_systems
            if coordinate_system in checked_coordinate_systems
        ]

    def _refresh_triplet_card_for_coordinate_system(
        self,
        coordinate_system: str,
        *,
        previous_label_identity: ElementIdentity | None = None,
        previous_image_identity: ElementIdentity | None = None,
    ) -> None:
        if previous_label_identity is None or previous_image_identity is None:
            remembered_label_identity, remembered_image_identity = (
                self._get_remembered_triplet_identities_for_coordinate_system(coordinate_system)
            )
            if previous_label_identity is None:
                previous_label_identity = remembered_label_identity
            if previous_image_identity is None:
                previous_image_identity = remembered_image_identity

        state = self._build_triplet_card_state(
            coordinate_system,
            previous_label_identity=previous_label_identity,
            previous_image_identity=previous_image_identity,
        )
        self._triplet_card_states_by_coordinate_system[coordinate_system] = state
        widgets = self._triplet_card_widgets_by_coordinate_system.get(coordinate_system)
        if widgets is not None:
            self._apply_triplet_card_state(widgets, state)

    def _sync_active_triplet_card_state(self) -> None:
        self._set_active_card_widgets(self.selected_coordinate_system)
        state = (
            None
            if self.selected_coordinate_system is None
            else self._triplet_card_states_by_coordinate_system.get(self.selected_coordinate_system)
        )
        if state is None:
            self._triplet_card_state = _FeatureExtractionTripletCardState(
                coordinate_system=self.selected_coordinate_system,
                label_options=[],
                selected_label_option=None,
                image_options=[],
                selected_image_option=None,
            )
            self._label_options = []
            self._selected_label_option = None
            self._image_options = []
            self._selected_image_option = None
            self._clear_image_channel_options()
            return

        self._triplet_card_state = state
        self._label_options = state.label_options
        self._selected_label_option = state.selected_label_option
        self._image_options = state.image_options
        self._selected_image_option = state.selected_image_option
        self._refresh_image_channel_options()

    def _refresh_triplet_cards(self) -> None:
        selected_coordinate_systems = self._checked_coordinate_system_names()
        self._clear_triplet_cards()
        for coordinate_system in selected_coordinate_systems:
            widgets = self._create_triplet_card_widgets(coordinate_system)
            self._triplet_card_widgets_by_coordinate_system[coordinate_system] = widgets
            self.triplet_cards_layout.addWidget(widgets.container)
            self._refresh_triplet_card_for_coordinate_system(coordinate_system)

        if self.selected_coordinate_system not in selected_coordinate_systems:
            next_active_coordinate_system = selected_coordinate_systems[0] if selected_coordinate_systems else None
            self._set_selected_coordinate_system_by_name(next_active_coordinate_system)

        self._sync_active_triplet_card_state()

    def _on_triplet_card_segmentation_changed(self, coordinate_system: str, index: int) -> None:
        self._set_selected_coordinate_system_by_name(coordinate_system)
        card_state = self._triplet_card_states_by_coordinate_system.get(coordinate_system)
        previous_image_identity = (
            None
            if card_state is None or card_state.selected_image_option is None
            else card_state.selected_image_option.identity
        )

        selected_label_option = None
        if card_state is not None and 0 <= index < len(card_state.label_options):
            selected_label_option = card_state.label_options[index]

        self._remembered_label_identity_by_coordinate_system[coordinate_system] = (
            None if selected_label_option is None else selected_label_option.identity
        )
        self._refresh_triplet_card_for_coordinate_system(
            coordinate_system,
            previous_label_identity=None if selected_label_option is None else selected_label_option.identity,
            previous_image_identity=previous_image_identity,
        )
        refreshed_state = self._triplet_card_states_by_coordinate_system.get(coordinate_system)
        self._remembered_image_identity_by_coordinate_system[coordinate_system] = (
            None
            if refreshed_state is None or refreshed_state.selected_image_option is None
            else refreshed_state.selected_image_option.identity
        )
        self._sync_active_triplet_card_state()
        self._refresh_table_names()
        self._update_intensity_features_hint()
        self._bind_current_selection()

    def _on_segmentation_changed(self, index: int) -> None:
        if self.selected_coordinate_system is not None:
            self._on_triplet_card_segmentation_changed(self.selected_coordinate_system, index)

    def _find_label_option_index_in_options(
        self,
        label_options: list[SpatialDataLabelsOption],
        identity: ElementIdentity | None,
    ) -> int | None:
        if identity is None:
            return None

        for index, option in enumerate(label_options):
            if option.identity == identity:
                return index

        return None

    def _find_label_option_index(self, identity: ElementIdentity | None) -> int | None:
        return self._find_label_option_index_in_options(self._label_options, identity)

    def _refresh_label_options(self) -> None:
        for coordinate_system in self._checked_coordinate_system_names():
            self._refresh_triplet_card_for_coordinate_system(coordinate_system)
        self._sync_active_triplet_card_state()

    def _refresh_image_options(self) -> None:
        for coordinate_system in self._checked_coordinate_system_names():
            self._refresh_triplet_card_for_coordinate_system(coordinate_system)
        self._sync_active_triplet_card_state()

    def _on_triplet_card_image_changed(self, coordinate_system: str, index: int) -> None:
        self._set_selected_coordinate_system_by_name(coordinate_system)
        card_state = self._triplet_card_states_by_coordinate_system.get(coordinate_system)
        if card_state is None:
            return

        selected_image_option = None
        if 0 < index <= len(card_state.image_options):
            selected_image_option = card_state.image_options[index - 1]

        self._triplet_card_states_by_coordinate_system[coordinate_system] = replace(
            card_state,
            selected_image_option=selected_image_option,
        )
        self._remembered_image_identity_by_coordinate_system[coordinate_system] = (
            None if selected_image_option is None else selected_image_option.identity
        )
        self._sync_active_triplet_card_state()
        self._update_intensity_features_hint()
        self._bind_current_selection()

    def _on_image_changed(self, index: int) -> None:
        if self.selected_coordinate_system is not None:
            self._on_triplet_card_image_changed(self.selected_coordinate_system, index)

    def _set_selected_image_option(self, index: int) -> None:
        if index <= 0 or index > len(self._image_options):
            self._selected_image_option = None
        else:
            self._selected_image_option = self._image_options[index - 1]

    def _find_image_option_index_in_options(
        self,
        image_options: list[SpatialDataImageOption],
        identity: ElementIdentity | None,
    ) -> int | None:
        if identity is None:
            return None

        for index, option in enumerate(image_options):
            if option.identity == identity:
                return index

        return None

    def _find_image_option_index(self, identity: ElementIdentity | None) -> int | None:
        return self._find_image_option_index_in_options(self._image_options, identity)

    def _clear_image_channel_options(self) -> None:
        self._image_channel_names = []
        self._image_channel_checkboxes = []

        while self.channel_selection_list_layout.count():
            item = self.channel_selection_list_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        self.channel_selection_label.hide()
        self.channel_selection_container.hide()
        self.channel_selection_scroll_area.setMaximumHeight(0)

    def _refresh_image_channel_options(self) -> None:
        self._clear_image_channel_options()
        self._image_channel_error = None

        if self.selected_spatialdata is None or self._selected_image_option is None:
            return

        try:
            channel_names = get_image_channel_names_from_sdata(self.selected_spatialdata, self.selected_image_name)
        except ValueError as error:
            self._image_channel_error = str(error)
            return
        if not channel_names:
            return

        self._image_channel_names = channel_names
        current_image_identity = self._selected_image_option.identity
        selected_channel_names = self._selected_channel_names_by_image_identity.get(current_image_identity)
        if selected_channel_names is None:
            selected_channel_name_set = set(channel_names)
        else:
            selected_channel_name_set = set(selected_channel_names)

        channel_rows: list[QWidget] = []
        for channel_name in channel_names:
            checkbox = QCheckBox(channel_name)
            checkbox.setObjectName(f"feature_extraction_channel_checkbox_{channel_name}")
            checkbox.setStyleSheet(_FEATURE_CHECKBOX_STYLESHEET)
            checkbox.setChecked(channel_name in selected_channel_name_set)
            checkbox.toggled.connect(self._on_channel_selection_changed)
            self.channel_selection_list_layout.addWidget(checkbox)
            self._image_channel_checkboxes.append(checkbox)
            channel_rows.append(checkbox)

        self._set_channel_selection_scroll_height(channel_rows)
        self.channel_selection_label.show()
        self.channel_selection_container.show()
        self._store_selected_channel_names_for_current_image()

    def _on_channel_selection_changed(self, _checked: bool) -> None:
        self._store_selected_channel_names_for_current_image()
        self._bind_current_selection()

    def _store_selected_channel_names_for_current_image(self) -> None:
        if self._selected_image_option is None or not self._image_channel_names:
            return

        self._selected_channel_names_by_image_identity[self._selected_image_option.identity] = tuple(
            checkbox.text() for checkbox in self._image_channel_checkboxes if checkbox.isChecked()
        )

    def _set_channel_selection_scroll_height(self, channel_rows: list[QWidget]) -> None:
        visible_rows = channel_rows[:_MAX_VISIBLE_EXTRACTION_CHANNELS]
        if not visible_rows:
            self.channel_selection_scroll_area.setMaximumHeight(0)
            return

        visible_height = sum(row.sizeHint().height() for row in visible_rows)
        visible_height += self.channel_selection_list_layout.spacing() * max(0, len(visible_rows) - 1)
        margins = self.channel_selection_list_layout.contentsMargins()
        visible_height += margins.top() + margins.bottom()
        visible_height += self.channel_selection_scroll_area.frameWidth() * 2
        self.channel_selection_scroll_area.setMaximumHeight(visible_height)

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
        if self.selected_spatialdata is None:
            self._coordinate_systems = []
        else:
            self._coordinate_systems = get_coordinate_system_names_from_sdata(self.selected_spatialdata)

        checked_coordinate_systems = {
            coordinate_system
            for coordinate_system in self._checked_coordinate_systems
            if coordinate_system in self._coordinate_systems
        }
        self._checked_coordinate_systems = [
            coordinate_system
            for coordinate_system in self._coordinate_systems
            if coordinate_system in checked_coordinate_systems
        ]

        if self.selected_coordinate_system not in checked_coordinate_systems:
            next_active_coordinate_system = (
                self._checked_coordinate_systems[0] if self._checked_coordinate_systems else None
            )
            self._set_selected_coordinate_system_by_name(next_active_coordinate_system)

        self._sync_coordinate_system_combo_items()
        self._sync_coordinate_system_checkboxes()
        self._refresh_triplet_cards()

    def _on_coordinate_system_changed(self, index: int) -> None:
        coordinate_system = self.coordinate_system_combo.itemData(index)
        next_coordinate_system = coordinate_system if isinstance(coordinate_system, str) else None
        if next_coordinate_system is not None and next_coordinate_system not in self._checked_coordinate_systems:
            self._checked_coordinate_systems.append(next_coordinate_system)
            self._checked_coordinate_systems = self._checked_coordinate_system_names()
            checkbox = self._coordinate_system_checkboxes.get(next_coordinate_system)
            if checkbox is not None:
                with QSignalBlocker(checkbox):
                    checkbox.setChecked(True)
            self._refresh_triplet_cards()

        self._set_selected_coordinate_system_by_name(next_coordinate_system)
        self._sync_active_triplet_card_state()
        self._refresh_table_names()
        self._update_intensity_features_hint()
        self._bind_current_selection()

    def _set_selected_coordinate_system(self, index: int) -> None:
        coordinate_system = self.coordinate_system_combo.itemData(index)
        self._selected_coordinate_system = coordinate_system if isinstance(coordinate_system, str) else None

    def _on_coordinate_system_checkbox_toggled(self, coordinate_system: str, checked: bool) -> None:
        checked_coordinate_systems = set(self._checked_coordinate_systems)
        if checked:
            checked_coordinate_systems.add(coordinate_system)
        else:
            checked_coordinate_systems.discard(coordinate_system)
        self._checked_coordinate_systems = [
            current_coordinate_system
            for current_coordinate_system in self._coordinate_systems
            if current_coordinate_system in checked_coordinate_systems
        ]

        next_active_coordinate_system = self.selected_coordinate_system
        if checked:
            next_active_coordinate_system = coordinate_system
        elif next_active_coordinate_system == coordinate_system:
            next_active_coordinate_system = (
                self._checked_coordinate_systems[0] if self._checked_coordinate_systems else None
            )

        self._set_selected_coordinate_system_by_name(next_active_coordinate_system)
        self._refresh_triplet_cards()
        self._refresh_table_names()
        self._update_intensity_features_hint()
        self._bind_current_selection()

    def _on_feature_selection_changed(self, _checked: bool) -> None:
        self._update_intensity_features_hint()
        self._bind_current_selection()

    def _on_feature_key_changed(self, _text: str) -> None:
        self._bind_current_selection()

    def _calculate_feature_matrix(self) -> None:
        feature_key = self.selected_feature_key
        # `overwrite_feature_key` is effectively tri-state in this method:
        # `False` means run normally without overwrite, `True` means the user
        # confirmed replacing an existing key, and `None` means the overwrite
        # prompt was canceled so calculation should abort.
        overwrite_feature_key = False
        can_check_existing_feature_key = (
            feature_key is not None
            and self.selected_spatialdata is not None
            and self.selected_table_name is not None
            and self._table_binding_error is None
        )
        # Only inspect `.obsm` when the current table binding is valid enough
        # to read the selected table safely. The controller still decides
        # whether calculation can actually run.
        if can_check_existing_feature_key:
            table = get_table(self.selected_spatialdata, self.selected_table_name)
            if feature_key in table.obsm:
                overwrite_feature_key = self._prompt_overwrite_feature_key_confirmation(
                    feature_key,
                    self.selected_table_name,
                )
                if overwrite_feature_key is None:
                    return

        self._feature_extraction_controller.calculate(overwrite_feature_key=overwrite_feature_key)

    def _prompt_overwrite_feature_key_confirmation(
        self,
        feature_key: str,
        table_name: str | None,
    ) -> bool | None:
        dialog = QDialog(self)
        dialog.setWindowTitle("Feature Matrix Already Exists")
        dialog.setModal(True)
        dialog.setMinimumWidth(560)

        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(14)

        warning_message = (
            f"Table `{escape(table_name)}` already contains `.obsm[{feature_key!r}]`."
            if table_name is not None
            else f"The selected table already contains `.obsm[{feature_key!r}]`."
        )
        warning_card = QLabel(
            "<div>"
            "<span style='font-size: 11px; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase;'>"
            "Existing Feature Matrix</span><br>"
            f"<span>{warning_message}</span><br>"
            "<span>Do you want to replace it?</span>"
            "</div>"
        )
        warning_card.setWordWrap(True)
        warning_card.setStyleSheet(
            "font-weight: 500; "
            "color: #b45309; "
            "background-color: #fffbeb; "
            "border: 1px solid #fde68a; "
            "border-radius: 10px; "
            "padding: 12px 14px;"
        )
        layout.addWidget(warning_card)

        button_row = QHBoxLayout()
        button_row.setSpacing(10)
        button_row.addStretch(1)

        overwrite_button = QPushButton("Replace feature matrix")
        cancel_button = QPushButton("Cancel")

        overwrite_button.setStyleSheet(
            "QPushButton {"
            "background-color: #166534; "
            "color: white; "
            "border: 1px solid #166534; "
            "border-radius: 6px; "
            "padding: 7px 14px; "
            "font-weight: 600;}"
            "QPushButton:hover { background-color: #15803d; border-color: #15803d; }"
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

        button_row.addWidget(overwrite_button)
        button_row.addWidget(cancel_button)
        layout.addLayout(button_row)

        overwrite_button.clicked.connect(dialog.accept)
        cancel_button.clicked.connect(dialog.reject)
        cancel_button.setDefault(True)

        return True if dialog.exec() == QDialog.DialogCode.Accepted else None

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
        effective_image_name = None if self._image_channel_error is not None else self.selected_image_name
        effective_channels = None if self._image_channel_error is not None else self.selected_extraction_channel_names
        self._feature_extraction_controller.bind(
            self.selected_spatialdata,
            self.selected_segmentation_name,
            effective_image_name,
            effective_table_name,
            self.selected_coordinate_system,
            self.selected_feature_names,
            self.selected_feature_key,
            channels=effective_channels,
            overwrite_feature_key=False,
        )
        self._update_selection_status()

    def _update_selection_status(self) -> None:
        self._update_primary_status_card()
        self._update_calculate_controls()

    def _update_primary_status_card(self) -> None:
        if self._app_state.sdata is None:
            self._set_selection_status(
                "No SpatialData Loaded",
                [
                    "Load a SpatialData object through the Harpy Viewer widget, reader, or `Interactive(sdata)`.",
                    "This form updates automatically from the shared Harpy state.",
                ],
                kind="warning",
            )
            return

        if self.selected_coordinate_system is None:
            self._set_selection_status(
                "Choose Coordinate Systems",
                ["Choose one or more coordinate systems to start building extraction targets."],
                kind="warning",
            )
            return

        if self.selected_spatialdata is None or self.selected_segmentation_name is None:
            self._set_selection_status(
                "Selection Needed",
                ["Choose a segmentation available in the selected coordinate system."],
                kind="warning",
            )
            return

        if self.selected_table_name is None:
            segmentation_name, segmentation_shortened = format_feedback_identifier(self.selected_segmentation_name)
            lines = [
                f"Segmentation `{segmentation_name}` is not linked to an annotation table.",
                "Support for creating a new linked table from this widget is coming soon.",
            ]
            self._set_selection_status(
                "No Table Linked",
                lines,
                tooltip_message="\n".join(
                    [
                        f"Segmentation `{self.selected_segmentation_name}` is not linked to an annotation table.",
                        "Support for creating a new linked table from this widget is coming soon.",
                    ]
                )
                if segmentation_shortened
                else None,
                kind="warning",
            )
            return

        if self._table_binding_error is not None:
            table_name, table_shortened = format_feedback_identifier(self.selected_table_name)
            segmentation_name, segmentation_shortened = format_feedback_identifier(self.selected_segmentation_name)
            lines = [
                f"Table `{table_name}` cannot currently be used for segmentation `{segmentation_name}`.",
                "Choose a different table or segmentation.",
            ]
            self._set_selection_status(
                "Table Binding Issue",
                lines,
                tooltip_message=self._table_binding_error,
                kind="warning",
            )
            return

        if self._image_channel_error is not None and self.selected_image_name is not None:
            image_name, _ = format_feedback_identifier(self.selected_image_name)
            self._set_selection_status(
                "Image Channel Issue",
                [
                    f"Image `{image_name}` has duplicate channel names, which Harpy does not support.",
                    "Use `sdata.set_channel_names(...)` to rename the channels, or choose a different image.",
                ],
                tooltip_message=self._image_channel_error,
                kind="warning",
            )
            return

        segmentation_name, segmentation_shortened = format_feedback_identifier(self.selected_segmentation_name)
        table_name, table_shortened = format_feedback_identifier(self.selected_table_name)
        coordinate_system_name, coordinate_system_shortened = format_feedback_identifier(
            self.selected_coordinate_system
        )

        shortened = [segmentation_shortened, table_shortened, coordinate_system_shortened]
        tooltip_lines = [
            f"Segmentation: {self.selected_segmentation_name}",
            f"Table: {self.selected_table_name}",
        ]

        if self.selected_image_name is None:
            image_line = "Image: none selected yet"
            tooltip_lines.append(image_line)
        else:
            image_name, image_shortened = format_feedback_identifier(self.selected_image_name)
            image_line = f"Image: {image_name}"
            tooltip_lines.append(f"Image: {self.selected_image_name}")
            shortened.append(image_shortened)

        tooltip_lines.append(f"Coordinate system: {self.selected_coordinate_system}")
        self._set_selection_status(
            "Selection Ready",
            [
                f"Segmentation: {segmentation_name}",
                f"Table: {table_name}",
                image_line,
                f"Coordinate system: {coordinate_system_name}",
            ],
            tooltip_message="\n".join(tooltip_lines) if any(shortened) else None,
            kind="success",
        )

    def _set_selection_status(
        self,
        title: str,
        lines: list[str],
        *,
        kind: StatusCardKind,
        tooltip_message: str | None = None,
    ) -> None:
        set_status_card(
            self.selection_status,
            title=title,
            lines=lines,
            kind=kind,
            tooltip_message=tooltip_message,
        )

    def _set_feature_extraction_feedback(self, message: str, *, kind: StatusCardKind = "info") -> None:
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
        set_status_card(
            self.controller_feedback,
            title=title_by_kind.get(kind, "Feature Extraction"),
            lines=[body],
            kind=kind,
        )

    def _update_calculate_controls(self) -> None:
        self.calculate_button.setEnabled(False)
        tooltip = (
            "Calculation is temporarily disabled while the multi-sample feature-extraction widget is being refactored."
        )
        self._set_tooltip(self.calculate_button, tooltip)

    def _on_controller_state_changed(self) -> None:
        self._set_feature_extraction_feedback(
            self._feature_extraction_controller.status_message,
            kind=self._feature_extraction_controller.status_kind,
        )
        self._update_calculate_controls()

    def _on_controller_table_state_changed(self) -> None:
        self._refresh_table_names()
        self._bind_current_selection()

    def _on_controller_feature_matrix_written(self, event: FeatureMatrixWrittenEvent) -> None:
        self._app_state.emit_feature_matrix_written(event)
