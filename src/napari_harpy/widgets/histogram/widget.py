from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING

from qtpy.QtCore import QSignalBlocker, QSize, Qt, Signal
from qtpy.QtGui import QPixmap
from qtpy.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QStyle,
    QToolButton,
    QVBoxLayout,
    QWidget,
)
from xarray import DataArray, DataTree

from napari_harpy._app_state import CoordinateSystemChangedEvent, HarpyAppState, get_or_create_app_state
from napari_harpy._resources import get_logo_path
from napari_harpy.core.histogram import HistogramSettings, HistogramTarget
from napari_harpy.core.spatialdata import (
    get_coordinate_system_names_from_sdata,
    get_image_channel_names_from_sdata,
    get_spatialdata_image_options_for_coordinate_system_from_sdata,
)
from napari_harpy.widgets.shared_styles import (
    ACTION_BUTTON_STYLESHEET,
    CHECKBOX_STYLESHEET,
    DISCLOSURE_CHEVRON_SIZE,
    SECONDARY_BUTTON_STYLESHEET,
    WIDGET_ACCENT_BORDER_COLOR,
    WIDGET_ACCENT_SOFT_COLOR,
    WIDGET_BORDER_COLOR,
    WIDGET_BORDER_STRONG_COLOR,
    WIDGET_MIN_WIDTH,
    WIDGET_PANEL_COLOR,
    WIDGET_PANEL_MUTED_COLOR,
    WIDGET_PANEL_SUBTLE_COLOR,
    WIDGET_SURFACE_COLOR,
    WIDGET_TEXT_COLOR,
    WIDGET_TEXT_MUTED_COLOR,
    CompactComboBox,
    apply_scroll_content_surface,
    apply_widget_surface,
    build_input_control_stylesheet,
    create_disclosure_chevron_icon,
    create_form_label,
    format_tooltip,
    set_status_card,
)

if TYPE_CHECKING:
    import napari
    from spatialdata import SpatialData


_INPUT_CONTROL_STYLESHEET = build_input_control_stylesheet("QComboBox, QLineEdit, QSpinBox")
_CARD_STYLESHEET = (
    "QFrame[histogramCard='true'] {"
    f"background-color: {WIDGET_PANEL_COLOR}; "
    f"border: 1px solid {WIDGET_BORDER_COLOR}; "
    "border-radius: 8px;}"
)
_CARD_TITLE_STYLESHEET = f"color: {WIDGET_TEXT_COLOR}; font-weight: 700; font-size: 13px; background: transparent;"
_SETTINGS_PANEL_STYLESHEET = "QWidget { background: transparent; }"
_SETTINGS_TOGGLE_STYLESHEET = (
    "QToolButton {"
    f"background-color: {WIDGET_PANEL_SUBTLE_COLOR}; "
    f"border: 1px solid {WIDGET_BORDER_COLOR}; "
    "border-radius: 8px; "
    f"color: {WIDGET_TEXT_COLOR}; "
    "font-size: 13px; "
    "font-weight: 600; "
    "padding: 3px 10px; "
    "min-height: 26px; "
    "text-align: left;}"
    f"QToolButton:hover {{ background-color: {WIDGET_PANEL_MUTED_COLOR}; border-color: {WIDGET_BORDER_STRONG_COLOR}; }}"
    f"QToolButton:checked {{ background-color: {WIDGET_ACCENT_SOFT_COLOR}; border-color: {WIDGET_ACCENT_BORDER_COLOR}; }}"
)
_ICON_BUTTON_STYLESHEET = (
    "QToolButton {"
    f"background-color: {WIDGET_PANEL_SUBTLE_COLOR}; "
    f"border: 1px solid {WIDGET_BORDER_COLOR}; "
    "border-radius: 6px; "
    "padding: 4px;}"
    f"QToolButton:hover {{ background-color: {WIDGET_PANEL_MUTED_COLOR}; border-color: {WIDGET_BORDER_STRONG_COLOR}; }}"
)
_EMPTY_STATE_STYLESHEET = (
    f"color: {WIDGET_TEXT_MUTED_COLOR}; "
    f"background-color: {WIDGET_PANEL_SUBTLE_COLOR}; "
    f"border: 1px dashed {WIDGET_BORDER_COLOR}; "
    "border-radius: 8px; "
    "padding: 12px;"
)
_SETTING_SCALE_CONTROL_WIDTH = 150
_SETTING_NUMERIC_CONTROL_WIDTH = 112
_SETTING_BINS_CONTROL_WIDTH = 96
_DEFAULT_SCALE = "scale0"
_DEFAULT_SETTINGS = HistogramSettings()


@dataclass(frozen=True)
class HistogramCalculationRequest:
    """Card-local request emitted when a histogram card is ready to calculate."""

    card_id: str
    target: HistogramTarget
    settings: HistogramSettings


@dataclass(frozen=True)
class _HistogramCardWidgets:
    card_id: str
    container: QFrame
    title_label: QLabel
    coordinate_system_combo: CompactComboBox
    image_combo: CompactComboBox
    channel_combo: CompactComboBox
    calculate_button: QPushButton
    remove_button: QToolButton
    status_label: QLabel
    settings_toggle: QToolButton
    settings_panel: QWidget
    scale_combo: CompactComboBox
    bins_spin: QSpinBox
    value_range_low_edit: QLineEdit
    value_range_high_edit: QLineEdit
    density_checkbox: QCheckBox
    exclude_nan_checkbox: QCheckBox
    exclude_zeros_checkbox: QCheckBox
    log_y_checkbox: QCheckBox
    percentile_min_edit: QLineEdit
    percentile_max_edit: QLineEdit
    reset_settings_button: QPushButton


class HistogramWidget(QWidget):
    """Qt shell for explicit per-card image histogram requests."""

    calculation_requested = Signal(object)

    def __init__(self, napari_viewer: napari.Viewer | None = None) -> None:
        super().__init__()
        self.setObjectName("histogram_widget")
        apply_widget_surface(self)
        self.setMinimumWidth(WIDGET_MIN_WIDTH)

        self._app_state = get_or_create_app_state(napari_viewer)
        self._cards: dict[str, _HistogramCardWidgets] = {}
        self._card_channel_errors: dict[str, str] = {}
        self._last_emitted_requests: dict[str, HistogramCalculationRequest] = {}
        self._logo_path = get_logo_path()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.scroll_area = QScrollArea()
        self.scroll_area.setObjectName("histogram_scroll_area")
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll_area.setStyleSheet("QScrollArea { border: 0px; background: transparent; }")

        self.scroll_content = QWidget()
        self.scroll_content.setObjectName("histogram_scroll_content")
        apply_scroll_content_surface(self.scroll_content)
        self.scroll_area.setWidget(self.scroll_content)
        layout.addWidget(self.scroll_area)

        content_layout = QVBoxLayout(self.scroll_content)
        content_layout.setContentsMargins(12, 12, 12, 12)
        content_layout.setSpacing(10)

        header_logo = self._create_header_logo()

        action_row = QWidget()
        action_row.setObjectName("histogram_header_action_row")
        action_layout = QHBoxLayout(action_row)
        action_layout.setContentsMargins(0, 0, 0, 0)
        action_layout.setSpacing(8)
        self.add_button = QPushButton("Add histogram")
        self.add_button.setObjectName("histogram_add_button")
        self.add_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.add_button.setStyleSheet(ACTION_BUTTON_STYLESHEET)
        self.add_button.clicked.connect(self.add_histogram_card)
        action_layout.addWidget(self.add_button)
        action_layout.addStretch(1)

        self.empty_state_label = QLabel("No histograms")
        self.empty_state_label.setObjectName("histogram_empty_state")
        self.empty_state_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_state_label.setStyleSheet(_EMPTY_STATE_STYLESHEET)

        self.cards_container = QWidget()
        self.cards_container.setObjectName("histogram_cards_container")
        self.cards_container.setStyleSheet(
            f"QWidget#histogram_cards_container {{ background: {WIDGET_SURFACE_COLOR}; }}"
        )
        self.cards_layout = QVBoxLayout(self.cards_container)
        self.cards_layout.setContentsMargins(0, 0, 0, 0)
        self.cards_layout.setSpacing(10)

        content_layout.addWidget(header_logo)
        content_layout.addWidget(action_row)
        content_layout.addWidget(self.empty_state_label)
        content_layout.addWidget(self.cards_container)
        content_layout.addStretch(1)

        self._app_state.sdata_changed.connect(self._on_sdata_changed)
        self._app_state.coordinate_system_changed.connect(self._on_coordinate_system_changed)
        self._update_empty_state()

    @property
    def app_state(self) -> HarpyAppState:
        """Return the shared Harpy app state for this widget."""
        return self._app_state

    @property
    def selected_spatialdata(self) -> SpatialData | None:
        """Return the loaded SpatialData object backing the widget."""
        return self._app_state.sdata

    @property
    def card_count(self) -> int:
        """Return the number of histogram cards currently shown."""
        return len(self._cards)

    @property
    def card_ids(self) -> tuple[str, ...]:
        """Return stable card ids in display order."""
        return tuple(self._cards)

    def add_histogram_card(self) -> str:
        """Add one histogram target card and return its stable card id."""
        card_id = uuid.uuid4().hex
        widgets = self._create_card_widgets(card_id)
        self._cards[card_id] = widgets
        self.cards_layout.addWidget(widgets.container)
        # Prefer the shared app-state coordinate system for new cards, while
        # still preserving explicit card selections during later refreshes.
        self._refresh_card_selectors(widgets, preferred_coordinate_system=self._app_state.coordinate_system)
        self._update_card_state(card_id)
        self._update_empty_state()
        return card_id

    def remove_histogram_card(self, card_id: str) -> None:
        """Remove a card-local histogram request UI without touching SpatialData."""
        widgets = self._cards.pop(card_id, None)
        self._card_channel_errors.pop(card_id, None)
        self._last_emitted_requests.pop(card_id, None)
        if widgets is None:
            return

        self.cards_layout.removeWidget(widgets.container)
        widgets.container.deleteLater()
        self._update_empty_state()

    def build_request_for_card(self, card_id: str) -> HistogramCalculationRequest:
        """Build a validated calculation request for tests and later controllers."""
        widgets = self._get_card(card_id)
        return self._build_request(widgets)

    def _create_header_logo(self) -> QLabel:
        logo_label = QLabel()
        logo_label.setObjectName("histogram_header_logo")
        logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        logo_pixmap = QPixmap(str(self._logo_path))
        if not logo_pixmap.isNull():
            logo_label.setPixmap(logo_pixmap.scaledToWidth(120, Qt.TransformationMode.SmoothTransformation))
            return logo_label

        logo_label.setText("napari-harpy")
        logo_label.setStyleSheet(f"color: {WIDGET_TEXT_COLOR}; font-size: 18px; font-weight: 600;")
        return logo_label

    def _create_card_widgets(self, card_id: str) -> _HistogramCardWidgets:
        container = QFrame()
        container.setObjectName(f"histogram_card_{card_id}")
        container.setProperty("histogramCard", "true")
        container.setStyleSheet(_CARD_STYLESHEET)
        container.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)

        layout = QVBoxLayout(container)
        layout.setContentsMargins(12, 10, 12, 12)
        layout.setSpacing(10)

        header = QWidget()
        header.setObjectName(f"histogram_card_header_{card_id}")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(8)

        title_label = QLabel("Histogram")
        title_label.setObjectName(f"histogram_card_title_{card_id}")
        title_label.setStyleSheet(_CARD_TITLE_STYLESHEET)
        header_layout.addWidget(title_label, 1)

        remove_button = QToolButton()
        remove_button.setObjectName(f"histogram_remove_button_{card_id}")
        remove_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_TrashIcon))
        remove_button.setCursor(Qt.CursorShape.PointingHandCursor)
        remove_button.setToolTip(format_tooltip("Remove histogram"))
        remove_button.setAccessibleName("Remove histogram")
        remove_button.setStyleSheet(_ICON_BUTTON_STYLESHEET)
        remove_button.clicked.connect(
            lambda _checked=False, current_card_id=card_id: self.remove_histogram_card(current_card_id)
        )
        header_layout.addWidget(remove_button)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(8)

        coordinate_system_combo = self._create_combo(
            f"histogram_coordinate_system_combo_{card_id}",
            placeholder="Choose coordinate system",
        )
        coordinate_system_combo.currentIndexChanged.connect(
            lambda _index, current_card_id=card_id: self._on_card_coordinate_system_changed(current_card_id)
        )
        image_combo = self._create_combo(f"histogram_image_combo_{card_id}", placeholder="Choose image")
        image_combo.currentIndexChanged.connect(
            lambda _index, current_card_id=card_id: self._on_card_image_changed(current_card_id)
        )
        channel_combo = self._create_combo(f"histogram_channel_combo_{card_id}", placeholder="Choose channel")
        channel_combo.currentIndexChanged.connect(
            lambda _index, current_card_id=card_id: self._on_card_target_or_settings_changed(current_card_id)
        )

        form.addRow(create_form_label("Coordinate system"), coordinate_system_combo)
        form.addRow(create_form_label("Image"), image_combo)
        form.addRow(create_form_label("Channel"), channel_combo)

        settings_toggle = QToolButton()
        settings_toggle.setObjectName(f"histogram_settings_toggle_{card_id}")
        settings_toggle.setCheckable(True)
        settings_toggle.setChecked(False)
        settings_toggle.setArrowType(Qt.ArrowType.NoArrow)
        settings_toggle.setIconSize(QSize(DISCLOSURE_CHEVRON_SIZE, DISCLOSURE_CHEVRON_SIZE))
        settings_toggle.setIcon(create_disclosure_chevron_icon(expanded=False))
        settings_toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        settings_toggle.setStyleSheet(_SETTINGS_TOGGLE_STYLESHEET)
        settings_toggle.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        settings_panel = QWidget()
        settings_panel.setObjectName(f"histogram_settings_panel_{card_id}")
        settings_panel.setStyleSheet(_SETTINGS_PANEL_STYLESHEET)
        settings_panel.setVisible(False)
        settings_layout = QFormLayout(settings_panel)
        settings_layout.setContentsMargins(0, 0, 0, 0)
        settings_layout.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        settings_layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.FieldsStayAtSizeHint)
        settings_layout.setHorizontalSpacing(12)
        settings_layout.setVerticalSpacing(8)

        scale_combo = self._create_combo(f"histogram_scale_combo_{card_id}", placeholder=_DEFAULT_SCALE)
        self._set_compact_setting_control(scale_combo, _SETTING_SCALE_CONTROL_WIDTH)
        scale_combo.currentIndexChanged.connect(
            lambda _index, current_card_id=card_id: self._on_card_target_or_settings_changed(current_card_id)
        )

        bins_spin = QSpinBox()
        bins_spin.setObjectName(f"histogram_bins_spin_{card_id}")
        bins_spin.setRange(1, 1_000_000)
        bins_spin.setValue(_DEFAULT_SETTINGS.bins)
        bins_spin.setStyleSheet(_INPUT_CONTROL_STYLESHEET)
        self._set_compact_setting_control(bins_spin, _SETTING_BINS_CONTROL_WIDTH)
        bins_spin.valueChanged.connect(
            lambda _value, current_card_id=card_id: self._on_card_target_or_settings_changed(current_card_id)
        )

        value_range_low_edit = self._create_line_edit(f"histogram_value_range_low_edit_{card_id}", "auto")
        value_range_high_edit = self._create_line_edit(f"histogram_value_range_high_edit_{card_id}", "auto")
        percentile_min_edit = self._create_line_edit(f"histogram_percentile_min_edit_{card_id}", "optional")
        percentile_max_edit = self._create_line_edit(f"histogram_percentile_max_edit_{card_id}", "optional")
        for numeric_edit in (
            value_range_low_edit,
            value_range_high_edit,
            percentile_min_edit,
            percentile_max_edit,
        ):
            self._set_compact_setting_control(numeric_edit, _SETTING_NUMERIC_CONTROL_WIDTH)

        density_checkbox = self._create_checkbox(f"histogram_density_checkbox_{card_id}", "Density")
        exclude_nan_checkbox = self._create_checkbox(f"histogram_exclude_nan_checkbox_{card_id}", "Exclude NaN")
        exclude_nan_checkbox.setChecked(_DEFAULT_SETTINGS.exclude_nan)
        exclude_zeros_checkbox = self._create_checkbox(f"histogram_exclude_zeros_checkbox_{card_id}", "Exclude zeros")
        log_y_checkbox = self._create_checkbox(f"histogram_log_y_checkbox_{card_id}", "Log y")
        checkbox_panel = QWidget()
        checkbox_panel.setObjectName(f"histogram_filter_panel_{card_id}")
        checkbox_panel.setStyleSheet(_SETTINGS_PANEL_STYLESHEET)
        checkbox_layout = QGridLayout(checkbox_panel)
        checkbox_layout.setContentsMargins(0, 0, 0, 0)
        checkbox_layout.setHorizontalSpacing(14)
        checkbox_layout.setVerticalSpacing(6)
        checkbox_layout.addWidget(density_checkbox, 0, 0)
        checkbox_layout.addWidget(exclude_nan_checkbox, 0, 1)
        checkbox_layout.addWidget(exclude_zeros_checkbox, 1, 0)
        checkbox_layout.addWidget(log_y_checkbox, 1, 1)

        for line_edit in (
            value_range_low_edit,
            value_range_high_edit,
            percentile_min_edit,
            percentile_max_edit,
        ):
            line_edit.textChanged.connect(
                lambda _text, current_card_id=card_id: self._on_card_target_or_settings_changed(current_card_id)
            )
        for checkbox in (density_checkbox, exclude_nan_checkbox, exclude_zeros_checkbox, log_y_checkbox):
            checkbox.toggled.connect(
                lambda _checked, current_card_id=card_id: self._on_card_target_or_settings_changed(current_card_id)
            )

        reset_settings_button = QPushButton("Reset settings")
        reset_settings_button.setObjectName(f"histogram_reset_settings_button_{card_id}")
        reset_settings_button.setCursor(Qt.CursorShape.PointingHandCursor)
        reset_settings_button.setStyleSheet(SECONDARY_BUTTON_STYLESHEET)
        reset_settings_button.clicked.connect(
            lambda _checked=False, current_card_id=card_id: self._reset_card_settings(current_card_id)
        )

        settings_layout.addRow(create_form_label("Scale"), scale_combo)
        settings_layout.addRow(create_form_label("Bins"), bins_spin)
        settings_layout.addRow(create_form_label("Range min"), value_range_low_edit)
        settings_layout.addRow(create_form_label("Range max"), value_range_high_edit)
        settings_layout.addRow(create_form_label("Percentile min"), percentile_min_edit)
        settings_layout.addRow(create_form_label("Percentile max"), percentile_max_edit)
        settings_layout.addRow(checkbox_panel)
        settings_layout.addRow(reset_settings_button)

        settings_toggle.toggled.connect(
            lambda checked, panel=settings_panel, toggle=settings_toggle: self._set_settings_panel_visible(
                toggle, panel, checked
            )
        )

        action_row = QWidget()
        action_row.setObjectName(f"histogram_action_row_{card_id}")
        action_layout = QHBoxLayout(action_row)
        action_layout.setContentsMargins(0, 0, 0, 0)
        action_layout.setSpacing(8)

        calculate_button = QPushButton("Calculate")
        calculate_button.setObjectName(f"histogram_calculate_button_{card_id}")
        calculate_button.setCursor(Qt.CursorShape.PointingHandCursor)
        calculate_button.setStyleSheet(ACTION_BUTTON_STYLESHEET)
        calculate_button.setEnabled(False)
        calculate_button.clicked.connect(
            lambda _checked=False, current_card_id=card_id: self._emit_calculation_request(current_card_id)
        )
        action_layout.addWidget(calculate_button, 1)

        status_label = QLabel()
        status_label.setObjectName(f"histogram_status_label_{card_id}")
        status_label.setWordWrap(True)

        layout.addWidget(header)
        layout.addLayout(form)
        layout.addWidget(settings_toggle)
        layout.addWidget(settings_panel)
        layout.addWidget(action_row)
        layout.addWidget(status_label)

        widgets = _HistogramCardWidgets(
            card_id=card_id,
            container=container,
            title_label=title_label,
            coordinate_system_combo=coordinate_system_combo,
            image_combo=image_combo,
            channel_combo=channel_combo,
            calculate_button=calculate_button,
            remove_button=remove_button,
            status_label=status_label,
            settings_toggle=settings_toggle,
            settings_panel=settings_panel,
            scale_combo=scale_combo,
            bins_spin=bins_spin,
            value_range_low_edit=value_range_low_edit,
            value_range_high_edit=value_range_high_edit,
            density_checkbox=density_checkbox,
            exclude_nan_checkbox=exclude_nan_checkbox,
            exclude_zeros_checkbox=exclude_zeros_checkbox,
            log_y_checkbox=log_y_checkbox,
            percentile_min_edit=percentile_min_edit,
            percentile_max_edit=percentile_max_edit,
            reset_settings_button=reset_settings_button,
        )
        self._refresh_settings_summary(widgets)
        return widgets

    def _create_combo(self, object_name: str, *, placeholder: str) -> CompactComboBox:
        combo = CompactComboBox()
        combo.setObjectName(object_name)
        combo.setPlaceholderText(placeholder)
        combo.setStyleSheet(_INPUT_CONTROL_STYLESHEET)
        return combo

    def _create_line_edit(self, object_name: str, placeholder: str) -> QLineEdit:
        line_edit = QLineEdit()
        line_edit.setObjectName(object_name)
        line_edit.setPlaceholderText(placeholder)
        line_edit.setStyleSheet(_INPUT_CONTROL_STYLESHEET)
        return line_edit

    @staticmethod
    def _set_compact_setting_control(widget: QWidget, width: int) -> None:
        widget.setMinimumWidth(0)
        widget.setMaximumWidth(width)
        widget.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

    def _create_checkbox(self, object_name: str, text: str) -> QCheckBox:
        checkbox = QCheckBox(text)
        checkbox.setObjectName(object_name)
        checkbox.setStyleSheet(CHECKBOX_STYLESHEET)
        return checkbox

    def _get_card(self, card_id: str) -> _HistogramCardWidgets:
        try:
            return self._cards[card_id]
        except KeyError as error:
            raise ValueError(f"Histogram card `{card_id}` is not available.") from error

    def _on_sdata_changed(self, _sdata: SpatialData | None) -> None:
        for widgets in self._cards.values():
            self._refresh_card_selectors(widgets)
            self._update_card_state(widgets.card_id)

    def _on_coordinate_system_changed(self, event: CoordinateSystemChangedEvent) -> None:
        if event.sdata is not self._app_state.sdata or event.coordinate_system is None:
            return

        for widgets in self._cards.values():
            if self._current_text_data(widgets.coordinate_system_combo) is None:
                self._refresh_card_selectors(widgets, preferred_coordinate_system=self._app_state.coordinate_system)
                self._update_card_state(widgets.card_id)

    def _on_card_coordinate_system_changed(self, card_id: str) -> None:
        widgets = self._get_card(card_id)
        self._refresh_card_image_options(widgets)
        self._refresh_card_channel_options(widgets)
        self._refresh_card_scale_options(widgets)
        self._update_card_state(card_id)

    def _on_card_image_changed(self, card_id: str) -> None:
        widgets = self._get_card(card_id)
        self._refresh_card_channel_options(widgets)
        self._refresh_card_scale_options(widgets)
        self._update_card_state(card_id)

    def _on_card_target_or_settings_changed(self, card_id: str) -> None:
        widgets = self._get_card(card_id)
        self._refresh_settings_summary(widgets)
        self._update_card_state(card_id)

    def _refresh_card_selectors(
        self,
        widgets: _HistogramCardWidgets,
        *,
        preferred_coordinate_system: str | None = None,
    ) -> None:
        coordinate_systems = (
            []
            if self.selected_spatialdata is None
            else get_coordinate_system_names_from_sdata(self.selected_spatialdata)
        )
        current_coordinate_system = self._current_text_data(widgets.coordinate_system_combo)
        selected_coordinate_system = None
        if preferred_coordinate_system in coordinate_systems:
            selected_coordinate_system = preferred_coordinate_system
        elif current_coordinate_system in coordinate_systems:
            selected_coordinate_system = current_coordinate_system

        self._set_combo_items(widgets.coordinate_system_combo, coordinate_systems, selected_coordinate_system)
        self._refresh_card_image_options(widgets)
        self._refresh_card_channel_options(widgets)
        self._refresh_card_scale_options(widgets)

    def _refresh_card_image_options(self, widgets: _HistogramCardWidgets) -> None:
        sdata = self.selected_spatialdata
        coordinate_system = self._current_text_data(widgets.coordinate_system_combo)
        current_image_name = self._current_text_data(widgets.image_combo)
        items: list[tuple[str, str]] = []
        if sdata is not None and coordinate_system is not None:
            items = [
                (option.display_name, option.image_name)
                for option in get_spatialdata_image_options_for_coordinate_system_from_sdata(
                    sdata=sdata,
                    coordinate_system=coordinate_system,
                )
            ]

        valid_image_names = {image_name for _display_name, image_name in items}
        selected_image_name = current_image_name if current_image_name in valid_image_names else None
        self._set_combo_items(widgets.image_combo, items, selected_image_name)

    def _refresh_card_channel_options(self, widgets: _HistogramCardWidgets) -> None:
        sdata = self.selected_spatialdata
        image_name = self._current_text_data(widgets.image_combo)
        current_channel_name = self._current_text_data(widgets.channel_combo)
        channel_names: list[str] = []
        channel_error: str | None = None

        if sdata is not None and image_name is not None:
            try:
                channel_names = get_image_channel_names_from_sdata(sdata, image_name)
            except ValueError as error:
                channel_error = str(error)
            else:
                if not channel_names:
                    channel_error = f"Image `{image_name}` does not expose channel names."

        selected_channel_name = current_channel_name if current_channel_name in channel_names else None
        self._set_combo_items(widgets.channel_combo, channel_names, selected_channel_name)
        self._card_channel_errors[widgets.card_id] = channel_error or ""

    def _refresh_card_scale_options(self, widgets: _HistogramCardWidgets) -> None:
        sdata = self.selected_spatialdata
        image_name = self._current_text_data(widgets.image_combo)
        current_scale = self._current_text_data(widgets.scale_combo)
        scales = [_DEFAULT_SCALE]

        if sdata is not None and image_name is not None:
            image_element = sdata.images[image_name]
            if isinstance(image_element, DataArray):
                scales = [_DEFAULT_SCALE]
            elif isinstance(image_element, DataTree):
                scales = [str(scale) for scale in image_element.keys()]

        selected_scale = (
            current_scale if current_scale in scales else (_DEFAULT_SCALE if _DEFAULT_SCALE in scales else None)
        )
        self._set_combo_items(widgets.scale_combo, scales, selected_scale)
        widgets.scale_combo.setEnabled(sdata is not None and image_name is not None and bool(scales))
        self._refresh_settings_summary(widgets)

    def _set_combo_items(
        self,
        combo: CompactComboBox,
        items: list[str] | list[tuple[str, str]],
        selected_data: str | None,
    ) -> None:
        with QSignalBlocker(combo):
            combo.clear()
            for item in items:
                if isinstance(item, tuple):
                    combo.addItem(item[0], item[1])
                else:
                    combo.addItem(item, item)

            selected_index = -1
            if selected_data is not None:
                for index in range(combo.count()):
                    if combo.itemData(index) == selected_data:
                        selected_index = index
                        break
            combo.setCurrentIndex(selected_index)

    @staticmethod
    def _current_text_data(combo: CompactComboBox) -> str | None:
        data = combo.currentData()
        return data if isinstance(data, str) and data else None

    def _set_settings_panel_visible(self, toggle: QToolButton, panel: QWidget, checked: bool) -> None:
        toggle.setIcon(create_disclosure_chevron_icon(expanded=checked))
        panel.setVisible(checked)

    def _reset_card_settings(self, card_id: str) -> None:
        widgets = self._get_card(card_id)
        with QSignalBlocker(widgets.bins_spin):
            widgets.bins_spin.setValue(_DEFAULT_SETTINGS.bins)
        for line_edit in (
            widgets.value_range_low_edit,
            widgets.value_range_high_edit,
            widgets.percentile_min_edit,
            widgets.percentile_max_edit,
        ):
            with QSignalBlocker(line_edit):
                line_edit.clear()
        checkbox_defaults = {
            widgets.density_checkbox: _DEFAULT_SETTINGS.density,
            widgets.exclude_nan_checkbox: _DEFAULT_SETTINGS.exclude_nan,
            widgets.exclude_zeros_checkbox: _DEFAULT_SETTINGS.exclude_zeros,
            widgets.log_y_checkbox: _DEFAULT_SETTINGS.log_y,
        }
        for checkbox, checked in checkbox_defaults.items():
            with QSignalBlocker(checkbox):
                checkbox.setChecked(checked)
        if widgets.scale_combo.count():
            selected_scale = _DEFAULT_SCALE if self._find_combo_data(widgets.scale_combo, _DEFAULT_SCALE) >= 0 else None
            with QSignalBlocker(widgets.scale_combo):
                widgets.scale_combo.setCurrentIndex(self._find_combo_data(widgets.scale_combo, selected_scale))

        self._refresh_settings_summary(widgets)
        self._update_card_state(card_id)

    @staticmethod
    def _find_combo_data(combo: CompactComboBox, data: str | None) -> int:
        if data is None:
            return -1
        for index in range(combo.count()):
            if combo.itemData(index) == data:
                return index
        return -1

    def _refresh_settings_summary(self, widgets: _HistogramCardWidgets) -> None:
        widgets.settings_toggle.setText("Settings")
        widgets.settings_toggle.setToolTip(format_tooltip("\n".join(self._settings_tooltip_lines(widgets))))

    def _settings_tooltip_lines(self, widgets: _HistogramCardWidgets) -> tuple[str, ...]:
        scale = self._current_text_data(widgets.scale_combo) or _DEFAULT_SCALE
        return (
            f"scale: {scale}",
            f"bins: {widgets.bins_spin.value()}",
            f"value_range: {self._value_range_tooltip_text(widgets)}",
            f"density: {widgets.density_checkbox.isChecked()}",
            f"exclude_nan: {widgets.exclude_nan_checkbox.isChecked()}",
            f"exclude_zeros: {widgets.exclude_zeros_checkbox.isChecked()}",
            f"log_y: {widgets.log_y_checkbox.isChecked()}",
            f"percentiles: {self._percentiles_tooltip_text(widgets)}",
        )

    def _value_range_tooltip_text(self, widgets: _HistogramCardWidgets) -> str:
        low_text = widgets.value_range_low_edit.text().strip()
        high_text = widgets.value_range_high_edit.text().strip()
        if not low_text and not high_text:
            return "auto"
        if not low_text or not high_text:
            return f"incomplete (low={low_text or 'auto'}, high={high_text or 'auto'})"
        return f"({low_text}, {high_text})"

    def _percentiles_tooltip_text(self, widgets: _HistogramCardWidgets) -> str:
        percentiles = tuple(
            text
            for text in (
                widgets.percentile_min_edit.text().strip(),
                widgets.percentile_max_edit.text().strip(),
            )
            if text
        )
        if not percentiles:
            return "none"
        return ", ".join(percentiles)

    def _update_card_state(self, card_id: str) -> None:
        widgets = self._get_card(card_id)
        try:
            request = self._build_request(widgets)
        except ValueError as error:
            widgets.calculate_button.setEnabled(False)
            set_status_card(
                widgets.status_label,
                title="Histogram Incomplete",
                lines=[str(error)],
                kind="warning",
            )
            return

        widgets.calculate_button.setEnabled(True)
        last_request = self._last_emitted_requests.get(card_id)
        if last_request is None:
            message = "Ready to calculate."
        elif last_request == request:
            message = "Calculation request emitted."
        else:
            message = "Target or settings changed. Calculate again."
        set_status_card(
            widgets.status_label,
            title="Histogram Ready",
            lines=[message],
            kind="success" if last_request == request else "info",
        )

    def _build_request(self, widgets: _HistogramCardWidgets) -> HistogramCalculationRequest:
        if self.selected_spatialdata is None:
            raise ValueError("No SpatialData loaded.")

        coordinate_system = self._current_text_data(widgets.coordinate_system_combo)
        if coordinate_system is None:
            raise ValueError("Choose a coordinate system.")

        image_name = self._current_text_data(widgets.image_combo)
        if image_name is None:
            raise ValueError("Choose an image.")

        channel_error = self._card_channel_errors.get(widgets.card_id)
        if channel_error:
            raise ValueError(channel_error)

        channel_name = self._current_text_data(widgets.channel_combo)
        if channel_name is None:
            raise ValueError("Choose a channel.")

        target = HistogramTarget(
            coordinate_system=coordinate_system,
            image_name=image_name,
            channel_name=channel_name,
        )
        settings = self._build_settings(widgets)
        return HistogramCalculationRequest(card_id=widgets.card_id, target=target, settings=settings)

    def _build_settings(self, widgets: _HistogramCardWidgets) -> HistogramSettings:
        value_range = self._parse_optional_pair(
            widgets.value_range_low_edit.text(),
            widgets.value_range_high_edit.text(),
            field_name="value range",
        )
        percentiles = tuple(
            self._parse_float(text, field_name="percentile")
            for text in (
                widgets.percentile_min_edit.text().strip(),
                widgets.percentile_max_edit.text().strip(),
            )
            if text
        )
        return HistogramSettings(
            bins=widgets.bins_spin.value(),
            value_range=value_range,
            density=widgets.density_checkbox.isChecked(),
            exclude_nan=widgets.exclude_nan_checkbox.isChecked(),
            exclude_zeros=widgets.exclude_zeros_checkbox.isChecked(),
            log_y=widgets.log_y_checkbox.isChecked(),
            scale=self._current_text_data(widgets.scale_combo) or _DEFAULT_SCALE,
            percentiles=percentiles,
        )

    def _parse_optional_pair(
        self,
        low_text: str,
        high_text: str,
        *,
        field_name: str,
    ) -> tuple[float, float] | None:
        low = low_text.strip()
        high = high_text.strip()
        if not low and not high:
            return None
        if not low or not high:
            raise ValueError(f"Optional {field_name} requires both low and high values.")
        return (
            self._parse_float(low, field_name=f"{field_name} low"),
            self._parse_float(high, field_name=f"{field_name} high"),
        )

    @staticmethod
    def _parse_float(text: str, *, field_name: str) -> float:
        try:
            return float(text)
        except ValueError as error:
            raise ValueError(f"Histogram {field_name} must be a number.") from error

    def _emit_calculation_request(self, card_id: str) -> None:
        widgets = self._get_card(card_id)
        request = self._build_request(widgets)
        self._last_emitted_requests[card_id] = request
        self.calculation_requested.emit(request)
        self._update_card_state(card_id)

    def _update_empty_state(self) -> None:
        has_cards = bool(self._cards)
        self.empty_state_label.setVisible(not has_cards)
        self.cards_container.setVisible(has_cards)
