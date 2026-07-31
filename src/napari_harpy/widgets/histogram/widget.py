from __future__ import annotations

import math
import uuid
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Literal

from qtpy.QtCore import QSignalBlocker, QSize, QStringListModel, Qt
from qtpy.QtGui import QPixmap
from qtpy.QtWidgets import (
    QCheckBox,
    QCompleter,
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
from napari_harpy.core.histogram import (
    HistogramResult,
    HistogramSettings,
    HistogramTarget,
    resolve_default_histogram_scale,
)
from napari_harpy.core.spatialdata import (
    get_coordinate_system_names_from_sdata,
    get_image_channel_names_from_sdata,
    get_spatialdata_image_options_for_coordinate_system_from_sdata,
)
from napari_harpy.viewer.adapter import ImageLayerBinding
from napari_harpy.viewer.image_styling import DEFAULT_OVERLAY_COLORS
from napari_harpy.widgets.histogram.controller import HistogramController
from napari_harpy.widgets.histogram.plot_widget import _HistogramPlotWidget
from napari_harpy.widgets.histogram.status_card import (
    _HistogramStatusCardSpec,
    build_histogram_calculated_card_spec,
    build_histogram_error_card_spec,
    build_histogram_incomplete_card_spec,
    build_histogram_ready_card_spec,
    build_histogram_running_card_spec,
)
from napari_harpy.widgets.overlay_channel_row import (
    _binding_channel_index,
    _binding_channel_name,
    _normalized_color_or_none,
    _OverlayChannelRow,
    _solid_color_from_layer,
)
from napari_harpy.widgets.overlay_color_button import OverlayColorButton
from napari_harpy.widgets.shared_styles import (
    ACTION_BUTTON_STYLESHEET,
    CALCULATE_BUTTON_STYLESHEET,
    CHECKBOX_STYLESHEET,
    COMPLETER_POPUP_STYLESHEET,
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
    WIDGET_WARNING_TEXT_COLOR,
    CompactComboBox,
    CompleterPopupLineEdit,
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
_CARD_SUBCONTAINER_STYLESHEET = "QWidget { background: transparent; }"
_SETTINGS_PANEL_STYLESHEET = "QWidget { background: transparent; }"
_SETTINGS_TOGGLE_STYLESHEET = (
    "QToolButton {"
    f"background-color: {WIDGET_PANEL_COLOR}; "
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


@dataclass
class _HistogramContrastSyncState:
    layer: object
    contrast_limits_callback: Callable[[object], None]
    updating_plot: bool = False


@dataclass(frozen=True)
class _HistogramOverlayMatch:
    """Describe how one Histogram target matches live overlay bindings.

    ``invalid`` means the card does not define a valid overlay target.
    ``missing`` means the target is valid but no corresponding layer is loaded.
    ``unique`` means exactly one live binding exists and is available in
    ``binding``. ``ambiguous`` means multiple live bindings match the target, so
    Histogram cannot safely select one.
    """

    state: Literal["invalid", "missing", "unique", "ambiguous"]
    binding: ImageLayerBinding | None = None
    message: str | None = None

    def __post_init__(self) -> None:
        if self.state not in ("invalid", "missing", "unique", "ambiguous"):
            raise ValueError(
                f"`state` must be one of: 'invalid', 'missing', 'unique', or 'ambiguous'; got {self.state!r}."
            )


@dataclass
class _HistogramCard:
    """Own one Histogram target's UI and optional fixed-binding overlay row.

    The card's Viewer section is reconciled from
    ``_HistogramOverlayMatch.state``:

    - ``missing`` shows ``pending_overlay_controls``, containing the initial
      color picker and `Load in viewer` button;
    - ``unique`` replaces the pending controls with ``overlay_row``, which
      presents visibility, channel name, colormap, and removal controls;
    - ``ambiguous`` replaces the editing controls with
      ``overlay_state_label``;
    - ``invalid`` shows the pending controls in a disabled state.

    ``viewer_controls`` and ``viewer_controls_layout`` provide the stable
    container into which the live overlay row is inserted. ``overlay_row``
    exists only while the current target resolves to exactly one live binding.
    The row owns visibility/colormap presentation callbacks;
    ``HistogramWidget`` owns membership reconciliation and validated layer
    mutations.

    ``channel_search_input`` may contain transient search text.
    ``accepted_channel_name`` is the persistent selection used for Histogram
    and overlay target resolution; typing alone never changes that target.

    Contrast synchronization remains separate because it belongs to the
    calculated Histogram plot rather than the overlay row.
    """

    card_id: str
    container: QFrame
    coordinate_system_combo: CompactComboBox
    image_combo: CompactComboBox
    channel_search_input: CompleterPopupLineEdit
    channel_completer_model: QStringListModel
    accepted_channel_name: str | None
    viewer_controls: QWidget
    viewer_controls_layout: QVBoxLayout
    pending_overlay_controls: QWidget
    overlay_state_label: QLabel
    overlay_color_button: OverlayColorButton
    load_overlay_button: QPushButton
    calculate_button: QPushButton
    sync_percentiles_button: QPushButton
    remove_button: QToolButton
    plot_widget: _HistogramPlotWidget
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
    viewer_action_message: str | None = None
    overlay_row: _OverlayChannelRow | None = None
    contrast_sync_state: _HistogramContrastSyncState | None = None
    contrast_sync_message: str | None = None

    @property
    def channel_names(self) -> tuple[str, ...]:
        """Return the channels currently exposed by the completer model."""
        return tuple(self.channel_completer_model.stringList())


class HistogramWidget(QWidget):
    """Qt shell for explicit per-card image histogram and overlay requests.

    Adapter ``image_layers_changed`` events trigger card membership
    reconciliation. Histogram intersects registry matches with layers currently
    present in napari, and ignores stack-only membership. Each live
    ``_OverlayChannelRow`` listens directly to its
    bound layer's visibility and colormap events. User intent signals are handled
    by ``HistogramWidget``, which validates the current live binding before
    changing its visibility or colormap, or requesting its removal through
    ``ViewerAdapter``. Histogram cards never exchange presentation events with
    Viewer cards.
    """

    def __init__(self, napari_viewer: napari.Viewer | None = None) -> None:
        super().__init__()
        self.setObjectName("histogram_widget")
        apply_widget_surface(self)
        self.setMinimumWidth(WIDGET_MIN_WIDTH)

        self._app_state = get_or_create_app_state(napari_viewer)
        self._histogram_controller = HistogramController(on_state_changed=self._on_controller_state_changed)
        self._cards: dict[str, _HistogramCard] = {}
        self._card_channel_errors: dict[str, str] = {}
        self._card_scale_errors: dict[str, str] = {}
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
        action_row.setObjectName("histogram_add_action_row")
        action_layout = QHBoxLayout(action_row)
        action_layout.setContentsMargins(0, 0, 0, 0)
        action_layout.setSpacing(8)
        self.add_button = QPushButton("Add")
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
        self._app_state.viewer_adapter.image_layers_changed.connect(self._on_image_layers_changed)
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
        histogram_card = self._create_histogram_card(card_id)
        self._cards[card_id] = histogram_card
        self.cards_layout.addWidget(histogram_card.container)
        # Prefer the shared app-state coordinate system for new cards, while
        # still preserving explicit card selections during later refreshes.
        self._refresh_card_selectors(histogram_card, preferred_coordinate_system=self._app_state.coordinate_system)
        self._update_card_state(card_id)
        self._update_empty_state()
        return card_id

    def remove_histogram_card(self, card_id: str) -> None:
        """Remove a card-local histogram request UI without touching SpatialData."""
        histogram_card = self._cards.pop(card_id, None)
        self._card_channel_errors.pop(card_id, None)
        self._card_scale_errors.pop(card_id, None)
        self._histogram_controller.remove_card(card_id)
        if histogram_card is None:
            return

        self._clear_card_contrast_sync(histogram_card)
        self._dispose_card_overlay_row(histogram_card)
        self.cards_layout.removeWidget(histogram_card.container)
        histogram_card.container.deleteLater()
        self._update_empty_state()

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

    def _create_histogram_card(self, card_id: str) -> _HistogramCard:
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
        header.setStyleSheet(_CARD_SUBCONTAINER_STYLESHEET)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(8)

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
        header_layout.addStretch(1)
        header_layout.addWidget(remove_button)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
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
        channel_search_input = CompleterPopupLineEdit()
        channel_search_input.setObjectName(f"histogram_channel_search_input_{card_id}")
        channel_search_input.setPlaceholderText("Search or select channel")
        channel_search_input.setStyleSheet(build_input_control_stylesheet("QLineEdit"))
        channel_search_input.setMinimumWidth(0)
        channel_search_input.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        channel_search_input.set_completion_popup_on_entry_enabled(True)

        channel_completer_model = QStringListModel(channel_search_input)
        channel_completer = QCompleter(channel_completer_model, channel_search_input)
        channel_completer.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
        channel_completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        channel_completer.setFilterMode(Qt.MatchFlag.MatchContains)
        channel_completer.setMaxVisibleItems(10)
        channel_completer.popup().setStyleSheet(COMPLETER_POPUP_STYLESHEET)
        channel_search_input.setCompleter(channel_completer)
        channel_completer.activated[str].connect(
            lambda text, current_card_id=card_id: self._accept_card_channel_text(current_card_id, text)
        )
        channel_search_input.returnPressed.connect(
            lambda current_card_id=card_id, current_input=channel_search_input: self._accept_card_channel_text(
                current_card_id,
                current_input.text(),
            )
        )
        channel_search_input.editingFinished.connect(
            lambda current_card_id=card_id: self._restore_card_channel_text_by_id(current_card_id)
        )

        overlay_color_button = OverlayColorButton(DEFAULT_OVERLAY_COLORS[0])
        overlay_color_button.setObjectName(f"histogram_overlay_color_button_{card_id}")
        overlay_color_button.setEnabled(False)

        load_overlay_button = QPushButton("Load in viewer")
        load_overlay_button.setObjectName(f"histogram_load_overlay_button_{card_id}")
        load_overlay_button.setCursor(Qt.CursorShape.PointingHandCursor)
        load_overlay_button.setStyleSheet(ACTION_BUTTON_STYLESHEET)
        load_overlay_button.setToolTip(format_tooltip("Load selected channel as a napari overlay layer"))
        load_overlay_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        load_overlay_button.setEnabled(False)
        load_overlay_button.clicked.connect(
            lambda _checked=False, current_card_id=card_id: self._load_card_overlay(current_card_id)
        )

        viewer_controls = QWidget()
        viewer_controls.setObjectName(f"histogram_viewer_controls_{card_id}")
        viewer_controls.setStyleSheet(_CARD_SUBCONTAINER_STYLESHEET)
        viewer_controls.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        viewer_controls_layout = QVBoxLayout(viewer_controls)
        viewer_controls_layout.setContentsMargins(0, 0, 0, 0)
        viewer_controls_layout.setSpacing(8)

        pending_overlay_controls = QWidget(viewer_controls)
        pending_overlay_controls.setObjectName(f"histogram_pending_overlay_controls_{card_id}")
        pending_overlay_controls_layout = QHBoxLayout(pending_overlay_controls)
        pending_overlay_controls_layout.setContentsMargins(0, 0, 0, 0)
        pending_overlay_controls_layout.setSpacing(8)
        pending_overlay_controls_layout.addWidget(load_overlay_button, 1)
        pending_overlay_controls_layout.addWidget(overlay_color_button)

        overlay_state_label = QLabel(viewer_controls)
        overlay_state_label.setObjectName(f"histogram_overlay_state_label_{card_id}")
        overlay_state_label.setWordWrap(True)
        overlay_state_label.setStyleSheet(
            f"color: {WIDGET_WARNING_TEXT_COLOR}; "
            f"background-color: {WIDGET_PANEL_SUBTLE_COLOR}; "
            f"border: 1px solid {WIDGET_BORDER_COLOR}; "
            "border-radius: 6px; "
            "padding: 6px;"
        )
        overlay_state_label.setVisible(False)

        viewer_controls_layout.addWidget(pending_overlay_controls)
        viewer_controls_layout.addWidget(overlay_state_label)

        form.addRow(create_form_label("Coordinate system"), coordinate_system_combo)
        form.addRow(create_form_label("Image"), image_combo)
        form.addRow(create_form_label("Channel"), channel_search_input)
        form.addRow(create_form_label("Viewer"), viewer_controls)

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
        log_y_checkbox = self._create_checkbox(f"histogram_log_y_checkbox_{card_id}", "Log scale")
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
        action_row.setStyleSheet(_CARD_SUBCONTAINER_STYLESHEET)
        action_layout = QHBoxLayout(action_row)
        action_layout.setContentsMargins(0, 0, 0, 0)
        action_layout.setSpacing(8)

        calculate_button = QPushButton("Show histogram")
        calculate_button.setObjectName(f"histogram_calculate_button_{card_id}")
        calculate_button.setCursor(Qt.CursorShape.PointingHandCursor)
        calculate_button.setStyleSheet(CALCULATE_BUTTON_STYLESHEET)
        calculate_button.setEnabled(False)
        calculate_button.clicked.connect(
            lambda _checked=False, current_card_id=card_id: self._calculate_histogram(current_card_id)
        )
        action_layout.addWidget(calculate_button, 2)

        sync_percentiles_button = QPushButton("Sync contrast limits")
        sync_percentiles_button.setObjectName(f"histogram_sync_percentiles_button_{card_id}")
        sync_percentiles_button.setCursor(Qt.CursorShape.PointingHandCursor)
        sync_percentiles_button.setStyleSheet(CALCULATE_BUTTON_STYLESHEET)
        sync_percentiles_button.setEnabled(False)
        sync_percentiles_button.clicked.connect(
            lambda _checked=False, current_card_id=card_id: self._sync_percentiles_to_contrast_limits(current_card_id)
        )
        action_layout.addWidget(sync_percentiles_button, 1)

        status_label = QLabel()
        status_label.setObjectName(f"histogram_status_label_{card_id}")
        status_label.setWordWrap(True)

        plot_widget = _HistogramPlotWidget()
        plot_widget.setObjectName(f"histogram_plot_widget_{card_id}")
        plot_widget.contrast_limits_dragged.connect(
            lambda low, high, current_card_id=card_id: self._on_plot_contrast_limits_dragged(current_card_id, low, high)
        )

        layout.addWidget(header)
        layout.addLayout(form)
        layout.addWidget(settings_toggle)
        layout.addWidget(settings_panel)
        layout.addWidget(action_row)
        layout.addWidget(plot_widget)
        layout.addWidget(status_label)

        histogram_card = _HistogramCard(
            card_id=card_id,
            container=container,
            coordinate_system_combo=coordinate_system_combo,
            image_combo=image_combo,
            channel_search_input=channel_search_input,
            channel_completer_model=channel_completer_model,
            accepted_channel_name=None,
            viewer_controls=viewer_controls,
            viewer_controls_layout=viewer_controls_layout,
            pending_overlay_controls=pending_overlay_controls,
            overlay_state_label=overlay_state_label,
            overlay_color_button=overlay_color_button,
            load_overlay_button=load_overlay_button,
            calculate_button=calculate_button,
            sync_percentiles_button=sync_percentiles_button,
            remove_button=remove_button,
            plot_widget=plot_widget,
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
        self._refresh_settings_summary(histogram_card)
        return histogram_card

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

    def _get_card(self, card_id: str) -> _HistogramCard:
        try:
            return self._cards[card_id]
        except KeyError as error:
            raise ValueError(f"Histogram card `{card_id}` is not available.") from error

    def _on_sdata_changed(self, _sdata: SpatialData | None) -> None:
        for histogram_card in self._cards.values():
            self._dispose_card_overlay_row(histogram_card)
            self._refresh_card_selectors(histogram_card, select_default_scale=True)
            self._update_card_state(histogram_card.card_id)

    def _on_coordinate_system_changed(self, event: CoordinateSystemChangedEvent) -> None:
        if event.sdata is not self._app_state.sdata or event.coordinate_system is None:
            return

        for histogram_card in self._cards.values():
            if self._current_text_data(histogram_card.coordinate_system_combo) is None:
                self._refresh_card_selectors(
                    histogram_card, preferred_coordinate_system=self._app_state.coordinate_system
                )
                self._update_card_state(histogram_card.card_id)

    def _on_card_coordinate_system_changed(self, card_id: str) -> None:
        histogram_card = self._get_card(card_id)
        self._dispose_card_overlay_row(histogram_card)
        histogram_card.viewer_action_message = None
        self._refresh_card_image_options(histogram_card)
        self._refresh_card_channel_options(histogram_card)
        self._refresh_card_scale_options(histogram_card)
        self._update_card_state(card_id)

    def _on_card_image_changed(self, card_id: str) -> None:
        histogram_card = self._get_card(card_id)
        self._dispose_card_overlay_row(histogram_card)
        histogram_card.viewer_action_message = None
        self._refresh_card_channel_options(histogram_card)
        self._refresh_card_scale_options(histogram_card, select_default=True)
        self._update_card_state(card_id)

    def _accept_card_channel_text(self, card_id: str, text: str) -> None:
        histogram_card = self._cards.get(card_id)
        if histogram_card is None or not histogram_card.channel_search_input.isEnabled():
            return

        channel_name = self._resolve_card_channel_text(histogram_card, text)
        if channel_name is None:
            self._restore_card_channel_text(histogram_card)
            return
        if channel_name == histogram_card.accepted_channel_name:
            histogram_card.channel_search_input.setText(channel_name)
            return

        # A row belongs to the previously accepted target. Disconnect it before
        # changing target identity; reconciliation may construct a replacement.
        self._dispose_card_overlay_row(histogram_card)
        histogram_card.accepted_channel_name = channel_name
        histogram_card.channel_search_input.setText(channel_name)
        histogram_card.viewer_action_message = None
        self._refresh_overlay_color_default(histogram_card)
        self._on_card_target_or_settings_changed(card_id)

    @staticmethod
    def _resolve_card_channel_text(histogram_card: _HistogramCard, text: str) -> str | None:
        normalized_text = text.strip()
        if not normalized_text:
            return None

        for channel_name in histogram_card.channel_names:
            if channel_name == normalized_text:
                return channel_name

        matches = [
            channel_name
            for channel_name in histogram_card.channel_names
            if channel_name.casefold() == normalized_text.casefold()
        ]
        return matches[0] if len(matches) == 1 else None

    def _restore_card_channel_text_by_id(self, card_id: str) -> None:
        histogram_card = self._cards.get(card_id)
        if histogram_card is not None:
            self._restore_card_channel_text(histogram_card)

    @staticmethod
    def _restore_card_channel_text(histogram_card: _HistogramCard) -> None:
        accepted_text = histogram_card.accepted_channel_name or ""
        if histogram_card.channel_search_input.text() != accepted_text:
            histogram_card.channel_search_input.setText(accepted_text)

    def _on_card_target_or_settings_changed(self, card_id: str) -> None:
        histogram_card = self._get_card(card_id)
        self._refresh_settings_summary(histogram_card)
        self._update_card_state(card_id)

    def _refresh_card_selectors(
        self,
        histogram_card: _HistogramCard,
        *,
        preferred_coordinate_system: str | None = None,
        select_default_scale: bool = False,
    ) -> None:
        coordinate_systems = (
            []
            if self.selected_spatialdata is None
            else get_coordinate_system_names_from_sdata(self.selected_spatialdata)
        )
        current_coordinate_system = self._current_text_data(histogram_card.coordinate_system_combo)
        selected_coordinate_system = None
        if preferred_coordinate_system in coordinate_systems:
            selected_coordinate_system = preferred_coordinate_system
        elif current_coordinate_system in coordinate_systems:
            selected_coordinate_system = current_coordinate_system

        self._set_combo_items(histogram_card.coordinate_system_combo, coordinate_systems, selected_coordinate_system)
        self._refresh_card_image_options(histogram_card)
        self._refresh_card_channel_options(histogram_card)
        self._refresh_card_scale_options(histogram_card, select_default=select_default_scale)

    def _refresh_card_image_options(self, histogram_card: _HistogramCard) -> None:
        sdata = self.selected_spatialdata
        coordinate_system = self._current_text_data(histogram_card.coordinate_system_combo)
        current_image_name = self._current_text_data(histogram_card.image_combo)
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
        self._set_combo_items(histogram_card.image_combo, items, selected_image_name)

    def _refresh_card_channel_options(self, histogram_card: _HistogramCard) -> None:
        sdata = self.selected_spatialdata
        image_name = self._current_text_data(histogram_card.image_combo)
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

        if (
            histogram_card.accepted_channel_name is not None
            and histogram_card.accepted_channel_name not in channel_names
        ):
            self._dispose_card_overlay_row(histogram_card)
            histogram_card.accepted_channel_name = None

        histogram_card.channel_completer_model.setStringList(channel_names)
        histogram_card.channel_search_input.setEnabled(bool(channel_names) and channel_error is None)
        self._restore_card_channel_text(histogram_card)
        self._card_channel_errors[histogram_card.card_id] = channel_error or ""
        self._refresh_overlay_color_default(histogram_card)

    def _refresh_card_scale_options(
        self,
        histogram_card: _HistogramCard,
        *,
        select_default: bool = False,
    ) -> None:
        sdata = self.selected_spatialdata
        image_name = self._current_text_data(histogram_card.image_combo)
        current_scale = self._current_text_data(histogram_card.scale_combo)
        scales = [_DEFAULT_SCALE]
        default_scale: str | None = _DEFAULT_SCALE
        scale_error: str | None = None

        if sdata is not None and image_name is not None:
            image_element = sdata.images[image_name]
            if isinstance(image_element, DataArray):
                scales = [_DEFAULT_SCALE]
            elif isinstance(image_element, DataTree):
                scales = [str(scale) for scale in image_element.keys()]
            try:
                default_scale = resolve_default_histogram_scale(image_element, image_name=image_name)
            except ValueError as error:
                default_scale = None
                scale_error = str(error)

        selected_scale = current_scale if not select_default and current_scale in scales else default_scale
        self._set_combo_items(histogram_card.scale_combo, scales, selected_scale)
        self._card_scale_errors[histogram_card.card_id] = scale_error or ""
        histogram_card.scale_combo.setEnabled(
            sdata is not None and image_name is not None and bool(scales) and scale_error is None
        )
        self._refresh_settings_summary(histogram_card)

    def _refresh_overlay_color_default(self, histogram_card: _HistogramCard) -> None:
        channel_name = histogram_card.accepted_channel_name
        if channel_name is None:
            histogram_card.overlay_color_button.set_color(DEFAULT_OVERLAY_COLORS[0])
            return

        channel_index = histogram_card.channel_names.index(channel_name)
        histogram_card.overlay_color_button.set_color(
            DEFAULT_OVERLAY_COLORS[channel_index % len(DEFAULT_OVERLAY_COLORS)]
        )

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
        histogram_card = self._get_card(card_id)
        with QSignalBlocker(histogram_card.bins_spin):
            histogram_card.bins_spin.setValue(_DEFAULT_SETTINGS.bins)
        for line_edit in (
            histogram_card.value_range_low_edit,
            histogram_card.value_range_high_edit,
            histogram_card.percentile_min_edit,
            histogram_card.percentile_max_edit,
        ):
            with QSignalBlocker(line_edit):
                line_edit.clear()
        checkbox_defaults = {
            histogram_card.density_checkbox: _DEFAULT_SETTINGS.density,
            histogram_card.exclude_nan_checkbox: _DEFAULT_SETTINGS.exclude_nan,
            histogram_card.exclude_zeros_checkbox: _DEFAULT_SETTINGS.exclude_zeros,
            histogram_card.log_y_checkbox: _DEFAULT_SETTINGS.log_y,
        }
        for checkbox, checked in checkbox_defaults.items():
            with QSignalBlocker(checkbox):
                checkbox.setChecked(checked)
        self._refresh_card_scale_options(histogram_card, select_default=True)

        self._update_card_state(card_id)

    @staticmethod
    def _find_combo_data(combo: CompactComboBox, data: str | None) -> int:
        if data is None:
            return -1
        for index in range(combo.count()):
            if combo.itemData(index) == data:
                return index
        return -1

    def _refresh_settings_summary(self, histogram_card: _HistogramCard) -> None:
        histogram_card.settings_toggle.setText("Histogram Settings")
        histogram_card.settings_toggle.setToolTip(
            format_tooltip("\n".join(self._settings_tooltip_lines(histogram_card)))
        )

    def _settings_tooltip_lines(self, histogram_card: _HistogramCard) -> tuple[str, ...]:
        scale = self._current_text_data(histogram_card.scale_combo) or _DEFAULT_SCALE
        return (
            f"scale: {scale}",
            f"bins: {histogram_card.bins_spin.value()}",
            f"value_range: {self._value_range_tooltip_text(histogram_card)}",
            f"density: {histogram_card.density_checkbox.isChecked()}",
            f"exclude_nan: {histogram_card.exclude_nan_checkbox.isChecked()}",
            f"exclude_zeros: {histogram_card.exclude_zeros_checkbox.isChecked()}",
            f"log_y: {histogram_card.log_y_checkbox.isChecked()}",
            f"percentiles: {self._percentiles_tooltip_text(histogram_card)}",
        )

    def _value_range_tooltip_text(self, histogram_card: _HistogramCard) -> str:
        low_text = histogram_card.value_range_low_edit.text().strip()
        high_text = histogram_card.value_range_high_edit.text().strip()
        if not low_text and not high_text:
            return "auto"
        if not low_text or not high_text:
            return f"incomplete (low={low_text or 'auto'}, high={high_text or 'auto'})"
        return f"({low_text}, {high_text})"

    def _percentiles_tooltip_text(self, histogram_card: _HistogramCard) -> str:
        percentiles = tuple(
            text
            for text in (
                histogram_card.percentile_min_edit.text().strip(),
                histogram_card.percentile_max_edit.text().strip(),
            )
            if text
        )
        if not percentiles:
            return "none"
        return ", ".join(percentiles)

    @staticmethod
    def _apply_status_card_spec(label: QLabel, spec: _HistogramStatusCardSpec | None) -> None:
        if spec is None:
            label.setText("")
            label.setToolTip("")
            label.setVisible(False)
            return

        set_status_card(
            label,
            title=spec.title,
            lines=list(spec.lines),
            kind=spec.kind,
            tooltip_message=spec.tooltip_message,
        )

    def _update_card_state(self, card_id: str) -> None:
        self._bind_card_state(card_id)
        self._refresh_card_after_controller_update(card_id)

    def _bind_card_state(self, card_id: str) -> None:
        histogram_card = self._get_card(card_id)
        target, validation_error = self._resolve_card_target(histogram_card)
        settings = None
        if target is not None:
            try:
                settings = self._build_settings(histogram_card)
            except ValueError as error:
                validation_error = str(error)

        # Bind invalid states too, so the controller clears stale calculable
        # requests/results and owns the current warning message for the card.
        self._histogram_controller.bind(
            card_id,
            self.selected_spatialdata,
            target,
            settings,
            validation_error=validation_error,
        )

    def _refresh_card_after_controller_update(self, card_id: str) -> None:
        try:
            histogram_card = self._get_card(card_id)
        except ValueError:
            return

        self._update_card_plot(histogram_card)
        self._refresh_card_overlay_state(histogram_card)
        self._update_card_status(card_id)

    def _update_card_status(self, card_id: str) -> None:
        try:
            histogram_card = self._get_card(card_id)
        except ValueError:
            return

        histogram_card.calculate_button.setEnabled(self._histogram_controller.can_calculate(card_id))
        self._refresh_sync_percentiles_button(histogram_card)
        message = self._histogram_controller.status_message(card_id)
        kind = self._histogram_controller.status_kind(card_id)
        if self._histogram_controller.is_running(card_id):
            spec = build_histogram_running_card_spec(message)
        elif kind == "warning":
            spec = build_histogram_incomplete_card_spec(message)
        elif kind == "error":
            spec = build_histogram_error_card_spec(message)
        elif kind == "success":
            spec = build_histogram_calculated_card_spec(message)
        else:
            spec = build_histogram_ready_card_spec(message)
        extra_lines: list[str] = []
        if histogram_card.viewer_action_message:
            extra_lines.append(histogram_card.viewer_action_message)
        result = self._histogram_controller.result_for_card(card_id)
        percentile_status_line = None if result is None else _format_percentile_status_line(result.percentile_values)
        if kind == "success" and percentile_status_line:
            extra_lines.append(percentile_status_line)
        if kind == "success" and histogram_card.contrast_sync_message:
            extra_lines.append(histogram_card.contrast_sync_message)
        if extra_lines:
            spec = _HistogramStatusCardSpec(
                title=spec.title,
                lines=(*spec.lines, *extra_lines),
                kind=spec.kind,
                tooltip_message=spec.tooltip_message,
            )
        self._apply_status_card_spec(histogram_card.status_label, spec)

    def _on_controller_state_changed(self, card_id: str) -> None:
        self._refresh_card_after_controller_update(card_id)

    def _update_card_plot(self, histogram_card: _HistogramCard) -> None:
        card_id = histogram_card.card_id
        if self._histogram_controller.is_running(card_id):
            # The status card reports progress; keep any existing histogram visible while recalculating.
            return

        result = self._histogram_controller.result_for_card(card_id)
        if result is not None:
            # Avoid re-rendering the same cached result on no-op UI refreshes
            # such as Reset settings; set_histogram(...) also refits pan/zoom.
            if not histogram_card.plot_widget.is_showing_result(result):
                histogram_card.plot_widget.set_histogram(result)
            self._refresh_card_contrast_sync(histogram_card, result)
            return

        self._clear_card_contrast_sync(histogram_card)
        histogram_card.plot_widget.clear_histogram()

    def _on_image_layers_changed(self) -> None:
        for histogram_card in self._cards.values():
            self._refresh_card_overlay_state(histogram_card)
            result = self._histogram_controller.result_for_card(histogram_card.card_id)
            if result is not None:
                self._refresh_card_contrast_sync(histogram_card, result)
            self._update_card_status(histogram_card.card_id)

    def _resolve_card_overlay_match(self, histogram_card: _HistogramCard) -> _HistogramOverlayMatch:
        sdata = self.selected_spatialdata
        target, validation_message = self._resolve_card_target(histogram_card)
        if sdata is None or target is None:
            return _HistogramOverlayMatch(
                state="invalid",
                message=validation_message or "Choose a valid image channel.",
            )

        adapter = self._app_state.viewer_adapter
        live_layer_ids = {
            id(layer)
            for layer in adapter.get_loaded_image_layers(
                sdata,
                target.image_name,
            )
        }
        matches = adapter.layer_bindings.find_bindings(
            sdata=sdata,
            element_type="image",
            element_name=target.image_name,
            coordinate_system=target.coordinate_system,
            image_display_mode="overlay",
            channel_name=target.channel_name,
        )
        image_bindings = [
            binding
            for binding in matches
            if isinstance(binding, ImageLayerBinding) and id(binding.layer) in live_layer_ids
        ]
        if not image_bindings:
            return _HistogramOverlayMatch(state="missing")
        if len(image_bindings) > 1:
            return _HistogramOverlayMatch(
                state="ambiguous",
                message=(
                    f"Multiple matching overlay layers exist for channel "
                    f'"{target.channel_name}". Remove the duplicate layers in napari.'
                ),
            )

        binding = image_bindings[0]
        try:
            _binding_channel_index(binding)
            _binding_channel_name(binding)
        except ValueError as error:
            return _HistogramOverlayMatch(state="invalid", message=str(error))
        return _HistogramOverlayMatch(state="unique", binding=binding)

    def _refresh_card_overlay_state(self, histogram_card: _HistogramCard) -> None:
        """Reconcile the complete Viewer section from current overlay membership."""
        match = self._resolve_card_overlay_match(histogram_card)
        can_load_overlay = (
            self._app_state.viewer is not None and self.selected_spatialdata is not None and match.state == "missing"
        )
        histogram_card.overlay_color_button.setEnabled(can_load_overlay)
        histogram_card.load_overlay_button.setEnabled(can_load_overlay)

        binding = match.binding
        if match.state == "unique" and binding is not None:
            current_row = histogram_card.overlay_row
            if current_row is not None and current_row.binding is binding:
                current_row.refresh_presentation()
            else:
                self._dispose_card_overlay_row(histogram_card)
                row = _OverlayChannelRow(
                    binding,
                    histogram_card.viewer_controls,
                )
                row.remove_requested.connect(partial(self._remove_card_overlay_channel, histogram_card.card_id))
                row.visibility_change_requested.connect(
                    partial(self._change_card_overlay_visibility, histogram_card.card_id)
                )
                row.color_change_requested.connect(partial(self._change_card_overlay_color, histogram_card.card_id))
                histogram_card.overlay_row = row
                histogram_card.viewer_controls_layout.addWidget(row)

            histogram_card.pending_overlay_controls.setVisible(False)
            histogram_card.overlay_state_label.setVisible(False)
            return

        cache_color = match.state in {"missing", "ambiguous"} and self._card_row_matches_current_target(histogram_card)
        self._dispose_card_overlay_row(histogram_card, cache_color=cache_color)
        if match.state == "ambiguous":
            histogram_card.pending_overlay_controls.setVisible(False)
            histogram_card.overlay_state_label.setText(match.message or "Multiple matching overlay layers.")
            histogram_card.overlay_state_label.setVisible(True)
            return

        # Both "missing" and "invalid" use the pending presentation. Its
        # controls are enabled only for "missing".
        histogram_card.overlay_state_label.setVisible(False)
        histogram_card.pending_overlay_controls.setVisible(True)

    def _dispose_card_overlay_row(
        self,
        histogram_card: _HistogramCard,
        *,
        cache_color: bool = False,
    ) -> None:
        row = histogram_card.overlay_row
        if row is None:
            return

        histogram_card.overlay_row = None
        if cache_color:
            histogram_card.overlay_color_button.set_color(row.color_button.current_color)
        row.dispose()
        histogram_card.viewer_controls_layout.removeWidget(row)
        row.deleteLater()

    def _card_row_matches_current_target(self, histogram_card: _HistogramCard) -> bool:
        row = histogram_card.overlay_row
        sdata = self.selected_spatialdata
        target, _message = self._resolve_card_target(histogram_card)
        if row is None or sdata is None or target is None:
            return False

        binding = row.binding
        return (
            binding.sdata_id == id(sdata)
            and binding.element_name == target.image_name
            and binding.coordinate_system == target.coordinate_system
            and binding.image_display_mode == "overlay"
            and binding.channel_name == target.channel_name
        )

    def _resolve_exact_card_overlay_binding(
        self,
        histogram_card: _HistogramCard,
        channel_index: int,
    ) -> ImageLayerBinding | None:
        match = self._resolve_card_overlay_match(histogram_card)
        row = histogram_card.overlay_row
        binding = match.binding
        if (
            match.state == "unique"
            and binding is not None
            and row is not None
            and row.binding is binding
            and row.channel_index == channel_index
        ):
            return binding

        self._refresh_card_overlay_state(histogram_card)
        return None

    def _change_card_overlay_visibility(
        self,
        card_id: str,
        channel_index: int,
        visible: bool,
    ) -> None:
        """Apply row intent; the napari ``visible`` event renders the result."""
        histogram_card = self._cards.get(card_id)
        if histogram_card is None:
            return

        binding = self._resolve_exact_card_overlay_binding(histogram_card, channel_index)
        if binding is None:
            self._update_card_status(card_id)
            return

        layer = binding.layer
        try:
            if layer.visible == visible:
                if histogram_card.overlay_row is not None:
                    histogram_card.overlay_row.refresh_presentation()
                histogram_card.viewer_action_message = None
                self._update_card_status(card_id)
                return
            layer.visible = visible
        except (TypeError, RuntimeError, ValueError) as error:
            if histogram_card.overlay_row is not None:
                histogram_card.overlay_row.refresh_presentation()
            histogram_card.viewer_action_message = f"Overlay visibility could not be updated: {error}"
            self._update_card_status(card_id)
            return

        histogram_card.viewer_action_message = None
        self._update_card_status(card_id)

    def _change_card_overlay_color(
        self,
        card_id: str,
        channel_index: int,
        color: str,
    ) -> None:
        """Apply row intent; the napari ``colormap`` event renders the result."""
        histogram_card = self._cards.get(card_id)
        if histogram_card is None:
            return

        binding = self._resolve_exact_card_overlay_binding(histogram_card, channel_index)
        if binding is None:
            self._update_card_status(card_id)
            return

        normalized_color = _normalized_color_or_none(color)
        if normalized_color is None:
            if histogram_card.overlay_row is not None:
                histogram_card.overlay_row.refresh_presentation()
            histogram_card.viewer_action_message = f'Overlay color could not be updated: "{color}" is invalid.'
            self._update_card_status(card_id)
            return

        layer = binding.layer
        try:
            if _solid_color_from_layer(layer) == normalized_color:
                if histogram_card.overlay_row is not None:
                    histogram_card.overlay_row.refresh_presentation()
                histogram_card.viewer_action_message = None
                self._update_card_status(card_id)
                return
            layer.colormap = normalized_color
        except (TypeError, RuntimeError, ValueError) as error:
            if histogram_card.overlay_row is not None:
                histogram_card.overlay_row.refresh_presentation()
            histogram_card.viewer_action_message = f"Overlay color could not be updated: {error}"
            self._update_card_status(card_id)
            return

        histogram_card.viewer_action_message = None
        self._update_card_status(card_id)

    def _remove_card_overlay_channel(
        self,
        card_id: str,
        channel_index: int,
    ) -> None:
        histogram_card = self._cards.get(card_id)
        sdata = self.selected_spatialdata
        if histogram_card is None or sdata is None:
            return

        binding = self._resolve_exact_card_overlay_binding(histogram_card, channel_index)
        row = histogram_card.overlay_row
        if binding is None or row is None:
            self._update_card_status(card_id)
            return

        histogram_card.overlay_color_button.set_color(row.color_button.current_color)
        try:
            removed_layer = self._app_state.viewer_adapter.remove_image_overlay_channel(
                sdata,
                binding.element_name,
                binding.coordinate_system,
                channel_index=channel_index,
            )
        except (TypeError, RuntimeError, ValueError) as error:
            row.refresh_presentation()
            histogram_card.viewer_action_message = f"Overlay could not be removed: {error}"
            self._update_card_status(card_id)
            return

        if removed_layer is None:
            # The adapter treats an already-absent target as a successful no-op.
            histogram_card.viewer_action_message = "Overlay is no longer loaded in the viewer."
        else:
            # A returned layer is the napari layer removed by this request.
            histogram_card.viewer_action_message = "Overlay removed from viewer."
        self._refresh_card_overlay_state(histogram_card)
        self._update_card_status(card_id)

    def _refresh_card_contrast_sync(self, histogram_card: _HistogramCard, result: HistogramResult) -> None:
        """Bind the card to the resolved napari layer for contrast-limit sync.

        If the card is already connected to the correct layer, refresh the plot only.
        Otherwise disconnect any old layer, connect the card to the resolved layer,
        store the sync state, and draw the current contrast limits.
        """
        card_id = histogram_card.card_id
        binding, unavailable_message = self._resolve_contrast_sync_binding(result)
        if binding is None:
            self._mark_card_contrast_sync_unavailable(
                histogram_card,
                unavailable_message or "Contrast sync unavailable.",
            )
            return

        state = histogram_card.contrast_sync_state
        if state is not None and state.layer is binding.layer:
            histogram_card.contrast_sync_message = None
            self._apply_layer_contrast_limits_to_plot(card_id)
            return

        self._disconnect_card_contrast_sync(histogram_card)
        callback = lambda _event, current_card_id=card_id: self._on_layer_contrast_limits_changed(current_card_id)
        binding.layer.events.contrast_limits.connect(callback)
        histogram_card.contrast_sync_state = _HistogramContrastSyncState(
            layer=binding.layer,
            contrast_limits_callback=callback,
        )
        # Successful sync is already visible via the contrast lines; keep the
        # status card focused on warnings/errors instead of repeating that state.
        histogram_card.contrast_sync_message = None
        self._apply_layer_contrast_limits_to_plot(card_id)

    def _resolve_contrast_sync_binding(self, result: HistogramResult) -> tuple[ImageLayerBinding | None, str | None]:
        sdata = self.selected_spatialdata
        if sdata is None:
            return None, "Contrast sync unavailable: no SpatialData loaded."

        target = result.target
        matches = self._app_state.viewer_adapter.layer_bindings.find_bindings(
            sdata=sdata,
            element_type="image",
            element_name=target.image_name,
            coordinate_system=target.coordinate_system,
            image_display_mode="overlay",
            channel_name=target.channel_name,
        )
        image_bindings = [binding for binding in matches if isinstance(binding, ImageLayerBinding)]
        if not image_bindings:
            return None, "Contrast sync unavailable: open this image in overlay mode."
        if len(image_bindings) > 1:
            return None, "Contrast sync unavailable: multiple matching overlay layers."

        binding = image_bindings[0]
        if bool(getattr(binding.layer, "rgb", False)):
            return None, "Contrast sync unavailable for RGB image layers."
        if _normalized_contrast_limits(getattr(binding.layer, "contrast_limits", None)) is None:
            return None, "Contrast sync unavailable: layer contrast limits are invalid."

        return binding, None

    def _mark_card_contrast_sync_unavailable(self, histogram_card: _HistogramCard, message: str) -> None:
        self._disconnect_card_contrast_sync(histogram_card)
        histogram_card.contrast_sync_message = message

    def _clear_card_contrast_sync(self, histogram_card: _HistogramCard) -> None:
        self._disconnect_card_contrast_sync(histogram_card)
        histogram_card.contrast_sync_message = None

    def _disconnect_card_contrast_sync(self, histogram_card: _HistogramCard) -> None:
        state = histogram_card.contrast_sync_state
        histogram_card.contrast_sync_state = None

        if state is not None:
            try:
                state.layer.events.contrast_limits.disconnect(state.contrast_limits_callback)
            except (TypeError, RuntimeError, ValueError):
                pass

        histogram_card.plot_widget.set_contrast_limits(None)

    def _on_layer_contrast_limits_changed(self, card_id: str) -> None:
        self._apply_layer_contrast_limits_to_plot(card_id)

    def _apply_layer_contrast_limits_to_plot(self, card_id: str) -> None:
        histogram_card = self._cards.get(card_id)
        state = None if histogram_card is None else histogram_card.contrast_sync_state
        if state is None or histogram_card is None:
            return

        limits = _normalized_contrast_limits(getattr(state.layer, "contrast_limits", None))
        if limits is None:
            histogram_card.contrast_sync_message = "Contrast sync unavailable: layer contrast limits are invalid."
            histogram_card.plot_widget.set_contrast_limits(None)
            self._update_card_status(card_id)
            return

        state.updating_plot = True
        try:
            histogram_card.plot_widget.set_contrast_limits(limits)
        finally:
            state.updating_plot = False

    def _on_plot_contrast_limits_dragged(self, card_id: str, low: float, high: float) -> None:
        histogram_card = self._cards.get(card_id)
        state = None if histogram_card is None else histogram_card.contrast_sync_state
        if state is None or state.updating_plot:
            return

        limits = _normalized_contrast_limits((low, high))
        if limits is None:
            return

        state.layer.contrast_limits = limits
        # Setting layer.contrast_limits normally emits the layer's contrast_limits event.
        # During sync setup, that event is connected to _on_layer_contrast_limits_changed,
        # which already calls _apply_layer_contrast_limits_to_plot(card_id). Keep this
        # direct refresh as a local reconciliation step in case the layer adjusts the
        # accepted limits.
        self._apply_layer_contrast_limits_to_plot(card_id)

    def _refresh_sync_percentiles_button(self, histogram_card: _HistogramCard) -> None:
        limits, reason = self._resolve_percentile_sync_limits(histogram_card)
        can_sync = limits is not None
        histogram_card.sync_percentiles_button.setEnabled(can_sync)
        if can_sync:
            histogram_card.sync_percentiles_button.setToolTip(
                format_tooltip("Synchronize contrast limits to calculated percentile values.")
            )
            return

        histogram_card.sync_percentiles_button.setToolTip(format_tooltip(reason or "Percentiles cannot be synced."))

    def _sync_percentiles_to_contrast_limits(self, card_id: str) -> None:
        histogram_card = self._get_card(card_id)
        limits, _reason = self._resolve_percentile_sync_limits(histogram_card)
        state = histogram_card.contrast_sync_state
        if limits is None or state is None:
            self._refresh_sync_percentiles_button(histogram_card)
            return

        state.layer.contrast_limits = limits
        # Setting layer.contrast_limits should emit the layer event that drives
        # _apply_layer_contrast_limits_to_plot(card_id). Keep the direct refresh
        # as local reconciliation in case napari clamps or normalizes the
        # accepted layer contrast limits.
        self._apply_layer_contrast_limits_to_plot(card_id)
        self._update_card_status(card_id)

    def _resolve_percentile_sync_limits(
        self,
        histogram_card: _HistogramCard,
    ) -> tuple[tuple[float, float] | None, str | None]:
        percentile_min_text = histogram_card.percentile_min_edit.text().strip()
        percentile_max_text = histogram_card.percentile_max_edit.text().strip()
        if not percentile_min_text or not percentile_max_text:
            return None, "Enter both Percentile min and Percentile max, then show the histogram."

        try:
            percentile_min = self._parse_float(percentile_min_text, field_name="percentile min")
            percentile_max = self._parse_float(percentile_max_text, field_name="percentile max")
        except ValueError as error:
            return None, str(error)

        if percentile_min >= percentile_max:
            return None, "Percentile min must be lower than Percentile max."

        result = self._histogram_controller.result_for_card(histogram_card.card_id)
        if result is None:
            return None, "Show the histogram after entering percentile values."

        if percentile_min not in result.percentile_values or percentile_max not in result.percentile_values:
            return None, "Show the histogram again to calculate both percentile values."

        low = float(result.percentile_values[percentile_min])
        high = float(result.percentile_values[percentile_max])
        if not math.isfinite(low) or not math.isfinite(high) or low >= high:
            return None, "Calculated percentile contrast limits are invalid."

        if histogram_card.contrast_sync_state is None:
            return None, histogram_card.contrast_sync_message or "Open this image in overlay mode before syncing."

        return (low, high), None

    def _resolve_card_target(self, histogram_card: _HistogramCard) -> tuple[HistogramTarget | None, str | None]:
        if self.selected_spatialdata is None:
            return None, "No SpatialData loaded."

        coordinate_system = self._current_text_data(histogram_card.coordinate_system_combo)
        if coordinate_system is None:
            return None, "Choose a coordinate system."

        image_name = self._current_text_data(histogram_card.image_combo)
        if image_name is None:
            return None, "Choose an image."

        scale_error = self._card_scale_errors.get(histogram_card.card_id)
        if scale_error:
            return None, scale_error

        channel_error = self._card_channel_errors.get(histogram_card.card_id)
        if channel_error:
            return None, channel_error

        channel_name = histogram_card.accepted_channel_name
        if channel_name is None:
            return None, "Choose a channel."

        target = HistogramTarget(
            coordinate_system=coordinate_system,
            image_name=image_name,
            channel_name=channel_name,
        )
        return target, None

    def _load_card_overlay(self, card_id: str) -> None:
        histogram_card = self._get_card(card_id)
        target, validation_error = self._resolve_card_target(histogram_card)
        if target is None or self.selected_spatialdata is None:
            histogram_card.viewer_action_message = (
                validation_error or "Choose an image and channel before loading an overlay."
            )
            self._update_card_status(card_id)
            return

        try:
            result = self._app_state.viewer_adapter.ensure_image_overlay_channel_loaded(
                self.selected_spatialdata,
                target.image_name,
                target.coordinate_system,
                channel=target.channel_name,
                channel_color=histogram_card.overlay_color_button.current_color,
            )
        except Exception as error:  # noqa: BLE001 - surface adapter validation/load errors in the card status.
            histogram_card.viewer_action_message = f"Overlay could not be loaded: {error}"
            self._update_card_status(card_id)
            return

        self._app_state.viewer_adapter.activate_layer(result.primary_layer)
        action = "loaded" if result.created else "updated"
        histogram_card.viewer_action_message = f"Overlay {action} in viewer."
        self._refresh_card_overlay_state(histogram_card)

        calculated_result = self._histogram_controller.result_for_card(card_id)
        if calculated_result is not None:
            self._refresh_card_contrast_sync(histogram_card, calculated_result)
        self._update_card_status(card_id)

    def _calculate_histogram(self, card_id: str) -> None:
        self._bind_card_state(card_id)
        cached_result = self._histogram_controller.result_for_card(card_id)
        if cached_result is not None and not self._histogram_controller.is_running(card_id):
            histogram_card = self._get_card(card_id)
            histogram_card.plot_widget.reset_view(cached_result)
            self._refresh_card_contrast_sync(histogram_card, cached_result)
            self._update_card_status(card_id)
            return

        self._histogram_controller.calculate(card_id)
        self._refresh_card_after_controller_update(card_id)

    def _build_settings(self, histogram_card: _HistogramCard) -> HistogramSettings:
        value_range = self._parse_optional_pair(
            histogram_card.value_range_low_edit.text(),
            histogram_card.value_range_high_edit.text(),
            field_name="value range",
        )
        percentiles = tuple(
            self._parse_float(text, field_name="percentile")
            for text in (
                histogram_card.percentile_min_edit.text().strip(),
                histogram_card.percentile_max_edit.text().strip(),
            )
            if text
        )
        return HistogramSettings(
            bins=histogram_card.bins_spin.value(),
            value_range=value_range,
            density=histogram_card.density_checkbox.isChecked(),
            exclude_nan=histogram_card.exclude_nan_checkbox.isChecked(),
            exclude_zeros=histogram_card.exclude_zeros_checkbox.isChecked(),
            log_y=histogram_card.log_y_checkbox.isChecked(),
            scale=self._current_text_data(histogram_card.scale_combo) or _DEFAULT_SCALE,
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

    def _update_empty_state(self) -> None:
        has_cards = bool(self._cards)
        self.empty_state_label.setVisible(not has_cards)
        self.cards_container.setVisible(has_cards)


def _normalized_contrast_limits(limits: object) -> tuple[float, float] | None:
    if not isinstance(limits, Iterable) or isinstance(limits, (str, bytes)):
        return None

    values = tuple(limits)
    if len(values) != 2:
        return None
    low, high = values

    try:
        low = float(low)
        high = float(high)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(low) or not math.isfinite(high):
        return None

    low, high = sorted((low, high))
    if low == high:
        return None

    return low, high


def _format_percentile_status_line(percentile_values: Mapping[float, float]) -> str | None:
    if not percentile_values:
        return None

    values = ", ".join(
        f"p{float(percentile):g} = {_format_compact_number(float(value))}"
        for percentile, value in sorted(percentile_values.items())
    )
    return f"Percentiles: {values}"


def _format_compact_number(value: float) -> str:
    if not math.isfinite(value):
        return str(value)
    return f"{value:.4g}"
