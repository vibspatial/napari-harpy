from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from qtpy.QtCore import QSignalBlocker, Qt, Signal
from qtpy.QtGui import QPixmap
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLayout,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from spatialdata import read_zarr
from spatialdata.models import get_axes_names
from spatialdata.transformations import get_transformation
from xarray import DataArray

from napari_harpy._app_state import HarpyAppState, get_or_create_app_state
from napari_harpy._spatialdata import get_annotating_table_names
from napari_harpy._viewer_adapter import DEFAULT_OVERLAY_COLORS
from napari_harpy.widgets._shared_styles import (
    ACTION_BUTTON_STYLESHEET as _ACTION_BUTTON_STYLESHEET,
)
from napari_harpy.widgets._shared_styles import (
    CHECKBOX_STYLESHEET as _CHECKBOX_STYLESHEET,
)
from napari_harpy.widgets._shared_styles import (
    WIDGET_MIN_WIDTH as _WIDGET_MIN_WIDTH,
)
from napari_harpy.widgets._shared_styles import (
    apply_scroll_content_surface,
    apply_widget_surface,
    build_input_control_stylesheet,
    create_form_label,
    format_tooltip,
)

if TYPE_CHECKING:
    import napari
    from spatialdata import SpatialData


_INPUT_CONTROL_STYLESHEET = build_input_control_stylesheet("QComboBox")
_CARD_STYLESHEET = "QFrame {background-color: #f8eeea; border: 1px solid #eadfd8; border-radius: 10px;}"
_CARD_TITLE_STYLESHEET = (
    "QLabel {"
    "background-color: #EFDCCF; "
    "border: 1px solid #D3B19E; "
    "border-radius: 8px; "
    "color: #374151; "
    "font-weight: 700; "
    "padding: 6px 10px;}"
)
_SECTION_TITLE_STYLESHEET = "color: #374151; font-size: 14px; font-weight: 700;"
_SECTION_GROUP_STYLESHEET = (
    "QFrame#viewer_widget_images_group, QFrame#viewer_widget_labels_group {"
    "background-color: #f7ece7; "
    "border: 1px solid #e3d2c8; "
    "border-radius: 12px;}"
)
_SUMMARY_LABEL_STYLESHEET = "color: #374151; font-weight: 500;"
_EMPTY_STATE_STYLESHEET = "color: #6b7280; font-weight: 500;"
_CHANNEL_PANEL_STYLESHEET = "QWidget { background: transparent; }"


@dataclass(frozen=True)
class ImageLoadRequest:
    image_name: str
    mode: str
    channels: list[int]
    channel_colors: list[str]


class _ElidedLabel(QLabel):
    """Single-line label that shows a tooltip only when the text is elided."""

    def __init__(self, text: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._full_text = text
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.setMinimumWidth(0)
        self.setMinimumHeight(36)
        self.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self._update_elided_text()

    def set_full_text(self, text: str) -> None:
        self._full_text = text
        self._update_elided_text()

    def resizeEvent(self, event: object) -> None:
        super().resizeEvent(event)
        self._update_elided_text()

    def _update_elided_text(self) -> None:
        available_width = max(0, self.contentsRect().width())
        elided_text = self.fontMetrics().elidedText(self._full_text, Qt.TextElideMode.ElideRight, available_width)
        super().setText(elided_text)
        self.setToolTip(format_tooltip(self._full_text) if elided_text != self._full_text else "")


class _LabelsCardWidget(QFrame):
    """Card UI for one labels element in the selected coordinate system."""

    add_update_requested = Signal(str)

    def __init__(
        self,
        *,
        label_name: str,
        table_names: list[str],
    ) -> None:
        super().__init__()
        self.label_name = label_name
        self.setObjectName(f"viewer_widget_labels_card_{label_name}")
        self.setStyleSheet(_CARD_STYLESHEET)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        self.title_label = _ElidedLabel(label_name)
        self.title_label.setObjectName(f"viewer_widget_labels_card_title_{label_name}")
        self.title_label.setStyleSheet(_CARD_TITLE_STYLESHEET)

        form_layout = QFormLayout()
        form_layout.setContentsMargins(0, 0, 0, 0)
        form_layout.setHorizontalSpacing(8)
        form_layout.setVerticalSpacing(6)

        linked_table_label = _create_form_label("Linked table")
        self.linked_table_combo = QComboBox()
        self.linked_table_combo.setObjectName(f"viewer_widget_linked_table_combo_{label_name}")
        self.linked_table_combo.setStyleSheet(_INPUT_CONTROL_STYLESHEET)
        if table_names:
            self.linked_table_combo.addItems(table_names)
        else:
            self.linked_table_combo.addItem("No linked tables")
            self.linked_table_combo.setEnabled(False)

        self.add_update_button = QPushButton("Add / Update in viewer")
        self.add_update_button.setObjectName(f"viewer_widget_add_update_labels_button_{label_name}")
        self.add_update_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.add_update_button.setMinimumHeight(28)
        self.add_update_button.setStyleSheet(_ACTION_BUTTON_STYLESHEET)
        self.add_update_button.clicked.connect(self._emit_add_update_request)

        form_layout.addRow(linked_table_label, self.linked_table_combo)

        layout.addWidget(self.title_label)
        layout.addLayout(form_layout)
        layout.addWidget(self.add_update_button)

    def _emit_add_update_request(self, _checked: bool = False) -> None:
        self.add_update_requested.emit(self.label_name)


class _ImageCardWidget(QFrame):
    """Card UI for one image element in the selected coordinate system."""

    add_update_requested = Signal(object)

    def __init__(
        self,
        *,
        image_name: str,
        channel_names: list[str],
    ) -> None:
        super().__init__()
        self.image_name = image_name
        self.channel_names = channel_names
        self.setObjectName(f"viewer_widget_image_card_{image_name}")
        self.setStyleSheet(_CARD_STYLESHEET)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        self.title_label = _ElidedLabel(image_name)
        self.title_label.setObjectName(f"viewer_widget_image_card_title_{image_name}")
        self.title_label.setStyleSheet(_CARD_TITLE_STYLESHEET)

        mode_layout = QHBoxLayout()
        mode_layout.setContentsMargins(0, 0, 0, 0)
        mode_layout.setSpacing(16)

        self.stack_toggle = QCheckBox("stack")
        self.stack_toggle.setObjectName(f"viewer_widget_stack_toggle_{image_name}")
        self.stack_toggle.setStyleSheet(_CHECKBOX_STYLESHEET)
        self.stack_toggle.setChecked(True)

        self.overlay_toggle = QCheckBox("overlay")
        self.overlay_toggle.setObjectName(f"viewer_widget_overlay_toggle_{image_name}")
        self.overlay_toggle.setStyleSheet(_CHECKBOX_STYLESHEET)

        mode_layout.addWidget(self.stack_toggle)
        mode_layout.addWidget(self.overlay_toggle)
        mode_layout.addStretch(1)

        self.channel_panel = QWidget()
        self.channel_panel.setObjectName(f"viewer_widget_channel_panel_{image_name}")
        self.channel_panel.setStyleSheet(_CHANNEL_PANEL_STYLESHEET)
        self.channel_panel.setVisible(False)
        channel_layout = QVBoxLayout(self.channel_panel)
        channel_layout.setContentsMargins(0, 0, 0, 0)
        channel_layout.setSpacing(6)

        self.channel_checkboxes: list[QCheckBox] = []
        self.channel_color_combos: list[QComboBox] = []

        if channel_names:
            for index, channel_name in enumerate(channel_names):
                row = QWidget()
                row_layout = QHBoxLayout(row)
                row_layout.setContentsMargins(0, 0, 0, 0)
                row_layout.setSpacing(8)

                checkbox = QCheckBox(channel_name)
                checkbox.setObjectName(f"viewer_widget_channel_checkbox_{image_name}_{channel_name}")
                checkbox.setStyleSheet(_CHECKBOX_STYLESHEET)

                color_combo = QComboBox()
                color_combo.setObjectName(f"viewer_widget_channel_color_combo_{image_name}_{channel_name}")
                color_combo.setStyleSheet(_INPUT_CONTROL_STYLESHEET)
                for color in DEFAULT_OVERLAY_COLORS:
                    color_combo.addItem(color, color)
                color_combo.setCurrentIndex(index % color_combo.count())

                row_layout.addWidget(checkbox, 1)
                row_layout.addWidget(color_combo)

                channel_layout.addWidget(row)
                self.channel_checkboxes.append(checkbox)
                self.channel_color_combos.append(color_combo)
        else:
            no_channels_label = QLabel("No channel axis available for this image.")
            no_channels_label.setObjectName(f"viewer_widget_no_channels_label_{image_name}")
            no_channels_label.setWordWrap(True)
            no_channels_label.setStyleSheet(_EMPTY_STATE_STYLESHEET)
            channel_layout.addWidget(no_channels_label)

        self.add_update_button = QPushButton("Add / Update in viewer")
        self.add_update_button.setObjectName(f"viewer_widget_add_update_image_button_{image_name}")
        self.add_update_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.add_update_button.setMinimumHeight(28)
        self.add_update_button.setStyleSheet(_ACTION_BUTTON_STYLESHEET)
        self.add_update_button.clicked.connect(self._emit_add_update_request)
        self.add_update_button.setToolTip("")

        self.stack_toggle.toggled.connect(self._on_stack_toggled)
        self.overlay_toggle.toggled.connect(self._on_overlay_toggled)

        layout.addWidget(self.title_label)
        layout.addLayout(mode_layout)
        layout.addWidget(self.channel_panel)
        layout.addWidget(self.add_update_button)

    def _on_stack_toggled(self, checked: bool) -> None:
        if checked:
            with QSignalBlocker(self.overlay_toggle):
                self.overlay_toggle.setChecked(False)
            self.channel_panel.setVisible(False)
            return

        if not self.overlay_toggle.isChecked():
            with QSignalBlocker(self.stack_toggle):
                self.stack_toggle.setChecked(True)

    def _on_overlay_toggled(self, checked: bool) -> None:
        if checked:
            with QSignalBlocker(self.stack_toggle):
                self.stack_toggle.setChecked(False)
            self.channel_panel.setVisible(True)
            return

        self.channel_panel.setVisible(False)
        if not self.stack_toggle.isChecked():
            with QSignalBlocker(self.stack_toggle):
                self.stack_toggle.setChecked(True)

    def display_mode(self) -> str:
        return "overlay" if self.overlay_toggle.isChecked() else "stack"

    def get_selected_overlay_channels(self) -> list[int]:
        return [index for index, checkbox in enumerate(self.channel_checkboxes) if checkbox.isChecked()]

    def get_selected_overlay_channel_names(self) -> list[str]:
        return [checkbox.text() for checkbox in self.channel_checkboxes if checkbox.isChecked()]

    def get_selected_overlay_colors(self) -> list[str]:
        return [
            str(color_combo.currentData() or color_combo.currentText())
            for checkbox, color_combo in zip(self.channel_checkboxes, self.channel_color_combos, strict=False)
            if checkbox.isChecked()
        ]

    def _emit_add_update_request(self, _checked: bool = False) -> None:
        self.add_update_requested.emit(
            ImageLoadRequest(
                image_name=self.image_name,
                mode=self.display_mode(),
                channels=self.get_selected_overlay_channels(),
                channel_colors=self.get_selected_overlay_colors(),
            )
        )


class ViewerWidget(QWidget):
    """Shared viewer widget backed by `HarpyAppState` and `ViewerAdapter`."""

    def __init__(self, napari_viewer: napari.Viewer | None = None) -> None:
        super().__init__()
        self.setObjectName("viewer_widget")
        apply_widget_surface(self)
        self.setMinimumWidth(_WIDGET_MIN_WIDTH)
        self._viewer = napari_viewer
        self._app_state = get_or_create_app_state(napari_viewer)
        self._labels_cards: list[_LabelsCardWidget] = []
        self._image_cards: list[_ImageCardWidget] = []
        self._logo_path = Path(__file__).resolve().parents[3] / "docs" / "_static" / "logo.png"

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        self.scroll_area = QScrollArea()
        self.scroll_area.setObjectName("viewer_widget_scroll_area")
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll_area.setStyleSheet("QScrollArea { border: 0px; background: transparent; }")

        self.scroll_content = QWidget()
        self.scroll_content.setObjectName("viewer_widget_scroll_content")
        apply_scroll_content_surface(self.scroll_content)
        self.content_layout = QVBoxLayout(self.scroll_content)
        self.content_layout.setContentsMargins(12, 12, 12, 12)
        self.content_layout.setSpacing(10)

        title = QLabel("Viewer")
        title.setObjectName("viewer_widget_title")
        title.setStyleSheet("color: #111827; font-size: 18px; font-weight: 700;")
        title.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        header_logo = self._create_header_logo()

        self.open_sdata_button = QPushButton("Load SpatialData")
        self.open_sdata_button.setObjectName("viewer_widget_open_sdata_button")
        self.open_sdata_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.open_sdata_button.setMinimumHeight(28)
        self.open_sdata_button.setStyleSheet(_ACTION_BUTTON_STYLESHEET)
        self.open_sdata_button.clicked.connect(self._open_spatialdata)

        self.empty_state_label = QLabel(
            "No SpatialData loaded. Use `Interactive(sdata)` for now; an in-widget open action will follow later."
        )
        self.empty_state_label.setObjectName("viewer_widget_empty_state")
        self.empty_state_label.setWordWrap(True)
        self.empty_state_label.setStyleSheet(_EMPTY_STATE_STYLESHEET)

        self.summary_label = QLabel("No SpatialData loaded.")
        self.summary_label.setObjectName("viewer_widget_summary")
        self.summary_label.setWordWrap(True)
        self.summary_label.setStyleSheet(_SUMMARY_LABEL_STYLESHEET)

        selector_layout = QFormLayout()
        selector_layout.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        selector_layout.setHorizontalSpacing(12)
        selector_layout.setVerticalSpacing(10)

        self.coordinate_system_combo = QComboBox()
        self.coordinate_system_combo.setObjectName("viewer_widget_coordinate_system_combo")
        self.coordinate_system_combo.currentIndexChanged.connect(self._on_coordinate_system_changed)
        self.coordinate_system_combo.setStyleSheet(_INPUT_CONTROL_STYLESHEET)
        selector_layout.addRow(_create_form_label("Coordinate system"), self.coordinate_system_combo)

        self.action_feedback_label = QLabel("")
        self.action_feedback_label.setObjectName("viewer_widget_action_feedback")
        self.action_feedback_label.setWordWrap(True)
        self.action_feedback_label.hide()

        self.images_section_title = QLabel("Images")
        self.images_section_title.setObjectName("viewer_widget_images_section_title")
        self.images_section_title.setStyleSheet(_SECTION_TITLE_STYLESHEET)

        self.images_empty_label = QLabel("No images available in the selected coordinate system.")
        self.images_empty_label.setObjectName("viewer_widget_images_empty_state")
        self.images_empty_label.setWordWrap(True)
        self.images_empty_label.setStyleSheet(_EMPTY_STATE_STYLESHEET)

        self.images_section = QWidget()
        self.images_section.setObjectName("viewer_widget_images_section")
        self.images_section_layout = QVBoxLayout(self.images_section)
        self.images_section_layout.setContentsMargins(0, 0, 0, 0)
        self.images_section_layout.setSpacing(8)
        self.images_group = QFrame()
        self.images_group.setObjectName("viewer_widget_images_group")
        self.images_group.setStyleSheet(_SECTION_GROUP_STYLESHEET)
        self.images_group_layout = QVBoxLayout(self.images_group)
        self.images_group_layout.setContentsMargins(12, 12, 12, 12)
        self.images_group_layout.setSpacing(10)
        self.images_group_layout.addWidget(self.images_section_title)
        self.images_group_layout.addWidget(self.images_empty_label)
        self.images_group_layout.addWidget(self.images_section)

        self.labels_section_title = QLabel("Segmentations")
        self.labels_section_title.setObjectName("viewer_widget_labels_section_title")
        self.labels_section_title.setStyleSheet(_SECTION_TITLE_STYLESHEET)

        self.labels_empty_label = QLabel("No segmentation masks available in the selected coordinate system.")
        self.labels_empty_label.setObjectName("viewer_widget_labels_empty_state")
        self.labels_empty_label.setWordWrap(True)
        self.labels_empty_label.setStyleSheet(_EMPTY_STATE_STYLESHEET)

        self.labels_section = QWidget()
        self.labels_section.setObjectName("viewer_widget_labels_section")
        self.labels_section_layout = QVBoxLayout(self.labels_section)
        self.labels_section_layout.setContentsMargins(0, 0, 0, 0)
        self.labels_section_layout.setSpacing(8)
        self.labels_group = QFrame()
        self.labels_group.setObjectName("viewer_widget_labels_group")
        self.labels_group.setStyleSheet(_SECTION_GROUP_STYLESHEET)
        self.labels_group_layout = QVBoxLayout(self.labels_group)
        self.labels_group_layout.setContentsMargins(12, 12, 12, 12)
        self.labels_group_layout.setSpacing(10)
        self.labels_group_layout.addWidget(self.labels_section_title)
        self.labels_group_layout.addWidget(self.labels_empty_label)
        self.labels_group_layout.addWidget(self.labels_section)

        self.content_layout.addWidget(header_logo)
        self.content_layout.addWidget(title)
        self.content_layout.addWidget(self.open_sdata_button)
        self.content_layout.addWidget(self.empty_state_label)
        self.content_layout.addWidget(self.summary_label)
        self.content_layout.addLayout(selector_layout)
        self.content_layout.addWidget(self.action_feedback_label)
        self.content_layout.addWidget(self.images_group)
        self.content_layout.addWidget(self.labels_group)
        self.content_layout.addStretch(1)

        self.scroll_area.setWidget(self.scroll_content)
        root_layout.addWidget(self.scroll_area)

        self._app_state.sdata_changed.connect(self._on_sdata_changed)
        self.refresh_from_sdata(self._app_state.sdata)

    @property
    def app_state(self) -> HarpyAppState:
        """Return the shared Harpy app state for this widget."""
        return self._app_state

    @property
    def labels_cards(self) -> list[_LabelsCardWidget]:
        """Return the currently visible labels cards."""
        return list(self._labels_cards)

    @property
    def image_cards(self) -> list[_ImageCardWidget]:
        """Return the currently visible image cards."""
        return list(self._image_cards)

    def _on_sdata_changed(self, sdata: SpatialData | None) -> None:
        """Refresh the widget when the shared loaded `SpatialData` changes."""
        self.refresh_from_sdata(sdata)

    def _on_coordinate_system_changed(self) -> None:
        """Refresh the filtered image and labels cards when the coordinate system changes."""
        self._refresh_coordinate_system_content()

    def _open_spatialdata(self, _checked: bool = False) -> None:
        selected_path = QFileDialog.getExistingDirectory(
            self,
            "Load SpatialData",
            "",
            QFileDialog.Option.ShowDirsOnly | QFileDialog.Option.DontResolveSymlinks,
        )
        if not selected_path:
            return

        try:
            sdata = read_zarr(selected_path)
        except (OSError, ValueError) as error:
            self._set_action_feedback(f"Could not load SpatialData store: {error}", is_error=True)
            return

        self._app_state.set_sdata(sdata)
        self._clear_action_feedback()

    def refresh_from_sdata(self, sdata: SpatialData | None) -> None:
        """Refresh the viewer widget from the currently loaded `SpatialData`."""
        with QSignalBlocker(self.coordinate_system_combo):
            previous_coordinate_system = self.coordinate_system_combo.currentText()
            self.coordinate_system_combo.clear()

            if sdata is None:
                self.empty_state_label.show()
                self.summary_label.setText("No SpatialData loaded.")
                self.coordinate_system_combo.setEnabled(False)
                self._clear_cards()
                self._update_section_empty_states([], [])
                return

            coordinate_systems = _get_coordinate_systems_from_sdata(sdata)
            self.coordinate_system_combo.addItems(coordinate_systems)

            if previous_coordinate_system in coordinate_systems:
                self.coordinate_system_combo.setCurrentIndex(coordinate_systems.index(previous_coordinate_system))
            elif coordinate_systems:
                self.coordinate_system_combo.setCurrentIndex(0)

            self.empty_state_label.hide()
            self.coordinate_system_combo.setEnabled(bool(coordinate_systems))

        self._refresh_coordinate_system_content()

    def _refresh_coordinate_system_content(self) -> None:
        sdata = self._app_state.sdata
        coordinate_system = self.coordinate_system_combo.currentText()

        if sdata is None or not coordinate_system:
            self._clear_cards()
            self._update_section_empty_states([], [])
            if sdata is None:
                self.summary_label.setText("No SpatialData loaded.")
            else:
                self.summary_label.setText("No coordinate system selected.")
            return

        label_names = _get_labels_in_coordinate_system(sdata, coordinate_system)
        image_names = _get_images_in_coordinate_system(sdata, coordinate_system)

        self.summary_label.setText(
            f"In coordinate system `{coordinate_system}`: "
            f"{len(image_names)} image element(s) and {len(label_names)} segmentation mask(s)."
        )
        self._rebuild_image_cards(sdata, image_names)
        self._rebuild_labels_cards(sdata, label_names)
        self._update_section_empty_states(image_names, label_names)

    def _rebuild_image_cards(self, sdata: SpatialData, image_names: list[str]) -> None:
        _clear_layout(self.images_section_layout)
        self._image_cards = []

        for image_name in image_names:
            card = _ImageCardWidget(
                image_name=image_name,
                channel_names=_get_image_channel_names(sdata, image_name),
            )
            card.add_update_requested.connect(self._add_or_update_image_layer)
            self.images_section_layout.addWidget(card)
            self._image_cards.append(card)

    def _rebuild_labels_cards(self, sdata: SpatialData, label_names: list[str]) -> None:
        _clear_layout(self.labels_section_layout)
        self._labels_cards = []

        for label_name in label_names:
            card = _LabelsCardWidget(
                label_name=label_name,
                table_names=get_annotating_table_names(sdata, label_name),
            )
            card.add_update_requested.connect(self._add_or_update_labels_layer)
            self.labels_section_layout.addWidget(card)
            self._labels_cards.append(card)

    def _add_or_update_labels_layer(self, label_name: str) -> None:
        sdata = self._app_state.sdata
        coordinate_system = self.coordinate_system_combo.currentText()

        if sdata is None or not coordinate_system:
            self._set_action_feedback("Load a SpatialData object and select a coordinate system first.", is_error=True)
            return

        try:
            layer = self._app_state.viewer_adapter.ensure_labels_loaded(sdata, label_name, coordinate_system)
        except ValueError as error:
            self._set_action_feedback(str(error), is_error=True)
            return

        self._app_state.viewer_adapter.activate_layer(layer)
        self._set_action_feedback(
            f"Loaded segmentation `{label_name}` in coordinate system `{coordinate_system}`.",
            is_error=False,
        )

    def _add_or_update_image_layer(self, request: ImageLoadRequest) -> None:
        sdata = self._app_state.sdata
        coordinate_system = self.coordinate_system_combo.currentText()
        image_name = request.image_name

        if sdata is None or not coordinate_system:
            self._set_action_feedback("Load a SpatialData object and select a coordinate system first.", is_error=True)
            return

        mode = request.mode
        try:
            layer_or_layers = self._app_state.viewer_adapter.ensure_image_loaded(
                sdata,
                image_name,
                coordinate_system,
                mode=mode,
                channels=request.channels if mode == "overlay" else None,
                channel_colors=request.channel_colors if mode == "overlay" else None,
            )
        except ValueError as error:
            self._set_action_feedback(str(error), is_error=True)
            return

        if mode == "stack":
            if isinstance(layer_or_layers, list):
                self._set_action_feedback(
                    f"Expected one stack image layer for `{image_name}`, but received multiple layers.",
                    is_error=True,
                )
                return

            self._app_state.viewer_adapter.activate_layer(layer_or_layers)
            self._set_action_feedback(
                f"Loaded image `{image_name}` in stack mode for coordinate system `{coordinate_system}`.",
                is_error=False,
            )
            return

        if not isinstance(layer_or_layers, list):
            self._set_action_feedback(
                f"Expected overlay image layers for `{image_name}`, but received a single layer.",
                is_error=True,
            )
            return
        if not layer_or_layers:
            self._set_action_feedback(f"No overlay layers were returned for image `{image_name}`.", is_error=True)
            return

        self._app_state.viewer_adapter.activate_layer(layer_or_layers[0])
        self._set_action_feedback(
            f"Loaded image `{image_name}` in overlay mode for channels {request.channels} "
            f"in coordinate system `{coordinate_system}`.",
            is_error=False,
        )

    def _update_section_empty_states(self, image_names: list[str], label_names: list[str]) -> None:
        self.images_empty_label.setVisible(not image_names)
        self.labels_empty_label.setVisible(not label_names)
        self.images_section.setVisible(bool(image_names))
        self.labels_section.setVisible(bool(label_names))

    def _clear_cards(self) -> None:
        _clear_layout(self.images_section_layout)
        _clear_layout(self.labels_section_layout)
        self._image_cards = []
        self._labels_cards = []

    def _set_action_feedback(self, message: str, *, is_error: bool) -> None:
        self.action_feedback_label.setText(message)
        self.action_feedback_label.setStyleSheet(
            "color: #b91c1c; font-weight: 600;" if is_error else "color: #166534; font-weight: 600;"
        )
        self.action_feedback_label.show()

    def _clear_action_feedback(self) -> None:
        self.action_feedback_label.clear()
        self.action_feedback_label.hide()

    def _create_header_logo(self) -> QLabel:
        logo_label = QLabel()
        logo_label.setObjectName("viewer_widget_header_logo")
        logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        logo_pixmap = QPixmap(str(self._logo_path))
        if not logo_pixmap.isNull():
            logo_label.setPixmap(logo_pixmap.scaledToWidth(120, Qt.TransformationMode.SmoothTransformation))
            return logo_label

        logo_label.setText("napari-harpy")
        logo_label.setStyleSheet("color: #111827; font-size: 18px; font-weight: 600;")
        return logo_label


def _clear_layout(layout: QLayout) -> None:
    while layout.count():
        item = layout.takeAt(0)
        widget = item.widget()
        child_layout = item.layout()
        if widget is not None:
            widget.deleteLater()
        elif child_layout is not None:
            _clear_layout(child_layout)


def _create_form_label(text: str) -> QLabel:
    return create_form_label(text)


def _get_coordinate_systems_from_sdata(sdata: SpatialData) -> list[str]:
    coordinate_systems: set[str] = set()

    for collection_name in ("labels", "images"):
        collection = getattr(sdata, collection_name, {})
        for element in collection.values():
            coordinate_systems.update(get_transformation(element, get_all=True).keys())

    return sorted(coordinate_systems)


def _get_labels_in_coordinate_system(sdata: SpatialData, coordinate_system: str) -> list[str]:
    labels = getattr(sdata, "labels", {})
    return sorted(
        label_name
        for label_name, element in labels.items()
        if coordinate_system in get_transformation(element, get_all=True).keys()
    )


def _get_images_in_coordinate_system(sdata: SpatialData, coordinate_system: str) -> list[str]:
    images = getattr(sdata, "images", {})
    return sorted(
        image_name
        for image_name, element in images.items()
        if coordinate_system in get_transformation(element, get_all=True).keys()
    )


def _get_image_channel_names(sdata: SpatialData, image_name: str) -> list[str]:
    images = getattr(sdata, "images", {})
    image_element = images[image_name]
    axes = get_axes_names(image_element)
    if "c" not in axes:
        return []

    if isinstance(image_element, DataArray):
        channel_values = list(image_element.coords.indexes["c"])
    else:
        scale0 = next(iter(image_element["scale0"].values()))
        channel_values = list(scale0.coords.indexes["c"])

    return [str(channel_value) for channel_value in channel_values]
