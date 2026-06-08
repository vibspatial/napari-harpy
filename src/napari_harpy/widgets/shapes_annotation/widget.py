from __future__ import annotations

from typing import TYPE_CHECKING

from qtpy.QtCore import QSignalBlocker, Qt
from qtpy.QtGui import QPixmap
from qtpy.QtWidgets import (
    QDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from napari_harpy._app_state import CoordinateSystemChangedEvent, HarpyAppState, get_or_create_app_state
from napari_harpy._resources import get_logo_path
from napari_harpy.core.spatialdata import get_coordinate_system_names_from_sdata
from napari_harpy.core.validation import normalize_spatialdata_name
from napari_harpy.widgets.shared_styles import (
    PRIMARY_BUTTON_STYLESHEET,
    SECONDARY_BUTTON_STYLESHEET,
    WARNING_BUTTON_STYLESHEET,
    WIDGET_MIN_WIDTH,
    WIDGET_TEXT_COLOR,
    CompactComboBox,
    apply_scroll_content_surface,
    apply_widget_surface,
    build_input_control_stylesheet,
    create_form_label,
    set_status_card,
)

if TYPE_CHECKING:
    import napari
    from napari.layers import Shapes
    from spatialdata import SpatialData


_SOURCE = "shapes_annotation_widget"
_INPUT_CONTROL_STYLESHEET = build_input_control_stylesheet("QComboBox, QLineEdit")


class ShapesAnnotation(QWidget):
    """Widget shell for creating new SpatialData shapes elements."""

    def __init__(self, napari_viewer: napari.Viewer | None = None) -> None:
        super().__init__()
        self.setObjectName("shapes_annotation_widget")
        apply_widget_surface(self)
        self.setMinimumWidth(WIDGET_MIN_WIDTH)

        self._viewer = napari_viewer
        self._app_state = get_or_create_app_state(napari_viewer)
        self._coordinate_systems: list[str] = []
        self._selected_coordinate_system: str | None = None
        self._validated_shapes_name: str | None = None
        self._annotation_layer: Shapes | None = None
        self._annotation_shapes_name: str | None = None
        self._annotation_coordinate_system: str | None = None
        self._logo_path = get_logo_path()

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        self.scroll_area = QScrollArea()
        self.scroll_area.setObjectName("shapes_annotation_scroll_area")
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll_area.setStyleSheet("QScrollArea { border: 0px; background: transparent; }")

        self.scroll_content = QWidget()
        self.scroll_content.setObjectName("shapes_annotation_scroll_content")
        apply_scroll_content_surface(self.scroll_content)
        self.content_layout = QVBoxLayout(self.scroll_content)
        self.content_layout.setContentsMargins(12, 12, 12, 12)
        self.content_layout.setSpacing(10)

        header_logo = self._create_header_logo()

        self.status_label = QLabel()
        self.status_label.setObjectName("shapes_annotation_status_label")
        self.status_label.setWordWrap(True)

        form_layout = QFormLayout()
        form_layout.setContentsMargins(0, 0, 0, 0)
        form_layout.setSpacing(8)
        form_layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        self.coordinate_system_combo = CompactComboBox(minimum_contents_length=18)
        self.coordinate_system_combo.setObjectName("shapes_annotation_coordinate_system_combo")
        self.coordinate_system_combo.setPlaceholderText("Choose coordinate system")
        self.coordinate_system_combo.setStyleSheet(_INPUT_CONTROL_STYLESHEET)

        self.name_edit = QLineEdit()
        self.name_edit.setObjectName("shapes_annotation_name_edit")
        self.name_edit.setPlaceholderText("new_shapes")
        self.name_edit.setStyleSheet(_INPUT_CONTROL_STYLESHEET)

        form_layout.addRow(create_form_label("Coordinate System"), self.coordinate_system_combo)
        form_layout.addRow(create_form_label("Shapes Name"), self.name_edit)

        button_row = QHBoxLayout()
        button_row.setContentsMargins(0, 0, 0, 0)
        button_row.setSpacing(8)

        self.create_layer_button = QPushButton("Create layer")
        self.create_layer_button.setObjectName("shapes_annotation_create_layer_button")
        self.create_layer_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.create_layer_button.setStyleSheet(PRIMARY_BUTTON_STYLESHEET)

        self.save_shapes_button = QPushButton("Save shapes")
        self.save_shapes_button.setObjectName("shapes_annotation_save_shapes_button")
        self.save_shapes_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.save_shapes_button.setStyleSheet(SECONDARY_BUTTON_STYLESHEET)

        button_row.addWidget(self.create_layer_button)
        button_row.addWidget(self.save_shapes_button)

        self.content_layout.addWidget(header_logo)
        self.content_layout.addWidget(self.status_label)
        self.content_layout.addLayout(form_layout)
        self.content_layout.addLayout(button_row)
        self.content_layout.addStretch(1)

        self.scroll_area.setWidget(self.scroll_content)
        root_layout.addWidget(self.scroll_area)

        self._app_state.sdata_changed.connect(self._on_sdata_changed)
        self._app_state.coordinate_system_changed.connect(self._on_app_state_coordinate_system_changed)
        self.coordinate_system_combo.currentIndexChanged.connect(self._on_coordinate_system_changed)
        self.name_edit.textChanged.connect(self._on_shapes_name_changed)
        self.create_layer_button.clicked.connect(self._on_create_layer_clicked)
        self.refresh_from_sdata(self._app_state.sdata)

    @property
    def app_state(self) -> HarpyAppState:
        """Return the shared per-viewer Harpy app state."""
        return self._app_state

    @property
    def selected_spatialdata(self) -> SpatialData | None:
        """Return the loaded SpatialData object backing this widget."""
        return self._app_state.sdata

    @property
    def selected_coordinate_system(self) -> str | None:
        """Return the selected coordinate system."""
        return self._selected_coordinate_system

    @property
    def selected_shapes_name(self) -> str | None:
        """Return the validated new shapes element name."""
        return self._validated_shapes_name

    def refresh_from_sdata(self, sdata: SpatialData | None) -> None:
        """Refresh coordinate-system choices from shared SpatialData state."""
        # App-state sdata replacement removes registered layers before this
        # widget refreshes, so clear stale annotation UI state when our tracked
        # layer has already disappeared from the Harpy binding registry.
        if (
            self._annotation_layer is not None
            and self._app_state.viewer_adapter.layer_bindings.get_binding(self._annotation_layer) is None
        ):
            self._clear_annotation_state()

        with QSignalBlocker(self.coordinate_system_combo):
            self.coordinate_system_combo.clear()

            if sdata is None:
                self._coordinate_systems = []
                self.coordinate_system_combo.setEnabled(False)
            else:
                self._coordinate_systems = get_coordinate_system_names_from_sdata(sdata)
                for coordinate_system in self._coordinate_systems:
                    self.coordinate_system_combo.addItem(coordinate_system, coordinate_system)
                self.coordinate_system_combo.setEnabled(bool(self._coordinate_systems))

        self._sync_coordinate_system_combo_selection(self._app_state.coordinate_system)
        self._set_selected_coordinate_system(self.coordinate_system_combo.currentIndex())
        self._refresh_create_layer_state()

    def _on_sdata_changed(self, sdata: SpatialData | None) -> None:
        self.refresh_from_sdata(sdata)

    def _on_app_state_coordinate_system_changed(self, event: CoordinateSystemChangedEvent) -> None:
        del event
        self._sync_coordinate_system_combo_selection(self._app_state.coordinate_system)
        self._set_selected_coordinate_system(self.coordinate_system_combo.currentIndex())
        self._refresh_create_layer_state()

    def _on_coordinate_system_changed(self, index: int) -> None:
        coordinate_system = self.coordinate_system_combo.itemData(index)
        next_coordinate_system = coordinate_system if isinstance(coordinate_system, str) else None
        if self._annotation_layer is not None:
            if next_coordinate_system == self._app_state.coordinate_system:
                return
            # `False` means the user cancelled the discard warning, so restore
            # the previous coordinate-system selection and keep the layer.
            if not self._confirm_discard_annotation_layer():
                self._sync_coordinate_system_combo_selection(self._app_state.coordinate_system)
                self._set_selected_coordinate_system(self.coordinate_system_combo.currentIndex())
                self._refresh_create_layer_state()
                return
            self._remove_annotation_layer()
            self._clear_annotation_state()

        # Publish the UI choice to shared app state. `_on_app_state_coordinate_system_changed(...)`
        # owns local selection and create-layer refresh so all sources follow one path.
        self._app_state.set_coordinate_system(next_coordinate_system, source=_SOURCE)

    def _on_shapes_name_changed(self, _text: str) -> None:
        self._refresh_create_layer_state()

    def _on_create_layer_clicked(self) -> None:
        if self._annotation_layer is not None:
            return

        self._refresh_create_layer_state()
        sdata = self._app_state.sdata
        shapes_name = self._validated_shapes_name
        coordinate_system = self._selected_coordinate_system
        if sdata is None or shapes_name is None or coordinate_system is None:
            return

        try:
            layer = self._app_state.viewer_adapter.create_empty_primary_shapes_layer(
                sdata,
                shapes_name,
                coordinate_system,
            )
        except ValueError as error:
            self._set_status(title="Could Not Create Layer", lines=[str(error)], kind="warning")
            self._set_action_enabled(create_enabled=False)
            return

        self._annotation_layer = layer
        self._annotation_shapes_name = shapes_name
        self._annotation_coordinate_system = coordinate_system
        self.name_edit.setEnabled(False)
        self._app_state.viewer_adapter.activate_layer(layer)
        self._refresh_create_layer_state()

    def _sync_coordinate_system_combo_selection(self, coordinate_system: str | None) -> None:
        with QSignalBlocker(self.coordinate_system_combo):
            if coordinate_system is None:
                self.coordinate_system_combo.setCurrentIndex(-1)
                return

            index = self.coordinate_system_combo.findData(coordinate_system)
            self.coordinate_system_combo.setCurrentIndex(index)

    def _set_selected_coordinate_system(self, index: int) -> None:
        coordinate_system = self.coordinate_system_combo.itemData(index)
        self._selected_coordinate_system = coordinate_system if isinstance(coordinate_system, str) else None

    def _refresh_create_layer_state(self) -> None:
        """Update layer-creation readiness from current sdata, coordinate system, and name."""
        sdata = self._app_state.sdata
        coordinate_system = self._selected_coordinate_system
        self._validated_shapes_name = None

        if self._annotation_layer is not None:
            self._validated_shapes_name = self._annotation_shapes_name
            self._set_status(
                title="Annotation Layer Created",
                lines=["Draw shapes in the viewer, then save them when the save step is available."],
                kind="info",
            )
            self._set_action_enabled(create_enabled=False)
            return

        if sdata is None:
            self._set_status(
                title="No SpatialData Loaded",
                lines=["Load a SpatialData object before creating shapes."],
                kind="warning",
            )
            self._set_action_enabled(create_enabled=False)
            return

        if not self._coordinate_systems:
            self._set_status(
                title="No Coordinate Systems",
                lines=["The loaded SpatialData object does not expose any coordinate systems."],
                kind="warning",
            )
            self._set_action_enabled(create_enabled=False)
            return

        if coordinate_system is None:
            self._set_status(
                title="Choose Coordinate System",
                lines=["Select a coordinate system before creating shapes."],
                kind="warning",
            )
            self._set_action_enabled(create_enabled=False)
            return

        try:
            shapes_name = normalize_spatialdata_name(self.name_edit.text(), "Shapes element name")
        except ValueError as error:
            self._set_status(
                title="Invalid Shapes Name",
                lines=[str(error)],
                kind="warning",
            )
            self._set_action_enabled(create_enabled=False)
            return

        if shapes_name in sdata.shapes:
            self._set_status(
                title="Name Already Exists",
                lines=[f'Shapes element "{shapes_name}" already exists. Choose a new name.'],
                kind="warning",
            )
            self._set_action_enabled(create_enabled=False)
            return

        self._validated_shapes_name = shapes_name
        self._set_status(
            title="Ready",
            lines=[f'Create shapes layer "{shapes_name}" in coordinate system "{coordinate_system}".'],
            kind="info",
        )
        self._set_action_enabled(create_enabled=True)

    def _set_action_enabled(self, *, create_enabled: bool) -> None:
        self.create_layer_button.setEnabled(create_enabled)
        self.save_shapes_button.setEnabled(False)

    def _set_status(self, *, title: str, lines: list[str], kind: str) -> None:
        set_status_card(self.status_label, title=title, lines=lines, kind=kind)

    def _confirm_discard_annotation_layer(self) -> bool:
        dialog = QDialog(self)
        dialog.setWindowTitle("Discard Unsaved Shape Annotations")
        dialog.setModal(True)
        dialog.setMinimumWidth(560)

        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        warning_card = QLabel()
        warning_card.setWordWrap(True)
        set_status_card(
            warning_card,
            title="Discard Unsaved Annotations",
            lines=["Changing coordinate system will delete the current unsaved shape annotations."],
            kind="warning",
        )
        layout.addWidget(warning_card)

        button_row = QHBoxLayout()
        button_row.setSpacing(10)
        button_row.addStretch(1)

        discard_button = QPushButton("Discard annotations")
        cancel_button = QPushButton("Cancel")

        discard_button.setStyleSheet(WARNING_BUTTON_STYLESHEET)
        cancel_button.setStyleSheet(SECONDARY_BUTTON_STYLESHEET)

        button_row.addWidget(discard_button)
        button_row.addWidget(cancel_button)
        layout.addLayout(button_row)

        discard_button.clicked.connect(dialog.accept)
        cancel_button.clicked.connect(dialog.reject)
        cancel_button.setDefault(True)

        return dialog.exec() == QDialog.DialogCode.Accepted

    def _remove_annotation_layer(self) -> None:
        sdata = self._app_state.sdata
        if sdata is None or self._annotation_shapes_name is None or self._annotation_coordinate_system is None:
            return
        self._app_state.viewer_adapter.remove_shapes_layer(
            sdata,
            self._annotation_shapes_name,
            self._annotation_coordinate_system,
        )

    def _clear_annotation_state(self) -> None:
        self._annotation_layer = None
        self._annotation_shapes_name = None
        self._annotation_coordinate_system = None
        self.name_edit.setEnabled(True)

    def _create_header_logo(self) -> QLabel:
        logo_label = QLabel()
        logo_label.setObjectName("shapes_annotation_header_logo")
        logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        logo_pixmap = QPixmap(str(self._logo_path))
        if not logo_pixmap.isNull():
            logo_label.setPixmap(logo_pixmap.scaledToWidth(120, Qt.TransformationMode.SmoothTransformation))
            return logo_label

        logo_label.setText("napari-harpy")
        logo_label.setStyleSheet(f"color: {WIDGET_TEXT_COLOR}; font-size: 18px; font-weight: 600;")
        return logo_label
