from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype, is_object_dtype, is_string_dtype
from qtpy.QtCore import QSignalBlocker, Qt
from qtpy.QtGui import QPixmap
from qtpy.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QLabel,
    QLayout,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)
from spatialdata import read_zarr

from napari_harpy._app_state import CoordinateSystemChangedEvent, HarpyAppState, get_or_create_app_state
from napari_harpy.core.spatialdata import (
    get_annotating_table_names,
    get_coordinate_system_names_from_sdata,
    get_image_channel_names_from_sdata,
    get_shape_column_color_source_options,
    get_spatialdata_image_options_for_coordinate_system_from_sdata,
    get_spatialdata_labels_options_for_coordinate_system_from_sdata,
    get_spatialdata_points_options_for_coordinate_system_from_sdata,
    get_spatialdata_shapes_options_for_coordinate_system_from_sdata,
    get_table_color_source_options,
)
from napari_harpy.viewer.adapter import ShapesLayerBinding, ViewerAdapter
from napari_harpy.viewer.points_styling import PointsLayerResult
from napari_harpy.widgets.shared_styles import (
    ACTION_BUTTON_STYLESHEET,
    WIDGET_MIN_WIDTH,
    StatusCardKind,
    apply_scroll_content_surface,
    apply_widget_surface,
    create_form_label,
    set_status_card,
)
from napari_harpy.widgets.viewer.disclosure import _CollapsibleSectionWidget, _DisclosureElementWidget
from napari_harpy.widgets.viewer.image_widget import ImageLoadRequest, _ImageCardWidget
from napari_harpy.widgets.viewer.labels_widget import LabelsLoadRequest, _LabelsCardWidget
from napari_harpy.widgets.viewer.points_controller import PointsController, PointsLoadResult, PointsValueSource
from napari_harpy.widgets.viewer.points_widget import PointsValueWidget
from napari_harpy.widgets.viewer.shapes_widget import ShapesLoadRequest, _ShapesCardWidget
from napari_harpy.widgets.viewer.status_card import (
    _ViewerStatusCardSpec,
    build_image_loaded_card_spec,
    build_points_layer_card_spec,
    build_primary_labels_loaded_card_spec,
    build_primary_shapes_loaded_card_spec,
    build_styled_labels_card_spec,
    build_styled_shapes_card_spec,
)
from napari_harpy.widgets.viewer.styles import (
    EMPTY_STATE_STYLESHEET,
    INPUT_CONTROL_STYLESHEET,
    SUMMARY_LABEL_STYLESHEET,
)

if TYPE_CHECKING:
    import napari
    from spatialdata import SpatialData


class ViewerWidget(QWidget):
    """Shared viewer widget backed by `HarpyAppState` and `ViewerAdapter`."""

    def __init__(self, napari_viewer: napari.Viewer | None = None) -> None:
        super().__init__()
        self.setObjectName("viewer_widget")
        apply_widget_surface(self)
        self.setMinimumWidth(WIDGET_MIN_WIDTH)
        self._viewer = napari_viewer
        self._app_state = get_or_create_app_state(napari_viewer)
        self._points_controller = PointsController(
            on_state_changed=self._on_points_controller_state_changed,
            on_value_source_loaded=self._on_points_value_source_loaded,
            on_points_loaded=self._on_points_loaded,
        )
        self._last_points_load_result: PointsLoadResult | None = None
        self._last_points_layer_result: PointsLayerResult | None = None
        self._labels_cards: list[_LabelsCardWidget] = []
        self._image_cards: list[_ImageCardWidget] = []
        self._shape_cards: list[_ShapesCardWidget] = []
        self._labels_rows: list[_DisclosureElementWidget] = []
        self._image_rows: list[_DisclosureElementWidget] = []
        self._shape_rows: list[_DisclosureElementWidget] = []
        self._expanded_labels_names: set[str] = set()
        self._expanded_image_names: set[str] = set()
        self._expanded_shapes_names: set[str] = set()
        self._logo_path = Path(__file__).resolve().parents[4] / "docs" / "_static" / "logo.png"

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

        header_logo = self._create_header_logo()

        self.open_sdata_button = QPushButton("Load SpatialData")
        self.open_sdata_button.setObjectName("viewer_widget_open_sdata_button")
        self.open_sdata_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.open_sdata_button.setMinimumHeight(28)
        self.open_sdata_button.setStyleSheet(ACTION_BUTTON_STYLESHEET)
        self.open_sdata_button.clicked.connect(self._open_spatialdata)

        self.empty_state_label = QLabel(
            "No SpatialData loaded. Use `Interactive(sdata)` for now; an in-widget open action will follow later."
        )
        self.empty_state_label.setObjectName("viewer_widget_empty_state")
        self.empty_state_label.setWordWrap(True)
        self.empty_state_label.setStyleSheet(EMPTY_STATE_STYLESHEET)

        self.summary_label = QLabel("No SpatialData loaded.")
        self.summary_label.setObjectName("viewer_widget_summary")
        self.summary_label.setWordWrap(True)
        self.summary_label.setStyleSheet(SUMMARY_LABEL_STYLESHEET)

        selector_layout = QFormLayout()
        selector_layout.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        selector_layout.setHorizontalSpacing(12)
        selector_layout.setVerticalSpacing(10)

        self.coordinate_system_combo = QComboBox()
        self.coordinate_system_combo.setObjectName("viewer_widget_coordinate_system_combo")
        self.coordinate_system_combo.currentIndexChanged.connect(self._on_coordinate_system_changed)
        self.coordinate_system_combo.setStyleSheet(INPUT_CONTROL_STYLESHEET)
        selector_layout.addRow(create_form_label("Coordinate system"), self.coordinate_system_combo)

        self.global_action_feedback_label = QLabel("")
        self.global_action_feedback_label.setObjectName("viewer_widget_action_feedback")
        self.global_action_feedback_label.setWordWrap(True)
        self.global_action_feedback_label.hide()

        self.images_empty_label = QLabel("No images available in the selected coordinate system.")
        self.images_empty_label.setObjectName("viewer_widget_images_empty_state")
        self.images_empty_label.setWordWrap(True)
        self.images_empty_label.setStyleSheet(EMPTY_STATE_STYLESHEET)

        self.images_section = QWidget()
        self.images_section.setObjectName("viewer_widget_images_section")
        self.images_section_layout = QVBoxLayout(self.images_section)
        self.images_section_layout.setContentsMargins(0, 0, 0, 0)
        self.images_section_layout.setSpacing(8)
        self.images_group = _CollapsibleSectionWidget(
            title="Images",
            object_name="viewer_widget_images_group",
            toggle_object_name="viewer_widget_images_section_toggle",
            expanded=False,
        )
        self.images_group.content_layout.addWidget(self.images_empty_label)
        self.images_group.content_layout.addWidget(self.images_section)
        self.images_section_toggle = self.images_group.toggle_button
        self.images_section_title = self.images_section_toggle

        self.labels_empty_label = QLabel("No labels available in the selected coordinate system.")
        self.labels_empty_label.setObjectName("viewer_widget_labels_empty_state")
        self.labels_empty_label.setWordWrap(True)
        self.labels_empty_label.setStyleSheet(EMPTY_STATE_STYLESHEET)

        self.labels_section = QWidget()
        self.labels_section.setObjectName("viewer_widget_labels_section")
        self.labels_section_layout = QVBoxLayout(self.labels_section)
        self.labels_section_layout.setContentsMargins(0, 0, 0, 0)
        self.labels_section_layout.setSpacing(8)
        self.labels_group = _CollapsibleSectionWidget(
            title="Labels",
            object_name="viewer_widget_labels_group",
            toggle_object_name="viewer_widget_labels_section_toggle",
            expanded=False,
        )
        self.labels_group.content_layout.addWidget(self.labels_empty_label)
        self.labels_group.content_layout.addWidget(self.labels_section)
        self.labels_section_toggle = self.labels_group.toggle_button
        self.labels_section_title = self.labels_section_toggle

        self.shapes_empty_label = QLabel("No shapes available in the selected coordinate system.")
        self.shapes_empty_label.setObjectName("viewer_widget_shapes_empty_state")
        self.shapes_empty_label.setWordWrap(True)
        self.shapes_empty_label.setStyleSheet(EMPTY_STATE_STYLESHEET)

        self.shapes_section = QWidget()
        self.shapes_section.setObjectName("viewer_widget_shapes_section")
        self.shapes_section_layout = QVBoxLayout(self.shapes_section)
        self.shapes_section_layout.setContentsMargins(0, 0, 0, 0)
        self.shapes_section_layout.setSpacing(8)
        self.shapes_group = _CollapsibleSectionWidget(
            title="Shapes",
            object_name="viewer_widget_shapes_group",
            toggle_object_name="viewer_widget_shapes_section_toggle",
            expanded=False,
        )
        self.shapes_group.content_layout.addWidget(self.shapes_empty_label)
        self.shapes_group.content_layout.addWidget(self.shapes_section)
        self.shapes_section_toggle = self.shapes_group.toggle_button
        self.shapes_section_title = self.shapes_section_toggle

        self.points_empty_label = QLabel("No points available in the selected coordinate system.")
        self.points_empty_label.setObjectName("viewer_widget_points_empty_state")
        self.points_empty_label.setWordWrap(True)
        self.points_empty_label.setStyleSheet(EMPTY_STATE_STYLESHEET)

        self.points_widget = PointsValueWidget()
        self.points_widget.source_changed.connect(self._on_points_source_changed)
        self.points_widget.add_update_requested.connect(self._add_or_update_points_selection)
        self.points_group = _CollapsibleSectionWidget(
            title="Points",
            object_name="viewer_widget_points_group",
            toggle_object_name="viewer_widget_points_section_toggle",
            expanded=False,
        )
        self.points_group.content_layout.addWidget(self.points_empty_label)
        self.points_group.content_layout.addWidget(self.points_widget)
        self.points_section_toggle = self.points_group.toggle_button
        self.points_section_title = self.points_section_toggle

        self.content_layout.addWidget(header_logo)
        self.content_layout.addWidget(self.open_sdata_button)
        self.content_layout.addWidget(self.empty_state_label)
        self.content_layout.addWidget(self.summary_label)
        self.content_layout.addLayout(selector_layout)
        self.content_layout.addWidget(self.global_action_feedback_label)
        self.content_layout.addWidget(self.images_group)
        self.content_layout.addWidget(self.labels_group)
        self.content_layout.addWidget(self.shapes_group)
        self.content_layout.addWidget(self.points_group)
        self.content_layout.addStretch(1)

        self.scroll_area.setWidget(self.scroll_content)
        root_layout.addWidget(self.scroll_area)

        self._app_state.sdata_changed.connect(self._on_sdata_changed)
        self._app_state.coordinate_system_changed.connect(self._on_app_state_coordinate_system_changed)
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

    @property
    def shape_cards(self) -> list[_ShapesCardWidget]:
        """Return the currently visible shapes cards."""
        return list(self._shape_cards)

    @property
    def image_rows(self) -> list[_DisclosureElementWidget]:
        """Return the currently visible compact image rows."""
        return list(self._image_rows)

    @property
    def labels_rows(self) -> list[_DisclosureElementWidget]:
        """Return the currently visible compact labels rows."""
        return list(self._labels_rows)

    @property
    def shape_rows(self) -> list[_DisclosureElementWidget]:
        """Return the currently visible compact shapes rows."""
        return list(self._shape_rows)

    def _on_sdata_changed(self, sdata: SpatialData | None) -> None:
        """Refresh the widget when the shared loaded `SpatialData` changes."""
        self.refresh_from_sdata(sdata)

    def _on_coordinate_system_changed(self, index: int) -> None:
        """Publish explicit user coordinate-system changes to shared app state."""
        coordinate_system = self.coordinate_system_combo.itemData(index)
        self._app_state.set_coordinate_system(
            coordinate_system if isinstance(coordinate_system, str) else None,
            source="viewer_widget",
        )

    def _on_app_state_coordinate_system_changed(self, event: CoordinateSystemChangedEvent) -> None:
        """Refresh the combo and cards when the shared coordinate system changes."""
        del event
        self._clear_action_feedback()
        self._sync_coordinate_system_combo_selection(self._app_state.coordinate_system)
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
            self._set_action_feedback(
                title="SpatialData Load Error",
                lines=[f"Could not load SpatialData store: {error}"],
                kind="error",
            )
            return

        self._app_state.set_sdata(sdata)
        self._clear_action_feedback()

    def refresh_from_sdata(self, sdata: SpatialData | None) -> None:
        """Refresh the viewer widget from the currently loaded `SpatialData`."""
        with QSignalBlocker(self.coordinate_system_combo):
            self.coordinate_system_combo.clear()

            if sdata is None:
                self.empty_state_label.show()
                self.summary_label.setText("No SpatialData loaded.")
                self.coordinate_system_combo.setEnabled(False)
                self._clear_cards()
                self._update_section_empty_states([], [], [], [])
                return

            coordinate_systems = get_coordinate_system_names_from_sdata(sdata)
            for coordinate_system in coordinate_systems:
                self.coordinate_system_combo.addItem(coordinate_system, coordinate_system)

            self.empty_state_label.hide()
            self.coordinate_system_combo.setEnabled(bool(coordinate_systems))

        self._sync_coordinate_system_combo_selection(self._app_state.coordinate_system)
        self._refresh_coordinate_system_content()

    def _refresh_coordinate_system_content(self) -> None:
        sdata = self._app_state.sdata
        coordinate_system = self._app_state.coordinate_system

        if sdata is None or not coordinate_system:
            self._clear_cards()
            self._update_section_empty_states([], [], [], [])
            if sdata is None:
                self.summary_label.setText("No SpatialData loaded.")
            else:
                self.summary_label.setText("No coordinate system selected.")
            return

        labels_names = _get_labels_in_coordinate_system(sdata, coordinate_system)
        image_names = _get_images_in_coordinate_system(sdata, coordinate_system)
        shapes_names = _get_shapes_in_coordinate_system(sdata, coordinate_system)
        points_names = _get_points_in_coordinate_system(sdata, coordinate_system)

        self.summary_label.setText(
            f"In coordinate system `{coordinate_system}`: "
            f"{len(image_names)} image element(s), {len(labels_names)} labels element(s), "
            f"{len(shapes_names)} shapes element(s), and {len(points_names)} points element(s)."
        )
        self._rebuild_image_cards(sdata, image_names)
        self._rebuild_labels_cards(sdata, labels_names)
        self._rebuild_shapes_cards(sdata, shapes_names)
        self._refresh_points_section(sdata, points_names)
        self._update_section_empty_states(image_names, labels_names, shapes_names, points_names)

    def _rebuild_image_cards(self, sdata: SpatialData, image_names: list[str]) -> None:
        _clear_layout(self.images_section_layout)
        self._image_cards = []
        self._image_rows = []
        self._expanded_image_names.intersection_update(image_names)

        for image_name in image_names:
            channel_error = None
            channel_names: list[str] = []
            try:
                channel_names = get_image_channel_names_from_sdata(sdata, image_name)
            except ValueError as error:
                channel_error = str(error)
            card = _ImageCardWidget(
                image_name=image_name,
                channel_names=channel_names,
                channel_error=channel_error,
            )
            card.add_update_requested.connect(self._add_or_update_image_layer)
            row = _DisclosureElementWidget(
                title=image_name,
                object_name=f"viewer_widget_image_row_{image_name}",
                toggle_object_name=f"viewer_widget_image_row_toggle_{image_name}",
                detail_widget=card,
                expanded=image_name in self._expanded_image_names,
            )
            row.expanded_changed.connect(
                lambda expanded, *, name=image_name: self._on_image_row_expanded(
                    name,
                    expanded,
                )
            )
            self.images_section_layout.addWidget(row)
            self._image_cards.append(card)
            self._image_rows.append(row)

    def _rebuild_labels_cards(self, sdata: SpatialData, labels_names: list[str]) -> None:
        _clear_layout(self.labels_section_layout)
        self._labels_cards = []
        self._labels_rows = []
        self._expanded_labels_names.intersection_update(labels_names)

        for labels_name in labels_names:
            table_names = get_annotating_table_names(sdata, labels_name)
            table_color_sources_by_table = {
                table_name: get_table_color_source_options(sdata, table_name) for table_name in table_names
            }
            card = _LabelsCardWidget(
                labels_name=labels_name,
                table_names=table_names,
                table_color_sources_by_table=table_color_sources_by_table,
            )
            card.add_update_requested.connect(self._add_or_update_labels_layer)
            row = _DisclosureElementWidget(
                title=labels_name,
                object_name=f"viewer_widget_labels_row_{labels_name}",
                toggle_object_name=f"viewer_widget_labels_row_toggle_{labels_name}",
                detail_widget=card,
                expanded=labels_name in self._expanded_labels_names,
            )
            row.expanded_changed.connect(
                lambda expanded, *, name=labels_name: self._on_labels_row_expanded(
                    name,
                    expanded,
                )
            )
            self.labels_section_layout.addWidget(row)
            self._labels_cards.append(card)
            self._labels_rows.append(row)

    def _rebuild_shapes_cards(self, sdata: SpatialData, shapes_names: list[str]) -> None:
        _clear_layout(self.shapes_section_layout)
        self._shape_cards = []
        self._shape_rows = []
        self._expanded_shapes_names.intersection_update(shapes_names)

        for shapes_name in shapes_names:
            card = _ShapesCardWidget(
                shapes_name=shapes_name,
                shape_color_sources=get_shape_column_color_source_options(sdata, shapes_name),
            )
            card.add_update_requested.connect(self._add_or_update_shapes_layer)
            row = _DisclosureElementWidget(
                title=shapes_name,
                object_name=f"viewer_widget_shapes_row_{shapes_name}",
                toggle_object_name=f"viewer_widget_shapes_row_toggle_{shapes_name}",
                detail_widget=card,
                expanded=shapes_name in self._expanded_shapes_names,
            )
            row.expanded_changed.connect(
                lambda expanded, *, name=shapes_name: self._on_shapes_row_expanded(
                    name,
                    expanded,
                )
            )
            self.shapes_section_layout.addWidget(row)
            self._shape_cards.append(card)
            self._shape_rows.append(row)

    def _refresh_points_section(self, sdata: SpatialData, points_names: list[str]) -> None:
        self.points_widget.set_points_names(points_names)
        self._refresh_points_index_columns(sdata)
        self._bind_points_source()

    def _refresh_points_index_columns(self, sdata: SpatialData | None) -> None:
        points_name = self.points_widget.selected_points_name()
        index_columns = [] if sdata is None or points_name is None else _get_points_index_columns(sdata, points_name)
        self.points_widget.set_index_columns(index_columns)

    def _on_points_source_changed(self) -> None:
        sdata = self._app_state.sdata
        self._refresh_points_index_columns(sdata)
        self._bind_points_source()

    def _bind_points_source(self) -> None:
        sdata = self._app_state.sdata
        coordinate_system = self._app_state.coordinate_system
        points_name = self.points_widget.selected_points_name()
        index_column = self.points_widget.selected_index_column()

        changed = self._points_controller.bind_source(
            sdata,
            points_name,
            coordinate_system,
            index_column,
        )
        if changed:
            self._last_points_load_result = None
            self._last_points_layer_result = None
        if changed and self._points_controller.can_load_values:
            self._points_controller.load_value_source()
        else:
            self.points_widget.render_controller_state(self._points_controller)

    def _add_or_update_points_selection(self, values: Sequence[str] | Literal["all"], render_point_budget: int) -> None:
        self._last_points_load_result = None
        self._last_points_layer_result = None
        self._points_controller.load_selection(
            values,
            render_point_budget=render_point_budget,
        )

    def _on_points_controller_state_changed(self) -> None:
        self.points_widget.render_controller_state(self._points_controller)
        if self._last_points_load_result is not None and self._last_points_layer_result is not None:
            self._render_points_loaded_status(self._last_points_load_result, self._last_points_layer_result)

    def _on_points_value_source_loaded(self, value_source: PointsValueSource) -> None:
        self.points_widget.set_value_source(value_source)

    def _on_points_loaded(self, load_result: PointsLoadResult) -> None:
        try:
            layer_result = self._app_state.viewer_adapter._ensure_points_layer_from_selection(
                load_result.identity,
                selection=load_result.selection,
                categorical_colors=load_result.selected_value_colors or None,
            )
        except ValueError as error:
            self._set_action_feedback(
                title="Points Layer Error",
                lines=[str(error)],
                kind="error",
            )
            return

        self._last_points_load_result = load_result
        self._last_points_layer_result = layer_result
        self._app_state.viewer_adapter.activate_layer(layer_result.layer)
        self._render_points_loaded_status(load_result, layer_result)

    def _render_points_loaded_status(
        self,
        load_result: PointsLoadResult,
        layer_result: PointsLayerResult,
    ) -> None:
        self._apply_status_card_spec(
            self.global_action_feedback_label,
            build_points_layer_card_spec(load_result, layer_result),
        )

    def _on_image_row_expanded(
        self,
        image_name: str,
        expanded: bool,
    ) -> None:
        if expanded:
            self._expanded_image_names.add(image_name)
            return

        self._expanded_image_names.discard(image_name)

    def _on_labels_row_expanded(
        self,
        labels_name: str,
        expanded: bool,
    ) -> None:
        if expanded:
            self._expanded_labels_names.add(labels_name)
            return

        self._expanded_labels_names.discard(labels_name)

    def _on_shapes_row_expanded(
        self,
        shapes_name: str,
        expanded: bool,
    ) -> None:
        if expanded:
            self._expanded_shapes_names.add(shapes_name)
            return

        self._expanded_shapes_names.discard(shapes_name)

    def _add_or_update_labels_layer(self, request: LabelsLoadRequest) -> None:
        if request.selected_source_kind is None:
            self._add_or_update_primary_labels_layer(request)
            return

        self._add_or_update_styled_labels_layer(request)

    def _add_or_update_primary_labels_layer(self, request: LabelsLoadRequest) -> None:
        sdata = self._app_state.sdata
        coordinate_system = self._app_state.coordinate_system
        labels_name = request.labels_name

        if sdata is None or not coordinate_system:
            self._set_action_feedback(
                title="Labels Load Error",
                lines=["Load a SpatialData object and select a coordinate system first."],
                kind="error",
            )
            return

        try:
            layer = self._app_state.viewer_adapter.ensure_labels_loaded(sdata, labels_name, coordinate_system)
        except ValueError as error:
            self._set_action_feedback(title="Labels Load Error", lines=[str(error)], kind="error")
            return

        self._app_state.viewer_adapter.activate_layer(layer)
        self._apply_status_card_spec(
            self.global_action_feedback_label,
            build_primary_labels_loaded_card_spec(request, coordinate_system),
        )

    def _add_or_update_styled_labels_layer(self, request: LabelsLoadRequest) -> None:
        sdata = self._app_state.sdata
        coordinate_system = self._app_state.coordinate_system

        if sdata is None or not coordinate_system:
            self._set_action_feedback(
                title="Colored Overlay Error",
                lines=["Load a SpatialData object and select a coordinate system first."],
                kind="error",
            )
            return

        if request.table_name is None:
            self._set_action_feedback(
                title="Colored Overlay Error",
                lines=[f"Labels element `{request.labels_name}` has no linked table for table-driven coloring."],
                kind="error",
            )
            return

        if request.selected_color_source is None:
            missing_source_label = "observation column" if request.selected_source_kind == "obs_column" else "var"
            missing_source_article = "an" if request.selected_source_kind == "obs_column" else "a"
            self._set_action_feedback(
                title="Colored Overlay Error",
                lines=[
                    f"Select {missing_source_article} {missing_source_label} "
                    f"to create a colored overlay for `{request.labels_name}`."
                ],
                kind="error",
            )
            return

        try:
            result = self._app_state.viewer_adapter.ensure_styled_labels_loaded(
                sdata,
                request.labels_name,
                coordinate_system,
                request.selected_color_source,
            )
        except ValueError as error:
            self._set_action_feedback(title="Colored Overlay Error", lines=[str(error)], kind="error")
            return

        self._app_state.viewer_adapter.activate_layer(result.layer)
        self._apply_status_card_spec(
            self.global_action_feedback_label,
            build_styled_labels_card_spec(request, result, coordinate_system),
        )

    def _add_or_update_shapes_layer(self, request: ShapesLoadRequest) -> None:
        if request.selected_source_kind is None:
            self._add_or_update_primary_shapes_layer(request.shapes_name)
            return

        self._add_or_update_styled_shapes_layer(request)

    def _add_or_update_primary_shapes_layer(self, shapes_name: str) -> None:
        sdata = self._app_state.sdata
        coordinate_system = self._app_state.coordinate_system

        if sdata is None or not coordinate_system:
            self._set_action_feedback(
                title="Shapes Load Error",
                lines=["Load a SpatialData object and select a coordinate system first."],
                kind="error",
            )
            return

        try:
            layer = self._app_state.viewer_adapter.ensure_shapes_loaded(sdata, shapes_name, coordinate_system)
        except ValueError as error:
            self._set_action_feedback(title="Shapes Load Error", lines=[str(error)], kind="error")
            return

        self._app_state.viewer_adapter.activate_layer(layer)
        skipped_geometry_count = _get_layer_skipped_geometry_count(self._app_state.viewer_adapter, layer)
        self._apply_status_card_spec(
            self.global_action_feedback_label,
            build_primary_shapes_loaded_card_spec(shapes_name, coordinate_system, skipped_geometry_count),
        )

    def _add_or_update_styled_shapes_layer(self, request: ShapesLoadRequest) -> None:
        sdata = self._app_state.sdata
        coordinate_system = self._app_state.coordinate_system

        if sdata is None or not coordinate_system:
            self._set_action_feedback(
                title="Styled Shapes Error",
                lines=["Load a SpatialData object and select a coordinate system first."],
                kind="error",
            )
            return

        if request.selected_color_source is None:
            self._set_action_feedback(
                title="Styled Shapes Error",
                lines=[f"Select a shapes column to create a styled shapes layer for `{request.shapes_name}`."],
                kind="error",
            )
            return

        try:
            result = self._app_state.viewer_adapter.ensure_styled_shapes_loaded(
                sdata,
                request.shapes_name,
                coordinate_system,
                request.selected_color_source,
                fill=request.fill_shapes,
            )
        except ValueError as error:
            self._set_action_feedback(title="Styled Shapes Error", lines=[str(error)], kind="error")
            return

        self._app_state.viewer_adapter.activate_layer(result.layer)
        skipped_geometry_count = _get_layer_skipped_geometry_count(self._app_state.viewer_adapter, result.layer)
        self._apply_status_card_spec(
            self.global_action_feedback_label,
            build_styled_shapes_card_spec(request, result, coordinate_system, skipped_geometry_count),
        )

    def _add_or_update_image_layer(self, request: ImageLoadRequest) -> None:
        sdata = self._app_state.sdata
        coordinate_system = self._app_state.coordinate_system
        image_name = request.image_name

        if sdata is None or not coordinate_system:
            self._set_action_feedback(
                title="Image Load Error",
                lines=["Load a SpatialData object and select a coordinate system first."],
                kind="error",
            )
            return

        mode = request.mode
        if mode == "overlay" and not request.channels:
            self._app_state.viewer_adapter.remove_image_layers(sdata, image_name, coordinate_system)
            self._set_action_feedback(
                title="Image Load Error",
                lines=["Overlay mode requires at least one selected channel."],
                kind="error",
            )
            return

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
            self._set_action_feedback(title="Image Load Error", lines=[str(error)], kind="error")
            return

        if mode == "stack":
            if isinstance(layer_or_layers, list):
                self._set_action_feedback(
                    title="Image Load Error",
                    lines=[f"Expected one stack image layer for `{image_name}`, but received multiple layers."],
                    kind="error",
                )
                return

            self._app_state.viewer_adapter.activate_layer(layer_or_layers)
            self._apply_status_card_spec(
                self.global_action_feedback_label,
                build_image_loaded_card_spec(image_name, coordinate_system, "stack"),
            )
            return

        if not isinstance(layer_or_layers, list):
            self._set_action_feedback(
                title="Image Load Error",
                lines=[f"Expected overlay image layers for `{image_name}`, but received a single layer."],
                kind="error",
            )
            return
        if not layer_or_layers:
            self._set_action_feedback(
                title="Image Load Error",
                lines=[f"No overlay layers were returned for image `{image_name}`."],
                kind="error",
            )
            return

        self._app_state.viewer_adapter.activate_layer(layer_or_layers[0])
        self._apply_status_card_spec(
            self.global_action_feedback_label,
            build_image_loaded_card_spec(image_name, coordinate_system, "overlay", request.channels),
        )

    def _update_section_empty_states(
        self,
        image_names: list[str],
        labels_names: list[str],
        shapes_names: list[str],
        points_names: list[str],
    ) -> None:
        self.images_group.set_count(len(image_names))
        self.labels_group.set_count(len(labels_names))
        self.shapes_group.set_count(len(shapes_names))
        self.points_group.set_count(len(points_names))
        self.images_empty_label.setVisible(not image_names)
        self.labels_empty_label.setVisible(not labels_names)
        self.shapes_empty_label.setVisible(not shapes_names)
        self.points_empty_label.setVisible(not points_names)
        self.images_section.setVisible(bool(image_names))
        self.labels_section.setVisible(bool(labels_names))
        self.shapes_section.setVisible(bool(shapes_names))
        self.points_widget.setVisible(bool(points_names))

    def _clear_cards(self) -> None:
        _clear_layout(self.images_section_layout)
        _clear_layout(self.labels_section_layout)
        _clear_layout(self.shapes_section_layout)
        self._image_cards = []
        self._labels_cards = []
        self._shape_cards = []
        self._image_rows = []
        self._labels_rows = []
        self._shape_rows = []
        self._expanded_image_names.clear()
        self._expanded_labels_names.clear()
        self._expanded_shapes_names.clear()
        self.points_widget.set_points_names([])
        self.points_widget.set_index_columns([])
        self.points_widget.set_value_source(None)
        self._last_points_load_result = None
        self._last_points_layer_result = None
        self._points_controller.bind_source(None, None, None, None)

    def _set_action_feedback(
        self,
        message: str | None = None,
        *,
        title: str | None = None,
        lines: list[str] | None = None,
        kind: StatusCardKind | None = None,
        is_error: bool | None = None,
        tooltip_message: str | None = None,
    ) -> None:
        """Render viewer action feedback as the shared status-card pattern."""
        if lines is None:
            if message is None:
                lines = []
            else:
                lines = [message]
        if kind is None:
            kind = "error" if is_error else "success"
        if title is None:
            title = "Viewer Error" if kind == "error" else "Viewer Updated"

        self._apply_status_card_spec(
            self.global_action_feedback_label,
            _ViewerStatusCardSpec(
                title=title,
                lines=tuple(lines),
                kind=kind,
                tooltip_message=tooltip_message,
            ),
        )

    def _apply_status_card_spec(self, label: QLabel, spec: _ViewerStatusCardSpec | None) -> None:
        if spec is None:
            label.clear()
            label.setToolTip("")
            label.setStyleSheet("")
            label.hide()
            return

        set_status_card(
            label,
            title=spec.title,
            lines=list(spec.lines),
            kind=spec.kind,
            tooltip_message=spec.tooltip_message,
        )

    def _clear_action_feedback(self) -> None:
        self._apply_status_card_spec(self.global_action_feedback_label, None)

    def _sync_coordinate_system_combo_selection(self, coordinate_system: str | None) -> None:
        with QSignalBlocker(self.coordinate_system_combo):
            if coordinate_system is None:
                self.coordinate_system_combo.setCurrentIndex(-1)
                return

            index = self.coordinate_system_combo.findData(coordinate_system)
            self.coordinate_system_combo.setCurrentIndex(index)

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


def _get_labels_in_coordinate_system(sdata: SpatialData, coordinate_system: str) -> list[str]:
    return [
        option.labels_name
        for option in get_spatialdata_labels_options_for_coordinate_system_from_sdata(
            sdata=sdata,
            coordinate_system=coordinate_system,
        )
    ]


def _get_images_in_coordinate_system(sdata: SpatialData, coordinate_system: str) -> list[str]:
    return [
        option.image_name
        for option in get_spatialdata_image_options_for_coordinate_system_from_sdata(
            sdata=sdata,
            coordinate_system=coordinate_system,
        )
    ]


def _get_shapes_in_coordinate_system(sdata: SpatialData, coordinate_system: str) -> list[str]:
    return [
        option.shapes_name
        for option in get_spatialdata_shapes_options_for_coordinate_system_from_sdata(
            sdata=sdata,
            coordinate_system=coordinate_system,
        )
    ]


def _get_points_in_coordinate_system(sdata: SpatialData, coordinate_system: str) -> list[str]:
    return [
        option.points_name
        for option in get_spatialdata_points_options_for_coordinate_system_from_sdata(
            sdata=sdata,
            coordinate_system=coordinate_system,
        )
    ]


def _get_points_index_columns(sdata: SpatialData, points_name: str) -> list[str]:
    points = getattr(sdata, "points", {}).get(points_name)
    if points is None:
        return []

    meta = getattr(points, "_meta", points)
    columns = list(getattr(meta, "columns", ()))
    index_columns: list[str] = []
    for column in columns:
        column_name = str(column)
        if column_name in {"x", "y"}:
            continue

        dtype = getattr(meta[column], "dtype", None)
        if dtype is None or is_bool_dtype(dtype) or is_numeric_dtype(dtype):
            continue
        if isinstance(dtype, pd.CategoricalDtype) or is_string_dtype(dtype) or is_object_dtype(dtype):
            index_columns.append(column_name)

    return index_columns


def _get_layer_skipped_geometry_count(viewer_adapter: ViewerAdapter, layer: object) -> int:
    binding = viewer_adapter.layer_bindings.get_binding(layer)
    if not isinstance(binding, ShapesLayerBinding):
        return 0

    try:
        return int(binding.skipped_geometry_count)
    except (TypeError, ValueError):
        return 0
