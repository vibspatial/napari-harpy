"""Context-driven Spatial Query child widget shell."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

from qtpy.QtCore import QSignalBlocker, Qt, Signal
from qtpy.QtWidgets import QFormLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QVBoxLayout, QWidget

from napari_harpy._app_state import HarpyAppState, get_or_create_app_state
from napari_harpy.core.spatial_query import (
    CanonicalCacheReport,
    build_canonical_source_signature,
    get_compatible_spatial_annotation_column_names,
    inspect_canonical_cache,
)
from napari_harpy.core.spatialdata import (
    get_annotating_table_names,
    get_spatialdata_labels_options_for_coordinate_system_from_sdata,
    get_table_metadata,
)
from napari_harpy.core.validation import normalize_spatialdata_dataframe_column_name
from napari_harpy.widgets.annotation.models import AnnotationContext
from napari_harpy.widgets.shared_styles import (
    ACTION_BUTTON_STYLESHEET,
    PRIMARY_BUTTON_STYLESHEET,
    CompactComboBox,
    build_input_control_stylesheet,
    create_form_label,
    format_tooltip,
    set_status_card,
)
from napari_harpy.widgets.spatial_query.status_card import (
    _SpatialQueryStatusCardSpec,
    build_spatial_query_cache_status_card_spec,
    build_spatial_query_readiness_status_card_spec,
)
from napari_harpy.widgets.spatial_query.viewer_styling import load_and_style_spatial_annotation_labels

if TYPE_CHECKING:
    import napari
    from spatialdata import SpatialData

type _TargetColumnMode = Literal["existing", "new"]

_DEFAULT_NEW_COLUMN_NAME = "spatial_annotation"
_INPUT_CONTROL_STYLESHEET = build_input_control_stylesheet("QComboBox, QLineEdit")
_FIELD_MIN_WIDTH = 180


class SpatialQuery(QWidget):
    """Select Spatial Query inputs and emit validated action intents.

    This independently embeddable child receives coordinate-system and Shapes
    state only through ``apply_annotation_context()``. It owns no worker or
    table mutation path: Run and Recalculate merely emit parameterless intents
    for the future execution layer.
    """

    run_requested = Signal()
    recalculate_centers_requested = Signal()

    def __init__(self, napari_viewer: napari.Viewer | None = None) -> None:
        super().__init__()
        self.setObjectName("spatial_query_widget")

        self._app_state = get_or_create_app_state(napari_viewer)
        self._annotation_context = AnnotationContext(
            sdata=None,
            coordinate_system=None,
            shapes_target=None,
            has_unsaved_shapes_changes=False,
        )
        self._canonical_cache_report: CanonicalCacheReport | None = None
        self._canonical_cache_inspection_error: str | None = None
        self._layer_styling_error: str | None = None

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(10)

        form_layout = QFormLayout()
        form_layout.setContentsMargins(0, 0, 0, 0)
        form_layout.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        form_layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.FieldsStayAtSizeHint)
        form_layout.setHorizontalSpacing(12)
        form_layout.setVerticalSpacing(10)

        self.labels_combo = self._create_combo(
            "spatial_query_labels_combo",
            "Select a supported 2D labels element.",
        )
        self.table_combo = self._create_combo(
            "spatial_query_table_combo",
            "Select a table that annotates the labels element.",
        )
        self.column_mode_combo = self._create_combo(
            "spatial_query_column_mode_combo",
            "Choose whether to annotate an existing categorical column or create a new one.",
        )
        self.column_mode_combo.addItem("Existing column", "existing")
        self.column_mode_combo.addItem("New column", "new")

        self.existing_column_combo = self._create_combo(
            "spatial_query_existing_column_combo",
            "Select an existing categorical column with string categories.",
        )
        self.new_column_edit = QLineEdit(_DEFAULT_NEW_COLUMN_NAME)
        self.new_column_edit.setObjectName("spatial_query_new_column_edit")
        self.new_column_edit.setAccessibleName("New annotation column name")
        self.new_column_edit.setMinimumWidth(_FIELD_MIN_WIDTH)
        self.new_column_edit.setStyleSheet(_INPUT_CONTROL_STYLESHEET)
        self.new_column_edit.setToolTip(
            format_tooltip("Enter a new categorical annotation column. It is created only after an effective Apply.")
        )

        self.existing_column_label = create_form_label("Existing column")
        self.new_column_label = create_form_label("New column name")
        form_layout.addRow(create_form_label("Labels"), self.labels_combo)
        form_layout.addRow(create_form_label("Table"), self.table_combo)
        form_layout.addRow(create_form_label("Target mode"), self.column_mode_combo)
        form_layout.addRow(self.existing_column_label, self.existing_column_combo)
        form_layout.addRow(self.new_column_label, self.new_column_edit)
        root_layout.addLayout(form_layout)

        self.cache_status_label = QLabel()
        self.cache_status_label.setObjectName("spatial_query_cache_status_label")
        self.cache_status_label.setWordWrap(True)
        root_layout.addWidget(self.cache_status_label)

        button_layout = QHBoxLayout()
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.setSpacing(8)

        self.recalculate_centers_button = QPushButton("Recalculate centroids")
        self.recalculate_centers_button.setObjectName("spatial_query_recalculate_centers_button")
        self.recalculate_centers_button.setAccessibleName("Recalculate centroids")
        self.recalculate_centers_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.recalculate_centers_button.setStyleSheet(ACTION_BUTTON_STYLESHEET)
        self.recalculate_centers_button.setToolTip(
            format_tooltip("Force recalculation for the selected labels region, even when cached centers are valid.")
        )

        self.run_button = QPushButton("Run spatial query")
        self.run_button.setObjectName("spatial_query_run_button")
        self.run_button.setAccessibleName("Run spatial query")
        self.run_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.run_button.setStyleSheet(PRIMARY_BUTTON_STYLESHEET)
        self.run_button.setToolTip(
            format_tooltip("Evaluate canonical label centers against the selected saved Shapes geometry.")
        )

        button_layout.addWidget(self.recalculate_centers_button)
        button_layout.addWidget(self.run_button)
        root_layout.addLayout(button_layout)

        self.readiness_status_label = QLabel()
        self.readiness_status_label.setObjectName("spatial_query_readiness_status_label")
        self.readiness_status_label.setWordWrap(True)
        root_layout.addWidget(self.readiness_status_label)

        self.labels_combo.currentIndexChanged.connect(self._on_labels_changed)
        self.table_combo.currentIndexChanged.connect(self._on_table_changed)
        self.column_mode_combo.currentIndexChanged.connect(self._on_column_mode_changed)
        self.existing_column_combo.currentIndexChanged.connect(self._on_existing_column_changed)
        self.new_column_edit.textChanged.connect(self._on_new_column_changed)
        self.recalculate_centers_button.clicked.connect(self.recalculate_centers_requested.emit)
        self.run_button.clicked.connect(self.run_requested.emit)

        self._set_column_control_visibility()
        self._refresh_labels(preferred_labels=None)
        self._refresh_tables(preferred_table=None)
        self._refresh_columns(
            preferred_mode=None,
            preferred_existing_column=None,
            preferred_new_column=_DEFAULT_NEW_COLUMN_NAME,
        )
        self._refresh_controls_and_status()

    @property
    def app_state(self) -> HarpyAppState:
        """Return the shared per-viewer Harpy app state."""
        return self._app_state

    @property
    def annotation_context(self) -> AnnotationContext:
        """Return the latest parent-supplied Annotation context."""
        return self._annotation_context

    @property
    def selected_spatialdata(self) -> SpatialData | None:
        return self._annotation_context.sdata

    @property
    def selected_coordinate_system(self) -> str | None:
        return self._annotation_context.coordinate_system

    @property
    def selected_labels_name(self) -> str | None:
        value = self.labels_combo.currentData()
        return value if isinstance(value, str) else None

    @property
    def selected_table_name(self) -> str | None:
        value = self.table_combo.currentData()
        return value if isinstance(value, str) else None

    @property
    def selected_column_mode(self) -> _TargetColumnMode | None:
        value = self.column_mode_combo.currentData()
        return cast(_TargetColumnMode, value) if value in ("existing", "new") else None

    @property
    def selected_column_name(self) -> str | None:
        column_name, error, _description = self._resolve_target_intent()
        return column_name if error is None else None

    @property
    def cache_report(self) -> CanonicalCacheReport | None:
        """Return the report captured for the current labels/table selection."""
        return self._canonical_cache_report

    def apply_annotation_context(self, context: AnnotationContext) -> None:
        """Adopt parent-owned context and refresh dependent controls without styling."""
        if not isinstance(context, AnnotationContext):
            raise TypeError("Spatial Query requires an AnnotationContext.")

        preserve_selection = context.sdata is self._annotation_context.sdata
        preferred_labels = self.selected_labels_name if preserve_selection else None
        preferred_table = self.selected_table_name if preserve_selection else None
        preferred_mode = self.selected_column_mode if preserve_selection else None
        preferred_existing_column = self._selected_existing_column_name() if preserve_selection else None
        preferred_new_column = self.new_column_edit.text() if preserve_selection else _DEFAULT_NEW_COLUMN_NAME

        self._annotation_context = context
        self._layer_styling_error = None
        self._refresh_labels(preferred_labels)
        self._refresh_tables(preferred_table)
        self._refresh_columns(
            preferred_mode=preferred_mode,
            preferred_existing_column=preferred_existing_column,
            preferred_new_column=preferred_new_column,
        )
        self._inspect_canonical_centers_cache()
        self._refresh_controls_and_status()

    def _create_combo(self, object_name: str, tooltip: str) -> CompactComboBox:
        combo = CompactComboBox()
        combo.setObjectName(object_name)
        combo.setAccessibleName(object_name.replace("spatial_query_", "").replace("_", " "))
        combo.setMinimumWidth(_FIELD_MIN_WIDTH)
        combo.setStyleSheet(_INPUT_CONTROL_STYLESHEET)
        combo.setToolTip(format_tooltip(tooltip))
        return combo

    def _refresh_labels(self, preferred_labels: str | None) -> None:
        options = []
        sdata = self.selected_spatialdata
        coordinate_system = self.selected_coordinate_system
        if sdata is not None and coordinate_system is not None:
            candidates = get_spatialdata_labels_options_for_coordinate_system_from_sdata(
                sdata=sdata,
                coordinate_system=coordinate_system,
            )
            for option in candidates:
                try:
                    build_canonical_source_signature(sdata, option.labels_name)
                except (KeyError, TypeError, ValueError):
                    continue
                options.append(option)

        with QSignalBlocker(self.labels_combo):
            self.labels_combo.clear()
            for option in options:
                self.labels_combo.addItem(option.display_name, option.labels_name)
            self._select_combo_data(self.labels_combo, preferred_labels)
            self.labels_combo.setEnabled(bool(options))

    def _refresh_tables(self, preferred_table: str | None) -> None:
        table_names: list[str] = []
        sdata = self.selected_spatialdata
        labels_name = self.selected_labels_name
        if sdata is not None and labels_name is not None:
            table_names = get_annotating_table_names(sdata, labels_name)

        with QSignalBlocker(self.table_combo):
            self.table_combo.clear()
            for table_name in table_names:
                self.table_combo.addItem(table_name, table_name)
            self._select_combo_data(self.table_combo, preferred_table)
            self.table_combo.setEnabled(bool(table_names))

    def _refresh_columns(
        self,
        *,
        preferred_mode: _TargetColumnMode | None,
        preferred_existing_column: str | None,
        preferred_new_column: str,
    ) -> None:
        sdata = self.selected_spatialdata
        table_name = self.selected_table_name
        column_names = (
            []
            if sdata is None or table_name is None
            else get_compatible_spatial_annotation_column_names(sdata, table_name)
        )

        with QSignalBlocker(self.existing_column_combo):
            self.existing_column_combo.clear()
            for column_name in column_names:
                self.existing_column_combo.addItem(column_name, column_name)

        default_mode: _TargetColumnMode = "existing" if _DEFAULT_NEW_COLUMN_NAME in column_names else "new"
        next_mode = preferred_mode
        if next_mode == "existing" and preferred_existing_column not in column_names:
            next_mode = None
        if next_mode is None:
            next_mode = default_mode

        with QSignalBlocker(self.column_mode_combo):
            self.column_mode_combo.setCurrentIndex(self.column_mode_combo.findData(next_mode))
        with QSignalBlocker(self.existing_column_combo):
            preferred = preferred_existing_column if next_mode == "existing" else _DEFAULT_NEW_COLUMN_NAME
            self._select_combo_data(self.existing_column_combo, preferred)
        with QSignalBlocker(self.new_column_edit):
            self.new_column_edit.setText(preferred_new_column or _DEFAULT_NEW_COLUMN_NAME)

        self.column_mode_combo.setEnabled(table_name is not None)
        self.existing_column_combo.setEnabled(bool(column_names))
        self.new_column_edit.setEnabled(table_name is not None)
        self._set_column_control_visibility()

    def _inspect_canonical_centers_cache(self) -> None:
        self._canonical_cache_report = None
        self._canonical_cache_inspection_error = None
        sdata = self.selected_spatialdata
        labels_name = self.selected_labels_name
        table_name = self.selected_table_name
        if sdata is None or labels_name is None or table_name is None:
            return

        try:
            self._canonical_cache_report = inspect_canonical_cache(
                sdata,
                table_name=table_name,
                labels_name=labels_name,
            )
        except (KeyError, TypeError, ValueError) as error:
            self._canonical_cache_inspection_error = str(error)

    def _on_labels_changed(self, index: int) -> None:
        del index
        preferred_table = self.selected_table_name
        preferred_mode = self.selected_column_mode
        preferred_existing_column = self._selected_existing_column_name()
        preferred_new_column = self.new_column_edit.text()
        self._layer_styling_error = None
        self._refresh_tables(preferred_table)
        self._refresh_columns(
            preferred_mode=preferred_mode,
            preferred_existing_column=preferred_existing_column,
            preferred_new_column=preferred_new_column,
        )
        self._inspect_canonical_centers_cache()
        self._apply_explicit_labels_styling()
        self._refresh_controls_and_status()

    def _on_table_changed(self, index: int) -> None:
        del index
        self._layer_styling_error = None
        self._refresh_columns(
            preferred_mode=None,
            preferred_existing_column=None,
            preferred_new_column=_DEFAULT_NEW_COLUMN_NAME,
        )
        self._inspect_canonical_centers_cache()
        self._refresh_controls_and_status()

    def _on_column_mode_changed(self, index: int) -> None:
        del index
        self._layer_styling_error = None
        self._set_column_control_visibility()
        if self.selected_column_mode == "existing":
            self._apply_existing_column_styling()
        self._refresh_controls_and_status()

    def _on_existing_column_changed(self, index: int) -> None:
        del index
        self._layer_styling_error = None
        self._apply_existing_column_styling()
        self._refresh_controls_and_status()

    def _on_new_column_changed(self, text: str) -> None:
        del text
        self._refresh_controls_and_status()

    def _apply_explicit_labels_styling(self) -> None:
        if self.selected_column_mode == "existing" and self._selected_existing_column_name() is not None:
            self._apply_existing_column_styling()
            return

        sdata = self.selected_spatialdata
        coordinate_system = self.selected_coordinate_system
        labels_name = self.selected_labels_name
        if sdata is None or coordinate_system is None or labels_name is None:
            return
        try:
            result = self._app_state.viewer_adapter.ensure_labels_loaded(
                sdata,
                labels_name,
                coordinate_system,
            )
            self._app_state.viewer_adapter.activate_layer(result.layer)
        except (KeyError, TypeError, ValueError) as error:
            self._layer_styling_error = str(error)

    def _apply_existing_column_styling(self) -> None:
        sdata = self.selected_spatialdata
        coordinate_system = self.selected_coordinate_system
        labels_name = self.selected_labels_name
        table_name = self.selected_table_name
        column_name = self._selected_existing_column_name()
        if (
            sdata is None
            or coordinate_system is None
            or labels_name is None
            or table_name is None
            or column_name is None
        ):
            return
        try:
            load_and_style_spatial_annotation_labels(
                self._app_state.viewer_adapter,
                sdata=sdata,
                coordinate_system=coordinate_system,
                labels_name=labels_name,
                table_name=table_name,
                column_name=column_name,
            )
        except (KeyError, TypeError, ValueError) as error:
            self._layer_styling_error = str(error)

    def _resolve_target_intent(self) -> tuple[str | None, str | None, str | None]:
        mode = self.selected_column_mode
        sdata = self.selected_spatialdata
        table_name = self.selected_table_name
        if table_name is None or sdata is None:
            return None, "Choose a linked table before configuring the annotation target.", None
        if mode == "existing":
            column_name = self._selected_existing_column_name()
            if column_name is None:
                return None, "Choose a compatible existing categorical column.", None
            return column_name, None, f'Existing column "{column_name}"'
        if mode != "new":
            return None, "Choose Existing column or New column.", None

        try:
            column_name = normalize_spatialdata_dataframe_column_name(
                self.new_column_edit.text(),
                "Annotation column name",
            )
        except ValueError as error:
            return None, str(error), None

        table = sdata.tables[table_name]
        table_metadata = get_table_metadata(sdata, table_name)
        if column_name in (table_metadata.region_key, table_metadata.instance_key):
            return None, f'Annotation column "{column_name}" cannot be a table linkage column.', None
        if column_name in table.obs.columns:
            return None, f'New annotation column "{column_name}" already exists.', None
        return column_name, None, f'New column "{column_name}"'

    def _refresh_controls_and_status(self) -> None:
        column_name, target_error, target_description = self._resolve_target_intent()
        del column_name
        has_report = self._canonical_cache_report is not None
        self.recalculate_centers_button.setEnabled(has_report)
        self.run_button.setEnabled(
            has_report
            and self._annotation_context.saved_shapes_name is not None
            and not self._annotation_context.has_unsaved_shapes_changes
            and target_error is None
        )

        if self._canonical_cache_inspection_error is not None:
            cache_status_spec = _SpatialQueryStatusCardSpec(
                title="Centroid Inspection Error",
                lines=(self._canonical_cache_inspection_error,),
                kind="error",
            )
        else:
            cache_status_spec = build_spatial_query_cache_status_card_spec(self._canonical_cache_report)
        self._apply_status_card_spec(self.cache_status_label, cache_status_spec)
        self._apply_status_card_spec(
            self.readiness_status_label,
            build_spatial_query_readiness_status_card_spec(
                has_spatialdata=self.selected_spatialdata is not None,
                coordinate_system=self.selected_coordinate_system,
                saved_shapes_name=self._annotation_context.saved_shapes_name,
                has_unsaved_shapes_changes=self._annotation_context.has_unsaved_shapes_changes,
                labels_name=self.selected_labels_name,
                table_name=self.selected_table_name,
                has_cache_report=has_report,
                target_error=target_error,
                target_description=target_description,
                layer_styling_error=self._layer_styling_error,
            ),
        )

    def _set_column_control_visibility(self) -> None:
        existing = self.selected_column_mode == "existing"
        self.existing_column_label.setVisible(existing)
        self.existing_column_combo.setVisible(existing)
        self.new_column_label.setVisible(not existing)
        self.new_column_edit.setVisible(not existing)

    def _selected_existing_column_name(self) -> str | None:
        value = self.existing_column_combo.currentData()
        return value if isinstance(value, str) else None

    @staticmethod
    def _select_combo_data(combo: CompactComboBox, preferred: str | None) -> None:
        index = combo.findData(preferred) if preferred is not None else -1
        combo.setCurrentIndex(index if index >= 0 else (0 if combo.count() else -1))

    @staticmethod
    def _apply_status_card_spec(label: QLabel, spec: _SpatialQueryStatusCardSpec) -> None:
        set_status_card(
            label,
            title=spec.title,
            lines=list(spec.lines),
            kind=spec.kind,
            tooltip_message=spec.tooltip_message,
        )
