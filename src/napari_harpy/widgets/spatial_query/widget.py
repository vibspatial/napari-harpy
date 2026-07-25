"""Context-driven Spatial Query child widget shell."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

from qtpy.QtCore import QSignalBlocker, Qt
from qtpy.QtWidgets import QFormLayout, QLabel, QLineEdit, QPushButton, QVBoxLayout, QWidget

from napari_harpy._app_state import (
    HarpyAppState,
    TableChangeKind,
    TableStateChangedEvent,
    get_or_create_app_state,
)
from napari_harpy.core.object_classification.annotation import USER_CLASS_COLUMN
from napari_harpy.core.object_classification.classifier import (
    PRED_CLASS_COLUMN,
    PRED_CONFIDENCE_COLUMN,
)
from napari_harpy.core.spatial_query import (
    CANONICAL_CACHE_PATHS,
    CANONICAL_OBSM_KEY,
    SPATIAL_COORDINATES_KEY,
    CanonicalCacheReport,
    CanonicalCacheUpdateAction,
    CanonicalCenterQueryResult,
    CanonicalCentersResult,
    build_canonical_source_signature,
    get_compatible_spatial_annotation_column_names,
    inspect_canonical_cache,
    parse_canonical_metadata,
    require_compatible_spatial_annotation_column,
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
    CompactComboBox,
    build_input_control_stylesheet,
    create_form_label,
    format_tooltip,
    set_status_card,
)
from napari_harpy.widgets.spatial_query.controller import SpatialQueryController
from napari_harpy.widgets.spatial_query.status_card import (
    _SpatialQueryStatusCardSpec,
    build_spatial_query_execution_status_card_spec,
    build_spatial_query_status_card_spec,
)
from napari_harpy.widgets.spatial_query.viewer_styling import (
    load_and_style_spatial_annotation_labels,
    load_and_style_unannotated_spatial_annotation_labels,
)

if TYPE_CHECKING:
    import napari
    from spatialdata import SpatialData

type _TargetColumnMode = Literal["existing", "new"]


@dataclass(frozen=True)
class _AnnotationTargetResolution:
    mode: _TargetColumnMode | None
    column_name: str | None
    error: str | None

    @property
    def is_ready(self) -> bool:
        return self.error is None

    @property
    def description(self) -> str | None:
        if self.mode is None or self.column_name is None:
            return None

        label = "Existing" if self.mode == "existing" else "New"
        return f'{label} column "{self.column_name}"'


@dataclass(frozen=True)
class _SpatialQueryRunIntent:
    """Immutable UI and domain inputs accepted by one Run action."""

    sdata: SpatialData
    shapes_name: str
    coordinate_system: str
    labels_name: str
    table_name: str
    column_mode: _TargetColumnMode
    column_name: str


_PREFERRED_ANNOTATION_COLUMN_NAME = "spatial_annotation"
_OBJECT_CLASSIFICATION_COLUMNS = frozenset((USER_CLASS_COLUMN, PRED_CLASS_COLUMN, PRED_CONFIDENCE_COLUMN))
_INPUT_CONTROL_STYLESHEET = build_input_control_stylesheet("QComboBox, QLineEdit")
_FIELD_MIN_WIDTH = 180
CANONICAL_CACHE_UPDATE_SOURCE = "spatial_query_canonical_centers"
_CACHE_UPDATE_CHANGE_KINDS: dict[CanonicalCacheUpdateAction, TableChangeKind] = {
    CanonicalCacheUpdateAction.CREATE: "created",
    CanonicalCacheUpdateAction.EXTEND: "updated",
    CanonicalCacheUpdateAction.REFRESH: "updated",
    CanonicalCacheUpdateAction.REBUILD: "rebuilt",
}


class SpatialQuery(QWidget):
    """Query a Labels element against the polygons of a saved Shapes element.

    The widget finds Labels instances whose canonical centers are contained by
    the selected Shapes geometry. It selects the relevant SpatialData, Shapes,
    Labels, table, and annotation-column inputs and orchestrates the
    mutation-free center-calculation and containment-query work.

    As a child of ``AnnotationWidget``, this independently embeddable widget
    receives coordinate-system and Shapes state only through
    ``apply_annotation_context()``. Run captures one immutable intent and
    delegates sequential center calculation and containment work to its
    controller. The child owns shared cache-event publication; annotation
    review and Apply are added by the next slice.

    CanonicalCacheReport lifecycle
    ------------------------------
    configuration-time CanonicalCacheReport
    (Labels selection change, table selection change, or selected-table reload)
        → produced by ``_inspect_canonical_centers_cache()``
        → stored on the widget for status-card information
        → not trusted to start later work

    Run-time CanonicalCacheReport
        → produced by ``_run_spatial_query()``
        → authoritative operation starting state
        → passed directly to ``SpatialQueryController.start_spatial_query()``
        → performed for every accepted Run

    post-calculation CanonicalCacheReport
        → produced inside ``apply_canonical_cache_update()``
        → only when centers were calculated
        → verifies that the source and table binding did not change
        → guards cache installation
    """

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
        self._canonical_input_inspection_error: str | None = None
        self._layer_styling_error: str | None = None
        self._active_run_intent: _SpatialQueryRunIntent | None = None
        self._show_execution_status = False
        self._controller = SpatialQueryController(
            on_state_changed=self._on_controller_state_changed,
            on_centers_ready=self._on_centers_ready,
            on_query_ready=self._on_query_ready,
        )

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
        self.labels_combo.setPlaceholderText("Choose a labels element")
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
            "Select an existing categorical column with string or positive-integer categories.",
        )
        self.existing_column_combo.setPlaceholderText("Choose an existing column")
        self.new_column_edit = QLineEdit()
        self.new_column_edit.setObjectName("spatial_query_new_column_edit")
        self.new_column_edit.setAccessibleName("New annotation column name")
        # Suggest the conventional name without turning it into an implicit
        # schema target; only text entered by the user is validated or applied.
        self.new_column_edit.setPlaceholderText(_PREFERRED_ANNOTATION_COLUMN_NAME)
        self.new_column_edit.setMinimumWidth(_FIELD_MIN_WIDTH)
        self.new_column_edit.setStyleSheet(_INPUT_CONTROL_STYLESHEET)
        self.new_column_edit.setToolTip(
            format_tooltip(
                "Enter a new string-categorical annotation column. It is created only after an effective Apply."
            )
        )

        self.existing_column_label = create_form_label("Existing column")
        self.new_column_label = create_form_label("New column name")
        form_layout.addRow(create_form_label("Labels"), self.labels_combo)
        form_layout.addRow(create_form_label("Table"), self.table_combo)
        form_layout.addRow(create_form_label("Target mode"), self.column_mode_combo)
        form_layout.addRow(self.existing_column_label, self.existing_column_combo)
        form_layout.addRow(self.new_column_label, self.new_column_edit)
        root_layout.addLayout(form_layout)

        self.status_label = QLabel()
        self.status_label.setObjectName("spatial_query_status_label")
        self.status_label.setWordWrap(True)
        root_layout.addWidget(self.status_label)

        self.run_button = QPushButton("Run spatial query")
        self.run_button.setObjectName("spatial_query_run_button")
        self.run_button.setAccessibleName("Run spatial query")
        self.run_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.run_button.setStyleSheet(ACTION_BUTTON_STYLESHEET)
        self.run_button.setToolTip(
            format_tooltip("Evaluate canonical label centers against the selected saved Shapes geometry.")
        )

        root_layout.addWidget(self.run_button)

        self.labels_combo.currentIndexChanged.connect(self._on_labels_changed)
        self.table_combo.currentIndexChanged.connect(self._on_table_changed)
        self.column_mode_combo.currentIndexChanged.connect(self._on_column_mode_changed)
        self.existing_column_combo.currentIndexChanged.connect(self._on_existing_column_changed)
        self.new_column_edit.textChanged.connect(self._on_new_column_changed)
        self.run_button.clicked.connect(self._run_spatial_query)
        self._app_state.viewer_adapter.primary_labels_layers_changed.connect(self._on_primary_labels_layers_changed)
        self._app_state.table_state_changed.connect(self._on_table_state_changed)
        self.destroyed.connect(self._controller.shutdown)

        self._set_column_control_visibility()
        self._refresh_labels()
        self._refresh_tables(preferred_table=None)
        self._refresh_columns(
            preferred_mode=None,
            preferred_existing_column=None,
            preferred_new_column="",
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
        resolution = self._resolve_annotation_target()
        return resolution.column_name if resolution.is_ready else None

    @property
    def cache_report(self) -> CanonicalCacheReport | None:
        """Return the report captured for the current labels/table selection."""
        return self._canonical_cache_report

    def apply_annotation_context(self, context: AnnotationContext) -> None:
        """Adopt parent context, refreshing selectors only when their dependencies change.

        SpatialData identity and coordinate-system changes clear the labels
        selection and its complete dependent state. Shapes-target and
        dirty-state changes affect Run readiness only and therefore reuse the
        current selections and report.
        """
        if not isinstance(context, AnnotationContext):
            raise TypeError("Spatial Query requires an AnnotationContext.")

        previous_context = self._annotation_context
        context_changed = (
            context.sdata is not previous_context.sdata
            or context.coordinate_system != previous_context.coordinate_system
            or context.shapes_target != previous_context.shapes_target
            or context.has_unsaved_shapes_changes != previous_context.has_unsaved_shapes_changes
        )
        selection_dependencies_changed = (
            context.sdata is not previous_context.sdata
            or context.coordinate_system != previous_context.coordinate_system
        )
        if context_changed:
            self._invalidate_run()
        self._annotation_context = context

        if selection_dependencies_changed:
            # Labels choices belong to one SpatialData and coordinate system.
            # Rebuild them and clear every dependent table/column selection
            # instead of carrying those choices into another display context.
            self._refresh_labels()
            self._clear_labels_selection_and_dependents()
            self._refresh_controls_and_status()
            return

        # Changes to the selected Shapes target or its dirty state may enable
        # or disable Run, but they do not affect the selected Labels, table, or
        # captured cache report. Refresh only the controls and status to avoid
        # recalculating the table instance-set digest.
        self._refresh_controls_and_status()

    def _create_combo(self, object_name: str, tooltip: str) -> CompactComboBox:
        combo = CompactComboBox()
        combo.setObjectName(object_name)
        combo.setAccessibleName(object_name.replace("spatial_query_", "").replace("_", " "))
        combo.setMinimumWidth(_FIELD_MIN_WIDTH)
        combo.setStyleSheet(_INPUT_CONTROL_STYLESHEET)
        combo.setToolTip(format_tooltip(tooltip))
        return combo

    def _refresh_labels(self) -> None:
        options = []
        sdata = self.selected_spatialdata
        coordinate_system = self.selected_coordinate_system
        if sdata is not None and coordinate_system is not None:
            candidates = get_spatialdata_labels_options_for_coordinate_system_from_sdata(
                sdata=sdata,
                coordinate_system=coordinate_system,
            )
            for option in candidates:
                # Expose only labels sources accepted by the canonical-centers
                # contract; this currently excludes 3D labels.
                try:
                    build_canonical_source_signature(sdata, option.labels_name)
                except (KeyError, TypeError, ValueError):
                    continue
                options.append(option)

        with QSignalBlocker(self.labels_combo):
            self.labels_combo.clear()
            for option in options:
                self.labels_combo.addItem(option.display_name, option.labels_name)
            self.labels_combo.setCurrentIndex(-1)
            self.labels_combo.setEnabled(bool(options))

    def _clear_labels_selection_and_dependents(self) -> None:
        """Clear labels-owned UI state without inspecting or mutating the cache."""
        with QSignalBlocker(self.labels_combo):
            self.labels_combo.setCurrentIndex(-1)
        self._refresh_tables(preferred_table=None)
        self._refresh_columns(
            preferred_mode=None,
            preferred_existing_column=None,
            preferred_new_column="",
        )
        self._canonical_cache_report = None
        self._canonical_input_inspection_error = None
        self._layer_styling_error = None

    def _on_primary_labels_layers_changed(self) -> None:
        """Clear the selection when its corresponding primary layer disappears."""
        sdata = self.selected_spatialdata
        coordinate_system = self.selected_coordinate_system
        labels_name = self.selected_labels_name
        if sdata is None or coordinate_system is None or labels_name is None:
            return

        loaded_layer = self._app_state.viewer_adapter.get_loaded_primary_labels_layer(
            sdata,
            labels_name,
            coordinate_system,
        )
        if loaded_layer is not None:
            return

        self._invalidate_run()
        self._clear_labels_selection_and_dependents()
        self._refresh_controls_and_status()

    def _on_table_state_changed(self, value: object) -> None:
        """Invalidate work when its table components are replaced from storage."""
        if not isinstance(value, TableStateChangedEvent):
            return
        # This consumer owns reload invalidation only. A reload replaces table
        # components from storage and may invalidate the captured Run inputs.
        # Ordinary mutation events, including the canonical-cache update event
        # emitted by the current Spatial Query Run, follow their own provenance
        # contracts.
        if value.change_kind != "reloaded":
            return
        if value.sdata is not self.selected_spatialdata or value.table_name != self.selected_table_name:
            return

        self._invalidate_run()
        self._inspect_canonical_centers_cache()
        self._refresh_controls_and_status()

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

        preferred_conventional_column = (
            _PREFERRED_ANNOTATION_COLUMN_NAME if _PREFERRED_ANNOTATION_COLUMN_NAME in column_names else None
        )
        if preferred_mode == "existing" and preferred_existing_column in column_names:
            next_mode: _TargetColumnMode = "existing"
            next_existing_column = preferred_existing_column
        elif preferred_mode != "new" and preferred_conventional_column is not None:
            next_mode = "existing"
            next_existing_column = preferred_conventional_column
        else:
            next_mode = "new"
            next_existing_column = None

        with QSignalBlocker(self.column_mode_combo):
            self.column_mode_combo.setCurrentIndex(self.column_mode_combo.findData(next_mode))
        with QSignalBlocker(self.existing_column_combo):
            next_existing_index = (
                self.existing_column_combo.findData(next_existing_column) if next_existing_column is not None else -1
            )
            self.existing_column_combo.setCurrentIndex(next_existing_index)
        with QSignalBlocker(self.new_column_edit):
            self.new_column_edit.setText(preferred_new_column)

        self.column_mode_combo.setEnabled(table_name is not None)
        self.existing_column_combo.setEnabled(bool(column_names))
        self.new_column_edit.setEnabled(table_name is not None)
        self._set_column_control_visibility()

    def _inspect_canonical_centers_cache(self) -> None:
        self._canonical_cache_report = None
        self._canonical_input_inspection_error = None
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
            self._canonical_input_inspection_error = str(error)

    def _on_labels_changed(self, index: int) -> None:
        del index
        self._invalidate_run()
        previous_table = self.selected_table_name
        previous_mode = self.selected_column_mode
        previous_existing_column = self._selected_existing_column_name()
        previous_new_column = self.new_column_edit.text()
        self._layer_styling_error = None
        self._refresh_tables(previous_table)
        same_table_context = previous_table is not None and self.selected_table_name == previous_table
        # A New-column draft belongs to its table. Preserve it only when the
        # refreshed labels selection still resolves to that exact table.
        self._refresh_columns(
            preferred_mode=previous_mode if same_table_context else None,
            preferred_existing_column=previous_existing_column if same_table_context else None,
            preferred_new_column=previous_new_column if same_table_context else "",
        )
        self._inspect_canonical_centers_cache()
        self._apply_explicit_labels_styling()
        self._refresh_controls_and_status()

    def _on_table_changed(self, index: int) -> None:
        del index
        self._invalidate_run()
        self._layer_styling_error = None
        self._refresh_columns(
            preferred_mode=None,
            preferred_existing_column=None,
            preferred_new_column="",
        )
        self._inspect_canonical_centers_cache()
        # This callback represents an explicit user table choice. Programmatic
        # table refreshes block the combo signal and therefore do not reclaim
        # the shared primary layer's presentation.
        self._apply_explicit_labels_styling()
        self._refresh_controls_and_status()

    def _on_column_mode_changed(self, index: int) -> None:
        del index
        self._invalidate_run()
        self._layer_styling_error = None
        self._set_column_control_visibility()
        if self.selected_column_mode == "new":
            with QSignalBlocker(self.existing_column_combo):
                self.existing_column_combo.setCurrentIndex(-1)
        self._apply_explicit_labels_styling()
        self._refresh_controls_and_status()

    def _on_existing_column_changed(self, index: int) -> None:
        del index
        self._invalidate_run()
        self._layer_styling_error = None
        self._apply_explicit_labels_styling()
        self._refresh_controls_and_status()

    def _on_new_column_changed(self, text: str) -> None:
        del text
        self._invalidate_run()
        self._refresh_controls_and_status()

    def _run_spatial_query(self) -> None:
        """Start Spatial Query from the current UI selections.

        Validate the current UI selections
            ↓
        perform a fresh canonical-cache inspection
            ↓
        capture the accepted selections in a frozen Run-intent record
            ↓
        ask the controller to start the query
        """
        if self._controller.is_running:
            return

        sdata = self.selected_spatialdata
        shapes_name = self._annotation_context.saved_shapes_name
        coordinate_system = self.selected_coordinate_system
        labels_name = self.selected_labels_name
        table_name = self.selected_table_name
        target = self._resolve_annotation_target()
        if (
            sdata is None
            or shapes_name is None
            or coordinate_system is None
            or labels_name is None
            or table_name is None
            or self._annotation_context.has_unsaved_shapes_changes
            or not target.is_ready
            or target.mode is None
            or target.column_name is None
        ):
            self._refresh_controls_and_status()
            return

        # The cache or table binding may have changed since the configuration
        # report was captured. Do not start work from that display report;
        # inspect the current state again for this Run.
        try:
            report = inspect_canonical_cache(
                sdata,
                table_name=table_name,
                labels_name=labels_name,
            )
        except (KeyError, TypeError, ValueError) as error:
            self._canonical_cache_report = None
            self._canonical_input_inspection_error = str(error)
            self._show_execution_status = False
            self._refresh_controls_and_status()
            return

        self._canonical_cache_report = report
        self._canonical_input_inspection_error = None
        self._active_run_intent = _SpatialQueryRunIntent(
            sdata=sdata,
            shapes_name=shapes_name,
            coordinate_system=coordinate_system,
            labels_name=labels_name,
            table_name=table_name,
            column_mode=target.mode,
            column_name=target.column_name,
        )
        self._show_execution_status = True
        operation_accepted = self._controller.start_spatial_query(
            sdata,
            report,
            shapes_name=shapes_name,
            coordinate_system=coordinate_system,
        )
        if operation_accepted:
            return

        self._active_run_intent = None
        self._show_execution_status = False
        self._refresh_controls_and_status()

    def _on_centers_ready(self, result: CanonicalCentersResult) -> None:
        """Publish an accepted cache mutation before containment review."""
        intent = self._active_run_intent
        if intent is None or not self._run_intent_matches_current_selection(intent):
            self._invalidate_run()
            return

        cache_update = result.cache_update
        if cache_update is None:
            # Reused centers did not mutate the canonical cache, so there is
            # no table-state change to publish or mark as dirty.
            return
        event = TableStateChangedEvent(
            sdata=intent.sdata,
            table_name=result.table_name,
            paths=CANONICAL_CACHE_PATHS,
            regions=(result.labels_name,),
            change_kind=_CACHE_UPDATE_CHANGE_KINDS[cache_update.action],
            source=CANONICAL_CACHE_UPDATE_SOURCE,
        )
        self._app_state.record_table_mutation(event)

        # Keep the configuration-time status snapshot in line with the cache
        # just installed without repeating the instance-set digest. Every
        # future Run still performs its own fresh authoritative inspection.
        stored_metadata = parse_canonical_metadata(
            intent.sdata.tables[result.table_name].uns[SPATIAL_COORDINATES_KEY][CANONICAL_OBSM_KEY]
        )
        self._canonical_cache_report = CanonicalCacheReport(
            stored_metadata=stored_metadata,
            source_signature=result.source_signature,
            binding=result.binding,
        )

    def _on_query_ready(self, result: CanonicalCenterQueryResult) -> None:
        """Accept a non-empty current query result without annotating the table."""
        intent = self._active_run_intent
        if intent is None or not self._run_intent_matches_current_selection(intent):
            self._invalidate_run()
            return
        # Slice 7a deliberately ends here. The retained controller result and
        # captured target intent become the review inputs in the next slice.

    def _run_intent_matches_current_selection(self, intent: _SpatialQueryRunIntent) -> bool:
        target = self._resolve_annotation_target()
        return (
            self.selected_spatialdata is intent.sdata
            and self.selected_coordinate_system == intent.coordinate_system
            and self._annotation_context.saved_shapes_name == intent.shapes_name
            and not self._annotation_context.has_unsaved_shapes_changes
            and self.selected_labels_name == intent.labels_name
            and self.selected_table_name == intent.table_name
            and target.mode == intent.column_mode
            and target.column_name == intent.column_name
            and target.is_ready
        )

    def _invalidate_run(self) -> None:
        """Invalidate the accepted Run and return to configuration status.

        This deliberately hides the cancelled Run's execution outcome: the
        status card should describe the newly selected inputs instead. Active
        cancellation refreshes the widget through the controller callback.
        When no controller operation is active, the caller must refresh the
        controls and status after applying its configuration change.
        """
        self._active_run_intent = None
        self._show_execution_status = False
        self._controller.cancel_active_operation()

    def _on_controller_state_changed(self) -> None:
        if not self._controller.is_running:
            self._active_run_intent = None
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
            load_and_style_unannotated_spatial_annotation_labels(
                self._app_state.viewer_adapter,
                sdata=sdata,
                coordinate_system=coordinate_system,
                labels_name=labels_name,
            )
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

    def _resolve_annotation_target(self) -> _AnnotationTargetResolution:
        """Resolve the selected Existing or New annotation target.

        For New mode with an empty draft:

        ```text
        empty draft
            ↓
        preferred "spatial_annotation" column exists?
            ├── no
            │      → request a New-column name
            │
            └── yes
                   ↓
               compatible?
                   ├── yes
                   │      → request a New-column name because the user
                   │        explicitly chose New mode
                   │
                   └── no
                          → explain why the conventional column was excluded
                          → request another Existing column or a different
                            New-column name
        ```

        Placeholder text is visual guidance only and never becomes a target
        value.
        """
        mode = self.selected_column_mode
        sdata = self.selected_spatialdata
        table_name = self.selected_table_name
        if table_name is None or sdata is None:
            return _AnnotationTargetResolution(
                mode=None,
                column_name=None,
                error="Choose a linked table before configuring the annotation target.",
            )
        if mode == "existing":
            column_name = self._selected_existing_column_name()
            if column_name is None:
                return _AnnotationTargetResolution(
                    mode=None,
                    column_name=None,
                    error="Choose a compatible existing categorical column.",
                )
            return _AnnotationTargetResolution(
                mode=mode,
                column_name=column_name,
                error=None,
            )
        if mode != "new":
            return _AnnotationTargetResolution(
                mode=None,
                column_name=None,
                error="Choose Existing column or New column.",
            )

        table = sdata.tables[table_name]
        new_column_draft = self.new_column_edit.text()
        if not new_column_draft.strip():
            # Placeholder text is not a target value. Still explain when the
            # conventional column exists but was excluded as incompatible.
            if _PREFERRED_ANNOTATION_COLUMN_NAME in table.obs.columns:
                try:
                    require_compatible_spatial_annotation_column(
                        table,
                        _PREFERRED_ANNOTATION_COLUMN_NAME,
                    )
                except ValueError:
                    return _AnnotationTargetResolution(
                        mode=None,
                        column_name=None,
                        error=(
                            f'Existing annotation column "{_PREFERRED_ANNOTATION_COLUMN_NAME}" cannot be used because '
                            "Spatial Query requires a categorical column containing only strings or positive "
                            "integers. Choose another compatible Existing column or enter a different New-column name."
                        ),
                    )
            return _AnnotationTargetResolution(
                mode=None,
                column_name=None,
                error="Enter a new annotation column name.",
            )

        try:
            column_name = normalize_spatialdata_dataframe_column_name(
                new_column_draft,
                "Annotation column name",
            )
        except ValueError as error:
            return _AnnotationTargetResolution(
                mode=None,
                column_name=None,
                error=str(error),
            )

        table_metadata = get_table_metadata(sdata, table_name)
        if column_name in (table_metadata.region_key, table_metadata.instance_key):
            return _AnnotationTargetResolution(
                mode=None,
                column_name=None,
                error=f'Annotation column "{column_name}" cannot be a table linkage column.',
            )
        if column_name in _OBJECT_CLASSIFICATION_COLUMNS:
            return _AnnotationTargetResolution(
                mode=None,
                column_name=None,
                error=(
                    f'New annotation column "{column_name}" is reserved for Object Classification and cannot be '
                    "created by Spatial Query."
                ),
            )
        if column_name in table.obs.columns:
            try:
                require_compatible_spatial_annotation_column(table, column_name)
            except ValueError:
                return _AnnotationTargetResolution(
                    mode=None,
                    column_name=None,
                    error=(
                        f'Annotation column "{column_name}" already exists but cannot be used because Spatial Query '
                        "requires a categorical column containing only strings or positive integers. Choose another "
                        "compatible Existing column or enter a different New-column name."
                    ),
                )
            return _AnnotationTargetResolution(
                mode=None,
                column_name=None,
                error=f'New annotation column "{column_name}" already exists.',
            )
        return _AnnotationTargetResolution(
            mode=mode,
            column_name=column_name,
            error=None,
        )

    def _refresh_controls_and_status(self) -> None:
        """Refresh Run readiness and status without reinspecting the cache."""
        if (
            self.selected_labels_name is not None
            and self.selected_table_name is not None
            and self._canonical_cache_report is None
            and self._canonical_input_inspection_error is None
        ):
            raise RuntimeError(
                "Canonical cache inspection produced neither a report nor an error "
                "for the complete Spatial Query selection."
            )

        target_resolution = self._resolve_annotation_target()
        has_report = self._canonical_cache_report is not None
        self.run_button.setEnabled(
            has_report
            and self._canonical_input_inspection_error is None
            and self._annotation_context.saved_shapes_name is not None
            and not self._annotation_context.has_unsaved_shapes_changes
            and target_resolution.is_ready
            and not self._controller.is_running
        )

        if self._show_execution_status:
            self._apply_status_card_spec(
                self.status_label,
                build_spatial_query_execution_status_card_spec(
                    message=self._controller.status_message,
                    kind=self._controller.status_kind,
                    is_running=self._controller.is_running,
                ),
            )
            return

        self._apply_status_card_spec(
            self.status_label,
            build_spatial_query_status_card_spec(
                has_spatialdata=self.selected_spatialdata is not None,
                coordinate_system=self.selected_coordinate_system,
                saved_shapes_name=self._annotation_context.saved_shapes_name,
                has_unsaved_shapes_changes=self._annotation_context.has_unsaved_shapes_changes,
                labels_name=self.selected_labels_name,
                table_name=self.selected_table_name,
                cache_report=self._canonical_cache_report,
                canonical_input_inspection_error=self._canonical_input_inspection_error,
                target_error=target_resolution.error,
                target_description=target_resolution.description,
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
