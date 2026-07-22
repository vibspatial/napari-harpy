"""Parent Annotation widget coordinating shared selection and Shapes editing."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from qtpy.QtCore import QSignalBlocker, Qt, Signal
from qtpy.QtGui import QPixmap
from qtpy.QtWidgets import QFormLayout, QFrame, QLabel, QScrollArea, QVBoxLayout, QWidget

from napari_harpy._app_state import (
    CoordinateSystemChangedEvent,
    CoordinateSystemChangeRequest,
    HarpyAppState,
    get_or_create_app_state,
)
from napari_harpy._resources import get_logo_path
from napari_harpy.core.spatialdata import (
    get_coordinate_system_names_from_sdata,
    get_spatialdata_shapes_options_for_coordinate_system_from_sdata,
)
from napari_harpy.widgets.annotation.models import AnnotationContext, ShapesAnnotationTarget
from napari_harpy.widgets.shapes_annotation.widget import ShapesAnnotation
from napari_harpy.widgets.shared_styles import (
    WIDGET_MIN_WIDTH,
    WIDGET_TEXT_COLOR,
    CompactComboBox,
    apply_scroll_content_surface,
    apply_widget_surface,
    build_input_control_stylesheet,
    create_form_label,
    format_feedback_identifier,
    format_tooltip,
)

if TYPE_CHECKING:
    import napari
    from spatialdata import SpatialData


_SOURCE = "annotation_widget"
_INPUT_CONTROL_STYLESHEET = build_input_control_stylesheet("QComboBox")
_ANNOTATION_FIELD_MIN_WIDTH = 180
_STATUS_IDENTIFIER_MAX_LENGTH = 32
_CREATE_SHAPES_OPTION_TEXT = "Create shapes..."


class AnnotationWidget(QWidget):
    """Own shared Annotation selection and coordinate the Shapes child.

    Parent-to-child coordination intentionally uses direct method calls, while
    child-to-parent communication uses Qt signals. The parent owns and knows
    its child, and its commands require synchronous completion, return values,
    exception propagation, or signal blocking. The child does not know its
    concrete parent; it reports events through signals so it remains reusable
    and does not acquire a dependency on ``AnnotationWidget``.

    Child → parent signals
    ----------------------
    ShapesAnnotation.edit_session_dirty_changed(dirty)
        → AnnotationWidget._on_child_dirty_state_changed()
        → publish updated AnnotationContext

    ShapesAnnotation.shapes_target_change_requested(target)
        → AnnotationWidget._on_child_shapes_target_change_requested()
        → close the current session if allowed
        → select the requested target in the parent
        → apply and publish the resulting AnnotationContext

    ShapesAnnotation.edit_session_saved(target)
        → AnnotationWidget._on_child_edit_session_saved()
        → refresh the parent Shapes choices
        → select the newly saved element as an edit-existing target
        → apply and publish the resulting AnnotationContext

    Parent → child calls
    --------------------
    AnnotationWidget
        → ShapesAnnotation.try_close_edit_session()
          before committing an incompatible coordinate system or Shapes target
        → ShapesAnnotation.apply_annotation_context(context)
          after the parent has accepted the selection and built its AnnotationContext

    Parent → other child widgets and observers
    ------------------------------------------
    AnnotationWidget.annotation_context_changed(context)
        → publishes the final shared context
        → allows other child widgets and observers to react to context changes
    """

    annotation_context_changed = Signal(object)

    def __init__(self, napari_viewer: napari.Viewer | None = None) -> None:
        super().__init__()
        self.setObjectName("annotation_widget")
        apply_widget_surface(self)
        self.setMinimumWidth(WIDGET_MIN_WIDTH)

        self._app_state = get_or_create_app_state(napari_viewer)
        self._annotation_context = AnnotationContext(
            sdata=self._app_state.sdata,
            coordinate_system=None,
            shapes_target=None,
            has_unsaved_shapes_changes=False,
        )

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        self.scroll_area = QScrollArea()
        self.scroll_area.setObjectName("annotation_scroll_area")
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll_area.setStyleSheet("QScrollArea { border: 0px; background: transparent; }")

        self.scroll_content = QWidget()
        self.scroll_content.setObjectName("annotation_scroll_content")
        apply_scroll_content_surface(self.scroll_content)
        self.content_layout = QVBoxLayout(self.scroll_content)
        self.content_layout.setContentsMargins(12, 12, 12, 12)
        self.content_layout.setSpacing(10)

        self.content_layout.addWidget(self._create_header_logo())

        selector_layout = QFormLayout()
        selector_layout.setContentsMargins(0, 0, 0, 0)
        selector_layout.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        selector_layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.FieldsStayAtSizeHint)
        selector_layout.setHorizontalSpacing(12)
        selector_layout.setVerticalSpacing(10)

        self.coordinate_system_combo = CompactComboBox()
        self.coordinate_system_combo.setObjectName("annotation_coordinate_system_combo")
        self.coordinate_system_combo.setStyleSheet(_INPUT_CONTROL_STYLESHEET)
        self.coordinate_system_combo.setMinimumWidth(_ANNOTATION_FIELD_MIN_WIDTH)

        self.shapes_combo = CompactComboBox()
        self.shapes_combo.setObjectName("annotation_shapes_combo")
        self.shapes_combo.setStyleSheet(_INPUT_CONTROL_STYLESHEET)
        self.shapes_combo.setMinimumWidth(_ANNOTATION_FIELD_MIN_WIDTH)

        selector_layout.addRow(create_form_label("Coordinate System"), self.coordinate_system_combo)
        selector_layout.addRow(create_form_label("Shapes"), self.shapes_combo)
        self.content_layout.addLayout(selector_layout)

        self.shapes_annotation = ShapesAnnotation(napari_viewer)
        self.content_layout.addWidget(self.shapes_annotation)
        self.content_layout.addStretch(1)

        self.scroll_area.setWidget(self.scroll_content)
        root_layout.addWidget(self.scroll_area)

        self._app_state.sdata_changed.connect(self._on_sdata_changed)
        self._app_state.coordinate_system_changed.connect(self._on_app_state_coordinate_system_changed)
        self.coordinate_system_combo.currentIndexChanged.connect(self._on_coordinate_system_changed)
        self.shapes_combo.currentIndexChanged.connect(self._on_shapes_target_changed)
        self.shapes_annotation.edit_session_dirty_changed.connect(self._on_child_dirty_state_changed)
        self.shapes_annotation.shapes_target_change_requested.connect(self._on_child_shapes_target_change_requested)
        self.shapes_annotation.edit_session_saved.connect(self._on_child_edit_session_saved)

        self.refresh_from_sdata(self._app_state.sdata)

        self._app_state.register_coordinate_system_change_participant(self)
        app_state = self._app_state
        participant = self
        self.destroyed.connect(
            lambda *_args, app_state=app_state, participant=participant: (
                app_state.unregister_coordinate_system_change_participant(participant)
            )
        )

    @property
    def app_state(self) -> HarpyAppState:
        """Return the shared per-viewer Harpy app state."""
        return self._app_state

    @property
    def annotation_context(self) -> AnnotationContext:
        """Return the last parent-committed context supplied to children."""
        return self._annotation_context

    def refresh_from_sdata(self, sdata: SpatialData | None) -> None:
        """Refresh parent-owned selectors and publish one resulting context."""
        with QSignalBlocker(self.coordinate_system_combo):
            self.coordinate_system_combo.clear()
            coordinate_systems = [] if sdata is None else get_coordinate_system_names_from_sdata(sdata)
            for coordinate_system in coordinate_systems:
                self.coordinate_system_combo.addItem(coordinate_system, coordinate_system)
            self.coordinate_system_combo.setEnabled(bool(coordinate_systems))

        self._sync_coordinate_system_combo_selection(self._app_state.coordinate_system)
        self._refresh_shapes_targets()
        self._apply_and_publish_context()

    def _on_sdata_changed(self, sdata: SpatialData | None) -> None:
        self.refresh_from_sdata(sdata)

    def _on_app_state_coordinate_system_changed(self, event: CoordinateSystemChangedEvent) -> None:
        del event
        self._sync_coordinate_system_combo_selection(self._app_state.coordinate_system)
        self._refresh_shapes_targets()
        self._apply_and_publish_context()

    def _on_coordinate_system_changed(self, index: int) -> None:
        coordinate_system = self.coordinate_system_combo.itemData(index)
        next_coordinate_system = coordinate_system if isinstance(coordinate_system, str) else None
        if next_coordinate_system == self._app_state.coordinate_system:
            return

        # Result of the set_coordinate_system() call immediately below:
        #
        # returns True: request accepted
        #     → pre-change participant accepted the request
        #     → app state changed its coordinate system
        #     → coordinate_system_changed was emitted
        #     → shared event handler refreshed the widget
        #
        # returns False: request rejected because
        #     → a dirty Shapes Annotation session exists
        #     → the coordinate-system change triggers discard confirmation
        #     → the user cancels that confirmation
        #     → try_close_edit_session() returns False
        #     → set_coordinate_system() returns False
        # rejected-request handling
        #     → no event exists
        #     → restore the combo and local derived state here
        if not self._app_state.set_coordinate_system(next_coordinate_system, source=_SOURCE):
            self._sync_coordinate_system_combo_selection(self._app_state.coordinate_system)

    def prepare_coordinate_system_change(self, request: CoordinateSystemChangeRequest) -> bool:
        """Prepare the Shapes child for a coordinate change from any widget.

        Viewer, Object Classification, and other widgets share this app state.
        Unsaved polygon edits still exist only in the Shapes child's editable
        napari layer, so the parent must reject the change before Harpy removes
        old-coordinate-system layers when the user cancels discard.
        """
        del request
        # This QSignalBlocker is the final-only AnnotationContext publication
        # boundary: closing the child can synchronously emit dirty=False while
        # the parent still retains the old selection.
        with QSignalBlocker(self.shapes_annotation):
            return self.shapes_annotation.try_close_edit_session(reason="coordinate_system")

    def _on_shapes_target_changed(self, index: int) -> None:
        next_target = self._shapes_target_from_combo_index(index)
        accepted_target = self._annotation_context.shapes_target
        if next_target == accepted_target:
            return
        # This QSignalBlocker is the final-only AnnotationContext publication
        # boundary: closing the child can synchronously emit dirty=False while
        # the parent still retains the old selection.
        with QSignalBlocker(self.shapes_annotation):
            closed = self.shapes_annotation.try_close_edit_session(reason="shapes_target")
        if not closed:
            self._sync_shapes_target_combo_selection(accepted_target)
            return

        self._refresh_shapes_combo_tooltip()
        self._apply_and_publish_context()

    def _on_child_shapes_target_change_requested(self, target: object) -> None:
        if not isinstance(target, ShapesAnnotationTarget):
            raise TypeError("Shapes target-change requests must carry a ShapesAnnotationTarget.")
        # This QSignalBlocker is the final-only AnnotationContext publication
        # boundary: closing the child can synchronously emit dirty=False while
        # the parent still retains the old selection.
        with QSignalBlocker(self.shapes_annotation):
            closed = self.shapes_annotation.try_close_edit_session(reason="shapes_target")
        if not closed:
            self._sync_shapes_target_combo_selection(self._annotation_context.shapes_target)
            return

        self._refresh_shapes_targets(preferred_target=target)
        if self._shapes_target_from_combo_index(self.shapes_combo.currentIndex()) != target:
            return
        self._apply_and_publish_context()

    def _on_child_edit_session_saved(self, target: object) -> None:
        if not isinstance(target, ShapesAnnotationTarget) or target.mode != "edit_existing":
            raise TypeError("Successful Shapes saves must carry an edit-existing ShapesAnnotationTarget.")
        self._refresh_shapes_targets(preferred_target=target)
        self._apply_and_publish_context()

    def _on_child_dirty_state_changed(self, dirty: bool) -> None:
        self._annotation_context = replace(self._annotation_context, has_unsaved_shapes_changes=dirty)
        # Parent publication boundary: expose the child-reported dirty state
        # through the shared AnnotationContext rather than the Shapes child.
        self.annotation_context_changed.emit(self._annotation_context)

    def _apply_and_publish_context(self) -> None:
        context = AnnotationContext(
            sdata=self._app_state.sdata,
            coordinate_system=self._app_state.coordinate_system,
            shapes_target=self._shapes_target_from_combo_index(self.shapes_combo.currentIndex()),
            has_unsaved_shapes_changes=self.shapes_annotation.has_unsaved_changes,
        )
        # apply_annotation_context() may clear a stale session or open/refresh
        # an annotation layer. Those operations can synchronously emit
        # edit_session_dirty_changed. This QSignalBlocker suppresses those
        # intermediate emissions; the parent reads the child's final dirty state
        # and publishes one AnnotationContext below.
        with QSignalBlocker(self.shapes_annotation):
            self.shapes_annotation.apply_annotation_context(context)

        self._annotation_context = replace(
            context,
            has_unsaved_shapes_changes=self.shapes_annotation.has_unsaved_changes,
        )
        # Parent → sibling children/observers: publish only the final context
        # after the Shapes child has adopted the committed parent selection.
        self.annotation_context_changed.emit(self._annotation_context)

    def _sync_coordinate_system_combo_selection(self, coordinate_system: str | None) -> None:
        with QSignalBlocker(self.coordinate_system_combo):
            if coordinate_system is None:
                self.coordinate_system_combo.setCurrentIndex(-1)
                return
            self.coordinate_system_combo.setCurrentIndex(self.coordinate_system_combo.findData(coordinate_system))

    def _refresh_shapes_targets(self, *, preferred_target: ShapesAnnotationTarget | None = None) -> None:
        previous_target = preferred_target if preferred_target is not None else self._annotation_context.shapes_target
        sdata = self._app_state.sdata
        coordinate_system = self._app_state.coordinate_system
        if sdata is None or coordinate_system is None:
            eligible_existing_shapes_names = []
        else:
            eligible_existing_shapes_names = [
                option.shapes_name
                for option in get_spatialdata_shapes_options_for_coordinate_system_from_sdata(
                    sdata=sdata,
                    coordinate_system=coordinate_system,
                )
            ]

        with QSignalBlocker(self.shapes_combo):
            self.shapes_combo.clear()
            for shapes_name in eligible_existing_shapes_names:
                visible_shapes_name, shortened = format_feedback_identifier(
                    shapes_name,
                    max_length=_STATUS_IDENTIFIER_MAX_LENGTH,
                )
                target = ShapesAnnotationTarget.edit_existing(shapes_name)
                self.shapes_combo.addItem(visible_shapes_name, target)
                if shortened:
                    self.shapes_combo.setItemData(
                        self.shapes_combo.count() - 1,
                        format_tooltip(shapes_name),
                        Qt.ItemDataRole.ToolTipRole,
                    )

            if sdata is not None and coordinate_system is not None:
                self.shapes_combo.addItem(_CREATE_SHAPES_OPTION_TEXT, ShapesAnnotationTarget.create_new())
            self.shapes_combo.setEnabled(self.shapes_combo.count() > 0)

            next_index = self._find_shapes_target_combo_index(previous_target)
            if next_index < 0:
                next_index = self._find_shapes_target_combo_index(ShapesAnnotationTarget.create_new())
            self.shapes_combo.setCurrentIndex(next_index)

        self._refresh_shapes_combo_tooltip()

    def _find_shapes_target_combo_index(self, target: ShapesAnnotationTarget | None) -> int:
        if target is None:
            return -1
        for index in range(self.shapes_combo.count()):
            if self.shapes_combo.itemData(index) == target:
                return index
        return -1

    def _sync_shapes_target_combo_selection(self, target: ShapesAnnotationTarget | None) -> None:
        with QSignalBlocker(self.shapes_combo):
            if target is None:
                self.shapes_combo.setCurrentIndex(-1)
            else:
                self.shapes_combo.setCurrentIndex(self._find_shapes_target_combo_index(target))
        self._refresh_shapes_combo_tooltip()

    def _shapes_target_from_combo_index(self, index: int) -> ShapesAnnotationTarget | None:
        item_data = self.shapes_combo.itemData(index) if 0 <= index < self.shapes_combo.count() else None
        return item_data if isinstance(item_data, ShapesAnnotationTarget) else None

    def _refresh_shapes_combo_tooltip(self) -> None:
        target = self._shapes_target_from_combo_index(self.shapes_combo.currentIndex())
        if target is not None and target.mode == "edit_existing" and target.existing_shapes_name is not None:
            self.shapes_combo.setToolTip(format_tooltip(target.existing_shapes_name))
            return
        self.shapes_combo.setToolTip("")

    def _create_header_logo(self) -> QLabel:
        logo_label = QLabel()
        logo_label.setObjectName("annotation_header_logo")
        logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        logo_path = get_logo_path()
        logo_pixmap = QPixmap(str(logo_path))
        if not logo_pixmap.isNull():
            logo_label.setPixmap(logo_pixmap.scaledToWidth(120, Qt.TransformationMode.SmoothTransformation))
            return logo_label

        logo_label.setText("napari-harpy")
        logo_label.setStyleSheet(f"color: {WIDGET_TEXT_COLOR}; font-size: 18px; font-weight: 600;")
        return logo_label
