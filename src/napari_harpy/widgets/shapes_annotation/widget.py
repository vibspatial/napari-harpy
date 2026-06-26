"""Annotation widget for creating or editing SpatialData shapes elements.

The widget keeps target selection separate from active edit sessions:
`_ShapesAnnotationTarget` represents the current UI dropdown choice before a
layer is opened, while `_ShapesAnnotationSession` represents the locked save
target and source metadata after a layer has been created, loaded, or adopted.
This prevents changing the dropdown from silently changing the active layer's
save target.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from napari.layers import Shapes
from napari.layers.base._base_constants import ActionType
from napari.layers.shapes._shapes_constants import Mode
from qtpy.QtCore import QSignalBlocker, Qt, QTimer
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

from napari_harpy._app_state import (
    CoordinateSystemChangedEvent,
    HarpyAppState,
    ShapesElementWrittenEvent,
    get_or_create_app_state,
)
from napari_harpy._resources import get_logo_path
from napari_harpy.core.shapes_annotation import (
    DEFAULT_SHAPES_INDEX_NAME,
    DEFAULT_SHAPES_INDEX_PREFIX,
    AnnotateShapesElementResult,
    CreateShapesElementRequest,
    EditShapesElementRequest,
    create_shapes_element_from_napari_shapes_layer,
    edit_shapes_element_from_napari_shapes_layer,
    validate_existing_shapes_source_geodataframe,
)
from napari_harpy.core.shapes_geometry import (
    NapariPolygonTopology,
    delete_napari_polygon_vertex,
    napari_polygon_vertices_to_topology,
    sync_napari_polygon_anchor_vertex,
)
from napari_harpy.core.spatialdata import (
    get_annotating_table_names,
    get_coordinate_system_names_from_sdata,
    get_spatialdata_shapes_options_for_coordinate_system_from_sdata,
)
from napari_harpy.core.validation import (
    normalize_spatialdata_name,
    spatialdata_element_name_exists,
)
from napari_harpy.viewer.adapter import ShapesLayerBinding
from napari_harpy.widgets.shapes_annotation._create_holes import (
    _apply_create_holes_plan,
    _create_holes_plan_from_selection,
)
from napari_harpy.widgets.shapes_annotation._layer_style import (
    _capture_shapes_layer_style,
    _restore_shapes_layer_current_style,
    _restore_shapes_layer_row_styles,
    _trim_stale_private_color_rows_before_rebuild,
)
from napari_harpy.widgets.shapes_annotation._snapshot import (
    _annotation_layer_snapshots_equal,
    _capture_annotation_layer_snapshot,
    _empty_annotation_layer_snapshot,
    _ShapesAnnotationLayerSnapshot,
)
from napari_harpy.widgets.shapes_annotation.status_card import (
    _ShapesAnnotationStatusCardSpec,
    build_annotation_coordinate_system_missing_card_spec,
    build_annotation_create_layer_error_card_spec,
    build_annotation_create_target_ready_card_spec,
    build_annotation_edit_warning_card_spec,
    build_annotation_existing_shapes_opened_card_spec,
    build_annotation_existing_target_ready_card_spec,
    build_annotation_invalid_shapes_name_card_spec,
    build_annotation_layer_ready_card_spec,
    build_annotation_no_coordinate_systems_card_spec,
    build_annotation_no_spatialdata_card_spec,
    build_annotation_open_shapes_error_card_spec,
    build_annotation_reload_shapes_error_card_spec,
    build_annotation_save_error_card_spec,
    build_annotation_save_success_card_spec,
    build_annotation_save_unavailable_card_spec,
    build_annotation_shapes_name_exists_card_spec,
    build_annotation_shapes_unavailable_card_spec,
    build_annotation_target_missing_card_spec,
    build_create_holes_error_card_spec,
    build_create_holes_success_card_spec,
)
from napari_harpy.widgets.shared_styles import (
    ACTION_BUTTON_STYLESHEET,
    SECONDARY_BUTTON_STYLESHEET,
    WARNING_BUTTON_STYLESHEET,
    WIDGET_MIN_WIDTH,
    WIDGET_TEXT_COLOR,
    CompactComboBox,
    apply_scroll_content_surface,
    apply_widget_surface,
    build_input_control_stylesheet,
    create_form_label,
    format_feedback_identifier,
    format_tooltip,
    set_status_card,
)

if TYPE_CHECKING:
    import geopandas as gpd
    import napari
    from spatialdata import SpatialData


_SOURCE = "shapes_annotation_widget"
_INPUT_CONTROL_STYLESHEET = build_input_control_stylesheet("QComboBox, QLineEdit")
_ANNOTATION_FIELD_MIN_WIDTH = 180
_STATUS_IDENTIFIER_MAX_LENGTH = 32
_CREATE_SHAPES_OPTION_TEXT = "Create shapes..."
_DEFAULT_NEW_SHAPES_NAME = "new_shapes"
_ShapesAnnotationTargetMode = Literal["create_new", "edit_existing"]
_ShapesAnnotationLayerOrigin = Literal[
    "created_by_annotation",
    "loaded_by_annotation",
    "adopted_primary",
]


def _normalize_native_shapes_layer_transform(layer: Shapes) -> None:
    """Bake one native napari Shapes-layer transform into its vertices.

    Napari's own "new shapes" action can clone the active layer's transform
    onto the new Shapes layer. Harpy's save path serializes polygon vertices
    from ``layer.data`` directly, so adopted native layers must be normalized
    to identity-like transform semantics before a save/edit session begins.
    """
    baked_data = [
        # `Shapes.data_to_world(...)` is a single-position helper; use the
        # array-capable underlying transform so one call can transform the
        # whole `(n_vertices, ndim)` array at once.
        np.asarray(layer._data_to_world(vertices), dtype=float)
        for vertices in layer.data
    ]

    layer.scale = (1.0,) * layer.ndim
    layer.translate = (0.0,) * layer.ndim
    layer.rotate = 0.0
    layer.shear = (0.0,) * max(layer.ndim * (layer.ndim - 1) // 2, 0)
    layer.affine = np.eye(layer.ndim + 1, dtype=float)

    if baked_data:
        layer.data = baked_data


@dataclass(frozen=True)
class _ShapesAnnotationTarget:
    mode: _ShapesAnnotationTargetMode
    existing_shapes_name: str | None = None

    def __post_init__(self) -> None:
        if self.mode == "create_new":
            if self.existing_shapes_name is not None:
                raise ValueError("Create-new shapes targets cannot carry an existing shapes name.")
            return

        if self.mode == "edit_existing":
            if self.existing_shapes_name is None or not self.existing_shapes_name.strip():
                raise ValueError("Edit-existing shapes targets require a shapes name.")
            return

        raise ValueError(f"Unknown shapes annotation target mode: {self.mode!r}.")

    @classmethod
    def create_new(cls) -> _ShapesAnnotationTarget:
        return cls(mode="create_new")

    @classmethod
    def edit_existing(cls, shapes_name: str) -> _ShapesAnnotationTarget:
        return cls(mode="edit_existing", existing_shapes_name=shapes_name)


@dataclass(frozen=True)
class _ActivePrimaryShapesCandidate:
    """Compatible primary Shapes layer selected through napari's active layer."""

    layer: Shapes
    shapes_name: str
    coordinate_system: str


@dataclass(frozen=True)
class _ShapesAnnotationSession:
    """Locked save target and source metadata for one annotation session.

    Attributes
    ----------
    mode
        Whether the active layer belongs to create-new or edit-existing
        annotation workflow.
    layer_origin
        How the active napari layer entered the session. This determines
        discard behavior: create-new layers can be removed, while existing
        layers should be reloaded from saved SpatialData after discard.
    shapes_name
        The locked `sdata.shapes[...]` element name this session saves to.
    coordinate_system
        The locked coordinate system whose transformed coordinates are shown in
        the editable napari layer and used when saving.
    source_shapes_index_feature_name
        Feature column on the napari layer that stores each row's source
        GeoDataFrame index value. SpatialData keeps row identity in the
        GeoDataFrame index; napari keeps it in `layer.features`.
    source_geodataframe
        Defensive snapshot of the source GeoDataFrame for edit-existing
        sessions. It is used to preserve non-geometry columns and source index
        metadata when saving edits. Create-new sessions keep this as `None`
        because there is no source GeoDataFrame before the first save. After
        first save, the saved element can be treated as the source for later
        overwrite saves.
    source_geodataframe_index_name
        Derived name of the source GeoDataFrame index, if any. It is exposed as
        a property so the session keeps `source_geodataframe` as the single
        source of truth for source index metadata.
    table_linked
        Whether one or more SpatialData tables annotate this shapes element.
        Linked tables do not block editing, but the widget can warn that row
        additions or deletions may leave tables out of sync.
    """

    mode: _ShapesAnnotationTargetMode
    layer_origin: _ShapesAnnotationLayerOrigin
    shapes_name: str
    coordinate_system: str
    source_shapes_index_feature_name: str
    source_geodataframe: gpd.GeoDataFrame | None = None
    table_linked: bool = False

    @property
    def reload_on_discard(self) -> bool:
        return self.layer_origin in {"loaded_by_annotation", "adopted_primary"}

    @property
    def source_geodataframe_index_name(self) -> str | None:
        if self.source_geodataframe is None:
            return None
        return self.source_geodataframe.index.name


@dataclass(frozen=True)
class _AnnotationLayerReadiness:
    """Readiness of the widget-owned annotation layer for annotation actions.

    Attributes
    ----------
    actionable
        Whether the current widget-owned layer has enough locked session state
        to run actions such as Save shapes and Create holes.
    status
        Optional status-card message that explains the current readiness.
    """

    actionable: bool
    status: _ShapesAnnotationStatusCardSpec | None = None


@dataclass(frozen=True)
class _AnchorDragState:
    row_index: int
    moved_vertex_index: int
    topology: NapariPolygonTopology


@dataclass(frozen=True)
class _VertexDeleteState:
    row_index: int
    deleted_vertex_index: int
    vertices: np.ndarray
    topology: NapariPolygonTopology


def _shape_type_at(layer: Shapes, row_index: int) -> object:
    try:
        return layer.shape_type[row_index]
    except (IndexError, TypeError):
        return None


class _AnnotationLayerEditGuard:
    """Install and restore annotation-specific Shapes direct-edit hooks."""

    def __init__(self, *, warning_callback: Callable[[str], None] | None = None) -> None:
        self._layer: Shapes | None = None
        self._original_drag_modes: dict[object, Callable[..., Any]] | None = None
        self._had_instance_drag_modes = False
        self._wrapped_direct_callback: Callable[..., Any] | None = None
        self._wrapped_vertex_remove_callback: Callable[..., Any] | None = None
        self._is_syncing_anchor_drag = False
        self._warning_callback = warning_callback

    @property
    def layer(self) -> Shapes | None:
        return self._layer

    def attach(self, layer: Shapes) -> None:
        if self._layer is layer:
            return

        self.disconnect()
        drag_modes = getattr(layer, "_drag_modes", None)
        if not isinstance(drag_modes, dict) or Mode.DIRECT not in drag_modes or Mode.VERTEX_REMOVE not in drag_modes:
            raise ValueError("Shapes layer does not expose napari annotation edit mode hooks.")

        original_direct_callback = drag_modes[Mode.DIRECT]
        original_vertex_remove_callback = drag_modes[Mode.VERTEX_REMOVE]

        def wrapped_direct_callback(*args: Any, **kwargs: Any) -> Any:
            direct_drag = original_direct_callback(*args, **kwargs)
            if not hasattr(direct_drag, "__next__"):
                return direct_drag
            event = args[1] if len(args) > 1 else kwargs.get("event")
            return self._iter_direct_drag_with_anchor_sync(layer, direct_drag, event)

        def wrapped_vertex_remove_callback(*args: Any, **kwargs: Any) -> Any:
            event = args[1] if len(args) > 1 else kwargs.get("event")
            return self._route_vertex_remove(
                layer,
                original_vertex_remove_callback,
                args,
                kwargs,
                event,
            )

        patched_drag_modes = dict(drag_modes)
        patched_drag_modes[Mode.DIRECT] = wrapped_direct_callback
        patched_drag_modes[Mode.VERTEX_REMOVE] = wrapped_vertex_remove_callback

        self._layer = layer
        self._original_drag_modes = drag_modes
        # `layer._drag_modes` may be inherited from napari rather than stored
        # on this layer instance. Remember that distinction so disconnect can
        # either restore the original instance mapping or delete our temporary
        # instance override and fall back to napari's default mapping.
        self._had_instance_drag_modes = "_drag_modes" in vars(layer)
        self._wrapped_direct_callback = wrapped_direct_callback
        self._wrapped_vertex_remove_callback = wrapped_vertex_remove_callback
        layer._drag_modes = patched_drag_modes

    def disconnect(self) -> None:
        layer = self._layer
        original_drag_modes = self._original_drag_modes
        had_instance_drag_modes = self._had_instance_drag_modes

        self._layer = None
        self._original_drag_modes = None
        self._had_instance_drag_modes = False
        self._wrapped_direct_callback = None
        self._wrapped_vertex_remove_callback = None
        self._is_syncing_anchor_drag = False

        if layer is None:
            return
        if had_instance_drag_modes:
            if original_drag_modes is None:
                return
            layer._drag_modes = original_drag_modes
            return
        # Case where the layer did not originally own `_drag_modes`: attach
        # created an instance override only for this guard. Remove it so napari
        # resolves the normal inherited/default mapping again.
        if "_drag_modes" in vars(layer):
            delattr(layer, "_drag_modes")

    def _iter_direct_drag_with_anchor_sync(
        self,
        layer: Shapes,
        direct_drag: Iterator[Any],
        event: object,
    ) -> Iterator[Any]:
        """Mirror napari's direct-drag generator while repairing anchor copies.

        The first ``next(direct_drag)`` runs napari's mouse-press setup and
        pauses at its first ``yield``. At that point napari has populated
        ``layer._moving_value`` but has not moved any vertex yet, so after a
        mouse-press step we can safely cache the pre-move hole topology. Later
        ``next(...)`` calls let napari process mouse moves first; after each
        mouse move we synchronize any duplicated anchor/separator vertices from
        the cached topology.
        """
        active_drag: _AnchorDragState | None = None
        try:
            try:
                yielded = next(direct_drag)
            except StopIteration:
                return

            # Defensive guard: the normal napari path reaches this point from
            # mouse press, but unexpected event phases should fall back to
            # napari's original behavior instead of caching topology.
            if getattr(event, "type", None) == "mouse_press":
                active_drag = self._capture_anchor_drag_state(layer)
            yield yielded

            while True:
                try:
                    yielded = next(direct_drag)
                except StopIteration:
                    return

                if getattr(event, "type", None) == "mouse_move":
                    self._sync_anchor_drag(layer, active_drag)
                yield yielded
        finally:
            close = getattr(direct_drag, "close", None)
            if callable(close):
                close()

    def _capture_anchor_drag_state(self, layer: Shapes) -> _AnchorDragState | None:
        moving_value = getattr(layer, "_moving_value", None)
        if not isinstance(moving_value, tuple) or len(moving_value) != 2:
            return None

        # `moving_value` is `(row_index, vertex_index)`: the rendered napari row
        # and moving vertex indices.
        row_index, moved_vertex_index = moving_value
        if not isinstance(row_index, (int, np.integer)) or not isinstance(moved_vertex_index, (int, np.integer)):
            return None

        row_index = int(row_index)
        moved_vertex_index = int(moved_vertex_index)
        if row_index < 0 or moved_vertex_index < 0 or row_index >= len(layer.data):
            return None
        if _shape_type_at(layer, row_index) != "polygon":
            return None

        try:
            topology = napari_polygon_vertices_to_topology(layer.data[row_index])
        except ValueError:
            return None

        for group in topology.synchronized_anchor_groups:
            if moved_vertex_index in group:
                return _AnchorDragState(
                    row_index=row_index,
                    moved_vertex_index=moved_vertex_index,
                    topology=topology,
                )
        return None

    def _sync_anchor_drag(self, layer: Shapes, active_drag: _AnchorDragState | None) -> None:
        if active_drag is None or self._is_syncing_anchor_drag:
            return
        if active_drag.row_index >= len(layer.data):
            return

        vertices = np.asarray(layer.data[active_drag.row_index], dtype=float)
        if active_drag.moved_vertex_index >= len(vertices):
            return

        moved_coordinate = vertices[active_drag.moved_vertex_index]
        try:
            synchronized_vertices = sync_napari_polygon_anchor_vertex(
                vertices,
                active_drag.topology,
                active_drag.moved_vertex_index,
                moved_coordinate,
            )
        except ValueError:
            return

        if np.array_equal(vertices, synchronized_vertices):
            return

        self._is_syncing_anchor_drag = True
        try:
            # Mirror napari's direct-drag write path. Anchor synchronization
            # only changes coordinates, so the row length and vertex cache stay
            # stable.
            layer._data_view.edit(active_drag.row_index, synchronized_vertices)
            layer.refresh()
        finally:
            self._is_syncing_anchor_drag = False

    def _route_vertex_remove(
        self,
        layer: Shapes,
        original_vertex_remove_callback: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        event: object,
    ) -> Any:
        delete_state = self._capture_vertex_delete_state(layer, event)
        if delete_state is None:
            return original_vertex_remove_callback(*args, **kwargs)

        try:
            updated_vertices, _updated_topology = delete_napari_polygon_vertex(
                delete_state.vertices,
                delete_state.topology,
                delete_state.deleted_vertex_index,
            )
        except ValueError as error:
            self._warn(str(error))
            return None

        # Mirror napari's native `vertex_remove(...)` event contract around
        # our topology-preserving edit path.
        layer.events.data(
            value=layer.data,
            action=ActionType.CHANGING,
            data_indices=(delete_state.row_index,),
            vertex_indices=((delete_state.deleted_vertex_index,),),
        )
        # Successful vertex deletion currently shortens the row, which needs a
        # cache rebuild. The `else` branch is kept only for a future helper
        # path that might rewrite vertices without changing row length.
        if len(updated_vertices) != len(delete_state.vertices):
            self._replace_shape_row_rebuilding_vertex_cache(layer, delete_state.row_index, updated_vertices)
        else:
            with layer.events.set_data.blocker():
                layer._data_view.edit(delete_state.row_index, updated_vertices)
        layer.events.data(
            value=layer.data,
            action=ActionType.CHANGED,
            data_indices=(delete_state.row_index,),
            vertex_indices=((delete_state.deleted_vertex_index,),),
        )
        layer.refresh()
        return None

    def _replace_shape_row_rebuilding_vertex_cache(
        self,
        layer: Shapes,
        row_index: int,
        updated_vertices: np.ndarray,
    ) -> None:
        row_count = len(layer.data)
        current_mode = layer.mode
        selected_data = set(layer.selected_data)
        style_snapshot = _capture_shapes_layer_style(layer, row_count=row_count)
        rebuilt_data = list(layer.data)
        rebuilt_data[row_index] = updated_vertices

        _trim_stale_private_color_rows_before_rebuild(layer, style_snapshot)

        # The caller emits the napari-style CHANGING/CHANGED data events for
        # this vertex deletion, so block intermediate events triggered by
        # `layer.data = rebuilt_data`.
        with layer.events.data.blocker(), layer.events.features.blocker():
            # Work around napari's private vertex cache after row-shortening
            # edits: low-level `_data_view.edit(...)` updates the shape data
            # but can leave old clickable vertex indices behind, so a later
            # hit-test may report an index that no longer exists in
            # `layer.data[row_index]`.
            layer.data = rebuilt_data

        layer.opacity = style_snapshot.opacity
        layer.mode = current_mode
        # Defensive guard: this helper only replaces a row today, but avoid
        # restoring impossible selections if a future caller removes rows.
        selected_data = {index for index in selected_data if index < len(layer.data)}
        if row_index < len(layer.data):
            selected_data.add(row_index)
        layer.selected_data = selected_data

        # Restore current draw defaults without emitting Harpy's style sync
        # callbacks, then reapply row styles last so callback side effects
        # cannot overwrite the final styling.
        _restore_shapes_layer_current_style(layer, style_snapshot)
        _restore_shapes_layer_row_styles(layer, style_snapshot, row_indices=range(len(layer.data)))

    def _capture_vertex_delete_state(self, layer: Shapes, event: object) -> _VertexDeleteState | None:
        """Return delete state only for hole-bearing polygon rows we own.

        Every ``Mode.VERTEX_REMOVE`` click enters the edit guard, but most
        clicks should still use napari's original deletion behavior. This method
        is the routing gate: it returns ``None`` for no-vertex clicks,
        non-polygon rows, malformed rows, simple polygons without encoded holes,
        and out-of-range hit-test results. For valid polygon rows with encoded
        holes, we intentionally return a state object so deletion is delegated
        to Harpy's topology-preserving path instead of napari's raw vertex
        deletion.
        """
        try:
            value = layer.get_value(getattr(event, "position", None), world=True)
        except (AttributeError, TypeError, ValueError, IndexError):
            return None
        if not isinstance(value, tuple) or len(value) != 2:
            return None

        row_index, deleted_vertex_index = value
        if deleted_vertex_index is None:
            return None
        if not isinstance(row_index, (int, np.integer)) or not isinstance(deleted_vertex_index, (int, np.integer)):
            return None

        row_index = int(row_index)
        deleted_vertex_index = int(deleted_vertex_index)
        if row_index < 0 or deleted_vertex_index < 0 or row_index >= len(layer.data):
            return None
        if _shape_type_at(layer, row_index) != "polygon":
            return None

        try:
            vertices = np.asarray(layer.data[row_index], dtype=float).copy()
        except (TypeError, ValueError):
            return None
        if deleted_vertex_index >= len(vertices):
            return None

        try:
            topology = napari_polygon_vertices_to_topology(vertices)
        except ValueError:
            return None
        # Important routing guard: only hole-bearing polygon rows use Harpy's
        # custom deletion path. Simple polygons and other shapes stay napari-owned.
        if not topology.hole_anchor_groups:
            return None

        return _VertexDeleteState(
            row_index=row_index,
            deleted_vertex_index=deleted_vertex_index,
            vertices=vertices,
            topology=topology,
        )

    def _warn(self, message: str) -> None:
        if self._warning_callback is not None:
            self._warning_callback(message)


class ShapesAnnotation(QWidget):
    """Widget shell for annotating SpatialData shapes elements."""

    def __init__(self, napari_viewer: napari.Viewer | None = None) -> None:
        super().__init__()
        self.setObjectName("shapes_annotation_widget")
        apply_widget_surface(self)
        self.setMinimumWidth(WIDGET_MIN_WIDTH)

        self._viewer = napari_viewer
        self._app_state = get_or_create_app_state(napari_viewer)
        self._coordinate_systems: list[str] = []
        self._selected_coordinate_system: str | None = None
        self._selected_shapes_target: _ShapesAnnotationTarget | None = None
        self._eligible_existing_shapes_names: list[str] = []
        self._validated_shapes_name: str | None = None
        self._annotation_session: _ShapesAnnotationSession | None = None
        self._annotation_layer: Shapes | None = None
        self._annotation_edit_guard = _AnnotationLayerEditGuard(warning_callback=self._set_annotation_edit_warning)
        self._annotation_has_been_saved = False
        self._annotation_clean_snapshot: _ShapesAnnotationLayerSnapshot | None = None
        # Suppress `_on_viewer_layer_removed(...)` while this widget removes
        # an annotation layer itself during discard or clean session teardown.
        self._is_handling_annotation_layer_removal = False
        # Ignore primary-shapes registration events emitted by open operations
        # that this widget started itself.
        self._is_opening_annotation_layer = False
        # Active-layer adoption can activate layers again, so keep this handler
        # single-entry before follow-up slices add real adoption behavior.
        self._is_handling_active_layer_change = False
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
        form_layout.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        form_layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.FieldsStayAtSizeHint)
        form_layout.setHorizontalSpacing(12)
        form_layout.setVerticalSpacing(10)

        self.coordinate_system_combo = CompactComboBox()
        self.coordinate_system_combo.setObjectName("shapes_annotation_coordinate_system_combo")
        self.coordinate_system_combo.setStyleSheet(_INPUT_CONTROL_STYLESHEET)
        self.coordinate_system_combo.setMinimumWidth(_ANNOTATION_FIELD_MIN_WIDTH)

        self.shapes_combo = CompactComboBox()
        self.shapes_combo.setObjectName("shapes_annotation_shapes_combo")
        self.shapes_combo.setStyleSheet(_INPUT_CONTROL_STYLESHEET)
        self.shapes_combo.setMinimumWidth(_ANNOTATION_FIELD_MIN_WIDTH)

        self.new_shapes_name_label = create_form_label("New shapes name")
        self.new_shapes_name_label.setObjectName("shapes_annotation_new_shapes_name_label")

        self.name_edit = QLineEdit()
        self.name_edit.setObjectName("shapes_annotation_new_shapes_name_edit")
        self.name_edit.setPlaceholderText(_DEFAULT_NEW_SHAPES_NAME)
        self.name_edit.setStyleSheet(_INPUT_CONTROL_STYLESHEET)
        self.name_edit.setMinimumWidth(_ANNOTATION_FIELD_MIN_WIDTH)

        form_layout.addRow(create_form_label("Coordinate System"), self.coordinate_system_combo)
        form_layout.addRow(create_form_label("Shapes"), self.shapes_combo)
        form_layout.addRow(self.new_shapes_name_label, self.name_edit)

        button_row = QHBoxLayout()
        button_row.setContentsMargins(0, 0, 0, 0)
        button_row.setSpacing(8)

        self.create_layer_button = QPushButton("Create layer")
        self.create_layer_button.setObjectName("shapes_annotation_create_layer_button")
        self.create_layer_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.create_layer_button.setStyleSheet(ACTION_BUTTON_STYLESHEET)

        self.create_holes_button = QPushButton("Create holes")
        self.create_holes_button.setObjectName("shapes_annotation_create_holes_button")
        self.create_holes_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.create_holes_button.setStyleSheet(ACTION_BUTTON_STYLESHEET)

        self.save_shapes_button = QPushButton("Save shapes")
        self.save_shapes_button.setObjectName("shapes_annotation_save_shapes_button")
        self.save_shapes_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.save_shapes_button.setStyleSheet(ACTION_BUTTON_STYLESHEET)

        button_row.addWidget(self.create_layer_button)
        button_row.addWidget(self.create_holes_button)
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
        self._app_state.viewer_adapter.primary_shapes_layer_registered.connect(self._on_primary_shapes_layer_registered)
        self.coordinate_system_combo.currentIndexChanged.connect(self._on_coordinate_system_changed)
        self.shapes_combo.currentIndexChanged.connect(self._on_shapes_target_changed)
        self.name_edit.textChanged.connect(self._on_shapes_name_changed)
        self.create_layer_button.clicked.connect(self._on_create_layer_clicked)
        self.create_holes_button.clicked.connect(self._on_create_holes_clicked)
        self.save_shapes_button.clicked.connect(self._on_save_shapes_clicked)
        viewer_layers = getattr(napari_viewer, "layers", None)
        layer_events = getattr(viewer_layers, "events", None)
        layer_inserted_event = getattr(layer_events, "inserted", None)
        layer_inserted_connect = getattr(layer_inserted_event, "connect", None)
        if callable(layer_inserted_connect):
            layer_inserted_connect(self._on_viewer_layer_inserted)
        layer_removed_event = getattr(layer_events, "removed", None)
        layer_removed_connect = getattr(layer_removed_event, "connect", None)
        if callable(layer_removed_connect):
            layer_removed_connect(self._on_viewer_layer_removed)
        selection = getattr(viewer_layers, "selection", None)
        active_layer_event = getattr(getattr(selection, "events", None), "active", None)
        active_layer_connect = getattr(active_layer_event, "connect", None)
        if callable(active_layer_connect):
            active_layer_connect(self._on_active_layer_changed)
        self.refresh_from_sdata(self._app_state.sdata)

    @property
    def app_state(self) -> HarpyAppState:
        """Return the shared per-viewer Harpy app state."""
        return self._app_state

    @property
    def _annotation_shapes_name(self) -> str | None:
        session = self._annotation_session
        return None if session is None else session.shapes_name

    @property
    def _annotation_coordinate_system(self) -> str | None:
        session = self._annotation_session
        return None if session is None else session.coordinate_system

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
        """Return the currently validated shapes element name."""
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
        self._refresh_shapes_targets()
        self._refresh_create_layer_state()

    def _on_sdata_changed(self, sdata: SpatialData | None) -> None:
        self.refresh_from_sdata(sdata)

    def _on_app_state_coordinate_system_changed(self, event: CoordinateSystemChangedEvent) -> None:
        del event
        self._sync_coordinate_system_combo_selection(self._app_state.coordinate_system)
        self._set_selected_coordinate_system(self.coordinate_system_combo.currentIndex())
        self._refresh_shapes_targets()
        self._refresh_create_layer_state()

    def _on_coordinate_system_changed(self, index: int) -> None:
        coordinate_system = self.coordinate_system_combo.itemData(index)
        next_coordinate_system = coordinate_system if isinstance(coordinate_system, str) else None
        if self._annotation_layer is not None:
            if next_coordinate_system == self._app_state.coordinate_system:
                return
            if self._annotation_layer_has_unsaved_changes():
                # `False` means the user cancelled the discard warning, so
                # restore the previous coordinate-system selection and keep the
                # layer.
                if not self._confirm_discard_annotation_layer(context="coordinate_system"):
                    self._sync_coordinate_system_combo_selection(self._app_state.coordinate_system)
                    self._set_selected_coordinate_system(self.coordinate_system_combo.currentIndex())
                    self._refresh_create_layer_state()
                    return
                self._discard_annotation_layer()
            else:
                self._close_clean_annotation_session()
            self._app_state.set_coordinate_system(next_coordinate_system, source=_SOURCE)
            return

        # Publish the UI choice to shared app state. `_on_app_state_coordinate_system_changed(...)`
        # owns local selection and create-layer refresh so all sources follow one path.
        self._app_state.set_coordinate_system(next_coordinate_system, source=_SOURCE)

    def _on_shapes_target_changed(self, index: int) -> None:
        next_target = self._shapes_target_from_combo_index(index)
        if self._annotation_layer is not None:
            if next_target == self._selected_shapes_target:
                return
            if self._annotation_layer_has_unsaved_changes():
                if not self._confirm_discard_annotation_layer(context="target"):
                    self._sync_shapes_target_combo_selection(self._selected_shapes_target)
                    self._refresh_create_layer_state()
                    return
                self._discard_annotation_layer()
            else:
                self._close_clean_annotation_session()

        self._set_selected_shapes_target_from_combo(index)
        self._refresh_create_layer_state()
        if next_target is not None and next_target.mode == "edit_existing" and self._annotation_layer is None:
            self._open_existing_annotation_layer()

    def _on_shapes_name_changed(self, _text: str) -> None:
        self._refresh_create_layer_state()

    def _on_active_layer_changed(self, event: object) -> None:
        """Observe every active-layer change; later adoption only cares about compatible Shapes layers."""
        if self._is_handling_active_layer_change:
            return
        if self._is_handling_annotation_layer_removal:
            # `_discard_annotation_layer(...)` may call
            # `ensure_shapes_loaded(...)` to reload a clean copy of the old
            # saved layer before `_clear_annotation_state()` runs. That reload
            # can transiently activate the old layer; do not treat it as a
            # user-driven active-layer switch.
            return

        layer = getattr(event, "value", None)
        if layer is None or layer is self._annotation_layer:
            return

        self._is_handling_active_layer_change = True
        try:
            self._maybe_adopt_active_shapes_layer(layer)
        finally:
            self._is_handling_active_layer_change = False

    def _maybe_adopt_active_shapes_layer(self, layer: object) -> None:
        candidate = self._active_primary_shapes_candidate(layer)
        if candidate is None:
            return
        if self._annotation_layer is not None:
            current_layer = self._annotation_layer
            if self._annotation_layer_has_unsaved_changes():
                if not self._confirm_discard_annotation_layer(context="target"):
                    # The user cancelled the dirty-session switch while we are
                    # inside napari's active-layer-change handling for the
                    # layer the user just clicked. If we reactivate the old
                    # annotation layer immediately, Qt may still finish the
                    # original click afterward and leave the new layer selected
                    # visually. Defer reactivation of the old layer to the
                    # next event-loop turn so cancel keeps napari and the
                    # widget on the old layer.
                    QTimer.singleShot(0, lambda: self._app_state.viewer_adapter.activate_layer(current_layer))
                    self._refresh_create_layer_state()
                    return
                self._discard_annotation_layer()
            else:
                self._close_clean_annotation_session()

        target = _ShapesAnnotationTarget.edit_existing(candidate.shapes_name)
        self._sync_shapes_target_combo_selection(target)
        self._refresh_create_layer_state()
        # In the happy path this is still `None`: we either had no session or
        # just closed a clean one. Keep the guard so a future target-refresh
        # side effect cannot open the same layer twice.
        if self._annotation_layer is None:
            self._open_existing_annotation_layer()

    def _active_primary_shapes_candidate(self, layer: object) -> _ActivePrimaryShapesCandidate | None:
        if not isinstance(layer, Shapes):
            return None

        sdata = self._app_state.sdata
        coordinate_system = self._selected_coordinate_system
        if sdata is None or coordinate_system is None:
            return None

        binding = self._app_state.viewer_adapter.layer_bindings.get_binding(layer)
        if not isinstance(binding, ShapesLayerBinding):
            return None

        if (
            binding.element_type != "shapes"
            or binding.sdata_id != id(sdata)
            or binding.coordinate_system != coordinate_system
            or binding.shapes_role != "primary"
            or binding.shapes_rendering_mode != "shapes"
            or binding.style_spec is not None
            or binding.element_name not in self._eligible_existing_shapes_names
        ):
            return None

        return _ActivePrimaryShapesCandidate(
            layer=layer,
            shapes_name=binding.element_name,
            coordinate_system=coordinate_system,
        )

    def _on_primary_shapes_layer_registered(self, binding: object) -> None:
        if self._is_opening_annotation_layer:
            return
        if self._is_handling_annotation_layer_removal:
            # `_discard_annotation_layer(...)` may call
            # `ensure_shapes_loaded(...)` to reload a clean copy of the old
            # saved layer. That internal reload registers the layer before the
            # dirty session has been cleared, so the registration repair must
            # not adopt it.
            return
        if not isinstance(binding, ShapesLayerBinding):
            return

        active_layer = getattr(getattr(getattr(self._viewer, "layers", None), "selection", None), "active", None)
        # napari may expose the active layer through a PublicOnlyProxy in
        # plugin widgets; Harpy bindings keep the real layer object.
        active_layer = getattr(active_layer, "__wrapped__", active_layer)
        if binding.layer is active_layer:
            # Catch primary shapes loaded through the Viewer widget Add/Update
            # flow. `ensure_shapes_loaded(...)` calls
            # `_add_layer_to_viewer(...)` before `register_shapes_layer(...)`.
            # Napari can make that inserted layer active immediately, so
            # `_on_active_layer_changed` sees it before the binding exists and
            # rejects it as unbound. Once this registration signal fires, the
            # binding exists. Repair adoption here because the Viewer widget's
            # later
            # `activate_layer(...)` call may set `selection.active` to the
            # already-active layer. Napari returns early in that case, so it
            # may not emit a second active event for `_on_active_layer_changed`;
            # catch that missed active-layer adoption here.
            self._maybe_adopt_active_shapes_layer(binding.layer)
            return

        if self._annotation_layer is not None or self._annotation_session is not None:
            return

        sdata = self._app_state.sdata
        target = self._selected_shapes_target
        coordinate_system = self._selected_coordinate_system
        if (
            sdata is None
            or target is None
            or target.mode != "edit_existing"
            or target.existing_shapes_name is None
            or coordinate_system is None
        ):
            return

        if (
            binding.sdata_id != id(sdata)
            or binding.element_name != target.existing_shapes_name
            or binding.coordinate_system != coordinate_system
            or binding.shapes_role != "primary"
        ):
            return

        self._refresh_create_layer_state()
        self._open_existing_annotation_layer()

    def _on_viewer_layer_inserted(self, event: object) -> None:
        """Notice Shapes layers created or imported through napari's own UI."""
        layer = getattr(event, "value", None)
        if not isinstance(layer, Shapes):
            return

        # This hook lets Annotation react when the user creates or imports a
        # Shapes layer through napari itself. Harpy-managed insertions are
        # filtered out by the deferred binding check below.
        # Harpy-managed shapes paths in `ViewerAdapter` call
        # `_add_layer_to_viewer(self._viewer, layer)` before
        # `self.register_shapes_layer(...)`. `_add_layer_to_viewer(...)` emits
        # this raw insertion event, but Annotation should not adopt layers that
        # were added by the ViewerAdapter. `QTimer.singleShot(0, ...)` queues the
        # callback for the end of the current Qt event-loop turn, giving Harpy
        # time to register its own binding before we decide whether this is still
        # an unbound native napari layer.
        QTimer.singleShot(0, lambda layer=layer: self._maybe_adopt_native_shapes_layer(layer))

    def _maybe_adopt_native_shapes_layer(self, layer: object) -> None:
        if not isinstance(layer, Shapes):
            return
        if not self._viewer_contains_layer(layer):
            return
        if self._app_state.viewer_adapter.layer_bindings.get_binding(layer) is not None:
            return
        if getattr(layer, "ndim", 2) != 2:
            return

        sdata = self._app_state.sdata
        coordinate_system = self._selected_coordinate_system
        if sdata is None or coordinate_system is None:
            return

        if self._annotation_layer is not None:
            if self._annotation_layer_has_unsaved_changes():
                if not self._confirm_discard_annotation_layer(context="target"):
                    return
                self._discard_annotation_layer()
            else:
                self._close_clean_annotation_session()

        if not self._viewer_contains_layer(layer):
            return
        if self._app_state.viewer_adapter.layer_bindings.get_binding(layer) is not None:
            return

        shapes_name = self._unique_new_shapes_name_for_native_layer(layer)
        self._adopt_native_shapes_layer(layer, shapes_name=shapes_name, coordinate_system=coordinate_system)

    def _viewer_contains_layer(self, layer: object) -> bool:
        layers = getattr(self._viewer, "layers", None)
        if layers is None:
            return False
        try:
            return any(candidate is layer for candidate in layers)
        except TypeError:
            return False

    def _unique_new_shapes_name_for_native_layer(self, layer: Shapes) -> str:
        sdata = self._app_state.sdata
        if sdata is None:
            return _DEFAULT_NEW_SHAPES_NAME

        try:
            base_name = normalize_spatialdata_name(getattr(layer, "name", ""), "Shapes element name")
        except ValueError:
            base_name = _DEFAULT_NEW_SHAPES_NAME

        shapes_name = base_name
        suffix = 1
        while spatialdata_element_name_exists(sdata, shapes_name):
            shapes_name = f"{base_name}_{suffix}"
            suffix += 1
        return shapes_name

    def _adopt_native_shapes_layer(self, layer: Shapes, *, shapes_name: str, coordinate_system: str) -> None:
        sdata = self._app_state.sdata
        if sdata is None:
            return

        self._refresh_shapes_targets(preferred_target=_ShapesAnnotationTarget.create_new())
        with QSignalBlocker(self.name_edit):
            self.name_edit.setText(shapes_name)
        layer.name = shapes_name
        _normalize_native_shapes_layer_transform(layer)
        # Normalize native napari layers into Annotation's visual contract before
        # registering the layer and capturing the clean annotation baseline.
        self._app_state.viewer_adapter.apply_primary_shapes_layer_style(layer)

        self._is_opening_annotation_layer = True
        try:
            self._app_state.viewer_adapter.register_shapes_layer(
                layer,
                sdata=sdata,
                shapes_name=shapes_name,
                coordinate_system=coordinate_system,
                shapes_rendering_mode="shapes",
                source_shapes_index_feature_name=DEFAULT_SHAPES_INDEX_NAME,
            )
        finally:
            self._is_opening_annotation_layer = False

        self._annotation_layer = layer
        self._annotation_edit_guard.attach(layer)
        self._annotation_session = _ShapesAnnotationSession(
            mode="create_new",
            layer_origin="created_by_annotation",
            shapes_name=shapes_name,
            coordinate_system=coordinate_system,
            source_shapes_index_feature_name=DEFAULT_SHAPES_INDEX_NAME,
        )
        self._annotation_has_been_saved = False
        self._annotation_clean_snapshot = self._initial_native_layer_clean_snapshot(layer)
        self._set_create_name_controls_visible(True)
        self._app_state.viewer_adapter.activate_layer(layer)
        self._refresh_create_layer_state()

    def _initial_native_layer_clean_snapshot(self, layer: Shapes) -> _ShapesAnnotationLayerSnapshot:
        # Empty native layers should be clean as they are, including any empty
        # feature schema. Non-empty native/imported layers are unsaved Harpy
        # annotations, so compare them against an empty baseline and mark them
        # dirty immediately until the user saves them.
        if len(layer.data) == 0:
            return _capture_annotation_layer_snapshot(layer)
        return _empty_annotation_layer_snapshot()

    def _on_create_layer_clicked(self) -> None:
        if self._annotation_layer is not None:
            return

        self._refresh_create_layer_state()
        target = self._selected_shapes_target
        if target is None or target.mode != "create_new":
            return

        self._open_create_new_annotation_layer()

    def _open_create_new_annotation_layer(self) -> None:
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
                source_shapes_index_feature_name=DEFAULT_SHAPES_INDEX_NAME,
            )
        except ValueError as error:
            self._apply_status_card_spec(build_annotation_create_layer_error_card_spec(str(error)))
            self.create_layer_button.setEnabled(False)
            self._refresh_save_shapes_state()
            return

        self._annotation_layer = layer
        self._annotation_edit_guard.attach(layer)
        self._annotation_session = _ShapesAnnotationSession(
            mode="create_new",
            layer_origin="created_by_annotation",
            shapes_name=shapes_name,
            coordinate_system=coordinate_system,
            source_shapes_index_feature_name=DEFAULT_SHAPES_INDEX_NAME,
        )
        self._annotation_has_been_saved = False
        self._annotation_clean_snapshot = _capture_annotation_layer_snapshot(layer)
        self._set_create_name_controls_visible(True)
        self._app_state.viewer_adapter.activate_layer(layer)
        self._refresh_create_layer_state()

    def _open_existing_annotation_layer(self) -> None:
        sdata = self._app_state.sdata
        target = self._selected_shapes_target
        coordinate_system = self._selected_coordinate_system
        shapes_name = self._validated_shapes_name
        if (
            sdata is None
            or coordinate_system is None
            or target is None
            or target.mode != "edit_existing"
            or shapes_name is None
        ):
            return

        load_result = None
        try:
            source_geodataframe = validate_existing_shapes_source_geodataframe(sdata.shapes[shapes_name])
            # `ensure_shapes_loaded(...)` emits `primary_shapes_layer_registered`
            # when it creates/registers a layer. This widget handles that
            # signal by calling `_open_existing_annotation_layer(...)` for
            # matching selected targets, so suppress the emitted event while
            # this method is already setting up the session.
            self._is_opening_annotation_layer = True
            try:
                load_result = self._app_state.viewer_adapter.ensure_shapes_loaded(
                    sdata,
                    shapes_name,
                    coordinate_system,
                )
            finally:
                self._is_opening_annotation_layer = False
            layer = load_result.layer
            binding = self._validate_opened_existing_shapes_layer(
                layer,
                source_row_count=len(source_geodataframe),
                shapes_name=shapes_name,
                coordinate_system=coordinate_system,
            )
        except ValueError as error:
            if load_result is not None and load_result.created:
                self._app_state.viewer_adapter.remove_shapes_layer(sdata, shapes_name, coordinate_system)
            self._apply_status_card_spec(build_annotation_open_shapes_error_card_spec(str(error)))
            self.create_layer_button.setEnabled(False)
            self._refresh_save_shapes_state()
            return

        table_linked = bool(get_annotating_table_names(sdata, shapes_name))
        self._annotation_layer = layer
        self._annotation_edit_guard.attach(layer)
        self._annotation_session = _ShapesAnnotationSession(
            mode="edit_existing",
            layer_origin="loaded_by_annotation" if load_result.created else "adopted_primary",
            shapes_name=shapes_name,
            coordinate_system=coordinate_system,
            source_shapes_index_feature_name=binding.source_shapes_index_feature_name,
            source_geodataframe=source_geodataframe.copy(deep=True),
            table_linked=table_linked,
        )
        self._annotation_has_been_saved = True
        self._annotation_clean_snapshot = _capture_annotation_layer_snapshot(layer)
        self._app_state.viewer_adapter.activate_layer(layer)
        self._refresh_create_layer_state()

    def _validate_opened_existing_shapes_layer(
        self,
        layer: object,
        *,
        source_row_count: int,
        shapes_name: str,
        coordinate_system: str,
    ) -> ShapesLayerBinding:
        if not isinstance(layer, Shapes):
            raise ValueError(
                "This shapes element is rendered as points and cannot be edited as polygon annotations yet."
            )

        binding = self._app_state.viewer_adapter.layer_bindings.get_binding(layer)
        if not isinstance(binding, ShapesLayerBinding):
            raise ValueError("Opened shapes layer is missing its Harpy shapes binding.")

        if (
            binding.element_type != "shapes"
            or binding.element_name != shapes_name
            or binding.coordinate_system != coordinate_system
            or binding.sdata_id != id(self._app_state.sdata)
            or binding.shapes_role != "primary"
            or binding.shapes_rendering_mode != "shapes"
            or binding.style_spec is not None
        ):
            raise ValueError("Opened shapes layer is not a compatible primary shapes layer.")

        if binding.skipped_geometry_count:
            raise ValueError(
                "This shapes element cannot be edited because one or more source geometries were skipped while loading."
            )

        # `validate_existing_shapes_source_geodataframe(...)` already rejects
        # MultiPolygons explicitly. This binding check is the rendered-layer
        # guardrail: every source row must still map to exactly one napari row,
        # with no splitting, skipped rows, or reordering.
        if list(binding.source_row_id_by_rendered_row) != list(range(source_row_count)):
            raise ValueError(
                "This shapes element cannot be edited because one source row does not map to exactly one napari shape."
            )

        if binding.source_shapes_index_feature_name not in layer.features.columns:
            raise ValueError("Opened shapes layer is missing the source row identity feature column.")

        return binding

    def _on_save_shapes_clicked(self) -> None:
        readiness = self._refresh_save_shapes_state()
        if not readiness.actionable:
            if readiness.status is not None:
                self._apply_status_card_spec(readiness.status)
            return

        sdata = self._app_state.sdata
        layer = self._annotation_layer
        session = self._annotation_session
        if sdata is None or layer is None or session is None:
            return

        try:
            # The locked session mode is the save contract. Create-new gets one
            # guarded first save (`overwrite=False`); after success the session
            # is rebuilt as edit-existing so later saves overwrite through the
            # edit path.
            if session.mode == "create_new":
                request = CreateShapesElementRequest(
                    sdata=sdata,
                    shapes_name=session.shapes_name,
                    coordinate_system=session.coordinate_system,
                    overwrite=False,
                    index_name=DEFAULT_SHAPES_INDEX_NAME,
                    index_prefix=DEFAULT_SHAPES_INDEX_PREFIX,
                )
                result = create_shapes_element_from_napari_shapes_layer(request, layer)
            else:
                if session.source_geodataframe is None:
                    raise ValueError("Existing shapes session is missing its source GeoDataFrame metadata.")
                request = EditShapesElementRequest(
                    sdata=sdata,
                    shapes_name=session.shapes_name,
                    coordinate_system=session.coordinate_system,
                    source_geodataframe=session.source_geodataframe,
                    source_shapes_index_feature_name=session.source_shapes_index_feature_name,
                    index_prefix=DEFAULT_SHAPES_INDEX_PREFIX,
                )
                result = edit_shapes_element_from_napari_shapes_layer(request, layer)
            self._update_annotation_session_after_successful_save(
                sdata=sdata,
                layer=layer,
                result=result,
                previous_session=session,
            )
        except ValueError as error:
            self._refresh_save_shapes_state()
            self._apply_status_card_spec(build_annotation_save_error_card_spec(str(error)))
            return

        self._refresh_shapes_targets(preferred_target=_ShapesAnnotationTarget.edit_existing(result.shapes_name))
        self._refresh_save_shapes_state()
        self._app_state.emit_shapes_element_written(
            ShapesElementWrittenEvent(
                sdata=sdata,
                shapes_name=result.shapes_name,
                coordinate_system=result.coordinate_system,
                source=_SOURCE,
            )
        )
        self._apply_status_card_spec(
            build_annotation_save_success_card_spec(
                result=result,
                table_linked=self._annotation_session.table_linked if self._annotation_session is not None else False,
            )
        )

    def _on_create_holes_clicked(self) -> None:
        readiness = self._refresh_save_shapes_state()
        if not readiness.actionable:
            if readiness.status is not None:
                self._apply_status_card_spec(readiness.status)
            return

        layer = self._annotation_layer
        session = self._annotation_session
        if layer is None or session is None:
            return

        try:
            plan = _create_holes_plan_from_selection(layer)
            hole_count = len(plan.hole_row_indices)
            _apply_create_holes_plan(layer, plan)
        except ValueError as error:
            self._apply_status_card_spec(build_create_holes_error_card_spec(str(error)))
            return

        self._refresh_save_shapes_state()
        self._apply_status_card_spec(
            build_create_holes_success_card_spec(
                hole_count=hole_count,
                table_linked=session.table_linked,
            )
        )

    def _update_annotation_session_after_successful_save(
        self,
        *,
        sdata: SpatialData,
        layer: Shapes,
        result: AnnotateShapesElementResult,
        previous_session: _ShapesAnnotationSession,
    ) -> None:
        saved_geodataframe = validate_existing_shapes_source_geodataframe(sdata.shapes[result.shapes_name])
        layer_origin: _ShapesAnnotationLayerOrigin = previous_session.layer_origin
        source_shapes_index_feature_name = previous_session.source_shapes_index_feature_name
        if previous_session.mode == "create_new":
            layer_origin = "loaded_by_annotation"
            source_shapes_index_feature_name = DEFAULT_SHAPES_INDEX_NAME
            # The typed create-new name has now been consumed by the saved
            # element, so reset the hidden field before it is shown again.
            self._clear_consumed_new_shapes_name()

        # Bring the Harpy registry back in sync with the just-saved
        # SpatialData element. The same live layer can now represent a
        # different row count and source-index feature name than when it was
        # first registered.
        self._app_state.viewer_adapter.sync_primary_shapes_layer_binding(
            layer,
            sdata=sdata,
            shapes_name=result.shapes_name,
            coordinate_system=result.coordinate_system,
            source_row_id_by_rendered_row=range(len(saved_geodataframe)),
            source_shapes_index_feature_name=source_shapes_index_feature_name,
        )

        self._annotation_session = _ShapesAnnotationSession(
            mode="edit_existing",
            layer_origin=layer_origin,
            shapes_name=result.shapes_name,
            coordinate_system=result.coordinate_system,
            source_shapes_index_feature_name=source_shapes_index_feature_name,
            source_geodataframe=saved_geodataframe.copy(deep=True),
            table_linked=bool(get_annotating_table_names(sdata, result.shapes_name)),
        )
        self._annotation_has_been_saved = True
        self._annotation_clean_snapshot = _capture_annotation_layer_snapshot(layer)

    def _clear_consumed_new_shapes_name(self) -> None:
        with QSignalBlocker(self.name_edit):
            self.name_edit.clear()

    def _on_viewer_layer_removed(self, event: object) -> None:
        # Coordinate-system and shapes-target changes can remove the annotation
        # layer programmatically through `_discard_annotation_layer(...)` or
        # `_close_clean_annotation_session(...)`; those paths own cleanup, so
        # ignore the layer-removal event they emit.
        if self._is_handling_annotation_layer_removal:
            return

        layer = getattr(event, "value", None)
        if layer is not self._annotation_layer:
            return

        # The adapter also listens for layer removals, but callback order is not
        # part of the contract. Unregister defensively in case the widget
        # observes this event first.
        self._app_state.viewer_adapter.unregister_layer(layer)
        self._clear_annotation_state()
        self._refresh_shapes_targets(preferred_target=_ShapesAnnotationTarget.create_new())
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

    def _refresh_shapes_targets(self, *, preferred_target: _ShapesAnnotationTarget | None = None) -> None:
        previous_target = preferred_target if preferred_target is not None else self._selected_shapes_target
        sdata = self._app_state.sdata
        coordinate_system = self._selected_coordinate_system

        if sdata is None or coordinate_system is None:
            self._eligible_existing_shapes_names = []
        else:
            self._eligible_existing_shapes_names = [
                option.shapes_name
                for option in get_spatialdata_shapes_options_for_coordinate_system_from_sdata(
                    sdata=sdata,
                    coordinate_system=coordinate_system,
                )
            ]

        with QSignalBlocker(self.shapes_combo):
            self.shapes_combo.clear()
            for shapes_name in self._eligible_existing_shapes_names:
                visible_shapes_name, shortened = format_feedback_identifier(
                    shapes_name,
                    max_length=_STATUS_IDENTIFIER_MAX_LENGTH,
                )
                target = _ShapesAnnotationTarget.edit_existing(shapes_name)
                self.shapes_combo.addItem(visible_shapes_name, target)
                if shortened:
                    self.shapes_combo.setItemData(
                        self.shapes_combo.count() - 1,
                        format_tooltip(shapes_name),
                        Qt.ItemDataRole.ToolTipRole,
                    )

            if sdata is not None and coordinate_system is not None:
                self.shapes_combo.addItem(_CREATE_SHAPES_OPTION_TEXT, _ShapesAnnotationTarget.create_new())

            self.shapes_combo.setEnabled(self.shapes_combo.count() > 0)

            next_index = self._find_shapes_target_combo_index(previous_target)
            if next_index < 0:
                next_index = self._find_shapes_target_combo_index(_ShapesAnnotationTarget.create_new())
            self.shapes_combo.setCurrentIndex(next_index)

        self._set_selected_shapes_target_from_combo(self.shapes_combo.currentIndex())

    def _find_shapes_target_combo_index(self, target: _ShapesAnnotationTarget | None) -> int:
        if target is None:
            return -1
        for index in range(self.shapes_combo.count()):
            if self.shapes_combo.itemData(index) == target:
                return index
        return -1

    def _sync_shapes_target_combo_selection(self, target: _ShapesAnnotationTarget | None) -> None:
        with QSignalBlocker(self.shapes_combo):
            if target is None:
                self.shapes_combo.setCurrentIndex(-1)
                return

            self.shapes_combo.setCurrentIndex(self._find_shapes_target_combo_index(target))
        self._set_selected_shapes_target_from_combo(self.shapes_combo.currentIndex())

    def _shapes_target_from_combo_index(self, index: int) -> _ShapesAnnotationTarget | None:
        item_data = self.shapes_combo.itemData(index) if 0 <= index < self.shapes_combo.count() else None
        return item_data if isinstance(item_data, _ShapesAnnotationTarget) else None

    def _set_selected_shapes_target_from_combo(self, index: int) -> None:
        target = self._shapes_target_from_combo_index(index)
        self._selected_shapes_target = target
        is_create_new = target is not None and target.mode == "create_new"
        self._set_create_name_controls_visible(is_create_new)
        self._refresh_shapes_combo_tooltip()

    def _set_create_name_controls_visible(self, is_visible: bool) -> None:
        self.new_shapes_name_label.setVisible(is_visible)
        self.name_edit.setVisible(is_visible)
        self.name_edit.setEnabled(is_visible and self._annotation_layer is None)

    def _refresh_shapes_combo_tooltip(self) -> None:
        target = self._selected_shapes_target
        if target is not None and target.mode == "edit_existing" and target.existing_shapes_name is not None:
            self.shapes_combo.setToolTip(format_tooltip(target.existing_shapes_name))
            return
        self.shapes_combo.setToolTip("")

    def _refresh_create_layer_state(self) -> None:
        """Update layer-creation readiness from current sdata, coordinate system, and name."""
        sdata = self._app_state.sdata
        coordinate_system = self._selected_coordinate_system
        target = self._selected_shapes_target
        self._validated_shapes_name = None
        self.create_layer_button.setText("Create layer")

        if self._annotation_layer is not None:
            self._validated_shapes_name = self._annotation_shapes_name
            self.create_layer_button.setEnabled(False)
            readiness = self._refresh_save_shapes_state()
            self._apply_status_card_spec(readiness.status)
            return

        if sdata is None:
            self._apply_status_card_spec(build_annotation_no_spatialdata_card_spec())
            self.create_layer_button.setEnabled(False)
            self._refresh_save_shapes_state()
            return

        if not self._coordinate_systems:
            self._apply_status_card_spec(build_annotation_no_coordinate_systems_card_spec())
            self.create_layer_button.setEnabled(False)
            self._refresh_save_shapes_state()
            return

        if coordinate_system is None:
            self._apply_status_card_spec(build_annotation_coordinate_system_missing_card_spec())
            self.create_layer_button.setEnabled(False)
            self._refresh_save_shapes_state()
            return

        if target is None:
            self._apply_status_card_spec(build_annotation_target_missing_card_spec())
            self.create_layer_button.setEnabled(False)
            self._refresh_save_shapes_state()
            return

        if target.mode == "edit_existing":
            shapes_name = target.existing_shapes_name
            if (
                shapes_name is None
                or shapes_name not in sdata.shapes
                or shapes_name not in self._eligible_existing_shapes_names
            ):
                self._apply_status_card_spec(build_annotation_shapes_unavailable_card_spec())
                self.create_layer_button.setEnabled(False)
                self._refresh_save_shapes_state()
                return

            self._validated_shapes_name = shapes_name
            self._apply_status_card_spec(
                build_annotation_existing_target_ready_card_spec(
                    shapes_name=shapes_name,
                    coordinate_system=coordinate_system,
                )
            )
            self.create_layer_button.setEnabled(False)
            self._refresh_save_shapes_state()
            return

        try:
            shapes_name = normalize_spatialdata_name(self.name_edit.text(), "Shapes element name")
        except ValueError as error:
            self._apply_status_card_spec(build_annotation_invalid_shapes_name_card_spec(str(error)))
            self.create_layer_button.setEnabled(False)
            self._refresh_save_shapes_state()
            return

        if spatialdata_element_name_exists(sdata, shapes_name):
            self._apply_status_card_spec(build_annotation_shapes_name_exists_card_spec(shapes_name))
            self.create_layer_button.setEnabled(False)
            self._refresh_save_shapes_state()
            return

        self._validated_shapes_name = shapes_name
        self._apply_status_card_spec(
            build_annotation_create_target_ready_card_spec(
                shapes_name=shapes_name,
                coordinate_system=coordinate_system,
            )
        )
        self.create_layer_button.setEnabled(True)
        self._refresh_save_shapes_state()

    def _refresh_save_shapes_state(self) -> _AnnotationLayerReadiness:
        """Update action button readiness for the widget-owned annotation layer."""
        readiness = self._evaluate_annotation_layer_readiness()
        self.save_shapes_button.setEnabled(readiness.actionable)
        self.create_holes_button.setEnabled(readiness.actionable)
        return readiness

    def _evaluate_annotation_layer_readiness(self) -> _AnnotationLayerReadiness:
        """Return whether the widget-owned annotation layer can be edited and saved."""
        layer = self._annotation_layer
        if layer is None:
            return _AnnotationLayerReadiness(actionable=False)

        if self._app_state.sdata is None:
            return _AnnotationLayerReadiness(
                actionable=False,
                status=build_annotation_save_unavailable_card_spec("Load a SpatialData object before saving shapes."),
            )

        if self._annotation_shapes_name is None or self._annotation_coordinate_system is None:
            return _AnnotationLayerReadiness(
                actionable=False,
                status=build_annotation_save_unavailable_card_spec(
                    "The annotation layer is missing its locked save target."
                ),
            )

        if not self._annotation_layer_binding_matches():
            return _AnnotationLayerReadiness(
                actionable=False,
                status=build_annotation_save_unavailable_card_spec(
                    "The annotation layer is no longer registered as the widget-owned primary shapes layer."
                ),
            )

        session = self._annotation_session
        if session is None:
            return _AnnotationLayerReadiness(
                actionable=False,
                status=build_annotation_save_unavailable_card_spec(
                    "The annotation layer is missing its locked save session."
                ),
            )

        if session.mode == "edit_existing":
            if session.source_geodataframe is None:
                return _AnnotationLayerReadiness(
                    actionable=False,
                    status=build_annotation_save_unavailable_card_spec(
                        "The edit session is missing its source shapes metadata."
                    ),
                )

            return _AnnotationLayerReadiness(
                actionable=True,
                status=build_annotation_existing_shapes_opened_card_spec(
                    shapes_name=session.shapes_name,
                    coordinate_system=session.coordinate_system,
                    table_linked=session.table_linked,
                ),
            )

        return _AnnotationLayerReadiness(actionable=True, status=build_annotation_layer_ready_card_spec())

    def _annotation_layer_binding_matches(self) -> bool:
        layer = self._annotation_layer
        sdata = self._app_state.sdata
        if layer is None or sdata is None:
            return False

        binding = self._app_state.viewer_adapter.layer_bindings.get_binding(layer)
        source_shapes_index_feature_name = (
            self._annotation_session.source_shapes_index_feature_name
            if self._annotation_session is not None
            else DEFAULT_SHAPES_INDEX_NAME
        )
        return (
            binding is not None
            and binding.element_type == "shapes"
            and binding.element_name == self._annotation_shapes_name
            and binding.coordinate_system == self._annotation_coordinate_system
            and binding.sdata_id == id(sdata)
            and getattr(binding, "shapes_role", None) == "primary"
            and getattr(binding, "shapes_rendering_mode", None) == "shapes"
            and getattr(binding, "style_spec", None) is None
            and getattr(binding, "source_shapes_index_feature_name", None) == source_shapes_index_feature_name
        )

    def _apply_status_card_spec(self, spec: _ShapesAnnotationStatusCardSpec | None) -> None:
        if spec is None:
            self.status_label.clear()
            self.status_label.setToolTip("")
            self.status_label.setStyleSheet("")
            return

        set_status_card(
            self.status_label,
            title=spec.title,
            lines=list(spec.lines),
            kind=spec.kind,
            tooltip_message=spec.tooltip_message,
        )

    def _set_annotation_edit_warning(self, message: str) -> None:
        self._apply_status_card_spec(build_annotation_edit_warning_card_spec(message))

    def _confirm_discard_annotation_layer(self, *, context: Literal["coordinate_system", "target"]) -> bool:
        if context == "coordinate_system":
            message = "Changing coordinate system will discard the current unsaved shape annotations."
        elif context == "target":
            message = "Switching shapes target will discard the current unsaved shape annotations."
        else:
            raise ValueError(f"Unknown discard context: {context!r}.")

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
            lines=[message],
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

    def _discard_annotation_layer(self) -> None:
        """Discard the active annotation session, reloading saved layers when needed."""
        sdata = self._app_state.sdata
        shapes_name = self._annotation_shapes_name
        coordinate_system = self._annotation_coordinate_system
        reload_on_discard = (
            self._annotation_session.reload_on_discard if self._annotation_session is not None else False
        )

        self._is_handling_annotation_layer_removal = True
        try:
            self._remove_annotation_layer()
            if reload_on_discard and sdata is not None and shapes_name is not None and coordinate_system is not None:
                try:
                    self._app_state.viewer_adapter.ensure_shapes_loaded(sdata, shapes_name, coordinate_system)
                except ValueError as error:
                    self._apply_status_card_spec(build_annotation_reload_shapes_error_card_spec(str(error)))
            self._clear_annotation_state()
        finally:
            self._is_handling_annotation_layer_removal = False

    def _close_clean_annotation_session(self) -> None:
        """Release a clean annotation session without dirty discard/reload work."""
        should_remove_layer = self._annotation_layer is not None and not self._annotation_has_been_saved

        self._is_handling_annotation_layer_removal = True
        try:
            if should_remove_layer:
                self._remove_annotation_layer()
            self._clear_annotation_state()
        finally:
            self._is_handling_annotation_layer_removal = False

    def _annotation_layer_has_unsaved_changes(self) -> bool:
        layer = self._annotation_layer
        clean_snapshot = self._annotation_clean_snapshot
        if layer is None or clean_snapshot is None:
            return False

        try:
            current_snapshot = _capture_annotation_layer_snapshot(layer)
        except ValueError:
            return True
        return not _annotation_layer_snapshots_equal(current_snapshot, clean_snapshot)

    def _clear_annotation_state(self) -> None:
        self._annotation_edit_guard.disconnect()
        self._annotation_session = None
        self._annotation_layer = None
        self._annotation_has_been_saved = False
        self._annotation_clean_snapshot = None
        self._set_create_name_controls_visible(
            self._selected_shapes_target is not None and self._selected_shapes_target.mode == "create_new"
        )

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
