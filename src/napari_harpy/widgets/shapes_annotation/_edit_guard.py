"""Private napari edit-hook adapter for the shapes annotation widget."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

import numpy as np
from napari.layers import Shapes
from napari.layers.base._base_constants import ActionType
from napari.layers.shapes._shapes_constants import Mode
from napari.layers.shapes._shapes_models import Ellipse, Path
from napari.layers.shapes._shapes_utils import point_to_lines
from napari.utils.key_bindings import coerce_keybinding

from napari_harpy._shapes_triangulation import ensure_shapes_triangulation_backend
from napari_harpy.core.shapes_geometry import (
    NapariPolygonTopology,
    delete_napari_polygon_vertex,
    insert_napari_polygon_vertex,
    move_napari_polygon_vertex,
    napari_polygon_vertices_to_shapely_polygon,
    napari_polygon_vertices_to_topology,
)
from napari_harpy.widgets.shapes_annotation._layer_state import (
    _capture_shapes_layer_baseline,
    _restore_shapes_layer_baseline,
)
from napari_harpy.widgets.shapes_annotation._layer_style import (
    _capture_shapes_layer_style,
    _restore_shapes_layer_current_style,
    _restore_shapes_layer_row_styles,
    _trim_stale_private_color_rows_before_rebuild,
)

_SPACE_KEYBINDING = coerce_keybinding("Space")
_INVALID_POLYGON_DRAG_WARNING = "Polygon edit was rejected because it would create invalid geometry."
_POLYGON_DRAG_RENDERING_WARNING = (
    "The polygon edit could not be rendered, so the previous accepted position was restored."
)
_POLYGON_DELETE_RENDERING_WARNING = (
    "The polygon deletion could not be rendered, so the previous layer state was restored."
)
_POLYGON_INSERT_RENDERING_WARNING = (
    "The polygon insertion could not be rendered, so the previous layer state was restored."
)
_ALREADY_INVALID_POLYGON_DRAG_WARNING = "This polygon is already invalid, so this edit cannot be safely rolled back."
_SPACE_PAN_RESUMABLE_DRAW_MODES = frozenset(
    {
        Mode.ADD_POLYGON_LASSO,
        Mode.ADD_PATH,
        Mode.ADD_POLYGON,
        Mode.ADD_POLYLINE,
    }
)
_POLYGON_GUARDED_EDIT_MODES = frozenset(
    {
        Mode.DIRECT,
        Mode.VERTEX_REMOVE,
        Mode.VERTEX_INSERT,
    }
)


@dataclass
class _PolygonVertexDragState:
    row_index: int
    moved_vertex_index: int
    topology: NapariPolygonTopology
    last_valid_vertices: np.ndarray
    warning_emitted: bool = False
    has_accepted_move: bool = False


class _PolygonVertexDragRoute(Enum):
    DELEGATE = auto()
    GUARD = auto()
    REJECT = auto()


class _PolygonVertexDeleteRoute(Enum):
    DELEGATE = auto()
    GUARD = auto()
    REJECT = auto()


class _PolygonVertexInsertRoute(Enum):
    DELEGATE = auto()
    GUARD = auto()
    REJECT = auto()


@dataclass(frozen=True)
class _VertexDeleteState:
    row_index: int
    deleted_vertex_index: int
    vertices: np.ndarray
    topology: NapariPolygonTopology


@dataclass(frozen=True)
class _VertexInsertState:
    row_index: int
    insert_index: int
    vertices: np.ndarray
    topology: NapariPolygonTopology
    inserted_coordinate: np.ndarray


def _shape_type_at(layer: Shapes, row_index: int) -> object:
    try:
        return layer.shape_type[row_index]
    except (IndexError, TypeError):
        return None


def _replace_callback(
    callbacks: list[Callable[..., Any]],
    *,
    old_callback: Callable[..., Any],
    new_callback: Callable[..., Any],
) -> None:
    for index, callback in enumerate(callbacks):
        if callback is old_callback:
            callbacks[index] = new_callback


class _AnnotationLayerEditGuard:
    """Install and restore annotation-specific Shapes direct-edit hooks."""

    def __init__(
        self,
        *,
        warning_callback: Callable[[str], None] | None = None,
        polygon_edit_finished_callback: Callable[[], None] | None = None,
        can_space_pan_draw: Callable[[Shapes], bool] | None = None,
    ) -> None:
        self._layer: Shapes | None = None
        self._original_drag_modes: dict[object, Callable[..., Any]] | None = None
        self._original_move_modes: dict[object, Callable[..., Any]] | None = None
        self._had_instance_drag_modes = False
        self._had_instance_move_modes = False
        self._wrapped_direct_callback: Callable[..., Any] | None = None
        self._wrapped_vertex_remove_callback: Callable[..., Any] | None = None
        self._wrapped_vertex_insert_callback: Callable[..., Any] | None = None
        self._wrapped_space_keybinding: Callable[..., Any] | None = None
        # Track Space and mouse release independently so drawing resumes only
        # after both halves of a temporary Space-pan have ended.
        self._space_pan_key_held = False
        self._space_pan_mouse_gesture_active = False
        self._previous_mouse_pan: bool | None = None
        self._previous_space_keybinding: object | None = None
        self._had_instance_space_keybinding = False
        self._warning_callback = warning_callback
        self._polygon_edit_finished_callback = polygon_edit_finished_callback
        self._can_space_pan_draw = can_space_pan_draw

    @property
    def layer(self) -> Shapes | None:
        return self._layer

    def attach(self, layer: Shapes) -> None:
        if self._layer is layer:
            return

        self.disconnect()
        drag_modes = getattr(layer, "_drag_modes", None)
        move_modes = getattr(layer, "_move_modes", None)
        if (
            not isinstance(drag_modes, dict)
            or not isinstance(move_modes, dict)
            or Mode.DIRECT not in drag_modes
            or Mode.VERTEX_REMOVE not in drag_modes
            or Mode.VERTEX_INSERT not in drag_modes
            or not _SPACE_PAN_RESUMABLE_DRAW_MODES.issubset(drag_modes)
            or not _SPACE_PAN_RESUMABLE_DRAW_MODES.issubset(move_modes)
        ):
            raise ValueError("Shapes layer does not expose napari annotation edit mode hooks.")

        original_direct_callback = drag_modes[Mode.DIRECT]
        original_vertex_remove_callback = drag_modes[Mode.VERTEX_REMOVE]
        original_vertex_insert_callback = drag_modes[Mode.VERTEX_INSERT]

        def wrapped_direct_callback(*args: Any, **kwargs: Any) -> Any:
            direct_drag = original_direct_callback(*args, **kwargs)
            if not hasattr(direct_drag, "__next__"):
                return direct_drag
            event = args[1] if len(args) > 1 else kwargs.get("event")
            return self._iter_direct_drag_with_polygon_validation(layer, direct_drag, event)

        def wrapped_vertex_remove_callback(*args: Any, **kwargs: Any) -> Any:
            event = args[1] if len(args) > 1 else kwargs.get("event")
            return self._route_vertex_remove(
                layer,
                original_vertex_remove_callback,
                args,
                kwargs,
                event,
            )

        def wrapped_vertex_insert_callback(*args: Any, **kwargs: Any) -> Any:
            event = args[1] if len(args) > 1 else kwargs.get("event")
            return self._route_vertex_insert(
                layer,
                original_vertex_insert_callback,
                args,
                kwargs,
                event,
            )

        def wrapped_space_keybinding(bound_layer: Shapes) -> Iterator[None]:
            yield from self._handle_space_keybinding(bound_layer)

        patched_drag_modes = dict(drag_modes)
        patched_drag_modes[Mode.DIRECT] = wrapped_direct_callback
        patched_drag_modes[Mode.VERTEX_REMOVE] = wrapped_vertex_remove_callback
        patched_drag_modes[Mode.VERTEX_INSERT] = wrapped_vertex_insert_callback
        patched_move_modes = dict(move_modes)
        for mode in _SPACE_PAN_RESUMABLE_DRAW_MODES:
            patched_drag_modes[mode] = self._wrap_supported_draw_drag_callback(
                layer,
                drag_modes[mode],
            )
            patched_move_modes[mode] = self._wrap_supported_draw_move_callback(
                move_modes[mode],
            )

        self._capture_space_keybinding(layer)

        self._layer = layer
        self._original_drag_modes = drag_modes
        self._original_move_modes = move_modes
        # `layer._drag_modes` may be inherited from napari rather than stored
        # on this layer instance. Remember that distinction so disconnect can
        # either restore the original instance mapping or delete our temporary
        # instance override and fall back to napari's default mapping.
        self._had_instance_drag_modes = "_drag_modes" in vars(layer)
        self._had_instance_move_modes = "_move_modes" in vars(layer)
        self._wrapped_direct_callback = wrapped_direct_callback
        self._wrapped_vertex_remove_callback = wrapped_vertex_remove_callback
        self._wrapped_vertex_insert_callback = wrapped_vertex_insert_callback
        self._wrapped_space_keybinding = wrapped_space_keybinding
        layer._drag_modes = patched_drag_modes
        layer._move_modes = patched_move_modes
        self._replace_current_guarded_edit_callback(
            layer,
            old_drag_modes=drag_modes,
            new_drag_modes=patched_drag_modes,
        )
        # Defensive guard for layers already in a supported draw mode: napari's
        # active mouse callback lists may still point at the original callbacks
        # even after we replace `_drag_modes` / `_move_modes`.
        self._replace_current_supported_draw_callbacks(
            layer,
            original_drag_modes=drag_modes,
            original_move_modes=move_modes,
            patched_drag_modes=patched_drag_modes,
            patched_move_modes=patched_move_modes,
        )
        # `bind_key(...)` accepts raw strings and performs napari's key coercion;
        # this ends up roughly like assigning
        # `layer.keymap[_SPACE_KEYBINDING] = wrapped_space_keybinding`, but keeps
        # installation on napari's public keybinding API. Direct `layer.keymap`
        # inspection/restoration uses `_SPACE_KEYBINDING` for exact cleanup.
        layer.bind_key("Space", wrapped_space_keybinding, overwrite=True)

    def disconnect(self) -> None:
        layer = self._layer
        original_drag_modes = self._original_drag_modes
        original_move_modes = self._original_move_modes
        had_instance_drag_modes = self._had_instance_drag_modes
        had_instance_move_modes = self._had_instance_move_modes
        previous_mouse_pan = self._previous_mouse_pan
        previous_space_keybinding = self._previous_space_keybinding
        had_instance_space_keybinding = self._had_instance_space_keybinding

        self._layer = None
        self._original_drag_modes = None
        self._original_move_modes = None
        self._had_instance_drag_modes = False
        self._had_instance_move_modes = False
        self._wrapped_direct_callback = None
        self._wrapped_vertex_remove_callback = None
        self._wrapped_vertex_insert_callback = None
        self._wrapped_space_keybinding = None
        self._space_pan_key_held = False
        self._space_pan_mouse_gesture_active = False
        self._previous_mouse_pan = None
        self._previous_space_keybinding = None
        self._had_instance_space_keybinding = False

        if layer is None:
            return
        # If disconnect interrupts an active Space-pan, restore the layer
        # mouse-pan setting that this guard temporarily changed.
        if previous_mouse_pan is not None:
            layer.mouse_pan = previous_mouse_pan
        self._restore_space_keybinding(
            layer,
            previous_space_keybinding=previous_space_keybinding,
            had_instance_space_keybinding=had_instance_space_keybinding,
        )
        if original_drag_modes is not None and original_move_modes is not None:
            self._replace_current_guarded_edit_callback(
                layer,
                old_drag_modes=layer._drag_modes,
                new_drag_modes=original_drag_modes,
            )
            # Edge case: if attach happened while this layer was already in a
            # supported draw mode, attach swapped the active mouse callback
            # lists from napari's originals to our wrappers. Swap those entries
            # back before restoring the mode mappings so disconnect leaves the
            # currently active mode usable without requiring a mode change.
            self._restore_supported_draw_callbacks(
                layer,
                original_drag_modes=original_drag_modes,
                original_move_modes=original_move_modes,
            )
        if had_instance_drag_modes:
            if original_drag_modes is not None:
                layer._drag_modes = original_drag_modes
        # Cases where the layer did not originally own the mode mappings:
        # attach created instance overrides only for this guard. Remove them so
        # napari resolves the normal inherited/default mappings again.
        elif "_drag_modes" in vars(layer):
            delattr(layer, "_drag_modes")
        if had_instance_move_modes:
            if original_move_modes is not None:
                layer._move_modes = original_move_modes
        elif "_move_modes" in vars(layer):
            delattr(layer, "_move_modes")

    def _replace_current_guarded_edit_callback(
        self,
        layer: Shapes,
        *,
        old_drag_modes: dict[object, Callable[..., Any]],
        new_drag_modes: dict[object, Callable[..., Any]],
    ) -> None:
        mode = layer._mode
        if mode not in _POLYGON_GUARDED_EDIT_MODES:
            return

        # Lifecycle handled here: a guarded mode is already active, so napari
        # has copied its callback from `_drag_modes` into
        # `mouse_drag_callbacks`. Harpy then replaces `_drag_modes`, but that
        # does not update the copied active callback. Without this explicit
        # swap, napari's native mutating callback would remain active and
        # bypass Harpy's movement, deletion, or insertion guard. Disconnect
        # calls the same helper with old/new mappings reversed to restore the
        # active native callback.
        _replace_callback(
            layer.mouse_drag_callbacks,
            old_callback=old_drag_modes[mode],
            new_callback=new_drag_modes[mode],
        )

    def _replace_current_supported_draw_callbacks(
        self,
        layer: Shapes,
        *,
        original_drag_modes: dict[object, Callable[..., Any]],
        original_move_modes: dict[object, Callable[..., Any]],
        patched_drag_modes: dict[object, Callable[..., Any]],
        patched_move_modes: dict[object, Callable[..., Any]],
    ) -> None:
        mode = layer._mode
        if mode not in _SPACE_PAN_RESUMABLE_DRAW_MODES:
            return
        _replace_callback(
            layer.mouse_drag_callbacks,
            old_callback=original_drag_modes[mode],
            new_callback=patched_drag_modes[mode],
        )
        _replace_callback(
            layer.mouse_move_callbacks,
            old_callback=original_move_modes[mode],
            new_callback=patched_move_modes[mode],
        )

    def _restore_supported_draw_callbacks(
        self,
        layer: Shapes,
        *,
        original_drag_modes: dict[object, Callable[..., Any]],
        original_move_modes: dict[object, Callable[..., Any]],
    ) -> None:
        patched_drag_modes = layer._drag_modes
        patched_move_modes = layer._move_modes
        for mode in _SPACE_PAN_RESUMABLE_DRAW_MODES:
            _replace_callback(
                layer.mouse_drag_callbacks,
                old_callback=patched_drag_modes[mode],
                new_callback=original_drag_modes[mode],
            )
            _replace_callback(
                layer.mouse_move_callbacks,
                old_callback=patched_move_modes[mode],
                new_callback=original_move_modes[mode],
            )

    def _capture_space_keybinding(self, layer: Shapes) -> None:
        # Napari stores keymap entries under coerced KeyBinding objects, not the
        # raw `"Space"` string accepted by `bind_key(...)`.
        self._had_instance_space_keybinding = _SPACE_KEYBINDING in layer.keymap
        if self._had_instance_space_keybinding:
            self._previous_space_keybinding = layer.keymap[_SPACE_KEYBINDING]
        else:
            self._previous_space_keybinding = None

    def _restore_space_keybinding(
        self,
        layer: Shapes,
        *,
        previous_space_keybinding: object | None,
        had_instance_space_keybinding: bool,
    ) -> None:
        if had_instance_space_keybinding:
            layer.keymap[_SPACE_KEYBINDING] = previous_space_keybinding
        else:
            layer.keymap.pop(_SPACE_KEYBINDING, None)

    def _handle_space_keybinding(self, layer: Shapes) -> Iterator[None]:
        """Handle the layer instance napari passes to the Space keybinding.

        Napari calls layer keybindings with the bound layer as the first
        argument. Checking that argument against `self._layer` keeps the custom
        Space-pan branch scoped to the layer this guard currently owns.
        """
        if self._layer is layer and self._can_space_pan_draw_mode(layer) and self._can_use_custom_space_pan(layer):
            # Custom resumable drawing path: active supported drawings use
            # Space to toggle mouse-pan without leaving the drawing mode.
            self._begin_space_pan_key_hold(layer)
            try:
                yield
            finally:
                self._end_space_pan_key_hold(layer)
            return

        # Fallback path: keep napari-equivalent temporary pan/zoom behavior for
        # unsupported modes, inactive drawing modes, and any state where the
        # custom resumable drawing path is not ready.
        yield from self._fallback_pan_zoom_key_hold(layer)

    def _can_use_custom_space_pan(self, layer: Shapes) -> bool:
        """Return whether widget ownership allows the custom Space-pan path."""
        if self._can_space_pan_draw is None:
            return False
        return bool(self._can_space_pan_draw(layer))

    def _fallback_pan_zoom_key_hold(self, layer: Shapes) -> Iterator[None]:
        """Delegate Space to napari-equivalent fallback pan/zoom behavior."""
        previous_mode = layer.mode
        pan_zoom = layer._modeclass.PAN_ZOOM
        if previous_mode == pan_zoom:
            yield
            return

        layer.mode = pan_zoom
        try:
            yield
        finally:
            layer.mode = previous_mode

    def _drawing_is_suspended(self) -> bool:
        """Return whether draw callbacks should stay paused during Space-pan.

        Drawing remains suspended while either Space is still held or the
        Space-started mouse pan gesture has not released yet. Tracking both
        states lets either release order recover safely.
        """
        return self._space_pan_key_held or self._space_pan_mouse_gesture_active

    def _can_space_pan_draw_mode(self, layer: Shapes) -> bool:
        return bool(layer._is_creating and layer._mode in _SPACE_PAN_RESUMABLE_DRAW_MODES)

    def _wrap_supported_draw_drag_callback(
        self,
        layer: Shapes,
        original_callback: Callable[..., Any],
    ) -> Callable[..., Any]:
        def wrapped_supported_draw_drag_callback(*args: Any, **kwargs: Any) -> Any:
            if not self._drawing_is_suspended():
                return original_callback(*args, **kwargs)
            event = args[1] if len(args) > 1 else kwargs.get("event")
            if event.type == "mouse_press":
                # Return a small generator so napari advances us on mouse
                # move/release and we can clear the Space-pan mouse gesture.
                return self._iter_suppressed_space_pan_drag(layer, event)
            return None

        return wrapped_supported_draw_drag_callback

    def _wrap_supported_draw_move_callback(
        self,
        original_callback: Callable[..., Any],
    ) -> Callable[..., Any]:
        def wrapped_supported_draw_move_callback(*args: Any, **kwargs: Any) -> Any:
            if self._drawing_is_suspended():
                return None
            return original_callback(*args, **kwargs)

        return wrapped_supported_draw_move_callback

    def _iter_suppressed_space_pan_drag(
        self,
        layer: Shapes,
        event: object,
    ) -> Iterator[None]:
        """Track a Space-pan mouse gesture while suppressing drawing.

        Napari keeps drag callbacks alive by storing the generator returned from
        mouse press and advancing it on mouse move/release. Returning this
        generator lets us mark the mouse gesture active without running napari's
        original draw callback, then clear that state when the gesture ends so
        drawing resumes safely regardless of whether Space or mouse releases
        first.
        """
        self._begin_space_pan_mouse_gesture()
        try:
            yield
            while event.type == "mouse_move":
                yield
        finally:
            self._end_space_pan_mouse_gesture(layer)

    def _begin_space_pan_key_hold(self, layer: Shapes) -> None:
        self._space_pan_key_held = True
        if self._previous_mouse_pan is None:
            self._previous_mouse_pan = layer.mouse_pan
        layer.mouse_pan = True

    def _end_space_pan_key_hold(self, layer: Shapes) -> None:
        # This handles Space release, whether it happens before or after the
        # mouse pan gesture ends; restoration is gated by both state flags.
        self._space_pan_key_held = False
        self._restore_space_pan_if_complete(layer)

    def _begin_space_pan_mouse_gesture(self) -> None:
        self._space_pan_mouse_gesture_active = True

    def _end_space_pan_mouse_gesture(self, layer: Shapes) -> None:
        # This handles mouse release, whether it happens before or after Space
        # release; restoration is gated by both state flags.
        self._space_pan_mouse_gesture_active = False
        self._restore_space_pan_if_complete(layer)

    def _restore_space_pan_if_complete(self, layer: Shapes) -> None:
        # Wait until both release flags are clear: Space is no longer held and
        # the Space-started mouse pan gesture has ended.
        if self._space_pan_key_held or self._space_pan_mouse_gesture_active:
            return
        if self._previous_mouse_pan is None:
            return
        layer.mouse_pan = self._previous_mouse_pan
        self._previous_mouse_pan = None

    def _iter_direct_drag_with_polygon_validation(
        self,
        layer: Shapes,
        direct_drag: Iterator[Any],
        event: object,
    ) -> Iterator[Any]:
        """Run the press, move, and release contract for one direct drag.

        Advance napari's native generator exactly once so it performs the
        mouse-press setup and records the vertex under the cursor, then classify
        that initial press. Delegated gestures continue through the native
        generator. For Harpy-owned polygon gestures, keep the native generator
        suspended and validate and transactionally apply each mouse move
        ourselves. Finally, close the suspended generator and perform the
        release cleanup that napari can no longer perform. Completion is
        notified only after a normal release with at least one accepted move.
        """
        route = _PolygonVertexDragRoute.DELEGATE
        active_drag: _PolygonVertexDragState | None = None
        harpy_owned = False
        normal_release = False
        try:
            # This is the only advancement shared by every route. It lets
            # napari perform hit testing and press setup, including populating
            # `_moving_value`, and stops at its first yield before mutation.
            try:
                yielded = next(direct_drag)
            except StopIteration:
                return

            # Classify the vertex/shape recorded by napari's press setup before
            # deciding who is allowed to process subsequent mouse moves.
            if event.type == "mouse_press":
                route, active_drag = self._classify_polygon_vertex_drag(layer)

            if route is _PolygonVertexDragRoute.DELEGATE:
                # Napari retains ownership of both move and release handling
                # for gestures outside Harpy's polygon-vertex guard.
                yield yielded
                while True:
                    try:
                        yield next(direct_drag)
                    except StopIteration:
                        return

            # GUARD and REJECT leave the native generator suspended at its
            # press yield. Yield that press step to napari's event dispatcher,
            # then handle every later event without advancing native code.
            harpy_owned = True
            yield yielded

            while event.type == "mouse_move":
                if route is _PolygonVertexDragRoute.GUARD:
                    if active_drag is None:  # pragma: no cover - routing invariant
                        raise RuntimeError("Guarded polygon drag has no captured state.")
                    self._apply_polygon_vertex_drag_move(layer, active_drag, event)
                # REJECT deliberately consumes the move without mutation;
                # GUARD has already validated and transactionally applied it.
                yield None
            normal_release = event.type == "mouse_release"
        finally:
            try:
                # Harpy-owned routes never resume native code after its press
                # yield, so close that suspended generator before local cleanup.
                direct_drag.close()
            finally:
                if harpy_owned:
                    # Replace napari's unexecuted release path. The finisher
                    # emits completion only for a normal, accepted mutation.
                    self._finish_polygon_vertex_drag(
                        layer,
                        active_drag,
                        emit_completed=normal_release,
                    )

    def _classify_polygon_vertex_drag(
        self,
        layer: Shapes,
    ) -> tuple[_PolygonVertexDragRoute, _PolygonVertexDragState | None]:
        moving_value = layer._moving_value
        if not isinstance(moving_value, tuple) or len(moving_value) != 2:
            return _PolygonVertexDragRoute.DELEGATE, None

        # `moving_value` is `(row_index, vertex_index)`: the rendered napari row
        # and moving vertex indices.
        row_index, moved_vertex_index = moving_value
        if moved_vertex_index is None:
            return _PolygonVertexDragRoute.DELEGATE, None
        if row_index is None:
            self._warn(_ALREADY_INVALID_POLYGON_DRAG_WARNING)
            return _PolygonVertexDragRoute.REJECT, None
        if not isinstance(row_index, (int, np.integer)) or not isinstance(moved_vertex_index, (int, np.integer)):
            self._warn(_ALREADY_INVALID_POLYGON_DRAG_WARNING)
            return _PolygonVertexDragRoute.REJECT, None

        row_index = int(row_index)
        moved_vertex_index = int(moved_vertex_index)
        if row_index < 0 or moved_vertex_index < 0 or row_index >= len(layer.data):
            self._warn(_ALREADY_INVALID_POLYGON_DRAG_WARNING)
            return _PolygonVertexDragRoute.REJECT, None
        if _shape_type_at(layer, row_index) != "polygon":
            return _PolygonVertexDragRoute.DELEGATE, None

        try:
            vertices = np.asarray(layer.data[row_index], dtype=float).copy()
        except (TypeError, ValueError):
            self._warn(_ALREADY_INVALID_POLYGON_DRAG_WARNING)
            return _PolygonVertexDragRoute.REJECT, None
        if moved_vertex_index >= len(vertices):
            self._warn(_ALREADY_INVALID_POLYGON_DRAG_WARNING)
            return _PolygonVertexDragRoute.REJECT, None

        try:
            topology = napari_polygon_vertices_to_topology(vertices)
        except ValueError:
            self._warn(_ALREADY_INVALID_POLYGON_DRAG_WARNING)
            return _PolygonVertexDragRoute.REJECT, None

        try:
            _ = napari_polygon_vertices_to_shapely_polygon(vertices)
        except ValueError:
            self._warn(_ALREADY_INVALID_POLYGON_DRAG_WARNING)
            return _PolygonVertexDragRoute.REJECT, None

        return (
            _PolygonVertexDragRoute.GUARD,
            _PolygonVertexDragState(
                row_index=row_index,
                moved_vertex_index=moved_vertex_index,
                topology=topology,
                last_valid_vertices=vertices,
            ),
        )

    def _apply_polygon_vertex_drag_move(
        self,
        layer: Shapes,
        active_drag: _PolygonVertexDragState,
        event: object,
    ) -> None:
        """Apply one polygon move as a prevalidated rendering transaction.

        Build and validate a candidate with all encoded vertex aliases already
        synchronized, so an invalid or unsynchronized row never reaches the
        live layer. Applying that valid candidate can still fail inside napari,
        particularly during triangulation, and may partially mutate its state.
        In that case, restore the last accepted row. Only record the candidate
        as accepted after both the edit and refresh succeed.
        """
        try:
            candidate_vertices, moved_coordinate = self._build_polygon_vertex_drag_candidate(layer, active_drag, event)
        except (AttributeError, IndexError, TypeError, ValueError):
            self._warn_polygon_drag_once(active_drag, _INVALID_POLYGON_DRAG_WARNING)
            return

        if np.array_equal(candidate_vertices, active_drag.last_valid_vertices):
            return

        baseline = active_drag.last_valid_vertices.copy()
        layer._moving_coordinates = moved_coordinate
        try:
            layer._data_view.edit(active_drag.row_index, candidate_vertices)
            layer.refresh(thumbnail=False)
        except Exception as application_error:  # noqa: BLE001 - transaction boundary
            try:
                # Napari may have partially written the candidate before edit,
                # triangulation, or refresh raised. Restore the cached accepted
                # row so every encoded alias is synchronized and failed
                # rendering never leaves malformed live polygon data behind.
                self._restore_polygon_drag_vertices(layer, active_drag.row_index, baseline)
            except Exception as restoration_error:  # noqa: BLE001 - retain both failures
                raise ExceptionGroup(
                    "Polygon move failed and restoring the previous accepted row also failed.",
                    [application_error, restoration_error],
                ) from application_error
            self._warn_polygon_drag_once(active_drag, _POLYGON_DRAG_RENDERING_WARNING)
            return

        active_drag.last_valid_vertices = candidate_vertices.copy()
        active_drag.has_accepted_move = True
        layer._is_moving = True

    def _build_polygon_vertex_drag_candidate(
        self,
        layer: Shapes,
        active_drag: _PolygonVertexDragState,
        event: object,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build a synchronized, valid candidate without mutating the layer.

        Convert the event position to one data-space coordinate, then delegate
        ordinary movement, alias synchronization, and geometry validation to
        ``move_napari_polygon_vertex(...)``. Return the validated row together
        with the coordinate needed for napari's interaction state.
        """
        moved_coordinate = np.asarray(layer.world_to_data(event.position), dtype=float)
        candidate_vertices = move_napari_polygon_vertex(
            active_drag.last_valid_vertices,
            active_drag.topology,
            active_drag.moved_vertex_index,
            moved_coordinate,
        )
        return candidate_vertices, moved_coordinate

    def _restore_polygon_drag_vertices(
        self,
        layer: Shapes,
        row_index: int,
        vertices: np.ndarray,
    ) -> None:
        layer._data_view.edit(row_index, vertices)
        layer.refresh(thumbnail=False)

    def _warn_polygon_drag_once(self, active_drag: _PolygonVertexDragState, message: str) -> None:
        if active_drag.warning_emitted:
            return
        self._warn(message)
        active_drag.warning_emitted = True

    def _finish_polygon_vertex_drag(
        self,
        layer: Shapes,
        active_drag: _PolygonVertexDragState | None,
        *,
        emit_completed: bool,
    ) -> None:
        """Finish a Harpy-owned drag in place of napari's suspended generator.

        Always clear napari's temporary interaction state and restore its
        highlight. After a normal release, emit the native ``CHANGED`` event
        and update the thumbnail only if at least one candidate was accepted.
        A drag whose candidates were all rejected or rolled back therefore
        performs cleanup without notifying listeners of a data change.
        """
        has_accepted_move = active_drag is not None and active_drag.has_accepted_move
        if emit_completed and has_accepted_move:
            data_indices = tuple(layer.selected_data)
            vertex_indices = tuple(tuple(range(len(layer.data[index]))) for index in data_indices)
            layer.events.data(
                value=layer.data,
                action=ActionType.CHANGED,
                data_indices=data_indices,
                vertex_indices=vertex_indices,
            )

        layer._is_moving = False
        layer._drag_start = None
        layer._drag_box = None
        layer._fixed_vertex = None
        layer._moving_value = (None, None)
        layer._set_highlight()

        if emit_completed and has_accepted_move:
            layer._update_thumbnail()
        if self._polygon_edit_finished_callback is not None:
            self._polygon_edit_finished_callback()

    def _route_vertex_insert(
        self,
        layer: Shapes,
        original_vertex_insert_callback: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        event: object,
    ) -> Any:
        """Route and transactionally apply one polygon-vertex insertion.

        Candidate validation happens before the live layer is changed. Because
        insertion lengthens a row and rebuilds napari's complete public data
        and derived vertex caches, capture the shared row-change baseline and
        restore it if rebuild, triangulation, or final refresh raises.
        """
        route, insert_state = self._classify_vertex_insert(layer, event)
        if route is _PolygonVertexInsertRoute.DELEGATE:
            return original_vertex_insert_callback(*args, **kwargs)
        if route is _PolygonVertexInsertRoute.REJECT:
            return None
        if insert_state is None:  # pragma: no cover - routing invariant
            raise RuntimeError("Guarded polygon insertion has no captured state.")

        try:
            candidate_vertices, _ = insert_napari_polygon_vertex(
                insert_state.vertices,
                insert_state.topology,
                insert_state.insert_index,
                insert_state.inserted_coordinate,
            )
        except ValueError:
            self._warn(_INVALID_POLYGON_DRAG_WARNING)
            return None

        baseline = _capture_shapes_layer_baseline(layer)

        # Mirror napari's native `vertex_insert(...)` event contract, but only
        # after the complete candidate has passed topology and geometry checks.
        layer.events.data(
            value=layer.data,
            action=ActionType.CHANGING,
            data_indices=(insert_state.row_index,),
            vertex_indices=((insert_state.insert_index,),),
        )
        try:
            self._replace_shape_row_preserving_layer_state(
                layer,
                insert_state.row_index,
                candidate_vertices,
            )
            layer.refresh()
        except Exception as application_error:  # noqa: BLE001 - transaction boundary
            try:
                # Roll back a partially committed longer row after rebuild,
                # triangulation, refresh, or another rendering step raises.
                _restore_shapes_layer_baseline(layer, baseline)
            except Exception as restoration_error:  # noqa: BLE001 - retain both failures
                raise ExceptionGroup(
                    "Polygon insertion failed and restoring the previous layer state also failed.",
                    [application_error, restoration_error],
                ) from application_error
            self._warn(_POLYGON_INSERT_RENDERING_WARNING)
            return None

        layer.events.data(
            value=layer.data,
            action=ActionType.CHANGED,
            data_indices=(insert_state.row_index,),
            vertex_indices=((insert_state.insert_index,),),
        )

        if self._polygon_edit_finished_callback is not None:
            self._polygon_edit_finished_callback()
        return None

    def _classify_vertex_insert(
        self,
        layer: Shapes,
        event: object,
    ) -> tuple[_PolygonVertexInsertRoute, _VertexInsertState | None]:
        """Select napari's nearest raw edge before applying polygon rules."""
        all_edges = np.empty((0, 2, 2))
        all_edge_targets = np.empty((0, 2), dtype=int)
        try:
            for selected_index in layer.selected_data:
                if isinstance(selected_index, (bool, np.bool_)) or not isinstance(
                    selected_index,
                    (int, np.integer),
                ):
                    raise ValueError("Selected shape index is not an integer.")
                row_index = int(selected_index)
                if row_index < 0 or row_index >= len(layer._data_view.shapes):
                    raise IndexError("Selected shape index is outside the layer.")

                shape_type = type(layer._data_view.shapes[row_index])
                if shape_type is Ellipse:
                    continue
                vertices = layer._data_view.displayed_vertices[
                    layer._data_view.displayed_vertices_to_shape_num == row_index
                ]
                closed = shape_type is not Path
                vertex_count = len(vertices)
                if closed:
                    edges = np.asarray(
                        [[vertices[index], vertices[(index + 1) % vertex_count]] for index in range(vertex_count)]
                    )
                else:
                    edges = np.asarray([[vertices[index], vertices[index + 1]] for index in range(vertex_count - 1)])
                if len(edges) == 0:
                    continue
                all_edges = np.append(all_edges, edges, axis=0)
                edge_targets = np.asarray(
                    [np.repeat(row_index, len(edges)), np.arange(len(edges))],
                    dtype=int,
                ).T
                all_edge_targets = np.append(all_edge_targets, edge_targets, axis=0)
        except (AttributeError, IndexError, TypeError, ValueError):
            self._warn(_ALREADY_INVALID_POLYGON_DRAG_WARNING)
            return _PolygonVertexInsertRoute.REJECT, None

        if len(all_edges) == 0:
            return _PolygonVertexInsertRoute.DELEGATE, None

        try:
            inserted_coordinate = np.asarray(
                layer.world_to_data(event.position),
                dtype=float,
            )
            displayed_coordinate = inserted_coordinate[list(layer._slice_input.displayed)]
            if (
                inserted_coordinate.shape != (2,)
                or displayed_coordinate.shape != (2,)
                or not np.isfinite(inserted_coordinate).all()
            ):
                raise ValueError("Inserted polygon coordinate must be finite and 2D.")
            nearest_edge_index, _ = point_to_lines(displayed_coordinate, all_edges)
            row_index = int(all_edge_targets[nearest_edge_index, 0])
            insert_index = int(all_edge_targets[nearest_edge_index, 1]) + 1
        except (AttributeError, IndexError, TypeError, ValueError):
            self._warn(_ALREADY_INVALID_POLYGON_DRAG_WARNING)
            return _PolygonVertexInsertRoute.REJECT, None

        shape_type = _shape_type_at(layer, row_index)
        if shape_type is None:
            self._warn(_ALREADY_INVALID_POLYGON_DRAG_WARNING)
            return _PolygonVertexInsertRoute.REJECT, None
        if shape_type != "polygon":
            return _PolygonVertexInsertRoute.DELEGATE, None

        try:
            vertices = np.asarray(layer.data[row_index], dtype=float).copy()
        except (IndexError, TypeError, ValueError):
            self._warn(_ALREADY_INVALID_POLYGON_DRAG_WARNING)
            return _PolygonVertexInsertRoute.REJECT, None
        if vertices.ndim != 2 or vertices.shape[1:] != (2,) or not np.isfinite(vertices).all():
            self._warn(_ALREADY_INVALID_POLYGON_DRAG_WARNING)
            return _PolygonVertexInsertRoute.REJECT, None

        try:
            topology = napari_polygon_vertices_to_topology(vertices)
            _ = napari_polygon_vertices_to_shapely_polygon(vertices)
        except ValueError:
            self._warn(_ALREADY_INVALID_POLYGON_DRAG_WARNING)
            return _PolygonVertexInsertRoute.REJECT, None

        return (
            _PolygonVertexInsertRoute.GUARD,
            _VertexInsertState(
                row_index=row_index,
                insert_index=insert_index,
                vertices=vertices,
                topology=topology,
                inserted_coordinate=inserted_coordinate,
            ),
        )

    def _route_vertex_remove(
        self,
        layer: Shapes,
        original_vertex_remove_callback: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        event: object,
    ) -> Any:
        """Route and transactionally apply one vertex-removal gesture.

        Like ``_apply_polygon_vertex_drag_move(...)``, restore the previously
        accepted state if applying or rendering a valid candidate fails. A
        deletion rollback is broader than a movement rollback: shortening a
        row or removing a shape can shift geometry, features, styles,
        selection, and derived vertex caches. Capture the complete
        ``_ShapesLayerBaseline`` instead of movement's single-row
        vertex baseline so all of that row-aligned state can be restored.
        """
        route, delete_state = self._classify_vertex_delete(layer, event)
        if route is _PolygonVertexDeleteRoute.DELEGATE:
            return original_vertex_remove_callback(*args, **kwargs)
        if route is _PolygonVertexDeleteRoute.REJECT:
            return None
        if delete_state is None:  # pragma: no cover - routing invariant
            raise RuntimeError("Guarded polygon deletion has no captured state.")

        try:
            candidate = delete_napari_polygon_vertex(
                delete_state.vertices,
                delete_state.topology,
                delete_state.deleted_vertex_index,
            )
        except ValueError as error:
            self._warn(str(error))
            return None

        baseline = _capture_shapes_layer_baseline(layer)

        # Mirror napari's native `vertex_remove(...)` event contract around
        # our topology-preserving edit path.
        layer.events.data(
            value=layer.data,
            action=ActionType.CHANGING,
            data_indices=(delete_state.row_index,),
            vertex_indices=((delete_state.deleted_vertex_index,),),
        )

        try:
            if candidate.removes_shape:
                self._remove_shape_row_preserving_layer_state(layer, delete_state.row_index)
            else:
                if candidate.vertices is None:  # pragma: no cover - candidate invariant
                    raise RuntimeError("Shortened polygon deletion has no candidate vertices.")
                self._replace_shape_row_preserving_layer_state(
                    layer,
                    delete_state.row_index,
                    candidate.vertices,
                )

            layer.refresh()
        except Exception as application_error:  # noqa: BLE001 - transaction boundary
            try:
                # Roll back any partial deletion commit after triangulation,
                # refresh, or another rendering/application step raises.
                _restore_shapes_layer_baseline(layer, baseline)
            except Exception as restoration_error:  # noqa: BLE001 - retain both failures
                raise ExceptionGroup(
                    "Polygon deletion failed and restoring the previous layer state also failed.",
                    [application_error, restoration_error],
                ) from application_error
            self._warn(_POLYGON_DELETE_RENDERING_WARNING)
            return None

        layer.events.data(
            value=layer.data,
            action=ActionType.CHANGED,
            data_indices=(delete_state.row_index,),
            vertex_indices=((delete_state.deleted_vertex_index,),),
        )
        if self._polygon_edit_finished_callback is not None:
            # Without this reset, a previous rejected-delete warning would
            # keep showing in the status card even after this successful
            # guarded delete.
            self._polygon_edit_finished_callback()
        return None

    def _replace_shape_row_preserving_layer_state(
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
        # the guarded row-length-changing edit, so block intermediate events
        # triggered by `layer.data = rebuilt_data`.
        with layer.events.data.blocker(), layer.events.features.blocker():
            # Rebuild napari's private vertex cache after row-length-changing
            # edits. Low-level `_data_view.edit(...)` can leave old clickable
            # boundaries behind after deletion and does not provide the public
            # rebuild needed by guarded insertion.
            ensure_shapes_triangulation_backend()
            layer.data = rebuilt_data

        layer.opacity = style_snapshot.opacity
        layer.mode = current_mode
        layer.selected_data = {index for index in selected_data if index < len(layer.data)}

        # Restore current draw defaults without emitting Harpy's style sync
        # callbacks, then reapply row styles last so callback side effects
        # cannot overwrite the final styling.
        _restore_shapes_layer_current_style(layer, style_snapshot)
        _restore_shapes_layer_row_styles(layer, style_snapshot, row_indices=range(len(layer.data)))

    def _remove_shape_row_preserving_layer_state(self, layer: Shapes, row_index: int) -> None:
        """Remove one polygon row while preserving all surviving layer state."""
        row_count = len(layer.data)
        current_mode = layer.mode
        selected_data = set(layer.selected_data)
        style_snapshot = _capture_shapes_layer_style(layer, row_count=row_count)
        surviving_row_indices = [index for index in range(row_count) if index != row_index]

        _trim_stale_private_color_rows_before_rebuild(layer, style_snapshot)
        with layer.events.data.blocker(), layer.events.features.blocker():
            layer.remove([row_index])

        layer.opacity = style_snapshot.opacity
        layer.mode = current_mode
        layer.selected_data = {
            index if index < row_index else index - 1 for index in selected_data if index != row_index
        }
        _restore_shapes_layer_current_style(layer, style_snapshot)
        _restore_shapes_layer_row_styles(
            layer,
            style_snapshot,
            row_indices=surviving_row_indices,
        )

    def _classify_vertex_delete(
        self,
        layer: Shapes,
        event: object,
    ) -> tuple[_PolygonVertexDeleteRoute, _VertexDeleteState | None]:
        """Classify one vertex-removal click before allowing any mutation."""
        try:
            value = layer.get_value(getattr(event, "position", None), world=True)
        except (AttributeError, TypeError, ValueError, IndexError):
            self._warn(_ALREADY_INVALID_POLYGON_DRAG_WARNING)
            return _PolygonVertexDeleteRoute.REJECT, None
        if value is None:
            return _PolygonVertexDeleteRoute.DELEGATE, None
        if not isinstance(value, tuple) or len(value) != 2:
            self._warn(_ALREADY_INVALID_POLYGON_DRAG_WARNING)
            return _PolygonVertexDeleteRoute.REJECT, None

        row_index, deleted_vertex_index = value
        if deleted_vertex_index is None:
            return _PolygonVertexDeleteRoute.DELEGATE, None
        if (
            isinstance(row_index, (bool, np.bool_))
            or isinstance(deleted_vertex_index, (bool, np.bool_))
            or not isinstance(row_index, (int, np.integer))
            or not isinstance(deleted_vertex_index, (int, np.integer))
        ):
            self._warn(_ALREADY_INVALID_POLYGON_DRAG_WARNING)
            return _PolygonVertexDeleteRoute.REJECT, None

        row_index = int(row_index)
        deleted_vertex_index = int(deleted_vertex_index)
        if row_index < 0 or deleted_vertex_index < 0 or row_index >= len(layer.data):
            self._warn(_ALREADY_INVALID_POLYGON_DRAG_WARNING)
            return _PolygonVertexDeleteRoute.REJECT, None
        shape_type = _shape_type_at(layer, row_index)
        if shape_type is None:
            self._warn(_ALREADY_INVALID_POLYGON_DRAG_WARNING)
            return _PolygonVertexDeleteRoute.REJECT, None
        if shape_type != "polygon":
            return _PolygonVertexDeleteRoute.DELEGATE, None

        try:
            vertices = np.asarray(layer.data[row_index], dtype=float).copy()
        except (TypeError, ValueError):
            self._warn(_ALREADY_INVALID_POLYGON_DRAG_WARNING)
            return _PolygonVertexDeleteRoute.REJECT, None
        if vertices.ndim != 2 or not np.isfinite(vertices).all() or deleted_vertex_index >= len(vertices):
            self._warn(_ALREADY_INVALID_POLYGON_DRAG_WARNING)
            return _PolygonVertexDeleteRoute.REJECT, None

        try:
            topology = napari_polygon_vertices_to_topology(vertices)
        except ValueError:
            self._warn(_ALREADY_INVALID_POLYGON_DRAG_WARNING)
            return _PolygonVertexDeleteRoute.REJECT, None

        try:
            _ = napari_polygon_vertices_to_shapely_polygon(vertices)
        except ValueError:
            self._warn(_ALREADY_INVALID_POLYGON_DRAG_WARNING)
            return _PolygonVertexDeleteRoute.REJECT, None

        return (
            _PolygonVertexDeleteRoute.GUARD,
            _VertexDeleteState(
                row_index=row_index,
                deleted_vertex_index=deleted_vertex_index,
                vertices=vertices,
                topology=topology,
            ),
        )

    def _warn(self, message: str) -> None:
        if self._warning_callback is not None:
            self._warning_callback(message)
