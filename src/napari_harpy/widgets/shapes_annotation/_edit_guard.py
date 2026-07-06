"""Private napari edit-hook adapter for the shapes annotation widget."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
from napari.layers import Shapes
from napari.layers.base._base_constants import ActionType
from napari.layers.shapes._shapes_constants import Mode
from napari.utils.key_bindings import coerce_keybinding

from napari_harpy._shapes_triangulation import ensure_shapes_triangulation_backend
from napari_harpy.core.shapes_geometry import (
    NapariPolygonTopology,
    delete_napari_polygon_vertex,
    napari_polygon_vertices_to_shapely_polygon,
    napari_polygon_vertices_to_topology,
    sync_napari_polygon_anchor_vertex,
)
from napari_harpy.widgets.shapes_annotation._layer_style import (
    _capture_shapes_layer_style,
    _restore_shapes_layer_current_style,
    _restore_shapes_layer_row_styles,
    _trim_stale_private_color_rows_before_rebuild,
)

_SPACE_KEYBINDING = coerce_keybinding("Space")
_INVALID_POLYGON_DRAG_WARNING = "Polygon edit was rejected because it would create invalid geometry."
_ALREADY_INVALID_POLYGON_DRAG_WARNING = "This polygon is already invalid, so this edit cannot be safely rolled back."
_SPACE_PAN_RESUMABLE_DRAW_MODES = frozenset(
    {
        Mode.ADD_POLYGON_LASSO,
        Mode.ADD_PATH,
        Mode.ADD_POLYGON,
        Mode.ADD_POLYLINE,
    }
)


@dataclass
class _PolygonVertexDragState:
    row_index: int
    moved_vertex_index: int
    topology: NapariPolygonTopology
    last_valid_vertices: np.ndarray
    warned_invalid_drag: bool = False


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
        polygon_vertex_drag_finished_callback: Callable[[], None] | None = None,
        polygon_vertex_delete_finished_callback: Callable[[], None] | None = None,
        can_space_pan_draw: Callable[[Shapes], bool] | None = None,
    ) -> None:
        self._layer: Shapes | None = None
        self._original_drag_modes: dict[object, Callable[..., Any]] | None = None
        self._original_move_modes: dict[object, Callable[..., Any]] | None = None
        self._had_instance_drag_modes = False
        self._had_instance_move_modes = False
        self._wrapped_direct_callback: Callable[..., Any] | None = None
        self._wrapped_vertex_remove_callback: Callable[..., Any] | None = None
        self._wrapped_space_keybinding: Callable[..., Any] | None = None
        self._is_applying_polygon_drag_edit = False
        # Track Space and mouse release independently so drawing resumes only
        # after both halves of a temporary Space-pan have ended.
        self._space_pan_key_held = False
        self._space_pan_mouse_gesture_active = False
        self._previous_mouse_pan: bool | None = None
        self._previous_space_keybinding: object | None = None
        self._had_instance_space_keybinding = False
        self._warning_callback = warning_callback
        self._polygon_vertex_drag_finished_callback = polygon_vertex_drag_finished_callback
        self._polygon_vertex_delete_finished_callback = polygon_vertex_delete_finished_callback
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
            or not _SPACE_PAN_RESUMABLE_DRAW_MODES.issubset(drag_modes)
            or not _SPACE_PAN_RESUMABLE_DRAW_MODES.issubset(move_modes)
        ):
            raise ValueError("Shapes layer does not expose napari annotation edit mode hooks.")

        original_direct_callback = drag_modes[Mode.DIRECT]
        original_vertex_remove_callback = drag_modes[Mode.VERTEX_REMOVE]

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

        def wrapped_space_keybinding(bound_layer: Shapes) -> Iterator[None]:
            yield from self._handle_space_keybinding(bound_layer)

        patched_drag_modes = dict(drag_modes)
        patched_drag_modes[Mode.DIRECT] = wrapped_direct_callback
        patched_drag_modes[Mode.VERTEX_REMOVE] = wrapped_vertex_remove_callback
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
        self._wrapped_space_keybinding = wrapped_space_keybinding
        layer._drag_modes = patched_drag_modes
        layer._move_modes = patched_move_modes
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
        self._wrapped_space_keybinding = None
        self._is_applying_polygon_drag_edit = False
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
        if (
            self._layer is layer
            and self._can_space_pan_draw_mode(layer)
            and self._can_use_custom_space_pan(layer)
        ):
            # Custom resumable drawing path: once Slice 4 enables callback
            # suppression, active supported drawings use Space to toggle
            # mouse-pan without leaving the current drawing mode.
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
        """Mirror napari's direct-drag generator while validating polygon rows.

        The first ``next(direct_drag)`` runs napari's mouse-press setup and
        pauses at its first ``yield``. At that point napari has populated
        ``layer._moving_value`` but has not moved any vertex yet, so after a
        mouse-press step we can safely cache the current valid polygon row.
        Later ``next(...)`` calls let napari process mouse moves first; after
        each mouse move we optionally synchronize duplicated anchor/separator
        vertices, then validate or roll back the candidate row.

        The cached drag state lives only for this one generator/gesture. When
        mouse release ends the generator, the state is discarded, so per-drag
        flags such as ``warned_invalid_drag`` reset naturally on the next mouse
        press.
        """
        active_drag: _PolygonVertexDragState | None = None
        try:
            try:
                yielded = next(direct_drag)
            except StopIteration:
                return

            # The normal napari path reaches this point from mouse press. Use
            # `event.type` directly so a broken mouse-event contract fails
            # loudly instead of silently skipping validation setup.
            if event.type == "mouse_press":
                active_drag = self._capture_polygon_vertex_drag_state(layer)
            yield yielded

            while True:
                try:
                    yielded = next(direct_drag)
                except StopIteration:
                    return

                if event.type == "mouse_move":
                    # Validate every mouse move, not only release, so invalid
                    # polygon edits are rolled back immediately.
                    self._validate_polygon_vertex_drag(layer, active_drag)
                yield yielded
        finally:
            direct_drag.close()
            if active_drag is not None and self._polygon_vertex_drag_finished_callback is not None:
                # The guarded drag may have shown a transient rollback
                # warning; once the gesture ends, let the widget restore its
                # normal annotation status card.
                self._polygon_vertex_drag_finished_callback()

    def _capture_polygon_vertex_drag_state(self, layer: Shapes) -> _PolygonVertexDragState | None:
        moving_value = layer._moving_value
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
            vertices = np.asarray(layer.data[row_index], dtype=float).copy()
        except (TypeError, ValueError):
            return None
        if moved_vertex_index >= len(vertices):
            return None

        try:
            topology = napari_polygon_vertices_to_topology(vertices)
        except ValueError:
            # Malformed path encoding means Harpy cannot reliably identify
            # topology/anchors for this drag, so we fall back to napari's
            # native direct-drag behavior.
            return None

        try:
            # Capture only starts from valid polygon rows; this establishes the
            # baseline that later mouse moves can roll back to. If the napari
            # vertices cannot convert to a Shapely polygon, there is no valid
            # baseline for this guard, so the ValueError path returns None and
            # falls back to napari's native direct-drag behavior.
            _ = napari_polygon_vertices_to_shapely_polygon(vertices)
        except ValueError:
            self._warn(_ALREADY_INVALID_POLYGON_DRAG_WARNING)
            return None

        return _PolygonVertexDragState(
            row_index=row_index,
            moved_vertex_index=moved_vertex_index,
            topology=topology,
            last_valid_vertices=vertices,
        )

    def _validate_polygon_vertex_drag(
        self,
        layer: Shapes,
        active_drag: _PolygonVertexDragState | None,
    ) -> None:
        if active_drag is None or self._is_applying_polygon_drag_edit:
            return
        if active_drag.row_index >= len(layer.data):
            return

        vertices = np.asarray(layer.data[active_drag.row_index], dtype=float)
        if active_drag.moved_vertex_index >= len(vertices):
            return

        candidate_vertices = vertices
        is_synchronized_anchor_vertex = any(
            active_drag.moved_vertex_index in group for group in active_drag.topology.synchronized_anchor_groups
        )
        if is_synchronized_anchor_vertex:
            moved_coordinate = candidate_vertices[active_drag.moved_vertex_index]
            try:
                candidate_vertices = sync_napari_polygon_anchor_vertex(
                    candidate_vertices,
                    active_drag.topology,
                    active_drag.moved_vertex_index,
                    moved_coordinate,
                )
            except ValueError:
                return

        # This Shapely conversion intentionally runs on every mouse move for
        # the active polygon row. Rejecting invalid geometry immediately keeps
        # `layer.data` out of an invalid intermediate state, and avoids possible
        # broken triangulation, instead of waiting until mouse release.
        try:
            _ = napari_polygon_vertices_to_shapely_polygon(candidate_vertices)
        except ValueError:
            self._restore_polygon_drag_last_valid_vertices(layer, active_drag)
            if not active_drag.warned_invalid_drag:
                self._warn(_INVALID_POLYGON_DRAG_WARNING)
                active_drag.warned_invalid_drag = True
            return

        if is_synchronized_anchor_vertex and not np.array_equal(vertices, candidate_vertices):
            self._is_applying_polygon_drag_edit = True
            try:
                # Mirror napari's direct-drag write path. Anchor synchronization
                # only changes coordinates, so the row length and vertex cache
                # stay stable.
                layer._data_view.edit(active_drag.row_index, candidate_vertices)
                layer.refresh()
            finally:
                self._is_applying_polygon_drag_edit = False

        active_drag.last_valid_vertices = np.asarray(candidate_vertices, dtype=float).copy()

    def _restore_polygon_drag_last_valid_vertices(
        self,
        layer: Shapes,
        active_drag: _PolygonVertexDragState,
    ) -> None:
        self._is_applying_polygon_drag_edit = True
        try:
            layer._data_view.edit(active_drag.row_index, active_drag.last_valid_vertices)
            layer.refresh()
        finally:
            self._is_applying_polygon_drag_edit = False

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
        if self._polygon_vertex_delete_finished_callback is not None:
            # Without this reset, a previous rejected-delete warning would
            # keep showing in the status card even after this successful
            # guarded delete.
            self._polygon_vertex_delete_finished_callback()
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
            ensure_shapes_triangulation_backend()
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
