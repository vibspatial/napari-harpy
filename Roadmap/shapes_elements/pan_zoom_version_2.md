# Pan And Zoom While Drawing Shapes: Version 2

Status: investigation

This note updates `Roadmap/shapes_elements/pan_zoom.md` after a closer look at
the annotation widget, napari's Shapes mode lifecycle, and the installed napari
0.7.0 mouse/key handling.

## Summary

The current lasso interruption is caused by napari's default Space shortcut.
Space temporarily switches the active layer to `pan_zoom`. For a Shapes layer,
changing mode while a shape is being created calls `_finish_drawing()`, so the
in-progress lasso is finalized or cancelled before the user can continue.

A true upstream-quality "pause lasso, pan, continue the same lasso" workflow
still belongs in napari because it needs cooperation between the canvas, camera,
mouse event stream, and layer creation state. However, napari-harpy can likely
provide a scoped annotation-widget workaround:

- intercept Space only on the widget-owned annotation Shapes layer;
- do not set `layer.mode = Mode.PAN_ZOOM` while a lasso is active;
- temporarily set `layer.mouse_pan = True`;
- suppress the annotation layer's lasso mouse callbacks while Space-pan is
  active;
- restore the previous mouse-pan state on Space release and continue the same
  in-progress lasso.

This keeps the implementation focused on the user's expected Space-pan workflow
and remains contained if implemented through the annotation layer guard.

## Root Cause

Napari's viewer Space shortcut is implemented by
`.venv/lib/python3.13/site-packages/napari/components/_viewer_key_bindings.py`:

```python
def hold_for_pan_zoom(viewer):
    selected_layer = viewer.layers.selection.active
    ...
    previous_mode = selected_layer.mode
    pan_zoom = selected_layer._modeclass.PAN_ZOOM
    if previous_mode != pan_zoom:
        selected_layer.mode = pan_zoom
        yield
        selected_layer.mode = previous_mode
```

For Shapes, `layer.mode = pan_zoom` routes through
`.venv/lib/python3.13/site-packages/napari/layers/shapes/shapes.py`. The mode
setter contains:

```python
if self._is_creating:
    with self.block_thumbnail_update():
        self._finish_drawing()
```

That means the lasso is not being stopped by the Harpy annotation widget
directly. It is stopped by napari's normal mode-switch behavior.

The lasso itself is handled in
`.venv/lib/python3.13/site-packages/napari/layers/shapes/_shapes_mouse_bindings.py`:

- `Mode.ADD_POLYGON_LASSO` uses `add_path_polygon_lasso`;
- while drawing, mouse moves call `polygon_creating`;
- if `_is_creating` is already true, a lasso mouse press finishes drawing;
- while moving, `polygon_creating` keeps the active vertex under the cursor and
  adds vertices after enough screen-pixel movement.

So preventing the mode switch is necessary, but not sufficient. If Space enables
camera panning and the layer still receives normal lasso mouse events, the lasso
continues to draw during the pan or a mouse press can finish it.

## Local Confirmation

A small headless simulation in the repository environment confirmed the key
split:

- setting `layer.mouse_pan = True` does not finish an active lasso;
- setting `layer.mode = Mode.PAN_ZOOM` does finish the active lasso;
- while `layer.mouse_pan = True`, lasso mouse callbacks still add vertices
  unless they are explicitly suppressed.

This makes `mouse_pan` the useful lever. The implementation should avoid mode
changes during an active lasso and should gate lasso callbacks during the
temporary pan state.

## Existing Harpy Fit

The annotation widget already has a good extension point:
`_AnnotationLayerEditGuard` in
`src/napari_harpy/widgets/shapes_annotation/widget.py`.

Today it:

- attaches to every annotation-owned Shapes layer;
- creates an instance-local copy of napari's `_drag_modes`;
- wraps `Mode.DIRECT` for hole-anchor synchronization;
- wraps `Mode.VERTEX_REMOVE` for hole-aware vertex deletion;
- restores the previous instance/class mapping on disconnect.

The guard is attached on all relevant entry paths:

- opening or adopting an existing annotation layer;
- creating a new annotation layer;
- adopting an active primary Shapes layer.

It is disconnected through `_clear_annotation_state(...)`. That lifecycle is a
good place to keep any Space-pan changes scoped to annotation-owned layers.

`ShapesAnnotation._annotation_layer_binding_matches()` is also already available
to prove that the tracked layer is still the widget-owned primary Shapes layer
for the locked SpatialData target. That should be part of the behavior gate if
the final implementation needs widget-level context.

## Recommended Design

Add a lasso-aware Space-pan path to the annotation layer guard.

### Space Keybinding

When the guard attaches, install an instance-level `Space` keybinding on the
annotation layer. Store and restore any previous instance binding so user or
plugin customizations on that layer are not lost.

The keybinding should behave like this:

1. If the guarded layer is actively creating a polygon lasso:
   - mark `space_pan_active = True`;
   - remember the previous `layer.mouse_pan` value;
   - set `layer.mouse_pan = True`;
   - yield until Space release;
   - restore the previous `layer.mouse_pan`;
   - mark `space_pan_active = False`.
2. Otherwise, delegate to napari-equivalent Space behavior:
   - store the current mode;
   - switch to `PAN_ZOOM`;
   - yield;
   - restore the previous mode.

The custom branch must not set `layer.mode = Mode.PAN_ZOOM` while
`layer._is_creating` is true in `Mode.ADD_POLYGON_LASSO`.

Napari's Qt viewer puts the active layer before the viewer in the keymap
provider chain, so a layer instance binding should win over the viewer Space
shortcut when the annotation layer is active. User-level keymap overrides still
win, which is acceptable.

### Lasso Callback Suppression

The guard should also create an instance-local copy of `_move_modes`, not only
`_drag_modes`.

Wrap:

- `_drag_modes[Mode.ADD_POLYGON_LASSO]`;
- `_move_modes[Mode.ADD_POLYGON_LASSO]`.

When `space_pan_active` is true and the layer is actively creating a lasso:

- the lasso move callback should no-op, so pointer movement during Space-pan
  does not add lasso vertices;
- the lasso drag callback should no-op on mouse press, so dragging the canvas
  while Space is held does not finish the lasso.

When `space_pan_active` is false, both wrappers should delegate to napari's
original callbacks.

If the annotation layer is already in lasso mode when the guard attaches, the
implementation may also need to update `layer.mouse_drag_callbacks` and
`layer.mouse_move_callbacks` so the active callback lists point at the wrappers,
not the original functions.

### Behavior Contract

The custom Space-pan path should only run when all of these are true:

- the guarded layer is still attached;
- the layer is a napari `Shapes` layer;
- `layer._mode == Mode.ADD_POLYGON_LASSO`;
- `layer._is_creating is True`;
- the layer is active in the viewer, if viewer context is available;
- the layer still matches the widget-owned annotation binding, if widget
  context is available.

It should not:

- call `layer._finish_drawing()`;
- change `layer.mode`;
- add or remove shapes;
- save shapes;
- affect styled Shapes layers or non-annotation Shapes layers;
- change behavior for other layers in the viewer.

## Expected UX

Target workflow:

1. User selects the annotation Shapes layer.
2. User selects napari's polygon lasso tool.
3. User clicks to start a lasso.
4. User holds Space.
5. User drags the canvas to pan.
6. User releases Space.
7. User continues the same lasso and finishes it normally.

Mouse-wheel zoom should continue to be left to napari unless testing shows a
separate issue.

## Known Limitations

This is still a workaround.

Changing the camera while a lasso is active changes the data coordinate under
the cursor. When drawing resumes, the next lasso move may connect the last
pre-pan coordinate to the new post-pan coordinate. That can create a long edge
after large pans. Small pans should be much less risky, but manual QA is needed.

The harder case is tablet-style lasso drawing where the mouse button is already
held down when Space is pressed. A first implementation can focus on the common
mouse-draw workflow: click to start, move without holding the mouse button,
hold Space and drag to pan, release Space, then continue drawing. Supporting
mid-drag pause perfectly may require wrapping an already-running lasso drag
generator and deciding how release events should behave while paused.

## Suggested Implementation Slices

### Slice 1: Guard Plumbing

Extend `_AnnotationLayerEditGuard` to store and restore:

- instance `_drag_modes`;
- instance `_move_modes`;
- the previous instance `Space` keybinding;
- `space_pan_active`;
- the previous `mouse_pan` value.

Add tests that attaching and disconnecting the guard restores all mappings and
keybindings.

### Slice 2: Space Keybinding State

Add the guarded layer Space keybinding.

Headless tests should prove:

- active lasso Space press sets `space_pan_active`;
- active lasso Space press sets `layer.mouse_pan = True`;
- Space release restores the previous mouse-pan value;
- active lasso Space does not change `layer.mode`;
- non-lasso Space still behaves like normal temporary pan-zoom.

### Slice 3: Lasso Callback Suppression

Wrap lasso drag and move callbacks.

Headless tests should prove:

- while `space_pan_active` is true, lasso move callback does not add vertices;
- while `space_pan_active` is true, lasso drag callback does not finish the
  active lasso;
- after Space release, original lasso callbacks resume;
- layer data, selected rows, and current mode are preserved.

### Slice 4: Widget Gating

If the guard needs widget context, pass a small predicate into the guard, for
example `can_space_pan_lasso()`. It can reuse
`_annotation_layer_binding_matches()` and the viewer active-layer check.

Tests should prove the custom behavior no-ops when:

- the annotation layer is no longer active;
- the binding no longer matches;
- the layer is not the widget-owned primary Shapes layer.

### Slice 5: Manual Napari QA

Manual matrix:

- create a new annotation layer;
- open an existing annotation layer;
- adopt an active primary Shapes layer;
- start a polygon lasso;
- hold Space and drag the canvas;
- release Space and continue the same lasso;
- finish and save the annotation;
- inspect geometry for unwanted long segments;
- repeat after selecting a non-annotation layer;
- repeat after removing or clearing the annotation layer.

Record whether the Vispy camera pans correctly when `layer.mouse_pan` is toggled
without changing layer mode. The headless simulation proves this does not finish
the lasso, but real canvas panning still needs GUI verification.

## Recommendation

Implement the scoped Space-pan workaround through `_AnnotationLayerEditGuard`
if manual proof-of-concept confirms that toggling `layer.mouse_pan` is enough
for the real napari canvas to pan while the layer remains in lasso mode.
