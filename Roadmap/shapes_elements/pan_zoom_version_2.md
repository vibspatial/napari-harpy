# Pan And Zoom While Drawing Shapes: Version 2

Status: implemented through Slice 4

This note updates `Roadmap/shapes_elements/pan_zoom.md` after a closer look at
the annotation widget, napari's Shapes mode lifecycle, and the installed napari
mouse/key handling. The local environment was rechecked with napari 0.7.1; the
Space/drawing behavior described below still matches the earlier 0.7.0
investigation.

## Summary

The current drawing interruption is caused by napari's default Space shortcut.
Space temporarily switches the active layer to `pan_zoom`. For a Shapes layer,
changing mode while a shape is being created calls `_finish_drawing()`, so the
in-progress shape is finalized or cancelled before the user can continue.

A true upstream-quality "pause drawing, pan, continue the same shape" workflow
still belongs in napari because it needs cooperation between the canvas, camera,
mouse event stream, and layer creation state. However, napari-harpy can likely
provide a scoped annotation-widget workaround:

- intercept Space only on the widget-owned annotation Shapes layer;
- do not set `layer.mode = Mode.PAN_ZOOM` while a resumable draw mode is active;
- temporarily set `layer.mouse_pan = True`;
- suppress the annotation layer's draw-mode mouse callbacks while Space-pan is
  active;
- restore the previous mouse-pan state on Space release and continue the same
  in-progress shape.

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

That means drawing is not being stopped by the Harpy annotation widget
directly. It is stopped by napari's normal mode-switch behavior.

The lasso itself is handled in
`.venv/lib/python3.13/site-packages/napari/layers/shapes/_shapes_mouse_bindings.py`:

- `Mode.ADD_POLYGON_LASSO` uses `add_path_polygon_lasso`;
- while drawing, mouse moves call `polygon_creating`;
- if `_is_creating` is already true, a lasso mouse press finishes drawing;
- while moving, `polygon_creating` keeps the active vertex under the cursor and
  adds vertices after enough screen-pixel movement.

The same mode-switch risk applies to other Shapes draw modes that keep an
unfinished shape between mouse gestures: `Mode.ADD_PATH`,
`Mode.ADD_POLYGON`, and `Mode.ADD_POLYLINE`.

So preventing the mode switch is necessary, but not sufficient. If Space enables
camera panning and the layer still receives normal draw-mode mouse events, the
shape can continue to draw during the pan or a mouse press can finish it.

The custom Space-pan hook should intentionally support only the draw modes that
can resume after a click/release:

```python
SPACE_PAN_RESUMABLE_DRAW_MODES = {
    Mode.ADD_POLYGON_LASSO,
    Mode.ADD_PATH,
    Mode.ADD_POLYGON,
    Mode.ADD_POLYLINE,
}
```

Do not include `Mode.ADD_LINE`, `Mode.ADD_RECTANGLE`, or `Mode.ADD_ELLIPSE` in
this first design. They are drag-to-create modes whose mouse release normally
finishes the shape, so supporting a Space-pan pause while that same drag is
already active would require a different, riskier generator-interruption
workflow.

## Local Confirmation

A small headless simulation in the repository environment confirmed the key
split:

- setting `layer.mouse_pan = True` does not finish an active draw;
- setting `layer.mode = Mode.PAN_ZOOM` does finish the active draw;
- while `layer.mouse_pan = True`, draw-mode mouse callbacks still add vertices
  unless they are explicitly suppressed.

This makes `mouse_pan` the useful lever. The implementation should avoid mode
changes during an active resumable draw and should gate draw callbacks during the
temporary pan state.

One important constraint: `layer.mouse_pan` only affects
`viewer.camera.mouse_pan` when that layer is the active viewer layer. The custom
Space-pan path must therefore prove the annotation layer is active before it
sets `layer.mouse_pan = True`.

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

The guard itself currently has no viewer or widget context. Later slices should
pass a small predicate from `ShapesAnnotation` into the guard rather than making
the guard discover widget state on its own.

## Recommended Design

Add a resumable-draw-aware Space-pan path to the annotation layer guard.

### Space Keybinding

When the guard attaches, install an instance-level `Space` keybinding on the
annotation layer. Store and restore any previous instance binding so user or
plugin customizations on that layer are not lost.

Napari coerces keymap keys into `KeyBinding` objects, so implementation should
use `layer.bind_key("Space", ...)` or napari's keybinding coercion utilities
when capturing/restoring Space. Direct `layer.keymap["Space"]` lookup will not
find an existing Space binding.

The keybinding should track Space-key state separately from mouse-gesture state.
Do not model the workflow with only one boolean.

Suggested state fields:

```python
self._space_pan_key_held = False
self._space_pan_mouse_gesture_active = False
self._previous_mouse_pan: bool | None = None
```

The derived "drawing is suspended" state is:

```python
self._space_pan_key_held or self._space_pan_mouse_gesture_active
```

Here, a Space-pan mouse gesture means the mouse button is down for a pan that
began while drawing was suspended by Space. It is intentionally separate from
the Space-key state so release order is harmless: the user can release Space
first or release the mouse first, and normal drawing input resumes only after
both states are false. Starting that mouse gesture does not enable pan by
itself; pan is enabled by the Space-key hold.

This routing was checked against the locally installed napari 0.7.1 source.
Setting `layer.mouse_pan = True` on the active layer propagates to
`viewer.camera.mouse_pan`, enabling camera pan without changing
`layer.mode`. Vispy's camera may mark the canvas mouse event as handled, but
Vispy's event emitter keeps calling later callbacks unless the event is
blocked. Napari's canvas mouse dispatcher still calls the active layer's mouse
callbacks after the viewer callbacks. Therefore the guard can detect the
temporary pan mouse gesture from wrapped drag callbacks for
`SPACE_PAN_RESUMABLE_DRAW_MODES` on `mouse_press` while drawing is suspended;
no separate canvas-level event hook is expected for this state.

The keybinding should behave like this:

1. If the guarded layer is actively creating a shape in
   `SPACE_PAN_RESUMABLE_DRAW_MODES`:
   - mark `_space_pan_key_held = True`;
   - remember the previous `layer.mouse_pan` value;
   - set `layer.mouse_pan = True`;
   - yield until Space release;
   - mark `_space_pan_key_held = False`;
   - if `_space_pan_mouse_gesture_active` is false, restore the previous
     `layer.mouse_pan` and fully resume draw callbacks;
   - if `_space_pan_mouse_gesture_active` is true, keep draw callbacks
     suspended and keep pan enabled until that mouse gesture releases.
2. Otherwise, delegate to napari-equivalent Space behavior:
   - store the current mode;
   - switch to `PAN_ZOOM`;
   - yield;
   - restore the previous mode.

The custom branch must not set `layer.mode = Mode.PAN_ZOOM` while
`layer._is_creating` is true in one of `SPACE_PAN_RESUMABLE_DRAW_MODES`.

Napari's Qt viewer puts the active layer before the viewer in the keymap
provider chain, so a layer instance binding should win over the viewer Space
shortcut when the annotation layer is active. User-level keymap overrides still
win, which is acceptable.

### Draw Callback Suppression

The guard should also create an instance-local copy of `_move_modes`, not only
`_drag_modes`.

Wrap:

- `_drag_modes[mode]` for every mode in `SPACE_PAN_RESUMABLE_DRAW_MODES`;
- `_move_modes[mode]` for every mode in `SPACE_PAN_RESUMABLE_DRAW_MODES`.

When drawing is suspended and the layer is actively creating a shape in one of
the supported modes:

- the move callback should no-op, so pointer movement during Space-pan does not
  move the active vertex or add path/lasso vertices;
- the drag callback should no-op on mouse press, so pressing the mouse while
  Space is held does not add a vertex or finish the active shape;
- the drag callback should mark `_space_pan_mouse_gesture_active = True`
  when it suppresses a mouse press during Space-pan;
- the drag callback should clear `_space_pan_mouse_gesture_active` on the
  matching mouse release/end of the wrapper generator.

`Mode.ADD_PATH` and `Mode.ADD_POLYGON_LASSO` use generator drag callbacks.
`Mode.ADD_POLYGON` and `Mode.ADD_POLYLINE` use non-generator press callbacks.
For the non-generator modes, the wrapper may need to return a tiny generator
when drawing is suspended so the guard can observe the matching mouse release
without delegating the suppressed press to napari.

When `_space_pan_mouse_gesture_active` becomes false after mouse release, the
guard should fully resume draw callbacks only if `_space_pan_key_held` is also
false. If Space is still held, keep drawing suspended.

When drawing is not suspended, wrappers should delegate to napari's original
callbacks.

If the annotation layer is already in one of `SPACE_PAN_RESUMABLE_DRAW_MODES`
when the guard attaches, the implementation may also need to update
`layer.mouse_drag_callbacks` and `layer.mouse_move_callbacks` so the active
callback lists point at the wrappers, not the original functions.

The same callback-list concern applies whenever wrappers are installed while
the layer is already in a supported draw mode: changing `_drag_modes` and
`_move_modes` alone does not rewrite callback functions already present in the
active callback lists.

### Behavior Contract

The custom Space-pan path should only run when all of these are true:

- the guarded layer is still attached;
- the layer is a napari `Shapes` layer;
- `layer._mode in SPACE_PAN_RESUMABLE_DRAW_MODES`;
- `layer._is_creating is True`;
- the layer is active in the viewer;
- the layer still matches the widget-owned annotation binding.

It should not:

- call `layer._finish_drawing()`;
- change `layer.mode`;
- add or remove shapes;
- save shapes;
- affect styled Shapes layers or non-annotation Shapes layers;
- change behavior for other layers in the viewer.

### Release-Order Contract

The implementation must recover cleanly from both possible release orders after
the user starts a mouse pan while holding Space.

Case A: Space release happens before mouse release.

1. User starts a shape in a supported resumable draw mode.
2. User holds Space.
3. User presses the mouse and drags to pan.
4. User releases Space while the mouse is still down.
5. The guard keeps draw callbacks suspended and keeps pan enabled.
6. User releases the mouse.
7. The guard clears `_space_pan_mouse_gesture_active`, restores the previous
   mouse-pan value, and resumes normal draw callbacks.

Case B: mouse release happens before Space release.

1. User starts a shape in a supported resumable draw mode.
2. User holds Space.
3. User presses the mouse and drags to pan.
4. User releases the mouse while Space is still down.
5. The guard clears `_space_pan_mouse_gesture_active` but keeps draw callbacks
   suspended because `_space_pan_key_held` is still true.
6. User releases Space.
7. The guard restores the previous mouse-pan value and resumes normal draw
   callbacks.

In both cases the active shape must remain unfinished, `layer.mode` must remain
the original supported draw mode, and the next normal mouse move after recovery
should continue the same shape.

## Expected UX

Target workflow:

1. User selects the annotation Shapes layer.
2. User selects napari's polygon lasso, path, polygon, or polyline tool.
3. User clicks to start drawing.
4. User holds Space.
5. User presses the mouse and drags the canvas to pan.
6. User releases Space and mouse in either order.
7. Once both are released, the user continues the same shape and finishes it
   normally.

Mouse-wheel zoom should continue to be left to napari unless testing shows a
separate issue.

## Known Limitations

This is still a workaround.

Changing the camera while drawing is active changes the data coordinate under
the cursor. When drawing resumes, the next move or click may connect the last
pre-pan coordinate to the new post-pan coordinate. That can create a long edge
after large pans, especially for lasso/path drawing. Small pans should be much
less risky, but manual QA is needed.

The harder case is tablet-style lasso drawing where the mouse button is already
held down before Space is pressed. A first implementation can focus on the
common mouse-draw workflow: click to start a resumable draw mode, move without
holding the mouse button, hold Space, press the mouse to pan, release Space and
mouse in either order, then continue drawing. Supporting pause after an
already-running drag generator has started may require additional handling and
can remain
outside the first Space-pan behavior slice.

Line, rectangle, and ellipse drawing remain outside this design. They finish on
mouse release and do not have the same stable "unfinished but no mouse button
held" state as lasso, path, polygon, and polyline.

## Suggested Implementation Slices

### Slice 1: Guard Plumbing

Status: implemented.

Goal: extend `_AnnotationLayerEditGuard` so it can safely own all state that
later Space-pan slices need, without changing draw behavior yet.

This slice should keep the current direct-edit and vertex-remove behavior
unchanged. It is plumbing only.

#### Scope

The guard should continue to be the only object that mutates annotation-layer
instance interaction state. It should own and restore:

- `_drag_modes`, as it already does today;
- `_move_modes`, using the same instance-local override pattern as
  `_drag_modes`;
- reserved Space-pan state fields;
- reserved previous Space keybinding state.

This slice must not:

- bind or override Space yet;
- change `layer.mode`;
- set `layer.mouse_pan`;
- suppress draw callbacks;
- mutate `layer.mouse_drag_callbacks` or `layer.mouse_move_callbacks`;
- patch napari class-level mappings;
- alter behavior for non-annotation Shapes layers.

#### State To Add

Add fields to `_AnnotationLayerEditGuard` for move-mode ownership:

```python
self._original_move_modes: dict[object, Callable[..., Any]] | None = None
self._had_instance_move_modes = False
```

Add fields reserved for the Space-pan lifecycle:

```python
self._space_pan_key_held = False
self._space_pan_mouse_gesture_active = False
self._previous_mouse_pan: bool | None = None
self._previous_space_keybinding: Callable[..., Any] | object | None = None
self._had_instance_space_keybinding = False
```

The exact type of `_previous_space_keybinding` can follow napari's keymap value
types. The important contract is that the guard can distinguish "there was no
instance Space binding" from "there was an instance Space binding whose value
was `None` or another sentinel-like value".

Using a private sentinel object is acceptable if it makes that distinction
clearer than relying on `None`.

#### Attach Contract

`attach(layer)` should still be idempotent when called with the current layer.

When attaching a new layer:

- call `disconnect()` first, preserving the existing replacement behavior;
- read `layer._drag_modes` and `layer._move_modes`;
- validate both are dict-like mappings;
- keep the existing validation that `_drag_modes` exposes `Mode.DIRECT` and
  `Mode.VERTEX_REMOVE`;
- additionally validate that both mappings expose every mode in
  `SPACE_PAN_RESUMABLE_DRAW_MODES`;
- copy both mappings into instance-local dictionaries;
- keep wrapping only `Mode.DIRECT` and `Mode.VERTEX_REMOVE` in `_drag_modes`;
- leave every `_move_modes` callback unchanged in this slice;
- assign both copied mappings back to the layer instance;
- capture whether each mapping was originally present in `vars(layer)`;
- leave `layer.mouse_drag_callbacks` and `layer.mouse_move_callbacks`
  unchanged.

The copied `_move_modes` mapping is intentionally behavior-neutral in this
slice. Its purpose is to prove that the guard can own and restore the mapping
before later slices install draw-mode wrappers.

#### Disconnect Contract

`disconnect()` should restore everything it owns and should remain safe to call
multiple times.

On disconnect:

- clear the tracked layer reference;
- clear original mapping references;
- clear wrapped direct and vertex-remove callback references;
- reset `_space_pan_key_held` to `False`;
- reset `_space_pan_mouse_gesture_active` to `False`;
- reset `_previous_mouse_pan` to `None`;
- reset previous Space keybinding bookkeeping;
- restore `_drag_modes` exactly as today;
- restore `_move_modes` with the same semantics:
  - if `_move_modes` was originally instance-local, restore the original object;
  - if `_move_modes` was inherited/class-level, delete the guard-created
    instance override;
- do not change `layer.mouse_pan` in this slice, because Slice 1 never changes
  it.

If restoring one mapping fails because the layer has already been partially
destroyed, prefer the same defensive style as the current guard: avoid raising
from expected cleanup paths where possible, but do not silently hide programming
errors during normal attach validation.

#### Space Keybinding Reservation

Slice 1 should reserve fields for Space keybinding restoration but should not
install the Space keybinding yet. Actual binding belongs to Slice 2.

If helper methods are added in Slice 1, they should be private and inert unless
explicitly called by future slices. For example:

```python
def _capture_space_keybinding(self, layer: Shapes) -> None: ...
def _restore_space_keybinding(self, layer: Shapes) -> None: ...
```

These helpers may be implemented now if they simplify tests, but `attach(...)`
should not change the layer's keymap yet.

#### Tests

Add or update tests in `tests/test_shapes_annotation_widget.py`.

Required coverage:

- attaching to a layer that originally inherited `_move_modes` creates an
  instance `_move_modes` override;
- disconnecting that layer deletes the guard-created `_move_modes` override and
  falls back to napari's inherited mapping;
- attaching to a layer with an existing instance `_move_modes` restores the same
  original mapping object on disconnect;
- existing `_drag_modes` tests still prove direct and vertex-remove wrapping and
  restoration;
- `attach(layer)` remains idempotent and does not create new wrapper objects on
  the second attach;
- attaching to a second layer restores `_drag_modes` and `_move_modes` on the
  first layer before patching the second layer;
- Slice 1 does not bind Space: a layer's instance keymap is unchanged after
  attach and disconnect;
- Slice 1 does not change `layer.mouse_pan`;
- Slice 1 does not change `layer.mode`;
- Slice 1 does not replace move callbacks for
  `SPACE_PAN_RESUMABLE_DRAW_MODES` yet.
- Slice 1 does not mutate `layer.mouse_drag_callbacks` or
  `layer.mouse_move_callbacks`.

Suggested focused test names:

```text
test_annotation_layer_edit_guard_restores_instance_move_modes
test_annotation_layer_edit_guard_attach_is_idempotent_for_move_modes
test_annotation_layer_edit_guard_replacing_layer_disconnects_previous_move_modes
test_annotation_layer_edit_guard_slice_one_does_not_bind_space_or_change_mouse_pan
```

#### Definition Of Done

- `_AnnotationLayerEditGuard` owns `_drag_modes` and `_move_modes` with matching
  attach/disconnect semantics.
- Existing direct-edit and vertex-remove tests still pass.
- New tests prove move-mode restoration and behavior neutrality.
- No Space keybinding behavior is active yet.
- No draw behavior changes are introduced in this slice.
- Active mouse callback lists are unchanged in this slice.

### Slice 2: Space-Pan State Machine

Status: implemented.

Add private state-machine helpers to `_AnnotationLayerEditGuard`, but do not
install the Space keybinding and do not wrap draw callbacks yet.

Implemented in `src/napari_harpy/widgets/shapes_annotation/widget.py` as
internal guard helpers and covered by direct headless tests in
`tests/test_shapes_annotation_widget.py`. The implementation includes the
resumable draw-mode set, two release-order flags, key-hold and mouse-gesture
begin/end helpers, restore-on-both-released logic, and disconnect cleanup.
No user-facing Space keybinding or draw callback suppression is active yet.

Suggested private helpers:

```python
def _drawing_is_suspended(self) -> bool: ...
def _can_space_pan_draw_mode(self, layer: Shapes) -> bool: ...
def _begin_space_pan_key_hold(self, layer: Shapes) -> None: ...
def _end_space_pan_key_hold(self, layer: Shapes) -> None: ...
def _begin_space_pan_mouse_gesture(self) -> None: ...
def _end_space_pan_mouse_gesture(self, layer: Shapes) -> None: ...
def _restore_space_pan_if_complete(self, layer: Shapes) -> None: ...
```

#### Implementation Skeleton

Slice 2 should replace any single `_space_pan_active` boolean with the two
explicit release-order flags reserved in Slice 1:

```python
self._space_pan_key_held = False
self._space_pan_mouse_gesture_active = False
self._previous_mouse_pan: bool | None = None
```

Do not keep `_space_pan_active` as the source of truth. A single boolean cannot
represent the two half-complete recovery states:

- Space was released, but the mouse button is still down for the pan;
- the mouse button was released, but Space is still held.

The state machine should be intentionally small and live on
`_AnnotationLayerEditGuard`:

```python
def _drawing_is_suspended(self) -> bool:
    return self._space_pan_key_held or self._space_pan_mouse_gesture_active


def _can_space_pan_draw_mode(self, layer: Shapes) -> bool:
    return (
        layer._is_creating
        and layer._mode in SPACE_PAN_RESUMABLE_DRAW_MODES
    )


def _begin_space_pan_key_hold(self, layer: Shapes) -> None:
    self._space_pan_key_held = True
    if self._previous_mouse_pan is None:
        self._previous_mouse_pan = layer.mouse_pan
    layer.mouse_pan = True


def _end_space_pan_key_hold(self, layer: Shapes) -> None:
    self._space_pan_key_held = False
    self._restore_space_pan_if_complete(layer)


def _begin_space_pan_mouse_gesture(self) -> None:
    self._space_pan_mouse_gesture_active = True


def _end_space_pan_mouse_gesture(self, layer: Shapes) -> None:
    self._space_pan_mouse_gesture_active = False
    self._restore_space_pan_if_complete(layer)


def _restore_space_pan_if_complete(self, layer: Shapes) -> None:
    if self._space_pan_key_held or self._space_pan_mouse_gesture_active:
        return
    if self._previous_mouse_pan is None:
        return
    layer.mouse_pan = self._previous_mouse_pan
    self._previous_mouse_pan = None
```

The key invariant is that restoration happens only when both state flags are
false. This is what makes the user release order irrelevant.

Expected helper behavior:

- `_drawing_is_suspended()` returns true when either `_space_pan_key_held` or
  `_space_pan_mouse_gesture_active` is true;
- `_can_space_pan_draw_mode(...)` returns true only when the layer is creating
  a shape in `SPACE_PAN_RESUMABLE_DRAW_MODES`;
- `_begin_space_pan_key_hold(...)` sets `_space_pan_key_held`, captures the
  previous `layer.mouse_pan` only once per Space-pan session, and sets
  `layer.mouse_pan = True`;
- `_end_space_pan_key_hold(...)` clears `_space_pan_key_held` and restores the
  previous mouse-pan value only when no Space-pan mouse gesture is active;
- `_begin_space_pan_mouse_gesture()` sets
  `_space_pan_mouse_gesture_active` to record that the mouse button is now down
  for the temporary Space-pan; it does not enable pan by itself;
- `_end_space_pan_mouse_gesture(...)` clears `_space_pan_mouse_gesture_active`
  and restores the previous mouse-pan value only when Space is no longer held;
- `_restore_space_pan_if_complete(...)` is the only helper that restores
  `layer.mouse_pan` and clears `_previous_mouse_pan`.

This slice may change `layer.mouse_pan` only when helper methods are called
directly by tests. It must not connect those helpers to real key or mouse
events yet.

Headless tests should prove:

- beginning a Space-pan key hold sets `_space_pan_key_held`;
- beginning a Space-pan key hold sets `layer.mouse_pan = True`;
- ending a Space-pan key hold restores the previous mouse-pan value when no
  Space-pan mouse gesture is active;
- ending a Space-pan key hold does not restore the previous mouse-pan value
  while a Space-pan mouse gesture is still active;
- beginning and ending a Space-pan mouse gesture toggles
  `_space_pan_mouse_gesture_active`;
- if Space ends first, restoration happens only after the mouse gesture ends;
- if the mouse gesture ends first, restoration happens only after Space ends;
- repeated begin calls do not overwrite the captured previous mouse-pan value;
- `SPACE_PAN_RESUMABLE_DRAW_MODES` includes only
  `Mode.ADD_POLYGON_LASSO`, `Mode.ADD_PATH`, `Mode.ADD_POLYGON`, and
  `Mode.ADD_POLYLINE`;
- `_can_space_pan_draw_mode(...)` excludes `Mode.ADD_LINE`,
  `Mode.ADD_RECTANGLE`, and `Mode.ADD_ELLIPSE`;
- helper calls do not change `layer.mode`, layer data, selected rows, keymaps,
  or active callback lists.

### Slice 3: Space Keybinding State

Status: implemented.

Add the guarded layer Space keybinding, but keep the custom Space-pan drawing
branch disabled until Slice 4 installs draw-callback suppression.

Implemented in `src/napari_harpy/widgets/shapes_annotation/widget.py` as
Space keymap capture/install/restore logic on `_AnnotationLayerEditGuard`.
The implementation uses napari's coerced Space key for direct keymap
inspection, installs the binding through `layer.bind_key(...)`, restores or
removes the instance binding on disconnect, and kept the custom Space-pan draw
branch disabled until Slice 4 added callback suppression.

The goal of this slice is to prove keybinding ownership, capture/restoration,
and fallback behavior without exposing half-working drawing behavior. In this
slice, pressing Space while actively drawing a supported mode must still follow
napari-equivalent temporary pan-zoom behavior, because normal draw callbacks
are not suppressed yet.

Implementation contract:

- use napari's keybinding coercion when inspecting the keymap:
  `coerce_keybinding("Space")` is the key stored in `layer.keymap`, not the
  raw string `"Space"`;
- install the custom binding with `layer.bind_key("Space", callback,
  overwrite=True)` rather than manually inserting a raw string key;
- keep real key events on the fallback path until Slice 4 wraps draw callbacks;
  this may be done with a temporary private feature flag or predicate in
  Slice 3;
- install an instance-level Space keybinding on the guarded layer during
  `attach(...)`;
- capture whether the guarded layer had an instance Space binding before
  attach with `space_key = coerce_keybinding("Space")`;
- store the previous value from `layer.keymap[space_key]` only if that coerced
  key existed in the instance keymap;
- on `disconnect()`, restore the previous value if an instance binding existed
  before attach;
- on `disconnect()`, remove the guard-created binding with
  `layer.keymap.pop(space_key, None)` if no instance binding existed before
  attach;
- when Space is pressed before draw suppression is installed, delegate to
  napari-equivalent temporary pan-zoom, even if
  `_can_space_pan_draw_mode(layer)` is true;
- do not call `_begin_space_pan_key_hold(...)` or
  `_end_space_pan_key_hold(...)` from real key events until the draw callbacks
  are wrapped in Slice 4;
- keep the direct helper tests from Slice 2 as the only tests that manually
  call the state-machine helpers;
- preserve `layer.mode`, layer data, selected rows, active callback lists, and
  keymap restoration after disconnect.

Slice 4 can then replace the temporary not-ready gate with the final guarded
layer check after the draw callback wrappers are installed.

Headless tests should prove:

- attaching installs an instance Space binding on the guarded layer;
- raw `"Space"` lookup is not used for capture/restoration; tests should use
  napari's coerced `KeyBinding` or `layer.bind_key(...)`;
- disconnect restores a pre-existing instance Space binding;
- disconnect removes the guard-created Space binding when no instance binding
  existed before attach;
- active supported-draw Space does not call the state-machine key-hold helpers
  before draw suppression is installed;
- active supported-draw Space still delegates to napari-equivalent temporary
  pan-zoom before draw suppression is installed;
- unsupported modes still behave like normal temporary pan-zoom;
- Space keybinding capture/restoration preserves any pre-existing instance
  Space binding whose value is `None` or sentinel-like;
- helper calls and real Space key events do not mutate layer data, selected
  rows, or active callback lists.

### Slice 4: Draw Callback Suppression

Status: implemented.

Wrap drag and move callbacks for every mode in `SPACE_PAN_RESUMABLE_DRAW_MODES`.

Implemented in `src/napari_harpy/widgets/shapes_annotation/widget.py` as
supported draw/move callback wrappers on `_AnnotationLayerEditGuard`. The
wrappers delegate normally when drawing is not suspended, suppress supported
draw behavior during Space-pan, track the Space-started mouse gesture through a
small drag generator, and make active supported drawing use the custom
Space-pan keybinding branch.

This slice is where the Space keybinding switches active supported drawing away
from the Slice 3 fallback. After the supported draw callbacks are wrapped,
`_handle_space_keybinding(...)` should require the key event layer to still be
the guarded layer, then use the custom state-machine branch for active supported
drawing:

```python
if self._layer is layer and self._can_space_pan_draw_mode(layer):
    self._begin_space_pan_key_hold(layer)
    try:
        yield
    finally:
        self._end_space_pan_key_hold(layer)
    return

yield from self._fallback_pan_zoom_key_hold(layer)
```

`_fallback_pan_zoom_key_hold(...)` remains the fallback path for unsupported
modes, inactive drawing modes, and any future not-ready guard condition. It
should no longer be used for active drawing in `Mode.ADD_POLYGON_LASSO`,
`Mode.ADD_PATH`, `Mode.ADD_POLYGON`, or `Mode.ADD_POLYLINE` once this slice is
complete.

Implementation contract:

- in `attach(...)`, capture the original supported-mode callbacks before
  replacing them;
- install wrapped callbacks in the instance `_drag_modes` and `_move_modes`
  mappings for every mode in `SPACE_PAN_RESUMABLE_DRAW_MODES`;
- keep existing `Mode.DIRECT` and `Mode.VERTEX_REMOVE` wrappers unchanged;
- when `_drawing_is_suspended()` is false, each supported-mode wrapper must
  delegate to napari's original callback exactly as before;
- when `_drawing_is_suspended()` is true, each supported move callback must
  suppress napari's original move callback and return without adding vertices,
  moving the active vertex, or changing layer data;
- when `_drawing_is_suspended()` is true, each supported drag callback must
  suppress napari's original drag callback for the Space-pan mouse gesture;
- after these wrappers are installed, active supported drawing should take the
  custom Space-pan branch only when the key event belongs to the guarded layer.

The drag wrapper is the expected place to track the temporary pan mouse gesture.
When `_drawing_is_suspended()` is true and a wrapped supported-mode drag callback
receives a mouse press, call `_begin_space_pan_mouse_gesture()` and suppress
napari's original drag callback for that gesture. On the matching mouse
release/end of the wrapper generator, call `_end_space_pan_mouse_gesture(layer)`.
This should still work while `layer.mouse_pan = True`, because napari dispatches
canvas mouse events to the active layer callbacks even when camera pan is
enabled.

The gesture lifetime must be independent of release order:

- if Space is released first, `_space_pan_key_held` becomes `False` but
  `_space_pan_mouse_gesture_active` keeps drawing suspended until the mouse
  gesture ends;
- if the mouse is released first, `_space_pan_mouse_gesture_active` becomes
  `False` but `_space_pan_key_held` keeps drawing suspended until Space is
  released;
- only after both flags are false should `layer.mouse_pan` restore to its
  previous value and drawing callbacks delegate normally again.

Implementation note: the wrapper should be robust to napari callbacks that
return either a generator or a plain value. When delegating, preserve the
original callback's return shape. When suppressing during Space-pan, do not run
the original callback body.

Headless tests should prove:

- when drawing is not suspended, supported drag and move wrappers delegate to
  napari's original callbacks;
- while drawing is suspended, supported move callbacks do not add vertices or
  move the active vertex;
- while drawing is suspended, supported drag callbacks do not add vertices or
  finish the active shape;
- mouse press during Space-pan calls the state-machine mouse-gesture begin
  helper;
- mouse release calls the state-machine mouse-gesture end helper;
- draw-callback suppression never calls the original supported-mode callback
  while `_drawing_is_suspended()` is true;
- active supported-draw Space calls the state-machine key-hold begin/end
  helpers instead of delegating to `_fallback_pan_zoom_key_hold(...)`;
- `_fallback_pan_zoom_key_hold(...)` remains the behavior for unsupported modes
  and non-active drawing;
- if Space is released first, draw callbacks resume only after mouse release;
- if mouse is released first, draw callbacks resume only after Space release;
- unsupported draw modes (`ADD_LINE`, `ADD_RECTANGLE`, `ADD_ELLIPSE`) remain
  delegated to napari behavior;
- layer data, selected rows, and current mode are preserved.

### Slice 5: Active Callback List Refresh

Status: implemented.

Slice 4 wraps `_drag_modes` and `_move_modes`. That covers the normal
annotation-widget lifecycle, where the guard is attached before the user enters
a supported draw mode. However, napari also keeps active callback lists for the
current mode:

```python
layer.mouse_drag_callbacks
layer.mouse_move_callbacks
```

Those lists contain callback objects copied from the current mode mappings. If
the guard attaches while the layer is already in
`SPACE_PAN_RESUMABLE_DRAW_MODES`, replacing `_drag_modes[mode]` and
`_move_modes[mode]` may not update the already-active callback lists. The layer
could then keep calling napari's original draw callbacks until the mode changes
or napari otherwise rebuilds those lists.

Do not solve this by toggling `layer.mode` away and back. Mode switching while
`layer._is_creating` is true is the root cause of the original lasso
interruption.

Implementation direction:

- when `attach(...)` installs supported-mode wrappers, check the current
  `layer._mode`;
- if the current mode is in `SPACE_PAN_RESUMABLE_DRAW_MODES`, replace only the
  matching original callbacks in `layer.mouse_drag_callbacks` and
  `layer.mouse_move_callbacks`;
- do not clear or rebuild the whole callback lists, because other napari/plugin
  callbacks may be present;
- on `disconnect(...)`, reverse that replacement if the active callback lists
  still contain the guard wrappers;
- keep this replacement best-effort and callback-object-specific: if the
  expected original callback is not present, leave the list unchanged;
- preserve the existing `_drag_modes` and `_move_modes` restoration semantics.

Suggested helper shape:

```python
def _replace_callback(
    callbacks: list[Callable[..., Any]],
    *,
    old_callback: Callable[..., Any],
    new_callback: Callable[..., Any],
) -> None:
    ...
```

Headless tests should prove:

- attaching while `layer.mode` is already `Mode.ADD_POLYGON_LASSO` replaces the
  active drag and move callbacks with the guard wrappers;
- disconnect restores the original active callbacks for that mode;
- attaching in a non-resumable mode does not mutate active callback lists;
- callback-list replacement preserves unrelated callbacks in the same list;
- no test or helper changes `layer.mode` to force callback-list refresh.

### Slice 6: Widget Gating

If the guard needs widget context, pass a small predicate into the guard, for
example `can_space_pan_draw()`. It can reuse
`_annotation_layer_binding_matches()` and the viewer active-layer check.

Tests should prove the custom behavior no-ops when:

- the annotation layer is no longer active;
- the binding no longer matches;
- the layer is not the widget-owned primary Shapes layer.

### Slice 7: Manual Napari QA

Manual matrix:

- create a new annotation layer;
- open an existing annotation layer;
- adopt an active primary Shapes layer;
- start a polygon lasso, path, polygon, and polyline;
- hold Space, press the mouse, and drag the canvas;
- release Space and mouse in both possible orders;
- continue the same shape after both are released;
- finish and save the annotation;
- inspect geometry for unwanted long segments;
- verify line, rectangle, and ellipse still use napari's normal behavior;
- repeat after selecting a non-annotation layer;
- repeat after removing or clearing the annotation layer.

Record whether the Vispy camera pans correctly when `layer.mouse_pan` is toggled
without changing layer mode. The headless simulation proves this does not finish
the active drawing state, but real canvas panning still needs GUI verification.

## Recommendation

Implement the scoped Space-pan workaround through `_AnnotationLayerEditGuard`
if manual proof-of-concept confirms that toggling `layer.mouse_pan` is enough
for the real napari canvas to pan while the layer remains in the original
supported draw mode.
