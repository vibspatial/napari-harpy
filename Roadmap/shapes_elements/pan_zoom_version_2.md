# Pan And Zoom While Drawing Shapes: Version 2

Status: investigation

This note updates `Roadmap/shapes_elements/pan_zoom.md` after a closer look at
the annotation widget, napari's Shapes mode lifecycle, and the installed napari
mouse/key handling. The local environment was rechecked with napari 0.7.1; the
Space/lasso behavior described below still matches the earlier 0.7.0
investigation.

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

Add a lasso-aware Space-pan path to the annotation layer guard.

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

The derived "lasso is suspended" state is:

```python
self._space_pan_key_held or self._space_pan_mouse_gesture_active
```

Here, a Space-pan mouse gesture means the mouse button is down for a pan that
began while the lasso was suspended by Space. It is intentionally separate from
the Space-key state so release order is harmless: the user can release Space
first or release the mouse first, and normal lasso input resumes only after
both states are false. Starting that mouse gesture does not enable pan by
itself; pan is enabled by the Space-key hold.

The keybinding should behave like this:

1. If the guarded layer is actively creating a polygon lasso:
   - mark `_space_pan_key_held = True`;
   - remember the previous `layer.mouse_pan` value;
   - set `layer.mouse_pan = True`;
   - yield until Space release;
   - mark `_space_pan_key_held = False`;
   - if `_space_pan_mouse_gesture_active` is false, restore the previous
     `layer.mouse_pan` and fully resume lasso callbacks;
   - if `_space_pan_mouse_gesture_active` is true, keep lasso callbacks
     suspended and keep pan enabled until that mouse gesture releases.
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

When lasso is suspended and the layer is actively creating a lasso:

- the lasso move callback should no-op, so pointer movement during Space-pan
  does not add lasso vertices;
- the lasso drag callback should no-op on mouse press, so pressing the mouse
  while Space is held does not finish the lasso;
- the lasso drag callback should mark `_space_pan_mouse_gesture_active = True`
  when it suppresses a mouse press during Space-pan;
- the lasso drag callback should clear `_space_pan_mouse_gesture_active` on the
  matching mouse release.

When `_space_pan_mouse_gesture_active` becomes false after mouse release, the
guard should fully resume lasso callbacks only if `_space_pan_key_held` is also
false. If Space is still held, keep lasso suspended.

When lasso is not suspended, both wrappers should delegate to napari's original
callbacks.

If the annotation layer is already in lasso mode when the guard attaches, the
implementation may also need to update `layer.mouse_drag_callbacks` and
`layer.mouse_move_callbacks` so the active callback lists point at the wrappers,
not the original functions.

The same callback-list concern applies whenever wrappers are installed while
the layer is already in `Mode.ADD_POLYGON_LASSO`: changing `_drag_modes` and
`_move_modes` alone does not rewrite callback functions already present in the
active callback lists.

### Behavior Contract

The custom Space-pan path should only run when all of these are true:

- the guarded layer is still attached;
- the layer is a napari `Shapes` layer;
- `layer._mode == Mode.ADD_POLYGON_LASSO`;
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

1. User starts a lasso.
2. User holds Space.
3. User presses the mouse and drags to pan.
4. User releases Space while the mouse is still down.
5. The guard keeps lasso callbacks suspended and keeps pan enabled.
6. User releases the mouse.
7. The guard clears `_space_pan_mouse_gesture_active`, restores the previous
   mouse-pan value, and resumes normal lasso callbacks.

Case B: mouse release happens before Space release.

1. User starts a lasso.
2. User holds Space.
3. User presses the mouse and drags to pan.
4. User releases the mouse while Space is still down.
5. The guard clears `_space_pan_mouse_gesture_active` but keeps lasso callbacks
   suspended because `_space_pan_key_held` is still true.
6. User releases Space.
7. The guard restores the previous mouse-pan value and resumes normal lasso
   callbacks.

In both cases the active lasso must remain unfinished, `layer.mode` must remain
`Mode.ADD_POLYGON_LASSO`, and the next normal mouse move after recovery should
continue the same lasso.

## Expected UX

Target workflow:

1. User selects the annotation Shapes layer.
2. User selects napari's polygon lasso tool.
3. User clicks to start a lasso.
4. User holds Space.
5. User presses the mouse and drags the canvas to pan.
6. User releases Space and mouse in either order.
7. Once both are released, the user continues the same lasso and finishes it
   normally.

Mouse-wheel zoom should continue to be left to napari unless testing shows a
separate issue.

## Known Limitations

This is still a workaround.

Changing the camera while a lasso is active changes the data coordinate under
the cursor. When drawing resumes, the next lasso move may connect the last
pre-pan coordinate to the new post-pan coordinate. That can create a long edge
after large pans. Small pans should be much less risky, but manual QA is needed.

The harder case is tablet-style lasso drawing where the mouse button is already
held down before Space is pressed. A first implementation can focus on the
common mouse-draw workflow: click to start, move without holding the mouse
button, hold Space, press the mouse to pan, release Space and mouse in either
order, then continue drawing. Supporting pause after an already-running lasso
drag generator has started may require additional handling and can remain
outside the first Space-pan behavior slice.

## Suggested Implementation Slices

### Slice 1: Guard Plumbing

Goal: extend `_AnnotationLayerEditGuard` so it can safely own all state that
later Space-pan slices need, without changing lasso behavior yet.

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
- suppress lasso callbacks;
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
- additionally validate that `_drag_modes` exposes `Mode.ADD_POLYGON_LASSO`;
- additionally validate that `_move_modes` exposes `Mode.ADD_POLYGON_LASSO`;
- copy both mappings into instance-local dictionaries;
- keep wrapping only `Mode.DIRECT` and `Mode.VERTEX_REMOVE` in `_drag_modes`;
- leave every `_move_modes` callback unchanged in this slice;
- assign both copied mappings back to the layer instance;
- capture whether each mapping was originally present in `vars(layer)`;
- leave `layer.mouse_drag_callbacks` and `layer.mouse_move_callbacks`
  unchanged.

The copied `_move_modes` mapping is intentionally behavior-neutral in this
slice. Its purpose is to prove that the guard can own and restore the mapping
before later slices install a lasso move wrapper.

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
- Slice 1 does not replace `_move_modes[Mode.ADD_POLYGON_LASSO]` yet.
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
- No lasso behavior changes are introduced in this slice.
- Active mouse callback lists are unchanged in this slice.

### Slice 2: Space-Pan State Machine

Add private state-machine helpers to `_AnnotationLayerEditGuard`, but do not
install the Space keybinding and do not wrap lasso callbacks yet.

Suggested private helpers:

```python
def _lasso_is_suspended(self) -> bool: ...
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
def _lasso_is_suspended(self) -> bool:
    return self._space_pan_key_held or self._space_pan_mouse_gesture_active


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

- `_lasso_is_suspended()` returns true when either `_space_pan_key_held` or
  `_space_pan_mouse_gesture_active` is true;
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
- helper calls do not change `layer.mode`, layer data, selected rows, keymaps,
  or active callback lists.

### Slice 3: Space Keybinding State

Add the guarded layer Space keybinding.

Headless tests should prove:

- active lasso Space press calls the state-machine key-hold begin helper;
- active lasso Space release calls the state-machine key-hold end helper;
- active lasso Space does not change `layer.mode`;
- non-lasso Space still behaves like normal temporary pan-zoom;
- Space keybinding capture/restoration preserves any pre-existing instance
  Space binding.

### Slice 4: Lasso Callback Suppression

Wrap lasso drag and move callbacks.

Headless tests should prove:

- while lasso is suspended, lasso move callback does not add vertices;
- while lasso is suspended, lasso drag callback does not finish the active
  lasso;
- mouse press during Space-pan calls the state-machine mouse-gesture begin
  helper;
- mouse release calls the state-machine mouse-gesture end helper;
- if Space is released first, lasso callbacks resume only after mouse release;
- if mouse is released first, lasso callbacks resume only after Space release;
- layer data, selected rows, and current mode are preserved.

### Slice 5: Widget Gating

If the guard needs widget context, pass a small predicate into the guard, for
example `can_space_pan_lasso()`. It can reuse
`_annotation_layer_binding_matches()` and the viewer active-layer check.

Tests should prove the custom behavior no-ops when:

- the annotation layer is no longer active;
- the binding no longer matches;
- the layer is not the widget-owned primary Shapes layer.

### Slice 6: Manual Napari QA

Manual matrix:

- create a new annotation layer;
- open an existing annotation layer;
- adopt an active primary Shapes layer;
- start a polygon lasso;
- hold Space, press the mouse, and drag the canvas;
- release Space and mouse in both possible orders;
- continue the same lasso after both are released;
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
