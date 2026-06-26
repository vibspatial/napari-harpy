# Listen To Active Layer Change

Status: specification.

Goal: make annotation-layer edit behavior consistent when a user selects a
compatible Harpy primary shapes layer directly in the napari layer list, not
only through the Shapes Annotation widget.

## Observed Behavior

The hole-aware anchor/separator editing fixes work when the shapes element is
selected in both places:

- the Shapes Annotation widget has opened/adopted the layer as its active
  annotation session
- the same layer is active/selected in the napari UI

If the layer is selected only in the napari UI, but the Shapes Annotation widget
has not opened/adopted it as the active annotation layer, the behavior falls
back to napari's native Shapes editing:

- dragging anchor/separator copies does not use Harpy's live synchronization
- deleting vertices does not use Harpy's hole-aware deletion helper
- row-shortening deletion does not use Harpy's cache-rebuild workaround

This reproduces the behavior from before the hole-editing fixes because those
fixes are installed only through the widget-owned edit guard.

## Current Code Path

The relevant guard is `_AnnotationLayerEditGuard`.

It is attached only when `ShapesAnnotation` owns an active annotation session:

- create-new annotation layer:
  `_open_create_new_annotation_layer(...)`
- edit-existing annotation layer:
  `_open_existing_annotation_layer(...)`
- native/imported layer adoption:
  `_adopt_native_shapes_layer(...)`

In each of these paths the widget sets:

```python
self._annotation_layer = layer
self._annotation_edit_guard.attach(layer)
```

The guard is disconnected in `_clear_annotation_state(...)`:

```python
self._annotation_edit_guard.disconnect()
self._annotation_session = None
self._annotation_layer = None
```

Therefore, a Harpy-managed shapes layer that is merely active in the napari UI
does not automatically receive the annotation edit guard. It keeps napari's
original `Mode.DIRECT` and `Mode.VERTEX_REMOVE` callbacks.

## Event Source

The implementation should listen to napari's real active-layer selection event:

```python
viewer.layers.selection.events.active
```

In the current napari version this event exists and emits the active layer as
`event.value`. The handler may read it defensively with
`getattr(event, "value", None)`, treating a missing value as no active layer.

Do not rely on `ViewerAdapter.active_layer_changed` as the primary event
source. That signal is emitted by `ViewerAdapter.activate_layer(...)`, so it
covers Harpy-driven activation but not necessarily a user clicking a layer in
napari's layer list. Do not connect to it for this feature; the widget should
listen only to napari's active-layer selection event.

## Important Design Boundary

We should not globally patch every napari `Shapes` layer. The guard changes
interactive edit behavior and, for deletion, depends on Harpy's annotation
semantics. It should remain scoped to layers that the annotation workflow can
save safely.

The missing behavior is not "always guard all shapes layers"; it is:

> when the user selects a compatible Harpy primary shapes layer in napari, the
> annotation widget should safely adopt/open that layer as the active annotation
> session, then attach the guard.

Unbound native Shapes layers are still handled by the existing inserted-layer
adoption flow. Active-layer changes should not newly adopt arbitrary unbound
native layers.

## Reuse Existing Session Setup

Do not create a second edit-session setup path.

When a compatible active layer is selected, update the widget target to the
binding's shapes element and reuse `_open_existing_annotation_layer(...)`.

That existing path already handles:

- validating the source GeoDataFrame
- validating the compatible primary shapes binding
- rejecting points-rendered shapes, skipped rows, and non-one-row source
  mappings
- creating `_ShapesAnnotationSession`
- attaching `_AnnotationLayerEditGuard`
- capturing the clean snapshot
- refreshing save/create state

## Implementation Slices

### Slice 1A - Active Layer Event Listener

Status: implemented.

Add a small listener for napari active-layer changes. This slice is plumbing
only: it should observe active-layer changes and route them to a placeholder
hook, but it should not inspect bindings, open/adopt layers, attach the edit
guard, show status messages, or change dirty-session behavior yet.

Implementation notes:

- Connect to `viewer.layers.selection.events.active` when available.
- Keep the event connection logic in a small private helper so `__init__` does
  not grow more event-connection boilerplate.
- The shared handler should resolve the active layer from the payload shape it
  receives:
  - for napari selection events, read `event.value`, using
    `getattr(event, "value", None)` if we want the handler to ignore malformed
    events gracefully
- Ignore `None`.
- Ignore if the active layer is already `self._annotation_layer`.
- Add a private reentrancy guard. Later slices may adopt/open layers, and those
  paths call `viewer_adapter.activate_layer(...)`, which can itself update
  active-layer state.
- Route valid active-layer changes to a placeholder private method, for example
  `_maybe_adopt_active_shapes_layer(layer)`. In this slice that method should be
  a no-op; Slice 1B and Slice 1C will add candidate/adoption behavior.
- Disconnecting is not currently needed because the widget has the same
  lifetime as the viewer-side dock widget, matching the existing layer
  inserted/removed event connections.

Tests:

- Active-layer selection events are connected when the viewer exposes
  `layers.selection.events.active`.
- Emitting/changing the active layer calls the placeholder adoption hook with
  the resolved active layer.
- If the active layer is already `self._annotation_layer`, the placeholder hook
  is not called.
- If the reentrancy guard is active, the placeholder hook is not called.

### Slice 1B - Compatible Primary Shapes Candidate

Status: implemented.

Add a small private helper that decides whether an active layer can be adopted
as an edit-existing annotation session.

Candidate requirements:

- active layer is a `napari.layers.Shapes`
- active layer has a `ShapesLayerBinding`
- binding belongs to the current `SpatialData` object
- `binding.element_type == "shapes"`
- `binding.shapes_role == "primary"`
- `binding.shapes_rendering_mode == "shapes"`
- `binding.style_spec is None`
- binding coordinate system matches the widget/app active coordinate system
- bound shapes name is currently eligible for that coordinate system

Do not perform full source-GeoDataFrame or row-mapping validation in this helper.
Those checks should remain in `_open_existing_annotation_layer(...)` and
`_validate_opened_existing_shapes_layer(...)`.

Tests:

- compatible primary shapes layer returns a candidate with shapes name and
  coordinate system
- unbound native layer is rejected
- non-shapes layer is rejected
- styled shapes layer is rejected
- points-rendered shapes layer is rejected
- different coordinate system is rejected
- different SpatialData object is rejected

### Slice 1C - Adopt Clean Active Layer

Status: implemented.

When a compatible active layer is selected and adoption is safe, open it as the
active edit-existing annotation session.

Behavior:

- If there is no annotation session, adopt/open the active layer.
- If a clean annotation session exists for a different layer, close the clean
  session and adopt/open the active layer.
- Update the Shapes combo to
  `_ShapesAnnotationTarget.edit_existing(binding.element_name)`.
- Reuse `_open_existing_annotation_layer(...)`.
- After adoption, the annotation layer, session, edit guard, status card, Save
  button, and Create holes button should reflect the active layer.

Important nuance:

- `_open_existing_annotation_layer(...)` may call
  `viewer_adapter.ensure_shapes_loaded(...)`, but because the selected active
  layer already has the matching binding, the adapter should return that layer
  with `created=False`.

Tests:

- selecting a compatible loaded primary shapes layer in napari opens/adopts it
  in the Shapes Annotation widget
- `_AnnotationLayerEditGuard` is attached to the active layer
- widget combo selection matches the active layer's bound shapes element
- Save/Create holes buttons are enabled
- selecting a compatible layer while a clean different annotation session is
  active closes the old clean session and adopts the selected layer

### Slice 1D - Dirty Session Protection

Status: implemented.

Do not silently switch annotation sessions when the current session has unsaved
changes.

Behavior:

- If a dirty annotation session exists for a different layer, show the existing
  discard confirmation dialog.
- If the user confirms, discard the dirty session and adopt/open the active
  compatible layer.
- If the user cancels, keep the current annotation session.
- After cancel, re-activate the current annotation layer so the napari active
  layer and annotation widget state are brought back into sync.
- Show the existing annotation status for the kept session; do not change the
  save target.

Tests:

- dirty current session plus active compatible layer asks for discard
  confirmation
- cancel keeps the old annotation session and edit guard
- cancel reselects/reactivates the old annotation layer
- confirm discards the dirty session and adopts the selected compatible layer

### Slice 1E - Adopt Viewer-Loaded Active Primary Shapes After Binding Registration

Status: implemented.

Handle the case where a primary shapes layer is loaded through the Viewer
widget while the Shapes Annotation widget is open but still on `Create shapes`.

Observed behavior:

- The Viewer widget loads the primary shapes layer through
  `viewer_adapter.ensure_shapes_loaded(...)`.
- `ensure_shapes_loaded(...)` adds the napari layer to the viewer before it
  registers the Harpy `ShapesLayerBinding`.
- Napari makes newly inserted layers active immediately.
- The Shapes Annotation active-layer listener can therefore see the layer while
  it is still unbound and reject it in `_active_primary_shapes_candidate(...)`.
- Harpy registers the binding afterwards and emits
  `primary_shapes_layer_registered`.
- `_on_primary_shapes_layer_registered(...)` currently only opens the layer if
  the Annotation widget is already on the matching edit-existing target. If the
  widget is still on `Create shapes`, it returns.
- The Viewer widget later calls `viewer_adapter.activate_layer(result.layer)`,
  but real napari may not emit another active-layer event because the newly
  inserted layer is already active.

Why this ordering happens:

- Harpy must add the built layer to napari before it can treat it as a loaded
  viewer layer. In the current adapter flow, `ensure_shapes_loaded(...)` calls
  `_add_layer_to_viewer(...)` first and only then calls
  `register_shapes_layer(...)`.
- `_add_layer_to_viewer(...)` uses napari's normal `viewer.add_layer(layer)`
  API. In napari, `add_layer(...)` appends the layer to `viewer.layers`.
- `viewer.layers` is a selectable layer list. Its insert path has
  `_activate_on_insert = True`, so inserting/appending a layer sets
  `viewer.layers.selection.active` to the new layer.
- Setting `selection.active` emits napari's active-layer event immediately,
  before control returns to Harpy's adapter and before
  `register_shapes_layer(...)` has installed the `ShapesLayerBinding`.
- Therefore this is not a napari rendering/editing bug. It is a natural
  ordering boundary between napari's layer-list selection behavior and Harpy's
  post-insertion binding metadata.

The first missed-adoption timeline is:

```text
Harpy ensure_shapes_loaded
  build layer
  add layer to napari
    napari inserts layer
    napari makes inserted layer active
    napari emits active-layer event
      annotation widget sees active layer
      Harpy binding does not exist yet
      candidate is rejected
  return from add_layer
  Harpy register_shapes_layer(...)
  Harpy emits primary_shapes_layer_registered
```

The later Viewer-widget activation does not reliably repair this:

```text
ensure_shapes_loaded(...)
  add layer to napari
    napari auto-activates inserted layer
    active-layer event fires before binding exists
  register_shapes_layer(...)
    binding now exists
    primary_shapes_layer_registered fires
return result

Viewer widget calls activate_layer(result.layer)
```

By that final `activate_layer(...)` call, napari may already consider
`result.layer` active. Napari's `Selection.active` setter returns early in that
case:

```python
if value == self._active:
    return
```

So `selection.active = layer` may not emit a second napari active-layer event
after the Harpy binding exists. That is why the registration callback is the
right place to repair the missed adoption: at
`primary_shapes_layer_registered` time the binding exists, and the callback can
check whether the just-registered layer is already napari's active layer.

Implementation nuance:

- In real plugin widgets, napari may expose `viewer.layers.selection.active`
  through a `PublicOnlyProxy`. Harpy's `ShapesLayerBinding` stores the real
  layer object, so the registration repair must unwrap proxy active layers
  before doing the identity comparison. Otherwise the active-layer repair works
  in plain model tests but fails in the live plugin widget context.
- The active-layer repair must not run while the widget is already discarding
  or closing its own annotation session. Edit-existing discard removes the dirty
  layer and reloads the saved source layer. That reload can register and
  auto-activate the old layer while `_discard_annotation_layer(...)` is still
  running and the old dirty session has not yet been cleared.

The dirty discard loop to avoid is:

```text
User edits new_shapes_1
User selects new_shapes_2 in the Annotation widget
Annotation asks whether to discard dirty new_shapes_1
User clicks Discard annotations
_discard_annotation_layer(...)
  remove dirty new_shapes_1 layer
  reload saved new_shapes_1 layer
    napari auto-activates reloaded new_shapes_1
    Harpy registers reloaded new_shapes_1
    active-layer repair tries to adopt new_shapes_1
      old dirty session is still present
      Annotation asks whether to discard dirty new_shapes_1 again
```

This is a reentrant widget-owned discard/reload path, not a user-driven active
layer switch. The active-layer handler and registration repair should return
while `self._is_handling_annotation_layer_removal` is true. After discard
finishes, the original target-change flow can continue and open the selected
target, for example `new_shapes_2`, exactly once.

Expected behavior:

- If a compatible primary, unstyled shapes layer is loaded through the Viewer
  widget and is already the active napari layer once its binding is registered,
  the Shapes Annotation widget should adopt/open it just as if the user had
  selected that layer directly in napari after it was registered.

Suggested implementation:

- Extend `_on_primary_shapes_layer_registered(...)` with a second path for
  active-layer adoption.
- Split the current early guard. Keep `self._is_opening_annotation_layer` as an
  immediate return, but do not return immediately just because
  `self._annotation_layer` or `self._annotation_session` exists; the active
  adoption path already handles no-session, clean-session, and dirty-session
  cases.
- Before the existing target-matching branch, check whether
  `binding.layer is viewer.layers.selection.active`.
- If the registered layer is currently active, call
  `_maybe_adopt_active_shapes_layer(binding.layer)` and return.
- Do not run active-layer adoption or registration repair while
  `self._is_handling_annotation_layer_removal` is true. That flag marks
  widget-owned discard/close/reload operations where transient napari active
  layers must not change the annotation target.
- Keep the existing target-specific registration behavior for the case where
  the widget already selected a matching edit-existing target. This existing
  branch should remain after the active-layer branch and can keep returning
  early when an annotation session already exists.
- Do not adopt styled shapes layers, point-rendered shapes layers, or unbound
  native napari Shapes layers.
- Do not reintroduce listening to `viewer_adapter.active_layer_changed`; this is
  specifically a binding-registration race for a layer that napari already made
  active.

Tests:

- Viewer-widget-style primary shapes load while Annotation is on `Create shapes`
  is adopted after binding registration if the newly registered layer is active.
- If the registered layer is not the active napari layer, Annotation does not
  adopt it.
- Styled or point-rendered shapes registrations are ignored.
- Dirty-session cancel/confirm behavior remains the same as Slice 1D when the
  already-active newly registered layer would switch annotation targets.
- Switching from dirty `new_shapes_1` to `new_shapes_2` through the Annotation
  widget asks for discard confirmation once. Clicking Discard annotations must
  not reenter the active-layer repair for the reloaded `new_shapes_1`; the
  widget should end on `new_shapes_2`.

### Slice 1F - Pending Native Import While Dirty

Status: implemented.

Problem:

- Native napari file imports, such as loading a shapes CSV through napari's
  own file-open UI, insert a plain `napari.layers.Shapes` layer before the
  Shapes Annotation widget can decide whether to adopt it.
- The widget only observes that layer after insertion via the viewer layer-list
  `inserted` event and then runs the deferred native adoption path.
- If the current annotation session is dirty, `_maybe_adopt_native_shapes_layer(...)`
  asks whether to discard the existing unsaved annotations. During that dialog,
  the newly imported native layer is already visible in napari.
- Because Harpy styling is applied only after successful adoption, the pending
  native layer appears with napari's default Shapes styling while the discard
  dialog is open.
- This is confusing because the UI visually looks as if the target has already
  changed before the user has answered the discard prompt.

Observed flow:

```text
User edits new_shapes_1
User opens a shapes CSV through napari's file-open UI
napari reads the CSV
napari inserts a native Shapes layer immediately
Annotation receives the layer-list inserted event
Annotation defers native adoption to the next Qt event-loop turn
_maybe_adopt_native_shapes_layer(...)
  sees dirty new_shapes_1
  opens the discard confirmation
pending native Shapes layer is already visible with default napari styling
```

Chosen behavior:

- Treat an unbound native Shapes layer inserted while the annotation session is
  dirty as a pending native adoption.
- If the user confirms discard:
  - discard the current dirty annotation session
  - adopt the pending native layer
  - normalize its transform
  - apply Harpy primary-shapes styling
  - register it as the new annotation-owned shapes layer
- If the user cancels:
  - keep the current dirty annotation session active
  - reactivate the current annotation layer
  - remove the pending imported native layer from the viewer, because the user
    declined switching Annotation to that import

Suggested implementation:

- Keep the existing deferred native-adoption mechanism in
  `_on_viewer_layer_inserted(...)`; do not try to intercept napari's file-open
  machinery before insertion.
- In `_maybe_adopt_native_shapes_layer(...)`, handle the dirty-session branch
  explicitly before adoption.
- For the cancel path, remove only the still-unbound pending native layer that
  triggered this adoption attempt. Do not remove Harpy-bound layers or layers
  that are no longer present in the viewer.
- After cancel, reactivate `self._annotation_layer` so the napari UI and the
  Annotation widget return to the dirty session the user kept.
- Guard against reentrant layer-removal/adoption callbacks while removing the
  pending native layer, using the existing annotation-layer removal guards or a
  narrowly scoped new guard if needed.
- Do not apply Harpy styling to a pending native layer before the user confirms
  adoption. Styling should remain a signal that Annotation owns the layer.

Tests:

- Dirty annotation session plus native Shapes import; user cancels discard:
  current annotation session remains dirty and active, the native import is
  removed, and the native layer remains unbound.
- Dirty annotation session plus native Shapes import; user confirms discard:
  current dirty layer is discarded, the native layer is adopted, Harpy styling
  is applied, and the layer is registered as annotation-owned.
- Clean annotation session plus native Shapes import keeps the existing
  behavior: the clean session can close and the native layer is adopted.

### Slice 1G - Hide Pending Native Import During Dirty Confirmation

Status: deferred.

Current status:

- Slice 1F fixes the most problematic half-adopted state. If the user cancels
  the dirty-session confirmation, Annotation removes the pending unbound native
  import and reactivates the existing dirty annotation layer.
- If the user confirms, the existing adoption path continues: the imported
  native layer is adopted, Harpy styling is applied, and the layer becomes the
  annotation-owned target.
- One visual artifact remains: napari inserts the native Shapes layer before
  Annotation can ask the dirty-session confirmation. Therefore, while the
  confirmation dialog is open, the pending native layer can briefly be visible
  with napari's default Shapes styling.

Why this remains:

- Annotation currently observes native file imports through the viewer
  layer-list `inserted` event. By the time this event reaches the widget, napari
  has already created and inserted the layer.
- Harpy styling is intentionally applied only after the user confirms adoption,
  because styling should signal that Annotation owns the layer.
- Intercepting napari's file-open/read path before layer insertion would be a
  broader integration change and is not needed for the current correctness fix.

Possible future behavior:

- Treat the inserted native layer as a hidden or temporarily removed pending
  adoption candidate while the dirty-session confirmation is shown.
- If the user confirms:
  - restore or keep the pending layer
  - adopt it
  - apply Harpy styling
  - register it as annotation-owned
- If the user cancels:
  - keep the current dirty annotation session active
  - leave the pending import removed or hidden
  - reactivate the current annotation layer

Implementation notes:

- Prefer a narrow widget-level solution over intercepting napari's reader
  machinery.
- If temporarily removing the pending layer, keep enough local state to restore
  the same layer object on confirm without losing its imported data, metadata,
  features, or transform.
- Guard any temporary removal/restoration from layer-removal and active-layer
  adoption callbacks so the widget does not reset the current dirty session.
- Keep Slice 1F's cancel behavior intact: cancel must not leave an unbound
  default-styled native import behind.

## Non-Goals

- Do not attach `_AnnotationLayerEditGuard` to arbitrary unbound native Shapes
  layers just because they are active.
- Do not make napari's layer selection silently change the save target when the
  current annotation layer has unsaved changes.
- Do not change global napari Shapes behavior outside annotation-owned sessions.
- Do not duplicate the anchor-drag or vertex-delete behavior tests here. Once
  the edit guard is attached, those paths are covered by the edit-anchor tests.

## Acceptance Criteria

- A user can select a compatible Harpy primary shapes layer in the napari UI and
  get the same hole-aware edit behavior as if they had selected it through the
  Shapes Annotation widget.
- The annotation widget remains the owner of save-target/session state.
- Dirty-session protection remains intact.
- The edit guard remains scoped to annotation-owned sessions.
