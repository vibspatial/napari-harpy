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

Status: proposed.

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
