# Listen To Active Layer Change

Status: investigation.

Goal: make annotation-layer edit behavior consistent when a user selects a
compatible shapes layer directly in the napari layer list, not only through the
Shapes Annotation widget.

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

## Why This Happens

The Shapes Annotation widget currently treats its own annotation session as the
source of truth. This is intentional for saving: an annotation session locks the
target shapes name, coordinate system, source GeoDataFrame metadata, table-link
warning state, and clean snapshot.

The widget does not currently subscribe to raw napari active-layer selection
changes. Selecting a layer in the napari layer list does not by itself call
`_open_existing_annotation_layer(...)`, `_adopt_native_shapes_layer(...)`, or
`_annotation_edit_guard.attach(...)`.

The viewer adapter has an `active_layer_changed` signal, but it is emitted by
`ViewerAdapter.activate_layer(...)`. It is not currently used by the Shapes
Annotation widget as a general bridge from napari UI layer selection into an
annotation session.

## Important Design Boundary

We should not globally patch every napari `Shapes` layer. The guard changes
interactive edit behavior and, for deletion, depends on Harpy's annotation
semantics. It should remain scoped to layers that the annotation workflow can
save safely.

The missing behavior is not "always guard all shapes layers"; it is:

> when the user selects a compatible Harpy primary shapes layer in napari, the
> annotation widget should safely adopt/open that layer as the active annotation
> session, then attach the guard.

## Suggested Follow-Up Slice

### Slice 1 - Adopt Compatible Active Shapes Layer

Status: not implemented.

Goal: when the napari active layer changes to a compatible primary shapes layer,
open/adopt it as the Shapes Annotation widget's active edit session when it is
safe to do so.

Suggested scope:

- Listen to active layer changes from the napari viewer.
- Prefer napari's real selection event if available, likely
  `viewer.layers.selection.events.active`.
- If using `ViewerAdapter.active_layer_changed`, make sure it covers user-driven
  napari UI selection, not only programmatic `ViewerAdapter.activate_layer(...)`.
- On active-layer change, inspect the active layer:
  - it must be a `napari.layers.Shapes` layer
  - it must have a `ShapesLayerBinding`
  - binding must belong to the current `SpatialData`
  - binding must be `element_type == "shapes"`
  - binding must be `shapes_role == "primary"`
  - binding must be `shapes_rendering_mode == "shapes"`
  - binding must match the currently selected coordinate system
  - binding must have no incompatible style spec
- If the active layer is already `self._annotation_layer`, do nothing.
- If there is no current annotation session, adopt/open the active layer as an
  edit-existing session:
  - update the Shapes widget combo to the bound shapes element
  - validate the source GeoDataFrame
  - create `_ShapesAnnotationSession`
  - attach `_AnnotationLayerEditGuard`
  - capture the clean snapshot
  - refresh save/create state
- If a clean annotation session is active for a different layer, close it and
  adopt the new active layer.
- If a dirty annotation session is active for a different layer, do not switch
  silently:
  - prompt the user with the existing discard confirmation flow, or
  - leave the current session active and show a warning/status message

Non-goals:

- Do not attach `_AnnotationLayerEditGuard` to arbitrary unbound native Shapes
  layers just because they are active.
- Do not make napari's layer selection silently change the save target when the
  current annotation layer has unsaved changes.
- Do not change global napari Shapes behavior outside annotation-owned sessions.

Tests for this slice:

- Selecting a compatible loaded primary shapes layer in napari opens/adopts it
  in the Shapes Annotation widget and attaches `_AnnotationLayerEditGuard`.
- After napari-driven adoption, anchor drag synchronization works on the active
  layer.
- After napari-driven adoption, hole-aware vertex deletion and the Slice 6C
  cache rebuild work on the active layer.
- Selecting an incompatible layer does not attach the guard.
- Selecting a layer for a different coordinate system does not attach the guard.
- Selecting a compatible layer while a clean different annotation session is
  active closes the clean session and adopts the selected layer.
- Selecting a compatible layer while a dirty different annotation session is
  active does not silently switch without confirmation.
- The Shapes widget combo and save button reflect the adopted active layer.

## Acceptance Criteria

- A user can select a compatible Harpy primary shapes layer in the napari UI and
  get the same hole-aware edit behavior as if they had selected it through the
  Shapes Annotation widget.
- The annotation widget remains the owner of save-target/session state.
- Dirty-session protection remains intact.
- The edit guard remains scoped to annotation-owned sessions.
