# Highlight Picked Object Before User-Class Annotation

Status: specified.

## Goal

When the Object Classification widget is used to add or remove custom
`user_class` annotations, the currently picked label instance should be visually
highlighted before the user presses `Add (A)` or `Remove (R)`.

The highlight should make the picked object easy to confirm without disturbing
the existing annotation flow:

- labels are still picked through the active labels layer;
- `Add (A)` and `Remove (R)` keep writing to the selected table row;
- existing `Color by` modes continue to show `user_class`, `pred_class`, or
  `pred_confidence`;
- multiscale labels still use the current custom mouse-pick fallback.

## Current Flow

The relevant selection path is already centralized.

`ObjectClassificationWidget` creates an `AnnotationController` with
`on_selected_instance_changed=self._on_selected_instance_changed`. The widget
then binds the controller from `_bind_current_selection(...)` whenever the
selected SpatialData, labels element, table, or coordinate system changes.

`AnnotationController` owns the picked instance id:

- it resolves the live primary labels layer through
  `ViewerAdapter.get_loaded_primary_labels_layer(...)`;
- it connects `layer.events.selected_label` when available;
- it also attaches a `mouse_drag_callbacks` picker because multiscale labels are
  not editable and cannot rely on napari pick mode;
- it clears napari's default `selected_label == 1` during bind so the first real
  click on label `1` is not ambiguous;
- it calls back into the widget via `_set_selected_instance_id(...)`.

The widget currently reacts in `_on_selected_instance_changed(...)` by clearing
annotation feedback and refreshing the selection card and annotation buttons.
There is no viewer-side visual highlight beyond napari's internal selected label
state.

Layer coloring is owned separately by `ViewerStylingController`. It builds a
`DirectLabelColormap` for the active `Color by` mode and refreshes labels
features. Annotation writes use row-scoped refresh paths where possible, so
selected-label highlighting should avoid forcing unnecessary full restyles after
every `Add`/`Remove`.

## Napari Capabilities

The local environment uses napari `0.7.1`. `Labels` layers expose the relevant
properties:

- `selected_label`
- `show_selected_label`
- `contour`

`show_selected_label` is not quite the desired UX. It filters the layer so only
the selected label remains visible. That is a very small implementation, but it
hides context after the first click, making it harder to choose the next object.

`contour` is useful for a highlight overlay. A second `Labels` layer can share
the primary layer's data object, use a transparent default colormap, and map only
the selected instance id to a bright color. With `contour > 0`, only the selected
object boundary is visible. This avoids copying the segmentation data and keeps
the primary layer's class/prediction color visible underneath.

One caveat: napari labels contours are a 2D display feature. For 3D rendering,
the overlay should fall back to a semi-transparent filled highlight, or the first
implementation can scope the contour behavior to 2D labels.

## Options

### Option 1: Native `show_selected_label`

Implementation would be tiny:

- set `labels_layer.show_selected_label = True` after a positive pick;
- set it back to `False` when selection clears or the widget unbinds.

This is not recommended as the primary implementation. It isolates the selected
label rather than highlighting it in context, so the user loses the surrounding
segmentation view while deciding what to annotate next.

### Option 2: Temporary Primary-Colormap Override

This would keep a `highlighted_instance_id` inside `ViewerStylingController` and
inject a bright color into the existing `DirectLabelColormap`.

Pros:

- no extra napari layer;
- no new viewer-adapter lifecycle work;
- works with all current `Color by` modes because they already use direct label
  colormaps.

Cons:

- the selected object is filled with the highlight color, so its real class or
  prediction color is hidden while selected;
- restoring the previous color has to be coordinated with full refreshes,
  row-scoped annotation refreshes, and color-mode changes.

This is a reasonable MVP if we want the smallest code change, but it is not the
best fit for annotation because it masks the semantic color exactly when the
user is checking the object.

### Option 3: Selected-Label Outline Overlay

This is the recommended implementation.

Create a small controller, for example `SelectedLabelHighlightController`, that
owns a viewer-only labels overlay above the active primary labels layer.

Behavior:

- bind to the current primary labels layer;
- create or update one highlight `Labels` layer that shares the primary layer's
  `data`;
- copy the primary layer transform/state needed for alignment, preferably from
  `primary_layer.as_layer_data_tuple()` with metadata adjusted for the overlay;
- set the overlay colormap to transparent for `None` and `0`, and bright for the
  selected instance id;
- set `overlay.contour = 2` for a boundary highlight;
- keep `overlay.visible = False` while no positive instance is selected;
- when a new instance is selected, update the overlay colormap and show it;
- keep the primary labels layer active after creating the overlay so picking
  continues to target the real labels layer.

Suggested default presentation:

- name: `<labels layer name> [selected object]`;
- color: high-contrast yellow, for example `#ffff00ff`;
- contour thickness: `2`;
- blending: `additive` or `translucent`;
- opacity: `1.0`;
- editable: `False`.

The important part is that the overlay does not own annotation state. It only
visualizes `AnnotationController.selected_instance_id`.

## Viewer Adapter Lifecycle

The cleanest version should integrate with the existing viewer adapter instead
of leaving the overlay as an unregistered layer.

Current labels bindings support only:

- `labels_role="primary"`
- `labels_role="styled"`

`styled` bindings require a table color `style_spec`, so they are not a natural
fit for a selection highlight. Add a third labels role, for example
`labels_role="highlight"`.

Expected adapter behavior:

- primary labels lookups remain unchanged because they already request
  `labels_role="primary"`;
- `primary_labels_layers_changed` should only emit for primary labels bindings;
- highlight labels should be removed by `remove_layers_for_sdata(...)` and
  `remove_layers_outside_coordinate_system(...)` along with other bindings for
  the same SpatialData/coordinate system;
- removing the highlight layer should not produce the current warning for
  unregistered layer removal;
- if the user manually removes the highlight layer, the controller can recreate
  it on the next selection.

A quick prototype could keep the overlay unregistered, but that risks stale
layers on SpatialData changes and warning logs during removal. A registered
`highlight` role is a better long-term fit.

## Widget Wiring

Add the highlight controller next to the existing annotation and styling
controllers in `ObjectClassificationWidget.__init__(...)`.

Bind it from `_bind_current_selection(...)` after the annotation controller has
resolved the active labels layer:

- pass the current primary labels layer;
- pass the selected SpatialData, labels name, and coordinate system if the
  adapter role needs them;
- clear or remove the previous overlay when the bound primary labels layer
  changes.

Update `_on_selected_instance_changed(...)`:

- call `highlight_controller.set_selected_instance_id(instance_id)`;
- then keep the existing feedback/status/button updates.

Do not clear the highlight after `Add (A)` or `Remove (R)` for the outline
overlay. The selected object remains picked, and the primary layer underneath can
still show the updated class color. If we choose the primary-colormap override
instead, then clearing the highlight after an annotation write may be preferable
so the user can see the new class color immediately.

On widget destruction or full unbind, call the controller shutdown/clear method
so the helper layer is removed or hidden.

## Tests

Add focused tests around selection and lifecycle:

- selecting a label creates/shows a highlight layer and keeps the primary labels
  layer active;
- the highlight colormap maps `0` and unknown labels to transparent and maps the
  selected id to the highlight color;
- selecting a different label updates the same highlight layer rather than
  creating duplicates;
- clearing or rebinding the labels selection hides/removes the highlight;
- multiscale mouse-pick fallback updates the highlight after calling the custom
  mouse callback;
- `Add (A)` and `Remove (R)` still update table state and primary labels styling
  while the highlight remains viewer-only;
- changing `Color by` refreshes the primary labels layer without removing the
  highlight;
- removing the primary labels layer cleans up or hides the highlight layer;
- if a `labels_role="highlight"` binding is added, viewer-adapter tests should
  confirm primary labels lookup ignores highlight layers and cleanup removes
  them with the selected SpatialData/coordinate system.

Useful verification commands:

- `.venv/bin/pytest tests/test_widget.py tests/test_viewer_styling.py tests/test_viewer_adapter.py`
- `.venv/bin/ruff check src/napari_harpy/widgets/object_classification src/napari_harpy/viewer/adapter.py tests/test_widget.py tests/test_viewer_adapter.py`

## Non-Goals

- Do not write highlight state into the annotation table.
- Do not change the meaning of `user_class`, `pred_class`, or
  `pred_confidence` colors.
- Do not copy or materialize large lazy labels arrays just to build the
  highlight.
- Do not replace the existing `AnnotationController` selection model.

## Open Decisions

- Confirm the desired highlight style: outline-only is recommended, but filled
  highlight is cheaper.
- Confirm whether 3D labels should get a filled fallback in the first pass.
- Decide whether the highlight layer should be visible in the napari layer list
  or named in a way that makes it clearly temporary.
