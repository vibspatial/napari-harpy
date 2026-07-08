# Highlight Picked Object Before User-Class Annotation

Status: investigation updated; chosen direction specified.

## Goal

When the Object Classification widget is used to add or remove custom
`user_class` annotations, the currently picked label instance should be visually
highlighted before the user presses `Add (A)` or `Remove (R)`.

The highlight should make the picked object easy to confirm without disturbing
the existing annotation flow:

- labels are still picked through the active primary labels layer;
- `Add (A)` and `Remove (R)` keep writing to the selected table row;
- existing `Color by` modes continue to show `user_class`, `pred_class`, or
  `pred_confidence`;
- multiscale labels still use the current custom mouse-pick fallback;
- no extra labels layer should be created just to render the highlight.

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
features. Annotation writes use row-scoped refresh paths where possible, so the
highlight should integrate with these existing color refresh paths instead of
adding a separate viewer layer.

## Napari Capability Check

The local environment uses napari `0.7.1`. `Labels` layers expose the relevant
properties:

- `selected_label`
- `show_selected_label`
- `contour`

`show_selected_label` is not the desired UX. It filters the layer so only the
selected label remains visible. That hides surrounding objects, which makes it
harder to decide what to annotate next.

`contour` is also not enough for the no-extra-layer requirement. Napari's
`Labels.contour` is a layer-wide display mode: it renders contours for labels on
that layer, not only for the currently selected label. A selected-only outline
can be built with a second labels overlay, but we explicitly do not want to
create an extra labels layer for this feature.

The practical no-extra-layer route is therefore to highlight the selected
instance through the primary layer's existing `DirectLabelColormap`. This keeps
the object visible in context and avoids copying or sharing segmentation data in
a helper layer. The tradeoff is that the selected object is filled with the
highlight color while selected, rather than receiving an outline-only highlight.

## Chosen Direction: Primary-Layer Highlight

Implement the highlight inside `ViewerStylingController` by injecting one
selected-instance color into the primary labels layer colormap.

Behavior:

- add selected-highlight state to `ViewerStylingController`, for example
  `set_highlighted_instance_id(instance_id: int | None)`;
- keep `AnnotationController` as the source of truth for which object is picked;
- have `ObjectClassificationWidget._on_selected_instance_changed(...)` forward
  the picked instance id to `ViewerStylingController`;
- when a positive instance id is selected, rebuild or update the primary labels
  layer colormap so that instance id maps to a high-contrast highlight color;
- when selection clears, rebinding happens, or the widget unbinds, restore the
  normal semantic colormap by refreshing the primary layer colors;
- do not create a highlight `Labels` layer;
- do not add a new viewer-adapter `labels_role`;
- do not write highlight state into the table.

Suggested default presentation:

- color: high-contrast yellow, for example `#ffff00ff`;
- no opacity/layer changes;
- no `show_selected_label`;
- no `contour` changes.

The highlight is viewer-only state. It should not affect annotation table
columns, classifier state, persistence state, or exported classifier bundles.

## Styling Integration

All color application should flow through `ViewerStylingController` so the
highlight is consistently preserved across current styling paths.

Full refresh path:

- `refresh_layer_colors(...)` should build the normal color dictionary for the
  active `Color by` mode;
- just before assigning `DirectLabelColormap`, apply the selected-instance
  highlight override if one is active.

Row-scoped user-class refresh path:

- `refresh_user_class_colormap_and_feature(...)` should keep using the narrow
  direct-annotation update when possible;
- if the edited instance is highlighted, the candidate color dictionary should
  first reflect the new semantic `user_class` value, then apply the highlight
  override;
- this keeps the highlighted object highlighted while selected, and the updated
  semantic color becomes visible when the highlight clears.

Prediction color modes:

- `refresh_user_class_feature(...)` can remain feature-only because prediction
  color modes do not repaint on user-class edits;
- the active highlight remains in the existing colormap until selection changes,
  color mode changes, prediction colors refresh, or selection clears.

Color mode changes:

- changing `Color by` should call the normal full refresh path;
- the selected highlight override should be applied on top of the newly selected
  semantic color mode.

This first implementation can choose correctness over micro-optimization by
rebuilding the colormap on selection changes. That touches label color metadata,
not the labels array itself. If selection repaint becomes noticeably expensive
on very large tables, a later optimization can cache and restore only the
previous highlighted colormap entry.

## Widget Wiring

Update `_on_selected_instance_changed(...)`:

- call `self._viewer_styling_controller.set_highlighted_instance_id(instance_id)`;
- then keep the existing feedback/status/button updates.

Bind/unbind behavior:

- when `ViewerStylingController.bind(...)` moves to a different primary labels
  layer, clear highlight state from the previous binding;
- if there is no active labels layer, highlight state should be empty;
- widget teardown or full unbind should clear the highlight by rebinding or
  explicitly calling the clear path.

Annotation writes:

- do not clear the highlight after `Add (A)` or `Remove (R)`;
- the object remains selected, so it should remain highlighted;
- the updated class color is still stored in the normal colormap calculation and
  becomes visible when the user selects a different object or clears selection.

## Tests

Add focused tests around selection and styling:

- selecting a label updates the primary labels layer colormap with the highlight
  color for that instance;
- no extra labels layer is created when a label is selected;
- clearing selection restores the normal semantic colormap;
- selecting a different label moves the highlight to the new id and restores the
  previous id's normal color;
- full `Color by` refreshes preserve the active selected-instance highlight;
- row-scoped `user_class` annotation refresh preserves the highlight while
  keeping the new semantic color available after highlight clear;
- prediction color modes keep their existing feature-only annotation refresh
  behavior;
- the multiscale mouse-pick fallback still routes through
  `_on_selected_instance_changed(...)`, so it updates the same highlight state.

Useful verification commands:

- `.venv/bin/pytest tests/test_viewer_styling.py tests/test_widget.py`
- `.venv/bin/ruff check src/napari_harpy/widgets/object_classification tests/test_viewer_styling.py tests/test_widget.py`

## Non-Goals

- Do not create a second labels layer for selected-object highlighting.
- Do not write highlight state into the annotation table.
- Do not change the meaning of `user_class`, `pred_class`, or
  `pred_confidence` colors.
- Do not copy or materialize large lazy labels arrays just to build the
  highlight.
- Do not replace the existing `AnnotationController` selection model.
- Do not implement selected-only outline rendering through private napari/vispy
  internals in this slice.

## Open Decisions

- Confirm the exact highlight color after trying it against the dark napari
  theme and the current class palettes.
- Decide later whether the filled highlight is sufficient, or whether a future
  custom visual is worth the added maintenance cost for outline-only rendering.
