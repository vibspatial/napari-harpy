# Annotation Discoverability: Tooltips And User Guidance

Status: implemented.

Goal: make the Shapes Annotation widget easier to discover for normal napari
users, especially for two non-obvious workflows:

- using Space-pan while drawing large precise annotations;
- using `Create holes` to convert selected inner polygons into polygon holes.

## Current Situation

The annotation widget currently keeps the UI compact:

- it has a status card;
- it has `Create layer`, `Create holes`, and `Save shapes` action buttons;
- it gives useful success/error feedback after actions run;
- it does not currently explain Space-pan drawing or the Create holes workflow
  before the user tries them.

That means `Create holes` is discoverable mostly by clicking the button and
reading validation feedback. Space-pan is even less discoverable because the
feature intentionally behaves like normal napari canvas interaction once the
user knows to hold Space.

## Recommendation

Use lightweight in-widget hints for immediate discovery.

Do not add a large permanent instruction block to the widget. The widget is
already compact, and always-visible tutorial text would become noise after the
first successful use.

Do not add README or external documentation in this slice. The first fix should
stay inside the widget.

## In-Widget Help

Add concise tooltips to the action buttons.

Suggested `Create layer` tooltip:

```text
Create an editable annotation Shapes layer for the selected coordinate system.
```

Suggested `Create holes` tooltip:

```text
Select one shell polygon and one or more polygons fully inside it, then click Create holes. Shift-click polygons to add them to the selection. The largest selected polygon becomes the shell; selected inner polygons become holes.
```

Suggested `Save shapes` tooltip:

```text
Save the current annotation layer back to the selected SpatialData shapes element.
```

For Space-pan, there is no dedicated button, so a tooltip alone is not enough.
Add a short tip to the normal annotation-ready status card when an annotation
layer is open.

Do not add a separate `TIP` status card for now. The current widget has a
single status-card label, and adding a second persistent card would introduce
extra layout and lifecycle questions. Instead, keep the existing one-card
pattern and add the Space-pan tip to both normal editable-layer readiness
states:

- create-new annotation layer:

```text
Draw shapes in the viewer, then click Save shapes.
Tip: while drawing in polygon, path, polyline or lasso mode, hold Space and drag to pan without ending the shape.
```

- edit-existing annotation layer:

```text
Edit shapes layer "..." in coordinate system "...".
Tip: while drawing in polygon, path, polyline or lasso mode, hold Space and drag to pan without ending the shape.
```

This should appear only in active annotation-layer states, not in
target-selection readiness cards. Existing warning/success cards, such as
Create holes errors or save results, should continue to replace the normal
readiness card while they are current.

## Implementation Notes

- Use the existing shared `format_tooltip(...)` helper for button tooltips.
- Keep tooltip text concise enough to fit Qt's normal tooltip behavior.
- Keep Create holes validation/error messages as they are; the tooltip should
  guide the happy path, while the status card should continue to explain
  rejected selections.
- Do not mention unsupported internals such as anchor/separator encoding in the
  user-facing text.
- Say "selected polygons" or "selected shape rows", not "selected vertices";
  Create holes consumes `layer.selected_data`, not transient vertex edit
  targets.

## Suggested Slice

Slice 1: Widget Tooltips And Status Tip

Status: implemented.

- Add tooltips to `Create layer`, `Create holes`, and `Save shapes`.
- Extend both normal editable-layer status cards with the Space-pan tip:
  `build_annotation_layer_ready_card_spec()` for create-new sessions and
  `build_annotation_existing_shapes_opened_card_spec(...)` for edit-existing
  sessions.
- Add tests that assert the button tooltips are present and that the ready
  status card includes the Space-pan tip for both create-new and edit-existing
  annotation layers.
