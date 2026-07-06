# Annotation Discoverability: Tooltips And User Guidance

Status: recommendation.

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
Select one shell polygon and one or more polygons fully inside it, then click Create holes. The largest selected polygon becomes the shell; selected inner polygons become holes and their rows are removed.
```

Suggested `Save shapes` tooltip:

```text
Save the current annotation layer back to the selected SpatialData shapes element.
```

For Space-pan, there is no dedicated button, so a tooltip alone is not enough.
Add a short tip to the normal annotation-ready status card when an annotation
layer is open.

Suggested status-card text:

```text
Tip: while drawing polygon, path, polyline, or lasso, hold Space and drag to pan without ending the shape.
```

This should appear only in the active annotation-layer state, not in every
target-selection readiness card.

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

- Add tooltips to `Create layer`, `Create holes`, and `Save shapes`.
- Extend the annotation-layer-ready status card with the Space-pan tip.
- Add tests that assert the button tooltips are present and that the ready
  status card includes the Space-pan tip when a layer is open.
