# Viewer Widget Status Card Roadmap

Date: 2026-05-18

## Goal

Introduce a dedicated viewer status-card module, matching the pattern already
used by object classification and feature extraction, so that viewer status
copy and severity decisions are no longer embedded throughout
`src/napari_harpy/widgets/viewer/widget.py`.

The first implementation should be behavior-preserving. It should not change
where feedback appears, the current titles, or the current message wording
unless a test explicitly needs updating for a clearer contract.

## Findings

### Shared status-card rendering

All widgets share the same final renderer:

- `src/napari_harpy/widgets/shared_styles.py`
- `StatusCardKind = Literal["info", "warning", "success", "error"]`
- `set_status_card(label, title=..., lines=..., kind=..., tooltip_message=...)`

This helper owns the actual Qt label styling, HTML formatting, tooltip
formatting, and visibility. The dedicated widget-specific `status_card.py`
modules should continue to return data-only specs and leave rendering to the
widget.

### Object classification pattern

`src/napari_harpy/widgets/object_classification/status_card.py` defines:

- `_ObjectClassificationStatusCardSpec`
- `_LabelsLayerPreparationResult`
- `build_object_classification_selection_status_card_spec(...)`
- `build_object_classification_classifier_preparation_card_spec(...)`
- `build_object_classification_classifier_feedback_card_spec(...)`

The widget gathers current UI/controller state, asks the status-card module for
a spec, then renders through `_apply_status_card_spec(...)` in
`src/napari_harpy/widgets/object_classification/widget.py`.

Important pattern:

- state collection stays in the widget
- status decision logic and user-facing copy live in `status_card.py`
- the spec can be `None` when a card should be hidden
- the renderer is a tiny adapter from spec to `set_status_card(...)`

Object classification still has a few direct feedback paths, such as
annotation and persistence feedback, but the complex selection and classifier
status policy has already been extracted.

### Feature extraction pattern

`src/napari_harpy/widgets/feature_extraction/status_card.py` defines:

- `_FeatureExtractionStatusCardEntry`
- `_FeatureExtractionStatusCardSpec`
- `build_feature_extraction_status_card_entries(...)`
- `build_feature_extraction_selection_status_card_spec(...)`
- `build_feature_extraction_controller_feedback_card_spec(...)`

The widget builds small input structures from its internal state, then applies
the resulting spec in
`src/napari_harpy/widgets/feature_extraction/widget.py`.

Important pattern:

- the module is Qt-free
- long or repeated identifiers are shortened through `format_feedback_identifier`
- visible lines and tooltip lines are built together so shortened text still has
  full context
- controller feedback messages are normalized by stripping a fixed prefix before
  rendering

This is the closest pattern for the viewer widget, because the viewer has many
status variants with similar titles, palette fallback messages, and optional
tooltips.

### Current viewer status-card state

`src/napari_harpy/widgets/viewer/widget.py` currently mixes UI construction,
viewer actions, adapter calls, and feedback copy.

Existing real status cards:

- `ViewerWidget.action_feedback_label` is created near the top of the widget and
  rendered by `_set_action_feedback(...)`.
- `_set_action_feedback(...)` is a small wrapper around `set_status_card(...)`.
- `_clear_action_feedback(...)` manually clears text, tooltip, style, and
  visibility.
- `PointsValueWidget` in `src/napari_harpy/widgets/viewer/points_widget.py`
  renders its section-local status with `set_status_card(...)` directly.

Existing plain helper/status text:

- `_LabelsCardWidget.action_status_label` shows action-preview text such as
  `Action: add/update primary labels layer`.
- `_ShapesCardWidget.action_status_label` shows action-preview text such as
  `Action: add/update styled shapes layer for column "..."`.
- `_ImageCardWidget.channel_warning_label` is a styled warning label, but not a
  shared status card.

Viewer feedback copy currently lives inside action methods:

- points layer success/warning copy in `_render_points_loaded_status(...)`
- primary labels success/error copy in `_add_or_update_primary_labels_layer(...)`
- styled labels overlay success, palette, coercion, and error copy in
  `_add_or_update_styled_labels_layer(...)`
- primary shapes success/skipped-geometry copy in `_add_or_update_primary_shapes_layer(...)`
- styled shapes palette, coercion, skipped-geometry, and error copy in
  `_add_or_update_styled_shapes_layer(...)`
- image stack/overlay success and error copy in `_add_or_update_image_layer(...)`
- SpatialData load errors in `_open_spatialdata(...)`

The tests in `tests/test_viewer_widget.py` already assert many of these status
titles, severities, style colors, and message fragments. That is good coverage
for a behavior-preserving extraction, but it also means the refactor should
avoid accidental copy changes.

## Recommendation

Add:

```text
src/napari_harpy/widgets/viewer/status_card.py
```

Use it as the viewer-specific home for data-only status specs and builders.
Keep `set_status_card(...)` in `shared_styles.py`; the new module should not
import Qt widgets or render anything directly.

Recommended first-pass scope:

- extract global viewer action feedback specs
- preserve the current global `action_feedback_label`
- preserve existing message wording and severities
- leave card-local feedback placement for a follow-up ticket, but make the spec
  reusable for any target `QLabel`
- leave `PointsValueWidget.show_status(...)` in place unless points feedback is
  being refactored in the same pass

This gives the viewer the same separation as object classification and feature
extraction without turning the change into a UI redesign.

## Proposed Module Shape

Start with a small data structure:

```python
@dataclass(frozen=True)
class _ViewerStatusCardSpec:
    title: str
    lines: tuple[str, ...]
    kind: StatusCardKind
    tooltip_message: str | None = None
```

Then add focused builders. Suggested first set:

- `build_viewer_error_card_spec(title: str, error: str) -> _ViewerStatusCardSpec`
- `build_missing_context_card_spec(title: str) -> _ViewerStatusCardSpec`
- `build_points_layer_card_spec(load_result, layer_result) -> _ViewerStatusCardSpec`
- `build_primary_labels_loaded_card_spec(labels_name, coordinate_system) -> _ViewerStatusCardSpec`
- `build_styled_labels_card_spec(request, result, coordinate_system) -> _ViewerStatusCardSpec`
- `build_primary_shapes_loaded_card_spec(shapes_name, coordinate_system, skipped_geometry_count) -> _ViewerStatusCardSpec`
- `build_styled_shapes_card_spec(request, result, coordinate_system, skipped_geometry_count) -> _ViewerStatusCardSpec`
- `build_image_loaded_card_spec(image_name, coordinate_system, mode, channels=None) -> _ViewerStatusCardSpec`

Keep generic error builders boring. The more valuable extraction is the
success/warning/info decision logic currently repeated in the labels, shapes,
image, and points action paths.

Optionally add plain text builders for action previews:

- `build_labels_action_preview_text(...) -> str`
- `build_shapes_action_preview_text(...) -> str`

Those previews are not shared status cards today, so they can be moved later if
the first pass should stay narrower.

## Implementation Plan

1. Add `src/napari_harpy/widgets/viewer/status_card.py`.

   Define `_ViewerStatusCardSpec` and helper builders. Use
   `format_feedback_identifier(...)` inside the builders so tooltip behavior
   stays with the message it belongs to.

2. Add a renderer adapter to `ViewerWidget`.

   Mirror the other widgets:

   ```python
   def _apply_status_card_spec(self, label: QLabel, spec: _ViewerStatusCardSpec | None) -> None:
       ...
   ```

   Then make `_set_action_feedback(...)` either delegate to that helper or
   replace it with `_set_action_feedback_spec(...)`.

3. Refactor action methods one family at a time.

   Suggested order:

   - image loaded/error feedback
   - primary labels and primary shapes loaded feedback
   - styled labels and styled shapes palette/coercion feedback
   - points loaded feedback
   - SpatialData load error feedback

   After each family, run the focused viewer-widget tests so copy regressions
   are caught early.

4. Keep adapter calls and side effects in `widget.py`.

   The status module should not call `ViewerAdapter`, activate layers, inspect
   Qt labels, or know about layout. It should receive already-computed facts
   such as `created`, `palette_source`, `coercion_applied`, selected color
   source, and skipped geometry count.

5. Preserve all current title/kind combinations.

   Important current combinations include:

   - `Image Loaded` as success
   - `Image Load Error` as error
   - `Labels Loaded` as success
   - `Colored Overlay Created` / `Updated` as success
   - `Colored Overlay Created With Warning` as warning
   - `Colored Overlay Created` as info for missing stored palette
   - `Shapes Loaded With Warning` when geometries are skipped
   - `Styled Shapes Created With Warning` for invalid palettes, coercion, or
     skipped geometries
   - `Points Layer Created With Warning` when sampling or categorical coloring
     limits produce warnings

6. Add unit tests for the new builders.

   Suggested file:

   ```text
   tests/test_viewer_status_card.py
   ```

   Cover:

   - identifier shortening and tooltip preservation
   - styled labels palette source variants
   - styled shapes palette source variants
   - skipped geometry warning escalation
   - image stack versus overlay text
   - points warning escalation
   - hidden/no-op specs if any builder returns `None`

7. Keep existing widget tests as integration coverage.

   `tests/test_viewer_widget.py` should continue to assert that clicking viewer
   controls renders the expected status card. Once the builder tests exist,
   the widget tests can focus on plumbing and fewer exact message combinations.

## Follow-Up: Card-Local Feedback

`Roadmap/viewer_widget/tickets.md` already describes moving viewer card actions
to card-local feedback. A dedicated `viewer/status_card.py` makes that easier:
the same spec builders can render into either the global
`action_feedback_label` or a local card label.

For that follow-up, add card methods such as:

```python
def set_feedback(self, spec: _ViewerStatusCardSpec | None) -> None: ...
def clear_feedback(self) -> None: ...
```

Then route image, labels, and shapes action feedback to the card that emitted
the request while keeping global feedback for SpatialData loading and other
dataset-level failures.

## Acceptance Criteria

- [ ] `src/napari_harpy/widgets/viewer/status_card.py` exists and is Qt-free.
- [ ] Viewer status-card copy and severity decisions are built by that module.
- [ ] `ViewerWidget` keeps responsibility for state collection, adapter calls,
      layer activation, and rendering specs into labels.
- [ ] Existing viewer feedback titles, severities, and tooltips are preserved.
- [ ] Builder-level tests cover success, info, warning, and error variants.
- [ ] Existing `tests/test_viewer_widget.py` feedback assertions still pass.
- [ ] The implementation does not change feedback placement unless the
      card-local feedback follow-up is intentionally included.

