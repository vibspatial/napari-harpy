# Annotation Performance Roadmap

## Context

Annotating `cell_labels_global_ROI1` in
`/Users/arne.defauw/VIB/DATA/test_data/sdata_xenium_full_data_core.zarr` feels laggy
because the current object-classification path performs multiple broad updates
after every add/remove annotation.

The selected Xenium table has roughly 407k cell rows. For a single user-class
edit, the current flow can:

- normalize and rewrite the whole `user_class` categorical column
- rebuild a label-to-color mapping for every cell in the active segmentation
- refresh layer features for every cell
- mark classifier outputs dirty and schedule classifier retraining

The goal is to isolate these issues and fix them one at a time. Avoid a large
"smart fast path" that couples annotation, coloring, classifier scheduling, and
napari lifecycle concerns in one difficult-to-maintain change.

## Principles

- Keep every phase independently reviewable and revertible.
- Keep the existing full-refresh paths for binding, table changes, color-mode
  changes, reload, and classifier prediction updates.
- Optimize the common `user_class` add/remove interaction first.
- Prefer simple data structures and explicit tests over hidden mutable state.
- Do not introduce private napari-event manipulation unless a phase explicitly
  proves it is needed.
- Preserve persisted table semantics: `user_class` remains categorical, class
  categories remain sorted, and `user_class_colors` remains aligned to
  categories.

## Phase 0: Guardrails

### Goal

Keep the annotation-performance work scoped and easy to review before changing
production code.

### Scope

- Do not add benchmark code yet.
- Do not commit large benchmark data.
- Do not make production changes in this phase.
- Use the current qualitative lag report and previous investigation as enough
  context to start with the smallest low-risk implementation slice.

### Deferred Benchmarking

Benchmark/dev-note work is deferred to the last phase, after the independent
fixes have landed. This avoids spending time polishing measurement scaffolding
before we know which fixes remain necessary.

### Acceptance Criteria

- The roadmap is split into independent phases.
- Phase 1 can start without introducing benchmark code.

## Phase 1: Sparse `user_class` Labels Colormap

### Goal

Make `user_class` coloring sparse so unlabeled cells use the default color and
only explicitly annotated cells receive explicit color entries.

This is the easiest win because most cells are usually unlabeled during manual
annotation. We do not need one `DirectLabelColormap.color_dict` entry per cell
just to say "unlabeled".

### Current Problem

`ViewerStylingController.refresh_layer_colors()` builds a full dictionary:

```text
{None: unlabeled_color, 0: transparent, 1: ..., 2: ..., ..., 407124: ...}
```

When almost all cells are unlabeled, this duplicates the same color hundreds of
thousands of times.

### Proposed Behavior

For `color_by == "user_class"`:

- Build `DirectLabelColormap` with:
  - `None: unlabeled_color`
  - `0: transparent`
  - one explicit entry only for labels whose `user_class != 0`
- Let napari use the `None` default color for all missing/unlabeled label ids.
- Keep full explicit mappings for:
  - `pred_class`, unless separately optimized later
  - `pred_confidence`, because continuous values are expected to differ per
    label

### Non-Goals

- Do not add benchmark code in this phase.
- Do not change the annotation table write path.
- Do not add the auto-training checkbox.
- Do not mutate an existing napari colormap in place.
- Do not update only one `layer.features` row.
- Do not change classifier prediction refresh behavior.
- Do not change viewer-widget styled overlay coloring outside object
  classification.

This phase is intentionally only about reducing the size of the `user_class`
labels colormap produced by the existing full-refresh path.

### Files

- `src/napari_harpy/widgets/object_classification/viewer_styling.py`
- `tests/test_widget.py`

### Behavior Matrix

| Color mode | Colormap strategy in Phase 1 | Reason |
| --- | --- | --- |
| `user_class` | Sparse: default color for unlabeled, explicit entries for labeled objects only | Most cells are unlabeled during manual annotation |
| `pred_class` | Keep existing full explicit mapping | Predictions can affect many cells and will be considered separately |
| `pred_confidence` | Keep existing full explicit mapping | Continuous confidence values can differ per cell |

### Implementation Notes

- Introduce a small helper such as:

```python
def _base_labels_color_dict(default_color: Any) -> dict[int | None, Any]:
    return {None: default_color, 0: "transparent"}
```

- Build the base color dictionary after resolving the effective unlabeled color
  from `_get_class_color_lookup(...)`.
- In the `user_class` branch, skip entries where `class_id == UNLABELED_CLASS`.
- Keep the existing loop shape for `pred_class` and `pred_confidence`.
- Keep current class-palette lookup behavior unchanged.
- Do not mutate an existing colormap in this phase; assign a new sparse
  `DirectLabelColormap` through the existing full-refresh path.
- Keep the layer `refresh()` call exactly where it currently happens.
- Avoid clever abstractions. A small helper plus a clearly separated
  `user_class` branch is preferable to a generalized color-strategy framework.

Suggested structure:

```python
if self._color_by == COLOR_BY_PRED_CONFIDENCE:
    # existing continuous behavior
elif self._color_by == COLOR_BY_USER_CLASS:
    class_color_lookup = ...
    unlabeled_color = ...
    color_dict = _base_labels_color_dict(unlabeled_color)
    for instance_id in instance_ids:
        class_id = int(class_by_instance.at[instance_id])
        if class_id == UNLABELED_CLASS:
            continue
        color_dict[instance_id] = class_color_lookup.get(class_id, unlabeled_color)
else:
    # existing pred_class behavior
```

### Edge Cases

- If `user_class` is missing, `_get_region_feature_rows()` already returns all
  unlabeled values. The sparse colormap should then contain only `{None, 0}`.
- If all objects are unlabeled, the sparse colormap should contain only
  `{None, 0}`.
- If a class palette is missing or incomplete, existing fallback behavior from
  `_get_class_color_lookup(...)` should still choose stable default colors.
- Tests should use `layer.colormap.map(unlabeled_label_id)` for unlabeled cells,
  because unlabeled label ids are no longer expected to be keys in
  `color_dict`.
- Background label `0` must remain transparent and must not inherit the
  unlabeled object color.

### Tests

Add or update widget tests so that after annotating label `5`:

- `layer.colormap` is a `DirectLabelColormap`
- `layer.colormap.color_dict` has only `{None, 0, 5}` for one labeled object
- `layer.colormap.map(6)` returns the unlabeled/default color
- label `5` and label `6` have different colors when label `5` is assigned a
  positive user class
- background label `0` remains transparent

Add or keep regression coverage that:

- `pred_class` mode still colors predicted classes as before
- `pred_confidence` mode still creates distinct colors for different confidence
  values
- layer features still expose `instance_id` and `user_class` after annotation

### Manual QA

After implementation, manually inspect:

- loading `cell_labels_global_ROI1`
- selecting object classification
- applying a positive user class to one object
- verifying the selected object changes color while other unlabeled objects keep
  the unlabeled color
- switching to `pred_class` and `pred_confidence` modes still works

Manual QA should not require writing to zarr.

### Review Checklist

- The diff is limited to `viewer_styling.py` and widget tests.
- No classifier scheduling code changes are included.
- No annotation table mutation code changes are included.
- No private napari cache/event manipulation is introduced.
- The sparse behavior is expressed in tests, not only in comments.

### Acceptance Criteria

- A full `user_class` color refresh no longer creates entries for every
  unlabeled object.
- Existing user-class color behavior is unchanged visually.
- Existing prediction and confidence coloring tests still pass.
- Phase 1 remains independently revertible without affecting later phases.

## Phase 2A: Split Classifier Status From Prediction Table Refresh

### Goal

Avoid refreshing the labels layer for classifier status-only changes, while
still refreshing prediction coloring whenever classifier outputs actually change.

This phase clarifies the overloaded phrase "classifier changed". A classifier
status change is not the same thing as a prediction table change.

### Current Problem

`ClassifierController._set_status(...)` calls the widget's classifier-state
callback for every status transition:

- model became stale
- training was scheduled
- training started
- training failed
- training finished
- inputs are not trainable

The widget currently handles that callback by updating feedback, refreshing the
labels layer, and updating classifier controls. That means status-only events
such as "model is stale" or "training is scheduled" rebuild label colors and
features even though `table.obs["pred_class"]` and
`table.obs["pred_confidence"]` did not change.

### Event Model

Treat these as separate event types:

| Event | Meaning | Labels-layer refresh? |
| --- | --- | --- |
| User annotation changed | `user_class` changed for the selected object | Yes, for annotation coloring/features |
| Classifier status changed | text/button state changed, but prediction columns did not | No |
| Classifier prediction table changed | `pred_class`/`pred_confidence` were written or cleared | Yes |
| Binding/reload/color mode changed | the layer/table/color source changed | Yes |

So yes: when retraining produces new predictions, we should recolor prediction
views. The key is that the recolor should be tied to the prediction-table update,
not to every classifier status message.

### How We Know Prediction/Table State Changed

Do not infer table changes from classifier status text. The classifier
controller is the only code path that writes classifier outputs in the widget,
so it should emit explicit callbacks immediately after those writes.

Relevant classifier-side table mutations are:

- prediction columns changed:
  - `table.obs["pred_class"]`
  - `table.obs["pred_confidence"]`
- classifier metadata changed:
  - `table.uns[CLASSIFIER_CONFIG_KEY]`
  - fitted/exportable model snapshot state

Prediction-column changes affect labels-layer styling. Metadata-only changes
affect persistence/export state but do not require recoloring.

Current broad notification points already exist:

- `_apply_ineligible_state(...)` clears predictions and writes classifier config
- `_on_worker_returned(...)` writes predictions and classifier config
- `_on_worker_errored(...)` writes failure config, but does not write new
  predictions

Phase 2A should make this distinction explicit instead of treating every
classifier table mutation as a styling event.

### Proposed Behavior

Use separate widget callbacks for status, persistence-relevant table changes,
and prediction-column changes:

```python
def _on_classifier_state_changed(self) -> None:
    self._update_classifier_feedback()
    self._update_classifier_controls()

def _on_classifier_table_state_changed(self) -> None:
    self._mark_persistence_dirty()
    self._update_persistence_controls()

def _on_classifier_prediction_state_changed(self) -> None:
    self._refresh_layer_styling()
```

Then the classifier controller should call:

- status callback after status-only changes
- table-state callback after any classifier metadata or prediction-table write
- prediction-state callback only after `pred_class` or `pred_confidence` were
  written or cleared

This preserves prediction recoloring because prediction writes still emit a
styling callback. It removes redundant styling refreshes from
stale/scheduled/training status updates and from metadata-only failure updates.

### Expected Annotation Flow

Current auto-training path:

1. User applies or clears one `user_class`.
2. The annotation path updates user-class styling once.
3. The classifier is marked stale.
4. The stale/scheduled status updates refresh only classifier feedback/controls.
5. If the debounce later retrains and writes predictions, the table-state
   callback marks persistence dirty and the prediction-state callback refreshes
   prediction styling once.

After Phase 2B, with auto training disabled:

1. User applies or clears one `user_class`.
2. The annotation path updates user-class styling once.
3. The classifier is marked stale.
4. No retrain is scheduled, so no prediction-table refresh happens until the
   user clicks `Train Classifier`.

### Files

- `src/napari_harpy/widgets/object_classification/widget.py`
- `tests/test_widget.py`
- `tests/test_classifier.py` if controller callback behavior needs a focused
  test

### Non-Goals

- Do not suppress classifier status messages.
- Do not skip prediction recoloring after predictions are written.
- Do not change model training behavior.
- Do not add incremental single-row layer updates in this phase.
- Do not change the `ViewerStylingController` refresh implementation yet.

### Tests

Add or update tests for:

- `mark_dirty(...)` updates classifier feedback/controls without refreshing
  labels-layer styling
- scheduled debounce status updates do not refresh labels-layer styling
- successful classifier prediction writes still refresh labels-layer styling
- ineligible classifier runs that clear predictions still refresh labels-layer
  styling
- metadata-only classifier failure updates mark persistence dirty without
  refreshing labels-layer styling
- prediction color modes still update after `pred_class` or `pred_confidence`
  changes

### Acceptance Criteria

- Annotation no longer triggers labels-layer refreshes from classifier
  status-only events.
- Prediction coloring still updates when classifier predictions are written or
  cleared.
- Classifier feedback and retrain/export controls stay accurate.
- The change is independently revertible from the auto-training checkbox and
  row-scoped table-edit work.

## Phase 2B: Manual/Automatic Classifier Training Toggle

### Goal

Add a checkbox that lets users disable automatic classifier retraining after
each annotation.

This builds on Phase 2A. Once status-only classifier callbacks no longer refresh
the labels layer, the toggle can focus on training policy: "I am annotating
quickly" versus "I want predictions updated after each click".

### Proposed UI

Add a checkbox near the classifier controls:

```text
[x] Auto train
```

Default: checked, preserving existing behavior.

When unchecked:

- annotation edits still mark persistence dirty
- annotation edits still mark classifier outputs dirty/stale
- annotation edits still refresh annotation coloring
- annotation edits do not call `schedule_retrain()`
- the existing `Train Classifier` button remains the manual path

### Files

- `src/napari_harpy/widgets/object_classification/widget.py`
- `tests/test_widget.py`

### Implementation Notes

- Store widget state as something like:

```python
self._auto_train_enabled = True
```

- Add a `QCheckBox` with object name:

```text
auto_train_checkbox
```

- In `_on_annotation_changed()`:

```python
self._classifier_controller.mark_dirty(reason="the annotations changed")
if self._auto_train_enabled:
    self._classifier_controller.schedule_retrain()
```

- Keep `mark_dirty()` outside the checkbox condition so classifier status and
  export availability remain honest.
- With Phase 2A in place, `mark_dirty()` should update classifier
  feedback/controls but must not refresh labels-layer styling through the
  classifier status callback.
- Update tooltip/status text to make the manual path clear when auto training
  is disabled.

### Tests

Add tests for:

- default state preserves current auto-retrain behavior
- unchecking the checkbox prevents `schedule_retrain()` from being called after
  annotation
- unchecking the checkbox still calls `mark_dirty()`
- unchecking the checkbox does not refresh labels-layer styling through
  classifier status callbacks
- clicking `Train Classifier` still invokes manual retraining when auto training
  is disabled

### Acceptance Criteria

- Users can annotate repeatedly without triggering classifier work on each edit.
- The classifier state clearly indicates stale predictions.
- Manual training remains available and unchanged.
- Phase 2B remains a small policy change because Phase 2A already separated
  classifier status from prediction styling.

## Phase 3: Row-Scoped `user_class` Table Edits

### Goal

Update only the selected row(s) when applying or removing a user class, instead
of normalizing and rewriting the whole `user_class` column on every edit.

### Current Problem

`AnnotationController._set_current_class()` currently does:

```python
self.ensure_annotation_column(USER_CLASS_COLUMN)
user_class_values = _to_user_class_values(state.table.obs[USER_CLASS_COLUMN])
user_class_values.loc[matching_rows] = int(class_id)
_set_user_class_annotation_state(state.table, user_class_values)
```

This is simple and robust, but expensive for large tables because it converts
the whole column and rebuilds categorical state on each edit.

### Proposed Behavior

Add a focused helper in `core/annotation.py`, for example:

```python
def set_user_class_for_rows(table: AnnData, rows: pd.Series, class_id: int) -> None:
    ...
```

The helper should:

- ensure `user_class` exists if missing
- recover through full normalization if the existing column is invalid
- add the requested category if absent
- assign only `rows`
- remove unused categories after clearing
- refresh `table.uns["user_class_colors"]` to match the resulting categories

### Files

- `src/napari_harpy/core/annotation.py`
- `src/napari_harpy/widgets/object_classification/annotation_controller.py`
- `tests/test_widget.py`
- `tests/test_class_palette.py` or a new focused annotation-core test if useful

### Design Constraints

- Keep `_set_user_class_annotation_state()` for full normalization on bind,
  reload, and recovery.
- The row-scoped helper must not duplicate the full palette logic in multiple
  places.
- Preserve exact current category behavior:
  - assigning class `3` to an all-unlabeled table yields categories `[0, 3]`
  - clearing the only labeled class yields categories `[0]`
  - colors remain `default_class_colors(categories)`
- If the existing column is not categorical or contains invalid values, fall
  back to full normalization once, then perform the row edit.

### Tests

Add tests for:

- assigning a new class updates the selected row only
- clearing the only labeled object removes the unused class category
- assigning another class replaces categories as expected
- existing non-categorical `user_class` state is recovered safely
- `user_class_colors` remains aligned with `user_class.cat.categories`

### Acceptance Criteria

- A single annotation no longer converts the entire column in the normal valid
  state.
- Persisted table state remains compatible with existing write/reload tests.
- Classifier training still reads the expected integer class values.

## Phase 4: Classifier Debounce Timer Lifecycle

### Goal

Fix the stale Qt timer cleanup race independently of annotation performance.

### Current Problem

The classifier controller owns a debounce `QTimer`. Faster annotation paths can
make tests expose a race where a scheduled timer fires after the widget has
already been deleted, and the callback tries to update deleted Qt labels.

### Proposed Behavior

Make timer ownership and shutdown explicit.

Options to evaluate:

1. Parent the debounce timer to the widget.
2. Add a `shutdown()` method to `ClassifierController` that stops pending work
   and clears UI callbacks.
3. Connect the widget's `destroyed` signal to the controller shutdown path.

The implementation can use both timer parenting and explicit shutdown if that is
the clearest lifecycle model.

### Files

- `src/napari_harpy/widgets/object_classification/controller.py`
- `src/napari_harpy/widgets/object_classification/widget.py`
- `tests/test_widget.py`

### Implementation Notes

- Keep controller usable in tests without a widget parent.
- Consider constructor shape:

```python
ClassifierController(..., timer_parent: QObject | None = None)
```

- `shutdown()` should:
  - stop the debounce timer
  - cancel/quit active workers if any
  - invalidate pending job ids
  - clear `on_state_changed` and `on_table_state_changed` callbacks
- Avoid relying only on Python object destruction; Qt signal timing is the risky
  part.

### Tests

Add or adjust tests so that:

- pending debounced retrain does not call widget callbacks after widget deletion
- calling `shutdown()` is idempotent
- existing classifier retraining tests still pass

### Acceptance Criteria

- No Qt event-loop exception from stale classifier timers.
- Debounced classifier behavior is unchanged while the widget is alive.
- Worker cancellation/reload behavior remains unchanged.

## Phase 5: Final Benchmark And Optional Incremental Layer Feature/Color Updates

### Goal

Measure the remaining annotation cost after Phases 1-4, including Phase 2A and
Phase 2B. Only pursue incremental layer updates if those measurements show they
are still needed.

This phase would update a single row in `layer.features` and a single entry in
an existing `DirectLabelColormap` after a user-class edit.

### Why It Is Deferred

This is where maintainability risk rises:

- it may require mutating napari colormap internals or clearing private caches
- it couples annotation changes to layer feature layout assumptions
- it is easy to accidentally diverge from full-refresh behavior

Sparse `user_class` coloring and disabling auto-training should already remove
the largest visible costs. Row-scoped table edits should remove another broad
table operation. We should re-measure in this phase before adding any
incremental layer-update code.

### Benchmark Scope

Add a short benchmark/dev note in this roadmap, or keep a local script in a
scratch area, that measures:

- table row count for the active segmentation
- time to apply one `user_class` edit
- time to refresh `user_class` layer coloring
- size of `layer.colormap.color_dict`
- whether classifier work is scheduled after annotation

Use the project environment:

```bash
source .venv/bin/activate
```

Measure against:

```text
/Users/arne.defauw/VIB/DATA/test_data/sdata_xenium_full_data_core.zarr
labels: cell_labels_global_ROI1
table: table_global_ROI1
```

### Acceptance Criteria Before Starting

- Phases 1-4, including Phase 2A and Phase 2B, have landed.
- Benchmarks still show annotation lag dominated by `layer.features` refresh or
  colormap assignment.
- We have a small, documented napari-compatible API surface for refreshing a
  single labels layer styling change.

## Suggested Landing Order

1. Phase 1: Sparse `user_class` colormap.
2. Phase 2A: Split classifier status from prediction-table refresh.
3. Phase 2B: Auto-training checkbox.
4. Phase 4: Classifier timer lifecycle.
5. Phase 3: Row-scoped `user_class` edits.
6. Phase 5: benchmark, then only add incremental layer updates if still needed.

The timer fix can land before Phase 2A, Phase 2B, or Phase 3 if tests expose
the stale callback race earlier.

## Verification Matrix

Run focused checks after each phase:

```bash
source .venv/bin/activate
ruff check src/napari_harpy tests
pytest -q tests/test_widget.py
```

Run broader checks before merging the full annotation-performance set:

```bash
source .venv/bin/activate
pytest -q tests/test_widget.py tests/test_classifier.py tests/test_persistence.py
```
