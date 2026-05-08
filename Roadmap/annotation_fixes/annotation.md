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

A prediction change is also a table-state change, but not every table-state
change is a prediction change. Keep both notifications because they drive
different downstream work:

| Callback | Meaning | Widget work |
| --- | --- | --- |
| `on_table_state_changed` | Classifier-owned table state changed, including metadata/config changes | Mark persistence dirty and update persistence/export controls |
| `on_prediction_state_changed` | Visual prediction columns changed or were cleared | Refresh labels-layer styling |

For example, a failed training run can update `table.uns[CLASSIFIER_CONFIG_KEY]`
without changing `pred_class` or `pred_confidence`. That should mark persistence
dirty, but it should not force a labels-layer refresh.

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
    # A prediction change is also a table-state change, but not every
    # table-state change affects labels-layer styling. This callback is for
    # persistence/export state.
    self._mark_persistence_dirty()
    self._update_persistence_controls()

def _on_classifier_prediction_state_changed(self) -> None:
    # Prediction changes are the classifier-owned table changes that affect
    # labels-layer coloring/features.
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

### Implementation Details

Add a third optional controller callback:

```python
ClassifierController(
    ...,
    on_state_changed=self._on_classifier_state_changed,
    on_table_state_changed=self._on_classifier_table_state_changed,
    on_prediction_state_changed=self._on_classifier_prediction_state_changed,
)
```

Add a small helper so prediction-output paths cannot forget one of the two
notifications:

```python
def _notify_prediction_table_state_changed(self) -> None:
    self._notify_table_state_changed()
    if self._on_prediction_state_changed is not None:
        self._on_prediction_state_changed()
```

Use this helper only for high-level classifier output changes:

- `_apply_ineligible_state(...)`, after predictions are cleared and classifier
  config is written
- `_on_worker_returned(...)`, after predictions are written, classifier config is
  written, and the model snapshot is stored

Keep using `_notify_table_state_changed()` without prediction notification for
metadata-only changes:

- `_on_worker_errored(...)`, after failure config is written

Do not emit prediction-state callbacks from passive setup or selection paths:

- `bind(...)` calls `_ensure_prediction_columns(table)` so existing tables have
  normalized prediction columns, but binding already has an explicit full layer
  refresh in the widget.
- `reset_after_reload(...)` also normalizes prediction columns, but reload
  already rebinds and refreshes the layer before resetting classifier status.

This keeps Phase 2A focused on retrain/apply results, not passive controller
normalization.

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
- worker-finished control refresh updates retrain/export controls without
  refreshing labels-layer styling
- successful classifier prediction writes still refresh labels-layer styling
- ineligible classifier runs that clear predictions still refresh labels-layer
  styling
- metadata-only classifier failure updates mark persistence dirty without
  refreshing labels-layer styling
- `bind(...)` and `reset_after_reload(...)` do not emit prediction-state
  callbacks just because they normalize missing prediction columns
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

**Status: Implemented.**

Implemented details:

- Added an `Auto train` checkbox before `Train Classifier`.
- Auto train is unchecked by default.
- Annotation edits always mark persistence dirty, mark classifier outputs stale,
  and refresh annotation styling.
- Annotation edits only call `schedule_retrain()` when auto train is checked.
- `Train Classifier` remains the manual training path when auto train is
  unchecked.
- The checkbox uses the shared widget checkbox stylesheet so it stays readable
  on the object-classification panel.
- The manual train button is enabled only when classifier preparation is
  trainable, preserving the previous behavior for one-class/no-label states.

### Goal

Add a checkbox that lets users disable automatic classifier retraining after
each annotation.

This builds on Phase 2A. Once status-only classifier callbacks no longer refresh
the labels layer, the toggle can focus on training policy: "I am annotating
quickly" versus "I want predictions updated after each click".

### User-Facing Behavior

There are two separate actions:

- Annotation: update `user_class` and immediately refresh user-class styling.
- Training: update `pred_class`/`pred_confidence` from the current annotations.

Phase 2B lets the user choose whether annotation automatically schedules the
training action.

When auto training is enabled, the current behavior remains:

1. User applies or clears one `user_class`.
2. The widget marks persistence dirty.
3. The classifier is marked stale.
4. User-class styling refreshes.
5. A debounced classifier retrain is scheduled.

When auto training is disabled:

1. User applies or clears one `user_class`.
2. The widget marks persistence dirty.
3. The classifier is marked stale.
4. User-class styling refreshes.
5. No classifier retrain is scheduled.
6. Existing predictions remain visible but stale until the user clicks
   `Train Classifier`.

### Proposed UI

Add a checkbox near the classifier controls:

```text
[ ] Auto train
```

Default: unchecked, favoring fast manual annotation.

Suggested placement:

- Add the checkbox to `retrain_action_row`, before `Train Classifier`.
- Keep `Train Classifier` as the manual path.
- Keep `Export Classifier` unchanged.

The checkbox is a preference for future annotation edits, so it can remain
enabled even when no table or feature matrix is selected.

When unchecked:

- annotation edits still mark persistence dirty
- annotation edits still mark classifier outputs dirty/stale
- annotation edits still refresh annotation coloring
- annotation edits do not call `schedule_retrain()`
- the existing `Train Classifier` button remains the manual path

### Files

- `src/napari_harpy/widgets/object_classification/widget.py`
- `tests/test_widget.py`

### State Model

Store the preference on the widget:

```python
self._auto_train_enabled = False
```

The checkbox is the source of truth for user interaction, and the widget state
should mirror it through a small slot such as:

```python
def _on_auto_train_toggled(self, checked: bool) -> None:
    self._auto_train_enabled = bool(checked)
    self._update_classifier_controls()
```

The setting is not persisted to zarr in this phase. A new widget starts with
auto training disabled.

### Implementation Notes

- Import and create a `QCheckBox`.
- Add a `QCheckBox` with object name:

```text
auto_train_checkbox
```

- Set it unchecked by default:

```python
self.auto_train_checkbox.setChecked(False)
```

- In `_on_annotation_changed()`:

```python
self._mark_persistence_dirty()
self._classifier_controller.mark_dirty(reason="the annotations changed")
self._refresh_layer_styling()
if self._auto_train_enabled:
    self._classifier_controller.schedule_retrain()
self._update_selection_status()
```

- Keep `mark_dirty()` outside the checkbox condition so classifier status and
  export availability remain honest.
- Keep `_refresh_layer_styling()` outside the checkbox condition. Annotation
  coloring is independent from classifier training.
- With Phase 2A in place, `mark_dirty()` should update classifier
  feedback/controls but must not refresh labels-layer styling through the
  classifier status callback.
- Update the checkbox tooltip when toggled:
  - checked: "Automatically train the classifier after each annotation."
  - unchecked: "Keep predictions stale while annotating; click Train Classifier
    to update predictions."
- The existing classifier stale status is sufficient. Do not add a second
  warning card just because auto training is disabled.

### Cancellation Policy

The checkbox controls whether future annotation edits call `schedule_retrain()`.

Phase 2B should not add new cancellation behavior:

- Unchecking does not cancel an active classifier worker.
- Unchecking does not need to cancel a debounce that was already scheduled while
  auto training was enabled.
- Clicking `Train Classifier` still runs the manual retrain path even when auto
  training is unchecked.

If pending-job cancellation becomes important, handle it later with the Phase 7
timer lifecycle work rather than mixing it into the training-policy toggle.

### Edge Cases

- If auto training is disabled and the user is viewing `pred_class` or
  `pred_confidence`, annotation can still refresh the layer using the selected
  color source, but predictions remain stale because no classifier run happens.
- If auto training is disabled and no feature matrix is selected, annotations
  still work and classifier status should remain honest about the missing
  feature matrix/manual training availability.
- Re-enabling auto training does not immediately train. It only affects
  subsequent annotation edits.

### Tests

Add tests for:

- the checkbox exists, is named `auto_train_checkbox`, and is unchecked by
  default
- default unchecked state prevents auto-retrain after annotation
- unchecking the checkbox prevents `schedule_retrain()` from being called after
  annotation
- unchecking the checkbox still calls `mark_dirty()`
- unchecking the checkbox still refreshes annotation styling through
  `_on_annotation_changed()`
- unchecking the checkbox still marks persistence dirty after annotation
- checking the checkbox makes subsequent annotations schedule retraining
  again
- clicking `Train Classifier` still invokes manual retraining when auto training
  is disabled
- toggling the checkbox by itself does not mark classifier or persistence state
  dirty

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

### Fast-Path Requirements

The normal valid-column path is the performance-sensitive path. It must avoid
the broad work that Phase 3 is meant to remove:

- Do not call `ensure_annotation_column()` before the row edit when
  `user_class` already exists as a valid categorical class column.
- Do not call `_to_user_class_values(...)` on the full `user_class` column in
  the happy path.
- Do not call `_set_user_class_annotation_state(...)` on the full table in the
  happy path.
- Do not normalize or rewrite the whole `user_class` column when assigning one
  selected row to a class that already exists in the categorical categories.
- Do not resync categories or `user_class_colors` when assigning an
  already-existing class and the edit does not make any category unused.
- Fall back to full normalization only when `user_class` is missing, not
  categorical, has invalid categories/values, or when category cleanup is
  actually required.

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
- Category cleanup may still scan categorical codes to detect whether a class
  became unused. That is acceptable; the primary win is avoiding full
  string/numeric conversion and categorical-column reconstruction for every
  annotation.

### Tests

Add tests for:

- assigning a new class updates the selected row only
- assigning an existing class does not rewrite the whole `user_class` column or
  resync `user_class_colors`
- clearing the only labeled object removes the unused class category
- assigning another class replaces categories as expected
- existing non-categorical `user_class` state is recovered safely
- `user_class_colors` remains aligned with `user_class.cat.categories`

### Acceptance Criteria

- A single annotation no longer converts the entire column in the normal valid
  state.
- Persisted table state remains compatible with existing write/reload tests.
- Classifier training still reads the expected integer class values.

## Phase 4: Faster `user_class` Viewer Styling Refresh

### Goal

Reduce the remaining `ViewerStylingController.refresh()` cost for the common
case where the user is annotating with `color_by == "user_class"` and auto
training is disabled.

Phase 1 made the resulting colormap sparse, but the refresh path still discovers
that sparse state by scanning and rebuilding full-region table state. Phase 4
keeps the public behavior as a full labels-layer refresh, but removes avoidable
full-table work inside that refresh.

### Measurement Snapshot

Measured against:

```text
/Users/arne.defauw/VIB/DATA/test_data/sdata_xenium_full_data_core.zarr
labels: cell_labels_global_ROI1
table: table_global_ROI1
rows: 406,611
user_class categories: [0, 1]
nonzero user_class rows: 11
```

Current approximate timings:

```text
ViewerStylingController.refresh()              ~1.28s
refresh_layer_colors()                         ~0.93s
refresh_layer_features()                       ~0.35s
_get_region_rows_by_instance()                 ~0.11s
_get_region_feature_rows()                     ~0.34s
_get_class_color_lookup(user_class)            ~0.24s
sparse color loop over all 406k rows           ~0.35s
sparse color loop over only labeled rows       ~0.0005s
DirectLabelColormap construction/assignment    ~0.0003s
```

### Current Problem

The sparse `user_class` colormap still pays several full-region costs:

- `refresh()` calls `refresh_layer_colors()` and `refresh_layer_features()`;
  each call rebuilds region feature rows independently.
- `_get_region_feature_rows()` normalizes `user_class`, `pred_class`, and
  `pred_confidence` for every row in the selected region.
- `_get_class_color_lookup(...)` normalizes the full table column to discover
  categories, even when the column is already a valid categorical column with a
  stored palette.
- `refresh_layer_colors()` loops over every instance id and skips unlabeled
  rows one by one, even though the resulting colormap contains only the
  annotated labels.

### Proposed Behavior

Optimize the valid `user_class` refresh path without introducing private napari
state mutation:

- In `refresh()`, compute region feature rows once and reuse them for both color
  and feature refresh work.
- For `color_by == "user_class"`, build the sparse color dict from only rows
  whose normalized `user_class != 0`.
- In `_get_class_color_lookup(...)`, use categorical categories and
  `table.uns["user_class_colors"]` directly when the selected column is already
  a valid categorical class column.
- Fall back to the current full-normalization lookup path for non-categorical or
  invalid table state.
- Keep full explicit mappings for `pred_class` and continuous
  `pred_confidence` unless a later measurement justifies optimizing them too.

### Implementation Details

Keep this phase as a full layer refresh. The implementation should reduce how
much table work is repeated inside that refresh, not introduce incremental
napari state mutation.

Suggested structure:

```python
def refresh(self) -> None:
    if self._labels_layer is None:
        return

    feature_rows = self._get_region_feature_rows()
    self.refresh_layer_colors(feature_rows=feature_rows)
    self.refresh_layer_features(feature_rows=feature_rows)
```

`refresh_layer_colors(...)` and `refresh_layer_features(...)` can keep their
public no-argument behavior by accepting an optional `feature_rows` argument and
computing it only when missing:

```python
def refresh_layer_colors(self, *, feature_rows: pd.DataFrame | None = None) -> None:
    if feature_rows is None:
        feature_rows = self._get_region_feature_rows()
```

That keeps existing callers working while letting `refresh()` avoid calling
`_get_region_feature_rows()` twice.

For user-class coloring:

```python
class_by_instance = feature_rows[USER_CLASS_COLUMN]
labeled_class_by_instance = class_by_instance[class_by_instance != UNLABELED_CLASS]
for instance_id, class_id in labeled_class_by_instance.items():
    color_dict[int(instance_id)] = class_color_lookup.get(int(class_id), unlabeled_color)
```

Do not use this sparse-row-only loop for `pred_class`. Prediction coloring
should keep explicit entries for every prediction row because class `0` can be a
real cleared/unknown prediction state and because prediction columns are written
in bulk by the classifier.

For user-class color lookup, add a valid-categorical fast path. A safe first
version can be private to `viewer_styling.py`, for example:

```python
def _read_valid_categorical_class_categories(values: pd.Series, *, unlabeled_class: int) -> list[int] | None:
    ...
```

The fast path should require:

- column dtype is `pd.CategoricalDtype`
- categories are integer class ids
- categories are sorted
- categories include `UNLABELED_CLASS`
- category codes contain no missing values
- stored palette length matches the category count

If any condition fails, keep the existing robust path:

```python
normalize_class_values(...)
read_series_class_categories(...)
stored_palette_to_lookup(...)
backfill_missing_class_colors(...)
```

The helper added in Phase 3 already keeps `user_class` valid in the common
annotation path, so the fast path should be hit after normal annotation edits.

### Expected Impact

This phase should mainly remove:

- the second `_get_region_feature_rows()` call inside `refresh()`
- the full-table user-class normalization inside `_get_class_color_lookup(...)`
- the Python loop over all 406k rows for sparse user-class coloring

It will not remove:

- the remaining full-region feature-table build
- assignment to `layer.features`
- explicit full-row prediction coloring for `pred_class` / `pred_confidence`

### Non-Goals

- Do not mutate an existing napari colormap in place.
- Do not update only one `layer.features` row.
- Do not skip feature refreshes yet; this phase only removes redundant
  rebuilding/scanning inside the current full-refresh model.
- Do not change classifier prediction refresh behavior.

### Files

- `src/napari_harpy/widgets/object_classification/viewer_styling.py`
- `tests/test_widget.py`
- Add a focused viewer-styling test file if that keeps tests clearer.

### Tests

Add or update tests for:

- sparse `user_class` color dict only contains nonzero user-class labels plus
  base entries
- existing `user_class` categorical categories and `user_class_colors` are used
  without full-column normalization in the valid path
- invalid/non-categorical state still falls back to the robust full
  normalization behavior
- `refresh()` reuses one region feature-row snapshot for color and feature
  refresh
- `refresh_layer_colors()` and `refresh_layer_features()` still work when called
  directly without a precomputed `feature_rows`
- `pred_class` and `pred_confidence` behavior remains unchanged

### Acceptance Criteria

- User-class color refresh no longer loops over every label id just to skip
  unlabeled cells.
- User-class color lookup does not normalize the full table column when the
  categorical column and palette are valid.
- A full `refresh()` does not rebuild region feature rows twice.
- Layer colors/features remain equivalent to the previous full-refresh
  behavior.

## Phase 5: Avoid Repeated Classifier Preparation Summaries

### Goal

Avoid recomputing classifier preparation summaries multiple times during one
annotation interaction, especially when auto training is disabled.

The preparation summary is useful for honest feedback and train-button
availability, but it scans enough table/feature state that repeated calls add
visible latency on large tables.

### Measurement Snapshot

Measured against:

```text
/Users/arne.defauw/VIB/DATA/test_data/sdata_xenium_full_data_core.zarr
labels: cell_labels_global_ROI1
table: table_global_ROI1
feature matrix: features_testing
rows: 406,611
```

Current approximate timing:

```text
ClassifierController.describe_current_preparation()    ~0.120s
```

### Current Problem

With auto training disabled, annotation still marks classifier outputs stale.
That status/control update path can recompute preparation summaries more than
once for the same underlying table state:

1. `_on_annotation_changed()` calls `mark_dirty(...)`.
2. `mark_dirty(...)` updates classifier status.
3. The classifier status callback updates classifier feedback/controls.
4. `_update_classifier_controls()` calls
   `describe_current_preparation()` to build the preparation card and decide
   whether `Train Classifier` is enabled.
5. `_on_annotation_changed()` then calls `_update_selection_status()`, which
   calls `_update_classifier_controls()` again and can recompute the same
   summary.

On the measured table, each summary is only about 120ms, but repeated summaries
stack up and become noticeable after the larger table-edit and styling costs
are reduced.

### Proposed Behavior

Keep classifier preparation feedback accurate, but compute it at most once for
one UI refresh cycle:

- Let `_update_classifier_controls(...)` accept an optional precomputed
  `ClassifierPreparationSummary`.
- When `_on_annotation_changed()` already knows it will call both
  `mark_dirty(...)` and `_update_selection_status()`, avoid doing the same
  preparation work twice.
- Consider having `_on_classifier_state_changed()` update feedback/status text
  only, and let the explicit end-of-annotation UI refresh update preparation
  cards and controls once.
- Alternatively, cache the latest preparation summary with a simple invalidation
  key based on the selected table object, feature key, scopes, and a table-state
  revision owned by the widget/controller.
- Keep the cache local and explicit. Do not infer correctness from object
  identity alone if the underlying table can mutate in place.

### Non-Goals

- Do not make classifier feedback stale or hide trainability warnings.
- Do not skip preparation summaries after selection, feature-matrix, training
  scope, prediction scope, reload, or classifier-result changes.
- Do not change classifier training eligibility rules.
- Do not add benchmark code in this phase.

### Files

- `src/napari_harpy/widgets/object_classification/widget.py`
- `src/napari_harpy/widgets/object_classification/controller.py` if a small
  controller-side cache or invalidation hook is clearer
- `tests/test_widget.py`
- `tests/test_classifier.py` only if controller-side behavior changes

### Tests

Add or update tests for:

- one annotation with auto training disabled does not call
  `describe_current_preparation()` redundantly
- classifier preparation feedback and `Train Classifier` enablement still update
  after annotation
- selection/table/feature/scope changes still refresh the preparation summary
- manual `Train Classifier` eligibility remains unchanged

### Acceptance Criteria

- The auto-training-disabled annotation path computes classifier preparation at
  most once for the resulting UI refresh.
- Classifier preparation cards and train/export controls remain accurate.
- The change stays independent from row-scoped table edits and viewer-styling
  refresh optimizations.

## Phase 6: Row-Scoped `user_class` Viewer Refresh

### Goal

Avoid calling `_get_region_feature_rows()` during the common annotation path when
`color_by == "user_class"`, while still keeping both labels-layer colors and
hover/properties features up to date for the annotated object.

Phase 4 still performs a full public styling refresh and therefore still builds
one selected-region feature table per annotation. Phase 6 adds a narrower
annotation-specific path that updates only the affected user-class color entry
and only the affected `layer.features` row.

### Current Problem

Even after Phase 4, this call remains in the full refresh path:

```python
feature_rows = self._get_region_feature_rows()
```

On the measured Xenium table, `_get_region_feature_rows()` costs roughly
`~0.34s` because it filters the selected labels region, normalizes
`user_class`, `pred_class`, and `pred_confidence`, and builds a fresh dataframe
for all selected-region label ids.

For a simple user annotation with auto training disabled and
`color_by == "user_class"`, the full feature table rebuild is unnecessary. The
annotation operation already knows:

```text
selected instance id -> new user_class
```

It does not need normalized prediction columns, prediction confidence, or a full
selected-region feature dataframe.

However, hover/properties data must remain current. Stale `layer.features` after
annotation is not acceptable because users should be able to inspect the newly
annotated cell immediately.

### Proposed Behavior

Add a row-scoped user-class annotation refresh path that is used only for
annotation edits when the viewer is currently colored by `user_class`.

The fast path eligibility should stay intentionally narrow:

```text
event source: direct user annotation
color mode: user_class
viewer state: valid row-scoped labels features and sparse DirectLabelColormap
table state: valid user_class categorical column and palette
```

Auto-training state does not decide whether the immediate annotation refresh can
use this path. If auto train is enabled, the immediate `user_class` annotation can
still be row-scoped, and the later classifier prediction write must still use the
full refresh path.

The annotation callback should carry a small payload describing what changed,
instead of being a no-argument signal:

Suggested shape:

```python
@dataclass(frozen=True)
class UserClassAnnotationChange:
    instance_id: int
    class_id: int
```

`AnnotationController._set_current_class(...)` already has both values:

- `state.instance_id`
- `class_id`

so it can call the annotation-changed callback with that payload after
`set_user_class_for_rows(...)` succeeds.

`ObjectClassificationWidget._on_annotation_changed(...)` can then choose:

```python
if (
    change is not None
    and self._viewer_styling_controller.color_by == COLOR_BY_USER_CLASS
    and self._viewer_styling_controller.refresh_user_class_annotation(change)
):
    ...
else:
    self._refresh_layer_styling()
```

`ViewerStylingController.refresh_user_class_annotation(...)` should update two
public napari layer properties without rebuilding all region features:

1. Labels colors:
   - copy the current sparse `DirectLabelColormap.color_dict`
   - if `class_id == UNLABELED_CLASS`, remove the explicit entry for
     `instance_id`
   - otherwise resolve the new class color from the valid `user_class`
     categorical palette and set `color_dict[instance_id]`
   - assign a new `DirectLabelColormap`

2. Labels features:
   - copy `layer.features`
   - locate the row where the feature `"index"` column equals `instance_id`
   - update only `USER_CLASS_COLUMN` for that row
   - assign the updated dataframe back to `layer.features`

This still avoids private napari internals. The controller should replace public
properties with updated copies rather than mutating hidden caches.

### Color Lookup

The row-scoped path should not call `normalize_class_values(...)` or
`_get_region_feature_rows()`.

It should use the strict Phase 4 valid-categorical palette fast path to resolve
the color for the one changed class id. If the `user_class` column or palette is
invalid, return `False` so the widget can fall back to the full refresh path.

For clearing annotations:

```python
class_id == UNLABELED_CLASS
```

remove the explicit color entry for the instance id. The sparse colormap default
then colors the object as unlabeled.

For assigning annotations:

```python
class_id > UNLABELED_CLASS
```

write only the explicit entry for the changed instance id.

### Where It Is Used

In the annotation changed path:

```python
if self._viewer_styling.color_by == COLOR_BY_USER_CLASS:
    handled = self._viewer_styling.refresh_user_class_annotation(change)
    if not handled:
        self._viewer_styling.refresh()
else:
    self._viewer_styling.refresh()
```

The exact call site should preserve current behavior for:

- `color_by == "pred_class"`
- `color_by == "pred_confidence"`
- auto-train classifier prediction writes
- classifier prediction updates
- selection changes
- table/feature/scope changes
- initial binding and full reloads

Those paths should continue to use the full refresh path.

In other words:

- direct annotation + `color_by == "user_class"`: try row-scoped refresh, then
  fall back to full refresh if unsafe
- direct annotation + `color_by != "user_class"`: full refresh
- classifier prediction/state output changed: full refresh
- selection/binding/table structure changed: full refresh

### Fallback Conditions

Return `False` and let the widget perform a full refresh when:

- no labels layer is bound
- current colormap is not a `DirectLabelColormap`
- `layer.features` is missing, empty, or lacks `"index"` / `USER_CLASS_COLUMN`
- the annotated `instance_id` is not present exactly once in `layer.features`
- the `user_class` column or palette is invalid
- color lookup for the changed positive class id cannot be resolved

The fallback preserves correctness while the happy path stays fast for normal
Phase 3-maintained `user_class` state.

### Maintainability Guardrails

Keep the branching contained and easy to debug:

- Put the fast-path decision in one small widget helper, for example
  `_refresh_after_user_class_annotation(change) -> bool`.
- Put the row-scoped layer mutation in one `ViewerStylingController` method,
  for example `refresh_user_class_annotation(change) -> bool`.
- The viewer-styling method should return a simple boolean: `True` means the
  row-scoped update was fully applied; `False` means the caller must perform a
  normal full refresh.
- Avoid mixing classifier state, auto-train policy, persistence, and styling
  fallback decisions in the same conditional block.
- Prefer early returns for fallback checks over deeply nested branches.
- Add a short debug-level log only when falling back would be useful during
  development; do not surface fallback details in the user UI.

### Non-Goals

- Do not mutate napari private colormap internals.
- Do not rebuild the full selected-region feature table in the happy path.
- Do not scan the selected region to rediscover all annotated cells in the happy
  path.
- Do not change prediction coloring or prediction feature refresh behavior.
- Do not skip full refreshes for selection, binding, color-mode, classifier, or
  table-structure changes.
- Do not change classifier dirty/retrain behavior.

### Files

- `src/napari_harpy/widgets/object_classification/viewer_styling.py`
- `src/napari_harpy/widgets/object_classification/annotation_controller.py`
- `src/napari_harpy/widgets/object_classification/widget.py`
- `tests/test_widget.py`
- Add a focused viewer-styling test file if that keeps tests clearer.

### Tests

Add or update tests for:

- annotation with `color_by == "user_class"` uses the row-scoped refresh path
- annotation with `color_by == "pred_class"` still uses the full refresh path
- annotation with `color_by == "pred_confidence"` still uses the full refresh
  path
- auto train enabled still uses full refresh when classifier predictions are
  written after training
- row-scoped refresh does not call `_get_region_feature_rows()`
- row-scoped refresh updates the sparse colormap for assigning a class
- row-scoped refresh removes the explicit sparse colormap entry for clearing a
  class
- row-scoped refresh updates only the matching `layer.features` row's
  `USER_CLASS_COLUMN`
- hover/properties feature state is current immediately after annotation
- invalid/missing `user_class`, missing feature rows, or invalid colormap state
  falls back to full refresh
- classifier prediction changes still use full refresh
- selection/table/feature/color-mode changes still use full refresh

### Acceptance Criteria

- The common `user_class` annotation path no longer calls
  `_get_region_feature_rows()`.
- `pred_class`, `pred_confidence`, classifier prediction writes, and
  selection/binding changes still call the full refresh path.
- Visual labels-layer coloring remains correct after adding/removing an
  annotation.
- `layer.features` has the new `user_class` value for the annotated instance
  immediately after annotation.
- Prediction coloring, classifier updates, and full viewer refreshes remain
  equivalent to the previous behavior.

## Phase 7: Classifier Debounce Timer Lifecycle

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

## Phase 8: Final Benchmark And Optional Deeper Layer Updates

### Goal

Measure the remaining annotation cost after Phases 1-7, including Phase 2A and
Phase 2B. Only pursue incremental layer updates if those measurements show they
are still needed.

Phase 6 should already update a single `layer.features` row and labels colormap
entry through public layer-property assignment. This final phase is for
benchmarking the remaining cost and only considering deeper incremental napari
state work if it is still needed.

### Why It Is Deferred

This is where maintainability risk rises:

- it may require mutating napari colormap internals or clearing private caches
- it is easy to accidentally diverge from full-refresh behavior

Sparse `user_class` coloring, disabling auto-training, row-scoped table edits,
faster user-class styling refreshes, row-scoped annotation refresh, and avoiding
repeated classifier preparation summaries should remove the largest visible
costs. We should re-measure in this phase before adding any private
layer-update code.

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

- Phases 1-7, including Phase 2A and Phase 2B, have landed.
- Benchmarks still show annotation lag dominated by `layer.features` refresh or
  colormap assignment.
- We have a small, documented napari-compatible API surface for refreshing a
  single labels layer styling change.

## Suggested Landing Order

1. Phase 1: Sparse `user_class` colormap.
2. Phase 2A: Split classifier status from prediction-table refresh.
3. Phase 2B: Auto-training checkbox.
4. Phase 3: Row-scoped `user_class` edits.
5. Phase 4: Faster `user_class` viewer styling refresh.
6. Phase 6: Row-scoped `user_class` viewer refresh.
7. Phase 5: Avoid repeated classifier preparation summaries.
8. Phase 7: Classifier timer lifecycle.
9. Phase 8: benchmark, then only add deeper private layer updates if still needed.

Phase 5 is independent and can be deferred until after Phase 6 because the
remaining visible annotation lag is now dominated by viewer refresh work.

The timer fix can land before Phase 2A, Phase 2B, Phase 3, Phase 4, Phase 5, or
Phase 6 if tests expose the stale callback race earlier.

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
