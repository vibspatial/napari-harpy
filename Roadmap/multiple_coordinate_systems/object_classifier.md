# Object Classifier Scopes

## Purpose

This document breaks roadmap item `4. Add Classifier Training and Prediction
Scopes` into smaller implementation slices.

It is a companion to:

- `Roadmap/multiple_coordinate_systems/cross_sample_tables.md`

The goal is to avoid landing one large change that rewrites classifier row
selection, widget controls, status text, metadata persistence, and reload
semantics all at once.

Unlike the feature-extraction roadmap, this classifier roadmap does not require
the train/predict action to remain usable at every intermediate step. During
the refactor, it is acceptable to temporarily disable `Train Classifier` if
that lets the controller and widget contracts move more quickly and more
cleanly.

## Completion Summary

This roadmap is complete. The object-classifier pipeline now separates
training scope from prediction scope throughout the controller, widget,
metadata, reload logic, and tests.

What landed:

- `src/napari_harpy/_classifier.py` resolves explicit training and prediction
  scopes with `ResolvedClassifierScope` / `ResolvedClassifierScopes`;
- training defaults to all eligible labeled regions in the selected table, while
  prediction defaults to the selected segmentation only;
- classifier row eligibility now excludes rows with non-finite or missing
  feature values before training or prediction arrays are prepared;
- `ClassifierPreparationSummary` is the shared structured preparation object
  used by jobs, results, metadata counts, controller status, and widget status;
- classifier metadata records explicit training and prediction scope fields,
  including resolved regions and row counts, instead of relying on one legacy
  active-region field;
- reload-state checks now compare feature key, table name, prediction-scope
  mode, and prediction-region coverage;
- `src/napari_harpy/widgets/_object_classification_widget.py` exposes training
  and prediction scope controls and marks classifier outputs stale when either
  scope changes;
- `src/napari_harpy/widgets/_object_classification_status_card.py` owns the
  selection, classifier-preparation, and classifier-feedback status-card specs;
- multi-region controller, widget, and persistence coverage verifies the final
  scope behavior.

## Guiding Rules

Implement the scope work in slices that keep the refactor understandable and
testable, without forcing classifier training to remain available at every
intermediate step.

In practice, that means:

- keep annotation interactions sample-local and viewer-local;
- separate training scope from prediction scope in code before adding complex UI
  around them;
- allow the widget to temporarily disable `Train Classifier` while classifier
  scope resolution, metadata, and reload semantics are in flux;
- prefer an explicitly disabled classifier action over preserving a partially
  correct or ambiguous training path;
- do not keep overloading `active_*` names once training rows and prediction
  rows can diverge;
- make hidden cross-sample writes explicit in the UI before enabling them;
- treat metadata / reload behavior as part of the feature, not as cleanup after
  the controller change;
- add multi-region test coverage early enough that later slices can build on it
  safely.

## Temporary Refactor Mode

During slices 1 through 5, it is acceptable for the widget to disable
`Train Classifier` entirely while the classifier pipeline is being reworked.

Recommended rule:

- keep annotation, table selection, feature-matrix selection, and viewer
  styling working;
- disable only the train/predict action, with explicit UI copy that classifier
  scope refactoring is in progress;
- re-enable the button once both scope controls and persisted metadata semantics
  are coherent enough to expose one unambiguous user-facing training action
  again.

## Recommended Scope Vocabulary

Use one stable pair of user-facing scope values throughout controller, widget,
tests, and `table.uns[CLASSIFIER_CONFIG_KEY]`:

- `selected_segmentation_only`
- `all`

Interpretation:

- `selected_segmentation_only` resolves to the one `region_key` value for the
  segmentation currently selected in `ObjectClassificationWidget`;
- `all` resolves at execution time to the eligible set of table regions for the
  selected table and feature matrix.

`all` should remain the compact persisted mode value, while the fully expanded
  resolved region list is stored separately.

## Slices

### 1. Extract Explicit Scope Resolution in the Classifier Controller

Status: [x] Implemented

Goal:

- move the controller from one implicit `active_mask` to explicit training and
  prediction scope resolution while preserving today’s behavior.
- make this primarily a controller-internal refactor of scope resolution,
  eligibility derivation, and job payload structure rather than a widget-level
  behavior change.

Scope:

- extend `ClassifierController.bind(...)` to accept:
  - `training_scope: ClassifierScopeMode = "selected_segmentation_only"`
  - `prediction_scope: ClassifierScopeMode = "selected_segmentation_only"`
- add controller-level scope types or literals for `selected_segmentation_only` and
  `all`;
- introduce explicit resolved-scope dataclasses, for example:

```python
ClassifierScopeMode = Literal["selected_segmentation_only", "all"]


@dataclass(frozen=True)
class ResolvedClassifierScope:
    mode: ClassifierScopeMode
    regions: tuple[str, ...]
    table_row_positions: np.ndarray
    n_rows_in_regions: int

    @property
    def n_eligible_rows(self) -> int:
        return int(self.table_row_positions.size)

    @property
    def n_excluded_feature_invalid_rows(self) -> int:
        return self.n_rows_in_regions - int(self.table_row_positions.size)


@dataclass(frozen=True)
class ResolvedClassifierScopes:
    training: ResolvedClassifierScope
    prediction: ResolvedClassifierScope
```

- use the following invariant:
  - `regions` stores the resolved semantic scope requested by the user;
  - `n_rows_in_regions` stores the total number of table rows whose
    `region_key` is in `regions`, before feature-validity filtering;
  - `table_row_positions` stores the in-scope table rows that are feature-valid
    for the currently selected feature matrix, meaning finite and non-missing,
    in original table order.
  - `n_eligible_rows` and `n_excluded_feature_invalid_rows` are derived from
    `n_rows_in_regions` and `table_row_positions` rather than stored
    separately.
- introduce a resolved-scope data model that can hold:
  - requested training scope mode;
  - resolved training regions;
  - training raw in-scope row count;
  - training feature-valid table-row positions;
  - requested prediction scope mode;
  - resolved prediction regions;
  - prediction raw in-scope row count;
  - prediction feature-valid table-row positions;
- treat `_prepare_classifier_job(...)` as the main refactor target and factor
  its current responsibilities around helpers such as:
  - `_get_finite_feature_row_mask(...)`
  - `_resolve_classifier_scopes(...)`
  - a small local per-scope resolver inside `_resolve_classifier_scopes(...)`
    if that keeps training and prediction resolution readable without
    reintroducing controller-wide helper indirection;
- keep scope membership and feature-valid row filtering in the same resolution
  path so each resolved scope has one authoritative `table_row_positions`
  result rather than several competing row-position concepts;
- training-specific labeled-row filtering may still happen later in classifier
  preparation, but it should build on the already resolved
  `table_row_positions` set rather than recomputing scope validity elsewhere;
- rename internal `active_positions` / `active_row_count` concepts toward
  prediction-specific names so later slices do not keep using `active` as a
  proxy for both concerns;
- replace `TrainingEligibility` with a broader preparation/result summary type
  such as `ClassifierPreparationSummary`, and carry that summary directly on
  `ClassifierJob` and `ClassifierJobResult`;
- let `_prepare_classifier_job(...)` return `ClassifierJob | None` once the
  summary is attached to the job, rather than returning a parallel eligibility
  object;
- apply the prediction-specific rename through:
  - `ClassifierJob`
  - `ClassifierJobResult`
  - `ClassifierPreparationSummary`
  - `_apply_ineligible_state(...)`
  - `_on_worker_returned(...)`
- keep runtime behavior equivalent to today by defaulting both scopes to
  `selected_segmentation_only`;
- make finite, non-missing feature-row filtering part of scope resolution in
  this slice, so `table_row_positions` already excludes rows that are unusable
  for the currently selected feature matrix rather than relying only on
  matrix-shape checks and downstream scikit-learn failures;
- it is acceptable for the feature-valid mask to be typed explicitly in code,
  for example with `BoolArray = NDArray[np.bool_]`, to make it clear that scope
  filtering is driven by a boolean NumPy mask;
- do not expand this slice to solve reload metadata semantics yet;
  `_update_status_from_reloaded_table(...)` may stay mostly single-region until
  slice 5, when stored classifier metadata becomes richer;
- widget work in this slice should stay minimal:
  - keep annotation and selection flows working;
  - if helpful, temporarily disable `Train Classifier` rather than preserving
    the old single-region execution path during the refactor.

Files:

- `src/napari_harpy/_classifier.py`
- optionally `src/napari_harpy/widgets/_object_classification_widget.py` if
  `Train Classifier` is temporarily disabled in this slice
- `tests/test_classifier.py`
- `tests/test_widget.py`

Expected outcome:

- the controller has a clean seam for later cross-sample behavior;
- no user-visible classifier-scope behavior changes yet;
- no widget-level scope controls are required yet, and this slice may either
  keep the existing train path working through safe defaults or temporarily
  disable `Train Classifier` if that keeps the refactor cleaner;
- `ResolvedClassifierScope.table_row_positions` becomes the one authoritative
  feature-valid row-position set per resolved scope;
- the controller already exposes `training_scope` and `prediction_scope`
  arguments with safe defaults, so later UI slices can wire into that seam
  without another signature refactor;
- existing single-region controller and widget tests still pass after the
  refactor, while dedicated multi-region coverage can land in slice 2.

### 2. Add Multi-Region Test Builders for Classifier Work

Status: [x] Implemented

Goal:

- stop depending on a single-region fixture for the new controller behavior.

Scope:

- add a small reusable way to construct a classifier table that spans at least
  two labels regions;
- keep the data builder narrow and deterministic even if it is promoted into
  shared test infrastructure once reuse is obvious;
- make it easy to create scenarios where:
  - the selected region has no local labels but another region does;
  - duplicate `instance_key` values appear across regions but not within a
    region;
  - one region has valid features while another has invalid feature rows.

Files:

- `src/napari_harpy/datasets.py`
- `tests/test_classifier.py`
- `tests/conftest.py`
- optionally `tests/test_widget.py` once widget-level multi-region assertions
  need the same fixture surface

Expected outcome:

- later slices can prove true multi-region behavior instead of reusing
  single-region assumptions;
- the dataset helper may be shared once reuse justifies it; in the current
  implementation that promotion is acceptable because both package-level smoke
  tests and classifier tests already consume the same `blobs_multi_region()`
  dataset shape.

### 3. Switch Default Training to Table-Wide Rows in the Controller

Status: [x] Implemented

Goal:

- implement the roadmap’s most important behavior change first:
  training becomes table-level by default while prediction remains region-local.

Scope:

- keep the existing controller selection path for `training_scope`, but change
  the controller defaults so training and prediction no longer share one
  default constant;
- introduce explicit default scope constants such as:
  - `DEFAULT_TRAINING_SCOPE: ClassifierScopeMode = "all"`
  - `DEFAULT_PREDICTION_SCOPE: ClassifierScopeMode = "selected_segmentation_only"`
- use those separate defaults in:
  - `ClassifierController.__init__(...)`
  - `ClassifierController.bind(...)`
  - any controller-level fallback state that still assumes one shared default;
- resolve training rows from all eligible labeled rows in the selected table
  when training scope is `all`;
- keep prediction scope fixed at `selected_segmentation_only` in this slice;
- keep the resolved-scope logic unchanged in spirit:
  - `training_scope="all"` resolves all feature-valid rows across the table’s
    annotated regions;
  - `prediction_scope="selected_segmentation_only"` continues to resolve only
    the currently selected segmentation region;
- update training eligibility and status reporting so it distinguishes:
  - number of resolved training rows;
  - number of labeled training rows;
  - number of resolved prediction rows;
  - number of training regions involved;
- prefer extending `ClassifierPreparationSummary` rather than adding another
  summary object just for slice 3;
- add summary fields for the new controller-facing counts, for example:
  - `resolved_training_row_count`
  - `resolved_prediction_row_count`
  - `training_region_count`
  - keep existing `labeled_count`, `class_labels`, and `n_features`;
- do not duplicate full `training_scope` / `prediction_scope` objects inside
  `ClassifierPreparationSummary`;
  `ClassifierJob` and `ClassifierJobResult` already carry the authoritative
  scope objects, so the summary should store only the derived reporting counts;
- derive those new counts directly from the resolved scopes:
  - `resolved_training_row_count` from `training_scope.n_eligible_rows`
  - `resolved_prediction_row_count` from `prediction_scope.n_eligible_rows`
  - `training_region_count` from `len(training_scope.regions)`;
- use those fields to make preflight status text more precise while keeping
  success text prediction-focused;
- recommended status behavior in this slice:
  - training start / preflight should mention labeled rows, resolved training
    rows, and participating regions, for example:

```text
Classifier: training RandomForest on 4 labeled objects across 2 classes from 26 eligible rows in 2 regions.
```

  - successful completion should still focus on prediction scope writes, for
    example:

```text
Classifier: model is up to date. Updated predictions for 13 objects.
```

- keep the widget button model unchanged for now: one train action still kicks
  off one retrain/predict cycle.

Files:

- `src/napari_harpy/_classifier.py`
- `tests/test_classifier.py`

Expected outcome:

- labeling objects in several regions can contribute to one shared training set
  immediately;
- prediction writes are still restricted to the selected region, so hidden-row
  writes do not expand yet;
- controller defaults now encode the intended asymmetry directly:
  training is table-wide by default, prediction stays selected-region-only by
  default;
- controller summaries and status text distinguish training-scope size from
  prediction-scope size, which later UI slices can surface directly;
- the widget may still keep `Train Classifier` disabled until the scope UI and
  metadata contract catch up;
- the biggest behavior change lands before UI control proliferation.

### 4. Add an Explicit Training-Scope Control to the Widget

Status: [x] Implemented

Goal:

- expose the new training default while still allowing region-local training as
  an explicit override.

Scope:

- add a `Training scope` control to
  `src/napari_harpy/widgets/_object_classification_widget.py`;
- default it to `all`;
- support the two modes only:
  - `Selected segmentation only`
  - `All eligible labeled regions in table`
- propagate the chosen mode into controller binding;
- mark the classifier dirty when the training scope changes;
- keep the existing `classifier_feedback` / tooltip model for now rather than
  introducing a dedicated classifier preflight status card in this slice;
- if small wording tweaks are needed, keep them limited to the existing
  feedback card and button tooltips;
- defer richer classifier preflight messaging and status-card composition until
  a later UX-focused slice, after training/prediction scope behavior and
  metadata semantics have settled;
- keep prediction behavior unchanged in this slice;
- keep annotation and styling flows unchanged.

Files:

- `src/napari_harpy/widgets/_object_classification_widget.py`
- `src/napari_harpy/_classifier.py`
- `tests/test_widget.py`

Expected outcome:

- the user can see that training is table-level by default;
- the user can opt back into selected-segmentation-only training when desired;
- depending on implementation order, the train button may remain temporarily
  disabled until prediction-scope and metadata behavior are also ready;
- the widget wiring stays focused on functional scope selection rather than
  introducing new presentation structures too early.

### 5. Add Prediction Scope Support in the Controller and Metadata

Status: [x] Implemented

Goal:

- enable the second axis of scope change without hiding the write target.

Scope:

- add a controller selection path for `prediction_scope`;
- keep the default as `selected_segmentation_only`;
- support `all` as an explicit mode that resolves prediction rows across all
  eligible table regions;
- update worker result application so prediction writes target the resolved
  prediction table-row positions, not the old single-region `active_positions`;
- make prediction writeback semantics explicit for partially eligible scopes:
  - first clear predictions for all rows covered by the requested
    `prediction_regions`
  - then write fresh predictions only for
    `prediction_scope.table_row_positions`
  - as a result:
    - eligible in-scope rows receive fresh predictions
    - in-scope but feature-invalid rows become unlabeled / `NaN`
    - out-of-scope rows remain untouched
- update ineligible-state handling so it reasons about prediction-target rows
  explicitly;
- make ineligible runs persist the full attempted scope metadata too:
  `_prepare_classifier_job(...)` already returns a `ClassifierJob` even when
  `summary.eligible` is `False`, so `_apply_ineligible_state(...)` should use
  that resolved training/prediction scope information when writing
  `classifier_config`;
- preserve the attempted classifier snapshot for worker-error metadata too:
  the controller already stores `self._active_worker`, but that only captures
  worker lifecycle, not the original `ClassifierJob` payload;
  introduce `self._active_job: ClassifierJob | None` for the running attempt so
  `_on_worker_errored(...)` can still write the attempted `training_scope`,
  `prediction_scope`, table, feature key, and summary-derived counts even if
  the run fails before returning predictions;
- do not rebuild failed-run metadata from the controller's current
  `self._selected_*` state:
  by the time an error arrives, the widget may already have rebound to a
  different segmentation, table, feature matrix, or scope choice;
  even when stale late errors are guarded by job id, the correct metadata
  source is still the original failed `ClassifierJob` snapshot, not the
  current live selection;
- clear `self._active_job` alongside worker cleanup and job invalidation so it
  always reflects the currently active classifier attempt only;
- replace the remaining single-scope classifier metadata contract with an
  explicit multi-scope one;
- keep the still-useful general metadata fields:
  - `model_type`
  - `feature_key`
  - `table_name`
  - `roi_mode`
  - `trained`
  - `eligible`
  - `reason`
  - `training_timestamp`
  - `n_labeled_objects`
  - `n_features`
  - `class_labels_seen`
  - `rf_params`
- add explicit scope metadata:
  - `training_scope`
  - `training_regions`
  - `n_training_rows`
  - `prediction_scope`
  - `prediction_regions`
  - `n_predicted_rows`
- drop legacy single-scope fields that no longer describe the real write
  target cleanly:
  - `label_name`
  - `n_active_objects`
- update reload-state comparison so it no longer relies only on one stored
  `label_name`;
  staleness should be based on:
  - whether `feature_key` matches
  - whether `table_name` matches
  - whether stored `prediction_scope` matches the controller's current
    `prediction_scope` exactly
  - whether the current selected segmentation is covered by stored
    `prediction_regions`

Files:

- `src/napari_harpy/_classifier.py`
- `tests/test_classifier.py`
- `tests/test_persistence.py`
- `tests/test_widget.py`

Expected outcome:

- the controller can run selected-segmentation-only or complete-table prediction
  intentionally;
- persisted metadata describes what was actually trained and what was actually
  predicted;
- reload status can represent multi-region predictions correctly.
- after this slice, the codebase should be in a good position to re-enable
  `Train Classifier` if the widget wiring is already ready.

### 6. Add a Prediction-Scope Control and Hidden-Write Warning in the Widget

Status: [x] Implemented

Goal:

- make expanded write scope visible before it is ever executed.

Scope:

- add a `Prediction scope` control to the widget;
- place it directly below `Training scope` in the selector form;
- default it to `Selected segmentation only`;
- expose the two modes only:
  - `Selected segmentation only`
  - `All eligible regions in table`
- store the widget selection as `self._selected_prediction_scope`, defaulting
  to `DEFAULT_PREDICTION_SCOPE`;
- thread `prediction_scope=self.selected_prediction_scope` through both
  classifier `bind(...)` call sites in
  `_object_classification_widget.py`;
- mark the classifier stale when prediction scope changes, exactly like the
  existing training-scope control does now;
- enable and disable the prediction-scope combo under the same conditions as
  the training-scope combo;
- add one small inline warning label near the retrain area, separate from the
  existing `classifier_feedback` card;
- keep the warning UX lightweight in this slice:
  - no modal confirmation
  - no richer slice-8 status-card system yet
- show the inline warning only when:
  - `prediction_scope == "all"`
  - and the requested prediction scope reaches at least one region beyond the
    selected segmentation
- hide the inline warning when:
  - `prediction_scope == "selected_segmentation_only"`
  - or the selected table effectively resolves only to the current
    segmentation
- use warning copy that reflects slice-5 writeback semantics, not only that
  "more rows get predictions";
  recommended visible shape:
  - `Prediction scope covers N row(s) across M region(s).`
  - `Eligible rows will receive fresh predictions; other in-scope rows with invalid features will be cleared. Some updates may not be visible in the current selection.`
- add one small controller-facing preparation API so the widget can render that
  warning and count information before the user clicks `Train Classifier`;
- do not parse `status_message` in the widget for this;
- do not introduce a separate `ClassifierPreflight` dataclass; keep the
  controller-facing preparation state in `ClassifierPreparationSummary`;
- do not use `ClassifierJob` as the widget-facing preparation object:
  `ClassifierJob` is the worker execution payload, and constructing one during
  ordinary widget refresh would mix status rendering with `job_id` assignment,
  feature slicing/copying, and train/predict array construction;
- fold the useful part of the planned preparation-state refactor into this
  slice by making `ClassifierPreparationSummary` the single structured object
  that carries both resolved scopes and scalar preparation state; derive row
  and region counts from the resolved scopes rather than storing duplicate
  count fields:

```python
@dataclass(frozen=True)
class ClassifierPreparationSummary:
    training_scope: ResolvedClassifierScope
    prediction_scope: ResolvedClassifierScope
    eligible: bool
    reason: str
    labeled_count: int
    class_labels: tuple[int, ...]
    n_features: int | None

    @property
    def resolved_training_row_count(self) -> int:
        return self.training_scope.n_eligible_rows

    @property
    def resolved_prediction_row_count(self) -> int:
        return self.prediction_scope.n_eligible_rows

    @property
    def training_region_count(self) -> int:
        return len(self.training_scope.regions)
```

- change `ClassifierJob` and `ClassifierJobResult` to carry this summary as the
  authoritative preparation object, rather than storing parallel
  `training_scope` / `prediction_scope` fields beside it;
- use this target `ClassifierJob` shape, where `training_scope` and
  `prediction_scope` are convenience properties over the summary rather than
  independent dataclass fields:

```python
@dataclass(frozen=True)
class ClassifierJob:
    job_id: int
    feature_key: str
    label_name: str
    table_name: str
    predict_features: Any
    train_features: Any
    train_labels: np.ndarray
    summary: ClassifierPreparationSummary

    @property
    def training_scope(self) -> ResolvedClassifierScope:
        return self.summary.training_scope

    @property
    def prediction_scope(self) -> ResolvedClassifierScope:
        return self.summary.prediction_scope
```

- treat the properties above as the preferred compatibility surface for existing
  code and tests; do not reintroduce duplicate stored scope fields on
  `ClassifierJob`;
- apply the same rule to `ClassifierJobResult`: store the summary once, and read
  resolved scopes from `result.summary` instead of persisting a second copy on
  the result object;
- keep the intended object roles explicit:
  - `ClassifierPreparationSummary`: lightweight, side-effect-free description of
    the current bound classifier state for widgets, status, metadata counts, and
    worker setup;
  - `ClassifierJob`: heavier immutable worker payload that wraps the preparation
    summary plus `job_id`, `train_features`, `predict_features`, and
    `train_labels`;
  - `ClassifierJobResult`: worker output that carries predictions/confidences
    plus the original preparation summary used to produce them;
- extract the side-effect-free, non-worker, non-feature-slicing preparation work
  from `_prepare_classifier_job(...)` into a helper such as
  `_prepare_classifier_summary(...)`;
- `_prepare_classifier_summary(...)` should resolve scopes, feature-valid row
  positions, labeled-row counts, class labels, feature counts, and ineligible
  reasons, but should not construct full training or prediction feature arrays;
- keep `_prepare_classifier_job(...)` as the worker-payload builder: it should
  call `_prepare_classifier_summary(...)`, then slice `train_features` and
  `predict_features` only when a job is actually being launched;
- add a lightweight side-effect-free accessor on `ClassifierController`:

```python
def describe_current_preparation(self) -> ClassifierPreparationSummary | None:
    ...
```

- return `None` only when the controller is not sufficiently bound to describe
  the current selection at all; otherwise return a
  `ClassifierPreparationSummary` even when training is currently ineligible;
- update controller status, metadata persistence, error handling, and reload
  logic to read resolved scope information from the summary where practical;
- update tooltips so the button text still describes one action even when the
  training and prediction scopes differ.

Files:

- `src/napari_harpy/widgets/_object_classification_widget.py`
- `src/napari_harpy/_classifier.py`
- `tests/test_widget.py`
- `tests/test_classifier.py`

Expected outcome:

- complete-table prediction is opt-in and obvious;
- selected-segmentation-only prediction remains the safe default;
- the widget owns an explicit prediction-scope selection and passes it into the
  existing controller binding flow;
- the widget can render hidden-write warning copy from structured controller
  preparation data rather than parsing controller status text;
- the widget communicates hidden-row writes before they happen, not only after.
- classifier preparation state is described by one summary object rather than a
  scope pair plus a separate summary, so no follow-up preparation-state refactor
  is needed.
- this is the natural point to re-enable `Train Classifier` if it was disabled
  during earlier refactor slices.

### 7. Add End-to-End Multi-Region Classifier Coverage

Status: [x] Implemented

Goal:

- lock the final behavior in with controller, widget, and persistence tests.

Scope:

- add controller coverage for:
  - default training on labeled rows from multiple regions;
  - selected-segmentation-only training override;
  - selected-segmentation-only prediction leaving hidden-region predictions unchanged;
  - complete-table prediction updating all eligible regions only when selected;
  - reload status when the stored prediction scope covers multiple regions;
- add widget coverage for:
  - training-scope and prediction-scope controls;
  - row-count summary text;
  - hidden-write warning copy;
  - dirty-state invalidation when either scope changes;
- add persistence coverage for the expanded `classifier_config` fields.

Files:

- `tests/test_classifier.py`
- `tests/test_widget.py`
- `tests/test_persistence.py`

Expected outcome:

- the roadmap acceptance criteria are covered by tests that actually span
  multiple regions;
- later refactors cannot silently collapse back to one-region classifier
  assumptions.

### 8. Add a Rich Classifier Status Card UX

Status: [x] Implemented

Goal:

- present object-classification selection status and classifier preparation
  context through status-card specs, similar in spirit to the feature extraction
  widget's spec-builder pattern.

Scope:

- add `src/napari_harpy/widgets/_object_classification_status_card.py` with a
  small spec dataclass mirroring the feature extraction helper shape:

```python
@dataclass(frozen=True)
class _ObjectClassificationStatusCardSpec:
    title: str
    lines: tuple[str, ...]
    kind: StatusCardKind
    tooltip_message: str | None = None
```

- keep the helper module presentation-focused: it should build status-card
  specs from structured controller/widget state and should not own Qt widget
  classes;
- mirror the feature extraction widget's status-card wiring in
  `_object_classification_widget.py`:
  - keep `self.selection_status` as the persistent selection/annotation context
    card;
  - replace `self.prediction_scope_warning` with
    `self.classifier_preparation_status` for persistent classifier preparation
    context;
  - set the new preparation label's object name to
    `classifier_preparation_status`;
  - remove the old `prediction_scope_warning` widget attribute outright rather
    than keeping a compatibility alias;
  - keep `self.classifier_feedback` for transient classifier controller
    feedback;
  - add an `_apply_status_card_spec(label, spec)` helper that applies
    `_ObjectClassificationStatusCardSpec | None` to any of those `QLabel`
    cards, matching the feature extraction pattern;
- replace the loose `_labels_layer_preparation_message` /
  `_labels_layer_preparation_error` widget attributes with a structured
  labels-layer preparation result defined in
  `_object_classification_status_card.py`:

```python
@dataclass(frozen=True)
class _LabelsLayerPreparationResult:
    kind: Literal["none", "loaded", "activated", "error"]
    label_name: str | None = None
    coordinate_system: str | None = None
    error: str | None = None
```

- make `_prepare_selected_labels_layer()` produce that structured result while
  keeping the actual layer load/activate/register side effects in the widget;
- store/pass that result from the widget, but keep result formatting in the
  status-card helper;
- move the existing primary selection-card decision tree out of
  `_object_classification_widget.py` into a helper builder such as
  `build_object_classification_selection_status_card_spec(...)`;
- pass plain widget state into the selection-card builder, including:
  SpatialData availability, selected coordinate system, selected segmentation,
  loaded-labels state, the structured labels-layer preparation result, selected
  table, table-binding errors, missing-table-row messages, selected instance id,
  instance-key name, and current user class;
- let the status-card helper format labels-layer preparation messages such as
  "Loaded segmentation ..." or "Activated segmentation ..." from the structured
  result rather than storing already formatted message strings in the widget;
- introduce a dedicated widget status area for classifier preparation context,
  separate from the transient `classifier_feedback` card;
- add a classifier-preparation builder such as
  `build_object_classification_classifier_preparation_card_spec(...)`;
- reuse the structured preparation accessor introduced in slice 6 rather than
  adding another controller-facing preparation model;
- show the classifier preparation card only once classifier inputs are specific
  enough to make the card useful: selected segmentation, selected table,
  selected feature matrix, valid table binding, and an available
  `ClassifierPreparationSummary`;
- leave earlier setup blockers to the existing selection card. For example,
  "choose a segmentation", "no SpatialData loaded", "no linked annotation
  table", and table-binding errors should stay selection-card responsibilities
  until classifier preparation is meaningful;
- render richer preparation lines such as:

```text
Training: 182 labeled rows across 4 regions.
Prediction: 3,110 eligible rows in selected region.
Feature matrix: features_1, 12 features.
```

- use `success` for eligible classifier preparation;
- use `warning` when preparation is ineligible and include `summary.reason`;
- fold the existing table-wide hidden-write notice into the classifier
  preparation card rather than keeping a separate `prediction_scope_warning`
  card;
- when table-wide prediction covers regions beyond the selected segmentation,
  keep the preparation card in `success` state and include a line explaining that
  some prediction updates may not be visible in the current selection;
- keep the existing `classifier_feedback` card for transient run-time events
  such as training started, training failed, model stale, and model up to date;
- route `classifier_feedback` through the same helper module with a
  `build_object_classification_classifier_feedback_card_spec(...)` builder, while
  keeping it conceptually separate from persistent preparation context;
- align the resulting UX with the feature extraction widget's separation
  between selection/preparation status and controller feedback status;
- add widget tests for:
  - primary selection-card rendering after the helper move;
  - hidden classifier preparation card while setup blockers are still active;
  - eligible classifier preparation lines;
  - ineligible classifier preparation reason;
  - table-wide hidden-write notice folded into the preparation card;
  - transient classifier feedback remaining separate from preparation context.

Files:

- `src/napari_harpy/widgets/_object_classification_widget.py`
- `src/napari_harpy/widgets/_object_classification_status_card.py`
- `src/napari_harpy/_classifier.py`
- `tests/test_widget.py`

Expected outcome:

- classifier scope choices are visible in a stable, readable preparation card
  before the user clicks `Train Classifier`;
- transient controller feedback remains separate from persistent preparation
  context;
- the classifier widget UX matches the overall status-card style already used
  elsewhere in the codebase, especially feature extraction.

## Recommended Order

Recommended landing order:

1. slice 1
2. slice 2
3. slice 3
4. slice 4
5. slice 5
6. slice 6
7. slice 7
8. slice 8

Why this order:

- it creates the controller seam before changing behavior;
- it adds real multi-region test data before the UI becomes more complex;
- it delivers the highest-value behavior change first by making training
  table-level while keeping prediction safe;
- it treats metadata and reload semantics as first-class scope work instead of
  post-merge cleanup.
- it folds the preparation-summary cleanup into the prediction-scope widget slice,
  so there is no separate follow-up refactor between the UI warning and final
  multi-region coverage.
- it defers richer classifier UX composition until the scope semantics and
  metadata contract are stable enough to present cleanly.

## Notes

- `src/napari_harpy/_annotation.py` should need little or no behavior change
  for this roadmap item; its current `(region_key, instance_key)` write rule is
  already the right one.
- `src/napari_harpy/_classifier_viewer_styling.py` should mostly continue to
  work unchanged, because it already reads `pred_class` rows for the currently
  selected segmentation only.
- if implementation reveals ambiguous behavior for rows inside the requested
  prediction scope that are feature-invalid, prefer making that rule explicit in
  the controller and tests rather than leaving stale predictions in place
  silently.
