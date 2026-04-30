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

## Current Codebase Baseline

The current code is already close in a few important ways, but the remaining
classifier work is real and mostly centered in one controller.

What already fits the cross-sample direction:

- `src/napari_harpy/_annotation.py` already writes user labels by
  `region_key + instance_key`, so interactive annotation is sample-local while
  still writing into a shared table correctly.
- `src/napari_harpy/_spatialdata.py` already supports multi-region table
  metadata and validates duplicate `instance_key` values per region rather than
  globally.
- the object-classification widget already binds to one selected segmentation,
  one selected table, and one selected feature matrix in a way that can remain
  compatible with a table-global training request.

What is still single-region today:

- `src/napari_harpy/_classifier.py` computes one `active_mask` from
  `self._selected_label_name` and uses that same row subset for both training
  and prediction.
- the worker job payload stores `active_positions`, and success / ineligible
  paths both write predictions only for those active rows.
- classifier status and metadata still describe one active region using fields
  such as `label_name` and `n_active_objects`.
- reload logic treats stored predictions as stale whenever the currently
  selected segmentation differs from the one recorded in
  `classifier_config["label_name"]`.
- `src/napari_harpy/widgets/_object_classification_widget.py` has no explicit
  training-scope or prediction-scope controls, and its copy still assumes a
  single-region training action.

Additional gap to address explicitly:

- the roadmap says classifier-eligible rows should have finite, non-missing
  feature values, but the current classifier path validates matrix shape only.
  Scope work is a good moment to make row eligibility explicit instead of
  leaving NaN / inf handling implicit inside scikit-learn behavior.

Testing baseline:

- `tests/test_classifier.py` and the classifier-focused widget tests in
  `tests/test_widget.py` currently assume one selected region is both the
  training set and the prediction target.
- `tests/conftest.py` still provides a single-region `sdata_blobs` fixture.
- classifier metadata is also asserted in reload / persistence flows, so
  `tests/test_persistence.py` will likely need coverage updates even though it
  is not named in the roadmap item.

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

- `selected_region_only`
- `all`

Interpretation:

- `selected_region_only` resolves to the one `region_key` value for the
  segmentation currently selected in `ObjectClassificationWidget`;
- `all` resolves at execution time to the eligible set of table regions for the
  selected table and feature matrix.

`all` should remain the compact persisted mode value, while the fully expanded
  resolved region list is stored separately.

## Slices

### 1. Extract Explicit Scope Resolution in the Classifier Controller

Status: [ ] Planned

Goal:

- move the controller from one implicit `active_mask` to explicit training and
  prediction scope resolution while preserving today’s behavior.

Scope:

- add controller-level scope types or literals for `selected_region_only` and
  `all`;
- introduce a resolved-scope data model that can hold:
  - requested training scope mode;
  - resolved training regions;
  - training-row positions;
  - requested prediction scope mode;
  - resolved prediction regions;
  - prediction-row positions;
- factor row-resolution logic out of `_prepare_classifier_job(...)` into helper
  functions;
- rename internal `active_positions` / `active_row_count` concepts toward
  prediction-specific names so later slices do not keep using `active` as a
  proxy for both concerns;
- keep runtime behavior equivalent to today by defaulting both scopes to
  `selected_region_only`;
- add explicit row eligibility helpers for:
  - valid scope membership;
  - finite feature rows for prediction;
  - finite feature rows plus non-zero `user_class` for training.

Files:

- `src/napari_harpy/_classifier.py`
- `tests/test_classifier.py`

Expected outcome:

- the controller has a clean seam for later cross-sample behavior;
- no user-visible classifier-scope behavior changes yet;
- if keeping the old train path would slow the refactor, the widget may disable
  `Train Classifier` during this slice rather than preserving the old
  single-region action;
- multi-region tests can assert scope resolution without needing widget changes.

### 2. Add Multi-Region Test Builders for Classifier Work

Status: [ ] Planned

Goal:

- stop depending on a single-region fixture for the new controller behavior.

Scope:

- add a small reusable way to construct a classifier table that spans at least
  two labels regions;
- keep the data builder narrow and deterministic rather than expanding the
  global fixture immediately;
- make it easy to create scenarios where:
  - the selected region has no local labels but another region does;
  - duplicate `instance_key` values appear across regions but not within a
    region;
  - one region has valid features while another has invalid feature rows.

Files:

- `tests/test_classifier.py`
- `tests/test_widget.py`
- optionally `tests/conftest.py` if the helper clearly becomes shared

Expected outcome:

- later slices can prove true multi-region behavior instead of reusing
  single-region assumptions;
- the new fixture surface stays local until reuse justifies promoting it.

### 3. Switch Default Training to Table-Wide Rows in the Controller

Status: [ ] Planned

Goal:

- implement the roadmap’s most important behavior change first:
  training becomes table-level by default while prediction remains region-local.

Scope:

- add a controller selection path for `training_scope`;
- resolve training rows from all eligible labeled rows in the selected table
  when training scope is `all`;
- keep prediction scope fixed at `selected_region_only` in this slice;
- update training eligibility and status reporting so it distinguishes:
  - number of resolved training rows;
  - number of labeled training rows;
  - number of resolved prediction rows;
  - number of training regions involved;
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
- the widget may still keep `Train Classifier` disabled until the scope UI and
  metadata contract catch up;
- the biggest behavior change lands before UI control proliferation.

### 4. Add an Explicit Training-Scope Control to the Widget

Status: [ ] Planned

Goal:

- expose the new training default while still allowing region-local training as
  an explicit override.

Scope:

- add a `Training scope` control to
  `src/napari_harpy/widgets/_object_classification_widget.py`;
- default it to `all`;
- support the two modes only:
  - `Selected region only`
  - `All eligible labeled regions in table`
- propagate the chosen mode into controller binding;
- mark the classifier dirty when the training scope changes;
- surface a preflight summary line such as:

```text
Training: 182 labeled rows across 4 regions
Prediction: 3,110 rows in selected region
```

- keep prediction behavior unchanged in this slice;
- keep annotation and styling flows unchanged.

Files:

- `src/napari_harpy/widgets/_object_classification_widget.py`
- `src/napari_harpy/_classifier.py`
- `tests/test_widget.py`

Expected outcome:

- the user can see that training is table-level by default;
- the user can opt back into selected-region-only training when desired;
- depending on implementation order, the train button may remain temporarily
  disabled until prediction-scope and metadata behavior are also ready;
- the widget makes the training set size visible before a run starts.

### 5. Add Prediction Scope Support in the Controller and Metadata

Status: [ ] Planned

Goal:

- enable the second axis of scope change without hiding the write target.

Scope:

- add a controller selection path for `prediction_scope`;
- keep the default as `selected_region_only`;
- support `all` as an explicit mode that resolves prediction rows across all
  eligible table regions;
- update worker result application so prediction writes target the resolved
  prediction-row positions, not the old single-region `active_positions`;
- update ineligible-state handling so it reasons about prediction-target rows
  explicitly;
- expand classifier metadata in `table.uns[CLASSIFIER_CONFIG_KEY]` to include:
  - `training_scope`
  - `training_regions`
  - `n_training_rows`
  - `prediction_scope`
  - `prediction_regions`
  - `n_predicted_rows`
- update reload-state comparison so it no longer relies only on one stored
  `label_name`; staleness should be based on table, feature matrix, and whether
  the current selected region is covered by the stored prediction scope.

Files:

- `src/napari_harpy/_classifier.py`
- `tests/test_classifier.py`
- `tests/test_persistence.py`
- `tests/test_widget.py`

Expected outcome:

- the controller can run selected-region-only or complete-table prediction
  intentionally;
- persisted metadata describes what was actually trained and what was actually
  predicted;
- reload status can represent multi-region predictions correctly.
- after this slice, the codebase should be in a good position to re-enable
  `Train Classifier` if the widget wiring is already ready.

### 6. Add a Prediction-Scope Control and Hidden-Write Warning in the Widget

Status: [ ] Planned

Goal:

- make expanded write scope visible before it is ever executed.

Scope:

- add a `Prediction scope` control to the widget;
- default it to `Selected region only`;
- expose the two modes only:
  - `Selected region only`
  - `All eligible regions in table`
- when `all` is selected, show clear warning copy that rows outside the visible
  coordinate system may be updated;
- reflect the resolved prediction-row count in the existing status / feedback
  area before the user clicks `Train Classifier`;
- update tooltips so the button text still describes one action even when the
  training and prediction scopes differ.

Files:

- `src/napari_harpy/widgets/_object_classification_widget.py`
- `src/napari_harpy/_classifier.py`
- `tests/test_widget.py`

Expected outcome:

- complete-table prediction is opt-in and obvious;
- selected-region-only prediction remains the safe default;
- the widget communicates hidden-row writes before they happen, not only after.
- this is the natural point to re-enable `Train Classifier` if it was disabled
  during earlier refactor slices.

### 7. Add End-to-End Multi-Region Classifier Coverage

Status: [ ] Planned

Goal:

- lock the final behavior in with controller, widget, and persistence tests.

Scope:

- add controller coverage for:
  - default training on labeled rows from multiple regions;
  - selected-region-only training override;
  - selected-region-only prediction leaving hidden-region predictions unchanged;
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

## Recommended Order

Recommended landing order:

1. slice 1
2. slice 2
3. slice 3
4. slice 4
5. slice 5
6. slice 6
7. slice 7

Why this order:

- it creates the controller seam before changing behavior;
- it adds real multi-region test data before the UI becomes more complex;
- it delivers the highest-value behavior change first by making training
  table-level while keeping prediction safe;
- it treats metadata and reload semantics as first-class scope work instead of
  post-merge cleanup.

## Notes

- `src/napari_harpy/_annotation.py` should need little or no behavior change
  for this roadmap item; its current `(region_key, instance_key)` write rule is
  already the right one.
- `src/napari_harpy/_classifier_viewer_styling.py` should mostly continue to
  work unchanged, because it already reads `pred_class` rows for the currently
  selected region only.
- if implementation reveals ambiguous behavior for rows inside the requested
  prediction scope that are feature-invalid, prefer making that rule explicit in
  the controller and tests rather than leaving stale predictions in place
  silently.
