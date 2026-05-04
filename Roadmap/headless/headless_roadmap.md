# Headless Classifier Roadmap

This roadmap covers two related goals:

1. export the trained object classifier from the Object Classification widget;
2. apply that exported classifier in a headless workflow, optionally computing
   the matching feature matrix on the target data first.

The first goal should be implemented as a small, UI-visible capability. The
second goal should be built as a reusable Python API first, with any CLI or
workflow wrapper layered on top afterwards.

## Current Baseline

The Object Classification widget currently trains a `RandomForestClassifier`
inside the worker path in `src/napari_harpy/_classifier.py`.

Important current properties:

- `_fit_classifier_job(...)` creates a local `RandomForestClassifier`, fits it,
  predicts immediately, and returns only predictions, confidences, timestamps,
  params, and the preparation summary;
- `ClassifierController` writes prediction columns and `table.uns["classifier_config"]`
  after a worker returns;
- the fitted estimator is not retained after the worker finishes, so there is
  currently no model object to export without retraining;
- classifier preparation already resolves training and prediction scopes,
  feature-valid rows, class labels, feature count, table name, and feature key;
- feature extraction already has a validated request object
  (`FeatureExtractionRequest`) and delegates matrix creation to
  `hp.tb.add_feature_matrix(...)`;
- Harpy feature-matrix metadata is stored under
  `table.uns["feature_matrices"][feature_key]` and includes the feature columns
  that a future headless apply path must match.

## Design Direction

### Export Format

Use a single joblib-backed artifact for the first implementation, for example:

```text
my_classifier.harpy-classifier.joblib
```

`joblib` is a good fit for scikit-learn estimators and NumPy arrays and is
already available through the scikit-learn stack. It still uses pickle semantics,
so exported classifiers must be treated as trusted files only. Loading an
untrusted classifier artifact should be documented as unsafe.

Do not export only a raw estimator. Export a structured bundle with explicit
metadata beside the estimator. A target shape:

```python
@dataclass(frozen=True)
class ClassifierExportBundle:
    schema_version: int
    created_at: str
    napari_harpy_version: str | None
    sklearn_version: str | None
    model_type: str
    estimator: RandomForestClassifier
    feature_key: str
    feature_columns: tuple[str, ...]
    feature_names: tuple[str, ...]
    n_features: int
    class_labels_seen: tuple[int, ...]
    rf_params: dict[str, object]
    source_classifier_config: dict[str, object]
    source_table_name: str
    source_training_scope: str
    source_training_regions: tuple[str, ...]
    source_prediction_scope: str
    source_prediction_regions: tuple[str, ...]
    source_feature_metadata: dict[str, object]
```

`ClassifierExportBundle` is the in-memory representation used by napari-harpy.
The on-disk joblib payload should be a plain dict with the same semantic fields,
for example:

```python
{
    "bundle_type": "napari_harpy_classifier",
    "schema_version": 1,
    "metadata": {...},
    "estimator": fitted_random_forest,
}
```

`write_classifier_export_bundle(...)` should convert the dataclass to this dict
payload before calling `joblib.dump(...)`, and `read_classifier_export_bundle(...)`
should validate the payload before reconstructing `ClassifierExportBundle`. This
keeps the artifact easier to inspect and migrate than pickling the dataclass
instance directly, while still using joblib for the fitted scikit-learn model.

The important invariant is:

> The bundle stores the exact fitted estimator and the exact feature schema it
> expects. Any headless apply path must verify that the target feature matrix has
> the same number and order of feature columns before predicting.

### Source Metadata and Drift Prevention

The classifier export should combine two metadata sources from the selected
table, but neither source should be treated as a live link after export:

- `table.uns["classifier_config"]` is UI/training audit metadata. It records the
  feature key, table name, class labels, training/prediction scopes, row counts,
  random-forest params, and training timestamp. It does not currently store
  `feature_columns`.
- `table.uns["feature_matrices"][feature_key]` is Harpy feature-schema metadata.
  It records `features`, `feature_columns`, `source_label`, `source_image`,
  `coordinate_system`, dtype/backend metadata, and the feature-matrix schema
  version.

For export, `table.uns["feature_matrices"][feature_key]` should be the source of
truth for `feature_columns` and `feature_names`. Export should fail if this
metadata is missing or if `feature_columns` does not match the selected
`.obsm[feature_key]` column count. A classifier that only knows `n_features` is
not safe to reuse headlessly, because column identity and order matter.

To avoid metadata drift, do not build the export by casually rereading whatever
is in `.uns` at save time. Instead, create an in-memory snapshot when training
succeeds:

```python
@dataclass(frozen=True)
class ClassifierModelSnapshot:
    estimator: RandomForestClassifier
    classifier_config: dict[str, object]
    feature_metadata: dict[str, object]
    feature_columns: tuple[str, ...]
    feature_key: str
    trained_at: str
```

The snapshot should be built from the same successful worker result that writes
`table.uns["classifier_config"]`, plus the current
`table.uns["feature_matrices"][feature_key]` entry. Export then uses this
snapshot as its source of truth.

Before writing an artifact, validate that the current table still matches the
snapshot:

- selected `feature_key` equals `snapshot.feature_key`;
- current `.obsm[feature_key]` exists and has `len(snapshot.feature_columns)`
  columns;
- current `table.uns["feature_matrices"][feature_key]["feature_columns"]` equals
  `snapshot.feature_columns`;
- the controller is not dirty;
- no training job is active.

If any of these checks fail, export should refuse with a clear stale-model or
metadata-mismatch message. After reloading a table from zarr we may have
predictions and `classifier_config`, but we no longer have the fitted estimator
snapshot; export should remain unavailable until the user trains again in the
UI. This prevents exporting metadata-only "ghost models".

### Feature Recipe Versus Target Mapping

The exported bundle should include the source feature metadata for provenance,
but it should not assume that a new target dataset uses the same labels, images,
coordinate-system names, or table name.

Headless mode should therefore accept an explicit target mapping:

```python
@dataclass(frozen=True)
class HeadlessFeatureTarget:
    table_name: str
    feature_key: str
    triplets: tuple[FeatureExtractionTriplet, ...]
    overwrite_feature_key: bool = False
```

`HeadlessFeatureTarget` describes the target dataset wiring for feature
calculation. It answers: which target table should receive the feature matrix,
which `.obsm` key should be used, and which target coordinate-system /
segmentation / image triplets should be passed to `hp.tb.add_feature_matrix(...)`.
It intentionally does not carry `feature_columns`: the exported classifier
bundle is the authority on the required feature schema, and the headless apply
path must compare the feature matrix produced for this target against
`bundle.feature_columns` before predicting.

This lets a bundle trained on `blobs_labels` be applied to a different dataset
whose labels element might be named `cells`, with a different coordinate system
or image name.

## 1. Export the Classifier From the Widget

Status: [ ] Not started

### Implementation Plan

- add a new module, likely `src/napari_harpy/_classifier_export.py`, containing:
  - `ClassifierExportBundle`;
  - `ClassifierModelSnapshot`;
  - `build_classifier_export_bundle(...)`;
  - `write_classifier_export_bundle(path, bundle)`;
  - `read_classifier_export_bundle(path)`;
- store a fitted model snapshot in the controller after training:
  - change `ClassifierJobResult` to carry the fitted estimator, or add a
    dedicated worker result field such as `estimator`;
  - build `source_classifier_config` with `_build_classifier_config(...)`;
  - read `source_feature_metadata` from
    `table.uns["feature_matrices"][feature_key]`;
  - normalize and validate `feature_columns` from that feature metadata;
  - set `self._model_snapshot` in `_on_worker_returned(...)`;
  - clear that snapshot whenever the classifier becomes dirty, inputs change,
    training starts, training fails, reload happens, or feature matrices are
    overwritten;
- expose a controller method such as:

```python
def export_classifier(self, path: str | Path) -> ClassifierExportBundle:
    ...
```

- export should fail with a clear `ValueError` when:
  - no fitted model is available;
  - the classifier is dirty/stale;
  - a training job is running;
  - the selected feature matrix metadata is missing or incompatible;
  - current table metadata no longer matches the stored model snapshot;
- add an `Export Classifier` button to the Object Classification widget near
  `Train Classifier`;
- use `QFileDialog.getSaveFileName(...)` with a default extension such as
  `.harpy-classifier.joblib`;
- add a compact status-card message for export success/error, probably reusing
  the existing classifier feedback card rather than introducing a new large UI
  surface;
- keep export independent from zarr persistence:
  - exporting a classifier writes a model artifact to a chosen path;
  - it should not require writing the annotation table to zarr first;
  - the artifact metadata should still record source table/feature/scope details.

### Tests

- controller exports after a successful train and the artifact reloads with:
  - a usable estimator;
  - matching feature count;
  - matching feature columns and feature names from
    `table.uns["feature_matrices"][feature_key]`;
  - matching class labels;
  - `source_classifier_config`;
  - `source_feature_metadata`;
- export is disabled or raises when the model is dirty;
- export is disabled or raises while training is active;
- export is cleared after feature matrix overwrite or reload;
- export refuses if current feature metadata has drifted from the stored model
  snapshot;
- widget test covers button enabled state and a mocked save dialog path;
- round-trip artifact can predict the same classes as the in-memory estimator
  on the same feature matrix.

### Acceptance Criteria

- a user can train in the widget and save one classifier artifact;
- reloading that artifact through the Python API returns a model bundle with
  explicit feature schema metadata;
- no headless apply logic is required yet, but the artifact contains enough
  metadata to support it in a follow-up slice.

## 2. Add a Headless Apply API for Existing Feature Matrices

Status: [ ] Not started

This slice should apply an exported classifier to a target table that already
contains a compatible feature matrix.

### Proposed API

Create a small public module, for example `src/napari_harpy/headless.py`:

```python
def load_classifier(path: str | Path) -> ClassifierExportBundle:
    ...

def apply_classifier(
    sdata: SpatialData,
    bundle: ClassifierExportBundle,
    *,
    table_name: str,
    feature_key: str | None = None,
    prediction_regions: Sequence[str] | None = None,
    output_pred_class_column: str = "pred_class",
    output_pred_confidence_column: str = "pred_confidence",
) -> HeadlessClassificationResult:
    ...
```

Default behavior:

- use `feature_key=bundle.feature_key` unless overridden;
- predict for all eligible rows in the chosen table unless
  `prediction_regions` is provided;
- skip rows with non-finite feature values;
- clear in-scope invalid rows to unlabeled/NaN, matching the widget behavior;
- write prediction columns to the target table;
- write a target-side config entry documenting:
  - artifact path or artifact id;
  - bundle schema version;
  - source table/feature/scope metadata;
  - target table/feature/prediction regions;
  - number of predicted rows;
  - number of skipped feature-invalid rows;
  - apply timestamp.

### Required Refactor

Extract or reuse classifier helpers so the headless path does not need a Qt
controller:

- feature matrix normalization;
- finite-row mask calculation;
- prediction-column setup;
- region row-position resolution;
- prediction writing;
- classifier-config building, or a headless-specific config builder.

Avoid importing Qt-only widget/controller code in `headless.py`.

### Tests

- apply a saved bundle to the same table/feature matrix and reproduce widget
  predictions;
- apply a saved bundle to a second compatible table;
- reject missing feature key;
- reject feature matrix with wrong column count;
- reject feature metadata whose `feature_columns` order does not match the
  bundle when metadata is present;
- skip/clear rows with non-finite feature values;
- support multi-region prediction and selected-region prediction.

### Acceptance Criteria

- a Python script can load a bundle and apply it to an existing feature matrix
  without creating a napari viewer or Qt widget;
- prediction outputs and metadata are written to the target table consistently
  with the interactive classifier semantics.

## 3. Add Headless Feature Calculation Before Apply

Status: [ ] Not started

This slice wires `hp.tb.add_feature_matrix(...)` into the headless pipeline so a
target dataset can compute the required feature matrix before applying the
exported classifier.

### Proposed API

```python
def compute_features_for_classifier(
    sdata: SpatialData,
    bundle: ClassifierExportBundle,
    *,
    target: HeadlessFeatureTarget,
) -> HeadlessFeatureExtractionResult:
    ...

def apply_classifier_with_features(
    sdata: SpatialData,
    bundle: ClassifierExportBundle,
    *,
    target: HeadlessFeatureTarget,
    prediction_regions: Sequence[str] | None = None,
) -> HeadlessClassificationResult:
    ...
```

Implementation rules:

- reuse `FeatureExtractionTriplet` and the same Harpy parameter resolution logic
  used by `_run_feature_extraction_job(...)`;
- call `hp.tb.add_feature_matrix(...)` with:
  - `table_name=target.table_name`;
  - `feature_key=target.feature_key`;
  - `features=list(bundle.feature_names)`;
  - target triplet labels/images/coordinate systems/channels;
  - `overwrite_feature_key=target.overwrite_feature_key`;
- after feature extraction, verify the target matrix has exactly the expected
  feature schema before applying the model;
- if Harpy metadata is present, compare `feature_columns` to
  `bundle.feature_columns`;
- if Harpy metadata is missing, require at least `matrix.shape[1] == bundle.n_features`,
  but report that the schema could not be fully verified.

### Tests

- compute a matching feature matrix and apply the exported classifier in one
  call;
- target labels/image/coordinate-system names can differ from the source names;
- selected target triplets can cover one or multiple table regions;
- incompatible feature columns fail before predictions are written;
- feature extraction overwrite behavior is explicit and tested;
- backed SpatialData writes are reloadable.

### Acceptance Criteria

- a script can load a target SpatialData object, compute the needed features,
  and apply a UI-exported classifier without napari;
- the target mapping is explicit enough that source and target datasets do not
  need identical element names.

## 4. Optional CLI Wrapper

Status: [ ] Not started

Once the Python API is stable, add a CLI wrapper. This should be thin and should
call the same `headless.py` functions.

Possible shape:

```bash
napari-harpy-classifier apply \
  --sdata input.zarr \
  --classifier model.harpy-classifier.joblib \
  --table table \
  --feature-key features_classifier \
  --labels cells \
  --image image \
  --coordinate-system global \
  --write
```

Keep CLI design for later. The first priority is a tested Python API.

## Open Questions

- Should the artifact include class color palettes from `user_class_colors` or
  `pred_class_colors`, or should colors stay table/viewer-local?
- Should headless apply write into the existing `classifier_config` key, or use
  a separate key such as `classifier_apply_config` to distinguish training from
  applying an external model?
- Should export be allowed when the current model is stale but an older fitted
  estimator still exists? Recommendation: no. Export only when the controller is
  not dirty and the exported artifact represents the visible prediction state.
- Should we add `joblib` as an explicit project dependency even though it is
  available through scikit-learn? Recommendation: yes once the export API becomes
  public, so the dependency is declared intentionally.
- Do we need a safer serialization backend such as `skops.io`? Recommendation:
  not for the first implementation, but document that joblib/pickle artifacts
  are trusted-code artifacts and must not be loaded from untrusted sources.
