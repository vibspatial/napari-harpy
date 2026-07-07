# Object Classification on Custom `.obsm` Feature Matrices

Status: investigation

## Goal

Support object classification on any suitable feature matrix stored in
`AnnData.obsm`, not only matrices created through `hp.tb.add_feature_matrix`.

The classifier should still keep the current safety model: exported and
headless-applied classifiers must know the exact feature-column schema they were
trained on. For custom `.obsm` matrices, napari-harpy should provide explicit UI
and non-UI paths to register the required metadata under
`table.uns["feature_matrices"][feature_key]`.

## Current Behavior

The Object Classification widget already discovers all `.obsm` keys:

- `get_table_obsm_keys(...)` returns `sorted(table.obsm.keys())`;
- `_refresh_feature_matrix_keys(...)` populates the feature matrix combo from
  those keys.

So the current limitation is not feature-key discovery.

Training currently only needs a numeric, row-aligned matrix:

- `_prepare_classifier_summary(...)` normalizes `table.obsm[feature_key]` and
  checks row count, dimensionality, finite rows, and feature count;
- `_prepare_classifier_job(...)` copies the selected feature matrix into the
  worker job.

The stricter metadata requirement appears when napari-harpy needs an exportable
model snapshot or headless apply compatibility:

- `_store_model_snapshot(...)` reads
  `table.uns["feature_matrices"][feature_key]`;
- `normalize_feature_columns(...)` requires non-empty `feature_columns`;
- export validates that current metadata still matches the fitted snapshot;
- headless `apply_classifier(...)` requires the target feature matrix metadata
  to match the classifier bundle feature schema exactly.

This means a custom `.obsm` matrix can be trainable today, but after training
the controller cannot create an exportable classifier snapshot if the metadata
is missing.

## Required Metadata Contract

Harpy's `hp.tb.add_feature_matrix(...)` writes:

```python
table.obsm[feature_key] = matrix
table.uns["feature_matrices"][feature_key] = {
    "feature_columns": list(columns),
    "schema_version": 1,
    "backend": "numpy",
    "dtype": str(matrix.dtype),
    "source_label": [...],
    "source_image": [...],
    "source_channels": ...,
    "coordinate_system": [...],
    "features": list(requested_features),
}
```

For custom `.obsm` matrices, the minimum classifier/export contract is:

- `feature_columns`: exact ordered feature-column names;
- `features`: required by classifier export as the source feature names, but
  for custom matrices this should be a stable custom marker rather than
  inferred Harpy feature names;
- `schema_version`: useful for future migration;
- `dtype` and `backend`: useful diagnostics;
- `source_kind`: explicitly marks that the matrix came from a custom `.obsm`
  registration path.

Do not invent Harpy feature-extraction provenance for custom matrices.
Fields such as `source_image`, `source_channels`, `source_label`, and
`coordinate_system` are meaningful for matrices produced by
`hp.tb.add_feature_matrix(...)`, but they are usually meaningless for a custom
`.obsm` matrix. They should be omitted unless a future custom-registration API
can provide real, validated provenance.

The crucial invariant is that `feature_columns` must have exactly the same
length and order as the selected `.obsm[feature_key]` columns. A classifier
bundle should remain reusable only when the target metadata declares the same
ordered feature schema.

For applying an exported classifier to a custom `.obsm` matrix, compatibility
is stricter than "the key exists". The target table must provide the same
ordered schema and a live matrix with the same width:

1. `table.obsm[feature_key]` exists.
2. `table.uns["feature_matrices"][feature_key]["feature_columns"]` exists.
3. Target `feature_columns` exactly equals the classifier bundle's
   `feature_columns`.
4. The live target matrix has `shape[1] == len(feature_columns)`.
5. If the estimator exposes `n_features_in_`, it must match the live matrix
   width.

This means that a custom `.obsm` matrix with the same key but a different
number or order of columns is not compatible.

## `source_kind` Contract

`source_kind` is a required metadata marker for disambiguating how feature
metadata should be interpreted. Harpy feature-matrix metadata written by
`hp.tb.add_feature_matrix(...)` now includes this key.

Allowed source kinds:

- `"custom_obsm"`: metadata was registered for an arbitrary existing `.obsm`
  matrix, and should be treated as a feature schema only;
- `"harpy_add_feature_matrix"`: metadata was produced by
  `hp.tb.add_feature_matrix(...)`, and can be treated as a Harpy
  feature-extraction recipe for the recompute path.

napari-harpy should be strict: missing, non-string, or unknown `source_kind`
values are invalid metadata. We do not need a legacy fallback for metadata
without `source_kind`.

## Proposed Non-UI Path

Add a Qt-free helper in `napari_harpy.core.feature_matrix_metadata`.

This module should own generic feature-matrix concerns that are not specific to
classifier training, feature extraction, or Qt widgets:

- normalizing/validating `.obsm` feature-matrix shape;
- inferring matrix width and dtype/backend facts;
- registering feature-matrix schema metadata.

Suggested API:

```python
def register_feature_matrix_metadata(
    table: AnnData,
    feature_key: str,
    *,
    feature_columns: Sequence[str] | None = None,
    features: Sequence[str] | None = None,
    overwrite: bool = False,
) -> dict[str, object]:
    ...
```

Behavior:

1. Validate that `feature_key` exists in `table.obsm`.
2. Reuse the classifier feature-matrix normalization rules so 1D arrays,
   dense arrays, and sparse matrices follow the same shape contract as
   training.
3. Infer `n_features` from the normalized matrix.
4. If `feature_columns` is omitted, generate deterministic names. Suggested
   default: `f"{feature_key}_{index}"`, or another stable, documented pattern.
5. Validate that generated or supplied column names are non-empty, unique, and
   match `n_features`.
6. If `features` is omitted, set it to a stable custom marker such as
   `["custom_obsm"]`. Do not default it to `feature_columns`: names such as
   `"mean"` could accidentally be interpreted as Harpy intensity features by
   headless feature-extraction code.
7. Refuse to replace existing metadata unless `overwrite=True`.
8. Write `table.uns["feature_matrices"][feature_key]`.
9. Return the written metadata.

Suggested metadata for a custom matrix:

```python
{
    "feature_columns": list(feature_columns),
    "schema_version": 1,
    "backend": "sparse" if issparse(matrix) else "numpy",
    "dtype": str(matrix.dtype),
    "features": ["custom_obsm"],
    "source_kind": "custom_obsm",
}
```

The helper should not silently train or export anything. It only registers the
selected `.obsm` matrix as a classifier feature matrix by creating the schema
metadata that export/apply already require.

## Headless API Semantics

Custom `.obsm` support should distinguish two headless paths:

- `headless.apply_classifier(...)` applies an exported classifier to an
  existing compatible feature matrix. This should support custom `.obsm`
  matrices, as long as the target table already contains `.obsm[feature_key]`
  and matching `feature_columns` metadata. Matching metadata means the exact
  same ordered feature schema as the exported classifier, and the live target
  matrix width must match that schema.
- `headless.apply_classifier_with_feature_extraction(...)` recomputes features
  by calling `hp.tb.add_feature_matrix(...)`. This should not support custom
  `.obsm` classifier bundles, because napari-harpy does not know how to
  reconstruct an arbitrary custom matrix from image/labels elements.

The current recompute path inspects `classifier.feature_names` and, for
intensity-derived Harpy features, requires `classifier.source_channels`.
For custom `.obsm` metadata, `features` must therefore avoid names that look
like Harpy feature names, and the recompute path should fail clearly when
`source_kind == "custom_obsm"`. Missing or unknown `source_kind` is invalid
feature metadata.

Suggested error:

```python
raise ValueError(
    "This classifier was trained on a registered custom `.obsm` feature matrix. "
    "napari-harpy cannot recompute that matrix with `hp.tb.add_feature_matrix`. "
    "Use `headless.apply_classifier(...)` on a table that already contains a "
    "compatible `.obsm` matrix."
)
```

For `headless.apply_classifier(...)`, custom `.obsm` compatibility failures
should also use clearer custom-specific messages. Generic errors such as
"columns do not match the classifier bundle feature schema" are technically
correct, but they do not explain the custom matrix contract. When
`source_kind == "custom_obsm"` is involved, errors should explicitly say that
custom `.obsm` classifiers require the same feature columns in the same order
and a live matrix with the same width.

Suggested schema-mismatch message shape:

```text
Custom feature matrix `my_features` is not compatible with this classifier.
Custom `.obsm` classifiers require the same feature columns in the same order.
Register or select a feature matrix with matching metadata, or retrain the classifier.
```

Suggested metadata/matrix-width mismatch message shape:

```text
Custom feature matrix `my_features` is not internally consistent.
Its metadata describes 10 feature columns, but `.obsm["my_features"]` has 12 columns.
Update the metadata or replace the matrix before applying this classifier.
```

## Proposed UI Path

Add an explicit button to the Object Classification widget, probably near the
feature matrix selector or classifier action row:

- label: `Register Feature Matrix`;
- tooltip: explain that this records feature-column metadata for the selected
  `.obsm` matrix so classifiers can be exported and reused;
- enabled only when a table and feature key are selected and the table binding
  is valid.

UI states:

- If the selected `.obsm` key has valid matching metadata, the button can be
  disabled with a tooltip such as "This feature matrix is already registered."
- If metadata is missing, enable the button.
- If metadata exists but its `feature_columns` count does not match the matrix,
  surface this as a warning and require an explicit repair/overwrite action
  before replacing metadata.

On click:

1. Call the core registration helper for the current table and feature key.
2. Mark the table dirty so the metadata can be written to disk.
3. Rebind or refresh classifier state.
4. If a fitted model snapshot exists or the classifier is currently clean,
   mark the classifier stale because the selected feature schema has changed.
5. Show a success status explaining that the feature matrix metadata was
   registered and the classifier should be trained again before export if
   needed.

The UI should not auto-register metadata during training. Keeping registration
explicit makes the feature schema choice visible and preserves the current
export/apply safety model.

## Persistence Requirement

The current `Write Table State` path writes `obs` and a whitelist of selected
`uns` entries. That whitelist currently includes classifier configs and color
metadata, but not `feature_matrices`.

If the UI writes custom feature metadata only in memory, it would be lost after
reloading from zarr unless `feature_matrices` is persisted too.

Therefore this feature should also update the persistence path so
`table.uns["feature_matrices"]` is included when writing table state.

This also means button text/tooltips that currently say "classifier metadata"
may remain acceptable, but the implementation should make clear that feature
matrix metadata is part of the persisted table state.

## Things To Watch

- Synthetic column names become part of the classifier contract. If a user
  exports a classifier trained on a custom matrix, any target dataset must use
  the same ordered `feature_columns` and the same matrix width to apply it.
- Custom `.obsm` classifier bundles are compatible with existing-matrix
  headless apply, not with headless feature recomputation. The UI and errors
  should avoid suggesting that napari-harpy can reconstruct arbitrary custom
  matrices.
- Auto-generated names must be deterministic and documented.
- Existing Harpy-created metadata should not be overwritten accidentally.
- Missing metadata and mismatched metadata are different states:
  - missing metadata can be registered;
  - mismatched metadata may indicate stale or corrupted schema and should be an
    explicit repair action.
- Sparse matrices should preserve sparse training behavior but still expose a
  reliable `dtype` and column count.
- Backed datasets need persistence of `feature_matrices`; unbacked datasets
  only keep the registration in memory.

## Proposed Implementation Slices

### Slice 1: Core Metadata Helper

Status: implemented.

Add `napari_harpy.core.feature_matrix_metadata` as the Qt-free home for generic
feature-matrix logic.

Move the existing `_normalize_feature_matrix(...)` implementation out of
`napari_harpy.core.classifier` and into
`napari_harpy.core.feature_matrix_metadata`.
Update callers to import the moved function from the new module. Do not keep a
compatibility alias in `classifier.py`; tests and callers should use the new
module path.

Then add the non-UI metadata registration helper. This is the foundation for
the widget path.

Suggested public/core API:

```python
register_feature_matrix_metadata(
    table,
    feature_key,
    *,
    feature_columns=None,
    features=None,
    overwrite=False,
)
```

The helper should:

- validate that `table.obsm[feature_key]` exists;
- normalize the matrix with the moved `normalize_feature_matrix(...)` behavior;
- infer matrix width;
- generate deterministic `feature_columns` when none are provided;
- write `table.uns["feature_matrices"][feature_key]`;
- use `features == ["custom_obsm"]` by default;
- use `source_kind == "custom_obsm"`;
- avoid inventing Harpy extraction provenance fields.

Tests:

- missing `.obsm` key raises a clear error;
- 1D array registers as one feature column;
- dense 2D array registers deterministic feature columns;
- sparse 2D matrix registers with matching column count and dtype;
- supplied `feature_columns` are preserved;
- duplicate, empty, or wrong-length `feature_columns` are rejected;
- existing metadata is not overwritten unless `overwrite=True`;
- written metadata satisfies `normalize_feature_columns(...)`;
- written metadata uses `features == ["custom_obsm"]` by default and
  `source_kind == "custom_obsm"`;
- written metadata does not invent `source_image`, `source_channels`,
  `source_label`, or `coordinate_system`;
- existing Harpy metadata with `source_kind == "harpy_add_feature_matrix"` is
  not treated as custom metadata.

### Slice 2: Metadata State Inspection

Status: implemented.

Add a Qt-free helper/model that describes the metadata state for one selected
table and feature key. This should not mutate the table.

The state should distinguish:

- missing from `.obsm`;
- present in `.obsm` but not a usable feature matrix;
- present but unregistered;
- registered and valid;
- registered but mismatched with the live matrix shape.

The helper should also report useful details for UI and error text, such as:

- live matrix width;
- live matrix validation error, when the `.obsm` value is present but unusable;
- number of metadata columns;
- the validated `source_kind`;
- whether metadata has `source_kind == "custom_obsm"`.

Only the allowed source kinds are valid: `"harpy_add_feature_matrix"` and
`"custom_obsm"`. Missing or unknown `source_kind` values should report
registered-but-mismatched metadata.

Tests:

- missing matrix key reports a missing-matrix state;
- present but non-2D or wrong-row-count matrix reports an invalid-matrix state;
- matrix without `feature_matrices` metadata reports unregistered;
- valid custom metadata reports registered/valid;
- valid Harpy metadata with `source_kind == "harpy_add_feature_matrix"` reports
  registered/valid but not custom;
- metadata without `source_kind` reports mismatched metadata;
- metadata column count mismatch reports a mismatched state.

### Slice 3: Classifier/Headless Compatibility Errors

Status: implemented.

Keep the existing validation semantics, but add clearer custom-specific error
messages for compatibility failures involving `source_kind == "custom_obsm"`.
The custom-specific path should apply when either side of the compatibility
check is custom:

- the exported classifier bundle has `source_kind == "custom_obsm"`;
- the target feature matrix metadata has `source_kind == "custom_obsm"`.

The validation should still require:

- target metadata `feature_columns` exactly matches the classifier bundle's
  `feature_columns`;
- target live matrix width matches the target metadata;
- estimator `n_features_in_`, when present, matches live matrix width.

Custom-specific messages should explain:

- same feature columns are required;
- the order must be the same;
- live matrix width must match the metadata/schema;
- users should register/select a matching matrix or retrain.

Implementation notes:

- `_validate_feature_matrix_compatible_with_bundle(...)` is the central place
  for source-versus-target `feature_columns` schema compatibility.
- `_validate_current_feature_matrix_matches_columns(...)` is the central place
  for checking that the live `.obsm` matrix width still matches its target
  metadata.
- `ClassifierExportBundle.source_kind` and the target metadata
  `source_kind` should decide whether to use custom-specific wording.
- Harpy metadata with `source_kind == "harpy_add_feature_matrix"` can keep the
  current generic wording unless we separately improve all messages.

Tests:

- custom `.obsm` apply is rejected when target metadata has the same key but a
  different ordered `feature_columns` schema, with a message explaining that
  custom matrices require the same columns in the same order;
- custom `.obsm` target metadata gets the same custom-specific schema error
  even if the classifier bundle itself was trained from Harpy metadata;
- custom `.obsm` apply is rejected when target metadata matches but the live
  target matrix has a different number of columns, with a message explaining
  that the metadata and live matrix are internally inconsistent;
- Harpy metadata with `source_kind == "harpy_add_feature_matrix"` keeps the
  current generic behavior unless separately improved.

### Slice 4: Headless Recompute Guard

Status: implemented.

Make the headless API behavior explicit:

- `headless.apply_classifier(...)` should continue to accept custom `.obsm`
  classifier bundles when the target table already has a compatible matrix and
  matching metadata.
- `headless.compute_features_for_classifier(...)` should reject classifier
  bundles whose source metadata has `source_kind == "custom_obsm"`, because it
  recomputes features through `hp.tb.add_feature_matrix(...)` and therefore
  requires a Harpy feature-extraction recipe.
- `headless.apply_classifier_with_feature_extraction(...)` should reject
  classifier bundles whose source metadata has `source_kind == "custom_obsm"`,
  with a clear message telling users to use `headless.apply_classifier(...)`
  on an existing compatible matrix.
- Missing or unknown `source_kind` should be rejected as invalid feature
  metadata.

Implementation notes:

- Put the guard in `compute_features_for_classifier(...)`, before
  `_resolve_classifier_source_channels(...)`. Custom `.obsm` metadata should
  not be inspected as if it were Harpy feature-extraction metadata.
- `apply_classifier_with_feature_extraction(...)` and
  `apply_classifier_with_feature_extraction_from_path(...)` should inherit this
  behavior through `compute_features_for_classifier(...)`.
- The error should explain that custom `.obsm` classifiers cannot recompute
  features and that users should call `headless.apply_classifier(...)` on a
  table that already contains a compatible registered matrix.
- `source_kind == "harpy_add_feature_matrix"` should continue through the
  current recompute path.

Tests:

- exported custom `.obsm` classifier can be applied with
  `headless.apply_classifier(...)` to a table that contains matching
  `feature_columns` and a matching live matrix width;
- the same classifier is rejected by `headless.compute_features_for_classifier(...)`;
- the same classifier is rejected by
  `headless.apply_classifier_with_feature_extraction(...)`;
- the rejection message points users to `headless.apply_classifier(...)` with
  an existing compatible matrix;
- no `source_channels` requirement is triggered for custom `.obsm` metadata;
- Harpy classifier metadata with `source_kind == "harpy_add_feature_matrix"`
  still follows the existing recompute behavior.

### Slice 5: Persistence

Status: implemented.

Include `feature_matrices` in the selected table-state write path.

This slice is persistence plumbing only. The core
`register_feature_matrix_metadata(...)` helper intentionally has no app-state
context, so it should not mark a backed table dirty by itself. Dirty-state
marking belongs to the later UI/action slice that calls the registration helper.

Implementation notes:

- add `feature_matrices` to the `uns_keys` whitelist used by
  `PersistenceController.write_table_state(...)`;
- keep reload behavior unchanged, because reload already reads the full table
  `uns` mapping from zarr;
- do not write arbitrary new `.obsm` matrices from this path. Feature
  extraction remains responsible for feature-matrix data writes. This slice only
  ensures registration metadata for an already-present feature key is persisted.

Tests:

- `Write Table State` writes `uns["feature_matrices"]` to zarr;
- reload preserves the registered custom metadata;
- writing table state clears an already-dirty table after the metadata has been
  persisted.

### Slice 6: Widget UI State

Add the widget state, status text, and tooltips needed to expose custom metadata
registration without yet wiring the button action.

This slice is UI state only. It should add the visible registration affordance
and all state/tooltip/warning behavior, but the button must not register
metadata yet. Actual registration, dirty marking, classifier stale marking, and
success feedback belong to Slice 7.

Implementation notes:

- add a `Register Feature Matrix` button near the feature matrix selector;
- give the button a stable object name such as
  `register_feature_matrix_button`;
- do not connect the button to a click action yet;
- use `inspect_feature_matrix_metadata(...)` to derive the selected feature
  matrix metadata state;
- add a focused widget helper such as
  `_update_feature_matrix_metadata_controls()`;
- call that helper from the central `_update_selection_status()` refresh path
  so table, feature-key, reload, and shared feature-matrix events all refresh
  the button consistently;
- reuse the existing validation/status area for invalid or mismatched metadata
  warnings rather than introducing a second warning card in this slice.

The widget should use the metadata-state helper to drive:

- whether the selected matrix is already registered;
- whether registration is available;
- whether existing metadata is mismatched.

State flow:

1. The user selects a feature matrix.
2. The widget calls `inspect_feature_matrix_metadata(table, feature_key)`.
3. If the state is `unregistered`, enable `Register Feature Matrix`.
4. If the state is `registered_valid`, disable the button because no
   registration action is needed.
5. If the state is `registered_mismatched`, disable the button and show a
   warning. Do not offer a UI repair/overwrite path.
6. If the state is `invalid_matrix` or `missing_matrix`, disable the button and
   show/tooltip the problem. Do not offer a UI repair path.

The widget should not treat `registered_mismatched` as "safe to register."
That state means metadata already exists but does not match the live matrix or
does not satisfy the current metadata contract. Silently overwriting it could
discard meaningful schema information, so the UI should only report the problem.
Fixing mismatched metadata remains a non-UI/manual responsibility for now.

Suggested UI behavior:

- no selected table/key: button disabled with normal selection tooltip;
- missing `.obsm` key or invalid matrix: button disabled and warning status;
- unregistered valid matrix: button enabled, tooltip explains that registering
  records feature-column metadata for export/reuse;
- valid custom metadata: button disabled, tooltip says it is already registered
  as a custom `.obsm` feature matrix;
- valid Harpy metadata with `source_kind == "harpy_add_feature_matrix"`:
  button disabled, tooltip says it is already registered from Harpy feature
  extraction;
- mismatched metadata: button disabled, warning status, and no silent overwrite.

Tests:

- button exists with object name `register_feature_matrix_button`;
- no selected table/key disables the button with a selection tooltip;
- button/status state is correct for unregistered matrix;
- button/status state is correct for valid custom metadata;
- button/status state is correct for valid Harpy metadata with
  `source_kind == "harpy_add_feature_matrix"`;
- invalid live `.obsm` matrix disables the button and produces a warning
  state;
- missing or unknown `source_kind` disables the button and produces a warning
  state;
- mismatched metadata disables the button and produces a warning state.

### Slice 7: Register Feature Matrix Button

Wire the UI button to the core registration helper.

On success:

- mark table state dirty;
- clear/refresh any stale warning;
- mark classifier stale if appropriate;
- refresh classifier/export controls.

Tests:

- button is enabled for a selected unregistered `.obsm` matrix;
- clicking registers metadata under `table.uns["feature_matrices"][feature_key]`;
- classifier export becomes possible after registering metadata and retraining;
- already registered matrices do not enable accidental overwrite.

### Slice 8: Optional Explicit Column-Name UI

Only if needed later, add a small advanced dialog for editing feature column
names before registration. The first implementation can use deterministic names
without a dialog, because this keeps the feature small and testable.

### Future Harpy Metadata Schema Cleanup

If we want every feature matrix to carry an explicit source kind, adapt
`hp.tb.add_feature_matrix(...)` upstream in Harpy to write:

```python
"source_kind": "harpy_add_feature_matrix"
```

This should be a separate Harpy metadata-schema change, not a napari-harpy
post-processing step. napari-harpy should remain compatible with both new
Harpy metadata that includes this key and older metadata that does not.
