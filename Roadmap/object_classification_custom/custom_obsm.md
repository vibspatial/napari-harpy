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
- `features`: required by classifier export as the source feature names;
- `schema_version`: useful for future migration;
- `dtype` and `backend`: useful diagnostics;
- source/provenance fields may be empty or explicitly marked as custom because
  there may be no Harpy feature-extraction source.

The crucial invariant is that `feature_columns` must have exactly the same
length and order as the selected `.obsm[feature_key]` columns. A classifier
bundle should remain reusable only when the target metadata declares the same
ordered feature schema.

## Proposed Non-UI Path

Add a Qt-free helper, likely in a small core module such as
`napari_harpy.core.feature_matrix_metadata`.

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
6. If `features` is omitted, set it to the same list as `feature_columns`, or to
   a stable custom marker if we decide export should distinguish source feature
   groups from concrete columns.
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
    "source_label": [],
    "source_image": [],
    "source_channels": None,
    "coordinate_system": [],
    "features": list(features),
    "source_kind": "custom_obsm",
}
```

The helper should not silently train or export anything. It only registers the
selected `.obsm` matrix as a classifier feature matrix by creating the schema
metadata that export/apply already require.

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
  the same ordered `feature_columns` to apply it.
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

### Slice 1: Core Metadata Registration Helper

Add a Qt-free helper to register metadata for an existing `.obsm` matrix.

Tests:

- missing `.obsm` key raises a clear error;
- 1D array registers as one feature column;
- dense 2D array registers deterministic feature columns;
- sparse 2D matrix registers with matching column count and dtype;
- supplied `feature_columns` are preserved;
- duplicate, empty, or wrong-length `feature_columns` are rejected;
- existing metadata is not overwritten unless `overwrite=True`;
- written metadata satisfies `normalize_feature_columns(...)`.

### Slice 2: Controller/Widget Metadata State

Add a small metadata-state check for the selected table and feature key.

The widget should know whether the selected feature matrix is:

- missing from `.obsm`;
- present but unregistered;
- registered and valid;
- registered but mismatched with the live matrix shape.

Use this state to drive the button, warning text, and tooltip.

### Slice 3: Register Feature Matrix Button

Add the UI button and wire it to the core helper.

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

### Slice 4: Persist Feature Matrix Metadata

Include `feature_matrices` in the selected table-state write path.

Tests:

- registering custom metadata marks the backed table dirty;
- `Write Table State` writes `uns["feature_matrices"]` to zarr;
- reload preserves the registered custom metadata.

### Slice 5: Optional Explicit Column-Name UI

Only if needed later, add a small advanced dialog for editing feature column
names before registration. The first implementation can use deterministic names
without a dialog, because this keeps the feature small and testable.
