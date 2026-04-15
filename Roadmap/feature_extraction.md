# Feature Extraction Widget Roadmap

## Goal

Add a second napari widget to `napari-harpy`:

- `Feature Extraction`
- `Object Classification`

The new widget should let the user:

1. select a labels layer loaded through `napari-spatialdata`
2. optionally select an image layer from the same `SpatialData`
3. choose which feature families to calculate
4. choose an output `.obsm` key
5. write the calculated features into the selected `AnnData` table
6. create a new linked `AnnData` table when no suitable table is selected

The end state we want is simple:

- the feature widget computes features into `sdata.tables[table_name].obsm[feature_key]`
- the object classification widget can then use that `feature_key` immediately or after a refresh

## Versions Investigated

The findings below were checked against the local environment used by this repo:

- `harpy-analysis 0.3.6`
- `napari-spatialdata 0.7.0`
- `spatialdata 0.7.2`
- `anndata 0.12.10`
- `napari 0.7.0`

## Findings

### 1. `napari-spatialdata` gives us the right metadata on both labels and image layers

From the installed `napari-spatialdata` source:

- labels layers carry:
  - `metadata["sdata"]`
  - `metadata["name"]`
  - `metadata["_current_cs"]`
  - `metadata["adata"]`
  - `metadata["table_names"]`
  - `metadata["region_key"]`
  - `metadata["instance_key"]`
  - `metadata["indices"]`
- image layers carry:
  - `metadata["sdata"]`
  - `metadata["name"]`
  - `metadata["_current_cs"]`

This is enough to build the new widget from viewer state without asking the user to browse the `SpatialData` object manually.

### 2. The current object-classification path already matches the intended integration target

The current classifier flow reads:

- the selected table from `sdata[table_name]`
- feature matrix keys from `table.obsm.keys()`
- the selected feature matrix from `table.obsm[selected_feature_key]`

The classifier only requires that the selected `.obsm` entry:

- is 2-dimensional
- has `n_obs` rows
- can be converted to a numeric array

That means the feature widget does not need any classifier-specific serialization format. A standard `.obsm` matrix with `n_obs` rows is enough, and feature names/provenance can live in companion metadata in `.uns`.

### 3. The Harpy high-level wrappers are close, but not quite the right abstraction for this widget

Available Harpy APIs:

- `harpy.utils.RasterAggregator(...)`
- `harpy.table._regionprops._calculate_regionprop_features(...)`
- `harpy.table._allocation_intensity.allocate_intensity(...)`
- `harpy.table._regionprops.add_regionprops(...)`
- `harpy.table._table.add_table_layer(...)`

Important behavior:

- `RasterAggregator` is the right low-level primitive for intensity statistics.
- `_calculate_regionprop_features` is the right low-level primitive for morphology features.
- `allocate_intensity(...)` writes to `.X` and `.obs`, not `.obsm`.
- `add_regionprops(...)` writes to `.obs`, not `.obsm`.

Conclusion:

- for this widget, we should use the low-level Harpy calculators
- we should own the final merge into `.obsm`

### 4. Harpy has strict array-shape expectations we should design around

`RasterAggregator` expects:

- labels mask shape: `(z, y, x)`
- image shape: `(c, z, y, x)`
- matching spatial shape
- matching spatial chunking

So for 2D data we will need to promote:

- labels: `(y, x)` -> `(1, y, x)`
- image: `(c, y, x)` -> `(c, 1, y, x)`

This is the same pattern Harpy already uses internally in `allocate_intensity(...)`.

### 5. Morphology features are not fully out-of-core

`_calculate_regionprop_features(...)` uses `skimage.measure.regionprops_table(...)` on in-memory masks.

That means:

- morphology calculations are fine for typical label layers
- very large labels layers may need memory warnings
- some properties such as `perimeter` are not supported for 3D labels

So the widget should separate:

- intensity features
- morphology features

and validate the chosen set against the selected labels dimensionality.

### 6. Creating a new linked table is feasible with the current libraries

If the user has no selected table, or wants a new one, we can create it ourselves:

1. build an `AnnData`
2. add `region_key` and `instance_key` to `adata.obs`
3. parse it with `TableModel.parse(...)`
4. attach it with `harpy.table._table.add_table_layer(...)`

This is especially useful because `add_table_layer(...)` also handles backed `SpatialData`.

### 7. Prefer an array-first feature-matrix schema

For long-term flexibility, the feature extractor should treat `.obsm[key]` as an array-like matrix store, not as a pandas-backed table abstraction.

Recommended schema:

- `adata.obsm[feature_key]`: `np.ndarray | scipy.sparse.spmatrix | dask.array.Array`
- `adata.uns["harpy_feature_matrices"][feature_key]`:
  - `columns`
  - `backend`
  - `dtype`
  - `source_label`
  - `source_image`
  - `coordinate_system`
  - `schema_version`

Why this is preferable:

- it avoids depending on DataFrame-specific semantics such as `.columns`, `.index`, and pandas casting behavior
- it keeps the stored feature representation aligned with the conceptual model of `.obsm` as an `n_obs x n_features` matrix
- it leaves a clean path for sparse and dask-backed matrices later

For the roadmap below, assume that feature names and provenance are read from `uns["harpy_feature_matrices"]`, not from a `DataFrame` stored in `.obsm`.

## Recommended MVP Decisions

### 1. Use a separate widget and controller

Add:

- `src/napari_harpy/widgets/_feature_extraction_widget.py`
- `src/napari_harpy/_feature_extraction.py`

The widget owns UI state.
The controller owns validation, background jobs, feature calculation, and the merge into the selected table.

### 2. Resolve from viewer metadata, but compute from `sdata`

The widget should use viewer layers only to resolve:

- the selected `SpatialData`
- the selected labels element name
- the selected image element name
- the active coordinate system

The actual computation should then read from the underlying `sdata` elements, not from `napari` layer buffers. That keeps the implementation aligned with Harpy and avoids display-level downsampling concerns.

### 3. Write an array-like feature matrix into `.obsm`

Recommended storage format:

- `adata.obsm[feature_key] = feature_matrix`

where `feature_matrix` is one of:

- `np.ndarray`
- `scipy.sparse.spmatrix`
- `dask.array.Array`

And store feature metadata separately in:

- `adata.uns["harpy_feature_matrices"][feature_key]`

with at least:

- `columns`
- `schema_version`

and preferably also:

- `backend`
- `dtype`
- `source_label`
- `source_image`
- `coordinate_system`

Expected feature names remain stable, for example:

- `mean__channel_name`
- `var__channel_name`
- `min__channel_name`
- `max__channel_name`
- `area`
- `eccentricity`
- `major_axis_length`

Those names should be stored in the metadata schema, not inferred from DataFrame columns.

### 4. Merge by `(region_key, instance_key)`, not by row order

This is the most important data-model rule for the feature widget.

We should not assume:

- selected labels row order == `adata.obs` row order
- region-only row order == full-table row order

Instead:

1. compute features keyed by label id
2. build a region-aware merge key from:
   - `region_key`
   - `instance_key`
3. align the result onto the full selected table
4. write the full aligned matrix into `.obsm[feature_key]`

This keeps the widget safe when:

- the table contains multiple annotated regions
- the table rows were filtered or reordered earlier

### 5. Preserve non-selected rows when possible

Recommended MVP behavior when writing into an existing table:

- for rows belonging to the selected labels region:
  - overwrite with the newly computed feature values
- for rows outside the selected region:
  - preserve existing values if `.obsm[feature_key]` already exists and the stored feature metadata reports the same columns
  - otherwise fill with `NaN`

This gives us a practical path for multi-region tables without requiring that all regions be recomputed together every time.

### 6. Default table keys for a new table

If no table exists yet for the selected labels layer, use defaults aligned with `napari-spatialdata` conventions:

- `region_key = "region"`
- `instance_key = "instance_id"`

For the new table rows:

- include one row per nonzero label id
- set `obs[region_key] = selected_label_name`
- set `obs[instance_key] = instance_id`
- use a stable obs index such as `f"{selected_label_name}:{instance_id}"`

## Proposed User Flow

### Inputs

- `Segmentation`
- `Image` (optional)
- `Table`
- `New table name` when creating a table
- `Output feature key`

### Feature groups

- `Intensity features`
  - requires labels + image
  - initial MVP:
    - `mean`
    - `var`
    - `min`
    - `max`
- `Morphology features`
  - requires labels only
  - initial MVP:
    - `area`
    - `eccentricity`
    - `major_axis_length`
    - `minor_axis_length`
    - `perimeter`
    - `equivalent_diameter`
    - `centroid` # -> calculate this via harpy rasteraggregator.

### Action

- `Calculate`

### UI terminology cleanup

As part of the multi-widget Phase 2 UI cleanup, rename the existing object-classification actions:

- `Retrain` -> `Train`
- `Rescan Viewer` -> `Scan Viewer`

Use the new labels consistently in:

- the object classification widget
- the feature extraction widget if it exposes the same viewer-refresh action
- the README, debug script, and any user-facing status or tooltip text

### Feedback

- validation banner
- calculation status
- success message including:
  - table name
  - output `.obsm` key
  - number of objects
  - number of generated features

## Validation Rules

Before calculation, require:

- a selected labels layer
- at least one selected feature
- a non-empty output `.obsm` key

If any intensity feature is selected, also require:

- a selected image layer
- same `SpatialData` object as the labels layer
- same spatial shape
- compatible transform / coordinate system

If an existing table is selected:

- validate table binding with the current helper:
  - `validate_table_binding(...)`

If a new table is being created:

- require a non-empty new table name
- reject table-name collisions unless the user explicitly chose overwrite behavior

If the output `.obsm` key already exists:

- if the stored feature metadata columns match, allow in-place replacement of the selected region rows
- if the stored feature metadata columns differ, require explicit overwrite confirmation or a new key

## Proposed Implementation Plan

### Phase 1: Extend SpatialData discovery helpers

Add image discovery helpers alongside the existing labels helpers in `src/napari_harpy/_spatialdata.py`.

Suggested additions:

- `SpatialDataImageOption`
- `SpatialDataViewerBinding.get_image_options()`
- `SpatialDataViewerBinding.get_image_layer(...)`
- `get_spatialdata_image_options(...)`

Rules:

- only expose image layers whose metadata links back to a viewer-loaded `SpatialData`
- show dataset names when multiple `SpatialData` objects are present
- keep the image selection scoped to the same `SpatialData` as the selected labels layer

### Phase 2: Build the feature-extraction controller

Create `src/napari_harpy/_feature_extraction.py`.

Responsibilities:

- bind the selected labels, image, table, and output key
- validate the current selection
- prepare calculation jobs on the main thread
- run feature calculation in a worker
- merge results into the selected table
- expose status strings for the widget

Recommended structure:

- `FeatureExtractionJob`
- `FeatureExtractionResult`
- `FeatureExtractionController`

This should mirror the current classifier design:

- explicit bind step
- worker-based long-running work
- stale-job protection if the selection changes mid-run

### Phase 3: Implement feature computation

Use Harpy low-level primitives directly.

#### Intensity path

1. resolve canonical `sdata.images[image_name]` and `sdata.labels[label_name]`
2. rechunk when needed (`sdata.images[image_name]` and `sdata.labels[label_name]` should have same chunk size).
3. normalize 2D data to singleton-z arrays when needed
4. build `RasterAggregator(...)`
5. compute the selected stats for nonzero labels only
6. rename columns to stable feature names

#### Morphology path

1. materialize the labels mask into memory
2. call `_calculate_regionprop_features(...)`
3. keep only the user-selected morphology properties
4. rename columns to stable output names

#### Assembly

1. outer-join the intensity and morphology feature frames on `instance_key`
2. add the selected `region_key`
3. align onto the target table rows
4. convert the aligned features to the chosen array-like backend
5. write the matrix into `.obsm[feature_key]`
6. write companion metadata into `uns["harpy_feature_matrices"][feature_key]`

### Phase 4: Create the widget

Create `src/napari_harpy/widgets/_feature_extraction_widget.py`.

The widget should look and behave like the current object classification widget:

- same overall styling
- same selector layout
- same status-card pattern
- same explicit button-driven flow

Suggested controls:

- labels combo
- image combo
- table combo
- optional new-table name line edit
- output-key line edit
- grouped feature checkboxes
- calculate button
- status labels

### Phase 5: Add the napari manifest entry

Update `src/napari_harpy/napari.yaml` to contribute:

- `Feature Extraction`
- `Object Classification`

This keeps the plugin structure aligned with the Phase 2 plan.

### Phase 6: Integrate with the object-classification workflow

Minimum viable integration:

1. update the authoritative table in `sdata.tables[table_name]`
2. refresh the loaded labels layer metadata with:
   - `refresh_layer_table_metadata(...)`
3. tell the user that the feature key is now available for object classification

Important nuance:

- the current object classification widget only refreshes its feature-key dropdown when it rebinds or rescans
- so MVP integration can rely on:
  - reopening the classification widget
  - or clicking `Scan Viewer`

Nice-to-have follow-up:

- add a small plugin-local signal bus so both widgets can refresh automatically when a table changes

### Follow-up ticket: stop classifier retraining from the widget

The current `ObjectClassificationWidget` supports retraining, but not user-initiated cancellation once training has started.

Add a follow-up ticket to:

- expose a `Stop training` action while classifier retraining is running
- cancel or ignore the active retraining job safely when the user stops it
- keep the last successfully trained classifier active if cancellation happens mid-run
- restore the widget controls and status message to a clear cancelled state
- allow retraining to be started again immediately after cancellation

### Phase 7: Persistence semantics

Feature extraction should follow the same authority model as the rest of the repo:

- authoritative in-memory table: `sdata.tables[table_name]`
- optional later write to backed zarr

MVP recommendation:

- feature extraction updates the in-memory table only
- the widget clearly reports that the new feature matrix is not yet written to disk

Open integration detail:

- if the object classification widget is open at the same time, its dirty-state indicator will not automatically know that features changed

Good follow-up options:

- give the feature widget its own `Write` button backed by `PersistenceController`
- or add a shared table-state signal so both widgets can mark the same table dirty

## Testing Plan

Add `tests/test_feature_extraction.py`.

Minimum test coverage:

- discovers image and labels options from viewer metadata
- requires image selection only when intensity features are chosen
- computes morphology-only features into `.obsm[key]`
- computes intensity + morphology features into `.obsm[key]`
- writes feature metadata with stable column names into `uns["harpy_feature_matrices"]`
- creates a new linked table when none is selected
- merges correctly into an existing table by `region_key` and `instance_key`
- preserves non-selected rows when updating one region of a multi-region table
- refreshes layer metadata so the updated table is visible through napari-spatialdata
- persists and reloads the new `.obsm` key and companion `uns["harpy_feature_matrices"]` metadata through backed zarr

## Recommended Initial Scope

To keep the first implementation tractable, I would intentionally limit MVP to:

- one selected labels layer
- one optional selected image layer
- one selected or newly created table
- one user-provided output `.obsm` key
- a small curated feature set
- CPU execution by default
- manual refresh in the object-classification widget after calculation

I would explicitly defer:

- ROI-aware feature extraction
- channel subsetting UI
- GPU toggle UI
- automatic cross-widget refresh
- deep feature extraction
- batch computation across multiple labels layers

## Summary Recommendation

The cleanest implementation path is:

1. add a dedicated `FeatureExtractionWidget`
2. extend the existing viewer-binding helpers with image discovery
3. use Harpy low-level calculators, not Harpy's higher-level table writers
4. merge feature results into `.obsm[key]` on the authoritative in-memory table and record companion metadata in `uns`
5. create a new linked table with `add_table_layer(...)` when no table exists
6. reuse the current object-classification widget as the consumer of the new feature matrix

That gives us a solid MVP without fighting either `napari-spatialdata` or Harpy's current table abstractions.
