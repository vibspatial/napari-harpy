# Feature Extraction Widget Roadmap

## Goal

Add a second napari widget to `napari-harpy`:

- `Feature Extraction`
- `Object Classification`

The new widget should let the user:

1. select a labels element discovered from viewer-linked `SpatialData` metadata
2. optionally select an image element from the same `SpatialData` and a compatible coordinate system
3. choose which feature families to calculate
4. choose an output `.obsm` key
5. write the calculated features into the selected `AnnData` table

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

The important implication is that viewer layers should act as entry points into
viewer-linked `SpatialData` objects, not as the complete list of selectable
elements. Once we know which `SpatialData` object and coordinate system the
user is bound to, we can discover sibling labels and image elements directly
from `sdata`.

Recommended discovery model:

- both widgets scan labels from the current viewer-linked `SpatialData` objects
- selecting a labels element binds us to:
  - a specific `SpatialData` object
  - the coordinate system represented by the selected labels layer when it is loaded
- the feature-extraction image dropdown should then list image elements from:
  - the same `SpatialData` object
  - coordinate systems compatible with the selected labels layer
- if the selected labels or image element is not currently loaded in the viewer,
  the widget should still allow selection but surface a warning to the user

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

### 3. Harpy now exposes the right high-level abstraction for this widget

Available Harpy APIs:

- `harpy.tb.add_feature_matrix(...)`
- `harpy.utils.RasterAggregator(...)`
- `harpy.table._regionprops._calculate_regionprop_features(...)`
- `harpy.table._allocation_intensity.allocate_intensity(...)`
- `harpy.table._regionprops.add_regionprops(...)`
- `harpy.table._table.add_table_layer(...)`

Important behavior:

- `harpy.tb.add_feature_matrix(...)` computes requested intensity and/or morphology features and writes the numeric result into `adata.obsm[feature_key]`.
- `harpy.tb.add_feature_matrix(...)` stores companion metadata in `adata.uns[feature_matrices_key][feature_key]`.
- `harpy.tb.add_feature_matrix(...)` aligns rows by `(region_key, instance_key)`, not by table row order.
- `harpy.tb.add_feature_matrix(...)` updates existing tables in place and requires `overwrite_feature_key=True` before replacing an existing `.obsm[feature_key]`.
- on backed `SpatialData`, `harpy.tb.add_feature_matrix(...)` persists the changed `.obsm` key and companion metadata with targeted `anndata.io.write_elem(...)` calls and reconsolidates zarr metadata.
- `RasterAggregator` remains the underlying intensity primitive.
- `_calculate_regionprop_features(...)` remains the underlying morphology primitive.

Conclusion:

- napari-harpy should call `hp.tb.add_feature_matrix(...)`, not reimplement feature extraction, row alignment, or backed persistence locally.
- the napari side should own viewer-driven selection, validation, worker orchestration, stale-job/UI-state handling, and cross-widget refresh behavior.
- napari-harpy should follow Harpy's default metadata namespace: `uns["feature_matrices"]`.

### 4. Harpy still has strict array-shape expectations internally, but the widget no longer needs to manage them directly

`RasterAggregator` expects:

- labels mask shape: `(z, y, x)`
- image shape: `(c, z, y, x)`
- matching spatial shape
- matching spatial chunking

So for 2D data Harpy needs to promote:

- labels: `(y, x)` -> `(1, y, x)`
- image: `(c, y, x)` -> `(c, 1, y, x)`

This is the same pattern Harpy already uses internally.

The important implication for napari-harpy is:

- the widget/controller should pass canonical `sdata` element names and the chosen coordinate system into `hp.tb.add_feature_matrix(...)`
- napari-harpy does not need its own array-shape normalization path for MVP
- `chunks` remains only an optional performance knob; Harpy already warns that rechunking on disk ahead of time is usually preferable

### 5. Morphology features are not fully out-of-core

`_calculate_regionprop_features(...)` uses `skimage.measure.regionprops_table(...)` on in-memory masks.

That means:

- morphology calculations are fine for typical label layers
- very large labels layers may need memory warnings
- some properties such as `perimeter` are not supported for 3D labels

So the widget should still separate:

- intensity features
- morphology features

and validate the chosen set against the selected labels dimensionality, even though Harpy owns the actual computation.

### 6. Creating a new linked table is feasible, but we should defer it from MVP

If we later want table creation in the feature widget, Harpy can create it for us through `hp.tb.add_feature_matrix(...)`:

1. pass `table_layer=None`
2. provide `output_layer`
3. let Harpy create the annotated table and attach it to `sdata`

That path is useful, but it adds widget state and overwrite branches that are not needed for the first end-to-end feature-to-classifier workflow. For MVP, we should assume the selected labels layer is already linked to a table.

### 7. Prefer an array-first feature-matrix schema

For long-term flexibility, the feature extractor should treat `.obsm[key]` as an array-like matrix store, not as a pandas-backed table abstraction.

Recommended schema:

- `adata.obsm[feature_key]`: `np.ndarray | scipy.sparse.spmatrix | dask.array.Array`
- `adata.uns["feature_matrices"][feature_key]`:
  - `feature_columns`
  - `backend`
  - `dtype`
  - `source_label`
  - `source_image`
  - `coordinate_system`
  - `schema_version`
  - `features`

Why this is preferable:

- it avoids depending on DataFrame-specific semantics such as `.columns`, `.index`, and pandas casting behavior
- it keeps the stored feature representation aligned with the conceptual model of `.obsm` as an `n_obs x n_features` matrix
- it leaves a clean path for sparse and dask-backed matrices later

For the roadmap below, assume that feature names and provenance are read from `uns["feature_matrices"]`, not from a `DataFrame` stored in `.obsm`. In Harpy's current implementation that means reading `feature_columns`, not `columns`.

## Recommended MVP Decisions

### 1. Use a separate widget and controller

Add:

- `src/napari_harpy/widgets/_feature_extraction_widget.py`
- `src/napari_harpy/_feature_extraction.py`

The widget owns UI state.
The controller owns validation, background jobs, the Harpy call boundary, and widget-facing status/refresh behavior.

### 2. Resolve from viewer metadata, but compute from `sdata`

The widgets should use viewer layers only to resolve:

- which `SpatialData` objects are currently linked into the viewer
- the selected `SpatialData`
- the selected labels element name
- the active coordinate system bound to the selected labels layer when it is loaded
- whether the selected labels/image element is currently loaded in the viewer

The actual computation should then read from the underlying `sdata` elements,
not from `napari` layer buffers. That keeps the implementation aligned with
Harpy and avoids display-level downsampling concerns.

For selection flow, this means:

- labels should be discovered dataset-centrically from viewer-linked `SpatialData` objects
- images should also be discovered dataset-centrically from `sdata.images`
- image choices should be filtered after labels selection to the same `SpatialData` and compatible coordinate systems
- loaded viewer layers are still useful for UX warnings and activation, but they should not define the complete set of selectable sibling elements

### 3. Write an array-like feature matrix into `.obsm`

Recommended storage format:

- `adata.obsm[feature_key] = feature_matrix`

where `feature_matrix` is one of:

- `np.ndarray`
- `scipy.sparse.spmatrix`
- `dask.array.Array`

And store feature metadata separately in:

- `adata.uns["feature_matrices"][feature_key]`

with at least:

- `feature_columns`
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

Implementation note:

- napari-harpy should reach this storage format by calling `hp.tb.add_feature_matrix(...)`
- metadata should be read from Harpy's default `feature_matrices` namespace

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

Harpy's current `add_feature_matrix(...)` implementation already does this alignment for us. It is still worth calling out explicitly because the widget/controller should assume that rule when validating table selections and explaining results.

This keeps the widget safe when:

- the table contains multiple annotated regions
- the table rows were filtered or reordered earlier

### 5. Preserve non-selected rows when possible

Recommended MVP behavior when writing into an existing table:

- for rows belonging to the selected labels region:
  - overwrite with the newly computed feature values
- for rows outside the selected region:
  - preserve existing values if `.obsm[feature_key]` already exists, `overwrite_feature_key=True`, and the stored feature metadata reports the same columns
  - otherwise fill with `NaN`

This gives us a practical path for multi-region tables without requiring that all regions be recomputed together every time. Harpy's current `add_feature_matrix(...)` implementation already follows this rule.

### 6. Future new-table path

If we later expose table creation in the widget, use defaults aligned with `napari-spatialdata` conventions:

- `region_key = "region"`
- `instance_key = "instance_id"`

When calling Harpy in new-table mode, it should then:

- include one row per nonzero label id
- set `obs[region_key] = selected_label_name`
- set `obs[instance_key] = instance_id`
- own obs-index creation internally

## Proposed User Flow

### Selection flow

1. both widgets scan the current viewer for layers carrying `metadata["sdata"]`
2. from those viewer-linked `SpatialData` objects, expose labels elements as selectable segmentations
3. when the user selects a segmentation:
   - bind to the selected `SpatialData`
   - resolve the coordinate system from the loaded labels layer when available
4. in the feature-extraction widget, populate the image dropdown from `sdata.images`
   filtered to image elements that are selectable for that labels choice:
   - same `SpatialData`
   - shared coordinate system
5. if the chosen labels or image element is not currently loaded in the viewer,
   keep it selectable but show a warning similar to the current labels-layer
   feedback path

### Inputs

- `Segmentation`
- `Image` (optional, filtered to the same `SpatialData` and compatible coordinate systems)
- `Table` (required, limited to tables that annotate the selected segmentation)
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
- warning when the selected labels/image element is not currently loaded in the viewer
- warning when the selected segmentation has no linked annotation table
- calculation status
- success message including:
  - table name
  - output `.obsm` key
  - number of objects
  - number of generated features

## Validation Rules

Before calculation, require:

- a selected labels element
- a selected existing annotation table linked to that labels element
- at least one selected feature
- a non-empty output `.obsm` key

If any intensity feature is selected, also require:

- a selected image element
- same `SpatialData` object as the labels element
- compatible coordinate system / transform
- same spatial shape after resolving the selected `sdata` elements

Loaded napari layers are not required for computation. If the selected labels or
image element is not currently loaded in the viewer, surface a warning rather
than treating that as a hard validation error.

If an existing table is selected:

- validate table binding with the current helper:
  - `validate_table_binding(...)`

If the output `.obsm` key already exists:

- require explicit overwrite confirmation before replacing it
- once `overwrite_feature_key=True`, let Harpy decide whether non-selected rows can be preserved or need to be filled with `NaN` based on schema compatibility

## Proposed Implementation Plan

### Phase 1: Extend SpatialData discovery helpers

Align labels and image discovery helpers in `src/napari_harpy/_spatialdata.py`
around viewer-linked `SpatialData` datasets rather than only currently loaded
layers.

Suggested additions:

- [x] `SpatialDataImageOption`
- [x] `SpatialDataViewerBinding.get_image_options()`
- [x] `SpatialDataViewerBinding.get_image_layer(...)`
- [x] `get_spatialdata_image_options(...)`

Rules:

- [x] labels discovery should continue to start from viewer-linked `SpatialData` objects
- [x] image discovery should expose `sdata.images[...]` entries from those same viewer-linked `SpatialData` objects
- [x] once a labels element is selected, image options should be filtered to:
  - [x] the same `SpatialData`
  - [x] coordinate systems compatible with the selected labels element
- [x] show dataset names when multiple `SpatialData` objects are present
- [x] for the feature extraction widget it is not necessary that the spatial elements (labels/images) are loaded in the viewer, so we should not raise a warning if they are not loaded.

### Phase 2: Build the feature-extraction controller

Create `src/napari_harpy/_feature_extraction.py`.

Responsibilities:

- [ ] bind the selected labels, optional image, target table, requested feature set, and output key
- [ ] validate the current bound selection against the authoritative `SpatialData` state
- [ ] prepare `hp.tb.add_feature_matrix(...)` call arguments on the main thread
- [ ] run the Harpy feature-extraction call in a worker
- [ ] accept the successful result on the main thread and treat the updated `sdata.tables[...]` entry as authoritative
- [ ] notify the widget when table state changed so dependent UI can refresh and persistence state can stay coherent
- [ ] expose status strings and simple run-state flags for the widget

Recommended structure:

- [ ] `FeatureExtractionJob`
- [ ] `FeatureExtractionResult`
- [ ] `FeatureExtractionController`

This should mirror the current classifier design:

- [ ] explicit, passive bind step that does not compute until the user clicks `Calculate`
- [ ] worker-based long-running work
- [ ] stale-job protection if the selection changes mid-run
- [ ] authoritative writes back into `sdata[table_name]`, not napari layer metadata

Important differences from the classifier:

- [ ] feature extraction is a one-shot calculate flow, so we do not need classifier-style dirty/debounce semantics
- [ ] for MVP, the controller only needs to support updating an existing linked table
- [ ] because Harpy writes directly into `sdata`, we should avoid overlapping feature-extraction runs against the same dataset and treat stale-job protection primarily as a UI/state-adoption guard

### Phase 3: Integrate Harpy feature calculation

Use `hp.tb.add_feature_matrix(...)` as the authoritative compute-and-write path.

#### Call contract

- [ ] resolve canonical labels, image, and table names from the bound selection
- [ ] pass `feature_key`, `features`, `to_coordinate_system`, and the current labels/image/table selection into `hp.tb.add_feature_matrix(...)`
- [ ] use Harpy's default `feature_matrices` metadata namespace
- [ ] pass `overwrite_feature_key` when replacing an existing `.obsm` key in an existing table
- [ ] keep `channels`, `chunks`, and `run_on_gpu` as optional later extensions unless the widget exposes them in MVP

#### What Harpy already owns

- [ ] intensity feature extraction via `RasterAggregator(...)`
- [ ] morphology feature extraction via `_calculate_regionprop_features(...)`
- [ ] validation that intensity features require an image input
- [ ] alignment by `(region_key, instance_key)`
- [ ] writing the matrix into `.obsm[feature_key]`
- [ ] writing companion metadata into `uns[feature_matrices_key][feature_key]`
- [ ] backed write-through persistence with targeted `anndata.io.write_elem(...)`

#### napari-harpy responsibilities around the call

- [ ] surface clear validation and overwrite UX before dispatching the Harpy call
- [ ] serialize runs so we do not have concurrent `add_feature_matrix(...)` mutations against the same `sdata`
- [ ] translate Harpy exceptions into widget status strings
- [ ] notify table-state change and refresh widget choices after a successful call

### Phase 4: Create the widget

Create `src/napari_harpy/widgets/_feature_extraction_widget.py`.

The widget should look and behave like the current object classification widget:

- [ ] same overall styling
- [ ] same selector layout
- [ ] same status-card pattern
- [ ] same explicit button-driven flow

Suggested controls:

- [ ] labels combo
- [ ] image combo
- [ ] table combo
- [ ] output-key line edit
- [ ] grouped feature checkboxes
- [ ] calculate button
- [ ] status labels

### Phase 5: Add the napari manifest entry

Update `src/napari_harpy/napari.yaml` to contribute:

- [ ] `Feature Extraction`
- [ ] `Object Classification`

This keeps the plugin structure aligned with the Phase 2 plan.

### Phase 6: Integrate with the object-classification workflow

Minimum viable integration:

- [ ] treat the table updated by `hp.tb.add_feature_matrix(...)` in `sdata.tables[table_name]` as authoritative
- [ ] emit a small shared table-changed signal so the object-classification widget can refresh from the same in-memory table state
- [ ] optionally refresh loaded labels layer metadata as a best-effort compatibility step:
  - [ ] `refresh_layer_table_metadata(...)`
- [ ] rely on Harpy's write-through behavior on backed datasets rather than duplicating feature persistence in napari-harpy
- [ ] tell the user that the feature key is now available for object classification

Important nuance:

- [ ] the authoritative source of truth is `sdata[table_name]`; napari layer metadata is only a cache
- [ ] the object classification widget should refresh feature-key choices from shared in-memory table state, not rely on a disk reload as the primary sync path
- [ ] reload from disk remains a fallback when the widget needs to resync from persisted state

Nice-to-have follow-up:

- [ ] make the shared table-changed signal general enough that both widgets can respond to future table updates, not only feature extraction

### Follow-up ticket: stop classifier retraining from the widget

The current `ObjectClassificationWidget` supports retraining, but not user-initiated cancellation once training has started.

Add a follow-up ticket to:

- [ ] expose a `Stop training` action while classifier retraining is running
- [ ] cancel or ignore the active retraining job safely when the user stops it
- [ ] keep the last successfully trained classifier active if cancellation happens mid-run
- [ ] restore the widget controls and status message to a clear cancelled state
- [ ] allow retraining to be started again immediately after cancellation

### Phase 7: Persistence semantics

Feature extraction should follow the same authority model as the rest of the repo:

- [ ] authoritative in-memory table: `sdata.tables[table_name]`
- [ ] on backed datasets, write through to zarr after a successful feature-extraction run
- [ ] on unbacked datasets, keep the result in memory only

MVP recommendation:

- [ ] feature extraction always updates the in-memory table via `hp.tb.add_feature_matrix(...)`
- [ ] if the selected dataset is backed, Harpy then persists the new `.obsm[key]` entry and `uns["feature_matrices"][key]`
- [ ] if the selected dataset is not backed, the widget clearly reports that the new feature matrix exists only in memory

Open integration detail:

- [ ] the object classification widget needs a lightweight refresh hook so a newly written feature key becomes selectable without forcing a full reload

Good follow-up options:

- [ ] keep reload available as a fallback for recovering persisted state
- [ ] avoid introducing a second napari-harpy-specific feature-persistence path unless we later need centralization beyond Harpy's write-through behavior
- [ ] use a shared table-state signal so both widgets can react to table updates in the same session

### Phase 8: Support the "No Table Linked" branch

Once the existing-table MVP is settled, add a second branch for segmentations
that do not yet annotate any table.

Goals:

- [ ] detect when the selected segmentation has no linked annotation table
- [ ] surface a dedicated "no table linked" UX state instead of only a generic validation block
- [ ] let the user choose a new table name when they want to continue from that state
- [ ] call `hp.tb.add_feature_matrix(...)` in new-table mode by passing `table_layer=None`
- [ ] pass the user-provided table name as `output_layer`
- [ ] keep `overwrite_output_layer=False` by default and require explicit confirmation before replacing an existing table layer
- [ ] after successful creation, refresh table choices so the new table becomes the selected authoritative table immediately
- [ ] emit the shared table-changed signal so the object-classification widget can discover the new table and feature key in the same session

Validation rules for this branch:

- [ ] require a non-empty new table name
- [ ] reject collisions with existing table-layer names unless overwrite was explicitly enabled
- [ ] continue to require an image only when intensity features are selected
- [ ] continue to apply the same labels/image coordinate-system validation as the existing-table path

Why this is a good follow-up phase:

- [ ] it is the most natural expansion of the MVP without changing the core compute path
- [ ] Harpy already owns the underlying table-creation machinery, so napari-harpy mainly needs extra UI, validation, and refresh handling
- [ ] it keeps the initial implementation focused on the feature-to-classifier workflow while still giving us a clear next step for datasets that start from labels alone

## Testing Plan

Add `tests/test_feature_extraction.py`.

Minimum test coverage:

- discovers image and labels options from viewer metadata
- requires image selection only when intensity features are chosen
- requires that the selected labels layer is already linked to a table
- computes morphology-only features into `.obsm[key]`
- computes intensity + morphology features into `.obsm[key]`
- writes feature metadata with stable column names into `uns["feature_matrices"]`
- updates an existing table by `region_key` and `instance_key`
- preserves non-selected rows when updating one region of a multi-region table with `overwrite_feature_key=True`
- raises cleanly when `feature_key` already exists and overwrite was not explicitly allowed
- on backed datasets, Harpy persists the new `.obsm` key and companion metadata immediately after calculation
- on unbacked datasets, keeps the new feature matrix in memory without attempting persistence
- refreshes the object-classification widget from shared in-memory table state after feature extraction
- refreshes layer metadata so the updated table is visible through napari-spatialdata
- persists and reloads the new `.obsm` key and companion `uns["feature_matrices"]` metadata through backed zarr

## Recommended Initial Scope

To keep the first implementation tractable, I would intentionally limit MVP to:

- one selected labels layer
- one optional selected image layer
- one selected existing table
- one user-provided output `.obsm` key
- a small curated feature set
- CPU execution by default
- a lightweight same-session refresh path in the object-classification widget after calculation

I would explicitly defer:

- ROI-aware feature extraction
- channel subsetting UI
- GPU toggle UI
- automatic cross-widget refresh
- deep feature extraction
- batch computation across multiple labels layers
- create-a-new-table flow from the feature widget, to be handled in Phase 8

## Summary Recommendation

The cleanest implementation path is:

1. add a dedicated `FeatureExtractionWidget`
2. extend the existing viewer-binding helpers with image discovery
3. call `hp.tb.add_feature_matrix(...)` from the controller rather than reimplementing feature extraction in napari-harpy
4. treat the updated `sdata.tables[table_name]` entry as authoritative and read companion metadata from `uns["feature_matrices"]`
5. require that the selected labels layer is already linked to an annotation table for MVP
6. reuse the current object-classification widget as the consumer of the new feature matrix

That gives us a solid MVP without duplicating logic that Harpy now already owns.
