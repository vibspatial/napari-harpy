# napari-harpy Roadmap

## Current Status

`napari-harpy` now has a working MVP for interactive object classification on `SpatialData`
datasets loaded through `napari-spatialdata`.

Current workflow in the repository:

- discover viewer-linked `SpatialData` datasets and segmentation layers
- select a segmentation, linked annotation table, and `adata.obsm` feature matrix
- validate `TableModel` linkage metadata and available feature matrices
- pick segmented objects in single-scale and multiscale labels layers
- store manual annotations in `adata.obs["user_class"]`
- retrain a background `RandomForestClassifier` with debounce and stale-job protection
- write predictions into:
  - `adata.obs["pred_class"]`
  - `adata.obs["pred_confidence"]`
  - `adata.uns["classifier_config"]`
- recolor the active labels layer by:
  - `user_class`
  - `pred_class`
  - `pred_confidence`
- explicitly write the current table state back to the backed zarr store
- explicitly reload on-disk table state back into the current in-memory session
- protect reload against unsynced edits and late classifier worker results

Repository verification at the time of this update:

- `source .venv/bin/activate && pytest -q` passes
- current result: `80 passed` on April 14, 2026

## What Is Still Deferred

The main planned feature that is still not implemented is ROI handling:

- ROI-based subsetting with `spatialdata.bounding_box_query`
- ROI-restricted annotation behavior
- ROI-restricted training and prediction behavior
- ROI-aware persistence semantics

## Implemented Architecture

The repository now centers around these components:

- `HarpyWidget`: UI state, selection flow, and user actions
- `SpatialDataAdapter`: table discovery, validation, and linkage metadata
- `SpatialDataViewerBinding`: napari layer lookup and viewer compatibility helpers
- `AnnotationController`: pick-based selection and `user_class` writes
- `ClassifierController`: debounced async `RandomForestClassifier` retraining and prediction writes
- `ViewerStylingController`: `Color by` styling and layer feature refresh
- `PersistenceController`: manual write/reload of the selected table state to and from backed zarr

## Phase Status

### Phase 0: Project setup and plugin skeleton

Status: complete.

Implemented:

- installable `src/napari_harpy` package
- npe2 manifest in `src/napari_harpy/napari.yaml`
- dock widget discoverable from napari
- local development notes and debug workflow in `README.md` and `scripts/debug_widget.py`

### Phase 1: SpatialData discovery and validation

Status: complete.

Implemented:

- discovery of viewer-linked `SpatialData` labels layers from `napari-spatialdata`
- linked table discovery via `TableModel` metadata
- validation of region and instance linkage
- validation of available `adata.obsm` keys
- user-facing validation feedback for invalid or incomplete datasets

### Phase 2: Basic annotation workflow

Status: complete.

Implemented:

- pick-based selection using `layer.selected_label`
- custom mouse-picking support for multiscale labels layers
- manual class assignment and clearing
- `adata.obs["user_class"]` with `0` meaning unlabeled
- stable class palette handling
- recoloring of the labels layer from table-backed annotation state
- labels-layer feature/status refresh from current table state

### Phase 3: Manual table sync to zarr

Status: complete.

Implemented:

- manual `Write` action in the widget
- partial write-back to the selected backed table path
- persistence of current `obs` plus Harpy-owned palette/config metadata in `uns`
- explicit success and failure feedback without automatic reload

### Phase 4: Background random forest training

Status: complete.

Implemented:

- training eligibility checks
- feature matrix normalization and shape validation
- training on labeled rows only
- worker-based retraining with debounce
- stale-job cancellation / stale-result dropping
- write-back of:
  - `pred_class`
  - `pred_confidence`
  - `classifier_config`

### Phase 5: Live prediction updates in the viewer

Status: complete for the full-dataset workflow.

Implemented:

- prediction over active segmentation rows
- `Color by` modes for `user_class`, `pred_class`, and `pred_confidence`
- stable predicted-class palette handling
- continuous confidence colormap with missing-value fallback
- main-thread layer refresh when classifier results arrive

### Phase 6: Reload from zarr

Status: complete.

Implemented:

- manual `Reload` action in the widget
- partial reload of the selected table `obs`, `obsm`, and `uns`
- dirty-reload decision flow:
  - write local edits and reload
  - reload and discard local edits
  - cancel
- classifier freeze/reset around reload so late worker results are ignored
- feature-key and widget-state refresh after reload

### Phase 7: ROI selection and subsetting

Status: deferred.

Deferred work:

- ROI layer discovery and validation
- ROI UI and selection semantics
- ROI membership tracking
- ROI-restricted annotation, training, prediction, and persistence

### Phase 8: Persistence to SpatialData / zarr

Status: complete.

Implemented:

- persisted `user_class`, `pred_class`, `pred_confidence`, and `classifier_config`
- persisted class-palette metadata for user and predicted classes
- reload-safe in-memory replacement of selected table state

Deferred with ROI support:

- writing only the active ROI subset while leaving rows outside the ROI untouched

## Data Model In Repo Today

### Required inputs

- one loaded labels layer linked to a `SpatialData` object
- one linked `AnnData` table
- one selected feature matrix in `adata.obsm`

### Written outputs

- `adata.obs["user_class"]`
- `adata.obs["pred_class"]`
- `adata.obs["pred_confidence"]`
- `adata.uns["classifier_config"]`

### Related palette metadata

- `adata.uns["user_class_colors"]`
- `adata.uns["pred_class_colors"]`

## Next Meaningful Roadmap Work

The next high-value roadmap items are continue planned integration with Harpy for feature extraction workflows.
