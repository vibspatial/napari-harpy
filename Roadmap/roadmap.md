# napari-harpy Roadmap

## Goal

Build `napari-harpy`, a napari plugin for interactive object classification on `SpatialData` datasets backed by zarr.

For the first minimal viable product, the plugin should:

- work on a `SpatialData` object loaded through `napari-spatialdata`
- require at least one segmentation mask
- require one `AnnData` table linked to that segmentation mask through `TableModel` region and instance keys
- let the user choose a feature matrix from `adata.obsm`, for example `adata.obsm["features"]`
- optionally let the user choose an ROI from a shapes layer; if ROI is `None`, use the full segmentation, and if ROI is set, subset the `SpatialData` object with `spatialdata.bounding_box_query`
- let the user annotate segmented objects with user classes such as `1, 2, 3, ...`
- train a background `RandomForestClassifier` on `adata.obsm[selected_key]` plus user annotations
- update predictions live in the viewer by coloring objects with predicted class
- write the following fields back into the table:
  - `adata.obs["user_class"]`
  - `adata.obs["pred_class"]`
  - `adata.obs["pred_confidence"]`
  - `adata.uns["classifier_config"]`
- persist the updated table back to disk in the zarr-backed `SpatialData` store
- ignore annotations outside the active ROI subset
- retrain asynchronously with debounce and stale-job cancellation rather than trying to make random forest itself streaming

## Guiding Principles

- Keep the first version simple and reliable.
- Prefer explicit validation and clear user feedback over hidden magic.
- Separate UI state, table state, and model-training state cleanly.
- Treat background retraining as async refit jobs with cancellation, not online learning.
- Make disk persistence safe and predictable.

## MVP Scope

### In scope

- Selecting one segmentation layer and its linked annotation table
- Selecting one `adata.obsm` entry as features
- Optional ROI selection from a compatible shapes layer
- Manual class annotation of segmented objects
- Background random forest training and prediction refresh
- Live object recoloring by predicted class
- Writing annotations and predictions back into `adata`
- Persisting the updated table to the zarr-backed `SpatialData`

### Out of scope for MVP

- Multiple classifiers
- Hyperparameter tuning UI
- Active learning strategies
- Batch annotation workflows
- Undo/redo history beyond simple state updates
- Multi-user editing
- Advanced performance optimization for very large datasets
- Streaming or partial-fit tree models

## Proposed MVP Architecture

### Data flow

1. User loads a `SpatialData` object in napari via `napari-spatialdata`.
2. Plugin discovers available segmentation masks, linked tables, and candidate ROI shapes layers.
3. User selects:
   - segmentation mask
   - linked table
   - `adata.obsm` feature key
   - ROI setting: `None` or a compatible shapes layer
4. If ROI is set, plugin creates a working subset via `spatialdata.bounding_box_query` and uses only objects inside that ROI.
5. User annotates objects with class labels.
6. Plugin stores labels in `adata.obs["user_class"]` for the active subset and ignores annotations outside the ROI.
7. Background worker retrains a random forest on labeled objects only.
8. Model predicts all objects in the active working set and updates:
   - `adata.obs["pred_class"]`
   - `adata.obs["pred_confidence"]`
   - `adata.uns["classifier_config"]`
9. Viewer refreshes colors using `pred_class`.
10. Updated table is persisted back to the `SpatialData` store.

### Main plugin components

- `SpatialDataAdapter`
  - Finds compatible segmentation layers, linked tables, and candidate ROI shapes layers.
  - Resolves `region_key` and `instance_key`.
  - Exposes available `adata.obsm` keys.

- `ROIController`
  - Exposes an ROI menu with `None` and compatible shapes layers.
  - Validates that the selected ROI shape is acceptable for MVP, for example a rectangular bounding region.
  - Applies `spatialdata.bounding_box_query` to derive the working subset.
  - Tracks which table rows are in or out of the active ROI.

- `AnnotationController`
  - Handles current class selection.
  - Maps selected objects to table rows.
  - Writes `user_class` values into `adata.obs`.
  - Ignores or blocks edits for objects outside the active ROI.

- `ClassifierController`
  - Validates training prerequisites.
  - Schedules retraining with debounce.
  - Cancels stale jobs.
  - Applies latest predictions only if job result is still current.

- `ViewerStylingController`
  - Colors segmentation objects by `pred_class`.
  - Refreshes layer properties when predictions change.

- `PersistenceController`
  - Writes table updates back to disk.
  - Centralizes save policy and error handling.

## Step-by-Step Implementation Plan

### Phase 0: Project setup and plugin skeleton

### Outcome

A minimal napari plugin package that can be discovered and opened in napari.

### Tasks

- [x] Clean up package layout so the project is installable as a napari plugin.
- [x] Add a minimal `src/` package structure for `napari_harpy`.
- [x] Register the plugin entry point.
- [x] Add a first widget or dock widget placeholder.
- [x] Add a small developer note on how to run the plugin locally.

### Exit criteria

- `pip install -e ".[dev]"` succeeds.
- napari can discover the plugin.
- A dock widget opens without errors.

### Phase 1: SpatialData discovery and validation

### Outcome

The plugin can detect whether the active napari session contains a compatible `SpatialData` dataset for classification.

### Tasks

- [ ] Identify how `napari-spatialdata` exposes the loaded `SpatialData` object and linked layers.
- [ ] Detect available segmentation masks.
- [ ] Detect annotation tables linked through `TableModel`.
- [ ] Detect candidate ROI shapes layers.
- [ ] Validate that the selected table contains:
  - [ ] valid region and instance mapping
  - [ ] at least one `.obsm` entry
- [ ] Validate ROI inputs:
  - [ ] `None` means full dataset
  - [ ] shapes layer is compatible with the segmentation coordinate system
  - [ ] selected ROI geometry is acceptable for MVP, for example a square or rectangular region
- [ ] Define how ROI selection is presented in the UI.
- [ ] Surface clear validation errors in the UI.

### Exit criteria

- User can select a valid segmentation/table pair.
- User can select a valid `adata.obsm` key.
- User can select `None` or a valid ROI layer.
- Invalid datasets fail with understandable messages.

### Phase 2: Basic annotation workflow

### Outcome

The user can assign class labels to segmented objects from napari.

### Tasks

- [ ] Define the simplest annotation interaction model for MVP.
- [ ] Add UI elements for:
  - [ ] current class label
  - [ ] apply label to current selection
  - [ ] clear label for current selection
- [ ] Respect active ROI state:
  - [ ] if ROI is `None`, annotations operate on the full segmentation
  - [ ] if ROI is set, annotations only apply to objects inside the subset
- [ ] Resolve napari object selection to segmentation instance ids.
- [ ] Map instance ids to `adata.obs` rows via `instance_key`.
- [ ] Initialize `adata.obs["user_class"]` if missing.
- [ ] Store labels as nullable values until annotated.

### Exit criteria

- User can label selected objects with class ids.
- Labels are stored in `adata.obs["user_class"]`.
- Relabeling an object updates the table correctly.
- Objects outside the active ROI are ignored or clearly blocked from annotation.

### Phase 3: Background random forest training

### Outcome

The plugin retrains a classifier in the background whenever annotations change.

### Tasks

- [ ] Define training eligibility rules:
  - [ ] enough labeled samples
  - [ ] at least two classes
  - [ ] feature matrix shape matches table rows
- [ ] If ROI is active, train only on rows inside the current ROI subset.
- [ ] Add a `RandomForestClassifier` training pipeline.
- [ ] Train on labeled rows only.
- [ ] Use async worker execution with napari threading.
- [ ] Use a debounced background training loop:
  - [ ] every annotation change schedules a retrain
  - [ ] wait about 200 to 500 ms so bursts of clicks collapse into one job
  - [ ] train in a worker thread
  - [ ] when training finishes, run prediction on all objects
  - [ ] update layer coloring and overlays on the main thread
- [ ] Prefer napari's `@thread_worker` pattern for training jobs and UI-safe completion handling.
- [ ] Implement stale-job cancellation or stale-result dropping so only the newest fit is applied.
- [ ] Capture training metadata in `adata.uns["classifier_config"]`.

### Exit criteria

- Label edits trigger background retraining.
- UI remains responsive during training.
- Older jobs do not overwrite newer results.
- ROI changes trigger a clean recomputation of the working subset and model state.

### Phase 4: Live prediction updates in the viewer

### Outcome

Objects are recolored live by predicted class.

### Tasks

- [ ] Predict classes for all objects after each successful fit.
- [ ] If ROI is active, predict only for objects in the active subset and define how out-of-ROI objects are displayed.
- [ ] Compute and store:
  - [ ] `pred_class`
  - [ ] `pred_confidence`
- [ ] Decide representation for unlabeled or not-yet-predicted objects.
- [ ] Push updated layer properties into the segmentation layer.
- [ ] Add a stable colormap for class ids.

### Exit criteria

- Prediction updates are visible immediately after training finishes.
- Objects are colored by `pred_class`.
- Untrained or invalid states are visually clear.
- ROI-restricted predictions are visually understandable.

### Phase 5: Persistence to SpatialData / zarr

### Outcome

User annotations and predictions survive reloads.

### Tasks

- [ ] Confirm the exact Harpy write-back API to persist tables into the zarr-backed `SpatialData` store.
- [ ] Update or replace the linked table safely.
- [ ] If ROI is active, write back changes only for rows inside the ROI subset and leave rows outside untouched.
- [ ] Decide when saves happen:
  - [ ] immediately after annotation
  - [ ] after prediction refresh
  - [ ] manual save button
- [ ] Add minimal error handling for write failures.
- [ ] Ensure persisted metadata includes `classifier_config`.

### Exit criteria

- After reload, `user_class`, `pred_class`, `pred_confidence`, and `classifier_config` are present.
- Persisted data remains linked to the segmentation via table metadata.
- ROI-limited edits do not accidentally overwrite rows outside the ROI.

### Phase 6: MVP hardening

### Outcome

The first usable version is stable enough for iterative testing.

### Tasks

- [ ] Add tests for:
  - [ ] table validation
  - [ ] annotation-to-row mapping
  - [ ] ROI subsetting and row membership
  - [ ] training eligibility
  - [ ] stale-job protection
- [ ] Add lightweight logging for key transitions.
- [ ] Improve user-facing error messages.
- [ ] Test on one small real zarr-backed `SpatialData` example.

### Exit criteria

- Core flows work end to end on a real dataset.
- Common failure modes produce clear messages.

## Data Model for MVP

### Required inputs

- one segmentation mask
- one linked `AnnData` table
- one selected feature matrix in `adata.obsm`
- optional ROI shapes layer or `None`

### Required outputs

- `adata.obs["user_class"]`
- `adata.obs["pred_class"]`
- `adata.obs["pred_confidence"]`
- `adata.uns["classifier_config"]`

### Suggested `classifier_config` contents

- model type
- selected `obsm` key
- ROI mode and ROI source
- training timestamp
- number of labeled objects
- number of objects inside the active ROI
- class labels seen during training
- important RF hyperparameters

## Key Technical Decisions

### 1. ROI strategy

MVP assumption:
The plugin exposes an `ROI` menu with `None` and compatible shapes layers. If `ROI` is `None`, the full segmentation is used. If an ROI is selected, the plugin derives a working subset using `spatialdata.bounding_box_query` and only operates on objects inside that subset.

Reason:
This gives users a focused workflow for local annotation and classification without forcing the whole dataset through every interaction.

### 2. Annotation model

MVP assumption:
Use class assignment to currently selected segmentation objects in napari.

Reason:
This is the simplest mental model and avoids building a more complex review UI too early.

### 3. Training model

MVP assumption:
Use `sklearn.ensemble.RandomForestClassifier` with full refit in a worker thread.

Reason:
It is robust, easy to explain, and fits the async refit plus debounce pattern you want.

### 4. Async strategy

MVP assumption:
Debounce annotation changes, launch retraining in a background worker, and discard stale results. Use napari's `@thread_worker` style execution so training happens off the main thread and viewer updates happen safely afterward.

Reason:
This gives near-live feedback without pretending RF supports online learning.

### 5. Persistence strategy

MVP assumption:
Persist the linked table back into the `SpatialData` store after successful updates, and if ROI mode is active only apply writes to rows inside the active subset.

Reason:
Keeping `adata` authoritative is the most straightforward way to preserve state across reloads.

## Risks and Watchouts

- Mapping napari selections back to table rows may be the trickiest integration point.
- `napari-spatialdata` layer metadata conventions need to be inspected early.
- ROI geometry validation and coordinate-system alignment may be tricky.
- `bounding_box_query` may include edge cases around partially intersecting objects.
- Large feature matrices may make retraining feel sluggish without debounce.
- Rewriting tables in-place may need careful handling to avoid corrupting or duplicating table entries.
- Feature matrices in `.obsm` may be sparse, dense, or unexpectedly shaped.

## First Implementation Slice

The first slice we should implement after this roadmap:

1. make the package installable as a napari plugin
2. add a dock widget skeleton
3. inspect the active `SpatialData` object from `napari-spatialdata`
4. list available segmentation masks, linked tables, ROI shapes layers, and `.obsm` keys in the UI

This gives us the integration backbone before we build annotation and training.

## Open Questions To Refine Next

- What is the exact napari interaction for selecting segmented objects in your preferred workflow?
- What exact ROI shapes do we want to support in MVP: only axis-aligned rectangles, or any polygon with a rectangular bounding box fallback?
- Should classes be free integer input, a predefined list, or both?
- Should persistence be automatic or user-triggered in MVP?
- What is the exact Harpy API for writing the updated table back to `SpatialData`?
- Do we want predictions for all objects immediately, or only after a minimum annotation threshold is reached?
- How should out-of-ROI objects be shown while an ROI is active: unchanged, dimmed, or hidden?
