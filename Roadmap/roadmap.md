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
- Optional ROI selection from a compatible shapes layer, added after the core annotation, training, and prediction flow
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
  - Stays intentionally thin and delegates table semantics to `spatialdata` rather than reimplementing them.
  - Finds compatible segmentation layers, linked tables, and candidate ROI shapes layers.
  - Resolves table linkage metadata such as `region`, `region_key`, and `instance_key`.
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
  - Writes table updates back to disk through an explicit sync action.
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

- [x] `pip install -e ".[dev]"` succeeds.
- [x] napari can discover the plugin.
- [x] A dock widget opens without errors.

### Phase 1: SpatialData discovery and validation

### Outcome

The plugin can detect whether the active napari session contains a compatible `SpatialData` dataset for classification.

### Tasks

- [x] Identify how `napari-spatialdata` exposes the loaded `SpatialData` object and linked layers.
- [x] Detect available segmentation masks.
- [X] Detect annotation tables linked through `TableModel`.
- [X] Validate that the selected table contains:
  - [X] valid region and instance mapping
  - [X] at least one `.obsm` entry
- [x] Surface clear validation errors in the UI.

### Exit criteria

- [x] User can select a valid segmentation/table pair.
- [x] User can select a valid `adata.obsm` key.
- [x] Invalid datasets fail with understandable messages.

### Phase 2: Basic annotation workflow

### Outcome

The user can assign class labels to segmented objects from napari.

### Implementation note

Keep `HarpyWidget` focused on UI state. Viewer and `SpatialData` discovery should live in a thin
`SpatialDataAdapter` so Phase 2 annotation logic does not accumulate inside the widget.

For MVP, "current selection" means the currently selected object in the napari `Labels` layer.
The controller should keep `layer.selected_label` in sync with viewer clicks and treat that value
as the selected segmentation instance id.

Because napari does not support native pick mode for multiscale labels layers, the plugin may need
its own mouse-picking callback to keep selection working consistently across both single-scale and
multiscale segmentations.

Do not edit the segmentation data itself. The segmentation layer remains an immutable instance-id
map and should be recolored via a direct instance-id-to-color mapping.

Use `adata.obs["user_class"]` as an integer annotation column with `0` meaning "unlabeled" and
`1, 2, 3, ...` reserved for user classes. Clearing a label resets it to `0`.

For now, treat `adata.uns["user_class_colors"]` as derived state from the current user classes.
If an existing palette is present, `napari-harpy` may overwrite it with its generated palette and we log when this happens.

### Suggested controller API

- `AnnotationController.bind(...)`: resolve the concrete `Labels` layer for the current widget selection, connect
  to `layer.events.selected_label`, and attach any custom mouse-picking callbacks needed for
  multiscale labels support.
- `ensure_annotation_column("user_class")`: create or fill the annotation column with `0`.
- `apply_current_class()`: read `labels_layer.selected_label`; if it is `> 0`, update the backing table row(s) where
  `region_key == selected_segmentation_name` and `instance_key == selected_label`.
- `clear_current_label()`: set the current object's `user_class` back to `0`.
- `refresh_layer_colors()`: build `instance_id -> color` for all visible ids in `layer.metadata["indices"]`, then
  assign a direct colormap.
- `refresh_layer_features()`: optionally set
  `layer.features = DataFrame({"user_class": ..., "index": instance_ids})` so hover and status reflect the
  annotation state.

### Tasks

- [x] Define the simplest annotation interaction model for MVP:
  - [x] use viewer clicks to choose the current object, including multiscale labels layers
  - [x] treat `layer.selected_label` as the current instance id
- [ ] Add UI elements for:
  - [x] current class label
  - [x ] apply label to current picked object
  - [x] clear label for current picked object
  - [x] optional readout of the currently picked instance id
- [x] Resolve the picked napari label to a segmentation instance id.
- [x] Map instance ids to `adata.obs` rows via `instance_key` and `region_key`.
- [x] Initialize `adata.obs["user_class"]` if missing, using `0` for unlabeled.
- [x] Recolor the active segmentation layer from `user_class` values without modifying segmentation ids.
- [x] Keep background transparent and define a stable unlabeled color for class `0`.

### Exit criteria

- [x] User can pick a segmented object and assign it a class id.
- [x] Labels are stored in `adata.obs["user_class"]` with `0` representing unlabeled.
- [x] Clearing an annotation resets `user_class` to `0`.
- [x] The existing segmentation layer is recolored from `user_class` without changing instance ids.
- [x] Relabeling an object updates the table and viewer correctly.



### Phase 3: Manual table sync to zarr

### Outcome

The user can explicitly persist `adata.obs[...]` updates from the active annotation table back to the
zarr-backed `SpatialData` store.

### Implementation note

Start with a manual sync button rather than autosave. After Phase 2, the in-memory table is the source
of truth for annotation edits, and this phase adds an explicit write-back step to disk.

For MVP, this phase only needs to guarantee persistence of annotation columns such as
`adata.obs["user_class"]`. The same sync pathway can later be extended to prediction fields and
classifier metadata.

`Write to zarr` should be strictly memory -> disk. It should not also imply reloading the store back
into the viewer.

### Tasks

- [x] Add a `Write to zarr` action in the widget.
- [x] Define the write-back source of truth:
  - [x] treat the selected in-memory `SpatialData` table as authoritative while the session is open
  - [x] persist the current `adata.obs[...]` values for the selected table
- [x] Implement a `PersistenceController` or equivalent write-back helper.
- [x] Support partial table persistence:
  - [x] locate the selected table path in the backed `SpatialData` store
  - [x] rewrite only the `obs` element for the selected table rather than rewriting the whole store
- [x] Write the updated table back into the zarr-backed `SpatialData` store safely.
- [x] Surface success and failure states clearly in the UI.
- [x] Decide and document what happens after sync:
  - [x] keep the current in-memory `SpatialData` table authoritative after a successful sync
  - [x] do not automatically reload the store as part of sync
- [x] Keep the sync step manual for MVP rather than automatically syncing on every click.

### Exit criteria

- [x] User can annotate objects, click `Write to zarr`, and persist `adata.obs["user_class"]` to disk.
- [x] Sync failures are visible and do not silently discard in-memory edits.
- [x] The roadmap clearly distinguishes viewer rescan, disk sync, and disk reload.

### Phase 4: Background random forest training

### Outcome

The plugin retrains a classifier in the background whenever annotations change.

### Tasks

- [x] Define training eligibility rules:
  - [x] enough labeled samples
  - [x] at least two classes
  - [x] feature matrix shape matches table rows -> I believe this is by construction if you use an AnnData table.
- [x] Add a `RandomForestClassifier` training pipeline.
- [x] Train on labeled rows only.
- [x] Use async worker execution with napari threading.
- [x] Use a debounced background training loop:
  - [x] every annotation change schedules a retrain
  - [x] wait about 200 to 500 ms so bursts of clicks collapse into one job
  - [x] train in a worker thread
  - [x] when training finishes, run prediction on all objects
  - [x] update layer coloring and overlays on the main thread
- [x] Prefer napari's `@thread_worker` pattern for training jobs and UI-safe completion handling.
- [x] Implement stale-job cancellation or stale-result dropping so only the newest fit is applied.
- [x] Capture training metadata in `adata.uns["classifier_config"]`.

### Exit criteria

- Label edits trigger background retraining.
- UI remains responsive during training.
- Older jobs do not overwrite newer results.

### Phase 5: Live prediction updates in the viewer

### Outcome

Objects are recolored live from selectable table-derived state.

### Status update

The full-table prediction and viewer-coloring workflow described in this phase is implemented.
ROI-restricted prediction behavior is intentionally deferred to Phase 7.

### Implementation note

Add a `Color by` dropdown to the widget so the user can explicitly choose how the active labels layer is
styled. For MVP, start with:

- `user_class`
- `pred_class`
- `pred_confidence`

Default to `user_class` so annotation remains the clearest first interaction.

For `pred_class`, reuse the same class-id-to-color mapping as `user_class` rather than generating a new
palette from only the predicted categories currently present. This keeps class `1`, class `2`, and so on
visually stable even when some categories are temporarily absent from predictions.

For `pred_confidence`, treat the values as continuous in `[0, 1]` and use a continuous colormap.
`NaN` or missing confidence values should map to a clear fallback color so the "not yet predicted" state
remains visible.

The recoloring step should happen on the main thread after worker completion. We can defer richer
"overlays" beyond layer recoloring and basic viewer status readouts if they are not needed for MVP.

### Tasks

- [x] Predict classes for all objects after each successful fit.
- [ ] If ROI is active, predict only for objects in the active subset and define how out-of-ROI objects are displayed.
  Deferred to Phase 7 ROI support.
- [x] Compute and store:
  - [x] `pred_class`
  - [x] `pred_confidence`
- [x] Decide representation for unlabeled or not-yet-predicted objects.
- [x] Add a `Color by` dropdown to the widget.
- [x] Support `Color by = user_class`.
- [x] Support `Color by = pred_class`.
- [x] Support `Color by = pred_confidence`.
- [x] Update layer coloring on the main thread when classifier results arrive.
- [x] Reuse the stable user-class palette when coloring by `pred_class`, even if some categories are absent.
- [x] Add a continuous colormap for `pred_confidence` in `[0, 1]`.
- [x] Define fallback styling for missing/`NaN` prediction confidence values.
- [x] Refresh the active labels layer when the `Color by` selection changes.

### Exit criteria

- [x] Prediction updates are visible immediately after training finishes.
- [x] The user can switch layer styling between `user_class`, `pred_class`, and `pred_confidence`.
- [x] `pred_class` uses the same stable class colors as `user_class`.
- [x] `pred_confidence` uses a continuous colormap with a clear missing-value fallback.
- [x] Untrained or invalid states are visually clear.
- [ ] ROI-restricted predictions are visually understandable.
  Deferred to Phase 7 ROI support.

### Phase 6: Reload from zarr

### Outcome

The user can explicitly reload backed table state from zarr into the active napari session when the
on-disk store changed outside the current in-memory workflow.

### Implementation note

`Reload from zarr` should be strictly disk -> memory. Keep it separate from `Write to zarr`.

For MVP, reload only the currently selected table state from disk by reading `obs`, `obsm`, and `uns`
directly from the backed zarr store. Keep `sdata.tables[table_name]` authoritative, refresh Harpy-owned
widget and styling state after reload, and defer any full-`SpatialData` rebuild or napari-spatialdata
cache regeneration to later follow-up work. This phase is mainly about safely picking up external changes
such as newly written `.obs` columns or new `.obsm[...]` entries.

### Tasks

- [x] Add a `Reload from zarr` action in the widget.
- [x] Read the selected table snapshot directly from the backed zarr store instead of rereading the full `SpatialData`.
- [x] Keep reload scope to the currently selected table for MVP and defer full-`SpatialData` reload.
- [x] Refresh the in-memory selected table state after reload.
- [x] Trigger the necessary widget and layer refresh so new `.obs` columns or `.obsm[...]` keys become visible.
- [x] Surface clear UI messaging if reload would discard unsynced in-memory edits.
- [x] Freeze async classifier work around reload so stale worker results cannot overwrite reloaded state.

### Exit criteria

- [x] User can click `Reload from zarr` and pick up on-disk table changes in the widget.
- [x] Reload updates relevant table-derived UI state such as feature keys.
- [x] Reload behavior is clearly separated from viewer rescan and from sync.
- [x] Stale classifier jobs cannot overwrite freshly reloaded table state.

### Phase 7: ROI selection and subsetting

### Outcome

The user can optionally constrain annotation and model updates to a valid ROI chosen from a shapes layer.

### Tasks

- [ ] Detect candidate ROI shapes layers.
- [ ] Define how ROI selection is presented in the UI.
- [ ] Add an ROI menu with `None` and compatible shapes layers.
- [ ] Validate ROI inputs:
  - [ ] `None` means full dataset
  - [ ] shapes layer is compatible with the segmentation coordinate system
  - [ ] selected ROI geometry is acceptable for MVP, for example a square or rectangular region
- [ ] Apply `spatialdata.bounding_box_query` to derive the working subset when ROI is selected.
- [ ] Track table rows and segmentation objects inside the active ROI.
- [ ] Respect active ROI state for annotation:
  - [ ] if ROI is `None`, annotations operate on the full segmentation
  - [ ] if ROI is set, annotations only apply to objects inside the subset
- [ ] Respect active ROI state for training:
  - [ ] if ROI is active, train only on rows inside the current ROI subset
- [ ] Trigger a clean recomputation of the working subset and model state when ROI changes.

### Exit criteria

- User can select `None` or a valid ROI layer.
- Invalid ROI inputs fail with understandable messages.
- Objects outside the active ROI are ignored or clearly blocked from annotation.
- ROI changes trigger a clean recomputation of the working subset and model state.

### Phase 8: Persistence to SpatialData / zarr

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

### Phase 9: MVP hardening

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
4. list available segmentation masks, linked tables, and `.obsm` keys in the UI

This gives us the integration backbone before we build annotation and training.

## Open Questions To Refine Next

- What is the exact napari interaction for selecting segmented objects in your preferred workflow?
- What exact ROI shapes do we want to support in MVP: only axis-aligned rectangles, or any polygon with a rectangular bounding box fallback?
- Should classes be free integer input, a predefined list, or both?
- Should persistence be automatic or user-triggered in MVP?
- What is the exact Harpy API for writing the updated table back to `SpatialData`?
- Do we want predictions for all objects immediately, or only after a minimum annotation threshold is reached?
- How should out-of-ROI objects be shown while an ROI is active: unchanged, dimmed, or hidden?
