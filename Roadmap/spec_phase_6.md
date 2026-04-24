# Phase 6 Spec: Reload From Zarr

## Goal

Provide a safe, explicit `Reload from zarr` action that refreshes the currently selected table from disk
without rebuilding the full `SpatialData` object and without letting Harpy, `SpatialData`, and napari layer
metadata drift out of sync.

This phase is intentionally separate from `Write to zarr`.

- `Write to zarr` is memory -> disk
- `Reload from zarr` is disk -> memory

## Investigation Context

The main risk in this phase is state divergence across three places:

1. the on-disk zarr table under `tables/<table_name>`
2. the in-memory table inside `sdata.tables[table_name]`
3. the in-memory `adata` cached in `layer.metadata["adata"]` by `napari-spatialdata`

If phase 6 does not define one authoritative in-memory object and one cache refresh strategy, the user can
end up in a very confusing state where the widget, the layer styling, and the disk contents all disagree.

## Versions Investigated

The findings below were checked against the local environment in this repo:

- `napari-spatialdata 0.7.0`
- `spatialdata 0.7.2`
- `anndata 0.12.10`

## Findings

### 1. Harpy currently uses `sdata[table_name]` as the working table

Current Harpy code reads the selected table through `SpatialDataAdapter.get_table()` and therefore works
against the table stored inside the selected in-memory `SpatialData` object.

Relevant code paths:

- `src/napari_harpy/_spatialdata.py`
- `src/napari_harpy/_annotation.py`
- `src/napari_harpy/_classifier.py`
- `src/napari_harpy/_classifier_viewer_styling.py`
- `src/napari_harpy/_persistence.py`

This is good news for phase 6, because there is already a natural authoritative object to keep.

### 2. `napari-spatialdata` stores an `adata` object in layer metadata

`napari-spatialdata` adds metadata such as:

- `layer.metadata["sdata"]`
- `layer.metadata["adata"]`
- `layer.metadata["region_key"]`
- `layer.metadata["instance_key"]`
- `layer.metadata["table_names"]`

This confirms the user's expectation that table state is cached on the napari layer.

### 3. `layer.metadata["adata"]` is not the same object as `sdata[table_name]`

This is the most important finding.

`napari-spatialdata` does not simply store the full table object from `sdata.tables[table_name]` in the layer
metadata. Instead, it constructs a joined `AnnData` using:

`join_spatialelement_table(..., how="left", match_rows="left")`

So `layer.metadata["adata"]` is a derived cache, not the authoritative table object.

This means phase 6 must not try to keep both objects independently editable. One of them needs to be
authoritative and the other must be treated as disposable derived state.

### 4. `napari-spatialdata` itself updates both metadata `adata` and `sdata[table_name]`

The plugin code in `napari-spatialdata` updates `selected_layer.metadata["adata"]` and also updates
`selected_layer.metadata["sdata"][selected_table]`.

This confirms that the dual-state problem is real and not hypothetical.

### 5. Full `SpatialData` reload is not needed for MVP

The MVP roadmap already prefers reloading the current table instead of rebuilding the whole viewer, and the
investigation supports that direction.

For phase 6 we only need:

- `obs`
- `obsm`
- `uns`

We do not need to reload images, labels, shapes, or points.

### 6. Reading the whole table through `read_zarr()` is still heavier than needed

`spatialdata.read_zarr(..., selection=("tables",))` still reads each table via `anndata.read_zarr()`, which
reloads the full table object.

That is functional, but it is broader than necessary for this feature.

For MVP, the cleaner approach is to reopen the selected table zarr group directly and read only:

- `obs`
- `obsm`
- `uns`

with `anndata.io.read_elem`.

### 7. Partial reload is safe only if row identity and row order are unchanged

Reloading only `obs`, `obsm`, and `uns` without reloading the full `AnnData` object is safe only if the table
still has the same rows in the same order.

That means phase 6 must validate:

- same `n_obs`
- same `obs_names`
- same `obs_names` order

If those checks fail, MVP should abort reload instead of trying to reconcile the mismatch.

### 8. `layer.metadata["adata"]` should be treated as a cache

Because the metadata `adata` is derived from the selected table and selected spatial element, it should be
regenerated after reload instead of patched column by column.

This is the safest approach and best matches how `napari-spatialdata` created it in the first place.

#### How the cache should be regenerated

Harpy should rebuild `layer.metadata["adata"]` from the refreshed in-memory `SpatialData` table after a
successful reload.

The intended implementation is:

1. reload disk state into `sdata.tables[table_name]`
2. resolve the currently bound napari layer for the selected spatial element
3. construct a fresh joined `AnnData` for that layer using the selected `sdata`, `table_name`, and
   `label_name`
4. use the same join semantics as `napari-spatialdata`:
   - `join_spatialelement_table(...)`
   - `how="left"`
   - `match_rows="left"`
5. replace only the table-derived metadata fields:
   - `layer.metadata["adata"]`
   - `layer.metadata["region_key"]`
   - `layer.metadata["instance_key"]`
   - `layer.metadata["table_names"]` if it is being refreshed
6. leave non-table metadata untouched:
   - `layer.metadata["sdata"]`
   - `layer.metadata["name"]`
   - `layer.metadata["indices"]`
   - coordinate-system metadata
   - affine/view state
7. run the normal Harpy refresh steps after the metadata cache replacement:
   - feature matrix dropdown refresh
   - selection status refresh
   - layer color refresh
   - layer features refresh
   - classifier status refresh

Important implementation note:

- Harpy should not import or depend on a private helper from `napari-spatialdata` for this.
- Harpy should reproduce the same behavior with the public `spatialdata.join_spatialelement_table(...)`
  API.

Fallback rule:

- if the selected table is reloaded successfully but the corresponding labels layer is not currently loaded in
  the viewer, skip the metadata-cache refresh step
- in that case the authoritative state in `sdata.tables[table_name]` is still updated, and the next bind/load
  of that layer should see the refreshed table

## Core Decision For Phase 6

Phase 6 should formalize this authority model:

- authoritative in-memory table: `sdata.tables[table_name]`
- derived napari cache: `layer.metadata["adata"]`

Harpy should read and write through the in-memory `SpatialData` table.
The napari layer metadata should be refreshed from that table after reload.

Harpy should not treat `layer.metadata["adata"]` as a second editable source of truth.

## MVP Scope

Phase 6 reload should support:

- reloading the currently selected backed table only
- reloading only `obs`, `obsm`, and `uns`
- refreshing the active bound layer metadata after reload
- refreshing widget state that depends on table contents

Phase 6 reload should explicitly not support:

- reloading the entire `SpatialData` object
- rebuilding the full viewer
- handling row-count changes
- handling reordered rows
- handling table deletion or rename on disk
- auto-merging local unsynced edits with disk edits

## Required Behavior

### User-visible behavior

The widget gets a new explicit button:

- `Reload from zarr`

This action must:

- operate only on the currently selected backed table
- never silently discard unsynced in-memory edits
- refresh the current widget and layer state after success
- stay clearly separate from `Rescan Viewer`
- stay clearly separate from `Write to zarr`

### Dirty-state rule

If the current table has unsynced in-memory edits, `Reload from zarr` must not proceed silently.

MVP interaction:

1. If there are no unsynced edits, reload immediately.
2. If there are unsynced edits, show a confirmation dialog with:
  - `Write`
   - `Reload and discard local edits`
   - `Cancel`

Required semantics:

- `Write` performs memory -> disk first, then reloads from disk
- `Reload and discard local edits` replaces in-memory state from disk
- `Cancel` leaves everything unchanged

## Reload Algorithm

### Preconditions

Reload is allowed only when:

- a `SpatialData` object is selected
- a table is selected
- the selected `SpatialData` is backed by zarr
- the selected table can be resolved to a unique zarr path

### Step 1. Freeze active async work

Before reload:

- cancel pending classifier debounce timers
- cancel active classifier workers
- block stale worker results from being applied after reload

This prevents a background prediction job from writing stale results into a freshly reloaded table.

### Step 2. Read the current table snapshot from disk

Resolve the selected table path as Harpy already does for sync.

Open the selected table group from zarr and read:

- `obs`
- `obsm`
- `uns`

Use direct zarr access plus `anndata.io.read_elem`.

For consistency with the current persistence path, prefer `use_consolidated=False` when opening the zarr group.

### Step 3. Validate the disk snapshot

Before applying the disk snapshot, validate:

- `obs` is a `DataFrame`
- `obsm` is a mapping
- `uns` is a mapping
- `uns["spatialdata_attrs"]` exists
- the reloaded table still has valid `region_key` and `instance_key`
- the selected table still annotates the selected segmentation
- `len(obs)` matches the current in-memory table
- `obs.index` matches the current in-memory table index exactly and in order
- every `obsm[key]` has first dimension equal to `len(obs)`

If any validation fails, abort reload and show a clear error message.

### Step 4. Replace the in-memory table state atomically

If validation passes, apply the disk snapshot atomically to the current selected table:

- replace `table.obs`
- replace `table.obsm`
- replace `table.uns`

Do not merge per-column.
Do not partially update only one of the three.
Do not leave the table half-updated if an error occurs.

For MVP, the apply logic should behave like all-or-nothing.

### Step 5. Refresh the `SpatialData` table reference

After reloading the snapshot, refresh the selected table reference inside the current `SpatialData` object so
all Harpy controllers continue to read the new data through `sdata[table_name]`.

For MVP, replacing `sdata.tables[table_name]` with a validated updated table is acceptable.

### Step 6. Regenerate layer metadata cache

After the authoritative in-memory table is refreshed, regenerate the active bound layer metadata cache.

Do not patch `layer.metadata["adata"]` in place.

Instead:

1. recreate the joined `adata` for the active layer using the same join semantics that
   `napari-spatialdata` uses
2. replace `layer.metadata["adata"]`
3. refresh `layer.metadata["region_key"]`
4. refresh `layer.metadata["instance_key"]`
5. preserve `layer.metadata["table_names"]` if still valid

This keeps Harpy aligned with the way `napari-spatialdata` constructs layer metadata.

## Widget And Controller Refresh Rules

After a successful reload, Harpy must refresh all state derived from the selected table.

### Feature matrix selection

Recompute the available `adata.obsm` keys.

Rules:

- if the previously selected feature key still exists, keep it selected
- if it no longer exists, fall back to the first available key
- if no keys remain, clear the selection and show validation feedback

### Annotation state

Recompute:

- selected instance status text
- current user class for the selected object
- apply/clear button enabled state

### Viewer styling

Refresh:

- direct label colormap
- layer features table
- current `Color by` mode rendering

This ensures that any on-disk change to:

- `user_class`
- `pred_class`
- `pred_confidence`
- color-related values in `uns`

becomes visible immediately after reload.

### Classifier state

After reload:

- there must be no active pending worker from before reload
- do not auto-retrain
- recompute classifier status against the reloaded table state
- preserve on-disk predictions if they exist

The classifier should become clean again relative to the newly reloaded table state.
Later in-memory edits can mark it dirty again as usual.

### Sync state

After a successful reload, Harpy should consider the table synchronized with disk.

That means:

- clear the unsynced-edit flag
- update sync/reload feedback text

## Failure Modes And Required Messages

Reload should fail clearly and without mutating in-memory state when:

- the selected dataset is not backed by zarr
- the selected table path cannot be resolved
- the table disappeared from disk
- `obs_names` no longer match
- `n_obs` changed
- required `spatialdata_attrs` are missing
- the selected segmentation is no longer annotated by the table
- an `obsm` entry has invalid shape

Suggested message pattern:

- explain what changed
- explain that partial reload is unsafe
- tell the user that MVP reload requires unchanged row identity/order

## Suggested Internal API Shape

The simplest shape is a dedicated reload helper, separate from the write-back method used for sync.

Possible controller responsibilities:

- `can_reload`
- `selected_table_store_path`
- `has_unsynced_changes`
- `reload_table_state()`
- `read_table_snapshot_from_disk()`
- `validate_reload_snapshot()`
- `apply_reload_snapshot()`
- `refresh_layer_metadata_cache()`

This can live in a new reload controller or in an expanded persistence controller, but the disk -> memory
path should remain clearly distinct from the memory -> disk path.

## Acceptance Criteria

Phase 6 is complete when all of the following are true:

- the user can click `Reload from zarr` for the active backed table
- new `.obsm` keys written on disk become visible in the feature-matrix dropdown
- reloaded `user_class`, `pred_class`, and `pred_confidence` values update the layer display
- Harpy refreshes `layer.metadata["adata"]` after reload
- unsynced local edits are never discarded without explicit confirmation
- reload aborts safely when row identity or row order changed
- stale classifier jobs cannot overwrite reloaded table state
- reload remains separate from viewer rescan and from sync

## Tests To Add

At minimum, phase 6 should add tests for:

1. reload picks up a new `obsm` key from disk
2. reload picks up changed `obs["user_class"]` from disk
3. reload refreshes prediction-driven layer styling
4. reload preserves the selected feature key when it still exists
5. reload falls back cleanly when the selected feature key was removed
6. reload refreshes `layer.metadata["adata"]`
7. reload is blocked or confirmed when local unsynced edits exist
8. reload aborts on `obs_names` mismatch
9. stale classifier worker results are ignored after reload

## Deferred Work

The following are intentionally deferred beyond MVP phase 6:

- full `SpatialData` reload from disk
- viewer reconstruction
- table discovery refresh when new tables are added or removed on disk
- support for row-count changes
- support for reordered rows
- diff or merge workflows between local and disk state
- ROI-aware reload semantics

## Final Recommendation

Phase 6 should be implemented as a targeted current-table reload with strong guards.

The safest mental model is:

- disk snapshot -> validated replacement of `sdata.tables[table_name]`
- then regenerate `layer.metadata["adata"]` from that refreshed table

This gives Harpy one authoritative in-memory table and one derived napari cache, which is the cleanest way
to avoid an out-of-sync state.
