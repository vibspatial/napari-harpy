# Phase 6 Tickets: Reload From Zarr

## Purpose

This document turns [spec_phase_6.md](/Users/arne.defauw/VIB/napari_harpy/Roadmap/spec_phase_6.md) into
concrete implementation tickets for Phase 6.

The tickets are ordered to support incremental delivery and low-risk implementation.

## Guiding Rule

For phase 6, the authoritative in-memory table remains:

- `sdata.tables[table_name]`

Harpy intentionally does not try to regenerate the `napari-spatialdata`
table cache stored on the napari layer as:

- `layer.metadata["adata"]`

That cache is derived viewer state owned by `napari-spatialdata`, and it may be
overwritten there again from `sdata[table_name]`. Phase 6 therefore focuses on
reloading the selected in-memory table and refreshing Harpy-owned UI and
styling around that authoritative state.

## Status Snapshot

- Completed: `P6-01`, `P6-02`, `P6-03`
- Removed from scope: former `P6-04` cache-regeneration work
- Partially completed ahead of the original order: `P6-07`, `P6-08`
- Remaining core work: `P6-05`, `P6-06`, then finish `P6-07` and `P6-08`

## Suggested Delivery Order

1. P6-01
2. P6-02
3. P6-03
4. P6-05
5. P6-06
6. P6-07
7. P6-08

## P6-01: Add Reload-Specific Table IO API

### Status

Completed.

### Goal

Create a dedicated disk -> memory API for the selected table that is clearly separate from the current
memory -> disk sync path.

### Scope

- extend the persistence layer or add a dedicated reload helper
- expose reload-specific capabilities
- keep sync and reload as separate code paths

### Required work

- add `can_reload`
- add `selected_table_store_path`
- add a method to read a disk snapshot of the selected table
- ensure this method reads only:
  - `obs`
  - `obsm`
  - `uns`
- continue using the existing selected `sdata` and `table_name` binding pattern

### Suggested files

- `src/napari_harpy/_persistence.py`
- possibly `src/napari_harpy/_spatialdata.py`

### Acceptance criteria

- Harpy has a dedicated API for reload
- reload logic is not mixed into `sync_table_state()`
- the selected table zarr path is resolved through the same authoritative table path logic as sync

### Depends on

- none

## P6-02: Implement Disk Snapshot Read And Validation

### Status

Completed.

### Goal

Read the current table snapshot from zarr and reject unsafe partial reloads before mutating memory.

### Scope

- direct zarr read of the selected table group
- `anndata.io.read_elem` usage
- validation of row identity, shape, and required metadata

### Required work

- open the selected table group with `use_consolidated=False`
- read:
  - `obs`
  - `obsm`
  - `uns`
- validate:
  - `obs` is a `DataFrame`
  - `obsm` is a mapping
  - `uns` is a mapping
  - `uns["spatialdata_attrs"]` exists
  - current `n_obs` matches disk `n_obs`
  - current `obs_names` match disk `obs_names`
  - current `obs_names` order matches disk `obs_names` order
  - all `obsm` entries have leading dimension `n_obs`
  - the selected table still annotates the selected segmentation

### Suggested files

- `src/napari_harpy/_persistence.py`
- possibly `src/napari_harpy/_spatialdata.py`

### Acceptance criteria

- reload snapshot can be read without reloading the full `SpatialData`
- unsafe row mismatches fail before in-memory state is touched
- failure messages are specific and user-facing

### Depends on

- P6-01

## P6-03: Apply Validated Reload Snapshot To In-Memory Table

### Status

Completed.

### Goal

Replace the selected in-memory table state from the validated disk snapshot in a controlled in-place update.

### Scope

- update `sdata.tables[table_name]`
- keep `sdata` authoritative
- keep mutation lightweight and limited to the validated reload payload

### Required work

- define a snapshot payload object or equivalent internal structure
- apply the validated snapshot to the selected table without rebuilding the full `AnnData`
- replace:
  - `table.obs`
  - `table.obsm`
  - `table.uns`
- refresh the selected table reference inside `sdata.tables[table_name]`
- ensure `locate_element()` and path resolution still work after replacement

### Suggested files

- `src/napari_harpy/_persistence.py`

### Acceptance criteria

- successful reload updates the selected `sdata.tables[table_name]`
- all expected validation failures occur before any in-memory mutation
- Harpy controllers continue to read the refreshed table through `sdata[table_name]`

### Depends on

- P6-02

## P6-05: Add Dirty-State Tracking And Reload Decision Flow

### Status

Open.

### Goal

Prevent confusing loss of unsynced edits when the user requests reload.

### Scope

- track whether current table state differs from last synced/reloaded disk state
- define the reload decision flow before disk -> memory replacement

### Required work

- introduce a per-selected-table dirty flag
- mark dirty on:
  - annotation edits
  - classifier outputs written into table state
  - any other in-memory table mutation owned by Harpy
- clear dirty on:
  - successful sync
  - successful reload
- support three user decisions when reload is requested while dirty:
  - `Sync`
  - `Reload and discard local edits`
  - `Cancel`

### Suggested files

- `src/napari_harpy/_widget.py`
- `src/napari_harpy/_persistence.py`
- possibly controller files that mutate the table

### Acceptance criteria

- reload never silently discards unsynced changes
- sync followed by reload works as one user path
- cancel leaves the current in-memory state untouched

### Depends on

- P6-03

## P6-06: Freeze Async Classifier Work Around Reload

### Status

Open.

### Goal

Prevent stale background classifier work from writing into a freshly reloaded table.

### Scope

- cancel pending debounce
- cancel active worker
- ignore stale worker returns after reload

### Required work

- add a classifier-side API to freeze or reset work before reload
- stop pending retrain timers
- cancel active worker where possible
- guarantee stale results are ignored even if the worker returns late
- recompute classifier status against the reloaded table state
- do not auto-retrain after reload

### Suggested files

- `src/napari_harpy/_classifier.py`
- `src/napari_harpy/_widget.py`

### Acceptance criteria

- no stale worker result can overwrite reloaded data
- classifier status remains coherent after reload
- on-disk prediction columns remain visible after reload without forced retraining

### Depends on

- P6-03

## P6-07: Add Widget UI And Full Refresh Flow

### Status

Partially complete.

Already implemented:

- `Reload from zarr` button in the widget
- basic reload wiring to `PersistenceController.reload_table_state()`
- refresh of feature-matrix choices and selection status after reload
- refresh of layer styling and persistence feedback after reload

Still blocked on `P6-05` and `P6-06`:

- dirty-state confirmation flow
- classifier-safe reload behavior when async work is in flight

### Goal

Expose `Reload from zarr` in the widget and refresh all table-derived UI state after success.

### Scope

- add button
- hook button to reload flow
- refresh dependent UI pieces

### Required work

- add `Reload from zarr` button to the widget
- enable it only when a backed selected table can be reloaded
- wire in dirty-state confirmation behavior
- refresh after successful reload:
  - feature matrix dropdown
  - selected feature key fallback behavior
  - selection status
  - annotation feedback
  - sync/reload feedback
  - layer colors
  - classifier feedback
- preserve previously selected feature key if it still exists
- otherwise fall back to the first available key or clear the selection

### Suggested files

- `src/napari_harpy/_widget.py`

### Acceptance criteria

- the widget exposes `Reload from zarr`
- the reload flow is clearly separate from `Rescan Viewer` and `Sync to zarr`
- new on-disk `obsm` keys become visible after reload
- styling updates reflect reloaded `user_class`, `pred_class`, and `pred_confidence`

### Depends on

- P6-05
- P6-06

## P6-08: Add Phase 6 Test Coverage

### Status

Partially complete.

Already implemented:

- persistence tests for disk snapshot reads and validation failures
- persistence tests for in-place reload of `obs`, `obsm`, and `uns`
- widget test coverage for `Reload from zarr`
- widget test coverage that new on-disk `obsm` keys become visible after reload

Still missing:

- dirty-state / confirmation coverage from `P6-05`
- stale-classifier-result coverage from `P6-06`

### Goal

Lock in the reload semantics with controller-level and widget-level tests.

### Scope

- unit tests for snapshot reading and validation
- integration tests for widget reload behavior

### Required tests

- reload picks up a new `obsm` key from disk
- reload picks up changed `obs["user_class"]` from disk
- reload refreshes prediction-driven layer styling
- reload preserves selected feature key when still present
- reload falls back when selected feature key disappears
- reload is blocked or confirmed when local unsynced edits exist
- reload aborts on `obs_names` mismatch
- stale classifier worker results are ignored after reload

### Suggested files

- `tests/test_persistence.py`
- `tests/test_widget.py`
- possibly new targeted tests for reload helpers

### Acceptance criteria

- the critical phase 6 flows are covered by automated tests
- failing validation paths are tested explicitly
- UI-visible reload behavior is verified end to end

### Depends on

- P6-01
- P6-02
- P6-03
- P6-05
- P6-06
- P6-07

## Nice-To-Have Follow-Ups

These are not MVP phase 6 tickets, but should stay visible:

- support row-count changes during reload
- support row reorder reconciliation
- support table discovery refresh after external table add/remove
- support full-`SpatialData` reload when needed
- evaluate whether dirty tracking should become a general persistence-status model used beyond phase 6
