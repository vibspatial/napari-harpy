# Enable `Write Table State` Only For Dirty Tables

Status: specified.

## Goal

Make the Object Classification widget enable `Write Table State` only when the
selected backed table has unsynced in-memory changes.

This should reduce no-op writes and make the persistence controls communicate
state more clearly:

- clean backed table: `Write Table State` disabled;
- dirty backed table: `Write Table State` enabled;
- `Reload Table State` remains available for backed tables, because reload can
  still be useful even when the local table is clean.

## Current Behavior

The current behavior is effectively:

- `PersistenceController.can_sync` returns `True` when a SpatialData object and
  table are selected and the SpatialData object is backed by zarr;
- `ObjectClassificationWidget._update_persistence_controls()` sets
  `sync_button.setEnabled(can_sync)`;
- dirty state only affects tooltip text, not button enablement.

So yes: for a valid backed table selection, `Write Table State` is currently
clickable even when `PersistenceController.is_dirty` is `False`.

One nuance: if the selected table cannot validly annotate the selected labels
layer, `_bind_current_selection(...)` passes `effective_table_name=None` to the
persistence controller. In that case `can_sync` is already `False`, so the button
is disabled. The current issue is specifically the clean-but-valid backed table
case.

Dirty state already exists and is shared:

- `HarpyAppState` tracks dirty table keys;
- `PersistenceController.is_dirty` reads that shared state;
- annotation edits call `_mark_persistence_dirty()`;
- classifier result writes call `_mark_persistence_dirty()`;
- feature-matrix write events mark the affected table dirty;
- `PersistenceController.write_table_state()` clears dirty after a successful
  write;
- `PersistenceController.reload_table_state()` clears dirty after a successful
  reload.

Because this state already exists, we do not need a table diff or disk comparison
for this feature.

## Recommendation

This is a good UX change.

The button currently reads like an action the user should take, even when there
is nothing local to persist. Disabling it while clean makes the write/reload
controls easier to scan:

- `Write Table State` means "persist my local changes";
- `Reload Table State` means "replace local table state from disk";
- dirty state decides whether writing is actionable.

We should not change `PersistenceController.write_table_state()` itself to reject
clean writes. Programmatic callers and the dirty-reload flow can keep using the
write path as an idempotent persistence operation. This slice should primarily
change the widget action state.

## Implementation Plan

1. Keep `PersistenceController.can_sync` as the low-level capability check:
   "is there a selected backed table that can be written?"
2. Add a clearer UI-facing property to `PersistenceController`, for example:

   ```python
   @property
   def can_write_table_state(self) -> bool:
       return self.can_sync and self.is_dirty
   ```

   This avoids changing the meaning of `can_sync`, which is also useful for
   tooltips and lower-level persistence logic.
3. Update `ObjectClassificationWidget._update_persistence_controls()`:

   - compute `can_sync = self._persistence_controller.can_sync`;
   - compute `can_write = self._persistence_controller.can_write_table_state`;
   - keep `can_reload = self._persistence_controller.can_reload`;
   - set `self.sync_button.setEnabled(can_write)`;
   - keep `self.reload_button.setEnabled(can_reload)`.
4. Update the clean-backed-table tooltip:

   - if backed and clean, tooltip should explain that there are no unsynced local
     table changes to write;
   - if backed and dirty, keep the current destination tooltip and dirty warning;
   - if unbacked or no valid table is selected, keep the existing unavailable
     tooltips.

   Suggested clean tooltip:

   ```text
   The selected table has no unsynced local in-memory changes to write.
   ```

5. Confirm successful writes still disable the button:

   - `_write_selected_table_to_zarr(...)` calls
     `PersistenceController.write_table_state()`;
   - `write_table_state()` clears dirty;
   - `_write_selected_table_to_zarr(...)` calls `_update_selection_status()`;
   - `_update_selection_status()` refreshes persistence controls.

6. Confirm dirty transitions still enable the button:

   - annotation edits end with `_update_selection_status()`;
   - classifier table-state changes explicitly call `_update_persistence_controls()`;
   - feature-matrix write events call `_update_selection_status()`;
   - registration success calls `_update_selection_status()`.

## Tests

Update or add widget tests around the persistence controls:

- a valid backed table selection starts clean with `Write Table State` disabled
  and `Reload Table State` enabled;
- after an annotation edit, dirty state is `True` and `Write Table State` becomes
  enabled;
- after clicking `Write Table State`, dirty state is cleared and the button
  becomes disabled again;
- classifier prediction writes still mark the table dirty and enable
  `Write Table State`;
- registering feature metadata still marks the table dirty and enables
  `Write Table State`;
- feature-matrix write events still mark the table dirty and enable
  `Write Table State`;
- unbacked data keeps both persistence actions unavailable as today;
- invalid table binding keeps `Write Table State` unavailable as today.

Existing tests likely needing adjustment:

- `test_widget_enables_sync_for_backed_spatialdata` should become a clean-state
  test that expects `sync_button.isEnabled() is False` while reload remains
  enabled;
- `test_widget_syncs_user_class_to_backed_zarr` should expect the sync button to
  be disabled after the successful write, because the table is clean again;
- `test_widget_marks_persistence_dirty_on_annotation_change_and_clears_it_on_sync`
  should assert the button toggles enabled -> disabled alongside dirty state.

## Non-Goals

- Do not implement deep table diffing against disk.
- Do not disable `Reload Table State` just because the table is clean.
- Do not change lower-level `write_table_state()` into an error for clean writes.
- Do not change how dirty state is recorded in `HarpyAppState`.
