# Multiple Coordinate Systems

## Problem

Harpy currently lets each widget keep its own coordinate-system selection:

- `ViewerWidget` filters its image and segmentation cards by its local combo.
- `ObjectClassificationWidget` filters segmentations by its local combo and auto-loads the selected labels layer.
- `FeatureExtractionWidget` filters labels/images by its local combo for calculation.

That works for simple data with only `global`, but it breaks down once one
`SpatialData` object exposes multiple meaningful coordinate systems.

Examples:

- `global` and `micron_coordinate_system`
- `sample_1` and `sample_2`
- multiple acquisition regions represented as separate coordinate systems

In those cases, showing images and segmentation masks from multiple coordinate
systems in the same napari viewer is usually misleading. Harpy should treat the
coordinate system as a shared viewer context: one active coordinate system is
shown and operated on at a time.

## Current Findings

The existing architecture already has most of the right ingredients:

- `HarpyAppState` is the shared per-viewer state hub.
- `HarpyAppState.sdata_changed` already synchronizes loaded `SpatialData`.
- `ViewerAdapter` owns Harpy layer loading, lookup, activation, and removal.
- `LayerBindingRegistry` is already the authoritative in-memory mapping from
  napari layer objects to Harpy layer identity.
- `LayerBinding` already stores `coordinate_system` for images, primary labels,
  and styled labels overlays.

The missing piece is that coordinate-system selection is still widget-local.
Because of that:

- changing the coordinate system in `ViewerWidget` does not update object
  classification or feature extraction;
- changing it in object classification does not update the viewer widget;
- Harpy can leave old layers from a previous coordinate system visible;
- the registry can correctly know which coordinate system each layer belongs to,
  but no shared policy uses that information to enforce a single active view.

## Recommended Invariant

There should be one active coordinate system per `HarpyAppState`.

This active coordinate system means:

- widget selectors are synchronized to the same coordinate system;
- newly loaded Harpy-managed layers use that coordinate system;
- Harpy-managed layers from other coordinate systems are removed from the viewer;
- feature extraction and object classification bind against the same coordinate
  context that the viewer displays.

This does not mean Harpy has only one coordinate system in the data model. It
means the viewer has one active coordinate-system context at a time.

## Cross-Sample Tables

There is a second, related requirement when coordinate systems represent
different samples.

Example:

- coordinate system `sample_1` contains image and labels for sample 1;
- coordinate system `sample_2` contains image and labels for sample 2;
- one AnnData table annotates segmentation masks from both samples.

This is a valid and important workflow. The viewer should still show only one
coordinate system at a time, but the table may span multiple coordinate systems
and multiple labels regions.

This mainly affects:

- `FeatureExtractionWidget`
- `ObjectClassificationWidget`

It should not change the viewer invariant. The active coordinate system remains
the currently visible / interactive sample context.

## Multi-Region Feature Extraction

For feature extraction, avoid modeling this as two independent multi-select
lists:

- multiple segmentation masks
- multiple images

That would be ambiguous because the correct pairing matters. An image should be
paired with a segmentation mask only when both are registered in the same
coordinate system.

The better unit is a per-region extraction target:

```text
coordinate system / sample
segmentation mask
registered image
selected channels
shared AnnData table
```

The intended workflow should be:

1. user selects an AnnData table;
2. Harpy reads the table metadata and discovers all annotated labels regions;
3. Harpy derives candidate extraction targets from those regions;
4. for each target, Harpy offers images available in the same coordinate system;
5. user confirms which targets to calculate;
6. Harpy runs feature extraction per target;
7. Harpy writes all features back into the same AnnData table, aligned by
   `region_key` plus `instance_key`.

This means the table remains the authoritative cross-sample object table, while
feature extraction becomes a batch of per-region jobs.

Important behavior:

- do not run a Cartesian product of all selected labels and all selected images;
- require an explicit labels/image pairing per coordinate system;
- validate that each selected labels element is actually annotated by the table;
- validate duplicate `instance_key` values within each `region_key` region, not
  globally across the entire table;
- write feature rows back to the matching table rows for that labels region.

Suggested UI direction:

- keep the current single-coordinate-system mode as the simple path;
- add an explicit batch / multi-region mode later;
- in batch mode, show a target list grouped by coordinate system;
- each row represents one labels region and its selected image;
- default image selection can be inferred only when exactly one registered image
  is available for that labels region's coordinate system;
- otherwise require the user to choose the image for that target.

## Object Classification With Cross-Sample Tables

Object classification is less problematic.

The user will usually annotate one sample at a time because the viewer only has
one active coordinate system. That is fine. The selected AnnData table can still
contain rows for multiple samples or labels regions.

Recommended behavior:

- object classification remains bound to the active coordinate system;
- annotation writes update the shared AnnData table rows for the active
  segmentation region;
- table lookup continues to use `region_key` and `instance_key`;
- classifier training can use labeled rows from the whole selected table;
- classifier prediction scope should be configurable in the UI.

The first version should keep the interaction sample-by-sample:

- choose active coordinate system;
- choose segmentation in that coordinate system;
- annotate objects;
- write to the shared table;
- switch coordinate system;
- continue annotating another sample into the same table.

This avoids showing multiple samples in the viewer while still allowing the
table to accumulate annotations across samples.

Open design question:

- Should classifier prediction be table-wide by default, or scoped to the active
  segmentation region?

Recommendation for now:

- training can use the whole selected table by default;
- prediction scope should be an explicit object-classification setting;
- the first UI can offer:
  - active segmentation only
  - all regions in selected table
- default to active segmentation only if we want the safer viewer-local behavior;
- default to all regions if we want the classifier to behave as a table-level
  batch prediction tool.

The UI setting should make the write scope visible before predictions are
generated, because table-wide prediction can modify rows that are not currently
visible in the viewer.

## App-State Signal

Yes, adding a coordinate-system signal to `HarpyAppState` is the right direction.

Use a signal on shared app state, ideally with a small event object rather than
emitting a bare string:

```python
@dataclass(frozen=True)
class CoordinateSystemChangedEvent:
    sdata: SpatialData | None
    previous_coordinate_system: str | None
    coordinate_system: str | None
    source: str | None = None
```

Then add to `HarpyAppState`:

```python
coordinate_system_changed = Signal(object)
```

and shared state:

```python
self.coordinate_system: str | None = None
```

with an API like:

```python
def set_coordinate_system(
    self,
    coordinate_system: str | None,
    *,
    source: str | None = None,
) -> bool:
    ...
```

The method should:

1. normalize empty strings to `None`;
2. no-op if the normalized value is unchanged;
3. update `self.coordinate_system`;
4. emit `CoordinateSystemChangedEvent`;
5. ask `ViewerAdapter` to remove Harpy-managed layers outside the new coordinate
   system.

The event should be emitted before pruning layers so widgets can first update
their local selection state. This avoids object classification reacting to old
layer removals while it still thinks the old coordinate system is selected.

## Layer Removal Policy

On coordinate-system switch, Harpy should remove Harpy-managed layers whose
binding does not match the new active coordinate system.

Concretely, when switching to `next_coordinate_system`:

- remove registered layers where `binding.coordinate_system != next_coordinate_system`;
- keep registered layers where `binding.coordinate_system == next_coordinate_system`;
- if the new coordinate system is `None`, remove all Harpy-managed layers for
  the active `sdata`;
- leave unregistered napari layers alone.

The last point is important. Harpy should only remove layers it owns through
`LayerBindingRegistry`. External layers loaded manually or by another plugin
should not disappear just because Harpy changes coordinate systems.

For the first implementation, removal is better than hiding:

- it keeps napari visually unambiguous;
- it frees large image/labels data from the viewer;
- it avoids accidentally annotating hidden stale labels layers;
- it lets the existing viewer removal event path unregister bindings.

If we later want to preserve per-coordinate-system layer display settings, that
should be a separate display-intent cache, not the current live-layer registry.

## Registry Policy

Do not sync the registry to the active coordinate system by mutating bindings.

The registry should remain a registry of live Harpy-managed napari layers.
Each binding records the coordinate system in which that concrete layer was
created. If the layer is removed, the binding is removed. If the same element is
later shown in another coordinate system, Harpy creates a new napari layer and a
new binding.

Recommended additions to `ViewerAdapter`:

```python
def remove_layers_outside_coordinate_system(
    self,
    *,
    sdata: SpatialData | None,
    coordinate_system: str | None,
) -> list[LayerBinding]:
    ...
```

and possibly:

```python
def remove_layers_for_sdata(self, sdata: SpatialData | None) -> list[LayerBinding]:
    ...
```

Implementation detail:

- iterate over a snapshot of `layer_bindings.iter_bindings()`;
- filter by `sdata_id` when `sdata` is not `None`;
- remove matching layers through the existing `_remove_layer_from_viewer_and_registry(...)`;
- rely on the existing viewer `removed` event path to unregister bindings;
- keep the fallback unregister path for test/dummy viewers.

## Widget Responsibilities

### `ViewerWidget`

`ViewerWidget` should become both a source and a listener.

When the user changes the coordinate-system combo:

- call `self._app_state.set_coordinate_system(selected_coordinate_system, source="viewer_widget")`;
- do not directly refresh cards as the primary path;
- let `coordinate_system_changed` drive the refresh.

When `coordinate_system_changed` is received:

- update the combo with `QSignalBlocker`;
- refresh image and labels cards for the active coordinate system;
- clear or update coordinate-system-level feedback.

### `ObjectClassificationWidget`

`ObjectClassificationWidget` should also be both a source and a listener.

When the user changes its coordinate-system combo:

- publish the new coordinate system through `HarpyAppState`;
- then rely on the shared signal to refresh local segmentation/table/feature
  state.

When `coordinate_system_changed` is received:

- update the combo with `QSignalBlocker`;
- refresh segmentation options for the new coordinate system;
- preserve the selected segmentation only if it is still valid in the new
  coordinate system;
- refresh linked tables and feature matrices;
- rebind annotation, classifier, styling, and persistence controllers;
- auto-load/activate the selected labels layer only if a valid segmentation is
  still selected.

The controllers should not subscribe directly to the app-state signal. The
widget remains the place where UI selection is translated into controller
bindings.

### `FeatureExtractionWidget`

Even though the original symptom is viewer/object-classification mismatch,
feature extraction should also participate because it has its own coordinate
system selector.

When the user changes the feature-extraction coordinate-system combo:

- publish through `HarpyAppState`;
- let the shared signal refresh local labels/images/tables;
- let the existing controller `bind(...)` path cancel stale active work when
  the coordinate system changes.

When `coordinate_system_changed` is received:

- update the combo with `QSignalBlocker`;
- refresh labels, images, channels, and tables;
- rebind `FeatureExtractionController`.

This keeps calculation context aligned with the viewer context.

## `sdata` Lifecycle

Coordinate-system state should be reset or revalidated when the shared `sdata`
changes.

Recommended policy:

- `set_sdata(new_sdata)` clears the active coordinate system if the previous
  coordinate system is not available in `new_sdata`;
- when `sdata` is cleared, active coordinate system becomes `None`;
- when `sdata` changes, remove Harpy-managed layers from the previous dataset;
- widgets choose the first available coordinate system only when app state has
  no valid active coordinate system.

This keeps `HarpyAppState` from owning coordinate-system discovery details while
still keeping the active coordinate system coherent.

Possible first-version simplification:

- on every `set_sdata(...)`, clear `coordinate_system` to `None`;
- widgets refresh from `sdata_changed`;
- the first attached widget with available coordinate systems sets the first
  sorted coordinate system through `set_coordinate_system(...)`.

That is deterministic if all widgets use the same sorted helper,
`get_coordinate_system_names_from_sdata(...)`.

## Implementation Plan

### 1. Add shared coordinate-system state

Files:

- `src/napari_harpy/_app_state.py`
- `tests/test_app_state.py`

Work:

- add `CoordinateSystemChangedEvent`;
- add `coordinate_system_changed = Signal(object)`;
- add `self.coordinate_system`;
- add `set_coordinate_system(...)` and `clear_coordinate_system(...)`;
- decide whether `set_sdata(...)` clears coordinate system immediately or lets
  widgets revalidate it.

Acceptance:

- setting a new coordinate system emits one event;
- setting the same coordinate system is a no-op;
- clearing `sdata` clears coordinate system;
- event contains previous and next coordinate systems.

### 2. Add viewer-layer pruning by coordinate system

Files:

- `src/napari_harpy/_viewer_adapter.py`
- `tests/test_viewer_adapter.py`

Work:

- add `remove_layers_outside_coordinate_system(...)`;
- add focused tests with image, primary labels, and styled labels bindings;
- verify matching-coordinate layers remain;
- verify nonmatching-coordinate layers are removed and unregistered;
- verify unregistered layers are not touched.

Acceptance:

- switching active coordinate system leaves the registry containing only live
  Harpy-managed layers for the active coordinate system;
- no binding is mutated from one coordinate system to another.

### 3. Wire `ViewerWidget`

Files:

- `src/napari_harpy/widgets/_viewer_widget.py`
- `tests/test_viewer_widget.py`

Work:

- connect `self._app_state.coordinate_system_changed` in `__init__`;
- make combo changes call `app_state.set_coordinate_system(...)`;
- make the signal handler update the combo and refresh cards;
- on `sdata_changed`, populate combo choices around `app_state.coordinate_system`;
- if no active coordinate system is valid, select and publish the first
  available coordinate system.

Acceptance:

- changing the viewer widget coordinate system emits app-state change;
- cards refresh from the shared active coordinate system;
- layers from the old coordinate system are removed.

### 4. Wire `ObjectClassificationWidget`

Files:

- `src/napari_harpy/widgets/_object_classification_widget.py`
- `tests/test_widget.py`

Work:

- connect `coordinate_system_changed`;
- make local combo changes publish to app state;
- move local refresh logic into an app-state signal handler;
- make object classification preserve selected segmentation only when valid in
  the new coordinate system;
- ensure layer pruning does not incorrectly clear a still-valid selected
  segmentation after the widget has rebound to the new coordinate system.

Acceptance:

- changing coordinate system in object classification updates `ViewerWidget`;
- the selected labels layer from the old coordinate system is removed;
- if the selected segmentation is not available in the new coordinate system,
  annotation and classifier state are unbound;
- if the selected segmentation is available, it is loaded/activated in the new
  coordinate system only.

### 5. Wire `FeatureExtractionWidget`

Files:

- `src/napari_harpy/widgets/_feature_extraction_widget.py`
- `tests/test_feature_extraction_widget.py`

Work:

- connect `coordinate_system_changed`;
- make local combo changes publish to app state;
- refresh labels/images/tables from the shared coordinate system;
- rely on `FeatureExtractionController.bind(...)` to cancel stale jobs when the
  coordinate context changes.

Acceptance:

- changing coordinate system in feature extraction updates the other widgets;
- labels and images remain filtered to the shared active coordinate system;
- feature extraction cannot keep a stale coordinate system after viewer context
  changes.

### 6. Add cross-widget integration tests

Files:

- `tests/test_app_state.py`
- `tests/test_viewer_widget.py`
- `tests/test_widget.py`
- `tests/test_feature_extraction_widget.py`

Recommended scenarios:

- same viewer, `ViewerWidget` plus `ObjectClassificationWidget`: change in one
  updates the other;
- same viewer, all three widgets: all coordinate-system combos stay in sync;
- switching coordinate system removes old Harpy-managed image and labels layers;
- registry contains no stale bindings after coordinate switch;
- unregistered external layers remain in the viewer;
- clearing or replacing `sdata` clears active coordinate state.

### 7. Add multi-region feature-extraction design

Files:

- `src/napari_harpy/widgets/_feature_extraction_widget.py`
- `src/napari_harpy/_feature_extraction.py`
- `src/napari_harpy/_spatialdata.py`
- `tests/test_feature_extraction_widget.py`
- `tests/test_feature_extraction.py`

Work:

- add helpers that derive table-annotated labels regions from
  `SpatialDataTableMetadata.regions`;
- for each labels region, derive available coordinate systems;
- for each `(region, coordinate_system)`, list candidate images registered in
  the same coordinate system;
- model a batch extraction request as explicit per-region targets, not as
  independent labels/image selections;
- run feature extraction target by target and write into the same table;
- keep row alignment based on `region_key` and `instance_key`.

Acceptance:

- one AnnData table can receive features for labels regions from multiple
  coordinate systems;
- each feature-extraction target has one labels element and zero or one image
  in the same coordinate system;
- duplicate instance ids are rejected within a region, but the same instance id
  can appear in another region;
- no feature extraction run accidentally pairs a labels element with an image
  from a different coordinate system.

### 8. Add object-classification prediction-scope setting

Files:

- `src/napari_harpy/widgets/_object_classification_widget.py`
- `src/napari_harpy/_classifier.py`
- `tests/test_widget.py`
- `tests/test_classifier.py`

Work:

- add a prediction-scope UI control to `ObjectClassificationWidget`;
- support at least:
  - active segmentation only
  - all regions in selected table
- make the selected scope part of the classifier/controller binding;
- ensure active-region prediction only writes rows matching the selected
  `region_key`;
- ensure table-wide prediction can still use labels from all regions while
  writing predictions for all eligible table rows;
- surface status text that makes the current write scope clear.

Acceptance:

- users can choose whether classifier predictions update only the active sample
  / segmentation or the whole selected table;
- active-region prediction does not modify prediction columns for other table
  regions, except where existing classifier metadata explicitly requires a
  reset;
- table-wide prediction updates eligible rows across all regions in the selected
  table;
- the UI clearly communicates which prediction scope will be used.

## Open Questions

1. Should `HarpyAppState.set_sdata(...)` choose the default coordinate system
   centrally, or should widgets choose the first available coordinate system
   after `sdata_changed`?

   Recommendation: keep discovery in widgets for now, but make all widgets use
   the same sorted helper and publish the first available coordinate system only
   when app state has no valid active coordinate system.

2. Should switching coordinate systems remove layers or hide them?

   Recommendation: remove them for the first implementation. Add a separate
   display-state cache later if preserving per-coordinate-system display choices
   becomes important.

3. Should external/unregistered napari layers be removed?

   Recommendation: no. Only remove Harpy-managed registered layers. External
   layers are outside Harpy's ownership unless we explicitly adopt/register them.

4. For object classification, should classifier prediction be table-wide or
   active-region-only when the table spans multiple samples?

   Recommendation: make this configurable in the object-classification UI.
   Prediction write scope should be explicit because table-wide prediction can
   update rows that are not currently visible.

## Summary

`coordinate_system_changed = Signal(object)` in `HarpyAppState` is a good idea,
but it should represent a shared active viewer context, not just another widget
notification.

The registry should stay global and live-layer oriented. Do not mutate bindings
to match the active coordinate system. Instead, use the existing
`coordinate_system` stored on each binding to remove Harpy-managed layers that
do not belong to the newly active coordinate system.

All widgets with a coordinate-system selector should listen to and publish this
shared state. Controllers should stay passive and be rebound by their widgets.

For cross-sample AnnData tables, keep the viewer single-coordinate-system, but
let table-driven workflows span multiple labels regions. Feature extraction
should grow a batch mode based on explicit per-region labels/image targets.
Object classification can remain sample-by-sample in the viewer while updating
and learning from the shared table.
