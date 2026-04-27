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

## Relationship To Cross-Sample Tables

Cross-sample AnnData tables are a related but separate problem.

This document stays focused on shared viewer context:

- one active coordinate system per `HarpyAppState`;
- synchronized widget selectors;
- pruning Harpy-managed layers from inactive coordinate systems;
- keeping viewer state and controller bindings aligned with the active
  coordinate system.

Table-level behavior that spans multiple regions or samples is covered in:

- `Roadmap/multiple_coordinate_systems/cross_sample_tables.md`

That companion document keeps the same viewer invariant: one visible /
interactive coordinate system at a time.

## Testing Requirement

The testing gap for this work is real and should be treated as part of the
first implementation, not as follow-up coverage.

Current tests already prove useful pieces of the existing behavior, including:

- shared app-state identity for widgets attached to the same viewer;
- widget-local coordinate filtering in `ViewerWidget`;
- widget-local coordinate filtering in `ObjectClassificationWidget`;
- widget-local coordinate filtering in `FeatureExtractionWidget`.

What is still missing for this roadmap is coordinated phase-1 coverage for:

- `coordinate_system_changed` events on shared app state;
- bulk pruning of Harpy-managed layers by coordinate system;
- bulk pruning of Harpy-managed layers when `sdata` is replaced or cleared;
- cross-widget synchronization of coordinate-system selectors on the same
  viewer;
- registry cleanup guarantees after coordinate-system switches;
- protection of unregistered external napari layers during Harpy-owned cleanup.

These tests should be treated as a hard acceptance gate for phase 1, not as an
optional later test pass after implementation is already considered complete.

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

Signal/pruning order for a coordinate-system-only switch:

1. update `app_state.coordinate_system`;
2. emit `coordinate_system_changed`;
3. prune Harpy-managed layers outside the new coordinate system.

## Default Selection Ownership

`HarpyAppState` should own the default active coordinate system.

This should not be left to widgets, because widget-local fallback selection
would make the resulting active coordinate system depend on widget
initialization order and signal timing.

Chosen policy:

- `HarpyAppState.coordinate_system` is the authoritative active value;
- `HarpyAppState.set_sdata(...)` resolves the active coordinate system for the
  new dataset by using the shared helper
  `get_coordinate_system_names_from_sdata(...)` from `_spatialdata.py`;
- if the previous coordinate system is still available in the new `sdata`, keep
  it;
- otherwise select the first sorted coordinate system returned by the shared
  helper;
- if no coordinate systems are available, set the active coordinate system to
  `None`;
- widgets must not invent or publish a default coordinate system on
  `sdata_changed`;
- widgets call `set_coordinate_system(...)` only on explicit user action.

This also means the viewer widget should stop using its own local coordinate
system discovery helper and should instead rely on the same shared helper used
by app state.

## Layer Removal Policy

On coordinate-system switch, Harpy should remove Harpy-managed layers whose
binding does not match the new active coordinate system.

Concretely, when switching to `next_coordinate_system`:

- remove registered layers where `binding.coordinate_system != next_coordinate_system`;
- keep registered layers where `binding.coordinate_system == next_coordinate_system`;
- if the new coordinate system is `None`, remove all Harpy-managed layers for
  the active `sdata`;
- when replacing or clearing `sdata`, remove all Harpy-managed layers whose
  binding belongs to the previous `sdata`;
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

and:

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

- `set_sdata(new_sdata)` owns coordinate-system resolution for the new dataset;
- it uses `get_coordinate_system_names_from_sdata(new_sdata)` as the shared
  discovery helper;
- if the previous coordinate system is still available in `new_sdata`, keep it;
- otherwise set the active coordinate system to the first sorted available
  coordinate system;
- when `sdata` is cleared, active coordinate system becomes `None`;
- when `sdata` changes, remove all Harpy-managed napari layers from the previous
  dataset by using the registry's `sdata_id` binding metadata;
- widgets refresh around `app_state.coordinate_system` and do not choose a
  fallback locally.

The cleanup must happen before `sdata_changed` is emitted. Widgets already listen
to `sdata_changed` and immediately refresh/rebind local UI and controllers, so
`HarpyAppState.set_sdata(...)` or an adjacent app-state-owned helper should
first prune old registered layers, then assign `self.sdata`, resolve the active
coordinate system for the new dataset, and finally emit `sdata_changed(new_sdata)`.

Recommended first-version order:

1. keep `old_sdata = self.sdata`;
2. keep `old_coordinate_system = self.coordinate_system`;
3. remove Harpy-managed layers for `old_sdata` through
   `viewer_adapter.remove_layers_for_sdata(old_sdata)`;
4. assign `self.sdata = new_sdata`;
5. resolve `next_coordinate_system` with
   `get_coordinate_system_names_from_sdata(new_sdata)`;
6. keep `old_coordinate_system` if it is still valid in `new_sdata`;
7. otherwise choose the first sorted available coordinate system, or `None` if
   none exist;
8. update `self.coordinate_system`;
9. emit `coordinate_system_changed` if the active coordinate system changed;
10. emit `sdata_changed(new_sdata)`.

In short:

```text
coordinate-system switch:
    coordinate_system_changed -> prune old-coordinate layers

sdata switch:
    prune old-sdata layers -> resolve active coordinate system in app state -> sdata_changed(new_sdata)
```

This makes default selection deterministic and keeps widgets from racing to
choose their own fallback coordinate system.

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
- make `set_sdata(...)` resolve the default active coordinate system centrally
  by using `get_coordinate_system_names_from_sdata(...)`;
- make `set_sdata(...)` remove Harpy-managed layers for the previous `sdata`
  and update shared coordinate-system state before emitting `sdata_changed`.

Acceptance:

- [x] setting a new coordinate system emits one event;
- [x] setting the same coordinate system is a no-op;
- [x] `set_sdata(...)` keeps the previous coordinate system when still valid in the
  new dataset;
- [x] otherwise `set_sdata(...)` selects the first sorted available coordinate
  system, or `None` when none exist;
- [x] clearing `sdata` clears coordinate system;
- [x] replacing or clearing `sdata` removes registered napari layers belonging to
  the previous `sdata`;
- [x] event contains previous and next coordinate systems.

### 2. Add viewer-layer pruning by coordinate system

Files:

- `src/napari_harpy/_viewer_adapter.py`
- `tests/test_viewer_adapter.py`

Work:

- add `remove_layers_outside_coordinate_system(...)`;
- add `remove_layers_for_sdata(...)`;
- add focused tests with image, primary labels, and styled labels bindings;
- verify matching-coordinate layers remain;
- verify nonmatching-coordinate layers are removed and unregistered;
- verify all registered layers for a removed/replaced `sdata` are removed and
  unregistered;
- verify unregistered layers are not touched.

Acceptance:

- [x] switching active coordinate system leaves the registry containing only live
  Harpy-managed layers for the active coordinate system;
- [x] no binding is mutated from one coordinate system to another.

### 3. Wire `ViewerWidget`

Files:

- `src/napari_harpy/widgets/_viewer_widget.py`
- `tests/test_viewer_widget.py`

Work:

- connect `self._app_state.coordinate_system_changed` in `__init__`;
- replace local coordinate-system discovery with the shared helper from
  `_spatialdata.py`;
- make combo changes call `app_state.set_coordinate_system(...)`;
- make the signal handler update the combo and refresh cards;
- on `sdata_changed`, populate combo choices around `app_state.coordinate_system`;
- never select or publish a default coordinate system locally.

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
- refresh from `app_state.coordinate_system` on `sdata_changed` without
  choosing a widget-local default;
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
- refresh from `app_state.coordinate_system` on `sdata_changed` without
  choosing a widget-local default;
- rely on `FeatureExtractionController.bind(...)` to cancel stale jobs when the
  coordinate context changes.

Acceptance:

- changing coordinate system in feature extraction updates the other widgets;
- labels and images remain filtered to the shared active coordinate system;
- feature extraction cannot keep a stale coordinate system after viewer context
  changes.

### 6. Hard acceptance gate: add cross-widget integration tests

Files:

- `tests/test_app_state.py`
- `tests/test_viewer_widget.py`
- `tests/test_widget.py`
- `tests/test_feature_extraction_widget.py`

Recommended scenarios:

- same viewer, `ViewerWidget` plus `ObjectClassificationWidget`: change in one
  updates the other;
- same viewer, all three widgets: all coordinate-system combos stay in sync;
- replacing `sdata` keeps the previous coordinate system when it is still
  available;
- replacing `sdata` selects the first sorted available coordinate system when
  the previous one is no longer valid;
- switching coordinate system removes old Harpy-managed image and labels layers;
- registry contains no stale bindings after coordinate switch;
- unregistered external layers remain in the viewer;
- clearing or replacing `sdata` clears active coordinate state.

Acceptance:

- phase 1 is not complete until these tests exist and pass;
- the shared coordinate-system behavior is verified across app state, viewer
  layer cleanup, and widget synchronization rather than only through
  widget-local filtering tests.

## Open Questions

1. Should switching coordinate systems remove layers or hide them?

   Recommendation: remove them for the first implementation. Add a separate
   display-state cache later if preserving per-coordinate-system display choices
   becomes important.

2. Should external/unregistered napari layers be removed?

   Recommendation: no. Only remove Harpy-managed registered layers. External
   layers are outside Harpy's ownership unless we explicitly adopt/register them.

## Summary

`coordinate_system_changed = Signal(object)` in `HarpyAppState` is a good idea,
but it should represent a shared active viewer context, not just another widget
notification.

The registry should stay global and live-layer oriented. Do not mutate bindings
to match the active coordinate system. Instead, use the existing
`coordinate_system` stored on each binding to remove Harpy-managed layers that
do not belong to the newly active coordinate system.

`HarpyAppState` should also own default coordinate-system selection when `sdata`
changes, using the shared `get_coordinate_system_names_from_sdata(...)` helper.
Widgets should mirror that state and only publish coordinate-system changes in
response to explicit user interaction.

All widgets with a coordinate-system selector should listen to and publish this
shared state. Controllers should stay passive and be rebound by their widgets.

Cross-sample table behavior is intentionally out of scope here and is described
separately in `Roadmap/multiple_coordinate_systems/cross_sample_tables.md`.
