# Viewer Widget Tickets

## Purpose

This document turns
[viewer_widget.md](/Users/arne.defauw/VIB/napari_harpy/Roadmap/viewer_widget/viewer_widget.md)
into concrete implementation tickets for the first version of the Harpy viewer
workflow.

The tickets are ordered for incremental delivery:

1. shared `HarpyAppState`
2. `_viewer_adapter`
3. viewer widget skeleton
4. feature-extraction integration
5. object-classification integration

## First-Version Ground Rules

For the first implementation:

- Harpy assumes exactly one loaded `SpatialData` object
- the only shared cross-widget state is:
  - `sdata`
  - viewer adapter / layer-binding services
- widget selections remain local
- layer metadata is lightweight and non-authoritative

## Suggested Delivery Order

1. VW-01
2. VW-02
3. VW-03
4. VW-04
5. VW-05

## VW-01: Add Shared `HarpyAppState`

### Goal

Introduce a very small shared app-state object so all Harpy widgets can access
the same loaded `SpatialData` object, without putting widget discovery or
loading logic into shared state.

### Scope

- add a shared app-state module
- store one loaded `SpatialData`
- provide a stable way for multiple dock widgets to access the same state for
  a napari viewer
- provide a small API for setting and clearing the loaded `SpatialData`
- add a minimal public `Interactive(sdata)` entrypoint so it is clear how the
  app should be launched programmatically
- keep `Interactive(sdata)` thin; it should initialize app state and Harpy UI,
  not own element discovery logic

### Required work

- add a new module, for example:
  - `src/napari_harpy/_app_state.py`
- define `HarpyAppState`
- suggested fields:
  - `viewer`
  - `sdata: SpatialData | None`
- add helper API such as:
  - `get_or_create_app_state(napari_viewer)`
  - `set_sdata(sdata)`
  - optional `clear_sdata()`
- add a simple change notification path so widgets can react when `sdata`
  changes:
  - use a `QObject`-based `HarpyAppState` with an `sdata_changed` signal
- ensure the same viewer returns the same shared state object across widgets
- keep widget-specific selections out of shared app state
- keep coordinate-system discovery, labels/image listing, and UI population out
  of `HarpyAppState`; those belong in widgets and viewer services
- add a new module, for example:
  - `src/napari_harpy/_interactive.py`
- implement a thin public launcher, for example:
  - `Interactive(sdata, viewer=None, headless=False)`
- the launcher should:
  - get or create a napari viewer
  - get or create `HarpyAppState`
  - call `set_sdata(sdata)`
  - ensure Harpy dock widgets are present idempotently
  - use `viewer.window.add_plugin_dock_widget(...)` for Harpy dock widgets so
    the napari plugin manifest remains the source of truth
  - expose `run()` similar to `napari-spatialdata.Interactive`
- for the first version, the launcher can ensure the currently available Harpy
  dock widgets are present:
  - `Feature Extraction`
  - `Object Classification`
- later tickets can extend that launcher to add the dedicated `Viewer` widget
  once it exists, also via `add_plugin_dock_widget(...)`
- `Interactive(..., headless=False)` should auto-call `run()` by default
- `Interactive(..., headless=True)` should initialize state and widgets without
  starting the napari event loop
- `clear_sdata()` should only clear shared state and emit `sdata_changed`; it
  should not remove napari layers in `VW-01`

### Suggested files

- `src/napari_harpy/_app_state.py`
- `src/napari_harpy/_interactive.py`
- `src/napari_harpy/__init__.py`
- possibly small follow-up wiring in widget constructors to consume app state

### Acceptance criteria

- [x] Harpy has one shared app-state object per napari viewer
- [x] multiple widgets can access the same loaded `SpatialData`
- [x] `HarpyAppState` exposes an `sdata_changed` signal when the loaded
  `SpatialData` changes
- [x] no feature-extraction or object-classification selection state is stored in
  `HarpyAppState`
- [x] `HarpyAppState` does not own coordinate-system discovery, image/labels
  option listing, or napari dock-widget creation
- [x] Harpy exposes a programmatic entrypoint such as:
  - `from napari_harpy import Interactive`
  - `ui = Interactive(sdata)`
- [x] `Interactive` initializes shared app state with the provided `sdata`
- [x] `Interactive` is a thin facade and does not duplicate widget discovery logic
- [x] `Interactive(sdata, headless=False)` auto-starts napari
- [x] `Interactive(sdata, headless=True)` leaves the viewer initialized without
  entering the event loop
- [x] repeated launcher calls do not create duplicate Harpy dock widgets

### Depends on

- none

## VW-02: Add `_viewer_adapter` And Layer Binding Registry

### Goal

Create a dedicated viewer adapter that owns napari layer loading, layer lookup,
activation, and minimal layer-binding state.

### Scope

- add viewer-facing load/activate helpers
- add a layer-binding registry
- add shared viewer services to `HarpyAppState`
- keep metadata lightweight
- support both labels loading and image loading
- support stack image mode and overlay image mode

### Required work

- add a new module:
  - `src/napari_harpy/_viewer_adapter.py`
- extend `HarpyAppState` with:
  - `viewer_adapter`
  - `layer_bindings`
- move napari-specific layer creation and lookup responsibilities here rather
  than expanding `_spatialdata.py`
- add a layer-binding model or equivalent internal mapping
- track at least:
  - layer -> `element_name`
  - layer -> `element_type`
  - layer -> `coordinate_system`
- add helpers such as:
  - `get_loaded_labels_layer(label_name)`
  - `get_loaded_image_layers(image_name)`
  - `ensure_labels_loaded(sdata, label_name, coordinate_system)`
  - `ensure_image_loaded(...)`
  - `activate_layer(layer)`
- treat `viewer.layers` as the source of truth for what is currently loaded in
  napari
- keep `LayerBindingRegistry` synchronized with viewer removal events:
  - listen to `viewer.layers.events.removed`
  - call `unregister_layer(...)` when a Harpy-managed layer is removed from
    napari
- support image loading modes:
  - stack image mode:
    - one napari `Image` layer
  - overlay mode:
    - one napari `Image` layer per selected channel
    - additive blending
    - user-selected colormaps / colors
- add a small number of targeted real-viewer integration tests once the adapter
  loading paths are stable, especially for:
  - `Interactive(...)` plus dock-widget behavior
  - real layer insertion / activation behavior
  - overlay image loading paths
- attach only lightweight metadata when useful:
  - `element_name`
  - `element_type`
  - `coordinate_system`

### Suggested files

- `src/napari_harpy/_viewer_adapter.py`
- possibly `src/napari_harpy/_app_state.py`
- possibly shared utility modules
- `tests/test_viewer_adapter.py`
- possibly a small real-viewer integration test module

### Acceptance criteria

- [x] Harpy can load a labels element from `sdata` into napari without relying on
  `napari-spatialdata`
- [x] Harpy can load an image from `sdata` into napari in stack mode
- [x] Harpy can load selected image channels in overlay mode
- [x] the adapter can answer whether a requested labels layer is already loaded
- [x] metadata stays lightweight and no authoritative analysis state is stored on
  layers
- [x] the registry stays synchronized when a user removes a Harpy-managed layer
  from napari
- [x] Harpy has a small number of real-viewer integration tests covering the
  most important viewer-specific adapter flows

### Depends on

- VW-01

## VW-03: Add Viewer Widget Skeleton

### Goal

Introduce the new `Viewer` widget and wire it to the shared app state and
viewer adapter.

### Scope

- add the widget class
- register it in the napari manifest
- support opening one `SpatialData` store
- expose coordinate-system-first browsing
- expose labels/image selection and basic load actions

### Required work

- add a new widget file:
  - `src/napari_harpy/widgets/_viewer_widget.py`
- add a dock-widget command in:
  - `src/napari_harpy/napari.yaml`
- add `Open SpatialData`
- load the selected store into `HarpyAppState.sdata`
- subscribe to `HarpyAppState.sdata_changed` so the viewer widget refreshes its
  available coordinate systems, labels, images, channels, and linked tables
  when the loaded `sdata` changes
- expose all coordinate systems from the loaded `sdata`
- make coordinate system selection required before element selection
- after coordinate-system selection:
  - show labels available in that coordinate system
  - show images available in that coordinate system
- for labels:
  - show correctly linked tables
  - add `Add / Update in viewer`
- for images:
  - show one image card per image in the selected coordinate system
  - show channel names
  - show display mode controls:
    - stack
    - overlay
  - show per-channel color controls for overlay mode
  - add `Add / Update in viewer`
- use the viewer adapter for all napari loading actions

### Suggested files

- `src/napari_harpy/widgets/_viewer_widget.py`
- `src/napari_harpy/widgets/__init__.py`
- `src/napari_harpy/napari.yaml`
- `tests/test_viewer_widget.py`

### Acceptance criteria

- [x] the new `Viewer` dock widget is available in napari
- [x] the widget can open a `SpatialData` store
- [x] changing the loaded `sdata` updates the viewer widget through
  `sdata_changed`
- [x] the widget lists coordinate systems from the loaded `sdata`
- [x] labels and images are filtered by the selected coordinate system
- [x] the widget can load a selected labels element into napari
- [x] the widget can load a selected image in stack mode
- [x] the widget can load selected channels in overlay mode

### Depends on

- VW-01
- VW-02

## VW-04: Integrate Feature Extraction With Shared `sdata`

### Goal

Refactor the feature-extraction widget so it derives options from the shared
loaded `SpatialData` instead of viewer scanning, while keeping feature
selection explicit and local to the feature-extraction widget.

### Scope

- read `sdata` from `HarpyAppState`
- remove viewer-scanning as the primary source of options
- make selection flow coordinate-system-first
- expose explicit extraction channel selection
- keep extraction channel state separate from viewer display channel state

### Required work

- update the feature-extraction widget to use shared app state
- subscribe to `HarpyAppState.sdata_changed` so the widget refreshes its
  available options when the loaded `sdata` changes
- when no `sdata` is loaded:
  - show a clear empty-state message
  - disable calculation
- expose:
  - all coordinate systems from the loaded `sdata`
  - labels in the selected coordinate system
  - images in the selected coordinate system
  - channels for the selected image
  - correctly linked tables for the selected labels element
- keep feature-extraction channel selection local to this widget
- ensure the calculation path uses:
  - selected coordinate system
  - selected labels element
  - selected image element
  - selected extraction channels
  - selected table
- do not derive extraction channels from viewer display state
- preserve the existing controller-centric calculation flow where possible
- remove or repurpose `Rescan Viewer` semantics if they are no longer the right
  primary UX

### Suggested files

- `src/napari_harpy/widgets/_feature_extraction_widget.py`
- `src/napari_harpy/_feature_extraction.py`
- `src/napari_harpy/_spatialdata.py`
- `tests/test_feature_extraction_widget.py`
- possibly new tests for channel-selection state and `sdata`-driven binding

### Acceptance criteria

- [x] feature extraction options come from the shared loaded `sdata`
- [x] changing the loaded `sdata` refreshes the widget through `sdata_changed`
- [x] coordinate-system selection happens inside the feature-extraction widget
- [x] labels and images are filtered to the selected coordinate system
- [x] tables are filtered to those correctly linked to the selected labels element
- [x] extraction channels are explicitly selected in the feature-extraction widget
- [x] extraction does not depend on viewer display channel choices

### Depends on

- VW-01
- VW-02
- VW-03

## VW-05: Integrate Object Classification With Shared `sdata`

### Goal

Refactor the object-classification widget so it derives options from the shared
loaded `SpatialData` and automatically ensures the selected labels layer is
loaded and activated in napari.

### Scope

- read `sdata` from `HarpyAppState`
- expose segmentation masks from `sdata`
- auto-load the chosen labels layer when needed
- expose correctly linked tables and feature matrices
- keep annotation dependent on a live napari labels layer

### Required work

- update the object-classification widget to use shared app state
- subscribe to `HarpyAppState.sdata_changed` so the widget refreshes its
  available segmentations, linked tables, and feature matrices when the loaded
  `sdata` changes
- when no `sdata` is loaded:
  - show a clear empty-state message
  - disable annotation/classifier actions
- expose segmentation masks from the loaded `sdata`
- when a segmentation is selected:
  - load it through the viewer adapter if not already loaded
  - activate the labels layer automatically
- expose only correctly linked tables for the selected segmentation
- expose feature matrices from the selected linked table
- keep pick-based annotation bound to the live napari labels layer
- adjust status messaging so the widget explains when it is loading/activating
  the labels layer for the user
- remove or repurpose `Rescan Viewer` semantics if they are no longer the right
  primary UX

### Suggested files

- `src/napari_harpy/widgets/_object_classification_widget.py`
- `src/napari_harpy/_annotation.py`
- `src/napari_harpy/_classifier_viewer_styling.py`
- `src/napari_harpy/_spatialdata.py`
- `tests/test_widget.py`
- possibly new viewer-adapter integration tests

### Acceptance criteria

- [ ] object-classification segmentation choices come from the shared loaded `sdata`
- [ ] changing the loaded `sdata` refreshes the widget through `sdata_changed`
- [ ] choosing a segmentation auto-loads it into napari if needed
- [ ] the loaded labels layer is activated and ready for annotation
- [ ] linked tables are filtered to those correctly linked to the chosen
  segmentation
- [ ] feature matrices come from the selected linked table
- [ ] annotation remains disabled until a compatible labels layer is actually bound

### Depends on

- VW-01
- VW-02
- VW-03

## VW-06: Reassess Shared SpatialData Selection Option Models

### Goal

Clean up the shared selection-model layer after both `VW-04` and `VW-05` are
complete, so we can decide whether `SpatialDataLabelsOption` and
`SpatialDataImageOption` should remain as shared cross-widget models or be
renamed/refined.

### Scope

- revisit `SpatialDataLabelsOption`
- revisit `SpatialDataImageOption`
- keep this as a code-cleanup / architecture ticket rather than a user-facing
  feature
- avoid churn during `VW-04`; do this only once both widgets have migrated

### Required work

- review how these dataclasses are used after `VW-04` and `VW-05`
- confirm whether they still provide the right shared abstraction for:
  - widget-local selection state
  - stable selection identity across refreshes
  - `display_name` handling
  - coordinate-system metadata
- decide whether to:
  - keep them as-is
  - rename them
  - split them into more explicit models
  - reduce/reshape their fields
- remove any now-obsolete viewer-scanning assumptions that are still embedded
  in their naming or docstrings

### Suggested files

- `src/napari_harpy/_spatialdata.py`
- `src/napari_harpy/widgets/_feature_extraction_widget.py`
- `src/napari_harpy/widgets/_object_classification_widget.py`
- possibly related tests in:
  - `tests/test_feature_extraction_widget.py`
  - `tests/test_widget.py`
  - `tests/test_spatialdata.py`

### Acceptance criteria

- [x] the codebase has a deliberate post-migration decision on whether
  `SpatialDataLabelsOption` / `SpatialDataImageOption` remain the shared
  selection models
- [x] any renaming or reshaping is reflected consistently across both widgets
- [x] their responsibilities and naming match the post-`VW-05` architecture

### Depends on

- VW-04
- VW-05

## VW-07: Add Table-Driven Labels Coloring In The Viewer Widget

### Goal

Allow the Harpy `Viewer` widget to color loaded labels layers directly from a
selected linked table column, so table annotations can be inspected visually
without switching to a separate workflow first.

### Scope

- extend the `Viewer` widget with table-driven labels coloring controls
- use linked-table information already exposed for labels elements
- apply coloring to the loaded napari labels layer for the selected
  segmentation
- keep the viewer adapter / shared `sdata` architecture intact

### Required work

- add a labels-coloring control path in `ViewerWidget`
- expose available linked tables for the selected segmentation mask
- expose selectable table columns suitable for coloring
- load or reuse the labels layer through `ViewerAdapter`
- apply table-driven coloring/styling to the active labels layer
- define how the widget behaves when:
  - no linked table exists
  - no compatible column exists
  - the labels layer is not yet loaded

### Suggested files

- `src/napari_harpy/widgets/_viewer_widget.py`
- `src/napari_harpy/_classifier_viewer_styling.py`
- `src/napari_harpy/_spatialdata.py`
- `tests/test_viewer_widget.py`
- possibly new viewer-styling tests

### Acceptance criteria

- [ ] the `Viewer` widget can expose linked-table-driven coloring controls for a labels element
- [ ] a loaded labels layer can be colored from a selected linked-table column
- [ ] the coloring flow works with the shared loaded `sdata`
- [ ] the viewer widget handles missing linked tables / incompatible columns gracefully

### Depends on

- VW-01
- VW-02
- VW-03

## VW-08: Clarify Shared App-State Lifecycle And Dataset-Switching Policy

### Goal

Keep the current `HarpyAppState` / `ViewerAdapter` architecture, but make the
remaining lifecycle rules explicit so the shared-state design stays clean as
the plugin grows.

### Scope

- treat this as an architecture / code-cleanup ticket, not a user-facing
  feature
- keep the current per-viewer shared-state model
- clarify what happens when the loaded `SpatialData` object changes
- reassess the boundary between:
  - shared state
  - viewer services
  - widget-local state

### Why this ticket exists

The current design is already a strong improvement over viewer scanning:

- one shared `HarpyAppState` per napari viewer
- one shared loaded `sdata`
- widgets synchronized through `sdata_changed`
- `ViewerAdapter` as the central viewer-facing layer service
- `LayerBindingRegistry` as the authoritative in-memory mapping for
  Harpy-managed napari layers

The main remaining architectural question is lifecycle policy, especially:

- what should happen to existing Harpy-managed layers when `set_sdata(...)`
  switches to a different dataset?
- should that policy live explicitly in `HarpyAppState`?
- should `HarpyAppState` continue to own both shared state and service wiring,
  or should those responsibilities be separated more clearly?

### Required work

- document the intended lifecycle of `HarpyAppState`
- make the dataset-switching policy explicit, including whether old
  Harpy-managed layers and bindings are:
  - kept
  - cleared
  - or handled conditionally
- review whether `HarpyAppState` should continue to hold:
  - `sdata`
  - `layer_bindings`
  - `viewer_adapter`
  or whether some of that wiring should move elsewhere
- confirm that widget-local selection state stays out of shared app state
- review whether any remaining code still depends on implicit assumptions about
  `id(sdata)` as the only runtime identity
- update tests and module docstrings so the chosen lifecycle policy is obvious

### Suggested files

- `src/napari_harpy/_app_state.py`
- `src/napari_harpy/_viewer_adapter.py`
- `src/napari_harpy/_interactive.py`
- `src/napari_harpy/widgets/_viewer_widget.py`
- `tests/test_app_state.py`
- `tests/test_viewer_adapter.py`
- possibly small integration tests covering dataset switches

### Acceptance criteria

- [ ] the codebase has an explicit policy for what happens when shared `sdata`
  is replaced or cleared
- [ ] the responsibilities of `HarpyAppState` versus `ViewerAdapter` are
  documented and intentional
- [ ] widget-local selection state remains outside shared app state
- [ ] the chosen lifecycle policy is covered by tests
- [ ] module docstrings/comments reflect the final architecture clearly

### Depends on

- VW-01
- VW-02
- VW-03
- VW-04
- VW-05

## VW-09: Allow Feature Extraction To Create A Linked Table

### Goal

Allow `FeatureExtractionWidget` to create a new linked table when the selected
segmentation currently has no usable annotation table.

### Scope

- extend `FeatureExtractionWidget` with an explicit table-creation flow
- keep the existing shared-`sdata`, coordinate-system-first selection model
- support creating a new table only when the selected segmentation has no
  usable linked table
- keep the completed slice-7 feature-matrix synchronization model intact

### Why this ticket exists

Today `FeatureExtractionWidget` stops at a warning when the selected
segmentation has no linked table:

- the widget already tells the user that table creation is "coming soon"
- there is no explicit flow yet for creating and binding that table
- future local widget table-context refresh flows, such as creating a table or
  relinking to a newly created table, are the main reason the controller still
  keeps a separate local `on_table_state_changed` refresh hook

This work was considered during `VW-05`, but it is now being tracked as a
separate follow-up rather than part of the object-classification migration.

### Required work

- replace the current dead-end warning with an explicit table-creation path
  inside `FeatureExtractionWidget`
- require an explicit user action to create the table; do not silently create
  one as a side effect of simply selecting a segmentation
- let the user review or choose the new table name before creation
- create the table in the selected shared in-memory `sdata`
- link the new table to the selected segmentation so the normal
  table-binding helpers recognize it
- after successful table creation:
  - refresh local table context in `FeatureExtractionWidget`
  - bind the widget to the newly created table
  - continue to use the normal feature-extraction write path so later
    `.obsm[feature_key]` writes still flow through the shared
    `feature_matrix_written` event
- keep the first version focused on table creation from `Feature Extraction`;
  broader table-management or relinking workflows can remain future work

### Suggested files

- `src/napari_harpy/widgets/_feature_extraction_widget.py`
- `src/napari_harpy/_feature_extraction.py`
- `src/napari_harpy/_spatialdata.py`
- possibly `src/napari_harpy/_app_state.py`
- `tests/test_feature_extraction_widget.py`
- `tests/test_feature_extraction.py`
- possibly a focused new test module for table-creation flow

### Acceptance criteria

- [ ] when the selected segmentation has no linked table, the widget exposes an
  explicit table-creation action instead of only a passive warning
- [ ] table creation is explicit and user-visible, with clear feedback about
  what table will be created and which segmentation it will be linked to
- [ ] after table creation, `FeatureExtractionWidget` refreshes local table
  options and binds the new table without requiring reopening or manual refresh
- [ ] after table creation, running feature extraction writes the requested
  feature matrix into the new table
- [ ] the new table is linked to the selected segmentation so the normal
  table-binding helpers recognize it
- [ ] shared dirty-table semantics remain correct after table creation and
  later feature extraction writes
- [ ] the flow remains compatible with the completed slice-7 behavior:
  creating the table is a local widget table-context refresh, while later
  feature-matrix writes are still propagated cross-widget through the shared
  `feature_matrix_written` event

### Depends on

- VW-01
- VW-04
- VW-05

## VW-10: Move Viewer Card Actions To Card-Local Feedback

### Goal

Make `ViewerWidget` action feedback appear close to the image or segmentation
card that triggered it, while keeping the existing global feedback area for
global viewer actions.

### Scope

- keep the global viewer feedback card for global actions such as:
  - loading a `SpatialData` store
  - dataset-level or coordinate-system-level errors
- add local feedback cards to image and segmentation cards for actions that
  originate from those cards
- reuse the shared `set_status_card(...)` helper from `_shared_styles.py`
- do not add a full legend, palette editor, or persistent notification system

### Why this ticket exists

`VW-07` introduced structured status-card feedback for the viewer, but the card
currently lives near the top of the scrollable widget. That is acceptable for a
single global "last action" message, but it can be easy to miss when the user
clicks `Add / Update in viewer` from a lower image or segmentation card.

For card-originated actions, feedback should feel attached to the object the
user just acted on:

- styled-overlay creation / update messages should appear on the relevant
  segmentation card
- palette fallback and categorical-coercion warnings should appear on that same
  segmentation card
- stack / overlay image loading messages should appear on the relevant image
  card
- global feedback should remain available for global actions and failures that
  are not naturally tied to a card

### Required work

- add a hidden local feedback `QLabel` to `_LabelsCardWidget`
- add a hidden local feedback `QLabel` to `_ImageCardWidget`
- add small card methods such as:
  - `set_feedback(...)`
  - `clear_feedback()`
- wire `ViewerWidget` card signals so the parent knows which card emitted the
  action
- route card-originated feedback through the emitting card when available
- keep `_set_action_feedback(...)` as the global feedback path for global
  viewer actions
- clear stale card feedback when cards are rebuilt after changing
  `SpatialData`, coordinate system, image selection context, or labels context
- keep feedback styling centralized through `set_status_card(...)`

### Suggested files

- `src/napari_harpy/widgets/_viewer_widget.py`
- `src/napari_harpy/widgets/_shared_styles.py` only if a tiny helper extension
  is needed
- `tests/test_viewer_widget.py`

### Acceptance criteria

- [ ] clicking a segmentation card's `Add / Update in viewer` shows feedback on
  that segmentation card
- [ ] styled-overlay palette-source and categorical-coercion messages are
  visible on the relevant segmentation card
- [ ] clicking an image card's `Add / Update in viewer` shows feedback on that
  image card
- [ ] successful card-local actions do not need to populate the global feedback
  card
- [ ] global actions, such as `Load SpatialData`, still use the global feedback
  card
- [ ] changing coordinate system or rebuilding cards clears stale local card
  feedback
- [ ] tests cover card-local success, warning, info, and error feedback paths

### Depends on

- VW-07

## Nice-To-Have Follow-Ups

- add image visibility toggles and layer grouping behavior
- add contrast presets for image channels
- support more than one loaded `SpatialData` object
