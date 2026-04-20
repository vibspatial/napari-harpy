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
- [ ] Harpy can load selected image channels in overlay mode
- [x] the adapter can answer whether a requested labels layer is already loaded
- [x] metadata stays lightweight and no authoritative analysis state is stored on
  layers
- [ ] the registry stays synchronized when a user removes a Harpy-managed layer
  from napari
- [ ] Harpy has a small number of real-viewer integration tests covering the
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
  - add `Show segmentation`
- for images:
  - show image choices
  - show channel names
  - show display mode:
    - stack
    - overlay
  - show per-channel color controls for overlay mode
  - add `Show image`
  - add `Show selected channels`
- use the viewer adapter for all napari loading actions

### Suggested files

- `src/napari_harpy/widgets/_viewer_widget.py`
- `src/napari_harpy/widgets/__init__.py`
- `src/napari_harpy/napari.yaml`
- `tests/test_viewer_widget.py`

### Acceptance criteria

- [ ] the new `Viewer` dock widget is available in napari
- [ ] the widget can open a `SpatialData` store
- [ ] changing the loaded `sdata` updates the viewer widget through
  `sdata_changed`
- [ ] the widget lists coordinate systems from the loaded `sdata`
- [ ] labels and images are filtered by the selected coordinate system
- [ ] the widget can load a selected labels element into napari
- [ ] the widget can load a selected image in stack mode
- [ ] the widget can load selected channels in overlay mode

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

- [ ] feature extraction options come from the shared loaded `sdata`
- [ ] changing the loaded `sdata` refreshes the widget through `sdata_changed`
- [ ] coordinate-system selection happens inside the feature-extraction widget
- [ ] labels and images are filtered to the selected coordinate system
- [ ] tables are filtered to those correctly linked to the selected labels element
- [ ] extraction channels are explicitly selected in the feature-extraction widget
- [ ] extraction does not depend on viewer display channel choices

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
- `src/napari_harpy/_viewer_styling.py`
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

## Nice-To-Have Follow-Ups

- add table-driven labels coloring directly in the viewer widget
- add image visibility toggles and layer grouping behavior
- add contrast presets for image channels
- support more than one loaded `SpatialData` object
- add a migration/cleanup ticket to remove now-obsolete viewer-scanning helpers
  after the new path is stable
