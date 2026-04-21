# Viewer Widget Roadmap

## Goal

Add a separate `Viewer` widget to `napari-harpy` that becomes the shared entry
point for:

- opening `SpatialData`
- choosing what should be shown in napari
- selecting multiplex image channels
- loading and activating image and segmentation layers
- exposing Harpy-owned `SpatialData` and layer services to downstream widgets

This widget is motivated by reuse.

We do not only need this for classical feature extraction.
We will likely want the same viewing and channel-selection workflow for:

- deep-learning feature extraction
- quality control / inspection
- future image-driven analysis tools
- future multi-sample browsing workflows

## Why A Separate Viewer Widget

The current repository has two task-oriented widgets:

- `Feature Extraction`
- `Object Classification`

Both currently discover `SpatialData` context from viewer-linked napari layers.
That was a good MVP path, but it mixes two concerns:

- task-specific logic
- dataset/image/view configuration

A dedicated `Viewer` widget gives us one reusable place for:

- image selection
- channel selection
- coordinate-system selection
- segmentation loading
- image loading
- viewer activation / visibility behavior

This keeps the task widgets narrower and makes the viewing flow reusable across
future features.

Another important reason is that `napari-spatialdata`'s default image-loading
behavior is not the exact UX we want for Harpy.

For multiplex images, `napari-spatialdata` currently loads the image as a
single napari `Image` layer unless the channels are interpreted as RGB(A).
That is a sensible generic browsing default, but it does not provide:

- explicit channel subset selection
- per-channel color assignment
- additive multiplex overlays for selected markers
- a clean separation between visualization-oriented channel choices and
  analysis-oriented channel choices

Harpy's viewer should therefore go beyond generic image loading and support a
channel-aware multiplex display workflow.

## Core Principle

For the first version, Harpy should assume that only one `SpatialData` object
is loaded at a time.

The `Viewer` widget should own:

- opening the active `SpatialData`
- viewer-local viewing context
- channel-selection UI
- loading data into napari

The task widgets should consume that context:

- `Feature Extraction` reads the shared loaded `SpatialData` and loaded-layer
  information, but keeps its own selection state
- `Object Classification` reads the shared loaded `SpatialData` and loaded-layer
  information, but keeps its own selection state

napari layers remain the visualization surface.
Layer metadata, if used at all, should stay lightweight and non-authoritative.

The intended coupling between widgets should be minimal.

What should be shared:

- the loaded `SpatialData` object
- layer-binding registry
- helper actions such as `ensure_labels_loaded(...)` and `ensure_image_loaded(...)`

What should remain widget-local:

- selected coordinate system
- selected image
- selected channels
- selected table
- feature-extraction inputs
- object-classification inputs

## Proposed User Flow

### 1. Viewer widget

The user:

- opens a `SpatialData` store
- chooses a coordinate system
- sees only the labels and images that are available in that coordinate system
- chooses a segmentation mask
- sees the tables that are correctly linked to that labels element
- loads that segmentation into napari
- chooses an image
- sees the channels for that image
- chooses one or more channels from that image
- loads or updates the image in napari to show exactly those selected channels

The viewer widget may attach lightweight identifiers to the napari layers it
loads, but the authoritative `SpatialData` object and selection state should
live in Harpy, not inside the layer metadata dictionary.

For multiplex images, the viewer should support two display modes:

- `Raw image mode`: load the selected image as a single napari `Image` layer
- `Overlay mode`: load only the selected channels as separate additive image
  layers with user-selected colors

The selection order should be explicit in the UI:

1. open a `SpatialData` store
2. choose coordinate system
3. choose either:
   - a labels element, then a linked table
   - an image element, then channels
4. choose how the selected element should be shown in napari

### 2. Feature extraction widget

The widget reads the currently loaded `SpatialData` object and shared layer
services and uses them to decide:

- which coordinate systems are available
- which segmentations are selectable in the chosen coordinate system
- which images are selectable in the chosen coordinate system
- which channels are selectable for the chosen image
- which tables are linked to the selected segmentation

Feature extraction then uses the underlying `SpatialData` elements, not the
rendered napari buffers, as the computation inputs.

Recommended feature-extraction flow:

1. choose a coordinate system from those available in the loaded `SpatialData`
2. choose a labels element available in that coordinate system
3. choose an image available in that coordinate system
4. choose the channels from that image to use for feature extraction
5. choose a correctly linked table for the selected labels element
6. choose feature families and output key

### 3. Object classification widget

The widget reads the same shared loaded `SpatialData` object and layer
services.

Extra requirement:

- the selected segmentation must be loaded as a napari `Labels` layer, because
  pick-based annotation depends on a live labels layer in the viewer

So object classification should validate both:

- the selected `SpatialData` / table context
- the presence of a compatible loaded labels layer

Recommended object-classification flow:

1. choose a segmentation mask from those available in the loaded `SpatialData`
2. load that segmentation into napari automatically if needed
3. activate the labels layer so it is ready for annotation
4. choose a correctly linked table for that segmentation
5. choose a feature matrix from the selected table

## Scope Of The Viewer Widget

### Required responsibilities

- open a `SpatialData` store
- keep track of the active loaded `SpatialData`
- discover labels, images, and coordinate systems
- choose the active coordinate system
- filter visible element choices by the active coordinate system
- choose the active segmentation
- expose only the tables that are correctly linked to the selected labels
  element
- choose the active image
- expose channel names when available
- expose fallback channel names such as `channel_0`, `channel_1`, ...
- load segmentation layers into napari
- load image layers into napari
- update the visible image channels in napari
- maintain enough Harpy-owned layer binding information so downstream widgets
  can resolve the originating `SpatialData` element names

### Labels-specific viewing controls

For labels, the viewer should expose controls that are separate from image
channel controls.

For the first version:

- `Labels`
- `Linked table`
- `Show segmentation`

Future-facing controls:

- `Color by obs column`
- `Color by gene / var feature`

The idea is similar to `napari-spatialdata`: once a labels element and a
correctly linked table are selected, the viewer should be able to color the
segmentation using table-derived values.

### Multiplex image display modes

The viewer should distinguish between two valid ways of showing multiplex
images.

#### 1. Raw image mode

Load the chosen image similarly to `napari-spatialdata`:

- one napari `Image` layer
- full channel stack preserved
- useful for generic browsing and quick inspection

#### 2. Overlay mode

Load a selected subset of channels as separate napari `Image` layers:

- one napari layer per selected channel
- one user-selected color / colormap per channel
- additive blending so the selected channels appear overlaid

This is likely the more important mode for Harpy because it supports explicit
marker visualization and aligns naturally with feature-extraction inputs.

### Useful but not required for the first version

- visibility toggles
- activate existing loaded layers instead of reloading duplicates
- layer naming rules for multiple datasets
- sample-aware browsing
- contrast presets
- RGB / grayscale display presets
- layer grouping conventions

### Non-goals for the first version

- replacing all of `napari-spatialdata`
- custom rendering inside the widget itself
- advanced viewer layout management
- implementing analysis logic in the viewer widget

## Minimal UI Sketch

Suggested first layout:

- `Open SpatialData`
- `Coordinate system`
- `Segmentation mask`
- `Linked table`
- `Show segmentation`
- `Image`
- `Display mode`
- `Channels`
- `Channel colors`
- `Show image`
- `Show selected channels`
- compact status summary

Example summary lines:

- `SpatialData: sample_01.zarr`
- `Coordinate system: global`
- `Segmentation: nuclei`
- `Linked table: table`
- `Image: multiplex_raw`
- `Display: overlay`
- `Channels: DAPI, CD3, CD8`

One important UI rule should be:

- users choose a coordinate system first
- only then do image and labels choices become available
- after choosing an image, channel controls become available
- after choosing a labels element, linked-table controls become available

## Architectural Sketch

## 1. Shared services vs widget-local state

We should introduce a very small shared app state object alongside the viewer
widget, while keeping most UI selections local to each widget.

Suggested concepts:

- `HarpyAppState`
- `LayerBindingRegistry`
- `ViewerSelectionState`
- `FeatureExtractionSelectionState`
- `ObjectClassificationSelectionState`

### Shared services

The following should be shared across widgets:

- `HarpyAppState`
- `LayerBindingRegistry`
- viewer adapter helper actions

This gives every widget access to:

- the currently loaded `SpatialData`
- the elements available in that `SpatialData`
- whether a requested labels or image layer is already loaded
- helper actions to load a required labels or image layer on demand

For the first version, `HarpyAppState` can stay very small.

Suggested fields:

- `sdata: SpatialData | None`
- `viewer_adapter`
- `layer_bindings`

### Widget-local state

The following should remain local to each widget.

#### Viewer widget

- selected coordinate system for viewing
- selected image for viewing
- selected labels element for viewing
- selected linked table for viewing
- selected display mode
- selected visible channels and colors

#### Feature Extraction widget

- selected coordinate system for feature extraction
- selected labels element
- selected image element
- selected feature-extraction channels
- selected feature families
- selected output key

#### Object Classification widget

- selected labels element
- selected linked table
- selected feature matrix
- selected coloring mode

This is intentionally weak coupling: widgets share access to the loaded
`SpatialData` and layer-loading services, but not each other's detailed
selections.

### Viewer-local selection state

Possible `ViewerSelectionState` fields:

- `coordinate_system`
- `label_name`
- `table_name`
- `image_name`
- `display_mode`
- `visible_channel_indices`
- `visible_channel_names`
- `visible_channel_colors`

Even if early implementation steps still use some viewer-linked metadata as a
compatibility bridge, the viewer widget should not hide all state in ad hoc
napari layer scanning. It should own an explicit state model and project that
state into napari.

### Feature-extraction-local selection state

Possible `FeatureExtractionSelectionState` fields:

- `coordinate_system`
- `label_name`
- `table_name`
- `image_name`
- `feature_channel_indices`
- `feature_channel_names`
- `feature_names`
- `feature_key`

### Object-classification-local selection state

Possible `ObjectClassificationSelectionState` fields:

- `label_name`
- `table_name`
- `feature_key`
- `color_by`

Possible `LayerBindingRegistry` responsibilities:

- map loaded napari layers to `element_name`
- map loaded napari layers to `element_type`
- map loaded napari layers to `coordinate_system`
- answer whether a requested segmentation or image is already loaded

## 2. Viewer adapter

A viewer-facing adapter should be responsible for:

- loading labels into napari
- loading images into napari
- updating displayed channels
- supporting both single-layer image loading and additive channel-overlay loading
- activating existing layers
- resolving whether a requested segmentation is already loaded
- maintaining the loaded-layer registry
- attaching only lightweight metadata when useful for interoperability

Suggested module:

- `src/napari_harpy/_viewer_adapter.py`

## 3. Layer binding and minimal metadata

The primary contract should be an in-memory Harpy layer-binding registry, not
layer metadata.

That means downstream widgets should preferably ask Harpy:

- whether a `SpatialData` object is loaded
- which layers are already loaded
- how to load or activate a required layer
- whether a matching labels layer is currently loaded

rather than scraping those answers from `layer.metadata`.

If we still attach metadata to layers, it should stay minimal and lightweight.

Good candidates:

- `metadata["element_name"]`
- `metadata["element_type"]`
- `metadata["coordinate_system"]`

Metadata we should avoid treating as authoritative:

- `metadata["sdata"]`
- `metadata["adata"]`
- copied table linkage state
- copied channel-selection state
- any large or mutable analysis state

The rule should be:

- Harpy app state is authoritative
- viewer adapter layer bindings are operational state
- layer metadata is optional compatibility / debugging state

## Relationship To Other Widgets

### Feature Extraction

The feature-extraction widget should use the shared loaded `SpatialData` object
and layer services, then compute from canonical `SpatialData` elements.

Expected behavior:

- coordinate-system choice remains explicit in the feature-extraction widget
- all coordinate systems available in the loaded `SpatialData` are offered
- after a coordinate system is selected, labels and images are filtered to that
  coordinate system
- labels, images, and tables come from the currently loaded `SpatialData`, not
  from the viewer widget's current selection
- tables are filtered to those correctly linked to the selected labels element
- channel choices reflect the chosen image
- feature extraction channel selection is explicit in the feature-extraction
  widget and is independent from viewer display channels
- viewer display state should not silently define or override feature
  extraction inputs
- stored feature-matrix metadata records the feature-extraction-selected image,
  channels, and coordinate system

### Object Classification

The object-classification widget should use the shared loaded `SpatialData`
object and layer services, but it has a stronger live-viewer dependency.

Expected behavior:

- segmentation choices come from the currently loaded `SpatialData`
- when a segmentation is selected, the widget should load it into napari if it
  is not already present
- once loaded, the labels layer should be activated automatically so annotation
  can start immediately
- table and feature-matrix choices come from the linked `SpatialData` table
- tables are filtered to those correctly linked to the selected segmentation
- annotation is only enabled when the chosen segmentation is loaded in napari as
  a compatible labels layer
- when the segmentation is not loaded, the widget should preferably be able to
  load and activate it for the user rather than only warning

### Future viewer coloring

The viewer itself should eventually support table-driven coloring for labels.

Expected future behavior:

- after selecting a labels element, only correctly linked tables are offered
- after selecting a linked table, the user can choose a coloring source
- coloring sources should include:
  - an `.obs` column
  - a gene / `.var` feature represented on the linked table
- the selected labels layer is recolored directly in napari from the chosen
  table-derived values

## Recommended First Milestone

### “Viewer-first Harpy”

Deliver:

- a new `Viewer` widget in the napari plugin menu
- direct opening of a local `SpatialData` store from Harpy
- one loaded `SpatialData` object shared across widgets
- image discovery from `sdata.images`
- segmentation discovery from `sdata.labels`
- coordinate-system selection
- coordinate-system-first element filtering
- image display mode selection
- linked-table selection for labels
- channel-selection UI for images
- per-channel color selection for overlay mode
- `Show segmentation`
- `Show image`
- `Show selected channels`
- a Harpy-owned layer-binding registry
- only lightweight metadata attached to layers when needed for compatibility

This would already provide value even before a full shared session model is
needed.

## Suggested Phases

### Phase 1: Viewer widget skeleton

- add a new dock widget: `Viewer`
- add `Open SpatialData`
- store one loaded `SpatialData` in shared app state
- expose selectable labels, images, and coordinate systems from that `SpatialData`
- make coordinate system selection the first required choice before element
  selection
- filter labels and images to the active coordinate system

### Phase 2: Native loading into napari

- load selected labels into napari
- load selected image into napari
- register created layers in a Harpy-owned layer-binding registry
- attach only lightweight metadata when useful
- avoid unnecessary duplicate layers when possible
- support raw image mode as a single napari image layer
- expose linked-table selection for labels elements

### Phase 3: Channel-aware image display

- discover image channels
- add channel-selection UI
- add per-channel color selection
- support overlay mode with one napari image layer per selected channel
- use additive blending so selected channels are shown on top of each other
- update napari image display to reflect the selected channel subset

### Phase 4: Labels table-aware viewing

- discover correctly linked tables for the selected labels element
- add linked-table dropdown for labels
- prepare the viewer for table-driven coloring controls
- add future-facing hooks for:
  - color by `.obs` column
  - color by gene / `.var` feature

### Phase 5: Downstream widget integration

- make `Feature Extraction` read the shared loaded `SpatialData` rather than
  viewer widget selections
- make `Feature Extraction` expose coordinate-system-first selection, then
  filter labels/images to the chosen coordinate system
- make `Feature Extraction` expose linked tables for the selected labels element
- make `Feature Extraction` expose image channels explicitly for extraction
- make `Object Classification` read the shared loaded `SpatialData` rather than
  viewer widget selections
- make `Object Classification` expose segmentations from the loaded `SpatialData`
- make `Object Classification` auto-load and activate the chosen labels layer
- make `Object Classification` expose linked tables for the selected
  segmentation
- add helper actions so object classification can load the required labels layer
  when needed

### Phase 6: Future state-model evolution

- keep the shared loaded `SpatialData` and layer services minimal
- keep detailed widget selections local
- reduce direct viewer scanning inside downstream widgets
- keep napari as the projection of Harpy-owned state

Future extension:

- if Harpy later needs to support multiple loaded `SpatialData` objects, the
  simple `HarpyAppState` can grow into a fuller session model at that point

## Testing Expectations

We should add tests for:

- opening a `SpatialData` store from the viewer widget
- coordinate-system-first selection flow
- element filtering by coordinate system
- segmentation option discovery
- image option discovery
- linked-table discovery for labels elements
- channel discovery and fallback naming
- raw image mode for multiplex images
- overlay mode for multiplex images
- per-channel color assignment in overlay mode
- `Show segmentation`
- `Show image`
- `Show selected channels`
- downstream widget compatibility with viewer-loaded layers
- object-classification validation when a segmentation is not loaded

## Final Recommendation

The viewer widget should become the shared visualization and channel-selection
entry point for `napari-harpy`.

That gives us:

- a reusable image-viewing workflow
- a clear home for multiplex channel selection
- less crowding in task-specific widgets
- a better foundation for future deep-learning and QC workflows
- a clearer separation between viewing and analysis

The first implementation can still use lightweight layer metadata as a
compatibility bridge, but it should be designed so that the shared Harpy app
state and viewer adapter remain the real source of truth.
