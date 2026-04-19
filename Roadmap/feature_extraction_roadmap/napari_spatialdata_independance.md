# Plan for Becoming Independent of `napari-spatialdata`

## Goal

Make `napari-harpy` independent of `napari-spatialdata` for dataset loading, selection state, and widget operation.

The objective is **not** to immediately replace every visualization convenience provided by `napari-spatialdata`. The objective is to make `napari-harpy` own the authoritative application state:

- opening `SpatialData`
- tracking loaded datasets and samples
- discovering labels, images, and tables
- managing selection state
- deciding what should be shown in napari

In this model, napari layers become a **projection of Harpy state**, not the source of truth.

## Why this is worth doing

Today, both widgets depend on discovering `SpatialData` objects indirectly by scanning napari layers for metadata such as:

- `layer.metadata["sdata"]`
- `layer.metadata["name"]`

This makes `napari-harpy` dependent on:

- `napari-spatialdata` loading behavior
- `napari-spatialdata` metadata conventions
- the user first loading the right layers through another plugin
- viewer state being the entry point into the workflow

That dependency was useful for the MVP, but it now limits the architecture.

Becoming independent of `napari-spatialdata` would give us:

- direct dataset opening from within `napari-harpy`
- explicit, stable application state
- less fragility if external metadata conventions change
- a cleaner path to multi-sample workflows
- clearer control over image, channel, labels, and table loading
- a more coherent user experience

## Core design principle

`SpatialData` itself should be the authoritative data model.

`napari-harpy` should own:

- dataset/session state
- selected sample
- selected segmentation
- selected image
- selected table
- selected coordinate system

napari should only be responsible for visualization.

## Recommended target architecture

Split responsibilities into three layers.

### 1. Session / data model layer

This layer owns authoritative state for currently opened datasets.

Responsibilities:

- open `SpatialData` stores directly
- register loaded datasets in the current session
- expose dataset, sample, labels, image, and table choices
- keep the current selection state
- provide a stable identity for datasets and samples
- surface metadata needed by widgets and controllers

Suggested concepts:

- `SpatialDataSession`
- `DatasetHandle`
- `SampleHandle`
- `SelectionState`

This layer should not depend on napari viewer layers.

### 2. Viewer adapter layer

This layer translates session state into napari visualization.

Responsibilities:

- load selected labels or image elements into napari
- activate layers
- manage visibility
- update image display state, including channel display
- maintain best-effort linkage between viewer layers and session selections

This layer should be the only place where napari-specific layer manipulation happens.

### 3. Controller / widget layer

This layer should continue to operate on authoritative `SpatialData` and selection information.

Responsibilities:

- annotation controller
- classifier controller
- feature extraction controller
- widget status and validation logic

Controllers should not need to know whether a dataset was discovered through viewer metadata or loaded directly by Harpy.

## What should change first

Do **not** begin by trying to recreate all of `napari-spatialdata`.

The first milestone should be much smaller:

- Harpy can open a `SpatialData` object directly
- Harpy can enumerate labels, images, and tables from it
- the widgets can bind to those objects without requiring `napari-spatialdata`
- Harpy can load selected elements into napari itself when needed

This gives us independence where it matters, without taking on unnecessary rendering scope.

## Proposed implementation phases

## Phase 1: Introduce a session-native data model

Create a new internal module, for example:

- `src/napari_harpy/_session.py`
- or `src/napari_harpy/_spatialdata_session.py`

This module should introduce the concept of an application session.

### Responsibilities

- store opened `SpatialData` objects
- assign stable identities to datasets
- expose available labels, images, and tables
- keep track of selection state independently of napari
- provide helper methods for widgets and controllers

### Suggested objects

#### `SpatialDataSession`

Owns the current Harpy session and all opened datasets.

Possible responsibilities:

- `open(path)`
- `add_dataset(sdata, dataset_name=None)`
- `remove_dataset(dataset_id)`
- `list_datasets()`
- `list_labels(dataset_id)`
- `list_images(dataset_id)`
- `list_tables(dataset_id)`

#### `DatasetHandle`

Represents one opened dataset.

Possible fields:

- `dataset_id`
- `dataset_name`
- `sdata`
- `path`
- `samples` or sample metadata if available

#### `SelectionState`

Represents the active selection in the UI.

Possible fields:

- `dataset_id`
- `sample_id` (future-facing)
- `label_name`
- `image_name`
- `table_name`
- `coordinate_system`
- `channel_indices` (future-facing)

## Phase 2: Separate pure SpatialData helpers from viewer-driven helpers

Today `_spatialdata.py` mixes several responsibilities:

- table metadata and validation helpers
- viewer scanning for layers
- option construction for widgets
- layer lookup helpers

This should be split.

### Keep as pure data helpers

These should remain independent of napari:

- table validation
- table metadata normalization
- image/label/table enumeration from `SpatialData`
- coordinate-system enumeration
- region / instance linkage checks

### Move into a viewer adapter module

These should become napari-facing helpers:

- find or activate a layer
- load selected labels into napari
- load selected image into napari
- update image display state
- synchronize widget state with layer visibility

Suggested module:

- `src/napari_harpy/_viewer_adapter.py`

## Phase 3: Replace viewer-discovery as the primary source of options

Currently, widget options are discovered by scanning the viewer for linked layers.

This should change so that widget options come from the session layer.

### New rule

Widgets should populate choices from:

- currently opened session datasets
- selected dataset
- selected sample
- canonical `SpatialData` elements

not from:

- whatever layers happen to already be in the viewer

### Viewer layers should become optional

A labels or image element may be selectable even if it is not currently loaded in napari.

The widget should then offer explicit actions like:

- `Show image`
- `Show segmentation`

This is already aligned with the direction we want for channel-aware image selection.

## Phase 4: Add explicit dataset opening UX

Once the session layer exists, add user-facing entry points.

Possible actions:

- `Open SpatialData`
- `Add dataset to session`
- `Close dataset`
- `Choose dataset`

This can begin with a minimal file/path-driven flow.

### Minimal first version

- one button to open a local `SpatialData` store
- dataset appears in session
- widgets can bind to it immediately
- no dependency on `napari-spatialdata`

## Phase 5: Implement minimal native visualization support

At this stage, Harpy should be able to show selected elements in napari without relying on `napari-spatialdata`.

This does not need to be feature-complete.

### Required support

- load labels layers into napari
- load image layers into napari
- preserve basic naming and identity
- activate layers from widget actions

### Future-facing support

Later, the viewer adapter can also own:

- channel visualization
- selected-channel synchronization
- contrast presets
- better visibility and layer grouping behavior

## Phase 6: Keep compatibility optional, not mandatory

We do not need to drop compatibility with `napari-spatialdata` immediately.

A practical transition plan is:

- make Harpy able to operate without it
- keep best-effort interoperability when present
- gradually stop depending on its metadata conventions internally

This avoids a disruptive rewrite and gives maintainers room to validate the new architecture incrementally.

## Relationship to multi-sample support

Independence from `napari-spatialdata` is the right foundation for future multi-sample work.

Right now, multiple datasets are mostly represented implicitly by multiple viewer-linked `SpatialData` objects. That is not enough for a strong multi-sample design.

A native session model makes it possible to introduce explicit concepts such as:

- dataset
- sample
- sample-specific images
- sample-specific segmentations
- sample-specific coordinate systems

Without this, multi-sample features will remain awkward and viewer-driven.

## Why this matters for the authoritative table idea

If we want one authoritative table that stores features for multiple samples, then Harpy must be able to:

- load and understand those samples explicitly
- map labels/images/tables to sample identity
- coordinate feature extraction across samples
- surface those choices clearly in the widget

That is much easier if Harpy owns the session model instead of inheriting all context from napari layer metadata.

## Recommended boundaries for the first refactor

To keep this manageable, I would avoid a big-bang rewrite.

### First refactor target

Introduce a session layer and make widgets read from it.

That means:

- widgets stop relying on viewer scanning as the primary discovery mechanism
- controllers continue to receive `sdata + label/image/table names`
- viewer integration remains, but becomes optional

### Second refactor target

Introduce a viewer adapter that can load and activate labels and images from Harpy state.

### Third refactor target

Add dataset-opening UX and phase out the assumption that another plugin loads the data first.

## What not to do yet

To keep scope under control, do **not** start with:

- a full replacement for all `napari-spatialdata` visualization features
- a custom image canvas inside widget UI
- advanced sample pooling semantics
- automatic migration of every existing viewer-linked workflow all at once

The main goal is independence of **state and loading**, not immediate feature parity.

## Concrete first milestone

A good first milestone would be:

### “Session-native Harpy”

Deliverables:

- Harpy can open `SpatialData` directly
- Harpy stores opened datasets in a session model
- widgets read labels/images/tables from session state
- user can choose segmentation/image/table without `napari-spatialdata`
- Harpy can load the chosen labels/image into napari on demand

If that is done well, the rest of the system becomes much easier to evolve.

## Suggested roadmap

### Milestone 1: Session model
- add `SpatialDataSession`
- add dataset handles
- add selection state

### Milestone 2: Pure helper split
- split `_spatialdata.py` into pure-data helpers and viewer adapter code

### Milestone 3: Widget rebinding
- make widgets use session-native discovery
- stop relying on viewer scanning as the primary source

### Milestone 4: Open dataset UX
- add explicit open/add dataset flow

### Milestone 5: Native layer loading
- add minimal image/labels loading into napari

### Milestone 6: Optional interoperability
- keep compatibility with `napari-spatialdata` where useful
- stop depending on it internally

## Final recommendation

The correct architectural move is not:

**“reimplement napari-spatialdata”**

The correct architectural move is:

**“make Harpy own dataset lifecycle, selection state, and what gets shown in napari”**

That is the key step that will:

- reduce external dependency
- simplify future feature work
- make multi-sample support realistic
- support channel-aware image selection cleanly
- move `napari-harpy` from a plugin add-on toward a coherent application layer
