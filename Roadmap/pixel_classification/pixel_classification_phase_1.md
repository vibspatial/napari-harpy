Roadmap for pixel classification Phase 1.

This document turns the high-level design in `pixel_classification.md` into a scoped implementation plan for the
first production slice. Phase 1 is not a prototype. It should be a durable, tested, user-facing feature with a narrow
resolution contract: everything runs on the highest-resolution image grid.

**Phase 1 Scope**
Phase 1 delivers interactive pixel classification at the highest available resolution from inside one integrated
`Pixel Classification` widget.

Included:

- select one or more coordinate systems;
- configure one pixel-classification target card per selected coordinate system;
- select a SpatialData image and one or more channels inside each target card;
- load selected channels as napari overlays for visual inspection;
- create/manage a high-resolution napari `Labels` annotation layer;
- train from annotated pixels only, with `0 = unlabeled` and `1..N = classes`;
- extract tile-wise pixel features from selected channels as an internal cache-building step;
- append normalized marker intensities to reduced deep features;
- write/reuse/delete a manifest-keyed sidecar zarr feature cache;
- train a classifier and predict class labels over each configured target image;
- display predicted labels as a high-resolution napari labels layer;
- explicitly save predicted labels to SpatialData as `sdata.labels[...]`;
- report progress, errors, stale state, and cache reuse clearly in the UI.

Excluded from Phase 1:

- no multiscale image-scale selector;
- no coarser-scale compute path;
- no annotation downsampling;
- no prediction upsampling;
- no DINOv2/JAFAR backend;
- no classifier export/reuse across different source images;
- no separate feature-extraction widget or feature-key workflow for pixel classification;
- no hidden feature cache inside the SpatialData zarr store.

If the selected image element is multiscale, Phase 1 should resolve and use the highest-resolution scale only.
Scale selection belongs to Phase 2.

**Target UX Contract**
The pixel-classification widget should feel like the feature-extraction target builder and the object-classification
training workflow combined into one purpose-built surface.

Feature extraction should not be exposed as a separate workflow here. Pixel features are implementation-specific,
large, cache-backed, and not usually meaningful to users as reusable table features. The user-facing concept is
therefore `feature cache`, not `feature matrix` or `feature key`.

Top-level flow:

1. User chooses one or more coordinate systems.
2. The widget creates one target card for each selected coordinate system.
3. In each target card, the user chooses an image element.
4. In each target card, the user chooses one or more channels.
5. The user can load any chosen channel as a napari overlay for visual inspection.
6. The widget checks whether a compatible feature cache already exists for that card.
7. If a cache exists, the card clearly reports that it can be reused.
8. If no compatible cache exists, the card prompts the user to build the feature cache.
9. The user can explicitly run or rerun feature extraction/cache building.
10. Once a compatible cache exists, annotation and classifier controls become available.
11. The user creates/opens the annotation labels layer, paints classes, trains, predicts, reviews, and saves.

Coordinate-system selection:

- mirror the feature-extraction widget pattern: the coordinate-system selector controls which target cards are visible;
- preserve card state when users temporarily uncheck and recheck a coordinate system where possible;
- if a coordinate system has no eligible images, show a disabled target card or status entry that explains why;
- target cards should be independent. Phase 1 does not need cross-card classifier training or cross-card prediction.

Target card layout:

- coordinate system heading;
- image selector filtered to that coordinate system;
- selected image summary, including highest-resolution shape and whether the source image is multiscale;
- channel selector with multi-select behavior;
- per-channel `Load overlay` action so users can inspect channels in napari before committing to cache generation;
- cache status block;
- feature-cache actions;
- annotation/classifier block, disabled until cache readiness;
- prediction/save block, disabled until a prediction exists.

Channel overlay behavior:

- `Load overlay` should be available per channel, not only for the whole selected channel set;
- overlays are for visual inspection and should not modify the cache manifest by themselves;
- channel overlay layers should use stable names that include image and channel identifiers;
- if a channel is already loaded, pressing `Load overlay` should update/reuse the existing overlay rather than creating
  confusing duplicates.

Cache UX states:

- `No compatible cache`: valid image/channel selection exists, but no matching cache was found;
- `Cache found`: compatible cache exists and can be reused;
- `Cache building`: feature extraction/cache writing is running;
- `Cache ready`: cache is available for annotation and classifier training;
- `Cache stale`: current image/channel/settings selection differs from the cache used by the current annotation or
  classifier state;
- `Cache invalid/partial`: a matching cache path exists but failed validation and should not be reused.

Cache actions:

- `Build cache` when no compatible cache exists;
- `Reuse cache` should be the default when a compatible cache exists, but the user should be aware of it;
- `Rebuild cache` should be explicit and should not silently overwrite a valid cache without confirmation or a new
  cache id;
- `Delete cache` should be available for visible sidecar caches, with confirmation.

Annotation/classifier gating:

- annotation controls are disabled until the target card has a compatible feature cache;
- classifier training is disabled until a compatible cache exists and the annotation layer has enough nonzero labels
  from at least two classes;
- changing image, channels, normalization settings, feature extractor settings, or cache id marks existing annotation
  and classifier state as needing review;
- changing annotations marks classifier predictions stale;
- the annotation layer is always highest-resolution in Phase 1.

Classifier/prediction UX:

- training and prediction live in the same target card as source and cache selection;
- status messaging should mirror object classification: insufficient labels, training running, prediction ready,
  prediction stale, and errors should be explicit;
- predicted labels are displayed as a high-resolution labels layer;
- saving predicted labels to SpatialData is explicit and disabled until a prediction exists.

**Target Package Layout**
Add a separate pixel-classification package instead of mixing this into object classification.

```text
src/napari_harpy/core/pixel_classification/
  __init__.py
  cache_store.py
  classifier.py
  features.py
  manifest.py
  normalization.py
  prediction.py
  reducer.py

src/napari_harpy/widgets/pixel_classification/
  __init__.py
  annotation_controller.py
  controller.py
  status_card.py
  widget.py
```

Register the new widget in:

- `src/napari_harpy/napari.yaml`;
- `src/napari_harpy/widgets/__init__.py`;
- `src/napari_harpy/_interactive.py`.

The core package should be importable without Qt and should contain the testable data/model/cache logic. The widget
package should own napari layers, Qt controls, thread workers, and user-facing state.

Do not add `spatialdata_io.py` or a central `types.py` by default. The repository already has shared SpatialData
helpers in `src/napari_harpy/core/spatialdata.py`, plus validation helpers in `src/napari_harpy/core/validation.py`.
Phase 1 should extend and reuse those shared modules for generic SpatialData concerns instead of introducing a
parallel pixel-classification-specific I/O layer. Pixel-specific dataclasses should live beside the behavior that owns
them, following existing patterns such as `core/histogram.py`, `core/classifier.py`, and
`core/feature_extraction.py`.

**Core Data Contracts**
Define immutable request/result objects where they are owned:

- shared image/source resolution structures, if generic enough, should live in `core/spatialdata.py`;
- `PixelFeatureManifest` should live in `core/pixel_classification/manifest.py`;
- `PixelFeatureCache` should live in `core/pixel_classification/cache_store.py`;
- classifier training inputs/results should live in `core/pixel_classification/classifier.py`;
- `PixelPredictionResult` should live in `core/pixel_classification/prediction.py`;
- widget worker jobs and UI binding snapshots should live in `widgets/pixel_classification/controller.py`.

Reuse existing shared helpers where possible:

- `get_coordinate_system_names_from_sdata(...)`;
- `get_spatialdata_image_options_for_coordinate_system_from_sdata(...)`;
- `get_image_channel_names_from_sdata(...)`;
- `validate_new_spatialdata_element_name(...)`;
- `normalize_spatialdata_name(...)`.

If Phase 1 needs generic image helpers that do not yet exist, add them to `core/spatialdata.py` rather than hiding them
inside the pixel-classification package. Good candidates are public versions of the existing image-scale logic used by
histogram calculation: resolving `DataArray` versus `DataTree`, picking `scale0` for highest-resolution Phase 1
execution, validating coordinate-system availability, and returning channel-aware image arrays.

The Phase 1 resolution contract is simple:

- annotation shape equals highest-resolution image shape;
- feature cache shape is `(C + F_reduced, y, x)` at highest resolution;
- predicted class labels shape equals highest-resolution image shape;
- saved `sdata.labels[...]` shape equals highest-resolution image shape.

The manifest should record enough information to reject incompatible caches:

- source SpatialData path / URI;
- source image element name;
- source image element zarr path resolved through SpatialData metadata;
- coordinate system;
- selected channels and channel names;
- highest-resolution source shape, axes, and dtype;
- normalization settings;
- feature extractor name, weights, package versions, layers, and tile settings;
- reducer type, parameters, fitted state identity, and output dimension;
- feature cache dtype, shape, chunks, and schema version.

`cache_id` should be a hash of the canonical manifest excluding runtime fields such as creation time, last-used time,
and UI-only labels.

**Implementation Slices**
1. Package skeleton and widget registration

Create the package structure, register `PixelClassificationWidget`, and add a minimal dock widget that binds to
`HarpyAppState`. The widget can initially show disabled controls and a status card, but it must be discoverable from
napari and from `Interactive(..., widgets=...)`.

Slice 1 is only the product shell and registration layer. It should not implement cache building, annotation handling,
feature extraction, classifier training, prediction, or saving. The goal is to create a clean, importable, registered
entry point that later slices can fill in without changing public names.

Files to add:

```text
src/napari_harpy/core/pixel_classification/__init__.py
src/napari_harpy/widgets/pixel_classification/__init__.py
src/napari_harpy/widgets/pixel_classification/controller.py
src/napari_harpy/widgets/pixel_classification/status_card.py
src/napari_harpy/widgets/pixel_classification/widget.py
tests/test_pixel_classification_widget.py
```

Files to update:

```text
src/napari_harpy/napari.yaml
src/napari_harpy/widgets/__init__.py
src/napari_harpy/_interactive.py
tests/test_package.py
tests/test_app_state.py
```

Core package requirements:

- `napari_harpy.core.pixel_classification` exists and imports without importing Qt, napari, torch, or torchvision;
- `__init__.py` should stay minimal, with no eager imports of heavy or optional dependencies;
- no placeholder feature/cache/classifier modules should be added in Slice 1 unless they are needed by the widget shell.

Widget package requirements:

- expose `PixelClassificationWidget` from `widgets/pixel_classification/__init__.py`;
- define `PixelClassificationWidget(QWidget)` in `widgets/pixel_classification/widget.py`;
- accept `napari_viewer: napari.Viewer | None = None`, matching the existing widget constructors;
- call `get_or_create_app_state(napari_viewer)` and keep the returned shared app state;
- set a stable object name, `pixel_classification_widget`;
- use the shared widget styling helpers already used by the other widgets;
- show a minimal status/card area with an initial state such as “No SpatialData loaded” or “Choose a coordinate system
  to start”;
- reserve the intended shell structure: coordinate-system selection area, target-card container, and global status
  feedback, even if real target cards are populated in Slice 2;
- all action controls in this first slice should be disabled or absent until later slices implement real behavior;
- destroying the widget should not leave timers/workers/callbacks running. Since Slice 1 should not start workers, this
  is mostly a constraint on not introducing unnecessary asynchronous objects yet.

Controller/status requirements:

- `controller.py` should contain only light widget-facing state needed by the shell, for example selected `SpatialData`
  presence and a status message;
- `status_card.py` should contain the minimal status-card builder or helper used by the widget shell;
- do not introduce source-resolution, annotation, cache, feature, or classifier state in Slice 1;
- the controller should be easy to replace/extend when Slice 2 adds source resolution.

Napari manifest registration:

- add command id `napari-harpy.pixel_classification`;
- command title should be `Open pixel classification widget`;
- `python_name` should be `napari_harpy.widgets.pixel_classification.widget:PixelClassificationWidget`;
- add a widget contribution with display name `Pixel Classification`;
- keep existing widget display names unchanged.

Lazy widget export:

- add `PixelClassificationWidget` to the `TYPE_CHECKING` block in `widgets/__init__.py`;
- add `"pixel_classification.widget": ["PixelClassificationWidget"]` to the lazy loader mapping;
- add `"PixelClassificationWidget"` to `__all__`;
- importing `napari_harpy.widgets` should not import the pixel widget module until the attribute is requested.

Interactive launcher registration:

- add `"pixel_classification"` to `HarpyWidgetId`;
- add `"pixel_classification": "Pixel Classification"` to `_WIDGET_NAMES`;
- add `"pixel_classification"` to `_ALL_WIDGET_IDS`;
- update the `Interactive` docstring so the valid widget list includes `pixel_classification`;
- `Interactive(..., widgets="pixel_classification")` should dock only `("napari-harpy", "Pixel Classification", True)`;
- `Interactive(..., widgets="all")` should include the new widget exactly once.

Slice 1 tests:

- `tests/test_package.py`: assert the napari manifest contributes the `Pixel Classification` widget and command;
- `tests/test_package.py`: assert lazy import exposes `PixelClassificationWidget` from `napari_harpy.widgets`;
- `tests/test_app_state.py`: assert `Interactive(..., widgets="pixel_classification")` docks the new widget;
- `tests/test_app_state.py`: update the `"all"` expected dock list to include `Pixel Classification`;
- `tests/test_pixel_classification_widget.py`: instantiate `PixelClassificationWidget()` and with a viewer, assert object
  name and shared app-state binding;
- add or extend an import hygiene test so importing `napari_harpy.core.pixel_classification` does not import
  `torch`, `torchvision`, `napari`, or Qt bindings.

Acceptance criteria:

- the widget appears in the napari manifest;
- `Interactive(..., widgets=("pixel_classification",))` can open it;
- importing `napari_harpy.core.pixel_classification` does not import Qt, napari, torch, or torchvision.

2. Source resolution and target-card UX

Reuse and, where needed, extend `core/spatialdata.py` for source resolution. The pixel-classification controller should
call shared helpers to resolve coordinate systems, image elements, axes, channel names, selected channels, and
highest-resolution array shape. This slice should introduce the target-card interaction model: one visible card per
selected coordinate system, with image and channel selection inside each card. If the image is a multiscale `DataTree`,
pick `scale0` / highest-resolution scale in Phase 1 and expose that decision in status/metadata.

The card should also provide per-channel `Load overlay` actions. Overlay loading is a viewer-inspection action; it
should not create a feature cache, mark a cache stale, or change the feature manifest unless the user changes the
actual selected channel set.

Acceptance criteria:

- checking/unchecking coordinate systems creates/removes target cards while preserving card state when practical;
- single-scale and multiscale images resolve to a concrete highest-resolution 2D channel image;
- image choices are filtered to the target card coordinate system;
- channel choices reflect the selected image and support selecting one or more channels;
- every selected channel has a visible `Load overlay` action;
- loading an overlay reuses or updates an existing image/channel overlay layer instead of creating duplicate clutter;
- invalid channel selections produce actionable validation messages;
- in-memory, remote, read-only, and backed SpatialData stores are distinguished for cache-location policy.

3. Annotation layer lifecycle

Implement `annotation_controller.py`. Create or reuse a napari labels layer matching the highest-resolution image grid.
Use `0` as unlabeled. Keep class labels, class names, and colors stable while the user edits annotations.

Acceptance criteria:

- annotation actions remain disabled until the card has a compatible feature cache;
- annotation layer shape always matches the highest-resolution image shape;
- annotation layers are associated with the target card image/channel cache context;
- nonzero label counts per class are tracked;
- training is disabled until at least two classes have enough annotated pixels;
- changing annotations marks classifier predictions stale.

4. Normalization and feature schema

Implement `normalization.py`, `features.py`, and `reducer.py`. Normalize selected marker channels before caching, extract
deep features tile-wise, reduce only the deep-feature axis, then append normalized marker planes unchanged. In the UI,
this work is presented as building or rebuilding a feature cache, not as producing a user-facing feature matrix.

Recommended Phase 1 defaults:

- selected channels: user-selected subset, with a product warning for very large selections;
- marker normalization: deterministic channel-wise clipping/scaling recorded in the manifest;
- deep feature backend: ConvNeXt-Tiny early-layer features behind the optional `torch` dependency group;
- reducer: deterministic fixed random projection or fitted incremental reducer, recorded in the manifest;
- persistent feature dtype: `float16` unless validation shows it harms classifier quality.

Acceptance criteria:

- feature cache planes are ordered as selected normalized marker planes followed by reduced deep feature planes;
- raw high-dimensional deep features are streamed tile-wise and are not persisted blindly;
- tests can run with a fake feature extractor so CI does not require downloading model weights;
- missing optional torch dependencies produce a clear install/action message.

5. Sidecar cache store

Implement `cache_store.py` and `manifest.py`. Store feature caches in an explicit sidecar zarr store:

```text
sample.harpy-cache.zarr/
  pixel_classification/
    feature_caches/
      <cache_id>/
        features
        reducer/
        manifest
```

Default cache-location behavior:

- backed writable local SpatialData: sibling `<source>.harpy-cache.zarr`;
- in-memory, remote, or read-only SpatialData: user must choose a writable cache location;
- cache discovery, reuse, and deletion must be visible in the widget;
- compatible cache discovery runs automatically after image/channel/cache-setting changes;
- annotation and classifier controls unlock only when the target card has a compatible cache.

Acceptance criteria:

- identical compatible requests resolve to the same `cache_id`;
- incompatible requests cannot silently reuse a cache;
- cache status is reported per target card as missing, found, building, ready, stale, or invalid/partial;
- cache creation is atomic enough that interrupted writes do not look valid;
- stale/partial caches are reported and can be deleted from the UI.

6. Classifier training and prediction

Implement `classifier.py` and `prediction.py`. Train from annotated pixels only, using rows read from the feature cache.
Predict over the full high-resolution feature cache tile-wise and return a high-resolution predicted label map.

Recommended Phase 1 classifier:

- `RandomForestClassifier` with deterministic `random_state`;
- class labels are integer label IDs from the annotation layer;
- confidence is max predicted probability;
- prediction writes should not modify the annotation layer.

Acceptance criteria:

- training rejects empty/unbalanced/one-class annotation states with clear messages;
- prediction is tile-wise and does not require flattening the whole image into memory at once;
- predicted class labels have the highest-resolution image shape;
- classifier metadata records feature cache id, training class counts, training time, and classifier parameters.

7. Widget workflow and background jobs

Implement `controller.py`, `status_card.py`, and `widget.py`. The UI should guide the user through source selection,
annotation, cache generation/reuse, training, prediction, and save.

Required states:

- no SpatialData loaded;
- no coordinate system selected;
- source incomplete;
- no eligible images for a selected coordinate system;
- no compatible cache;
- cache found/reusable;
- cache building;
- cache ready;
- cache stale;
- cache invalid/partial;
- annotations insufficient;
- training;
- prediction ready;
- prediction stale;
- save succeeded;
- error with recovery action.

Long-running operations should run in background workers and expose progress where possible. Cache building, training,
prediction, and save should not freeze napari.

Acceptance criteria:

- controls are enabled only when the current state is valid;
- annotation and classifier controls stay disabled until cache readiness;
- cancel/shutdown prevents orphan callbacks from updating destroyed widgets;
- errors leave the widget in a recoverable state;
- feature cache reuse is explicit rather than surprising.

8. Save to SpatialData

Implement save logic by reusing shared SpatialData validation helpers first. The user-facing prediction result is a
high-resolution labels element. If the save path grows beyond a thin call-site, add a narrowly named
`core/pixel_classification/output.py`; do not create a broad duplicate `spatialdata_io.py`.

Save behavior:

- default output name should be generated from source image and classifier context;
- existing labels element names require explicit overwrite or a new name;
- saved labels include coordinate-system metadata compatible with the source image;
- prediction metadata records cache id, selected channels, normalization, feature extractor, reducer, classifier
  parameters, class labels, and created timestamp.

Acceptance criteria:

- saved predictions reload as normal SpatialData labels;
- labels align with the highest-resolution source image in napari;
- backed SpatialData writes are persisted;
- failed writes do not leave ambiguous UI state.

9. Tests and verification

Add focused tests before broadening UI behavior:

- manifest canonicalization and `cache_id` stability;
- cache compatibility/rejection;
- coordinate-system selection creates target-card state;
- image/channel target-card selection validates against coordinate-system availability;
- per-channel overlay load reuses existing layers;
- highest-resolution selection for single-scale and multiscale image elements;
- annotation validation and class-count tracking;
- normalization and feature-plane ordering;
- fake extractor plus reducer writes expected feature-cache shape;
- classifier training/prediction on small synthetic data;
- tile-wise prediction shape/dtype;
- widget state transitions with mocked workers;
- save-to-SpatialData labels alignment and overwrite policy.

Use the repository environment directly:

```bash
.venv/bin/pytest
.venv/bin/pre-commit run ruff --all-files
```

Torch/TorchVision integration tests should be optional or skipped unless the optional dependency group and model weights
are available. Core behavior should be testable without network access.

**Suggested Delivery Order**
1. Package skeleton, manifest registration, and empty widget.
2. Source resolution, target-card UX, per-channel overlays, and Phase 1 highest-resolution contract.
3. Annotation layer controller and class-count validation.
4. Manifest/cache-id model and sidecar cache store.
5. Fake feature extractor path with cache writing and reuse.
6. Real ConvNeXt feature extractor behind optional dependencies.
7. Classifier training and tile-wise prediction.
8. Viewer display and stale-state handling.
9. Save predicted labels to SpatialData.
10. Cache management UI, progress/cancel polish, and final test hardening.

This order keeps each slice independently testable while building toward the real user workflow.

**Phase 1 Completion Definition**
Phase 1 is done when a user can choose coordinate systems, configure target cards with image/channel selections, load
channel overlays for inspection, build or reuse a visible sidecar feature cache, annotate pixels at highest resolution,
train and predict without blocking napari, inspect high-resolution predicted labels in the viewer, and explicitly save
those labels back into SpatialData with reproducible metadata.

The implementation must leave clear extension points for Phase 2 multiscale compute, but Phase 1 should not expose
scale selection or save transformed low-resolution prediction layers.
