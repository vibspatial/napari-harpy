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
- concatenate normalized marker intensities with 64 deterministically projected deep-feature planes;
- write/reuse/delete a manifest-keyed sidecar zarr feature cache;
- train one pooled classifier and predict class labels for the active viewer target;
- display predicted labels as a high-resolution napari labels layer;
- explicitly save predicted labels to SpatialData as `sdata.labels[...]`;
- report progress, errors, stale state, and cache reuse clearly in the UI.

Phase 1 treats positive integer annotation values as the complete class identity. `0` means unlabeled, and `1..N` are
user-defined classes. There is no semantic class-name schema or cross-target class mapping. Pooled training assumes
that users apply the same integer label consistently across participating targets.

Excluded from Phase 1:

- no multiscale image-scale selector;
- no coarser-scale compute path;
- no annotation downsampling;
- no prediction upsampling;
- no DINOv2/JAFAR backend;
- no classifier export/reuse across different source images;
- no separate feature-extraction widget or feature-key workflow for pixel classification;
- no interactive UI action that predicts across all selected coordinate systems at once;
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
10. Once a compatible cache exists for a target card, annotation controls become available for that card.
11. The user creates/opens annotation labels layers and paints classes on one or more eligible target cards.
12. A shared classifier-training panel lets the user train one pixel classifier from all eligible annotated cards, or
    from a selected subset of eligible cards.
13. Interactive prediction is viewer-bound: the UI predicts only for the coordinate system/sample currently loaded or
    active in the viewer.
14. The user reviews the active prediction and explicitly saves it.

Coordinate-system selection:

- mirror the feature-extraction widget pattern: the coordinate-system selector controls which target cards are visible;
- preserve card state when users temporarily uncheck and recheck a coordinate system where possible;
- if a coordinate system has no eligible images, show a disabled target card or status entry that explains why;
- target cards are independently configured and cached, but they are not independent classifier workflows. Eligible
  target cards can contribute annotated pixels to one shared classifier training run.

Target card layout:

- coordinate system heading;
- image selector filtered to that coordinate system;
- selected image summary, including highest-resolution shape and whether the source image is multiscale;
- channel selector with multi-select behavior;
- per-channel `Load overlay` action so users can inspect channels in napari before committing to cache generation;
- cache status block;
- feature-cache actions;
- annotation block, disabled until cache readiness;
- training eligibility summary, for example cache ready, annotated classes, annotated pixel count, and whether this card
  is included in the current training scope;
- prediction status for this target, if it is the current viewer-bound prediction target.

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
- `Cache ready`: cache is available for annotation and can contribute to classifier training;
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
- classifier training is controlled by a shared training panel, not by individual target cards;
- a target card is eligible for training only when it has a compatible feature cache and enough annotation to contribute
  useful class examples;
- classifier training is disabled until the selected training scope contains eligible cached/annotated target cards with
  enough nonzero labels from at least two classes overall;
- changing image, channels, normalization settings, feature extractor settings, or cache id marks existing annotation
  and classifier state as needing review for the affected target card;
- changing annotations marks classifier predictions stale;
- the annotation layer is always highest-resolution in Phase 1.

Classifier training UX:

- classifier training lives in a shared panel below or beside the target cards;
- the training-scope control should mirror the object-classification idea of training on all eligible samples or on a
  selected subset;
- `All eligible selected samples` should include every selected coordinate-system target card that has a compatible
  cache and enough annotation;
- `Selected samples` should let the user choose a subset of eligible target cards;
- the classifier is trained from annotated pixels pooled across the selected training scope;
- status messaging should mirror object classification: insufficient labels, training running, trained classifier ready,
  stale classifier, and errors should be explicit.

Prediction and save UX:

- interactive prediction should not mirror object classification's multi-region prediction scope;
- the UI should allow prediction only for the coordinate system/sample currently loaded or active in the viewer;
- if several target cards are selected, the user should still run interactive prediction one active viewer target at a
  time;
- predicted labels are displayed as a high-resolution labels layer for the active target;
- saving predicted labels to SpatialData is explicit and disabled until a prediction exists for the active target;
- prediction status should report whether the active target has no compatible cache, no trained classifier, a ready
  prediction, or a stale prediction;
- multi-sample/batch inference should be a headless workflow, not a Phase 1 interactive UI action.

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
  projection.py

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
- feature cache shape is `(C + 64, y, x)` at highest resolution, where `C` is the number of selected marker channels;
- predicted class labels shape equals highest-resolution image shape;
- saved `sdata.labels[...]` shape equals highest-resolution image shape.

The manifest should record enough information to reject incompatible caches:

- source SpatialData path / URI;
- source image element name;
- source image element zarr path resolved through SpatialData metadata;
- coordinate system;
- selected channels and channel names;
- highest-resolution source shape, axes, dtype, and the complete physical Zarr/Dask chunk tuple for every axis,
  including the smaller terminal chunks;
- ConvNeXt common output stride, minimum input height/width, required context, stride-rounded halo, transient
  right/bottom padding, actual post-overlap core/expanded layouts, and final regular cache layout;
- normalization settings;
- feature extractor name, weights, package versions, layers, common stride, context halo, padding, and dense-upsampling
  settings;
- projection algorithm, parameters, implementation version, matrix identity, and output dimension (`K = 64`);
- raw feature schema id;
- projection id;
- final feature schema id;
- feature cache dtype, shape, chunks, and schema version.

All ids should use a versioned canonical hash contract:

- hash algorithm: `sha256`;
- id format: `<kind>:<schema_version>:sha256:<hex_digest>`, for example
  `final_feature_schema:v1:sha256:...`;
- structured payload: canonical JSON with sorted keys, no whitespace dependence, and normalized scalar types;
- array payloads: hash numeric arrays separately using canonical dtype, shape, memory order, and raw bytes, then include
  those array hashes in the JSON payload;
- excluded runtime fields: creation time, last-used time, writer hostname, progress state, local cache-store path,
  napari layer names, UI labels, runtime marker-batch size, source-chunk compatibility classification, lazy-rechunk
  diagnostics, and other fields that do not change feature values or compatibility;
- included version fields: cache schema version and package/model/library versions that can change feature values.

**Deterministic Projection and Cache Compatibility**
Feature caches should be physically separate per target card / sample / coordinate system, but pooled classifier
training requires a shared final feature schema.

Phase 1 should use one deterministic fixed random projection for the deep-feature axis. It must not fit PCA or another
data-dependent reducer. Given the same raw feature schema, every target independently obtains the same projection
matrix and final feature-plane semantics. Consequently, users can build caches one target at a time, add another target
later, and reuse existing compatible caches without a reducer-refit workflow or a reducer control in the UI.

Use a dense Rademacher projection as the Phase 1 default. For raw deep-feature dimension `D` and output dimension
`K = 64`, generate a matrix `R` with shape `(K, D)` whose entries are deterministically sampled from
`{-1 / sqrt(K), +1 / sqrt(K)}`. For a raw per-pixel feature vector `x`, compute `z = R @ x`. The generator algorithm,
seed-derivation rule, matrix orientation, scaling, input/output dtype policy, and implementation version are part of the
persisted projection contract. Generation must not depend on cache-build order, target data, annotations, training
scope, process-global random state, or hardware.

For the initial projection contract, derive the seed by hashing a versioned projection namespace together with the
`raw_feature_schema_id`, then convert a specified digest prefix to an unsigned integer with specified byte order. Use a
named, versioned pseudo-random generator rather than a library-global RNG. This makes matrix reproduction an explicit
algorithm rather than an accidental consequence of whichever process generated the first cache.

`K = 64` is a fixed Phase 1 product default, not a theorem-derived optimum or a user-facing setting. Later phases may
change it based on ablation results. Such a change creates a new `projection_id` and `final_feature_schema_id`; it does
not mutate an existing cache schema.

Use three levels of identity:

- `raw_feature_schema_id`: hash of fields that define the unreduced deep-feature stream and normalized marker planes,
  excluding target-specific source identity. This includes selected channel names and order, normalization settings,
  feature extractor name/weights/layers, minimum-input/stride/halo/padding/upsampling contract, raw deep-feature plane
  order, and raw deep-feature dimension.
- `projection_id`: hash of the deterministic projection contract and generated matrix. The same raw feature schema must
  produce the same projection id on every target and in every process.
- `final_feature_schema_id`: hash of `raw_feature_schema_id`, `projection_id`, final feature plane order, final dtype
  policy, and cache schema version.

Exact `raw_feature_schema_id` inputs:

- hash kind and hash schema version;
- pixel-classification cache schema version;
- phase/resolution mode, for Phase 1 `highest_resolution`;
- axes convention and feature array layout, for example `(features, y, x)`;
- selected channel names and order;
- channel identity policy, for example match by channel name;
- marker normalization policy, parameters, and whether fitted normalization state is per-target or shared;
- raw marker plane order;
- feature extractor backend name, implementation version, model name, weights name or digest, selected layers, scale
  pyramid settings, input channel handling, RGB replication strategy, pretrained input mean/std, padding policy,
  common output stride, minimum input height/width, required context, stride-rounded halo, dense-upsampling mode and
  coordinate convention, preprocessing, Dask overlap/trim contract version, and output raw feature plane order;
- raw deep-feature dimension before projection;
- projection input dtype/precision policy;
- relevant package versions for feature generation.

Exact `projection_id` inputs:

- hash kind and projection hash schema version;
- `raw_feature_schema_id`;
- projection implementation and package version;
- projection parameters, including input dimension, output dimension `K = 64`, Rademacher distribution and scaling,
  generator algorithm, seed-derivation rule, matrix orientation, and input/output dtype policy;
- projection matrix array hash.

Exact `final_feature_schema_id` inputs:

- hash kind and final schema hash version;
- `raw_feature_schema_id`;
- `projection_id`;
- final feature plane order: selected normalized marker planes first, then projected deep-feature planes;
- final feature names or deterministic plane labels;
- final feature count;
- final feature dtype policy;
- cache schema version.

Target-specific `cache_id` should still include source identity:

- hash kind and target-cache hash schema version;
- `final_feature_schema_id`;
- source SpatialData URI/path;
- source image element name;
- source element zarr path resolved through SpatialData metadata;
- coordinate system;
- compute scale, for Phase 1 `scale0` / highest resolution;
- highest-resolution shape and axes;
- complete source Dask/Zarr chunk tuples for every axis, including terminal chunks;
- selected channel names and order as resolved in this target;
- target-specific fitted normalization state, if the selected normalization policy is per-target;
- feature array shape, chunks, dtype, compressor/store policy, and cache schema version.

Recommended sidecar layout:

```text
sample.harpy-cache.zarr/
  pixel_classification/
    feature_schemas/
      <final_feature_schema_id>/
        raw_feature_manifest
        projection/
          <projection_id>/
            manifest
            matrix
    feature_caches/
      <target_cache_id>/
        features
        manifest              # points to raw_feature_schema_id, projection_id, final_feature_schema_id
```

Cache building is independent per target:

1. Resolve the raw feature schema and deterministically generate or load its projection artifact.
2. Stream raw deep features tile-wise for the target card.
3. Apply the projection only along the deep-feature axis, for example as a fixed `1 x 1` linear operation.
4. Concatenate selected normalized marker planes followed by the 64 projected deep-feature planes.
5. Write a separate target cache pointing to the shared `projection_id` and `final_feature_schema_id`.

The raw high-dimensional deep features should not be persisted. Building a new target cache neither reads nor changes
other targets' data or caches.

Training-scope validation:

- all selected training target caches must have the same `final_feature_schema_id`;
- all selected training target caches must point to the same `projection_id`;
- same projection type or same output count alone is not sufficient; the complete projection identity must match;
- per-target source fields may differ, but schema fields must match exactly;
- adding a target with a compatible raw feature schema must not invalidate or rebuild existing target caches.

Prediction validation:

- an active prediction target must use the same `final_feature_schema_id` as the trained classifier;
- if the active target cache has a different projection, even with the same output dimension, prediction must be
  blocked with a clear rebuild/reuse action.

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
Use `0` as unlabeled. Preserve integer class ids while the user edits annotations. Use consistent per-label colors
across target layers as a visual aid; colors and optional display names are not classifier compatibility metadata.

Acceptance criteria:

- annotation actions remain disabled until the card has a compatible feature cache;
- annotation layer shape always matches the highest-resolution image shape;
- annotation layers are associated with the target card image/channel cache context;
- nonzero label counts per class are tracked;
- target-card training eligibility is tracked separately from global classifier readiness;
- changing annotations marks classifier predictions stale.

4. Normalization and feature schema

Implement `normalization.py`, `features.py`, and `projection.py`. Normalize selected marker channels before caching,
extract deep features tile-wise, project only the deep-feature axis to 64 planes, then append those planes after the
normalized marker planes. In the UI, this work is presented as building or rebuilding a feature cache, not as producing
a user-facing feature matrix. The projection has no fit action or user-facing configuration. Tile-wise extraction must
be expressed as one lazy Dask graph from the source Zarr-backed array to the sidecar feature-cache Zarr. Do not call
`.compute()` once per spatial tile in a Python loop and do not materialize the complete source or feature array.

The two feature groups are intentionally complementary:

```text
normalized marker planes       direct biological intensity at the pixel
projected CNN feature planes   morphology and local-neighbourhood context
```

Keeping the marker planes is a feature-level skip connection around the pretrained CNN and projection. It preserves
simple marker rules and exact high-resolution intensities that may otherwise be transformed, spatially smoothed, or
mixed across projected coordinates. This matters because ConvNeXt was pretrained on natural RGB images, its nonlinear
and strided operations do not preserve every original marker value, and a 64-dimensional projection is deliberately
lossy. Marker normalization must be deterministic and recorded, but aggressive per-target normalization should be
avoided or explicitly evaluated because it can erase meaningful cross-target intensity differences.

Recommended Phase 1 defaults:

- selected channels: user-selected subset, with a product warning for very large selections;
- marker normalization: deterministic channel-wise clipping/scaling recorded in the manifest;
- deep feature backend: ConvNeXt-Tiny early-layer features behind the optional `torch` dependency group;
- ConvNeXt input handling: process each marker independently by transiently replicating it over RGB and applying only
  the pretrained weights' fixed ImageNet mean/std normalization;
- projection: deterministic dense Rademacher projection to `K = 64`, recorded as a reusable projection artifact;
- persistent feature dtype: `float16` unless validation shows it harms classifier quality.

For one concrete Dask block with `C` selected markers, reshape `(C, H, W)` to `(C, 1, H, W)` and treat marker identity
as the Torch batch coordinate during feature extraction. Restore the marker coordinate afterward and emit raw features
in a fixed marker-major order before projection. Replicate the marker batch over the three input channels, apply the
pretrained weights' mean/std, and discard that RGB tensor as soon as its selected intermediate features have been
extracted. The replicated tensor is a transient implementation detail: it must never be written to the feature cache.
Marker passes may be split into smaller batches when device memory is constrained. This intentionally does not mix
markers inside ConvNeXt; cross-marker combinations are learned later by the classifier from the projected features and
appended marker planes.

Use the following as the initial production batching pattern:

```python
import torch
from torchvision.models import ConvNeXt_Tiny_Weights


# marker_tile: Torch tensor (C, H, W), converted from one normalized Dask block
C, H, W = marker_tile.shape
marker_batch = marker_tile.reshape(C, 1, H, W)

# expand is a view, but the normalized three-channel result is transiently
# materialized. Bound the number of markers processed together.
input_transform = ConvNeXt_Tiny_Weights.DEFAULT.transforms()
mean = marker_batch.new_tensor(input_transform.mean).reshape(1, 3, 1, 1)
std = marker_batch.new_tensor(input_transform.std).reshape(1, 3, 1, 1)
max_marker_batch = 4
feature_chunks: dict[str, list] = {}

with torch.inference_mode():
    for marker_chunk in marker_batch.split(max_marker_batch, dim=0):
        rgb_chunk = (marker_chunk.expand(-1, 3, -1, -1) - mean) / std
        chunk_features = intermediate_extractor(rgb_chunk)
        del rgb_chunk
        for name, values in chunk_features.items():
            feature_chunks.setdefault(name, []).append(values)

# The extractor returns selected intermediate ConvNeXt feature maps. Restore the
# marker coordinate independently for every selected layer before spatial
# alignment, concatenation and deterministic projection.
layer_features = {
    name: torch.cat(chunks, dim=0).reshape(C, *chunks[0].shape[1:])
    for name, chunks in feature_chunks.items()
}
```

This marker-batching code runs inside the Dask block-inference function. Dask supplies one concrete NumPy block,
including its spatial halo; the block function converts that bounded block to Torch, runs the frozen extractor under
`torch.inference_mode()`, returns a NumPy feature block, and releases its CPU/GPU intermediates. Construct the model
once per execution worker, not once per Dask block.

Do not call the complete TorchVision classification transform on dense tiles: its resize/crop operations would change
the source grid. Use only the mean/std supplied by the selected weights. Persist the weights identifier, exact mean/std
values, RGB replication policy, and preprocessing implementation version in the raw feature schema. Marker chunk size
is a runtime memory control and is excluded from feature identity; tests must show that changing it preserves output
values within the declared tolerance.

**Dask-to-PyTorch execution contract**

Use the physical source chunk layout as the Phase 1 inference layout. Do not add an independently configurable spatial
tile planner. Keep these units separate:

```text
physical Zarr chunk    storage unit and initial spatial ConvNeXt inference core
expanded Dask block    one inference core plus the context halo
Torch marker batch     number of marker planes sent through ConvNeXt together
```

The source image remains a lazy Dask array. Select the requested markers, lazily combine only the marker axis into one
block, preserve the source spatial chunks, apply marker clipping/scaling lazily from the already computed percentile
statistics, add the required spatial halo, map the PyTorch extractor over those blocks, trim the halo and transient
edge padding, and regularize the resulting feature chunks for direct storage in the target cache Zarr. Build this
complete graph before executing it.

ConvNeXt contains strided operations. If two inference cores begin on different phases of that stride lattice, the
same source pixel can be grouped with different neighbours and acquire different features. Halo overlap supplies
neighbourhood context, but it cannot repair a shifted stride lattice. Define `stride` as the least common multiple of
the output strides of every selected ConvNeXt feature layer. For the usual power-of-two ConvNeXt stages this is simply
the largest selected output stride. Before graph construction, require every internal source-chunk boundary on both
spatial axes to lie on that common lattice:

```python
from itertools import accumulate


def internal_chunk_boundaries(chunks: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(accumulate(chunks[:-1]))


assert all(
    boundary % stride == 0
    for chunks in source_markers.chunks[-2:]
    for boundary in internal_chunk_boundaries(chunks)
)
```

For a regular source layout, this normally means that the non-terminal chunk size is a multiple of `stride`. The
naturally smaller terminal chunk need not itself have the original nominal size: its start is what matters. If an
internal boundary is not aligned, reject cache construction before model execution with a message that reports the
source layout, required stride, and offending boundary. Merely recording a misaligned layout would make the result
reproducible, but would not remove chunk-boundary seams.

The padded image dimensions must also be divisible by `stride`, otherwise local dense upsampling at the right or
bottom edge can use a different scale from a whole-image extraction. Add only the necessary `0 .. stride - 1`
right/bottom pixels, lazily and transiently, using zero in normalized-marker space:

```python
height, width = normalized.shape[-2:]
pad_bottom = (-height) % stride
pad_right = (-width) % stride

padded = da.pad(
    normalized,
    ((0, 0), (0, pad_bottom), (0, pad_right)),
    mode="constant",
    constant_values=0,
)

# da.pad may represent the new edge as a separate tiny Dask block. Fold that
# block into the terminal source core so the original, possibly unaligned
# image edge does not become a new internal inference boundary.
padded_chunks = (
    (C,),
    normalized.chunks[1][:-1]
    + (normalized.chunks[1][-1] + pad_bottom,),
    normalized.chunks[2][:-1]
    + (normalized.chunks[2][-1] + pad_right,),
)
padded = padded.rechunk(padded_chunks)
```

The `rechunk` immediately after `da.pad` is intentional and local to the image boundary. For example:

```text
source chunks                         (512, 512, 101)
after adding three padding pixels     (512, 512, 101, 3)
after folding the padding block       (512, 512, 104)
```

It must preserve every non-terminal chunk and concatenate only the small zero-padding blocks onto the rightmost
column, bottom row, and bottom-right corner of source chunks. It is a lazy graph transformation: it does not rewrite
the source Zarr, materialize the complete image, or redistribute the interior chunks. At execution time the affected
boundary tasks assemble the terminal source block and at most `stride - 1` zero-valued rows or columns in memory. Those
source blocks would already be read for feature extraction.

Keep this padding-fold rechunk conceptually separate from the later `allow_rechunk=True` overlap repair:

```text
padding-fold rechunk     removes the standalone padding block and aligns the padded image edge
overlap rechunk          merges/redistributes any inference cores too small to support the halo
```

The resulting terminal core has a size divisible by `stride` because both its start and the padded image end are
stride-aligned. Round the required model context up to the same lattice,
`halo = ceil(required_context / stride) * stride`. Consequently, a stride-aligned core start plus a stride-aligned halo
also gives a stride-aligned expanded-block start. Run tiled and canonical reference extraction with the same
right/bottom padding and fixed dense-upsampling convention, then crop the feature result back to the original
`(height, width)` before storage.

Chunk size and model input size are separate constraints. A physical source core may be smaller than the halo. In that
case `allow_rechunk=True` must lazily merge or redistribute adjacent cores until Dask can construct the overlap; this
applies to any undersized spatial core, not only the terminal core. The merged cores are logical inference blocks only
and do not rewrite the source Zarr. Their actual boundaries must still pass the stride checks, and their expanded sizes
must still fit the memory budget.

Independently, the selected extractor has a minimum spatial input size. Set the Phase 1 ConvNeXt-Tiny extractor
contract to `minimum_input_height = minimum_input_width = 32` and persist these values in the raw feature schema. Every
expanded block passed to Torch must be at least `32 x 32`. First validate the complete padded source, because an axis
held in one block has no neighbouring block with which it can be merged:

```python
minimum_input_height = 32
minimum_input_width = 32
padded_height, padded_width = padded.shape[-2:]

if padded_height < minimum_input_height:
    raise FeatureExtractionInputSizeError(
        "Feature extraction requires an input height of at least "
        f"{minimum_input_height} pixels; the padded source height is "
        f"{padded_height} pixels."
    )

if padded_width < minimum_input_width:
    raise FeatureExtractionInputSizeError(
        "Feature extraction requires an input width of at least "
        f"{minimum_input_width} pixels; the padded source width is "
        f"{padded_width} pixels."
    )
```

For example, when the common stride is `4`, a source width of `20` is already stride-aligned and receives no edge
padding. It must therefore fail before graph execution with:

```text
Feature extraction requires an input width of at least 32 pixels;
the padded source width is 20 pixels.
```

After overlap construction, also validate every actual expanded spatial block against the same `32 x 32` minimum.
This covers small physical chunks that Dask merged as well as edge blocks that have a halo on only one side.

Because block inference changes the leading axis from selected markers to `(C + 64)` cache planes, prefer explicit
overlap, block mapping, and trimming stages rather than relying on automatic output-shape inference:

```python
import dask.array as da
import numpy as np


# source_markers is Zarr-backed and remains lazy: (C, Y, X). Rechunk only the
# marker axis; its spatial chunks remain the source Zarr chunks.
source_markers = source_markers.rechunk({0: C})
normalized = normalize_dask(source_markers, percentile_statistics)

# Validate internal source boundaries and create `padded` as specified above.
# halo_y and halo_x are required-context values rounded up to `stride`.

# An axis held in one block has no internal block boundary and needs no Dask
# overlap. ConvNeXt still applies its normal model-boundary behavior there.
depth = {
    0: 0,
    1: 0 if padded.numblocks[1] == 1 else halo_y,
    2: 0 if padded.numblocks[2] == 1 else halo_x,
}
boundary = {0: "none", 1: "none", 2: "none"}

overlapped = da.overlap.overlap(
    padded,
    depth=depth,
    boundary=boundary,
    # Any source core smaller than the overlap requirement may need to be
    # merged/redistributed. Validate the resulting core lattice below.
    allow_rechunk=True,
)

if min(overlapped.chunks[1]) < minimum_input_height:
    raise FeatureExtractionInputSizeError(
        "At least one expanded inference block is shorter than the "
        f"required {minimum_input_height} pixels."
    )
if min(overlapped.chunks[2]) < minimum_input_width:
    raise FeatureExtractionInputSizeError(
        "At least one expanded inference block is narrower than the "
        f"required {minimum_input_width} pixels."
    )

# infer_convnext_block receives one in-memory NumPy block and returns
# (C + 64, block_y, block_x). Supplying meta prevents Dask from invoking
# PyTorch on synthetic 0-dimensional inputs during graph construction. Derive
# spatial output metadata from the actual post-rechunk overlap array, never
# from the original source chunk shape.
actual_pretrim_chunks = (
    (C + 64,),
    overlapped.chunks[1],
    overlapped.chunks[2],
)
mapped = overlapped.map_blocks(
    infer_convnext_block,
    dtype=np.float16,
    chunks=actual_pretrim_chunks,
    meta=np.empty((0, 0, 0), dtype=np.float16),
)

features_padded = da.overlap.trim_internal(
    mapped,
    depth,
    boundary=boundary,
)

# Remove only the transient right/bottom padding. Cache and annotation shapes
# remain exactly aligned with the original highest-resolution image.
features = features_padded[:, :height, :width]

# Automatic overlap rechunking may leave irregular inference-core chunks.
# Rechunk only the projected C + 64 output to the fixed cache layout; this does
# not rewrite the source or rerun ConvNeXt.
actual_inference_chunks = features_padded.chunks
cache_chunks = (C + 64, cache_chunk_y, cache_chunk_x)
features_for_store = features.rechunk(cache_chunks)

# target_cache_zarr is created with cache_chunks. This creates a lazy write
# graph; it does not load the complete feature array.
write_job = da.store(
    features_for_store,
    target_cache_zarr,
    compute=False,
)

# Execute exactly once in the existing background worker. The synchronous
# scheduler keeps GPU inference concurrency at one for the initial local path.
write_job.compute(scheduler="synchronous")
```

The final `write_job.compute(...)` is not an eager full-array conversion: it executes the store graph and writes one
output block at a time. A separate preliminary Dask reduction may compute the small set of full-image percentile
statistics once. The prohibited pattern is repeated spatial-tile `.compute()` calls.

`allow_rechunk=True` remains necessary whenever any spatial source core is smaller than the halo or otherwise cannot
support Dask's overlap construction. Dask may merge or redistribute several adjacent cores and may change the number
of blocks. Because the source internal boundaries, padded total shape, and halo are all multiples of `stride`, the
resulting cores are expected to remain on the common lattice, but do not rely on that as an undocumented Dask
guarantee. Treat `overlapped.chunks` and the chunks obtained after `trim_internal` as authoritative and reject the plan
before model execution if any resulting padded-core size or internal boundary is not divisible by `stride`, or if any
expanded block is smaller than the extractor's `32 x 32` minimum.

Do not pass output chunks predicted from the original source layout into `da.map_overlap`. When automatic rechunking
changes the block grid, stale `chunks=` metadata can fail graph construction or describe different boundaries from the
arrays returned at runtime, making downstream region writes unsafe. For this shape-changing operation from `(C, ...)`
to `(C + 64, ...)`, the explicit public sequence `overlap(..., allow_rechunk=True) -> map_blocks(chunks derived from
overlapped.chunks) -> trim_internal(...)` is intentionally safer than the `da.map_overlap` convenience wrapper. It is
the same overlap/map/trim algorithm while exposing the actual intermediate layout needed for correct output metadata.

Before execution, validate the actual plan rather than the requested plan:

- the largest expanded block, including all selected markers and transient Torch activations, fits the CPU/GPU memory
  budget;
- the complete padded source and every actual expanded block meet the extractor's `32 x 32` minimum input size;
- every source internal spatial chunk boundary is divisible by the common ConvNeXt stride;
- the halo, padded image dimensions, actual padded-core sizes, and actual padded-core boundaries are divisible by that
  stride;
- `features_padded` has shape `(C + 64, padded_Y, padded_X)` and cropping restores exactly `(C + 64, Y, X)`;
- `features_for_store` has regular chunks matching the fixed Zarr cache chunks.

The complete source chunk tuple is part of the target-specific cache manifest and cache identity. Compare the current
layout with the persisted tuple whenever opening or reusing a cache. Any difference invalidates that target's cache and
the UI should say that it is being recalculated because the source chunk layout changed. Record the actual
post-overlap core/expanded layout as well for diagnostics. The source layout is deliberately not part of the shared
`raw_feature_schema_id` or `final_feature_schema_id`: targets with different but valid stride-aligned source layouts
must still be poolable for classifier training. Cache invalidation is conservative even though the equivalence tests
below require valid aligned layouts to produce the same interior feature values.

Do not physically rechunk or rewrite the source image in Phase 1. The marker-axis combine, terminal padding fold, any
undersized-core repair required by Dask overlap, and final output regularization are lazy graph operations. An on-disk
staging/rechunk cache would require a full source read, duplicate storage, another progress and invalidation workflow,
and is outside this phase. Also do not introduce a PyTorch `DataLoader` for dense cache generation: Dask already owns
block scheduling, overlap, shared reads, and chunk-wise Zarr output. A DataLoader remains an option for future
independent-sample training workflows. The final `features.rechunk(cache_chunks)` is also lazy and operates only on the
already projected `(C + 64)` output so the Zarr cache has one regular physical chunk shape. Pre-create
`target_cache_zarr` with that exact shape/dtype/chunk layout and use `da.store`; do not ask `to_zarr` to select another
automatic write layout after this explicit regularization.

Acceptance criteria:

- feature cache planes are ordered as selected normalized marker planes followed by 64 projected deep-feature planes;
- raw high-dimensional deep features are streamed tile-wise and are not persisted blindly;
- production feature extraction replicates each marker transiently over RGB, applies the selected weights' mean/std,
  and never writes replicated RGB planes to the cache;
- marker batching can be reduced without changing feature-plane semantics when device memory is constrained;
- multichannel extraction preserves the declared marker-major raw-feature order when flattening and restoring the
  marker batch coordinate;
- cache building constructs one lazy Dask graph and does not call `.compute()` per spatial tile or on the full source
  or feature array;
- Dask overlap supplies the declared halo, block inference returns the expanded spatial shape, and trimming restores
  the padded shape without seams in the tested valid region; removing transient right/bottom padding then restores the
  exact highest-resolution source shape;
- every source internal spatial chunk boundary is stride-aligned, the halo is rounded up to the common extractor
  stride, and only `0 .. stride - 1` transient right/bottom pixels are added;
- `da.pad` edge blocks are folded into the terminal source cores so the original image edge does not become a new
  internal inference boundary;
- the padding-fold rechunk preserves all non-terminal spatial chunks and only assembles the right/bottom boundary
  blocks; it remains distinct from any subsequent undersized-core overlap rechunk;
- overlap may lazily merge or redistribute any source cores that are too small for the overlap requirement;
  block-output metadata is derived from the actual post-rechunk `overlapped.chunks`, never from the original source
  layout;
- the complete padded source and every expanded inference block are at least `32 x 32`; an undersized source or block
  is rejected before Torch inference with its actual and required dimensions;
- the actual expanded/core layouts pass memory, output-shape, and stride-grid validation before execution;
- irregular trimmed inference chunks are lazily rechunked only after projection to a regular `(C + 64, y, x)` cache
  layout matching the target Zarr chunks;
- a source with a non-stride-aligned internal boundary is rejected with the offending boundary and required stride;
- the complete source chunk tuple is stored in the target cache manifest, and changing it makes that cache stale and
  produces a clear UI explanation before recalculation;
- source padding and any undersized-core overlap repair remain lazy and never create an on-disk rechunked source copy;
- the initial local execution path keeps a single frozen ConvNeXt instance and at most one GPU inference task active;
- projection generation is deterministic and independent of target data, annotations, and cache-build order;
- independently built compatible target caches resolve to the same projection id and final feature schema id;
- adding a compatible target does not invalidate or rebuild existing caches;
- tests can run with a fake feature extractor so CI does not require downloading model weights;
- missing optional torch dependencies produce a clear install/action message.

5. Sidecar cache store

Implement `cache_store.py` and `manifest.py`. Store feature caches in an explicit sidecar zarr store:

```text
sample.harpy-cache.zarr/
  pixel_classification/
    feature_schemas/
      <final_feature_schema_id>/
        raw_feature_manifest
        projection/
          <projection_id>/
            manifest
            matrix
    feature_caches/
      <cache_id>/
        features
        manifest
```

Default cache-location behavior:

- backed writable local SpatialData: sibling `<source>.harpy-cache.zarr`;
- in-memory, remote, or read-only SpatialData: user must choose a writable cache location;
- cache discovery, reuse, and deletion must be visible in the widget;
- compatible cache discovery runs automatically after image/channel/cache-setting changes;
- annotation controls unlock only when the target card has a compatible cache;
- global classifier training can use only target cards with compatible caches.

Acceptance criteria:

- identical compatible requests resolve to the same `cache_id`;
- incompatible requests cannot silently reuse a cache;
- cache status is reported per target card as missing, found, building, ready, stale, or invalid/partial;
- pooled-training cache compatibility is validated through `final_feature_schema_id`, not through target-specific
  `cache_id`;
- caches with different projection ids are rejected even when both contain 64 projected planes;
- adding a target with a compatible schema reuses the existing projection identity without rebuilding other caches;
- cache creation is atomic enough that interrupted writes do not look valid;
- stale/partial caches are reported and can be deleted from the UI.

6. Shared classifier training and viewer-bound prediction

Implement `classifier.py` and `prediction.py`. Train from annotated pixels only, using rows read from the feature caches
belonging to the selected training scope. Predict over the active viewer target's high-resolution feature cache tile-wise
and return a high-resolution predicted label map for that target.

Recommended Phase 1 classifier:

- a small per-pixel multilayer perceptron (MLP) implemented behind the optional `torch` dependency group;
- input width `C + 64`, one hidden `Linear(C + 64, 64)` layer, `GELU`, `Dropout(p=0.1)`, and a final
  `Linear(64, N_classes)` layer;
- no batch normalization; fit a per-feature mean and standard deviation on the sampled training rows, persist that
  state with the classifier, and apply it unchanged during prediction;
- deterministic class- and target-balanced training-row sampling, a fixed training seed, `float32` training, and
  bounded epochs with validation-based early stopping where a valid spatial holdout can be formed;
- class labels are integer label IDs from the annotation layer;
- one fitted classifier is trained from pooled annotated pixels across the selected training scope;
- confidence is the maximum softmax score, presented as an uncalibrated model confidence rather than a calibrated
  probability;
- prediction writes should not modify the annotation layer.

The MLP is a fixed Phase 1 product choice, not another expert configuration surface. The UI should expose a single
training action and useful progress/error status, not hidden-layer, optimizer, dropout, epoch, projection, or seed
controls. Exact defaults must be versioned and recorded in classifier metadata.

This classifier matches the hybrid cache representation: explicit marker planes retain direct per-pixel biological
signals, projected CNN planes provide neighbourhood and morphology context, and the dense first layer can learn
arbitrary combinations across both groups. The fixed projection is lossy, so the roadmap should retain ablation tests
for marker-only, projected-CNN-only, and combined inputs; the neural network cannot recover information discarded by
the projection.

Training scope:

- default scope: all eligible selected samples / coordinate-system target cards;
- optional scope: user-selected subset of eligible target cards;
- eligible means cache-ready and sufficiently annotated;
- training status should report which target cards contributed training pixels and how many pixels/classes each
  contributed.

Prediction scope:

- interactive UI prediction is limited to the active viewer target;
- no Phase 1 UI control should run prediction across all selected target cards;
- multi-sample inference belongs to a headless workflow.

Acceptance criteria:

- training rejects empty, one-class, or otherwise insufficient selected training scopes with clear messages;
- the fitted classifier records the target cards and cache ids used for training;
- training rejects target-cache selections whose `final_feature_schema_id` or `projection_id` differ;
- training-row sampling prevents a large target or heavily annotated class from dominating by default and records the
  sampled counts per target and class;
- classifier input-standardization state, integer-class/output-index mapping, architecture version, optimizer and
  stopping parameters, and deterministic seed are persisted and reused for prediction;
- validation rows are separated by target or spatial block where practical rather than randomly interleaving adjacent
  pixels between training and validation;
- prediction is tile-wise and does not require flattening the whole image into memory at once;
- predicted class labels have the active target's highest-resolution image shape;
- classifier metadata records the final feature schema and projection ids, training cache ids, training scope, training
  class counts, training time, classifier architecture/parameters, fitted input-standardization state, class-index
  mapping, and model-state identity.

7. Widget workflow and background jobs

Implement `controller.py`, `status_card.py`, and `widget.py`. The UI should guide the user through source selection,
annotation, cache generation/reuse, shared classifier training, viewer-bound prediction, and save.

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
- training scope incomplete;
- training;
- trained classifier ready;
- no active prediction target;
- prediction ready;
- prediction stale;
- save succeeded;
- error with recovery action.

Long-running operations should run in background workers and expose progress where possible. Cache building, training,
prediction, and save should not freeze napari.

Acceptance criteria:

- controls are enabled only when the current state is valid;
- annotation controls stay disabled until target-card cache readiness;
- global classifier training stays disabled until the selected training scope is eligible;
- prediction stays disabled until a classifier is trained and an active viewer target has a compatible cache;
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
- prediction metadata records cache id, selected channels, normalization, feature extractor, projection, classifier
  parameters, training scope, active prediction target, class labels, and created timestamp.

Acceptance criteria:

- saved predictions reload as normal SpatialData labels;
- labels align with the highest-resolution source image in napari;
- backed SpatialData writes are persisted;
- failed writes do not leave ambiguous UI state.

9. Tests and verification

Add focused tests before broadening UI behavior:

- manifest canonicalization and `cache_id` stability;
- cache compatibility/rejection;
- deterministic projection generation and matrix-hash stability;
- projection id and final feature schema id validation;
- caches with different projection identities are rejected even when their output dimensions match;
- adding a compatible target preserves existing cache ids and does not rebuild other target caches;
- coordinate-system selection creates target-card state;
- image/channel target-card selection validates against coordinate-system availability;
- per-channel overlay load reuses existing layers;
- highest-resolution selection for single-scale and multiscale image elements;
- annotation validation and class-count tracking;
- training-scope selection pools eligible annotated target cards;
- normalization and feature-plane ordering;
- RGB replication applies the selected weights' exact mean/std without resize/crop and never writes RGB cache planes;
- changing the marker batch/chunk size preserves marker-major feature ordering and feature values within tolerance;
- fake extractor plus projection writes the expected `(C + 64, y, x)` feature-cache shape;
- a Zarr-backed Dask source reaches a Zarr feature target through one overlap/map/trim/store graph without per-block
  `.compute()` calls or full-array materialization;
- fake block inference is not invoked during graph construction when explicit `meta` is supplied;
- internal source-chunk boundaries divisible by the common extractor stride are accepted, while a deliberately
  misaligned boundary is rejected before the extractor runs with an actionable error;
- arbitrary source image dimensions are lazily padded on the right/bottom to the next stride multiple, the padding
  block is folded into the terminal source core, and the stored feature array is cropped back to the exact source
  shape;
- after padding and its explicit fold, every non-terminal chunk equals the corresponding source chunk, the terminal
  chunk equals `source_terminal_chunk + padding`, and no standalone padding chunk remains on either spatial axis;
- computing an interior padded-array block does not require a boundary source block, and padding-fold execution does
  not modify or create a rechunked source Zarr;
- padded/overlap/trim extraction matches a canonical equally padded whole-image reference for multiple valid physical
  source chunk layouts, including edge chunks;
- changing any entry in the complete source chunk tuple changes the target `cache_id`, marks the previous cache stale,
  and does not change the shared `final_feature_schema_id`;
- terminal and non-terminal cores smaller than the overlap requirement are lazily merged or redistributed without
  rewriting the source Zarr; both same-block-count and changed-block-count cases expose stride-aligned pre-trim and
  trimmed chunk metadata;
- a padded source axis smaller than 32 pixels is rejected before graph execution with the required and actual size,
  including the exact 20-pixel-width error-message contract documented above;
- every actual expanded block is validated against the `32 x 32` extractor minimum before the fake or real Torch
  extractor is invoked;
- shape-changing block inference derives `chunks=` from the actual overlapped array; deliberately stale metadata is
  rejected by validation rather than reaching Zarr storage;
- irregular inference-core chunks are lazily regularized after projection, output Zarr chunks match that regular layout,
  and partial/failed graph execution remains identifiable;
- MLP training/prediction on small synthetic data, including deterministic seeds and persisted input standardization;
- class- and target-balanced sampling behavior;
- integer class-id/output-index round trips for non-contiguous positive label ids;
- marker-only, projected-CNN-only, and combined-representation ablation harnesses;
- interactive prediction rejects non-active/batch target scopes;
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

10. Follow-up optimization: single-channel ConvNeXt stem

After the standard RGB-replication feature path is implemented and validated, add an optimized backend that folds RGB
replication and the pretrained weights' mean/std into a one-channel ConvNeXt stem. This is an implementation
optimization, not a different feature model: for a single normalized marker `x`, it should reproduce the reference
RGB stem within the declared numerical tolerance.

For original RGB stem weights `W_r`, bias `b`, pretrained input means `mean_r`, and standard deviations `std_r`, use:

```text
W_single = sum_r(W_r / std_r)
b_single = b - sum_r((mean_r / std_r) * sum_spatial(W_r))
```

The replacement is `Conv2d(1, stem_width, kernel_size=4, stride=4, padding=0)`. All subsequent pretrained ConvNeXt
layers remain unchanged and frozen. Use the following as the canonical initial TorchVision implementation; retain the
explicit structural checks so a future TorchVision layout change fails rather than silently converting the wrong
layer.

```python
import torch
from torch import nn


def convert_convnext_stem_to_single_channel(
    model: nn.Module,
    *,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> nn.Module:
    """Fold replicated-marker ImageNet normalization into ConvNeXt's stem."""
    try:
        old_conv = model.features[0][0]
    except (AttributeError, IndexError, TypeError) as exc:
        raise RuntimeError("Unsupported TorchVision ConvNeXt stem layout") from exc

    if not isinstance(old_conv, nn.Conv2d):
        raise RuntimeError("Expected model.features[0][0] to be Conv2d")
    if old_conv.in_channels != 3 or old_conv.groups != 1:
        raise RuntimeError("Expected an ungrouped three-channel ConvNeXt stem")
    if old_conv.padding != (0, 0) or old_conv.padding_mode != "zeros":
        raise RuntimeError("Exact stem conversion requires the unpadded ConvNeXt stem")
    if old_conv.bias is None:
        raise RuntimeError("Exact stem conversion requires a stem bias")

    weight = old_conv.weight
    mean_t = weight.new_tensor(mean).reshape(1, 3, 1, 1)
    std_t = weight.new_tensor(std).reshape(1, 3, 1, 1)
    if not torch.all(std_t > 0):
        raise ValueError("ConvNeXt input standard deviations must be positive")

    new_conv = nn.Conv2d(
        in_channels=1,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        dilation=old_conv.dilation,
        groups=1,
        bias=True,
        padding_mode=old_conv.padding_mode,
        device=weight.device,
        dtype=weight.dtype,
    )

    with torch.no_grad():
        new_conv.weight.copy_((weight / std_t).sum(dim=1, keepdim=True))
        normalization_offset = (weight * (mean_t / std_t)).sum(dim=(1, 2, 3))
        new_conv.bias.copy_(old_conv.bias - normalization_offset)

    model.features[0][0] = new_conv
    return model
```

Construct and freeze the optimized extractor using the metadata attached to the selected pretrained weights:

```python
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny


weights = ConvNeXt_Tiny_Weights.DEFAULT
input_transform = weights.transforms()
model = convnext_tiny(weights=weights)
model = convert_convnext_stem_to_single_channel(
    model,
    mean=tuple(input_transform.mean),
    std=tuple(input_transform.std),
)
model.requires_grad_(False).eval()
```

The optimized production path passes `(C, 1, H, W)` percentile-normalized marker batches from each Dask block directly
to the converted model. It must not replicate RGB or additionally apply ImageNet normalization because both operations
are already represented by the converted stem.

Pros:

- avoids materializing the transient normalized three-channel tensor;
- reduces input bandwidth and the first stem convolution from three input channels to one;
- retains independent per-marker processing and is mathematically equivalent to the reference preprocessing;
- can reduce peak device memory when many markers or tiles are batched together.

Cons:

- mutates a standard pretrained architecture and depends on TorchVision's internal stem layout;
- adds conversion code, compatibility checks, parameter hashing, and numerical-equivalence tests;
- may complicate checkpoint inspection, model tooling, export, and future TorchVision upgrades;
- floating-point summation order can produce small differences even though the operations are algebraically equivalent;
- likely provides limited end-to-end speedup because most ConvNeXt compute and memory occur after the stem;
- is unnecessary when RGB replication fits comfortably in memory with a suitably small marker batch.

Treat input handling as part of the raw feature schema. Record `single_channel_fused_stem`, the conversion
implementation version, original weights identifier/digest, exact mean/std, converted stem parameter hash, and numeric
precision. Do not claim bitwise compatibility with RGB-built caches; changing from RGB replication to the optimized
path creates a new `raw_feature_schema_id` unless a future cache-migration contract explicitly proves stronger
compatibility.

Slice 10 acceptance criteria:

- a focused test compares the converted stem with RGB replication plus mean/std over multiple inputs and shapes using
  `torch.testing.assert_close` with an explicit tolerance;
- selected intermediate feature maps from the complete frozen extractor also match the reference path within tolerance;
- the optimized production path accepts `(C, 1, H, W)` directly and does not create an RGB tensor;
- structural validation rejects unsupported ConvNeXt/TorchVision stem layouts;
- benchmarks report peak memory and end-to-end tile throughput for representative tile sizes and marker counts;
- the optimization remains disabled when its maintenance cost or measured benefit does not justify activation.

**Suggested Delivery Order**
1. Package skeleton, manifest registration, and empty widget.
2. Source resolution, target-card UX, per-channel overlays, and Phase 1 highest-resolution contract.
3. Annotation layer controller and class-count validation.
4. Manifest/cache-id model and sidecar cache store.
5. Fake feature extractor path with cache writing and reuse.
6. Real ConvNeXt feature extractor behind optional dependencies.
7. Shared classifier training and viewer-bound tile-wise prediction.
8. Viewer display and stale-state handling.
9. Save predicted labels to SpatialData.
10. Optional single-channel ConvNeXt stem optimization with equivalence tests and benchmarks.
11. Cache management UI, progress/cancel polish, and final test hardening.

This order keeps each slice independently testable while building toward the real user workflow.

**Phase 1 Completion Definition**
Phase 1 is done when a user can choose coordinate systems, configure target cards with image/channel selections, load
channel overlays for inspection, build or reuse a visible sidecar feature cache, annotate pixels at highest resolution,
train and predict without blocking napari, inspect high-resolution predicted labels in the viewer, and explicitly save
those labels back into SpatialData with reproducible metadata.

The implementation must leave clear extension points for Phase 2 multiscale compute, but Phase 1 should not expose
scale selection or save transformed low-resolution prediction layers.
