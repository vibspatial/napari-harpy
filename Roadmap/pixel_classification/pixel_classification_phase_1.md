Roadmap for pixel classification Phase 1.

This document turns the high-level design in `pixel_classification.md` into a scoped implementation plan for the
first production slice. Phase 1 is not a prototype. It should be a durable, tested, user-facing feature with a narrow
resolution contract: everything runs on the highest-resolution image grid.

**Phase 1 Scope**
Phase 1 delivers interactive pixel classification for one selected SpatialData image element at the highest available
resolution.

Included:

- select SpatialData image, coordinate system, and channels;
- create/manage a high-resolution napari `Labels` annotation layer;
- train from annotated pixels only, with `0 = unlabeled` and `1..N = classes`;
- extract tile-wise pixel features from selected channels;
- append normalized marker intensities to reduced deep features;
- write/reuse/delete a manifest-keyed sidecar zarr feature cache;
- train a classifier and predict class labels over the selected image;
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
- no hidden feature cache inside the SpatialData zarr store.

If the selected image element is multiscale, Phase 1 should resolve and use the highest-resolution scale only.
Scale selection belongs to Phase 2.

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
  spatialdata_io.py
  types.py

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

**Core Data Contracts**
Define immutable request/result objects in `core/pixel_classification/types.py`:

- `PixelClassificationSource`: SpatialData identity, image name, coordinate system, highest-resolution shape,
  axes, channel names, selected channels, dtype, and source element location.
- `PixelFeatureManifest`: canonical cache schema metadata used to compute a stable `cache_id`.
- `PixelFeatureCache`: opened zarr feature array plus manifest metadata.
- `PixelTrainingData`: sampled feature rows and nonzero class labels.
- `PixelPredictionResult`: high-resolution predicted class map, optional confidence/probability maps, and metadata.

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

Acceptance criteria:

- the widget appears in the napari manifest;
- `Interactive(..., widgets=("pixel_classification",))` can open it;
- importing `napari_harpy.core.pixel_classification` does not import Qt, napari, torch, or torchvision.

2. Source resolution and validation

Implement `spatialdata_io.py` and source-selection controller logic. Resolve image elements, coordinate systems, axes,
channel names, selected channels, and highest-resolution array shape. If the image is a multiscale `DataTree`, pick the
highest-resolution scale in Phase 1 and expose that decision in status/metadata.

Acceptance criteria:

- single-scale and multiscale images resolve to a concrete highest-resolution 2D channel image;
- invalid channel selections produce actionable validation messages;
- in-memory, remote, read-only, and backed SpatialData stores are distinguished for cache-location policy.

3. Annotation layer lifecycle

Implement `annotation_controller.py`. Create or reuse a napari labels layer matching the highest-resolution image grid.
Use `0` as unlabeled. Keep class labels, class names, and colors stable while the user edits annotations.

Acceptance criteria:

- annotation layer shape always matches the highest-resolution image shape;
- nonzero label counts per class are tracked;
- training is disabled until at least two classes have enough annotated pixels;
- changing annotations marks classifier predictions stale.

4. Normalization and feature schema

Implement `normalization.py`, `features.py`, and `reducer.py`. Normalize selected marker channels before caching, extract
deep features tile-wise, reduce only the deep-feature axis, then append normalized marker planes unchanged.

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
- cache discovery, reuse, and deletion must be visible in the widget.

Acceptance criteria:

- identical compatible requests resolve to the same `cache_id`;
- incompatible requests cannot silently reuse a cache;
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
- source incomplete;
- cache missing;
- cache building;
- cache reusable;
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
- cancel/shutdown prevents orphan callbacks from updating destroyed widgets;
- errors leave the widget in a recoverable state;
- feature cache reuse is explicit rather than surprising.

8. Save to SpatialData

Implement save logic in `spatialdata_io.py` or a small dedicated output module. The user-facing prediction result is a
high-resolution labels element.

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
2. Source resolution and Phase 1 highest-resolution contract.
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
Phase 1 is done when a user can open a SpatialData image, choose channels, annotate pixels at highest resolution, build
or reuse a visible sidecar feature cache, train and predict without blocking napari, inspect high-resolution predicted
labels in the viewer, and explicitly save those labels back into SpatialData with reproducible metadata.

The implementation must leave clear extension points for Phase 2 multiscale compute, but Phase 1 should not expose
scale selection or save transformed low-resolution prediction layers.
