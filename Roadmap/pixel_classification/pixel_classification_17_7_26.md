# Pixel Classification Roadmap: Usability-First Raw-Intensity Classifier

Date: 17 July 2026

This document replaces the implementation direction proposed in
`pixel_classification_phase_1.md` for the first usable pixel-classification
release. The earlier document remains useful as research for future feature
enrichment, but it front-loads deep-feature extraction, a large persistent
feature cache, and extensive cache-compatibility machinery before users can
train their first classifier.

The first release should instead optimize for a short, understandable workflow:

1. choose an image, image scale, and marker channels;
2. create or reload an annotation layer at exactly that image scale;
3. paint two or more classes;
4. train a Random Forest directly from the annotated raw marker intensities;
5. predict at the same selected scale;
6. review the result in napari;
7. explicitly write or reload the annotation and prediction Labels state in
   Zarr.

There is no feature-extraction step in this release. There is no pixel-feature
cache. A user should be able to understand why the classifier produced its
result: every training row is simply the vector of selected marker intensities
at one annotated pixel.

## Product Decision

Implement a practical raw-intensity pixel classifier before investigating
handcrafted or deep features.

The initial classifier contract is:

- one selected SpatialData image element;
- one selected coordinate system available on that image;
- one explicitly selected image scale;
- one or more selected image channels, with a stable order;
- one editable annotation raster at the selected scale;
- annotation and prediction dtype is `uint8`;
- `0 = unlabeled`, never implicit background, and `1..255` are explicit class
  IDs;
- one `sklearn.ensemble.RandomForestClassifier` trained only from nonzero
  annotation pixels;
- deterministic per-class sampling capped at 50,000 annotated candidates,
  followed by non-finite-row exclusion and combined with
  `class_weight="balanced_subsample"`;
- one prediction raster at the selected scale;
- annotation and prediction remain separate napari layers and separate
  SpatialData labels elements;
- annotation and prediction persistence is exposed as one paired Labels-state
  workflow with explicit write and reload actions;
- no intensity normalization in the first implementation;
- no handcrafted features, CNN features, PCA, projection, or feature cache;
- no automatic upsampling to `scale0`.

The first usable milestone supports one active classification target. Pooled
multi-target training is a follow-up slice after the single-target workflow is
reliable, followed by supported headless training and apply APIs. This is an
implementation order, not a permanent product limitation: first prove the
complete annotation, training, prediction, transform, and persistence contract
for one target, then compose the same Qt-free core across targets.

The single-target implementation must keep target identity explicit in its core
inputs rather than reading it from global widget state. It does not need to
implement pooled sampling early, but it must avoid an API that can only ever
describe one hard-coded image. This leaves pooled training additive rather than
a rewrite.

## Answer to the Scale and Rendering Question

Yes, napari supports the proposed approach.

If the user selects `scale3` with shape `(5000, 5000)`, the annotation layer can
be a normal, single-scale napari `Labels` layer with data shape `(5000, 5000)`.
The layer does not need to be upsampled to the shape of `scale0`. Napari layers
have `scale`, `translate`, and `affine` transformations, and napari uses those
transformations when rendering layers together in world coordinates.

For a simple pyramid where `scale3` is downsampled by 8 on both axes relative to
`scale0`, the napari annotation and prediction layers use an intrinsic layer
scale equivalent to `(8, 8)`, followed by the same image-to-coordinate-system
affine used by the source image. They therefore cover the same field of view as
the full-resolution image while storing and editing only `(5000, 5000)` pixels.

The same design applies to prediction:

```text
multiscale source image
  scale0: high-resolution display source
  scale3: selected classification source, shape (Y3, X3)

annotation Labels layer
  data shape: (Y3, X3)
  editable: yes
  displayed through: selected-grid transform -> image transform -> viewer world

prediction Labels layer
  data shape: (Y3, X3)
  editable: no
  displayed through: the same composed transform
```

### Editing coordinate contract

Napari does not paint in screen coordinates. A mouse event has a position in
viewer/world coordinates, and napari applies the inverse Labels-layer transform
to obtain the position in the annotation array. Painting then changes a pixel
in the selected-scale `(Y3, X3)` array.

For a regular 8x-downsampled `scale3`, the forward and inverse mappings are
conceptually:

```text
rendering:
annotation pixel (y, x) -> scale by (8, 8) -> source-image/world position

editing:
mouse world position -> inverse source-image transform -> divide by (8, 8)
                     -> annotation-array position (y, x)
```

Napari-harpy must calculate and supply this transform. Napari does not infer an
8x factor from the name `scale3`. If the transform is absent or incorrect, the
annotation will be rendered and edited in the wrong location.

Training does not require a world-coordinate lookup because the selected image
and annotation use the same selected-scale grid:

```text
annotation[y, x] = class ID
image[:, y, x]   = raw marker-intensity training row
```

Thus annotation pixel `(y, x)` always labels image pixel `(y, x)` at the
selected scale. At 8x downsampling, that one annotation pixel represents the
corresponding approximately `8 x 8` block of `scale0` pixels. Boundaries will
therefore look blocky when viewed at scale0 resolution; this is the expected
meaning of classifying at `scale3`, not a rendering error.

This is preferable to storing annotation or prediction as a multiscale labels
pyramid. The user edits one explicitly chosen grid, and a single-scale napari
Labels layer is the simplest editable representation. A multiscale output could
be generated later for export or faster overview rendering, but it must not be
the source of truth for annotation.

When persisted into SpatialData, each annotation or prediction is a single-scale
`xarray.DataArray` parsed as a `Labels2DModel`. Its transformation maps that
labels element's intrinsic pixel grid into the selected coordinate system. The
transformation is composed from:

1. the selected-level grid to the source image's intrinsic grid; then
2. the source image's existing transformation to the selected coordinate
   system.

Conceptually:

```text
labels pixel grid
    -- selected_grid_to_image_intrinsic --> source image intrinsic coordinates
    -- source_image_to_coordinate_system --> selected coordinate system
```

For a regular full-image pyramid, the first component is normally a `Scale`.
The implementation must nevertheless derive it from the selected scale's
regular `x` and `y` coordinates and its relationship to `scale0`; it must not
assume that the name `scale3` means a factor of 8. This supports anisotropic
downsampling and detects unexpected offsets or irregular coordinates.

If the selected grid has a real offset as well as a different spacing, the
selected-grid transform is scale plus translation, represented as an affine or
a SpatialData `Sequence`. The napari layer transform and the persisted
SpatialData transform must describe the same mapping.

Pixel-center conventions require focused alignment tests. SpatialData raster
coordinates represent pixel centers, while napari renders array indices through
its layer transform chain. The implementation should use the conventions of the
existing viewer adapter and verify alignment at corners, the center, and class
boundaries rather than adding an untested half-pixel correction.

References:

- napari layer `scale`, `translate`, and `affine` transformations:
  <https://napari.org/stable/getting_started/layers.html#scaling-layers>
- SpatialData transformations and composition:
  <https://spatialdata.scverse.org/en/stable/tutorials/notebooks/notebooks/examples/transformations.html>
- SpatialData raster models:
  <https://spatialdata.scverse.org/en/stable/api/models.html>

## User Workflow

The first production workflow should be linear and require few decisions.

### 1. Choose the target

The user chooses:

- coordinate system;
- image element;
- image scale;
- one or more channels.

For a single-scale `DataArray`, the only scale is presented as `scale0`. For a
multiscale `DataTree`, the scale selector lists the actual keys in the element,
such as `scale0`, `scale1`, and `scale3`.

Every scale option should show its shape and relative spacing:

```text
scale0 — 40000 x 40000 — highest resolution
scale1 — 20000 x 20000 — 2x downsample
scale3 —  5000 x  5000 — 8x downsample — recommended
```

Use one simple recommendation heuristic: mark the highest-resolution available
scale whose total spatial pixel count does not exceed `8192 * 8192`. The user
may still choose any available scale. If no scale satisfies the heuristic, mark
the coarsest available scale as recommended and do not introduce a separate
warning or hard-stop policy in the first implementation.

The selected scale is part of the classifier context. Changing scale invalidates
the current trained classifier and prediction. Dirty Labels state must be
written, explicitly discarded, or retained by cancelling before the target grid
changes.

### 2. Inspect selected channels

The user can load selected markers as channel overlays through the existing
viewer-adapter behavior. Loading an overlay does not alter training state.
Changing the actual selected channel set or order makes the classifier and
prediction stale, but it does not invalidate annotations because annotations
are bound to the source image grid, not to a feature schema.

The channel selector should preserve image channel order by default. The UI
should show the order because it is the Random Forest feature-column order.

### 3. Create or reload annotations

The user chooses either:

- `Create annotations`; or
- `Reload Labels State` from the active backed SpatialData Zarr store.

Creating annotations allocates one zero-filled, in-memory `uint8` array with the
selected `(y, x)` shape and adds it to napari as a single-scale editable Labels
layer. Annotation does not depend on a cache or trained classifier.

Reloading persisted annotations loads their values into an editable working layer.
The implementation must not rely on mutating a backed Dask/Zarr array directly
for every brush operation. The working annotation array is an explicit editable
session copy, and `Write Labels State` persists the accepted state.

Changing the source image or selected scale requires a new compatible
annotation layer. The implementation must not silently resample annotations.

### 4. Define classes and paint

The widget provides a small shared class editor:

- class ID in the range `1..255`;
- class name;
- class color;
- annotated-pixel count.

`0` is reserved for unlabeled pixels and is never a trainable class. The first
release supports at most 255 classes because annotation and prediction use
`uint8`. Class names and colors are not Random Forest inputs, but they are
required product metadata and prevent users from confusing integer meanings.

Background is an ordinary explicit class with an ID in `1..255`, for example
`1 = Background`. Unpainted pixels remain unlabeled and must never be inferred
to be background training data. Users should paint small, diverse background
regions rather than one large homogeneous area. After reviewing a prediction,
they should be able to add misclassified regions as hard background examples
and retrain.

The native napari paint, erase, fill, and polygon-paint interactions should be
used where possible. Brush size is expressed in selected-grid pixels; the UI
may additionally show its approximate footprint in `scale0` pixels or physical
units.

Annotation edits mark the trained classifier and prediction stale. They do not
automatically retrain on every brush stroke in the initial release. The user
presses `Train` explicitly, which keeps expensive work predictable.

### 5. Train

Training reads only nonzero annotation positions. For each annotated position,
the training row is:

```text
[raw intensity channel_1, raw intensity channel_2, ..., raw intensity channel_C]
```

No image neighborhood, filter response, normalization, or deep feature is
included.

The initial product defaults should match the established object-classification
defaults where appropriate:

```text
RandomForestClassifier(
    n_estimators=100,
    random_state=0,
    n_jobs=-2,
    class_weight="balanced_subsample",
)
```

These are versioned product defaults, not expert controls in the initial UI.

Sampling and extraction follow an explicit bounded contract:

1. Use the selected-scale annotation raster already loaded in memory for napari
   editing. If persisted annotations were reloaded, complete that reload into
   the editable Labels layer before training. Scan this entire `uint8` raster to
   calculate the exact number of annotated pixels for every nonzero class ID;
   this step must not read any marker-image data.
2. From the annotation raster alone, select without replacement at most 50,000
   candidate positions for each class using a fixed seed. Do not construct a
   complete coordinate array or `sparse.COO` representation for a large densely
   painted class merely to select those candidates. Bounded
   reservoir/batch sampling or deterministic rank-based multi-pass selection
   are acceptable implementation strategies.
3. Only after candidate positions have been selected, group them by source-image
   chunk and read the selected marker channels from chunks containing those
   positions. Do not read every marker over the complete selected-scale image,
   and avoid one independent Dask random-indexing task per position or channel.
4. Convert gathered rows to `float32` with shape `samples x channels` and retain
   a row only when every selected channel value is finite.
5. Exclude a candidate when any selected channel contains `NaN`, `+inf`, or
   `-inf`. Do not sample replacements in the first implementation; train with
   the valid rows remaining from the single candidate sample.

Training requires at least two classes with at least one valid row after
extraction. To keep training bounded and retain diverse examples, every class
contributes at most 50,000 annotated candidates before finite-value filtering.
Classes with 50,000 or fewer annotated pixels sample all of them. Do not reduce
every class to the size of the smallest class, because a small foreground
annotation should not force the classifier to discard useful background
diversity. The cap is a versioned product default, not a main-UI control; later
benchmarks may justify changing that default in a subsequent version.

The remaining imbalance after capping is handled by
`class_weight="balanced_subsample"`. Each Random Forest tree calculates
inverse-frequency class weights from its bootstrap sample, so an arbitrarily
large painted Background region does not dominate split decisions merely due to
its area. This weighting compensates for annotation imbalance; it does not make
limited or homogeneous annotations representative.

The immutable extraction result and UI status card report, per class:

- exact annotated-pixel count;
- number of candidate pixels sampled;
- valid samples used;
- non-finite sampled candidates excluded;
- whether the annotated-candidate cap was applied.

For a capped class, do not report an exact total number of valid annotated
pixels because only the sampled candidates had their marker values read. A
suitable status is:
`Background (class 1): 50,000 candidates sampled from 812,430 annotated pixels
(capped); 37 non-finite rows excluded; 49,963 used`.

When a class is not capped, every annotated position was sampled, so its
complete valid and non-finite counts are known. These counts are informational;
training is blocked only when fewer than two classes retain a valid sample.

Training therefore performs a complete scan of the small editable annotation
raster, followed by bounded reads from the much larger multiplex source. It
reads only the selected marker channels and source chunks required by candidate
pixel positions. It must not materialize the complete multiplex image merely to
gather training rows.

### 6. Predict

`Predict` applies the trained Random Forest to all pixels at the selected image
scale. Allocate one in-memory `uint8` output array with the selected `(y, x)`
shape, then compute prediction in chunks or tiles:

1. read a bounded block of selected raw channels;
2. reshape it from `(C, block_y, block_x)` to `(pixels, C)`;
3. calculate a row mask for which every selected channel is finite;
4. initialize the output rows to class ID `0` and predict integer class IDs only
   for finite rows;
5. reshape to `(block_y, block_x)`;
6. write into the selected-scale output array;
7. report progress, the number of non-finite pixels left as `0`, and honor
   cancellation between blocks.

The complete multiplex source and a full `pixels x channels` feature matrix
must never be held in memory simultaneously. Tile-wise execution bounds source
input and temporary feature-matrix memory; it is not a mechanism for making the
prediction output lazy or disk-backed.

At the `8192 * 8192` recommendation threshold, the output array uses 64 MiB.
The first implementation does not introduce temporary prediction stores,
direct block writes to Zarr, or lazy Dask prediction layers. The user may still
select another scale under the general scale-selection contract.

The first implementation produces only the integer class map. A confidence or
probability image is a later addition because it doubles output, display,
persistence, and provenance concerns without being required for the core
workflow.

The prediction appears as a separate, read-only napari Labels layer with the
same shape and transform as the annotation layer. Prediction never modifies the
annotation data.

### 7. Write and reload Labels state

Follow the persistence interaction already used by Object Classification, but
treat the pixel annotation and prediction elements as one UI-level consistency
unit. The widget exposes two explicit actions:

- `Write Labels State` writes the editable annotation element and any present,
  complete prediction element to the active backed SpatialData Zarr store;
- `Reload Labels State` replaces the in-memory annotation and prediction layers
  with their persisted state from that store.

The annotation and prediction remain distinct single-scale SpatialData labels
elements with different element names and roles. The annotation element is the
required part of the pair. A prediction is optional: writing before prediction
exists writes annotations only, and reloading a state with no persisted
prediction restores annotations without creating a prediction layer. If a local
prediction layer exists but no prediction is present in the persisted state,
reload removes the local prediction layer.

Brush strokes and prediction generation update only the in-memory layers; they
never write through to Zarr automatically. Annotation edits, class-metadata
edits, and creating, replacing, or removing a prediction mark the Labels state
dirty. A write clears dirty state only after every element required for that
write succeeds. A failed write keeps the in-memory layers intact and the state
dirty.

`Reload Labels State` reloads immediately when the current state is clean. When
it is dirty, it uses the Object Classification three-way decision:

1. `Write labels state and reload`;
2. `Reload labels state and discard local edits`;
3. `Cancel`.

Leaving the current image, scale, coordinate system, or other target-defining
selection while dirty uses the equivalent choices: write and continue, discard
and continue, or cancel the target change. Reload and discard applies to both
the annotation and prediction layers; it never retains one local member of the
pair while reloading the other.

A complete but stale prediction may remain visible or be written only if it is
clearly marked stale and retains the classifier and annotation-revision
provenance needed to explain that status. Reload must not present such a
prediction as current. No periodic recovery snapshot is part of the first
implementation.

Persisted metadata should include:

- Harpy pixel-classification schema version;
- role: `annotation` or `prediction`;
- source image element name;
- the selected-resolution descriptor defined below;
- selected labels shape;
- selected coordinate system;
- ordered channel names;
- selected-grid-to-image transform description;
- class IDs, names, and colors;
- for predictions, Random Forest parameters and model identity;
- for predictions, training annotation identity and class counts;
- creation time;
- napari-harpy, SpatialData, scikit-learn, NumPy, and Dask versions relevant
  to reproduction.

The first implementation does not create a multiscale prediction pyramid and
does not upsample the result to `scale0`. A later explicit export action may
offer nearest-neighbor upsampling or pyramid construction, with provenance that
records the operation.

## Annotation and Prediction Lifecycle

Annotations and predictions are intentionally separate objects.

```text
source image + selected scale
            |
            +--> editable annotation layer ----> optional annotation labels element
            |
            +--> trained Random Forest
                       |
                       +--> read-only prediction layer ----> prediction labels element
```

### Annotation identity

An annotation is compatible with a target when all of the following match:

- source image element identity;
- selected-resolution descriptor;
- selected coordinate system binding;
- annotation schema version.

Channel selection is deliberately not part of annotation identity. A user may
paint tissue classes once and compare classifiers trained from different marker
subsets on the same grid.

### Classifier identity

A trained classifier records:

- source image and selected-resolution descriptor used for the training run;
- ordered selected channel names;
- class schema;
- sampled class counts;
- Random Forest parameters and fitted estimator;
- library versions;
- annotation revision used for training.

Changing annotations, channel selection, channel order, image, or scale marks
the classifier stale. Changing only display colors does not.

### Selected-resolution descriptor

Annotation, prediction, and classifier metadata use the same resolution
descriptor. For example:

```yaml
resolution:
  scale_key: scale3
  shape_yx: [5000, 5000]
  scale0_shape_yx: [40000, 40000]
  relative_spacing_to_scale0_yx: [8.0, 8.0]
  selected_grid_to_scale0_affine: [...]
```

`scale_key` records the exact DataTree key selected by the user, but the key is
provenance local to that image and is not a portable resolution identifier.
`relative_spacing_to_scale0_yx` is dimensionless and is derived from regular
grid coordinates rather than inferred from the scale name. The affine preserves
the complete selected-grid mapping, including anisotropic scale and any offset.

For a single-scale `DataArray`, use `scale_key: null`, set both shapes to the
same `(y, x)` shape, use relative spacing `[1.0, 1.0]`, and store the identity
selected-grid-to-scale0 affine.

Do not add physical-pixel-spacing or physical-unit fields. They are not assumed
to be available, and napari-harpy must not infer physical units from a
coordinate-system name or an otherwise unitless transform.

### Prediction identity

A prediction records:

- the classifier identity;
- target source image and selected-resolution descriptor;
- output shape and dtype;
- creation state: running, complete, cancelled, or failed.

Only complete predictions can be written. A cancelled or failed working output
array is discarded and cannot be mistaken for a valid prediction.

## Scope of the First Usable Release

Included:

- one active classification target;
- exact scale selection for multiscale images;
- selected scale shape and downsampling summary;
- multi-channel raw-intensity input;
- channel overlays;
- editable single-scale annotation layer;
- named and colored classes backed by `uint8` IDs in the range `1..255`;
- explicit Random Forest training;
- deterministic per-class capped sampling and class-balanced Random Forest
  weighting;
- tile-wise full-target prediction at the selected scale;
- read-only prediction layer;
- paired `Write Labels State` and `Reload Labels State` actions for backed
  SpatialData;
- explicit stale-state and dirty-state handling;
- background-worker training, prediction, and persistence operations;
- focused model, transform, widget-state, and persistence tests.

Excluded:

- handcrafted intensity, edge, texture, or morphology features;
- ConvNeXt, DINO, JAFAR, or other deep features;
- feature caches or sidecar Zarr stores;
- normalization, clipping, log, or asinh transforms;
- automatic retraining after every brush stroke;
- confidence/probability-map output;
- prediction upsampling to `scale0`;
- automatic multiscale prediction-pyramid generation;
- 3D, time, or arbitrary non-`(c, y, x)` image axes;
- classifier hyperparameter controls in the main UI;
- pooled multi-target training in the first usable milestone;
- public headless pixel-classifier training and apply APIs in the first usable
  milestone;
- interactive batch prediction over several targets.

## Package Direction

Keep pixel classification separate from object classification, but reuse shared
SpatialData, validation, styling, palette, and viewer-adapter helpers.
Reuse the current core-classifier semantics for ordered input columns,
finite-row masking, class ID `0` for rows that cannot be classified, and
structured result summaries. Implement pixel sampling and block reshaping as
explicit, testable functions; do not introduce a generic N-dimensional sklearn
wrapper or copy the exploratory `NDSparseClassifier`/`NDDaskClassifier`
abstractions from `ilastik-napari`.

Proposed package direction:

```text
src/napari_harpy/core/pixel_classification/
  __init__.py
  source.py          # image/scale/channel resolution and grid transforms
  annotations.py     # class schema and annotation validation
  classifier.py      # training-row extraction and Random Forest training
  prediction.py      # tile planning and prediction
  output.py          # SpatialData labels creation and provenance

src/napari_harpy/widgets/pixel_classification/
  __init__.py
  controller.py      # jobs, state snapshots, and stale/dirty transitions
  status_card.py
  widget.py

src/napari_harpy/headless.py  # later thin public wrappers over the core
```

The core package remains importable without Qt or napari. It may depend on
NumPy, Dask, xarray, SpatialData, and scikit-learn through existing project
dependencies. Napari layer creation, Qt workers, user prompts, and viewer event
connections belong in the widget package or viewer adapter.

Follow the object-classification headless architecture: training, compatibility
validation, prediction, bundle IO, and SpatialData output belong in shared
Qt-free core helpers. The interactive widget adds background workers, progress,
cancellation, prompts, and viewer layers. Public headless functions are thin,
synchronous wrappers around the same core and must not maintain a second
training or prediction implementation.

Generic image-scale and transformation helpers should be added to
`core/spatialdata.py` when they are useful beyond pixel classification. Reuse
the histogram widget's real scale-key selection pattern and the viewer adapter's
SpatialData-to-napari affine conversion rather than introducing a parallel
interpretation of image elements.

## Implementation Slices

Each slice should leave the code in a coherent, testable state. The first
end-to-end usable milestone is complete after Slice 6. Slice 7 adds a portable
bundle, Slice 8 adds pooled training, and Slice 9 exposes the same single- and
multi-target core through supported headless APIs. These follow-up slices do not
block validation of the first single-target release.

### Slice 1: Selected-scale and transformation foundation

Implement the non-Qt source-grid contract first.

Deliver:

- resolve `DataArray` versus `DataTree` image elements;
- enumerate actual scale keys;
- resolve one scale to a concrete `DataArray`;
- require supported `(c, y, x)` data and regular `x`/`y` coordinates;
- resolve channel names and indices in stable order;
- calculate selected scale shape, dtype, chunks, and relative spacing;
- select the highest-resolution scale with at most `8192 * 8192` spatial pixels
  as the recommended default, falling back to the coarsest scale;
- derive selected-grid to source-image-intrinsic scale/affine from coordinate
  vectors;
- compose that mapping with the image's SpatialData transformation;
- convert the result into equivalent napari layer transform components;
- reject irregular, non-finite, or unsupported grids with actionable errors.

Acceptance criteria:

- single-scale images expose only `scale0`;
- every `DataTree` scale is selectable by its actual key;
- the recommendation heuristic is deterministic and does not prevent selecting
  another scale;
- anisotropic scale factors are preserved;
- no scale factor is inferred from the numeric suffix of a key;
- a single-scale labels layer aligns with the selected level of a multiscale
  image at corners, center, and known boundaries;
- alignment works with identity, translation, scale, and general supported 2D
  affine image transformations;
- the persisted SpatialData transformation and napari rendering transform
  represent the same mapping;
- tests cover pixel-center behavior and fail on a deliberate half-pixel shift.

### Slice 2: Widget shell and target selection

Register `PixelClassificationWidget` and implement the first real selection
surface.

Deliver:

- shared `HarpyAppState` binding;
- coordinate-system selector;
- image selector filtered to the coordinate system;
- scale selector populated from the selected image;
- scale summary containing shape, relative spacing, and a recommended marker;
- multi-select channel selector preserving source order;
- per-channel `Load overlay` action using stable viewer layer names;
- one status card with the next valid action;
- disabled annotation/training/prediction controls until their prerequisites
  are met.

Acceptance criteria:

- target changes are reflected without duplicate viewer overlays;
- invalid axes, missing channel names, or irregular grids are explained in the
  widget;
- the recommended scale is preselected when the target has no remembered scale
  choice;
- no feature or cache terminology appears in the UI.

### Slice 3: Editable annotation lifecycle and class editor

Deliver:

- create a zero-filled, single-scale in-memory `uint8` Labels layer at selected
  shape;
- apply selected-grid and image transforms to the layer;
- mark it as the active editable layer;
- reserve `0` for unlabeled pixels;
- add, rename, recolor, and select class IDs in the range `1..255`;
- show live annotated-pixel counts per class;
- track annotation revision and dirty state;
- prompt to write, discard, or cancel when switching target, scale, or closing
  a dirty Labels state;
- keep annotation available before training and without any cache.

Acceptance criteria:

- painting at selected scale changes only selected-grid pixels;
- the visual brush footprint aligns with the source image;
- erasing restores `0`;
- unlabeled `0` pixels are excluded from training and never treated as implicit
  background;
- an explicitly painted Background class behaves like any other trainable class;
- two classes can be painted and counted;
- predictions cannot write into the annotation layer;
- channel changes preserve annotations but mark any classifier stale;
- scale changes never silently resample annotations.

### Slice 4: Raw-intensity Random Forest training

Implement training independently of prediction.

Deliver:

- scan the complete in-memory annotation raster and count annotations per class
  without reading marker data or constructing unbounded coordinate arrays;
- deterministically generate bounded candidate positions per class from that
  annotation raster;
- only then group candidate positions by source chunk and read the selected
  marker values needed for those positions;
- construct a bounded `samples x channels` float32 matrix;
- exclude non-finite rows without replacement sampling and train with the valid
  rows remaining from the single candidate sample;
- validate at least two classes with valid training rows;
- report per-class annotated, sampled, used, non-finite-excluded, and capped
  fields in the UI status card;
- train the fixed Random Forest with `class_weight="balanced_subsample"` in a
  background worker;
- retain class mapping, channel order, sample counts, parameters, versions, and
  annotation revision in an immutable training result;
- expose clear states: insufficient annotation, ready to train, training,
  trained, stale, and error.

Acceptance criteria:

- the training matrix contains raw selected intensities only;
- selected channel order equals model input-column order;
- deterministic inputs produce deterministic sampling and predictions;
- a class with more than 50,000 annotated pixels samples exactly 50,000
  candidates and is marked as capped;
- its used count equals the sampled count minus non-finite exclusions, with no
  replacement sampling;
- a capped class reports its exact annotated count but does not claim an exact
  total valid count for the unsampled annotations;
- a class at or below the cap samples all its annotations, reports complete
  valid and non-finite counts, and is not marked as capped;
- densely painted classes do not require a complete coordinate array or sparse
  label copy;
- annotation counting and initial position sampling do not trigger marker-image
  reads;
- sampled source reads are grouped by chunk rather than expressed as one Dask
  random-indexing task per position or channel;
- imbalanced sampled counts receive inverse-frequency per-tree class weights;
- a large painted class does not dominate a small class merely because of
  annotation area;
- training does not load the full image;
- worker cancellation and widget destruction cannot apply late results.

### Slice 5: Tile-wise prediction and review

Deliver:

- plan bounded prediction blocks from the selected scale's Dask/Zarr layout;
- allocate one in-memory selected-scale `uint8` prediction array;
- read selected channels block-wise;
- finite-mask every `pixels x channels` block and predict only finite rows;
- leave rows containing any `NaN`, `+inf`, or `-inf` as class ID `0` and report
  their count;
- fill the in-memory output with predicted class IDs;
- show progress and permit cancellation between blocks;
- add or update one read-only prediction Labels layer with the same transform
  as annotations;
- track prediction freshness against classifier and target revisions;
- keep annotation and prediction color mappings synchronized by class ID.

Acceptance criteria:

- prediction never flattens the complete multiplex source into memory;
- output shape is exactly the selected scale's `(y, x)` shape;
- output is one in-memory `uint8` array, not a temporary Zarr store or lazy Dask
  layer;
- output contains declared `uint8` class IDs and no classifier-output indices;
- non-finite input rows are consistently represented by class ID `0` and are
  never passed to the classifier;
- prediction aligns with annotation and source image;
- a changed annotation or channel selection marks prediction stale;
- cancellation cannot produce a persistable apparently complete result;
- re-predict updates the existing prediction layer instead of adding duplicates.

### Slice 6: SpatialData annotation and prediction persistence

This slice completes the first end-to-end usable milestone.

Deliver:

- `Write Labels State`, which writes the annotation and any present complete
  prediction as one UI-level persistence action;
- `Reload Labels State`, which reloads the annotation and prediction together;
- the same write/reload/discard/cancel interaction pattern as Object
  Classification;
- validated, distinct annotation and prediction element names with overwrite
  confirmation;
- `Labels2DModel.parse(...)` creation from selected-scale arrays;
- composed transformation from labels intrinsic coordinates to the selected
  coordinate system;
- Harpy provenance metadata for annotation and prediction roles;
- backed `SpatialData.write_element(...)` support;
- dirty tracking for annotation edits, class-metadata edits, and prediction
  creation, replacement, or removal;
- clear behavior for in-memory, read-only, and failed writes;
- reload-and-align verification.

Acceptance criteria:

- persisted elements are single-scale `xarray.DataArray` labels, not hidden
  caches;
- annotations reload into an editable working layer at the same selected grid;
- predictions reload as normal SpatialData labels and align with the source;
- writing succeeds when annotations exist but no prediction exists;
- reloading a state without a persisted prediction removes any local prediction
  and does not create a replacement prediction layer;
- dirty reload offers write-and-reload, reload-and-discard, and cancel;
- leaving a dirty target offers write-and-continue, discard-and-continue, and
  cancel;
- cancelling either prompt leaves the target and both layers unchanged;
- dirty state clears only when all elements required by the write succeed;
- no upsampling occurs during write;
- annotation and prediction names cannot collide silently;
- failed writes leave the in-memory working layers intact and the UI
  recoverable;
- persisted metadata is sufficient to explain the source image, scale, channels,
  classes, and model used.

### Slice 7: Classifier bundle export and usability hardening

Classifier persistence is not required to prove the first end-to-end flow, but
it is important for a genuinely useful product and should follow immediately.

Deliver:

- export the fitted Random Forest and its compatibility metadata in a versioned
  Harpy classifier bundle;
- load and validate a bundle against an active target;
- match channels by unique name and reorder inputs to the saved model schema;
- serialize and validate the selected-resolution descriptor without physical
  spacing or unit fields;
- show a non-blocking warning for cross-image relative-spacing mismatches;
- warn that raw-intensity classifiers assume comparable acquisition and
  intensity ranges;
- prediction-only use without retaining the original annotation layer;
- focused progress, cancellation, dirty-state, and error-message polish;
- user documentation describing scale choice and raw-intensity limitations.

Compatibility must not require the new target to have the same spatial shape.
It requires a compatible ordered channel schema after name-based resolution and
the selected-resolution descriptor described above.

For reuse on the original source image, require the recorded scale key and grid
descriptor to match. For reuse on another image, do not compare scale-key strings
as if they had shared meaning. Compare
`relative_spacing_to_scale0_yx` and show a non-blocking status warning when it
differs; prediction remains allowed because the first classifier uses raw pixel
intensities without spatial-neighborhood features. Always retain the training
and prediction resolution descriptors in provenance so the mismatch remains
explainable. Pyramid resampling may still alter intensity distributions, so the
warning should accompany the existing raw-intensity comparability warning.

### Slice 8: Pooled multi-target training

Add multi-sample behavior only after the single-target model is stable. Follow
the object-classification direction for explicit training scope, reusable model
bundles, deterministic summaries, and compatibility validation.

Deliver:

- coordinate-system target cards based on the feature-extraction widget pattern;
- independent image and scale selection per target;
- shared channel schema resolved by unique channel names;
- shared class schema across all annotation layers;
- deterministic class- and target-balanced sampling;
- one pooled Random Forest;
- one active-target prediction at a time;
- training summary showing contributed pixels by target and class;
- compatibility checks for channel schema and raw-intensity assumptions.

Targets may have different `(y, x)` shapes and different scale-key strings.
Compare their dimensionless relative spacing and report mismatches, but do not
require physical pixel spacing or physical units. Raw-intensity comparability
across batches is not guaranteed; the UI must state that clearly.

Acceptance criteria:

- a one-target pooled request produces the same sampled rows and fitted model as
  the validated single-target path;
- every target contributes through an explicit target descriptor rather than
  hidden widget state;
- channel order and class meaning are identical across targets;
- sampling is deterministic and its per-target, per-class contribution is
  visible;
- one invalid target fails validation clearly and cannot silently disappear
  from training;
- widget and future headless pooled training use the same core function.

### Slice 9: Public headless pixel-classification APIs

Expose supported synchronous APIs after the classifier bundle and pooled core
contracts are stable. Mirror the object-classification headless pattern while
using pixel-specific target descriptors.

Deliver:

- immutable explicit descriptors for training and prediction targets, including
  SpatialData image name, annotation Labels name where applicable, coordinate
  system, selected-resolution descriptor, and ordered channel names;
- `train_pixel_classifier(...)` accepting one or more training targets and
  returning the same versioned classifier bundle used by the widget;
- `apply_pixel_classifier(...)` accepting a loaded bundle and one explicit
  prediction target;
- path-based load/apply convenience wrappers consistent with the existing
  object-classifier headless API;
- the same per-class cap, `balanced_subsample` weighting, compatibility checks,
  block-wise source reading, in-memory `uint8` output, and provenance as the
  widget path;
- explicit optional persistence of the prediction Labels element through the
  same SpatialData output helper used by `Write Labels State`;
- result objects that report target identity, output identity, resolution
  descriptor, class counts, warnings, and whether persistence occurred.

Headless functions are synchronous and must not import napari, Qt, widget
controllers, or `thread_worker`. The caller decides whether to run them in a
process, thread, notebook, or batch system. They must not infer selections from
viewer or app state, and they must never silently choose an image, scale,
channel, annotation element, or coordinate system.

Acceptance criteria:

- a script can train from one target, train from several pooled targets, load a
  bundle, and predict without creating napari or Qt objects;
- widget and headless calls with the same targets and seed produce equivalent
  training samples, fitted-model predictions, and metadata;
- headless apply rejects incompatible channel schemas and reports non-blocking
  relative-spacing warnings exactly like the widget;
- prediction reads the multiplex source block-wise and returns one in-memory
  selected-scale `uint8` result;
- optional persistence writes the same Labels model, transform, and provenance
  as the interactive path;
- importing the public headless module does not import napari, Qt, widget
  modules, or worker machinery.

### Slice 10: Feature enrichment experiments

Do not add feature enrichment until the raw-intensity baseline has real-world
benchmarks.

Candidate experiments, in order of increasing complexity:

1. raw intensity plus inexpensive local mean or Gaussian features;
2. raw intensity plus gradient, Laplacian, or texture features;
3. configurable multiscale shallow features;
4. pretrained deep feature planes;
5. learned dimensionality reduction or deterministic projection.

Every enriched representation should be compared with the raw-intensity
baseline on held-out spatial regions and held-out samples. Only a representation
with a material quality benefit should justify a persistent feature cache.

## State and Gating Rules

The widget should expose a small number of understandable states.

```text
No SpatialData
  -> Choose coordinate system
  -> Choose image
  -> Choose scale
  -> Choose channels
  -> Create/reload annotations
  -> Paint at least two classes
  -> Train
  -> Predict
  -> Write/reload Labels state
```

Rules:

- annotation is enabled as soon as target image, scale, and transform are valid;
- training does not depend on prediction or persisted annotations;
- prediction requires a fresh trained classifier compatible with the target;
- writing Labels state requires a working annotation layer and includes a
  prediction only when a complete prediction exists;
- reloading Labels state requires a persisted annotation element; a persisted
  prediction element is optional;
- changing only viewer contrast, colormap, opacity, or channel-overlay
  visibility does not stale anything;
- changing class colors does not stale the model;
- changing class IDs or semantic class assignment does stale the model;
- changing selected channels or their order stales model and prediction;
- changing scale, image, or coordinate system changes the annotation grid and
  requires an explicit write/discard/cancel decision when Labels state is
  dirty.

## Performance Contract

The simpler model does not remove the need for bounded execution.

- Annotation memory is one byte per selected-scale pixel.
- Training may scan the complete in-memory annotation raster before accessing
  the multiplex source.
- Training memory is bounded by sampled pixels times selected channels.
- Annotation counting and candidate selection do not construct an unbounded
  coordinate array for densely annotated classes.
- Prediction input memory is bounded by one source block times selected
  channels.
- Prediction output is one in-memory `uint8` array and uses one byte per
  selected-scale pixel: 64 MiB at `8192 * 8192`.
- No `C + F` feature raster is stored.
- Source images remain Dask/Zarr-backed where available.
- Long-running work runs outside the Qt main thread.
- Progress is based on training extraction stages or prediction blocks.
- Cancellation leaves annotations untouched.

The `8192 * 8192` recommendation heuristic is a usability default, not a memory
limit. The first implementation does not add memory estimation, confirmation
thresholds, or hard-stop behavior.

## Test Strategy

Use focused tests for each slice, following repository test-scope guidance.

Core tests:

- DataArray and DataTree scale discovery;
- actual scale-key selection;
- channel selection and stable ordering;
- regular coordinate-spacing validation;
- isotropic and anisotropic selected-grid transforms;
- image affine composition;
- pixel-center alignment;
- raw training-row extraction;
- exclusion of unlabeled `0` pixels and inclusion of an explicit Background
  class;
- deterministic per-class capped sampling;
- annotation counting and position sampling without marker-source reads;
- `balanced_subsample` Random Forest weighting for imbalanced sampled counts;
- all-class status-card reporting with annotated, sampled, used,
  non-finite-excluded, and capped fields;
- no false exact-valid-total claim for a capped class whose unsampled marker
  values were not read;
- bounded candidate selection for densely painted classes;
- single-pass non-finite exclusion without replacement sampling;
- non-finite training and prediction handling, including prediction class `0`;
- Random Forest class-ID round trip;
- block prediction equivalence with whole-array prediction on small data;
- cancellation/partial prediction state;
- SpatialData labels parse, write, reload, and alignment;
- metadata compatibility and stale-state decisions;
- selected-resolution descriptor round-trip, including anisotropic spacing and
  affine values;
- same-image exact scale-key validation;
- cross-image scale-key independence and non-blocking relative-spacing mismatch
  warnings;
- one-target pooled-core equivalence with the single-target path;
- deterministic per-target, per-class pooled sampling and contribution
  summaries;
- pooled validation failure when any requested target is invalid;

Widget tests:

- target-control gating;
- scale summary and recommendation heuristic;
- annotation creation and layer reuse;
- class creation, selection, coloring, and counts;
- all-class annotated, sampled, used, non-finite-excluded, and capped training
  fields in the status card;
- non-blocking cross-image relative-spacing mismatch warning;
- dirty Labels-state prompts for reload and target changes;
- worker result revision guards;
- separate annotation and prediction layers;
- prediction layer read-only behavior;
- paired write/reload behavior and overwrite confirmation;
- destroyed widget ignores late callbacks.

Headless tests:

- public-module import without napari, Qt, widgets, or worker machinery;
- single-target and pooled training through explicit target descriptors;
- widget/core/headless equivalence for sampling, prediction, and metadata;
- explicit failure for missing or incompatible target selections;
- block-wise prediction with the same in-memory `uint8` result as the widget;
- optional prediction persistence with the same Labels transform and
  provenance as the interactive path.

At least one integration test should use a real multiscale SpatialData image
whose selected scale has a known non-unit factor and whose image has a nontrivial
2D transformation. The test should write annotations and predictions, reload
the Labels state, and verify world-coordinate alignment.

## Completion Definition

The first usable release is complete when a user can:

1. select a SpatialData image, an actual multiscale level, and marker channels;
2. create an editable annotation layer with the selected level's shape;
3. see that layer correctly aligned over the full-resolution image in napari;
4. define named classes and paint examples;
5. train a deterministic Random Forest from raw annotated marker intensities;
6. predict the complete selected-scale target without loading the full
   multiplex image into memory;
7. review a separate aligned prediction layer;
8. write and reload annotation and prediction as paired, single-scale
   SpatialData labels elements with correct transformations and provenance.

Deep features, shallow contextual features, and persistent feature caches are
explicitly postponed until this baseline is usable and benchmarked.
