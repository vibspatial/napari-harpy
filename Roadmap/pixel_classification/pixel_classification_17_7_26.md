# Pixel Classification Roadmap: Usability-First Raw-Intensity Classifier

Date: 17 July 2026

This document replaces the implementation direction proposed in
`pixel_classification_phase_1.md` for the first usable pixel-classification
release. The earlier document remains useful as research for future feature
enrichment, but it front-loads deep-feature extraction, a large persistent
feature cache, and extensive cache-compatibility machinery before users can
train their first classifier.

The first release should instead optimize for a short, understandable workflow:

1. choose a coordinate system, image, and image scale;
2. create a new pixel-classification workflow or select an eligible existing
   workflow from the explicit Harpy sidecar;
3. restore or select the ordered marker channels;
4. create or reload the workflow's annotation layer at exactly that image
   scale;
5. paint two or more classes;
6. train a Random Forest directly from the annotated raw marker intensities;
7. predict at the same selected scale;
8. review the result in napari;
9. explicitly write or reload the workflow's annotation and prediction Labels
   state.

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
- one active, persistent single-sample workflow identified independently of
  its display name and Labels element names;
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
- an explicit Harpy sidecar workflow manifest binds the target, channels, class
  schema, annotation element, optional prediction element, revisions, and
  provenance;
- annotation and prediction persistence is exposed through explicit workflow
  write and reload actions;
- no intensity normalization in the first implementation;
- no handcrafted features, CNN features, PCA, projection, or feature cache;
- no automatic upsampling to `scale0`.

The first usable milestone supports one active single-sample workflow. Pooled
multi-target training is a follow-up slice after that workflow is reliable,
followed by supported headless training and apply APIs. This is an
implementation order, not a permanent product limitation: first prove the
complete annotation, training, prediction, transform, sidecar discovery, and
persistence contract for one target, then compose the same Qt-free core across
several selected workflows.

The single-target implementation must keep target identity explicit in its core
inputs rather than reading it from global widget state. It does not need to
implement pooled sampling early, but it must avoid an API that can only ever
describe one hard-coded image. This leaves pooled training additive rather than
a rewrite.

## Workflow and Sidecar Model

A pixel-classification workflow is the persistent unit that the user creates,
continues, writes, and reloads. One workflow describes one sample target; it is
not the pooled experiment and is not synonymous with either Labels element.

The target identity of one workflow is:

```text
coordinate system
+ image element
+ selected image scale and resolution descriptor
```

The workflow manifest additionally records:

- a stable `workflow_id`, independent of mutable display and element names;
- an editable workflow display name;
- ordered selected channel names;
- one required annotation Labels element binding;
- one optional prediction Labels element binding;
- the shared class schema;
- annotation, classifier, and prediction revisions and provenance;
- creation and update timestamps and the workflow schema version.

The annotation and prediction arrays remain normal, user-visible SpatialData
Labels elements. The workflow manifest lives in an explicit Harpy sidecar and
owns their association. Element-name conventions help users but never establish
identity or pairing.

For a local backed SpatialData store, use a visible sibling sidecar by default:

```text
sample.zarr
sample.harpy-cache.zarr/
  pixel_classification/
    workflows/
      <workflow_id>/
        manifest.json
    classifier_artifacts/       # reserved for later slices
    feature_caches/             # reserved for later feature enrichment
```

Workflow manifests are durable project metadata even though they live in the
Harpy sidecar. A generic feature-cache cleanup action must never delete them.
Feature arrays remain absent from the first release. If no default writable
sidecar can be derived, workflow persistence requires an explicit writable
sidecar location.

The first release enforces a one-to-one ownership relationship: every workflow
binds exactly one annotation Labels element, and an annotation Labels element
cannot belong to more than one workflow. It does not allow arbitrary annotation
and prediction elements to be combined. The prediction binding is absent until
a prediction destination is created and belongs to the same workflow as its
annotation.

An existing workflow is eligible for a selected card when its manifest and
live elements validate against the source SpatialData association, coordinate
system, image element, and selected-resolution descriptor. Channel selection is
deliberately not an eligibility key: selecting an existing workflow restores
its saved channels, and changing them afterward keeps the annotation valid but
marks its classifier and prediction stale.

The widget discovers eligible workflow manifests after the target grid is
known:

- no eligible workflow: offer `Create new workflow`;
- one eligible workflow: preselect it but wait for an explicit reload action;
- several eligible workflows: require an explicit workflow selection;
- invalid or incomplete manifest: show it as invalid with an actionable reason
  and never guess a replacement from element names.

Normal continuation lists sidecar workflows rather than every Labels element,
because an arbitrary Labels element may be a segmentation mask rather than
pixel-class annotations. A separate `Attach existing annotation Labels` action
validates an unregistered element, collects or confirms its class schema,
optionally attaches a compatible prediction, and creates a new workflow
manifest. This is also the recovery path when valid Labels elements survive but
their sidecar workflow manifest is missing.

Multi-sample training later selects several single-sample workflows. Each
selected coordinate system is represented by a target card similar to the
Feature Extraction widget. Shared channel and class compatibility is validated
across those workflow cards before pooled training; the workflows themselves
remain independently editable and persistable.

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

### 1. Choose the target grid

The user chooses:

- coordinate system;
- image element;
- image scale.

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

### 2. Create or select a workflow

After the target grid is valid, the workflow selector offers
`Create new workflow` plus the eligible manifests discovered from the Harpy
sidecar. A single eligible workflow is preselected but not automatically
reloaded. Several eligible workflows require an explicit choice.

Creating a workflow starts with editable defaults:

```text
Workflow name:     Pixel classification 1
Annotation element: <image_name>_<scale_key>_pixel_annotations
Prediction element: <annotation_name>_prediction
```

Names are normalized through the existing SpatialData element-name validation.
They never overwrite an existing element silently. The prediction name reserves
a destination but does not require a prediction element to exist before a
complete prediction is produced. A manually edited prediction name stops
following later edits to the annotation-name default.

Selecting an existing workflow restores its saved channel selection, class
schema, annotation binding, optional prediction binding, and revision status.
The annotation and prediction choices are constrained by that workflow; the UI
must not let the user combine an annotation from one workflow with a prediction
from another. The user still presses `Reload Labels State` before persisted
arrays replace the working layers.

`Attach existing annotation Labels` is a separate recovery/adoption action, not
an entry in the normal workflow list. It creates a workflow manifest only after
the selected Labels element and optional prediction pass target-grid, transform,
dtype, role, and class-schema validation.

### 3. Inspect selected channels

The user can load selected markers as channel overlays through the existing
viewer-adapter behavior. Loading an overlay does not alter training state.
Changing the actual selected channel set or order makes the classifier and
prediction stale, but it does not invalidate annotations because annotations
are bound to the source image grid, not to a feature schema.

The channel selector should preserve image channel order by default. The UI
should show the order because it is the Random Forest feature-column order.
For a new workflow, the user makes this selection explicitly. For an existing
workflow, the saved selection is restored first and remains editable.

### 4. Create or reload annotations

The user chooses either:

- create the new workflow's annotation layer; or
- `Reload Labels State` for the selected existing workflow.

Creating annotations allocates one zero-filled, in-memory `uint8` array with the
selected `(y, x)` shape and adds it to napari as a single-scale editable Labels
layer. Annotation does not depend on an extracted-feature cache or trained
classifier. The sidecar manifest is used for persistent workflow discovery and
association, not for storing the editable annotation pixels.

Reloading persisted annotations loads their values into an editable working layer.
The implementation must not rely on mutating a backed Dask/Zarr array directly
for every brush operation. The working annotation array is an explicit editable
session copy, and `Write Labels State` persists the accepted state.

Changing the source image or selected scale requires a new compatible
annotation layer. The implementation must not silently resample annotations.

### 5. Define classes and paint

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

### 6. Train

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

### 7. Predict

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

### 8. Write and reload workflow Labels state

Follow the persistence interaction already used by Object Classification, but
treat the selected workflow manifest and its annotation and prediction elements
as one UI-level consistency unit. The widget exposes two explicit actions:

- `Write Labels State` writes the editable annotation element and any present,
  complete prediction element to the active backed SpatialData Zarr store and
  records their bindings and provenance in the selected workflow manifest;
- `Reload Labels State` resolves the selected workflow manifest and replaces
  the in-memory annotation and prediction layers with exactly the SpatialData
  elements referenced by it.

The annotation and prediction remain distinct single-scale SpatialData labels
elements with different element names and roles. The annotation element is the
required part of the pair. A prediction is optional: writing before prediction
exists writes annotations only, and reloading a state with no persisted
prediction restores annotations without creating a prediction layer. If a local
prediction layer exists but no prediction is present in the persisted state,
reload removes the local prediction layer.

The sidecar manifest is the authoritative discovery and pairing record. Do not
use `SpatialData.attrs`, arbitrary Labels `DataArray.attrs`, classifier metadata,
or matching element-name suffixes as the primary workflow registry. Classifier
and prediction provenance may reference `workflow_id`, but those references do
not replace the workflow manifest. When several compatible annotation elements
exist, `Reload Labels State` acts only on the workflow explicitly selected in
the workflow selector.

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

The versioned sidecar workflow manifest should include:

- Harpy pixel-classification schema version;
- stable workflow ID and editable display name;
- source SpatialData path or URI association;
- source image element name;
- the selected-resolution descriptor defined below;
- selected labels shape;
- selected coordinate system;
- ordered channel names;
- selected-grid-to-image transform description;
- class IDs, names, and colors;
- annotation element name, role, and revision;
- optional prediction element name, role, creation state, and annotation
  revision used;
- for predictions, Random Forest parameters and model identity;
- for predictions, training annotation identity and class counts;
- creation and update times;
- napari-harpy, SpatialData, scikit-learn, NumPy, and Dask versions relevant
  to reproduction.

The first implementation does not create a multiscale prediction pyramid and
does not upsample the result to `scale0`. A later explicit export action may
offer nearest-neighbor upsampling or pyramid construction, with provenance that
records the operation.

## Annotation and Prediction Lifecycle

Annotations and predictions are intentionally separate SpatialData objects that
are associated by one explicit sidecar workflow manifest.

```text
source image + selected scale
            |
            +--> workflow manifest
                    |
                    +--> editable annotation layer ----> annotation labels element
                    |
                    +--> trained Random Forest
                               |
                               +--> read-only prediction layer
                                             |
                                             +--> optional prediction labels element
```

### Annotation identity

An annotation is compatible with a target when all of the following match:

- workflow ID and annotation-element binding;
- source image element identity;
- selected-resolution descriptor;
- selected coordinate system binding;
- annotation schema version.

Channel selection is deliberately not part of annotation identity. A user may
paint tissue classes once and compare classifiers trained from different marker
subsets on the same grid.

### Classifier identity

A trained classifier records:

- workflow ID used for the training run;
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

- owning workflow ID and prediction-element binding;
- the classifier identity;
- target source image and selected-resolution descriptor;
- output shape and dtype;
- creation state: running, complete, cancelled, or failed.

Only complete predictions can be written. A cancelled or failed working output
array is discarded and cannot be mistaken for a valid prediction.

## Scope of the First Usable Release

Included:

- one active single-sample workflow and target card;
- explicit create/select/reload workflow interaction;
- one versioned workflow manifest in a visible Harpy sidecar;
- editable default names for annotation and prediction Labels elements;
- sidecar-based discovery of eligible workflows and explicit selection when
  several match;
- `Attach existing annotation Labels` recovery/adoption;
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
- extracted-feature raster caches; the small workflow-manifest sidecar is part
  of the first release;
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
  workflow.py        # workflow identity, manifests, eligibility, and validation
  sidecar.py         # explicit sidecar discovery and manifest IO
  annotations.py     # class schema and annotation validation
  classifier.py      # training-row extraction and Random Forest training
  prediction.py      # tile planning and prediction
  output.py          # SpatialData Labels creation and workflow provenance

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
surface as one single-sample target card whose structure can later be repeated
for several coordinate systems.

Deliver:

- shared `HarpyAppState` binding;
- coordinate-system selector;
- image selector filtered to the coordinate system;
- scale selector populated from the selected image;
- scale summary containing shape, relative spacing, and a recommended marker;
- an in-memory new-workflow draft with a stable workflow ID, editable display
  name, and target binding; persisted workflow discovery is added in Slice 6;
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
- the card structure can later represent one independently selected workflow per
  coordinate system;
- no feature-extraction or feature-cache terminology appears in the UI.

### Slice 3: Editable annotation lifecycle and class editor

Deliver:

- keep the editable annotation owned by the active in-memory workflow draft;
- derive editable default annotation and prediction element names from the
  selected image and scale;
- create a zero-filled, single-scale in-memory `uint8` Labels layer at selected
  shape;
- apply selected-grid and image transforms to the layer;
- mark it as the active editable layer;
- reserve `0` for unlabeled pixels;
- add, rename, recolor, and select class IDs in the range `1..255`;
- show live annotated-pixel counts per class;
- track annotation revision and dirty state;
- protect a dirty in-memory annotation when switching target, switching scale,
  or closing the widget by offering `Discard and continue` or `Cancel`;
- defer all write and reload actions to Slice 6, where persistence is actually
  implemented;
- keep annotation available before training and without any extracted-feature
  cache.

Acceptance criteria:

- painting at selected scale changes only selected-grid pixels;
- the visual brush footprint aligns with the source image;
- erasing restores `0`;
- unlabeled `0` pixels are excluded from training and never treated as implicit
  background;
- an explicitly painted Background class behaves like any other trainable class;
- two classes can be painted and counted;
- cancelling a dirty-state guard leaves the target and annotation layer
  unchanged;
- accepting discard clears the in-memory annotation state and completes the
  requested target change or close;
- Slice 3 does not show a write option or imply that annotations can already be
  persisted;
- changing the draft's display or element names does not change its stable
  workflow ID;
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
  workflow ID and annotation revision in an immutable training result;
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
- bind that prediction to the active workflow and its annotation revision;
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

- resolve `sample.harpy-cache.zarr` as the default sibling for a local backed
  source named `sample.zarr`, with explicit writable sidecar selection when no
  default can be used;
- implement the versioned
  `pixel_classification/workflows/<workflow_id>/manifest.json` contract;
- discover and validate eligible workflows from the sidecar after coordinate
  system, image, and scale selection;
- expose `Create new workflow`, eligible existing workflow choices, and
  `Attach existing annotation Labels`;
- restore saved channels and class schema when an existing workflow is selected;
- `Write Labels State`, which writes the annotation and any present complete
  prediction and updates the workflow manifest as one UI-level persistence
  action;
- `Reload Labels State`, which resolves the selected manifest and reloads its
  bound annotation and optional prediction together;
- the same write/reload/discard/cancel interaction pattern as Object
  Classification;
- upgrade the Slice 3 discard/cancel dirty-state guards to
  write/discard/cancel now that persistence is available;
- validated, distinct annotation and prediction element names with overwrite
  confirmation;
- exactly one annotation binding per workflow, no annotation element owned by
  several workflows, and no arbitrary cross-workflow annotation/prediction
  combinations;
- `Labels2DModel.parse(...)` creation from selected-scale arrays;
- composed transformation from labels intrinsic coordinates to the selected
  coordinate system;
- sidecar workflow provenance for annotation and prediction roles, revisions,
  target identity, channels, classes, and model identity;
- backed `SpatialData.write_element(...)` support;
- dirty tracking for annotation edits, class-metadata edits, and prediction
  creation, replacement, or removal;
- clear behavior for in-memory, read-only, and failed writes;
- reload-and-align verification.

Acceptance criteria:

- no eligible workflow offers creation, one is preselected without automatic
  reload, and several require explicit selection;
- workflow eligibility uses source association, coordinate system, image, and
  selected-resolution descriptor but not channel selection;
- selecting an existing workflow restores its channels; later channel changes
  preserve annotations and stale the classifier and prediction;
- reload resolves only the annotation and prediction named by the selected
  workflow manifest and never guesses a pair from element names;
- arbitrary SpatialData Labels do not appear as normal workflows;
- `Attach existing annotation Labels` creates a workflow only after target,
  transform, dtype, role, and class-schema validation;
- missing, incomplete, or incompatible manifests are reported as invalid with
  actionable reasons;
- workflow discovery and pairing do not depend on `SpatialData.attrs`, custom
  Labels attrs, or classifier metadata;
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
- editable element names do not change the stable workflow identity;
- generic feature-cache cleanup cannot remove workflow manifests;
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
- record the originating workflow ID and annotation revision, or every
  contributing workflow ID and revision for a pooled classifier;
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
- one independently selected eligible single-sample workflow per target card;
- independent image and scale selection per workflow target;
- shared channel schema resolved by unique channel names;
- shared class schema across all annotation layers;
- deterministic class- and target-balanced sampling;
- one pooled Random Forest;
- one active-target prediction at a time;
- training summary showing contributed pixels by target and class;
- pooled classifier provenance listing every contributing workflow ID and
  annotation revision;
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
- each target descriptor resolves from one selected workflow manifest;
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
  workflow ID, SpatialData and sidecar associations, image name, annotation
  Labels name where applicable, coordinate system, selected-resolution
  descriptor, and ordered channel names;
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
  -> Create/select workflow
  -> Choose channels
  -> Create/reload workflow annotations
  -> Paint at least two classes
  -> Train
  -> Predict
  -> Write/reload workflow Labels state
```

Rules:

- workflow creation or selection is enabled as soon as target image, scale, and
  transform are valid;
- annotation creation is enabled for a valid new workflow draft; annotation
  reload is enabled only for a selected valid persisted workflow;
- changing channels does not change workflow eligibility or invalidate its
  annotation binding;
- training does not depend on prediction or persisted annotations;
- prediction requires a fresh trained classifier compatible with the target;
- writing Labels state requires a working annotation layer and includes a
  prediction only when a complete prediction exists; persistent writes also
  require a selected workflow and writable sidecar destination;
- reloading Labels state requires a persisted annotation element; a persisted
  prediction element is optional, and both are resolved through the selected
  workflow manifest;
- changing only viewer contrast, colormap, opacity, or channel-overlay
  visibility does not stale anything;
- changing class colors does not stale the model;
- changing class IDs or semantic class assignment does stale the model;
- changing selected channels or their order stales model and prediction;
- changing workflow selection, scale, image, or coordinate system changes the
  active annotation context and requires an explicit write/discard/cancel
  decision when Labels state is dirty;
- changing only the workflow display name or future element-name destination
  marks workflow metadata dirty but does not stale a compatible classifier.

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

- workflow manifest serialization, schema validation, and stable identity;
- default sidecar and explicit sidecar path resolution;
- workflow eligibility by source, coordinate system, image, and resolution but
  not channel selection;
- zero/one/several eligible-workflow discovery behavior;
- invalid and incomplete workflow-manifest reporting;
- exact annotation/prediction association without name-based guessing;
- adoption and recovery through `Attach existing annotation Labels`;
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

- new versus existing workflow selection;
- existing workflow channel and class-schema restoration;
- several eligible workflows require explicit selection;
- arbitrary unregistered Labels are excluded from normal workflow choices;
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

1. select a coordinate system, SpatialData image, and actual multiscale level;
2. create a new workflow or select an eligible workflow discovered from the
   explicit Harpy sidecar;
3. select or restore ordered marker channels;
4. create or reload the workflow's editable annotation layer with the selected
   level's shape;
5. see that layer correctly aligned over the full-resolution image in napari;
6. define named classes and paint examples;
7. train a deterministic Random Forest from raw annotated marker intensities;
8. predict the complete selected-scale target without loading the full
   multiplex image into memory;
9. review a separate aligned prediction layer;
10. write and reload annotation and prediction as paired, single-scale
    SpatialData Labels elements whose association and provenance are explicit in
    the selected workflow manifest.

Deep features, shallow contextual features, and persistent extracted-feature
raster caches are explicitly postponed until this baseline is usable and
benchmarked.
