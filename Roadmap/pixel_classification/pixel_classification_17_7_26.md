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

- a required `workflow_mode` discriminator;
- a stable `workflow_id`, independent of mutable display and element names;
- an editable workflow display name;
- ordered selected channel names;
- the shared class schema;
- mode-applicable annotation, classifier, and prediction revisions and
  provenance;
- creation and update timestamps and the workflow schema version.

The manifest is a tagged union with two explicit modes:

| Mode | Annotation binding | Classifier binding | Prediction binding |
|---|---|---|---|
| `annotation` | Required | Optional until trained or imported | Optional |
| `prediction_only` | Absent | Required classifier-bundle reference | Required for a persisted workflow |

Slices 1–6 implement only `workflow_mode: annotation`. They must nevertheless
write and validate that discriminator so Slice 7 can add `prediction_only`
without reinterpreting old manifests. An in-memory `prediction_only` draft may
exist before prediction completes, but its manifest is not published as a valid
persisted workflow until a complete, fresh prediction has been written and its
classifier-bundle reference has been validated successfully.

An `annotation` workflow owns editable examples and can train or import a
classifier. A `prediction_only` workflow represents application of an existing
classifier to one target and never creates, requires, or pretends to own an
annotation Labels element. Its class-ID membership and channel requirements
come from the classifier bundle. Class names and colors may remain editable
display metadata, but class IDs cannot be added, removed, or reassigned in this
mode.

The annotation and prediction arrays remain normal, user-visible SpatialData
Labels elements. The workflow manifest lives in an explicit Harpy sidecar and
owns their association. Element-name conventions help users but never establish
identity or pairing.

The first release supports only a SpatialData source opened from a
path-addressable Zarr store on the local filesystem. Reject an unbacked or
purely in-memory SpatialData object and any remote, URI-backed, object-store, or
network-filesystem abstraction that does not resolve to a supported local Zarr
path. The widget must explain this prerequisite in the status card before
enabling workflow creation, annotation, training, or prediction. This source
restriction does not conflict with using editable in-memory NumPy working arrays
for annotation and prediction.

For a supported local backed SpatialData store, use a visible sibling sidecar
by default:

```text
sample.zarr
sample.harpy-cache.zarr/
  pixel_classification/
    workflows/
      <workflow_id>/
        manifest.json
    feature_caches/             # reserved for later feature enrichment
```

Workflow manifests are durable project metadata even though they live in the
Harpy sidecar. A generic feature-cache cleanup action must never delete them.
Feature arrays remain absent from the first release. If the default sibling
sidecar cannot be used, the user may choose another local writable sidecar
location.

### Canonical classifier bundles and workflow references

Slice 7 uses one versioned, user-owned classifier-bundle format. A saved bundle
is the canonical classifier file; applying it to a target never copies it into
the workflow sidecar and does not create any managed classifier-model
directory.

The bundle contains a stable `classifier_id`, fitted estimator, exact ordered
channel schema, class-schema snapshot, training resolution descriptor, Random
Forest parameters, source workflow and annotation revisions, relevant library
versions, and creation metadata. Bundle loading uses trusted joblib/pickle
semantics and the UI and documentation must warn users to load only bundles
from trusted sources.

A newly trained or retrained classifier receives a new `classifier_id`; saved
bundles are treated as immutable model revisions. A current-session classifier
may be used for an in-memory prediction immediately, but before that result can
be persisted as a `prediction_only` workflow the user must explicitly save the
bundle once to a chosen local path. Napari-harpy must not silently choose a
classifier-file location.

The destination workflow manifest stores the bundle's `classifier_id`, bundle
schema version, and normalized bundle path relative to the sidecar root. The
same canonical file may therefore be referenced by several workflows and by
sidecars belonging to different SpatialData stores without creating duplicate
model files. On reload or re-prediction, resolve the recorded path from the
sidecar root and validate at least the bundle schema, `classifier_id`, channel
schema, and class-ID schema. Never guess another file with a similar name or ID.

If the bundle later cannot be found or its identity no longer matches, a
persisted prediction can still be reloaded and displayed from its Labels
element. Mark the workflow degraded, explain the missing or mismatched path in
the status card, and disable re-prediction until the reference is repaired.
The manifest retains a lightweight classifier and prediction-provenance summary
so an existing prediction remains interpretable without loading the estimator.
An explicit `Locate classifier` recovery action or self-contained packaging
operation may be added later; both are outside the first implementation.

Classifier-reference validation, prediction Labels writing, and manifest
publication are not one filesystem transaction. Prevalidate the classifier
reference and complete request, write the complete fresh prediction, and
publish the workflow manifest last. A failed later stage keeps the in-memory
workflow recoverable and does not trigger cross-element rollback. Because the
classifier file is user-owned and read-only to this operation, workflow writes
never create, overwrite, delete, or clean up classifier bundles.

### Source association and relocation

Every workflow manifest stores the source Zarr path relative to the sidecar
root, never relative to the process working directory. For the default sibling
layout, the association is:

```yaml
source_zarr_relative_path: ../sample.zarr
```

An explicitly selected local sidecar uses the equivalent normalized relative
path, which may contain additional parent components. Do not store an absolute
source path or remote URI as a fallback in the first release.

On discovery and reload, resolve this path from the sidecar root and require it
to identify the active local SpatialData Zarr store. Then validate the recorded
source image, coordinate system, selected-resolution shape, scale descriptor,
and transform against the live store. Moving the dataset and sidecar together
while preserving their relative layout continues to work. Moving or renaming
only one of them makes the association invalid.

Fail closed and explain the mismatch in the status card. Report the sidecar
path, stored relative reference, resolved source path, and the first missing or
incompatible target descriptor. Never guess by basename, scan nearby
directories, silently bind another dataset, or provide a `Relink workflow
sidecar` action in the first release. Recovery consists of restoring the
expected relative layout or creating or attaching a new workflow explicitly.

This deliberately does not detect a different dataset placed at the same
relative path with an identical target structure. Persistent dataset UUIDs,
content fingerprints, remote source associations, and relocation tooling are
postponed until there is evidence they are needed.

For `annotation` workflows, the first release enforces a one-to-one ownership
relationship: every such workflow binds exactly one annotation Labels element,
and an annotation Labels element cannot belong to more than one workflow. It
does not allow arbitrary annotation and prediction elements to be combined.
The prediction binding is absent until a prediction destination is created and
belongs to the same workflow as its annotation.

Annotation and prediction element names are draft-time choices until the
corresponding element is written successfully for the first time. That first
successful write fixes the element binding in the workflow manifest. A fixed
binding is read-only in the first release: renaming, cloning, or redirecting a
persisted workflow element requires a later explicit action and is out of
scope. The workflow display name remains editable because it does not identify
or relocate a SpatialData element. If annotations are written before any
prediction exists, only the annotation binding becomes fixed; the prediction
destination remains editable until its own first successful write.

A `prediction_only` workflow has no annotation name or annotation-name control.
Its prediction name follows the same draft-until-first-successful-write rule.
Its classifier binding is fixed when a manifest referencing the validated
bundle is first published successfully. Applying a different classifier creates
a new prediction-only workflow rather than silently retargeting a persisted
one.

An existing workflow is eligible for a selected card when its manifest and
live elements validate against the source SpatialData association, coordinate
system, image element, selected-resolution descriptor, and requested workflow
mode. For annotation workflows, channel selection is deliberately not an
eligibility key: selecting an existing workflow restores its saved channels,
and changing them afterward keeps the annotation valid but marks its classifier
and prediction stale. A prediction-only workflow instead restores the required
channel schema from its bound classifier bundle. If that file is unavailable,
the manifest summary can restore display metadata and the persisted prediction,
but re-prediction remains disabled.

The widget discovers eligible workflow manifests after the target grid is
known:

- no eligible workflow: offer `Create new workflow`;
- one eligible workflow: preselect it but wait for an explicit reload action;
- several eligible workflows: require an explicit workflow selection;
- invalid or incomplete manifest: show it as invalid with an actionable reason
  and never guess a replacement from element names.

Normal annotation continuation lists sidecar workflows rather than every Labels element,
because an arbitrary Labels element may be a segmentation mask rather than
pixel-class annotations. A separate `Attach existing annotation Labels` action
validates an unregistered element, collects or confirms its class schema,
optionally attaches a compatible prediction, and creates a new workflow
manifest. This is also the recovery path when valid Labels elements survive but
their sidecar workflow manifest is missing.

Multi-sample training later selects several `annotation` workflows. Each
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
following later edits to the annotation-name default. Each name remains editable
only until that element's first successful write. Existing workflow bindings
are displayed read-only; changing them requires a future explicit rename or
clone action rather than editing the workflow form.

Selecting an existing workflow restores its saved channel selection, class
schema, annotation binding, optional prediction binding, and revision status.
The annotation and prediction choices are constrained by that workflow; the UI
must not let the user combine an annotation from one workflow with a prediction
from another. The user still presses `Reload Labels State` before persisted
arrays replace the working layers.

`Attach existing annotation Labels` is a separate recovery/adoption action, not
an entry in the normal workflow list. It creates a workflow manifest only after
the selected Labels element and optional prediction pass target-grid, transform,
dtype, role, and class-schema validation. Because attached elements already
exist persistently, their bindings are fixed as soon as attachment succeeds.

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

#### Class identity and schema edits

A nonzero class ID is the stable identity of a class while that class is present
in the workflow's current schema. The annotation and prediction arrays encode
only these IDs. Names and colors are editable presentation metadata; they do not
change the integer meaning learned or emitted by the classifier.

The following membership-edit rules apply to `annotation` workflows.
Prediction-only workflows allow rename and recolor display edits but disable
class addition, removal, ID editing, and ID reuse because their output IDs are
fixed by the classifier bundle.

For annotation workflows, apply these rules:

| Action | Allowed behavior | Persistence state | Classifier and prediction state |
|---|---|---|---|
| Rename a class | Allow; keep its ID and pixels unchanged | Class schema and manifest dirty | Remain fresh |
| Change a class color | Allow and update annotation and prediction layer colormaps | Class schema and manifest dirty | Remain fresh |
| Remove a class with zero annotated pixels | Remove it from the current class schema | Class schema and manifest dirty | Become stale |
| Remove a class with annotated pixels | Reject and show the exact annotated-pixel count | Unchanged | Unchanged |
| Change an existing class ID | Reject | Unchanged | Unchanged |
| Reuse an ID that is no longer in the current schema | Allow for a newly created class | Class schema and manifest dirty | Existing artifacts remain stale and retain their own schema snapshots |

Renaming is a display-name correction, not a semantic reassignment. Retraining
after a rename would learn the same mapping from the same integer-labelled
pixels. If the user intends a genuinely different biological meaning, they must
create a new class ID and repaint or explicitly reassign annotations; bulk
class reassignment is a possible later feature.

Removing a class must never silently erase its annotated pixels, convert them
to `0`, or reinterpret them as another class. Once its annotated-pixel count is
zero, removal deletes that definition from the workflow's current class schema.
The manifest does not maintain a retired-class registry or retain unused IDs,
old names, or old colors in its current class list. Adding or removing a class
changes active class-ID membership and therefore stales any existing classifier
and prediction.

A class ID is immutable while its class definition exists, but an ID that is no
longer present in the current schema may be assigned to a newly created class.
The provenance for each existing classifier and prediction stores the class
schema snapshot used to create that artifact. Therefore an older retained stale
prediction continues to interpret and render its integer IDs using its own
snapshot, not a later workflow class that happens to reuse the same ID. Once
such an artifact is removed, its historical names and colors need not be
retained elsewhere.

The workflow manifest is the authoritative source for the **current** class
schema. Its class list contains only the IDs, names, and colors currently shown
in the class editor. `Reload Labels State` reconstructs the annotation editor
and editable annotation layer's napari color mapping from that list.
`Reload Prediction State` reconstructs the prediction-only display schema and
prediction colormap without creating annotation controls. A fresh prediction
uses the applicable current mapping. A retained stale prediction instead uses
its artifact-specific schema snapshot when the old and current mappings differ.
Do not store the current class schema in `SpatialData.attrs` or Labels
`DataArray.attrs`; the Labels arrays contain only integer IDs.

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

#### Prediction replacement and cancellation

Re-prediction must use a private, unpublished `uint8` NumPy output array. Do not
clear, resize, or fill the array currently displayed by the prediction Labels
layer while the worker is running.

Apply this lifecycle:

1. retain the existing prediction layer, its array, and its provenance;
2. present that existing prediction as stale with a `recalculating` status while
   the worker fills the private candidate array block by block;
3. publish only a complete candidate whose captured workflow, annotation,
   classifier, channel, and target revisions still match the active workflow;
4. after that validation succeeds, replace the existing layer's data in one UI
   update, or create the prediction layer if none existed, and mark the new
   prediction complete, fresh, and dirty;
5. on cancellation, worker failure, or obsolete-result validation failure,
   discard the private candidate array and leave the previous layer data and
   provenance unchanged.

Cancellation or failure must never replace the existing layer with a partial
array and must never create an apparently complete first prediction. After the
worker stops, recompute the retained prediction's freshness from its preserved
provenance and the current workflow revisions: a prediction that was already
stale remains stale, while one whose dependencies are still current may return
from the transient `recalculating` presentation to fresh. If no prior prediction
existed, cancellation or failure leaves no prediction layer.

This intentionally permits at most two selected-scale prediction arrays during
re-prediction: the displayed previous NumPy array and the private candidate
array. At the recommended maximum shape, each is 64 MiB. The first
implementation accepts this temporary memory use in exchange for
non-destructive cancellation and simple layer semantics.

### 8. Write and reload workflow Labels state

Follow the persistence interaction already used by Object Classification, but
treat the selected workflow manifest and its annotation and prediction elements
as one UI-level consistency unit. For `workflow_mode: annotation`, the widget
exposes two explicit actions:

- `Write Labels State` writes the editable annotation element and writes or
  overwrites prediction data only when the local prediction is complete and
  fresh, then records the resulting bindings, freshness, and provenance in the
  selected workflow manifest;
- `Reload Labels State` resolves the selected workflow manifest and replaces
  the in-memory annotation and prediction layers with exactly the SpatialData
  elements referenced by it.

For `workflow_mode: prediction_only`, Slice 7 exposes `Write Prediction State`
and `Reload Prediction State`. They use the same dirty guards, manifest-last
publication, and failure reporting, but operate on the classifier-bundle
reference and prediction only. No annotation layer participates.

Within an annotation workflow, annotation and prediction remain distinct
single-scale SpatialData labels elements with different element names and
roles. The annotation element is required and prediction is optional: writing
before prediction exists writes annotations only, and reloading a state with no
persisted prediction restores annotations without creating a prediction layer.
If a local prediction layer exists but no prediction is present in the
persisted state, reload removes the local prediction layer.

The sidecar manifest is the authoritative discovery and pairing record. Do not
use `SpatialData.attrs`, arbitrary Labels `DataArray.attrs`, classifier metadata,
or matching element-name suffixes as the primary workflow registry. Classifier
and prediction provenance may reference `workflow_id`, but those references do
not replace the workflow manifest. When several compatible annotation elements
exist, `Reload Labels State` acts only on the workflow explicitly selected in
the workflow selector.

Brush strokes and prediction generation update only the in-memory working
layers; they never write through to Zarr automatically. As with the table-based
classification workflow, the controller compares this working state with the
last successfully persisted state. It must not infer persistence state from the
presence of a napari layer alone.

#### Dirty state versus stale state

`Dirty` and `stale` describe different properties and must be tracked
independently:

- **dirty** means that a persistable part of the in-memory workflow differs from
  the last successfully persisted workflow state;
- **stale** means that a derived classifier or prediction no longer represents
  the workflow's current annotation revision where applicable, classifier
  identity, channels, class schema, target descriptor, or other declared
  inputs. Staleness does not imply that the prediction pixel array was edited.

The workflow controller tracks at least:

- `annotation_dirty`: annotation pixel values changed;
- `class_schema_dirty`: current class membership, class names, or colors
  changed;
- `prediction_dirty`: a complete, fresh prediction was created or replaced
  relative to the persisted state;
- `manifest_dirty`: workflow bindings, revisions, provenance, names, or
  freshness metadata changed;
- aggregate `labels_state_dirty`: at least one persistable component above is
  dirty. This aggregate state drives write/discard/cancel prompts.

An annotation workflow may use all of these flags. A prediction-only workflow
has no `annotation_dirty`; its class-ID membership is fixed by its classifier,
and its aggregate state is derived from prediction, permitted display-schema,
and manifest changes.

These are controller-level workflow flags. Napari layer events may trigger
them, but the Labels arrays themselves are not the authoritative state machine.
In particular, merely marking an otherwise unchanged prediction stale does not
make its pixel array dirty. It normally makes the manifest dirty so that the
changed dependency relationship can be recorded.

| Event | Annotation working state | Prediction working state | Workflow persistence state |
|---|---|---|---|
| User paints annotations | Dirty | Becomes stale; its pixel array does not become dirty merely because the annotations changed | Dirty |
| User runs `Predict` | Unchanged | Complete, fresh, and dirty until written | Dirty |
| User changes channels | Unchanged | Becomes stale; its pixel array is unchanged | Manifest dirty |
| User renames or recolors a class | Unchanged | Remains fresh | Class schema and manifest dirty |
| User removes an unused class | Unchanged | Becomes stale; its pixel array is unchanged | Class schema and manifest dirty |
| Successful write with a fresh prediction | Clean | Clean and fresh | Clean |
| Successful annotation-only write retaining an older persisted prediction | Clean | Persisted array remains unchanged and stale | Clean after the manifest records the stale relationship |

A prediction that was generated locally but became stale before its first write
is an **ephemeral stale prediction**, not a persistable dirty prediction. It may
remain visible for comparison, but the status card must warn that it will not
be saved and will disappear on reload. After a successful annotation and
manifest write, that ephemeral layer does not keep `labels_state_dirty` set;
otherwise the widget would repeatedly offer to save a prediction that the
persistence policy deliberately excludes.

A write clears only the dirty flags covered by the successfully finalized
workflow write. A failed write keeps the in-memory layers and their dirty flags
intact. Freshness remains orthogonal: a clean persisted prediction may still be
stale, and a newly computed fresh prediction is dirty until it is successfully
persisted.

Workflow writes are deliberately best-effort rather than a cross-store
transaction. For `annotation`, use this order:

1. prevalidate the complete request, including sidecar destination, fixed or
   draft element bindings, overwrite decisions, array dtype and shape,
   transformations, and manifest serialization;
2. write the annotation Labels element;
3. write the prediction Labels element only when the local prediction is both
   complete and fresh;
4. stage, validate, and finalize the workflow manifest last as the
   workflow-level completion record; an incomplete temporary manifest is never
   eligible for discovery.

Use `harpy.im.add_labels(...)` as the Labels-element write boundary, including
its explicit overwrite support. Do not duplicate Harpy's element creation and
cleanup machinery in napari-harpy. Before relying on overwrite for this
workflow, verify and, if necessary, extend Harpy's single-element replacement
so a failed replacement cleans up temporary state safely and preserves or
restores the previous canonical element when possible.

Napari-harpy does not promise atomic rollback across annotation, prediction,
and sidecar writes. If any stage fails, stop the remaining stages, keep the
in-memory workflow and layers unchanged and dirty, and show the failing stage,
element or manifest path, and underlying error in the status card. The message
must state that an earlier disk stage may already have succeeded and that
retrying `Write Labels State` is the normal recovery action. Do not silently
delete or restore already written user-facing Labels elements from a later
napari-harpy stage.

Prediction-only writes use the classifier-reference validation, prediction,
then manifest-last order defined in the classifier-storage and Slice 7
contracts. Their equivalent recovery action is `Write Prediction State`.

Because the manifest is written last, a failed workflow write never publishes a
new manifest revision as complete. A later discovery pass reports missing,
incomplete, or incompatible manifests and referenced elements, but does not
attempt automatic data repair. Unreferenced temporary artifacts produced by a
failed Harpy element write are Harpy cleanup concerns and are never interpreted
as valid pixel-classification workflows.

`Reload Labels State` reloads immediately when the current state is clean. When
it is dirty, it uses the Object Classification three-way decision:

1. `Write labels state and reload`;
2. `Reload labels state and discard local edits`;
3. `Cancel`.

`Reload Prediction State` uses the equivalent prediction-only choices: write
prediction state and reload, reload and discard local changes, or cancel.

Leaving the current image, scale, coordinate system, or other target-defining
selection while dirty uses the equivalent choices: write and continue, discard
and continue, or cancel the target change. Reload and discard applies to both
the annotation and prediction layers; it never retains one local member of the
pair while reloading the other.

Only complete, fresh prediction arrays are written or overwritten. For
`annotation`, apply this policy:

| Prediction state | `Write Labels State` behavior |
|---|---|
| Complete and fresh | Write or overwrite annotation and prediction, then update the manifest. |
| Stale, with a previously persisted prediction | Write annotation only; leave the persisted prediction unchanged and record it as stale with its original provenance. |
| Stale, never previously persisted | Write annotation only; do not persist or bind the stale prediction. |
| Missing | Write annotation only. |

The first persisted write of a `prediction_only` workflow requires a complete,
fresh prediction and a validated canonical classifier-bundle reference. For an
already persisted prediction-only workflow, a stale persisted prediction may
remain unchanged while permitted display metadata and the manifest's stale
status are written. A stale prediction that has never been persisted cannot
create a prediction-only workflow.

A retained stale prediction remains associated with its workflow rather than
becoming an orphan, but its manifest entry continues to record the classifier
ID and annotation revision that produced it. The updated workflow revision marks
it stale relative to the current annotation, channels, class schema, or
classifier. Reload may display it for comparison but must not present it as
current. The status card explains that the prediction was not rewritten, names
its originating annotation revision when available, and directs the user to run
`Predict` to refresh it.

An unpersisted stale prediction may remain visible during the current session,
but it has no persisted workflow binding and disappears on reload. Writing must
never delete or unlink an older persisted prediction merely because it became
stale. No periodic recovery snapshot is part of the first implementation.

The versioned sidecar workflow manifest should include:

- Harpy pixel-classification schema version;
- `workflow_mode: annotation | prediction_only`;
- stable workflow ID and editable display name;
- normalized source Zarr path relative to the sidecar root;
- source image element name;
- the selected-resolution descriptor defined below;
- selected labels shape;
- selected coordinate system;
- ordered channel names;
- selected-grid-to-image transform description;
- current class IDs, names, and colors; this is the authoritative display schema
  used to reconstruct the mode-appropriate class editor and Labels colormap;
- for `annotation`, required annotation element name, role, and revision;
- for `annotation`, optional prediction element name, role, creation state, and
  annotation revision used;
- for `prediction_only`, no annotation fields and a required prediction element
  name, role, creation state, and prediction revision;
- for `prediction_only`, required classifier ID, bundle schema version, and
  classifier-bundle path relative to the sidecar root;
- for every prediction, classifier identity, training annotation provenance,
  training class counts, target resolution descriptor, and application time;
- creation and update times;
- napari-harpy, SpatialData, scikit-learn, NumPy, and Dask versions relevant
  to reproduction.

A representative persisted prediction-only manifest is:

```yaml
schema_version: 1
workflow_mode: prediction_only
workflow_id: <workflow-uuid>
display_name: Target A - imported pixel classifier
source_zarr_relative_path: ../sample.zarr

target:
  coordinate_system: target_coordinate_system
  image_element: target_image
  resolution:
    scale_key: scale3
    shape_yx: [5000, 5000]
    scale0_shape_yx: [40000, 40000]
    relative_spacing_to_scale0_yx: [8.0, 8.0]
    selected_grid_to_scale0_affine: [...]
  selected_grid_to_image_transform: [...]

channels:
  ordered_names: [DAPI, CD3, CD8]

classes:
  - {id: 1, name: Background, color: "#4A4A4A"}
  - {id: 2, name: Tumor, color: "#E64B35"}

classifier:
  classifier_id: <classifier-uuid>
  bundle_schema_version: 1
  bundle_path_relative_to_sidecar: ../models/tissue_classifier.harpy-classifier.joblib

prediction:
  element_name: target_image_pixel_prediction_scale3
  role: prediction
  revision: 1
  state: complete
  freshness: fresh
  classifier_id: <classifier-uuid>
  target_resolution: {...}
  applied_at: ...

created_at: ...
updated_at: ...
```

The manifest contains no annotation element, annotation revision, or annotation
dirty state. Training workflow IDs, training annotation revisions, training
class counts, estimator parameters, and library versions live in the referenced
classifier bundle and may be summarized in prediction provenance without
duplicating the estimator payload in the manifest.

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
            +--> annotation workflow manifest
                    |
                    +--> editable annotation layer ----> annotation labels element
                    |
                    +--> trained Random Forest
                               |
                               +--> read-only prediction layer
                                             |
                                             +--> optional prediction labels element
```

Slice 7 adds a separate application path without inventing annotations:

```text
portable or current-session classifier
            |
            +--> explicitly saved canonical classifier bundle
                         |
target image + selected scale
            |
            +--> prediction-only workflow manifest
                         |   (relative bundle path + classifier ID)
                         |
                         +--> read-only prediction layer
                                      |
                                      +--> required prediction labels element
```

The Qt-free core operation is annotation-independent:

```text
classifier bundle + explicit prediction target -> prediction result
```

An unpersisted interactive or headless result may remain only in memory. Once a
prediction result is persisted through the supported workflow API, it receives
a `prediction_only` manifest; the implementation must not write an
undiscoverable standalone Labels element as the normal persisted path.

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
- class-schema snapshot used for that training run;
- sampled class counts;
- Random Forest parameters and fitted estimator;
- library versions;
- annotation revision used for training.

Changing annotations, active class-ID membership, channel selection, channel
order, image, or scale marks the classifier stale. Changing only a class name or
color does not; those metadata edits leave the fitted integer-ID mapping intact.

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

- owning workflow ID, workflow mode, and prediction-element binding;
- classifier ID and training provenance;
- source annotation revision for an annotation workflow, when applicable;
- class-schema snapshot used for rendering and provenance;
- target source image and selected-resolution descriptor;
- output shape and dtype;
- application timestamp;
- creation state: running, complete, cancelled, or failed.

The `running`, `cancelled`, and `failed` states describe the private candidate
operation; they must not be applied to or overwrite the previously displayed
complete prediction. The displayed layer changes identity and provenance only
when a complete candidate passes the active-revision guard.

Only complete, fresh predictions can be written. A cancelled or failed working
output array is discarded and cannot be mistaken for a valid prediction. A
complete stale prediction can remain visible under the persistence policy above
but is never written or overwritten.

## Scope of the First Usable Release

Included:

- local filesystem-backed SpatialData Zarr sources only;
- `workflow_mode: annotation`; manifests carry the discriminator even though
  the first milestone implements no other mode;
- one active single-sample workflow and target card;
- explicit create/select/reload workflow interaction;
- one versioned workflow manifest in a visible Harpy sidecar;
- editable draft names for annotation and prediction Labels elements, fixed
  independently by each element's first successful write;
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

- unbacked or purely in-memory SpatialData sources;
- remote, URI-backed, or object-store SpatialData sources;
- classifier reuse and `prediction_only` workflows until Slice 7;
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
  classifier_bundle.py  # versioned bundle serialization and validation
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

- resolve and validate the local filesystem Zarr backing path for the source
  SpatialData object;
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
- tests cover pixel-center behavior and fail on a deliberate half-pixel shift;
- unbacked, in-memory, and remote or URI-backed SpatialData sources are rejected
  with an actionable unsupported-source result before image-scale work begins.

### Slice 2: Widget shell and target selection

Register `PixelClassificationWidget` and implement the first real selection
surface as one single-sample target card whose structure can later be repeated
for several coordinate systems.

Deliver:

- shared `HarpyAppState` binding;
- source-backing prerequisite status that disables the pixel-classification
  workflow for unbacked, in-memory, or remote SpatialData;
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

- a supported local Zarr-backed SpatialData source enables target selection;
- unsupported source backing disables workflow creation, annotation, training,
  and prediction and explains that a local on-disk Zarr source is required;
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
- add, rename, recolor, remove, and select classes with stable IDs in the range
  `1..255`;
- show live annotated-pixel counts per class;
- keep an existing class ID immutable while allowing an ID absent from the
  current schema to be used for a newly created class;
- reject removal while annotated pixels remain and report their exact count;
- track annotation revision, `annotation_dirty`, and `class_schema_dirty`
  independently;
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
- renaming or recoloring a class changes persisted schema metadata without
  staling an otherwise compatible classifier or prediction;
- removing a zero-count class removes it from the current schema and stales
  existing derived artifacts;
- attempting to remove a class with annotated pixels is rejected without
  changing the class schema or annotation array;
- an existing class ID cannot be edited, while an ID absent from the current
  schema can be assigned to a newly created class;
- current class names and colors reload from the workflow manifest rather than
  SpatialData or Labels attrs;
- a retained stale prediction with an older or reused ID renders from its own
  class-schema snapshot;
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
- allocate one private in-memory selected-scale `uint8` candidate array without
  mutating any currently displayed prediction;
- read selected channels block-wise;
- finite-mask every `pixels x channels` block and predict only finite rows;
- leave rows containing any `NaN`, `+inf`, or `-inf` as class ID `0` and report
  their count;
- fill the private candidate output with predicted class IDs;
- show progress and permit cancellation between blocks;
- retain the previous prediction layer while prediction runs and present it as
  stale with a `recalculating` status;
- only after successful completion and revision validation, create or update
  one read-only prediction Labels layer with the same transform as annotations;
- discard a cancelled, failed, or obsolete candidate and retain the previous
  layer data and provenance;
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
- re-prediction never mutates the displayed prediction array in place;
- successful re-prediction replaces the existing layer data only after the
  private candidate is complete and current, without adding a duplicate layer;
- cancellation, failure, or obsolete-result rejection cannot produce a
  persistable apparently complete result and leaves the previous prediction
  array and provenance unchanged;
- cancellation or failure without a previous prediction leaves no prediction
  layer;
- after cancellation or failure, the retained prediction returns to the
  freshness implied by its preserved provenance and current workflow revisions;
- re-prediction uses at most the previous and private candidate full-size
  prediction arrays.

### Slice 6: SpatialData annotation and prediction persistence

This slice completes the first end-to-end usable milestone.

Deliver:

- resolve `sample.harpy-cache.zarr` as the default sibling for a local backed
  source named `sample.zarr`, with explicit writable sidecar selection when no
  default can be used;
- store a normalized source Zarr path relative to the sidecar root;
- resolve the stored reference from the sidecar root and validate it against the
  active source and target descriptors;
- fail closed with an actionable status-card error when the source reference is
  missing or mismatched, without basename guessing, directory scanning, silent
  relinking, or a first-release relink action;
- implement the versioned
  `pixel_classification/workflows/<workflow_id>/manifest.json` contract;
- write and validate `workflow_mode: annotation` in every Slice 6 manifest and
  reject unsupported modes explicitly rather than treating their missing
  annotation fields as corrupt annotation workflows;
- discover and validate eligible workflows from the sidecar after coordinate
  system, image, and scale selection;
- expose `Create new workflow`, eligible existing workflow choices, and
  `Attach existing annotation Labels`;
- restore saved channels and class schema when an existing workflow is selected;
- `Write Labels State`, which writes the annotation and any present complete
  fresh prediction and updates the workflow manifest as one UI-level
  persistence action;
- prevalidate the complete write, call `harpy.im.add_labels(...)` for annotation
  and prediction in that order, and publish the workflow manifest last;
- `Reload Labels State`, which resolves the selected manifest and reloads its
  bound annotation and optional prediction together;
- the same write/reload/discard/cancel interaction pattern as Object
  Classification;
- upgrade the Slice 3 discard/cancel dirty-state guards to
  write/discard/cancel now that persistence is available;
- validated, distinct annotation and prediction element names with overwrite
  confirmation;
- freeze each annotation or prediction binding independently after that
  element's first successful write while keeping the workflow display name
  editable;
- exactly one annotation binding per Slice 6 annotation workflow, no annotation
  element owned by several workflows, and no arbitrary cross-workflow
  annotation/prediction combinations;
- prepare selected-scale arrays, `("y", "x")` dims, chunks, and composed
  transformations for `harpy.im.add_labels(...)`, which creates the
  `Labels2DModel` element;
- composed transformation from labels intrinsic coordinates to the selected
  coordinate system;
- sidecar workflow provenance for annotation and prediction roles, revisions,
  target identity, channels, classes, and model identity;
- backed `SpatialData.write_element(...)` support;
- controller-level `annotation_dirty`, `class_schema_dirty`,
  `prediction_dirty`, `manifest_dirty`, and aggregate `labels_state_dirty`
  tracking, kept separate from classifier and prediction freshness;
- persistence handling for fresh, previously persisted stale, unpersisted stale,
  and missing prediction states;
- clear behavior for editable working layers, read-only prediction layers, and
  failed writes;
- stage-specific status-card errors that keep the workflow dirty and direct the
  user to retry without claiming cross-element rollback;
- reload-and-align verification.

Acceptance criteria:

- every Slice 6 workflow round-trips with `workflow_mode: annotation`, while an
  unsupported mode receives a clear version-or-mode error;
- adjacent dataset and sidecar stores use `../sample.zarr` as the source
  association and remain valid when moved together;
- moving or renaming only the dataset or sidecar invalidates the association
  with a status-card error containing the sidecar path, stored reference,
  resolved source path, and first failed validation;
- source references resolve relative to the sidecar root and never to the
  process working directory;
- no absolute-path or URI fallback, nearby-directory search, basename guessing,
  silent relinking, or `Relink workflow sidecar` action is present;
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
- a complete fresh prediction is written and recorded as current;
- a stale previously persisted prediction is not rewritten or deleted, remains
  bound with its original provenance, and is recorded as stale;
- a stale never-persisted prediction is not written and receives no persisted
  binding;
- after a successful annotation and manifest write, a visible ephemeral stale
  prediction does not keep `labels_state_dirty` set and the status card states
  that reload will discard it;
- a missing prediction writes annotations only;
- the status card explains every skipped stale prediction and offers `Predict`
  as the refresh action;
- reloading a state without a persisted prediction removes any local prediction
  and does not create a replacement prediction layer;
- dirty reload offers write-and-reload, reload-and-discard, and cancel;
- leaving a dirty target offers write-and-continue, discard-and-continue, and
  cancel;
- cancelling either prompt leaves the target and both layers unchanged;
- dirty state clears only when all required Labels writes and the final manifest
  write succeed;
- annotation-write, prediction-write, and manifest-write failures each stop the
  remaining stages, preserve the in-memory workflow, and identify the failing
  stage in the status card;
- failure after an earlier successful disk stage explicitly warns that disk may
  be partially updated and leaves retry as the recovery action;
- napari-harpy performs no cross-element rollback or automatic repair;
- Harpy single-element overwrite cleanup is validated independently, including
  failure after staging and during canonical replacement;
- no upsampling occurs during write;
- annotation and prediction names cannot collide silently;
- draft element names do not change the stable workflow identity;
- a fixed annotation or prediction binding cannot be edited, renamed, cloned,
  or redirected through the first-release workflow form;
- writing annotations without a prediction fixes only the annotation binding;
- a failed first write does not fix the attempted element binding or make its
  name field read-only;
- attaching an existing annotation or prediction creates an immediately fixed
  binding;
- generic feature-cache cleanup cannot remove workflow manifests;
- failed writes leave the in-memory working layers intact and the UI
  recoverable;
- persisted metadata is sufficient to explain the source image, scale, channels,
  classes, and model used.

### Slice 7: Classifier bundle export and usability hardening

Classifier persistence is not required to prove the first end-to-end flow, but
it is important for a genuinely useful product and should follow immediately.
This slice adds `workflow_mode: prediction_only` as a first-class persisted
application workflow; it does not weaken the annotation-binding requirements of
`workflow_mode: annotation`.

Deliver:

- export the fitted Random Forest and its compatibility metadata in a versioned
  Harpy classifier bundle with a stable `classifier_id`;
- record the originating workflow ID and annotation revision, or every
  contributing workflow ID and revision for a pooled classifier;
- expose an `Apply existing classifier` UI mode that accepts either a trusted
  portable bundle or the classifier from the current session;
- materialize a current-session classifier through the same bundle contract
  before application so file-based, interactive, and headless paths share one
  validator;
- load and validate the bundle before asking the user to predict;
- select one explicit local target coordinate system, image, and scale without
  requiring any annotation element on that target;
- match channels by unique name and reorder inputs to the saved model schema;
- block on missing or duplicate required channel names and show the complete
  resolution result in the status card;
- serialize and validate the selected-resolution descriptor without physical
  spacing or unit fields;
- recommend, without forcing, the target scale whose relative spacing is
  closest to the training descriptor among the scales that satisfy the normal
  practical-size recommendation;
- show a non-blocking warning for cross-image relative-spacing mismatches;
- warn that raw-intensity classifiers assume comparable acquisition and
  intensity ranges;
- predict directly on the selected target image grid and render the Labels
  result with the target image's transformation into the selected coordinate
  system; do not warp or resample the target into the training coordinate
  system;
- create an in-memory `prediction_only` workflow draft with no annotation
  binding, annotation array, annotation revision, or training controls;
- initialize the target class IDs, names, and colors from the bundle; keep ID
  membership read-only while permitting name and color display edits under the
  normal non-staling metadata rules;
- use the Slice 5 private-candidate prediction, cancellation, replacement, and
  revision-guard contract unchanged;
- provide mode-appropriate `Write Prediction State` and
  `Reload Prediction State` actions backed by the common workflow persistence
  machinery;
- require an explicitly saved, validated local classifier-bundle path before
  persisting a current-session result; record its normalized path relative to
  the sidecar together with its `classifier_id` and bundle schema version;
- do not copy, import, overwrite, or delete the referenced classifier bundle
  when applying or persisting a workflow, and do not create a managed
  classifier-bundle directory in the sidecar;
- validate the classifier reference, write the complete fresh prediction
  Labels element, and publish the `prediction_only` manifest last;
- persist no standalone prediction Labels through the normal widget workflow:
  a persisted interactive result must be discoverable through its
  `prediction_only` manifest;
- reload an existing prediction-only workflow and its prediction without the
  original annotation dataset; load the referenced bundle when re-prediction is
  requested;
- if the referenced bundle is missing or its identity does not validate,
  reload the persisted prediction in a degraded workflow, explain the problem,
  and disable re-prediction;
- create one prediction-only workflow per target application; several target
  coordinate systems and sidecars may reference the same canonical classifier
  bundle;
- create a new prediction-only workflow when applying a different classifier
  rather than mutating a fixed classifier binding;
- focused progress, cancellation, dirty-state, and error-message polish;
- user documentation describing scale choice and raw-intensity limitations.

The plug-and-play UI order is:

```text
Apply existing classifier
  -> Choose trusted portable bundle or current-session classifier
  -> Save the current-session classifier once if persistence will be requested
  -> Choose target coordinate system
  -> Choose target image
  -> Accept or change recommended target scale
  -> Review automatically resolved and reordered channels
  -> Review compatibility warnings
  -> Predict
  -> Review prediction
  -> Write Prediction State
```

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

An unpersisted prediction-only result may remain an in-memory result and may be
discarded without creating a workflow. If it uses a current-session classifier
that has not yet been saved, `Write Prediction State` asks the user to save the
canonical bundle first. Once persistence is requested, the validated
classifier reference, prediction Labels, and final manifest form one UI-level
consistency unit. A failed reference validation, prediction write, or manifest
write keeps the in-memory result recoverable and reports the failing stage; no
cross-element rollback is promised.

If the user later wants to annotate the target, a future explicit
`Start annotation workflow from this target` action may create a separate
`annotation` workflow and copy the class display schema. Slice 7 does not create
an empty annotation array, silently convert workflow modes, or treat a
prediction-only target as training data.

Acceptance criteria:

- a bundle trained in one coordinate system can predict a compatible image in
  another coordinate system without access to the original annotation Labels;
- prediction shape and transform come from the selected target grid, with no
  resampling to the training target and no requirement that spatial shapes
  match;
- required channels are resolved by unique name and passed to the estimator in
  saved bundle order;
- missing or duplicate required channels block prediction; spacing and
  raw-intensity comparability concerns warn without blocking;
- the scale recommendation uses the closest available relative spacing within
  the practical-size recommendation and remains user-overridable;
- a prediction-only workflow contains no annotation binding, annotation
  revision, annotation layer, annotation dirty state, or training action;
- class-ID membership is fixed by the bundle, while name and color edits remain
  display metadata and do not stale a compatible prediction;
- persistence records the validated canonical bundle's `classifier_id`, schema
  version, and path relative to the sidecar without copying the bundle;
- a current-session classifier can predict in memory before it is saved, but
  persistence requires the user to choose and save its canonical bundle path;
- several workflows and sidecars may reference the same canonical bundle
  without creating duplicate model files;
- moving a destination SpatialData store and sidecar together preserves the
  reference only when their relative relationship to the canonical bundle is
  also preserved; otherwise reload enters the defined degraded state;
- a persisted prediction-only workflow requires one complete fresh prediction
  and one valid classifier-bundle reference and is published only after both
  exist;
- cancellation or failure before persistence leaves no partial prediction-only
  manifest or apparently complete prediction;
- reload restores the prediction, class display schema, target descriptor, and
  freshness without creating an annotation layer, even if the classifier file
  has become unavailable;
- a missing classifier file, schema mismatch, or `classifier_id` mismatch is
  reported loudly and disables re-prediction rather than triggering a search or
  implicit copy;
- applying a different classifier creates a new workflow;
- normal application and persistence never create a managed classifier-model
  directory or any sidecar-owned model copy;
- no normal interactive persistence path writes an undiscoverable standalone
  prediction Labels element.

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
- deterministic pooled sampling with one global 50,000-candidate cap per class,
  allocated fairly across contributing targets as defined below;
- one pooled Random Forest;
- one active-target prediction at a time;
- training summary showing contributed pixels by target and class;
- pooled classifier provenance listing every contributing workflow ID and
  annotation revision;
- compatibility checks for channel schema and raw-intensity assumptions.

The 50,000-candidate limit applies **per class across the complete training
pool**, not per target and class. For each class:

1. count its annotated pixels independently in every selected target without
   reading marker intensities;
2. exclude targets with zero annotated pixels for that class from its allocation
   calculation;
3. set the class budget to the smaller of 50,000 and the class's total annotated
   count across all targets;
4. divide that budget as evenly as possible among contributing targets;
5. when a target has fewer annotated pixels than its provisional allocation,
   use all of them and redistribute the unused allocation evenly among targets
   that still have remaining capacity;
6. assign any indivisible integer remainder in stable workflow-ID order;
7. sample the resulting allocation without replacement within each target,
   using the same bounded annotation-first selection contract as single-target
   training;
8. read only the selected marker rows and exclude non-finite rows without
   replacement sampling.

This is a deterministic max-min-fair allocation. It allows small targets to
contribute all their examples, prevents a densely annotated target from
dominating while other targets have capacity, and keeps the total training
matrix bounded by `50,000 * number_of_classes` regardless of target count. Do
not oversample small targets or duplicate rows. Continue to use
`class_weight="balanced_subsample"` for remaining class imbalance; the first
pooled implementation does not add separate per-target estimator weights.

Canonicalize target processing by stable workflow ID before allocation and
sampling so changing target-card order cannot change the result. A one-target
pooled request must delegate to the same sampling path and produce the same
candidate rows as single-target training.

Targets may have different `(y, x)` shapes and different scale-key strings.
Compare their dimensionless relative spacing and report mismatches, but do not
require physical pixel spacing or physical units. Raw-intensity comparability
across batches is not guaranteed; the UI must state that clearly.

Acceptance criteria:

- a one-target pooled request produces the same sampled rows and fitted model as
  the validated single-target path;
- every target contributes through an explicit target descriptor rather than
  hidden widget state;
- each training target descriptor resolves from one selected annotation
  workflow manifest; prediction-only workflows are never pooled training
  inputs;
- channel order and class meaning are identical across targets;
- each class contributes at most 50,000 candidates across the complete pool,
  independent of the number of selected targets;
- per-target, per-class allocations follow the deterministic max-min-fair rule,
  redistribute unused capacity, and do not oversample or duplicate rows;
- changing target-card order does not change allocations or sampled candidates;
- the training summary reports annotated, allocated, sampled,
  non-finite-excluded, and used counts for every target and class;
- one invalid target fails validation clearly and cannot silently disappear
  from training;
- widget and future headless pooled training use the same core function.

### Slice 9: Public headless pixel-classification APIs

Expose supported synchronous APIs after the classifier bundle and pooled core
contracts are stable. Mirror the object-classification headless pattern while
using pixel-specific target descriptors.

Deliver:

- immutable explicit descriptors for training and prediction targets, including
  optional existing workflow ID, local SpatialData Zarr and sidecar paths, image
  name, annotation Labels name for training targets only, coordinate system,
  selected-resolution descriptor, and ordered channel names;
- `train_pixel_classifier(...)` accepting one or more training targets and
  returning the same versioned classifier bundle used by the widget;
- `apply_pixel_classifier(...)` accepting a loaded bundle and one explicit
  prediction target;
- path-based load/apply convenience wrappers consistent with the existing
  object-classifier headless API;
- the same per-class cap, `balanced_subsample` weighting, compatibility checks,
  block-wise source reading, in-memory `uint8` output, and provenance as the
  widget path;
- explicit optional persistence through the same prediction output and
  workflow-reference helpers as the widget: persistence requires and validates
  the canonical classifier-bundle path, writes the prediction Labels element,
  and publishes a `prediction_only` manifest last without copying the bundle;
- result objects that report target identity, output identity, resolution
  descriptor, class counts, warnings, and whether persistence occurred.

Headless functions are synchronous and must not import napari, Qt, widget
controllers, or `thread_worker`. The caller decides whether to run them in a
process, thread, notebook, or batch system. They must not infer selections from
viewer or app state, and they must never silently choose an image, scale,
channel, annotation element, or coordinate system.

Acceptance criteria:

- headless targets reject unbacked, in-memory, and remote SpatialData inputs
  under the same public source contract as the widget;
- a script can train from one target, train from several pooled targets, load a
  bundle, and predict without creating napari or Qt objects;
- widget and headless calls with the same targets and seed produce equivalent
  training samples, fitted-model predictions, and metadata;
- headless apply rejects incompatible channel schemas and reports non-blocking
  relative-spacing warnings exactly like the widget;
- prediction reads the multiplex source block-wise and returns one in-memory
  selected-scale `uint8` result;
- a non-persisted application returns the in-memory result without creating a
  workflow or Labels element;
- optional persistence writes the same Labels model, transform, sidecar
  classifier reference, prediction-only manifest, and provenance as the
  interactive path, never copies the classifier bundle, and never creates an
  annotation element;
- persistence of a loaded in-memory bundle without a canonical path fails with
  a clear request to save the bundle or provide its path first;
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
Supported local SpatialData
  -> Choose workflow mode
       |
       +--> Annotate and train
       |      -> Choose coordinate system, image, and scale
       |      -> Create/select annotation workflow
       |      -> Choose channels
       |      -> Create/reload workflow annotations
       |      -> Paint at least two classes
       |      -> Train
       |      -> Predict
       |      -> Write/reload workflow Labels state
       |
       +--> Apply existing classifier
              -> Load portable or current-session classifier
              -> Choose coordinate system, image, and scale
              -> Resolve channels and review compatibility
              -> Predict
              -> Write/reload prediction state
```

Rules:

- workflow creation or selection is enabled only when the source has a
  supported local Zarr backing path and the target image, scale, and transform
  are valid;
- annotation creation is enabled for a valid new workflow draft; annotation
  reload is enabled only for a selected valid persisted workflow;
- `prediction_only` creation is enabled only after a trusted classifier bundle
  and compatible explicit target validate successfully;
- changing channels does not change workflow eligibility or invalidate its
  annotation binding;
- training does not depend on prediction or persisted annotations;
- prediction requires either a fresh trained classifier or a validated loaded
  classifier bundle compatible with the target;
- writing Labels state for `annotation` requires a working annotation layer and
  includes a prediction data write only when a complete, fresh prediction
  exists; persistent writes also require a selected workflow, writable local
  source Zarr, and writable local sidecar destination;
- writing prediction state for `prediction_only` requires a valid canonical
  classifier-bundle path, matching `classifier_id`, and one complete, fresh
  prediction; no annotation state is created or written and the bundle is not
  copied;
- a stale persisted prediction remains bound but is not rewritten, while a
  stale unpersisted prediction remains unbound and is not written;
- dirty state records differences from the last persisted workflow, whereas
  stale state records invalid derived provenance; neither state implies the
  other;
- a newly generated complete, fresh prediction is dirty until written; making
  an unchanged prediction stale does not make its pixel array dirty;
- a running prediction candidate is private and non-persistable; the previous
  prediction remains displayed as stale/recalculating until a complete,
  revision-valid candidate replaces it;
- cancelling, failing, or rejecting a prediction candidate discards only that
  candidate and retains the previous array and provenance;
- an ephemeral stale prediction does not keep the aggregate Labels state dirty
  after the persistable annotation and manifest state has been written;
- reloading Labels state for `annotation` requires a persisted annotation
  element; a persisted prediction element is optional, and both are resolved
  through the selected workflow manifest;
- reloading prediction state for `prediction_only` requires its persisted
  prediction element and must not create an annotation layer; an unavailable
  or mismatched classifier reference produces a degraded reload with
  re-prediction disabled rather than hiding an otherwise valid prediction;
- changing only viewer contrast, colormap, opacity, or channel-overlay
  visibility does not stale anything;
- changing a class name or color marks class-schema and manifest state dirty but
  does not stale the classifier or prediction;
- adding a class or removing a zero-count class changes current class-ID
  membership and stales the classifier and prediction;
- prediction-only workflows prohibit class-ID membership changes because the
  output schema is fixed by the classifier bundle;
- removing a class with annotated pixels and editing an existing class ID are
  rejected without a state change; an ID absent from the current schema may be
  assigned to a newly created class;
- changing biological meaning requires a new class ID and annotation changes;
  a rename alone is never treated as semantic reassignment;
- changing selected channels or their order stales model and prediction;
- changing workflow selection, scale, image, or coordinate system changes the
  active target context and requires an explicit write/discard/cancel decision
  when the current workflow state is dirty;
- selecting a different classifier for application creates a new
  prediction-only workflow instead of mutating a persisted classifier binding;
- changing only the workflow display name or an as-yet-unwritten element-name
  destination marks workflow metadata dirty but does not stale a compatible
  classifier;
- successfully writing an element changes its name field to read-only; no
  first-release state transition renames or clones a fixed binding.

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
- One active prediction is one in-memory `uint8` array and uses one byte per
  selected-scale pixel: 64 MiB at `8192 * 8192`. Re-prediction may temporarily
  hold the previous displayed array plus one private candidate, for a maximum
  of two full-size prediction arrays.
- No `C + F` feature raster is stored.
- Source images remain Dask/Zarr-backed where available.
- Long-running work runs outside the Qt main thread.
- Progress is based on training extraction stages or prediction blocks.
- Cancellation leaves annotations and any previous prediction untouched.

The `8192 * 8192` recommendation heuristic is a usability default, not a memory
limit. The first implementation does not add memory estimation, confirmation
thresholds, or hard-stop behavior.

## Test Strategy

Use focused tests for each slice, following repository test-scope guidance.

Core tests:

- workflow manifest serialization, schema validation, and stable identity;
- tagged-manifest validation for `annotation` and `prediction_only`, including
  required and forbidden mode-specific fields;
- Slice 6 annotation-manifest round-trip and explicit rejection of unsupported
  modes before Slice 7;
- local filesystem Zarr backing-path resolution and rejection of unbacked,
  in-memory, remote, and URI-backed sources;
- default adjacent and explicit local sidecar relative-source serialization;
- source-reference resolution from the sidecar root rather than the process
  working directory;
- successful workflow discovery after moving dataset and sidecar together;
- loud invalidation when only the dataset or sidecar is moved or renamed;
- target-descriptor mismatch reporting without basename guessing, directory
  scanning, absolute fallback, or silent relinking;
- complete write prevalidation before any Labels or manifest mutation;
- annotation, optional prediction, then staged-manifest write ordering;
- annotation-, prediction-, and manifest-stage failure results that remain dirty
  and identify possible partial disk updates without cross-element rollback;
- fresh prediction write and manifest provenance;
- retained, unchanged persisted prediction becoming explicitly stale after an
  annotation-only write;
- stale never-persisted prediction exclusion from disk and manifest bindings;
- independent annotation, prediction, manifest, and aggregate dirty-state
  transitions;
- independent class-schema dirty-state transitions for current class names,
  colors, and membership;
- class rename and recolor persistence without classifier or prediction
  staleness;
- zero-count class removal from the current manifest with derived-artifact
  staleness and no retired-class registry;
- rejection of class removal with annotated pixels without annotation or schema
  mutation;
- rejection of existing class-ID editing and acceptance of an ID absent from
  the current schema for a newly created class;
- authoritative current class-schema round-trip through the manifest without
  dependence on SpatialData or Labels attrs;
- artifact-specific class-schema snapshots in classifier and prediction
  provenance, including correct rendering of a stale prediction after an ID is
  reused by the current schema;
- stale-state transitions that leave unchanged prediction pixels clean;
- successful annotation-only write clearing aggregate dirty state while an
  ephemeral stale prediction remains visible with a discard-on-reload warning;
- annotation-only write with no prediction;
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
- private-candidate cancellation and failure without a previous prediction;
- non-destructive re-prediction cancellation and failure with a previous
  prediction;
- successful atomic prediction-layer data replacement after completion and
  active-revision validation;
- obsolete prediction-result rejection with the previous array and provenance
  retained;
- previous-prediction stale/recalculating presentation and post-failure
  freshness restoration;
- SpatialData labels parse, write, reload, and alignment;
- metadata compatibility and stale-state decisions;
- selected-resolution descriptor round-trip, including anisotropic spacing and
  affine values;
- same-image exact scale-key validation;
- cross-image scale-key independence and non-blocking relative-spacing mismatch
  warnings;
- portable classifier-bundle round-trip with stable classifier ID, channels,
  class schema, training resolution, and training provenance;
- current-session and reloaded portable bundles producing equivalent target
  compatibility results and predictions;
- unique-name target channel resolution and saved-order reordering, including
  hard failures for missing and duplicate required channels;
- closest-relative-spacing scale recommendation within the practical-size
  recommendation and explicit user override;
- prediction on a differently shaped image and coordinate system using the
  target grid and target transform without training-grid resampling;
- prediction-only manifest validation with no annotation fields and fixed
  classifier-defined class-ID membership;
- canonical classifier-path normalization relative to the sidecar and
  round-trip resolution;
- several workflows and sidecars referencing one canonical classifier file
  without producing model copies;
- rejection of prediction-only persistence for a current-session classifier
  until a canonical bundle path has been saved or provided;
- classifier-reference validation, prediction Labels, then manifest-last write
  ordering and stage-specific failure recovery;
- prediction-only reload and re-prediction without the original annotation
  Labels when the referenced canonical bundle remains valid;
- degraded prediction-only reload with the persisted prediction retained and
  re-prediction disabled when the bundle is missing or its `classifier_id` or
  schema does not match;
- rejection of normal standalone persisted prediction Labels without a
  prediction-only workflow record;
- one-target pooled-core equivalence with the single-target path;
- deterministic global per-class pooled caps, max-min-fair target allocation,
  unused-capacity redistribution, and integer-remainder handling;
- pooled sampling invariance to target-card order, without replacement or
  duplicated rows;
- annotated, allocated, sampled, non-finite-excluded, and used contribution
  summaries for every target and class;
- pooled validation failure when any requested target is invalid;

Widget tests:

- unsupported-source status disables all pixel-classification workflow actions;
- source-association mismatch status reports the sidecar, stored relative
  reference, resolved source, and failed descriptor;
- new versus existing workflow selection;
- existing workflow channel and class-schema restoration;
- several eligible workflows require explicit selection;
- arbitrary unregistered Labels are excluded from normal workflow choices;
- target-control gating;
- scale summary and recommendation heuristic;
- annotation creation and layer reuse;
- class creation, selection, renaming, coloring, removal, stable IDs, and
  annotated-pixel counts;
- all-class annotated, sampled, used, non-finite-excluded, and capped training
  fields in the status card;
- non-blocking cross-image relative-spacing mismatch warning;
- dirty Labels-state prompts for reload and target changes;
- stage-specific write-failure status cards with retry guidance;
- stale-prediction write status with `Predict` refresh guidance;
- worker result revision guards;
- separate annotation and prediction layers;
- prediction layer read-only behavior;
- previous prediction remains visible with stale/recalculating status during
  re-prediction;
- cancelled or failed re-prediction leaves the previous layer data unchanged;
- successful re-prediction updates the existing layer only after completion and
  does not add a duplicate;
- paired write/reload behavior and overwrite confirmation;
- independent annotation and prediction binding freeze on first successful
  write;
- failed first writes leave the corresponding draft name editable;
- fixed binding fields are read-only while the workflow display name remains
  editable;
- `Apply existing classifier` accepts a trusted portable or current-session
  bundle and exposes no annotation layer or training controls;
- application mode automatically resolves channels, recommends scale, and
  reports blocking compatibility errors separately from non-blocking warnings;
- prediction-only class IDs are fixed while names and colors remain editable
  display metadata;
- `Write Prediction State` records the validated canonical classifier path and
  ID without copying the bundle, then publishes a discoverable prediction-only
  workflow;
- `Reload Prediction State` restores the prediction without creating
  annotations and clearly degrades when the classifier cannot be resolved;
- applying a different classifier creates a new prediction-only workflow;
- destroyed widget ignores late callbacks.

Headless tests:

- public-module import without napari, Qt, widgets, or worker machinery;
- rejection of unbacked, in-memory, and remote source targets;
- single-target and pooled training through explicit target descriptors;
- widget/core/headless equivalence for sampling, prediction, and metadata;
- explicit failure for missing or incompatible target selections;
- block-wise prediction with the same in-memory `uint8` result as the widget;
- non-persisted application leaves SpatialData, sidecar, and workflow discovery
  unchanged;
- optional prediction persistence records the same classifier reference, Labels
  transform, prediction-only manifest, and provenance as the interactive path
  without copying the bundle or creating annotation state;
- persistence without a canonical classifier path fails clearly, while one
  canonical bundle can be referenced by multiple headless applications.

At least one integration test should use a real multiscale SpatialData image
whose selected scale has a known non-unit factor and whose image has a nontrivial
2D transformation. The test should write annotations and predictions, reload
the Labels state, and verify world-coordinate alignment.

## Completion Definition

The first usable release is complete when a user can:

1. open a supported local filesystem-backed SpatialData Zarr and select a
   coordinate system, image, and actual multiscale level;
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
