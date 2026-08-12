# Napari/VisPy Integration for Tiled Transcript Rendering

Status: focused implementation specification and roadmap

Initial compatibility target: the repository's currently installed
`napari 0.7.1` and `vispy 0.16.2`

Last updated: 2026-08-10

## Authority and relationship to the main roadmap

This document expands the napari/VisPy boundary described in
[`multi_tile_cache_29_7_26.md`](multi_tile_cache_29_7_26.md). It is authoritative
for the initial napari integration spike, the first tile-retaining VisPy
backend, and their integration into Harpy.

The main roadmap remains authoritative for:

- persistent cache construction and publication;
- cache schemas and source identity;
- cache reading, viewport planning, and request scheduling;
- CPU cache and LOD policy;
- the complete product workflow.

This document deliberately targets the currently installed napari version.
Supporting or migrating to a later napari version is follow-up work and must not
delay proving the present integration boundary.

## Executive decision

Harpy will start napari integration before the persistent cache reader, planner,
and scheduler are complete.

The first implementation is a deliberately narrow vertical spike:

```text
TranscriptLayerModel
        │
        │ private napari 0.7.1 registration
        ▼
VispyTranscriptLayer
        │
        ▼
one synthetic in-memory tile
        │
        ▼
napari canvas, transforms, layer controls, blending, and cleanup
```

The spike validates the most fragile external boundary while the cache work
continues independently. It does not imitate a cache on disk and does not
introduce temporary Parquet, planner, or scheduler behavior.

The synthetic tile must use the intended runtime payload shape closely enough
that replacing it with a real `TilePayload` does not require redesigning the
layer or visual boundary.

## Goals

The initial integration work must establish that:

- napari can host a custom logical transcript layer without a complete
  `N x 2` coordinate array;
- the layer reports complete dataset extent independently of resident tiles;
- napari automatically constructs the corresponding VisPy layer and Qt
  controls;
- one tile of local `float32` point coordinates renders at the correct dataset
  location;
- the existing napari layer transform is applied exactly once;
- camera and canvas changes can be converted into a stable viewport event for
  later planning;
- visibility, opacity, blending, layer order, and transform changes behave like
  normal napari layer state;
- removing the layer closes the rendering object and disconnects all Harpy
  callbacks;
- repeated add/remove cycles do not retain application-owned tile resources;
- later multi-tile rendering can retain independently addressable GPU buffers.

## Non-goals of the initial spike

The first spike does not implement:

- persistent cache validation or reading;
- manifest or Parquet access;
- viewport-to-tile planning or LOD selection;
- background tile loading;
- a CPU tile LRU;
- multiple active tiles;
- GPU memory budgeting or eviction;
- cross-level snapshot transitions;
- the production value-palette shader;
- per-transcript picking;
- cache build/rebuild UI;
- automatic direct/tiled mode selection;
- compatibility with napari versions other than the currently installed
  `0.7.1`.

The spike may define interfaces needed by those components, but it must not
implement speculative versions of them.

## Current repository facts

The current environment contains:

```text
napari 0.7.1
vispy 0.16.2
```

The project dependency remains broad at `napari[all]>=0.4.18`. That declaration
continues to describe the existing application as a whole. It does not imply
that the private tiled-renderer integration works on every allowed napari
version.

For the first implementation:

- tiled transcript rendering explicitly supports napari `0.7.1`;
- registration feature-detects the exact private objects it needs;
- an unsupported environment receives a clear tiled-renderer error;
- failure to enable tiled rendering must not break the existing direct Points
  path;
- widening or changing the supported version range is a separate review.

Napari 0.7.1 selects custom visual implementations through the private
`napari._vispy.utils.visual.layer_to_visual` mapping and custom controls through
the private
`napari._qt.layer_controls.qt_layer_controls_container.layer_to_controls`
mapping. All imports and mutations of those registries must remain isolated.

## Package and dependency boundary

Napari-facing code belongs under a dedicated viewer package:

```text
src/napari_harpy/viewer/multi_scale_points/
    __init__.py
    _layer.py
    _models.py
    _napari_registration.py
    _qt_controls.py
    _vispy_layer.py
```

Only add later modules when their responsibility is implemented:

```text
    _session.py
    _upload_pump.py
    _vispy_backend.py
    _visuals.py
```

The core cache package at
`napari_harpy.core.multi_scale_cache_points` must not import Qt, napari, or
VisPy. Renderer-independent runtime models may live in the core package once
the cache reader/scheduler owns them. The initial synthetic models stay at the
viewer boundary until that ownership is concrete.

Private napari symbols may be imported only by `_napari_registration.py`,
`_qt_controls.py`, and `_vispy_layer.py`. The registration module is the only
module allowed to mutate napari's private registries.

## Locked layer-model contract

### Model role

`TranscriptLayerModel` is the persistent, view-independent object shown in
napari's layer list.

It owns:

- an immutable transcript dataset reference;
- complete dataset bounds in native point coordinates;
- active value-selection and palette state when those features are added;
- transcript point-style state;
- render preferences and user-visible status when those features are added;
- normal napari visibility, opacity, blending, and transforms;
- a small custom event carrying a deduplicated viewport description.

It does not own:

- a complete transcript coordinate array;
- current tile payloads;
- camera position as persistent model state;
- a Parquet reader or open file handles;
- worker objects;
- pending requests;
- VisPy nodes, buffers, or OpenGL state;
- CPU or GPU LRU state.

The model subclasses `napari.layers.Layer`, not `Points`. Its required `data`
property returns the immutable dataset reference or another explicitly named
logical-data object. It must never return the currently resident points and
must not suggest that resident tiles are the canonical dataset.

### Dimensionality and editing

The first transcript layer is two-dimensional:

```text
ndim = 2
axis order = y, x
```

It is read-only. Its supported interaction mode is pan/zoom; transcript adding,
deleting, moving, selection, and transform handles are not part of the spike.
Programmatic updates of the normal layer transform remain supported.

The spike must define explicit behavior when the viewer switches to
`ndisplay == 3`. The initial acceptable policy is to keep the layer as a flat
2D plane in the last two world dimensions or hide it with a clear status. The
behavior must not be accidental or crash-prone.

### Extent

`_extent_data` always returns the complete transcript dataset bounds in napari
`y, x` order:

```text
[[y_min, x_min],
 [y_max, x_max]]
```

The extent is independent of:

- the synthetic tile;
- active or pending snapshots;
- selected values;
- CPU or GPU eviction;
- visibility of individual tile visuals.

Loading, replacing, hiding, or evicting a tile must not emit an extent change.
Changing the dataset reference may replace the extent as one explicit model
operation.

## Synthetic payload contract

The spike uses one immutable, already-decoded tile. It should mirror the
eventual reader output rather than the physical Parquet column order.

Conceptually:

```python
@dataclass(frozen=True)
class SyntheticTilePayload:
    tile_key: TileKey
    tile_origin_yx: tuple[float, float]
    positions_yx_local: ndarray  # float32, C-contiguous, shape (N, 2)
    value_ids: ndarray           # uint32, C-contiguous, shape (N,)
    point_ids: ndarray           # uint64, C-contiguous, shape (N,)
```

The arrays are immutable after construction. Their first dimensions match and
they contain no NaN or infinite positions. `point_ids` remain CPU-side in the
spike.

Use recognizable, asymmetric geometry rather than only random points:

- all four tile corners inset by a known amount;
- a diagonal that distinguishes `y, x` from `x, y`;
- an off-centre cross;
- at least two value IDs with visibly different colors.

Suggested geometry:

```text
complete extent: y = 0 ... 20,000; x = 0 ... 30,000
tile origin:     y = 4,000;       x = 8,000
tile edge:       512
```

The persistent writer stores physical columns in `x_rel, y_rel` order. The
runtime reader is responsible for returning the explicitly named
`positions_yx_local` representation. The VisPy boundary reverses that array to
VisPy `x, y` ordering exactly once.

## Runtime identity contract to freeze before multi-tile work

A logical on-disk tile is not a sufficient GPU-residency identity. Filtered
payloads for two value selections may refer to the same logical tile while
containing different vertices.

Before a fake or production multi-tile backend is implemented, define one
immutable residency key containing at least:

```text
cache_generation_id
logical TileKey, including level
decode_contract
selection_key
```

All readiness, upload, lookup, activation, pinning, and eviction operations use
that full key. In particular, future backend methods must not expose ambiguous
operations such as:

```python
is_ready(tile_key)
evict(tile_key)
```

when several payload identities for that logical tile can exist.

This refinement must be reflected in the main runtime roadmap before the Phase
2 fake backend and the multi-tile VisPy backend share an interface.

## Coordinate and transform contract

There are exactly three transform stages:

```text
tile-local y/x + tile origin y/x
        -> dataset-native y/x
        -> napari layer transform
        -> world coordinates
        -> camera/canvas coordinates
```

The tile visual applies only the tile origin. `VispyBaseLayer` applies the
normal napari layer transform to the root layer node. The camera applies the
world-to-canvas transform.

The SpatialData transform must not be baked into tile coordinates. Harpy should
reuse the same conversion currently used by the direct points path:

```python
transform.to_affine_matrix(
    input_axes=("y", "x"),
    output_axes=("y", "x"),
)
```

The first visual should use a VisPy `CompoundVisual` root with one marker
subvisual for the synthetic tile. Tile origin is applied to that subvisual in
VisPy `x, y` order. This structure is intentional: napari may rewrite direct
scene-node child transforms during layer matrix updates, while compound
subvisual transforms remain independently controlled.

The spike must test:

- identity transform;
- translation;
- non-unit and anisotropic scale;
- rotation or shear through an affine matrix;
- transform changes after the tile was uploaded;
- negative or nonzero dataset origins;
- alignment with a small reference image layer;
- absence of half-pixel or double-transform offsets.

A transform change updates scene transforms only. It must not call the tile's
coordinate upload path.

## Viewport bridge contract

### Why an explicit bridge is required

Napari 0.7.1 does not send ordinary two-dimensional pan and zoom changes to
`VispyBaseLayer._on_camera_move`. During canvas drawing, napari instead calls
the private `Layer._update_draw(...)` method with:

- world-coordinate viewport corners;
- the world-units-per-canvas-pixel scale factor;
- the current viewbox size.

The transcript layer must convert this private callback into a small
Harpy-owned contract. The planner and scheduler must not import napari canvas
internals.

### `ViewportState`

The initial viewer-side viewport value should contain enough information for a
later pure planner:

```python
@dataclass(frozen=True)
class ViewportState:
    displayed_axes: tuple[int, ...]
    data_bounds_yx: tuple[float, float, float, float]
    canvas_size_yx: tuple[int, int]
    world_units_per_canvas_pixel: float
    ndisplay: int
```

If the planner ultimately needs more than one scalar for anisotropic or sheared
screen scale, the contract may replace the scalar with an explicitly named
projected-scale representation. It must not silently treat a scalar camera zoom
as data units per pixel under an arbitrary layer affine.

`TranscriptLayerModel._update_draw(...)` must:

1. call `super()._update_draw(...)`;
2. inverse-transform all four world viewport corners into layer data
   coordinates;
3. construct a conservative floating-point data-coordinate AABB;
4. clip it to the complete dataset bounds;
5. create a normalized immutable `ViewportState`;
6. emit a custom viewport event only when the normalized state changed.

The bridge reports state only. It does not select tiles, start IO, own workers,
or mutate the renderer.

Deduplication is mandatory because uploading a tile requests another canvas
draw. An upload-induced draw with the same viewport must not recursively create
another planning generation.

Transform changes, canvas resize, displayed-axis changes, and camera changes
must all produce a new viewport state when they materially change the plan.

## Private registration contract

Registration is explicit and occurs before adding the first transcript layer.
Importing the package alone must not silently mutate napari global registries.

The registration function must:

- verify `napari.__version__ == 0.7.1` for the initial implementation;
- feature-detect both required registry objects;
- verify that the expected factory functions remain callable;
- add the model-to-visual and model-to-controls mappings;
- be idempotent when the desired mappings already exist;
- reject a conflicting pre-existing mapping rather than overwrite it;
- return a small result describing whether registration was newly installed or
  already present;
- raise one Harpy-owned exception with an actionable message when unsupported.

Tests may provide an explicit unregister helper to isolate global state. Product
code must not rely on unregistering while live transcript layers exist.

All failures must leave both napari registries in their original state. If the
first insertion succeeds and the second fails, registration rolls back the
first insertion.

## Minimal Qt controls

The spike registers a real controls class because a bare custom `Layer` cannot
be selected safely in the Qt viewer without one.

The initial controls should expose only:

- opacity;
- blending;
- a read-only indication that this is a tiled transcript layer;
- optional synthetic-spike diagnostics such as resident point count.

Transcript editing controls are absent. Point-size and palette controls may be
added only if they are needed to validate style propagation in the spike.

The controls must disconnect model events when Qt destroys them. Their removal
is tested independently from VisPy resource cleanup.

## Initial VisPy layer contract

`VispyTranscriptLayer` subclasses `VispyBaseLayer` and owns one root compound
visual. For the spike it owns exactly one marker subvisual and uploads the
synthetic positions once.

It relies on `VispyBaseLayer` for:

- root visibility;
- root opacity;
- blending state;
- layer ordering;
- napari layer transforms;
- detaching the root node on close.

It owns:

- conversion of local `y, x` positions to VisPy `x, y`;
- the tile-origin subvisual transform;
- synthetic value colors;
- upload instrumentation;
- a closed/generation guard;
- clearing its application references during close.

The spike records at least:

```text
coordinate_upload_count
resident_tile_count
resident_point_count
closed
```

Changing opacity, blending, visibility, layer order, or layer transform must not
increment `coordinate_upload_count`.

### Point-size decision

The first spike must make point-size units explicit and compare the two useful
semantics:

- constant canvas-pixel diameter;
- constant dataset/world-space diameter.

The review gate selects one user-facing default before the compact production
shader is implemented. Until then, tests must state which semantics they expect
rather than inheriting a VisPy default accidentally.

## Cleanup and ownership

Napari calls the VisPy layer's `close()` when it removes a layer. Harpy uses
that callback as the final GUI/OpenGL cleanup boundary.

The spike's `close()` is idempotent and must:

- set `_closed` before releasing resources;
- increment or invalidate its generation;
- disconnect every additional event callback it installed;
- clear references to payload arrays and tile subvisuals;
- request deletion or release of application-owned GPU objects;
- call `super().close()` so napari events disconnect and the root node detaches;
- make late callbacks harmless;
- tolerate a second call.

The later production session owns store, planner, scheduler, and workers. The
VisPy layer owns the canvas-specific backend. Layer removal closes both through
one idempotent session shutdown path; neither object may resurrect the other.

Every asynchronous completion added later checks both:

```text
not closed
completion generation == current generation
```

## Upload-pump and readiness contract for follow-up work

The one-tile spike may upload synchronously during visual construction. The
multi-tile backend must not assume napari's optional `_on_poll` integration is
available: in napari 0.7.1 it is normally connected only for the experimental
monitor.

The production backend therefore owns a GUI-thread upload pump, initially a
short-lived `QTimer`:

- start it when the upload queue changes from empty to nonempty;
- process at most the configured byte or elapsed-time budget per tick;
- request a canvas redraw after submitting work;
- stop it when the queue becomes empty;
- stop and disconnect it during close;
- never perform Parquet or decoding work on its callback.

Before implementing atomic snapshot activation, define observable upload
states precisely:

```text
CPU_READY
UPLOAD_QUEUED
GL_COMMANDS_SUBMITTED
RENDERABLE
```

VisPy may defer actual GL commands until drawing. A hidden pending visual must
not wait forever for a draw that can occur only after it is activated. The
multi-tile renderer slice must test the selected pre-upload strategy in a real
OpenGL context before `RENDERABLE` is used as a scheduler condition.

## Integration with the existing Harpy viewer

The current points controller and `PointsLayerBinding` represent the direct
napari Points workflow. They should remain working and independently testable.

Tiled mode should add a distinct binding or an explicit rendering-mode field
rather than making helper functions that require `isinstance(layer, Points)`
silently accept a different layer type.

The recommended ownership is:

```text
points selection/value UI
        ├── direct Points workflow
        └── TiledTranscriptSession
                ├── TranscriptLayerModel
                ├── store/planner/scheduler
                └── canvas-specific VisPy backend
```

Direct and tiled modes may share:

- SpatialData points identity;
- coordinate-system selection;
- normalized value vocabulary;
- stable value colors;
- user-facing point-style defaults.

They do not share:

- a live napari layer instance;
- worker state;
- renderer buffers;
- direct-mode sampled coordinate arrays;
- tiled CPU/GPU cache state.

The product workflow and automatic mode selection are implemented only after
the synthetic integration spike and fake-backend scheduler tests pass.

## Implementation slices

Each slice is independently reviewable and has focused tests. Later cache work
may proceed in parallel because slices N0-N5 do not read a real cache.

### N0: freeze the viewer-side contracts

Deliverables:

- define the synthetic dataset reference and payload models;
- freeze `y, x` runtime axis naming;
- define the initial `ViewportState`;
- reconcile the full GPU residency key with the main runtime roadmap;
- record the napari 0.7.1-only compatibility policy;
- define point-size decision criteria.

Exit criteria:

- models are immutable and validate shape/dtype invariants;
- no Qt, napari, or VisPy object appears in a payload model;
- readiness and eviction APIs cannot be keyed by logical tile alone;
- the slice performs no rendering and no IO.

### N1: custom layer model without rendering

Deliverables:

- implement the minimal `TranscriptLayerModel` subclass;
- implement its minimal slicing-state support;
- expose the logical dataset reference through `data`;
- report complete data/world extent;
- make the layer read-only;
- define 2D behavior in a higher-dimensional viewer;
- provide minimal state serialization semantics or explicitly mark unsupported
  export behavior.

Exit criteria:

- the model can be added, selected, hidden, fit to view, and removed from a
  `ViewerModel` without Qt or VisPy;
- it never requires a complete coordinate array;
- affine changes update world extent correctly;
- changing synthetic resident data cannot change extent;
- abstract napari layer methods have intentional behavior and tests.

### N2: private registration and minimal controls

Deliverables:

- implement transactional, idempotent napari 0.7.1 registration;
- register a minimal controls class;
- add unsupported-version and missing-private-API errors;
- isolate registry mutations between tests.

Exit criteria:

- adding the custom layer resolves the intended controls class;
- repeated registration does not change the registries again;
- conflicting mappings fail without partial mutation;
- controls are removed and disconnected with the layer;
- the existing built-in layer registrations remain unchanged.

### N3: one-tile VisPy vertical spike

Deliverables:

- implement `VispyTranscriptLayer` with one compound marker subvisual;
- upload one recognizable synthetic tile;
- apply the tile origin at the subvisual boundary;
- instrument coordinate upload count;
- test normal layer styling propagation;
- add a transformed reference image fixture.

Exit criteria:

- napari automatically constructs the custom visual;
- the tile appears at the expected image coordinates;
- `y, x`/`x, y` swaps are visibly and numerically excluded;
- identity, translation, scale, affine/shear, and post-add transform changes
  preserve alignment;
- transform and style changes do not reupload coordinates;
- fit-to-view continues to use complete dataset bounds;
- layer ordering, visibility, opacity, and blending work;
- at least one test runs with a real OpenGL canvas or the same checks are
  completed as a documented manual acceptance run.

### N4: viewport bridge

Deliverables:

- override the napari draw callback narrowly;
- create and emit immutable `ViewportState` values;
- deduplicate identical draw states;
- handle transform, resize, dims, pan, and zoom changes;
- add a recorder in place of a planner.

Exit criteria:

- one camera gesture may cause many draws but does not emit unbounded identical
  viewport events;
- upload-induced redraws do not create a loop;
- inverse-transformed bounds conservatively cover all visible data;
- rotation/shear never omit a visible corner;
- viewport state is clipped to complete data bounds;
- no IO, planner, or worker is introduced.

### N5: lifecycle and compatibility hardening

Deliverables:

- implement idempotent visual close;
- add closed/generation guards;
- instrument retained payload and visual counts;
- repeat add/remove and viewer-close scenarios;
- document the selected point-size semantics;
- record the napari 0.7.1 private APIs on which the feature depends.

Exit criteria:

- repeated add/remove does not accumulate Harpy callbacks, payload references,
  controls, or tile subvisuals;
- a removed layer cannot be changed by a late callback;
- the root visual detaches from the scene graph;
- application-owned upload timers are absent after close;
- unsupported environments fail before inserting a transcript layer;
- the normal direct Points workflow still passes its focused tests.

### Gate N: napari integration boundary accepted

Gate N approves moving from the one-tile spike to a tile-retaining backend.

Approval requires:

- every N1-N5 exit criterion;
- visual evidence of image/transcript alignment;
- an agreed point-size convention;
- an agreed viewport event contract;
- an agreed full residency key;
- no unresolved add/remove leak;
- no dependency on a real cache, reader, planner, or scheduler.

### R1: fake-backend scheduler contract

This slice belongs primarily to the main roadmap's runtime phase but must use
the integration contracts frozen here.

Deliverables:

- update the backend protocol to use full residency keys;
- implement a deterministic fake backend;
- test generations, stale results, pinning, and activation;
- define failed-snapshot and core-tile failure behavior.

Required deterministic sequence:

```text
view 1: A B
view 2: B C
view 3: C D
```

Exit criteria:

- B is not uploaded twice;
- obsolete payloads cannot satisfy a new selection;
- incomplete cross-level snapshots do not replace the active snapshot;
- a failed core tile cannot leave a snapshot waiting forever;
- tests import neither Qt nor VisPy.

### R2: multi-tile marker backend

Deliverables:

- replace the one synthetic subvisual with independently retained tile
  subvisuals;
- implement active and pending snapshot visibility;
- retain overlaps across same-level pans;
- implement byte accounting and explicit eviction;
- use the GUI-thread upload pump;
- test renderable/readiness behavior in a real context.

Exit criteria:

- resident overlapping tiles retain their buffers;
- only entering or selection-replacement payloads upload;
- cross-level activation is atomic;
- active and pending core tiles cannot be evicted;
- the upload pump respects its per-tick budget;
- close releases all backend-owned resources.

### R3: compact production point visual

Deliverables:

- compare standard marker-per-tile rendering with a compact transcript visual;
- upload local `float32` positions and a compatible dense `value_id` attribute;
- implement palette and enabled/highlight lookup;
- implement point size and global opacity as uniforms;
- keep `point_id` CPU-side;
- benchmark draw calls, upload time, frame time, and GPU bytes.

Exit criteria:

- palette/style-only changes never reupload coordinates;
- compact attribute conversion is exact for the supported value-ID range;
- large dataset coordinates retain tile-local precision;
- the selected visual strategy has measured evidence over the marker baseline;
- GPU memory remains within the configured budget.

### P1: Harpy tiled-session integration

Deliverables:

- add the tiled layer binding and lifecycle session;
- connect the real store/planner/scheduler/backend;
- reuse the existing SpatialData affine conversion;
- connect value-selection changes to replanning;
- connect layer visibility and removal to scheduling shutdown;
- retain direct mode as fallback and correctness comparison.

Exit criteria:

- tiled mode opens without scanning or materializing the complete source;
- the complete extent remains stable during all tile changes;
- exact tiled views agree with the direct path on deterministic fixtures;
- selection changes cannot show a stale selection snapshot;
- removing the layer stops IO, upload, and callbacks;
- direct mode remains usable when tiled mode is unavailable or fails.

### P2: product workflow and recovery

Deliverables:

- cache detected/compatible/stale/absent states;
- build and rebuild actions;
- auto, tiled, and direct rendering choices;
- progress and diagnostics;
- corrupt-tile and failed-snapshot recovery;
- actionable unsupported-renderer messages.

This slice is outside the initial integration spike.

## Test strategy

### Non-GL model and contract tests

Test without constructing a Qt viewer where possible:

- dataset reference instead of coordinate-array data;
- complete data and transformed world extent;
- fit-to-view center and scale;
- 2D layer inside a higher-dimensional `ViewerModel`;
- immutable payload validation;
- residency-key separation across selection and generation;
- viewport normalization and deduplication;
- unsupported-version and registration-rollback behavior;
- idempotent close state.

### Qt tests without renderer assertions

Test:

- custom controls creation and selection;
- opacity/blending widget propagation;
- controls removal and event disconnection;
- repeated registration isolation;
- repeated layer add/remove;
- viewer close with an active transcript layer.

### Real OpenGL rendering tests

At least one supported local or CI environment must exercise a real canvas.
Test or manually accept:

- known point/image pixel alignment;
- tile-origin translation;
- non-unit scale and affine/shear;
- transform changes after upload;
- opacity, blending, and layer ordering;
- viewport event behavior during interactive pan and zoom;
- pending-tile pre-upload/readiness behavior;
- repeated add/remove resource release;
- coordinate-upload counters during style and transform changes.

Headless tests that mock the OpenGL context do not replace this acceptance run.

## Diagnostics required before production integration

The development backend should expose inexpensive counters for:

- current viewport generation;
- active and pending snapshot identity;
- resident tile and point count;
- resident GPU bytes;
- coordinate upload count by residency key;
- queued upload count and bytes;
- last upload-pump duration;
- stale completion count;
- eviction count;
- failed snapshot or tile state;
- closed state.

These are development diagnostics, not necessarily permanent public UI.

## Known risks and required decisions

### Private napari APIs

The layer, visual, controls, and draw-callback integration all depend on private
napari 0.7.1 behavior. Isolation, focused compatibility tests, and clear
fallback are mandatory. Redesign for a later napari version is separate work.

### Viewport scale under affine transforms

A scalar inverse camera zoom is not automatically equivalent to intrinsic data
units per screen pixel when the layer transform is anisotropic, rotated, or
sheared. N4 must either provide a conservative projected scale or explicitly
carry enough transform information for the planner.

### Deferred VisPy uploads

Creating or updating a VisPy buffer does not by itself prove that GPU commands
have executed. R2 must define and test the readiness boundary used for atomic
activation.

### Dense translucent points

Ordinary alpha blending can appear additive where many markers overlap. N3
must compare supported blending modes on a dense synthetic patch and document
the default.

### Scene-node count

One visual per tile is the first measured strategy, not a permanent format
requirement. Pooling or multi-draw is introduced only if real viewport traces
show that node/draw-call overhead is material.

### Failure during an atomic transition

A corrupt or failed core tile must move the pending snapshot to an explicit
failed state, retain the previous active snapshot when possible, and report the
error. It must not wait indefinitely.

## Architecture invariants

The following invariants apply from the first spike onward:

- resident tiles are rendering state, never `Layer.data`;
- layer extent describes the complete dataset;
- cache coordinates remain in native source space;
- the napari layer transform is applied exactly once;
- axis ordering is explicit at every boundary;
- camera changes do not replace a monolithic Points array;
- transform and style changes do not upload coordinates;
- an unchanged residency key uploads at most once before eviction;
- readiness cannot be satisfied by another cache generation or selection;
- worker threads never touch VisPy objects;
- all VisPy mutation occurs on the GUI thread;
- upload-induced draws do not create planning loops;
- layer removal disconnects viewport events and closes backend resources;
- late callbacks cannot resurrect a removed layer;
- unsupported tiled rendering never disables the direct Points path.

## Immediate next actions

1. Review and approve the N0 contracts in this document.
2. Reconcile the full residency key with the backend protocol in the main
   roadmap.
3. Implement N1 and N2 with non-GL and Qt-focused tests.
4. Implement N3 using the recognizable one-tile fixture and a reference image.
5. Complete the real-canvas alignment and cleanup acceptance run.
6. Implement N4 and N5, then hold Gate N.
7. Continue persistent cache construction independently throughout N0-N5.
8. Begin multi-tile backend work only after Gate N and the fake-backend runtime
   contract agree on identity and readiness semantics.
