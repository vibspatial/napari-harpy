# Napari Integration for the Zarr Multiscale Transcript Cache

**Date:** 2026-08-21

**Status:** proposed implementation roadmap

**Target environment:** napari 0.7.1, VisPy 0.16.2, Qt through `qtpy`

## Authority and relationship to earlier roadmaps

This document is the implementation roadmap for integrating the adopted
Zarr-backed transcript cache with napari and then replacing napari-harpy's
current materialized `Points` workflow.

It supersedes
`Roadmap/transcripts_visualization/napari_integration_10_8_26.md` where the two
documents differ. That earlier document was written before the production
cache reader existed and deliberately planned a cache-free rendering spike.
Its findings about napari's private registration boundary, viewport callback,
coordinate transforms, GUI-thread ownership, and cleanup remain useful.

`Roadmap/transcripts_visualization/cache_construction_zarr.md` remains
authoritative for the cache format, construction, validation, publication,
reader semantics, and recorded Xenium measurements. This roadmap does not
introduce another cache backend, migrate cache data, or reopen the decision to
adopt Zarr.

`Roadmap/transcripts_visualization/multi_tile_cache_29_7_26.md` remains useful
as architectural history. Its Parquet-specific runtime assumptions and its
older selected-value coverage policy are not part of this implementation.

## Executive decision

Implement transcript visualization through a generic, read-only tiled-points
napari layer with a Harpy-owned VisPy renderer. The initial product consumer is
the transcript workflow, but the layer, runtime contracts, and renderer do not
encode transcript- or gene-specific behavior. Do not repeatedly replace the
data of a native napari `Points` layer as the camera moves.

The production path is:

```text
published transcripts_vis_zarr cache
        ↓
one long-lived cache reader on one worker thread
        ↓
resident catalog, selected-value, and bucket lookup indexes
        ↓
viewport and LOD plan
        ↓
byte-bounded CPU tile cache; read only missing tiles
        ↓
immutable render snapshot
        ↓ Qt queued delivery
tile-retaining VisPy renderer on the GUI/OpenGL thread
```

The final napari-harpy points workflow uses this path instead of scanning the
source dataframe, sampling it into one in-memory coordinate array, and
constructing a replacement `napari.layers.Points` object for every value
selection.

This decision follows from the native napari 0.7.1 points renderer. Its
`VispyPointsLayer._on_data_change()` reads the complete current points view and
calls one marker `set_data()` operation. Replacing `Points.data` for every
viewport would therefore rebuild napari's point view state and replace the
complete visible GPU payload. It would also make the layer extent depend on the
currently resident rows unless additional workarounds were added. That is not
the correct ownership model for independently reusable cache tiles.

## Current evidence and its implications

The retained Xenium evaluation contains 136,578,750 source points and nine
serialized levels. The published cache is approximately 1.69 GB. The relevant
reader observations are:

| Operation | Current observation |
|---|---:|
| Enter reader and load compact catalog indexes | 87.4 ms; 821,488 resident bytes |
| Load selected-value index for 1 value | 28.6 ms; 0.23 MiB retained |
| Load selected-value index for 10 values | 43.7 ms; 2.11 MiB retained |
| Load selected-value index for 100 values | 160.7 ms; 19.95 MiB retained |
| Repeated selected LOD planning for 1 value | 0.34--0.67 ms |
| Repeated selected LOD planning for 10 values | 1.00--1.44 ms |
| Repeated selected LOD planning for 100 values | 7.72--9.89 ms |
| Dense Exact tile, 108,598 points | 44.3 ms first; 11.1 ms repeated |
| Four Exact tiles, 360,291 points | 94.0 ms |
| Nine Exact tiles, 804,187 points | 190.0 ms |
| Full-extent common-value smoke request | L4; 78,789 points in 127 tiles |

The complete bucket lookup index is approximately 596 MB for this cache:

| Level | Approximate lookup bytes |
|---:|---:|
| Exact, L0 | 296 MB |
| Bridge, L1 | 185 MB |
| Spatial, L2 | 75 MB |
| All nine levels | 596 MB |

These measurements are engineering evidence, not permanent pass/fail timing
thresholds. They lead to the following runtime decisions:

- reader entry, bucket-index loading, selected-index loading, LOD planning, and
  point reads never execute on the Qt GUI thread;
- selected-value catalog IO is paid once when the selected values change, not
  on every pan or zoom;
- bucket lookup arrays are explicitly primed before interaction and remain
  resident for the cache generation;
- the initial integration supports eager loading of all bucket lookup indexes;
  a configured metadata budget can guard their complete projected size, while
  an explicit `None` permits loading without that product-level limit;
- an application CPU tile cache is needed so a warm pan does not reread point
  payloads merely because `read_viewport()` was called again;
- one logical tile is a useful reuse unit, but 127 visible tiles are enough that
  VisPy scene-node and draw-call overhead must be measured;
- an over-budget terminal level is evidence that the request cannot safely be
  executed, not permission to read it anyway.

## Goals

- Display transcript-scale SpatialData points without materializing the full
  source dataframe.
- Select LOD from the cache catalog for every material viewport change.
- Reuse selected-value and bucket lookup indexes across viewport changes.
- Read and upload only missing tile payloads during nearby pans.
- Keep complete dataset extent independent of resident tiles.
- Keep point coordinates tile-local until the scene transform is applied.
- Preserve stable value colors across selections and LODs.
- Keep all disk and codec work away from the GUI thread.
- Keep all napari model, Qt widget, VisPy node, and OpenGL mutation on the GUI
  thread.
- Make stale asynchronous results harmless.
- Replace the current Dask-to-NumPy-to-native-Points transcript workflow after
  the cache-backed layer passes its acceptance gate.
- Fail clearly when the cache, napari version, renderer, or memory budget is
  unsupported.

## Non-goals of the adopted integration

- Supporting the removed tiled-Parquet backend.
- Automatically falling back to the old materialized-points workflow.
- Editing, adding, or deleting individual points in the tiled-points layer.
- Point picking or hover metadata in the first integrated release. Display
  payloads deliberately omit `point_id`.
- Arbitrary 3D transcript rendering. The first supported contract is a 2D
  points element in a 2D napari display.
- Query-time random thinning when no serialized level fits the runtime budget.
- Adding an x/y index inside a logical tile. The camera clips complete tile
  payloads during rendering.
- Rebuilding or exhaustively validating the cache during an interactive read.
- Concurrent reads across buckets. The adopted reader batches requests within
  a bucket and lets Zarr own chunk concurrency.
- Remote/object-store caches in the first integration.
- Supporting several value-column cache variants under one points element in
  the first product workflow.

## Napari and VisPy API facts

The installed environment is the initial compatibility target:

```text
napari 0.7.1
vispy 0.16.2
qtpy 2.4.3
```

The integration relies on these facts from that environment:

1. `Layer._update_draw(scale_factor, corner_pixels_displayed,
   shape_threshold)` receives the current world-coordinate canvas bounds on
   every draw. It is the correct viewport bridge for ordinary 2D pan, zoom,
   resize, and transform changes in napari 0.7.1.
2. `Layer.world_to_data()` applies the inverse layer transform. All four world
   viewport corners must be transformed before taking the intrinsic AABB.
3. A custom layer must implement napari's base layer and slicing-state abstract
   contracts even though transcript slicing itself is a no-op.
4. napari maps layer classes to VisPy layers through
   `napari._vispy.utils.visual.layer_to_visual`.
5. napari maps layer classes to Qt controls through
   `napari._qt.layer_controls.qt_layer_controls_container.layer_to_controls`.
6. `VispyBaseLayer` owns normal visibility, opacity, blending, ordering, and
   napari layer-transform propagation.
7. napari invokes `VispyBaseLayer.close()` when a layer is removed.
8. VisPy scene visuals and buffers must be created and mutated on the
   GUI/OpenGL thread.

Items 1, 4, 5, and the `VispyBaseLayer` subclass boundary are private napari
APIs. There is no stable plugin manifest contribution for this renderer in the
target version. Keep every such import and registry mutation inside one small
compatibility package. The rest of the runtime must be testable without those
private imports.

The initial feature check targets exactly napari 0.7.1 and VisPy 0.16.2. The
project's broader napari dependency does not imply that this private renderer
works on every admitted napari version. Supporting another version requires a
compatibility test and an intentional extension of the accepted range.

Primary references used for this design are napari's
[rendering guide](https://napari.org/stable/guides/rendering.html),
[threading guide](https://napari.org/stable/guides/threading.html),
[layer API](https://napari.org/stable/api/napari.layers.html), and VisPy's
[visual API](https://vispy.org/api/vispy.visuals.html). The exact 0.7.1 behavior
must be checked against the installed source and the
[napari v0.7.1 source tree](https://github.com/napari/napari/tree/v0.7.1/napari)
rather than inferred from newer documentation. In particular, keep compatibility
tests anchored to the 0.7.1
[`Layer._update_draw()` implementation](https://github.com/napari/napari/blob/v0.7.1/napari/layers/base/base.py),
[visual registry](https://github.com/napari/napari/blob/v0.7.1/napari/_vispy/utils/visual.py),
[controls registry](https://github.com/napari/napari/blob/v0.7.1/napari/_qt/layer_controls/qt_layer_controls_container.py),
and [native points renderer](https://github.com/napari/napari/blob/v0.7.1/napari/_vispy/layers/points.py).

## Target package structure

Keep the cache implementation independent of napari, and avoid another large,
flat viewer module:

```text
src/napari_harpy/
  core/
    multi_scale_cache_points_zarr/
      reader.py                         # storage-neutral cache reader

  viewer/
    tiled_points/
      __init__.py
      contracts.py                      # immutable runtime values only
      runtime/
        __init__.py
        cache_session.py                # worker-owned reader lifecycle
        coordinator.py                  # generations and latest-request policy
        residency.py                    # decoded CPU tile LRU
      napari/
        __init__.py
        layer.py                         # TiledPointsLayerModel
        registration.py                  # private napari compatibility boundary
        controls.py                      # minimal layer controls
        viewport.py                      # draw callback to intrinsic viewport
      vispy/
        __init__.py
        layer.py                         # VispyTiledPointsLayer
        visuals.py                       # tiled-points tile visual
        residency.py                     # tile GPU residency and eviction

tests/
  viewer/
    tiled_points/
      runtime/
      napari/
      vispy/
```

The exact filenames may be adjusted to match code size, but the `runtime`,
`napari`, and `vispy` ownership boundaries should remain visible.

## Runtime contracts

### Cache dataset information

The viewer must not reach through `_PointsCacheReader._attributes`. Add one
small immutable integration-facing value exposing only:

```text
cache generation ID
points element name and source value column
canonical value names in value_id order
intrinsic x/y bounds and aligned origin
serialized level summaries
construction overview budget
```

This is enough to create the logical layer, populate the value UI, calculate
tile origins, report LOD state, and key all resident data. It does not expose
Zarr arrays, bucket descriptors, or writer models.

Opening this information performs the reader's existing strict root/layout
checks. It does not run staged exhaustive validation and does not scan the
source point contents. A cache with `publication_state="complete"` is trusted
for interactive use because independent validation was part of construction.

### Value selection identity

Normalize a selected value set once against canonical cache IDs:

```text
all values      -> one explicit ALL_VALUES identity
selected values -> sorted unique uint32 IDs plus an immutable selection key
```

The full identity, not only the logical tile, participates in CPU and GPU
residency:

```text
TileResidencyKey = (
    cache_generation_id,
    requested_value_ids,
    level,
    tile_x,
    tile_y,
)
```

Using the full selected-ID tuple is acceptable for the initial key. If a digest
is introduced later, equality must retain enough canonical evidence that a hash
collision cannot make one selection display another selection's payload.

### Viewport state

The GUI-side viewport bridge emits an immutable value shaped as:

```text
displayed axes
intrinsic half-open x_min, y_min, x_max, y_max
canvas width and height in physical or consistently defined logical pixels
device-pixel ratio where needed
world-units-per-canvas-pixel evidence
```

Only the intrinsic rectangle and effective point budget are passed to the
cache reader. Canvas and scale evidence remain viewer policy, not cache-reader
state.

Viewport states are normalized and deduplicated. A draw caused by uploading a
new snapshot must not recursively schedule the same request.

### Effective runtime point budget

Keep two policies separate:

- `hard_render_point_budget`: the user's absolute maximum visible payload;
- `screen_density_budget`: a canvas-derived target that may be lower when a
  large overview would plot many markers onto the same pixels.

The cache reader receives only:

```text
effective_point_budget = min(
    hard_render_point_budget,
    screen_density_budget,
)
```

For constant screen-space markers, calculate the density budget from canvas
pixel area and an explicit target pixels-per-point value. The target should be
calibrated together with the adopted point diameter and HiDPI behavior in the
real-canvas gate. It is viewer configuration, not a cache schema value.

Do not assume that the terminal 100,000-point construction overview fits this
runtime budget. When `_LevelSelection.within_budget` is false:

- do not call a point-payload read;
- retain the last valid snapshot if one exists;
- otherwise show an empty tiled-points visual with the complete logical extent;
- report that the user must zoom in or raise the hard budget.

An explicit future “render anyway” action may override this, but automatic
viewport handling must not.

### Planned viewport and subset reads

The existing `read_viewport()` operation is correct for one complete request,
but it returns every positive visible tile. Calling it on every pan would reread
overlapping tiles before the renderer could reuse their VBOs.

Introduce a narrow viewer-independent reader seam:

```text
select_level(viewport, budget, value_index)
        ↓
plan_viewport(level, viewport, value_index)
        ↓
ordered positive tile requests and required bucket keys
        ↓
CPU residency lookup
        ↓
read_planned_tiles(plan, tile_keys_to_read)
```

The plan is immutable and bound to the reader's cache generation and selected
value index. It contains sufficient internal request information to read a
subset without repeating catalog discovery. It exposes stable logical tile
keys and required bucket keys but does not expose writable bucket arrays.

`read_planned_tiles()` preserves the adopted physical behavior:

- group missing requests by `(level, bucket_id)`;
- issue one coordinated Zarr selection per display array for all missing tiles
  in a bucket;
- let Zarr own chunk concurrency;
- split the returned arrays back into logical immutable tile payloads;
- return tiles in plan order;
- never read `point_id` for visualization.

`read_viewport()` remains a convenience and acceptance API. The napari session
uses the plan/subset path.

### Render snapshot

One immutable render snapshot contains:

```text
cache generation
viewport/request generation
selection identity
selected level and LOD kind
within-budget evidence
omitted value IDs
ordered core tile residency keys
new CPU tile payloads, if any
```

It never mixes levels, selections, or cache generations. A zero-tile snapshot
is a valid atomic result that clears the previous visual.

## Coordinate and transform contract

The cache is constructed from the intrinsic `x` and `y` columns stored in the
SpatialData points element at `points/<points_name>/points.parquet`. Those
source-native coordinates are tiled without applying any SpatialData transform;
the cache stores each position relative to its intrinsic logical tile origin.

Coordinate ordering is explicit at every boundary:

```text
cache tile payload location      (x, y), float32, tile-local
cache viewport                   x_min, y_min, x_max, y_max
napari layer data axes           (y, x)
SpatialData affine axes          input/output (y, x)
VisPy positions and translation  (x, y)
```

The complete transform is:

```text
tile-local VisPy (x, y)
        + intrinsic tile origin (x, y)
        ↓
dataset-native coordinates
        ↓
one napari layer affine declared in (y, x)
        ↓
world coordinates
        ↓
camera/canvas coordinates
```

Do not bake the SpatialData transform into cached coordinates or returned tile
payloads. Reuse the current affine conversion:

```python
transform.to_affine_matrix(
    input_axes=("y", "x"),
    output_axes=("y", "x"),
)
```

For viewport conversion:

1. receive the world-coordinate bounds from `Layer._update_draw()`;
2. construct all four world corners;
3. inverse-transform every corner through the layer transform;
4. take a conservative floating-point data-coordinate AABB;
5. convert napari `(y, x)` bounds to the reader's `(x, y)` viewport;
6. let the reader clip against observed cache geometry.

Transforming only top-left and bottom-right is incorrect under rotation or
shear. Rounding to integer data coordinates before planning is also incorrect
for point data.

The VisPy renderer keeps positions tile-local for float32 precision. It applies
the intrinsic tile origin at the tile visual, and `VispyBaseLayer` applies the
napari layer transform at the root exactly once. Transform changes update scene
transforms; they do not reread cache payloads or reupload coordinate buffers.

## Threading, scheduling, and cancellation

### One long-lived reader thread

Create one dedicated, serial worker for each open tiled-points layer/session. The
worker creates, enters, uses, and closes `_PointsCacheReader` on that same
thread.

Do not decorate every operation with an independent `thread_worker`: a pool may
run successive calls on different threads and would obscure reader-handle
ownership. A dedicated `QThread` worker or a single-worker executor with a Qt
signal bridge is acceptable if tests prove:

- all reader calls use one worker thread;
- results arrive on the GUI thread;
- close runs on the reader thread after any active call finishes;
- no worker callback touches a napari layer, Qt control, VisPy node, or GL
  buffer directly.

### Latest-request mailbox

Camera interaction must not enqueue an unbounded FIFO of obsolete viewports.
The coordinator owns:

```text
at most one active worker request
at most one pending request, always replaced by the newest
one monotonically increasing request generation
```

Cancellation is cooperative. A synchronous Zarr selection already in progress
may finish. Its result may populate the CPU LRU if its residency identity is
still valid, but it must not activate a stale render snapshot.

A zero-delay Qt timer may coalesce duplicate draw callbacks within one event
loop turn. Do not add an arbitrary long pan/zoom debounce before measuring it;
the mailbox and tile-set deduplication already prevent request backlogs.

### Request identity and deduplication

Before point IO, compare the planned identity with the active or pending plan:

```text
cache generation
selection identity
level
ordered positive logical tile keys
```

If these are unchanged, a camera draw needs no cache read and no GPU upload.
This is especially important while the camera moves inside the same set of
logical tiles.

### CPU tile cache

The worker session owns a byte-bounded LRU of immutable display payloads keyed
by `TileResidencyKey`. Account at least:

```text
location.nbytes + value_id.nbytes
```

Active and pending snapshot tiles are protected during assembly. A payload
larger than the configured cache budget may be returned transiently but must
not make byte accounting dishonest. Selection changes naturally use different
keys. The cache does not contain VisPy objects.

The reader's filesystem/codec cache and the Harpy CPU tile cache solve
different problems. The latter is what guarantees that a warm overlapping pan
does not call point-array IO for tiles already decoded by the application.

## Reader initialization and selected-value lifecycle

### Session startup

Session startup runs in this order:

```text
enter trusted completed reader
        ↓
publish small cache dataset information to GUI
        ↓
create empty logical layer with complete extent
        ↓
project all bucket lookup-index bytes
        ↓
configured metadata budget is present and projection exceeds it?
        ↓
load all bucket lookup indexes on worker thread
        ↓
mark viewport reads READY
```

For the evaluated Xenium cache this deliberately retains approximately 596 MB
of bucket tile/range metadata while leaving all point payload arrays on disk.
This is the default product policy because it moves bucket lookup IO out of
every future pan, zoom, LOD, and value selection.

The metadata budget is explicit configuration and its complete projection is
checked before arrays are read. If it does not fit, the first integration fails
with an actionable message rather than loading an unbounded subset silently.
Plan-driven incremental lookup residency may be added later, but it also needs
eviction semantics that the current reader intentionally does not expose.

Measure all-index startup time and peak RSS on the retained Xenium cache in the
acceptance slice. Keep the empty layer and status UI responsive while priming.

### Value selection change

The cache root supplies canonical value names without scanning the source
dataframe. A value selection change performs:

```text
selected labels
        ↓ map once through canonical cache vocabulary
sorted unique uint32 value IDs
        ↓
load_selected_value_index() once on reader thread
        ↓
immutable selected-value index retained in session
        ↓
new selection generation and immediate viewport replan
```

Selecting all canonical values uses the reader's `None` all-values path and
does not construct a selected index.

The current measurements show why this boundary matters: a 100-value index can
take approximately 161 ms to load, whereas repeated indexed planning performs
no catalog Zarr reads and was approximately 8--10 ms for the measured
full-extent cases. The old snapshot remains visible while a new selection index
and snapshot are prepared.

When a sampled level omits requested values, the level remains eligible. The
layer status reports `omitted_value_ids`; it must not present absence at that
LOD as biological absence. This preserves the budget-first cache policy and
prevents one rare gene from forcing a large multi-value request back to Exact.

## Napari layer model

Implement a dedicated `TiledPointsLayerModel(Layer)` with these semantics:

- fixed `ndim=2`;
- read-only, pan/zoom interaction only;
- `data` is a small logical dataset reference, never an `N x 2` point array;
- `_extent_data` is the complete observed cache extent in napari `(y, x)`
  order;
- extent never changes when snapshots, selections, or LODs change;
- no-op layer slicing state suitable for a 2D logical layer;
- `_get_value()` returns `None` until picking is designed;
- a deterministic placeholder thumbnail;
- explicit unsupported behavior for standard layer-data serialization;
- custom events for normalized viewport state, render snapshot, point style,
  and session status;
- no ownership of the cache reader or worker thread.

`TiledPointsLayerModel._update_draw()` is the narrow viewport bridge. It calls
`super()._update_draw()` first, constructs the normalized intrinsic viewport,
and emits only when the state materially changed. It never performs LOD
planning, disk IO, or renderer mutation.

Gene colour has three distinct owners. The napari-harpy points controller owns
the stable colour assignment for the canonical cache vocabulary. The layer
model owns the resulting immutable presentation state as a dense
`value_palette`, aligned so row `value_id` contains that value's RGBA colour,
and emits a palette-change event. `VispyTiledPointsLayer` owns the corresponding
GPU lookup resource. Point payloads and VBOs retain only `value_id`; they never
duplicate per-point RGBA rows. A palette change therefore performs no Zarr
point read and does not reupload point positions.

For the initial implementation, use a complete `(G, 4)` `uint8` palette, where
`G` is the canonical cache vocabulary size. At 5,122 values this occupies about
20 KiB and preserves colour identity as values enter or leave a selection. I2
deliberately does not add this model property before a renderer can consume it.
I6 adds `TiledPointsLayerModel.value_palette`, its event, validation, and the GPU
lookup together; I8 supplies the stable palette from the existing points-panel
colour policy.

The layer's private napari abstract-method implementation should follow the
smallest behavior supported by napari 0.7.1. Do not inherit from `Points` merely
to avoid those methods: doing so would reintroduce point editing, feature,
slicing, and view-cache semantics that do not describe a logical tiled layer.

## Private napari registration and controls

Registration is explicit and occurs before adding the first tiled-points layer.
Importing `napari_harpy` must not mutate global napari registries.

The registration operation:

- checks the supported napari and VisPy versions;
- feature-detects both private registries and the expected factory behavior;
- installs the model-to-visual and model-to-controls mappings;
- is idempotent when the desired mappings already exist;
- rejects a conflicting mapping;
- rolls back the first mapping if the second mapping fails;
- raises one actionable Harpy-owned compatibility exception;
- provides a test-only unregister helper for registry isolation.

The custom controls use napari's base opacity/blending controls and add only:

- point diameter in canvas pixels;
- a read-only Exact/Bridge/Spatial level indicator;
- rendered point and tile counts;
- loading/over-budget/error status;
- a sampled-LOD warning and omitted selected-value summary.

Value selection and hard render budget remain in napari-harpy's points panel,
where the source element is chosen. Do not duplicate them in layer controls.

## VisPy renderer

### Ownership

`VispyTiledPointsLayer` subclasses `VispyBaseLayer` and owns one compound root
visual plus Harpy-owned tile visuals. It relies on the base class for:

- root visibility and opacity;
- blending state;
- layer ordering;
- the napari layer transform;
- detaching the root on close.

It owns:

- tile-local `(x, y)` coordinate buffers;
- intrinsic tile-origin transforms;
- value palette and point-size state;
- active and pending snapshot membership;
- GPU byte accounting and LRU eviction;
- upload counters and cleanup.

### Tile-retaining strategy

Use one independently retained tile visual as the initial production strategy.
The renderer maps `TileResidencyKey` to a GPU resource and reuses overlapping
resources across same-selection pans. Only newly entering or evicted tiles are
uploaded.

The 127-tile full-extent common-value observation makes scene-node and draw-call
cost a required acceptance measurement. It does not justify abandoning tile
identity before measuring. If draw-call overhead is material, pool tile VBOs
into renderer-owned pages without changing the cache format or worker payload
contract. Do not group GPU residency by physical bucket: buckets are an IO
layout, not a spatial rendering identity.

### Visual representation

Qualify a standard VisPy markers-per-tile implementation as the correctness
reference, then adopt a compact tiled-points visual if it passes the real-canvas
gate. The recommended compact payload is:

```text
tile-local position   float32 x 2
dense value ID        compact exact GPU-compatible representation
```

Point diameter and global opacity are uniforms. Value color is resolved through
a small palette lookup resource so a palette edit does not reupload positions.
A 2D nearest-filtered palette texture is preferable to relying on 1D texture
support across GL profiles.

The renderer receives the complete immutable `value_palette` from
`TiledPointsLayerModel` and translates it into that GPU resource. It does not
choose gene colours or mutate the palette. The controller-to-model-to-renderer
flow is:

```text
napari-harpy points controller
    stable value_id -> RGBA assignment
                ↓
TiledPointsLayerModel.value_palette
    complete immutable (G, 4) uint8 presentation state
                ↓ palette-change event
VispyTiledPointsLayer
    small GPU lookup texture
                ↓
tile VBO value_id -> displayed RGBA
```

If value IDs are converted to float32 attributes, enforce the exact-integer
range explicitly. The evaluated 5,122-value vocabulary is safe, but the visual
must not silently accept a cache vocabulary whose IDs cannot be represented
exactly. An integer attribute path may be adopted instead if VisPy/gloo support
is verified on the supported GL environments.

Use constant canvas-pixel diameter initially. It gives points a stable
visual size across zoom levels and makes the screen-density budget meaningful.
World-space diameter may be considered later as a separate presentation mode.

### Snapshot activation

The currently active snapshot remains visible while missing resources for a
new snapshot are prepared. Activate a new snapshot only when every core tile is
available. The activation changes visibility/membership as one GUI-thread
operation; it never presents a mixture of LODs or selected-value identities.

For an ordinary same-level pan, overlapping tile visuals remain resident and
only entering tiles are added. For selection or level changes, the old snapshot
remains until the complete replacement is ready.

The visible point budget is at most approximately 100,000 in the current
policy, so begin with one complete GUI-thread snapshot application and measure
it. Do not introduce a complex upload pump solely on the assumption that
`VertexBuffer.set_data()` corresponds to immediate GPU transfer; VisPy may
defer GL work until draw. If real-canvas measurements show an interaction pause,
add a GUI-thread upload queue with an explicit measured byte/time budget and a
tested definition of readiness.

### GPU residency and eviction

Bound GPU residency by bytes, not tile count. Pin active and pending snapshot
resources. Evict only inactive resources, on the GUI/OpenGL thread. Record:

```text
resident tile count
resident point count
resident GPU bytes
coordinate uploads by TileResidencyKey
evictions
active and pending snapshot IDs
```

These counters support acceptance and diagnosis; they need not all become
permanent user-facing UI.

## End-to-end request flow

### Pan, zoom, resize, or transform

```text
napari draw on GUI thread
        ↓
TiledPointsLayerModel._update_draw()
        ↓ normalize and deduplicate
TiledPointsLayerModel.events.viewport(state)
        ↓ GUI listener submits or replaces the pending request; no cache IO
latest-request coordinator
        ↓ assign request generation
reader worker: select_level()
        ↓
within_budget=False ──→ warning result; no payload read
        ↓ true
reader worker: plan_viewport()
        ↓
same active/pending plan ──→ no work
        ↓
CPU LRU lookup
        ↓
reader worker: batch-read only missing tiles
        ↓
immutable render snapshot
        ↓ Qt queued signal
GUI generation/closed check
        ↓
VisPy pending resources and atomic activation
```

The core logical tile already extends beyond the exact viewport at its edges.
Do not add a prefetch halo in the initial integration. First measure whether
logical-tile overlap and CPU/GPU retention are sufficient. Any future halo must
participate in memory and read budgets and must not make a request silently
exceed the hard render limit.

### Value selection change

```text
napari-harpy value UI
        ↓
canonical value IDs and new selection generation
        ↓
worker loads selected-value index once
        ↓
latest viewport is replanned
        ↓
old selection remains visible until new snapshot is complete
        ↓
atomic selection replacement and omitted-value status
```

### Layer removal

```text
invalidate session and request generation on GUI thread
        ↓
disconnect layer, controls, timers, and result signals
        ↓
VisPy close releases/detaches GPU resources on GUI thread
        ↓
active worker call may finish but its result is ignored
        ↓
reader closes on its worker thread
        ↓
worker thread terminates
```

Every close operation is idempotent. A late result cannot recreate a node,
restart a timer, or mutate a removed layer.

## Napari-harpy product integration

### Cache location and trust

Use the adopted nested location:

```text
<sdata.zarr>/points/<points_name>/transcripts_vis_zarr/
```

The cache root identifies its points element, element path, x/y/value columns,
source signature, cache generation, and complete publication state. Interactive
opening validates the cache contract but does not rescan the source Parquet
contents. Expensive source/content validation belongs to cache construction.

The first product workflow supports one published transcript cache per points
element. Its stored value column is the selectable transcript category. If the
user chooses another index column, the UI must explain that the cache was built
for a different column and offer an explicit rebuild/replacement. It must not
silently interpret value IDs under another column.

The SpatialData coordinate system is a runtime layer-transform choice and does
not require another cache generation. Reuse the same cache-native coordinates
with the selected SpatialData affine.

### Replace the current direct points path

The existing workflow currently:

1. validates a points element;
2. scans it to build `PointsValueTable`;
3. filters and samples through Dask for every selection;
4. materializes `PointsValueSelection.coordinates` and a pandas feature table;
5. creates a new native `Points` layer;
6. replaces the old layer to avoid stale napari private point-view caches.

The final transcript workflow instead:

1. opens the completed cache and reads its canonical vocabulary;
2. creates one persistent `TiledPointsLayerModel`;
3. keeps one reader session for the layer lifetime;
4. changes the selected-value index and render snapshots in place;
5. never constructs a complete pandas feature table for display;
6. never replaces the layer on camera or selection changes.

Add a distinct `TiledPointsLayerBinding`; do not make
`PointsLayerBinding` pretend that a custom tiled-points layer is a native
`Points` layer. Reuse the source identity, SpatialData transform helper,
selected-value UI, stable value colors, and render-budget input where their
semantics still match.

After the cache-backed acceptance gate, remove the direct transcript loading
jobs and adapter path from the points panel. Native napari `Points` remains
available for other features that genuinely own a materialized point array,
including any shapes-as-points behavior. There is no hidden old-backend
fallback for transcript-scale SpatialData points.

### Cache absent, incompatible, or failed

The points panel distinguishes:

```text
CACHE_ABSENT
CACHE_INCOMPATIBLE
CACHE_READY
CACHE_OPENING
CACHE_BUILDING
VIEW_READY
VIEW_LOADING
VIEW_OVER_BUDGET
VIEW_FAILED
```

When absent or incompatible, offer Build/Rebuild and do not silently run the
old full-source materialization. Cache creation runs off the GUI thread using
the guarded builder and its publication lock. The UI reports indeterminate or
structured progress without treating an incomplete staging generation as
readable.

Do not rebuild a cache while its reader session is active unless publication
and reader quiescence have an explicit protocol. The initial safe workflow
closes the tiled-points layer/session before replacement, builds and publishes,
then opens the new generation.

If cache construction is cancelled or fails, the guarded builder leaves the
previous completed generation intact. If viewport IO fails, retain the last
valid snapshot where possible and expose a recoverable error; do not activate a
partial snapshot.

## Implementation slices

Each slice is production-quality, independently reviewable, and covered by
focused tests. A slice may be merged while hidden behind an internal entrypoint,
but the existing points workflow is not removed until Gate I and the product
migration slice pass.

### Slice I0: freeze integration contracts and compatibility — resolved

Deliverables:

- make this document authoritative over the earlier integration roadmap;
- freeze 2D-only, read-only transcript-layer scope;
- freeze all coordinate orders and the full tile residency key;
- freeze budget-first handling and the no-read over-budget rule;
- freeze napari 0.7.1 and VisPy 0.16.2 as the initial renderer target;
- define the cache dataset information, viewport, selection, plan, tile, and
  render-snapshot contracts;
- document that normal reading trusts completed construction and never runs
  exhaustive validation.

Exit criteria:

- contracts contain no Qt, napari, VisPy, Zarr array, or mutable writer object;
- no contract confuses physical bucket identity with logical tile identity;
- a residency key cannot alias cache generations or value selections;
- the screen-density budget is explicitly viewer-owned;
- later slices do not need to infer coordinate ordering.

### Slice I1: add the viewer-oriented cache planning seam — resolved

The current `read_viewport()` operation discovers the positive logical tiles
and immediately reads every corresponding point payload. That is correct for a
complete standalone viewport request, but it gives a later CPU residency cache
no opportunity to remove tiles that are already decoded.

I1 separates discovery from physical point reads:

```text
current read_viewport()
    discover positive tiles
    + immediately read every tile

I1
    plan_viewport()
        -> immutable ordered tile plan
        -> no point-payload IO

    application chooses the tile keys to read

    read_planned_tiles(plan, tile_keys_to_read)
        -> validate the subset against the plan
        -> group missing tiles by physical bucket
        -> one coordinated read per bucket
        -> return tiles in plan order

    read_viewport()
        -> plan all
        -> read all
```

For example:

```text
planned tiles:   A B C D
resident tiles:  A   C
physical reads:    B   D
```

The immutable plan contains the ordered logical tile identity and the internal
read instructions needed for all-values or selected-value access. It exposes
stable logical tile and required bucket keys, but callers do not construct or
modify manifest rows, bucket descriptors, or sparse-range records. The plan is
bound to the cache generation and records the complete requested value IDs once.
Each private tile instruction retains only the requested value IDs applicable to
that tile. The public subset keys are therefore purely logical
`(level, tile_x, tile_y)` identities; supplying a subset cannot alter the value
selection already frozen in the plan. The private instruction stores those
three coordinates directly and exposes the tuple through a property; there is
no additional logical-key wrapper object.

The later viewer runtime combines the plan's cache generation and complete
value-selection identity with each logical tile key when constructing the CPU
and GPU `TileResidencyKey`. Keeping that policy out of the core reader avoids
coupling every core tile instruction to viewer residency policy. An empty
`tile_keys_to_read` tuple is a valid no-IO read, and caller key order never
changes the plan-order result.

Subset reads reuse the existing one-batch-per-bucket implementation; they must
not loop through the singleton `read_tile()` convenience API. Physical bucket
execution may differ from logical tile order, so results are restored to plan
order before they are returned.

I1 provides this reader capability but does not implement the CPU LRU or decide
which tiles are resident. That policy belongs to I5. Keeping the boundary here
makes I1 independently testable without napari, Qt, workers, or a renderer.

Deliverables:

- expose immutable cache dataset information from the reader;
- implement catalog-only `plan_viewport()`;
- expose ordered logical tile keys and required bucket keys on the plan;
- implement a generation-bound batch read for a requested plan subset;
- preserve one coordinated read per bucket and exact Zarr row selection;
- keep existing `read_viewport()` as the complete-request convenience API;
- add focused reader tests without napari or Qt.

Exit criteria:

- planning performs no point-payload IO;
- a subset read cannot request a tile absent from its plan;
- adjacent missing tiles in one bucket use one bucket batch;
- returned tile order follows the plan, not physical bucket iteration;
- visualization never accesses `point_id`;
- the old complete viewport API remains correct;
- one warm-pan test proves already supplied tiles can be omitted from the next
  physical read.

### Slice I2: implement the custom napari layer boundary — resolved

I2 creates the empty but fully functional napari shell that the later cache
runtime and renderer will drive. It does not yet read Zarr point payloads or
render tiled points. The change from the current native-`Points` boundary is:

```text
current transcript display
    materialized N x 2 coordinates
        -> napari Points.data
        -> native Points renderer

I2 boundary
    small immutable cache dataset description + layer affine
        -> TiledPointsLayerModel.data
        -> complete stable dataset extent
        -> registered empty tiled-points visual and custom controls
```

`TiledPointsLayerModel` subclasses napari's base `Layer` directly. Its `data`
property is a small logical dataset reference derived from the immutable cache
dataset information exposed in I1; it is never a resident coordinate array.
The model reports the complete cache extent in napari `(y, x)` data order, and
napari applies the normal layer transform to obtain the world extent. Fit to
view and other layer-list operations therefore remain correct before any tile
is loaded and remain unchanged as the viewport, selection, LOD, or resident
snapshot changes.

The model implements only the smallest napari 0.7.1 base-layer behavior needed
for a fixed two-dimensional logical layer: no-op slicing, no picking, a
deterministic placeholder thumbnail, and explicit unsupported standard
layer-data serialization. It does not inherit from `Points` and consequently
does not acquire point editing, feature-table, selection, slicing, or native
point-view-cache semantics.

I2 also establishes the napari lifecycle boundary needed to add the model to a
real viewer. Explicit registration, performed before inserting the first
tiled-points layer, maps `TiledPointsLayerModel` to:

- a thin `VispyBaseLayer` adapter that supports ordinary visibility, opacity,
  blending, transforms, ordering, and close, but does not yet own point
  buffers; and
- minimal tiled-points layer controls built on napari's normal base controls.

The same visual boundary is completed into the tile-retaining renderer in I6.
Registration is never an import side effect: it checks the supported napari and
VisPy versions, is idempotent for the desired mappings, rejects conflicts, and
rolls back the first private-registry mutation if the second fails.
`register_tiled_points_layer()` deliberately has no application call site at
the end of I2; only its focused tests exercise it at this stage. I8 becomes its
first production consumer and must call it before constructing or inserting the
first `TiledPointsLayerModel`. This dormant interval is an intentional slice
boundary, not evidence that the registration function is dead code.

The slice boundary is deliberately narrow:

```text
I2  logical napari layer, empty visual lifecycle, and controls
I3  canvas viewport -> intrinsic cache viewport
I4  worker-owned cache reader and resident lookup indexes
I5  scheduling, level planning, and CPU tile residency
I6  tile-retaining VisPy point renderer
I7  complete cache-to-canvas session
I8  replacement of the current napari-harpy Points workflow
```

Consequently, I2 does not call `plan_viewport()`, open a cache reader, select an
LOD, read a bucket, upload a point buffer, or replace the existing materialized
`PointsLoadRequest` path. Its acceptance result is an empty logical tiled-points
layer that napari can manage correctly without any point coordinates in
`Layer.data`.

Deliverables:

- implement `TiledPointsLayerModel` and its minimal no-op slicing state;
- report complete data/world extent independently of resident points;
- implement explicit private visual/control registration;
- implement minimal custom controls;
- add compatibility and rollback errors;
- add model-only and Qt-focused tests.

Exit criteria:

- an empty logical tiled-points layer can be added, selected, transformed,
  hidden, fit to view, and removed;
- no coordinate array is stored in `Layer.data`;
- the correct visual and controls are selected automatically;
- repeated registration is idempotent and conflicts fail atomically;
- unsupported versions fail before a layer is inserted;
- the layer cannot enter an edit mode.

### Slice I3: implement viewport conversion and effective budgets — resolved

I3 is the GUI-only adapter between the napari layer created in I2 and the
reader-planning seam created in I1. At the start of I3, the two existing ends
are deliberately not connected yet:

```text
napari draw state
    TiledPointsLayerModel.events.viewport exists, but nothing emits it

cache-reader planning
    select_level(_IntrinsicViewport, point_budget)
    plan_viewport(_IntrinsicViewport, ...)
```

I3 fills that gap by adding an immutable `TiledPointsViewportState` and a narrow
`TiledPointsLayerModel._update_draw()` override. The override first calls
`super()._update_draw(...)` so normal napari scale and corner bookkeeping still
runs. Napari's base implementation calculates a data-space bounding box but
rounds it to integer array coordinates; I3 must independently preserve the
floating-point bounds required by point data.

I3 only defines and emits `TiledPointsLayerModel.events.viewport`; it does not
install a production listener. I5 implements the latest-request coordinator,
and I7 connects this event to that coordinator during session composition. The
event is emitted synchronously on the GUI thread, so its listener may only
submit or replace pending work. It must never perform Zarr or codec work
directly.

Napari supplies the top-left and bottom-right viewbox corners in world `(y, x)`
coordinates. Form all four world corners, inverse-transform each through
`Layer.world_to_data()`, take the enclosing floating-point data-coordinate AABB,
and only then convert `(y, x)` to the reader's half-open `(x, y)` viewport.
Transforming only two corners is invalid under rotation or shear. Do not round,
clip, or otherwise quantize these bounds in the layer; cache-geometry clipping
remains the reader's responsibility.

Freeze the immutable state as:

```text
TiledPointsViewportState
    displayed_axes: tuple[int, int]
    x_min, y_min, x_max, y_max: float
    canvas_width, canvas_height: int
    hard_render_point_budget: int
    screen_density_budget: int
    effective_point_budget: int  # derived property: min(hard, screen density)
```

The two budget inputs remain stored in the state as diagnostic and test
evidence; `effective_point_budget` is a property derived from their minimum and
cannot become inconsistent with them. Later reader code receives only the
intrinsic rectangle and that effective budget. Treat `shape_threshold` as the
consistently defined logical viewbox-pixel dimensions supplied by napari 0.7.1.
The marker renderer and real-canvas gate must use the same logical-pixel
convention and explicitly qualify HiDPI behavior; introduce a device-pixel
conversion only if that gate proves it necessary.

Budgeting remains viewer policy:

```text
screen_density_budget = max(
    1,
    floor(canvas_pixel_area / target_pixels_per_point),
)

effective_point_budget = min(
    hard_render_point_budget,
    screen_density_budget,
)
```

Use the existing `100_000` points-panel default as the initial hard limit. Set
the initial configurable `target_pixels_per_point` to `9.0`, corresponding to
the approximate `3 x 3` canvas-pixel footprint of the default point diameter.
This value is an initial viewer policy rather than an acceptance threshold or
cache-format value; I6 may adjust its default after real-canvas and HiDPI
qualification. I8 supplies the validated points-panel hard limit.

Keep `target_pixels_per_point` independent of live `point_diameter` edits in the
initial integration. A diameter change remains a style-only GPU-uniform update
and must not silently select another LOD or trigger a cache read. A future
diameter-aware density policy would be a separate behavioral change.

Normalize axis order, scalar types, and canvas dimensions before comparing a
new state with the last emitted state. Use exact immutable-state equality after
normalization; do not obtain deduplication by rounding coordinates. An identical
draw emits nothing, preventing a point-buffer upload from recursively scheduling
the same viewport request. A changed camera, transform, canvas size, hard budget,
or density evidence emits one `viewport` event.

Retain the latest emitted immutable viewport state; do not introduce a second
private geometry object containing the same bounds and canvas dimensions. If
the hard budget or density configuration changes while the camera is
stationary, retain those geometry fields, recalculate the state's budget
fields, and pass the replacement state through the same deduplicating emitter.
Before the first draw there is no state to update and therefore no viewport
event.

The I3 tests connect a recorder to that event instead of creating a cache
session. I3 performs no cache open, Zarr IO, LOD selection, viewport planning,
worker scheduling, tile read, or VisPy point-buffer mutation. I4 through I7
consume the emitted state and implement those responsibilities.

Deliverables:

- bridge `Layer._update_draw()` to immutable viewport states;
- inverse-transform all four corners and preserve floating-point bounds;
- handle pan, zoom, resize, transform, and HiDPI canvas changes;
- deduplicate identical normalized states;
- implement hard-versus-screen-density budget calculation;
- recompute from retained draw geometry when budget policy changes;
- add a recorder instead of a real cache session.

Exit criteria:

- identity, translate, scale, anisotropic scale, rotation, and shear cases
  conservatively cover the visible intrinsic region;
- y/x and x/y swaps are excluded by asymmetric fixtures;
- upload-induced redraws do not emit another identical request;
- a small canvas can produce a budget below the construction overview;
- a stationary viewport emits a replacement state when its budget changes;
- changing point diameter alone does not change the effective budget;
- the callback performs no IO and remains fast on the GUI thread.

### Slice I4: implement the worker-owned cache session — resolved

I4 introduces the long-lived runtime wrapper around `_PointsCacheReader`. It
does not yet schedule viewports or read point payloads. Its responsibility is
to create, enter, use, and close one reader on one dedicated serial worker
thread while exposing only immutable information and queued lifecycle events
to the GUI thread:

```text
GUI thread
    cache-session facade
        ↓ queued commands
dedicated serial reader worker
    _PointsCacheReader
    resident bucket lookup indexes
    current selected-value index
        ↑ queued immutable results/events
GUI thread
```

Implement a GUI-thread `_TiledPointsCacheSession(QObject)` facade and a private
`_TiledPointsCacheWorker(QObject)` moved to one dedicated `QThread`. Do not
construct the reader on the GUI thread and move it afterwards. The worker
creates and enters `_PointsCacheReader`, performs every subsequent reader
operation, and calls `__exit__()` on that same thread. Independent
`thread_worker` invocations and a general thread pool do not establish this
ownership guarantee and are not part of I4.

Construction remains passive. Creating the session does not start a thread or
perform IO; an explicit `start()` transition creates/starts the worker
lifecycle. Use a small private state machine:

```text
NEW
STARTING
PRIMING
READY
LOADING_SELECTION
FAILED
CLOSING
CLOSED
```

`start()` is valid only from `NEW`. Selection requests are accepted only from
`READY`; they return to `READY` after success or recoverable selection failure.
The session does not expose its reader or accept future reader commands before
readiness. `close()` is safe and idempotent from every non-closed state.

Add immutable `_CacheSessionSettings` with two required `int | None` fields and
no implicit defaults:

```text
max_bucket_lookup_bytes
max_selected_value_index_bytes
```

I4 must not guess a machine-wide memory policy. I7/I8 will supply the adopted
product configuration. The decoded point-payload/CPU-LRU budget belongs to I5,
not these settings.

A positive integer is a preflight upper bound. `None` explicitly disables that
configured limit; it does not disable size projection, byte reporting, or
post-load reconciliation, and it does not imply that process memory is
unlimited. Propagate this `int | None` contract through the bucket-lookup and
selected-value reader APIs rather than translating `None` to an artificial
large integer.

Session startup is ordered and guarded:

```text
enter trusted completed reader
        ↓
return immutable cache dataset information
        ↓
project bytes for every bucket lookup index
        ↓
configured limit exists and projection exceeds it?
    yes → fail before loading lookup arrays
    no or no configured limit ↓
load all bucket lookup indexes with progress
        ↓
mark session READY for later viewport commands
```

The resident lookup indexes contain bucket tile/range metadata, not point
coordinates or point-level value IDs. Priming all of them is deliberate: it
moves this metadata IO out of later pan, zoom, LOD, and value-selection paths.
For the evaluated Xenium cache the expected retained lookup footprint is about
596 MB; choosing a finite metadata budget or explicitly choosing `None` is
therefore a product decision rather than an implicit allocation.

The session also owns the current selected-value index. Its command boundary
uses `None` for all values and a sorted unique nonempty `tuple[int, ...]` for a
subset. A changed subset is converted to `uint32`, loaded once through
`load_selected_value_index()` on the reader thread, and retained across
subsequent viewports. Selecting all canonical values stores `None` and uses the
reader's all-values path. An unchanged identity reuses the current index; the
session does not maintain an unbounded cache of historical selections. The
`_SelectedValueIndex` object never crosses to the GUI thread.

Expose focused Qt signals carrying only immutable lifecycle evidence:

```text
state_changed(state)
dataset_available(_CacheDatasetInfo)
lookup_progress(completed_buckets, total_buckets)
ready()
selection_ready(selection_identity, resident_bytes)
failed(phase, exception_type, message)
closed()
```

Log the original exception and traceback on the worker side; do not transport
a live traceback object as GUI state. A startup or priming failure is fatal,
closes the reader on its owning thread, and transitions through `FAILED` to
`CLOSED`. A selected-index failure is recoverable: retain the previous
selection/index, report the failure phase, and return to `READY`.

Worker-originated results cross to the GUI using queued signals; no worker
callback may mutate a napari layer, Qt control, VisPy node, or OpenGL resource.

Deliverables:

- start one long-lived serial reader worker;
- enter the reader and return cache dataset information;
- project and eagerly load all bucket lookup indexes, enforcing an explicit
  byte budget when one is configured;
- load and retain selected-value indexes when selection changes;
- close the reader on its owning thread;
- expose structured startup, ready, progress, and error events.

Non-goals for I4:

- do not connect `TiledPointsLayerModel.events.viewport`;
- do not run `select_level()` or `plan_viewport()`;
- do not read point payload arrays;
- do not implement decoded CPU tile residency or render snapshots;
- do not mutate the layer or renderer.

I5 adds viewport scheduling, payload reads, and CPU tile residency on top of
this already-open, already-primed worker session. I7 later connects the layer's
viewport event to the composed coordinator.

Shutdown is cooperative rather than falsely instantaneous. `close()` sets a
thread-safe cancellation flag and schedules reader closure on the owner
thread. Lookup priming checks the flag through its per-bucket progress callback
and may use that callback to trigger the reader's atomic rollback. A
synchronous selected-index load has no progress boundary and may finish before
closure runs; its result must not be published after closing begins. In every
case, the worker first runs `_PointsCacheReader.__exit__()` and reports its
private completion, the facade asks the `QThread` event loop to quit, and the
public `closed()` signal is emitted only after `QThread.finished` reaches the
GUI-side facade.

Exit criteria:

- injected fake-reader thread-ID tests prove construction, entry, operations,
  and exit all use the worker thread while facade callbacks use the GUI thread;
- the session cannot reach `READY` before complete lookup priming;
- an over-budget metadata projection fails before lookup arrays are loaded;
- selecting all values avoids a selected index;
- requesting an unchanged value selection reuses the retained selected index;
- a selected-index failure retains the previous ready selection;
- session shutdown during startup, priming, index loading, and idle state is
  safe and idempotent;
- focused state-machine tests use an injected fake reader, while one small real
  cache test proves the session boundary works with `_PointsCacheReader`;
- tests assert project behavior, event order, ownership, and cleanup rather
  than generic `QThread` or Qt signal internals.

### Slice I5: implement viewport scheduling and CPU tile residency

Deliverables:

- implement one-active/one-latest-pending request coordination;
- run level selection and viewport planning on the reader worker;
- reject over-budget plans before point IO;
- implement the byte-bounded CPU tile LRU;
- read only plan tiles missing from CPU residency;
- assemble immutable render snapshots;
- reject stale completions by session, selection, and request generation.

Required deterministic sequences include:

```text
same selection: view 1 = A B; view 2 = B C; view 3 = C D
selection S1 in flight; selection changes to S2
level L2 in flight; viewport changes and selects L3
```

Exit criteria:

- B is physically read at most once while resident;
- obsolete S1 or L2 results cannot activate;
- intermediate camera states do not form an unbounded queue;
- CPU bytes never exceed the configured retained budget;
- active/pending assembly is not broken by LRU eviction;
- `within_budget=False` produces zero point-array calls;
- the complete scheduler is testable with a fake renderer and no OpenGL.

### Slice I6: implement and qualify the tile-retaining VisPy renderer

Deliverables:

- implement `VispyTiledPointsLayer` and compound root;
- render recognizable synthetic local-coordinate tiles;
- implement one tile visual per `TileResidencyKey`;
- add the complete immutable `TiledPointsLayerModel.value_palette` and its
  palette-change event;
- implement GPU palette lookup, point diameter, and opacity behavior without
  rereading points or reuploading positions after palette-only changes;
- implement active/pending atomic snapshots;
- implement GPU byte accounting, pinning, and eviction;
- implement idempotent close;
- compare the standard marker reference with the compact tiled-points visual;
- run real-canvas alignment and performance tests.

Exit criteria:

- tile origins align with a reference image under identity and affine
  transforms;
- transforms do not reupload coordinates;
- same-selection pans retain overlapping VBOs;
- selection and LOD changes never show mixed snapshots;
- style-only changes do not reupload positions in the adopted visual;
- point diameter is stable under zoom and correct on HiDPI displays;
- GPU residency obeys its byte budget;
- 127 visible tile nodes have measured acceptable frame behavior, or a
  renderer-owned pooling follow-up is completed before Gate I;
- repeated add/remove releases scene nodes, callbacks, and GPU references.

### Slice I7: compose the real cache-to-canvas session

Deliverables:

- bind the real worker session and coordinator to `TiledPointsLayerModel`;
- connect `TiledPointsLayerModel.events.viewport` to the GUI-side coordinator
  callback, and disconnect it during layer/session teardown;
- keep that callback non-blocking: it only submits or replaces the latest
  request and performs no cache IO;
- deliver worker results through queued GUI-thread signals;
- connect render snapshots to the VisPy backend;
- report Exact/Bridge/Spatial, selected counts, omitted values, and errors;
- retain the prior snapshot during reads and selection changes;
- exercise the complete flow with a small real Zarr cache.

Exit criteria:

- opening the layer never scans the source dataframe;
- first view, warm pan, LOD change, selection change, empty view, over-budget
  view, and close all follow the documented flow;
- a viewport whose tile set is unchanged performs neither point IO nor upload;
- a one-tile pan reads/uploads only the entering tile;
- all VisPy mutation is observed on the GUI thread;
- no late result can mutate a removed layer.

### Gate I: cache-backed napari layer accepted

Hold this gate before replacing napari-harpy's existing points workflow.

Approval requires:

- I1 through I7 exit criteria;
- visual evidence of transcript/image alignment;
- recorded main-thread, worker-thread, and cleanup evidence;
- recorded CPU/GPU residency and warm-pan behavior;
- no automatic over-budget read;
- no complete source materialization;
- one accepted standard-marker or compact-visual strategy;
- one accepted policy for the 127-tile scene-node observation;
- no unresolved unsupported-version or registry-cleanup behavior.

### Slice I8: replace the napari-harpy points selection workflow

Deliverables:

- add `TiledPointsLayerBinding` and cache-backed controller state;
- call `register_tiled_points_layer()` before constructing or inserting the
  first `TiledPointsLayerModel`; do not register through an import side effect;
- derive the nested cache path for the selected SpatialData points element;
- populate value selection from cache `value_names`;
- reuse the SpatialData affine and existing stable value colours, construct the
  complete value-ID-aligned palette, and assign it to the persistent tiled-points
  layer;
- connect the current points panel to persistent layer selection changes;
- replace `PointsLoadRequest`/materialized selection application in the
  transcript path;
- activate and preserve one existing tiled-points layer instead of replacing it;
- update status cards for cache, LOD, sampled omission, and over-budget state.

Exit criteria:

- selecting values updates the existing tiled-points layer;
- ordinary selection and camera changes do not execute Dask;
- the layer remains at the same layer-list identity and camera state;
- changing coordinate system applies the transform exactly once;
- a value-column/cache mismatch is actionable;
- native Points behavior used by unrelated features remains unchanged.

### Slice I9: integrate cache discovery, build, and rebuild

Deliverables:

- detect absent, complete-compatible, and incompatible cache states;
- add explicit Build/Rebuild actions;
- run source validation and guarded construction away from the GUI thread;
- provide useful build status without exposing staging generations;
- coordinate layer/session closure before cache replacement;
- open the newly published generation after success;
- preserve a previous completed cache on failed construction.

Exit criteria:

- an absent cache never triggers hidden source materialization;
- a staging or incomplete cache is never opened;
- the UI remains responsive during the measured multi-minute Xenium build;
- two builders respect the existing publication lock;
- failed/cancelled construction cannot invalidate a previous completed cache;
- rebuild cannot leave a reader using a replaced path.

### Slice I10: remove the old transcript display path

Deliverables:

- remove direct transcript-selection Dask jobs and native-Points replacement
  code no longer used by another feature;
- remove the superseded transcript-cache helper in
  `src/napari_harpy/_transcript_tiles.py` once a consumer scan confirms the
  adopted builder/session owns every required use case;
- reduce or remove `src/napari_harpy/_points_value_index.py` according to its
  remaining non-transcript consumers;
- retain or relocate general point styling only where native Points still use
  it;
- remove obsolete tests that specify the superseded transcript workflow;
- add stale-import and package-boundary checks;
- update user and developer documentation.

Exit criteria:

- the SpatialData transcript points panel has one cache-backed implementation;
- no backend selector or forwarding compatibility layer remains;
- no normal transcript view builds `PointsValueSelection.coordinates`;
- focused existing viewer tests and all new tiled-points tests pass;
- native Points and shapes-as-points features still pass their focused tests.

### Slice I11: full-Xenium product evaluation

Use the retained completed cache without rebuilding it for renderer and reader
evaluation. Record one coherent run covering:

- time to open the reader and create the empty logical layer;
- time and peak RSS to prime all 108 bucket lookup indexes;
- baseline resident catalog and lookup bytes;
- selected-index time/bytes for 1, 10, and 100 abundant values;
- first and repeated viewport latency for all-values and selected requests;
- warm pan with no tile change;
- warm pan with entering/leaving tiles;
- Exact-to-coarse and coarse-to-Exact transitions;
- selection change while a viewport request is active;
- CPU and GPU tile residency, bytes, evictions, and coordinate uploads;
- GUI callback duration and visible frame behavior;
- full-extent common-value rendering with approximately 127 positive tiles;
- small-canvas screen-density budget and `within_budget=False` behavior;
- repeated layer add/remove and viewer shutdown.

The evaluation is an engineering review, not an exhaustive parameter sweep or
a comparison with the removed tiled-Parquet cache. If the layer is not
responsive enough, attribute time separately to:

```text
viewport conversion
LOD and tile planning
selected-index preparation
bucket lookup priming
point payload reads
CPU snapshot assembly
Qt delivery
VisPy upload/submission
draw-call/frame cost
```

Optimize the measured boundary rather than adding speculative concurrency.

## Test strategy

### Viewer-independent tests

- immutable dataset, viewport, selection, plan, key, tile, and snapshot models;
- exact coordinate-order conversions;
- budget calculation;
- plan-subset validation and bucket batching;
- selected-value index lifecycle;
- latest-request replacement and stale-result rejection;
- CPU LRU accounting, pinning, and eviction;
- over-budget no-read behavior;
- session close during every state.

### Napari model and Qt tests

- complete extent and fit-to-view without resident coordinates;
- layer transforms and world extent;
- custom registration, idempotence, rollback, and unsupported versions;
- viewport emission and deduplication;
- minimal controls and event disconnection;
- selected-value and status-panel integration;
- repeated layer add/remove and viewer close;
- proof that GUI callbacks do not run reader operations.

### Real VisPy/OpenGL tests

At least one supported local or CI environment must use a real canvas. Verify:

- asymmetric tile/image alignment and axis order;
- identity, translation, scale, rotation, and shear;
- tile-local precision at large dataset origins;
- constant screen-space point size and HiDPI behavior;
- palette, opacity, blending, and layer order;
- coordinate-upload counts across transforms, styles, and pans;
- atomic selection and LOD transitions;
- scene-node/draw-call behavior at realistic tile counts;
- GPU resource release after removal.

Mocked GL tests do not replace this gate.

### Deterministic cache integration fixtures

Build small real Zarr caches containing:

- asymmetric x/y geometry;
- empty grid regions;
- values present in one, several, and every tile;
- a value omitted by a sampled level;
- a terminal level that exceeds a supplied runtime budget;
- several tiles sharing one bucket and tiles in separate buckets;
- a pan sequence with overlapping tile sets;
- translated and rotated SpatialData coordinate systems.

## Diagnostics

Keep inexpensive development diagnostics available for:

```text
cache and selection generation
viewport/request generation
active and pending snapshot
selected level and within-budget state
omitted selected values
reader state and worker thread identity
selected-index bytes
bucket lookup-index bytes
CPU resident tile/point/byte counts
point-read count by tile identity
GPU resident tile/point/byte counts
coordinate upload and eviction counts
stale result count
last planning, read, delivery, upload, and activation durations
```

Do not put benchmark-only physical chunk statistics back into normal point
payloads.

## Risks and mitigations

### Private napari APIs

The custom layer visual, controls, and viewport callback depend on private
napari 0.7.1 behavior. Isolate imports, feature-detect, test exact supported
versions, and fail before layer insertion. Do not disguise incompatibility as a
cache error.

### Large resident bucket metadata

The evaluated full lookup index is approximately 596 MB. Project before load,
run load off the GUI thread, account actual bytes, and expose the configuration.
Do not call it the application's total RAM bound. If future caches routinely
exceed the metadata budget, design lookup eviction as a separate reader change.

### Many tile scene nodes

Tile nodes enable precise reuse but add draw calls. Measure the 127-tile case.
If necessary, pool renderer resources while retaining logical tile keys and
cache-independent snapshot contracts.

### Deferred GL work

VisPy buffer mutation may defer physical GPU work until draw. Define readiness
from observed real-canvas behavior. Do not claim that a timer meters GPU upload
unless the measurement proves it.

### Stale asynchronous state

Camera, selection, coordinate system, layer lifetime, and cache generation may
all change while IO runs. Every result carries generations, and GUI activation
checks them again immediately before mutation.

### Sampled value omission

A coarse level may contain zero rows for a requested value. Keep budget-first
selection, surface omitted IDs, and never describe sampled omission as exact
absence.

### Runtime budget below terminal overview

The reader truthfully returns `within_budget=False`. Do not read automatically.
Retain the old view or show an empty one and provide an actionable status.

### Cache replacement with an active reader

Lazy bucket paths can become invalid during directory replacement. Close and
quiesce the reader before rebuild publication in the initial workflow.

## Architecture invariants

- Camera movement never reads or mutates the canonical transcript dataframe.
- Camera movement never replaces a monolithic napari `Points` array.
- Cache validation and construction do not run during interactive viewing.
- `Layer.data` never contains resident transcript coordinates.
- Layer extent always describes the complete cache geometry.
- Cache `(x, y)`, napari `(y, x)`, and VisPy `(x, y)` ordering is explicit.
- The SpatialData transform is applied exactly once.
- All reader operations use one non-GUI owner thread.
- All napari, Qt, VisPy, and OpenGL mutations use the GUI thread.
- Selected-value catalog IO occurs only when the selection changes.
- Bucket lookup metadata is primed before point reads.
- Warm overlapping pans read only missing logical tiles.
- Requests sharing a bucket remain one coordinated bucket batch.
- No visualization read accesses `point_id`.
- No active snapshot mixes cache generations, selections, or levels.
- `within_budget=False` never triggers an automatic payload read.
- Style and transform changes never reupload coordinate buffers.
- A resident tile uploads at most once before GPU eviction.
- Upload-induced draws do not schedule an identical request.
- Stale worker results cannot activate themselves.
- Layer removal disconnects events and releases worker and GPU resources.
- The final SpatialData transcript workflow has one cache-backed renderer, not
  a hidden direct fallback.

## Recommended implementation order

1. Review and freeze I0.
2. Implement I1 before napari work so tile retention can avoid redundant reads.
3. Implement I2 and I3 as the narrow napari compatibility boundary.
4. Implement I4 and I5 with fake rendering and deterministic cache fixtures.
5. Implement I6 and hold the real-canvas renderer review.
6. Compose I7 and hold Gate I using a small real cache.
7. Replace the napari-harpy points workflow in I8.
8. Add product cache construction/rebuild handling in I9.
9. Remove the superseded direct transcript path in I10.
10. Run and record the retained full-Xenium product evaluation in I11.
