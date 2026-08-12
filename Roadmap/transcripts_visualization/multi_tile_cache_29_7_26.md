# Multiscale Transcript Tile Cache and Tiled Renderer

Status: authoritative investigation and implementation roadmap

Last updated: 2026-07-30

## Authority and relationship to older documents

This document is the authoritative roadmap for:

- the persistent multiscale visualization cache for transcript-like points;
- viewport-driven tile selection and asynchronous loading;
- CPU and GPU tile residency;
- the dedicated napari/VisPy transcript renderer;
- the boundary between the existing direct points path and the tiled path.

Where this document conflicts with the following notes, this document wins:

- `Transcripts_Tile_cache.md`
- `phase_1_spatial_first_cache.md`
- `visualizing_transcripts.md`
- `visualizing_transcripts_tiled_multiscale.md`
- `neuroglancer_napari_points_cache_recommendations.md`
- `ticket_napari_harpy.md`

Those files remain useful research and implementation history. They should not
be treated as current contracts for sampling, budgets, physical layout, or
napari integration.

`transcripts_gene_index_napari.md` remains authoritative for the existing
direct points-selection fallback. It is not the architecture for the tiled
renderer.

## Executive decision

Harpy will keep SpatialData points as the canonical transcript data and build a
separate, deletable visualization cache. Every cache level remains a point
representation; there is no raster fallback at coarse zoom.

The production tiled path will not repeatedly replace one monolithic
`napari.layers.Points.data` array. It will use a dedicated, read-only transcript
layer and a tiled rendering backend whose GPU buffers survive viewport changes.
The normal napari Points path remains available for small selections, fallback,
debugging, and correctness comparison.

The cache core uses `value` for the selected categorical point attribute.
Transcript datasets normally select the physical source column named `gene`, so
the public default is `value="gene"`. Internal schemas and algorithms use
`value`, `value_id`, and `value_table`; transcript-facing UI may still present
those values as genes when that matches the dataset semantics.

The durable investment is:

1. the cache contract;
2. immutable tile payloads;
3. viewport and LOD planning;
4. request scheduling and cache lifecycles;
5. a small rendering-backend interface.

VisPy is the first rendering backend, but VisPy-specific decisions must not leak
into cache construction or tile planning.

## Goals

The completed system must:

- visualize transcript datasets far larger than can be materialized in one
  napari Points layer;
- show points at every zoom level;
- guarantee a bounded whole-dataset overview;
- show exact source membership when the visible exact tiles fit the render
  budget;
- read only the Parquet row groups needed for the viewport;
- retain overlapping CPU and GPU tiles across pan and zoom;
- preserve stable value colors, cheap palette/style changes, and responsive
  selection-aware replanning when the active value set changes;
- keep the GUI responsive while disk reads and decoding happen in background
  workers;
- reject stale asynchronous results and prevent mixed-source or mixed-LOD
  displays;
- keep the direct points path working when no tiled cache is available.

## Non-goals for the first production version

The first production version does not need:

- raster or density-image rendering at coarse zoom;
- editing, adding, moving, or deleting transcripts through napari tools;
- a complete public napari extension API for custom layer renderers;
- 3D transcript rendering;
- remote/object-store cache construction;
- exact value-selective Parquet reads;
- per-transcript GPU picking;
- Morton ordering as a required part of the format;
- a second on-disk warm-cache format.

Here, exact value-selective Parquet reads means a physical layout that can avoid
reading mixed-value row groups. The first production path may read complete
manifest-selected tile row groups, filter their `value_id` rows in CPU/Arrow,
and upload only the active values. Physical value sharding, richer picking,
remote stores, and alternative rendering backends are later extensions. The
initial format must not make them impossible.

## Current repository state

### Historical implementation evidence

`src/napari_harpy/_transcript_tiles.py` contains historical implementations of:

- cache and level dataclasses;
- validation of backed SpatialData points elements;
- coordinate, gene, and transcript-id validation;
- bounds and regular-grid metadata;
- deterministic gene dictionary construction;
- `genes.parquet` writing;
- conversion to tile-local `float32` coordinates;
- finest-level tile annotation;
- tile-specific Parquet row-group writing;
- manifest-row collection for physical row groups;
- staged replacement and rollback helpers.

`tests/test_transcript_tiles.py` covers those historical behaviors. At the time
this roadmap was written, its focused test module passed 103 tests. That result
shows consistency with the legacy tests; it does not make the legacy models,
schemas, Dask execution pattern, or writer architecture part of the new
specification.

These names describe the legacy implementation only. The new package
generalizes the categorical column to `value` and does not retain its
caller-supplied transcript-id path. Nothing in the old module is presumed
reusable. Every retained invariant, algorithm, schema field, or test case needs
an independent justification from the new cache and runtime requirements.

### Not implemented

The repository does not yet contain:

- a public end-to-end cache builder;
- sampled coarse-level construction;
- a Harpy-owned stable internal `uint64 point_id`;
- final `metadata.json`, `manifest.parquet`, and tile/value count-index writers;
- a completed-cache marker and complete reader validation;
- source-staleness inspection;
- a runtime tile store;
- viewport-to-tile planning;
- LOD selection;
- a request scheduler or byte-bounded CPU tile cache;
- a dedicated transcript layer;
- a tile-retaining VisPy renderer or GPU cache;
- tiled-mode UI and lifecycle integration.

The active viewer path still validates and scans the source Dask dataframe,
filters selected values, applies global random sampling, materializes one
selection, and creates a normal napari Points layer. The current controller
reports tiled-cache construction as unavailable.

### Consequence

The existing code is best described as tested writer primitives, not as an
almost-complete multiscale feature. The production replacement starts from a
fresh package at:

```text
src/napari_harpy/core/multi_scale_cache_points/
```

The new package may inspect `_transcript_tiles.py` as implementation history, but
it must not import from it or copy its models and schemas by default. A retained
idea must be expressed independently and justified by the new contracts. This
keeps the replacement independently testable and makes eventual removal of the
old module straightforward.

`src/napari_harpy/_transcript_tiles.py` and
`tests/test_transcript_tiles.py` remain temporarily as implementation history
and a source of possible edge cases, not as an authoritative behavioral
specification. They are removed only
after the new builder, reader, and product integration have replaced every
required use. The removal is a dedicated cleanup change, not part of the first
implementation slice.

## Locked design decisions

These decisions should be treated as requirements unless a later ADR explicitly
changes them.

### Implementation ownership and work blocks

The work is divided into four explicit blocks:

1. physical source resolution and validation;
2. persistent cache construction;
3. cache reading, tile planning, and scheduling;
4. napari/VisPy rendering and Harpy integration.

Block 3 is deliberately separate from rendering. The renderer consumes
immutable tile payloads and snapshots; it does not know about Dask, Parquet
file discovery, manifests, source validation, or cache publication.

The first implementation slice is Block 1. Cache writing does not begin until
the source-resolution and validation result is a stable, tested contract.

The core cache package must not import Qt, napari, or VisPy. Napari-specific
code belongs under a viewer-facing package such as:

```text
src/napari_harpy/viewer/multi_scale_points/
```

Private napari registration details remain isolated at that boundary.

### Canonical data and cache ownership

- The SpatialData points element remains canonical.
- `transcripts_vis/` is a Harpy-owned derived cache.
- The cache may be deleted and rebuilt without changing canonical data.
- Cache coordinates are stored in the native coordinate space of the points
  element.
- The selected SpatialData transform is applied by the napari layer, not baked
  into every cached point.

### Point-only LODs

- Every level contains point rows.
- The finest level has full source membership.
- Coarse rows are actual source transcript representatives, not centroids with
  invented identities.
- Every representative retains the Harpy-owned internal `point_id`.
- No coarse level is replaced with a raster.

### Self-contained, nested levels

Each level is independently renderable:

```text
level_n ⊆ ... ⊆ level_2 ⊆ level_1 ⊆ level_0
level_0 = exact source membership
level_n = terminal coarsest overview
```

The same representative may therefore occur in several levels. This modest
storage duplication keeps runtime semantics simple: choose one level and render
it. It also reduces visual instability during LOD changes.

Harpy will not use Neuroglancer-style residual/disjoint levels in the first
format. Residual levels require cumulative multi-level reads or mixed-level
rendering, which conflicts with the initial one-active-LOD contract.

### Value-neutral coarse sampling and value-aware planning

The sampled pyramid is spatially stratified but value-neutral. It must not
deliberately oversample rare values or use `value_id` to influence membership:
doing so would distort apparent abundance when all values are displayed.
Disappearance from a sampled level is an expected consequence of
representative sampling, not evidence that a value is biologically absent.

Every sampled level uses a 16 × 16 microgrid relative to its current logical
tile, as approved at Gate C after the focused C5a and C5d spikes. The
bridge is the first concrete demonstration; later spatial levels use the same
sampler after immediate-finer coordinates have been rebased into their coarser
tile. C5a also freezes the concrete priority-hash payload. The following
structural guarantees are already fixed:

- deterministic results for the same source identity and build parameters;
- actual source rows as representatives;
- stable pseudo-random priority based on a named, versioned hash algorithm;
- spatial stratification within a tile;
- proportional allocation across occupied spatial strata;
- monotonically increasing membership from coarse to fine;
- no level or tile budget overrun;
- no dependence on Python's randomized `hash()`;
- deterministic tie-breaking;
- deterministic use of the Harpy-owned internal `point_id`;
- no influence from `value_id` on allocation, priority, or winner selection.

Value awareness belongs in cache indexing and runtime planning. Every level
therefore publishes sparse per-tile counts by `value_id`. When the user selects
one or more values, the planner uses those counts to skip zero-count tiles and
choose the finest level satisfying screen-scale, selected-render-count, and
physical-row-read budgets. The initial reader may still read a complete
mixed-value tile row group, filter by `value_id`, and upload only selected rows;
physically value-sharded point IO is a later measured optimization.

### Tile geometry and sampling density are different concepts

Tile size answers:

> Which spatial payloads should be read for this viewport?

Sampling density answers:

> How many representative points should be drawn at this LOD?

They must not be inferred from each other. Dataset pixels are data-coordinate
units, not screen pixels. A tile's projected screen size depends on the current
camera transform and is computed from the viewport's data-units-per-screen-pixel
value.

Consequently:

- level count must not be derived only from dataset extent;
- the format supports two levels with the same tile size and different sampling
  densities, but the initial schedule uses that capability only for the sampled
  finest bridge;
- level metadata records both grid geometry and sampling semantics;
- the planner uses both screen scale and manifest counts.

### Initial construction schedule

The first implementation targets the following schedule, written from finest
source geometry toward coarser sampled geometry:

| Design label | Tile geometry | Maximum representatives stored per logical tile |
|---|---:|---:|
| Exact | 512 | uncapped: all source points belonging to the tile |
| Sampled finest bridge | 512 | 4,096 |
| L1 | 1,024 | 8,192 |
| L2 | 2,048 | 16,384 |
| L3 | 4,096 | 32,768 |
| Later spatial levels | double the preceding tile edge | initially double the preceding per-tile capacity |

Serialized level numbers follow construction from finest to coarsest:

```text
Exact → sampled finest bridge → L1 → L2 → ... → overview
  0               1             2     3              n
```

`L1`, `L2`, and later `L*` names remain spatial design labels, so `L1` has
serialized level number 2 because the same-geometry bridge occupies level 1.
The exact-only case contains only level 0, which is then both finest and
coarsest.

The sampled finest bridge is intentional. A dense exact 512-unit tile can
exceed the runtime render budget while it is still large on screen. The bridge
provides a bounded representation without prematurely moving to 1,024-unit
spatial payloads. The initial implementation does not add arbitrary
`Density A`, `Density B`, or further equal-geometry levels.

For the non-terminal sampled progression:

```text
tile_size(k) = 512 * 2**k
tile_capacity(k) = 4,096 * 2**k
```

where `k = 0` is the sampled finest bridge. Doubling the tile edge combines
approximately four immediate-finer tiles, while doubling rather than
quadrupling the capacity. A fully populated coarser tile therefore retains
approximately half the representatives in its four contributing finer tiles.
This targets an approximately twofold point-count change between adjacent
sampled LODs rather than an automatic fourfold change.

Capacity is a hard maximum, not a fill target. Sparse tiles retain all available
candidates. Every sampled level is built from representatives retained by the
next finer level.

The doubling rule is not allowed to violate the global coarsest-level contract.
Construction continues until a complete whole-dataset level satisfies
`overview_point_budget`. If the grid reaches one tile while its scheduled
per-tile capacity still exceeds that budget, the terminal level's effective
per-tile capacity is clamped to `overview_point_budget`. For a one-tile level,
that capacity is also the complete-level capacity, so no separate global
allocation field is required.

This schedule is the initial implementation and benchmark target, not an
immutable file-format restriction. Changing it later requires benchmark
evidence from real viewport traces, screen-space density, value-distribution
fidelity without deliberate categorical reweighting, build cost, and LOD
transition quality.

### Separate budgets

The following settings have distinct meanings:

`overview_point_budget`
: Maximum total point count in the complete coarsest level.

`max_rows_per_row_group`
: Maximum physical Parquet IO shard size, initially 1,000,000 rows. It does not
  control visual sampling.

`level_sampling_target`
: A level-specific effective maximum per-tile capacity recorded in level
  metadata. The initial sampled targets are 4,096, 8,192, 16,384, 32,768, and
  so on; a terminal one-tile target is capped by `overview_point_budget` when
  necessary.

`render_point_budget`
: Runtime maximum for visible core tiles plus the configured prefetch policy.
  It may be larger than the build-time overview budget. The initial runtime
  range to benchmark is 100,000-200,000 visible transcripts; it is a safety
  ceiling, not a target that the renderer must fill.

`cpu_cache_byte_budget`
: Maximum decoded CPU tile-cache memory.

`gpu_cache_byte_budget`
: Maximum retained GPU tile-buffer memory for one rendering context.

These values must not be collapsed into one `coarse_tile_budget`.

The coarsest-level invariant is:

```text
coarsest_level = max(level in planned levels)
sum(manifest.n_points where level == coarsest_level) <= overview_point_budget
```

The cache advertises the actual coarsest count as its minimum supported
whole-dataset render budget.

### Runtime IO

- Dask may be used for offline construction when a measured construction design
  justifies it; it is not required merely because the legacy writer uses it or
  because it is already a dependency.
- Runtime tile reads use PyArrow against known Parquet files and row groups.
- Interactive camera updates do not construct or execute a Dask graph.
- Runtime payloads are immutable, contiguous arrays suitable for CPU caching
  and GPU upload.

### Rendering

- A normal napari Points layer is not the production camera-driven hot path.
- The renderer retains independently addressable GPU tile payloads.
- Camera movement reuses resident buffers.
- Palette, color, opacity, and point-size style updates do not reupload
  coordinate buffers.
- Changing the active selected-value set may trigger a new plan, filtered tile
  reads, and an atomic coordinate-buffer replacement.
- Disk and Parquet work never touches VisPy objects.
- GPU creation, upload, and deletion occur on the GUI/OpenGL thread.
- The physical GPU representation is private to the rendering backend.

### LOD transitions

- Same-level pan retains overlapping tiles and loads only entering tiles.
- A cross-level transition keeps the active level visible until all new core
  tiles are GPU-ready.
- A new LOD activates as one immutable snapshot.
- No active snapshot mixes cache generations, source signatures, levels, or
  value selections.
- Small zoom changes use hysteresis so they do not oscillate between levels.

## Target architecture

The complete system has an offline construction side and a runtime side:

```text
Backed SpatialData points element
                         │
                         ▼
              PointsSourceResolver
                         │
                         ▼
              PointsSourceValidator
          file metadata + bounded scans
                         │
                         ▼
              ValidatedPointsSource
                         │
                         ▼
              MultiscaleCacheBuilder
       exact level → sampled levels → publish
                         │
                         ▼
              completed transcripts_vis/
                         │
          ┌──────────────┴──────────────┐
          ▼                             ▼
 TranscriptTileStore             freshness/status
          │
          ▼
 planner → scheduler → backend → napari
```

The initial source and construction package is:

```text
src/napari_harpy/core/multi_scale_cache_points/
  __init__.py
  models.py
  source.py
  validation.py
  signature.py
```

Only add modules when their responsibility becomes concrete. Cache
construction is expected to add:

```text
  builder.py
  build_plan.py
  hashing.py
  sampling.py
  manifest.py
  publication.py
  writer/
    __init__.py
    models.py
    support.py
    exact.py
    bridge.py
    spatial.py
```

The private `writer/` subpackage groups physical writer contracts and
implementations. Planning, sampling, shared hashing, manifest assembly, and
publication remain at the parent package level because they are not physical
writer implementations.

Runtime cache consumption is expected to add:

```text
  store.py
  planner.py
  scheduler.py
```

`__init__.py` exposes a deliberately small public API. Implementation modules
remain private unless a stable external use case is identified.

The runtime flow is:

```text
                         napari camera + canvas + dims
                                      │
                                      ▼
TranscriptLayerModel ───────► TranscriptTilePlanner
persistent user state             viewport + LOD policy
         │                              │
         │                              ▼
         │                     TranscriptTileScheduler
         │                 generations + priorities + CPU LRU
         │                              │
         │                    tile requests / immutable payloads
         │                              │
         └──────────────► TranscriptTileStore
                           metadata + manifest + PyArrow
                                      │
                                      ▼
                         RenderSnapshot + TilePayloads
                                      │
                                      ▼
                         TranscriptRenderBackend
                                      │
                                      └── VisPy backend
                                          GPU LRU + upload queue
                                          compact point visual
                                          value palette
```

### `TranscriptLayerModel`

The layer model is the persistent, view-independent object shown in napari's
layer list.

It owns:

- an immutable transcript dataset reference;
- active selected value IDs, driven by the napari value-selection/visibility UI;
- value palette and style/highlight state;
- point size;
- render and prefetch settings;
- user-visible status;
- normal napari visibility, opacity, blending, and transforms.

It does not own:

- a full `N x 2` point array;
- camera position;
- current viewport tiles;
- Dask dataframes;
- open Parquet readers;
- VisPy nodes or GPU buffers.

The model should subclass napari `Layer`, not `Points`. Presenting transient
viewport/LOD contents as `layer.data` would incorrectly imply that they are the
canonical transcript collection.

Napari already uses `Layer.source` for provenance. Use a name such as
`transcript_dataset` or `transcript_source_ref` for the transcript source.

The layer extent always reports complete dataset bounds. It must not change as
tiles enter and leave the viewport.

### `TranscriptDatasetRef`

This immutable value object identifies one cache and its canonical source:

```python
@dataclass(frozen=True)
class TranscriptDatasetRef:
    spatialdata_identity: object
    points_name: str
    coordinate_system: str
    cache_location: object
    cache_schema_version: str
    cache_generation_id: str
    source_signature: str
```

The concrete `cache_location` API must leave room for a future filesystem/URI
abstraction even if the first writer supports local paths only.

### `TranscriptTileStore`

The store is independent of Qt, napari, and VisPy.

It owns:

- parsed and validated cache metadata;
- the value dictionary;
- the tile/row-group manifest index;
- the sparse per-level tile/value count index;
- PyArrow row-group reads;
- decoding of tile-local coordinates and features;
- a byte-bounded CPU LRU, if the cache is shared at store scope.

Core operations:

```python
class TranscriptTileStore:
    @classmethod
    def from_path(cls, path: Path) -> "TranscriptTileStore": ...

    def tiles_intersecting(
        self,
        *,
        level: int,
        data_bounds: tuple[float, float, float, float],
    ) -> tuple["TileKey", ...]: ...

    def estimated_point_count(
        self,
        tile_keys: tuple["TileKey", ...],
    ) -> int: ...

    def selected_point_counts(
        self,
        *,
        tile_keys: tuple["TileKey", ...],
        value_ids: tuple[int, ...],
    ) -> tuple[int, ...]: ...

    def load_tile(
        self,
        key: "TileKey",
        *,
        value_ids: tuple[int, ...] | None = None,
    ) -> "TilePayload": ...
```

`value_ids=None` means all values and uses manifest totals directly. An explicit
selection queries the sparse count index; tiles with a selected count of zero
need no point-row read. A positive tile is initially loaded from all manifest
row-group shards, filtered by `value_id` in CPU/Arrow, and returned with only the
selected rows. A decoded unfiltered CPU tile may be retained and reused when its
byte cost fits the CPU-cache policy.

The store must not scan all tile/value rows on every camera or selection event.
It opens an immutable lookup over the sorted sidecar so a query touches only the
requested level/value ranges and tile keys. Whether that lookup is compactly
materialized in memory or backed by Arrow row-group pruning is selected from
Phase 2 measurements; all-values planning bypasses it entirely.

One logical tile may consist of several physical row-group shards. The store
combines those shards into one immutable `TilePayload`; the renderer does not
need to know how many files or row groups backed the tile.

### `TranscriptTilePlanner`

The planner is pure policy. Given a view description and cache metadata, it
returns a render plan without starting IO.

Inputs include:

- inverse-transformed viewport bounds in cache data coordinates;
- canvas size and data-units-per-screen-pixel;
- available levels and their sampling metadata;
- manifest point counts;
- active selected value IDs and sparse per-level tile/value counts;
- render budget;
- physical row-read budget as the initial IO-cost proxy;
- prefetch margin;
- previous level for hysteresis.

LOD selection chooses the finest level satisfying all of:

1. its sampling density is appropriate for the screen scale;
2. selected rows in visible core tiles fit the render budget;
3. complete mixed-value row groups that must be read fit the physical row-read
   budget;
4. every explicitly selected value with a nonzero exact core-view count is not
   silently represented as absent when a finer affordable level contains it.

For all-values rendering, selected-row and physical-row counts are the manifest
totals and the sparse value index is bypassed. For an explicit selection, the
planner sums only matching tile/value counts for its render estimate, while its
initial physical IO estimate sums complete manifest rows for tiles whose
matching count is nonzero. The exact level is chosen whenever these core-view
constraints fit.

If exact-level counts show a selected value in the viewport but a sampled level
contains none, that sampled level cannot be presented as evidence of biological
absence. The planner chooses a finer level when budgets permit. If no level can
satisfy scale, render, IO, and selection-coverage constraints, it retains a safe
bounded representation and reports the limitation explicitly rather than
silently changing sampling semantics.

The initial implementation may budget core plus prefetch together. If this
causes unnecessary coarse LOD selection, core tiles must remain hard-bounded
while prefetch becomes a soft, separately capped budget. That choice is part of
the Phase 2 planner benchmark.

### `TranscriptTileScheduler`

There is one scheduler per `(viewer canvas, transcript layer)` pair.

It owns:

- monotonically increasing view generations;
- core and prefetch priorities;
- bounded concurrency;
- cancellation intent;
- in-flight request bookkeeping;
- stale-result rejection;
- CPU cache interaction;
- renderer upload requests;
- pending and active render snapshots.

Conceptual tile lifecycle:

```text
ABSENT
  -> REQUESTED
  -> LOADING
  -> CPU_READY
  -> UPLOAD_QUEUED
  -> GPU_READY
  -> ACTIVE
  -> CACHED
  -> EVICTED
```

Stale results may populate the CPU cache if their source identity is still
valid. They may never activate an obsolete render plan.

Rapid camera events are coalesced. The scheduler must not start an unbounded
sequence of reads during a wheel gesture or animated pan.

### `RenderSnapshot`

The scheduler hands the renderer immutable snapshots:

```python
@dataclass(frozen=True)
class RenderSnapshot:
    generation: int
    cache_generation_id: str
    source_signature: str
    selection_key: str
    level: int
    core_tile_keys: tuple["TileKey", ...]
    prefetch_tile_keys: tuple["TileKey", ...]
    expected_point_count: int
```

`selection_key` is a deterministic identity for the normalized active value-ID
set, with one reserved identity for all values. Logical `TileKey` identifies an
on-disk tile, but filtered CPU payloads and GPU readiness are keyed by both
`TileKey` and `selection_key`. This prevents an asynchronous payload for an old
selection from satisfying or activating a new selection's snapshot.

The renderer activates a pending cross-level snapshot only when all core tiles
are GPU-ready.

### `TranscriptRenderBackend`

The scheduler depends on a small protocol rather than on VisPy:

```python
class TranscriptRenderBackend(Protocol):
    def enqueue_upload(self, tile: "TilePayload") -> None: ...
    def is_ready(self, key: "TileKey") -> bool: ...
    def activate(self, snapshot: RenderSnapshot) -> None: ...
    def update_style(self, style: "TranscriptStyle") -> None: ...
    def evict(self, key: "TileKey") -> None: ...
    def close(self) -> None: ...
```

This boundary allows tests to use a fake backend and allows the VisPy
implementation to change its internal buffer strategy without changing the
store or scheduler.

### VisPy backend

The first production backend owns:

- per-context GPU tile handles;
- a byte-bounded GPU LRU;
- active and pending snapshot pinning;
- a GUI-thread upload queue;
- a per-frame upload byte/time budget;
- tile visibility;
- a compact point shader;
- value palette and style/highlight lookup;
- point-size and opacity uniforms;
- context loss and cleanup.

The intended GPU vertex payload is approximately:

```text
x_rel: float32
y_rel: float32
value_id: uint16, uint32, or exactly represented float32
```

Disk and CPU dtypes do not have to equal GPU attribute dtypes. In particular,
the cache may store `value_id` as `uint32` while a compatibility-oriented VisPy
prototype uses `float32` in the vertex buffer.

Internal point ids remain CPU-side unless a measured picking design requires them
on the GPU.

Whether the backend uses one scene node per tile, several VBOs in one visual,
pooled buffer ranges, or a future multi-draw path is deliberately not part of
the architecture contract.

## Physical source resolution and validation

### First implementation slice

Source resolution and validation are implemented before cache writing.
Validation returns an immutable build input containing all deterministic source
facts needed by later stages.

A representative model is:

```python
@dataclass(frozen=True)
class ValidatedPointsSource:
    source: ParquetPointsSource
    files: tuple[ParquetSourceFile, ...]
    selected_schema: pa.Schema
    row_count: int
    bounds: PointsBounds
    value_table: pa.Table
    source_signature: str
    source_signature_method: str
    value_normalization_method: str
    point_id_policy: str
```

`selected_schema` contains only the caller-selected `x`, `y`, and `value` fields
in canonical semantic order. Unselected source columns are not part of the
build contract and need not match across source files.

The nested `source` retains the selected columns and canonical SpatialData
identity without duplicating path state. Its `element_path` and `parquet_path`
remain derived properties.

The precise names may change during implementation, but the separation between
an unresolved source and a validated immutable build input is required.

### Canonical physical source

For the initial SpatialData contract, the resolver derives the physical
Parquet dataset from the backed store path and points-element name. The
resulting source description exposes that concrete path explicitly to the
validator and builder. It does not reverse-engineer an arbitrary Dask
expression graph.

The intended API separation is:

```python
def resolve_spatialdata_points_source(
    sdata: SpatialData,
    points_name: str,
    *,
    x: str = "x",
    y: str = "y",
    value: str = "gene",
) -> ParquetPointsSource: ...


def validate_parquet_points_source(
    source: ParquetPointsSource,
    *,
    max_batch_rows: int = 1_048_576,
) -> ValidatedPointsSource: ...
```

The cache builder accepts the returned `ValidatedPointsSource` directly.
Validation does not return or persist a diagnostics report, performance
timings, transient counters, machine information, or a generation timestamp.
Performance measurements belong to dedicated benchmark tooling.
The first validation implementation is synchronous and exposes no progress or
cancellation protocol. Those orchestration concerns may be added later without
changing the validated-source or cache-format contracts.

The API does not accept a source transcript-id column. Validation establishes
deterministic source-file offsets, and cache construction generates a Harpy-owned
`uint64 point_id` from each source-file offset and row position.
`point_id_policy` stores the complete versioned policy name as a string; the
initial parameter-free policy does not require a wrapper model.

For the supported SpatialData contract, the resolver derives the element path
as `points/<points_name>` and its `points.parquet` dataset from the backed store
path. `/` is not allowed inside `points_name`. `ParquetPointsSource` stores only
`spatialdata_path`, `points_name`, and the selected columns; its `element_path`
and `parquet_path` are derived properties. The former is the
SpatialData-root-relative logical path, while the latter is the concrete
filesystem path ending in `points.parquet`.

A future standalone or non-canonical Parquet entry point should use a separate
source contract. Arbitrary filtered or transformed Dask dataframes are not
silently assumed to map one-to-one to the canonical physical dataset.

### Private Parquet source inventory

The validator first constructs a private `_ParquetSourceInventory` that:

- verifies that the path is a readable Parquet dataset;
- creates a deterministic relative file inventory;
- validates compatible schemas across files and row groups;
- validates required column names and physical types;
- collects file sizes, row counts, ordered row-group row counts and compressed
  sizes;
- computes deterministic source-file row offsets;
- provides the complete metadata input for the versioned source signature.

This private object does not contain preliminary coordinate bounds, file-metadata
null statistics, or decisions about whether scanning can be skipped. It reads
Parquet file metadata without decoding point rows and is not accepted by the cache
builder.

### Bounded content validation

The validator always performs one bounded streaming scan over the actual
Parquet data pages for the selected `x`, `y`, and `value` columns. It establishes:

- missing, NaN, or infinite coordinates;
- missing or normalized-empty values;
- normalized value counts;
- exact coordinate bounds;
- scanned row counts per row group, source file, and complete source.

Reads use bounded PyArrow batches. Dictionary-encoded values are handled by
normalizing each returned dictionary once and aggregating its integer indices
rather than converting every point value through Python strings. Plain-string
batches use eager, bounded Arrow normalization and `value_counts` kernels; only
their distinct-label/count results enter the Python source-wide accumulator.
These calls do not construct a Dask graph or reread the Parquet source. Raw
labels that normalize to the same value are merged; no collision count is
returned or persisted. Dictionary indices remain local physical codes and are
never treated as dataset-wide value identities. Counts are merged globally by
normalized label, so dictionary membership and index ordering may differ across
batches or row groups without changing the final `value_table`.

Content validation is fail-fast. A file-open, decode, structural, coordinate,
or value failure stops the scan immediately; validation does not traverse the
remaining source to produce exhaustive diagnostics. Errors include file,
row-group, selected-column, and failure-category context. A naturally available
invalid count may be reported only as local to the failing batch.

The scan reconciles counts without another data pass: batch rows sum to each
inventory row-group count, observed row groups sum to each source-file count,
observed source files sum to the inventory total, and normalized value counts
sum to both the observed and inventory totals. Batch-local value counts must
also sum to the current batch row count. Scope counters are transient Python
integers; no reconciliation report is returned or persisted. The returned scan
row count is the successfully observed total rather than a copied metadata
value. These checks detect traversal and aggregation disagreement, not
same-count content mutation.

The authoritative source for each build fact is fixed:

- row count and source-file offsets come from Parquet metadata;
- coordinate bounds and normalized value counts come from the streaming scan;
- stale-source detection uses the versioned source signature.

The validator must fail clearly when required correctness cannot be established.
It constructs `ValidatedPointsSource` only after the scanned counts agree with
the source inventory, value counts sum to the source row count, and a repeated
inventory inspection produces the same source signature.

### Validation acceptance dataset

The first real-data acceptance target is:

```text
sdata_xenium_full_data_core.zarr/
  points/transcripts_global_ROI1/points.parquet
```

At the time of investigation it contains:

- 136,578,750 rows;
- 65 Parquet files and 168 row groups;
- 5,122 normalized values from the physical `gene` column;
- `x`, `y`, and `gene` source columns;
- `x` bounds approximately `[38.3088, 54047.2059]`;
- `y` bounds approximately `[22.7206, 37581.4706]`;

The validated source must contain those counts and measured coordinate bounds
without materializing the complete dataframe, produce deterministic source-file
offsets and a repeatable source signature, and remain within a documented
bounded-memory envelope.

## Persistent cache contract

### Target layout

The logical cache layout is:

```text
<sdata.zarr>/
  points/
    <points_name>/
      points.parquet
      transcripts_vis/
        metadata.json
        manifest.parquet
        values.parquet
        tile_value_counts.parquet
        levels/
          level_0/
            part-00000.parquet
            ...
          level_1/
            ...
          level_n/
            ...
        COMPLETED
```

`ParquetPointsSource.element_path` derives `points/<points_name>` from the
validated name, and `ParquetPointsSource.parquet_path` derives
`<spatialdata_path>/<element_path>/points.parquet`. Do not infer a different
physical Parquet path by inspecting Dask graph internals.

All paths stored in metadata or the manifest are relative to the cache root.

`COMPLETED` is written only after every required file has been written and
validated. A reader rejects caches without it.

For local filesystems, every build—including the first build—uses a unique
sibling staging directory and installs the completed directory with a rename.
An incomplete build is never written directly into the final visible cache
path.

Remote/object-store transactional publication needs a generation-directory plus
pointer design and is deferred.

### Schema versioning

The existing constant is `harpy-transcripts-vis-0.1`, but no public
end-to-end builder currently exists.

- If no `0.1` cache has been used outside development, the format may be
  redefined before the first public builder.
- If any such cache must remain readable, introduce
  `harpy-transcripts-vis-0.2`.
- Readers reject unsupported versions; they do not guess.

### `metadata.json`

Required cache identity:

- `schema_version`
- `cache_generation_id`
- `created_by` package and version

Required source identity:

- points element name;
- resolved element path;
- source row count;
- source schema summary;
- coordinate and value column names;
- source-signature method and value.

Required geometry:

- `x_origin`, `y_origin`;
- `x_min`, `x_max`, `y_min`, `y_max`;
- axis convention;
- coordinate dtype contract.

Required level records, ordered by ascending serialized level from finest to
coarsest:

- `level`;
- `tile_size`;
- grid shape or equivalent validated grid bounds;
- `is_exact`;
- total stored point count;
- sampling-policy name and version;
- effective maximum per-tile capacity;
- sampling target or density semantics;
- level directory.

Required build parameters:

- `leaf_tile_size`;
- `overview_point_budget`;
- `max_rows_per_row_group`, initially `1_000_000`;
- stable hash algorithm and seed;
- sampler name/version and parameters;
- value-normalization method;
- internal point-identity policy.

Required auxiliary-index records:

- tile/value count-index path;
- tile/value count-index method/version.

`metadata.json` is the source of truth for cache semantics. The manifest is the
source of truth for physical tile/row-group locations and actual stored counts.
The tile/value count index is the source of truth for selection-aware count
estimates, but never for locating point row groups.

### Cache and source identity

Two identities solve different problems.

`cache_generation_id`
: A fresh UUID or equivalent for each successful build. It prevents runtime
  mixing of decoded or GPU-resident tiles from different cache builds.

`source_signature`
: Evidence that the cache still corresponds to the canonical points source.

The first source-signature method is
`harpy-parquet-source-inventory-sha256-v1`. It hashes the exact canonical UTF-8
JSON representation specified by the validation roadmap. The payload contains
the SpatialData-relative element path, selected `x`, `y`, and `value` names and
normalized type descriptors, ordered dataset-relative source files, file sizes,
available nanosecond modification times, file-metadata and row-group row counts,
row-group compressed sizes, and total row count.

It excludes the absolute host path, Parquet min/max statistics, performance
measurements, and generation timestamps. It is not a cryptographic hash of
every Parquet data page, so the UI and API must not claim stronger guarantees
than it provides.

### Construction-time source-signature guards

Cache construction trusts the content facts already established by
`ValidatedPointsSource` when the current source signature matches its
`source_signature`. The builder does not reconstruct `ValidatedPointsSource`,
repeat the bounded content scan, or independently recompute source row counts,
bounds, and normalized value counts from point data.

`_read_current_source_signature(validated)` is metadata-only. It freshly
discovers the current Parquet files, reads their filesystem and file metadata,
constructs `_ParquetSourceInventory`, and applies the versioned signature method. It
does not decode the `x`, `y`, or `value` data pages and must not reuse the stale
source inventory retained from validation.

The initial builder flow is:

```python
expected_signature = validated.source_signature

signature_at_start = _read_current_source_signature(validated)

if signature_at_start != expected_signature:
    raise PointsSourceValidationError(
        "The points source changed after it was validated."
    )

staging = create_staging_cache()

try:
    build_exact_level(validated, staging)
    build_sampled_levels(staging)
    write_metadata_and_manifest(staging)
    validate_staged_cache(staging)

    signature_before_publish = _read_current_source_signature(
        validated
    )

    if signature_before_publish != expected_signature:
        raise PointsSourceValidationError(
            "The points source changed while the cache was being built."
        )

    write_completion_marker(staging)
    publish_staged_cache(staging)

except Exception:
    reject_incomplete_staging_cache(staging)
    raise
```

Both comparisons use the original signature stored in
`ValidatedPointsSource`, not merely the two freshly calculated signatures. A
failure to inspect the current source is also a failed guard. The initial guard
runs before staging work begins. The final guard runs after staged-cache
validation and immediately before the completion marker and atomic publication.

A failed guard never publishes the staged generation and preserves any existing
completed cache. Normal staged-cache validation still checks the cache's own
metadata, manifest, files, row groups, and writer accounting. It does not repeat
source-content validation or rebuild source row-count, bounds, and value-count
aggregates from the Parquet point data.

The reader reports at least:

```text
VALID
STALE
UNVERIFIABLE
INVALID
ABSENT
```

A cache generation id is mandatory even when source freshness is
`UNVERIFIABLE`.

After publication, the reader uses the same metadata-only signature method to
classify a later source change as `STALE`. This runtime freshness check does not
replace the two construction-time guards.

### Grid convention

Coordinates and tile indices use the points element's native data space.

The proposed initial builder aligns one shared origin to the leaf grid:

```text
x_origin = floor(x_min / leaf_tile_size) * leaf_tile_size
y_origin = floor(y_min / leaf_tile_size) * leaf_tile_size
```

All levels reuse this origin. The rule keeps adjacent finer and coarser grids
aligned and produces non-negative tile indices for finite negative or positive
source coordinates. C1 implements this contract in its IO-free build planner.

For one level:

```text
tile_x = floor((x - x_origin) / tile_size)
tile_y = floor((y - y_origin) / tile_size)
tile_id = f"{level}/{tile_x}/{tile_y}"
```

Tile cells are half-open in each dimension. Grid shape must be computed from
the maximum assigned tile indices, not assumed solely from the numeric extent.
This handles points that lie exactly on a tile boundary, including the source
maximum: such a point belongs to the tile beginning at that boundary. The
initial implementation uses direct `float64` arithmetic and `floor`, without an
epsilon or `nextafter` adjustment.

The coarsest grid should normally cover the complete dataset in one tile by
choosing a tile size strictly greater than the maximum coordinate span. This
makes the global overview budget straightforward. The format must still support
more than one coarsest tile if a later builder deliberately chooses that
layout; the global coarsest budget remains mandatory.

### `values.parquet`

Required columns:

```text
value_id: uint32, non-nullable
value: string, non-nullable
n_points: uint64, non-nullable
```

The file contains exactly those columns in that order and carries no custom
Arrow schema or field metadata. For `N` normalized values, its rows use
contiguous `value_id` values from zero through `N - 1`; normalized labels are
unique, non-empty, and sorted lexicographically by their UTF-8 bytes; every
`n_points` is positive; and the counts sum to the exact original source row
count. `n_points` is not a sampled-level count. Empty sources are rejected, and
the initial format supports at most `2**32` distinct values.

Values are normalized according to one documented policy, assigned stable ids
in deterministic order, and never inferred from GPU palette order. The initial
policy is
`harpy-string-trim-unicode-white-space-case-sensitive-v1`: it trims an explicit
versioned set of Unicode `White_Space` code points, remains case-sensitive,
performs no other Unicode normalization, and orders labels by ascending UTF-8
bytes. Null logical values and referenced normalized-empty values are rejected;
invalid unreferenced physical dictionary entries are ignored. Arrow `string`,
`large_string`, and dictionary-string source encodings all produce the same
canonical `string` output field; an unrepresentable normalized label is a
validation failure rather than a schema change.

### `tile_value_counts.parquet`

This sparse planning index contains one row for every nonzero
`(level, value_id, tile_x, tile_y)` combination:

```text
level: int16
value_id: uint32
tile_x: uint32
tile_y: uint32
n_points: uint64
```

Rows are unique, positive-count, and sorted by
`(level, value_id, tile_y, tile_x)`. Zero-count combinations are omitted. Every
`value_id` must exist in `values.parquet`, and every tile key must exist in the
manifest. The file is organized so contiguous value-ID ranges can be read or
indexed without loading a dense `n_levels × n_tiles × n_values` matrix; exact
row-group sizing is frozen from C7/C9 measurements rather than hard-coded here.

For each logical tile, summing its value counts must equal the sum of
`manifest.n_points` over all physical shards for that tile. On exact level 0,
summing each value over all tiles must equal `values.parquet.n_points`. Sampled
levels reconcile to their own manifest totals but are not expected to preserve
exact per-value totals.

Exact and sampled bucket finalizers emit flat intermediate count files while
their tile rows are already available. A finalizer may use a transient
per-tile `{value_id: count}` mapping, but it appends those counts to its
bucket-local intermediate file and releases the mapping rather than retaining
one Python object or dictionary per sparse count across the level. Level results
retain only small intermediate-file descriptors. This avoids both an additional
scan of the canonical source or completed levels and unsafe concurrent appends
from independent bucket finalizers into one shared Parquet file. The
intermediate counts are exact, but each file covers only one finalization unit
and is not the complete, globally value-ordered and reconciled index. C7 reads
the intermediate files in bounded batches, sorts and validates the complete
index, writes `tile_value_counts.parquet`, and then removes the intermediate
files. Because every tile belongs to exactly one bucket, its finalizer must
aggregate across any physical tile shards and emit each nonzero
`(level, value_id, tile_x, tile_y)` key exactly once. C7 rejects duplicate keys
rather than silently combining them.
Dask shuffle files are separate execution scratch and may be removed as soon as
their bucket is finalized.

The index answers which tiles and levels contain selected values and how many
selected representatives they contain. It does not locate point rows inside a
mixed-value row group. The initial point files remain tile-co-located rather
than value-sharded.

### `manifest.parquet` contract

One row describes one physical Parquet row group:

```text
level: int16
level_file: string
tile_x: uint32
tile_y: uint32
n_points: int64
row_group: int32
tile_shard: int32
```

The initial manifest contains exactly this logical column set. It does not
contain `tile_id` or `schema_version`:

- `tile_id` is derived as `f"{level}/{tile_x}/{tile_y}"` from the manifest's
  numeric tile key;
- `schema_version` is stored once in `metadata.json`, which owns the schema
  version for the complete cache generation.

Repeating either value in every manifest row adds storage without adding
information. Gate D freezes the remaining Arrow details, including nullability
and metadata policy, but does not reopen these two exclusions. Adding either
column later requires an explicit cache-format revision.

Requirements:

- every row group contains rows for exactly one logical tile;
- one tile may have several row-group shards;
- manifest rows are deterministically ordered;
- shard numbering is deterministic;
- all `level_file` values are cache-root-relative;
- summing `n_points` for a tile equals the rows read from all of its shards;
- summing `n_points` for a level equals the level count in metadata;
- no manifest row points outside the completed cache root.

The manifest should be sorted/indexable by `(level, tile_y, tile_x,
tile_shard)`. Whether the whole compact manifest is loaded into memory or
queried as an Arrow dataset is decided from benchmarked manifest size.

### Level point-payload contract

Every exact and sampled level uses this physical per-point payload:

```text
x_rel: float32
y_rel: float32
value_id: uint32
point_id: uint64
```

`tile_id`, `tile_x`, and `tile_y` are not point-payload columns. Every row group
contains one logical tile, and its manifest row supplies `level`, `tile_x`, and
`tile_y`; `tile_id` is derived from that numeric key. A reader reconstructs
global coordinates from the manifest tile origin plus `x_rel` and `y_rel`.
Repeating tile identity for every point would add storage and decoding work
without adding information. Changing this contract later requires an explicit
cache-format revision.

The first cache format stores tile-local `x_rel` and `y_rel` as `float32`.
Validation and tile assignment operate from `float64` working coordinates and
`float64` source bounds; the writer subtracts the tile origin before converting
the relative values. Global point coordinates are not stored as per-row
`float32` values.

Raw tile-local `float16` is not part of the initial format. Its quantization
step grows with tile size and it cannot represent relative values above 65,504,
which is insufficient for a possible approximately 100,000-unit overview tile.
It may also be expanded to `float32` by the initial rendering backend, removing
the intended bandwidth benefit. A later measured optimization may use
level-restricted `float16` or normalized `uint16`, but requires an explicit
format version and renderer-quality evidence.

`point_id` is the Harpy-owned identity generated from deterministic source-file
offsets and row positions before sampling and propagated unchanged through
every level. It is never written back to canonical SpatialData. The renderer
does not upload it to the GPU unless a measured picking design requires that.

Tile-local coordinates reconstruct native coordinates:

```text
x = x_origin + tile_x * tile_size + x_rel
y = y_origin + tile_y * tile_size + y_rel
```

The exact level means full membership, not full-precision coordinate storage.
Canonical full-precision coordinates remain in `points.parquet`.

Optional future columns such as representative weight, cell id, or quality
metrics require explicit schema evolution. Do not copy arbitrary source columns
into every visualization level by default.

## Level and sampling construction

### Level discovery

The builder considers both:

- spatial extent relative to the requested leaf tile size;
- source point count;
- the explicit initial capacity progression;
- the global overview budget.

It must create at least:

- one sampled coarsest level when exact source count exceeds the overview
  budget;
- one exact finest level.

For a small source whose entire exact representation is within the overview
budget, one exact level is sufficient.

Otherwise, the initial builder creates:

1. the exact 512-unit level;
2. the sampled 512-unit bridge with capacity 4,096;
3. 1,024-, 2,048-, and 4,096-unit spatial levels with capacities 8,192,
   16,384, and 32,768 respectively;
4. further spatial levels following the same edge-doubling and initial
   capacity-doubling rule;
5. termination at the first complete level whose conservative point-count
   upper bound satisfies `overview_point_budget`; if necessary, a one-tile
   terminal level whose effective per-tile capacity is capped by that budget.

The format continues to support other density-only levels, but they are not
part of the initial default schedule and require benchmark evidence before
being added.

### Stable internal point identity

Sampling begins by assigning every source row a Harpy-owned `uint64 point_id`:

```text
point_id = source_file_row_offset + row_position_within_file
```

The policy name is `harpy-source-file-row-offset-uint64-v1`.

V3 exposes that complete name as the `POINT_ID_POLICY` string constant but does
not add a scalar row-to-id helper. Such a helper would have no production
consumer and must not be called once per source row. The exact-level writer owns
the batch-oriented implementation: it combines `source_file.row_offset` with a
batch's zero-based start and row count, then materializes a bounded NumPy or
Arrow `uint64` array. A shared batch helper should be introduced only if the
concrete writer demonstrates a need for one.

Validation establishes deterministic dataset-relative source-file ordering and
row offsets. The builder generates ids batch by batch without materializing a
full-source identity array. Reproducibility is guaranteed only while the
validated file inventory and row order remain stable; the versioned source
signature and point-identity policy record that scope.

### Sampling contract

The C5a and C5d sampling slices produce one concrete, versioned sampler
specification. They refine, but do not reopen, this fixed algorithm family:

1. start from candidates retained by the next finer level;
2. annotate candidates with the current level's tile and spatial stratum;
3. allocate the tile target proportionally across occupied spatial strata using
   integer largest-remainder allocation with deterministic remainder ties;
4. rank candidates by ascending stable pseudo-random priority derived from
   level, spatial stratum, `point_id`, and a seed;
5. keep the first candidates in that ascending order, up to each allocation;
6. sort output deterministically before writing.

Every sampled logical tile uses a fixed 16 × 16 microgrid relative to that
current output tile. The intrinsic cell edge therefore scales with tile
geometry: 32 for the 512-unit bridge, 64 for L1, 128 for L2, and 256 for L3.
Immediate-finer tiles remain the manifest and IO units used to assemble a
coarser spatial tile's candidates, but they are not sampling strata. Their
retained coordinates are rebased into the coarser tile, and the combined
candidates are assigned directly to that tile's 16 × 16 microgrid.

Sampling randomness comes only from
`harpy-value-neutral-stratified-splitmix64-v1` with the initial fixed uint64 seed
zero. Sequentially mix a sampling-domain constant, seed, serialized level,
current tile key, stable stratum key, and `point_id`. A stateful random-number
generator, input row or partition arrival order, and Python's randomized
`hash()` must not affect membership. C5a freezes the shared current-tile
microgrid, priority payload encoding, integer allocation, collision
tie-breaking, and final deterministic sort. C5d approves immediate-finer tile
assembly and coordinate rebasing into the coarser tile before the same sampler
is invoked.

For stratum count `n_i`, total candidate count `N`, and target `K`, allocate
`(K * n_i) // N` representatives first, then assign remaining slots by descending
`(K * n_i) % N` with deterministic stratum-priority and numeric-key ties. Since
`K <= N`, this allocation cannot exceed an observed stratum count and requires
no unconsumed-allocation redistribution phase.

Coordinates assign points to strata before this calculation; the calculation
then allocates sample slots in proportion to each stratum's observed point
count. It does not allocate an equal number to every occupied stratum. The
microgrid consequently preserves approximate local spatial abundance while
reducing random within-tile clumping; it is not intended to flatten spatial
density or guarantee representation of every occupied cell.

`value_id` is deliberately excluded from target allocation and the priority
payload. The sampler must not impose a rarity modifier, minimum per-value
allocation, or another categorical reweighting. Values remain in the retained
point payload and are counted per tile and level for runtime planning, but they
do not influence which representatives win.

The sampler must define behavior when:

- occupied spatial strata exceed the available target;
- all points share one coordinate;
- one value dominates a tile;
- many singleton values occur in one tile;
- a hash collision occurs;
- one logical Exact tile is split across several physical row groups.

The initial bridge builder may load and concatenate every physical shard of one
logical Exact tile before sampling. It processes one complete logical tile at a
time and does not promise a source-size-independent bound for a pathological
tile. C5c measured the densest Xenium tile before Gate C; a two-pass or streaming
top-k sampler is deferred unless that evidence shows the initial assumption is
insufficient.

### Why value neutrality and selection-aware planning are separate

Rarity-aware sampling can prevent rare values from disappearing, but it also
overrepresents them relative to common values. That makes an all-values sampled
view a poorer qualitative representation of local abundance. The initial cache
therefore keeps sampling value-neutral and treats every sampled view explicitly
as an LOD representation rather than an abundance measurement.

The usability problem is handled without changing sample membership. Sparse
per-level tile/value counts let the runtime detect where selected values exist,
estimate their selected render count, and choose an appropriate level. A value
missing from one sampled level can cause a switch to a finer affordable level;
it is not automatically interpreted as absent. If later measurements show that
reading mixed-value row groups prevents an acceptable exact or fine selection,
physical value sharding may be added as a new format optimization without
replacing the value-neutral sampling contract.

## Physical Parquet layout: tile-co-located bucketed shuffle

Source partitions and source-file ordering must not determine cache locality.
A partition-local writer, previously called Layout A, is rejected as the
production direction: a shuffled source could scatter one logical tile across
nearly every output file and make routine viewport reads depend on many file
opens and small row-group reads.

For an arbitrarily ordered source, this guarantee necessarily requires a full
logical redistribution of the exact-level rows. A direct one-pass Parquet writer
cannot append later rows to an already completed row group, while retaining an
unfinished buffer for every logical tile would be unbounded. The implementation
therefore does not claim to avoid the shuffle; it makes that shuffle local,
disk-backed, bounded, and deterministic at the final cache boundary.

The production requirement is the tile co-location property previously called
Layout B:

- all rows for one logical tile are redistributed to one deterministic writer
  bucket;
- one ordinary tile is co-located in one final bucket file;
- a tile exceeding `max_rows_per_row_group` uses a deterministic sequence of
  row groups or physical shards in that same bucket;
- every row group contains exactly one logical tile;
- one bucket file may contain row groups for several logical tiles;
- source partition boundaries have no influence on the final tile locality.

Deterministic tile buckets, previously described separately as Layout C, are the
initial bounded implementation of this Layout B requirement rather than a
competing physical layout. The initial writer uses Dask's local disk shuffle
plus Arrow finalization. A direct-PyArrow alternative is investigated only if
the measured Dask implementation exposes a concrete limitation. The
engine-independent construction flow is:

```text
read and annotate bounded batches from the validated physical inventory
→ calculate tile_x and tile_y
→ generate point_id and map value_id
→ calculate a deterministic integer bucket_id
→ redistribute all rows into local disk-backed bucket storage
→ group or sort each complete bucket by (tile_y, tile_x, point_id)
→ write one or more row groups per tile
→ record each row group in the manifest
```

The bucket mapping uses the versioned method
`harpy-tile-splitmix64-v1`; Python's built-in `hash()` is not suitable. Pack the
non-negative `uint32` tile coordinates into one collision-free `uint64` key and
apply the SplitMix64 finalizer:

```python
tile_key = (uint64(tile_y) << uint64(32)) | uint64(tile_x)

z = tile_key + uint64(0x9E3779B97F4A7C15)
z = (z ^ (z >> uint64(30))) * uint64(0xBF58476D1CE4E5B9)
z = (z ^ (z >> uint64(27))) * uint64(0x94D049BB133111EB)
tile_hash = z ^ (z >> uint64(31))

bucket_id = tile_hash % uint64(bucket_count)
```

All addition and multiplication use explicit modulo-`2**64` wrapping. The
mapping has no runtime-random seed, so identical tile coordinates and
`bucket_count` produce identical placement across processes and platforms. All
points in one tile receive the same bucket. This method controls only physical
bucket placement and is not implicitly reused for the separately versioned
sampling-priority hash.

Tiles and rows receive deterministic final ordering, with `point_id` as the
within-tile tie-breaker.

Bucket filenames use a minimum three-digit width, expanding only when required:

```python
filename_width = max(3, len(str(bucket_count - 1)))
filename = f"bucket-{bucket_id:0{filename_width}d}.parquet"
```

The point bucket is stored below `levels/level_<level>/`; its intermediate
tile/value-count file uses the same filename below
`intermediate_tile_value_counts/level_<level>/`. Empty buckets create neither
file. Point buckets use Snappy compression and dictionary encoding only for
`value_id`; intermediate count files use Snappy without dictionary encoding.

### Initial engine: Dask disk shuffle plus Arrow finalizer

Dask is a leading implementation candidate because it provides an on-disk
single-machine shuffle, not merely because the legacy writer uses Dask or it is
already installed. In this candidate, Harpy constructs the dataframe itself from
`ValidatedPointsSource`; it does not accept or inspect an arbitrary caller graph.
The initial source graph contains one partition per validated physical Parquet
file. For every file, Harpy creates a separate selected-column read with
`split_row_groups=False`, attaches that file's validated provenance, and then
concatenates the annotated reads. This uses Dask's public Parquet API for
decoding while preserving the file row offsets needed for deterministic
`point_id` construction.

The writer intentionally does not reuse the SpatialData points dataframe even
when it currently happens to have the same partition count: a caller may have
filtered, repartitioned, or shuffled that graph. It also does not advertise an
arbitrary source-row partition limit. Input partition size is initially set by
the physical source files, and the decoded row count of every partition is
reconciled with its validated file record.

For the initial Xenium source, the validated 65 Parquet files therefore produce
65 Dask input partitions. Its 168 physical row groups remain internal to the
file reads. The acceptance benchmark decides whether these file-aligned
partitions have acceptable peak memory. If they do not, the first refinement is
row-group-aligned Dask reads with `split_row_groups=True` and explicit row-group
provenance for unchanged point IDs—not an arbitrary row-range reader and not an
automatic reason to replace Dask with direct PyArrow.

With `B` integer buckets, the intended partitioning is equivalent to:

```python
bucketed = annotated.set_index(
    "bucket_id",
    divisions=list(range(B + 1)),
    shuffle_method="disk",
    drop=False,
)
```

Explicit divisions avoid a quantile-discovery pass and make output partition
`i` correspond to bucket `i`. Dask shuffle arrival order is not part of the
cache contract; the final `(tile_y, tile_x, point_id)` sort establishes the
deterministic order before Parquet writing.

#### Intermediate Dask contracts

The names `annotated`, `bucketed`, and `bucket file` refer to distinct stages:

```text
validated physical source
→ annotated: file-partitioned lazy Dask dataframe
→ bucketed: bucket-partitioned lazy Dask dataframe
→ ordered bucket: one computed and deterministically sorted output partition
→ bucket-<id>.parquet: final persistent level file
```

`annotated` is not a stored cache artifact. Its partitions still correspond to
Harpy-owned complete-file reads from the validated source inventory. Its
minimum hot columns are:

```text
tile_x
tile_y
x_rel
y_rel
value_id
point_id
bucket_id
```

Additional source-provenance columns may exist only while constructing
`point_id`; they are removed before final level writing. No serialized per-row
`tile_id`, `tile_x`, or `tile_y` is created during annotation or finalization.

`bucketed` is also not a stored cache artifact. It is the lazy result of the
disk-shuffle graph. When that graph executes, every annotated input partition is
split into temporary fragments by `bucket_id`; Dask's local shuffle storage
collects all fragments for bucket `i` into output partition `i`. Those temporary
fragments are internal, disposable shuffle data and are not Parquet files named
`bucket-<id>.parquet`.

For example:

```text
annotated source partition 0: A1(bucket_id=7), B1(bucket_id=3)
annotated source partition 1: C1(bucket_id=5), A2(bucket_id=7)
annotated source partition 2: B2(bucket_id=3), A3(bucket_id=7)

computed bucketed partition 7: A2, A3, A1
```

The shuffle guarantees that all of tile A is in output partition 7, but does not
guarantee arrival order. The bucket finalizer sorts the computed partition by
`(tile_y, tile_x, point_id)`, producing contiguous, deterministic tile runs:

```text
ordered bucket 7: A1, A2, A3, ...other complete tiles in bucket 7...
```

Only then does a `ParquetWriter` create the persistent file. It processes each
contiguous logical-tile run in deterministic order, splits the run by
`max_rows_per_row_group`, writes each resulting shard as one row group, and emits
the corresponding manifest row:

```text
bucket-007.parquet
  row group 0: complete tile A, or tile A shard 0
  row group 1: tile A shard 1, or the next complete tile
  ...
```

Thus, a Dask shuffle bucket is a temporary logical output partition; a final
bucket Parquet file is created only by the subsequent grouped writer.

### Optional follow-up: direct PyArrow spill and compaction

The focused alternative uses the Phase 0 physical inventory and PyArrow batches
directly:

```text
read bounded PyArrow batch
→ calculate the numeric exact-level payload and bucket_id
→ partition the batch indices by bucket_id
→ append bounded temporary fragments to deterministic bucket spill storage
→ compact one complete bucket through bounded grouping or sorting
→ write final Parquet row groups and provisional level-manifest rows
```

This alternative still performs the required full logical redistribution, but it
avoids reconstructing the source as a Dask/Pandas execution graph. It must define
bounded file-handle use, temporary-fragment consolidation, oversized-bucket
handling, deterministic ordering, concurrency, single-owner bucket output, and
cleanup.

Direct PyArrow is not implemented in parallel with the initial Dask writer. It
is opened as optional C4 only when C3's small acceptance benchmark records a
specific Dask limitation and a measurable success criterion for the
alternative. Do not expand the investigation to DuckDB, Polars, Spark, or other
new execution dependencies.

Bounded source reads alone do not guarantee bounded finalization memory because
the shuffle may collect points from every source file into one output bucket.
Each bucket is an independent finalization unit: one finalizer materializes,
sorts, writes, and releases one complete output bucket.

The pragmatic first implementation uses a deterministic target of 2,000,000
points per physical output bucket on average:

```python
target_rows_per_output_bucket = 2_000_000
bucket_count = max(
    1,
    math.ceil(
        level.point_count_upper_bound / target_rows_per_output_bucket
    ),
)
```

The target is an internal construction default rather than a public builder
argument. For Exact, `level.point_count_upper_bound` equals the validated source
row count. For the 136,578,750-point Xenium source, the calculation therefore
produces 69 buckets, averaging approximately 1.98 million points per bucket
before hash skew, and at most 69 Exact-level bucket files. Empty buckets do not
create files. `dask_worker_count=1` remains the initial benchmark setting.

The output-bucket count is intentionally independent of the 65 input Parquet
files. There is no one-to-one mapping after the shuffle: each source file may
contribute to many output buckets. Physical source packaging is not a logical
cache or memory property and must not determine cache layout.

Using `ceil` guarantees only that the average does not exceed the two-million-
row target. It is not a hard per-bucket limit because hash skew or one very
dense tile may still produce a larger bucket.

`target_rows_per_output_bucket` is a physical file-packing and finalization
target; it is not a logical tile capacity. The bridge and every coarser sampled
level independently derive their bucket count from their own planned
`point_count_upper_bound`, while `max_points_per_tile` continues to control the
representatives retained in each logical tile. Coarser levels therefore use
fewer output buckets as their complete-level upper bounds decrease, normally
reaching one bucket for the overview.

The independent physical row-group maximum is initially:

```python
max_rows_per_row_group = 1_000_000
```

Every row group contains exactly one logical tile. A tile containing more than
1,000,000 points is divided into deterministic row-group shards, while one
bucket file may contain multiple row groups from one or several tiles. This
does not conflict with `target_rows_per_output_bucket`: the row-group setting
limits one tile shard, whereas the bucket target describes the average total
rows packed across the complete output file.

This first implementation does not add recursive bucket spilling or an external
bounded sort and therefore does not claim a strict worst-case finalization-
memory guarantee. Dense tiles are still divided into deterministic output row
groups after sorting, but output row-group sharding does not reduce the memory
needed to materialize and sort their bucket. The acceptance benchmark records
bucket-size skew and peak RSS; only a demonstrated problem justifies changing
the bucket count or concurrency or implementing an oversized-bucket fallback.

The C3 Dask implementation and its small acceptance benchmark must select
practical values and algorithms for:

- correct implementation of `harpy-tile-splitmix64-v1` and deterministic
  bucket/file names;
- `target_rows_per_output_bucket=2_000_000` and its resulting 69-bucket Xenium
  Exact acceptance configuration;
- `max_rows_per_row_group=1_000_000` as the physical tile-shard limit;
- Dask partition and shuffle configuration;
- observed average and maximum bucket size and peak finalization memory;
- dense-tile row-group and shard creation;
- deterministic single-owner bucket output under the local no-task-retry
  execution contract;
- one-at-a-time bucket finalization for the initial benchmark;
- whether measured results justify future file rollover, higher concurrency, or
  a strict oversized-bucket mechanism.

Measure:

- total build time and peak memory;
- shuffle and temporary spill volume;
- total disk size and temporary peak disk usage;
- average and maximum output-bucket rows and bytes;
- largest logical-tile row count;
- finalization throughput and peak memory at the evaluated bounded concurrency
  settings;
- number of final files, row groups, and manifest rows;
- row groups and files touched per logical tile;
- cold and warm single-tile latency;
- viewport latency for small, medium, and large views;
- behavior on local SSD and, if relevant, networked storage.

Gate B accepts Dask when its correctness and benchmark results are sufficient.
Only otherwise does it open optional C4, with the named limitation and success
criterion recorded before direct-PyArrow work begins. Neither outcome reopens
partition-local Layout A as a production fallback. Different physical source
orders may still produce different Harpy `point_id` values under the initial
source-row identity policy, so canonical here means deterministic tile-local
organization for one validated source, not byte-identical caches after source
rows are reordered.

Gate B accepted Dask on 2026-08-07 after one complete Xenium Exact build. The
writer processed all 136,578,750 points into 69 nonempty bucket files and 7,294
tile-owned row groups in 57.92 seconds. Peak process RSS was 3.69 GiB, with 3.36
GiB incremental RSS above baseline; peak benchmark workspace usage was 1.74
GiB. Exact point files occupied 1.65 GiB and intermediate tile/value-count
files occupied 0.08 GiB. Buckets averaged 1,979,402 points and the largest
contained 2,547,160 points, which remained acceptable for the measured memory
envelope despite exceeding the average two-million-row target through ordinary
hash skew.

An independent output pass verified all 136,578,750 point IDs as unique and
complete in 12.69 seconds. A warm representative read loaded a 108,598-point
tile from one row group in 1.15 ms. Every source partition satisfied the
float32 tile-relative coordinate tolerance of `6.103515625e-05` intrinsic
units. The run does not justify recursive spilling, external sorting, file
rollover, a different bucket heuristic, or optional C4. All generated benchmark
artifacts were removed after recording these measurements.

## View planning

### Viewport calculation

For each plan:

1. obtain the visible canvas rectangle in world coordinates;
2. inverse-transform its corners through the transcript layer transform;
3. construct a conservative data-coordinate AABB;
4. intersect it with cache bounds;
5. compute core tiles;
6. add the configured prefetch halo.

For rotation or shear, the inverse-transformed viewport is a polygon. Querying
its AABB may load extra tiles but must never omit visible data.

Axis order must be explicit at every boundary:

- cache metadata and Parquet columns use `x`, `y`;
- napari coordinate arrays use `y`, `x`;
- transforms declare their input and output axes;
- `TilePayload.positions_yx_local` is named accordingly.

### LOD selection

For every candidate level:

- find intersecting core and prefetch tiles;
- for all values, sum manifest counts;
- for an explicit value selection, query tile/value counts, skip zero-count
  tiles, sum selected representative counts for the render estimate, and sum
  complete manifest rows for the initial physical IO estimate;
- evaluate sampling spacing against data-units-per-screen-pixel;
- evaluate render and physical row-read budgets;
- evaluate explicit-selection coverage against finer and exact counts.

Choose the finest eligible level. Exact wins whenever its core selected-render
count and complete-row-group IO cost fit. A selection change triggers immediate
replanning with the same hysteresis and atomic-snapshot rules as a camera or LOD
change.

Do not apply normal query-time random trimming after choosing a level. A level
is a trusted deterministic representation. If no level satisfies the supported
runtime budget, use the coarsest level and report a cache/build configuration
error rather than silently changing sampling semantics.

### Hysteresis

Switching thresholds must include hysteresis based on both screen scale and
point budget. Small camera changes must not alternate between adjacent levels.

## Scheduling and cache policy

### Request priorities

Order work approximately as:

1. missing core tiles for the pending snapshot;
2. same-level tiles entering the current viewport;
3. core tiles needed to improve from fallback to target LOD;
4. prefetch halo tiles;
5. speculative work, if ever enabled.

Priorities are recomputed when a new generation arrives.

### Cancellation and stale results

Cancellation is cooperative. A read already executing inside PyArrow may
finish. On completion:

- validate cache generation and source signature;
- allow a still-useful payload into the CPU cache;
- never activate it for an obsolete render generation.

### CPU cache

The CPU LRU is bounded by decoded bytes, not tile count.

It stores immutable renderer-independent `TilePayload`s and never stores VisPy
objects. A filtered payload key includes the cache generation, logical tile,
decode contract, and `selection_key`. An optional unfiltered decoded-tile cache
uses the same identity without `selection_key` and may be reused to derive
several filtered payloads. Active and pending core tiles are pinned. Prefetch
tiles may be evicted before core tiles.

Sharing a CPU cache across multiple canvases is allowed only when keys include
the full cache generation and decode contract.

### GPU cache

The GPU cache belongs to one OpenGL context/canvas. Its payload identity includes
the cache generation, logical tile, level/decode contract, and `selection_key`.

It is byte-bounded and pins:

- active snapshot tiles;
- pending core tiles;
- buffers currently uploading.

Eviction and deletion happen only with the correct GL context active.

### Upload metering

GPU uploads are queued and limited per frame by bytes and/or elapsed time.
Large bursts must not freeze camera interaction. Upload completion notifies the
scheduler so pending snapshots can activate.

## Value selection, palette, and filtering

The GPU receives a dense value id per resident point and a small lookup resource:

```text
value_id -> RGBA + enabled
```

Changing:

- value color;
- global opacity;
- point size;
- selected-value highlighting

must not reupload point coordinates.

Those are style-only changes over the current active selection. Changing the
active set of selected value IDs is a data-selection change: it immediately
replans the level and required tiles, may read and filter point rows, and
atomically replaces the active coordinate payload when the pending snapshot is
ready.

For all-values rendering, the runtime reads the manifest-selected tile shards.
For an explicit selection, it first uses `tile_value_counts.parquet` to omit
zero-count tiles and estimate the level. For each positive tile, the first
production reader may read all values in the tile's mixed-value row-group
shards, filter `value_id` in CPU/Arrow, and upload only selected rows. This keeps
the GPU render budget tied to selected rows even though physical IO initially
depends on complete row groups.

Physically value-selective reads require a later row-group layout or point index;
the tile/value count index deliberately does not claim to provide that. A
byte-bounded decoded-tile CPU cache may make repeated selection changes cheap
without changing the on-disk layout.

## Picking

Initial picking is CPU-side and separate from rendering:

1. convert the cursor from world to cache data coordinates;
2. identify the active tile;
3. search an optional small spatial index over its CPU payload;
4. ignore disabled values;
5. return `point_id`, value, coordinates, and LOD status.

At a sampled level, the result is a real representative transcript and must be
reported as sampled. At the exact level, it is an exact visible source row.

Picking is not required for the first rendering spike, but the payload contract
must retain enough identity to add it without rebuilding the cache.

## Napari integration boundary

Napari currently lacks a stable public plugin API for mapping an arbitrary
custom Layer subclass to custom Qt controls and a custom VisPy layer.

All private napari integration must be isolated in a narrow adapter such as:

```text
napari_harpy/viewer/_napari_transcript_registration.py
```

The rest of the cache, store, planner, scheduler, and backend protocol must be
testable without importing napari private modules.

The tiled feature needs an explicit napari compatibility policy. The project's
broad existing `napari>=0.4.18` dependency cannot imply that a private custom
renderer is supported across every historical minor version.

Before product integration:

- select the napari minor versions supported by the tiled renderer;
- add compatibility tests for those versions;
- feature-detect or fail clearly outside that range;
- track upstream custom-renderer registration work;
- align scheduler concepts with napari progressive-loading work without making
  an experimental image/labels implementation a runtime dependency.

## Implementation phases

### Phase 0: physical source resolution and validation

Deliverables:

- create `core/multi_scale_cache_points/` as the new implementation home;
- implement immutable unresolved and validated source models;
- resolve backed SpatialData points elements to explicit Parquet datasets;
- implement deterministic Parquet file inventory and source-file offsets;
- construct the private metadata-backed source inventory and validate the selected
  schema;
- implement one bounded PyArrow scan over the selected point data;
- build the normalized value table efficiently;
- implement and version the source-signature method;
- implement and version the internal point-identity policy;
- measure stage time, decoded bytes, and peak memory in dedicated benchmark
  tooling without widening the public validation result;
- add tiny, adversarial, and real-data validation tests.

Exit criteria:

- the implementation does not import `_transcript_tiles.py`;
- the fast path never reverse-engineers a Dask graph;
- the validation package imports neither Qt, napari, nor VisPy;
- missing or incompatible Parquet sources fail before cache writing;
- metadata-only facts avoid full scans;
- all streaming reads have bounded batch sizes;
- internal point ids can be derived deterministically from validated source-file
  offsets;
- the public validation API accepts no caller-supplied identity column;
- the initial validation API exposes no progress or cancellation protocol;
- the Xenium acceptance dataset reports 136,578,750 rows and 5,122 normalized
  values;
- repeated validation produces the same ordered inventory and source signature.

The Phase 0 review sequence is defined by the validation roadmap:

- Gate C follows V5 and freezes the functional validation contract;
- V6 benchmarks, profiles, and hardens that implementation on the Xenium
  acceptance source;
- Gate D follows V6 and is the go/no-go decision for beginning the exact-level
  cache writer.

A V6 finding that requires a functional semantic change reopens the affected
Gate C decision. Phase 1 does not begin before Gate D.

### Phase 1: persistent cache construction

The implementation is divided into independently reviewable slices in
[persistent_cache_construction_5_8_26.md](persistent_cache_construction_5_8_26.md).

Gate A approved C1's private immutable IO-free level plan. C2 froze the minimal
exact-writer contracts, and C3 implemented and benchmarked the Dask Exact writer:

- use the agreed initial 512-unit exact tiles on the Xenium acceptance dataset;
- use the locked four-column numeric point payload without per-row tile columns;
- implement the tile-co-located Layout B contract through deterministic writer
  buckets;
- use Dask local disk shuffle plus Arrow finalization as the initial engine;
- group or sort each completed bucket deterministically before final Parquet
  writing;
- make source partition boundaries irrelevant to final tile locality;
- use one-at-a-time finalization and verify the resulting measured memory and
  temporary-disk envelope;
- run only through a Harpy-controlled local threaded or synchronous scheduler,
  without distributed execution, automatic task retries, or speculation;
- give exactly one finalizer ownership of each deterministic bucket path and
  reject the complete staging generation on any finalizer failure;
- retain the row-group-per-logical-tile manifest invariant.

Attempt-local bucket files, coordinated winner installation, task-retry
idempotence, and reuse of incomplete staging output are not Phase 1
requirements. They are deferred until Harpy deliberately supports a distributed
or resumable builder, automatic retries or speculation, multiple writers for
one staging generation, or object-store publication. A failed initial build is
restarted by creating a fresh staging generation.

Then complete the cache builder:

- define new dataclasses and cache schemas only after their owning review gates;
- accept the returned `ValidatedPointsSource` as the cache-construction input;
- freshly recompute the metadata-only source signature and require it to match
  the validated signature before creating the staged cache;
- generate a fresh cache-generation id and consume the validated source
  signature;
- generate and propagate stable internal `point_id` values;
- write the exact level from the validated source;
- implement the initial 512-all → 512-at-4,096 → 1,024-at-8,192 →
  2,048-at-16,384 → 4,096-at-32,768 construction schedule;
- construct sampled levels from retained finer-level candidates or normalized
  exact tiles, not by repeatedly rescanning the original source;
- implement value-neutral, spatially stratified nested sampled levels;
- enforce the global coarsest-level budget;
- emit nonzero tile/value counts during exact and sampled tile finalization;
- write metadata, manifest, values, and the sparse tile/value count index;
- validate the complete staged cache;
- freshly recompute the metadata-only source signature again and require it to
  match the validated signature immediately before completion and publication;
- publish with completion marker and atomic replacement;
- expose the public backed-points-element builder.

Exit criteria:

- exact coordinate reconstruction is within documented tolerance;
- exact level has full membership and identity coverage;
- generated non-terminal levels have the required geometry and capacity
  progression;
- every sampled level is deterministic and nested;
- coarsest total never exceeds the overview budget;
- manifest accounting matches physical row groups;
- tile/value counts reconcile to every logical tile, and exact per-value counts
  reconcile to `values.parquet`;
- the builder performs no second source-content validation pass and trusts
  `ValidatedPointsSource` while both source-signature guards pass;
- a source-signature mismatch before or during construction publishes nothing
  and preserves any existing completed cache;
- rebuilding cannot expose an incomplete cache;
- building never writes an incomplete cache at the final visible path;
- full build time, peak memory, level sizes, and fragmentation are recorded on
  the Xenium acceptance dataset.

An exact-only artifact may be used internally for performance work, but it is
not published as a completed multiscale cache when the source exceeds the
overview budget.

### Phase 2: runtime store, planner, and scheduler

Implement:

- validate metadata, manifest, files, row groups, and completion marker;
- expose cache freshness state;
- build the manifest lookup;
- build the sparse per-level tile/value count lookup;
- load and combine physical row-group shards with PyArrow;
- skip zero-count tiles for explicit selections, filter complete mixed-value
  row-group reads by selected `value_id`, and return immutable selected payloads;
- perform no Dask computation;
- viewport transform and conservative bounds;
- core and prefetch tile selection;
- screen-scale, selected-render-budget, physical-row-read-budget, and
  selection-coverage-aware LOD;
- hysteresis;
- immutable plans and snapshots;
- generations and stale-result rejection;
- bounded worker concurrency;
- CPU byte LRU and pinning;
- same-level incremental pan behavior;
- atomic cross-level activation;
- lifecycle shutdown.

Exit criteria:

- reader tests do not import Qt, napari, or VisPy;
- real-data cold tile latency meets the measured target;
- deterministic planner tests cover boundary-touching viewports and transforms;
- stale loads cannot activate;
- snapshots cannot mix cache generations or levels;
- snapshots cannot mix active value selections;
- repeated nearby views hit the CPU cache;
- explicit value selections can select a finer affordable level without reading
  tiles known to have zero matching rows;
- a sampled zero is never silently reported as biological absence when exact
  viewport counts are nonzero;
- invisible or removed layers stop scheduling work;
- all scheduler tests run against a fake backend.

### Phase 3: production VisPy backend

Start with an in-memory renderer spike using synthetic immutable tiles. It must
demonstrate:

- resident tile buffers survive pan;
- only entering tiles upload;
- palette and other style-only changes do not upload coordinates;
- active value-selection changes can atomically replace filtered coordinates;
- cross-level activation is atomic;
- upload metering keeps interaction responsive;
- cleanup releases GPU resources.

The spike compares one standard VisPy marker visual per tile with one lean
transcript point visual using compact attributes. The result informs the first
VisPy backend without changing the backend protocol.

Implement:

- transcript tile visual;
- compact vertex payload;
- value palette/visibility lookup;
- per-context GPU LRU;
- GUI-thread upload queue;
- per-frame upload metering;
- active/pending pinning;
- atomic snapshot visibility;
- context cleanup;
- diagnostics for resident points, bytes, queue depth, and latency.

Exit criteria:

- an already resident tile is uploaded at most once before eviction;
- camera transforms do not rebuild resident VBOs;
- style changes do not upload coordinates;
- GPU memory stays within its configured budget;
- LOD changes do not show random mixed-level tiles;
- removal releases resources and disconnects events;
- benchmarks cover supported GPUs and napari versions.

### Phase 4: napari and Harpy product integration

Implement:

- dedicated TranscriptLayerModel;
- private registration adapter;
- layer controls and status;
- active value selection wired to immediate selection-aware replanning;
- viewer widget entry point;
- cache build/rebuild/status workflow;
- tiled-mode controller lifecycle;
- coordinate-system transform integration;
- direct-path fallback;
- user-visible diagnostics and error recovery.

The existing direct points path remains:

- the behavior when no cache is available;
- a small selection workflow;
- a correctness comparison for exact tiled views;
- a fallback when the tiled renderer is unsupported.

Exit criteria:

- opening tiled mode does not materialize the full source dataframe;
- fit-to-view uses the bounded point overview;
- choosing one or more values immediately replans and displays the finest level
  satisfying scale, selected-render, physical-row-read, and coverage
  constraints;
- zoomed exact views match the direct path for the same rows;
- pan and zoom reuse resident GPU tiles;
- layer visibility/removal and SpatialData replacement are safe;
- tiled-mode failures do not corrupt canonical data or the previous cache.

Migration cleanup occurs only after those criteria pass:

- switch all required imports and entry points to the new package;
- inspect legacy tests for useful edge cases and write independently specified
  replacement tests where the new contracts require them;
- verify the direct fallback remains intact;
- remove `src/napari_harpy/_transcript_tiles.py` in a dedicated cleanup change.

### Phase 5: physically value-selective IO and advanced interaction

Only after measured need:

- change physical row-group layout or add value-aware shards;
- read only selected-value row groups;
- add efficient picking and metadata lookup;
- add remote-store publication and reads.

Subset-aware planning, fine/exact selection when budgets permit, and full-row-
group filtering are core Phase 1/2/4 requirements, not deferred here. This phase
only optimizes the physical read amplification if measurements justify the
additional layout complexity. It requires a new schema version if physical row-
group semantics change.

## Test strategy

### Source resolution and validation

Use a small number of focused, multi-row-group fixtures to cover the following
contracts. The bullets are not a requirement for one test per Arrow dtype,
invalid-value variant, encoding combination, or PyArrow guarantee:

- missing, non-Parquet, and unreadable paths;
- deterministic file ordering;
- incompatible file and row-group schemas;
- missing required columns and unsupported physical types;
- metadata-derived and scan-derived row counts and bounds;
- missing, NaN, and infinite coordinates;
- missing, empty, and whitespace-padded normalized values;
- normalization-equivalent raw values merge without collision telemetry;
- dictionary-encoded and plain-string values;
- deterministic source-file offsets and internal point ids;
- repeatable source signatures and inventory-change detection;
- bounded batch reads;
- no dependency on Dask graph inspection;
- no Qt, napari, or VisPy imports.

### Cache writer

Test:

- source-signature mismatch before staging begins;
- source-signature mismatch immediately before publication;
- source-metadata-inspection failure at either construction guard;
- failed guards preserve the existing completed cache and never expose staging;
- construction guards perform no point-data scan or source-content
  reconciliation;
- stable internal point ids;
- tile-boundary coordinates;
- deterministic value ids;
- exact membership;
- coordinate reconstruction;
- deterministic nested sampling;
- spatial-stratum coverage and absence of deliberate `value_id` influence on
  sampled membership;
- sparse tile/value count uniqueness and reconciliation to logical tiles,
  levels, and exact `values.parquet` totals;
- global and per-level budgets;
- dense-tile sharding;
- cross-partition tile co-location semantics;
- reconstruction from the locked point payload and manifest tile key;
- local no-task-retry execution, single-owner bucket paths, and whole-staging
  rejection after a finalizer failure;
- manifest accounting;
- metadata-only source-signature guards;
- first build and rebuild rollback;
- incomplete-cache rejection.

### Store and planner

Test:

- unsupported schema;
- absent/stale/unverifiable/invalid states;
- missing or corrupt files;
- row-group schema mismatch;
- tile intersection at every boundary;
- rotated/sheared conservative bounds;
- YX/XY conventions;
- point-count estimates;
- selected-value point-count estimates and zero-count tile elimination;
- distinct selected-render-count and complete-row-group IO estimates;
- explicit-selection coverage when a value is absent from a sampled level but
  present at a finer or exact level;
- core versus prefetch budgeting;
- exact-level shortcut;
- hysteresis;
- the exact-to-sampled 512-unit bridge;
- the initial 4,096 → 8,192 → 16,384 → 32,768 capacity progression;
- optional density-only levels with equal tile sizes when supplied by a
  non-default cache schedule.

### Scheduler

Test:

- request priority;
- concurrency bounds;
- cancellation;
- generation changes;
- stale completion;
- CPU LRU pinning and eviction;
- same-level overlap reuse;
- atomic cross-level transitions;
- errors retaining the previous display;
- invisible/removed layer shutdown.

### Renderer

Test or instrument:

- GUI-thread-only GPU mutation;
- upload count per tile;
- coordinate buffers unaffected by palette changes;
- atomic coordinate replacement after active value-selection changes;
- point-size and opacity uniforms;
- GPU LRU byte accounting;
- active/pending pinning;
- context cleanup;
- no source or LOD mixing;
- large coordinate precision using tile-local positions.

## Benchmark datasets and metrics

Use at least:

1. a tiny deterministic fixture for exact correctness;
2. a dense compact fixture where extent alone would incorrectly produce one
   LOD;
3. a spatially skewed fixture with hotspots and empty regions;
4. a value-skewed fixture with dominant and rare values;
5. a medium real Xenium or equivalent transcript dataset;
6. a large real or synthetic dataset exercising many tiles and row groups.

Record:

- build time and peak memory;
- size per level and total cache overhead;
- manifest size;
- tile/value count-index rows, size, build overhead, and selected-value lookup
  latency;
- cold and warm tile latency;
- files and row groups touched per viewport;
- planner time;
- CPU cache hit rate and bytes;
- GPU upload bytes and time per frame;
- active/resident point counts;
- same-level pan latency;
- cold and warm LOD transition latency;
- palette/style-only update latency;
- active-value selection replan, filtered-read, and atomic-transition latency;
- resource cleanup.

Initial runtime targets should be treated as hypotheses until measured:

- a hard visible render budget configurable in the 100,000-200,000 range;
- a separate screen-space density target so coarse views are not filled merely
  because unused render budget remains;
- warm pan/zoom median under 100 ms;
- common cold tile view under 300 ms;
- no interaction-blocking upload burst.

The cache format must not hard-code one machine's runtime budget.

## Architecture invariants

These invariants are suitable as tests and review gates:

- camera movement never mutates a canonical transcript dataframe;
- camera movement never replaces one monolithic Points layer in tiled mode;
- camera movement never rebuilds an already resident tile VBO;
- a tile uploads at most once between GPU insertion and eviction;
- palette and style-only changes never reupload coordinates;
- active value-selection changes may replan, read, filter, and atomically replace
  coordinate payloads;
- no render snapshot mixes cache generations;
- no active snapshot mixes LOD levels;
- no active snapshot mixes value selections;
- sampled rows always identify real source transcripts;
- the exact level has full source membership;
- the complete coarsest level fits the declared overview budget;
- interactive reads touch only manifest-selected row groups;
- no interactive read executes a Dask graph;
- active and pending tiles are protected from LRU eviction;
- stale asynchronous results cannot activate themselves;
- worker threads never touch VisPy objects;
- the layer extent is independent of current resident tiles;
- layer removal disconnects camera events and releases GPU resources;
- incomplete caches never validate as usable;
- canonical SpatialData points are never mutated by cache construction or
  rendering.

## Upstream alignment

Relevant upstream and neighboring work:

- napari progressive loading:
  https://github.com/napari/napari/pull/9067
- napari multiresolution non-image layers:
  https://github.com/napari/napari/issues/1019
- napari custom layer-to-visual registration:
  https://github.com/napari/napari/issues/4121
- napari large-points discussion:
  https://github.com/napari/napari/issues/6148
- Neuroglancer precomputed annotation spatial index:
  https://github.com/google/neuroglancer/blob/master/src/datasource/precomputed/annotations.md
- deck.gl TileLayer scheduling and refinement:
  https://deck.gl/docs/api-reference/geo-layers/tile-layer
- Odon point renderer and in-memory LOD reference:
  https://github.com/alexcoulton/odon

Harpy should reuse concepts and align interfaces where practical. It should not
wait for napari to deliver a transcript-specific renderer, and it should not
make experimental napari progressive-loading internals a required dependency.

## Immediate next actions

Phase 0 validation and its Gate D are complete. Do not add the new builder to
`_transcript_tiles.py` or treat its schemas and tests as the Phase 1
specification.

The next work follows the construction companion roadmap. Planning, the Dask
Exact writer, the shared value-neutral sampler, level-neutral physical writer
support, the persistent Bridge writer, and the complete nested spatial writer
are implemented. The one-shot Xenium Bridge build retained 21,722,305
representatives in 25.99 seconds with 76,087,296 bytes of incremental peak RSS.
Gate B accepted Dask, Gate C accepted the finer-to-coarser contract, and
optional direct-PyArrow investigation remains deferred indefinitely unless new
evidence identifies a concrete Dask limitation. Continue with:

1. freeze the published cache artifact schemas and consolidate sparse
   tile/value counts in C7;
2. write and validate metadata, values, manifest, and the complete staged cache
   in C7;
3. implement guarded end-to-end construction and publication in C8;
4. run the complete multilevel Xenium benchmark and hardening in C9.

This order keeps logical cache requirements stable while deferring the physical
writer and the remaining manifest and metadata schema choices until focused
evidence exists.
