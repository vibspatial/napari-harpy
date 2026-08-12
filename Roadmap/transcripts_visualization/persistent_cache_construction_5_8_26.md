# Persistent Points Cache Construction

Status: implementation roadmap for Phase 1 cache construction

Roadmap date: 2026-08-05

## Authority and scope

This document expands Phase 1, persistent cache construction, from
[multi_tile_cache_29_7_26.md](multi_tile_cache_29_7_26.md).

The parent roadmap remains authoritative for the complete multiscale cache,
runtime store, scheduler, renderer, and napari integration. The completed
[validation_cache_29_7_26.md](validation_cache_29_7_26.md) remains authoritative
for source resolution, source signatures, value normalization, and
`ValidatedPointsSource`. If this document conflicts with either contract, the
parent roadmap wins and the conflict must be resolved before implementation.

Phase 0 is complete and Gate D has approved beginning the exact-level writer.
This roadmap covers only the work needed to turn one `ValidatedPointsSource`
into a complete, local, persistent multiscale points cache.

It does not implement:

- runtime cache opening or freshness classification;
- viewport-to-tile planning or LOD selection;
- request scheduling or CPU/GPU caches;
- napari, Qt, VisPy, or GPU rendering;
- runtime value-selection planning, point filtering, or physically
  value-selective tile IO;
- remote/object-store construction or publication;
- distributed Dask schedulers, automatic task retries, or speculative task
  execution;
- resumable construction or reuse of files from an incomplete staging
  generation;
- mutation of the canonical SpatialData points element;
- removal of `_transcript_tiles.py` or its legacy tests.

The canonical `points.parquet` dataset remains read-only. Every cache artifact is
derived, Harpy-owned, and disposable.

## Outcome

Phase 1 must produce a completed cache rooted at:

```text
<sdata.zarr>/points/<points_name>/transcripts_vis/
```

with:

```text
metadata.json
manifest.parquet
values.parquet
tile_value_counts.parquet
levels/
  level_0/
  ...
  level_n/
COMPLETED
```

The completed cache must provide:

- one exact finest level containing every validated source row exactly once;
- deterministic Harpy-owned `uint64 point_id` values;
- tile-local `float32` coordinates with documented reconstruction tolerance;
- self-contained, nested, spatially stratified, value-neutral sampled levels;
- the initial 512-all → 512-at-4,096 → 1,024-at-8,192 →
  2,048-at-16,384 → 4,096-at-32,768 progression;
- further spatial levels when needed to satisfy the global overview budget;
- deterministic `values.parquet`, sparse per-level tile/value counts, metadata,
  manifest, files, row groups, and shard numbering;
- source-signature guards before staging and immediately before publication;
- staged validation and a completion marker;
- local publication that never exposes an incomplete cache and preserves an
  existing completed cache on failure;
- measured build time, peak memory, disk usage, and fragmentation on the Xenium
  acceptance source.

An exact-only artifact may be used internally by implementation spikes. It is
not a completed multiscale cache for a source whose exact representation exceeds
the overview budget.

## Locked inputs from Phase 0

Construction accepts the public immutable `ValidatedPointsSource` and trusts its
content facts while source-signature guards pass. It does not:

- resolve SpatialData again;
- inspect a Dask graph;
- repeat the validation content scan;
- recompute source bounds or the global value table from point data;
- accept a caller-supplied transcript or point identity column.

The builder consumes:

- `source` and its canonical physical Parquet path;
- ordered source files, row offsets, and row-group metadata;
- the selected source schema and columns;
- source row count and exact scan-derived bounds;
- the canonical normalized `value_table`;
- source-signature method and value;
- value-normalization method;
- point-identity policy.

Construction necessarily reads the selected source point rows again to create
the cache. That is a construction pass, not a second validation pass. It must be
bounded and must not independently recreate Phase 0 diagnostics.

## Locked construction contracts

### Source identity and point identity

The expected source signature is always
`validated.source_signature`. A fresh metadata-only signature must match before
staging begins and again after staged-cache validation, immediately before the
completion marker and publication.

The initial point identity policy is:

```text
point_id = source_file.row_offset + row_position_within_file
```

with method name:

```text
harpy-source-file-row-offset-uint64-v1
```

IDs are generated in bounded arrays by the exact writer and propagated unchanged
through every sampled level. Construction never writes them back to SpatialData.

### Point-only, self-contained levels

Every level stores actual source representatives. No level stores invented
centroids or a raster. Serialized levels follow construction from finest to
coarsest:

```text
Exact → sampled finest bridge → L1 → L2 → ... → overview
  0               1             2     3              n

level_n ⊆ ... ⊆ level_2 ⊆ level_1 ⊆ level_0
level_0 = exact source membership
level_n = terminal coarsest overview
```

`L1`, `L2`, and later `L*` names are spatial design labels rather than
serialized level numbers. In an exact-only cache, `n == 0`, and level 0 is both
finest and coarsest.

The runtime will render one self-contained level at a time. Residual or disjoint
levels are not part of the first format.

### Initial level schedule

The initial target is:

| Design label | Tile geometry | Maximum representatives stored per logical tile |
|---|---:|---:|
| Exact | 512 | uncapped: all source points belonging to the tile |
| Sampled finest bridge | 512 | 4,096 |
| L1 | 1,024 | 8,192 |
| L2 | 2,048 | 16,384 |
| L3 | 4,096 | 32,768 |
| Later spatial levels | double preceding edge | initially double preceding capacity |

Capacity is a maximum, not a fill target. Sparse tiles retain all candidates.
Construction stops at the first complete sampled level whose conservative
point-count upper bound satisfies:

```text
coarsest_level = max(level in planned levels)
sum(manifest.n_points where level == coarsest_level) <= overview_point_budget
```

For a source whose exact membership already satisfies the overview budget, one
exact level is sufficient.

### Locked level point payload

Every exact and sampled level uses this physical per-point payload:

```text
x_rel: float32
y_rel: float32
value_id: uint32
point_id: uint64
```

`tile_id`, `tile_x`, and `tile_y` are not point-payload columns. One row group
contains one logical tile, and its manifest row supplies `level`, `tile_x`, and
`tile_y`; `tile_id` is derived from that numeric tile key. Repeating these values
for every point would add storage and decoding work without adding information.

Construction uses numeric tile keys internally. Per-row Python `tile_id` strings
are not part of the hot construction path. Global coordinates can be
reconstructed from the manifest tile key as:

```text
x = x_origin + tile_x * tile_size + x_rel
y = y_origin + tile_y * tile_size + y_rel
```

Arbitrary source attributes are not copied into visualization levels.

### Physical row-group invariant

Every physical level row group contains rows for exactly one logical tile. A
logical tile may have multiple deterministic shards. Manifest rows describe
those physical shards and reconcile to logical-tile and complete-level counts.

The production writer must co-locate a tile independently of source partition
boundaries. All rows for a tile map to one deterministic writer bucket; an
ordinary tile is written into one final bucket file, while a pathological dense
tile may use a deterministic sequence of row groups or physical shards in that
same bucket. A bucket file may contain row groups for several tiles.

This is the tile-shuffled Layout B contract. Deterministic tile buckets are its
bounded implementation, not a separate competing layout. The initial Dask
implementation and its acceptance benchmark select the bucket, shuffle,
grouping, and sharding parameters; they do not compare against partition-local
Layout A as a production candidate.

### Staging and publication

Every local build, including the first, uses a unique sibling staging directory.
`COMPLETED` is written only after all required artifacts have been validated and
the final source-signature guard passes. Failure publishes nothing incomplete
and preserves any previously completed cache.

## Decisions deliberately left for slice refinement

The high-level slices below are ready, but these remaining details must be frozen
at their named review gates rather than guessed during implementation. C7a has
already frozen the new `harpy-multiscale-points-cache-0.1` format identifier and
the complete Arrow and metadata contracts.

- the public build-result model and any later public builder parameters beyond
  the two logical planning arguments;
- any future change to C1's implemented grid-origin normalization or exact
  maximum-boundary behavior;
- whether measured C3 results justify the optional C4 direct-PyArrow
  investigation;
- deterministic bucket filename width; the bucket hash and the initial
  bucket-count heuristic are locked below;
- Dask partition and shuffle configuration, plus direct-PyArrow spill and
  compaction configuration only if optional C4 is opened;
- whether measured C3 bucket skew or peak memory justifies a later maximum
  in-memory bucket size, recursive spill or external-grouping fallback, and
  file rollover;
- C5a approval of proportional allocation, deterministic integer-remainder
  behavior, the shared 16 × 16 sampled-tile microgrid, sampling priority
  payload, seed representation, and collision tie-breaking;
- C5c evidence that one complete logical Exact tile is a viable initial memory
  unit on Xenium;
- C5d approval of immediate-finer tile assembly and coordinate rebasing into
  the coarser tile's shared microgrid;
- bounded worker concurrency and memory limits;
- whether the first public builder exposes progress and cancellation or remains
  a synchronous core API wrapped later by product integration;
- exact local replacement and rollback mechanics on each supported platform.

## Expected package shape

Construction is expected to add modules only as their responsibilities become
concrete:

```text
src/napari_harpy/core/multi_scale_cache_points/
  builder.py
  build_plan.py
  cache_format.py
  hashing.py
  sampling.py
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
implementations without creating another exported API. `writer/support.py` is
introduced only in C5b, when the already implemented Exact physical schemas and
helpers have an agreed second consumer. The future complete manifest and
metadata schemas remain deferred to C7a rather than being added merely to
mirror the legacy module.

C1 owns `build_plan.py` and its private plan records. Small helpers should remain
in their consuming module rather than creating speculative modules. `hashing.py`
becomes concrete in C5a because the Exact bucket mapping and sampling priority
share the same vectorized SplitMix64 transform while retaining separate
versioned method names and payload contracts.

Focused tests remain under:

```text
tests/multi_scale_cache_points/
```

The new implementation may inspect `_transcript_tiles.py` as historical evidence
and a source of possible edge cases, but it must not import it, copy its models
or schemas by default, or treat its tests as the new specification. Every
retained idea requires independent justification from this roadmap.

## Slice overview

| Slice | Status | Deliverable | Reads source point rows | Publishes cache |
|---|---|---|---:|---:|
| C1 | Implemented; Gate A approved | Pure grid and level build planning | No | No |
| C2 | Implemented | Minimal exact-level construction contracts | No | No |
| C3 | Implemented; Gate B approved Dask | Dask exact-level writer and acceptance benchmark | Yes | No |
| C4 | Deferred indefinitely; not justified by Gate B | Direct-PyArrow exact-writer investigation, reopened only for a concrete Dask limitation | Yes | No |
| C5a | Implemented | Generic 16 × 16 sampled-tile contract and pure in-memory sampler | No original-source rescan | No |
| C5b | Implemented | Level-neutral physical writer-support extraction | No | No |
| C5c | Implemented | Persistent bridge-level construction and acceptance check | No original-source rescan | No |
| C5d | Implemented; Gate C approved | Immediate-finer tile assembly and coarser-coordinate-rebasing spike | No original-source rescan | No |
| C6 | Implemented | Complete nested spatial pyramid from the bridge | No original-source rescan | No |
| C7a | Planned | Published cache artifact contracts | No | No |
| C7b | Planned | Artifact writing and tile/value-count consolidation | No | No |
| C7c | Planned; Gate D follows | Staged-cache validation | No | No |
| C8 | Planned | Guarded end-to-end builder and local publication | Through level builders | Yes |
| C9 | Planned | Xenium construction benchmark and hardening | Yes | Benchmark only |

Each slice must be independently reviewable. C3 implements one credible writer
rather than two competing engines. Gate B accepted its measured Dask writer, so
C4 is deferred indefinitely unless new evidence identifies a concrete Dask
limitation and measurable PyArrow success criterion. It does not block C5a or
later work. C5a is the implemented pure sampled-tile selector, C5b has extracted
the now-concrete level-neutral physical writer support, and C5c has implemented
the first persistent sampled level. C5d implements the pure finer-to-coarser
assembly boundary used by later spatial levels, and C6 implements the complete
nested spatial pyramid. C7 is divided into separately reviewable artifact
contracts, artifact writing, and staged validation rather than introducing all
three concerns in one implementation step.

## Slice C1: pure grid and level build planning

### Goal

Convert `ValidatedPointsSource` facts plus two explicit logical planning
arguments into a deterministic, IO-free build plan.

The contract below is implemented in `build_plan.py` and covered by the focused
`test_build_plan.py` module. Gate A reviews the resulting behavior before C2
depends on it.

The agreed eventual public defaults are:

```text
leaf_tile_size = 512
overview_point_budget = 100_000
```

They are applied only by the public builder introduced in C8. The private C1
planner has no defaults and always receives resolved values explicitly:

```python
def _plan_points_cache(
    validated: ValidatedPointsSource,
    *,
    leaf_tile_size: int,
    overview_point_budget: int,
) -> _PointsCacheBuildPlan: ...
```

The C2 construction contracts and all later writers consume immutable records from
`_PointsCacheBuildPlan`; they do not independently reinterpret these logical
arguments or apply their own defaults.

### Grid and boundary contract

All levels share one origin aligned to the leaf grid:

```text
x_origin = floor(validated.bounds.x_min / leaf_tile_size) * leaf_tile_size
y_origin = floor(validated.bounds.y_min / leaf_tile_size) * leaf_tile_size
```

The calculations use Python integer arithmetic for `leaf_tile_size` and
`float64` coordinate/bounds arithmetic. Do not add an epsilon or `nextafter`
adjustment. For every level:

```text
tile_x = floor((x - x_origin) / tile_size)
tile_y = floor((y - y_origin) / tile_size)
```

Cells are half-open. A point exactly on a boundary belongs to the tile beginning
at that boundary. The observed maximum therefore determines:

```text
max_tile_x = floor((x_max - x_origin) / tile_size)
max_tile_y = floor((y_max - y_origin) / tile_size)
grid_width = max_tile_x + 1
grid_height = max_tile_y + 1
```

This rule applies unchanged to negative source coordinates. The shared origin
and edge-doubling schedule keep adjacent finer and coarser grids aligned.

### Private plan contracts

C1 defines these private logical records with the following frozen names:

```python
class _LevelKind(Enum):
    EXACT = "exact"
    BRIDGE = "bridge"
    SPATIAL = "spatial"


@dataclass(frozen=True)
class _LevelBuildPlan:
    level: int
    kind: _LevelKind
    tile_size: int
    grid_width: int
    grid_height: int
    max_points_per_tile: int | None
    point_count_upper_bound: int

    @property
    def relative_directory(self) -> str:
        return f"levels/level_{self.level}"


@dataclass(frozen=True)
class _PointsCacheBuildPlan:
    x_origin: float
    y_origin: float
    leaf_tile_size: int
    overview_point_budget: int
    levels: tuple[_LevelBuildPlan, ...]
```

`kind` describes only the logical role needed to construct the level. C1 does
not record a C5a sampler name, version, or parameters. The relative level
directory is derived from the serialized level; no absolute output or staging
path belongs to either plan record.

For the exact level, `max_points_per_tile` is `None`, and
`point_count_upper_bound` equals `validated.row_count`. A sampled bridge or
regular spatial level has its effective `max_points_per_tile`. This normally
equals the scheduled capacity. The terminal one-tile fallback described below
instead caps it at `overview_point_budget`.

### Level progression and termination

If:

```text
validated.row_count <= overview_point_budget
```

C1 emits only exact level 0.

Otherwise it emits exact level 0 and exactly one same-geometry sampled bridge.
The bridge is the first sampled candidate and participates in the same
upper-bound calculation as every later spatial candidate:

```text
candidate_upper_bound = min(
    finer_level.point_count_upper_bound,
    grid_width * grid_height * max_points_per_tile,
)
```

If the bridge upper bound is at most `overview_point_budget`, the bridge is
terminal and C1 emits no spatial levels. Otherwise C1 emits the
edge/capacity-doubling spatial progression. The first sampled spatial level
whose upper bound is at most `overview_point_budget` is terminal; no extra
overview level is appended. More than one terminal tile is allowed in either
case because the complete-level upper bound already satisfies the whole-dataset
overview budget.

If neither the bridge nor any normally capped spatial candidate satisfies the
budget before the grid reaches one tile, that one-tile candidate becomes the
terminal level with an effective per-tile capacity capped by the overview
budget:

```text
effective_max_points_per_tile = min(
    scheduled_max_points_per_tile,
    overview_point_budget,
)
point_count_upper_bound = min(
    finer_level.point_count_upper_bound,
    effective_max_points_per_tile,
)
```

Because this fallback contains exactly one tile, its per-tile capacity is also
the complete-level capacity. A separate global allocation field would add no
information and would require a more complex multi-tile allocation contract
that the initial builder does not need. This rule guarantees finite planning
without a point-row scan. The terminal condition is represented by the final
position in `levels`; `OVERVIEW` is not a separate `_LevelKind`.

### Expected files

```text
src/napari_harpy/core/multi_scale_cache_points/build_plan.py
tests/multi_scale_cache_points/test_build_plan.py
```

### Implement

- define the private immutable level and complete-cache build-plan records once
  their fields and invariants above pass the pre-implementation review;
- validate that `leaf_tile_size` and `overview_point_budget` are positive
  integers and are not `bool`, raising ordinary `ValueError` otherwise;
- implement the shared aligned origin and exact half-open maximum-boundary rules;
- distinguish tile geometry from sampling capacity;
- create the exact-only plan when source count fits the overview budget;
- otherwise create the exact level, sampled finest bridge, required spatial
  progression, and terminal level whose complete-level upper bound satisfies
  the overview budget;
- assign contiguous serialized levels in construction order: exact is 0, the
  bridge is 1, the first spatial level is 2, and the terminal coarsest level is
  `n`;
- order serialized level records by ascending level from finest to coarsest
  while preserving clear spatial design labels internally;
- record only the specified logical level kind, geometry, effective per-tile
  capacity, count upper bound, and derived relative directory;
- reject impossible integer grid shapes or serialized level identifiers before
  construction starts.

The planner uses only the specified conservative count recurrence. It does not
inspect source rows or predict the exact sampled count of every tile.

Reject planning when:

- either argument is not a positive non-boolean integer;
- a grid width or height exceeds `2**32`, because the largest tile index would
  not fit the cache's `uint32` tile coordinates;
- a serialized level exceeds the non-negative `int16` range;
- a row-count upper bound exceeds the supported non-negative `int64` cache-count
  range.

Use ordinary `ValueError`; do not add a public planning-error hierarchy.

Do not introduce `PointsCacheBuildConfig`, expose the private plan records,
create directories, read source point rows, define Arrow cache schemas, or add
physical writer and publication settings in this slice.

### Focused tests

Cover focused argument validation, a small exact-only source, a bridge-terminal
source, a large source requiring the bridge and several spatial levels, aligned
and non-aligned bounds, an observed maximum exactly on a tile boundary, negative
coordinates, regular spatial upper-bound termination, the one-tile
effective-capacity clamp, and one focused overflow case. Avoid combinatorial
extent/budget tests and do not test Python/dataclass behavior.

### Exit criteria

- identical validated facts and explicit logical arguments produce an identical
  plan;
- the plan contains enough information for the exact writer and sampler without
  consulting a viewport or renderer;
- no filesystem or point-row IO occurs;
- no new public package export is introduced.

## Slice C2: minimal exact-level construction contracts

### Goal

Freeze only the private records and engine-independent boundaries required by
the first exact-level writer. This slice performs no point-row IO and does not
implement or compare writer engines.

The contracts below are implemented in `writer/models.py`, covered by the
focused `test_writer_models.py` module, and remain private package internals.
The existing `max_source_rows_per_partition` field is removed before C3: the
file-aligned source contract below deliberately does not expose a nominal row
limit that the initial reader cannot enforce without splitting physical files.

### Minimal contracts

C2 documents the private callable boundary that C3 implements:

```python
def _write_exact_level(
    validated: ValidatedPointsSource,
    plan: _PointsCacheBuildPlan,
    *,
    staging_directory: Path,
    temporary_directory_root: Path,
    config: _ExactLevelWriterConfig,
) -> _LevelWriteResult: ...
```

C2 does not add an unimplemented function stub. C3 introduces the actual
function together with its first consumer.

The complete build plan is authoritative. The writer uses `plan.x_origin`,
`plan.y_origin`, and `plan.levels[0]`; it requires that record to have
`level == 0` and `kind is _LevelKind.EXACT`. Tile size, grid dimensions, level
identity, and relative output directory come from that record. They are not
duplicated in `config` or supplied as additional function arguments. The
initial builder and Xenium benchmark create a plan with 512-unit leaf tiles,
but the writer itself must not hard-code 512.

Define only these private frozen records:

```python
@dataclass(frozen=True)
class _ExactLevelWriterConfig:
    """Private physical execution settings for the Exact-level writer.

    Parameters
    ----------
    bucket_count
        Number of deterministic logical output buckets used by the local disk
        shuffle. Every logical tile maps to exactly one bucket, while one bucket
        may contain several tiles. Empty buckets need not create final files.
        This controls redistribution granularity and the potential final bucket
        file count, not the logical tile grid.
    max_rows_per_row_group
        Maximum points written to one physical Parquet row group. Because every
        row group contains exactly one logical tile, a denser tile is split into
        deterministic row-group shards of at most this size. This is physical
        sharding only; it never samples or removes Exact points. The initial
        construction default is 1,000,000 rows.
    dask_worker_count
        Number of local threads available to the Dask scheduler for the complete
        read, annotation, disk-redistribution, sorting, and writing graph. These
        are local threaded-scheduler workers, not distributed processes. This
        bounds graph parallelism and its combined memory and storage pressure.
    """

    bucket_count: int
    max_rows_per_row_group: int
    dask_worker_count: int


@dataclass(frozen=True)
class _ManifestRow:
    level: int
    level_file: str
    tile_x: int
    tile_y: int
    n_points: int
    row_group: int
    tile_shard: int


@dataclass(frozen=True)
class _IntermediateTileValueCountFile:
    level: int
    relative_path: str
    row_count: int


@dataclass(frozen=True)
class _LevelWriteResult:
    manifest_rows: tuple[_ManifestRow, ...]
    intermediate_tile_value_count_files: tuple[_IntermediateTileValueCountFile, ...]
```

The result is level-neutral: the Exact writer and every later sampled-level
writer return the same record type. Exact-specific behavior remains in the
writer entry point and `_ExactLevelWriterConfig`, while C7b can consolidate a
homogeneous collection of `_LevelWriteResult` records.

All configuration values are positive integers, excluding `bool`. C2 gives the
configuration fields no defaults: C3 supplies one explicit initial Dask
configuration, and Gate B approves or revises eventual production defaults.
Do not add a writer-engine enum, performance-metrics model, or separate
accounting dataclass. Exact row count, file count, and row-group count are
derivable from the returned manifest rows; C3 benchmark measurements remain
outside the logical writer result.

`_ManifestRow` follows the parent manifest's logical field meanings and integer
ranges. `level_file` is a normalized, cache-root-relative POSIX path directly
inside `levels/level_{level}` for the row's own `level`. C2 does not constrain
the filename itself; C3 freezes deterministic bucket naming. Rows are returned
in deterministic `(level, tile_y, tile_x, tile_shard)` order. Each row
describes exactly one physical row group containing exactly one logical tile;
`n_points` is positive, row-group and shard indices are non-negative, and shard
indices for a tile are contiguous from zero.

`_IntermediateTileValueCountFile` describes one flat, construction-only count
file emitted by one level-writer finalization unit. `level` follows the parent
manifest's non-negative `int16` range, `relative_path` is a normalized
cache-root-relative POSIX path, and `row_count` is positive. Descriptors are
unique and returned in deterministic `(level, relative_path)` order. C3 freezes
the intermediate-file directory and filename convention.

Each intermediate file contains flat nonzero
`(level, value_id, tile_x, tile_y, n_points)` rows with the logical field types
required by the final tile/value-count index. A tile finalizer may temporarily
aggregate `{value_id: count}` for its current tile, or use an equivalent Arrow
aggregation, but it must append those counts to a bucket-local intermediate file
and release the per-tile mapping. It must not retain one Python object or one
Python dictionary for every nonzero combination across the complete level.

Every logical tile belongs to exactly one writer bucket. That bucket finalizer
therefore owns the complete value counts for the tile, including points written
through several dense-tile row-group shards. It must aggregate those shards and
emit exactly one intermediate-file row for each nonzero
`(level, value_id, tile_x, tile_y)` key. The same logical key must not occur in
another intermediate file.

For example, the repeated values in the point rows for logical tile
`(tile_x=4, tile_y=8)` are the aggregation input:

```text
value_id
--------
3
3
8
3
```

The finalizer may transiently represent them as `{3: 3, 8: 1}`, but its flat
intermediate file stores:

```text
level  value_id  tile_x  tile_y  n_points
-----  --------  ------  ------  --------
0      3         4       8       3
0      8         4       8       1
```

One bucket-local intermediate file normally contains such rows for several
logical tiles. `_IntermediateTileValueCountFile.row_count` is the number of
aggregated nonzero `(level, value_id, tile_x, tile_y)` rows in that file; it is
not the number of original point rows. The example file therefore contributes
`row_count=2` for the displayed tile.

Taken together, a level's intermediate files contain every nonzero tile/value
count. For every exact logical tile, those counts sum to the `n_points` total
across the tile's manifest rows. Across Exact level 0, each `value_id` total
reconciles to the corresponding exact count in
`ValidatedPointsSource.value_table`.

#### Why the count files are intermediate

Counts are derived while a bucket finalizer already has the tile rows needed to
write the point payload. Persisting them at that moment prevents C7b from
rereading every completed point row group, or the canonical source, solely to
recalculate counts. For Exact on the Xenium acceptance source, such a rescan
would revisit all 136,578,750 points.

Independent bucket finalizers must not append concurrently to one shared
`tile_value_counts.parquet`. Parquet has one writer-owned footer, and routing
all counts through a central in-memory writer would add coordination, buffering,
and failure-handling complexity. Each finalizer therefore writes one
bucket-local flat intermediate file that it owns exclusively.

The counts in an intermediate file are exact, not approximate, but the file
covers only one finalization unit. It is not yet the complete, globally
value-ordered, reconciled runtime index. After every required level has been
written, C7b performs the reduction step:

```text
all bucket-local intermediate count files
→ read their already aggregated sparse rows
→ concatenate those count rows
→ reject duplicate logical keys
→ order by (level, value_id, tile_y, tile_x)
→ reconcile with manifest rows and exact value totals
→ write and reconcile tile_value_counts.parquet
→ remove the intermediate files
```

Intermediate count files are distinct from Dask shuffle-temporary files.
Shuffle files are execution scratch and are removed once their bucket has been
finalized. Intermediate count files contain semantic construction results and
must survive until C7b has successfully written and reconciled the final index.

The result retains tuples of small descriptors, not all count rows or Arrow
tables in memory. C7b initially reads the already aggregated intermediate rows,
concatenates them into one compact Arrow table, rejects duplicate logical keys,
and sorts that table once for the final index. This is not an end-to-end bounded
memory guarantee. C9 measures the Xenium index row count and peak memory; a
bounded external merge remains a local replacement if the evidence requires
it. C7b removes the intermediate files after the final index has been written
and reconciled. C2 itself does not materialize Arrow cache schemas.

### Cross-slice construction handoff

The private records connect the later construction slices as follows:

```text
C8 end-to-end builder
    │
    ├── creates staging generation and _ExactLevelWriterConfig
    │
    ├── calls C3 Exact writer
    │       │
    │       ├── Dask redistributes points into buckets
    │       │
    │       ├── each bucket finalizer:
    │       │       ├── writes the staged persistent Exact point bucket
    │       │       ├── produces _ManifestRow records
    │       │       ├── counts values per tile
    │       │       ├── writes one intermediate count file
    │       │       └── produces one _IntermediateTileValueCountFile descriptor
    │       │
    │       └── returns _LevelWriteResult
    │               ├── manifest_rows
    │               └── intermediate_tile_value_count_files
    │
    ├── calls C5c Bridge writer
    │       └── returns the Bridge _LevelWriteResult
    │
    ├── calls C6 spatial-level writers
    │       └── each spatial level returns another _LevelWriteResult
    │
    ├── passes all _LevelWriteResult objects to C7b
    │       ├── writes values.parquet and metadata.json
    │       ├── writes manifest.parquet from _ManifestRow records
    │       ├── consolidates the intermediate count files
    │       ├── writes tile_value_counts.parquet
    │       └── removes the intermediate count files
    │
    └── calls C7c
            └── independently validates the complete staged cache
```

The point bucket files are persistent members of the staged cache generation;
they are not Dask shuffle scratch. The intermediate count files are
construction-only handoff artifacts and disappear after C7b has created and
reconciled the final index. C7c then validates only the final staged artifacts.
After this handoff returns successfully, C8 performs
its final source-signature guard, completion-marker write, and local publication
steps.

### Directory ownership

The caller creates and owns one unique, initially empty
`staging_directory`. The exact writer may create only the Exact level subtree
derived from the plan and must fail rather than overwrite an existing target.
It never removes the caller's staging root, writes `COMPLETED`, publishes the
generation, or touches an existing completed cache. The caller rejects and
removes the complete staging generation after any construction failure.

Intermediate tile/value-count files are staged construction artifacts, not
shuffle-temporary files. They remain inside `staging_directory` after a level
writer returns, appear in `_LevelWriteResult` only through cache-root-relative
descriptors, and survive until C7b has written and reconciled the consolidated
index. C7b then removes the intermediate files before C7c validates the final
staged generation and before C8 publication.

`temporary_directory_root` is a caller-selected location for disposable local
shuffle storage. The writer creates a unique child beneath it, owns that child
exclusively, closes all resources, and removes the child after both success and
failure. Shuffle-temporary paths never appear in `_LevelWriteResult`. A cleanup
failure must not hide the original construction exception.

These ownership rules describe later C3 behavior. C2 itself creates or removes
no directories.

### Output invariants fixed for C3

- every validated source point occurs exactly once in the Exact output;
- every expected `point_id` occurs exactly once;
- the physical payload is exactly `x_rel: float32`, `y_rel: float32`,
  `value_id: uint32`, and `point_id: uint64`, all non-nullable;
- every ordinary tile is co-located in one deterministic final bucket file,
  independently of source partitions;
- every physical row group contains one logical tile, while a dense tile may
  use several deterministic row-group shards;
- returned manifest rows and intermediate tile/value-count file descriptors
  completely describe the written Exact level and its construction counts;
- all output paths are cache-root-relative and all writes remain inside the
  caller-owned staging generation.

Detailed source traversal, annotation, normalization-helper refactoring,
bucket hashing, Dask shuffle construction, bucket-size measurement,
concurrency, and measurements belong to C3. Strict oversized-bucket handling is
added later only if the acceptance benchmark demonstrates the need. Direct
PyArrow belongs only to optional C4.

### Focused tests

Test only semantic record validation: invalid configuration counts, unsafe or
non-normalized manifest and intermediate-file paths, invalid integer ranges,
and invalid nonpositive counts. Do not test dataclass immutability, Python tuple
behavior, Dask, or PyArrow. Intermediate-file contents, deterministic ordering,
duplicate logical keys, and cross-record reconciliation require actual writer
output and are tested across C3, C7b, and C7c.

### Exit criteria

- C3 can implement the documented entry contract without inventing another
  build plan or duplicating logical geometry arguments;
- the staging and temporary-directory owner is unambiguous on success and
  failure;
- provisional row-group records and intermediate tile/value-count file
  descriptors are sufficient for later staged-cache validation without
  retaining every sparse count as a Python object;
- no source rows are read and no speculative public API is exported.

## Slice C3: Dask exact-level writer and acceptance benchmark

### Goal

Implement one credible exact-level writer using Dask's local disk shuffle plus
Arrow finalization. Run a small acceptance benchmark after correctness is
established. Do not implement a second engine in this slice.

### File-aligned Dask source contract

The initial writer relies on Dask's public Parquet reader for physical decoding,
but constructs its own graph from `ValidatedPointsSource`. It must not consume
the existing SpatialData points dataframe: that graph may have been filtered,
repartitioned, or shuffled and therefore is not authoritative for physical-file
provenance or source row positions.

For each validated source file, in deterministic inventory order, the writer
creates a separate lazy read equivalent to:

```python
dd.read_parquet(
    physical_file_path,
    columns=[x_column, y_column, value_column],
    split_row_groups=False,
)
```

Each such read has exactly one Dask input partition containing that complete
physical file. The writer annotates it with the corresponding validated
`ParquetSourceFile` record and then concatenates the annotated reads. Dask may
execute these partitions in any order; inventory order remains the source of
identity, not execution or arrival order.

Dask and PyArrow handle the row groups inside each file. The initial writer does
not create one partition per row group and does not implement arbitrary
row-range partitions. It reconciles each decoded partition length with the
validated file row count and constructs point IDs as:

```text
source_file.row_offset + physical row position within that file
```

Consequently, `_ExactLevelWriterConfig` has no source-partition row-limit
setting. The physical source files determine input-partition size. On the
initial Xenium dataset this means 65 input partitions for 65 Parquet files;
although the files contain 168 row groups, row groups are not separate Dask
partitions in the initial design.

The acceptance benchmark must record the memory behavior of this file-aligned
contract. Only if one-file partitions prove too memory-intensive do we introduce
a row-group-aligned fallback using `split_row_groups=True`, together with the
row-group provenance needed for identical point IDs. That is a targeted reader
refinement, not a reason by itself to implement the optional direct-PyArrow
writer.

### Initial bucket and finalizer configuration

Before adding the reader, C3 removes the obsolete
`max_source_rows_per_partition` field from `_ExactLevelWriterConfig` and its
focused tests. The file-aligned source contract has no replacement row-limit
field.

The initial deterministic heuristic targets at most 2,000,000 points per
physical output bucket on average:

```python
target_rows_per_output_bucket = 2_000_000
bucket_count = max(
    1,
    math.ceil(
        exact_level.point_count_upper_bound / target_rows_per_output_bucket
    ),
)
```

The target is an internal construction default, not a new public builder
argument. `_ExactLevelWriterConfig` retains the resulting integer
`bucket_count`. For Exact, `exact_level.point_count_upper_bound` equals
`validated.row_count`; the C3 benchmark and the later end-to-end builder
calculate the bucket count before invoking the Exact writer.

For the 136,578,750-point Xenium source this produces 69 buckets, averaging
about 1.98 million points per bucket before hash skew, and at most 69
Exact-level bucket Parquet files. An empty bucket does not create a file. This
output count is independent of the 65 physical source files: every input file
may contribute rows to many output buckets after the shuffle. Source-file count
describes physical source packaging and must not determine cache layout.

Using `ceil` guarantees only that the arithmetic average is no greater than the
two-million-row target. It is not a hard per-bucket limit: hash skew or one very
dense tile may still produce a larger bucket.

This physical target is distinct from `max_points_per_tile`, which controls
logical sampled-level membership. Every later sampled level applies the same
physical heuristic independently using its own planned
`point_count_upper_bound`; it must not reuse the Exact level's bucket count. As
levels become smaller, their output bucket counts therefore decrease, normally
reaching one for the overview.

The initial physical row-group maximum is independently fixed at:

```python
max_rows_per_row_group = 1_000_000
```

Every row group contains rows from exactly one logical tile. A tile containing
more than 1,000,000 points is written as deterministic row-group shards; a
bucket may contain multiple row groups from one or several tiles. This maximum
does not conflict with `target_rows_per_output_bucket`: the former limits one
tile shard, while the latter targets the total rows across the complete bucket
file.

For this pragmatic implementation, one finalizer materializes, sorts, writes,
and releases one complete bucket at a time. C3 does not yet implement recursive
bucket spilling or an external bounded sort and therefore does not claim a
strict worst-case finalization-memory bound. Its benchmark measures average and
maximum bucket sizes and peak RSS. Only a demonstrated problem justifies adding
an oversized-bucket fallback or changing the bucket count or concurrency.

### Stable tile-to-bucket mapping

The physical bucket mapping uses the versioned method identifier:

```python
BUCKET_HASH_METHOD = "harpy-tile-splitmix64-v1"
```

For non-negative `uint32` tile coordinates, form one collision-free `uint64`
tile key and apply the SplitMix64 finalizer with explicit modulo-`2**64`
arithmetic:

```python
tile_key = (uint64(tile_y) << uint64(32)) | uint64(tile_x)

z = tile_key + uint64(0x9E3779B97F4A7C15)
z = (z ^ (z >> uint64(30))) * uint64(0xBF58476D1CE4E5B9)
z = (z ^ (z >> uint64(27))) * uint64(0x94D049BB133111EB)
tile_hash = z ^ (z >> uint64(31))

bucket_id = tile_hash % uint64(bucket_count)
```

Every addition and multiplication wraps as `uint64`. The result is converted to
the bucket-id integer dtype used by the Dask shuffle. This mapping has no
runtime-random seed: identical tile coordinates and `bucket_count` always
produce the same bucket. All points in one tile therefore share one bucket,
while the mixer avoids coupling neighbouring tile coordinates directly to
neighbouring or identical modulo buckets.

This hash controls only physical tile-to-bucket placement. It must not be
silently reused as the later sampling-priority hash, whose payload, seed, and
version remain separate sampling-contract decisions.

Bucket filenames use:

```python
filename_width = max(3, len(str(bucket_count - 1)))
filename = f"bucket-{bucket_id:0{filename_width}d}.parquet"
```

The persistent point file is `levels/level_<level>/<filename>`. Its intermediate
count file is
`intermediate_tile_value_counts/level_<level>/<filename>`. Empty buckets create
neither file. Point buckets use Snappy compression and dictionary encoding only
for `value_id`; intermediate count files use Snappy without dictionary
encoding.

### Implementation responsibilities

- remove the obsolete `max_source_rows_per_partition` configuration field and
  its focused test inputs before constructing the file-aligned reader;
- construct one Harpy-owned Dask input partition per validated physical file,
  requesting only the selected columns and explicitly using
  `split_row_groups=False`;
- construct `uint64 point_id` arrays batch-wise from validated source-file row
  offsets and physical row positions;
- factor the existing private Arrow value-normalization operations into one
  narrow helper shared by validation and construction, rather than reimplement
  the Unicode trimming policy;
- map normalized values vectorially against
  `ValidatedPointsSource.value_table`, including dictionary-encoded batches
  without Python-per-row string processing;
- assign numeric tile coordinates from `float64` working coordinates using the
  Exact record and shared origin in `_PointsCacheBuildPlan`, then convert only
  tile-local coordinates to `float32`;
- implement and test `harpy-tile-splitmix64-v1`, calculate the initial bucket
  count, and use the locked deterministic bucket filename width;
- use the locked output row-group size and Dask worker count, and configure
  the Dask shuffle;
- measure output-bucket skew and peak finalization memory, without adding a
  speculative recursive-spill or external-sort mechanism;
- shard dense tiles into deterministic output row groups after the containing
  bucket has been sorted, while acknowledging that this output sharding does
  not itself bound bucket-finalization memory;
- produce the C2 result records and enforce their deterministic ordering,
  uniqueness, membership, identity, and reconciliation invariants;
- honor the C2 staging and temporary-directory ownership contract.

### Dask disk shuffle plus Arrow finalizer

Harpy constructs this Dask dataframe only from the validated ordered physical
inventory, using owned readers that preserve source-file offsets and row
positions. It never accepts or inspects an arbitrary caller graph. The writer
uses the integer `bucket_id`, explicit divisions, and Dask's local disk shuffle
so one output partition corresponds to one writer bucket without a
quantile-discovery scan.

It must keep these stages distinct:

```text
validated physical source
→ annotated: file-partitioned lazy Dask dataframe
→ bucketed: bucket-partitioned lazy Dask dataframe
→ ordered bucket: one computed and sorted output partition
→ bucket-<id>.parquet: final level file inside the caller-owned staging artifact
```

`annotated` retains the Harpy-owned one-file source partitions and contains at
least `tile_x`, `tile_y`, `x_rel`, `y_rel`, `value_id`, `point_id`, and
`bucket_id`. It is neither shuffled nor persisted as a cache level.

`bucketed` is the lazy result of the disk-shuffle graph. Output Dask partition
`i` contains all rows assigned to integer bucket `i`, possibly in
nondeterministic arrival order. Dask's temporary on-disk fragments are internal
shuffle state; they are not final bucket Parquet files.

The finalizer computes one bucket partition, applies the deterministic
`(tile_y, tile_x, point_id)` sort, groups the resulting contiguous rows by
logical tile, and writes one Parquet row group per capacity-bounded tile shard.
Only this finalizer creates `bucket-<id>.parquet`, its provisional
level-manifest rows, and its flat intermediate tile/value-count file. Counts
are derived while tile rows are already available. A per-tile value-count
mapping may exist only transiently while that tile is finalized; after its flat
rows are appended to the bucket's intermediate file, the mapping is released.
If the tile uses several physical row-group shards, the finalizer aggregates
across those shards before emitting one row per nonzero logical key. The result
returns one small intermediate-file descriptor rather than one Python record
per nonzero count. No additional source or completed-level scan is allowed.

The finalizer should consume numeric data and write through PyArrow. It must not
copy the legacy per-partition Pandas string construction, schema, or direct
side-effect pattern merely because that code exists.

After all bucket finalizers complete, the Exact writer performs only the cheap
level-wide conservation checks available from their compact results:

```text
sum(bucket point_count)       == ValidatedPointsSource.row_count
sum(bucket value_count_total) == ValidatedPointsSource.row_count
```

This proves that the completed buckets and their intermediate counts account
for the expected total number of points. It does not prove that every
individual `value_id` has the canonical `n_points` recorded during source
validation. Do not add a potentially large per-value mapping to every bucket
result or reread all intermediate count files inside the Exact writer merely to
perform that later reduction twice. Exact per-value reconciliation belongs to
C7b, which consumes the actual intermediate files. A cache cannot be completed
or published before that check succeeds.

### Locked physical point payload

The Dask writer must emit the locked exact-level payload:

```text
x_rel: float32
y_rel: float32
value_id: uint32
point_id: uint64
```

Do not repeat `tile_id`, `tile_x`, or `tile_y` per point. Validate coordinate
reconstruction and a representative tile read using the manifest tile key and
the locked payload. Changing this contract later requires an explicit cache-
format revision; it is not a writer-engine decision at Gate B.

Use the locked manifest column set from the parent roadmap. Do not copy
`schema_version` or the derived string `tile_id` from the legacy metadata
dataframe: `schema_version` belongs once in `metadata.json`, while `tile_id` is
derived from `(level, tile_x, tile_y)`. C3 records the physical row-group facts
needed by C7a; Gate D owns the remaining manifest Arrow details but does not
reopen these exclusions.

### Initial local execution and failure contract

Phase 1 is a local, no-task-retry builder. The Dask writer may use only a
Harpy-controlled local threaded or synchronous scheduler and local disk shuffle;
it must not use `distributed.Client`, automatic retries, or speculative task
execution. The finalization graph is computed once. Exactly one finalizer owns
each deterministic `bucket-<id>.parquet` path, and different concurrent
finalizers own different paths.

Each build or spike uses a fresh, unique, initially empty staging directory. A
finalizer writes its owned bucket path, closes the writer, and returns
provisional level-manifest rows. It must fail if that deterministic path already
exists. If a finalizer or any later construction step fails, the complete
staging generation is rejected; a partial or otherwise successfully written
bucket is never recovered, installed, or reused. A user-initiated rebuild starts
from a new staging generation rather than retrying a task in the failed one.
Only a fully closed and subsequently validated staging generation can receive
`COMPLETED` and be published, so failure cannot replace an existing completed
cache.

Attempt-local files, coordinated winner installation, task-retry idempotence,
and resumable staging are explicitly deferred. They must be designed before a
future implementation enables distributed schedulers, automatic retries,
speculative execution, multiple writers for one staging generation, resumable
builds, or object-store publication.

The one initial acceptance run records at least:

- exact build time and peak RSS;
- peak temporary disk usage and shuffle volume when Dask exposes it cheaply;
- written bytes, average and maximum bucket rows and bytes, bucket/file count,
  row-group count, and manifest rows;
- coordinate reconstruction error;
- membership and point-id coverage;
- one representative tile-read latency from the staged artifact.

The initial benchmark is an acceptance check, not an engine tournament. Use the
512-unit geometry, derive `bucket_count` using
`target_rows_per_output_bucket=2_000_000`, and use
`max_rows_per_row_group=1_000_000` and `dask_worker_count=1`. For the Xenium
target this means 69 buckets and at most 69 Exact-level bucket files. One
representative benchmark run is sufficient for the first decision; do not
require a parameter sweep,
concurrency sweep, or repeated statistical benchmark before the design has
demonstrated a concrete problem.

### Focused tests

Use small multi-file and multi-row-group fixtures to cover exact membership,
point identity, tile boundaries, deterministic output, coordinate
reconstruction, and one dense-tile shard case. Deliberately spread one logical
tile across source files and verify that the shuffle co-locates it. Include one
small set of fixed SplitMix64 vectors so implementation changes cannot silently
alter bucket placement. Include one finalizer-failure case proving that
incomplete staging output is rejected. Test Harpy's accounting and ownership
rules; do not retest Dask or Parquet internals.

### Exit criteria

- 512-unit exact construction is demonstrated;
- observed peak memory and maximum bucket size are acceptable for the initial
  Xenium target, with no claim yet of a source-size-independent worst-case
  bound;
- temporary disk use and cleanup are measured and bounded by a documented
  construction policy;
- exact membership, identity, and coordinate reconstruction are correct;
- the Dask writer uses the locked exact-level payload and reconstructs
  coordinates correctly from its manifest tile key;
- bucket mapping, shuffle/grouping, single-owner output, file, and dense-tile
  output-sharding policies are approved for downstream construction;
- benchmark artifacts are removed after measurements are recorded.

### Implemented C3 acceptance result

One acceptance run completed on 2026-08-07 against the validated Xenium source
with 136,578,750 points, 65 source files, 168 source row groups, and 5,122
normalized values. It used 512-unit Exact tiles,
`harpy-tile-splitmix64-v1`, 69 output buckets,
`max_rows_per_row_group=1_000_000`, and `dask_worker_count=1`.

Measured construction results were:

- 57.92 seconds for Exact construction, excluding the separately measured
  2.00-second validation;
- 3.69 GiB peak process RSS, 3.36 GiB above the pre-build baseline;
- 1.74 GiB peak benchmark workspace usage;
- 69 nonempty Exact bucket files containing 136,578,750 points in 7,294 row
  groups;
- 1.65 GiB of Exact point Parquet files plus 0.08 GiB of intermediate
  tile/value-count files;
- 1,979,402 average rows and 2,547,160 maximum rows per bucket;
- 25.70 MB average and 32.97 MB maximum bucket-file size;
- exact point-ID verification of all 136,578,750 unique contiguous IDs in 12.69
  seconds;
- one representative 108,598-point tile read from one row group in 1.15 ms;
- every source partition passed the locked float32 tile-relative reconstruction
  tolerance of `6.103515625e-05` intrinsic coordinate units.

The representative tile read followed the complete point-ID verification scan
and is therefore a warm-cache measurement, not a cold-storage latency claim.
The bucket maximum exceeded the two-million-row average target as expected from
hash skew, but remained small enough for the measured memory envelope. No
recursive spill, external sort, file rollover, larger bucket count, or direct-
PyArrow investigation is justified by this run. All benchmark staging, shuffle,
and JSON result artifacts were removed after the measurements were recorded.

Gate B records one of two outcomes:

1. **Dask accepted:** correctness and the small benchmark are sufficient; keep
   Dask as the Phase 1 exact writer and proceed directly to C5a. **Selected on
   2026-08-07.**
2. **PyArrow investigation justified:** record the concrete Dask limitation and
   the success criterion that a direct-PyArrow experiment must meet; open
   optional C4 before proceeding.

## Slice C4: optional direct-PyArrow exact-writer investigation

### Goal

Investigate a direct-PyArrow spill-and-compaction writer only when Gate B or
later evidence records a concrete Dask limitation and a measurable PyArrow
success criterion. This slice remains deferred indefinitely while the Dask
result is acceptable and does not block subsequent construction work.

### Conditional experiment

```text
bounded source batch
→ numeric exact-level annotation and bucket_id
→ batch-local partitioning of row indices by bucket_id
→ deterministic temporary bucket fragments
→ bounded bucket compaction and grouping
→ final Parquet row groups and provisional level-manifest rows
```

The experiment must retain C2's entry/result contracts and C3's exact payload,
tile co-location, deterministic ordering, accepted measured memory envelope,
single-owner output, and correctness requirements. It must define bounded
file-handle use, temporary-fragment consolidation, concurrency, and cleanup. If
the named C3 limitation is oversized-bucket memory, it must additionally define
and measure the replacement spill or external-grouping policy. It performs the
same complete logical redistribution as Dask; it is not a partition-local
shortcut.

Compare only against the concrete C3 limitation and the success criterion
recorded at Gate B. Do not add DuckDB, Polars, Spark, or another engine, and do
not require feature parity beyond what is needed to make the decision.

### Exit criteria

- record whether direct PyArrow materially resolves the named Dask limitation;
- retain Dask unless the alternative has a meaningful measured advantage that
  justifies its additional implementation and maintenance cost;
- if PyArrow is selected, rerun the complete C3 exact-writer correctness checks
  before downstream construction consumes it;
- remove all experimental artifacts after recording the decision.

## Slice C5a: generic 16 × 16 sampled-tile contract and pure in-memory sampler

### Goal

Freeze and implement one deterministic membership rule for any sampled logical
tile. The first concrete case is one 512-unit Exact tile becoming one 512-unit
bridge tile capped at 4,096 representatives. This slice is a pure selection
spike: it does not read Parquet, use Dask, or write a cache level.

### Fixed first implementation

For one complete current-level candidate tile:

```text
all candidate rows contributing to the output tile
→ assign each point to one cell in a 16 × 16 within-tile microgrid
→ count candidates in every occupied cell
→ allocate the tile target proportionally across occupied cells
→ calculate a stable value-independent uint64 priority per point
→ retain the lowest-priority points within every cell allocation
→ sort retained indices by point_id
```

Use these initial versioned constants:

```python
SAMPLING_METHOD = "harpy-value-neutral-stratified-splitmix64-v1"
SAMPLING_SEED = 0
SAMPLED_TILE_MICROGRID_EDGE = 16
```

The grid is fixed relative to the **current output tile**, not at one fixed
intrinsic cell size. Its cells therefore scale with the level:

| Level | Tile edge | Microgrid | Cell edge |
|---|---:|---:|---:|
| Bridge | 512 | 16 × 16 | 32 |
| L1 | 1,024 | 16 × 16 | 64 |
| L2 | 2,048 | 16 × 16 | 128 |
| L3 | 4,096 | 16 × 16 | 256 |

The logical tile hierarchy becomes progressively coarser:

```text
Bridge: many small tiles
L1:     fewer, larger tiles
L2:     fewer again
...
Overview: very few tiles, eventually one
```

Four 512-unit Bridge tiles form one 1,024-unit L1 tile. Each finer Bridge tile
therefore covers an 8 × 8 quadrant of the coarser L1 tile's 16 × 16 microgrid:

```text
L1 microgrid: 16 × 16

┌──────────┬──────────┐
│ 8 × 8    │ 8 × 8    │
│ cells    │ cells    │
├──────────┼──────────┤
│ 8 × 8    │ 8 × 8    │
│ cells    │ cells    │
└──────────┴──────────┘
```

The logical tiles are persistent storage, manifest, and loading units. The
microgrid is only a transient sampling structure within one output tile; its
cells are not stored as cache tiles or loaded independently.

Later levels follow the same 16 × 16 current-tile-relative grid. This is an
internal initial policy, not a public builder option. The spike may reject it
only with a named correctness or clearly demonstrated spatial-representation
problem.

The pure entry point has this architectural shape:

```python
def _select_sampled_tile_indices(
    x_rel: np.ndarray,
    y_rel: np.ndarray,
    point_ids: np.ndarray,
    *,
    level: int,
    tile_x: int,
    tile_y: int,
    tile_size: int,
    target: int,
) -> np.ndarray: ...
```

The implementation docstring must preserve this distinction between the
logical tile hierarchy and the transient microgrid. It should include the
Bridge/L1/L2 tile-and-cell-size table and the four-finer-tile L1 schematic
above, or
an equivalently clear compact explanation, so callers do not mistake microgrid
cells for cache tiles.

`x_rel` and `y_rel` are relative to the current output tile. For the bridge,
Exact and output tile geometry match. For a later spatial level, its writer
first rebases immediate-finer tile coordinates into the coarser tile before
calling this same function.

The function deliberately does not accept `value_id`. The caller applies the
returned indices to the complete payload, thereby propagating `value_id`
unchanged while making it impossible for the selection kernel to use values
during allocation or ranking.

The selector contract is:

- `x_rel`, `y_rel`, and `point_ids` are one-dimensional NumPy arrays with the
  same length;
- coordinate arrays are numeric and finite, and `point_ids` has dtype
  `uint64`;
- `level` is an integer in the supported non-negative serialized `int16` range,
  `tile_x` and `tile_y` are integers in the `uint32` range, and `tile_size` and
  `target` are positive integers;
- coordinates must lie in the closed interval `[0, tile_size]`; the inclusive
  upper edge exists only to accept a source coordinate just below the boundary
  that rounded to `tile_size` in the stored `float32` payload;
- globally unique `point_id` values are an upstream cache invariant and are not
  rescanned for uniqueness inside every sampler call;
- empty input returns an empty `np.intp` array;
- nonempty input returns exactly `min(candidate_count, target)` original row
  indices, ordered by the corresponding ascending `point_id` values.

Assign coordinates to the fixed microgrid as:

```text
cell_x = min(floor(float64(x_rel) * 16 / tile_size), 15)
cell_y = min(floor(float64(y_rel) * 16 / tile_size), 15)
cell_id = cell_y * 16 + cell_x
```

Reject negative coordinates and coordinates greater than `tile_size`; do not
silently clamp general out-of-range input. Only an exact upper-edge value maps
to cell 15. Calculate the cell coordinate in `float64` so the classification is
not changed by another intermediate `float32` rounding.

If a tile contains at most `target` candidates, retain every candidate and
return its indices in ascending `point_id` order. Otherwise, use these cell IDs
for proportional allocation and selection.

### Exact proportional allocation

For occupied-stratum count `n_i`, total candidate count `N`, and target `K`:

```text
base_i      = (K * n_i) // N
remainder_i = (K * n_i) % N
```

The coordinates have already assigned every point to one microgrid cell. These
formulas allocate **sample slots**, not points, to those cells. A cell containing
10% of the tile's candidates should receive approximately 10% of the retained
representatives. Allocation is proportional to observed cell population; it is
not an equal number per occupied cell.

For example, consider four occupied cells with candidate counts `(5, 4, 2, 1)`,
so `N=12`, and a simplified target `K=7`. Applying
`divmod(K * n_i, N)` gives quotient/remainder pairs `(2, 11)`, `(2, 4)`,
`(1, 2)`, and `(0, 7)`. The base allocation `(2, 2, 1, 0)` accounts for five
representatives. The two remaining slots go to the first and fourth cells,
which have the largest remainders, producing:

```text
cell candidate counts:  5  4  2  1
retained allocations:   3  2  1  1
```

After allocation, stable point priorities decide which three, two, one, and one
actual candidates win inside their respective cells.

Assign the `K - sum(base_i)` remaining slots to the largest remainders. Resolve
equal remainders with a stable pseudo-random stratum priority derived from the
versioned method, fixed seed, level, tile key, and microgrid-cell ID, followed
by the numeric cell ID as a final collision tie-breaker. Do not use iteration
order or always favor the numerically first cells.

Because `K <= N` and the allocation is proportional to observed candidate
counts, no allocation can exceed its stratum's available candidates. The first
implementation therefore has no redistribution phase for unconsumed
allocations. When occupied strata outnumber `K`, the same largest-remainder rule
deterministically assigns positive allocations to at most `K` strata.

The microgrid is therefore not an equal-coverage mechanism and must not flatten
real spatial density. Dense cells retain proportionally more representatives;
sparse cells retain proportionally fewer. Compared with ranking the complete
tile as one unstratified population, the grid constrains every local cell to
remain close to its observed share and reduces random local clumping or holes.

### Stable point priority

Sampling randomness is a deterministic random-looking `uint64`, not a stateful
random-number-generator sequence. Extract the existing SplitMix64 transform
from the Exact writer into the internal `hashing.py` module and use that shared
primitive for both bucket placement and sampling. Its arithmetic remains the
standard wraparound `uint64` transform already protected by the Exact bucket
fixed vectors:

```text
z = value + 0x9E3779B97F4A7C15
z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
z = (z ^ (z >> 27)) * 0x94D049BB133111EB
splitmix64(value) = z ^ (z >> 31)
```

Freeze separate domains for candidate-point priorities and the cell priorities
used to resolve equal allocation remainders:

```python
_POINT_PRIORITY_DOMAIN = np.uint64(0x48504F494E543031)  # "HPOINT01"
_CELL_PRIORITY_DOMAIN = np.uint64(0x4843454C4C303031)   # "HCELL001"
```

The current tile and microgrid cell keys are:

```text
tile_key = (uint64(tile_y) << 32) | uint64(tile_x)
cell_id = uint64(cell_y * 16 + cell_x)
```

For every candidate point, calculate its priority by this exact sequence:

```text
state = splitmix64(_POINT_PRIORITY_DOMAIN ^ uint64(SAMPLING_SEED))
state = splitmix64(state ^ uint64(level))
state = splitmix64(state ^ tile_key)
state = splitmix64(state ^ cell_id)
point_priority = splitmix64(state ^ point_id)
```

For deterministic ordering of cells with equal allocation remainders, calculate:

```text
state = splitmix64(_CELL_PRIORITY_DOMAIN ^ uint64(SAMPLING_SEED))
state = splitmix64(state ^ uint64(level))
state = splitmix64(state ^ tile_key)
cell_priority = splitmix64(state ^ cell_id)
```

Sort equal-remainder cells by `(cell_priority, cell_id)`. The separate domains
prevent the point and cell priorities from sharing one hash namespace. This
sampling method is versioned separately from bucket placement even though both
reuse the same SplitMix64 primitive.

Within each allocated cell, rank by `(priority, point_id)` and retain the first
allocated number. The globally unique `point_id` supplies deterministic
collision handling. Sort the selected indices by `point_id` before returning
them. Input row order, physical shard boundaries, Python's randomized `hash()`,
and Dask partitioning must not influence membership.

### Initial memory contract

The first implementation may load and concatenate every candidate contributing
to one current output tile before sampling. For the bridge this means all
physical row-group shards belonging to one logical Exact tile. For a later
spatial level it means candidates from up to four immediate-finer tiles.
It does not promise a source-size-independent bound for a pathological tile.
Peak candidate memory is therefore bounded by the densest current-level
candidate tile encountered, not by the complete dataset and not by
`max_rows_per_row_group`. A future two-pass or streaming top-k sampler requires
measured evidence that this assumption is insufficient.

### Focused tests and spike

Keep C5a synthetic and pure. Cover sparse pass-through and exact capacity,
proportional integer allocation, equal-remainder ties, controlled priority
collisions, microgrid boundaries and scaling, coincident coordinates, and
input-order invariance. Add fixed vectors for point and cell priorities so the
versioned method cannot change accidentally. Compare selected `point_id` values,
not raw returned index numbers, when candidate input order changes.

Do not add value-label permutation tests: `value_id` is absent from the API, so
value neutrality is enforced structurally. Do not test physical row-group shard
division or perform a Xenium inspection in this pure slice. C5c already owns one
Exact tile split across several row groups and its focused Xenium acceptance
check.

### Exit criteria

- the method name, seed, shared 16 × 16 current-tile microgrid, payload
  encoding, and SplitMix64 priority are frozen;
- the same logical candidates and parameters always produce the same winners;
- every winner is an unchanged input candidate and capacity is never exceeded;
- input order does not affect membership;
- `value_id` is absent from the selection API and has no influence on winners;
- the one-complete-current-tile memory assumption is accepted for C5c and C5d.

### Implemented C5a result

Implemented on 2026-08-10 in the new internal `sampling.py` and `hashing.py`
modules. The Exact bucket mapper now imports the shared vectorized SplitMix64
transform without changing `harpy-tile-splitmix64-v1` output. The pure selector
implements the fixed 16 × 16 current-tile microgrid, proportional
largest-remainder allocation, separate versioned point and cell priority
domains, deterministic point-ID collision handling, and ascending-point-ID
output indices. It performs no Parquet IO, Dask work, or cache writing.

Focused synthetic tests cover fixed priority vectors, sparse and exact-capacity
pass-through, proportional allocations, equal-remainder cell ordering,
microgrid boundary scaling, input-order invariance, controlled priority
collisions, and representative invalid inputs. The existing Exact bucket fixed
vector remains unchanged after extracting the shared SplitMix64 transform.

## Slice C5b: level-neutral physical writer-support extraction

### Goal

Extract the physical cache-format machinery that is already concrete in the
Exact writer before the persistent Bridge writer becomes its second consumer.
This is a behavior-preserving internal refactor: it creates
`writer/support.py`, but it does not create `writer/bridge.py`, read staged
Exact rows, construct a sampled level, or change any persistent output.

### Module responsibilities

After this refactor, the construction modules have these boundaries:

```text
writer/models.py
    immutable configuration and result records

writer/support.py
    shared physical schemas
    bucket-count calculation
    tile-to-bucket hashing
    intermediate tile/value-count writer
    bucket-file validation
    level-result reconciliation

writer/exact.py
    original source -> Exact
    source annotation and value-ID mapping
    Dask disk shuffle and Exact bucket finalization

writer/bridge.py
    not created until C5c
```

Move the immutable `_BucketWriteResult` from the Exact writer into
`writer/models.py` beside `_ManifestRow`, `_IntermediateTileValueCountFile`, and
`_LevelWriteResult`. Its documentation and accounting semantics become
level-neutral; `_ExactLevelWriterConfig` remains explicitly Exact-specific.

Move only the established physical primitives needed by both Exact and future
sampled writers into `writer/support.py`:

- rename `_EXACT_PAYLOAD_SCHEMA` to the level-neutral
  `_POINT_PAYLOAD_SCHEMA` without changing its four fields, order, types, or
  nullability;
- move `_TILE_VALUE_COUNT_SCHEMA`, the intermediate-count directory and buffer
  constants, `BUCKET_HASH_METHOD`, `TARGET_ROWS_PER_OUTPUT_BUCKET`, and the
  shared maximum-row-group default;
- move `_bucket_count_for_level` and `_tile_bucket_ids` without changing their
  formulas, versioned hash, or fixed-vector output;
- move `_IntermediateTileValueCountWriter` without changing its buffering,
  Parquet encoding, row accounting, or close behavior;
- make bucket-file validation level-neutral while preserving its schema,
  footer, manifest-row, and intermediate-descriptor checks;
- make level-result reconciliation level-neutral while preserving point-count,
  value-count, manifest ordering, duplicate physical-key, and duplicate
  intermediate-path checks.

`writer/exact.py` imports those primitives and retains all original-source,
normalization, identity, annotation, Dask, shuffle, and Exact-finalization
logic. `DEFAULT_DASK_WORKER_COUNT`, `_ExactLevelWriterConfig`, and the
reconciliation against `ValidatedPointsSource.row_count` remain Exact-owned.

Do not introduce an abstract base writer, writer-engine hierarchy, callback
pipeline, generic tile scheduler, placeholder Bridge module, or new public
export. This slice extracts only code with a concrete Exact consumer and an
agreed C5c Bridge consumer.

### Focused tests and exit criteria

Run the existing focused Exact-writer and writer-model tests after the move.
Retain the fixed tile-hash vector and focused intermediate-count, schema,
manifest-ordering, row-count, and duplicate-key coverage. Add direct tests only
where moving a shared boundary exposes a real untested behavior; do not test
module import structure or Python itself.

C5b is complete when:

- the Exact writer produces the same schemas, hash assignments, filenames,
  row-group layout, manifest rows, intermediate counts, and reconciliation
  results as before;
- no persistent method name, default, or cache-format field changes;
- `writer/exact.py` no longer owns level-neutral physical schemas, hashing,
  count writing, file validation, or result reconciliation;
- `writer/support.py` contains no source-reading, Dask-shuffle, sampling, or
  Bridge-specific behavior;
- `writer/bridge.py` does not yet exist;
- focused tests and lint checks pass. A new Xenium benchmark is unnecessary for
  this behavior-preserving refactor.

### Implemented C5b result

Implemented on 2026-08-11. The internal `writer/support.py` owns the shared
point and tile/value-count schemas, physical writer constants, deterministic
tile bucket mapping and bucket-count calculation, intermediate count writer,
bucket-file validation, and level-result reconciliation. `_BucketWriteResult`
now lives with the other immutable records in `writer/models.py`.

`writer/exact.py` retains source annotation, value and point identity mapping,
Dask redistribution, and Exact bucket finalization while importing the shared
physical support. The Exact benchmark script and focused tests now import the
level-neutral constants and helpers from their owning module. The fixed bucket
hash vector, Exact point payload, row-group layout, manifest rows, and
intermediate counts remain unchanged. No Bridge module or sampled output was
introduced, and the large Xenium benchmark was not rerun.

## Slice C5c: persistent bridge-level construction and acceptance check

### Goal

Use the C5a sampler to construct the real 512-at-4,096 bridge from the staged
Exact level, without rescanning the original Parquet source.

### Implement

The minimal private entry point is:

```python
def _write_bridge_level(
    exact_result: _LevelWriteResult,
    plan: _PointsCacheBuildPlan,
    *,
    staging_directory: Path,
) -> _LevelWriteResult:
    ...
```

The bridge writer consumes only the staged Exact-level result, the immutable
logical build plan, and the caller-owned staging directory. It does not receive
`ValidatedPointsSource`: constructing the bridge reads the staged Exact point
payload and must not revisit the original physical source. It also receives no
shuffle-temporary directory or Dask worker setting because this first bridge
implementation performs no shuffle and processes tiles sequentially. The
sampling target is not a separate argument; it is the bridge level's
`max_points_per_tile` in `_PointsCacheBuildPlan`.

`exact_result.intermediate_tile_value_count_files` are not bridge inputs. They
remain staged and unchanged for later consolidation with the intermediate
count files produced by the bridge and subsequent sampled levels.

Before reading or writing point rows, `_write_bridge_level(...)` must require:

- `plan.levels[0]` is serialized Exact level 0;
- `plan.levels[1]` exists and is the serialized Bridge level;
- Exact and Bridge have identical `tile_size`, `grid_width`, and `grid_height`;
- every input manifest row belongs to Exact level 0;
- `staging_directory` is an existing directory;
- neither the Bridge point directory nor its intermediate tile/value-count
  directory already exists.

An Exact-only plan has no Bridge to construct and must fail with a clear error.
These checks make the standalone private entry point reject a mismatched plan,
input result, or staging generation before creating partial Bridge output.

#### Logical Exact-tile reconstruction

`_LevelWriteResult` is level-neutral, so the bridge writer must first require
that every input `_ManifestRow` belongs to Exact level 0. It then groups the
manifest rows by logical `(tile_y, tile_x)` and processes those logical tiles in
deterministic `(tile_y, tile_x)` order.

One dense logical Exact tile may occupy several physical Parquet row groups.
Within each tile group, order the manifest rows by `tile_shard` and require the
logical shard numbers to be exactly `0, 1, ..., n - 1`. `row_group` identifies
the physical row-group index inside `level_file`; it is not the tile-local
ordering key and need not start at zero for a tile. For example:

```text
level_file          row_group  tile_y  tile_x  tile_shard  n_points
------------------  ---------  ------  ------  ----------  ---------
bucket-007.parquet          1       8       4           0  1,000,000
bucket-007.parquet          2       8       4           1    145,108
```

Read only the referenced row groups and their complete point payload:

```text
x_rel:    float32
y_rel:    float32
value_id: uint32
point_id: uint64
```

Concatenate the decoded shards in `tile_shard` order into one complete
candidate tile. Its decoded row count must equal the sum of the grouped
manifest `n_points`; a missing, duplicate, or inconsistent shard fails before
sampling. The C5a selector receives `x_rel`, `y_rel`, and `point_id`, and its
returned original-row positions are applied to the complete four-column table.
This preserves each selected point's `value_id` without allowing values to
influence membership.

Exact and Bridge use the same planned `tile_size`, which is 512 under the
initial default, and the same logical tile coordinates. Copy `x_rel` and `y_rel`
unchanged into the Bridge payload. Coordinate rebasing begins only when several
immediate-finer tiles are assembled into one larger spatial tile.

#### Deterministic bucket-major traversal

Derive the Bridge output bucket count independently from the Bridge level's
conservative point-count upper bound:

```python
bucket_count = max(
    1,
    ceil(bridge.point_count_upper_bound / 2_000_000),
)
```

`bucket_count` sizes the physical Parquet output-file groups; it does not define
the logical tile grid. Exact and Bridge share the same logical tile geometry,
but Exact retains every point while Bridge retains at most 4,096 points per
tile. Bridge therefore normally has a much smaller point-count upper bound and
needs fewer physical buckets. Reusing the larger Exact bucket count would
unnecessarily fragment the sampled level into many small files.

The denominator is the accepted `target_rows_per_output_bucket`. Map each
complete logical tile to one of these buckets with the existing versioned
deterministic tile hash. Group only the small logical-tile descriptors at this
stage; do not redistribute or buffer their point rows.

The Exact manifest and the Bridge hash serve different purposes. A grouped
tile's Exact manifest rows identify the staged `level_file` and physical
`row_group` values from which its candidate shards must be read. Its logical
`(tile_y, tile_x)` coordinates are then hashed with the independently derived
Bridge `bucket_count` to determine where the sampled tile will be written. Do
not reuse or infer the Bridge destination from the Exact bucket filename: the
same logical tile can belong to different physical bucket numbers at the two
levels because their bucket counts differ.

Construction then follows this exact order:

```text
Exact manifest rows
    -> group into complete logical tiles
    -> assign every tile descriptor to its Bridge output bucket
    -> process output buckets by ascending bucket_id
    -> process each bucket's tiles by (tile_y, tile_x)
    -> read one complete Exact tile
    -> sample it
    -> append it to the current Bridge point and intermediate-count files
```

For each tile descriptor assigned to the current Bridge bucket, the sampling
handoff is conceptually:

```python
candidate_table = concatenate_exact_shards_in_tile_shard_order(...)
selected_indices = _select_sampled_tile_indices(
    candidate_table["x_rel"],
    candidate_table["y_rel"],
    candidate_table["point_id"],
    level=bridge.level,
    tile_x=tile_x,
    tile_y=tile_y,
    tile_size=bridge.tile_size,
    target=bridge.max_points_per_tile,
)
sampled_table = candidate_table.take(selected_indices)
```

The selector therefore determines membership from the complete Exact tile,
while applying its returned positions to `candidate_table` carries the matching
`value_id` values into the Bridge payload unchanged.

#### Bridge output contract

For every reconstructed Exact tile, the persistent Bridge tile contains
exactly:

```python
bridge_tile_count = min(
    exact_tile_count,
    bridge.max_points_per_tile,
)
```

The initial Bridge plan sets `bridge.max_points_per_tile = 4_096`. Sparse Exact
tiles therefore pass through with all their points, while denser tiles retain
exactly 4,096 representatives. In both cases, write the selected rows with the
unchanged shared point payload:

```text
x_rel:    float32
y_rel:    float32
value_id: uint32
point_id: uint64
```

Because one Bridge tile contains at most 4,096 rows and the accepted physical
row-group limit is 1,000,000 rows, every Bridge tile fits in exactly one
physical Parquet row group. Every Bridge `_ManifestRow` consequently has
`tile_shard = 0`. Its `row_group` remains the physical index within the current
Bridge bucket file and advances across tiles, for example:

```text
row_group 0 -> tile A, tile_shard 0
row_group 1 -> tile B, tile_shard 0
row_group 2 -> tile C, tile_shard 0
```

Open one Bridge point writer and its companion intermediate tile/value-count
writer for the current nonempty output bucket. Close both before advancing to
the next bucket. This keeps only one output-writer pair and one complete
candidate tile active at a time. It also gives deterministic physical output
ordering without another point-level shuffle or a shuffle-temporary directory.

Use the same deterministic filename convention as the Exact writer. Let
`filename_width = max(3, len(str(bucket_count - 1)))`, and write each nonempty
Bridge bucket to:

```text
levels/level_{bridge.level}/bucket-{bucket_id:0{filename_width}d}.parquet
intermediate_tile_value_counts/level_{bridge.level}/bucket-{bucket_id:0{filename_width}d}.parquet
```

For the initial Bridge this yields paths such as:

```text
levels/level_1/bucket-000.parquet
intermediate_tile_value_counts/level_1/bucket-000.parquet
```

Empty bucket IDs produce no files. This path policy is part of deterministic
manifest and intermediate-file descriptors; equivalent builds must not choose
filenames from task arrival order or filesystem state.

With this reconstruction contract, the bridge writer must:

- keep only one complete candidate tile in memory at a time;
- apply the C5a indices to the complete four-column point payload, preserving
  `point_id` and `value_id` unchanged;
- write self-contained bridge point files with the C3 payload and tile-owned
  row-group contract;
- emit bridge `_ManifestRow` records and intermediate tile/value-count files and
  return one `_LevelWriteResult`.

The source manifest already identifies complete logical tiles independently of
their physical shards. C5c therefore routes tile descriptors to bridge output
buckets and reads the referenced rows; it does not redistribute individual
points merely because C3 required a source-to-Exact shuffle.

#### Bridge reconciliation

Calculate the expected complete Bridge row count from the grouped Exact input
manifest before returning:

```python
expected_bridge_rows = sum(
    min(exact_tile_rows, bridge.max_points_per_tile)
    for exact_tile_rows in logical_exact_tile_row_counts
)
```

After every Bridge bucket has been written and its physical files validated,
require:

```text
sum(Bridge manifest n_points)
    = sum(written Bridge point rows)
    = sum(intermediate tile/value-count n_points)
    = expected_bridge_rows
```

The level result must also satisfy all of the following:

- every output manifest row has `level == bridge.level`;
- every logical Bridge tile contains at most
  `bridge.max_points_per_tile` rows;
- every physical `(level_file, row_group)` key is unique;
- every intermediate tile/value-count file path is unique.

Intermediate counts are exact for the representatives retained in the Bridge
output, and their complete `n_points` total must reconcile as above. Do not
compare sampled per-`value_id` totals with
`ValidatedPointsSource.value_table`: sampling intentionally changes those
totals, and `ValidatedPointsSource` is not an input to `_write_bridge_level`.
Exact source per-value reconciliation remains a later C7b responsibility and
must not be duplicated here.

### Focused tests and acceptance check

Cover a sparse tile, a dense tile, one Exact tile split across several row
groups, several tiles routed into one bridge output file, deterministic
rebuilds, and value-neutral membership.

For focused synthetic determinism coverage, two equivalent Bridge builds must
produce identical:

- selected `point_id` membership for every logical tile;
- `point_id` ordering within every tile;
- `_ManifestRow` records;
- decoded intermediate tile/value-count rows and their ordering.

Determinism is a logical cache contract. Do not compare raw Parquet bytes or
require byte-identical files.

Prepare the required Xenium Exact staging generation once, before starting the
Bridge measurement interval. Exclude Exact construction time and memory from
the reported Bridge measurements, then run `_write_bridge_level(...)` once and
record its build time, peak RSS, largest logical Exact tile rows and decoded
bytes, Bridge point count, bucket skew, and output size. The synthetic focused
test owns the repeated-build determinism check; the Xenium Bridge does not need
to be constructed more than once. Remove the prepared Exact generation, Bridge
output, intermediate count files, and measurement artifacts afterward.

### Exit criteria

- every bridge tile is a deterministic subset of its matching Exact tile;
- every bridge tile contains at most 4,096 representatives;
- bridge files, manifest rows, and intermediate counts reconcile at level-total
  scope;
- no original-source content scan or point-level reshuffle occurs;
- the measured densest-tile memory cost supports the initial in-memory policy.

### Implemented C5c result

Implemented on 2026-08-11 in `writer/bridge.py`. The sequential Bridge writer
groups Exact manifest rows into complete logical tiles, validates contiguous
tile shards, assigns tile descriptors to independently sized Bridge buckets,
reads only the referenced staged Exact row groups, applies the C5a selector,
and writes the shared four-column payload plus intermediate tile/value counts.
It does not receive `ValidatedPointsSource`, revisit the original source, or
perform a point-level shuffle.

Focused synthetic coverage reconstructs a sparse tile and a 4,100-point Exact
tile split across three physical row groups. It verifies sparse pass-through,
the 4,096 representative cap, one Bridge row group per tile, deterministic
membership and ordering across equivalent builds, identical decoded
intermediate counts across equivalent builds, and unchanged selected
`point_id` membership after changing every candidate's `value_id`. The focused
planner, sampler, writer-model, writer-support, Exact-writer, and Bridge-writer
suite passed 54 tests.

The Xenium acceptance source contained 136,578,750 Exact points. One prepared
Exact staging generation was reused for one separately measured Bridge build
with the accepted defaults. Results were:

- Bridge build time: 25.99 seconds;
- Bridge representatives: 21,722,305;
- Bridge point files: 17, containing 7,294 tile-owned row groups;
- largest, average, and smallest bucket rows: 1,440,970, 1,277,783, and
  1,157,736 respectively;
- Bridge point-file bytes: 331,497,230;
- intermediate tile/value-count files: 17, containing 9,253,957 rows and
  50,453,095 bytes;
- total incremental staged bytes: 381,950,325;
- largest logical Exact tile: 108,598 rows and 2,171,960 decoded Arrow bytes;
- measured incremental Bridge peak RSS: 76,087,296 bytes, from a
  2,704,228,352-byte post-Exact baseline to 2,780,315,648 bytes.

Validation and the one-time Exact preparation took 2.44 and 56.53 seconds
respectively and were excluded from the Bridge interval. The modest decoded
dense-tile size and incremental peak RSS support the initial one-complete-tile
in-memory policy. The benchmark's prepared Exact generation, Bridge output,
intermediate count files, JSON report, and temporary directories were removed
after the measurement.

## Slice C5d: immediate-finer tile assembly and coarser-coordinate-rebasing spike

### Goal

Demonstrate that retained immediate-finer tiles can be assembled into one
coarser-coordinate candidate table and passed to the same C5a sampler before
the complete spatial pyramid writer is implemented.

### Contract

Immediate-finer tiles are manifest and IO units used to assemble the coarser
candidate set; they are **not** sampling strata. A coarser tile has twice the
finer tile edge and receives candidates from up to four finer tiles. Rebase
each finer tile's coordinates into the coarser tile before sampling:

```text
coarser_x_rel = tile_offset_x * finer_tile_size + finer_x_rel
coarser_y_rel = tile_offset_y * finer_tile_size + finer_y_rel
```

where each tile offset is zero or one and follows deterministically from the
coarser and finer tile indices. Assign the combined candidates to the coarser
tile's own 16 × 16 microgrid and invoke `_select_sampled_tile_indices` with the
coarser level, tile key, tile size, and target. Each finer tile geometrically
covers an 8 × 8 region of that coarser grid, but allocation is performed over
the complete set of occupied coarser-tile microgrid cells. There is no separate
finer-tile allocation stage and no `value_id` influence.

The pure spike consumes bounded in-memory finer-tile candidate tables and
returns one sampled, coarser-relative payload table. It writes no persistent
level. Every coarser-level winner must already belong to the immediate finer
level, establishing nested membership by construction.

### Private in-memory boundary

Implement the pure finer-to-coarser assembly in `writer/spatial.py`. This
module is introduced only when the concrete assembly behavior is implemented;
it contains no persistent spatial writer yet.

Represent each supplied finer tile with a small private helper record:

```python
@dataclass(frozen=True)
class _FinerLevelTile:
    tile_x: int
    tile_y: int
    points: pa.Table
```

The logical coordinates are carried separately because the shared four-column
point payload does not repeat `tile_x` or `tile_y`. `points` must be a nonempty
table with the level-neutral point payload:

```text
x_rel:    float32
y_rel:    float32
value_id: uint32
point_id: uint64
```

The minimal pure entry point is:

```python
def _assemble_and_sample_coarser_tile(
    finer_tiles: tuple[_FinerLevelTile, ...],
    *,
    finer_level: _LevelBuildPlan,
    coarser_level: _LevelBuildPlan,
    coarser_tile_x: int,
    coarser_tile_y: int,
) -> pa.Table:
    ...
```

Return the complete sampled coarser-level payload rather than indices into an
internal concatenated candidate table. The future persistent spatial writer
needs the rebased four-column rows, while indices into this helper's temporary
table have no useful external meaning.

### Finer and coarser geometry

Require `coarser_level` to be the immediate planned level after `finer_level`,
to have `kind == _LevelKind.SPATIAL`, and to satisfy:

```text
coarser_level.level     = finer_level.level + 1
coarser_level.tile_size = 2 * finer_level.tile_size
```

The coarser tile coordinates must lie inside the coarser level's grid and its
effective `max_points_per_tile` must be present. For coarser tile
`(coarser_tile_x, coarser_tile_y)`, the only valid immediate-finer coordinates
are:

```text
(2 * coarser_tile_x,     2 * coarser_tile_y)
(2 * coarser_tile_x + 1, 2 * coarser_tile_y)
(2 * coarser_tile_x,     2 * coarser_tile_y + 1)
(2 * coarser_tile_x + 1, 2 * coarser_tile_y + 1)
```

Accept one through four nonempty finer tiles because dataset edges and sparse
regions may omit tiles. Reject duplicate finer-tile coordinates, tiles outside
the finer grid, and tiles that do not contribute to the requested coarser tile.
Ignore input tuple order: process valid finer tiles deterministically by
`(tile_y, tile_x)`.

### Coordinate rebasing and selection

For each finer tile, derive offsets in `{0, 1}` and rebase its coordinates:

```python
tile_offset_x = finer_tile.tile_x - 2 * coarser_tile_x
tile_offset_y = finer_tile.tile_y - 2 * coarser_tile_y

coarser_x_rel = tile_offset_x * finer_level.tile_size + finer_x_rel
coarser_y_rel = tile_offset_y * finer_level.tile_size + finer_y_rel
```

Perform rebasing calculations in `float64`, then represent the coarser-relative
payload coordinates as `float32`. Finer-tile upper-edge values are allowed by
the existing coordinate contract: a coarser-tile upper-edge value may equal
`coarser_level.tile_size` and the sampler assigns it to the final microgrid
cell.

Concatenate the rebased finer-tile payloads in deterministic tile order and call
the existing `_select_sampled_tile_indices(...)` with:

```text
level     = coarser_level.level
tile_x    = coarser_tile_x
tile_y    = coarser_tile_y
tile_size = coarser_level.tile_size
target    = coarser_level.max_points_per_tile
```

Apply the returned positions to all four columns. `value_id` is carried into
the output but is never supplied to the selector. The resulting table must:

- use the shared four-column point schema;
- contain exactly
  `min(sum(finer_tile.points.num_rows), coarser_level.max_points_per_tile)` rows;
- be ordered by ascending `point_id`;
- preserve every retained candidate's `point_id` and `value_id`;
- contain coarser-relative `x_rel` and `y_rel`;
- be a subset of the supplied immediate-finer candidates.

### Explicit exclusions

This is a pure bounded in-memory spike. It performs no Parquet IO, manifest
construction, bucket assignment, intermediate value counting, Dask execution,
persistent level writing, or Xenium benchmark. C6 owns those concerns after the
finer-to-coarser assembly and rebasing contract is approved.

### Focused tests and exit criteria

Keep the focused coverage compact:

- sparse four-finer-tile assembly covering all coordinate quadrants and
  unchanged membership;
- dense coarser-tile sampling with hard capacity compliance, nested membership,
  deterministic point ordering, and unchanged membership after changing only
  `value_id`;
- identical output after permuting finer-tile input order;
- one through three occupied finer tiles at sparse or edge coarser tiles,
  including coarser upper-edge clamping;
- rejection of duplicate, out-of-grid, or geometrically unrelated finer tiles.

C5d is complete when the rebasing rule is approved and the existing generic
sampler is shown to produce deterministic, value-neutral, nested coarser-level
output with exact capacity behavior, without introducing persistent
construction.

### Implemented C5d result

Implemented on 2026-08-11 in `writer/spatial.py`. `_FinerLevelTile` carries one
nonempty, schema-compatible finer-level payload with its logical tile
coordinates. `_assemble_and_sample_coarser_tile(...)` validates adjacent
sampled-level geometry, accepts one through four unique contributing finer
tiles, orders them deterministically, rebases coordinates in `float64`, and
returns the shared `float32` four-column payload selected by the existing C5a
sampler. It performs no Parquet IO, manifest construction, bucket assignment,
or persistent writing.

Focused coverage verifies all four coordinate quadrants, deterministic
`point_id` ordering, exact dense-tile capacity, nested and value-neutral
membership, input-order invariance, sparse edge behavior at the closed coarser
boundary, and rejection of invalid level or tile geometry. The spatial,
sampler, and build-plan focused suite passed 29 tests.

## Slice C6: complete nested spatial pyramid from the bridge

### Goal

Starting from the completed bridge, build every planned spatial level from
retained immediate-finer candidates without rescanning `points.parquet`.

### Private entry points

The complete spatial writer consumes the staged Bridge result and the immutable
build plan:

```python
def _write_spatial_levels(
    bridge_result: _LevelWriteResult,
    plan: _PointsCacheBuildPlan,
    *,
    staging_directory: Path,
) -> tuple[_LevelWriteResult, ...]:
    ...
```

The returned tuple contains one result per planned spatial level, ordered from
L1 toward the terminal overview. A valid Bridge-terminal plan returns an empty
tuple. `ValidatedPointsSource`, original source paths, Dask settings, and the
preceding levels' intermediate tile/value-count files are not inputs.

Keep one internal single-level operation:

```python
def _write_spatial_level(
    finer_result: _LevelWriteResult,
    *,
    finer_level: _LevelBuildPlan,
    coarser_level: _LevelBuildPlan,
    staging_directory: Path,
) -> _LevelWriteResult:
    ...
```

The level writer requires the implemented C5d finer-to-coarser transition
contract. It creates one complete staged spatial level and returns the same
level-neutral `_LevelWriteResult` used by Bridge construction.

### Level-by-level construction

Build only from the immediately preceding sampled level:

```text
Bridge result
→ construct L1
→ use L1 result to construct L2
→ use L2 result to construct L3
→ ...
→ terminal overview
```

This preserves the nesting rule by construction: every coarser representative
already belongs to the immediately finer level. Construct the 1,024-at-8,192,
2,048-at-16,384, and 4,096-at-32,768 levels when present in the plan, continue
with later edge/capacity-doubling levels, and honor the terminal one-tile
capacity clamp recorded by C1. Do not hard-code a maximum number of spatial
levels.

### Manifest-driven finer-tile reconstruction

The immediately finer `_LevelWriteResult.manifest_rows` is the physical input
index. Group rows into complete logical finer tiles by `(tile_y, tile_x)`, order
each tile's descriptors by `tile_shard`, and require shard indices to be
contiguous from zero. Every descriptor must belong to `finer_level`, lie inside
its grid, reference a compatible point-payload file, and agree with its physical
row-group row count.

For each requested coarser tile:

```text
identify its one through four nonempty logical finer tiles
→ read and concatenate every physical shard of each finer tile
→ construct one _FinerLevelTile per complete logical tile
→ call _assemble_and_sample_coarser_tile(...)
```

The preceding level's intermediate tile/value-count files remain untouched for
C7b and are never used to reconstruct points. Cache open `ParquetFile` handles
within the single-level writer and close every handle before returning or
propagating an exception.

### Deterministic bucket-major output

Calculate each spatial level's bucket count independently:

```python
bucket_count = max(
    1,
    ceil(coarser_level.point_count_upper_bound / target_rows_per_output_bucket),
)
```

Map the coarser logical tile coordinates through the existing versioned tile
hash. Process nonempty output buckets in ascending `bucket_id`, and each
bucket's coarser tiles in `(tile_y, tile_x)` order:

```text
coarser logical-tile descriptors
→ assign by coarser (tile_x, tile_y) to output buckets
→ open one point writer and one intermediate-count writer
→ reconstruct and sample one complete coarser candidate set at a time
→ append the sampled tile and its exact sampled value counts
```

This is sequential in the initial implementation. It performs no point-level
shuffle and adds no Dask dependency or worker setting. At most one complete
coarser candidate set and one output writer pair are active at a time.

### Physical payload, sharding, and intermediate counts

Every spatial level writes the shared self-contained point payload:

```text
x_rel:    float32
y_rel:    float32
value_id: uint32
point_id: uint64
```

For every complete sampled coarser tile, count `value_id` once before physical
sharding and append those counts to the bucket's flat intermediate
tile/value-count file. The intermediate file is returned only through its
descriptor and is not rescanned by the level writer.

Write a sampled tile in consecutive physical shards of at most
`max_rows_per_row_group=1_000_000` rows. Assign `tile_shard=0, 1, ...` and one
`_ManifestRow` per physical row group. Current planned spatial capacities are
normally below this physical limit, but retaining the general sharding contract
keeps spatial output compatible with Exact and future capacity policies.

Output bucket filenames, compression, point schema, dictionary encoding for
`value_id`, intermediate-count schema, file-footer validation, and
cache-root-relative paths reuse the accepted level-neutral writer support.

### Explicit reconciliation

For each coarser tile:

```python
expected_tile_rows = min(
    sum(complete_finer_tile_rows),
    coarser_level.max_points_per_tile,
)
```

Require its sampled table and the sum of its manifest shard rows to equal this
count. At complete-level scope require:

```text
sum(expected_tile_rows)
    = written point rows
    = manifest n_points total
    = intermediate tile/value-count n_points total
```

Also require unique physical `(level_file, row_group)` keys, unique intermediate
count-file paths, correct output level numbers, contiguous tile shards, in-grid
tile coordinates, and per-tile capacity compliance. Do not compare coarser
per-value totals with the finer level: value-neutral sampling intentionally
changes those totals.

### Focused tests

Keep persistent coverage compact and reuse the pure C5d tests for detailed
coordinate and sampler behavior. Cover:

- a Bridge-terminal plan returning no spatial results;
- one small persistent build containing multiple spatial levels;
- sparse and edge coarser tiles with one through four occupied finer tiles;
- dense capacity truncation and the terminal one-tile overview clamp;
- nested `point_id` membership across every generated level;
- unchanged `point_id` membership after changing only `value_id`;
- deterministic point order, manifest records, and decoded intermediate counts
  across equivalent builds;
- rejection of invalid finer manifest levels, non-contiguous shards, missing or
  incompatible physical row groups, and row-count disagreement.

Exact-only plan handling belongs to the later end-to-end builder because C6
requires a completed Bridge result. Do not add one test per number of strata,
values, or planned spatial levels, and do not require byte-identical Parquet
files.

### Exit criteria

- every generated spatial level is a subset of the next finer level;
- all representatives retain their exact-level identity and value;
- every sampled tile respects its effective per-tile capacity;
- the coarsest total respects the global overview budget;
- sampled construction performs no original-source content rescan;
- all planned levels are written and accounted for.

### Implemented C6 result

Implemented on 2026-08-12 in `writer/spatial.py`. The spatial coordinator
walks every planned spatial level from L1 toward the terminal overview, using
only the completed immediately finer `_LevelWriteResult` as point input. It
reconstructs complete logical finer tiles from manifest row groups, validates
contiguous physical shards and decoded row counts, rebases one through four
contributing tiles, and applies the shared deterministic value-neutral sampler.

Each spatial level independently assigns logical output tiles to deterministic
buckets, writes the shared four-column point payload in bounded physical row
groups, emits exact intermediate tile/value counts before sharding, and returns
the level-neutral manifest and count-file descriptors. Complete-level
reconciliation checks the expected sampled rows, written rows, manifest totals,
intermediate value-count totals, grid membership, shard continuity, per-tile
capacity, and terminal overview budget. Construction remains sequential,
performs no point-level shuffle, and never revisits the original points source.

Focused persistent coverage verifies a deterministic two-level spatial build,
nested and value-neutral `point_id` membership, physical output sharding,
intermediate counts, the Bridge-terminal case, and rejection of invalid finer
shards or physical row counts. The focused planning, sampling, and writer suite
passed 61 tests. The complete Xenium performance measurement remains assigned
to the end-to-end cache benchmark and hardening work.

## Slice C7a: published cache artifact contracts

### Goal

Freeze the first public cache-generation contract before implementation writes
metadata, values, the manifest, or the sparse tile/value-count index.

### Specify and freeze

- use the distinct format identifier:

  ```python
  POINTS_CACHE_SCHEMA_VERSION = "harpy-multiscale-points-cache-0.1"
  ```

  This value-generic cache is incompatible with the legacy
  `harpy-transcripts-vis-0.1` artifact and must not be accepted by its reader;
- freeze exact non-nullable Arrow schemas, column order, and metadata policy for
  `values.parquet`, `manifest.parquet`, and `tile_value_counts.parquet`. All
  three schemas and all their fields carry no custom Arrow metadata;
- retain the canonical `values.parquet` schema:

  ```text
  value_id: uint32
  value: string
  n_points: uint64
  ```

- retain the manifest schema with exactly one row per physical point row group:

  ```text
  level: int16
  level_file: string
  tile_x: uint32
  tile_y: uint32
  n_points: int64
  row_group: int32
  tile_shard: int32
  ```

  The manifest contains neither the derived `tile_id` nor `schema_version`;
- retain the sparse tile/value-count schema:

  ```text
  level: int16
  value_id: uint32
  tile_x: uint32
  tile_y: uint32
  n_points: uint64
  ```

- freeze `metadata.json` to this exact nested structure and these JSON value
  types:

  ```json
  {
    "schema_version": "harpy-multiscale-points-cache-0.1",
    "cache_generation_id": "00000000-0000-0000-0000-000000000000",
    "created_by": {
      "package": "napari-harpy",
      "version": "0.0.0"
    },
    "source": {
      "points_name": "transcripts",
      "element_path": "points/transcripts",
      "row_count": 136578750,
      "columns": {
        "x": "x",
        "y": "y",
        "value": "gene"
      },
      "selected_schema": [
        {
          "role": "x",
          "name": "x",
          "nullable": false,
          "type": {"kind": "float", "bit_width": 32}
        },
        {
          "role": "y",
          "name": "y",
          "nullable": false,
          "type": {"kind": "float", "bit_width": 32}
        },
        {
          "role": "value",
          "name": "gene",
          "nullable": false,
          "type": {"kind": "string", "offset_width": 32}
        }
      ],
      "signature_method": "harpy-parquet-source-inventory-sha256-v1",
      "signature": "...",
      "value_normalization_method": "harpy-string-trim-unicode-white-space-case-sensitive-v1",
      "point_id_policy": "harpy-source-file-row-offset-uint64-v1"
    },
    "geometry": {
      "x_origin": 0.0,
      "y_origin": 0.0,
      "x_min": 0.0,
      "x_max": 100000.0,
      "y_min": 0.0,
      "y_max": 100000.0,
      "coordinate_axes": ["x", "y"],
      "relative_coordinate_dtype": "float32"
    },
    "build": {
      "leaf_tile_size": 512,
      "overview_point_budget": 100000,
      "max_rows_per_row_group": 1000000,
      "target_rows_per_output_bucket": 2000000,
      "bucket_hash_method": "harpy-tile-splitmix64-v1",
      "sampling_method": "harpy-value-neutral-stratified-splitmix64-v1",
      "sampling_seed": 0,
      "sampling_microgrid_edge": 16
    },
    "levels": [
      {
        "level": 0,
        "kind": "exact",
        "tile_size": 512,
        "grid_width": 200,
        "grid_height": 200,
        "point_count": 136578750,
        "max_points_per_tile": null,
        "relative_directory": "levels/level_0"
      }
    ],
    "artifacts": {
      "values": "values.parquet",
      "manifest": "manifest.parquet",
      "tile_value_counts": "tile_value_counts.parquet"
    }
  }
  ```

  Numeric examples are illustrative; the field names, nesting, JSON types, and
  ordering semantics are normative. `cache_generation_id` is a canonical
  lowercase hyphenated UUID string. `levels` is ordered by ascending serialized
  level. Every level records `kind` as `exact`, `bridge`, or `spatial`;
- serialize source `selected_schema` in semantic role order `x`, `y`, `value`
  using the same normalized Arrow-type representation already frozen for the
  versioned source signature. Do not use `str(pa.DataType)` and do not serialize
  Arrow schema or field metadata;
- serialize metadata as UTF-8 bytes produced by:

  ```python
  json.dumps(
      payload,
      sort_keys=True,
      separators=(",", ":"),
      ensure_ascii=False,
      allow_nan=False,
  ) + "\n"
  ```

  Non-finite numeric metadata is therefore rejected, keys are deterministic,
  and the file ends with exactly one newline;
- freeze artifact names and require every stored path to be a normalized
  cache-root-relative POSIX path;
- implement the shared format contracts in the level-neutral
  `multi_scale_cache_points/cache_format.py`, not under `writer/`. Keep the
  private model surface to `_CacheLevelMetadata` and `_CacheMetadata`, plus the
  format and artifact constants, the three Arrow schemas, and pure
  metadata-to-payload and payload-to-metadata conversion. Do not expose or
  reuse legacy cache models;
- define successful staged validation as returning `None`; failures raise and
  no diagnostics report or partial-success object is persisted.

`metadata.json` is the source of truth for generation semantics. The manifest
is authoritative for physical point row groups and their actual stored counts.
The tile/value-count index is authoritative for selection-aware count estimates,
but never for locating point rows inside mixed-value row groups.

### Focused tests

Keep contract tests in `tests/multi_scale_cache_points/test_cache_format.py`.
Cover valid model and payload conversion plus one representative failure for
unsupported schema version, malformed metadata, invalid path, and unexpected
Arrow columns or metadata. Do not test JSON or PyArrow themselves.

### Exit criteria

- every public artifact has one exact versioned schema;
- metadata field names, types, ordering semantics, and path ownership are
  unambiguous;
- C7b and C7c can consume the contracts without inventing format details.

## Slice C7b: artifact writing and tile/value-count consolidation

### Goal

Turn the completed per-level writer results into every required final artifact
of one unpublished staging generation.

### Private entry point

Use one level-neutral operation shaped as:

```python
def _write_staged_cache_artifacts(
    validated: ValidatedPointsSource,
    plan: _PointsCacheBuildPlan,
    level_results: tuple[_LevelWriteResult, ...],
    *,
    staging_directory: Path,
    cache_generation_id: str,
) -> None:
    ...
```

C8 owns generation-ID creation and passes it into this operation. C7b requires
exactly one result for every planned level in ascending serialized-level order;
it does not infer missing levels or reconstruct the plan.

### Write final artifacts

C7b does not rewrite Exact, Bridge, or spatial point files and does not scan
their point payloads. Those files were completed by the level writers. This
slice serializes their descriptors and converts their already aggregated
intermediate counts into the final cache-level artifacts.

- write `values.parquet` from `ValidatedPointsSource.value_table` using the C7a
  schema rather than preserving arbitrary Arrow metadata;
- flatten all level results and write deterministic `manifest.parquet` rows
  sorted by `(level, tile_y, tile_x, tile_shard)`;
- read every described intermediate count file, require its descriptor and
  physical row count to agree, and retain one row for every nonzero
  `(level, value_id, tile_x, tile_y)` key;
- reject duplicate logical count keys rather than combining or repairing them;
  duplicates violate the single-bucket ownership contract;
- sort valid tile/value-count rows by
  `(level, value_id, tile_y, tile_x)` and write
  `tile_value_counts.parquet` using the C7a schema;
- write `metadata.json` from the validated source, immutable plan, actual level
  results, generation ID, and versioned constants owned by the implementation;
- write no completion marker and expose no staging path through a public API;
- remove the complete intermediate tile/value-count directory only after all
  final artifacts have been written successfully. A later failure rejects the
  whole staging generation rather than trying to restore intermediate files.

The initial count-index flow is deliberately simple:

```text
read each intermediate count file
→ concatenate its already aggregated sparse rows
→ reject duplicate (level, value_id, tile_x, tile_y) keys
→ sort once by (level, value_id, tile_y, tile_x)
→ write tile_value_counts.parquet
```

No point rows are counted again. Materializing the compact count rows for this
sort is an explicitly measurable initial policy, not a claim of bounded peak
memory. C9 records the Xenium index row count and memory cost. If that evidence
is unacceptable, replace only this consolidation step with sorted runs and a
bounded external merge.

The sparse index remains a planning index rather than a physical point locator.
Point files remain tile-co-located and are not reorganized by `value_id`.

### Explicit reconciliation before return

Require:

```text
one level result per planned level
manifest rows == all writer manifest records
manifest level totals == metadata level totals
intermediate count descriptor rows == decoded count rows
sum(tile/value n_points) == sum(manifest n_points)
```

Aggregate Exact tile/value counts by `value_id` and require them to equal the
canonical `n_points` values written to `values.parquet`. Sampled per-value totals
are allowed to change, but each sampled logical tile's value counts must equal
its own manifest total.

This is deliberately stronger than the Exact writer's earlier total-only
conservation check. Canonical counts `{0: 100, 1: 50}` must not be accepted as
`{0: 90, 1: 60}` merely because both distributions sum to 150.

### Focused tests

Use compact synthetic level results and intermediate files to cover one valid
multilevel generation, deterministic artifact rows, duplicate count keys,
descriptor disagreement, Exact per-value disagreement, wrong result ordering,
and cleanup of intermediate files after successful writing. Do not require
byte-identical Parquet output.

### Exit criteria

- the staging root contains canonical metadata, values, manifest, tile/value
  counts, and all referenced level files;
- no intermediate count directory remains after successful writing;
- artifact writing performs no original-source content rescan;
- no `COMPLETED` marker exists.

## Slice C7c: staged-cache validation

### Goal

Independently validate the complete unpublished generation written by C7b
without trusting its in-memory writer results or consulting the original Dask
graph.

### Private entry point

```python
def _validate_staged_cache(
    validated: ValidatedPointsSource,
    plan: _PointsCacheBuildPlan,
    *,
    staging_directory: Path,
) -> None:
    ...
```

Success returns `None`. Any disagreement raises and invalidates the complete
staging generation. The validator returns no diagnostics report, repairs no
artifact, and creates no completion marker.

### Validation contract

- parse `metadata.json` and require the supported schema version and exact C7a
  object contract;
- validate exact Arrow schemas, column order, nullability, and absence of
  unexpected schema or field metadata where the format forbids it;
- require every serialized path to be normalized, cache-root-relative, and
  contained by the staging root after resolution;
- validate every manifest-referenced point file and physical row group against
  the shared point-payload schema and recorded row count;
- require every physical point row group under `levels/` to appear exactly once
  in the manifest;
- reconcile physical row groups → tile shards → logical tiles → levels → the
  complete cache, including contiguous shard numbering and metadata level
  totals;
- validate `values.parquet` IDs, labels, counts, source-wide total, and equality
  with `ValidatedPointsSource.value_table`;
- validate positive and unique tile/value-count keys, known values, known
  manifest tiles, per-tile totals, and Exact per-value totals;
- validate level ordering, geometry, effective capacities, Exact membership
  totals, and the terminal overview budget;
- decode point payloads in bounded logical-tile units to validate coordinate
  ranges, `value_id` ranges, and immediate-coarser `point_id` membership as a
  subset of the corresponding finer tiles;
- require `COMPLETED` to be absent. C8 alone writes it after staged validation
  and the final fresh source-signature guard.

The staged validator checks the cache that was written. It does not rescan the
canonical source to recompute bounds, values, or row counts.

### Focused tests

Start from one tiny valid staged generation and derive a small set of
corruptions: missing files, escaped paths, wrong row-group references, count
disagreement, duplicate or invalid tile/value-count records, tile/count
reconciliation failure, Exact per-value disagreement, schema mismatch or an
unexpected manifest column such as `tile_id` or `schema_version`, budget
overflow, and non-nested membership. Avoid one test per metadata field.

### Exit criteria

- one staged generation is self-consistent without consulting a Dask graph;
- metadata, manifest, values, and tile/value counts are sufficient for the
  future Phase 2 store and selection-aware planner;
- every physical row group is represented exactly once in the manifest;
- validation returns no partial success and writes no completion marker.

## Slice C8: guarded end-to-end builder and local publication

### Goal

Compose planning, exact writing, sampled writing, cache metadata, staged
validation, source guards, and local publication into the first supported builder.

### Required flow

```text
ValidatedPointsSource + resolved logical planning arguments
→ fresh source signature == validated signature
→ C1 immutable build plan
→ create unique sibling staging generation
→ write exact and sampled levels
→ write values, tile/value counts, metadata, and manifest
→ validate complete staging generation
→ fresh source signature == validated signature
→ write COMPLETED
→ install completed generation at transcripts_vis/
```

### Implement

- fail the initial metadata-only source guard before staging is created;
- apply the public defaults `leaf_tile_size=512` and
  `overview_point_budget=100_000`, validate them through C1, and invoke C1 exactly
  once after the initial source guard;
- generate a fresh cache-generation ID;
- create and own a unique sibling staging directory;
- pass the resulting immutable plan records to C3, C5c, C6, C7b, and C7c without
  rebuilding validation facts;
- fail the final metadata-only source guard after staged validation and before
  completion;
- write `COMPLETED` only after every preceding step succeeds;
- publish the completed local directory with the approved replacement protocol;
- preserve an existing completed cache when any build or guard fails;
- reject and clean incomplete staging according to the frozen recovery policy;
- expose a small public builder accepting `ValidatedPointsSource` plus keyword-
  only `leaf_tile_size` and `overview_point_budget` arguments with those defaults;
- expose a backed-SpatialData convenience entry point that delegates visibly
  through resolution, validation, and the primary builder.

Progress, cancellation, overwrite behavior, and the exact returned build result
must be frozen before C8 implementation. They must not leak temporary paths or
partially completed metadata into the public contract.

### Focused tests

Cover first publication, successful replacement, failure before staging, failure
during writing, staged-validation failure, final source-signature mismatch,
publication failure, preservation of an existing completed cache, cleanup, and
absence of canonical-source mutation. Inject failures at Harpy boundaries rather
than testing operating-system rename implementation details exhaustively.

### Exit criteria

- the final path is absent or a complete validated generation, never staging;
- both source guards use fresh inventories and the original validated signature;
- no guard repeats the point-content validation scan;
- failures publish nothing incomplete and preserve the previous generation;
- the primary builder requires no SpatialData object or Dask graph after it
  receives `ValidatedPointsSource`.

## Slice C9: Xenium construction benchmark and hardening

### Goal

Demonstrate that the complete Phase 1 builder is correct and operationally
reasonable on the 136,578,750-row Xenium acceptance source before Phase 2 begins.

### Benchmark tool

Add an explicit, opt-in developer script that receives the SpatialData path,
points name, build configuration, output location, run label, and JSON result
path. It must not hardcode private data paths or run in normal CI.

Record:

- source signature and all build parameters;
- exact, sampled, metadata, validation, guard, and publication times;
- peak RSS and configured concurrency;
- rows, bytes, files, row groups, and logical tiles per level;
- complete cache size and overhead relative to the source;
- manifest rows and bytes;
- tile/value count rows, bytes, construction overhead, and representative
  selected-value lookup latency;
- sampled counts, capacity utilization, nesting, and coarsest total;
- coordinate reconstruction error;
- files and row groups touched for representative tiles and viewports;
- representative cold and warm PyArrow tile-read latency;
- cleanup and replacement behavior;
- package, machine, and storage context.

The benchmark output is diagnostic and is not stored in cache metadata. Failed
or benchmark-only cache generations are removed after their results are recorded.

### Hardening policy

- profile before introducing concurrency or native extensions;
- optimize only measured bottlenecks;
- reopen the bucket-count, spill, grouping, or sharding decision if locality,
  build cost, or tile-read latency is poor;
- reopen the sampler gate if spatial coverage, nesting, determinism, or measured
  value neutrality is unacceptable;
- record target misses with an explicit accept, optimize, or redesign decision;
- do not weaken correctness, determinism, the accepted measured memory envelope,
  or publication safety merely to reduce build time.

### Exit criteria

- all exact and sampled correctness invariants pass on the acceptance cache;
- build time, memory, disk size, and fragmentation are recorded;
- the exact writer and sampler decisions remain supported or are explicitly
  revised;
- the resulting completed cache is suitable as the Phase 2 runtime-store
  acceptance artifact;
- Gate E approves beginning runtime store, planner, and scheduler work.

## Review gates

### Gate A: after C1

Approve:

- minimal logical construction models;
- grid origin, boundary, and serialized-level conventions;
- exact-only and multilevel planning behavior;
- ownership and suitability of the two logical public defaults needed by the
  Dask writer.

### Gate B: after C3

Decision on 2026-08-07: approved. The Dask writer satisfies the initial Xenium
acceptance target; keep Dask, defer optional C4 indefinitely, and proceed to C5a.
The measured bucket skew and peak RSS do not justify a strict oversized-bucket
fallback or file rollover in the first implementation.

Approve:

- exact tile size;
- Dask exact-writer correctness and the small acceptance benchmark;
- whether Dask is accepted or optional C4 is opened for a named limitation and
  measurable success criterion;
- correct implementation of `harpy-tile-splitmix64-v1`,
  `target_rows_per_output_bucket=2_000_000`, its resulting 69-bucket Xenium
  Exact configuration, and deterministic file naming;
- `max_rows_per_row_group=1_000_000` as the initial physical tile-shard limit;
- Dask shuffle configuration, one-at-a-time finalization, observed maximum
  bucket size, and peak RSS;
- whether those measurements justify changing the bucket configuration or
  implementing a strict oversized-bucket fallback or file rollover;
- dense-tile output row-group sharding, without treating it as a
  bucket-finalization memory bound;
- the local no-task-retry execution contract and deterministic single-owner
  bucket output;
- bounded read/write strategy and concurrency envelope;
- coordinate reconstruction tolerance;
- exact-writer performance viability.

### Gate C: after C5d

Decision on 2026-08-12: approved. The implemented C5a sampler, C5c persistent
Bridge result, and C5d finer-to-coarser spike establish the initial sampled
pyramid contract. Proceed to C6 without adding a streaming sampler or
value-aware membership policy.

Approved:

- sampler name and version;
- the shared C5a 16 × 16 current-tile microgrid at every sampled level;
- C5d finer-tile assembly, coarser-coordinate rebasing, and boundary behavior;
- the initial one-complete-current-tile in-memory policy and C5c Xenium memory
  evidence;
- proportional microgrid-cell target allocation with no `value_id` influence;
- hash, seed, tie-breaking, and output ordering;
- deterministic, nested, spatial, value-neutral, and capacity behavior.

### Gate D: after C7c

Approve:

- cache schema version;
- payload, values, tile/value counts, metadata, and manifest contracts;
- staged-cache validation and accounting;
- compatibility boundary expected by the future Phase 2 reader.

### Gate E: after C9

Approve:

- complete Xenium build correctness;
- measured build time, memory, size, and fragmentation;
- physical-layout and sampler decisions after real-data evidence;
- publication and cleanup behavior;
- readiness for the runtime store, planner, and scheduler.

## Phase 1 definition of done

Phase 1 is complete when:

- construction accepts `ValidatedPointsSource` and imports no legacy writer;
- the canonical SpatialData source remains unchanged;
- source-signature guards run before staging and before publication;
- the exact level has complete membership and point identity;
- exact tiles are physically co-located independently of source partition
  boundaries;
- tile-local coordinate reconstruction meets the frozen tolerance;
- sampled levels are deterministic, nested, spatially stratified, and
  value-neutral;
- every tile and level respects its capacity or global budget;
- `values.parquet`, `tile_value_counts.parquet`, metadata, manifest, level files,
  and row groups reconcile;
- every stored path is cache-root-relative and contained by the cache root;
- a cache without `COMPLETED` is never accepted as complete;
- first publication and replacement cannot expose an incomplete generation;
- failed construction preserves any existing completed cache;
- construction time, peak memory, disk size, and fragmentation are documented on
  the Xenium acceptance source;
- Gate E approves the completed cache as the Phase 2 acceptance artifact;
- all focused construction tests pass.

## Immediate next slice

Specify **C7a: published cache artifact contracts** around the completed Exact,
Bridge, and spatial writer results. Then implement C7b artifact writing and C7c
staged-cache validation without reopening the frozen format during those
implementation slices.
Optional C4 remains deferred indefinitely unless new evidence identifies a
concrete Dask limitation and measurable PyArrow success criterion.
