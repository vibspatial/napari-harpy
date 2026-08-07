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

The high-level slices below are ready, but these details must be frozen at their
named review gates rather than guessed during implementation:

- whether the first public format redefines `harpy-transcripts-vis-0.1` or uses
  a new `0.2` schema version;
- the public build-result model and any later public builder parameters beyond
  the two logical planning arguments;
- any future change to C1's implemented grid-origin normalization or exact
  maximum-boundary behavior;
- the remaining Arrow details for the locked manifest column set;
- whether measured C3 results justify the optional C4 direct-PyArrow
  investigation;
- deterministic bucket filename width; the bucket hash and the initial
  bucket-count heuristic are locked below;
- Dask partition and shuffle configuration, plus direct-PyArrow spill and
  compaction configuration only if optional C4 is opened;
- whether measured C3 bucket skew or peak memory justifies a later maximum
  in-memory bucket size, recursive spill or external-grouping fallback, and
  file rollover;
- proportional spatial-stratum allocation and deterministic integer-remainder
  behavior;
- same-geometry bridge stratification;
- sampler priority-hash algorithm, seed representation, and collision
  tie-breaking;
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
  writer_models.py
  exact_level.py
  sampling.py
  parquet_writer.py
  manifest.py
  publication.py
```

`schema.py` is added only when C3 needs to materialize the locked point payload
as an Arrow schema or after Gate D freezes the complete manifest and
metadata schema; C1 must not create it merely to mirror the legacy module.

C1 owns `build_plan.py` and its private plan records. Small helpers should remain
in their consuming module rather than creating speculative modules.

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
| C2 | Implemented; obsolete source-partition setting removal pending | Minimal exact-level construction contracts | No | No |
| C3 | Planned | Dask exact-level writer and acceptance benchmark | Yes | No |
| C4 | Optional | Direct-PyArrow exact-writer investigation, only if C3 justifies it | Yes | No |
| C5 | Planned | Versioned value-neutral sampling contract and spike | No original-source rescan | No |
| C6 | Planned | Complete nested sampled pyramid | No original-source rescan | No |
| C7 | Planned | Metadata, values, manifest, tile/value counts, and staged-cache validation | No | No |
| C8 | Planned | Guarded end-to-end builder and local publication | Through level builders | Yes |
| C9 | Planned | Xenium construction benchmark and hardening | Yes | Benchmark only |

Each slice must be independently reviewable. C3 implements one credible writer
rather than two competing engines. C4 is not on the critical path: it is opened
only when Gate B concludes that the measured Dask writer is insufficient or
that a direct-PyArrow alternative has a concrete, testable advantage. C5 is a
deliberate sampler spike whose artifacts are internal and disposable.

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
and edge-doubling schedule keep child and parent grids aligned.

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
not record a C5 sampler name, version, or parameters. The relative level
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

The contracts below are implemented in `writer_models.py`, covered by the
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
    finalizer_concurrency
        Maximum number of complete shuffle buckets that may be computed,
        sorted, grouped by tile, and written concurrently. This bounds local
        writer parallelism and its combined memory and storage pressure; fewer
        finalizers may be active when less work is available.
    """

    bucket_count: int
    max_rows_per_row_group: int
    finalizer_concurrency: int


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
class _TileValueCountFragment:
    level: int
    relative_path: str
    row_count: int


@dataclass(frozen=True)
class _LevelWriteResult:
    manifest_rows: tuple[_ManifestRow, ...]
    tile_value_count_fragments: tuple[_TileValueCountFragment, ...]
```

The result is level-neutral: the Exact writer and every later sampled-level
writer return the same record type. Exact-specific behavior remains in the
writer entry point and `_ExactLevelWriterConfig`, while C7 can consolidate a
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

`_TileValueCountFragment` describes one staged, flat, provisional count file
emitted by one level-writer finalization unit. `level` follows the parent
manifest's non-negative `int16` range, `relative_path` is a normalized
cache-root-relative POSIX path, and `row_count` is positive. Descriptors are
unique and returned in deterministic `(level, relative_path)` order. C3 freezes
the provisional fragment directory and filename convention.

Each fragment contains flat nonzero
`(level, value_id, tile_x, tile_y, n_points)` rows with the logical field types
required by the final tile/value-count index. A tile finalizer may temporarily
aggregate `{value_id: count}` for its current tile, or use an equivalent Arrow
aggregation, but it must append those counts to a bucket-local flat fragment
and release the per-tile mapping. It must not retain one Python object or one
Python dictionary for every nonzero combination across the complete level.

Every logical tile belongs to exactly one writer bucket. That bucket finalizer
therefore owns the complete value counts for the tile, including points written
through several dense-tile row-group shards. It must aggregate those shards and
emit exactly one fragment row for each nonzero
`(level, value_id, tile_x, tile_y)` key. The same logical key must not occur in
another fragment.

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

The finalizer may transiently represent them as `{3: 3, 8: 1}`, but the flat
provisional fragment stores:

```text
level  value_id  tile_x  tile_y  n_points
-----  --------  ------  ------  --------
0      3         4       8       3
0      8         4       8       1
```

One bucket-local fragment normally contains such rows for several logical
tiles. `_TileValueCountFragment.row_count` is the number of aggregated nonzero
`(level, value_id, tile_x, tile_y)` rows in that fragment; it is not the number
of original point rows. The example fragment therefore contributes
`row_count=2` for the displayed tile.

Taken together, a level's fragments contain every nonzero tile/value count. For
every exact logical tile, those counts sum to the `n_points` total across the
tile's manifest rows. Across Exact level 0, each `value_id` total reconciles to
the corresponding exact count in `ValidatedPointsSource.value_table`.

#### Why the count fragments are provisional

Counts are derived while a bucket finalizer already has the tile rows needed to
write the point payload. Persisting them at that moment prevents C7 from
rereading every completed point row group, or the canonical source, solely to
recalculate counts. For Exact on the Xenium acceptance source, such a rescan
would revisit all 136,578,750 points.

Independent bucket finalizers must not append concurrently to one shared
`tile_value_counts.parquet`. Parquet has one writer-owned footer, and routing
all counts through a central in-memory writer would add coordination, buffering,
and failure-handling complexity. Each finalizer therefore writes one
bucket-local flat fragment that it owns exclusively.

The counts in a fragment are exact, not approximate, but the fragment is
provisional because it covers only one finalization unit. It is not yet the
complete, globally value-ordered, reconciled runtime index. After every required
level has been written, C7 performs the reduction step:

```text
all bucket-local count fragments
→ read in bounded batches
→ order by (level, value_id, tile_y, tile_x)
→ validate that every logical key occurs exactly once
→ reconcile with manifest rows and exact value totals
→ write and validate tile_value_counts.parquet
→ remove the provisional fragments
```

Provisional count fragments are distinct from Dask shuffle-temporary files.
Shuffle files are execution scratch and are removed once their bucket has been
finalized. Count fragments contain semantic construction results and must
survive until C7 has successfully written and validated the final index.

The result retains tuples of small descriptors, not all count rows or Arrow
tables in memory. C7 reads the provisional fragments in bounded batches,
consolidates and sorts them into the final Arrow tile/value-count index, and
removes them. C2 itself does not materialize Arrow cache schemas.

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
    │       │       ├── writes one provisional count fragment
    │       │       └── produces one _TileValueCountFragment descriptor
    │       │
    │       └── returns _LevelWriteResult
    │               ├── manifest_rows
    │               └── tile_value_count_fragments
    │
    ├── calls C6 sampled-level writers
    │       └── each level returns another _LevelWriteResult
    │
    └── passes all _LevelWriteResult objects to C7
            ├── writes manifest.parquet from _ManifestRow records
            ├── streams the provisional count fragments
            ├── writes tile_value_counts.parquet
            ├── removes the provisional count fragments
            └── validates the staged cache
```

The point bucket files are persistent members of the staged cache generation;
they are not Dask shuffle scratch. The count fragments are construction-only
handoff artifacts and disappear after C7 has created and validated the final
index. After this handoff returns successfully, C8 performs its final source-
signature guard, completion-marker write, and local publication steps.

### Directory ownership

The caller creates and owns one unique, initially empty
`staging_directory`. The exact writer may create only the Exact level subtree
derived from the plan and must fail rather than overwrite an existing target.
It never removes the caller's staging root, writes `COMPLETED`, publishes the
generation, or touches an existing completed cache. The caller rejects and
removes the complete staging generation after any construction failure.

Provisional tile/value-count fragments are staged construction artifacts, not
shuffle-temporary files. They remain inside `staging_directory` after a level
writer returns, appear in `_LevelWriteResult` only through cache-root-relative
descriptors, and survive until C7 has written and validated the consolidated
index. C7 then removes the provisional fragments before publication.

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
- returned manifest rows and tile/value-count fragment descriptors completely
  describe the written Exact level and its provisional planning counts;
- all output paths are cache-root-relative and all writes remain inside the
  caller-owned staging generation.

Detailed source traversal, annotation, normalization-helper refactoring,
bucket hashing, Dask shuffle construction, bucket-size measurement,
concurrency, and measurements belong to C3. Strict oversized-bucket handling is
added later only if the acceptance benchmark demonstrates the need. Direct
PyArrow belongs only to optional C4.

### Focused tests

Test only semantic record validation: invalid configuration counts, unsafe or
non-normalized manifest and fragment paths, invalid integer ranges, and invalid
nonpositive counts. Do not test dataclass immutability, Python tuple behavior,
Dask, or PyArrow. Fragment contents, deterministic ordering, duplicate logical
keys, and cross-record reconciliation require actual writer output and are
tested in C3 and C7.

### Exit criteria

- C3 can implement the documented entry contract without inventing another
  build plan or duplicating logical geometry arguments;
- the staging and temporary-directory owner is unambiguous on success and
  failure;
- provisional row-group records and tile/value-count fragment descriptors are
  sufficient for later staged-cache validation without retaining every sparse
  count as a Python object;
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
  count, and select the deterministic bucket filename width;
- use the locked output row-group size and finalizer concurrency, and configure
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
level-manifest rows, and its flat provisional tile/value-count fragment. Counts
are derived while tile rows are already available. A per-tile value-count
mapping may exist only transiently while that tile is finalized; after its flat
rows are appended to the bucket fragment, the mapping is released. If the tile
uses several physical row-group shards, the finalizer aggregates across those
shards before emitting one row per nonzero logical key. The result returns one
small fragment descriptor rather than one Python record per nonzero count. No
additional source or completed-level scan is allowed.

The finalizer should consume numeric data and write through PyArrow. It must not
copy the legacy per-partition Pandas string construction, schema, or direct
side-effect pattern merely because that code exists.

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
needed by C7; Gate D owns the remaining manifest Arrow details but does not
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
`max_rows_per_row_group=1_000_000` and `finalizer_concurrency=1`. For the Xenium
target this means 69 buckets and at most 69 Exact-level bucket files. One
representative benchmark run is
sufficient for the first decision; do not require a parameter sweep,
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

Gate B records one of two outcomes:

1. **Dask accepted:** correctness and the small benchmark are sufficient; keep
   Dask as the Phase 1 exact writer and proceed directly to C5.
2. **PyArrow investigation justified:** record the concrete Dask limitation and
   the success criterion that a direct-PyArrow experiment must meet; open
   optional C4 before proceeding.

## Slice C4: optional direct-PyArrow exact-writer investigation

### Goal

Investigate a direct-PyArrow spill-and-compaction writer only when Gate B records
a concrete reason that the C3 Dask writer is insufficient. This slice is
skipped when the Dask result is acceptable.

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

## Slice C5: versioned value-neutral sampling contract and spike

### Goal

Freeze and demonstrate the deterministic sampling algorithm before building the
complete sampled pyramid.

### Contracts fixed before C5

The sampler uses the following fixed algorithm family:

```text
finer candidates
→ current-level tile and spatial stratum
→ tile target allocated across occupied strata
→ stable value-independent point priority
→ deterministic winners
→ deterministic output order
```

Every sampled level consumes only representatives retained by its immediate
finer level, so membership is nested and every winner remains an actual source
point with its unchanged `point_id` and `value_id`. For L1 and every later
spatial level, the immediate finer child tiles are the top-level spatial
strata. The tile capacity is allocated proportionally across occupied strata,
and candidates are selected within those allocations by ascending stable
pseudo-random priority. Stateful random-number-generator order, input partition
order, and Python's randomized `hash()` must not affect membership.

`value_id` must not affect stratum allocation, point priority, or winner
selection. The sampler intentionally represents local source abundance rather
than guaranteeing preservation of rare values. Explicit value selections are
handled later by the per-level tile/value count index and selection-aware LOD
planner, not by scientifically distorting the sampled membership.

### Details C5 must refine and freeze

C5 must settle:

- same-geometry bridge stratification, with a fixed micro-grid evaluated as a
  candidate rather than assumed;
- exact proportional integer allocation, deterministic remainder distribution,
  and redistribution when a stratum cannot consume its provisional allocation;
- the versioned stable hash algorithm, seed encoding, and priority payload;
- deterministic collision and final tie-breaking;
- the deterministic physical output sort after winner selection;
- behavior when occupied spatial strata outnumber the target;
- behavior for coincident coordinates, dominant values, singleton-heavy values,
  sparse tiles, and tiles split across physical shards.

The spike operates on bounded candidate tables or exact-cache tiles. It does not
rescan the original canonical source and does not write a completed cache.

### Evaluation

Use small exact fixtures plus value-skewed, spatially skewed, and dense synthetic
tiles. Include representative Xenium tiles. Measure:

- deterministic and nested membership;
- spatial coverage;
- sampled value proportions compared with the finer candidates, as a diagnostic
  for unintended systematic value bias rather than a rare-value preservation
  target;
- hard capacity compliance;
- runtime and peak memory;
- adjacent-level count ratios and transition stability proxies.

### Exit criteria

- the complete sampler has one name and versioned parameter contract;
- every winner is an actual finer-level candidate with unchanged `point_id`;
- the same candidates and parameters always produce the same winners;
- no tile or level target can be exceeded;
- `value_id` has no influence on allocation, priority, or membership;
- the bridge and spatial-level stratification rules are approved for C6.

## Slice C6: complete nested sampled pyramid

### Goal

Build every planned sampled level from retained finer-level candidates without
repeatedly scanning `points.parquet`.

### Implement

- the 512-at-4,096 sampled finest bridge from exact candidates;
- 1,024-at-8,192, 2,048-at-16,384, and 4,096-at-32,768 spatial levels;
- later edge/capacity-doubling levels when required by the plan;
- the terminal one-tile capacity clamp when required by the plan;
- deterministic parent formation from finer child tiles;
- the C5 value-neutral spatial allocation and stable point priorities;
- unchanged `point_id` and `value_id` propagation;
- self-contained payloads at every level;
- the C3 physical sharding and manifest-row contract for sampled levels;
- derive each sampled level's physical `bucket_count` independently as
  `ceil(level.point_count_upper_bound / target_rows_per_output_bucket)`, with a
  minimum of one, rather than reusing the Exact bucket count;
- flat provisional tile/value-count fragments emitted while sampled tiles are
  finalized, with only fragment descriptors retained and no additional level
  scan;
- bounded candidate memory and bounded writer concurrency.

### Focused tests

Cover exact-only small sources, sparse tiles, four-child parent formation, the
same-geometry bridge, dense capacity truncation, a value-skewed fixture,
the terminal one-tile capacity clamp, nested membership, and deterministic
rebuilds. Verify that changing only value labels does not change sampled
membership. Do not require one test for every possible number of occupied
strata or values.

### Exit criteria

- every generated level is a subset of the next finer level;
- all representatives retain their exact-level identity and value;
- every sampled tile respects its effective per-tile capacity;
- the coarsest total respects the global overview budget;
- sampled construction performs no original-source content rescan;
- all planned levels are written and accounted for.

## Slice C7: metadata, values, manifest, tile/value counts, and staged-cache validation

### Goal

Turn writer outputs into a complete but unpublished cache generation whose
semantics and physical accounting can be validated independently.

### Implement

- freeze the cache schema version before writing publicly consumable artifacts;
- write `values.parquet` directly from the validated canonical value table;
- write deterministic `manifest.parquet` rows sorted by
  `(level, tile_y, tile_x, tile_shard)`;
- read the writers' provisional count fragments in bounded batches and
  consolidate them into
  `tile_value_counts.parquet`, with exactly one row per nonzero
  `(level, value_id, tile_x, tile_y)` tuple and this logical schema:

  ```text
  level: int16
  value_id: uint32
  tile_x: uint32
  tile_y: uint32
  n_points: uint64
  ```

  Reject duplicate logical keys rather than combining or repairing them; a
  duplicate violates the single-bucket ownership contract. Sort the valid rows
  by `(level, value_id, tile_y, tile_x)`. It is a planning index, not a
  physical point locator; `manifest.parquet` remains authoritative for files and
  row groups. The first physical point layout remains tile-co-located and is not
  value-sharded;
- write no manifest `tile_id` or `schema_version` column; derive the former from
  the numeric tile key and store the latter once in `metadata.json`;
- write `metadata.json` with cache identity, source identity, geometry, ordered
  level records, build parameters, value-normalization method, point-id policy,
  sampler version, writer layout, coordinate dtype contract, and the tile/value
  count-index path and method;
- use cache-root-relative POSIX paths only;
- validate exact Arrow schemas and absence of unexpected metadata where the
  format requires it;
- validate every referenced file and row group;
- reconcile shard → tile → level → cache row counts;
- validate that every tile/value count is positive, every value ID and tile key
  exists, and no nonzero tuple is duplicated;
- reconcile tile/value counts to the manifest total for every logical tile and
  reconcile exact-level per-value totals to `values.parquet.n_points`;
- validate exact membership totals, nested sampled counts, capacities, terminal
  overview budget, level ordering, and path containment;
- reject an absent or premature artifact without creating `COMPLETED`.

The staged validator checks the cache that was written. It does not rescan the
canonical source to recompute bounds, values, or row counts.

### Focused tests

Start from one tiny valid staged generation and derive a small set of corruptions:
missing files, escaped paths, wrong row-group references, count disagreement,
duplicate or invalid tile/value count records, tile/count reconciliation
failure, exact per-value disagreement, schema mismatch or an unexpected
manifest column such as `tile_id` or `schema_version`, budget overflow, and
non-nested membership where validated at this phase. Avoid one test per metadata
field.

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
- pass the resulting immutable plan records to C3, C6, and C7 without rebuilding
  validation facts;
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

### Gate C: after C5

Approve:

- sampler name and version;
- bridge and spatial stratification;
- proportional spatial-stratum target allocation with no `value_id` influence;
- hash, seed, tie-breaking, and output ordering;
- deterministic, nested, spatial, value-neutral, and capacity behavior.

### Gate D: after C7

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

Review the implemented **C2: minimal exact-level construction contracts**, then
write the detailed specifications for **C3: Dask exact-level writer and
acceptance benchmark**. Open optional C4 only if Gate B records a concrete Dask
limitation and a measurable success criterion for direct PyArrow. The public
builder still applies the agreed logical defaults later; version strings,
public result models, and construction-specific errors remain deferred to their
consuming slices. Do not settle sampler details prematurely.
