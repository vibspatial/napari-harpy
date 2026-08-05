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
- value-selective tile IO;
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
- self-contained, nested, value-aware sampled levels;
- the initial 512-all → 512-at-4,096 → 1,024-at-8,192 →
  2,048-at-16,384 → 4,096-at-32,768 progression;
- further spatial levels when needed to satisfy the global overview budget;
- deterministic `values.parquet`, metadata, manifest, files, row groups, and
  shard numbering;
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
centroids or a raster. Serialized levels are ordered from coarsest to finest:

```text
level_0 ⊆ level_1 ⊆ ... ⊆ level_n
level_n = exact source membership
```

The runtime will render one self-contained level at a time. Residual or disjoint
levels are not part of the first format.

### Initial level schedule

The initial target is:

| Design label | Tile geometry | Maximum rows per tile |
|---|---:|---:|
| Exact | 512 | all source rows |
| Sampled finest bridge | 512 | 4,096 |
| L1 | 1,024 | 8,192 |
| L2 | 2,048 | 16,384 |
| L3 | 4,096 | 32,768 |
| Later spatial levels | double preceding edge | initially double preceding capacity |

Capacity is a maximum, not a fill target. Sparse tiles retain all candidates.
The terminal coarsest level is globally allocated so that:

```text
sum(manifest.n_points where level == 0) <= overview_point_budget
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
bounded implementation, not a separate competing layout. The construction spike
selects the bucket, spill, grouping, and sharding parameters; it does not compare
against partition-local Layout A as a production candidate.

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
- the exact public build configuration and result models;
- the default `overview_point_budget` and `max_rows_per_row_group`;
- grid-origin normalization and exact maximum-boundary behavior;
- the provisional minimal manifest fields and whether any derived field must be
  persisted;
- writer engine selection between the focused Dask and direct-PyArrow
  candidates;
- the stable bucket hash, bucket count, and deterministic file naming;
- engine-specific partition, spill, shuffle, and compaction configuration;
- maximum in-memory finalization bucket size, recursive spill or external-
  grouping fallback, file rollover, and cleanup;
- exact value-aware allocation weights and minimum-allocation behavior;
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
  exact_level.py
  sampling.py
  parquet_writer.py
  manifest.py
  publication.py
```

`schema.py` is added only when C2/C3 needs to materialize the locked point
payload as an Arrow schema or after Gate D freezes the complete manifest and
metadata schema; C0 must not create it merely to mirror the legacy module.

A pure planner may justify `build_plan.py`; do not add it before Slice C1 makes
that boundary concrete. Small helpers should remain in their consuming module
rather than creating speculative modules.

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
| C0 | Next | Minimal logical construction contracts | No | No |
| C1 | Planned | Pure grid and level build planning | No | No |
| C2 | Planned | Exact-level writer architecture and bucketed-shuffle spike | Yes | No |
| C3 | Planned | Production exact-level writer | Yes | No |
| C4 | Planned | Versioned value-aware sampling contract and spike | No original-source rescan | No |
| C5 | Planned | Complete nested sampled pyramid | No original-source rescan | No |
| C6 | Planned | Metadata, values, manifest, and staged-cache validation | No | No |
| C7 | Planned | Guarded end-to-end builder and local publication | Through level builders | Yes |
| C8 | Planned | Xenium construction benchmark and hardening | Yes | Benchmark only |

Each slice must be independently reviewable. C2 and C4 are deliberate spikes;
their artifacts are internal and disposable, but their measured decisions are
recorded before production code depends on them.

## Slice C0: minimal logical construction contracts

### Goal

Create only the immutable logical construction contracts required by the pure
planner and the C2 spike without freezing a physical cache schema.

### Expected files

```text
src/napari_harpy/core/multi_scale_cache_points/models.py
src/napari_harpy/core/multi_scale_cache_points/errors.py
tests/multi_scale_cache_points/test_cache_contracts.py
```

### Implement

- the minimal build configuration needed by C1 and C2;
- immutable level and cache-plan records only when their fields are frozen;
- ownership placeholders for later cache-schema, sampling-policy, and
  writer-policy versions without inventing their values or physical fields;
- construction-specific errors rooted in the existing points validation error
  family only when callers need to distinguish them;
- narrow package exports; internal planning and writer records remain private.

### Do not

- open source Parquet files;
- create directories or write cache artifacts;
- expose speculative runtime-store or renderer models;
- define level-payload, manifest, or metadata Arrow schemas;
- freeze metadata fields whose semantics are still owned by C6;
- copy all legacy cache dataclasses merely because they exist.

### Focused tests

Use a few direct tests for construction parameters that affect correctness, such
as positive budgets and incompatible combinations. Do not test dataclass or
PyArrow behavior itself, and do not add schema tests before the schema-owning
gate.

### Exit criteria

- C1 and C2 can accept typed configuration without dictionaries of unrelated
  options;
- concrete Arrow schema objects remain deferred to their consuming slices even
  though the point-payload columns and types are already locked;
- the public API remains small and independent of Qt, napari, and VisPy.

## Slice C1: pure grid and level build planning

### Goal

Convert `ValidatedPointsSource` facts plus construction configuration into a
deterministic, IO-free build plan.

### Implement

- freeze `x_origin` and `y_origin` rules;
- calculate exact tile indices from validated bounds using half-open cells;
- handle points on tile boundaries, including the source maximum;
- distinguish tile geometry from sampling capacity;
- create the exact-only plan when source count fits the overview budget;
- otherwise create the exact level, sampled finest bridge, required spatial
  progression, and terminal global-budget level;
- order serialized level records from coarsest to finest while preserving clear
  design labels internally;
- record per-level tile size, capacity or global allocation, exactness, sampling
  policy, and output directory;
- reject impossible integer grid shapes or serialized level identifiers before
  construction starts.

The planner may use conservative count bounds. It does not inspect source rows
or predict the exact sampled count of every tile.

### Focused tests

Cover a small exact-only source, a large source requiring the bridge and several
spatial levels, maximum-boundary coordinates, negative coordinates, and the
terminal overview-budget rule. Avoid combinatorial extent/budget tests.

### Exit criteria

- identical validated facts and configuration produce an identical plan;
- the plan contains enough information for the exact writer and sampler without
  consulting a viewport or renderer;
- no filesystem or point-row IO occurs.

## Slice C2: exact-level writer architecture and bucketed-shuffle spike

### Goal

Select and demonstrate an efficient bounded writer architecture for the required
tile-co-located exact level without presuming that the legacy Dask/Pandas writer
or its physical schemas are the correct starting point.

### Spike contract

- consume `ValidatedPointsSource` directly;
- use the agreed initial 512-unit exact tile geometry; do not spend this spike
  comparing 256-unit tiles unless 512 demonstrates a concrete failure;
- read only the selected columns through bounded operations driven by the
  validated ordered physical inventory;
- generate batch-oriented internal `point_id` arrays;
- normalize and map source values to the validated `value_table` without
  Python-per-row string handling;
- assign tiles from `float64` working coordinates and produce tile-local
  `float32` coordinates;
- use numeric construction keys;
- map each logical tile to a deterministic writer bucket using an explicit
  stable hash rather than Python's built-in `hash()`;
- perform the unavoidable full logical redistribution through bounded local
  disk-backed bucket storage;
- group or sort every completed bucket by `(tile_y, tile_x, point_id)` so engine
  arrival order cannot affect final cache ordering;
- co-locate every ordinary tile in one final bucket file, independently of
  source partitions;
- finalize one bucket in memory only when it is within the configured memory
  envelope;
- recursively repartition an oversized bucket on disk by further deterministic
  tile-key bits, or use an equivalent bounded external grouping/sort;
- stream a pathological single tile into deterministic row groups or physical
  shards without retaining the complete tile in memory;
- treat each bucket as an independent compute-sort-write-release unit and run
  only a configured, bounded number of finalizers concurrently;
- preserve the row-group-per-logical-tile invariant;
- retain bounded concurrency, memory, and temporary disk usage;
- give exactly one finalizer ownership of each deterministic bucket path and
  never recompute or retry a finalizer within the same staging generation;
- write only a unique benchmark/staging artifact, never `COMPLETED` or the final
  visible cache path.

### Focused candidate A: Dask disk shuffle plus Arrow finalizer

Harpy constructs this Dask dataframe only from the validated ordered physical
inventory, using owned readers that preserve source-file offsets and row
positions. It never accepts or inspects an arbitrary caller graph. The candidate
uses the integer `bucket_id`, explicit divisions, and Dask's local disk shuffle
so one output partition corresponds to one writer bucket without a
quantile-discovery scan.

It must keep these stages distinct:

```text
validated physical source
→ annotated: source-partitioned lazy Dask dataframe
→ bucketed: bucket-partitioned lazy Dask dataframe
→ ordered bucket: one computed and sorted output partition
→ bucket-<id>.parquet: final level file inside the disposable spike artifact
```

`annotated` retains the Harpy-owned bounded source partitions and contains at
least `tile_x`, `tile_y`, `x_rel`, `y_rel`, `value_id`, `point_id`, and
`bucket_id`. It is neither shuffled nor persisted as a cache level.

`bucketed` is the lazy result of the disk-shuffle graph. Output Dask partition
`i` contains all rows assigned to integer bucket `i`, possibly in
nondeterministic arrival order. Dask's temporary on-disk fragments are internal
shuffle state; they are not final bucket Parquet files.

The finalizer computes one bucket partition, applies the deterministic
`(tile_y, tile_x, point_id)` sort, groups the resulting contiguous rows by
logical tile, and writes one Parquet row group per capacity-bounded tile shard.
Only this finalizer creates `bucket-<id>.parquet` and its provisional
level-manifest rows. Oversized buckets follow the bounded fallback above before
final writing.

The finalizer should consume numeric data and write through PyArrow. It must not
copy the legacy per-partition Pandas string construction, schema, or direct
side-effect pattern merely because that code exists.

### Focused candidate B: direct PyArrow spill and compaction

This candidate stays on the validated PyArrow path:

```text
bounded source batch
→ numeric exact-level annotation and bucket_id
→ batch-local partitioning of row indices by bucket_id
→ deterministic temporary bucket fragments
→ bounded bucket compaction and grouping
→ final Parquet row groups and provisional level-manifest rows
```

It must define bounded file-handle use, temporary-fragment consolidation,
oversized-bucket handling, deterministic ordering, concurrency, single-owner
bucket output, and cleanup. It performs the same complete logical redistribution
as the Dask candidate; it is not a partition-local shortcut.

C2 compares only these two approaches based on installed dependencies. Do not
introduce DuckDB, Polars, Spark, or another execution dependency unless both
focused candidates fail the locked requirements.

### Locked physical point payload

Both writer candidates must emit the locked exact-level payload:

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

Use the provisional minimal manifest semantics from the parent roadmap. In
particular, do not copy per-row `schema_version` or the derived string `tile_id`
from the legacy metadata dataframe without independent justification. C2 records
the physical row-group facts needed by C6; Gate D owns the final manifest schema.

### Initial local execution and failure contract

Phase 1 is a local, no-task-retry builder. The Dask candidate may use only a
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

Measure at least:

- exact build time and peak RSS;
- shuffle volume, spill bytes, and peak temporary disk usage;
- average and maximum output-bucket rows and bytes;
- largest logical-tile row count;
- finalization throughput and peak RSS at each evaluated bounded concurrency;
- written bytes, bucket/file count, row-group count, and manifest rows;
- rows, files, and row groups touched per logical tile;
- coordinate reconstruction error;
- membership and point-id coverage;
- representative cold and warm tile reads from the spike artifact.

Compare the two candidates with the same 512-unit geometry, payload semantics,
bucket policy, source, and correctness checks. Vary only parameters that
materially affect selection, such as bucket count, engine-specific spill or
shuffle configuration, finalization-memory limit, oversized-bucket fallback,
writer concurrency, and row-group size. Do not implement partition-local Layout
A or add unrelated writer engines for comparison.

### Exit criteria

- 512-unit exact construction is demonstrated;
- peak memory remains bounded independently of source row count;
- temporary disk use and cleanup are measured and bounded by a documented
  construction policy;
- exact membership, identity, and coordinate reconstruction are correct;
- one writer engine is selected from the two focused candidates with recorded
  correctness, complexity, performance, and operational reasons;
- both writer candidates use the locked exact-level payload and reconstruct
  coordinates correctly from its manifest tile key;
- bucket mapping, spill/grouping, single-owner output, file, and dense-tile
  sharding policies are approved for C3;
- spike artifacts are removed after measurements are recorded.

## Slice C3: production exact-level writer

### Goal

Turn the C2 decisions into a deterministic, focused exact-level writer that can
populate a caller-owned staging generation.

### Implement

- the approved bucketed tile-shuffle writer and deterministic file naming;
- the Gate B-selected Dask or direct-PyArrow construction engine, without
  importing or wrapping the legacy writer;
- deterministic per-bucket sorting and finalization;
- bounded oversized-bucket repartitioning or external grouping;
- bounded concurrent finalization governed by the approved combined memory and
  storage-throughput policy;
- bounded selected-column reads using the validated physical inventory;
- vectorized `uint64 point_id` construction from file offsets and row positions;
- normalized-value to `uint32 value_id` mapping from the validated value table;
- numeric tile assignment and tile-local `float32` conversion;
- the locked four-column exact-level point payload;
- deterministic ordering within physical shards;
- dense-tile splitting by `max_rows_per_row_group`;
- one logical tile per row group and deterministic `tile_shard` numbering;
- exact-level counts and provisional level-manifest rows;
- one finalizer owner per deterministic bucket path, with no task retry or
  recomputation inside the staging generation;
- writer accounting sufficient for C6 staged validation.

Unexpected source values, decoding failures, ID overflow, or source facts that
cannot be mapped to the validated contract fail construction. The writer does
not silently extend `values.parquet` or repair the canonical source.

### Focused tests

Use tiny multi-file and multi-row-group fixtures to cover file offsets, batch
boundaries, tile boundaries, dense-tile sharding, dictionary/plain values, exact
membership, deterministic output, and coordinate reconstruction tolerance.
Deliberately spread one logical tile across several source files and verify that
the exact writer co-locates it in one final bucket file or its documented dense-
tile shard sequence. Inject a finalizer failure and verify that the incomplete
staging generation is rejected, no completed cache is replaced, and a subsequent
user-initiated build uses a fresh staging generation. Verify that duplicate
ownership or a pre-existing bucket target fails rather than overwriting it. Test
Harpy's accounting; do not retest Parquet compression internals.

### Exit criteria

- every source point appears exactly once in the exact level;
- every expected `point_id` appears exactly once;
- output rows use only canonical value IDs;
- reconstructed coordinates meet the frozen tolerance;
- rerunning the writer produces the same logical rows, shards, and manifest
  records, apart from generation-owned paths not included in logical identity.

## Slice C4: versioned value-aware sampling contract and spike

### Goal

Freeze and demonstrate the deterministic sampling algorithm before building the
complete sampled pyramid.

### Contract to refine and freeze

The spike starts from the parent-roadmap family:

```text
finer candidates
→ current-level tile and spatial stratum
→ tile target allocated across occupied strata
→ stratum target allocated across values
→ stable priority and deterministic winners
→ deterministic output order
```

It must settle:

- same-geometry bridge stratification, with a fixed micro-grid evaluated as a
  candidate rather than assumed;
- child-tile spatial strata for L1 and later levels;
- the bounded concave value-frequency transform;
- any clipped global-rarity modifier and minimum allocation;
- the versioned stable hash algorithm, seed encoding, and priority payload;
- deterministic integer allocation and remainder distribution;
- deterministic collision and final tie-breaking;
- behavior when strata or values outnumber the target;
- behavior for coincident coordinates, dominant values, singleton-heavy values,
  sparse tiles, and tiles split across physical shards.

The spike operates on bounded candidate tables or exact-cache tiles. It does not
rescan the original canonical source and does not write a completed cache.

### Evaluation

Use small exact fixtures plus value-skewed, spatially skewed, and dense synthetic
tiles. Include representative Xenium tiles. Measure:

- deterministic and nested membership;
- spatial coverage;
- preservation of rare and dominant values;
- hard capacity compliance;
- runtime and peak memory;
- adjacent-level count ratios and transition stability proxies.

### Exit criteria

- the complete sampler has one name and versioned parameter contract;
- every winner is an actual finer-level candidate with unchanged `point_id`;
- the same candidates and parameters always produce the same winners;
- no tile or level target can be exceeded;
- the bridge and spatial-level stratification rules are approved for C5.

## Slice C5: complete nested sampled pyramid

### Goal

Build every planned sampled level from retained finer-level candidates without
repeatedly scanning `points.parquet`.

### Implement

- the 512-at-4,096 sampled finest bridge from exact candidates;
- 1,024-at-8,192, 2,048-at-16,384, and 4,096-at-32,768 spatial levels;
- later edge/capacity-doubling levels when required by the plan;
- terminal global allocation satisfying `overview_point_budget`;
- deterministic parent formation from finer child tiles;
- the C4 value-aware allocation and stable priorities;
- unchanged `point_id` and `value_id` propagation;
- self-contained payloads at every level;
- the C3 physical sharding and manifest-row contract for sampled levels;
- bounded candidate memory and bounded writer concurrency.

### Focused tests

Cover exact-only small sources, sparse tiles, four-child parent formation, the
same-geometry bridge, dense capacity truncation, rare values, terminal global
allocation, nested membership, and deterministic rebuilds. Do not require one
test for every possible number of occupied strata or values.

### Exit criteria

- every generated level is a subset of the next finer level;
- all representatives retain their exact-level identity and value;
- each non-terminal tile respects its capacity;
- the coarsest total respects the global overview budget;
- sampled construction performs no original-source content rescan;
- all planned levels are written and accounted for.

## Slice C6: metadata, values, manifest, and staged-cache validation

### Goal

Turn writer outputs into a complete but unpublished cache generation whose
semantics and physical accounting can be validated independently.

### Implement

- freeze the cache schema version before writing publicly consumable artifacts;
- write `values.parquet` directly from the validated canonical value table;
- write deterministic `manifest.parquet` rows sorted by
  `(level, tile_y, tile_x, tile_shard)`;
- write `metadata.json` with cache identity, source identity, geometry, ordered
  level records, build parameters, value-normalization method, point-id policy,
  sampler version, writer layout, and coordinate dtype contract;
- use cache-root-relative POSIX paths only;
- validate exact Arrow schemas and absence of unexpected metadata where the
  format requires it;
- validate every referenced file and row group;
- reconcile shard → tile → level → cache row counts;
- validate exact membership totals, nested sampled counts, capacities, terminal
  overview budget, level ordering, and path containment;
- reject an absent or premature artifact without creating `COMPLETED`.

The staged validator checks the cache that was written. It does not rescan the
canonical source to recompute bounds, values, or row counts.

### Focused tests

Start from one tiny valid staged generation and derive a small set of corruptions:
missing files, escaped paths, wrong row-group references, count disagreement,
schema mismatch, budget overflow, and non-nested membership where validated at
this phase. Avoid one test per metadata field.

### Exit criteria

- one staged generation is self-consistent without consulting a Dask graph;
- metadata and manifest are sufficient for the future Phase 2 store;
- every physical row group is represented exactly once in the manifest;
- validation returns no partial success and writes no completion marker.

## Slice C7: guarded end-to-end builder and local publication

### Goal

Compose planning, exact writing, sampled writing, cache metadata, staged
validation, source guards, and local publication into the first supported builder.

### Required flow

```text
fresh source signature == validated signature
→ create unique sibling staging generation
→ write exact and sampled levels
→ write values, metadata, and manifest
→ validate complete staging generation
→ fresh source signature == validated signature
→ write COMPLETED
→ install completed generation at transcripts_vis/
```

### Implement

- fail the initial metadata-only source guard before staging is created;
- generate a fresh cache-generation ID;
- create and own a unique sibling staging directory;
- invoke C1, C3, C5, and C6 without rebuilding validation facts;
- fail the final metadata-only source guard after staged validation and before
  completion;
- write `COMPLETED` only after every preceding step succeeds;
- publish the completed local directory with the approved replacement protocol;
- preserve an existing completed cache when any build or guard fails;
- reject and clean incomplete staging according to the frozen recovery policy;
- expose a small public builder accepting `ValidatedPointsSource`;
- expose a backed-SpatialData convenience entry point that delegates visibly
  through resolution, validation, and the primary builder.

Progress, cancellation, overwrite behavior, and the exact returned build result
must be frozen before C7 implementation. They must not leak temporary paths or
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

## Slice C8: Xenium construction benchmark and hardening

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
- reopen the sampler gate if value or spatial preservation is unacceptable;
- record target misses with an explicit accept, optimize, or redesign decision;
- do not weaken correctness, determinism, bounded memory, or publication safety
  merely to reduce build time.

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
- build configuration ownership and defaults needed by the writer spike.

### Gate B: after C2

Approve:

- exact tile size;
- writer-engine selection and the reasons for rejecting the other focused
  candidate;
- stable bucket hash, bucket count, and deterministic file naming;
- engine-specific shuffle/spill configuration and finalization-memory limit;
- bounded oversized-bucket fallback, file-rollover, and dense-tile sharding
  policies;
- the local no-task-retry execution contract and deterministic single-owner
  bucket output;
- bounded read/write strategy and concurrency envelope;
- coordinate reconstruction tolerance;
- exact-writer performance viability.

### Gate C: after C4

Approve:

- sampler name and version;
- bridge and spatial stratification;
- value-aware target allocation;
- hash, seed, tie-breaking, and output ordering;
- deterministic, nested, spatial, value-preservation, and capacity behavior.

### Gate D: after C6

Approve:

- cache schema version;
- payload, values, metadata, and manifest contracts;
- staged-cache validation and accounting;
- compatibility boundary expected by the future Phase 2 reader.

### Gate E: after C8

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
- sampled levels are deterministic, nested, spatially stratified, and value aware;
- every tile and level respects its capacity or global budget;
- `values.parquet`, metadata, manifest, level files, and row groups reconcile;
- every stored path is cache-root-relative and contained by the cache root;
- a cache without `COMPLETED` is never accepted as complete;
- first publication and replacement cannot expose an incomplete generation;
- failed construction preserves any existing completed cache;
- construction time, peak memory, disk size, and fragmentation are documented on
  the Xenium acceptance source;
- Gate E approves the completed cache as the Phase 2 acceptance artifact;
- all focused construction tests pass.

## Immediate next slice

Start with **C0: minimal logical construction contracts**. Before writing code,
refine only the decisions C0 genuinely requires: minimal build configuration,
level-plan records, ownership of later version strings, and which construction
errors must be public. Do not materialize Arrow schema objects before their
consuming slices, and do not settle the sampler or writer-engine details owned
by later slices prematurely.
