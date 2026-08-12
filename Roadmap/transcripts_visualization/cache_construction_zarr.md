# Hybrid Parquet/Zarr multiscale points cache construction

Status: implementation roadmap replacing the draft Parquet point-payload
backend before the first public cache format is frozen

Roadmap date: 2026-08-12

## Authority and relationship to the existing roadmaps

This document refactors the persistent point-payload backend described in
[persistent_cache_construction_5_8_26.md](persistent_cache_construction_5_8_26.md)
from tile-owned Parquet row groups to bucket-local Zarr v3 arrays. It applies to
every persistent level: Exact, Bridge, every spatial level, and the terminal
overview.

The following completed work remains authoritative and must be reused:

- source resolution, validation, normalized values, source signatures, and
  deterministic source-row `point_id` assignment;
- the C1 immutable multiscale build plan and aligned grid geometry;
- the Dask Exact source-to-bucket redistribution;
- the C5a value-neutral spatial sampler;
- the C5c Bridge membership and capacity semantics;
- the C5d coordinate-rebasing semantics;
- the C6 immediate-finer-to-coarser pyramid construction and nesting contract;
- staging, source guards, atomic local publication, and the `COMPLETED` rule.

This document supersedes the following not-yet-published parts of the parent
roadmaps:

- Parquet point files under `levels/`;
- one manifest row per physical Parquet row group;
- `level_file`, `row_group`, and `tile_shard` as published manifest fields;
- `max_rows_per_row_group` as a persistent cache-format parameter;
- complete mixed-value row-group reads as the initial selected-value runtime
  strategy;
- the draft C7a, C7b, C7c, C8, and C9 details where they assume a Parquet point
  payload.

The canonical SpatialData `points.parquet` input remains read-only. Parquet is
still used for the compact tabular cache indexes. Only the derived multiscale
point payload moves to Zarr.

No completed `harpy-multiscale-points-cache-0.1` generation or compatible
runtime reader has been published yet. Consequently, the first completed hybrid
format may retain that schema-version string after the new C7 contract is
frozen. Incomplete development artifacts using the draft Parquet payload are
not caches and are never migrated or accepted.

## Decision summary

Use a hybrid representation:

```text
Parquet
  values.parquet
  manifest.parquet
  tile_value_counts.parquet

Zarr v3
  aligned numeric point arrays
  tile offsets
  sparse tile/value row ranges
```

The physical design borrows the relevant ideas from Xenium Explorer's
visualization-oriented transcript Zarr:

- spatially partitioned multiscale point payloads;
- separate typed arrays for coordinates, identities, and categorical values;
- points arranged contiguously by gene/value inside a spatial tile;
- direct per-gene row ranges instead of scanning an unordered tile;
- independently compressed array chunks.

Harpy does not copy Xenium's biological aggregation semantics. Bridge and
spatial levels remain deterministic, nested, value-neutral samples of exact
source points. They do not become per-gene clusters, do not acquire
`cluster_count`, and do not claim exact abundance at coarse levels. The change
is a physical storage and access refactor.

The external format basis is documented by the
[10x Xenium output-Zarr documentation](https://www.10xgenomics.com/support/software/xenium-onboard-analysis/latest/advanced/xoa-output-zarr)
and the [Zarr v3 array and sharding documentation](https://zarr.readthedocs.io/en/latest/user-guide/arrays/#sharding).

## Goals

1. Use one Zarr point-payload backend for Exact, Bridge, spatial, and overview
   levels.
2. Preserve all source rows, identities, tile-local coordinate precision,
   sampled membership, nesting, capacities, and overview-budget invariants.
3. Preserve the existing independent bucket construction model so Exact bucket
   finalizers do not coordinate writes into one global array.
4. Make every logical tile directly readable as one contiguous point interval.
5. Make every nonempty `(tile, value_id)` directly addressable as one contiguous
   point interval.
6. Let a selected-value reader avoid decoding complete mixed-value tiles when
   the selected ranges touch fewer chunks.
7. Keep gene-to-tile planning compact and efficient through
   `tile_value_counts.parquet`.
8. Keep Zarr details behind a narrow storage boundary so sampling and pyramid
   algorithms continue to operate on Arrow/NumPy payloads.
9. Use Zarr v3 sharding to bound filesystem-object growth without making a
   complete tile or bucket the smallest read unit.
10. Complete the format change before C7 freezes published artifacts and before
    Phase 2 implements the runtime store.

## Non-goals

- Do not change the canonical source from Parquet to Zarr.
- Do not mutate the SpatialData points element.
- Do not convert old or incomplete development cache artifacts in place; rebuild
  derived caches from the validated canonical source.
- Do not maintain two public point-payload backends.
- Do not use one Zarr array or group per logical tile.
- Do not use one Zarr array or chunk per gene.
- Do not use a dense `n_tiles x (n_values + 1)` gene-offset matrix in the first
  format.
- Do not use experimental rectilinear Zarr chunks.
- Do not use `ZipStore` for construction or normal runtime access. Xenium's zip
  is a packaging choice; Harpy requires independent local writers and efficient
  random reads.
- Do not change value-neutral sampling into per-value sampling or clustering.
- Do not implement the complete Phase 2 scheduler, CPU/GPU cache, napari layer,
  or renderer in this construction refactor.
- Do not add density-raster fallback in this roadmap.

## Why the payload is bucket-local rather than level-global

The existing Exact construction assigns each spatial tile deterministically to
one of a bounded number of Dask output buckets. Every finalizer owns one bucket
and can run independently. The full Xenium acceptance source produces 69 Exact
buckets and 7,294 nonempty Exact tiles.

A single flattened Zarr array for the complete level would require a global
prefix-sum pass and coordinated region writes from all finalizers. That would
replace a working construction boundary and complicate failure isolation.

Instead, each existing physical bucket becomes one independent Zarr v3 store:

```text
levels/level_0/bucket-000.zarr/
levels/level_0/bucket-001.zarr/
...
```

Within a bucket, all of its tiles are flattened into aligned arrays. Across
buckets, no shared mutable Zarr object exists. Bridge and spatial construction
already assign their output tiles to deterministic buckets, so they use the same
backend and naming rule.

The bucket hash remains a construction and packaging policy. It does not affect
logical tile coordinates, sampling membership, or runtime rendering semantics.

## Target cache layout

```text
<sdata.zarr>/
  points/
    <points_name>/
      points.parquet                         # canonical source, unchanged
      transcripts_vis/
        metadata.json
        values.parquet
        manifest.parquet
        tile_value_counts.parquet
        levels/
          level_0/
            bucket-000.zarr/
            bucket-001.zarr/
            ...
          level_1/
            bucket-000.zarr/
            ...
          level_n/
            bucket-000.zarr/
        COMPLETED
```

Each `bucket-<id>.zarr` is an independent Zarr v3 `LocalStore` rooted at the
shown cache-relative directory. The store contains one root group and one
`ranges` child group.

## One bucket's Zarr contract

For a bucket containing `K` nonempty logical tiles, `N` points, and `M` nonempty
tile/value combinations:

```text
bucket-000.zarr/
  location             shape=(N, 2)   dtype=float32
  point_id             shape=(N,)     dtype=uint64
  value_id             shape=(N,)     dtype=uint32
  tile_offset          shape=(K + 1,) dtype=uint64

  ranges/
    tile_indptr        shape=(K + 1,) dtype=uint64
    value_id           shape=(M,)     dtype=uint32
    row_start          shape=(M,)     dtype=uint64
    row_count          shape=(M,)     dtype=uint64
```

`location[:, 0]` is `x_rel` and `location[:, 1]` is `y_rel`. Coordinates remain
tile-local `float32`; the global level origin, tile coordinates, and tile size
remain sufficient to reconstruct data coordinates.

`point_id`, point `value_id`, and `location` are aligned by row. Row `i` in all
three arrays describes the same point.

The persisted point order is:

```text
(bucket-local tile order, value_id, point_id)
```

Bucket-local tiles are ordered by `(tile_y, tile_x)`. Points belonging to one
tile are contiguous. Points belonging to one value inside that tile are also
contiguous. Sorting by value happens only after sampling membership is fixed, so
it never changes which representatives are retained.

The bucket root attributes contain only versioned physical facts needed to
reject incompatible stores:

```text
payload_schema_version
level
bucket_id
tile_count
point_count
range_count
point_order
coordinate_encoding
point_chunk_rows
point_shard_rows
range_chunk_rows
codec identifier and codec parameters
```

The exact attribute names and JSON-compatible types are frozen in slice Z2 and
shared by writers, staged validation, and the later runtime store. Semantic
cache metadata remains in `metadata.json`; Zarr attributes do not duplicate the
source signature, build plan, or value vocabulary.

### `tile_offset`

`tile_offset` is a CSR-style pointer array:

```text
tile i rows = tile_offset[i] : tile_offset[i + 1]
```

It supports a complete all-values tile read and reconciles the Zarr payload with
`manifest.parquet`.

Required invariants:

- `tile_offset[0] == 0`;
- `tile_offset[-1] == N`;
- offsets are strictly increasing because empty tiles are not stored;
- `tile_offset[i + 1] - tile_offset[i]` equals the manifest `n_points` for
  bucket-local tile `i`.

### Sparse value ranges

`tile_indptr` partitions the three range arrays by bucket-local tile:

```text
tile i range records = tile_indptr[i] : tile_indptr[i + 1]
```

Each range record stores one nonempty value run:

```text
value_id[j]
row_start[j] : row_start[j] + row_count[j]
```

`row_start` is absolute within the bucket point arrays. Keeping it explicit
slightly duplicates cumulative counts, but makes runtime lookup direct, staged
validation clearer, and future physical-layout changes representable without
changing the query model.

Required invariants for every tile:

- its range `value_id` values are strictly increasing and unique;
- every `row_count` is positive;
- the first range starts at the tile's `tile_offset` start;
- consecutive ranges meet without gaps or overlaps;
- the final range ends at the tile's `tile_offset` stop;
- the point-level `value_id` array is constant and equal to the range value over
  every described row interval;
- the sum of range counts equals the tile's manifest count.

The sparse representation is the Harpy equivalent of Xenium's `gene_offset`.
Xenium can afford dense per-tile gene ranges in its private layout. Harpy's
first format records only nonempty `(tile, value)` combinations because the
acceptance source has 5,122 values and thousands of tiles.

### Chunk and shard contract

The three point arrays use aligned regular chunks along their first dimension:

```text
location chunks = (point_chunk_rows, 2)
point_id chunks = (point_chunk_rows,)
value_id chunks = (point_chunk_rows,)
```

Their Zarr v3 shard shapes are aligned in the same way:

```text
location shards = (point_shard_rows, 2)
point_id shards = (point_shard_rows,)
value_id shards = (point_shard_rows,)
```

`point_shard_rows` must be an integer multiple of `point_chunk_rows`. Chunks are
the independent compression/read unit; shards group many chunks into fewer
filesystem objects.

Initial benchmark candidates are:

```text
point_chunk_rows: 2,048; 4,096; 8,192; 16,384
chunks per shard: 16; 32; 64
codec: Zstd and Blosc/Zstd with byte/bit shuffle where supported
```

The first implementation starts with `point_chunk_rows=4,096` and 32 chunks per
shard as a benchmark baseline, not as a frozen public default. Z3 freezes the
production point chunk, shard, and codec settings from Exact-level evidence
before C7 publishes metadata.

Index arrays use independent, larger regular chunks. Their configuration is not
required to match point chunks. `tile_offset` and `tile_indptr` are normally
small enough for one chunk per bucket. Range arrays share one aligned
`range_chunk_rows` setting. Z2 must prevent scalar reads from producing an
unbounded number of tiny objects.

Fixed chunks may cross tile and value boundaries. A selected range therefore
causes Zarr to decode every chunk it touches, including adjacent unselected
rows. The format promises direct logical row ranges, not one physical chunk per
gene. Read amplification is measured explicitly in Z3 and Z9.

## Parquet artifact contracts

### `values.parquet`

Retain the existing canonical non-nullable schema:

```text
value_id: uint32
value: string
n_points: uint64
```

Rows are in ascending `value_id` order. This file maps UI labels to stable cache
value IDs and records exact source totals.

### `manifest.parquet`

Replace the draft row-group manifest with one row per nonempty logical tile:

```text
level: int16
bucket_id: uint32
bucket_path: string
bucket_tile_index: uint32
tile_x: uint32
tile_y: uint32
n_points: int64
```

Rows are sorted by `(level, tile_y, tile_x)`. Required invariants:

- `(level, tile_x, tile_y)` is unique;
- `(bucket_path, bucket_tile_index)` is unique;
- every path is normalized, cache-root-relative, and directly inside the
  matching `levels/level_<level>` directory;
- all rows sharing `bucket_path` agree on `level` and `bucket_id`;
- for each bucket, `bucket_tile_index` is exactly `0..K-1` in
  `(tile_y, tile_x)` order;
- `n_points` is positive and equals its `tile_offset` interval;
- every Zarr bucket store is referenced by at least one manifest row, and every
  manifest bucket path exists exactly once.

The manifest answers spatial lookup:

```text
(level, tile_x, tile_y)
    -> bucket_path, bucket_tile_index, n_points
```

It no longer describes row groups or physical tile shards. A logical tile is
one contiguous Zarr interval even when its interval spans many chunks or shards.

### `tile_value_counts.parquet`

Retain the existing sparse non-nullable schema:

```text
level: int16
value_id: uint32
tile_x: uint32
tile_y: uint32
n_points: uint64
```

Rows remain sorted by `(level, value_id, tile_y, tile_x)` and contain one row
for every nonempty combination. This is the gene-first planning index:

```text
value_id -> positive spatial tiles and selected point estimates
```

The Zarr range arrays are the tile-first physical index:

```text
bucket-local tile -> value_id -> exact point row interval
```

The two representations deliberately serve opposite query directions. Their
keys and counts must reconcile exactly. The Parquet index does not duplicate
`row_start`; runtime first uses it to choose positive tiles, then uses the
bucket-local sparse ranges to locate payload rows.

## End-to-end lookup flows

### All-values tile read

```text
viewport + chosen level
  -> manifest rows for visible tiles
  -> bucket_path and bucket_tile_index
  -> tile_offset[i:i+2]
  -> location[start:stop]
  -> value_id[start:stop]
  -> point_id[start:stop] when picking requires identity
```

### Explicit value selection

```text
selected labels
  -> values.parquet -> selected value_ids
  -> tile_value_counts.parquet -> positive visible tiles and render estimate
  -> manifest.parquet -> bucket_path and bucket_tile_index
  -> tile_indptr[i:i+2] -> tile's sparse range records
  -> binary/search-merge selected value_ids against sorted range value_ids
  -> location[row_start:row_stop] for matching runs
  -> point_id ranges when picking requires identity
  -> synthesize or concatenate known selected value_ids without reading the
     point-level value_id array solely to rediscover them
```

For multi-value selections, adjacent or overlapping physical chunk requests
should be coalesced before reading. Returned point rows are immutable and may be
reordered for rendering only if `point_id` correspondence is preserved.

The speedup comes only when selected ranges touch fewer chunks than the complete
positive tiles. The indexes eliminate large-array gene searches; Zarr chunking
reduces physical decoding. Neither property alone guarantees faster viewing.

## Reuse and refactor boundaries

### Reuse without semantic changes

The following modules or responsibilities remain substantially unchanged:

```text
models.py
  source descriptions, bounds, validated-source facts

validation.py
  canonical Parquet source inventory and content validation

signature.py
  source signatures and freshness guards

value_normalization.py
  canonical value normalization

build_plan.py
  level geometry, kinds, capacities, overview planning

hashing.py
  deterministic tile-to-bucket and sampling hash primitives

sampling.py
  value-neutral representative selection

writer/exact.py
  source annotation, value mapping, point IDs, tile coordinates, Dask shuffle

writer/bridge.py
  logical Bridge grouping and sample membership

writer/spatial.py
  finer-tile grouping, coordinate rebasing, sampling, level progression
```

### Replace or isolate

The following physical concerns change:

- `_ManifestRow` becomes a logical-tile/bucket location record rather than a
  Parquet row-group descriptor;
- `_POINT_PAYLOAD_SCHEMA` becomes a storage-neutral in-memory payload contract,
  not a Parquet file schema;
- `_read_logical_tile` delegates to a Zarr bucket reader;
- Parquet bucket writers become Zarr bucket writers;
- row-group sharding becomes Zarr chunking and sharding;
- Parquet physical-file validation becomes Zarr array/range validation;
- tests that assert Parquet files and row groups are rewritten to assert logical
  tiles, Zarr schemas, offsets, ranges, chunks, and shards.

### Shared in-memory payload

Keep the current four-column Arrow table as the interchange type used by
construction algorithms:

```text
x_rel: float32
y_rel: float32
value_id: uint32
point_id: uint64
```

The Zarr writer converts `x_rel` and `y_rel` into `location[:, 0:2]`. The Zarr
reader reconstructs the same Arrow table for Bridge and spatial sampling. This
keeps persistent layout decisions out of the sampler and coordinate-rebasing
code.

### Proposed module boundaries

Use narrow level-neutral boundaries rather than spreading Zarr calls across all
writers:

```text
multi_scale_cache_points/payload.py
  storage-neutral Arrow payload schema and logical tile records

multi_scale_cache_points/zarr_payload.py
  Zarr v3 layout constants
  bucket open/read/validate operations shared with the later runtime store

multi_scale_cache_points/writer/zarr_bucket.py
  construction-only bucket creation and append/finalize operations

multi_scale_cache_points/cache_format.py
  metadata models, Parquet schemas, versioned public artifact contracts
```

Names may be adjusted once imports are tested, but the dependency direction is
normative:

```text
sampling and pyramid logic
        |
        v
storage-neutral Arrow/NumPy payload
        |
        v
Zarr bucket read/write boundary
```

The future runtime store may import `zarr_payload.py`; it must not import
construction code from `writer/`.

## Construction flow by level

### Exact

Preserve the implemented flow through Dask redistribution:

```text
validated source Parquet files
  -> annotate tile_x, tile_y, x_rel, y_rel, value_id, point_id, bucket_id
  -> disk shuffle by bucket_id
  -> one materialized bucket partition per finalizer
```

Change the finalizer:

```text
materialized bucket
  -> stable sort by (tile_y, tile_x, value_id, point_id)
  -> group contiguous logical tiles
  -> write aligned Zarr point arrays
  -> write tile_offset
  -> derive sparse value runs and write ranges
  -> emit the existing bucket-local intermediate tile/value counts
  -> return logical manifest records and compact descriptors
```

The Exact finalizer already owns the complete materialized bucket, so it can
calculate `N`, `K`, and `M` before creating fixed-shape arrays. It must write in
bounded chunk- or shard-aligned batches and must not materialize another
complete copy of the `(N, 2)` location array. Construct only a bounded
`(batch_rows, 2)` location buffer for each sequential write.

### Bridge

Preserve logical grouping and capacity semantics:

```text
Exact manifest tile
  -> Zarr complete-tile read through tile_offset
  -> existing value-neutral sampler
  -> reorder selected rows by (value_id, point_id)
  -> append to Bridge Zarr bucket
  -> emit tile/value counts and ranges
```

Bridge construction still needs complete Exact tiles because sampling is
spatial and value-neutral. Gene-selective reads are a runtime optimization, not
a construction shortcut.

The expected Bridge output count for each bucket is known from Exact tile counts
and the Bridge capacity, so point arrays can be created with their final `N`.
Sparse range arrays may be buffered and written in bounded blocks, or created
after the bucket-local intermediate count stream closes. The implementation
must not resample a tile solely to discover `M`.

### Spatial levels and overview

Preserve the implemented immediate-finer flow:

```text
one through four complete finer Zarr tiles
  -> reconstruct storage-neutral Arrow tables
  -> rebase tile-local coordinates into the coarser tile
  -> concatenate candidates
  -> existing value-neutral sampler
  -> reorder selected rows by (value_id, point_id)
  -> append to the current spatial Zarr bucket
```

Every level uses the identical Zarr bucket contract. No special Parquet fallback
exists for sparse, dense, or terminal overview levels.

## Implementation slices

Each slice ends in a focused green test set and leaves the repository in a
coherent state. Do not implement the full runtime viewer during these slices.

### Slice Z0: freeze the refactor boundary and preserve a baseline

#### Goal

Record the current Parquet writer as a benchmark baseline and make this roadmap
authoritative before changing physical artifacts.

#### Work

- record the current focused C1–C6 test commands and results;
- preserve the existing Exact, Bridge, and spatial benchmark reports needed for
  before/after comparison;
- record current full-Xenium build time, peak RSS, bytes, bucket count, tile
  count, representative complete-tile read latency, and sampled counts;
- explicitly mark the parent roadmap's draft C7 Parquet manifest and point-file
  contracts as superseded by this document;
- confirm that no completed public `0.1` hybrid-incompatible cache must be
  migrated;
- confirm the direct runtime dependency policy for Zarr v3. Zarr is currently
  present transitively, but production imports require a direct project
  dependency with a tested minimum version and `<4` upper bound until a future
  compatibility review.

#### Exit criteria

- baseline evidence is retained outside disposable benchmark workspaces;
- the source and logical construction contracts are unchanged;
- the team agrees that bucket-local Zarr v3 is the only production point
  backend targeted by the remaining slices.

### Slice Z1: introduce storage-neutral payload and tile-location contracts

#### Goal

Separate logical tile construction from Parquet row-group descriptors without
changing output membership yet.

#### Work

- move the four-column Arrow payload schema into a level-neutral module;
- define an immutable logical tile location containing at least `level`,
  `bucket_id`, `bucket_path`, `bucket_tile_index`, `tile_x`, `tile_y`, and
  `n_points`;
- change `_BucketWriteResult` and `_LevelWriteResult` to carry logical tile
  records plus intermediate count-file descriptors;
- introduce one private `read_complete_tile(location) -> pa.Table` boundary;
- temporarily wrap the current Parquet reader behind that boundary only as an
  incremental scaffold; do not expose a public backend switch;
- change Bridge and spatial grouping code to reason about one logical tile
  record rather than one or more row-group shards;
- preserve existing point counts, tile order, capacity checks, and sampled
  membership tests.

#### Focused tests

- immutable model validation and path containment;
- unique and contiguous bucket-local tile indexes;
- unchanged build-plan and sampler tests;
- Bridge and spatial logical grouping independent of row groups;
- current Parquet scaffold reads the same Arrow tile payload during this slice.

#### Exit criteria

- no sampling or rebasing function accepts a Parquet row-group descriptor;
- Parquet-specific code is isolated behind the temporary physical adapter;
- all C1–C6 logical invariants remain green.

### Slice Z2: implement and freeze the Zarr bucket primitive

#### Goal

Implement one production-quality bucket-local Zarr writer, reader, and validator
independently of Dask and the multiscale coordinators.

#### Work

- add the direct Zarr v3 dependency and use environment binaries according to
  repository policy;
- define the exact root attributes, arrays, dtypes, shapes, ordering, chunking,
  sharding, codec, and path rules described above;
- implement bucket creation from an ordered sequence of nonempty logical Arrow
  tiles;
- implement bounded point-array writes without a second full-bucket location
  copy;
- implement `tile_offset`, `tile_indptr`, sorted sparse ranges, and consistency
  checks;
- implement complete-tile reads returning the storage-neutral Arrow payload;
- implement selected-value range lookup and reads for a small standalone
  acceptance reader;
- implement structural validation without trusting caller-provided in-memory
  descriptors;
- reject Zarr v2 stores, unsupported codecs, missing arrays, unexpected dtypes,
  incompatible shapes, malformed attributes, non-monotonic pointers, range
  gaps/overlaps, and point/range value disagreement;
- do not consolidate metadata until measurements show a benefit; each bucket has
  few arrays and is already one independent store.

#### Focused tests

Use tiny deterministic fixtures covering:

- several tiles and values in one bucket;
- values absent from some tiles;
- one tile containing one value;
- a value run crossing a point-chunk boundary;
- a tile crossing point chunks and Zarr shards;
- all-values tile roundtrip;
- one- and multi-value selected reads;
- `point_id` correspondence and tile-local coordinate equality;
- every pointer/range corruption class named above;
- cleanup and close behavior after an injected write failure.

#### Microbenchmark

Compare candidate point chunk/shard/codec settings on synthetic tiles matching:

- the average Exact tile;
- the measured 108,598-point dense Exact tile;
- a 4,096-point Bridge tile;
- high- and low-read-amplification value distributions.

Record complete-tile and selected-range cold/warm latency, compressed bytes,
chunks touched, shard objects, and decoded rows.

#### Exit criteria

- the Zarr primitive roundtrips the canonical Arrow payload exactly;
- selected lookup never scans the point-level `value_id` array;
- all structural corruption tests fail closed;
- one provisional configuration is selected for the Exact integration slice.

### Slice Z3: replace the Exact Parquet finalizer

#### Goal

Write the complete Exact level through the Zarr bucket backend while retaining
the proven source annotation and Dask shuffle.

#### Work

- change the bucket sort key from `(tile_y, tile_x, point_id)` to
  `(tile_y, tile_x, value_id, point_id)`;
- replace `ParquetWriter`, row-group sharding, and `_ManifestRow` emission in the
  Exact finalizer with the Zarr bucket writer;
- keep bucket naming, bucket hashing, Dask partition ownership, staging
  ownership, point conservation, and intermediate tile/value counts;
- reconcile every logical tile record with `tile_offset` and every intermediate
  count row with one sparse range;
- ensure each Dask finalizer writes only its own Zarr store and intermediate
  count file;
- keep failure cleanup at staging-generation scope; never expose a partially
  written bucket as a completed cache;
- replace Exact tests that inspect Parquet row groups with Zarr payload, offset,
  range, chunk, and logical-manifest assertions.

#### Focused tests

- every exact source point appears once with the same deterministic `point_id`;
- coordinates retain the existing tolerance;
- points are tile- and value-contiguous;
- tile and per-value totals reconcile;
- bucket output is deterministic across source partition arrival order;
- independent delayed finalizers cannot write the same Zarr store;
- failure in one bucket cannot create a publishable cache.

#### Gate Z3: full-Xenium Exact benchmark

Run the complete 136,578,750-point Exact build and compare it with the preserved
Parquet baseline. Record:

- build time and peak RSS;
- total and per-array compressed bytes;
- buckets, arrays, chunks, shards, and filesystem objects;
- all 7,294 logical tile counts and complete point-ID coverage;
- coordinate reconstruction error;
- complete-tile cold/warm latency;
- selected-value latency for common, median, rare-localized, and
  rare-distributed genes;
- logical selected rows, physically decoded chunk rows, and read amplification.

Freeze `point_chunk_rows`, `point_shard_rows`, range chunking, and codec settings
only after this gate. A configuration is rejected if it obtains selection speed
by causing unacceptable complete-tile latency, build memory, file count, or
storage growth. Initial thresholds are acceptance hypotheses and must be
recorded before the run; they must not be silently changed after results are
known.

#### Exit criteria

- Exact correctness matches the current implementation;
- the production Zarr physical settings are frozen for all levels;
- selected-value evidence justifies proceeding with the hybrid backend;
- any observed regression and accepted tradeoff is documented explicitly.

### Slice Z4: move Bridge construction to Zarr

#### Goal

Consume Exact Zarr tiles and persist Bridge through the same Zarr backend.

#### Work

- remove the temporary Parquet tile reader from Bridge;
- read each complete Exact tile through `tile_offset`;
- apply the existing value-neutral Bridge sampler unchanged;
- reorder sampled output by `(value_id, point_id)` only after membership is
  chosen;
- write Bridge bucket arrays, tile offsets, ranges, and intermediate count files;
- retain the 4,096-point capacity, exact/Bridge equal geometry, bucket ownership,
  deterministic membership, and one-complete-candidate-tile memory policy;
- update Bridge tests and benchmark tooling to inspect Zarr rather than Parquet.

#### Focused tests

- sparse Exact tiles remain complete in Bridge;
- dense tiles contain exactly the planned capacity;
- changing values without changing coordinates and identities does not alter
  sampled `point_id` membership;
- stored Bridge values and counts match the selected representatives;
- complete-tile reads from Zarr reconstruct the same input to the sampler;
- Zarr ordering differs only physically and never changes membership.

#### Acceptance check

Repeat the existing full-Xenium Bridge measurement. Reconcile the previously
observed representative total, nested identity membership, time, peak memory,
bytes, chunks, shards, and complete-tile reads.

#### Exit criteria

- Exact and Bridge use no Parquet point payloads;
- Bridge logical behavior and measured memory remain acceptable;
- both levels use one frozen Zarr schema and physical configuration.

### Slice Z5: move every spatial and overview level to Zarr

#### Goal

Complete a uniform Zarr-backed multiscale pyramid.

#### Work

- replace spatial finer-level Parquet reads with the shared Zarr complete-tile
  reader;
- preserve grouping of one through four immediate-finer tiles;
- preserve coordinate rebasing, value-neutral sampling, capacities, and nested
  point identity;
- reorder every selected coarser tile by `(value_id, point_id)` before storage;
- write every spatial bucket and the terminal overview with the same Zarr
  primitive;
- remove row-group limits, row-group shard grouping, cached `ParquetFile`
  handles, and Parquet point-file validation from the production pyramid path;
- keep `pa.Table` as the assembly and sampling interchange type;
- rewrite spatial persistent tests around logical tiles and Zarr stores.

#### Focused tests

- deterministic two- and multi-level builds;
- every coarser level is a `point_id` subset of its immediate finer level;
- all coordinate-rebasing boundary cases remain green;
- every tile respects its level capacity;
- every level's ranges reconcile with its tile/value counts;
- the coarsest total respects the overview budget;
- no spatial writer reads the canonical source again;
- all levels use identical array schemas and backend version.

#### Gate Z5

Run a complete small end-to-end pyramid and the focused C1–C6 suite. After this
gate, remove the temporary Parquet point adapter and do not support mixed
generations where some levels are Parquet and others are Zarr.

#### Exit criteria

- every planned level is Zarr-backed;
- production code contains no Parquet point-level reader or writer;
- Parquet remains only for the source and compact cache indexes.

### Slice Z6: freeze the hybrid published cache contract

#### Goal

Replace draft C7a with one versioned public hybrid contract before writing final
artifacts.

#### Work

- implement `cache_format.py` with the exact schemas for `values.parquet`, the
  logical-tile `manifest.parquet`, and `tile_value_counts.parquet`;
- freeze bucket Zarr schema and attribute constants in the level-neutral Zarr
  module;
- retain the following identifier only after verifying that no incompatible
  completed format was published:

  ```python
  POINTS_CACHE_SCHEMA_VERSION = "harpy-multiscale-points-cache-0.1"
  ```

- add a required backend identifier such as
  `harpy-zarr-v3-bucket-sparse-value-ranges-v1` to `metadata.json`;
- replace `max_rows_per_row_group` metadata with frozen point/range chunk,
  shard, codec, and point-order fields;
- retain source, geometry, build-plan, sampling, level, artifact-path, and
  generation identity metadata from the parent roadmap;
- require every persisted path to be normalized and contained by the cache
  root;
- state explicitly that the manifest is authoritative for logical tiles and
  bucket locations, while each Zarr store is authoritative for physical point
  and range arrays;
- reject the draft Parquet point payload rather than guessing its format from
  files.

#### Focused tests

- metadata roundtrip and deterministic JSON serialization;
- exact Arrow schemas and nullability;
- supported backend and payload versions;
- malformed or escaping paths;
- invalid chunk/shard/codec combinations;
- logical manifest uniqueness and bucket-local tile continuity;
- unsupported or mixed physical backends fail closed.

#### Exit criteria

- writers, staged validation, and future runtime readers share one contract;
- no C7 implementation decision remains implicit;
- the schema version identifies only the hybrid layout.

### Slice Z7: write final artifacts and validate complete staging

#### Goal

Implement revised C7b/C7c for the hybrid generation.

#### Artifact writing

- write `values.parquet` from the validated canonical value table;
- flatten all level results into the logical-tile manifest sorted by
  `(level, tile_y, tile_x)`;
- consolidate intermediate count files into the gene-first
  `tile_value_counts.parquet` exactly as already planned;
- reject duplicate tile/value keys;
- reconcile every consolidated count with exactly one Zarr sparse range;
- write deterministic `metadata.json` including actual level and bucket totals;
- remove intermediate count files only after final artifacts have been written;
- write no `COMPLETED` marker.

#### Independent staged validation

Open the staging generation without trusting writer objects and validate:

- metadata, backend version, and artifact schemas;
- exact set equality between manifest bucket paths and physical Zarr stores;
- Zarr v3 root/group/array contracts and root attributes;
- array shapes, dtypes, chunks, shards, codecs, and aligned point lengths;
- bucket-local tile index continuity and `tile_offset` reconciliation;
- sparse-range structure, coverage, value order, and point-level value agreement;
- manifest totals by tile and level;
- tile/value index equality with Zarr range keys and counts;
- exact source-wide per-value totals against `values.parquet`;
- bounded coordinate validity and reconstruction tolerance;
- exact point-ID completeness and immediate-coarser subset membership;
- level geometry, capacities, and terminal overview budget;
- absence of unreferenced bucket stores, unexpected point Parquet files,
  intermediate count files, and `COMPLETED`.

Validation must process one logical tile or bounded chunk batch at a time. It
must not load the complete Exact level or all sparse ranges into Python objects
simultaneously.

#### Focused tests

- valid exact-only and multilevel staged generations;
- one representative corruption for every structural layer;
- mismatches between manifest, offsets, ranges, point values, and Parquet counts;
- extra/missing Zarr bucket stores and wrong levels;
- truncated arrays or malformed Zarr metadata;
- validator performs no canonical-source content rescan.

#### Exit criteria

- a valid staging generation is fully self-describing;
- corruption fails closed before publication;
- point payload and both Parquet indexes reconcile independently.

### Slice Z8: compose the guarded end-to-end builder and publication

#### Goal

Implement revised C8 using the uniform Zarr backend.

#### Required flow

```text
ValidatedPointsSource
  -> fresh metadata-only source-signature guard
  -> immutable build plan
  -> unique sibling staging generation
  -> Exact Zarr buckets
  -> Bridge Zarr buckets
  -> all spatial/overview Zarr buckets
  -> Parquet indexes + metadata
  -> independent staged validation
  -> final fresh source-signature guard
  -> write COMPLETED
  -> atomic local publication
```

#### Work

- preserve failure cleanup and replacement semantics from the parent roadmap;
- close all Zarr stores before staged validation and before directory rename;
- ensure no open Dask task, Zarr store, memory map, or file handle refers to the
  staging path during publication;
- preserve an existing completed generation on every construction, validation,
  guard, and publication failure;
- expose no backend selector in the public builder;
- treat any existing draft Parquet development artifact as unsupported and
  disposable; do not attempt in-place conversion;
- never publish mixed or incomplete levels.

#### Focused tests

- first build and replacement;
- failures before staging, during each level, during index writing, during Zarr
  validation, at the final source guard, and during publication;
- existing generation preservation;
- cleanup of incomplete Zarr directories;
- all stores closed before rename;
- `COMPLETED` is the final staged write and mandatory for readers.

#### Exit criteria

- the public cache path is absent or a complete validated hybrid generation;
- no incomplete or mixed physical backend is observable;
- the canonical SpatialData source remains unchanged.

### Slice Z9: selected-read acceptance benchmark and physical tuning

#### Goal

Prove that the new physical representation provides the intended access benefit
without making normal all-values navigation unacceptable.

This slice implements only a small backend-level acceptance reader. The Phase 2
planner, scheduler, LRU caches, napari integration, and renderer remain separate.

#### Reader operations

```text
read_complete_tile(level, tile_x, tile_y)
read_selected_values(level, tile_x, tile_y, value_ids)
```

The selected reader must:

- use `tile_value_counts.parquet` to skip zero-count tiles;
- use manifest bucket locations and Zarr sparse ranges;
- avoid scanning the point-level `value_id` array;
- merge adjacent chunk reads where practical;
- return selected coordinates, IDs when requested, and known value IDs;
- report logical rows, chunks touched, decoded-row estimate, and bytes where the
  store API exposes them.

#### Full-Xenium scenarios

Measure at Exact, Bridge, representative spatial levels, and overview:

- one complete dense tile and representative average tiles;
- all-values viewports at several zoom levels;
- one common gene;
- one median-abundance gene;
- one rare localized gene;
- one rare but spatially distributed gene;
- several selected genes with adjacent and nonadjacent ranges;
- repeated selection changes with cold and warm OS/Zarr caches;
- panning with overlapping bucket and chunk reuse.

For every selected case calculate:

```text
logical selected rows
complete positive-tile rows
chunks touched
estimated decoded chunk rows
tile read amplification
chunk read amplification
latency
```

Compare against the preserved Parquet baseline where reproducible. Report cases
where small tiles make the difference negligible and cases where broadly
distributed sparse genes produce material improvement.

#### Acceptance

Correctness is mandatory. Performance thresholds are frozen before the run and
include:

- complete build time and peak RSS;
- total cache bytes and filesystem-object count;
- complete-tile cold/warm latency;
- selected-value cold/warm latency at high read amplification;
- all-values throughput;
- no unbounded growth in chunks touched per visible tile.

If tuning changes chunk, shard, or codec parameters after Z6, the cache format
metadata and schema-version decision must be revisited before any public release.
Do not silently change a published physical contract.

#### Exit criteria

- the selected reader demonstrates direct range lookup and chunk-selective IO;
- all-values performance is acceptable for the planned viewer;
- final build and runtime measurements are recorded;
- the hybrid generation is accepted as the Phase 2 input artifact.

### Slice Z10: remove transitional code and synchronize documentation

#### Goal

Leave one supported implementation and one coherent set of roadmaps.

#### Work

- delete the temporary Parquet point adapter and obsolete row-group payload
  models after benchmark evidence is retained;
- remove `max_rows_per_row_group` from public cache construction configuration
  and metadata while retaining source-Parquet row-group validation concepts;
- remove writer tests whose only contract was Parquet packaging and replace
  them with logical/Zarr coverage;
- retain a clearly named opt-in historical benchmark only if it remains useful;
- update `multi_tile_cache_29_7_26.md` and
  `persistent_cache_construction_5_8_26.md` to point to this roadmap for the
  physical payload and revised C7–C9 work;
- update `compare_to_xenium.md` with measured rather than proposed Harpy
  behavior;
- document that cache rebuild, not artifact conversion, is the supported
  migration path.

#### Exit criteria

- production code has one point backend: bucket-local Zarr v3;
- Parquet is used only for canonical input and compact cache indexes;
- documentation contains no active claim that point payloads are Parquet row
  groups;
- focused construction and backend-reader tests are green.

## Test strategy

Run only focused tests during each slice according to repository policy. The
complete focused construction set is appropriate at Z3, Z5, Z7, Z8, and Z10
because those are broad backend boundaries. The full repository test suite is
not required unless separately approved.

Test responsibilities are layered:

```text
pure logic
  build plans, hashes, sampling, coordinate rebasing

Zarr primitive
  arrays, pointers, ranges, chunks, shards, corruption

level writers
  Exact, Bridge, spatial membership and counts

cache artifacts
  metadata, manifest, values, tile/value counts

staged validation
  independent cross-artifact reconciliation

builder
  guards, cleanup, completion, publication

acceptance reader
  complete and selected physical reads
```

Do not assert compressed bytes for tiny fixtures: codec output may vary across
compatible library versions. Assert declared codec configuration, logical
content, and structural invariants. Performance belongs to opt-in benchmark
scripts, not normal CI.

## Memory and concurrency rules

- Exact retains the accepted one-materialized-bucket finalization policy until
  measurements require a different bounded sort.
- Each Exact finalizer owns one Zarr bucket store; no concurrent writers target
  the same store or shard.
- Bridge and spatial construction retain one complete candidate tile at a time.
- Zarr point arrays use known final shapes. Do not grow them once per point or
  once per tile.
- the bucket writer maintains bounded sequential point buffers across logical
  tile boundaries and preferentially flushes complete Zarr shards. Do not issue
  one partial-shard write per small tile, because repeatedly updating the same
  shard can erase the object-count benefit with rewrite overhead;
- Sparse range records are written in bounded blocks. Unknown final `M` must not
  cause one Python object per range to remain live for the complete level.
- `location` is written from existing coordinate columns in bounded batches;
  avoid a second full-bucket `(N, 2)` NumPy allocation.
- Close stores promptly and before another level consumes them if the backend
  requires it for consistent metadata visibility.
- Runtime reads may be concurrent, but construction writes are coordinated only
  at independent bucket-store granularity.

## Failure and publication rules

- Every cache remains derived and disposable.
- All writes occur under a unique sibling staging generation.
- A bucket failure invalidates the complete staging generation.
- The builder does not attempt to repair or resume a partially written Zarr
  store in the first implementation.
- Staged validation opens stores afresh after writers close them.
- `COMPLETED` is absent during all construction and validation work.
- The final source guard precedes `COMPLETED` and publication.
- Readers reject missing completion, unsupported backend versions, missing
  bucket stores, mixed payload backends, and inconsistent Zarr/Parquet indexes.

## Risks and mitigations

### Small-read overhead

Many tiny selected ranges can touch many chunks. Mitigate with measured chunk
sizes, range coalescing, Zarr v3 sharding, and the existing CPU-cache plan. Do
not create a chunk per gene.

### All-values regression

Gene ordering is still contiguous within tiles, so complete tile slices remain
sequential. Benchmark complete-tile and viewport throughput before freezing
physical parameters.

### Filesystem-object growth

Use bucket-local stores and v3 shards. Record actual files, directories, chunks,
and shards at full scale. Do not infer object count only from logical chunk
count.

### Duplicate indexes

`tile_value_counts.parquet` and Zarr sparse ranges intentionally duplicate keys
and counts for different query directions. Independent staged validation makes
their equality a cache invariant rather than trusting either writer.

### Zarr API/version instability

Use a direct bounded dependency, a narrow internal adapter, explicit Zarr v3
validation, and no experimental rectilinear chunks. Keep Zarr calls out of
sampling and planning modules.

### Nested `.zarr` directories inside SpatialData

The payload stores live below Harpy-owned `transcripts_vis/` and are not
SpatialData elements. Validate path ownership and confirm SpatialData read/write
operations ignore these derived directories. Publication treats the complete
cache directory as opaque derived content.

### Range selectivity smaller than expected

Small Harpy tiles may make offsets irrelevant for many views. Z3/Z9 measure
read amplification by gene and viewport. The hybrid format is accepted only
with evidence and an explicit record of where it helps and where it does not.

## Expected reuse

The refactor should preserve approximately:

| Area | Expected reuse |
|---|---:|
| Source resolution and validation | 95–100% |
| Signatures and value normalization | 100% |
| Logical cache planning | 90–95% |
| Tile coordinates and `point_id` generation | 100% |
| Dask Exact redistribution | about 90% |
| Deterministic sampling | 100% |
| Spatial rebasing and membership | 85–95% |
| Intermediate count generation | 80–100% |
| Physical point writer/reader | 10–30% |
| Physical manifest models | 20–40% |
| Physical validation tests | 20–40% |

The expected direct reuse across the current core is about 70–75%; within the
writer package it is closer to 45–60%. The replaced code is primarily Parquet
packaging and row-group reconstruction, not the logical multiscale algorithms.

## Estimated implementation effort

For one engineer familiar with the current code:

| Work | Estimate |
|---|---:|
| Z0–Z2 contracts, abstraction, Zarr primitive | 4–7 working days |
| Z3 Exact integration and full-data gate | 3–5 days |
| Z4–Z5 Bridge and spatial integration | 3–5 days |
| Z6–Z8 artifacts, validation, builder, publication | 4–7 days |
| Z9–Z10 benchmark, tuning, cleanup, documentation | 3–5 days |

Some work overlaps with the unimplemented C7–C9 tasks that were already
required. A production-quality hybrid Phase 1 is expected to take roughly two
to three working weeks, with benchmark tuning as the largest uncertainty.

## Definition of done

The refactor is complete when:

- every planned level uses the same versioned bucket-local Zarr v3 payload;
- no published cache point payload is Parquet;
- source validation, identities, coordinates, sampling, capacities, nesting,
  and overview-budget invariants remain satisfied;
- every manifest tile resolves to one contiguous Zarr interval;
- every nonempty tile/value key resolves to one contiguous sparse range;
- `values.parquet`, `manifest.parquet`, `tile_value_counts.parquet`, and all Zarr
  stores reconcile under independent staged validation;
- selected-value reads avoid point-level value scans and decode only intersecting
  chunks;
- all-values reads remain acceptable under measured viewer scenarios;
- construction is bounded, failure-safe, and atomically published;
- the full Xenium acceptance build and selected-read benchmark are recorded;
- the hybrid cache is approved as the sole Phase 2 runtime-store input.

## Immediate next slice

Execute Z0, then implement Z1 before changing any writer output. Do not implement
the draft Parquet-based C7a contract first: it would freeze fields that this
backend deliberately removes. The first code-producing Zarr slice is Z2, and
the first full-data decision gate is the Exact-only Z3 benchmark.
