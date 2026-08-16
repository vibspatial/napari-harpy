# Independent hybrid Parquet/Zarr multiscale points cache

Status: implementation roadmap for an isolated production-quality Zarr-backed cache

Roadmap date: 2026-08-13

## High-level decision

Implement the Zarr-backed cache as a new package:

```text
src/napari_harpy/core/
  multi_scale_cache_points/          # existing tiled-Parquet implementation
  multi_scale_cache_points_zarr/     # new isolated Zarr implementation
```

The new package borrows proven ideas and logical invariants from the existing
implementation, but it is not a refactor of that implementation. It starts with
fresh models, planning, writers, readers, validation, artifact construction, and
orchestration.

The existing `multi_scale_cache_points` package remains unchanged while the new
architecture is developed and evaluated. It is a reference and rollback point,
not a runtime fallback and not a backend selected by the new package.

The only intentional code sharing is the canonical read-only source boundary:

```text
ParquetPointsSource
ValidatedPointsSource
validate_parquet_points_source(...)
source inventory and content validation
canonical normalized value table
source signature facts
```

Everything downstream of a `ValidatedPointsSource` is owned by
`multi_scale_cache_points_zarr`, even where the implementation initially looks
similar to the Parquet-backed code. Duplication is acceptable and preferred over
coupling the new architecture to existing writer internals.

In particular, the new package must not import from:

```text
multi_scale_cache_points.writer.exact
multi_scale_cache_points.writer.bridge
multi_scale_cache_points.writer.spatial
multi_scale_cache_points.writer.models
multi_scale_cache_points.writer.support
```

There is no transitional Parquet point cache, no `_ParquetCompleteTileReader`,
no conversion of `_ManifestRow`, and no generation containing a mixture of
Parquet-backed and Zarr-backed levels.

## Authority and relationship to earlier roadmaps

This document defines the production-quality Zarr-backed cache candidate. It
replaces earlier proposals to incrementally migrate the existing Exact, Bridge,
and spatial Parquet writers.

The following source and semantic requirements remain authoritative:

- the canonical SpatialData points element remains read-only Parquet;
- source validation and normalized values define the accepted input;
- `point_id` is deterministic and uniquely identifies one canonical source row;
- Exact preserves every accepted source row exactly once;
- Bridge and spatial levels use deterministic value-neutral sampling;
- a coarser level is a `point_id` subset of its immediate finer level;
- tile-local coordinate rebasing preserves the existing spatial meaning;
- level capacities and the terminal overview budget are enforced;
- all construction happens under staging and is published only after independent
  validation;
- `COMPLETED` is written only after the final source guard and successful staged
  validation.

The new implementation may reproduce algorithms to satisfy those requirements.
It does not need to call the corresponding implementation in the existing
package.

## Product-quality and compatibility policy

This is an isolated implementation path for a professional product, evaluated
at full Xenium scale before integration. Isolation does not lower the quality
bar: every implemented slice must be correct, deterministic, memory-bounded,
failure-safe, independently validated, maintainable, and suitable for continued
production ownership. Build time, peak RSS, cache size, object count, and
representative read behavior are recorded and reviewed together; there are no
pre-registered numerical pass/fail thresholds.

The architecture-adoption decision remains binary:

```text
Zarr cache is correct, memory-bounded, and practically satisfactory
    -> adopt the new implementation in a later integration decision

Zarr cache is unsatisfactory
    -> do not adopt it; retain the existing package
```

Do not add:

- an artifact migration path;
- a reader that guesses the backend from files;
- a public Parquet/Zarr backend selector;
- an automatic fallback from Zarr to the existing implementation;
- compatibility tests between the two derived cache formats.

The direct Zarr dependency and version bounds are not frozen before the bucket
primitive and Xenium evidence establish what is required.

## Goals

1. Write Exact, Bridge, every spatial level, and overview directly to the same
   bucket-local Zarr v3 representation.
2. Preserve source identities, coordinate precision, sampling membership,
   nesting, capacities, and overview-budget invariants.
3. Let independent Exact finalizers own independent Zarr stores; no level-global
   mutable array is introduced.
4. Make a complete logical tile directly addressable as one contiguous row
   interval.
5. Make every nonempty `(tile, value_id)` directly addressable as one contiguous
   row interval.
6. Avoid scanning mixed-value point payloads for selected-value reads.
7. Keep construction memory bounded at a materialized Exact bucket or a small
   number of immediate-finer tiles, never a complete level.
8. Independently validate the final Zarr stores and compact Parquet indexes
   before publication.
9. Keep the new implementation understandable and maintainable in isolation,
   even when that duplicates logic from `multi_scale_cache_points`.

## Non-goals

- Do not change the canonical source from Parquet to Zarr.
- Do not mutate the SpatialData points element.
- Do not refactor the existing `multi_scale_cache_points` package as part of
  this roadmap.
- Do not import existing writer or writer-support internals.
- Do not create one Zarr store, group, array, or chunk per logical tile or gene.
- Do not use a dense `n_tiles x (n_values + 1)` value-offset matrix initially.
- Do not use `ZipStore`, experimental rectilinear chunks, or a level-global
  mutable Zarr array.
- Do not change value-neutral sampling into per-value sampling or clustering.
- Do not implement the full Phase 2 napari scheduler, renderer, or CPU/GPU LRU
  cache here.
- Do not promise backward compatibility for derived development artifacts.

## Package boundary

The target package layout is:

```text
multi_scale_cache_points_zarr/
  __init__.py
  models.py                 # Zarr-cache-specific immutable models
  build_plan.py             # fresh level geometry and capacities
  hashing.py                # fresh deterministic tile/bucket helpers
  sampling.py               # fresh value-neutral sampler implementation
  payload.py                # storage-neutral in-memory point payload
  cache_format.py           # metadata and final Parquet schemas

  storage/
    __init__.py
    _schema.py              # private bucket constants and Zarr codec/layout map
    models.py               # bucket plans/results and physical settings
    bucket_writer.py        # create and finalize one Zarr bucket
    bucket_reader.py        # complete and selected tile reads
    bucket_validation.py    # independent structural bucket validation

  writer/
    __init__.py
    exact.py                # canonical source -> Exact Zarr buckets
    bridge.py               # Exact Zarr -> Bridge Zarr
    spatial.py              # finer Zarr -> coarser Zarr
    artifacts.py            # values/manifest/tile-value-count artifacts
    staging_validation.py   # complete cross-artifact validation
    build.py                # guards, staging, composition, publication
```

Corresponding tests live under:

```text
tests/multi_scale_cache_points_zarr/
```

The exact module split may be adjusted to avoid tiny files, but these dependency
rules are normative:

```text
canonical source validation
          |
          v
multi_scale_cache_points_zarr models and build plan
          |
          +------------------------+
          |                        |
          v                        v
fresh logical construction     Zarr storage primitive
          |                        |
          +------------+-----------+
                       v
              artifacts and validation
                       |
                       v
                 guarded publication
```

Logical construction may depend on the new storage API. Storage must not depend
on Exact, Bridge, or spatial writers. The later runtime store may import the
bucket reader and cache-format contracts; it must not import construction code.

## What is reused and what is duplicated

### Reused directly

The new builder accepts the existing validated source object rather than
reimplementing canonical source discovery and scanning:

```text
multi_scale_cache_points.models
  ParquetPointsSource
  ValidatedPointsSource
  source-file and bounds facts

multi_scale_cache_points.validation
  validate_parquet_points_source

multi_scale_cache_points.value_normalization
  VALUE_NORMALIZATION_METHOD
  _normalized_row_values

multi_scale_cache_points.signature
  source-signature facts and fresh metadata-only checks, where suitable
```

Value normalization is canonical source semantics rather than derived-cache
writer behavior. Reusing its version identifier and row-aligned normalizer keeps
validation and Exact annotation on the same definition. If importing another
private source helper would pull in cache-writer assumptions, the new package
duplicates the small helper instead. The reuse boundary must remain obvious from
imports.

### Implemented fresh in the Zarr package

- build-plan models and validation;
- level geometry and capacities;
- tile-to-bucket hashing policy;
- source annotation and deterministic `point_id` assignment;
- value-neutral sampling;
- coordinate rebasing;
- Exact Dask graph and finalizers;
- Bridge and spatial level construction;
- all writer-result and tile-descriptor models;
- all Zarr read, write, and validation code;
- final artifact construction;
- staged validation, builder composition, and publication;
- all tests of derived-cache behavior.

The existing implementation may be read as a specification aid. Copying and
adapting small pieces is acceptable, but the resulting code and tests belong to
the new package and must not create cross-writer dependencies.

## Target cache layout

```text
<sdata.zarr>/
  points/
    <points_name>/
      points.parquet                         # canonical source, unchanged
      transcripts_vis_zarr/                  # Harpy-owned derived cache
        metadata.json
        values.parquet
        manifest.parquet
        tile_value_counts.parquet
        levels/
          level_0/
            bucket-000.zarr/
            bucket-001.zarr/
          level_1/
            bucket-000.zarr/
          level_n/
            bucket-000.zarr/
        COMPLETED
```

The final cache directory name is intentionally deferred until the public
integration decision. During isolated development it must not collide with the
existing derived-cache path. Every `bucket-<id>.zarr` is an independent Zarr v3
`LocalStore`.

Parquet is used only for:

- the canonical source;
- `values.parquet`;
- `manifest.parquet`;
- `tile_value_counts.parquet`.

There is no point-payload Parquet file and no construction-only point-payload
Parquet stage.

## Storage-neutral point payload

Fresh logical construction uses one aligned in-memory payload:

```text
x_rel: float32
y_rel: float32
value_id: uint32
point_id: uint64
```

The internal representation is the frozen NumPy `_PointPayload` specified in
Z1. The contract requires:

- equal row counts;
- exact dtypes;
- no non-finite coordinates;
- tile-local coordinates;
- aligned rows across all fields;
- at least one row for a stored tile.

Sampling and rebasing operate on this payload. The Zarr writer converts the two
coordinate arrays to `location[:, 0:2]`. Arrow remains appropriate for the
canonical value table and final compact Parquet indexes, but is not the point
interchange boundary.

## Logical tile descriptor

Every nonempty stored tile has one descriptor:

```text
level: int
bucket_id: int
bucket_tile_index: int
tile_x: int
tile_y: int
n_points: int
```

`bucket_path` is not stored independently on the descriptor. It is the
canonical cache-relative property
`levels/level_<level>/bucket-<bucket_id, minimum three digits>.zarr`, derived
only from `level` and `bucket_id`.

`bucket_tile_index` is a zero-based tile ordinal inside a bucket. It is not a
point offset, chunk number, shard number, or Parquet row group. Physical point
offsets remain inside the Zarr store.

Descriptors are construction results and later become manifest rows. They must
be unique by `(level, tile_x, tile_y)` and by
`(bucket_path, bucket_tile_index)`. Within one bucket they are ordered by
`(tile_y, tile_x)` and indexed exactly `0..K-1`.

## One bucket's Zarr contract

For `K` nonempty tiles, `N` points, and `M` nonempty tile/value combinations:

```text
bucket-000.zarr/
  location             shape=(N, 2)   dtype=float32
  point_id             shape=(N,)     dtype=uint64
  value_id             shape=(N,)     dtype=uint32
  tile_x               shape=(K,)     dtype=uint32
  tile_y               shape=(K,)     dtype=uint32
  tile_offset          shape=(K + 1,) dtype=uint64

  ranges/
    tile_indptr        shape=(K + 1,) dtype=uint64
    value_id            shape=(M,)     dtype=uint32
    row_start           shape=(M,)     dtype=uint64
    row_count           shape=(M,)     dtype=uint64
```

Point rows are ordered by:

```text
(tile_y, tile_x, value_id, point_id)
```

Sorting by `value_id` happens only after sampling membership is fixed. It changes
physical order but never changes representative membership.

The root attributes contain the compact application-level bucket contract:

```text
payload_schema_version = 1
level                  = <JSON integer>
bucket_id              = <JSON integer>
tile_count             = <JSON integer>
point_count            = <JSON integer>
range_count            = <JSON integer>
point_order            = ["tile_y", "tile_x", "value_id", "point_id"]
coordinate_encoding    = "tile-relative-xy-float32-v1"
codec_id               = "zstd-v1"
```

These exact keys and value encodings are part of payload schema version 1.
NumPy scalar objects are not written as attributes. Semantic cache metadata
remains in `metadata.json`.

Chunk and shard shapes are deliberately not duplicated in the root attributes.
Each array's Zarr v3 metadata is the authoritative source for its physical
layout. Readers obtain the point read granularity from the point-array metadata;
independent validation derives the point and range row layouts from their
canonical `value_id` arrays and requires every parallel array to match. The
versioned `codec_id` remains an application-level compatibility profile whose
declared pipeline is checked against the physical array codec metadata.

### Tile identity and offsets

Bucket-local tile `i` is:

```text
(tile_x[i], tile_y[i])
```

Its complete point interval is:

```text
tile_offset[i] : tile_offset[i + 1]
```

Required invariants:

- `tile_x.shape == tile_y.shape == (K,)`;
- coordinate pairs are unique and ordered by `(tile_y, tile_x)`;
- `tile_offset.shape == (K + 1,)`;
- `tile_offset[0] == 0` and `tile_offset[-1] == N`;
- offsets are strictly increasing because empty tiles are omitted;
- the interval length equals the descriptor and manifest `n_points`;
- the manifest coordinates for bucket-local index `i` equal `tile_x[i]` and
  `tile_y[i]`.

The coordinate arrays deliberately duplicate the manifest coordinates. This
small duplication lets independent validation detect a permuted or mislabeled
manifest.

### Sparse value ranges and `tile_indptr`

`tile_indptr` partitions the range records by bucket-local tile:

```text
tile i range records = tile_indptr[i] : tile_indptr[i + 1]
```

Each range record identifies one nonempty value run in the point arrays:

```text
value_id[j]
row_start[j] : row_start[j] + row_count[j]
```

There are two distinct pointer layers:

```text
tile_offset
    bucket-local tile -> point rows

tile_indptr
    bucket-local tile -> sparse value-range records
```

Worked example:

```text
tile_x      = [4,  5,  4]
tile_y      = [2,  2,  3]
tile_offset = [0, 13, 25, 31]

tile 0 values: 0 -> [0:10],  2 -> [10:13]
tile 1 values: 1 -> [13:21], 2 -> [21:25]
tile 2 values: 0 -> [25:31]

ranges/value_id  = [0,  2,  1,  2,  0]
ranges/row_start = [0, 10, 13, 21, 25]
ranges/row_count = [10, 3,  8,  4,  6]
tile_indptr       = [0, 2, 4, 5]
```

For bucket-local tile 1, `tile_indptr[1:3] == [2, 4]`, so its range records are
indices `2:4`. Value ID 1 resolves to point rows `13:21` without scanning the
point-level `value_id` array.

Required invariants:

- `tile_indptr.shape == (K + 1,)`;
- `tile_indptr[0] == 0` and `tile_indptr[-1] == M`;
- pointers are strictly increasing for stored nonempty tiles;
- all three range arrays have shape `(M,)`;
- value IDs are strictly increasing and unique within each tile;
- every `row_count` is positive;
- ranges cover the corresponding tile interval without gaps or overlaps;
- the point-level `value_id` array is constant and equal to the range value over
  each range;
- range counts reconcile with `tile_value_counts.parquet`.

This sparse representation is the Harpy equivalent of Xenium's gene-offset
idea without a dense tile-by-gene matrix.

### Chunking and Zarr-internal sharding

Logical tiles are not shards, and construction code does not create or name
shard files. Sharding is an internal Zarr v3 physical packing choice: chunks
remain the independent compression/read units, while each shard stores several
chunks in one filesystem object and is the efficient write unit.

This two-scale layout is deliberate: shards keep the number of physical storage
objects small, inner chunks preserve fine-grained selected reads, and
shard-sized staging buffers provide aligned, memory-bounded writes.

Point arrays use aligned chunks and shards along the point dimension:

```text
location chunks = (point_chunk_rows, 2)
point_id chunks = (point_chunk_rows,)
value_id chunks = (point_chunk_rows,)

location shards = (point_shard_rows, 2)
point_id shards = (point_shard_rows,)
value_id shards = (point_shard_rows,)
```

Sparse range arrays use their own aligned chunks and shards:

```text
ranges/value_id chunks  = (range_chunk_rows,)
ranges/row_start chunks = (range_chunk_rows,)
ranges/row_count chunks = (range_chunk_rows,)

ranges/value_id shards  = (range_shard_rows,)
ranges/row_start shards = (range_shard_rows,)
ranges/row_count shards = (range_shard_rows,)
```

These sizes live canonically in each array's Zarr metadata. The names above
describe `_ZarrWriteSettings` construction inputs and the cross-array alignment
contract; they are not repeated as root attributes.

`point_shard_rows` is an integer multiple of `point_chunk_rows`, and
`range_shard_rows` is an integer multiple of `range_chunk_rows`. Fixed chunk and
shard boundaries may cross tile and value boundaries. The small `tile_x`,
`tile_y`, `tile_offset`, and `tile_indptr` arrays each use one unsharded chunk
per bucket.

Current production-candidate values are:

```text
point_chunk_rows = 4,096
point_shard_rows = 131,072       # 32 point chunks
range_chunk_rows = 8,192
range_shard_rows = 131,072       # 16 range chunks
```

They are not frozen until Slice Z3 records Exact-level evidence. This is
benchmark-guided configuration selection, not permission for a temporary or
lower-quality implementation. At these values, the three aligned point shard
buffers occupy approximately 2.5 MiB in aggregate, and the three aligned range
shard buffers occupy approximately another 2.5 MiB.

Writers buffer aligned point fields across logical tile boundaries and flush
complete `point_shard_rows` blocks where possible. The final partial shard is
flushed once during finalization. A `write_tile` call therefore advances the
logical tile contract but does not imply an immediate tile-sized Zarr write.
This keeps memory bounded while avoiding repeated updates of the same partially
filled shard when a bucket contains many small tiles.

Range records are buffered and flushed in the same way using
`range_shard_rows`. A selected read still resolves and decodes only the inner
chunks touched by its sparse row ranges; it does not decode the complete
containing shard. The reader deduplicates touched chunk IDs so two selected
ranges in the same inner chunk do not request that chunk twice. Intervals whose
touched chunks overlap or are consecutive share one read using the minimal row
envelope from the first exact selected row to the last; read bounds are not
expanded to the outer chunk edges.

All arrays are created with empty-chunk storage enabled. This is necessary
because a completely zero-valued chunk can be legitimate cache data. Readers
and validators use strict missing-chunk reads so an absent chunk or shard raises
instead of being silently reconstructed from the array fill value.

Every array uses an explicit zero fill value of its declared dtype. The store
uses the standard Zarr v3 default chunk-key encoding with `/` as the separator.
Because empty chunks are written, a valid all-zero chunk remains physically
present; because reads are strict, the zero fill value cannot conceal a missing
chunk or shard.

Z2 owns a small internal registry from a versioned `codec_id` to an exact public
Zarr v3 codec pipeline. The initial mapping is:

```text
zstd-v1
  inner chunk data:
    little-endian bytes
    -> Zstd level 3 with checksum enabled

  shard index:
    little-endian bytes
    -> CRC32C
    -> index stored at the end of the shard
```

Unknown codec IDs are rejected. The roadmap does not add or freeze a direct
Zarr constraint in `pyproject.toml` at this stage.

## Compact Parquet indexes

### `values.parquet`

```text
value_id: uint32
value: string
n_points: uint64
```

Rows are ordered by `value_id`. Counts are exact canonical-source totals.

### `manifest.parquet`

One row describes one nonempty logical tile:

```text
level: int16
bucket_id: uint32
bucket_path: string
bucket_tile_index: uint32
tile_x: uint32
tile_y: uint32
n_points: int64
```

Rows are sorted by `(level, tile_y, tile_x)`. The manifest maps a logical tile
to its bucket and bucket-local ordinal; it never exposes Zarr offsets, chunks,
or shards. `bucket_path` is materialized from the descriptor's canonical
property for direct lookup; it is never accepted as independent construction
state. Validation recomputes it from `level` and `bucket_id` and requires exact
equality.

### `tile_value_counts.parquet`

```text
level: int16
value_id: uint32
tile_x: uint32
tile_y: uint32
n_points: uint64
```

Rows are sorted by `(level, value_id, tile_y, tile_x)`. They are generated in a
bounded pass over finalized Zarr range arrays. No construction-only point
Parquet or per-bucket Parquet count fragment is required.

The two value indexes serve different query directions:

```text
tile_value_counts.parquet
    selected value -> positive tiles

bucket ranges
    bucket-local tile -> selected value -> point rows
```

Independent validation requires their keys and counts to agree exactly.

## End-to-end construction flow

```text
ParquetPointsSource
  -> shared canonical source validation
  -> ValidatedPointsSource
  -> fresh Zarr-cache build plan
  -> unique staging generation
  -> Exact Zarr buckets
  -> Bridge Zarr buckets read from Exact Zarr
  -> spatial/overview Zarr buckets read from immediate-finer Zarr
  -> values.parquet
  -> manifest.parquet
  -> tile_value_counts.parquet derived from Zarr ranges
  -> independent staged validation
  -> final fresh source-signature guard
  -> COMPLETED
  -> atomic publication
```

At no point does the derived point payload pass through Parquet.

## Construction flow by level

### Exact

Implement a fresh Dask construction graph:

```text
validated source row groups
  -> one input partition per validated physical row group
  -> annotate tile_x, tile_y, x_rel, y_rel, value_id, point_id, bucket_id
  -> disk shuffle by bucket_id
  -> one complete materialized partition per nonempty bucket
  -> stable sort/group by (tile_y, tile_x)
  -> bucket writer orders each tile by (value_id, point_id)
  -> write one independent Zarr store
```

Each finalizer owns exactly one store. It derives ordered tile identities and
counts from the sorted bucket and uses them to construct `_BucketPlan`.
`_BucketWriter` then owns tile-internal ordering, offsets, sparse ranges, and
physical writing. It writes `location` in bounded batches and does not allocate
a second full-bucket `(N, 2)` array.

### Bridge

```text
Exact tile descriptor
  -> read complete Exact tile from Zarr through tile_offset
  -> fresh value-neutral sampler
  -> take the selected aligned payload rows
  -> bucket writer orders by (value_id, point_id) and appends
  -> finalize Bridge Zarr store
```

Bridge sampling still consumes complete Exact tiles because membership is
spatial and value-neutral. Sparse value ranges optimize later reads, not sample
selection.

### Spatial levels and overview

```text
one through four immediate-finer Zarr tiles
  -> complete tile reads
  -> rebase tile-local coordinates into the coarser tile
  -> concatenate candidates
  -> fresh value-neutral sampler
  -> take the selected aligned payload rows
  -> bucket writer orders by (value_id, point_id) and appends
```

Every level uses exactly the same bucket writer and reader. Overview is the last
planned spatial level, not a special storage backend.

## Bucket construction API

The storage primitive accepts a complete immutable bucket plan before point
writing:

```text
BucketPlan
  level
  bucket_id
  ordered tile coordinates
  expected point count per tile
  total N
  physical chunk/shard/codec settings
```

Writers then provide one complete logical tile at a time in plan order:

```text
with BucketWriter(path, plan) as writer:
    writer.write_tile(tile_x, tile_y, payload)
    ...
    result = writer.finalize()
```

The primitive validates each payload count against the plan, orders its rows by
`(value_id, point_id)`, writes aligned point arrays sequentially, and appends
sparse range records in bounded blocks. Range arrays may grow in coarse blocks
because `M` is not always known before sampling; they are trimmed and frozen at
finalization. They must not resize once per point or retain one Python object per
range for a complete level.

### Finalization count reconciliation

Finalization obtains counts from the finalized physical arrays; it must not
derive the observed counts from the plan or descriptors. After all planned
tiles have been written, define:

```text
physical_point_count = location.shape[0]
physical_range_count = ranges/value_id.shape[0]
```

Before publishing root attributes or returning a result, require:

```text
location.shape                         == (physical_point_count, 2)
point_id.shape                         == (physical_point_count,)
value_id.shape                         == (physical_point_count,)
writer point cursor                    == physical_point_count
tile_offset[-1]                        == physical_point_count
sum(descriptor.n_points)               == physical_point_count
BucketPlan.point_count                 == physical_point_count

ranges/value_id.shape                  == (physical_range_count,)
ranges/row_start.shape                 == (physical_range_count,)
ranges/row_count.shape                 == (physical_range_count,)
writer range cursor                    == physical_range_count
tile_indptr[-1]                        == physical_range_count
```

The growable range arrays are first trimmed to the writer's completed range
cursor and are then measured through their finalized shapes. The point cursor
is checked separately because a preallocated point array can have its planned
shape even after an incomplete write.

Only after every equality succeeds does finalization write
`point_count=physical_point_count` and `range_count=physical_range_count` to the
root attributes and return the same observed values in `_BucketWriteResult`.
Thus `point_count` is not `sum(descriptor.n_points)` under another name: the two
values come from independent physical and logical sources and are reconciled.
Any mismatch fails finalization and leaves the staging generation incomplete.

The independent on-disk validator reopens the store without trusting
`_BucketWriteResult`. It repeats the array-shape, root-attribute, offset, and
pointer checks. Later cross-artifact validation additionally reconciles the
manifest descriptor counts with these validated physical counts.

Exact can derive its bucket plan from the materialized bucket. Bridge and
spatial writers derive expected output tile counts from candidate counts and
level capacities before opening their output bucket.

## Lookup semantics

### Complete tile

```text
(level, tile_x, tile_y)
  -> manifest row
  -> bucket_path, bucket_tile_index
  -> verify bucket tile_x[i], tile_y[i]
  -> tile_offset[i:i+2]
  -> aligned location/value_id/point_id slices
```

### Selected values

```text
selected labels
  -> values.parquet -> value_ids
  -> tile_value_counts.parquet -> positive visible tiles
  -> manifest.parquet -> bucket path and tile index
  -> tile_indptr -> tile's sorted range records
  -> selected value range(s)
  -> only intersecting aligned point-array chunks
```

Opening a Zarr bucket loads metadata and required small index chunks; it must not
materialize the complete point arrays. Selection is beneficial only when the
selected intervals touch fewer chunks than complete positive tiles. A rare value
present in every visible tile remains a meaningful worst case and is measured.
The Z2 reader returns a complete `_PointPayload`, so its selected read includes
`point_id`. A later acceptance/runtime API may add a coordinates-only read that
omits `point_id`; that optimization is deliberately outside the primitive.

## Implementation slices

Each slice is implemented only under `multi_scale_cache_points_zarr` and its
tests unless it explicitly changes documentation or a future public entrypoint.
Every slice ends in a coherent focused test set. The existing package and its
tests must remain untouched.

The dependency sequence is:

| Slice | Delivers | Depends on |
|---|---|---|
| Z0 | isolated implementation boundary | roadmap decision |
| Z1 | fresh models, planning, hashes, and payload contracts | shared validated source |
| Z2 | standalone Zarr bucket writer, reader, and validator | Z1 |
| Z3 | Exact source-to-Zarr construction and Xenium Exact gate | Z1–Z2 |
| Z4 | fresh sampler and Bridge Zarr construction | Z3 |
| Z5 | fresh rebasing and all spatial/overview Zarr levels | Z4 |
| Z6 | metadata and compact Parquet indexes | Z3–Z5 |
| Z7 | independent complete-generation validation | Z6 |
| Z8 | guarded end-to-end build and publication | Z7 |
| Z9 | acceptance reader and full-Xenium evaluation | Z8 |
| Z10 | explicit architecture-adoption decision | Z9 |

No slice depends on an adapted Parquet point writer or a compatibility reader.

### Slice Z0: freeze the isolated implementation boundary — resolved

#### Goal

Make the new-package strategy authoritative before implementation.

#### Work

- declare `multi_scale_cache_points_zarr` the sole implementation location for
  this Zarr-backed candidate;
- reuse only canonical source models, validation, value-normalization, and
  source-signature facts;
- allow duplication of all derived-cache logic;
- forbid imports from the existing writer package;
- forbid transitional or mixed point-payload backends;
- retain the existing package unchanged as the reference and rollback point;
- use `transcripts_vis_zarr` as a noncolliding isolated output path until public
  integration is decided.

#### Exit criteria

- this roadmap contains no incremental Parquet-to-Zarr migration plan;
- Z1 can create a new package without changing existing writers;
- adoption or non-adoption has a simple package-level boundary.

### Slice Z1: scaffold fresh contracts and build planning — resolved

**Status:** implemented and verified on 2026-08-13.

#### Goal

Create a standalone, testable logical foundation without writing persistent
point payloads yet.

Z1 freezes only pure Python and NumPy contracts. It opens no source Parquet
file, imports no Zarr module, creates no Dask graph, and writes no cache object.

#### Package scaffold and dependency rule

Create:

```text
src/napari_harpy/core/multi_scale_cache_points_zarr/
  __init__.py
  models.py
  build_plan.py
  hashing.py
  payload.py
  storage/
    __init__.py
    models.py

tests/multi_scale_cache_points_zarr/
  test_build_plan.py
  test_hashing.py
  test_payload.py
  test_models.py
  test_import_boundary.py
```

All Z1 implementation symbols are private. `__init__.py` does not expose a
builder prematurely.

The only permitted imports from `multi_scale_cache_points` in Z1 are source
types needed at the validation boundary:

```text
PointsBounds
ValidatedPointsSource
```

Z1 may accept a `ValidatedPointsSource` but must not call source scanning or
inspect Parquet content. It must not import any existing build-plan, hashing,
sampling, writer, or writer-support implementation. In particular, imports
whose module path begins with either of the following fail the boundary test:

```text
napari_harpy.core.multi_scale_cache_points.build_plan
napari_harpy.core.multi_scale_cache_points.writer
```

The test scans imports in the new package rather than relying only on which
branches happened to execute.

#### NumPy point-payload contract

Define a frozen container `_PointPayload` in `payload.py`:

```text
_PointPayload
  x_rel: np.ndarray      shape=(N,) dtype=float32
  y_rel: np.ndarray      shape=(N,) dtype=float32
  value_id: np.ndarray   shape=(N,) dtype=uint32
  point_id: np.ndarray   shape=(N,) dtype=uint64
```

The constructor validates rather than silently coercing:

- every field is an `np.ndarray`, not an arbitrary array-like value;
- every array is one-dimensional, C-contiguous, and has its exact dtype;
- all four lengths are equal;
- `N >= 1` because empty logical tiles are never stored;
- `x_rel` and `y_rel` contain only finite values;
- the four arrays do not share a semantic field through accidental broadcasting
  or shape normalization;
- boolean, signed, wider, and platform-dependent integer dtypes are rejected
  rather than cast.

The dataclass is a frozen container. It exposes read-only array views so payload
consumers cannot mutate through the payload fields. Construction does not make
a defensive full-tile copy merely to enforce ownership; callers must not mutate
the original backing arrays while the payload is in use. This borrowed-buffer
rule is documented and tested.

Provide only small storage-neutral operations needed by later slices:

```text
n_points -> int
take(indices: np.ndarray[int64]) -> _PointPayload
ordered_by_value_and_point_id() -> _PointPayload
```

`take` applies one index vector to all four arrays and preserves alignment. It
requires a one-dimensional integer index vector with in-bounds unique indices;
the sampler, not `_PointPayload`, determines membership. The ordering helper is
stable and uses `(value_id, point_id)`; it does not change membership.

Generic payload validation does not know a tile size, so it does not enforce an
upper coordinate bound. Exact, Bridge, and spatial writers later require
`0 <= x_rel <= tile_size` and `0 <= y_rel <= tile_size` in the relevant level
context, retaining the accepted upper-edge tolerance.

Z1 deliberately chooses NumPy rather than Arrow for this internal boundary.
The canonical value table and final compact indexes may still use Arrow and
Parquet. No logical writer needs to create a four-column Arrow table merely to
pass points to Zarr.

#### Fresh build-plan contracts

Define a fresh enum:

```text
_LevelKind
  EXACT = "exact"
  BRIDGE = "bridge"
  SPATIAL = "spatial"
```

Define `_LevelBuildPlan`:

```text
level: int
kind: _LevelKind
tile_size: int
grid_width: int
grid_height: int
max_points_per_tile: int | None
point_count_upper_bound: int
```

Required invariants:

- `level` is a nonnegative `int16`-compatible integer;
- `tile_size`, `grid_width`, and `grid_height` are positive integers;
- `grid_width` and `grid_height` are at most `2**32`, so their maximum valid
  tile index fits `uint32`;
- `point_count_upper_bound` is positive and at most `int64_max`;
- Exact has `max_points_per_tile is None`;
- Bridge and spatial levels have a positive `int64`-compatible capacity;
- `relative_directory` is derived as `levels/level_<level>` and is not stored as
  independent mutable data.

Define `_PointsCacheBuildPlan`:

```text
x_origin: float
y_origin: float
leaf_tile_size: int
overview_point_budget: int
levels: tuple[_LevelBuildPlan, ...]
```

Required invariants:

- origins are finite floats;
- `leaf_tile_size` and `overview_point_budget` are positive integers;
- levels are nonempty and numbered consecutively from zero;
- level 0 is uncapped Exact;
- if present, level 1 is Bridge and has Exact's tile size and grid geometry;
- every later level is spatial, doubles the preceding tile size, and has
  `grid_width == ceil(finer.grid_width / 2)` and
  `grid_height == ceil(finer.grid_height / 2)`;
- upper bounds never increase toward coarser levels;
- the terminal upper bound does not exceed `overview_point_budget`.

The serialized cache-level number and the spatial-level number are distinct.
Cache level 1 is Bridge, so Spatial 1 starts at cache level 2:

```text
validated source points
        |
        v
cache level 0: Exact
  tile size: S
  grid: W x H
  capacity: uncapped
        |
        | sample within the same logical tiles
        v
cache level 1: Bridge
  tile size: S
  grid: W x H
  scheduled capacity: 4,096 points/tile
        |
        | combine each 2 x 2 group of finer tiles and sample
        v
cache level 2: Spatial 1
  tile size: 2S
  grid: ceil(W / 2) x ceil(H / 2)
  scheduled capacity: 8,192 points/tile
        |
        v
cache level 3: Spatial 2
  tile size: 4S
  grid: ceil(W / 4) x ceil(H / 4)
  scheduled capacity: 16,384 points/tile
        |
        v
cache level 4: Spatial 3
  tile size: 8S
  scheduled capacity: 32,768 points/tile
        |
       ...
        v
terminal overview level
  complete point-count upper bound <= overview_point_budget
```

Exact and Bridge therefore have identical logical tile geometry; Bridge adds
sampling without changing spatial resolution. Each Spatial level doubles the
preceding tile edge and normally doubles its scheduled capacity. Once one tile
covers the dataset, the final capacity may instead be clamped to
`overview_point_budget`. This is the logical level hierarchy and is independent
of physical Zarr arrays, buckets, and chunking. If the Exact row count already
fits the overview budget, the hierarchy stops at cache level 0.

Implement one pure planner:

```text
_plan_points_cache(
    validated: ValidatedPointsSource,
    *,
    leaf_tile_size: int,
    overview_point_budget: int,
) -> _PointsCacheBuildPlan
```

It reads only validated aggregate facts: `row_count` and `bounds`. It must not
read `validated.source.parquet_path` or any source row.

Freeze the planning policy:

1. Align `x_origin` and `y_origin` downward to integer multiples of
   `leaf_tile_size` using the validated minima.
2. Compute each grid dimension as
   `floor((maximum - origin) / tile_size) + 1`.
3. Create uncapped Exact level 0 with the source row count as its upper bound.
4. If the source row count already fits the overview budget, stop with Exact.
5. Otherwise create Bridge level 1 with the same tile size and geometry as
   Exact and a scheduled capacity of 4,096 points per tile.
6. Create spatial levels by doubling both the preceding tile size and scheduled
   capacity.
7. For every sampled level, calculate
   `min(finer_upper_bound, grid_width * grid_height * capacity)`.
8. Once one tile covers the dataset, reduce that terminal tile's effective
   capacity to the overview budget if necessary.
9. Stop at the first level whose complete upper bound is at most the overview
   budget.

Reject row counts above `int64_max`, levels above `int16_max`, grid dimensions
above the `uint32` coordinate space, nonfinite bounds, and arithmetic that would
escape the serialized ranges.

#### Fresh bucket policy

Implement deterministic bucket routing in `hashing.py` without importing the
existing helper:

```text
tile_key = (uint64(tile_y) << 32) | uint64(tile_x)
tile_hash = SplitMix64(tile_key)
bucket_id = tile_hash % bucket_count
```

The policy has a new-package-owned version identifier and fixed golden vectors.
It accepts one-dimensional C-contiguous `uint32` NumPy arrays for `tile_x` and
`tile_y`, requires identical shapes, and returns `uint64` bucket IDs. Invalid
dtypes, a nonpositive bucket count, or a bucket count above `2**32` fail rather
than being silently cast.

Use the current production-candidate construction target of 2,000,000 planned
points per bucket:

```text
bucket_count = max(1, ceil(level.point_count_upper_bound / 2_000_000))
```

This is a construction policy, not part of the logical tile identity. The Z3
full-Xenium Exact run retained it for the remaining construction slices: it
produced 69 nonempty stores averaging 1,979,402 points, with a largest bucket of
2,547,160 points, a 44.09-second build, 4.17 GiB peak build RSS, and 4,746
filesystem objects across the complete Exact payload. Those results do not show
an immediate construction or object-count problem that requires changing the
policy before Bridge and spatial construction are implemented.

The value is nevertheless inherited from the Parquet-backed design and is
probably conservative for sharded Zarr. Physical read and decode granularity is
defined by inner chunks, while shards group those chunks into storage objects;
increasing the logical bucket size therefore does not make a reader decode a
complete bucket. For the same Xenium Exact level, a target of 10,000,000 points
would plan 14 buckets averaging approximately 9.76 million points. This is the
leading Z9 alternative because a multi-tile viewport or distributed-value query
could open materially fewer stores and read fewer bucket metadata documents.

The corresponding risk is construction memory: one finalizer materializes and
stable-sorts one complete shuffled bucket, and up to `dask_worker_count` such
buckets may be active. A ten-million target makes that per-finalizer unit roughly
five times larger than the current average. Do not change the target or the
inner chunk/shard settings during Z4--Z8 based on speculation. Slice Z9 owns one
explicit decision between the retained two-million policy and the ten-million
candidate, using viewport buckets-opened/latency evidence together with build
peak memory. Changing bucket size alone is not a reason to change point or range
chunk and shard dimensions.

Bucket filenames use a canonical minimum width of three digits and do not
depend on the complete planned bucket count:

```text
levels/level_<level>/bucket-<bucket_id:03d>.zarr
```

The width is a minimum: bucket IDs above 999 expand normally. Numeric ordering
uses `bucket_id`, never lexicographic filename order. This count-independent
rule lets every construction model derive the path from bucket identity alone.

Empty bucket IDs create no plan, descriptor, or store.

#### Tile and bucket models

Define `_TileDescriptor` in `models.py`:

```text
level: int
bucket_id: int
bucket_tile_index: int
tile_x: int
tile_y: int
n_points: int
```

Validate:

- serialized integer ranges: level `int16`, bucket/tile/index `uint32`, and
  `n_points` in `[1, int64_max]`;
- booleans are rejected as integers;
- `bucket_path` is a read-only property derived canonically from `level` and
  `bucket_id`; callers cannot supply a conflicting path.

Define `_PlannedTile` in `storage/models.py`:

```text
tile_x: int
tile_y: int
n_points: int
```

Define `_ZarrWriteSettings` without importing Zarr:

```text
point_chunk_rows: int
point_shard_rows: int
range_chunk_rows: int
range_shard_rows: int
codec_id: str
```

All row settings are positive integers. `point_shard_rows` is an integer
multiple of `point_chunk_rows`, `range_shard_rows` is an integer multiple of
`range_chunk_rows`, and `codec_id` is a nonempty versioned string. The settings
describe requested physical behavior; Z2 maps the supported ID to the exact
codec objects frozen above. Because Z1 was scaffolded before range sharding was
selected, adding `range_shard_rows` to the implemented model and its focused
tests is the narrow prerequisite adjustment at the start of Z2.

Define `_BucketPlan`:

```text
level: int
bucket_id: int
tiles: tuple[_PlannedTile, ...]
settings: _ZarrWriteSettings
```

Required invariants:

- a plan contains at least one nonempty tile;
- `bucket_path` is the same canonical property of `level` and `bucket_id` used
  by `_TileDescriptor`;
- tile coordinates are unique and strictly ordered by `(tile_y, tile_x)`;
- tile fields fit the serialized ranges;
- the sum of tile counts is positive and at most `int64_max`;
- derived properties expose `tile_count`, `point_count`, and the exact
  `tile_offset` prefix sums without storing a second independent count.

Define `_BucketWriteResult`:

```text
tile_descriptors: tuple[_TileDescriptor, ...]
point_count: int
range_count: int
```

It represents a finalized nonempty store. The complete bucket identity remains
on the standalone descriptors because they later become manifest rows.
`level` and `bucket_id` are taken from their shared descriptor identity, and
`bucket_path` is derived canonically from those values. None is duplicated as
stored result state. Descriptors must all have the same identity; their
bucket-local indexes are exactly
`0..K-1` in tile order. `point_count` remains an explicit independent physical
total so finalization can reconcile it with the sum of descriptor counts.
`range_count` is at least the tile count and at most the point count.

Define `_LevelWriteResult`:

```text
buckets: tuple[_BucketWriteResult, ...]
```

Its `level` is derived from the shared level of its nonempty bucket results,
again leaving the descriptors as the single source of identity. Buckets are
ordered by unique `bucket_id`. Other derived properties flatten globally
ordered tile descriptors and calculate point, tile, bucket, and range totals.
Across the level, tile coordinates and `(bucket_id, bucket_tile_index)` keys
are unique. An empty level result is invalid because a validated source is
nonempty and every planned constructed level retains at least one point.

#### Z1 non-goals

Z1 does not:

- import or configure Zarr;
- create directories, arrays, Parquet files, or metadata;
- implement `_BucketWriter` or `_BucketReader`;
- read canonical source rows;
- create a Dask graph;
- implement value-neutral sampling or coordinate rebasing;
- define final published metadata or Arrow index schemas;
- expose a public builder before the guarded integration slice;
- alter any file under `multi_scale_cache_points` or its existing tests.

#### Focused tests

- `_PointPayload` accepts exact aligned arrays and rejects wrong type, rank,
  dtype, length, contiguity, emptiness, and nonfinite coordinates;
- payload fields are read-only views, `take` keeps all fields aligned, and
  value/point ordering is deterministic;
- aligned origins for positive, negative, and exact-boundary minima;
- Exact-only, Exact-plus-Bridge, and multi-spatial build plans;
- one-tile overview capacity reduction;
- grid, level, count, and capacity overflow rejection;
- exact Bridge geometry and spatial doubling invariants;
- fixed SplitMix64 bucket vectors, dtype rejection, and deterministic bucket
  paths;
- descriptor-derived paths, integer ranges, uniqueness, order, and bucket
  ownership;
- exact `BucketPlan` prefix sums and total reconciliation;
- valid and invalid bucket/level results;
- monkeypatched source access proving the pure planner does not read source
  files.

#### Exit criteria

- a validated source can produce a complete fresh Zarr-cache build plan;
- `_PointPayload` is the only point interchange contract planned for Exact,
  Bridge, spatial construction, and the Zarr primitive;
- tile, bucket-plan, bucket-result, and level-result ownership is unambiguous;
- logical contracts contain no Arrow-table requirement and no Parquet row-group
  concept;
- hashing and bucket naming are deterministic and independently versioned;
- Z1 imports no Zarr implementation and writes no files;
- no existing writer module has changed;
- Z2 can be developed entirely against tiny NumPy payloads and bucket plans.

### Slice Z2: implement the standalone Zarr bucket primitive — resolved

**Status:** implemented and verified on 2026-08-13.

Focused verification completed with all 102 tests in
`tests/multi_scale_cache_points_zarr` passing. The opt-in synthetic bucket
characterization also completed with the current production-candidate
chunk/shard settings for 131,394 points across representative average Exact,
dense Exact, and Bridge tiles. These measurements establish operability only;
they are not a numerical acceptance threshold and do not replace the
full-Xenium Z3 gate.

#### Goal

Implement, read, and independently validate one bucket using only a
`_BucketPlan` and one `_PointPayload` at a time. Z2 remains independent of Dask,
canonical source readers, sampling, rebasing, pyramid writers, manifests, and
publication.

#### Files and ownership

Implement the primitive under:

```text
multi_scale_cache_points_zarr/storage/
  _schema.py
  bucket_writer.py
  bucket_reader.py
  bucket_validation.py
```

`_schema.py` is the single private source for payload schema version, exact
root-attribute literals, array names/dtypes/fill values, chunk-key encoding,
and the `codec_id` registry. Writer, reader, and validator import these facts
rather than reproducing them independently.

The existing Z1 models remain the ownership boundary:

```text
_BucketPlan       expected bucket identity, tile order, counts, and settings
_PointPayload     actual aligned rows supplied for one logical tile
Zarr arrays       observed physical output
_BucketWriteResult
                  finalized descriptors and observed physical totals
```

The plan, supplied payloads, and finalized arrays are independent count sources
that must be reconciled. Z2 does not import or adapt any writer or reader from
`multi_scale_cache_points`.

Before opening a Zarr store, Z2 makes the already documented narrow extension
to the implemented Z1 settings model: add `range_shard_rows`, validate it as a
positive serialized integer, require exact divisibility by `range_chunk_rows`,
and update only the focused Zarr-package model tests. This is the first Z2 task,
not a reopening of Z1 planning or payload design.

#### Writer lifecycle

The intended interface is:

```text
with _BucketWriter(staging_root, plan) as writer:
    writer.write_tile(tile_x, tile_y, payload)
    ...
    result = writer.finalize()
```

`staging_root` is the generation root, not a caller-composed bucket path. The
writer derives the only target from `plan.bucket_path` and refuses to overwrite
an existing target.

The lifecycle is explicit:

```text
NEW -> OPEN -> FINALIZED -> CLOSED
           \-> FAILED -> CLOSED
```

- writes are accepted only while open;
- every planned tile is supplied exactly once, in plan order;
- coordinates and `payload.n_points` must match the next `_PlannedTile`;
- `finalize` succeeds exactly once and only after every planned tile;
- root count attributes are absent until all final checks succeed;
- successful finalization writes final attributes, closes the store, and
  returns `_BucketWriteResult`;
- an ordinary write/finalization exception marks the writer failed, closes its
  handles, and removes that exact partial bucket from the isolated staging
  generation;
- a process crash may leave a partial bucket, but the enclosing generation has
  no `COMPLETED` marker and cannot be published or read as a valid cache.

#### Point writes and shard buffering

Create the three point arrays immediately at the final `plan.point_count` shape.
They are never resized. Write `location[:, 0] = x_rel` and
`location[:, 1] = y_rel` without assembling a second full-bucket location
matrix.

For each tile, the writer orders rows by `(value_id, point_id)` after membership
has been fixed. It appends the four aligned fields to one bounded
`point_shard_rows` buffer shared across tile boundaries. Complete shard-sized
blocks are flushed once; only the final partial shard is flushed at
finalization. Large payloads may stream through that buffer and need not be
copied in full.

At the current production-candidate settings, the four aligned buffers occupy
about 2.6 MiB:

```text
131,072 * (2 * float32 + uint32 + uint64)
```

The small `tile_x`, `tile_y`, and planned `tile_offset` arrays are written once.
Their values are derived from `_BucketPlan`, but finalization still reconciles
their terminal offset against the independent writer cursor and physical array
shapes.

#### Sparse-range writes

After a tile is ordered, contiguous equal `value_id` runs become bucket-global
`ranges/value_id`, `ranges/row_start`, and `ranges/row_count` records. The
writer records the completed range cursor for that tile in `tile_indptr`.

The final range count `M` is not part of `_BucketPlan`. The writer therefore
uses one bounded `range_shard_rows` NumPy buffer shared across logical tile
boundaries. The range arrays:

- start with one `range_shard_rows` capacity block;
- grow geometrically, rounded to a multiple of `range_shard_rows`;
- receive complete shard-sized writes where possible rather than one
  resize/write per range;
- write the final partial range shard once during finalization;
- are trimmed to the exact completed range cursor before physical counts are
  measured;
- never retain one Python object per range for the complete bucket.

`tile_indptr` is a small `(K + 1,)` in-memory prefix array and is written once
during finalization.

#### Zarr creation and strict reads

Create a new local Zarr v3 group with unconsolidated metadata. Apply the aligned
point chunks/shards and range chunks/shards from `_ZarrWriteSettings`. Each
small tile/index array uses one unsharded chunk. Writer code supplies sequential
shard-sized array slices, while Zarr owns shard encoding, indexing, file naming,
and inner-chunk placement.

Every array is created with its exact schema dtype, `fill_value=0`, the standard
Zarr v3 default chunk-key encoding with `/` separator, and the `zstd-v1` pipeline
defined above. For sharded arrays this means independently compressed inner
chunks plus the checksummed end-of-shard index. These choices are explicit
rather than inherited from mutable library defaults.

Z2 maps the versioned settings `codec_id` through the registry above rather
than accepting arbitrary codec objects or parameters. The reader uses the
canonical point-array chunk metadata for selected reads. The validator derives
point and range chunk/shard row sizes from their canonical `value_id` arrays,
requires their parallel arrays to use the same layouts, and verifies dtype and
codec properties against the format contract.

Every array is created with `write_empty_chunks=True`. All reopened arrays used
by `_BucketReader` and the independent validator set
`read_missing_chunks=False`. Thus valid all-zero chunks are physically present,
while a missing chunk or shard is structural corruption rather than an implicit
fill-value region.

#### Reader contract

The reader is scoped to one bucket so repeated tile reads reuse open store and
array handles:

```text
with _BucketReader(cache_root, level, bucket_id) as reader:
    complete = reader.read_complete(descriptor)
    selected = reader.read_selected(descriptor, selected_value_ids)
```

Both methods verify descriptor bucket identity, bucket-local index, stored tile
coordinates, and descriptor count before reading point data. Calls after close
fail.

`read_complete` resolves `tile_offset[i:i+2]` and returns an exact
`_PointPayload`. `read_selected` requires a nonempty, one-dimensional,
strictly increasing unique `uint32` ID array and performs:

```text
tile_indptr[i:i+2]
  -> this tile's sorted sparse range records
  -> binary search selected value IDs
  -> map selected row intervals to inner point-chunk IDs
  -> group overlapping or consecutive touched chunks
  -> read each group's minimal exact-row envelope
  -> extract the exact selected intervals
```

It does not scan the tile's point-level `value_id` slice. It returns `None` when
no requested value is present because `_PointPayload` deliberately cannot be
empty. Otherwise it returns a complete payload including `point_id`. Splitting
the stored `(N, 2)` `location` slice into contiguous `x_rel` and `y_rel` arrays
is an accepted Z2 conversion cost and is measured rather than optimized now.
The inner chunk remains the selected-read granularity even though several
chunks share one physical shard.

#### Independent bucket validation

The validator's input and result are:

```text
_validate_bucket(cache_root, level, bucket_id) -> _BucketWriteResult
```

It receives neither `_BucketPlan` nor the writer's result. After reopening the
canonical bucket read-only, it reconstructs descriptors and observed counts
from the physical store and validates:

- Zarr format 3 and the exact logical group/array hierarchy;
- exact required root attributes, supported schema version, and `codec_id`;
- dtypes, ranks, shapes, zero fill values, chunk-key encoding, chunks, shards,
  codecs, shard-index configuration, and strict chunk presence;
- ordered unique `(tile_y, tile_x)` pairs and exact local indexes;
- monotonic `tile_offset` and `tile_indptr` with correct terminal counts;
- finite, nonnegative relative coordinates;
- per-tile `(value_id, point_id)` ordering;
- strictly increasing unique range values within each tile;
- positive counts and exact gap-free, overlap-free coverage of each tile;
- agreement between every range value and its point-level `value_id` rows;
- root counts against finalized physical shapes.

Validation proceeds by tile, range, or chunk and does not materialize a complete
bucket. Unexpected logical Zarr groups or arrays are rejected. The validator
does not depend on raw chunk filenames or other private Zarr storage details;
strict reads and codec checksums detect missing or corrupt physical payloads.

Complete `_validate_bucket` scans are used by focused corruption tests and
explicit exhaustive acceptance or diagnostic validation. Slice Z7 may factor
its metadata, hierarchy, layout, root-count, and compact-pointer checks into a
shared structural helper, but its normal publication path must not obtain a
`_BucketWriteResult` by rereading every point and range payload. It instead
reconciles independently reopened compact on-disk facts with the manifest and
other cache artifacts while relying on each writer's controlled construction
proof for complete payload semantics. `_BucketReader` does not run complete
bucket validation during interactive reads, and `_BucketWriter.finalize()`
retains only its immediate writer-side reconciliation responsibilities.

Upper coordinate bounds remain a level-writer responsibility because neither
`_PointPayload` nor `_BucketPlan` knows the tile size. Exact, Bridge, and spatial
writers enforce `x_rel <= tile_width` and `y_rel <= tile_height` before calling
the bucket writer. Explicit exhaustive validation may repeat that point-level
check; normal publication validation does not reread all coordinates solely to
replay an invariant already enforced during controlled construction.

#### Work

- implement the writer, reader, codec mapping, and independent validator exactly
  against the contracts above;
- use only public Zarr v3 APIs and explicitly close local stores;
- keep all construction and validation memory bounded independently of bucket
  point count;
- make structural failure closed and deterministic without adding generation
  publication or repair behavior to Z2.

#### Implementation order

Implement Z2 in these reviewable stages without creating additional roadmap
slices:

1. add `range_shard_rows` to `_ZarrWriteSettings` and its focused tests;
2. define `_schema.py` and test the exact attributes, array specifications,
   chunk-key encoding, and `zstd-v1` codec map;
3. implement store/array creation plus bounded point and range shard buffers;
4. implement finalization, physical/logical count reconciliation, close, and
   partial-target cleanup;
5. implement complete and chunk-aware selected reads;
6. implement independent on-disk validation;
7. add lifecycle, corruption, chunk-boundary, and shard-boundary tests;
8. run the opt-in small synthetic characterization benchmark.

#### Focused tests

- writer state transitions: before enter, after failure, double finalization,
  and calls after close;
- missing, duplicate, out-of-order, coordinate-mismatched, and count-mismatched
  tile writes;
- one and several tiles per bucket;
- one and several values per tile;
- values absent from some tiles;
- complete-tile round trips;
- one- and multi-value selected reads, adjacent-range coalescing, and `None` for
  no match;
- tiles and value runs crossing point and range chunk/shard boundaries;
- selected ranges sharing an inner point chunk cause that chunk to be read only
  once;
- final partial point and range shards, plus payloads and range streams larger
  than their respective shard buffers;
- range capacity growth across several `range_shard_rows` boundaries and exact
  trimming;
- a legitimate all-zero chunk remains readable because it was physically
  stored;
- deleting a chunk or shard causes strict reader and validator failure rather
  than fill-value reconstruction;
- exact schema-version, point-order, coordinate-encoding, fill-value,
  chunk-key, inner-codec, and shard-index metadata;
- coordinate and `point_id` alignment;
- physical point-array shapes, writer cursor, terminal tile offset, descriptor
  totals, plan total, root `point_count`, and result `point_count` all reconcile;
- finalized range-array shapes, writer cursor, terminal `tile_indptr`, root
  `range_count`, and result `range_count` all reconcile;
- partial preallocated point writes and deliberately mismatched count sources
  fail finalization without producing a valid result;
- corrupt schema/codec attributes, dtype or shape metadata, offsets, pointers,
  range coverage, range values, point ordering, and unexpected logical nodes;
- the independent validator reconstructs the same valid result without
  receiving the plan or writer result;
- unknown descriptor, closed reader, failed write, exact partial-target cleanup,
  and refusal to overwrite an existing target.

Tests use deliberately tiny chunks, shards, and range blocks so every physical
boundary is exercised with small payloads. Normal unit tests assert declared
configuration, logical content, shapes, and invariants; they do not assert exact
compressed byte counts.

#### Microbenchmark

Use synthetic payloads representing:

- an average Exact tile;
- a dense approximately 108,598-point Exact tile;
- a 4,096-point Bridge tile;
- localized and distributed value ranges.

Record complete and selected cold/warm reads, compressed bytes, chunks touched,
decoded-row estimates, shard objects, and write time for a small configuration
matrix. This is an opt-in characterization benchmark, not a numerical unit-test
gate.

#### Exit criteria

- the primitive roundtrips aligned payloads exactly;
- selected lookup does not scan the point-level `value_id` array;
- point and range shard-buffered writing remains bounded by settings rather
  than bucket size;
- valid all-zero data roundtrips while missing chunks, shards, and structural
  corruption fail closed;
- validation reconstructs a result without trusting in-memory construction
  objects;
- a measured physical configuration is ready for full-scale Exact construction.

### Slice Z3: implement fresh Exact construction directly to Zarr

#### Goal

Build a complete Exact level from `ValidatedPointsSource` without using any
existing writer code or intermediate point Parquet.

Z3 owns source annotation, redistribution, Exact bucket planning, and
coordination of the Z2 storage primitive. It does not implement Bridge or
spatial sampling, cache indexes, metadata, completion markers, publication, or
the final cross-artifact validator.

#### Files and dependency boundary

Create:

```text
src/napari_harpy/core/multi_scale_cache_points_zarr/
  writer/
    __init__.py
    exact.py

tests/multi_scale_cache_points_zarr/
  test_exact_writer.py
```

All Z3 symbols remain private. `exact.py` may import the existing canonical
source models, `VALUE_NORMALIZATION_METHOD`, `_normalized_row_values`, and the
point-ID policy identifier. It must not import any module from
`multi_scale_cache_points.writer`. Source annotation, the Dask graph, bucket
finalizers, and level reconciliation are implemented fresh in the Zarr package.

#### Exact writer API and execution settings

Define an Exact-specific execution configuration:

```text
_ExactWriterConfig
  zarr_settings: _ZarrWriteSettings
  dask_worker_count: int
```

`dask_worker_count` is a positive local threaded-scheduler worker count.
`bucket_count` is deliberately absent: the writer derives it from Exact's
`_LevelBuildPlan` through `_bucket_count_for_level`, so it cannot disagree with
the versioned routing policy. Chunk, shard, and codec choices remain explicit in
the supplied `_ZarrWriteSettings` rather than becoming Exact-writer constants.

The construction entry point is:

```text
_write_exact_level(
    validated,
    plan,
    *,
    staging_root,
    temporary_directory_root,
    config,
) -> _LevelWriteResult
```

Before creating a Dask graph it requires:

- `validated` is a `ValidatedPointsSource` using the supported normalization and
  point-ID policies;
- `plan` is a `_PointsCacheBuildPlan` whose first level is uncapped Exact level
  zero;
- `staging_root` is an existing isolated generation root;
- `levels/level_0` does not already exist; the Exact coordinator owns creation
  of this directory;
- `temporary_directory_root` is an existing caller-owned scratch root and is
  distinct from staged cache output;
- `config` contains valid Zarr settings and a positive worker count.

The function returns the existing storage-neutral `_LevelWriteResult` containing
only nonempty finalized bucket results ordered by numeric `bucket_id`.

#### Row-group-aligned source partitions and point IDs

Create one input partition per validated physical Parquet row group, not one per
complete source file. `ValidatedPointsSource` does not bound physical file size;
row-group alignment prevents one large file from becoming an unnecessarily
large construction partition while preserving physical source order.

Do not obtain these partitions by calling `dd.read_parquet` over the complete
dataset and inferring physical row-group identity from Dask partition order.
Instead, construct one small immutable read specification per validated row
group in canonical file and row-group order:

```text
_SourceRowGroupReadSpec
  relative_path
  row_group_index
  expected_row_count
  point_id_start
```

`point_id_start` is calculated by the coordinator from validated metadata before
the graph is created. Build the annotated Dask DataFrame conceptually as:

```text
dd.from_map(
    _read_and_annotate_row_group,
    row_group_read_specs,
    meta=_annotated_meta(),
)
```

Each `_read_and_annotate_row_group` task resolves the canonical source path,
opens `pyarrow.parquet.ParquetFile`, calls `read_row_group(row_group_index, ...)`
for only the selected x, y, and value columns, and closes that handle within the
task. It verifies the decoded row count before annotation and assigns IDs from
the explicit `point_id_start`. The physical row-group index therefore remains a
direct input to the read rather than being inferred through
`partition_info["number"]` or another Dask ordering detail.

For row group `r` in one validated source file, define:

```text
row_group_offset_within_file = sum(row_count of row groups before r in this file)

point_id = source_file.row_offset
           + row_group_offset_within_file
           + row_index_within_row_group
```

This exactly implements the canonical file-row-offset point-ID policy while
allowing row groups to execute and arrive in any order. Each task reads only the
selected x, y, and value columns and requires the decoded row count to equal the
validated row-group count. Empty physical row groups may produce empty input
partitions but never output tiles or stores.

Opening one Parquet handle per row-group task is an accepted correctness and
memory tradeoff for Z3 and is measured in the full-Xenium gate. If evidence later
shows that task or footer-open overhead is material, a read specification may be
extended to cover a bounded consecutive group of explicitly identified row
groups with precomputed offsets. It must not regress to an unbounded complete
source-file partition or make point identity depend on implicit Dask partition
ordering.

The annotated Dask metadata contract is exact:

```text
tile_x     uint32
tile_y     uint32
x_rel      float32
y_rel      float32
value_id   uint32
point_id   uint64
bucket_id  uint64
```

Unexpected columns, nulls, incompatible dtypes, nonfinite coordinates, decoded
row-count mismatches, arithmetic overflow, or IDs outside
`[0, validated.row_count)` fail annotation.

#### Canonical value-ID mapping

Materialize the validated vocabulary once in canonical `value_id` order from
`ValidatedPointsSource.value_table`. Every row-group task applies the canonical
`_normalized_row_values` helper and maps the normalized row-aligned strings to
positions in that vocabulary. Those positions are the serialized `uint32`
`value_id` values.

The writer verifies `validated.value_normalization_method` before graph
construction. A null or empty normalized label, a label absent from the
validated vocabulary, or a mapping outside the `uint32` value space fails
rather than being silently recoded.

#### Exact coordinate annotation

Calculate tile identity in float64 before encoding tile-relative coordinates:

```text
tile_x = floor((x - plan.x_origin) / exact.tile_size)
tile_y = floor((y - plan.y_origin) / exact.tile_size)

x_rel = float32(x - (plan.x_origin + tile_x * exact.tile_size))
y_rel = float32(y - (plan.y_origin + tile_y * exact.tile_size))
```

Signed temporary tile indexes must be nonnegative and strictly below Exact's
planned grid width and height before conversion to `uint32`. Relative
coordinates must remain finite and nonnegative. Because casting a coordinate
immediately below a boundary to float32 may round it to the tile size, the
accepted stored upper bound is inclusive:

```text
x_rel <= exact.tile_size
y_rel <= exact.tile_size
```

Do not clamp or move coordinates to make them pass. Reconstructing an intrinsic
coordinate from origin, tile index, and stored relative coordinate must agree
with the source coordinate within an absolute tolerance of
`spacing(float32(exact.tile_size))` and zero relative tolerance.

Map the exact `uint32` tile arrays through `_tile_bucket_ids`. A logical tile
therefore has one deterministic bucket regardless of which source row group
provided its points.

#### Disk shuffle and concurrency

Build a fresh Dask DataFrame from the annotated row-group partitions and use a
local disk shuffle with explicit integer divisions for all bucket IDs:

```text
annotated row-group partitions
  -> set_index(bucket_id, explicit divisions, disk shuffle, keep bucket_id)
  -> exactly one destination partition for each planned bucket ID
  -> one side-effecting delayed finalizer per destination
```

Every nonempty destination contains all rows for every logical tile assigned to
that bucket. A finalizer verifies that every retained `bucket_id` equals its
destination ID. Empty destinations return an empty private outcome and create no
`_BucketPlan`, descriptor, directory, or Zarr store.

Run finalizers with the local threaded scheduler and the configured worker
count. Every finalizer exclusively owns one canonical bucket path; concurrent
tasks never write the same store or shard. The worker count is therefore also a
bound on concurrently materialized bucket partitions and active shard buffers.

Create one uniquely named temporary child directory beneath
`temporary_directory_root` and install it as Dask's temporary directory only for
this computation. It contains no cache artifact and is removed when computation
unwinds after success or an ordinary failure. The caller-owned temporary root
remains.

#### Sorting and bucket-plan derivation

After resetting the shuffled bucket index, a nonempty finalizer stable-sorts the
materialized bucket only by:

```text
(tile_y, tile_x)
```

This makes every logical tile contiguous and orders tiles exactly as required by
`_BucketPlan`. Derive one `_PlannedTile(tile_x, tile_y, n_points)` from each
complete nonempty run and construct:

```text
_BucketPlan(
    level=0,
    bucket_id=destination_bucket_id,
    tiles=ordered_planned_tiles,
    settings=config.zarr_settings,
)
```

Do not globally sort the bucket by `(value_id, point_id)`. For each tile, create
one C-contiguous `_PointPayload` and pass it to `_BucketWriter.write_tile` in
plan order. The common storage primitive remains the single owner of the final
tile-internal `(value_id, point_id)` ordering and corresponding sparse-range
construction. This avoids performing the same value-major sort both in Z3 and
again in Z2.

Before writing a tile, Z3 enforces the level-aware relative-coordinate upper
bounds that `_PointPayload` and `_BucketPlan` cannot know. `_BucketWriter`
continues to enforce payload alignment, nonnegative coordinates, expected tile
identity and count, physical ordering, offsets, ranges, and final count
reconciliation.

#### Finalizer outcome and level reconciliation

Use a private Exact-finalizer outcome containing:

```text
bucket_result: _BucketWriteResult | None
value_id: sorted unique uint32 IDs present in this destination
value_count: aligned uint64 point totals
```

The sparse value totals are construction facts used only to reconcile Exact
against the canonical source-wide counts. They are not written as an
intermediate file and do not replace the bucket's physical sparse range arrays.

After all finalizers complete, the coordinator:

- discards empty outcomes and orders nonempty `_BucketWriteResult` values by
  numeric `bucket_id`;
- constructs `_LevelWriteResult` and requires `level == 0`;
- requires the sum of physical bucket point counts to equal
  `validated.row_count` and Exact's planned point-count upper bound;
- requires all point IDs observed by a finalizer to lie in the canonical range
  and to be unique within that complete bucket partition;
- requires the summed sparse per-value totals to equal every
  `ValidatedPointsSource.value_table["n_points"]` entry exactly;
- relies on `_LevelWriteResult` to reject duplicate bucket IDs, logical tile
  coordinates, and bucket-local descriptor keys;
- requires every descriptor coordinate to lie within Exact's planned grid and
  every descriptor count to be positive.

The controlled construction proof consists of disjoint canonical point-ID
ranges at annotation, a row-preserving routing operation, destination-ID checks,
per-bucket uniqueness, and exact total reconciliation. Z3 does not add a second
global shuffle solely to sort point IDs. The full-Xenium gate independently
proves global `0..N-1` coverage from the finalized Zarr arrays. Normal Slice Z7
validation relies on this controlled construction proof and persisted compact
accounting facts rather than repeating a complete point-payload scan for every
cache build. Exhaustive acceptance validation retains the independent global-ID
proof when it is explicitly requested.

`_validate_bucket` is not called automatically by `_BucketWriter.finalize()` or
by every normal Exact finalizer. Z2 writer-side reconciliation remains the
immediate construction check. Focused Z3 tests and the full-Xenium Z3 gate reopen
the completed stores with `_validate_bucket`. Complete `_validate_bucket` scans
remain consumers of focused corruption tests and explicit exhaustive
acceptance/diagnostic validation; normal Slice Z7 publication validation reuses
or factors out its metadata/layout checks without rereading every point row.

#### Memory contract

- One source task materializes at most one validated physical row group.
- One finalizer may materialize and stable-sort one complete shuffled bucket,
  never a complete level.
- Bucket size is a balancing target rather than a hard maximum because one
  logical tile cannot be split across bucket IDs.
- The finalizer retains at most its materialized bucket, sorting/grouping working
  memory, one complete logical tile payload, and the Z2 writer's bounded point
  and range shard buffers.
- It must not allocate a second full-bucket `(N, 2)` location matrix or retain
  one Python object per point or sparse range.
- Temporary tile arrays and payload references are released as their tile is
  appended; successful buckets are not reread during ordinary construction.
- Peak concurrent memory scales with `dask_worker_count`; the full-Xenium gate
  records the largest input row group, largest output bucket, and observed peak
  RSS before selecting the downstream worker configuration.

#### Failure and cleanup contract

- Annotation, shuffle, finalizer, bucket-writer, or reconciliation failures
  propagate from `_write_exact_level`.
- `_BucketWriter` removes the exact partially written bucket it owns after an
  ordinary write or finalization failure.
- Other finalizers may already have completed valid bucket stores. Z3 does not
  attempt cross-task rollback or resume; the enclosing staging generation is
  invalid and the later builder owns its removal.
- Dask scratch cleanup is independent of staged-output cleanup and occurs before
  `_write_exact_level` returns or raises.
- All Dask tasks, Zarr stores, and source handles are closed before control
  returns to the caller.
- Z3 never writes `COMPLETED`, publishes a generation, repairs partial output, or
  deletes a caller-owned staging or temporary root.
- No point-payload or intermediate tile/value-count Parquet file is created.

#### Focused tests

- multiple source files, multiple row groups, empty row groups, and exact
  row-group-offset point IDs;
- points for one tile arriving from several input partitions and being
  co-located in exactly one output bucket;
- dictionary-encoded and ordinary UTF-8 values, canonical whitespace
  normalization, canonical value IDs, and exact per-value totals;
- null, empty, unknown, or changed normalized values failing closed;
- negative and nonzero origins, fractional coordinates immediately around tile
  boundaries, grid bounds, inclusive stored upper bounds, and float32
  reconstruction tolerance;
- exact one-to-one source-row/`point_id` coverage, in-range IDs, and injected
  duplicate or missing rows;
- empty destination buckets producing no store and nonempty bucket results being
  numerically ordered;
- sparse tiles, dense tiles, several tiles sharing one bucket, and tile payloads
  crossing several tiny test chunks and shards;
- finalized point rows ordered by `(tile_y, tile_x, value_id, point_id)` through
  the combined Z3/Z2 responsibilities;
- sparse ranges supporting complete and selected roundtrips through
  `_BucketReader` without a point-level value scan;
- every Z3-produced bucket independently passing `_validate_bucket`;
- deterministic descriptors and decoded Zarr content under changed input-task
  and finalizer arrival order; normal tests do not require deterministic codec
  bytes;
- more than one worker producing several independent buckets without shared
  paths or handles;
- existing output paths, row-count mismatches, invalid coordinates, wrong bucket
  destinations, and unsupported source policies failing closed;
- injected annotation, shuffle, bucket-write, finalizer, and reconciliation
  failures, with Dask scratch removed and the staging generation left
  unpublished;
- inspection proving that no derived point Parquet or intermediate
  tile/value-count file was created.

#### Gate Z3: full-Xenium Exact evaluation

Provide an opt-in Exact evaluation script separate from normal unit tests. Build
the complete 136,578,750-point Xenium Exact level with the production-candidate
bucket, chunk, shard, codec, and worker settings. Reopen every produced bucket
through `_validate_bucket` and require the reconstructed result to agree with
the construction result.

Independently verify:

- every canonical source row appears exactly once;
- finalized `point_id` values are exactly the complete `0..N-1` set, using a
  bounded external or memory-mapped structure rather than a Python set of all
  IDs;
- all logical tile counts and Exact's total equal the validated source count;
- physical per-value totals equal the canonical validated value table;
- tile identities, grid bounds, and coordinate reconstruction tolerance hold
  across the complete level;
- every point payload is Zarr and no derived point Parquet exists.

Record:

- build time and peak RSS;
- validated source-file and row-group counts, the largest materialized input row
  group, the largest output bucket, and the largest logical tile;
- total and per-array compressed bytes;
- bucket, chunk, shard, and filesystem-object counts;
- coordinate reconstruction error distribution and maximum;
- complete-tile reads;
- common, median, rare-localized, and rare-distributed selected reads;
- logical selected rows, chunks touched, and decoded-row amplification.

Use the evidence to select point chunks/shards, range chunks/shards, and codecs
for the remaining slices and to choose a practically memory-bounded default
worker count. This is an engineering decision, not a comparison gate against
the existing Parquet implementation. There is no fixed numerical pass/fail
benchmark threshold: the build must complete, remain practically memory bounded
and performant, and produce independently valid Exact buckets.

#### Exit criteria

- Exact is independently correct and practically viable on the full Xenium
  example;
- one canonical source row produces exactly one finalized Exact point with its
  deterministic ID, value ID, tile identity, and reconstructable coordinate;
- all nonempty Exact buckets use the common Z2 storage primitive and pass its
  independent validator;
- construction is bounded by source-row-group, shuffled-bucket, logical-tile,
  shard-buffer, and configured-worker limits rather than complete-level size;
- the next level can consume Exact only through the Zarr bucket reader;
- no point-payload or intermediate tile/value-count Parquet artifact exists;
- Z3 leaves cache artifacts, publication, and complete-generation validation to
  their planned later slices.

### Slice Z4: implement fresh Bridge construction

#### Goal

Construct Bridge Zarr buckets directly from Exact Zarr buckets.

Z4 owns fresh value-neutral sampling, deterministic Bridge bucket planning,
bounded Exact-reader reuse, and coordination of the Z2 storage primitive. It
does not reread the canonical source, implement coarser coordinate rebasing,
write compact Parquet indexes, publish a generation, or import an existing
Parquet-backed writer or sampler.

#### Files and dependency boundary

Create:

```text
src/napari_harpy/core/multi_scale_cache_points_zarr/
  sampling.py
  storage/
    reader_cache.py
  writer/
    bridge.py

tests/multi_scale_cache_points_zarr/
  test_sampling.py
  test_reader_cache.py
  test_bridge_writer.py
```

The fresh sampler may import `_splitmix64` from this package's `hashing.py`.
Bridge construction may import only the new package's plans, descriptors,
payload, hashes, reader cache, bucket reader/writer, and storage results. It
must not import `multi_scale_cache_points.sampling`,
`multi_scale_cache_points.writer.bridge`, or any existing writer-support
module. Independent fixed-vector tests freeze the fresh algorithm directly;
they are not cross-backend compatibility tests.

#### Bridge writer API and execution settings

Define a Bridge-specific physical configuration:

```text
_BridgeWriterConfig
  zarr_settings: _ZarrWriteSettings
  max_open_exact_readers: int
```

`max_open_exact_readers` is a positive bound on entered Exact bucket readers.
Do not freeze a numeric default in Z4; the caller supplies the production
candidate value. A value of one is valid and exercises the strictest reader
lifetime bound. Z4 does not compare cached and uncached construction, require a
particular cache hit rate, or select this value through a standalone metadata
benchmark.

The construction entry point is:

```text
_write_bridge_level(
    exact_result,
    plan,
    *,
    staging_root,
    config,
) -> _LevelWriteResult
```

Z4 uses a deterministic sequential output-bucket coordinator. It does not add
Dask or a worker-count setting: the expected memory bound is one materialized
Exact candidate tile, one at-most-Bridge-capacity retained payload, one active
output bucket writer, and a bounded reader-metadata cache. If full-scale
evidence later shows that bucket concurrency is necessary, add it as an
explicit bounded execution policy rather than allowing concurrent writes
implicitly.

Before opening an input store or creating output, require:

- `exact_result` is a nonempty `_LevelWriteResult` for serialized Exact level
  zero;
- the build plan contains Exact level zero followed by Bridge level one;
- Exact and Bridge have identical tile size, grid width, and grid height;
- Exact is uncapped and Bridge has a positive `max_points_per_tile`;
- every Exact descriptor lies inside the planned Exact grid;
- the staged Exact level exists below `staging_root`;
- `levels/level_1` does not exist; Z4 owns creation of that directory;
- `config` contains valid Zarr settings and a positive reader-cache bound.

#### One Exact descriptor is one complete input tile

Do not reproduce the Parquet writer's `_ExactTileDescriptor` or group several
physical manifest rows into one logical tile. The Zarr `_TileDescriptor`
already identifies the complete point interval of one logical tile through its
bucket-local `tile_offset` entry. Zarr chunks and shards are internal physical
units and never become additional tile descriptors.

`_LevelWriteResult.tile_descriptors` already provides one descriptor per
nonempty Exact tile in global `(tile_y, tile_x)` order and rejects duplicate
tile coordinates. Z4 validates that contract and routes those descriptors
directly. Every nonempty Exact tile produces exactly one nonempty Bridge tile
with the same `(tile_x, tile_y)`.

#### Fresh value-neutral sampler

Implement the established versioned policy freshly in `sampling.py`:

```text
SAMPLING_METHOD = "harpy-value-neutral-stratified-splitmix64-v1"
SAMPLING_SEED = 0
SAMPLED_TILE_MICROGRID_EDGE = 16
```

The primary contract is conceptually:

```text
_select_sampled_tile_indices(
    x_rel: float32[N],
    y_rel: float32[N],
    point_id: uint64[N],
    *,
    level,
    tile_x,
    tile_y,
    tile_size,
    target,
) -> int64[min(N, target)]
```

The sampler receives no `value_id`. Membership therefore cannot depend on a
gene/value label, value frequency, or physical value-major input order.
Coordinates and point IDs are candidate facts; the serialized level, logical
tile identity, fixed seed, and versioned hash domains provide deterministic
tie-breaking.

Validate aligned one-dimensional arrays, exact coordinate and point-ID dtypes,
finite coordinates in the inclusive interval `[0, tile_size]`, supported level
and tile-coordinate ranges, and positive `tile_size` and `target`. Point-ID
uniqueness is an accepted immediate-finer-level invariant and is verified by
level acceptance; do not add a second full sort solely to rediscover it inside
every sampler call.

Sampling uses a transient 16-by-16 tile-local microgrid:

1. assign every candidate to one of 256 cells, clamping an exactly represented
   upper tile edge to the final cell;
2. allocate `target` representatives proportionally through integer largest
   remainders, never exceeding a cell's candidate count;
3. resolve equal cell remainders by versioned SplitMix64 cell priority and then
   numeric cell ID;
4. rank candidates inside a cell by versioned SplitMix64 point priority and
   then `point_id`;
5. return exactly `min(N, target)` unique original row indices as a
   C-contiguous `int64` array, ordered by ascending retained `point_id`.

When `N <= target`, retain every candidate and return only the deterministic
point-ID ordering. Membership must be invariant to candidate arrival order.
Fixed tests cover point and cell priority vectors so a later algorithm change
requires a new `SAMPLING_METHOD` identifier.

The resulting `_select_sampled_tile_indices` implementation must carry a
substantial NumPy-style docstring rather than leaving this spatial policy only
in the roadmap. Its `Notes` section must make clear that cache tiles are the
persistent storage/loading units while the 16-by-16 microgrid is transient
sampling state inside one current tile. Include the scaling scheme:

```text
Level    tile edge    microgrid    cell edge
Bridge         512      16 x 16           32
L1           1,024      16 x 16           64
L2           2,048      16 x 16          128
```

Also retain the coarser-level spatial relationship in the code documentation.
After rebasing, four immediate-finer tiles occupy four 8-by-8 quadrants of one
coarser tile's transient 16-by-16 microgrid:

```text
coarser 16 x 16 microgrid
+---------+---------+
| finer   | finer   |
| 8 x 8   | 8 x 8   |
+---------+---------+
| finer   | finer   |
| 8 x 8   | 8 x 8   |
+---------+---------+
```

The docstring must explain the five-stage cell assignment, proportional
allocation, deterministic cell tie-break, deterministic within-cell point
ranking, and final point-ID ordering. It must also state directly that
`value_id` is absent by design, so candidate values cannot influence sampling
membership. Helper docstrings and inline comments should explain the largest-
remainder and priority calculations where they occur rather than requiring a
future developer to reconstruct them from tests.

Do not lose the useful local reasoning currently carried by the reference
sampler's inline comments. Rephrase it for the fresh implementation and retain
it next to the corresponding operations, especially:

- why `target` is a maximum capacity and sparse tiles retain every candidate;
- how point-level microgrid cell IDs become the fixed 256-entry occupancy
  histogram;
- what each `cell_targets[cell_id]` value represents;
- why candidate point IDs and cell IDs are parallel arrays when priorities are
  calculated;
- that `np.lexsort` treats its last key as primary, together with the precise
  cell/priority/point-ID key hierarchy;
- why sorting by cell makes each cell's candidates contiguous;
- how adjacent-cell comparisons find group starts, retaining the small concrete
  example used to make that vectorized step understandable;
- that selected values are original candidate-row positions before their final
  point-ID ordering;
- how integer division and remainder implement proportional allocation without
  floating-point rounding;
- how remaining slots are assigned by remainder, versioned cell priority, and
  final numeric cell-ID tie-breaking;
- which fixed domains, seed, level, tile key, cell ID, and point ID contribute
  to deterministic hash state.

These comments need not be copied verbatim. They must describe the fresh code
accurately, use its final names and dtypes, and remain adjacent to the non-obvious
vectorized or hashing operations they explain. Remove or rewrite any reference
that only makes sense for the Parquet-backed implementation.

#### Deterministic Bridge bucket planning

Derive the planned Bridge bucket count from Bridge's conservative
`point_count_upper_bound` through `_bucket_count_for_level`. Map the Exact tile
coordinate arrays through `_tile_bucket_ids` with that count. Different Exact
tiles may share one Bridge bucket, but one logical tile remains indivisible.

Group only the small descriptors by output `bucket_id`; never materialize or
shuffle point payloads for this planning step. Omit planned bucket IDs that
receive no tiles. Process nonempty output buckets in numeric bucket-ID order and
their descriptors in `(tile_y, tile_x)` order.

The final Bridge count is known without reading the candidate payload:

```text
bridge_tile_count = min(exact_descriptor.n_points,
                        bridge.max_points_per_tile)
```

Use that count to create one `_PlannedTile` per descriptor and a complete
`_BucketPlan` before entering `_BucketWriter`:

```text
_BucketPlan(
    level=1,
    bucket_id=bridge_bucket_id,
    tiles=ordered_planned_tiles,
    settings=config.zarr_settings,
)
```

This gives point arrays their final shape up front. Bridge does not resize point
arrays per tile and does not introduce an intermediate point or count file.

#### Bounded Exact reader reuse

Implement `_BucketReaderCache` as a small generic context-managed LRU of
entered `_BucketReader` instances keyed by `(level, bucket_id)`. Its cache root
and positive maximum size are fixed at construction.

- a hit returns the existing entered reader and marks it most recently used;
- a miss enters one strict reader;
- if the bound is already reached, close and remove the least-recently-used
  reader before admitting the miss;
- failed opens do not remain cached;
- context exit and every exceptional unwind close all cached readers exactly
  once;
- callers never retain a reader beyond the immediate sequential tile operation.

The primary contract is explicit, bounded reader lifetime. Reusing an entered
reader also avoids repeating root-attribute validation and reconstruction of
the Zarr array metadata objects when successive tiles revisit a bucket. Those
operations are normally inexpensive on a local filesystem, so this is a
modest reuse optimization rather than a performance-critical part of Bridge
construction. In particular, a local Zarr reader is not an expensive
database-style connection.

The cache contains initialized readers and array metadata only. It must not
cache decoded point chunks, complete `_PointPayload` values, or let point
memory grow with the level. Keeping it generic allows Z5 to reuse the same
bounded reader-lifetime policy for immediate-finer spatial reads.

#### Per-tile read, sample, and write flow

For every Exact descriptor in the current Bridge bucket:

1. obtain its Exact bucket reader from the bounded cache;
2. call `read_complete(descriptor)` exactly once;
3. pass only `x_rel`, `y_rel`, and `point_id` to the fresh sampler with Bridge's
   level, unchanged tile coordinates, unchanged tile size, and capacity;
4. call `_PointPayload.take(selected_indices)` so all four fields remain
   aligned;
5. require the retained count to equal the `_PlannedTile` count;
6. pass the retained payload and unchanged tile coordinates to
   `_BucketWriter.write_tile`;
7. release candidate and retained payload references before reading the next
   tile.

Bridge owns membership selection only. It must not sort the retained payload by
`(value_id, point_id)` before writing. The common bucket writer remains the
single owner of deterministic value-major ordering and aligned sparse-range
construction. This avoids sorting the selected rows twice and preserves the
same responsibility boundary established by Exact.

Bridge coordinates require no rebasing because Exact and Bridge have identical
tile geometry. Sampling changes membership only: every retained row keeps its
Exact `x_rel`, `y_rel`, `value_id`, and `point_id` values.

#### Result and fast level reconciliation

Bridge needs no `_ExactBucketOutcome` equivalent. Exact's auxiliary global
value totals prove source preservation; Bridge has no corresponding requirement
to preserve every value or source row. Each finalized `_BucketWriteResult`
already contains the generic persisted bucket facts needed to construct the
Bridge `_LevelWriteResult`.

After all readers and writers close, reconcile without rescanning every stored
point during ordinary construction:

- bucket results are nonempty, ordered by `bucket_id`, and belong to level one;
- the output tile-coordinate set equals the Exact input tile-coordinate set;
- every output tile has `min(exact_count, bridge_capacity)` rows;
- the level point count equals the sum of those expected tile counts and does
  not exceed Bridge's planned upper bound;
- every descriptor lies inside the unchanged Bridge grid;
- sparse Exact tiles preserve every candidate and dense tiles fill the capacity
  exactly;
- output paths contain Zarr buckets only and no derived point Parquet.

Membership is selected by unique indices into the complete Exact payload, so
construction verifies subset alignment before handing the payload to the
writer. Focused tests and Gate Z4 independently reopen persisted Bridge and
Exact tiles to prove `point_id` subset membership, unchanged retained fields,
and valid physical ranges. Do not add an exhaustive `_validate_bucket` scan to
ordinary Bridge construction; complete persisted-store validation remains an
opt-in gate here and a mandatory publication check in Z7.

#### Memory, failure, and cleanup contract

At steady state Z4 retains only:

- small level-wide descriptors and bucket assignments;
- at most `max_open_exact_readers` reader metadata/handle objects;
- one complete Exact candidate `_PointPayload`;
- one selected payload capped by Bridge capacity;
- one active `_BucketWriter` and its fixed point/range shard buffers.

It never materializes a complete Bridge bucket's point rows in memory, never
rereads the canonical Parquet source, and never creates Dask scratch.

`_BucketWriter` removes its own partial current bucket after an ordinary write
failure. The reader cache closes all input stores on every exit. Previously
finalized Bridge buckets may remain inside the isolated unpublished staging
generation; the later top-level builder owns removal of that complete failed
generation rather than Z4 attempting partial repair or resume. `COMPLETED` is
never created by Z4.

#### Focused tests

- fixed point-priority and cell-priority vectors, proportional allocation,
  upper-edge cell assignment, controlled priority collisions, and exact
  capacity vectors;
- sparse, empty sampler-input, and over-capacity candidate arrays;
- sampler membership invariant to candidate permutation and unavailable to
  `value_id` by API construction;
- invalid coordinate, dtype, shape, capacity, level, and tile identities;
- reader-cache hits, LRU eviction at bounds one and greater than one, failed
  opens, exceptional unwind, and deterministic closure;
- tiny staged Exact buckets with sparse and dense tiles crossing point chunks
  and shards, built directly through the common Z2 primitive rather than the
  canonical source reader;
- Bridge and Exact geometry equality, unchanged retained fields, exact sparse
  pass-through, dense capacity, and deterministic membership;
- exactly one Bridge descriptor per nonempty Exact descriptor and no
  Parquet-style tile-shard grouping;
- multiple Exact input buckets routed into multiple Bridge output buckets,
  including empty planned bucket IDs and reader-cache reuse/eviction;
- persisted Bridge buckets passing `_validate_bucket` and independently
  matching the corresponding Exact `point_id` subsets;
- preexisting output, invalid level transitions, descriptor/count mismatch,
  injected read, sampling, and write failures, current partial-bucket cleanup,
  reader closure, and absence of `COMPLETED`.

Tests use small chunks, shards, capacities, and reader-cache bounds so every
physical and lifecycle boundary is exercised without depending on the
Parquet-backed implementation.

#### Gate Z4: full-Xenium Bridge evaluation

Provide an opt-in Bridge evaluation separate from unit tests. Build or retain a
current-tree full-Xenium Exact staging level, construct Bridge with the
production-candidate settings, and measure Bridge separately from Exact.

Record:

- Bridge construction time and peak incremental RSS;
- expected and observed point, tile, bucket, range, shard, and filesystem-object
  counts;
- largest Exact candidate tile and largest Bridge output tile;
- configured reader-cache bound and confirmation that the number of entered
  readers remains within it;
- compressed bytes by array and total Bridge output size.

Reopen every Bridge bucket through `_validate_bucket`. For every logical tile,
read the corresponding Exact and Bridge payloads and independently require:

- the Bridge count is `min(exact_count, bridge_capacity)`;
- Bridge `point_id` values are a unique subset of that Exact tile;
- retained coordinates and values agree exactly by `point_id`;
- recomputing the fresh sampler from Exact produces the persisted Bridge
  membership;
- no source Parquet read occurs during the Bridge phase and no point Parquet is
  written.

This is a current-format engineering gate, not a comparison against the old
Parquet-derived cache and not a fixed numerical benchmark threshold. It should
complete with practical time, bounded memory, bounded open readers, and fully
valid persisted membership before Z5 uses Bridge as its immediate-finer input.
There is no separate cached-versus-uncached run, cache-hit target, or exhaustive
metadata-cache benchmark: metadata reuse is not a Gate Z4 acceptance criterion.

#### Exit criteria

- Bridge reads and writes only Zarr point payloads;
- the fresh sampler is deterministic, value-neutral, versioned, and
  independently fixed by vectors and logical tests;
- each complete Exact tile is sampled exactly once and produces one same-geometry
  Bridge tile;
- Bridge membership, retained-field alignment, capacity, and physical storage
  semantics are independently verified;
- construction memory and open readers are bounded by explicit contracts rather
  than complete-level size;
- every reader and writer closes deterministically on success and failure;
- the full-Xenium Bridge gate is practically viable and independently correct;
- Z5 can treat Bridge like any other immediate-finer Zarr level.

### Slice Z5: implement fresh spatial pyramid and overview

#### Goal

Build every coarser level through the uniform Zarr path.

#### Work

- group one through four immediate-finer tile descriptors into each coarser
  tile;
- read complete finer tiles through the Zarr reader;
- implement fresh coordinate-rebasing helpers inside the new package;
- rebase coordinates into the coarser frame;
- concatenate candidates and sample once at the coarser capacity;
- leave retained-row `(value_id, point_id)` ordering to the common bucket
  writer after membership selection;
- plan and write every spatial bucket with the common bucket primitive;
- use the same flow for the terminal overview;
- build levels strictly from their immediate predecessor;
- enforce per-tile capacities, nested `point_id` membership, and the overview
  budget;
- never reread the canonical source.

#### Focused tests

- two- and multi-level pyramids;
- sparse edge regions and every rebasing quadrant;
- deterministic membership across repeated builds;
- each level is a subset of its immediate predecessor;
- all capacities and overview budget;
- uniform array schemas and backend versions across all levels;
- multiple buckets and bounded handle lifetime;
- injected failure at each level.

#### Gate Z5

Run a complete small pyramid and the new package's full focused logical and
storage tests. Inspect the staged directory and prove that every point payload
under every level is Zarr.

#### Exit criteria

- all planned levels are written by the fresh package;
- the construction path contains no point-level Parquet reader or writer;
- Exact, Bridge, spatial, and overview share one storage contract.

### Slice Z6: freeze final cache-format and artifact contracts

#### Goal

Define the complete hybrid cache contract and write its compact indexes.

#### Work

- implement exact metadata and Arrow schemas in `cache_format.py`;
- freeze the payload/backend identifiers and selected physical settings after
  the Z3 evidence;
- write `values.parquet` from the canonical validated value table;
- write `manifest.parquet` from all level descriptors;
- derive `tile_value_counts.parquet` in a bounded pass over finalized bucket
  range arrays;
- reject duplicate logical tiles and tile/value keys;
- reconcile artifact totals with bucket attributes and offsets;
- write deterministic `metadata.json` containing source signature, build plan,
  level summaries, artifact paths, physical settings, and generation identity;
- do not write `COMPLETED`.

The first schema version is chosen only after verifying that no incompatible
completed format used the same identifier. Metadata includes a required backend
identifier such as:

```text
harpy-zarr-v3-bucket-sparse-value-ranges-v1
```

#### Focused tests

- exact Parquet schemas, nullability, sort order, and deterministic bytes where
  the writer guarantees them;
- metadata roundtrip and deterministic JSON;
- normalized contained paths;
- manifest/bucket coordinate and count reconciliation;
- tile/value-count/range equality;
- unsupported versions and physical settings fail closed.

#### Exit criteria

- one staged generation is self-describing without writer objects;
- all compact indexes can be regenerated from validated source facts,
  descriptors, and finalized Zarr stores;
- no format decision needed by independent validation remains implicit.

### Slice Z7: implement independent staged validation

#### Goal

Validate the publication-critical structure and accounting of a complete
generation by reopening it from disk and trusting no in-memory writer result
for those persisted facts. Do not replay the full source-equivalence acceptance
gate during normal cache creation.

#### Performance evidence and validation tiers

The completed full-Xenium Z3 gate measured 44.09 seconds for construction and
166.03 seconds for exhaustive verification of 136,578,750 points. The latter
reopened every bucket with `_validate_bucket`, read every logical tile again,
and rescanned every canonical Parquet source row to compare point IDs, values,
and reconstructed coordinates. It was approximately 3.8 times the construction
time and was predominantly single-threaded. This is valuable release and format
evidence, but it is not an acceptable unconditional addition to normal cache
creation.

Z7 therefore has two deliberately separate validation tiers:

1. **Normal publication validation** is mandatory in the Z8 build flow. It
   reopens and validates metadata, hierarchy, array layouts, compact pointer and
   index arrays, artifact schemas, paths, counts, versions, and cross-artifact
   accounting. It must not read every `location`, point-level `value_id`, and
   `point_id` row, construct a global point-ID bitmap, or rescan canonical source
   content.
2. **Exhaustive acceptance/diagnostic validation** is opt-in. It may run the
   complete `_validate_bucket` scan, prove global Exact point-ID coverage with a
   bounded external structure, check cross-level point-ID membership, and
   compare finalized values and reconstructed coordinates with the canonical
   source. Use it for format or algorithm changes, release qualification,
   benchmarks, and investigation of suspected corruption, not every cache
   build.

The tiers share low-level parsers and structural checks where useful, but they
have distinct entry points so the exhaustive path cannot accidentally become a
normal publication cost.

#### Work

Validate, in bounded batches:

- enumerate the bucket stores referenced by the manifest and reopen every
  expected physical bucket through a metadata/layout validation path that does
  not decode the complete point payload;
- metadata, backend version, build plan, and artifact schemas;
- exact equality between manifest bucket paths and physical stores;
- Zarr v3 hierarchy, attributes, shapes, dtypes, chunks, shards, and codecs;
- bucket tile coordinates, offsets, and manifest counts;
- compact sparse range keys, ordering, counts, and pointer bounds without
  rereading point-level values during normal validation;
- equality between range keys/counts and `tile_value_counts.parquet`;
- Exact per-value totals and `values.parquet`;
- level geometry, capacities, and overview budget;
- absence of unreferenced stores, unexpected point Parquet files,
  construction scratch, and premature `COMPLETED`.

The optional exhaustive entry point additionally owns:

- complete bucket payload validation, including sparse-range agreement with
  point-level values;
- Exact point-ID completeness and uniqueness;
- immediate-coarser point-ID subset membership;
- point-level coordinate validity and source-coordinate reconstruction
  tolerance;
- canonical source-row value and coordinate equivalence when requested.

Validation must not load a complete level or all Exact IDs into one Python
collection. Use bounded scans, sorted merge checks, or temporary external data
structures where necessary in the exhaustive tier. Normal publication
validation must remain bounded by compact metadata/index batches and must not
perform a complete point-payload or canonical-source scan.

#### Focused tests

- valid Exact-only and multilevel generations;
- missing/extra buckets and artifacts;
- corrupted metadata and backend versions;
- malformed arrays, offsets, ranges, and attributes;
- manifest/bucket and range/index mismatches;
- normal-tier capacity, geometry, and overview violations detectable from
  compact persisted facts;
- exhaustive-tier point-ID duplication, loss, nesting, coordinate, and
  point/range-semantic violations;
- proof that normal publication validation neither opens canonical point
  Parquet nor reads complete Zarr point arrays;
- explicit invocation tests proving exhaustive checks cannot run implicitly
  through the normal Z8 publication path.

#### Exit criteria

- corruption at every storage or cross-artifact layer fails closed in the
  validation tier that owns that semantic check;
- normal publication validation is memory bounded and avoids complete point and
  source scans;
- exhaustive validation remains memory bounded and opt-in;
- a successful staging result is safe to publish after the final source guard.

### Slice Z8: compose the guarded builder and publication

#### Goal

Expose one isolated candidate builder that creates only complete Zarr-backed
generations.

#### Required flow

```text
ValidatedPointsSource
  -> fresh metadata-only source guard
  -> fresh Zarr-cache build plan
  -> unique sibling staging generation
  -> Exact
  -> Bridge
  -> all spatial/overview levels
  -> final artifacts and metadata
  -> independent staged validation
  -> final fresh source guard
  -> write COMPLETED
  -> atomic local publication
```

#### Work

- own staging creation, cleanup, replacement, and publication in the new
  package;
- preserve an existing completed candidate generation on every failure;
- close all Dask tasks, Zarr stores, memory maps, and file handles before
  validation and rename;
- make `COMPLETED` the final staged write;
- reject incomplete generations when opening;
- expose no public backend selector and do not call the existing builder;
- treat derived cache data as regenerable while preserving product-quality
  failure and publication guarantees.

#### Focused tests

- first build and successful replacement;
- failures before staging and during every major construction phase;
- failure during artifacts, validation, final guard, completion, and rename;
- preservation of an existing generation;
- cleanup of incomplete stores and Dask scratch;
- all handles closed before publication;
- canonical source remains unchanged.

#### Exit criteria

- the isolated output path is absent or contains one complete independently
  validated generation;
- no mixed or incomplete payload is observable;
- the end-to-end builder never imports or invokes the existing writer package.

### Slice Z9: implement the acceptance reader and Xenium evaluation

#### Goal

Measure whether the physical design provides useful selected-value access while
keeping normal all-values navigation practical.

This slice implements a small backend-level reader, not the complete Phase 2
napari store.

#### Reader operations

```text
read_complete_tile(level, tile_x, tile_y)
read_selected_values(level, tile_x, tile_y, value_ids, include_point_id)
read_selected_viewport(level, tile_bounds, value_ids, include_point_id)
```

The reader must:

- use `tile_value_counts.parquet` to prune zero-count tiles;
- group positive visible tiles by bucket and open each bucket once per request;
- use sparse range records instead of scanning point-level values;
- read only intersecting coordinate chunks and optional ID chunks;
- coalesce adjacent ranges where practical;
- distinguish metadata/index work from point-payload work;
- report logical rows, positive tiles, buckets, chunks touched, and decoded-row
  estimates.

#### Full-Xenium scenarios

Measure at Exact, Bridge, representative spatial levels, and overview:

- dense and average complete tiles;
- all-values viewports at several zoom levels;
- common, median, rare-localized, and rare-distributed values;
- several selected values with adjacent and separated ranges;
- repeated selection changes with cold and warm caches;
- panning with overlapping tiles, buckets, and chunks.

Record:

- complete build time and peak RSS;
- total bytes and filesystem-object count;
- complete-tile and viewport latency;
- selected-value latency;
- logical selected rows;
- complete positive-tile rows;
- positive visible tiles and buckets opened;
- chunks touched and estimated decoded rows;
- tile- and chunk-read amplification.

#### Bucket-target decision

Retain `TARGET_POINTS_PER_BUCKET = 2_000_000` through construction and use its
completed Xenium measurements as the baseline physical configuration. Evaluate
`10_000_000` as the single leading alternative rather than opening an unbounded
parameter sweep. The decision must account for both sides of the tradeoff:

- complete and selected viewport latency, especially the number of distinct
  bucket stores opened for common and rare-distributed values;
- metadata/open-handle work per request;
- Exact and multilevel construction peak RSS with the configured worker count;
- largest materialized shuffled bucket and finalizer duration;
- total object count and storage bytes;
- unchanged inner-chunk decoded-row amplification.

Choose ten million only if the reduced store-open and metadata work is material
for realistic navigation and its larger construction unit remains practically
memory bounded. Otherwise retain two million. Record the chosen target and
evidence as a versioned construction policy before the format is proposed for
Phase 2.

#### Acceptance decision

Correctness is mandatory. Review build reliability, bounded memory, construction
speed, storage behavior, object count, complete reads, and selected reads as one
engineering decision without fixed numerical thresholds and without requiring a
Parquet comparison run.

If adopted, record the format and physical settings as the proposed Phase 2
input. If not adopted, stop work on this package and retain the existing cache
implementation. Do not add a fallback path.

#### Exit criteria

- direct sparse-range lookup is demonstrated at full scale;
- its useful and non-useful cases are documented honestly;
- all-values access remains practical for the planned viewer;
- the candidate architecture has an explicit adoption decision.

### Slice Z10: architecture-adoption decision

#### Goal

Conclude the isolated architecture evaluation without blurring the two
implementations.

#### If adopted

- update the higher-level transcript-visualization roadmaps with measured Zarr
  behavior;
- choose the future public entrypoint and final cache directory name;
- document the supported schema and dependency bounds;
- plan removal or archival of the existing derived-cache implementation as a
  separate integration task, not inside this roadmap;
- keep only one public backend after integration.

#### If not adopted

- document the measured reason;
- remove or archive `multi_scale_cache_points_zarr` as a non-adopted candidate;
- retain the existing package without adding compatibility code.

#### Exit criteria

- there is one explicit project direction;
- no runtime backend selector or automatic fallback exists;
- documentation distinguishes historical/reference code from the supported
  implementation.

## Test strategy

Normal development uses only focused tests under
`tests/multi_scale_cache_points_zarr`. The existing package's focused tests may
be run at broad gates to confirm that the new sibling package did not disturb
it, but the new tests do not compare cache artifacts or performance.

Test layers are:

```text
fresh logical logic
  planning, hashes, sampling, rebasing

Zarr primitive
  arrays, offsets, ranges, chunks, shards, corruption

level writers
  Exact, Bridge, spatial membership and counts

artifacts
  metadata, manifest, values, tile/value counts

staged validation
  independent cross-artifact reconciliation

builder
  guards, cleanup, completion, publication

acceptance reader
  complete and selected physical reads
```

Do not assert codec-compressed byte counts in normal unit tests. Assert declared
configuration, logical content, shapes, and invariants. Performance and object
counts belong to opt-in benchmark scripts.

## Memory and concurrency rules

- Each Exact source task materializes at most one validated physical row group;
  each Exact finalizer may materialize one complete shuffled bucket partition,
  never a complete level.
- Each finalizer owns one Zarr store; concurrent writers never target the same
  store or shard.
- `location` is assembled and written in bounded batches; avoid a second
  full-bucket `(N, 2)` allocation.
- Bridge holds at most one complete Exact candidate tile plus bounded output
  buffers.
- Spatial construction holds at most the one-through-four immediate-finer tiles
  needed for one coarser tile plus bounded output buffers.
- Point arrays use final planned shapes. Do not resize them once per tile or
  point.
- Range arrays may grow only in coarse shard-aligned blocks and are trimmed at
  finalization.
- Small consecutive tiles and their range records are buffered across tile
  boundaries so partially filled point or range shards are not repeatedly
  rewritten.
- Readers reuse handles within a request/context and close them deterministically.
- All writers and readers close before staged validation or publication.

## Failure and publication rules

- The cache data are derived and regenerable; the implementation is held to the
  same professional quality standard as other product code.
- All writes occur under a unique sibling staging generation.
- A failure in one bucket invalidates the whole staging generation.
- This architecture deliberately does not resume or repair partial stores;
  failed staging generations are rebuilt from the canonical source.
- Independent validation reopens all required artifacts after writers close.
- `COMPLETED` is absent throughout construction and validation.
- The final source guard precedes `COMPLETED` and publication.
- Readers reject missing completion, unsupported versions, missing stores, and
  inconsistent indexes.
- Existing completed candidate generations survive failed replacements.

## Risks and mitigations

### Duplication drift

Fresh planning and sampling code may diverge unintentionally from the intended
semantics. Mitigate with explicit invariant tests and small golden vectors, not
imports from the existing writer package. Differences are acceptable only when
documented as deliberate Zarr-architecture decisions.

### Small selected reads

A one-point interval still decompresses a complete chunk. Measure read
amplification and coalesce ranges; do not create a chunk per gene.

### All-values regression

Value-major ordering remains sequential within a complete tile, but it may
alter compression and access behavior. Benchmark complete tiles and viewports
before adoption.

### Filesystem-object growth

Use independent bucket stores with Zarr v3 sharding. Measure actual files and
directories at full scale rather than inferring object count from logical
chunks. The initial 4,096-row unsharded Exact point arrays would create roughly
100,000 chunk files before range arrays and coarser levels; 131,072-row point
shards reduce the corresponding physical point objects to roughly 3,300 after
allowing for the 69 bucket boundaries. Range arrays are sharded for the same
reason. These estimates motivate the initial layout but do not replace the
full-Xenium measurement.

### Duplicate indexes

`tile_value_counts.parquet` and Zarr sparse ranges intentionally duplicate
tile/value keys for opposite query directions. Derive the Parquet index from
finalized ranges and independently validate equality.

### Zarr API and version stability

Keep Zarr calls inside the new storage package, use explicit v3 checks, and
defer dependency bounds until the working primitive and Xenium build establish
requirements.

### Nested Zarr stores inside SpatialData

The derived stores are not SpatialData elements. Use a distinct Harpy-owned
path, validate containment, and verify that SpatialData operations ignore the
isolated cache directory.

### Sparse values distributed everywhere

`tile_value_counts.parquet` cannot prune a rare value present in every visible
tile. Viewport clipping, pyramid selection, sparse ranges, and chunk caching
still bound work, but Z9 must measure this case.

## Definition of done

The production-candidate evaluation is complete when:

- `multi_scale_cache_points_zarr` builds every planned level without importing
  existing writer code;
- every point payload is stored in the same bucket-local Zarr v3 schema;
- Exact contains every source point exactly once;
- Bridge and spatial membership, nesting, capacities, coordinates, and overview
  budget are correct;
- every manifest tile resolves to one verified Zarr interval;
- every nonempty tile/value key resolves to one verified sparse range;
- final Parquet indexes reconcile with all Zarr stores;
- independent staged validation fails closed on corruption;
- construction is bounded, failure-safe, and atomically published;
- full-Xenium build and read measurements are recorded;
- the project explicitly adopts or does not adopt the candidate architecture.

## Immediate next slice

Z0, Z1, and Z2 are resolved. Implement Z3 against the standalone bucket
primitive. Build Exact directly from the validated canonical source into Zarr;
do not modify or adapt the existing Exact, Bridge, spatial, or writer-support
modules, and do not introduce a transitional Parquet point reader.
