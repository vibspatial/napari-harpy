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

multi_scale_cache_points.signature
  source-signature facts and fresh metadata-only checks, where suitable
```

If importing a private source helper would pull in cache-writer assumptions, the
new package duplicates the small helper instead. The reuse boundary must remain
obvious from imports.

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

The root attributes contain only versioned physical facts:

```text
payload_schema_version = 1
level                  = <JSON integer>
bucket_id              = <JSON integer>
tile_count             = <JSON integer>
point_count            = <JSON integer>
range_count            = <JSON integer>
point_order            = ["tile_y", "tile_x", "value_id", "point_id"]
coordinate_encoding    = "tile-relative-xy-float32-v1"
point_chunk_rows       = <JSON integer>
point_shard_rows       = <JSON integer>
range_chunk_rows       = <JSON integer>
range_shard_rows       = <JSON integer>
codec_id               = "zstd-v1"
```

These exact keys and value encodings are part of payload schema version 1.
NumPy scalar objects are not written as attributes. Semantic cache metadata
remains in `metadata.json`.

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
ranges in the same inner chunk do not request that chunk twice.

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
validated source files
  -> one input partition per validated physical file
  -> annotate tile_x, tile_y, x_rel, y_rel, value_id, point_id, bucket_id
  -> disk shuffle by bucket_id
  -> one complete materialized partition per nonempty bucket
  -> stable sort by (tile_y, tile_x, value_id, point_id)
  -> write one independent Zarr store
```

Each finalizer owns exactly one store. It derives tile identities, offsets, and
sparse ranges from the sorted bucket and returns compact descriptors. It writes
`location` in bounded batches and does not allocate a second full-bucket
`(N, 2)` array.

### Bridge

```text
Exact tile descriptor
  -> read complete Exact tile from Zarr through tile_offset
  -> fresh value-neutral sampler
  -> reorder selected rows by (value_id, point_id)
  -> append to planned Bridge bucket
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
  -> reorder selected rows by (value_id, point_id)
  -> append to planned coarser bucket
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
- reuse only canonical source models, validation, and source-signature facts;
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

This is a construction policy, not part of the logical tile identity. Z3 may
change it from Xenium evidence before the cache format is frozen.

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
than accepting arbitrary codec objects or parameters. Both the reader and
validator verify the public array chunk, shard, dtype, and codec properties
against the declared settings.

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
  -> deduplicate and group touched chunk IDs
  -> read every touched inner chunk once
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

Upper coordinate bounds remain a level-writer responsibility because neither
`_PointPayload` nor `_BucketPlan` knows the tile size. Exact, Bridge, and spatial
writers will enforce `x_rel <= tile_width` and `y_rel <= tile_height` before
calling the bucket writer. Cross-artifact validation repeats those checks once
the build plan and metadata are available.

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

#### Work

- implement one Dask input partition per validated source file;
- assign deterministic source-row `point_id` values from validated file
  offsets;
- map normalized labels to canonical `value_id` values;
- calculate tile coordinates, tile-local coordinates, and bucket IDs;
- disk-shuffle by bucket ID;
- stable-sort each materialized bucket by
  `(tile_y, tile_x, value_id, point_id)`;
- derive a `BucketPlan` and write one independent Zarr store per nonempty
  bucket;
- return one descriptor per nonempty logical tile;
- reconcile Exact point count, unique IDs, bounds, tile totals, and per-value
  totals;
- clean Dask scratch independently of staging output;
- make any finalizer failure invalidate the staging generation.

#### Focused tests

- multiple source files and source row groups;
- points for one tile arriving from several input partitions;
- sparse and dense tiles;
- negative/nonzero origins and coordinate tolerance;
- exact one-to-one source-row/`point_id` coverage;
- canonical value IDs and value totals;
- deterministic output under changed partition arrival order;
- several independent bucket finalizers;
- injected shuffle/finalizer failures and cleanup.

#### Gate Z3: full-Xenium Exact evaluation

Build the complete 136,578,750-point Exact level and record:

- build time and peak RSS;
- total and per-array compressed bytes;
- bucket, chunk, shard, and filesystem-object counts;
- all logical tile counts and complete point-ID coverage;
- coordinate reconstruction error;
- complete-tile reads;
- common, median, rare-localized, and rare-distributed selected reads;
- logical selected rows, chunks touched, and decoded-row amplification.

Use the evidence to select point chunks/shards, range chunks/shards, and codecs
for the remaining slices. This is an engineering decision, not a comparison
gate against the existing Parquet implementation.

#### Exit criteria

- Exact is independently correct and practically viable on Xenium;
- the next level can consume Exact only through the Zarr bucket reader;
- no point-payload Parquet artifact exists.

### Slice Z4: implement fresh Bridge construction

#### Goal

Construct Bridge Zarr buckets directly from Exact Zarr buckets.

#### Work

- group Exact descriptors by logical tile and deterministic Bridge bucket;
- read one complete Exact tile through the new bucket reader;
- implement the fresh value-neutral sampler inside the new package;
- apply it exactly once per logical tile;
- preserve sparse tiles and cap dense tiles at the Bridge capacity;
- sort retained rows by `(value_id, point_id)` only after membership selection;
- plan and write Bridge buckets with the same storage primitive;
- retain at most one complete candidate tile plus bounded output buffers;
- validate Bridge counts, tile identities, ranges, and `point_id` membership;
- never reread the canonical source.

#### Focused tests

- sparse and over-capacity Exact tiles;
- sampler allocation, spatial coverage, tie-breaking, and capacity vectors;
- deterministic sampling and independence from value labels;
- Bridge and Exact geometry equality;
- unchanged tile-local coordinates;
- one Bridge descriptor per nonempty Exact tile;
- Bridge `point_id` subset membership;
- multiple input/output buckets and handle reuse;
- injected read/write failures and cleanup.

#### Exit criteria

- Bridge reads and writes only Zarr point payloads;
- Bridge membership and capacity semantics are independently verified;
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
- sort retained rows by `(value_id, point_id)` after membership selection;
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

Validate a complete generation by reopening it from disk and trusting no
in-memory writer results.

#### Work

Validate, in bounded batches:

- metadata, backend version, build plan, and artifact schemas;
- exact equality between manifest bucket paths and physical stores;
- Zarr v3 hierarchy, attributes, shapes, dtypes, chunks, shards, and codecs;
- bucket tile coordinates, offsets, and manifest counts;
- sparse range ordering, coverage, and point-level value agreement;
- equality between range keys/counts and `tile_value_counts.parquet`;
- Exact per-value totals and `values.parquet`;
- Exact point-ID completeness;
- immediate-coarser subset membership;
- coordinate validity and reconstruction tolerance;
- level geometry, capacities, and overview budget;
- absence of unreferenced stores, unexpected point Parquet files,
  construction scratch, and premature `COMPLETED`.

Validation must not load a complete level or all Exact IDs into one Python
collection. Use bounded scans, sorted merge checks, or temporary external data
structures where necessary.

#### Focused tests

- valid Exact-only and multilevel generations;
- missing/extra buckets and artifacts;
- corrupted metadata and backend versions;
- malformed arrays, offsets, ranges, and attributes;
- manifest/bucket and range/index mismatches;
- point-ID duplication, loss, and nesting violations;
- capacity, coordinate, and overview violations;
- proof that the validator performs no canonical-source content rescan.

#### Exit criteria

- corruption at every storage or cross-artifact layer fails closed;
- validation is memory-bounded;
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

- Exact may materialize one complete Dask bucket partition, not a complete
  level.
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
