# Independent Zarr multiscale points cache

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
fresh models, planning, writers, readers, validation, catalog construction, and
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
- the root `publication_state` changes from `"staging"` to `"complete"` only
  after the final source guard and successful staged validation.

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
8. Independently validate the complete Zarr hierarchy, catalog arrays, and
   cross-index accounting before publication.
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
  mutable point-payload Zarr array. Fixed final catalog arrays are permitted.
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
  cache_format.py           # root attributes and catalog-array contracts

  storage/
    __init__.py
    _schema.py              # private bucket constants and Zarr codec/layout map
    models.py               # bucket plans/results and physical settings
    bucket_writer.py        # create and finalize one Zarr bucket
    bucket_reader.py        # complete and selected tile reads
    bucket_validation.py    # independent structural bucket validation
    catalog_writer.py       # fixed final cache-wide Zarr catalog arrays
    catalog_reader.py       # strict root metadata and catalog-array reads

  writer/
    __init__.py
    exact.py                # canonical source -> Exact Zarr buckets
    bridge.py               # Exact Zarr -> Bridge Zarr
    spatial.py              # finer Zarr -> coarser Zarr
    catalog.py              # source/results -> root attributes and catalog
    staging_validation.py   # compact publication hierarchy and cross-index validation
    build.py                # guards, staging, composition, publication
```

Developer-only exhaustive acceptance tooling lives outside the installed
package:

```text
scripts/
  validate_multi_scale_cache_points_zarr_exhaustive.py
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
               catalog and validation
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
- final Zarr catalog construction;
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
        zarr.json                            # root group + semantic attributes
        values/
          zarr.json
          n_points/                          # exact source counts by value_id
        manifest/
          zarr.json
          level_indptr/
          bucket_id/
          bucket_tile_index/
          tile_x/
          tile_y/
          n_points/
        value_tiles/
          zarr.json
          indptr/                            # (level, value_id) -> entries
          manifest_index/                    # entry -> manifest row
          n_points/                          # entry count
        levels/
          zarr.json
          level_0/
            zarr.json
            bucket-000.zarr/
            bucket-001.zarr/
          level_1/
            zarr.json
            bucket-000.zarr/
          level_n/
            zarr.json
            bucket-000.zarr/
```

The final cache directory name is intentionally deferred until the public
integration decision. During isolated development it must not collide with the
existing derived-cache path. The cache root is one Zarr v3 group whose
attributes contain the small semantic contract. `values`, `manifest`, and
`value_tiles` are ordinary child groups containing typed numeric arrays.

Every `bucket-<id>.zarr` remains independently openable as a Zarr v3
`LocalStore`, while its `zarr.json` also makes it a child group at that path when
the complete cache root is opened as one hierarchy. Ancestor `levels` and
`level_<n>` group metadata are written once; independent bucket writers never
mutate shared parent metadata.

Parquet is used only for the canonical source. No derived-cache metadata,
catalog, count index, point payload, or publication marker is persisted outside
the Zarr hierarchy. Publication state is a root Zarr attribute, so generic Zarr
tools see no unrelated root sidecar.

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
coordinate arrays to `location[:, 0:2]`. Arrow remains the validated canonical
source-table boundary but is neither the point interchange boundary nor a
persisted derived-cache catalog format.

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

Descriptors are construction results and later become rows in the typed
`manifest` catalog arrays. They must
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
point_row_order        = ["tile_y", "tile_x", "value_id", "point_id"]
coordinate_encoding    = "tile-relative-xy-float32-v1"
codec_id               = "zstd-v1"
```

These exact keys and value encodings are part of payload schema version 1.
NumPy scalar objects are not written as attributes. Cache-wide semantic
metadata lives in the cache root group's Zarr v3 attributes, not in an external
JSON sidecar.

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
- the interval length equals the descriptor and manifest-catalog `n_points`;
- the manifest-catalog coordinates for bucket-local index `i` equal `tile_x[i]`
  and `tile_y[i]`.

The coordinate arrays deliberately duplicate the manifest-catalog coordinates.
This small duplication lets independent validation detect a permuted or
mislabeled catalog row.

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
- range counts reconcile with the cache-wide `value_tiles` CSR arrays.

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

The Z3 through Z5 full-Xenium gates retained these values as the initial
production defaults. Z6 records the actual common values in metadata and
requires every bucket in a generation to agree; the numeric sizes remain a
tunable physical profile rather than part of the logical cache-schema identity.
At these values, the three aligned point shard buffers occupy approximately 2.5
MiB in aggregate, and the three aligned range shard buffers occupy approximately
another 2.5 MiB.

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

## Cache-wide Zarr catalog

The cache root is a Zarr v3 group. Small, generation-wide semantic facts live in
its JSON-serializable attributes. Potentially large or sliceable indexes remain
typed Zarr arrays rather than being embedded in attributes or written as JSON
tables.

This follows the useful Xenium distinction: versions, dataset identity, value
labels, and grid summaries are attributes; scalable point, offset, and sparse
index structures are arrays. A separate `manifest.json` would still require a
complete verbose JSON parse and would not solve the much larger value-to-tile
index.

### Values

Canonical value labels are stored once in root attributes as `value_names`, in
ascending implicit `value_id` order. With `G` values:

```text
values/n_points       shape=(G,) dtype=uint64
```

`value_id` is the zero-based array position and is not redundantly stored.
`n_points` contains exact canonical-source totals. Labels are unique, nonempty,
canonically normalized, and ordered by ascending UTF-8 bytes; counts are
positive and sum to the Exact point total.

### Manifest catalog

For `L` levels and `T` nonempty logical tiles over all levels:

```text
manifest/level_indptr         shape=(L + 1,) dtype=uint64
manifest/bucket_id            shape=(T,)     dtype=uint32
manifest/bucket_tile_index    shape=(T,)     dtype=uint32
manifest/tile_x               shape=(T,)     dtype=uint32
manifest/tile_y               shape=(T,)     dtype=uint32
manifest/n_points             shape=(T,)     dtype=uint64
```

The five length-`T` arrays are parallel. Their shared zero-based array position
is the global `manifest_index`; it is not stored as a separate manifest array.
`manifest/level_indptr` is a level-boundary pointer array, not a sixth
row-aligned column. For example:

```text
implicit       bucket_   bucket_tile_   tile_   tile_   n_
manifest_index id        index          x       y       points
----------------------------------------------------------------
0              1         0              0       0       120
1              0         0              1       0        85
2              1         1              2       0       103
```

Manifest rows are globally ordered by `(level, tile_y, tile_x)`.
`level_indptr[level:level + 2]` gives the half-open manifest-row interval for
one level, so the non-negative serialized level need not be repeated per row.
Within that interval, coordinate pairs are unique and ordered by
`(tile_y, tile_x)`.

The aligned arrays map a logical tile to its bucket and bucket-local ordinal.
They never expose point offsets, chunks, or shards. `bucket_path` is not stored
as a string array: it remains the canonical derivation from serialized `level`
and `bucket_id`. Validation requires every catalog address and coordinate to
agree with the bucket's compact tile arrays.

### Value-to-tile CSR index

The bucket sparse ranges answer `tile -> value -> point rows`. Runtime planning
also benefits from the inverse direction, `(level, value) -> positive tiles`,
especially for selected-value count estimates before point chunks are read.

For `M` nonempty `(level, value_id, tile)` combinations:

```text
value_tiles/indptr            shape=(L, G + 1)   dtype=uint64
value_tiles/manifest_index    shape=(M,)         dtype=uint64
value_tiles/n_points          shape=(M,)         dtype=uint64
```

For one level and value, entries are:

```text
start = value_tiles/indptr[level, value_id]
stop  = value_tiles/indptr[level, value_id + 1]

manifest rows = value_tiles/manifest_index[start:stop]
counts        = value_tiles/n_points[start:stop]
```

Every level row is nondecreasing. The final pointer of level `l` equals the
first pointer of level `l + 1`, because the aligned entry arrays are globally
ordered by `(level, value_id, manifest_index)`. The first pointer is zero and
the final pointer of the final level is `M`.

Each segment is strictly ordered by its referenced manifest row. Every
`manifest_index` points into the matching level interval, and every count is
positive. This normalized representation does not repeat level, value, tile
coordinates, or bucket paths on every sparse row.

This cache-wide index does not replace the bucket-local `ranges/tile_indptr`,
and it does not point directly into the point arrays. Reuse the three tiles from
the bucket example above and assume that they are manifest rows `0`, `1`, and
`2` of level `0`:

```text
manifest row 0 / tile 0: value 0 -> 10 points, value 2 -> 3 points
manifest row 1 / tile 1: value 1 ->  8 points, value 2 -> 4 points
manifest row 2 / tile 2: value 0 ->  6 points
```

Inverting those tile-local ranges gives:

```text
value 0 -> manifest rows [0, 2], counts [10, 6]
value 1 -> manifest row  [1],    count  [8]
value 2 -> manifest rows [0, 1], counts [3, 4]

value_tiles/manifest_index = [0,  2, 1, 0, 1]
value_tiles/n_points       = [10, 6, 8, 3, 4]
value_tiles/indptr[0]      = [0,  2, 3, 5]
```

For example, value `2` at level `0` selects entries `3:5`, which identifies
manifest rows `0` and `1` before either bucket's point arrays are read. For each
visible selected manifest row, its `bucket_id` and `bucket_tile_index` locate
the physical bucket and bucket-local tile. Only then does that bucket's
`ranges/tile_indptr` locate the value-specific `row_start` and `row_count` in
the point arrays.

The two sparse pointer layers intentionally serve opposite query directions:

```text
(level, selected value)
    -> value_tiles/indptr
    -> positive manifest rows and counts
    -> manifest bucket_id + bucket_tile_index
    -> bucket ranges/tile_indptr
    -> selected value's exact point rows
```

`value_tiles` is retained as a derived query accelerator, not because it is
required for correctness. Without it, a viewport query must locate all visible
manifest tiles, open their buckets, inspect each tile's sparse value ranges, and
then discard tiles that do not contain a selected value. With it, the runtime
can first intersect the visible manifest rows with the positive manifest rows
for the selected values. Its aligned `n_points` also provides the selected-point
count needed for LOD budgeting before point payloads or bucket range records are
read.

The expected benefit is strongest for sparse values across a large viewport,
where many tiles and potentially entire buckets can be skipped. It is small for
a zoomed-in viewport, a value present in nearly every tile, or an all-values
query. The bucket-local sparse ranges remain the authoritative physical index;
`value_tiles` may always be reconstructed from them and must never be the sole
record of a tile/value relationship.

Independent validation requires their keys and counts to agree exactly. The
`value_tiles` arrays retain a Harpy-specific planning feature that is not
publicly documented in Xenium's transcript store; Xenium can derive tile paths
directly and inspect each tile's `gene_offset`, whereas Harpy groups logical
tiles into buckets and performs selected-value-aware point-budget planning.

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
  -> cache-root Zarr group and semantic attributes
  -> values and manifest catalog arrays
  -> value_tiles CSR arrays derived from bucket ranges
  -> independent staged validation
  -> final fresh source-signature guard
  -> root publication_state = "complete"
  -> atomic publication
```

At no point does derived cache state pass through Parquet; only the canonical
source remains Parquet.

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
pointer checks. Later cache-wide validation additionally reconciles the manifest
catalog counts with these validated physical counts.

Exact can derive its bucket plan from the materialized bucket. Bridge and
spatial writers derive expected output tile counts from candidate counts and
level capacities before opening their output bucket.

## Lookup semantics

### Complete tile

```text
(level, tile_x, tile_y)
  -> manifest level interval and coordinate row
  -> bucket_id, bucket_tile_index
  -> derive canonical bucket_path
  -> verify bucket tile_x[i], tile_y[i]
  -> tile_offset[i:i+2]
  -> aligned location/value_id/point_id slices
```

### Selected values

```text
selected labels
  -> root value_names attribute -> value_ids
  -> value_tiles indptr -> positive manifest rows and counts
  -> manifest arrays -> bucket ID and bucket-local tile index
  -> derive canonical bucket path
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
| Z6 | root metadata and cache-wide Zarr catalog | Z3–Z5 |
| Z7 | independent complete-generation validation | Z6 |
| Z8 | guarded end-to-end build and publication | Z7 |
| Z9 | acceptance reader and full-Xenium evaluation | Z8 |
| Z10 | batched multi-value catalog lookup | Z9 |
| Z11 | resident selected-value runtime index | Z10 |
| Z12 | resident bucket lookup indexes and eager reader initialization | Z11 |
| Z13 | bucket-wide batched point-payload reads | Z12 |
| Z14 | exact per-level selected-value catalog reads | Z13 |
| Z15 | explicit architecture-adoption decision | Z14 |
| Z16 | self-contained Zarr package and removal of the tiled-Parquet cache | Z15 |

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
on the standalone descriptors because they later become aligned manifest
catalog rows.
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
- a process crash may leave a partial bucket, but the enclosing generation
  cannot satisfy the completed root-attribute contract and cannot be published
  or read as a valid cache.

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
    construction = reader.read_construction_payload(descriptor)
    selected = reader.read_display_payload(descriptor, selected_value_ids)
```

Both methods verify descriptor bucket identity, bucket-local index, stored tile
coordinates, and descriptor count before reading point data. Calls after close
fail.

`read_construction_payload` resolves `tile_offset[i:i+2]` and returns an exact
construction `_PointPayload` including `point_id`. `read_display_payload` reads
either all display rows when value IDs are `None` or a nonempty,
one-dimensional, strictly increasing unique `uint32` selection. Its selected
path performs:

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
no requested value is present and otherwise returns only aligned `location` and
point-level `value_id` arrays. Visualization never slices or decodes `point_id`.
Splitting the stored `(N, 2)` `location`
slice into contiguous `x_rel` and `y_rel` arrays remains a construction-only
conversion in `read_construction_payload`. The inner chunk remains the selected-read
granularity even though several chunks share one physical shard.

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
spatial sampling, the cache-wide catalog, root attributes, publication state,
publication, or the final cross-index validator.

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
- Z3 never marks a generation complete, publishes a generation, repairs partial
  output, or deletes a caller-owned staging or temporary root.
- No derived-cache Parquet or standalone JSON sidecar is created.

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
- inspection proving that no derived-cache Parquet or standalone JSON sidecar
  was created.

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
- no derived-cache Parquet or standalone JSON sidecar exists;
- Z3 leaves cache artifacts, publication, and complete-generation validation to
  their planned later slices.

### Slice Z4: implement fresh Bridge construction

#### Goal

Construct Bridge Zarr buckets directly from Exact Zarr buckets.

Z4 owns fresh value-neutral sampling, deterministic Bridge bucket planning,
bounded Exact-reader reuse, and coordination of the Z2 storage primitive. It
does not reread the canonical source, implement coarser coordinate rebasing,
write the cache-wide Zarr catalog, publish a generation, or import an existing
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
  max_open_exact_readers: int | None = None
```

`None` means that Bridge construction retains one entered reader for every
nonempty physical Exact bucket; the effective capacity is therefore derived
from `exact_result.bucket_count` rather than from a hard-coded dataset-specific
number. The full-Xenium profile observed 69 Exact buckets, a 99.05% reader-cache
hit rate, and no evictions under this policy. An explicit positive integer
remains available as a stricter metadata-lifetime bound for unusually large
inputs, and is clamped to the actual nonempty Exact bucket count. A value of one
is valid and exercises the strictest reader lifetime bound. The cache retains
initialized Zarr readers and metadata, never decoded point chunks or complete
point payloads.

This reader reuse materially accelerates Bridge construction; it is not only a
metadata convenience. Bridge output-bucket order revisits Exact buckets in an
interleaved pattern. In the full-Xenium profile, retaining 16 readers caused
5,597 reader misses and 5,581 evictions, while retaining all 69 Exact readers
caused only the initial 69 misses and no evictions. Bridge construction fell
from 139.05 seconds to 99.63 seconds (28.4%), and time attributed to reader-cache
lookup and admission fell from 31.84 seconds to 2.12 seconds. These measurements
justify the default policy; they are profiling observations, not numerical
acceptance thresholds. The remaining read time is dominated by tile interval
and point-payload access, which this metadata-reader cache does not attempt to
store.

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
2. call `read_construction_payload(descriptor)` exactly once;
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
generation rather than Z4 attempting partial repair or resume. Z4 never marks
the generation complete.

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
  reader closure, and absence of a completed publication state.

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

Build every planned coarser level through the same Zarr payload path, using the
completed Bridge as the sole first input and each completed Spatial level as the
sole input to its successor. Preserve nested membership without rereading the
canonical source or introducing a second overview backend.

The implementation is fresh and isolated under:

```text
src/napari_harpy/core/multi_scale_cache_points_zarr/writer/spatial.py
tests/multi_scale_cache_points_zarr/test_spatial_writer.py
```

It may reuse only this package's build plans, descriptors, payload, sampler,
hashing, reader cache, bucket reader/writer, and storage results. It must not
import the Parquet-backed spatial writer or any existing writer-support module.

#### Writer API and execution settings

Define a Spatial-specific physical configuration:

```text
_SpatialWriterConfig
  zarr_settings: _ZarrWriteSettings
  max_open_finer_readers: int | None = None
```

`None` retains one entered reader for every nonempty physical bucket of the
current immediate-finer level. An explicit positive integer imposes a stricter
bound and is clamped to that level's actual nonempty bucket count. Reader caches
are scoped to one level: every finer-level reader is closed before the next
completed Spatial level becomes the input. The cache retains initialized Zarr
readers and metadata only, never decoded chunks or point payloads.

The construction entry point is:

```text
_write_spatial_levels(
    bridge_result,
    plan,
    *,
    staging_root,
    config,
) -> tuple[_LevelWriteResult, ...]
```

Return completed Spatial results in ascending serialized-level order. If Bridge
is already the terminal planned level, require its observed count to fit
`overview_point_budget` and return an empty tuple. An Exact-only plan is handled
by the future end-to-end coordinator and does not call this entry point.

Use a deterministic sequential coordinator. Do not introduce Dask, output-
bucket threading, new chunk or shard settings, or intermediate point files in
Z5. Process output buckets in numeric order and parent tiles within each bucket
in `(tile_y, tile_x)` order. If later full-scale evidence warrants concurrency,
add it as a separate explicit bounded policy.

#### Required transition validation

Before opening a finer bucket or creating a coarser directory, require:

- `bridge_result` describes the planned nonempty level-one Bridge;
- every planned level after Bridge has `_LevelKind.SPATIAL` and immediately
  follows its finer level;
- the coarser tile edge is exactly twice the finer edge;
- each coarser grid dimension is `ceil(finer_dimension / 2)`;
- both level results and descriptors lie inside their planned grids;
- the coarser level has a positive per-tile capacity and a non-increasing
  point-count upper bound;
- `staging_root` and the complete finer-level directory exist;
- the coarser output directory is absent.

The build-plan dataclasses already reject these inconsistencies globally. The
writer repeats the transition-local requirements at its trust boundary so a
standalone internal call fails before opening input or creating output.

#### Descriptor-only parent planning

Introduce a small immutable planning record:

```text
_CoarserTileInput
  tile_x: int
  tile_y: int
  finer_descriptors: tuple[_TileDescriptor, ...]

  candidate_count = sum(descriptor.n_points)
```

Each nonempty finer descriptor maps to exactly one parent:

```text
coarser_tile_x = finer_tile_x // 2
coarser_tile_y = finer_tile_y // 2
```

Group one through four nonempty immediate-finer descriptors per parent. Missing
or empty edge tiles are absent rather than represented by placeholders. Require
contributors to be unique, inside the finer grid, mapped to the stated parent,
and ordered by `(tile_y, tile_x)`. This planning phase handles descriptors only;
it does not open Zarr, allocate point arrays, rebase coordinates, or sample.

Route the resulting parent records to coarser output buckets by hashing the
parent `(tile_x, tile_y)` through the existing versioned tile hash and the
coarser `_bucket_count_for_level`. Empty planned bucket IDs remain absent. Build
one `_BucketPlan` per nonempty output bucket with each planned tile count equal
to:

```text
min(coarser_tile.candidate_count, coarser_level.max_points_per_tile)
```

#### Coordinate rebasing contract

Implement the rebasing logic fresh in `writer/spatial.py`. For a finer tile and
its containing parent:

```text
quadrant_x = finer_tile_x - 2 * coarser_tile_x  # exactly 0 or 1
quadrant_y = finer_tile_y - 2 * coarser_tile_y  # exactly 0 or 1

coarser_x_rel = finer_x_rel + quadrant_x * finer_tile_size
coarser_y_rel = finer_y_rel + quadrant_y * finer_tile_size
```

The four valid quadrants are:

```text
finer coordinates             one coarser tile

(2x,   2y)   (2x+1, 2y)       +---------+---------+
                                |  (0,0)  |  (1,0)  |
(2x, 2y+1)   (2x+1,2y+1)      +---------+---------+
                                |  (0,1)  |  (1,1)  |
                                +---------+---------+
```

Require finite finer-relative coordinates in the closed interval
`[0, finer_tile_size]`. Rebased coordinates must be C-contiguous `float32` in
`[0, coarser_tile_size]`. Copy `value_id` and `point_id` without modification.
Keep the pure coordinate/quadrant helper in the Spatial writer; do not broaden
the general `_PointPayload` API solely for rebasing.

#### Bounded tile assembly and sampling

For each parent tile:

1. acquire each contributor's reader from the current level-scoped
   `_BucketReaderCache`;
2. call `read_construction_payload(descriptor)` once for each of its one through four
   contributors;
3. rebase every contributor into the shared coarser-relative frame;
4. concatenate the aligned fields in deterministic finer-tile order;
5. call the fresh value-neutral sampler exactly once with the coarser serialized
   level, parent coordinates, coarser tile size, and coarser capacity;
6. take the same selected rows from every aligned field;
7. pass the retained `_PointPayload` to the common `_BucketWriter`.

Assembly should preallocate the combined aligned arrays from the descriptor-
derived `candidate_count`, fill them with a checked cursor, and release each
complete finer payload after copying it. This avoids retaining four decoded
finer payloads plus a second complete concatenation unnecessarily. The
steady-state coordinator bound is therefore:

```text
one complete immediate-finer payload
    + one combined parent-candidate payload
    + one at-most-capacity retained payload
    + one active output bucket writer and its shard buffers
    + bounded entered-reader metadata
```

Do not materialize a complete finer or coarser level, a complete output bucket
payload, one Python object per point, or any point-level shuffle. Contributor
candidate counts are bounded by their planned capacities. The common sampler
returns unique original candidate-row positions; nested construction from
disjoint immediate-finer tiles preserves unique `point_id` membership, which
focused tests and the Gate verify across every assembled parent.

Membership selection remains independent of `value_id`. The sampler returns
retained original row positions in canonical `point_id` order; `_BucketWriter`
remains the sole owner of persisted tile-internal `(value_id, point_id)` ordering
and sparse-range construction.

#### Per-level construction and reconciliation

For each planned Spatial level, perform:

```text
immediate-finer _LevelWriteResult
    -> descriptor-only parent groups
    -> deterministic destination buckets
    -> one level-scoped finer-reader cache
    -> read/rebase/assemble/sample one parent at a time
    -> common Zarr bucket writers
    -> reconciled coarser _LevelWriteResult
    -> sole input to the next transition
```

Fast normal reconciliation operates on finalized results and descriptor facts;
it does not replay every persisted point or call `_validate_bucket` for every
bucket during ordinary construction. Require:

- the output coordinate set equals the descriptor-derived nonempty parent set;
- each parent count equals `min(sum(finer counts), coarser capacity)`;
- every output descriptor belongs to the planned level and lies inside its grid;
- bucket, tile, and bucket-local descriptor identities remain unique and
  ordered through the common result contracts;
- observed level rows equal the sum of expected parent counts, do not exceed the
  coarser upper bound, and do not exceed the immediate-finer observed total;
- the terminal observed count does not exceed `overview_point_budget`.

Focused tests and the Gate independently prove that persisted output
`point_id` values are a subset of the union of the contributing immediate-finer
tiles and that coordinates and values agree after applying the specified
rebasing. Normal reconciliation does not substitute a source-level equivalence
scan for these bounded construction facts.

#### Overview semantics

Do not introduce `_LevelKind.OVERVIEW`, an overview-specific writer, or a second
payload format. “Overview” is the role of the terminal planned level:

- Exact is terminal when the validated source already fits the overview budget;
- Bridge is terminal when its planned and observed output fits the budget;
- otherwise the last Spatial level is terminal.

The terminal Spatial level is built through the same parent grouping, rebasing,
sampling, hashing, reader, and writer path as every preceding Spatial level. A
one-tile terminal level may use the planner-clamped whole-dataset overview
capacity.

#### Failure and ownership behavior

- `_BucketReaderCache` alone enters, retains, evicts, and closes finer readers.
- `_BucketWriter` owns active-bucket cleanup and removes its partial Zarr store
  after read, rebase, sampling, or write failure.
- A failure in a later bucket or level may leave previously finalized buckets
  and prerequisite levels in the isolated staging generation; it is not marked
  complete, and the future end-to-end coordinator owns generation cleanup.
- Do not delete or mutate the completed immediate-finer level.
- No source object, Dask collection, or point-payload Parquet path is accepted by
  a Spatial writer API.

#### Focused tests

- pure quadrant and coordinate rebasing, including closed upper edges and
  invalid coordinates;
- parent grouping with one, two, three, and four nonempty contributors;
- odd finer-grid dimensions, sparse edge regions, missing tiles, invalid parent
  membership, and duplicate contributors;
- descriptor-derived candidate counts and planned output counts;
- deterministic routing with multiple input and output buckets, including empty
  destination IDs;
- real reader-cache reuse, bounded explicit capacity, all-bucket default, and
  closure between successive levels;
- sparse pass-through and over-capacity sampling with recomputed deterministic
  membership;
- exact field preservation by `point_id` after coordinate rebasing;
- two- and multi-level pyramids whose every level is a subset of its immediate
  predecessor;
- terminal Bridge, terminal Spatial, and one-tile capacity-clamped overview
  plans;
- every per-tile capacity, level upper bound, and overview budget;
- uniform array schema, codec, chunk, shard, offset, and sparse-range contracts
  across Bridge and all Spatial levels;
- preexisting output, invalid transitions, descriptor/count mismatches, and
  injected read, rebase, sampling, write, and later-level failures;
- active partial-bucket cleanup, prerequisite preservation, deterministic reader
  closure, and absence of point Parquet and a completed publication state.

#### Gate Z5

Run the full focused logical and storage tests and construct one complete small
pyramid with odd grids, sparse edges, several buckets, every rebasing quadrant,
and at least two Spatial transitions. Reopen its buckets independently and prove
the complete nested membership, field rebasing, capacities, overview budget,
and uniform Zarr storage contract.

Also provide one opt-in full-Xenium run that reuses or constructs current-tree
Exact and Bridge prerequisites, then constructs all remaining Spatial levels
once. Record per level:

- construction time and peak incremental RSS;
- input candidate, output point, nonempty tile, bucket, range, shard, and
  filesystem-object counts;
- maximum contributing parent-candidate count and maximum output tile count;
- configured and peak entered finer readers;
- compressed bytes by array and total level size.

Reopen every Spatial bucket through `_validate_bucket`. Independently verify
descriptor-derived parent coordinates and counts, immediate-predecessor
`point_id` nesting, retained field equality under rebasing, deterministic
sampler membership for representative sparse, dense, edge, and terminal tiles,
and the final overview budget. Confirm that construction never reads the
canonical source and writes no point Parquet.

This is one current-format engineering gate, not a backward-compatibility run,
a Parquet comparison, an exhaustive source-equivalence replay, or a fixed
numerical benchmark threshold. The complete pyramid must be correct,
deterministic, practically fast, and memory bounded before Z6 freezes final
catalog contracts.

#### Exit criteria

- every planned Spatial level is built only from its completed immediate
  predecessor and returned in serialized order;
- parent grouping, rebasing, membership, capacities, and terminal overview
  accounting are independently correct;
- construction has explicit tile-memory and reader-lifetime bounds and closes
  resources deterministically on success and failure;
- the construction path contains no source reread or point-level Parquet reader
  or writer;
- Exact, Bridge, Spatial, and terminal overview payloads share one physical Zarr
  contract;
- the small-pyramid and full-Xenium gates establish that Z6 can freeze one
  complete Zarr cache format without unresolved level-construction behavior.

### Slice Z6: freeze final cache-format and catalog contracts — resolved

**Status:** implemented with focused verification on 2026-08-17 and accepted by
the current level-at-a-time full-Xenium Gate Z6 run on 2026-08-18.

The initial focused implementation suite completed with all 184 tests in
`tests/multi_scale_cache_points_zarr` passing. The initial full-Xenium Gate Z6
processed 136,578,750 source points, 9 cache levels, 5,122 values, 17,149
manifest rows, and 29,787,508 nonempty tile/value rows. Its external-merge
catalog constructor took 68.93 seconds with 58.28 MB incremental peak RSS.

The persisted catalog occupied 44,162,968 bytes across 79 filesystem objects;
the root `zarr.json`, including all value labels and level summaries, occupied
84,723 bytes. Strict reopened validation plus an independent comparison of nine
representative bucket indexes, covering 1,991,105 sparse range records and
1,049 manifest tiles, completed in 9.03 seconds. Bucket stores and the canonical
source remained unchanged, no point payload array or canonical source data page
was read by Z6, no derived Parquet was written, and the generation remained in
the staging publication state.

A follow-up design review on 2026-08-17 replaced that external merge with the
level-at-a-time NumPy sort specified below. The persisted catalog contract is
unchanged, and all 183 focused tests in `tests/multi_scale_cache_points_zarr`
pass after the refactor. The earlier 68.93-second and 58.28-MB observations
describe the superseded constructor and remain only as historical evidence;
they motivated the current-constructor full-Xenium timing and memory rerun
recorded below.

That current-format rerun completed on 2026-08-18. Catalog construction over
29,787,508 value/tile rows took 21.08 seconds. The largest individual level
contained 14,790,090 range records; its compact input arrays and NumPy order
permutation were estimated at 295,801,800 and 118,320,720 bytes respectively.
The catalog interval started at 1,553,907,712 bytes RSS, peaked at
2,305,064,960 bytes RSS, and therefore added 751,157,248 bytes at peak. This is
consistent with memory proportional to the largest level and remains practical
on the 32-GiB evaluation machine.

The strict reopened catalog validation and representative compact-index check
took 7.76 seconds. It sampled one bucket from every level, covering 1,991,105
range records and 1,049 manifest tiles. All 9 levels, 5,122 values, 17,149
manifest rows, and 29,787,508 value/tile rows reconciled. The persisted catalog
contract remained 44,162,968 bytes across 79 filesystem objects, with an
84,723-byte root `zarr.json`. No point payload array or canonical source data
page was read, and the source, evaluated buckets, and reusable pyramid remained
unchanged. No derived Parquet, standalone cache JSON sidecar, catalog-sort
scratch was produced, and the generation remained in the staging publication
state.

The retained engineering artifacts are:

```text
/Users/arne.defauw/VIB/DATA/test_data/
  sdata_xenium_full_data_core.transcripts-cache-workspace/
    pyramid-base/
    z6-20260818-current/
    reports/gate-z6-20260818.json
```

The prerequisite build took 38.62 seconds for Exact, 107.02 seconds for Bridge,
and 77.41 seconds for all Spatial levels. These prerequisite timings and the
5.02-second hard-link tree creation are recorded separately and are not part of
the 21.08-second Z6 catalog measurement. The catalog-free `pyramid-base`
occupies approximately 1.6 GiB physically. The evaluated generation shares its
immutable bucket files and adds approximately 42 MiB of private hierarchy and
catalog data, so future Z6 experiments can reuse the pyramid without rerunning
level construction.

#### Goal

Freeze the first complete Zarr-only derived-cache contract and write its root
attributes and cache-wide typed catalog arrays from validated source facts and
completed level results. After this slice, one unpublished staging generation
is self-describing without retaining writer objects or reading point payload
arrays.

Z6 owns root-group creation, cache-format models, catalog construction, and
writer-time reconciliation. It does not own independent publication validation,
completion, publication, runtime level selection, or runtime LOD/render-budget
policy; those remain Z7 through Z9 responsibilities.

The implementation remains isolated under:

```text
src/napari_harpy/core/multi_scale_cache_points_zarr/
  cache_format.py
  storage/catalog_writer.py
  storage/catalog_reader.py
  writer/catalog.py

tests/multi_scale_cache_points_zarr/
  test_cache_format.py
  test_catalog_writer.py
  test_catalog_reader.py
```

All Zarr opening, array creation, compact-index iteration, and physical layout
validation remain storage responsibilities. `writer/catalog.py` coordinates
validated facts and level results through those storage APIs; it must not reach
through private Zarr group or array handles owned by another object.

#### Xenium-aligned metadata policy

Follow the same broad division used by Xenium's visualization store:

- small versions, identities, semantic settings, value labels, and level
  summaries are Zarr group attributes;
- scalable, typed, or sliceable structures are Zarr arrays;
- sparse indexes use `indptr` plus aligned data arrays;
- Parquet remains a canonical analysis/source format, not an internal
  visualization-catalog format.

The public Xenium transcript-store specification places versions, dataset
identity, gene names, grid keys, and grid counts in Zarr attributes, while
transcript fields, `gene_offset`, and sparse density CSR structures are arrays:
[10x Xenium Zarr output format](https://www.10xgenomics.com/support/software/xenium-onboard-analysis/latest/advanced/xoa-output-zarr).
Zarr v3 stores a group's JSON-compatible user attributes in that group's
`zarr.json`: [Zarr v3 core specification](https://zarr-specs.readthedocs.io/en/latest/v3/core/v3.0.html).

Do not add `metadata.json` or `manifest.json`. Zarr v3 already serializes group
attributes in `zarr.json`; a separate JSON manifest would be verbose, untyped,
and all-or-nothing to parse. Do not add final derived-cache Parquet sidecars.

#### Three independent version identities

Do not use one version for three compatibility boundaries. Freeze:

```text
cache schema version
    harpy-multiscale-points-zarr-cache-0.1

backend identifier
    harpy-zarr-v3-bucket-sparse-value-ranges-v1

bucket payload schema version
    1
```

The cache schema version identifies the root-attribute and catalog-array
contract. The backend identifier selects the bucket family. The integer payload
version remains the root-attribute contract of each bucket. The initial reader
supports only these versions and fails closed on unknown values; it does not
guess another cache family or fall back to the existing backend.

#### Cache-format module boundary

`cache_format.py` owns:

- cache, backend, catalog, and group-path constants;
- exact root-attribute keys and typed metadata models;
- catalog array names, dtypes, dimensionality, ordering, and pointer semantics;
- normalized cache-relative POSIX path validation;
- version and supported-storage-profile validation.

It performs no filesystem I/O, imports no level writer, and does not open Zarr.
Zarr-Python owns JSON encoding of `zarr.json`; the application contract is the
exact parsed attribute values, not byte ordering of the metadata document.

#### Root-group attributes

Create the staging root as a Zarr v3 group. Its `attributes` object has exactly
these top-level keys:

```text
schema_version
cache_generation_id
publication_state
created_by
backend
source
geometry
build
levels
value_names
catalog
```

The nested value types and structure are:

```json
{
  "schema_version": "harpy-multiscale-points-zarr-cache-0.1",
  "cache_generation_id": "00000000-0000-0000-0000-000000000000",
  "publication_state": "staging",
  "created_by": {
    "package": "napari-harpy",
    "version": "0.0.0"
  },
  "backend": {
    "identifier": "harpy-zarr-v3-bucket-sparse-value-ranges-v1",
    "zarr_format": 3,
    "payload_schema_version": 1,
    "point_row_order": ["tile_y", "tile_x", "value_id", "point_id"],
    "coordinate_encoding": "tile-relative-xy-float32-v1",
    "codec_id": "zstd-v1",
    "point_chunk_rows": 4096,
    "point_shard_rows": 131072,
    "range_chunk_rows": 8192,
    "range_shard_rows": 131072
  },
  "source": {
    "points_name": "transcripts",
    "element_path": "points/transcripts",
    "row_count": 136578750,
    "columns": {"x": "x", "y": "y", "value": "gene"},
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
    "target_points_per_bucket": 2000000,
    "bucket_hash_method": "harpy-zarr-tile-splitmix64-v1",
    "sampling_method": "harpy-value-neutral-stratified-splitmix64-v1",
    "sampling_seed": 0,
    "sampling_microgrid_edge": 16
  },
  "levels": [
    {
      "level": 0,
      "kind": "exact",
      "tile_size": 512,
      "grid_width": 106,
      "grid_height": 74,
      "max_points_per_tile": null,
      "point_count_upper_bound": 136578750,
      "bucket_count": 69,
      "tile_count": 7844,
      "point_count": 136578750,
      "range_count": 1000000,
      "relative_directory": "levels/level_0"
    }
  ],
  "value_names": ["ACTB", "EPCAM", "MALAT1"],
  "catalog": {
    "value_count": 3,
    "level_count": 9,
    "manifest_row_count": 20000,
    "value_tile_row_count": 6000000,
    "values_group": "values",
    "manifest_group": "manifest",
    "value_tiles_group": "value_tiles",
    "manifest_row_order": ["level", "tile_y", "tile_x"],
    "value_tile_key_order": ["level", "value_id"],
    "manifest_chunk_rows": 65536,
    "manifest_shard_rows": 262144,
    "value_tile_chunk_rows": 65536,
    "value_tile_shard_rows": 1048576
  }
}
```

Numeric values and labels are illustrative; keys, nesting, list order, and JSON
types are normative. `levels` contains exactly one entry per planned and
completed level in ascending order. It retains both planned bounds and actual
bucket, tile, point, and range totals.

`value_names` is the canonical label dictionary in implicit zero-based
`value_id` order, matching Xenium's use of a root gene-name attribute. Labels
are suitable for attributes at the expected vocabulary size; exact counts remain
a typed array. If future vocabularies make the attribute materially large, that
requires a versioned format decision rather than an implicit string-array
change.

`source.selected_schema` uses semantic role order `x`, `y`, `value` and the same
normalized Arrow-type representation as the source signature. Exclude absolute
host paths, scratch paths, Dask configuration, timings, RSS, and creation time.

The backend records the actual common bucket settings. Every bucket must match.
Numeric chunk and shard sizes are recorded and cross-validated but are not part
of the cache schema identifier. `cache_generation_id` is a canonical lowercase
hyphenated UUID created by Z8 and passed unchanged into Z6.

Write all root attributes, including `publication_state = "staging"`, in one
final `update_attributes` operation after the catalog arrays reconcile. The
empty root group necessarily has a `zarr.json` during catalog construction. Z8
later changes only this root attribute to `"complete"` after validation and the
final source guard; no non-Zarr completion sidecar is part of the format.

#### Exact catalog-array contract

All catalog numeric arrays use little-endian core Zarr dtypes, zero fill values,
default `/` chunk-key separation, and the declared `zstd-v1` profile. Arrays
and the `values`, `manifest`, and `value_tiles` groups have no user attributes;
their semantics are frozen centrally by cache schema and root attributes.

Values, for `G` canonical labels:

```text
values/n_points              shape=(G,)         dtype=uint64
```

Manifest, for `L` levels and `T` nonempty tiles:

```text
manifest/level_indptr        shape=(L + 1,)     dtype=uint64
manifest/bucket_id           shape=(T,)         dtype=uint32
manifest/bucket_tile_index   shape=(T,)         dtype=uint32
manifest/tile_x              shape=(T,)         dtype=uint32
manifest/tile_y              shape=(T,)         dtype=uint32
manifest/n_points            shape=(T,)         dtype=uint64
```

Value-to-tile index, for `M` nonempty `(level, value, tile)` combinations:

```text
value_tiles/indptr           shape=(L, G + 1)   dtype=uint64
value_tiles/manifest_index   shape=(M,)         dtype=uint64
value_tiles/n_points         shape=(M,)         dtype=uint64
```

`values/n_points`, `manifest/level_indptr`, and `value_tiles/indptr` each use
one unsharded chunk because they are compact pointer/dictionary structures for
the supported dimensions. The five row-aligned manifest arrays share one row
chunk/shard layout. The two row-aligned `value_tiles` arrays share a separate
row chunk/shard layout. Initial writer settings are:

```text
manifest_chunk_rows    = 65,536
manifest_shard_rows    = 262,144
value_tile_chunk_rows  = 65,536
value_tile_shard_rows  = 1,048,576
```

Tests inject smaller positive multiples. These are tunable recorded physical
settings, not logical schema identities. `value_tile_chunk_rows` also supplies
the bounded batch size used while reading compact range metadata and writing
ordered catalog rows; that transient use does not add another format setting.
Gate Z6 records whether the initial values remain practical.

#### Catalog-writer API and preconditions

Implement one level-neutral operation shaped as:

```text
_write_staged_cache_catalog(
    validated,
    plan,
    level_results,
    *,
    staging_root,
    cache_generation_id,
    settings,
) -> None
```

Require before creating root or catalog metadata:

- exact validated source, build plan, level-result, and catalog-settings
  contracts;
- exactly one result per planned level in ascending order;
- each result's descriptors, geometry, capacities, and totals satisfy its plan;
- every completed bucket path exists and no additional bucket path is present;
- bucket metadata and layouts expose one common supported settings profile;
- staging is an existing directory tree;
- root `zarr.json`, `values`, `manifest`, `value_tiles`, and the catalog groups
  do not already exist.

The staging root already contains finalized bucket directories. Create group
metadata for the cache root, `levels`, every `level_<n>`, and the three catalog
groups without overwriting any bucket. A bucket's existing root `zarr.json`
simultaneously represents its independently openable `LocalStore` and its child
group node in the complete hierarchy.

The operation returns `None`. Persisted root attributes and catalog arrays, not
an in-memory result, are the handoff to Z7.

#### Values and manifest construction

Copy canonical labels from `ValidatedPointsSource.value_table` into root
`value_names` and exact counts into a fresh contiguous `uint64` array. Revalidate
contiguous implicit IDs, UTF-8 label order, positive counts, and equality among
their sum, validated source rows, and Exact points.

Flatten all descriptors in `(level, tile_y, tile_x)` order and assign each one a
zero-based global `manifest_index`. Reject duplicate logical tile keys and
duplicate `(level, bucket_id, bucket_tile_index)` addresses. Write the aligned
manifest arrays and `level_indptr`; do not store `level` or `bucket_path` per
row because both are derived from pointers and integer identity.

The complete manifest is small relative to payload data and may be assembled in
memory. Reconcile every row with its result, plan, grid, canonical bucket path,
bucket-local ordering, and point count before writing.

#### Level-at-a-time `value_tiles` construction

Do not materialize all sparse bucket ranges in one table. For each finalized
bucket in `(level, bucket_id)` order, use a storage-owned bounded iterator that
reads only:

```text
root attributes
tile_x
tile_y
tile_offset
ranges/tile_indptr
ranges/value_id
ranges/row_start
ranges/row_count
```

It must not decode `location`, point-level `value_id`, or `point_id`. The
iterator verifies compact identity, pointers, per-tile value order, positive
counts, contiguous point-row coverage, and descriptor/catalog agreement while
carrying boundary state across batches.

Map every range to one typed scratch record inside the containing level stream:

```text
(value_id, manifest_index, n_points)
```

The level is structural context supplied once by the stream's position in the
level-ordered input tuple; do not allocate or repeat a level value for every
range record. Processing streams in ascending level order and sorting each
stream by `(value_id, manifest_index)` still produces the cache-wide persisted
key order `(level, value_id, manifest_index)`.

The mapping is explicit rather than inferred from iteration order. Manifest
construction creates an address map from
`(level, bucket_id, bucket_tile_index)` to the global `manifest_index`. For
bucket-local tile `i`, use `ranges/tile_indptr[i:i + 2]` to visit its range
records. Each range record contributes:

```text
containing stream level = bucket level
record value_id         = ranges/value_id[j]
record manifest_index   = manifest address map[level, bucket_id, i]
record n_points         = ranges/row_count[j]
```

`ranges/row_start[j]` is not copied into `value_tiles`; it is used with
`row_count` and `tile_offset` to validate that the range describes the expected
contiguous physical point rows.

For the three-tile example above, after assigning manifest rows `0`, `1`, and
`2`, traversal of the level 0 stream emits:

```text
tile 0: (value=0, manifest=0, count=10)
        (value=2, manifest=0, count= 3)
tile 1: (value=1, manifest=1, count= 8)
        (value=2, manifest=1, count= 4)
tile 2: (value=0, manifest=2, count= 6)
```

Sorting the level 0 stream by `(value_id, manifest_index)` produces these
conceptual cache-wide rows, with level supplied by the stream:

```text
(0, 0, 0, 10)
(0, 0, 2,  6)
(0, 1, 1,  8)
(0, 2, 0,  3)
(0, 2, 1,  4)
```

The final sequential write therefore yields:

```text
value_tiles/manifest_index = [0,  2, 1, 0, 1]
value_tiles/n_points       = [10, 6, 8, 3, 4]
value_tiles/indptr[0]      = [0,  2, 3, 5]
```

Bucket traversal naturally arrives in approximately
`(level, bucket, tile, value)` order, while the inverted index requires
`(level, value, manifest tile)` order. That transpose requires sorting rather
than writing the final arrays directly during bucket traversal.

The primary persisted key is `level`, and level results are already serialized
in ascending order. Therefore sort one complete level at a time by
`(value_id, manifest_index)` and concatenate those ordered level streams. This
is exactly equivalent to one cache-wide sort by
`(level, value_id, manifest_index)`; records from different levels never need to
participate in the same NumPy sort.

For level `l`, preallocate arrays at its already reconciled `range_count`:

```text
value_id       uint32
manifest_index uint64
n_points       uint64
```

Fill those arrays from the bounded compact-range iterator, requiring every
batch to identify level `l`, reference only that level's manifest interval, and
produce exactly the declared range count. Then calculate:

```python
order = np.lexsort((manifest_index, value_id))
```

`np.lexsort` treats the last key as primary, so this groups by value and orders
manifest rows inside every value. Reject duplicate
`(level, value_id, manifest_index)` keys, including duplicates crossing an
output-batch boundary. Use `np.bincount(value_id, minlength=G)` and a cumulative
sum to construct the level's complete `indptr` row, including empty values and
the global cursor offset from preceding levels.

Write `manifest_index[order]` and `n_points[order]` in slices of at most
`value_tile_chunk_rows`. Materialize only each output slice as contiguous arrays;
do not create full level-sized ordered copies. Release the three input arrays
and the permutation before collecting the next level:

```text
one level's compact bucket ranges
    -> preallocate exactly range_count compact records
    -> fill and reconcile those records
    -> np.lexsort by (value_id, manifest_index)
    -> validate and write ordered bounded slices
    -> build this level's indptr row
    -> release the level workspace
    -> continue with the next level
```

Peak catalog-sort memory therefore scales with the largest individual level,
not with all `M` cache records. The input arrays require 20 bytes per range and
the usual 64-bit NumPy permutation requires another 8 bytes per range, in
addition to NumPy sorting workspace and small output slices. A few gigabytes of
construction memory are acceptable for large production datasets; avoiding
temporary Zarr runs, multipass I/O, open-run management, and a Python k-way
merge is the preferred complexity and performance trade-off. This is a
construction policy only and does not change the persisted catalog schema.

#### Reconciliation before final root attributes

Require:

- every catalog group and array has exactly the frozen hierarchy, dtype, shape,
  layout, codec, fill value, and empty attributes;
- pointer arrays start at zero, are nondecreasing, and terminate at their
  declared row totals;
- every manifest level slice has unique ordered coordinates and valid grid,
  bucket, and bucket-local identities;
- every manifest tile has at least one value entry and no value entry references
  an absent or wrong-level manifest row;
- each manifest tile's value counts sum to its `n_points`;
- every bucket contributes exactly its root and result `range_count` records;
- each level's value counts equal its manifest and result point totals;
- Exact per-value totals equal `values/n_points`;
- sampled levels reconcile to their own totals without claiming canonical
  per-value preservation;
- level summaries and physical settings in the proposed root attributes equal
  the reconciled arrays and stores exactly;
- no final derived-cache Parquet or standalone JSON sidecar exists.

This is writer-time accounting over compact structures. It is not the
independent Z7 validation pass and does not read or compare every point row.

#### Failure and ownership contract

Z6 writes only inside the unique unpublished staging generation. It records the
root as `publication_state = "staging"`; it never marks that state complete,
publishes a path, updates a public pointer, rereads canonical point rows, or
removes an existing completed cache.

The operation closes all catalog and bucket readers and removes its private
temporary Zarr runs on every exit. A failure may leave an incomplete root group
or catalog arrays in unpublished staging; Z8 owns removal of the entire failed
generation. Z6 and Z7 do not repair or resume it.

#### Focused tests

- exact separation of cache, backend, and payload versions;
- root-attribute exact-key parsing, canonical UUIDs, JSON-compatible scalar
  types, value-label order, absent timestamps/host paths, and unknown versions;
- root, ancestor, catalog, and existing independent bucket groups forming one
  discoverable Zarr v3 hierarchy without bucket rewrites;
- exact catalog group/array names, dtypes, shapes, chunks, shards, codecs, fill
  values, dimension counts, and empty node attributes;
- values with invalid labels, counts, implicit IDs, and Exact totals;
- Exact-only, terminal Bridge, and multi-Spatial manifest pointers and rows;
- duplicate tile identities, duplicate bucket-local addresses, wrong derived
  paths, missing/extra buckets, out-of-grid tiles, and count mismatches;
- compact range iteration across chunks and shards without a point-array read,
  including batch boundaries inside and between tiles;
- level-at-a-time sorting with deliberately unordered input, multiple bounded
  output batches, duplicate rejection across output boundaries, empty
  `(level, value)` segments, and continuous cross-level pointers;
- equality among bucket ranges, `value_tiles`, manifest counts, level results,
  root summaries, and canonical Exact value totals;
- unsupported codec/payload/layout settings and mixed bucket settings fail
  closed;
- absence of derived Parquet, standalone metadata/manifest JSON, point-array
  reads, completed publication state, publication, and overwrite behavior.

Use real small Z2 buckets for storage and lifecycle claims. Assert semantic root
attributes and logical array content, not `zarr.json` key ordering or compressed
bytes.

#### Gate Z6: full-Xenium catalog evaluation

Provide one opt-in current-tree full-Xenium run that constructs or reuses one
complete valid staging pyramid and writes the Zarr root and catalog once.
Measure Z6 separately from Exact, Bridge, Spatial, and Z7.

Retain this expensive engineering fixture beside, rather than inside, the
canonical SpatialData store. The benchmark workspace has two distinct roles:

```text
sdata_xenium_full_data_core.transcripts-cache-workspace/
  pyramid-base/
    _benchmark_pyramid_inventory.json
    levels/                         # Exact, Bridge, and every Spatial level
  z6-<run-name>/
    levels/                         # cheap local clone of pyramid-base/levels
    values/
    manifest/
    value_tiles/
    zarr.json
  reports/
    gate-z6-<run-name>.json
```

`pyramid-base` is a benchmark-owned, catalog-free prerequisite template. Its
inventory records the validated source signature, logical plan, physical Zarr
settings, and finalized level results needed to reconstruct the in-memory Z6
input contracts. Reuse requires exact equality with the freshly validated
source, plan, and settings, plus exact agreement between inventoried and
physical bucket paths. The inventory is not copied into an evaluated cache and
is not part of the persisted cache format.

Each Z6 run creates a new evaluation generation from the immutable level
template. On the local same-filesystem benchmark workspace, hard links avoid a
second physical copy of the point payload while giving the run private root,
ancestor, and catalog metadata. Z6 opens every bucket read-only and verifies
that both the evaluation bucket snapshots and `pyramid-base` remain unchanged.
Never rerun catalog construction in place: the production writer remains
write-once and continues to reject existing root or catalog targets. Retain the
evaluated generation for Z7 and later slices, while retaining `pyramid-base`
for isolated catalog reruns.

Record:

- catalog construction time and peak incremental and process RSS;
- value count, manifest rows, and value-tile rows;
- largest per-level range count, estimated compact input/permutation bytes, and
  observed peak incremental and process RSS;
- bytes, chunks, shards, and filesystem objects by catalog array and in total;
- root `zarr.json` size, including the complete value-label attribute;
- reconciled per-level bucket, tile, point, and range summaries;
- confirmation that no source data page or Zarr point payload array was read and
  no derived Parquet or standalone JSON sidecar was written;
- absence of catalog-sort scratch data and closure of every Zarr handle.
- paths of the retained pyramid, evaluated generation, and report; whether the
  prerequisite pyramid was built or reused; and the time required to clone its
  level tree. Prerequisite construction and cloning remain outside the measured
  Z6 catalog interval.

Reopen the cache root and catalog through the strict Z6 reader and repeat
hierarchy, attributes, layout, ordering, uniqueness, pointer, and aggregate
checks. Independently sample representative bucket compact indexes against
their `value_tiles` entries. Exhaustive source/point equivalence remains an
opt-in Z7 operation.

This is one current-format engineering evaluation, not a comparison with the
Parquet-backed cache, a fixed numerical threshold, or a runtime LOD benchmark.
Catalog construction must be correct and practically fast, with peak memory
proportional to the largest level rather than the complete cache, before Z7
treats it as the publication contract.

#### Exit criteria

- one exact versioned Zarr contract covers root metadata, values, manifest,
  value-to-tile planning, and bucket payloads;
- one staged generation is self-describing without writer objects or sidecars;
- the catalog is reproducible from validated source facts, build plan, level
  results, and compact bucket indexes without point-payload reads;
- catalog sorting, memory, scratch space, and handles are bounded explicitly;
- all catalog and physical totals reconcile before final root attributes;
- no format decision needed by independent Z7 validation remains implicit;
- the full-Xenium gate establishes that the Zarr-only catalog is viable.

### Slice Z7: implement independent staged validation

**Status:** implemented with focused verification and accepted by the
full-Xenium normal-publication gate on 2026-08-18. Exhaustive whole-cache and
source-equivalence orchestration is retained only as non-packaged developer
tooling under `scripts/`.

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

Z7 therefore distinguishes the production publication gate from optional
developer tooling:

1. **Normal publication validation** is mandatory in the Z8 build flow. It
   reopens and validates root attributes, hierarchy, array layouts, compact
   pointer and catalog arrays, paths, counts, versions, and cross-index
   accounting. It must not read every `location`, point-level `value_id`, and
   `point_id` row, construct a global point-ID bitmap, or rescan canonical
   source content.
2. **Exhaustive acceptance/diagnostic validation** is a non-packaged script. It
   may run the complete `_validate_bucket` scan, prove global Exact point-ID
   coverage with a bounded external structure, check cross-level point-ID
   membership, and compare finalized values and reconstructed coordinates with
   the canonical source. Use it for format or algorithm changes, release
   qualification, benchmarks, and investigation of suspected corruption, not
   as a runtime or publication API.

The tiers share low-level parsers and structural checks where useful, but they
have distinct ownership: only the compact validator is installed with
`napari_harpy`, so the exhaustive path cannot accidentally become a normal
publication cost or production maintenance contract.

#### Entry points and ownership

Add the orchestration module already reserved by the package boundary:

```text
multi_scale_cache_points_zarr/
  writer/
    staging_validation.py
```

Its mandatory normal entry point is:

```python
_validate_staged_cache(staging_root: Path) -> None
```

The path is the only input. Do not accept `ValidatedPointsSource`,
`_PointsCacheBuildPlan`, `_LevelWriteResult`, `_BucketWriteResult`, an open
writer, or any caller-supplied catalog facts. The validator reconstructs every
expected fact from the reopened root attributes, catalog arrays, and physical
bucket stores. Success returns `None`; any structural, logical, or accounting
mismatch raises and leaves the generation unpublished. Z8 is the first
production consumer and calls this entry point before its final source guard
and before changing the root publication state to `"complete"`.

The optional whole-cache diagnostic is the repository script
`scripts/validate_multi_scale_cache_points_zarr_exhaustive.py`. It first calls
the installed normal validator, then composes private bucket readers and
validation primitives. Without source arguments it checks complete on-cache
payload, point-ID, and cross-level facts. Supplying the source SpatialData path,
points name, and selected columns additionally enables freshly validated
source-row equivalence.

Neither the production validator nor the developer script writes, repairs,
removes, or publishes the cache. Both open stores read-only and close every
handle on success and failure. Keep storage layout parsing in `storage/`; keep
publication-critical complete-generation coordination in
`writer/staging_validation.py`; keep global payload/source acceptance
orchestration in `scripts/` so it is not included in the wheel.

#### Normal publication-validation flow

The mandatory path is:

```text
staging_root only
  -> open a fresh read-only _CatalogReader
  -> parse and validate root attributes, hierarchy, and catalog layouts
  -> _CatalogReader.validate_contents()
  -> reconstruct manifest descriptors grouped by physical bucket
  -> enumerate and validate the exact physical bucket inventory
  -> reopen every bucket through the compact validation path
  -> validate persisted geometry, hierarchy, capacities, and bucket hashing
  -> compare compact bucket ranges exactly with value_tiles, one level at a time
  -> require root publication_state = "staging" and reject forbidden artifacts
  -> close all handles
  -> return None
```

`_CatalogReader.validate_contents()` is the cache-wide catalog primitive owned
by Z6. It validates value totals, manifest pointers and ordering, bucket-local
addresses, `value_tiles` pointers and ordering, and per-manifest, per-level, and
Exact per-value totals. Z7 consumes it; it does not duplicate those checks in
the staging coordinator. It then adds the physical-bucket and cross-index facts
that the catalog alone cannot establish.

#### Compact bucket-validation primitive

Do not use `_validate_bucket` in normal validation. That existing exhaustive
primitive decodes every `location`, point-level `value_id`, and `point_id` row.
Add or factor a separate read-only compact path that:

- accepts bucket identity and the expected ordered manifest descriptors
  reconstructed by Z7, never a writer result;
- validates the exact bucket hierarchy, root attributes, array dtypes, shapes,
  chunks, shards, codecs, fill values, and chunk-key encoding;
- reads `tile_x`, `tile_y`, `tile_offset`, and `ranges/tile_indptr` completely;
- requires tile coordinates and point counts to equal the corresponding
  manifest rows at every `bucket_tile_index`;
- requires pointer origins, strict monotonicity, and terminals to agree with
  physical point and range shapes;
- streams only `ranges/value_id`, `ranges/row_start`, and `ranges/row_count` in
  bounded batches;
- validates strictly increasing values per tile, positive counts, contiguous
  range-row coverage, tile boundaries, and final point/range totals;
- yields or exposes compact `(value_id, manifest_index, n_points)` batches for
  the level cross-index comparison;
- opens point payload arrays only far enough to validate their metadata and
  layouts, with `read_missing_chunks=False`, but never indexes or decodes their
  data in the normal tier.

The current `_iter_bucket_range_batches` behavior already establishes most of
the compact logical invariants, but its construction-time
`_BucketWriteResult` input must not become a trust dependency for Z7. Factor or
wrap the storage logic so expected descriptors originate from the reopened
manifest and observed range totals originate from the reopened bucket.

#### Reconstruct and validate the persisted build plan

Root metadata and the manifest must independently describe one valid hierarchy:

- the Exact origin is the aligned floor of the stored source bounds at
  `leaf_tile_size`, and its grid is the exact covering grid for those bounds;
- Exact tile size equals `leaf_tile_size` and Exact point count equals the
  stored source row count;
- when present, Bridge has the Exact tile size, grid, and exact nonempty tile
  coordinate set;
- every Spatial tile size doubles its immediate finer tile size and every grid
  dimension is `ceil(finer / 2)`;
- every Spatial nonempty coordinate is present exactly when at least one finer
  coordinate maps to it through `(finer_x // 2, finer_y // 2)`;
- every sampled tile stores exactly
  `min(sum(contributing_finer_n_points), max_points_per_tile)` points; Bridge
  uses its same-coordinate Exact tile as the sole contributor;
- all tile coordinates lie inside their stated grids, every sampled tile obeys
  its capacity, level point totals are nonincreasing, and the terminal level's
  point total does not exceed `overview_point_budget`;
- stored point-count upper bounds, capacities, level kinds, and level numbering
  follow the versioned build policy;
- the planned bucket count is derived from the stored upper bound and
  `target_points_per_bucket`, and every manifest `bucket_id` equals the result
  of the stored versioned tile-hash policy for that level and coordinate;
- within each bucket, manifest `bucket_tile_index` values are exactly contiguous
  from zero in `(tile_y, tile_x)` order.

These are compact catalog checks. They prove geometry and count consequences of
construction, not sampled point membership; the developer-only exhaustive tool
owns point-level membership diagnostics.

#### Exact compact cross-index comparison

Require exact equality between the bucket sparse-range records and the
persisted `value_tiles` inverted index. Do not use an order-independent checksum
or probabilistic multiset hash, and do not perform one random Zarr lookup per
range record.

Use the level-at-a-time policy accepted and measured in Z6:

```text
compact bucket ranges for one level, in bounded input batches
  -> materialize value_id, manifest_index, n_points for that level
  -> reconcile observed row count with root level.range_count
  -> np.lexsort by (value_id, manifest_index)
  -> stream-compare ordered slices with that level's value_tiles interval
  -> require exact key and n_points equality
  -> release all level arrays and the permutation
  -> continue with the next level
```

The comparison must also require every manifest row to receive exactly its
stored `manifest/n_points`, every level to receive exactly its root point total,
and Exact per-value totals to equal `values/n_points`. Some of these totals are
already checked by `validate_contents()`; retaining them at the independently
derived bucket side is intentional cross-index reconciliation.

Normal validation may therefore materialize one complete level of compact
range records and its NumPy permutation. It must never materialize one complete
level of point payloads. Peak memory scales with the largest individual
`level.range_count`, not the complete cache. The current full-Xenium Z6 gate
measured the same largest-level policy at 14,790,090 compact records and
751,157,248 bytes incremental peak RSS, which is the accepted initial bound for
this professional validation path. If a future dataset makes that policy
impractical, revisit an external exact merge without changing the persisted
format.

#### Work

Validate, in bounded batches:

- enumerate the bucket stores referenced by the manifest catalog and reopen every
  expected physical bucket through a metadata/layout validation path that does
  not decode the complete point payload;
- root attributes, backend version, build plan, and catalog schema;
- exact equality between catalog-derived bucket paths and physical stores;
- Zarr v3 hierarchy, attributes, shapes, dtypes, chunks, shards, and codecs;
- bucket tile coordinates, offsets, and manifest-catalog counts;
- compact sparse range keys, ordering, counts, and pointer bounds without
  rereading point-level values during normal validation;
- equality between range keys/counts and `value_tiles` CSR entries;
- Exact per-value totals and `values/n_points`;
- level geometry, capacities, and overview budget;
- absence of unreferenced stores, unexpected derived Parquet/JSON sidecars,
  construction scratch, and a premature completed publication state.

The optional exhaustive developer script additionally owns:

- complete bucket payload validation, including sparse-range agreement with
  point-level values;
- Exact point-ID completeness and uniqueness;
- immediate-coarser point-ID subset membership;
- point-level coordinate validity and source-coordinate reconstruction
  tolerance;
- canonical source-row value and coordinate equivalence when requested.

Validation must not load a complete point-payload level or all Exact IDs into
one Python collection. The normal tier may hold one complete compact
range-record level as specified above. The developer script uses bounded scans
and temporary external data structures for complete point-ID and payload facts.
Normal publication validation must not perform a complete point-payload or
canonical-source scan.

#### Exhaustive acceptance/diagnostic flow

This is an engineering-script contract, not an installed `napari_harpy` API.
After normal validation succeeds, the opt-in script may reuse
`_validate_bucket` to decode and validate every complete bucket payload. Its
additional checks are separate phases with explicit scratch ownership:

1. validate every point array and its agreement with the sparse range index;
2. prove Exact `point_id` values are unique and cover exactly
   `0..source.row_count - 1` using a bounded external representation rather
   than a Python set or full in-memory bitmap;
3. prove every immediate-coarser level's point IDs are a subset of its
   immediate finer level and that retained `value_id` and reconstructed source
   coordinates are unchanged;
4. prove tile-relative coordinates are finite and inside the logical tile,
   including the defined upper-edge tolerance;
5. when source arguments are supplied, freshly validate that source and compare
   Exact point IDs, normalized values, and reconstructed coordinates with the
   canonical source in bounded batches.

Use only caller-owned `temporary_directory_root` for external runs or merge
state. Remove private scratch and close all handles on every exit. A failed
script run leaves the staged generation intact for diagnosis; it does not
attempt repair or participate in publication.

#### Failure and artifact policy

Normal validation requires root `publication_state = "staging"` because Z8
changes it to `"complete"` only after validation and its final source guard.
Reject missing or unreferenced buckets, additional level or catalog nodes,
derived Parquet, standalone JSON sidecars, known construction scratch, and
unexpected generation-level files. Do not mistake valid Zarr metadata and
chunk/shard objects for sidecars.

Every failure is fail-closed and names the violated layer and relevant logical
identity where practical: root/catalog, level, bucket, tile, value, pointer, or
manifest row. Z7 does not weaken the exact layout checks merely to produce a
more specific downstream error.

#### Focused tests

- valid Exact-only and multilevel generations;
- missing/extra buckets, groups, and catalog arrays;
- corrupted metadata and backend versions;
- malformed arrays, offsets, ranges, and attributes;
- manifest-catalog/bucket and range/`value_tiles` mismatches;
- normal-tier capacity, geometry, and overview violations detectable from
  compact persisted facts;
- proof that normal publication validation neither opens canonical point
  Parquet nor reads complete Zarr point arrays;
- a compact valid catalog whose `value_tiles` records cross validation batch
  boundaries;
- exact compact cross-index corruption in `value_id`, `manifest_index`, and
  `n_points`, including corruption that preserves aggregate totals;
- manifest-derived bucket descriptors disagreeing with tile coordinates,
  offsets, local indexes, hashing, or physical root attributes;
- proof, through guarded/missing payload shards and guarded source readers,
  that normal validation opens payload metadata but never decodes point rows or
  opens canonical point Parquet.

Use real small Zarr v3 buckets for storage, missing-shard, codec, and lifecycle
claims. Pure geometry, hierarchy, ordering, and accounting helpers may use
small immutable arrays. Do not mock Zarr behavior that is central to the
contract. The low-level `_validate_bucket` tests remain the installed format
coverage used by the developer script; the global exhaustive orchestration is
not duplicated as a production-package unit-test contract.

#### Gate Z7: full-Xenium normal publication validation

Run the mandatory normal validator once against the retained current-format Z6
generation:

```text
/Users/arne.defauw/VIB/DATA/test_data/
  sdata_xenium_full_data_core.transcripts-cache-workspace/
    z6-20260818-current/
```

Record total time, baseline/peak/incremental RSS, largest compact level and
per-level comparison times, bucket and range records scanned, Zarr objects and
bytes read where observable, and closure of every handle. Confirm independently
that no `location`, point-level `value_id`, or `point_id` chunk is decoded; no
canonical Parquet data page is opened; no scratch or output artifact is
created; and the generation's bucket and catalog filesystem inventories, file
sizes, and modification times remain unchanged.

This gate evaluates the normal publication tier only. Do not repeat the
166.03-second full source-equivalence scan merely to accept Z7: the earlier Z3
gate remains evidence for the exhaustive algorithm, while focused corruption
tests qualify the installed storage primitives it composes. A new full-Xenium
exhaustive script run is opt-in for a format/algorithm change, release
qualification, or suspected corruption.

The accepted current-tree run used the retained Z6 generation and completed in
25.74 seconds. It reopened 108 physical buckets across nine levels and compared
29,787,508 compact sparse-range records; Exact was the largest individual
workspace at 14,790,090 range records. Process RSS was 169,541,632 bytes before
the measured call and peaked at 907,345,920 bytes, for an incremental peak of
737,804,288 bytes. The before/after filesystem inventory was identical across
10,940 entries, 7,059 files, and 1,690,639,035 file bytes. Focused tests with a
missing point-payload shard and with removed canonical Parquet files establish
that the normal path validates payload metadata without decoding point rows and
does not open the source. The machine-readable run report is retained at:

```text
/Users/arne.defauw/VIB/DATA/test_data/
  sdata_xenium_full_data_core.transcripts-cache-workspace/
    reports/gate-z7-20260818.json
```

#### Exit criteria

- corruption at every storage or cross-index layer fails closed in the
  validation tier that owns that semantic check;
- normal publication validation is memory bounded and avoids complete point and
  source scans;
- exhaustive engineering validation remains disk-bounded, opt-in, and outside
  the installed package;
- a successful staging result is safe to publish after the final source guard.

### Slice Z8: compose the guarded builder and publication

**Status:** implemented with focused verification on 2026-08-18. The full
Xenium acceptance and interactive-read evaluation remain the explicit Z9 gate.

#### Goal

Expose one isolated candidate builder that creates only complete Zarr-backed
generations. Z8 is a lifecycle coordinator over the already implemented Z1--Z7
primitives; it does not introduce a new point format, sampler, catalog, or
validation algorithm.

Keep the builder private until Z15 decides whether to adopt this architecture.
Add:

```text
src/napari_harpy/core/multi_scale_cache_points_zarr/
  builder.py

tests/multi_scale_cache_points_zarr/
  test_builder.py
```

Do not expose a backend selector through the existing public API and do not
import or invoke the existing Parquet-backed writer package.

#### Builder API and configuration

Implement one coordinator shaped as:

```text
_build_points_cache_zarr(
    validated,
    *,
    output_path,
    temporary_directory_root,
    config,
) -> Path
```

`validated` is the existing canonical `ValidatedPointsSource`. `output_path`
is an explicit local `Path`; integration may later derive the isolated default
`points/<points_name>/transcripts_vis_zarr`, but this candidate builder does
not infer or select a backend. `temporary_directory_root` is an existing
caller-owned local directory, separate from staging and published output. A
successful call returns the exact published `output_path`.

Define one frozen `_PointsCacheBuilderConfig` containing:

```text
leaf_tile_size: int
overview_point_budget: int
dask_worker_count: int
zarr_settings: _ZarrWriteSettings
catalog_settings: _CatalogWriteSettings
max_open_exact_readers: int | None = None
max_open_finer_readers: int | None = None
```

The logical tile size, overview budget, and Exact worker count remain explicit.
The builder derives `_ExactWriterConfig`, `_BridgeWriterConfig`, and
`_SpatialWriterConfig` so callers cannot accidentally provide different Zarr
settings to different levels. Use the current evaluated defaults for physical
storage:

```text
point_chunk_rows   = 4,096
point_shard_rows   = 131,072
range_chunk_rows   = 8,192
range_shard_rows   = 131,072
codec_id           = zstd-v1
```

`_CatalogWriteSettings()` retains its current catalog defaults. Reader bounds
default to `None`, preserving the measured all-input-buckets metadata-cache
policy. Validate the complete coordinator configuration before acquiring any
output ownership or creating a staging directory.

#### Metadata-only source guards

Add one canonical reusable source-validation helper beside the existing source
validation code:

```text
_require_parquet_source_unchanged(validated) -> None
```

This is the only narrow Z8 addition outside the isolated candidate package. It
reuses `_read_parquet_source_inventory()` and the versioned source-signature
builder; it does not decode a Parquet data page. Require the supplied object to
use the supported signature method, reconstruct a fresh deterministic metadata
inventory, calculate its signature, and compare it with
`validated.source_signature`. A mismatch raises a dedicated
`PointsSourceValidationError` code identifying that the source changed after
content validation.

The signature already covers the selected element and columns, normalized
selected schema, relative file inventory, file sizes and modification times,
file and row-group counts, compressed row-group sizes, and total rows. Do not
duplicate those fields as a second coordinator-owned comparison contract.

Call the guard twice:

```text
before planning or staging
        and
after independent staged validation, immediately before marking the root complete
```

The initial guard rejects an already-stale `ValidatedPointsSource`. The final
guard detects a source change during construction. Exact row-group tasks retain
their existing decoded-row checks as a third, local read-boundary defense.

#### Output preflight, lock, and staging ownership

Before construction, require:

- exact `Path` inputs and an existing local `output_path.parent`;
- an existing caller-owned `temporary_directory_root`;
- output, staging, and temporary roots to be separate directory trees;
- `output_path` not to be a symbolic link;
- no unresolved parent creation, cross-filesystem copy, or remote-store
  publication.

Serialize builders targeting the same output through a non-blocking,
platform-aware inter-process lock on one sibling coordination path:

```text
<output-name>.build-lock
```

The lock and generation UUID have separate responsibilities:

```text
cache_generation_id / staging UUID
  -> identifies one generation and keeps concurrent staging trees distinct

<output-name>.build-lock
  -> grants one builder exclusive ownership of the final output path
```

A unique staging UUID alone does not serialize publication. Without the lock,
two builders could safely construct different staging generations and then
race while installing, replacing, or restoring the same `output_path`. The
lock is therefore required for a first build as well as replacement of an
existing cache. Holding it for the complete operation also prevents two
expensive generations for the same output from being built concurrently only
for one to supersede the other.

Acquire the lock before the first source guard and retain it through
publication and cleanup. Declare `filelock>=3.20.1` as a direct dependency
rather than relying on its transitive presence, and use `FileLock` with an
immediate timeout. Failure to acquire the active lock reports the coordination
path and aborts before planning or staging. The platform lock is released on
ordinary exit and when the holding process terminates unexpectedly.

Preserve this distinction in the implementation: the lock helper or context
manager must have a docstring explaining that the UUID provides generation
identity and staging-path uniqueness, whereas the sibling lock coordinates
exclusive ownership of the final publication path. The `.build-lock` pathname
is only a coordination object and may remain after active ownership is
released. Its presence is therefore not evidence that a builder is running;
only a failed non-blocking acquisition establishes contention. Do not unlink
the path independently of `FileLock`, because pathname removal can race a new
holder using that same coordination object.

While holding the lock, preflight an existing output before creating staging:

- an absent output permits a first build;
- a regular file, symlink, unsupported cache, or incomplete directory is
  rejected without mutation;
- a replaceable directory must have root `publication_state = "complete"` and a
  valid parsed root `cache_generation_id`;
- opening its root must validate the supported attributes, hierarchy, and
  catalog array layouts, but need not repeat the complete compact-range scan;
- suspected incomplete or foreign directories are never silently deleted or
  repaired by the builder.

Create one canonical lowercase UUID and one absent same-parent staging sibling:

```text
<output-name>.staging-<cache_generation_id>
```

Create the staging directory with exclusive semantics. The same parent keeps
publication on one local filesystem. Pass the UUID unchanged to Z6 as
`cache_generation_id`. Z8 owns the entire staging generation and removes it
recursively if it still exists on every failed or successful unwind. Individual
writers continue to own active partial buckets, readers, and private Dask
scratch while they are running.

#### Required flow

```text
ValidatedPointsSource
  -> acquire output build lock and preflight existing output
  -> fresh metadata-only source guard
  -> fresh Zarr-cache build plan
  -> unique sibling staging generation
  -> Exact
  -> Bridge
  -> all spatial/overview levels
  -> cache-root attributes and catalog arrays
  -> independent staged validation
  -> final fresh source guard
  -> set root publication_state = "complete"
  -> generation-atomic staged replacement
  -> release output build lock
```

The construction branch is explicit:

```text
Exact-only plan
  -> level_results = (exact_result,)

multilevel plan
  -> Exact
  -> Bridge
  -> zero or more Spatial levels
  -> level_results = (exact_result, bridge_result, *spatial_results)
```

Never call the Bridge or Spatial entry points for an Exact-only plan. Build the
catalog from the complete ordered result tuple, then release all construction
graphs, readers, and writer scopes before calling `_validate_staged_cache()`.
The normal validator receives only the staging path and reconstructs its own
facts; Z8 must not pass writer results or the build plan into validation.

#### Publication-state attribute

Root `publication_state` is the Zarr-native publication signal. Z6 initially
writes `"staging"`. Z8 changes it to `"complete"` as the final content mutation
inside a successfully validated staging generation, only after staged
validation, closure of all storage and Dask work, and the successful final
source guard. No catalog, Zarr metadata, or payload write may follow it.

The root `cache_generation_id` remains the single canonical generation UUID;
publication state does not duplicate it. A completion preflight parses the root
attributes and requires both a supported cache contract and
`publication_state = "complete"`. A directory name or lock pathname alone is
not evidence of completion. Keeping the signal in `zarr.json` avoids an
unrecognized root sidecar and the warnings generic Zarr hierarchy traversal can
otherwise produce.

Z8 supplies this small completed-generation preflight for replacement safety.
The runtime acceptance reader and its incomplete-publication-state rejection
remain Z9 work.

#### Publication semantics

Use generation-atomic staged replacement, borrowing the proven lifecycle idea
from the existing cache while implementing it independently in the candidate
package:

```text
new staging is complete
        -> existing output renamed to unique sibling backup, when present
        -> staging renamed to output
        -> backup removed after the install commits
```

Every rename is a same-parent local directory rename. A first publication into
an absent output is one atomic rename. Portable replacement of an existing
nonempty directory cannot provide uninterrupted old-or-new visibility through
one POSIX rename: there is a short interval between the backup and install
renames when `output_path` is absent. Therefore do not claim uninterrupted
atomic replacement. The guaranteed property is that no partial or mixed
generation is ever installed at `output_path`.

If installation of staging fails after moving the old output, immediately
restore the backup before propagating the error. If rollback itself fails,
retain the complete old generation at the reported backup path, retain or clean
the unpublished staging as safely possible, and report both filesystem errors;
do not delete either recoverable generation. Once staging has successfully
become `output_path`, publication is committed. A later backup-cleanup failure
must not remove or roll back the new complete output; leave the complete backup
with a clear cleanup warning.

This direct-directory contract is preferred for the isolated experiment. True
uninterrupted atomic switching would require a generation pointer, symlink, or
platform-specific exchange operation and would change the cache-root contract;
do not introduce that architecture implicitly in Z8.

#### Failure and cleanup contract

- any failure before publication leaves an existing completed output byte-for-
  byte and path-for-path unchanged;
- a failed staging generation is never repaired or resumed and is removed by
  the coordinator after all nested writers unwind;
- the caller-owned temporary root remains, while Exact's private shuffle child
  and all candidate-owned scratch are absent after return or error;
- all Dask computations are synchronous within the Exact writer, every reader
  cache and Zarr store exits before validation, and validation closes every
  reopened handle before the final guard and rename;
- the root publication state never becomes `"complete"` on a construction,
  catalog, validation, or final-source-guard failure;
- the canonical Parquet source is opened read-only and is never mutated;
- cleanup acts only on the exact uniquely created staging and backup paths
  owned by this invocation; lock ownership is released through `FileLock`;
- no broad glob, unresolved environment variable, or caller-owned directory is
  a cleanup target.

Derived cache data remain regenerable, but unknown existing outputs fail closed.
Z8 does not resume partial output, publish in place, update a public backend
selector, or call the existing builder.

#### Focused tests

- builder configuration and path validation before any filesystem mutation;
- metadata-only source guard accepting an unchanged source and rejecting file,
  row-group, schema, size, timestamp, and inventory changes without decoding
  data pages;
- first Exact-only build and one complete small multilevel build through the
  real Zarr writers, catalog, staged validator, publication-state transition,
  and rename;
- successful replacement of a completed generation, UUID agreement between the
  root and staging identity, and normal backup removal;
- rejection and preservation of an existing file, symlink, incomplete
  directory, foreign directory, malformed root publication state, and invalid
  root generation UUID;
- competing held-lock rejection and successful lock reacquisition after both
  ordinary return and failure; tests must not infer ownership from pathname
  presence;
- injected failure at the initial guard, Exact, Bridge, Spatial, catalog,
  staged validation, final guard, publication-state update, backup rename, and
  staging install rename boundaries;
- rollback restoring the old completed generation after install failure;
- failure cleanup removing only the owned staging generation and private Dask
  scratch while preserving the caller temporary root;
- proof that all readers, stores, Dask work, and memory maps are closed before
  validation and publication;
- before/after canonical source inventory equality and absence of source writes;
- absence of a completed root publication state in every unpublished failure
  tree and absence of mixed Parquet/Zarr derived payloads.

Use real small Parquet sources and Zarr generations for both successful flows.
Inject failures by replacing coordinator phase callables or rename boundaries;
do not mock Zarr behavior central to a successful build. Publication helpers
may be tested independently with small completed directory trees.

#### Exit criteria

- after an ordinary return, the isolated output path contains one independently
  validated generation with root `publication_state = "complete"` and the UUID
  assigned to its unique staging identity;
- after an ordinary pre-publication failure, output is absent or the previous
  completed generation remains unchanged, with no staging artifact and with
  the publication lock available for immediate reacquisition;
- no partial or mixed generation is installed at the public output path;
- publication and rollback guarantees are named accurately rather than
  claiming an unavailable portable one-rename replacement of a nonempty tree;
- the source guard detects metadata changes before staging and immediately
  before completion without repeating the content scan;
- the end-to-end builder never imports or invokes existing derived-cache writer
  code and exposes no public backend selector.

### Slice Z9: implement the acceptance reader and Xenium evaluation

**Status:** implemented and evaluated on the full 136,578,750-point Xenium
dataset on 2026-08-19. The retained generation and machine-readable report are
listed under **Gate Z9 results** below.

#### Goal

Measure whether the physical design provides useful selected-value access while
keeping normal all-values navigation practical.

This slice implements a small backend-level reader, not the complete Phase 2
napari store. It also runs the first complete full-Xenium build through the Z8
coordinator and retains that published generation for repeatable read
evaluation. Do not introduce a public backend selector or a Parquet fallback.

#### Files and dependency boundary

Create:

```text
src/napari_harpy/core/multi_scale_cache_points_zarr/reader.py
tests/multi_scale_cache_points_zarr/test_reader.py
scripts/benchmark_multi_scale_cache_points_zarr_acceptance.py
```

`reader.py` owns the high-level cache, tile, viewport, and level-selection
contracts. It composes the existing strict `_CatalogReader`, `_BucketReader`,
and `_BucketReaderCache`; it does not duplicate Zarr schema parsing or import
the existing Parquet-backed cache reader. Low-level visualization reads may be
added to `storage/bucket_reader.py`, but existing construction-facing
`_PointPayload` methods and writer behavior remain unchanged.

The benchmark script owns timing, RSS, filesystem-object accounting, scenario
selection, and retained-run reporting. No benchmark-only counters, machine
paths, or pass/fail timings become cache-format metadata.

#### Reader lifecycle and publication contract

Implement one private `_PointsCacheReader` context manager. On entry it:

1. opens the cache through `_CatalogReader` and therefore validates the frozen
   root, hierarchy, and catalog array layouts;
2. requires root `publication_state = "complete"` and rejects a staging or
   unsupported generation;
3. enters one `_BucketReaderCache` whose capacity is the sum of the physical
   bucket counts declared by all levels;
4. loads only the compact runtime indexes described below.

Bucket readers are admitted lazily when a request first touches their bucket;
the acceptance reader does not eagerly open every store. Once opened, retain a
reader for the lifetime of `_PointsCacheReader`. This keeps repeated panning and
selection changes from reopening bucket metadata while bounding retained
readers by the cache's finite physical bucket inventory. Do not expose a
speculative smaller `max_open_readers` setting in Z9. Reconsider a stricter LRU
bound only if the full-Xenium evaluation demonstrates a material handle or
metadata-resource problem.

Reader identity is the composite key `(level, bucket_id)`: bucket zero at Exact
is distinct from bucket zero at Bridge or a Spatial level. The one reader cache
therefore retains lazily opened readers from all levels together, up to the sum
of their declared bucket counts. This supports metadata reuse when navigation
changes LOD and later returns to a previously visited level:

```text
Exact buckets opened
    -> zoom out and open Spatial buckets
    -> zoom in and reuse the retained Exact readers
```

Only the entered store/group/array reader objects and their initialized Zarr
metadata are retained. Point payloads and decoded chunks are not stored in
`_BucketReaderCache`; operating-system and codec caching remain separate.

Opening an accepted published cache must not call
`_CatalogReader.validate_contents()` or `_validate_staged_cache()`. Publication
already ran the independent complete-generation validator; replaying its full
compact reconciliation on every viewer open would add seconds of unnecessary
startup work. Runtime reads still fail closed on malformed root/layout metadata
and on any bucket or sparse-range inconsistency encountered while reading.

The runtime trust boundary is:

```text
cache construction
    -> build every level and catalog
    -> run independent staged validation once
    -> set publication_state = "complete"
    -> publish

viewer runtime
    -> require publication_state = "complete"
    -> parse supported root, catalog, and bucket layouts
    -> perform only the requested catalog/index lookup
    -> read only the requested point chunks
```

Once published, the cache's globally reconciled semantic contents are trusted.
Neither reader entry nor any tile, viewport, selection, panning, or LOD request
may revalidate the canonical Parquet source, call `_validate_staged_cache()`,
call `_CatalogReader.validate_contents()`, reconcile the complete manifest and
`value_tiles`, scan all bucket ranges, or scan a complete point payload.

“Trusted” does not disable cheap defensive checks local to data being opened or
requested. Runtime readers still require supported versions and array layouts,
validate requested identifiers and sparse-range bounds, and fail on a missing
chunk or other inconsistency they encounter. This work must remain proportional
to reader metadata plus the requested indexes, buckets, and point rows, never to
the complete cache.

The reader closes the catalog store and every retained bucket reader on normal
exit and on any failed open or read. It is entered once and is not thread-safe;
Phase 2 may later define per-worker or synchronized ownership.

#### Runtime metadata residency

At reader entry, materialize these compact arrays and build immutable lookup
state:

```text
manifest/level_indptr
manifest/bucket_id
manifest/bucket_tile_index
manifest/tile_x
manifest/tile_y
manifest/n_points
value_tiles/indptr
values/n_points
```

Also retain root level metadata and `value_names`, and build a mapping from
`(level, tile_x, tile_y)` to the implicit manifest row. The complete manifest
and two-dimensional pointer table are small relative to point payloads and make
tile and viewport planning independent of bucket opens.

Do not materialize complete `value_tiles/manifest_index` or
`value_tiles/n_points`; the Xenium generation has tens of millions of those
rows. Read only the half-open slices selected by `value_tiles/indptr`.

#### Input and result contracts

Represent a spatial viewport with a small immutable `_IntrinsicViewport` using
finite intrinsic-source `x_min`, `y_min`, `x_max`, and `y_max`. Bounds are
half-open, must have positive width and height, and are clipped to the cache
geometry. A disjoint viewport has no positive tiles.

`_IntrinsicViewport` is deliberately a storage-neutral cache-reader contract,
not a napari viewport or camera object. Napari does not pass such a rectangle to
an external point backend. The later Phase 2 napari adapter is responsible for
observing the camera, canvas size, displayed dimensions, and relevant layer or
SpatialData transforms; calculating the visible world-coordinate rectangle;
and transforming that rectangle back into the transcript source's intrinsic
coordinates before calling this reader:

```text
napari camera + canvas size + displayed axes
                    -> napari-specific adapter
                    -> visible world rectangle
                    -> inverse layer/SpatialData transform
                    -> _IntrinsicViewport(x_min, y_min, x_max, y_max)
                    -> read_viewport(level, viewport, value_ids=selected_value_ids)
```

Consequently, Z9 tests and benchmarks construct `_IntrinsicViewport` directly
and do not need a napari viewer. The acceptance reader must not import napari or
infer a viewport from camera state; that integration remains a Phase 2 concern.

When provided, value IDs are a one-dimensional C-contiguous `uint32` array,
strictly increasing, unique, nonempty, and inside the serialized value
vocabulary. The reader exposes the immutable `value_names` tuple so callers can
map labels to their implicit IDs; it does not repeat source-value normalization
at read time.

Return one immutable `_TileReadResult` per positive tile. It contains:

```text
level, tile_x, tile_y, tile_size
location       (N, 2) float32, stored tile-relative (x, y) coordinates
value_id       (N,)   uint32
```

Arrays are aligned, C-contiguous, read-only, and nonempty. Tile identity and the
root origin provide the unambiguous intrinsic-coordinate reconstruction:

```text
x = root.x_origin + tile_x * tile_size + location[:, 0]
y = root.y_origin + tile_y * tile_size + location[:, 1]
```

This result is deliberately separate from construction `_PointPayload`, whose
mandatory `point_id` and four-array construction invariants must not be weakened
for display. The acceptance reader never returns or reads point IDs. A strict
bucket open may validate `point_id` array metadata as part of the frozen bucket
layout, but no viewer tile, viewport, selection, panning, or LOD request may
slice or decode a `point_id` payload chunk. Construction, validation, and opt-in
diagnostics continue to use the existing mandatory-ID bucket-reader methods.

Viewport methods return one immutable `_ViewportReadResult` containing the
selected level and the tuple of positive `_TileReadResult` objects in manifest
order. `select_level()` returns an immutable `_LevelSelection` containing the
chosen level, estimated point count, positive visible-tile count, and
`within_budget` flag rather than returning an unexplained integer. For a
value-filtered request it also returns read-only sorted `omitted_value_ids`:
requested values with a positive Exact visible count and zero count at the
selected level. The field is an empty `uint32` array when none were omitted and
`None` when no value filter was supplied. The LOD evidence is part of that
decision contract; physical-read diagnostics are not part of tile or viewport
payloads.

#### Reader operations

```text
read_tile(level, tile_x, tile_y, *, value_ids=None)
read_viewport(level, viewport, *, value_ids=None)
select_level(viewport, point_budget, *, value_ids=None)
```

Use one high-level operation per spatial request rather than exposing the
physical all-row versus sparse-range distinction in method names. For all three
operations, `value_ids=None` means all values. A supplied `value_ids` array
follows the strict contract above and restricts the result or estimate to those
values. Keep `value_ids` keyword-only so a caller cannot pass an unexplained
positional array:

```python
reader.read_viewport(level, viewport)
reader.read_viewport(level, viewport, value_ids=selected_value_ids)
```

`read_tile()` and `read_viewport()` dispatch internally through
`_BucketReader.read_display_payload()` to the contiguous all-values path or sparse
value-range path. `_BucketReader.read_construction_payload()` remains the distinct
construction-facing operation because Bridge and Spatial require mandatory
point IDs; there is no selected construction consumer.

`point_budget` is a positive caller-computed effective limit, not a napari
canvas policy embedded in the cache reader. Z9 accepts and tests that one number
without importing napari or introducing separate maximum-render and
screen-density parameters. The later Phase 2 napari adapter owns the dynamic
calculation:

```text
hard maximum render points
        +
canvas size, projected marker size, and visual-density policy
        -> screen-density budget
        -> min(hard maximum, screen-density budget)
        -> point_budget
        -> select_level(viewport, point_budget, value_ids=...)
```

This keeps screen pixels and display policy outside the storage-neutral reader
while allowing a fully zoomed-out request to target materially fewer points
than the hard rendering maximum. Do not add a second fixed reader budget in Z9.

Do not conflate the viewport's spatial bounds with the bucket's sparse value
ranges. They address two different stages of a request:

```text
viewport x_min, y_min, x_max, y_max
    -> identify intersecting logical tiles

selected value_ids
    -> identify sparse point-row intervals inside those tiles
```

A logical tile is the cache's smallest spatially indexed unit. An all-values
request resolves `tile_offset[i:i + 2]` and reads every stored point row for an
intersecting positive tile. A selected-value request instead resolves that
tile's `ranges/tile_indptr` records and reads only the `row_start:row_stop`
intervals belonging to requested values. Thus “complete tile” means all rows of
one logical tile, not all rows of its physical bucket.

There is no additional coordinate index inside a tile: its point rows are
value-major rather than ordered by `x` or `y`. A viewport cutting through part
of a tile therefore cannot skip the points outside the exact viewport boundary
without first reading and filtering the applicable tile rows. Z9 deliberately
does not add that second spatial index or point-level clipping step.

Logical tiles must also not be confused with Zarr chunks. A tile defines the
smallest spatial lookup and reuse boundary, whereas a chunk is the smallest
physical decoding unit. A tile can span several chunks, and a boundary chunk
can contain rows adjacent to the requested logical interval. This physical
read amplification may be investigated with opt-in diagnostics, but is not
returned with normal display payloads.

Direct tile reads return `_TileReadResult | None`; an empty manifest tile or a
tile with none of the selected values returns `None`. Viewport reads return
positive tile results in deterministic manifest order `(tile_y, tile_x)` and
never create placeholder results for empty grid cells. They materialize only
the selected level and viewport, never a complete cache level. The caller uses
`select_level()` before normal viewport rendering and follows the budget-first
policy below; explicit-level methods remain available for
correctness tests and physical benchmarks.

A viewport selects logical tiles whose half-open spatial extent intersects the
clipped viewport. It returns all applicable all-values or value-filtered rows
stored in those tiles; it does not apply a second point-coordinate clip at the
viewport edge. The napari layer can clip rendered points, while tile reuse
during small pans remains possible. Consequently, LOD estimates use complete
positive-tile rows and may conservatively exceed the number of points strictly
inside the viewport rectangle.

`select_level()` follows the same optional `value_ids` contract as the read
methods. It chooses the finest serialized level whose estimated visible point
count is at most the positive
`point_budget`:

- all-values estimates sum `manifest/n_points` for intersecting manifest rows;
- value-filtered estimates sum the corresponding positive `value_tiles/n_points`
  entries after viewport intersection;
- unique selected values are disjoint point categories, so their counts may be
  summed without point-level deduplication;
- a disjoint viewport selects Exact and produces no tile results;
- when no level fits for an all-values request, select the terminal overview
  and set `within_budget` from its actual estimate rather than assuming the
  construction overview limit also fits this runtime budget.

The construction and runtime limits are different contracts. For example, a
cache built with `overview_point_budget = 100_000` may have an 82,000-point
terminal overview, while a small canvas produces `point_budget = 25_000`:

```text
level                 estimated viewport rows    fits point_budget=25,000
Exact                              12,000,000     no
Bridge                              2,000,000     no
Spatial                               500,000     no
terminal overview                       82,000     no

selected level      = terminal overview
estimated rows      = 82,000
within_budget       = False
```

The terminal overview is still the smallest available all-values
representation, but returning it does not mean that `select_level()` satisfied
the caller's effective budget. If its estimate were 18,000 instead,
`within_budget` would be `True`. Z9 reports this state truthfully; it does not
thin the 82,000 rows at read time.

Use the same budget-first policy for all-values and value-filtered requests:

1. evaluate serialized levels incrementally from Exact toward the coarsest
   level;
2. for a value-filtered request, sum visible points only for requested values
   represented at that level;
3. return the first level whose estimate is at most `point_budget`, with
   `within_budget = True`;
4. do not make a sampled level ineligible because it omits one or more requested
   values;
5. if no serialized level fits, return the coarsest level with
   `within_budget = False`.

This makes the render budget authoritative for a multi-value request: one rare
value lost during sampling cannot force every other selected value back to
Exact. The accepted trade-off is that a coarse LOD may omit selected values.
For a rare value selected alone, Exact still wins whenever its visible Exact
count fits the budget. If a sampled level has zero rows for every requested
value, its zero estimate is a valid fit; level selection does not replace it
with an unbounded Exact read.

Report that trade-off without changing eligibility. `_LevelSelection` stores
the sorted IDs that were visible at Exact but have zero visible count at the
selected level. The viewer can derive their count or map them through the
canonical value table for optional messaging. Values already absent from the
Exact viewport are not reported as LOD omissions, and computing this evidence
must not trigger additional catalog reads.

Return as soon as the first fit is known, so the ordinary path does not slice
`value_tiles` or construct per-tile value selections for coarser levels that
cannot affect the answer. A value count need not be monotonic across levels:
complete coarser tiles have larger spatial footprints and may include an
existing requested value outside the Exact tiles intersecting the viewport.

Level selection reads only catalog metadata and must not open a bucket or point
payload array. Exact remains eligible when its visible count fits; Bridge and
Spatial are not preferred merely because they are sampled.

#### Tile discovery, bucket grouping, and ordering

The high-level value-filtered read path owns the complete two-index lookup
below. Keep this scheme in its implementation docstring so the distinction
between cache-wide tile discovery and bucket-local point-row resolution remains
explicit:

```text
selected value and level
        ↓
value_tiles/indptr
        ↓
manifest indexes of tiles containing the value
        ↓
manifest
        ↓
bucket_id + bucket_tile_index
        ↓
ranges/tile_indptr
        ↓
range record for the selected value
        ↓
row_start + row_count
        ↓
exact point rows
```

The `value_tiles` half prunes tiles and buckets before point payloads are read.
The bucket-range half starts only after a manifest tile has been resolved and
maps that tile/value pair to half-open rows in the aligned `location` and
point-level `value_id` arrays. `_BucketReader.read_display_payload()` owns only
this second half; the high-level reader must not collapse the two
responsibilities into one undocumented lookup.

For each selected `(level, value_id)`, read its exact `value_tiles` slice,
intersect the strictly ordered manifest indexes with visible manifest rows, and
accumulate the requested value IDs per positive tile. Deduplicate manifest rows
across values, group positive tiles by `(level, bucket_id)`, and process each
bucket as one contiguous request group. This guarantees at most one bucket open
per request even when several selected values or tiles resolve to it; the
reader-scoped cache additionally reuses that entered reader across later
requests.

All-values viewport reads discover positive tiles directly from the resident
manifest lookup and do not touch `value_tiles`. Both all-values and
value-filtered paths preserve manifest tile order in returned results and
bucket-local value-major, then `point_id`, row order inside each tile.

The reader must therefore:

- use `value_tiles/indptr` and `manifest_index` to prune zero-count tiles;
- group positive visible tiles by bucket and open each bucket once per request;
- use sparse range records instead of scanning point-level values;
- read only intersecting `location` and point-level `value_id` chunks;
- coalesce adjacent ranges where practical;
- distinguish metadata/index work from point-payload work.

#### Visualization bucket primitive

Factor the aligned point-array reading inside `_BucketReader` so the acceptance
path reads only `location` and `value_id`. Preserve the existing
`read_construction_payload()` contract as the mandatory-ID operation used by
Bridge, Spatial, validation, diagnostics, and their tests. `read_display_payload()`
owns both complete and selected display reads without point IDs. Shared private
planning continues to own sparse-range lookup, exact intervals, inner-chunk
deduplication, and coalesced minimal envelopes.

#### Minimal display payloads

Keep physical-read accounting outside the production result contract.
`_PointDisplayPayload` contains only `location` and `value_id`; the private read
plan contains only the exact output intervals and coalesced physical blocks
needed to perform the read. `_TileReadResult` and `_ViewportReadResult` do not
carry benchmark statistics.

Logical point and positive-tile counts remain directly derivable from the
returned arrays and tile tuple. Timing, chunk, shard, and decoded-row
investigations belong in opt-in benchmark or profiling instrumentation. The
original Z9 evaluation used such information to assess the format, but normal
viewer requests must not calculate and propagate it when no product consumer
requires it. The reader cache retains `open_reader_count` for its bounded
resource-lifetime contract, but does not maintain cumulative hit or miss
counters.

#### Focused tests

Add focused real-Zarr tests for:

- rejection of staging publication state, unsupported roots, invalid levels,
  tiles, viewports, point budgets, and value-ID arrays;
- all-values and value-filtered `read_tile()` calls, empty logical tiles, absent
  requested values, multiple values, deterministic row order, and coordinate
  reconstruction;
- all-values and value-filtered `read_viewport()` calls across sparse rows,
  bucket boundaries, and level boundaries without duplicate tiles or points;
- exact `value_tiles` pruning, including proof that a zero-count tile's bucket
  and point payload are not opened;
- several values and tiles sharing one bucket, with one open per request and
  reader-cache reuse across requests, lazy admission, and no eager opening of
  untouched buckets;
- all-values and value-filtered visualization reads succeeding without a point-ID
  payload-chunk read, including when such a chunk is deliberately unavailable;
- Exact, Bridge, Spatial, terminal-overview, all-values, selected-value, and
  no-level-fits LOD decisions from catalog counts only;
- an all-values `point_budget` below the terminal overview count selecting that
  terminal level with `within_budget = False` rather than claiming a fit;
- a selected value that exceeds the budget at Exact and disappears at the next
  sampled level, proving that the zero-count sampled level is a valid fit;
- multiple requested values where one disappears while another remains,
  proving that represented values alone determine the level estimate;
- requested values absent at Exact selecting an empty Exact result rather than
  points introduced only by a coarser tile's larger intersecting footprint;
- exact returned rows for complete, selected, coalesced, and partial-final-chunk
  reads;
- deterministic closure after successful reads and injected catalog, bucket,
  sparse-range, and payload failures.

Use small real buckets and catalogs for physical claims. Do not add timing,
compressed-byte, or operating-system cache thresholds to unit tests.

#### Full-Xenium scenarios

Run one current-tree baseline build through `_build_points_cache_zarr()` with
`TARGET_POINTS_PER_BUCKET = 2_000_000` and the retained
`overview_point_budget = 100_000` construction policy. Do not lower the
construction overview budget merely to anticipate a Phase 2 canvas-density
policy. Time source content validation separately; the measured build interval
begins with the validated source and includes planning, every level writer,
catalog construction, normal staged validation, the final source guard,
publication-state transition, and directory publication. Retain the published
cache and machine-readable report under the existing sibling workspace rather
than modifying the canonical SpatialData source:

```text
/Users/arne.defauw/VIB/DATA/test_data/
  sdata_xenium_full_data_core.transcripts-cache-workspace/
    z9-current/
      transcripts_vis_zarr/
    reports/
      gate-z9-current.json
```

Remove or replace only the explicitly named candidate output through the Z8
builder contract; do not delete the workspace broadly. Record whether the run
created or replaced that generation and its `cache_generation_id`. Subsequent
read scenarios reuse this exact published output without rebuilding it.

First establish reader correctness on representative Exact, Bridge, Spatial,
and overview tiles by comparing a value-filtered read with an in-memory value
filter of the corresponding all-values read, comparing viewport results with
the ordered union of their tile results, and reconciling returned counts with
manifest and `value_tiles` counts. Do not repeat the complete canonical-source
equivalence scan merely to qualify runtime reads.

Then measure at Exact, Bridge, representative spatial levels, and overview:

- dense and average all-values tiles;
- all-values viewports at several zoom levels;
- common, median, rare-localized, and rare-distributed values;
- several selected values with adjacent and separated ranges;
- repeated selection changes with cold and warm caches;
- panning with overlapping tiles, buckets, and chunks.

For this gate, **application-cold** means a newly entered `_PointsCacheReader`
with an empty reader-scoped bucket cache. **Application-warm** means repeating
the request through the same entered reader. Do not claim that either state
clears or controls the operating-system page cache, Zarr/codec internals, or
filesystem cache; record that limitation explicitly. Repeated measurements
report their individual observations or a stated summary, without fixed
pass/fail latency thresholds.

Exercise automatic `select_level()` for realistic all-values and selected-value
viewports, while also using explicit-level methods to expose the physical Exact,
Bridge, Spatial, and overview read behavior. Include a rare-distributed value
that is positive in many visible tiles so the limits of `value_tiles` pruning
are measured rather than inferred. Also identify at least one value/viewport
combination that disappears at a sampled level, when present in the retained
generation, and verify that level selection treats its zero count according to
the budget-first policy. If the dataset contains no such combination, record
that fact rather than manufacturing one for the full-scale timing run; the
focused test still freezes the behavior.

Exercise several caller-supplied `point_budget` values below and at the retained
100,000-point overview limit, including budgets derived from documented example
canvas sizes and screen-space densities. Report when the terminal overview is
the best available all-values level but still has `within_budget = False`.
These calculations provide evidence for whether a later construction-policy
experiment should lower `overview_point_budget`; Z9 does not change that policy,
claim visual acceptance without a napari integration, or implement runtime
display thinning.

Record:

- source-validation time, complete builder time, baseline/peak/incremental RSS,
  and publication result;
- total bytes and filesystem-object count;
- reader-open latency and resident compact-index bytes;
- all-values tile and viewport latency;
- selected-value latency;
- logical selected rows;
- complete positive-tile rows;
- positive visible tiles and retained bucket-reader handles;
- application-cold and application-warm request timings;
- proof that no acceptance-reader request accesses a `point_id` payload chunk;
- chosen LOD, estimated point count, actual returned point count, and
  `within_budget` decision.

#### Bucket-target decision

Retain `TARGET_POINTS_PER_BUCKET = 2_000_000` through construction and use its
completed Xenium measurements as the baseline physical configuration. Evaluate
`10_000_000` as the single leading alternative rather than opening an unbounded
parameter sweep. Run that alternative only as a separately identified Z9
experiment after the complete two-million baseline is accepted as correct; do
not overwrite the retained baseline or change inner chunk/shard sizes at the
same time. The decision must account for both sides of the tradeoff:

- all-values and value-filtered viewport latency, especially the number of distinct
  bucket stores opened for common and rare-distributed values;
- metadata/open-handle work per request;
- Exact and multilevel construction peak RSS with the configured worker count;
- largest materialized shuffled bucket and finalizer duration;
- total object count and storage bytes;
- opt-in inner-chunk amplification diagnostics if the alternative is run.

Choose ten million only if the reduced store-open and metadata work is material
for realistic navigation and its larger construction unit remains practically
memory bounded. Otherwise retain two million. Record the chosen target and
evidence as a versioned construction policy before the format is proposed for
Phase 2. The alternative may use an explicitly isolated experiment change; do
not generalize the production reader to arbitrary bucket targets or retain two
supported construction policies after the decision. The adopted code and
format profile support only the chosen target and fail closed on the other.

#### Acceptance assessment

Correctness is mandatory. Review build reliability, bounded memory, construction
speed, storage behavior, object count, all-values reads, and value-filtered reads
as one engineering decision without fixed numerical thresholds and without
requiring a Parquet comparison run.

If the evidence supports adoption, record the format and physical settings as
the recommended Phase 2 input. If it does not, record the measured reason and
recommend retaining the existing implementation. Z9 produces the evidence and
recommendation; Z15 owns the explicit architecture-adoption decision and any
follow-up archival or integration plan. Do not add a fallback path in either
case.

#### Gate Z9 results

One current-tree run built and published all nine levels through the Z8
coordinator. The retained artifacts are:

```text
/Users/arne.defauw/VIB/DATA/test_data/
  sdata_xenium_full_data_core.transcripts-cache-workspace/
    z9-current/
      transcripts_vis_zarr/
    reports/
      gate-z9-current.json
```

The run created generation `bebfadbc-bb8b-4858-bd5e-811801a8a837` with the
frozen two-million-point bucket target, 4,096-row inner point chunks,
131,072-row point shards, and a 100,000-point construction overview budget.
Observed build and storage results were:

- source content validation: 2.34 seconds;
- complete guarded build, staged validation, and publication: 243.35 seconds;
- baseline process RSS: 353 MB; peak RSS: 4.33 GB; incremental peak: 3.98 GB;
- published storage: 1,690,639,072 bytes in 7,059 files;
- nine levels, with the terminal overview containing exactly 100,000 points.

Reader entry took 87.4 ms and retained 821,488 bytes of compact NumPy catalog
indexes. Representative all-values tile observations included:

| request | rows | first request | repeat through same reader |
| --- | ---: | ---: | ---: |
| Exact dense tile | 108,598 | 44.3 ms | 11.1 ms |
| Exact average tile | 18,698 | 42.3 ms | 5.15 ms |
| Bridge dense tile | 4,096 | 41.7 ms | 3.71 ms |
| representative spatial L5 tile | 65,536 | 59.1 ms | 10.5 ms |
| terminal overview tile | 100,000 | 38.1 ms | 11.0 ms |

The detailed chunk and shard observations below were captured by the original
acceptance instrumentation. They remain historical evidence, but the later
minimal-payload refactor deliberately removed them from production read
results.

An application-cold selected-value request opened one Exact bucket and returned
3,514 logical rows in 19.4 ms after a separate 56.7 ms reader entry. Repeating
the request through that entered reader took 10.1 ms. It read 6,375 compact
`value_tiles` rows, touched two inner point chunks in one shard, and had an
estimated decoded-row amplification of 2.33. “Cold” and “warm” here describe
only the reader-scoped bucket metadata cache; operating-system, filesystem, and
codec caches were not controlled.

The explicit all-values pan scenarios returned four positive Exact tiles and
360,291 rows in 94.0 ms, then nine positive tiles and 804,187 rows in 190.0 ms.
The second viewport was spatially offset and overlapped the first; the retained
reader reused previously opened bucket metadata while admitting newly touched
buckets.

Sparse lookup was exercised for common, median, rare-localized,
rare-distributed, adjacent, and separated values. The physical omission test
proved that visualization reads succeed when the requested `point_id` payload
shard is unavailable, while construction reads fail, freezing that acceptance
requests slice only `location` and point-level `value_id`.

Catalog-only LOD selection chose the terminal 100,000-point overview for a
full-dataset all-values viewport. It reported `within_budget = True` for a
100,000 runtime budget and `False` for 50,000 and 25,000, rather than claiming
that the terminal level satisfied those smaller screen-derived budgets. For a
common selected value it chose L4 with 78,789 estimated points; median,
rare-distributed, and rare-localized examples fit at Exact.

The generation contains sampled value loss. `FGF9` (`value_id = 1591`) has
per-level counts `[11627, 1011, 552, 285, 154, 76, 43, 22, 0]`. The original Z9
run used the earlier value-preserving policy and returned L7 for a one-point
runtime budget. A later multi-value design review made the budget authoritative:
the current policy selects the empty terminal L8 because its zero represented
rows fit. This deliberate revision prevents one omitted rare value from forcing
all other values in a large selection into an unbounded finer-level read.

The two-million-point bucket target remains the recommendation for this
baseline. Store-open work was visible but practical: one application-cold
selected request opened one bucket, and its repeat saved about 9 ms. The
observed metadata and handle behavior does not justify paying for a second
full-scale ten-million-point build before Z15. Inner-chunk amplification, not
the number of bucket directories alone, dominates the smallest sparse reads;
for example, a single selected row still decodes one 4,096-row inner chunk.

These results satisfy the Z9 engineering gate without fixed latency thresholds:
the builder completed with bounded memory, the published artifact reopened
quickly, normal all-values access remained practical for representative tiles
and viewports, and sparse selection avoided unrelated point rows and point IDs.
The subsequent budget-first selected-value revision is frozen by focused tests.
Z15 still owns the explicit architecture-adoption decision.

#### Exit criteria

- a published cache is built once end to end through Z8 at full Xenium scale;
- the reader rejects incomplete generations and closes all catalog and bucket
  handles deterministically;
- all-values and value-filtered tile and viewport results reconcile with the
  frozen catalog and each other;
- acceptance reads demonstrably avoid point-ID payload access;
- catalog-only LOD selection is correct for all-values and selected-value
  requests;
- value-filtered LOD treats only represented selected rows as budget consumers,
  permits sampled value loss, and reports a coarsest over-budget fallback only
  when no serialized level fits;
- direct sparse-range lookup is demonstrated at full scale;
- its useful and non-useful cases are documented honestly;
- all-values access remains practical for the planned viewer;
- baseline and optional bucket-target evidence are isolated and reproducible;
- one evidence-backed recommendation is ready for the explicit Z15 decision.

### Slice Z10: batch and coalesce multi-value catalog lookup — resolved

#### Goal

Make selected-value planning scale with the compact catalog chunks touched,
rather than issuing two Zarr reads for every `(level, value_id)` pair. Keep this
physical lookup optimization separate from the product policy that decides
whether one level may omit requested values.

The current `_value_filtered_manifest()` loops over every requested value. For
each nonempty value interval at each evaluated level it independently slices
both `value_tiles/manifest_index` and `value_tiles/n_points`. A request for `G`
values over `L` levels can therefore issue up to `2 * G * L` Zarr selections,
even when many intervals occupy the same inner chunks. The retained Xenium
generation demonstrated that this is acceptable for one value but not a
scalable many-value query plan.

#### Lookup plan

For one level and one sorted unique `value_ids` request:

1. use the resident `(L, G + 1)` `value_tiles/indptr` array to resolve every
   requested value to its half-open catalog interval;
2. retain the requested-value position with each nonempty interval so results
   remain aligned with caller order;
3. map the intervals to inner catalog chunks, deduplicate shared chunks, and
   coalesce overlapping or adjacent chunk work into shard-bounded read blocks;
4. read each block once from `value_tiles/manifest_index` and once from
   `value_tiles/n_points`;
5. recover each value's exact interval from the in-memory blocks, intersect its
   manifest indexes with one compact level-local map from manifest row to
   visible-row position, and derive its visible count and positive tiles;
6. process large dense selections in bounded sequential blocks rather than
   materializing a complete cache-wide `value_tiles` array.

Use the physical catalog shard boundary as the explicit maximum read-block
boundary. A block may combine connected touched inner chunks only while they
remain inside one `value_tiles` shard. Split an interval or connected chunk run
that crosses a shard boundary and recover the logical value interval across the
successive blocks. Do not introduce a separate runtime block-size setting.

With the current defaults, one maximum block reads at most 1,048,576 rows from
each of two `uint64` arrays:

```text
1,048,576 rows * 8 bytes * 2 arrays = 16 MiB
```

This is the maximum uncompressed payload of the two catalog result arrays for
one block, not a claim about total process RSS: Zarr decoding, masks, exact
interval fragments, and consumer output require additional bounded or
output-proportional memory. Both `value_tiles` arrays have already passed strict
layout validation and share the same inner-chunk and shard row boundaries.

Do not merge across large unselected gaps merely to reduce the number of Zarr
calls. The read planner must balance call count against decoded and materialized
row amplification. Reuse the established minimal-envelope principle: values
sharing a touched chunk may share one read, while widely separated chunks
remain separate or are handled in bounded consecutive runs.

#### Separate result needs

Expose two internal consumers over the same batched lookup primitive:

- level selection needs only per-requested-value visible counts and the number
  of positive visible tiles;
- viewport reading additionally needs the selected value IDs present in every
  positive manifest row.

The summary path must not build the per-manifest-row dictionary or thousands of
small value-ID arrays. The viewport path may build that mapping after the
batched catalog records are in memory. This separation changes neither the
persisted format nor point-payload reading.

#### Correctness and resource constraints

- preserve sorted unique input validation and caller-order count alignment;
- preserve exact half-open `indptr` semantics, manifest-level bounds, positive
  counts, and strictly increasing manifest indexes within every value interval;
- produce the same logical counts and tile/value mapping as the current
  per-value implementation for any fixed level-selection policy;
- do not open bucket stores or point arrays during catalog-only LOD planning;
- keep temporary memory explicitly bounded for dense value selections;
- do not add an unbounded decoded catalog cache as part of this slice;
- do not change the Zarr catalog schema or chunk dimensions in this slice.

#### Focused tests

Use real sharded Zarr catalog arrays and cover:

- one value, several adjacent values, and widely separated values;
- several value intervals sharing one inner chunk;
- intervals spanning adjacent and nonadjacent chunks;
- empty value intervals and values with no visible manifest rows;
- multi-value counts and per-tile mappings matching the existing logical
  result;
- bounded processing of a dense selection;
- proof at the read-plan or instrumented-store boundary that shared catalog
  chunks are not independently fetched once per selected value.

#### Xenium evaluation

Measure full- and partial-viewport planning for one, ten, one hundred, and all
available values. Record latency, peak temporary memory, requested intervals,
coalesced read blocks, and touched inner chunks. This is a single current-tree
evaluation rather than a strict pass/fail benchmark. Retain the current catalog
format unless the measurements separately justify a future format revision.

#### Implementation and Gate Z10 results

The reader now resolves all requested intervals from the resident `indptr`,
splits them at catalog shard boundaries, and combines connected touched chunks
within each shard. Each resulting block is sliced once from each parallel
`value_tiles` array. One shared block-level iterator serves two consumers:
`select_level()` accumulates only per-value totals and a positive-tile union,
while viewport reading additionally constructs the manifest-row-to-value map.

Within a level, the implementation creates one small direct map from manifest
row to visible-row position. This is bounded by the number of nonempty logical
tiles in that level (7,294 rows at the largest Xenium levels), not by the much
larger number of value-tile records. It lets every catalog record be intersected
in constant time after a block read and avoids repeating a binary search for
each record.

Focused reader tests use real sharded Zarr arrays and pass all nine scenarios.
They freeze logical selected counts and viewport mappings, sampled-value loss,
summary-only early stopping, chunk coalescing without crossing shards, and an
instrumented proof that two values sharing one block cause one slice of each
parallel catalog array.

The retained Z9 Xenium generation contains 5,122 values and was evaluated
without rebuilding it. The partial viewport is the central 50% of the source
extent along each axis. Values in the 1/10/100 cases are the most abundant
Exact values. `intervals`, `blocks`, and `chunks` below are totals over every
level evaluated before the selected LOD was reached. Timings are one first
observation followed by the median of three repeats through the same entered
reader; operating-system and codec caches were not controlled.

| viewport | requested values | selected LOD | first | repeat median | intervals | blocks | touched chunks |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full | 1 | L4 | 17.1 ms | 13.0 ms | 5 | 5 | 5 |
| full | 10 | L7 | 87.4 ms | 78.9 ms | 80 | 30 | 37 |
| full | 100 | L8 | 350.4 ms | 333.4 ms | 900 | 107 | 243 |
| full | 5,122 | L8 | 756.6 ms | 749.6 ms | 45,598 | 37 | 463 |
| central partial | 1 | L2 | 8.40 ms | 7.85 ms | 3 | 3 | 3 |
| central partial | 10 | L4 | 71.3 ms | 70.3 ms | 50 | 27 | 33 |
| central partial | 100 | L8 | 333.2 ms | 330.3 ms | 900 | 107 | 243 |
| central partial | 5,122 | L8 | 724.9 ms | 709.9 ms | 45,598 | 37 | 463 |

The largest planned block was exactly one 1,048,576-row shard, so the two
uncompressed catalog result arrays respected their 16 MiB block-payload bound.
A separate `tracemalloc` run, whose instrumentation increased latency and is
therefore not used in the timing table, observed peak temporary allocations of
0.96, 6.48, 32.26, and 88.67 MiB for the full-viewport 1/10/100/all cases. The
corresponding partial-viewport peaks were 0.83, 6.47, 31.93, and 73.46 MiB.
These totals correctly exceed 16 MiB because they also include exact-fragment
indexes, viewport positions, masks, decoded Zarr working state, and result
accumulators.

The result is bounded and practical for ordinary single- and small-multi-value
interaction. A 100-value selection remains a noticeable catalog-planning
operation, and an explicit list of every value is slower than the dedicated
`value_ids=None` all-values path. Viewer integration should use that dedicated
path when no filtering is intended. No catalog-format or chunk-dimension change
is justified by this gate alone.

#### Exit criteria

- many-value lookup no longer performs two independent Zarr selections per
  nonempty `(level, value_id)` interval;
- summary-only level planning avoids constructing the viewport tile mapping;
- focused tests prove logical equivalence and bounded behavior;
- the full-Xenium evaluation documents whether many-value planning is practical
  for viewer integration.

### Slice Z11: load a selected-value runtime index — resolved

**Status:** implemented with focused verification and evaluated against the
retained 136,578,750-point Xenium cache on 2026-08-20.

#### Goal

Make viewport planning a catalog-I/O-free operation for an unchanged explicit
value selection. Pay the Z10 Zarr lookup cost once when the selected value IDs
change, retain only the exact selected catalog records in an immutable bounded
runtime index, and reuse that index for subsequent pan, zoom, and selected-tile
discovery.

This slice closes the gap exposed by the Z10 Xenium evaluation. Calling the
current selected-value `select_level()` directly for every accepted camera
update would repeat up to tens of synchronous Zarr selections. Event coalescing
prevents an unbounded queue, but it does not make a 70--80 ms catalog operation
appropriate for a viewport hot path. The backend contract must make the I/O
boundary explicit before napari integration.

#### Relationship to Z10

Z11 does not replace the Z10 physical lookup work. It moves that work from
every viewport change to one explicit selected-index loading boundary:

```text
Z10 interval resolution + shard-bounded block reading
        |
        | run once when selected value IDs change
        v
Z11 immutable resident selected-value index
        |
        | reuse for every accepted viewport change
        v
in-memory LOD planning + positive-tile discovery
```

Retain and reuse the Z10 contracts that:

- resolve selected values to exact `value_tiles` intervals;
- split intervals at physical shard boundaries;
- coalesce connected touched chunks without crossing a shard;
- read each block once from both parallel catalog arrays;
- discard rows that belong only to the coalesced envelope rather than an exact
  selected interval.

Refactor the current Z10 block-reading portion into a viewport-independent
index-loading primitive where necessary. The existing on-disk
`_value_filtered_manifest_summary()` and `_value_filtered_manifest()` flows must not
remain as alternative viewport-time paths: their logical aggregation and
tile-mapping responsibilities move to operations over the resident index arrays.
After Z11, there is one selected-value runtime architecture, not an indexed fast
path alongside a synchronous fallback. A raw-value convenience helper may exist
only when it is explicitly named and documented as loading the index from storage;
it must not be callable accidentally from camera-driven planning.

#### Explicit index-loading boundary

Separate selection-dependent catalog I/O from viewport-dependent policy:

```text
selected value IDs change
        |
        v
load selected-value index          catalog Zarr I/O is allowed here
        |
        v
immutable resident selected-value index
        |
        +------------------------------+
        |                              |
        v                              v
select LOD for viewport          discover positive viewport tiles
catalog-I/O-free                 no repeated value_tiles reads
```

Introduce one explicit index-loading operation:

```python
load_selected_value_index(
    value_ids,
    *,
    max_resident_bytes,
) -> _SelectedValueIndex | None
```

The API and semantic boundary are:

- index loading accepts the same sorted, unique `uint32` value IDs as Z9--Z10;
- it uses the Z10 shard-bounded block reader to load every serialized level;
- it discards unselected envelope gaps after decoding and retains only exact
  selected catalog records;
- it reads no bucket store and no point payload;
- the returned object is immutable, generation-bound, and independent of any
  particular viewport;
- it can be constructed on a worker by the later scheduler without accessing a
  napari or VisPy object.

Do not hide first-use I/O inside a nominally pure planner. A synchronous helper
may remain for focused acceptance use only if its name makes the I/O explicit;
the production-facing viewport contract must require either `None` for all
values or an already loaded selected-value index.

#### Resident index representation

Use a small immutable root object and one immutable record set per serialized
level. The recommended logical representation is:

```text
_SelectedValueIndex
  cache_generation_id
  value_ids                 # (S,) read-only uint32
  levels                    # tuple[_SelectedValueLevelIndex, ...]
  resident_bytes

_SelectedValueLevelIndex
  value_indptr              # (S + 1,) read-only uint64
  manifest_index            # (R,) read-only uint64
  n_points                  # (R,) read-only uint64
```

Records remain grouped by the caller's selected-value order. `value_indptr`
preserves empty value intervals and resolves each selected value without a
per-record value-ID array. `manifest_index` uses the global manifest row so the
same index can serve both LOD counts and positive-tile discovery.
All arrays must be C-contiguous, read-only, and validated when the object is
constructed.

The retained record count and byte cost are known before catalog payload I/O:
the resident `value_tiles/indptr` supplies every exact interval length. Reject a
index-loading request before reading the large catalog arrays if its projected
representation exceeds the caller-supplied positive byte budget. Do not fall
back to synchronous per-viewport Zarr reads, silently broaden the selection, or
retain a partially loaded index.

An explicit selection containing every canonical value is normalized to the
existing all-values `None` path and does not construct a selected-value index. Other
dense explicit selections preserve their exact meaning and must satisfy the
configured byte budget or fail explicitly. The later viewer owns the user-facing
policy for such a failure.

The current Xenium catalog provides useful scale evidence for the two core
`uint64` record arrays across all nine levels:

| selected values | exact selected records | two-array bytes |
| ---: | ---: | ---: |
| 10 most abundant | 138,121 | 2.11 MiB |
| 100 most abundant | 1,307,246 | 19.95 MiB |
| all 5,122 | 29,787,508 | 454.52 MiB |

The final deterministic `resident_bytes` total includes the retained value IDs,
every per-level pointer array, and both per-level record arrays. It deliberately
counts owned NumPy buffer bytes rather than platform-dependent Python object
headers. Report process RSS separately so temporary decoding workspace and
Python/NumPy ownership overhead are not mistaken for retained index buffers. Do
not describe the two-array values above as either complete `resident_bytes` or
total process memory.

#### Runtime consumption

Refactor the selected-value planning path so its frequent operation consumes a
selected-value index rather than raw value IDs:

1. compute visible manifest rows from the resident manifest arrays;
2. intersect those rows with the indexed level records in memory;
3. derive per-selected-value counts, the positive-tile union, omitted-value
   evidence, and the budget-first LOD decision;
4. reuse the same indexed level records to map positive manifest tiles to the
   selected values required by bucket-local sparse-range reads.

Repeated viewport planning for one selected-value index must not call
`CatalogReader.array(...)`, slice `value_tiles/manifest_index` or
`value_tiles/n_points`, open a bucket reader, or read a point payload. Viewport
payload loading is still I/O by definition, but it must not repeat the
cache-wide selected-value catalog lookup before opening the already identified
positive buckets.

All-values planning remains unchanged: it sums resident `manifest/n_points`
rows and does not create a selected-value index. The index does not alter
the budget-first policy, sampled-value omission semantics, complete-tile
viewport convention, or persisted cache format.

#### Ownership and lifecycle

Keep the selected-value index explicit rather than storing one hidden mutable
"current selection" on `_PointsCacheReader`. Its identity includes the cache
generation and normalized selected value IDs. Reader operations reject an index
from a different generation.

Z11 implements the immutable index-loading and consumption contracts, not the
complete Phase 2 scheduler or cache manager. The later runtime should:

- load a new selected-value index away from the UI thread;
- keep the previous render snapshot active until index loading and payload reads
  for the new generation are ready;
- pin the active selected-value index;
- optionally retain recent selected-value indexes in a byte-bounded LRU keyed by
  `(cache_generation_id, selection_key)`;
- coalesce camera events, apply LOD hysteresis, and reject stale results.

Do not add an unbounded decoded-catalog cache in this slice. The explicit
index object and `resident_bytes` accounting allow the Phase 2 owner to apply
one coherent memory policy without duplicating hidden caches inside the reader.

#### Focused tests

Use real sharded Zarr catalogs and cover:

- index loading for one, adjacent, separated, sampled-away, and multi-value
  selections;
- preservation of empty intervals and caller-order count alignment;
- exact logical equivalence with the existing Z10 results for several
  viewports and point budgets;
- immutable C-contiguous arrays and complete resident-byte accounting;
- projected over-budget rejection before either large catalog array is sliced;
- normalization of the complete canonical value set to the all-values path;
- rejection of a selected-value index from another cache generation;
- instrumented proof that repeated selected-value `select_level()` calls over
  changing viewports perform zero Zarr selections and open zero buckets;
- instrumented proof that selected `read_viewport()` reuses the indexed tile
  mapping and does not reread either cache-wide `value_tiles` array;
- no change to bucket-local sparse point-range correctness.

Do not put timing thresholds in unit tests.

#### Xenium evaluation

Reuse the retained Z9 cache. For one, ten, and one hundred selected values,
record separately:

- one-time index-loading latency;
- projected and actual resident bytes;
- decoded envelopes, exact retained records, blocks, chunks, and peak temporary
  memory during index loading;
- repeated full and partial viewport LOD-planning latency after index loading;
- positive-tile discovery latency after index loading;
- instrumented catalog Zarr selections during each repeated viewport operation,
  which must be zero.

Exercise a sequence of overlapping pan and zoom viewports through the same
selected-value index. This is a backend evaluation, not a simulation of napari frame
timing: record observations without fixed latency thresholds and do not claim
interactive visual acceptance before Phase 2 integration.

#### Implementation and Gate Z11 results

The reader now exposes one explicit `load_selected_value_index(...)` catalog-I/O
boundary. It calculates exact selected record counts and retained NumPy-buffer
bytes from the resident `value_tiles/indptr` before opening either large catalog
array. Over-budget requests therefore fail before catalog payload I/O. Selecting
the complete canonical vocabulary normalizes to the existing all-values path.

Index loading reuses the Z10 shard-bounded catalog envelopes and writes exact
selected records into one immutable `_SelectedValueLevelIndex` per serialized level.
Each level retains `value_indptr`, `manifest_index`, and `n_points`; the root
`_SelectedValueIndex` retains the canonical value IDs and cache generation
UUID. Every array owns its C-contiguous buffer and is read-only. Reader methods
reject indexes from another generation.

`select_level()` and `read_viewport()` now accept only `None` or an explicit
selected-value index for their selected-value planning contract. The earlier
on-disk `_value_filtered_manifest_summary()` and `_value_filtered_manifest()`
viewport paths have been removed. Direct `read_tile(..., value_ids=...)` remains
available because a known logical tile needs no cache-wide value-to-tile catalog
discovery.

Thirteen focused reader tests use real sharded Zarr catalogs. They cover one,
adjacent, separated, sampled-away, and multi-value selections; immutable arrays;
complete retained-byte accounting; over-budget rejection before catalog payload
access; complete-vocabulary normalization; generation mismatch; LOD policy;
bucket-local sparse correctness; and instrumented proof that repeated indexed
`select_level()` and `read_viewport()` calls cannot reread either `value_tiles`
payload array.

The retained nine-level Xenium cache was evaluated without rebuilding it. The
selected sets are the 1, 10, and 100 most abundant values. Index-loading timings
are single current-tree observations; RSS sampling used a 10 ms interval. The
catalog-envelope figures cover all nine levels.

| selected values | index load | retained records | decoded envelope rows | envelopes | chunks | `resident_bytes` | incremental peak RSS |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 117 ms | 15,087 | 15,087 | 9 | 9 | 0.23 MiB | 0.19 MiB |
| 10 | 247 ms | 138,121 | 652,281 | 31 | 38 | 2.11 MiB | 7.72 MiB |
| 100 | 570 ms | 1,307,246 | 9,905,147 | 107 | 243 | 19.95 MiB | 72.33 MiB |

Projected and actual `resident_bytes` agreed exactly in every case. The larger
RSS deltas correctly include final retained arrays, temporary decoded envelopes,
copy/freeze workspace, codec state, and sampling noise. In particular, the
100-value index load reads substantially more envelope rows than it retains;
that bounded amplification is now paid once per value-selection change rather
than once per camera change.

Three overlapping full and partial viewports were then evaluated through each
selected-value index with a 100,000-point runtime budget:

| selected values | LOD planning range | positive-tile discovery range | runtime catalog Zarr selections |
| ---: | ---: | ---: | ---: |
| 1 | 0.34--0.67 ms | 0.098--0.100 ms | 0 |
| 10 | 1.00--1.44 ms | 0.059--0.072 ms | 0 |
| 100 | 7.72--9.89 ms | 0.373--0.400 ms | 0 |

This establishes the intended boundary: selection changes may perform a
noticeable, memory-bounded index load away from the UI thread, while repeated
viewport planning over the unchanged selection is an in-memory operation. The
measurements are backend evidence, not fixed UI timing thresholds or a claim
about complete napari frame latency.

#### Exit criteria

- selected-value catalog I/O occurs once at the explicit index-loading boundary,
  not once per viewport change;
- selected LOD planning is a pure in-memory operation over resident manifest
  arrays and the selected-value index;
- positive-tile discovery reuses the same indexed records without a second
  cache-wide lookup;
- the loaded index is immutable, generation-safe, and byte-bounded;
- all-values planning continues to bypass the selected-value index;
- focused tests and the retained Xenium evaluation demonstrate zero catalog
  Zarr reads during repeated viewport planning.

### Slice Z12: load resident bucket lookup indexes

**Status:** implemented with focused verification on 2026-08-21; the retained
full-Xenium evaluation remains pending.

#### Goal

Remove bucket initialization and sparse-range metadata reads from the viewport
hot path. Keep bucket readers alive for the cache-reader lifetime, explicitly
initialize them away from the UI thread, and materialize the bucket-local tile
and value-range lookup arrays as immutable, byte-accounted NumPy indexes. Point
payload arrays remain chunked on disk and are still read only for requested
tiles.

This slice addresses a different I/O boundary from Z11:

```text
Z11 selected-value index
  cache-wide value -> positive manifest tiles

Z12 bucket lookup index
  bucket-local tile/value -> exact point-row intervals
```

Neither index contains point coordinates. Z11 changes when selected values
change; Z12 indexes remain reusable across value selections, viewports, and LOD
decisions for one cache generation.

#### Evidence motivating the slice

The retained Xenium cache contains 108 buckets across all nine levels. A
100-abundant-value request over one viewport intersecting nine L1 tiles used
nine distinct L1 buckets and returned 10,294 points. A diagnostic decomposition
observed:

| Work | Total | Approximate per tile |
|---|---:|---:|
| initialize nine bucket readers | 339 ms | 23--48 ms per bucket |
| construct nine first selected read plans | 281 ms | 23--36 ms per tile |
| read and assemble aligned point arrays | 99 ms | 10--14 ms per tile |

The read-plan cost was primarily bucket-local Zarr lookup I/O, not NumPy search
or interval construction. For every selected tile the current reader performs
small selections from `tile_x`, `tile_y`, `tile_offset`,
`ranges/tile_indptr`, `ranges/value_id`, `ranges/row_start`, and
`ranges/row_count` before reading point arrays.

The nine L1 tiles contained 1,591--1,728 range records each. The 100-value
selection retained 93--99 intervals per tile. Those intervals coalesced to one
point-read block per tile but spanned 4,000--4,023 of each tile's 4,096 point
rows. This means the first selected-read latency has two independent targets:
make lookup metadata resident, and later consider a complete-read-and-filter
policy when sparse intervals already span almost a complete tile. Z12 owns the
first target only.

#### Distinguish reader initialization from resident lookup data

Opening a `_BucketReader` initializes the Zarr group, root attributes, array
objects, shapes, chunks, shards, codecs, and dtype/layout checks. It does not
load array contents. Initializing every reader therefore moves the per-bucket
open cost to cache startup but does not by itself remove viewport-time reads of
the sparse lookup arrays.

Use one explicit startup priming flow, run by the later viewer scheduler away
from the UI thread:

```text
open long-lived _PointsCacheReader
        |
        v
initialize required bucket readers
        |  Zarr group and array metadata
        v
load immutable bucket lookup indexes
        |  tile offsets and sparse range records
        v
publish reader as ready for viewport payload requests
```

The current reader already sizes `_BucketReaderCache` for every serialized
bucket and retains each entered reader until `_PointsCacheReader` closes. Z12
must preserve that lifetime. It must not recreate the cache reader when the
viewport or selected value IDs change.

The explicit priming API may initialize all buckets when requested, but must
report progress and remain callable from a worker. Do not hide multi-second
all-bucket initialization inside a nominally cheap constructor or camera-driven
method.

#### Resident bucket lookup contract

The implementation introduces one immutable lookup object per loaded bucket:

```text
_BucketLookupIndex
  level
  bucket_id
  tile_offset             # (K + 1,) read-only uint64
  tile_indptr             # (K + 1,) read-only uint64
  range_value_id          # (M,)     read-only uint32
  range_row_start         # (M,)     read-only uint64
  range_row_count         # (M,)     read-only uint64
  resident_bytes
```

`K` is the bucket tile count and `M` is its sparse range-record count. The
object owns C-contiguous read-only NumPy buffers and remains private to its
owning bucket reader. It is released when that reader closes, so it does not
need an additional cache-generation identity.

`tile_x` and `tile_y` remain serialized bucket arrays required by the physical
format and independent validation. The trusted runtime catalog already supplies
logical coordinates through `_TileDescriptor`, so viewport reads must not reread
bucket `tile_x` or `tile_y` merely to reconfirm each descriptor. Runtime lookup
loading therefore neither reads nor retains these duplicate coordinate buffers.

After loading, pure lookup operations resolve complete and selected tile
requests into bucket-global point intervals as:

```text
descriptor.bucket_tile_index
        |
        +-> tile_offset[i:i+2]       complete point-row interval
        |
        +-> tile_indptr[i:i+2]       tile's sparse range-record interval
                    |
                    v
          range_value_id
          range_row_start
          range_row_count
                    |
                    v
          exact selected point-row intervals
```

Keep this boundary independent of physical point-array reading. The
lookup-facing operations are:

```text
_BucketReader.resolve_complete_tile_interval(descriptor)
    -> (row_start, row_stop)

_BucketReader.resolve_selected_tile_intervals(descriptor, selected_value_ids)
    -> exact point intervals + selected row count
```

Both operations are pure queries over the resident lookup arrays. They do not
inspect point chunks, construct `_PointReadPlan`, choose between slice and
integer row selections, or read `location`, point-level `value_id`, or
`point_id`.

No array in this flow may be sliced from Zarr after the bucket lookup index is
resident. The only viewport-time Zarr selections should be from point payload
arrays such as `location` and point-level `value_id`; visualization continues to
omit `point_id`.

#### Memory policy

The current physical schema stores one range record as:

```text
ranges/value_id    uint32     4 bytes
ranges/row_start   uint64     8 bytes
ranges/row_count   uint64     8 bytes
                             --------
                              20 bytes
```

The retained tile pointers add two `uint64` arrays. Excluding small object
headers, codecs, and temporary decode workspace, projected lookup bytes for a
set of buckets are therefore known before loading:

```text
20 * total_range_count
    + 8 * sum(tile_count_per_bucket + 1)    # tile_offset
    + 8 * sum(tile_count_per_bucket + 1)    # tile_indptr
```

For the retained Xenium cache this is approximately:

| Level | Kind | Range records | Projected resident lookup data |
|---:|---|---:|---:|
| L0 | Exact | 14,790,090 | 296 MB |
| L1 | Bridge | 9,253,957 | 185 MB |
| L2 | Spatial | 3,744,119 | 75 MB |
| all levels | — | 29,787,508 | 596 MB |

The reader must expose exact projected and actual `resident_bytes` accounting.
Loading all levels is acceptable when the caller's explicit metadata budget can
hold them; it must not be an unconditional assumption for every machine or
future dataset.

Support both:

- all-bucket/all-level priming when the complete projection fits the supplied
  positive byte budget;
- explicit level or bucket priming so the later viewer can load the active and
  likely adjacent levels under a smaller budget.

The implemented cache-level boundary is:

```text
project_bucket_lookup_index_bytes(levels=... | bucket_keys=...)
    -> exact requested-set bytes without lookup-array reads

load_bucket_lookup_indexes(
    max_resident_bytes=...,
    levels=... | bucket_keys=...,
    progress=...,
)
    -> exact total resident bucket-lookup bytes
```

With neither selector supplied, both operations address every serialized
bucket. Level and explicit bucket selectors are mutually exclusive. Loading
checks the projected final total, including previously resident indexes, before
reading large lookup arrays. If a bucket load or progress callback fails, every
index newly introduced by that call is released while earlier resident indexes
remain available.

Reject an over-budget request before loading large range arrays. Do not silently
partially prime a requested set and do not fall back to viewport-time Zarr
lookup reads. A level must be ready before it becomes eligible for payload
execution; the later scheduler keeps the previous rendered snapshot active
while a newly required level is initialized.

#### Reader integration

Refactor `_BucketReader` so:

- opening still performs strict structural and array-layout checks;
- an explicit operation loads and freezes its `_BucketLookupIndex`;
- complete-interval lookup resolves `tile_offset` from the resident index;
- selected-interval lookup resolves `tile_indptr` and all range records from the
  resident index and returns exact bucket-global intervals plus their logical
  row count;
- interval lookup remains independent of physical display-read planning and
  point-array I/O;
- runtime coordinate trust comes from the already loaded manifest descriptor;
- closing the reader releases both Zarr objects and resident lookup buffers.

Do not repeat publication-time logical validation while priming a trusted
cache. `_validate_staged_cache()` has already reconciled bucket pointers,
sparse ranges, manifest descriptors, value IDs, and point totals before
publication. Runtime priming retains the representation checks on
`_BucketLookupIndex`, verifies exact projected-versus-actual byte accounting,
and lets strict Zarr reads surface missing physical data, but it does not
revalidate those logical relationships or read `tile_x` and `tile_y`.

Do not add another transient physical planner to Z12. Application-level chunk
IDs, coalesced read blocks, per-tile orthogonal selectors,
`batch_tile_indptr`, and point-payload caching are outside this slice. Z13 owns
combining the intervals for all requested tiles in one bucket, choosing the
slice-or-integer row representation, executing the coordinated Zarr selections,
and splitting the result back into tile payloads.

`_BucketReaderCache` remains the single owner of reader and lookup-index
lifetime. The selected-value index and bucket indexes must not duplicate or own
one another. Their connection remains the manifest address:

```text
selected-value index
  -> manifest row
  -> level + bucket_id + bucket_tile_index
  -> resident bucket lookup index
  -> point-row intervals
  -> Zarr point payload
```

#### Focused tests

Use real sharded Zarr buckets and cover:

- exact projected and actual byte accounting;
- immutable C-contiguous lookup arrays;
- one bucket, one complete level, and all-level priming;
- over-budget rejection before range-array payload reads;
- immutable representation checks and exact projected-versus-actual byte
  accounting while the index is loaded;
- logical equivalence of complete and selected interval resolution before and
  after the refactor, including aligned selected row counts;
- integration coverage showing that the existing tile-scoped point reader can
  consume the resolved intervals without making interval resolution itself own
  physical read planning;
- instrumented proof that initialized viewport reads never slice `tile_x`,
  `tile_y`, `tile_offset`, or any `ranges/*` Zarr array;
- instrumented proof that point arrays remain lazy and are not read by priming;
- reuse across viewport and selected-value changes;
- deterministic release when `_PointsCacheReader` closes.

Do not add latency thresholds to unit tests.

#### Xenium evaluation

Reuse the retained cache without rebuilding it. Record:

- complete and per-level projected lookup bytes;
- reader-initialization and lookup-index loading latency;
- retained and incremental peak RSS;
- all-bucket versus active-level priming observations;
- the same nine-L1-tile 100-value request with initialization and lookup loading
  outside viewport timing, explicitly labelled as a pre-Z13 tile-at-a-time
  point-read baseline rather than Z12's final display-read architecture;
- instrumentation proving that interval planning performs no Zarr selections
  and that the subsequent viewport payload execution selects point arrays only;
- first and repeated point-payload latency after lookup metadata is resident.

This gate records observations without fixed thresholds. It must distinguish
application reader state from uncontrolled operating-system, filesystem, and
codec caches.

#### Exit criteria

- bucket initialization is an explicit startup/worker operation rather than
  hidden viewport work;
- selected and complete tile intervals come exclusively from resident bucket
  lookup indexes;
- interval resolution performs no point-array I/O and creates no physical
  point-read plan or row selector;
- no viewport request rereads bucket tile or sparse-range metadata from Zarr;
- point payload arrays remain lazy and chunked on disk;
- lookup residency is immutable, private to its owning reader, explicitly
  byte-bounded, and released with that reader;
- focused tests and retained-Xenium evidence document memory and latency.

### Slice Z13: batch requested tiles within each bucket

**Status:** implemented with focused real-Zarr verification on 2026-08-21; the
retained full-Xenium evaluation remains pending.

#### Goal

Replace repeated tile-at-a-time point selections with one coordinated read batch
for all requested tiles belonging to the same physical bucket. Resolve every
tile and selected-value interval from the resident Z12 lookup index, expose the
complete bucket batch to Zarr's existing chunk concurrency, and reconstruct the
same immutable tile results in the caller's original order.

Retain the physical grouping already constructed by
`_read_manifest_requests()`:

```text
logical manifest requests
        |
        v
group by (level, bucket_id)
        |
        +-> bucket A -- one coordinated multi-tile Zarr batch
        +-> bucket B -- one coordinated multi-tile Zarr batch
        +-> bucket C -- one coordinated multi-tile Zarr batch
        |
        v
split each batch back into tile payloads
        |
        v
restore original manifest-request order
```

Process bucket batches sequentially. Do not introduce a thread pool, worker task
per bucket or tile, application-managed `asyncio` fan-out, or any other outer
concurrency layer. Zarr alone owns concurrency among the chunks participating in
each coordinated array selection. Outer bucket concurrency and spatially
clustered bucket construction remain outside this roadmap unless later evidence
opens a separate decision.

#### Relationship to Z12

Z13 depends on Z12. Each requested bucket reader and its immutable lookup index
must already be resident before a bucket batch reads point arrays. Batch planning
resolves `tile_offset`, `tile_indptr`, and sparse value ranges exclusively from
that in-memory index; it must not slice bucket tile or range metadata from Zarr.

Z12 and Z13 solve different parts of the same viewport path:

```text
Z12 resident lookup index
  tile/value request -> exact bucket-global point intervals

Z13 bucket batch
  all exact intervals -> coordinated point-array selections -> tile payloads
```

Point arrays remain lazy. Loading a bucket lookup index must not decode
`location`, point-level `value_id`, or `point_id`.

#### Reader refactor boundary

Split the current tile-scoped display-planning responsibility rather than
extending `_selected_read_plan()` to perform bucket I/O. Its selected-value
matching logic remains necessary per logical tile, but its physical display-read
plan does not:

```text
current tile-scoped display path
  _selected_read_plan(descriptor, selected_value_ids)
      -> exact tile intervals
      -> tile-specific _PointReadPlan
      -> tile-specific point-array reads

Z13 display path
  resolve_selected_tile_intervals(descriptor, selected_value_ids)
      -> exact bucket-global intervals only
      -> no _PointReadPlan and no point-array read

  plan_bucket_display_selection(all requested tile intervals)
      -> slice or int64 selected_rows
      -> batch_tile_indptr

  execute one bucket display batch
      -> both row representations use Zarr's orthogonal-selection API
      -> one location selection and one value_id selection
      -> immutable per-tile display payloads

construction path
  resolve one complete tile interval
      -> direct location[start:stop, :]
      -> direct value_id[start:stop]
      -> direct point_id[start:stop]
```

The private names above are descriptive rather than mandatory, but the ownership
boundary is frozen. Per-tile interval resolution:

- validates the descriptor and selected IDs;
- uses only the resident Z12 lookup index;
- returns exact bucket-global intervals and their logical point count, or an
  explicit empty result;
- does not inspect chunk geometry, create `_PointReadPlan`, or access a point
  payload array.

Introduce one canonical plural bucket-reader operation for display payloads.
`_read_manifest_requests()` calls it exactly once for each nonempty physical
bucket group. The direct `read_tile()` API routes its single descriptor through
the same plural operation as a one-request batch; do not retain a second
tile-specific display-I/O implementation. A singular private convenience method
may exist only as a thin wrapper around that canonical batch path.

Keep construction reads separate and unchanged in semantics.
`read_construction_payload()` still reads one complete logical tile including
mandatory `point_id` for Bridge and spatial construction. Because that complete
tile is one contiguous bucket-global interval, read its three aligned arrays
directly with the same `[start:stop]` bounds. A slice may span several chunks;
Zarr still owns their physical chunk processing. Do not redirect construction
through the display batch, which deliberately omits `point_id`.

The current `_PointReadPlan`, `_point_read_plan()`,
`_coalesced_read_blocks_for_intervals()`, and general `_read_aligned_rows()`
exist to turn several tile-scoped sparse intervals into application-planned
basic-slice envelopes and then trim those envelopes in memory. After the display
batch uses one exact basic or orthogonal selection and construction uses one
direct complete slice, they have no production consumer and should be removed
rather than retained as a second physical-read architecture. Replace their
private block-planner tests with the Z13 selection-path tests.

#### Bucket-batch contract

One bucket batch accepts an entered `_BucketReader` and the requests already
grouped for that reader:

```text
tuple[(descriptor, selected_value_ids_or_none), ...]
```

Manifest rows remain a cache-wide catalog concern and are deliberately not
passed into the physical bucket reader. `_read_manifest_requests()` retains
those rows, zips them with the aligned bucket payload results, and restores the
complete cross-bucket request order.

The plural operation returns one aligned payload or `None` per request. `None`
preserves the direct `read_tile()` behavior when a requested value is absent.
The catalog-driven viewport path treats the same result as an inconsistency,
because `value_tiles` had declared that request positive. Empty requests add a
repeated entry to `batch_tile_indptr` and no point rows to the physical
selection; an entirely empty bucket result performs no point-array selection.

For every request, resolve either its exact nonempty bucket-global point
intervals or an explicit empty result in deterministic bucket-request order.
Preserve each tile's exact returned-row count, then merge physically touching
intervals for selection planning. Choose the physical selection only after this
merge:

```text
exact requested intervals
        |
        v
merge touching intervals
        |
        +-- one merged interval
        |       |
        |       v
        |   row_selection = slice(start, stop)
        |   no selected_rows allocation
        |
        +-- several disjoint intervals
                |
                v
            row_selection = one C-contiguous int64 selected_rows array
```

This merge is exact interval normalization, not the old chunk-envelope
coalescing. Merge only when `previous_stop == next_start`. Do not inspect
`chunk_rows`, expand a requested interval to chunk boundaries, or join two
intervals merely because they touch the same or consecutive chunks. Zarr owns
the mapping from the final basic or orthogonal selection to physical chunks.

Construct one transient `uint64 batch_tile_indptr` of length
`requested_tile_count + 1`; its adjacent entries identify the returned row
interval belonging to each requested tile on either physical path. This is
in-memory read workspace and is unrelated to the persisted
`ranges/tile_indptr` that addresses sparse range records. For example, the
disjoint path uses:

```text
tile A requires [10:13] and [20:22]
tile B requires [100:104]

selected_rows      = [10, 11, 12, 20, 21, 100, 101, 102, 103]
batch_tile_indptr  = [0, 5, 9]
```

Reject overlapping, out-of-order, out-of-bucket, or out-of-array exact
intervals. Logical tiles are disjoint, so a valid batch never needs the same
point row twice. Validate all counts and the chosen physical selection before
opening point payload chunks.

Issue exactly one coordinated selection from `location` and one from point-level
`value_id` for each nonempty bucket batch. Pass either row-selection
representation through Zarr's orthogonal-selection entry point; it accepts both
slices and integer arrays. A slice follows the simple contiguous selector path,
while an integer array follows the disjoint advanced-index path. Visualization
continues to omit `point_id`. Zarr maps the selection to its inner chunks,
applies its configured asynchronous chunk concurrency, and may coalesce nearby
reads inside a shard. The application does not create concurrent selection calls
around it.

```text
row_selection:
  slice(start, stop)            # one exact contiguous interval
  selected_rows                 # several exact disjoint intervals

physical reads:
  location.get_orthogonal_selection((row_selection, slice(None)))
  value_id.get_orthogonal_selection((row_selection,))
```

On the disjoint path, do not replace the exact selector with one basic slice
spanning its minimum and maximum row. Tiles assigned to one bucket may be
separated by large runs of unrequested tiles; a broad envelope would decode
unrelated chunks and make memory depend on their gap rather than the requested
payload.

#### Maintainability decision

Do not introduce an adaptive basic-envelope path for disjoint rows. In
particular, do not retain application logic that calculates touched chunk IDs,
coalesces them into blocks, loops over those blocks, and trims unwanted envelope
rows. Orthogonal selection may not be the fastest operation for every dense
within-chunk pattern, but one Zarr-owned selection architecture is preferred to
maintaining two competing physical planners and a policy for choosing between
them.

A narrow read-only diagnostic on the retained sharded Xenium L1 cache motivated
this choice. These were same-process medians with uncontrolled filesystem,
operating-system, and codec caches, not acceptance thresholds:

| Request shape | Current per-tile blocks | Bucket blocks | Bucket orthogonal |
|---|---:|---:|---:|
| 1,258 rows, 2 chunks, 1 block | 9.57 ms | 2.31 ms | 2.87 ms |
| 5,094 rows, 12 chunks, 7 blocks | 13.84 ms | 13.81 ms | 9.29 ms |

The first observation shows a small possible advantage for an ideal basic
envelope; the second shows the orthogonal path benefiting when rows span several
separated blocks. The product decision is that removing repeated per-tile calls
provides the important structural gain, while the remaining sub-millisecond
one-block difference does not justify preserving the chunk-block planner. Reopen
this decision only with representative evidence showing a material product
problem, not because a basic envelope can win an isolated microbenchmark.

Split the two returned arrays at `batch_tile_indptr` and construct one
`_PointDisplayPayload` and `_TileReadResult` per request. Prefer C-contiguous
read-only views into the batch result when that preserves the payload contract;
do not copy the complete batch once per tile. The returned tuple follows the
original bucket-request order. `_read_manifest_requests()` then restores the
complete cross-bucket `requests` order exactly as it does now.

#### Sequential bucket execution and failure

The outer bucket loop remains synchronous and sequential:

```text
bucket A coordinated batch -- wait
bucket B coordinated batch -- wait
bucket C coordinated batch -- wait
```

If planning or reading any bucket fails, fail the complete viewport request and
publish no partial `_ViewportReadResult`. Previously created local batch results
are released normally. Reader lifetime remains owned solely by
`_BucketReaderCache`; Z13 adds no executor, task lifecycle, cancellation model,
or cross-thread close behavior.

#### Memory accounting

For `N` returned rows and `K` requested tiles, the deterministic additional
selection workspace is:

```text
both paths:
  8 * (K + 1)     batch_tile_indptr uint64

disjoint orthogonal path only:
  8 * N           selected_rows int64
```

This is in addition to the returned `(N, 2) float32 location` and `(N,) uint32
value_id` buffers and Zarr's bounded decoded-chunk workspace. The contiguous
basic path allocates no row-selector array. On the disjoint path, construct the
selector as one NumPy allocation rather than one Python integer per point,
report its bytes in the Xenium gate, and release it after the returned arrays
have been split. Explicit-level acceptance reads can exceed the viewer's normal
render budget, so do not describe selector memory as intrinsically capped at
100,000 rows.

The exact physical selection prevents gap amplification in the returned arrays,
but inner chunks remain the minimum independently decoded units. Record
decoded-row amplification separately from selector and logical output bytes. Do
not add a decoded point-payload cache in this slice.

#### Focused tests

Use real sharded Zarr buckets and cover:

- one tile in one bucket;
- several requested tiles sharing one bucket;
- complete and selected tiles with adjacent and separated value ranges;
- touching intervals merging to one slice row selection, passed through the
  same Zarr orthogonal-selection API, with no `selected_rows` allocation;
- disjoint intervals using one exact orthogonal row selector;
- no application-level chunk-ID calculation or coalesced read-block planning;
- requested rows sharing an inner chunk, crossing chunks and shards, and ending
  in the final partial chunk;
- large unrequested row gaps that are absent from the orthogonal result;
- exactly one `location` and one point-level `value_id` selection per nonempty
  bucket batch and no `point_id` selection;
- several buckets executing in deterministic sequential order;
- stable tile order across bucket grouping and result reconstruction;
- logical equivalence with a sequential reference execution;
- direct singleton `read_tile()` and one-request bucket batching using the same
  display-I/O implementation;
- direct construction slices spanning one and several chunks while retaining
  complete aligned `point_id` semantics;
- failure during one bucket producing no partial viewport result;
- deterministic reader cleanup after completed and failed requests;
- C-contiguous immutable point payloads and read-only lookup indexes;
- no reintroduction of bucket metadata Zarr reads after Z12 priming.

Instrument selection objects and logical outputs rather than asserting timing in
unit tests.

#### Xenium evaluation

Reuse the retained cache after Z12 priming without rebuilding it. Evaluate at
least one request containing several tiles from the same bucket so the batch
contract is exercised. Retain the nine-L1-tile, 100-abundant-value request as an
honest counterexample: its nine tiles currently occupy nine distinct buckets,
so Z13 creates nine one-tile batches and should not claim a multi-tile batching
benefit for it.

Record:

- Zarr's configured internal async concurrency;
- bucket-batch count and tiles per batch;
- coordinated Zarr selection count per point array;
- selected level, positive tiles, logical returned points, and decoded-row
  amplification;
- selected-row bytes when the disjoint path is used, `batch_tile_indptr` bytes,
  returned payload bytes, and incremental peak RSS;
- payload latency with readers and lookup indexes resident;
- exact logical equality and stable output order;
- confirmation that no tile/range metadata Zarr arrays were selected.

Do not tune Zarr concurrency, change bucket construction, or compare worker
counts in this gate. Record observations without numerical pass/fail thresholds.

#### Exit criteria

- all requested tiles in one bucket are resolved and read as one logical batch;
- touching intervals use one slice without row-selector allocation, while
  disjoint intervals use one exact integer row selector without materializing
  unrelated row gaps; both use the same Zarr orthogonal-selection API;
- each nonempty bucket batch performs one coordinated selection per display
  point array and delegates chunk concurrency to Zarr;
- separate immutable tile results preserve original request order and logical
  content without full-batch-per-tile copies;
- one canonical plural display-read path serves both viewport and singleton tile
  requests, while construction reads remain tile-scoped and point-ID-complete;
- the obsolete application-level `_PointReadPlan` and chunk-block coalescing
  path have no production or private-test remainder;
- lookup metadata remains resident and display reads never access point IDs;
- buckets remain sequential and no application concurrency layer is introduced;
- selector, decoded-chunk, output-memory, and retained-Xenium latency evidence
  are documented for the final adoption decision.

### Slice Z14: replace catalog envelopes with exact per-level Zarr selections — resolved

#### Goal

Apply the maintainability decision established for Z13 point payloads to the
one-time selected-value catalog load. Replace application-planned,
shard-bounded contiguous envelopes with one exact basic or orthogonal selection
per nonempty serialized level and parallel `value_tiles` array.

This changes only construction of the in-memory `_SelectedValueIndex` when a
value selection changes. Repeated `select_level()` and `read_viewport()` calls
remain catalog-I/O-free and continue to consume the same immutable index
contracts.

#### Motivation and measured evidence

The current Z11 path resolves one contiguous catalog interval per selected
value, splits intervals at shard boundaries, coalesces fragments whose touched
chunks are connected, reads each resulting basic-slice envelope, and then
discards unselected gap rows. It is correct and bounds each individual returned
envelope, but it retains application-level knowledge of chunk and shard
geometry and can materialize many unrelated catalog rows.

For the retained Xenium cache, the 100-abundant-value index contains 1,307,246
exact records while the current envelope path returns 9,905,147 temporary
envelope rows. A same-process warm-cache diagnostic compared the current code
with one exact orthogonal selection per level; outputs agreed record for record:

| selected values | current envelopes | per-level exact selection |
|---:|---:|---:|
| 1 | 23.9 ms | 23.5 ms |
| 10 | 85.0 ms | 42.8 ms |
| 100 | 335.3 ms | 164.9 ms |

These observations motivate the design and are not acceptance thresholds. They
do not represent application-cold latency and must be rerun after the production
refactor.

#### Exact per-level selection contract

For each serialized level, use resident `value_tiles/indptr` to resolve every
selected value to its exact half-open interval in the two aligned catalog
arrays. Selected IDs and their intervals are already ordered by value, so the
physical selector and returned records preserve value-major order:

```text
selected value IDs for one level
        -> exact value_tiles intervals
        -> merge only intervals whose boundaries touch
        -> one resulting interval?
             yes -> slice(start, stop)
             no  -> exact C-contiguous int64 row selector
        -> one selection from value_tiles/manifest_index
        -> one aligned selection from value_tiles/n_points
        -> _SelectedValueLevelIndex
```

Pass either selector representation through Zarr's orthogonal-selection API.
Issue exactly one selection per parallel catalog array for every nonempty level.
Do not expand intervals to chunk or shard boundaries, form a broad slice across
unselected value gaps, calculate touched chunk IDs, or split selections at
shard boundaries. Zarr owns mapping the exact logical selector to inner chunks,
shards, asynchronous work, and decoding.

The returned record order must remain compatible with the already calculated
level-local `value_indptr`. Equal pointers continue to represent a selected
value with no records at that level. Validate aligned result shapes, positive
point counts, level-local manifest bounds, strictly increasing manifest rows
within every selected value, and final projected-versus-observed record counts.

#### Memory policy

For `R` exact records selected from one level, the disjoint path adds an
`8 * R` byte `int64` row selector. Zarr returns two exact `uint64` arrays with a
combined raw footprint of `16 * R` bytes; no unselected gap rows participate in
those returned arrays. Levels are loaded sequentially, so selectors are never
constructed for all levels at once.

For the retained 100-value Xenium selection, the largest level contains 552,033
records: approximately 4.2 MiB for its selector and 8.4 MiB for its two raw
selected results. This is small relative to the accepted selected-index budget
and removes the 9.9-million-row envelope amplification. The existing
`max_resident_bytes` remains a bound on the final immutable index buffers, not a
strict process-RSS or transient-workspace bound. Continue projecting retained
bytes before catalog payload I/O and report selector, returned-result, retained,
and peak-RSS evidence separately.

An explicit selection containing every canonical value still normalizes to the
all-values `None` path. Do not construct a nearly complete catalog selector for
that exact case.

#### Code retirement

Remove `_ValueTileCatalogEnvelope`, `_value_tile_catalog_envelopes()`, their
chunk/shard coalescing tests, and the envelope-fragment write-cursor path from
`_load_selected_value_level_index()`. Do not retain the envelope planner as an
adaptive alternative. Keep one catalog physical-selection architecture unless
representative product evidence later demonstrates a material problem.

Do not reuse a bucket-storage helper across module boundaries merely because
the selector policy is analogous. The catalog implementation may own a small
storage-neutral helper whose validation and return contract are explicit for
catalog row counts.

#### Focused tests

Use real sharded Zarr catalogs and cover:

- one value whose interval uses the contiguous slice path;
- adjacent selected values whose touching intervals merge to one slice;
- separated values using one exact sorted `int64` selector;
- empty selected-value intervals at one or more levels;
- intervals crossing inner chunks, shard boundaries, and the final partial
  chunk or shard;
- large unselected value gaps absent from the returned arrays;
- exactly one `manifest_index` and one `n_points` selection per nonempty level;
- no application-level chunk-ID, shard-splitting, or envelope planning;
- stable value-major ordering and `value_indptr` partitioning;
- immutable C-contiguous outputs, exact resident-byte accounting, and rejection
  before payload I/O when the retained index exceeds its budget;
- record-for-record equivalence with direct expected catalog rows;
- zero catalog Zarr selections during repeated indexed LOD and viewport calls.

Instrument selectors and selection counts rather than timing unit tests.

#### Xenium evaluation

Reuse the retained cache without rebuilding it. Evaluate the 1, 10, and 100
abundant-value sets and record:

- selected records and their distribution by level;
- slice versus integer-selector levels and selector bytes;
- touched chunks and shards as diagnostics reported from array metadata, without
  application read planning;
- catalog selection count, one-time index-load latency, returned-result bytes,
  final `resident_bytes`, and incremental peak RSS;
- exact logical equality with the published catalog;
- confirmation that subsequent LOD and viewport operations perform zero catalog
  selections.

#### Exit criteria

- every nonempty level uses one exact selection per parallel catalog array;
- contiguous intervals use a slice without allocating an integer selector and
  disjoint intervals use one exact `int64` selector without gap rows;
- the immutable selected-value index and every downstream reader result retain
  their current logical contracts;
- the application no longer plans catalog chunks, shards, envelopes, or
  fragment write cursors;
- retained and transient memory evidence is explicit and the full Xenium gate
  is recorded for the final architecture decision.

#### Implemented result

The selected-value loader now resolves one exact row selector per nonempty
level and passes it once to each aligned `value_tiles` array. One selected value
uses the slice path at every level; separated value sets use one C-contiguous
`int64` selector per level. `_ValueTileCatalogEnvelope`, its chunk/shard planner,
fragment write cursors, and their tests have been removed. Focused reader tests
use real sharded catalogs and verify selector form, exact aligned selection
counts, empty levels, immutable output, budget rejection, and catalog-I/O-free
runtime reuse.

The retained nine-level Xenium cache was evaluated without rebuilding it. These
are one current-tree observations with a 10 ms RSS sampling interval; operating
system, filesystem, and codec caches were not reset:

| selected values | index load | exact records | selections per array | slice / `int64` levels | total / maximum selector | `resident_bytes` | incremental peak RSS |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 28.6 ms | 15,087 | 9 | 9 / 0 | 0 / 0 MiB | 0.23 MiB | 0.12 MiB |
| 10 | 43.7 ms | 138,121 | 9 | 0 / 9 | 1.05 / 0.45 MiB | 2.11 MiB | 7.20 MiB |
| 100 | 160.7 ms | 1,307,246 | 9 | 0 / 9 | 9.97 / 4.21 MiB | 19.95 MiB | 77.17 MiB |

Projected and actual retained bytes agreed exactly. The aligned catalog arrays
received identical selectors, the selected and retained record counts agreed,
and repeated indexed LOD and viewport operations performed zero catalog Zarr
selections. The peak-RSS observation includes selected Zarr results, immutable
index copies, selector workspace, codec state, and sampling noise; it is not a
strict bound derived from `max_resident_bytes`.

### Slice Z15: architecture-adoption decision — resolved: adopt Zarr

#### Goal

Conclude the isolated architecture evaluation without blurring the two
implementations. Include the Z11 proof that selected-value viewport planning no
longer performs catalog I/O, the Z12 proof that bucket-local lookup metadata
does not remain on the viewport hot path, the Z13 bucket-batched-read evidence,
and the Z14 exact selected-catalog-read evidence before deciding whether the
backend is suitable for Phase 2 integration.

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

#### Decision

Proceed with the Zarr-backed cache as the product architecture for Phase 2
napari integration. The retained full-Xenium build, independent validation,
resident catalog and bucket lookup indexes, exact selected-value catalog reads,
and budget-controlled viewport payload observations provide sufficient backend
evidence to begin integration. End-to-end napari scheduling and rendering remain
an integration acceptance gate rather than a reason to retain two cache
backends.

Do not add a runtime backend selector, automatic fallback, or compatibility
reader for the experimental tiled-Parquet cache. Slice Z16 makes the adopted
implementation self-contained and gives Phase 2 one unambiguous Python package.

### Slice Z16: make the adopted backend self-contained and remove the tiled-Parquet cache — resolved

#### Goal

Remove the implementation split before napari integration. Make the adopted
Zarr cache independent of the deprecated tiled-Parquet cache and remove the
latter. Keep the adopted implementation under its existing
`multi_scale_cache_points_zarr` package name; renaming it to the former package
name would blur the distinction between the two implementations in code review,
history, documentation, and future maintenance.

The two Parquet roles must remain distinct:

```text
Parquet source data
  remains supported and validated as cache input

tiled-Parquet derived cache
  is deprecated and removed as an output backend
```

Source resolution, schema validation, source signatures, and value
normalization are therefore retained. Parquet cache planning, sampling, writers,
and their storage-specific tests and scripts are removed.

#### Current dependency boundary

The Zarr package still imports the following source-facing behavior from
`multi_scale_cache_points`:

- `PointsBounds`, `ValidatedPointsSource`, and the associated Parquet source
  contracts;
- `_require_parquet_source_unchanged` and the public source validation flow;
- `POINT_ID_POLICY`, `SOURCE_SIGNATURE_METHOD`, and normalized Arrow type
  handling;
- `VALUE_NORMALIZATION_METHOD` and source-value normalization.

These imports do not justify retaining the old cache backend. Move the required
source-ingestion contracts into the adopted package first, together with their
relevant tests, and only then remove the old package.

#### Source subpackage

Keep the adopted package root focused on cache construction, storage, and
reading. Move the retained input-facing behavior into one dedicated `source`
subpackage:

```text
multi_scale_cache_points_zarr/
  source/
    __init__.py
    errors.py
    models.py
    resolution.py
    validation.py
    signature.py
    value_normalization.py

  storage/
  writer/

  build_plan.py
  builder.py
  cache_format.py
  hashing.py
  models.py
  payload.py
  reader.py
  sampling.py
```

Use `source`, not `discovery`, for this boundary. Discovery describes only the
step that locates a SpatialData point element and its physical Parquet dataset;
the subpackage owns that resolution plus source models, validation, signatures,
and logical-value normalization.

The module responsibilities are:

```text
source/resolution.py
  SpatialData point element -> ParquetPointsSource

source/validation.py
  ParquetPointsSource -> ValidatedPointsSource

source/models.py
  PointColumnSelection
  PointsBounds
  ParquetPointsSource
  ParquetSourceFile
  ParquetSourceRowGroup
  ValidatedPointsSource

source/signature.py
  stable source-inventory signatures
  normalized Arrow schema descriptions

source/value_normalization.py
  canonical logical values and value IDs

source/errors.py
  source-resolution and source-validation exceptions
```

`source/__init__.py` is the intentional source-facing facade and exports only
the contracts and operations needed to resolve and validate an input source:

```text
ParquetPointsSource
PointColumnSelection
ValidatedPointsSource
resolve_spatialdata_points_source
validate_parquet_points_source
```

Implementation modules may import additional private source contracts directly
from their defining modules. In particular, use
`multi_scale_cache_points_zarr.source.models.ValidatedPointsSource` internally
rather than re-exporting every source implementation detail at the adopted
package root.

This hierarchy deliberately keeps two model domains separate:

```text
source/models.py
  contracts describing the original point source

models.py
  contracts describing the constructed Zarr cache
```

Do not introduce deeper `discovery`, `validation`, or `normalization`
subdirectories. The `source` package is the ownership boundary; its small
modules provide sufficient separation without additional nesting.

#### Package transition

Perform the transition in one coherent slice:

```text
before
  core/multi_scale_cache_points/          tiled-Parquet cache plus source validation
  core/multi_scale_cache_points_zarr/     adopted cache borrowing source helpers

after
  core/multi_scale_cache_points_zarr/     adopted cache plus source validation
```

Keep the focused tests under `tests/multi_scale_cache_points_zarr`. Do not reuse
the deleted Parquet test-directory name for the adopted implementation.

The safe implementation order is:

1. move the required source-facing models, errors, resolution, validation,
   signatures, and normalization into the Zarr implementation's `source`
   subpackage;
2. update production code, Zarr tests, retained validation tools, and retained
   benchmark tools so no adopted path imports the old package;
3. verify the self-contained Zarr package before removing anything;
4. remove the deprecated tiled-Parquet writers, planning, sampling, hashing,
   storage-specific tests, and backend-only benchmark tools;
5. verify that the repository contains no executable import or backend selector
   referring to removed tiled-Parquet implementation symbols.

Do not leave forwarding modules under the deleted `multi_scale_cache_points`
namespace. This project has explicitly declined backward compatibility for the
experimental cache backend, and a compatibility layer would preserve the
ambiguity this slice is intended to remove.

#### Retained and removed coverage

Retain and relocate tests covering input behavior that remains part of the
product contract:

- SpatialData point-element resolution;
- Parquet source inventory and schema validation;
- selected-column and bounds contracts;
- source-change detection and source signatures;
- logical value normalization;
- all adopted Zarr construction, publication, validation, and reader behavior.

Remove tests whose subject is exclusively the tiled-Parquet derived-cache
architecture. Apply the same rule to scripts: retain tools that validate or
measure the adopted cache or its source input, update their imports, and remove
tools that only construct or profile the deprecated cache.

Historical roadmap discussion and recorded measurements may continue to name
the tiled-Parquet implementation. They are documentation, not executable
dependencies, and should not be rewritten as if the historical evaluation used
Zarr.

#### Format and artifact compatibility

Moving the retained source-ingestion code must not change the adopted on-disk
schema, schema version, cache generation identity, publication protocol,
hashing policies, or serialized method identifiers. Existing published Zarr
caches must reopen with `multi_scale_cache_points_zarr` after the cleanup. No
full-Xenium rebuild or new performance comparison is required solely because
source-validation modules moved.

#### Focused verification

- run the relocated source-resolution and Parquet-input-validation tests;
- run the complete adopted-cache focused tests under
  `tests/multi_scale_cache_points_zarr`;
- reopen the retained full-Xenium Zarr cache and exercise catalog entry,
  selected-value index loading, bucket lookup priming, LOD selection, and one
  viewport payload read;
- scan `src`, `tests`, and retained `scripts` for stale imports and removed
  backend symbols;
- confirm that importing `multi_scale_cache_points_zarr` cannot import a
  tiled-Parquet writer transitively.

#### Exit criteria

- one cache implementation remains under `multi_scale_cache_points_zarr`;
- the adopted implementation has no dependency on deleted cache code;
- Parquet input resolution and validation retain their focused coverage;
- tiled-Parquet cache construction code, storage-specific tests, and
  backend-only tools are removed;
- no compatibility shim, runtime backend selector, or automatic fallback is
  introduced;
- retained Zarr caches reopen without a schema or artifact migration;
- Phase 2 napari integration imports only `multi_scale_cache_points_zarr`.

#### Implemented result

The retained source-input boundary now lives under
`multi_scale_cache_points_zarr/source`. Its public facade exposes source
resolution and validation, while implementation modules own source models,
errors, signatures, and value normalization. Cache-internal models remain at the
adopted package root.

All adopted production modules, focused tests, validation tools, and retained
benchmark tools import the new source boundary. The deprecated
`multi_scale_cache_points` package, its tiled-Parquet planning and writers, its
storage-specific tests, and its two backend construction benchmarks have been
removed. The source-validation benchmark remains available under the explicit
`benchmark_multi_scale_cache_points_zarr_source_validation.py` name. No
forwarding package or backend selector remains.

The complete focused suite passed with 272 tests. A stale-import scan over
`src`, `tests`, and `scripts` found no executable reference to the removed
package, and Python package discovery no longer resolves it. The retained
full-Xenium generation reopened without migration; one common-value full-extent
smoke request selected L4 within budget, primed three required bucket indexes,
and returned the expected 78,789 points across 127 positive tiles. No cache
rebuild was required.

## Test strategy

Normal development uses focused tests under
`tests/multi_scale_cache_points_zarr`. Z16 retains that location, moves the
source-input validation coverage into it, and removes tests belonging only to
the deprecated cache. The tests do not compare against tiled-Parquet artifacts
or performance.

Test layers are:

```text
fresh logical logic
  planning, hashes, sampling, rebasing

Zarr primitive
  arrays, offsets, ranges, chunks, shards, corruption

level writers
  Exact, Bridge, spatial membership and counts

cache-wide Zarr catalog
  root attributes, values, manifest, value-to-tile CSR

staged validation
  independent hierarchy and cross-index reconciliation

builder
  guards, cleanup, completion, publication

acceptance reader
  complete and selected physical reads

selected-value index
  one exact per-level catalog selection per aligned array, immutable bounded
  index, IO-free viewport planning

bucket lookup indexes
  eager reader initialization, resident tile/range metadata, lazy point payloads

bucket-wide point reads
  one exact slice-or-integer multi-tile selection per display array through
  Zarr's orthogonal-selection API,
  Zarr-owned chunk concurrency, stable splitting, sequential bucket execution
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
- Cache-wide `value_tiles` construction materializes and sorts one level's
  compact range records at a time, writes ordered bounded slices, and releases
  that workspace before advancing to the next level.
- A selected-value index retains only exact selected catalog records,
  reports its complete resident byte cost, and is rejected before payload I/O
  when its projected representation exceeds the supplied runtime budget. Load
  each level through one exact slice-or-`int64` selection per aligned catalog
  array; do not retain application-level catalog chunk, shard, or envelope
  planning. Account the transient per-level selector separately from retained
  index bytes.
- Resident bucket lookup indexes retain only tile offsets, sparse-range
  pointers, and sparse range records. Their projected and actual bytes are
  explicitly accounted, and point payload arrays remain on disk.
- Viewport requests are grouped by bucket and all requested tiles in one bucket
  are resolved into one exact row selection per display point array. Represent
  one merged interval as a slice and disjoint intervals as an `int64` selector;
  pass either representation through the same Zarr orthogonal-selection API.
- Bucket batches execute sequentially. Zarr alone owns concurrency among chunks
  participating in one selection; do not add a bucket or tile executor.
- Transient batch tile pointers and any disjoint-path row selector are explicitly
  byte-accounted, and a batch must not replace disjoint rows with one
  gap-amplifying envelope.
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
- Independent validation reopens the complete hierarchy and catalog after
  writers close.
- Root `publication_state` remains `"staging"` throughout construction and
  validation.
- The final source guard precedes the transition to `"complete"` and
  publication.
- Readers reject an incomplete publication state, unsupported versions, missing
  stores, and inconsistent indexes.
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

The cache-wide `value_tiles` CSR arrays and bucket sparse ranges intentionally
duplicate tile/value keys for opposite query directions. Derive `value_tiles`
from finalized ranges and independently validate exact key/count equality.

### Root-attribute growth

Root attributes contain only small semantic structures and the value-label
dictionary. Tile rows and value-to-tile rows remain typed arrays. Gate Z6 records
the root `zarr.json` size; if a future vocabulary makes `value_names` materially
large, introduce a versioned string-storage contract rather than silently
placing manifest-like tables in attributes.

### Zarr API and version stability

Keep Zarr calls inside the new storage package, use explicit v3 checks, and
defer dependency bounds until the working primitive and Xenium build establish
requirements.

### Nested Zarr stores inside SpatialData

The cache root is a Harpy-owned Zarr group, not a SpatialData element. Its bucket
directories are both independently openable stores and child groups in that
hierarchy. Use a distinct contained path, validate every ancestor and bucket
node, avoid consolidated metadata over the complete bucket hierarchy initially,
and verify that SpatialData operations ignore the isolated cache group.

### Sparse values distributed everywhere

The `value_tiles` CSR index cannot prune a rare value present in every visible
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
- every manifest-catalog tile resolves to one verified Zarr interval;
- every nonempty tile/value key resolves to one verified sparse range;
- root attributes and every cache-wide catalog array reconcile with all bucket
  stores;
- independent staged validation fails closed on corruption;
- construction is bounded, failure-safe, and published with the documented
  generation-replacement and rollback semantics;
- full-Xenium build and read measurements are recorded;
- repeated selected-value viewport planning is catalog-I/O-free after one
  explicit byte-bounded selected-index loading step;
- selected-value index loading uses exact per-level catalog selections without
  materializing unselected value gaps or planning catalog chunks and shards;
- initialized viewport reads resolve tile and sparse value ranges from
  byte-bounded resident bucket lookup indexes without Zarr metadata-array reads;
- all requested tiles sharing one bucket are read through coordinated exact
  point-array selections, split back into immutable results in original order,
  and processed without application-managed read concurrency;
- the project explicitly adopts or does not adopt the candidate architecture.

## Immediate next slice

Z0 through Z16 are resolved. The Zarr-backed cache is adopted, self-contained,
and the deprecated tiled-Parquet cache has been removed. The next phase is
public napari integration against the unambiguous
`multi_scale_cache_points_zarr` package; no cache-backend compatibility or
selection layer is required.
