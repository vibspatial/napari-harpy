# Independent hybrid Parquet/Zarr multiscale points cache

Status: implementation roadmap for an isolated Zarr-backed cache experiment

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

The existing `multi_scale_cache_points` package remains unchanged while the
experiment is developed. It is a reference and rollback point, not a runtime
fallback and not a backend selected by the new package.

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
coupling the experiment to existing writer internals.

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

This document defines the Zarr experiment. It replaces earlier proposals to
incrementally migrate the existing Exact, Bridge, and spatial Parquet writers.

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

## Experiment and compatibility policy

This is an isolated engineering experiment evaluated on the Xenium example.
Correctness is mandatory. Build time, peak RSS, cache size, object count, and
representative read behavior are recorded and reviewed together; there are no
pre-registered numerical pass/fail thresholds.

The decision remains binary:

```text
Zarr cache is correct, memory-bounded, and practically satisfactory
    -> promote the new implementation in a later integration decision

Zarr cache is unsatisfactory
    -> abandon multi_scale_cache_points_zarr and retain the existing package
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
9. Keep the experiment understandable in isolation, even when that duplicates
   logic from `multi_scale_cache_points`.

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

The final cache directory name is provisional until the public integration
decision. During the experiment it must not collide with the existing derived
cache path. Every `bucket-<id>.zarr` is an independent Zarr v3 `LocalStore`.

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
bucket_path: str
bucket_tile_index: int
tile_x: int
tile_y: int
n_points: int
```

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
codec identifier and parameters
```

Semantic cache metadata remains in `metadata.json`.

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

### Chunking and sharding

Point arrays use aligned chunks and Zarr v3 shards along the point dimension:

```text
location chunks = (point_chunk_rows, 2)
point_id chunks = (point_chunk_rows,)
value_id chunks = (point_chunk_rows,)

location shards = (point_shard_rows, 2)
point_id shards = (point_shard_rows,)
value_id shards = (point_shard_rows,)
```

`point_shard_rows` is an integer multiple of `point_chunk_rows`. Chunks are the
compression/read unit; shards reduce filesystem-object growth. Fixed chunks may
cross tile and value boundaries.

Initial experimental values are:

```text
point_chunk_rows = 4,096
point_shard_rows = 131,072       # 32 chunks
```

They are provisional. Slice Z3 records Exact-level evidence before physical
settings are frozen. Range arrays share a separate `range_chunk_rows` setting;
small tile arrays may use one chunk per bucket.

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
or shards.

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
  -> only intersecting location chunks
  -> point_id chunks only when identity is required
```

Opening a Zarr bucket loads metadata and required small index chunks; it must not
materialize the complete point arrays. Selection is beneficial only when the
selected intervals touch fewer chunks than complete positive tiles. A rare value
present in every visible tile remains a meaningful worst case and is measured.

## Implementation slices

Each slice is implemented only under `multi_scale_cache_points_zarr` and its
tests unless it explicitly changes documentation or a future public entrypoint.
Every slice ends in a coherent focused test set. The existing package and its
tests must remain untouched.

The dependency sequence is:

| Slice | Delivers | Depends on |
|---|---|---|
| Z0 | isolated experiment boundary | roadmap decision |
| Z1 | fresh models, planning, hashes, and payload contracts | shared validated source |
| Z2 | standalone Zarr bucket writer, reader, and validator | Z1 |
| Z3 | Exact source-to-Zarr construction and Xenium Exact gate | Z1–Z2 |
| Z4 | fresh sampler and Bridge Zarr construction | Z3 |
| Z5 | fresh rebasing and all spatial/overview Zarr levels | Z4 |
| Z6 | metadata and compact Parquet indexes | Z3–Z5 |
| Z7 | independent complete-generation validation | Z6 |
| Z8 | guarded end-to-end build and publication | Z7 |
| Z9 | acceptance reader and full-Xenium evaluation | Z8 |
| Z10 | explicit promotion or retirement decision | Z9 |

No slice depends on an adapted Parquet point writer or a compatibility reader.

### Slice Z0: freeze the isolated experiment boundary — resolved

#### Goal

Make the new-package strategy authoritative before implementation.

#### Work

- declare `multi_scale_cache_points_zarr` the sole implementation location for
  this experiment;
- reuse only canonical source models, validation, and source-signature facts;
- allow duplication of all derived-cache logic;
- forbid imports from the existing writer package;
- forbid transitional or mixed point-payload backends;
- retain the existing package unchanged as the reference and rollback point;
- use `transcripts_vis_zarr` as a noncolliding experimental output path until
  promotion is decided.

#### Exit criteria

- this roadmap contains no incremental Parquet-to-Zarr migration plan;
- Z1 can create a new package without changing existing writers;
- acceptance or rejection of the experiment has a simple package-level
  boundary.

### Slice Z1: scaffold fresh contracts and build planning

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

Use a provisional construction target of 2,000,000 planned points per bucket:

```text
bucket_count = max(1, ceil(level.point_count_upper_bound / 2_000_000))
```

This is a construction policy, not part of the logical tile identity. Z3 may
change it from Xenium evidence before the cache format is frozen.

Bucket filenames use the complete planned bucket count for deterministic width:

```text
width = max(3, len(str(bucket_count - 1)))
levels/level_<level>/bucket-<zero-padded bucket_id>.zarr
```

Empty bucket IDs create no plan, descriptor, or store.

#### Tile and bucket models

Define `_TileDescriptor` in `models.py`:

```text
level: int
bucket_id: int
bucket_path: str
bucket_tile_index: int
tile_x: int
tile_y: int
n_points: int
```

Validate:

- serialized integer ranges: level `int16`, bucket/tile/index `uint32`, and
  `n_points` in `[1, int64_max]`;
- booleans are rejected as integers;
- `bucket_path` is a normalized cache-root-relative POSIX path;
- it is directly inside `levels/level_<level>`;
- it has the `.zarr` suffix;
- it contains no absolute root, `..`, or noncanonical spelling.

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
codec_id: str
```

All row settings are positive integers, `point_shard_rows` is an integer
multiple of `point_chunk_rows`, and `codec_id` is a nonempty versioned string.
The settings describe requested physical behavior; Z2 maps them to concrete
Zarr codec objects and may extend the model with JSON-compatible codec
parameters before the bucket contract is frozen.

Define `_BucketPlan`:

```text
level: int
bucket_id: int
bucket_path: str
tiles: tuple[_PlannedTile, ...]
settings: _ZarrWriteSettings
```

Required invariants:

- a plan contains at least one nonempty tile;
- the path matches its level and ends in `.zarr`;
- tile coordinates are unique and strictly ordered by `(tile_y, tile_x)`;
- tile fields fit the serialized ranges;
- the sum of tile counts is positive and at most `int64_max`;
- derived properties expose `tile_count`, `point_count`, and the exact
  `tile_offset` prefix sums without storing a second independent count.

Define `_BucketWriteResult`:

```text
level: int
bucket_id: int
bucket_path: str
tile_descriptors: tuple[_TileDescriptor, ...]
point_count: int
range_count: int
```

It represents a finalized nonempty store. Descriptors must all belong to its
level, bucket ID, and path; their bucket-local indexes are exactly `0..K-1` in
tile order; their point total equals `point_count`; and `range_count` is at
least the tile count and at most the point count.

Define `_LevelWriteResult`:

```text
level: int
buckets: tuple[_BucketWriteResult, ...]
```

Buckets are ordered by unique `bucket_id`. Derived properties flatten globally
ordered tile descriptors and calculate point, tile, bucket, and range totals.
Across the level, tile coordinates and `(bucket_path, bucket_tile_index)` keys
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
- expose an experimental public builder;
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
- descriptor paths, suffixes, integer ranges, uniqueness, order, and bucket
  ownership;
- exact `BucketPlan` prefix sums and total reconciliation;
- valid and invalid bucket/level results;
- an import scan proving no forbidden existing implementation dependency;
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

### Slice Z2: implement the standalone Zarr bucket primitive

#### Goal

Implement and validate one bucket independently of Dask and pyramid writers.

#### Work

- create Zarr v3 stores with the exact array hierarchy above;
- implement planned sequential point writes and bounded range-array growth;
- persist tile identity arrays, offsets, sparse value ranges, and root
  attributes;
- implement deterministic finalization, close, and incomplete-write behavior;
- implement complete-tile reads returning the storage-neutral payload;
- implement selected-value range lookup and selected payload reads;
- reuse open store/array handles within a reader context and close them
  deterministically;
- implement structural bucket validation from disk without trusting writer
  result objects;
- reject Zarr v2, unsupported attributes/codecs, missing arrays, dtype/shape
  disagreements, nonmonotonic pointers, range gaps/overlaps, point/range value
  disagreements, and unexpected objects.

#### Focused tests

- one and several tiles per bucket;
- one and several values per tile;
- values absent from some tiles;
- complete-tile round trips;
- one- and multi-value selected reads;
- tiles and value runs crossing chunk and shard boundaries;
- coordinate and `point_id` alignment;
- every structural corruption class above;
- unknown descriptor, closed reader, failed write, and cleanup behavior.

#### Microbenchmark

Use synthetic payloads representing:

- an average Exact tile;
- a dense approximately 108,598-point Exact tile;
- a 4,096-point Bridge tile;
- localized and distributed value ranges.

Record complete and selected cold/warm reads, compressed bytes, chunks touched,
decoded-row estimates, shard objects, and write time for a small configuration
matrix.

#### Exit criteria

- the primitive roundtrips aligned payloads exactly;
- selected lookup does not scan the point-level `value_id` array;
- corruption fails closed;
- a provisional physical configuration is ready for Exact construction.

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

Use the evidence to select point chunks, shard rows, range chunks, and codecs for
the remaining slices. This is an engineering decision, not a comparison gate
against the existing Parquet implementation.

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

Expose one experimental builder that creates only complete Zarr-backed
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
- preserve an existing completed experimental generation on every failure;
- close all Dask tasks, Zarr stores, memory maps, and file handles before
  validation and rename;
- make `COMPLETED` the final staged write;
- reject incomplete generations when opening;
- expose no public backend selector and do not call the existing builder;
- treat the experimental cache as disposable and rebuildable.

#### Focused tests

- first build and successful replacement;
- failures before staging and during every major construction phase;
- failure during artifacts, validation, final guard, completion, and rename;
- preservation of an existing generation;
- cleanup of incomplete stores and Dask scratch;
- all handles closed before publication;
- canonical source remains unchanged.

#### Exit criteria

- the experimental output path is absent or contains one complete independently
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

If accepted, record the format and physical settings as the proposed Phase 2
input. If rejected, stop work on this package and retain the existing cache
implementation. Do not add a fallback path.

#### Exit criteria

- direct sparse-range lookup is demonstrated at full scale;
- its useful and non-useful cases are documented honestly;
- all-values access remains practical for the planned viewer;
- the experiment has an explicit accept/reject result.

### Slice Z10: promotion or retirement decision

#### Goal

Conclude the isolated experiment without blurring the two implementations.

#### If accepted

- update the higher-level transcript-visualization roadmaps with measured Zarr
  behavior;
- choose the future public entrypoint and final cache directory name;
- document the supported schema and dependency bounds;
- plan removal or archival of the existing derived-cache implementation as a
  separate integration task, not inside the experiment;
- keep only one public backend after promotion.

#### If rejected

- document the measured reason;
- remove or archive `multi_scale_cache_points_zarr` as experimental work;
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
- Range arrays may grow only in coarse bounded blocks and are trimmed at
  finalization.
- Small consecutive tiles should be buffered across tile boundaries so one
  partially filled shard is not repeatedly rewritten.
- Readers reuse handles within a request/context and close them deterministically.
- All writers and readers close before staged validation or publication.

## Failure and publication rules

- The cache is derived and disposable.
- All writes occur under a unique sibling staging generation.
- A failure in one bucket invalidates the whole staging generation.
- The first implementation does not resume or repair partial stores.
- Independent validation reopens all required artifacts after writers close.
- `COMPLETED` is absent throughout construction and validation.
- The final source guard precedes `COMPLETED` and publication.
- Readers reject missing completion, unsupported versions, missing stores, and
  inconsistent indexes.
- Existing completed experimental generations survive failed replacements.

## Risks and mitigations

### Duplication drift

Fresh planning and sampling code may diverge unintentionally from the intended
semantics. Mitigate with explicit invariant tests and small golden vectors, not
imports from the existing writer package. Differences are acceptable only when
documented as deliberate Zarr-experiment decisions.

### Small selected reads

A one-point interval still decompresses a complete chunk. Measure read
amplification and coalesce ranges; do not create a chunk per gene.

### All-values regression

Value-major ordering remains sequential within a complete tile, but it may
alter compression and access behavior. Benchmark complete tiles and viewports
before promotion.

### Filesystem-object growth

Use independent bucket stores with Zarr v3 sharding. Measure actual files and
directories at full scale rather than inferring object count from logical
chunks.

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
experimental cache directory.

### Sparse values distributed everywhere

`tile_value_counts.parquet` cannot prune a rare value present in every visible
tile. Viewport clipping, pyramid selection, sparse ranges, and chunk caching
still bound work, but Z9 must measure this case.

## Definition of done

The experiment is complete when:

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
- the project explicitly accepts or rejects the experiment.

## Immediate next slice

Z0 is resolved by this roadmap. Implement Z1 by creating
`multi_scale_cache_points_zarr` and its fresh logical contracts. Do not modify or
adapt the existing Exact, Bridge, spatial, or writer-support modules, and do not
introduce a transitional Parquet point reader.
