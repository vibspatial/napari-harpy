# Multiscale points Zarr cache format

This document explains the logical format implemented by
`multi_scale_cache_points_zarr`. It uses one small, fully reconciled cache to
show how logical tiles, physical buckets, manifest rows, value-to-tile records,
bucket point ranges, and value-major coordinate rows relate to one another.

The schema parser and storage validators remain the authoritative executable
contract. In particular, see [`cache_format.py`](cache_format.py),
[`storage/_schema.py`](storage/_schema.py), and
[`storage/catalog_reader.py`](storage/catalog_reader.py). The example values in
this document are illustrative, but every array shape, pointer, count, and
relationship follows the current `harpy-multiscale-points-zarr-cache-0.2`
format.

## 1. The three address spaces

The most important distinction is that the cache uses three different address
spaces:

```text
logical tile address
    (level, tile_x, tile_y)
        |
        v
global manifest address
    manifest_index
        |
        v
physical tile address
    (level, bucket_id, bucket_tile_index)
        |
        v
physical point address
    bucket-global point row
```

The value index provides another direction into the same logical tiles:

```text
(level, value_id)
        |
        v
value_tiles record interval
        |
        v
manifest_index values for tiles containing that value
```

The principal index names therefore mean:

| Name | Scope | What it addresses |
|---|---|---|
| `manifest_index` | Complete cache | One logical nonempty tile represented by one row in the parallel manifest arrays |
| `bucket_tile_index` | One level/bucket | One tile inside that physical bucket |
| `tile_offset` | One level/bucket | Point-row boundaries for bucket-local tiles |
| `ranges/tile_indptr` | One level/bucket | Sparse range-record boundaries for bucket-local tiles |
| `ranges/row_start` | One level/bucket | First bucket-global point row for one tile/value run |
| `value_tiles/indptr` | Complete cache, partitioned by level/value | Rows in the flat value-to-tile catalog arrays |
| `value_point_indptr` | One value-major level | Coordinate-row boundaries for canonical values |

An integer such as `2` has no meaning without its address space. Manifest row
2, bucket-local tile 2, bucket point row 2, and value-tile row 2 are unrelated
addresses.

## 2. Logical hierarchy

Ignoring Zarr metadata files, chunk keys, and shard objects, one cache has this
logical hierarchy:

```text
cache/
├── zarr.json                         root attributes
├── values/
│   └── n_points                      uint64[V]
├── manifest/
│   ├── level_indptr                  uint64[L + 1]
│   ├── bucket_id                     uint32[T]
│   ├── bucket_tile_index             uint32[T]
│   ├── tile_x                        uint32[T]
│   ├── tile_y                        uint32[T]
│   └── n_points                      uint64[T]
├── value_tiles/
│   ├── indptr                        uint64[L, V + 1]
│   ├── manifest_index                uint64[R]
│   └── n_points                      uint64[R]
├── value_major/
│   ├── level_0/
│   │   ├── location                  float32[N_0, 2]
│   │   └── value_point_indptr        uint64[V + 1]
│   └── ... one group per level
└── levels/
    ├── level_0/
    │   ├── bucket-000.zarr/
    │   │   ├── location              float32[N_b, 2]
    │   │   ├── point_id              uint64[N_b]
    │   │   ├── value_id              uint32[N_b]
    │   │   ├── tile_x                uint32[T_b]
    │   │   ├── tile_y                uint32[T_b]
    │   │   ├── tile_offset           uint64[T_b + 1]
    │   │   └── ranges/
    │   │       ├── tile_indptr       uint64[T_b + 1]
    │   │       ├── value_id          uint32[R_b]
    │   │       ├── row_start         uint64[R_b]
    │   │       └── row_count         uint64[R_b]
    │   ├── bucket-001.zarr/
    │   └── ...
    └── ... one group per level
```

Here:

- `L` is the number of serialized levels.
- `V` is the canonical value count.
- `T` is the number of nonempty manifest tiles across all levels.
- `R` is the number of nonempty `(level, value, tile)` records.
- `N_0` is the point count at level 0; every level has its own `N_L`.
- `N_b`, `T_b`, and `R_b` are the point, tile, and sparse-range counts in one
  bucket.

Root attributes store the schema version, generation and publication identity,
source signature, geometry, build settings, level descriptors, canonical
`value_names`, catalog descriptor, and value-major descriptor. Value names are
therefore not repeated in a Zarr array. `values/n_points` stores their aligned
Exact/source counts.

Each bucket also has attributes declaring its payload schema version, level,
bucket ID, tile/point/range counts, point ordering, coordinate encoding, and
codec. Array chunks and shards are physical storage choices recorded and
validated separately from the logical relationships below.

## 3. One complete example

The example cache has:

```text
levels: 1
values: 3
tiles:  3
buckets at level 0: 2
points at level 0: 8
```

The canonical values are:

```text
value_id  value name
0         alpha
1         beta
2         gamma
```

The root attributes contain:

```text
value_names = ["alpha", "beta", "gamma"]
```

The logical points are:

| Tile | `(tile_x, tile_y)` | Value | Point ID | Tile-relative location |
|---|---:|---:|---:|---:|
| A | `(0, 0)` | 0 | 10 | `(1, 1)` |
| A | `(0, 0)` | 0 | 11 | `(2, 1)` |
| A | `(0, 0)` | 2 | 12 | `(4, 3)` |
| B | `(2, 0)` | 1 | 20 | `(0.5, 2)` |
| B | `(2, 0)` | 2 | 21 | `(3, 1)` |
| B | `(2, 0)` | 2 | 22 | `(4, 1)` |
| C | `(3, 0)` | 0 | 30 | `(1, 4)` |
| C | `(3, 0)` | 1 | 31 | `(2, 4)` |

The coordinates are relative to their tile. A cache-relative point position is
reconstructed as:

```text
x_cache = x_relative + tile_x * tile_size
y_cache = y_relative + tile_y * tile_size
```

The layer transform later places those cache-relative coordinates in the
napari/world coordinate system.

The deterministic tile hash assigns:

```text
tile B    -> bucket 0
tile A,C  -> bucket 1
```

This assignment is intentionally not global spatial order. Within each bucket,
however, tiles follow `(tile_y, tile_x)` order, and points within each tile
follow `(value_id, point_id)` order.

## 4. Tile-major bucket payloads

### Bucket 0

Bucket 0 contains only tile B, so its point arrays are:

```text
bucket point row   tile   location    value_id   point_id
0                  B      (0.5, 2)        1         20
1                  B      (3, 1)          2         21
2                  B      (4, 1)          2         22
```

The physical arrays are:

```text
location    = [(0.5, 2), (3, 1), (4, 1)]
value_id    = [1, 2, 2]
point_id    = [20, 21, 22]

tile_x      = [2]
tile_y      = [0]
tile_offset = [0, 3]
```

`tile_offset[0:2] = [0, 3]` says that bucket-local tile 0 occupies bucket point
rows `[0:3]`.

Because the tile contains values 1 and 2, its sparse range arrays are:

```text
ranges/tile_indptr = [0, 2]
ranges/value_id    = [1, 2]
ranges/row_start   = [0, 1]
ranges/row_count   = [1, 2]
```

The two aligned range rows mean:

```text
value 1 -> bucket point rows [0:1]
value 2 -> bucket point rows [1:3]
```

### Bucket 1

Bucket 1 contains tile A followed by tile C:

```text
bucket point row   tile   location   value_id   point_id
0                  A      (1, 1)         0         10
1                  A      (2, 1)         0         11
2                  A      (4, 3)         2         12
3                  C      (1, 4)         0         30
4                  C      (2, 4)         1         31
```

The physical arrays are:

```text
location    = [(1, 1), (2, 1), (4, 3), (1, 4), (2, 4)]
value_id    = [0, 0, 2, 0, 1]
point_id    = [10, 11, 12, 30, 31]

tile_x      = [0, 3]
tile_y      = [0, 0]
tile_offset = [0, 3, 5]
```

The tile point intervals are:

```text
bucket-local tile 0, tile A -> point rows [0:3]
bucket-local tile 1, tile C -> point rows [3:5]
```

The sparse range arrays are:

```text
ranges/tile_indptr = [0, 2, 4]
ranges/value_id    = [0, 2, 0, 1]
ranges/row_start   = [0, 2, 3, 4]
ranges/row_count   = [2, 1, 1, 1]
```

They expand to:

```text
range row   tile   value   bucket point interval
0           A       0          [0:2]
1           A       2          [2:3]
2           C       0          [3:4]
3           C       1          [4:5]
```

`ranges/tile_indptr` points into the range arrays. `ranges/row_start` and
`ranges/row_count` point into the bucket point arrays. They are different
address spaces.

## 5. Cache-wide value totals

At Exact level, the value totals are:

```text
alpha, value 0 -> 3 points
beta,  value 1 -> 2 points
gamma, value 2 -> 3 points
```

The aligned root array is therefore:

```text
values/n_points = [3, 2, 3]
```

Coarser sampled levels can have different per-value totals. Their counts are
represented by their `value_tiles/n_points` records and value-major pointers,
not by replacing these canonical Exact/source totals.

## 6. Manifest: global logical tiles to physical buckets

The manifest contains one row per nonempty logical tile, ordered globally by
`(level, tile_y, tile_x)`. The array position is the implicit
`manifest_index`:

```text
manifest_  tile   bucket_  bucket_tile_  tile_  tile_  n_
index             id       index         x      y      points
----------------------------------------------------------------
0          A      1        0             0      0      3
1          B      0        0             2      0      3
2          C      1        1             3      0      2
```

The physical arrays are:

```text
manifest/level_indptr      = [0, 3]
manifest/bucket_id         = [1, 0, 1]
manifest/bucket_tile_index = [0, 0, 1]
manifest/tile_x            = [0, 2, 3]
manifest/tile_y            = [0, 0, 0]
manifest/n_points          = [3, 3, 2]
```

For example, manifest row 2 says:

```text
logical tile C at (3, 0)
    -> level 0, bucket 1
    -> bucket-local tile 1
    -> 2 complete-tile points
```

`manifest/level_indptr = [0, 3]` says level 0 owns manifest rows `[0:3]`.
With more levels, all manifest arrays remain flat and each adjacent pointer
pair gives one level's global manifest interval.

At runtime, the reader retains these compact arrays and additionally constructs
an in-memory mapping:

```text
(level, tile_x, tile_y) -> manifest_index
```

The manifest therefore supports both directions needed by planning:

```text
logical viewport tile -> manifest row
manifest row -> logical coordinates and physical bucket address
```

## 7. `value_tiles`: values to manifest tiles

The manifest does not answer which tiles contain a requested value.
`value_tiles` is the cache-wide inverted index from `(level, value_id)` to
manifest tiles.

The conceptual records for the example are:

```text
value_id   manifest_index   n_points   meaning
0          0                2          alpha has 2 points in tile A
0          2                1          alpha has 1 point  in tile C
1          1                1          beta  has 1 point  in tile B
1          2                1          beta  has 1 point  in tile C
2          0                1          gamma has 1 point  in tile A
2          1                2          gamma has 2 points in tile B
```

The physical arrays omit `value_id` because the pointer table makes it
implicit:

```text
value_tiles/indptr         = [[0, 2, 4, 6]]
value_tiles/manifest_index = [0, 2, 1, 2, 0, 1]
value_tiles/n_points       = [2, 1, 1, 1, 1, 2]
```

The pointer row is interpreted as:

```text
level 0, value 0 -> value_tiles rows [0:2]
level 0, value 1 -> value_tiles rows [2:4]
level 0, value 2 -> value_tiles rows [4:6]
```

For example:

```text
value_tiles/manifest_index[4:6] = [0, 1]
value_tiles/n_points[4:6]       = [1, 2]
```

says that gamma occurs in manifest tiles A and B with one and two points,
respectively.

The `value_tiles` row number is not a manifest index. It is a position in the
value-to-tile record arrays; the value stored at
`value_tiles/manifest_index[value_tile_row]` is the associated manifest index.

With multiple levels, `value_tiles/indptr` has shape `(L, V + 1)`. Its pointers
are global offsets into the two flat record arrays, whose level sections are
concatenated in serialized level order.

## 8. Current tile-major lookups

### All values in one visible tile

For all values in tile C:

```text
(level 0, tile_x 3, tile_y 0)
    -> in-memory manifest lookup
    -> manifest_index 2
    -> bucket_id 1, bucket_tile_index 1
    -> bucket 1 tile_offset[1:3] = [3, 5]
    -> bucket 1 location[3:5]
    -> bucket 1 value_id[3:5]
```

No sparse value ranges are needed because the complete tile is selected.

### One selected value in visible tiles

For gamma, value 2, in visible tiles A and B:

```text
(level 0, value 2)
    -> value_tiles rows [4:6]
    -> manifest rows [0, 1], counts [1, 2]
    -> intersect with visible manifest rows
```

Manifest row 0 resolves to bucket 1, bucket-local tile 0:

```text
ranges/tile_indptr[0:2] = [0, 2]
    -> search ranges/value_id[0:2] = [0, 2]
    -> value 2 has row_start 2, row_count 1
    -> bucket 1 location[2:3]
```

Manifest row 1 resolves to bucket 0, bucket-local tile 0:

```text
ranges/tile_indptr[0:2] = [0, 2]
    -> search ranges/value_id[0:2] = [1, 2]
    -> value 2 has row_start 1, row_count 2
    -> bucket 0 location[1:3]
```

This path explains the distinct responsibilities:

```text
value_tiles   answers: which manifest tiles contain the value?
manifest      answers: which physical bucket and bucket-local tile?
bucket ranges answer: where are that value's point rows in the bucket?
```

## 9. Construction transpose and `ordered_row_start`

The bucket range records reach catalog construction in physical bucket order,
then tile order within each bucket, then value order within each tile.

For this example, bucket 0 is visited before bucket 1:

```text
input  value_id  manifest_index  row_start  n_points  source
0      1         1               0          1         bucket 0, tile B
1      2         1               1          2         bucket 0, tile B
2      0         0               0          2         bucket 1, tile A
3      2         0               2          1         bucket 1, tile A
4      0         2               3          1         bucket 1, tile C
5      1         2               4          1         bucket 1, tile C
```

Catalog construction sorts these records by `(value_id, manifest_index)`:

```text
output  value_id  manifest_index  row_start  n_points
0       0         0               0          2
1       0         2               3          1
2       1         1               0          1
3       1         2               4          1
4       2         0               2          1
5       2         1               1          2
```

It publishes:

```text
value_tiles/manifest_index = [0, 2, 1, 2, 0, 1]
value_tiles/n_points       = [2, 1, 1, 1, 1, 2]
value_tiles/indptr         = [[0, 2, 4, 6]]
```

It writes the identically permuted source addresses to a temporary disk-backed
array:

```text
ordered_row_start = [0, 3, 0, 4, 2, 1]
```

This array is meaningful only when aligned with the published value-tile rows:

```text
manifest_index   ordered_row_start   n_points
0                0                   2
2                3                   1
1                0                   1
2                4                   1
0                2                   1
1                1                   2
```

`manifest_index` resolves the source bucket, `ordered_row_start` resolves the
first row in that bucket's point arrays, and `n_points` gives the consecutive
row count. The same `row_start` can occur in different buckets, which is why it
cannot be interpreted without the manifest record.

`ordered_row_start` is not part of the published cache. It is a construction
aid used to copy coordinates into the sidecar and is deleted with its temporary
directory afterward.

## 10. Value-major sidecar

Every serialized level contains a coordinate-only sidecar ordered by
`(value_id, manifest_index, point_id)`. Following the sorted records above
produces:

```text
sidecar row   value   manifest tile   point ID   location
0             0       A               10         (1, 1)
1             0       A               11         (2, 1)
2             0       C               30         (1, 4)
3             1       B               20         (0.5, 2)
4             1       C               31         (2, 4)
5             2       A               12         (4, 3)
6             2       B               21         (3, 1)
7             2       B               22         (4, 1)
```

Only the locations and compact value pointers are persisted:

```text
value_major/level_0/location = [
    (1, 1), (2, 1), (1, 4),
    (0.5, 2), (2, 4),
    (4, 3), (3, 1), (4, 1),
]

value_major/level_0/value_point_indptr = [0, 3, 5, 8]
```

The pointer intervals are:

```text
value 0 -> sidecar coordinate rows [0:3]
value 1 -> sidecar coordinate rows [3:5]
value 2 -> sidecar coordinate rows [5:8]
```

Point-level value IDs are unnecessary because each value is implicit from its
pointer interval. Point IDs are unnecessary because they established the
deterministic order during bucket construction. Manifest identities are not
duplicated because the existing ordered `value_tiles` records supply them.

For gamma, the sidecar returns three coordinates from `[5:8]`. The aligned
gamma value-tile records are:

```text
manifest_index = [0, 1]
n_points       = [1, 2]
```

Those counts partition the coordinate interval:

```text
first 1 coordinate  -> manifest tile A
next  2 coordinates -> manifest tile B
```

The manifest supplies each tile's logical coordinates, allowing the consumer
to add the correct tile origin to the tile-relative sidecar coordinates.

The current schema, writer, and validation require this sidecar at every level.
The current visualization reader still uses the tile-major bucket path; routing
proper-subset display reads through the sidecar is a separate runtime change.

## 11. Persisted, resident, selected, and temporary state

The current implementation treats these structures differently:

| Structure | Published on disk | Normally retained in runtime memory | Notes |
|---|:---:|:---:|---|
| Root attributes and level descriptors | Yes | Yes | Generation, geometry, vocabulary, and physical contract |
| `values/n_points` | Yes | Yes | Canonical Exact/source value totals |
| Manifest arrays | Yes | Yes | Small cache-wide logical-tile catalog |
| `value_tiles/indptr` | Yes | Yes | Compact level/value pointer table |
| Complete `value_tiles/manifest_index` and `n_points` | Yes | No | Selected intervals are loaded when the active proper subset changes |
| Selected-value index | No | Yes, for the active proper subset | Compact copies of selected manifest/count records, reused across viewports |
| Bucket lookup arrays | Yes | Yes in the current cache session | `tile_offset` and four `ranges/*` arrays; point payloads remain on disk |
| Bucket `location`, point-level `value_id`, and `point_id` | Yes | No as lookup metadata | Decoded only for requested payloads; decoded tiles may enter CPU residency |
| Value-major `location` and `value_point_indptr` | Yes | Not yet used by the display reader | Mandatory format payload at every level |
| `bucket_manifest_indexes` | No | Construction only | In-memory translation from bucket-local tile index to manifest index |
| `ordered_row_start` | No | Construction only, disk-backed | Cache-wide temporary companion aligned with sorted value-tile rows |

The current bucket lookup object retains exactly:

```text
tile_offset
ranges/tile_indptr
ranges/value_id
ranges/row_start
ranges/row_count
```

It does not retain coordinates, point-level value IDs, or point IDs.

## 12. Pointer glossary

Every pointer uses a half-open interval `[start:stop]`:

```text
manifest/level_indptr[level : level + 2]
    -> rows in the flat manifest arrays for one level

bucket/tile_offset[bucket_tile_index : bucket_tile_index + 2]
    -> rows in that bucket's point arrays for one complete tile

bucket/ranges/tile_indptr[bucket_tile_index : bucket_tile_index + 2]
    -> rows in that bucket's sparse range arrays for one tile

value_tiles/indptr[level, value_id : value_id + 2]
    -> rows in the flat value-to-tile arrays for one level/value

value_major/level_L/value_point_indptr[value_id : value_id + 2]
    -> rows in that level's value-major location array for one value
```

The resulting lookup responsibilities are:

```text
manifest_index
    answers: which logical tile and physical bucket?

bucket_tile_index
    answers: which tile-local pointer entries inside that bucket?

ranges/row_start + ranges/row_count
    answer: which consecutive point rows inside that bucket?

value_tiles
    answers: which manifest tiles contain a level/value, and how many points
             of that value does each tile contribute?

value_point_indptr
    answers: which contiguous sidecar coordinate rows belong to one value?
```

Together, these structures provide two physical coordinate orderings under one
logical cache generation:

```text
tile-major buckets
    optimized for complete logical tiles and all-value reads

value-major sidecars
    optimized for proper value subsets that span many tiles
```
