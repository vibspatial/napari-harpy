## Bottom line

Xenium Explorer is architecturally close to Harpy in one important respect: both use a precomputed, spatially tiled point pyramid with an exact finest level and progressively coarser levels.

However, its coarse-level semantics are fundamentally different:

- Harpy retains a deterministic, value-neutral subset of real transcripts.
- Xenium clusters transcripts separately by gene and stores a `cluster_count` for each displayed aggregate.

Consequently, Xenium preserves the exact transcript count represented by a coarse point and avoids losing rare genes. Harpy preserves representative transcript identity and unbiased dot-density sampling, but individual genes can disappear at coarse levels.

Explorer also has a precomputed density-map fallback. Harpy explicitly remains point-only. That is probably the largest reason Explorer can remain responsive under extreme all-gene loads.

## What Xenium Explorer stores

Explorer normally reads `transcripts.zarr.zip`, not `transcripts.parquet`. The latter is the canonical tabular output for downstream analysis; the Zarr is the purpose-built visualization representation. [10x’s format documentation](https://www.10xgenomics.com/support/software/xenium-onboard-analysis/latest/advanced/xoa-output-zarr) describes:

```text
transcripts.zarr.zip
├── grids/                       # tiled transcript pyramid
│   ├── 0/                       # exact, most zoomed-in level
│   │   ├── x,y/
│   │   │   ├── location
│   │   │   ├── gene_identity
│   │   │   ├── quality_score
│   │   │   ├── id / uuid
│   │   │   └── gene_offset
│   ├── 1/
│   └── ...
├── density/gene/                # 10-µm sparse gene-density grid
├── density/codeword/
├── gene_category
└── codeword_category
```

The grid has a nominal 250-µm tile size at level 0 and doubles spatially at coarser levels. Level 0 contains every transcript; the most zoomed-out level typically fits in one spatial tile.

Exact points use global `float32` XYZ coordinates, a `uint16` gene identity, Q-score, codeword, ID, UUID, and status fields. Arrays are stored separately and compressed as Zarr chunks using Blosc/Zstd.

### Modern scaled-transcript levels

Since XOA/Explorer 3.0, coarse levels are not ordinary random subsamples. For every gene, neighboring transcripts are grouped using a radius that starts at 16 image pixels and doubles as the view becomes coarser. Explorer draws one larger point whose size is proportional to the square root of its `cluster_count`, and can label it with the represented count. [10x describes this behavior here](https://www.10xgenomics.com/support/software/xenium-explorer/latest/tutorials/interface-and-features/nav-transcripts).

I inspected the official tiny XOA 3.0 example from the [10x example-dataset page](https://www.10xgenomics.com/support/software/xenium-onboard-analysis/latest/resources/xenium-example-data). In that archive:

- Level counts were 1,985 exact points → 1,180 aggregates → 1,110 aggregates.
- `cluster_count` was `uint32`.
- The sum of `cluster_count` remained exactly 1,985 at both coarse levels.
- The low-Q/high-Q totals also remained exactly preserved: 1,445 and 540.
- Coarse locations were nested representatives: every aggregate location matched a real exact transcript location in this example, rather than an invented centroid.
- Coarse points had no transcript ID, however, and Explorer explicitly treats them as aggregates without transcript-specific tooltips.
- `gene_offset` had four range fields per gene: low-Q start/end and high-Q start/end. Points in each tile are therefore arranged so Explorer can locate a selected gene and quality class by ranges rather than filtering an unordered mixed collection.

The last point is especially relevant to Harpy. Xenium’s layout is both spatially tiled and gene-addressable within each tile.

### Density fallback

Explorer separately stores a sparse, per-gene 10-µm density grid. Its default maximum is five million transcript points in the viewport; above that it switches automatically to the density-map view. Thus Explorer’s robust overview is a hybrid point/raster solution, not solely the point pyramid. [10x documents the limit and fallback](https://www.10xgenomics.com/support/software/xenium-explorer/latest/tutorials/interface-and-features/nav-transcripts).

## Comparison with Harpy

| Aspect | Xenium Explorer | Harpy design |
|---|---|---|
| Finest level | Exact tiled transcripts | Exact tiled transcripts |
| Coarse representation | Per-gene spatial aggregates with exact `cluster_count` | Value-neutral spatial sample of real transcripts |
| Level nesting | Observed as nested in the inspected v3 archive | Explicit contract: coarse level is a subset of finer level |
| Rare genes | Protected because clustering is per gene | May disappear through neutral sampling |
| Abundance | Exactly represented through aggregate weights | Approximately represented through retained dot density |
| Point identity | Exact level only; aggregates are not individual transcripts | Every sampled representative retains `point_id` |
| Value selection | `gene_offset` gives direct per-gene ranges within each tile | Sidecar counts skip empty tiles, but positive mixed-value row groups are initially read completely |
| Extreme overview | Density-map fallback | Hard-bounded point-only overview |
| Tile size | Initially 250 µm, then doubles | Initially 512 native units, then doubles; includes an equal-geometry Bridge |
| Storage | Zipped Zarr; separate typed arrays per tile | Parquet row groups plus manifest and sidecar indexes |
| Coordinates | Global `float32` XYZ | Tile-local `float32` XY, reconstructed with tile origin |
| Q-score | First-class filtering/index dimension | Initial cache selects only X, Y, and value |
| Runtime internals | Proprietary/undocumented | Explicit CPU/GPU LRUs, scheduler and retained tile VBO design |

For the current Harpy Xenium dataset, 512 native units correspond to about 108.8 µm because the SpatialData transform is 0.2125 µm/unit: [zarr.json](/Users/arne.defauw/VIB/DATA/test_data/sdata_xenium_full_data_core.zarr/zarr.json:1771). Harpy’s finest tiles are therefore considerably smaller than Explorer’s 250-µm tiles.

Harpy’s physical payload is leaner—tile-local coordinates, value ID and stable point ID only: [support.py](/Users/arne.defauw/VIB/napari_harpy/src/napari_harpy/core/multi_scale_cache_points/writer/support.py:27). Tile-local coordinates should also retain better `float32` precision over large specimens than Explorer’s global coordinates.

The implemented Harpy Bridge caps each 512-unit tile at 4,096 representatives and uses the value-neutral sampler: [bridge.py](/Users/arne.defauw/VIB/napari_harpy/src/napari_harpy/core/multi_scale_cache_points/writer/bridge.py:276). That sampler explicitly excludes `value_id`: [sampling.py](/Users/arne.defauw/VIB/napari_harpy/src/napari_harpy/core/multi_scale_cache_points/sampling.py:81). The measured Bridge retained 21.7 million of the 136.6 million transcripts; the remaining pyramid and runtime viewer are not implemented yet: [roadmap](/Users/arne.defauw/VIB/napari_harpy/Roadmap/transcripts_visualization/multi_tile_cache_29_7_26.md:2586).

## What I would take from Xenium

I would not replace the current Harpy architecture wholesale. Its tiling, manifest, deterministic identities, local-coordinate precision, source freshness, CPU/GPU cache separation and retained VBO plan are stronger and more explicit than anything publicly documented for Explorer.

I would revisit three design questions before freezing the file format:

1. Aggregate weights and visual semantics

   Explorer demonstrates that count-bearing aggregates provide a much more interpretable coarse view: a point says how many transcripts it represents. Harpy’s sampled points cannot make that claim.

   This is not solved merely by adding a `weight` column. Exact per-gene weights require clustering separately by value, as Xenium does. That changes the current value-neutral semantics and may greatly increase the number of coarse points for 5,000-gene panels.

2. Physically value-selective reads

   Harpy’s `tile_value_counts.parquet` is excellent for planning, but the initial reader still loads every value from a positive tile and filters afterward: [roadmap](/Users/arne.defauw/VIB/napari_harpy/Roadmap/transcripts_visualization/multi_tile_cache_29_7_26.md:2059).

   Xenium’s `gene_offset` layout is concrete evidence that sorting/indexing tile payloads by value is useful. A Harpy equivalent could be value-contiguous row groups, offsets, or value shards. It should be benchmark-driven because it may hurt all-values reads and increase metadata.

3. Density fallback

   If the requirement is specifically “behave as robustly as Explorer for all genes over the complete specimen,” Harpy’s point-only constraint should be recognized as a deliberate product difference. Explorer does not solve every extreme view with points; it falls back to a precomputed density representation.

   Harpy’s bounded point overview can still work well, but it will show sampled transcript representatives rather than exact mass. That tradeoff should be evaluated visually against Explorer using the same full dataset.

One limitation of the investigation is that Explorer’s renderer and scheduler are closed. The format strongly implies viewport-based tile selection and zoom-level switching, but public evidence does not establish whether it retains GPU buffers, uses an LRU, performs atomic LOD transitions, or how it schedules asynchronous reads. Those parts of the Harpy roadmap should be evaluated on their own merits rather than claimed as copies of Explorer.

No repository files were modified.



Other comments:

Zarr is better suited to exploiting `gene_offset`, but not especially to storing the offset table itself. The advantage comes from independently readable chunks and natural array slicing.

A Xenium-style Zarr tile can look like:

```text
location      (N, 2) float32
value_id      (N,)   uint16/uint32
point_id      (N,)   uint64
gene_offset   (G+1,) uint32
```

Points are sorted by gene. For gene `g`:

```text
start = gene_offset[g]
stop  = gene_offset[g + 1]

coordinates = location[start:stop]
point_ids   = point_id[start:stop]
```

This maps naturally onto Zarr’s array slicing API. The important part is that Zarr chunks are independently compressed and are the smallest independently readable units. Zarr v3 sharding can place many such chunks in one storage object while retaining independent chunk reads, avoiding a file-per-chunk explosion. [Zarr array and sharding documentation](https://zarr.readthedocs.io/en/latest/user-guide/arrays/)

### Zarr versus the current Parquet layout

| Property | Zarr | Current Harpy Parquet |
|---|---|---|
| Select arbitrary row range | Natural array slice | Normally read row group, then slice |
| Physical read unit | Chunk | Row group/column chunk |
| Many small independent units | Zarr v3 shards can contain many chunks | Many row groups increase footer and scheduling overhead |
| Separate coordinate/ID arrays | Natural | Already columnar, but columns share row-group boundaries |
| Tabular inspection/querying | Less convenient | Excellent |
| Existing Harpy implementation | Would require a new payload writer/store | Already implemented |

PyArrow exposes individual Parquet row-group reads, but not an equivalently straightforward “read rows 1234–1250 from these columns” operation. [Apache Arrow Parquet documentation](https://arrow.apache.org/docs/python/parquet.html)

### Chunking remains decisive

Zarr does not make `gene_offset` magically selective.

Suppose a tile has 20,000 points:

```text
gene range:       rows 8,120–8,135
Zarr chunk size:  4,096 rows
```

Only the chunk containing those rows needs to be read—roughly 4,096 points instead of 20,000.

But if each per-tile Zarr array is stored as one chunk:

```text
chunk size = complete 20,000-point tile
```

then selecting the gene still decompresses the entire tile. The offset only saves the in-memory mask and gather, exactly like gene-sorted Parquet with one tile-sized row group.

So the useful combination is:

```text
gene-contiguous rows
+ gene offsets
+ multiple independently compressed chunks along the point dimension
```

Zarr’s chunking model represents that combination more naturally than Parquet.

### Is it better for Harpy specifically?

Potentially, but I would not replace Parquet wholesale yet.

Harpy’s average exact tile is only about 18,725 points, and the largest measured tile was 108,598 points. The largest warm Parquet tile read took 1.15 ms ([roadmap](/Users/arne.defauw/VIB/napari_harpy/Roadmap/transcripts_visualization/multi_tile_cache_29_7_26.md:1911)). Sampled Bridge tiles are capped at only 4,096 points.

For a few visible tiles, Zarr is unlikely to make a dramatic difference. The gain would appear when a selected gene occurs sparsely across many positive tiles: Zarr could read only the chunks covering that gene’s ranges instead of decompressing every complete tile.

There are also costs:

- Small chunks can cause many reads.
- A group containing separate arrays for every tile can create considerable metadata.
- Fixed-size chunks will not align perfectly with variable-length gene ranges.
- All-gene views must read every coordinate chunk.
- Harpy would need a new payload storage implementation alongside its existing Parquet writer, manifest and validation machinery.

Zarr’s own guidance emphasizes that chunk shape should be chosen around the expected access pattern; smaller chunks improve selectivity but create more chunk operations. [Zarr performance guidance](https://zarr.readthedocs.io/en/latest/user-guide/performance/)

### The most attractive design may be hybrid

I would keep the tabular metadata in Parquet:

```text
manifest.parquet
values.parquet
tile_value_counts.parquet
```

and experimentally store only the point payload in Zarr:

```text
level_0/
  location
  point_id
  gene_offset or sparse value ranges
```

The current environment already contains Zarr 3.2.1, so Zarr v3 sharding is technically available. The experiment should compare:

1. Current one-Parquet-row-group-per-tile layout.
2. Parquet with a bounded number of value-partitioned row groups.
3. Zarr with gene-sorted arrays, fixed point chunks and v3 sharding.

Measure both single-gene and all-gene views. My expectation is:

- Zarr wins ergonomically for true offset-based slicing.
- It may win cold selected-gene latency when read amplification is high.
- Parquet likely remains perfectly adequate for small or cached tiles.
- The choice will make little difference unless the physical Zarr chunks are appreciably smaller than a tile.

So: **yes, Zarr is the more natural representation for Xenium-style access, but the performance benefit comes from chunk layout—not from the `gene_offset` array itself.**



And then next:

Zarr does not require image-shaped data. A point cloud is simply an array whose first dimension is the point index:

```text
image:     (height, width, channels)
locations: (number_of_points, coordinate_dimensions)
```

For 2D transcripts, `location` is an `(N, 2)` `float32` array.

## A single tile

Conceptually, one tile could contain:

```text
level_0/
  tile_0003_0007/
    location       shape=(N, 2)   dtype=float32
    point_id       shape=(N,)     dtype=uint64
    value_id       shape=(N,)     dtype=uint32
    gene_offset    shape=(G + 1,) dtype=uint32
```

All point arrays are aligned row-by-row:

```text
row       location       point_id    value_id
0         [12.3, 17.8]   9001        0
1         [18.1, 21.4]   9017        0
2         [40.2, 11.2]   8402        2
3         [41.7, 13.9]   9128        2
4         [45.0, 18.1]   9344        2
5         [82.2, 74.1]   8701        3
```

Because the points are sorted by `value_id`, the corresponding offsets are:

```text
gene_offset = [0, 2, 2, 5, 6]
```

This means:

```text
gene 0 → rows [0:2]
gene 1 → rows [2:2]  # absent
gene 2 → rows [2:5]
gene 3 → rows [5:6]
```

Selecting gene 2 means taking the identical slice from every aligned array:

```text
location[2:5]
point_id[2:5]
```

Reading `value_id` is unnecessary for a single-gene selection because the range already identifies the gene.

Harpy could continue storing tile-relative coordinates, as it does now when calculating `x_rel` and `y_rel` ([exact.py](/Users/arne.defauw/VIB/napari_harpy/src/napari_harpy/core/multi_scale_cache_points/writer/exact.py:306)). The tile origin is added when loading.

## Avoiding thousands of per-tile groups

The layout above is intuitive, but Harpy has 7,294 exact tiles. Four arrays per tile would mean roughly 30,000 Zarr arrays and their associated metadata.

A more scalable layout is to flatten all tiles within a level:

```text
level_0/
  location           shape=(N_total, 2)
  point_id           shape=(N_total,)
  value_id           shape=(N_total,)
  tile_offset        shape=(T + 1,)
```

Points are ordered first by tile and then by gene:

```text
tile 0: gene 0, gene 1, gene 2, ...
tile 1: gene 0, gene 1, gene 2, ...
...
```

`tile_offset` works like a CSR `indptr` array:

```text
tile_offset = [0, 12000, 30725, 45002, ...]
```

Therefore:

```text
tile 0 → rows [0:12000]
tile 1 → rows [12000:30725]
tile 2 → rows [30725:45002]
```

This “flattened values plus offsets” pattern is how variable-length/ragged data is normally represented in ordinary numeric arrays.

## Dense versus sparse gene offsets

A dense index could be:

```text
gene_offset shape=(number_of_tiles, number_of_genes + 1)
```

Each row contains gene offsets relative to that tile.

With approximately 7,300 tiles and 5,122 genes, however, that is about 37 million offsets. It would compress because many adjacent offsets repeat, but it is unnecessarily large.

A sparse range representation is preferable:

```text
tile_gene_indptr  shape=(T + 1,)
gene_id           shape=(M,)
row_start         shape=(M,)
row_count         shape=(M,)
```

Here, `M` is the number of nonempty `(tile, gene)` combinations.

For example:

```text
tile_gene_indptr = [0, 3, 5]

gene_id   = [0, 2, 7,   1, 7]
row_start = [0, 9, 14,  20, 26]
row_count = [9, 5, 6,    6, 4]
             tile 0       tile 1
```

This says:

```text
tile 0:
  gene 0 → rows [0:9]
  gene 2 → rows [9:14]
  gene 7 → rows [14:20]

tile 1:
  gene 1 → rows [20:26]
  gene 7 → rows [26:30]
```

This sparse index is effectively the physical counterpart of Harpy’s current `tile_value_counts`, which currently stores only counts and does not locate point rows ([roadmap](/Users/arne.defauw/VIB/napari_harpy/Roadmap/transcripts_visualization/multi_tile_cache_29_7_26.md:1294)).

I would retain the Parquet count index for gene-to-tile planning and use the Zarr sparse ranges for tile-to-gene payload lookup.

## Chunking the point arrays

The arrays should use aligned chunks along the point dimension:

```text
location   chunks=(4096, 2)
point_id   chunks=(4096,)
value_id   chunks=(4096,)
```

The number 4,096 is illustrative, not a recommendation yet.

If a gene occupies rows `[8120:8140]`, Zarr reads the chunk containing those rows. If it spans a chunk boundary, it reads two chunks. It does not need to decompress the complete tile.

There is an unavoidable tradeoff:

- Smaller chunks: better single-gene selectivity, more chunk operations.
- Larger chunks: better sequential/all-gene throughput, more unwanted points per gene read.

Candidate sizes such as 2,048, 4,096, 8,192 and 16,384 rows should be benchmarked.

Zarr v3 sharding lets multiple independently compressed chunks live inside a larger storage object. Thus, for example, 32 point chunks could share one shard while remaining independently readable. This reduces filesystem-object proliferation. [Zarr sharding documentation](https://zarr.readthedocs.io/en/latest/user-guide/arrays/#sharding)

## Recommended Harpy experiment

I would prototype this payload:

```text
level_0/
  location             (N, 2) float32
  point_id             (N,)   uint64
  value_id             (N,)   uint32
  tile_offset          (T+1,) uint64

  ranges/
    tile_indptr        (T+1,) uint64
    gene_id            (M,)   uint32
    row_start          (M,)   uint64
    row_count          (M,)   uint32
```

Keep these as Parquet:

```text
manifest.parquet
values.parquet
tile_value_counts.parquet
```

This avoids thousands of tile groups, preserves efficient gene-to-tile planning, and uses Zarr specifically where it is strongest: aligned numeric arrays and range-based chunk reads. The essential mental model is not “points stored like an image”; it is “a columnar point table stored as several aligned one-dimensional arrays.”



Example:

Think of the format as three cooperating indexes around one flattened point table:

```text
viewport → manifest → visible tiles
selected gene → tile_value_counts → positive visible tiles
(tile, gene) → ranges → rows in the Zarr point arrays
```

The Zarr arrays contain the actual points. The Parquet files tell the viewer which parts of those arrays it should read.

## Small example

Suppose level 0 has three spatial tiles and three genes:

| `value_id` | Gene |
|---:|---|
| 0 | ACTB |
| 1 | MALAT1 |
| 2 | EPCAM |

The tiles contain:

```text
tile 0:
    ACTB   point 101 at (1, 1)
    ACTB   point 102 at (2, 1)
    EPCAM  point 103 at (4, 4)

tile 1:
    MALAT1 point 104 at (0.5, 3)
    MALAT1 point 105 at (1, 3)
    EPCAM  point 106 at (3, 2)
    EPCAM  point 107 at (3.5, 2)

tile 2:
    ACTB   point 108 at (1, 4)
    MALAT1 point 109 at (2, 4)
    MALAT1 point 110 at (2.5, 4)
```

Before writing, points are ordered by:

```text
(tile_index, value_id, point_id)
```

All points are then flattened into one Zarr array.

### The flattened point arrays

```text
row   tile   value_id   location      point_id
---   ----   --------   ------------  --------
 0      0        0      [1.0, 1.0]       101
 1      0        0      [2.0, 1.0]       102
 2      0        2      [4.0, 4.0]       103

 3      1        1      [0.5, 3.0]       104
 4      1        1      [1.0, 3.0]       105
 5      1        2      [3.0, 2.0]       106
 6      1        2      [3.5, 2.0]       107

 7      2        0      [1.0, 4.0]       108
 8      2        1      [2.0, 4.0]       109
 9      2        1      [2.5, 4.0]       110
```

Physically, these are three aligned arrays:

```text
location = [
    [1.0, 1.0],
    [2.0, 1.0],
    [4.0, 4.0],
    [0.5, 3.0],
    [1.0, 3.0],
    [3.0, 2.0],
    [3.5, 2.0],
    [1.0, 4.0],
    [2.0, 4.0],
    [2.5, 4.0],
]

value_id = [
    0, 0, 2,
    1, 1, 2, 2,
    0, 1, 1,
]

point_id = [
    101, 102, 103,
    104, 105, 106, 107,
    108, 109, 110,
]
```

The same row always identifies the same point across all three arrays.

## What `tile_offset` does

```text
tile_offset = [0, 3, 7, 10]
```

This has `number_of_tiles + 1` entries.

It means:

```text
tile 0 → rows tile_offset[0]:tile_offset[1] → [0:3]
tile 1 → rows tile_offset[1]:tile_offset[2] → [3:7]
tile 2 → rows tile_offset[2]:tile_offset[3] → [7:10]
```

Therefore, an all-gene request for tile 1 is simply:

```text
location[3:7]
value_id[3:7]
point_id[3:7]
```

No gene-range index is needed for the all-gene case.

## What the sparse ranges do

Within each tile, points are grouped by gene. There are six nonempty `(tile, gene)` combinations:

```text
tile 0, ACTB
tile 0, EPCAM
tile 1, MALAT1
tile 1, EPCAM
tile 2, ACTB
tile 2, MALAT1
```

Therefore, `M = 6`.

The sparse range arrays contain:

```text
gene_id   = [0, 2,  1, 2,  0, 1]
row_start = [0, 2,  3, 5,  7, 8]
row_count = [2, 1,  2, 2,  1, 2]
```

They describe:

| Range entry | Tile | Gene | Point rows |
|---:|---:|---|---|
| 0 | 0 | ACTB | `[0:2]` |
| 1 | 0 | EPCAM | `[2:3]` |
| 2 | 1 | MALAT1 | `[3:5]` |
| 3 | 1 | EPCAM | `[5:7]` |
| 4 | 2 | ACTB | `[7:8]` |
| 5 | 2 | MALAT1 | `[8:10]` |

But those arrays do not themselves say which range records belong to each tile. That is the purpose of `tile_indptr`.

### `tile_indptr`

```text
tile_indptr = [0, 2, 4, 6]
```

It partitions the range arrays:

```text
tile 0 → range entries [0:2]
tile 1 → range entries [2:4]
tile 2 → range entries [4:6]
```

For tile 1:

```text
start = tile_indptr[1]     # 2
stop  = tile_indptr[2]     # 4

gene_id[2:4]   = [1, 2]
row_start[2:4] = [3, 5]
row_count[2:4] = [2, 2]
```

Therefore:

```text
tile 1, MALAT1 → location[3:5]
tile 1, EPCAM  → location[5:7]
```

This is the sparse equivalent of storing a dense `gene_offset` array for every gene in every tile.

## What each Parquet file does

### `values.parquet`

This is the gene dictionary:

| `value_id` | `value` | Exact point count |
|---:|---|---:|
| 0 | ACTB | 3 |
| 1 | MALAT1 | 4 |
| 2 | EPCAM | 3 |

When the user selects `MALAT1`, Harpy translates that label to `value_id = 1`.

### `manifest.parquet`

This is the spatial tile directory:

| `level` | `tile_index` | `tile_x` | `tile_y` | `n_points` |
|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 0 | 3 |
| 0 | 1 | 1 | 0 | 4 |
| 0 | 2 | 0 | 1 | 3 |

The planner intersects the viewport with the grid and determines which tiles are visible.

For a Zarr-backed payload, the current Parquet-specific manifest would have to change. Its `level_file`, `row_group`, and `tile_shard` fields describe the current Parquet payload; Zarr would instead need a stable `tile_index` connecting each spatial tile to `tile_offset` and `tile_indptr`.

### `tile_value_counts.parquet`

This is the gene-to-tile index:

| `level` | `value_id` | `tile_index` | `n_points` |
|---:|---:|---:|---:|
| 0 | 0 | 0 | 2 |
| 0 | 0 | 2 | 1 |
| 0 | 1 | 1 | 2 |
| 0 | 1 | 2 | 2 |
| 0 | 2 | 0 | 1 |
| 0 | 2 | 1 | 2 |

It answers:

```text
Which tiles contain MALAT1?
→ tiles 1 and 2

How many selected points will be rendered?
→ 2 + 2 = 4
```

This sidecar is organized gene-first, while the Zarr range index is organized tile-first:

```text
tile_value_counts:
    gene → positive tiles

Zarr ranges:
    tile → gene → point rows
```

They contain some overlapping information, but serve opposite query directions.

## Complete MALAT1 lookup

Suppose the viewport intersects all three tiles and the user selects MALAT1.

### 1. Translate the gene

From `values.parquet`:

```text
MALAT1 → value_id 1
```

### 2. Find positive tiles

From `tile_value_counts.parquet`:

```text
value_id 1 exists in:
    tile 1: 2 points
    tile 2: 2 points
```

Tile 0 is eliminated without reading any point payload.

### 3. Locate MALAT1 inside tile 1

```text
tile_indptr[1:3] = [2, 4]
```

So tile 1 uses range records `[2:4]`.

Within those records:

```text
gene_id = [1, 2]
```

Gene 1 has:

```text
row_start = 3
row_count = 2
```

Read:

```text
location[3:5]
point_id[3:5]
```

### 4. Locate MALAT1 inside tile 2

Tile 2 uses range records:

```text
tile_indptr[2:4] = [4, 6]
```

Gene 1 has:

```text
row_start = 8
row_count = 2
```

Read:

```text
location[8:10]
point_id[8:10]
```

The logical result is only four coordinates:

```text
[
    [0.5, 3.0],
    [1.0, 3.0],
    [2.0, 4.0],
    [2.5, 4.0],
]
```

## Where the speedup comes from

With the current Harpy Parquet design:

```text
tile 1: read all 4 points, retain 2
tile 2: read all 3 points, retain 2

total decoded: 7
total returned: 4
```

With the Zarr ranges:

```text
tile 1: request only rows [3:5]
tile 2: request only rows [8:10]

total logically requested: 4
total returned: 4
```

On a real dataset, the difference might be:

```text
Current:
    40 visible positive tiles
    20,000 total points per tile
    800,000 coordinates decoded
    1,200 selected points retained

Range-selective:
    only chunks intersecting the 1,200 selected rows are decoded
```

The actual physical saving depends on chunk size. Zarr always reads complete compressed chunks:

```text
requested gene range: 20 rows
chunk size:           4,096 rows
physically decoded:   up to 4,096 rows
```

It is still potentially much better than decoding a 20,000- or 100,000-point tile, but it is not exactly 20 rows of physical I/O.

## Why both `tile_offset` and sparse ranges?

They optimize different interactions:

```text
Show every gene in tile 1:
    tile_offset → one direct [3:7] slice

Show MALAT1 in tile 1:
    tile_indptr/ranges → direct [3:5] slice
```

`tile_offset` also provides useful validation:

```text
tile_offset[t + 1] - tile_offset[t]
    == manifest.n_points for tile t
```

The sparse range records must cover that same interval without gaps or overlaps.

## A possible simplification

`row_start` is technically derivable from `tile_offset` and the preceding `row_count` values. For example:

```text
first range start = tile_offset[tile]
next range start  = previous start + previous count
```

I would nevertheless keep `row_start` initially because:

- lookup becomes direct;
- validation is clearer;
- the storage cost is small compared with coordinates;
- future sharding or oversized-range rules become easier to represent.

## The mental model

The complete design can be summarized as:

```text
values.parquet
    gene label → numeric gene ID

manifest.parquet
    spatial position → tile index

tile_value_counts.parquet
    gene ID → tiles that contain it

tile_indptr + gene_id
    tile index → its nonempty gene ranges

row_start + row_count
    gene range → exact slice in the point arrays

location/value_id/point_id
    actual point payload
```

The speedup is therefore not that Zarr searches genes faster. No searching happens in the large point arrays. The small indexes calculate the required row slices first, and Zarr reads only the chunks intersecting those slices.