# Visualizing Transcripts In napari

This note is based on the discussion in:

- [scverse/napari-spatialdata#372](https://github.com/scverse/napari-spatialdata/issues/372)
- [scverse/spatialdata#789](https://github.com/scverse/spatialdata/issues/789)
- [scverse/spatialdata#974](https://github.com/scverse/spatialdata/issues/974)

It is intentionally not based on `Roadmap/issue_points_rendering.md`.

## Main takeaway

For transcript visualization at the scale of tens of millions up to one billion points, we should not try to render a generic `SpatialData` points table directly.

We need a dedicated visualization path with:

1. a spatial index;
2. explicit tiling;
3. multiple levels of detail;
4. a small render budget in napari;
5. lazy viewport-driven loading.

The current `points.parquet` + `dask.dataframe` representation is fine as a canonical exact table, but it is not a good interactive rendering format.

## What the GitHub threads imply

### 1. The current path is fundamentally O(N)

The upstream code confirms the same problem described in the issues:

- `spatialdata` writes points as a plain `points.parquet` file inside the store.
- `spatialdata` bounding-box queries on points still materialize the full dataframe.
- `napari-spatialdata` calls `.compute()` on the full points dataframe and then randomly subsamples above `POINT_THRESHOLD=100000`.

So today there is no notion of "load only visible points". There is only "load all points, then throw most of them away".

### 2. Two viable directions came out of the discussion

The issues point to two families of solutions:

- Minimal-change approach: keep Parquet, but sort points by Morton code / Z-order and use Parquet row groups as the unit of spatial loading.
- Storage redesign approach: store points in an explicitly tiled layout, ideally NGFF-like / OME-Zarr-like, so spatial locality is encoded on disk.

The discussion in `spatialdata#974` and the Vitessce example show that Morton-sorted Parquet is already practical for loading millions of points. The discussion in `spatialdata#789` shows that a true tiled store is likely the right long-term direction, but it also highlights real design risks around chunk shape, integer coordinates, and filtering by gene.

### 3. For 1B points, Morton sort alone is not enough

Morton sorting fixes the query problem, but not the rendering problem.

Even if viewport queries become cheap, we still cannot push arbitrarily many exact points into a napari `Points` layer and expect smooth pan/zoom. For whole-slide views, we need a bounded number of representative points. This is exactly why the Neuroglancer spatial-index design is interesting: it combines spatial tiles with multi-resolution sampled levels.

## Recommendation

We should build a hybrid design:

1. keep the canonical exact points table in SpatialData / Parquet;
2. add a visualization-specific tiled cache beside it;
3. drive napari from that cache, not from the raw `dask.dataframe`.

This gives us an implementable path now, without waiting for a new upstream point-storage spec.

## Recommended storage design

### A. Canonical exact representation

Keep the exact transcript table as the source of truth, but preprocess it for spatial locality:

- required columns: `x`, `y`, `gene_id`
- optional columns: `transcript_id`, `cell_id`, quality metrics
- convert gene names to an integer `gene_id`
- optionally add `morton_code`
- sort by `morton_code`
- write with small Parquet row groups

This part aligns with `spatialdata#974` and keeps us compatible with the current ecosystem.

Important detail:

- use `dask` for preprocessing and writing;
- do not use `dask` for the interactive viewport read path.

For interactive reads we should use `pyarrow` directly, because the viewer needs low-latency row-group reads, not dataframe-style lazy algebra.

### B. Visualization cache

Add a second, rendering-oriented store, for example:

`<dataset>/transcripts_vis/`

Recommended contents:

- `manifest.parquet`
- `genes.parquet` or `genes.json`
- `levels/level_0.parquet`
- `levels/level_1.parquet`
- ...
- `levels/level_n.parquet`

#### `manifest.parquet`

One row per tile, with columns such as:

- `level`
- `tile_id`
- `x_min`, `x_max`, `y_min`, `y_max`
- `n_points`
- `row_group`
- `is_exact`

This lets us determine which row groups to read for a viewport without scanning data files.

#### `level_k.parquet`

Each level file stores points for one resolution level.

Recommended columns:

- `tile_id`
- `x_rel`
- `y_rel`
- `gene_id`
- optional `transcript_id`

Recommended encoding:

- one row group per tile, or one row group per small tile shard
- `x_rel`, `y_rel` stored relative to the tile origin
- start with `float32` for `x_rel`, `y_rel`
- use `uint16`/`uint32` for `gene_id`

Why relative coordinates:

- they compress better;
- they keep values small;
- they leave room for later quantization.

Why `float32` first instead of forcing integers:

- it avoids blocking on the integer-coordinate question raised in `spatialdata#789`;
- it avoids coupling correctness to tile size;
- it is simple and safe for a first implementation.

If benchmarks later show that coordinates are effectively integer-valued, we can add an optional quantized mode using `uint16`.

### C. Multi-level point pyramid

The visualization cache should not store only exact leaf tiles. It should also store coarse sampled levels.

This should follow the idea behind the Neuroglancer spatial index:

- coarse levels store a bounded, spatially uniform sample per tile;
- finer levels store the remaining points;
- the finest level stores the exact data.

This is the crucial difference between "queryable" and "snappy".

Without these coarse sampled levels, zoomed-out views will still try to render far too many points.

### D. Concrete tile and level construction

For implementation, we should make `tile_id` explicit and define levels from a fixed finest tile size.

Assume:

- one chosen 2D render coordinate system
- dataset bounds `[x_min, x_max) x [y_min, y_max)`
- a finest tile size, for example `512`
- levels `0..L`, where `L` is the finest exact level

Define:

```text
tile_size(l) = leaf_tile_size * 2^(L - l)
```

So:

- `level L` has the smallest tiles and stores exact points
- moving toward `level 0` doubles tile width and height each step

For a point `(x, y)` at level `l`:

```text
tx(l) = floor((x - x_min) / tile_size(l))
ty(l) = floor((y - y_min) / tile_size(l))
tile_id(l) = (l, tx(l), ty(l))
```

In storage, `tile_id` can be encoded as a string or integer, but logically it should be treated as the tuple `(level, tx, ty)`.

At the finest level:

- each point belongs to exactly one leaf tile
- `level L` stores the exact data

Parent membership is then automatic:

```text
tx_parent(l) = tx_leaf // 2^(L - l)
ty_parent(l) = ty_leaf // 2^(L - l)
```

This means the pyramid can be built bottom-up.

Recommended construction:

- `level L`: exact leaf tiles, grouped by finest `tile_id`
- `level < L`: sampled parent tiles, built from child tiles

For each parent tile:

1. gather points from its child tiles
2. choose a bounded representative sample
3. store only that sample at the parent level

The first implementation should prefer a spatially stratified sample over naive random sampling, so zoomed-out views remain spatially even and do not clump.

One important caveat is gene filtering:

- if coarse levels are only a global spatial sample, rare genes may disappear at low zoom
- the simplest first version is still to sample spatially first and benchmark gene-filter behavior before adding gene-aware summaries

## Level selection at runtime

The viewer should not choose the level from zoom alone.

Zoom only tells us the viewport size in data coordinates, but transcript density can vary strongly across the tissue. The better rule is:

1. compute the current viewport bounds
2. find intersecting tiles for each candidate level
3. estimate visible points from `manifest.parquet`
4. choose the finest level whose estimated visible point count stays under a render budget

Example:

```text
budget = 150_000 visible points
```

Then:

```text
for l from finest to coarsest:
    visible_tiles = tiles_intersecting_viewport(l)
    estimate = sum(tile_manifest[l, tile].n_points_stored for tile in visible_tiles)
    if estimate <= budget:
        choose level l
        break
```

This gives the desired behavior:

- zoomed in: exact leaf tiles fit under the budget, so use them
- zoomed out: exact tiles exceed the budget, so switch to a coarser sampled level

The runtime controller should also use a small amount of hysteresis so the chosen level does not flicker when the estimated visible count sits near the threshold.

## Recommended loading algorithm in napari

Do not load transcripts through the standard `napari-spatialdata` point path.

Instead, add a Harpy-owned controller, for example `TranscriptTilesController`, that:

1. listens to camera pan / zoom changes;
2. computes the current viewport in the transcript coordinate system;
3. chooses the coarsest level that keeps the visible point count below a render budget;
4. finds intersecting tiles from `manifest.parquet`;
5. loads only those row groups with `pyarrow`;
6. filters by `gene_id` in memory;
7. updates a single napari `Points` layer with the visible sample.

### Render budget

We should treat napari as a renderer for the current view, not as a container for the full dataset.

Suggested starting targets:

- visible points budget: 100k to 250k
- tile size at finest level: 512 px or 1024 px
- warm-cache pan latency target: under 150 ms
- cold tile-load target: under 500 ms

These numbers should be benchmarked, but they are good starting values.

### Runtime behavior

At low zoom:

- show only coarse sampled levels
- never try to show exact global points

At medium zoom:

- switch to finer sampled levels

At high zoom:

- switch to exact leaf tiles

This is the only realistic way to make one-billion-point navigation feel responsive.

### Caching and prefetching

The controller should also maintain:

- an LRU cache of recently decoded tiles
- a small prefetch halo around the viewport
- request cancellation / stale-request dropping when the camera moves again

This is standard image-tile behavior and we should apply the same idea to points.

## Gene filtering

The GitHub discussion correctly points out that filtering by a value column such as gene can become the weak point.

The right first implementation is:

- keep `gene_id` in every tile
- load only visible tiles
- apply the gene filter after tile load

This is acceptable because the expensive step is spatially restricting the data. Once we have only a handful of visible tiles in memory, filtering by `gene_id` is cheap.

Only if benchmarks show that gene filtering is still dominant should we add a secondary per-tile gene index.

Possible phase-2 optimizations:

- store rows within each tile grouped by `gene_id`
- store per-tile counts per gene for UI summaries
- store per-tile offsets for fast gene subsets

I would not build these before measuring.

## Why this is better than the alternatives

### Better than the current plain Parquet table

- adds spatial locality
- adds a cheap viewport query path
- avoids full-table `.compute()`
- avoids random subsampling of the global dataset

### Better than "just Morton-sort one global file"

- still lets us use Morton sorting for the exact data
- but also adds the multi-scale sampled levels needed for whole-slide interaction

### Better than inventing a new NGFF point spec immediately

- we can implement it now
- we do not need to block on upstream standardization
- we can still migrate later to an official tiled point format if SpatialData adopts one

## Concrete implementation phases

### Phase 0: benchmark and settle file layout

Build a prototype converter from a transcript table with `x`, `y`, `gene` to:

- Morton-sorted exact Parquet
- a multi-level tile cache with one Parquet file per level and one row group per tile

Benchmark:

- disk size
- cold read latency
- warm read latency
- pan / zoom responsiveness
- gene switch latency

Use at least one dataset in the 10M to 100M range before extrapolating to 1B.

### Phase 1: preprocessing utilities

Add Harpy utilities to:

- encode `gene -> gene_id`
- compute the global bounds
- sort the exact table by Morton code
- build the tile manifest
- build the sampled levels
- write the cache

This can live entirely in Harpy first.

### Phase 2: napari runtime integration

Add:

- a tile reader backed by `pyarrow`
- a camera-aware controller
- a single `Points` layer used as the render surface
- a minimal UI for gene selection and cache status

At this point the raw `SpatialData` points element should no longer be used for transcript rendering.

### Phase 3: exact selection and metadata plumbing

Add:

- optional transcript picking from the exact leaf tile
- stable mapping back to the canonical transcript row / id
- support for overlays and linked metadata

### Phase 4: upstream alignment

Once the Harpy version is stable:

- upstream Morton-sort utilities to `spatialdata`
- upstream metadata conventions for sorted points
- decide whether the visualization cache should remain Harpy-specific or become a wider `SpatialData` convention

## Explicit design choices

### Use Parquet row groups for the visualization cache

This is the most practical starting point.

It directly follows the discussion in `napari-spatialdata#372`:

- Luca explicitly suggested Parquet row groups for chunks and spatial levels;
- Keller explicitly mentioned one Parquet table per tile as a possible approach.

I prefer one file per level plus row groups over one file per tile because it avoids pathological file counts.

### Do not use `dask.dataframe` in the interactive path

`dask` is helpful when writing and transforming the dataset offline.

It is not the right abstraction for:

- repeated low-latency viewport reads
- row-group-level access
- camera-driven tile fetches

The interactive path should be `pyarrow` + an explicit tile manifest.

### Do not rely on datashader for the base solution

That is a different problem. The issue here is not only rasterization quality, but also storage layout, query cost, and lazy loading.

## Minimal acceptance criteria

We can consider the first implementation successful if it achieves all of the following:

- opening a transcript layer no longer computes the entire points dataframe
- pan and zoom load only visible tiles
- zoomed-out views stay under a fixed point budget
- zoomed-in views can display exact points
- gene filtering does not require scanning the full dataset
- the design works without modifying the canonical exact transcript table schema beyond adding `gene_id` and optional `morton_code`

## References

- [napari-spatialdata issue 372](https://github.com/scverse/napari-spatialdata/issues/372)
- [spatialdata issue 789](https://github.com/scverse/spatialdata/issues/789)
- [spatialdata issue 974](https://github.com/scverse/spatialdata/issues/974)
- [Vitessce data troubleshooting: points](https://vitessce.io/docs/data-troubleshooting/#points)
- [Neuroglancer annotation spatial index](https://github.com/google/neuroglancer/blob/c1fd0b036198f1f5218c9731bdba4061179887de/src/datasource/precomputed/annotations.md#spatial-index)
