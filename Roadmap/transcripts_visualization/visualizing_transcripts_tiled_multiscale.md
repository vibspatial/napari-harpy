# Visualizing Transcripts With Tiled Multiscale Storage

This note captures the design direction we discussed for transcript visualization in napari:

- explicit `tile_id`-based spatial partitioning
- multiple sampled levels of detail
- optional Morton ordering inside tiles

This is a viewer-oriented design. It is preferred over a single globally Morton-sorted table when the main goal is snappy pan/zoom in napari for very large transcript datasets.

## Main idea

For interactive visualization, the important pieces are:

1. `tile_id`: make space explicit
2. multiscale sampled levels: make rendering bounded

Global Morton sort alone improves locality, but it still leaves us with one large flat table. A tiled multiscale design gives us an actual spatial access model and an actual rendering model.

## Why this is preferred over Morton sort alone

### Morton sort alone

Morton sort helps because nearby points tend to be stored near each other in one file. That makes viewport queries much better than scanning a random global Parquet table.

But Morton sort alone does not give us:

- an explicit mapping from viewport to tiles
- a natural way to load by zoom level
- a bounded point budget for zoomed-out views

It solves part of the data access problem, but not the full viewer problem.

### Tiled multiscale design

With explicit tiling:

- viewport -> intersecting `tile_id`s is trivial
- storage is spatially organized by construction
- loading logic matches how image tiling already works

With multiple levels:

- zoomed-out views load coarse representative samples
- zoomed-in views load finer samples
- sufficiently zoomed-in views load exact leaf tiles

This is the important step from "queryable" to "snappy".

## Recommended structure

We should treat transcript visualization storage as a dedicated cache beside the canonical transcript table.

The canonical table can remain:

- one exact transcript table
- columns such as `x`, `y`, `gene_id`
- optionally `transcript_id`, `cell_id`, quality metrics

The visualization cache should be separate and optimized for viewport-driven reads.

## Proposed on-disk layout

Example:

```text
<dataset>/
  transcripts.parquet                  # canonical exact table
  transcripts_vis/
    manifest.parquet
    genes.parquet
    levels/
      level_0.parquet
      level_1.parquet
      ...
      level_n.parquet
```

Recommended meaning:

- `level_0`: coarsest sampled overview
- `level_n`: finest exact level
- one Parquet file per level
- one row group per tile, or per tile shard if a tile is too dense

This avoids creating a huge number of tiny files while still letting us read tiles independently.

## Tile model

Each point gets:

- `tile_id`
- `level`
- coordinates relative to tile origin: `x_rel`, `y_rel`
- `gene_id`
- optional `transcript_id`

The tile itself has metadata:

- `x_min`, `x_max`
- `y_min`, `y_max`
- `n_points`
- `row_group`
- `is_exact`

That metadata lives in `manifest.parquet`.

## Why `tile_id` is essential

`tile_id` is the core access primitive.

It lets the viewer do:

1. compute the viewport bounds
2. find intersecting tiles
3. read only those tiles

This is much simpler than:

1. compute Morton intervals
2. map intervals to row groups
3. load candidate ranges
4. refine in memory

Morton sort is still useful as an optimization, but `tile_id` is what makes the design easy to reason about and easy to implement correctly in a napari controller.

## Why multiscale levels are essential

Exact tiling alone is not enough.

If we only store exact leaf tiles, then zoomed-out views still intersect many tiles and may still contain far too many points to render smoothly.

Multiscale sampled levels solve this:

- coarse levels store bounded representative samples
- fine levels store denser samples
- the finest level stores exact data

At runtime, the viewer chooses the coarsest level that keeps the visible point count under a target render budget.

This means:

- low zoom: use coarse overview points
- medium zoom: use finer sampled points
- high zoom: use exact points

Without this level selection, explicit tiling still does not fully solve the rendering problem.

## Morton inside tile

Morton ordering inside a tile is optional.

Possible benefits:

- slightly better compression
- stable spatial ordering
- possible future sub-tile querying

But once points are already grouped by `tile_id`, most of the important spatial benefit has already been achieved.

So the priorities should be:

1. `tile_id`
2. multiscale levels
3. optional Morton ordering inside tile

Morton should be treated as a nice optimization, not the central design.

## Suggested runtime behavior in napari

We should not rely on the standard `napari-spatialdata` points loading path for transcript rendering.

Instead, add a Harpy-owned controller that:

1. listens to camera pan / zoom changes
2. computes the viewport in transcript coordinates
3. selects an appropriate `level`
4. resolves intersecting `tile_id`s
5. reads only the relevant row groups with `pyarrow`
6. filters by `gene_id` in memory
7. updates one napari `Points` layer with only the visible points

Additional runtime behavior:

- LRU cache of recently used tiles
- small prefetch halo around the viewport
- stale-request dropping if the camera moves again

## Gene filtering

Spatial tiling solves the spatial access problem first.

For a first implementation, the simplest plan is:

- store `gene_id` in every tile
- load only visible tiles
- apply gene filtering after tile load

This is acceptable because by the time filtering happens, the data has already been reduced to the visible tiles.

If this is still too slow, phase-2 optimizations could include:

- rows grouped by `gene_id` within each tile
- per-tile gene counts
- per-tile offsets for gene subsets

These should be driven by benchmarks, not added up front.

## Handling dense regions

Fixed grid tiling can lead to strongly uneven densities.

Some regions may be sparse, while transcript-dense regions may create very heavy tiles.

This can be handled by:

- choosing a small enough tile size
- allowing a tile to be split into multiple row-group shards
- optionally adding subtiles later for pathological dense regions

So the design does not need to assume uniform density.

## Tradeoff summary

### Prefer global Morton sort when

- we want the smallest change to the current Parquet-based exact points representation
- we want an upstream-friendly improvement to `SpatialData`
- we are optimizing one global exact table

### Prefer tiled multiscale storage when

- we want the best napari viewing experience
- we want direct viewport -> tile lookup
- we need strict render budgets
- we expect very large datasets, including hundreds of millions to one billion points

## Recommendation

For transcript visualization in Harpy, the preferred design should be:

1. keep the canonical exact transcript table separate
2. build a tiled multiscale visualization cache
3. use `tile_id` as the primary access primitive
4. use multiscale sampled levels as the primary rendering control
5. treat Morton ordering as optional inside-tile optimization

In short:

- for visualization, prefer `tile_id + multiscale`
- for exact table locality, Morton is still useful
- do not make global Morton sort the main design if the goal is responsive napari interaction

## Minimal acceptance criteria

The approach is successful if it gives us:

- no full-table `.compute()` during viewer interaction
- viewport-driven tile reads only
- bounded visible point counts at every zoom level
- exact points when sufficiently zoomed in
- acceptable gene filtering latency on visible tiles
- stable interaction on datasets much larger than current napari point limits
