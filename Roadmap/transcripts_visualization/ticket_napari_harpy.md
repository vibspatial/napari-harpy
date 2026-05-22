## Implement tiled multiscale point cache for transcript visualization

We currently support the direct in-memory path for points: the user picks a
render budget, selects genes, and we load/render that selection directly in
napari. This remains useful as a fallback when no cache has been built, but it
does not scale well for large transcript datasets.

We want a Neuroglancer-inspired tiled cache for large point selections, while
preserving exact rendering when the finest cached level fits the viewport and
render budget.

Related upstream discussions:

- [spatialdata#974: Easily enable sorting points along a Z-order curve](https://github.com/scverse/spatialdata/issues/974)
- [napari-spatialdata#372: Points subsampling](https://github.com/scverse/napari-spatialdata/issues/372)
- [spatialdata#789: idea: point on-disk representation using OME-NGFF](https://github.com/scverse/spatialdata/issues/789)

## Desired behavior

On `Add / Update in viewer`:

```text
valid cache available
  -> cached/live path
  -> query tile cache using render_budget as the target for level selection
  -> update one live napari Points layer on pan/zoom

no valid cache, or cached rendering disabled
  -> direct fallback path
  -> current behavior
  -> static napari Points layer
```

In cached mode, `render_budget` is used to choose the cache level. If the finest
visible level fits, cached rendering is exact for the viewport. If not, we choose
a coarser sampled level. `render_budget` should not normally be used as a second
query-time trim after loading tiles. We trust the selected cache level.

## Cache design

Use a sharded Parquet tile pyramid:

```text
.sdata_points_cache/
  cache_metadata.json
  genes.parquet
  levels/
    level=0/
      manifest.parquet
      gene_tile_counts.parquet
      shards/
        shard_00000.parquet
    level=1/
      ...
```

Each logical tile is one Parquet row group inside a shard.

Each tile stores:

```text
id
x
y
gene_code
sample_rank
```

Rows inside tiles are sorted by:

```text
gene_code, sample_rank
```

This is important for snappy gene switching. `sample_rank =
stable_hash(point_id)` gives deterministic ordering and stable sampling behavior
across levels.

## Level construction

Build each level as a self-contained representation of the full normalized
points source. Do not build residual Neuroglancer-style levels for now.

For each level:

```python
points = scan_parquet(normalized_path)

points = points.with_columns([
    tile_x_expr(level),
    tile_y_expr(level),
    tile_id_expr(level),
    sample_rank_expr(point_id),
])

ranked = rank_within_tile(points, by="sample_rank")

if level.is_finest:
    emitted = ranked
else:
    emitted = ranked.filter(rank_in_tile <= level.max_points_per_tile)

write_level_tiles_and_manifest(emitted, level=level)
```

This duplicates sampled points in coarser levels, but expected overhead is
modest compared with the full finest level, and the query path becomes much
simpler.

## Query path

For each cached viewport update:

```text
bbox + selected genes + render_budget
  -> estimate visible point count per level using manifest/gene_tile_counts
  -> choose finest level with estimated_count <= render_budget
  -> if none fit, use coarsest level as-is
  -> read visible tile row groups
  -> exact bbox/gene filter
  -> update live Points layer
```

No normal query-time resampling/trimming.

## Napari layer strategy

Each `Add / Update in viewer` creates a new Harpy-managed Points layer and
retires the previous one for that source.

In cached/live mode, pan/zoom updates mutate the same live layer in-place. We
need a guarded update helper because naive `layer.data` / `layer.features`
mutation can leave napari async slicing caches stale.

## Notes

This is related to the same problem space as SpatialData Z-order sorting and
on-demand point loading, but the cache here is app-level and optimized for
napari-harpy transcript visualization.

Compared with Neuroglancer's residual spatial index, this design stores sampled
coarser levels as self-contained representations. That costs some extra storage,
but keeps query behavior simple: choose one level, load visible tiles, filter,
render.
