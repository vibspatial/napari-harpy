## Optional Exact Gene Cache

This is a deferred extension, not part of the current tile-pyramid-first
implementation plan.

The tile pyramid is tile-major. It is optimized for normal spatial browsing,
especially when no gene is selected or many genes are visible.

However, selected-gene rendering has a useful special case:

```text
if the user selects one or a few genes
and the exact visible count is <= points_render_budget
then show those genes exactly, even at zoomed-out levels
```

With only the tile-major cache, this can require reading many tile row groups
that mostly contain other genes, then filtering to the selected genes in memory.

### Optional Layout Extension

To support exact selected-gene rendering efficiently, add an optional gene-major
cache alongside the tile pyramid:

```text
exact_genes/
  manifest.parquet
  shards/
    shard_00000.parquet
    shard_00001.parquet
    ...
```

This cache stores full-resolution points organized primarily by `gene_code`.
It is not a replacement for the tile pyramid. It is a complementary read path
for cases where exact selected-gene rendering fits the point render budget.

If this extension is enabled, the top-level cache layout becomes:

```text
.sdata_points_cache/
  cache_metadata.json
  genes.parquet
  levels/
    ...
  exact_genes/
    manifest.parquet
    shards/
      shard_00000.parquet
      shard_00001.parquet
      ...
```

The `cache_metadata.json` should record whether the exact gene cache was built
and should invalidate it if the source data, category column, coordinate system,
coordinate transform, sampling/id strategy, or cache version changes.

### Manifest Schema

Recommended manifest schema:

```text
gene_code: int32
chunk_id: int32

xmin: float64
xmax: float64
ymin: float64
ymax: float64

n_points: int64

shard_path: string
row_group: int32
```

Each row group contains full-resolution point rows:

```text
id: uint64
x: float32 or float64
y: float32 or float64
gene_code: int32
tile_x: int32
tile_y: int32
tile_id: int64
sample_rank: uint64
```

### Build Extension

For each gene, write full-resolution points to `exact_genes/`.

For a simple implementation, use one row group per gene.

For large or abundant genes, split each gene into chunks, for example targeting
64k-256k points per row group. Rows within each gene can be sorted by:

```text
tile_id, sample_rank
```

or by a spatial key such as Morton/Hilbert order if spatial pruning inside very
large genes becomes important.

Avoid using one row group per `(tile_id, gene_code)` as the default. That layout
can create a very large number of tiny row groups for transcript datasets with
many genes and many tiles.

A future implementation plan for this extension would be:

1. Add an explicit build option, for example `enable_exact_gene_cache=False`.
2. Write `exact_genes/manifest.parquet`.
3. Write exact gene shards with one row group per gene or gene chunk.
4. Add query-time exact-count estimation.
5. Add the exact gene-major query path.
6. Add an optional in-memory gene chunk LRU cache.

### Query-Time Use

This extension only applies when the user selects one or a few genes. Normal
spatial browsing should keep using the tile pyramid.

First estimate a conservative full-resolution visible count using the finest
full tile-pyramid level and `n_points_total`:

```python
estimated_exact_count = gene_tile_counts[
    (gene_tile_counts.level == finest_level) &
    (gene_tile_counts.tile_x >= tile_x_min) &
    (gene_tile_counts.tile_x <= tile_x_max) &
    (gene_tile_counts.tile_y >= tile_y_min) &
    (gene_tile_counts.tile_y <= tile_y_max) &
    (gene_tile_counts.gene_code.isin(selected_gene_codes))
]["n_points_total"].sum()
```

This is an upper bound because it counts whole tiles that intersect the
viewport. If the exact gene cache is chunked, `exact_genes/manifest.parquet`
can provide a second, usually tighter, upper bound by summing `n_points` for
chunks whose bounding boxes intersect the viewport:

```python
estimated_exact_count = exact_gene_manifest[
    (exact_gene_manifest.gene_code.isin(selected_gene_codes)) &
    (exact_gene_manifest.xmax >= xmin) &
    (exact_gene_manifest.xmin <= xmax) &
    (exact_gene_manifest.ymax >= ymin) &
    (exact_gene_manifest.ymin <= ymax)
]["n_points"].sum()
```

If few genes are selected, `exact_genes/` exists, and:

```text
estimated_exact_count <= points_render_budget
```

prefer the exact gene-major read path instead of a sampled zoomed-out
tile-pyramid level. Otherwise, fall back to the normal tile-pyramid path.

### Exact Gene-Major Query Path

Use `exact_genes/manifest.parquet` to find row groups for the selected genes.

If the exact gene cache uses chunks, prune chunks by their bounding boxes:

```python
chunks = exact_gene_manifest[
    (exact_gene_manifest.gene_code.isin(selected_gene_codes)) &
    (exact_gene_manifest.xmax >= xmin) &
    (exact_gene_manifest.xmin <= xmax) &
    (exact_gene_manifest.ymax >= ymin) &
    (exact_gene_manifest.ymin <= ymax)
]
```

Read those gene row groups, then apply the exact viewport filter:

```python
points = points[
    (points.x >= xmin) &
    (points.x <= xmax) &
    (points.y >= ymin) &
    (points.y <= ymax)
]
```

Return the same napari-ready shape as the tile-pyramid path:

```python
coords = points[["y", "x"]].to_numpy()
features = {"gene": gene_names}
```

The runtime branch becomes:

```text
viewport
  -> estimate selected-gene exact count
  -> if exact gene path fits budget: load exact_genes chunks
  -> otherwise: tile range -> manifest lookup -> optional gene index lookup -> load tile chunks
  -> exact bbox/gene filter
  -> update napari Points layer
```

### Optional Gene Chunk LRU Cache

If users repeatedly toggle or pan around selected genes, add an in-memory LRU
cache for exact gene chunks.

Key:

```python
(gene_code, chunk_id)
```

Value:

```text
full-resolution points for that gene chunk
```

This is separate from the normal tile LRU cache, whose key remains:

```python
(level, tile_x, tile_y)
```

### Interface Extension

If this extension is implemented, expose it as an explicit opt-in option rather
than making it part of the default tile cache:

```python
class SpatialDataPointTileSource:
    def __init__(
        self,
        sdata,
        points_key: str,
        gene_key: str = "gene",
        coordinate_system: str | None = None,
        cache_dir: str | Path | None = None,
        max_points_per_tile: int = 4096,
        max_visible_points: int = 100_000,
        enable_exact_gene_cache: bool = False,
    ):
        ...
```

The UI should only offer exact selected-gene rendering when the cache metadata
confirms that `exact_genes/` exists and matches the active cache identity.
