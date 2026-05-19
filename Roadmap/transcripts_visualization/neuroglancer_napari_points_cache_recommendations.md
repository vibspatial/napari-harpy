# Neuroglancer-Inspired Point Visualization Cache for napari + SpatialData

## Goal

We start from a `points` element in a `SpatialData` object, for example transcript-like points with:

```text
x
y
gene
```

The goal is to visualize these points efficiently in **napari**, without loading all points into a `Points` layer at once.

The recommended design is inspired by Neuroglancer's annotation rendering model:

- Use a **regular spatial tile pyramid**.
- Store **bounded / sampled point subsets** at coarse levels.
- Store complete or nearly complete point data at fine levels.
- Use a **manifest** to map viewport queries to tile data.
- Use a **gene/category index** to avoid loading irrelevant tiles.
- Feed napari only a **small visible subset** of points.

---

## Core Principle

Do **not** pass the full SpatialData points table directly to napari.

Instead:

```text
SpatialData points element
        ↓
persistent point cache
        ↓
viewport + gene query
        ↓
small NumPy/Pandas visible subset
        ↓
ordinary napari Points layer
```

The napari `Points` layer should only contain the currently visible subset.

---

## Recommended On-Disk Layout

A practical cache layout:

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
        shard_00001.parquet
        ...

    level=1/
      manifest.parquet
      gene_tile_counts.parquet
      shards/
        shard_00000.parquet
        shard_00001.parquet
        ...

    level=2/
      manifest.parquet
      gene_tile_counts.parquet
      shards/
        shard_00000.parquet
        shard_00001.parquet
        ...
```

Where:

```text
level 0 = coarsest / most downsampled
level N = finest / most complete
```

A simpler first implementation can use one file per tile:

```text
.sdata_points_cache/
  cache_metadata.json
  genes.parquet

  level_0_manifest.parquet
  level_0_gene_tile_counts.parquet
  level_0_tiles/
    tile_0_0.parquet
    tile_0_1.parquet
    ...

  level_1_manifest.parquet
  level_1_gene_tile_counts.parquet
  level_1_tiles/
    tile_0_0.parquet
    tile_0_1.parquet
    ...
```

For production-scale datasets, prefer sharded Parquet files with one row group per tile.

---

## `cache_metadata.json`

This file describes how the cache was built and whether it can be reused.

Example:

```json
{
  "cache_version": 1,
  "points_element": "transcripts",
  "coordinate_system": "global",
  "x_column": "x",
  "y_column": "y",
  "category_column": "gene",
  "coordinate_order_for_napari": "yx",
  "bounds": {
    "xmin": 0.0,
    "xmax": 50000.0,
    "ymin": 0.0,
    "ymax": 30000.0
  },
  "levels": [
    {
      "level": 0,
      "tile_size_x": 4096.0,
      "tile_size_y": 4096.0,
      "max_points_per_tile": 2048,
      "sampled": true
    },
    {
      "level": 1,
      "tile_size_x": 2048.0,
      "tile_size_y": 2048.0,
      "max_points_per_tile": 4096,
      "sampled": true
    },
    {
      "level": 2,
      "tile_size_x": 1024.0,
      "tile_size_y": 1024.0,
      "max_points_per_tile": 8192,
      "sampled": true
    },
    {
      "level": 3,
      "tile_size_x": 512.0,
      "tile_size_y": 512.0,
      "max_points_per_tile": null,
      "sampled": false
    }
  ],
  "gene_dictionary_path": "genes.parquet"
}
```

Invalidate the cache if any of these change:

```text
SpatialData object/path
points element name
x/y columns
gene/category column
coordinate system
coordinate transform
tile sizes
sampling parameters
cache version
```

---

## `genes.parquet`

Store a dictionary mapping gene strings to integer codes.

Schema:

```text
gene_code: int32
gene: string
```

Example:

| gene_code | gene |
|---:|---|
| 0 | Actb |
| 1 | Malat1 |
| 2 | Sox2 |

Tile data should store `gene_code`, not the gene string.

---

## `manifest.parquet`

This is the main spatial index.

One row per tile, per level.

Schema:

```text
level: int16
tile_x: int32
tile_y: int32
tile_id: int64

xmin: float64
xmax: float64
ymin: float64
ymax: float64

n_points_total: int64
n_points_stored: int64

sampled: bool

shard_path: string
row_group: int32
```

Example:

| level | tile_x | tile_y | tile_id | xmin | xmax | ymin | ymax | n_points_total | n_points_stored | shard_path | row_group |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| 2 | 10 | 5 | 42949672965 | 5120 | 5632 | 2560 | 3072 | 18320 | 18320 | shards/shard_00007.parquet | 12 |
| 1 | 5 | 2 | 21474836482 | 5120 | 6144 | 2048 | 3072 | 42500 | 4096 | shards/shard_00002.parquet | 44 |
| 0 | 2 | 1 | 8589934593 | 4096 | 8192 | 2048 | 6144 | 240000 | 2048 | shards/shard_00000.parquet | 7 |

The manifest tells the query engine:

```text
which tiles intersect the viewport
how many points each tile contains
where the tile's physical data lives
```

This is the spatial index.

Because the tiles form a regular grid, you usually do not need an R-tree.

---

## Viewport to Tile Lookup

Given a viewport:

```text
xmin, xmax, ymin, ymax
```

and a chosen level with tile size:

```text
tile_size_x
tile_size_y
```

Compute tile ranges directly:

```python
tile_x_min = floor((xmin - origin_x) / tile_size_x)
tile_x_max = floor((xmax - origin_x) / tile_size_x)

tile_y_min = floor((ymin - origin_y) / tile_size_y)
tile_y_max = floor((ymax - origin_y) / tile_size_y)
```

Then filter the manifest:

```python
tiles = manifest[
    (manifest.level == level) &
    (manifest.tile_x >= tile_x_min) &
    (manifest.tile_x <= tile_x_max) &
    (manifest.tile_y >= tile_y_min) &
    (manifest.tile_y <= tile_y_max)
]
```

This gives the list of tile chunks to load.

---

## `gene_tile_counts.parquet`

This is the gene/category index.

One row per `(tile, gene)` pair that has at least one point.

Schema:

```text
level: int16
tile_x: int32
tile_y: int32
tile_id: int64
gene_code: int32

n_points_total: int64
n_points_stored: int64

start_in_tile: int32
count_in_tile: int32
```

Example:

| level | tile_id | gene_code | n_points_total | n_points_stored | start_in_tile | count_in_tile |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 42949672965 | 0 | 210 | 210 | 0 | 210 |
| 2 | 42949672965 | 1 | 850 | 850 | 210 | 850 |
| 2 | 42949672965 | 2 | 18 | 18 | 1060 | 18 |

This file is useful for:

```text
checking whether a visible tile contains selected genes
estimating visible point counts
choosing the appropriate level of detail
skipping irrelevant tiles
finding per-gene ranges inside a tile
```

This assumes rows inside each tile are sorted by `gene_code`.

Important: `start_in_tile` and `count_in_tile` are logical row offsets inside the
tile, not physical Parquet byte offsets and not global shard offsets.

The recommended storage granularity is still:

```text
one Parquet row group per tile
```

The recommended gene index granularity is:

```text
one row in gene_tile_counts.parquet per tile/gene pair
```

This avoids creating one row group per `(tile, gene)`, which would explode the
number of row groups for datasets with many genes. The gene index is used to
skip visible tiles that do not contain selected genes, estimate visible point
counts, and slice contiguous gene ranges after a tile row group has been loaded
into memory. In the normal Parquet layout, `start_in_tile` / `count_in_tile`
should not be treated as a promise that only that small range can be read from
disk.

---

## Tile / Shard Data

Each physical tile row group should contain:

```text
id: uint64
x: float32 or float64
y: float32 or float64
gene_code: int32
sample_rank: uint64
```

Recommended row order inside each tile:

```text
gene_code, sample_rank
```

or:

```text
gene_code, id
```

For visualization, prefer:

```text
gene_code, sample_rank
```

This allows deterministic downsampling.

---

## Sorting Strategy

At build time, derive:

```python
tile_x = floor((x - origin_x) / tile_size_x)
tile_y = floor((y - origin_y) / tile_size_y)
tile_id = make_tile_id(tile_x, tile_y)
gene_code = categorical_code(gene)
sample_rank = stable_hash(id, level)
```

Then sort by:

```text
level, tile_id, gene_code, sample_rank
```

Within one level, the effective order is:

```text
tile_id, gene_code, sample_rank
```

Within one tile, the important order is:

```text
gene_code, sample_rank
```

This gives:

```text
fast tile grouping
fast gene filtering
stable point order
deterministic downsampling
```

---

## Tile ID

A simple tile ID can be constructed from `tile_x` and `tile_y`:

```python
tile_id = (tile_x << 32) | tile_y
```

This assumes non-negative tile coordinates.

If tile coordinates may be negative, either offset them first or use a signed-safe encoding.

---

## Building the Cache

### Step 1: Normalize Input

Start from the SpatialData points element:

```text
x, y, gene
```

Create normalized columns:

```text
id
x
y
gene_code
```

Apply the correct SpatialData coordinate transform before caching.

The cache should be built in the coordinate system used by napari.

---

### Step 2: Build Gene Dictionary

Convert gene strings to integer codes:

```python
gene_code = categorical_encode(gene)
```

Write:

```text
genes.parquet
```

---

### Step 3: Build Each Level

For each level:

```python
tile_x = floor((x - origin_x) / tile_size_x)
tile_y = floor((y - origin_y) / tile_size_y)
tile_id = make_tile_id(tile_x, tile_y)
```

Group by tile:

```python
groupby(tile_id)
```

For each tile:

```python
if n_points <= max_points_per_tile:
    keep all points
else:
    keep deterministic sample of max_points_per_tile
```

Use a stable hash for sampling:

```python
sample_rank = stable_hash(id, level)
```

Keep points with the lowest sample ranks:

```python
tile_points = tile_points.nsmallest(max_points_per_tile, "sample_rank")
```

Then sort:

```python
tile_points = tile_points.sort_values(["gene_code", "sample_rank"])
```

---

### Step 4: Write Tile Data

For a first implementation:

```text
one Parquet file per tile
```

For a production implementation:

```text
one Parquet row group per tile
multiple tiles per shard file
```

The manifest row should point to the physical location:

```text
shard_path
row_group
```

or, in a simpler implementation:

```text
tile_path
```

---

### Step 5: Write `manifest.parquet`

For each tile, write:

```text
level
tile_x
tile_y
tile_id
xmin
xmax
ymin
ymax
n_points_total
n_points_stored
sampled
shard_path
row_group
```

This file is used for spatial lookup and level-of-detail planning.

---

### Step 6: Write `gene_tile_counts.parquet`

For each tile and each gene present in that tile, write:

```text
level
tile_x
tile_y
tile_id
gene_code
n_points_total
n_points_stored
start_in_tile
count_in_tile
```

Because tile rows are sorted by `gene_code`, each gene occupies a contiguous range.

---

## Choosing a Level at Query Time

Given:

```text
viewport bbox
selected genes
max_visible_points
```

Estimate the number of points at each level.

Without a gene filter:

```python
estimated_count = manifest[
    (manifest.level == level) &
    (manifest.tile_x >= tile_x_min) &
    (manifest.tile_x <= tile_x_max) &
    (manifest.tile_y >= tile_y_min) &
    (manifest.tile_y <= tile_y_max)
]["n_points_stored"].sum()
```

With a gene filter:

```python
estimated_count = gene_tile_counts[
    (gene_tile_counts.level == level) &
    (gene_tile_counts.tile_x >= tile_x_min) &
    (gene_tile_counts.tile_x <= tile_x_max) &
    (gene_tile_counts.tile_y >= tile_y_min) &
    (gene_tile_counts.tile_y <= tile_y_max) &
    (gene_tile_counts.gene_code.isin(selected_gene_codes))
]["n_points_stored"].sum()
```

Pick the finest tile-pyramid level where:

```text
estimated_count <= max_visible_points
```

If all levels exceed the limit, pick the coarsest level and apply a final deterministic cap.

---

## Query Path

Given a napari viewport and selected genes:

```text
xmin, xmax, ymin, ymax
selected_gene_codes
max_visible_points
```

### 1. Choose query path and level

Use the manifest and gene index to estimate visible point counts.

Choose a tile-pyramid level based on the visible-count estimate and render
budget.

### 2. Tile-pyramid path

Find intersecting tiles.

Use tile arithmetic and `manifest.parquet`.

### 3. Optionally skip irrelevant tiles

If genes are selected, semi-join visible tiles with `gene_tile_counts.parquet`.

### 4. Read tile row groups

For each selected tile:

```text
read shard_path + row_group
```

or:

```text
read tile_path
```

### 5. Apply exact filtering

Filter by viewport:

```python
tile = tile[
    (tile.x >= xmin) &
    (tile.x <= xmax) &
    (tile.y >= ymin) &
    (tile.y <= ymax)
]
```

Filter by gene:

```python
tile = tile[tile.gene_code.isin(selected_gene_codes)]
```

### 6. Cap if needed

If the result is still too large:

```python
result = result.nsmallest(max_visible_points, "sample_rank")
```

### 7. Return napari-ready data

napari usually expects coordinates as:

```text
y, x
```

So return:

```python
coords = result[["y", "x"]].to_numpy()
features = {"gene": gene_names}
```

---

## napari Integration

Create one `Points` layer:

```python
layer = viewer.add_points(
    np.empty((0, 2), dtype=np.float32),
    features={"gene": np.array([], dtype=object)},
    size=2,
    face_color="gene",
)
```

On viewport or gene-filter change:

```python
visible = tile_source.query(
    bbox=current_view_bbox(viewer),
    selected_genes=selected_genes,
    max_visible_points=100_000,
)

update_live_points_layer(
    layer,
    coords_yx=visible.coords_yx,
    features={"gene": visible.gene_names},
)
```

### Safe In-Place Points Layer Updates

The live tiled mode should keep one persistent napari `Points` layer and update
its payload in place. Replacing the layer on every pan or zoom would reset
viewer state and is unlikely to feel snappy.

However, plain in-place updates are not always safe:

```python
layer.data = coords_yx
layer.features = features
layer.refresh()
```

In napari 0.7.0, this can leave stale private view indices during the update
when experimental async slicing is enabled. If the new data array is shorter
than the old one, hover/status/render callbacks can temporarily read old
`_indices_view` values against the new shorter `data` array and raise an
`IndexError`.

Use a guarded update sequence:

```python
from contextlib import ExitStack


def update_live_points_layer(layer, *, coords_yx, features) -> None:
    with ExitStack() as stack:
        stack.enter_context(layer.events.blocker_all())
        stack.enter_context(layer._face.events.blocker_all())
        stack.enter_context(layer._border.events.blocker_all())
        stack.enter_context(layer.text.events.blocker_all())

        layer.selected_data = set()
        layer._value = None
        layer.data = coords_yx
        layer.features = features

    layer.set_view_slice()
    layer._refresh_sync(
        thumbnail=False,
        data_displayed=True,
        highlight=False,
        extent=False,
        force=True,
    )
```

Important details:

```text
block layer, face-color, border-color, and text events while the layer payload
is internally inconsistent

assign data before features, because napari validates feature length against
the current data length

clear transient selection/hover state before the payload swap

force a synchronous set_view_slice() before emitting the single visual update

do not reset face_color on every viewport update; configure the color mode once
when creating the live layer, then update data/features only
```

This uses napari private APIs (`_face`, `_border`, `_value`, `_refresh_sync`),
so it should be isolated in one Harpy helper and covered by regression tests.
The minimum regression test should enable napari async slicing, update a points
layer from many rows to fewer rows, and assert that `_view_data`, `_view_size`,
and `get_status(...)` do not raise after the helper returns.

Layer-model timing measured locally was roughly:

```text
10k points:   ~5 ms
50k points:   ~23 ms
100k points:  ~44 ms
200k points:  ~88 ms
```

These numbers exclude real GPU upload cost, but they suggest that a 50-150 ms
camera debounce and a ~100k visible point budget are plausible starting points.

Use debouncing:

```text
camera event
  -> debounce 50-150 ms
  -> submit async query
  -> ignore stale query results
  -> update layer only with the newest result
```

---

## Cache Layers

Use three cache layers.

### 1. Persistent Disk Cache

The files described above.

### 2. In-Memory Tile LRU Cache

Key:

```python
(level, tile_x, tile_y)
```

Value:

```python
{
    "coords_yx": np.ndarray,
    "gene_codes": np.ndarray,
    "ids": np.ndarray,
    "sample_rank": np.ndarray,
    "gene_offsets": dict[int, tuple[int, int]],
}
```

Evict based on byte size.

### 3. Query Result Cache

Key:

```python
(level, rounded_bbox, selected_gene_codes, max_visible_points)
```

Value:

```text
coords_yx
gene_names
ids
```

This is useful because napari can emit multiple camera events for nearly identical views.

## Gene Filtering Strategy

### Few selected genes

Use `gene_tile_counts.parquet` to estimate the visible count for those genes
and to find visible tiles containing those genes.

Then use per-tile `start_in_tile` / `count_in_tile` ranges.

### Many selected genes

Load visible spatial tiles and filter after reading.

### No gene filter

Ignore the gene index.

### Very sparse genes

Use the gene index aggressively to avoid reading empty tiles.

---

## Full Level vs Sampled Levels

Recommended:

```text
coarse levels: sampled
finest level: full
```

Example:

```text
level 0: tile size 4096, max 2,000 points/tile
level 1: tile size 2048, max 4,000 points/tile
level 2: tile size 1024, max 8,000 points/tile
level 3: tile size 512, full points
```

At low zoom, show representative points.

At high zoom, show complete local detail.

---

## Why a Manifest Instead of an R-tree?

Because the tiles form a regular grid.

Viewport lookup is simple arithmetic:

```python
tile_x_min = floor((xmin - origin_x) / tile_size_x)
tile_x_max = floor((xmax - origin_x) / tile_size_x)
tile_y_min = floor((ymin - origin_y) / tile_size_y)
tile_y_max = floor((ymax - origin_y) / tile_size_y)
```

Then the manifest tells you:

```text
which of those tiles exist
how many points they contain
where their data is stored
```

An R-tree is optional. It is not necessary for the main case.

---

## First Implementation Plan

Build the simplest useful version first:

1. Encode genes into `gene_code`.
2. Build one finest-level tiled cache.
3. Write one Parquet file per tile.
4. Write `manifest.parquet`.
5. Implement viewport query.
6. Add the guarded live `Points` layer update helper and async-slicing regression test.
7. Add `gene_tile_counts.parquet`.
8. Add coarser sampled levels.
9. Replace tile files with sharded Parquet row groups.

---

## Production Implementation Plan

For larger data:

1. Use multiple levels.
2. Store tile data in shard files.
3. Use one Parquet row group per tile.
4. Keep `manifest.parquet` and `gene_tile_counts.parquet` memory-mapped or loaded in memory.
5. Use async loading.
6. Use an LRU cache for recently viewed tiles.
7. Debounce napari camera events.
8. Ignore stale query results.

---

## Minimal Python Interfaces

```python
from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass(frozen=True)
class ViewQuery:
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    genes: tuple[str, ...] | None
    max_points: int
    coordinate_system: str


@dataclass
class PointTile:
    level: int
    tile_x: int
    tile_y: int
    coords_yx: np.ndarray
    gene_codes: np.ndarray
    ids: np.ndarray
    sample_rank: np.ndarray
    gene_offsets: dict[int, tuple[int, int]]


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
    ):
        ...

    def build_cache(self, overwrite: bool = False) -> None:
        ...

    def query(self, query: ViewQuery):
        ...
```

---

## Summary

Use:

```text
manifest.parquet
```

as the spatial index.

Use:

```text
gene_tile_counts.parquet
```

as the categorical/gene index.

Store point data sorted by:

```text
tile_id, gene_code, sample_rank
```

or inside each tile:

```text
gene_code, sample_rank
```

Use a tile pyramid:

```text
coarse levels = sampled
fine levels = complete
```

At runtime:

```text
viewport
  -> tile range -> manifest lookup -> optional gene index lookup -> load tile chunks
  -> exact bbox/gene filter
  -> update napari Points layer
```

This gives you a Neuroglancer-like point visualization system while still using an ordinary napari `Points` layer for rendering.

---

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
