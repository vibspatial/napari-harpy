# Neuroglancer-Inspired Point Visualization Cache for napari + SpatialData

## Goal

We start from a `points` element in a `SpatialData` object, for example transcript-like points with:

```text
x
y
gene
```

The goal is to visualize these points efficiently in **napari**, without
loading an over-budget point selection into a `Points` layer at once.

The recommended design is inspired by Neuroglancer's annotation rendering model:

- Use a **regular spatial tile pyramid**.
- Store **bounded / sampled point subsets** at coarse levels.
- Store complete or nearly complete point data at fine levels.
- Use a **manifest** to map viewport queries to tile data.
- Use a **gene/category index** to avoid loading irrelevant tiles.
- Feed napari only a **small visible subset** of points.

---

## Core Principle

Do **not** pass an over-budget SpatialData points selection directly to napari.
Exact direct rendering is still the preferred path when the selected points fit
within the user's render budget.

For over-budget selections, use:

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

## Hybrid Rendering Policy

Users keep the existing render-budget workflow:

```text
user chooses render budget
user selects genes, a few genes, or all/no genes
user clicks Add / Update in viewer
```

On each `Add / Update in viewer` click, compute the total selected source-point
count before deciding how to render:

```text
selected_total_count <= render_budget
  -> exact/direct mode
  -> reuse the current direct implementation
  -> create a fresh static napari Points layer with all selected points
  -> no viewport-driven updates

selected_total_count > render_budget
  -> tiled/live mode
  -> query the persistent tile cache with max_visible_points = render_budget
  -> create a fresh live napari Points layer for the render request
  -> update that layer on debounced viewport changes
```

For no gene filter / all genes, `selected_total_count` is the total number of
points in the points element. This means small datasets still render exactly,
while large datasets automatically use the tiled path.

Mode selection happens only when the user changes the selection or render
budget and clicks `Add / Update in viewer`. Camera movement must not switch
between exact/direct and tiled/live mode. Camera movement only updates the
current tiled/live layer.

Once a selection enters tiled/live mode because its global count exceeds the
render budget, zoomed-in views can still become exact when the finest visible
tiles fit within the viewport budget. In other words, multiscale rendering does
not mean "always sampled"; it means "sampled when needed, exact when locally
affordable."

Layer lifecycle should stay simple:

```text
each Add / Update click creates a new napari Points layer object
the previous Harpy-managed points render layer for the same source is retired
any previous live tiled query controller/listener is stopped
the new request is either exact/static or tiled/live
```

This matches the current direct points adapter behavior, which replaces the old
Harpy-managed points layer with a fresh layer object for the same source
identity. The tiled/live implementation should keep that same lifecycle rule
and additionally retire any old viewport-update worker/controller.

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

## Choosing a Tiled Level at Query Time

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

## Tiled Query Path

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

On each `Add / Update in viewer` click, create a fresh `Points` layer for the
new render request.

In exact/direct mode, use the current direct path:

```text
_points_value_index.load_points(...)
  -> fully materialized selected points
  -> fresh static napari Points layer
  -> no camera listener
```

In tiled/live mode, create one fresh live `Points` layer for the request:

```python
layer = viewer.add_points(
    np.empty((0, 2), dtype=np.float32),
    features={"gene": np.array([], dtype=object)},
    size=2,
    face_color="gene",
)
```

On viewport changes, query the tile source and update that same live layer:

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

When the user clicks `Add / Update in viewer` again, the new request should
create a new `Points` layer object and retire the previous Harpy-managed points
render layer for the same source. If the previous request was tiled/live, also
stop its debounced camera listener and ignore any stale query results.

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

1. Preserve the current exact/direct path for selections whose total count is
   within the render budget.
2. Add the render-mode decision on `Add / Update in viewer`.
3. Encode genes into `gene_code`.
4. Build one finest-level tiled cache.
5. Write one Parquet file per tile.
6. Write `manifest.parquet`.
7. Implement tiled viewport query.
8. Add the guarded live `Points` layer update helper and async-slicing regression test.
9. Add live tiled layer lifecycle management: fresh layer per request, retire
   previous layer, stop stale viewport listeners.
10. Add `gene_tile_counts.parquet`.
11. Add coarser sampled levels.
12. Replace tile files with sharded Parquet row groups.

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
Add / Update in viewer
  -> compute selected_total_count
  -> if selected_total_count <= render_budget: exact/direct static layer
  -> otherwise: tiled/live layer

tiled/live viewport update
  -> tile range -> manifest lookup -> optional gene index lookup -> load tile chunks
  -> exact bbox/gene filter
  -> guarded in-place update of the live napari Points layer
```

This gives you a Neuroglancer-like point visualization system while still using an ordinary napari `Points` layer for rendering.
