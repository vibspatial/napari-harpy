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

Prefer the persistent point cache whenever a valid cache exists for the selected
points element, coordinate system, coordinate transform, and category column.
The direct SpatialData path remains the fallback for users who have not built a
cache or explicitly disable cached rendering.

When a valid cache exists, use:

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

In cached mode, the napari `Points` layer should only contain the currently
visible subset.

---

## Cache-First Rendering Policy

Users keep the existing render-budget workflow:

```text
user chooses render budget
user selects genes, a few genes, or all/no genes
user clicks Add / Update in viewer
```

On each `Add / Update in viewer` click, choose the render path from cache
availability:

```text
valid cache available
  -> cached/live mode
  -> query the persistent tile cache using render_budget as the target for level selection
  -> trust the selected cache level rather than resampling the query result
  -> create a fresh live napari Points layer for the render request
  -> update that layer on debounced viewport changes

no valid cache, or cached rendering disabled
  -> direct fallback mode
  -> reuse the current direct implementation and its render-budget behavior
  -> create a fresh static napari Points layer
  -> no viewport-driven updates
```

With a valid cache, selected genes and all/no-gene selections use the same
cached query path. If the finest visible cached level fits the render budget,
the cache returns exact points for the viewport. If not, the query chooses a
coarser sampled level.

Render-path selection happens only when the user changes the selection or render
budget and clicks `Add / Update in viewer`. Camera movement must not switch
between cached and direct modes. Camera movement only updates the current
cached/live layer and may choose a different cached level for the new viewport.

Cached rendering does not mean "always sampled"; it means "sampled when needed,
exact when locally affordable."

Layer lifecycle should stay simple:

```text
each Add / Update click creates a new napari Points layer object
the previous Harpy-managed points render layer for the same source is retired
any previous cached/live query controller/listener is stopped
the new request is either cached/live or direct/static fallback
```

This matches the current direct points adapter behavior, which replaces the old
Harpy-managed points layer with a fresh layer object for the same source
identity. The cached/live implementation should keep that same lifecycle rule
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

Use sharded Parquet files with one row group per logical tile from the start.

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

Required row order inside each tile:

```text
gene_code, sample_rank
```

Rows should be grouped primarily by `gene_code` so gene switches can be handled
snappily. Within each gene range, `sample_rank` provides a deterministic order
for stable display and any future gene-local operations.

---

## Sorting Strategy

At build time, derive:

```python
tile_x = floor((x - origin_x) / tile_size_x)
tile_y = floor((y - origin_y) / tile_size_y)
tile_id = make_tile_id(tile_x, tile_y)
gene_code = categorical_code(gene)
sample_rank = stable_hash(id)
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

Use a level-independent `sample_rank`:

```python
sample_rank = stable_hash(id)
```

This makes samples nested across levels: points that are visible in a coarser
sample tend to remain present as the viewer moves to finer levels. That should
reduce visual popping during zoom changes.

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

Build every level as a self-contained representation of the full normalized
points source. Do not build residual levels where points emitted at one level
are removed from the input for the next level. The viewport query displays one
chosen level at a time, so each level must be able to answer the query on its
own.

Use this build shape:

```python
def build_levels(normalized_path, levels, cache_dir):
    for level in levels:
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

        write_level_tiles_and_manifest(
            emitted,
            level=level,
            cache_dir=cache_dir,
        )
```

Conceptually, for each level:

```python
points = points.with_columns([
    tile_x_expr(level),
    tile_y_expr(level),
    tile_id_expr(level),
    sample_rank_expr(point_id),
])
```

Then rank points within each tile by `sample_rank`:

```python
ranked = points.with_columns(
    row_number()
    .over("tile_id")
    .sort_by("sample_rank")
    .alias("rank_in_tile")
)
```

For sampled levels, keep only the first `level.max_points_per_tile` points in
each tile. For the finest level, keep all points:

```python
if level.is_finest:
    emitted = ranked
else:
    emitted = ranked.filter(rank_in_tile <= level.max_points_per_tile)
```

Use a stable hash for sampling:

```python
sample_rank = stable_hash(id)
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

```text
one Parquet row group per tile
multiple tiles per shard file
```

The manifest row should point to the physical location:

```text
shard_path
row_group
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
render_budget
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
estimated_count <= render_budget
```

If all levels exceed the limit, pick the coarsest level and display that cached
representation as-is. Do not apply a second deterministic cap at query time.
Cache parameters should be chosen so the coarsest level remains acceptable for
overview rendering.

---

## Tiled Query Path

Given a napari viewport and selected genes:

```text
xmin, xmax, ymin, ymax
selected_gene_codes
render_budget
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

### 6. Trust the selected cache level

Do not apply an additional viewport-wide budget trim after loading tiles. The
cache already encodes the approximation through its level choice, tile size, and
`max_points_per_tile`.

Unexpected manifest/cache mismatches should be treated as diagnostics, not as a
normal reason to resample the query result.

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

If a valid cache is available, create one fresh live `Points` layer for the
request:

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
    render_budget=100_000,
)

update_live_points_layer(
    layer,
    coords_yx=visible.coords_yx,
    features={"gene": visible.gene_names},
)
```

The cached query may choose the finest complete level, in which case the visible
points are exact. If the finest visible level exceeds the render budget, the
query chooses a coarser sampled level.

If no valid cache is available, use the current direct path:

```text
_points_value_index.load_points(...)
  -> materialized direct selection using the current render-budget behavior
  -> fresh static napari Points layer
  -> no camera listener
```

When the user clicks `Add / Update in viewer` again, the new request should
create a new `Points` layer object and retire the previous Harpy-managed points
render layer for the same source. If the previous request was cached/live, also
stop its debounced camera listener and ignore any stale query results.

### Safe In-Place Points Layer Updates

Cached/live mode should keep one persistent napari `Points` layer and update
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
camera debounce and a ~100k render-budget level-selection target are plausible
starting points.

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
(level, rounded_bbox, selected_gene_codes, render_budget)
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

## Implementation Plan

### Slice 1: Cache Contracts

Define the cache metadata, cache validity checks, and public query interfaces.
This should include:

```text
cache_metadata.json
cache identity / invalidation rules
ViewQuery
PointTileSource
manifest schema
gene dictionary schema
tile row schema
```

At the end of this slice, the code should be able to answer whether a valid
cache exists for a points element, coordinate system, coordinate transform, and
category column. No viewer behavior needs to change yet.

### Slice 2: Finest Exact Cache

Build the finest complete level using the final storage layout:

```text
sharded Parquet
one row group per logical tile
manifest.parquet
genes.parquet
rows sorted by gene_code, sample_rank
```

This gives an exact cached representation and validates the storage/query shape
before adding sampled levels.

### Slice 3: Cached Exact Query

Implement viewport queries against the finest complete level:

```text
bbox + selected genes
  -> tile ids
  -> manifest lookup
  -> read row groups
  -> bbox filter
  -> gene filter
  -> napari-ready coords/features
```

At the end of this slice, a valid cache can render exact visible points from
the finest level.

### Slice 4: Cache-First Viewer Integration

Change `Add / Update in viewer` behavior to:

```text
valid cache available
  -> cached/live layer

no valid cache, or cached rendering disabled
  -> current direct fallback
```

Every click should still create a fresh Harpy-managed `Points` layer and retire
the previous one for the same source.

### Slice 5: Live Layer Updates

Add the guarded in-place `Points` layer update helper and make cached rendering
use one persistent live layer per request.

This slice should include:

```text
debounced viewport updates
safe layer.data / layer.features mutation
napari async-slicing regression test
stale query result protection
```

### Slice 6: Gene Index

Add:

```text
gene_tile_counts.parquet
```

Use it to:

```text
skip irrelevant tiles
estimate visible selected-gene counts
make gene switching fast
```

This is where `gene_code, sample_rank` row ordering starts paying off.

### Slice 7: Coarser Sampled Levels

Add sampled levels built independently from the normalized source:

```text
sample_rank = stable_hash(point_id)
rank within tile by sample_rank
keep rank_in_tile <= level.max_points_per_tile
```

Then `render_budget` can select the finest acceptable cached level.

### Slice 8: Performance Polish

Add the interaction and IO pieces needed for large datasets:

```text
tile payload LRU cache
async row-group reads
camera debounce
ignore stale queries
basic timing/logging
```

Benchmark at least:

```text
all genes overview
single gene overview
pan/zoom at multiple zoom levels
large selection switching
```

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
    render_budget: int
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
  -> if valid cache is available: cached/live layer using render_budget for level selection
  -> otherwise: direct/static fallback using the current implementation

cached/live viewport update
  -> tile range -> manifest lookup -> choose level -> optional gene index lookup -> load tile chunks
  -> exact bbox/gene filter
  -> guarded in-place update of the live napari Points layer
```

This gives you a Neuroglancer-like point visualization system while still using an ordinary napari `Points` layer for rendering.
