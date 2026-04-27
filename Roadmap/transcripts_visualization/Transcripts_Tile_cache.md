# Transcript Tile Cache Implementation Plan

This document describes the first concrete implementation of the Harpy transcript visualization cache.

The starting point is a backed `SpatialData` zarr store with a points element. For the first implementation, Harpy will not support unbacked `SpatialData` objects for cache generation. The cache is built directly from the stored coordinates in the points element's on-disk `points.parquet` data, and the writer does not inspect or apply `SpatialData` transformations.

The cache builder itself starts from a `dask.dataframe.DataFrame` with `x`, `y`, and `gene` columns and writes a dedicated visualization cache to disk.

## Target Layout

For a points element inside a SpatialData zarr store, write the cache beside the canonical points Parquet file:

```text
<sdata.zarr>/
  points/
    <points_key>/
      points.parquet
      transcripts_vis/
        metadata.json
        manifest.parquet
        genes.parquet
        levels/
          level_0.parquet
          level_1.parquet
          ...
          level_n.parquet
```

`points.parquet` remains the canonical exact table. `transcripts_vis/` is a Harpy-owned visualization cache and can be deleted or rebuilt without changing the source data.

## Public API

Add the first implementation to a focused module:

```text
src/napari_harpy/_transcript_tiles.py
```

Primary public function:

```python
def build_transcript_visualization_cache(
    points: dask.dataframe.DataFrame,
    output_path: Path,
    *,
    x: str = "x",
    y: str = "y",
    gene: str = "gene",
    transcript_id: str | None = None,
    leaf_tile_size: float = 1024.0,
    n_levels: int | None = None,
    max_points_per_tile: int = 50_000,
) -> TranscriptTileCache:
    ...
```

Return a small dataclass:

```python
@dataclass(frozen=True)
class TranscriptTileCache:
    path: Path
    metadata_path: Path
    manifest_path: Path
    genes_path: Path
    levels_path: Path
    schema_version: str
    n_levels: int
    finest_level: int
    leaf_tile_size: float
    x_origin: float
    y_origin: float
    x_min: float
    x_max: float
    y_min: float
    y_max: float
```

Add a thin SpatialData-facing helper after the dataframe builder is stable:

```python
def build_transcript_visualization_cache_for_points_element(
    sdata: SpatialData,
    points_key: str,
    *,
    x: str = "x",
    y: str = "y",
    gene: str = "gene",
    transcript_id: str | None = None,
    leaf_tile_size: float = 1024.0,
    n_levels: int | None = None,
    max_points_per_tile: int = 50_000,
) -> TranscriptTileCache:
    ...
```

This helper should:

- require `sdata.is_backed()` and `sdata.path is not None`;
- require `points_key in sdata.points`;
- read the stored points dataframe as-is from the selected points element;
- call `build_transcript_visualization_cache(...)` with `output_path = Path(sdata.path) / "points" / points_key / "transcripts_vis"`.

The resulting cache is therefore in the native stored coordinate space of `points.parquet`.
Future reader or controller code can either use that same coordinate space directly or add explicit coordinate-system handling later.

## Dependencies

The active environment already has `dask.dataframe` and `pyarrow`, but they are not direct project dependencies. The implementation should add direct dependencies for the features it owns:

- `dask[dataframe]`
- `pyarrow`

This avoids relying on `spatialdata` transitive dependencies for Harpy-owned IO.

## Cache Schema

Use a module constant:

```python
TRANSCRIPT_TILE_CACHE_SCHEMA_VERSION = "harpy-transcripts-vis-0.1"
```

### `metadata.json`

One JSON object for cache-level metadata:

```text
schema_version: string
n_levels: int
finest_level: int
leaf_tile_size: float
x_origin: float
y_origin: float
x_min: float
x_max: float
y_min: float
y_max: float
```

This file stores global cache bounds and grid parameters once for the whole cache.
Do not repeat these values on every manifest row.

### `genes.parquet`

One row per gene:

```text
schema_version: string
gene_id: uint32
gene: string
n_transcripts: int64
```

Sort genes lexicographically for deterministic `gene_id` assignment in the first implementation.

### `manifest.parquet`

One row per Parquet row group. In the common case, that means one row per tile. If a tile is split into multiple row-group shards, there are multiple rows with the same `level`, `tile_x`, and `tile_y`.

Required columns:

```text
schema_version: string
level: int16
tile_id: string
tile_x: int64
tile_y: int64
n_points: int64
row_group: int32
level_file: string
is_exact: bool
```

Recommended extra column:

```text
tile_shard: int32
```

`tile_shard` is `0` for ordinary tiles. If a dense tile is split into several row groups, shard indices are `0..k`. The reader can still address data by `row_group`, but `tile_shard` makes the manifest easier to inspect and test.

Manifest rows are row-group lookup entries, not a place for repeated cache-level metadata.
For the first implementation, tile geometry is reconstructed from `level`, `tile_x`, `tile_y`,
and the cache-wide metadata in `metadata.json`.

### `levels/level_k.parquet`

Each level file stores tile-local point rows:

```text
tile_id: string
tile_x: int64
tile_y: int64
x_rel: float32
y_rel: float32
gene_id: uint32
transcript_id: optional original dtype
```

The absolute coordinates are reconstructed as:

```text
x = x_origin + tile_x * tile_size + x_rel
y = y_origin + tile_y * tile_size + y_rel
```

The first implementation should not quantize coordinates. Keep `x_rel` and `y_rel` as `float32`; add quantized integer storage only after benchmarking.

## Tile Grid

Compute global bounds from the source dataframe:

```text
x_min = min(points[x])
x_max = max(points[x])
y_min = min(points[y])
y_max = max(points[y])
```

Set the grid origin to the minimum bounds for the first version:

```text
x_origin = x_min
y_origin = y_min
```

For deterministic cache rebuilds, we may later floor origins to an integer or user-provided grid. Do not add that complexity initially.

Let `L` be the finest level:

```text
finest_level = n_levels - 1
```

Use:

```text
tile_size(level) = leaf_tile_size * 2 ** (finest_level - level)
```

For a point at one level:

```text
tile_x = floor((x - x_origin) / tile_size(level))
tile_y = floor((y - y_origin) / tile_size(level))
tile_id = f"{level}/{tile_x}/{tile_y}"
```

The finest level stores exact points. Coarser levels store sampled representative points.

## Level Count

If `n_levels` is provided, use it directly and validate that it is at least `1`.

If `n_levels is None`, choose it from the data extent:

```text
extent = max(x_max - x_min, y_max - y_min)
n_levels = max(1, ceil(log2(extent / leaf_tile_size)) + 1)
```

This makes `level_0` roughly cover the whole dataset in a small number of coarse tiles while `level_n` uses `leaf_tile_size`.

## Build Algorithm

### 1. Validate Inputs

Validate:

- `points` is a Dask dataframe;
- `x`, `y`, and `gene` columns exist;
- `x` and `y` are numeric;
- `transcript_id` exists when provided;
- `leaf_tile_size > 0`;
- `max_points_per_tile > 0`;
- `n_levels is None or n_levels >= 1`.

For the SpatialData helper, additionally validate:

- the SpatialData object is backed by zarr;
- the points element exists.

### 2. Prepare Output Directory

Write into a temporary sibling directory first:

```text
transcripts_vis.tmp-<uuid>/
```

On success, atomically replace the final `transcripts_vis/` directory. This prevents a failed build from leaving a half-valid cache.

For the first implementation, if `output_path` already exists, remove or replace it only inside the final atomic swap step. Do not mutate the canonical `points.parquet`.

### 3. Compute Bounds

Use one Dask reduction to compute:

```text
x_min, x_max, y_min, y_max
```

Persist these values in the returned `TranscriptTileCache` and in `metadata.json`.

### 4. Build Gene Mapping

Compute gene counts:

```python
gene_counts = points[gene].value_counts().compute()
```

Normalize gene labels to strings for the first implementation. Sort by gene name and assign stable integer `gene_id`.

Write `genes.parquet` with `pyarrow`.

Apply the mapping to the source dataframe before level construction. For the first implementation, this can be done partition-wise:

```python
gene_to_id = {"Actb": 0, "Gapdh": 1, ...}

def encode_gene_partition(partition):
    partition = partition.copy()
    partition["gene_id"] = partition[gene].astype(str).map(gene_to_id).astype("uint32")
    return partition
```

Avoid carrying the original gene string into level files.

### 5. Build Finest Exact Level

For `level = finest_level`:

1. Compute `tile_x`, `tile_y`, `tile_id`, `x_rel`, and `y_rel`.
2. Keep columns: `tile_id`, `tile_x`, `tile_y`, `x_rel`, `y_rel`, `gene_id`, optional `transcript_id`.
3. Partition or group rows by tile.
4. Write `levels/level_<finest_level>.parquet` using `pyarrow.parquet.ParquetWriter`.
5. Write one row group per tile, or split a dense tile into shards of at most `max_points_per_tile`.
6. Add one manifest row per written row group with `is_exact = True`.

The first implementation can favor correctness over maximum scalability: group tile data with Dask, materialize one tile at a time, and write row groups through pyarrow. If this becomes too slow for very large inputs, optimize the grouping/shuffle path after the schema is proven.

### 6. Build Coarser Sampled Levels

For each level from `finest_level - 1` down to `0`:

1. Compute tile membership at that level from the exact source dataframe, not from already sampled parent levels.
2. Within each tile, choose at most `max_points_per_tile` representative points.
3. Compute tile-local coordinates for that level.
4. Write one row group per tile or tile shard.
5. Add manifest rows with `is_exact = False`.

Initial sampling strategy:

- deterministic hash sampling by transcript row identity when `transcript_id` is available;
- otherwise deterministic hash sampling from `(x, y, gene_id)`;
- cap each tile at `max_points_per_tile`.

This is simpler than a full spatially stratified sampler and gives stable cache rebuilds. Once the reader path exists, add spatial stratification if overview points visibly clump.

### 7. Write Manifest

Collect manifest rows while writing level files, then write `manifest.parquet` last.

Write `metadata.json` before `manifest.parquet`.

The manifest is the cache validity anchor: if it is missing, the cache is invalid. Writing it last means readers never see a complete-looking cache before level files, genes, and metadata are present.

### 8. Atomic Finalization

After all files are written and basic validation passes:

1. remove any existing backup temp directory from this build;
2. rename existing `output_path` to a temporary old path if present;
3. rename the completed temporary cache directory to `output_path`;
4. remove the old cache directory.

Use only paths inside the selected points element directory.

## Reader-Oriented Invariants

The writer should guarantee these invariants because the later napari controller will rely on them:

- every manifest row points to an existing `level_file`;
- every manifest row's `row_group` exists in that level file;
- all rows in a row group belong to the manifest row's `level`, `tile_x`, and `tile_y`;
- `level_0` is the coarsest level;
- `finest_level = n_levels - 1` is exact;
- `metadata.json` stores the global bounds and shared grid parameters for the whole cache;
- all level files use the same `x_origin` and `y_origin` from `metadata.json`;
- `tile_size` is derived from `metadata.json` and is constant within each level;
- the cache coordinates are the native stored coordinates from the points element's `points.parquet`;
- `gene_id` values are defined in `genes.parquet`;
- no full-table `.compute()` is needed by future viewport reads.

## Testing Plan

Add focused tests in:

```text
tests/test_transcript_tiles.py
```

Test cases:

1. Builds the expected directory layout from a small Dask dataframe.
2. Writes deterministic `genes.parquet` with stable `gene_id` values and counts.
3. Writes `metadata.json` with global bounds, origins, and level metadata.
4. Writes `manifest.parquet` with one row per row group.
5. Finest level reconstructs exact source coordinates and gene IDs.
6. Dense tiles split into multiple row groups when `max_points_per_tile` is small.
7. Coarse levels are sampled and stay within the per-tile budget.
8. Invalid inputs raise clear `ValueError`s for missing columns, bad `n_levels`, bad tile size, and bad transcript id column.
9. SpatialData helper rejects unbacked `SpatialData`.
10. SpatialData helper writes to `points/<points_key>/transcripts_vis`.
11. SpatialData helper builds the cache from the stored points coordinates without inspecting transformations.

For tests, use small pandas dataframes converted to Dask dataframes. Read written Parquet files with `pyarrow.parquet.ParquetFile` so tests can assert row-group counts directly.

## First Implementation Milestones

### Milestone 1: Pure dataframe writer

Implement:

- `TranscriptTileCache`;
- input validation;
- bounds computation;
- gene mapping;
- exact finest-level writer;
- manifest writer;
- tests for exact reconstruction and manifest row groups.

This milestone does not need sampled coarser levels yet if `n_levels=1`.

### Milestone 2: Multiscale sampled levels

Implement:

- automatic `n_levels`;
- coarser level construction;
- deterministic sampling;
- per-tile budget tests.

### Milestone 3: SpatialData entry point

Implement:

- backed-zarr validation;
- points element lookup;
- native stored-coordinate contract;
- output path resolution;
- tests using a backed SpatialData fixture.

### Milestone 4: Reader prototype

Add a separate reader class after the writer format is stable:

```python
class TranscriptTileStore:
    def from_path(path: Path) -> TranscriptTileStore: ...
    def tiles_for_bounds(self, bounds, level: int) -> pandas.DataFrame: ...
    def choose_level(self, bounds, budget: int) -> int: ...
    def read_manifest_rows(self, rows, gene_ids=None) -> pandas.DataFrame: ...
```

This reader is the bridge to the future napari controller, but it should remain independent from Qt and napari.

## Deliberately Deferred

Do not include these in the first writer:

- applying or resolving `SpatialData` coordinate transformations;
- Morton ordering inside tiles;
- coordinate quantization;
- per-gene offsets inside each tile;
- gene-aware overview sampling;
- napari camera/controller integration;
- remote zarr stores.

These are all useful, but they should come after the on-disk contract is proven and benchmarked.
