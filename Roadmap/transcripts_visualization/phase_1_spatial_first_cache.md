# Phase 1A: Spatial-First Cache Offline Writer

This document turns Phase 1A from [Transcripts_Tile_cache.md](/Users/arne.defauw/VIB/napari_harpy_transcrips/Roadmap/transcripts_visualization/Transcripts_Tile_cache.md) into a concrete implementation plan.

Phase 1A is the offline writer only. It does not include the napari runtime, the transcript controller, gene subset UI, `tile_gene_index.parquet`, or gene-aware overview sampling.

## Goal

Build a working offline writer that converts a transcript-like points table into the first Harpy transcript visualization cache:

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

The writer must:

- start from a `dask.dataframe.DataFrame` with at least `x`, `y`, and `gene`;
- write a spatial-first multiscale cache;
- keep the finest level unsampled, with one cache row per source row;
- build coarser levels by deterministic spatially stratified sampling per tile;
- finalize through a staged replacement so incomplete new caches are never exposed as valid.

## Deliverables

Phase 1A should produce:

- `src/napari_harpy/_transcript_tiles.py`
- `tests/test_transcript_tiles.py`
- direct project dependencies for `dask[dataframe]` and `pyarrow` if still needed

The module should expose at least:

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
    max_rows_per_row_group: int = 50_000,
    coarse_tile_budget: int = 50_000,
) -> TranscriptTileCache:
    ...


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
    max_rows_per_row_group: int = 50_000,
    coarse_tile_budget: int = 50_000,
) -> TranscriptTileCache:
    ...
```

`max_rows_per_row_group` controls physical Parquet sharding for unsampled and sampled level files.
`coarse_tile_budget` controls how many representative points may be stored in one sampled coarse tile.
Keep these separate so IO layout tuning and overview-density tuning do not become coupled.

All level files store tile-local `x_rel` and `y_rel` coordinates as `float32`.
In this cache, "exact" means unsampled/full-membership rather than full-precision coordinate storage.
The canonical full-precision coordinates remain in `points.parquet`.

## Non-Goals

Do not include these in Phase 1A:

- napari UI or camera integration
- transcript runtime reader/controller
- warm cache implementation
- gene subset selection
- `tile_gene_index.parquet`
- gene-aware overview sampling
- coordinate transform resolution beyond stored points coordinates
- Morton ordering or coordinate quantization

## Proposed Module Shape

Keep the public API small and move most logic into internal helpers. A reasonable first structure inside `src/napari_harpy/_transcript_tiles.py` is:

```text
TranscriptTileCache                      # return dataclass
build_transcript_visualization_cache     # main public writer
build_transcript_visualization_cache_for_points_element

_validate_points_input
_validate_backed_points_element
_compute_bounds_and_level_config
_build_gene_table
_encode_gene_partition
_annotate_partition_for_level
_sample_partition_for_coarse_level
_write_level_file
_write_metadata_json
_write_manifest_parquet
_finalize_cache_with_staged_replacement
```

The internal function names do not need to match this exactly, but Phase 1A should keep the writer decomposed into testable units instead of one monolithic function.

## Recommended Implementation Order

Implement Phase 1A in the order below.

### 1. Define the Core Types and Constants

Add:

- `TRANSCRIPT_TILE_CACHE_SCHEMA_VERSION`
- `TranscriptTileCache` dataclass

The dataclass should include:

- final cache paths
- schema version
- level metadata
- bounds and origins

This gives the rest of the module a stable return contract from the start.

### 2. Input Validation

Implement validation first so later failures are clearer.

Validate for dataframe entry point:

- `points` is a Dask dataframe
- `x`, `y`, and `gene` columns exist
- `x` and `y` are numeric
- `transcript_id` exists if requested
- `leaf_tile_size > 0`
- `max_rows_per_row_group > 0`
- `coarse_tile_budget > 0`
- `n_levels is None or n_levels >= 1`

Also decide explicit behavior for:

- empty dataframes
- null gene values
- `NaN` / `inf` coordinates

Recommended first behavior:

- reject empty inputs with a clear `ValueError`
- reject rows with invalid coordinates rather than silently dropping them
- coerce `gene` to string during gene mapping, but reject missing gene values if they would become ambiguous

Validate for SpatialData entry point:

- `sdata.is_backed()`
- `sdata.path is not None`
- `points_key in sdata.points`
- selected points element resolves to exactly one on-disk element path

Use `sdata.locate_element(...)` or the equivalent SpatialData API to resolve the selected points element path.
Do not assume the path is always `points/<points_key>`.
Raise a clear `ValueError` if the element cannot be located or resolves to multiple zarr paths.

### 3. Output Directory Setup and Staged Finalization

Before writing data files, implement the temp-directory and staged replacement path. This should exist early so later integration work does not need to be rewritten.

Recommended behavior:

- write to `transcripts_vis.tmp-<uuid>/`
- only write into the temp directory during the build
- write `metadata.json`, `genes.parquet`, level files, then `manifest.parquet`
- move into place only once the build is complete
- if replacing an existing cache, move the old cache to a sibling backup path first
- if final replacement fails after moving the old cache aside, restore the old cache whenever possible

Implementation notes:

- keep all temp and replacement paths inside the points-element directory
- avoid mutating `points.parquet`
- treat `manifest.parquet` as the final validity anchor
- do not describe directory replacement as strictly atomic; there may be a brief missing-output window during the staged swap

### 4. Bounds, Origins, and Level Configuration

Implement one function that computes:

- `x_min`, `x_max`, `y_min`, `y_max`
- `x_origin`, `y_origin`
- `n_levels`
- `finest_level`

Recommended first implementation:

- use `x_origin = x_min`
- use `y_origin = y_min`
- if `n_levels is None`, derive it from the max extent and `leaf_tile_size` using:

```text
extent = max(x_max - x_min, y_max - y_min)
if extent <= leaf_tile_size:
    n_levels = 1
else:
    n_levels = ceil(log2(extent / leaf_tile_size)) + 1
```

- this means:
  - if the dataset extent is zero, smaller than one leaf tile, or exactly one leaf tile, use `n_levels = 1`;
  - otherwise add coarser levels until `level_0` covers the whole dataset in a small number of tiles

This logic should be unit-tested separately because it drives every later tile calculation.
Use the same tile assignment formula at every level:

```text
tile_x = floor((x - x_origin) / tile_size(level))
tile_y = floor((y - y_origin) / tile_size(level))
```

Tiles are half-open intervals: `[tile_start, tile_start + tile_size)`.
Points exactly on an internal tile boundary belong to the tile on the positive x/y side.
Do not clamp points at `x == x_max` or `y == y_max` into the previous tile; they follow the same floor rule even when that creates a max-edge tile.
This keeps tile-local coordinates in `[0, tile_size)` instead of allowing `x_rel == tile_size` or `y_rel == tile_size`.

### 5. Gene Dictionary and `genes.parquet`

Implement deterministic gene mapping next.

Required behavior:

- compute gene counts with Dask
- normalize genes to strings
- sort genes lexicographically
- assign `gene_id: uint32`
- write `genes.parquet`

The resulting table should contain:

- `schema_version`
- `gene_id`
- `gene`
- `n_transcripts`

Then add the partition-wise `gene_id` encoding step to the working dataframe.

At the end of this step, all later level-writing code should work with:

- `x`
- `y`
- `gene_id`
- optional `transcript_id`

and not carry the original gene string into the level files.

### 6. Tile Annotation Utilities

Before writing any levels, implement reusable level-annotation helpers.

For a given level, compute:

- `tile_size(level)`
- `tile_x`
- `tile_y`
- `tile_id`
- `x_rel`
- `y_rel`

These helpers should be shared by:

- finest unsampled level writing
- coarser sampled level writing

This is also the right place to standardize dtypes for:

- `tile_x`, `tile_y`
- `x_rel`, `y_rel`
- `gene_id`

### 7. Finest Unsampled Level Writer

Implement the finest unsampled level before coarser sampled levels.

Required behavior:

1. annotate rows for `level = finest_level`
2. keep only the level-file columns
3. group by tile
4. write one row group per tile, or split dense tiles into shards
5. collect one manifest row per written row group

Important constraints:

- use `pyarrow.parquet.ParquetWriter`
- row-group sharding must respect `max_rows_per_row_group`
- all rows in a written row group must belong to exactly one `(level, tile_x, tile_y)`

Recommended simplification for Phase 1A:

- correctness first
- materialize one tile group at a time if needed
- accept that the grouping strategy may be optimized later

### 8. Coarser Sampled Level Writer

Once the finest unsampled level works, add coarse levels.

For each level from `finest_level - 1` down to `0`:

1. derive tile membership from the canonical source dataframe, not from already sampled parent levels
2. subdivide each coarse tile into a fixed micro-grid
3. allocate `coarse_tile_budget` across occupied micro-grid cells
4. choose deterministic representative points within each occupied cell
5. annotate sampled rows with tile-local coordinates for that level
6. write row groups and manifest rows

Recommended first implementation choices:

- keep the micro-grid size as a private constant and start with `8 x 8`
- compute a deterministic `cell_id` from the per-tile micro-grid coordinates
- if occupied cells `<= coarse_tile_budget`, assign quota `1` to each occupied cell, then distribute the remaining quota approximately by occupancy using a largest-remainder rule with stable tie-break on `cell_id`
- if occupied cells `> coarse_tile_budget`, assign quota `1` only to the `coarse_tile_budget` occupied cells with highest occupancy, with stable tie-break on `cell_id`
- if `transcript_id` is available, compute a stable per-row ordering key from a canonical encoding of `transcript_id`
- otherwise, compute a stable per-row ordering key by hashing a stable binary encoding of `(x, y, gene_id)`
- for the fallback encoding, pack `float64(x)`, `float64(y)`, and `uint32(gene_id)` in canonical little-endian order before hashing with a stable hash function from the standard library
- never use Python's built-in `hash()` for sampling
- after selecting rows from each cell, sort the final sampled rows deterministically before writing so rebuilds do not depend on Dask partition order
- cap each sampled coarse tile at `coarse_tile_budget`

Keep Phase 1A sampling spatial-only. Do not add gene-aware overview sampling yet.

Recommended concrete deterministic algorithm for one coarse tile:

1. compute the micro-grid cell coordinates for every candidate row in the tile
2. compute `cell_id` from those cell coordinates
3. count rows per occupied cell
4. assign cell quotas with the rules above
5. compute one stable per-row ordering key
6. within each occupied cell, sort rows by that ordering key and keep the first `quota[cell]`
7. concatenate selected rows from all occupied cells
8. sort the selected rows deterministically before writing

The key requirement is that repeating the build with the same inputs must produce the same sampled rows even if Dask partition order changes.

### 9. `metadata.json` and `manifest.parquet`

Once all levels can be written, add the cache metadata writers.

`metadata.json` should be written once with:

- schema version
- `n_levels`
- `finest_level`
- `leaf_tile_size`
- `x_origin`, `y_origin`
- `x_min`, `x_max`, `y_min`, `y_max`

`manifest.parquet` should be written from the collected row-group metadata and include:

- `schema_version`
- `level`
- `tile_id`
- `tile_x`
- `tile_y`
- `n_points`
- `row_group`
- `level_file`
- `is_exact`
- optional `tile_shard`

Write `manifest.parquet` last.

### 10. SpatialData Helper

After the dataframe writer is stable, implement the backed SpatialData wrapper.

Responsibilities:

- validate backed SpatialData input
- resolve the points element
- resolve the points element's unique on-disk path with `sdata.locate_element(...)` or equivalent
- read the stored points dataframe as-is
- compute `output_path = Path(sdata.path) / resolved_points_element_path / "transcripts_vis"`
- delegate to `build_transcript_visualization_cache(...)`

Phase 1A should keep the cache contract in the stored coordinate space of `points.parquet`.

## Testing Plan

Keep all main writer tests in:

```text
tests/test_transcript_tiles.py
```

Recommended test groups:

### Group A: Input Validation

- missing `x`, `y`, or `gene`
- invalid `transcript_id`
- invalid `leaf_tile_size`
- invalid `max_rows_per_row_group`
- invalid `coarse_tile_budget`
- invalid `n_levels`
- zero-extent data computes `n_levels = 1`
- tile-boundary coordinates follow the documented half-open grid convention
- unbacked SpatialData rejection

### Group B: Cache Layout and Metadata

- expected directory layout is created
- `metadata.json` has the expected bounds, origins, and level metadata
- `genes.parquet` has stable deterministic ids and counts
- `manifest.parquet` exists only after successful build

### Group C: Finest Unsampled Level Correctness

- finest-level coordinates reconstruct the source coordinates within `float32` tolerance
- `gene_id` values match `genes.parquet`
- dense tiles split into multiple row groups when `max_rows_per_row_group` is small
- manifest row-group accounting matches the actual Parquet file

### Group D: Coarse Sampled Levels

- coarse levels stay within `coarse_tile_budget`
- the number of stored points decreases or stays bounded at coarser levels
- sampling is deterministic across rebuilds with identical inputs

### Group E: SpatialData Entry Point

- cache path resolves under `<resolved_points_element_path>/transcripts_vis`
- path resolution rejects missing or ambiguous points-element locations
- stored points coordinates are used directly
- no transformation logic is applied

### Group F: Staged Finalization

- failed writes do not leave a complete-looking cache behind
- rebuilding over an existing cache either installs the completed new cache or restores the previous valid cache when replacement fails

Testing notes:

- use small pandas fixtures converted to Dask dataframes
- inspect row groups with `pyarrow.parquet.ParquetFile`
- prefer direct file assertions over indirectly testing through later readers

## Benchmark Plan

Phase 1A should include at least lightweight benchmark scripts or notebooks, even if they are not part of automated tests.

Recommended datasets:

1. tiny synthetic fixture
   Purpose:
   finest-level membership and coordinate reconstruction correctness

2. medium synthetic skewed fixture
   Purpose:
   stress uneven tile density and coarse-level sampling

3. one real transcript dataset with multiple meaningful levels
   Purpose:
   sanity-check write time, disk footprint, and level counts on real data

Recommended benchmark outputs:

- total build time
- per-level write time
- total disk size
- per-level file sizes
- manifest row count
- number of row groups in the finest level

Phase 1A does not yet need interactive napari benchmarks. Those belong to later phases.

## Suggested Work Packages

If Phase 1A is implemented incrementally, the cleanest checkpoints are:

1. skeleton module + dataclass + validation
2. bounds/level config + metadata writer
3. gene dictionary + `genes.parquet`
4. finest unsampled level writer + manifest
5. coarse sampled level writer
6. staged finalization with rollback
7. SpatialData helper
8. test completion
9. benchmark pass on synthetic and real data

Each checkpoint should leave the branch in a runnable, testable state.

## Phase 1A Exit Criteria

Phase 1A is complete when:

- the offline writer can build the full spatial-first cache from a Dask dataframe;
- the backed SpatialData helper can build the same cache from a points element;
- all core writer tests pass;
- the resulting cache layout matches the roadmap contract;
- coarse sampled levels exist and respect the configured budget;
- the implementation is stable enough that the next phase can build a runtime reader on top of it without revisiting the on-disk format.
