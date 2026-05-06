# Phase 1A: Spatial-First Cache Offline Writer

This document turns Phase 1A from [Transcripts_Tile_cache.md](/Users/arne.defauw/VIB/napari_harpy/Roadmap/transcripts_visualization/Transcripts_Tile_cache.md) into a concrete implementation plan.

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

- start from a backed `SpatialData` points element whose stored points table resolves to a `dask.dataframe.DataFrame` with at least `x`, `y`, and `gene`;
- write a spatial-first multiscale cache;
- keep the finest level unsampled, with one cache row per source row;
- build coarser levels by deterministic spatially stratified sampling per tile;
- finalize through a staged replacement so incomplete new caches are never exposed as valid.

## Deliverables

Phase 1A should produce:

- `src/napari_harpy/_transcript_tiles.py`
- `tests/test_transcript_tiles.py`
- direct project dependencies for `dask[dataframe]` and `pyarrow` if still needed

For now, Phase 1A should expose only the backed points-element builder:

```python
def build_transcript_visualization_cache_for_points_element(
    sdata: SpatialData,
    points_key: str,
    *,
    output_path: str | PathLike[str] | None = None,
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

Do not expose a standalone dataframe builder in Phase 1A.
The implementation may still use internal dataframe helpers after the points element is resolved.

### Points Element Builder Construction Contract

`build_transcript_visualization_cache_for_points_element(...)` is the main and only public entry point for Phase 1A.
It validates that `sdata` is backed, resolves `points_key` to exactly one stored points element path, and uses that stored points dataframe as the cache source.
By default, the final cache root is:

```text
Path(sdata.path) / resolved_points_element_path / "transcripts_vis"
```

If an explicit `output_path` is provided, normalize it with `Path(output_path)` and use it as the final cache root.

The points-element builder accepts `leaf_tile_size` and `n_levels` as construction inputs.
For Phase 1A, it uses them to create a regular level pyramid.
`leaf_tile_size` is in the stored coordinate units of the resolved points dataframe, not screen pixels.

If `n_levels` is provided, validate that it is at least `1`.
If `n_levels is None`, derive it from the source bounds and `leaf_tile_size` in Step 4.
After the final `n_levels` value is known:

```text
finest_level = n_levels - 1
```

Each level record is created as:

```text
tile_size = leaf_tile_size * 2 ** (finest_level - level)
is_exact = level == finest_level
```

For example, with `leaf_tile_size = 1024` and `n_levels = 3`, the builder creates:

```text
level  tile_size  is_exact
0      4096       false
1      2048       false
2      1024       true
```

These generated records are then stored explicitly in `TranscriptTileCache.levels` and in `metadata.json["levels"]`.
Readers should use those explicit records.
Level file paths follow the fixed layout convention `levels/level_<level>.parquet` and are derived from `level`.

`transcript_id` is optional.
If `transcript_id` is provided, Phase 1A validates that it is non-null and unique, and uses it as the stable row identity for deterministic coarse-level sampling.
If `transcript_id` is not provided, the builder creates an internal unique row id for the current build.
This id is sufficient to avoid duplicate sampling keys within one build, but deterministic coarse-level sampling across rebuilds is guaranteed only when the input dataframe row order and partitioning are stable.

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
TranscriptTileLevel                      # per-level return metadata
TranscriptTileCache                      # return dataclass
build_transcript_visualization_cache_for_points_element  # main public writer

_validate_points_element
_validate_cache_build_parameters
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

### 1. Define the Core Types and Constants — Implemented

Status:

- Implemented in `src/napari_harpy/_transcript_tiles.py`
- Covered by `tests/test_transcript_tiles.py`
- Verified with `pytest tests/test_transcript_tiles.py`
- Verified with `ruff check src/napari_harpy/_transcript_tiles.py tests/test_transcript_tiles.py`

Add:

- `TRANSCRIPT_TILE_CACHE_SCHEMA_VERSION`
- `TranscriptTileLevel` dataclass
- `TranscriptTileCache` dataclass

`TranscriptTileLevel` should include:

- `level`
- `tile_size`
- `is_exact`

`TranscriptTileCache` should include:

- final cache root path
- derived cache paths as properties
- schema version
- explicit per-level metadata as `levels: tuple[TranscriptTileLevel, ...]`
- bounds and origins

Use these exact dataclasses as the Phase 1A return contract:

```python
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TranscriptTileLevel:
    level: int
    tile_size: float
    is_exact: bool

    def __post_init__(self) -> None:
        if self.level < 0:
            raise ValueError("Transcript tile level must be non-negative.")
        if not math.isfinite(self.tile_size) or self.tile_size <= 0:
            raise ValueError("Transcript tile level tile_size must be finite and positive.")

    @property
    def level_file(self) -> str:
        return f"levels/level_{self.level}.parquet"


@dataclass(frozen=True)
class TranscriptTileCache:
    path: Path
    schema_version: str
    levels: tuple[TranscriptTileLevel, ...]
    x_origin: float
    y_origin: float
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    def __post_init__(self) -> None:
        if self.schema_version != TRANSCRIPT_TILE_CACHE_SCHEMA_VERSION:
            raise ValueError("Unsupported transcript tile cache schema version.")
        if not self.levels:
            raise ValueError("Transcript tile cache must contain at least one level.")

        level_ids = [level.level for level in self.levels]
        if level_ids != sorted(level_ids):
            raise ValueError("Transcript tile cache levels must be sorted by ascending level.")
        if len(set(level_ids)) != len(level_ids):
            raise ValueError("Transcript tile cache levels must not contain duplicate level ids.")
        if level_ids != list(range(level_ids[-1] + 1)):
            raise ValueError("Transcript tile cache levels must be contiguous from 0.")

        exact_levels = [level for level in self.levels if level.is_exact]
        if len(exact_levels) != 1:
            raise ValueError("Expected exactly one exact transcript tile level.")
        if exact_levels[0].level != level_ids[-1]:
            raise ValueError("The exact transcript tile level must be the finest level.")

        bounds_and_origins = [self.x_origin, self.y_origin, self.x_min, self.x_max, self.y_min, self.y_max]
        if not all(math.isfinite(value) for value in bounds_and_origins):
            raise ValueError("Transcript tile cache bounds and origins must be finite.")
        if self.x_min > self.x_max:
            raise ValueError("Transcript tile cache requires x_min <= x_max.")
        if self.y_min > self.y_max:
            raise ValueError("Transcript tile cache requires y_min <= y_max.")

    @property
    def metadata_path(self) -> Path:
        return self.path / "metadata.json"

    @property
    def manifest_path(self) -> Path:
        return self.path / "manifest.parquet"

    @property
    def genes_path(self) -> Path:
        return self.path / "genes.parquet"

    @property
    def levels_path(self) -> Path:
        return self.path / "levels"

    @property
    def n_levels(self) -> int:
        return len(self.levels)

    @property
    def finest_level(self) -> int:
        exact_levels = [level for level in self.levels if level.is_exact]
        if len(exact_levels) != 1:
            raise ValueError("Expected exactly one exact transcript tile level.")
        return exact_levels[0].level

    @property
    def leaf_tile_size(self) -> float:
        exact_levels = [level for level in self.levels if level.is_exact]
        if len(exact_levels) != 1:
            raise ValueError("Expected exactly one exact transcript tile level.")
        return exact_levels[0].tile_size
```

Do not store `leaf_tile_size`, `n_levels`, or `finest_level` as independent dataclass fields.
They are derived from `levels`, which avoids a second source of truth in the Python API.
Likewise, do not store `metadata_path`, `manifest_path`, `genes_path`, or `levels_path` as independent fields.
They are derived from the single cache root `path`.
When a caller needs an absolute path to a level file, it can combine `cache.path / level.level_file`.

Keep these dataclass validations even though the builder and future reader also validate their inputs.
They prevent tests or internal helpers from constructing an invalid cache object accidentally.

This gives the rest of the module a stable return contract from the start.

### 2. Input Validation — Implemented

Status:

- Implemented in `src/napari_harpy/_transcript_tiles.py`
- Covered by `tests/test_transcript_tiles.py`
- Verified with `pytest tests/test_transcript_tiles.py`
- Verified with `ruff check src/napari_harpy/_transcript_tiles.py tests/test_transcript_tiles.py`
- The points-element validation contract currently lives in the private `_validate_points_element(...)` helper
- Cache build parameter validation lives in the private `_validate_cache_build_parameters(...)` helper
- Public writer wiring remains part of Step 10

Implement validation before any output directory is created so later failures are clearer and failed inputs do not leave cache artifacts behind.

#### Points Element Parameter And Schema Validation

Validate the public points-element entry point before triggering a full dataframe compute where possible:

- `sdata.is_backed()`
- `sdata.path is not None`
- `points_key` is a string
- `points_key in sdata.points`
- selected points element resolves to exactly one on-disk element path
- resolved points element is a `dask.dataframe.DataFrame`
- if `output_path` is provided, it is path-like and represents the final `transcripts_vis/` cache root
- if `output_path` is provided, normalize it with `Path(output_path)` before returning or using it internally
- if `output_path` is not provided, compute it as `Path(sdata.path) / resolved_points_element_path / "transcripts_vis"`
- `x`, `y`, and `gene` are strings
- `x`, `y`, and `gene` columns exist
- `x` and `y` are numeric according to dataframe metadata
- `transcript_id is None` or is a string
- if `transcript_id` is provided, the column exists

Use `sdata.locate_element(...)` or the equivalent SpatialData API to resolve the selected points element path.
Do not assume the path is always `points/<points_key>`.
Raise a clear `ValueError` if the element cannot be located or resolves to multiple zarr paths.

#### Cache Build Parameter Validation

Validate cache construction parameters separately from points-element validation:

- `leaf_tile_size` is finite and `> 0`
- `max_rows_per_row_group` is an `int` and `> 0`, but not `bool`
- `coarse_tile_budget` is an `int` and `> 0`, but not `bool`
- `n_levels is None` or is an `int >= 1`, but not `bool`

#### Resolved Points Data-Quality Validation

Validate with Dask reductions before writing cache files:

- reject empty dataframes
- reject non-finite coordinates in `x` or `y`
- reject missing gene values before string coercion
- reject gene values whose string form is empty after stripping whitespace
- if `transcript_id` is provided:
  - reject missing values
  - reject duplicate values

Recommended implementation notes:

- use one or a small number of Dask reductions for these checks
- prefer clear `ValueError` messages that name the failing column
- do not silently drop invalid rows
- coerce `gene` to string only after missing and stripped-empty gene validation
- the builder may compute row count during validation because later metadata and empty-input rejection need it anyway

#### Transcript Identity Policy

If `transcript_id` is provided:

- use the validated column as the stable row identity for sampling
- deterministic coarse-level sampling should not depend on Dask partition order

If `transcript_id` is not provided:

- validation records that the internal-row-id fallback policy applies
- create an internal unique row id later during dataframe preparation or level writing, not during validation
- the internal id must be unique within that build
- deterministic coarse-level sampling across rebuilds is only guaranteed when input row order and partitioning are stable
- do not expose the internal id as a public source transcript identity

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
- derived `TranscriptTileLevel` records for every level

This function implements the level construction described in the Points Element Builder Construction Contract.

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

For Phase 1A, `x_origin` and `y_origin` intentionally equal `x_min` and `y_min`.
Still store origins separately because origins define the tile grid, while bounds define the data extent.
Future schema versions may use stable grid origins that differ from the data bounds.
Readers must use `x_origin` and `y_origin` for tile assignment and coordinate reconstruction rather than inferring origins from `x_min` and `y_min`.

After computing bounds and origins, validate that all values are finite, `x_min <= x_max`, and `y_min <= y_max`.
The builder should reject invalid bounds before constructing `TranscriptTileCache` or writing `metadata.json`.

This logic should be unit-tested separately because it drives every later tile calculation.
The same function should build the returned level records from `0` through `finest_level`.

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
- row identity for sampling, using validated `transcript_id` when provided or an internal unique row id for this build otherwise

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
3. process Dask partitions independently
4. within each partition, sort or group rows by `(tile_x, tile_y)`
5. write one row group per partition-local tile group, or split large groups into shards
6. allow the same `(level, tile_x, tile_y)` to appear in multiple row groups when that tile has rows in multiple input partitions
7. collect one manifest row per written row group

Important constraints:

- use `pyarrow.parquet.ParquetWriter`
- row-group sharding must respect `max_rows_per_row_group`
- all rows in a written row group must belong to exactly one `(level, tile_x, tile_y)`
- Phase 1A should not require a global Dask shuffle by tile before writing

Recommended simplification for Phase 1A:

- correctness first
- write partition-local tile shards first
- accept that one tile may produce multiple manifest rows across input partitions
- optionally compact or merge row groups by tile in a later optimization pass if benchmarks show it is needed

### 8. Coarser Sampled Level Writer

Once the finest unsampled level works, add coarse levels.

For each level from `finest_level - 1` down to `0`:

1. derive tile membership from the canonical source dataframe, not from already sampled parent levels
2. subdivide each coarse tile into a fixed micro-grid
3. allocate `coarse_tile_budget` across occupied micro-grid cells
4. choose deterministic representative points within each occupied cell
5. annotate sampled rows with tile-local coordinates for that level
6. write row groups and manifest rows

The micro-grid is used to make overview sampling spatially even.
Without it, a deterministic whole-tile sample can be dominated by the densest hotspot in a coarse tile and may erase sparse spatial regions.
The micro-grid splits each coarse tile into small bins, gives occupied bins some representation when the budget allows, and then gives dense bins more of the remaining budget.

`cell_id` is the stable identifier for one of those micro-grid bins.
For an `8 x 8` grid, one simple implementation is:

```text
cell_x = floor((x_rel / tile_size) * 8)
cell_y = floor((y_rel / tile_size) * 8)
cell_id = cell_y * 8 + cell_x
```

Clamp `cell_x` and `cell_y` to `0..7` as a defensive guard against floating-point edge cases.
Sampling then happens within each `cell_id`, which keeps sparse occupied areas visible while still allowing dense areas to contribute more representatives.

Example quota allocation for one coarse tile:

```text
coarse_tile_budget = 10

cell_id  occupancy
0        900
1        80
2        20
3        1
```

First give each occupied cell quota `1`, using `4` of the `10` slots.
The remaining `6` slots are distributed in proportion to occupancy.
The proportional extra quotas are approximately `5.39`, `0.48`, `0.12`, and `0.006`.
Take the integer floors first, then give the one leftover slot to the largest fractional remainder.
The final quotas are:

```text
cell_id  quota
0        6
1        2
2        1
3        1
```

If two cells have the same fractional remainder, break ties deterministically by `cell_id`.

Recommended first implementation choices:

- keep the micro-grid size as a private constant and start with `8 x 8`
- compute a deterministic `cell_id` from the per-tile micro-grid coordinates
- if occupied cells `<= coarse_tile_budget`, assign quota `1` to each occupied cell, then distribute the remaining quota approximately by occupancy using a largest-remainder rule with stable tie-break on `cell_id`
- if occupied cells `> coarse_tile_budget`, assign quota `1` only to the `coarse_tile_budget` occupied cells with highest occupancy, with stable tie-break on `cell_id`
- if `transcript_id` is provided, compute a stable per-row ordering key from a canonical encoding of the validated `transcript_id`
- otherwise, compute a per-row ordering key from the internal unique row id created for this build
- internal row ids must be unique within one build, but they only support deterministic coarse-level sampling across rebuilds when input dataframe row order and partitioning are stable
- never use Python's built-in `hash()` for sampling
- after selecting rows from each cell, sort the final sampled rows deterministically before writing so rebuilds do not depend on Dask partition order
- cap each sampled coarse tile at `coarse_tile_budget`

Keep Phase 1A sampling spatial-only. Do not add gene-aware overview sampling yet.

Recommended concrete deterministic algorithm for one coarse tile:

1. compute the micro-grid cell coordinates for every candidate row in the tile
2. compute `cell_id` from those cell coordinates
3. count rows per occupied cell
4. assign cell quotas with the rules above
5. compute one per-row ordering key from validated `transcript_id` or the internal build row id
6. within each occupied cell, sort rows by that ordering key and keep the first `quota[cell]`
7. concatenate selected rows from all occupied cells
8. sort the selected rows deterministically before writing

When `transcript_id` is provided and validated, repeating the build with the same inputs must produce the same sampled rows even if Dask partition order changes.
When `transcript_id` is not provided, deterministic coarse-level sampling across rebuilds is guaranteed only when input dataframe row order and partitioning are stable.

### 9. `metadata.json` and `manifest.parquet`

Once all levels can be written, add the cache metadata writers.

`metadata.json` should be written once with:

- schema version
- `n_levels`
- `finest_level`
- `x_origin`, `y_origin`
- `x_min`, `x_max`, `y_min`, `y_max`
- `levels`
- `build_parameters`

`x_origin` and `y_origin` are grid origins, not merely aliases for the minimum bounds.
For Phase 1A they are written with the same values as `x_min` and `y_min`, but readers must use the explicit origin fields for tile coordinate reconstruction.

`levels` should be an array of explicit per-level records sorted by ascending `level`.
Each record should contain:

- `level`
- `tile_size`
- `is_exact`

Do not write `level_file` in `metadata.json`; level files follow the fixed layout convention `levels/level_<level>.parquet`.
Readers should treat `metadata.json["levels"]` as the source of truth for per-level tile size and exact/sampled status.
Do not write `leaf_tile_size` as a separate metadata field; it is the `tile_size` of the single level where `is_exact = true`.
The cache-wide `n_levels` and `finest_level` remain useful summary and validation fields, but readers should not need to reconstruct level metadata from them.
If those summary fields disagree with `levels`, readers should reject the cache rather than trying to choose which value wins.

`build_parameters` is provenance for how the cache was produced, not a source of truth for interpreting stored tile contents.
Readers should use `manifest.parquet` for actual stored point counts.
`build_parameters` should include:

- `max_rows_per_row_group`
- `coarse_tile_budget`
- `x`
- `y`
- `gene`
- `transcript_id`

`max_rows_per_row_group` and `coarse_tile_budget` have different meanings:

- `max_rows_per_row_group` is a physical Parquet layout setting. It limits how many point rows may be written into one row group before the writer creates another row group shard.
- `coarse_tile_budget` is a build-time overview sampling setting. It caps how many representative point rows may be stored for one sampled coarse tile.

For example, if `max_rows_per_row_group = 50_000` and one exact finest-level tile contains `120_000` points, that tile may produce three manifest rows:

```text
level  tile_x  tile_y  row_group  tile_shard  n_points
2      3       1       0          0           50000
2      3       1       1          1           50000
2      3       1       2          2           20000
```

If `coarse_tile_budget = 50_000` and one sampled coarse tile contains `3_000_000` source points, the writer stores at most `50_000` representative points for that coarse tile before row-group sharding is applied.
The actual stored count is recorded in `manifest.parquet` as `n_points` per row group; if a tile has multiple row groups, sum their `n_points` values to get the tile total.
Exact/sampled status is not repeated in the manifest; readers derive it from `metadata.json["levels"]`.

Readers must validate `metadata.json` before using the cache.
If validation fails, treat the cache as invalid and do not attempt partial recovery.

For Phase 1A, reject metadata when:

- `schema_version` is missing or unsupported
- `levels` is missing, empty, unsorted, or contains duplicate level ids
- `n_levels != len(levels)`
- `finest_level` is not present in `levels`
- level ids are not exactly `0..finest_level`
- there is not exactly one `is_exact = true` level
- the exact level is not `finest_level`
- any `tile_size <= 0`
- required bounds or origins are missing or non-finite
- `x_min > x_max` or `y_min > y_max`

For schema version `harpy-transcripts-vis-0.1`, the writer should validate before finalization that:

- `len(levels) == n_levels`
- level ids are exactly `0..finest_level`
- exactly one level has `is_exact = true`, and it is `finest_level`
- the default Phase 1A construction formula produced the recorded `tile_size` values

`manifest.parquet` should be written from the collected row-group metadata and include:

- `schema_version`
- `level`
- `tile_id`
- `tile_x`
- `tile_y`
- `n_points`
- `row_group`
- `tile_shard`

Do not write `level_file` in the manifest.
Readers derive the level file from the manifest row's `level` using `levels/level_<level>.parquet`.
Do not write `is_exact` in the manifest.
Readers derive exact/sampled status from `metadata.json["levels"]`.

For Phase 1A, `tile_shard` should be written even when a tile only has one shard.
With partition-local writing, multiple row groups may have the same `level`, `tile_x`, and `tile_y`; `tile_shard` gives those row groups a deterministic per-tile shard index.

Write `manifest.parquet` last.

### 10. Public Points Element Builder

Wire the public `build_transcript_visualization_cache_for_points_element(...)` entry point around the internal writer helpers.

Responsibilities:

- call `_validate_points_element(...)`
- call `_validate_cache_build_parameters(...)`
- use the resolved points dataframe as-is
- build all level files, `genes.parquet`, `metadata.json`, and `manifest.parquet`
- return the final `TranscriptTileCache`

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
- null `transcript_id` values when requested
- duplicate `transcript_id` values when requested
- invalid `leaf_tile_size`
- invalid `max_rows_per_row_group`
- invalid `coarse_tile_budget`
- invalid `n_levels`
- zero-extent data computes `n_levels = 1`
- tile-boundary coordinates follow the documented half-open grid convention
- unbacked SpatialData rejection

### Group B: Cache Layout and Metadata

- expected directory layout is created
- `metadata.json` has the expected bounds, origins, and explicit `levels` records
- `genes.parquet` has stable deterministic ids and counts
- `manifest.parquet` exists only after successful build

### Group C: Finest Unsampled Level Correctness

- finest-level coordinates reconstruct the source coordinates within `float32` tolerance
- `gene_id` values match `genes.parquet`
- dense tiles split into multiple row groups when `max_rows_per_row_group` is small
- partition-local writing can produce multiple manifest rows for the same tile
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
7. public points-element builder wiring
8. test completion
9. benchmark pass on synthetic and real data

Each checkpoint should leave the branch in a runnable, testable state.

## Phase 1A Exit Criteria

Phase 1A is complete when:

- the public points-element builder can build the full spatial-first cache from a backed SpatialData points element;
- all core writer tests pass;
- the resulting cache layout matches the roadmap contract;
- coarse sampled levels exist and respect the configured budget;
- the implementation is stable enough that the next phase can build a runtime reader on top of it without revisiting the on-disk format.
