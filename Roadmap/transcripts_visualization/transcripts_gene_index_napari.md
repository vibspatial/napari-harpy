# Transcript Value-Index MVP For napari

This note describes a smaller first working version of transcript visualization in napari.

The existing transcript visualization roadmap is spatial-first: build a tiled multiscale cache so pan and zoom can stay responsive for all transcripts. That is still the right long-term direction for whole-slide transcript rendering.

This document proposes a narrower MVP:

- start from `sdata.points["transcripts"]`;
- validate that the points table has `x`, `y`, and `gene` columns, or another configured index column;
- build a value-first cache optimized for quickly loading selected genes or another configured string/categorical value;
- visualize the selected result in one napari `Points` layer;
- if the selected subset is too large, show a best-effort shuffled preview and warn the user.

The goal is not to solve every zoom level and viewport problem yet. The goal is to make selected-gene transcript visualization feel useful quickly.

## Main Recommendation

Yes, a value index is worth building for this MVP if the primary interaction is:

```text
select one or more genes -> show those transcripts in napari
```

The default index column is `gene`, so the first user-facing workflow can still feel like selected-gene visualization. However, the storage contract should be generic because the same mechanism can index `target`, `feature_name`, `probe`, or another eligible string/categorical column.

The important part is not just the existence of an index file. The cache must also be physically organized by indexed value.

If the original `points.parquet` row groups contain a random mixture of values, a `value -> row groups` lookup will not help much. The reader would still need to scan most row groups and filter in memory.

For selected-value visualization to be snappy, the MVP cache should enforce this invariant:

```text
Each data row group contains rows for exactly one value_id.
```

Large values can span many row groups. Small values can fit in one small row group. The reader can then resolve selected values to a short list of row groups, read only those row groups with `pyarrow`, and update a napari `Points` layer.

This makes selected-value subsetting fast. It does not make arbitrary spatial viewport queries fast. That remains the job of the spatial tiled cache.

## Relationship To The Spatial-First Cache

This MVP should be treated as a separate cache, not as a replacement for `transcripts_vis/`.

Recommended layout:

```text
<sdata.zarr>/
  points/
    <points_name>/
      points.parquet
      transcripts_value_index/
        <index_column_cache_key>/
          metadata.json
          values.parquet
          value_index.parquet
          data/
            shard-00000.parquet
            shard-00001.parquet
            ...
```

`points.parquet` remains the canonical exact table.

`transcripts_value_index/` is a Harpy-owned visualization cache root that can be deleted and rebuilt.

Cache identity is:

```text
points element + index column
```

This means a cache built for `gene` does not overwrite a cache built for `target`. Each index column gets its own cache directory below `transcripts_value_index/`.

`<index_column_cache_key>` is a filesystem-safe key derived from the selected index column. Simple safe column names can be used directly:

```text
gene
target
feature_name
```

If a column name contains spaces, slashes, or other awkward filesystem characters, the cache key should be escaped or slugged. `metadata.json` must store both the original `index_column` and the derived `index_column_cache_key`.

Why separate from `transcripts_vis/`:

- the spatial cache wants rows grouped by tile and level;
- the value-index MVP wants rows grouped by selected value;
- one physical ordering cannot be optimal for both access patterns;
- keeping the caches separate avoids complicating the existing spatial-first implementation while we learn from the MVP.

Later, the two directions can meet in a value-aware tiled cache, such as a `tile_value_index.parquet` version of the idea from the broader roadmap.

## Scope

The MVP supports:

- backed `SpatialData`;
- one points element, initially `sdata.points["transcripts"]`;
- required coordinate columns, default `x="x"` and `y="y"`;
- configurable string/categorical index column, default `gene="gene"`;
- exact visualization when the selected subset is within the render budget;
- sampled visualization when the selected subset exceeds the render budget;
- one napari `Points` layer for the selected subset;
- a minimal UI for choosing the points element, choosing the index-value column, building/rebuilding the cache, selecting cache-backed values, and visualizing the selected points.

The MVP does not support:

- viewport-driven loading;
- multiscale spatial overview levels;
- exact all-transcript rendering for huge datasets;
- coordinate transformation handling beyond the stored points coordinates;
- cell segmentation joins or transcript-to-cell overlays;
- full integration with the current spatial tile cache.

## Input Contract

The builder starts from a backed points element:

```python
points = sdata.points["transcripts"]
```

The selected points element must be a `dask.dataframe.DataFrame`.

Required columns by default:

```text
x
y
gene
```

The first implementation should already allow the user to choose the index column. The default is `gene`, but any eligible value column can be used.

Eligible index columns:

- are not the configured `x` or `y` coordinate columns;
- have string-like values or a categorical dtype;
- can be normalized to non-empty string values.

Examples:

```text
gene
feature_name
target
probe
```

The on-disk cache should use generic `value` terminology. In this document, "value" means one normalized value from the configured index column.

### Terminology

Use generic `value` terminology for storage, internal code, and public API names.

The default index column is still `gene`, and the first user-facing workflow can still be selected-gene visualization. However, a cache built for `index_column="target"` or `index_column="probe"` should not expose gene-specific storage or API names.

Storage schema:

- use `values.parquet`, not `genes.parquet`;
- use `value_index.parquet`, not `gene_index.parquet`;
- use `value_id`, not `gene_id`;
- use `value`, not `gene`, for the normalized string column in `values.parquet`.

Public API names should also be generic:

- `TranscriptValueIndexCache`;
- `TranscriptValueIndexSelection`;
- `build_transcript_value_index_cache_for_points_element`;
- `load_transcripts_for_values`;
- `add_transcript_value_points_layer`;
- `TRANSCRIPT_VALUE_INDEX_SCHEMA_VERSION`.

napari-visible features should use the selected source column name where possible. The reader can work with generic value data internally, but the layer should expose the semantic column selected by the user:

```text
index_column = "gene"   -> features["gene"]
index_column = "target" -> features["target"]
```

The layer may also include `value_id` as an internal/debug feature. It should not expose only a generic `value` feature when a more meaningful selected column name is available.

Widget labels should not hard-code "gene" once arbitrary index columns are supported. Prefer labels based on the selected index column, for example:

```text
index_column = "gene"         -> Search gene values
index_column = "target"       -> Search target values
index_column = "feature_name" -> Search feature_name values
```

### Value Normalization Rules

Index values are normalized before building `values.parquet`.

Normalization is intentionally minimal:

1. Missing values are invalid.
2. Values are converted to strings.
3. Leading and trailing whitespace is stripped.
4. Empty strings after stripping are invalid.
5. Case is preserved.
6. Internal whitespace is preserved.
7. No Unicode normalization, lowercasing, or symbol rewriting is performed in the MVP.

Examples:

```text
" Actb " -> "Actb"
"ACTB"   -> "ACTB"
"Actb"   -> "Actb"
"Act b"  -> "Act b"
```

`ACTB`, `Actb`, and `actb` remain distinct values.

Eligible index column dtypes:

- pandas string dtype;
- object dtype where all non-missing values are string-like;
- categorical dtype whose categories are string-like after normalization.

Object dtype handling:

- object columns are accepted only if every non-missing value can be safely normalized as a string value;
- mixed string/object columns are allowed when all observed non-missing values normalize cleanly;
- numeric, boolean, list, dict, tuple, or other structured Python objects are rejected for the MVP.

Categorical handling:

- categorical values are normalized from their category labels;
- unused categories do not appear in `values.parquet`;
- category order is ignored;
- normalized values are sorted lexicographically before assigning `value_id`.

Bytes handling:

- bytes values are not accepted by default in the MVP;
- if bytes support is added later, bytes should be decoded explicitly as UTF-8 and undecodable values should be rejected;
- do not use Python's default `str(bytes_value)`, because it produces values like `"b'ACTB'"`.

Collision handling:

If multiple source values normalize to the same string, they are treated as the same indexed value.

Examples:

```text
"Actb"
" Actb "
"Actb\t"
```

all normalize to:

```text
"Actb"
```

and therefore produce one row in `values.parquet` with a combined `n_points`.

The cache stores only normalized values. `values.parquet.value` is always the normalized value, not the raw source value.

Validation should reject:

- missing coordinate columns;
- missing index column;
- missing `transcript_id` column when `transcript_id` is configured;
- an index column that is neither string-like nor categorical;
- non-numeric `x` or `y`;
- non-finite `x` or `y`;
- missing index values;
- index values whose string form is empty after stripping whitespace;
- empty dataframes.

Optional input:

```text
transcript_id
```

If present, `transcript_id` can be passed through to the data shards for future lookup of canonical source rows. It is not required for MVP sampling. The MVP does not require `transcript_id` to be non-null or unique.

Before writing the data shards, the builder should globally group rows by the selected index column and shuffle rows within each value group:

```python
def shuffle_value_group(pdf):
    return pdf.sample(frac=1, random_state=42)

shuffled = points.groupby(index_column, group_keys=False).apply(
    shuffle_value_group,
    meta=points._meta,
)
```

The implementation should use the configured `index_column`, not hard-code `gene`.

This global groupby/apply is allowed to be the expensive step in the MVP. The goal is a much simpler cache contract:

```text
group rows by value
shuffle rows within each value group as a best effort
write row groups in that order
```

The fixed random state is a best-effort repeatability aid, not a strict deterministic sampling contract. The MVP does not require stable sampling across rebuilds.

## Cache Lifecycle

Use the same staged replacement pattern as `_transcript_tiles.py`: build in a unique sibling temporary directory, validate required files, move any existing valid cache to a backup, rename the completed build into place, restore the backup on failure, and remove temporary/backup directories after completion.

The builder should never write directly into the final cache directory:

```text
transcripts_value_index/<index_column_cache_key>/
```

Instead, it should build into a temporary sibling directory under the same cache root, for example:

```text
transcripts_value_index/.building-<index_column_cache_key>-<uuid>/
```

The temporary build directory should contain the same final structure:

```text
metadata.json
values.parquet
value_index.parquet
data/
```

Build finalization:

1. Write data shards, `values.parquet`, and `value_index.parquet` into the temporary build directory.
2. Validate the temporary cache internally.
3. Write `metadata.json` last.
4. Replace the final cache directory for that index column with the completed temporary cache.

Replacement should happen as a same-filesystem directory move or swap, not as a file-by-file copy into the final directory. If an existing cache is present, the implementation can move the existing cache to a backup name, move the completed temporary cache into the final path, and then delete the backup. If the final move fails, the implementation should restore the previous cache when possible.

Failed builds should remove their temporary build directory and leave any existing final cache untouched. The final cache directory should be considered valid only when `metadata.json` exists and the cache validator passes.

Rebuilds should be scoped to one cache identity:

```text
points element + index column
```

Rebuilding the `gene` cache must not delete or stale the `target` cache, and rebuilding the `target` cache must not affect the `gene` cache.

Only one build should run for the same cache identity at a time. While a cache build is running, the UI should disable the build/rebuild button for that cache identity and report that the cache is building. Builds for different index columns can be allowed independently later, but the first implementation can serialize all transcript value-index builds if that is simpler.

If a valid cache already exists for the selected points element and index column, clicking rebuild should show a confirmation dialog before overwriting it. If the user cancels, the existing cache should remain unchanged and no temporary build should start.

## Cache Files

### `metadata.json`

One JSON object with cache-level metadata:

```text
schema_version: string
source_points_name: string
source_element_path: string
x: string
y: string
index_column: string
index_column_cache_key: string
transcript_id: string | null
source_n_points: int
n_values: int
target_rows_per_row_group: int
default_render_point_budget: int
shuffle_random_state: int | null
shuffle_policy: string
```

Suggested schema version:

```text
harpy-transcripts-value-index-0.1
```

Write `metadata.json` last, or write a separate completion marker last. The cache should not look valid until all required files have been written.

MVP shuffle metadata:

```text
shuffle_policy: "groupby-value-apply-sample-v1"
shuffle_random_state: 42
```

`shuffle_random_state` may be `null` if no fixed random state was used.

### `values.parquet`

One row per indexed value:

```text
value_id: uint32
value: string
n_points: uint64
```

Sort values lexicographically for deterministic `value_id` assignment in the first implementation.

The reader uses this file to:

- resolve selected values to `value_id`;
- reject selected values that are not present in the cache vocabulary;
- estimate the selected point count before reading data;
- decide whether the selection should be exact or sampled.

### `value_index.parquet`

One row per data row group:

```text
value_id: uint32
data_file: string
row_group: int32
value_shard: int32
n_points: int64
```

`data_file` should be relative to the cache root, for example:

```text
data/shard-00003.parquet
```

`row_group` is the zero-based physical row group index within `data_file`.

`value_shard` is the zero-based logical shard/read order for a given `value_id` after the builder has grouped rows by value and shuffled rows within the value group. This mirrors `tile_shard` in the spatial cache and keeps the physical Parquet row group index separate from the per-value preview order.

`row_group` and `value_shard` answer different questions:

```text
data_file + row_group = where this chunk lives physically
value_id + value_shard = where this chunk sits in the read/preview order for that value
```

`row_group` is local to one Parquet file, so many files can each have `row_group = 0`. `value_shard` gives the reader one continuous order across all files for a selected value. Exact reads can read all shards for a value, while sampled preview reads can read `value_shard = 0, 1, 2, ...` until enough points are loaded.

This makes preview sampling simple:

```text
read the first k value_shards for this value
```

Assign `value_shard` as the zero-based shard number within each `value_id` in write order.

Required invariant:

```text
Each `value_index.parquet` row points to a Parquet row group containing exactly one value_id.
```

Deferred optional columns:

```text
x_min: float64
x_max: float64
y_min: float64
y_max: float64
```

Bounding box columns are not needed for the first static selected-value display, so they are not part of the required MVP schema. They can be added later because they are useful for:

- future "zoom to selected values";
- rough spatial extent display;
- debugging cache correctness;
- possible later hybrid spatial filtering.

### `data/shard-*.parquet`

The data files store the displayable transcript rows.

Required columns:

```text
x: float32
y: float32
value_id: uint32
```

Optional columns:

```text
transcript_id: source dtype
```

If `transcript_id` was provided, store `transcript_id` using the source column dtype when possible.

Store display coordinates as `float32`. This is good enough for napari transcript visualization and reduces memory pressure in both the cache and the `Points` layer. The canonical full-precision values remain in `points.parquet`. If exact coordinate preservation is needed for picked transcripts later, use `transcript_id` to look up canonical rows.

Rows inside each value should be physically ordered by the groupby/apply shuffle output.

This matters because it lets the reader build a preview without reading the full value:

```text
rows are grouped by value_id
within each value_id, rows are best-effort shuffled
first row groups for a value are therefore a best-effort shuffled preview
```

The MVP should use a fixed `random_state` for best-effort repeatability, but it does not promise exact reproducibility across Dask versions, partitioning changes, or rebuilds.

## Cache Availability And Staleness Checks

The viewer should not allow transcript visualization directly from `sdata.points[points_name]`.

Instead, the transcript UI should follow this flow:

1. The user selects the points element and index-value column.
2. The user clicks `Create cache` or `Rebuild cache`.
3. Harpy builds the corresponding cache under `transcripts_value_index/<index_column_cache_key>/`, including `values.parquet`.
4. The value search box is enabled only when a valid cache is available.
5. The value search box reads available values from `values.parquet`, not from the source points dataframe.

This avoids offering values that are not present in the cache and avoids expensive source-data scans during interactive use.

For the MVP, use cheap validation checks only. Do not compute a source Parquet footer digest, full content hash, or fresh source `value_counts` when opening the viewer.

Required cache checks before enabling visualization:

```text
cache exists
metadata.json exists
values.parquet exists
value_index.parquet exists
metadata schema_version matches
metadata shuffle_policy matches supported MVP value
metadata source_element_path matches selected points element
metadata x/y/index column names and index_column_cache_key match current UI selection
metadata source_n_points == sum(values.n_points)
metadata source_n_points == sum(value_index.n_points)
metadata n_values == len(values.parquet)
```

If any check fails, mark the cache as missing or stale, disable the value search box and visualization button, and ask the user to build or rebuild the cache.

These checks mostly validate that the selected points element and UI column choices match the cache, and that the cache is internally consistent. They are not a cryptographic guarantee that the source points table has not changed. That stronger source fingerprint can be added later if stale-cache bugs become common, but it is intentionally out of scope for the MVP.

## Row Group Size

Use this default for the value-index MVP:

```text
target_rows_per_row_group = 25_000
```

This is intentionally smaller than a bulk-analytics Parquet row group. The cache is for interactive selected-value display, where read granularity matters more than maximum compression.

Recommended tuning range:

```text
10_000 rows: more responsive previews and less sample overshoot
25_000 rows: default MVP choice
50_000 rows: fewer row groups and better compression for very large datasets
```

The default pairs well with:

```text
render_point_budget = 100_000
```

With those values, a large selected value needs about four row groups to reach the default preview size, while rare values still usually fit in one row group.

The row group size is a target, not a reason to mix values. The stronger invariant is:

```text
Never mix values inside a row group in the MVP.
```

Consequences:

- a value with fewer than `25_000` points gets one smaller row group;
- a value with more than `25_000` points is split across multiple row groups;
- a row group should not be padded with rows from another value;
- the final row group for a large value may contain fewer than `25_000` rows.

This is acceptable for the standalone value-index MVP. With `1_000_000_000` points, at most about `25_000` values, and `target_rows_per_row_group = 25_000`, the rough row-group count is bounded by:

```text
1_000_000_000 / 25_000 + 25_000 = about 65_000 row groups
```

Those are row groups inside Parquet shard files, not separate files. That scale is reasonable for the MVP and keeps rare-value reads exact and simple.

## Future Spatial Plus Value Layout

Do not carry the strict single-value row-group rule directly into a future spatial plus value cache.

A future `(tile, value)` layout should avoid creating one tiny row group for every non-empty `(tile, value_id)` pair. That could produce millions of small row groups once spatial tiling is introduced.

Recommended future layout:

```text
write tile-major data
within each tile, sort rows by value_id
within each value_id, sort by sampling key or spatial key
write row groups up to a target size
allow multiple small values in one row group
keep each value's rows contiguous within a row group
split one value across row groups only when that (tile, value_id) group exceeds the target size
```

The future `tile_value_index.parquet` would then point to ranges, not necessarily whole single-value row groups:

```text
level
tile_x
tile_y
value_id
data_file
row_group
row_offset
n_points
```

For a rare value, the reader may load one row group that also contains neighboring rare values, then slice or filter in memory. The false-positive read is bounded by the row-group size. That tradeoff is better than exploding the number of row groups.

So the storage policy is:

```text
value-only MVP: keep row groups single-value
future spatial plus value cache: use tile-major, value-contiguous indexed ranges
```

## Build Algorithm

Recommended public entry point:

```python
def build_transcript_value_index_cache_for_points_element(
    sdata: SpatialData,
    points_name: str = "transcripts",
    *,
    output_path: str | PathLike[str] | None = None,
    x: str = "x",
    y: str = "y",
    index_column: str = "gene",
    transcript_id: str | None = None,
    target_rows_per_row_group: int = 25_000,
    shuffle_random_state: int | None = 42,
    default_render_point_budget: int = 100_000,
) -> TranscriptValueIndexCache:
    ...
```

Implementation steps:

1. Validate the backed `SpatialData` object and resolve the points element path with `sdata.locate_element(...)`.
2. Validate the points dataframe schema and data quality.
3. Normalize selected index values by stripping whitespace and converting valid values to strings.
4. Build `values.parquet` using a Dask `value_counts`.
5. Assign deterministic `value_id` values from the sorted value table.
6. Create a working dataframe with `x`, `y`, `value_id`, and optional `transcript_id`.
7. Globally group by `value_id` and shuffle rows within each value group using `shuffle_random_state`.
8. Write Parquet row groups so each row group contains only one `value_id`.
9. Split very large values across multiple row groups of at most `target_rows_per_row_group`.
10. Assign `value_shard` as the shard order within each value.
11. Build `value_index.parquet` from the final row-group metadata.
12. Finalize through staged replacement so incomplete caches are not exposed as valid.

For the first implementation, it is acceptable for one value to appear in multiple shard files as long as every row group is single-value and every row group is listed in `value_index.parquet`.

The MVP deliberately accepts the cost of the global groupby/apply shuffle because it makes the cache contract much simpler and avoids row-group explosion from partition-local value writing. The implementation should avoid calling `.compute()` on the full shuffled dataframe when possible; the cache writer should keep the shuffled result as a Dask dataframe and write from Dask tasks.

The shuffled order is best effort. It is good enough to make the first row groups for a large value useful as a preview, but it is not a strict reproducibility guarantee.

## Runtime Reader

Recommended reader entry point:

```python
def load_transcripts_for_values(
    cache_path: str | PathLike[str],
    values: Sequence[str] | Literal["all"],
    *,
    render_point_budget: int = 100_000,
    columns: Sequence[str] = ("x", "y", "value_id"),
) -> TranscriptValueIndexSelection:
    ...
```

The reader should use `pyarrow`, not Dask, in the interactive path.

Runtime flow:

1. Load `metadata.json`, `values.parquet`, and `value_index.parquet`.
2. Resolve selected values to `value_id`.
3. Sum `n_points` for the selected values before reading data.
4. If the selected count is `<= render_point_budget`, read all row groups for those values.
5. If the selected count is `> render_point_budget`, force a best-effort shuffled preview capped at `render_point_budget`.
6. Return coordinates and features for one napari `Points` layer.

Selected values should already come from `values.parquet`. If a selected value is not present in `values.parquet`, the reader should raise an error instead of warning and skipping it. That means the UI or controller allowed stale or invalid selection state to reach the reader, which is an internal consistency bug.

The UI should make this rare by only allowing selections resolved from the cache-backed value search box. If the error still happens, surface it as a cache/selection consistency problem and ask the user to refresh the selection or rebuild the cache.

## Reader Return Object

Use a generic value-index return object:

```python
@dataclass(frozen=True, kw_only=True)
class TranscriptValueIndexSelection:
    coordinates: np.ndarray
    features: pd.DataFrame
    selected_values: tuple[str, ...]
    selected_value_ids: tuple[int, ...]
    total_count: int
    loaded_count: int
    render_point_budget: int
    is_sampled: bool
    warning: str | None
```

`coordinates` is an `Nx2` `float32` array in napari display order:

```text
y, x
```

The cache stores source coordinate columns as `x` and `y`, but the reader returns coordinates already ordered for napari `Points`.

`features` is a `pandas.DataFrame` with exactly `loaded_count` rows. Its row order must match `coordinates`. It must include:

- the configured index column using its source column name, for example `gene`, `target`, or `probe`;
- `value_id`.

The configured index-column feature contains the normalized string value for each point, populated by mapping `value_id` through `values.parquet`. For example, when `index_column == "gene"`, `features["gene"]` contains gene names such as `"MALAT1"`; `value_id` remains an internal/cache feature.

If the cache contains `transcript_id`, the reader should include it in `features` as well.

`selected_values` contains the normalized unique selected values that were resolved against `values.parquet`. Duplicate requested values are removed before quota allocation and before reading. For `values="all"`, `selected_values` contains all values in `value_id` order.

`selected_value_ids` contains the resolved `value_id` values in the same order as `selected_values`.

`total_count` is the sum of `n_points` for all resolved selected values before render-budget sampling.

`loaded_count` is the number of points returned. It must equal both `len(coordinates)` and `len(features)`. It is always `<= total_count`; when `is_sampled` is true, it is also `<= render_point_budget`.

`is_sampled` is true when `total_count > render_point_budget`, and false when exact selected points are returned.

`warning` is `None` for exact reads. For sampled reads, it should contain user-visible text such as:

```text
Showing 100,000 of 2,431,912 selected points.
```

Use small custom exception types for the reader:

```python
class TranscriptValueIndexError(Exception):
    ...


class TranscriptValueIndexInvalidCacheError(TranscriptValueIndexError):
    ...


class TranscriptValueIndexInvalidSelectionError(TranscriptValueIndexError, ValueError):
    ...


class TranscriptValueIndexUnknownValueError(TranscriptValueIndexInvalidSelectionError):
    ...


class TranscriptValueIndexReadError(TranscriptValueIndexError):
    ...
```

Expected error behavior:

- unknown selected value: raise `TranscriptValueIndexUnknownValueError`;
- missing or invalid metadata, missing required files, schema mismatch, or count mismatch: raise `TranscriptValueIndexInvalidCacheError`;
- invalid reader arguments, such as `render_point_budget <= 0`: raise `TranscriptValueIndexInvalidSelectionError`;
- Parquet IO or read failure: raise `TranscriptValueIndexReadError`.

An empty selected value list should return an empty exact selection:

```text
coordinates.shape == (0, 2)
features has 0 rows
total_count == 0
loaded_count == 0
is_sampled == false
warning is None
```

## Render Budget And Sampling Policy

The default runtime render budget should start at:

```text
render_point_budget = 100_000
```

`render_point_budget` is the maximum number of points returned to napari for display. It is a hard runtime UI safety limit, not a physical Parquet read-size limit.

If the selected values contain at most `render_point_budget` points:

```text
show exact selected points
```

If the selected values contain more than `render_point_budget` points:

```text
force a best-effort shuffled preview and warn the user
```

The MVP should not expose `sample=False` or exact-at-any-cost loading. Selections above the render budget are always sampled before being returned to napari.

The warning should include:

```text
Showing 100,000 of 2,431,912 selected points.
```

Recommended sampling policy:

The MVP default is proportional sampling by point count, with a minimum of one point per selected value when possible. This preserves the visual meaning of density while avoiding complete disappearance of selected rare values in sampled previews.

Before quota allocation, resolve selected values to unique `value_id` values. Duplicate selected values are ignored after the first occurrence, and points are never duplicated in the returned preview.

1. Allocate a sample quota per selected value, proportional to its point count.
2. If `number_of_selected_values <= render_point_budget`, give every selected value at least one point when possible, then distribute the remaining budget proportionally.
3. If `number_of_selected_values > render_point_budget`, do not guarantee one point per selected value. Allocate quotas proportionally by `n_points`; values with quota `0` are omitted from the preview.
4. Use deterministic quota rounding. Start from floor quotas, then distribute remaining points by largest fractional remainder. Break ties by `value_id` ascending.
5. For each value with quota `> 0`, read the first row groups in `value_shard` order until at least the quota is available.
6. If the loaded rows exceed the quota, keep the first quota rows in physical cache order.
7. Concatenate the sampled rows across values.

Balanced sampling and user-selectable sampling modes are deferred until the UI needs an explicit comparison mode. They are useful for comparing spatial patterns across values, but they intentionally distort abundance and should not be the default.

Because rows within each value are shuffled before writing, reading the first row groups gives a best-effort preview without scanning the full selected subset.

This is not a strict reproducibility guarantee. The preview quality depends on the groupby/apply shuffle and the selected random state.

This is important. If we simply read every selected row and then sample, large value selections will still be slow.

MVP read amplification tradeoff:

The sampled output is capped by `render_point_budget`, but the number of rows read from Parquet is not strictly capped. The reader loads whole Parquet row groups. If many selected values each have a tiny sample quota, reading the first row group for each value can load many more rows than the final displayed preview.

This is accepted for the MVP to keep the cache layout simple. If this becomes a practical problem, add a small-preview-shard optimization:

```text
preview_rows_per_value = 512
target_rows_per_row_group = 25_000
```

For each value, write one or more initial small `value_shard`s before the normal large row groups. Sampled reads would consume those small preview shards first, while exact reads would still read all shards. This reduces read amplification for large multi-value selections such as `values="all"` without adding a separate preview cache.

## All-Values Selection

Selecting all values can be allowed.

It should use the same count and sampling policy:

```text
values = "all"
total_count = sum(values_table.n_points)
if total_count <= render_point_budget:
    show exact
else:
    show sampled preview and warn
```

For all-values sampling, proportional quotas are the MVP default. They preserve the global abundance distribution.

A possible alternative is a more balanced per-value sample, where rare values get more visibility than proportional sampling would give them. That is useful for exploratory biology, but it changes the visual meaning of density. Start with proportional sampling and make balanced sampling an explicit option later.

## napari Integration

The MVP should create or update one napari `Points` layer.

Layer behavior:

- layer name: `transcripts` or `transcripts: selected values`;
- data: `Nx2` array in napari display order `y, x`;
- features: at least the configured index column and `value_id`, plus `transcript_id` when present in the cache;
- face color: categorical by the configured index column for small selections;
- warning: displayed when the layer is sampled.

Important implementation detail:

```text
Do not create one napari layer per value.
```

A single layer is easier to update, hide, remove, and later connect to a controller.

For many selected values, categorical coloring may become visually noisy. The first version can still attach the configured value feature and use a simple color cycle, then refine the UI later.

## Why This Should Feel Snappy

This MVP avoids the current slow path:

```text
load full dask dataframe -> compute all rows -> filter/subsample
```

Instead, selected-value display becomes:

```text
selected values
-> value_id values
-> row groups listed in value_index.parquet
-> pyarrow read of only those row groups
-> napari Points layer update
```

For sampled large selections:

```text
selected values
-> per-value quotas
-> first few build-time-sample-key-sorted row groups per value
-> pyarrow read of a bounded number of rows
-> napari Points layer update
```

This should be fast for:

- one rare gene;
- a handful of marker genes;
- a moderate gene panel;
- all values as a sampled preview.

It will not be fast for:

- exact display of millions of selected points;
- viewport-exact all-transcript rendering;
- spatial queries such as "only visible transcripts in this rectangle".

Those remain spatial-cache problems.

## Design Tradeoffs

### Benefit

- Much simpler than the multiscale spatial cache.
- Gives an early user-visible transcript workflow.
- Supports exact selected-value display when counts are modest.
- Avoids scanning `points.parquet` for every value switch.
- Gives predictable behavior through a hard render budget.

### Cost

- Requires an offline shuffle by the selected value column.
- Duplicates a subset of the canonical transcript data in another cache.
- Does not solve pan/zoom-scaled loading.
- Exact display is still bounded by napari `Points` performance.
- A row-group-per-value policy can create many small row groups for rare values.

The many-small-row-groups concern is acceptable for the standalone value-index MVP. Most transcript datasets have thousands to tens of thousands of values for typical index columns such as `gene`, not millions of values. Rare values should not be packed together in this MVP, because exact rare-value reads are one of the main benefits of the cache. Packing small value groups belongs to a future spatial plus value layout where `(tile, value)` row-group counts could otherwise explode.

## Implementation Slices

These slices split the MVP into implementation units with clear dependencies. The first slices intentionally establish contracts before the widget work begins, because the UI should depend on stable cache, validation, and reader behavior.

Recommended order:

```text
0 -> 1 -> 2 -> 3 -> 5 -> 4 -> 6 -> 7 -> 8 -> 9 -> 10
```

The validator is pulled forward before the full data writer so the viewer can rely on a cache-status contract early.

### Slice 0: Lock Minimal Decisions Before Code

Goal: decide the few contracts that would otherwise cause rework.

Includes:

- cache path identity: one cache per points element versus one cache per points element plus index column;
- generic naming: use `value` internally and on disk while keeping `gene` as the default user-facing index column;
- coordinate order for napari: `y, x`;
- first-pass sampling behavior for selections above `render_point_budget`;
- basic cache lifecycle: temporary build path, final cache path, rebuild replacement.

Done when:

- the implementation has enough fixed vocabulary and paths to avoid renaming churn later.

### Slice 1: Core Module Skeleton And Data Contracts

Goal: create the standalone transcript value-index module without building the full cache yet.

Includes:

- new module, likely `src/napari_harpy/_transcript_value_index.py`;
- schema version constant;
- dataclasses or typed return objects:
  - `TranscriptValueIndexCache`;
  - `TranscriptValueIndexSelection`;
- cache path helpers;
- metadata read/write helpers;
- custom errors for invalid source data, stale cache, invalid selection, and cache read failures.

Tests:

- metadata round trip;
- cache path resolution;
- basic object construction;
- error types.

Done when:

- later slices can depend on stable public objects and helper functions.

### Slice 2: Source Points Validation And Value Normalization

Goal: validate whether a selected points element can produce a cache.

Includes:

- resolve `sdata.points[points_name]`;
- require backed `SpatialData`;
- require Dask dataframe points element;
- validate coordinate columns;
- validate configured index column;
- normalize index values;
- reject missing, empty, invalid, or unsupported values;
- validate optional `transcript_id` exists when requested.

Tests:

- missing `x`, `y`, index column;
- non-numeric coordinates;
- non-finite coordinates;
- empty dataframe;
- invalid index dtype;
- missing or blank index values;
- categorical, string, and object value handling.

Done when:

- the builder can fail early with clear errors before doing expensive work.

### Slice 3: Vocabulary Cache

Goal: build the first useful cache artifact: the selectable value vocabulary.

Includes:

- compute normalized value counts with Dask;
- sort values deterministically;
- assign stable `value_id`;
- write `values.parquet`;
- write `metadata.json` last;
- store source points name and path, selected columns, transcript id column, counts, row-group target, and sampling policy.

Tests:

- deterministic value id assignment;
- correct point counts;
- metadata consistency;
- `metadata.json` written only after required cache files;
- selected values come from `values.parquet`, not the source dataframe.

Done when:

- a cache can expose searchable values, even before point loading exists.

### Slice 4: Full Cache Builder With Data Shards

Goal: write displayable transcript rows grouped by value.

Includes:

- create a working dataframe with:
  - `x`;
  - `y`;
  - value id;
  - optional `transcript_id`;
- globally group by value and shuffle rows within each value group;
- convert display coordinates to `float32`;
- write `data/shard-*.parquet`;
- enforce single-value row groups;
- split large values across row groups;
- assign `value_shard` as the shard order within each value;
- write `value_index.parquet`;
- finalize through staged replacement.

Tests:

- required data columns exist;
- coordinates are `float32`;
- every indexed row group contains exactly one value id;
- row groups do not exceed target size except where explicitly allowed;
- all row groups are listed in `value_index.parquet`;
- rows are grouped by value and shuffled within each value group before writing;
- `value_shard` follows row-group write order within each value;
- `sum(value_index.n_points) == source_n_points`.

Done when:

- the full on-disk cache can be built and inspected without napari.

### Slice 5: Cache Validator And Staleness Checks

Goal: decide whether an existing cache is usable for the current UI selection.

Includes:

- check required files;
- check schema version;
- check selected points element;
- check selected `x`, `y`, and index column;
- check count consistency:
  - metadata versus `values.parquet`;
  - metadata versus `value_index.parquet`;
  - metadata `n_values` versus `values.parquet`;
- return structured status instead of only raising.

Tests:

- valid cache passes;
- missing files fail;
- wrong schema fails;
- wrong selected column fails;
- inconsistent counts fail;
- stale cache disables visualization state.

Done when:

- the viewer can safely enable or disable value selection based on cache state.

### Slice 6: Exact Runtime Reader

Goal: load selected values within the render budget using PyArrow only.

Includes:

- load metadata, `values.parquet`, and `value_index.parquet`;
- resolve selected values to ids;
- reject unknown selected values;
- compute total selected point count before reading data;
- read only listed row groups;
- return a `TranscriptValueIndexSelection`;
- include coordinates, features, selected values, selected value ids, loaded count, total count, render budget, sampled flag, and warning text.

Tests:

- one selected value;
- multiple selected values;
- unknown value raises;
- exact load returns all selected rows;
- reader does not use Dask;
- coordinates are returned as `float32` in napari `y, x` order;
- features include the configured index column and `value_id`;
- empty selections return an empty exact result.

Done when:

- selected-value loading works without napari.

### Slice 7: Sampled Runtime Reader

Goal: load bounded best-effort shuffled previews when selected values exceed `render_point_budget`.

Includes:

- proportional quota allocation;
- minimum one point per selected value when possible;
- deterministic quota rounding;
- proportional omission of values when `number_of_selected_values > render_point_budget`;
- deterministic tie-breaking by `value_id` ascending;
- read first required row groups in `value_shard` order;
- trim excess loaded rows by keeping the first quota rows in physical cache order;
- support `values="all"`;
- produce warning text like:
  - `Showing 100,000 of 2,431,912 selected points.`;

Tests:

- sampled result has at most `render_point_budget`;
- sampled reads use the cache's shuffled row-group order;
- rare selected values are preserved when possible;
- all-values selection works;
- duplicate selected values are handled predictably;
- when `render_point_budget < number_of_selected_values`, some selected values may receive quota `0` and be omitted from the preview.

Done when:

- large selections return fast, bounded, best-effort shuffled previews.

### Slice 8: napari Points Layer Integration

Goal: convert a reader result into one napari `Points` layer.

Includes:

- helper such as `add_transcript_value_points_layer`;
- create or update one existing layer;
- use the reader's `y, x` coordinates directly;
- attach the configured index column and `value_id`, plus `transcript_id` when present;
- apply categorical coloring for small selections;
- set point size, opacity, name, and metadata;
- surface sampled warning.

Tests:

- creates layer when missing;
- updates same layer on repeated calls;
- does not create one layer per value;
- features are attached;
- sampled warning metadata or status is present.

Done when:

- code can display exact or sampled cache-backed transcripts in napari through a thin helper.

### Slice 9: Viewer UI Integration

Goal: expose the workflow in the existing Harpy viewer widget.

Includes:

- points element selector;
- index column selector;
- widget labels based on the selected index column rather than hard-coded "gene" wording;
- create or rebuild cache button;
- confirmation dialog before rebuilding an existing valid cache;
- cache status display;
- value search and select control backed by `values.parquet`;
- visualize selected values button;
- all-values option;
- disable the build/rebuild button while a cache build is running;
- disable visualization controls when cache is missing, stale, building, or loading;
- run build and read without freezing the UI where practical.

Tests:

- widget initializes with no cache;
- valid cache enables value search;
- stale cache disables visualization;
- create or rebuild triggers builder;
- selected values trigger reader and layer update;
- sampled warning is visible.

Done when:

- the user-facing MVP workflow exists end to end.

### Slice 10: Integration Tests And Hardening

Goal: make the whole path reliable enough to iterate on real data.

Includes:

- fixture `SpatialData` with transcript points;
- full build, validate, exact read, sampled read, and layer update flow;
- failure cases for corrupt or partial cache;
- rebuild behavior;
- basic performance sanity checks on moderate synthetic data;
- documentation updates or roadmap status notes.

Done when:

- the MVP has a tested end-to-end path and clear known limitations.

### Deferred Follow-Up: Small Preview Shards

Goal: reduce sampled-read amplification if large multi-value previews are too slow.

Problem:

- sampled output is capped by `render_point_budget`;
- Parquet reads happen at row-group granularity;
- when many selected values each need only a few points, reading one normal row group per value can load far more rows than are displayed.

Possible solution:

- add `preview_rows_per_value`, for example `512`;
- for each value, write initial small `value_shard`s before normal `target_rows_per_row_group` shards;
- make sampled reads consume preview shards first;
- keep exact reads unchanged by reading all shards for the selected values.

Status:

- deferred until benchmarks show read amplification is a real problem.

## Deliverables

The MVP should produce these implementation pieces:

1. Cache builder: builds `transcripts_value_index/<index_column_cache_key>/` from a backed points element and a selected string/categorical index column.
2. Cache validator: checks required files, schema version, selected source element, selected columns, and internal count consistency before visualization is enabled.
3. Cache reader: loads exact or sampled selected values from `values.parquet`, `value_index.parquet`, and the data shard files using PyArrow.
4. Viewer UI controls: lets the user choose the points element and index-value column, create or rebuild the cache, search/select values from `values.parquet`, and request visualization.
5. napari layer integration: creates or updates one `Points` layer for the selected values, with sampled-state warning text when applicable.
6. Tests: cover input validation, cache layout, staleness checks, value resolution, exact reads, sampled reads, and row-group invariants.

## Suggested Module Shape

Keep this separate from `_transcript_tiles.py` at first:

```text
src/napari_harpy/_transcript_value_index.py
```

Suggested objects:

```text
TRANSCRIPT_VALUE_INDEX_SCHEMA_VERSION
TranscriptValueIndexCache
TranscriptValueIndexSelection
TranscriptValueIndexError
TranscriptValueIndexInvalidCacheError
TranscriptValueIndexInvalidSelectionError
TranscriptValueIndexUnknownValueError
TranscriptValueIndexReadError
build_transcript_value_index_cache_for_points_element
load_transcripts_for_values
add_transcript_value_points_layer
```

The build path can use Dask.

The read path should use PyArrow.

The napari path should be thin and should not know how the cache is built internally.

## Minimal Acceptance Criteria

A first implementation is successful when:

- it can build `transcripts_value_index/<index_column_cache_key>/` from a backed `sdata.points["transcripts"]`;
- it validates `x`, `y`, and the selected index-value column;
- it writes `values.parquet`, `value_index.parquet`, and data Parquet files;
- it validates cache availability and staleness before enabling visualization;
- the value search box reads selectable values from `values.parquet`;
- every data row group listed in `value_index.parquet` contains exactly one `value_id`;
- selecting values within the render budget loads exact points;
- selecting values above the render budget loads at most `render_point_budget` points;
- all-values selection is allowed and sampled when needed;
- sampled results show a user-visible warning;
- the napari integration updates one `Points` layer rather than creating many layers.
