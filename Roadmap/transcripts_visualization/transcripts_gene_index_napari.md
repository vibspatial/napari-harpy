# Transcript Value-Index MVP For napari

This note describes a smaller first working version of transcript visualization in napari.

The existing transcript visualization roadmap is spatial-first: build a tiled multiscale cache so pan and zoom can stay responsive for all transcripts. That is still the right long-term direction for whole-slide transcript rendering.

This document proposes a narrower MVP:

- start from `sdata.points["transcripts"]`;
- validate that the points table has `x`, `y`, and `gene` columns, or another configured index column;
- first implement selected-value visualization directly from the Dask points dataframe, without requiring a cache;
- visualize the selected result in one napari `Points` layer;
- if the selected subset is too large, show a best-effort shuffled preview and warn the user.

The goal is not to solve every zoom level and viewport problem yet. The goal is to make selected-gene transcript visualization feel useful quickly.

## Main Recommendation

Start with the no-cache path.

The primary interaction is still:

```text
select one or more genes -> show those transcripts in napari
```

The default index column is `gene`, so the first user-facing workflow can still feel like selected-gene visualization. However, the implementation should be generic because the same mechanism can select by `target`, `feature_name`, `probe`, or another eligible string/categorical column.

The first implementation should:

1. compute the selected total count first;
2. if the selection is within `render_point_budget`, compute the exact selection;
3. if the selection exceeds `render_point_budget`, filter first, sample/downsample before compute when possible, then final-trim in memory;
4. return the same `TranscriptValueSelection` object that a future cache-backed reader would return.

This keeps the first implementation much simpler and lets us test whether the direct Dask path is already good enough for common selected-gene workflows.

The value-index cache remains the planned acceleration path if direct filtering is not responsive enough on real datasets or remote/backed stores. If we build that cache later, the important invariant remains:

```text
Each data row group contains rows for exactly one value_id.
```

Large values can span many row groups. Small values can fit in one small row group. The reader can then resolve selected values to a short list of row groups, read only those row groups with `pyarrow`, and update a napari `Points` layer.

That cache would make selected-value subsetting faster. It still would not make arbitrary spatial viewport queries fast. That remains the job of the spatial tiled cache.

## Relationship To The Spatial-First Cache

The no-cache MVP does not write any transcript visualization cache. If the optional value-index cache is implemented later, it should be treated as a separate cache, not as a replacement for `transcripts_vis/`.

Deferred value-index cache layout:

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

`transcripts_value_index/` would be a Harpy-owned visualization cache root that can be deleted and rebuilt.

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

Why the deferred cache should stay separate from `transcripts_vis/`:

- the spatial cache wants rows grouped by tile and level;
- the value-index MVP wants rows grouped by selected value;
- one physical ordering cannot be optimal for both access patterns;
- keeping the caches separate avoids complicating the existing spatial-first implementation while we learn from the MVP.

Later, the two directions can meet in a value-aware tiled cache, such as a `tile_value_index.parquet` version of the idea from the broader roadmap.

## Scope

The MVP supports:

- backed and unbacked `SpatialData` for the direct no-cache path;
- one points element, initially `sdata.points["transcripts"]`;
- required coordinate columns, default `x="x"` and `y="y"`;
- configurable string/categorical index column, default `gene="gene"`;
- direct Dask filtering by selected values, without requiring a cache;
- exact visualization when the selected subset is within the render budget;
- sampled visualization when the selected subset exceeds the render budget;
- one napari `Points` layer for the selected subset;
- a minimal UI for choosing the points element, choosing the index-value column, selecting values, setting `render_point_budget`, and visualizing the selected points.

The MVP does not support:

- requiring a prebuilt value-index cache before visualization;
- viewport-driven loading;
- multiscale spatial overview levels;
- exact all-transcript rendering for huge datasets;
- coordinate transformation handling beyond the stored points coordinates;
- cell segmentation joins or transcript-to-cell overlays;
- full integration with the current spatial tile cache.

## Input Contract

The direct reader starts from a points element:

```python
points = sdata.points["transcripts"]
```

The optional cache builder, if implemented later, starts from the same points element but requires backed `SpatialData`.

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

The implementation should use generic `value` terminology. If the optional cache is built later, the on-disk cache should use the same generic terminology. In this document, "value" means one normalized value from the configured index column.

Direct mode should produce an in-memory value table with the same logical schema as future `values.parquet`:

```text
value_id: uint32
value: string
n_points: uint64
```

This table is a compact summary of the selected index column, not the full points dataframe. The UI uses it to populate value search, the direct reader uses it to resolve selected strings to `value_id`, and the returned layer features use it to attach compact `value_id` values. Keeping this schema aligned with future `values.parquet` lets the direct reader and optional cache reader share the same downstream napari layer path.

Represent this table with a small immutable object:

```python
@dataclass(frozen=True, kw_only=True)
class TranscriptValueTable:
    values: pd.DataFrame
    index_column: str
    total_count: int
```

`values` must contain exactly:

```text
value_id
value
n_points
```

`TranscriptValueTable.__post_init__` should validate:

```text
values contains exactly value_id, value, n_points
value_id is unique
value is unique
n_points is non-negative
total_count == sum(values.n_points)
```

Returned selections and napari layer features should include only the MVP feature columns:

```text
<index_column>: pandas categorical
value_id: integer
```

Do not include `transcript_id` in returned features or layer features for the MVP.

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
- `TranscriptValueSelection`;
- `TranscriptValueTable`;
- `build_transcript_value_index_cache_for_points_element`;
- `load_transcripts_for_values_direct`;
- `add_transcript_value_points_layer`;
- `TRANSCRIPT_VALUE_INDEX_SCHEMA_VERSION`.

napari-visible features should use the selected source column name where possible. The reader can work with generic value data internally, but the layer should expose the semantic column selected by the user:

```text
index_column = "gene"   -> features["gene"]
index_column = "target" -> features["target"]
```

The layer should also include `value_id` as a compact internal/debug feature. It should not expose only a generic `value` feature when a more meaningful selected column name is available.

Widget labels should not hard-code "gene" once arbitrary index columns are supported. Prefer labels based on the selected index column, for example:

```text
index_column = "gene"         -> Search gene values
index_column = "target"       -> Search target values
index_column = "feature_name" -> Search feature_name values
```

### Value Normalization Rules

Index values are normalized before direct value selection and before any optional cache build.

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
- unused categories do not appear in the value table or in `values.parquet` if the optional cache is built;
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

and therefore produce one row in the value table, and in `values.parquet` if the optional cache is built, with a combined `n_points`.

The direct reader and optional cache use normalized values as their selection contract. If the cache is built later, `values.parquet.value` is always the normalized value, not the raw source value.

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

If the optional cache is built later, then before writing data shards the builder should globally group rows by the selected index column and shuffle rows within each value group:

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

## Optional Cache Lifecycle

This section applies only if the deferred value-index cache is implemented.

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

## Optional Cache Files

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
- reject selected values that are not present in the cache value table;
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

## Direct Mode And Optional Cache Availability

The viewer should allow transcript visualization directly from `sdata.points[points_name]`.

The direct MVP flow is:

1. The user selects the points element and index-value column.
2. Harpy validates the selected points dataframe and selected index column.
3. Harpy computes or refreshes the selectable value table directly from the source Dask dataframe.
4. The user selects one or more values, or chooses all values.
5. Harpy loads the direct selection, sampling before compute when the selected count exceeds `render_point_budget`.

The optional cache flow can be added later:

1. The user selects the points element and index-value column.
2. The user clicks `Create cache` or `Rebuild cache`.
3. Harpy builds the corresponding cache under `transcripts_value_index/<index_column_cache_key>/`, including `values.parquet`.
4. If a valid cache exists, Harpy may read selectable values from `values.parquet` and load selections through the PyArrow cache reader.
5. If no valid cache exists, direct mode remains available.

Missing or stale caches must not disable direct visualization. They only disable the cache-backed acceleration path.

For the optional cache, use cheap validation checks only. Do not compute a source Parquet footer digest, full content hash, or fresh source `value_counts` when opening the viewer.

Required cache checks before enabling the cache-backed path:

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

If any check fails, mark the cache as missing or stale and use the direct source-data path instead. The UI can still offer `Create cache` or `Rebuild cache` as an optional acceleration action.

These checks mostly validate that the selected points element and UI column choices match the cache, and that the cache is internally consistent. They are not a cryptographic guarantee that the source points table has not changed. That stronger source fingerprint can be added later if stale-cache bugs become common, but it is intentionally out of scope for the direct-first MVP.

## Optional Cache Row Group Size

If the value-index cache is implemented, use this default:

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

If the value-index cache is implemented, do not carry the strict single-value row-group rule directly into a future spatial plus value cache.

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

## Optional Cache Build Algorithm

This section is deferred until the direct no-cache path has been implemented and benchmarked.

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

The optional cache design deliberately accepts the cost of the global groupby/apply shuffle because it makes the cache contract much simpler and avoids row-group explosion from partition-local value writing. The implementation should avoid calling `.compute()` on the full shuffled dataframe when possible; the cache writer should keep the shuffled result as a Dask dataframe and write from Dask tasks.

The shuffled order is best effort. It is good enough to make the first row groups for a large value useful as a preview, but it is not a strict reproducibility guarantee.

## Runtime Reader

Recommended direct no-cache reader entry point:

```python
def load_transcripts_for_values_direct(
    sdata: SpatialData,
    points_name: str,
    values: Sequence[str] | Literal["all"],
    *,
    x: str = "x",
    y: str = "y",
    index_column: str = "gene",
    render_point_budget: int = 100_000,
    random_state: int | None = 42,
) -> TranscriptValueSelection:
    ...
```

The direct reader uses Dask in the interactive path. Its job is to avoid materializing the full selected subset when the selected count exceeds the render budget.

Runtime flow:

1. Validate the selected points element and selected index column.
2. Normalize requested values with the same value-normalization rules used everywhere else.
3. Build a selected Dask dataframe by filtering the configured index column. For `values="all"`, the selected dataframe is the full points dataframe.
4. Compute the selected total count before materializing selected rows.
5. If `total_count <= render_point_budget`, compute the exact selected dataframe with only the needed columns.
6. If `total_count > render_point_budget`, filter first, then call Dask `sample(frac=render_point_budget / total_count, random_state=random_state)` on the selected dataframe before compute.
7. After compute, final-trim in memory to at most `render_point_budget` rows.
8. Return coordinates and features for one napari `Points` layer.

The important direct-mode rule is:

```text
filter selected values first, then sample selected rows
```

Do not sample the full points dataframe before filtering, because that can drop rare selected values before they have a chance to appear.

The direct no-cache reader should return the same `TranscriptValueSelection` object shape as a future cache-backed reader. This keeps the napari layer integration independent of the read backend.

Recommended optional cache-backed reader entry point:

```python
def load_transcripts_for_values_from_cache(
    cache_path: str | PathLike[str],
    values: Sequence[str] | Literal["all"],
    *,
    render_point_budget: int = 100_000,
    columns: Sequence[str] = ("x", "y", "value_id"),
) -> TranscriptValueSelection:
    ...
```

If implemented later, the cache-backed reader should use `pyarrow`, not Dask, in the interactive path.

For direct mode, selected values should come from the direct value table computed from the selected points dataframe. For cache mode, selected values should come from `values.parquet`. If an unknown selected value reaches either reader, raise an error instead of warning and skipping it. This means the UI or controller allowed stale or invalid selection state to reach the reader, which is an internal consistency bug. Surface it as a selection consistency problem and ask the user to refresh the value list.

## Reader Return Object

Use a generic value-selection return object:

```python
@dataclass(frozen=True, kw_only=True)
class TranscriptValueSelection:
    coordinates: np.ndarray
    features: pd.DataFrame
    index_column: str
    selected_values: tuple[str, ...]
    selected_value_ids: tuple[int, ...]
    total_count: int
    render_point_budget: int
    is_sampled: bool
    warning: str | None

    @property
    def loaded_count(self) -> int:
        return len(self.coordinates)
```

`coordinates` is an `Nx2` `float32` array in napari display order:

```text
y, x
```

The source dataframe and optional cache store coordinate columns as `x` and `y`, but the reader returns coordinates already ordered for napari `Points`.

`index_column` is the configured source column used for value selection, for example `gene`, `target`, or `probe`. Layer code should use this field to find the categorical feature column for coloring and status text.

`features` is a `pandas.DataFrame` with exactly `loaded_count` rows. Its row order must match `coordinates`. It must include:

- the configured index column using its source column name, for example `gene`, `target`, or `probe`;
- `value_id`.

The configured index-column feature contains the normalized string value for each point. In direct mode this comes from the filtered source dataframe after normalization; in cache mode this is populated by mapping `value_id` through `values.parquet`. For example, when `index_column == "gene"`, `features["gene"]` contains gene names such as `"MALAT1"`; `value_id` remains an internal/debug feature.

Store the configured index-column feature as a pandas `Categorical` column whose categories are the normalized values from the current value table. This keeps repeated gene, target, or probe labels compact for large render budgets.

Do not include `transcript_id` in `features` for the MVP. If picked-point canonical lookup is needed later, use a separate lookup path rather than carrying transcript identifiers in every rendered feature row.

`selected_values` contains the normalized unique selected values that were resolved against the current value table. Duplicate requested values are removed before sampling and before reading. For `values="all"`, `selected_values` contains all values in `value_id` order.

`selected_value_ids` contains the resolved `value_id` values in the same order as `selected_values`.

`total_count` is the sum of `n_points` for all resolved selected values before render-budget sampling.

`loaded_count` is a derived property, not a stored dataclass field. It is the number of points returned and should be computed from `len(coordinates)`. It must equal `len(features)`. It is always `<= total_count`; when `is_sampled` is true, it is also `<= render_point_budget`.

`is_sampled` is true when `total_count > render_point_budget`, and false when exact selected points are returned.

`warning` is `None` for exact reads. For sampled reads, it should contain user-visible text such as:

```text
Showing 100,000 of 2,431,912 selected points.
```

`TranscriptValueSelection.__post_init__` should validate:

```text
coordinates is an Nx2 array
coordinates dtype is float32
len(features) == len(coordinates) == loaded_count
features contains index_column
features contains value_id
loaded_count <= total_count
loaded_count <= render_point_budget when is_sampled
```

Expected error behavior:

- follow the existing napari-harpy style and use `ValueError` for expected validation and precondition failures;
- invalid source dataframe or source schema: raise `ValueError`;
- unknown selected value: raise `ValueError`;
- invalid reader arguments, such as `render_point_budget <= 0`: raise `ValueError`;
- missing or invalid optional-cache metadata, missing required cache files, schema mismatch, or count mismatch: raise `ValueError`;
- unexpected Dask, Parquet, or layer-update failures can propagate to the controller/worker, which converts them into an error status message.

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

Direct-mode sampling policy:

The direct no-cache MVP uses simple selected-row sampling:

1. Resolve and deduplicate selected values.
2. Filter the Dask dataframe to the selected values.
3. Compute `total_count`.
4. If `total_count <= render_point_budget`, compute the exact selected rows.
5. If `total_count > render_point_budget`, compute:

```python
target_fraction = render_point_budget / total_count
sampled = selected.sample(frac=target_fraction, random_state=42)
```

6. Compute the sampled dataframe.
7. Final-trim in memory to at most `render_point_budget` rows.

This policy is intentionally simple. It does not guarantee one point per selected value, and it does not implement proportional per-value quotas. It is the fastest MVP path to test whether direct Dask filtering gives an acceptable user experience.

The direct-mode rule is to filter before sampling. Calling `points.sample(frac=...)` before filtering selected values is not equivalent and can remove rare selected values before the value filter is applied.

Optional cache sampling policy:

If the cache path is implemented later, it can use proportional sampling by point count, with a minimum of one point per selected value when possible. That policy preserves the visual meaning of density while avoiding complete disappearance of selected rare values in sampled previews.

Before quota allocation, resolve selected values to unique `value_id` values. Duplicate selected values are ignored after the first occurrence, and points are never duplicated in the returned preview.

1. Allocate a sample quota per selected value, proportional to its point count.
2. If `number_of_selected_values <= render_point_budget`, give every selected value at least one point when possible, then distribute the remaining budget proportionally.
3. If `number_of_selected_values > render_point_budget`, do not guarantee one point per selected value. Allocate quotas proportionally by `n_points`; values with quota `0` are omitted from the preview.
4. Use deterministic quota rounding. Start from floor quotas, then distribute remaining points by largest fractional remainder. Break ties by `value_id` ascending.
5. For each value with quota `> 0`, read the first row groups in `value_shard` order until at least the quota is available.
6. If the loaded rows exceed the quota, keep the first quota rows in physical cache order.
7. Concatenate the sampled rows across values.

Balanced sampling and user-selectable sampling modes are deferred until the UI needs an explicit comparison mode. They are useful for comparing spatial patterns across values, but they intentionally distort abundance and should not be the default.

Because rows within each value are shuffled before writing, the optional cache reader can read the first row groups as a best-effort preview without scanning the full selected subset.

This is not a strict reproducibility guarantee. The optional cache preview quality depends on the groupby/apply shuffle and the selected random state.

This is important. If we simply read every selected row and then sample, large value selections will still be slow.

Optional cache read amplification tradeoff:

The sampled output is capped by `render_point_budget`, but the number of rows read from Parquet is not strictly capped. The reader loads whole Parquet row groups. If many selected values each have a tiny sample quota, reading the first row group for each value can load many more rows than the final displayed preview.

This is accepted for the optional cache design to keep the cache layout simple. If this becomes a practical problem, add a small-preview-shard optimization:

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

For all-values sampling, the direct MVP samples rows from the selected dataframe after counting. This approximates the global abundance distribution without explicit per-value quotas.

In cache mode, proportional all-values sampling can be implemented with explicit per-value quotas. A possible alternative is a more balanced per-value sample, where rare values get more visibility than proportional sampling would give them. That is useful for exploratory biology, but it changes the visual meaning of density. Start with the simple direct-mode policy and make balanced sampling an explicit option later.

## napari Integration

The MVP should create or update one napari `Points` layer.

Coordinate behavior:

- the reader returns coordinates as an `Nx2` `float32` array in napari display order `y, x`;
- the napari helper uses `selection.coordinates` directly and must not reorder coordinates again;
- coordinates are in the stored points coordinate system used to build the cache. Coordinate transform handling beyond that remains out of scope for the MVP.

Layer identity and update behavior:

- create or update one Harpy-owned transcript value-selection `Points` layer;
- do not create one layer per value;
- do not create one layer per visualization request;
- changing the selected value set, for example from `gene=MALAT1` to `gene=TEST`, updates the same layer object by replacing its data, features, name, colors, and sampled status;
- ignore unregistered napari layers even if they happen to have the same name.

Use `LayerBindingRegistry` as the authoritative way to find the existing layer, following the existing viewer adapter pattern. Do not rely on napari layer names or metadata as the primary lookup contract.

Add a transcript value-selection points binding to the registry. The exact class name can change, but the binding should capture:

```text
layer: Points
element_type: "points"
points_role: "transcript_value_selection"
sdata_id
points_name
coordinate_system: string | null
index_column
index_column_cache_key
```

The binding identity should not include `selected_values`. This is what lets repeated visualization requests replace the current selection in one layer instead of accumulating stale layers.

Layer naming:

```text
one selected value:
  <points_name>: <index_column>=<value>

multiple selected values:
  <points_name>: <n> <index_column> values

all values:
  <points_name>: all <index_column> values
```

Keep sampled/exact state in the widget status card or controller state, not in the main layer name. Do not use layer metadata as the source of truth for sampled/build/read state.

Layer feature behavior:

- `layer.data`: `selection.coordinates`;
- `layer.features`: `selection.features`;
- features contain the configured index column as a pandas categorical feature, plus `value_id`;
- do not include `transcript_id` in layer features for the MVP.

Point visual defaults:

```text
size: 1.0
opacity: 0.8
symbol: "disc"
edge_width: 0
solid_color: "#00FFFF"
```

Categorical coloring:

- if `len(selection.selected_values) == 1`, use `solid_color`;
- if `2 <= len(selection.selected_values) <= 102`, color categorically by `features[index_column]`;
- use Harpy's existing `default_categorical_colors(n)` helper, which already falls through to Scanpy's `default_102` / `godsnot_102` palette for larger category counts;
- category and palette order should follow `selection.selected_values`, which are ordered by `value_id`;
- if `len(selection.selected_values) > 102`, disable categorical coloring and use `solid_color`;
- base the categorical-coloring threshold on the number of resolved selected values, not only on values that happen to remain visible after sampling.

When categorical coloring is disabled above 102 values, show a widget status-card warning such as:

```text
Selected 5,000 gene values. Categorical coloring is disabled above 102 values; point values remain available in layer features.
```

On update, preserve user-toggled visibility when possible and do not reset the camera.

## UI State Machine

The transcript value-index UI should be driven by explicit states rather than ad hoc button toggles.

Follow the existing napari-harpy controller pattern for this workflow. The state machine should live in a widget-local controller, not in `HarpyAppState` and not directly in the Qt widget class.

Recommended module split:

```text
src/napari_harpy/_transcript_value_index.py
  Pure source validation, value normalization, direct reader, optional cache builder/validator/reader, and dataclasses.
  No QWidget logic.

src/napari_harpy/viewer/adapter.py
  Points-layer binding support.
  Add/update the transcript value-selection Points layer.

src/napari_harpy/widgets/viewer/transcript_value_index_controller.py
  UI state machine.
  Async value-list/read workers.
  Optional async cache build/read workers.
  Status message and status kind.
  can_load_values / can_visualize / can_build_cache / can_rebuild_cache state.

src/napari_harpy/widgets/viewer/widget.py
  Qt controls and layout.
  Status-card rendering.
  Optional rebuild confirmation dialog.
  Calls into the controller.
```

This mirrors the existing controller split used by `FeatureExtractionController` and `ClassifierController`: the controller owns long-running work, state transitions, and status text, while the widget renders that state and forwards user actions.

`HarpyAppState` should remain the shared viewer/session hub for loaded `SpatialData`, coordinate system, layer bindings, and viewer adapter. Do not put transcript value-index UI state there.

Core states:

```text
NO_SDATA
  No SpatialData object is loaded.

NO_POINTS_ELEMENT
  SpatialData is loaded, but no eligible points element is selected.

LOADING_VALUES
  The direct value table for the selected points element and index column is being computed.

VALUES_READY
  The selected points element and index column are valid, and direct value selection can be used.

MISSING_CACHE
  A points element and index column are selected, but no valid optional cache exists. Direct mode can still be used.

STALE_CACHE
  An optional cache exists, but validation says it does not match the current points element, index column, schema, or required files. Direct mode can still be used.

VALID_CACHE
  An optional cache exists and passes validation. The controller may use the cache-backed acceleration path.

BUILDING_CACHE
  A cache build is running.

BUILD_FAILED
  The last build failed. Any previously valid cache should remain untouched.

LOADING_SELECTION
  The reader is loading selected values and updating the napari layer.

LOADED_SELECTION
  A selection has been loaded or updated in napari.

LOAD_FAILED
  The selected values could not be read or rendered.
```

Represent the states with an explicit enum, for example:

```python
class TranscriptValueIndexUiState(Enum):
    NO_SDATA = "no_sdata"
    NO_POINTS_ELEMENT = "no_points_element"
    LOADING_VALUES = "loading_values"
    VALUES_READY = "values_ready"
    MISSING_CACHE = "missing_cache"
    STALE_CACHE = "stale_cache"
    VALID_CACHE = "valid_cache"
    BUILDING_CACHE = "building_cache"
    BUILD_FAILED = "build_failed"
    LOADING_SELECTION = "loading_selection"
    LOADED_SELECTION = "loaded_selection"
    LOAD_FAILED = "load_failed"
```

The controller should expose read-only state for the widget to render:

```text
state
status_message
status_kind
can_load_values
can_build_cache
can_rebuild_cache
can_visualize
is_loading_values
is_building
is_loading
cache_status
selection
```

Async implementation should follow the existing controller worker pattern:

- use `thread_worker(start_thread=False, ignore_errors=True)`;
- create immutable value-list, build, and read job dataclasses;
- increment job ids for each launched job;
- ignore stale worker results whose job id no longer matches the active job;
- call `worker.quit()` only for shutdown or invalidation, not as a user-facing cancel feature;
- notify the widget through an `on_state_changed` callback.

Control behavior:

```text
NO_SDATA / NO_POINTS_ELEMENT
  Disable points-dependent transcript controls.

LOADING_VALUES
  Disable points element and index-column changes if simple to implement.
  Disable Visualize selected values.
  Show value-loading status.

VALUES_READY
  Enable value search.
  Enable Visualize selected values when at least one value is selected or the all-values option is active.
  Enable optional Build cache when no valid cache exists.

MISSING_CACHE / STALE_CACHE
  Enable optional Build cache.
  Keep direct value search and visualization available when direct values are loaded.
  Show cache status as optional acceleration, not as a blocker.

VALID_CACHE
  Enable Rebuild cache.
  Enable value search.
  Enable Visualize selected values when at least one value is selected or the all-values option is active.
  Prefer the cache-backed reader if implemented; otherwise keep using direct mode.

BUILDING_CACHE
  Disable Build/Rebuild cache.
  Keep direct visualization available unless the first implementation chooses to serialize all transcript work.
  Disable points element and index-column changes if simple to implement.
  Show build progress/status.

BUILD_FAILED
  Enable Build cache again.
  Keep direct value search and visualization enabled when the direct value table is still valid.
  Show the build error in the status card.

LOADING_SELECTION
  Disable Visualize selected values.
  Keep value search visible.
  Do not start another read until the current read finishes.

LOADED_SELECTION
  Re-enable Visualize selected values.
  Show loaded/sampled status in the status card.

LOAD_FAILED
  Re-enable Visualize selected values when the direct value table and selection are still valid.
  Show the read or layer-update error in the status card.
```

Direct value-list computation and selection reads must run asynchronously so the Qt UI does not freeze. Optional cache builds must also run asynchronously if implemented. For the MVP, only one value-list job and one selection read should run at a time for the transcript value-selection layer. The first implementation can simply disable the relevant buttons while work is running rather than queueing multiple requests.

Optional cache build progress can be indeterminate. Prefer clear phase text over fake percentages:

```text
Preparing cache...
Normalizing values...
Counting values...
Shuffling rows by value...
Writing shards...
Writing index...
Finalizing cache...
```

Selection read progress can also be indeterminate:

```text
Loading selected values...
Showing 100,000 of 2,431,912 selected points.
```

Use the existing shared status-card style for transcript cache and selection feedback. Warning and error placement should be in the transcript UI section's status card, not only in logs and not only in napari layer state.

Status-card messages should cover:

- direct value loading;
- direct values ready;
- missing optional cache;
- stale optional cache;
- build in progress;
- build failed;
- optional cache ready;
- loading selection;
- sampled preview warning;
- categorical coloring disabled above 102 selected values;
- read or layer-update failure.

Do not expose a user-facing cancel button in the MVP. Build cancellation can be added later, but it must cancel the worker, clean the temporary build directory, and leave any existing valid cache untouched. Read cancellation can also be added later if real data shows selection reads are long enough to need it.

`render_point_budget` should be user-configurable as a runtime UI setting:

```text
default: 100_000
minimum: 1_000
maximum: 1_000_000
```

Use a numeric text field with the sensible default prefilled. Validate the value before visualization; if it is missing, non-integer, or outside the allowed range, keep visualization disabled and show a status-card warning.

Changing `render_point_budget` must not rebuild the cache and must not affect cache validity. It only affects the next visualization request. `metadata.json["default_render_point_budget"]` is build provenance and a suggested default, not the source of truth for the current UI setting.

## Why This Should Feel Snappy

The direct-first MVP avoids the worst current slow path:

```text
load full dask dataframe -> compute all rows -> filter/subsample
```

Instead, direct selected-value display becomes:

```text
selected values
-> Dask filter on the selected index column
-> compute selected total count
-> exact compute if within render_point_budget
-> otherwise Dask sample before compute and final-trim in memory
-> napari Points layer update
```

The optional cache path, if needed later, would make the same user workflow faster by replacing repeated Dask scans with indexed row-group reads:

```text
selected values
-> value_id values
-> row groups listed in value_index.parquet
-> pyarrow read of only those row groups
-> napari Points layer update
```

For cache-backed sampled large selections:

```text
selected values
-> per-value quotas
-> first few shuffled value_shard row groups per value
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

- Much simpler than the multiscale spatial cache and the value-index cache.
- Gives an early user-visible transcript workflow.
- Supports exact selected-value display when counts are modest.
- Avoids building and maintaining a cache before we know it is needed.
- Gives predictable behavior through a hard render budget.

### Cost

- Direct mode may still scan source partitions for every value switch.
- Direct mode depends on Dask/backing-store performance.
- Direct mode sampled previews do not guarantee one point per selected value.
- Does not solve pan/zoom-scaled loading.
- Exact display is still bounded by napari `Points` performance.
- If the optional cache is implemented later, it will require an offline shuffle by the selected value column and duplicate a subset of the canonical transcript data.
- If the optional cache is implemented later, a row-group-per-value policy can create many small row groups for rare values.

If the value-index cache is implemented, the many-small-row-groups concern is acceptable for the standalone value-index cache. Most transcript datasets have thousands to tens of thousands of values for typical index columns such as `gene`, not millions of values. Rare values should not be packed together in that cache, because exact rare-value reads are one of its main benefits. Packing small value groups belongs to a future spatial plus value layout where `(tile, value)` row-group counts could otherwise explode.

## Implementation Slices

These slices split the MVP into implementation units with clear dependencies. The first slices intentionally establish contracts before the widget work begins, because the UI should depend on stable validation, value table, reader, and layer-update behavior.

The direct no-cache path comes first. The value-index cache is deferred until we have measured whether direct Dask filtering is not good enough for the target datasets and storage backends.

Recommended order:

```text
0 -> 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7
```

Optional cache acceleration slices can follow after the direct workflow is working:

```text
A -> B -> C -> D
```

### Slice 0: Lock Minimal Decisions Before Code

Goal: decide the few contracts that would otherwise cause rework.

Includes:

- direct no-cache path is the first implementation path;
- optional cache path identity, if implemented later: one cache per points element plus index column;
- generic naming: use `value` internally and on disk while keeping `gene` as the default user-facing index column;
- coordinate order for napari: `y, x`;
- first-pass sampling behavior for selections above `render_point_budget`;
- returned/layer feature columns: configured `<index_column>` as categorical plus `value_id`, with no `transcript_id` for the MVP;
- shared return object for direct and optional cache readers.

Done when:

- the implementation has enough fixed terminology and API shape to avoid renaming churn later.

### Slice 1: Core Module Skeleton And Data Contracts

Goal: create the standalone transcript value-selection module without building a cache.

Status: implemented.

Implementation:

- `src/napari_harpy/_transcript_value_index.py`;
- `tests/test_transcript_value_index.py`.

Includes:

- new module `src/napari_harpy/_transcript_value_index.py`;
- constants:
  - `DEFAULT_X = "x"`;
  - `DEFAULT_Y = "y"`;
  - `DEFAULT_INDEX_COLUMN = "gene"`;
  - `DEFAULT_RENDER_POINT_BUDGET = 100_000`;
  - `DEFAULT_RANDOM_STATE = 42`;
  - `TRANSCRIPT_VALUE_INDEX_SCHEMA_VERSION = "harpy-transcripts-value-index-0.1"`;
- dataclasses or typed return objects:
  - `TranscriptValueTable`;
  - `TranscriptValueSelection`;
- direct read job/config objects if useful;
- error handling aligned with the existing codebase:
  - use `ValueError` for expected validation and precondition failures;
  - let unexpected Dask/IO failures propagate to the controller/worker so they become error status messages.

Tests:

- basic object construction;
- `TranscriptValueTable` validates required columns;
- `TranscriptValueSelection` validates coordinate shape/dtype, feature row count, required feature columns, and count invariants;
- `ValueError` is raised for invalid dataclass inputs.

Done when:

- later slices can depend on stable public objects and helper functions.

### Slice 2: Source Points Validation And Value Normalization

Goal: validate whether a selected points element can be visualized directly.

Status: implemented.

Implementation:

- `src/napari_harpy/_transcript_value_index.py`;
- `tests/test_transcript_value_index.py`.

Includes:

- public validation helper:

```python
def validate_points_element_for_value_selection(
    sdata: SpatialData,
    points_name: str,
    *,
    x: str = DEFAULT_X,
    y: str = DEFAULT_Y,
    index_column: str = DEFAULT_INDEX_COLUMN,
    transcript_id: str | None = None,
) -> _ValidatedPointsElement:
    ...
```

- private validated return object:

```python
@dataclass(frozen=True)
class _ValidatedPointsElement:
    points: dd.DataFrame
    points_name: str
    source_path: Path | None
    source_n_points: int
    x: str
    y: str
    index_column: str
    transcript_id: str | None

    @property
    def is_backed(self) -> bool:
        return self.source_path is not None

    @property
    def element_path(self) -> str | None:
        if self.source_path is None:
            return None
        return f"points/{self.points_name}"
```

`points_name` and `source_path` should be stored state. `source_path` is the root zarr store path for backed `SpatialData`, and `None` for unbacked `SpatialData`. `element_path` should be a derived property that returns `None` for unbacked data and `points/<points_name>` for backed data.

- resolve `sdata.points[points_name]`;
- allow backed and unbacked `SpatialData` for direct mode;
- store `source_path = Path(sdata.path)` when the `SpatialData` object is backed, otherwise `None`;
- require `points_name` is a non-empty string;
- require Dask dataframe points element;
- validate `x`, `y`, `index_column`, and optional `transcript_id` are non-empty strings;
- validate `x`, `y`, and `index_column` columns exist;
- reject `index_column == x` or `index_column == y`;
- validate coordinate columns are numeric;
- compute `source_n_points` and reject empty dataframes;
- reject missing, NaN, or infinite coordinate values;
- normalize index values;
- reject missing, empty, invalid, or unsupported index values;
- if `transcript_id` is provided:
  - require the column exists;
  - reject missing `transcript_id` values;
  - require `transcript_id` values are unique.

Value normalization helpers:

```python
def normalize_index_value(value: object) -> str:
    ...


def normalize_index_values(values: pd.Series) -> pd.Series:
    ...
```

Normalization rules:

- reject missing values;
- reject bytes values;
- reject numeric and boolean values;
- reject list, dict, tuple, and other structured Python objects;
- accept string-like values;
- convert accepted values to `str`;
- strip leading and trailing whitespace;
- reject empty strings after stripping;
- preserve case;
- preserve internal whitespace.

Tests:

- missing `x`, `y`, index column;
- non-numeric coordinates;
- non-finite coordinates;
- empty dataframe;
- invalid index dtype;
- missing or blank index values;
- categorical, string, and object value handling.
- backed and unbacked SpatialData;
- missing points element;
- points element that is not a Dask dataframe;
- `index_column == x` or `index_column == y`;
- bytes, numeric, boolean, list, dict, and tuple index values are rejected;
- normalization preserves case and internal whitespace while stripping edges;
- optional `transcript_id` rejects missing, non-unique, and missing-column values.

Done when:

- the direct reader can fail early with clear errors before doing expensive work.

### Slice 3: Direct Value Table And Counts

Goal: compute the selectable value table directly from the selected Dask points dataframe.

Includes:

- public value-table helper:

```python
def build_direct_value_table(
    validated: _ValidatedPointsElement,
) -> TranscriptValueTable:
    ...
```

- input is a `_ValidatedPointsElement` from Slice 2;
- use `validated.points[validated.index_column]`;
- normalize values with the same `normalize_index_value` rules from Slice 2;
- compute normalized value counts with Dask;
- merge source values that normalize to the same string into one value;
- exclude unused categorical categories;
- raise `ValueError` if invalid values are encountered during value-table construction;
- return a `TranscriptValueTable` with:

```text
value_id: uint32
value: string
n_points: uint64
```

- sort normalized values lexicographically by `value`;
- assign `value_id` from `0..n_values-1` after sorting;
- ensure `value_id` is deterministic for the same normalized value table;
- ensure `total_count == sum(n_points) == validated.source_n_points`;
- keep the value table in controller state for search and value resolution;
- do not require `values.parquet`;
- rerun or invalidate the value list when the selected `SpatialData` object, points element, or index column changes;
- do not recompute the value table when only selected values change.

Tests:

- whitespace-normalized duplicates are merged, for example `" AAMP "` and `"AAMP"`;
- case is preserved, so `"ACTB"` and `"actb"` remain separate values;
- counts are correct;
- values are sorted lexicographically;
- deterministic `value_id` assignment;
- `value_id` has dtype `uint32`;
- `n_points` has dtype `uint64`;
- categorical input includes only observed values;
- string and object input work;
- invalid values still raise if they reach value-table construction;
- `total_count == validated.source_n_points`.

Done when:

- the UI can expose searchable values without requiring a cache.

### Slice 4: Direct Runtime Reader

Goal: load selected values directly from the Dask dataframe.

Includes:

- resolve selected values against the direct value table;
- reject unknown selected values;
- compute total selected point count before materializing rows;
- if `total_count <= render_point_budget`, compute exact selected rows;
- if `total_count > render_point_budget`, filter selected rows first, then Dask-sample before compute;
- final-trim in memory to at most `render_point_budget`;
- return `TranscriptValueSelection`;
- include coordinates, features, selected values, selected value ids, loaded count, total count, render budget, sampled flag, and warning text.

Tests:

- one selected value;
- multiple selected values;
- unknown value raises;
- exact load returns all selected rows;
- sampled load returns at most `render_point_budget`;
- direct sampling filters before sampling;
- coordinates are returned as `float32` in napari `y, x` order;
- features include the configured index column and `value_id`;
- empty selections return an empty exact result;
- `values="all"` works and samples when needed.

Done when:

- selected-value loading works without napari and without a cache.

### Slice 5: napari Points Layer Integration

Goal: convert a reader result into one napari `Points` layer.

Includes:

- helper such as `add_transcript_value_points_layer`;
- create or update one existing layer;
- use the reader's `y, x` coordinates directly;
- attach the configured index column as a pandas categorical feature, plus `value_id`;
- register and find the layer through `LayerBindingRegistry`, not by layer name alone;
- reuse the same registered layer when selected values change;
- apply categorical coloring for `2..102` selected values with `default_categorical_colors(n)`;
- use a single solid color for one selected value or more than 102 selected values;
- surface a status-card warning when categorical coloring is disabled above 102 selected values;
- set point size, opacity, and name;
- surface sampled warning.

Tests:

- creates layer when missing;
- updates same layer on repeated calls;
- changing from one selected value to another replaces the existing layer instead of creating a second layer;
- does not create one layer per value;
- ignores unregistered same-name layers;
- features are attached;
- categorical coloring is used for up to 102 selected values;
- solid coloring is used above 102 selected values;
- sampled warning status is present in the widget/controller state.

Done when:

- code can display exact or sampled direct selections in napari through a thin helper.

### Slice 6: Viewer UI Integration

Goal: expose the direct no-cache workflow in the existing Harpy viewer widget.

Includes:

- controller module `src/napari_harpy/widgets/viewer/transcript_value_index_controller.py`;
- explicit `TranscriptValueIndexUiState` enum;
- immutable value-list and read job dataclasses for worker inputs;
- controller-owned status message, status kind, current value table, and current selection;
- points element selector;
- index column selector;
- numeric text-field `render_point_budget` control with default `100_000`, minimum `1_000`, and maximum `1_000_000`;
- widget labels based on the selected index column rather than hard-coded "gene" wording;
- direct value search and select control backed by the direct value table;
- visualize selected values button;
- all-values option;
- explicit UI state machine for `NO_SDATA`, `NO_POINTS_ELEMENT`, `LOADING_VALUES`, `VALUES_READY`, `LOADING_SELECTION`, `LOADED_SELECTION`, and `LOAD_FAILED`;
- optional cache status display if cache helpers already exist;
- run value-list computation asynchronously;
- run selection reads asynchronously;
- ignore stale async value/read results by job id;
- no user-facing cancel button in the MVP;
- progress/status phase text for value loading and read;
- status-card warning placement for sampled previews and categorical-coloring disablement;
- keep UI/controller state as the source of truth rather than layer metadata.

Tests:

- widget initializes without requiring a cache;
- value loading runs when points element or index column changes;
- value search is enabled when direct values are ready;
- selected values trigger direct reader and layer update;
- loading selection disables duplicate visualize requests;
- stale async value/read worker results are ignored;
- `render_point_budget` changes affect the next reader call without rebuilding anything;
- sampled warning is visible.

Done when:

- the user-facing direct workflow exists end to end.

### Slice 7: Integration Tests And Hardening

Goal: make the direct path reliable enough to iterate on real data.

Includes:

- fixture `SpatialData` with transcript points;
- direct validate, value table, exact read, sampled read, and layer update flow;
- failure cases for invalid source data;
- basic performance sanity checks on moderate synthetic data;
- documentation updates or roadmap status notes.

Done when:

- the direct MVP has a tested end-to-end path and clear known limitations.

### Optional Cache Slice A: Cache Validator And Value Table Cache

Goal: add the first useful optional cache artifact: the selectable value table and cache status contract.

Includes:

- cache path helpers;
- metadata read/write helpers;
- compute normalized value counts with Dask;
- sort values deterministically;
- assign stable `value_id`;
- write `values.parquet`;
- write `metadata.json` last;
- check required files, schema version, selected source element, selected columns, and count consistency.

Done when:

- the UI can choose between the direct value table and a valid cache value table.

### Optional Cache Slice B: Full Cache Builder With Data Shards

Goal: write displayable transcript rows grouped by value.

Includes:

- create a working dataframe with `x`, `y`, value id, and optional `transcript_id`;
- globally group by value and shuffle rows within each value group;
- convert display coordinates to `float32`;
- write `data/shard-*.parquet`;
- enforce single-value row groups;
- split large values across row groups;
- assign `value_shard` as the shard order within each value;
- write `value_index.parquet`;
- finalize through staged replacement.

Done when:

- the full on-disk cache can be built and inspected without napari.

### Optional Cache Slice C: Cache-Backed Runtime Reader

Goal: load exact and sampled selected values from the optional cache using PyArrow.

Includes:

- load metadata, `values.parquet`, and `value_index.parquet`;
- resolve selected values to ids;
- reject unknown selected values;
- compute total selected point count before reading data;
- read only listed row groups;
- implement proportional quota allocation for sampled reads;
- support `values="all"`;
- return the same `TranscriptValueSelection` object as direct mode.

Done when:

- cache-backed loading can replace direct loading behind the same napari layer helper.

### Optional Cache Slice D: UI Cache Acceleration Mode

Goal: expose cache build/rebuild and cache-backed loading as an optional acceleration path.

Includes:

- create or rebuild cache button;
- confirmation dialog before rebuilding an existing valid cache;
- cache status display;
- cache build worker and job ids;
- prefer cache-backed reads when a valid cache exists;
- fall back to direct reads when cache is missing or stale;
- disable the build/rebuild button while a cache build is running;
- show build progress/status phase text.

Done when:

- the user can use direct mode by default and opt into cache acceleration when needed.

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

1. Source validator: validates a points element, coordinate columns, and a selected string/categorical index column.
2. Direct value table: computes selectable values and point counts from the Dask points dataframe.
3. Direct reader: loads exact or sampled selected values from the Dask points dataframe, sampling before compute when needed.
4. Viewer UI controls: lets the user choose the points element and index-value column, search/select values, set `render_point_budget`, and request visualization.
5. napari layer integration: creates or updates one `Points` layer for the selected values, with sampled-state warning text when applicable.
6. Tests: cover input validation, direct value resolution, exact reads, sampled reads, all-values selection, async stale-result handling, and layer updates.

Optional later deliverables:

1. Cache builder: builds `transcripts_value_index/<index_column_cache_key>/` from a backed points element and a selected string/categorical index column.
2. Cache validator: checks required files, schema version, selected source element, selected columns, and internal count consistency before cache-backed reads are used.
3. Cache reader: loads exact or sampled selected values from `values.parquet`, `value_index.parquet`, and the data shard files using PyArrow.
4. UI cache acceleration mode: lets the user create or rebuild the cache and uses it when valid, while preserving direct mode as a fallback.

## Suggested Module Shape

Keep this separate from `_transcript_tiles.py` at first:

```text
src/napari_harpy/_transcript_value_index.py
```

Suggested objects:

```text
DEFAULT_X
DEFAULT_Y
DEFAULT_INDEX_COLUMN
DEFAULT_RENDER_POINT_BUDGET
DEFAULT_RANDOM_STATE
TRANSCRIPT_VALUE_INDEX_SCHEMA_VERSION
TranscriptValueTable
TranscriptValueSelection
TranscriptValuePointsLayerBinding
load_transcripts_for_values_direct
build_direct_value_table
add_transcript_value_points_layer
```

Optional cache objects:

```text
TranscriptValueIndexCache
build_transcript_value_index_cache_for_points_element
load_transcripts_for_values_from_cache
```

The direct read path uses Dask.

The optional cache build path can use Dask.

The optional cache read path should use PyArrow.

The napari path should be thin and should not know whether the selection came from direct Dask filtering or the optional cache.

Add the UI controller separately from the cache module:

```text
src/napari_harpy/widgets/viewer/transcript_value_index_controller.py
```

Suggested controller objects:

```text
TranscriptValueIndexUiState
TranscriptValueIndexValueJob
TranscriptValueIndexReadJob
TranscriptValueIndexController
```

Optional cache controller objects:

```text
TranscriptValueIndexBuildJob
```

The controller should own widget-facing state and async worker orchestration. The Qt widget should render controller state and forward user actions; it should not own the cache build/read state machine directly.

## Minimal Acceptance Criteria

A first implementation is successful when:

- it validates `x`, `y`, and the selected index-value column;
- it computes selectable values and counts directly from `sdata.points["transcripts"]`;
- the value search box works without requiring a cache;
- selecting values within the render budget loads exact points;
- selecting values above the render budget filters selected rows first, samples before compute, and loads at most `render_point_budget` points;
- all-values selection is allowed and sampled when needed;
- sampled results show a user-visible warning;
- the napari integration updates one `Points` layer rather than creating many layers.

The optional cache path becomes successful later when:

- it can build `transcripts_value_index/<index_column_cache_key>/` from a backed points element;
- it writes `values.parquet`, `value_index.parquet`, and data Parquet files;
- it validates cache availability and staleness before using the cache-backed reader;
- every data row group listed in `value_index.parquet` contains exactly one `value_id`;
- cache-backed reads return the same `TranscriptValueSelection` object shape as direct reads.
