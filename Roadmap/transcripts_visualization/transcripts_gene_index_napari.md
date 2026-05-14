# Transcript Gene-Index MVP For napari

This note describes a smaller first working version of transcript visualization in napari.

The existing transcript visualization roadmap is spatial-first: build a tiled multiscale cache so pan and zoom can stay responsive for all transcripts. That is still the right long-term direction for whole-slide transcript rendering.

This document proposes a narrower MVP:

- start from `sdata.points["transcripts"]`;
- validate that the points table has `x`, `y`, and `gene` columns, or another configured index column;
- build a value-first cache optimized for quickly loading selected genes or another configured string/categorical value;
- visualize the selected result in one napari `Points` layer;
- if the selected subset is too large, show a deterministic sample and warn the user.

The goal is not to solve every zoom level and viewport problem yet. The goal is to make selected-gene transcript visualization feel useful quickly.

## Main Recommendation

Yes, a gene index is worth building for this MVP if the primary interaction is:

```text
select one or more genes -> show those transcripts in napari
```

However, the important part is not just the existence of an index file. The cache must also be physically organized by gene.

If the original `points.parquet` row groups contain a random mixture of genes, a `gene -> row groups` lookup will not help much. The reader would still need to scan most row groups and filter in memory.

For selected-gene visualization to be snappy, the MVP cache should enforce this invariant:

```text
Each data row group contains rows for exactly one gene_id.
```

Large genes can span many row groups. Small genes can fit in one small row group. The reader can then resolve selected genes to a short list of row groups, read only those row groups with `pyarrow`, and update a napari `Points` layer.

This makes gene subsetting fast. It does not make arbitrary spatial viewport queries fast. That remains the job of the spatial tiled cache.

## Relationship To The Spatial-First Cache

This MVP should be treated as a separate cache, not as a replacement for `transcripts_vis/`.

Recommended layout:

```text
<sdata.zarr>/
  points/
    <points_name>/
      points.parquet
      transcripts_gene_index/
        metadata.json
        genes.parquet
        gene_index.parquet
        data/
          shard-00000.parquet
          shard-00001.parquet
          ...
```

`points.parquet` remains the canonical exact table.

`transcripts_gene_index/` is a Harpy-owned visualization cache that can be deleted and rebuilt.

Why separate from `transcripts_vis/`:

- the spatial cache wants rows grouped by tile and level;
- the gene MVP wants rows grouped by gene;
- one physical ordering cannot be optimal for both access patterns;
- keeping the caches separate avoids complicating the existing spatial-first implementation while we learn from the MVP.

Later, the two directions can meet in a gene-aware tiled cache, such as the `tile_gene_index.parquet` idea from the broader roadmap.

## Scope

The MVP supports:

- backed `SpatialData`;
- one points element, initially `sdata.points["transcripts"]`;
- required coordinate columns, default `x="x"` and `y="y"`;
- configurable string/categorical index column, default `gene="gene"`;
- exact visualization when the selected subset is below a render threshold;
- sampled visualization when the selected subset exceeds the threshold;
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

The cache can keep the existing `gene` terminology in file names and internal fields for the MVP, but it should store the actual source column name in `metadata.json`. In this document, "value" means one value from the configured index column.

Validation should reject:

- missing coordinate columns;
- missing index column;
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

If present, `transcript_id` should be used for stable sampling. If absent, the builder can create a best-effort internal row identity, but deterministic sampling across rebuilds is only guaranteed when the source row order and partitioning are stable.

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
transcript_id: string | null
source_n_transcripts: int
n_genes: int
target_rows_per_row_group: int
default_max_points: int
sample_key_policy: string
```

Suggested schema version:

```text
harpy-transcripts-gene-index-0.1
```

Write `metadata.json` last, or write a separate completion marker last. The cache should not look valid until all required files have been written.

### `genes.parquet`

One row per gene:

```text
gene_id: uint32
gene: string
n_transcripts: uint64
```

Sort genes lexicographically for deterministic `gene_id` assignment in the first implementation.

The reader uses this file to:

- resolve selected values to `gene_id`;
- reject selected values that are not present in the cache vocabulary;
- estimate the selected transcript count before reading data;
- decide whether the selection should be exact or sampled.

### `gene_index.parquet`

One row per data row group:

```text
gene_id: uint32
data_file: string
row_group: int32
gene_row_group: int32
n_points: int64
x_min: float64
x_max: float64
y_min: float64
y_max: float64
```

`data_file` should be relative to the cache root, for example:

```text
data/shard-00003.parquet
```

`gene_row_group` is the row group order within that gene after sorting by the stable sample key. This makes preview sampling simple:

```text
read the first k row groups for this gene
```

Required invariant:

```text
Each `gene_index.parquet` row points to a Parquet row group containing exactly one gene_id.
```

The bounding box columns are not needed for the first static selected-gene display, but they are cheap to compute and useful for:

- future "zoom to selected genes";
- rough spatial extent display;
- debugging cache correctness;
- possible later hybrid spatial filtering.

### `data/shard-*.parquet`

The data files store the displayable transcript rows.

Required columns:

```text
x: float32
y: float32
gene_id: uint32
sample_key: uint64
```

Optional columns:

```text
transcript_id
```

Store display coordinates as `float32`. This is good enough for napari transcript visualization and reduces memory pressure in both the cache and the `Points` layer. The canonical full-precision values remain in `points.parquet`. If exact coordinate preservation is needed for picked transcripts later, use `transcript_id` to look up canonical rows.

Rows inside each gene should be ordered by a stable random-looking `sample_key`.

This matters because it lets the reader build a preview without reading the full gene:

```text
gene rows are grouped by gene_id
within each gene_id, rows are sorted by sample_key
first row groups for a gene are therefore a deterministic random-like preview
```

Do not use Python's built-in `hash`, because it is intentionally not stable across processes.

Possible first implementation:

```text
sample_key = pandas.util.hash_pandas_object(..., index=False, hash_key=<fixed 16 byte key>)
```

If cross-language stability becomes important, switch to a named stable hash such as xxhash64 or BLAKE2b-64.

## Cache Availability And Staleness Checks

The viewer should not allow transcript visualization directly from `sdata.points[points_name]`.

Instead, the transcript UI should follow this flow:

1. The user selects the points element and index-value column.
2. The user clicks `Create cache` or `Rebuild cache`.
3. Harpy builds `transcripts_gene_index/`, including `genes.parquet`.
4. The value search box is enabled only when a valid cache is available.
5. The value search box reads available values from `genes.parquet`, not from the source points dataframe.

This avoids offering values that are not present in the cache and avoids expensive source-data scans during interactive use.

For the MVP, use cheap validation checks only. Do not compute a source Parquet footer digest, full content hash, or fresh source `value_counts` when opening the viewer.

Required cache checks before enabling visualization:

```text
cache exists
metadata.json exists
genes.parquet exists
gene_index.parquet exists
metadata schema_version matches
metadata source_element_path matches selected points element
metadata x/y/index column names match current UI selection
metadata source_n_transcripts == sum(genes.n_transcripts)
metadata source_n_transcripts == sum(gene_index.n_points)
metadata n_genes == len(genes.parquet)
```

If any check fails, mark the cache as missing or stale, disable the value search box and transcript visualization button, and ask the user to build or rebuild the cache.

These checks mostly validate that the selected points element and UI column choices match the cache, and that the cache is internally consistent. They are not a cryptographic guarantee that the source points table has not changed. That stronger source fingerprint can be added later if stale-cache bugs become common, but it is intentionally out of scope for the MVP.

## Row Group Size

Use this default for the gene-index MVP:

```text
target_rows_per_row_group = 25_000
```

This is intentionally smaller than a bulk-analytics Parquet row group. The cache is for interactive selected-gene display, where read granularity matters more than maximum compression.

Recommended tuning range:

```text
10_000 rows: more responsive previews and less sample overshoot
25_000 rows: default MVP choice
50_000 rows: fewer row groups and better compression for very large datasets
```

The default pairs well with:

```text
max_points = 100_000
```

With those values, a large selected gene needs about four row groups to reach the default preview size, while rare genes still usually fit in one row group.

The row group size is a target, not a reason to mix genes. The stronger invariant is:

```text
Never mix genes inside a row group in the MVP.
```

Consequences:

- a gene with fewer than `25_000` transcripts gets one smaller row group;
- a gene with more than `25_000` transcripts is split across multiple row groups;
- a row group should not be padded with rows from another gene;
- the final row group for a large gene may contain fewer than `25_000` rows.

This is acceptable for the standalone gene-index MVP. With `1_000_000_000` transcripts, at most about `25_000` genes, and `target_rows_per_row_group = 25_000`, the rough row-group count is bounded by:

```text
1_000_000_000 / 25_000 + 25_000 = about 65_000 row groups
```

Those are row groups inside Parquet shard files, not separate files. That scale is reasonable for the MVP and keeps rare-gene reads exact and simple.

## Future Spatial Plus Gene Layout

Do not carry the strict single-gene row-group rule directly into a future spatial plus gene cache.

A future `(tile, gene)` layout should avoid creating one tiny row group for every non-empty `(tile, gene_id)` pair. That could produce millions of small row groups once spatial tiling is introduced.

Recommended future layout:

```text
write tile-major data
within each tile, sort rows by gene_id
within each gene_id, sort by sample_key or spatial key
write row groups up to a target size
allow multiple small genes in one row group
keep each gene's rows contiguous within a row group
split one gene across row groups only when that (tile, gene_id) group exceeds the target size
```

The future `tile_gene_index.parquet` would then point to ranges, not necessarily whole single-gene row groups:

```text
level
tile_x
tile_y
gene_id
data_file
row_group
row_offset
n_points
```

For a rare gene, the reader may load one row group that also contains neighboring rare genes, then slice or filter in memory. The false-positive read is bounded by the row-group size. That tradeoff is better than exploding the number of row groups.

So the storage policy is:

```text
gene-only MVP: keep row groups single-gene
future spatial plus gene cache: use tile-major, gene-contiguous indexed ranges
```

## Build Algorithm

Recommended public entry point:

```python
def build_transcript_gene_index_cache_for_points_element(
    sdata: SpatialData,
    points_name: str = "transcripts",
    *,
    output_path: str | PathLike[str] | None = None,
    x: str = "x",
    y: str = "y",
    index_column: str = "gene",
    transcript_id: str | None = None,
    target_rows_per_row_group: int = 25_000,
    default_max_points: int = 100_000,
) -> TranscriptGeneIndexCache:
    ...
```

Implementation steps:

1. Validate the backed `SpatialData` object and resolve the points element path with `sdata.locate_element(...)`.
2. Validate the points dataframe schema and data quality.
3. Normalize selected index values by stripping whitespace and converting valid values to strings.
4. Build `genes.parquet` using a Dask `value_counts`.
5. Assign deterministic `gene_id` values from the sorted gene table.
6. Create a working dataframe with `x`, `y`, `gene_id`, optional `transcript_id`, and `sample_key`.
7. Shuffle or repartition by `gene_id` so each gene is written from as few partitions as practical.
8. Within each output partition, sort by `gene_id` and `sample_key`.
9. Write Parquet row groups so each row group contains only one `gene_id`.
10. Split very large genes across multiple row groups of at most `target_rows_per_row_group`.
11. Build `gene_index.parquet` from the written row-group metadata.
12. Finalize through staged replacement so incomplete caches are not exposed as valid.

For the first implementation, it is acceptable for one gene to appear in multiple shard files as long as every row group is single-gene and every row group is listed in `gene_index.parquet`.

## Runtime Reader

Recommended reader entry point:

```python
def load_transcripts_for_values(
    cache_path: str | PathLike[str],
    values: Sequence[str] | Literal["all"],
    *,
    max_points: int = 100_000,
    sample: bool = True,
    columns: Sequence[str] = ("x", "y", "gene_id"),
) -> TranscriptGeneIndexSelection:
    ...
```

The reader should use `pyarrow`, not Dask, in the interactive path.

Runtime flow:

1. Load `metadata.json`, `genes.parquet`, and `gene_index.parquet`.
2. Resolve selected values to `gene_id`.
3. Sum `n_transcripts` for the selected values before reading data.
4. If the selected count is `<= max_points`, read all row groups for those values.
5. If the selected count is `> max_points`, read a deterministic sample.
6. Return coordinates and features for one napari `Points` layer.

Selected values should already come from `genes.parquet`. If a selected value is not present in `genes.parquet`, the reader should raise an error instead of warning and skipping it. That means the UI or controller allowed stale or invalid selection state to reach the reader, which is an internal consistency bug.

The UI should make this rare by only allowing selections resolved from the cache-backed value search box. If the error still happens, surface it as a cache/selection consistency problem and ask the user to refresh the selection or rebuild the cache.

## Sampling Policy

The default threshold should start at:

```text
max_points = 100_000
```

If the selected values contain at most `max_points` transcripts:

```text
show exact selected transcripts
```

If the selected values contain more than `max_points` transcripts:

```text
show a deterministic sample and warn the user
```

The warning should include:

```text
Showing 100,000 of 2,431,912 selected transcripts.
```

Recommended sampling policy:

The MVP default is proportional sampling by transcript count, with a minimum of one point per selected value when possible. This preserves the visual meaning of density while avoiding complete disappearance of selected rare values in sampled previews.

1. Allocate a sample quota per selected value, proportional to its transcript count.
2. If the number of selected values is less than `max_points`, give each selected value at least one point when possible.
3. For each value, read the first row groups in `gene_row_group` order until at least the quota is available.
4. If the loaded rows exceed the quota, downsample by `sample_key`.
5. Concatenate the sampled rows across values.

Balanced sampling and user-selectable sampling modes are deferred until the UI needs an explicit comparison mode. They are useful for comparing spatial patterns across values, but they intentionally distort abundance and should not be the default.

Because rows within each gene are sorted by a stable random-looking `sample_key`, reading the first row groups gives a deterministic preview without scanning the full selected subset.

This is important. If we simply read every selected row and then sample, large gene selections will still be slow.

## All-Values Selection

Selecting all values can be allowed.

It should use the same count and sampling policy:

```text
values = "all"
total_count = sum(genes_table.n_transcripts)
if total_count <= max_points:
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
- data: `Nx2` array from `y, x` or `x, y`, matching the coordinate convention used elsewhere in Harpy;
- features: at least the configured index-value column;
- face color: categorical by the configured index-value column for small selections;
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
-> gene_id values
-> row groups listed in gene_index.parquet
-> pyarrow read of only those row groups
-> napari Points layer update
```

For sampled large selections:

```text
selected values
-> per-value quotas
-> first few sample-key-sorted row groups per value
-> pyarrow read of a bounded number of rows
-> napari Points layer update
```

This should be fast for:

- one rare gene;
- a handful of marker genes;
- a moderate gene panel;
- all values as a sampled preview.

It will not be fast for:

- exact display of millions of selected transcripts;
- viewport-exact all-transcript rendering;
- spatial queries such as "only visible transcripts in this rectangle".

Those remain spatial-cache problems.

## Design Tradeoffs

### Benefit

- Much simpler than the multiscale spatial cache.
- Gives an early user-visible transcript workflow.
- Supports exact selected-value display when counts are modest.
- Avoids scanning `points.parquet` for every value switch.
- Gives predictable behavior through a hard render threshold.

### Cost

- Requires an offline shuffle by the selected value column.
- Duplicates a subset of the canonical transcript data in another cache.
- Does not solve pan/zoom-scaled loading.
- Exact display is still bounded by napari `Points` performance.
- A row-group-per-value policy can create many small row groups for rare values.

The many-small-row-groups concern is acceptable for the standalone gene-index MVP. Most transcript datasets have thousands to tens of thousands of genes, not millions of genes. Rare genes should not be packed together in this MVP, because exact rare-gene reads are one of the main benefits of the cache. Packing small gene groups belongs to a future spatial plus gene layout where `(tile, gene)` row-group counts could otherwise explode.

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
- generic naming: keep `gene` everywhere or introduce internal `value`;
- coordinate order for napari: explicit `y, x` or `x, y`;
- first-pass sampling behavior for selections above `max_points`;
- basic cache lifecycle: temporary build path, final cache path, rebuild replacement.

Done when:

- the implementation has enough fixed vocabulary and paths to avoid renaming churn later.

### Slice 1: Core Module Skeleton And Data Contracts

Goal: create the standalone transcript gene-index module without building the full cache yet.

Includes:

- new module, likely `src/napari_harpy/_transcript_gene_index.py`;
- schema version constant;
- dataclasses or typed return objects:
  - `TranscriptGeneIndexCache`;
  - `TranscriptGeneIndexSelection`;
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
- validate optional `transcript_id`.

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
- assign stable `gene_id` or value id;
- write `genes.parquet`;
- write `metadata.json` last;
- store source points name and path, selected columns, transcript id column, counts, row-group target, and sampling policy.

Tests:

- deterministic value id assignment;
- correct transcript counts;
- metadata consistency;
- `metadata.json` written only after required cache files;
- selected values come from `genes.parquet`, not the source dataframe.

Done when:

- a cache can expose searchable values, even before point loading exists.

### Slice 4: Full Cache Builder With Data Shards

Goal: write displayable transcript rows grouped by value.

Includes:

- create a working dataframe with:
  - `x`;
  - `y`;
  - value id;
  - `sample_key`;
  - optional `transcript_id`;
- convert display coordinates to `float32`;
- write `data/shard-*.parquet`;
- enforce single-value row groups;
- split large values across row groups;
- sort rows within each value by `sample_key`;
- write `gene_index.parquet`;
- finalize through staged replacement.

Tests:

- required data columns exist;
- coordinates are `float32`;
- every indexed row group contains exactly one value id;
- row groups do not exceed target size except where explicitly allowed;
- all row groups are listed in `gene_index.parquet`;
- `sum(gene_index.n_points) == source_n_transcripts`.

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
  - metadata versus `genes.parquet`;
  - metadata versus `gene_index.parquet`;
  - metadata `n_genes` versus `genes.parquet`;
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

Goal: load selected values below the threshold using PyArrow only.

Includes:

- load metadata, `genes.parquet`, and `gene_index.parquet`;
- resolve selected values to ids;
- reject unknown selected values;
- compute total selected transcript count before reading data;
- read only listed row groups;
- return a `TranscriptGeneIndexSelection`;
- include coordinates, features, selected values, loaded count, total count, and sampled flag.

Tests:

- one selected value;
- multiple selected values;
- unknown value raises;
- exact load returns all selected rows;
- reader does not use Dask;
- features include configured index-value column.

Done when:

- selected-gene loading works without napari.

### Slice 7: Sampled Runtime Reader

Goal: load bounded deterministic previews when selected values exceed `max_points`.

Includes:

- proportional quota allocation;
- minimum one point per selected value when possible;
- deterministic quota rounding and tie-breaking;
- read first required row groups in `gene_row_group` order;
- downsample excess rows by `sample_key`;
- support `values="all"`;
- produce warning text like:
  - `Showing 100,000 of 2,431,912 selected transcripts.`;

Tests:

- sampled result has at most `max_points`;
- sampling is deterministic;
- rare selected values are preserved when possible;
- all-values selection works;
- duplicate selected values are handled predictably;
- `max_points < number_of_selected_values` behavior is defined and tested.

Done when:

- large selections return fast, bounded, deterministic previews.

### Slice 8: napari Points Layer Integration

Goal: convert a reader result into one napari `Points` layer.

Includes:

- helper such as `add_transcript_gene_points_layer`;
- create or update one existing layer;
- use explicit coordinate order;
- attach feature column for selected value;
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
- create or rebuild cache button;
- cache status display;
- value search and select control backed by `genes.parquet`;
- visualize selected values button;
- all-values option;
- disable controls when cache is missing, stale, building, or loading;
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

## Deliverables

The MVP should produce these implementation pieces:

1. Cache builder: builds `transcripts_gene_index/` from a backed points element and a selected string/categorical index column.
2. Cache validator: checks required files, schema version, selected source element, selected columns, and internal count consistency before visualization is enabled.
3. Cache reader: loads exact or sampled selected values from `genes.parquet`, `gene_index.parquet`, and the data shard files using PyArrow.
4. Viewer UI controls: lets the user choose the points element and index-value column, create or rebuild the cache, search/select values from `genes.parquet`, and request visualization.
5. napari layer integration: creates or updates one `Points` layer for the selected values, with sampled-state warning text when applicable.
6. Tests: cover input validation, cache layout, staleness checks, value resolution, exact reads, sampled reads, and row-group invariants.

## Suggested Module Shape

Keep this separate from `_transcript_tiles.py` at first:

```text
src/napari_harpy/_transcript_gene_index.py
```

Suggested objects:

```text
TRANSCRIPT_GENE_INDEX_SCHEMA_VERSION
TranscriptGeneIndexCache
TranscriptGeneIndexSelection
build_transcript_gene_index_cache_for_points_element
load_transcripts_for_values
add_transcript_gene_points_layer
```

The build path can use Dask.

The read path should use PyArrow.

The napari path should be thin and should not know how the cache is built internally.

## Minimal Acceptance Criteria

A first implementation is successful when:

- it can build `transcripts_gene_index/` from a backed `sdata.points["transcripts"]`;
- it validates `x`, `y`, and the selected index-value column;
- it writes `genes.parquet`, `gene_index.parquet`, and data Parquet files;
- it validates cache availability and staleness before enabling visualization;
- the value search box reads selectable values from `genes.parquet`;
- every data row group listed in `gene_index.parquet` contains exactly one `gene_id`;
- selecting values below the threshold loads exact points;
- selecting values above the threshold loads at most `max_points` points;
- all-values selection is allowed and sampled when needed;
- sampled results show a user-visible warning;
- the napari integration updates one `Points` layer rather than creating many layers.
