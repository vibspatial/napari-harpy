# Transcript Gene-Index MVP For napari

This note describes a smaller first working version of transcript visualization in napari.

The existing transcript visualization roadmap is spatial-first: build a tiled multiscale cache so pan and zoom can stay responsive for all transcripts. That is still the right long-term direction for whole-slide transcript rendering.

This document proposes a narrower MVP:

- start from `sdata.points["transcripts"]`;
- validate that the points table has `x`, `y`, and `gene` columns, or another configured index column;
- build a label-first cache optimized for quickly loading selected genes or another configured string/categorical label;
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
- programmatic label selection first, with a simple UI for choosing the index column and values later.

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

The first implementation should already allow the user to choose the index column. The default is `gene`, but any eligible label column can be used.

Eligible index columns:

- are not the configured `x` or `y` coordinate columns;
- have string-like values or a categorical dtype;
- can be normalized to non-empty string labels.

Examples:

```text
gene
feature_name
target
probe
```

The cache can keep the existing `gene` terminology in file names and internal fields for the MVP, but it should store the actual source column name in `metadata.json`. In this document, "gene" often means "the configured index-label column".

Validation should reject:

- missing coordinate columns;
- missing index column;
- an index column that is neither string-like nor categorical;
- non-numeric `x` or `y`;
- non-finite `x` or `y`;
- missing index-label values;
- index-label values whose string form is empty after stripping whitespace;
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
gene: string
transcript_id: string | null
n_transcripts: int
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

- resolve selected labels to `gene_id`;
- report unknown labels;
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
    gene: str = "gene",
    transcript_id: str | None = None,
    target_rows_per_row_group: int = 25_000,
    default_max_points: int = 100_000,
) -> TranscriptGeneIndexCache:
    ...
```

Implementation steps:

1. Validate the backed `SpatialData` object and resolve the points element path with `sdata.locate_element(...)`.
2. Validate the points dataframe schema and data quality.
3. Normalize selected index labels by stripping whitespace and converting valid values to strings.
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
def load_transcripts_for_genes(
    cache_path: str | PathLike[str],
    genes: Sequence[str] | Literal["all"],
    *,
    max_points: int = 100_000,
    sample: bool = True,
    columns: Sequence[str] = ("x", "y", "gene_id"),
) -> TranscriptPointsSelection:
    ...
```

The reader should use `pyarrow`, not Dask, in the interactive path.

Runtime flow:

1. Load `metadata.json`, `genes.parquet`, and `gene_index.parquet`.
2. Resolve selected labels to `gene_id`.
3. Sum `n_transcripts` for the selected labels before reading data.
4. If the selected count is `<= max_points`, read all row groups for those labels.
5. If the selected count is `> max_points`, read a deterministic sample.
6. Return coordinates and features for one napari `Points` layer.

Unknown selected labels should not crash the viewer. They should be reported in a warning and skipped.

## Sampling Policy

The default threshold should start at:

```text
max_points = 100_000
```

If the selected labels contain at most `max_points` transcripts:

```text
show exact selected transcripts
```

If the selected labels contain more than `max_points` transcripts:

```text
show a deterministic sample and warn the user
```

The warning should include:

```text
Showing 100,000 of 2,431,912 selected transcripts.
```

Recommended sampling policy:

The MVP default is proportional sampling by transcript count, with a minimum of one point per selected label when possible. This preserves the visual meaning of density while avoiding complete disappearance of selected rare labels in sampled previews.

1. Allocate a sample quota per selected label, proportional to its transcript count.
2. If the number of selected labels is less than `max_points`, give each selected label at least one point when possible.
3. For each label, read the first row groups in `gene_row_group` order until at least the quota is available.
4. If the loaded rows exceed the quota, downsample by `sample_key`.
5. Concatenate the sampled rows across labels.

Balanced sampling and user-selectable sampling modes are deferred until the UI needs an explicit comparison mode. They are useful for comparing spatial patterns across labels, but they intentionally distort abundance and should not be the default.

Because rows within each gene are sorted by a stable random-looking `sample_key`, reading the first row groups gives a deterministic preview without scanning the full selected subset.

This is important. If we simply read every selected row and then sample, large gene selections will still be slow.

## All-Genes Selection

Selecting all genes can be allowed.

It should use the same count and sampling policy:

```text
genes = "all"
total_count = sum(genes.n_transcripts)
if total_count <= max_points:
    show exact
else:
    show sampled preview and warn
```

For all-label sampling, proportional quotas are the MVP default. They preserve the global abundance distribution.

A possible alternative is a more balanced per-label sample, where rare labels get more visibility than proportional sampling would give them. That is useful for exploratory biology, but it changes the visual meaning of density. Start with proportional sampling and make balanced sampling an explicit option later.

## napari Integration

The MVP should create or update one napari `Points` layer.

Layer behavior:

- layer name: `transcripts` or `transcripts: selected genes`;
- data: `Nx2` array from `y, x` or `x, y`, matching the coordinate convention used elsewhere in Harpy;
- features: at least the configured index-label column;
- face color: categorical by the configured index-label column for small selections;
- warning: displayed when the layer is sampled.

Important implementation detail:

```text
Do not create one napari layer per gene.
```

A single layer is easier to update, hide, remove, and later connect to a controller.

For many selected labels, categorical coloring may become visually noisy. The first version can still attach the configured label feature and use a simple color cycle, then refine the UI later.

## Why This Should Feel Snappy

This MVP avoids the current slow path:

```text
load full dask dataframe -> compute all rows -> filter/subsample
```

Instead, selected-label display becomes:

```text
selected labels
-> gene_id values
-> row groups listed in gene_index.parquet
-> pyarrow read of only those row groups
-> napari Points layer update
```

For sampled large selections:

```text
selected labels
-> per-gene quotas
-> first few sample-key-sorted row groups per gene
-> pyarrow read of a bounded number of rows
-> napari Points layer update
```

This should be fast for:

- one rare gene;
- a handful of marker genes;
- a moderate gene panel;
- all genes as a sampled preview.

It will not be fast for:

- exact display of millions of selected transcripts;
- viewport-exact all-transcript rendering;
- spatial queries such as "only visible transcripts in this rectangle".

Those remain spatial-cache problems.

## Design Tradeoffs

### Benefit

- Much simpler than the multiscale spatial cache.
- Gives an early user-visible transcript workflow.
- Supports exact selected-gene display when counts are modest.
- Avoids scanning `points.parquet` for every gene switch.
- Gives predictable behavior through a hard render threshold.

### Cost

- Requires an offline shuffle by gene.
- Duplicates a subset of the canonical transcript data in another cache.
- Does not solve pan/zoom-scaled loading.
- Exact display is still bounded by napari `Points` performance.
- A row-group-per-gene policy can create many small row groups for rare genes.

The many-small-row-groups concern is acceptable for an MVP. Most transcript datasets have thousands to tens of thousands of genes, not millions of genes. If metadata overhead becomes a problem, we can later pack very rare genes together and accept filtered false-positive reads for that rare-gene tail.

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
load_transcripts_for_genes
add_transcript_gene_points_layer
```

The build path can use Dask.

The read path should use PyArrow.

The napari path should be thin and should not know how the cache is built internally.

## Minimal Acceptance Criteria

A first implementation is successful when:

- it can build `transcripts_gene_index/` from a backed `sdata.points["transcripts"]`;
- it validates `x`, `y`, and `gene`;
- it writes `genes.parquet`, `gene_index.parquet`, and data Parquet files;
- every data row group listed in `gene_index.parquet` contains exactly one `gene_id`;
- selecting a gene below the threshold loads exact points;
- selecting genes above the threshold loads at most `max_points` points;
- all-genes selection is allowed and sampled when needed;
- sampled results show a user-visible warning;
- the napari integration updates one `Points` layer rather than creating many layers.

## Open Questions

Questions to answer during implementation:

- Should missing selected labels be warnings only, or should strict mode raise an error?
- Should cache staleness detection compare source file metadata, source row count, or a stronger fingerprint?
- Is one row group per rare gene acceptable for the datasets we care about?

The only question that should block the MVP is the row-group invariant. Without gene-organized row groups, the index will not deliver the intended speedup.
