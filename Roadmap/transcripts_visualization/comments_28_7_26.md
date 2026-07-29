The existing implementation should not be discarded wholesale. Its data contracts, tile arithmetic, row-group writer, and tests are valuable. However, I would redesign the orchestration and high-volume dataflow before adding a public builder.

In short: reuse the foundations, replace the pipeline.

## Findings from the real dataset

I inspected:

`/Users/arne.defauw/VIB/DATA/test_data/sdata_xenium_full_data_core.zarr/points/transcripts_global_ROI1/points.parquet`

| Property | Measured value |
|---|---:|
| Rows | 136,578,750 |
| Genes | 5,122 |
| Compressed source size | approximately 1.17 GiB |
| Parquet files | 65 |
| Row groups | 168 |
| Median row-group size | 1,048,576 rows |
| Coordinate extent | 54,009 × 37,559 source units |
| Source transcript ID | None |
| Dask partitions | 65 |

The source contains a `__null_dask_index__`, but it resets between files and is not globally unique. It must not be used as transcript identity.

A streaming Arrow pass over `x`, `y`, and `gene` took approximately 12.5 seconds on warm local storage. Footer inventory took only 0.02 seconds, and an Arrow gene-count pass took about 7.6 seconds.

These are useful lower bounds rather than final build predictions.

## The current pipeline is doing too many source scans

The existing validation performs a full compute for row counts and value checks ([`_validate_points_element`](/Users/arne.defauw/VIB/napari_harpy/src/napari_harpy/_transcript_tiles.py:188)). On this dataset it took about 22.6 seconds.

After that:

- Bounds computation performs another scan and took approximately 8.8 seconds ([metadata calculation](/Users/arne.defauw/VIB/napari_harpy/src/napari_harpy/_transcript_tiles.py:312)).
- Gene dictionary construction performs another scan and took approximately 32.9 seconds ([gene-table construction](/Users/arne.defauw/VIB/napari_harpy/src/napari_harpy/_transcript_tiles.py:380)).
- Encoding, tiling, and writing would then scan the source again.

That is already roughly 64 seconds of warm-cache preprocessing before writing the first cache row. Coarse levels could add further reevaluation unless the intermediate dataflow is carefully controlled.

A Parquet-specific builder should instead use:

1. A footer preflight for file inventory, row counts, schema, statistics, source signature, and deterministic partition offsets.
2. One streaming gene pass to establish normalized global gene IDs and counts.
3. One exact-level construction pass.
4. Coarse-level construction from the normalized exact cache, not from the original source.

The public API can still accept the Dask dataframe, but the fast path should receive or resolve its physical Parquet source. We should not depend on reverse-engineering arbitrary transformed Dask graphs.

## What should be reused

| Existing idea | Recommendation |
|---|---|
| SpatialData element and output-path resolution | Keep |
| Coordinate/gene/schema validation contracts | Keep, but separate cheap structural validation from streaming value validation |
| Deterministic normalized gene table | Keep |
| `genes.parquet` schema and writer | Keep |
| `uint32` gene/tile IDs and `float32` local coordinates | Keep |
| Tile index and relative-coordinate arithmetic | Keep |
| One logical tile per Parquet row group | Keep |
| Dense-tile row-group sharding | Keep |
| Manifest rows describing physical row groups | Keep |
| Partition-local output files | Keep initially for this dataset |
| Staged replacement and rollback | Keep and strengthen |
| Existing 103 tests | Preserve and adapt |

The tile-local conversion is implemented cleanly in [`_annotate_tile_partition()`](/Users/arne.defauw/VIB/napari_harpy/src/napari_harpy/_transcript_tiles.py:506), while the row-group invariant and manifest collection are sound ideas in [`_write_level_dataset()`](/Users/arne.defauw/VIB/napari_harpy/src/napari_harpy/_transcript_tiles.py:549) and [`_write_level_partition()`](/Users/arne.defauw/VIB/napari_harpy/src/napari_harpy/_transcript_tiles.py:597).

The focused suite remains healthy: **103 tests passed**.

## What should be redesigned

### 1. Level discovery

The current implementation derives every level from spatial extent and doubles tile size between levels ([current formula](/Users/arne.defauw/VIB/napari_harpy/src/napari_harpy/_transcript_tiles.py:345)). This conflicts with the roadmap’s separation of tile geometry and sampling density.

Levels must consider both point count and extent, and multiple density levels may share the same tile geometry.

### 2. Exact tile size

The current default of 1,024 source units is too large for dense exact tiles in this dataset.

| Exact tile size | Populated tiles | Median points/tile | P90 | Maximum |
|---:|---:|---:|---:|---:|
| 256 | 27,310 | 1,417 | 17,316 | 30,533 |
| 512 | 7,296 | 5,014 | 66,117 | 108,143 |
| 1,024 | 1,879 | 18,937 | 250,740 | 404,574 |

I would benchmark **256 and 512** as the real candidates. A 1,024 tile can exceed the intended render budget fourfold before neighboring tiles are considered.

My initial preference is 512 for lower manifest/row-group overhead, unless viewport benchmarks show that its 108k dense tiles make exact views too coarse. The roadmap correctly requires this to remain a measured choice.

### 3. Per-row string tile IDs

The current annotator creates a Python string for every row:

[`tile_id` list construction](/Users/arne.defauw/VIB/napari_harpy/src/napari_harpy/_transcript_tiles.py:525)

For a 1.145M-row source partition, the annotated dataframe occupied approximately 90.6 MiB; the `tile_id` column alone consumed 68.8 MiB.

This is the clearest avoidable memory cost.

During construction, use a packed numeric tile key derived from `tile_x` and `tile_y`. Sort and group on that key. If the on-disk contract continues to require `tile_id`, construct it as a constant/dictionary Arrow column only when writing each row-group shard.

The repeated tile columns compress well on disk: my 1.145M-row write occupied about 13.6 MiB. The problem is build-time pandas memory and Python object creation, not final cache size.

### 4. Stable internal identity

For this dataset, assign:

```text
point_identity = cumulative source-fragment row offset + row position within fragment
```

Fragment ordering must be deterministic and included in the source signature. This matches the roadmap’s fallback identity policy ([stable identity contract](/Users/arne.defauw/VIB/napari_harpy/Roadmap/transcripts_visualization/multi_tile_cache.md:806)).

### 5. Staging and completion

The current first build writes directly into the final directory ([current behavior](/Users/arne.defauw/VIB/napari_harpy/src/napari_harpy/_transcript_tiles.py:290)). The new builder must always use a sibling staging directory, including the first build, and publish only after metadata, manifest, validation, and `COMPLETED` are present.

## Keep the no-shuffle layout initially

For 512-unit tiles:

- 7,296 logical exact tiles
- 10,121 file–tile fragments
- median one file per tile
- 90% of tiles in at most two files
- maximum four files per tile
- approximately 11,045 row groups at a 50k row-group limit

This is good enough to justify starting with the existing partition-local layout. It avoids an expensive 136.6M-row shuffle while producing modest fragmentation for this particular source.

That aligns with roadmap Layout A ([physical-layout comparison](/Users/arne.defauw/VIB/napari_harpy/Roadmap/transcripts_visualization/multi_tile_cache.md:861)). Layout C should remain available if cold viewport benchmarks show that two-to-four file reads are too expensive. I would not implement the global tile shuffle first.

## Recommended construction pipeline

```text
Parquet footer preflight
        ↓
global normalized gene dictionary
        ↓
bounded parallel exact-level writer
        ↓
exact manifest
        ↓
tile-by-tile gene-aware sampling
        ↓
next finer sampled level
        ↓
repeat fine → coarse
        ↓
metadata + manifest validation
        ↓
COMPLETED + atomic publication
```

The key idea is to use the exact cache as the normalized construction intermediate.

Each exact logical tile can be loaded from its one-to-four shards, sampled independently using micro-grid/gene counts and stable hash ranking, then written to the next coarser level. Subsequent levels operate only on representatives retained by the previous level. This naturally gives the required nesting:

```text
level_0 ⊆ level_1 ⊆ ... ⊆ exact
```

It also avoids a global Dask shuffle and avoids rescanning the original 136.6M-row dataframe once per level.

## Suggested first implementation slice

I would split the work into two internal milestones:

1. **Exact-cache performance spike**

   - Parquet source preflight and signature.
   - Global gene dictionary.
   - Stable fallback identities.
   - Exact 256/512 tile writer.
   - Numeric construction keys.
   - Bounded worker concurrency.
   - Manifest, metadata, staging, and completion validation.
   - Full cold and warm benchmark on this dataset.

2. **First valid multiscale cache**

   - Deterministic gene-aware sampler.
   - One sampled intermediate level.
   - One bounded whole-dataset overview.
   - Nested-membership validation.
   - Global overview-budget validation.

The exact-only artifact should remain an internal staging/benchmark artifact, not be published as a completed multiscale cache.

A reasonable initial performance hypothesis on this machine is:

- exact stage below roughly 3 minutes;
- complete initial multiscale cache below roughly 5 minutes;
- bounded peak memory, ideally under 3–4 GiB.

Those are targets to validate, not results—I did not write the complete 1.5–2+ GiB cache during this investigation.

No repository files were changed.