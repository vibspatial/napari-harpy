# Interactive transcript metadata cache

Status: architecture proposal for metadata-aware transcript visualization

Roadmap date: 2026-09-01

## High-level decision

The dual-order multiscale cache must distinguish transcript identity from
transcript attributes and must distinguish attributes by their interactive
access pattern.

`point_id` is the canonical cache-local identity and the correct join key for
one or a small number of transcripts. It is not, by itself, a suitable bulk
join mechanism for every visible point. Gathering metadata in source-row order
for tens of thousands of point IDs can touch thousands of unrelated chunks and
recreate the sparse-read amplification that the value-major coordinate sidecar
was introduced to remove.

The metadata-aware cache therefore uses three primary storage classes:

1. feature-level metadata, stored once per normalized `value_id`;
2. hot per-transcript attributes, physically aligned with both display orders;
3. cold per-transcript details, stored once in dense `point_id` order.

Optional relationship indexes, such as a `cell_id` inverted index, are a
separate fourth class and are added only for demonstrated high-cardinality
selection workloads.

The canonical SpatialData Parquet points element remains the complete source of
truth. The Zarr hierarchy is a derived, deletable, and rebuildable visualization
index. Dask remains a construction tool and never participates in interactive
metadata lookup.

## Motivation and current limitation

The current cache construction contract selects only `x`, `y`, and the
categorical value column ([`PointColumnSelection`](/Users/arne.defauw/VIB/napari_harpy/src/napari_harpy/core/multi_scale_cache_points_zarr/source/models.py:24)).
It does not retain quality, control classification, cell assignment, field of
view, nucleus overlap, vendor transcript identity, or other source columns.

The existing tile-major payload does persist a `point_id`, but the display
reader intentionally never selects point IDs. In selected-value mode it also
synthesizes `value_id` rather than gathering the aligned array, because that
gather would reintroduce sparse many-chunk decoding
([`read_display_payloads()`](/Users/arne.defauw/VIB/napari_harpy/src/napari_harpy/core/multi_scale_cache_points_zarr/storage/bucket_reader.py:313)).

The proposed value-major sidecar currently persists only `location` and
`value_point_indptr`. Its construction-time `ordered_row_start` maps every
catalog record back to tile-major storage, but that mapping is deliberately not
published ([`_write_value_major_sidecars()`](/Users/arne.defauw/VIB/napari_harpy/src/napari_harpy/core/multi_scale_cache_points_zarr/writer/value_major.py:108)).
Consequently, a coordinate returned by the sidecar is not independently
self-identifying after it reaches the display pipeline.

That coordinate-only contract is adequate for gene-colored rendering. It is
not adequate for a general transcript visualization cache in which a user can:

- click or hover over one transcript and inspect its source attributes;
- color visible transcripts by quality or another per-point field;
- apply an interactive quality or category filter;
- retain point selections across camera changes and LOD transitions;
- export selected transcript identities back to the canonical table.

## Identity contract

### Dense cache identity

Harpy currently synthesizes point IDs consecutively from canonical physical
source-row order ([Exact annotation](/Users/arne.defauw/VIB/napari_harpy/src/napari_harpy/core/multi_scale_cache_points_zarr/writer/exact.py:605)):

```text
point_id = 0, 1, 2, ..., N_exact - 1
```

Within one validated cache generation, `point_id` is therefore both:

- a stable cache-local identity; and
- a direct row address into exact point-metadata arrays.

No hash table, B-tree, Dask graph, or Parquet predicate scan is required for:

```text
point_metadata/<field>[point_id]
```

The dense cache ID must remain distinct from an optional vendor-provided
`transcript_id`. A vendor identity can be a string, can use a different numeric
domain, and need not be dense. It is an ordinary cold metadata field unless a
separate demonstrated workload requires a source-ID index.

The source signature and cache generation ID bind the dense point-ID assignment
to one canonical source inventory. A rebuilt cache may assign different IDs if
the canonical physical source ordering changes.

### Point identity in the value-major sidecar

The initial metadata-capable schema should persist `point_id` beside
`location` at every value-major level:

```text
value_major/level_<n>/
    location                 float32 [N_level, 2]
    point_id                 uint64  [N_level]
    value_point_indptr       uint64  [N_values + 1]
```

The aligned sidecar `point_id` makes every rendered coordinate independently
self-identifying. It supports picking, asynchronous detail lookup, persistent
selection, LOD reconciliation, and export without reopening tile-major sparse
range metadata.

This is real storage duplication. The existing tile-major `point_id` arrays
occupy approximately 300 MiB across all supplied-cache levels; reordered
sidecar IDs may have a similar or worse compression ratio and must be measured.
The cost is justified initially by the simpler and more portable identity
contract.

A later optimization may replace sidecar point IDs with compact run-level
provenance. A rendered vertex can in principle be mapped through its
`(value_id, manifest_index)` record and offset to the corresponding tile-major
range, whose rows already follow point-ID order. That alternative avoids one ID
per sidecar row but complicates picking and makes readers dependent on a second
physical representation. It is not the initial recommendation.

## Metadata class 1: feature-level metadata

Some columns repeated on every transcript are semantically properties of the
feature or codeword rather than properties of an individual decoded point.
Examples include:

- gene versus non-gene;
- negative-control probe;
- negative-control codeword;
- genomic control;
- unassigned or deprecated codeword;
- panel target name, target identifier, or feature category.

These fields belong in the normalized value vocabulary:

```text
values/
    label                       existing canonical value label
    n_points                    existing exact source-wide count
    feature_type_code           integer code per value
    is_gene                     boolean per value
    is_control                  boolean per value
    feature_type_vocabulary     dictionary or schema metadata
```

A control filter then becomes a value-set operation:

```text
feature metadata predicate
    -> allowed value_ids
    -> existing value_tiles index
    -> value-major coordinate ranges
```

It requires no per-point attribute read and preserves the current fast selected
value path.

The builder must validate that a declared feature-level source column is
constant for every normalized `value_id`. If the value varies between
transcripts with the same label, it cannot be normalized into `values/` and
must be treated as a per-transcript attribute.

Recent Xenium outputs expose fields such as `is_gene` and
`codeword_category` specifically to simplify control filtering. The complete
source still contains one transcript row with fields such as `qv`, `cell_id`,
`overlaps_nucleus`, `fov_name`, and `nucleus_distance`; these do not all have
feature-level scope.

## Metadata class 2: hot per-transcript attributes

Hot attributes are consumed for many or all visible points during ordinary
interaction. Typical uses are coloring, opacity, marker shape, filtering, or
worker-side render selection.

Candidate hot fields include:

- `qv` or another confidence value;
- `overlaps_nucleus`;
- a small assay/category flag that truly varies per transcript;
- `cell_id` only when coloring or filtering by cell is a primary workflow;
- `nucleus_distance` only when interactive coloring or thresholding is required.

These fields must be stored in the same row order and with the same row count as
the coordinates they qualify:

```text
tile-major bucket payload
    location
    point_id
    value_id
    attributes/qv
    attributes/overlaps_nucleus

value-major/level_<n>/
    location
    point_id
    attributes/qv
    attributes/overlaps_nucleus
    value_point_indptr
```

Every point array within one physical ordering uses aligned chunk and shard
boundaries where practical. A reader projects only the fields required by the
current visual operation. Ordinary coordinate rendering does not read `qv`;
quality coloring or filtering reads `location` and the aligned `qv` intervals.

For example:

```text
selected gene and viewport
    -> value/tile records
    -> contiguous sidecar row intervals
       -> location[start:stop]
       -> qv[start:stop]
       -> optional point_id[start:stop]
```

No point-ID metadata gather occurs on this hot path.

Hot attributes on sampled levels are inherited from the retained source
representative identified by `point_id`. They are not recomputed or aggregated
unless a future attribute specification explicitly defines aggregate semantics.

The attribute representation must be compact and explicit:

- categorical strings use a dictionary and fixed-width integer codes;
- booleans may use one byte initially or a documented bit field after evidence;
- nullable fields use a validity array or a declared non-colliding sentinel;
- lossy integer quantization requires declared scale, offset, units, and error
  tolerance;
- source values must not be silently rounded merely to reduce cache size.

## Metadata class 3: cold per-transcript details

Cold fields are normally accessed for one or a small number of transcripts
after a click, hover, selection, or explicit details request. They are not read
for every frame and should not be duplicated across every LOD and physical
ordering.

Examples include:

- vendor `transcript_id`;
- `fov_name`;
- `cell_id` when it is detail-only;
- exact `nucleus_distance`;
- decoding diagnostics and vendor-specific flags;
- provenance needed for export or source reconciliation.

Store cold metadata once in exact dense point-ID order:

```text
point_metadata/
    attributes/
        cell_id_code             [N_exact]
        fov_code                 [N_exact]
        nucleus_distance         [N_exact]
        source_transcript_code   [N_exact] or variable-length encoding
    validity/
        cell_id                  [N_exact]
        nucleus_distance         [N_exact]
    vocabularies/
        cell_id
        fov
    schema                       metadata describing every field
```

A point tooltip follows this route:

```text
picked vertex
    -> aligned value-major point_id
    -> point_metadata/<requested fields>[point_id]
    -> asynchronous detail update
```

The metadata reader keeps a small LRU of recently decoded metadata chunks. For
a multi-point request it sorts IDs, groups them by chunk, reads each touched
chunk once, gathers the requested rows, and restores caller order.

This route does not execute Dask. If a cold field is deliberately omitted from
the derived cache, the fallback is an asynchronous projected read from the
canonical Parquet source using the known source file, row group, and row offset;
it is never a viewport-scale Dask gather. A standalone cache profile should
include every cold field promised by its metadata schema and should not require
the source to remain present for normal inspection.

## Optional class 4: relationship and predicate indexes

A high-cardinality attribute can become a first-class selection dimension. The
main example is `cell_id`: a user may request every transcript assigned to one
or several cells rather than merely inspect the cell assignment of a clicked
transcript.

Neither duplicating every possible column into a new physical ordering nor
performing a global point-ID gather should be automatic. A demonstrated
relationship workload may add an optional inverted index analogous to the gene
catalog:

```text
cell_index/
    vocabulary
    cell_tiles/indptr
    cell_tiles/manifest_index
    cell_tiles/n_points
    optional cell-major payload or bounded range mapping
```

This is a separate schema extension with its own storage and benchmark gate.
The base metadata architecture does not promise fast global predicates over
arbitrary cold columns. Those remain analytical queries against canonical
Parquet.

## Recommended cache architecture

```text
canonical SpatialData points element
    points.parquet                         complete source of truth

derived Zarr visualization cache
    root metadata
        schema version
        source signature
        cache generation ID
        publication state
        attribute schema and profiles

    values/
        label and exact counts
        feature-level metadata
        feature dictionaries

    manifest/
        level and spatial tile facts

    value_tiles/
        value -> level/tile records and counts

    levels/level_<n>/bucket-<id>.zarr/
        tile-major location
        point_id
        value_id
        declared hot attributes
        construction/validation indexes

    value_major/level_<n>/
        location
        point_id
        value_point_indptr
        declared hot attributes

    point_metadata/
        exact point-ID-major cold attributes
        validity arrays
        dictionaries

    optional relationship indexes/
        only for separately accepted interactive dimensions
```

The four normal runtime routes are:

| User operation | Physical route |
|---|---|
| Render all values | Tile-major coordinates and required hot fields |
| Render selected values | Value-major coordinates and required hot fields |
| Inspect one or a few points | Aligned `point_id` followed by dense `point_metadata` lookup |
| Analyze or filter arbitrary source columns | Canonical Parquet outside the frame hot path |

## Attribute schema contract

Every cached attribute must have a versioned declaration. A conceptual
`AttributeSpec` contains at least:

```text
name
source_column or derivation method
scope                  value | hot_point | cold_point | relationship
logical_type
storage_dtype
nullable
missing_value_encoding
dictionary metadata, if categorical
units, if applicable
scale and offset, if quantized
physical representations
LOD inheritance or aggregation semantics
```

Readers discover available attributes from cache metadata and request an
explicit projection. Unknown optional attributes may be ignored; unknown
required semantics or unsupported encodings cause a clear validation failure.

The format must not assume that all assays expose Xenium column names. The
builder maps source-specific columns into cache attribute declarations, while
the cache exposes normalized logical names where a cross-assay meaning is well
defined.

## Filtering and LOD selection

Feature-level filters map directly to selected `value_id` sets, so existing
value/tile counts continue to give exact conservative LOD estimates.

A hot per-point predicate such as `qv >= 20` is initially applied after reading
the aligned attribute rows. LOD selection may use counts before the predicate:

```text
estimated selected-gene count: 60,000
read 60,000 aligned location and qv rows
apply qv >= 20
return 47,000 rows
```

This cannot exceed the point budget, although it may select a coarser level
than necessary and under-fill the renderer. Correctness and bounded work take
priority over an immediately exact predicate-aware estimate.

Later evidence may justify:

- compact per-tile histograms for an all-values quality filter;
- per-range min/max values for cheap rejection;
- a controlled retry at a finer LOD when a predicate leaves substantial budget;
- a dedicated inverted index for a heavily used categorical relationship.

Do not initially persist combined gene-by-tile-by-quality histograms. The
existing cache contains tens of millions of value/tile records; even a small
histogram on every record can add gigabytes of metadata.

## Construction pipeline

Metadata-aware construction should proceed as follows:

```text
validate source schema, metadata declarations, and source inventory
    -> normalize value vocabulary and feature-level attributes
    -> assign dense point IDs from canonical source-row order
    -> stream cold point metadata directly into point-ID-major arrays
    -> carry hot attributes through Exact tile annotation and ordering
    -> retain representative hot attributes through Bridge/spatial sampling
    -> write aligned tile-major arrays
    -> transpose location, point_id, and hot attributes into value-major order
    -> reconcile every row count, point ID, dictionary, validity array, and value
    -> independently validate the staged hierarchy
    -> publish one complete generation atomically
```

The exact cold metadata writer does not require a global shuffle: the dense
point-ID interval for each validated source row group is already known. Hot
attributes do follow the same shuffle and deterministic sorting as coordinates
because alignment, not source order, is their performance contract.

## Storage accounting

Let:

```text
N = exact canonical transcript count
M = sum of stored point rows across all serialized levels
H = total byte width of declared hot attributes per point
C = total byte width of fixed-width cold attributes per exact point
```

The main uncompressed costs are approximately:

```text
feature metadata                 O(number of values)
point-ID-major cold metadata     C * N
value-major point identity       8 * M
hot attributes in both orders    2 * H * M
```

Dictionaries, validity arrays, relationship indexes, Zarr metadata, chunk
boundaries, and compression add or reduce the physical result. These formulas
are capacity-planning guides rather than predictions of compressed bytes.

A field already available in the active display payload does not also need to
be fetched from `point_metadata` for an ordinary tooltip. The point-ID-major
copy is needed only when the field is part of the declared cold detail profile
or must be available independently of a current rendered payload.

## Neuroglancer comparison

Neuroglancer Precomputed Annotations makes every annotation self-contained by
encoding geometry, declared properties, and IDs together and by duplicating
geometry/property data across its spatial, relationship, and annotation-ID
indexes. That is a valid brute-force availability strategy, but it couples the
bytes read for all declared properties and can multiply storage.

Harpy should preserve column projection:

- coordinate-only rendering reads coordinates;
- quality coloring adds only the aligned quality column;
- point inspection reads only requested cold fields;
- optional relationship indexes are introduced only for accepted workloads.

The lesson to retain from Neuroglancer is that fast interaction requires
physical indexes appropriate to the query, not that every property must be
copied into every index.

## Implementation slices

### M0: freeze metadata semantics

- Define `AttributeSpec`, supported scopes, normalized logical types, null
  semantics, dictionary rules, and quantization metadata.
- Distinguish dense cache `point_id` from source `transcript_id`.
- Define the mandatory metadata-capable value-major `point_id` array.
- Define cache capability discovery and projection behavior.

### M1: feature-level metadata

- Accept declared feature-level source columns.
- Validate invariance per normalized `value_id`.
- Persist typed value-level arrays and dictionaries.
- Route feature-category filters through existing selected-value planning.

### M2: point identity and cold details

- Persist sidecar `point_id` for every level.
- Add exact point-ID-major metadata arrays and validity/dictionary support.
- Add projected scalar and chunk-coalesced batch reads with a bounded LRU.
- Add asynchronous point-inspection contracts without Dask.

### M3: one hot attribute

- Use `qv` or a synthetic equivalent as the first hot field.
- Carry it through Exact, sampling, tile-major writing, and value-major
  transposition.
- Add explicit display-field projection and worker-side threshold filtering.
- Preserve coordinate-only reads when the field is not requested.

### M4: integrated acceptance

- Benchmark coordinate-only rendering before and after the schema change.
- Benchmark selected-gene plus quality filtering for 1, 10, and 100 values.
- Benchmark all-values quality filtering for small, medium, and full viewports.
- Measure cold and warm single-point tooltip latency and a 100-point batch.
- Record point chunks, attribute chunks, decoded bytes, physical operations,
  cache startup, peak RSS, build time, and compressed size per attribute.
- Verify selection identity across LOD transitions and cache reopen.

### M5: optional relationship-index decision

- Characterize `cell_id` selection and coloring separately.
- Add an inverted relationship index only if aligned hot storage or cold lookup
  cannot satisfy the measured workflow.
- Keep it out of the base schema until its query and storage benefit are proven.

## Acceptance criteria

The metadata-aware architecture is accepted when:

1. every rendered point in both physical orders can be reconciled to one exact
   cache-local `point_id`;
2. feature-level control filtering performs no per-point metadata reads;
3. a selected-gene hot-attribute filter touches only the aligned coordinate and
   requested attribute chunks, with no point-ID-major bulk gather;
4. point inspection performs no Dask computation and reads only bounded cached
   metadata chunks or one explicit projected canonical-source fallback;
5. coordinate-only rendering does not read unrequested attributes;
6. nulls, dictionaries, units, quantization, and LOD inheritance are validated
   against the attribute schema;
7. every Exact cold-metadata row reconciles to the canonical source row bound to
   the cache generation;
8. actual cache-size increases are reported per physical representation and are
   accepted explicitly rather than hidden inside a general metadata total;
9. incomplete or mismatched attribute representations prevent publication;
10. the canonical SpatialData points element remains unchanged and sufficient
    for arbitrary downstream analysis.

## References

- [10x Genomics: Understanding Xenium outputs](https://www.10xgenomics.com/support/software/xenium-onboard-analysis/latest/analysis/xoa-output-understanding-outputs)
- [10x Genomics: Xenium output format changes and transcript filtering fields](https://www.10xgenomics.com/support/jp/software/xenium-onboard-analysis/latest/release-notes/release-notes-for-xoa)
- [Neuroglancer Precomputed Annotation representation](https://github.com/google/neuroglancer/blob/master/src/datasource/precomputed/annotations.md)
- [Dual-ordering proposal and measured sparse-read amplification](/Users/arne.defauw/VIB/napari_harpy/Roadmap/transcripts_visualization/comments_slow_rendering_26_8_26.md:188)
- [Current Zarr cache construction roadmap](/Users/arne.defauw/VIB/napari_harpy/Roadmap/transcripts_visualization/cache_construction_zarr.md:1)

