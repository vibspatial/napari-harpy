# Points Cache Source Validation

Status: implementation roadmap for the validation block

Roadmap date: 2026-07-29

## Authority and scope

This document expands Phase 0, physical source resolution and validation, from
[multi_tile_cache_29_7_26.md](multi_tile_cache_29_7_26.md).

The parent roadmap remains authoritative for the complete multiscale cache,
runtime store, scheduler, renderer, and napari integration. If this document
conflicts with that roadmap, the parent roadmap wins.

This roadmap covers only the work needed to turn a backed SpatialData points
element and its canonically located physical Parquet dataset into a validated,
immutable input for cache construction.

It does not implement:

- exact or sampled cache levels;
- tile geometry or sampling policy;
- cache metadata, manifests, or publication;
- runtime tile reads;
- viewport planning or scheduling;
- napari, Qt, VisPy, or GPU behavior;
- remote/object-store sources;
- arbitrary transformed Dask dataframe materialization.

Validation is read-only with respect to the canonical SpatialData store.

## Outcome

The validation block must produce a build-ready object with:

- an explicit local Parquet dataset path;
- deterministic source-file ordering;
- a compatible canonical schema for the selected cache columns;
- selected coordinate and categorical-value columns;
- exact source row count;
- deterministic source-file row offsets;
- finite coordinate bounds;
- a deterministic normalized value dictionary and exact value counts;
- a versioned source signature;
- a versioned internal `uint64` point-identity policy.

The validation API returns the deterministic build input directly. The cache
builder must not rediscover the source layout or repeat validation scans.
Performance measurements belong to the V6 benchmark tooling, not to the public
validation result or persisted cache metadata.

## Locked decisions

### Fresh implementation package

All new work lives under:

```text
src/napari_harpy/core/multi_scale_cache_points/
```

The initial validation package is:

```text
src/napari_harpy/core/multi_scale_cache_points/
  __init__.py
  models.py
  source.py
  validation.py
  signature.py
```

The new package may re-express useful behavior from
`src/napari_harpy/_transcript_tiles.py`, but it must not import that module.

The package must not import Qt, napari, or VisPy. `models.py`, `validation.py`,
and `signature.py` should also remain independent of SpatialData and Dask.
SpatialData and Dask awareness is isolated in `source.py`.

### Generic categorical value

The cache treats the third selected source column as a generic categorical
`value`, not specifically as a gene. Transcript datasets normally select the
physical source column named `gene`, so the public default is `value="gene"`.
Callers may select another string-valued categorical column.

Internal models, normalized dictionaries, cache payloads, and sampling code use
`value`, `value_id`, and `value_table`. Transcript-facing UI may present those
values as genes when that is the dataset semantics, but the cache core does not
encode that assumption.

### Canonical physical Parquet source

For the initial SpatialData contract, the resolver derives the physical
Parquet dataset from the backed store path and points-element name. The
resulting source description exposes that concrete path explicitly to the
validator and builder.

It does not inspect or reverse-engineer a Dask expression graph to discover
files. A Dask dataframe may be retained as source context, but the validated
physical file inventory is authoritative for cache construction.

The first implementation supports local filesystem paths only. Remote URIs and
fsspec-backed stores fail with a clear unsupported-source error.

### Parquet source inventory plus one fused scan

Validation performs two kinds of work:

1. construct a private `_ParquetSourceInventory` from Parquet file metadata
   without decoding point columns;
2. perform one bounded streaming scan over the actual `x`, `y`, and `value`
   data.

V5 repeats the same private metadata inspection after the content scan only to
recompute and compare the source signature. It does not perform another content
scan.

Do not implement independent full scans for row count, coordinate validation,
bounds, value validation, and value counts.

The inventory stage handles only structural facts. It does not derive preliminary
bounds, inspect file-metadata null statistics, or decide whether the content scan can
be skipped. The content scan always establishes coordinate validity, exact
bounds, value validity, and normalized value counts.

The authoritative source for each build fact is fixed:

- row count and source-file offsets come from Parquet metadata;
- coordinate bounds and normalized value counts come from the streaming scan;
- stale-source detection uses the versioned source signature.

### No materialization of the full dataframe

PyArrow reads use a configurable maximum batch-row count. Validation retains
only compact aggregates:

- scalar row counts;
- coordinate minima and maxima;
- the value count dictionary;
- file and row-group metadata.

No operation constructs a pandas or Arrow table containing all transcripts.

### Deterministic source-file order and internal point identity

Source files are ordered lexicographically by their normalized
Parquet-dataset-root-relative POSIX path strings. This is ordinary string
ordering, not natural sorting: for example, `part.1.parquet`,
`part.10.parquet`, `part.2.parquet`. The dataset root is the resolved canonical
`points.parquet/` directory. Order is not inherited from a Dask graph or
filesystem directory iteration.

Every source row receives a Harpy-owned `uint64` identity:

```text
point_id = source_file_row_offset + row_position_within_file
```

`source_file_row_offset` is the cumulative row count of all preceding source files.
Validation establishes the ordering and offsets without materializing one id per
row. The builder generates `point_id` batch by batch and propagates it unchanged
through every cache level. It is never written back to canonical SpatialData.

This identity is reproducible only while file inventory and row order remain
stable. That limitation is recorded in the validated source and eventual cache
metadata, and the source signature changes when the validated inventory or
ordering changes.

The public API does not accept a source `transcript_id` or other caller-supplied
identity column. Validation performs no source-id uniqueness scan and needs no
scratch storage for identity validation.

### Value normalization version 1

The method name is:

```text
harpy-string-trim-unicode-white-space-case-sensitive-v1
```

The initial policy freezes the following Unicode `White_Space` code points as
part of version 1:

```text
U+0009-U+000D, U+0020, U+0085, U+00A0, U+1680,
U+2000-U+200A, U+2028-U+2029, U+202F, U+205F, U+3000
```

The implementation constructs one string containing exactly those code points
and passes it as `characters` to `pyarrow.compute.utf8_trim()`. It does not use
`pyarrow.compute.utf8_trim_whitespace()` or Python `str.strip()`, whose behavior
could follow the Unicode version of the installed runtime rather than this
frozen contract.

The remaining policy is:

- accepts Arrow `string`, `large_string`, or dictionary encoding of those
  value types;
- rejects any null logical point value;
- trims only leading and trailing occurrences of the frozen whitespace set;
- rejects values that become empty;
- remains case-sensitive;
- performs no Unicode normalization, case folding, or internal-whitespace
  rewriting;
- orders normalized labels by ascending UTF-8 bytes, independent of locale;
- assigns contiguous `uint32` value ids in that order after the full scan;
- merges raw labels that normalize to the same value;
- never calls Python `hash()`.

For a dictionary array, a null index and an index that references a null
dictionary value are both invalid logical point values. A referenced dictionary
value that normalizes to an empty string is also invalid. Null or
normalized-empty dictionary entries that are not referenced by any row in that
array are ignored. Dictionary values are normalized once per returned dictionary
array, and counts are aggregated from its integer indices rather than by
materializing one Python string per row. Different dictionary entries or raw
labels that normalize to the same label contribute to the same global count.

Numeric and binary value columns are not silently converted to strings in the
first version.

### Build-ready versus private intermediate results

`_ParquetSourceInventory` and any content-scan result are private intermediate
objects. Neither is a validated source.

Only successful completion of every required validation stage may produce
`ValidatedPointsSource`. Partial results must use distinct types and cannot be
passed accidentally to the cache builder.

## Proposed model boundaries

The precise field names may evolve during implementation. The following
separation is the intended contract.

### `PointColumnSelection`

```python
@dataclass(frozen=True)
class PointColumnSelection:
    x: str
    y: str
    value: str
```

It validates non-empty, distinct column names without performing IO.

### `ParquetPointsSource`

```python
@dataclass(frozen=True)
class ParquetPointsSource:
    spatialdata_path: Path
    points_name: str
    columns: PointColumnSelection

    @property
    def element_path(self) -> str:
        return f"points/{self.points_name}"

    @property
    def parquet_path(self) -> Path:
        return self.spatialdata_path / self.element_path / "points.parquet"
```

This means that SpatialData resolution succeeded. It does not mean that the
Parquet dataset or its row values are valid.

The supported SpatialData storage contract places points at
`points/<points_name>/points.parquet` and disallows `/` inside an element name.
Store `spatialdata_path` and `points_name` once; derive both `element_path` and
`parquet_path`. Do not retain independently supplied path state that could
disagree with either input.

`element_path` and `parquet_path` are related but have different meanings:
`element_path` is the SpatialData-root-relative logical element path, such as
`points/transcripts`; `parquet_path` is the concrete filesystem path to that
element's Parquet dataset, such as
`/data/example.zarr/points/transcripts/points.parquet`.

Supporting a standalone or non-canonical Parquet dataset later requires a
separate source contract rather than adding an independently configurable
`parquet_path` to this SpatialData-specific model.

Do not store mutable SpatialData or Dask objects in this immutable physical
source description.

### `ParquetSourceRowGroup`

```python
@dataclass(frozen=True)
class ParquetSourceRowGroup:
    row_count: int
    compressed_size_bytes: int
```

This is compact file-metadata-derived signature material, not point data. Row groups
are retained in validated physical order. `compressed_size_bytes` is the sum of
`total_compressed_size` over every physical column chunk in the row group.

The model is not exported as a top-level public API entry point. It is carried
by source files so V3 can construct the source signature from V2's validated
inventory without reopening Parquet files.

### `ParquetSourceFile`

```python
@dataclass(frozen=True)
class ParquetSourceFile:
    relative_path: str
    size_bytes: int
    modified_time_ns: int | None
    row_count: int
    row_offset: int
    row_groups: tuple[ParquetSourceRowGroup, ...]

    @property
    def row_group_count(self) -> int:
        return len(self.row_groups)
```

Do not store a separate mutable or independently supplied row-group count.
Construction enforces:

- at least one row-group record;
- non-negative row counts and compressed sizes;
- `sum(row_group.row_count) == source_file.row_count`;
- tuple order equal to physical Parquet row-group order.

The row-group records are needed by the source signature. They do not determine
`point_id`, which depends on source-file ordering, source-file offsets, and row
positions.

### `PointsBounds`

```python
@dataclass(frozen=True)
class PointsBounds:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
```

Construction rejects non-finite values and inverted ranges.

### `_ParquetSourceInventory`

```python
@dataclass(frozen=True)
class _ParquetSourceInventory:
    source: ParquetPointsSource
    files: tuple[ParquetSourceFile, ...]
    selected_schema: pa.Schema
    row_count: int
```

This private object is constructed solely from the file inventory and Parquet
file metadata. Its files contain the row offsets and ordered row-group metadata
required by the source-signature method. V3 computes the signature from this
object without reopening files.

It is not accepted by the cache builder.

### `_ScannedPointsContent`

The bounded content scan may return this small private result:

```python
@dataclass(frozen=True)
class _ScannedPointsContent:
    row_count: int
    bounds: PointsBounds
    value_table: pa.Table
```

Invalid-value counters remain transient. If content is invalid, the scan raises
an error and returns no `_ScannedPointsContent`.

### `ValidatedPointsSource`

```python
@dataclass(frozen=True)
class ValidatedPointsSource:
    source: ParquetPointsSource
    files: tuple[ParquetSourceFile, ...]
    selected_schema: pa.Schema
    row_count: int
    bounds: PointsBounds
    value_table: pa.Table
    source_signature: str
    source_signature_method: str
    value_normalization_method: str
    point_id_policy: str
```

`selected_schema` is the canonical Arrow schema for only the caller-selected
cache columns. Its fields retain their physical source names and are ordered by
semantic role: `x`, `y`, `value`. Unselected source columns are not included.

`value_table` uses the cache contract:

```text
value_id: uint32
value: string
n_points: uint64
```

`ValidatedPointsSource` describes the source version identified by its
`source_signature`; it is not a permanent assertion that the physical files
remain unchanged. At construction handoff, the cache builder freshly recomputes
the metadata-only signature before staging and again immediately before
publication. While both guards match, the builder trusts the validated content
facts and does not reconstruct `ValidatedPointsSource`, repeat the content scan,
or independently recompute source row counts, bounds, and normalized value
counts from point data. The parent roadmap specifies the complete construction
flow.

Performance timings, transient counters, machine information, and validation
generation timestamps are not fields of `ValidatedPointsSource`, are not
returned in a public report wrapper, and are not persisted as validation
metadata. Source-file modification times remain deterministic inputs to the
source signature. V6 benchmark tooling owns any measurements needed to evaluate
or optimize validation.

`point_id_policy` stores the complete versioned policy name. V3 does not add a
`PointIdentityPolicy` wrapper model; a string is sufficient for the initial
parameter-free policy and avoids exposing a speculative contract.

## Error contract

Use a small package-specific exception hierarchy rooted in `ValueError`:

```text
PointsSourceValidationError
  ├── PointsSourceResolutionError
  ├── ParquetMetadataValidationError
  └── PointContentValidationError
```

Exceptions should carry a stable short error code and human-readable message.
Path and column context should be included without dumping row data.

Validation is fail-fast. Structural failures stop before content scanning, and
the fused scan raises immediately when a Parquet read, row-count invariant,
coordinate check, or value check fails. It does not continue through the source
to construct exhaustive error counts. A vectorized check may naturally produce
an invalid count for the current batch; an error may include that count only when
it is clearly described as batch-local rather than source-wide.

No `ValidatedPointsSource` is returned alongside an error.

## Public API target

The eventual small public surface is expected to be:

```python
def resolve_spatialdata_points_source(
    sdata: SpatialData,
    points_name: str,
    *,
    x: str = "x",
    y: str = "y",
    value: str = "gene",
) -> ParquetPointsSource: ...


def validate_parquet_points_source(
    source: ParquetPointsSource,
    *,
    max_batch_rows: int = 524_288,
) -> ValidatedPointsSource: ...
```

The returned `ValidatedPointsSource` is passed directly to cache construction.
Validation returns no result wrapper or diagnostics report.

An optional convenience function may later resolve and validate in one call.
The private source-inventory reader and content scanner remain independently testable
without becoming public API.

## Slice overview

Each slice should be independently reviewable, tested, and mergeable.

| Slice | Status | Deliverable | Full row scan |
|---|---|---|---:|
| V0 | Implemented | Package scaffold, immutable models, errors | No |
| V1 | Implemented | Explicit SpatialData-to-Parquet source resolution | No |
| V2 | Implemented | Private Parquet source inventory and schema validation | No |
| V3 | Implemented | Source signature and internal point-identity contract | No |
| V4 | Next | Fused coordinate/value content scan | Once |
| V5 | Planned | Public validation orchestration and build-ready source | No additional scan |
| V6 | Planned | Xenium benchmark, profiling, and hardening | Once per benchmark run |

V0 through V5 deliver the build-ready validation path. The API always uses the
Harpy-owned internal point identity and does not accept a source identity
column.

## Slice V0: scaffold immutable contracts

Implementation status: **complete as of 2026-07-31**. The package scaffold,
minimal immutable contracts, error hierarchy, and focused model tests are in
place. V0 performs no SpatialData, Dask, filesystem, or Parquet IO.

### Goal

Create the new package and type boundaries without performing IO.

### Files

```text
src/napari_harpy/core/multi_scale_cache_points/__init__.py
src/napari_harpy/core/multi_scale_cache_points/models.py
tests/multi_scale_cache_points/test_models.py
```

Add an errors module only if keeping the hierarchy in `models.py` creates
unhelpful coupling:

```text
src/napari_harpy/core/multi_scale_cache_points/errors.py
```

### Implement

- `PointColumnSelection`;
- `ParquetPointsSource`;
- the minimal validation-error base and source-resolution errors required by V1;
- narrow exports from `__init__.py`.

Do not implement or expose source-file, bounds, identity-policy, source-inventory,
scan-result, or validated-source models in V0. Add each contract in the first
slice that uses it, once that slice has established its concrete requirements.

### Tests

- non-empty and distinct selected column names;
- immutable physical-source description without SpatialData or Dask objects;
- minimal validation-error and source-resolution-error inheritance.

### Exit criteria

- imports succeed in the headless test environment;
- the two immutable contracts and minimal errors required by source resolution
  are stable enough for V1;
- no filesystem or dataframe IO occurs.

## Slice V1: explicit SpatialData source resolution

Implementation status: **complete as of 2026-07-31**. The resolver accepts a
backed local SpatialData points element, validates selected Dask metadata
columns, and returns the V0 `ParquetPointsSource` without computing rows,
inspecting Dask graphs, or opening Parquet files.

### Goal

Resolve one backed SpatialData points element to a physical local Parquet
dataset without scanning rows or inspecting Dask internals.

### Files

```text
src/napari_harpy/core/multi_scale_cache_points/source.py
tests/multi_scale_cache_points/test_source.py
```

### Implement

- require a backed SpatialData object with a local path;
- require a string points-element name;
- require the named points element to exist;
- validate that the element is represented as a Dask dataframe, without
  computing it;
- derive `element_path` as `points/<points_name>` according to the supported
  SpatialData points-storage contract;
- reject a points name containing `/`;
- use the derived `parquet_path` property as the physical dataset path;
- validate requested column names against dataframe metadata as an early
  user-facing check;
- return `ParquetPointsSource`;
- reject remote stores explicitly in the first version.

The dataframe metadata check is advisory structural validation. V2 validates
the authoritative physical Parquet schema.

### Do not

- call `.compute()`;
- enumerate Parquet files;
- inspect Dask layers, expressions, tasks, or partition inputs;
- mutate the SpatialData object or dataframe.

### Tests

- resolve `blobs_points` from the real `backed_sdata_blobs` fixture with
  `value="genes"`, and verify the returned contract and canonical Parquet path;
- reject an unbacked SpatialData object;
- reject an unknown points element;
- reject a missing selected column.

These tests cover Harpy's resolver behavior. They do not retest SpatialData's
backing, path, or points-container API, and they do not use mocks to prove the
absence of `.compute()` calls.

### Exit criteria

- the Xenium element resolves to its physical
  `points/transcripts_global_ROI1/points.parquet` dataset;
- resolution time is independent of source row count;
- source resolution contains no cache-building logic.

## Slice V2: deterministic Parquet source inventory

Implementation status: **complete as of 2026-07-31**. The private reader builds
the deterministic file inventory, selected schema, row-group records, and
source-file offsets from local Parquet file metadata without decoding point columns.

The metadata-only acceptance check against
`sdata_xenium_full_data_core.zarr/points/transcripts_global_ROI1/points.parquet`
returned 65 source files, 168 row groups, and 136,578,750 rows with one compatible
selected schema across every file. Timing remains a V6 benchmark concern;
the manual local check only supports the existing performance hypothesis.

### Goal

Construct the private `_ParquetSourceInventory`: a deterministic physical inventory
and selected schema obtained from Parquet metadata without decoding point
columns.

### Files

```text
src/napari_harpy/core/multi_scale_cache_points/validation.py
tests/multi_scale_cache_points/test_inventory.py
```

### Implement

- require the canonical `points.parquet` path to exist and be a directory;
- recursively include every regular file whose name ends in `.parquet`;
- ignore non-Parquet files and the standard `_metadata` and
  `_common_metadata` files;
- treat every other discovered `.parquet` file as a source file rather
  than trying to classify some Parquet files as unrelated;
- normalize source-file paths to Parquet-dataset-root-relative POSIX strings;
- sort those strings lexicographically, without natural sorting;
- open each Parquet source file and read its file metadata with PyArrow;
- require at least one file, row group, and source row;
- validate the selected Arrow fields in every source file;
- require compatible selected Arrow data types and nullability across source files;
- require supported integer/floating coordinate fields and supported
  string/dictionary-string value fields;
- collect file sizes, modification times, row counts, and ordered row-group
  records containing row counts and compressed sizes;
- calculate cumulative source-file row offsets with `uint64` overflow checks;
- return `_ParquetSourceInventory`.

V2 supports only the canonical directory-backed SpatialData source established
by V1. A future standalone or non-canonical Parquet entry point requires its
own source contract; V2 does not add single-file source behavior implicitly.

Because the first version is local-filesystem-only, deterministic inventory
discovery may use `pathlib.Path`. PyArrow opens and validates the file metadata
of each discovered source file. Inventory and ordering remain Harpy contracts rather than incidental
filesystem or PyArrow ordering.

### Schema compatibility

The selected columns are the caller-configured `x`, `y`, and `value` columns.
Compatibility is evaluated only for those columns.

Exact equality of complete source-file schemas is not required. Unselected columns
may be present, absent, or have different types across source files without making
the selected schema incompatible. They are neither read nor retained in
`selected_schema`.

Compatibility compares each selected Arrow field's data type and nullability.
The first version accepts signed and unsigned Arrow integer or floating-point
types for `x` and `y`. It accepts Arrow `string`, `large_string`, or a dictionary
with an integer index and one of those string value types for `value`.

The first version rejects a selected Arrow data-type or nullability change
between source files rather than attempting implicit coercion. Parquet
compression and encoding choices are not schema compatibility criteria. After
all source files pass, `_ParquetSourceInventory` records the selected fields in
canonical semantic order `x`, `y`, `value` as `selected_schema`.

### Tests

- one controlled multi-file, multi-row-group valid dataset covering lexical
  source-file ordering, offsets, selected schema, dictionary-string values, and
  ordered row-group records;
- empty-directory and corrupt-file-metadata rejection;
- missing or incompatible selected-field rejection, while unrelated fields do
  not affect compatibility;
- zero-row source rejection.

These tests exercise Harpy's inventory, compatibility, and aggregation
contracts. They do not retest PyArrow's file-metadata parser, compression
implementation, or every supported dtype combination. The implementation's
metadata-only control flow, rather than mocks around PyArrow internals, keeps
V2 from decoding data pages.

### Exit criteria

- `_ParquetSourceInventory` contains exact file-metadata row counts and deterministic
  offsets;
- no selected data column is decoded;
- the Xenium file-metadata inspection is expected to complete in under one second on
  the reference local SSD, recorded as a hypothesis until benchmarked formally.

## Slice V3: source signature and internal point identity

Implementation status: **complete as of 2026-08-02**. V3 constructs and hashes
the canonical metadata-only source-signature payload directly from V2's
immutable inventory. It freezes the signature-method and point-identity-policy
constants without reopening Parquet files or adding a row-wise identity helper.
Focused tests freeze the canonical bytes and digest and cover the intended
included and excluded source facts.

### Goal

Version the source identity metadata and finalize the Harpy-owned `point_id`
contract without scanning rows.

### Files

```text
src/napari_harpy/core/multi_scale_cache_points/signature.py
tests/multi_scale_cache_points/test_signature.py
```

### Source-signature method

The initial method is:

```text
harpy-parquet-source-inventory-sha256-v1
```

Build this versioned JSON shape:

```json
{
  "method": "harpy-parquet-source-inventory-sha256-v1",
  "element_path": "points/example",
  "columns": [
    {"role": "x", "name": "x", "nullable": true, "type": {"kind": "float", "bit_width": 64}},
    {"role": "y", "name": "y", "nullable": true, "type": {"kind": "float", "bit_width": 64}},
    {"role": "value", "name": "gene", "nullable": true, "type": {"kind": "string", "offset_width": 32}}
  ],
  "files": [
    {
      "path": "part-00000.parquet",
      "size_bytes": 123,
      "modified_time_ns": null,
      "row_count": 10,
      "row_groups": [
        {"row_count": 10, "compressed_size_bytes": 100}
      ]
    }
  ],
  "row_count": 10
}
```

`element_path` and source-file paths are normalized POSIX paths. `columns` always
uses the semantic order `x`, `y`, `value`; `files` and their `row_groups`
use validated physical order. `compressed_size_bytes` is the sum of
`total_compressed_size` over every physical column chunk in that row group.
V3 consumes the immutable row-group records produced by V2 and does not reopen
Parquet files to reconstruct this payload.

Normalized type descriptors are limited to the initially supported selected
types:

```text
signed integer:
  {"kind": "integer", "signed": true, "bit_width": 8|16|32|64}
unsigned integer:
  {"kind": "integer", "signed": false, "bit_width": 8|16|32|64}
floating point:
  {"kind": "float", "bit_width": 16|32|64}
string:
  {"kind": "string", "offset_width": 32|64}
dictionary string:
  {
    "kind": "dictionary",
    "index": <normalized signed/unsigned integer descriptor>,
    "value": <normalized string descriptor>,
    "ordered": true|false
  }
```

Arrow field metadata is excluded. A selected type outside this descriptor
contract is rejected before signature construction.

The v1 payload deliberately excludes absolute source paths, Parquet min/max
statistics, performance measurements, and generation timestamps. In
particular, it does not serialize raw `str(pa.Schema)` or Arrow object
representations whose formatting may change independently of this contract.

Serialize the object exactly as:

```python
canonical_bytes = json.dumps(
    payload,
    sort_keys=True,
    separators=(",", ":"),
    ensure_ascii=False,
    allow_nan=False,
).encode("utf-8")
```

There is no byte-order mark or trailing newline. Arrays retain their specified
order, JSON numbers in this payload are integers, unavailable values use
`null`, and the SHA-256 result is lowercase hexadecimal. Golden tests freeze the
exact canonical bytes as well as the digest.

The SHA-256 digest protects the canonical metadata representation. It is not a
cryptographic hash of every Parquet data page. Documentation and UI must not
claim full content-hash guarantees. A data-page edit that preserves every v1
inventory field can remain undetected; this limitation is part of the method
documentation.

### Internal `point_id`

Expose the signature method and point-identity policy as module constants:

```python
SOURCE_SIGNATURE_METHOD = "harpy-parquet-source-inventory-sha256-v1"
POINT_ID_POLICY = "harpy-source-file-row-offset-uint64-v1"
```

The signature entry point is:

```python
def build_source_signature(
    inventory: _ParquetSourceInventory,
) -> str: ...
```

It returns only the lowercase hexadecimal digest. Callers record
`SOURCE_SIGNATURE_METHOD` separately where the method name is required.

Do not introduce a `PointIdentityPolicy` dataclass or enum in V3. The complete
versioned string is the policy value later stored by `ValidatedPointsSource` and
cache metadata. Do not add a scalar `point_id_for_row` helper: it has no
production consumer, and cache construction must not call Python once per
source row.

The policy fixes the identity formula as:

```text
source_file.row_offset + row_position
```

Here `row_position` is zero-based within the source file. V2 already establishes
contiguous source-file offsets and restricts the complete source row count to at
most `2**64 - 1`, so the formula defines unique ids in `[0, row_count)` without
materializing them. V3 records the versioned policy but adds no row-wise identity
implementation.

The exact-level writer owns the eventual batch-oriented implementation. It will
derive a half-open id range from `source_file.row_offset`, the batch's zero-based
start within that file, and the batch row count, then materialize a bounded
NumPy or Arrow `uint64` array. Define a shared batch helper only if the concrete
writer has more than one real consumer or otherwise benefits from one.

Implement:

- build and hash the canonical source-signature payload;
- expose the signature-method and point-id-policy constants;
- document that file inventory or row-order changes invalidate reproducibility.

The source signature must change when source-file inventory or ordering changes,
preventing internal point ids from being mixed across such source versions.
Validation stores only the policy and source-file offsets. Cache construction
materializes ids batch by batch and propagates them unchanged through all
levels.

### Tests

- repeated source-inventory inspection produces the same signature;
- golden canonical JSON bytes and SHA-256 digest;
- construction order of Python dictionaries does not affect the digest;
- path separator normalization;
- file addition, removal, rename, size, row-group count, schema, or selected-column
  change affects the digest;
- modification-time absence is represented deterministically;
- absolute source-store relocation alone does not enter the canonical payload;
- Parquet statistics are excluded from the canonical payload;
- signature limitations are documented;
- the exact `POINT_ID_POLICY` version string is frozen.

### Exit criteria

- the signature method and point-identity policy have explicit version strings;
- signature computation reads no point data pages;
- the same validated inventory produces the same signature across processes.

## Slice V4: one fused coordinate and value scan

### Goal

Perform all build-ready content validation in one bounded pass over `x`, `y`,
and `value`.

### Files

```text
src/napari_harpy/core/multi_scale_cache_points/validation.py
tests/multi_scale_cache_points/test_scan.py
```

### Implementation shape

The private scan entry point is:

```python
def _scan_points_content(
    inventory: _ParquetSourceInventory,
    *,
    max_batch_rows: int = 524_288,
) -> _ScannedPointsContent: ...
```

Do not accept source paths, selected column names, schemas, file inventories, or
row counts as separate arguments. `_ParquetSourceInventory` is the single source
of those already-validated structural inputs. Keeping one immutable input avoids
independently supplied state that could disagree during the scan.

Iterate deterministic source files, row groups, and batches using PyArrow. The
scan maintains compact accumulators for:

- rows scanned;
- authoritative `x` and `y` minima and maxima;
- normalized value counts;
- normalized value cardinality.

The physical traversal contract is:

1. traverse `inventory.files` in its existing deterministic order;
2. resolve each physical file as
   `inventory.source.parquet_path / source_file.relative_path`;
3. open that file as a `pyarrow.parquet.ParquetFile`;
4. traverse physical row-group indices from zero through
   `source_file.row_group_count - 1` in order;
5. call `ParquetFile.iter_batches()` for exactly one row group at a time, using
   `row_groups=[row_group_index]`, `batch_size=max_batch_rows`, and only the
   selected physical columns in semantic `x`, `y`, `value` order;
6. process each returned record batch before requesting the next one.

The selected column names come from `inventory.source.columns`. Coordinates and
values must be requested together in the same batch traversal. Do not run a
separate coordinate pass, value pass, bounds pass, null-count pass, or
value-count pass, and do not reread any selected column for another aggregate.

The implementation should process Arrow arrays directly. Converting a bounded
batch to NumPy is acceptable where it remains faster and memory-bounded.
Converting complete source files to pandas is not.

The scan reads the actual Parquet data pages for only the selected `x`, `y`,
and `value` columns. It validates each row-group and source-file row count against
`_ParquetSourceInventory` while traversing them, then returns the compact private
`_ScannedPointsContent`. It does not retain point batches after their
aggregates have been updated.

### Coordinate behavior

- accept supported numeric Arrow integer and floating types;
- convert each bounded coordinate batch to `float64` for validation and bounds;
- reject null, NaN, positive infinity, and negative infinity;
- do not require finite source coordinates to be exactly representable as
  `float32`;
- do not quantize or downcast source coordinates during validation.

Scan-derived `float64` bounds are authoritative for the first build-ready mode.
The initial datasets are expected to use intrinsic pixel coordinates, commonly
within roughly a 100,000-by-100,000 space, where global `float32` would already
have negligible rendering error. That rendering assumption does not require V4
to discard source precision: validation retains only four bounds and bounded
working batches, so using `float64` here has no material cache or GPU cost.

Coordinate quantization belongs to cache construction. The initial writer
calculates tile membership and tile-local coordinates from the validated source,
then stores `x_rel` and `y_rel` as `float32`. V4 does not reject integer or
floating coordinates merely because their original values would change in that
later representation.

### Value behavior

- handle plain `string`, `large_string`, and dictionary-encoded string arrays;
- implement the frozen
  `harpy-string-trim-unicode-white-space-case-sensitive-v1` policy;
- for a plain-string array, reject nulls, normalize the complete bounded array
  with `pyarrow.compute.utf8_trim()`, reject normalized-empty values with Arrow
  compute kernels, and run `pyarrow.compute.value_counts()` on that normalized
  array;
- for a dictionary array, reject null indices, normalize its dictionary once,
  run `pyarrow.compute.value_counts()` on its integer indices, validate only the
  referenced normalized dictionary entries, and map the aggregate index counts
  to normalized labels;
- convert only the distinct-label/count output of those Arrow kernels to Python
  while merging it into the source-wide count mapping; never convert a complete
  point-value batch to a Python list or process it row by row;
- ignore invalid unreferenced physical dictionary entries, while rejecting a
  referenced null or normalized-empty entry;
- merge counts from normalization-equivalent labels;
- sort the final normalized labels by ascending `label.encode("utf-8")`, without
  locale-aware comparison;
- assign contiguous `uint32` value ids;
- reject more values than `uint32` can represent;
- build the required Arrow `value_table`;
- require value counts to sum to the exact row count.

These Arrow compute calls are eager, bounded in-memory kernels over the current
record batch. Calling them inside the batch loop does not construct a Dask task
graph, reread Parquet data, or retain preceding batches. Multiple in-memory
kernel passes over the current bounded array are allowed; the one-content-pass
rule prohibits a second traversal of the Parquet source.

Normalization-equivalent raw labels are merged by definition. Validation tests
that behavior, but it does not count, return, or persist normalization
collisions.

### Execution policy

Start with deterministic sequential source-file traversal and PyArrow's bounded
decode threading. Add bounded multi-file concurrency only after profiling
demonstrates a benefit and memory remains predictable.

The default `max_batch_rows` is a build parameter and must be validated as a
positive integer. The largest observed batch must never exceed it.
Counters needed to enforce scan invariants remain internal and transient.

The scan uses deterministic fail-fast behavior:

1. validate `max_batch_rows` before opening a source file;
2. fail immediately on file-open, decode, schema, or structural errors;
3. for each decoded batch, validate selected `x`, then `y`, then `value` before
   committing that batch to successful bounds or value-count aggregates;
4. on the first invalid selected array, raise `PointContentValidationError` and
   request no further batches;
5. include the dataset-relative source file, physical row-group index, selected
   column role and name, and failure category in the error message;
6. reconcile the scanned row count immediately after each row group and source
   file, failing at the first disagreement;
7. return `_ScannedPointsContent` only after the complete source and final value
   counts reconcile successfully.

If a batch-local vectorized check yields an invalid count, the message may
include it for context but must not present it as a complete-source diagnostic.
The scan retains no cross-batch invalid-data counters and returns no partial
content result.

### Tests

- all supported numeric coordinate types;
- coordinate null, NaN, and infinities;
- negative and very large finite coordinates;
- plain-string, large-string, and dictionary values;
- different dictionaries and dictionary orders across row groups;
- every boundary in the frozen whitespace set, plus a nearby non-whitespace
  code point;
- case sensitivity and the absence of Unicode normalization;
- deterministic UTF-8-byte label order;
- null and normalized-empty plain-string values;
- null dictionary indices and referenced null or normalized-empty dictionary
  entries;
- ignored unreferenced null and normalized-empty dictionary entries;
- merging normalization-equivalent raw labels without collision telemetry;
- confirmation that plain-string aggregation converts only distinct batch
  labels, not every point value, to Python;
- value-count and row-count reconciliation;
- empty batches and fragmented chunk boundaries;
- exact batch-size bound;
- row-group, source-file, and total scanned-row reconciliation with file-metadata counts;
- confirmation that only selected columns are read;
- confirmation that there is only one content pass.

### Exit criteria

- a valid source produces exact row count, bounds, and value table;
- every read, structural, coordinate, or value failure stops the scan without
  requesting another batch;
- memory use scales with batch size and value cardinality, not point count;
- no Dask compute occurs.

## Slice V5: validation orchestration and build-ready source

### Goal

Connect resolution, private source-inventory inspection, signature construction, internal
point identity, and fused scanning into the stable validation API that provides
the input to cache construction.

### Files

```text
src/napari_harpy/core/multi_scale_cache_points/validation.py
tests/multi_scale_cache_points/test_validation.py
```

### Implement

The orchestration shape is deliberately small:

```python
inventory_before = _read_parquet_source_inventory(source)
signature_before = build_source_signature(inventory_before)

scanned = _scan_points_content(
    inventory_before,
    max_batch_rows=max_batch_rows,
)

inventory_after = _read_parquet_source_inventory(source)
signature_after = build_source_signature(inventory_after)

if signature_before != signature_after:
    raise PointsSourceValidationError("source changed during validation")

return _construct_validated_points_source(
    inventory=inventory_before,
    scanned=scanned,
    source_signature=signature_before,
)
```

- `validate_parquet_points_source(...)`;
- consistent error translation and context;
- construct `_ParquetSourceInventory` and its source signature before scanning;
- reconcile scanned row-group, source-file, total, and value-table counts with the
  source inventory;
- repeat the private source-inventory inspection after scanning and reject a changed
  source signature;
- final immutable `ValidatedPointsSource` returned directly.

`ValidatedPointsSource` is constructed only after the inventory facts, scanned
content, and source-stability check agree. It combines the source inventory's
physical inventory and selected schema with the scan's exact bounds and value
table. Partial intermediate objects are never returned as build-ready input.

The validated source contains deterministic build facts only. Validation does
not return or persist timings, machine information, transient counters, or a
generation timestamp. V6 measures performance outside the public API.

The first implementation is synchronous and exposes no progress callback or
cancellation-token protocol. Batch boundaries leave room to add orchestration
hooks later without changing `ValidatedPointsSource`, source identity, or cache
format semantics.

### Failure behavior

- resolution or source-metadata-validation errors start no content scan;
- content errors return no build-ready source;
- validation creates no files beside the canonical dataset;
- errors do not mutate the SpatialData object or dataframe.

### Tests

- complete valid workflow using internal point ids;
- one content scan with source-inventory inspection before and after it;
- inventory/scan row-count disagreement;
- row-group and source-file scan-count disagreement;
- value-table count disagreement;
- source-signature change during validation;
- deterministic validated source across runs;
- error code and path/column context;
- headless import and operation;
- no cache files created.

### Exit criteria

- V0 through V5 validate the Xenium acceptance source;
- the returned `ValidatedPointsSource` contains everything needed to start
  exact-level construction;
- rerunning validation produces the same files, offsets, values, bounds,
  point-id policy, and source signature;
- the cache builder does not need the original SpatialData object or Dask graph
  after receiving the validated source.

## Slice V6: Xenium benchmark and hardening

### Acceptance source

```text
sdata_xenium_full_data_core.zarr/
  points/transcripts_global_ROI1/points.parquet
```

Expected investigated values:

- 136,578,750 rows;
- 65 Parquet files;
- 168 row groups;
- 5,122 normalized values from the physical `gene` column;
- `x` bounds approximately `[38.3088, 54047.2059]`;
- `y` bounds approximately `[22.7206, 37581.4706]`;

The dataset path is not hardcoded into normal tests. A benchmark script or
opt-in test receives it through an explicit command-line argument or environment
variable.

### Benchmark entry point

Add an explicit developer tool such as:

```text
scripts/benchmark_multi_scale_cache_points_validation.py
```

It reports machine-readable JSON plus a concise console summary containing:

- package and dependency versions;
- machine and storage context when available;
- cold/warm run label;
- source path and signature method;
- file, row-group, row, batch, and value counts;
- initial inventory, scan, final inventory, and total times;
- largest batch rows and decoded bytes;
- peak resident memory when measured reliably;
- validated-source summary.

The benchmark is read-only. These measurements belong to the benchmark output;
they are not returned by `validate_parquet_points_source(...)` and are not
stored in `ValidatedPointsSource` or cache metadata. Benchmark-only
instrumentation or profiling hooks may be added without widening the public
validation API.

### Initial performance hypotheses

On the investigated local SSD and current development machine:

- each source-inventory inspection should complete in under 1 second;
- full validation should complete in under 30 seconds on warm storage;
- a cold validation target of under 60 seconds is reasonable;
- incremental validation memory should remain below 512 MiB;
- no stage should materialize a complete source file or dataframe.

These are hypotheses, not permanent product guarantees. Record actual results,
hardware, batch size, and cache state before changing them.

### Profile before optimizing

Profile separately:

- file discovery and file-metadata reading;
- coordinate decoding and finiteness checks;
- dictionary versus plain-string value decoding;
- value aggregation;
- Arrow-to-NumPy conversion;
- signature serialization;
- Python allocation and peak memory.

Do not introduce concurrency, custom native code, or persistent validation
sidecars until the profile identifies a material bottleneck.

### Exit criteria

- all expected source facts match;
- repeated runs produce the same validated source and signature;
- performance and memory hypotheses are measured;
- any target miss has a profile-backed explanation and follow-up decision;
- validation is fast enough to proceed to the exact-level writer benchmark.

## Test fixture strategy

### Tiny Parquet fixtures

Build deterministic temporary Parquet datasets that vary:

- file count and filename ordering;
- row-group count and size;
- statistics presence;
- coordinate dtypes and invalid values;
- plain and dictionary value encodings;
- schema mismatch.

Use PyArrow directly so fixture physical properties are explicit.

### SpatialData resolver fixture

Use one small backed SpatialData fixture to verify real element location and
physical path resolution. Resolver unit tests may use minimal fakes for error
branches, but at least one integration test must exercise the supported
SpatialData version.

### Real-data tests

The 136-million-row Xenium validation is opt-in and excluded from the regular
unit suite. CI must not depend on a private local dataset.

## Focused test commands

As slices are added, prefer:

```bash
.venv/bin/pytest -q tests/multi_scale_cache_points/test_models.py
.venv/bin/pytest -q tests/multi_scale_cache_points/test_source.py
.venv/bin/pytest -q tests/multi_scale_cache_points/test_inventory.py
.venv/bin/pytest -q tests/multi_scale_cache_points/test_signature.py
.venv/bin/pytest -q tests/multi_scale_cache_points/test_scan.py
.venv/bin/pytest -q tests/multi_scale_cache_points/test_validation.py
```

Run linting only on the added package and tests.

Keep subsystem-specific shared fixtures in
`tests/multi_scale_cache_points/conftest.py` when they become necessary. The
directory does not need an `__init__.py`.

The legacy `tests/test_transcript_tiles.py` remains independent. New validation
tests must not import its implementation helpers.

## Review gates

### Gate A: after V1

Approve:

- package and model names;
- the explicit local Parquet source boundary;
- SpatialData resolution behavior;
- unsupported remote-source behavior.

### Gate B: after V3

Approve:

- deterministic source-file ordering;
- source-file schema compatibility;
- source-signature method and limitations;
- internal point-identity method.

Changing these later may invalidate built caches.

### Gate C: after V5

Approve:

- the public validation API and private intermediate boundaries;
- build-ready validated source;
- value normalization and value-table contract;
- source-signature and internal point-identity integration;
- error semantics and focused correctness tests.

Gate C freezes the functional validation contract. V6 may benchmark, profile,
harden, and optimize the implementation without casually changing those
semantics. A V6 finding that requires a functional contract change explicitly
reopens the affected Gate C decision.

### Gate D: after V6

Approve:

- the expected Xenium source facts;
- measured initial-inventory, scan, final-inventory, total-time, and peak-memory
  behavior;
- confirmation that batch and materialization bounds hold on the acceptance
  source;
- a profile-backed decision for any missed performance hypothesis;
- readiness to begin the exact-level cache writer.

The performance hypotheses are evaluation targets, not automatic pass/fail
thresholds. Gate D records whether to optimize, accept the measured behavior, or
revise a hypothesis with supporting measurements.

Only after Gate D should exact-level cache construction begin.

## Phase 0 definition of done

The validation block is complete when:

- the new package is independent of `_transcript_tiles.py`;
- a backed local SpatialData points element resolves without Dask graph
  inspection;
- `_ParquetSourceInventory` construction is deterministic and does not decode data
  pages;
- build-ready validation uses one bounded content pass;
- coordinates are finite and bounds are exact;
- values are valid, normalized, deterministic, and exactly counted;
- source signature and internal point-identity methods are versioned;
- the public API accepts no caller-supplied identity column;
- the Xenium acceptance dataset matches expected facts;
- time and memory behavior are measured and documented by V6 benchmark tooling;
- `ValidatedPointsSource` contains every source fact required by cache
  construction;
- no validation operation writes to or mutates the canonical SpatialData store;
- all focused tests pass.

## Immediate next slice

Proceed with V4 only:

1. add the minimal `PointsBounds`, `_ScannedPointsContent`, and content-error
   contracts required by the fused scan;
2. implement one bounded traversal of the selected `x`, `y`, and `value` data
   pages in deterministic inventory order;
3. establish exact finite coordinate bounds and normalized value counts in that
   same traversal;
4. reconcile row-group, source-file, total, and value counts with V2's inventory;
5. add focused scan tests for Harpy's validation, normalization, and bounded-read
   behavior.

Do not implement V5 orchestration, cache writing, sampling, rendering, or
legacy-module migration in this slice.
