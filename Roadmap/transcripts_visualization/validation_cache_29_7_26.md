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
element and an explicit physical Parquet dataset into a validated, immutable
input for cache construction.

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
- compatible physical schemas;
- selected coordinate, gene, and optional transcript-id columns;
- exact source row count;
- deterministic source-fragment row offsets;
- finite coordinate bounds;
- a deterministic normalized gene dictionary and exact gene counts;
- a versioned source signature;
- either a validated source transcript id or a documented fallback identity
  policy;
- evidence describing how important facts were established;
- diagnostics sufficient to benchmark time, rows, batches, and memory.

The cache builder must consume this result directly. It must not rediscover the
source layout or repeat validation scans.

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

### Explicit physical Parquet source

The fast path receives or resolves the physical Parquet dataset explicitly.

It does not inspect or reverse-engineer a Dask expression graph to discover
files. A Dask dataframe may be retained as source context, but the validated
physical file inventory is authoritative for cache construction.

The first implementation supports local filesystem paths only. Remote URIs and
fsspec-backed stores fail with a clear unsupported-source error.

### Footer preflight plus one fused scan

Validation has two execution stages:

1. a metadata-first Parquet footer preflight;
2. one bounded streaming scan over only the columns required for facts that
   footers cannot prove safely.

Do not implement independent full scans for row count, coordinate validation,
bounds, gene validation, and gene counts.

The footer stage may provide preliminary bounds and null information. The fused
scan remains authoritative for coordinate finiteness and gene counts in the
initial build-ready validation mode.

### No materialization of the full dataframe

PyArrow reads use a configurable maximum batch-row count. Validation retains
only compact aggregates:

- scalar row and error counts;
- coordinate minima and maxima;
- the gene count dictionary;
- file and row-group metadata;
- bounded transcript-id uniqueness state when that feature is enabled.

No operation constructs a pandas or Arrow table containing all transcripts.

### Deterministic fragment order and fallback identity

Source files are ordered by their normalized cache-root-relative POSIX paths
using a documented bytewise ordering. The order is not inherited from a Dask
graph or filesystem directory iteration.

For a source without a transcript-id column:

```text
point_identity = fragment_row_offset + row_position_within_fragment
```

`fragment_row_offset` is the cumulative row count of all preceding fragments.
The builder must read fragments and rows in the same validated order.

This identity is reproducible only while file inventory and row order remain
stable. That limitation is recorded in the validated source and eventual cache
metadata.

### Gene normalization version 1

The initial policy:

- accepts Arrow `string`, `large_string`, or dictionary encoding of those
  value types;
- rejects null gene values;
- trims leading and trailing whitespace;
- rejects values that become empty;
- remains case-sensitive;
- assigns gene ids by deterministic normalized-label ordering;
- merges raw labels that normalize to the same value;
- reports the number of raw-label normalization collisions;
- never calls Python `hash()`.

Dictionary values are normalized once per dictionary, not once per row.

Numeric and binary gene columns are not silently converted to strings in the
first version.

### Build-ready versus preflight-only results

A footer preflight result is useful but is not a validated source.

Only successful completion of every required validation stage may produce
`ValidatedPointsSource`. Partial results must use distinct types and cannot be
passed accidentally to the cache builder.

### Exactness over probabilistic acceptance

Validation may use stable hashes for bucketing, signatures, or duplicate
detection, but it must not declare transcript ids unique based only on a
probabilistic sketch or hash comparison.

Approximate cardinality may be recorded as a diagnostic. It is not a
correctness gate.

## Proposed model boundaries

The precise field names may evolve during implementation. The following
separation is the intended contract.

### `PointColumnSelection`

```python
@dataclass(frozen=True)
class PointColumnSelection:
    x: str
    y: str
    gene: str
    transcript_id: str | None
```

It validates non-empty, distinct column names without performing IO.

### `ParquetPointsSource`

```python
@dataclass(frozen=True)
class ParquetPointsSource:
    spatialdata_path: Path
    points_name: str
    element_path: str
    parquet_path: Path
    columns: PointColumnSelection
```

This means that SpatialData resolution succeeded. It does not mean that the
Parquet dataset or its row values are valid.

Do not store mutable SpatialData or Dask objects in this immutable physical
source description.

### `ParquetSourceFragment`

```python
@dataclass(frozen=True)
class ParquetSourceFragment:
    relative_path: str
    size_bytes: int
    modified_time_ns: int | None
    row_count: int
    row_group_count: int
    row_offset: int
```

Additional schema and footer fingerprints may be added if they are required by
the versioned source-signature method.

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

### Evidence

```python
class ValidationEvidence(str, Enum):
    PARQUET_METADATA = "parquet_metadata"
    STREAMING_SCAN = "streaming_scan"
    CALLER_PROVIDED = "caller_provided"
```

Evidence is recorded for facts where the distinction affects trust,
performance, or stale-source reporting.

### `ParquetPointsPreflight`

This internal result contains:

- the unresolved source;
- ordered fragments;
- compatible selected-column schema;
- total footer row count;
- fragment row offsets;
- available row-group statistics;
- preliminary bounds, if trustworthy statistics exist;
- footer and inventory signature material;
- decisions about which content scans remain required.

It is not accepted by the cache builder.

### `ValidatedPointsSource`

```python
@dataclass(frozen=True)
class ValidatedPointsSource:
    source: ParquetPointsSource
    fragments: tuple[ParquetSourceFragment, ...]
    source_schema: pa.Schema
    row_count: int
    bounds: PointsBounds
    gene_table: pa.Table
    source_signature: str
    source_signature_method: str
    gene_normalization_method: str
    identity_policy: PointIdentityPolicy
```

`gene_table` uses the cache contract:

```text
gene_id: uint32
gene: string
n_transcripts: uint64
```

Performance timings and transient counters do not participate in equality,
source identity, or cache identity. They belong in a separate report.

### `PointsSourceValidationReport`

The report may contain:

- footer-preflight elapsed time;
- content-scan elapsed time;
- total elapsed time;
- files, row groups, batches, and rows scanned;
- requested maximum batch rows;
- largest observed batch rows and decoded bytes;
- facts obtained from metadata versus scanning;
- raw gene labels and normalized genes;
- gene normalization collision count;
- temporary spill bytes, when transcript-id uniqueness uses spill storage;
- warnings that do not invalidate the source.

The report is diagnostic and must not be required to interpret cached point
rows.

## Error contract

Use a small package-specific exception hierarchy rooted in `ValueError`:

```text
PointsSourceValidationError
  ├── PointsSourceResolutionError
  ├── ParquetPreflightError
  └── PointContentValidationError
```

Exceptions should carry a stable short error code and human-readable message.
Path and column context should be included without dumping row data.

Structural failures stop before content scanning. The fused scan aggregates
independent value-error counts where practical so a user does not need to
rerun a 136-million-row validation for every invalid column.

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
    gene: str = "gene",
    transcript_id: str | None = None,
) -> ParquetPointsSource: ...


def validate_parquet_points_source(
    source: ParquetPointsSource,
    *,
    max_batch_rows: int = 524_288,
    scratch_path: Path | None = None,
) -> PointsSourceValidationResult: ...
```

Where:

```python
@dataclass(frozen=True)
class PointsSourceValidationResult:
    source: ValidatedPointsSource
    report: PointsSourceValidationReport
```

An optional convenience function may later resolve and validate in one call.
The separate functions remain available so source resolution and footer
preflight can be tested and diagnosed independently.

`scratch_path` is unused for the no-transcript-id Xenium path. It is reserved
for exact, bounded transcript-id uniqueness validation.

## Slice overview

Each slice should be independently reviewable, tested, and mergeable.

| Slice | Deliverable | Full row scan |
|---|---|---:|
| V0 | Package scaffold, immutable models, errors | No |
| V1 | Explicit SpatialData-to-Parquet source resolution | No |
| V2 | Deterministic footer inventory and schema preflight | No |
| V3 | Source signature and fallback identity contract | No |
| V4 | Fused coordinate/gene content scan | Once |
| V5 | Public validation orchestration and reporting | No additional scan |
| V6 | Supplied transcript-id validation | Integrated scan plus bounded spill |
| V7 | Xenium benchmark, profiling, and hardening | Once per benchmark run |

V0 through V5 deliver the build-ready no-source-transcript-id path needed by
the Xenium acceptance dataset. V6 completes the optional supplied-id path
before the public cache builder advertises that capability.

## Slice V0: scaffold immutable contracts

### Goal

Create the new package and type boundaries without performing IO.

### Files

```text
src/napari_harpy/core/multi_scale_cache_points/__init__.py
src/napari_harpy/core/multi_scale_cache_points/models.py
tests/test_multi_scale_cache_points_models.py
```

Add an errors module only if keeping the hierarchy in `models.py` creates
unhelpful coupling:

```text
src/napari_harpy/core/multi_scale_cache_points/errors.py
```

### Implement

- `PointColumnSelection`;
- `ParquetPointsSource`;
- `ParquetSourceFragment`;
- `PointsBounds`;
- evidence and identity-policy enums;
- preflight, validated-source, report, and result shells;
- the exception hierarchy;
- narrow exports from `__init__.py`.

Do not expose models that are only speculative and unused by the next slice.

### Tests

- frozen dataclass behavior;
- non-empty and distinct selected column names;
- finite and ordered bounds;
- valid relative fragment paths;
- non-negative file, row, and offset counts;
- enum serialization values;
- exception inheritance;
- no import of Qt, napari, VisPy, or `_transcript_tiles`.

### Exit criteria

- imports succeed in the headless test environment;
- all contracts required by source resolution are stable enough for V1;
- no filesystem or dataframe IO occurs.

## Slice V1: explicit SpatialData source resolution

### Goal

Resolve one backed SpatialData points element to a physical local Parquet
dataset without scanning rows or inspecting Dask internals.

### Files

```text
src/napari_harpy/core/multi_scale_cache_points/source.py
tests/test_multi_scale_cache_points_source.py
```

### Implement

- require a backed SpatialData object with a local path;
- require a string points-element name;
- require the named points element to exist;
- validate that the element is represented as a Dask dataframe, without
  computing it;
- use `sdata.locate_element(points)` and require one resolved element path;
- construct the physical `points.parquet` path from the resolved element path;
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
- assume the element path is `points/<points_name>`;
- mutate the SpatialData object or dataframe.

### Tests

- valid backed local SpatialData points element;
- default and explicitly selected columns;
- unbacked SpatialData;
- missing points element;
- non-Dask points value;
- no located path and multiple located paths;
- missing dataframe metadata column;
- custom nested resolved element path;
- unsupported remote/non-local source;
- proof that no Dask compute is called.

### Exit criteria

- the Xenium element resolves to its physical
  `points/transcripts_global_ROI1/points.parquet` dataset;
- resolution time is independent of source row count;
- source resolution contains no cache-building logic.

## Slice V2: deterministic Parquet footer preflight

### Goal

Produce a deterministic physical inventory and validate everything available
from Parquet metadata without decoding point columns.

### Files

```text
src/napari_harpy/core/multi_scale_cache_points/validation.py
tests/test_multi_scale_cache_points_validation.py
```

### Implement

- validate that the Parquet path exists and is a directory or supported single
  Parquet file;
- discover physical data files while excluding metadata and unrelated files;
- normalize cache-root-relative POSIX paths;
- sort fragments deterministically;
- open each footer with PyArrow;
- require at least one file, row group, and source row;
- validate selected columns in every physical schema;
- validate compatible Arrow logical types across fragments;
- require numeric `x` and `y`;
- require supported gene string/dictionary types;
- validate the optional transcript-id physical type against its initial policy;
- collect file sizes, modification times, row counts, and row-group counts;
- calculate cumulative fragment row offsets with `uint64` overflow checks;
- collect available null counts and coordinate statistics;
- calculate preliminary bounds only when all required statistics are present
  and trustworthy;
- return `ParquetPointsPreflight`.

File discovery should use supported PyArrow dataset APIs where practical. The
normalized inventory and ordering remain Harpy contracts rather than incidental
PyArrow ordering.

### Schema compatibility

Exact equality of unrelated source columns is not required. Compatibility is
evaluated only for selected columns and any fields needed to establish source
identity.

The first version rejects a selected column whose physical/logical type changes
between fragments rather than attempting implicit coercion.

### Tests

- directory and single-file datasets;
- empty dataset directory;
- corrupt footer;
- zero-row and zero-row-group cases;
- deterministic ordering with names such as `part.1`, `part.10`, `part.2`;
- metadata files excluded from fragments;
- missing selected column in one fragment;
- incompatible coordinate or gene types;
- dictionary-encoded strings;
- complete, incomplete, and absent statistics;
- valid and invalid footer-derived bounds;
- deterministic offsets;
- offset overflow;
- no row-group data-page reads.

### Exit criteria

- preflight returns exact footer row counts and deterministic offsets;
- no selected data column is decoded;
- the Xenium footer preflight is expected to complete in under one second on
  the reference local SSD, recorded as a hypothesis until benchmarked formally.

## Slice V3: source signature and fallback identity

### Goal

Version the source identity evidence and finalize the no-source-transcript-id
identity contract without scanning rows.

### Files

```text
src/napari_harpy/core/multi_scale_cache_points/signature.py
tests/test_multi_scale_cache_points_signature.py
```

### Source-signature method

The initial candidate is:

```text
harpy-parquet-footer-inventory-sha256-v1
```

Hash a canonical UTF-8 JSON representation containing at least:

- signature method;
- resolved points-element path;
- selected column names;
- selected-column Arrow schema fingerprint;
- deterministically ordered relative file paths;
- file sizes and available nanosecond modification times;
- footer row and row-group counts;
- relevant row-group compressed sizes and available selected-column
  statistics;
- total row count.

JSON key ordering, separators, number representation, path normalization, and
missing-value representation are part of the versioned method.

The SHA-256 digest protects the canonical metadata representation. It is not a
cryptographic hash of every Parquet data page. Documentation and UI must not
claim full content-hash guarantees.

### Fallback identity

Implement helpers that:

- map `(fragment index, row position)` to cumulative `uint64` identity;
- reject out-of-range row positions;
- validate identity coverage at fragment boundaries;
- expose the identity-policy name and version;
- document that file inventory or row-order changes invalidate reproducibility.

The source signature must change when fragment inventory or ordering changes,
preventing fallback identities from being mixed across such source versions.

### Tests

- repeated preflight produces the same signature;
- construction order of Python dictionaries does not affect the digest;
- path separator normalization;
- file addition, removal, rename, size, footer count, schema, or selected-column
  change affects the digest;
- modification-time absence is represented deterministically;
- signature limitations are documented;
- first, last, and cross-fragment fallback identities;
- identity uniqueness across fragment boundaries;
- `uint64` overflow rejection.

### Exit criteria

- signature and fallback identity methods have explicit version strings;
- signature computation reads no point data pages;
- the same validated inventory produces the same signature across processes.

## Slice V4: one fused coordinate and gene scan

### Goal

Perform all build-ready validation needed for the no-source-transcript-id path
in one bounded pass over `x`, `y`, and `gene`.

### Implementation shape

Iterate deterministic fragments, row groups, and batches using PyArrow. The
scan maintains compact accumulators for:

- rows scanned;
- missing and non-finite `x` values;
- missing and non-finite `y` values;
- authoritative `x` and `y` minima and maxima;
- missing and normalized-empty genes;
- normalized gene counts;
- raw and normalized gene cardinalities;
- gene normalization collisions;
- batch and decoded-byte diagnostics.

The implementation should process Arrow arrays directly. Converting a bounded
batch to NumPy is acceptable where it remains faster and memory-bounded.
Converting complete fragments to pandas is not.

### Coordinate behavior

- accept supported numeric Arrow integer and floating types;
- convert safely to `float64` for validation and bounds;
- reject null, NaN, positive infinity, and negative infinity;
- do not downcast canonical coordinates during validation;
- compare scan bounds with footer bounds and report disagreement.

Scan-derived bounds are authoritative for the first build-ready mode.

### Gene behavior

- handle plain and dictionary-encoded string arrays;
- validate nulls before normalization;
- normalize dictionary values once per encountered dictionary;
- aggregate counts by normalized label;
- sort normalized labels deterministically;
- assign contiguous `uint32` gene ids;
- reject more genes than `uint32` can represent;
- build the required Arrow `gene_table`;
- require gene counts to sum to the exact row count.

### Execution policy

Start with deterministic sequential fragment traversal and PyArrow's bounded
decode threading. Add bounded multi-file concurrency only after profiling
demonstrates a benefit and memory remains predictable.

The default `max_batch_rows` is a build parameter and must be validated as a
positive integer. The largest observed batch must never exceed it.

### Tests

- all supported numeric coordinate types;
- coordinate null, NaN, and infinities;
- negative and very large finite coordinates;
- plain-string, large-string, and dictionary genes;
- different dictionaries and dictionary orders across row groups;
- whitespace trimming and case sensitivity;
- null and normalized-empty genes;
- normalized-label collisions;
- gene-count and row-count reconciliation;
- empty batches and fragmented chunk boundaries;
- exact batch-size bound;
- scan-derived bounds versus footer statistics;
- evidence and report counters;
- confirmation that only selected columns are read;
- confirmation that there is only one content pass.

### Exit criteria

- a valid no-id source produces exact row count, bounds, and gene table;
- every independent invalid-value count is reported after the same scan;
- memory use scales with batch size and gene cardinality, not transcript count;
- no Dask compute occurs.

## Slice V5: validation orchestration and build-ready result

### Goal

Connect resolution, preflight, signature, fallback identity, and fused scanning
into the stable API consumed by cache construction.

### Implement

- `validate_parquet_points_source(...)`;
- `PointsSourceValidationResult`;
- consistent error translation and context;
- validation-stage timings and counters;
- reconciliation between footer and scan row counts;
- reconciliation between footer and scan bounds;
- final immutable `ValidatedPointsSource`;
- optional non-Qt progress callbacks at stage boundaries;
- safe cancellation points between batches if cancellation is included.

The validated source contains deterministic facts only. Timing, machine, and
progress information stays in the report.

### Failure behavior

- resolution or preflight errors start no content scan;
- content errors return no build-ready source;
- cancellation returns no build-ready source;
- validation creates no files beside the canonical dataset;
- errors do not mutate the SpatialData object or dataframe.

### Tests

- complete valid no-id workflow;
- every stage called exactly once;
- footer/scan row-count disagreement;
- footer/scan bound disagreement policy;
- deterministic validated source across runs;
- diagnostic timings excluded from identity and equality;
- error code and path/column context;
- progress ordering, if exposed;
- cancellation cleanup, if exposed;
- headless import and operation;
- no cache files created.

### Exit criteria

- V0 through V5 validate the Xenium acceptance source;
- the result contains everything needed to start exact-level construction;
- rerunning validation produces the same fragments, offsets, genes, bounds,
  identity policy, and source signature;
- the cache builder does not need the original SpatialData object or Dask graph
  after receiving the result.

## Slice V6: supplied transcript-id validation

### Goal

Support a caller-selected transcript-id column without sacrificing exactness or
bounded memory.

This slice is not required for the initial Xenium dataset, which has no supplied
globally unique transcript id. It is required before the public builder claims
general `transcript_id=` support.

### Type and encoding policy

Initially support:

- signed and unsigned Arrow integers;
- Arrow string and large-string;
- dictionary encoding of supported strings.

Define a versioned canonical byte encoding for hashing and later sampling.
Reject nulls and unsupported mixed physical types.

### Exact uniqueness

An in-memory Python set of all ids is not acceptable for unbounded sources.
Dask `nunique()` and a global Dask shuffle are not part of the Parquet fast
path.

The first exact bounded approach should use deterministic hash buckets with
temporary spill storage:

1. canonicalize ids batch by batch;
2. compute a named stable wide hash;
3. assign each id to a deterministic bucket;
4. write bounded bucket chunks under an explicit scratch directory;
5. process one bucket at a time;
6. compare canonical values, not hashes alone, to detect duplicates;
7. clean scratch data after success, failure, or cancellation.

Bucket count and memory targets are configuration and benchmark decisions, not
cache-format semantics.

If an alternative exact bounded algorithm is selected, document it before
implementation.

### Tests

- integer and string ids;
- dictionary strings;
- null and unsupported ids;
- duplicates within one batch, across batches, files, and buckets;
- deliberately forced hash collisions;
- canonical encoding stability;
- scratch cleanup after success and failure;
- bounded per-bucket memory;
- no false duplicate or false unique result.

### Exit criteria

- supplied ids are proven non-null and unique exactly;
- no correctness decision relies only on hash uniqueness;
- peak memory is independent of total id count within the configured bucket
  envelope;
- scratch lifecycle is safe and documented.

## Slice V7: Xenium benchmark and hardening

### Acceptance source

```text
sdata_xenium_full_data_core.zarr/
  points/transcripts_global_ROI1/points.parquet
```

Expected investigated values:

- 136,578,750 rows;
- 65 Parquet files;
- 168 row groups;
- 5,122 normalized genes;
- `x` bounds approximately `[38.3088, 54047.2059]`;
- `y` bounds approximately `[22.7206, 37581.4706]`;
- no supplied globally unique transcript-id column.

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
- file, row-group, row, batch, and gene counts;
- footer, scan, and total times;
- largest batch rows and decoded bytes;
- peak resident memory when measured reliably;
- validation result summary.

The benchmark is read-only.

### Initial performance hypotheses

On the investigated local SSD and current development machine:

- footer preflight should complete in under 1 second;
- full no-id validation should complete in under 30 seconds on warm storage;
- a cold validation target of under 60 seconds is reasonable;
- incremental validation memory should remain below 512 MiB;
- no stage should materialize a complete fragment or dataframe.

These are hypotheses, not permanent product guarantees. Record actual results,
hardware, batch size, and cache state before changing them.

### Profile before optimizing

Profile separately:

- file discovery and footer opening;
- coordinate decoding and finiteness checks;
- dictionary versus plain-string gene decoding;
- gene aggregation;
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
- plain and dictionary gene encodings;
- schema mismatch;
- optional transcript ids.

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
.venv/bin/pytest -q tests/test_multi_scale_cache_points_models.py
.venv/bin/pytest -q tests/test_multi_scale_cache_points_source.py
.venv/bin/pytest -q tests/test_multi_scale_cache_points_validation.py
.venv/bin/pytest -q tests/test_multi_scale_cache_points_signature.py
```

Run linting only on the added package and tests.

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

- deterministic fragment ordering;
- footer schema compatibility;
- source-signature method and limitations;
- fallback identity method.

Changing these later may invalidate built caches.

### Gate C: after V5

Approve:

- build-ready validation result;
- gene normalization and gene-table contract;
- evidence and error semantics;
- measured Xenium no-id performance.

Only after Gate C should exact-level cache construction begin.

### Gate D: after V6

Approve:

- supplied transcript-id types;
- canonical id encoding;
- exact bounded uniqueness algorithm;
- scratch-space behavior.

Only after Gate D should the public cache builder advertise supplied
transcript-id support.

## Phase 0 definition of done

The validation block is complete when:

- the new package is independent of `_transcript_tiles.py`;
- a backed local SpatialData points element resolves without Dask graph
  inspection;
- footer preflight is deterministic and does not decode data pages;
- build-ready validation uses one bounded content pass for the no-id path;
- coordinates are finite and bounds are exact;
- genes are valid, normalized, deterministic, and exactly counted;
- source signature and fallback identity methods are versioned;
- optional supplied ids are either exactly validated or explicitly unsupported
  by the exposed API;
- the Xenium acceptance dataset matches expected facts;
- time and memory behavior are measured and documented;
- the result contains every source fact required by cache construction;
- no validation operation writes to or mutates the canonical SpatialData store;
- all focused tests pass.

## Immediate next slice

Begin with V0 only:

1. create `core/multi_scale_cache_points/`;
2. add the minimal immutable model and error contracts used by V1;
3. keep `__init__.py` exports narrow;
4. add focused headless model tests;
5. review the contracts before implementing SpatialData or Parquet IO.

Do not implement cache writing, sampling, rendering, or legacy-module migration
in this slice.
