# Spatial Query Annotation Using Canonical Centers

## Status

Final product specification and implementation plan.

This document supersedes the raster-overlap query algorithm described in
spatial_query.md. It retains the agreed user interface, table mutation,
overwrite protection, undo, dirty-state, persistence, reload, asynchronous
execution, and cross-widget behavior, but changes the spatial membership rule
and execution strategy.

The production query is based on one canonical center per labels instance:

- canonical centers are stored row-aligned in
  AnnData.obsm["spatial_canonical"];
- accompanying Harpy-owned metadata describes their meaning, provenance,
  coverage, and validity;
- if a valid canonical-center cache exists for the selected labels region, the
  query reuses it and reads no label pixels;
- if the cache is absent, incomplete for the selected region, or known to be
  stale, the centers are calculated from the zarr-backed scale0 labels through
  Harpy's lazy RasterAggregator before the query is evaluated;
- all polygons in the selected Shapes element form one annotation region;
- an instance matches when its canonical center is inside or on the boundary of
  that region.

The exact any-pixel-overlap algorithm in spatial_query.md is not an alternative
execution path for this feature. It may remain as design history or become a
separate future query mode, but implementations of this specification must not
silently fall back to it.

## Product Goal

Let a user bulk-annotate segmented objects by selecting an existing polygon
annotation:

1. select one Shapes element created by, or valid for, the Shapes Annotation
   widget;
2. treat all polygons in that Shapes element as one annotation region;
3. select a labels element in the same SpatialData object and selected
   coordinate system;
4. select a table that annotates that labels element;
5. select an existing compatible AnnData.obs column or configure a new one,
   defaulting to spatial_annotation;
6. run a centroid-containment spatial query;
7. transparently calculate and cache canonical centers first when they are not
   already valid and reusable;
8. review the affected instances and any values that will be overwritten;
9. provide the annotation value, defaulting to the Shapes element name;
10. apply the value to the matching table rows;
11. explicitly write or reload the shared table state when working with the
    backed zarr store.

The workflow targets zarr-backed SpatialData. The highest-resolution labels
level, scale0, is always used and is always expected to be backed by a lazy Dask
array. NumPy-backed labels and lower pyramid levels are outside this contract.

The feature must be suitable for large production datasets. It must not load
the complete labels raster into RAM, freeze the napari UI, silently replace
user data, trust ambiguous coordinate metadata, accept stale asynchronous
results, or confuse in-memory state with persisted state.

## Product Principles

- **Fast repeated queries:** the expensive labels scan is amortized by a
  validated, persisted canonical-center cache.
- **Out of core cache generation:** center calculation operates through Dask
  and RasterAggregator over zarr-backed scale0 labels. The full labels raster is
  never materialized in RAM.
- **Precisely defined membership:** the result follows centroid containment,
  including a documented boundary rule. It is not label-pixel overlap.
- **Explicit coordinate frames:** canonical centers are stored in the intrinsic
  coordinate frame of their source labels element. Query geometry is
  transformed into that frame.
- **Validated reuse:** a matrix named spatial_canonical is not trusted without
  matching metadata and source/coverage validation.
- **Preview before annotation mutation:** spatial querying does not change the
  annotation column. The user reviews affected and overwritten counts before
  applying.
- **No silent data loss:** overwriting values, replacing an ambiguous cache,
  reloading a dirty table, and leaving a dirty dataset are explicit decisions.
- **One shared table state:** all Harpy widgets see the same in-memory AnnData
  and the same per-table dirty marker.
- **Deterministic and testable:** identical geometry, transforms, canonical
  centers, linkage metadata, and table state produce identical results.
- **Accessible:** all important states, warnings, progress, and errors are
  conveyed textually and are keyboard accessible.

## Terminology and Normative Decisions

### Annotation

An annotation is one selected SpatialData Shapes element. Every geometry row in
the element contributes to the same annotation region. This workflow does not
offer per-polygon row selection.

The Shapes element must satisfy the existing Shapes Annotation edit-validity
contract:

- it is a GeoDataFrame with an active geometry column;
- it contains at least one geometry row;
- every geometry is a two-dimensional Shapely Polygon;
- every polygon is non-empty, valid, finite, and has positive area;
- polygon holes are supported.

Rectangles and ellipses drawn in napari are eligible after the Shapes Annotation
save path converts them to valid polygons. Lines, paths, points, unpolygonized
circles, MultiPolygon values, and geometry collections are not eligible unless
the shared Shapes Annotation validity contract is extended for them. Spatial
Query must not define a conflicting geometry-validity rule.

The effective annotation region is the geometric union of all polygon rows.
Overlapping polygons do not duplicate results. Disjoint polygons are allowed.
Holes remain excluded according to Shapely/OGC polygon semantics.

### Labels instance

A labels instance is one positive, non-background integer value in the selected
labels raster. Label value zero is background and never represents an instance.

The linked table is the searchable and annotatable universe for the centroid
query. Each eligible table row identifies an instance through the selected
region key and instance key. A labels value for which no table row exists has
no row-aligned location in spatial_canonical and cannot be returned or
annotated by this workflow.

This differs from the raster-overlap design, which could first discover labels
and then report labels missing from the table. The centroid query must not claim
to detect or count labels that are absent from the selected table.

### Canonical center

The canonical center is the center of mass of all scale0 pixels belonging to a
labels instance, using equal weight for every pixel:

    center_x = mean(pixel column indices)
    center_y = mean(pixel row indices)

For two-dimensional labels, the stored coordinate order is always x, y.
RasterAggregator may internally use z, y, x; the singleton z coordinate is
discarded and y, x is explicitly reordered to x, y before storage.

Pixel indices denote pixel centers in the same convention used by Harpy's
RasterAggregator: x=0, y=0 is the center represented by array position [0, 0].
There is no implicit addition of 0.5.

The term center of mass is mathematically precise because label pixels have
uniform mass. The UI may use the more familiar word centroid, but metadata must
record method="center_of_mass".

The canonical center is a representative point, not a promise that the point
lies on a label pixel. For a concave, disconnected, or ring-shaped instance,
the center of mass can lie in background or outside the instance's pixel
support. Spatial membership is nevertheless determined solely by that
representative point.

### Spatial predicate: canonical center inside

An instance matches when its canonical center intersects the effective
annotation region:

    inside = shapely.intersects_xy(region_in_labels, center_x, center_y)

For a point and polygon, this includes points in the polygon interior and on its
exterior or interior-ring boundary. Points strictly inside a polygon hole are
excluded.

Consequences:

- a label can overlap the annotation but be excluded when its canonical center
  is outside;
- a label can have no pixels inside the annotation but be included when its
  canonical center is inside;
- a center on a polygon boundary is included;
- a center inside a hole is excluded;
- overlapping polygons never duplicate a result;
- every table instance is evaluated at most once.

The UI must describe the rule as Canonical center inside annotation, with a
tooltip explaining the boundary behavior. It must not describe the result as
any-pixel overlap, full label containment, or percentage overlap.

Other predicates may be added later behind explicit strategy identifiers. They
must never alter the meaning of this predicate without a metadata/query schema
version change.

### Source-of-truth rule

The query reads the selected Shapes element and labels/table binding from the
current in-memory SpatialData object, not arbitrary similarly named napari
layers.

If Shapes Annotation has an open dirty edit session for the selected Shapes
element, Spatial Query must not query the last stored geometry silently. Run is
blocked with guidance to save or discard the shape edits. After a successful
Shapes Annotation save, the Spatial Query selections refresh and the new
in-memory geometry becomes queryable.

The current AnnData object is authoritative for spatial_canonical and its
metadata. Persisted zarr state becomes authoritative again only after an
explicit reload.

### Coordinate-system rule

Canonical centers are stored in each source labels element's intrinsic scale0
x, y coordinate frame. They are not stored in the currently selected named
coordinate system.

The Shapes and labels elements must:

- belong to the same SpatialData object;
- both be available in the selected coordinate system;
- expose supported, finite, invertible two-dimensional transformations between
  their intrinsic frames and that coordinate system.

Let M_shapes_to_cs map Shapes intrinsic x, y coordinates to the selected
coordinate system, and M_labels_to_cs map labels intrinsic x, y coordinates to
the same coordinate system. The query geometry is transformed using:

    M_shapes_to_labels =
        inverse(M_labels_to_cs) @ M_shapes_to_cs

The unioned annotation is transformed with M_shapes_to_labels before point
membership is evaluated. Matrix conversion must explicitly use x, y order for
Shapely coordinates. Napari's array-axis y, x order must not leak into this
calculation.

Identity, translation, anisotropic scale, rotation, reflection, and other
supported invertible 2D affine compositions must work. A transform must not be
assumed to be identity merely because both elements are displayed together.

Storing intrinsic coordinates has two important effects:

- changing the labels-to-global transformation does not invalidate the
  calculated centers;
- the current SpatialData transformation graph, rather than a copied affine in
  AnnData metadata, remains authoritative at query time.

The canonical-center metadata therefore does not store a list of coordinate
systems or one affine per coordinate system. If coordinates in a named
coordinate system are ever materialized for another feature, they require a
separate obsm key and separate metadata. They must not change the semantics of
spatial_canonical.

This release supports 2D x/y labels queries. Labels with a spatial z dimension
are rejected rather than implicitly querying the current napari slice or
projecting a volume.

### Table binding and row identity

Selectable tables are discovered using the existing annotating-table helper.
Before cache inspection or query execution, the selected binding must pass the
shared table-binding validator and the additional canonical-center
requirements.

The table row for an instance is identified by both linkage keys:

    obs[region_key] == selected_labels_name
    and
    obs[instance_key] == instance_id

Matching by obs row order alone or by obs_names alone is forbidden.

Within the selected labels region, instance-key values must be non-missing,
positive, integer-like, and unique. Booleans, fractional numbers, non-finite
numbers, and strings that only resemble numbers are rejected rather than
silently coerced. Duplicate instance IDs in another region are allowed.

Canonical-center calculation uses the validated instance IDs from the selected
table region as RasterAggregator's requested index. This avoids a separate
global unique-label discovery pass. The calculated result is joined back to
AnnData rows by instance ID; code must not assume that RasterAggregator returns
the same row order as AnnData.

If a requested table instance has no pixels in the selected labels element, its
center is undefined. This is a table/labels binding inconsistency. Cache
installation and the spatial query fail with an actionable error; incomplete
or infinite center values must not be silently stored.

## Scope

### In scope

- one loaded, zarr-backed SpatialData object at a time;
- one selected coordinate system;
- one valid polygon Shapes element, with all rows forming one region;
- one two-dimensional labels element whose scale0 data is a zarr-backed Dask
  array;
- one table linked to that labels element;
- creation, validation, reuse, refresh, and persistence of
  AnnData.obsm["spatial_canonical"];
- center-of-mass calculation through Harpy RasterAggregator;
- canonical-center containment using vectorized Shapely;
- assignment to one existing compatible obs column or one newly created column;
- mandatory overwrite disclosure and confirmation;
- asynchronous center calculation and query execution, cancellation, progress,
  and stale-result protection;
- per-table shared clean/dirty state;
- writing the complete current table state, including canonical centers and
  their metadata, to the backed zarr store;
- reloading the current table state from zarr with dirty-state protection;
- undo of the most recently applied spatial annotation while its source
  binding remains valid;
- cross-widget refresh after cache installation, annotation mutation, write,
  reload, or Shapes element write.

### Non-goals

- modifying labels pixels or Shapes geometry;
- querying NumPy-backed labels or lower-resolution pyramid levels;
- creating missing table rows or a new linked table;
- returning labels instances that have no linked table row;
- querying arbitrary unregistered napari Shapes layers;
- querying unsaved Shapes Annotation edits;
- per-polygon annotation values within one Shapes element;
- 3D center calculation/query semantics;
- any-pixel overlap, full-label containment, or percentage-overlap predicates;
- boolean combinations selected interactively across Shapes elements;
- automatically writing to zarr after every cache or annotation change;
- automatically recoloring labels;
- treating raw out-of-band zarr mutations that bypass the required labels
  revision contract as safely detectable;
- using an existing obsm["spatial"] array without verified canonical metadata.

## Canonical-Center Data Contract

### Storage locations

Coordinates are stored at:

    adata.obsm["spatial_canonical"]

Metadata is stored in a Harpy-owned registry at:

    adata.uns["spatial_coordinates"]["spatial_canonical"]

The registry pattern leaves room for future named spatial coordinate arrays
without overloading feature-matrix metadata.

The matrix:

- has shape n_obs by 2;
- is a dense floating-point NumPy-compatible array;
- uses columns x, y in that order;
- is row-aligned through AnnData's obsm contract;
- contains finite coordinates for every row covered by a valid region entry;
- may contain NaN for rows belonging to table regions whose centers have not
  yet been calculated;
- must not contain inf;
- must not be sparse.

spatial_canonical is a reserved Harpy key. An existing array at that key without
valid matching metadata is ambiguous user data and is never silently trusted or
overwritten.

### Metadata schema

A representative schema is:

    adata.uns["spatial_coordinates"]["spatial_canonical"] = {
        "schema_version": 1,
        "obsm_key": "spatial_canonical",
        "axes": ["x", "y"],
        "dtype": "float64",
        "region_key": "region",
        "instance_key": "instance_id",
        "regions": {
            "nuclei": {
                "source_element": "nuclei",
                "source_element_type": "labels",
                "source_scale": "scale0",
                "coordinate_frame": {
                    "type": "element_intrinsic",
                    "element": "nuclei",
                    "axes": ["x", "y"],
                },
                "calculation": {
                    "method": "center_of_mass",
                    "weighting": "uniform_label_pixels",
                    "background_value": 0,
                    "pixel_coordinate_convention":
                        "integer_indices_are_pixel_centers",
                    "implementation":
                        "harpy.utils.RasterAggregator.center_of_mass",
                    "algorithm_version": 1,
                },
                "coverage": {
                    "scope": "all_rows_for_region",
                    "n_obs": 125000,
                    "row_identity_digest": "sha256:...",
                },
                "source": {
                    "store_identity": "...",
                    "element_path": "labels/nuclei",
                    "content_revision": "...",
                    "shape_yx": [50000, 70000],
                    "dtype": "uint32",
                    "chunks_yx": [1024, 1024],
                },
                "generated_by": {
                    "package": "napari-harpy",
                    "version": "...",
                },
            },
        },
    }

The final serialized representation must use values supported by AnnData's zarr
writer. Versioned parsing and validation belong in a dedicated domain module;
Qt code must not interpret arbitrary dictionaries directly.

### Linkage fields

region_key and instance_key are strings naming the linkage columns in obs.

The actual region values are keys of the regions mapping. Do not store region
and coordinate_system as parallel str-or-list fields. Parallel lists are
fragile and cannot safely represent per-region provenance. A mapping also lets
one multi-region table acquire canonical centers incrementally.

Each region entry describes exactly one source labels element. The mapping key,
source_element, coordinate-frame element, and selected labels name must agree.

### Required provenance and semantics

Metadata validation requires:

- supported schema version;
- matching obsm key, two-dimensional x/y axes, matrix shape, and dtype;
- current table region_key and instance_key;
- a selected-region entry with matching source labels element;
- scale0 as the source level;
- element_intrinsic as the coordinate frame;
- center_of_mass with uniform label-pixel weighting;
- background value zero;
- the supported pixel-center convention and algorithm version;
- coverage of all current rows in the selected table region;
- a matching labels content revision and source store/element identity;
- finite x/y coordinates on every covered row.

Shape, dtype, chunks, and element path are useful diagnostics, but they do not
prove that pixel contents are unchanged. They must not substitute for a content
revision.

generated_by and an optional generation timestamp are provenance only. A cache
is not rejected merely because a newer napari-harpy package is running, unless
the supported algorithm/schema version or documented semantics changed.

### Coverage and multi-region tables

One table may annotate multiple labels regions. The single n_obs by 2 matrix can
therefore contain coordinates expressed in different labels-intrinsic frames.
The row's obs[region_key] value selects the metadata entry that defines its
frame.

This heterogeneous-frame representation is valid because coordinates from
different regions are never compared directly. Before querying, the controller
filters the table to the selected labels region and transforms the Shapes
geometry into that region's labels-intrinsic frame.

When a new spatial_canonical matrix is created, initialize all rows to NaN and
fill the selected region's rows. When another region is later calculated, fill
only that region and preserve valid existing region coordinates and metadata.

Coverage is complete only when every current table row in that region has one
finite center. The row-identity digest is calculated from stable row identity
and linkage values using a versioned canonical encoding. It detects membership
or identity changes without storing a potentially enormous ID list in uns.
Normal AnnData slicing/reordering must preserve obsm alignment; unsupported
external mutations that break AnnData's alignment guarantees are outside the
contract.

Subsetting a table, changing linkage values, replacing obs, or changing
region/instance-key metadata must cause coverage revalidation. Semantic table
events should invalidate affected region entries immediately rather than
waiting until Run.

### Labels content revision

Reliable persisted cache reuse requires a stable labels content-revision token.
Every supported Harpy operation that creates or changes a labels element must
write or advance that token. The token identifies label-pixel content, not the
element's coordinate transformation.

The selected region metadata snapshots the token used for center calculation.
On reuse, it must equal the current labels token.

If an existing labels element has no reliable content revision:

- a newly calculated cache can be reused safely for the current in-memory
  labels identity and session;
- it must not be treated as reliably reusable after reload/reopen solely from
  shape, dtype, or chunk metadata;
- the UI explains that persisted reuse requires labels revision metadata;
- the implementation roadmap must add/backfill the revision contract before
  declaring cross-session cache reuse production-ready.

Raw external writes that change zarr label chunks without advancing the
revision cannot be detected without hashing/reading the full raster. Such writes
violate the supported cache-coherency contract. A user-accessible Rebuild
canonical centers action remains available as recovery.

Changing only a SpatialData coordinate transformation does not invalidate
canonical centers because the stored frame is labels intrinsic. It does
invalidate an active query result whose geometry was transformed with the old
transform.

### Cache states

The cache inspector returns one of these typed states for the selected region:

- absent: neither usable matrix nor selected-region metadata exists;
- partial: a valid matrix/registry exists, but the selected region is not yet
  covered;
- valid: matrix, metadata, coverage, source revision, and finite values pass;
- stale: the cache was once valid but a known source or coverage revision no
  longer matches;
- invalid: matrix/metadata is malformed, contradictory, unsupported, or only
  one half of the pair exists.

Run behavior:

- valid: reuse immediately;
- absent or partial: calculate the selected region, install it atomically, then
  query;
- stale: recalculate and atomically replace only the selected region, then
  query;
- invalid: block automatic replacement and offer Rebuild canonical centers
  with explicit confirmation that the reserved array/metadata will be replaced.

The widget shows the current state before Run, including First query will
calculate canonical centers when appropriate.

### Atomic cache installation

Center calculation occurs off the Qt main thread against immutable request
inputs. The worker returns a compact result keyed by instance ID; it never
mutates AnnData.

After worker completion, the main-thread controller revalidates the request,
table identity, linkage, row coverage, labels identity/revision, and cache
generation. It then installs or updates the matrix and metadata as one
all-or-nothing table mutation.

If validation or installation fails:

- restore the complete prior obsm value and metadata registry;
- emit no accepted mutation event;
- leave dirty state unchanged;
- do not continue to the annotation dialog.

A successful cache create, extend, refresh, or rebuild changes table state and
therefore marks the shared table dirty. This remains true if the query finds no
instances or the user later cancels the Apply dialog. The cache is useful
derived data, but it is still an in-memory table change that must be written to
persist.

Undo last annotation reverses only the annotation-column mutation. It does not
remove or roll back a canonical-center cache created by the same Run action.

## User Experience

### Dock widget

Add a Spatial Query dock widget with the visual language and status-card
patterns used by existing Harpy widgets.

Recommended control order:

1. Coordinate system combo.
2. Annotation shapes combo.
3. Labels element combo.
4. Linked table combo.
5. Target column mode: Existing column or New column.
6. Existing-column combo or new-column line edit.
7. Canonical centers status.
8. Run Spatial Query button.
9. Query/action status and progress area with Cancel while running.
10. Undo last annotation button.
11. Write Table State and Reload Table from zarr buttons.
12. Persistent shared clean/dirty table-state status.

The target controls use stable identities rather than display strings so valid
selections survive refreshes.

### Selection dependencies

- Coordinate-system choices come from the shared HarpyAppState SpatialData.
- Annotation choices include only Shapes elements available in the selected
  coordinate system and valid under the Shapes Annotation contract.
- Labels choices include only supported 2D labels elements available in that
  coordinate system.
- Table choices include only tables whose SpatialData TableModel metadata
  declares the selected labels element as an annotated region.
- Existing-column choices include only compatible text/categorical annotation
  columns and exclude region_key and instance_key.
- Changing an upstream selection refreshes downstream options, closes pending
  dialogs, and cancels or invalidates active work.
- Preserve a downstream selection if its stable identity remains valid.
  Otherwise choose the first valid option or show a disabled placeholder with
  an explanation.

The coordinate-system combo participates in the shared active-coordinate-system
model. A change here updates HarpyAppState and follows the same layer cleanup
and refresh behavior as other coordinate-aware widgets.

### Target column behavior

If spatial_annotation already exists and is compatible, default to Existing
column and select it. Otherwise default to New column with spatial_annotation
prefilled.

A new column name:

- is trimmed;
- must pass SpatialData dataframe-column-name validation;
- must not equal region_key or instance_key;
- must not collide with an existing obs column;
- is not created until the user confirms Apply with at least one changed row.

An existing target column is compatible when it is:

- pandas categorical with string categories;
- pandas StringDtype;
- object/string containing only strings and missing values;
- an all-missing object/string column.

Numeric, boolean, datetime, mixed-object, and non-string categorical columns
are not writable and must never be converted implicitly.

### Canonical-center status

The status area reports one of:

- Ready: valid cached canonical centers will be reused;
- Not calculated: Run will calculate centers from scale0 first;
- Partial: Run will calculate centers for this labels region;
- Stale: Run will refresh centers for this labels region;
- Invalid: canonical-center data needs an explicit rebuild;
- Running: calculating canonical centers;
- Running: querying canonical centers.

Tooltips explain that the first calculation scans all scale0 chunks lazily and
may take substantially longer than later queries. The UI must not promise that
only chunks near the polygon will be read: center calculation is a global
labels aggregation.

### Run and result flow

1. The user configures a complete valid selection.
2. Run Spatial Query becomes enabled unless cache state is invalid and rebuild
   confirmation has not been obtained.
3. Clicking Run captures an immutable request containing stable source
   identities, selected coordinate system, table linkage, target intent, cache
   state, labels content revision, and a generation token.
4. If cache state is absent, partial, or stale, the worker calculates centers
   for all rows of the selected table region from scale0.
5. The UI shows phase-specific progress and offers Cancel. Selection controls
   remain readable, but changes invalidate the active request.
6. The worker evaluates the transformed annotation against either the validated
   cached coordinates or the newly calculated coordinates.
7. The worker returns instance IDs, diagnostics, and, when needed, a cache
   installation payload. It never mutates SpatialData, AnnData, or Qt state.
8. The controller revalidates all captured identities and revisions.
9. If a cache payload exists, install it atomically on the main thread and mark
   the table dirty before accepting the query result.
10. Resolve returned IDs to current table rows using region_key and
    instance_key.
11. If no canonical centers match, show No instance centers found in the
    annotation and make no annotation-column changes.
12. Otherwise open the Apply Spatial Annotation dialog.

Cancellation before cache installation makes no changes. Cancellation of the
Apply dialog after a newly calculated cache was installed leaves that cache in
memory and leaves the table dirty.

### Apply Spatial Annotation dialog

The modal dialog contains:

- annotation source name;
- labels element, table, and target column;
- inclusion rule: Canonical center inside annotation;
- number of eligible instances in the selected table region;
- number of canonical centers inside the annotation;
- number of matching rows currently missing a value;
- number already equal to the proposed value;
- number with a non-missing different value that would be overwritten;
- a QLineEdit labeled Annotation value, prefilled with the Shapes element name;
- Apply and Cancel actions.

It does not show labels missing from the table, because the centroid-based query
does not enumerate labels outside the table.

The annotation value is trimmed and must be a non-empty string. Unicode and
internal spaces are allowed. It is a user-facing category value rather than a
SpatialData element key, so element-name restrictions do not apply.

Changing the value updates equal/overwrite counts live. If different
non-missing values will be replaced, the dialog shows a prominent mandatory
warning and uses explicit action text such as Overwrite 12 and apply to 35.

Cancel closes the dialog without creating or changing the annotation column.
It does not undo a canonical-center cache installed earlier in the Run flow.

Immediately before Apply, revalidate:

- SpatialData, Shapes, labels, coordinate system, table, and target intent;
- table linkage and row identities;
- Shapes and transformation generations;
- spatial_canonical matrix/metadata generation;
- target-column values used for the displayed counts.

If only target values changed, refresh counts and require confirmation of the
updated summary. If source geometry, transform, centers, binding, selection, or
table identity changed, discard the result and require a new query.

### Successful apply

Applying is one main-thread, all-or-nothing obs mutation:

- create the new column only if requested and still absent;
- update only rows in the selected labels region whose instance IDs were
  returned;
- leave all other rows, regions, columns, obsm, and uns untouched;
- add a category without discarding existing category order or ordered state;
- represent unannotated values as actual missing values, never stringified
  missing values or an empty string;
- preserve the compatible target dtype;
- update each matching row at most once.

A new target column is categorical. Non-target rows are missing and the applied
value is its first category.

If every matching row already has the requested value, report a no-op. Do not
replace the column object, emit an annotation mutation event, create an undo
record, or alter dirty state. Any dirty state caused earlier by cache creation
remains.

After an effective annotation mutation:

- mark the selected table dirty through shared HarpyAppState;
- emit a semantic table-state event with the changed obs column, selected
  labels region, source, and change kind;
- refresh this widget and table-column/color-source consumers;
- show updated, overwritten, and unchanged counts;
- enable undo for this annotation operation.

### Undo last annotation

Keep one in-memory undo record for the most recent successful spatial
annotation. It contains exact affected row identities, prior values,
dtype/category metadata, whether the column was created, table mutation and
persistence revisions, and the table dirty state before annotation apply.

Undo:

- restores previous annotation-column values exactly;
- removes a column created by Apply when no later mutation touched it;
- never removes or rewinds spatial_canonical or its metadata;
- clears dirty state only when the complete table exactly returns to the
  unchanged persisted baseline;
- therefore remains dirty when the same Run first created/refreshed canonical
  centers;
- emits the semantic table event with source spatial_query_undo.

Undo is invalidated by reload, another spatial Apply, incompatible row/linkage
changes, table replacement, or a later mutation of the same target column. An
unrelated mutation may leave undo available but prevents it from clearing the
table's dirty marker.

## Canonical Center Calculation

### RasterAggregator execution

Use Harpy's RasterAggregator center-of-mass implementation as the canonical
calculation path.

For a 2D scale0 labels Dask array:

1. require integer dtype and known 2D shape/chunks;
2. add a singleton z dimension lazily, without copying or rechunking the raster,
   to satisfy RasterAggregator's z, y, x input contract;
3. pass the selected table-region instance IDs as index;
4. exclude background zero before aggregation;
5. let RasterAggregator construct and execute its Dask aggregation;
6. receive the compact per-instance z, y, x result;
7. require z to be the expected singleton-plane value;
8. drop z and reorder y, x to x, y;
9. validate one finite result for every requested ID;
10. join results back to table rows by instance ID.

Passing a known index avoids RasterAggregator's separate global unique-label
discovery. It does not avoid scanning scale0: every labels chunk may contribute
pixels to an instance, so center-of-mass calculation is a global aggregation.

RasterAggregator may currently use more than one full lazy pass, for example to
calculate counts and coordinate moments. The product contract is out-of-core
execution and bounded working memory, not a promise of exactly one storage
pass. The implementation must be benchmarked on representative local and remote
stores. If the current per-chunk intermediate layout exceeds the declared
production memory budget for high instance counts, RasterAggregator must be
optimized before release; falling back to loading labels eagerly is forbidden.

### Cache generation result

Use a UI-independent worker result, conceptually:

    CanonicalCenterBuildResult:
        labels_name
        table_name
        instance_ids
        centers_xy
        source_revision
        coverage_digest
        diagnostics
        generation

The arrays are compact and eager only after Dask completes. They contain one
row per table instance, not one row per label pixel.

The worker must not receive Qt or napari layer objects and must not mutate table
state. Main-thread installation is a separate domain operation.

### Performance and cancellation

- Run all labels I/O and Dask calculation off the Qt main thread.
- Preserve zarr-aligned scale0 chunks; do not rechunk as part of this feature.
- Never compute the full labels array, a full coordinate raster, or a full
  boolean mask.
- Bound local Dask concurrency against a documented memory budget.
- Use scheduler diagnostics for progress where feasible; otherwise show
  indeterminate progress with the current phase.
- Cancellation immediately prevents cache installation, result dialogs, and
  table mutation. Already scheduled local chunk reads may finish in the
  background; the UI must not claim hard I/O cancellation.
- Errors become actionable UI feedback and structured logs, with controls
  restored to a usable state.

## Spatial Query Engine Contract

### Inputs and results

Use typed, UI-independent contracts, for example:

    SpatialCenterQueryRequest:
        sdata
        shapes_name
        labels_name
        table_name
        coordinate_system
        predicate = "canonical_center_inside"
        cache_generation
        source_revision
        generation

    SpatialCenterQueryResult:
        shapes_name
        labels_name
        table_name
        coordinate_system
        instance_ids
        eligible_instance_count
        matched_instance_count
        cache_action
        cache_build_result or None
        generation

The exact Python types may differ. The separation is normative: center
calculation and geometry querying return immutable data; cache installation,
row preparation, and annotation mutation are distinct operations.

### Query algorithm

1. Validate and snapshot the selected Shapes polygons.
2. Union all polygons into one effective region.
3. resolve M_shapes_to_cs and M_labels_to_cs from current SpatialData;
4. calculate inverse(M_labels_to_cs) @ M_shapes_to_cs;
5. transform the union into labels-intrinsic x, y coordinates;
6. prepare the Shapely geometry for repeated predicates;
7. inspect spatial_canonical and metadata for the selected region;
8. if valid, snapshot the selected region's row identities and x/y values;
9. if absent, partial, or stale, calculate centers through RasterAggregator and
   use the resulting x/y values for this query;
10. reject non-finite values or a mismatch between row identities, instance IDs,
    centers, and declared coverage;
11. use the annotation bounds as a cheap vectorized prefilter;
12. evaluate Shapely only for candidate centers:

        candidates = (
            (x >= min_x) & (x <= max_x)
            & (y >= min_y) & (y <= max_y)
        )
        inside = shapely.intersects_xy(
            region_in_labels,
            x[candidates],
            y[candidates],
        )

13. map true values back to instance IDs;
14. return unique sorted positive integer IDs and diagnostics.

No labels raster data is read during step 11-14. With a valid cache, the entire
Run path after validation is an eager vectorized point query over the in-memory
table coordinates.

The bounding box is an optimization only. Shapely intersection is the
authoritative membership test, including holes and boundaries.

### Complexity

With a valid cache:

- labels I/O: zero;
- time: linear in the number of table rows for the selected region, with
  Shapely applied only to bounding-box candidates;
- memory: one coordinate snapshot and boolean/index arrays proportional to that
  region's table rows.

Without a valid cache:

- labels I/O: a global lazy aggregation over scale0, potentially multiple
  passes depending on RasterAggregator;
- time: center calculation plus the same in-memory point query;
- memory: bounded Dask chunk working state plus compact per-instance
  aggregation/results, never the full labels raster.

This tradeoff is intentional: the first query may be expensive, while repeated
queries against the same labels/table region should be snappy.

### Stale-result protection

Every Run has a monotonically increasing generation token and captured source
identities/revisions. Discard all worker output if, while it runs:

- the SpatialData object changes;
- coordinate system, Shapes, labels, table, or target-column intent changes;
- the Shapes element is written or replaced;
- a relevant Shapes/labels transformation changes;
- the labels element or content revision changes;
- the table is reloaded/replaced or linkage/row coverage changes;
- spatial_canonical is installed, rebuilt, or modified by another operation;
- a newer query starts;
- the widget closes.

For simplicity and predictable side effects, a stale Run installs neither its
cache payload nor its query result, even when center calculation itself would
still be reusable. A later Run may recalculate.

Worker completion callbacks perform UI and table work on the main thread only.

## Table Mutation Contract

Use a pure, testable preparation/apply boundary. Preparation contains:

- sorted queried instance IDs;
- exact matching table row positions/identities;
- current values at those positions;
- missing/equal/overwrite counts for a candidate annotation value;
- whether the target column will be created;
- binding, cache, table-mutation, and persistence revisions sufficient for
  apply-time validation.

Apply validates preparation against current state before mutating. If validation
or assignment fails, restore the entire prior target-column state and leave
dirty state/events unchanged. Partial row updates are forbidden.

Large counts use locale-aware formatting. The UI must never render an unbounded
instance-ID list.

## Shared State and Cross-Widget Events

Because this feature mutates obs, obsm, and uns, an event named only
TableObsWrittenEvent is too narrow. Introduce or generalize to a semantic event
such as TableStateChangedEvent with payload including:

    sdata
    table_name
    components       # obs, obsm, uns
    keys              # changed columns/array/registry keys
    regions
    change_kind       # created, updated, removed, rebuilt, reloaded
    source            # spatial_query, spatial_query_canonical, undo, ...

Accepted mutation events mark the table dirty. Reload is a state replacement
event or an explicitly non-dirty event. Producers do not invoke widgets
directly.

Shared app state also maintains monotonic per-table mutation and persistence
revisions. Every accepted in-memory table mutation increments the mutation
revision. Successful writes/reloads advance the persistence revision; reload
also establishes a new accepted baseline. These session counters supplement the
user-facing dirty boolean and are not stored in AnnData.

Expected consumers:

- Spatial Query refreshes cache/column state and invalidates unsafe dialogs or
  undo;
- Viewer refreshes linked table column/color-source choices while preserving
  valid selections;
- Object Classification refreshes only state affected by the changed
  components/keys;
- future table widgets consume the same general event.

Handlers guard against feedback loops using event source and object identity.

## Clean/Dirty and Persistence Semantics

### Shared dirty state

Reuse HarpyAppState's per-table dirty tracking and PersistenceController. Do not
introduce a widget-local dirty truth.

| Action | Table mutation | Dirty-state result |
| --- | --- | --- |
| Bind/select inputs | No | Unchanged |
| Query using valid centers | No | Unchanged |
| Cancel center calculation | No accepted mutation | Unchanged |
| Center calculation fails | No accepted mutation | Unchanged |
| Install/extend/refresh/rebuild centers | obsm and uns | Dirty |
| Query returns no matching centers after cache install | No further mutation | Remains dirty |
| Cancel Apply after cache install | No further mutation | Remains dirty |
| Apply annotation is a no-op | No | Unchanged from current state |
| Apply creates/changes annotation column | obs | Dirty |
| Undo annotation | obs | Clean only if complete table equals persisted baseline |
| Successful write | Disk updated | Clean |
| Failed write | Possibly attempted | Remains dirty |
| Successful reload | Memory replaced from disk | Clean |
| Failed/cancelled reload | No accepted replacement | Unchanged |

Dirty status belongs to the entire table. Changes from Object Classification,
canonical-center generation, Spatial Query annotation, and other widgets
coexist and are written together.

### Write Table State

Write Table State is enabled only for a backed dirty table. It writes the
complete accepted current table state required by Harpy, including:

- obs and categorical metadata;
- obsm["spatial_canonical"];
- uns["spatial_coordinates"]["spatial_canonical"];
- other existing table matrices/metadata covered by the shared persistence
  contract.

The spatial annotation column and canonical cache do not get independent
writers. The action and success message identify the table and resolved store
path and state that the shared table state is being written.

A successful write makes newly generated centers reusable after reload/reopen,
provided labels content-revision validation succeeds. A failure shows an error
and leaves the table dirty.

Unbacked SpatialData is outside scope. Run and persistence actions remain
disabled with an explanation that the feature requires zarr-backed SpatialData,
Dask-backed scale0 labels, and a backed linked table.

### Reload Table from zarr

Reuse the Object Classification dirty-reload decision:

- Write table state and reload;
- Reload table state and discard local edits;
- Cancel.

Before replacement, invalidate active center calculations, queries, Apply
dialogs, and undo. Reload performs the existing validated in-place replacement
of obs, obsm, and uns.

After success:

- re-inspect spatial_canonical and its metadata;
- refresh table metadata and compatible target columns;
- preserve a target selection only if still valid;
- notify consumers through the semantic table replacement event;
- clear dirty state;
- show the source path and outcome.

Late worker results created before reload must never install a cache, open a
dialog, or mutate the reloaded table.

### Leaving a dirty dataset

Changing table selection may leave each table's shared dirty marker intact.
Before replacing or closing a SpatialData object with dirty tables, the shared
application lifecycle warns and offers write/discard/cancel behavior. A
Spatial Query-only warning is insufficient because other widgets can also
replace the dataset.

## Validation and Error States

Run remains disabled, with concise status and a detailed tooltip, when:

- no SpatialData is loaded;
- SpatialData/table is not backed by zarr;
- no coordinate system is selected;
- no eligible Shapes element is selected;
- the selected Shapes has unsaved Shapes Annotation edits;
- Shapes geometry is invalid, empty, unsupported, or cannot be unioned;
- no eligible 2D labels element is selected;
- labels has no scale0 or scale0 is not a known-chunk 2D Dask array;
- Shapes or labels is unavailable in the selected coordinate system;
- a required transform is missing, non-finite, unsupported, or non-invertible;
- no linked table is selected;
- table linkage metadata is missing/inconsistent;
- selected-region instance IDs are missing, duplicated, non-positive, or not
  integer-like;
- no valid target mode/column is configured;
- a new column name is empty, invalid, reserved, or colliding;
- an existing target column has an incompatible dtype;
- spatial_canonical is invalid/ambiguous and rebuild has not been confirmed;
- a calculation/query is already running for the same request.

Runtime outcomes are distinct:

- no centers inside annotation: neutral information;
- table instance absent from labels during center calculation: binding error;
- stale cache detected: informational refresh state;
- ambiguous/corrupt cache: explicit rebuild warning;
- overwrite: mandatory confirmation warning;
- cancellation or stale result: neutral/cancelled state;
- Dask, zarr, transform, geometry, aggregation, persistence, or validation
  failure: error with retry/recovery guidance.

Never expose a traceback in the widget. Log technical context and show a stable
user-facing message.

## Accessibility and Interaction Quality

- Controls have visible labels, accessible names, and logical keyboard order.
- Enter applies only when validation passes; Escape cancels.
- Warning meaning uses text/icon as well as color.
- Status cards are word-wrapped and copyable where practical.
- Long names are elided visually and shown fully in tooltips.
- Running phase, progress, cancellation, success, and failure are textual.
- Destructive rebuild, overwrite, and reload actions use explicit verbs.
- Modal dialogs have the widget as parent and cannot appear behind napari.
- The first-run cost and reuse state are understandable without reading logs.

## Architecture

Spatial Query is a feature domain rather than one utility module. Its core code
belongs in a dedicated package so the top-level core directory does not acquire
one large module per concern:

    core/
        spatial_query/
            __init__.py
            models.py
            canonical.py
            engine.py
            annotation.py

            __init__.py
                intentional, stable public exports for the feature domain

            models.py
                immutable request/result/preparation types
                cache-state and metadata value types
                shared spatial-query enums and literals

            canonical.py
                spatial_canonical metadata schema/parser/validator
                cache-state inspection and coverage/source fingerprints
                RasterAggregator adapter
                atomic cache installation and rollback payloads

            engine.py
                Shapes validation and union
                coordinate transformation
                vectorized center-containment query

            annotation.py
                target-column validation
                row resolution and conflict summaries
                atomic annotation apply and undo payloads

The package's consumers import its intentional API from
napari_harpy.core.spatial_query rather than importing implementation modules
directly. The __init__.py facade must remain small and explicit; it must not use
wildcard exports. This lets internal modules be reorganized later without
changing controllers or other consumers.

The corresponding widget package is:

    widgets/
        spatial_query/
            __init__.py
            controller.py
            widget.py
            dialogs.py
            status_card.py

            controller.py
                binding/cache validation
                worker lifecycle and generations
                stale-result handling
                cache install, apply, and undo orchestration

            widget.py
                selectors, cache status, progress, persistence actions

            dialogs.py
                rebuild confirmation
                Apply Spatial Annotation dialog

            status_card.py
                pure status-card specification builders

The core spatial_query package must remain UI independent. It must not import
Qt, napari widgets, HarpyAppState, or widget controllers. The widget controller
orchestrates the pure core operations with shared application services.

General concerns stay outside the feature package:

- shared table persistence and reload services;
- shared application state and mutation/persistence revisions;
- general cross-widget table events;
- generic SpatialData/table-binding helpers;
- Shapes Annotation's shared geometry-validity contract.

These dependencies are consumed by Spatial Query; they are not reimplemented
inside it.

Other existing modules in the flat core directory may eventually be moved into
corresponding feature-domain packages. That broader reorganization is a
separate cleanup and is explicitly not part of this feature. New Spatial Query
work adopts the package structure without introducing unrelated import churn.

Reuse rather than copy:

- annotating-table discovery and table/linkage metadata helpers;
- Shapes Annotation geometry validity helpers and write events;
- Harpy RasterAggregator;
- HarpyAppState dirty tracking and semantic events;
- PersistenceController write/reload behavior;
- active-coordinate-system selection patterns;
- styles, status cards, worker cleanup, and generation-token patterns.

The widget must not depend on Object Classification internals. Shared
persistence/reload UI belongs in a reusable component or service.

Do not register spatial_canonical as a classifier feature matrix merely because
both live in obsm. It has a distinct spatial-coordinate schema and lifecycle.

## Testing Strategy

### Metadata and cache-state tests

- missing matrix/metadata is absent;
- valid matrix plus missing region is partial;
- matrix without metadata and metadata without matrix are invalid;
- wrong shape, row count, axes, dtype, method, scale, frame, or schema rejected;
- linkage-key mismatch rejected;
- source store/element/revision mismatch is stale;
- transform-only changes do not stale intrinsic centers;
- coverage digest and finite-coordinate validation;
- multi-region incremental fill preserves other valid regions;
- rebuild replacement requires confirmation for ambiguous data;
- cache installation rollback restores matrix/metadata exactly;
- cache create/extend/refresh/rebuild marks shared dirty once;
- cache parser never mutates during inspection.

### RasterAggregator adapter tests

- 2D Dask scale0 is wrapped as singleton-z without eager computation;
- integer dtype and known chunks required;
- table instance IDs are passed as index and background is excluded;
- no global unique-label discovery when index is supplied;
- one-pixel labels, irregular labels, disjoint components, concave labels, and
  labels spanning chunks have correct centers;
- y/x to x/y conversion and integer pixel-center convention;
- output is joined by instance ID, not aggregator row order;
- absent label ID, duplicate result, NaN, and inf rejected;
- no NumPy/full-array labels fallback;
- only scale0 is read;
- cancellation/stale completion cannot install;
- memory and concurrency remain within the declared budget.

### Geometry/query tests

- one polygon and one included center;
- multiple disjoint polygons form one union;
- overlapping polygons do not duplicate IDs;
- hole excludes a center in its interior;
- exterior and hole boundaries are included;
- center outside while label pixels overlap is excluded;
- center inside while label pixels do not overlap is included;
- concave/disconnected-label center semantics are explicit;
- empty bounding-box candidate set avoids Shapely point work;
- identity, translation, scale, rotation, reflection, and composed transforms;
- x/y versus array y/x correctness;
- missing/non-invertible transform rejection;
- cached query reads zero labels chunks;
- result IDs are sorted, unique, positive, and limited to selected table region;
- multi-region coordinates are never compared in mixed frames.

### Table-domain tests

- row matching uses region_key plus instance_key;
- duplicate IDs across regions allowed and within selected region rejected;
- missing/boolean/fractional/string/non-finite IDs rejected;
- new categorical annotation column with missing non-target rows;
- compatible categorical, StringDtype, and object updates;
- category addition preserves order/ordered state;
- numeric/mixed/reserved target columns rejected;
- missing/equal/overwrite counts update with proposed value;
- no-op leaves column identity/events/current dirty state unchanged;
- apply rollback prevents partial mutation;
- undo restores values/dtype/categories/column absence;
- undo never removes canonical cache;
- undo dirty derivation covers cache-created-on-run, pre-dirty table,
  intervening write, and unrelated later mutation.

### Controller and async tests

- valid cache follows query-only phase;
- absent/partial/stale cache follows calculate-install-query phases;
- worker never mutates AnnData or Qt;
- cache payload installs only after main-thread revalidation;
- result accepted only for unchanged request;
- every selection/source/revision invalidation drops late results;
- older run cannot supersede newer run;
- cancellation prevents installation/dialog/mutation;
- reload freezes and invalidates pending work;
- worker errors restore usable controls and give feedback;
- cleanup disconnects workers/signals when widget closes.

### Widget tests

- dependent combo filtering and stable selection preservation;
- default spatial_annotation existing/new behavior;
- Run enablement/tooltips for every blocker;
- dirty Shapes session blocker;
- all canonical cache status states and phase text;
- explicit ambiguous-cache rebuild confirmation;
- result dialog contents and canonical predicate wording;
- live conflict recount and value validation;
- mandatory overwrite warning and explicit action text;
- no-result flow does not change annotation column;
- cancel after cache install leaves cache dirty state visible;
- apply/undo states and summaries;
- shared dirty indicator across widgets;
- write enabled only for backed dirty table;
- dirty reload write/discard/cancel branches;
- reload re-inspects cache and invalidates undo;
- accessibility, keyboard behavior, and long-name tooltips.

### Backed-zarr integration tests

- first Run calculates centers lazily and installs them in memory;
- first Run reads scale0 but not lower-resolution levels;
- second Run with valid cache reads no label chunks;
- cache is not on disk until Write Table State;
- write persists matrix, metadata, annotation columns, missing values, and
  categorical metadata;
- reopen/reload reuses cache when labels revision matches;
- stale labels revision triggers refresh rather than reuse;
- reload discards an unpersisted cache after confirmation;
- write failure retains dirty state and usable in-memory cache;
- reload validation failure preserves current table and dirty state;
- canonical-center, spatial-annotation, and object-classification changes
  coexist in one write;
- late center/query completion after reload cannot affect the table.

### Performance tests

- benchmark first-run aggregation for representative instance counts, raster
  sizes, chunk shapes, local stores, and remote-like latency;
- benchmark cached point queries for representative table sizes and polygon
  complexity;
- measure Dask task count, storage passes, peak memory, concurrency, and
  cancellation latency;
- establish regression thresholds from measured supported hardware;
- test repeated queries to demonstrate cache amortization;
- fail the release gate if current RasterAggregator intermediates exceed the
  declared production memory budget.

## Observability

Log structured diagnostic context for:

- cache inspection state and reason;
- center calculation start/cancel/stale-drop/success/failure;
- source scale0 shape, dtype, chunks, content revision, requested instance
  count, Dask task count, elapsed time, and peak/concurrency diagnostics;
- cache installation action, covered region/count, schema/algorithm version;
- query start/cancel/stale-drop/success/failure;
- coordinate system, transform identities, annotation bounds, eligible and
  bounding-box candidate counts, matched count, and elapsed time;
- annotation apply/undo counts and changed key;
- write/reload path and outcome.

Do not log full user annotation values, instance-ID arrays, coordinate arrays,
label chunks, polygons, or dataframes by default.

## Implementation Slices

Each slice ends with integrated tests, error handling, documentation, and
reviewable behavior. A happy path alone does not complete a slice.

### Slice 1: Canonical contracts, metadata schema, and labels revision

Deliverables:

- typed cache state, metadata, build request/result, query request/result, and
  annotation preparation contracts;
- spatial_coordinates/spatial_canonical schema version 1;
- parser, normalizer, non-mutating inspector, and stable error messages;
- coverage identity-digest definition;
- labels content-revision read/write contract and integration with supported
  labels mutation paths;
- fixtures for single/multi-region tables and every cache state;
- metadata/cache unit tests.

Exit criteria:

- valid versus absent/partial/stale/invalid is deterministic;
- structural metadata cannot masquerade as content freshness;
- all invalid input fails before Dask work or table mutation;
- domain modules have no Qt dependency.

### Slice 2: Lazy canonical-center calculation

Deliverables:

- scale0-only Dask labels resolver;
- lazy 2D-to-singleton-3D RasterAggregator adapter;
- validated table-region index handling;
- center-of-mass execution, x/y normalization, instance-ID join, and finite
  output validation;
- bounded concurrency, progress diagnostics, cancellation, and worker-safe API;
- representative zarr-backed correctness and memory tests;
- upstream RasterAggregator optimizations if benchmarks show production limits.

Exit criteria:

- output matches an independent eager reference on test-sized fixtures;
- no full labels computation, NumPy fallback, rechunk, or lower-scale read;
- index-supplied execution avoids unique-label discovery;
- large fixtures demonstrate bounded working memory under configured
  concurrency;
- missing labels or invalid results never produce an installable payload.

### Slice 3: Atomic cache lifecycle and persistence payload

Deliverables:

- creation of NaN-initialized row-aligned matrix;
- incremental multi-region fill and selected-region refresh/rebuild;
- atomic matrix plus metadata installation with rollback;
- shared dirty/revision changes and semantic table-state event;
- explicit ambiguous-cache rebuild confirmation domain action;
- persistence/reload support for spatial_canonical and metadata;
- cache lifecycle and backed-zarr tests.

Exit criteria:

- table consumers never observe half-installed matrix/metadata;
- unrelated table regions and metadata are preserved;
- successful install is dirty and round-trips through zarr;
- reload/reopen reuse is allowed only with validated source revision.

### Slice 4: Vectorized centroid-containment query

Deliverables:

- shared Shapes validation/snapshot/union;
- inverse(M_labels_to_cs) @ M_shapes_to_cs transformation;
- bounding-box prefilter and vectorized Shapely intersects_xy predicate;
- region filtering and sorted unique ID result;
- valid-cache and fresh-build input paths using one query implementation;
- geometry/transform/predicate tests and zero-label-I/O instrumentation for
  cached queries.

Exit criteria:

- results match an independent point-in-polygon reference;
- boundary, hole, transform, and x/y semantics are proven;
- multi-region coordinate frames never mix;
- valid cached queries read no labels chunks.

### Slice 5: Atomic annotation preparation, apply, and undo

Deliverables:

- exact row resolution through region and instance keys;
- missing/equal/overwrite summaries;
- compatible existing-column mutation and new categorical-column creation;
- apply-time cache/binding/table revalidation and rollback;
- single-operation undo with cache-aware dirty derivation;
- table-domain tests.

Exit criteria:

- no cancel, no-result, no-op, or failed apply changes the annotation column;
- all effective mutations are atomic;
- undo restores the annotation column exactly and never removes cache data.

### Slice 6: Widget selection and validation shell

Deliverables:

- registered Spatial Query dock widget and plugin manifest entry;
- coordinate system, Shapes, labels, table, target-column controls;
- canonical-center status and rebuild interaction;
- dependent filtering with stable identity preservation;
- default spatial_annotation behavior;
- Shapes Annotation dirty-session blocker;
- status cards, tooltips, accessible names, and selector-state tests.

Exit criteria:

- Run is enabled only for a complete valid/rebuild-authorized request;
- first-run cost and cache reuse state are clear before execution;
- selection/inspection never mutates or dirties a table.

### Slice 7: Async calculate-query-review-apply flow

Deliverables:

- worker orchestration with generation tokens and two execution phases;
- progress, cancellation, cleanup, and error routing;
- main-thread cache installation/revalidation;
- no-result outcome;
- Apply dialog with live counts and mandatory overwrite warning;
- main-thread annotation apply and undo UI;
- controller/dialog async tests.

Exit criteria:

- napari remains responsive during global aggregation;
- cancel/stale/error paths cannot install, open late dialogs, or annotate;
- cache installation and annotation application are visibly distinct state
  changes;
- users always see affected and overwrite counts before annotation mutation.

### Slice 8: Shared events, persistence UX, and cross-widget synchronization

Deliverables:

- general table-state event supporting obs, obsm, and uns;
- per-table mutation/persistence revisions;
- Viewer and Object Classification targeted refresh behavior;
- reusable Write Table State and Reload Table from zarr UI/service;
- dirty reload write/discard/cancel behavior;
- dirty-dataset close/replacement guard;
- multi-widget and backed-zarr integration tests.

Exit criteria:

- all widgets observe one current in-memory table state;
- canonical centers and annotation columns persist/reload together;
- no event loop or unrelated classifier invalidation;
- no late task affects a reloaded/replaced table;
- there is no competing widget-local dirty truth.

### Slice 9: Production hardening and release gate

Deliverables:

- representative first-build and cached-query benchmarks;
- documented memory/concurrency budgets and regression thresholds;
- cancellation-latency and remote-store testing;
- structured logging and diagnostic coverage;
- accessibility and keyboard pass;
- user documentation with screenshots and exact centroid semantics;
- migration notes explaining why unregistered spatial/spatial_canonical data is
  not reused;
- full test, lint, formatting, and manual napari smoke tests on supported
  platforms.

Exit criteria:

- all Definition of Done items pass;
- no correctness, data-loss, stale-worker, cache-coherency, or persistence
  defect is deferred as polish;
- performance limitations are measured and documented;
- first-build resource usage meets the declared professional-product budget.

## Definition of Done

The feature is complete when a user can select a valid stored polygon
annotation, 2D labels element, and linked table; transparently create or reuse
validated canonical centers; run a responsive center-containment query; review
affected and overwritten rows; apply a string annotation to a compatible
existing or new obs column; undo the last annotation; and safely write/reload
the complete shared table state from zarr.

Completion additionally requires:

- spatial_canonical has a versioned, validated, persisted metadata contract;
- centers are scale0 uniform-pixel centers of mass in labels-intrinsic x/y;
- missing/partial/stale/invalid cache states have deterministic behavior;
- valid cached queries read no label pixels;
- cache generation remains lazy/out-of-core and never loads full labels;
- the first-run global scan and later reuse are clear in the UI;
- polygon union, holes, boundaries, and affine transformations are tested;
- query membership is canonical-center containment, never silently pixel
  overlap;
- no annotation mutation occurs before explicit Apply;
- cache installation is atomic and visibly marks the shared table dirty;
- overwrite is never silent;
- no stale/cancelled worker installs data, opens a dialog, or mutates a table;
- row identity always uses region_key and instance_key;
- table rows with no source label are rejected as binding inconsistencies;
- labels absent from the table are not claimed as queryable or counted;
- annotation apply/rollback and undo are safe and atomic;
- shared cross-widget dirty state and general table events work for obs, obsm,
  and uns;
- canonical cache and annotation columns round-trip through backed zarr;
- reload and dirty-dataset replacement protect local changes;
- accessible validation/status/error feedback is complete;
- repository tests, lint, formatting, benchmarks, and manual smoke tests pass.

## Final Architectural Decision

spatial_canonical is a reusable, row-aligned spatial index for the instances in
a linked AnnData table. It stores one x/y center of mass per covered table row
in the intrinsic coordinate frame of that row's source labels region.

On Run:

    inspect and validate spatial_canonical
                    |
          valid ----+---- missing, partial, or stale
            |                         |
      reuse centers             calculate centers
            |                 lazily from scale0
            |                         |
            |                 atomically install
            |                   and mark dirty
            +------------+------------+
                         |
              transform annotation into
              labels-intrinsic coordinates
                         |
              vectorized intersects_xy
                         |
              review and explicitly Apply

The expensive global raster aggregation is paid only when the canonical index
is unavailable or invalidated. The frequent operation is a fast vectorized
point-in-polygon query over coordinates already held by the table.
