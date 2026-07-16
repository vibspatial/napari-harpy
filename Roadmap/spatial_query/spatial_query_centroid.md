# Spatial Query Annotation Using Canonical Centers

## Status

Final product specification and implementation plan.

Canonical metadata/cache lifecycle (Slice 1a) and blocking Harpy centroid
construction/cache ensure (Slice 1b) are implemented. Later slices remain
planned.

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
- **No silent data loss:** overwriting values, rebuilding a mismatched managed
  cache, reloading a dirty table, and leaving a dirty dataset are reported
  explicitly.
- **One shared table state:** all Harpy widgets see the same in-memory AnnData
  and the same per-table dirty marker.
- **Deterministic and testable:** identical geometry, transforms, canonical
  centers, linkage metadata, and table state produce identical results.
- **Accessible:** all important states, warnings, busy states, and errors are
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

The canonical cache always uses a fixed z, y, x storage layout, including for
two-dimensional labels. RasterAggregator already returns z, y, x. For a 2D
source, the lazily added singleton z plane produces and stores z=0.0 for every
covered row. This z value is a storage-padding convention, not evidence that
the source labels element is three-dimensional. The source signature continues
to record the actual source dims, for example ("y", "x").

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

The UI must describe the rule as Centroid inside annotation, with a
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
projecting a volume. The fixed z/y/x cache layout is deliberately 3D-ready, but
does not add 3D calculation, transformation, or query semantics in this
release. A 2D query reads y from column 1 and x from column 2 and ignores the
synthetic z column; it does not pass z into a 2D transformation.

### Table binding and instance membership

Selectable tables are discovered using the existing annotating-table helper.
Before cache inspection or query execution, the selected binding must pass the
shared table-binding validator and the additional canonical-center
requirements.

The table row for an instance is identified by both linkage keys:

    obs[region_key] == selected_labels_name
    and
    obs[instance_key] == instance_id

Matching by obs row order or by obs_names is forbidden. Spatial identity is the
validated `(region_key value, instance_key value)` pair; the AnnData index does
not participate in canonical cache identity.

Within the selected labels region, instance-key values must be non-missing,
positive, integer-like, and unique. Booleans, fractional numbers, non-finite
numbers, and strings that only resemble numbers are rejected rather than
silently coerced. Duplicate instance IDs in another region are allowed.
The selected-region row set must be non-empty. An empty selection is a binding
error: this feature does not create table rows. Missing or duplicate obs_names
do not affect this contract because they are not SpatialData linkage values.

Canonical-center calculation uses the validated instance IDs from the selected
table region as RasterAggregator's requested index. This avoids a separate
global unique-label discovery pass. The calculated result is joined back to
AnnData rows by instance ID; code must not assume that RasterAggregator returns
the same row order as AnnData.

If a requested table instance has no pixels in the selected labels element, its
aggregate count is zero and its temporary center is non-finite. This is a
table/labels binding inconsistency. The adapter validates the complete requested
result before the cache update and reports the missing-ID count plus a bounded
preview of IDs. The cache update and spatial query fail; a non-finite
center for a covered region row is never committed.

Labels values that are absent from the selected table region are intentionally
outside the operation. They are not requested from RasterAggregator, do not get
centers, and are neither discovered nor reported. Validation therefore requires
no global unique-label pass over the raster, although calculating the requested
centers still scans all scale0 chunks lazily.

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
- asynchronous center calculation and query execution, cancellation, textual
  busy status, and stale-result protection;
- per-table shared clean/dirty state;
- selective AnnData element-level persistence of all supported dirty table
  components, including canonical centers and their metadata, without rewriting
  the complete AnnData table;
- reloading the current table state from zarr with dirty-state protection;
- undo of the most recently applied spatial annotation while its source
  binding remains valid;
- cross-widget refresh after a cache update, annotation mutation, write,
  reload, or Shapes element write.

### Non-goals

- modifying labels pixels or Shapes geometry;
- querying NumPy-backed labels or lower-resolution pyramid levels;
- creating missing table rows or a new linked table;
- returning labels instances that have no linked table row;
- querying arbitrary unregistered napari Shapes layers;
- querying unsaved Shapes Annotation edits;
- per-polygon annotation values within one Shapes element;
- 3D center calculation/query semantics; the storage layout is 3D-ready, but
  3D sources remain unsupported in this release;
- any-pixel overlap, full-label containment, or percentage-overlap predicates;
- boolean combinations selected interactively across Shapes elements;
- automatically writing to zarr after every cache or annotation change;
- automatically recoloring labels;
- detecting out-of-band labels pixel changes when dimensions, shape, dtype, and
  table linkage all remain unchanged;
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

- has shape n_obs by 3;
- is a dense floating-point NumPy-compatible array;
- uses columns z, y, x in that order;
- is row-aligned through AnnData's obsm contract;
- contains finite coordinates for every row covered by a valid region entry;
- contains z=0.0 for every covered row belonging to a 2D labels source;
- may contain NaN for rows belonging to table regions whose centers have not
  yet been calculated;
- must not contain inf;
- must not be sparse.

spatial_canonical is a reserved Harpy key. An existing array at that key without
valid matching metadata is never trusted. When the user explicitly requests
canonical centers, the mismatch is reported, recalculation completes first, and
the reserved pair is replaced only after a valid result is available.

### Metadata schema

A representative schema is:

    adata.uns["spatial_coordinates"]["spatial_canonical"] = {
        "schema_version": 1,
        "obsm_key": "spatial_canonical",
        "axes": ["z", "y", "x"],
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
                    "axes": ["z", "y", "x"],
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
                    "instance_set_digest": "sha256:...",
                },
                "source": {
                    "element_path": "labels/nuclei",
                    "dims": ["y", "x"],
                    "shape": [50000, 70000],
                    "dtype": "uint32",
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
- matching obsm key, fixed z/y/x axes, three-column matrix shape, and dtype;
- current table region_key and instance_key;
- a selected-region entry with matching source labels element;
- scale0 as the source level;
- element_intrinsic as the coordinate frame;
- center_of_mass with uniform label-pixel weighting;
- background value zero;
- the supported pixel-center convention and algorithm version;
- coverage of all current rows in the selected table region;
- matching source element, scale0 dimensions, shape, and dtype;
- finite z/y/x coordinates on every covered row, with z=0.0 for a 2D source.

Dimensions, shape, dtype, and table coverage form the structural cache
signature. A mismatch is reported and triggers recalculation. Dask/zarr
chunking is deliberately excluded: it affects execution and performance but not
the mathematical center or the validity of cached coordinates. Live chunks are
validated and inspected when calculation is required, but are not persisted or
compared for cache reuse.

generated_by and an optional generation timestamp are provenance only. A cache
is not rejected merely because a newer napari-harpy package is running, unless
the supported algorithm/schema version or documented semantics changed.

### Coverage and multi-region tables

One table may annotate multiple labels regions. The single n_obs by 3 matrix can
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
NaN is a placeholder only for rows in table regions that have no valid region
entry yet. Every row covered by a registered region entry must have finite
z/y/x coordinates, and every covered row for a 2D source must have z=0.0; a
region is never registered with partial coverage.

Coverage is complete only when every current table row in that region has one
finite center. Each labels element has a separate instance_set_digest stored in
its own regions[labels_name].coverage entry; there is no table-global digest.
The digest is calculated from a domain/schema tag, the selected labels name,
and the sorted set of normalized instance IDs using a versioned canonical
binary encoding and SHA-256. Including the labels name domain-separates
otherwise identical instance sets in different regions. The digest avoids
storing a potentially enormous instance-ID list in uns, and obs_names do not
participate.

Instance-ID normalization is numeric only. Positive integral scalars and finite
integer-like real scalars such as 1.0 normalize to the same canonical integer.
Booleans, strings such as "1", missing values, non-finite values, zero, negative
values, and fractional values are rejected. Every normalized ID must fit in an
unsigned 64-bit integer, which covers the positive range of every supported
integer labels dtype.

Normal AnnData slicing/reordering must preserve obsm alignment. Row reordering
and obs-name changes do not alter the instance-set digest. Unsupported external
mutations that break AnnData's alignment guarantees are outside the contract.

An unordered instance-set digest cannot detect a reassignment that preserves
the complete per-region instance set, such as swapping two instance-key values
between existing rows. Supported mutations to obs, region_key, or instance_key
must therefore emit semantic table events that invalidate every affected region
entry before the cache can be reused, even when the resulting instance sets are
unchanged. An out-of-band same-set reassignment that bypasses those events is an
explicit structural-validation limitation, like an out-of-band labels-pixel
edit that preserves the labels structural signature. Forced recalculation is
the recovery path.

Subsetting a table, changing linkage values, replacing obs, or changing
region/instance-key metadata must cause coverage revalidation. Semantic table
events must invalidate affected region entries immediately rather than waiting
until Run; this is required for same-set linkage reassignments that structural
inspection cannot detect.

### Structural cache validity

Canonical metadata is authoritative for deciding whether coordinates can be
reused. Validation compares it with the current labels element and linked table.
It does not add Harpy-specific attributes to the SpatialData object.

A selected-region cache is structurally reusable when:

- the live matrix and top-level metadata agree;
- the source is the selected labels element at scale0;
- dimensions, shape, and dtype agree;
- the calculation semantics and algorithm version agree;
- region_key, instance_key, selected-row count, and the selected region's
  instance-set digest agree;
- all selected-region x/y coordinates are finite.

Changing a SpatialData coordinate transformation does not invalidate centers
because their frame is labels intrinsic. The current transformations are
validated and applied later when the annotation geometry is queried.

This contract cannot detect an out-of-band label-pixel edit that preserves the
complete structural signature. Detecting that case would require reading or
hashing the raster again and would defeat fast cache reuse. The explicit
Recalculate centroids action described below is the user-facing recovery path;
the core ensure operation also exposes the same forced-recalculation mode to
non-UI callers.

### Cache states

The cache inspector returns one of these typed states for the selected region:

- absent: neither usable matrix nor selected-region metadata exists;
- partial: a valid matrix/registry exists, but the selected region is not yet
  covered;
- valid: matrix, metadata, structural source signature, table coverage, and
  finite values pass;
- stale: the cache is interpretable but its selected-region structural
  source/coverage/algorithm signature no longer matches;
- invalid: matrix/metadata is malformed, contradictory, unsupported, or only
  one managed component exists.

Run behavior:

- valid: reuse immediately;
- absent or partial: calculate the selected region, apply the cache update atomically, then
  query;
- stale: recalculate and atomically replace only the selected region, then
  query;
- invalid: report the all-regions mismatch, calculate the selected region, and
  rebuild the managed matrix/metadata pair. The rebuild is conservative: an
  existing region is preserved only when its coordinates and metadata can be
  fully substantiated against the shared matrix, current table, and current
  source signature. If an all-regions inconsistency prevents that proof,
  rebuild a selected-region-only cache rather than carrying the entry forward.

The widget shows the current state before Run, including First query will
calculate centroids when appropriate.

### Atomic cache update

Center calculation occurs off the Qt main thread against immutable request
inputs. The worker returns a compact result keyed by instance ID; it never
mutates AnnData.

After worker completion, the main-thread controller revalidates the request,
table identity, linkage, row coverage, labels structural signature, and cache
generation. It then applies the matrix and metadata update as one
all-or-nothing table mutation.

If validation or the cache update fails:

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
7. Centroid cache status and Recalculate centroids button.
8. Run Spatial Query button.
9. Query/action status card that reports when calculation is busy.
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

### Centroid status

The status area reports one of:

- Ready: valid cached centroids will be reused;
- Not calculated: Run will calculate centers from scale0 first;
- Partial: Run will calculate centers for this labels region;
- Stale: Run will refresh centers for this labels region;
- Invalid: Run will report the mismatch and rebuild centroid data;
- Running: calculating centroids;
- Running: querying centroids.

Tooltips explain that the first calculation scans all scale0 chunks lazily and
may take substantially longer than later queries. The UI must not promise that
only chunks near the polygon will be read: center calculation is a global
labels aggregation.

### Recalculate centroids action

Recalculate centroids is an explicit standalone action for refreshing the
selected labels region when the user knows or suspects that label pixels have
changed without a detectable structural change. It is user-facing terminology;
internal code and metadata continue to use the spatial_canonical contract.

The action is enabled when a supported labels element and a valid non-empty
linked-table region are selected and no conflicting table/calculation operation
is running. It does not require a valid Shapes selection or annotation target,
does not perform a spatial query, and never changes an annotation column.

On activation:

1. capture the selected labels/table-region identities, structural signatures,
   instance-set digest, current cache generation, and a new operation ID;
2. bypass valid-cache reuse and calculate all requested table-region centroids
   lazily from scale0;
3. validate that every requested table ID has exactly one finite result;
4. revalidate the captured table, labels, binding, and cache generation on the
   main thread;
5. atomically replace only the selected region's current row positions and
   metadata when the shared matrix/top-level contract is valid, preserving all
   other valid regions;
6. emit the shared table-state event, mark the table dirty, refresh the centroid
   status, and report completion.

If calculation is cancelled, fails, produces a missing/non-finite requested ID,
or becomes stale before the cache update, the previous obsm/uns state is preserved
exactly, no mutation event is emitted, and dirty state is unchanged. An
all-regions invalid matrix/metadata state follows the existing rebuild rules,
but is not replaced until a complete valid calculation result is available.

### Run and result flow

1. The user configures a complete valid selection.
2. Run Spatial Query becomes enabled for valid, absent, partial, stale, and
   rebuildable-invalid cache states.
3. Clicking Run performs a fresh selected-region cache inspection, including
   one instance-set digest, then captures an immutable request containing stable
   source identities, selected coordinate system, table linkage, target intent,
   cache state, structural signatures, and one operation ID.
4. A valid cache supplies the selected rows of `spatial_canonical` directly and
   skips centroid calculation. If the cache is absent, partial, stale, or
   rebuildable-invalid, a worker calculates centers for all rows of the selected
   table region from scale0 and returns a `CanonicalCacheUpdatePayload` without
   mutating the table.
5. The main-thread controller accepts the payload only for the current operation
   ID and applies it through `apply_canonical_cache_update()`. Its fresh
   source/binding validation re-resolves current rows and rejects outdated work.
6. Once canonical centers are valid or the cache update has succeeded, a worker
   evaluates them against the transformed Shapes geometry. It returns matching
   instance IDs without mutating SpatialData, AnnData, or Qt state.
7. Throughout either worker phase, the status card reports that calculation is
   busy without a progress bar or percentage. Selection changes invalidate the
   operation ID.
8. The main-thread controller accepts the query result only for the current
   operation ID, revalidates the captured request, and resolves returned IDs to
   current table rows using `region_key` and `instance_key`.
9. If no centroids match, show No instance centroids found in the
   annotation and make no annotation-column changes.
10. Otherwise open the Apply Spatial Annotation dialog.

Cancellation before the cache update makes no changes. If a newly calculated
cache was already applied before the query phase is cancelled or invalidated,
that useful cache remains in memory and remains dirty, but no result dialog or
annotation mutation follows. Cancelling the Apply dialog likewise leaves an
already-applied cache intact.

### Apply Spatial Annotation dialog

The modal dialog contains:

- annotation source name;
- labels element, table, and target column;
- inclusion rule: Centroid inside annotation;
- number of eligible instances in the selected table region;
- number of centroids inside the annotation;
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
It does not undo a canonical-center cache updated earlier in the Run flow.

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
dtype/category metadata, whether the column was created, relevant component
revisions, and the table dirty state before annotation apply.

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

1. select the rows for the requested labels region and reject an empty set;
2. validate non-missing, positive, unique integer-like instance IDs within that
   region;
3. use SpatialData's validated 2D Dask raster representation and require an
   integer labels dtype;
4. add a singleton z dimension lazily, without copying or rechunking the raster,
   to satisfy RasterAggregator's z, y, x input contract;
5. pass exactly the selected table-region instance IDs as index;
6. exclude background zero before aggregation;
7. let RasterAggregator construct and execute its Dask aggregation;
8. receive the compact per-requested-instance z, y, x result;
9. require z to equal the expected singleton-plane value 0.0;
10. retain the float64 z, y, x result unchanged for canonical storage;
11. validate exactly one finite result for every requested ID; a zero-count or
    non-finite result means that a table instance is absent from the raster and
    is a binding error;
12. require RasterAggregator to return the requested IDs in the same order, as
    guaranteed by its `center_of_mass(index=...)` contract; retain centers in
    that binding order.

Passing a known index avoids RasterAggregator's separate global unique-label
discovery. It does not avoid scanning scale0: every labels chunk may contribute
pixels to an instance, so center-of-mass calculation is a global aggregation.
Raster IDs outside the requested table index do not produce output and are not
validated; this is the intended table-defined query universe.

RasterAggregator may currently use more than one full lazy pass, for example to
calculate counts and coordinate moments. The product contract is out-of-core
execution through Harpy, not a promise of exactly one storage pass. Harpy owns
RasterAggregator scheduling, concurrency, memory, and performance. napari-harpy
does not add local performance budgets or eager fallbacks.

### Background calculation result

The worker calls `calculate_canonical_centers()` and returns its existing
`CanonicalCacheUpdatePayload`; do not introduce a second result type that
duplicates the binding, source signature, instance IDs, or centers. The arrays
are compact and eager only after Harpy's Dask calculation completes. They
contain one row per selected table instance, not one row per label pixel.

The worker receives no Qt or napari layer objects and never mutates table state.
Applying the payload through `apply_canonical_cache_update()` is a separate
main-thread domain operation.

### Background execution and cancellation

- Run all labels I/O and Dask calculation off the Qt main thread.
- Preserve the scale0 Dask representation; do not rechunk or eagerly load the
  labels raster as part of this feature.
- Follow the existing Feature Extraction and Object Classification worker
  lifecycle: assign a monotonically increasing operation ID, keep the active
  operation phase, call `worker.quit()` when invalidating it, clear the active
  reference, and ignore every late signal whose operation ID is no longer
  current.
- `worker.quit()` is best-effort invalidation, not hard interruption of Harpy's
  synchronous calculation. The underlying calculation may finish, but its
  result cannot update the cache, open a later dialog, or mutate a table.
- Do not add a Dask cancellation token, scheduler callback, concurrency policy,
  task diagnostics, progress bar, percentage, or performance telemetry.
- While current work is running, show a textual busy message in the status card.
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
        source_signature
        operation_id

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
        operation_id

The exact Python types may differ. The separation is normative: center
calculation and geometry querying return immutable data; cache updates,
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

Every Run has a monotonically increasing operation ID and captured
source identities/structural signatures. Discard all worker output if, while it
runs:

- the SpatialData object changes;
- coordinate system, Shapes, labels, table, or target-column intent changes;
- the Shapes element is written or replaced;
- a relevant Shapes/labels transformation changes;
- the labels element is replaced or its structural signature changes;
- the table is reloaded/replaced or linkage/row coverage changes;
- spatial_canonical is created, rebuilt, or modified by another operation;
- a newer query starts;
- the widget closes.

For simplicity and predictable side effects, a stale Run applies neither its
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
- binding, cache, and relevant table-component revisions sufficient for
  apply-time validation.

Apply validates preparation against current state before mutating. If validation
or assignment fails, restore the entire prior target-column state and leave
dirty state/events unchanged. Partial row updates are forbidden.

Large counts use locale-aware formatting. The UI must never render an unbounded
instance-ID list.

## Shared State and Cross-Widget Events

Because table mutations can affect obs, obsm, and nested uns state, introduce a
general `TableStateChangedEvent`. A logical table-component path has this
concrete validated shape:

    TableComponent = Literal["obs", "obsm", "uns"]


    @dataclass(frozen=True, order=True)
    class TableComponentPath:
        component: TableComponent
        keys: tuple[str, ...]

        def __post_init__(self) -> None:
            if self.component not in ("obs", "obsm", "uns"):
                raise ValueError("Unsupported table component.")
            if not self.keys or any(
                not isinstance(key, str) or not key
                for key in self.keys
            ):
                raise ValueError("Table component path keys must be non-empty strings.")
            if self.component in ("obs", "obsm") and len(self.keys) != 1:
                raise ValueError("obs and obsm paths must identify exactly one key.")


Examples are:

    TableComponentPath("obs", ("spatial_annotation",))
    TableComponentPath("obsm", ("spatial_canonical",))
    TableComponentPath(
        "uns",
        ("spatial_coordinates", "spatial_canonical"),
    )

The event has this conceptual shape:

    TableStateChangedEvent:
        sdata
        table_name
        paths              # unique logical component paths
        regions
        change_kind        # created, updated, removed, rebuilt, reloaded
        source             # spatial_query, spatial_query_canonical, undo, ...

The event describes what changed; emitting it does not by itself decide whether
the paths are dirty. HarpyAppState exposes three explicit acceptance paths:

    record_table_mutation(event)
        # emit the event and assign a new dirty revision to every path

    record_persisted_table_change(event)
        # emit the event and establish its exact paths as already persisted

    record_table_reload(event)
        # emit a replacement event and establish the complete table as clean

Object Classification uses `record_table_mutation()` because it changes the
in-memory table and relies on PersistenceController for the later write. Harpy
Feature Extraction writes its feature matrix and metadata directly when
SpatialData is backed and therefore uses `record_persisted_table_change()`
after Harpy returns successfully. Unbacked Feature Extraction uses
`record_table_mutation()`. Reload uses `record_table_reload()`. Producers do not
invoke widgets directly.

`TableStateChangedEvent` is the shared persistence and low-level table-state
contract. Existing domain events such as `FeatureMatrixWrittenEvent` and
`ClassificationTableWrittenEvent` remain available because they carry useful
feature-specific meaning for consumers. A producer may therefore publish its
domain event and one table-state event for the same accepted change. Only the
explicit HarpyAppState acceptance method updates or clears the shared
dirty-component manifest.

Shared app state maintains, for each in-memory table, a mapping from dirty
component path to its latest monotonic revision. The user-facing table dirty
boolean is derived from whether that mapping is non-empty. Every accepted
mutation allocates one new table revision and assigns it to all paths in that
event. A write captures the current `path -> revision` mapping and clears a path
only when that exact revision was successfully persisted and is still current.
A newer mutation of the same path therefore remains dirty. Separate mutation
and persistence revision counters are not required. These session values are
not stored in AnnData.

The persistence handoff has one explicit snapshot/write/acknowledge boundary:

    snapshot = app_state.snapshot_table_dirty_state(sdata, table_name)
    result = write_table_components(
        sdata,
        table_name=table_name,
        paths=snapshot.paths,
    )
    app_state.acknowledge_table_write(
        snapshot,
        persisted_paths=result.persisted_paths,
    )

`TableDirtySnapshot` captures the table identity and an immutable
`TableComponentPath -> revision` mapping. `TableComponentWriteResult` reports
the table store and exact logical paths that were successfully persisted. The
Qt-independent writer receives paths, not HarpyAppState or revision state;
HarpyAppState alone decides whether the captured revisions are still current
and may be cleared. Unscoped `mark_table_dirty()` and `clear_table_dirty()`
operations are therefore not primary mutation or persistence APIs.

The deferred-write flow is:

    widget/controller accepts a table mutation
        ↓
    record_table_mutation(TableStateChangedEvent)
        ↓
    HarpyAppState records path revisions
        ↓
    PersistenceController captures a dirty snapshot
        ↓
    core persistence writes supported AnnData elements
        ↓
    HarpyAppState clears only unchanged persisted revisions

An operation that already persisted its result uses the shorter flow:

    producer completes its element writes and metadata consolidation
        ↓
    record_persisted_table_change(TableStateChangedEvent)
        ↓
    HarpyAppState emits the event and establishes only those paths as persisted

The general event, component manifest, and persistence foundation are
introduced before canonical-cache integration. Full Viewer, Object
Classification, and Spatial Query refresh behavior, feedback-loop guards, and
shared persistence UI are completed in the later cross-widget integration
slice.

## Clean/Dirty and Persistence Semantics

### Shared dirty state

Generalize HarpyAppState's per-table dirty tracking and the existing
`PersistenceController` with the shared component manifest described above. Do
not create a Spatial Query-specific persistence controller or a widget-local
dirty truth. Multiple widgets may own selection-aware controller instances, but
they coordinate through the same HarpyAppState manifest and generic core
persistence operations.

| Action | Table mutation | Dirty-state result |
| --- | --- | --- |
| Bind/select inputs | No | Unchanged |
| Query using valid centers | No | Unchanged |
| Invalidate/cancel center calculation | No accepted mutation | Unchanged |
| Center calculation fails | No accepted mutation | Unchanged |
| Create/extend/refresh/rebuild centers | obsm and uns | Dirty |
| Successful Recalculate centroids | selected-region obsm and uns | Dirty |
| Failed/cancelled Recalculate centroids | No accepted mutation | Unchanged |
| Query returns no matching centers after cache update | No further mutation | Remains dirty |
| Cancel Apply after cache update | No further mutation | Remains dirty |
| Apply annotation is a no-op | No | Unchanged from current state |
| Apply creates/changes annotation column | obs | Dirty |
| Undo annotation | obs | Clean only if complete table equals persisted baseline |
| Successful write, no remaining/newer dirty components | Captured components persisted | Clean |
| Successful write with remaining/newer dirty components | Captured components persisted | Remains dirty |
| Failed write | No accepted persistence completion | Remains dirty |
| Successful reload | Memory replaced from disk | Clean |
| Failed/cancelled reload | No accepted replacement | Unchanged |

Dirty status belongs to the entire table. Changes from Object Classification,
canonical-center generation, Spatial Query annotation, and other widgets
coexist and are written together. Creating or updating
obsm["spatial_canonical"] and its required metadata is an accepted AnnData table
mutation and therefore always records both component paths as dirty. Merely
computing a worker result, reusing an existing cache, or rejecting a result does
not.

### Write Table State

Write Table State is enabled only for a backed dirty table. It snapshots the
shared dirty-component manifest and uses AnnData's element-level zarr API to
persist the union of supported dirty components. It must not call
AnnData.write_zarr, rewrite the complete AnnData object, or rewrite unrelated X,
layers, var, varm, obsp, or uns/obsm entries.

Preflight must resolve every dirty path to a supported element writer. An
unknown or unsupported dirty path blocks the operation with an actionable error;
it is never silently skipped, and the table remains dirty.

The current `PersistenceController` is generalized rather than duplicated. Its
selection, reload, and user-facing coordination remain in the application
layer; path resolution and AnnData element writing live in Qt-independent core
persistence functions. The existing `write_table_prediction_state()` may
remain temporarily as a compatibility wrapper that delegates to the generic
writer with explicit Object Classification paths.

Logical paths map to physical AnnData write units as follows:

- any dirty obs-column path writes the complete encoded `obs` dataframe,
  because its index, column order, missing values, and categorical encodings
  form one AnnData element;
- an obsm path writes only the named obsm entry;
- an uns path writes only the named top-level or nested metadata entry while
  preserving unrelated siblings;
- a removed live path deletes only its corresponding encoded element;
- multiple logical paths that belong to one consistency contract may be
  grouped into one persistence unit.

After a successful set of element creates, updates, or removals, the generic
writer consolidates zarr metadata before returning a successful result. This is
required because direct element writes may be readable through non-consolidated
access while a normal reopened SpatialData still sees the old consolidated
hierarchy. If element writing or metadata consolidation fails, no dirty
revision is acknowledged; all captured paths remain dirty and retryable.

Object Classification is the first deferred-write consumer of this generalized
path. It records every obs column and uns configuration/color key it actually
changes with `record_table_mutation()` and later uses PersistenceController.

Feature Extraction has different persistence semantics. For backed SpatialData,
Harpy's feature-matrix operation already writes both
`TableComponentPath("obsm", (feature_key,))` and
`TableComponentPath("uns", ("feature_matrices", feature_key))` and consolidates
metadata before returning. Feature Extraction must not write those elements a
second time through PersistenceController; it calls
`record_persisted_table_change()` after Harpy succeeds. For unbacked
SpatialData, the same logical paths are in-memory changes and are passed to
`record_table_mutation()`. Existing semantic events remain available to
domain-specific consumers in both cases.

spatial_canonical is an obsm entry, not an obs column. Its persisted pair is:

    adata.obsm["spatial_canonical"]
    adata.uns["spatial_coordinates"]["spatial_canonical"]

When that pair is dirty, the persistence service writes only those two logical
elements using AnnData encodings, conceptually:

    ad.io.write_elem(table_group["obsm"], "spatial_canonical", ...)
    ad.io.write_elem(
        table_group["uns"]["spatial_coordinates"],
        "spatial_canonical",
        ...,
    )

The implementation creates a correctly encoded spatial_coordinates mapping
when absent and preserves every unrelated sibling entry when it already exists.
It does not rewrite all of obsm or uns.

The spatial annotation target, for example
adata.obs["spatial_annotation"], is separate. If any obs column is dirty, the
existing AnnData element-level persistence path writes the obs dataframe element
with its index, column-order, missing-value, and categorical encodings. Writing
obs is still a selective table-component write; it does not serialize the full
AnnData object. Existing supported classifier metadata and future components are
included only when their paths are present in the same shared dirty snapshot.

There is one shared Write Table State action rather than competing widget-local
writers. A centroid-only dirty table writes only the spatial_canonical pair; an
annotation-only dirty table writes obs; a table containing both changes writes
both in the same operation.

The matrix and its spatial-coordinate metadata are one persistence unit. The
service stages/backs up as needed, validates both encoded elements, and must not
leave a newly written matrix paired with old or missing metadata after a handled
failure. The table remains dirty unless every component in the captured write
set succeeds. Captured component revisions are cleared only if they have not
changed during the write; later mutations remain dirty.

The action and success message identify the table, resolved store path, and
components written. A successful write makes newly generated centers reusable
after reload/reopen when their stored structural metadata still matches the
labels element and table. A failure shows an error, preserves usable in-memory
state, and leaves the table dirty.

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

The persistence-foundation slice retains this full validated table-state
reload. It does not introduce selective per-path reload: independently
replacing parts of obs, obsm, and uns could create a mixed in-memory state.
Canonical metadata is parsed and validated from the disk snapshot before the
later canonical integration accepts the replacement.

After success:

- re-inspect spatial_canonical and its metadata;
- refresh table metadata and compatible target columns;
- preserve a target selection only if still valid;
- notify consumers through the semantic table replacement event;
- clear dirty state;
- show the source path and outcome.

Late worker results created before reload must never update a cache, open a
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
- labels has no readable 2D scale0 array;
- Shapes or labels is unavailable in the selected coordinate system;
- a required transform is missing, non-finite, unsupported, or non-invertible;
- no linked table is selected;
- table linkage metadata is missing/inconsistent;
- selected-region instance IDs are missing, duplicated, non-positive, or not
  integer-like;
- no valid target mode/column is configured;
- a new column name is empty, invalid, reserved, or colliding;
- an existing target column has an incompatible dtype;
- a calculation/query is already running for the same request.

Runtime outcomes are distinct:

- no centers inside annotation: neutral information;
- table instance absent from labels during center calculation: binding error;
- stale cache detected: informational refresh state;
- invalid/corrupt managed cache: mismatch report followed by recalculation and
  replacement only after a valid result is available;
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
- Busy, cancellation, success, and failure states are textual.
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
                atomic cache updates and rollback support

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
                worker lifecycle and operation IDs
                stale-result handling
                cache update, apply, and undo orchestration

            widget.py
                selectors, cache status, busy state, persistence actions

            dialogs.py
                cache mismatch reporting
                Apply Spatial Annotation dialog

            status_card.py
                pure status-card specification builders

The core spatial_query package must remain UI independent. It must not import
Qt, napari widgets, HarpyAppState, or widget controllers. The widget controller
orchestrates the pure core operations with shared application services.

General concerns stay outside the feature package:

- shared table persistence and reload services;
- shared application state and per-component revisions;
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
- the generalized PersistenceController and Qt-independent
  ad.io.write_elem-based component persistence path;
- active-coordinate-system selection patterns;
- styles, status cards, worker cleanup, and operation-ID patterns.

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
- source element, dimensions, shape, or dtype mismatch triggers a reported
  refresh;
- rechunking alone does not invalidate canonical centers;
- transform-only changes do not stale intrinsic centers;
- instance-set digest and finite-coordinate validation;
- a valid Run calculates one selected-region digest and never hashes unrelated
  registered regions;
- applying a cache update rechecks the selected-region digest and hashes other regions
  only when deciding whether to preserve them;
- multi-region incremental fill preserves other valid regions;
- an all-regions mismatch rebuilds the managed matrix and metadata only after recalculation
  succeeds;
- forced recalculation bypasses valid reuse, replaces only the selected region,
  and preserves all other valid regions;
- failed, cancelled, or stale forced recalculation preserves the prior pair
  exactly;
- cache-update rollback restores matrix/metadata exactly;
- cache create/extend/refresh/rebuild marks shared dirty once;
- cache parser never mutates during inspection.

### RasterAggregator adapter tests

- 2D Dask scale0 is wrapped as singleton-z without eager computation;
- integer labels dtype is required, while SpatialData's Dask raster contract is
  trusted rather than revalidated locally;
- table instance IDs are passed as index and background is excluded;
- no global unique-label discovery when index is supplied;
- one-pixel labels, irregular labels, disjoint components, concave labels, and
  labels spanning chunks have correct centers;
- y/x to x/y conversion and integer pixel-center convention;
- output instance IDs must exactly match the requested IDs and order;
- absent label ID, mismatched result IDs, NaN, and inf are rejected;
- no NumPy/full-array labels fallback;
- only scale0 is read;
- cancellation/stale completion cannot update the cache;
- worker wrapping does not change Harpy's Dask scheduling policy.

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
- absent/partial/stale cache follows calculate-update-query phases;
- worker never mutates AnnData or Qt;
- cache-update payload is applied only after main-thread revalidation;
- result accepted only for unchanged request;
- every selection/source-signature invalidation drops late results;
- older run cannot supersede newer run;
- cancellation prevents cache update/dialog/mutation;
- reload freezes and invalidates pending work;
- worker errors restore usable controls and give feedback;
- cleanup disconnects workers/signals when widget closes.

### Widget tests

- dependent combo filtering and stable selection preservation;
- default spatial_annotation existing/new behavior;
- Run enablement/tooltips for every blocker;
- dirty Shapes session blocker;
- all centroid cache status states and phase text;
- Recalculate centroids enablement is based on labels/table binding rather than
  Shapes or annotation-target validity;
- Recalculate centroids bypasses cache reuse, shows textual busy/cancellation
  status, refreshes status on success, and never starts a query or annotation
  flow;
- invalid-cache mismatch reporting and automatic rebuild state;
- result dialog contents and centroid predicate wording;
- live conflict recount and value validation;
- mandatory overwrite warning and explicit action text;
- no-result flow does not change annotation column;
- cancel after cache update leaves cache dirty state visible;
- apply/undo states and summaries;
- shared dirty indicator across widgets;
- write enabled only for backed dirty table;
- dirty reload write/discard/cancel branches;
- reload re-inspects cache and invalidates undo;
- accessibility, keyboard behavior, and long-name tooltips.

### Backed-zarr integration tests

- first Run calculates centers lazily and applies the cache update in memory;
- first Run reads scale0 but not lower-resolution levels;
- second Run with valid cache reads no label chunks;
- cache is not on disk until Write Table State;
- a centroid-only write uses AnnData element encoding for only
  obsm/spatial_canonical and
  uns/spatial_coordinates/spatial_canonical, preserving unrelated table
  elements and sibling metadata;
- an annotation-only write persists obs, including column order, missing values,
  categorical values/categories, and the dataframe index, without rewriting the
  full AnnData table;
- a mixed dirty manifest writes obs plus the canonical pair and clears dirty
  state only after all captured component revisions succeed;
- an unknown dirty component blocks persistence and is never silently cleared;
- no path calls AnnData.write_zarr or rewrites X, layers, var, varm, obsp, or
  unrelated obsm/uns entries;
- reopen/reload reuses cache when structural metadata still matches;
- structural labels/table mismatch triggers refresh rather than reuse;
- reload discards an unpersisted cache after confirmation;
- injected failure between canonical matrix and metadata writes restores or
  preserves a consistent on-disk pair, retains dirty state, and keeps the
  in-memory cache usable;
- a mutation accepted during persistence remains dirty after the older write
  completes;
- reload validation failure preserves current table and dirty state;
- canonical-center, spatial-annotation, and object-classification changes
  coexist in one write;
- late center/query completion after reload cannot affect the table.

### Background calculation tests

- calculation runs through a worker and returns a cache-update payload without
  mutating AnnData;
- only the current operation ID can apply a returned payload;
- invalidated, cancelled, and late worker signals are ignored;
- accepted payloads are revalidated and applied on the main thread;
- the status card reports a textual busy state without progress telemetry;
- Harpy's Dask scheduler and performance behavior are not reimplemented or
  regression-gated in napari-harpy.

## Observability

Log structured diagnostic context for:

- cache inspection state and reason;
- center calculation start/cancel/stale-drop/success/failure;
- source scale0 dimensions, shape, dtype, and requested instance count;
- cache-update action, covered region/count, schema/algorithm version;
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

### Slice 1a: Canonical metadata and cache lifecycle

**Implementation status: Implemented.**

Slice 1a defines and safely manages the canonical cache without calculating any
centroids. Tests use synthetic x/y arrays and must not invoke RasterAggregator
or read labels chunks.

Deliverables:

- typed cache state, mismatch report, metadata, source signature,
  selected-region binding, cache-update payload, and cache-update-result
  contracts;
- spatial_coordinates/spatial_canonical schema version 1 using values supported
  by AnnData's zarr encoding;
- strict parser/builder plus a non-mutating inspector;
- structural labels signature covering source element, scale0 dimensions,
  shape, and dtype; chunking is excluded from persisted metadata and cache
  validity;
- a selected-region binding identity covering region/instance keys,
  selected-row count, and a deterministic instance_set_digest over the sorted
  unique instance IDs for that region; the canonical digest input also includes
  the labels name and a schema/domain tag;
- one exact, versioned digest encoding implemented and pinned by test vectors,
  including vectorized sorted big-endian uint64 instance-ID representation;
- selected-region binding validation that rejects zero matching rows and
  missing, non-positive, non-integer-like, duplicate, or uint64-overflowing
  instance IDs without creating or deleting table rows;
- matrix/metadata validation for fixed z/y/x shape, dtype, axes, finite
  registered-region coordinates, 2D z=0.0, and multi-region coverage;
- deterministic absent, partial, valid, stale, and invalid cache-state
  classification with structured mismatch reasons;
- synthetic cache-update payload construction and an atomic update operation that:
  - creates an n_obs by 3 NaN-initialized matrix when absent;
  - fills or replaces only the selected region's current row positions;
  - preserves every other still-valid region and its metadata;
  - rebuilds the managed matrix and metadata and drops unsubstantiated region
    entries after an all-regions matrix/top-level mismatch;
  - restores the complete previous obsm/uns pair if validation or assignment
    fails;
- cache-update results that report the performed create, extend, refresh, or
  rebuild action together with the fresh inspection's mismatch reasons;
- fixtures and tests for single- and multi-region missing, reusable, partial,
  stale, region-mismatched, and all-regions-invalid states.

#### Slice 1a typed API

The public contracts live in `core/spatial_query/models.py`, and the operations
that build, parse, inspect, and apply them live in
`core/spatial_query/canonical.py`. Stable consumers import the intentional
exports from `napari_harpy.core.spatial_query` rather than either implementation
module directly.

Use string enums for cache state, cache-update action, and mismatch code:

    CanonicalCacheState = absent | partial | valid | stale | invalid
    CanonicalCacheUpdateAction = create | extend | refresh | rebuild
    CanonicalMismatchCode:
        matrix_without_metadata
        metadata_without_matrix
        matrix_invalid
        metadata_invalid
        schema_version_unsupported
        top_level_contract_mismatch
        region_not_registered
        region_metadata_invalid
        source_signature_mismatch
        table_signature_mismatch
        algorithm_version_mismatch
        region_coordinates_invalid

Mismatch codes, rather than human-readable messages, drive tests and controller
behavior. All-regions versus region-local scope is derived from the mismatch
code rather than stored as a second independently configurable field. Keep the
code set at behaviorally meaningful categories; the bounded detail identifies
the particular malformed field or expected/actual value. For example, wrong
rank, shape, storage type, or matrix dtype uses `matrix_invalid`, while a
disagreement among supported top-level `obsm_key`, axes, coordinate dtype, or
linkage keys uses `top_level_contract_mismatch`.

Cache-state classification follows this deterministic evaluation order:

1. Inspect the two managed paths. If neither
   `obsm/spatial_canonical` nor
   `uns/spatial_coordinates/spatial_canonical` exists, return `absent`. The
   presence of the parent `spatial_coordinates` registry without its
   `spatial_canonical` entry still counts as absent.
2. If exactly one managed path exists, return `invalid` with
   `matrix_without_metadata` or `metadata_without_matrix`.
3. If both exist but the matrix, schema, top-level contract, or strict metadata
   structure is malformed, contradictory, or unsupported, return `invalid`.
   All-regions invalidity takes precedence over every region-local outcome.
4. Once the shared cache is valid, if the selected region has no metadata entry,
   return `partial` with `region_not_registered`.
5. If the selected region entry exists and is interpretable but its source
   signature, table coverage, supported algorithm version, or finite complete
   coordinate coverage does not match current state, return `stale` with the
   corresponding region-local mismatch code.
6. If all selected-region checks pass, return `valid`.

`CanonicalCacheReport.state` is a derived property rather than stored alongside
the mismatch tuple. All-regions mismatches derive `invalid`,
`region_not_registered` derives `partial`, other non-empty region-local
mismatches derive `stale`, no mismatches plus no readable stored metadata
derives `absent`, and no mismatches plus readable stored metadata derives
`valid`.

The report state and region-local mismatches describe the selected region only.
Ordinary inspection does not calculate live source signatures, rebuild region
bindings, or calculate instance-set digests for every other registered region. A stale but
structurally interpretable entry for another region therefore does not downgrade
an otherwise valid selected region and is evaluated only if a later
cache update proposes to preserve it. A malformed region entry discovered by
the strict metadata parser can still make the shared registry `invalid` for
`all_regions`. Cache-report mismatch tuples are ordered deterministically with
all-regions reasons first and selected-region reasons second, preserving
validation order within each group.

The immutable value contracts are conceptually:

    CanonicalSourceSignature:
        labels_name
        source_scale = "scale0"
        dims
        shape
        dtype

    CanonicalRegionBinding:
        table_name
        labels_name
        region_key
        instance_key
        row_positions
        instance_ids
        instance_set_digest
        n_obs property derived from instance_ids

    CanonicalRegionMetadata:
        source_signature: CanonicalSourceSignature
        n_obs
        instance_set_digest
        algorithm_version
        generated_by_package or None
        generated_by_version or None
        generated_at or None

    CanonicalMetadata:
        schema_version
        region_key
        instance_key
        regions: mapping[str, CanonicalRegionMetadata]

    CanonicalCacheMismatch:
        code: CanonicalMismatchCode
        region or None
        bounded user-facing detail or None
        scope property derived from code: all_regions | region

    CanonicalCacheReport:
        stored_metadata: CanonicalMetadata or None
        source_signature: CanonicalSourceSignature
        binding: CanonicalRegionBinding
        mismatches: tuple[CanonicalCacheMismatch, ...]
        state property derived from stored_metadata and mismatches
        table_name property derived from binding
        labels_name property derived from binding

    CanonicalCacheUpdatePayload:
        table_name property derived from binding
        labels_name property derived from binding
        binding: CanonicalRegionBinding
        centers with shape (n_instances, 3) in z, y, x order
        source_signature

    CanonicalCacheUpdateResult:
        action: CanonicalCacheUpdateAction
        mismatches: tuple[CanonicalCacheMismatch, ...]

The source-signature value type has this concrete shape:

```python
type SpatialDimension = Literal["z", "y", "x"]


@dataclass(frozen=True)
class CanonicalSourceSignature:
    labels_name: str
    source_scale: Literal["scale0"]
    dims: tuple[SpatialDimension, ...]
    shape: tuple[int, ...]
    dtype: str

    def __post_init__(self) -> None:
        if not self.labels_name:
            raise ValueError("Source labels name must not be empty.")

        if self.source_scale != "scale0":
            raise ValueError(
                "Canonical coordinates must use labels source scale `scale0`."
            )

        if not self.dims:
            raise ValueError("Source dims must not be empty.")

        if len(self.dims) != len(self.shape):
            raise ValueError("Source dims and shape must have equal lengths.")

        if len(set(self.dims)) != len(self.dims):
            raise ValueError("Source dims must be unique.")

        if any(dim not in ("z", "y", "x") for dim in self.dims):
            raise ValueError(
                "Source dims must contain only `z`, `y`, and `x`."
            )

        if any(
            isinstance(size, bool)
            or not isinstance(size, int)
            or size <= 0
            for size in self.shape
        ):
            raise ValueError("Source shape must contain positive integers.")

        if not self.dtype:
            raise ValueError("Source dtype must not be empty.")

    @property
    def ndim(self) -> int:
        return len(self.dims)
```

The builder normalizes source shape entries to built-in Python integers before
constructing this value. The dataclass enforces dimension-independent
structural invariants; schema- and algorithm-specific dimensionality rules
belong to the metadata builder and parser.

`CanonicalCacheReport` is both the non-mutating inspector result and the typed
mismatch report; do not add a second inspection-result wrapper with the same
information. `CanonicalRegionBinding` arrays and cache-update-payload arrays
are normalized eager NumPy arrays and are made read-only at the contract
boundary. The metadata regions mapping is defensively copied and exposed as a
read-only mapping rather than retaining a caller-owned mutable dictionary.

`CanonicalSourceSignature.dims` records the exact ordered scale0
`DataArray.dims`, and `shape` records the corresponding extent in the same
order. Construction requires equal tuple lengths, unique non-empty dimension
names, and positive integer shape entries. Schema version 1 additionally
requires `dims == ("y", "x")`; a future 3D schema can use
`dims == ("z", "y", "x")` and a three-entry shape without renaming the value
contract. Do not persist a redundant `ndim`; it is `len(dims)`. These source
dimensions are distinct from the fixed top-level
`axes == ("z", "y", "x")`, which describe the column order of the canonical
coordinate matrix. A 2D source therefore has dims ("y", "x") while its stored
centers use z, y, x with z=0.0.

##### Instance-set digest encoding

The instance-set digest hashes the semantic selected-region instance
membership, not AnnData row identities, an AnnData file, a zarr group, or any
other storage serialization. Implement it with Python's standard-library
`hashlib.sha256()` and a vectorized NumPy canonicalization path. Do not encode or
hash instance IDs one at a time in a Python loop.

Digest encoding version 1 uses these exact primitives:

- `U16(n)`: unsigned two-byte big-endian integer;
- `U64(n)`: unsigned eight-byte big-endian integer;
- `LP(value)`: `U64(len(value))` followed by the bytes in `value`;
- the labels name is encoded as exact UTF-8 without trimming, case folding, or
  Unicode normalization;
- every normalized instance ID is encoded as one fixed-width `U64` value.

The fixed domain bytes are:

    b"napari-harpy/spatial-canonical/instance-set"

For labels name `L` and normalized unique positive instance IDs `I_i`, sort the
IDs numerically and encode the sorted array as contiguous big-endian uint64.
The exact SHA-256 input is:

    LP(domain)
    || U16(1)
    || LP(L.encode("utf-8"))
    || U64(number_of_ids)
    || U64(I_1)
    || U64(I_2)
    || ...
    || U64(I_n)

The corresponding implementation is expected to follow this pseudocode:

```python
import hashlib
from collections.abc import Sequence

import numpy as np

_DIGEST_DOMAIN = b"napari-harpy/spatial-canonical/instance-set"
_DIGEST_ENCODING_VERSION = 1


def _encode_u64(value: int) -> bytes:
    return value.to_bytes(8, byteorder="big", signed=False)


def _update_length_delimited(hasher, value: bytes) -> None:
    hasher.update(_encode_u64(len(value)))
    hasher.update(value)


def build_instance_set_digest(
    labels_name: str,
    instance_ids: Sequence[int],
) -> str:
    canonical_ids = np.sort(
        np.asarray(instance_ids, dtype=np.uint64)
    ).astype(">u8", copy=False)

    hasher = hashlib.sha256()
    _update_length_delimited(hasher, _DIGEST_DOMAIN)
    hasher.update(
        _DIGEST_ENCODING_VERSION.to_bytes(
            2,
            byteorder="big",
            signed=False,
        )
    )
    _update_length_delimited(hasher, labels_name.encode("utf-8"))
    hasher.update(_encode_u64(len(canonical_ids)))
    hasher.update(memoryview(canonical_ids).cast("B"))
    return f"sha256:{hasher.hexdigest()}"
```

This pseudocode assumes `instance_ids` already contains unique, validated,
normalized positive integers that fit in uint64. The canonical binding
validator must reject invalid values before NumPy conversion so casting cannot
wrap or silently admit negative, fractional, Boolean, string, missing, or
overflowing values. The contiguous memoryview prevents an additional Python
bytes copy and supplies all sorted IDs to `hashlib` in one update.

Here `||` describes byte-sequence concatenation for the encoding contract. The
stored value is `"sha256:" + hasher.hexdigest()`, using the lowercase
64-character hexadecimal digest returned by `hashlib`.

Focused digest tests pin the exact encoding, order independence, labels-name
sensitivity, and instance-membership sensitivity. Binding tests separately
demonstrate that obs-name changes, table row reordering, and same-set
reassignment preserve the same identity. Non-integer values, including
integer-like floats, are rejected rather than normalized. File bytes, AnnData
serialization, zarr chunking, table row order, and obs_names never enter the
digest. A representative 400,000-ID benchmark must guard against regression to
per-ID Python hashing; observed performance should remain in the low tens of
milliseconds on a typical development machine rather than becoming a hard,
platform-sensitive CI timing assertion.

##### Digest frequency and cache/query flow

An authoritative cache inspection for Run calculates the instance-set digest
once for the selected labels region. It does not calculate digests for all
regions in a multi-region table. This selected-region calculation is required
for every spatial-query Run so an out-of-band addition, removal, or replacement
of an instance ID cannot silently reuse stale centers. With the vectorized
uint64 encoding, hashing 400,000 already-normalized IDs is expected to take low
tens of milliseconds; table filtering and numeric validation are additional
bounded table work and never read labels chunks.

The normal valid-cache path is:

1. resolve the selected region's current rows and normalize its instance IDs;
2. calculate its current instance-set digest and source signature;
3. compare them with the selected region's stored metadata;
4. when valid, reuse the selected rows of `spatial_canonical` and run the query
   without another digest calculation or any labels read.

The calculation-and-cache-update path calculates the selected-region digest
twice:

1. initial inspection calculates it and captures the selected-region binding
   in the immutable calculation request;
2. immediately before applying the cache update, the update operation resolves the current
   selected-region bindings and calculates it again;
3. a changed instance set rejects the stale payload without mutation;
4. an unchanged set lets the update operation map calculated centers by instance ID
   onto the current row positions and apply the update atomically.

Other registered regions are not hashed during an ordinary selected-region
query. When an extend or region-local refresh proposes to preserve existing
other-region coordinates, the update operation validates the source signature, table
signature, digest, and finite coverage of each preservation candidate at that
time. Regions that are not fully validated are dropped from the candidate
metadata. A conservative selected-region-only rebuild need not hash regions it
will not preserve.

The widget may perform an earlier inspection to display cache status. Such a
report may be memoized against the shared table/cache generation to prevent
incidental UI refreshes from repeating the digest, but it is not authoritative
for a later Run: Run performs its own fresh selected-region inspection. If an
annotation result remains open before Apply, apply-time table/cache
revalidation performs another selected-region digest unless the operation has
an equivalent authoritative fresh signature from the same guarded generation.
No status, Run, cache-update, or Apply path hashes every table region by
default.

Schema-v1 constants are not configurable dataclass fields. `obsm_key`, axes,
matrix dtype, source element type, scale, coordinate-frame type, calculation
method, weighting, background value, and pixel-coordinate convention are
written by the builder and required exactly by the strict parser. The typed
metadata therefore represents variable data without allowing callers to
construct unsupported calculation semantics. Provenance remains optional and
does not participate in cache validity.

The public operation surface is:

    build_canonical_source_signature(sdata, labels_name)
        -> CanonicalSourceSignature

    build_canonical_region_binding(table, table_metadata, labels_name)
        -> CanonicalRegionBinding

    build_instance_set_digest(labels_name, instance_ids)
        -> str

    build_canonical_metadata(...)
        -> CanonicalMetadata

    parse_canonical_metadata(value)
        -> CanonicalMetadata

    canonical_metadata_to_storage(metadata)
        -> dict[str, object]

    inspect_canonical_cache(sdata, *, table_name, labels_name)
        -> CanonicalCacheReport

    build_canonical_cache_update_payload(...)
        -> CanonicalCacheUpdatePayload

    apply_canonical_cache_update(sdata, payload)
        -> CanonicalCacheUpdateResult

The exact builder keyword layout may evolve during implementation, but these
operation boundaries and return contracts are stable. Parsing and inspection
must not mutate AnnData, SpatialData, or stored metadata. In particular, the
inspector must not call an existing helper through a code path that normalizes
SpatialData table attrs in place.

`CanonicalCacheUpdatePayload` deliberately carries the calculation-time
`CanonicalRegionBinding`, whose instance IDs are authoritative but whose row
positions are not. Immediately before applying the cache update, the update
operation rebuilds the
current region binding, verifies the source signature and binding identity, and
maps the payload binding's instance IDs onto current row positions. A normal
AnnData row reorder can therefore complete safely. A changed instance set
rejects the payload; a same-set row reassignment is remapped safely during
cache update but requires semantic invalidation to prevent reuse of an older
cache.

The cache-update operation derives its action from a fresh inspection rather than trusting
the state observed before calculation:

- absent produces create;
- partial produces extend;
- stale produces refresh;
- invalid produces rebuild;
- valid is reused by the caller and normally does not reach the cache-update operation;
  forced recalculation of a valid region produces refresh.

Before the first assignment, the cache-update operation constructs and validates the
complete candidate matrix and metadata registry in local values. An internal,
non-public assignment helper records in local rollback variables whether each
managed path existed and its complete prior value. If either assignment fails,
rollback restores both exact prior path states, including absence. A successful
update returns only its action and the fresh inspection's mismatch reasons; the
two mutated managed paths are fixed by the canonical-cache contract rather than
repeated in the result.

Reuse existing core types only where their contracts match. In particular,
`SpatialDataTableMetadata` remains the shared linkage value type. The canonical
binding validator adds the stricter selected-region rules required here.
Existing general feature-matrix normalization is not reused for
`spatial_canonical`, because it permits sparse matrices, one-dimensional
reshaping, and dtype coercion that the canonical schema must reject. The
inspector reuses the existing `get_table_metadata()` helper for read-only table
linkage metadata and accesses the AnnData table explicitly through
`sdata.tables[table_name]`.

Exit criteria:

- metadata can be built, serialized, parsed, and validated deterministically;
- cache inspection performs no labels computation or labels-chunk reads;
- every cache state and mismatch reason is covered by tests;
- each region entry carries its own instance-set digest; row reordering and
  obs-name changes preserve it, while a region's instance membership change
  invalidates it;
- same-set instance-key reassignment is pinned as an explicitly undetectable
  structural case, and the contract requires later semantic-event integration
  to invalidate affected region entries before reuse for supported obs/linkage
  mutations;
- a valid shared matrix refresh changes only the selected region's current row
  positions and metadata, leaving other valid regions byte-for-byte unchanged;
- NaN occurs only in rows for regions without valid coverage metadata;
- region-local refresh preserves other valid regions, while an all-regions rebuild
  never preserves metadata that no longer describes the shared matrix;
- no cache-update failure leaves partially replaced obsm/uns state;
- the structural-validation limitation for undetectable same-signature pixel
  edits is documented;
- no SpatialData-level Harpy revision attributes or affine snapshots are
  introduced;
- the slice has no RasterAggregator, Qt, napari-layer, or background-worker
  dependency.

### Slice 1b: Harpy centroid construction and cache ensure

**Implementation status: Implemented.**

Slice 1b calculates the values consumed by Slice 1a and supplies the blocking,
UI-independent ensure operation. The thin background calculation boundary and
late-result safety remain Slice 2 responsibilities; complete query-controller
orchestration remains Slice 7.

Deliverables:

- an exact-scale0 labels lookup that uses SpatialData's validated 2D Dask raster
  representation directly and never falls back to NumPy or a lower-resolution
  scale;
- preflight validation that every normalized positive table instance ID is
  representable by the labels dtype before starting the global aggregation;
- a RasterAggregator adapter that:
  - adds the singleton z axis lazily without copying or rechunking;
  - explicitly uses run_on_gpu=False so behavior does not depend on whether
    CuPy happens to be installed;
  - passes exactly the selected table-region instance IDs as index;
  - converts returned z/y/x centers to float64 without dropping or reordering
    axes and requires z=0.0 for the supported 2D source;
  - requires returned instance IDs to exactly match the requested IDs and order,
    following RasterAggregator's `center_of_mass(index=...)` contract;
- selective raster-membership validation from the requested aggregation result:
  every requested ID must have exactly one finite center, missing requested IDs
  raise before mutation, and raster IDs absent from the table are neither
  calculated nor globally enumerated;
- a UI-independent calculation operation with this exact public boundary:

      calculate_canonical_centers(
          sdata: SpatialData,
          report: CanonicalCacheReport,
      ) -> CanonicalCacheUpdatePayload

  The operation consumes the calculation-time source signature and selected-region
  binding already captured by `inspect_canonical_cache()`. It calculates and
  validates centers for exactly that binding, then returns the existing Slice 1a
  cache-update payload carrying the same calculation-time identity. It does not
  mutate the table or update the cache; applying the update remains the responsibility
  of `apply_canonical_cache_update()`;
- an immutable canonical-centers result with this concrete shape:

      @dataclass(frozen=True)
      class CanonicalCentersResult:
          source_signature: CanonicalSourceSignature
          binding: CanonicalRegionBinding
          centers: NDArray[np.float64] = field(
              repr=False,
              compare=False,
          )
          cache_update: CanonicalCacheUpdateResult | None

          @property
          def table_name(self) -> str:
              return self.binding.table_name

          @property
          def labels_name(self) -> str:
              return self.binding.labels_name

          @property
          def n_obs(self) -> int:
              return self.binding.n_obs

          @property
          def reused(self) -> bool:
              return self.cache_update is None

  `centers` is a read-only float64 array with shape `(binding.n_obs, 3)` and
  fixed z, y, x column order. Row `i` belongs to
  `binding.instance_ids[i]`. Consumers associate centers through these instance
  IDs, not through `binding.row_positions`; row positions are a snapshot and may
  become outdated after table reordering. `source_signature` describes the live
  labels raster, while `binding` describes the live linked table rows and
  instance IDs. They are independent calculation-time snapshots whose shared
  invariant is the selected labels name. The transient binding stores
  `table_name` once; the cache report, cache-update payload, and centers result
  expose it as a derived property;
- an ensure operation with this exact public boundary:

      ensure_canonical_centers(
          sdata: SpatialData,
          *,
          table_name: str,
          labels_name: str,
          force_recalculation: bool = False,
      ) -> CanonicalCentersResult

  The operation:
  - reuses a structurally valid selected-region cache without reading labels;
  - calculates spatial_canonical plus metadata and applies the cache update when absent;
  - accepts an explicit forced-recalculation mode that bypasses valid reuse;
  - treats a selected region's instance-set digest mismatch as stale and
    recalculates that complete region;
  - delegates every table mutation and rollback to the Slice 1a cache-update operation;
  - returns selected-region centers rather than the complete table matrix;
  - sets `cache_update` to `None` when it reuses a valid cache, otherwise to
    the `CanonicalCacheUpdateResult` returned by `apply_canonical_cache_update()`.
    That nested result contains only the performed action and the fresh
    inspection's mismatch reasons; table/region identity and row count remain
    available from the centers result's binding;
- representative zarr-backed Dask fixtures and integration tests, including
  labels spanning chunks and requested IDs absent from the raster.

Exit criteria:

- one UI-independent blocking operation can ensure valid canonical centers for
  a selected labels/table region;
- its forced mode recalculates a valid selected-region cache instead of reusing
  it and retains Slice 1a's validation and atomic cache-update guarantees;
- a valid cache is reused with zero labels-chunk reads;
- missing or mismatched cache data is calculated from Dask-backed scale0
  without loading the labels raster into RAM;
- no global unique-label discovery is used: only selected table IDs are
  requested, while all scale0 chunks may still be scanned lazily;
- a table ID absent from the raster produces an actionable binding error and no
  obsm/uns mutation; raster IDs absent from the table produce no center;
- calculated output exactly follows the requested instance-ID order and is
  finite for every selected-region row;
- instance IDs outside the labels dtype range fail before Dask work;
- no calculation failure reaches the Slice 1a cache-update operation or changes obsm/uns;
- domain modules have no Qt dependency.

### Slice 2: Background canonical calculation boundary

**Implementation status: Implemented.**

The required mutation boundary is:

    main thread
        inspect_canonical_cache() and capture CanonicalCacheReport
            ↓
    worker
        calculate_canonical_centers()
        return CanonicalCacheUpdatePayload
            ↓
    main thread
        reject a cancelled, invalidated, or outdated job
        apply_canonical_cache_update()

The worker never calls `ensure_canonical_centers()`, because that blocking
operation includes cache mutation. The existing blocking ensure remains the
UI-independent convenience operation for synchronous callers.

Deliverables:

- a thin worker wrapper that calls `calculate_canonical_centers()` and returns
  its `CanonicalCacheUpdatePayload` without calling
  `apply_canonical_cache_update()` or otherwise mutating AnnData;
- the same monotonically increasing operation ID, active-operation-phase,
  `worker.quit()`, and late-signal rejection pattern used by existing
  napari-harpy controllers;
- an accepted-result boundary that applies the payload only on the main thread,
  where the existing fresh source/binding validation remains authoritative;
- textual busy, cancellation, success, and failure status without a progress
  bar, percentage, scheduler diagnostics, or performance telemetry;
- focused worker tests for accepted, invalidated, cancelled, late, and errored
  results.

Exit criteria:

- labels I/O and centroid calculation run outside the Qt main thread;
- the worker returns the existing payload contract and never mutates AnnData;
- cancellation or invalidation immediately prevents result acceptance even if
  the underlying Harpy call later completes;
- only the current operation ID may reach
  `apply_canonical_cache_update()` on the main thread;
- napari-harpy does not override or regression-gate Harpy's Dask scheduling,
  concurrency, memory, or calculation performance.

### Slice 3a: General table-state events and component persistence

This slice generalizes the existing Object Classification-oriented persistence
infrastructure before Spatial Query depends on it. It introduces no canonical
cache behavior and no second persistence controller.

Deliverables:

- the validated immutable `TableComponentPath` contract for obs columns,
  individual obsm entries, and top-level or nested uns entries;
- `TableStateChangedEvent` carrying SpatialData/table identity, unique component
  paths, affected regions, change kind, and source;
- coexistence with existing domain events such as
  `FeatureMatrixWrittenEvent` and `ClassificationTableWrittenEvent`, which
  remain available for feature-specific consumers;
- a HarpyAppState per-table dirty manifest mapping each logical path to its
  latest revision, with table-level `is_table_dirty()` derived from that
  manifest;
- explicit `record_table_mutation()`, `record_persisted_table_change()`, and
  `record_table_reload()` acceptance methods so publishing an event does not
  implicitly mean that its paths are dirty;
- `TableDirtySnapshot`, `TableComponentWriteResult`, and acknowledgement
  operations implementing the documented snapshot/write/acknowledge boundary
  and clearing only successfully persisted path revisions that are still
  current;
- a generalized `PersistenceController` coordinating selection, dirty
  snapshots, full validated table-state reload, and Qt-independent core
  persistence functions;
- component writers for encoded obs, individual obsm entries, and top-level or
  nested uns entries, including removal, unsupported-path preflight, and zarr
  metadata consolidation before successful acknowledgement;
- migration of Object Classification as the first deferred-write consumer,
  declaring every logical path it changes while retaining its semantic domain
  event;
- publication of Feature Extraction's backed Harpy writes through
  `record_persisted_table_change()`, without routing those already-persisted
  elements through PersistenceController; unbacked changes use
  `record_table_mutation()`;
- a temporary compatibility path for `write_table_prediction_state()` if
  needed by existing headless callers;
- focused app-state, persistence, Object Classification, and Feature Extraction
  regression tests.

Exit criteria:

- one event accepted through `record_table_mutation()` assigns one new revision
  to all of its paths; publishing an event alone does not alter dirty state;
- a table is dirty exactly when its component manifest is non-empty;
- writing an obs-column path writes the complete encoded obs dataframe but no
  unrelated AnnData component;
- writing an obsm or uns path touches only its supported encoded element and
  preserves unrelated siblings;
- unsupported paths fail before any write and are never silently cleared;
- a mutation accepted during persistence remains dirty when its captured older
  revision completes;
- a write is acknowledged only after zarr metadata consolidation succeeds, and
  a normally reopened SpatialData sees created, updated, and removed elements;
- a consolidation failure leaves every captured path dirty;
- backed Feature Extraction remains clean for the exact paths Harpy already
  persisted, while unbacked Feature Extraction records those paths as dirty;
- existing Object Classification and Feature Extraction behavior and semantic
  events remain valid;
- no path calls `AnnData.write_zarr()` or introduces a competing widget-local
  dirty truth.

### Slice 3b: Canonical cache state, invalidation, and persistence

This slice wires the canonical cache into the shared foundation from Slice 3a.
Core canonical operations remain independent from Qt and HarpyAppState.

Deliverables:

- a Spatial Query controller callback that publishes a
  `TableStateChangedEvent` only after an accepted main-thread canonical cache
  update; synchronous core callers remain responsible for publishing an app
  event when used inside a shared UI session;
- one accepted cache event containing both
  `TableComponentPath("obsm", ("spatial_canonical",))` and
  `TableComponentPath(
      "uns",
      ("spatial_coordinates", "spatial_canonical"),
  )`, with the cache action and affected region represented by event change
  kind/source/regions;
- one canonical persistence unit that writes both paths through AnnData element
  encodings, preserves unrelated table elements and sibling metadata, and
  restores or preserves the prior on-disk pair after a handled partial failure;
- canonical metadata parsing and structural validation before accepting a disk
  reload or reusing a reopened cache;
- an explicit UI-independent affected-region invalidation operation for
  supported region-key, instance-key, or obs replacement events, including
  same-set instance reassignment that the instance-set digest cannot detect;
- clear mismatch/recalculation data for later controller and widget consumers;
- canonical cache lifecycle and backed-zarr tests.

Direct arbitrary AnnData edits that bypass napari-harpy semantic mutation APIs
remain undetectable. Supported producers must publish linkage changes with
sufficient affected-region context to invalidate every prior and current region
whose row-to-instance association may have changed.

Exit criteria:

- a successful cache update records both canonical paths as dirty in one shared
  table-state event and round-trips through zarr;
- a centroid-only write touches only the two canonical element paths and never
  serializes the complete AnnData table;
- the table is not marked clean when either half of the canonical persistence
  unit fails or rollback cannot substantiate a consistent pair;
- reload/reopen reuses only structurally valid canonical metadata;
- a supported linkage mutation invalidates every affected region before cache
  reuse even when its instance set is unchanged;
- calculating, cancelling, rejecting, or reusing centers without an accepted
  cache mutation records no dirty path and emits no table-state mutation event.

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
- apply-time selected-region signature plus cache/binding/table revalidation
  and rollback;
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
- centroid cache status and explicit Recalculate centroids action;
- dependent filtering with stable identity preservation;
- default spatial_annotation behavior;
- Shapes Annotation dirty-session blocker;
- status cards, tooltips, accessible names, and selector-state tests.

Exit criteria:

- Run is enabled only for a complete valid/rebuild-authorized request;
- first-run cost and cache reuse state are clear before execution;
- Recalculate centroids is available from a valid labels/table selection even
  when Shapes or annotation-target inputs are incomplete;
- selection/inspection never mutates or dirties a table.

### Slice 7: Async calculate-query-review-apply flow

Deliverables:

- worker orchestration with one monotonically increasing operation ID
  spanning an optional centroid-calculation phase and a spatial-query phase;
- standalone Recalculate centroids orchestration that bypasses valid reuse,
  calculates a payload in the worker, and applies it on the main thread, ending
  after the cache update rather than continuing into query or annotation
  review;
- textual busy status, cancellation, cleanup, and error routing;
- main-thread cache update/revalidation;
- no-result outcome;
- Apply dialog with live counts and mandatory overwrite warning;
- main-thread annotation apply and undo UI;
- controller/dialog async tests.

The centroid-calculation phase is:

    worker: calculate_canonical_centers()
        ↓
    main thread: accept and apply cache payload

It is skipped when the selected-region cache is already valid. Standalone
Recalculate centroids performs this phase with valid reuse bypassed and ends
after the main-thread cache update.

The spatial-query phase is:

    worker: evaluate canonical centers against the Shapes geometry
        ↓
    main thread: accept result and open the review dialog

The same operation ID governs both phases; do not introduce a separate
job-ID concept. The controller also records which phase owns the active worker.
Cancellation, selection changes, reload, a newer operation, or widget shutdown
invalidate that operation ID, and every late signal from either phase is ignored.

Exit criteria:

- napari remains responsive during global aggregation;
- cancel/stale/error paths cannot update the cache, open late dialogs, or annotate;
- successful Recalculate centroids applies only the refreshed cache update, marks the
  shared table dirty, and opens no query/result dialog;
- cache update and annotation application are visibly distinct state
  changes;
- users always see affected and overwrite counts before annotation mutation.

### Slice 8: Persistence UX and cross-widget synchronization

Deliverables:

- Spatial Query, Viewer, and Object Classification targeted refresh behavior
  consuming the shared table-state and existing domain events without feedback
  loops;
- reusable Write Table State and Reload Table from zarr UI components backed by
  the generalized PersistenceController;
- dirty reload write/discard/cancel behavior;
- dirty-dataset close/replacement guard;
- multi-widget and backed-zarr integration tests.

Exit criteria:

- all widgets observe one current in-memory table state;
- canonical centers and annotation columns persist/reload together;
- Write Table State persists the union of supported dirty components through
  AnnData element encodings and clears only successfully written unchanged
  component revisions;
- no full-AnnData rewrite occurs;
- no event loop or unrelated classifier invalidation;
- no late task affects a reloaded/replaced table;
- there is no competing widget-local dirty truth.

### Slice 9: Production hardening and release gate

Deliverables:

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
  defect is deferred as polish.

## Definition of Done

The feature is complete when a user can select a valid stored polygon
annotation, 2D labels element, and linked table; transparently create or reuse
validated canonical centers; run a responsive center-containment query; review
affected and overwritten rows; apply a string annotation to a compatible
existing or new obs column; undo the last annotation; and safely write/reload
all supported dirty table components from zarr without rewriting the complete
AnnData object.

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
- cache update is atomic and visibly marks the shared table dirty;
- creating or refreshing spatial_canonical records both its obsm path and
  metadata path as dirty, while calculation without a cache update does not;
- overwrite is never silent;
- no stale/cancelled worker updates the cache, opens a dialog, or mutates a table;
- spatial instance identity always uses region_key and instance_key; obs_names
  do not participate in canonical cache identity;
- table rows with no source label are rejected as binding inconsistencies;
- labels absent from the table are not claimed as queryable or counted;
- annotation apply/rollback and undo are safe and atomic;
- shared cross-widget dirty state and general table events work for obs, obsm,
  and uns;
- canonical cache and annotation columns round-trip through backed zarr;
- persistence uses AnnData element-level encodings for dirty obs/obsm/uns
  components and never rewrites unrelated AnnData elements;
- reload and dirty-dataset replacement protect local changes;
- accessible validation/status/error feedback is complete;
- repository tests, lint, formatting, and manual smoke tests pass.

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
            |                 atomically apply cache update
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
