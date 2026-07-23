# Spatial Query Annotation Using Canonical Centers

## Status

Final product specification and implementation plan.

Canonical metadata/cache lifecycle (Slice 1a) and blocking Harpy centroid
construction/cache ensure (Slice 1b) are implemented. Later slices remain
planned.

This document supersedes the raster-overlap query algorithm described in
spatial_query.md. It retains the agreed user interface, table mutation,
overwrite protection, dirty-state, persistence, reload, asynchronous
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
   child;
2. treat all polygons in that Shapes element as one annotation region;
3. select a labels element in the same SpatialData object and selected
   coordinate system;
4. select a table that annotates that labels element;
5. select an existing compatible AnnData.obs column or configure a new one,
   defaulting to spatial_annotation;
6. run a centroid-containment spatial query;
7. transparently calculate and cache canonical centers first when they are not
   already valid and reusable;
8. review the affected instances and any values that will be overwritten or
   removed;
9. choose Set annotation or Remove annotation; Set defaults the annotation
   value to the Shapes element name;
10. apply the string value or missing annotation state to the matching table
    rows;
11. visualize the selected annotation column on the primary labels layer and
    refresh that visualization after an effective annotation change;
12. explicitly write or reload the shared table state when working with the
    backed zarr store.

This workflow lives in one registered parent `AnnotationWidget`. The parent
composes two children with separate responsibilities:

    AnnotationWidget
        ├── ShapesAnnotation
        │     create, edit, validate, save, and discard polygon annotations
        └── SpatialQuery
              select labels/table/target, query centers, review, apply, and
              visualize table annotations

The parent owns the shared SpatialData, coordinate-system, and selected-Shapes
context. The children remain separate implementation components with their own
controllers and status logic; integrating the workflow must not merge their
domain responsibilities into one large widget class.

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
- **One primary labels presentation:** Spatial Query reuses the primary labels
  layer and the shared table-backed coloring infrastructure instead of creating
  a competing styled overlay or a separate palette system.
- **Deterministic and testable:** identical geometry, transforms, canonical
  centers, linkage metadata, and table state produce identical results.
- **Accessible:** all important states, warnings, busy states, and errors are
  conveyed textually and are keyboard accessible.

## Terminology and Normative Decisions

### Annotation

An annotation is one selected SpatialData Shapes element. Every geometry row in
the element contributes to the same annotation region. This workflow does not
offer per-polygon row selection.

The Shapes element must satisfy the existing Shapes Annotation child's
edit-validity contract:

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

Spatial Query must reuse the complete existing Shapes Annotation edit-validity
contract and its implementation rather than maintaining a second validator.
The selected Shapes element must be accepted by
`validate_existing_shapes_source_geodataframe()` and therefore be eligible for
editing by the Shapes Annotation child, including its active-geometry,
GeoDataFrame-index, and Polygon requirements. A Shapes element rejected for
editing must also be rejected by Spatial Query. If the shared edit-validity
contract is extended in the future, both workflows inherit that change through
the shared validator.

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

If the Shapes Annotation child has an open dirty edit session for the selected
Shapes element, the parent must not let the Spatial Query child silently query
the last saved geometry. Run is blocked with guidance to save or discard the
shape edits. Because both children belong to the same parent, this is a direct
parent-to-child context contract rather than cross-widget dirty-session state.
After a successful Shapes Annotation save, the parent refreshes the Spatial
Query child and the new in-memory geometry becomes queryable.

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

This equation defines the coordinate relationship; napari-harpy must not
implement the inversion or composition itself. Resolve the transformation
through SpatialData's transformation graph:

    shapes_to_labels = get_transformation_between_coordinate_systems(
        sdata,
        source_coordinate_system=shapes_element,
        target_coordinate_system=labels_element,
        intermediate_coordinate_systems=coordinate_system,
    )

Passing the SpatialElements selects their intrinsic coordinate systems, while
`intermediate_coordinate_systems=coordinate_system` requires the path to pass
through the coordinate system selected by the user. Missing, non-invertible,
or ambiguous paths are rejected by SpatialData rather than replaced with local
path-discovery or matrix-composition logic.

Convert the returned `BaseTransformation` explicitly with
`to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y"))`. The unioned
annotation is transformed with that matrix before point membership is
evaluated. Napari's array-axis y, x order must not leak into this calculation.
A small query adapter may verify the resulting 3 x 3 matrix shape and finite
values and convert it to the representation expected by Shapely; it must not
reimplement transformation-graph traversal, inversion, or composition. The
conversion must be covered by identity, translation, anisotropic-scale,
rotation, and reflection tests so matrix and axis conventions are not inferred
from display behavior.

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
- removal of annotations from matching rows in one existing compatible column;
- primary-label visualization of the selected annotation column with the shared
  categorical palette and missing-value styling;
- creation, extension, and persistence of the standard
  `AnnData.uns["<column>_colors"]` palette associated with an effectively
  mutated spatial-annotation column;
- mandatory overwrite/removal disclosure and confirmation;
- asynchronous center calculation and query execution, cancellation, textual
  busy status, and stale-result protection;
- per-table shared clean/dirty state;
- selective AnnData element-level persistence of all supported dirty table
  components, including canonical centers and their metadata, without rewriting
  the complete AnnData table;
- selective reload of supported table components from zarr, plus a full-table
  convenience reload, with dirty-state protection;
- refresh of the Annotation children and other widgets after a cache update,
  annotation mutation, write, reload, or Shapes element write.

### Non-goals

- modifying labels pixels, or modifying Shapes geometry from the Spatial Query
  child; Shapes geometry editing remains owned by the sibling Shapes Annotation
  child;
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
- creating or changing `<column>_colors` merely because a column was selected
  for viewer display, without an effective annotation mutation;
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
between existing rows. No operation currently planned in this roadmap changes
the row-to-region or row-to-instance association. An out-of-band same-set
reassignment is therefore an explicit structural-validation limitation, like
an out-of-band labels-pixel edit that preserves the labels structural
signature. The user-facing workflow does not attempt to recover from
unsupported out-of-band mutations.

Any future supported producer that changes row-to-region or row-to-instance
linkage must define its cache lifecycle as part of that feature. It must either
invalidate every previous and current affected region before reuse, or eagerly
force recalculation of the complete affected set. The invalidation API should
be introduced with that concrete producer rather than maintained speculatively.

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
forced-recalculation mode remains available to core callers and future
producer-specific recovery tooling, but it is not exposed as a routine widget
action. A future supported labels-pixel editor must publish an explicit cache
invalidation or eager-recalculation contract.

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

## User Experience

### Unified Annotation dock widget

Extend the existing registered Annotation dock into one parent
`AnnotationWidget`; do not register a second Spatial Query dock. The parent
composes a Shapes Annotation child and a Spatial Query child. Their visual
language, status cards, spacing, and validation feedback follow the existing
Harpy widget patterns.

The parent owns the coordinate-system and Shapes-target selectors and the
committed annotation context shared with both children. Those selectors remain
visible as common workflow inputs rather than belonging to either child.

The Shapes Annotation child owns polygon creation, editing, hole creation,
validation, save controls, dirty detection, and discard-confirmation behavior.
Before the parent commits another coordinate system or Shapes target, it asks
the Shapes child to resolve its current edit session. Cancellation restores the
old selector and context; acceptance lets the child perform its own layer
cleanup before the parent commits and publishes the new context. The parent
never manipulates the child's private edit-session layer directly.

The Spatial Query child consumes the parent's committed annotation context and
owns this dependent control order:

1. Labels element combo.
2. Linked table combo.
3. Target column mode: Existing column or New column.
4. Existing-column combo or new-column line edit.
5. One Spatial Query status card combining selection readiness and centroid
   cache behavior.
6. Run Spatial Query button.
7. Write Table State and Reload Table from zarr buttons.
8. Persistent shared clean/dirty table-state status.

The parent coordinates child context changes and cancellation, but each child
retains its own controller and status-building modules. Spatial Query logic must
not be added directly to the already substantial Shapes Annotation child.
Target controls use stable identities rather than display strings so valid
selections survive refreshes.

### Selection dependencies

- Coordinate-system choices come from the shared HarpyAppState SpatialData and
  are selected once on the parent Annotation widget.
- The parent's Shapes-target choice includes Shapes elements available in the
  selected coordinate system and valid under the Shapes Annotation edit
  contract, plus the existing create-new workflow rendered by the Shapes child.
- Spatial Query receives only a saved existing Shapes element. A create-new
  session becomes queryable after its first successful save, and any dirty edit
  session blocks Run until it is saved or discarded.
- Labels choices include only supported 2D labels elements available in that
  coordinate system.
- Table choices include only tables whose SpatialData TableModel metadata
  declares the selected labels element as an annotated region.
- Existing-column choices include only pandas categorical annotation columns
  whose categories are all strings, and exclude region_key and instance_key.
- Changing an upstream selection refreshes downstream options, closes pending
  dialogs, and cancels or invalidates active work.
- Preserve a downstream selection if its stable identity remains valid, except
  across an accepted coordinate-system change. That transition deliberately
  clears the Spatial Query labels selection and all dependent state even when
  the same labels element is available in both coordinate systems. The labels
  selector then shows `Choose a labels element`; it never silently selects or
  reloads an element. Removing the selected primary labels layer from napari
  has the same clearing behavior. Other dependent controls follow their
  documented defaults or show a disabled placeholder with an explanation.

The parent's coordinate-system combo participates in the shared
active-coordinate-system model. The parent commits an in-widget change to
HarpyAppState only after the Shapes child accepts the edit-session transition.
The resulting committed context refreshes both children, invalidates active
Spatial Query work, and follows the same layer cleanup behavior as other
coordinate-aware widgets.

### Target column behavior

If spatial_annotation already exists and is a compatible categorical target,
default to Existing column and select it. If it is absent, default to New
column with spatial_annotation prefilled. If it exists but is incompatible,
exclude it from the Existing-column choices and explain that Spatial Query will
neither convert nor overwrite it; the user must select another eligible
categorical column or enter a different non-colliding New column name.

A new column name:

- is trimmed;
- must pass SpatialData dataframe-column-name validation;
- must not equal region_key or instance_key;
- must not collide with an existing obs column;
- is not created until the user confirms Apply with at least one changed row.

An existing target column is compatible only when it uses
`pd.CategoricalDtype` and all declared categories are strings. An empty
categorical category set is valid. Missing row values are valid and mean that
those rows are not annotated. Preserve the categorical `ordered` flag and
category order.

StringDtype, object/string, numeric, boolean, datetime, mixed-object, and
non-string categorical columns are not writable and must never be converted
implicitly. Spatial Query supports convenience creation through New column,
but it does not normalize an existing user-owned column into its required
categorical contract.

Remove annotation is available only for an existing compatible target column.
It clears the matching row values to the column's missing state; it never
deletes the obs column. New column plus Remove annotation is invalid because it
would create no useful state.

### Annotation visualization

The selected labels element is loaded and activated as the primary labels
layer through `ViewerAdapter.ensure_labels_loaded()`, following the same layer
selection and activation behavior as Object Classification. The Spatial Query
child must reuse that primary layer; it must not create a second styled-labels
overlay for the annotation. Activating the labels layer switches napari's
active editing target away from the Shapes layer without discarding the sibling
Shapes Annotation session.

The selected annotation target column is also the Spatial Query child's color
source:

- explicitly selecting a labels element or compatible existing target column
  applies the current column values to the primary labels layer immediately;
- programmatic context, selector, or table refreshes update the available
  controls without silently reclaiming primary-layer styling from another
  workflow; only an explicit user coloring action establishes Spatial Query as
  the most recent color-source owner;
- a New column cannot be used as a color source before it exists; after the
  first effective Apply creates it, the new column remains selected and is
  applied as the color source;
- Set and Remove operations refresh the current visualization after the table
  mutation succeeds; removed or otherwise missing values use the shared
  semi-transparent missing/unlabelled color;
- a no-op Apply does not emit a table mutation or perform a mutation-driven
  refresh.

Use the generic table-backed labels styling path through
`apply_table_color_source_to_labels_layer()` and a `TableColorSourceSpec` for
the selected obs column. Do not reuse Object Classification's specialized
`ViewerStylingController`: it owns `user_class`, `pred_class`, prediction
confidence, integer class IDs, and classifier palette semantics that do not
belong to arbitrary spatial-annotation strings.

The annotation column and its standard `table.uns["<column>_colors"]` palette
form one annotation consistency unit. Palette entries align positionally with
the categorical column's category order. Resolve that palette as follows:

- a valid stored palette has exactly one valid color per category and remains
  authoritative; preserve every existing color value and its order;
- when no stored palette exists, derive one viewer-only from category positions
  using `default_labeled_class_color(position + 1)`;
- when stored palette metadata is malformed or incompatible with the category
  count, render with the same complete position-based default palette and show
  a non-blocking warning rather than trusting ambiguous color alignment;
- when Set appends a category to a valid stored palette, append exactly one
  `default_labeled_class_color(new_position + 1)` entry and never regenerate
  earlier entries;
- when a New column is created, create its complete position-based default
  palette at the same time.

This per-position mapping uses the same underlying default color logic as
Object Classification while guaranteeing that appending a category never
changes colors assigned to existing categories. Merely selecting or displaying
a column remains read-only: a missing or invalid palette is not written,
repaired, announced as a table mutation, or marked dirty until an effective
annotation Apply occurs. On such an Apply, store the derived palette when it was
missing, replace invalid palette metadata with the displayed complete default
palette, or append the new-category entry to a valid palette. Remove never
prunes categories or colors.

An effective annotation Apply publishes the ordinary
`TableStateChangedEvent` for the changed `obs/<column>` path. The Spatial Query
child refreshes table-backed coloring only when the event refers to the current
SpatialData object and table, covers the currently displayed annotation
column, and either includes the selected labels region or is explicitly
table-wide. Unrelated obs, obsm, uns, table, or region changes must not repaint
the layer. A relevant reload or a mutation published by another widget follows
the same targeted refresh rule.

Because Object Classification and the Spatial Query child can both style the
same primary labels layer, the most recent explicit user coloring action owns
its presentation. Selecting a Spatial Query target column or successfully
applying that annotation is such an action. A later Object Classification color
choice may replace it, and vice versa. Shared presentation state must retain
the currently applied primary-layer color-source identity so background table
events refresh that source only; a widget must not reclaim layer styling merely
because it received an unrelated refresh callback.

### Spatial Query status

One status card reports both the highest-priority selection/readiness state and
the centroid-cache behavior Run will use. For a complete request, it reports
one of:

- Ready: valid cached centroids will be reused;
- Not calculated: Run will calculate centers from scale0 first;
- Partial: Run will calculate centers for this labels region;
- Stale: Run will refresh centers for this labels region;
- Invalid: Run will report the mismatch and rebuild centroid data;
- Running: calculating centroids;
- Running: querying centroids.

The tooltip explains in user-facing terms that centers will first be
calculated for the selected labels element before the spatial query runs. The
UI does not expose a manual Recalculate action. Run owns the complete automatic
cache lifecycle: it reuses a valid report and calculates, refreshes, or rebuilds
centers for absent, partial, stale, or invalid states before querying.

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
- explicit Set annotation and Remove annotation modes, with Set selected by
  default;
- a QLineEdit labeled Annotation value, prefilled with the Shapes element name;
- a live mode-specific summary;
- a mode-specific primary action and Cancel.

It does not show labels missing from the table, because the centroid-based query
does not enumerate labels outside the table.

In Set annotation mode, the annotation value is trimmed and must be a non-empty
string. Unicode and internal spaces are allowed. It is a user-facing category
value rather than a SpatialData element key, so element-name restrictions do
not apply. The strings `"None"` and `"nan"` remain ordinary annotation values;
an empty string remains invalid and none of them acts as a removal sentinel.

Changing the value updates the summary live. It shows currently missing,
already equal, and different non-missing values. If different non-missing
values will be replaced, the dialog shows a prominent mandatory warning and
uses explicit action text such as Overwrite 12 and apply to 35.

In Remove annotation mode, the annotation-value field is disabled or hidden.
The summary shows Already empty and Annotations to remove. The dialog shows a
prominent removal warning and uses explicit primary action text such as Remove
annotation from 35. Remove annotation is disabled for New column targets.

Cancel closes the dialog without creating or changing the annotation column.
It does not roll back a canonical-center cache updated earlier in the Run flow.

Immediately before Apply, revalidate:

- SpatialData, Shapes, labels, coordinate system, table, and target intent;
- table linkage and row identities;
- the Shapes geometry and element-to-element transformation against the query
  snapshot retained by the controller;
- the canonical source, binding, selected-region metadata, and center rows
  against the exact `CanonicalCentersResult` used by the query;
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
- when setting, add a category without discarding existing category order or
  ordered state;
- when removing, assign the canonical in-memory missing scalar `pd.NA` and do
  not remove unused categories or reorder categorical metadata;
- preserve a valid existing `<column>_colors` palette and append one stable
  default color when Set adds a category;
- create or repair `<column>_colors` from the stable position-based default
  palette only as part of an effective annotation Apply;
- represent unannotated values as actual missing values, never stringified
  missing values or an empty string;
- preserve the compatible target dtype;
- update each matching row at most once.

A new target column is categorical. Non-target rows are missing, the applied
string value is its first category, and `<column>_colors` contains its first
stable default color. Removal never creates or deletes a column, category, or
palette entry.

If every matching row already has the requested string value, or every matching
row is already missing during removal, report a no-op. Do not replace the
column object, emit an annotation mutation event, or alter dirty state. Any
dirty state caused earlier by cache creation remains.

After an effective annotation mutation:

- mark the selected table dirty through shared HarpyAppState;
- emit a semantic table-state event with the changed obs column and, only when
  it was created, repaired, or extended, the associated palette uns path;
  include the selected labels region, source, and change kind; a new column
  uses `created`, while an existing-column Set or Remove uses `updated`;
- refresh this widget and table-column/color-source consumers;
- show updated, overwritten, and unchanged counts.

Spatial Query deliberately provides no operation-specific Undo command. The
review dialog, mandatory overwrite/removal confirmation, and atomic Apply are
the primary safeguards. An incorrect annotation can be corrected by another
Set annotation operation or cleared through Remove annotation. Unpersisted
table changes can also be discarded through the existing Reload Table from
zarr workflow, with its normal warning that all covered dirty table components
are replaced. This keeps annotation state, dirty tracking, and recovery within
the shared persistence model rather than introducing widget-local history.

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

The containment stage consumes `CanonicalCentersResult` as its only source of
canonical centers and row-to-instance identity. Cache reuse and fresh
calculation converge before containment begins. This stage never inspects,
calculates, applies, or persists the canonical cache and never reads labels
raster data.

The query returns an immutable result carrying the selected-region binding and
the matching instance IDs in ascending order. Controller operation identity
remains orchestration state. The concrete request, result, and thread-boundary
contracts are specified in Slice 4.

### Query algorithm

At a feature-contract level, the query validates and snapshots the selected
Shapes polygons, transforms their union into the selected labels element's
intrinsic x, y frame, applies a bounding-box prefilter followed by the
authoritative vectorized `shapely.intersects_xy()` predicate, and maps matches
back to the binding's instance IDs. Slice 4 specifies the exact execution
boundary and algorithm.

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
- explicit new/existing column intent;
- the canonical-center provenance needed for apply-time validation.

A separate pure summarization step classifies the current matched values for a
candidate string assignment or missing-value removal. Shared table-component
mutation tokens remain widget/app-state orchestration data and are not embedded
in the UI-independent domain preparation.

Apply validates preparation against current state before mutating. If validation
or assignment fails, restore the entire prior target-column and companion
palette state and leave dirty state/events unchanged. Partial row updates and
half-applied obs/uns annotation units are forbidden.

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
    TableComponentPath("uns", ("spatial_annotation_colors",))
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
        regions            # explicit semantic scope of row-scoped changes
        change_kind        # created, updated, removed, rebuilt, reloaded
        source             # spatial_query, spatial_query_canonical, ...

`regions` is required on every event. It contains the table regions
semantically targeted by any row-scoped change. `regions=()` means the event is
genuinely metadata-only and never means that the affected regions are unknown.
A producer that cannot prove a narrower row scope reports every region declared
by the table. For a mixed row-data/companion-metadata event, `regions` describes
the row-scoped part.

The event describes what changed; emitting it does not by itself decide whether
the paths are dirty. HarpyAppState exposes three explicit acceptance paths:

    record_table_mutation(event)
        # emit the event and assign one fresh mutation token to every path

    record_persisted_table_change(event, snapshot)
        # emit the event and establish only unchanged captured path tokens
        # as already persisted

    record_table_reload(event)
        # emit the event and clear only dirty paths covered by the accepted
        # component reload

Object Classification uses `record_table_mutation()` because it changes the
in-memory table and relies on PersistenceController for the later write. A user
annotation reports only its selected labels region. Classifier inference
reports the resolved prediction regions, which may contain every table region
even when one segmentation is selected in the widget. A classifier
metadata-only update reports `regions=()`. Harpy Feature Extraction writes its
feature matrix and metadata directly when
SpatialData is backed and therefore uses `record_persisted_table_change()`
after Harpy returns successfully. It captures the table's dirty snapshot on the
main thread immediately before launching Harpy and returns that immutable
snapshot to the main-thread acceptance boundary with the worker result. A path
is cleared only when its captured mutation token is still current; a same-path
mutation accepted while Harpy was working remains dirty. A path absent from the
pre-operation snapshot is not allowed to clear a newer dirty mutation.
Unbacked Feature Extraction uses `record_table_mutation()`. Reload uses
`record_table_reload()`. Producers do not invoke widgets directly.

`TableStateChangedEvent` is the single shared contract for AnnData table
changes. HarpyAppState exposes one `table_state_changed` signal, and each of the
three explicit acceptance methods emits exactly one event through that signal
while applying its corresponding dirty-state transition. Producers do not emit
a second feature-specific table event.

`FeatureMatrixWrittenEvent` and `ClassificationTableWrittenEvent` are removed
after their existing consumers have migrated to `TableStateChangedEvent`. They
do not contain information that is absent from the general event: a feature key
is the key of its changed obsm path, classification columns are the keys of its
changed obs paths, and change kind, source, SpatialData, and table identity are
already explicit. Object Classification can therefore detect an overwritten
selected feature matrix from the event's source, change kind, and obsm path;
Viewer refreshes can filter the same event by its paths. This avoids duplicate
signals and duplicate representations of one accepted table change.

Reload events also carry explicit, non-empty component paths; a `None` or empty
path set is not used as a full-table sentinel. For an obs reload, the event
reports the complete old/new obs-column coverage because obs is replaced as one
encoded dataframe. An uns parent path covers its complete subtree. Named obsm
entries and nested uns entries otherwise remain exact paths. HarpyAppState
clears only dirty paths covered by those accepted reload paths, so unrelated
dirty components remain dirty.

`ShapesElementWrittenEvent` remains separate because it describes a
`SpatialData.shapes` change rather than an AnnData table component. Only the
explicit HarpyAppState table acceptance method updates or clears the shared
dirty-component manifest.

Shared app state maintains, for each in-memory table, a mapping from dirty
component path to its latest opaque mutation token. The user-facing table dirty
boolean is derived from whether that mapping is non-empty. Every accepted
mutation creates one fresh identity token and assigns it to all paths in that
event. A write captures the current `path -> token` mapping and clears a path
only when that exact token was successfully persisted and is still current.
A newer mutation of the same path therefore remains dirty. Ordering is neither
encoded nor needed, and no mutation or persistence counters are maintained.
These session-only tokens are not stored in AnnData.

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
`TableComponentPath -> mutation token` mapping. `TableComponentWriteResult`
reports the table store and exact logical paths that were successfully
persisted. The Qt-independent writer receives paths, not HarpyAppState or
mutation-token state; HarpyAppState alone decides whether the captured tokens
are still current and may be cleared. Unscoped `mark_table_dirty()` and
`clear_table_dirty()` operations are therefore not primary mutation or
persistence APIs.

The deferred-write flow is:

    widget/controller accepts a table mutation
        ↓
    record_table_mutation(TableStateChangedEvent)
        ↓
    HarpyAppState records path mutation tokens
        ↓
    PersistenceController captures a dirty snapshot
        ↓
    core persistence writes supported AnnData elements
        ↓
    HarpyAppState clears only unchanged persisted tokens

An operation that already persisted its result uses the shorter flow:

    main thread captures the current dirty snapshot
        ↓
    producer completes its element writes and metadata consolidation
        ↓
    main thread accepts the result
        ↓
    record_persisted_table_change(TableStateChangedEvent, snapshot)
        ↓
    HarpyAppState emits the event and clears only event paths whose captured
    mutation tokens are still current

The general event, component manifest, and persistence foundation are
introduced before canonical-cache integration. Full Viewer, Object
Classification, and parent Annotation widget refresh behavior, including its
Spatial Query child, feedback-loop guards, and shared persistence UI are
completed in the later cross-widget integration slice.

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
| Query returns no matching centers after cache update | No further mutation | Remains dirty |
| Cancel Apply after cache update | No further mutation | Remains dirty |
| Apply annotation is a no-op | No | Unchanged from current state |
| Apply sets/removes annotations | obs, plus uns when the companion palette changes | Dirty |
| Successful write, no remaining/newer dirty components | Captured components persisted | Clean |
| Successful write with remaining/newer dirty components | Captured components persisted | Remains dirty |
| Failed write | No accepted persistence completion | Remains dirty |
| Successful component reload, no other dirty paths | Selected components replaced from disk | Clean |
| Successful component reload with unrelated dirty paths | Selected components replaced from disk | Remains dirty |
| Successful full-table reload | All supported table components replaced from disk | Clean |
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
mutation token is acknowledged; all captured paths remain dirty and retryable.

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
`record_table_mutation()`. Both cases publish their one
`TableStateChangedEvent` through `table_state_changed`.

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

The spatial annotation target and companion palette, for example
`adata.obs["spatial_annotation"]` and
`adata.uns["spatial_annotation_colors"]`, form one annotation consistency unit.
If the obs column is dirty, the existing AnnData element-level persistence path
writes the obs dataframe element with its index, column-order, missing-value,
and categorical encodings. The palette uns element is additionally written only
when its own path is dirty because it was created, repaired, or extended.
Writing these remains a selective table-component operation; it does not
serialize the full AnnData object. Existing supported classifier metadata and
future components are included only when their paths are present in the same
shared dirty snapshot.

There is one shared Write Table State action rather than competing widget-local
writers. A centroid-only dirty table writes only the spatial_canonical pair; an
annotation-only dirty table writes obs and its palette when that palette also
changed; a table containing both changes writes their union in the same
operation.

The matrix and its spatial-coordinate metadata are one persistence unit. The
service stages/backs up as needed, validates both encoded elements, and must not
leave a newly written matrix paired with old or missing metadata after a handled
failure. The table remains dirty unless every component in the captured write
set succeeds. Captured component mutation tokens are cleared only if they have
not changed during the write; later mutations remain dirty.

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

The Qt-independent reload operation accepts explicit
`TableComponentPath` values and returns a `TableComponentReloadResult` with the
resolved table path and exact logical path coverage restored from disk. Reload
uses these physical AnnData units:

- requesting any obs-column path reads, validates, and replaces the complete
  encoded obs dataframe; the result covers the union of old and reloaded obs
  columns, including locally created columns removed by the reload;
- an obsm path reads and replaces only its named entry, or removes that
  in-memory entry when it is absent on disk;
- an uns path reads and replaces only its named top-level or nested entry, or
  removes that in-memory entry when it is absent on disk;
- an uns parent path explicitly covers and replaces its complete subtree.

Selective uns reload is required. Replacing all of uns merely to restore one
obsm companion would risk discarding unrelated classifier, feature, or
canonical metadata changes. AnnData's element API supports reading named obsm
entries and top-level or nested uns entries directly, so a complete uns
replacement is not the default.

Logical paths that form one consistency contract are expanded before reload and
accepted together. In particular:

    feature matrix unit
        obsm[feature_key]
        uns["feature_matrices"][feature_key]

    canonical centers unit
        obsm["spatial_canonical"]
        uns["spatial_coordinates"]["spatial_canonical"]

The persistence foundation supports grouped paths without embedding feature
semantics in the generic reader. Feature Extraction supplies its pair; the
canonical integration slice supplies the canonical pair. A caller must pass the
complete domain-defined consistency unit rather than only its obsm half.

`Reload Table State` remains a convenience operation. It expands to complete
obs coverage, every in-memory, on-disk, or currently dirty obsm key, and every
in-memory, on-disk, or currently dirty top-level uns path, then uses the same
component reload operation. Including currently dirty paths ensures that a
locally or externally removed entry is still explicitly covered. Selective
workflows request only the consistency units they actually need. No special
`None` path or table-wide wildcard is introduced.

Before applying a reload, validate table identity, row count and order, region
key, instance key, obsm leading dimensions, and any affected feature-specific
metadata. In-memory replacement and `record_table_reload()` occur on the main
thread. The emitted `TableStateChangedEvent` contains the reload result's
explicit paths. A reload containing an obs or obsm path reports every current
table region because the replaced AnnData element is row-aligned and the
generic persistence layer cannot substantiate a narrower semantic scope. A
known uns-only metadata reload reports `regions=()`. HarpyAppState clears only
dirty paths covered by those paths. Unrelated dirty paths remain present.
Before a full-table reload, invalidate active center calculations, queries, and
Apply dialogs.
Canonical metadata is parsed and validated before the later canonical
integration accepts a reloaded canonical consistency unit.

After success:

- re-inspect spatial_canonical and its metadata;
- refresh table metadata and compatible target columns;
- preserve a target selection only if still valid;
- notify consumers through `table_state_changed` with the paths actually
  restored;
- clear only dirty paths covered by the accepted reload; a full-table reload is
  clean because it covers every supported component;
- show the source path and outcome.

Late worker results created before reload must never update a cache, open a
dialog, or mutate the reloaded table.

### Leaving a dirty dataset

Changing table selection may leave each table's shared dirty marker intact.
Before replacing or closing a SpatialData object with dirty tables, the shared
application lifecycle warns and offers write/discard/cancel behavior. A warning
owned only by the Spatial Query child is insufficient because other widgets can
also replace the dataset.

## Validation and Error States

The Spatial Query shell keeps Run disabled, with concise status and a detailed
tooltip, when:

- no SpatialData is loaded;
- no coordinate system is selected;
- no saved eligible Shapes element is selected;
- the selected Shapes has unsaved Shapes Annotation edits;
- no eligible 2D labels element is selected;
- labels has no readable 2D scale0 array;
- no linked table is selected;
- table linkage metadata is missing/inconsistent;
- selected-region instance IDs are missing, duplicated, non-positive, or not
  integer-like;
- no valid target mode/column is configured;
- a new column name is empty, invalid, reserved, or colliding;
- an existing target column has an incompatible dtype.

After Run is requested, the execution flow performs one synchronous preflight
before starting any worker. It rejects the intent when the Shapes geometry is
invalid, empty, unsupported, or cannot be unioned; Shapes or labels is no
longer available in the selected coordinate system; or the required transform
is missing, non-finite, unsupported, or non-invertible. Once Slice 7 connects
execution, an active calculation or query also disables Run until that
operation finishes or is cancelled.

An in-memory SpatialData object is a valid calculation, query, and annotation
target. Backing by zarr is required only for persistence actions: Write Table
State and Reload Table from zarr remain unavailable when the selected object
is not backed.

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
            canonical_models.py
            canonical.py
            centroids.py
            query_models.py
            query.py
            annotation.py

            __init__.py
                intentional, stable public exports for the feature domain

            canonical_models.py
                cache-state and metadata value types
                canonical-center request/result types
                canonical cache enums and literals

            query_models.py
                immutable containment-query request/result types

            canonical.py
                spatial_canonical metadata schema/parser/validator
                cache-state inspection and coverage/source fingerprints
                atomic cache updates and rollback support

            centroids.py
                RasterAggregator adapter
                canonical-center calculation and cache ensure

            query.py
                validated Shapes and transformation snapshot construction
                vectorized canonical-center containment

            engine.py
                Shapes validation and union
                coordinate transformation
                vectorized center-containment query

            annotation.py
                target-column validation
                row resolution and conflict summaries
                atomic annotation apply and rollback

The package's consumers import its intentional API from
napari_harpy.core.spatial_query rather than importing implementation modules
directly. The __init__.py facade must remain small and explicit; it must not use
wildcard exports. This lets internal modules be reorganized later without
changing controllers or other consumers.

The corresponding widget composition is:

    widgets/
        annotation/
            models.py
                shared UI-only ShapesAnnotationTarget and AnnotationContext

            widget.py
                registered parent AnnotationWidget
                shared coordinate-system and Shapes-target selectors
                committed AnnotationContext
                child composition and cross-child cancellation/refresh

        shapes_annotation/
            widget.py
                embedded ShapesAnnotation child
                polygon create/edit/save/discard session
                dirty-session preflight, cleanup, and context adoption

        spatial_query/
            __init__.py
            widget.py
            controller.py
            viewer_styling.py
            dialogs.py
            status_card.py

            widget.py
                embedded SpatialQuery child
                labels/table/target selectors, cache status, busy state,
                and persistence actions

            controller.py
                binding/cache validation
                worker lifecycle and operation IDs
                stale-result handling
                cache update and annotation apply orchestration

            viewer_styling.py
                thin primary-layer binding and generic annotation-column
                styling orchestration; no classifier-specific class semantics

            dialogs.py
                cache mismatch reporting
                Apply Spatial Annotation dialog

            status_card.py
                pure status-card specification builders

The exact extraction of the current `ShapesAnnotation` root widget into an
embedded child is an implementation refactor, not a domain merge. The existing
Annotation plugin command is retained and points to the new parent; no second
Spatial Query command or dock contribution is added.

The core spatial_query package must remain UI independent. It must not import
Qt, napari widgets, HarpyAppState, or widget controllers. The Spatial Query
child and its controller orchestrate the pure core operations with services
supplied by the parent Annotation widget and shared application state.

General concerns stay outside the feature package:

- shared table persistence and reload services;
- shared application state and per-component mutation tokens;
- general cross-widget table events;
- generic SpatialData/table-binding helpers;
- the Shapes Annotation child's shared geometry-validity contract.

These dependencies are consumed by Spatial Query; they are not reimplemented
inside it.

Other existing modules in the flat core directory may eventually be moved into
corresponding feature-domain packages. That broader reorganization is a
separate cleanup and is explicitly not part of this feature. New Spatial Query
work adopts the package structure without introducing unrelated import churn.

Reuse rather than copy:

- annotating-table discovery and table/linkage metadata helpers;
- Shapes Annotation child geometry-validity helpers and write events;
- Harpy RasterAggregator;
- HarpyAppState dirty tracking and the shared `table_state_changed` event;
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
- compatible string-categorical updates;
- category addition preserves order/ordered state;
- valid stored palettes are preserved and extended by one default color without
  changing existing entries;
- missing or invalid palettes are displayed through stable position-based
  defaults without mutation, then created or repaired only with an effective
  annotation Apply;
- adding a category never changes an existing category's resolved color;
- StringDtype, object/string, numeric, boolean, datetime, mixed-object,
  non-string categorical, and reserved target columns rejected;
- set summaries update missing/equal/overwrite counts with the proposed value;
- removal summaries report already-empty and annotations-to-remove counts;
- removal writes missing values for categorical columns while preserving dtype
  and categorical metadata;
- removal never deletes the column, prunes unused categories, or permits a New
  column target;
- typed strings such as `"None"` and `"nan"` remain literal annotation values;
- no-op leaves column and palette identity/events/current dirty state unchanged;
- apply rollback restores both column and palette and prevents partial
  obs/uns mutation.

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
- Existing-column choices contain only string-categorical columns and exclude
  reserved, StringDtype, object/string, numeric, and non-string categorical
  columns;
- an incompatible existing spatial_annotation column is never converted or
  reused as a New target with the colliding name;
- annotation coloring preserves a valid stored palette and otherwise derives
  the stable shared default palette without mutating table state merely on
  selection;
- Run enablement/tooltips for every blocker;
- dirty Shapes session blocker;
- all centroid cache status states and phase text;
- invalid-cache mismatch reporting and automatic rebuild state;
- result dialog contents and centroid predicate wording;
- live Set/Remove summaries and value validation;
- mandatory overwrite/removal warnings and explicit action text;
- Remove annotation disables the value field and is unavailable for New column;
- no-result flow does not change annotation column;
- cancel after cache update leaves cache dirty state visible;
- apply states and summaries;
- shared dirty indicator across widgets;
- write enabled only for backed dirty table;
- dirty reload write/discard/cancel branches;
- reload re-inspects cache and invalidates pending query/apply state;
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
- when the companion palette changed, the same write persists exactly the
  `<column>_colors` uns element and a reopened table retains the category-color
  association;
- removed annotations round-trip through zarr as missing values for every
  supported categorical target, regardless of the concrete missing scalar
  returned by AnnData on reload;
- a mixed dirty manifest writes obs, any changed companion palette, and the
  canonical pair, and clears dirty state only after all captured component
  mutation tokens succeed;
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
- reloading one obsm entry preserves unrelated obsm entries and reloads any
  registered companion uns path in the same consistency unit;
- reloading one nested uns path preserves unrelated uns siblings and unrelated
  dirty paths;
- requesting any obs path reloads the complete validated obs dataframe;
- a full Reload Table State expands to complete obs and all in-memory, on-disk,
  or currently dirty obsm and top-level uns paths through the same component
  API;
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
- annotation set/removal counts and changed key;
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

The public canonical contracts live in
`core/spatial_query/canonical_models.py`, and the operations that build, parse,
inspect, and apply them live in `core/spatial_query/canonical.py`. Query-only
contracts live in `core/spatial_query/query_models.py`. Stable consumers import
the intentional exports from `napari_harpy.core.spatial_query` rather than the
implementation modules directly.

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
report is not authoritative for a later Run: Run performs its own fresh
selected-region inspection. A future memoization layer is permitted only when
it has a concrete revision identity invalidated by every accepted relevant
table/cache event; do not assume an abstract cache-generation counter that the
application does not maintain. If an annotation result remains open before
Apply, apply-time table/cache revalidation performs another selected-region
digest unless the operation already owns an equivalent authoritative fresh
binding captured in the same uninterrupted main-thread validation turn. No
status, Run, cache-update, or Apply path hashes every table region by default.

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
cache update but remains structurally undetectable when considering an older
cache for reuse. No current roadmap operation performs such a reassignment; a
future producer must introduce invalidation or eager affected-region
recalculation as part of its own contract.

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
  structural case; no current roadmap operation performs it, and any future
  producer must introduce invalidation or eager affected-region recalculation
  before reuse;
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

**Implementation status: Implemented.**

This slice generalizes the existing Object Classification-oriented persistence
infrastructure before Spatial Query depends on it. It introduces no canonical
cache behavior and no second persistence controller.

Deliverables:

- the validated immutable `TableComponentPath` contract for obs columns,
  individual obsm entries, and top-level or nested uns entries;
- `TableStateChangedEvent` carrying SpatialData/table identity, unique component
  paths, required semantic row-scope regions, change kind, and source; an empty
  region tuple means metadata-only and never unknown;
- one HarpyAppState `table_state_changed` signal carrying that event;
- migration of every `FeatureMatrixWrittenEvent` and
  `ClassificationTableWrittenEvent` producer and consumer to the general event,
  followed by removal of those two redundant event classes and signals;
- retention of `ShapesElementWrittenEvent` as the separate contract for
  non-table shapes changes;
- a HarpyAppState per-table dirty manifest mapping each logical path to its
  latest opaque mutation token, with table-level `is_table_dirty()` derived
  from that manifest;
- explicit `record_table_mutation()`, `record_persisted_table_change()`, and
  `record_table_reload()` acceptance methods so publishing an event does not
  implicitly mean that its paths are dirty;
- pre-operation dirty-snapshot capture for already-persisting producers such as
  backed Feature Extraction, so their successful completion cannot clear a
  same-path mutation accepted while their worker was running;
- `TableDirtySnapshot`, `TableComponentWriteResult`, and acknowledgement
  operations implementing the documented snapshot/write/acknowledge boundary
  and clearing only successfully persisted path mutation tokens that are still
  current;
- a generalized `PersistenceController` coordinating selection, dirty
  snapshots, selective and full validated table-state reload, and
  Qt-independent core persistence functions;
- component writers for encoded obs, individual obsm entries, and top-level or
  nested uns entries, including removal, unsupported-path preflight, and zarr
  metadata consolidation before successful acknowledgement;
- a `TableComponentReloadResult` and component reader that reload the complete
  obs dataframe for any obs request, one named obsm entry, or one top-level or
  nested uns entry, including removal when a requested live entry is absent on
  disk;
- grouped reload of caller-expanded consistency units, with Feature Extraction
  supplying its feature-matrix obsm/uns pair and the later canonical integration
  supplying the canonical pair; the generic persistence layer has no global
  feature-specific registry;
- a full Reload Table State convenience operation that expands in-memory,
  on-disk, and currently dirty component paths and delegates to the same
  selective reader;
- migration of Object Classification as the first deferred-write consumer,
  declaring every logical path it changes and publishing one general table
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

- one event accepted through `record_table_mutation()` assigns one fresh
  identity token to all of its paths; publishing an event alone does not alter
  dirty state;
- a table is dirty exactly when its component manifest is non-empty;
- writing an obs-column path writes the complete encoded obs dataframe but no
  unrelated AnnData component;
- writing an obsm or uns path touches only its supported encoded element and
  preserves unrelated siblings;
- reloading any obs path replaces the complete validated obs dataframe, while
  named obsm and uns reloads preserve unrelated entries;
- component reload emits explicit restored paths, clears only covered dirty
  paths, and accepts consistency units already expanded by domain code;
- a full-table reload delegates to the component API and leaves the table clean
  without using an empty, `None`, or wildcard event path;
- unsupported paths fail before any write and are never silently cleared;
- a mutation accepted during persistence remains dirty when its captured older
  mutation token completes;
- a backed Feature Extraction result acknowledges only matching mutation tokens
  from its pre-operation snapshot and cannot clear a path first dirtied after
  that snapshot;
- a write is acknowledged only after zarr metadata consolidation succeeds, and
  a normally reopened SpatialData sees created, updated, and removed elements;
- a consolidation failure leaves every captured path dirty;
- backed Feature Extraction remains clean for the exact paths Harpy already
  persisted, while unbacked Feature Extraction records those paths as dirty;
- Feature Extraction reports its extraction regions, user annotation reports
  its selected region, classifier inference reports its resolved prediction
  regions, and classifier metadata-only changes report an empty region tuple;
- row-aligned reload events report every table region, while known uns-only
  metadata reloads report an empty region tuple;
- Object Classification, Feature Extraction, and Viewer retain their current
  targeted refresh and invalidation behavior through `table_state_changed`;
- one accepted table change emits one table event rather than parallel general
  and feature-specific events;
- no path calls `AnnData.write_zarr()` or introduces a competing widget-local
  dirty truth.

### Slice 3b: Canonical cache state and persistence

**Implementation status: Implemented.**

This slice wires the canonical cache into the shared foundation from Slice 3a.
Core canonical operations remain independent from Qt and HarpyAppState.

Deliverables:

- the existing Spatial Query controller accepted-result callback as the
  canonical mutation boundary: its consumer publishes a
  `TableStateChangedEvent` only after `apply_canonical_cache_update()` succeeds
  on the main thread; synchronous core callers remain responsible for
  publishing the same event when used inside a shared UI session;
- one accepted cache event containing both
  `TableComponentPath("obsm", ("spatial_canonical",))` and
  `TableComponentPath(
      "uns",
      ("spatial_coordinates", "spatial_canonical"),
  )`; the event reports the selected labels region, uses a canonical-cache
  source, and maps create, extend/refresh, and rebuild actions to `created`,
  `updated`, and `rebuilt`, respectively;
- one canonical consistency unit containing those two paths. Persistence writes
  both through AnnData element encodings and preserves unrelated table elements
  and sibling metadata;
- acknowledge-both-or-neither persistence semantics: the shared dirty manifest
  clears both canonical paths only after both element writes and zarr metadata
  consolidation succeed. A failure acknowledges neither path, even if zarr was
  partially changed before the failure;
- no claim of a transactional multi-element zarr rollback. A partial on-disk
  cache remains dirty in the current session and is classified as `INVALID` by
  canonical inspection after reload or reopen, so it is never reused and is
  rebuilt conservatively;
- canonical metadata parsing and structural validation through
  `inspect_canonical_cache()` after reload or reopen and before any cache reuse.
  The generic `PersistenceController` remains unaware of canonical metadata;
- clear mismatch/recalculation data for later controller and widget consumers;
- canonical cache lifecycle and backed-zarr tests.

The implemented public contracts are:

    CANONICAL_CACHE_PATHS
        # obsm/spatial_canonical plus its nested uns metadata path

    record_canonical_cache_update(app_state, sdata, result)
        # accepted controller-result consumer; reuse returns None

The shared-state integration function lives outside the Qt-independent core.

The accepted cache-update flow is:

    worker
        calculate_canonical_centers()
            ↓
    main thread
        reject cancelled or outdated result
        apply_canonical_cache_update()
            ↓ success only
        record one table mutation for both canonical paths

Cache reuse, cancellation, worker failure, or a rejected payload does not emit
a table mutation event and does not change shared dirty state.

The persistence boundary is:

    write both canonical paths
        ↓
    consolidate zarr metadata
        ↓
    success → acknowledge both paths
    failure → acknowledge neither path

The reload and reopen boundary is:

    reload or reopen table state
        ↓
    inspect_canonical_cache()
        ↓
    VALID → reuse
    otherwise → recalculate when canonical centers are next required

An invalid canonical cache does not make an otherwise valid full-table reload
fail. It is accepted as non-reusable stored state and handled by the existing
inspection and conservative-rebuild lifecycle.

Direct arbitrary AnnData edits that preserve the instance set while changing
row-to-region or row-to-instance association remain undetectable. No current
roadmap operation performs such a mutation. Any future supported producer must
introduce explicit invalidation or eager affected-region recalculation as part
of its implementation.

Exit criteria:

- a successful cache update records both canonical paths as dirty in one shared
  table-state event and round-trips through zarr;
- a centroid-only write touches only the two canonical element paths and never
  serializes the complete AnnData table;
- the table is not marked clean when either half of the canonical persistence
  unit or zarr metadata consolidation fails; a later reload/reopen classifies
  any resulting partial disk state as non-reusable;
- reload/reopen reuses only structurally valid canonical metadata;
- same-set linkage reassignment remains explicitly unsupported and
  structurally undetectable; introducing a producer for it requires a new
  invalidation or eager-recalculation contract;
- calculating, cancelling, rejecting, or reusing centers without an accepted
  cache mutation records no dirty path and emits no table-state mutation event.

### Slice 4: Vectorized centroid-containment query

**Implementation status: Implemented.**

#### Public contracts

Cache reuse and fresh calculation converge before the containment query:

    valid cache ------------------+
                                  |
                                  v
                         CanonicalCentersResult
                                  ^
                                  |
    freshly calculated centers ---+
                                  |
                                  v
                    centroid-containment query

Build a self-contained immutable request on the main thread:

    @dataclass(frozen=True)
    class CanonicalCenterQueryRequest:
        canonical_centers: CanonicalCentersResult
        polygons: tuple[Polygon, ...]
        polygons_to_labels_affine: NDArray[np.float64]

        @property
        def table_name(self) -> str:
            return self.canonical_centers.table_name

        @property
        def labels_name(self) -> str:
            return self.canonical_centers.labels_name

Return an immutable domain result that retains the exact canonical-center
snapshot used by the worker:

    @dataclass(frozen=True)
    class CanonicalCenterQueryResult:
        canonical_centers: CanonicalCentersResult
        matched_instance_ids: NDArray[np.integer]

        @property
        def binding(self) -> CanonicalRegionBinding:
            return self.canonical_centers.binding

        @property
        def eligible_instance_count(self) -> int:
            return self.binding.n_obs

        @property
        def matched_instance_count(self) -> int:
            return len(self.matched_instance_ids)

`canonical_centers` is retained by reference rather than copied. It provides
the source signature, selected-region binding, and exact center snapshot that
produced the query membership. `binding`, `table_name`, and `labels_name` are
derived properties rather than repeated fields. `matched_instance_ids`
contains the matching IDs in ascending order, while `binding.instance_ids`
contains every eligible instance evaluated by the query. The binding already
guarantees that IDs are positive and unique, so the query selects and sorts
them without a second uniqueness-normalization pass. This small result-contract
refinement is part of
Slice 5 because annotation apply-time validation is its first consumer.

The public core operations are:

    def build_canonical_center_query_request(
        sdata: SpatialData,
        *,
        shapes_name: str,
        coordinate_system: str,
        canonical_centers: CanonicalCentersResult,
    ) -> CanonicalCenterQueryRequest:
        ...

    def evaluate_canonical_center_query(
        request: CanonicalCenterQueryRequest,
    ) -> CanonicalCenterQueryResult:
        ...

#### Snapshot and worker boundary

`build_canonical_center_query_request()` fetches the selected Shapes element,
applies the shared Shapes Annotation edit-validity validation, snapshots its
polygons, and asks SpatialData for the element-to-element transformation:

    shapes_to_labels = get_transformation_between_coordinate_systems(
        sdata,
        source_coordinate_system=shapes_element,
        target_coordinate_system=labels_element,
        intermediate_coordinate_systems=coordinate_system,
    )

It converts the returned `BaseTransformation` with
`to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y"))` and snapshots
the result as `polygons_to_labels_affine`. The affine is a finite 3 x 3
homogeneous matrix using explicit x, y axes. SpatialData owns graph traversal,
path selection, inversion, and composition. The request stores only the
resulting matrix because the evaluator needs the exact coordinate relationship
captured for this query.

Shapely 2 geometries are immutable, so a tuple of validated Polygon objects is
the geometry snapshot. Do not copy unrelated GeoDataFrame columns or index
values into the request. Polygon union remains worker work because it can be
substantially more expensive than capturing the geometry tuple.

`evaluate_canonical_center_query()` receives only the request. It unions the
polygons, transforms the union into labels-intrinsic coordinates, applies the
bounding-box prefilter and vectorized predicate, and returns the matching
instance IDs. It does not access SpatialData, Qt, napari layers, or mutable
table state.

The thread boundary is:

    main thread
        build_canonical_center_query_request()
            -> validate and snapshot polygons
            -> ask SpatialData for the Shapes-to-labels transformation
            -> convert it to an explicit x/y affine
            -> no mutation
                    |
                    v
    worker
        evaluate_canonical_center_query()
            -> union and transform polygons
            -> bounding-box prefilter
            -> vectorized intersects_xy
            -> return sorted matching instance IDs
            -> no SpatialData access or mutation

The request deliberately does not store `sdata`, `shapes_name`,
`coordinate_system`, duplicate `table_name` or `labels_name`, or carry cache
state, cache action, or an operation ID. Table and labels names derive from
`canonical_centers`; cache handling has already finished; and selection plus
operation identity belongs to the controller.

#### Query algorithm

1. Receive a validated `CanonicalCentersResult` from the upstream cache or
   calculation flow.
2. On the main thread, call `build_canonical_center_query_request()`:
   - validate and snapshot the selected Shapes polygons through the shared
     Shapes Annotation edit-validity contract;
   - ask `get_transformation_between_coordinate_systems()` for the
     Shapes-intrinsic to labels-intrinsic transformation through the selected
     coordinate system;
   - convert and snapshot its affine using explicit x, y axes.
3. In the worker, call `evaluate_canonical_center_query()`:
   - union all polygons into one effective region;
   - transform the union into labels-intrinsic x, y coordinates;
   - prepare the Shapely geometry for repeated predicates;
   - read y from canonical-center column 1 and x from column 2 for the selected
     binding;
   - use the annotation bounds as a cheap vectorized prefilter;
   - evaluate Shapely only for candidate centers:

        candidates = (
            (x >= min_x) & (x <= max_x)
            & (y >= min_y) & (y <= max_y)
        )
        inside = shapely.intersects_xy(
            region_in_labels,
            x[candidates],
            y[candidates],
        )

4. Map true values back to the binding's instance IDs.
5. Sort the matching IDs and return `CanonicalCenterQueryResult`.

No labels raster data is read by this containment algorithm. With a valid
cache, the entire Run path after validation is an eager vectorized point query
over the in-memory table coordinates.

Controller operation IDs remain orchestration state and do not belong in the
domain request or result. Center calculation and geometry querying return
immutable data; cache updates, row preparation, and annotation mutation remain
distinct operations.

#### Deliverables

- shared Shapes validation/snapshot/union;
- `build_canonical_center_query_request()` as the main-thread snapshot and
  transformation boundary;
- `evaluate_canonical_center_query()` as the SpatialData-independent worker
  operation;
- SpatialData-owned Shapes-intrinsic to labels-intrinsic transformation through
  the selected coordinate system;
- bounding-box prefilter and vectorized Shapely intersects_xy predicate;
- selected-region filtering and a sorted ID result derived from the validated
  unique binding;
- one query implementation consuming `CanonicalCentersResult` after valid-cache
  and fresh-calculation paths converge;
- geometry/transform/predicate tests and zero-label-I/O instrumentation for
  cached queries.

#### Exit criteria

- results match an independent point-in-polygon reference;
- boundary, hole, transform, and x/y semantics are proven;
- multi-region coordinate frames never mix;
- valid cached queries read no labels chunks.

### Slice 4b: Stable categorical palette foundation

#### Responsibility boundary

Slice 4b is a small UI-independent infrastructure slice completed before
annotation mutation. It establishes one shared, append-stable categorical
palette contract for current Labels/Shapes color sources, the later Spatial
Query child styling controller, and future categorical color panels. It does
not implement a color panel or Spatial Query child and never mutates `.obs`,
`.uns`, dirty state, or persisted data.

The current generic palette fallback lives in `viewer/_styling.py` and derives
all colors from `default_categorical_colors(len(categories))`. Its palette
family can change as the category count crosses a threshold. Slice 5 must not
import that viewer implementation or duplicate it in the UI-independent
annotation domain.

Introducing append-stable defaults has one intentional viewer compatibility
effect: an existing generic Labels or Shapes visualization with no valid stored
`<column>_colors` palette may receive different default colors after Slice 4b.
This is a one-time, viewer-only migration effect and does not mutate the table.
Valid stored palettes remain authoritative and unchanged, and Object
Classification colors retain their existing specialized contract. After this
transition, adding categories no longer changes colors assigned to existing
category positions.

#### Shared core palette contract

Keep the implementation in the existing `core/class_palette.py`; do not create
a Spatial Query-specific palette module. Refactor the generic default palette
so category position is stable:

    def default_categorical_colors(length: int) -> list[str]:
        return [
            default_labeled_class_color(position + 1)
            for position in range(length)
        ]

Break the current internal call cycle by having
`default_labeled_class_color()` select directly from the private base-palette
resolver rather than calling `default_categorical_colors()`. The color returned
for every existing positive Object Classification class ID must remain
unchanged.

Move the pure standard AnnData categorical-palette inspection from
`viewer/_styling.py` into the core palette API. Its source classification
remains explicit:

    CategoricalPaletteSource = Literal[
        "stored",
        "default_missing",
        "default_invalid",
    ]


    def validate_categorical_palette_source(
        source: str,
    ) -> CategoricalPaletteSource:
        ...


    def resolve_table_categorical_palette(
        *,
        table: AnnData,
        column_name: str,
        categories: Sequence[object],
    ) -> tuple[CategoricalPaletteSource, list[str]]:
        ...


    def extend_categorical_palette(
        palette: Sequence[str],
        *,
        current_categories: Sequence[object],
        next_categories: Sequence[object],
    ) -> list[str]:
        ...

The shared resolver:

- reads `<column>_colors` without mutating the table;
- returns a valid stored palette unchanged when it has exactly one valid color
  per category;
- returns the complete append-stable default palette when stored metadata is
  absent or invalid;
- reports which of the three sources produced the result so viewers can warn
  without changing table state.

`extend_categorical_palette()` requires `next_categories` to start with the
exact complete `current_categories` sequence. Given a valid existing palette
aligned to that current sequence, it preserves every existing color value and
order and appends `default_labeled_class_color(position + 1)` for only the new
trailing positions. It rejects category removal, reordering, replacement,
invalid input colors, and a palette/current-category length mismatch. It
performs no AnnData assignment.

Labels and Shapes categorical styling consume this shared core resolver and
source type. Plain string/object viewer coercion, where still supported outside
Spatial Query, uses the same append-stable default palette. Object
Classification retains its specialized integer-class categories, unlabeled
class, stored `user_class_colors`/`pred_class_colors`, and strict canonical
palette validation; this slice changes none of those semantics.

Persistence requires no preparatory change: `write_table_components()` already
supports explicit top-level uns paths. Creating, repairing, extending, rolling
back, publishing, and persisting `<column>_colors` remain Slice 5 and later
widget responsibilities.

#### Implementation boundary

The production changes are limited to:

- `core/class_palette.py` for the shared type, validation, stable defaults,
  non-mutating resolver, and extension helper;
- `viewer/_styling.py` for removal of the viewer-local palette policy while
  retaining unrelated color conversion and rendering helpers;
- `viewer/labels_styling.py` and `viewer/shapes_styling.py` for consuming the
  core resolver and source type;
- focused `test_class_palette.py`, `test_styling.py`, Labels/Shapes styling,
  and existing Object Classification palette tests affected by those imports
  or defaults.

Do not modify Spatial Query annotation code, component persistence, table-state
events, dirty tracking, or widget orchestration in this slice.

#### Deliverables

- append-stable generic defaults in `core/class_palette.py`;
- core palette source type, validation, non-mutating resolver, and pure
  extension helper;
- Labels and Shapes styling migrated from viewer-local palette resolution to
  the shared core API;
- removal of duplicated viewer-local generic palette policy;
- focused palette and affected viewer-styling tests.

#### Exit criteria

- appending categories across the Matplotlib-cycle, 20, 28, and 102 category
  thresholds never changes any existing position's color;
- valid stored palettes and their order are returned unchanged;
- missing and invalid palettes resolve deterministically without mutating the
  table;
- extending a palette appends only the required stable colors and rejects
  invalid input or any non-prefix category transition;
- Labels and Shapes styling use the same resolver and source classification;
- focused tests acknowledge the intentional one-time default-color change for
  generic unstored palettes while proving that no table mutation occurs;
- Object Classification class-color behavior and stored palette contracts are
  unchanged;
- the slice emits no table event, records no dirty path, and performs no zarr
  write.

### Slice 5: Atomic annotation preparation and apply

**Implementation status: Implemented.**

#### Responsibility boundary

Slice 5 is a UI-independent table-domain slice. It owns exact row resolution,
target-column validation, live conflict summaries, and atomic apply/rollback.
It does not own dialog lifecycle, operation IDs, Shapes selection freshness, or
table-state event publication; those are controller/widget responsibilities
when the domain API is integrated.

Apply-time validation nevertheless proves that the query is still valid for
the current selected table region and canonical cache. It uses concrete source,
binding, metadata, and center snapshots rather than introducing an abstract
cache-generation counter. The later controller additionally rejects a result
when its Shapes selection, coordinate system, or operation identity changed.

Slice 5 consumes the stable core palette resolver and extension helper from
Slice 4b. For an effective annotation mutation, it decides whether the
companion palette must be created, repaired, or extended and applies that
`.uns` change atomically with the `.obs` column change. A no-op never changes
the palette. Slice 5 does not redefine default colors, stored-palette validity,
or extension semantics.

#### Typed contracts

Keep the column mode explicit because New and Existing represent different user
intent even when the table changes between preparation and Apply. The mode and
column name belong directly to the preparation; they do not need a one-use
wrapper dataclass:

    SpatialAnnotationColumnMode = Literal["existing", "new"]


    @dataclass(frozen=True)
    class SpatialAnnotationPreparation:
        query_result: CanonicalCenterQueryResult
        column_name: str
        column_mode: SpatialAnnotationColumnMode
        row_positions: NDArray[np.intp]
        current_values: pd.Series

        @property
        def binding(self) -> CanonicalRegionBinding:
            return self.query_result.binding


    @dataclass(frozen=True)
    class SpatialAnnotationSummary:
        annotation_value: str | None
        matched_count: int
        current_missing_count: int
        current_equal_count: int
        current_other_count: int

        @property
        def is_removal(self) -> bool:
            return self.annotation_value is None

        @property
        def changed_count(self) -> int:
            if self.is_removal:
                return self.current_other_count
            return self.current_missing_count + self.current_other_count

        @property
        def unchanged_count(self) -> int:
            if self.is_removal:
                return self.current_missing_count
            return self.current_equal_count

        @property
        def overwrite_count(self) -> int:
            return 0 if self.is_removal else self.current_other_count

        @property
        def removal_count(self) -> int:
            return self.current_other_count if self.is_removal else 0


    class SpatialAnnotationColumnChangedError(ValueError):
        """The reviewed column values changed and counts must be refreshed."""


    class SpatialAnnotationQueryOutdatedError(ValueError):
        """The binding or canonical-center query provenance is no longer current."""


    @dataclass(frozen=True)
    class SpatialAnnotationApplyResult:
        annotation_changed: bool
        palette_changed: bool

        def __post_init__(self) -> None:
            if self.palette_changed and not self.annotation_changed:
                raise ValueError(
                    "A spatial-annotation palette changes only with an effective annotation mutation."
                )

Arrays stored on these contracts are read-only defensive snapshots. Series are
defensive copies that domain functions never mutate after construction.
Names such as `table_name`, `labels_name`, `region_key`, and `instance_key` are
derived from the binding rather than duplicated. Summary construction validates
non-negative counts, the partition invariant, the action-specific equal count,
and the normalized string-or-None annotation value.

The public core operations are:

    def prepare_spatial_annotation(
        sdata: SpatialData,
        *,
        query_result: CanonicalCenterQueryResult,
        column_name: str,
        column_mode: SpatialAnnotationColumnMode,
    ) -> SpatialAnnotationPreparation:
        ...


    def summarize_spatial_annotation(
        preparation: SpatialAnnotationPreparation,
        annotation_value: str | None,
    ) -> SpatialAnnotationSummary:
        ...


    def apply_spatial_annotation(
        sdata: SpatialData,
        preparation: SpatialAnnotationPreparation,
        expected_summary: SpatialAnnotationSummary,
    ) -> SpatialAnnotationApplyResult:
        ...

Preparation and summarization never mutate SpatialData or AnnData. The dialog
can call `summarize_spatial_annotation()` repeatedly as the user edits the
string value or switches between Set and Remove without resolving table rows
again. `annotation_value=None` is an internal domain value meaning Remove
annotation; the UI never asks the user to type a sentinel.

Apply reports whether the obs column and associated palette actually changed.
The widget uses those booleans to publish only the corresponding shared table
component paths; the domain function still does not publish events or know
about `HarpyAppState`.

#### Exact row resolution

Preparation rebuilds the current selected-region binding from the live table
and requires it to match the query result exactly:

- the same SpatialData table name and labels region;
- the same region_key and instance_key;
- the same row positions and instance IDs in corresponding order;
- the same positive, unique selected-region instance set.

Returned query IDs must be a unique subset of the binding IDs. Resolve them to
row positions through the binding's instance IDs; never use obs_names and never
match instance IDs without also constraining the selected region. Duplicate
instance IDs in other regions remain valid and cannot receive this annotation.

The resolved row positions and current target values are copied into the
preparation. An empty query result does not create a preparation for Apply; the
controller reports the no-result outcome before opening the dialog.

#### Target validation and summaries

`column_mode="new"` requires the normalized column name to remain absent.
`column_mode="existing"` requires it to remain present and compatible. Both
modes reject region_key and instance_key. Compatibility follows the Target
column behavior contract above: the existing Series must use
`pd.CategoricalDtype`, and every declared category must be a string. No
implicit conversion of StringDtype, object/string, numeric, boolean, datetime,
mixed-object, or non-string categorical data is allowed.

A non-None annotation value is trimmed once and must be a non-empty string.
`None` means Remove annotation and is valid only with
`column_mode="existing"`. Summarization partitions every matched row into
neutral current-state counts:

- current_missing: the current value is missing;
- current_equal: the current non-missing value equals the proposed string;
- current_other: the current non-missing value differs from the proposed
  string, or any current non-missing value during removal.

Consequently:

    matched_count == (
        current_missing_count
        + current_equal_count
        + current_other_count
    )

For Set annotation, missing and other rows change, equal rows remain unchanged,
and `overwrite_count == current_other_count`. For Remove annotation,
`current_equal_count == 0`, missing rows remain unchanged, and
`removal_count == current_other_count`.

Changing the proposed string or switching action is a pure O(number of matched
rows) summary calculation. The UI requires explicit overwrite confirmation
when `overwrite_count > 0` and explicit removal confirmation when
`removal_count > 0`.

#### Apply-time freshness and atomic mutation

Immediately before mutation, `apply_spatial_annotation()` performs fresh
table/cache inspection without reading labels pixels and requires:

1. the current selected-region binding to match the preparation exactly,
   including row-position-to-instance-ID association;
2. the canonical cache to inspect as VALID for that region;
3. the current labels source signature and selected-region canonical metadata
   to match the `CanonicalCentersResult` retained by the query result;
4. the current selected rows of `spatial_canonical` to equal the exact center
   snapshot used by the query; this is an eager NumPy comparison and never reads
   labels pixels;
5. the column mode, compatible dtype/category state, and matched-row values to
   match the preparation;
6. a freshly computed summary to equal `expected_summary`.

Palette metadata does not affect query membership or overwrite/removal counts,
so a palette-only change while the dialog is open does not invalidate the
reviewed annotation. Apply reads the current palette immediately before
mutation, preserves it when valid, and resolves missing or invalid state through
the palette policy above. It must never overwrite a newer valid palette with an
older dialog snapshot.

If only relevant target values changed while the review dialog was open, Apply
raises `SpatialAnnotationColumnChangedError` without mutation. The controller
rebuilds the preparation and summary, updates the dialog, and requires
confirmation again. A changed source, binding, or cache raises
`SpatialAnnotationQueryOutdatedError` and requires a new query. An absent or
incompatible target follows normal target-validation handling without mutating
the table.

For an effective mutation, construct the complete replacement target Series
off-table. Existing compatible dtype, categorical order, and categorical
`ordered` state are preserved. Set appends a missing string category without
reordering existing categories. Remove writes `pd.NA` at the resolved rows and
does not remove categories that become unused. A new column is categorical,
contains actual missing values outside the matched rows, and uses the applied
string as its first category. Construct the complete replacement palette
off-table as well whenever the palette is missing, invalid, or must be extended
for a new category.

Assign the completed Series and, when required, its completed palette on the
main thread as one atomic annotation consistency-unit update. Keep exact
pre-assignment snapshots of both `table.obs[column_name]` and the presence/value
of `table.uns["<column>_colors"]`. If either assignment or post-assignment
validation fails, restore both snapshots; remove newly created entries during
rollback. Do not publish an event or change dirty state until the complete
domain mutation succeeds. Other obs columns, other uns keys, other regions,
obsm, and the canonical cache remain untouched.

If `changed_count == 0`, return without assignment. Do not replace the column
object, create/repair palette metadata, publish a mutation, or alter dirty
state, and return `SpatialAnnotationApplyResult(False, False)`. After an
effective successful Apply, return `annotation_changed=True` and report
`palette_changed=True` only when `<column>_colors` was created, repaired, or
extended. The widget then publishes one ordinary `TableStateChangedEvent`
through `record_table_mutation()` for the target obs path plus the palette uns
path only when `palette_changed` is true, scoped to the selected labels region.
Creating a new column reports `change_kind="created"`; setting or removing
values in an existing column reports `change_kind="updated"`.

#### Recovery boundary

Spatial Query does not maintain widget-local annotation history and exposes no
operation-specific Undo command. The review dialog and mandatory overwrite
confirmation are the user-facing safeguards; atomic Apply and rollback protect
against partial failure.

After Apply, the annotation is an ordinary shared in-memory table mutation. A
user can correct it with another annotation Apply, persist it through Write
Table State, or discard covered unpersisted table changes through Reload Table
from zarr. Reload retains its existing warning and scope: it may also discard
other dirty table components covered by that reload, including an unpersisted
canonical cache.

#### Deliverables

- provenance-carrying `CanonicalCenterQueryResult` refinement;
- exact row resolution through region and instance keys;
- immutable preparation and summary contracts;
- explicit Apply result reporting annotation-column and palette changes;
- mode-specific set/overwrite/removal summaries;
- compatible existing-column set/removal and new categorical-column creation;
- valid-palette preservation, stable default palette creation/repair/extension,
  and category-color stability;
- apply-time source/cache/binding/table revalidation and atomic obs/uns
  rollback;
- ordinary shared table-state publication after an effective Apply;
- focused table-domain tests.

#### Exit criteria

- no cancel, no-result, no-op, outdated preparation, or failed apply changes the
  annotation column or companion palette;
- all effective mutations assign one completed annotation-column/palette
  consistency unit atomically;
- displayed counts are the exact counts accepted by Apply;
- apply rejects any changed binding, canonical-center snapshot, or relevant
  target-column state;
- an effective Apply produces one ordinary dirty obs mutation, additionally
  reports the palette uns path exactly when that palette changed, and never
  changes canonical cache paths;
- focused tests cover set/remove, new/existing, valid/missing/invalid palette,
  stable palette extension, no-op, rollback, and outdated preparation paths.

### Slice 6a: Shared coordinate-system change guard

**Implementation status: Implemented.**

This slice fixes an existing Shapes Annotation data-loss bug before changing
the widget architecture. Unsaved polygon edits live only in the editable
napari Shapes layer until Save shapes succeeds. Viewer, Object Classification,
or another widget could previously change the shared coordinate system, after
which Harpy removed layers belonging to the old coordinate system without
giving Shapes Annotation an opportunity to reject the change. The implemented
guard now closes that pre-change boundary.

The slice adds no parent widget, moves no selectors, and introduces no Spatial
Query behavior. The currently registered standalone `ShapesAnnotation` owns
the guard for its viewer session.

#### Guard API

```python
@dataclass(frozen=True)
class CoordinateSystemChangeRequest:
    sdata: SpatialData | None
    previous_coordinate_system: str | None
    coordinate_system: str | None
    source: str | None


type CoordinateSystemChangeGuard = Callable[
    [CoordinateSystemChangeRequest],
    bool,
]


class HarpyAppState:
    def set_coordinate_system_change_guard(
        self,
        guard: CoordinateSystemChangeGuard | None,
    ) -> None: ...

    def clear_coordinate_system_change_guard(
        self,
        guard: CoordinateSystemChangeGuard,
    ) -> bool: ...
```

The per-viewer app state supports one optional synchronous guard. This is
intentionally not a general mutable-guard registry: no other guarded dirty
coordinate-system workflow exists in the current application.

`set_coordinate_system()` first validates and normalizes a genuinely new
requested coordinate system. It then invokes the guard before changing
app-state fields, emitting `coordinate_system_changed`, or removing viewer
layers. `clear_coordinate_system()` follows the same guarded path.

If the guard returns `False`, `set_coordinate_system()` returns `False` and the
old coordinate system, Shapes edit session, widget selectors, emitted events,
and viewer layers remain unchanged. A rejected change originating from the
Shapes Annotation, Viewer, or Object Classification combo resynchronizes that
control to the unchanged app-state coordinate system. If the guard returns
`True`, the existing commit, event, and layer-removal flow proceeds.

#### Standalone Shapes Annotation integration

The current widget exposes one reusable session boundary:

```python
class ShapesAnnotation(QWidget):
    def try_close_edit_session(
        self,
        *,
        reason: Literal["coordinate_system", "shapes_target"],
    ) -> bool:
        """End the edit session, prompting before discarding unsaved changes."""
```

It returns `True` when no session exists, after releasing a clean session, or
after the user accepts discard. It returns `False` when the user cancels and
leaves the session unchanged. The `reason` preserves the existing specific
warning for a coordinate-system change versus a Shapes-target switch.

The standalone widget installs a guard that calls:

```python
shapes_annotation.try_close_edit_session(
    reason="coordinate_system",
)
```

Its own coordinate-system combo no longer performs a separate local release;
it calls `HarpyAppState.set_coordinate_system()` and relies on the same guard
used by every external source. Shapes-target changes call
`try_close_edit_session(reason="shapes_target")` directly because they are
local widget state rather than HarpyAppState state.

The flow is:

    any widget requests another coordinate system through HarpyAppState
        ↓
    standalone ShapesAnnotation coordinate-system guard
        ↓
    ShapesAnnotation.try_close_edit_session(reason="coordinate_system")
        ├── cancelled
        │       ↓ no app-state, selector, event, session, or layer change
        └── accepted
                ↓ widget completes its own session/layer cleanup
    HarpyAppState commits and emits the coordinate-system change
        ↓
    existing widget refresh and old-coordinate-system layer cleanup continue

The standalone widget removes its guard during teardown. SpatialData
replacement remains outside this guard: `set_sdata()` has broader dataset and
layer lifecycle semantics, and the shared dirty-dataset replacement guard
remains a Slice 8 responsibility.

#### Required implementation documentation

The guard is a safety boundary against silent loss of unsaved Shapes geometry,
not a generic callback convenience. The implementation must include:

- a `HarpyAppState.set_coordinate_system()` docstring stating that the guard
  runs before state mutation, `coordinate_system_changed` emission, and layer
  removal, and that rejection leaves all three untouched;
- guard-installation documentation describing the single-owner contract and
  teardown responsibility;
- a Shapes Annotation guard-callback docstring explaining that a coordinate
  change can originate from another widget while edits exist only in the
  editable napari layer;
- a concise inline comment at the guard invocation explaining why moving it
  after mutation or layer removal would permit silent data loss;
- focused test names and assertions that act as executable documentation.

A generic comment such as "run change guard" is insufficient; the code must
name the unsaved-Shapes data-loss condition and ordering guarantee directly.

Deliverables:

- `CoordinateSystemChangeRequest`, the optional HarpyAppState guard, and
  guarded `set_coordinate_system()`/`clear_coordinate_system()` behavior;
- public `ShapesAnnotation.try_close_edit_session(reason=...)` extracted
  from the current private dirty confirmation and cleanup paths;
- standalone Shapes Annotation guard installation and teardown;
- exactly-once routing for local and external coordinate-system changes;
- existing Shapes-target behavior routed through the same public session
  release helper without using the app-state guard;
- the required docstrings and safety-ordering inline comment;
- focused app-state and current Shapes Annotation regression tests.

Exit criteria:

- a rejected coordinate-system request from Shapes Annotation, Viewer, Object
  Classification, or another source changes no app-state field, emits no
  `coordinate_system_changed` event, removes no viewer layer, and preserves the
  edit session and unsaved geometry;
- an accepted request invokes the guard exactly once before state mutation,
  event emission, and layer removal, then follows the existing refresh flow;
- the local Shapes Annotation coordinate-system combo does not perform a
  second release check;
- Shapes-target cancellation continues to preserve the old selection and edit
  session with its target-specific warning;
- widget teardown removes the guard;
- code documentation explicitly explains the data-loss condition and
  safety-critical ordering;
- no parent Annotation widget, selector move, canonical-center behavior,
  Spatial Query behavior, or table mutation is introduced.

### Slice 6b: Parent Annotation widget foundation

**Implementation status: Implemented.**

This slice performs only the architectural refactor needed to establish the
final dock hierarchy:

    AnnotationWidget
        └── ShapesAnnotation

The Spatial Query child is not introduced or integrated yet. Slice 6b reuses
the tested coordinate-system guard from Slice 6a unchanged and transfers guard
installation ownership from the standalone Shapes child to the registered
parent.

The parent owns the shared selectors and committed selection; the child owns
the edit session. The existing private Shapes target becomes a small shared UI
model:

```python
@dataclass(frozen=True)
class ShapesAnnotationTarget:
    mode: Literal["create_new", "edit_existing"]
    existing_shapes_name: str | None = None


@dataclass(frozen=True)
class AnnotationContext:
    sdata: SpatialData | None
    coordinate_system: str | None
    shapes_target: ShapesAnnotationTarget | None
    has_unsaved_shapes_changes: bool

    @property
    def saved_shapes_name(self) -> str | None:
        target = self.shapes_target
        if target is None or target.mode != "edit_existing":
            return None
        return target.existing_shapes_name
```

`saved_shapes_name` is `None` for a proposed create-new name before its first
successful save. An existing selected Shapes element remains identified during
editing; `has_unsaved_shapes_changes=True` tells downstream consumers that its
saved geometry must not currently be queried. Dirty state remains derived from
the Shapes child's existing clean-layer snapshot rather than a parallel dirty
registry. These shared UI-only models live in
`widgets/annotation/models.py`, not the core spatial-query package.

The parent exposes:

```python
class AnnotationWidget(QWidget):
    annotation_context_changed = Signal(object)

    @property
    def annotation_context(self) -> AnnotationContext: ...
```

The child retains the Slice 6a `try_close_edit_session()` API and additionally
exposes:

```python
class ShapesAnnotation(QWidget):
    edit_session_dirty_changed = Signal(bool)
    shapes_target_change_requested = Signal(object)
    edit_session_saved = Signal(object)

    @property
    def has_unsaved_changes(self) -> bool: ...

    def apply_annotation_context(self, context: AnnotationContext) -> None:
        """Adopt a context already committed by the parent."""
```

`apply_annotation_context()` prepares or opens the edit workflow for a context
already committed by the parent. It must not prompt, alter parent selectors, or
change HarpyAppState.

The child has one context-driven construction mode. It owns the Shapes edit
controls, edit session, editable layer, and status feedback, but never creates
its own coordinate-system or Shapes-target selectors. Direct
`ShapesAnnotation(...)` construction remains supported for tests and
programmatic embedding; the child is inactive until
`apply_annotation_context()` supplies a usable context. Do not introduce an
`embedded=True` option, a compatibility mode that reconstructs the old outer
dock, or two parallel selector-ownership paths. The registered napari command
constructs `AnnotationWidget`, which supplies the complete user-facing
workflow.

The three child signals have distinct responsibilities:

- `edit_session_dirty_changed` emits whenever the child's snapshot-derived
  boolean dirty state changes; the parent republishes one corresponding
  `AnnotationContext`;
- `shapes_target_change_requested` carries a `ShapesAnnotationTarget` when a
  compatible active primary Shapes layer asks to become the selected target;
  the parent runs the ordinary guarded Shapes-target transition before
  committing it;
- `edit_session_saved` carries the resulting
  `ShapesAnnotationTarget.edit_existing(...)` after a successful save. When
  the parent context still contains `create_new`, this promotes the target
  without closing or reopening the now-clean edit session. For an already
  selected existing target, it refreshes context without creating a target
  transition.

`edit_session_saved` is a local parent-child UI notification. The existing
`ShapesElementWrittenEvent` remains the shared cross-widget notification that
an in-memory Shapes element was written; neither signal replaces or duplicates
the other's responsibility.

For coordinate-system changes from any widget, the parent-owned Slice 6a guard
delegates once to
`try_close_edit_session(reason="coordinate_system")`. After acceptance,
HarpyAppState commits and emits the change; the parent commits its selectors and
`AnnotationContext`, asks the child to adopt that context, and publishes one
final context notification. A Shapes-target change follows the same local
sequence with `reason="shapes_target"` but does not pass through HarpyAppState.
Cancellation restores the old selector and publishes no intermediate context.

The parent suppresses child dirty/session notifications during an accepted
transition. Siblings observe the old context after cancellation or one final
new context after acceptance, never a transient clean version of the old
target created during cleanup.

A successful first save is not a request to leave the edit session. The child
emits `edit_session_saved`, and the parent promotes `create_new` to
`edit_existing`, updates context, and publishes without releasing or reopening
the clean session. Compatible active-primary-Shapes adoption instead emits
`shapes_target_change_requested` and requests the normal guarded parent
Shapes-target transition.

Deliverables:

- `widgets/annotation/widget.py` with the registered parent
  `AnnotationWidget` and `widgets/annotation/models.py` with the two UI models;
- change only the existing Annotation command's Python target to
  `AnnotationWidget`, retaining command ID `napari-harpy.shapes_annotation`,
  display name Annotation, `Interactive(..., widgets="shapes_annotation")`,
  and exactly one dock contribution;
- move coordinate-system and Shapes-target selectors, option discovery, outer
  dock surface, logo, scroll area, selector form, and committed selection state
  into the parent;
- embed the remaining Shapes edit workflow without duplicate shared controls
  or dock chrome; keep `ShapesAnnotation` independently constructible and
  publicly importable as one context-driven child that remains inactive until
  supplied a usable `AnnotationContext`, with no dual standalone mode;
- transfer Slice 6a guard installation and teardown ownership to the parent
  without changing the guard's app-state semantics or safety documentation;
- implement parent commit, child context adoption, final-only context
  publication, and dirty-state reporting;
- route dirty-state publication, first save, and active-primary-Shapes adoption
  through their explicit child signals and distinct parent paths described
  above;
- preserve all polygon create/edit/hole/validate/save/discard behavior, status
  feedback, table events, and persistence behavior;
- update lazy exports and add focused parent, manifest, compatibility,
  transition, dirty-state, first-save, adoption, and guard-ownership tests.

Exit criteria:

- napari exposes one Annotation dock and no Spatial Query dock, with the visible
  Shapes workflow behaving as before;
- the parent is the single source and publication boundary for shared
  selectors and `AnnotationContext`;
- a proposed unsaved create-new name is never exposed as
  `saved_shapes_name`;
- cancellation preserves selectors, HarpyAppState, the edit session, and the
  published context; acceptance publishes exactly one final context after
  child cleanup and adoption;
- first save promotes the saved target without releasing or reopening the
  current session, while active-primary-Shapes adoption cannot replace a dirty
  session without accepted preflight;
- direct child construction creates no duplicate shared selector or outer dock
  mode and becomes operational only after receiving a usable context;
- the three child signals retain their distinct dirty-state, adoption-request,
  and successful-save responsibilities, while `ShapesElementWrittenEvent`
  remains the shared Shapes-write notification;
- the parent owns the guard lifecycle, and closing it removes the guard;
- existing direct `ShapesAnnotation` construction, the historical command ID,
  and the historical `Interactive` selector remain valid;
- no canonical-center, Spatial Query, or table-annotation behavior is added.

### Slice 6b follow-up: Shapes dirty-event filtering

**Implementation status: Implemented.**

This small follow-up tightens the dirty-state publication introduced by Slice
6b. Napari Shapes operations emit pre-mutation `data` actions (`ADDING`,
`REMOVING`, or `CHANGING`) followed by the corresponding completed action
(`ADDED`, `REMOVED`, or `CHANGED`). Rebuilding the complete
geometry-and-features snapshot for a pre-mutation event performs unnecessary
work and still sees the previous clean state.

The Shapes child therefore listens only to the active annotation layer's
`data` emitter, ignores all three pre-mutation actions, and evaluates dirty
state for the three completed actions. The callback contract is intentionally
strict: a connected event must expose `type="data"` and an `action`. Missing
actions or unexpected event types fail loudly instead of being normalized or
silently accepted.

The child deliberately does not subscribe to `layer.events.features`.
Napari-harpy currently provides no feature-only Shapes editing workflow, and
napari updates row-aligned features before emitting the completed `data` event
for ordinary shape additions and removals. Listening to `features` as well
would therefore repeat the same complete snapshot comparison. Features remain
part of the authoritative clean snapshot, so an unexpected feature-only
mutation is still detected when an edit session is closed.

This is only an event-filtering optimization. It must retain
`_annotation_layer_has_unsaved_changes()` and the existing exact comparison
against `_annotation_clean_snapshot` as the source of truth. An event must not
unconditionally mark the session dirty: if the geometry and features are
restored exactly to their clean state, the parent context must be allowed to
return to `has_unsaved_shapes_changes=False`.

Focused tests must establish that:

- `ADDING`, `REMOVING`, and `CHANGING` perform no dirty-state evaluation or
  parent-context publication;
- `ADDED`, `REMOVED`, and `CHANGED` each trigger dirty-state evaluation;
- a data event without an action fails loudly, and an unexpected `features`
  event is rejected by the data-only callback;
- repeated events that do not change the derived boolean do not produce
  duplicate `edit_session_dirty_changed` notifications.

### Slice 6c: Explicit coordinate-system change participant

**Implementation status: Implemented.**

This slice makes the pre-change relationship introduced in Slice 6a explicit
after Slice 6b has established the parent Annotation widget as its long-term
owner. The pre-change boundary remains in `HarpyAppState` because a shared
coordinate-system transition can otherwise discard another workflow's unsaved
state. Viewer and Object Classification must not acquire a direct dependency
on the Annotation widget.

The refactor replaces the opaque stored `Callable` with a named participant
contract:

```python
class CoordinateSystemChangeParticipant(Protocol):
    def prepare_coordinate_system_change(
        self,
        request: CoordinateSystemChangeRequest,
    ) -> bool: ...


class HarpyAppState:
    def register_coordinate_system_change_participant(
        self,
        participant: CoordinateSystemChangeParticipant,
    ) -> None: ...

    def unregister_coordinate_system_change_participant(
        self,
        participant: CoordinateSystemChangeParticipant,
    ) -> bool: ...
```

`AnnotationWidget` implements
`prepare_coordinate_system_change(request)` and delegates exactly once to its
Shapes child:

```python
def prepare_coordinate_system_change(
    self,
    request: CoordinateSystemChangeRequest,
) -> bool:
    return self.shapes_annotation.try_close_edit_session(
        reason="coordinate_system",
    )
```

`HarpyAppState.set_coordinate_system()` then expresses the mediation in terms
of the named role:

```python
participant = self._coordinate_system_change_participant
if participant is not None:
    if not participant.prepare_coordinate_system_change(request):
        return False
```

The contract remains one optional participant per viewer session. This is not
a general callback registry: only the parent Annotation workflow currently
owns state that must synchronously approve a coordinate-system change. The
registration API rejects replacement by a different active participant, and
unregistration is identity-safe so teardown of an older parent cannot remove
a newer owner's protection.

Viewer and Object Classification continue to depend only on
`HarpyAppState.set_coordinate_system()` and its documented possibility of
rejection. They do not import, discover, or call `AnnotationWidget` or
`ShapesAnnotation`. The explicit participant name makes the runtime mediation
discoverable without pretending that the unavoidable cross-workflow
coordination has disappeared.

Deliverables:

- add `CoordinateSystemChangeParticipant` as the named structural contract;
- remove the `CoordinateSystemChangeGuard` callable alias and the
  `set_coordinate_system_change_guard()` / identity-based clear API;
- replace them with explicit participant registration and identity-safe
  unregistration;
- make the Slice 6b parent `AnnotationWidget` implement and register the
  participant contract, delegating preflight to its Shapes child exactly once;
- remove bound-method guard storage from the Annotation widget;
- preserve the Slice 6a validation, rejection, selector-resynchronization,
  event-ordering, layer-preservation, and SpatialData-replacement boundaries;
- update documentation and focused app-state, parent, Viewer, and Object
  Classification tests to use participant terminology.

Exit criteria:

- constructing the Shapes child alone no longer modifies shared app-state
  transition behavior;
- the registered parent is the explicit coordinate-system change participant
  for its viewer session;
- app-state code calls a named participant method rather than an arbitrary
  stored callable;
- rejected and accepted transitions remain behaviorally identical to Slice
  6a, including exactly-once preflight and unchanged state on rejection;
- Viewer and Object Classification retain no dependency on Annotation widget
  types or internals;
- participant teardown is identity-safe and leaves no stale preflight owner;
- no Spatial Query, canonical-center, table mutation, or viewer-styling
  behavior is introduced.

### Slice 6d: Spatial annotation viewer-styling foundation

**Implementation status: Implemented.**

This slice isolates the small amount of primary-label layer orchestration that
is specific to spatial annotation. It must build on the existing generic
table-backed labels styling API rather than reproduce palette resolution or
Object Classification semantics.

The boundary is one stateless function rather than a controller or another
state-owning class:

```python
def load_and_style_spatial_annotation_labels(
    viewer_adapter: ViewerAdapter,
    *,
    sdata: SpatialData,
    coordinate_system: str,
    labels_name: str,
    table_name: str,
    column_name: str,
) -> LabelsLoadResult:
    ...
```

It performs only the following orchestration:

```text
validate categorical annotation column
    ↓
ViewerAdapter.ensure_labels_loaded()               # primary layer
    ↓
apply_table_color_source_to_labels_layer()
    ↓
ViewerAdapter.sync_labels_display_after_colormap_change()
    ↓
ViewerAdapter.activate_layer()
```

The function constructs the following generic styling request internally; its
callers do not need to understand generic viewer color-source configuration:

```python
TableColorSourceSpec(
    table_name=table_name,
    source_kind="obs_column",
    value_key=column_name,
    value_kind="categorical",
)
```

The selected existing column must use pandas categorical dtype and contain
only string categories, matching the Spatial Annotation apply contract. Expose
the existing private validation in `core/spatial_query/annotation.py` as one
shared core helper and use it from annotation preparation, this styling
boundary, and the later column-discovery implementation. Column validation
must happen before loading or changing a viewer layer. Table-binding
validation remains owned by `apply_table_color_source_to_labels_layer()` and
must not be repeated as a separate preflight scan.

Deliverables:

- a thin `widgets/spatial_query/viewer_styling.py` boundary for loading or
  reusing the selected labels element as the primary labels layer, activating
  it, and applying one selected annotation column as its color source;
- reuse of the existing `LabelsLoadResult`, including the layer-created and
  palette-resolution information; no spatial-query-specific result dataclass;
- a shared core validator for categorical string annotation columns, replacing
  the private Slice 5-only validation path rather than duplicating its rules;
- valid stored `<column>_colors` palette preservation and stable
  position-derived display defaults when that palette is missing or invalid,
  through the shared styling and palette APIs;
- missing annotation values rendered through the shared categorical missing
  color;
- no table, palette, dirty-state, canonical-cache, or persisted-data mutation
  during layer loading or styling;
- focused boundary tests for primary-layer creation, reuse, activation,
  coloring, validation-before-loading, and no-mutation behavior. Existing
  shared palette and labels-colormap tests remain responsible for stored,
  missing, invalid, and missing-value color semantics.

Palette resolution during styling is read-only:

- a valid stored `<column>_colors` palette is used unchanged;
- a missing or invalid stored palette produces the shared stable viewer-only
  fallback;
- styling never creates, repairs, or replaces `<column>_colors` in `.uns`;
- palette persistence remains part of an effective Spatial Annotation apply;
- a configured New column is not styleable until its first effective Apply has
  created the categorical column.

Exit criteria:

- the styling boundary contains only spatial-annotation layer orchestration;
- categorical-string column validation precedes viewer-layer loading or
  restyling, while the generic styling helper performs table-binding
  validation exactly once;
- the selected labels element is loaded or reused as the registered primary
  labels layer, styled, synchronized, and activated;
- it does not introduce a second labels overlay, classifier-specific class
  semantics, a color-source ownership policy, or a widget controller;
- it does not call `ViewerAdapter.ensure_styled_labels_loaded()`, because that
  API represents a separate styled overlay rather than the shared primary
  annotation layer;
- all palette selection remains delegated to the shared core/viewer contracts;
- only napari layer presentation state may change; the AnnData table, app
  dirty manifest, canonical cache, and persisted data remain unchanged;
- if inspection shows that an operation is already fully expressed by an
  existing helper, the spatial-query boundary delegates to that helper instead
  of wrapping it with additional state.

### Slice 6e: Spatial Query child widget shell

**Implementation status: Implemented.**

Build the Spatial Query child as an independently testable, embeddable widget.
It is not registered as a napari dock and is not yet composed into the parent
Annotation widget.

The public child boundary is:

```python
class SpatialQuery(QWidget):
    run_requested = Signal()

    def apply_annotation_context(
        self,
        context: AnnotationContext,
    ) -> None:
        ...
```

The signal is a parameterless action intent in this slice. No calculation or
query request dataclass is introduced merely to transport current control
values. The future execution layer captures and validates immutable
computational inputs synchronously when it accepts the intent, before starting
background work.

The child stores only the last parent-supplied `AnnotationContext` and the
current `CanonicalCacheReport`, when inspection succeeds. Labels, table,
target-column mode, and column name remain control-owned selection state and
are read from combo `itemData()` or the new-column line edit. Do not mirror
them in parallel `_selected_*` fields.

The target-column controls are explicit rather than represented by another
target dataclass:

```text
Target column mode: Existing column | New column

Existing column
    → compatible categorical-string column combo

New column
    → line edit initially containing "spatial_annotation"
```

Add one shared core discovery helper:

```python
def get_compatible_spatial_annotation_column_names(
    sdata: SpatialData,
    table_name: str,
) -> list[str]:
    ...
```

It preserves `.obs` column order, excludes the table `region_key` and
`instance_key`, ignores non-string column names, and applies the same internal
categorical-string predicate as
`require_compatible_spatial_annotation_column()`. Discovery and fail-loud
validation must not maintain separate definitions of column compatibility.

Selection dependencies are applied in this order:

```text
AnnotationContext.sdata + coordinate_system
    → supported 2D labels elements in that coordinate system
        → tables declaring the labels element as an annotated region
            → existing compatible columns or validated new-column intent
                → canonical-cache inspection for the labels/table pair
```

Refresh dependent controls with signals blocked. Preserve an explicitly
selected labels element by stable identity when it remains valid; otherwise
leave the labels selector at `Choose a labels element` and do not populate or
inspect its downstream table/cache state. Other dependent controls use the
documented `spatial_annotation` default policy once labels and table context
exists. A create-new Shapes target has no saved query geometry. A supplied
context with no `saved_shapes_name`, or with
`has_unsaved_shapes_changes=True`, disables Run in this child. Slice 6f wires
the live parent publication into this already-defined behavior rather than
adding another dirty-session policy.

Inspect the canonical cache once after a labels/table selection settles and
retain that report until an upstream selection or explicit refresh invalidates
it. Do not rebuild the report during status-card rendering. Report states map
to the shell as follows:

```text
VALID
    → Run will reuse existing centers

ABSENT
    → Run will calculate centers first

PARTIAL
    → Run will calculate centers for the selected labels region

STALE
    → Run will refresh centers for the selected labels region

INVALID
    → Run will report the mismatch and rebuild conservatively
```

Every successfully constructed report is rebuild-authorized through Run. Run
requires a saved clean Shapes context and a valid target-column intent; it
automatically reuses, calculates, refreshes, or rebuilds canonical centers as
indicated by the report.

Viewer styling is user-driven in this shell. An explicit labels or existing
target-column selection uses
`load_and_style_spatial_annotation_labels()` to load or reuse, style, and
activate the primary labels layer. Programmatic context and dependent-selector
refreshes do not silently reapply Spatial Query coloring. A New column is not
a color source before its first effective Apply.

Add `widgets/spatial_query/status_card.py` as a pure status-spec builder. It
receives already-derived selection and cache state and returns presentation
data; it does not inspect SpatialData, mutate widget state, or own controller
messages. Busy, cancellation, execution success, and execution error cards are
added when the execution flow is connected.

Shell validation remains cheap and synchronous. Before enabling Run it checks
the saved/clean Shapes context, supported 2D labels selection, linked table and
successful canonical inspection, and target-column intent. Complete
Shapes-geometry and element-to-element transformation snapshot validation is
performed synchronously by the execution flow after Run is requested and
before any expensive centroid worker starts; the shell must not duplicate the
core query-construction contract.

Deliverables:

- an unregistered Spatial Query child accepting the parent-supplied
  `AnnotationContext` rather than owning duplicate coordinate-system or Shapes
  selectors;
- child controls for labels, linked table, and target-column intent;
- a shared core discovery helper for compatible categorical string annotation
  columns, excluding `region_key` and `instance_key`, so Qt code does not
  duplicate the Slice 5 target rules;
- dependent filtering with stable identity preservation and default
  `spatial_annotation` new-column behavior;
- centroid-cache inspection status and one explicit Run action intent;
- primary labels-layer load/activation and existing-column visualization
  through the Slice 6d styling boundary;
- status cards, tooltips, accessible names, and focused selector/state tests.

The action controls in this shell validate state and emit intent only. They do
not calculate centers, run a query, open a review dialog, mutate a table, or
publish dirty state; those execution paths belong to Slice 7. The existing
`SpatialQueryController` is deliberately not constructed or driven by this
shell.

Exit criteria:

- Run is enabled only for a complete valid or rebuild-authorized request;
- first-run cost and cache reuse state are clear before execution;
- create-new or dirty Shapes context disables Run directly in the child;
- a compatible existing annotation column can be visualized without mutating
  the table, and a configured New column is not treated as available before
  its first effective Apply;
- programmatic refresh cannot silently reclaim primary-label coloring;
- in-memory SpatialData remains eligible for calculation and annotation even
  though persistence actions require a backed object;
- selection, inspection, and styling never mutate or dirty a table;
- focused tests can drive the child with supplied context without constructing
  the parent widget.

### Slice 6f: Annotation parent/child integration

**Implementation status: Implemented.**

Compose the two independently established children into the final dock
hierarchy:

    AnnotationWidget
        ├── shared Coordinate System selector
        ├── shared Shapes selector
        ├── ShapesAnnotation
        └── SpatialQuery

Keep the children stacked in that order in the parent's existing scroll area:
Shapes Annotation first and Spatial Query directly below it. Do not introduce
tabs, another nested navigation control, a second top-level widget surface, or
a separate dock. Construct both children with the same napari viewer so they
resolve the same per-viewer `HarpyAppState`.

This slice wires shared selection and dirty-session context only. It does not
yet implement asynchronous calculate-query-review-apply execution.

Use the parent's existing final-context publication as the only Spatial Query
context input:

```python
self.annotation_context_changed.connect(
    self.spatial_query.apply_annotation_context,
)
```

Establish this connection before the parent's initial `refresh_from_sdata()`
so the Spatial Query child receives the first published context. Do not also
call `SpatialQuery.apply_annotation_context()` through a parallel path. The
parent continues to call Shapes child commands directly where it needs
synchronous completion, return values, exception propagation, or signal
blocking; `annotation_context_changed` is the already-established publication
boundary for the final immutable context consumed by other children and
observers.

The normal publication order is:

```text
AnnotationWidget builds the candidate AnnotationContext
    ↓
ShapesAnnotation.apply_annotation_context(context)
    ↓
parent reads the Shapes child's final dirty state
    ↓
parent stores and emits the final AnnotationContext
    ↓
SpatialQuery.apply_annotation_context(context)
```

Dirty-state publication follows the same boundary:

```text
ShapesAnnotation.edit_session_dirty_changed(dirty)
    ↓
AnnotationWidget replaces has_unsaved_shapes_changes in its context
    ↓
AnnotationWidget emits annotation_context_changed(context)
    ↓
SpatialQuery refreshes its readiness from that context
    ↓
dirty=True disables Run
```

Spatial Query must not read Shapes child state directly, and the parent must
not duplicate the child's Run blocker. A successful Shapes save, discard, or
other final dirty-to-clean transition republishes the resulting context so the
Spatial Query child observes the current saved in-memory geometry. The child
retains valid labels, table, target-mode, and column selections across context
updates for the same SpatialData object; a SpatialData replacement resets
those dependent selections through its existing
`apply_annotation_context()` contract.

`SpatialQuery.apply_annotation_context()` must distinguish context fields that
invalidate its selection dependencies from fields that affect only query
readiness. A SpatialData identity or coordinate-system change refreshes the
labels, linked tables, target columns, and canonical-centers cache report. A
Shapes-target or `has_unsaved_shapes_changes` change for the same SpatialData
and coordinate system updates readiness only; it must retain the current
selectors and captured cache report and must not recalculate the selected
region's instance-set digest:

```python
previous_context = self._annotation_context
selection_dependencies_changed = (
    context.sdata is not previous_context.sdata
    or context.coordinate_system != previous_context.coordinate_system
)
self._annotation_context = context

if not selection_dependencies_changed:
    # Shapes target and dirty state affect Run readiness, but they do not
    # invalidate the labels/table selection or its captured cache report.
    self._refresh_controls_and_status()
    return

# Refresh dependent selectors and inspect the canonical-centers cache.
```

Keep this distinction explicit in the method docstring and retain an inline
comment at the early return. The optimization is part of the lifecycle
contract, not merely a micro-optimization: Shapes dirty-state publication must
not trigger repeated table instance-set hashing. User-driven labels/table
changes continue to inspect through their existing handlers.

Deliverables:

- embed the Spatial Query child in the existing parent Annotation widget while
  retaining one Annotation manifest entry and adding no separate Spatial Query
  dock contribution;
- one parent-owned and published `AnnotationContext` shared with both children;
- connect the parent's existing `annotation_context_changed` signal to the
  Spatial Query child's `apply_annotation_context()` method before initial
  refresh, with no duplicate direct context-delivery path;
- unsaved edits to the selected Shapes element reach the child through
  `AnnotationContext.has_unsaved_shapes_changes`, exercising the child's
  existing Run blocker and explanation; no second parent-owned blocking rule
  or general cross-widget Shapes dirty-state registry is introduced;
- refresh the Spatial Query child after a successful Shapes save or discard so
  it observes the current saved in-memory geometry;
- preserve the Shapes edit session when the Spatial Query child activates the
  primary labels layer;
- keep `run_requested` as an unconnected intent-only signal in this integration
  slice; do not construct or drive the Spatial Query controller here;
- focused tests covering initial context delivery, coordinate-system and
  Shapes-target publication, dirty/clean transitions, successful-save refresh,
  same-SpatialData selection preservation, SpatialData-replacement reset,
  Shapes-session preservation during labels-layer activation, and existing
  Shapes Annotation regressions;
- a focused inspection-count test proving that dirty-only and Shapes-target-only
  context publications reuse the captured cache report, while SpatialData or
  coordinate-system changes follow the full dependent-selector and cache
  refresh path.

Exit criteria:

- napari exposes one Annotation dock containing both child workflows and no
  separate Spatial Query dock;
- coordinate-system and Shapes-target selection have one parent-owned source
  of truth published to both children;
- dirty selected Shapes geometry blocks Run, while a successful save refreshes
  the Spatial Query child and makes the saved in-memory geometry eligible;
- dirty-only and Shapes-target-only context changes never re-inspect the
  canonical-centers cache or recalculate its instance-set digest;
- parent context changes consistently refresh or invalidate dependent Spatial
  Query selections and intent;
- Spatial Query action signals remain execution-free and cause no calculation,
  query, dialog, table mutation, or dirty-state publication in this slice;
- the existing Shapes Annotation workflow remains behaviorally unchanged;
- integration itself does not calculate centers, run a query, apply an
  annotation, or dirty a table.

### Slice 6g: Spatial Query labels-selection and viewer-layer lifecycle

**Implementation status: Implemented.**

Align Spatial Query with the established Object Classification labels
selection lifecycle. A labels selection is an explicit request to load and
display that element in the active coordinate system; it must not survive a
coordinate-system transition or the removal of its corresponding primary
labels layer merely because the SpatialData element itself remains available.

This slice deliberately supersedes the shell-stage stable-identity
preservation described in Slice 6e and the same-SpatialData preservation from
Slice 6f for coordinate-system changes. Shapes-target and dirty-state-only
context publications still preserve all Spatial Query selections and reuse the
captured cache report.

The coordinate-system transition is:

```text
accepted coordinate-system change
    ↓
AnnotationWidget publishes the new AnnotationContext
    ↓
SpatialQuery repopulates supported labels choices
    ↓
clear the labels selection even if its previous name remains available
    ↓
show "Choose a labels element"
    ↓
clear linked table, target-column, cache-report, styling-error, and readiness state
    ↓
do not load a labels layer and do not inspect the canonical-centers cache
```

The app-state boundary continues to remove Harpy-managed viewer layers outside
the newly active coordinate system. Spatial Query must not attempt to preserve
or immediately reload the old primary labels layer in the new coordinate
system. The user makes a new explicit labels choice after the transition.

Spatial Query also subscribes to the existing
`ViewerAdapter.primary_labels_layers_changed` signal, as Object Classification
already does. It resolves live availability through
`get_loaded_primary_labels_layer()` rather than inspecting napari layer names
or maintaining a second layer reference. The removal flow is:

```text
selected primary labels layer is removed from napari
    ↓
ViewerAdapter unregisters its binding
    ↓
primary_labels_layers_changed is emitted
    ↓
SpatialQuery finds no matching loaded primary layer
    ↓
clear labels selection and every dependent control/report
    ↓
show "Choose a labels element" and disable Run
```

Removing an unrelated primary labels layer must not affect the Spatial Query
selection. Likewise, insertion of a primary labels layer by Viewer, Object
Classification, or another consumer must not auto-select it in Spatial Query.
The final invariant is one-way: a non-empty Spatial Query labels selection
requires the matching primary labels layer to be loaded for the current
SpatialData object and coordinate system, while a loaded layer does not imply
that Spatial Query selected it.

Keep this synchronization event-driven and control-owned:

- do not add a parallel selected-labels field or retain a napari layer object
  on `SpatialQuery`;
- clear the combo and downstream state through one explicit helper so
  coordinate changes and selected-layer removal follow the same contract;
- guard only the widget's own synchronous load/style path if necessary to
  avoid treating its successful layer registration as a disappearance;
- clearing the selection must not inspect or mutate the canonical cache,
  change the table, or affect the Shapes edit session;
- widget destruction relies on Qt receiver teardown in the same way as the
  existing Object Classification adapter-signal connection.

Deliverables:

- reset Spatial Query labels selection on every accepted coordinate-system
  change, including when the previous element is valid in the new system;
- retain the populated options but restore the `Choose a labels element`
  placeholder and require a new explicit choice;
- clear linked table, target-column controls, captured cache report and errors,
  and Run readiness without performing cache inspection;
- consume `primary_labels_layers_changed` and clear state when the selected
  primary labels layer is manually removed;
- ignore unrelated primary-label removal and all unrequested layer insertion;
- update the earlier preservation-focused test and add focused
  coordinate-change, selected-layer-removal, and unrelated-layer-removal
  coverage.

Exit criteria:

- coordinate-system changes never preserve or auto-reload the Spatial Query
  labels selection;
- manual removal of the selected primary labels layer leaves no selected
  labels/table/cache state behind and disables Run;
- unrelated viewer-layer changes do not disturb a valid current selection;
- no cache digest is recalculated until the user explicitly selects labels
  again;
- Object Classification and Spatial Query present the same labels-reset
  behavior without sharing widget-local state.

### Slice 6h: Unified Spatial Query status card

**Implementation status: Implemented.**

Replace the separate centroid-cache and Run-readiness cards introduced by the
Spatial Query shell with one status card. The cache is no longer an independent
user action: Run automatically reuses, calculates, extends, refreshes, or
rebuilds centers. Presenting cache state separately can therefore duplicate
selection blockers or show a warning card beside a successful `Spatial Query
Ready` card. The unified card must explain one coherent current state.

This slice deliberately supersedes the two-label presentation from Slice 6e.
It changes presentation only; Run enablement, cache inspection, labels-layer
lifecycle, controller boundaries, and table dirty state remain unchanged.

Use one widget label, named for the complete workflow rather than one
subsystem:

```python
self.status_label = QLabel()
self.status_label.setObjectName("spatial_query_status_label")
```

Remove `cache_status_label` and `readiness_status_label`. Place the unified
status card after the selector form and before the Run button, reusing the
existing shared status-card styling and word wrapping.

Consolidate the two pure builders into one:

```python
def build_spatial_query_status_card_spec(
    *,
    has_spatialdata: bool,
    coordinate_system: str | None,
    saved_shapes_name: str | None,
    has_unsaved_shapes_changes: bool,
    labels_name: str | None,
    table_name: str | None,
    cache_report: CanonicalCacheReport | None,
    canonical_input_inspection_error: str | None,
    target_error: str | None,
    target_description: str | None,
    layer_styling_error: str | None,
) -> _SpatialQueryStatusCardSpec:
    ...
```

The builder remains a presentation-only function. It must not inspect
SpatialData, calculate a digest, mutate widget state, decide whether Run is
enabled, or own controller messages. The widget continues to derive Run
enablement from the same validated state and passes already-derived values to
the builder.

Use deterministic priority for incomplete or exceptional state:

```text
1. no SpatialData
2. no coordinate system
3. no saved Shapes element
4. dirty Shapes edit session
5. no labels selection
6. no linked table
7. live labels-source or table-binding inspection failure
8. invalid annotation-target intent
9. non-blocking labels-layer styling warning
10. complete request, presented according to cache state
```

The first applicable state owns the title, message, and status kind. A live
canonical-input inspection failure uses:

```text
title: Labels or Table Validation Failed
kind: error
message: the captured user-facing labels-source or table-binding validation error
consequence: Spatial Query cannot calculate centers until this issue is resolved
Run: disabled
```

For a complete labels/table selection, the widget must supply exactly one
inspection outcome: either a `CanonicalCacheReport` or
`canonical_input_inspection_error`. Supplying neither is an inconsistent
internal orchestration state, not a user-facing cache condition.
`SpatialQuery._refresh_controls_and_status()` must fail loudly with
`RuntimeError` before deriving Run enablement or invoking the presentation-only
builder. As a secondary function-contract safeguard, the builder also rejects
both inconsistent argument combinations with `ValueError`: neither outcome
supplied, or both a report and an error supplied. A labels-layer styling
warning remains non-blocking and must say that Spatial Query can still run when
the computational request is otherwise valid.

For a complete request, incorporate cache behavior into the same card:

```text
VALID
    → title: Spatial Query Ready
    → kind: success
    → say cached centers will be reused

ABSENT
    → title: Spatial Query Ready
    → kind: info
    → say centers will be calculated before querying

PARTIAL
    → title: Spatial Query Ready
    → kind: info
    → say centers for the selected labels region will be added

STALE
    → title: Spatial Query Ready
    → kind: warning
    → say centers for the selected region will be refreshed

INVALID
    → title: Spatial Query Ready
    → kind: info
    → say centers for the selected labels element will be recalculated before querying
    → keep technical mismatch detail available only in the tooltip
```

Every complete-state card also identifies the saved Shapes element, selected
labels element, and target-column description. Warning or informational color
does not imply that Run is disabled: absent, partial, stale, and rebuildable
invalid reports remain valid Run prerequisites because their recovery is
automatic.

An `INVALID` report is a successfully inspected, automatically recoverable
cache state; it is not a user-actionable cache error. Run remains enabled. On
Run, the existing managed cache is treated conservatively, no previously
stored region is trusted, and centers for the currently selected labels
element are recalculated before the query continues. Other regions are
recalculated later only when selected. Do not expose the mismatch code in the
primary status text or imply that every region will be recalculated eagerly.
The mismatch detail may remain in the tooltip for diagnostics.

This is distinct from a live source or table-binding inspection failure.
`inspect_canonical_cache()` first needs to construct a trustworthy current
labels-source signature and selected-region table binding. A `ValueError`,
`TypeError`, or `KeyError` while resolving those live inputs means that no
`CanonicalCacheReport` can be constructed:

```text
current labels source and table binding validate
    → a CanonicalCacheReport is returned
    → INVALID stored cache remains automatically recoverable
    → Run stays enabled

current labels source or table binding does not validate
    → inspection raises before a report can be constructed
    → no trustworthy source signature or row/instance binding exists
    → recalculation cannot safely start
    → show "Labels or Table Validation Failed"
    → explain that centers cannot be calculated until the issue is resolved
    → Run stays disabled
```

Examples of the blocking branch include a removed or unreadable selected
labels element, malformed table linkage metadata, a missing region or instance
key column, no rows for the selected labels region, and non-integer,
non-positive, or duplicate region-local instance IDs. Discarding the existing
cache cannot repair these live input problems. Conversely, malformed managed
matrix or metadata contents are converted into deterministic mismatches on an
`INVALID` report and must not enter this blocking branch.

When Slice 7 connects execution, the same label displays busy, cancellation,
no-result, success, and error messages. Controller-owned execution status
temporarily takes presentation precedence over selection/cache readiness.
After execution finishes or is dismissed, the widget rebuilds the ordinary
unified status from current state. Do not reintroduce a second execution-only
status label.

Deliverables:

- replace `cache_status_label` and `readiness_status_label` with one
  `status_label`;
- replace the two status-spec builders with one pure unified builder;
- rename the captured failure to `canonical_input_inspection_error` and pass it
  through the builder rather than constructing an exceptional card directly
  in the widget;
- make `_refresh_controls_and_status()` fail loudly before status rendering
  when a complete labels/table selection supplies neither a cache report nor a
  canonical-input inspection error;
- make the pure builder explicitly require exactly one of `cache_report` and
  `canonical_input_inspection_error` after labels and table selection are
  complete;
- retain the existing `_SpatialQueryStatusCardSpec` and shared rendering
  helper;
- remove duplicate unavailable/required messages and contradictory adjacent
  success/warning cards;
- update focused shell and parent-integration assertions to inspect the unified
  label.

Exit criteria:

- Spatial Query renders exactly one workflow status card;
- every selection blocker and cache state has one deterministic presentation;
- a live labels-source or table-binding inspection failure is presented as
  `Labels or Table Validation Failed`, includes the validation message and its
  consequence for center calculation, and disables Run without calling the
  cache invalid;
- a returned `INVALID` report remains informational, recoverable, and
  Run-enabled;
- a complete rebuild-authorized request never shows a separate success card
  beside a cache warning;
- first-calculation, reuse, refresh, and rebuild behavior remain clear before
  Run;
- status rendering performs no cache inspection, hashing, viewer mutation, or
  table mutation;
- Run enablement is unchanged by the presentation refactor;
- the one label is ready to receive Slice 7 execution status without adding a
  parallel status surface.

### Slice 6i: Neutral unannotated primary-label styling

**Implementation status: Implemented.**

Align the Spatial Query New-column presentation with Object Classification's
neutral unannotated presentation. Before this slice, the shell loaded the
primary labels layer through `ViewerAdapter.ensure_labels_loaded()` but left
napari's per-instance default labels colors in place when the configured New
column did not exist yet. Those colors had no annotation meaning and could
suggest that instances already belonged to distinct annotation categories.

This slice deliberately refines the earlier rule that a New column is not a
color source before its first effective Apply. The nonexistent column still
must not be treated as a table-backed color source. Instead, Spatial Query
applies a viewer-only neutral state:

```text
New column, or no existing target column selected
    → load or reuse the primary labels layer
    → render background label 0 transparently
    → render every foreground label with the shared missing/unannotated color
    → do not create an obs column or palette

compatible Existing column selected
    → use its current categorical values
    → use its valid stored palette or the shared read-only fallback palette

Slice 7: first effective Apply creates the New column
    → store the categorical column and its palette atomically
    → keep that column selected as an Existing target
    → immediately replace the neutral presentation with table-backed coloring
```

Keep the current named-default behavior. `spatial_annotation` is the preferred
conventional target, but it is not an unconditional color source:

```text
user explicitly selects a labels element in Spatial Query
    → select its linked annotation table
    → discover compatible categorical string columns
    ↓
compatible "spatial_annotation" column exists
    → select Existing column mode
    → select "spatial_annotation"
    → color from its current categorical values and resolved palette

"spatial_annotation" column does not exist
    → select New column mode
    → propose "spatial_annotation" as the new column name
    → apply the neutral unannotated presentation

"spatial_annotation" column exists but is incompatible
    → exclude it from the Existing-column choices
    → do not convert, repair, or overwrite it
    → select New column mode with the colliding proposed default name
    → apply the neutral unannotated presentation
    → disable Run until the target collision is resolved
    → require another compatible Existing column or a different New name
```

Compatibility here means pandas categorical dtype with only string categories.
Plain integer, string, or object dtype is not compatible, and neither is a
categorical column with non-string categories.

The incompatible named-default branch must explain why the existing
`spatial_annotation` column was not selected. A generic message that the New
column name already exists is insufficient on its own. The status card should
state that the existing column cannot be used because Spatial Query requires a
categorical column with string categories, and direct the user to select
another compatible Existing column or enter a different New-column name.

Do not automatically select the first differently named compatible column.
Those columns remain available in the Existing-column dropdown, but choosing
one is an explicit user action. Once the user explicitly selects another
compatible Existing column, that column is both the annotation target and the
color source; later programmatic refreshes must not force the selection back
to `spatial_annotation`.

This convention parallels Object Classification without erasing the difference
between the workflows:

```text
Object Classification
    → fixed conventional annotation column: "user_class"
    → available: color from its values
    → absent: show the unannotated presentation

Spatial Query
    → preferred conventional annotation column: "spatial_annotation"
    → compatible and selected: color from its values
    → absent: propose it as New and show the neutral presentation
    → another compatible Existing column may be selected explicitly
```

Only an explicit labels or annotation-target selection inside Spatial Query
may claim the primary layer's presentation. A labels layer loaded by Viewer,
Object Classification, or another workflow must not cause Spatial Query to
select `spatial_annotation` or restyle that layer merely because it appeared
in napari.

Keep this as a stateless presentation boundary beside
`load_and_style_spatial_annotation_labels()` in
`widgets/spatial_query/viewer_styling.py`. Add a narrowly named helper rather
than making a nonexistent column look like a `TableColorSourceSpec`, for
example:

```python
def load_and_style_unannotated_spatial_annotation_labels(
    viewer_adapter: ViewerAdapter,
    *,
    sdata: SpatialData,
    coordinate_system: str,
    labels_name: str,
) -> LabelsLoadResult:
    ...
```

The helper reuses `ViewerAdapter.ensure_labels_loaded()`, the existing labels
colormap primitives, and the shared categorical missing/unannotated color. It
then synchronizes the labels display and activates the primary layer. It must
not import or reuse Object Classification's specialized
`ViewerStylingController`: that controller owns integer `user_class` and
classifier semantics, while Spatial Query needs only the same neutral visual
meaning.

The neutral colormap should use one constant default foreground color rather
than materializing one value or color per instance. Applying it must not scan
labels pixels, calculate canonical centers, derive table row mappings, or
install fake layer features for a column that does not exist.

The implementation ownership is intentionally split at the Apply boundary:

```text
Slice 6i
    → provide the stateless neutral-style helper
    → apply neutral or Existing-column styling from explicit selection actions
    → expose the styling paths needed after a later annotation Apply

Slice 7
    → perform the first effective annotation Apply
    → create the New categorical column and palette atomically
    → retain that column as the selected Existing target
    → invoke Existing-column styling for the newly created annotation
```

Slice 6i does not introduce or simulate annotation Apply merely to demonstrate
the final transition. Slice 7 owns that operational transition because it owns
the review-and-Apply flow. Slice 6i only establishes and tests the presentation
helpers and selection-driven behavior that Slice 7 will consume.

Styling remains driven by explicit user actions:

- explicitly selecting a labels element applies either its selected Existing
  column or the neutral New-column presentation;
- explicitly switching to New column, or entering Existing mode without
  selecting a compatible column, applies the neutral presentation;
- explicitly selecting a compatible Existing column applies table-backed
  coloring immediately;
- programmatic parent-context, selector, and table refreshes do not reclaim the
  primary layer's presentation from another workflow;
- changing only the proposed New-column text does not repeatedly restyle the
  layer.

The neutral state is strictly viewer-only. It must not:

- create the proposed `.obs` column;
- create, repair, or remove `<column>_colors` in `.uns`;
- emit `TableStateChangedEvent`, mark the table dirty, or write persisted data;
- mutate the canonical-centers cache;
- establish a fake table-backed color-source identity.

Deliverables:

- a stateless neutral-style helper in the existing Spatial Query
  `viewer_styling.py` module;
- Spatial Query control wiring that replaces raw napari labels colors with the
  neutral presentation whenever an explicit user coloring action has no
  selected Existing source;
- focused tests for transparent background, neutral foreground, primary-layer
  reuse and activation, no table/cache mutation, explicit Existing/New
  transitions, incompatible named-default diagnosis, and no styling
  reclamation during programmatic refresh.

Exit criteria:

- a newly selected labels element never shows semantically meaningless
  per-instance napari colors merely because the target column has not been
  created;
- every foreground instance uses the shared missing/unannotated color until an
  actual compatible annotation column supplies category values;
- selecting or displaying the neutral state remains read-only and leaves
  `.obs`, `.uns`, dirty state, persisted state, and the canonical cache
  unchanged;
- an incompatible existing `spatial_annotation` column is never silently
  coerced or overwritten, keeps Run disabled while its name collides with the
  proposed New target, and produces an actionable compatibility message;
- Existing-column palette behavior remains unchanged;
- the helper and control boundaries required for Slice 7 to replace neutral
  styling after the first effective Apply are available without Slice 6i
  implementing Apply itself;
- Object Classification and Spatial Query share the neutral visual convention
  without sharing widget-specific controllers or annotation-domain semantics.

### Slice 6j: Empty New-column draft and named-default separation

**Implementation status: Implemented.**

Separate the preferred conventional Existing-column name from the editable
New-column value. After Slice 6i, `spatial_annotation` was used for both roles,
which produced an avoidable collision when a compatible column already
existed and the user explicitly switched from Existing to New mode:

```text
compatible "spatial_annotation" exists
    → Existing mode initially colors from it
    ↓ user switches to New mode
hidden New-column value is also "spatial_annotation"
    → neutral styling is correct
    → proposed New target immediately collides with the existing column
    → Run is disabled for a name the user never entered
```

Follow the Shapes Annotation naming convention instead. Keep
`spatial_annotation` as the preferred Existing-column name and as placeholder
guidance, but do not install it as the `QLineEdit` value:

```text
New-column QLineEdit
    text        → ""
    placeholder → "spatial_annotation"

placeholder text
    → visual guidance only
    → is not a selected annotation target
    → is never passed to validation, query, Apply, or persistence as a value
```

Use a name that reflects this single remaining role, for example:

```python
_PREFERRED_ANNOTATION_COLUMN_NAME = "spatial_annotation"
```

Do not retain a constant whose name or usage implies that the New-column field
contains a valid default value.

The initial selection contract becomes:

```text
compatible "spatial_annotation" exists
    → select Existing mode
    → select and color from "spatial_annotation"
    → keep the hidden New-column draft empty

"spatial_annotation" does not exist
    → select New mode
    → keep the New-column draft empty
    → show "spatial_annotation" only as placeholder guidance
    → apply neutral styling
    → disable Run until the user enters a valid unused name

"spatial_annotation" exists but is incompatible
    → exclude it from Existing choices
    → select New mode with an empty draft
    → apply neutral styling
    → do not convert, repair, or overwrite the existing column
    → explain why it cannot be used
    → require another compatible Existing column or an explicitly entered,
      different New-column name
```

An empty draft normally produces a concise request to enter a New-column name.
When an incompatible `spatial_annotation` column exists, the empty-state
message must additionally explain that the conventional column was not
selected because Spatial Query requires pandas categorical dtype with only
string categories. Once the user enters a different valid unused name, that
unrelated incompatible column no longer blocks Run.

Treat entered New-column text as a user draft scoped to its current
SpatialData/table target:

- switching temporarily between Existing and New modes preserves a non-empty
  draft;
- programmatic selector refreshes that retain the same table preserve it;
- changing or clearing the selected table context clears it rather than
  carrying a proposed schema change into another table;
- changing only the draft text updates readiness but does not repeatedly
  reclaim or restyle the primary labels layer;
- New mode always uses the neutral presentation, regardless of whether the
  draft is empty, valid, or temporarily invalid.

Do not introduce a separate draft model or another source of widget state for
this behavior. The `QLineEdit` owns the draft, while the existing
signal-blocked selector-refresh paths decide whether its text is preserved or
cleared.

Slice 7 owns successful consumption of the draft:

```text
first effective Apply creates the explicitly named New column
    → store its categorical values and palette atomically
    → select it as an Existing target
    → replace neutral styling with table-backed styling
    → clear the consumed New-column draft
```

Deliverables:

- split the preferred conventional name from the New-column field value;
- initialize and reset the New-column field to empty while showing
  `spatial_annotation` as placeholder guidance;
- preserve a user-entered draft only while its SpatialData/table target
  remains current;
- retain neutral viewer styling throughout New mode;
- retain actionable incompatible-conventional-column feedback without relying
  on a prefilled colliding value;
- focused tests for compatible, absent, and incompatible conventional columns,
  mode-toggle draft preservation, table-context draft clearing, placeholder
  non-semantics, and unchanged viewer/table/cache state.

Exit criteria:

- switching from a compatible Existing `spatial_annotation` target to New mode
  never presents an immediate collision for text the user did not enter;
- `spatial_annotation` remains the preferred compatible Existing target but is
  only placeholder guidance in New mode;
- an empty New-column draft never enables Run or becomes an implicit Apply
  target;
- an incompatible existing `spatial_annotation` column remains visible as an
  actionable validation problem without being coerced or overwritten;
- explicit New-column drafts survive mode toggles only while they still belong
  to the same SpatialData/table target;
- neutral and Existing-column styling behavior from Slice 6i remains
  unchanged;
- this slice performs no annotation Apply, table mutation, dirty-state change,
  persistence write, or canonical-cache mutation.

### Slice 7: Async calculate-query-review-apply flow

This slice connects the validated action intents exposed by the integrated
Spatial Query child after Slice 6j to the existing core calculation, query, and
annotation APIs.

Deliverables:

- worker orchestration with one monotonically increasing operation ID
  spanning an optional centroid-calculation phase and a spatial-query phase;
- textual busy status, cancellation, cleanup, and error routing;
- main-thread cache update/revalidation;
- Spatial Query child-owned shared-state publication after an accepted cache
  update. The controller returns `CanonicalCentersResult` through
  `on_centers_ready` and remains unaware of `HarpyAppState`; the child uses the
  parent Annotation widget's shared app state to translate an actual
  `result.cache_update` into one `TableStateChangedEvent` for
  `CANONICAL_CACHE_PATHS` and calls `record_table_mutation(event)`;
- move the temporary `record_canonical_cache_update()` adapter into the Spatial
  Query child's accepted-centers callback and remove `cache_state.py` once the
  child owns this behavior, matching the Object Classification
  controller-to-widget event boundary;
- no-result outcome;
- Apply dialog with explicit Set annotation and Remove annotation modes, live
  mode-specific counts, and mandatory overwrite/removal warnings;
- main-thread atomic annotation Apply;
- Spatial Query child-owned publication of the obs path and the palette uns
  path indicated by `SpatialAnnotationApplyResult`;
- targeted primary labels-layer refresh after an effective annotation Apply,
  with missing values rendered through the shared missing/unlabelled color;
- controller/dialog async tests.

The centroid-calculation phase is:

    worker: calculate_canonical_centers()
        ↓
    main thread: accept and apply cache payload
        ↓ success
    AnnotationWidget
        ↓ delegates accepted result to embedded child
    SpatialQuery._on_canonical_centers_ready(result)
        ↓ result.cache_update is not None
    record one TableStateChangedEvent for both canonical paths
        ↓
    HarpyAppState marks the canonical consistency unit dirty

It is skipped when the selected-region cache is already valid. Valid cache
reuse, cancellation, worker failure, or rejected payloads never publish a
canonical table mutation event.

The spatial-query phase is:

    worker: evaluate canonical centers against the Shapes geometry
        ↓
    main thread: accept result and open the review dialog

The same operation ID governs both phases; do not introduce a separate
job-ID concept. The controller also records which phase owns the active worker.
Cancellation, selection changes, reload, a newer operation, or parent/child
shutdown invalidate that operation ID, and every late signal from either phase
is ignored. A parent context change or the selected Shapes session becoming
dirty has the same invalidating effect.

Exit criteria:

- napari remains responsive during global aggregation;
- cancel/stale/error paths cannot update the cache, open late dialogs, or annotate;
- the Spatial Query child, not `SpatialQueryController`, owns construction and
  publication of the accepted canonical `TableStateChangedEvent` through the
  parent Annotation widget's shared app state;
- cache update and annotation application are visibly distinct state
  changes;
- users always see affected and overwrite counts before annotation mutation.

### Slice 8: Persistence UX and cross-widget synchronization

Deliverables:

- the parent Annotation widget, including its Spatial Query child, Viewer, and
  Object Classification targeted refresh behavior consuming
  `table_state_changed` without feedback loops;
- shared primary-label color-source ownership implementing the rule that the
  most recent explicit coloring action controls the layer, so later unrelated
  table events cannot let another widget silently reclaim its styling;
- reusable Write Table State and Reload Table from zarr UI components backed by
  the generalized PersistenceController;
- dirty reload write/discard/cancel behavior;
- dirty-dataset close/replacement guard;
- multi-widget and backed-zarr integration tests.

Exit criteria:

- all widgets observe one current in-memory table state;
- canonical centers, annotation columns, and changed companion palettes
  persist/reload together;
- Write Table State persists the union of supported dirty components through
  AnnData element encodings and clears only successfully written unchanged
  component mutation tokens;
- no full-AnnData rewrite occurs;
- no event loop or unrelated classifier invalidation;
- annotation coloring follows only relevant current-table, current-region,
  current-obs-column events and never dirties the table merely by styling;
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

The feature is complete when one registered parent Annotation widget lets a
user create, edit, save, and select a polygon annotation through its Shapes
Annotation child, then select a 2D labels element and linked table through its
Spatial Query child; transparently create or reuse validated canonical centers;
run a responsive center-containment query; review affected rows; set a string
annotation in a compatible existing or new obs column or remove annotations
from an existing compatible column; visualize the selected annotation on the
shared primary labels layer; and safely write/reload all supported dirty table
components from zarr without rewriting the complete AnnData object.

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
- overwrite and annotation removal are never silent;
- no stale/cancelled worker updates the cache, opens a dialog, or mutates a table;
- spatial instance identity always uses region_key and instance_key; obs_names
  do not participate in canonical cache identity;
- table rows with no source label are rejected as binding inconsistencies;
- labels absent from the table are not claimed as queryable or counted;
- annotation set/removal and any associated palette update apply and roll back
  as one atomic obs/uns consistency unit;
- annotation coloring reuses the generic compact table-backed labels styling,
  preserves valid stored colors, derives stable shared defaults by category
  position, and never changes an existing category's color merely because a
  category was appended;
- missing or invalid `<column>_colors` state is not mutated merely for viewer
  display, but is stored or repaired with the next effective annotation Apply
  and recorded through the corresponding dirty uns path;
- the latest explicit primary-label coloring action wins across the parent
  Annotation widget's Spatial Query child and Object Classification, and
  unrelated table events do not override it;
- shared cross-widget dirty state and general table events work for obs, obsm,
  and uns;
- canonical cache, annotation columns, and their changed companion palettes
  round-trip through backed zarr;
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
