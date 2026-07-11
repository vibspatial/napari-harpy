# Spatial Query Annotation

## Status

Specification and implementation plan.

This document specifies a production-quality Spatial Query widget. The widget
uses a polygon annotation to find instances in a labels element and assigns a
user-provided annotation value to the corresponding rows of a linked
`AnnData.obs` column.

The feature is related to, but independent of, object classification. It reuses
the existing table-linkage, shared dirty-state, write-to-zarr, and reload-from-zarr
contracts rather than introducing a second persistence model.

## Product Goal

Let a user bulk-annotate segmented objects by drawing or selecting a spatial
region:

1. select one Shapes element created by, or valid for, the Shapes Annotation
   widget;
2. treat all polygons in that Shapes element as one annotation region;
3. select a labels element in the same `SpatialData` object and coordinate
   system;
4. select a table that annotates that labels element;
5. select an existing text/categorical `.obs` column or configure a new one,
   defaulting to `spatial_annotation`;
6. run an exact polygon query;
7. review the number of affected instances and any overwrites;
8. provide the annotation value, defaulting to the Shapes element name;
9. apply the value to the matching table rows;
10. explicitly write or reload the shared table state when working with a
    backed zarr store.

The workflow must be safe for large, lazy, and multiscale labels elements. It
must not freeze the napari UI, silently overwrite table values, lose edits when
a background result becomes stale, or confuse in-memory table state with
persisted state.

## Product Principles

- **Exact and explainable:** the result must follow a documented spatial
  predicate. A bounding box alone is never an acceptable final query.
- **Preview before mutation:** querying does not mutate the table. The user sees
  affected and overwritten counts before applying.
- **No silent data loss:** overwriting existing values, reloading a dirty table,
  and leaving a dirty dataset are explicit user decisions.
- **One shared table state:** all Harpy widgets see the same in-memory `AnnData`
  and the same per-table dirty marker.
- **Out-of-core by design:** query only the label chunks intersecting the
  annotation bounding box; do not eagerly materialize the full labels array.
- **Deterministic:** identical geometry, labels data, coordinate system, and
  table state produce identical instance IDs and table changes.
- **Accessible and testable:** controls have clear labels/tooltips, all states
  have textual feedback, and domain behavior lives outside Qt code.

## Terminology and Normative Decisions

### Annotation

An **annotation** is one selected SpatialData Shapes element. Every geometry row
in the element contributes to the same annotation region. There is no
per-polygon row selection in this workflow.

For the first production release, the element must satisfy the existing Shapes
Annotation edit-validity contract:

- it is a `GeoDataFrame` with an active geometry column;
- it contains at least one geometry row;
- every geometry is a 2D Shapely `Polygon`;
- every polygon is non-empty, valid, finite, and has positive area;
- polygon holes are supported and remain excluded from the positive annotation
  area.

Rectangles and ellipses drawn in napari are eligible after the Shapes Annotation
save path converts them to valid polygons. Lines, paths, points, circles that
have not been polygonized, `MultiPolygon`, and geometry collections are not
eligible. `MultiPolygon` support can be added later by extending the shared
Shapes Annotation validity contract; the Spatial Query widget must not create a
separate, conflicting definition of a valid annotation.

The effective annotation region is the geometric union of all polygon rows.
Overlapping polygons do not duplicate results. Disjoint polygons are allowed.
Holes remain holes after union according to normal Shapely/OGC geometry rules.

### Labels instance

A **labels instance** is one positive, non-background integer value in the
selected labels raster. Label value `0` is always background and is never
returned. The selected labels element is the authoritative source of instance
membership; the table is used only to resolve returned label values to rows.

### Spatial predicate: any pixel-center overlap

For this feature, an instance is “inside the annotation” when **at least one
pixel belonging to that instance has its pixel center in the effective
annotation region**.

Consequences of this definition:

- an instance that crosses the annotation boundary is included if at least one
  of its pixel centers is inside;
- an instance entirely inside a polygon hole is excluded;
- an instance spanning both a hole and positive annotation area is included;
- exterior polygon boundaries are included by Shapely's `covers` predicate;
- duplicate hits from multiple polygons or chunks collapse to one instance ID.

The UI must describe this as `Any pixel-center overlap` in the Run Query tooltip
and result dialog. Do not describe it as full containment or centroid
containment.

Alternative predicates such as `centroid inside`, `fully contained`, or a
minimum overlap fraction are useful future extensions. The query engine should
accept an internal predicate/strategy value so those modes can be added without
changing the table mutation contract, but they are not part of this feature.

### Source-of-truth rule

The query reads the selected Shapes and labels elements from the current
in-memory `SpatialData` object, not arbitrary similarly named napari layers.

If the Shapes Annotation widget has an open dirty edit session for the selected
Shapes element, Spatial Query must not silently query the last saved geometry.
The action is blocked with guidance to save or discard those shape edits first.
After a successful Shapes Annotation save emits `shapes_element_written`, the
Spatial Query selections refresh and the new in-memory geometry becomes
queryable immediately. This provides one reproducible source of truth without
attempting to interpret an unrelated or partially edited viewer layer.

### Coordinate-system rule

The annotation and labels element must:

- belong to the same `SpatialData` object;
- both be available in the selected coordinate system;
- expose a supported, invertible 2D transformation between the query coordinate
  system and labels intrinsic coordinates.

The annotation union is evaluated in labels intrinsic coordinates. Let
`M_shapes_to_cs` map Shapes intrinsic `(x, y)` coordinates to the selected
coordinate system, and let `M_labels_to_cs` map labels intrinsic `(x, y)`
coordinates to the same coordinate system. The required affine is:

```text
M_shapes_to_labels = inverse(M_labels_to_cs) @ M_shapes_to_cs
```

Apply `M_shapes_to_labels` to the Shapely annotation geometry before evaluating
pixel centers. Matrix conversion must explicitly use `(x, y)` axis order for
Shapely coordinates; napari's usual array-axis `(y, x)` order must not leak into
this calculation. Identity, translation, scale, rotation, and other supported
invertible 2D affine compositions must work. A transform must not be assumed to
be identity merely because both elements are displayed together.

This release supports 2D `x`/`y` labels queries. Labels with a `z` spatial axis
are rejected with a clear message rather than implicitly querying a current
napari slice or the full volume.

### Table binding and row identity

Selectable tables are discovered with `get_annotating_table_names(sdata,
labels_name)`. Before a query can run, the selected binding must pass
`validate_table_binding(...)` and spatial-query-specific instance validation.

The table row for label value `instance_id` is resolved by both linkage keys:

```text
obs[region_key] == selected_labels_name
and
obs[instance_key] == instance_id
```

Matching by `.obs` row order or `.obs_names` is forbidden.

Within the selected labels region, `instance_key` values must be non-missing,
positive, integer-like, and unique. Booleans, fractional numbers, non-finite
numbers, and strings that only happen to look numeric are rejected rather than
silently coerced. Duplicate instance IDs within another region are allowed.

A labels instance may legitimately have no row in the selected table. Such an
instance is reported as `not represented in the table` and is skipped. The
widget never creates new table rows. If the query finds label instances but
none resolve to table rows, it shows a warning and does not open the annotation
value dialog.

## Scope

### In scope

- one loaded `SpatialData` object at a time;
- one selected coordinate system;
- one valid polygon Shapes element, with all rows forming one region;
- one 2D labels element;
- one table linked to the labels element;
- exact any-pixel-center-overlap membership;
- assignment to one existing compatible `.obs` column or one newly created
  column;
- mandatory overwrite disclosure and confirmation;
- async query execution, cancellation, progress/status, and stale-result
  protection;
- per-table shared clean/dirty state;
- write current table state to backed zarr;
- reload current table state from backed zarr with dirty-state protection;
- undo of the most recently applied spatial annotation while its source table
  binding remains valid;
- cross-widget refresh after table mutation, write, reload, or Shapes element
  write.

### Non-goals

- modifying labels pixels or Shapes geometry;
- creating missing table rows or a new linked table;
- querying arbitrary, unregistered napari Shapes layers;
- querying unsaved Shapes Annotation edits;
- per-polygon annotation values within one Shapes element;
- 3D ROI/volume queries or current-z-slice semantics;
- points or shapes as the target instance element;
- centroid, full-containment, or percentage-overlap predicates;
- boolean combinations selected interactively across multiple Shapes elements;
- writing a subsetted SpatialData object;
- automatically writing to zarr after every annotation;
- storing an unbounded operation history or audit log in `table.uns`;
- automatically recoloring labels. Existing viewer color-source controls may
  use the new column after they refresh.

## User Experience

### Dock widget

Add a `Spatial Query` dock widget with the same visual language and status-card
patterns as the current Harpy widgets.

Recommended control order:

1. `Coordinate system` combo.
2. `Annotation shapes` combo.
3. `Labels element` combo.
4. `Linked table` combo.
5. `Target column` mode: `Existing column` or `New column`.
6. Existing-column combo or new-column line edit.
7. `Run Spatial Query` button.
8. Query/action status and progress area with `Cancel` while running.
9. `Undo last annotation` button.
10. `Write Table State` and `Reload Table from zarr` buttons.
11. A persistent clean/dirty table-state status.

The target controls use stable identities, not display strings, to preserve
valid selections across refreshes.

### Selection dependencies

- Coordinate-system choices come from the shared `HarpyAppState` SpatialData.
- Annotation choices contain only Shapes elements available in the selected
  coordinate system and valid under the Shapes Annotation contract.
- Labels choices contain only 2D labels elements available in the same
  coordinate system.
- Table choices contain only tables that declare the selected labels element in
  their SpatialData `TableModel` metadata.
- Existing-column choices contain only compatible text/categorical annotation
  columns and exclude `region_key` and `instance_key`.
- Changing an upstream selection refreshes downstream options, clears pending
  query results/dialogs, and cancels or invalidates active work.
- Preserve a downstream choice if its stable identity remains valid. Otherwise,
  choose the first valid option or show a disabled placeholder and explanation.

The coordinate-system combo participates in the existing shared active
coordinate-system model. A change made here should update `HarpyAppState` and
follow the same layer cleanup/refresh behavior as other coordinate-aware
widgets.

### Target column behavior

If `spatial_annotation` already exists and is compatible, default to `Existing
column` and select it. Otherwise default to `New column` with
`spatial_annotation` prefilled.

A new column name:

- is trimmed;
- must pass SpatialData dataframe-column-name validation;
- must not equal the selected table's `region_key` or `instance_key`;
- must not collide with any existing `.obs` column;
- is not created until the user confirms an apply with at least one updateable
  table row.

An existing target column is compatible when it is one of:

- pandas categorical with string categories;
- pandas `StringDtype`;
- object/string data containing only strings and missing values;
- an all-missing object/string column.

Numeric, boolean, datetime, mixed-object, and non-string categorical columns
are not writable by this feature. They should be omitted or visibly disabled
with explanatory feedback; they must never be converted implicitly. Harpy-owned
classification columns such as integer `user_class` are therefore protected by
the dtype rule.

### Run and result flow

1. User configures a complete valid selection.
2. `Run Spatial Query` becomes enabled.
3. Clicking it captures an immutable request containing stable source
   identities, the coordinate system, target column intent, and a generation
   token.
4. The UI reports progress and offers `Cancel`; selection controls remain
   readable but changes invalidate the active request.
5. The worker returns unique label IDs only. It does not mutate `SpatialData`,
   the table, or Qt state.
6. The controller resolves those IDs to the current linked-table rows and
   revalidates the captured binding.
7. If there are no non-background label instances, show `No instances found in
   the annotation` and make no changes.
8. If instances were found but none have table rows, report the number missing
   from the table and make no changes.
9. Otherwise open the Apply Spatial Annotation dialog.

### Apply Spatial Annotation dialog

The modal dialog contains:

- annotation source name;
- labels element, table, and target column;
- inclusion rule: `Any pixel-center overlap`;
- number of unique labels instances found;
- number of matching table rows that can be updated;
- number of labels instances missing from the table;
- number of matching rows currently missing a value;
- number already equal to the proposed value;
- number with a non-missing different value that would be overwritten;
- a `QLineEdit` labeled `Annotation value`, prefilled with the Shapes element
  name;
- `Apply` and `Cancel` actions.

The annotation value is trimmed and must be a non-empty string. Unicode and
internal spaces are allowed. It is a user-facing category value, not a
SpatialData element key, so SpatialData name restrictions do not apply.

Changing the line edit updates equal/overwrite counts live. If one or more
different non-missing values will be replaced, the dialog shows a prominent
warning and uses explicit action text such as `Overwrite 12 and apply to 35`.
The overwrite warning is mandatory; it is not a preference that can be disabled.

`Cancel` closes the dialog without creating a column, changing the table, or
marking it dirty.

Immediately before apply, revalidate:

- the selected `SpatialData`, Shapes, labels, coordinate system, table, and
  target column still match the captured request;
- the table binding and row identities remain valid;
- the source Shapes generation has not changed;
- the current target-column values used for overwrite counts have not changed.

If only target values changed while the dialog was open, refresh the counts and
require the user to confirm the updated summary. If source geometry, labels,
binding, or selection changed, discard the result and require a new query.

### Successful apply

Applying is one main-thread, all-or-nothing table mutation:

- create the new column only if requested and still absent;
- update only rows for the selected `region_key` and returned `instance_key`
  values;
- leave all other rows, regions, columns, `.obsm`, and `.uns` untouched;
- add a new category to an existing categorical column without discarding its
  category order or ordered flag;
- represent unannotated values as actual missing values (`pd.NA`/categorical
  missing), never as the strings `"nan"`, `"None"`, or an empty string;
- preserve the target column's compatible dtype;
- update each matching row at most once.

A newly created target column is categorical: all non-target rows are missing
and the applied value is its first category.

If every updateable row already has the requested value, report a no-op. Do not
replace the column object, emit a table-write event, create an undo record, or
change clean/dirty state.

After an effective mutation:

- mark the selected table dirty through shared `HarpyAppState` state;
- emit a semantic table-observation-written event containing `sdata`, table
  name, changed column, selected labels region, source, and created/updated
  change kind;
- refresh this widget and any table-column/color-source consumers;
- show a success summary with updated, overwritten, unchanged, and missing-table
  counts;
- enable undo for this operation.

### Undo last annotation

Bulk overwrites warrant an immediate recovery path in addition to reload.

The widget keeps one in-memory undo record for the most recent successful
spatial annotation. The record contains the exact affected row positions or
stable row identity pairs, the prior column values/dtype/category metadata,
whether the column was created by the operation, and whether the table was
already dirty before apply. It also captures the table mutation revision and
persistence revision described below; a dirty boolean alone is not sufficient
to determine the correct state after undo.

Undo is enabled only while the same `SpatialData` object, table object, linkage
metadata, row identities, and target column remain compatible. It:

- restores previous values exactly;
- removes the target column if the operation created it and no later mutation
  has touched it;
- clears the dirty marker only when the table was clean before apply, no other
  mutation occurred afterward, and no write occurred between apply and undo;
- otherwise leaves or marks the table dirty, because the restored in-memory
  state may differ from the latest persisted state;
- emits the same semantic table-observation-written event with source
  `spatial_query_undo`;
- is invalidated by reload, another spatial apply, an external write to the same
  column, table replacement, incompatible row changes, or dataset replacement.

An unrelated later mutation in another column does not have to disable undo,
but it prevents undo from clearing the table's dirty marker. Undo never attempts
to reverse unrelated edits made by another widget.

## Query Engine Contract

### Inputs

Use a UI-independent request/result API, for example:

```python
@dataclass(frozen=True)
class SpatialLabelQueryRequest:
    sdata: SpatialData
    shapes_name: str
    labels_name: str
    coordinate_system: str
    predicate: Literal["any_pixel_center_overlap"]
    generation: int


@dataclass(frozen=True)
class SpatialLabelQueryResult:
    shapes_name: str
    labels_name: str
    coordinate_system: str
    instance_ids: npt.NDArray[np.integer]
    inspected_pixel_count: int
    inspected_chunk_count: int
    generation: int
```

The final types may differ, but the separation is normative: geometry/query
code returns deterministic IDs and diagnostics; table resolution/mutation is a
separate operation.

### Algorithm

1. Validate the source Shapes element with the shared Shapes Annotation helper.
2. Copy/snapshot the polygon geometries for the request.
3. Union all polygons into one effective region.
4. Obtain the Shapes-to-coordinate-system and labels-to-coordinate-system 2D
   affine matrices from SpatialData, calculate
   `inverse(M_labels_to_cs) @ M_shapes_to_cs`, and transform the union into
   labels intrinsic `(x, y)` coordinates.
5. Compute a clipped integer index-space bounding box. If it does not overlap
   the label extent, return an empty result without reading label data.
6. Identify only dask chunks, or bounded tiles for NumPy data, intersecting that
   bounding box.
7. Prepare the transformed geometry once with Shapely for repeated predicates.
8. For each block, construct the intrinsic pixel-center coordinates. For array
   row `r` and column `c`, evaluate Shapely coordinate `(x=c, y=r)`; there is no
   implicit half-pixel offset.
9. Evaluate the complete block mask with vectorized
   `shapely.intersects_xy(region_in_labels, xx, yy)`. For point coordinates,
   intersection with the polygon is equivalent to asking whether the polygon
   covers the point: interior and boundary points are included, while points in
   hole interiors are excluded.
10. Read/inspect label values where that mask is true, collect unique positive
    integer IDs, and drop background `0`.
11. Merge and sort IDs deterministically before returning.

The intended core operation is therefore:

```python
region = shapely.union_all(shapes.geometry)

shapes_to_labels = numpy.linalg.inv(labels_to_coordinate_system) @ shapes_to_coordinate_system
region_in_labels = transform_shapely_geometry(region, shapes_to_labels)
shapely.prepare(region_in_labels)

instance_ids: set[int] = set()
for block_slice in blocks_intersecting(region_in_labels.bounds):
    labels_block = read_labels_block(block_slice)
    yy, xx = pixel_center_grids(block_slice)  # y=row index, x=column index
    inside = shapely.intersects_xy(region_in_labels, xx, yy)
    instance_ids.update(numpy.unique(labels_block[inside & (labels_block > 0)]))

return numpy.asarray(sorted(instance_ids))
```

`transform_shapely_geometry(...)`, `blocks_intersecting(...)`, and
`pixel_center_grids(...)` above are descriptive helper names rather than a
required public API. The affine composition, integer-center convention, and
`intersects_xy()` predicate are normative.

For multiscale labels, query the authoritative highest-resolution/full-detail
level (`scale0`) only. Lower-resolution pyramid levels may omit small instances
or merge labels and must not determine membership.

SpatialData's current polygon query for raster elements may be used to derive a
safe bounding-box crop, but its raster result must not be treated as an exact
polygon mask: the raster query currently returns the polygon's bounding-box
crop. The explicit `intersects_xy()` mask is required before collecting label
IDs.

Do not use `rasterio.features.rasterize()` as the normative membership test.
Its default boundary line-burning behavior is not identical to the documented
pixel-center predicate. It may be useful only as an independently tested
optimization if it can be proven bit-for-bit equivalent for exteriors, holes,
boundaries, transforms, and chunk edges; `intersects_xy()` remains the reference
implementation.

### Performance and execution

- Run data access and spatial computation off the Qt main thread.
- Never pass Qt/napari layer objects into the worker.
- Do not compute the full labels array or a full-size polygon mask.
- Keep peak memory proportional to the intersecting crop/chunk working set and
  returned unique IDs, not to the complete labels image.
- Deduplicate IDs incrementally to avoid storing one entry per selected pixel.
- Support NumPy and dask-backed labels with the same result semantics.
- Report indeterminate progress when total work is unknown; otherwise report
  chunks completed/total.
- Cancellation is cooperative between chunk reads. A cancelled job produces no
  dialog and no mutation.
- Exceptions become actionable status feedback and are logged with technical
  context; they do not leave the widget in a running state.

### Stale-result protection

Every run has a monotonically increasing generation token and captured source
identity. A result is discarded if, while it runs:

- the selected `SpatialData` object changes;
- coordinate system, Shapes, labels, table, or target-column intent changes;
- the source Shapes element is written/replaced;
- the labels element or relevant transform is replaced;
- the table is reloaded/replaced or its linkage metadata changes;
- a newer query starts;
- the widget closes.

Worker completion callbacks perform UI and table work on the main thread only.

## Table Mutation Contract

Use a pure, testable preparation/apply boundary. A preparation result should
contain at least:

- sorted queried instance IDs;
- exact matching table row positions;
- IDs missing from the table;
- current values for those positions;
- missing/equal/overwrite counts for a candidate value;
- whether the target column will be created;
- a binding/table revision fingerprint sufficient for apply-time validation.

Apply must validate the preparation against current state before mutating. If
validation or assignment fails, restore the complete pre-apply target-column
state and leave dirty state/events unchanged. Partial row updates are forbidden.

Large counts in dialogs/statuses should use locale-aware formatting. Missing-ID
previews may show the first few sorted values plus a total, but must not render
an unbounded list in Qt.

## Shared State and Cross-Widget Events

The current `ClassificationTableWrittenEvent` is too specific for this feature.
Introduce a general semantic event, for example `TableObsWrittenEvent`, or
generalize the existing event without breaking consumers. Its payload should
include:

```text
sdata
table_name
columns
regions
change_kind        # created / updated / removed / reloaded as applicable
source             # spatial_query, spatial_query_undo, object_classification, ...
```

Emitting the event marks the table dirty for mutation events. Reload is a
separate state replacement event or an explicitly non-dirty event. Producers do
not call other widgets directly.

To make undo and stale-dialog validation reliable, shared app state should also
maintain monotonically increasing per-table mutation and persistence revisions.
Every accepted in-memory table mutation increments the mutation revision;
successful write and reload operations advance the persistence revision, with
reload also replacing the accepted in-memory baseline. These counters are
session state, not data written to `AnnData`. They supplement, rather than
replace, the user-facing dirty boolean.

Expected consumers include:

- Spatial Query: refresh column choices/counts and invalidate unsafe undo;
- Viewer: refresh linked-table column/color-source options while preserving a
  still-valid selection;
- Object Classification: recompute any state that depends on changed `.obs`
  columns without treating an unrelated annotation column as a classifier
  feature change;
- future table-oriented widgets.

Event handlers must guard against feedback loops using source/identity checks.

## Clean/Dirty and Persistence Semantics

### Shared dirty state

Reuse the existing `HarpyAppState` per-`(id(sdata), table_name)` dirty tracking
and `PersistenceController`. Do not introduce widget-local truth for whether a
table is dirty.

State transitions:

| Action | Table mutation | Dirty-state result |
| --- | --- | --- |
| Bind/select inputs | No | Unchanged |
| Run query | No | Unchanged |
| Query returns no usable rows | No | Unchanged |
| Cancel result dialog | No | Unchanged |
| Apply is a no-op | No | Unchanged |
| Apply creates/changes a column | Yes | Dirty |
| Undo apply | Yes | Clean only when it exactly restores the unchanged persisted baseline; otherwise dirty |
| Successful write | Disk updated | Clean |
| Failed write | Possibly attempted | Remains dirty |
| Successful reload | Memory replaced from disk | Clean |
| Failed/cancelled reload | No accepted replacement | Unchanged |

Dirty status belongs to the whole table. If object classification or another
widget has already modified it, Spatial Query must show it as dirty. Likewise,
writing from Spatial Query persists the current shared table state, not only the
spatial annotation column.

### Write Table State

`Write Table State` is enabled only when the selected table is backed and dirty.
It uses `PersistenceController.write_table_state()`, which writes the complete
current `.obs` plus the existing selected Harpy metadata. Because the whole
`.obs` is written, the spatial annotation column requires no dedicated zarr
writer.

The tooltip and success message must identify the table and resolved store path
and make clear that the current shared table state is being written. A write
failure shows an error and leaves the table dirty.

For unbacked SpatialData, the in-memory annotation workflow remains available,
but write/reload controls are disabled with an explanation that persistence
requires a backed zarr store.

### Reload Table from zarr

Reuse the Object Classification dirty-reload decision contract:

- `Write table state and reload`;
- `Reload table state and discard local edits`;
- `Cancel`.

Reload uses the existing validated in-place replacement of `.obs`, `.obsm`, and
`.uns`. Before replacement, invalidate active queries, open result dialogs, and
undo state. After success:

- refresh table metadata and compatible column choices;
- preserve a target column selection only if it still exists and is compatible;
- rebind other table consumers through semantic events;
- clear dirty state;
- show the source path and success result.

Late worker results created before reload must never write into or open a dialog
for the reloaded table.

### Dirty table when leaving the dataset

Switching between tables may leave each table's shared dirty marker intact and
does not itself discard in-memory changes. Before the application replaces or
closes a `SpatialData` object that has any dirty tables, the shared application
lifecycle must warn the user and offer write/discard/cancel behavior. This is a
cross-widget requirement; implementing only a Spatial Query-local warning would
still permit data loss through another widget.

## Validation and Error States

The primary action remains disabled, with a concise status and full tooltip,
when any of the following applies:

- no SpatialData is loaded;
- no coordinate system is selected;
- no eligible Shapes element is available or selected;
- selected Shapes has unsaved edits in Shapes Annotation;
- Shapes geometry is empty, invalid, unsupported, or cannot be unioned;
- no eligible 2D labels element is available or selected;
- Shapes or labels is unavailable in the chosen coordinate system;
- their required transform is missing, unsupported, non-finite, or non-invertible;
- no linked table exists or is selected;
- table linkage metadata is missing/inconsistent;
- selected-region instance keys are missing, duplicated, or not positive
  integer-like values;
- no target mode/column is valid;
- new column name is empty, invalid, reserved, or collides;
- the selected existing column has an incompatible dtype;
- a query is already running for the same request.

Runtime outcomes are distinguished from configuration errors:

- no label instances in annotation: neutral information;
- instances found but absent from table: warning;
- some instances absent from table: apply remains possible and the omission is
  disclosed;
- overwrite: warning requiring explicit confirmation;
- cancellation or stale result: neutral/cancelled state, no error toast;
- I/O, transform, geometry, or compute failure: error with retry guidance.

Never expose a Python traceback in the widget. Log it for diagnostics and show a
stable user-facing message.

## Accessibility and Interaction Quality

- All controls have labels, accessible names, and keyboard focus order matching
  the visual order.
- Enter in the value dialog applies only when validation passes; Escape cancels.
- Warning meaning is conveyed by text/icon, not color alone.
- Status cards are word-wrapped and copyable where practical.
- Long element/table/column names are elided visually but fully available in a
  tooltip.
- Running/cancelled/succeeded/failed states are announced through textual status.
- Destructive reload and overwrite buttons have explicit action text.
- Modal dialogs have the widget as parent and cannot appear behind napari.

## Architecture

Recommended separation:

```text
core/spatial_query.py
    geometry validation/union
    coordinate transformation
    chunked exact labels query
    request/result types

core/spatial_annotation.py
    target-column validation
    table-row resolution
    preparation/conflict summaries
    atomic apply and undo payloads

widgets/spatial_query/controller.py
    binding validation
    worker lifecycle and generations
    stale-result handling
    apply/undo orchestration

widgets/spatial_query/widget.py
    selectors, dialogs, status/progress, persistence actions

widgets/spatial_query/status_card.py
    pure status-card specification builders
```

Reuse rather than copy:

- `get_annotating_table_names(...)`, `get_table(...)`,
  `get_table_metadata(...)`, and `validate_table_binding(...)`;
- Shapes Annotation geometry validity helpers;
- `HarpyAppState` table dirty tracking and semantic events;
- `PersistenceController` write/reload behavior;
- viewer/app-state coordinate-system selection patterns;
- shared control styles and status-card helpers;
- existing worker cleanup and generation-token patterns.

The widget should not depend on Object Classification widget internals. Any
shared reload dialog or table persistence UI should be extracted into a reusable
component/service rather than imported from one feature widget into another.

## Testing Strategy

### Core geometry/query tests

- one polygon, one included label, and background exclusion;
- multiple disjoint polygons produce a union of IDs;
- overlapping polygons and repeated chunk hits deduplicate IDs;
- polygon hole excludes labels entirely inside it;
- boundary-crossing instance follows any-pixel-center-overlap semantics;
- pixel centers use intrinsic `(x=column, y=row)` integer coordinates without a
  half-pixel shift;
- annotation completely outside label extent reads no chunks and returns empty;
- annotation clipped at label extent;
- identity, translation, scale, rotation, and composed affine transforms;
- missing/non-invertible transform rejection;
- invalid/empty/zero-area/unsupported source geometry rejection;
- NumPy and dask arrays return the same sorted IDs;
- chunk-boundary cases;
- multiscale query uses `scale0` and does not compute the complete pyramid;
- cancellation between chunks;
- no full-array compute in large lazy fixtures.

### Table-domain tests

- row matching by `(region_key, instance_key)`, never row order;
- duplicate IDs across regions allowed; duplicates within selected region
  rejected;
- missing, boolean, fractional, string, and non-finite instance keys rejected;
- IDs absent from table counted and skipped;
- multi-region table updates only the selected labels region;
- new categorical column creation with missing non-target rows;
- existing category addition and dtype/order preservation;
- existing StringDtype and compatible object-column updates;
- numeric/mixed/reserved target columns rejected;
- missing/equal/overwrite counts update when the proposed value changes;
- no-op apply leaves object identity/events/dirty state unchanged;
- exception during apply restores the complete original column;
- undo restores values, dtype/categories, and column absence, and derives dirty
  state correctly across prior dirtiness, unrelated mutations, intervening
  writes, and the unchanged-baseline case;
- undo invalidation after external mutation/reload.

### Controller/async tests

- worker never mutates the table;
- result accepted for unchanged request;
- results dropped after every selection/binding/source invalidation path;
- an older result cannot supersede a newer run;
- cancellation prevents dialog/mutation;
- reload freezes/invalidates pending work;
- worker errors restore usable controls and produce feedback;
- controller cleanup disconnects signals and workers on widget close.

### Widget tests

- dependent combo filtering and stable selection preservation;
- default `spatial_annotation` existing/new behavior;
- Run button enablement and tooltips for every validation blocker;
- dirty Shapes Annotation session blocker;
- result dialog counts, default value, live conflict recount, and validation;
- mandatory overwrite warning and explicit button text;
- cancel/no-result/missing-table-row flows do not dirty or create columns;
- apply/undo status and control states;
- clean/dirty indicator shared with another widget;
- write enabled only for backed dirty table;
- dirty reload write/discard/cancel branches;
- successful reload refreshes columns and invalidates undo;
- unbacked persistence explanation;
- table-observation event refreshes Viewer and relevant widgets without loops;
- keyboard behavior, accessible names, and long-name tooltips.

### Backed-zarr integration tests

- apply changes in-memory `.obs` only until Write Table State;
- write persists a newly created column, values, missing values, and categorical
  metadata;
- reopen/reload observes the persisted state;
- reload discards an unpersisted new column after confirmation;
- write failure retains dirty state;
- reload validation failure preserves the current table and dirty state;
- spatial annotations and object-classification changes coexist in one write;
- late query completion after reload cannot affect the table.

## Observability

Log structured diagnostic context for query start, cancellation, stale drop,
success, failure, apply, undo, write, and reload. Include names/identities,
coordinate system, generation, elapsed time, bounding-box/chunk counts, number of
unique IDs, number of table rows updated/skipped/overwritten, and exception type.
Do not log full annotation value lists, full instance-ID lists, or user data
frames by default.

## Implementation Slices

The slices below are ordered to establish correctness boundaries first while
still ending each slice with testable, integrated behavior. A slice is complete
only when its tests, error handling, and documentation are included; “happy
path implemented” is not sufficient.

### Slice 1: Domain contracts and fixtures

Deliverables:

- request/result/preparation/change-summary dataclasses;
- the normative `any_pixel_center_overlap` predicate identifier;
- shared fixtures covering polygons, holes, transforms, chunked/multiscale
  labels, multi-region tables, missing rows, and compatible/incompatible target
  columns;
- explicit validation helpers for source geometry, 2D labels, transforms,
  labels instance values, and target columns;
- focused unit tests for every validation branch.

Exit criteria:

- public/internal contracts are typed and documented;
- invalid inputs fail before computation or mutation with stable messages;
- no Qt dependency exists in the domain modules.

### Slice 2: Exact out-of-core spatial query engine

Deliverables:

- annotation union and the normative
  `inverse(M_labels_to_cs) @ M_shapes_to_cs` affine implementation;
- clipped bounding-box/chunk planning;
- exact vectorized pixel-center masking with `shapely.intersects_xy()` as the
  reference implementation;
- incremental unique positive label-ID collection;
- NumPy, dask, and multiscale `scale0` support;
- cooperative cancellation and diagnostics;
- geometry/query test matrix, including transformed holes and chunk boundaries.

Exit criteria:

- query output matches an independent eager reference implementation on all
  fixtures;
- no test path relies on bounding-box-only membership;
- large lazy tests prove only intersecting work is computed;
- execution API is safe to call in a worker.

### Slice 3: Atomic table preparation, apply, and undo

Deliverables:

- exact row resolution using region and instance keys;
- missing/equal/overwrite summaries for arbitrary proposed values;
- compatible existing-column mutation and new categorical-column creation;
- apply-time revalidation and rollback on error;
- one-operation undo records and invalidation rules;
- table-domain tests, including multi-region and pre-dirty scenarios.

Exit criteria:

- domain tests prove no partial mutations;
- no-op, cancel-equivalent, apply, overwrite, and undo semantics are distinct;
- unrelated table state is byte/semantic-equivalent before and after apply/undo.

### Slice 4: Widget selection and validation shell

Deliverables:

- registered Spatial Query dock widget and plugin manifest entry;
- coordinate system, Shapes, labels, table, and column controls;
- dependent filtering with stable identity preservation;
- default `spatial_annotation` behavior;
- status cards, tooltips, accessible names, and shared styles;
- Shapes Annotation dirty-session integration/blocker;
- widget tests for all selector and validation states.

Exit criteria:

- Run is enabled only for a complete valid request;
- every disabled state tells the user how to proceed;
- selection refreshes never mutate or dirty a table.

### Slice 5: Async run, progress, result review, and apply

Deliverables:

- worker orchestration with generation tokens, progress, cancel, cleanup, and
  error routing;
- stale-result invalidation for all source/binding changes;
- no-result and missing-table-row outcomes;
- Apply Spatial Annotation dialog with live counts and mandatory overwrite
  warning;
- main-thread revalidation and atomic apply;
- success summaries and undo UI wiring;
- async/controller/dialog tests.

Exit criteria:

- the main UI stays responsive during lazy queries;
- cancel/stale/error paths cannot open a late dialog or mutate a table;
- users always see final affected and overwrite counts before mutation.

### Slice 6: General table events and cross-widget synchronization

Deliverables:

- general `TableObsWrittenEvent` (or backward-compatible generalization);
- shared dirty marking, mutation/persistence revisions, and correct undo dirty
  derivation;
- Viewer refresh of linked table columns/color sources;
- targeted Object Classification refresh/invalidation behavior;
- source guards against event loops;
- multi-widget integration tests.

Exit criteria:

- all widgets observe one current in-memory table state;
- a newly created spatial column becomes available to relevant consumers without
  rescanning/reloading the dataset;
- unrelated classification state is not spuriously retrained or destroyed.

### Slice 7: Shared persistence and reload UX

Deliverables:

- reusable Write Table State and Reload Table from zarr controls/dialog service;
- `PersistenceController` binding from Spatial Query;
- dirty/clean indicator and store-path feedback;
- dirty reload write/discard/cancel behavior;
- query/dialog/undo invalidation around reload;
- backed and unbacked integration tests;
- shared dirty-dataset close/replacement guard, or a separately tracked blocker
  ticket if the application lifecycle owner must land it first.

Exit criteria:

- new and existing spatial annotation columns round-trip through zarr;
- write/reload failures preserve recoverable in-memory state and correct dirty
  status;
- no late query can affect a reloaded table;
- there is no widget-local competing dirty truth.

### Slice 8: Production hardening and release gate

Deliverables:

- representative large-data benchmarks and regression thresholds;
- bounded-memory verification and cancellation-latency checks;
- structured logging/diagnostic coverage;
- accessibility and keyboard pass;
- user documentation with screenshots and exact predicate explanation;
- migration/release notes for the generalized table event;
- full test suite, lint, formatting, and manual napari smoke test on macOS/Linux
  targets supported by the project.

Exit criteria:

- all Definition of Done items below pass;
- no known correctness, data-loss, stale-worker, or persistence defect is
  deferred as polish;
- performance limitations are measured and documented, not guessed.

## Definition of Done

The feature is complete when a user can select a valid stored polygon annotation
and 2D labels element in one coordinate system, run an exact responsive query,
review affected and overwritten rows, apply a string annotation to a compatible
existing or new `.obs` column, undo the last apply, and safely write/reload the
shared table state from zarr.

Completion additionally requires:

- exact holes, transforms, multiscale, chunking, background, and boundary
  behavior covered by tests;
- no mutation before explicit Apply;
- no silent overwrite;
- no full labels-array computation for a bounded lazy query;
- no stale/cancelled worker mutation or late dialog;
- row matching exclusively through `(region_key, instance_key)`;
- correct partial-table coverage reporting;
- atomic apply/rollback and safe undo;
- shared cross-widget dirty state and table events;
- backed/unbacked persistence behavior and dirty reload protection;
- accessible validation/status/error feedback;
- passing repository test, lint, and formatting checks.
