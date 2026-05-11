# Shapes Annotation Widget

Status: proposed

This roadmap covers shape write-back and annotation editing. It is intentionally
separate from `add_shapes_elements_to_viewer.md`, which should stay focused on
loading, viewing, and coloring existing shapes elements.

## Widget Contract

Add a dedicated widget:

```python
ShapesAnnotation()
```

The widget owns workflows that mutate or create `sdata.shapes[...]` elements.
It should not be folded into the viewer widget because annotation has a
different risk profile: it can create new SpatialData elements, overwrite
existing geometry, and eventually affect table linkage.

The widget must support two primary workflows:

- creating new shapes annotations;
- modifying existing shapes elements.

## Layer Role Contract

`ShapesAnnotation()` must treat primary and styled shapes layers differently:

- primary shapes layers are annotation-capable layers that can be listened to,
  edited, and written back;
- styled shapes layers are viewer-only color variants and must not be used as
  write-back sources;
- when selecting an existing layer to edit, the widget should filter for Harpy
  bindings with `shapes_role="primary"`;
- when creating a new annotation layer, the widget should register it as a
  primary shapes layer.

This keeps visual coloring separate from geometry editing and prevents
accidental write-back from a styled viewer overlay.

## Workflow A: Create New Shapes

This workflow means "create a new shapes annotation element".

User flow:

- the user opens `ShapesAnnotation()`;
- the user selects the active coordinate system;
- the user creates a new empty napari `Shapes` layer from the widget;
- the user draws polygons, circles, or other supported shapes in napari;
- Harpy converts that layer into a new `GeoDataFrame`;
- Harpy writes it into `sdata.shapes[new_name]`.

Recommended semantics:

- save the new geometry in the active coordinate system;
- store an `Identity()` transform to that active coordinate system, because the
  user drew the coordinates directly in that displayed coordinate frame;
- generate a fresh shape index;
- optionally initialize columns such as `annotation_class`, `created_by`,
  `created_at`, or other future annotation metadata;
- assume no existing table linkage;
- if the user wants a table later, make that a separate explicit action.

This is the safer first write-back feature because the user is creating a new
SpatialData element rather than mutating an existing one.

Recommended tests:

- a new napari shapes layer can be saved as `sdata.shapes[new_name]`;
- the new shapes element has an `Identity()` transform to the active coordinate
  system;
- generated indices are unique and stable;
- optional initialized columns are written with the expected length;
- backed `SpatialData` can persist the new element to zarr when supported.

## Workflow B: Modify Existing Shapes

This workflow means "modify an existing `SpatialData` shapes element".

User flow:

- the user opens `ShapesAnnotation()`;
- the user selects an existing `sdata.shapes["..."]` element;
- Harpy loads it into a napari `Shapes` layer in the selected coordinate
  system;
- the user edits geometry in the loaded napari layer;
- Harpy writes the edited geometry back to the existing shapes element, or
  saves it as a copy.

This path needs stricter rules because Harpy must preserve or intentionally
update existing state:

- the existing `GeoDataFrame` index;
- geometry type expectations, such as circles vs polygons vs multipolygons;
- scalar annotation columns such as `leiden` or `in_tumor`;
- companion color columns such as `leiden_colors`;
- any tables whose `region` metadata annotates the shapes element by index;
- coordinate transformations.

Recommended first behavior:

- support geometry-only edits that preserve the same shape count and source
  indices;
- always provide `Save as new shapes element`;
- require explicit confirmation before overwriting the original shapes element;
- block or warn on add/delete when a table annotates the shapes element;
- reject unsupported napari shape types such as lines or paths until there is a
  clear SpatialData representation;
- if inverse-transforming back to the original coordinate frame is not proven
  safe, save the edited copy in the active coordinate system with an
  `Identity()` transform.

Add/delete inside an existing shapes element is a separate problem from vertex
editing. If the user deletes or adds rows to an existing table-linked shapes
element, Harpy needs an explicit policy for table rows and metadata. That should
not happen silently.

Recommended tests:

- editing vertices with unchanged source indices can update the geometry;
- scalar columns and companion `_colors` columns survive geometry-only edits;
- overwrite requires confirmation;
- save-as-copy is available even when overwrite is blocked;
- add/delete is blocked or clearly reported when linked tables annotate the
  shapes element;
- unsupported napari shape types are rejected with actionable feedback.

## Workflow C: Add/Delete Existing Shape Rows

This workflow means "add new rows to, or delete rows from, an existing
`SpatialData` shapes element".

Initial policy:

- allow add/delete only when no table explicitly annotates the edited shapes
  element;
- generate unique indices for newly added rows;
- preserve existing scalar columns and fill new-row values with explicit
  missing values or configured defaults;
- preserve companion color columns without inventing new palette values unless
  the user explicitly edits the annotation column;
- require `Save as new shapes element` when overwrite would be ambiguous.

Table-linked add/delete is intentionally deferred until table reconciliation is
specified.

Recommended tests:

- adding rows to an unlinked shapes element generates unique indices;
- deleting rows from an unlinked shapes element removes the matching geometry
  rows;
- added rows preserve the existing column schema with missing/default values;
- add/delete is blocked when linked tables annotate the shapes element;
- save-as-copy remains available when overwrite is blocked.

## Follow-Up: Shape Table Reconciliation

This should come after annotation creation and strict geometry editing.

This workflow handles add/delete edits when one or more `AnnData` tables
explicitly annotate the edited shapes element.

Open policy questions:

- when a shape is deleted, should matching table rows be deleted, orphaned, or
  moved to an audit table?
- when a shape is added, should Harpy create table rows with missing values?
- how should generated instance ids avoid collisions with existing indices?
- how should these changes be represented in undo/redo or persistence
  feedback?

Until those policies are explicit, table-linked add/delete write-back should
remain out of scope.

## Non-Goals For The First Version

- table reconciliation for add/delete edits on table-linked shapes;
- using labels-linked tables to infer shape identities;
- palette editing or style authoring;
- automatic biological-object identity resolution across labels and shapes;
- silent mutation of existing shapes elements without user confirmation.

## References

- SpatialData object docs:
  https://spatialdata.scverse.org/en/latest/api/SpatialData.html
- SpatialData annotation/table tutorial:
  https://spatialdata.scverse.org/en/latest/tutorials/notebooks/notebooks/examples/tables.html
- SpatialData from-scratch tutorial, connecting `AnnData` to shapes:
  https://spatialdata.scverse.org/en/stable/tutorials/notebooks/notebooks/examples/sdata_from_scratch.html
