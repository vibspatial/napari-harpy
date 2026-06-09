# Edit Existing Shapes Annotation Elements

Status: proposed

This roadmap will specify the workflow for opening an existing
`sdata.shapes[...]` element as an editable annotation layer, modifying it in
napari, and writing the edits back to the same SpatialData element.

It builds on the create-new workflow documented in `add_new_shapes.md`, but it
has a different risk profile because the target element already exists and may
already be visible in the viewer, backed by zarr, linked to tables, or used by
styled layers.

## Current Annotation Widget State

The current `ShapesAnnotation()` widget implements Workflow A: create a brand
new shapes element from user-drawn napari shapes.

Current behavior:

- exposes one shared coordinate-system selector;
- exposes one shapes-name text input for a new element name;
- validates that the requested name does not already exist before layer
  creation;
- creates an empty primary napari `Shapes` layer through
  `ViewerAdapter.create_empty_primary_shapes_layer(...)`;
- tracks exactly one widget-owned annotation layer with:
  - `_annotation_layer`;
  - `_annotation_shapes_name`;
  - `_annotation_coordinate_system`;
  - `_annotation_has_been_saved`;
- locks the shapes name and coordinate system once the empty annotation layer is
  created;
- saves through `create_shapes_element_from_napari_shapes_layer(...)`;
- uses `overwrite=False` for the first save and `overwrite=True` only after the
  widget has successfully saved its own locked layer once;
- keeps the annotation layer editable after save;
- emits `ShapesElementWrittenEvent` after successful saves so the Viewer widget
  can refresh its shapes section;
- listens for manual annotation-layer removal and clears widget-owned annotation
  state;
- requires confirmation before changing coordinate system when an annotation
  layer is active, because changing coordinate system discards that editable
  layer.

Important current limitations:

- it cannot choose an existing shapes element as the edit target;
- it rejects existing shapes element names during layer creation;
- it does not load existing geometry into an editable annotation layer as an
  edit session;
- it does not distinguish create-new save state from edit-existing save state;
- it does not track whether an existing element is already loaded as a primary
  or styled layer elsewhere in the viewer;
- it does not handle table-linked shapes edit semantics;
- it does not provide conflict detection beyond the create-new first-save
  `overwrite=False` guard.

## Initial Goal

Define an edit-existing workflow that lets a user:

- select an existing shapes element;
- open it as a widget-owned editable primary shapes layer;
- edit, add, or delete supported polygon rows in napari;
- save the edited layer back to the same `sdata.shapes[...]` element;
- preserve stable source index values for unchanged rows;
- write through the same core conversion and Harpy-backed persistence path used
  by Workflow A.

First implementation scope:

- support editable flat polygon rows only;
- require a unique GeoDataFrame index for editable shapes elements;
- preserve the existing GeoDataFrame index values as row identity;
- allow new napari rows to receive generated IDs, for example `shape_N`, when
  they do not yet have a source index value;
- reject shapes elements whose source-to-rendered-row mapping is not one-to-one;
- reject shapes elements that contain `MultiPolygon`, point-radius circle rows,
  empty/skipped geometries, unsupported geometries, or duplicate index values.

## Widget Placement

Edit-existing should live in the current `Annotation` widget.

Rationale:

- create-new and edit-existing share the same user intent: author region
  annotations;
- both workflows need the same shared coordinate-system selector;
- both workflows need one widget-owned editable primary shapes layer;
- both workflows share save feedback, layer-removal handling, discard behavior,
  and `ShapesElementWrittenEvent`;
- keeping both workflows in one widget avoids duplicating layer ownership and
  save-state logic across two annotation widgets.

The widget should expose two explicit modes:

- `Create new`: the existing Workflow A. The user enters a new shapes element
  name, creates an empty layer, and first save uses `overwrite=False`;
- `Edit existing`: the user chooses an existing shapes element in the selected
  coordinate system, opens it as an editable annotation layer, and saves back to
  that same element.

The mode must be explicit in the UI and internal state. The save overwrite rules
are different:

- create-new keeps the current first-save guard:
  - first save uses `overwrite=False`;
  - later saves of the widget-owned locked layer use `overwrite=True`;
- edit-existing has an explicit existing target selected by the user, so save
  can use `overwrite=True` from the start.

## Proposed UI Contract

The Annotation widget should keep one compact form with these top-level fields:

- `Coordinate System`;
- `Shapes`;
- `New shapes name`, visible only when the `Shapes` selector is in create mode.

Replace the current `Shapes Name` text field with a `Shapes` dropdown.

The `Shapes` dropdown should behave like the Feature Extraction widget's
`Table` selector:

- list existing shapes elements available in the selected coordinate system;
- include one create option, for example `Create shapes...`;
- store structured item data rather than inferring mode from display text;
- preserve selection when possible during refreshes;
- use a compact combo box so long shapes names do not widen the widget;
- show the full shapes name in a tooltip when the visible name is shortened.

Suggested internal model:

```python
class _ShapesAnnotationTargetMode(Enum):
    EXISTING = "existing"
    CREATE = "create"


@dataclass(frozen=True)
class _ShapesAnnotationTarget:
    mode: _ShapesAnnotationTargetMode
    shapes_name: str | None = None
```

Selector semantics:

- selecting an existing shapes element enters the `Edit existing` branch;
- selecting `Create shapes...` enters the `Create new` branch;
- in `Create new`, show the `New shapes name` label and line edit;
- in `Edit existing`, hide the `New shapes name` label and line edit;
- in `Create new`, validate the new name against existing
  `sdata.shapes[...]`;
- in `Edit existing`, the selected existing name is already the save target;
- switching the `Shapes` dropdown while an annotation layer is active should use
  the same discard-confirmation pattern as coordinate-system changes.

Suggested labels:

- form label: `Shapes`;
- create-option text: `Create shapes...`;
- conditional create-name label: `New shapes name`;
- create button text can remain `Create layer` for now, but its status feedback
  should make the selected branch clear.

This means the `Shapes` selector becomes the branch point:

- `Create shapes...` + `New shapes name` -> Workflow A / create-new;
- existing shapes element -> edit-existing.

## Geometry And Identity Scope

The first edit-existing implementation should be conservative. It should only
open shapes elements that can round-trip one source `GeoDataFrame` row to one
napari `Shapes` row and back again.

Eligible source geometry:

- Shapely `Polygon` rows;
- Shapely `Polygon` rows with holes, because these are rendered as one napari
  polygon path and remain one source row;
- rows whose GeoDataFrame index is unique.

Rejected source geometry:

- `MultiPolygon` rows;
- geometry collections that expand into multiple polygons;
- point-plus-radius circle rows rendered as napari `Points` or ellipse-like
  shapes;
- empty geometries;
- invalid geometries that cannot be rendered as exactly one editable polygon
  row;
- unsupported geometry types such as `LineString`;
- duplicate GeoDataFrame index values.

Rationale:

- the viewer currently expands a `MultiPolygon` source row into multiple napari
  polygon rows;
- the viewer keeps enough source-row mapping for display and styling, but not
  enough edit semantics to decide how arbitrary add/delete/edit operations
  should be grouped back into multipart geometries;
- the current save conversion writes one saved polygon row per napari row;
- allowing multipart rows in the first edit-existing slice would risk silently
  changing one `MultiPolygon` source row into several independent polygon rows.

Identity rules:

- for existing rows, use the source GeoDataFrame index value already present in
  `layer.features`;
- use the source index feature column tracked by the layer binding, for example
  `binding.source_shapes_index_feature_name`;
- existing rows must keep their source index values across saves;
- new rows drawn during the edit session get generated IDs only if they are
  missing a source index value;
- deleted rows disappear on save;
- duplicate custom/manual IDs remain invalid.

This differs from create-new Workflow A only in where the first row IDs come
from:

- create-new starts with no source rows and generates `shape_0`, `shape_1`, ...;
- edit-existing starts from the existing shapes element index and preserves it.

## Open Specification Questions

We should resolve these together before implementation:

- What happens if the target shapes element is already loaded in the viewer as a
  primary layer?
- What happens if a styled layer exists for the same shapes element?
- Should opening an existing element reuse an existing compatible primary layer
  or create a dedicated editable annotation layer?
- Should edit sessions lock the coordinate system the same way create-new
  sessions do?
- Should saving always use `overwrite=True` once an existing element has been
  explicitly opened for editing?
- Do we need conflict detection for backed stores or external mutations between
  opening and saving?
- Should table-linked shapes edits be blocked, allowed with warnings, or handled
  in a later table-reconciliation workflow?

## Deferred: Multipart Edit Roundtrip

Full multipart editing should be a follow-up workflow.

To safely support `MultiPolygon` edit round-trips later, Harpy likely needs an
explicit part-aware edit model, such as hidden per-row features for:

- source instance ID;
- source row id;
- source part id;
- source geometry kind.

That follow-up must define how to handle:

- editing one part of a `MultiPolygon`;
- deleting one part of a `MultiPolygon`;
- adding a new part to an existing multipart row;
- splitting one multipart row into independent rows;
- merging independent rows into one multipart geometry;
- preserving table linkage for multipart rows.

Until that is specified and tested, edit-existing should reject shapes elements
that do not map one source row to one rendered editable napari row.

## Likely Reusable Pieces

- `ViewerAdapter.ensure_shapes_loaded(...)` or nearby shape-loading code for
  converting existing SpatialData shapes into napari layer data.
- `ViewerAdapter.register_shapes_layer(...)` for registering the editable layer
  as a primary shapes layer.
- `napari_shapes_layer_to_geodataframe(...)` for converting edited layer data
  back to `GeoDataFrame`.
- `create_shapes_element_from_napari_shapes_layer(...)` for write-back with
  `overwrite=True`.
- `ShapesElementWrittenEvent` for notifying the Viewer widget after save.
- Existing annotation-layer removal and coordinate-system discard guards from
  `ShapesAnnotation`.

## Non-Goals To Confirm

- Editing styled shapes layers directly.
- Inferring polygon holes from nested polygons.
- Creating or reconciling annotation tables automatically.
- Supporting unsupported napari shape types such as `line` and `path`.
- Manual zarr mutation outside Harpy's supported write APIs.
