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
  - after a successful first save, the saved element exists in
    `sdata.shapes[...]` and the active session can transition to edit-existing
    semantics for later saves;
  - the `Shapes` dropdown should switch from `Create shapes...` to the newly
    saved shapes element, and `New shapes name` should hide;
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

After a successful create-new first save, the newly created element should be
selected in the `Shapes` dropdown. The existing napari layer remains active and
editable, but the session is now treated as edit-existing for that saved shapes
element.

## Coordinate-System Session Lock

Both create-new and edit-existing annotation sessions should lock the coordinate
system when the editable layer is created or adopted.

An active annotation session has one fixed save target:

```python
(sdata, shapes_name, coordinate_system)
```

For edit-existing, this means opening `cells` in coordinate system `global`
must keep saving back to `cells` in `global` for the lifetime of that edit
session. The session should not silently follow later shared app-state changes
to another coordinate system.

Behavior:

- keep the coordinate-system selector visible and usable;
- if no annotation layer is active, coordinate-system changes follow the normal
  shared app-state behavior;
- once an annotation layer is active, changing coordinate system requires
  discard confirmation;
- canceling the discard restores the selector to the locked session coordinate
  system;
- confirming the discard clears or reverts the active annotation session and
  then applies the newly selected coordinate system;
- save always uses the locked session coordinate system, not the latest global
  app-state value.

Rationale:

- napari layer coordinates only make sense in the coordinate frame selected when
  the layer was created or opened;
- silently changing the save coordinate system would risk writing valid-looking
  geometry to the wrong coordinate frame;
- this matches the create-new workflow and keeps the Annotation widget's save
  target explicit.

## Existing Loaded Primary Layer Rule

If the selected edit target is already loaded in the viewer as a compatible
primary shapes layer, the Annotation widget should adopt that layer as the edit
session layer. It should not create a duplicate primary layer for the same
`sdata`, shapes element, and coordinate system.

Adoption requirements:

- the loaded layer has a Harpy `ShapesLayerBinding`;
- the binding matches the selected `sdata`, shapes name, and coordinate system;
- `binding.shapes_role == "primary"`;
- `binding.shapes_rendering_mode == "shapes"`;
- the layer is a napari `Shapes` layer, not a point-radius `Points` layer;
- `binding.style_spec is None`;
- `binding.skipped_geometry_count == 0`;
- `binding.source_row_id_by_rendered_row` is one-to-one with the current layer
  rows;
- the layer passes the edit-existing geometry and identity validation described
  below.

Behavior:

- do not create a second primary layer for the same edit target;
- store the adopted layer in `_annotation_layer`;
- record that the layer was adopted from a pre-existing viewer primary layer,
  not created by the Annotation widget;
- lock the selected shapes name and coordinate system for the edit session;
- lock the source index feature name and source GeoDataFrame index name for the
  edit session;
- activate the adopted layer in napari;
- save the adopted layer with the edit-existing save path and `overwrite=True`;
- keep styled layers separate. Styled layers may coexist, but they are never
  adopted as edit sources.

Rationale:

- the viewer adapter already treats primary shapes layers as the canonical
  visible layer for a shapes element in one coordinate system;
- creating a second primary layer for the same element would make ownership,
  registry lookup, and save behavior ambiguous;
- adopting a compatible primary layer keeps the viewer and Annotation widget
  pointed at the same editable object.

## Styled Layer Rule

Styled shapes layers should be ignored by the edit-existing workflow.

The viewer registry can distinguish them from editable primary layers through
the `ShapesLayerBinding`:

- primary editable shapes have `binding.shapes_role == "primary"` and
  `binding.style_spec is None`;
- styled shapes have `binding.shapes_role == "styled"` and a non-null
  `binding.style_spec`.

Behavior:

- if only styled layers exist for the selected shapes element, create or load a
  separate primary editable layer for the edit session;
- if both a primary layer and styled layers exist, adopt only the compatible
  primary layer;
- never adopt a styled layer as `_annotation_layer`;
- do not remove, restyle, or otherwise mutate styled layers when opening an
  edit session;
- after save, rely on the existing `ShapesElementWrittenEvent`/viewer refresh
  path to let viewer-owned styled layers refresh if that behavior is needed.

Rationale:

- styled layers are viewer presentation variants, not canonical edit targets;
- styled layers may be table-colored or shape-column-colored and carry
  presentation-specific state that should not become part of annotation save
  semantics;
- keeping styled and primary layers separate avoids ambiguous ownership.

## Discard And Target Switching

Switching coordinate system or switching the `Shapes` target while an annotation
layer is active should require discard confirmation.

Confirmed discard behavior depends on how the edit layer was created:

- create-new annotation layer: remove the layer and unregister its binding;
- edit-existing layer created by the Annotation widget: remove the layer and
  unregister its binding;
- edit-existing layer adopted from an already-loaded primary viewer layer:
  reload the saved shapes element from current `sdata` into the same napari
  layer object.

Rationale:

- discard should not leave unsaved edits visible in the viewer;
- adopted primary layers may have user-visible viewer state such as layer order,
  visibility, opacity, and selection context. Discard should not disrupt that
  state;
- reload-in-place behaves like reverting unsaved edits on the existing layer,
  which is the least surprising behavior for a professional annotation workflow.

Reload-in-place requirements for adopted primary layers:

- keep the same napari layer object;
- keep the same layer position in the viewer;
- keep the existing layer binding object or update it in place;
- preserve presentation state where possible, including layer name, visibility,
  opacity, blending, and selection;
- replace the layer's editable shape state from the saved SpatialData element,
  including `data`, `shape_type`, and `features`;
- rebuild the binding's source-row mapping metadata from the saved element so
  later saves continue to use the correct source identities;
- if the saved element no longer passes edit-existing validation, fail the
  discard with actionable feedback rather than leaving a half-reverted layer.

The implementation should reuse the same conversion logic as the viewer loading
path, but expose it as a helper that prepares napari shapes payloads without
adding a new layer. The Annotation widget can then apply that payload to the
adopted layer in place.

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
- generated IDs for new rows should use the existing create-new convention,
  for example `shape_0`, `shape_1`, ...;
- generated IDs must avoid collisions with all existing source index values;
- deleted rows disappear on save;
- duplicate custom/manual IDs remain invalid.

This differs from create-new Workflow A only in where the first row IDs come
from:

- create-new starts with no source rows and generates `shape_0`, `shape_1`, ...;
- edit-existing starts from the existing shapes element index and preserves it.

Index-name rules:

- preserve the original GeoDataFrame index name when saving the edited element;
- do not blindly use the napari feature column name as the saved GeoDataFrame
  index name;
- track both:

  ```python
  source_index_feature_name: str
  source_geodataframe_index_name: str | None
  ```

- `source_index_feature_name` is the napari `layer.features` column that stores
  row identity for editing and status display;
- `source_geodataframe_index_name` is the name to restore on the saved
  GeoDataFrame index. It can be tracked in widget session state, but the core
  conversion helper should derive it from `source_geodataframe.index.name`
  rather than taking it as a separate public argument.

This distinction matters for unnamed GeoDataFrame indexes. The viewer currently
stores unnamed source indexes in `layer.features["index"]` so napari can display
and track them, but saving the edited element should restore
`geodataframe.index.name is None`, not accidentally rename the index to
`"index"`.

Example:

```python
# Source element before editing:
source.index.name is None
source.index == ["cell_1", "cell_2"]

# Editable napari layer:
layer.features["index"] == ["cell_1", "cell_2"]

# After the user draws one new polygon:
layer.features["index"] == ["cell_1", "cell_2", "shape_0"]

# Saved edited element:
saved.index.name is None
saved.index == ["cell_1", "cell_2", "shape_0"]
```

## Non-Geometry Column Rules

Edit-existing should preserve non-geometry columns from the source shapes
element.

Rules:

- existing rows preserve all non-geometry column values;
- editing a row's geometry must not modify that row's metadata values;
- deleted rows remove both geometry and metadata values;
- newly added rows receive missing values for every copied non-geometry column;
- missing values for newly added rows should use `pd.NA` where possible;
- integer columns that need missing values should be converted to pandas
  nullable integer dtypes, for example `int64` -> `Int64`;
- boolean columns that need missing values should be converted to pandas
  nullable boolean dtype;
- categorical columns should keep their categories and use missing values for
  new rows;
- float columns may use regular `NaN`;
- string/object columns may use `pd.NA`;
- dtype changes must be deliberate and tested. Do not silently cast integer
  metadata columns to float just because a new row has missing metadata.

Example:

```python
# Source element before editing:
source.index.name == "instance_id"
source.index == ["cell_1", "cell_2"]
source["class_id"] == [1, 2]  # int64
source["name"] == ["a", "b"]

# After user edits cell_1 geometry and draws one new polygon:
saved.index == ["cell_1", "cell_2", "shape_0"]
saved["class_id"] == [1, 2, pd.NA]  # nullable Int64, not float64
saved["name"] == ["a", "b", pd.NA]
```

If preserving a source column's dtype is impossible or unsafe, the edit helper
should fail with actionable feedback or document the intentional conversion.

## Save Model

Edit-existing saves should rebuild the full shapes element from the current
napari layer state. They should not attempt partial row patching.

Overwrite policy:

- edit-existing saves always call `harpy.sh.add_shapes(...)` with
  `overwrite=True`;
- the target element already exists by definition, because the user explicitly
  selected it as the edit target;
- using `overwrite=False` would make accepted edit-existing saves fail by
  construction;
- create-new keeps its separate first-save guard with `overwrite=False`.

Conflict policy:

- edit-existing does not perform conflict detection for external mutations
  between opening and saving;
- once a shapes element is opened in the Annotation widget, the active edit
  session is authoritative for that element at save time;
- saving replaces the target `sdata.shapes[...]` element in memory and in the
  backed store through Harpy;
- source metadata is preserved from the snapshot used when the edit session was
  opened, so external metadata changes made after opening may be overwritten;
- external references, such as a notebook-held GeoDataFrame object, may become
  stale after the widget saves;
- users are responsible for coordinating external notebook writes with active
  Annotation edit sessions.

Future conflict detection or merge behavior should be specified as a separate
workflow. It should not change the accepted edit-existing write path from
`overwrite=True`.

Save behavior:

- treat the editable napari layer as the source of truth for the edit session;
- validate every current napari row before writing;
- convert every current `layer.data` row back to Shapely geometry;
- resolve row identity for every current napari row from
  `layer.features[source_index_feature_name]`;
- preserve non-geometry metadata for existing row identities;
- assign generated IDs and missing metadata to newly added rows;
- omit deleted rows from the rebuilt output;
- construct one complete replacement `GeoDataFrame`;
- restore `geodataframe.index.name` from `source_geodataframe.index.name`;
- when `source_geodataframe.index.name is None`, keep the saved GeoDataFrame
  index unnamed. Do not substitute `source_index_feature_name`;
- write the complete replacement through `harpy.sh.add_shapes(...)` with
  `overwrite=True`.

Rationale:

- napari does not provide a stable enough dirty-row contract for this workflow;
- rows can be inserted, deleted, reordered, duplicated, or have copied feature
  values;
- a partial patch would need a separate edit journal or dirty-row model;
- full replacement keeps the save behavior aligned with the existing
  create-new workflow and backed persistence path.

This means that even if the user only edits rows 5 and 7, the save helper still
recalculates Shapely geometries for all rows currently present in the napari
layer. Unchanged rows are expected to round-trip through napari geometry
conversion.

Conceptual save flow:

```python
geometries = convert_all_layer_rows_to_geometry(layer.data, layer.shape_type)
row_ids = resolve_all_row_ids(layer.features[source_index_feature_name])
metadata = align_source_metadata_to_row_ids(source_geodataframe, row_ids)
edited = GeoDataFrame(metadata, geometry=geometries, index=row_ids)
edited.index.name = source_geodataframe.index.name

# `source_geodataframe.index.name` may be None for source elements whose index
# was unnamed. In that case the saved index must remain unnamed.
_ = hp.sh.add_shapes(
    sdata,
    input=edited,
    output_shapes_name=shapes_name,
    transformations={coordinate_system: Identity()},
    instance_key=source_geodataframe.index.name,
    overwrite=True,
)
```

Tests must cover both named and unnamed source indexes so this save path does
not drift from `index.name is None` to `index.name == "index"`.

## Table-Linked Shapes Policy

Edit-existing should allow shapes elements that are annotated by one or more
tables, but it should warn the user that table reconciliation is out of scope
for this workflow.

Rules:

- do not block opening or saving a table-linked shapes element;
- when the selected shapes element has annotating tables, show a warning before
  or during the edit session;
- save only the shapes element;
- do not add, remove, reorder, or modify rows in linked tables;
- do not try to repair table metadata or table instance IDs;
- adding a shape may create a shapes row without a linked table row;
- deleting a shape may leave table rows whose instance IDs no longer exist in
  the shapes element.

Viewer responsibility:

- the Viewer widget already discovers linked tables from SpatialData table
  metadata;
- table-backed shapes styling already performs stricter table-to-shapes
  alignment checks when styling is requested;
- shapes rows without matching table rows can be rendered transparent with
  viewer status feedback;
- table rows whose instance IDs are missing from the shapes element are surfaced
  as viewer styling errors.

Rationale:

- blocking all table-linked shapes edits would make the edit workflow too
  restrictive;
- table reconciliation has workflow-specific semantics and should not be hidden
  inside the Annotation widget's geometry save path;
- keeping Annotation responsible only for shapes geometry and row identity keeps
  the save model understandable.

## Open Specification Questions

No open specification questions remain before implementation planning.

## Implementation Slices

The implementation should be split so each slice keeps the existing create-new
Workflow A usable. Avoid a large widget rewrite that mixes UI, conversion, and
persistence in one step.

### Slice 1: Core Edit Conversion And Metadata Alignment

Status: pending

Goal: add core helpers that can rebuild an edited shapes `GeoDataFrame` from a
napari `Shapes` layer while preserving existing source row identity and
non-geometry metadata.

Likely files:

- `src/napari_harpy/core/shapes_annotation.py`;
- `tests/test_shapes_annotation.py`.

Work:

- keep one public napari-layer-to-GeoDataFrame conversion helper:
  `napari_shapes_layer_to_geodataframe(...)`;
- extend `napari_shapes_layer_to_geodataframe(...)` with optional edit-existing
  inputs instead of adding a second public helper:

  ```python
  def napari_shapes_layer_to_geodataframe(
      layer: Shapes,
      *,
      index_name: str = DEFAULT_SHAPES_INDEX_NAME,
      index_prefix: str = DEFAULT_SHAPES_INDEX_PREFIX,
      source_geodataframe: gpd.GeoDataFrame | None = None,
      source_index_feature_name: str | None = None,
      ellipse_segments: int = DEFAULT_ELLIPSE_SEGMENTS,
  ) -> gpd.GeoDataFrame:
      ...
  ```

- in create-new mode, `source_geodataframe is None` and the helper keeps the
  current behavior: row identity comes from `index_name`, new IDs use
  `index_prefix`, and the saved index name is `index_name`;
- in edit-existing mode, `source_geodataframe` is provided and
  `source_index_feature_name` is required:
  - row identity is read from `layer.features[source_index_feature_name]`;
  - source metadata is copied from `source_geodataframe`;
  - the saved index name is `source_geodataframe.index.name`, including `None`;
  - there is no separate public `source_geodataframe_index_name` argument,
    because that value belongs to the source GeoDataFrame itself;
- add validation for edit-existing source shapes elements:
  - unique source index;
  - supported `Polygon` rows only;
  - no `MultiPolygon`, point-radius rows, empty geometries, unsupported
    geometries, or one-source-row-to-many-rendered-row mappings;
- add a row-identity helper that:
  - reads existing row IDs from `layer.features[source_index_feature_name]`;
  - preserves existing source index values;
  - assigns generated `shape_N` IDs only to new rows with missing row identity;
  - avoids collisions with all existing row IDs;
- add metadata alignment helpers that:
  - preserve non-geometry columns for existing row IDs;
  - drop metadata for deleted rows;
  - fill metadata for new rows with missing values;
  - convert integer and boolean metadata columns to nullable pandas dtypes when
    new missing values are needed;
  - preserve categorical columns where possible;
- set the output GeoDataFrame index name from `source_geodataframe.index.name`
  in edit-existing mode, including `None`;
- keep `napari_shapes_layer_to_geodataframe(...)` behavior for create-new
  unchanged unless edit-existing inputs are provided.

Done when:

- core tests cover named and unnamed source indexes;
- new rows get stable generated IDs without renaming unnamed indexes to
  `"index"`;
- deleted rows disappear;
- integer metadata columns do not silently become float columns;
- unsupported geometry fails before writing.

### Slice 2: Core Edit-Existing Save API

Status: pending

Goal: add an explicit core save helper for edit-existing rather than overloading
the create-new request contract.

Likely files:

- `src/napari_harpy/core/shapes_annotation.py`;
- `tests/test_shapes_annotation.py`.

Suggested API shape:

```python
@dataclass(frozen=True)
class EditShapesElementRequest:
    sdata: SpatialData
    shapes_name: str
    coordinate_system: str
    source_geodataframe: gpd.GeoDataFrame
    source_index_feature_name: str
    index_prefix: str = DEFAULT_SHAPES_INDEX_PREFIX
```

```python
def edit_shapes_element_from_napari_shapes_layer(
    request: EditShapesElementRequest,
    layer: Shapes,
) -> EditShapesElementResult:
    ...
```

Work:

- validate request-only fields before touching the napari layer;
- require `request.shapes_name` to exist in `request.sdata.shapes`;
- always write with `overwrite=True`;
- call `napari_shapes_layer_to_geodataframe(...)` with
  `source_geodataframe=request.source_geodataframe` and
  `source_index_feature_name=request.source_index_feature_name` to rebuild the
  complete replacement GeoDataFrame;
- write through `harpy.sh.add_shapes(...)`;
- return a result with `shapes_name`, `coordinate_system`, and row count;
- keep conflict detection out of scope.

Done when:

- edit-existing can save an edited in-memory shapes element;
- edit-existing can overwrite an existing backed zarr shapes element;
- external concurrent mutations are not checked or merged;
- create-new tests still pass unchanged.

### Slice 3: Viewer Adapter Payload And In-Place Revert Helpers

Status: pending

Goal: expose viewer-adapter helpers that prepare existing shapes as napari layer
payloads without necessarily adding a new layer, then use the same preparation
for load and revert.

Likely files:

- `src/napari_harpy/viewer/adapter.py`;
- viewer adapter tests if present or focused widget tests.

Work:

- extract the current `_prepare_napari_shapes_layer_inputs(...)` /
  `_build_shapes_layer(...)` path into a reusable payload helper;
- keep primary layer loading behavior unchanged;
- add a helper that can apply a saved-shapes payload to an existing napari
  `Shapes` layer in place;
- update or preserve the existing `ShapesLayerBinding` in place after revert;
- preserve layer object identity and viewer layer order during in-place revert;
- ensure the helper exposes:
  - `data`;
  - `shape_type`;
  - `features`;
  - `source_row_id_by_rendered_row`;
  - `source_shapes_index_feature_name`;
  - skipped geometry count;
  - rendering mode.

Done when:

- `ensure_shapes_loaded(...)` still behaves as before;
- an adopted primary shapes layer can be reverted to saved `sdata` state without
  creating a new napari layer object;
- binding metadata after revert matches the saved element.

### Slice 4: Annotation Target Selector UI

Status: pending

Goal: replace the create-new-only name field with a `Shapes` target selector
while preserving the current create-new workflow.

Likely files:

- `src/napari_harpy/widgets/shapes_annotation/widget.py`;
- `tests/test_shapes_annotation_widget.py`.

Work:

- add `_ShapesAnnotationTargetMode` and `_ShapesAnnotationTarget` item data;
- replace the visible `Shapes Name` row with:
  - `Shapes` compact combo box;
  - conditional `New shapes name` line edit;
- populate the combo with existing shapes in the selected coordinate system plus
  `Create shapes...`;
- follow the Feature Extraction widget's `Create table...` pattern;
- preserve selection across refreshes where possible;
- keep long names compact and expose full names through tooltips;
- update create-layer readiness so create-new validates the new name while
  edit-existing validates the selected existing target.

Done when:

- Workflow A still works through `Create shapes...`;
- selecting an existing shapes element enters edit-existing mode;
- switching the target while an annotation layer is active routes through
  discard confirmation.

### Slice 5: Edit Session Opening And Adoption

Status: pending

Goal: create or adopt the editable primary layer for edit-existing sessions.

Likely files:

- `src/napari_harpy/widgets/shapes_annotation/widget.py`;
- `src/napari_harpy/viewer/adapter.py` if a small adapter lookup helper is
  useful;
- `tests/test_shapes_annotation_widget.py`.

Work:

- add explicit annotation session state, for example:
  - mode: create-new or edit-existing;
  - layer origin: created-by-annotation, loaded-by-annotation, or adopted-primary;
  - locked shapes name;
  - locked coordinate system;
  - source index feature name;
  - source GeoDataFrame index name;
  - source metadata snapshot for edit-existing;
  - table-linked warning state;
- when opening an existing target:
  - adopt a compatible loaded primary layer if one exists;
  - ignore styled layers;
  - otherwise load/create one primary editable shapes layer through the viewer
    adapter;
  - reject incompatible targets with actionable status feedback;
- activate the editable layer in napari;
- lock the selected target for the active session.

Done when:

- existing compatible primary layers are adopted, not duplicated;
- styled layers are never adopted;
- only one annotation layer is active;
- the widget can open an eligible existing polygon-only shapes element.

### Slice 6: Edit Save Integration And Feedback

Status: pending

Goal: wire create-new and edit-existing save paths into one Annotation widget
without blurring their overwrite rules.

Likely files:

- `src/napari_harpy/widgets/shapes_annotation/widget.py`;
- `tests/test_shapes_annotation_widget.py`;
- `tests/test_shapes_annotation.py`.

Work:

- route save by active session mode:
  - create-new first save uses `create_shapes_element_from_napari_shapes_layer(...)`;
  - edit-existing uses the new edit-existing core helper;
- keep create-new first save at `overwrite=False`;
- after successful create-new first save, transition the active session to
  edit-existing semantics by storing the saved GeoDataFrame snapshot and source
  index feature name;
- after successful create-new first save, refresh the `Shapes` dropdown, select
  the newly saved shapes element, and hide `New shapes name`;
- use edit-existing `overwrite=True` for existing targets and for repeated saves
  after create-new first save;
- emit `ShapesElementWrittenEvent` after both create-new and edit-existing
  saves;
- show a table-linked warning when opening or saving table-linked shapes;
- keep save feedback concise and use shortened identifiers/tooltips for long
  names.

Done when:

- saving an edited existing layer updates `sdata.shapes[shapes_name]`;
- new rows, deleted rows, edited geometries, and preserved metadata are visible
  after reload;
- Viewer shapes cards refresh through the existing event path;
- table-linked saves warn but do not block.

### Slice 7: Discard, Target Switching, And Revert In Place

Status: pending

Goal: make coordinate-system changes, target changes, and manual layer removal
safe for both create-new and edit-existing sessions.

Likely files:

- `src/napari_harpy/widgets/shapes_annotation/widget.py`;
- `src/napari_harpy/viewer/adapter.py`;
- `tests/test_shapes_annotation_widget.py`.

Work:

- extend discard confirmation to `Shapes` target changes;
- keep the current coordinate-system session lock behavior;
- for create-new layers, discard removes the layer and unregisters it;
- for edit-existing layers loaded by Annotation, discard removes the layer and
  unregisters it;
- for adopted primary layers, discard reloads saved `sdata` geometry into the
  same napari layer object;
- use a scoped guard so programmatic layer removals/reverts do not double-handle
  viewer layer removal callbacks;
- preserve layer order and presentation state during adopted-layer revert.

Done when:

- canceling discard leaves the active edit session untouched;
- confirming discard clears or reverts the active session correctly;
- adopted-layer discard preserves the napari layer object and viewer order;
- manual deletion of the active annotation layer clears widget state.

### Slice 8: Viewer Integration Audit

Status: pending

Goal: verify that edit-existing saves interact correctly with the Viewer widget
and existing loaded layers.

Likely files:

- `src/napari_harpy/widgets/viewer/widget.py`;
- `src/napari_harpy/viewer/adapter.py`;
- `tests/test_viewer_widget.py`;
- `tests/test_shapes_annotation_widget.py`.

Work:

- confirm `ShapesElementWrittenEvent` refreshes the shapes section after
  edit-existing saves;
- confirm the edited primary layer remains registered and usable after save;
- confirm linked-table choices refresh from current `sdata`;
- document or implement behavior for already-loaded styled layers after the
  underlying shapes element changes. The current spec allows styled layers to
  be ignored as edit targets, but loaded styled layers may need explicit refresh
  or user-triggered update after save.

Done when:

- Viewer cards show the edited shapes element after save;
- adding/updating the primary layer from the Viewer does not steal Annotation
  session ownership;
- any styled-layer stale-state behavior is either tested or explicitly deferred.

### Slice 9: Backed Persistence And Regression Coverage

Status: pending

Goal: prove the whole edit-existing workflow survives backed stores and the
edge cases called out in this roadmap.

Likely files:

- `tests/test_shapes_annotation.py`;
- `tests/test_shapes_annotation_widget.py`;
- possibly `tests/test_viewer_widget.py`.

Work:

- write backed zarr tests for edit-existing save and reload;
- test named and unnamed GeoDataFrame indexes;
- test metadata preservation and nullable dtype behavior;
- test added rows, deleted rows, and geometry edits;
- test rejection of unsupported/multipart/empty geometries;
- test table-linked warning behavior without table mutation;
- test adopted primary layer discard/revert in place;
- keep create-new Workflow A regression coverage green.

Done when:

- edit-existing tests pass in memory and backed mode;
- create-new annotation tests still pass;
- viewer refresh tests cover the same-session save path.

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

- Viewer shape-loading conversion code for preparing napari `Shapes` payloads
  from existing SpatialData shapes without necessarily adding a new layer.
- `ViewerAdapter.register_shapes_layer(...)` for registering the editable layer
  as a primary shapes layer.
- `napari_shapes_layer_to_geodataframe(...)` for converting edited layer data
  back to `GeoDataFrame`.
- `create_shapes_element_from_napari_shapes_layer(...)` for create-new first
  save with `overwrite=False`.
- `ShapesElementWrittenEvent` for notifying the Viewer widget after save.
- Existing annotation-layer removal and coordinate-system discard guards from
  `ShapesAnnotation`.

## Non-Goals To Confirm

- Editing styled shapes layers directly.
- Inferring polygon holes from nested polygons.
- Creating or reconciling annotation tables automatically.
- Supporting unsupported napari shape types such as `line` and `path`.
- Manual zarr mutation outside Harpy's supported write APIs.
