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
- allow new napari rows to receive generated IDs, for example
  `__annotation_N`, when they do not yet have a source index value;
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
- edit-existing annotation layer, whether adopted from an already-loaded primary
  viewer layer or loaded by the Annotation widget: remove the dirty primary
  layer, unregister its binding, then reload a clean primary layer from current
  saved `sdata` through the normal viewer adapter load path.

Rationale:

- discard should not leave unsaved edits visible in the viewer;
- remove-and-reload is simpler than in-place revert and reuses the current,
  well-tested viewer adapter path;
- the tradeoff is accepted: the reloaded layer is a new napari layer object and
  may reset presentation state or layer order according to normal viewer loading
  behavior.

Reload requirements for edit-existing discard:

- call `ViewerAdapter.remove_shapes_layer(sdata, shapes_name, coordinate_system)`
  for the dirty primary layer;
- call `ViewerAdapter.ensure_shapes_loaded(sdata, shapes_name, coordinate_system)`
  to load a fresh primary layer from saved `sdata`;
- ensure the old dirty layer no longer has a Harpy binding;
- ensure the fresh clean layer has a primary `ShapesLayerBinding`;
- do not introduce a separate in-place payload helper for this workflow.

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

Editable napari row types:

- during an edit session, the napari layer may contain `polygon`, `rectangle`,
  or `ellipse` rows;
- all supported editable napari rows are converted back to saved Shapely
  `Polygon` geometries;
- this does not relax source eligibility. The source shapes element must still
  start from flat Shapely `Polygon` rows so one source row maps to one editable
  napari row.

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
  for example `__annotation_0`, `__annotation_1`, ...;
- generated IDs must avoid collisions with all existing source index values;
- deleted rows disappear on save;
- duplicate custom/manual IDs remain invalid.

This differs from create-new Workflow A only in where the first row IDs come
from:

- create-new starts with no source rows and generates `__annotation_0`, `__annotation_1`, ...;
- edit-existing starts from the existing shapes element index and preserves it.

Index-name rules:

- preserve the original GeoDataFrame index name when saving the edited element;
- do not blindly use the napari feature column name as the saved GeoDataFrame
  index name;
- track the napari source identity feature column, and derive the GeoDataFrame
  index name from the source snapshot:

  ```python
  source_shapes_index_feature_name: str
  source_geodataframe: gpd.GeoDataFrame
  ```

- `source_shapes_index_feature_name` is the napari `layer.features` column that stores
  row identity for editing and status display;
- `source_geodataframe_index_name` is the name to restore on the saved
  GeoDataFrame index. It should be derived from
  `source_geodataframe.index.name`, including in widget session state if exposed
  as a convenience property, rather than stored as a separate field that can
  drift from the snapshot.

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
layer.features["index"] == ["cell_1", "cell_2", "__annotation_0"]

# Saved edited element:
saved.index.name is None
saved.index == ["cell_1", "cell_2", "__annotation_0"]
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
saved.index == ["cell_1", "cell_2", "__annotation_0"]
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
  `layer.features[source_shapes_index_feature_name]`;
- preserve non-geometry metadata for existing row identities;
- assign generated IDs and missing metadata to newly added rows;
- omit deleted rows from the rebuilt output;
- construct one complete replacement `GeoDataFrame`;
- restore `geodataframe.index.name` from `source_geodataframe.index.name`;
- when `source_geodataframe.index.name is None`, keep the saved GeoDataFrame
  index unnamed. Do not substitute `source_shapes_index_feature_name`;
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
row_ids = resolve_all_row_ids(layer.features[source_shapes_index_feature_name])
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

Status: implemented

Goal: add core helpers that can rebuild an edited shapes `GeoDataFrame` from a
napari `Shapes` layer while preserving existing source row identity and
non-geometry metadata.

Likely files:

- `src/napari_harpy/core/shapes_annotation.py`;
- `tests/test_shapes_annotation.py`.

Work:

- keep one public napari-layer-to-GeoDataFrame conversion helper:
  `napari_shapes_layer_to_geodataframe(...)`;
- add explicit conversion context dataclasses so the helper supports create-new
  and edit-existing without a bloated keyword-only signature:

  ```python
  @dataclass(frozen=True)
  class NewShapesLayerConversion:
      index_name: str = DEFAULT_SHAPES_INDEX_NAME
      index_prefix: str = DEFAULT_SHAPES_INDEX_PREFIX


  @dataclass(frozen=True)
  class ExistingShapesLayerConversion:
      """Conversion context for saving edits to an existing shapes element.

      `source_shapes_index_feature_name` names the napari `layer.features` column that
      stores source row identity. It is intentionally separate from
      `source_geodataframe.index.name`, because unnamed GeoDataFrame indexes are
      stored in napari under a fallback feature column such as `"index"` but
      must still save back with `geodataframe.index.name is None`.
      """

      source_geodataframe: gpd.GeoDataFrame
      source_shapes_index_feature_name: str
      index_prefix: str = DEFAULT_SHAPES_INDEX_PREFIX
  ```

- extend `napari_shapes_layer_to_geodataframe(...)` with one optional
  conversion object instead of adding a second public helper:

  ```python
  def napari_shapes_layer_to_geodataframe(
      layer: Shapes,
      *,
      conversion: NewShapesLayerConversion | ExistingShapesLayerConversion | None = None,
      ellipse_segments: int = DEFAULT_ELLIPSE_SEGMENTS,
  ) -> gpd.GeoDataFrame:
      ...
  ```

- `conversion is None` should behave like
  `NewShapesLayerConversion()` for backwards-compatible create-new behavior;
- in create-new mode, row identity comes from
  `NewShapesLayerConversion.index_name`, new IDs use
  `NewShapesLayerConversion.index_prefix`, and the saved index name is
  `NewShapesLayerConversion.index_name`;
- in edit-existing mode:
  - row identity is read from
    `layer.features[ExistingShapesLayerConversion.source_shapes_index_feature_name]`;
  - source metadata is copied from
    `ExistingShapesLayerConversion.source_geodataframe`;
  - the saved index name is
    `ExistingShapesLayerConversion.source_geodataframe.index.name`, including
    `None`;
  - there is no separate public `source_geodataframe_index_name` argument,
    because that value belongs to the source GeoDataFrame itself;
- add validation for edit-existing source shapes elements:
  - unique source index;
  - source geometry rows are Shapely `Polygon` rows only;
  - no `MultiPolygon`, point-radius rows, empty geometries, unsupported
    geometries, or one-source-row-to-many-rendered-row mappings;
- keep editable napari row support aligned with create-new conversion:
  `polygon`, `rectangle`, and `ellipse` rows can be saved as Shapely
  `Polygon` geometries;
- add a row-identity helper that:
  - reads existing row IDs from `layer.features[source_shapes_index_feature_name]`;
  - preserves existing source index values;
  - assigns generated `__annotation_N` IDs only to new rows with missing row identity;
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

Status: implemented

Goal: add an explicit core save helper for edit-existing rather than overloading
the create-new request contract. The save helper should treat the edited napari
layer as the authoritative replacement for the target shapes element.

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
    source_shapes_index_feature_name: str
    index_prefix: str = DEFAULT_SHAPES_INDEX_PREFIX


@dataclass(frozen=True)
class AnnotateShapesElementResult:
    shapes_name: str
    coordinate_system: str
    row_count: int
```

```python
def edit_shapes_element_from_napari_shapes_layer(
    request: EditShapesElementRequest,
    layer: Shapes,
) -> AnnotateShapesElementResult:
    ...
```

Work:

- validate request-only fields before touching the napari layer;
- require `request.shapes_name` to exist in `request.sdata.shapes`;
- require `request.coordinate_system` to be available for the target shapes
  element;
- call `napari_shapes_layer_to_geodataframe(...)` with
  `ExistingShapesLayerConversion(...)` to rebuild the complete replacement
  GeoDataFrame:

  ```python
  geodataframe = napari_shapes_layer_to_geodataframe(
      layer,
      conversion=ExistingShapesLayerConversion(
          source_geodataframe=request.source_geodataframe,
          source_shapes_index_feature_name=request.source_shapes_index_feature_name,
          index_prefix=request.index_prefix,
      ),
  )
  ```

- write through `harpy.sh.add_shapes(...)` with `overwrite=True`;
- pass `instance_key=geodataframe.index.name`, including `None` for unnamed
  source indexes;
- document that the napari layer geometry is assumed to already be expressed in
  `request.coordinate_system`; this matches the viewer adapter, which transforms
  vector shapes into the selected coordinate system before creating the napari
  layer;
- before overwriting the target shapes element, derive the replacement
  transformations from the original target element:

  ```python
  target_element = request.sdata.shapes[request.shapes_name]
  original_transformations = get_transformation(target_element, get_all=True)
  transformations = {}
  for target_coordinate_system in original_transformations:
      if target_coordinate_system == request.coordinate_system:
          transformations[target_coordinate_system] = Identity()
      else:
          transformations[target_coordinate_system] = (
              get_transformation_between_coordinate_systems(
                  request.sdata,
                  request.coordinate_system,
                  target_coordinate_system,
                  intermediate_coordinate_systems=target_element,
              )
          )
  ```

- pass this full `transformations` dictionary to `harpy.sh.add_shapes(...)`;
- this flattens the edited geometry into `request.coordinate_system` while
  preserving the shapes element's availability in every coordinate system that
  was defined on the original target element;
- if the original element had transforms `E -> global` and `E -> global_micron`,
  and the edit happens in `global`, then the saved geometry is stored in
  `global` coordinates with `global: Identity()`, while `global_micron` is
  recalculated as `global -> E -> global_micron`, i.e. the inverse of the
  original `E -> global` followed by the original `E -> global_micron`;
- if SpatialData cannot derive a transform for one of the original coordinate
  systems, fail before writing rather than silently dropping that coordinate
  system;
- return a result with `shapes_name`, `coordinate_system`, and row count;
- keep conflict detection out of scope: if external code changed the same
  shapes element after the edit session was opened, the widget save still
  overwrites it.

Done when:

- edit-existing can save an edited in-memory shapes element;
- edit-existing can overwrite an existing backed zarr shapes element;
- unnamed source indexes round-trip with `geodataframe.index.name is None`;
- the source row identity feature column is synced back into `layer.features`;
- original coordinate-system availability is preserved by deriving replacement
  transforms before overwrite;
- external concurrent mutations are not checked or merged;
- create-new tests still pass unchanged.

### Slice 3: Viewer Adapter Reload On Discard

Status: implemented

Goal: keep discard handling for adopted existing shapes layers simple by
removing the dirty primary layer and loading a fresh primary shapes layer from
the saved `SpatialData` state.

Likely files:

- `src/napari_harpy/widgets/shapes_annotation/widget.py`;
- `tests/test_shapes_annotation_widget.py`;
- viewer adapter tests only if a small adapter convenience wrapper is added.

Work:

- do not refactor `_build_shapes_layer(...)` or
  `_prepare_napari_shapes_layer_inputs(...)` for this workflow;
- rely on the existing viewer adapter loading path:
  - `ViewerAdapter.remove_shapes_layer(sdata, shapes_name, coordinate_system)`
    removes the dirty primary layer and unregisters its binding;
  - `ViewerAdapter.ensure_shapes_loaded(sdata, shapes_name, coordinate_system)`
    loads a fresh primary layer from the current saved shapes element;
- use this reload path when the Annotation widget has adopted an existing
  primary shapes layer, the user edits it, then chooses another annotation
  target or coordinate system and confirms the discard dialog;
- keep create-new discard behavior separate:
  - unsaved create-new annotation layer: remove the unsaved layer;
  - saved or adopted existing annotation layer: remove and reload from saved
    `sdata`;
- accept that the reloaded layer is a new napari layer object. The simple reload
  path may reset layer presentation and may reinsert the layer according to the
  normal adapter loading behavior;
- ignore styled shapes layers for this workflow. Styled layers are separate
  viewer-only representations and are distinguishable from primary shapes via
  `ShapesLayerBinding.shapes_role`;
- point-radius shapes should not reach edit-existing annotation because source
  eligibility remains polygon-only. If they do appear, the normal open/edit
  validation should reject them before discard/reload logic matters.

The discard clean state is the current saved state in `sdata`. For backed
stores, this means the state visible through the active `SpatialData` object and
its backed store at the time of reload. The Annotation widget is not responsible
for conflict detection or merging external edits here.

Done when:

- `ensure_shapes_loaded(...)` still behaves as before;
- confirming discard for an adopted existing primary shapes layer removes the
  dirty layer and loads a fresh layer from saved `sdata`;
- the old dirty layer has no remaining Harpy binding after discard;
- the fresh layer has a correct primary `ShapesLayerBinding`;
- create-new discard still removes the unsaved annotation layer without trying
  to reload a saved element.

### Slice 4: Annotation Target Selector UI

Status: implemented

Goal: replace the create-new-only name field with a `Shapes` target selector
while preserving the current create-new workflow.

Likely files:

- `src/napari_harpy/widgets/shapes_annotation/widget.py`;
- `tests/test_shapes_annotation_widget.py`.

Work:

- add `_ShapesAnnotationTargetMode` and `_ShapesAnnotationTarget` item data;
- define the target item API concretely in
  `src/napari_harpy/widgets/shapes_annotation/widget.py`:

  ```python
  _ShapesAnnotationTargetMode = Literal["create_new", "edit_existing"]


  @dataclass(frozen=True)
  class _ShapesAnnotationTarget:
      mode: _ShapesAnnotationTargetMode
      existing_shapes_name: str | None = None

      @classmethod
      def create_new(cls) -> _ShapesAnnotationTarget:
          return cls(mode="create_new")

      @classmethod
      def edit_existing(cls, shapes_name: str) -> _ShapesAnnotationTarget:
          return cls(mode="edit_existing", existing_shapes_name=shapes_name)
  ```

- `create_new` targets must have `existing_shapes_name is None`. The new
  element name for create-new lives in the `New shapes name` text field, not in
  the target object;
- `edit_existing` targets must have a non-empty `existing_shapes_name`;
- the combo-box display text and tooltip should be derived from the target when
  populating the combo rather than stored in the target object;
- replace the current visible `Shapes Name` row with:
  - `Shapes` compact combo box;
  - conditional `New shapes name` line edit shown/enabled only when the selected
    target is `Create shapes...`;
- rename the existing create-new text field label from `Shapes Name` to
  `New shapes name`;
- populate the `Shapes` combo whenever `sdata` or the selected coordinate
  system changes:
  - include existing shapes elements available in the selected coordinate
    system;
  - append `Create shapes...`;
  - keep long names compact and expose full names through item tooltips;
  - preserve the previous target selection if it is still valid;
  - otherwise default to `Create shapes...`;
- follow the Feature Extraction widget's `Create table...` pattern;
- expose a single current-target state on the widget so later slices can open
  the selected target without reparsing combo-box text;
- after create-new first save, Slice 6 should update this target state to
  `_ShapesAnnotationTarget.edit_existing(result.shapes_name)`, refresh the
  `Shapes` combo, select the newly saved element, and hide `New shapes name`;
- update readiness rules:
  - create-new target validates the `New shapes name` text exactly as the
    current widget validates `Shapes Name`;
  - edit-existing target validates that the selected shapes element still exists
    and is available in the selected coordinate system;
- keep Slice 4 scoped to selector UI and readiness only. Opening or adopting
  the editable layer for an existing target belongs to Slice 5.

Target switching while an annotation session is active:

- if the selected target changes and there is no active annotation layer, update
  target state immediately;
- if an active annotation layer exists, show the same discard confirmation used
  for coordinate-system changes;
- canceling discard restores the `Shapes` combo to the locked session target and
  leaves the active edit session untouched;
- confirming discard uses the Slice 3 discard behavior, then applies the newly
  selected target;
- target-change discard must use the same scoped removal guard as
  coordinate-system discard so programmatic layer removals do not double-clear
  widget state.

Done when:

- Workflow A still works through `Create shapes...`;
- selecting an existing shapes element updates widget target state to
  edit-existing mode;
- selecting `Create shapes...` shows `New shapes name` and uses the existing
  create-new validation path;
- existing target selection hides/disables `New shapes name`;
- target selection is preserved across refreshes where possible;
- switching the target while an annotation layer is active routes through
  discard confirmation;
- canceling target-change discard restores the previous target selection.

### Slice 5: Edit Session Opening And Adoption

Status: implemented

Goal: create or adopt the editable primary layer for edit-existing sessions.

Likely files:

- `src/napari_harpy/widgets/shapes_annotation/widget.py`;
- `src/napari_harpy/viewer/adapter.py` if a small adapter lookup helper is
  useful;
- `tests/test_shapes_annotation_widget.py`.

Work:

- introduce target-aware opening from the existing action button:
  - `Create shapes...` target keeps the current create-new behavior;
  - edit-existing target opens the selected existing shapes element;
  - button text should be target-aware, for example `Create layer` for
    create-new and `Open layer` for edit-existing;
- add explicit active annotation session state. A concrete shape can be:

  ```python
  _ShapesAnnotationLayerOrigin = Literal[
      "created_by_annotation",
      "loaded_by_annotation",
      "adopted_primary",
  ]

  @dataclass(frozen=True)
  class _ShapesAnnotationSession:
      mode: _ShapesAnnotationTargetMode
      layer_origin: _ShapesAnnotationLayerOrigin
      shapes_name: str
      coordinate_system: str
      source_shapes_index_feature_name: str
      source_geodataframe: gpd.GeoDataFrame | None = None
      table_linked: bool = False

      @property
      def reload_on_discard(self) -> bool:
          return self.layer_origin in {"loaded_by_annotation", "adopted_primary"}

      @property
      def source_geodataframe_index_name(self) -> str | None:
          if self.source_geodataframe is None:
              return None
          return self.source_geodataframe.index.name
  ```

- `_ShapesAnnotationLayerOrigin` records where the active napari layer came
  from:
  - `created_by_annotation`: the Annotation widget created a new empty layer for
    Workflow A;
  - `loaded_by_annotation`: the widget loaded an existing saved shapes element
    that was not already visible;
  - `adopted_primary`: the target shapes element was already visible as a
    compatible primary layer, and the Annotation widget took that layer over for
    editing;
- `layer_origin` records how the edit layer entered the session, while
  `reload_on_discard` exposes the behavior discard needs:
  - `created_by_annotation` -> remove without reload;
  - `loaded_by_annotation` -> remove dirty layer and reload clean saved layer;
  - `adopted_primary` -> remove dirty layer and reload clean saved layer;
- for create-new opening:
  - create the empty primary shapes layer exactly as today through
    `ViewerAdapter.create_empty_primary_shapes_layer(...)`;
  - create a session with `mode="create_new"` and
    `layer_origin="created_by_annotation"`;
  - keep `source_geodataframe is None`;
- for edit-existing opening:
  - validate `sdata`, selected coordinate system, and
    `_ShapesAnnotationTarget.edit_existing(...)`;
  - validate the source shapes element before exposing it as editable:
    - source is a GeoDataFrame-like shapes element;
    - source index is unique and non-missing;
    - source geometries satisfy the Geometry And Identity Scope section;
    - the rendered layer maps one source row to one napari row;
  - reuse the core edit-existing validation rules rather than duplicating them
    in the widget. If needed, expose a small core helper so opening validation
    and save conversion reject the same source shapes elements;
  - call `ViewerAdapter.ensure_shapes_loaded(sdata, shapes_name, coordinate_system)`;
  - if `ShapesLoadResult.created is False`, treat the layer as
    `layer_origin="adopted_primary"`;
  - if `ShapesLoadResult.created is True`, treat the layer as
    `layer_origin="loaded_by_annotation"`;
  - require the returned layer to be a napari `Shapes` layer, not a
    point-radius `Points` compatibility layer;
  - require the Harpy binding to match the selected `sdata`, shapes name,
    coordinate system, primary role, `shapes_rendering_mode == "shapes"`,
    `style_spec is None`, and `skipped_geometry_count == 0`;
  - require `binding.source_row_id_by_rendered_row` to be one-to-one with the
    source GeoDataFrame rows;
  - store `binding.source_shapes_index_feature_name` in the session as
    `source_shapes_index_feature_name`;
  - store a defensive source GeoDataFrame snapshot in the session for Slice 6
    save metadata alignment;
  - store `source_geodataframe.index.name` in the session for clarity, while
    still letting the GeoDataFrame itself remain the source of truth;
  - detect whether the shapes element has annotating tables and store a
    `table_linked` flag so the UI can warn without blocking;
- ignore styled layers:
  - do not adopt styled shapes layers;
  - do not remove or restyle them;
  - if only styled layers exist, `ensure_shapes_loaded(...)` should still load
    a separate primary editable layer;
- activate the editable layer in napari after successful create/open;
- lock the active session target:
  - changing coordinate system or changing `Shapes` target routes through the
    existing discard confirmation;
  - cancel restores the locked combo target;
  - confirm uses Slice 3 discard/reload behavior;
- keep save disabled for edit-existing sessions until Slice 6 wires the
  edit-existing save path. This avoids accidentally using the create-new save
  helper on an existing source element.

Implementation notes:

- `ViewerAdapter.ensure_shapes_loaded(...)` already adopts compatible loaded
  primary layers and ignores styled layers through
  `get_loaded_primary_shapes_layer(...)`;
- rename the current `_on_create_layer_clicked(...)` handler to the
  target-neutral `_on_open_annotation_clicked(...)`. It should dispatch by the
  selected target:

  ```python
  if target.mode == "create_new":
      self._open_create_new_annotation_layer()
  else:
      self._open_existing_annotation_layer()
  ```

  The user-facing button text should also become target-aware;
- update the button text whenever the selected `Shapes` target changes or
  readiness is refreshed:
  - `Create shapes...` target -> `Create layer`;
  - existing shapes target -> `Open layer`;
- status feedback should be actionable:
  - incompatible geometry -> explain that only one-row-to-one-polygon shapes
    can be edited for now;
  - point-radius rendering mode -> explain that circle-like shapes are not
    editable through this workflow yet;
  - table-linked shapes -> warn that edits are allowed but linked tables may
    become out of sync if rows are removed.

Done when:

- existing compatible primary layers are adopted, not duplicated;
- styled layers are never adopted;
- only one annotation layer is active;
- the widget can open an eligible existing polygon-only shapes element;
- edit-existing open activates the editable layer and locks target/coordinate
  system;
- edit-existing save remains disabled until Slice 6.

### Slice 5a: Auto-Open Existing Shapes Target

Status: implemented

Goal: selecting an existing shapes element in the `Shapes` dropdown should open
it immediately as the active annotation layer. The user should not need to
select a shapes element and then click `Open layer`.

This slice supersedes the Slice 5 button behavior for edit-existing targets:

- the action button text should become `Create layer` again;
- `Create layer` is only for the `Create shapes...` branch;
- when the selected `Shapes` target is an existing shapes element, `Create
  layer` is disabled because the layer opens automatically from the dropdown
  selection;
- there should no longer be a user-facing `Open layer` button state.

Target-selection behavior:

- if no annotation layer is active and the user selects an existing shapes
  element:
  - set `_selected_shapes_target` to the selected edit-existing target;
  - immediately call the existing edit-opening path, for example
    `_open_existing_annotation_layer()`;
  - on success, activate the opened/adopted layer and lock the session target
    exactly as Slice 5 already specifies;
  - on failure, show the existing `Could Not Open Shapes` warning, keep no
    active annotation session, keep `Create layer` disabled, and let the user
    choose another target or `Create shapes...`;
- if no annotation layer is active and the user selects `Create shapes...`:
  - show `New shapes name`;
  - enable `Create layer` only when the new name is valid;
  - do not create a layer until the user clicks `Create layer`;
- if an annotation layer is active and the user selects a different existing
  target:
  - use the current discard confirmation behavior before changing targets;
  - cancel restores the previous `Shapes` selection and keeps the active session
    untouched;
  - confirm discards/reloads the current session, selects the requested existing
    target, and immediately opens that target;
- if an annotation layer is active and the user selects `Create shapes...`:
  - use the current discard confirmation behavior before changing targets;
  - cancel restores the previous `Shapes` selection and keeps the active session
    untouched;
  - confirm discards/reloads the current session, switches to create-new mode,
    shows `New shapes name`, and waits for the user to click `Create layer`.

Programmatic refresh behavior:

- combo refreshes and syncs must not accidentally auto-open layers;
- `_refresh_shapes_targets(...)` and `_sync_shapes_target_combo_selection(...)`
  should keep using signal blocking or an equivalent guard;
- after create-new first save, refreshing the `Shapes` selector to the saved
  element should pin the UI selection but must not discard/reopen the active
  layer;
- auto-open should run only from an actual target-selection change that the
  widget is ready to handle.

Status and button behavior:

- existing target selected and opened successfully:
  - status should say the existing shapes layer is opened for editing;
  - `Create layer` disabled;
  - `Save shapes` remains disabled until Slice 6 wires edit-existing save;
- existing target selected but opening failed:
  - status should explain why the target cannot be opened;
  - `Create layer` disabled;
  - `Save shapes` disabled;
- `Create shapes...` selected:
  - status should use the existing create-new readiness messages;
  - `Create layer` enabled only for a valid new shapes name;
  - `Save shapes` disabled until a layer exists and contains saveable shapes.

Done when:

- selecting an eligible existing shapes element opens/adopts the layer without
  pressing a second button;
- compatible already-loaded primary layers are still adopted, not duplicated;
- selecting an ineligible existing shapes element shows a warning and does not
  create an active session;
- selecting `Create shapes...` still requires pressing `Create layer`;
- `Create layer` is never used as an `Open layer` action;
- programmatic target refreshes do not trigger accidental open/discard flows;
- existing Slice 5 tests are updated to expect auto-open behavior.

### Slice 5b: Annotation Layer Snapshot/Fingerprint Helpers

Status: implemented

Goal: implement and test the low-level clean-state snapshot machinery without
changing Annotation widget discard behavior yet.

Do not introduce a dirty-row model. Napari does not provide a stable enough
row-level dirty contract, and the Annotation widget saves the whole layer
anyway. This slice should answer only one question: "does the current
save-relevant layer state match a previously captured clean state?"

Suggested widget state:

```python
@dataclass(frozen=True)
class _ShapesAnnotationLayerSnapshot:
    row_count: int
    geometry_hash: str
    features: pd.DataFrame
```

The snapshot should be a compact fingerprint of save-relevant state, not a full
copy of all geometry and feature data. This avoids keeping a second large copy
of dense annotations in memory while still allowing a robust equality check.
Geometry is the potentially large part, so store it as a hash. The current
viewer adapter keeps `layer.features` small, typically just the source row
identity column, so store a normalized copy of `layer.features` directly for
clarity.

- compute a stable hash from layer values rather than keeping references to
  mutable napari geometry objects;
- include `row_count` so row insertions/deletions are cheap to detect and easy
  to reason about;
- compute `geometry_hash` from every row's ordered napari shape type plus
  vertex array metadata and values. Do not store a separate shape-type hash;
  shape type is part of the geometry fingerprint;
- copy `layer.features`, reindex to `range(row_count)`, and reset its index so
  feature comparison is based on napari row order rather than DataFrame index
  identity;
- tolerate empty layers.

Geometry hash rules:

- for each geometry row, hash an explicit row separator, the shape type, the
  vertex array shape, normalized dtype, and raw contiguous bytes;
- normalize floating arrays to `float64` before hashing so equivalent numeric
  coordinates do not differ only because one array is `float32`;
- normalize integer arrays to `int64` if integer arrays appear;
- do not round floating values. A tiny vertex movement should count as an edit;
- include separators between shape, dtype, and value payloads so concatenated
  strings cannot collide accidentally. For example, hash labelled fields such
  as `b"shape:"`, `b"\0dtype:"`, and `b"\0values:"` so inputs like `"1", "23"`
  cannot be encoded the same way as `"12", "3"`.

Feature snapshot rules:

- store a copied `DataFrame`, not a digest;
- reset the `DataFrame` index to row order before storing it;
- compare with pandas-aware equality, for example
  `current.features.equals(clean.features)`;
- include column order, column names, dtypes, values, and missing values through
  the `DataFrame` itself;
- today this is expected to be small because loaded shapes layers store the
  source row identity feature column. Other source GeoDataFrame metadata is
  preserved from the session's `source_geodataframe`, not from `layer.features`.

The discard check should compare snapshots by value:

```python
current = _capture_annotation_layer_snapshot(layer)
clean = self._annotation_clean_snapshot
has_unsaved_changes = (
    current.row_count != clean.row_count
    or current.geometry_hash != clean.geometry_hash
    or not current.features.equals(clean.features)
)
```

The clean snapshot should include only data that affects persistence:

- napari shape geometry data;
- napari shape types;
- `layer.features`, especially the source row identity feature column.

The snapshot should ignore visual-only layer state that is not persisted
by the Annotation workflow, such as current selection, active mode, edge color,
face color, opacity, and other styling.

Suggested helpers:

Implement these helpers in a small widget-local module, for example:

```text
src/napari_harpy/widgets/shapes_annotation/_snapshot.py
```

Keep them out of `core/shapes_annotation.py`: the snapshot is about napari
layer UI/session state, not SpatialData conversion or persistence. Also avoid
growing `widget.py` with byte-level hashing details.
Use private names inside this private helper module as well, matching the
current Annotation widget style for internal dataclasses and helper functions.

```python
def _capture_annotation_layer_snapshot(layer: Shapes) -> _ShapesAnnotationLayerSnapshot:
    ...


def _annotation_layer_snapshots_equal(
    left: _ShapesAnnotationLayerSnapshot,
    right: _ShapesAnnotationLayerSnapshot,
) -> bool:
    ...
```

`_annotation_layer_snapshots_equal(...)` should compare:

```python
return (
    left.row_count == right.row_count
    and left.geometry_hash == right.geometry_hash
    and left.features.equals(right.features)
)
```

Done when:

- the snapshot helper handles empty layers;
- capturing the same unchanged layer twice produces equal snapshots;
- moving a vertex changes `geometry_hash`;
- changing a row's napari shape type changes `geometry_hash`;
- adding or deleting a row changes `row_count` and/or `geometry_hash`;
- changing the source row identity feature value changes the stored
  `features`;
- changing feature column order, names, dtypes, values, or missing values is
  detected by DataFrame equality;
- changing visual-only state such as selection, mode, edge color, face color,
  opacity, or edge width does not change the snapshot;
- no Annotation widget discard flow changes in this slice.

### Slice 5c: Clean Snapshot For Discard Warnings

Status: implemented

Goal: avoid showing the discard warning when the active annotation layer has no
save-relevant changes, using the Slice 5b snapshot helpers.

Current behavior is intentionally conservative: coordinate-system changes and
`Shapes` target changes ask for discard confirmation whenever
`_annotation_layer is not None`. This protects user edits, but it is annoying
when the layer is still clean, for example:

- a newly created empty annotation layer with no drawn shapes;
- an existing shapes layer opened for editing but not changed;
- a layer immediately after successful save;
- a create-new layer after first save when the selector has switched to the
  saved edit-existing target.

Concrete design:

- track a layer-level clean snapshot of save-relevant state;
- establish the clean snapshot when:
  - a create-new annotation layer is created;
  - an existing shapes layer is opened or adopted;
  - save succeeds;
- before coordinate-system or `Shapes` target discard, compare the current
  layer state with the clean snapshot;
- show the discard dialog only when the current layer differs from the clean
  snapshot;
- if the layer is clean, close the Annotation session silently without using
  the dirty discard/reload path for saved or existing layers.

Napari `Shapes` layers expose `data` and `features` events, so the widget may
also maintain an eager `_annotation_dirty` flag for immediate UI feedback.
However, the discard decision should still be allowed to recompute/compare the
snapshot directly. This keeps the behavior robust if a napari event is missed or
if widget code updates `layer.features` programmatically.

Implementation notes:

- `self._annotation_has_been_saved` must remain the overwrite/ownership flag; it
  should not be reused as the dirty flag;
- add a separate clean-state concept, for example
  `_annotation_clean_snapshot`, `_annotation_dirty`, or both;
- add a widget helper, for example
  `_annotation_layer_has_unsaved_changes()`, that uses the Slice 5b snapshot
  helpers;
- `_annotation_layer_has_unsaved_changes()` should return `False` when there is
  no active layer or no clean snapshot, and should compare the current layer to
  the stored clean snapshot otherwise;
- if an eager `_annotation_dirty` flag is added from `layer.events.data` or
  `layer.events.features`, use it as an optimization/UI hint only. The final
  discard decision should still be allowed to compare the snapshot directly;
- after successful save, replace the clean snapshot with the current layer
  state;
- after create-new first save, the `Shapes` selector still jumps to the saved
  element and the session enters edit-existing semantics, but the snapshot
  should also be refreshed so switching away immediately does not warn;
- target-switch and coordinate-system-switch code should ask:

  ```python
  if self._annotation_layer_has_unsaved_changes():
      show_discard_dialog()
  else:
      close_clean_annotation_session()
  ```

- make discard dialog copy context-aware:
  - coordinate-system switch: mention changing coordinate system;
  - `Shapes` target switch: mention switching annotation target.

Concrete discard-dialog API:

```python
def _confirm_discard_annotation_layer(
    self,
    *,
    context: Literal["coordinate_system", "target"],
) -> bool:
    if context == "coordinate_system":
        lines = ["Changing coordinate system will discard the current unsaved shape annotations."]
    else:
        lines = ["Switching shapes target will discard the current unsaved shape annotations."]
    ...
```

Call sites:

- `_on_coordinate_system_changed(...)` should call
  `_confirm_discard_annotation_layer(context="coordinate_system")`;
- `_on_shapes_target_changed(...)` should call
  `_confirm_discard_annotation_layer(context="target")`.

Clean close behavior:

- clean close is different from dirty discard;
- dirty discard restores the last saved state and may remove/reload a layer;
- clean close releases widget ownership of an already-clean annotation session;
- for a clean unsaved create-new layer, remove the layer because it has no
  corresponding saved `sdata.shapes[...]` element;
- for a clean existing edit session, do not remove or reload the layer. Leave
  the primary shapes layer in the viewer exactly where it is and clear only the
  Annotation widget's session state;
- for a clean create-new layer after first save, treat it like a clean existing
  edit session: leave the saved primary layer in place and clear the widget
  session state;
- leaving clean saved/existing layers in place preserves napari layer order and
  avoids the target-switch layer reordering caused by remove-and-reload;
- after clean close, the next selected target can open or adopt its own layer
  normally.

Coordinate-system switch behavior:

- if the active annotation layer has unsaved changes:
  - show the context-aware discard dialog;
  - cancel restores the previous coordinate-system selector value and keeps the
    active session untouched;
  - confirm uses existing discard/reload behavior, then applies the newly
    selected coordinate system to shared app state;
- if the active annotation layer is clean:
  - do not show a dialog;
  - clean-close the active annotation session;
  - apply the newly selected coordinate system to shared app state.

`Shapes` target switch behavior:

- if the active annotation layer has unsaved changes:
  - show the context-aware discard dialog;
  - cancel restores the previous target selector value and keeps the active
    session untouched;
  - confirm uses existing discard/reload behavior, then selects the newly
    requested target;
- if the active annotation layer is clean:
  - do not show a dialog;
  - clean-close the active annotation session;
  - select the newly requested target.

Clean close still needs to clear widget-owned annotation state. It should not
leave the widget thinking it owns a layer that has been released back to the
viewer, or an unsaved create-new layer that has been removed.

Implementation notes for clean close:

- add a separate helper rather than routing clean sessions through
  `_discard_annotation_layer()`, for example:

  ```python
  def _close_clean_annotation_session(self) -> None:
      ...
  ```

- `_close_clean_annotation_session()` should clear `_annotation_layer`,
  `_annotation_session`, clean snapshot state, save state, and any cached
  annotation name/coordinate-system fields owned by the widget;
- it should remove the layer only for clean unsaved create-new sessions;
- it should not call `ViewerAdapter.remove_shapes_layer(...)` or
  `ViewerAdapter.ensure_shapes_loaded(...)` for clean saved/existing sessions;
- manual layer-removal listeners should keep working: if the user removes the
  layer directly in napari, the widget still clears its session state as it does
  today.

Done when:

- switching away from an empty create-new layer does not show a warning;
- switching away immediately after save does not show a warning;
- switching between clean existing targets does not reorder the previously
  opened clean layer;
- switching away after adding, deleting, or editing shapes still shows a
  warning;
- canceling a dirty discard keeps the current session untouched;
- confirming a dirty discard keeps the existing Slice 3 discard/reload
  behavior;
- visual-only changes do not trigger discard warnings.

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
  shapes index feature name;
- this transition must rebuild or update `_annotation_session` itself so
  `_annotation_session.mode == "edit_existing"`. It is not enough for the
  `Shapes` dropdown target to point at `_ShapesAnnotationTarget.edit_existing`;
- after successful create-new first save, refresh the `Shapes` dropdown, select
  the newly saved shapes element, and hide `New shapes name`;
- remove the temporary `_annotation_reload_on_discard` bridge once create-new
  first save rebuilds the session as a real edit-existing session:
  - discard behavior should derive from `_annotation_session.reload_on_discard`;
  - `created_by_annotation` before first save still discards by removing the
    unsaved layer;
  - after first save, the rebuilt session should have an origin that reloads
    saved `sdata` on discard;
- use edit-existing `overwrite=True` for existing targets and for repeated saves
  after create-new first save;
- emit `ShapesElementWrittenEvent` after both create-new and edit-existing
  saves;
- show a table-linked warning when opening or saving table-linked shapes;
- keep save feedback concise and use shortened identifiers/tooltips for long
  names.

Done when:

- saving an edited existing layer updates `sdata.shapes[shapes_name]`;
- after create-new first save, the active layer remains editable and
  `_annotation_session.mode == "edit_existing"` for subsequent saves;
- new rows, deleted rows, edited geometries, and preserved metadata are visible
  after reload;
- Viewer shapes cards refresh through the existing event path;
- table-linked saves warn but do not block.

### Slice 7: Discard, Target Switching, And Reload

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
- for edit-existing layers, discard removes the dirty primary layer and reloads
  a clean primary layer from saved `sdata`;
- use a scoped guard so programmatic layer removals/reloads do not double-handle
  viewer layer removal callbacks;
- accept the normal viewer adapter reload behavior for layer object identity,
  order, and presentation.

Done when:

- canceling discard leaves the active edit session untouched;
- confirming discard clears or reverts the active session correctly;
- edit-existing discard replaces the dirty layer with a freshly loaded clean
  layer from saved `sdata`;
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

- `ViewerAdapter.ensure_shapes_loaded(...)` and
  `ViewerAdapter.remove_shapes_layer(...)` for reloading saved primary shapes
  layers on discard.
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
