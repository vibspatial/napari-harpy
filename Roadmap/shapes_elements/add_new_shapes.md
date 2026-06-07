# Add New Shapes Annotation Elements

Status: proposed

This roadmap extracts Workflow A from `annotation_shapes.md`: creating a brand
new `sdata.shapes[...]` element from user-drawn napari shapes.

It is intentionally narrower than the full `ShapesAnnotation()` roadmap. This
document does not cover editing existing shapes, overwriting existing
`SpatialData` elements, add/delete reconciliation for table-linked shapes, or
palette/style authoring.

## Goal

Add a create-new write-back workflow for shapes annotation:

- the user opens a dedicated `ShapesAnnotation()` widget;
- the user selects the coordinate system in which the new shapes should be
  drawn;
- the user chooses a new shapes element name;
- Harpy creates an empty napari `Shapes` layer registered as a primary shapes
  layer;
- the user draws supported shapes in napari;
- Harpy converts that layer into a `GeoDataFrame`;
- Harpy writes the new element into `sdata.shapes[new_name]`;
- the new shapes element stores an `Identity()` transform to the selected
  coordinate system.

This is a safe shape write-back workflow because it creates a new SpatialData
element instead of mutating an existing one.

## Non-Goals

- Editing or overwriting an existing `sdata.shapes[...]` element.
- Save-as-copy for an existing shapes element.
- Reconciling shapes rows with existing `AnnData` tables.
- Creating or linking a new annotation table.
- Inferring shape identity from labels-linked tables.
- Palette editing or styled-shapes authoring.
- Supporting every napari shape type.
- Persisting new shapes to backed zarr stores unless the SpatialData write API
  path is proven and tested.

## Current Codebase Fit

Useful existing pieces:

- `HarpyAppState` already owns the shared `sdata` and active coordinate system.
- The viewer and object-classification widgets already expose coordinate-system
  selectors that stay synchronized with `HarpyAppState`.
- `ViewerAdapter` already owns layer registration, lookup, activation, and
  removal.
- `LayerBindingRegistry` already supports `ShapesLayerBinding`.
- `ShapesLayerBinding` already distinguishes `shapes_role="primary"` from
  `shapes_role="styled"`.
- `ViewerAdapter.register_shapes_layer(...)` can register a manually-created
  napari shapes layer as a Harpy primary shapes layer.
- `core.spatialdata.get_coordinate_system_names_from_sdata(...)` and the viewer
  widget already use coordinate-system-oriented discovery.
- Shape viewing already converts SpatialData shapes into napari layers, but the
  reverse path from napari layer to `GeoDataFrame` does not exist yet.

Important current gap:

- current shape code is viewer-oriented. It loads and styles existing shapes;
  it does not convert edited napari data back to SpatialData, does not create
  new `sdata.shapes[...]` entries, and does not persist new shapes elements.

## Widget Contract

Add a dedicated widget contribution:

```python
ShapesAnnotation()
```

Widget controls:

- coordinate system selector;
- new shapes element name input;
- create empty layer button;
- save new shapes element button;
- status/feedback area for validation and write results.

The widget should share `HarpyAppState` with the existing viewer, feature
extraction, and object classification widgets.

The coordinate-system selector should follow the same shared-state pattern as
the viewer and object-classification widgets:

- populate from the coordinate systems available in the current `sdata`;
- initialize from `app_state.coordinate_system` when possible;
- update `app_state.coordinate_system` when the user chooses a different
  coordinate system;
- refresh itself when another widget changes `app_state.coordinate_system`.

The widget should not live inside `ViewerWidget`. The operation mutates
`sdata.shapes[...]`, so it has a different risk profile than viewing and
coloring existing elements.

## User Flow

1. User opens `ShapesAnnotation()`.
2. Widget reads `app_state.sdata` and populates a coordinate-system selector.
3. Widget selects `app_state.coordinate_system` when it is available in the
   selector.
4. User chooses the coordinate system in which the new shapes should be drawn.
   If this differs from `app_state.coordinate_system`, the widget updates shared
   app state.
5. User enters a new shapes element name.
6. User clicks `Create layer`.
7. Harpy validates:
   - `sdata` is loaded;
   - a coordinate system is selected;
   - the new element name is non-empty;
   - the name does not already exist in `sdata.shapes`.
8. Harpy creates an empty napari `Shapes` layer in the selected coordinate
   system.
9. Harpy registers that layer as a primary shapes layer with:
   - `element_type="shapes"`;
   - `element_name=new_name`;
   - `coordinate_system=selected_coordinate_system`;
   - `shapes_role="primary"`;
   - `style_spec=None`.
10. User draws supported shapes.
11. User clicks `Save new shapes element`.
12. Harpy converts the layer to a `GeoDataFrame`.
13. Harpy parses it into a SpatialData shapes model with an `Identity()`
    transform to the selected coordinate system.
14. Harpy writes it into `sdata.shapes[new_name]`.
15. Harpy refreshes relevant UI state and reports success.

If the user changes the coordinate-system selector after creating an annotation
layer with unsaved shapes, the widget should show a warning dialog:

```text
Changing coordinate system will delete the current unsaved shape annotations.
```

If the user cancels, keep the current coordinate-system selection and preserve
the annotation layer. If the user confirms, remove the pending unsaved layer,
unregister its Harpy layer binding, update `app_state.coordinate_system`, and
let the user create a new empty annotation layer in the newly selected
coordinate system.

## Layer Role Rules

The annotation-created layer must be primary:

- it is annotation-capable;
- it is the write-back source for the new element;
- it must not carry a style spec;
- it should be discoverable through bindings with `shapes_role="primary"`.

Styled shapes layers remain viewer-only:

- they must not be accepted as write-back sources;
- they must not be created by this workflow;
- they should remain under the viewer widget's coloring workflows.

After the user creates an empty layer, the widget should lock the new element
name for that pending layer. To use another name, the user should discard the
pending unsaved layer and create a new one.

## Data Semantics

The saved shapes element should be interpreted in the selected display coordinate
system.

Transform rule:

```python
ShapesModel.parse(geodataframe, transformations={coordinate_system: Identity()})
```

Rationale:

- the user drew directly in the displayed coordinate frame;
- no inverse transform is needed for this create-new workflow;
- the saved coordinate values are already in the selected coordinate system.

Index rule:

- generate a fresh index for every saved row;
- default index values should be deterministic and stable for the saved layer,
  for example `shape_0`, `shape_1`, ...;
- the generated index must be unique within the new `GeoDataFrame`;
- the generated index name should be `instance_id`.

Rationale:

- current table-backed shapes styling uses the `GeoDataFrame` index as the
  shape instance identity;
- naming the index `instance_id` makes later explicit table-linking less
  awkward, while still avoiding automatic table creation in this workflow.

Column rule:

- write a geometry column;
- initialize optional annotation metadata columns only when explicitly
  configured by the widget or future settings;
- do not create table-linkage columns by default;
- if a `radius` column is needed for circles, polygon rows should receive
  missing radius values.

Table rule:

- assume no existing table linkage;
- do not create an `AnnData` table automatically;
- creating/linking a table should be a separate explicit workflow later.

## Supported Shape Types

Supported napari shape types:

- `polygon` as Shapely `Polygon`;
- `rectangle` as Shapely `Polygon`, if napari exposes it as a shape type;
- circular `ellipse` as Shapely `Point` plus a positive `radius` column.

Polygon row semantics:

- each drawn napari polygon row is saved as one independent Shapely `Polygon`
  row;
- if the user draws one polygon inside another, save both as independent
  polygons;
- do not infer polygon holes from nested polygons.

Unsupported napari shape types:

- `line`;
- `path`;
- non-circular ellipses, unless Harpy explicitly converts them to polygonal
  approximations;
- shapes with too few vertices for a valid polygon;
- empty rows;
- rows containing non-finite coordinates.

Circle handling:

- napari circles are represented as `ellipse` rows;
- accept an ellipse as a circle only when its displayed width and height are
  equal within a small numeric tolerance;
- store the center as `Point(x, y)`;
- store the radius in display-coordinate units in a `radius` column.

Coordinate order:

- napari shape vertices are in `(y, x)` order;
- Shapely geometries are in `(x, y)` order;
- conversion helpers must perform this swap explicitly.

Geometry validity:

- polygon rings should be closed before constructing a `Polygon`;
- invalid polygons should be repaired only when the repair is predictable and
  tested;
- otherwise report an actionable validation error.

## Follow-Up: Polygon Hole Authoring

Status: proposed

Polygon holes should only be supported through an explicit UI action, such as
`Subtract selected polygon as hole`. The create-new workflow should save nested
polygons as independent polygons unless the user runs that explicit action.

Suggested user flow:

- user draws an outer polygon;
- user draws one or more inner polygons;
- user selects the outer polygon and inner polygon(s);
- user runs `Subtract selected polygon as hole`;
- Harpy replaces the selected outer polygon with one Shapely `Polygon` that has
  interior rings;
- Harpy removes or marks the subtracted inner polygons according to a clear UI
  policy.

Rules:

- never infer holes from containment alone;
- require an explicit user command;
- reject invalid holes that touch or cross the exterior boundary;
- reject holes that overlap each other unless a robust union policy is defined;
- preserve the outer polygon's `instance_id`;
- do not create table-linkage changes implicitly.

Recommended tests:

- subtracting one selected inner polygon creates a Shapely polygon with one
  interior ring;
- the outer polygon keeps its `instance_id`;
- the inner polygon row is removed or handled according to the chosen UI policy;
- nested polygons are saved independently when the subtract action is not used;
- invalid hole candidates produce actionable feedback.

## Proposed Core Helpers

Add a focused module, for example:

```text
src/napari_harpy/core/shapes_annotation.py
```

Suggested models:

```python
@dataclass(frozen=True)
class NewShapesElementRequest:
    sdata: SpatialData
    shapes_name: str
    coordinate_system: str
    index_name: str = "instance_id"
    index_prefix: str = "shape"


@dataclass(frozen=True)
class NewShapesElementResult:
    shapes_name: str
    coordinate_system: str
    row_count: int
```

Suggested helpers:

```python
def napari_shapes_layer_to_geodataframe(
    layer: Shapes,
    *,
    index_name: str = "instance_id",
    index_prefix: str = "shape",
) -> gpd.GeoDataFrame:
    ...


def save_new_shapes_element(
    request: NewShapesElementRequest,
    layer: Shapes,
) -> NewShapesElementResult:
    ...
```

Rules for `save_new_shapes_element(...)`:

- reject missing `sdata`;
- reject missing or unknown coordinate system;
- reject empty or duplicate shapes element names;
- reject empty layers;
- validate all napari rows before mutating `sdata`;
- parse the `GeoDataFrame` with `ShapesModel.parse(...)`;
- write only after validation succeeds;
- return a small result object for widget feedback.

## Widget Implementation Notes

Suggested package:

```text
src/napari_harpy/widgets/shapes_annotation/
```

Suggested files:

- `__init__.py`
- `widget.py`

Widget behavior:

- use `get_or_create_app_state(viewer)`;
- listen to `sdata_changed`;
- listen to `coordinate_system_changed`;
- expose a coordinate-system combo box populated from the current `sdata`;
- keep the combo box synchronized with `app_state.coordinate_system`;
- call `app_state.set_coordinate_system(...)` when the user selects a different
  coordinate system;
- when a pending unsaved annotation layer exists, confirm before changing
  coordinate system and discarding that layer;
- disable create/save controls when `sdata` or coordinate system is missing;
- validate the new name as the user types;
- create an empty napari `Shapes` layer only after a valid name is available;
- activate the created layer;
- save only from the widget-owned primary layer;
- leave styled viewer layers untouched.

Plugin registration:

- add a `napari-harpy.shapes_annotation` command to `napari.yaml`;
- expose the widget as `Shapes Annotation`;
- optionally add an `Interactive(..., widgets="shapes_annotation")` selection
  id once the widget is available.

## Persistence Policy

In-memory behavior:

- write the new shapes element into the in-memory `SpatialData` object;
- if `sdata` is backed, report that the new shapes element is currently
  in-memory unless a tested backed-write path is available.

Backed-store behavior:

- add explicit backed zarr persistence for new shapes elements;
- use SpatialData's supported write API rather than manual zarr mutation;
- add tests that write a backed store, save a new shapes element, reload, and
  confirm geometry plus transformations survive.

## Error Feedback

Use short, actionable feedback messages:

- "Load a SpatialData object before creating shapes."
- "Select a coordinate system before creating shapes."
- "Changing coordinate system will delete the current unsaved shape annotations."
- "Choose a new shapes element name."
- "Shapes element `<name>` already exists. Choose another name."
- "Draw at least one supported shape before saving."
- "Lines and paths cannot be saved as SpatialData shapes yet."
- "Only circular ellipses can be saved as radius-backed circles."
- "Shape row `<index>` contains non-finite coordinates."
- "Saved `<name>` with `<n>` shape(s) in coordinate system `<cs>`."

## Implementation Slices

Build these slices in order. Each slice should leave the project in a tested,
usable state and should avoid taking dependencies on later slices.

### Slice 1: Core Conversion

Status: proposed

Purpose:

- add the core module and models for this workflow;
- convert a napari `Shapes` layer into a valid `GeoDataFrame`;
- keep this slice free of `SpatialData` mutation and widget code.

Code:

- add `src/napari_harpy/core/shapes_annotation.py`;
- add `NewShapesElementRequest`;
- add `NewShapesElementResult`;
- add `napari_shapes_layer_to_geodataframe(...)`;
- add focused tests in `tests/test_shapes_annotation.py`.

Behavior:

- convert napari `polygon` rows into Shapely `Polygon` rows;
- convert napari `rectangle` rows into Shapely `Polygon` rows, if represented
  as rectangles by napari;
- convert circular napari `ellipse` rows into Shapely `Point` rows plus a
  `radius` column;
- reject `line` and `path` rows;
- reject non-circular ellipses;
- reject empty layers, empty rows, rows with too few vertices, and rows with
  non-finite coordinates;
- save each drawn polygon row as an independent polygon;
- do not infer holes from nested polygons;
- generate deterministic indices: `shape_0`, `shape_1`, ...;
- name the generated index `instance_id`;
- swap napari `(y, x)` vertices to Shapely `(x, y)` coordinates.

Acceptance:

- supported napari shape rows convert to a valid `GeoDataFrame`;
- unsupported rows raise clear `ValueError`s with actionable messages;
- mixed polygons and circles preserve expected geometry rows and `radius`
  values;
- generated indices are unique, stable, and named `instance_id`;
- no `SpatialData` mutation happens in this slice.

Tests:

- polygon conversion;
- rectangle conversion, if the napari representation is available in tests;
- circular ellipse conversion;
- non-circular ellipse rejection;
- line and path rejection;
- empty layer rejection;
- non-finite coordinate rejection;
- index naming and generation;
- coordinate order conversion;
- nested polygons remain independent rows.

### Slice 2: Core Save Into In-Memory SpatialData

Status: proposed

Purpose:

- add the write-back core for creating a new in-memory `sdata.shapes[...]`
  element;
- validate everything before mutating `sdata`.

Code:

- add `save_new_shapes_element(...)` in
  `src/napari_harpy/core/shapes_annotation.py`;
- reuse `napari_shapes_layer_to_geodataframe(...)`;
- use `ShapesModel.parse(...)` with an `Identity()` transform;
- extend `tests/test_shapes_annotation.py`.

Behavior:

- reject missing `sdata`;
- reject missing or unknown coordinate system;
- reject empty or duplicate shapes element names;
- reject empty layers;
- validate all napari rows before mutating `sdata`;
- parse the `GeoDataFrame` with:

  ```python
  ShapesModel.parse(geodataframe, transformations={coordinate_system: Identity()})
  ```

- write the parsed element into `sdata.shapes[new_name]`;
- return `NewShapesElementResult` with name, coordinate system, and row count;
- do not create or link an `AnnData` table;
- do not persist to a backed zarr store in this slice.

Acceptance:

- validates request and layer;
- rejects duplicate names before mutation;
- rejects validation failures before mutation;
- writes `sdata.shapes[new_name]`;
- stores `Identity()` to the selected coordinate system;
- preserves the generated row index;
- returns row-count feedback.

Tests:

- saving writes `sdata.shapes[new_name]`;
- duplicate names are rejected before mutation;
- missing `sdata` is rejected;
- missing coordinate system is rejected;
- unknown coordinate system is rejected;
- failed conversion does not mutate `sdata.shapes`;
- the new element has an `Identity()` transform to the selected coordinate
  system;
- no annotation table is created implicitly.

### Slice 3: Widget Shell And Shared Coordinate System

Status: proposed

Purpose:

- add the dedicated widget shell;
- make coordinate-system selection match existing shared app-state behavior;
- prepare validation and feedback without creating layers yet.

Code:

- add `src/napari_harpy/widgets/shapes_annotation/__init__.py`;
- add `src/napari_harpy/widgets/shapes_annotation/widget.py`;
- expose `ShapesAnnotation`;
- add widget tests, likely in `tests/test_shapes_annotation_widget.py`.

Behavior:

- use `get_or_create_app_state(viewer)`;
- listen to `sdata_changed`;
- listen to `coordinate_system_changed`;
- populate the coordinate-system selector from loaded `sdata`;
- initialize the selector from `app_state.coordinate_system` when available;
- update `app_state.coordinate_system` when the user chooses a different
  coordinate system;
- refresh when another widget changes `app_state.coordinate_system`;
- validate the new element name as the user types;
- disable create/save controls when `sdata` or coordinate system is missing;
- show clear status text for missing `sdata`, missing coordinate system, empty
  names, and duplicate names.

Acceptance:

- widget shares app state;
- coordinate-system controls reflect loaded `sdata`;
- coordinate-system controls reflect shared app-state changes;
- user coordinate-system selection updates shared app state;
- create/save controls reflect `sdata`, coordinate-system, and name validity;
- no napari layer is created in this slice.

Tests:

- widget shares `HarpyAppState` with the viewer;
- controls disable when no `sdata` is loaded;
- controls disable when no coordinate system is selected;
- coordinate-system selector is populated from loaded `sdata`;
- coordinate-system selector initializes from `app_state.coordinate_system`;
- user coordinate-system selection updates `app_state.coordinate_system`;
- external `coordinate_system_changed` events update the selector;
- duplicate-name validation is shown before layer creation.

### Slice 4: Empty Layer Lifecycle

Status: proposed

Purpose:

- create and track the widget-owned pending annotation layer;
- register the layer as a primary Harpy shapes layer;
- handle coordinate-system changes that would discard unsaved annotations.

Code:

- extend `ShapesAnnotation` layer creation behavior;
- use `ViewerAdapter.register_shapes_layer(...)`;
- use existing viewer adapter layer removal/unregistration APIs where possible;
- extend widget tests.

Behavior:

- create an empty napari `Shapes` layer only after the name, `sdata`, and
  coordinate system are valid;
- register the layer as:
  - `element_type="shapes"`;
  - `element_name=new_name`;
  - `coordinate_system=selected_coordinate_system`;
  - `shapes_role="primary"`;
  - `style_spec=None`;
- activate the created layer;
- lock the new element name while the pending layer exists;
- track the layer as widget-owned;
- leave viewer-created styled shapes layers untouched;
- when the user changes coordinate system with a pending unsaved layer, show:

  ```text
  Changing coordinate system will delete the current unsaved shape annotations.
  ```

- cancelling the warning keeps the current coordinate-system selection and
  preserves the pending layer;
- confirming the warning removes the pending layer, unregisters its Harpy
  binding, updates `app_state.coordinate_system`, and lets the user create a new
  layer.

Acceptance:

- new layer is created, registered as primary, and activated;
- element name is locked after layer creation;
- changing coordinate system after layer creation requires confirmation and
  discards the pending unsaved layer;
- styled shapes layers are not modified by this workflow.

Tests:

- create layer registers a primary shapes binding;
- create layer activates the new layer;
- create layer locks the name input;
- cancelling the coordinate-system warning preserves the pending layer and
  keeps the previous coordinate-system selection;
- confirming the coordinate-system warning removes the pending layer,
  unregisters its Harpy binding, and updates shared app state;
- styled shapes layers are not removed or re-registered by this workflow.

### Slice 5: Save Button And Feedback

Status: proposed

Purpose:

- connect the widget-owned pending layer to the core save helper;
- report save results and validation failures in the widget.

Code:

- connect the `Save new shapes element` button;
- call `save_new_shapes_element(...)`;
- refresh relevant widget/viewer state after success;
- extend widget tests.

Behavior:

- save only from the widget-owned primary layer;
- reject saving if there is no pending layer;
- reject saving if the pending layer binding no longer matches the widget
  request;
- pass the selected coordinate system and locked element name to the core save
  helper;
- show actionable conversion/save errors in the status area;
- show:

  ```text
  Saved `<name>` with `<n>` shape(s) in coordinate system `<cs>`.
  ```

- after successful save, keep the newly saved element registered as the source
  element for the layer or refresh the viewer state by a clearly documented
  policy;
- do not accept styled shapes layers as save sources.

Acceptance:

- save validates the widget-owned primary layer;
- successful save writes the new element;
- successful save reports row count and coordinate system;
- errors are shown as clear widget feedback;
- no styled layer can be saved by this workflow.

Tests:

- save calls the core save helper with the selected coordinate system;
- save passes the locked new shapes element name;
- save feedback reports row count and coordinate system;
- duplicate-name validation is still enforced before save;
- unsupported shape errors are displayed without mutating `sdata`;
- styled shapes layers are not offered or accepted as save sources.

### Slice 6: Plugin And Launcher Wiring

Status: proposed

Expose the widget through napari plugin metadata and, if desired, through
`Interactive`.

Code:

- add a `napari-harpy.shapes_annotation` command to
  `src/napari_harpy/napari.yaml`;
- expose the widget as `Shapes Annotation`;
- update lazy widget exports in `src/napari_harpy/widgets/__init__.py`;
- optionally add an `Interactive(..., widgets="shapes_annotation")` selection
  id once the widget is available.

Acceptance:

- `Shapes Annotation` appears as a napari widget contribution;
- lazy widget exports include `ShapesAnnotation`;
- launcher selection tests cover the new widget id if added;
- `Interactive(..., widgets="all")` behavior is updated intentionally, either
  including or excluding the annotation widget by documented choice.

Tests:

- `napari.yaml` contributes `Shapes Annotation`;
- package lazy exports include `ShapesAnnotation`;
- `Interactive(..., widgets="shapes_annotation")` docks the widget if launcher
  support is added;
- `Interactive(..., widgets="all")` behavior is covered if changed.

### Slice 7: Backed Persistence Follow-Up

Status: deferred

Add zarr persistence once the supported SpatialData write API for adding a new
shapes element to an existing backed store is verified.

Behavior:

- use SpatialData's supported write API rather than manual zarr mutation;
- keep in-memory and backed behavior distinct in user feedback;
- document whether saving to backed stores is immediate or requires a separate
  persist action.

Acceptance:

- backed-write tests pass against the canonical `.venv` environment;
- in-memory and backed behavior have distinct user feedback.

Tests:

- a backed `SpatialData` can persist a new shapes element to zarr;
- reloading the store preserves the new element name;
- reloading preserves geometry and row index;
- reloading preserves the `Identity()` transform for the selected coordinate
  system.

## Acceptance Criteria

- A user can create a new empty primary napari `Shapes` layer from
  `ShapesAnnotation()`.
- A user can draw supported shapes and save them as `sdata.shapes[new_name]`.
- The user can choose the coordinate system for the new shapes element.
- The widget stays synchronized with `app_state.coordinate_system`.
- The saved element uses an `Identity()` transform to the selected coordinate
  system.
- Generated shape indices are unique and stable.
- Unsupported shape types are rejected before mutation.
- Duplicate shapes element names are rejected before mutation.
- No annotation table is created implicitly.
- Existing shapes elements are not modified by this workflow.
- Styled shapes layers are never used as write-back sources.
