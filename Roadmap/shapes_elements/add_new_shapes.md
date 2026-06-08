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

- Editing or overwriting arbitrary pre-existing `sdata.shapes[...]` elements.
- Save-as-copy for an existing shapes element.
- Reconciling shapes rows with existing `AnnData` tables.
- Creating or linking a new annotation table.
- Inferring shape identity from labels-linked tables.
- Palette editing or styled-shapes authoring.
- Supporting every napari shape type.
- Manual backed zarr mutation outside `harpy.sh.add_shapes(...)`.

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
13. Harpy writes it into `sdata.shapes[new_name]` with
    `harpy.sh.add_shapes(..., overwrite=False)` and an `Identity()` transform to
    the selected coordinate system.
14. Harpy refreshes relevant UI state and reports success.
15. The layer remains editable as the widget-owned primary annotation layer.
16. User may continue editing, adding, or deleting shapes in the same layer.
17. Each later `Save new shapes element` validates the same layer again and
    writes with `overwrite=True` to replace the same locked
    `sdata.shapes[new_name]` element.

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

After the first successful save, the layer remains the editable source for the
locked shapes element name. Workflow A passes `overwrite=True` on later saves of
that widget-owned element, but it must not use `overwrite=True` to target any
unrelated pre-existing `sdata.shapes[...]` element.

## Data Semantics

The saved shapes element should be interpreted in the selected display coordinate
system.

Transform rule:

```python
harpy.sh.add_shapes(
    request.sdata,
    input=geodataframe,
    output_shapes_name=request.shapes_name,
    transformations={request.coordinate_system: Identity()},
    instance_key=request.index_name,
    overwrite=request.overwrite,
)
```

Rationale:

- the user drew directly in the displayed coordinate frame;
- no inverse transform is needed for this create-new workflow;
- the saved coordinate values are already in the selected coordinate system;
- the first save uses `overwrite=False`; repeated saves from the same
  widget-owned, locked layer use `overwrite=True`.

Index rule:

- generate a stable `instance_id` for each new napari row;
- store each generated `instance_id` in the napari layer features so repeated
  saves can preserve row identity;
- default index values should be deterministic for new rows, for example
  `shape_0`, `shape_1`, ...;
- after the first save, existing rows keep their assigned `instance_id`;
- new rows receive the next unused `shape_N` value;
- when napari copies a generated `shape_N` value into a newly drawn row, treat
  the later duplicate as a new row and assign a fresh `shape_N`;
- duplicate custom/manual `instance_id` values remain invalid;
- deleted rows disappear on the next save;
- the generated index must be unique within the new `GeoDataFrame`;
- the generated index name should be `instance_id`.

Rationale:

- current table-backed shapes styling uses the `GeoDataFrame` index as the
  shape instance identity;
- naming the index `instance_id` makes later explicit table-linking less
  awkward, while still avoiding automatic table creation in this workflow;
- preserving `instance_id` values across repeated saves prevents inserts,
  deletes, or row reordering from changing existing shape identities.

Column rule:

- write a geometry column;
- initialize optional annotation metadata columns only when explicitly
  configured by the widget or future settings;
- do not create table-linkage columns by default.

Table rule:

- assume no existing table linkage;
- do not create an `AnnData` table automatically;
- creating/linking a table should be a separate explicit workflow later.

## Supported Shape Types

Supported napari shape types:

- `polygon` as Shapely `Polygon`;
- `rectangle` as Shapely `Polygon`, if napari exposes it as a shape type;
- `ellipse` as a polygonal Shapely `Polygon` approximation.

Region row semantics:

- each supported napari shape row is saved as one independent Shapely `Polygon`
  row;
- all saved rows are region annotations represented as Shapely `Polygon`
  geometries;
- ellipse rows are saved as polygonal approximations of the ellipse boundary;
- if the user draws one polygon inside another, save both as independent
  polygons;
- do not infer polygon holes from nested polygons.

Unsupported napari shape types:

- `line`;
- `path`;
- shapes with too few vertices for a valid polygon;
- empty rows;
- rows containing non-finite coordinates.

Ellipse handling:

- napari ellipse rows are stored as four bounding-box vertices;
- convert each ellipse to a deterministic Shapely `Polygon` approximation;
- use a default of 64 boundary samples per ellipse unless implementation tests
  show another value is preferable;
- reject ellipse rows whose bounding box has zero, negative, or non-finite width
  or height;
- do not store circles or ellipses as `Point` plus `radius` in this workflow.

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
class CreateShapesElementRequest:
    sdata: SpatialData
    shapes_name: str
    coordinate_system: str
    overwrite: bool = False
    index_name: str = "instance_id"
    index_prefix: str = "shape"


@dataclass(frozen=True)
class CreateShapesElementResult:
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
    ellipse_segments: int = 64,
) -> gpd.GeoDataFrame:
    ...


def create_shapes_element_from_napari_shapes_layer(
    request: CreateShapesElementRequest,
    layer: Shapes,
) -> CreateShapesElementResult:
    ...
```

Rules for `create_shapes_element_from_napari_shapes_layer(...)`:

- reject missing `sdata`;
- reject missing or unknown coordinate system;
- reject empty shapes element names;
- reject invalid ellipse segment counts;
- reject duplicate shapes element names when `request.overwrite` is `False`;
- allow overwriting `request.shapes_name` when `request.overwrite` is `True`;
- reject empty layers;
- validate all napari rows before mutating `sdata`;
- treat `request.overwrite` as the explicit caller decision for replacement;
- the widget decides when it is legitimate to pass `overwrite=True`, usually
  after the first successful save of its locked layer;
- write the converted `GeoDataFrame` with
  `harpy.sh.add_shapes(..., overwrite=request.overwrite)`;
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

- write the shapes element through
  `harpy.sh.add_shapes(..., overwrite=request.overwrite)`;
- reject existing shapes element names when `request.overwrite` is `False`;
- allow repeated saves for the same widget-owned, locked element name by
  passing `request.overwrite=True`;
- reject attempts to use Workflow A to overwrite unrelated existing shapes
  elements.

Backed-store behavior:

- use `harpy.sh.add_shapes(..., overwrite=request.overwrite)`, which backs the
  resulting shapes element to zarr when `sdata` is backed;
- keep the same locked-name rule as in memory: repeated saves may overwrite the
  widget-owned element, but the initial create flow must not target a
  pre-existing unrelated element;
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
- "Ellipse row `<index>` cannot be converted to a valid polygon."
- "Shape row `<index>` contains non-finite coordinates."
- "Saved `<name>` with `<n>` shape(s) in coordinate system `<cs>`."

## Implementation Slices

Build these slices in order. Each slice should leave the project in a tested,
usable state and should avoid taking dependencies on later slices.

### Slice 1: Core Conversion

Status: implemented

Implemented in:

- `src/napari_harpy/core/shapes_annotation.py`;
- `tests/test_shapes_annotation.py`.

Verified with:

- `.venv/bin/pytest tests/test_shapes_annotation.py`;
- `.venv/bin/ruff check src/napari_harpy/core/shapes_annotation.py tests/test_shapes_annotation.py`.

Purpose:

- add the core module and models for this workflow;
- convert a napari `Shapes` layer into a valid `GeoDataFrame`;
- keep this slice free of `SpatialData` mutation and widget code.

Code:

- add `src/napari_harpy/core/shapes_annotation.py`;
- add `CreateShapesElementRequest`;
- add `CreateShapesElementResult`;
- add `napari_shapes_layer_to_geodataframe(...)`;
- add focused tests in `tests/test_shapes_annotation.py`.

Behavior:

- convert napari `polygon` rows into Shapely `Polygon` rows;
- convert napari `rectangle` rows into Shapely `Polygon` rows, if represented
  as rectangles by napari;
- convert napari `ellipse` rows into Shapely `Polygon` approximations;
- reject `line` and `path` rows;
- reject empty layers, empty rows, rows with too few vertices, and rows with
  non-finite coordinates;
- reject ellipse rows whose bounding box has zero, negative, or non-finite width
  or height;
- reject invalid ellipse segment counts;
- save each supported napari shape row as an independent polygon;
- do not infer holes from nested polygons;
- generate deterministic indices for new rows: `shape_0`, `shape_1`, ...;
- store generated IDs in napari layer features under `instance_id`;
- preserve existing `instance_id` feature values across repeated conversions;
- assign new rows the next unused `shape_N` value;
- assign fresh IDs for duplicate generated `shape_N` values copied by napari
  into newly drawn rows;
- reject duplicate custom/manual `instance_id` values;
- name the generated index `instance_id`;
- swap napari `(y, x)` vertices to Shapely `(x, y)` coordinates.

Acceptance:

- supported napari shape rows convert to a valid `GeoDataFrame`;
- unsupported rows raise clear `ValueError`s with actionable messages;
- all converted geometry rows are Shapely `Polygon` instances;
- generated indices are unique, stable, and named `instance_id`;
- existing feature-backed `instance_id` values are preserved;
- no `SpatialData` mutation happens in this slice.

Tests:

- polygon conversion;
- rectangle conversion, if the napari representation is available in tests;
- ellipse polygonization;
- ellipse bounding-box validation;
- invalid ellipse segment count rejection;
- line and path rejection;
- empty layer rejection;
- non-finite coordinate rejection;
- index naming and generation;
- preservation of existing `instance_id` feature values;
- next-unused `shape_N` assignment after deletion or insertion;
- copied generated `shape_N` duplicates receive fresh IDs;
- duplicate custom/manual `instance_id` values are rejected;
- coordinate order conversion;
- nested polygons remain independent rows.

### Slice 2: Core Save Into SpatialData

Status: implemented

Implemented in:

- `src/napari_harpy/core/shapes_annotation.py`;
- `tests/test_shapes_annotation.py`.

Verified with:

- `.venv/bin/pytest tests/test_shapes_annotation.py`;
- `.venv/bin/ruff check src/napari_harpy/core/shapes_annotation.py tests/test_shapes_annotation.py`.

Purpose:

- add the write-back core for creating or updating the locked `sdata.shapes[...]`
  element;
- validate everything before mutating `sdata`;
- keep this slice UI-free. Widget ownership and initial name locking are
  enforced by later widget slices.

Code:

- add `create_shapes_element_from_napari_shapes_layer(...)` in
  `src/napari_harpy/core/shapes_annotation.py`;
- reuse `napari_shapes_layer_to_geodataframe(...)`;
- import `get_coordinate_system_names_from_sdata` from
  `napari_harpy.core.spatialdata`;
- import `harpy as hp`;
- import `Identity` from `spatialdata.transformations`;
- extend `tests/test_shapes_annotation.py`.

Behavior:

- reject invalid request fields before touching the napari layer:
  - `sdata` is missing;
  - `shapes_name` is not a non-empty string after stripping whitespace;
  - `coordinate_system` is not a non-empty string after stripping whitespace;
  - `overwrite` is not a boolean;
  - `index_name` is not a non-empty string;
  - `index_prefix` is not a non-empty string;
- reject unknown coordinate systems by checking
  `get_coordinate_system_names_from_sdata(request.sdata)` before mutating
  `sdata`;
- reject existing `sdata.shapes[request.shapes_name]` targets when
  `request.overwrite` is `False`;
- allow replacing `sdata.shapes[request.shapes_name]` when
  `request.overwrite` is `True`;
- convert the napari layer with `napari_shapes_layer_to_geodataframe(...)`;
- treat conversion errors as validation failures and leave `sdata.shapes`
  unchanged;
- call `harpy.sh.add_shapes(...)` only after request validation and conversion
  succeed;
- write with:

  ```python
  _ = hp.sh.add_shapes(
      request.sdata,
      input=geodataframe,
      output_shapes_name=request.shapes_name,
      transformations={request.coordinate_system: Identity()},
      instance_key=request.index_name,
      overwrite=request.overwrite,
  )
  ```

- treat `request.overwrite` as the core replacement policy;
- do not infer widget ownership in this core helper. The widget slice decides
  when it is legitimate to pass `overwrite=True`;
- intentionally ignore the return value from `hp.sh.add_shapes(...)`. The helper
  treats the Harpy call as a side-effecting write and callers continue using
  `request.sdata`;
- return `CreateShapesElementResult` with name, coordinate system, and row
  count;
- do not create or link an `AnnData` table;
- use `harpy.sh.add_shapes(...)` as the backed-capable write path. Slice 2 does
  not add manual zarr mutation and does not own full backed-store roundtrip
  verification.

Save order:

1. Validate request-only fields before touching the napari layer: `sdata`,
   `shapes_name`, `coordinate_system`, `overwrite`, `index_name`,
   and `index_prefix`.
2. Validate target collision policy: if `request.overwrite` is `False` and
   `request.shapes_name` already exists in `request.sdata.shapes`, fail before
   conversion.
3. Convert the napari layer with `napari_shapes_layer_to_geodataframe(...)`.
   Conversion may update `layer.features` with stable `instance_id` values after
   all geometry rows validate.
4. If conversion succeeds, call `hp.sh.add_shapes(...)`. This mutates
   `request.sdata`.

Mutation boundary:

- `napari_shapes_layer_to_geodataframe(...)` may write stable `instance_id`
  values back to `layer.features`;
- `sdata` must not be mutated until request validation and conversion succeed;
- if `hp.sh.add_shapes(...)` raises, propagate the error and leave widget
  feedback to the caller.

Acceptance:

- validates request and layer;
- rejects validation failures before mutation;
- writes `sdata.shapes[new_name]`;
- stores `Identity()` to the selected coordinate system;
- preserves the generated row index;
- rejects name collisions when `request.overwrite` is `False`;
- calls `hp.sh.add_shapes(...)` with `overwrite=request.overwrite`;
- returns row-count feedback.

Tests:

- saving writes `sdata.shapes[new_name]`;
- missing `sdata` is rejected;
- blank shapes names are rejected;
- blank coordinate systems are rejected;
- missing coordinate system is rejected;
- unknown coordinate system is rejected;
- non-boolean overwrite values are rejected;
- existing shapes names are rejected when `overwrite=False`;
- invalid index names are rejected;
- invalid index prefixes are rejected;
- failed conversion does not mutate `sdata.shapes`;
- the new element has an `Identity()` transform to the selected coordinate
  system;
- repeated saves with `overwrite=True` overwrite the locked element and
  preserve existing `instance_id` values;
- `hp.sh.add_shapes(...)` is called with `overwrite=request.overwrite`;
- no manual zarr write path is introduced;
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
- call `create_shapes_element_from_napari_shapes_layer(...)`;
- refresh relevant widget/viewer state after success;
- extend widget tests.

Behavior:

- save only from the widget-owned primary layer;
- reject saving if there is no pending layer;
- reject saving if the pending layer binding no longer matches the widget
  request;
- reject initial layer creation if the requested name already exists in
  `sdata.shapes`;
- pass the selected coordinate system and locked element name to the core save
  helper;
- show actionable conversion/save errors in the status area;
- show:

  ```text
  Saved `<name>` with `<n>` shape(s) in coordinate system `<cs>`.
  ```

- after successful save, keep the newly saved element registered as the source
  element for the editable widget-owned layer;
- keep the layer editable after save;
- pass `overwrite=False` on the first save and `overwrite=True` on later saves
  from the same widget-owned layer;
- preserve existing `instance_id` values across repeated saves;
- do not accept styled shapes layers as save sources.

Acceptance:

- save validates the widget-owned primary layer;
- successful save writes the new element;
- successful save reports row count and coordinate system;
- a saved layer can be edited and saved again to the same locked element name;
- errors are shown as clear widget feedback;
- no styled layer can be saved by this workflow.

Tests:

- save calls the core save helper with the selected coordinate system;
- save passes the locked new shapes element name;
- save feedback reports row count and coordinate system;
- duplicate-name validation is enforced before initial layer creation;
- repeated save preserves `instance_id` values for existing rows;
- repeated save assigns new `instance_id` values only to newly-added rows;
- repeated save removes deleted rows from the saved element;
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

### Slice 7: Backed Persistence Verification

Status: proposed

Verify backed persistence for the `harpy.sh.add_shapes(...)` write path used by
Workflow A.

Slice 2 only ensures the save helper uses Harpy's backed-capable API; this
slice verifies the actual backed zarr behavior by writing, reloading, and
checking the saved element.

Behavior:

- use `harpy.sh.add_shapes(..., overwrite=request.overwrite)` rather than
  manual zarr mutation;
- verify the first-save `overwrite=False` path and the repeated-save
  `overwrite=True` path for the same widget-owned, locked shapes element;
- keep in-memory and backed behavior distinct in user feedback;
- document whether saving to backed stores is immediate or requires a separate
  persist action.

Acceptance:

- backed-write tests pass against the canonical `.venv` environment;
- in-memory and backed behavior have distinct user feedback.

Tests:

- a backed `SpatialData` can persist a new shapes element to zarr;
- repeated backed saves with `overwrite=True` overwrite the same locked shapes
  element;
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
- Existing shape `instance_id` values are preserved across repeated saves.
- Unsupported shape types are rejected before mutation.
- Duplicate shapes element names are rejected before initial layer creation.
- No annotation table is created implicitly.
- The widget-owned saved layer remains editable and can be saved again to the
  same locked element name.
- Arbitrary pre-existing shapes elements are not modified by this workflow.
- Styled shapes layers are never used as write-back sources.
