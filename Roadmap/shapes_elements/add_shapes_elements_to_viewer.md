# Add Shapes Elements To Viewer

## Goal

Add first-class viewer support for `sdata.shapes[...]` elements in
napari-harpy.

The viewer widget should discover shapes elements from the loaded
`SpatialData` object, show them beside images and segmentations, and let users
load them into napari. After plain loading is stable, the viewer should support
coloring shapes by scalar columns stored directly on the `GeoDataFrame`.

The initial design should be conservative around `AnnData` table-backed shape
coloring, because the current harpy and napari-harpy flows assume linked tables
annotate labels elements.

## SpatialData Table Semantics

SpatialData supports tables annotating shapes. The official docs describe
shapes as `geopandas.GeoDataFrame` objects, and the SpatialData object docs say
shapes are regions that can be annotated by tables.

The table metadata contract is the important part:

- `region` is a string or list of strings naming the annotated
  `SpatialElement` targets.
- `region_key` names a `table.obs` column that says, for each table row, which
  `SpatialElement` that row annotates.
- `instance_key` names a `table.obs` column that says, for each table row, which
  instance index inside that element is annotated.

Consequence:

- A single `AnnData` table can annotate multiple spatial elements overall.
- A single row in that table points to one spatial element and one instance via
  its `region_key` and `instance_key` values.
- SpatialData does not provide a native "one row annotates both this labels
  instance and this shapes instance" relation in the table metadata itself.

So for napari-harpy we should not infer that a row from a labels-linked table
also annotates a shapes element just because the labels instance id and shapes
index happen to match.

If users want one biological object represented both as labels and shapes, the
safe SpatialData-compatible options are:

- make the table annotate the shapes element instead of the labels element,
  when the shapes are the canonical representation for that workflow;
- create table rows for each annotated region, e.g. one row for the labels
  instance and one row for the shapes instance, with matching shared metadata
  stored in columns;
- keep separate labels and shapes tables;
- add an explicit crosswalk table or column, then build a higher-level harpy
  feature later that understands that crosswalk.

## Recommendation

Implement shape visualization in two stages.

First version:

- discover and load shapes elements;
- add a dedicated `Shapes` section to the viewer widget;
- support plain geometry display;
- support coloring by scalar columns directly stored on the shapes
  `GeoDataFrame`, for example `leiden`, `in_tumor`, `area`, or `score`;
- keep table-backed shape coloring out of the first implementation unless the
  table already explicitly annotates the selected shapes element.

Follow-up version:

- add table-backed shape coloring for tables whose `region` metadata includes
  the selected shapes element;
- reuse the table source rules already used for labels overlays;
- do not reuse labels-linked tables for shapes unless the table metadata says
  the table annotates the shapes element.

This gives users immediate value for common shape annotations while keeping the
SpatialData table contract honest.

## Current Codebase Fit

Relevant existing pieces:

- `ViewerWidget` already has coordinate-system-first browsing for images and
  labels.
- `ViewerAdapter` owns layer creation, lookup, activation, removal, and
  registry binding.
- `LayerBindingRegistry` is the source of truth for Harpy-managed napari
  layers.
- `core.spatialdata` already contains discovery helpers for labels, images,
  linked tables, and table color sources.
- Labels viewer coloring already separates primary labels layers from styled
  overlay variants, which is a useful pattern for future table-backed shapes
  coloring.

The shape implementation should extend these patterns rather than adding
shape-specific state directly to the widget.

## User Experience

The viewer widget should show three top-level element sections for a selected
coordinate system:

- `Images`
- `Segmentations`
- `Shapes`

The summary should mention all three counts, for example:

`In coordinate system global: 1 image element(s), 2 segmentation mask(s), and 3 shapes element(s).`

Each shapes row should be expandable, matching the existing image and labels
rows. The first version of the card should expose:

- shape element name;
- color source selector:
  - `None`;
  - `Shape column`;
- searchable column selector when `Shape column` is selected;
- one `Add / Update in viewer` button;
- concise action text, for example:
  - `Action: add/update shapes layer`;
  - `Action: add/update colored shapes layer for shapes["leiden"]`.

The first version should use one plain shapes layer for `None` and one styled
shape variant per selected column. That mirrors the labels overlay model and
allows users to compare multiple shape annotations without constantly
recoloring one layer.

Layer names should make the style source visible:

- plain: `shapes_name`
- styled column: `shapes_name[shape:leiden]`

## Data Model Additions

Add a shapes option model in `core.spatialdata`:

```python
@dataclass(frozen=True)
class SpatialDataShapesOption:
    shapes_name: str
    display_name: str
    sdata: SpatialData
    coordinate_systems: tuple[str, ...]

    @property
    def identity(self) -> tuple[int, str]:
        return (id(self.sdata), self.shapes_name)
```

Add shape color source metadata. Keep it separate from
`TableColorSourceSpec`, because first-version shape coloring is element-column
backed, not table-backed.

```python
ShapeColorSourceKind = Literal["shape_column"]

@dataclass(frozen=True)
class ShapeColorSourceSpec:
    source_kind: ShapeColorSourceKind
    value_key: str
    value_kind: ColorValueKind

    @property
    def identity(self) -> tuple[ShapeColorSourceKind, str]:
        return (self.source_kind, self.value_key)

    @property
    def display_name(self) -> str:
        return self.value_key
```

If follow-up table-backed shape coloring lands later, introduce a broader
`SpatialElementColorSourceSpec` or a union of `ShapeColorSourceSpec` and
`TableColorSourceSpec`. Do not overload `TableColorSourceSpec` with optional
shape-column fields.

## Discovery Helpers

Add helpers in `core.spatialdata`:

- `_get_shapes_names(sdata) -> list[str]`
- `get_spatialdata_shapes_options_from_sdata(sdata)`
- `get_spatialdata_shapes_options_for_coordinate_system_from_sdata(...)`
- `get_shape_column_color_source_options(sdata, shapes_name)`

Update:

- `get_coordinate_system_names_from_sdata(...)` so coordinate systems from
  shapes are included.

Column discovery rules:

- inspect `sdata.shapes[shapes_name]`;
- exclude the active geometry column;
- treat columns named `<value_column>_colors` as optional companion color
  columns for categorical shape columns, not as ordinary color-by choices;
- expose scalar columns that can be interpreted as categorical or continuous;
- allow `radius` to be exposed when present, because it is a valid scalar
  column, even though it is also part of circle geometry;
- classify values with the same broad rules as table color sources:
  - pandas categorical -> categorical;
  - bool -> categorical;
  - exact binary integer `{0, 1}` -> categorical;
  - other integers -> continuous;
  - floats -> continuous;
  - string/object columns containing scalar strings -> categorical;
  - mixed or non-scalar columns -> unsupported.

Missing or unsupported columns should produce clear disabled UI states, not
tracebacks.

Companion color-column convention:

- for a categorical shape column named `leiden`, napari-harpy should look for
  a sibling column named `leiden_colors`;
- `leiden_colors` is the only supported companion-column name because it
  mirrors the Scanpy/AnnData `uns["leiden_colors"]` convention;
- companion color columns are row-level color data, so they must have the same
  length as the shapes `GeoDataFrame`, not a compact palette list like
  `adata.uns["leiden_colors"]`;
- all rows with the same `leiden` category must map to the same color token;
- color tokens should be validated with the same color parser used by the
  rendering path, e.g. `#ff0000`, named colors such as `red`, and other
  formats accepted by napari / matplotlib;
- if the companion column is valid, use it as the stored categorical palette;
- if it is missing or invalid, warn and fall back to the deterministic default
  categorical palette;
- companion columns should remain hidden from the normal color-source selector
  when their base column exists, because users should choose `leiden`, not
  `leiden_colors`;
- if we later add an explicit direct-color-column mode, companion color columns
  can become selectable there without changing the categorical-coloring
  convention.

This mirrors the Scanpy/AnnData `_colors` naming convention while keeping the
storage format practical for `GeoDataFrame` IO, where normal columns round-trip
more reliably than arbitrary custom `GeoDataFrame.attrs`.

## Adapter And Binding Design

Extend the adapter binding model with shapes.

Recommended binding shape:

```python
@dataclass(frozen=True, kw_only=True)
class ShapesLayerBinding(BaseLayerBinding):
    element_type: Literal["shapes"] = "shapes"
    shapes_role: Literal["plain", "styled"] = "plain"
    style_spec: ShapeColorSourceSpec | None = None
```

Update the union:

```python
LayerBinding = LabelsLayerBinding | ImageLayerBinding | ShapesLayerBinding
```

Add `ViewerAdapter` methods:

- `get_loaded_plain_shapes_layer(...)`
- `get_loaded_styled_shapes_layer(...)`
- `get_loaded_styled_shapes_layers(...)`
- `ensure_shapes_loaded(...)`
- `ensure_styled_shapes_loaded(...)`
- `remove_shapes_layers(...)`

Shape layers should not emit `primary_labels_layers_changed`. Object
classification should remain labels-only.

## Geometry Conversion

SpatialData shapes are `GeoDataFrame` objects with `x`, `y` axes. Napari
`Shapes` layer data is coordinate-array based and should be supplied in napari
dimension order, which for this viewer is effectively `y`, `x`.

Recommended first implementation:

- use `spatialdata.transform(shape_element, to_coordinate_system=coordinate_system)`
  to materialize the shapes in the selected coordinate system;
- convert transformed geometry coordinates from `x, y` to napari `y, x`;
- create the napari `Shapes` layer without an additional affine transform;
- keep the transformed `GeoDataFrame` only as layer construction input, not as
  a replacement for `sdata.shapes[shapes_name]`.

This avoids subtle bugs from trying to apply an `x, y` affine matrix to
napari data ordered as `y, x`.

Supported geometry rules:

- `Polygon`: render the exterior ring as one napari polygon.
- `MultiPolygon`: render each polygon part as one napari polygon, while keeping
  a source-index mapping so all parts receive the same color. This is more
  complete than `napari-spatialdata`'s current viewer path, which explodes
  multipolygons and keeps only the largest polygon per source row.
- circles: SpatialData stores these as `Point` geometries plus a `radius`
  column. Render them as napari `ellipse` shapes from four bounding-box corner
  coordinates. This gives the viewer one plain `Shapes` layer contract for
  polygons, multipolygons, and circles.
- empty geometries: skip them and report a warning in the action feedback.
- invalid geometries: try a conservative `shapely.make_valid(...)` repair when
  available; flatten any repaired polygonal result into renderable polygons;
  skip still-empty, still-invalid, or non-polygonal results and report a warning.
- holes/interiors: preserve them for polygon and multipolygon parts by encoding
  each Shapely polygon as one napari polygon path with embedded interior rings.
  Napari represents holes by removing bridge edges that are traversed twice, so
  Harpy should orient the polygon and embed each interior ring in the exterior
  path before converting `x, y` coordinates to napari `y, x` coordinates.

Recommended hole encoding:

```python
def _polygon_to_napari_path(polygon: Polygon) -> list[tuple[float, float]]:
    """Encode a Shapely polygon as one napari path, preserving holes.

    Napari can render polygon holes when the interior rings are embedded in the
    same vertex path as the exterior ring and wind in the opposite direction.
    The repeated exterior anchor creates bridge edges that napari's
    triangulation removes because they are traversed twice.
    """
    oriented = shapely.geometry.polygon.orient(polygon, sign=1.0)
    path = list(oriented.exterior.coords)
    anchor = path[0]
    for interior in oriented.interiors:
        path.extend(interior.coords)
        path.append(anchor)
    return path
```

This keeps one source `Polygon` as one napari shape row even when it has holes.
For `MultiPolygon`, apply the same encoding to each polygon part and map every
part back to the same source row in
`layer.metadata["source_shapes_index_by_row"]`.

Keep a per-napari-shape-row source mapping as primary layer metadata:

```python
layer.metadata["source_shapes_index_by_row"] = tuple(...)
```

This is required because one SpatialData shape row can become several napari
shape rows when it is a `MultiPolygon`.

`napari-spatialdata` stores this concept under the generic
`layer.metadata["indices"]` key. Harpy should use
`source_shapes_index_by_row` as the primary key because it names the row-mapping
contract directly and avoids overloading the labels/table-oriented meaning of
`indices`.

## Shape Styling

Plain shapes layer defaults:

- deterministic edge color;
- transparent or lightly transparent fill;
- readable edge width;
- layer opacity that does not hide images underneath.

Styled shapes:

- categorical shape-column values should produce per-shape face and/or edge
  colors from either a valid `<column>_colors` companion column or a
  deterministic default palette;
- continuous values should use a continuous colormap;
- missing values should receive a neutral transparent or muted color;
- `MultiPolygon` parts should repeat the source row color;
- string/object columns should be colored through a temporary in-memory
  categorical representation, without mutating the `GeoDataFrame`;
- do not write palette columns or category cleanup back to `sdata`;
- do not consult companion color columns for continuous columns.

Shape-column styling should use
`layer.metadata["source_shapes_index_by_row"]` to align source values to
rendered napari shape rows. Napari expects one color per `layer.data` row, but
Harpy's values live on `sdata.shapes[shapes_name]`. If one source row expands
into several rendered rows, such as a `MultiPolygon`, every rendered row should
look up the same source index and therefore receive the same color.

For example:

```python
source_index_by_row = layer.metadata["source_shapes_index_by_row"]
source_values = sdata.shapes[shapes_name][column_name]
row_values = [source_values.loc[source_index] for source_index in source_index_by_row]
layer.face_color = [palette[value] for value in row_values]
```

For categorical companion colors, build the category-to-color mapping from the
full original shape column before expanding multipolygons. For example:

```text
geometry | leiden | leiden_colors
poly1    | 0      | #1f77b4
poly2    | 1      | #ff7f0e
poly3    | 0      | #1f77b4
```

This should produce a palette equivalent to:

```python
{"0": "#1f77b4", "1": "#ff7f0e"}
```

If one category maps to multiple colors, or if any color token is invalid, the
entire companion palette should be ignored for that source and the default
palette should be used instead. Do not try to partially salvage the mapping.

The implementation should return a structured result for feedback, similar to
styled labels:

```python
@dataclass(frozen=True)
class StyledShapesLoadResult:
    layer: Shapes
    created: bool
    value_kind: ColorValueKind
    palette_source: Literal["shape_column", "default", None]
    coercion_applied: bool
    skipped_geometry_count: int
```

## Table-Backed Shape Coloring Follow-Up

Once direct shape-column coloring is stable, add table-backed shapes coloring
only for tables that explicitly annotate the selected shapes element.

Rules:

- discover tables with `get_element_annotators(sdata, shapes_name)`;
- validate `region_key`, `instance_key`, and duplicate instance ids within the
  shapes region;
- align table values to shape rows through `instance_key` and the
  `GeoDataFrame` index;
- support table `obs` and `X[:, var_name]` sources using the existing
  `TableColorSourceSpec` semantics;
- do not offer a table for shapes if that table annotates only labels;
- do not infer labels-to-shapes identity from matching integer ids.

If we later want to support "label row also colors derived shape row", define
an explicit crosswalk feature. That should be a separate roadmap item because
it is a semantic join, not basic SpatialData table annotation.

## Implementation Slices

### Slice 1: Discovery And UI Shell

Status: completed

Purpose:

Establish shapes as a first-class browsable element type in the viewer widget,
without loading geometry into napari yet. This slice should make shapes visible
in the coordinate-system-first UI and settle the widget contract that later
slices will attach loading and styling behavior to.

Implement:

- add shape discovery helpers;
- include shapes coordinate systems in viewer coordinate-system discovery;
- add a `Shapes` collapsible section to `ViewerWidget`;
- add `_ShapesCardWidget` as a mostly presentational card for one shapes
  element;
- show a disabled `Add / Update in viewer` control or equivalent disabled
  action state until Slice 2 implements layer loading;
- update summary and empty states to include shapes counts;
- expose `shape_cards` and `shape_rows` test helpers on `ViewerWidget`,
  matching the existing image and labels helpers;
- preserve expanded shapes row state across coordinate-system refreshes.

Recommended concrete code touchpoints:

- add `SpatialDataShapesOption` beside `SpatialDataLabelsOption` and
  `SpatialDataImageOption`;
- add `_get_shapes_names(sdata) -> list[str]`;
- add `get_spatialdata_shapes_options_from_sdata(sdata)`;
- add `get_spatialdata_shapes_options_for_coordinate_system_from_sdata(...)`;
- update `get_coordinate_system_names_from_sdata(...)` so shapes-only
  SpatialData objects expose selectable coordinate systems;
- add `_get_shapes_in_coordinate_system(...)` in the viewer widget module, or
  delegate directly to the new core helper;
- extend `ViewerWidget._refresh_coordinate_system_content(...)`,
  `_update_section_empty_states(...)`, and `_clear_cards(...)` to handle
  images, labels, and shapes consistently.

Out of scope:

- creating napari `Shapes` layers;
- adding `ShapesLayerBinding` or other adapter registry changes;
- geometry conversion from `GeoDataFrame` to napari shape arrays;
- shape-column coloring;
- table-backed shape coloring;
- any labels-to-shapes table inference.

Recommended tests:

- coordinate systems are discovered from shapes-only data;
- shapes section appears when shapes are available;
- shapes section empty state appears when none are available;
- expanded-row state is preserved across refreshes;
- the disabled Slice 1 action does not call the adapter or mutate viewer
  layers.

### Slice 2: Plain Shapes Layer Loading

Status: proposed

Implement:

- add `ShapesLayerBinding`;
- add adapter lookup, registration, and removal paths for plain shapes layers;
- implement GeoDataFrame to napari `Shapes` conversion in the selected
  coordinate system;
- support polygons, multipolygons, and `Point` + `radius` circles;
- apply coordinate-system transforms by materializing transformed geometries;
- set `layer.metadata["source_shapes_index_by_row"]` as the primary rendered-row
  to source-row mapping;
- skip empty, invalid, or unsupported geometries with explicit feedback instead
  of failing the whole load when at least one shape can be rendered;
- preserve polygon interiors by encoding holes as embedded rings in the napari
  polygon path;
- activate the loaded shapes layer after `Add / Update`.

Recommended tests:

- plain polygons load as a napari `Shapes` layer;
- multipolygon rows create multiple napari shapes and duplicate their source row
  in `metadata["source_shapes_index_by_row"]`;
- circle rows render as napari ellipses;
- empty or invalid geometries are skipped with warning feedback;
- polygon interiors render as holes rather than filled exterior-only polygons;
- loading the same shapes element twice reuses the existing plain layer;
- removing layers for a coordinate system also removes shapes layers;
- object-classification labels-layer signals are not emitted for shapes.

### Slice 3: Shape-Column Coloring

Status: proposed

Implement:

- add `ShapeColorSourceSpec`;
- add shape-column source discovery;
- add styled shapes adapter lookup and load/update path;
- style shapes by categorical and continuous columns;
- use valid `<column>_colors` companion columns as stored categorical palettes;
- use `metadata["source_shapes_index_by_row"]` to align source shape-column
  values to rendered napari rows;
- repeat colors for multipolygon parts by repeating the source row lookup;
- provide feedback for created vs updated layers, palette source, invalid
  companion palettes, and skipped geometries.

Recommended tests:

- categorical columns produce per-shape categorical colors;
- categorical columns use valid `<column>_colors` companion palettes;
- invalid companion color values fall back to the default palette;
- categories with conflicting companion colors fall back to the default
  palette;
- bool and exact binary integer columns are categorical;
- non-binary integer and float columns are continuous;
- string/object scalar columns are temporarily categorical without mutating the
  `GeoDataFrame`;
- multipolygon parts repeat the source row color via
  `metadata["source_shapes_index_by_row"]`;
- mixed unsupported columns are hidden from the selector;
- styled layer identity includes the selected column and is reused on repeat.

### Slice 4: Explicit Shape-Table Coloring

Status: follow-up

Implement only after the first three slices are stable.

Implement:

- discover linked tables for shapes with `get_element_annotators`;
- expose table `obs` and `X[:, var_name]` color sources for shapes;
- align table values to the shapes index;
- share palette and continuous-color behavior with labels overlays where
  possible;
- report a clear unavailable state when no table explicitly annotates the
  shapes element.

Recommended tests:

- tables that annotate shapes are discovered;
- tables that annotate only labels are not offered for shapes;
- table values align by `instance_key`, not by table row order;
- duplicate `instance_key` values within one shapes region are rejected;
- missing shape indices receive the configured missing color.

## Shape Write-Back Follow-Up

Shape write-back should be treated as a separate roadmap area from first-pass
shape viewing and coloring. There are two distinct user workflows here, and
they should have separate write paths because they carry different risks.

### Follow-Up A: Shape Annotation Creation

This workflow means "create a new shapes annotation layer".

User flow:

- the user creates a new empty napari `Shapes` layer or clicks a Harpy action
  such as `New Shapes Annotation`;
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

### Follow-Up B: Shape Geometry Editing

This workflow means "modify an existing `SpatialData` shapes element".

User flow:

- the user loads `sdata.shapes["cell_boundaries"]`;
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

Add/delete is a separate problem from vertex editing. If the user deletes or
adds shapes, Harpy needs an explicit policy for table rows and metadata. That
should not happen silently.

Recommended tests:

- editing vertices with unchanged source indices can update the geometry;
- scalar columns and companion `_colors` columns survive geometry-only edits;
- overwrite requires confirmation;
- save-as-copy is available even when overwrite is blocked;
- add/delete is blocked or clearly reported when linked tables annotate the
  shapes element;
- unsupported napari shape types are rejected with actionable feedback.

### Follow-Up C: Shape Table Reconciliation

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

## Acceptance Criteria

- Users can load `sdata.shapes["..."]` elements from the viewer widget.
- Shapes are filtered by the selected coordinate system.
- Images, segmentations, and shapes can coexist in the viewer.
- Polygon, multipolygon, and circle shapes render in the correct coordinate
  system.
- Users can color shapes by scalar `GeoDataFrame` columns such as `leiden` or
  `in_tumor`.
- Categorical shape columns can use row-level companion color columns such as
  `leiden_colors`.
- Direct shape-column coloring does not mutate the `GeoDataFrame`.
- Labels annotation and object-classification behavior remain unchanged.
- Table-backed shapes coloring is only exposed when SpatialData metadata says
  the table annotates the selected shapes element.

## Non-Goals For The First Version

- editing shapes geometry in napari and writing it back to SpatialData;
- rich legends or palette editors;
- exact polygon-hole rendering unless napari supports it cleanly in our target
  version;
- using labels-linked tables to color shapes by implicit id matching;
- cross-element biological-object identity resolution.

## References

- SpatialData object docs:
  https://spatialdata.scverse.org/en/latest/api/SpatialData.html
- SpatialData annotation/table tutorial:
  https://spatialdata.scverse.org/en/latest/tutorials/notebooks/notebooks/examples/tables.html
- SpatialData from-scratch tutorial, connecting `AnnData` to shapes:
  https://spatialdata.scverse.org/en/stable/tutorials/notebooks/notebooks/examples/sdata_from_scratch.html
