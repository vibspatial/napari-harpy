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
    shape_name: str
    display_name: str
    sdata: SpatialData
    coordinate_systems: tuple[str, ...]

    @property
    def identity(self) -> tuple[int, str]:
        return (id(self.sdata), self.shape_name)
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

- `_get_shape_names(sdata) -> list[str]`
- `get_spatialdata_shape_options_from_sdata(sdata)`
- `get_spatialdata_shape_options_for_coordinate_system_from_sdata(...)`
- `get_shape_column_color_source_options(sdata, shape_name)`

Update:

- `get_coordinate_system_names_from_sdata(...)` so coordinate systems from
  shapes are included.

Column discovery rules:

- inspect `sdata.shapes[shape_name]`;
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
  a replacement for `sdata.shapes[shape_name]`.

This avoids subtle bugs from trying to apply an `x, y` affine matrix to
napari data ordered as `y, x`.

Supported geometry rules:

- `Polygon`: render the exterior ring as one napari polygon.
- `MultiPolygon`: render each polygon part as one napari polygon, while keeping
  a source-index mapping so all parts receive the same color.
- circles: SpatialData stores these as `Point` geometries plus a `radius`
  column. Render as napari ellipses if the coordinate handling is clean;
  otherwise approximate as polygons from `point.buffer(radius)`.
- empty or invalid geometries: skip them and report a warning in the action
  feedback.
- holes/interiors: do not promise exact hole rendering in the first version
  unless napari's `Shapes` layer can represent them correctly in our target
  version. Rendering exterior rings only is acceptable for the first version if
  the feedback or implementation note is explicit.

Keep a per-napari-shape-row source mapping in layer metadata or binding-adjacent
state:

```python
source_shape_index_by_row: tuple[object, ...]
```

This is required because one SpatialData shape row can become several napari
shape rows when it is a `MultiPolygon`.

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

- discover tables with `get_element_annotators(sdata, shape_name)`;
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

Status: proposed

Implement:

- add shape discovery helpers;
- include shapes coordinate systems in viewer coordinate-system discovery;
- add a `Shapes` collapsible section to `ViewerWidget`;
- add `_ShapesCardWidget` with plain load controls and disabled column-color
  controls if no colorable columns exist;
- update summary and empty states to include shapes counts.

Recommended tests:

- coordinate systems are discovered from shapes-only data;
- shapes section appears when shapes are available;
- shapes section empty state appears when none are available;
- expanded-row state is preserved across refreshes.

### Slice 2: Plain Shapes Layer Loading

Status: proposed

Implement:

- add `ShapesLayerBinding`;
- add adapter lookup, registration, and removal paths for plain shapes layers;
- implement GeoDataFrame to napari `Shapes` conversion;
- support polygons, multipolygons, and circles;
- apply coordinate-system transforms by materializing transformed geometries;
- activate the loaded shapes layer after `Add / Update`.

Recommended tests:

- plain polygons load as a napari `Shapes` layer;
- multipolygon rows create multiple napari shapes but keep source-index mapping;
- circle rows render as ellipses or polygon approximations;
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
- repeat colors for multipolygon parts;
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
