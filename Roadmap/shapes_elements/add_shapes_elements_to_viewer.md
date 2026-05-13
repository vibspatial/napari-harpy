# Add Shapes Elements To Viewer

## Goal

Add first-class viewer support for `sdata.shapes[...]` elements in
napari-harpy.

The viewer widget should discover shapes elements from the loaded
`SpatialData` object, show them beside images and segmentations, and let users
load them into napari. After basic loading is stable, the viewer should support
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
- support geometry display;
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

The first version should use one shapes layer for `None`. A later coloring
slice can decide whether styled shape variants should live in separate layers
or update an existing shapes layer.

Layer naming for the initial `None` path:

- `None`: `shapes_name`

If a later coloring slice chooses separate styled layer variants, specify their
names in that slice.

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

Add a neutral color-source module at `core/_color_source.py`. It should own the
shared color-source primitives and the source-specific spec dataclasses for
both table-backed labels and shape-column-backed shapes.

```python
ColorValueKind = Literal["categorical", "continuous", "instance"]
TableColorSourceKind = Literal["obs_column", "x_var"]
ShapeColorSourceKind = Literal["shape_column"]

@dataclass(frozen=True)
class TableColorSourceSpec:
    source_kind: TableColorSourceKind
    value_key: str
    value_kind: ColorValueKind

@dataclass(frozen=True)
class ShapeColorSourceSpec:
    source_kind: ShapeColorSourceKind
    value_key: str
    value_kind: Literal["categorical", "continuous"]

    @property
    def identity(self) -> tuple[ShapeColorSourceKind, str]:
        return (self.source_kind, self.value_key)

    @property
    def display_name(self) -> str:
        return self.value_key
```

`TableColorSourceSpec` and `ShapeColorSourceSpec` should remain distinct
dataclasses even though they live in the same module. The table spec describes
linked `AnnData` sources, while the shape spec describes direct scalar columns
on `sdata.shapes[shapes_name]`.

Move the existing table color-source spec definitions into
`core/_color_source.py`. Styling helpers such as high-cardinality string
warnings should live in `viewer/_styling.py` after the styling refactor slice.
Remove `core/table_color_source.py` and update imports to
`napari_harpy.core._color_source`; do not keep a compatibility shim or
re-export module.

If follow-up table-backed shape coloring lands later, introduce a broader
`SpatialElementColorSourceSpec` in `core/_color_source.py`, or use a union of
`ShapeColorSourceSpec` and `TableColorSourceSpec`. Do not overload
`TableColorSourceSpec` with optional shape-column fields.

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
    source_shapes_index_by_row: tuple[Any, ...] = ()
    source_shapes_index_feature_name: str = "index"
    skipped_geometry_count: int = 0
```

Update the union:

```python
LayerBinding = LabelsLayerBinding | ImageLayerBinding | ShapesLayerBinding
```

Add `ViewerAdapter` methods:

- `_get_loaded_shapes_layer_for_coordinate_system(...)`
- `ensure_shapes_loaded(...)`
- `remove_shapes_layer(...)`

Shape-column coloring should choose its adapter contract in Slice 6 instead of
baking a styled-layer model into Slice 2.

As the adapter grows, avoid extending one generic public
`ViewerAdapter.register_layer(...)` method with every layer-type-specific
argument. Prefer typed registration entrypoints:

```python
def register_labels_layer(...) -> LabelsLayerBinding: ...
def register_image_layer(...) -> ImageLayerBinding: ...
def register_shapes_layer(...) -> ShapesLayerBinding: ...
```

The typed methods should construct the appropriate binding and then call one
shared private registration helper, so insertion bookkeeping and viewer signals
stay centralized while labels-, image-, and shapes-specific parameters remain
readable.

Slice 6 should extend this binding with `shapes_role` and `style_spec` for
primary/styled layer variants.

After Slice 3, `ShapesLayerBinding.source_shapes_index_by_row` is the
authoritative Harpy mapping from rendered napari shape rows back to source
`GeoDataFrame` indices.
`layer.features[source_shapes_index_feature_name]` is the napari-visible
feature table used for status-bar display and manual inspection.
`ShapesLayerBinding.skipped_geometry_count` stores the skipped-geometry count
needed for viewer feedback without using `layer.metadata`.

After Slice 3, do not store Harpy shapes contracts in `layer.metadata`. This
includes source row mappings, source-index feature names, skipped-geometry
counts, shapes roles, and style specs. Shape layer identity and behavior should
be resolved through `LayerBindingRegistry`.

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
  coordinates. This gives the viewer one `Shapes` layer contract for
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
`ShapesLayerBinding.source_shapes_index_by_row`.

Keep a per-napari-shape-row source mapping in the adapter binding:

```python
binding.source_shapes_index_by_row = tuple(...)
```

This is required because one SpatialData shape row can become several napari
shape rows when it is a `MultiPolygon`.

`napari-spatialdata` stores this concept under the generic
`layer.metadata["indices"]` key. Harpy should not follow that storage contract
for shapes. The row mapping should live in `ShapesLayerBinding`, because Harpy
already treats `LayerBindingRegistry` as the authoritative layer contract and
uses layer metadata only as loose napari-facing state for other layer types.

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
`ShapesLayerBinding.source_shapes_index_by_row` to align source values to
rendered napari shape rows. Napari expects one color per `layer.data` row, but
Harpy's values live on `sdata.shapes[shapes_name]`. If one source row expands
into several rendered rows, such as a `MultiPolygon`, every rendered row should
look up the same source index and therefore receive the same color.

For example:

```python
binding = adapter.layer_bindings.get_binding(layer)
source_index_by_row = binding.source_shapes_index_by_row
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

The shapes-specific styling API should live in `viewer/shapes_styling.py`.
It should return a structured result for feedback, similar to styled labels:

```python
@dataclass(frozen=True)
class StyledShapesLoadResult:
    layer: Shapes
    created: bool
    value_kind: Literal["categorical", "continuous"]
    palette_source: Literal["stored", "default_missing", "default_invalid"] | None
    coercion_applied: bool
```

For categorical columns, `palette_source` should report whether colors came
from a valid stored companion palette (`"stored"`), whether no usable companion
palette existed (`"default_missing"`), or whether a companion palette was found
but rejected as invalid or conflicting (`"default_invalid"`). For continuous
columns, `palette_source` should be `None`.

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

### Slice 2: Shapes Layer Loading

Status: completed

Implement:

- add `ShapesLayerBinding`;
- add adapter lookup, registration, and removal paths for shapes layers;
- implement GeoDataFrame to napari `Shapes` conversion in the selected
  coordinate system;
- support polygons, multipolygons, and `Point` + `radius` circles;
- apply coordinate-system transforms by materializing transformed geometries;
- set `layer.metadata["source_shapes_index_by_row"]` as the initial
  rendered-row to source-row mapping;
- populate `layer.features` only with the source index column, using the
  GeoDataFrame index name or `"index"` fallback, so the status bar can expose
  source identity without copying every source column;
- skip empty, invalid, or unsupported geometries with explicit feedback instead
  of failing the whole load when at least one shape can be rendered;
- preserve polygon interiors by encoding holes as embedded rings in the napari
  polygon path;
- activate the loaded shapes layer after `Add / Update`.

Recommended tests:

- polygons load as a napari `Shapes` layer;
- multipolygon rows create multiple napari shapes and duplicate their source row
  in `metadata["source_shapes_index_by_row"]`;
- circle rows render as napari ellipses;
- `layer.features` contains only the source index column, including named
  GeoDataFrame indexes and the `"index"` fallback for unnamed indexes;
- empty or invalid geometries are skipped with warning feedback;
- polygon interiors render as holes rather than filled exterior-only polygons;
- loading the same shapes element twice reuses the existing shapes layer;
- removing layers for a coordinate system also removes shapes layers;
- object-classification labels-layer signals are not emitted for shapes.

### Slice 3: Move Shapes Row Mapping Into Bindings

Status: completed

Slice 2 made shapes loadable, but it still uses `layer.metadata` for some
Harpy-internal shapes contracts. Before adding styled shapes layers, move those
contracts into `ShapesLayerBinding` so later styling and annotation work is not
dependent on a loose napari metadata dictionary.

Implement:

- first split the public adapter registration API into typed entrypoints:
  - `ViewerAdapter.register_labels_layer(...)`;
  - `ViewerAdapter.register_image_layer(...)`;
  - `ViewerAdapter.register_shapes_layer(...)`;
- keep a shared private helper for the actual registry insertion and common
  side effects, so signal emission and viewer-loaded bookkeeping stay in one
  place;
- keep `unregister_layer(...)` generic because unregistering only needs the
  live napari layer object;
- keep binding lookup/filtering registry-backed;
- extend `ShapesLayerBinding` with:
  - `source_shapes_index_by_row: tuple[Any, ...] = ()`;
  - `source_shapes_index_feature_name: str = "index"`;
  - `skipped_geometry_count: int = 0`;
- treat `ShapesLayerBinding.source_shapes_index_by_row` as the authoritative
  rendered-napari-row to source-`GeoDataFrame`-index mapping;
- treat `ShapesLayerBinding.source_shapes_index_feature_name` as the name of
  the `layer.features` column that stores the source index for napari status
  display and manual inspection;
- treat `ShapesLayerBinding.skipped_geometry_count` as the authoritative source
  for skipped-geometry viewer feedback;
- keep `layer.features[source_shapes_index_feature_name]` as the
  napari-visible source-index feature table;
- remove all code paths that read
  `layer.metadata["source_shapes_index_by_row"]` for Harpy logic;
- stop writing shapes source-row mappings and source-index feature names into
  `layer.metadata`;
- keep skipped-geometry feedback out of shapes layer metadata as well; read it
  from the shapes binding when the viewer widget builds action feedback;
- update `ViewerAdapter.ensure_shapes_loaded(...)` so it registers the source
  row mapping, source-index feature name, and skipped-geometry count through
  `register_shapes_layer(...)` when the primary shapes layer is created;
- keep `ensure_shapes_loaded(...)` reuse behavior registry-backed, so an
  existing registered shapes layer returns the same binding-backed contract;
- keep object-classification labels-layer signals unchanged.

Out of scope:

- shape-column source discovery;
- styled shapes layers;
- `ShapeColorSourceSpec`;
- primary/styled shapes roles;
- shape coloring.

Recommended tests:

- `ShapesLayerBinding.source_shapes_index_by_row` matches the rendered napari
  rows for polygons, multipolygons, and circles;
- `ShapesLayerBinding.source_shapes_index_feature_name` is `"index"` for
  unnamed GeoDataFrame indexes and the GeoDataFrame index name for named
  indexes;
- `layer.features[source_shapes_index_feature_name]` still exposes source
  indices for the status bar;
- shapes layers do not store source-row mappings, source-index feature names,
  shapes roles, style specs, or skipped-geometry counts in `layer.metadata`;
- typed registration entrypoints create the expected binding subclasses and
  preserve existing primary-labels signal behavior;
- viewer skipped-geometry feedback still works without reading layer metadata;
- reusing an already-loaded shapes layer preserves the binding-backed mapping;
- removing shapes layers unregisters the binding and its source-row mapping.

### Slice 4: Styling Module Refactor

Status: proposed

Purpose:

Prepare for shape-column styling by splitting the current labels overlay styling
module into labels-specific code and reusable viewer styling primitives, without
changing styled-label behavior.

Implement:

- rename `viewer/overlay_styling.py` to `viewer/labels_styling.py`;
- update imports to the new module and remove `viewer/overlay_styling.py`
  rather than keeping a compatibility shim;
- keep labels-specific public API in `viewer/labels_styling.py`:
  - `StyledLabelsStyleResult`;
  - `StyledLabelsLoadResult`;
  - `apply_table_color_source_to_labels_layer(...)`;
  - `build_styled_labels_layer_name(...)`;
  - labels-only table alignment, feature-table construction, and
    `DirectLabelColormap` application helpers;
- create `viewer/_styling.py` for internal reusable styling primitives:
  - shared palette-source literal, for example
    `Literal["stored", "default_missing", "default_invalid"]`;
  - missing categorical and continuous colors;
  - continuous colormap name;
  - category-value normalization;
  - string coercion and high-cardinality warning helpers that are useful for
    viewer styling;
  - color validation;
  - default categorical palette materialization;
  - continuous value normalization and color materialization;
- keep `viewer/_styling.py` free of napari layer classes, `SpatialData`, and
  `AnnData`, so it can be reused by labels and shapes without depending on one
  layer type's alignment model;
- make `viewer/labels_styling.py` call the shared helpers from
  `viewer/_styling.py` while preserving the current labels behavior and
  feedback semantics.

Out of scope:

- `ShapeColorSourceSpec`;
- `viewer/shapes_styling.py`;
- shape-column source discovery;
- styled shapes layers;
- shape coloring.

Recommended tests:

- existing styled-labels tests pass unchanged after imports move from
  `viewer.overlay_styling` to `viewer.labels_styling`;
- no imports of `viewer.overlay_styling` remain;
- no `viewer/overlay_styling.py` compatibility shim remains;
- styled-label palette feedback still reports `stored`, `default_missing`, and
  `default_invalid` exactly as before;
- styled-label string/object coercion warnings and feedback remain unchanged;
- labels-only colormap application still uses `DirectLabelColormap`.

### Slice 5: Color Source Module Refactor

Status: proposed

Purpose:

Move shared color-source datatypes out of the table-specific module before
adding shape-column color sources. This keeps the later `ShapeColorSourceSpec`
addition small and avoids making `table_color_source.py` a mixed table/shapes
module.

Implement:

- create `core/_color_source.py` as the shared color-source module:
  - move the existing `TableColorSourceSpec`, `ColorValueKind`, and source-kind
    aliases there;
  - rename any table-only source-kind alias to an explicit
    `TableColorSourceKind`, if needed;
  - update all imports from `napari_harpy.core.table_color_source` to
    `napari_harpy.core._color_source`;
  - remove `core/table_color_source.py` instead of keeping a compatibility
    shim;
- keep behavior unchanged for existing table-backed labels coloring and table
  color-source discovery;
- keep viewer styling helpers in `viewer/_styling.py`, not in
  `core/_color_source.py`.

Out of scope:

- `ShapeColorSourceSpec`;
- shape-column source discovery;
- styled shapes layers;
- shape coloring.

Recommended tests:

- existing table color-source behavior still passes after imports move to
  `core/_color_source.py`;
- no imports of `napari_harpy.core.table_color_source` remain;
- no `core/table_color_source.py` compatibility shim remains;
- styled-label coloring behavior remains unchanged after the import move.

### Slice 6: Shape-Column Coloring

Status: proposed

Follow the styled-labels pattern, but use direct columns on the shapes
`GeoDataFrame` instead of linked tables. A shapes card should have a `Color
source` selector:

- `None` means `Add / Update` loads or reuses the primary shapes layer;
- `Shape column` means `Add / Update` loads or updates a separate styled shapes
  layer variant.

The second control should be labelled `Shape column`, not `Observations` or
`Vars`, because the source is a column on `sdata.shapes[shapes_name]` itself.

The primary/styled distinction is required for the future
`ShapesAnnotation()` widget: annotation should listen to and edit primary
shapes layers only, while styled shapes layers are viewer-only variants.

Implementation reuse guidance:

- follow the styled-labels adapter and widget structure for layer roles,
  `style_spec` identity, created-vs-updated feedback, and source selector UI;
- do not call the labels-specific `apply_table_color_source_to_labels_layer`
  path for shapes, because it assumes linked `AnnData`, instance-key alignment,
  and `DirectLabelColormap`;
- reuse the domain-neutral helpers introduced in Slice 4 for palette-source
  reporting, category-value normalization, color validation, default
  categorical palettes, string/object categorical warnings, and continuous
  value normalization;
- keep shapes-specific styling in `viewer/shapes_styling.py`, because shapes
  use direct `GeoDataFrame` columns, row-level companion `<column>_colors`, and
  one `face_color` / `edge_color` value per rendered napari shape row.

Implement:

- add `ShapeColorSourceSpec` in `core/_color_source.py`, modelled after
  `TableColorSourceSpec` but scoped to direct shapes columns:
  - `source_kind: Literal["shape_column"]`;
  - `value_key: str`;
  - `value_kind: Literal["categorical", "continuous"]`;
- create `viewer/shapes_styling.py` for shapes-specific styling:
  - `StyledShapesStyleResult`;
  - `StyledShapesLoadResult`;
  - `apply_shape_color_source_to_shapes_layer(...)`;
  - `build_styled_shapes_layer_name(...)`;
  - shapes-only source-row alignment and `face_color` / `edge_color`
    application helpers;
- add shape-column source discovery for `sdata.shapes[shapes_name]`:
  - include scalar columns that can be classified as categorical or continuous;
  - exclude the geometry column;
  - exclude explicit color/palette columns, including columns ending in
    `_colors`, `_color`, or `.color`;
  - keep companion `<column>_colors` columns available only as palettes for
    their corresponding value column;
- extend the shapes card UI:
  - add `Color source` with `None` and `Shape column`;
  - add a searchable/autocompleted `Shape column` input populated from
    `ShapeColorSourceSpec` options;
  - show `Action: add/update primary shapes layer` when `Color source = None`;
  - show `Action: add/update styled shapes layer for shape["<column>"]` when a
    shape column is selected;
- extend `ShapesLoadRequest` so it can carry an optional selected
  `ShapeColorSourceSpec`;
- dispatch from the viewer widget:
  - no selected shape color source -> `ViewerAdapter.ensure_shapes_loaded(...)`;
  - selected shape color source -> `ViewerAdapter.ensure_styled_shapes_loaded(...)`;
- extend `ShapesLayerBinding` with:
  - `shapes_role: Literal["primary", "styled"] = "primary"`;
  - `style_spec: ShapeColorSourceSpec | None = None`;
- enforce the binding invariant:
  - primary shapes bindings must have `shapes_role="primary"` and
    `style_spec is None`;
  - styled shapes bindings must have `shapes_role="styled"` and a non-`None`
    `ShapeColorSourceSpec`;
- reuse `source_shapes_index_by_row` and `source_shapes_index_feature_name`
  from the Slice 3 binding contract for both primary and styled shapes layers;
- keep `ensure_shapes_loaded(...)` as the primary-layer path and make all
  primary lookup/removal code filter `shapes_role="primary"`;
- add styled-shapes adapter paths matching styled labels:
  - `get_loaded_styled_shapes_layer(...)`;
  - `get_loaded_styled_shapes_layers(...)`;
  - `ensure_styled_shapes_loaded(...)`;
  - a user-facing layer name such as
    `build_styled_shapes_layer_name(shapes_name, style_spec)`;
- allow primary and styled shapes layers for the same shapes element to coexist;
- register styled shapes layers with `shapes_role="styled"` and `style_spec`;
- keep styled shapes identity and source-row mappings in `ShapesLayerBinding`,
  not in `layer.metadata`;
- style shapes by categorical and continuous columns:
  - apply one color per rendered napari shape row;
  - prefer coloring both `face_color` and `edge_color`, with a readable edge and
    a translucent face so image data remains visible underneath;
  - use a neutral missing color for missing values;
- use valid `<column>_colors` companion columns as stored categorical palettes:
  - build the category-to-color mapping from the full source shape column before
    expanding multipolygons;
  - if a category has conflicting companion colors, fall back to the default
    categorical palette;
  - if the companion palette is missing, incomplete, or invalid, fall back to
    the default categorical palette;
- classify shape columns like styled labels:
  - pandas categorical, bool, and exact binary integer columns are categorical;
  - non-binary integer and float columns are continuous;
  - string/object scalar columns are temporarily categorical without mutating
    the `GeoDataFrame`;
- use `ShapesLayerBinding.source_shapes_index_by_row` to align source
  shape-column values to rendered napari rows;
- add the selected style source column to the `layer.features` DataFrame,
  aligned to rendered napari rows, instead of copying all GeoDataFrame columns
  during Slice 2;
- if the selected style source column name collides with the source-index
  feature column, store it in a deterministic disambiguated feature column
  without overwriting the source-index feature;
- repeat colors and feature values for multipolygon parts by repeating the
  source row lookup;
- provide feedback for primary vs styled load paths, created vs updated styled
  layers, palette source, invalid companion palettes, string coercion, and
  skipped geometries.

Recommended tests:

- shapes cards expose `Color source = None | Shape column` and a `Shape column`
  autocomplete populated from the shapes element;
- geometry and explicit color/palette columns are hidden from the shape-column
  selector;
- `Add / Update` with no selected shape column dispatches to the primary
  `ensure_shapes_loaded(...)` path;
- `Add / Update` with a selected shape column dispatches to
  `ensure_styled_shapes_loaded(...)`;
- primary and styled shapes layers can coexist for the same shapes element;
- styled shapes lookup reuses a matching variant and creates distinct variants
  for different columns;
- styled shapes bindings have `shapes_role="styled"` and a non-`None`
  `ShapeColorSourceSpec`;
- primary shapes bindings have `shapes_role="primary"` and `style_spec is None`;
- primary and styled shapes bindings carry `source_shapes_index_by_row` and
  `source_shapes_index_feature_name`;
- shapes layers do not store shapes roles, style specs, source mappings, or
  source-index feature names in `layer.metadata`;
- categorical columns produce per-shape categorical colors;
- categorical columns use valid `<column>_colors` companion palettes;
- invalid companion color values fall back to the default palette;
- categories with conflicting companion colors fall back to the default
  palette;
- bool and exact binary integer columns are categorical;
- non-binary integer and float columns are continuous;
- string/object scalar columns are temporarily categorical without mutating the
  `GeoDataFrame`;
- the selected style source column is added to `layer.features` and repeated
  for multipolygon parts;
- style source feature-name collisions are disambiguated without overwriting the
  source-index feature;
- multipolygon parts repeat the source row color via
  `ShapesLayerBinding.source_shapes_index_by_row`;
- mixed unsupported columns are hidden from the selector;
- styled layer identity includes the selected column and is reused on repeat.

### Slice 7: Explicit Shape-Table Coloring

Status: not planned

Decision: do not implement explicit shape-table coloring in this roadmap.
The notes below are retained as historical context only.

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

## Shape Annotation And Write-Back

Shape annotation and write-back are tracked in
`Roadmap/shapes_elements/annotation_shapes.md`.

That separate roadmap introduces a dedicated `ShapesAnnotation()` widget for
creating new shapes annotations and modifying existing shapes elements. The
viewer roadmap should remain focused on loading, viewing, and coloring shapes.

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
- Table-backed shapes coloring is not exposed in this roadmap.

## Non-Goals For The First Version

- editing shapes geometry in napari and writing it back to SpatialData;
- rich legends or palette editors;
- using labels-linked tables to color shapes by implicit id matching;
- cross-element biological-object identity resolution.

## References

- SpatialData object docs:
  https://spatialdata.scverse.org/en/latest/api/SpatialData.html
- SpatialData annotation/table tutorial:
  https://spatialdata.scverse.org/en/latest/tutorials/notebooks/notebooks/examples/tables.html
- SpatialData from-scratch tutorial, connecting `AnnData` to shapes:
  https://spatialdata.scverse.org/en/stable/tutorials/notebooks/notebooks/examples/sdata_from_scratch.html
