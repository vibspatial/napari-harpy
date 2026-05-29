# Color Shapes By Linked Table `.obs` And `X[:, var]`

Status: investigation

## Goal

Support coloring a `sdata.shapes[...]` element by values stored in an
`AnnData` table that explicitly annotates that shapes element.

The useful user-facing cases are:

- table `.obs` columns, for example `cell_type`, `leiden`, `quality_score`;
- the table `instance_key`, shown as an observation source but rendered with
  instance/identity colors rather than as a categorical or continuous value;
- table expression/feature values, represented like the current labels path as
  `X[:, var_name]`, where `var_name` comes from `table.var_names`.

Important terminology note: the existing viewer code says `Vars`, but it does
not color by scalar metadata columns in `table.var`. It colors each spatial
instance by the corresponding value in `table.X[:, var_name]`. Direct
`table.var[...]` columns describe variables/features, not shapes rows, so they
are not directly colorable per shape without a second aggregation rule.

## Current State

Shape coloring already exists, but only for scalar columns stored directly on
the shapes `GeoDataFrame`.

Implemented pieces:

- `core/_color_source.py`
  - `ShapeColorSourceSpec` supports only `source_kind="shape_column"`;
  - `TableColorSourceSpec` supports `source_kind="obs_column"` and `"x_var"`,
    but it is currently used by labels overlays.
- `core/spatialdata.py`
  - `get_shape_column_color_source_options(...)` discovers direct shape
    columns;
  - `get_table_color_source_options(...)` discovers table `.obs` and
    `X[:, var_name]` sources;
  - `get_annotating_table_names(...)` currently has labels-oriented naming, but
    it calls SpatialData's generic `get_element_annotators(...)`.
- `viewer/shapes_styling.py`
  - `apply_shape_color_source_to_shapes_layer(...)` applies direct
    `GeoDataFrame` column values;
  - it already handles rendered-row alignment for MultiPolygons through
    `source_shapes_index_by_row`;
  - categorical values can use a row-level companion `<column>_colors` shape
    column;
  - string/object columns are coerced to temporary categorical values;
  - continuous values use the shared `viridis` path;
  - missing values render gray.
- `viewer/labels_styling.py`
  - `apply_table_color_source_to_labels_layer(...)` is the closest existing
    implementation for table-backed coloring;
  - it aligns table rows to label instances through `region_key` and
    `instance_key`;
  - current labels styling silently drops duplicate instance IDs with
    `keep="last"` during alignment, which should be fixed so duplicates within
    a selected region raise clearly;
  - categorical `.obs` columns use `table.uns["<column>_colors"]`;
  - `X[:, var_name]` values are continuous.
- `viewer/adapter.py`
  - primary and styled shapes layers can coexist;
  - styled shapes layer identity is currently keyed by `ShapeColorSourceSpec`;
  - styled shape layers keep the source shape index mapping in
    `ShapesLayerBinding`.
- `widgets/viewer/shapes_widget.py`
  - shapes cards expose `Color source = None | Shapes column`;
  - there is no linked-table selector for shapes.

Conclusion: the geometry/rendered-row machinery is already the hard part, and
it is in place. The missing work is mostly table discovery, a shape-specific
table alignment function, broader style-spec typing, and UI plumbing.

## SpatialData Semantics

Only offer table-backed shape coloring when a table explicitly annotates the
selected shapes element.

SpatialData table metadata gives us:

- `region_key`: the `table.obs` column naming the target spatial element for
  each row;
- `instance_key`: the `table.obs` column naming the instance inside that
  element;
- `region`: the declared element name or names the table annotates.

For shapes, table `obs[instance_key]` should align only to the
`sdata.shapes[shapes_name][instance_key]` column. Do not fall back to
`sdata.shapes[shapes_name].index`, because implicit index matching can produce
surprising styling when the GeoDataFrame index is just storage/order metadata.
Use exact value matching: do not coerce strings to numbers, normalize IDs, or do
fuzzy matching between table and shape instance values.

The GeoDataFrame index remains only a render-row bookkeeping key: it maps each
napari-rendered shape row back to its source GeoDataFrame row, especially when a
single source geometry expands into multiple rendered rows. It must not be used
as the semantic join key to the table.

Coverage is intentionally asymmetric. Table rows for the selected shapes region
must refer only to instances present in `shapes_element[instance_key]`; table
instances that are not present in the shapes element should raise clearly.
Shapes instances that have no table row are allowed and should render with the
missing color.

Unlike labels, shape instance values can be strings or other non-integer labels.
Do not reuse labels-only validation that coerces instance IDs to positive
integers.

Do not infer that a labels-linked table also annotates shapes because IDs happen
to match. A table that annotates `blobs_labels` should not appear as a color
source for `cell_boundaries` unless its SpatialData metadata also lists
`cell_boundaries`.

## Recommended User Experience

Extend each shapes card to mirror the labels card where table-backed sources
are available:

- `Linked table`
  - enabled only when `get_element_annotators(sdata, shapes_name)` returns one
    or more tables;
  - hidden or disabled with `No linked tables` when none exist.
- `Color source`
  - `None`;
  - `Shapes column`;
  - `Observations`;
  - `Vars`.
- `Value source`
  - direct shape column names when `Shapes column` is selected;
  - table `.obs` columns when `Observations` is selected;
  - `table.var_names` when `Vars` is selected.
- `Fill`
  - keep the existing behavior: enabled for styled shape sources, disabled for
    primary shapes.

Suggested action text:

- `Action: add/update primary shapes layer`
- `Action: add/update styled shapes layer for column "leiden"`
- `Action: add/update styled shapes layer for obs["cell_type"]`
- `Action: add/update styled shapes layer for X[:, "GeneA"]`
- `Action: table-backed coloring requires a linked table`

Layer names should remain stable and include the source kind:

- shape column: `cell_boundaries[shape:leiden]`
- table obs: `cell_boundaries[obs:cell_type]`
- table var: `cell_boundaries[X:GeneA]`

The table name should be part of styled layer identity, because multiple tables
can annotate the same shapes element and expose the same column names. It should
not be included in the displayed napari layer name.

## Data Model Changes

The current separation between `ShapeColorSourceSpec` and
`TableColorSourceSpec` is still useful. Do not add table fields to
`ShapeColorSourceSpec`.

Recommended typing:

```python
SpatialElementColorSourceSpec = ShapeColorSourceSpec | TableColorSourceSpec
```

Then update shape-facing contracts that can now accept either direct shape
columns or table-backed sources:

- `ShapesLayerBinding.style_spec`
- `LayerBindingRegistry.find_bindings(..., style_spec=...)`
- `ViewerAdapter.get_loaded_styled_shapes_layer(...)`
- `ViewerAdapter.ensure_styled_shapes_loaded(...)`
- `ShapesLoadRequest.selected_color_source`
- status-card helpers for styled shapes.

Alternative: create a new dataclass such as `ShapesTableColorSourceSpec`.
That is more explicit, but likely unnecessary because `TableColorSourceSpec`
already contains the table name, source kind, value key, and value kind needed
for table-backed shape coloring.

One caveat: `TableColorSourceSpec.value_kind == "instance"` is labels-specific
today, but the same value kind should be used for table-backed shapes when the
selected observation source is the table `instance_key`. For shapes, this means
identity coloring: one color per shape instance, not categorical or continuous
coloring of arbitrary table values.

## Discovery Helpers

Add shape-specific table helpers in `core/spatialdata.py`.

Recommended new functions:

```python
def get_shape_annotating_table_names(sdata: SpatialData, shapes_name: str) -> list[str]:
    ...

def get_shape_table_color_source_options(
    sdata: SpatialData,
    shapes_name: str,
    table_name: str,
) -> list[TableColorSourceSpec]:
    ...

def get_shape_color_source_options(
    sdata: SpatialData,
    shapes_name: str,
) -> list[ShapeColorSourceSpec | TableColorSourceSpec]:
    ...
```

Discovery rules:

- use `get_element_annotators(sdata, shapes_name)` for linked tables;
- validate that table metadata `annotates(shapes_name)`;
- reuse the existing table `.obs` and `X[:, var_name]` source classification;
- exclude `region_key` as before;
- include `instance_key` as an observation source with `value_kind="instance"`;
- keep direct shape-column discovery unchanged.

Some existing names are labels-specific but generic in implementation:

- `SpatialDataTableMetadata.annotates(labels_name)` can be renamed internally to
  `annotates(element_name)`;
- `get_annotating_table_names(sdata, labels_name)` can either be renamed to
  `get_element_annotating_table_names(...)` or wrapped by separate labels and
  shapes helper names to keep call sites readable.

Avoid reusing:

- `validate_table_binding(...)` for shapes, because it enforces labels-specific
  integer instance IDs;
- `_get_region_rows_by_instance(...)` from labels styling, because it drops
  non-positive numeric IDs and casts the instance index to `int64`.

Related labels fix:

- labels table alignment should stop silently dropping duplicate
  `instance_key` values with `keep="last"` and should instead raise when
  duplicates occur within the selected labels region, matching the intended
  shape behavior.

## Alignment Algorithm

Add a shape-specific table alignment function. Its output should be one value
per rendered napari shape row, because a source `MultiPolygon` can expand into
multiple napari rows.

Inputs:

- `sdata`
- `shapes_name`
- source shapes `GeoDataFrame`
- `TableColorSourceSpec`
- `source_shapes_index_by_row`
- `source_shapes_index_feature_name`

Recommended steps:

1. Load and validate the table with `get_table(...)` and `get_table_metadata(...)`.
2. Check `table_metadata.annotates(shapes_name)`.
3. Filter table rows where `table.obs[region_key] == shapes_name`.
4. Drop rows with missing `instance_key`.
5. Require the shapes element to contain a column named `instance_key`; if it is
   missing, raise a clear error instead of falling back to the GeoDataFrame
   index.
6. Preserve table and shapes `instance_key` values as labels; do not coerce to
   integers.
7. Require table `instance_key` values to be unique within the selected shapes
   region. Duplicates are ambiguous and should raise `ValueError`.
8. Build a table-level `pd.Series` indexed by table `instance_key` value:
   - for `instance_key`, use the aligned instance labels themselves;
   - for `.obs`, use `table.obs[value_key]`;
   - for `X[:, var_name]`, extract the selected `X` column at the matching
     table observation positions.
9. Validate that every table `instance_key` value for the selected shapes region
   exists in `shapes_element[instance_key]`; extra table instances should raise
   a clear error.
10. Align table values to source GeoDataFrame rows by mapping each source row to
    its `shapes_element[instance_key]` value. Shapes with no matching table row
    are allowed and receive missing values.
11. Align again to `source_shapes_index_by_row`, repeating values for
    MultiPolygon parts.
12. Apply colors to `layer.face_color` and `layer.edge_color` with the existing
    shape alpha behavior.
13. Add the selected table value to `layer.features` for hover/status display.

Missing table rows:

- If no table instances overlap with `shapes_element[instance_key]`, raise a
  clear error.
- If any table instances for the selected shapes region are missing from
  `shapes_element[instance_key]`, raise a clear error.
- If some shapes have no matching table row, keep rendering them with the
  existing missing gray color and consider reporting the missing count in the
  status card.

The all-missing case should not silently create an all-gray styled layer,
because it usually means the table metadata points at the shapes element but
the instance IDs use a different convention than the shapes `instance_key`
column.

## Styling Semantics

For table-backed shape `.obs` sources, reuse labels overlay semantics:

- the table `instance_key` -> instance/identity coloring;
- pandas categorical -> categorical;
- bool -> categorical;
- exact binary integer `{0, 1}` -> categorical;
- other integer/float -> continuous;
- string-like object columns -> temporary categorical with warning;
- unsupported object/mixed columns -> not offered.

For the table `instance_key` source:

- mirror labels conceptually: this is a special identity-coloring branch, not a
  table-value categorical or continuous colormap;
- do not use `table.uns["<instance_key>_colors"]`;
- set result metadata to `value_kind="instance"`, `palette_source=None`, and
  `coercion_applied=False`;
- for numeric shape instance IDs compatible with napari labels, the colors can
  follow napari's cyclic `label_colormap(...)` behavior;
- for string or otherwise non-numeric shape instance IDs, use an equivalent
  deterministic cyclic identity palette over the aligned instance labels;
- repeated rendered rows from a `MultiPolygon` source instance must receive the
  same instance color.

Palette behavior:

- categorical `.obs` columns should use `table.uns["<column>_colors"]` when
  valid, exactly like labels overlays;
- invalid or missing stored palettes fall back to the default categorical
  palette;
- string/object coercion should ignore stored palettes and use the default
  palette, matching labels and direct shape-column behavior;
- direct shape-column categorical sources should continue using row-level
  companion columns named `<column>_colors`.

For `X[:, var_name]` table sources:

- always treat as continuous;
- support dense and sparse `table.X`, matching labels;
- use the existing continuous color helper.

For missing values after alignment:

- use `SHAPES_MISSING_BASE_COLOR`;
- preserve existing alpha rules:
  - face alpha `0.0` by default;
  - face alpha `SHAPES_FACE_ALPHA` when `Fill` is enabled;
  - edge alpha `SHAPES_EDGE_ALPHA`.

## Implementation Sketch

Likely code changes:

- `core/_color_source.py`
  - add a union alias, for example
    `SpatialElementColorSourceSpec = ShapeColorSourceSpec | TableColorSourceSpec`;
  - decide whether to add shape-table-specific display helpers or keep those in
    the widget/status-card layer.
- `core/spatialdata.py`
  - add shape table discovery helpers;
  - add shape-specific table binding validation that preserves arbitrary shape
    index labels;
  - optionally make existing table annotation helper names element-generic.
- `viewer/shapes_styling.py`
  - keep `apply_shape_color_source_to_shapes_layer(...)` for direct columns;
  - add `apply_table_color_source_to_shapes_layer(...)`;
  - add a shape-specific instance-coloring branch for table `instance_key`
    sources;
  - extract shared categorical/continuous rendered-row application helpers so
    both direct and table paths share as much as possible;
  - reuse or move labels table palette resolution so table-backed shapes and
    labels behave identically.
- `viewer/labels_styling.py`
  - fix the existing labels alignment shortcoming by raising on duplicate
    `instance_key` values within the selected region instead of silently keeping
    the last duplicate row.
- `viewer/adapter.py`
  - broaden styled-shapes `style_spec` typing;
  - in `ensure_styled_shapes_loaded(...)`, dispatch to the direct shape-column
    styler or table-backed styler based on spec type/source kind;
  - include table source identity in lookup while keeping displayed layer names
    table-name-free.
- `widgets/viewer/shapes_widget.py`
  - add linked table support;
  - allow table source kinds in `selected_source_kind`;
  - preserve direct shape-column behavior for existing tests.
- `widgets/viewer/widget.py`
  - pass shape-linked table names and table color-source options when rebuilding
    shapes cards;
  - update styled-shapes error messages for table-backed selections.
- `widgets/viewer/status_card.py`
  - format direct shape columns and table sources differently;
  - optionally report partial table alignment/missing-shape counts.

## Test Plan

Core/discovery tests:

- a table that annotates a shapes element is discovered for that shapes card;
- a table that annotates only labels is not offered for shapes;
- direct shape-column options remain available unchanged;
- table `region_key` is not offered as an ordinary color option;
- table `instance_key` is offered as an observation source with
  `value_kind="instance"`;
- table `.obs` and `X[:, var_name]` sources produce stable identities including
  table name.

Styling tests:

- table `.obs` categorical colors use `table.uns["<column>_colors"]`;
- table `instance_key` colors shapes by instance identity rather than by table
  categorical/continuous semantics;
- table `instance_key` identity coloring works for string shape instance values
  without integer coercion;
- table-backed styling raises clearly when the shapes element lacks the
  `instance_key` column and does not fall back to the GeoDataFrame index;
- invalid/missing table palettes fall back like labels overlays;
- table `.obs` string/object values coerce to temporary categorical values;
- table `.obs` continuous values use the continuous colormap;
- `X[:, var_name]` values color shapes continuously for dense and sparse `X`;
- string shape `instance_key` values align without integer coercion;
- MultiPolygon-expanded rendered rows repeat the table value for each rendered
  part;
- duplicate table `instance_key` values within the shapes region raise a clear
  error;
- table `instance_key` values for the selected shapes region that are absent
  from `shapes_element[instance_key]` raise a clear error;
- zero overlap between table instances and shapes `instance_key` values raises a
  clear error;
- shape instances without table rows render gray;
- labels duplicate table `instance_key` values within the selected labels
  region raise a clear error instead of being silently de-duplicated.

Adapter/widget tests:

- shapes cards expose `None | Shapes column | Observations | Vars`;
- shapes cards show `No linked tables` when none annotate the shapes element;
- selecting `Observations` or `Vars` dispatches to
  `ViewerAdapter.ensure_styled_shapes_loaded(...)` with a `TableColorSourceSpec`;
- styled shape layer names differ for direct shape column, table `.obs`, and
  `X[:, var_name]`;
- styled shape variants for two tables with the same column name coexist;
- fill toggling reuses the same styled table-backed shapes layer;
- status feedback reports table source names, instance coloring, and palette
  fallback/coercion behavior.

## Suggested Slices

1. Discovery and model plumbing
   - add shape table discovery helpers;
   - broaden styled-shapes style-spec typing;
   - add tests without changing rendering behavior yet.

2. Table-backed styling core
   - implement table-to-shape alignment;
   - add `apply_table_color_source_to_shapes_layer(...)`;
   - cover instance, categorical, continuous, `X[:, var_name]`, string index,
     duplicate instance IDs, and MultiPolygon repetition.
   - fix labels alignment so duplicate instance IDs within one selected region
     raise instead of silently keeping the last duplicate row.

3. Adapter and layer lifecycle
   - dispatch styled shapes to direct or table-backed styler;
   - update layer naming and lookup;
   - ensure table-backed variants coexist with direct shape-column variants.

4. Widget and feedback
   - add linked table and `Observations`/`Vars` controls to shapes cards;
   - update error/status messages;
   - add widget tests for disabled and linked-table states.

## Open Questions

- For non-numeric shape instance IDs, should identity colors be generated by
  enumerating aligned instance labels into napari's cyclic label palette, or by
  using the existing default categorical palette? Either way, the mode should
  behave as identity coloring and not consult table categorical palettes.
- Should partial table coverage be warning-level feedback with a missing count?
  This would help users catch incomplete annotation tables without blocking
  useful partial visualization.
- Should table-backed values be added to `layer.features` under just
  `value_key`, or a disambiguated name like `table.obs.cell_type`? The direct
  shape-column path already handles collisions with the source index feature;
  table-backed sources additionally need to avoid collisions with direct shape
  columns and other table variants.
- If a shapes element has string-like `instance_key` values and a table stores
  numeric-looking instance IDs, should Harpy attempt string-normalized matching?
  Safer default is exact value matching plus a clear error when table instances
  do not exist in the shapes element.

## Recommendation

Implement this as a follow-up to the existing direct shape-column coloring path,
not as a replacement for it.

The best first version is:

- only tables that explicitly annotate the selected shapes element;
- the table `instance_key`, other `.obs` columns, and `X[:, var_name]` values;
- exact value matching between `table.obs[instance_key]` and
  `shapes_element[instance_key]`, with table instances required to be a subset
  of shape instances;
- direct reuse of existing shape rendered-row coloring;
- labels-compatible table palette behavior;
- no labels-table inference and no direct `.var` metadata coloring.
