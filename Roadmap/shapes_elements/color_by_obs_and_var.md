# Color Shapes By Linked Table `.obs` And `X[:, var]`

Status: investigation

## Goal

Support coloring a `sdata.shapes[...]` element by values stored in an
`AnnData` table that explicitly annotates that shapes element.

The useful user-facing cases are:

- table `.obs` columns, for example `cell_type`, `leiden`, `quality_score`;
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

For shapes, `instance_key` should align to `sdata.shapes[shapes_name].index`.
Unlike labels, shape indices can be strings or other non-integer labels. Do not
reuse labels-only validation that coerces instance IDs to positive integers.

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
- table obs: `cell_boundaries[table:table/obs:cell_type]`
- table var: `cell_boundaries[table:table/X:GeneA]`

The table name should be part of styled layer identity and the displayed name,
because multiple tables can annotate the same shapes element and expose the
same column names.

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
today. There is no direct shapes equivalent of napari's label colormap. For the
first table-backed shapes slice, exclude the table `instance_key` from shapes
table color options unless we add a deliberate "shape instance colors" mode.

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
- exclude `instance_key` for the first implementation, or expose it only after
  adding shape-specific instance color semantics;
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
5. Preserve `instance_key` values as labels; do not coerce to integers.
6. Require table `instance_key` values to be unique within the selected shapes
   region. Duplicates are ambiguous and should raise `ValueError`.
7. Build a source-level `pd.Series` indexed by shape instance ID:
   - for `.obs`, use `table.obs[value_key]`;
   - for `X[:, var_name]`, extract the selected `X` column at the matching
     table observation positions.
8. Align that series to `shapes_element.index`.
9. Align again to `source_shapes_index_by_row`, repeating values for
   MultiPolygon parts.
10. Apply colors to `layer.face_color` and `layer.edge_color` with the existing
    shape alpha behavior.
11. Add the selected table value to `layer.features` for hover/status display.

Missing table rows:

- If no table instances overlap with the shapes index, raise a clear error.
- If some shapes have no matching table row, keep rendering them with the
  existing missing gray color and consider reporting the missing count in the
  status card.

The all-missing case should not silently create an all-gray styled layer,
because it usually means the table metadata points at the shapes element but
the instance IDs use a different convention than the GeoDataFrame index.

## Styling Semantics

For table-backed shape `.obs` sources, reuse labels overlay semantics:

- pandas categorical -> categorical;
- bool -> categorical;
- exact binary integer `{0, 1}` -> categorical;
- other integer/float -> continuous;
- string-like object columns -> temporary categorical with warning;
- unsupported object/mixed columns -> not offered.

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
  - extract shared categorical/continuous rendered-row application helpers so
    both direct and table paths share as much as possible;
  - reuse or move labels table palette resolution so table-backed shapes and
    labels behave identically.
- `viewer/adapter.py`
  - broaden styled-shapes `style_spec` typing;
  - in `ensure_styled_shapes_loaded(...)`, dispatch to the direct shape-column
    styler or table-backed styler based on spec type/source kind;
  - include table source identity in lookup and layer names.
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
- table `region_key` and shape `instance_key` are not offered as ordinary
  first-pass table shape color options;
- table `.obs` and `X[:, var_name]` sources produce stable identities including
  table name.

Styling tests:

- table `.obs` categorical colors use `table.uns["<column>_colors"]`;
- invalid/missing table palettes fall back like labels overlays;
- table `.obs` string/object values coerce to temporary categorical values;
- table `.obs` continuous values use the continuous colormap;
- `X[:, var_name]` values color shapes continuously for dense and sparse `X`;
- string shape indices align without integer coercion;
- MultiPolygon-expanded rendered rows repeat the table value for each rendered
  part;
- duplicate table `instance_key` values within the shapes region raise a clear
  error;
- zero overlap between table instances and shapes index raises a clear error;
- partial overlap renders unmatched shapes gray.

Adapter/widget tests:

- shapes cards expose `None | Shapes column | Observations | Vars`;
- shapes cards show `No linked tables` when none annotate the shapes element;
- selecting `Observations` or `Vars` dispatches to
  `ViewerAdapter.ensure_styled_shapes_loaded(...)` with a `TableColorSourceSpec`;
- styled shape layer names differ for direct shape column, table `.obs`, and
  `X[:, var_name]`;
- styled shape variants for two tables with the same column name coexist;
- fill toggling reuses the same styled table-backed shapes layer;
- status feedback reports table source names and palette fallback/coercion
  behavior.

## Suggested Slices

1. Discovery and model plumbing
   - add shape table discovery helpers;
   - broaden styled-shapes style-spec typing;
   - add tests without changing rendering behavior yet.

2. Table-backed styling core
   - implement table-to-shape alignment;
   - add `apply_table_color_source_to_shapes_layer(...)`;
   - cover categorical, continuous, `X[:, var_name]`, string index, duplicate
     instance IDs, and MultiPolygon repetition.

3. Adapter and layer lifecycle
   - dispatch styled shapes to direct or table-backed styler;
   - update layer naming and lookup;
   - ensure table-backed variants coexist with direct shape-column variants.

4. Widget and feedback
   - add linked table and `Observations`/`Vars` controls to shapes cards;
   - update error/status messages;
   - add widget tests for disabled and linked-table states.

## Open Questions

- Should we ever expose `instance_key` as a shape color source? It is useful as
  an identity view, but labels' `instance` colormap does not transfer directly
  to shapes. First implementation should probably exclude it.
- Should partial table coverage be warning-level feedback with a missing count?
  This would help users catch incomplete annotation tables without blocking
  useful partial visualization.
- Should table-backed values be added to `layer.features` under just
  `value_key`, or a disambiguated name like `table.obs.cell_type`? The direct
  shape-column path already handles collisions with the source index feature;
  table-backed sources additionally need to avoid collisions with direct shape
  columns and other table variants.
- If a shapes element has an integer index and a table stores instance IDs as
  strings, should Harpy attempt string-normalized matching? Safer default is
  exact matching plus a clear error on zero overlap.

## Recommendation

Implement this as a follow-up to the existing direct shape-column coloring path,
not as a replacement for it.

The best first version is:

- only tables that explicitly annotate the selected shapes element;
- `.obs` columns and `X[:, var_name]` values;
- exact shape-index alignment;
- direct reuse of existing shape rendered-row coloring;
- labels-compatible table palette behavior;
- no labels-table inference and no direct `.var` metadata coloring.
