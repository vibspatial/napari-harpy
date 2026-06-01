# Color Shapes By Linked Table `.obs` And `X[:, var]`

Status: Slice 6 implemented

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
  - `ShapeColumnColorSourceSpec` supports direct shape-column sources with
    `source_kind="shape_column"`;
  - `TableColorSourceSpec` supports `source_kind="obs_column"` and `"x_var"`,
    and is the shared table-backed source model for labels and shapes.
- `core/spatialdata.py`
  - `get_shape_column_color_source_options(...)` discovers direct shape
    columns;
  - `get_table_color_source_options(...)` discovers table `.obs` and
    `X[:, var_name]` sources;
  - `get_annotating_table_names(sdata, element_name)` calls SpatialData's
    generic `get_element_annotators(...)`.
- `viewer/shapes_styling.py`
  - `apply_shape_column_color_source_to_shapes_layer(...)` applies direct
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
  - styled shapes layer identity accepts explicit
    `ShapeColumnColorSourceSpec | TableColorSourceSpec` unions;
  - `ensure_styled_shapes_loaded(...)` dispatches direct shape-column sources
    and table-backed sources to the appropriate styler;
  - styled shape layers keep the source shape index mapping in
    `ShapesLayerBinding`.
- `widgets/viewer/shapes_widget.py`
  - shapes cards expose `Color source = None | Shapes column | Observations |
    Vars`;
  - linked tables are discovered per shapes element, matching labels;
  - table-backed selections are represented in the UI and request model.

Conclusion: the geometry/rendered-row machinery is already the hard part, and
it is in place. The remaining work is mostly widget/status feedback polish for
table-backed shapes.

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
If table and shape instance IDs differ only by dtype or formatting, for example
`1` versus `"1"`, Harpy should still treat them as different IDs and raise a
clear error for table instances that do not exist in the shapes element.

The GeoDataFrame index remains only a render-row bookkeeping key: it maps each
napari-rendered shape row back to its source GeoDataFrame row, especially when a
single source geometry expands into multiple rendered rows. It must not be used
as the semantic join key to the table.

Coverage is intentionally asymmetric. Table rows for the selected shapes region
must refer only to instances present in `shapes_element[instance_key]`; table
instances that are not present in the shapes element should raise clearly.
Shapes instances that have no table row are allowed and should render
transparent, matching labels overlay behavior. Shapes that do have a table row
but whose selected value is missing should render with the missing color.

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

- shapes column: `cell_boundaries[shapes_column:leiden]`
- table obs: `cell_boundaries[obs:cell_type]`
- table var: `cell_boundaries[X:GeneA]`

The table name should be part of styled layer identity, because multiple tables
can annotate the same shapes element and expose the same column names. It should
not be included in the displayed napari layer name.

## Data Model Changes

The separation between direct shape-column sources and table-backed sources is
still useful. Do not add table fields to the direct shape-column dataclass.

Recommended typing:

```python
ShapeColorSourceKind = Literal["shape_column"]
ShapeColumnColorSourceSpec  # renamed from the former concrete ShapeColorSourceSpec
ShapeColumnColorSourceSpec | TableColorSourceSpec
```

Keep `ShapeColorSourceKind` even though it currently has only one allowed
value. It documents the direct shape-column source-kind domain and keeps the
shape-column dataclass parallel with the table-backed source typing.

Then update shape-facing contracts that can now accept either direct shape
columns or table-backed sources:

- `ShapesLayerBinding.style_spec`
- `LayerBindingRegistry.find_bindings(..., style_spec=...)`
- `ViewerAdapter.get_loaded_styled_shapes_layer(...)`
- `ViewerAdapter.ensure_styled_shapes_loaded(...)`
- `ShapesLoadRequest.selected_color_source`
- status-card helpers for styled shapes.
- `ShapesStyleResult` should carry optional table-alignment counts so Slice 6
  can report partial table coverage as informational feedback without rerunning
  alignment:
  ```python
  @dataclass(frozen=True)
  class ShapesStyleResult:
      value_kind: ShapeColorValueKind | None
      palette_source: StyledPaletteSource | None
      coercion_applied: bool
      unannotated_source_shape_count: int = 0
      unannotated_rendered_shape_count: int = 0
  ```
  Direct shape-column styling returns zeros for these counts. Table-backed
  styling fills them from `_ShapeTableRowAlignment.source_row_has_table_row`
  and `_ShapeTableRowAlignment.rendered_row_has_table_row`.

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

Mirror the current labels linked-table discovery flow for shapes.

Recommended discovery flow:

```python
table_names = get_annotating_table_names(sdata, shapes_name)
shape_column_color_sources = get_shape_column_color_source_options(sdata, shapes_name)
table_color_sources_by_table = {
    table_name: get_table_color_source_options(sdata, table_name)
    for table_name in table_names
}
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

Add a small shape-specific table alignment helper. Slice 3 should stop at
alignment: it should return aligned values and masks, but it should not apply
colors, mutate `layer.features`, or dispatch through the adapter yet. Those
steps belong to Slice 4+.

Avoid introducing a broad public alignment model. The current codebase keeps
alignment close to the styling function: direct shapes styling aligns inline in
`apply_shape_column_color_source_to_shapes_layer(...)`, and labels use a small private
helper returning only the pieces the styling function needs. Shape table
alignment should follow that pattern.

The helper output only needs to carry source-row and rendered-row views,
because a source `MultiPolygon` can expand into multiple napari rows:

```python
@dataclass(frozen=True)
class _ShapeTableRowAlignment:
    source_row_values: pd.Series
    rendered_row_values: pd.Series
    source_row_has_table_row: pd.Series
    rendered_row_has_table_row: pd.Series
```

Document these fields in the dataclass docstring or with adjacent inline
comments. In particular, make clear that `source_row_*` fields align to the
source `GeoDataFrame` rows, `rendered_row_*` fields align to napari shape rows
after MultiPolygon expansion, and `*_has_table_row` distinguishes unannotated
shapes from annotated rows whose selected table value is missing.

The table metadata, selected-region rows, table observation index, and
`value_by_instance` mapping can stay local implementation details unless Slice
4 shows they need to cross function boundaries.

Inputs:

- `shapes_name`
- validated table `AnnData`
- `SpatialDataTableMetadata`
- source shapes `GeoDataFrame`
- `TableColorSourceSpec`
- `source_shapes_index_by_row`

Recommended private helper shape:

```python
def _align_table_color_source_to_shapes_rows(
    *,
    table: AnnData,
    table_metadata: SpatialDataTableMetadata,
    shapes_name: str,
    shapes_element: gpd.GeoDataFrame,
    style_spec: TableColorSourceSpec,
    source_shapes_index_by_row: tuple[Any, ...],
) -> _ShapeTableRowAlignment:
    ...
```

Keep this helper in `viewer/shapes_styling.py`. It is analogous to labels'
`_get_region_rows_by_instance(...)`, but it is shape-specific because it uses
the rendered-row mapping from the shapes layer builder.

Recommended steps:

1. Check `table_metadata.annotates(shapes_name)`.
2. Require the shapes element to contain a column named `instance_key`; if it is
   missing, raise a clear error instead of falling back to the GeoDataFrame
   index.
3. Require the source GeoDataFrame index to be unique, because
   `source_shapes_index_by_row` maps rendered napari rows back through that
   index.
4. Filter table rows where `table.obs[region_key] == shapes_name`.
5. Require at least one table row for the selected shapes region after region
   filtering; if none remain, raise a clear error because the table metadata
   declares the shapes annotation but no rows annotate that shapes element.
6. Raise if any selected-region table row has a missing `instance_key`.
   Do not drop these rows silently.
7. Raise if any source shapes row in the GeoDataFrame has a missing
   `shapes_element[instance_key]` value. A missing shape instance value is
   malformed, while a present shape instance value with no matching table row is
   allowed and tracked separately.
8. Preserve table and shapes `instance_key` values as labels; do not coerce to
   integers.
9. Require table `instance_key` values to be unique within the selected shapes
   region. Duplicates are ambiguous and should raise `ValueError`.
10. Allow duplicate values in `shapes_element[instance_key]`. Multiple source
   geometries with the same shape instance identity should receive the same
   table value.
11. Build a table-level `pd.Series` indexed by table `instance_key` value:
   - for `instance_key`, use the aligned instance labels themselves;
   - for `.obs`, use `table.obs[value_key]`;
   - for `X[:, var_name]`, extract the selected `X` column at the matching
     table observation positions.
12. Validate that every table `instance_key` value for the selected shapes region
   exists in `shapes_element[instance_key]`; extra table instances should raise
   a clear error.
13. Align table values to source GeoDataFrame rows by mapping each source row to
    its `shapes_element[instance_key]` value.
14. Build `source_row_has_table_row` so shapes with no matching table row are
    tracked separately from rows whose selected table value is missing.
15. Align again to `source_shapes_index_by_row`, repeating values and masks for
    MultiPolygon parts.
16. Return `_ShapeTableRowAlignment` for Slice 4 styling.

Missing table rows:

- If no table rows remain for the selected shapes region after filtering by
  `region_key`, raise a clear error.
- If any table instances for the selected shapes region are missing from
  `shapes_element[instance_key]`, raise a clear error.
- If some shapes have no matching table row, allow the alignment and expose the
  count through `source_row_has_table_row` / `rendered_row_has_table_row`.
  Slice 4 should render those rows transparent and report the count as
  informational status feedback, not as a warning.

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
- for positive-integer shape instance IDs, derive colors from napari's cyclic
  `label_colormap(...)` using those integer IDs, matching labels conceptually;
- for string or otherwise non-numeric shape instance IDs, deterministically map
  each distinct instance label to a positive integer code, then sample the same
  cyclic `label_colormap(...)`;
- do not use the default categorical palette for `instance_key` identity
  coloring;
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

- use `SHAPES_MISSING_BASE_COLOR` only for shape instances that have a matching
  table row but whose selected table value is missing;
- use transparent colors for shape instances that have no matching table row;
- preserve existing alpha rules:
  - face alpha `0.0` by default;
  - face alpha `SHAPES_FACE_ALPHA` when `Fill` is enabled;
  - edge alpha `SHAPES_EDGE_ALPHA`.

## Implementation Sketch

Likely code changes:

- `core/_color_source.py`
  - rename the current direct-shape-column dataclass to
    `ShapeColumnColorSourceSpec`;
  - use explicit `ShapeColumnColorSourceSpec | TableColorSourceSpec` annotations
    at shape-facing boundaries instead of adding a broad union alias;
  - decide whether to add shape-table-specific display helpers or keep those in
    the widget/status-card layer.
- `core/spatialdata.py`
  - make existing table annotation helper parameter names element-generic where
    the implementation is already generic;
  - reuse `get_annotating_table_names(sdata, shapes_name)` and
    `get_table_color_source_options(sdata, table_name)` for Slice 2 discovery;
  - keep shape-specific table-to-shape validation for Slice 3.
- `viewer/shapes_styling.py`
  - keep `apply_shape_column_color_source_to_shapes_layer(...)` for direct columns;
  - add `apply_table_color_source_to_shapes_layer(...)` with an `sdata`-based
    API that mirrors labels styling and resolves the linked table internally:
    ```python
    def apply_table_color_source_to_shapes_layer(
        layer: Shapes,
        *,
        sdata: SpatialData,
        shapes_name: str,
        style_spec: TableColorSourceSpec,
        source_shapes_index_by_row: tuple[Any, ...],
        source_shapes_index_feature_name: str,
        fill: bool = False,
    ) -> ShapesStyleResult:
        ...
    ```
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
- table `instance_key` identity coloring maps non-numeric instance labels to
  deterministic positive integer codes and samples napari's cyclic label
  colormap, not the default categorical palette;
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
- an annotating table with no rows for the selected shapes region raises a clear
  error;
- shape instances without table rows render transparent;
- shape instances with table rows but missing selected values render gray;
- partial shape coverage reports the count of shape instances without table
  rows as informational feedback, not warning feedback;
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
- table-backed values are added to `layer.features` under `value_key`, with the
  same source-index-feature collision handling used by direct shape columns.

## Implementation Slices

Each slice should be independently testable and should keep the existing direct
shape-column coloring behavior unchanged.

1. Table metadata validation cleanup - completed
   - keep the existing public labels-facing table APIs stable for this slice:
     `get_annotating_table_names(...)`,
     `get_table_annotated_labels_names(...)`,
     `validate_table_annotation_coverage(...)`,
     `validate_table_region_instance_ids(...)`, and
     `validate_table_binding(...)`;
   - make table annotation terminology element-generic only where the existing
     behavior is already element-generic, for example
     `SpatialDataTableMetadata.annotates(element_name)`;
   - rename private helper arguments from labels-specific names to generic
     `element_name` where they validate table metadata rather than labels data;
   - add or reuse validation that rejects duplicate `instance_key` values within
     one selected region, without adding the shapes-specific validation API yet;
   - add a small private helper for required table `.obs` binding columns if it
     keeps the current validation functions consistent;
   - fix labels styling so duplicate labels-table `instance_key` values raise
     instead of being silently de-duplicated with `keep="last"`;
   - labels styling should call the existing table binding validation before
     alignment, and `_get_region_rows_by_instance(...)` should also reject
     duplicates after labels-specific numeric coercion;
   - keep shapes/table-to-shape alignment validation for Slice 3, because it
     needs `shapes_element[instance_key]`, exact matching, subset validation, and
     partial shape coverage behavior;
   - add labels regression tests for duplicate instance IDs within one region,
     including duplicate IDs that only become duplicates after labels-specific
     numeric coercion.

2. Shape table discovery and source typing - completed
   - mirror the current labels discovery flow instead of introducing a flat
     all-tables source list;
   - generalize the existing `get_annotating_table_names(...)` helper to use an
     `element_name` parameter, because SpatialData's
     `get_element_annotators(sdata, element_name)` is already element-generic;
     keep the function name stable unless a shape-specific alias is useful for
     readability;
   - use `get_annotating_table_names(sdata, shapes_name)` for shapes linked-table
     discovery, so only tables that explicitly annotate that shapes element are
     returned;
   - keep `get_table_color_source_options(sdata, table_name)` as the per-table
     source API, matching labels;
   - keep direct shape-column source discovery separate:
     `shape_column_color_sources = get_shape_column_color_source_options(sdata, shapes_name)`;
   - have the shapes widget build
     `table_color_sources_by_table: dict[str, list[TableColorSourceSpec]]`,
     matching `_LabelsCardWidget`;
   - use `table_color_sources_by_table` as the single source of truth for linked
     table choices in labels and shapes cards, deriving the table combo entries
     from the dictionary order instead of passing a separate `table_names`
     argument that could drift;
   - the shapes widget should therefore mirror labels with:
     `table_names = get_annotating_table_names(sdata, shapes_name)` and
     `table_color_sources_by_table = {table_name: get_table_color_source_options(sdata, table_name) for table_name in table_names}`;
   - reuse existing table source option classification: expose table
     `instance_key` as an observation source with `value_kind="instance"`, keep
     excluding `region_key`, include colorable `.obs` columns, and include
     `X[:, var_name]` sources;
   - rename the current concrete `ShapeColorSourceSpec` dataclass to
     `ShapeColumnColorSourceSpec`;
   - keep `TableColorSourceSpec` as the table-backed source model and update its
     docstring so it is no longer labels-specific;
   - broaden shape-facing type annotations to accept explicit
     `ShapeColumnColorSourceSpec | TableColorSourceSpec` unions without changing
     adapter dispatch behavior yet;
   - do not add shape table-to-shape alignment validation in this slice:
     `shapes_element[instance_key]`, exact matching, subset validation, missing
     shape coverage, and no-row checks belong to Slice 3;
   - test linked shapes table discovery, labels-only table exclusion, per-table
     source grouping, table
     `instance_key` as `value_kind="instance"`, `region_key` exclusion, `.obs`
     sources, `X[:, var_name]` sources, and unchanged direct shape-column source
     discovery.

3. Shape table-to-source-row alignment - completed
   - implement a small private `_ShapeTableRowAlignment` return object and
     `_align_table_color_source_to_shapes_rows(...)`;
   - document the difference between source-row values, rendered-row values,
     and table-coverage masks in docstrings or nearby inline comments;
   - avoid a broad public alignment dataclass; keep table metadata,
     selected-region rows, table observation index, and `value_by_instance` as
     local implementation details unless Slice 4 proves they need to cross a
     function boundary;
   - keep this slice alignment-only: return source-row/rendered-row values and
     table-coverage masks, but do not apply colors, write `layer.features`, or
     dispatch through the adapter yet;
   - require `shapes_element[instance_key]` and never fall back to
     `shapes_element.index`;
   - require the source GeoDataFrame index to be unique, because rendered-row
     mapping uses that index;
   - use exact value matching between table and shapes instance values;
   - require selected-region table instances to be a subset of shape instances;
   - raise if selected-region table rows have missing `instance_key` values;
   - raise if any source shapes row in the GeoDataFrame has a missing
     `shapes_element[instance_key]` value;
   - require selected-region table `instance_key` values to be unique;
   - allow duplicate values in `shapes_element[instance_key]`, so multiple
     geometries can share one table-backed instance value;
   - allow shape instances without table rows and track them separately from
     missing selected table values;
   - preserve string and non-integer instance values without coercion;
   - test missing shapes `instance_key` column, missing shapes `instance_key`
     values, non-unique source GeoDataFrame index, missing table `instance_key`
     values, duplicate table instances, duplicate shapes instance values, extra
     table instances, no selected-region table rows, partial shape coverage,
     rendered-row repetition, and string IDs.

4. Table-backed shapes styling - completed
   - add `apply_table_color_source_to_shapes_layer(...)` with the agreed API:
     ```python
     def apply_table_color_source_to_shapes_layer(
         layer: Shapes,
         *,
         sdata: SpatialData,
         shapes_name: str,
         style_spec: TableColorSourceSpec,
         source_shapes_index_by_row: tuple[Any, ...],
         source_shapes_index_feature_name: str,
         fill: bool = False,
     ) -> ShapesStyleResult:
         ...
     ```
   - resolve the table and `SpatialDataTableMetadata` from
     `style_spec.table_name` inside the function, matching the labels styling
     API shape and keeping adapter dispatch simple;
   - reuse the existing rendered-row expansion via `source_shapes_index_by_row`;
   - add the special `instance_key` identity-coloring branch;
   - support categorical `.obs` columns with `table.uns["<column>_colors"]`;
   - support string/object `.obs` coercion with default palette behavior;
   - support continuous `.obs` columns and dense/sparse `X[:, var_name]`;
   - preserve shape fill/edge alpha rules, missing gray behavior for table rows
     with missing selected values, and transparent behavior for shapes without
     table rows;
   - return `ShapesStyleResult` with
     `unannotated_source_shape_count` and
     `unannotated_rendered_shape_count`, using zero counts for direct
     shape-column styling and table coverage masks for table-backed styling;
   - store the selected table value in `layer.features` under `value_key`,
     disambiguating only if it collides with the source-index feature name;
   - add rendered-row tests for MultiPolygon repetition and feature-table values.

5. Adapter and layer lifecycle - completed
   - dispatch styled shapes to direct shape-column styling or table-backed
     styling based on the source spec;
   - include table name in styled-layer identity and lookup;
   - keep displayed layer names table-name-free:
     `shapes_column:leiden`, `obs:cell_type`, `X:GeneA`;
   - ensure styled variants from multiple linked tables can coexist even when
     their displayed names collide or napari suffixes them;
   - test creation, update, fill toggling, layer lookup, and coexistence.

6. Widget and status feedback - completed
   - no major widget API changes are expected in this slice: shape cards already
     support linked-table selection, `None | Shapes column | Observations |
     Vars`, `TableColorSourceSpec` dispatch, and linked-table disabled states;
   - keep `_add_or_update_styled_shapes_layer(...)` responsible for surfacing
     table-backed alignment and validation errors as `Styled Shapes Error`,
     including missing shape `instance_key` column, extra table instances, and
     no selected-region table rows;
   - update `build_styled_shapes_card_spec(...)` so the status text describes
     the selected source by source type:
     - direct shape-column source: `column "leiden"`;
     - table observation source: `obs["cell_type"]`;
     - table expression source: `X[:, "GeneA"]`;
   - report table-backed instance-key coloring explicitly in status feedback
     with the shared wording `Used instance colors.`;
   - report `unannotated_source_shape_count` and
     `unannotated_rendered_shape_count` as informational feedback for
     table-backed shapes, because shapes without a linked table row are valid
     and are rendered transparent by design;
   - do not turn partial table coverage into a warning. If the styled layer has
     no other warning condition, partial coverage may make the status kind
     `info`; existing warning conditions such as skipped geometries, invalid
     palettes, or palette coercion should remain warnings;
   - keep skipped-geometry feedback unchanged and stronger than partial table
     coverage info;
   - add focused tests for status-card source wording, table-backed instance
     feedback, partial table coverage info, the existing table-backed widget
     request/update path, and one representative table-alignment error reaching
     widget feedback.

7a. Internal source-row identity for shapes - follow-up
   - stop relying on the GeoDataFrame index as the unique internal source-row
     identity for rendered-row lookup;
   - generate an internal source-row identifier when building napari shapes
     layers. Prefer integer source row position, `0..len(shapes_element)-1`,
     because it is unique even when the GeoDataFrame index contains duplicates;
   - rename the rendered-row mapping to make the new meaning explicit:
     `source_shapes_index_by_row` should become
     `source_row_id_by_rendered_row`;
   - use `source_row_id_by_rendered_row` for all internal rendered-row to
     source-row style alignment, so rendered napari rows can always map back to
     the correct source GeoDataFrame row even when the GeoDataFrame index is
     duplicated;
   - keep the GeoDataFrame index as a visible feature for status display, using
     `source_shapes_index_feature_name` or a later clearer name. For example,
     if the index name is `cell_id`, keep showing `cell_id: ...` in the napari
     status bar, even if those displayed index values are duplicated;
   - keep this internal row id separate from user-facing shape features and
     separate from biological/table instance identity;
   - preserve existing MultiPolygon expansion behavior: multiple rendered rows
     from one source row should carry the same internal source-row identifier;
   - remove the current styling requirement that `shapes_element.index` is
     unique;
   - direct shape-column styling should align source values by integer source
     row id, for example with `iloc`, not by `reindex(...)` on the GeoDataFrame
     index;
   - table-backed styling should produce source-row values aligned to internal
     source row ids, then expand them to rendered rows with
     `source_row_id_by_rendered_row`;
   - keep validating that the rendered-row mapping has one entry per rendered
     napari shape row and that each internal source row id resolves to a source
     GeoDataFrame row;
   - add tests for primary shapes loading with duplicate GeoDataFrame indices,
     duplicated visible index values in `layer.features`, direct shape-column
     styling with duplicate GeoDataFrame indices, MultiPolygon expansion,
     adapter bindings storing internal row ids, and table-backed styling with an
     `instance_key` column while the GeoDataFrame index is duplicated.

7b. Shape instance identity source compatibility - follow-up
   - after 7a, align table-backed shapes styling with SpatialData's accepted
     shapes annotation patterns by resolving biological/table shape instance
     identity independently from rendered-row source identity:
     - if `shapes_element[instance_key]` exists as a GeoDataFrame column, use
       that column;
     - otherwise, if `shapes_element.index.name == instance_key`, use the
       GeoDataFrame index;
     - if both the column and a named index exist, require them to agree row by
       row, or raise a clear error when they disagree;
   - keep exact value matching against `table.obs[instance_key]`;
   - keep allowing duplicate shape instance values, because multiple source
     geometries may represent the same biological instance and should receive
     the same table-backed value;
   - when the GeoDataFrame index is used as the resolved instance identity, keep
     displaying that source index in `layer.features` under the real index name
     or `"index"` fallback. Do not hide or overwrite it just because it also
     serves as the biological/table instance identity;
   - if table-backed styling by `instance_key` would store the selected table
     value under the same feature name as the visible source-index feature,
     continue using the existing `__value` disambiguation, for example
     `instance_id` for the visible source index and `instance_id__value` for
     the selected table-backed style value;
   - update error messages so they mention both accepted locations when neither
     a matching column nor a matching index name is present;
   - add tests for column-backed shape instances, index-backed shape instances
     with unique and duplicate index values, matching column/index values,
     disagreeing column/index values, and table instances missing from the
     resolved shape instance identities.

Optional cleanup slice:

- Consider renaming `source_shapes_index_by_row` to
  `source_shapes_index_by_rendered_row` across the adapter, shapes styling,
  tests, and roadmap docs. The current name is established and documented, so
  this should be a behavior-free naming cleanup only if the clearer rendered-row
  wording is worth the churn.

## Recommendation

Implement this as a follow-up to the existing direct shape-column coloring path,
not as a replacement for it.

The best first version is:

- only tables that explicitly annotate the selected shapes element;
- the table `instance_key`, other `.obs` columns, and `X[:, var_name]` values;
- exact value matching between `table.obs[instance_key]` and
  the resolved shapes instance identities, with table instances required to be
  a subset of shape instances;
- direct reuse of existing shape rendered-row coloring;
- labels-compatible table palette behavior;
- no labels-table inference and no direct `.var` metadata coloring.
