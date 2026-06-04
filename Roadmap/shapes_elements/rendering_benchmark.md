# Shapes Rendering And Color Preparation Benchmark Notes

Status: historical benchmark note, updated for current code paths

## Scope

This note records a preliminary local benchmark for coloring many shapes. It is
not a full napari rendering benchmark.

Target scenario:

- 300k shapes;
- simple point geometries with a radius;
- one rendered napari row per source shape instance;
- measure the data preparation needed to color shapes, not the eventual GPU/UI
  rendering cost.

In this context, "coloring" means:

- align source values to rendered napari shape rows;
- build or update the `layer.features` values used for inspection/status;
- create face and edge RGBA arrays that are assigned to napari.

## Current Code Status

The code base has moved beyond the original benchmark assumptions in a few
important ways:

- qualifying point-radius shapes are rendered as napari `Points`, not as one
  ellipse row per source point;
- point-radius preparation is vectorized:
  `_prepare_napari_point_radius_shapes_layer_inputs(...)` extracts geometry
  coordinates and radius values as arrays, builds `size = 2 * radius`, and uses
  `source_row_id_by_rendered_row = range(n)`;
- non-qualifying shapes still use the generic napari `Shapes` path, where a
  source row can expand into multiple rendered rows, for example one
  `MultiPolygon` source row becoming several rendered polygons;
- rendered-row alignment is based on integer source row ids
  (`source_row_id_by_rendered_row`), not GeoDataFrame index labels. This keeps
  styling unambiguous when GeoDataFrame indexes contain duplicate labels;
- direct shape-column styling starts from
  `shapes_element[style_spec.value_key]` and aligns with
  `source_values.iloc[source_row_id_by_rendered_row]`;
- table-backed styling requires the GeoDataFrame index name to match the
  table `instance_key`. A same-named GeoDataFrame column is allowed only when it
  matches the index exactly.
- direct and table-backed shapes styling now build `Nx4` float RGBA arrays
  directly for categorical, continuous, and instance colors;
- styled `Shapes` and point-backed shapes assign alpha directly on RGBA arrays,
  avoiding the old row-wise `_with_alpha(...)` conversion.
- labels continuous coloring now builds vectorized RGBA values before
  constructing the `DirectLabelColormap` dictionary.

The remaining planned labels work is categorical implementation alignment:
`continuous_colors_for_values(...)` is still retained for existing callers, and
labels still need a final label-id-to-color dictionary for napari.

## Current Shape Coloring Path

Direct shape-column coloring does not color by the GeoDataFrame index. It uses
integer source row ids as render-row bookkeeping:

1. Shape geometry loading records `source_row_id_by_rendered_row`, one integer
   source GeoDataFrame row id per rendered napari row.
2. Direct coloring starts from the selected column,
   `shapes_element[style_spec.value_key]`.
3. The selected column is aligned with
   `source_values.iloc[source_row_id_by_rendered_row]`, repeating values when a
   source geometry expands into multiple rendered rows.
4. Categorical or continuous colors are computed for rendered rows.
5. Face and edge RGBA arrays are assigned to the layer.
6. The selected value is added to `layer.features`.

For table-backed shape coloring, the same render-row bookkeeping is used, but
table values need one extra semantic lookup:

```text
rendered napari row
 -> source GeoDataFrame row via source_row_id_by_rendered_row
 -> shape instance value via shapes_element.index, where index.name == instance_key
 -> table value via table.obs[instance_key]
```

The table join itself uses `instance_key`. The GeoDataFrame index provides the
shape instance identities, and the integer row-id mapping connects rendered
napari rows back to source rows.

## Historical Local Benchmark

Synthetic benchmark, `N = 300_000`, one rendered row per source shape. These
timings were gathered before the point-radius `Points` fast path and before the
current integer source-row-id alignment. They remain useful as a rough
color-preparation signal, not as a current end-to-end rendering benchmark.

Approximate timings:

- build `pd.Index` from old source-index tuple: `0.036s`;
- direct shape-column reindex to rendered rows: `0.001s`;
- table join by string `instance_key` plus rendered-row reindex: `0.009s`;
- copy `features` and assign one numeric value column: `0.001s`;
- copy `features` and assign one categorical/object value column: `0.002s`;
- current categorical normalization: `0.049s`;
- current categorical color lookup: `0.135s`;
- current `_with_alpha(...)` conversion for one RGBA array: `0.13s`;
- current continuous color lookup: `2.98s`.

Interpretation:

- The pandas alignment work is small at 300k rows.
- The table-backed `instance_key` join is also small in this synthetic case.
- The current continuous color path is the main color-preparation hotspot.

The continuous hotspot comes from row-wise Python work in
`continuous_colors_for_values(...)`. A vectorized reference implementation for
continuous RGBA generation took about `0.006s` on the same synthetic input, so
there is substantial optimization headroom if 300k-shape continuous coloring
becomes a required target.

## Pre-Optimization Synthetic Benchmark, June 3, 2026

Ran locally with `.venv/bin/python` on synthetic `N = 300_000` rows,
`32` categories, and `5%` missing values unless noted otherwise. Timings report
the best run from a small repeat count. These are color-preparation benchmarks,
not full napari rendering benchmarks.

Shared color creation:

- current `categorical_colors_for_values(...)`: `0.118s`;
- current `continuous_colors_for_values(...)`: `3.013s`;
- reference vectorized categorical RGBA creation: `0.0078s`;
- reference vectorized continuous RGBA creation: `0.0075s`.

Shapes color-array conversion:

- current `_with_alpha(...)` on categorical colors, once: `0.131s`;
- current `_with_alpha(...)` on continuous colors, once: `0.134s`;
- reference categorical RGBA with alpha already prepared: `0.0075s`;
- reference continuous RGBA with alpha already prepared: `0.0071s`.

Labels color-dictionary creation:

- current categorical labels color dict: `0.146s`;
- current continuous labels color dict: `3.106s`;
- reference categorical RGBA plus dict conversion: `0.151s`;
- reference continuous RGBA plus dict conversion: `0.155s`.

Napari `Points` categorical styling:

- current `apply_points_selection_style(...)` for `300k` points and
  `32` categories, no missing values: `0.159s` best run.

Interpretation:

- Continuous coloring is the clear current bottleneck for labels and shapes.
- For labels, categorical coloring is already dominated by unavoidable
  `DirectLabelColormap` dictionary creation, so categorical optimization is not
  the first priority there.
- For shapes and point-backed shapes, categorical coloring still has useful
  headroom because the current path creates object color `Series` and then loops
  through `_with_alpha(...)`.
- Real SpatialData points currently use categorical value-selection styling,
  not table-style continuous coloring. That path is reasonably fast at 300k
  rows. Point-backed shapes are covered by the shapes styling path.

## Concrete Implementation Plan

Goal: make viewer coloring reasonably fast for labels, generic shapes,
point-backed shapes, and real points, with continuous coloring as the primary
target and categorical shapes as a secondary target.

Non-goal:

- Do not optimize the explicit geometry loop that splits generic polygon and
  multipolygon geometries into napari `Shapes` rows in this plan. Large polygon
  geometry preparation is a separate concern, and point-radius shapes already
  use the napari `Points` fast path.

1. Add a reusable benchmark script - completed
   - added `scripts/benchmark_viewer_coloring.py`;
   - run with:
     `./.venv/bin/python scripts/benchmark_viewer_coloring.py --rows 100000 300000`;
   - benchmark synthetic `100k`, `300k`, and optionally `1M` rows;
   - include:
     - shared categorical and continuous color creation;
     - labels categorical and continuous color-dict creation;
     - shapes categorical and continuous RGBA-array creation;
     - point-backed shapes through the same shapes styling helpers;
     - real napari `Points` categorical styling;
   - keep geometry construction out of the benchmark.

2. Add vectorized RGBA helper APIs in `viewer/_styling.py` - completed
   - add a continuous helper that returns an `Nx4` float RGBA array directly,
     for example `continuous_rgba_for_values(...)`;
   - add a categorical helper that returns an `Nx4` float RGBA array directly,
     for example `categorical_rgba_for_values(...)`;
   - both helpers should return `float64` RGBA arrays aligned one-to-one with
     the input `values`;
   - preserve current continuous behavior:
     - missing values use `missing_color`;
     - all-missing input returns all `missing_color`;
     - constant non-missing input maps to the colormap midpoint `0.5`;
     - normalized values are clipped to `[0, 1]`;
   - preserve current categorical behavior:
     - missing values use `missing_color`;
     - values not found in `categories` use `missing_color`;
     - category matching still uses `normalize_category_value(...)`, including
       numpy scalar normalization;
   - preserve the existing public behavior of `continuous_colors_for_values(...)`
     and `categorical_colors_for_values(...)` until callers have moved over;
   - add focused tests, preferably in a dedicated styling test module, comparing
     the new helpers against the old color results after converting old colors
     with `matplotlib.colors.to_rgba(...)`;
   - cover:
     - continuous normal values with missing values;
     - continuous constant values;
     - continuous all-missing values;
     - categorical normal values with missing values;
     - categorical unknown values;
     - numpy scalar category normalization;
   - do not migrate labels/shapes callers in this slice. That is covered by the
     next slices.

3. Update shapes and point-backed shapes to use RGBA arrays directly - completed
   - update `_build_continuous_shape_style(...)` to use the vectorized
     continuous RGBA helper;
   - update categorical shape style paths to use the vectorized categorical
     RGBA helper where categories and palettes are already resolved;
   - update table-backed instance coloring to keep the RGBA array returned by
     `label_colormap(...).map(...)` directly, instead of wrapping it in a
     `pd.Series` of color tuples;
   - replace `_with_alpha(...)` in the shapes/points styling path with direct
     alpha assignment on the RGBA array;
   - preserve current semantics:
     - missing selected values remain gray;
     - table-backed unannotated shapes/points remain transparent;
     - `fill=False` keeps shapes faces transparent;
     - point-backed shapes keep face and border colors identical.

4. Update labels continuous coloring - completed
   - use vectorized continuous RGBA generation before constructing the
     `DirectLabelColormap` dictionary;
   - leave categorical labels to the next alignment slice, because the
     benchmark shows the dict creation itself dominates and current categorical
     performance is already acceptable;
   - preserve labels-specific behavior:
     - `None` and `0` stay transparent;
     - instance coloring still uses napari's label colormap;
     - features remain aligned by positive label id.

5. Align categorical labels with shared RGBA helper implementation - completed
   - refactor `_build_categorical_color_dict(...)` to use
     `categorical_rgba_for_values(...)` before constructing the
     `DirectLabelColormap` dictionary;
   - treat this as implementation alignment, not as a primary performance
     optimization, because labels still require a label-id-to-color dictionary;
   - preserve categorical label behavior:
     - `None` and `0` stay transparent;
     - missing and unknown category values use the existing missing color;
     - stored palettes from `table.uns` continue to resolve the same way;
     - features remain aligned by positive label id;
   - benchmark before/after to confirm categorical labels do not regress.

6. Keep real points styling mostly unchanged - completed
   - real SpatialData points continue to use categorical value-selection styling;
   - keep `apply_points_selection_style(...)` as-is unless a later continuous
     points-coloring feature is added;
   - if continuous points coloring is added later, reuse the new vectorized
     RGBA helpers instead of introducing another row-wise color path.

7. Acceptance criteria
   - existing viewer adapter, labels, shapes, and widget tests pass;
   - new helper tests prove color equivalence with current behavior;
   - benchmark script shows:
     - continuous shared RGBA creation near the vectorized reference time;
     - labels continuous color dict drops from about `3.1s` toward `0.15s` at
       `300k` rows;
     - shapes continuous color-array preparation drops from several seconds
       plus alpha-loop overhead to tens of milliseconds at `300k` rows;
     - categorical shapes avoid the object `Series` plus `_with_alpha(...)`
       overhead where practical;
     - real points categorical styling remains in the same ballpark.

## Memory Ballpark

For `N = 300_000` rendered rows:

- continuous value `Series`: about `2.3 MiB`;
- continuous color object `Series`: about `22.9 MiB`;
- face RGBA `float64` array: about `9.2 MiB`;
- edge RGBA `float64` array: about `9.2 MiB`;
- numeric `features` DataFrame with source index and one value column: about
  `4.6 MiB`;
- categorical/object value `Series`: about `14.4 MiB`;
- categorical color object `Series`: about `16.0 MiB`;
- object `features` DataFrame with source index and one value column: about
  `16.7 MiB`.

## Caveat: Geometry And Rendering Are Separate

This benchmark does not include building the shapes layer geometry or napari's
actual rendering.

For qualifying point-radius shapes, geometry preparation no longer converts
each point to an ellipse vertex array. The current fast path renders them as
napari `Points`, with vectorized coordinates and radius-derived sizes. This
removes the old 300k-small-ellipse-arrays bottleneck for all-Point elements with
a valid positive `radius` column.

For non-qualifying shapes, the generic napari `Shapes` path still builds shape
vertex arrays and can still involve Python-level geometry work, especially for
large polygon or multipolygon elements.

Napari rendering, interaction, and GPU/display behavior are also separate from
the color-preparation timings above. A full performance decision for 300k
shapes should measure:

- point-radius preparation into napari `Points` data, when the fast path
  qualifies;
- geometry conversion into napari `Shapes` data, when the generic path is used;
- assignment of `face_color`, `edge_color`, and `features`;
- napari layer rendering and interaction latency.

## Completed Status

For table-backed `.obs` and `X[:, var_name]` styling, the `instance_key` table
join is unlikely to be the limiting factor for 300k simple shape instances.
Point-radius geometry preparation has already been addressed by rendering
qualifying elements as napari `Points`.

The color-preparation refactor covered by this note is complete:

- shared categorical and continuous RGBA helpers exist;
- shapes and point-backed shapes use RGBA arrays directly;
- labels continuous and categorical table-backed styling use the shared RGBA
  helpers before constructing napari label colormaps;
- real SpatialData points remain on the existing napari property-coloring path.

Remaining performance questions should be tracked as separate follow-up issues:

- large categorical `Labels` styling is dominated by napari
  `DirectLabelColormap` construction/application for one color entry per label
  id;
- real `Points` categorical styling is dominated by napari property-color
  mapping when assigning `layer.face_color = selection.index_column`.
