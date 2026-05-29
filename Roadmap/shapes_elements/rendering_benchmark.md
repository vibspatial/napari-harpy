# Shapes Rendering And Color Preparation Benchmark Notes

Status: investigation

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

## Current Shape Coloring Path

Direct shape-column coloring does not color by the GeoDataFrame index. It uses
the GeoDataFrame index as render-row bookkeeping:

1. Shape geometry loading records `source_shapes_index_by_row`, one source
   GeoDataFrame index label per rendered napari row.
2. Direct coloring starts from the selected column,
   `shapes_element[style_spec.value_key]`.
3. The selected column is reindexed by `source_shapes_index_by_row`, repeating
   values when a source geometry expands into multiple rendered rows.
4. Categorical or continuous colors are computed for rendered rows.
5. Face and edge RGBA arrays are assigned to the layer.
6. The selected value is added to `layer.features`.

For table-backed shape coloring, the same render-row bookkeeping should remain,
but table values need one extra semantic lookup:

```text
rendered napari row
 -> source GeoDataFrame row via source_shapes_index_by_row
 -> shape instance value via shapes_element[instance_key]
 -> table value via table.obs[instance_key]
```

The table join itself should use `instance_key`; the GeoDataFrame index should
only map source rows to rendered rows.

## Local Benchmark

Synthetic benchmark, `N = 300_000`, one rendered row per source shape.

Approximate timings:

- build `pd.Index` from `source_shapes_index_by_row` tuple: `0.036s`;
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

For point shapes with a radius, the current geometry path converts each point
to an ellipse vertex array one by one. That means 300k point-radius shapes imply
300k small numpy arrays and a Python loop before napari receives the layer data.
This geometry construction cost is separate from color preparation and may
dominate initial layer creation.

Napari rendering, interaction, and GPU/display behavior are also separate from
the color-preparation timings above. A full performance decision for 300k
shapes should measure:

- geometry conversion into napari shapes data;
- assignment of `face_color`, `edge_color`, and `features`;
- napari layer rendering and interaction latency.

## Recommendation

For the table-backed `.obs` and `X[:, var_name]` roadmap, the planned
`instance_key` table join is unlikely to be the limiting factor for 300k simple
shape instances. The first performance improvement to consider is vectorizing
continuous color generation and avoiding large intermediate object color
`Series` where practical.
