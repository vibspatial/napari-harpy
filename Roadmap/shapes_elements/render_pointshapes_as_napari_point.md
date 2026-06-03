# Render Point-Radius Shapes As Napari Points

Status: Slice 5 implemented; Slices 6-8 pending

## Goal

Support snappy visualization of SpatialData shapes elements whose geometries
are all `Point` objects with a valid `radius` column by rendering them as a
napari `Points` layer with per-point size, instead of converting every point
into an ellipse row in a napari `Shapes` layer.

This is a separate, larger refactor from table-backed shapes coloring. The data
model remains a SpatialData shapes element; only the viewer representation uses
napari `Points` for this specific point-radius case.

## Motivation

The generic shapes path is appropriate for polygon and MultiPolygon geometries,
but it is not the right fast path for many simple point-radius objects. For
large point-radius elements, constructing one ellipse vertex array and one
napari `Shapes` row per source row is avoidable work.

`napari-spatialdata` follows this direction for circles: when circles can be
shown as points, it uses a napari `Points` layer with point size derived from
radius. Harpy should support the same idea for point-radius shapes.

This should not be treated as the solution for snappy loading of hundreds of
thousands of polygon boundaries. Polygon-heavy elements are better represented
as labels, simplified boundaries, ROI/viewport-limited geometry, or another
polygon-specific strategy.

## Trigger Conditions

Use the fast path only when all of these are true:

- the selected SpatialData element is a shapes element;
- every renderable geometry is a `Point`;
- a `radius` column exists;
- every rendered point has a finite positive radius;
- the selected coordinate system can be resolved for the shapes element.

If any condition fails, fall back to the existing napari `Shapes` path.

## Rendering Semantics

- Extract coordinates and radii as arrays, avoiding `iterrows()`.
- Build napari points coordinates in `(y, x)` display order.
- Set per-point size from radius, conceptually `size = 2 * radius * scale_factor`.
- Follow the existing shapes transform semantics: call SpatialData
  `transform_spatial_element(...)` first, then extract transformed point
  coordinates and transformed radii.
- Do not pass the shapes element transformation to napari as `layer.affine` for
  this mode. The napari `Points` layer should receive already-transformed
  coordinates and sizes, matching the current shapes path and avoiding double
  transforms.
- Preserve source-row traceability with internal row ids.
- Preserve visible source-index metadata/status behavior currently provided by
  shapes `layer.features`.

## Adapter And Binding Semantics

The viewer layer would be a napari `Points` layer, but semantically it still
represents a SpatialData shapes element.

The adapter therefore needs explicit binding semantics for this mode:

- keep `element_type="shapes"` or introduce a clear equivalent that says this
  is a shapes element rendered as points;
- keep the shapes element name and coordinate system in the binding;
- keep source row ids aligned one-to-one with rendered points;
- keep skipped/invalid geometry feedback;
- make widget/status feedback explicit enough that users understand they are
  viewing point-radius shapes as napari points.

## Styling Semantics

Shape-column coloring and table-backed coloring should remain unchanged at the
SpatialData/data-model level.

Support both unstyled primary display and styled variants through the points
fast path. Direct shape-column coloring and table-backed coloring should render
as napari `Points` layers whenever the selected shapes element qualifies for
point-radius rendering.

Styled point-radius shapes should follow the same semantic rules as styled
shapes:

- direct shape-column styling uses values from the source GeoDataFrame;
- table-backed styling uses the resolved shapes instance identity;
- table rows may partially cover the shapes element;
- shapes without table rows are transparent;
- shapes with table rows but missing selected values use the missing color;
- categorical palettes, continuous colormaps, and instance colors should match
  the existing shapes styling behavior.

## Fallback

If an element does not qualify for the point-radius fast path, use the existing
generic napari `Shapes` path. Do not add a separate vectorized ellipse rendering
mode unless a concrete workflow later requires actual napari `Shapes` semantics
for otherwise valid point-radius data.

## Tests

Add focused tests for:

- all-point valid radii using the points fast path;
- invalid, missing, non-finite, or non-positive radii falling back or raising
  with clear feedback, depending on final UX choice;
- mixed geometry types falling back to the existing shapes path;
- duplicate GeoDataFrame index values;
- named index display/status behavior;
- source-row id preservation;
- table-backed source identity compatibility;
- fallback to the generic `Shapes` path for non-qualifying elements.

## Implementation Slices

1. Point-radius detection and preparation - completed
   - add a private prepared-input dataclass, for example
     `_NapariPointRadiusShapesLayerInputs`, with:
     - `coordinates: np.ndarray` in napari `(y, x)` order;
     - `sizes: np.ndarray`, one diameter per rendered point;
     - `features: pd.DataFrame`, row-aligned to `coordinates`;
     - `source_shapes_index_feature_name: str`;
     - `source_row_id_by_rendered_row: tuple[int, ...]`;
     - `skipped_geometry_count: int`;
   - add a private helper, for example
     `_prepare_napari_point_radius_shapes_layer_inputs(transformed_shapes)`,
     that returns `_NapariPointRadiusShapesLayerInputs | None`;
   - this helper receives an already transformed shapes element. The caller is
     responsible for calling SpatialData `transform_spatial_element(...)`
     before detection/preparation, matching the generic shapes path;
   - return `None` when the element does not qualify, so the caller can fall
     back to `_prepare_napari_shapes_layer_inputs(...)`;
   - qualification rules:
     - the shapes element has a `radius` column;
     - every source row has a non-empty `Point` geometry;
     - every radius value can be converted to a finite positive float;
     - the number of extracted geometries, radii, index values, and source row
       ids all match;
   - mixed geometries, missing radius column, missing/empty geometries,
     non-finite radii, non-positive radii, or non-numeric radii should return
     `None` for this slice rather than raising. The existing generic shapes path
     remains the fallback;
   - extract geometry, radius, and source index values as arrays without
     `iterrows()`;
   - build `coordinates` with `y = geometry.y`, `x = geometry.x`, and return
     shape `(n, 2)`;
   - build `sizes` as `2 * radius`, because the transformed shapes element
     already contains transformed radius values;
   - keep `source_row_id_by_rendered_row == tuple(range(n))` for this point
     fast path, because each source row renders as exactly one napari point;
   - use `_get_shapes_index_feature_name(...)` and preserve duplicate/named
     GeoDataFrame index values in `features`, matching the generic shapes path;
   - do not create a napari layer in this slice;
   - add tests for:
     - valid points/radii returning coordinates, sizes, features, source row
       ids, and zero skipped count;
     - missing `radius` returning `None`;
     - invalid radius values returning `None`;
     - mixed geometries returning `None`;
     - empty or missing point geometry returning `None`;
     - duplicate GeoDataFrame index values preserved in `features`;
     - named GeoDataFrame index values using the index name as the feature
       column;
     - helper does not call `GeoDataFrame.iterrows()`.

2. Vectorized point-radius preparation optimization - completed
   - optimize `_prepare_napari_point_radius_shapes_layer_inputs(...)` before
     building layer lifecycle on top of it;
   - previous Slice 1 helper was correctness-oriented and looped over Shapely
     `Point` objects. Local synthetic timings before this optimization:
     - `100k` point-radius rows: about `0.53s`;
     - `300k` point-radius rows: about `1.4-1.6s`;
     - `1M` point-radius rows: about `4.7-5.1s`;
     - extrapolated `9M` point-radius rows: roughly `40s+` for preparation
       alone;
   - the implemented vectorized path using `geometry.x`, `geometry.y`,
     vectorized radius validation, and direct feature DataFrame construction
     measured about `0.07s` for `300k` and `0.23s` for `1M` on the same
     machine;
   - replace per-row coordinate/radius extraction with vectorized extraction:
     - verify all geometries are non-empty points with GeoPandas/Shapely vector
       operations where possible;
     - validate radii with `pd.to_numeric(..., errors="coerce")` and NumPy
       finite/positive checks;
     - build coordinates with vectorized `geometry.y` and `geometry.x`;
     - build sizes with vectorized `2 * radius`;
     - build source-index features directly from the GeoDataFrame index;
   - store `source_row_id_by_rendered_row` as `range(n)` for qualifying
     point-radius shapes, not `tuple(range(n))`. The mapping is always
     one-to-one in this mode, and `range(n)` avoids materializing millions of
     Python integer objects;
   - keep the generic `Shapes` path on `tuple[int, ...]`, because polygons and
     MultiPolygons can repeat source row ids, for example `(7, 7, 7, 8)`;
   - if downstream type annotations need to accept both representations,
     introduce a small alias such as
     `SourceRowIdByRenderedRow = tuple[int, ...] | range`;
   - add benchmark-oriented tests or focused unit tests that protect the
     vectorized behavior without making the normal test suite timing-sensitive;
   - keep the Slice 1 qualification/fallback semantics unchanged.

3. Primary point-radius layer lifecycle - completed
   - extend the adapter so `ensure_shapes_loaded(...)` can create a napari
     `Points` layer for qualifying point-radius shapes elements;
   - keep the existing transform behavior:
     - validate that the requested coordinate system exists;
     - call SpatialData `transform_spatial_element(...)`;
     - pass the transformed shapes element to
       `_prepare_napari_point_radius_shapes_layer_inputs(...)`;
     - do not pass a napari `affine` to the point-radius `Points` layer;
   - if point-radius preparation returns inputs, build a napari `Points` layer:
     - `data=coordinates`;
     - `size=sizes`;
     - `features=features`;
     - default point styling should visually match the current primary shapes
       presentation as much as reasonable, for example cyan face/border and
       opacity `0.8`;
     - use one `disc` symbol per point unless there is a stronger napari reason
       to choose another symbol;
   - if point-radius preparation returns `None`, fall back to the existing
     generic `_prepare_napari_shapes_layer_inputs(...)` and `_HarpyShapes`
     layer creation;
   - bind point-radius `Points` layers semantically as shapes elements, not as
     ordinary SpatialData points elements:
     - keep `element_type="shapes"`;
     - add a binding discriminator such as
       `shapes_rendering_mode: Literal["shapes", "points"]`;
     - use `shapes_rendering_mode="points"` for the point-radius fast path and
       `"shapes"` for the existing generic shapes path;
   - update types and helpers that currently assume primary shapes layers are
     actual napari `Shapes` instances:
     - `ShapesLoadResult.layer` may need to become `Shapes | Points`;
     - `ShapesLayerBinding.layer` already inherits from the generic napari
       `Layer`, but `register_shapes_layer(...)` should accept `Shapes | Points`;
     - `source_row_id_by_rendered_row` should accept the Slice 2 `range(n)`
       representation as well as generic shapes tuples;
     - lookup helpers such as `get_loaded_primary_shapes_layer(...)` and
       `_get_loaded_shapes_layer_for_coordinate_system(...)` must consider
       points-backed shapes layers, not only `_is_shapes_layer(layer)`;
   - preserve source row id mapping, source index display/status metadata,
     coordinate system, skipped count, and layer replacement/reuse behavior;
   - provide source-index hover/status behavior for the points-backed shapes
     layer. If napari `Points` does not already display the needed feature
     values in the same way as `_HarpyShapes`, add a small Harpy-specific
     `Points` subclass or equivalent helper;
   - do not implement styled point-radius layers in this slice. Direct
     shape-column styling and table-backed styling remain separate later
     slices;
   - add tests for:
     - primary point-radius shapes loading creates a napari `Points` layer;
     - the layer has coordinates, sizes, features, and source row ids from the
       prepared inputs;
     - the layer binding has `element_type="shapes"`,
       `shapes_role="primary"`, and `shapes_rendering_mode="points"`;
     - layer reuse works for an already loaded point-radius shapes layer;
     - duplicate source indices and named source indices are preserved in
       features/status;
     - unknown coordinate system errors remain unchanged;
     - non-qualifying shapes elements still create the existing napari `Shapes`
       layer with `shapes_rendering_mode="shapes"`.

4. Direct shape-column styling on points-backed shapes - completed
   - add styling support for `ShapeColumnColorSourceSpec` when the styled layer
     is a napari `Points` layer representing point-radius shapes;
   - keep this limited to direct shape-column styling. Table-backed
     point-radius styling remains Slice 6 work;
   - in `ensure_styled_shapes_loaded(...)`, allow the point-radius fast path
     only when `style_spec` is a `ShapeColumnColorSourceSpec`;
   - keep table-backed styled shapes on the existing generic napari `Shapes`
     path until Slice 6;
   - preserve the existing styled-layer identity:
     - layer name remains `build_styled_shapes_layer_name(...)`, for example
       `cell_centroids[shapes_column:cell_type]`;
     - binding keeps `element_type="shapes"`;
     - binding uses `shapes_role="styled"`;
     - binding stores the exact `ShapeColumnColorSourceSpec`;
     - binding uses `shapes_rendering_mode="points"` for the point-backed
       styled variant and `"shapes"` for the generic fallback;
   - update styled-shapes lookup helpers to consider semantic shapes layers
     (`Shapes | Points`) instead of only napari `Shapes` where needed;
   - keep generic `Shapes` styling behavior unchanged for non-qualifying
     elements;
   - extend the existing semantic-shapes styling API
     `apply_shape_column_color_source_to_shapes_layer(...)` to accept
     `layer: Shapes | Points`;
   - keep the function name because the source element is still a SpatialData
     shapes element, even when the viewer representation is a napari `Points`
     layer;
   - share the existing direct shape-column value/color logic with generic
     shapes styling:
     - validate `style_spec.source_kind == "shape_column"`;
     - validate `style_spec.value_key` exists in the source GeoDataFrame;
     - validate `source_row_id_by_rendered_row` against the source row count
       and rendered point count;
     - start from one `source_values` series per source GeoDataFrame row;
     - align values to rendered point rows by integer source row id;
     - reuse `_build_categorical_shape_style(...)` and
       `_build_continuous_shape_style(...)` so categorical palettes,
       companion color columns, string coercion, missing values, and
       continuous colormap behavior match generic styled shapes;
   - keep the existing color application path for napari `Shapes`;
   - add a layer-specific branch for napari `Points` that applies point-backed
     colors as data-driven colors:
     - set `layer.face_color` to the rendered-row color array;
     - set `layer.border_color` to the same rendered-row color array;
     - keep `layer.border_width = 0` unless a later presentation-control slice
       decides otherwise;
     - preserve point sizes from the source `radius` values;
     - preserve `symbol="disc"`;
   - treat `fill` as a generic shapes-only presentation option for this slice:
     point-backed direct styling should remain visibly filled, because the
     filled point glyph is the fast-path representation of the point-radius
     geometry;
   - update `layer.features` exactly like generic styled shapes:
     - preserve the source GeoDataFrame index feature;
     - add the selected shape-column style value;
     - disambiguate collisions with the source index feature via
       `disambiguate_shape_style_feature_name(...)`;
     - keep hover/status display useful through the point-backed Harpy `Points`
       subclass;
   - return the same `ShapesStyleResult` semantics as generic styled shapes:
     `value_kind`, `palette_source`, and `coercion_applied` should be
     indistinguishable for the same source values;
   - add tests for:
     - qualifying point-radius shapes with categorical shape-column styling
       create a napari `Points` styled layer;
     - qualifying point-radius shapes with continuous shape-column styling
       create a napari `Points` styled layer;
     - stored companion color columns and default palettes match generic
       shape-column styling;
     - missing selected values use the existing shapes missing color;
     - source index features, named source indices, and duplicate source index
       values are preserved;
     - style-feature name collisions with the source index feature are
       disambiguated;
     - `source_row_id_by_rendered_row == range(n)` works for point-backed
       styling;
     - styled point sizes remain radius-derived after applying colors;
     - the same shape-column style spec reuses the existing styled point-backed
       layer;
     - non-qualifying shapes elements still create a napari `Shapes` styled
       layer with `shapes_rendering_mode="shapes"`.

5. Table-backed styling on points-backed shapes - completed
   - extend the existing table-backed shapes styling API rather than adding a
     separate public points-specific API:
     `apply_table_color_source_to_shapes_layer(layer: Shapes | Points, ...)`;
   - keep the function name in shapes terminology, because the source element is
     still a SpatialData shapes element even when the viewer representation is a
     napari `Points` layer;
   - update the accepted rendered-row mapping type to
     `SourceRowIdByRenderedRow` (`tuple[int, ...] | range`), so the same
     styling path works for generic shapes and one-to-one point-radius shapes;
   - reuse the existing table-to-source-row alignment helper and semantics:
     - require `instance_key` to be stored in the shapes GeoDataFrame index,
       with `shapes_element.index.name == instance_key`;
     - allow a redundant GeoDataFrame column named `instance_key` only when it
       agrees with the named index row by row;
     - require exact matching between selected-region table instances and
       shapes instance identities;
     - allow duplicate shape instance identities, so multiple rendered points
       can share the same table-backed style;
     - reject duplicate selected-region table instances, because one table
       instance must map to exactly one selected table-backed value per region;
     - preserve partial table coverage, where shapes may exist without a table
       row;
   - keep the existing table-backed value builders for observation columns,
     `X[:, var_name]`, and instance colors. These builders should remain
     layer-agnostic and return rendered-row colors plus style metadata;
   - keep the existing generic `Shapes` color application path unchanged;
   - add a private point-specific application helper, for example
     `_apply_table_rendered_row_colors_to_points_layer(...)`, that:
     - writes the computed colors to both `face_color` and `border_color`;
     - keeps point size radius-derived and preserves the current symbol;
     - keeps `border_width=0`;
     - makes unannotated points transparent;
     - leaves annotated points with missing selected values in the normal
       missing-value color;
   - preserve the agreed table-backed coverage semantics:
     - not annotated by the selected table -> transparent;
     - annotated by the selected table but selected value missing -> gray;
     - annotated with a valid selected value -> palette, colormap, or instance
       color;
   - in the adapter lifecycle, allow `TableColorSourceSpec` styled variants to
     use the point-radius fast path when the shapes element qualifies. Remove
     the current assumption that table-backed styled shapes must be backed by a
     napari `Shapes` layer;
   - non-qualifying shapes elements should continue to fall back to the generic
     napari `Shapes` styled layer;
   - preserve `ShapesStyleResult` semantics, including `value_kind`,
     `palette_source`, `coercion_applied`, `unannotated_source_shape_count`,
     and `unannotated_rendered_shape_count`;
   - keep hover/status behavior useful on point-backed styled layers:
     - source-index metadata should remain visible;
     - table-backed style features should be added with the same
       disambiguation rules as generic styled shapes;
   - add tests for:
     - `.obs` categorical styling on qualifying point-radius shapes creates a
       napari `Points` styled layer;
     - `.obs` continuous styling on qualifying point-radius shapes creates a
       napari `Points` styled layer;
     - `X[:, var_name]` styling creates a point-backed styled layer with the
       same colormap/coercion semantics as generic shapes;
     - instance colors use the same labels-like cyclic instance coloring as
       generic table-backed shapes;
     - partial table coverage makes unannotated points transparent and reports
       unannotated counts;
     - annotated rows with missing selected values use the missing-value color;
     - duplicate shape instance identities are allowed and receive the same
       table-backed style;
     - duplicate selected-region table instances raise clearly;
     - named-index instance identity works when the shapes element stores
       `instance_key` in the GeoDataFrame index;
     - table-backed style feature name collisions are disambiguated;
     - the same table-backed style spec reuses the existing styled point-backed
       layer;
     - non-qualifying shapes elements still create a generic napari `Shapes`
       styled layer with `shapes_rendering_mode="shapes"`.

6. Point-backed primary presentation controls - completed
   - scope: primary point-radius shapes layers that are rendered as napari
     `Points`, i.e. `ShapesLayerBinding.shapes_role == "primary"` and
     `shapes_rendering_mode == "points"`;
   - keep generic napari `Shapes` behavior unchanged:
     - primary generic shapes continue to sync
       `_connect_current_edge_width_to_global_edge_width(layer)`;
     - primary generic shapes continue to sync
       `_connect_current_edge_color_to_global_edge_color(layer)` when
       `sync_edge_color=True`;
   - for primary point-backed shapes, reuse the existing point presentation
     sync helpers from `viewer/points_styling.py`, separate from the generic
     shapes helpers:
     - rename/export the point helpers as reusable functions, for example
       `connect_current_symbol_to_global_point_symbol(...)` and
       `connect_current_face_color_to_global_point_face_color(...)`;
     - keep these helpers in `points_styling.py`, because they wire napari
       `Points` events rather than generic palette utilities;
     - sync `current_symbol -> symbol` for all points, because symbol is a
       presentation choice and does not alter the source point/radius geometry;
     - sync `current_face_color -> face_color` and `border_color` for all
       points, because primary unstyled point-backed shapes use one
       presentation color and should keep border color equal to face color;
     - keep `border_width=0`, unless a later explicit design decision adds
       visible point borders;
   - do not sync `current_size -> size` as an absolute overwrite in this slice:
     - point size is derived from the source `radius` column
       (`size = 2 * radius`);
     - writing a single absolute `current_size` into `layer.size` would destroy
       variable-radius information;
     - implement size interaction separately as the radius-size scale factor in
       Slice 8;
   - attach symbol sync to both primary and styled point-backed shapes:
     - symbol is presentation-only and does not flatten data-driven colors;
     - direct shape-column and table-backed styled point-backed layers should
       still let the user change all point symbols from napari's UI;
   - do not attach the point color sync callback to styled point-backed shapes:
     - direct shape-column and table-backed styled layers have data-driven
       colors;
     - user color changes must not flatten styled palettes or transparent
       unannotated rows;
     - styled point-backed layers should not receive primary color sync
       callbacks;
   - callbacks should be stored on the layer under private Harpy attribute
     names, mirroring the existing generic shapes sync pattern, so they are not
     garbage-collected;
   - add tests for:
     - changing `current_symbol` on a primary point-backed shapes layer updates
       every point's `symbol`;
     - changing `current_face_color` on a primary point-backed shapes layer
       updates every point's `face_color` and `border_color`;
     - changing presentation color keeps `border_width=0`;
     - changing `current_size` does not overwrite radius-derived `size` in this
       slice;
     - styled point-backed direct shape-column layers do not receive primary
       color sync, keep data-driven colors, and still sync symbol changes;
     - styled point-backed table-backed layers do not receive primary color
       sync, keep palette/transparent-row semantics, and still sync symbol
       changes.

7. Widget and feedback polish - completed
   - keep the shapes card controls unchanged:
     - do not add a separate "render as points" control;
     - keep direct shape-column and linked-table color-source controls at the
       shapes data-model level;
     - point-radius rendering remains an adapter decision based on the selected
       shapes element geometry and `radius` column;
   - add rendering-mode information to the shapes load result:
     - extend `ShapesLoadResult` with
       `shapes_rendering_mode: Literal["shapes", "points"] = "shapes"` or the
       existing adapter `ShapesRenderingMode` alias;
     - set it from `_BuiltShapesLayer.shapes_rendering_mode` in
       `ViewerAdapter.ensure_shapes_loaded(...)`;
     - when returning an existing primary shapes layer, read the rendering mode
       from its `ShapesLayerBinding`;
     - set it from the styled layer binding in
       `ViewerAdapter.ensure_styled_shapes_loaded(...)`;
   - use the rendering mode only for user-facing feedback, not for color-source
     API branching in the widget:
     - primary regular shapes: keep the current message shape, e.g.
       `Created shapes layer for "...".`;
     - primary point-radius shapes: add a short second line such as
       `Rendered point-radius shapes as napari points for faster display.`;
     - styled regular shapes: keep the current message shape;
     - styled point-radius shapes: add a short second line such as
       `Rendered point-radius shapes as napari points while preserving shapes styling semantics.`;
   - keep all terminology shapes-first:
     - titles remain `Shapes Layer ...` and `Styled Shapes ...`;
     - do not call these "points layers" in the title, because the source
       element is still a SpatialData shapes element;
     - mention napari points only as the rendering representation in the detail
       line;
   - preserve existing warning/info behavior:
     - skipped geometry warnings still apply only to generic shapes fallback
       paths;
     - table-backed unannotated-shape info remains visible for styled layers;
     - palette/coercion/instance-color feedback remains unchanged;
   - errors should remain user-facing and shapes-first:
     - point-radius detection failures should normally be silent fallback to
       generic shapes rendering, not widget errors;
     - explicit styling/alignment failures should keep `Styled Shapes Error`
       and avoid raw napari-layer terminology where possible;
   - add tests for:
     - `ShapesLoadResult.shapes_rendering_mode` is `"points"` for primary
       point-radius shapes and `"shapes"` for generic shapes;
     - existing primary point-backed shapes return the saved rendering mode when
       updated/reused;
     - point-backed styled direct shape-column layers report
       `shapes_rendering_mode == "points"` in their load result;
     - point-backed styled table-backed layers report
       `shapes_rendering_mode == "points"` in their load result;
     - primary widget feedback for point-radius shapes includes the
       point-radius-as-points detail line;
     - styled widget feedback for point-backed shapes includes the
       point-radius-as-points detail line while preserving palette/unannotated
       feedback lines;
     - generic shapes feedback does not mention point-radius rendering.

8. Point-backed radius-size scale control - specified
   - make napari's point-size UI useful without destroying the source
     radius-derived sizes;
   - add a point-backed-shapes-specific size sync helper, separate from the
     existing real-points helper
     `_connect_current_size_to_global_point_size(layer)`:
     - do not reuse the real-points helper, because it writes one absolute size
       to every point;
     - keep the helper near the point-backed-shapes construction/styling path
       or in `points_styling.py` with a clearly point-radius-shapes-specific
       name, for example
       `connect_current_size_to_radius_scaled_point_size(...)`;
   - store the unscaled radius-derived point sizes on the point-backed shapes
     layer under private Harpy attributes:
     - `original_radius_sizes = 2 * radius`, i.e. the sizes passed by
       `_prepare_napari_point_radius_shapes_layer_inputs(...)`;
     - `reference_size`, used to translate napari's `current_size` control into
       a scale factor;
     - the callback itself, so it is not garbage-collected;
   - choose a stable positive `reference_size` for the layer:
     - if all radii are equal, use that single diameter;
     - otherwise use a representative diameter such as the median finite
       original size;
     - if no positive finite size exists, raise a clear error rather than
       installing an invalid scale callback. This should normally be defensive
       because point-radius preparation already rejects non-positive radii;
   - initialize `layer.current_size` to `reference_size`, so the napari UI
     starts from a value that matches the displayed radius-derived sizes;
   - when `current_size` changes, interpret it as a scale target rather than an
     absolute size overwrite:
     - `scale = current_size / reference_size`;
     - `layer.size = original_radius_sizes * scale`;
     - if the user enters an invalid/non-positive `current_size`, ignore it or
       restore the reference size using the same positive-size semantics as
       existing point-size handling;
   - preserve relative radius differences for variable-radius elements;
   - for fixed-radius elements, this should feel like the normal napari point
     size control because every point scales together;
   - apply the same radius-size scale behavior to both primary and styled
     point-backed shapes:
     - both layer kinds represent the same source point/radius geometry;
     - direct shape-column and table-backed styling should affect colors, not
       reset or remove radius-size scaling;
     - styled point-backed layers should therefore preserve relative radii when
       `current_size` changes, just like primary point-backed layers;
   - keep existing color and symbol behavior unchanged:
     - symbol sync continues to apply to primary and styled point-backed shapes;
     - color sync remains primary-only;
     - color/symbol updates must not reset the stored original radius sizes or
       reference size;
   - on styled-layer updates/reuse, do not overwrite an existing user-chosen
     size scale unless the layer is recreated:
     - if the styled layer already exists and the user changed `current_size`,
       reapplying color styling should keep the current scaled sizes;
     - newly created styled layers start from the source radius-derived sizes
       and reference size;
   - add tests for:
     - fixed-radius point-backed shapes scale like ordinary point-size changes;
     - variable-radius point-backed shapes preserve relative size ratios;
     - `current_size` initializes to the expected reference size;
     - invalid or zero reference sizes are rejected or fall back clearly;
     - invalid or zero `current_size` does not destroy the radius-derived sizes;
     - styled point-backed direct shape-column layers preserve relative radii
       when `current_size` changes;
     - styled point-backed table-backed layers preserve relative radii when
       `current_size` changes;
     - reapplying styled direct/table-backed colors to an existing point-backed
       layer preserves the user's current radius-size scale;
     - color/symbol updates do not reset the original radius-derived size
       state.
