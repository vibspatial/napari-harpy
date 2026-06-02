# Render Point-Radius Shapes As Napari Points

Status: Slice 2 implemented; Slices 3-6 pending

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

3. Primary point-radius layer lifecycle
   - extend the adapter so `ensure_shapes_loaded(...)` can create a napari
     `Points` layer for qualifying point-radius shapes elements;
   - bind the layer semantically as a shapes element while recording that the
     viewer representation is points-backed;
   - preserve source row id mapping, source index display/status metadata,
     coordinate system, skipped count, and layer replacement/reuse behavior;
   - make status text for hover match the current shapes source-index/status
     behavior as closely as napari `Points` allows;
   - add tests for primary loading, layer reuse, binding metadata, status text,
     duplicate indices, named index display, and fallback to `Shapes` when the
     element does not qualify.

4. Direct shape-column styling on points-backed shapes
   - add styling support for `ShapeColumnColorSourceSpec` when the styled layer
     is a napari `Points` layer representing point-radius shapes;
   - reuse the existing shape-column value alignment by source row id;
   - map categorical values, continuous values, instance-like values, missing
     values, and palette/coercion feedback to napari point face/edge colors;
   - keep styled layer naming and adapter identity consistent with the existing
     styled shapes layer names;
   - add tests for categorical, continuous, missing values, color companion
     columns, duplicate GeoDataFrame index values, named index display, and
     fill/edge behavior.

5. Table-backed styling on points-backed shapes
   - add styling support for `TableColorSourceSpec` when the styled layer is a
     napari `Points` layer representing point-radius shapes;
   - reuse the same table-to-source-row alignment rules as styled shapes,
     including column-backed and index-backed `instance_key` resolution;
   - preserve partial table coverage semantics: unannotated points transparent,
     annotated points with missing selected values gray;
   - support observation columns, `X[:, var_name]`, and instance colors;
   - preserve palette/coercion feedback and unannotated point counts;
   - add tests for `.obs` categorical/continuous values, `X[:, var_name]`,
     instance colors, partial table coverage, missing selected values,
     duplicate shape instance identities, duplicate table instance errors, and
     named-index instance identity.

6. Widget and feedback polish
   - make the shapes card/status feedback distinguish between regular shapes
     rendering and point-radius shapes rendered as points;
   - keep the existing color-source controls unchanged at the data-model level;
   - report point-radius rendering mode in successful load/status feedback;
   - ensure errors from detection, fallback, and styled point-radius alignment
     are user-facing and phrased in shapes terminology, not raw napari layer
     terminology;
   - add tests for widget request dispatch, status card feedback, and fallback
     messages.
