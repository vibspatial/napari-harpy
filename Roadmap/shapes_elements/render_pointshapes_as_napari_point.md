# Render Point-Radius Shapes As Napari Points

Status: Spec extracted; implementation pending

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
- Preserve the existing coordinate-system transform semantics before extracting
  coordinates and radii.
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

Keep a vectorized `Shapes` ellipse fallback for cases where actual napari
`Shapes` layer semantics are required. The fallback should build ellipse vertex
arrays from coordinate/radius arrays and avoid `iterrows()`.

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
- parity between the points fast path and vectorized ellipse fallback where
  behavior should match.

## Implementation Slices

1. Point-radius detection and preparation
   - add a small helper that determines whether a transformed shapes element can
     be rendered as point-radius shapes;
   - require all geometries to be `Point` and require a `radius` column with
     finite positive numeric values for every row;
   - extract coordinates, radii, source index values, and source row ids as
     arrays without `iterrows()`;
   - return a prepared-input object for point-radius shapes, separate from the
     existing `_NapariShapesLayerInputs`;
   - keep invalid/mixed cases falling back to the existing `Shapes` path;
   - add tests for valid points/radii, missing `radius`, invalid radius values,
     mixed geometries, duplicate GeoDataFrame index values, and named index
     values.

2. Primary point-radius layer lifecycle
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

3. Direct shape-column styling on points-backed shapes
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

4. Table-backed styling on points-backed shapes
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

5. Widget and feedback polish
   - make the shapes card/status feedback distinguish between regular shapes
     rendering and point-radius shapes rendered as points;
   - keep the existing color-source controls unchanged at the data-model level;
   - report point-radius rendering mode in successful load/status feedback;
   - ensure errors from detection, fallback, and styled point-radius alignment
     are user-facing and phrased in shapes terminology, not raw napari layer
     terminology;
   - add tests for widget request dispatch, status card feedback, and fallback
     messages.

6. Optional vectorized ellipse fallback
   - if actual napari `Shapes` semantics are needed for point-radius elements,
     add a vectorized ellipse fallback that avoids `iterrows()`;
   - build ellipse vertex arrays from coordinate/radius arrays;
   - keep source row ids and source-index features aligned with the points fast
     path;
   - add parity tests between the points fast path and ellipse fallback for the
     cases where behavior should match.
