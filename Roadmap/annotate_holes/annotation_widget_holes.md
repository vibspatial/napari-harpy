# Annotation Widget Support For Holes

Investigation date: 2026-06-18

Scope: investigate how the current napari-harpy shapes annotation widget could
create and save annotations with holes. No implementation changes are included
in this document.

## Summary

napari-harpy already has the important half of hole support: visualization.
When a SpatialData shapes row is a Shapely `Polygon` with interior rings, the
viewer adapter encodes the exterior and interior rings into one napari Shapes
row so napari can render the hole.

The missing half is the reverse path. The annotation save code currently treats
every napari polygon row as one exterior ring and calls `Polygon(coordinates)`.
For a hole-encoded row this produces an invalid polygon, so an existing
polygon-with-hole can be opened correctly but cannot be saved back unchanged.

Implementation slices overview:

1. Slice 1: preserve existing holes upon save by adding a shared
   encoder/decoder for napari hole paths. This first slice should explicitly
   reconstruct a Shapely polygon with `Polygon(shell, holes=[...])`; it should
   not add any new hole-creation UI yet.
2. Slice 2: create holes with a small QuPath-like `Subtract selected`
   operation that subtracts selected polygons from a containing polygon.
3. Slice 3: broaden complex geometry support only after the basic
   polygon-with-holes path is stable.

## Current napari-harpy State

The shapes annotation widget has three entry paths, but they all converge on
the same save converter:

- create-new: `ShapesAnnotation._open_create_new_annotation_layer(...)`
  creates an empty primary Shapes layer.
- edit-existing: `_open_existing_annotation_layer(...)` loads a shapes element
  through the viewer adapter and stores a defensive source GeoDataFrame copy.
- adopted native layer: the widget can adopt a napari Shapes layer created or
  imported outside the widget.

The save boundary is
[`napari_shapes_layer_to_geodataframe`](../../src/napari_harpy/core/shapes_annotation.py#L236-L305).
It walks each napari row, accepts `polygon`, `rectangle`, and `ellipse`, and
produces one Shapely `Polygon` per napari row.

The current geometry choke point is
[`_polygon_shape_to_polygon`](../../src/napari_harpy/core/shapes_annotation.py#L626-L630),
which converts the whole vertex array into one exterior ring. Validation then
uses
[`_make_valid_polygon`](../../src/napari_harpy/core/shapes_annotation.py#L666-L670).
That is correct for simple polygons, but not for the hole path produced by the
viewer adapter.

Visualization is implemented in the opposite direction in
[`_polygon_to_napari_path`](../../src/napari_harpy/viewer/adapter.py#L2247-L2261).
The adapter orients the polygon, appends the exterior ring, then appends each
interior ring followed by the exterior anchor. The comments there explicitly
describe this as the napari path encoding used to preserve holes.

The edit-existing validator currently accepts only Shapely `Polygon` rows:
[`_validate_existing_shapes_source_geodataframe`](../../src/napari_harpy/core/shapes_annotation.py#L416-L439).
That does not reject holes, because a polygon with interiors is still a
`Polygon`. It does reject `MultiPolygon`, and the widget has a test for this
guardrail in
[`test_shapes_annotation_widget_open_existing_target_rejects_multipolygon_source`](../../tests/test_shapes_annotation_widget.py#L784-L800).

The widget also requires a one-to-one source-row to napari-row mapping before
opening an existing shapes element for editing:
[`_validate_opened_existing_shapes_layer`](../../src/napari_harpy/widgets/shapes_annotation/widget.py#L725-L770).
This works for a single `Polygon` with holes because it renders as one napari
row. It does not work for `MultiPolygon`, where one source row can expand into
multiple rendered rows.

After a successful save, the widget refreshes the source snapshot and binding
as if each saved GeoDataFrame row maps to one napari row:
[`_update_annotation_session_after_successful_save`](../../src/napari_harpy/widgets/shapes_annotation/widget.py#L850-L891).
That is fine for Slice 1 if we stay with one `Polygon` row per
annotation.

Local verification with the current environment showed the failure mode:

- input: Shapely `Polygon` with one interior ring
- display encoding: `_polygon_to_napari_path(...)`
- save conversion: `napari_shapes_layer_to_geodataframe(...)`
- result: `ValueError: Shape row 0 cannot be converted to a valid polygon.`

Local versions checked:

- napari `0.7.0`
- Shapely `2.1.2`
- GeoPandas `1.1.3`
- SpatialData `0.7.2`
- harpy `0.4.1`

## How QuPath Supports This

QuPath source checked: current `main` branch at commit
[`bb65edf0e7c5ebf28076662cf320ba9a4078d0bb`](https://github.com/qupath/qupath/tree/bb65edf0e7c5ebf28076662cf320ba9a4078d0bb).

QuPath's core model is geometry-first. Annotation operations call JTS geometry
operations and then convert the result back to an ROI.

Relevant source references:

- `RoiTools.CombineOp` exposes `ADD`, `SUBTRACT`, and `INTERSECT`.
  `SUBTRACT` uses `area1.difference(area2)`:
  <https://github.com/qupath/qupath/blob/bb65edf0e7c5ebf28076662cf320ba9a4078d0bb/qupath-core/src/main/java/qupath/lib/roi/RoiTools.java#L76-L120>
- `RoiTools.subtract(...)` subtracts one or more ROIs from a main ROI, using
  the union of subtractor ROIs for multi-subtract:
  <https://github.com/qupath/qupath/blob/bb65edf0e7c5ebf28076662cf320ba9a4078d0bb/qupath-core/src/main/java/qupath/lib/roi/RoiTools.java#L270-L300>
- The selected-annotation command puts the main selected annotation first, then
  for `SUBTRACT` subtracts the union of the remaining selected ROIs from it:
  <https://github.com/qupath/qupath/blob/bb65edf0e7c5ebf28076662cf320ba9a4078d0bb/qupath-gui-fx/src/main/java/qupath/lib/gui/commands/Commands.java#L1089-L1164>
- The GUI exposes `Merge selected`, `Subtract selected`, and `Intersect
  selected` in a multiple-annotation edit menu:
  <https://github.com/qupath/qupath/blob/bb65edf0e7c5ebf28076662cf320ba9a4078d0bb/qupath-gui-fx/src/main/java/qupath/lib/gui/tools/GuiTools.java#L1145-L1158>
- The brush tool also has subtract behavior. In subtract mode it computes
  `shapeROI.getGeometry().difference(shapeDrawn)` and converts the result back
  through `GeometryTools.geometryToROI(...)`:
  <https://github.com/qupath/qupath/blob/bb65edf0e7c5ebf28076662cf320ba9a4078d0bb/qupath-gui-fx/src/main/java/qupath/lib/gui/viewer/tools/handlers/BrushToolEventHandler.java#L333-L400>
- QuPath treats complex outputs as geometry ROIs. `geometryToROI(...)` returns
  `GeometryROI` for multi-part geometries and polygons with interior rings:
  <https://github.com/qupath/qupath/blob/bb65edf0e7c5ebf28076662cf320ba9a4078d0bb/qupath-core/src/main/java/qupath/lib/roi/GeometryTools.java#L1381-L1419>
- QuPath also has explicit hole removal. `fillHoles(...)` removes interior
  rings and unions the result when needed:
  <https://github.com/qupath/qupath/blob/bb65edf0e7c5ebf28076662cf320ba9a4078d0bb/qupath-core/src/main/java/qupath/lib/roi/GeometryTools.java#L817-L833>

Lessons for napari-harpy:

- Holes should be real geometry, not a styling convention.
- The user-facing operation can be simple: subtract selected region(s) from a
  main region.
- The persistence model should store the result as one polygon with interiors
  when the result is one connected polygon.
- More complex boolean results, especially `MultiPolygon`, need a separate
  design because napari-harpy currently assumes one editable source row maps to
  one napari row.

## Recommended Geometry Contract

For Slice 1 and Slice 2, persist holes as Shapely `Polygon` objects with
interior rings in the GeoDataFrame geometry column. Do not support
`MultiPolygon` annotation in these slices.

Do not persist hole rings as separate shapes rows. Separate rows would break
table annotation semantics and would make a hole look like a positive region to
downstream SpatialData consumers.

Use one napari Shapes row to display one polygon-with-holes while editing. The
display encoding should remain compatible with the existing adapter:

- `shape_type == "polygon"`
- `layer.data[row]` is a 2D `(n_vertices, 2)` array in napari `(y, x)` order
- the first ring is the exterior
- each interior ring follows the exterior
- the exterior anchor separates rings

This contract should live in a shared geometry helper, not only in
`viewer/adapter.py`, so both loading and saving use the same encoding rules.

`MultiPolygon` is explicitly out of scope for annotation support in Slice 1 and
Slice 2. Existing `MultiPolygon` shapes elements should continue to be rejected
by the annotation widget, and any operation that would produce a `MultiPolygon`
must fail clearly rather than splitting, merging, or silently changing row
identity.

## Hole State Through The Current Pipeline

Given the current codebase, hole information should be tracked in geometry, not
in `layer.features` or in a separate sidecar structure.

The intended Slice 1 lifecycle is:

1. SpatialData stores one shapes row whose geometry is a Shapely `Polygon` with
   direct interior rings.
2. The viewer adapter renders that source row as one napari Shapes row.
3. The napari row keeps hole topology inside `layer.data[row]` using the
   existing `_polygon_to_napari_path(...)` encoding.
4. `layer.features.iloc[row]` keeps row metadata, especially the source
   GeoDataFrame index, but does not store hole boundaries.
5. On save, `napari_shapes_layer_to_geodataframe(...)` decodes
   `layer.data[row]` through `napari_path_to_polygon(...)`.
6. The helper validates the path, constructs `Polygon(shell, holes=holes)`
   internally, and the saved GeoDataFrame receives that Shapely `Polygon` as the
   row geometry.
7. `hp.sh.add_shapes(...)` writes that GeoDataFrame back into the SpatialData
   object.

The existing adapter already sets up this model when rendering source shapes.
`_prepare_napari_shapes_layer_inputs(...)` appends one rendered napari row for
each renderable source `Polygon`, and stores only the source index in
`features`. For a simple `Polygon` with holes this remains one source row to
one rendered row. That means the current widget validation and post-save
binding refresh can stay one-to-one for Slice 1.

The encoded napari path has this shape, after converting from napari `(y, x)`
to Shapely `(x, y)`:

```text
A B C D A   E F G H E   A
```

Here `A B C D A` is the closed exterior shell, `E F G H E` is the closed
interior ring, and the final `A` is the repeated shell-start anchor that
separates the hole from any later rings. This is not based on consecutive
duplicate vertices such as `vertex[i] == vertex[i + 1]`; the decoder looks for
the shell-start anchor recurring later in the path, and for each hole to close
on its own start coordinate.

The repeated shell start is the separator that lets napari render the polygon
with holes. Slice 1 should make the reverse operation explicit and shared:

```python
geometry = napari_path_to_polygon(vertices)
```

The helper constructs `Polygon(shell, holes=holes)` internally after validating
the decoded rings.

No new feature column is needed for hole bookkeeping. The source identity
continues to live in the existing `source_shapes_index_feature_name` column,
and the hole geometry lives in the Shapely geometry itself.

The important limitation is that this is only deterministic for paths that
match the documented encoding. If a user edits repeated vertices so the ring
separators are no longer parseable, the save path should raise a clear error.
It should not try to infer holes from arbitrary self-touching polygon paths.

## Implementation Slices

The slices are intentionally ordered so the first implementation does not
change the annotation UI. Slice 1 only makes the existing visualization format
round-trip through save.

## Slice 1 - Preserve Existing Holes Upon Save

Goal: an existing `Polygon` with interiors can be opened in the annotation
widget, saved without edits, and round-tripped without losing or corrupting the
holes.

This is the first implementation slice. The save path should stop treating the
entire napari vertex path as one exterior boundary. Instead, it should decode
the path into:

- `shell`: the exterior ring coordinates in Shapely `(x, y)` order
- `holes`: zero or more interior ring coordinate sequences in Shapely `(x, y)`
  order

and then construct the persisted geometry with:

```python
Polygon(shell, holes=holes)
```

The slice is complete when existing hole-bearing polygons can be saved back
unchanged. Creating new holes with a widget action is Slice 2.

Primary code areas:

- `src/napari_harpy/viewer/adapter.py`
- `src/napari_harpy/core/shapes_annotation.py`
- a possible shared helper such as `src/napari_harpy/core/shapes_geometry.py`
- `tests/test_shapes_annotation.py`
- `tests/test_shapes_annotation_widget.py`

Out of scope:

- no new widget buttons
- no `Subtract selected` operation
- no `MultiPolygon` annotation or editing support
- no nested hole support; a hole inside a hole represents island/multipolygon
  semantics and must be rejected or deferred
- no change to linked-table behavior

Slice 1 implementation breakdown:

### Slice 1A - Geometry Helper Only

Status: implemented.

Implemented in `src/napari_harpy/core/shapes_geometry.py` with
`polygon_to_napari_path(...)` and `napari_path_to_polygon(...)`. The viewer
adapter now delegates its polygon path encoding to the shared helper, and
`tests/test_shapes_geometry.py` covers the helper contract without touching the
annotation widget, SpatialData writes, or save behavior.

Goal: implement and unit-test the geometry conversion helper without changing
the annotation widget, SpatialData writes, or save behavior yet.

Suggested work:

1. Add a small helper module, for example
   `src/napari_harpy/core/shapes_geometry.py`.
2. Move or mirror the existing adapter encoder into that helper so loading and
   saving share one documented napari path contract.
3. Implement `napari_path_to_polygon(vertices) -> Polygon`, or a similarly
   named helper, that:
   - accepts a napari `(y, x)` vertex array
   - returns a Shapely `Polygon`
   - constructs holes explicitly with `Polygon(shell, holes=holes)`
   - keeps simple polygons without hole separators working
   - rejects malformed, nested-hole, overlapping-hole, and ambiguous paths
4. Add focused unit tests for the helper only. These tests should not require a
   napari `Shapes` layer or a SpatialData object.

Slice 1A acceptance criteria:

- simple polygon vertex arrays still decode to simple Shapely polygons
- adapter-encoded polygons with one or more holes decode to Shapely polygons
  with matching `interiors`
- ambiguous edited separator paths fail clearly
- hole-inside-hole / island-in-hole layouts fail clearly
- `MultiPolygon` is not introduced or accepted by this helper

### Slice 1B - Wire Helper Into `_polygon_shape_to_polygon`

Goal: make the annotation save converter use the helper for polygon rows while
keeping the surrounding save pipeline unchanged.

Status: specified; not implemented.

This is the first behavioral save-path change. It should preserve holes for
encoded napari polygon rows, but it should not change the annotation widget UI,
SpatialData write orchestration, edit-existing source validation, or
`MultiPolygon` policy.

Implementation shape:

```python
def _polygon_shape_to_polygon(vertices: object, *, row_index: int) -> Polygon:
    try:
        return napari_path_to_polygon(vertices)
    except ValueError as error:
        raise ValueError(
            f"Shape row `{row_index}` cannot be converted to a valid polygon: {error}"
        ) from error
```

The exact type annotation can follow the surrounding converter code, but the
important behavior is that `_polygon_shape_to_polygon(...)` becomes a thin
row-aware wrapper around `napari_path_to_polygon(...)`.

Layering contract:

- `src/napari_harpy/core/shapes_geometry.py` owns pure geometry/path
  interpretation. It knows about napari `(y, x)` path arrays and Shapely
  `Polygon`s, but it does not know about napari `Shapes` layers, row indexes,
  `layer.features`, SpatialData, or widgets.
- `src/napari_harpy/core/shapes_annotation.py` owns annotation save conversion.
  `_polygon_shape_to_polygon(...)` should keep existing row-aware error context
  and converter structure, but should stop duplicating polygon path parsing.
- `napari_path_to_polygon(...)` becomes the source of truth for interpreting one
  polygon or rectangle path row as a Shapely `Polygon`.
- `_coerce_vertices(...)` and `_make_valid_polygon(...)` in
  `shapes_annotation.py` can remain for ellipse conversion and other local save
  converter needs. They do not need to be removed or refactored in Slice 1B.

Suggested work:

1. Import `napari_path_to_polygon(...)` in
   `src/napari_harpy/core/shapes_annotation.py`.
2. Update `_polygon_shape_to_polygon(...)` to call the helper.
3. Wrap helper `ValueError`s with the napari shape row index while preserving
   the detailed helper message.
4. Keep `rectangle` rows flowing through the same polygon conversion as today.
   Rectangles are simple napari paths and should continue to save as simple
   Shapely polygons.
5. Keep `ellipse`, `line`, and `path` behavior unchanged.
6. Keep `MultiPolygon` edit-existing rejection unchanged.
7. Preserve the current failure behavior that invalid geometry conversion does
   not mutate `layer.features`.

Behavior to preserve:

- simple user-drawn polygon rows save as simple `Polygon`
- closed simple polygon rows save as simple `Polygon`
- rectangle rows save as simple `Polygon`
- ellipse rows still use `_ellipse_shape_to_polygon(...)`
- line and path rows remain rejected
- generated or preserved `layer.features` IDs are written only after all
  geometry rows validate

New behavior:

- adapter-encoded hole paths save as Shapely `Polygon` with `interiors`
- malformed hole encodings fail clearly before `layer.features` is rewritten
- shell-touching holes fail because the helper requires holes to be fully inside
  the shell
- nested holes, overlapping holes, and edge-sharing holes fail

Tests to add in `tests/test_shapes_annotation.py`:

1. `napari_shapes_layer_to_geodataframe(...)` preserves one hole from
   `polygon_to_napari_path(...)`.
2. `napari_shapes_layer_to_geodataframe(...)` preserves multiple holes.
3. Invalid or ambiguous encoded hole paths fail with row context and do not
   mutate `layer.features`.
4. Existing simple polygon, rectangle, ellipse, line/path rejection, and feature
   identity tests continue to pass.

Slice 1B non-goals:

- no annotation widget behavior changes
- no SpatialData round-trip test yet
- no edit-existing widget test yet
- no `MultiPolygon` support
- no hole creation or subtraction UI

Slice 1B acceptance criteria:

- `napari_shapes_layer_to_geodataframe(...)` converts an encoded hole path into
  a GeoDataFrame row whose geometry is `Polygon(shell, holes=holes)`
- simple polygon, rectangle, and ellipse conversion tests continue to pass
- invalid or ambiguous encoded paths fail before `layer.features` is rewritten

### Slice 1C - End-To-End Annotation Save Round Trip

Goal: prove that existing hole-bearing SpatialData shapes can be opened in the
annotation widget and saved back without losing hole geometry.

Suggested work:

1. Add create/edit tests that start from a Shapely `Polygon` with interiors.
2. Open the existing shapes element through the annotation widget.
3. Save without editing.
4. Assert the saved SpatialData shapes geometry still has the expected
   interiors, bounds, area, index, and non-geometry columns.
5. Keep `MultiPolygon` rejection covered by the existing widget behavior.

Slice 1C acceptance criteria:

- an unchanged polygon-with-hole round-trips through the annotation widget save
  path
- saved source metadata and row identity remain stable
- backed SpatialData save behavior is covered if practical

Decoder notes:

- Convert napari `(y, x)` to Shapely `(x, y)`.
- The adapter encoding repeats the exterior anchor after the exterior and after
  every hole.
- The first segment is the exterior ring.
- Later segments are interior rings.
- Pass the parsed rings to Shapely as `Polygon(shell, holes=holes)`, not as
  `Polygon(all_coordinates)`.
- Explicitly validate that all parsed holes are direct holes in the shell:
  each hole must be contained by the shell, holes must not contain other holes,
  and holes must not overlap each other. Cases that imply a hole inside a hole
  should fail with a clear unsupported-geometry error rather than being
  silently coerced.
- Validate with Shapely after decoding.
- Be conservative when parsing ambiguous paths. If a user edits the vertex
  sequence into an invalid ring layout, report a save error rather than trying
  to guess.

Decoder algorithm for adapter-encoded paths:

1. Coerce `vertices` to a finite `(n, 2)` array.
2. Convert from napari `(y, x)` to Shapely `(x, y)`.
3. Let `anchor = coordinates_xy[0]`.
4. Find the first later occurrence of `anchor`; this closes the exterior ring.
5. Use coordinates from the start through that occurrence as `shell`.
6. If there are no coordinates after the shell, return `Polygon(shell)`.
7. Otherwise, parse the remaining coordinates as zero or more hole chunks.
   Each hole chunk starts immediately after a shell-anchor separator and ends
   immediately before the next shell-anchor separator.
8. For each chunk, require a closed ring with at least four coordinates. The
   chunk itself should close on its own first coordinate.
9. Do not treat consecutive duplicate vertices as the hole marker. The hole
   marker is structural: a closed hole ring followed by the repeated shell
   anchor.
10. Require the path to end on the shell anchor. If a later shell-anchor
   separator is missing, fail.
11. Validate direct-hole topology:
    - every hole is contained by the shell
    - no hole contains another hole
    - holes do not overlap each other
12. Construct `Polygon(shell, holes=holes)` and validate the resulting Shapely
    geometry.

Simple user-drawn polygons may not have adapter-style hole separators. If no
valid shell-anchor separator pattern is present, the converter can keep the
existing simple path behavior and call `Polygon(coordinates_xy)` for that row.
Once an adapter-style separator is detected, the row must fully satisfy the
hole-path grammar above or fail clearly.

Strict failure policy:

- If repeated vertices or ring separators no longer match the documented
  encoding, the decoder must fail with a clear user-facing save error.
- The decoder must not infer hole boundaries from arbitrary self-touching
  paths, bridge edges, or vertex repetition patterns.
- Failed decoding must leave both `layer.features` and the SpatialData object
  unchanged.

Tests to add:

- converting a simple polygon is unchanged
- `_polygon_to_napari_path(Polygon(..., holes=[...]))` followed by the save
  converter preserves `len(geometry.interiors)`, area, bounds, and validity
- existing `MultiPolygon` shapes elements remain rejected by the annotation
  widget
- create-new save can write a polygon with an interior ring
- edit-existing save can round-trip an unchanged polygon with an interior ring
- a hole-inside-hole / island-in-hole path is rejected as unsupported
- an ambiguous edited separator path fails clearly rather than guessing a hole
  layout
- invalid encoded paths fail without mutating `layer.features` or `sdata`

Slice 1 acceptance criteria:

- An existing SpatialData shapes row with a Shapely `Polygon` interior opens in
  the annotation widget.
- Existing SpatialData shapes rows with Shapely `MultiPolygon` geometry remain
  unsupported for annotation and fail clearly.
- Saving that layer without edits preserves the number of interiors, bounds,
  area, validity, index, and non-geometry columns.
- Simple polygon, rectangle, and ellipse save behavior remains unchanged.
- Hole-inside-hole geometry is explicitly unsupported and fails clearly.
- Malformed hole paths produce a clear save error and do not mutate
  `layer.features` or `sdata`.

## Slice 2 - Create Holes With "Subtract Selected"

Goal: a user can draw an outer polygon and one or more inner polygons, then
turn the inner polygons into holes in the outer polygon.

Recommended user flow:

1. Draw the outer positive polygon.
2. Draw one or more polygons inside it.
3. Select the outer polygon and the inner polygons.
4. Click a new widget action such as `Subtract selected` or `Cut hole`.
5. The layer replaces the selected rows with one polygon row that contains
   interior rings.
6. Saving persists that row as one Shapely `Polygon` with holes.

Why this is the best first UX:

- It mirrors QuPath's `Subtract selected` command.
- It avoids relying on users to edit low-level feature columns.
- It works with the existing napari Shapes layer rather than a custom drawing
  tool.
- It keeps table row identity intuitive: the outer polygon remains the
  annotation row; the subtracted shapes are temporary construction geometry.

Main-shape selection in napari needs a deterministic rule. QuPath has a main
selected object. napari Shapes exposes selected rows, but ordering may not be a
stable public contract. Safer options:

- MVP rule: among selected polygons, use the polygon that contains all other
  selected polygons as the shell. If exactly one such shell exists, subtract the
  rest. If none or multiple exist, show a status warning.
- Alternative rule: use the largest-area selected polygon as the shell, but
  only if it contains all other selected polygons.
- Later rule: add an explicit `Set shell` or `Use active shape as shell`
  interaction if napari exposes a reliable active-shape API.

Geometry operation:

- Convert selected napari rows to Shapely polygons using the Slice 1 decoder.
- Compute `shell.difference(unary_union(holes))`.
- Accept the result only when it is a non-empty valid `Polygon`.
- If the result is `MultiPolygon` or `GeometryCollection`, fail with a clear
  message. Slice 2 must not create or annotate `MultiPolygon` geometries.
- Encode the resulting polygon back to one napari polygon row using the shared
  hole-path encoder.
- Remove the subtractor rows from `layer.data`, `layer.shape_type`, and
  `layer.features`.
- Preserve the shell row's feature values and source row identity.

Metadata behavior:

- For create-new sessions, the shell row keeps or receives the generated
  `instance_id` during save.
- For edit-existing sessions, the shell row keeps its
  `source_shapes_index_feature_name` value, so `_edited_shapes_geodataframe_from_source(...)`
  preserves existing metadata for that annotation.
- Removing selected subtractor rows is a real row deletion. The widget already
  warns that linked tables are not updated when rows are added or removed, and
  that warning remains relevant.

Widget integration:

- Add a compact geometry-operation row near the existing `Create layer` and
  `Save shapes` controls.
- Enable the operation only when an annotation Shapes layer is open and at
  least two shape rows are selected.
- Report success/failure through the existing status-card pattern.
- Treat the operation as a normal layer edit; the existing snapshot/dirty logic
  should detect the changed geometry and feature rows.

Tests to add:

- selected outer + one inner polygon becomes one row with one interior ring
- selected outer + two inner polygons becomes one row with two interior rings
- shell feature values and source index are preserved
- subtractor feature rows are removed
- operation fails without layer mutation when selected polygons do not have a
  unique containing shell
- operation fails without layer mutation when the boolean result is not a
  single `Polygon`
- operation fails without layer mutation when subtraction would produce a
  `MultiPolygon`
- edit-existing save after subtraction preserves source metadata for the shell
  row

Slice 2 acceptance criteria:

- Selected outer + one inner polygon becomes one annotation row with one
  interior ring.
- Selected outer + multiple inner polygons becomes one annotation row with
  multiple interior rings.
- The shell row keeps its feature values and source identity.
- Subtractor rows are removed intentionally and linked-table warnings still
  apply when relevant.
- Subtraction results that would be `MultiPolygon` are rejected; Slice 2 does
  not support annotating `MultiPolygon`.

## Slice 3 - Future Extensions

These are useful but should not block Slice 1 or Slice 2:

- Support `MultiPolygon` annotation sessions. This requires revisiting the
  one-source-row to one-rendered-row assumption in widget validation and save
  session refresh.
- Add brush/eraser subtraction similar to QuPath's brush tool.
- Add `Fill holes` for selected rows, matching QuPath's explicit hole-removal
  operation.
- Add `Merge selected` and `Intersect selected` operations if boolean geometry
  tools become a broader annotation feature.
- Support converting complex boolean results into multiple annotation rows,
  with a clear policy for copied features and generated IDs.
- Add a visual affordance for construction geometry if users want to preview
  hole cutters before applying subtraction.

## Risks And Decisions

The largest risk is path decoding. The current adapter encoding is documented
locally, but arbitrary user edits can produce vertex paths that no longer have
clean ring separators. The save path should be strict and give a clear error
for ambiguous paths.

The second risk is boolean output shape. A contained subtractor produces the
desired `Polygon` with an interior ring. A subtractor crossing the exterior can
produce a smaller polygon, a split `MultiPolygon`, or an empty result. For the
MVP, accepting only a valid single `Polygon` keeps the existing widget model
intact.

The third risk is row identity. The implementation should preserve the shell
row's source identity and delete temporary cutter rows. Grouping holes only at
save time would be harder to reason about because cutter rows would still look
like positive annotations while editing.

The fourth risk is linked tables. Current annotation editing already warns that
linked tables are not updated when rows are added or removed. Hole subtraction
can remove cutter rows, so the same warning should stay visible for linked
shapes elements.

## Roadmap Acceptance Criteria

Slice 1:

- Existing SpatialData shapes with Shapely `Polygon` interiors open in the
  annotation widget.
- Existing SpatialData shapes with Shapely `MultiPolygon` geometry remain
  rejected by the annotation widget.
- Saving an unchanged polygon-with-hole preserves the hole.
- Hole-inside-hole / island-in-hole geometry is explicitly rejected for Slice 1.
- The shared converter can persist hole-encoded polygon rows in both
  create-new and edit-existing save paths, without adding UI for creating
  those holes.
- Simple polygon, rectangle, and ellipse save behavior remains unchanged.

Slice 2:

- A selected shell plus selected contained polygons can be converted into one
  hole-bearing annotation row from the widget.
- Any subtraction result that would be a `MultiPolygon` is rejected.
- The shell row keeps its source identity and feature values.
- Invalid selections fail without mutating the layer.

Slice 3:

- `MultiPolygon` annotation remains explicitly unsupported until a separate
  roadmap item changes the row-mapping contract.
