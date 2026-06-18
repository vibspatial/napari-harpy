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

For Slice 1, persist holes as Shapely `Polygon` objects with
interior rings in the GeoDataFrame geometry column.

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
- no `MultiPolygon` editing support
- no nested hole support; a hole inside a hole represents island/multipolygon
  semantics and must be rejected or deferred
- no change to linked-table behavior

Suggested work:

1. Move or duplicate the adapter's hole-path encoder into a shared core helper,
   for example `core/shapes_geometry.py`.
2. Add a decoder that converts a napari polygon vertex path back into
   `Polygon(shell, holes=[...])`.
3. Update `_polygon_shape_to_polygon(...)` to use the decoder:
   - simple polygon paths with no hole separators still become simple polygons
   - adapter-encoded hole paths become polygons with interiors
   - malformed paths raise a clear `ValueError`
4. Keep `MultiPolygon` edit-existing rejection unchanged for Slice 1.
5. Keep `line` and `path` rows unsupported unless a separate design is made for
   them.

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

Tests to add:

- converting a simple polygon is unchanged
- `_polygon_to_napari_path(Polygon(..., holes=[...]))` followed by the save
  converter preserves `len(geometry.interiors)`, area, bounds, and validity
- create-new save can write a polygon with an interior ring
- edit-existing save can round-trip an unchanged polygon with an interior ring
- a hole-inside-hole / island-in-hole path is rejected as unsupported
- invalid encoded paths fail without mutating `layer.features` or `sdata`

Slice 1 acceptance criteria:

- An existing SpatialData shapes row with a Shapely `Polygon` interior opens in
  the annotation widget.
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
  message for the MVP.
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

## Slice 3 - Future Extensions

These are useful but should not block Slice 1 or Slice 2:

- Support `MultiPolygon` edit-existing sessions. This requires revisiting the
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
- Saving an unchanged polygon-with-hole preserves the hole.
- Hole-inside-hole / island-in-hole geometry is explicitly rejected for Slice 1.
- The shared converter can persist hole-encoded polygon rows in both
  create-new and edit-existing save paths, without adding UI for creating
  those holes.
- Simple polygon, rectangle, and ellipse save behavior remains unchanged.

Slice 2:

- A selected shell plus selected contained polygons can be converted into one
  hole-bearing annotation row from the widget.
- The shell row keeps its source identity and feature values.
- Invalid selections fail without mutating the layer.

Slice 3:

- `MultiPolygon` edit-existing behavior remains explicitly unsupported until a
  separate roadmap item changes the row-mapping contract.
