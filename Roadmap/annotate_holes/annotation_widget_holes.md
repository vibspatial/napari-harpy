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
   - Slice 1C proves unchanged hole-bearing annotations round-trip through the
     widget save path.
   - Slice 1D proves ordinary non-anchor vertices of existing holes can be
     edited and saved.
   - Slice 1E is a must-fix follow-up for editing anchor/separator vertices,
     because napari's direct-edit path treats the repeated anchors as ordinary
     duplicate vertices.
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
[`_napari_polygon_vertices_to_shapely_polygon`](../../src/napari_harpy/core/shapes_annotation.py#L626-L630),
which converts the whole vertex array into one exterior ring. Validation then
uses
[`_make_valid_polygon`](../../src/napari_harpy/core/shapes_annotation.py#L666-L670).
That is correct for simple polygons, but not for the hole path produced by the
viewer adapter.

Visualization is implemented in the opposite direction in
[`_shapely_polygon_to_napari_polygon_vertices`](../../src/napari_harpy/viewer/adapter.py#L2247-L2261).
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
- display encoding: `_shapely_polygon_to_napari_polygon_vertices(...)`
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
   existing `_shapely_polygon_to_napari_polygon_vertices(...)` encoding.
4. `layer.features.iloc[row]` keeps row metadata, especially the source
   GeoDataFrame index, but does not store hole boundaries.
5. On save, `napari_shapes_layer_to_geodataframe(...)` decodes
   `layer.data[row]` through `napari_polygon_vertices_to_shapely_polygon(...)`.
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
geometry = napari_polygon_vertices_to_shapely_polygon(vertices)
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

## Napari Direct-Edit Behavior For Hole Paths

Local testing with napari `0.7.0` shows that hole rendering and direct vertex
editing use different assumptions.

The render path understands napari's flat hole encoding. During triangulation,
napari normalizes duplicate coordinates and removes edges that are visited
twice. This is why an encoded path such as:

```text
A B C D A   E F G H E   A
```

renders as one exterior ring plus one hole. The bridge edge `A-E` is traversed
twice and removed from the rendered outline.

The direct-edit path does not preserve that topology. It exposes the raw vertex
array as editable vertices. In the local CSV test row, napari sees:

```text
exterior anchor copies: 0, 5, 12
hole anchor copies:     6, 11
```

Dragging an ordinary non-anchor hole vertex works because no separator
invariant changes. The helper still decodes the row, the geometry saves, and
the result can be reloaded.

Dragging an anchor/separator vertex breaks the encoding. For example, dragging
only vertex `11` moves the hole-closing copy but not vertex `6`, so the hole no
longer closes on itself and the duplicated bridge edge is no longer removed.
This produces the visible collapse/bridge lines in napari before save.

The relevant napari behavior is:

- rendering removes duplicate traversed edges in
  `normalize_vertices_and_edges(...)`
- direct-edit vertex picking uses `displayed_vertices`, which are raw vertex
  rows
- direct dragging only special-cases `vertices[0] == vertices[-1]` for simple
  closed polygons; it does not know about hole anchors or exterior separators

This means editable holes need two levels of support:

1. non-anchor vertex edits, which already match the current encoding model
2. anchor/separator vertex edits, which need explicit topology-preserving
   synchronization

Until the second level is implemented, anchor/separator edits should be treated
as unsupported or repaired immediately. Save-time guessing is not sufficient
because the user-visible napari rendering can collapse during the drag.

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
`shapely_polygon_to_napari_polygon_vertices(...)` and `napari_polygon_vertices_to_shapely_polygon(...)`. The viewer
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
3. Implement `napari_polygon_vertices_to_shapely_polygon(vertices) -> Polygon`, or a similarly
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

### Slice 1B - Wire Helper Into `_napari_polygon_vertices_to_shapely_polygon`

Goal: make the annotation save converter use the helper for polygon rows while
keeping the surrounding save pipeline unchanged.

Status: implemented.

This is the first behavioral save-path change. It should preserve holes for
encoded napari polygon rows, but it should not change the annotation widget UI,
SpatialData write orchestration, edit-existing source validation, or
`MultiPolygon` policy.

Implemented by making `_napari_polygon_vertices_to_shapely_polygon(...)` a row-aware wrapper
around `napari_polygon_vertices_to_shapely_polygon(...)` and adding converter-level tests in
`tests/test_shapes_annotation.py`. Slice 1C remains responsible for proving the
end-to-end annotation widget save round trip.

Implementation shape:

```python
def _napari_polygon_vertices_to_shapely_polygon(vertices: object, *, row_index: int) -> Polygon:
    try:
        return napari_polygon_vertices_to_shapely_polygon(vertices)
    except ValueError as error:
        raise ValueError(
            f"Shape row `{row_index}` cannot be converted to a valid polygon: {error}"
        ) from error
```

The exact type annotation can follow the surrounding converter code, but the
important behavior is that `_napari_polygon_vertices_to_shapely_polygon(...)` becomes a thin
row-aware wrapper around `napari_polygon_vertices_to_shapely_polygon(...)`.

Layering contract:

- `src/napari_harpy/core/shapes_geometry.py` owns pure geometry/path
  interpretation. It knows about napari `(y, x)` path arrays and Shapely
  `Polygon`s, but it does not know about napari `Shapes` layers, row indexes,
  `layer.features`, SpatialData, or widgets.
- `src/napari_harpy/core/shapes_annotation.py` owns annotation save conversion.
  `_napari_polygon_vertices_to_shapely_polygon(...)` should keep existing row-aware error context
  and converter structure, but should stop duplicating polygon path parsing.
- `napari_polygon_vertices_to_shapely_polygon(...)` becomes the source of truth for interpreting one
  polygon or rectangle path row as a Shapely `Polygon`.
- `_coerce_vertices(...)` and `_make_valid_polygon(...)` in
  `shapes_annotation.py` can remain for ellipse conversion and other local save
  converter needs. They do not need to be removed or refactored in Slice 1B.

Suggested work:

1. Import `napari_polygon_vertices_to_shapely_polygon(...)` in
   `src/napari_harpy/core/shapes_annotation.py`.
2. Update `_napari_polygon_vertices_to_shapely_polygon(...)` to call the helper.
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
- ellipse rows still use `_napari_ellipse_vertices_to_shapely_polygon(...)`
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
   `shapely_polygon_to_napari_polygon_vertices(...)`.
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

Status: not implemented until widget-level hole round-trip tests exist.

Suggested work:

1. Add an edit-existing widget test that starts from a Shapely `Polygon` with
   interiors in a synthetic SpatialData object.
2. Open that existing shapes element through the annotation widget.
3. Save without editing.
4. Assert the saved SpatialData shapes geometry still has the expected
   interiors, bounds, area, index, and non-geometry columns.
5. Add an adopted-native-layer widget test for a napari Shapes layer that came
   from napari's native shapes CSV path.
6. Keep `MultiPolygon` rejection covered by the existing widget behavior.

The Slice 1C tests should cover two end-to-end routes:

1. Existing SpatialData route:
   - Build a small in-memory `SpatialData` object containing only one shapes
     element.
   - The shapes element should contain one `Polygon(shell, holes=[...])` row
     plus at least one ordinary simple polygon row.
   - Open that shapes element through `ShapesAnnotation`.
   - Save without editing.
   - Assert the hole geometry and row metadata are preserved.
2. Native napari load/adoption route:
   - Use `napari_builtins.io.napari_write_shapes(...)` to write a temporary
     CSV with one encoded polygon-with-hole and one simple polygon.
   - Read it back with napari's built-in shapes CSV reader, mirroring a user
     loading the file through the napari UI.
   - Construct a native napari `Shapes` layer from the loaded layer data and add
     it to the viewer.
   - Let the annotation widget adopt that native layer.
   - Save it into SpatialData.
   - Assert the saved SpatialData shape has one interior ring for the first row
     and a simple polygon for the second row.

Use a synthetic version of the manually tested geometry, not the external
Xenium test data path. The test should keep the same non-axis-aligned shape
layout because it is realistic and catches coordinate-order mistakes, but it
should avoid depending on
`/Users/arne.defauw/VIB/DATA/test_data/sdata_xenium_3_6_26.zarr`.

Use the following example as the canonical Slice 1C geometry fixture, including
for tests that start from Shapely polygons with interiors. The fixture is
defined in napari `(y, x)` order first because it mirrors the manually tested
CSV/native-layer workflow, then tests should convert it to Shapely `(x, y)` as
needed.

```python
center_y = 1000.0
center_x = 2000.0

polygon_1_shell_yx = np.array(
    [
        [center_y - 350, center_x - 280],
        [center_y - 420, center_x + 120],
        [center_y - 120, center_x + 320],
        [center_y + 180, center_x + 140],
        [center_y + 120, center_x - 240],
    ],
    dtype=float,
)

polygon_1_hole_yx = np.array(
    [
        [center_y - 150, center_x - 40],
        [center_y - 170, center_x + 70],
        [center_y - 80, center_x + 130],
        [center_y - 10, center_x + 40],
        [center_y - 50, center_x - 70],
    ],
    dtype=float,
)

polygon_2_yx = np.array(
    [
        [center_y + 260, center_x - 40],
        [center_y + 180, center_x + 260],
        [center_y + 420, center_x + 340],
        [center_y + 520, center_x + 40],
        [center_y + 360, center_x - 180],
    ],
    dtype=float,
)
```

For SpatialData assertions, convert these from napari `(y, x)` to Shapely
`(x, y)` and construct:

```python
polygon_1 = Polygon(shell_xy, holes=[hole_xy])
polygon_2 = Polygon(polygon_2_xy)
```

The edit-existing SpatialData widget test should start from those Shapely
objects directly. The native napari adoption test should start from the same
fixture geometry but encode `polygon_1` into napari's flat polygon-with-hole
vertex path before writing the temporary CSV.

Slice 1C acceptance criteria:

- an unchanged polygon-with-hole round-trips through the annotation widget save
  path
- a native napari Shapes layer loaded from a napari shapes CSV with an encoded
  hole can be adopted by the annotation widget and saved into SpatialData
- saved source metadata and row identity remain stable
- the tests use synthetic in-memory or temporary-file data, not external local
  datasets
- backed SpatialData save behavior is covered if practical

### Slice 1D - Non-Anchor Hole Vertex Edit Round Trip

Goal: prove that users can edit ordinary vertices of an existing hole-bearing
annotation and save/reload the result, as long as they do not move any repeated
anchor/separator vertex.

This slice turns the observed working behavior into a supported contract. It
does not fix anchor editing yet.

Definitions:

- ordinary exterior vertex: a shell vertex that is not one of the repeated
  exterior anchor/separator copies
- ordinary hole vertex: an interior-ring vertex that is not the repeated hole
  start/end anchor
- anchor/separator vertex: any duplicated coordinate required by the napari
  hole encoding, such as exterior anchor copies or hole start/end copies

Suggested work:

1. Add widget-level or converter-level tests that load a polygon with one hole.
2. Move one ordinary shell vertex and one ordinary hole vertex.
3. Save through the annotation widget.
4. Assert the saved geometry is valid, still has the expected number of
   interiors, and reflects the edited vertex coordinates.
5. Reload the saved shape through the viewer/annotation path and assert the
   encoded napari row still decodes to the same Shapely polygon.

Slice 1D acceptance criteria:

- editing a non-anchor exterior vertex of a hole-bearing polygon round-trips
  through save/reload
- editing a non-anchor interior-ring vertex round-trips through save/reload
- anchor/separator edits remain explicitly out of scope for this slice
- malformed anchor/separator paths still fail clearly rather than being guessed

### Slice 1E - Anchor/Separator Vertex Editing Stability

Status: must-fix follow-up before editable holes are considered complete.

Goal: moving an anchor/separator vertex in napari direct-edit mode preserves the
hole path grammar and does not create transient or persisted bridge/collapse
artifacts.

Problem statement:

Napari's renderer supports hole paths by removing duplicate bridge edges, but
napari's direct-edit interaction moves only the selected raw vertex. For an
encoded polygon with a hole, the duplicated vertices are semantic groups. Moving
one member of such a group must move the other required copies:

- moving any exterior anchor/separator copy should update all exterior anchor
  copies for that row
- moving a hole start/end anchor should update both copies of that hole anchor
- moving an ordinary non-anchor vertex should update only that vertex

Concrete one-hole example:

```text
index:  0 1 2 3 4   5 6 7 8 9   10
value:  A B C D A   E F G H E   A
```

This flat napari row represents:

- `A B C D A`: the closed exterior shell
- `E F G H E`: the closed hole ring
- final `A`: the exterior-anchor separator after the hole

The semantic synchronization groups are:

```text
exterior anchor group: [0, 4, 10]
hole anchor group:     [5, 9]
ordinary vertices:     [1], [2], [3], [6], [7], [8]
```

If the user drags exterior-anchor index `0`, the broken napari behavior is:

```text
A' B C D A   E F G H E   A
```

The required repaired state is:

```text
A' B C D A'   E F G H E   A'
```

If the user drags hole-anchor index `5`, the broken napari behavior is:

```text
A B C D A   E' F G H E   A
```

The required repaired state is:

```text
A B C D A   E' F G H E'   A
```

If the user drags ordinary hole vertex `G` at index `7`, no synchronization is
needed:

```text
A B C D A   E F G' H E   A
```

For multiple holes, the same rule applies independently to each anchor group:

```text
index:  0 1 2 3 4   5 6 7   8   9 10 11   12
value:  A B C D A   E F E   A   I J  I    A

exterior anchor group: [0, 4, 8, 12]
first hole group:      [5, 7]
second hole group:     [9, 11]
```

This is the core Slice 1E contract: napari may report that one raw vertex moved,
but napari-harpy must treat some raw vertices as aliases of the same logical
topology vertex and keep those aliases identical.

Candidate implementation approach:

1. Add a pure helper that parses a valid encoded napari polygon row into
   topology metadata, not only a Shapely polygon. The metadata should include
   the index groups that must stay synchronized, for example:
   `[[0, shell_end, final_separator], [hole_start, hole_end], ...]`.
2. When an annotation layer is opened or adopted, attach a small edit guard for
   polygon rows that decode as hole-bearing paths.
3. Before a drag/edit starts, record the current synchronized index groups for
   the affected row. This is important because after one anchor copy moves, the
   row may no longer be parseable.
4. During direct vertex movement, or immediately after napari edits the row,
   detect whether the moved vertex belongs to one of those groups. If it does,
   write the same coordinate into all indices in the group and refresh the row.
5. Use a re-entrancy guard so the repair edit does not recursively trigger
   itself.
6. If the row can no longer be repaired deterministically, fail clearly and
   leave save behavior strict.

Implementation options to investigate:

- Live sync: subclass or wrap the annotation `Shapes` layer mouse interaction
  so anchor groups are synchronized during the drag. This best avoids the
  visible collapse shown in napari.
- Event repair: listen to `layer.events.data`, capture pre-edit topology, and
  repair on `CHANGED`. This may preserve save correctness, but napari emits the
  final direct-drag data event only on mouse release, so it may still show a
  transient collapse while dragging.
- Save-time repair only: not sufficient for this bug, because the rendering
  has already collapsed during user interaction and the broken row may be
  ambiguous.

Slice 1E acceptance criteria:

- dragging the exterior anchor of a hole-bearing polygon keeps all exterior
  anchor/separator copies synchronized
- dragging a hole anchor keeps the hole start/end copies synchronized
- napari rendering does not show persistent bridge/collapse artifacts after the
  edit
- saving after anchor edits preserves a valid Shapely `Polygon` with interiors
- malformed edits that cannot be synchronized fail clearly without guessing
- ordinary non-anchor vertex editing from Slice 1D continues to work

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
- `_shapely_polygon_to_napari_polygon_vertices(Polygon(..., holes=[...]))` followed by the save
  converter preserves `len(geometry.interiors)`, area, bounds, and validity
- existing `MultiPolygon` shapes elements remain rejected by the annotation
  widget
- create-new save can write a polygon with an interior ring
- edit-existing save can round-trip an unchanged polygon with an interior ring
- a hole-inside-hole / island-in-hole path is rejected as unsupported
- an ambiguous edited separator path fails clearly rather than guessing a hole
  layout
- non-anchor exterior and interior-ring vertex edits round-trip through the
  annotation widget save path
- anchor/separator vertex edits either stay synchronized or fail before
  corrupting the saved geometry; Slice 1E must make the synchronized path work
- invalid encoded paths fail without mutating `layer.features` or `sdata`

Slice 1 acceptance criteria:

- An existing SpatialData shapes row with a Shapely `Polygon` interior opens in
  the annotation widget.
- Existing SpatialData shapes rows with Shapely `MultiPolygon` geometry remain
  unsupported for annotation and fail clearly.
- Saving that layer without edits preserves the number of interiors, bounds,
  area, validity, index, and non-geometry columns.
- Editing ordinary non-anchor shell and hole vertices preserves the hole and
  round-trips through save/reload.
- Editing anchor/separator vertices is handled by explicit synchronization so
  napari does not leave a collapsed bridge-edge rendering.
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

The second risk is napari direct editing of duplicated anchor/separator
vertices. Rendering can represent holes by removing duplicated bridge edges,
but direct editing exposes and moves raw vertices. Anchor groups must be
synchronized explicitly; otherwise the path grammar can collapse before save.

The third risk is boolean output shape. A contained subtractor produces the
desired `Polygon` with an interior ring. A subtractor crossing the exterior can
produce a smaller polygon, a split `MultiPolygon`, or an empty result. For the
MVP, accepting only a valid single `Polygon` keeps the existing widget model
intact.

The fourth risk is row identity. The implementation should preserve the shell
row's source identity and delete temporary cutter rows. Grouping holes only at
save time would be harder to reason about because cutter rows would still look
like positive annotations while editing.

The fifth risk is linked tables. Current annotation editing already warns that
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
- Editing non-anchor vertices of the shell or hole preserves the hole after
  save and reload.
- Editing anchor/separator vertices is supported through explicit
  synchronization, without persistent bridge/collapse artifacts.
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
