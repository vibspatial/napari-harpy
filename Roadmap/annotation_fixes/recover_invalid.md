# Recovering Invalid Hole Edits

## Context

This note documents an observed failure while editing the `new_shapes`
element, annotation `__annotation_1`, from:

```text
/Users/arne.defauw/VIB/DATA/test_data/sdata_xenium_3_6_26.zarr
```

The polygon is represented in napari as one polygon row with encoded holes.
Harpy encodes holes by repeating anchor/separator coordinates in `layer.data`.
For `__annotation_1`, the extracted topology is:

```text
shell_anchor_group = (0, 16, 25, 33, 40)
hole_anchor_groups = ((17, 24), (26, 32), (34, 39))
```

Vertex `39` is therefore the closing copy of the third hole anchor. It is
synchronized with vertex `34`.

## Observed Behavior

If the user drags vertex `39` outside the shell, Harpy allows the move. The
rendered polygon becomes strange, but the anchor group is still synchronized:

```text
34 = moved outside
39 = moved outside
```

At this point the row is still structurally closed, but the geometry is invalid
because the hole is no longer contained by the exterior shell.

If the user then grabs the same visible vertex again and moves it, the result
can become worse. Vertex `39` moves again, but vertex `34` can be left behind:

```text
34 = first outside position
39 = second outside position
```

Saving then fails with:

```text
Malformed polygon hole encoding: each hole ring must be closed before the next separator.
```

## Current Failure Chain

The first drag starts from a valid polygon, so the edit guard can parse topology
and synchronize the duplicated anchor group `(34, 39)`.

On mouse move, `_AnnotationLayerEditGuard._sync_anchor_drag(...)` calls:

```python
sync_napari_polygon_anchor_vertex(...)
```

That keeps `34` and `39` together.

After this first drag, the polygon is geometrically invalid because the hole is
outside the shell. On the next drag, `_capture_anchor_drag_state(...)` tries to
rebuild topology from the current row:

```python
topology = napari_polygon_vertices_to_topology(layer.data[row_index])
```

But `napari_polygon_vertices_to_topology(...)` currently does both tasks:

1. parse the raw anchor/separator topology;
2. validate that the parsed polygon-with-holes is geometrically valid.

Specifically, it calls:

```python
if parsed.holes:
    _make_valid_polygon_with_holes(parsed.shell, list(parsed.holes))
```

For the outside-shell hole, this raises:

```text
Polygon holes must be contained by the exterior ring.
```

The edit guard catches this `ValueError` and returns `None`, so no active
anchor-drag state is captured for the second drag. Napari then falls back to
its raw direct-edit behavior and moves only the clicked vertex copy. That leaves
the duplicated hole anchor unsynchronized, so the hole ring no longer closes.

## Important Conclusion

This is not an inserted vertex. The row length remains unchanged.

The apparent extra point is an existing duplicated anchor/separator copy being
left in a different location after the second drag. The bug is that Harpy loses
its ability to synchronize duplicated anchors once the polygon is geometrically
invalid, even though the raw anchor/separator grammar may still be parseable.

## Rendering Behavior

Napari continues to remesh and render the invalid raw path. Rendering is
best-effort and does not validate Harpy's polygon-with-holes contract.

With the Numba triangulation backend, valid polygons with holes render
correctly. However, Numba does not prevent invalid geometry such as a hole being
moved outside the shell. Once the row is invalid, napari can still show a
triangulated shape that looks surprising.

## Save Behavior

The save path converts `layer.data` back to Shapely through:

```python
napari_polygon_vertices_to_shapely_polygon(...)
```

This conversion validates both the encoding grammar and the final polygon
geometry. Therefore:

- after the first outside-shell move, save fails because the hole is outside
  the shell;
- after the second unsynchronized move, save fails earlier because the hole
  ring is no longer closed before the next separator.

## Recommended Fix Direction

Harpy should reject or roll back edits that would leave a hole-bearing polygon
outside Harpy's valid polygon-with-holes contract. We should not depend on
napari rendering to communicate validity, and we should not defer this class of
problem until save time.

The fix should still separate topology parsing from full geometry validation.
The edit guard needs a way to recover anchor groups from a row that still has a
parseable anchor/separator grammar, even if the final polygon geometry is
invalid.

Do this by changing `napari_polygon_vertices_to_topology(...)` itself to parse
only. It should call `_parse_napari_polygon_vertices(...)` and return
`parsed.topology`; it should no longer call `_make_valid_polygon_with_holes(...)`.

Full geometry validation should be explicit at call sites that need a savable
polygon. For public/outside callers, the validation API remains
`napari_polygon_vertices_to_shapely_polygon(...)`, which parses the row and then
calls `_make_valid_polygon(...)` or `_make_valid_polygon_with_holes(...)`.
Internal geometry helpers may call `_make_valid_polygon_with_holes(...)` only
when they already have parsed `(x, y)` shell and hole arrays.

Do not add a `valid` flag to `NapariPolygonTopology`. That object should remain
an index-only description of synchronized anchor groups. Geometry validity
depends on the current vertex coordinates, not just on the anchor indices, so a
cached topology validity flag could become stale as soon as a vertex moves.

This also keeps call sites explicit:

- topology parsing answers: "which encoded row and anchor groups does this vertex
  array describe?";
- full validation answers: "can this current vertex row be saved as a Harpy
  polygon with holes?";
- `_PolygonVertexDragState` owns the previous fully valid vertex row used for
  rollback.

## Implementation Slices

### Slice 1: Separate Topology Parsing From Validation

Status: implemented.

Refactor the geometry helpers before changing drag behavior.

- Change `napari_polygon_vertices_to_topology(...)` so it only parses topology.
  It should still reject malformed Harpy path grammar, such as open hole rings
  or missing shell-anchor separators, because those failures come from parsing.
- Keep `NapariPolygonTopology` index-only. Do not add a validity flag.
- Keep `napari_polygon_vertices_to_shapely_polygon(...)` as the explicit full
  validation/conversion API.
- Add explicit validation calls where current code relied on
  `napari_polygon_vertices_to_topology(...)` for validation side effects.

Known places that need explicit validation after this refactor:

- `insert_napari_polygon_vertex(...)` after building `inserted_vertices`;
- the generic path inside `delete_napari_polygon_vertex(...)` after building
  `deleted_vertices`.

The special deletion helpers already call
`napari_polygon_vertices_to_shapely_polygon(...)` before returning rebuilt
vertices, so they are already close to the desired shape.

Slice 1 should include unit tests proving that:

- `napari_polygon_vertices_to_topology(...)` returns topology for a structurally
  closed row whose hole is outside the shell;
- `napari_polygon_vertices_to_shapely_polygon(...)` still rejects that same row;
- insert/delete helpers still reject edits whose resulting vertices cannot be
  converted to a valid Harpy polygon.

### Slice 2: Polygon Vertex Drag Rollback

After Slice 1, implement the actual edit recovery behavior:

- replace anchor-only drag tracking with `_PolygonVertexDragState`;
- track direct drags for polygon vertices, including simple polygon vertices,
  shell vertices, hole vertices, and duplicated anchors/separators;
- synchronize duplicated anchors/separators only when the moved vertex belongs
  to a synchronized anchor group;
- validate each candidate row after napari's native move;
- accept valid candidates and update the cached last-valid vertices;
- restore the cached last-valid vertices and warn once when the candidate is
  invalid.

Implementation detail against the current code:

- replace `_AnchorDragState` with `_PolygonVertexDragState`;
- rename `_iter_direct_drag_with_anchor_sync(...)` to reflect validation and
  rollback, for example `_iter_direct_drag_with_polygon_validation(...)`;
- replace `_capture_anchor_drag_state(...)` with
  `_capture_polygon_vertex_drag_state(...)`;
- replace `_sync_anchor_drag(...)` with a helper that applies conditional anchor
  synchronization and then validates or rolls back the candidate row.

Suggested state:

```python
@dataclass
class _PolygonVertexDragState:
    row_index: int
    moved_vertex_index: int
    topology: NapariPolygonTopology
    last_valid_vertices: np.ndarray
    warned_invalid_drag: bool = False
```

The state is captured after the first `next(direct_drag)`, exactly where the
current code captures `_AnchorDragState`. At that point napari has populated
`layer._moving_value`, but no vertex has moved yet.

Capture should return a state when:

- `layer._moving_value` is `(row_index, vertex_index)`;
- the row and vertex indices are in range;
- the shape type at the row is `"polygon"`;
- `layer.data[row_index]` can be coerced to numeric vertices;
- `napari_polygon_vertices_to_topology(...)` can parse the row;
- `napari_polygon_vertices_to_shapely_polygon(...)` can validate the row.

Capture should return `None` for malformed rows or rows that are already not
valid Shapely polygons. Slice 2 is primarily prevention: valid rows should not
be allowed to become invalid during a drag. Recovering rows that are already
invalid is out of scope and is not currently planned.

On each `mouse_move`, after napari's native direct-drag callback has advanced,
the helper should:

1. Read `candidate_vertices = np.asarray(layer.data[state.row_index], dtype=float)`.
2. If the row index or moved vertex index is no longer addressable, skip Harpy
   sync/validation/rollback for that event and return without additional
   mutation. This is a defensive branch for an unexpected structural change; it
   should not blindly restore `state.last_valid_vertices` because the stored row
   may no longer be safe to address.
3. Compute `is_synchronized_anchor_vertex` from
   `state.topology.synchronized_anchor_groups`.
4. If true, call `sync_napari_polygon_anchor_vertex(...)` and use the returned
   vertices as the candidate.
5. Validate the candidate with `napari_polygon_vertices_to_shapely_polygon(...)`.
6. If validation succeeds:
   - write synchronized vertices only when they differ from the layer row;
   - call `layer.refresh()` after such a write;
   - update `state.last_valid_vertices` to a copy of the accepted candidate.
7. If validation fails:
   - restore `state.last_valid_vertices` with `layer._data_view.edit(...)`;
   - call `layer.refresh()`;
   - warn once for the drag through `_warn(...)`;
   - do not update `state.last_valid_vertices`.

Because `last_valid_vertices` changes during a drag, `_PolygonVertexDragState`
should not be frozen, or the helper should return an updated state.

The implementation should include an inline comment next to the `mouse_move`
validation branch making the performance/correctness tradeoff explicit. The
comment should say that Shapely validation intentionally runs on every mouse
move for the active polygon row so invalid geometry is rejected immediately,
instead of allowing the layer to remain invalid until mouse release.

## Edit-Time Contract

For polygon rows, Harpy should own the direct-drag validation path instead of
falling back to napari's raw edit path. This should apply to:

- moving vertices of simple polygons;
- moving shell vertices of polygons with holes;
- moving duplicated shell or hole anchors;
- moving ordinary hole vertices;
- deleting vertices from shell or hole rings.

After each candidate edit, Harpy should validate against the same geometry
contract used for save/export:

- each encoded ring remains closed according to Harpy's path grammar;
- holes remain strictly inside the exterior shell;
- holes do not touch the exterior shell;
- holes are not nested;
- holes do not overlap or share edges, except allowed point contact.

If the candidate edit is valid, Harpy accepts it. If it is invalid, Harpy
restores the previous valid vertex row and warns the user once for that drag or
delete action.

This means a user should not be able to drag a hole vertex outside the shell and
leave the layer in that invalid state. It also means deleting a vertex that
would make a hole-bearing polygon invalid should be rejected without mutating
`layer.data`.

## Direct-Drag Plan

The current direct-drag wrapper only captures anchor drags. That is too narrow.
It should become a polygon vertex drag guard using a state object named
`_PolygonVertexDragState`.

Suggested shape:

```python
@dataclass
class _PolygonVertexDragState:
    row_index: int
    moved_vertex_index: int
    topology: NapariPolygonTopology
    last_valid_vertices: np.ndarray
    warned_invalid_drag: bool = False
```

For simple polygons, `topology` is the empty topology and no anchor
synchronization is needed. For polygons with holes, `topology` contains the
shell and hole anchor groups used to synchronize duplicated anchors/separators.
The drag state should track all polygon vertices, but anchor synchronization
should only run for vertices that are actually members of a synchronized anchor
group.

Use a local boolean at the sync point rather than a separate helper:

```python
is_synchronized_anchor_vertex = any(
    state.moved_vertex_index in group
    for group in state.topology.synchronized_anchor_groups
)

if is_synchronized_anchor_vertex:
    candidate_vertices = sync_napari_polygon_anchor_vertex(
        candidate_vertices,
        state.topology,
        state.moved_vertex_index,
        candidate_vertices[state.moved_vertex_index],
    )
```

1. On mouse press, let napari initialize the direct drag and populate
   `layer._moving_value`.
2. If the hit row is a polygon with parseable topology, cache:
   - the row index;
   - the moving vertex index;
   - the topology;
   - the last fully valid vertex row.
3. On each mouse move, let napari apply its native move first.
4. Read the candidate row from `layer.data`.
5. Compute `is_synchronized_anchor_vertex` from
   `state.topology.synchronized_anchor_groups`. If true, copy the moved
   coordinate to the other group members. If false, leave the candidate vertices
   unchanged and continue directly to validation.
6. Validate the synchronized candidate through the full Harpy geometry decoder.
7. If valid, write the synchronized candidate and update the cached last-valid
   row.
8. If invalid, restore the cached last-valid row and warn once for the drag.

This keeps napari's existing direct-manipulation mechanics, but prevents Harpy
from accepting a polygon row that it cannot later save. The same state object
covers regular polygon vertices, shell vertices of polygons with holes, ordinary
hole vertices, and duplicated anchor/separator vertices.

The invalid-drag warning should be intentionally generic and stable enough for
tests, for example:

```text
Polygon edit was rejected because it would create invalid geometry.
```

The warning should be emitted once per drag gesture, not once per mouse-move
event.

## Vertex-Delete Plan

The existing vertex deletion path already uses `delete_napari_polygon_vertex(...)`
for valid hole-bearing polygons. That helper validates the candidate after
deletion, so the basic shape is correct. For the normal case where the row
starts valid, the current code already behaves well: if deleting a vertex would
make the polygon invalid, the helper raises, the widget warns, and `layer.data`
is left unchanged.

The risky part is the routing gate. `_capture_vertex_delete_state(...)` currently
returns `None` when `napari_polygon_vertices_to_topology(...)` raises. Returning
`None` delegates to napari's raw deletion behavior, which is unsafe for a
hole-bearing row. This is mostly a recovery edge case for rows that are already
geometrically invalid but still have a parseable Harpy hole encoding.

After Slice 1, the delete capture should use
`napari_polygon_vertices_to_topology(...)` to decide whether the row is
Harpy-owned. If the row has parseable holes, Harpy should either:

- apply a valid topology-preserving delete; or
- reject the delete and leave `layer.data` unchanged.

It should not fall back to napari raw deletion for a row that Harpy recognizes
as a hole-bearing polygon.

## Why Not Auto-Repair

We should avoid silently repairing invalid polygons with a generic Shapely
operation. Auto-repair can split polygons, remove holes, change ring ownership,
or otherwise reinterpret the annotation. For annotation editing, explicit
reject/rollback behavior is safer and easier to explain.

## Implementation Worries

The main implementation risk is drag-time rollback. Napari's direct mode mutates
the row during its drag generator and keeps private drag state until mouse
release. Harpy's wrapper can restore `layer._data_view.edit(row_index,
last_valid_vertices)` after an invalid move, but this must be tested through a
full press, move, release sequence.

Another edge case is opening a row that is already invalid but still raw
parseable. Parse-only topology lets Harpy identify duplicated anchors, but a row
with no known valid baseline cannot be rolled back to a valid state. This is
out of scope: we do not currently plan to implement recovery from an
already-invalid row. The implementation should prioritize preventing valid rows
from becoming invalid.

## Regression Tests

Add focused tests for both geometry helpers and widget behavior:

- topology parsing succeeds for a closed hole row whose hole is outside the
  shell, while full geometry validation still raises;
- dragging hole anchor `39` outside the shell in the `__annotation_1` fixture is
  rejected or rolled back, and vertices `34` and `39` remain synchronized;
- dragging an ordinary hole vertex outside the shell is rejected or rolled back;
- dragging an ordinary shell vertex of a simple polygon into a self-intersecting
  shape is rejected or rolled back;
- dragging an ordinary vertex through a sequence of valid moves updates the
  cached last-valid row, so a later invalid move rolls back to the latest valid
  accepted position rather than the original press position;
- invalid drag warning emits once per drag, even if multiple mouse-move events
  produce invalid candidates;
- direct-drag release completes without leaving napari's private moving state
  visibly broken; at minimum, tests should advance the wrapped generator through
  press, one or more moves, and release;
- deleting a vertex that would make a hole-bearing polygon invalid leaves
  `layer.data` unchanged and emits a warning;
- deleting from a parseable hole-bearing row does not fall back to napari's
  raw deletion path.
