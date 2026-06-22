# Anchor/Separator Vertex Editing Stability

Source: detailed roadmap for Slice 1E from
[`annotation_widget_holes.md`](./annotation_widget_holes.md).

Status: specification in progress.

Goal: moving an anchor/separator vertex in napari direct-edit mode preserves the
hole path grammar and does not create transient or persisted bridge/collapse
artifacts.

## Problem Statement

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

This is the core contract: napari may report that one raw vertex moved, but
napari-harpy must treat some raw vertices as aliases of the same logical
topology vertex and keep those aliases identical.

## Current Napari Edit Path

In the local napari `0.7.0` environment, direct vertex editing uses
`layer._drag_modes[Mode.DIRECT]`, which points to napari's `select(...)` mouse
drag callback. During mouse movement, that callback repeatedly calls
`_move_active_element_under_cursor(...)`. That function only synchronizes the
first and last vertex of a simple closed polygon. It does not know that
hole-encoded paths contain more duplicated anchor/separator vertices.

Napari emits the public `layer.events.data(..., action=CHANGED, ...)` event only
after the drag has finished. Therefore an event-only repair can protect save
correctness after mouse release, but it is too late to prevent the visible
bridge/collapse artifact during dragging. Save-time repair is even later and
should not be used to guess the intended topology.

## Chosen Implementation Approach

1. Add a pure topology helper that parses a valid encoded napari polygon row
   into metadata, not only into a Shapely polygon. The metadata should include
   the index groups that must stay synchronized, for example:
   `[[0, shell_end, final_separator], [hole_start, hole_end], ...]`.
   Simple polygons should produce no synchronization groups. Malformed or
   ambiguous hole paths should fail clearly, using the same strict grammar as
   `napari_polygon_vertices_to_shapely_polygon(...)`.
2. Add a small annotation-layer edit guard. The guard should be attached by
   `ShapesAnnotation` whenever an annotation layer is opened, created, or
   adopted, and disconnected when annotation state is cleared. This matters
   because existing SpatialData shapes load as `_HarpyShapes`, while
   native/imported napari layers adopted by the widget remain plain
   `napari.layers.Shapes`.
3. Use live direct-mode synchronization as the primary fix. The guard should
   give the layer an instance-local copy of napari's `_drag_modes` mapping and
   replace only `Mode.DIRECT` with a wrapper around napari's normal direct-edit
   callback. This avoids global monkeypatching and works for both `_HarpyShapes`
   and native `Shapes` layers.
4. Before a direct drag starts, record the current synchronized index groups for
   the affected row. This is important because after one anchor copy moves, the
   row may no longer be parseable.
5. During the direct-drag loop, detect whether the moved raw vertex belongs to
   one of the recorded groups. If it does, write the moved coordinate into all
   indices in that group and refresh the row immediately:

   - moving exterior index `0`, `4`, or `10` updates all exterior anchor copies
   - moving hole index `5` or `9` updates both hole-anchor copies
   - moving ordinary vertices such as `1`, `2`, `3`, `6`, `7`, or `8` leaves all
     other vertices unchanged

6. Use a re-entrancy guard so the synchronization edit does not recursively
   trigger itself.
7. Keep event-time repair only as a defensive fallback if the live wrapper
   misses an edit. Keep the save path strict: if a row is ambiguous or cannot be
   synchronized deterministically, saving should fail clearly rather than
   guessing.

## Implementation Slices

### Slice 1 - Pure Topology Helper

Status: implemented.

Goal: parse one napari polygon vertex row into topology metadata that describes
which raw vertex indices are aliases of the same logical anchor.

Implemented in:

- `NapariPolygonTopology`
- `napari_polygon_vertices_to_topology(...)`
- shared private parser used by both `napari_polygon_vertices_to_topology(...)`
  and `napari_polygon_vertices_to_shapely_polygon(...)`

Suggested scope:

- Add a pure helper in the geometry layer, likely next to
  `napari_polygon_vertices_to_shapely_polygon(...)`.
- The helper should accept finite napari vertices in `(y, x)` order.
- It should use the same strict adapter-encoded path grammar as the existing
  hole decoder.
- It should return metadata, not mutate vertices and not import napari UI
  classes.

Suggested metadata:

```python
@dataclass(frozen=True)
class NapariPolygonTopology:
    shell_anchor_group: tuple[int, ...]
    hole_anchor_groups: tuple[tuple[int, ...], ...]

    @property
    def synchronized_anchor_groups(self) -> tuple[tuple[int, ...], ...]:
        if not self.shell_anchor_group:
            return self.hole_anchor_groups
        return (self.shell_anchor_group, *self.hole_anchor_groups)
```

Implemented behavior:

- simple polygon without holes returns `shell_anchor_group=()` and
  `hole_anchor_groups=()`
- one-hole example `A B C D A E F G H E A` returns
  `shell_anchor_group=(0, 4, 10)` and `hole_anchor_groups=((5, 9),)`
- multi-hole example `A B C D A E F E A I J I A` returns
  `shell_anchor_group=(0, 4, 8, 12)` and
  `hole_anchor_groups=((5, 7), (9, 11))`
- `synchronized_anchor_groups` returns the shell group followed by each hole
  group, so synchronization code can iterate over all alias groups without
  losing the shell-versus-hole distinction in the stored metadata
- malformed or ambiguous adapter-style paths raise `ValueError`

Tests for this slice:

- simple polygon row
- canonical one-hole row from the existing hole round-trip fixture
- multiple direct holes
- missing final exterior separator
- unclosed hole ring
- too-short shell or hole ring

### Slice 2 - Synchronization Core Without Napari UI

Status: implemented.

Goal: make the anchor synchronization behavior testable without napari mouse
events.

Implemented in:

- `sync_napari_polygon_anchor_vertex(...)`
- validation for moved vertex index, moved coordinate, topology group bounds,
  and overlapping topology groups

Suggested scope:

- Add a small pure function that applies one moved vertex to its synchronized
  group.
- The function should work on raw napari vertices in `(y, x)` order, not on the
  private `_ParsedNapariPolygonVertices` shell/hole bundle. The private parsed
  object stores Shapely-oriented rings and can go stale after an edit; the sync
  core needs to update the actual napari row that will be written back to
  `layer.data[row]`.
- Proposed API:

  ```python
  def sync_napari_polygon_anchor_vertex(
      vertices: ArrayLike,
      topology: NapariPolygonTopology,
      moved_vertex_index: int,
      moved_coordinate: ArrayLike,
  ) -> np.ndarray:
      ...
  ```

- `vertices` is the full napari polygon vertex row in `(y, x)` order.
- `topology` comes from `napari_polygon_vertices_to_topology(...)` and should be
  captured before one anchor copy is moved.
- `moved_vertex_index` is the raw vertex index napari moved.
- `moved_coordinate` is the moved coordinate in napari `(y, x)` order.
- The function should return updated vertices. For ordinary non-anchor vertices,
  returning a copy equal to the input is acceptable.
- It should copy data rather than mutate caller-owned arrays unexpectedly.

Suggested behavior:

- moving exterior index `0`, `4`, or `10` writes the moved coordinate into all
  exterior anchor copies
- moving hole index `5` or `9` writes the moved coordinate into both hole-anchor
  copies
- moving ordinary indices such as `1`, `2`, `3`, `6`, `7`, or `8` leaves all
  other vertices untouched
- out-of-range moved indices or inconsistent topology metadata fail clearly

Tests for this slice:

- exterior anchor synchronization
- hole anchor synchronization
- ordinary non-anchor no-op
- multiple-hole synchronization only updates the affected hole group
- invalid moved index raises a clear error

### Slice 3 - Vertex Insert Topology Update Without Napari UI

Status: implemented.

Goal: support adding ordinary vertices to hole-bearing polygon rows without
losing the synchronized-anchor topology needed by later UI slices.

Implemented in:

- `insert_napari_polygon_vertex(...)`
- insertion-index validation against real shell/hole ring ranges
- topology index shifting followed by fresh topology decoding/validation

Current napari behavior:

Napari's `VERTEX_INSERT` mode inserts one raw vertex into the selected polygon
row with `np.insert(vertices, ind, [coordinates], axis=0)`. It reports the
inserted raw index through `layer.events.data(..., vertex_indices=((ind,),))`.
Because `NapariPolygonTopology` stores raw vertex indices, every topology index
at or after the insertion point must shift by one.

Suggested scope:

- Add a pure helper for structural insertion, likely in the same geometry
  module as the move-sync helper.
- The helper should work on raw napari vertices in `(y, x)` order and return
  both the updated vertex row and the updated topology.
- The helper should insert an ordinary non-anchor vertex; inserted vertices
  should not become members of `shell_anchor_group` or `hole_anchor_groups`.
- The helper should update topology indices by shifting all anchor indices at
  or after the insertion index by `+1`.
- The helper should validate that insertion happened on a real shell or hole
  ring edge, not on an artificial bridge/separator edge.
- After insertion, the updated row should still decode through
  `napari_polygon_vertices_to_topology(...)`; if the inserted coordinate makes
  the encoding ambiguous, fail clearly.
- The helper should not import napari UI classes.

Possible API:

```python
def insert_napari_polygon_vertex(
    vertices: ArrayLike,
    topology: NapariPolygonTopology,
    insert_index: int,
    inserted_coordinate: ArrayLike,
) -> tuple[np.ndarray, NapariPolygonTopology]:
    ...
```

Allowed insertions:

- inserting into an ordinary shell edge
- inserting into an ordinary hole-ring edge
- inserting before a ring's closing anchor copy, because that still adds an
  ordinary vertex to the ring before it closes

Rejected insertions:

- inserting before the shell-start anchor at index `0`
- inserting between the shell-closing anchor and the first hole anchor
- inserting between a hole-closing anchor and the following exterior separator
- inserting between an exterior separator and the next hole anchor
- inserting on the final exterior-separator-to-shell-start closure edge
- inserting a coordinate that makes the row no longer satisfy the strict
  adapter-encoded grammar

Insertion-edge validation:

Napari reports the raw `insert_index`; the inserted vertex is placed before
that index. For a closed polygon row, this means the edge being split is usually
`(insert_index - 1, insert_index)`, with `insert_index == 0` representing the
closing edge from the last raw vertex back to the first.

Use topology to derive the real ring index ranges:

- shell ring: from `shell_anchor_group[0]` through `shell_anchor_group[1]`
- each hole ring: from each `(hole_start, hole_end)` pair in
  `hole_anchor_groups`

An insertion is allowed only when it splits a real ring edge:

```python
shell_start < insert_index <= shell_end
or any(hole_start < insert_index <= hole_end for hole_start, hole_end in hole_anchor_groups)
```

This permits inserting before a ring's closing anchor copy because that splits
the final real edge of that ring. It rejects the artificial bridge/separator
edges between the shell and holes, between holes and exterior separators, and
between the final exterior separator and the shell start.

One-hole example:

```text
index:  0 1 2 3 4   5 6 7 8 9   10
value:  A B C D A   E F G H E   A

shell anchor group: (0, 4, 10)
hole anchor group:  (5, 9)
```

Allowed insert indices:

```text
1, 2, 3, 4   shell ring edges
6, 7, 8, 9   hole ring edges
```

Rejected insert indices:

```text
0    final exterior separator -> shell start
5    shell closing anchor -> hole start
10   hole closing anchor -> exterior separator
11   out of range
```

If a vertex `X` is inserted into the shell before index `3`, the updated row is:

```text
index:  0 1 2 3 4 5   6 7 8 9 10   11
value:  A B C X D A   E F G H E    A

shell anchor group: (0, 5, 11)
hole anchor group:  (6, 10)
```

If a vertex `Y` is inserted into the hole before index `8`, the updated row is:

```text
index:  0 1 2 3 4   5 6 7 8 9 10   11
value:  A B C D A   E F G Y H E    A

shell anchor group: (0, 4, 11)
hole anchor group:  (5, 10)
```

Tests for this slice:

- inserting an ordinary shell vertex updates all later shell/hole anchor indices
- inserting an ordinary hole vertex updates that hole's closing anchor and all
  later separators
- insertion in one hole does not change earlier hole groups except for global
  index shifting where appropriate
- bridge/separator-edge insertions are rejected
- inserted coordinates that make the row ambiguous are rejected
- returned vertices and topology decode successfully through the existing
  topology helper

### Slice 4A - Ordinary Non-Anchor Vertex Delete Topology Update Without Napari UI

Status: implemented.

Goal: support deleting ordinary, non-anchor vertices from hole-bearing polygon
rows while preserving the encoded hole topology.

Implemented in:

- `delete_napari_polygon_vertex(...)`
- ordinary non-anchor deletion for shell and hole rings
- topology index shifting followed by fresh topology decoding/validation
- tests for ordinary shell deletion, ordinary hole deletion, multi-hole index
  shifting, invalid indices, simple polygon rejection, shell-too-short
  rejection, and minimal-hole removal

If ordinary deletion from a minimal triangular hole would make that hole too
short, the helper removes the entire hole ring instead of rejecting. This gives
users a natural way to delete a hole, since holes are not separate napari shape
rows. If ordinary deletion from the shell would make the shell too short, the
helper continues to reject clearly; deleting the shell means deleting the entire
annotation row and belongs to a later layer-row deletion workflow, not this
vertex-row helper.

Current napari behavior:

Napari's `VERTEX_REMOVE` mode removes exactly one raw vertex from the selected
polygon row with `np.delete(vertices, vertex_under_cursor, axis=0)`. It reports
the clicked raw index through `layer.events.data(..., vertex_indices=((index,),))`.
Deleting ordinary vertices can be handled by shifting topology indices, but
deleting anchor/separator vertices is harder because those raw vertices are
structural aliases required by the encoding grammar.

Suggested scope:

- Add a pure helper for ordinary vertex deletion, likely in the geometry module.
- The helper should work on raw napari vertices in `(y, x)` order and return
  both the updated vertex row and updated topology.
- The helper should support deleting ordinary non-anchor vertices from shell
  and hole rings.
- Structural shell anchors, hole anchors, and exterior separators are outside
  the ordinary-deletion path and are handled by Slice 4B.
- The helper should validate that affected shell rings still have enough
  coordinates to form a valid Shapely ring after deletion.
- If an affected hole ring would become too short, the helper should remove
  that entire hole ring instead of returning an invalid polygon.
- The helper should not import napari UI classes.

Possible API:

```python
def delete_napari_polygon_vertex(
    vertices: ArrayLike,
    topology: NapariPolygonTopology,
    deleted_vertex_index: int,
) -> tuple[np.ndarray, NapariPolygonTopology]:
    ...
```

Ordinary vertex deletion:

- deleting an ordinary shell vertex removes only that raw vertex
- deleting an ordinary hole vertex removes only that raw vertex
- all topology indices after the deleted index shift by `-1`
- deletion is rejected if the affected shell ring would become too short
- deletion from a minimal triangular hole removes the entire hole ring
- the updated row must still decode through `napari_polygon_vertices_to_topology(...)`
- structural shell anchors, hole anchors, and exterior separators follow the
  Slice 4B rebuild path

One-hole example:

```text
index:  0 1 2 3 4   5 6 7 8 9   10
value:  A B C D A   E F G H E   A
```

Deleting ordinary shell vertex `C` at index `2` can be handled by removal and
index shifting:

```text
value:  A B D A   E F G H E   A
shell anchor group: (0, 3, 9)
hole anchor group:  (4, 8)
```

Deleting ordinary hole vertex `G` at index `7` can also be handled by removal
and index shifting:

```text
value:  A B C D A   E F H E   A
shell anchor group: (0, 4, 9)
hole anchor group:  (5, 8)
```

Deleting ordinary vertex `F` from a minimal triangular hole should remove the
entire hole:

```text
index:  0 1 2 3 4   5 6 7 8   9
value:  A B C D A   E F G E   A

after deleting F:
value:  A B C D A
topology: no synchronized anchor groups
```

If other holes exist, only the affected minimal hole is removed; the remaining
holes are re-encoded and fresh topology is derived.

Tests for this slice:

- deleting an ordinary shell vertex updates topology and preserves holes
- deleting an ordinary hole vertex updates topology and preserves holes
- deleting an ordinary hole vertex from a minimal triangular hole removes that
  hole
- deleting an ordinary hole vertex from one minimal hole in a multi-hole row
  preserves unaffected holes
- deleting an ordinary shell vertex is rejected when it would make the shell
  too short
- the returned vertices and topology decode successfully through the existing
  topology helper

### Slice 4B - Anchor/Separator Vertex Delete Rebuild Without Napari UI

Status: implemented.

Goal: support deletion of anchor/separator vertices by rebuilding the encoded
row from logical rings, without allowing the hole topology to collapse into an
ambiguous row.

Implemented in:

- structural index routing inside `delete_napari_polygon_vertex(...)`
- `_delete_napari_polygon_shell_anchor(...)`
- `_delete_napari_polygon_hole_anchor(...)`
- `_encode_napari_polygon_vertices_from_rings(...)`
- rebuilt-row validation through existing Shapely/topology helpers
- tests for shell alias deletion, hole alias deletion, minimal-hole removal,
  multi-hole preservation, shell-too-short rejection, and shell-rebuild
  rejection when remaining holes no longer fit

Why this is separate from Slice 4A:

Anchor/separator vertices are structural aliases in the napari vertex row.
Deleting only one raw copy leaves the other copies behind and changes the row
grammar. Therefore this is not just index shifting; it requires rebuilding the
encoded row from logical rings and deriving fresh topology.

Suggested scope:

- Extend the deletion helper from Slice 4A, or add a dedicated rebuild helper
  that is called by the Slice 4A helper when the deleted index is structural.
- Treat deletion of anchor/separator vertices as deletion of one logical ring
  vertex, not as deletion of only one raw duplicate copy.
- Rebuild the encoded row from logical rings after choosing replacement shell
  or hole anchors deterministically.
- Reject the operation clearly if shell-anchor deletion would make the shell
  invalid or too short.
- If hole-anchor deletion would make that hole too short, remove the entire
  hole ring, matching the Slice 4A ordinary-hole deletion policy.
- After rebuilding, validate by decoding to a Shapely `Polygon` with interiors
  and deriving a fresh `NapariPolygonTopology`.
- Do not import napari UI classes.

Chosen policy:

- Shell structural deletion rebuilds the polygon when the shell remains valid.
- Shell structural deletion is rejected when the shell would become too short
  or invalid. Deleting the whole annotation row is a layer-row operation and is
  not handled by this vertex-row helper.
- Hole structural deletion rebuilds the polygon when the hole remains valid.
- Hole structural deletion removes the entire hole when the hole would become
  too short, matching Slice 4A ordinary-hole deletion.
- Removing the last hole returns a simple closed polygon row with empty
  `NapariPolygonTopology` groups.

Implementation shape:

Keep the public API as:

```python
def delete_napari_polygon_vertex(
    vertices: ArrayLike,
    topology: NapariPolygonTopology,
    deleted_vertex_index: int,
) -> tuple[np.ndarray, NapariPolygonTopology]:
    ...
```

Instead of rejecting structural indices immediately, route them:

```python
if deleted_vertex_index in shell_anchor_group:
    return _delete_napari_polygon_shell_anchor(...)
if deleted_vertex_index in a hole_anchor_group:
    return _delete_napari_polygon_hole_anchor(...)
```

Anchor/separator deletion policy:

The implementation should rebuild the encoded row from rings:

- if deleting any shell anchor/separator copy, remove the logical shell anchor
  vertex from the shell ring
- if deleting any hole anchor copy, remove the logical hole anchor vertex from
  that hole ring
- choose the replacement anchor deterministically as the next remaining vertex
  in the affected ring; if the deleted vertex was the last logical vertex before
  closure, wrap to the first remaining vertex
- rewrite every exterior anchor/separator copy to the replacement shell anchor
- rewrite both hole-anchor copies to the replacement hole anchor
- reject if shell-anchor deletion would make the shell invalid or too short
- remove the affected hole if hole-anchor deletion would make that hole too
  short
- after rebuilding, validate by decoding to a Shapely `Polygon` with interiors
  and deriving a fresh `NapariPolygonTopology`

Rebuild details:

- Work from logical unclosed rings in napari `(y, x)` coordinates.
- Do not rely only on `shapely_polygon_to_napari_polygon_vertices(...)` for this
  rebuild, because Slice 4B needs deterministic replacement-anchor selection.
- Add a small private encoder from logical napari rings, for example:

```python
def _encode_napari_polygon_vertices_from_rings(
    shell_yx: np.ndarray,
    holes_yx: tuple[np.ndarray, ...],
) -> np.ndarray:
    ...
```

The encoder should produce:

```text
shell + shell[0]
hole + hole[0]
shell[0]
...
```

Then validate the rebuilt row through the existing decode/topology helpers.

One-hole example:

```text
index:  0 1 2 3 4   5 6 7 8 9   10
value:  A B C D A   E F G H E   A
```

Deleting any shell anchor/separator copy, indices `0`, `4`, or `10`, should
delete logical shell vertex `A` and rebuild with the next shell vertex as the
replacement anchor:

```text
value:  B C D B   E F G H E   B
```

Deleting either hole anchor copy, indices `5` or `9`, should delete logical
hole vertex `E`. It must not produce
`A B C D A F G H E A`. Instead, it should rebuild the hole with a new anchor,
using the next hole vertex as the replacement anchor:

```text
value:  A B C D A   F G H F   A
```

For a minimal triangular hole:

```text
index:  0 1 2 3 4   5 6 7 8   9
value:  A B C D A   E F G E   A
```

Deleting either hole anchor copy, indices `5` or `8`, should remove the entire
hole:

```text
value:  A B C D A
topology: no synchronized anchor groups
```

Tests for this slice:

- deleting each shell anchor/separator copy rebuilds the same valid encoded row
  with the next remaining shell vertex as replacement anchor
- deleting either copy of a hole anchor rebuilds the same valid encoded row with
  the next remaining hole vertex as replacement anchor
- deleting either copy of a minimal triangular hole anchor removes that hole
- deleting a minimal triangular hole anchor in a multi-hole row preserves
  unaffected holes
- deleting any structural alias removes the logical vertex from the affected
  ring, not only the clicked raw duplicate
- deleting a shell structural alias is rejected when the shell would become too
  short or invalid
- deleting a structural alias in a multi-hole row preserves unaffected holes and
  derives fresh topology for all anchor groups
- deleting a shell anchor is rejected if the rebuilt shell no longer contains
  remaining holes
- unrecoverable ambiguous deletion never reaches the save path as a guessed
  geometry

### Slice 5 - Annotation Layer Edit Guard Lifecycle

Status: implemented.

Goal: establish the widget-owned guard lifecycle and direct-mode interception
plumbing without changing annotation edit behavior yet.

This slice is intentionally only infrastructure. It should prove that
`ShapesAnnotation` can attach a guard to the active annotation layer, wrap
napari direct-edit mode for that layer instance, and detach cleanly. Anchor
synchronization itself belongs to Slice 6.

Suggested scope:

- Add a small private guard object owned by `ShapesAnnotation`, likely stored as
  `self._annotation_edit_guard`.
  The guard should own the low-level napari patch/restore details: the guarded
  layer reference, the original direct-mode callback, the previous drag-mode
  mapping state, and later any re-entrancy or pre-drag topology cache. This
  keeps `ShapesAnnotation` responsible for deciding when an annotation session
  starts or ends, while the guard is responsible for how direct-edit
  interception is installed and removed.
- Attach the guard immediately after `self._annotation_layer` is assigned for
  every annotation entry path:
  - create-new layer assignment in `_open_create_new_annotation_layer(...)`
  - edit-existing layer assignment in `_open_existing_annotation_layer(...)`
  - native/imported layer adoption in `_adopt_native_shapes_layer(...)`
- Disconnect the guard in `_clear_annotation_state(...)` before dropping the
  widget's reference to `self._annotation_layer`. This covers discard, clean
  close, layer replacement, manual layer removal, and stale-state cleanup paths
  because they already route through centralized annotation-state cleanup.
- Support both `_HarpyShapes` layers loaded from SpatialData and native
  `napari.layers.Shapes` layers adopted by the widget.
- Give the guarded layer an instance-local copy of napari's `_drag_modes`
  mapping. Do not mutate napari's class-level/default drag-mode mapping.
- Replace only `Mode.DIRECT` with a wrapper that, in this slice, simply delegates
  to napari's original direct-edit callback.
- Preserve the original direct callback and enough previous layer state to
  restore it on disconnect where feasible.
- Make attach/disconnect idempotent:
  - attaching twice to the same layer should not stack wrappers
  - attaching to a new layer should disconnect the old layer first
  - disconnecting when no guard is attached should be harmless
- Do not parse hole topology or call the Slice 1-4 geometry edit helpers yet.
  Slice 5 should be behavior-preserving.

Tests for this slice:

- creating a new annotation layer attaches the guard
- opening an existing annotation layer attaches the guard
- adopting a native `Shapes` layer attaches the guard
- clearing annotation state disconnects the guard before the annotation layer
  reference is dropped
- discarding or closing an annotation session disconnects the guard through
  `_clear_annotation_state(...)`
- toggling the layer into direct mode uses the wrapped direct callback, and the
  wrapper delegates to napari's original callback unchanged
- attaching twice to the same layer does not stack duplicate wrappers
- replacing the annotation layer disconnects the old guard before attaching the
  new one
- disconnecting the guard restores the previous direct callback / drag-mode
  state where feasible

### Slice 6 - Live Direct-Drag Anchor Synchronization

Status: implemented.

Goal: make direct dragging of anchor/separator vertices stable during the drag,
not only after release.

Suggested scope:

- Extend the direct-mode wrapper from Slice 5.
- Napari's direct-edit callback is a generator. The Slice 6 wrapper should
  preserve that generator contract rather than replacing napari's interaction
  model.
- On mouse press, create the original napari direct-mode generator and advance
  it through its first `yield`. This lets napari run its normal press handling,
  including setting `layer._moving_value = (row_index, vertex_index)`.
- After that first `yield`, read `layer._moving_value` to determine the
  candidate napari `layer.data` row and raw vertex index that napari is about
  to move.
- This post-press/pre-move point is the safe cache window: napari has already
  identified the shape and vertex under the cursor, but no vertex coordinate
  has moved yet, so the row should still satisfy the hole-encoding grammar.
- If the candidate is a polygon row with adapter-encoded holes, parse and cache
  the pre-edit topology before any move occurs:
  - the napari `layer.data` row index
  - the moved raw vertex index
  - the pre-edit `NapariPolygonTopology`
  - the synchronized anchor group containing that vertex, if any
- Separators are handled through the same cached topology. In
  `A B C D A E F G H E A`, `shell_anchor_group=(0, 4, 10)` includes the
  exterior start vertex, the exterior closure vertex, and the exterior
  separator after the hole. If napari moves separator index `10`, the wrapper
  should treat it exactly like moving shell-anchor index `0` or `4`: copy the
  moved coordinate to every index in the shell group.
- Hole-anchor groups work the same way for the duplicated hole start/end
  coordinate. In `A B C D A E F G H E A`, `hole_anchor_groups=((5, 9),)` means
  moving either `5` or `9` copies the moved coordinate to both indices.
- If topology parsing fails on mouse press, do not guess. Delegate to napari's
  original direct-edit generator unchanged for the rest of the drag.
- During each mouse-move iteration, advance napari's original generator once
  first so napari performs its normal direct edit.
- After napari's edit, if the moved vertex belongs to a cached anchor group,
  read the moved coordinate from `layer.data[row_index][vertex_index]`, call
  `sync_napari_polygon_anchor_vertex(...)`, write the synchronized row back to
  the layer, and refresh the layer.
- If the moved vertex is ordinary/non-anchor, do nothing after napari's edit and
  leave Slice 1D non-anchor behavior unchanged.
- Use a re-entrancy guard so guard-triggered edits do not recursively trigger
  guard logic.
- Preserve napari's mouse-release behavior, including its existing
  `layer.events.data(...)` emission.

Important constraint:

The topology must be captured immediately after napari's press handling and
before the first move. Once napari moves one duplicate anchor copy, the row may
no longer satisfy the decoder grammar, so parsing after the broken edit is not
reliable.

Concrete example:

```text
before move:  A  B C D A   E F G H E   A
napari move:  A' B C D A   E F G H E   A
```

In the second row, napari has moved only raw vertex `0`. The three exterior
anchor copies are no longer equal, so a parser starting from the first
coordinate `A'` cannot rediscover that indices `0`, `4`, and `10` are the same
logical shell anchor. The wrapper must therefore reuse the cached pre-move
topology and synchronize the row to:

```text
repaired:     A' B C D A'  E F G H E   A'
```

Tests for this slice:

- simulated direct drag of an exterior anchor copy synchronizes all exterior
  copies
- simulated direct drag of an exterior separator copy synchronizes all exterior
  anchor/separator copies
- simulated direct drag of a hole anchor copy synchronizes both hole copies
- simulated direct drag of an ordinary hole vertex changes only that vertex
- repeated direct-mode toggles do not break the wrapper
- malformed rows are not guessed into a repaired topology

### Slice 6B - Vertex-Remove Mode Deletion UI Integration

Status: not implemented.

Goal: wire napari `Mode.VERTEX_REMOVE` clicks for hole-bearing annotation rows
into the existing pure deletion helper without broadening Slice 6's drag-only
scope.

This slice is separate from Slice 6 because deletion is a structural edit and
uses a different napari interaction path. Mouse dragging in `Mode.DIRECT` is a
generator that changes coordinates. Vertex deletion uses `Mode.VERTEX_REMOVE`
and napari's `vertex_remove(layer, event)` callback, which is a click handler
that changes the length of `layer.data[row]` and may remove or rebuild a shape
row.

Suggested scope:

- Extend `_AnnotationLayerEditGuard.attach(...)` to wrap `Mode.VERTEX_REMOVE` in
  addition to `Mode.DIRECT`, while still patching only the guarded annotation
  layer instance.
- Preserve the original `Mode.VERTEX_REMOVE` callback and restore it on guard
  disconnect, using the same instance-local `_drag_modes` restoration contract
  as Slice 5/6.
- In the vertex-remove wrapper, determine the clicked vertex before any
  deletion by calling the same napari lookup used by the original callback:

  ```python
  row_index, deleted_vertex_index = layer.get_value(event.position, world=True)
  ```

- Treat `row_index` as the rendered napari `layer.data` row index, not a source
  GeoDataFrame index.
- If `deleted_vertex_index is None`, delegate to napari's original
  `Mode.VERTEX_REMOVE` callback unchanged.
- If the clicked row is not a polygon row, delegate unchanged. This keeps
  rectangles, ellipses, and unsupported/ordinary napari behavior outside the
  hole-specific scope.
- Parse the pre-delete row with `napari_polygon_vertices_to_topology(...)`.
  This must happen before any vertex is removed because anchor/separator
  deletion can make post-delete topology ambiguous or impossible to infer.
- If the pre-delete row has no encoded holes, delegate unchanged to napari's
  original callback. The pure helper currently requires hole-bearing topology,
  and simple polygon deletion should remain napari-owned for this slice.
- If topology parsing fails, do not guess. Let the original callback handle the
  click unchanged; the save path remains strict for malformed rows.
- For a valid hole-bearing polygon row, call the existing pure helper:

  ```python
  updated_vertices, updated_topology = delete_napari_polygon_vertex(
      old_vertices,
      cached_topology,
      deleted_vertex_index,
  )
  ```

- `updated_topology` does not need to be stored by the UI guard; it is returned
  by the helper as a validation result and future-proofing for callers that need
  the new topology.
- If `delete_napari_polygon_vertex(...)` raises `ValueError`, do not silently
  fall back to napari's default deletion, because that can create an ambiguous
  hole row. Instead, fail clearly for the click, ideally by leaving
  `layer.data[row_index]` unchanged and routing a warning through
  `ShapesAnnotation` status UI.
- Give `_AnnotationLayerEditGuard` a small warning callback hook owned by
  `ShapesAnnotation`. The guard should call that hook with the helper error
  message when hole-aware deletion is rejected, while the widget decides how to
  present the message in its status card.
- For successful helper deletion, write `updated_vertices` back to
  `layer.data[row_index]` using the same low-level edit path as Slice 6
  (`layer._data_view.edit(row_index, updated_vertices)`) and refresh the layer.
- Preserve napari-style data notifications around the edit. The original
  callback emits `ActionType.CHANGING` before deletion and `ActionType.CHANGED`
  after deletion with `data_indices=(row_index,)` and
  `vertex_indices=((deleted_vertex_index,),)`. The wrapper should either reuse
  the original event contract or deliberately emit equivalent events for the
  custom helper path.
- Support the helper behavior already implemented in Slices 4A/4B:
  - ordinary shell or hole vertex deletion
  - shell anchor/separator deletion by rebuilding the shell with a replacement
    anchor
  - hole anchor/separator deletion by rebuilding that hole with a replacement
    anchor
  - deletion from a minimal triangular hole removes the whole hole
  - ambiguous or invalid rebuilt geometry fails clearly
- Keep this slice scoped to annotation layers guarded by `ShapesAnnotation`.
  Do not change global napari Shapes deletion behavior.

Chosen error-reporting policy:

- Route `delete_napari_polygon_vertex(...)` errors through the
  `ShapesAnnotation` status UI via the guard callback hook.
- Leave the row unchanged when deletion is rejected.
- Do not raise from the mouse callback for expected geometry/topology rejection,
  because that would make normal interactive editing brittle.

Tests for this slice:

- vertex-remove click with no vertex under the cursor delegates unchanged
- simple polygon row delegates unchanged to napari's original
  `Mode.VERTEX_REMOVE` callback
- hole-bearing polygon ordinary shell vertex deletion calls the helper and
  writes the updated row
- hole-bearing polygon ordinary hole vertex deletion calls the helper and writes
  the updated row
- exterior anchor/separator deletion rebuilds the shell and keeps holes valid
- hole anchor/separator deletion rebuilds the hole or removes a minimal
  triangular hole
- malformed topology is not guessed
- helper `ValueError` leaves the layer unchanged and shows a warning through
  `ShapesAnnotation` status UI
- the wrapper emits or preserves napari-style `CHANGING`/`CHANGED` data events
  for successful helper deletion
- disconnecting the guard restores the original `Mode.VERTEX_REMOVE` callback

### Slice 7 - Defensive Event-Time Repair

Status: not implemented.

Goal: provide a backup repair path in case an edit reaches `layer.events.data`
without having been synchronized live.

Suggested scope:

- Listen to `layer.events.data` for annotation layers only.
- Reuse cached pre-edit topology where available.
- On `CHANGED`, if exactly one member of a known anchor group changed, propagate
  that coordinate to the rest of the group.
- If the change is ambiguous, do not guess. Leave the row strict so the save
  path fails clearly.

Non-goal:

This slice is not the primary user-experience fix. Event-time repair happens
after mouse release and therefore cannot by itself prevent the visual
bridge/collapse during dragging.

Tests for this slice:

- one missed exterior-anchor edit is repaired after a data event
- one missed hole-anchor edit is repaired after a data event
- ambiguous edits are not guessed
- save remains strict for unrecoverable malformed rows

### Slice 8 - End-To-End Widget Round Trip And Interactive QA

Status: not implemented.

Goal: prove anchor editing works through the annotation widget, save path, and
reload path.

Suggested scope:

- Reuse the canonical polygon-with-hole fixture from Slice 1C/1D.
- Cover edit-existing SpatialData shapes loaded as `_HarpyShapes`.
- Cover native/imported napari `Shapes` layers adopted by the widget.
- Save after anchor edits and assert the stored geometry is a Shapely
  `Polygon` with the expected interior ring.
- Reload the saved shapes element and decode the napari row back to Shapely.
- Manually verify in napari that dragging exterior and hole anchors does not
  show the bridge/collapse artifact during the drag.

Tests for this slice:

- edit-existing layer: exterior anchor edit saves and reloads with holes
- edit-existing layer: hole anchor edit saves and reloads with holes
- native adopted layer: exterior anchor edit saves and reloads with holes
- native adopted layer: hole anchor edit saves and reloads with holes
- ordinary non-anchor edit regression from Slice 1D still passes

## Suggested Tests

- topology-helper tests for simple polygons, one-hole polygons, multi-hole
  polygons, and malformed ambiguous paths
- guard-level tests that simulate moving one exterior anchor copy and assert
  all exterior anchor/separator copies are synchronized
- guard-level tests that simulate moving one hole-anchor copy and assert both
  hole-anchor copies are synchronized
- widget round-trip tests showing that anchor edits save and reload as a valid
  Shapely `Polygon` with interiors
- coverage for both edit-existing `_HarpyShapes` layers and adopted native
  napari `Shapes` layers

## Acceptance Criteria

- dragging the exterior anchor of a hole-bearing polygon keeps all exterior
  anchor/separator copies synchronized
- dragging a hole anchor keeps the hole start/end copies synchronized
- napari rendering does not show bridge/collapse artifacts during or after the
  direct edit
- saving after anchor edits preserves a valid Shapely `Polygon` with interiors
- malformed edits that cannot be synchronized fail clearly without guessing
- ordinary non-anchor vertex editing from Slice 1D continues to work

## Decoder Notes

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

## Decoder Algorithm For Adapter-Encoded Paths

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
