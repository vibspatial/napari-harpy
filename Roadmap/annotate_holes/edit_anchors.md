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
