# Invalid Polygon State During Vertex Editing

## Status

Investigated and reproduced. Slice 1's passing native characterizations and
reusable regression fixtures are implemented. Slice 2's prevalidated,
transactional direct-polygon movement guard is also implemented. Deletion and
the shared recovery/unsafe-session work remain in Slices 3 and 4.

## Context

The failure was observed while editing annotation row `annotation_1` in the
SpatialData shapes element `annotation_1_hole_triangulation_regression` from:

`/Users/arne.defauw/VIB/DATA/test_data/sdata_xenium_3_6_26.zarr`

The user was dragging raw vertex index 39, which is one copy of an encoded hole
anchor, very close to a shell-anchor/separator vertex. In the four-hole row
visible during the original interaction, the user identified shell-anchor copy
54. The same shell anchor is repeated at indices 0, 16, 25, 33, 40, and 54.
The minimized three-hole reproducer ends at shell-anchor copy 40, so it uses 40
where the original UI observation referred to the equivalent copy at 54.

Napari raised during face triangulation and the Shapes annotation widget
subsequently considered the polygon invalid. The widget could not recover
through normal interaction, so the annotation session had to be discarded.

The reported traceback entered Harpy at:

```text
src/napari_harpy/widgets/shapes_annotation/_edit_guard.py:502
    yielded = next(direct_drag)
```

and failed in napari's Numba/VisPy Shapes mesh path:

```text
napari.layers.shapes._shapes_utils.triangulate_face_vispy(...)
vispy.geometry.triangulation.Triangulation.triangulate(...)

TypeError: argument of type 'NoneType' is not iterable
```

Napari saved the failed normalized triangulation input to temporary `.npz` and
text files. Those artifacts were still available during this investigation and
made it possible to reconstruct the exact candidate row.

## Executive Summary

Harpy's current edit guard validates a polygon only after napari has applied
the mouse move and rebuilt the polygon mesh. That ordering works only when
napari successfully renders the temporary candidate.

For a hole anchor, napari initially moves only the selected copy of the
duplicated anchor. Before Harpy can synchronize the other copy, napari attempts
face triangulation. In the reproduced case the two hole-anchor copies differed
by only about `0.0012` data units. This created an open, nearly degenerate
constraint path that caused VisPy triangulation to raise.

The exception prevents Harpy from reaching its synchronization, validation,
and rollback code. Napari has already assigned the candidate vertices to the
shape model at that point, so the layer is left with a malformed hole encoding.
Napari's normal mouse-release cleanup is also skipped. This combination
explains both the triangulation traceback and the widget's inability to recover.

The cached pre-drag row can be restored successfully with a low-level edit.
Discard is therefore not intrinsically required; the current interaction path
simply has no exception-safe rollback.

Follow-up investigation found related gaps in vertex deletion and insertion.
The first implementation phase in this roadmap therefore covers the already
partially guarded direct-move and vertex-delete paths together. Vertex
insertion follows in a separate phase because the annotation widget does not
currently guard that mode at all, even though a topology-aware insertion helper
already exists.

## Polygon Hole Encoding

Harpy stores a polygon with holes in one napari polygon row. Interior rings are
appended after the closed shell, with repeated shell anchors acting as
separators. Each ring anchor is also duplicated to close that ring.

For the three-hole regression geometry used to reproduce this failure, the
topology is:

```text
shell_anchor_group = (0, 16, 25, 33, 40)
hole_anchor_groups = ((17, 24), (26, 32), (34, 39))
```

Vertices 34 and 39 are two copies of the same third-hole anchor. They must stay
coordinate-identical for the row to describe a closed hole ring.

Raw index 40 is the terminal shell-anchor/separator copy immediately after the
third hole. It is coordinate-identical to every other member of
`shell_anchor_group`. Thus the relevant tail of the minimized encoded row is:

```text
index 34 = start copy of the third-hole anchor
...
index 39 = end copy of the third-hole anchor
index 40 = repeated shell anchor after the third hole
```

These are zero-based raw array indices. A UI that presents one-based vertex
numbers would label them differently.

Harpy already tracks these groups through `NapariPolygonTopology`. The problem
in the recorded implementation was not missing topology information. It was
that synchronization occurred after napari's native move and retriangulation.

## Exact Reproduction

The saved failed triangulation input was reproduced from
`POLYGON_WITH_HOLES_TRIANGULATION_FIXTURE_1` in
`tests/test_shapes_triangulation_backend.py`, but that polygon constant alone is
only the source geometry. It is not already the failing pre-drag row.

The valid pre-drag row is derived deterministically from its encoded 41-vertex
napari row:

1. encode the three-hole source polygon;
2. assign `P` to every shell-anchor copy in `(0, 16, 25, 33, 40)`;
3. assign `Q` to both copies of the third-hole anchor at `(34, 39)`;
4. validate that complete reconstructed row before using it as the drag
   baseline.

The failing native move then changes only raw index 39 from `Q` to `Q'`. It
does not move raw index 40; index 40 remains at the nearby shell coordinate
`P`.

Immediately before the failing mouse event, the relevant coordinates in napari
`(y, x)` order were:

```text
shell anchor P = (2350.921630859375, 1882.07861328125)
hole anchor  Q = (2350.9228515625,   1882.0780029296875)
```

Both copies of the hole anchor, vertices 34 and 39, were at `Q`. This row:

- decoded successfully through
  `napari_polygon_vertices_to_shapely_polygon(...)`;
- produced a valid Shapely polygon;
- constructed successfully as a napari `Shapes` polygon under Numba/VisPy;
- constructed successfully under Bermuda.

The distance from `P` to `Q` was approximately:

```text
0.0013647875839232116
```

The next native mouse move changed only vertex 39 to:

```text
Q' = (2350.921630859375, 1882.0780029296875)
```

The movement from `Q` to `Q'` was:

```text
0.001220703125
```

At that instant:

```text
vertex 34 = Q
vertex 39 = Q'
```

Normalizing this open candidate with napari's own triangulation normalization
produced an array exactly equal to the saved
`napari_vispy_triang_06vi_z0i.npz` data, including shape, dtype, values, and
order. Constructing a Numba/VisPy `Shapes` polygon from the candidate reproduced
the same `RuntimeError` from `triangulate_face_vispy(...)`.

Harpy rejects the candidate before considering geometric validity because its
encoded hole is no longer closed:

```text
ValueError: Malformed polygon hole encoding: each hole ring must be closed
before the next separator.
```

If both hole-anchor aliases are synchronized to `Q'`, the encoding becomes
closed again, but the resulting geometry is outside the valid containment
boundary and Harpy rejects it with:

```text
ValueError: Polygon holes must be contained by the exterior ring.
```

The correct result for this mouse move is therefore to reject it and restore
the last valid row. The current guard intends to do exactly that, but it never
regains control after native triangulation raises.

## Failure Sequence

The complete failure sequence is:

1. On mouse press, Harpy captures a valid topology and a copy of the current
   vertices as `last_valid_vertices`.
2. On mouse move, Harpy calls `next(direct_drag)`.
3. Napari reads the cursor coordinate and mutates only the selected vertex.
4. Napari calls `layer._data_view.edit(...)` with that temporarily
   unsynchronized row.
5. Napari's polygon data setter assigns the candidate to its internal `_data`.
6. The setter rebuilds the displayed face mesh.
7. VisPy triangulation fails on the nearly coincident open-ring constraints.
8. The exception propagates out of `next(direct_drag)`.
9. Harpy never calls `_validate_polygon_vertex_drag(...)` for this mouse move.
10. Harpy's `finally` block closes the native generator, but does not restore
    `last_valid_vertices`.
11. Closing the generator skips napari's normal release code that resets its
    private drag state.
12. The layer remains backed by the malformed row, so later Harpy conversion
    correctly reports that the polygon is invalid.

The key ordering in `_edit_guard.py` is currently:

```python
yielded = next(direct_drag)  # native edit and triangulation happen here

if event.type == "mouse_move":
    self._validate_polygon_vertex_drag(layer, active_drag)
```

The comment above the validation states that immediate rejection keeps
`layer.data` out of an invalid intermediate state and avoids broken
triangulation. In practice, the native generator has already written and
triangulated that intermediate state before the validation runs.

## Required New Execution Order

The primary fix is to reverse that ordering for guarded polygon vertex moves.
This is a design requirement, not an optional optimization:

> Napari's native polygon vertex-write step must not execute before Harpy has
> constructed, synchronized, and validated the complete candidate row.

The current order is:

```text
napari moves only the selected raw vertex
    -> napari writes the temporarily unsynchronized row
    -> napari triangulates the row
    -> Harpy synchronizes anchor aliases
    -> Harpy validates the synchronized candidate
```

The required order is:

```text
Harpy reads the cursor coordinate
    -> Harpy copies the latest accepted row
    -> Harpy moves the selected vertex and every synchronized alias in memory
    -> Harpy validates the complete candidate
        -> invalid: do not write anything; keep the latest accepted row
        -> valid: attempt to apply the complete row once with
           `_data_view.edit(...)`
            -> napari triangulation succeeds:
                accept the candidate as the new `last_valid_vertices`
            -> napari triangulation fails:
                restore the previous `last_valid_vertices`
                restore napari interaction state
                warn that the mouse move was rejected
                keep the annotation session editable
                -> restoring the previous row also fails:
                    preserve and report both failures
                    mark the annotation session as unsafe
                    block saving and further editing
                    require the user to discard and reload the session
```

For the reproduced drag, vertices 34 and 39 must therefore both receive the
new coordinate in the in-memory candidate before the first
`_data_view.edit(...)` call. Harpy then detects that the synchronized hole is
not contained by the shell and rejects the move without changing `layer.data`.
VisPy never receives the transient open-ring row.

The current wrapper can safely advance `direct_drag` once for native press
setup, because napari yields before its mouse-move mutation. It must not advance
that native generator a second time for a guarded polygon gesture: napari would
then calculate, write, and triangulate the candidate inside that call. The
chosen integration boundary is specified below.

`_data_view.edit(...)` is not internally atomic: napari stores the candidate
before it triangulates, and it does not restore the previous row if
triangulation raises. Harpy must therefore make the overall attempted write
transactional by retaining the previous accepted row until the write and
triangulation both succeed.

Exception-safe rollback remains mandatory, but it is a secondary safety net.
It protects against a rendering backend raising while applying a candidate that
Harpy already considers valid. Catching the current exception and restoring the
row without changing the write-before-validation ordering would improve
recovery, but would not remove the root transient-invalid-state bug.

## Chosen Direct-Drag Integration: Native Press, Harpy Move and Release

The movement implementation uses the existing wrapper as a hybrid generator.
Harpy advances napari's native direct generator exactly once:

```text
call native generator once
    -> napari performs hit testing and press setup
    -> napari populates `_moving_value`
    -> napari initializes selection, highlight, and `_drag_start`
    -> napari yields before moving or triangulating any vertex
```

Harpy then classifies the gesture:

```text
DELEGATE
    non-polygon, non-vertex, or other native interaction
    -> continue forwarding move and release events to the native generator

GUARD
    valid widget-owned polygon vertex
    -> keep the native generator suspended after its press yield
    -> Harpy owns all moves, writes, recovery, events, and release cleanup

REJECT
    widget-owned polygon vertex whose starting row is malformed or invalid
    -> do not delegate to napari's unvalidated move path
    -> consume the gesture without geometry mutation
    -> warn and perform Harpy-owned release cleanup
```

The current `state | None` capture result is too ambiguous for this routing,
because `None` can mean either "delegate this interaction" or "this polygon is
unsafe to edit." The implementation needs an equally small but explicit
three-way result. This distinction is required safety behavior, not a general
gesture framework.

For `GUARD`, the native generator remains suspended for the rest of the
gesture. On each mouse move, Harpy constructs and validates the candidate and
performs the transactional write described above. Harpy must never call
`next(native_drag)` for a guarded move.

On release, Harpy owns only the behavior required for a polygon-vertex gesture:

- emit one completed `ActionType.CHANGED` event if the gesture has an accepted
  state to report;
- emit no completed event if every candidate was rejected;
- reset `_is_moving`, `_drag_start`, `_drag_box`, `_fixed_vertex`, and
  `_moving_value`;
- restore highlight state;
- update the thumbnail only if a mutation was committed;
- call the annotation drag-finished callback.

The suspended native generator should be closed in `finally` before Harpy's
final state normalization. If a future napari generator adds cleanup on close,
Harpy's normalization remains the final authority.

If earlier moves succeeded and a later candidate fails but the last accepted
row is restored successfully, the gesture still has a completed change to
report. The event rule is:

```text
no accepted move
    -> no `CHANGED`

one or more accepted moves, followed only by rejected geometry
    -> `CHANGED` on release for the surviving last accepted row

one or more accepted moves, followed by a rendering failure and successful
restoration
    -> `CHANGED` when the gesture finishes for the restored last accepted row

restoration failure and unsafe session
    -> no successful `CHANGED`
```

### Required Napari Compatibility Contract

The hybrid design depends on one precise private napari contract:

> The first `next(select_generator)` performs press setup and yields before
> mutating polygon coordinates, calling `_data_view.edit(...)`, triangulating,
> or emitting a data-change event.

Slice 1 must verify this against napari's actual `Mode.DIRECT` callback, not a
Harpy fake. If a future napari version violates the contract, the compatibility
test must fail clearly rather than silently reintroducing
write-before-validation behavior.

### Deliberate Minimal Scope

This movement work should not introduce:

- a copied or forked implementation of napari's complete `select(...)`
  generator;
- a generic mouse-gesture framework;
- a broad napari-version abstraction layer;
- speculative automatic layer rebuilding after restoration failure;
- insertion-specific routing before the insertion slices;
- edge-only rendering changes as part of the movement fix.

The justified additions are limited to explicit gesture routing, the existing
per-drag state extended with accepted-move state, one guarded candidate/apply
path, one guarded release/cleanup path, and the shared transaction recovery
needed by the roadmap.

## Existing `last_valid_vertices` and Its New Role

The proposed fix does not introduce a second last-valid-row mechanism. Harpy
already stores the complete most recently valid polygon row in
`_PolygonVertexDragState.last_valid_vertices`. This field was introduced with
the current direct-drag validation guard.

The existing lifecycle is:

1. On mouse press, Harpy copies the current polygon row.
2. Harpy verifies that the copied row converts to a valid Shapely polygon.
3. The copy becomes the initial `last_valid_vertices` for this drag gesture.
4. Napari applies and triangulates each mouse-move candidate first.
5. If control returns to Harpy, Harpy synchronizes aliases and validates the
   resulting candidate.
6. If validation fails, Harpy rewrites `last_valid_vertices` to the layer.
7. If validation succeeds, the candidate replaces `last_valid_vertices` as the
   latest valid rollback point.
8. The per-drag state is discarded when the gesture finishes.

This already supports rollback to the latest valid position rather than only
to the position at mouse press. For example, if one mouse move is valid and the
next is invalid, the current guard restores the first move.

The limitation is not the absence of a valid baseline. The limitation is that
napari writes and triangulates inside `next(direct_drag)` before Harpy can use
that baseline. If triangulation raises, control never reaches the current
validation or restoration code even though `last_valid_vertices` is available.

The new lifecycle should reuse the same field as follows:

1. On mouse press, capture and validate the initial row as today.
2. For each mouse move, construct the candidate from
   `last_valid_vertices`, not from a row that napari has already mutated.
3. Change the selected vertex and all synchronized aliases only in that
   in-memory candidate.
4. Validate the complete candidate before any layer write.
5. If it is invalid, leave both `layer.data` and `last_valid_vertices`
   unchanged.
6. If it is valid, retain `last_valid_vertices` unchanged while attempting the
   single `_data_view.edit(...)` and its triangulation.
7. Only after that attempted write succeeds, replace `last_valid_vertices`
   with a defensive copy of the accepted candidate.
8. If the attempted write raises, use the still-unchanged
   `last_valid_vertices` as the restoration baseline.

The distinction is therefore:

```text
Current:
napari writes and triangulates first
    -> Harpy may later use `last_valid_vertices` to roll back

Required:
Harpy starts from `last_valid_vertices`
    -> constructs, synchronizes, and validates before any write
    -> retains the baseline throughout the attempted write and triangulation
    -> advances the baseline only after complete success
```

The name remains accurate because it contains the complete latest valid row,
not one individual vertex. A future rename to `last_accepted_vertices` could
emphasize that the planned baseline has passed both Harpy validation and napari
triangulation, but such a rename is optional and is not required for the fix.

## Why Napari Retains the Invalid Row

Napari's polygon data setter performs these operations in this order:

```python
self._data = data
self._bounding_box = ...
self._update_displayed_data()
```

Face triangulation occurs inside `_update_displayed_data()`. If it raises,
there is no transactional rollback of `_data`. Consequently, a rendering
failure can also become a data-state failure in the in-memory layer.

The previously generated mesh can remain partially or wholly cached, but the
public `layer.data` already exposes the malformed vertices. The next Harpy save
or validation attempt therefore sees a genuinely malformed napari row even
though the drag started from valid geometry.

## Why Normal Mouse Cleanup Does Not Run

Napari normally completes a direct drag by resetting state such as:

```text
layer._is_moving = False
layer._drag_start = None
layer._drag_box = None
layer._fixed_vertex = None
layer._moving_value = (None, None)
```

This cleanup is located after the native generator's mouse-move loop. When
triangulation raises, Harpy closes the generator from its own `finally` block.
Generator closure does not resume execution after the `yield`, so napari's
cleanup block is bypassed.

Any recovery path must therefore restore both polygon data and interaction
state. Restoring only the vertices may leave selection/highlight or subsequent
drag behavior inconsistent.

## Recovery Evidence

After reproducing the exception, the in-memory row had unequal aliases:

```text
vertex 34 = (2350.9229, 1882.0780)
vertex 39 = (2350.9216, 1882.0780)
aliases equal: False
```

Harpy conversion failed as expected. Reapplying the cached pre-drag vertices
through `layer._data_view.edit(...)` succeeded under Numba/VisPy, after which:

- the aliases were equal again;
- Harpy conversion succeeded;
- the restored Shapely polygon was valid.

This proves that the shape model can recover without rebuilding the entire
annotation session. The missing piece is an exception-safe rollback in the
guard.

## Backend Findings

### Numba / VisPy

The reported traceback is consistent with the user selecting the Numba
backend. Napari's Numba Shapes backend still uses VisPy for constrained face
triangulation. Numba accelerates supporting normalization and mesh operations;
it does not replace `triangulate_face_vispy(...)` in this path.

Observed environment:

```text
napari 0.7.1
vispy 0.16.2
shapely 2.1.2
settings backend: Numba
runtime backend: Numba
shape mesh method: _set_meshes_py
```

The exact open candidate reproduced the reported VisPy exception under this
backend.

### Bermuda

The exact saved open candidate did not raise when constructed under Bermuda.
Bermuda accepted it and produced a compiled mesh. That does not establish
correct rendering or safe behavior: the existing hole-triangulation regression
tests show that Bermuda can overdraw or render incorrect face coverage for
Harpy's repeated-anchor hole encoding.

The user's observation that the problem is easier to trigger with Bermuda may
therefore concern a related rendering or invalid-intermediate behavior, but it
was not the same exception for this exact saved candidate. These should remain
separate claims:

- the attached traceback is a reproduced Numba/VisPy exception;
- Bermuda remains unsafe for these rows for independently demonstrated mesh
  correctness reasons.

Changing backend is not a sufficient fix for the edit-ordering problem. A
temporary malformed row should not be sent to any triangulator if Harpy can
construct and validate the intended synchronized candidate first.

## Current SpatialData State

At investigation time,
`annotation_1_hole_triangulation_regression` contained two rows. The persisted
`annotation_1` row was a valid Shapely polygon with:

```text
4 holes
55 encoded napari vertices
shell_anchor_group = (0, 16, 25, 33, 40, 54)
hole_anchor_groups = ((17, 24), (26, 32), (34, 39), (41, 53))
```

The current persisted geometry is therefore not evidence of permanent Zarr
corruption. The failed `.npz` artifact represents the transient in-memory row
that napari attempted to triangulate. The overwrite warning in the original log
does not by itself establish that this malformed candidate was persisted.

The persisted four-hole topology and the minimized three-hole reproduction do
not identify different geometric anchors. In the four-hole row, indices 40 and
54 are both copies of the same shell anchor, with 54 occurring after the fourth
hole. Removing that unrelated fourth-hole suffix leaves index 40 as the final
shell separator. The minimized fixture therefore preserves the failing
third-hole-anchor relationship while avoiding dependence on the mutable Zarr
annotation.

## Related Vertex Mutation Findings

The annotation widget exposes three relevant polygon vertex mutations:

- direct movement through `Mode.DIRECT`;
- deletion through `Mode.VERTEX_REMOVE`;
- insertion through `Mode.VERTEX_INSERT`.

They currently have different levels of protection.

### Hole-Bearing Vertex Deletion Is Prevalidated but Not Transactional

The edit guard already wraps `Mode.VERTEX_REMOVE` for hole-bearing polygons.
It copies the source row and topology, calls
`delete_napari_polygon_vertex(...)`, and applies the result only if that helper
returns successfully.

The helper already handles:

- ordinary shell and hole vertices;
- synchronized shell-anchor aliases;
- synchronized hole-anchor aliases;
- removal of a minimal hole;
- topology index shifts after row shortening;
- Shapely validation of the complete candidate;
- comparison of parsed topology with expected topology.

Invalid deletion candidates therefore fail before the live row is mutated.
This part of the desired design already exists.

The remaining risk is the commit. Deletion shortens the row, and the current
widget rebuilds `layer.data` to refresh napari's private vertex cache. Napari's
bulk data setter replaces its existing `_data_view` before it constructs and
triangulates all replacement shapes. If triangulation raises during that
rebuild, the old data view has already been discarded and Harpy has no
transactional restoration path.

### Simple-Polygon Vertex Deletion Is Unguarded

The current delete router deliberately delegates simple polygons to napari
when no encoded hole topology is present. Napari removes the selected raw
vertex and writes the shortened row without Shapely validation.

A controlled reproduction started from a valid concave simple polygon,
deleted one vertex, and observed:

```text
napari error: none
resulting row: invalid according to Harpy/Shapely
```

The first implementation phase must therefore route all polygon deletions
through Harpy, not only hole-bearing polygons. Simple deletion can construct a
candidate with `np.delete(...)`, but it must validate that complete candidate
before committing it. Napari's existing behavior for removing an entire shape
when too few vertices remain should be preserved deliberately and tested as a
separate valid operation rather than treated as an invalid polygon candidate.

### Vertex Insertion Is Currently Unguarded

The edit guard does not wrap `Mode.VERTEX_INSERT`. Napari builds its list of
candidate edges from every consecutive pair in the raw encoded row. For a
hole-bearing Harpy polygon, that list includes artificial shell-to-hole and
hole-to-shell bridge edges used only by the flat hole encoding.

A controlled reproduction inserted a vertex at the midpoint of one such
bridge in a valid one-hole polygon. Napari accepted and rendered the edit
without raising, but Harpy could no longer decode the resulting row:

```text
before: 11 encoded vertices and a valid polygon
after:  12 encoded vertices
napari error: none
Harpy error: each hole ring must be closed before the next separator
```

#### Why Visible Boundary Insertion Currently Appears to Work

Insertion on a genuine shell or hole boundary usually succeeds despite the
missing widget guard. This does not happen because Harpy updates a persistent
topology table during insertion. No such persistent table exists on the layer.

Napari inserts the coordinate with `np.insert(...)`. Every raw vertex at or
after the insertion index therefore moves one array position forward
automatically. If the selected edge is a real ring edge and the cursor
coordinate keeps the geometry valid, the repeated shell and hole anchor
coordinates remain correctly paired; only their integer positions change.

Harpy's `NapariPolygonTopology` is gesture-local. When a later guarded direct
drag starts, `_capture_polygon_vertex_drag_state(...)` reads the current row and
calls `napari_polygon_vertices_to_topology(...)` again. The parser finds the
repeated anchor coordinates in the updated array and reconstructs the new
synchronization groups from scratch. Later anchor movement therefore uses the
shifted groups even though insertion itself performed no Harpy topology update.

A one-hole reproduction demonstrated this behavior:

```text
Before insertion:
shell anchors = (0, 4, 10)
hole anchors  = (5, 9)

After insertion on a real shell edge:
shell anchors = (0, 5, 11)
hole anchors  = (6, 10)

After insertion on a real hole edge:
shell anchors = (0, 4, 11)
hole anchors  = (5, 10)
```

Both inserted rows were valid, geometrically equal to the source polygon, and
the next direct-drag capture found exactly the shifted groups shown above.
Moving a hole anchor after either insertion synchronized the new pair
successfully.

The current behavior is therefore:

```text
napari inserts into the raw array
    -> later raw positions shift implicitly
    -> no topology is validated during insertion
    -> a future guarded operation reparses and rediscovers topology
```

This explains why ordinary visible-edge insertion can appear reliable, but it
does not make insertion safe. A bridge insertion, ambiguous anchor insertion,
off-edge cursor coordinate, invalid candidate, or triangulation failure can
still corrupt the live row before any future reparse occurs.

The planned guarded behavior is deliberately stronger:

```text
construct the insertion candidate in memory
    -> calculate the expected shifted topology
    -> reparse the candidate topology immediately
    -> require parsed topology to equal expected topology
    -> validate the complete Shapely polygon
    -> commit only after all checks succeed
```

The important change is not merely that indices shift. They already shift
today. The change is that Harpy verifies the shift, topology grammar, and
geometry inside the insertion transaction instead of relying on a later
operation to rediscover whatever napari committed.

Harpy already provides `insert_napari_polygon_vertex(...)`. For hole-bearing
rows it rejects bridge/separator insertion indices, shifts topology indices,
validates the resulting Shapely polygon, and verifies the parsed topology.
However, the annotation widget does not currently call this helper. The helper
also intentionally rejects simple-polygon topology, so the widget still needs
a separate candidate-construction and validation path for simple polygons.

Insertion increases row length and therefore has the same private vertex-cache
and transactional rebuild concerns as deletion. It is specified as a follow-up
phase after movement and deletion are complete.

## Polygon Mutation Safety Invariant

The shared target invariant for all widget-owned polygon mutations is:

> Starting from a valid polygon, Harpy constructs and validates the complete
> candidate before mutating the live layer. Applying an accepted candidate is
> transactional. The mutation either commits a valid, successfully rendered
> polygon state or leaves the previous accepted state unchanged.

This contract has three cumulative parts. They address different failure modes
and are not alternatives:

1. **Encoding integrity is mandatory.** Harpy must update every coordinate
   alias in the candidate before any live layer write or triangulation. No
   `_data_view.edit(...)` call may receive an open ring or an unsynchronized
   shell or hole anchor. This requirement applies even when the complete
   synchronized candidate will subsequently fail geometry validation.
2. **Geometry acceptance during editing is the chosen product policy.** A
   gesture that starts from a valid polygon may commit only another valid
   polygon. Harpy validates the complete synchronized candidate during the edit
   and rejects invalid candidates without writing them to the live layer.
   Save-time validation remains a final defense; it is not the first or only
   geometry-acceptance point.
3. **Rendering transaction safety is mandatory.** Passing Harpy/Shapely
   validation does not guarantee that a rendering backend will triangulate the
   candidate successfully. Harpy retains the previous accepted row or layer
   baseline throughout the attempted write. If editing or rendering raises, it
   restores that baseline and the interaction state instead of accepting the
   candidate.

The required order is therefore:

```text
synchronize the complete in-memory candidate
    -> validate its encoded topology and geometry
        -> invalid: reject without a live write
        -> valid: attempt the live write transactionally
            -> rendering succeeds: accept the candidate
            -> rendering fails: restore the previous accepted state
```

In particular, synchronization alone is necessary but not sufficient, and
geometric validation is not treated as proof that triangulation cannot fail.

For a failure to restore the previous accepted state, the existing fail-loud
policy applies: preserve both failures, mark the session unsafe, block saving
and further editing, and require discard/reload.

Slices 1–4 apply this invariant to direct movement and deletion. The subsequent
insertion slices extend the same invariant to `Mode.VERTEX_INSERT`.

## Existing Test Coverage and Gap

The existing direct-drag and triangulation-backend focused run completed with:

```text
13 passed
```

A second focused run covering the standalone insertion/deletion helpers and
the widget's current vertex-remove route completed with:

```text
36 passed
```

Current direct-move tests cover:

- synchronizing shell anchors;
- synchronizing hole anchors;
- rolling back invalid ordinary hole vertices;
- rolling back an invalid hole anchor;
- rolling back to the latest valid simple-polygon state;
- warning when a drag starts from an already-invalid row.

Those rollback tests use invalid candidates that napari happens to triangulate
successfully. Control returns to Harpy, so the post-edit validator can reject
and restore them.

Current deletion tests cover topology-aware hole-bearing deletion, minimal-hole
removal, anchor rebuilding, cache rebuilding, style preservation, and helper
rejection without mutation. They intentionally delegate simple-polygon
deletion to napari and do not cover a rendering failure during the live commit.

Current insertion tests cover the standalone hole-aware geometry helper. There
is no annotation-widget insertion routing or widget-level insertion test.

Missing movement and deletion coverage for Slices 1–4 includes:

- the real-napari first-`next(...)` press-before-mutation compatibility
  contract;
- the exact near-coincident duplicated-hole-anchor regression;
- a native triangulation exception before post-edit movement validation;
- simple-polygon deletion that would create invalid geometry;
- transactional failure during the row-length-changing deletion rebuild;
- restoration of row data, topology, features, styles, selection, mode, and
  private vertex-cache consistency after a failed deletion commit;
- cleanup of napari's private movement state after an aborted generator;
- continued editing after successful recovery;
- fail-loud behavior when baseline restoration itself fails;
- ensuring a rejected move or deletion emits no successful changed/save state.

Missing insertion coverage is specified separately in the follow-up insertion
slices below.

## Recommended Repair Direction

The repair should have both a preventative path and an exception-safety path.
The preventative path defines the required normal execution order and is the
primary correctness fix. Exception recovery supplements it; exception recovery
alone is not the completed fix.

### 1. Prevalidate and Apply Polygon Moves Transactionally

For guarded polygon vertex moves, Harpy must derive the candidate coordinate
from the mouse event before napari performs any vertex write or triangulation.

For a synchronized anchor group:

1. copy `last_valid_vertices` or the current accepted row;
2. write the new coordinate to every alias in the group;
3. convert and validate the synchronized candidate through Harpy;
4. reject invalid geometry without applying it;
5. attempt to apply the valid candidate with one `_data_view.edit(...)` call,
   retaining the previous accepted row until editing and triangulation finish;
6. accept it as the new `last_valid_vertices` only after that call succeeds;
7. if the call raises, restore the previous row and interaction state instead
   of accepting the candidate.

This prevents the triangulator from ever seeing an open hole ring.

The same pre-validation principle is desirable for ordinary polygon vertices.
Even without duplicated anchors, a self-intersecting or nearly degenerate
candidate may cause a rendering backend to raise before a post-edit validator
can roll it back.

Implement this through the chosen hybrid generator boundary: advance napari's
native direct generator once for press setup, then keep it suspended and let
Harpy own move and release for `GUARD` and `REJECT`. A post-edit forwarding
wrapper is insufficient because advancing the native generator a second time
would compute, write, and triangulate the candidate before Harpy can validate
it. `DELEGATE` interactions continue through the native generator unchanged.

### 2. Prevalidate All Polygon Vertex Deletions

The annotation edit guard must own deletion for every polygon row, rather than
only for rows with encoded holes.

For a hole-bearing polygon:

1. copy the current row and parse its topology;
2. construct the candidate with `delete_napari_polygon_vertex(...)`;
3. rely on the helper to rebuild anchors or remove a minimal hole as needed;
4. reject helper or Shapely validation errors without mutating the layer;
5. retain the original row and relevant layer state through the commit.

For a simple polygon:

1. copy the current row;
2. construct the shortened candidate in memory;
3. validate the complete candidate through
   `napari_polygon_vertices_to_shapely_polygon(...)`;
4. reject invalid geometry without mutating the layer;
5. preserve napari's deliberate whole-shape removal behavior when too few
   vertices remain, with explicit tests for that separate operation.

Deletion changes row length. Harpy currently rebuilds `layer.data` to avoid a
stale private vertex cache, so its transactional baseline must cover more than
the polygon coordinates. Before committing, retain everything required to
restore a usable layer:

- all shape rows and shape types;
- features and row identity;
- edge and face colors, edge widths, z-indices, and opacity;
- current draw style;
- selected rows and active mode;
- the affected row's topology and expected vertex-cache boundaries.

Candidate validation must finish before emitting `ActionType.CHANGING` or
replacing live layer data. Emit `ActionType.CHANGED` and the delete-finished
callback only after the rebuild and triangulation succeed. A rejected or failed
deletion must not be reported as a completed edit.

### 3. Add Shared Exception-Safe Transaction Handling

The attempted write that applies an accepted movement or deletion candidate
must be wrapped as a Harpy-managed transaction. Rendering errors should be
treated as recoverable edit failures when a valid cached baseline exists. Any
native calls retained for non-polygon paths need equivalent exception safety.

On failure, the guard should:

1. restore `last_valid_vertices` for movement, or the complete captured layer
   baseline for a row-length-changing deletion;
2. refresh the row or rebuilt layer and mesh;
3. reset napari's interaction state consistently;
4. restore features, styles, selection, mode, and highlight state as needed;
5. verify polygon validity, anchor topology, and private vertex-cache
   boundaries after restoration;
6. show a user-facing rejected-edit warning;
7. leave the annotation layer editable;
8. avoid reporting the candidate as a completed data change.

This fallback remains necessary because rendering backends can fail on input
that is geometrically valid according to Shapely. It must not be used as a
substitute for prevalidation: invalid polygon candidates should normally be
rejected without calling `_data_view.edit(...)` at all.

The rollback itself can also theoretically raise. Harpy should not attempt a
speculative automatic row or layer rebuild in that situation: rebuilding may
invoke the same failing triangulation again and risks disturbing features,
styles, selection, and other layer state.

If restoration fails, Harpy can no longer guarantee that `layer.data` is safe.
It should fail loudly by preserving and reporting both the original candidate
write failure and the restoration failure, marking the annotation session as
unsafe, blocking saving and further editing, and requiring the user to discard
and reload the session. It must not emit a successful completed-edit event or
allow the possibly malformed candidate to be saved.

### 4. Consider Edge-Only Rendering for Editable Polygons

The broader proposal in `Roadmap/annotation_fixes/slow_annotation.md` is to use
hole-aware edge-only rendering for primary editable polygon layers while
keeping their semantic shape type and data as polygons.

That direction would also remove face triangulation from the per-mouse-move
path and therefore avoid this particular VisPy face-triangulation failure. It
addresses the separate large-polygon performance problem at the same time.

It is, however, a broader architectural change with hit-testing and edge-mesh
requirements. Exception-safe edit rollback is still valuable even if edge-only
rendering is adopted. Row-length-changing deletion rebuilds must also construct
the same hole-aware edge-only polygon model and preserve its vertex-cache and
hit-testing invariants.

## Suggested Implementation Slices

Slices 1–4 form the first delivery phase and cover direct movement plus vertex
deletion. Insertion is deliberately handled afterward in Slices 5–7.

### Slice 1: Passing Characterizations and Reusable Regression Fixtures

Status: implemented as passing test-only coverage. Production behavior remains
unchanged.

Slice 1 changes tests and test data only. It introduces no production-code
changes, no `xfail` tests, and no assertions that describe behavior which only
exists after Slices 2–4. The focused suite must remain green at the end of the
slice.

Build the smallest useful set of reusable, hardcoded fixtures:

- retain `POLYGON_WITH_HOLES_TRIANGULATION_FIXTURE_1` as the source for the
  movement regression, or extract a clearly named shared source fixture from
  it; do not treat the unmodified polygon constant as the pre-drag row;
- derive and expose one reusable hardcoded 41-vertex pre-drag row by assigning
  `P` to shell-anchor group `(0, 16, 25, 33, 40)` and `Q` to third-hole-anchor
  group `(34, 39)`;
- name the regression roles explicitly: moved raw index 39, synchronized hole
  alias 34, nearby terminal shell separator 40, accepted coordinate `Q`, and
  failing target coordinate `Q'`;
- validate the reconstructed pre-drag row in the fixture setup, and keep it
  independent of the temporary failure files and full Zarr store;
- add one clearly named valid concave simple-polygon row for which native
  deletion of a specified vertex produces a Shapely-invalid shortened row;
- use a controlled vertex hit in the deletion characterization so the result
  does not depend on napari's display-scale-dependent vertex hit radius;
- run the movement characterization explicitly with the Numba backend and
  restore both the settings and runtime backend afterward; reuse existing
  backend-restoration support instead of adding another local variant;
- reuse the existing hole-bearing deletion fixtures where they already express
  the needed topology and state; move a fixture into shared test infrastructure
  only when more than one test module actually needs it;
- add only narrow event, generator-cleanup, or state-capture helpers needed by
  these tests. Do not introduce a generic gesture-testing framework.

Add a passing native press-contract test against napari's actual
`Mode.DIRECT` callback:

- construct its real `select(...)` generator for a polygon-vertex press and
  advance it exactly once;
- assert that this first advancement populates the expected `_moving_value`,
  initializes `_drag_start`, establishes selection/highlight state, and yields;
- assert that `layer.data` remains byte-for-byte unchanged, `_is_moving` remains
  false, `_data_view.edit(...)` is not called, no triangulation occurs, and no
  data-change event is emitted before that first yield;
- close the generator and dispose of the test layer without leaking native
  interaction state.

Add passing characterizations of the two broken native behaviors which justify
the later guards:

- exercise the original napari direct callback with the hardcoded
  near-coincident anchor regression, use a controlled press hit of `(row 0,
  raw vertex 39)`, move only raw vertex 39 to `Q'`, and assert the currently
  observed native Numba/VisPy triangulation exception;
- assert that the failed native move leaves index 34 at `Q`, index 39 at `Q'`,
  index 40 at `P`, and the live row malformed according to Harpy's hole grammar;
- assert that closing the failed native generator does not perform normal
  release cleanup: `_moving_value` still identifies `(0, 39)` and `_is_moving`
  remains true;
- exercise native simple-polygon deletion with the hardcoded concave fixture,
  assert that napari reports no error, and assert that Harpy/Shapely rejects
  the shortened row which napari accepted;
- call the original native callbacks directly in these characterizations so
  the tests continue to document napari's behavior after Harpy's wrappers are
  fixed.

Do not duplicate existing tests merely to collect them under this slice. Keep
the current successful hole-bearing deletion and helper-rejection tests where
they are, and add a new test only for a contract or regression that is not
already covered. Widget-level fixed-behavior assertions belong to Slices 2–4:

- synchronized prevalidation, rejection without a live write, interaction
  cleanup, and a subsequent successful drag belong to Slice 2;
- rejection of an invalid simple-polygon deletion and precise whole-shape or
  minimal-hole semantics belong to Slice 3;
- forced rendering failures, complete restoration, continued editing after
  recovery, and restoration-failure handling belong to Slice 4.

### Slice 2: Prevalidated Transactional Polygon Moves

Status: implemented.

Implement the chosen native-press/Harpy-move hybrid without copying napari's
complete `select(...)` generator. Slice 2 owns guarded polygon movement from
the native press yield through normal release. Native behavior remains
authoritative for interactions which are explicitly delegated.

#### Press Boundary and Three-Way Routing

Call the real native `Mode.DIRECT` generator and advance it exactly once. Slice
1 establishes that this performs hit testing and press setup, populates
`_moving_value`, initializes selection, highlight, and `_drag_start`, and then
yields before mutation or triangulation.

Classify the state immediately after that yield. The routing result must be an
explicit three-way value rather than the current ambiguous `state | None`:

- `DELEGATE`: `_moving_value` describes no raw vertex, no shape, a valid
  non-polygon row, or another clearly native non-polygon/non-vertex
  interaction. Resume the suspended native generator for every later move and
  release exactly as before.
- `GUARD`: `_moving_value` contains in-bounds integer row and raw-vertex
  indices, the row is a polygon, its vertices can be copied as a floating
  array, its encoded topology can be parsed, and the complete starting row can
  be decoded as valid Shapely geometry. Capture the topology and the copied row
  as the initial `last_valid_vertices`.
- `REJECT`: the press claims a raw polygon vertex, but Harpy cannot establish a
  safe starting state because an index is invalid, the row cannot be copied,
  its topology is malformed, or its geometry is already invalid. Warn once,
  consume the remaining gesture without mutation, and perform Harpy-owned
  release cleanup.

Only a clearly non-polygon or non-vertex interaction may use `DELEGATE`.
Failure to capture a polygon vertex safely must not silently fall through to
napari's unvalidated move path. `GUARD` and `REJECT` must never advance the
native generator a second time; keep it suspended until it is closed.

Do not add a generic gesture router. A small enum plus a movement-state object
is sufficient. Extend the existing `_PolygonVertexDragState`; do not introduce
a second accepted-row cache alongside `last_valid_vertices`.

#### Candidate Construction and Encoding Integrity

Keep the polygon-coordinate operation in the standalone
`move_napari_polygon_vertex(...)` geometry helper. The edit guard derives the
event coordinate and supplies the cached valid row, topology, and raw vertex
index; the helper owns ordinary movement, alias synchronization, and complete
candidate validation.

For every `mouse_move` in `GUARD`:

1. Derive the data-space coordinate exactly once with
   `layer.world_to_data(event.position)`, matching napari's direct-move path,
   and reject an incompatible coordinate rather than guessing dimensions.
2. Start from a copy of `last_valid_vertices`, never from a partially written
   live row.
3. Determine the complete raw-index alias set before writing the candidate:
   - for a shell or hole anchor in a hole-bearing encoding, use the cached
     topology's complete synchronized anchor group;
   - for an explicitly closed simple polygon, moving raw index `0` or the last
     raw index must update the pair `(0, last)`, matching napari's native
     closed-endpoint behavior even though simple topology has no hole-anchor
     groups;
   - for an ordinary vertex, update only the selected raw index.
4. Write the new coordinate to every member of that alias set in memory.
5. Require the candidate to retain the starting row's implicit or explicit
   closure form. A coordinate move can otherwise collide the first and last
   coordinates and silently reinterpret an implicit row as an explicit ring.
6. Reparse the candidate and require its encoded topology to equal the cached
   topology. This rejects coordinate collisions which would accidentally
   reinterpret an ordinary vertex as an anchor or separator.
7. Decode the complete candidate through
   `napari_polygon_vertices_to_shapely_polygon(...)`; reject malformed,
   empty, zero-area, self-intersecting, or otherwise Shapely-invalid geometry.

The live row, mesh, `last_valid_vertices`, and accepted-move flag must remain
unchanged when candidate construction or validation fails. Do not call
`_data_view.edit(...)`, `refresh()`, or a triangulation path for a rejected
candidate. Show `_INVALID_POLYGON_DRAG_WARNING` at most once per gesture and
continue consuming later move events so a later valid cursor position can
still be accepted.

A candidate which is byte-for-byte equal to `last_valid_vertices` is a no-op:
do not write it, mark it committed, update the thumbnail, or emit a completed
change solely because a mouse-move event occurred.

#### Accepted Move Transaction

For a valid candidate different from `last_valid_vertices`:

1. Retain `last_valid_vertices` as the restoration baseline.
2. Set the private moving coordinate needed for a direct gesture, but do not
   mark the layer as having an accepted move yet.
3. Apply the complete synchronized candidate with one
   `_data_view.edit(row_index, candidate)` call and refresh the layer. The
   triangulator must therefore see only the already-synchronized, validated
   row.
4. Only after edit, triangulation, and refresh succeed, copy the candidate into
   `last_valid_vertices`, mark that this gesture has committed a move, and set
   `_is_moving` consistently with napari's successful direct-drag behavior.

The transaction boundary includes the edit and refresh. If either raises,
immediately restore the retained `last_valid_vertices` and refresh before
continuing or finishing the gesture. A successful restoration rejects only
that candidate: it does not erase an earlier accepted move, advance
`last_valid_vertices`, or prevent a later valid move in the same gesture.

Slice 2 may implement this as a small movement-specific transaction; it must
not introduce a speculative general transaction framework. If restoration
itself raises, retain both the candidate-application and restoration errors,
terminate the gesture, and emit no successful completion for the failed
candidate. Slice 4 owns the shared movement/deletion recovery abstraction,
complete unsafe-session blocking, and exhaustive forced-renderer and
restoration-failure integration coverage.

#### Guarded Release and Event Contract

For `GUARD` and `REJECT`, consume each `mouse_move` with one wrapper yield, then
complete `mouse_release` without another yield. Perform release locally; do not
resume the suspended native generator to obtain native release behavior.

On normal release:

- emit no `ActionType.CHANGING`; native direct vertex movement does not use a
  `CHANGING` event;
- emit exactly one `ActionType.CHANGED` only if at least one move was committed
  and its accepted row still survives;
- mirror napari's `CHANGED` payload: `value=layer.data`, selected-row
  `data_indices`, and all raw vertex indices for each reported selected row;
- emit no `CHANGED` when every candidate was invalid, the gesture contained
  only no-ops, the starting row was rejected, or no accepted state survives;
- keep an earlier accepted row and emit one `CHANGED` when a later geometric
  candidate is rejected or a later write fails and restoration succeeds;
- update the thumbnail exactly once, and only when a committed mutation
  survives release;
- preserve the selection established by native press setup, reset
  `_is_moving`, `_drag_start`, `_drag_box`, `_fixed_vertex`, and
  `_moving_value`, and refresh the highlight;
- call the annotation drag-finished callback exactly once after cleanup for a
  Harpy-owned gesture, including a rejected gesture, so transient widget
  status can be normalized.

Close the suspended native generator in `finally` before Harpy performs its
final state normalization. Cleanup must also run when candidate validation or
application raises. `DELEGATE` continues to use the native generator's own
move, event, thumbnail, and release behavior unchanged.

#### Slice 2 Test Matrix

Adapt existing movement tests where they already cover the contract instead of
duplicating them. Use the actual native direct callback for the press boundary
and the completed wrapper path; use narrow spies or failure sentinels only to
prove that a forbidden edit or triangulation call did not occur.

Required passing coverage is:

- one representative `DELEGATE` interaction continues through native move and
  release behavior;
- malformed or already-invalid polygon-vertex rows take `REJECT`, never resume
  the native move path, remain unchanged, warn once, emit no data event, and
  receive complete Harpy-owned cleanup;
- the Slice 1 near-coincident regression moves raw vertex 39 toward `Q'`,
  constructs a candidate with indices 34 and 39 synchronized before
  validation, rejects that synchronized invalid candidate before
  `_data_view.edit(...)` or triangulation, preserves the original live row,
  emits no `CHANGED`, and closes with normalized interaction state;
- a valid later drag on the same layer succeeds, proving that rejection did
  not poison interaction state;
- a valid ordinary-vertex move performs one candidate write and emits one
  correct `CHANGED` on release;
- a valid hole-anchor or shell-anchor move presents a fully synchronized row to
  the single edit call and advances `last_valid_vertices` only after success;
- moving an endpoint of an explicitly closed simple polygon keeps raw indices
  `(0, last)` synchronized;
- moving an ordinary vertex onto a coordinate that changes the encoded anchor
  topology is rejected before a live write;
- an accepted move followed by an invalid move leaves the accepted row live
  and emits one `CHANGED` for that surviving row;
- an identical/no-op candidate performs no write, thumbnail update, or
  completed event;
- a narrow movement-transaction test makes candidate application fail after a
  possible partial write and verifies restoration to `last_valid_vertices`, no
  advancement of accepted state for the failed candidate, and no successful
  event unless an earlier accepted move survives.

This slice removes the original transient-open-ring window for direct polygon
movement. It applies to ordinary and duplicated anchor vertices in simple and
hole-bearing polygons. It deliberately excludes deletion, insertion,
edge-only rendering, a copied native generator, and a generic gesture
framework.

### Slice 3: Guard Every Polygon Vertex Deletion

Status: implemented.

Extend the existing `Mode.VERTEX_REMOVE` wrapper from hole-bearing polygons to
all polygon rows owned by the annotation widget. Slice 3 owns hit
classification, candidate construction, validation, successful row rebuilds,
and whole-shape removal. It completes normal-path deletion prevalidation;
forced rendering-failure recovery remains in Slice 4.

#### Hit Test and Three-Way Routing

Call `layer.get_value(event.position, world=True)` once and classify the result
before invoking any mutation path. Deletion needs the same explicit ownership
boundary as movement, even though it is a one-shot callback rather than a drag
generator:

- `DELEGATE`: no raw vertex was hit, or the hit resolves to a valid
  non-polygon row whose native behavior Harpy does not own. Call napari's
  original `vertex_remove(...)` callback unchanged.
- `GUARD`: the hit contains in-bounds integer row and raw-vertex indices, the
  row is a polygon, its vertices can be copied as a finite floating array, its
  encoded topology can be parsed, and the complete starting row decodes as
  valid Shapely geometry. Capture the copied row and topology before
  constructing a candidate.
- `REJECT`: a raw-vertex hit cannot be safely resolved, or it resolves to a
  polygon row with an invalid index, malformed encoding, non-numeric data, or
  already-invalid geometry. Warn without mutation and do not call napari's raw
  deletion callback.

Only a clearly non-polygon or no-vertex interaction may delegate. Failure to
capture a polygon deletion safely must not silently expose the row to native
unvalidated deletion. A small deletion-specific route result is sufficient;
do not introduce a generic gesture router or copy napari's complete
`vertex_remove(...)` implementation.

`GUARD` and `REJECT` are Harpy-owned from classification through return. A
rejected deletion emits no data event, performs no refresh or thumbnail update,
and does not invoke the delete-finished callback.

#### Hole-Bearing Candidate Construction

Keep all polygon-coordinate deletion semantics in the standalone
`delete_napari_polygon_vertex(...)` geometry helper. The helper branches
between hole-bearing and simple encodings and returns a validated
`NapariPolygonVertexDeletion`. The edit guard consumes that outcome but owns
only layer events, row replacement or removal, state preservation, and
refresh. It must not duplicate either geometry branch.

For a valid row whose cached topology contains hole-anchor groups:

1. Pass the copied starting row, cached topology, and clicked raw index to
   `delete_napari_polygon_vertex(...)`.
2. Let that helper handle ordinary shell and hole vertices, synchronized shell
   and hole anchors, separator-index shifts, and removal of an entire minimal
   hole.
3. Inside the helper, treat the returned vertices and topology as an in-memory
   candidate only. Require the candidate's parsed topology to equal its
   returned topology and decode the complete row through
   `napari_polygon_vertices_to_shapely_polygon(...)` before returning it to the
   edit guard for any live mutation.
4. Reject any helper, topology, or Shapely error without emitting
   `ActionType.CHANGING`.

Deleting any raw vertex from a minimal three-vertex hole removes that complete
hole and its encoding aliases; it does not remove the polygon row. A shell
deletion which cannot retain a valid hole-bearing polygon is rejected rather
than converted implicitly into whole-shape removal.

#### Simple-Polygon Candidate Construction

The simple-polygon branch also lives in
`delete_napari_polygon_vertex(...)`. Simple rows may be implicitly closed:

```text
A, B, C, D
```

or explicitly closed, as produced by Harpy's Shapely encoder:

```text
A, B, C, D, A
```

For an explicitly closed row, raw indices `0` and `last` are aliases of one
semantic shell vertex. Compute the semantic shell-vertex count as
`len(vertices) - 1` for an explicitly closed row and `len(vertices)` otherwise.

If deleting one semantic vertex would leave at least three shell vertices:

- delete an ordinary vertex from an implicitly closed row with one in-memory
  `np.delete(...)`;
- delete an ordinary non-endpoint vertex from an explicitly closed row while
  retaining its existing closing alias;
- when raw index `0` or `last` is clicked on an explicitly closed row, delete
  the shared semantic vertex rather than only one duplicate coordinate, then
  reclose the shortened ring on its new first vertex.

The endpoint rule prevents deletion of one closure copy from becoming a
geometric no-op. For example:

```text
A, B, C, D, A
    delete A
B, C, D, B
```

After construction, require the candidate to remain a simple-polygon encoding
with the same implicit or explicit closure form and validate it through
`napari_polygon_vertices_to_shapely_polygon(...)`. Reject coordinate
collisions, malformed rows, zero-area results, self-intersections, and all
other invalid Shapely candidates before a live write.

#### Semantic Triangle and Whole-Shape Removal

If the starting simple polygon has exactly three semantic shell vertices,
the helper returns a deletion outcome with no replacement vertices. The edit
guard interprets that outcome by removing the complete shape row. Apply this
rule to both representations:

```text
A, B, C       # implicit triangle
A, B, C, A    # explicitly closed triangle
```

This preserves the intent of napari's minimum-polygon behavior without copying
its raw-length bug. Napari removes a row only when its raw row length is three;
therefore it mishandles Harpy's four-coordinate closed triangle by either
removing only one closure alias or accepting a row with two unique vertices.
Slice 3 must use semantic vertices and remove the row directly instead.

Whole-shape removal is a deliberate valid operation, not an invalid-candidate
fallback. It applies only after a valid simple triangle has been captured. A
malformed or already-invalid short row takes `REJECT`.

#### Successful Commit and Layer Preservation

Candidate validation must complete before emitting any data event. For an
accepted shortened-row candidate:

1. Emit one native-compatible `ActionType.CHANGING` event with the original
   row index and clicked raw vertex index.
2. Replace the affected row through a Harpy-owned layer rebuild, because a
   low-level row-shortening `_data_view.edit(...)` can leave napari's private
   vertex-boundary cache stale.
3. Block intermediate `data` and `features` events from the bulk data setter;
   the guarded delete owns the public event pair.
4. Preserve every unaffected data row and shape type, the complete features
   table and source identities, per-row edge and face colors, edge widths,
   z-indices, layer opacity, current draw defaults, active mode, and selected
   rows.
5. Leave `_data_view._vertices_index` consistent with the rebuilt row lengths
   so later hit testing cannot report a stale raw index.
6. Refresh the layer, then emit one `ActionType.CHANGED` with the same payload
   and invoke the delete-finished callback exactly once.

For semantic-triangle whole-shape removal, use the same Harpy-owned rebuild
principle but remove the corresponding entries from every row-aligned
structure:

- remove the polygon data row and shape type;
- remove its feature/source-identity row;
- remove its edge color, face color, edge width, and z-index rows;
- preserve opacity and current draw defaults;
- remove the deleted row from selection and decrement selected indices above
  it;
- preserve the active mode and rebuild the private vertex-boundary cache for
  all surviving rows.

The successful whole-shape event payload still uses the original row and raw
vertex indices, matching napari's `vertex_remove(...)` contract. Refresh/update
the thumbnail, then emit `CHANGED` and call the delete-finished callback only
after the complete row removal succeeds.

If a rebuild raises, do not emit `CHANGED` or invoke the completion callback.
Slice 3 does not add a generic recovery framework or claim that a partially
failed rebuild is recoverable. Slice 4 captures and restores the complete
pre-commit layer baseline, tests forced renderer failures, and blocks an unsafe
session if restoration also fails.

#### Slice 3 Test Matrix

Adapt existing deletion tests where they already cover the contract. Required
passing coverage is:

- no-vertex and valid non-polygon hits remain delegated to napari;
- malformed, already-invalid, non-numeric, and invalid-index polygon hits are
  rejected without native fallback, mutation, data events, refresh, or the
  completion callback;
- the Slice 1 concave simple-polygon fixture rejects the characterized
  invalid deletion before any live write or triangulation and leaves the row
  unchanged;
- valid ordinary deletion succeeds for an implicitly closed simple polygon;
- valid ordinary deletion succeeds for an explicitly closed simple polygon
  and preserves its closing alias;
- deleting raw index `0` or `last` from an explicitly closed simple polygon
  removes the one semantic endpoint, recloses the row, and does not become a
  no-op;
- deleting any vertex from representative implicit and explicitly closed
  semantic triangles removes the complete polygon row;
- whole-shape removal from a multi-row layer removes and shifts the correct
  feature, identity, style, and selection rows while preserving mode, opacity,
  current draw defaults, and surviving geometry;
- existing hole-bearing ordinary-shell, ordinary-hole, shell-anchor,
  hole-anchor, and terminal-separator deletions continue to use the topology
  helper;
- deleting from a minimal hole removes the hole but retains the polygon row;
- a successful shortened-row rebuild preserves features and styles, produces
  correct private vertex-cache boundaries, and supports a correct subsequent
  hit test and deletion;
- each accepted deletion emits exactly `CHANGING` then `CHANGED` with the
  native payload and calls the completion callback once; every rejected
  deletion emits neither event and does not call the callback;
- a successful deletion after a rejected deletion clears the stale widget
  warning through the existing completion callback.

Do not add forced edit, rebuild, triangulation, or restoration failures to this
slice; those are Slice 4 recovery tests. Do not add insertion coverage or a
general mutation transaction abstraction.

### Slice 4: Shared Transaction Recovery and Rendering Integration

Generalize and harden the movement-local rollback introduced in Slice 2 into
shared exception-safe transaction handling for accepted movement and deletion
candidates.

- retain the movement row baseline or deletion layer snapshot until editing,
  rebuilding, and triangulation all succeed;
- catch edit, rebuild, and rendering exceptions;
- restore coordinates plus features, styles, selection, mode, highlight, and
  private vertex-cache consistency as appropriate;
- do not emit a completed change for a failed transaction;
- warn once and keep the session editable after successful restoration;
- test continued movement and deletion after recovery;
- force a restoration failure and assert that both errors are retained, the
  session is marked unsafe, saving and editing are blocked, and discard/reload
  is required.

Force failures during both an accepted movement write and a valid
row-length-changing deletion rebuild. Verify complete baseline restoration,
including data, topology, features, styles, selection, mode, highlight, and
private vertex-cache boundaries, followed by a successful later edit. These
are recovery tests, not Slice 1 characterizations of the pre-fix partial state.

Coordinate this transaction path with any future edge-only primary polygon
model:

- avoid duplicate topology implementations;
- ensure row rebuilds construct the same editable polygon model;
- keep save conversion unchanged;
- verify both filled and edge-only configurations where applicable;
- retain recovery around all remaining mesh updates.

At the end of Slice 4, direct movement and vertex deletion satisfy the polygon
mutation safety invariant. Vertex insertion remains explicitly pending.

### Slice 5: Vertex Insertion Regression Coverage and Widget Routing

Add widget-level insertion regressions and install a `Mode.VERTEX_INSERT`
wrapper in the annotation edit guard.

- reproduce insertion on an artificial shell-to-hole bridge and require
  rejection without mutation or completed data events;
- cover successful insertion on real shell and hole ring edges;
- cover a simple-polygon insertion whose cursor coordinate would make the
  candidate invalid;
- verify insertion with multiple holes and topology-index shifts;
- verify that no-eligible-edge and non-polygon interactions continue to follow
  napari behavior;
- ensure attach, repeated attach, layer switching, and disconnect restore the
  original insertion callback exactly as they do for the existing guarded
  modes.

This slice establishes routing and failing integration tests. It must not rely
only on the existing standalone helper tests.

### Slice 6: Prevalidate Simple and Hole-Bearing Insertions

Construct every polygon insertion candidate before mutating the live layer.

For hole-bearing rows:

- parse the current topology;
- map the chosen raw edge to the proposed insertion index;
- call `insert_napari_polygon_vertex(...)`;
- reject artificial bridge or separator edges;
- validate the returned topology against the expected shifted topology.

For simple polygons:

- insert the cursor coordinate into an in-memory row copy;
- validate the complete candidate through Shapely;
- reject invalid geometry without mutation.

For both forms:

- use the actual cursor coordinate rather than assuming it lies on the nearest
  edge;
- finish validation before emitting `ActionType.CHANGING`;
- attempt a live write only for an accepted candidate;
- never delegate a valid widget-owned polygon row to napari's unvalidated raw
  insertion path.

### Slice 7: Transactional Insertion Commit and Recovery

Apply accepted insertion candidates with the same row-length-changing
transaction contract as deletion.

- capture the complete restorable layer state;
- rebuild the affected row while preserving features, identity, styles,
  opacity, selection, mode, and current draw defaults;
- verify topology, private vertex-cache boundaries, and hit testing;
- emit `ActionType.CHANGED` only after rebuilding and triangulation succeed;
- restore the complete baseline after an edit, rebuild, or rendering failure;
- verify continued insertion, movement, and deletion after successful
  recovery;
- apply the same fail-loud unsafe-session policy if restoration fails.

At the end of Slice 7, direct movement, deletion, and insertion all satisfy the
shared polygon mutation safety invariant.

## Acceptance Criteria

### Phase 1: Movement and Deletion, Slices 1–4

The first delivery phase is complete when:

- a real-napari compatibility test proves that the first advancement of the
  native direct generator yields after press setup and before mutation,
  `_data_view.edit(...)`, triangulation, or data-change emission;
- `DELEGATE`, `GUARD`, and `REJECT` take the specified native or Harpy-owned
  gesture paths without a copied full `select(...)` implementation;
- no live write or triangulation receives an unsynchronized shell or hole
  anchor candidate;
- starting from a valid polygon, an invalid synchronized candidate is rejected
  during editing rather than retained until save time;
- reproducing the exact vertex-39 movement does not leave an invalid row;
- duplicated shell and hole anchors remain synchronized after every accepted
  move;
- every guarded polygon direct-vertex candidate is synchronized and validated
  before it can reach face triangulation;
- invalid simple and hole-bearing deletion candidates leave the live layer and
  data-event stream unchanged;
- valid ordinary, anchor, minimal-hole, and whole-shape deletions preserve their
  intended semantics;
- every widget-owned polygon deletion is validated before a live write and is
  never delegated to napari's raw unvalidated deletion path;
- successful row-length-changing deletion rebuilds preserve features, identity,
  styles, opacity, selection, mode, current draw defaults, topology, and private
  vertex-cache consistency;
- the last accepted movement row or deletion layer snapshot is restored when an
  accepted mutation fails during editing, rebuilding, or rendering;
- `_moving_value`, `_is_moving`, drag coordinates, selection, and highlight
  state are usable after rejection;
- after a successful baseline restoration, the user can continue editing
  through both movement and deletion without discarding the session;
- if baseline restoration itself fails, both errors are reported, the unsafe
  session cannot be saved or edited further, and discard/reload is required;
- no rejected or failed movement/deletion candidate is saved or emitted as a
  completed edit.

Vertex insertion remains explicitly out of the Phase 1 guarantee.

### Full Vertex Mutation Guard: Slices 5–7

The insertion follow-up is complete when:

- `Mode.VERTEX_INSERT` is installed and restored with the same lifecycle
  guarantees as the existing edit-guard callbacks;
- insertion on artificial shell/hole bridge or separator edges is rejected
  without mutation;
- valid insertion on real shell and hole edges preserves expected topology;
- simple-polygon insertion candidates are Shapely-validated before writing;
- successful insertion rebuilds preserve the same layer and vertex-cache state
  required for deletion;
- insertion commit failures restore the complete prior layer state;
- continued insertion, movement, and deletion work after successful recovery;
- failed insertion restoration applies the same unsafe-session policy;
- no valid widget-owned polygon insertion uses napari's raw unvalidated path.

### Shared Requirements

Across both phases:

- no invalid candidate is saved or emitted as a completed edit;
- Numba remains the required safe Harpy Shapes backend;
- existing Bermuda mesh-correctness regression coverage remains intact;
- valid polygon and hole editing behavior continues to pass existing tests.

## Out of Scope for This Investigation

No source code, tests, or SpatialData elements were modified as part of the
investigation. This report does not select the final private napari integration
mechanism. That choice may depend on whether the prevalidated transactional
mutation paths are implemented on the existing filled model or together with
the broader edge-only editable-polygon work. Exception recovery alone is
explicitly not the completed fix.
