# Invalid Polygon State During Vertex Editing

## Status

Investigated and reproduced. No implementation has been done for this issue
yet.

## Context

The failure was observed while editing annotation row `annotation_1` in the
SpatialData shapes element `annotation_1_hole_triangulation_regression` from:

`/Users/arne.defauw/VIB/DATA/test_data/sdata_xenium_3_6_26.zarr`

The user was dragging vertex 39, which is one copy of an encoded hole anchor,
very close to a shell-anchor/separator vertex. Napari raised during face
triangulation and the Shapes annotation widget subsequently considered the
polygon invalid. The widget could not recover through normal interaction, so
the annotation session had to be discarded.

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

Harpy already tracks these groups through `NapariPolygonTopology` and
`sync_napari_polygon_anchor_vertex(...)`. The problem is not missing topology
information. It is that synchronization currently occurs after napari's native
move and retriangulation.

## Exact Reproduction

The saved failed triangulation input was reproduced exactly from the existing
`annotation_1` regression fixture in
`tests/test_shapes_triangulation_backend.py`.

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

The current `next(direct_drag)` wrapper cannot provide this ordering because
napari calculates, writes, and triangulates the mouse-move candidate entirely
inside that call. The guarded polygon mouse-move path must therefore prevent or
replace that native write step. Harpy may still reproduce the relevant napari
press, release, selection, highlight, thumbnail, and data-event behavior, but
it must own candidate construction and the decision to write.

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

Implementing this requires preventing or replacing napari's native polygon
mouse-move write. A post-edit wrapper is insufficient: the current wrapper
cannot prevalidate a candidate that napari computes, writes, and triangulates
entirely inside `next(direct_drag)`.

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

### Slice 1: Movement and Deletion Regression Coverage

Add focused failing regressions before changing the interaction code.

For direct movement:

- start from the reconstructed valid pre-drag row;
- move only vertex 39 to `Q'` through the guarded drag path;
- demonstrate that the native Numba/VisPy operation raises without the fix;
- require the fixed path to synchronize vertices 34 and 39 in memory before
  validation;
- require the invalid synchronized candidate to be rejected without calling
  `_data_view.edit(...)` or mutating the live row;
- verify reset interaction state and a subsequent successful drag.

The movement fixture should be hardcoded or derived from
`POLYGON_WITH_HOLES_TRIANGULATION_FIXTURE_1`; it must not depend on the
temporary failure file or the full Zarr store.

For deletion:

- add a valid concave simple polygon where native deletion produces an invalid
  result, and require rejection without mutation or data events;
- retain the current successful hole-bearing ordinary, shell-anchor,
  hole-anchor, and minimal-hole deletion cases;
- add an invalid hole-bearing deletion candidate and verify that helper failure
  leaves data, features, styles, selection, mode, and events unchanged;
- add a forced rendering failure during a valid deletion rebuild and capture
  the expected pre-fix partial-state problem;
- define explicit coverage for deliberate whole-shape removal when too few
  polygon vertices remain.

### Slice 2: Prevalidated Transactional Polygon Moves

Move guarded polygon candidate construction ahead of napari's native write:

- prevent the native polygon mouse-move branch from writing first;
- derive the candidate coordinate from the mouse event;
- construct it from `last_valid_vertices`;
- synchronize all anchor aliases in memory;
- validate the complete intended candidate;
- leave the live row unchanged for rejected candidates;
- attempt one write only for accepted candidates;
- advance `last_valid_vertices` only after editing and triangulation succeed;
- preserve napari's press, release, selection, highlight, thumbnail, and
  data-event contract.

This removes the original transient-open-ring window. It applies to ordinary
and anchor vertices in both simple and hole-bearing polygons.

### Slice 3: Guard Every Polygon Vertex Deletion

Extend the existing `Mode.VERTEX_REMOVE` wrapper from hole-bearing polygons to
all polygon rows.

- capture the source row and the complete restorable layer state before
  mutation;
- use `delete_napari_polygon_vertex(...)` for hole-bearing rows;
- construct and Shapely-validate simple-polygon deletion candidates in memory;
- preserve intentional whole-shape removal at the minimum vertex count;
- reject invalid candidates before emitting `ActionType.CHANGING`;
- rebuild row-length-changing accepted candidates while preserving features,
  identity, styles, opacity, selection, mode, and current draw defaults;
- verify the rebuilt private vertex cache and hit-test indices;
- emit `ActionType.CHANGED` and the delete-finished callback only after the
  complete commit succeeds;
- never fall back to napari's raw polygon deletion for a valid widget-owned
  polygon row.

This slice completes normal-path prevalidation for deletion. Rendering-failure
recovery is completed in Slice 4.

### Slice 4: Shared Transaction Recovery and Rendering Integration

Complete exception-safe transaction handling for accepted movement and
deletion candidates.

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
