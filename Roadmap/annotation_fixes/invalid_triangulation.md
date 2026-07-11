# Invalid Polygon Triangulation During Vertex Drag

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

## Existing Test Coverage and Gap

The existing direct-drag and triangulation-backend tests pass. The focused run
completed with:

```text
13 passed
```

Current edit-guard tests cover:

- synchronizing shell anchors;
- synchronizing hole anchors;
- rolling back invalid ordinary hole vertices;
- rolling back an invalid hole anchor;
- rolling back to the latest valid simple-polygon state;
- warning when a drag starts from an already-invalid row.

Those rollback tests use invalid candidates that napari happens to triangulate
successfully. Control returns to Harpy, so the post-edit validator can reject
and restore them.

Missing coverage includes:

- a native triangulation exception before post-edit validation;
- restoration of `last_valid_vertices` after that exception;
- cleanup of napari's private interaction state after an aborted generator;
- a near-coincident duplicated hole anchor matching the saved regression;
- continued editing after recovery;
- ensuring a rejected drag does not emit a successful changed/save state.

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

### 2. Add Exception-Safe Rollback

The single attempted `_data_view.edit(...)` that applies an accepted candidate
must be wrapped as a Harpy-managed transaction. Rendering errors should be
treated as recoverable edit failures when a valid cached baseline exists. Any
native direct-drag calls retained for non-polygon paths need equivalent
exception safety.

On failure, the guard should:

1. restore `last_valid_vertices`;
2. refresh the row and mesh;
3. reset napari's drag state consistently;
4. restore selection/highlight state as needed;
5. show a user-facing rejected-edit warning;
6. leave the annotation layer editable;
7. avoid reporting the invalid candidate as a completed data change.

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

### 3. Consider Edge-Only Rendering for Editable Polygons

The broader proposal in `Roadmap/annotation_fixes/slow_annotation.md` is to use
hole-aware edge-only rendering for primary editable polygon layers while
keeping their semantic shape type and data as polygons.

That direction would also remove face triangulation from the per-mouse-move
path and therefore avoid this particular VisPy face-triangulation failure. It
addresses the separate large-polygon performance problem at the same time.

It is, however, a broader architectural change with hit-testing and edge-mesh
requirements. Exception-safe edit rollback is still valuable even if edge-only
rendering is adopted.

## Suggested Implementation Slices

### Slice 1: Exact Regression Test

Add a focused test derived from the saved artifact:

- start from the reconstructed valid pre-drag row;
- move only vertex 39 to `Q'` through the guarded drag path;
- demonstrate that the native Numba/VisPy operation raises without the fix;
- assert that the fixed guard synchronizes vertices 34 and 39 in memory before
  validation;
- assert that the invalid synchronized candidate is rejected without calling
  `_data_view.edit(...)` and without mutating the layer row;
- assert synchronized aliases and reset interaction state;
- assert that a second drag can start and finish normally.

The fixture should be hardcoded or derived from the existing
`POLYGON_WITH_HOLES_TRIANGULATION_FIXTURE_1`; it should not depend on the
temporary failure file or the full Zarr store.

### Slice 2: Prevalidated Transactional Polygon Moves

Move guarded polygon candidate construction ahead of napari's data write:

- prevent the native polygon mouse-move branch from writing first;
- derive the candidate coordinate from the mouse event;
- synchronize all anchor aliases in memory;
- validate the complete intended candidate;
- write only accepted candidates;
- preserve napari's expected press, release, selection, highlight, thumbnail,
  and data-event contract.

This is the primary correctness fix. It removes the root
transient-invalid-state window rather than relying on catching triangulation
errors after napari has already stored malformed data.

### Slice 3: Exception Recovery Safety Net

Complete the transactional application of an accepted polygon candidate:

- retain the last accepted baseline until the gesture is completely cleaned
  up;
- catch edit or rendering exceptions from the attempted write;
- restore the baseline and napari interaction state;
- warn once for the failed gesture;
- test both recovery and continued editability;
- use a separate test in which a candidate passes Harpy validation but the
  rendering write is forced to raise;
- use a separate restoration-failure test that asserts both errors are
  retained, saving and editing are blocked, and discard/reload is required.

This slice handles unexpected backend failures. It is mandatory defense in
depth, but it does not replace Slice 2 and must not reintroduce native
write-before-validation behavior.

### Slice 4: Integrate With Rendering Strategy

Coordinate the edit fix with any future edge-only primary polygon model:

- avoid duplicate implementations of hole topology handling;
- keep save conversion unchanged;
- verify filled read-only layers and edge-only editable layers consistently
  reject invalid geometry;
- retain exception recovery around any remaining mesh updates.

## Acceptance Criteria

The issue can be considered fixed when:

- reproducing the exact vertex-39 movement does not leave an invalid row;
- the last valid vertices are restored when rendering raises;
- duplicated shell and hole anchors remain synchronized after every accepted
  move;
- every guarded polygon direct-vertex candidate is synchronized and validated
  before it can reach face triangulation;
- `_moving_value`, `_is_moving`, drag coordinates, selection, and highlight
  state are usable after rejection;
- after a successful baseline restoration, the user can continue editing
  without discarding the annotation session;
- if baseline restoration itself fails, both errors are reported, the unsafe
  session cannot be saved or edited further, and discard/reload is required;
- no invalid candidate is saved or emitted as a completed edit;
- Numba remains the required safe Harpy Shapes backend;
- existing Bermuda mesh-correctness regression coverage remains intact;
- valid polygon and hole editing behavior continues to pass existing tests.

## Out of Scope for This Investigation

No source code, tests, or SpatialData elements were modified as part of the
investigation. This report does not select the final private napari integration
mechanism. That choice may depend on whether the prevalidated transactional
move path is implemented on the existing filled model or together with the
broader edge-only editable-polygon work. Exception recovery alone is explicitly
not the completed fix.
