# Slow shapes annotation

## Problem

Editing large primary shapes annotations in napari can feel unusably slow. On
the Xenium test store
`/Users/arne.defauw/VIB/DATA/test_data/sdata_xenium_3_6_26.zarr`, dragging one
vertex in the `smooth_muscle` annotation can pause for seconds. The same issue
is visible, though less severe, in `tumor`.

This is not caused by SpatialData zarr I/O and is not specifically caused by
holes. A standalone napari `Shapes` layer built from the same geometries
reproduces the delay.

Representative geometry sizes:

- `smooth_muscle`: one polygon, one hole, about 7.4k napari vertices.
- `tumor`: three polygons, no holes, largest row about 2k vertices.

## Cause

Napari rebuilds the polygon face mesh whenever a polygon row is edited. Under
the safe Numba/vispy path this retriangulation dominates the interaction cost.

Measured local timings:

- `smooth_muscle`, filled polygon: build about 3.2 s, vertex edit about 3.1 s.
- `tumor`, filled polygon: vertex edit about 360-390 ms.
- Harpy's per-move Shapely validity guard is much smaller: roughly 10 ms for
  `smooth_muscle` and 2-3 ms for `tumor`.

So deferring validation is not enough. The expensive work is napari's filled
polygon triangulation.

## Rejected Options

### Transparent face color

Setting primary shapes face color alpha to zero does not solve the problem.
Napari still treats the row as a filled `polygon`, so it still calculates the
face triangulation.

### Paths instead of polygons

Treating annotations as paths is not acceptable. The annotation layer must keep
the polygon contract, including holes, and saving must continue to produce
SpatialData shapes elements with polygon geometries.

### Bermuda backend

Bermuda is fast, but manual testing showed it is too buggy for this workflow.
It can fail triangulation or produce incorrect rendering for annotation shapes,
especially in the hole-related cases we care about. We should keep the safe
Numba path rather than falling back to Bermuda for polygons with holes.

### Defer filled triangulation until mouse release

This improves drag responsiveness but still leaves a multi-second pause on
release for large polygons such as `smooth_muscle`. That is still a poor user
experience and does not address slow initial layer construction.

## Recommended Direction

Use edge-only rendering for primary/editable Harpy polygon layers while keeping
the layer semantically polygon-backed.

Napari's polygon model has an internal filled flag. A shape can still report
`shape_type == "polygon"` and keep polygon vertex data while skipping face mesh
generation by using an unfilled polygon shape model. In a prototype:

- `smooth_muscle`, edge-only polygon: build about 8-10 ms after warmup, vertex
  edit about 6 ms.
- `tumor`, edge-only polygon: vertex edit about 2 ms.

The important distinction is that this is not a path fallback. The raw
`layer.data` stays as polygon vertices, `layer.shape_type` remains `"polygon"`,
and Harpy's save conversion still round-trips to Shapely/SpatialData polygons.

## Hole Handling

Harpy encodes polygon holes in one napari polygon row by appending interior
rings plus repeated shell-anchor separator vertices. A naive unfilled polygon
would draw connector/bridge edges from that encoding.

The primary-layer shape model therefore needs a hole-aware edge mesh:

- parse the existing encoded polygon row into shell and interior rings;
- generate edge meshes for each ring separately;
- do not generate face triangles;
- keep the original encoded polygon row unchanged for editing and saving.

The existing anchor synchronization guard remains important so moving a repeated
anchor keeps the encoded hole topology valid during direct vertex edits.

## Interaction Tradeoff

Napari's default inside-shape hit test relies on mesh triangles. Removing face
triangles means clicking in the filled interior will no longer select a shape by
default. Vertex and edge editing remain fast.

Possible mitigation:

- add Harpy-specific hit testing for primary shapes using cached Shapely
  polygons/prepared geometries;
- use this only for selection/picking, not for rendering;
- invalidate the cached geometry for the edited row after data changes.

Prepared Shapely point-in-polygon checks were negligible compared with
triangulation in local timing.

## Implementation Sketch

- Keep `ensure_shapes_triangulation_backend()` on Numba.
- Add a Harpy edge-only polygon shape class for primary/editable layers.
- In `_HarpyShapes`, override the layer construction/rebuild path so polygon
  rows are instantiated with the Harpy edge-only polygon class, while still
  exposing them as `polygon`.
- Keep styled/read-only shapes layers filled if desired.
- Ensure `layer.data = ...` rebuild paths, including create-holes and row
  replacement paths, also use the edge-only polygon class.
- Add tests covering:
  - large polygon edit avoids face triangles;
  - hole-bearing polygon draws shell and hole edges without bridge edges;
  - `layer.shape_type` remains `polygon`;
  - save conversion still produces polygons with holes;
  - primary-layer hit testing, if implemented.
