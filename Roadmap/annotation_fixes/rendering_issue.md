# Shape Hole Rendering Issue

## Status

Investigated. No implementation has been done for this issue yet.

## Context

The issue was observed for the shapes element `new_shapes`, annotation row
`__annotation_1`, from:

`/Users/arne.defauw/VIB/DATA/test_data/sdata_xenium_3_6_26.zarr`

In the viewer, the polygon holes appeared to render incorrectly. The visible
artifact looked like napari was not handling the repeated anchor/separator
vertices correctly for a polygon-with-holes row.

This investigation checked whether the problem came from:

- corrupted geometry in the SpatialData store;
- Harpy's Shapely-to-napari polygon encoding;
- Harpy's viewer adapter;
- recent annotation interaction changes;
- napari's runtime triangulation/rendering backend.

## Findings

The stored geometry is valid.

- `new_shapes` contains 8 rows.
- `__annotation_1` is a valid Shapely `Polygon`.
- It has 3 interior rings.
- Its area, bounds, exterior, and interior rings are coherent.
- No other `new_shapes` row overlaps `__annotation_1`.

Harpy's encoding is also valid.

- `shapely_polygon_to_napari_polygon_vertices(...)` encodes the polygon as one
  napari polygon row with embedded interior rings.
- The repeated shell anchor/separator topology for `__annotation_1` is:

```text
shell_anchor_group = (0, 16, 25, 33, 40)
hole_anchor_groups = ((17, 24), (26, 32), (34, 39))
```

- Decoding the encoded napari vertices with
  `napari_polygon_vertices_to_shapely_polygon(...)` roundtrips back to the
  original Shapely polygon.
- The exterior ring and interior rings have opposite winding as expected by the
  existing napari hole encoding.

The viewer adapter passes the correct row to napari.

- `_prepare_napari_shapes_layer_inputs(...)` maps row 1 to
  `instance_id == "__annotation_1"`.
- The adapter output row is exactly equal to the output of
  `shapely_polygon_to_napari_polygon_vertices(...)`.
- The adapter feature column is `instance_id`.
- The row shape type is `polygon`.

The recent Space-pan changes are unlikely to be related.

- Those changes affect active annotation interaction callbacks and keybindings.
- They do not alter load-time Shapely geometry conversion.
- They do not alter the viewer adapter.
- They do not alter napari's face triangulation.

## Reproduction Summary

Headless napari mesh inspection reproduces the issue when napari uses its
compiled `bermuda` triangulation backend.

Environment observed during investigation:

```text
napari 0.7.1
bermuda 0.1.7
spatialdata 0.7.2
shapely 2.1.2
```

For the exact `__annotation_1` vertex row:

```text
Backend: Pure python / VisPy
face mesh symmetric difference from Shapely polygon: 0.0
holes render geometrically correctly
```

```text
Backend: Fastest available / bermuda
face mesh symmetric difference from Shapely polygon: 23.224087
triangle area sum is too high
extra/overlapping triangles appear around an interior hole
```

When the whole `new_shapes` layer is loaded:

```text
Backend: Pure python
__annotation_1 mesh is exact

Backend: Fastest available
__annotation_1 mesh has extra/overlapping triangles
```

Direct calls into `bermuda.triangulate_polygons_face(...)` and
`bermuda.triangulate_polygons_with_edge(...)` reproduce the same bad mesh for
the same vertex row.

Changing ring orientation did not fix the `bermuda` result. Both orientation
variants were exact through the VisPy path and bad through the `bermuda` path.

## Likely Root Cause

Harpy stores direct polygon holes using napari's embedded-ring path encoding:

```text
shell, repeated shell anchor, hole, repeated hole anchor, repeated shell anchor
```

For example:

```text
A B C D A E F G H E A
```

Napari's VisPy/pure-Python path handles this by normalizing duplicate vertices
and removing duplicate/repeated bridge edges before triangulating the face.

In the installed napari stack, the compiled `bermuda` path sends the raw polygon
path directly to `bermuda.triangulate_polygons_face(...)` or
`bermuda.triangulate_polygons_with_edge(...)`. For this row, `bermuda` does not
produce an equivalent hole-aware face mesh.

So the problem appears to be a napari/bermuda rendering backend incompatibility
with this repeated-anchor direct-hole encoding.

## Relevant Code Paths

Harpy encoding:

- `src/napari_harpy/core/shapes_geometry.py`
- `shapely_polygon_to_napari_polygon_vertices(...)`

Harpy adapter:

- `src/napari_harpy/viewer/adapter.py`
- `_prepare_napari_shapes_layer_inputs(...)`
- `_shapely_polygon_to_napari_polygon_vertices(...)`

Napari compiled backend path:

- `.venv/lib/python3.13/site-packages/napari/layers/shapes/_shapes_models/shape.py`
- `_set_meshes_compiled_bermuda(...)`

Napari VisPy/pure path:

- `.venv/lib/python3.13/site-packages/napari/layers/shapes/_shapes_models/shape.py`
- `_set_meshes_py(...)`
- `triangulate_face_and_edges(...)`

## Things To Worry About

This issue is data-dependent. Simple polygons and some hole-bearing polygons may
look correct, while specific polygons with multiple holes or certain geometry
layouts may render incorrectly under `bermuda`.

The saved geometry may still be correct even when the viewer rendering looks
wrong. The artifact is in the displayed face mesh, not necessarily in the
SpatialData store.

If users edit a visually wrong shape, they may make decisions based on an
incorrect rendered fill even though the underlying geometry is valid.

The issue may become more visible when napari defaults to, or is configured for,
`Fastest available` triangulation and `bermuda` is installed.

## Potential Mitigations

The lowest-risk mitigation is to avoid napari's `bermuda` triangulation backend
for Harpy-rendered Shapes layers.

The preferred practical backend is `TriangulationBackend.numba`, not necessarily
`Pure python`. For the problematic `__annotation_1` row, both `Numba` and
`Pure python` rendered the hole-bearing polygon exactly, while
`Fastest available` selected `bermuda` and produced the bad mesh.

Harpy should use the same triangulation backend for all Shapes layers it
renders or manages. We should not first inspect every shapes element to decide
whether it currently contains polygon holes. A single Harpy Shapes policy is
simpler and safer because:

- napari's triangulation backend is process-global, not per layer;
- a layer can start without holes and later gain holes through create-holes or
  editing;
- per-layer or per-row backend behavior would be surprising to reason about;
- simple polygons, imported native Shapes, future hole-bearing polygons, and
  edited Shapes should render consistently;
- correctness is more important than selecting the fastest compiled backend for
  simple polygons.

Important implementation detail: calling
`napari.utils.triangulation_backend.set_backend(TriangulationBackend.numba)`
alone is not enough. Napari settings own a separate
`settings.experimental.triangulation_backend` value. When Harpy launches
`Interactive(..., async_slicing=False)`, it calls `get_settings()` to update
napari's async slicing setting. Loading napari settings can re-emit the saved
triangulation backend value, so a previous raw `set_backend(Numba)` call can be
overwritten back to `Fastest available`.

The Harpy-side mitigation should therefore update both napari's settings value
and the active runtime backend:

```python
from napari.settings import get_settings
from napari.utils.triangulation_backend import (
    TriangulationBackend,
    get_backend,
    set_backend,
)

_HARPY_SHAPES_TRIANGULATION_BACKEND = TriangulationBackend.numba
_UNSAFE_HARPY_SHAPES_TRIANGULATION_BACKENDS = {
    TriangulationBackend.fastest_available,
    TriangulationBackend.bermuda,
}


def _ensure_harpy_shapes_triangulation_backend() -> None:
    settings = get_settings()
    settings_backend = settings.experimental.triangulation_backend
    runtime_backend = get_backend()

    if settings_backend in _UNSAFE_HARPY_SHAPES_TRIANGULATION_BACKENDS:
        settings.experimental.triangulation_backend = _HARPY_SHAPES_TRIANGULATION_BACKEND

    if runtime_backend in _UNSAFE_HARPY_SHAPES_TRIANGULATION_BACKENDS:
        set_backend(_HARPY_SHAPES_TRIANGULATION_BACKEND)
```

The helper should be called whenever Harpy is about to create, open, import, or
mutate a napari `Shapes` layer that Harpy owns or manages. We should not call it
for point-radius shapes rendered as napari `Points`, because those do not use
Shapes face triangulation.

Likely integration points:

1. In `_build_shapes_layer(...)`, call
   `_ensure_harpy_shapes_triangulation_backend()` before constructing
   `_HarpyShapes(...)` for generic shapes rendering.
2. In `_build_empty_primary_shapes_layer(...)`, call the same helper before
   constructing an empty editable `_HarpyShapes` layer.
3. In `_replace_shapes_layer_preserving_state(...)`, call the same helper before
   constructing the replacement `_HarpyShapes` layer.
4. Before applying create-holes mutations, call the same helper because the
   operation creates a repeated-anchor hole-bearing polygon row.
5. When importing or attaching to an existing native napari Shapes layer, call
   the same helper before Harpy enables editing or rewrites layer data.
6. Investigate whether napari exposes a layer-local triangulation setting. If
   not, global backend switching must be treated carefully because it may affect
   other Shapes layers.
7. Report a minimal reproduction upstream to napari/bermuda using the
   `__annotation_1` vertex row.

The most important design constraint is that the mitigation should not corrupt
or rewrite the saved geometry. The stored Shapely polygon and Harpy's encoded
napari row are valid; the problem is the runtime triangulation backend.

## Backend Scope

Napari's triangulation backend is process-global state, not layer-local state.
That means a temporary wrapper like this is not robust for Harpy-managed Shapes
layers:

```python
previous = set_backend(TriangulationBackend.numba)
layer = _HarpyShapes(...)
set_backend(previous)
```

That pattern may fix initial layer construction, but later editing or refreshes
can trigger remeshing after the backend has been restored to an unsafe value.
For Harpy-managed Shapes, the Harpy backend must remain active while those
layers can be edited or remeshed.

Because updating `settings.experimental.triangulation_backend` may affect the
user's napari session preference, the implementation should make this behavior
explicit and log when Harpy switches from an unsafe backend to `Numba`.

## Suggested Next Step

Implement a small backend guard in the viewer/annotation code path, covered by
tests that verify:

- constructing any Harpy generic Shapes layer switches unsafe
  `Fastest available` or `bermuda` settings/runtime state to `Numba`;
- point-radius shapes rendered as napari `Points` do not need the Shapes
  triangulation guard;
- direct `set_backend(Numba)` is not the only operation performed; the napari
  settings value is also updated;
- the `__annotation_1` regression row triangulates exactly under the selected
  backend.
