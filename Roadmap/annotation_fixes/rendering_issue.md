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

## What `bermuda` Does Wrong

For a polygon fill, triangulation should produce a non-overlapping triangle mesh
whose union is exactly the intended filled region:

```text
triangle union == polygon shell minus polygon holes
triangle interiors do not overlap
no triangle area lies inside a hole
```

For `__annotation_1`, the `bermuda` result is not a clean partition of the
filled polygon. It creates overlapping triangles and a small amount of fill
outside the intended Shapely polygon.

Observed metrics for the original `__annotation_1` row:

```text
Backend: Numba / VisPy
target area:       188061.368643
triangle sum:      188061.368643
triangle union:    188061.368643
overdraw:          0.0
symdiff vs target: 0.0

Backend: Fastest available / bermuda
target area:       188061.368643
triangle sum:      189357.041160
triangle union:    188084.592730
overdraw:          1272.448430
symdiff vs target: 23.224087
```

`overdraw` is:

```text
sum(triangle.area) - unary_union(triangles).area
```

This is important because the visible artifact can come from overlapping
triangles even when the final triangle union is close to the intended polygon.
With semi-transparent face rendering, overlapping triangles are drawn multiple
times and appear as darker wedges or stripes.

The likely failure mode is that `bermuda` mishandles napari's repeated-anchor
multi-ring path. Harpy encodes holes as one path:

```text
shell, shell anchor, hole, hole anchor, shell anchor
```

The bridge edges between shell and holes are meant to cancel out or be removed
before face triangulation. Napari's VisPy path handles that correctly. For this
geometry layout, `bermuda` appears to keep or reinterpret part of that repeated
path structure and generates invalid diagonals between shell/hole vertices,
which produces overlapping face triangles.

## Regression Geometry

The regression test should not depend on loading the full SpatialData Zarr
store. Use a hardcoded minimal Shapely geometry derived from `__annotation_1`:

- translate the original geometry to the origin;
- scale by `0.01`;
- simplify while preserving topology;
- keep the two triangular holes that still reproduce the `bermuda` overdraw.

The resulting fixture is a quadrilateral shell with two triangular holes. Use
Shapely `(x, y)` coordinate order:

```python
from shapely.geometry import Polygon


BERMUDA_HOLE_TRIANGULATION_REGRESSION_POLYGON = Polygon(
    shell=[
        (1.855311, 0.000000),
        (5.102106, 1.443020),
        (2.679894, 4.999033),
        (0.000000, 2.679895),
        (1.855311, 0.000000),
    ],
    holes=[
        [
            (3.699100, 2.834912),
            (2.909122, 2.780432),
            (3.045325, 3.080078),
            (3.699100, 2.834912),
        ],
        [
            (3.099806, 1.500122),
            (3.562897, 1.881492),
            (3.889784, 1.336680),
            (3.099806, 1.500122),
        ],
    ],
)
```

Observed metrics for this reduced fixture:

```text
Backend: Numba / VisPy
target area:       12.959622812
triangle sum:      12.959621970
triangle union:    12.959621970
overdraw:          0.0
symdiff vs target: 0.000001194

Backend: Fastest available / bermuda
target area:       12.959622812
triangle sum:      13.089189184
triangle union:    12.960768406
overdraw:          0.128420778
symdiff vs target: 0.001147616
```

The exact floating-point values may vary slightly by platform or dependency
version, so tests should use tolerances rather than exact metric equality.

## Regression Test Contract

The Harpy regression test should validate the backend guard and the rendered
mesh invariant.

Suggested mesh assertion:

```python
from shapely.geometry import Polygon
from shapely.ops import unary_union


def _assert_shape_mesh_matches_polygon(layer, row, expected):
    shape = layer._data_view.shapes[row]
    vertices = shape._face_vertices
    triangles = shape._face_triangles

    triangle_polygons = [
        Polygon([(vertices[index][1], vertices[index][0]) for index in triangle])
        for triangle in triangles
    ]
    triangle_polygons = [polygon for polygon in triangle_polygons if polygon.area > 0]

    triangle_union = unary_union(triangle_polygons)
    triangle_area_sum = sum(polygon.area for polygon in triangle_polygons)
    tolerance = expected.area * 1e-6

    assert triangle_union.symmetric_difference(expected).area < tolerance
    assert triangle_area_sum - triangle_union.area < tolerance
```

The first assertion catches triangles that fill holes or miss shell area. The
second assertion catches overlapping triangles that can still make the final
union look almost correct but render visibly wrong because semi-transparent
faces are drawn multiple times.

Recommended Harpy test flow:

1. Set napari settings/runtime state to an unsafe backend such as
   `Fastest available`.
2. Construct a Harpy-managed generic `Shapes` layer from
   `BERMUDA_HOLE_TRIANGULATION_REGRESSION_POLYGON`.
3. Assert Harpy switched settings and runtime triangulation to `Numba`.
4. Assert the resulting napari shape mesh matches the Shapely polygon using the
   mesh invariant above.

An optional upstream-facing test can directly force `Fastest available` with
`bermuda` installed and assert that the reduced fixture produces nonzero
overdraw. That test documents the napari/bermuda failure itself, but Harpy's
own regression should focus on proving that Harpy avoids the unsafe backend.

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
