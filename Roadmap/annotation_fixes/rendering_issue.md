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

### Preferred Harpy Regression Fixture

The reduced fixture above is useful for explaining the `bermuda` failure, but
it is not the best primary Harpy regression fixture. In the napari UI, the
reduced CSV example can still be confusing because a layer may have been opened
before Harpy switches the backend, or may keep a cached `bermuda` mesh even
after `get_backend()` later reports `Numba`.

For Harpy's own unit test, use a hardcoded fixture that stays close to the real
`new_shapes` / `__annotation_1` geometry from:

`/Users/arne.defauw/VIB/DATA/test_data/sdata_xenium_3_6_26.zarr`

This fixture should keep the original coordinate scale and ring complexity:

- 17 shell coordinates including the closing coordinate;
- 3 interior rings;
- interior ring coordinate counts of 8, 7, and 6 including closures;
- a resulting napari direct-hole path with 41 `(y, x)` vertices.

Use this Shapely `(x, y)` fixture in the regression test:

```python
from shapely.geometry import Polygon


ANNOTATION_1_HOLE_TRIANGULATION_REGRESSION_POLYGON = Polygon(
    shell=[
        (1883.543213, 2352.524414),
        (2182.454590, 2429.829102),
        (2208.222656, 2496.826416),
        (2208.222656, 2615.360107),
        (2177.300781, 2713.279297),
        (2120.610840, 2780.276855),
        (2063.920654, 2821.505859),
        (1966.001465, 2852.427734),
        (1899.004150, 2852.427734),
        (1821.699463, 2831.813232),
        (1775.316772, 2800.891357),
        (1713.473022, 2697.818359),
        (1698.012085, 2620.513916),
        (1698.012085, 2543.209229),
        (1734.087524, 2460.750977),
        (1832.006836, 2373.138916),
        (1883.543213, 2352.524414),
    ],
    holes=[
        [
            (1841.824951, 2505.260742),
            (1814.584351, 2521.605225),
            (1790.067871, 2548.845703),
            (1803.688110, 2581.534424),
            (1847.273071, 2589.706543),
            (1877.237793, 2581.534424),
            (1888.134033, 2543.397705),
            (1841.824951, 2505.260742),
        ],
        [
            (2037.957397, 2584.258545),
            (2010.716797, 2600.602783),
            (1988.924316, 2630.567627),
            (2002.544556, 2660.532227),
            (2037.957397, 2663.256348),
            (2067.922119, 2636.015625),
            (2037.957397, 2584.258545),
        ],
        [
            (2007.992676, 2502.536621),
            (2016.164917, 2540.673584),
            (2054.301758, 2540.673584),
            (2081.542236, 2513.432861),
            (2086.990479, 2486.192383),
            (2007.992676, 2502.536621),
        ],
    ],
)
```

Observed mesh behavior for this original-scale fixture:

```text
Backend: Fastest available / bermuda
shape mesh path:  _set_meshes_compiled_bermuda
face vertices:    34
face triangles:   38
overdraw:          ~1272.448430
symdiff vs target: ~23.224523

Backend: Numba / VisPy path
shape mesh path:  _set_meshes_py
face vertices:    36
face triangles:   38
overdraw:          0.0
symdiff vs target: 0.0 when compared to the layer-decoded polygon
```

The test should compare the mesh to
`napari_polygon_vertices_to_shapely_polygon(layer.data[0])`, not only to the
original Python literal. Napari can introduce tiny internal floating-point
differences when creating a `Shapes` layer; comparing against the layer-decoded
polygon keeps the test focused on triangulation correctness rather than
coordinate storage noise.

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

Recommended Slice 1 test flow:

1. Set napari settings/runtime state to an unsafe backend such as
   `Fastest available`.
2. Import or reload `napari_harpy`.
3. Assert importing Harpy switched settings and runtime triangulation to
   `Numba`.
4. Construct a plain napari `Shapes` layer from
   `ANNOTATION_1_HOLE_TRIANGULATION_REGRESSION_POLYGON`.
5. Assert the shape mesh path is `_set_meshes_py`, not
   `_set_meshes_compiled_bermuda`.
6. Decode `layer.data[0]` back to a Shapely polygon and assert the resulting
   napari shape mesh matches that polygon using the mesh invariant above.

An optional upstream-facing test can directly force `Fastest available` with
`bermuda` installed and assert that the reduced fixture produces nonzero
overdraw. That test documents the napari/bermuda failure itself, but Harpy's
own regression should focus on proving that Harpy avoids the unsafe backend.

## Cached Mesh Finding

One important follow-up finding: `get_backend() == TriangulationBackend.numba`
does not prove that an already-loaded `Shapes` layer was triangulated with
Numba.

Napari builds and caches each shape's face mesh when the shape is constructed
or remeshed. If a layer was created while the active backend selected
`bermuda`, that layer can keep the bad `bermuda` triangles even after the
global backend later reports `Numba`.

Observed with the reduced regression fixture:

```text
Layer created under Numba:
current backend:  Numba
shape mesh path:  _set_meshes_py
face vertices:    12
face triangles:   12
overdraw:         0.0

Layer created under Fastest available / bermuda:
current backend:  Fastest available
shape mesh path:  _set_meshes_compiled_bermuda
face vertices:    10
face triangles:   12
overdraw:         ~0.128

Same bad layer after switching backend to Numba:
current backend:  Numba
shape mesh path:  _set_meshes_compiled_bermuda
face vertices:    10
face triangles:   12
overdraw:         ~0.128
```

Calling `layer.refresh()` did not rebuild the face mesh. Reassigning the layer
data under the safe backend did rebuild it:

```python
set_backend(TriangulationBackend.numba)
layer.data = [row.copy() for row in layer.data]
```

After data reassignment, the same layer rebuilt through `_set_meshes_py` and the
mesh invariant passed.

This means the Harpy fix must do more than make `get_backend()` print `Numba`.
Harpy must ensure the backend is safe before constructing Harpy-managed Shapes
layers, and any existing native napari Shapes layer that Harpy adopts or
rewrites should be reconstructed or have its data reassigned after the backend
guard has run. Existing cached bad meshes are not repaired by changing backend
state alone.

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

Checking only `get_backend()` can be misleading. The currently active backend
may be `Numba` while a specific already-loaded layer still contains a cached
`bermuda` mesh. Debugging should inspect the shape mesh path as well:

```python
shape = layer._data_view.shapes[0]
print(getattr(shape._set_meshes, "__name__", None))
```

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
    switched_from = {
        backend
        for backend in (settings_backend, runtime_backend)
        if backend in _UNSAFE_HARPY_SHAPES_TRIANGULATION_BACKENDS
    }

    if switched_from:
        settings.experimental.triangulation_backend = _HARPY_SHAPES_TRIANGULATION_BACKEND
        if get_backend() != _HARPY_SHAPES_TRIANGULATION_BACKEND:
            set_backend(_HARPY_SHAPES_TRIANGULATION_BACKEND)
```

### Slice 1: Import-Time Numba Backend Policy

Current code state after rollback:

- `src/napari_harpy/__init__.py` currently only resolves `__version__` and
  installs lazy module attributes through `lazy_loader`.
- There is no triangulation helper in `src/napari_harpy`.
- `viewer/adapter.py` and the annotation widget do not currently set napari's
  triangulation backend.

Slice 1 should be deliberately minimal: set Harpy's Shapes triangulation policy
as soon as `napari_harpy` is imported, then verify that a Shapes layer created
after that import uses the Numba/VisPy path for the `__annotation_1`-like
fixture.

Implementation specification:

1. Add a small private helper in `src/napari_harpy/__init__.py`, or in a tiny
   internal module imported by `__init__.py`, that applies the backend policy.
2. Call that helper during `napari_harpy` package import.
3. If either napari settings or runtime backend is unsafe
   (`Fastest available` or `bermuda`), set
   `settings.experimental.triangulation_backend` to
   `TriangulationBackend.numba` and ensure `get_backend()` also reports
   `Numba`.
4. Do not inspect individual shapes, rows, or polygons for holes.
5. Do not reconstruct or mutate any existing layers in this slice.

This import-time policy should help both Harpy-managed Shapes and plain napari
CSV/native Shapes workflows, as long as `napari_harpy` is imported before the
Shapes layer is constructed. If the backend setting is persisted by napari
settings, it may also help later napari processes, but the unit test should only
rely on behavior within the current Python process.

Important limitation: Slice 1 does not repair a Shapes layer that was already
constructed under `bermuda`. Such a layer can keep a cached bad face mesh until
it is reconstructed or its public `layer.data` is reassigned under a safe
backend. That repair path is explicitly out of scope for this first slice.

Out of scope for Slice 1:

- adding adapter-level guards before `_HarpyShapes(...)` construction;
- adding annotation-widget guards before `layer.data = ...` remeshes;
- reassigning `layer.data = [row.copy() for row in layer.data]`;
- native-layer adoption remesh repairs;
- per-shape or per-row hole detection;
- changing saved geometry or Harpy's direct-hole encoding.

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

Implement Slice 1 only, then report back before adding broader remesh hooks.

Slice 1 test coverage should verify:

- importing or reloading `napari_harpy` with unsafe napari settings/runtime
  backend state switches both to `Numba`;
- the implementation updates napari settings as well as runtime state, not only
  `set_backend(Numba)`;
- after `import napari_harpy`, constructing a plain napari `Shapes` layer from
  `ANNOTATION_1_HOLE_TRIANGULATION_REGRESSION_POLYGON` uses the Numba/VisPy
  mesh path rather than `_set_meshes_compiled_bermuda`;
- the same fixture triangulates without overdraw when the resulting mesh is
  compared to `napari_polygon_vertices_to_shapely_polygon(layer.data[0])`;
- tests restore both napari settings and runtime backend after each case, since
  triangulation backend state is process-global.

Manual verification after Slice 1:

1. Start from unsafe backend state (`Fastest available` / `bermuda`).
2. Import `napari_harpy`.
3. Create or open the `bermuda_hole_regression` Shapes layer only after that
   import.
4. Confirm `get_backend()` reports `Numba`.
5. Inspect the shape mesh path:

```python
shape = layer._data_view.shapes[0]
print(getattr(shape._set_meshes, "__name__", None))
```

The expected mesh path is `_set_meshes_py`. If the napari UI still renders the
fixture incorrectly under this setup, stop after Slice 1 and investigate the
remaining rendering path before adding remesh or adapter-level fixes.
