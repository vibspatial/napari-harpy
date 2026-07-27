# Create Holes: Bermuda Failure and Full Rollback

## Status

Investigation and both implementation phases are complete. The shared
`_ShapesLayerBaseline`, `_capture_shapes_layer_baseline(...)`, and
`_restore_shapes_layer_baseline(...)` API lives in `_layer_state.py`; guarded
insertion, deletion, and `Create holes` use it. The exact Bermuda reproducer,
deterministic rollback regression, restoration-failure regression, and widget
failure path are covered by passing tests. Opening the upstream Bermuda issue
remains the follow-up.

This document records the `Create holes` failure observed while editing the
`tumor` Shapes element in:

```text
/Users/arne.defauw/VIB/DATA/test_data/sdata_xenium_3_6_26.zarr
```

The agreed Harpy scope is deliberately narrow:

- keep Bermuda as the configured triangulation backend;
- do not alter, simplify, perturb, translate, scale, or cyclically re-anchor
  valid polygon geometry in an attempt to make Bermuda accept it;
- make `Create holes` a full transaction and restore the complete live layer
  when rendering or triangulation fails;
- after successful rollback, show the failure in the Shapes Annotation status
  card instead of allowing the application exception to escape the Qt
  callback;
- preserve the exact failing geometry as a regression fixture and use it to
  report the triangulation defect upstream to Bermuda.

Fixing Bermuda's sweep-line triangulator belongs in Bermuda. Harpy owns the
stronger application-level guarantee that an upstream rendering failure must
not destroy the user's current annotation session.

## User-visible Failure

The user selected a shell and two polygons inside it, then clicked `Create
holes`. Bermuda panicked while napari rebuilt the replacement polygon:

```text
thread '<unnamed>' panicked at crates/triangulation/src/face_triangulation.rs:531:60:
called `Option::unwrap()` on a `None` value
```

Napari caught the PyO3 panic and raised:

```text
RuntimeError: Triangulation failed. Data saved to ...npz and ...txt
```

The exception escaped the button callback. More importantly, the active
Shapes layer could no longer be used to annotate `tumor` after the error.

## Captured Reproducer

Napari saved the exact candidate passed to Bermuda at:

```text
/var/folders/sz/t3tgg4fs4tz9btm0fbqg_tzc0000gn/T/napari_bermuda_triang_hzq4dkn2.npz
/var/folders/sz/t3tgg4fs4tz9btm0fbqg_tzc0000gn/T/napari_bermuda_triang_sm7uznve.txt
```

The temporary paths must not be used by tests. The exact text artifact is
checked into the repository at:

```text
tests/fixtures/create_holes_triangulation_failure.txt
```

Artifact integrity at investigation time:

```text
NPZ SHA-256:       87c6ffe531d17510bbd17b4d24c277a36b8488bc31816791c1efb1a887f3d11e
TXT SHA-256:       7f09c0043050e518052e52736fd68d7cd400611697675b1e3dc6c4977f4639d2
data-byte SHA-256: 21df8806d99580c208b26e075297fdbefeac4ce8f134ac370328203af55975ec
```

The NPZ contains one `data` array with shape `(570, 2)` and dtype `float32`.
Coordinates are in napari `(y, x)` order. Its encoded topology is:

```text
indices 0:546     exterior ring: 545 distinct path vertices plus closure
indices 546:557   first hole: 10 distinct path vertices plus closure
index   557       repeated exterior anchor separator
indices 558:569   second hole: 10 distinct path vertices plus closure
index   569       repeated exterior anchor separator/final closure
```

The decoded Shapely geometry is a non-empty valid `Polygon`:

- exterior area: `140075382.17486215`;
- first-hole area: `277656.0446577072`;
- second-hole area: `377113.79895591736`;
- final area: `139420612.33124852`;
- minimum clearance: approximately `17.5421`;
- both holes are strictly contained by the exterior;
- the holes are approximately `630.1431` units apart;
- all coordinates are finite;
- there are no consecutive duplicate coordinates.

The exterior matches the persisted `tumor` row at source index `1`. The
persisted `tumor` element still contains three valid polygon rows and all three
render successfully with Bermuda. The failed `Create holes` operation did not
reach the save path and did not overwrite the zarr element.

## Root Cause Analysis

### Upstream triangulation failure

The installed versions were:

```text
napari  0.7.1
bermuda 0.1.7
shapely 2.1.2
numpy   2.3.5
```

The exact candidate reproduces the panic through both
`bermuda.triangulate_polygons_face(...)` and
`bermuda.triangulate_polygons_with_edge(...)`. It succeeds through napari's
Numba backend, establishing that the geometry and napari hole encoding are not
generally unrenderable.

The failure is isolated to Bermuda face triangulation:

- the exterior alone succeeds;
- each hole alone succeeds;
- exterior plus the second hole succeeds;
- exterior plus the first hole panics;
- exterior plus both holes panics regardless of the order of the two holes.

The panic occurs in Bermuda's sweep-line implementation at an unchecked
`bottom_begin.next_back().unwrap()`. Bermuda reaches a state with an eligible
top segment but no remaining bottom segment.

The outcome is sensitive to equivalent numeric/path representations:

- cyclically choosing hole vertex `0`, `1`, or `2` as the first-hole anchor
  panics;
- choosing any of vertices `3` through `9` succeeds;
- translating the coordinates can change failure to success;
- scaling the coordinates can change failure to success or back to failure.

Those observations support an event-ordering or floating-point robustness
defect inside Bermuda. They do not justify rewriting user geometry in Harpy.
The exact same unchecked unwrap remains on Bermuda's current `main` branch, and
no existing Bermuda issue was found for this reproducer during the
investigation.

### Harpy transaction failure

`_AnnotationLayerEditGuard` provides rollback for guarded native polygon
movement, insertion, and deletion. `Create holes` is a custom button action and
does not pass through those guarded mutation routes.

The current custom path is:

1. `_create_holes_plan_from_selection(...)` validates and creates a valid
   Shapely polygon with direct holes;
2. `_apply_create_holes_plan(...)` captures styles but not a restorable copy of
   the complete layer;
3. it assigns `layer.data = rebuilt_data`;
4. napari replaces `layer._data_view` with a new empty `ShapeList` before it
   constructs and triangulates all replacement rows;
5. Bermuda panics on the new hole-bearing shell;
6. napari raises `RuntimeError` before the new rows are installed;
7. `_on_create_holes_clicked(...)` catches only `ValueError`, so the rendering
   failure escapes and no rollback runs.

An exact minimal reproduction starts with three individually Bermuda-renderable
rows: the captured exterior and the two captured holes. After the failed apply,
the live napari layer contains:

```text
len(layer.data)              == 0
len(layer.shape_type)        == 0
len(layer._data_view.shapes) == 0
len(layer.features)          == 3
layer.selected_data          == set()
```

This split state explains why the Shapes Annotation UI is unusable after the
error. The source geometry was valid, but napari's public data setter is not an
atomic operation when construction of a replacement shape raises.

## Required Product Behavior

`Create holes` must have all-or-nothing semantics.

On success, existing behavior remains unchanged: the largest selected polygon
survives as the shell, selected child polygons become direct holes, child rows
are removed, row identities and styles remain aligned, the final shell remains
selected, and the annotation session becomes dirty.

On any exception during live application, triangulation, row removal, style
restoration, or final refresh:

1. restore the exact complete pre-action layer state;
2. do not emit or display a success result;
3. display a concise error status explaining that holes could not be created
   and that the original annotations were restored;
4. leave the layer selected, editable, and usable for further work;
5. leave the dirty state exactly as it was before the click;
6. if restoration succeeds, consume the application error and report a normal
   unsuccessful result, matching guarded vertex insertion and deletion;
7. only if restoration itself fails, raise an `ExceptionGroup` containing the
   original `application_error` followed by the `restoration_error`, with the
   application error retained as `__cause__`.

The complete restorable baseline must contain real copies, not only the hashed
`_ShapesAnnotationLayerSnapshot`. At minimum it must cover:

- every vertex array in `layer.data`;
- ordered shape types;
- `features` and `feature_defaults`;
- row edge colors, face colors, edge widths, and z indices;
- opacity;
- current drawing defaults;
- selected rows;
- layer mode;
- any other napari row-aligned state already handled by the existing
  row-length-changing edit rollback.

The layer binding and annotation session objects should remain the same. A
failed action is not a close/reopen operation and must not reload the saved
SpatialData element over unrelated unsaved edits.

## Proposed Implementation Direction

Implement this in two separately reviewable phases. The first phase is a
behavior-preserving extraction of the existing insertion/deletion recovery
mechanism. The second phase uses that established mechanism for `Create
holes`. Do not create a second, weaker rollback definition specifically for
holes.

### Phase 1: extract shared Shapes layer state recovery (complete)

Add the neutral module
`napari_harpy.widgets.shapes_annotation._layer_state` with this private API:

```python
@dataclass(frozen=True)
class _ShapesLayerBaseline:
    ...


def _capture_shapes_layer_baseline(layer: Shapes) -> _ShapesLayerBaseline:
    ...


def _restore_shapes_layer_baseline(
    layer: Shapes,
    baseline: _ShapesLayerBaseline,
) -> None:
    ...
```

This is an extraction, not a new recovery implementation:

- rename and generalize `_PolygonVertexRowChangeBaseline` to
  `_ShapesLayerBaseline`;
- move the existing
  `_restore_polygon_vertex_row_change_baseline(...)` body unchanged into
  `_restore_shapes_layer_baseline(...)`;
- extract the two identical inline baseline constructions in guarded vertex
  insertion and deletion into `_capture_shapes_layer_baseline(...)`;
- update insertion and deletion to call those two shared functions;
- update their type references, tests, and restoration-failure monkeypatch
  targets accordingly.

Phase 1 must not change `Create holes`. Its purpose is to prove that the
generalized API preserves existing insertion/deletion behavior before another
caller depends on it. Focused verification must cover successful insertion and
deletion, successful rollback after an application failure, and the existing
`ExceptionGroup` behavior when restoration also fails.

Do not use `_ShapesAnnotationLayerSnapshot` for this API. That snapshot exists
for dirty-state comparison and does not contain the complete copied state
required for restoration.

### Phase 2: make Create holes transactional (complete)

`_create_holes.py` imports the same `_ShapesLayerBaseline` capture and restore
functions established in phase 1. It contains no duplicate baseline or
restoration logic.

`_apply_create_holes_plan(...)` is one transaction and reports whether it
committed:

```python
baseline = _capture_shapes_layer_baseline(layer)

try:
    apply_rebuilt_data_and_remove_hole_rows(layer, plan)
    restore_success_styles_selection_and_mode(layer, baseline, plan)
    layer.refresh()
except Exception as application_error:
    try:
        _restore_shapes_layer_baseline(layer, baseline)
    except Exception as restoration_error:
        raise ExceptionGroup(
            "Create holes failed and restoring the previous layer state also failed.",
            [application_error, restoration_error],
        ) from application_error
    return False
return True
```

The transaction catches normal application exceptions, including napari's
`RuntimeError` wrapper around the PyO3 panic. If restoration succeeds, it does
not re-raise that application error; it returns `False`, just as guarded vertex
insertion and deletion warn and return after successfully restoring their
baseline. Harpy does not need to catch a raw `BaseException` from Bermuda
because napari already catches that at its triangulation boundary and raises
`RuntimeError`.

Only restoration failure raises. That `ExceptionGroup` must preserve the two
distinct exception objects in order, `(application_error,
restoration_error)`, and use `application_error` as its cause, matching the
existing insertion/deletion tests.

The button callback should treat a `False` result as an unsuccessful restored
operation: refresh button/dirty readiness and show an error card explaining
that the original annotations were restored. It must not call the success-card
path. An `ExceptionGroup` is allowed to escape this normal handling because it
means the safety guarantee itself failed and the layer may not be usable.

The implementation does not:

- switch to Numba;
- automatically simplify polygons;
- perturb or round vertices;
- translate or scale annotation coordinates;
- rotate ring anchors until Bermuda accepts the path;
- silently drop a hole;
- save or reload the SpatialData element as a substitute for in-memory
  rollback.

## Regression Fixture Design

The repository fixture should be self-contained and must not depend on the
external Xenium zarr store or macOS temporary files.

Keep all 570 text coordinates because the defect depends on the full exterior,
the selected hole anchor, and floating-point values. The loader reads them
directly as `float32` and verifies the resulting byte hash, proving that the
checked-in text reconstructs the original NPZ array exactly. A reduced or
rounded fixture could stop exercising the failure.

The fixture loader can split the encoded candidate into semantic component
rows:

```python
failed_vertices = np.loadtxt(fixture_path, dtype=np.float32)
shell_vertices = failed_vertices[:546]
hole_1_vertices = failed_vertices[546:557]
hole_2_vertices = failed_vertices[558:569]
```

The separator vertices at `557` and `569` belong only to the combined napari
hole encoding and are not separate source polygon rows.

Before using the fixture, the test should assert its shape, dtype, and byte
hash so an accidental fixture rewrite is visible:

```python
assert failed_vertices.shape == (570, 2)
assert failed_vertices.dtype == np.float32
assert hashlib.sha256(failed_vertices.tobytes()).hexdigest() == (
    "21df8806d99580c208b26e075297fdbefeac4ce8f134ac370328203af55975ec"
)
```

## Harpy Rollback Regression Test

The permanent Harpy regression must test Harpy's transaction contract, not
require a particular Bermuda version to remain broken forever.

Use the full captured exterior and holes as individually renderable starting
rows. Add at least one unrelated polygon and non-default features/styles so the
test proves that rollback restores the entire layer rather than only the
failing shell. Put the rows in an order where shell and hole removal require
index remapping, for example:

```text
row 0: captured hole 1       selected
row 1: unrelated polygon     unselected
row 2: captured exterior     selected shell
row 3: captured hole 2       selected
```

The currently implemented issue-specific support and tests are grouped in
`TestCreateHolesTriangulationFailure` in
`tests/test_shapes_annotation_create_holes.py`. Its class docstring records the
captured input, the Bermuda/napari failure boundary, and the purpose of each
test. The class contains the fixture metadata and loader, layer factory, and
complete-state capture/assertion helpers.

The rollback tests use the same no-raise contract as insertion and deletion:

- `test_full_geometry_bermuda_failure_restores_complete_layer` calls
  `Create holes` with real Bermuda, asserts that the transaction returns
  `False`, and then requires the full baseline to be restored. This is the
  temporary fixture-specific rollback characterization; when Bermuda accepts
  this input, the rollback expectation is no longer applicable and the test
  can be removed or converted to a success test.
- `test_artificial_render_failure_restores_complete_layer` injects a failure
  only for the exact combined candidate. Every other triangulation delegates
  to real Bermuda, including the baseline rebuild during rollback. This is the
  permanent Harpy transaction regression. It likewise asserts `False` and a
  complete restored baseline, without `pytest.raises(...)`.

Both rollback tests now pass without XFAIL markers. A third transaction test
covers the exceptional recovery path by independently injecting application
and restoration failures. It asserts that `_apply_create_holes_plan(...)`
raises an `ExceptionGroup`, that
`caught.value.exceptions == (application_error, restoration_error)`, and that
`caught.value.__cause__ is application_error`, mirroring the existing
insertion/deletion restoration-failure tests.

Together, the rollback and widget assertions cover:

- all four vertex arrays, byte-for-byte or with exact array equality;
- ordered shape types;
- complete features and feature defaults;
- edge/face colors, widths, z indices, opacity, and current defaults;
- `Mode.DIRECT`;
- selection `{0, 2, 3}`;
- a working `_data_view` with four shapes;
- unchanged annotation dirty state;
- continued ability to perform another valid annotation edit.

Using an injected candidate-specific failure is important:

- it deterministically exercises the rollback even after Bermuda is fixed;
- it avoids making the Harpy suite assert that an upstream bug must continue
  to exist;
- it proves restoration through the real Bermuda path because only the
  combined candidate is rejected;
- it separates Harpy's safety contract from the upstream algorithm defect.

A focused widget test additionally clicks `Create holes`, injects the same
candidate-specific failure, and asserts:

- no exception escapes the Qt callback;
- the unsuccessful transaction result is handled without entering the success
  path;
- the error status card is visible;
- no success card is shown;
- the layer equals its baseline;
- Save and Create holes readiness is refreshed and the annotation layer remains
  usable.

## Bermuda Characterization and Upstream Test

The temporary real-Bermuda Harpy rollback test and the artificial permanent
rollback test are separate methods in the grouped class. The former does not
need to assert an escaping `RuntimeError` once Harpy follows the
insertion/deletion contract. The upstream Bermuda issue/patch should add its
own desired-success test that calls Bermuda directly:

```python
def test_full_napari_hole_path_does_not_panic():
    vertices = np.loadtxt(fixture_path, dtype=np.float32)

    (triangles, face_vertices), edge_mesh = (
        bermuda.triangulate_polygons_with_edge([vertices])
    )

    assert len(triangles) > 0
    assert len(face_vertices) > 0
```

On Bermuda `0.1.7` this test panics at
`face_triangulation.rs:531`. Once Bermuda is corrected, it should pass without
Harpy changing or re-encoding the input.

Harpy's temporary real-Bermuda rollback characterization is isolated; the
deterministic artificial-failure rollback test is the required permanent CI
protection for Harpy. Bermuda's direct desired-success test remains the place
where the exact candidate is expected to triangulate without error after the
upstream fix.

## Upstream Bermuda Issue Follow-up

Open a Bermuda issue with:

- Bermuda, napari, NumPy, Python, OS, and architecture versions;
- the original exact NPZ attachment;
- its SHA-256 hashes;
- the minimal direct call to `triangulate_polygons_with_edge([data])`;
- the Rust backtrace identifying `face_triangulation.rs:531`;
- the Shapely validity, containment, area, and minimum-clearance diagnostics;
- confirmation that every component ring triangulates independently;
- confirmation that exterior plus hole 2 succeeds while exterior plus hole 1
  panics;
- the cyclic-anchor and translation/scale sensitivity as evidence of
  event-ordering or floating-point sensitivity;
- the expectation that library input should return a structured error rather
  than panic even if Bermuda ultimately considers the path unsupported.

Link the final upstream issue from this document and from any temporary
characterization marker.

## Acceptance Criteria

- The exact full-geometry fixture is checked into repository test data with an
  integrity assertion.
- `Create holes` remains on Bermuda and does not rewrite valid geometry.
- A forced rendering failure restores the complete live Shapes layer.
- A rendering/application failure followed by successful restoration returns
  an unsuccessful result and does not raise.
- A restoration failure raises an `ExceptionGroup` containing both the
  application and restoration errors, with the application error as its cause.
- When restoration succeeds, no application exception escapes the `Create
  holes` Qt callback; restoration failure remains exceptional.
- The user receives an explicit restored-original-annotations error status.
- Dirty state is unchanged by a failed operation.
- The annotation layer remains editable immediately after rollback.
- Successful `Create holes` behavior and save/reload behavior remain covered.
- Insertion and deletion use the shared `_layer_state.py` capture/restore API
  without a behavior change.
- `Create holes` uses that same capture/restore implementation and contains no
  duplicate baseline or restoration logic.
- Focused tests cover the transaction and widget paths.
- A separate Bermuda issue contains the exact upstream reproducer.
