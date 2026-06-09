# Pan And Zoom While Drawing Shapes

Status: proposed

This roadmap investigates whether `ShapesAnnotation()` can support moving the
napari camera while the user is drawing a lasso polygon, and proposes a scoped
implementation plan for napari-harpy.

## Summary

napari does not currently support a true "pause lasso, pan camera, continue the
same lasso stroke" workflow for Shapes layers.

The practical first napari-harpy feature should be an experimental camera-nudge
mode for the annotation widget:

- keep the active Shapes layer in `add_polygon_lasso` mode;
- do not switch the layer to napari's `pan_zoom` mode;
- bind a small set of keyboard shortcuts that directly update
  `viewer.camera.center`;
- gate those shortcuts so they only act on the widget-owned primary annotation
  Shapes layer;
- treat mouse-wheel zoom as napari-owned behavior unless later testing shows a
  real gap.

This gives annotators a useful small-movement workaround without monkeypatching
napari's drawing internals.

## Investigation

As of 2026-06-09, upstream napari issue
`napari/napari#6885` is still open:

https://github.com/napari/napari/issues/6885

The issue describes the exact workflow needed for large ROI annotation: drawing
a shape that extends beyond the current viewport, then shifting the camera while
drawing. The proposed upstream solutions are:

- auto-pan near the canvas edge;
- arrow-key camera movement while drawing;
- Space-pan behavior that does not finish or cancel the active shape.

The issue is open, labeled `feature`, has no assignee, and has no linked
implementation.

napari's current Shapes documentation says that pan is disabled while using
adding and editing tools, while mouse-wheel zoom typically continues to work:

https://napari.org/dev/howtos/layers/shapes.html

The same page describes the lasso workflow as an active drawing operation:

- click to begin;
- move the pointer to draw;
- click or press `Esc` to finish.

The current napari source explains why Space and `pan_zoom` mode cannot preserve
an in-progress lasso shape:

- `ADD_POLYGON_LASSO` is handled by `add_path_polygon_lasso`;
- while moving the mouse, `ADD_POLYGON_LASSO` calls `polygon_creating`;
- only `PAN_ZOOM` is listed in `_interactive_modes`;
- when the Shapes layer mode changes while `_is_creating` is true, the mode
  setter calls `_finish_drawing()`.

Local confirmation in this repository's canonical environment:

```text
napari 0.7.0
```

Relevant local napari default shortcuts in `0.7.0`:

- `Left` increments dimensions to the left;
- `Right` increments dimensions to the right;
- `Alt+Up` and `Alt+Down` move dimension slider focus;
- `Meta+Up` and `Meta+Down` select the layer above or below on macOS;
- `Shift+Up` and `Shift+Down` extend layer selection.

This means bare arrow keys are technically possible to override, but they are
not a safe first default because they conflict with established napari viewer
navigation.

Important local files inspected:

- `src/napari_harpy/widgets/shapes_annotation/widget.py`
- `src/napari_harpy/viewer/adapter.py`
- `src/napari_harpy/core/shapes_annotation.py`
- `.venv/lib/python3.13/site-packages/napari/layers/shapes/shapes.py`
- `.venv/lib/python3.13/site-packages/napari/layers/shapes/_shapes_mouse_bindings.py`
- `.venv/lib/python3.13/site-packages/napari/components/_viewer_key_bindings.py`

## Current Codebase Fit

Useful existing pieces:

- `ShapesAnnotation` already owns the create-new annotation layer lifecycle.
- The annotation-created layer is registered as a primary shapes layer through
  `ViewerAdapter.create_empty_primary_shapes_layer(...)`.
- The widget already tracks the widget-owned annotation layer in
  `self._annotation_layer`.
- `_annotation_layer_binding_matches()` already verifies that the tracked layer
  is the widget-owned primary shapes layer and still targets the locked
  SpatialData element.
- `ViewerAdapter` already has camera helper precedent through
  `_capture_viewer_camera_state(...)` and `_restore_viewer_camera_state(...)`.
- napari exposes `viewer.bind_key(...)`.
- napari exposes `viewer.camera.center` and `viewer.camera.zoom`.

Important current gaps:

- There is no harpy-owned viewer interaction helper for widget-level
  keybindings.
- `ShapesAnnotation` has no user-facing controls for camera nudge settings.
- Current tests use lightweight dummy viewers that do not expose
  `bind_key(...)` or `camera`.

## Non-Goals

- Do not monkeypatch napari's Shapes mouse handlers.
- Do not change napari's built-in Space behavior.
- Do not switch the active layer to `pan_zoom` during a lasso stroke.
- Do not implement full auto-pan near the canvas edge in the first slice.
- Do not support 3D shape editing; napari Shapes editing is a 2D workflow.
- Do not apply shortcuts globally to arbitrary napari Shapes layers.
- Do not make direct camera nudges a substitute for an upstream native
  lasso-aware pan implementation.

## Recommended UX

First version:

- enable camera nudges automatically after the widget creates an annotation
  Shapes layer;
- use keyboard shortcuts that do not require leaving the lasso tool;
- default to `Ctrl+Alt+Left/Right/Up/Down`;
- optionally also support `Ctrl+Alt+H/J/K/L` as an alternate mapping if testing
  shows the arrow mapping is unreliable on some platform;
- do not use bare `Left/Right/Up/Down` in the first implementation because
  napari and Qt already use arrow keys for viewer navigation and widget focus;
- use a default nudge size of `80` screen pixels;
- expose the nudge size as a small numeric setting later, not in the first
  implementation slice.

Shortcut mapping:

```text
Ctrl+Alt+Left     nudge left
Ctrl+Alt+Right    nudge right
Ctrl+Alt+Up       nudge up
Ctrl+Alt+Down     nudge down
```

The status text after layer creation should mention that mouse-wheel zoom still
comes from napari, and camera nudges are available while the annotation layer is
active.

## Behavior Contract

Camera nudges should only run when all of these are true:

- a napari viewer is available;
- `viewer.camera.center` and `viewer.camera.zoom` are available;
- the viewer is in 2D display mode, or no `ndisplay` API is available in tests;
- `ShapesAnnotation._annotation_layer` is not `None`;
- the viewer's active layer is that same annotation layer;
- `_annotation_layer_binding_matches()` returns `True`;
- the annotation layer is a napari `Shapes` layer, not a point-rendered shapes
  layer;
- the camera zoom is finite and greater than zero.

Camera nudges should not:

- change `layer.mode`;
- change `layer.selected_data`;
- call `layer._finish_drawing()`;
- create, remove, or save shapes;
- affect styled shapes layers or external Shapes layers.

The nudge calculation should operate in screen-pixel units:

```python
data_delta = pixel_delta / viewer.camera.zoom
```

For 2D viewing, update the last two entries in `viewer.camera.center`:

```python
center[-2] += dy_px / zoom
center[-1] += dx_px / zoom
viewer.camera.center = tuple(center)
```

This follows napari's camera API, where `center` is the camera center and, in 2D
viewing, the last two values are used.

## Limitations

This is a workaround, not native lasso-aware panning.

Changing `viewer.camera.center` while drawing changes the data coordinate under
the cursor. The next mouse move can therefore add a lasso segment from the old
data position to the new data position. For small nudges this can be acceptable;
for large viewport moves it may create a visibly long edge that the user should
inspect before saving.

The workaround also cannot preserve the exact "cursor remains locked to the same
data position while the camera moves" behavior proposed upstream. That behavior
belongs in napari because it requires cooperation between the canvas, mouse
event stream, layer mode state, and active shape creation state.

## Implementation Plan

### Slice 1: Pure Camera Nudge Helper

Add a small helper module, for example:

```text
src/napari_harpy/widgets/shapes_annotation/camera_nudge.py
```

Suggested public helper:

```python
def nudge_viewer_camera(viewer: object, *, dx_px: float, dy_px: float) -> bool:
    ...
```

Rules:

- return `True` when the camera center was changed;
- return `False` when required viewer/camera attributes are missing;
- treat invalid, missing, non-finite, or non-positive zoom as no-op;
- preserve the original tuple/list length of `camera.center`;
- pad very short centers to at least two values defensively;
- catch `AttributeError`, `RuntimeError`, `TypeError`, and `ValueError` and
  return `False`.

Tests:

- `dx_px` changes `center[-1]` by `dx_px / zoom`;
- `dy_px` changes `center[-2]` by `dy_px / zoom`;
- zoom scaling is honored;
- missing camera returns `False`;
- zero or invalid zoom returns `False`;
- a one-value center is padded before assignment.

### Slice 2: Widget-Owned Keybinding Controller

Add a tiny controller object, also in `camera_nudge.py`:

```python
class ShapesAnnotationCameraNudgeController:
    def __init__(
        self,
        viewer: object | None,
        *,
        can_nudge: Callable[[], bool],
        pixels: int = 80,
    ) -> None:
        ...

    def install(self) -> None:
        ...
```

The controller should:

- no-op cleanly when `viewer` is `None`;
- no-op cleanly when `viewer.bind_key` is unavailable;
- bind the four default shortcuts with `overwrite=True`;
- install at most once per controller instance;
- call `can_nudge()` before changing the camera;
- leave all shape data and layer mode untouched.

`ShapesAnnotation` should instantiate this controller in `__init__`:

```python
self._camera_nudge_controller = ShapesAnnotationCameraNudgeController(
    napari_viewer,
    can_nudge=self._can_nudge_annotation_camera,
)
self._camera_nudge_controller.install()
```

Add a widget method:

```python
def _can_nudge_annotation_camera(self) -> bool:
    ...
```

This method should use the behavior contract above. It can reuse
`_annotation_layer_binding_matches()`.

Tests:

- keybindings are registered when the viewer exposes `bind_key`;
- registration is skipped for dummy viewers without `bind_key`;
- pressing a registered handler nudges the dummy camera when the annotation
  layer is active and binding matches;
- pressing a registered handler does nothing when there is no annotation layer;
- pressing a registered handler does nothing when the active layer is different;
- pressing a registered handler does nothing when the binding no longer matches;
- invoking handlers does not change `layer.mode`.

### Slice 3: Status And Discoverability

Update `ShapesAnnotation._refresh_save_shapes_state(...)` after a layer is
created.

Suggested status:

```text
Draw shapes in the viewer, then click Save shapes. While this layer is active,
Ctrl+Alt+Arrow nudges the camera; mouse-wheel zoom remains handled by napari.
```

Keep this concise so it does not dominate the widget.

Tests:

- layer-ready status includes the camera nudge shortcuts when a viewer with
  keybinding support is present;
- existing save-state warning tests still pass unchanged.

### Slice 4: Manual Napari QA

Manual test matrix:

- create a new annotation layer;
- select napari's polygon lasso tool;
- start drawing a lasso polygon;
- press each nudge shortcut;
- verify the camera moves and the layer remains in lasso mode;
- continue drawing and finish the polygon;
- save shapes through `ShapesAnnotation`;
- inspect the saved `GeoDataFrame` shape count and index values;
- repeat with mouse-wheel zoom during drawing;
- repeat after removing the annotation layer;
- repeat after selecting a non-annotation layer.

Record the observed geometry behavior after nudging. If large jumps are too
easy to create, keep the feature hidden behind a setting or reduce the default
pixel step.

## Future Native-Like Options

### Auto-Pan Near Canvas Edge

This is the most natural UX for large whole-slide annotations, but it needs more
canvas integration.

Potential implementation:

- listen to mouse move events while the annotation layer is in lasso mode;
- detect pointer distance to the canvas edge;
- use a `QTimer` to repeatedly nudge the camera while the cursor remains near an
  edge;
- scale speed by edge proximity;
- stop immediately when drawing finishes or the active layer changes.

Reasons to defer:

- needs reliable access to canvas dimensions and cursor position;
- higher risk of depending on private Qt/Vispy details;
- greater risk of runaway camera movement;
- harder to test headlessly.

### Bare Arrow-Key Or Space Pan

These are closer to upstream proposals but should stay upstream-first.

Implementing them inside napari-harpy would require either overriding napari's
global keybindings or intercepting Shapes layer mode changes. Bare arrows also
conflict with napari's dimension and layer navigation shortcuts. That would be
fragile because Shapes currently finishes active drawing when mode changes.

## Definition Of Done

- The roadmap's first implementation slice provides camera nudges without
  changing layer mode.
- The feature acts only on the widget-owned primary annotation layer.
- Existing shape creation and save workflows continue to pass.
- Unit tests cover nudge math, gating, missing-viewer fallbacks, and keybinding
  registration.
- Manual QA confirms that lasso mode remains active after nudging.
- The status text makes the feature discoverable without claiming it is native
  lasso-aware panning.

## References

- napari issue "Shift camera when drawing":
  https://github.com/napari/napari/issues/6885
- napari Shapes layer documentation:
  https://napari.org/dev/howtos/layers/shapes.html
- napari Camera API:
  https://napari.org/dev/api/napari.components.Camera.html
- napari Shapes source:
  https://napari.org/stable/_modules/napari/layers/shapes/shapes.html
