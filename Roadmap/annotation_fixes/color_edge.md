# Primary Shapes Face Color Sync

## Current Behavior

- Primary polygon `Shapes` styling is centralized in
  `src/napari_harpy/viewer/shapes_styling.py`.
- The current defaults are:
  - `PRIMARY_SHAPES_EDGE_COLOR = "#00FFFF"`
  - `PRIMARY_SHAPES_FACE_COLOR = "#00FFFF20"`
  - `PRIMARY_SHAPES_EDGE_WIDTH = 1`
  - `PRIMARY_SHAPES_OPACITY = 0.8`
- `apply_primary_shapes_layer_style(...)` applies both current draw defaults
  and existing-row styles:
  - `layer.current_edge_color`
  - `layer.current_face_color`
  - `layer.edge_color`
  - `layer.face_color`
  - `layer.edge_width`
  - `layer.opacity`
- Primary shape edge-color syncing already exists via
  `_connect_current_edge_color_to_global_edge_color(...)`.
  It listens to `layer.events.current_edge_color` and assigns
  `layer.edge_color = layer.current_edge_color`, so a color change in napari's
  Shapes UI is applied to all existing rows.
- There is no equivalent face-color sync for `Shapes`. Changing
  `layer.current_face_color` in napari updates the draw default for future
  shapes, but does not repaint existing shapes unless Harpy also assigns
  `layer.face_color`.

## Proposed Default Face Color

Change the primary polygon face color from fully transparent black to a cyan
with very low alpha, matching the primary edge RGB:

```python
PRIMARY_SHAPES_FACE_COLOR = "#00FFFF20"
```

`#20` is about 12.5% alpha. This keeps the fill light while making the face
color easier to notice and easier to edit from napari's color picker.

## Implementation Plan

1. [x] Add a shapes face-color sync callback in
   `src/napari_harpy/viewer/shapes_styling.py`.

   - Add a sibling callback attribute constant, for example:
     `"_harpy_shapes_face_color_sync_callback"`.
   - Add `_connect_current_face_color_to_global_face_color(layer: Shapes)`.
   - Follow the existing edge-color implementation:
     - return early if the callback is already installed;
     - define `_sync_current_face_color_to_all_shapes(...)`;
     - assign `layer.face_color = layer.current_face_color`;
     - connect to `layer.events.current_face_color`;
     - store the callback on the layer to prevent duplicate connections.
   - Implemented in `src/napari_harpy/viewer/shapes_styling.py`; not wired
     into primary styling yet, so user-facing behavior is unchanged until
     slice 2.

2. [x] Wire the face-color callback into primary styling.

   - Update `apply_primary_shapes_layer_style(...)` so it installs the new
     face-color sync callback for primary shapes.
   - Keep styled shapes protected from global UI color flattening. Today
     `_build_shapes_layer(..., sync_edge_color=False)` prevents styled edge
     palettes from being flattened.
   - Rename the parameter to `sync_current_colors` because it will now guard
     both edge and face color sync callbacks.
   - Connect both `_connect_current_edge_color_to_global_edge_color(layer)` and
     `_connect_current_face_color_to_global_face_color(layer)` only when
     `sync_current_colors` is true.
   - Update all call sites that currently pass `sync_edge_color`:
     - primary shapes loading should pass `sync_current_colors=True`;
     - styled shapes loading should pass `sync_current_colors=False`;
     - direct calls to `apply_primary_shapes_layer_style(...)` can rely on the
       default true value unless they intentionally need styled/data-driven
       behavior.
   - Implemented in `src/napari_harpy/viewer/shapes_styling.py` and
     `src/napari_harpy/viewer/adapter.py`.

3. [x] Update the default face color constant.

   - Change `PRIMARY_SHAPES_FACE_COLOR` to the chosen cyan-with-alpha value.
   - Because `apply_primary_shapes_layer_style(...)` is shared, this covers:
     - empty annotation-created primary layers;
     - primary shapes loaded through the Viewer widget;
     - existing primary shapes opened by the Annotation widget;
     - native napari Shapes layers adopted by the Annotation widget;
     - Harpy replacement layers created by
       `_build_harpy_shapes_layer_from_native_layer(...)`.
   - Implemented as `PRIMARY_SHAPES_FACE_COLOR = "#00FFFF20"`.

4. [x] Preserve existing annotation edit behavior.

   - `_capture_shapes_layer_style(...)`,
     `_restore_shapes_layer_current_style(...)`, and
     `_restore_shapes_layer_row_styles(...)` already snapshot and restore
     `current_face_color` and `face_color`.
   - Their event blockers are important: when restoring current colors during
     hole/vertex edits, the new face-color sync callback must not overwrite the
     restored per-row `face_color`.
   - No structural change is expected in
     `src/napari_harpy/widgets/shapes_annotation/_layer_style.py`.
   - Existing snapshot/restore helpers already preserve `current_face_color`
     and per-row `face_color`, including event blockers around current color
     restore, so no code change was needed.

5. [x] Add or update tests.

   - Update existing expectations from `"#00000000"` to the chosen
     `PRIMARY_SHAPES_FACE_COLOR` value in:
     - annotation layer creation tests;
     - native layer adoption tests;
     - native non-empty save/import tests;
     - any direct style-preservation tests that intentionally assert the old
       default row face color.
   - Add a focused test for `apply_primary_shapes_layer_style(...)`:
     - create a `Shapes` layer with one or more polygons;
     - call `apply_primary_shapes_layer_style(layer)`;
     - set `layer.current_face_color` to a different color;
     - assert every row in `layer.face_color` changes to that color;
     - assert the callback attribute is installed.
   - Add or update an annotation-widget-level test:
     - create a primary annotation layer;
     - set `layer.current_face_color` as napari would from the UI;
     - assert `layer.face_color` updates for existing shapes.
   - Add a styled-shapes regression if not already covered:
     - load/apply styled shapes with `sync_edge_color=False`;
     - change `current_face_color`;
     - assert the data-driven `face_color` array is not flattened.
   - Implemented primary default and sync assertions in
     `tests/test_viewer_adapter.py` and primary annotation default assertions
     in `tests/test_shapes_annotation_widget.py`; existing styled-shapes
     regressions cover data-driven color preservation.

6. Verification commands.

```bash
.venv/bin/pytest tests/test_shapes_styling.py tests/test_shapes_annotation_widget.py
.venv/bin/pre-commit run ruff --all-files
```

## Notes

- Local napari behavior confirms that `Shapes.events.current_face_color`
  exists and emits when `current_face_color` changes.
- Local napari behavior also confirms that changing `current_face_color` alone
  does not repaint existing `face_color` rows, so Harpy needs an explicit sync
  callback just like it already has for edge color.
