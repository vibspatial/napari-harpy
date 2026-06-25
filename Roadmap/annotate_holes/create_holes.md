# Create Holes From Selected Polygons

Source: follow-up roadmap for Slice 2 from
[`annotation_widget_holes.md`](./annotation_widget_holes.md).

Status: specification.

Goal: let users create polygon holes from ordinary napari polygon rows without
introducing a complex boolean-geometry UI. The first implementation should be a
strict "Create holes" action, not a general-purpose subtraction tool.

The action is driven by napari's persistent selected shape rows
(`layer.selected_data`). It is not driven by selected vertices.

Users may prefer napari's `Select vertices` / direct mode because selected
polygon rows show their vertices there, making the intended shell and hole
polygons easier to inspect. That is only a visual/interaction affordance. The
button still reads selected polygon rows from `layer.selected_data`.

## Decision

Do not implement full `master_polygon.difference(union(children))` for the
first hole-creation slice.

Instead, implement a constrained operation:

1. The user selects one shell polygon row and one or more polygon rows fully
   inside it, using napari's native shape-row selection.
2. The widget infers the unique selected shell: the selected polygon that can
   validly contain all other selected polygons as direct holes.
3. The child polygons are converted into interior rings of the shell polygon.
4. The child polygon rows are removed from the napari layer.
5. The shell row keeps its feature values and source identity.

This creates a Shapely `Polygon(shell, holes=[...])` explicitly. It avoids the
harder boolean-subtraction cases where cutters cross the shell, split the
shell, produce a `MultiPolygon`, create geometry collections, or imply
hole-inside-hole topology.

The widget action can still be QuPath-inspired, but the user-facing label should
be clearer than full boolean subtraction. Prefer `Create holes` or
`Create holes from selected` over `Subtract selected`.

Do not support a stateful "click a shell vertex, click hole vertices, then
create holes" workflow in this slice. In napari Shapes, vertex hits in direct
mode are transient edit targets, while `layer.selected_data` is the stable
selection state we can read from the widget.

## Current-Code Constraints

The current annotation model is a good fit for this constrained operation:

- One editable annotation row is one napari Shapes row.
- One saved annotation row is one Shapely `Polygon`.
- Existing edit sessions reject `MultiPolygon` source geometries.
- Hole-bearing polygons are already encoded into one napari polygon row with
  repeated shell-anchor separators.
- The save path already decodes a hole-bearing napari row into one Shapely
  `Polygon`.
- Row identity lives in `layer.features`, especially the source index feature
  column used by edit-existing sessions.

The main implementation risk is therefore not Shapely. It is safe napari layer
mutation: replace one selected row with a longer hole-encoded row, remove the
consumed child rows, preserve the shell row's features, and keep napari's
internal shape/vertex caches consistent.

## User Flow

1. Open or create an annotation layer through the Shapes Annotation widget.
2. Draw one intended shell polygon.
3. Draw one or more intended hole polygons fully inside the shell polygon.
4. Select the shell polygon row and all intended hole polygon rows in napari.
   Users may use napari's shape selection affordances, such as shift-clicking
   rows/shapes. Direct/vertex mode may make vertices more visible, but the
   operation still consumes selected polygon rows, not selected vertices.
5. Click `Create holes`.
6. The widget replaces the selected rows with one hole-bearing polygon row.
7. The shell row remains selected.
8. Saving persists one Shapely `Polygon` with interior rings.

No extra modal dialog, "set shell" state, feature-column editing, custom drawing
mode, or vertex-picking state machine is needed for the first implementation.

## Selection UX Contract

`Create holes` is a one-off action:

1. The user prepares a napari Shapes selection.
2. The user clicks `Create holes`.
3. napari-harpy validates the current selection.
4. If the selection describes exactly one shell row and one or more valid hole
   rows, napari-harpy mutates the layer.
5. Otherwise, napari-harpy leaves the layer unchanged and reports the problem
   through the status card.

The action should not start, continue, or finish a custom stateful selection
workflow. In particular, do not implement:

- click a shell vertex, then click hole vertices, then create holes
- click a shell row, then click hole rows while the widget records state
- a separate "set shell" interaction
- a custom vertex-picking mode

The only UI state consumed by the button is the current selected shape-row set:

```python
selected_rows = set(layer.selected_data)
```

Those values are current napari row indices into `layer.data`,
`layer.shape_type`, and row-aligned `layer.features`. They are not vertex
indices, and they are not the source GeoDataFrame index values stored in
annotation metadata.

Napari direct mode can still be useful. In direct mode, selected shape rows show
their raw vertices, so users can visually verify that the shell and candidate
holes are the intended rows. But napari does not expose a durable "selected
vertices" collection for Shapes. Vertex hits in direct mode are transient edit
targets such as `layer._moving_value = (row_index, vertex_index)`, and should
not be used by `Create holes`.

Practical implication:

- use napari's native row/shape selection affordances to select the shell row
  and hole rows
- users may use whichever napari mode gives useful visual feedback, including
  direct/vertex mode if it helps them see which rows are selected
- `Create holes` infers shell and holes from selected rows only

## Geometry Contract

Supported input:

- selected shape rows must have `shape_type == "polygon"`
- at least two shape rows must be selected
- exactly one selected polygon row must have the largest Shapely area; this
  unique largest-area row is the shell candidate
- every other selected polygon row must be usable as a direct hole
- shell rows may already contain holes; preserve those existing holes and
  append the new direct holes only if the combined topology remains valid
- selected child rows with existing holes are rejected; using a
  polygon-with-hole as a hole cutter would imply an island-in-hole result,
  which is outside the current topology contract

Rejected input:

- no unique largest-area selected polygon row
- unique largest-area selected polygon row cannot contain every other selected
  row as a direct hole
- selected line/path/rectangle/ellipse rows
- candidate holes outside the shell
- candidate holes crossing or touching the shell boundary
- candidate holes inside an existing shell hole
- nested holes / islands-in-holes
- overlapping holes
- edge-sharing holes
- child polygons that already have interiors
- attempts to drive the operation from clicked/selected vertices rather than
  selected shape rows
- any operation that would require `MultiPolygon` annotation support

Important nuance: the operation should not call Shapely `difference(...)` as
the core implementation. It should construct:

```python
Polygon(shell.exterior.coords, holes=[existing_holes, child_exteriors, ...])
```

and then run the same direct-hole validation rules used by the save decoder.
Existing holes on the selected shell are therefore first-class geometry that
survives the operation; selected child rows only add more direct holes.

## Slice 2A - Pure Geometry Helper

Status: implemented.

Add a public core helper that constructs a hole-bearing Shapely polygon from a
shell polygon and direct child-hole polygons.

Suggested API:

```python
def create_polygon_with_direct_holes(
    shell: Polygon,
    holes: Sequence[Polygon],
) -> Polygon:
    """Return ``shell`` with the child polygons added as direct holes.

    ``shell`` is the Shapely polygon that survives the operation. If it already
    has holes, those existing interiors are read from ``shell.interiors`` and
    preserved.

    ``holes`` contains only the new child polygons selected to become
    additional holes. The child polygons must be simple polygons without their
    own interiors; each child exterior ring is appended after any existing
    ``shell.interiors``.
    """
    ...
```

Responsibilities:

- Validate that `shell` is a non-empty valid Shapely `Polygon`.
- Preserve any existing `shell.interiors` as existing holes.
- Reject child polygons that already have interiors; selected children must be
  simple polygons that can become direct holes.
- Use each child polygon's exterior ring as one new hole.
- Validate the combined existing and new holes with the same strict topology
  rules used by `napari_polygon_vertices_to_shapely_polygon(...)`.
- When the shell already has interiors, validate the combined holes against an
  exterior-only shell polygon, not against the original hole-bearing shell.
  Shapely treats existing holes as non-contained empty regions, so containment
  checks against the original shell would incorrectly reject preserved holes.
- Construct and return one Shapely `Polygon`.
- Return a polygon that can be encoded by
  `shapely_polygon_to_napari_polygon_vertices(...)`.

Implementation notes:

- This helper belongs in `src/napari_harpy/core/shapes_geometry.py`, close to
  the existing encoder/decoder.
- The current `_validate_direct_holes(...)` is private, but the new helper can
  reuse it internally.
- The validation shell should be:

  ```python
  exterior_shell = Polygon(shell.exterior.coords)
  ```

  Then validate:

  ```python
  all_holes = existing_shell_holes + new_child_holes
  _validate_direct_holes(exterior_shell, all_holes)
  ```

  Finally construct the output as:

  ```python
  Polygon(shell.exterior.coords, holes=all_hole_rings)
  ```

- The helper should not know anything about napari layers, features, selected
  rows, or widget state.

Unit tests:

- simple shell plus one child returns a polygon with one interior
- simple shell plus two children returns a polygon with two interiors
- shell with an existing hole preserves that hole and appends a new direct hole
- shell with an existing hole does not fail containment validation merely
  because the existing hole is not positive area inside the original
  hole-bearing shell
- child polygon with an interior is rejected
- child outside the shell is rejected
- child crossing or touching the shell is rejected
- nested child holes are rejected
- overlapping child holes are rejected
- edge-sharing child holes are rejected

Acceptance criteria:

- The helper creates valid one-level Shapely polygons with interiors.
- The helper does not introduce `MultiPolygon`.
- The helper enforces the same no-nested-hole contract as the existing save
  decoder.

## Slice 2B - Selection Planning Helper

Status: implemented.

Add a UI-independent helper that inspects a napari Shapes layer selection and
produces a create-holes plan without mutating the layer.

Selection here means napari shape-row selection, i.e. `layer.selected_data`.
The helper should not inspect transient vertex-edit state such as
`layer._moving_value`.

Suggested internal API:

```python
@dataclass(frozen=True)
class _CreateHolesShapesLayerPlan:
    """Planned create-holes mutation for one selected napari Shapes layer.

    Attributes
    ----------
    shell_row_index
        Current napari row in ``layer.data`` that survives the operation and
        receives the new hole-bearing vertices. This is a napari layer row
        index, not a source GeoDataFrame index value.
    hole_row_indices
        Current napari rows selected as child polygons. Their exterior rings
        become new holes, and the rows are removed after the shell row is
        replaced.
    vertices
        Napari ``(y, x)`` vertex row encoding of the new hole-bearing shell
        polygon. This array is the replacement data for
        ``layer.data[shell_row_index]``.
    """

    shell_row_index: int
    hole_row_indices: tuple[int, ...]
    vertices: np.ndarray


def _create_holes_plan_from_selection(layer: Shapes) -> _CreateHolesShapesLayerPlan:
    ...
```

Responsibilities:

- Read `layer.selected_data`.
- Treat `layer.selected_data` as the only semantic input from the napari UI.
- Interpret selected values as current napari row indices into `layer.data`,
  not as source GeoDataFrame indices.
- Normalize selected rows deterministically, for example:

  ```python
  selected_rows = tuple(sorted(int(index) for index in layer.selected_data))
  ```

  Sorting is only for deterministic planning and tests; it must not give
  selection order any semantic meaning.
- Validate that every selected row index is an integer index currently present
  in `layer.data`.
- Require at least two selected shape rows.
- Require every selected shape row to be a polygon row.
- Decode every selected row with
  `napari_polygon_vertices_to_shapely_polygon(...)`.
- Identify the unique largest-area selected polygon row using Shapely
  `polygon.area`; this row is the shell candidate.
- Fail clearly if the largest area is tied. Use exact area equality for the
  first implementation; if two selected rows share the maximum area, the shell
  is ambiguous.
- Treat every other selected polygon row as a proposed new hole. Return
  `hole_row_indices` sorted by current napari row index so later mutation is
  predictable.
- Build the output polygon with `create_polygon_with_direct_holes(...)`.
- Encode the output polygon with
  `shapely_polygon_to_napari_polygon_vertices(...)`.
- Keep the Shapely output polygon as a local planning intermediate; do not
  store it on the plan after it has been encoded.
- Return a plan containing row indices and replacement vertices.
- Do not mutate the layer if planning fails.

Shell inference rule:

- The shell candidate is the unique selected row whose decoded Shapely polygon
  has the largest area.
- Selection order is not semantic input. Do not use "first selected", "last
  selected", napari `Selection.active`, or private current-selection state.
- If the largest area is exactly tied, fail without mutation and show a clear
  ambiguity warning.
- If the unique largest-area row cannot contain every other selected row as a
  direct hole under the strict helper from Slice 2A, fail without mutation and
  show a clear warning such as "Select one shell polygon and one or more
  polygons fully inside it."

Implementation notes:

- This helper lives in
  `src/napari_harpy/widgets/shapes_annotation/_create_holes.py`, keeping the
  planning logic close to the annotation widget while avoiding more bulk in
  `widget.py`.
- Do not use selected-row ordering as semantic input; napari selection ordering
  should not be treated as a stable public contract.
- Do not support "first clicked shell vertex" or "clicked hole vertices" as an
  input mechanism. That would require a separate custom interaction mode.
- Planning failures should raise `ValueError` with user-facing messages. Slice
  2D should catch those errors and route them into the existing status card.

Unit tests:

- selected shell plus one child produces a plan
- selected shell plus two children produces a plan
- selected shell that already has holes produces a plan that preserves existing
  holes and appends new direct holes when the combined topology is valid
- selection order does not matter
- unique largest-area selected row is used as the shell candidate
- tied largest-area selected rows fail without mutation
- no selected rows fails without mutation
- one selected row fails without mutation
- selected row index outside `layer.data` fails without mutation
- non-polygon selected row fails without mutation
- unique largest-area row that cannot contain the other selected rows fails
  without mutation
- selected child with existing holes fails without mutation

Acceptance criteria:

- Planning is deterministic.
- Planning has no layer side effects.
- Invalid selections fail before any row is changed or removed.

## Slice 2C - Layer Row Replacement And Child Removal

Status: implemented.

Add a layer mutation helper that applies a successful create-holes plan.

Suggested internal API:

```python
def _apply_create_holes_plan(layer: Shapes, plan: _CreateHolesShapesLayerPlan) -> None:
    ...
```

Responsibilities:

- Validate that the plan still refers to the current napari layer state before
  mutating:

  - `plan.shell_row_index` is present in `layer.data`
  - `plan.hole_row_indices` are unique
  - no hole row equals `plan.shell_row_index`
  - every hole row is present in `layer.data`

- Replace the shell row's vertices with the encoded hole-bearing vertices.
- Remove the consumed child rows.
- Preserve the shell row's feature values and source identity.
- Remove the child rows' feature values.
- Preserve unselected rows.
- Restore selection to the resulting shell row.
- Keep the current napari layer mode when possible.
- Rebuild napari shape/vertex caches safely.

Recommended mutation strategy:

1. Validate plan indices against the current `layer.data` length.
2. Store current mode.
3. Replace the shell row by assigning a same-length rebuilt `layer.data` list:

   ```python
   rebuilt_data = list(layer.data)
   rebuilt_data[plan.shell_row_index] = plan.vertices
   # Assign through `layer.data`, not `_data_view.edit(...)`: create-holes can
   # change the shell row's vertex count, and the public setter rebuilds
   # napari's `ShapeList` bookkeeping used for rendering and hit-testing.
   layer.data = rebuilt_data
   ```

   This rebuilds napari's `ShapeList` after the shell row changes length. In
   earlier anchor-deletion work, low-level `_data_view.edit(...)` could update
   row data while leaving old clickable vertex indices in napari's internal
   hit-test cache after a row shortened. Create-holes is also a topology
   change, so it should use the same public-setter cache rebuild strategy.
4. Remove child rows with napari's public `layer.remove(...)`, sorted by row
   index:

   ```python
   layer.remove(sorted(plan.hole_row_indices))
   ```

5. Compute the shell row's new index after removing child rows that were before
   it:

   ```python
   new_shell_row_index = plan.shell_row_index - sum(
       row_index < plan.shell_row_index
       for row_index in plan.hole_row_indices
   )
   ```

6. Restore `layer.mode`.
7. Set `layer.selected_data = {new_shell_row_index}`.
8. Refresh the layer.

Rationale:

- The previous anchor-deletion work showed that low-level
  `_data_view.edit(...)` can leave napari vertex hit-test caches stale after
  row-shortening edits.
- `layer.data = ...` rebuilds the `ShapeList`.
- `layer.remove(...)` removes arbitrary rows while keeping napari's feature and
  style arrays row-aligned.
- Doing replacement before removal avoids having to remap the original shell
  row before its vertices are updated.

Implementation notes:

- This helper should live in
  `src/napari_harpy/widgets/shapes_annotation/_create_holes.py` next to the
  planning helper.
- Do not manually edit `layer.features` unless the public napari row-removal
  path proves insufficient.
- Do not use `_data_view.edit(...)` for shell replacement; the replacement row
  may have a different number of vertices.
- Do not call the save converter from this helper.
- Do not show status cards or touch `SpatialData`; the widget click handler
  will handle user feedback in a later step.
- The helper should be small enough that widget click handling remains easy to
  read.

Unit tests:

- shell row features are preserved
- child row features are removed
- unselected row features are preserved
- shell row remains selected after child rows before it are removed
- shell row remains selected after child rows after it are removed
- layer shape types remain aligned with `layer.data`
- arbitrary row order is handled correctly
- invalid plan indices fail before mutation

Acceptance criteria:

- After the helper runs, `len(layer.data)` decreases by the number of consumed
  child rows.
- The resulting shell row decodes to a Shapely polygon with the expected holes.
- `layer.features` remains row-aligned.
- The layer can be saved by the existing save path.

## Slice 2D - Widget Button And Status Integration

Status: implemented.

Add the user-facing widget action.

Responsibilities:

- Add a compact `Create holes` button near the existing `Create layer` and
  `Save shapes` controls.
- Connect the button to a new click handler.
- Run `_create_holes_plan_from_selection(self._annotation_layer)`.
- Apply the plan with `_apply_create_holes_plan(...)` if planning succeeds.
- Report invalid selection or geometry failures through the existing status
  card, without mutating the layer.
- Report success through the existing status card after the layer mutation.
- Refresh save readiness without overwriting the success status.
- Do not update `_annotation_clean_snapshot`; creating holes is an unsaved
  annotation edit until the user clicks `Save shapes`.
  Updating the clean snapshot here would incorrectly suppress discard warnings
  even though the changed layer has not yet been written back to `SpatialData`.

Suggested handler flow:

```python
def _on_create_holes_clicked(self) -> None:
    layer = self._annotation_layer
    session = self._annotation_session
    if layer is None or session is None:
        return

    try:
        plan = _create_holes_plan_from_selection(layer)
        hole_count = len(plan.hole_row_indices)
        _apply_create_holes_plan(layer, plan)
    except ValueError as error:
        self._set_status(title="Could Not Create Holes", lines=[str(error)], kind="warning")
        return

    self._refresh_save_shapes_state(update_status=False)
    lines = [
        f"Converted {hole_count} selected polygon(s) into hole(s) and removed their shape row(s)."
    ]
    if session.table_linked:
        lines.append(
            "Linked tables are not updated automatically; after saving, table annotations may no longer match the shapes rows."
        )
    self._set_status(title="Created Holes", lines=lines, kind="success")
```

Button enabled state:

- Enable only when an annotation layer is open and the same save
  binding/session checks used by `Save shapes` pass.
- Factor those shared checks into a small private helper, for example:

  ```python
  def _annotation_layer_is_actionable(self, *, update_status: bool = True) -> bool:
      ...
  ```

  `_refresh_save_shapes_state(...)` can use this helper to enable
  `save_shapes_button`, and the create-holes state refresh can use the same
  helper with `update_status=False`.
- Do not infer create-holes readiness from
  `self.save_shapes_button.isEnabled()`. The save button's visual enabled state
  should not become hidden business logic for a different action.
- Do not rely on live selection-change events for the first implementation, and
  do not disable the button based on the current row-selection count.
  napari's `selected_data` is validated when the user clicks `Create holes`.
- If the selected shape rows are invalid for creating holes, keep the layer
  unchanged and show a warning.
- Do not try to infer intent from the last clicked vertex or active vertex-edit
  target.

Success message:

- Include how many child polygons were converted into holes, and state that
  those child shape rows were removed from the annotation layer.
- If the session is table-linked, warn explicitly that this row-removing
  operation is not propagated to linked tables, so table annotations may no
  longer match the shapes rows after save.

Scope:

- The button acts on `self._annotation_layer`, not whichever layer is active in
  napari, until the active-layer synchronization roadmap is implemented.
- The operation consumes selected napari shape rows from
  `self._annotation_layer.selected_data`; selected vertices are ignored.
- Native layers adopted by the widget should work because they become the
  annotation layer and already use the same save path.
- No custom modal, custom drawing mode, or stateful "choose shell, then choose
  holes" workflow is introduced.

Widget tests:

- button is disabled when no annotation layer is open
- invalid selection shows `Could Not Create Holes`
- valid selection mutates the layer and shows success
- table-linked session includes the linked-table warning
- save button remains enabled after a successful operation
- dirty-state detection sees the changed layer

Acceptance criteria:

- Users can create holes without a modal dialog or custom drawing mode.
- Invalid selections fail clearly and do not mutate the layer.
- The operation is treated like any other unsaved annotation edit.

## Slice 2D - refactor

Status: implemented.

Make the widget button/readiness/status code easier to read before adding more
create-holes behavior.

Why this deserves a follow-up:

- The current code uses `_refresh_save_shapes_state(update_status=False)` to
  mean "update the Save shapes and Create holes button enabled state, but do
  not overwrite the operation-specific status card."
- That is hard to reason about because `_refresh_save_shapes_state(...)` calls
  `_annotation_layer_is_actionable(...)`, and `_annotation_layer_is_actionable`
  sounds like a pure boolean predicate but can also write status-card messages.
- Callers therefore need to understand hidden side effects and pass
  `update_status=False` at the right moments to avoid clobbering messages such
  as `Created Holes` or `Could Not Create Holes`.
- `_on_save_shapes_clicked(...)` also checks
  `self.save_shapes_button.isEnabled()` after refreshing readiness. That makes
  a UI property act as hidden business logic.

Recommended direction:

- Introduce a small private readiness object, for example:

  ```python
  @dataclass(frozen=True)
  class _AnnotationLayerReadiness:
      actionable: bool
      status: _ShapesAnnotationStatusCardSpec | None = None
  ```

  The `status` is a readiness status-card spec, not an operation-result status.
  It should come from the proposed `widgets/shapes_annotation/status_card.py`
  builders.

- Replace `_annotation_layer_is_actionable(update_status=...)` with a pure
  evaluator, for example:

  ```python
  def _evaluate_annotation_layer_readiness(self) -> _AnnotationLayerReadiness:
      ...
  ```

  This method should inspect the current annotation layer, app state, binding,
  and session, but should not mutate the UI.

- Add explicit UI application helpers:

  ```python
  def _apply_annotation_action_buttons(self, readiness: _AnnotationLayerReadiness) -> None:
      self.save_shapes_button.setEnabled(readiness.actionable)
      self.create_holes_button.setEnabled(readiness.actionable)

  def _apply_status_card_spec(self, spec: _ShapesAnnotationStatusCardSpec | None) -> None:
      ...
  ```

- Keep operation-specific handlers explicit. For example, `Create holes` should:
  - evaluate readiness
  - apply button state
  - show `readiness.status` only if not actionable
  - otherwise run create-holes
  - after success, evaluate readiness again and apply button state without
    showing the readiness status
  - build an operation-result status with a status-card helper, for example:

    ```python
    created_holes_status = build_create_holes_success_card_spec(
        hole_count=hole_count,
        table_linked=session.table_linked,
    )
    self._apply_status_card_spec(created_holes_status)
    ```

  This keeps the click handler focused on workflow while
  `status_card.py` owns wording such as `Created Holes`.

- `Save shapes` should similarly use the readiness result directly instead of
  checking `self.save_shapes_button.isEnabled()` as the source of truth.

Design notes:

- This refactor should not change user-visible behavior.
- It should remove or greatly reduce the need for `update_status=False`.
- It should make the distinction between "readiness status" and
  "operation-result status" explicit.
- Keep this scoped to the annotation widget readiness/action buttons. Do not
  fold in style preservation or active-layer synchronization.

Acceptance criteria:

- Readiness evaluation is side-effect free.
- Button state updates are handled by an explicit helper.
- Status-card writes happen only where the caller intentionally requests them.
- `Create holes` and `Save shapes` no longer infer business readiness from a
  button's enabled state.
- Existing widget tests continue to pass with no behavior changes.

## Slice 2E - Reject Create Holes During Napari Draw/Edit Transient State

Status: deferred.

Prevent create-holes from mutating a Shapes layer while napari is still inside
an in-progress draw/add interaction.

Deferred rationale:

- A reported traceback showed napari failing inside
  `add_path_polygon_lasso(...)` on mouse release because
  `layer._moving_value[0]` was already `None`.
- Napari uses `_moving_value`, `_is_creating`, and the current add/draw mode to
  finish shape creation. Mutating `layer.data` during that transient state can
  call `_finish_drawing()`, reset `_moving_value`, and leave napari's mouse
  callback with stale state.
- However, blocking create-holes, coordinate-system changes, or annotation-target
  changes based on napari draw/add modes gives a poor user experience: users can
  end up blocked without a clear place to click or action to take.
- The recurring style symptom is better explained by row/style array mismatch
  during the create-holes mutation itself. If napari's internal style arrays
  contain stale extra rows, `layer.data = rebuilt_data` can trigger napari's
  color fallback path and reset edge/face colors to defaults.
- Defer this guard unless we later isolate a separate, reproducible mouse-draw
  crash that cannot be solved by explicit style preservation and layer mutation
  hygiene.

Previously considered guard:

- Before planning or applying create-holes, reject clearly if the annotation
  layer is in an in-progress creation state:

  ```python
  if getattr(layer, "_is_creating", False):
      raise ValueError("Finish the current shape drawing before creating holes.")
  ```

- Also reject when the layer mode is one of napari's draw/add modes, because
  those modes indicate the user is preparing or completing shape creation:
  - `Mode.ADD_PATH`
  - `Mode.ADD_POLYGON`
  - `Mode.ADD_POLYGON_LASSO`
  - `Mode.ADD_RECTANGLE`
  - `Mode.ADD_ELLIPSE`
  - `Mode.ADD_LINE`
  - `Mode.ADD_POLYLINE`
- Keep `Mode.SELECT`, `Mode.DIRECT`, `Mode.VERTEX_INSERT`, and
  `Mode.VERTEX_REMOVE` eligible. The operation still validates selected shape
  rows on click.
- Surface the rejection through the existing `Could Not Create Holes` status
  card and leave the layer unchanged.

Implementation notes:

- Prefer a small helper close to the widget click handler or create-holes
  helper, for example `_validate_create_holes_layer_is_stable(layer)`.
- This helper should run before `_create_holes_plan_from_selection(...)`, so no
  geometry parsing or row mutation happens while napari is in a transient draw
  state.
- The guard is separate from active-layer synchronization. It prevents an
  unsafe operation on the widget-owned annotation layer, but it does not yet
  solve napari/widget active-layer divergence.

Regression tests:

- `Create holes` with `layer._is_creating = True` shows a warning and does not
  mutate rows/features.
- `Create holes` in each add/draw mode shows a warning and does not mutate
  rows/features.
- `Create holes` still works in an eligible mode such as `Mode.SELECT` or
  `Mode.DIRECT`.

Acceptance criteria:

- Create-holes cannot run while napari is midway through drawing/adding a
  shape.
- The reported `add_path_polygon_lasso(...)` `_moving_value[0] is None` failure
  is not reachable from the create-holes button path.
- Geometry/topology behavior from Slice 2D remains unchanged in stable modes.

## Slice 2F - Preserve Annotation Styling After Create Holes

Status: proposed; next priority.

Fix the visual styling regression observed after using the `Create holes`
button: the resulting hole-bearing polygon can render like a default filled
napari polygon instead of keeping Harpy's annotation style.

Observed behavior:

- The create-holes operation succeeds geometrically.
- The resulting row contains the expected hole topology.
- The rendered polygon can lose the expected annotation appearance, showing a
  filled gray/white face instead of Harpy's transparent-face, cyan-edge style.
- The issue is intermittent because it depends on napari's current internal
  row/style state. In a clean layer, create-holes can preserve styling; if
  napari's style arrays contain stale extra rows, napari can fall back to
  default colors.

Relevant current code path:

- Harpy's primary annotation style is defined in
  `src/napari_harpy/viewer/shapes_styling.py`:

  ```python
  PRIMARY_SHAPES_EDGE_COLOR = "#00FFFF"
  PRIMARY_SHAPES_FACE_COLOR = "#00000000"
  PRIMARY_SHAPES_EDGE_WIDTH = 1
  PRIMARY_SHAPES_OPACITY = 0.8
  ```

- `apply_primary_shapes_layer_style(...)` applies those defaults when an
  annotation layer is created, loaded, or adopted.
- `_apply_create_holes_plan(...)` then mutates the layer by assigning
  `layer.data = rebuilt_data`, removing child rows with `layer.remove(...)`,
  and selecting the resulting shell row.
- That geometry mutation rebuilds napari's internal `ShapeList` and removes
  rows. Even if model-level style arrays often remain correct, the operation
  should explicitly preserve/re-emit style state so the vispy layer cannot fall
  back to default filled-polygon rendering.
- Napari's `layer.data` setter reads existing `edge_width`, `edge_color`,
  `face_color`, and `z_index` state before rebuilding its `ShapeList`. If the
  color arrays have a length that does not match the number of logical shape
  rows, napari warns and normalizes colors to its default fallback. This matches
  warnings such as:

  ```text
  The provided edge_color parameter has 5 entries, while the data contains 4 entries.
  Setting edge_color to white.
  ```

Root-cause hypothesis:

- `Create holes` should not rely on napari's implicit style carry-over through
  `layer.data = rebuilt_data`.
- Before replacing/removing rows, Harpy should snapshot the style state that
  belongs to the current logical layer rows and ignore any stale extra style
  entries.
- After the geometry mutation, Harpy should explicitly restore the style state
  for the surviving rows.

Recommended fix:

- Capture row-aligned style state before mutation, sliced to the current
  logical row count `len(layer.data)`:
  - `edge_color`
  - `face_color`
  - `edge_width`
  - `z_index`
  - `opacity`
  - `current_edge_color`
  - `current_face_color`
  - `current_edge_width`
- Compute the pre-mutation row indices that survive:
  - the shell row survives and keeps the shell row's original style
  - selected child rows become holes and their style entries are removed
  - unselected rows survive and keep their original styles
  - final row order follows napari's remove semantics, so each final style row
    should be built from the corresponding surviving pre-mutation row index
- Before assigning `layer.data = rebuilt_data`, normalize the current layer's
  row style arrays to the current logical row count using the sliced snapshot.
  This prevents napari's `layer.data` setter from seeing stale extra style rows
  and falling back to defaults during the rebuild.
- Apply the create-holes geometry mutation.
- Restore surviving row styles through public napari setters after child rows
  have been removed, so napari emits the relevant color/width events.
- Restore layer-level/current annotation colors, width, opacity, and row
  `z_index` while preserving user customization. Then apply final row-aligned
  `edge_color`, `face_color`, and `edge_width` last, so Harpy's current-color
  synchronization callbacks cannot overwrite the final row styles.
- Keep the resulting shell row selected after style restoration.

Design notes:

- Prefer preserving the existing layer style over blindly calling
  `apply_primary_shapes_layer_style(...)`. Users may have changed the
  annotation edge color or edge width in napari; create-holes should not reset
  those choices to cyan/width 1 unless the layer was already using them.
- The helper should preserve styles for all surviving rows, not just the shell
  row.
- The consumed child rows' style entries should disappear along with the rows.
- If direct style restoration through public setters proves to change selected
  rows unintentionally, temporarily clear or block selection while restoring
  style, then restore `selected_data = {new_shell_row_index}`.
- Be careful with Harpy's existing current-edge-color synchronization callback:
  setting `current_edge_color` can update all row edge colors. The implementation
  should restore current colors/width and row colors in an order that leaves the
  final row-aligned arrays exactly as captured for the surviving rows.
- `current_edge_width` has the same callback pattern through napari's
  `edge_width` event, so edge-width restoration needs the same ordering care.
- Do not solve this by blindly calling `apply_primary_shapes_layer_style(...)`
  after create-holes; that would erase user-customized annotation styling.

Regression tests:

- A create-holes operation on a Harpy-styled annotation layer preserves:
  - transparent face color for the resulting shell row
  - annotation edge color for the resulting shell row
  - edge width
  - z index
  - opacity
- A layer with user-customized annotation edge color/width keeps those custom
  values after create-holes.
- Unselected surviving rows keep their row styles and z index.
- A regression test should deliberately simulate stale napari internal style
  arrays before create-holes by appending an extra row to
  `layer._data_view._edge_color` and `layer._data_view._face_color`. The
  operation should:
  - not emit napari's color fallback warning
  - preserve cyan/transparent Harpy styling for the surviving shell row
  - preserve custom styles for unselected surviving rows
  - remove the consumed child rows' style entries
- The geometry assertions from Slice 2D should remain in place: the shell row
  contains the new holes and child rows no longer exist as positive polygons.

Acceptance criteria:

- After clicking `Create holes`, the result renders with the same annotation
  styling as before the operation.
- Napari color fallback warnings caused by stale row/style array lengths are not
  emitted by the create-holes path.
- Geometry/topology behavior from Slice 2D remains unchanged.
- Style preservation is covered by focused tests.

## Slice 2G - End-To-End Save And Reload

Status: proposed.

Add focused integration coverage that proves the created holes persist through
the existing annotation save path.

Integration scenarios:

1. Create-new annotation layer:
   - draw/select shell plus child polygon rows
   - click `Create holes`
   - save
   - assert the SpatialData shapes element has one row with one or more
     interiors

2. Edit-existing annotation layer:
   - start from a SpatialData shapes element with multiple simple polygon rows
   - open it through the annotation widget
   - select one shell row and one child row
   - click `Create holes`
   - save
   - assert the saved element has one fewer row
   - assert the shell row source index and non-geometry metadata are preserved
   - assert the consumed child source row is gone

3. Adopted native napari layer:
   - create or import a native Shapes layer with selected polygon rows
   - let the annotation widget adopt it
   - click `Create holes`
   - save
   - assert the saved SpatialData element contains a polygon with interiors

Keep the test set focused. The pure helper tests should cover the geometry
matrix; widget integration only needs to prove that layer mutation, features,
save, and reload work together.

Acceptance criteria:

- Created holes save as Shapely `Polygon` interiors.
- The saved result reloads into napari as one hole-bearing polygon row.
- Shell metadata survives.
- Consumed child rows do not survive as positive annotation rows.

## Slice 2H - Shapes Annotation Status Card Helper

Status: proposed.

Introduce a dedicated `widgets/shapes_annotation/status_card.py` helper module
so annotation status-card text follows the same structure as the viewer, object
classification, and feature extraction widgets.

Why this helps:

- The annotation widget currently builds many status cards inline in
  `widget.py`: readiness messages, save success/failure, create-holes
  success/failure, target validation, and edit-guard warnings.
- Other widgets already centralize this pattern in a local `status_card.py`
  module with a small status spec dataclass and builder functions.
- Moving annotation status-card construction into a helper makes the proposed
  Slice 2D refactor cleaner: readiness evaluation can return a status-card spec
  without directly mutating the widget UI.
- It also keeps future create-holes status messages, such as style-preservation
  warnings or linked-table warnings, out of the button-handler control flow.

Recommended API shape:

```python
@dataclass(frozen=True)
class _ShapesAnnotationStatusCardSpec:
    title: str
    lines: tuple[str, ...]
    kind: StatusCardKind
    tooltip_message: str | None = None

    def __post_init__(self) -> None:
        validate_status_card_kind(self.kind)
```

Suggested builder functions:

- `build_annotation_no_spatialdata_card_spec(...)`
- `build_annotation_coordinate_system_missing_card_spec(...)`
- `build_annotation_target_missing_card_spec(...)`
- `build_annotation_target_ready_card_spec(...)`
- `build_annotation_existing_shapes_opened_card_spec(...)`
- `build_annotation_layer_ready_card_spec(...)`
- `build_annotation_save_success_card_spec(...)`
- `build_annotation_error_card_spec(...)`
- `build_create_holes_success_card_spec(...)`
- `build_create_holes_error_card_spec(...)`
- `build_annotation_edit_warning_card_spec(...)`

Integration with Slice 2D refactor:

- `_AnnotationLayerReadiness` can carry a
  `_ShapesAnnotationStatusCardSpec | None` instead of raw `title`, `lines`, and
  `kind` fields:

  ```python
  @dataclass(frozen=True)
  class _AnnotationLayerReadiness:
      actionable: bool
      status: _ShapesAnnotationStatusCardSpec | None = None
  ```

- The readiness evaluator stays pure: it returns readiness and a status spec,
  but does not call `_set_status(...)`.
- The widget gets one apply helper, matching existing project patterns:

  ```python
  def _apply_status_card_spec(self, spec: _ShapesAnnotationStatusCardSpec | None) -> None:
      ...
  ```

- Operation handlers choose explicitly whether to show a readiness status or an
  operation-result status. This should remove the confusing
  `_refresh_save_shapes_state(update_status=False)` pattern over time.

Design notes:

- Keep the status helper focused on message construction only. It should not
  inspect live napari layers or mutate widgets.
- Use `format_feedback_identifier(...)` in the helper for shortened
  shapes-name and coordinate-system display text, matching the current widget
  behavior.
- Keep table-linked warnings explicit in the create-holes and save status
  builders.
- Prefer one generic error builder only when the title/wording is truly shared;
  otherwise keep small named builders so call sites remain readable.

Acceptance criteria:

- Annotation status-card text construction lives in
  `widgets/shapes_annotation/status_card.py`.
- `widget.py` applies status-card specs rather than constructing every status
  inline.
- Existing user-visible status wording is preserved unless intentionally
  changed in the spec.
- The helper supports the Slice 2D readiness refactor without introducing UI
  side effects into readiness evaluation.

## Slice 2I - Documentation And Main Roadmap Alignment

Status: proposed.

After the implementation is stable, align the broader roadmap wording.

Tasks:

- Update `annotation_widget_holes.md` so Slice 2 points to this document.
- Replace older `difference(unary_union(...))` wording with the stricter
  "contained polygons become direct holes" contract.
- Keep full boolean subtraction as a future extension.
- Document that the first UI action is `Create holes`, not a full
  `Subtract selected` implementation.

Acceptance criteria:

- The roadmap documents no longer imply that Slice 2 performs arbitrary
  boolean subtraction.
- Future boolean operations remain explicitly out of scope.

## Out Of Scope

The following should not block the first implementation:

- arbitrary Shapely `difference(...)` subtraction
- cutters that touch or cross the shell boundary
- cutters that split a shell into multiple polygons
- `MultiPolygon` annotation sessions
- converting boolean results into multiple annotation rows
- selecting an explicit master polygon through a separate UI state
- selecting a shell through a clicked shell vertex
- selecting holes through clicked hole vertices
- brush/eraser-style hole creation
- fill-hole or remove-hole operations
- rectangle or ellipse rows as hole cutters

Rectangle and ellipse rows can remain save-supported annotations, but the
first `Create holes` action should require polygon rows. A later slice can
convert rectangle/ellipse cutters into polygons if that becomes important.

## Definition Of Done

- The geometry helper creates and validates direct holes without boolean
  subtraction.
- Selection planning is deterministic and side-effect free.
- Layer mutation preserves shell identity and removes consumed rows safely.
- The widget exposes a simple `Create holes` action.
- Create-new, edit-existing, and adopted-native workflows can save the result.
- Invalid selections fail clearly and leave the layer unchanged.
- `MultiPolygon` remains unsupported for annotation.
