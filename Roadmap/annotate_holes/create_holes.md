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
- direct/vertex mode is allowed as visual feedback if it helps users see which
  rows are selected
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

Status: proposed.

Add a public core helper that constructs a hole-bearing Shapely polygon from a
shell polygon and direct child-hole polygons.

Suggested API:

```python
def create_polygon_with_direct_holes(
    shell: Polygon,
    holes: Sequence[Polygon],
) -> Polygon:
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
- Construct and return one Shapely `Polygon`.
- Return a polygon that can be encoded by
  `shapely_polygon_to_napari_polygon_vertices(...)`.

Implementation notes:

- This helper belongs in `src/napari_harpy/core/shapes_geometry.py`, close to
  the existing encoder/decoder.
- The current `_validate_direct_holes(...)` is private, but the new helper can
  reuse it internally.
- The helper should not know anything about napari layers, features, selected
  rows, or widget state.

Unit tests:

- simple shell plus one child returns a polygon with one interior
- simple shell plus two children returns a polygon with two interiors
- shell with an existing hole preserves that hole and appends a new direct hole
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

Status: proposed.

Add a UI-independent helper that inspects a napari Shapes layer selection and
produces a create-holes plan without mutating the layer.

Selection here means napari shape-row selection, i.e. `layer.selected_data`.
The helper should not inspect transient vertex-edit state such as
`layer._moving_value`.

Suggested internal API:

```python
@dataclass(frozen=True)
class _CreateHolesPlan:
    shell_row_index: int
    hole_row_indices: tuple[int, ...]
    polygon: Polygon
    vertices: np.ndarray


def _create_holes_plan_from_selection(layer: Shapes) -> _CreateHolesPlan:
    ...
```

Responsibilities:

- Read `layer.selected_data`.
- Treat `layer.selected_data` as the only semantic input from the napari UI.
- Interpret selected values as current napari row indices into `layer.data`,
  not as source GeoDataFrame indices.
- Require at least two selected shape rows.
- Require every selected shape row to be a polygon row.
- Decode every selected row with
  `napari_polygon_vertices_to_shapely_polygon(...)`.
- Identify the unique largest-area selected polygon row using Shapely
  `polygon.area`; this row is the shell candidate.
- Fail clearly if the largest area is tied.
- Treat every other selected polygon row as a proposed new hole.
- Build the output polygon with `create_polygon_with_direct_holes(...)`.
- Encode the output polygon with
  `shapely_polygon_to_napari_polygon_vertices(...)`.
- Return a plan containing row indices and replacement vertices.
- Do not mutate the layer if planning fails.

Shell inference rule:

- The shell candidate is the unique selected row whose decoded Shapely polygon
  has the largest area.
- Selection order is not semantic input. Do not use "first selected", "last
  selected", napari `Selection.active`, or private current-selection state.
- If the largest area is tied, fail without mutation and show a clear ambiguity
  warning.
- If the unique largest-area row cannot contain every other selected row as a
  direct hole under the strict helper from Slice 2A, fail without mutation and
  show a clear warning such as "Select one shell polygon and one or more
  polygons fully inside it."

Implementation notes:

- This helper likely belongs in
  `src/napari_harpy/widgets/shapes_annotation/widget.py` at first, because it
  depends on napari `Shapes` layer shape types and selection state.
- If it grows, move it into a small private module such as
  `widgets/shapes_annotation/_create_holes.py`.
- Do not use selected-row ordering as semantic input; napari selection ordering
  should not be treated as a stable public contract.
- Do not support "first clicked shell vertex" or "clicked hole vertices" as an
  input mechanism. That would require a separate custom interaction mode.

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
- non-polygon selected row fails without mutation
- unique largest-area row that cannot contain the other selected rows fails
  without mutation
- selected child with existing holes fails without mutation

Acceptance criteria:

- Planning is deterministic.
- Planning has no layer side effects.
- Invalid selections fail before any row is changed or removed.

## Slice 2C - Layer Row Replacement And Child Removal

Status: proposed.

Add a layer mutation helper that applies a successful create-holes plan.

Suggested internal API:

```python
def _apply_create_holes_plan(layer: Shapes, plan: _CreateHolesPlan) -> None:
    ...
```

Responsibilities:

- Replace the shell row's vertices with the encoded hole-bearing vertices.
- Remove the consumed child rows.
- Preserve the shell row's feature values and source identity.
- Remove the child rows' feature values.
- Preserve unselected rows.
- Restore selection to the resulting shell row.
- Keep the current napari layer mode when possible.
- Rebuild napari shape/vertex caches safely.

Recommended mutation strategy:

1. Store current mode and selected rows.
2. Replace the shell row by assigning a same-length rebuilt `layer.data` list.
   This rebuilds napari's `ShapeList` after the shell row changes length.
3. Remove child rows with napari's public `layer.remove(...)`, sorted by row
   index.
4. Compute the shell row's new index after removing child rows that were before
   it.
5. Restore `layer.mode`.
6. Set `layer.selected_data = {new_shell_row_index}`.

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

- Do not manually edit `layer.features` unless the public napari row-removal
  path proves insufficient.
- Do not call the save converter from this helper.
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

Acceptance criteria:

- After the helper runs, `len(layer.data)` decreases by the number of consumed
  child rows.
- The resulting shell row decodes to a Shapely polygon with the expected holes.
- `layer.features` remains row-aligned.
- The layer can be saved by the existing save path.

## Slice 2D - Widget Button And Status Integration

Status: proposed.

Add the user-facing widget action.

Responsibilities:

- Add a compact `Create holes` button near the existing `Create layer` and
  `Save shapes` controls.
- Connect the button to a new click handler.
- Run the selection planning helper.
- Apply the plan if planning succeeds.
- Report invalid selection or geometry failures through the existing status
  card, without mutating the layer.
- Report success through the existing status card after the layer mutation.
- Refresh save readiness without overwriting the success status.

Suggested handler flow:

```python
def _on_create_holes_clicked(self) -> None:
    layer = self._annotation_layer
    if layer is None:
        return

    try:
        plan = _create_holes_plan_from_selection(layer)
        _apply_create_holes_plan(layer, plan)
    except ValueError as error:
        self._set_status(title="Could Not Create Holes", lines=[str(error)], kind="warning")
        return

    self._refresh_save_shapes_state(update_status=False)
    self._set_status(...)
```

Button enabled state:

- Enable only when an annotation layer is open and save binding/session checks
  pass.
- Do not rely on live selection-change events for the first implementation, and
  do not disable the button based on the current row-selection count.
  napari's `selected_data` is validated when the user clicks `Create holes`.
- If the selected shape rows are invalid for creating holes, keep the layer
  unchanged and show a warning.
- Do not try to infer intent from the last clicked vertex or active vertex-edit
  target.

Success message:

- Include how many child polygons were converted into holes.
- If the session is table-linked, keep or repeat the existing warning that
  linked tables are not updated when rows are added or removed.

Scope:

- The button acts on `self._annotation_layer`, not whichever layer is active in
  napari, until the active-layer synchronization roadmap is implemented.
- Native layers adopted by the widget should work because they become the
  annotation layer and already use the same save path.

Widget tests:

- button does nothing when no annotation layer is open
- invalid selection shows `Could Not Create Holes`
- valid selection mutates the layer and shows success
- table-linked session includes the linked-table warning
- save button remains enabled after a successful operation
- dirty-state detection sees the changed layer

Acceptance criteria:

- Users can create holes without a modal dialog or custom drawing mode.
- Invalid selections fail clearly and do not mutate the layer.
- The operation is treated like any other unsaved annotation edit.

## Slice 2E - End-To-End Save And Reload

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

## Slice 2F - Documentation And Main Roadmap Alignment

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

## Open Questions

- Should direct/vertex mode selection be treated differently from shape-select
  mode selection?
  Recommended answer: no. Users may use whichever napari mode gives useful
  visual feedback, but the operation consumes the selected shape rows in
  `layer.selected_data`.

## Definition Of Done

- The geometry helper creates and validates direct holes without boolean
  subtraction.
- Selection planning is deterministic and side-effect free.
- Layer mutation preserves shell identity and removes consumed rows safely.
- The widget exposes a simple `Create holes` action.
- Create-new, edit-existing, and adopted-native workflows can save the result.
- Invalid selections fail clearly and leave the layer unchanged.
- `MultiPolygon` remains unsupported for annotation.
