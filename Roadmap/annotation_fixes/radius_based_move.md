# Radius-Based Vertex Move

## Status

Investigation notes for a professional radius-based polygon editing tool in the
Shapes Annotation workflow.

## Product Goal

Allow users to reshape polygon annotations by pushing or pulling multiple nearby
vertices in one drag interaction. The tool should feel like a controlled local
deformation brush: the user chooses an influence radius, drags a vertex or point
on a polygon boundary, and nearby vertices move according to a clear falloff
policy.

This should be designed as a first-class annotation feature, not as a temporary
shortcut around napari's direct-edit mode. It must preserve Harpy's geometry
contracts, avoid silent topology corruption, and produce geometry that can be
saved back to `SpatialData.shapes[...]` through the existing annotation save
path.

## Current System Findings

- Napari 0.7.0 direct vertex editing for `Shapes` uses the private
  `layer._drag_modes[Mode.DIRECT]` callback. `Mode.DIRECT` currently maps to
  napari's `select(...)` mouse-drag callback.
- During a direct drag, napari calls `_move_active_element_under_cursor(...)`,
  which updates one active raw vertex via `layer._data_view.edit(...)` and then
  refreshes the layer.
- Napari emits the public `layer.events.data(..., action=CHANGED, ...)` event at
  the end of a drag, not continuously for every mouse move.
- Harpy already wraps the same private direct-edit path in
  `src/napari_harpy/widgets/shapes_annotation/widget.py` through
  `_AnnotationLayerEditGuard`.
- That guard is attached only to annotation-owned primary `Shapes` layers and is
  already used to keep hole-encoded polygon anchor vertices synchronized during
  live direct drags.
- Existing tests already simulate direct-drag generators and assert the guard's
  behavior, so the interaction layer is testable without full GUI automation.
- The annotation save path already accepts geometry-only edits as long as the
  resulting `Shapes.data` rows convert to valid polygons.

## Geometry Constraints

The central risk is polygon topology, not persistence.

Harpy stores polygon holes as one flat napari polygon row with repeated anchor
coordinates. For example:

```text
A B C D A E F G H E A
```

This represents:

- `A B C D A`: closed exterior shell
- `E F G H E`: closed hole ring
- final `A`: shell separator after the hole

The repeated coordinates are semantic aliases:

- shell anchor group: `A` copies must move together
- hole anchor group: `E` copies must move together
- ordinary vertices can move independently

A radius-based move cannot treat every raw row coordinate as an independent
vertex. It must operate on logical vertices or explicitly synchronize duplicated
raw vertices after each deformation step. Otherwise the tool can create transient
bridge artifacts, malformed hole encodings, or geometry that fails save-time
validation.

The existing helpers in `src/napari_harpy/core/shapes_geometry.py` are relevant:

- `napari_polygon_vertices_to_topology(...)`
- `move_napari_polygon_vertex(...)`
- `napari_polygon_vertices_to_shapely_polygon(...)`

The radius tool should add pure geometry helpers beside these rather than
embedding deformation math directly in the Qt/widget layer.

## Proposed Interaction Model

Add a radius-based move tool to the Shapes Annotation widget for annotation-owned
primary polygon `Shapes` layers.

Expected user controls:

- enable/disable radius-based move mode
- radius in the active coordinate-system units
- falloff policy selector
- possibly strength/sensitivity if testing shows one drag delta is too coarse
- visible influence-radius indicator around the cursor or active vertex

Expected interaction:

1. User opens or creates an annotation layer.
2. User enables radius-based move.
3. User chooses an influence radius and falloff policy.
4. User drags a polygon vertex or boundary handle.
5. Harpy deforms nearby logical vertices during the drag.
6. Harpy keeps duplicated anchor/separator vertices synchronized live.
7. On release, Harpy emits a coherent data-change event and leaves the layer in a
   normal saveable annotation state.

The tool should initially be scoped to annotation-owned primary polygon
`Shapes`. Point-radius shapes currently render as `Points` and are not editable
as polygon annotations. Styled shapes should remain viewer-only and must not be
used as geometry write-back sources.

## Weighted Movement Investigation

The falloff policy is a product and usability decision, not just an implementation
detail. We should investigate and choose deliberately between these behaviors:

| Policy | Behavior | Strengths | Risks |
| --- | --- | --- | --- |
| Constant | Every affected vertex receives the same drag delta. | Predictable, easy to explain, useful for moving local boundary segments rigidly. | Can create abrupt deformation at the radius edge. |
| Linear | Movement decreases linearly with distance from the drag origin. | Simple mental model and smoother than constant movement. | Still has a visible falloff edge unless clamped or eased carefully. |
| Smooth falloff | Movement follows an easing curve, such as smoothstep or Gaussian-like decay. | Feels most like sculpting; avoids harsh transitions. | Harder to explain and tune; may require strength/radius feedback. |

Questions to answer before implementation:

- Should the default feel like rigid local translation or soft sculpting?
- Should the active dragged vertex always follow the cursor exactly?
- Should vertices at the radius boundary move zero distance or a small non-zero
  amount?
- Should the falloff use Euclidean distance in displayed `(y, x)` coordinates
  only?
- Should the tool affect only the active polygon row, selected polygon rows, or
  all editable rows under the radius?
- Should hole vertices be affected by the same radius as shell vertices, and how
  should the UI communicate this?
- Should self-intersection or invalid hole movement be blocked live, reverted on
  release, or allowed until save-time validation reports the issue?

## Implementation Direction

### Geometry Layer

Add pure helpers for logical vertex grouping and deformation:

- parse a polygon row into logical vertex groups, including duplicated shell and
  hole anchors
- compute distances from the drag origin to logical vertices
- compute per-logical-vertex weights for the selected falloff policy
- apply the weighted drag delta to every affected logical group
- return updated napari `(y, x)` vertices without mutating caller-owned arrays
- validate finite coordinates and preserve row length

The pure helper should be testable without napari UI objects.

### Annotation Interaction Layer

Extend `_AnnotationLayerEditGuard` or add a sibling guard for radius-based move
state.

Likely flow:

1. On mouse press, capture the active row, active vertex, original vertices, and
   topology before napari mutates the row.
2. During mouse move, compute the drag delta from the original active vertex to
   the current cursor/vertex position.
3. Apply the radius deformation to the captured original vertices, not to already
   repeatedly-mutated vertices. This avoids accumulating numerical drift and
   makes the drag deterministic.
4. Write updated rows through `layer._data_view.edit(...)` when row length is
   unchanged.
5. Call `layer.refresh()` for live feedback.
6. On release, emit data events consistent with napari's expectations and refresh
   the clean/dirty annotation state through the existing snapshot logic.

If the tool eventually supports cross-row deformation, it should update every
affected row in one coherent operation and emit row/vertex indices for all
affected rows.

### Visual Radius Indicator

Investigate two options:

- napari cursor model: use a circular cursor with `viewer.cursor.size`, scaled to
  match the chosen radius when possible
- overlay layer: draw a temporary transparent circle or ring in data coordinates

The overlay layer is likely more precise for coordinate-system-unit radii, while
the cursor model may be simpler and less intrusive. The chosen approach must not
be saved as annotation geometry and must be cleaned up when the tool is disabled
or the annotation session closes.

## Validation and Failure Policy

Radius deformation can create invalid polygons more easily than single-vertex
editing. The feature needs an explicit policy.

Recommended behavior to investigate:

- During drag, keep the operation live and responsive.
- On release, validate affected rows with the existing geometry conversion
  helpers.
- If a row becomes invalid, show an actionable warning in the annotation status
  card.
- Decide whether invalid release should revert automatically or leave the edited
  layer dirty but unsaveable until the user repairs it.

Automatic live blocking sounds attractive but may feel sticky or unpredictable
unless the invalidity test is fast and the visual feedback is clear.

## Tests

Pure geometry tests:

- constant falloff moves all vertices inside radius by the full delta
- linear falloff gives decreasing movement by distance
- smooth falloff reaches full movement at the active vertex and near-zero at the
  radius edge
- vertices outside the radius do not move
- duplicated shell anchors move together
- duplicated hole anchors move together
- ordinary hole vertices can move independently
- malformed topology fails clearly
- input arrays are not mutated unexpectedly

Annotation interaction tests:

- attaching/disconnecting the radius tool restores the original layer hooks
- radius mode is limited to compatible annotation-owned primary `Shapes` layers
- a simulated drag updates multiple vertices in one polygon row
- direct anchor synchronization still works when radius mode is disabled
- row features, shape types, selected data, and current mode are preserved
- data events identify affected rows and vertices
- invalid deformation reports a status-card warning

Manual QA:

- create a new annotation layer and push a simple polygon boundary
- open an existing polygon shapes element and push a boundary, then save
- push near a polygon hole and verify hole topology remains stable
- switch radius/falloff settings mid-session
- disable the tool and confirm normal napari direct-edit behavior returns
- verify styled shapes and point-rendered shapes are not editable through this
  tool

## Open Product Decisions

- Default falloff policy.
- Default radius and radius step size.
- Whether radius is expressed in data coordinates, physical units, or screen
  pixels.
- Whether deformation scope is active row, selected rows, or all editable rows.
- Whether invalid release reverts automatically or leaves repairable dirty state.
- Whether the tool should be named "Push", "Radius Move", "Soft Move", or another
  term in the UI.
