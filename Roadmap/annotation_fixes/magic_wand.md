# Magic Wand Lasso Refinement

## Status

Specification draft.

This note proposes a magic-wand-style refinement feature for the Shapes
Annotation workflow. The feature is inspired by QuPath's wand/brush annotation
tools, but should be adapted to napari-harpy's current lasso-based shape editing
model rather than copied directly.

References:

- QuPath annotation tools:
  <https://qupath.readthedocs.io/en/stable/docs/starting/annotating.html>
- Existing lasso/pan investigation:
  [`../shapes_elements/pan_zoom_version_2.md`](../shapes_elements/pan_zoom_version_2.md)
- Existing radius-move investigation:
  [`radius_based_move.md`](./radius_based_move.md)
- Existing hole-annotation plan:
  [`../annotate_holes/create_holes.md`](../annotate_holes/create_holes.md)

## Product Goal

Let users quickly draw an approximate region with napari's polygon/lasso tool,
then refine that rough lasso into an image-aware boundary using a magic-wand
operation.

The desired mental model is:

1. draw a rough lasso around the intended object or tissue region;
2. click inside that lasso on a representative pixel;
3. Harpy grows a pixel region similar to the clicked pixel, constrained by the
   lasso;
4. Harpy converts the grown region back into an editable/savable polygon row.

The lasso should be the safety boundary. A classic magic wand can run away when
contrast is poor. In Harpy, the selected lasso polygon should cap how far the
wand is allowed to grow unless the user explicitly enables a margin outside the
lasso.

## Short Recommendation

Implement the first version as **completed-lasso refinement**, not as a custom
in-progress lasso mode.

That means:

- the user finishes a normal napari `ADD_POLYGON_LASSO` shape;
- the finished polygon row is selected in the annotation `Shapes` layer;
- the user enables wand refinement and clicks a seed point inside that selected
  polygon;
- Harpy computes a candidate mask inside the lasso constraint;
- Harpy previews the candidate boundary;
- the user applies it, replacing the selected polygon row while preserving row
  identity and features.

Do not try to intercept the live lasso generator for the first slice. Napari's
in-progress lasso state is private and already interacts delicately with Space
pan behavior. A completed-row workflow is much easier to test, much less likely
to corrupt drawing state, and still gives the main "rough lasso, smart cleanup"
experience.

## QuPath-Inspired Behavior To Keep

QuPath describes the wand as a more aggressive brush that expands from the click
point while pixels remain similar to the clicked pixel. It also describes brush
editing as an additive/subtractive way to refine ROIs drawn with other tools.

Useful ideas for Harpy:

- seed the operation from a user click;
- use pixel similarity in the currently relevant image/channel;
- support additive and subtractive refinement eventually;
- make the tool feel like a fast annotation aid, not a full segmentation
  workflow;
- let image contrast/transforms influence what the wand sees, if we can expose
  that clearly.

Important difference for Harpy:

- the first implementation should be lasso-constrained by default, because this
  gives users a predictable escape hatch when local contrast is weak.

## Non-Goals

The first implementation should not be:

- a pixel-classification pipeline;
- a trainable segmentation model;
- a full ImageJ/QuPath clone;
- a general boolean-geometry editing UI;
- a tool that modifies an in-progress lasso stroke before napari has finished
  creating the shape;
- an unconstrained flood-fill that can select an arbitrarily large connected
  region outside the user's rough annotation;
- a `MultiPolygon` annotation workflow.

## User Flow

### Primary Flow

1. User opens or creates a Shapes Annotation layer.
2. User loads or selects a compatible image layer in the same SpatialData object
   and coordinate system.
3. User draws a rough polygon with napari's polygon/lasso tool.
4. User clicks/selects exactly one valid polygon row in the annotation layer.
5. The magic-wand card opens inside the Shapes Annotation widget.
6. User chooses:
   - source image layer;
   - source channel or visible overlay channel;
   - tolerance;
   - optional lasso margin, meaning how far the wand is allowed to grow outside
     the original lasso.
7. User clicks `Refine`.
8. Harpy enters seed-pick mode.
9. User clicks a representative seed point for the selected polygon.
10. Harpy computes a constrained candidate region and shows a preview contour.
11. User applies the preview.
12. Harpy replaces the selected polygon row with the refined polygon, preserves
    row features and source identity, selects the refined row, and marks the
    annotation layer dirty.

### Magic-Wand Card

Implement the feature as a dedicated magic-wand card inside the Shapes
Annotation widget.

Card activation is driven by napari row selection, not by a separate target
picker:

- when the active annotation-owned primary `Shapes` layer has exactly one
  selected polygon row, the card opens or becomes enabled for that row;
- when there is no valid selected polygon row, the card stays collapsed or
  disabled with a short status;
- when multiple rows are selected, the card does not pick one implicitly;
- when the selected row changes, any pending preview or seed-pick state is
  cancelled and the card refreshes for the new row.

The selected row is the refinement target and lasso constraint. It is a current
napari row index from `layer.selected_data`, not a source GeoDataFrame index
value stored in `layer.features`.

User action inside the card:

1. User selects/refines card parameters.
2. User clicks `Refine`.
3. The next valid click in the viewer is interpreted as the seed point.
4. The seed click must land inside the selected polygon or its configured lasso
   margin.
5. Harpy shows a preview and exposes `Apply` / `Cancel`.

Do not start seed-pick mode merely because a polygon row is selected. The row
selection opens/enables the card; `Refine` arms the one-click seed interaction.

### Later Add/Subtract Flow

After replacement refinement is stable, add a mode selector:

- `Replace`: selected lasso row becomes the wand region.
- `Add`: selected row becomes `selected_polygon.union(wand_region)`.
- `Subtract`: selected row becomes `selected_polygon.difference(wand_region)`.

For add/subtract, reject results that produce unsupported `MultiPolygon` or
geometry collections unless and until annotation save/edit supports them. If a
subtract result creates a valid direct hole, encode it using the existing
hole-aware polygon helpers.

## User Controls

Recommended controls in the Shapes Annotation widget:

- `Wand refine` toggle or tool button.
- Source image combo, filtered to image layers bound to the same `SpatialData`
  object and coordinate system as the annotation layer.
- Channel selector:
  - for overlay mode, one loaded channel layer can be selected directly;
  - for stack mode, select one channel by index/name;
  - RGB/composite support can be deferred.
- Tolerance slider in normalized intensity units.
- Connectivity selector, default `8-connected`.
- Lasso margin numeric control:
  - default `0`, meaning the candidate cannot leave the selected lasso;
  - positive values dilate the lasso constraint, allowing the candidate to grow
    outside the original lasso by that many image pixels or coordinate-system
    units, depending on the final unit decision.
- Smooth/simplify controls:
  - smoothing radius for the mask before contouring;
  - polygon simplification tolerance after contouring.
- Preview/apply/cancel controls.

Avoid adding a new persistent napari layer to the save model. Preview layers
must be temporary viewer affordances and must be removed when refinement is
cancelled, applied, the tool is disabled, or the annotation session closes.

## Behavior Contract

### Annotation Selection

Required input:

- one active annotation-owned primary `Shapes` layer;
- exactly one selected row in `layer.selected_data`;
- selected row has `shape_type == "polygon"`;
- selected row can be decoded by `napari_polygon_vertices_to_shapely_polygon(...)`;
- selected row is the lasso constraint and the row that will be replaced on
  apply.

Selection-driven card behavior:

- listen to selection changes on the annotation-owned layer's `selected_data`;
- open/enable the magic-wand card only for exactly one selected polygon row;
- clear preview and seed-pick state when the selected row, annotation layer,
  target shapes element, coordinate system, or selected `SpatialData` object
  changes;
- do not use a clicked source feature value or GeoDataFrame index as the primary
  trigger, because edit-safe mutation happens against the current napari row.

Reject:

- no selected row;
- multiple selected rows;
- non-polygon selected row;
- malformed polygon rows;
- styled viewer-only shapes layers;
- point-backed shapes;
- layers not owned by the Shapes Annotation widget.

### Source Image Selection

The source image should be a loaded napari `Image` layer that has an
`ImageLayerBinding` matching:

- same `SpatialData` object id;
- same coordinate system;
- compatible transform into the annotation layer's displayed coordinate system.

For the first slice, support simple 2D source planes:

- single 2D image layer;
- one overlay channel layer;
- one channel selected from a stack-mode image layer.

Defer full support for:

- RGB composite similarity;
- multiscale image pyramids;
- z/time slicing beyond the currently visible 2D plane;
- rotated/sheared transforms;
- huge lazy dask reads that exceed the configured ROI pixel budget.

### Lasso Constraint

The selected polygon row defines a binary constraint mask.

Default behavior:

- candidate pixels must be inside the selected lasso polygon;
- the seed click must be inside the selected lasso polygon;
- flood growth is clipped to the lasso mask;
- the final polygon is also clipped to the lasso-derived constraint.

Optional margin behavior:

- if margin is positive, dilate the rasterized lasso mask before region growth;
- the seed still has to be near or inside the original selected polygon;
- the preview should make it clear when the result extends beyond the original
  lasso.

This makes the lasso a user-controlled safety guard instead of just an initial
guess.

### Output Geometry

On apply:

- replace only the selected row's vertices;
- preserve `layer.features` row values;
- preserve source identity feature values for edit-existing sessions;
- preserve row order;
- preserve current mode as much as possible;
- select the refined row;
- keep style arrays aligned;
- leave the existing save path responsible for writing to `SpatialData.shapes`.

The output must be one valid Shapely `Polygon` encodable by
`shapely_polygon_to_napari_polygon_vertices(...)`.

Polygons with holes are supported:

- the candidate may contain direct interior rings;
- valid direct holes should be preserved when contour extraction can identify
  one shell and unambiguous interior rings;
- small noisy holes may be filled according to the configured cleanup policy;
- nested holes, overlapping holes, and ambiguous hole topology should be
  rejected or filled before apply rather than saved as invalid geometry.

`MultiPolygon` is not supported:

- the implementation must keep only the connected component containing the seed;
- disconnected candidate components must not become multiple annotation parts;
- any polygonization or boolean result whose geometry type is `MultiPolygon` or
  `GeometryCollection` must fail clearly and leave the selected row unchanged;
- supporting one annotation as multiple napari rows requires a separate
  `MultiPolygon` annotation design and is outside this feature.

Reject or ask the user to adjust settings when:

- no candidate region is found;
- candidate region is too small;
- candidate region touches too much of the lasso boundary, suggesting runaway
  tolerance;
- candidate conversion produces an invalid polygon;
- candidate conversion produces multiple disconnected polygons;
- simplification makes the polygon invalid;
- result would require unsupported geometry types.

## Algorithm Direction

### 1. Resolve Coordinate Mapping

Use world coordinates as the bridge between the annotation layer and image
layer.

For the selected annotation polygon:

1. Convert annotation layer data coordinates to world coordinates if needed.
2. Convert world coordinates into source image data coordinates with
   `image_layer.world_to_data(...)`.
3. Work in source image `(y, x)` pixel coordinates for raster operations.
4. Convert the final contour from image data coordinates back through world
   coordinates into annotation layer data coordinates.

Even if current annotation layers often have normalized identity-like
transforms, using the layer transform APIs keeps the design robust.

The first implementation should explicitly validate that this mapping is
effectively 2D and axis-aligned. If not, report that wand refinement is
unavailable for the current image/annotation transform.

### 2. Extract A Bounded ROI

Compute the selected polygon's image-space bounding box, expanded by the lasso
margin and a small contouring pad.

Then:

- clip the bounding box to the source image extent;
- refuse the operation if the ROI is empty;
- refuse or downsample if the ROI exceeds a configured pixel budget;
- read only this ROI from dask-backed source data.

Suggested initial pixel budget:

```text
max_wand_roi_pixels = 5_000_000
```

This should be a guardrail, not a final performance target. Later multiscale
support can pick an image pyramid level based on lasso size and current zoom.

### 3. Rasterize The Lasso

Rasterize the selected polygon into the ROI grid.

Use `skimage.draw.polygon2mask(...)`. `scikit-image` is a direct dependency for
this feature, so the implementation should prefer its well-tested raster,
contour, morphology, and connected-component utilities over custom one-off
rasterization code.

### 4. Normalize The Source Signal

The first version should use one scalar image plane.

Recommended normalization:

1. Extract ROI pixels inside the lasso constraint.
2. Compute robust percentiles, for example 1st and 99th.
3. Clip and scale to `[0, 1]`.
4. Use the seed pixel value in normalized units.

The candidate similarity mask is:

```text
abs(normalized_roi - normalized_seed_value) <= tolerance
```

This is intentionally simple and explainable.

Future extensions:

- use multiple selected channels and Euclidean distance in normalized channel
  space;
- use the active image contrast limits instead of recomputed ROI percentiles;
- use an explicit transform such as inverted intensity, log/asinh, or a
  stain/color transform;
- seed from the median intensity of a small click-radius disk instead of one
  pixel.

### 5. Region Grow From Seed

Compute the connected component containing the seed inside:

```text
similarity_mask & lasso_constraint_mask
```

Implementation recommendation:

- compute `similarity_mask & lasso_constraint_mask`;
- label connected components with `skimage.measure.label(...)`;
- keep the label under the seed pixel;
- reject the result if the seed pixel is not in a valid component.

This keeps the lasso constraint explicit and avoids a custom flood-fill loop.

### 6. Clean The Mask

Mask cleanup should be conservative:

- remove tiny islands if a later operation can introduce them;
- fill holes smaller than a configured area;
- optionally close one-pixel gaps;
- preserve the seed-containing component only.

Avoid aggressive smoothing before the user can see a preview. Over-smoothing is
especially risky for thin biological structures.

### 7. Convert Mask To Polygon

Convert the candidate mask boundary to one Shapely `Polygon`.

Recommended path:

1. Find contours in ROI image coordinates with
   `skimage.measure.find_contours(...)`.
2. Convert contour coordinates to coordinate-system/world positions.
3. Build a Shapely polygon.
4. Optionally simplify with a small tolerance.
5. Validate with existing geometry rules.
6. Encode with `shapely_polygon_to_napari_polygon_vertices(...)`.

If the mask contains holes:

- preserve valid direct holes when contour extraction produces one unambiguous
  shell and interior rings;
- fill small noisy holes below a configurable area threshold;
- reject nested, overlapping, edge-touching, or otherwise ambiguous hole
  topology with a clear warning.

If contour extraction returns more than one exterior shell, keep only the shell
associated with the seed component. If a later polygonization or boolean step
still produces a `MultiPolygon`, reject it without mutating the annotation
layer.

### 8. Preview And Apply

Preview should show the candidate as a temporary contour or translucent filled
shape.

Apply should reuse the same safe row-mutation pattern used by create-holes:

- capture style state;
- rebuild `layer.data` if vertex count changes;
- preserve features and shape types;
- restore styles;
- restore/keep selected row;
- emit napari-compatible data events through public layer APIs when possible;
- mark the annotation layer dirty through the existing snapshot comparison.

The create-holes helper already shows the right pattern in
`src/napari_harpy/widgets/shapes_annotation/_create_holes.py`.

Consider extracting a shared helper such as:

```python
def _replace_shapes_layer_row_preserving_style(
    layer: Shapes,
    row_index: int,
    vertices: np.ndarray,
) -> None:
    ...
```

That helper could serve magic wand, create-holes, and future shape-edit tools.

## Integration With Existing Code

### Best Extension Point

The existing `_AnnotationLayerEditGuard` is the right lifecycle boundary for
annotation-owned layer hooks, but the wand operation itself does not need to be
implemented inside the guard initially.

Recommended split:

- widget owns UI controls, selected source image, preview layer, and apply/cancel
  state;
- pure helpers own ROI extraction, masking, region growing, and polygonization;
- a small layer-mutation helper owns safe replacement of one Shapes row;
- `_AnnotationLayerEditGuard` remains focused on private napari edit hooks
  such as direct vertex sync, vertex deletion, and Space-pan behavior.

Only add guard hooks later if we want a true click-mode that captures canvas
mouse events while `Wand refine` is active. Even then, route the event to the
widget/controller instead of putting image segmentation logic inside the guard.

The card can use row-selection events to decide whether it is open/enabled. The
seed-click interaction should be armed only after the user clicks `Refine`.
That keeps ordinary napari row selection and direct editing behavior intact.

### Candidate Modules

Possible new files:

```text
src/napari_harpy/widgets/shapes_annotation/_magic_wand.py
src/napari_harpy/core/magic_wand.py
tests/test_shapes_annotation_magic_wand.py
```

Keep pure image/geometry functions in `core` if they do not require Qt/napari
layer objects. Keep napari layer mutation and preview management under
`widgets/shapes_annotation`.

### Dependency Decision

Decision: add `scikit-image` as a direct project dependency for this feature.

Expected APIs:

- `skimage.draw.polygon2mask`;
- `skimage.measure.find_contours`;
- `skimage.morphology`;
- `skimage.measure.label`;

This avoids relying on napari's transitive dependencies and keeps the wand
implementation smaller and easier to test.

## Proposed Implementation Slices

### Slice 1 - Pure Constrained Region Grow

Add a UI-independent helper that takes:

- 2D image ROI;
- seed pixel;
- binary constraint mask;
- tolerance;
- connectivity;
- normalization options.

It returns one boolean candidate mask.

Tests:

- seed outside constraint is rejected;
- no similar pixels returns a clear error;
- 4-connected and 8-connected behavior differ as expected;
- growth cannot cross the lasso constraint;
- tolerance controls candidate size monotonically;
- non-finite image values are ignored or rejected deterministically.

### Slice 2 - Mask To Polygon

Add a helper that converts a boolean candidate mask and ROI transform into one
valid polygon in annotation coordinates.

Tests:

- simple disk/rectangle mask becomes a valid polygon;
- large direct holes are preserved as Shapely polygon interiors;
- small holes are filled according to explicit policy;
- nested or ambiguous hole topology is rejected or filled before apply;
- disconnected masks are rejected or reduced to the seed component;
- `MultiPolygon` and `GeometryCollection` results are rejected without mutation;
- simplification does not create invalid geometry;
- generated vertices round-trip through
  `napari_polygon_vertices_to_shapely_polygon(...)`.

### Slice 3 - Source Image And Coordinate Resolution

Add widget/controller helpers that discover compatible loaded image layers.

Tests:

- only image layers from the same `SpatialData` and coordinate system are
  selectable;
- unrelated native image layers are rejected unless an explicit "active native
  layer" fallback is intentionally designed;
- stack-mode channel selection resolves the correct 2D plane;
- overlay channel layers resolve as scalar source planes;
- incompatible transforms report a clear status message.

### Slice 4 - Preview Layer

Add temporary preview management.

Tests:

- card opens/enables when exactly one valid polygon row is selected;
- card disables when no row, multiple rows, or a non-polygon row is selected;
- clicking `Refine` arms exactly one seed-pick interaction;
- changing selection cancels an armed seed pick;
- preview layer is created after a successful seed click;
- preview layer is updated when tolerance changes;
- preview layer is removed on cancel/apply/session clear;
- preview layer is not registered as a SpatialData shapes layer;
- preview layer is never used by save.

### Slice 5 - Replace Selected Row

Apply the preview by replacing the selected polygon row.

Tests:

- row features are preserved;
- source identity feature is preserved;
- row order is preserved;
- style arrays remain aligned;
- selected row remains selected;
- annotation dirty-state detects the change;
- saving writes the refined geometry to `SpatialData.shapes[...]`;
- malformed candidate geometry leaves the layer unchanged.

### Slice 6 - Add/Subtract Modes

Add optional boolean refinement modes after replacement is reliable.

Tests:

- add mode unions a wand region into the selected polygon;
- subtract mode creates a valid hole when appropriate;
- unsupported `MultiPolygon` results are rejected without mutation;
- table-linked shape warnings remain explicit.

### Slice 7 - Multiscale And Large ROI Support

Support large images more gracefully.

Options:

- choose a multiscale level based on lasso size and target preview resolution;
- refine at lower resolution first, then optionally recompute boundary near the
  edge at higher resolution;
- expose a clear "ROI too large" warning with a suggested zoom/lasso adjustment.

## Failure Policy

Prefer "no mutation plus status-card warning" over partial mutation.

The operation should leave the selected row unchanged when:

- source image cannot be resolved;
- seed is outside the selected lasso;
- seed maps outside the image plane;
- ROI exceeds budget;
- no valid candidate mask exists;
- candidate polygon is invalid;
- candidate polygonization produces `MultiPolygon` or `GeometryCollection`;
- applying the candidate would break row/style/feature alignment.

Warnings should be short and actionable, for example:

- "Click inside the selected lasso."
- "Increase tolerance; no connected pixels matched the seed."
- "Decrease tolerance; the region filled the lasso boundary."
- "The selected image transform is not supported for wand refinement yet."
- "The candidate boundary could not be converted to a valid polygon."

## Manual QA

Manual checks:

- draw rough lasso around a bright object on dark background;
- refine with a seed click and apply;
- save and reload the shapes element;
- refine a lasso that deliberately includes neighboring tissue and verify the
  wand stays inside the lasso;
- set low tolerance and verify the candidate is small;
- set high tolerance and verify the candidate is clipped by the lasso;
- cancel preview and confirm the original polygon remains unchanged;
- apply to an edit-existing row and confirm source identity/features survive;
- try incompatible source image/layer and confirm the warning is clear.

Later QA:

- additive refinement expands an existing annotation without changing row
  identity;
- subtractive refinement creates a hole where valid;
- large ROI guard prevents accidental whole-slide reads;
- multiscale image layer chooses a sensible preview level;
- Space-pan and normal lasso drawing still behave as specified in the pan/zoom
  roadmap note.

## Open Product Decisions

- Should the first click apply immediately, or should all successful clicks
  create a preview that requires `Apply`?
- Should tolerance be expressed in normalized `[0, 1]` intensity units,
  percentile units, or raw image units?
- Should the default source signal follow the currently visible image contrast
  limits or recompute robust normalization inside the lasso?
- Should lasso margin be expressed in image pixels, coordinate-system units, or
  screen pixels?
- What default area threshold should separate preserved direct holes from small
  noisy holes that are filled during cleanup?
- Should the wand support multi-seed refinement in the first version?
- Should `Alt` eventually switch to subtract mode, following QuPath's brush
  spirit, or should subtract stay as an explicit mode selector?
- How much smoothing should be applied before polygonization by default?
- What is the acceptable maximum ROI size before requiring multiscale support?

## Initial Product Scope

The initial product scope is:

- one selected lasso polygon row;
- one scalar loaded image layer or selected channel;
- one seed click;
- lasso-constrained region growing;
- preview contour;
- replace selected row on apply;
- add/subtract modes are deferred until replacement refinement is robust;
- operate first on the selected 2D image plane; pyramid-level selection and
  multiscale refinement are a later extension;
- clear warnings and no mutation on failure.

This gets the core interaction into users' hands while preserving Harpy's
annotation identity, save semantics, and topology contracts.
