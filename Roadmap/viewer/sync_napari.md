# Live napari synchronization for Labels and Shapes

Status: proposed

## Motivation

The Viewer widget currently presents Labels and Shapes as staged forms. The
user selects a color source and then presses **Add / Update in viewer**. Once a
layer exists, napari independently exposes controls for visibility,
presentation, ordering, and removal.

This creates two competing control surfaces:

- the Harpy card suggests that its form is the authoritative viewer state;
- napari can change or remove the corresponding layer without the card
  reflecting that change;
- **Add / Update** combines initial membership with later presentation updates,
  even though these are different actions;
- the user cannot see which primary and styled variants are currently loaded.

Labels and Shapes should instead use a live membership model. Napari is the
source of truth for whether a bound layer exists and for its current native
presentation. Harpy remains responsible for SpatialData identity, color-source
selection, validated mutations, and table-driven styling.

## Goals

- Remove **Add / Update in viewer** from Labels and Shapes cards.
- Show the currently loaded primary and styled layers inside each card.
- Synchronize visibility and removal in both directions between Harpy and
  napari.
- Reflect relevant napari-side presentation changes in the Harpy rows.
- Make accepting a valid styled color source load that variant without a
  second confirmation action.
- Preserve the distinction between annotation-capable primary layers and
  viewer-only styled variants.
- Support multiple styled variants for the same SpatialData element.
- Keep existing Object Classification, Spatial Query, and Shapes annotation
  consumers working.

## Non-goals

- Inferring a Harpy table or column source from arbitrary colors selected in
  napari.
- Representing a categorical or direct palette as one editable solid color.
- Treating unbound native napari layers as Harpy-managed layers.
- Persisting visibility or other viewer-only presentation state to
  SpatialData.
- Replacing the separate continuous colorbar and range-control design.
- Generalizing all image, labels, and shapes rows into one class when their
  presentation semantics do not match.

## Findings from the current implementation

### Layer roles and identity

`LabelsLayerBinding` and `ShapesLayerBinding` already distinguish `primary`
and `styled` layers. Styled identity includes the selected color-source
specification.

This means one element can have:

- at most one primary layer for a SpatialData object and coordinate system;
- several styled layers, one for each distinct color-source specification.

The UI therefore cannot model a Labels or Shapes card as having one selected
layer. It must render primary membership separately and show every live styled
variant.

Primary Labels layers are shared with workflows such as Object Classification
and Spatial Query. Primary Shapes layers may be used for annotation. Styled
layers are viewer-only presentations and must never be offered as annotation
or write-back sources.

Shapes bindings can refer to either a napari `Shapes` layer or a point-backed
napari `Points` layer. Membership and visibility controls must work for both
rendering modes.

### Existing loading behavior

The current card emits one broad request:

- `LabelsLoadRequest` is handled by either `ensure_labels_loaded(...)` or
  `ensure_styled_labels_loaded(...)`;
- `ShapesLoadRequest` is handled by either `ensure_shapes_loaded(...)` or
  `ensure_styled_shapes_loaded(...)`.

The styled `ensure_*` methods create a missing variant or reapply table-driven
styling to an existing one. This is why the current button is labelled
**Add / Update**. The live design should retain those operations but invoke
them from narrower, contextual intents instead of one staged form submission.

### Adapter lifecycle gaps

The adapter currently exposes `primary_labels_layers_changed` for consumers of
primary Labels membership and `primary_shapes_layer_registered` for primary
Shapes registration. Neither signal represents the complete primary-plus-
styled membership required by the Viewer cards.

The existing Labels and Shapes removal helpers resolve the primary layer for an
element. They cannot safely remove one exact styled variant. Existing styled
lookup helpers can also return the first matching layer without surfacing a
duplicate semantic match.

Complete live synchronization therefore requires:

- membership invalidation for all usable Labels bindings;
- membership invalidation for all usable Shapes bindings;
- ordered queries for complete live binding snapshots;
- exact, identity-validated removal of a primary or styled binding;
- explicit duplicate detection rather than first-match mutation.

The current primary-specific signals must remain available because they have a
different consumer contract.

## Proposed card UX

Each Labels or Shapes card has two conceptually separate areas.

### Primary layer

When the primary layer is absent, show a contextual **Load in viewer** action.
This is an explicit initial membership action, not a staged update action.

After a successful load, replace it with a live row:

```text
eye  Primary  presentation preview  ×
```

The row remains visible for as long as the exact bound layer remains in
napari. Removing the layer through either UI restores the unloaded state.

Separating primary loading from styled color-source selection also removes the
need for the current `No color source` option to stand in for primary-layer
membership.

### Styled variants

Keep the linked-table, source-kind, and searchable value-source controls as a
composer. They define the identity of a styled variant; they do not mirror a
napari color property.

When the user accepts a complete and valid color source:

1. validate the current SpatialData object and coordinate system;
2. create and style the corresponding variant if it is absent;
3. if that exact variant already exists, activate it without treating source
   selection as an implicit restyle operation;
4. allow membership reconciliation to create or retain the live row;
5. clear the value-source search field after a successful new load.

If users need to re-read changed table values later, provide a contextual
**Refresh style from data** action for that live variant or reconcile it from a
known data-change event. Do not bring back a combined Add/Update action.

Every loaded variant is shown independently:

```text
eye  obs["cell_type"]  palette preview  ×
eye  X[:, "marker_score"]  gradient preview  ×
```

Multiple variants may coexist. Accepting or removing one variant must not
remove its siblings.

For Shapes, **Fill** is presentation state rather than styled-layer identity.
The composer can provide its initial value. Once a variant is loaded, any Fill
control should be associated with that live row and update the exact layer
immediately; it must not require an Update button or create another variant.
The row should refresh from the resulting native face-color state.

## Live synchronization contract

### Membership

The adapter should expose two payload-free invalidation signals, for example:

```python
labels_layers_changed = Signal()
shapes_layers_changed = Signal()
```

Each signal means that the set or order of usable live bindings may have
changed. Consumers must re-query the adapter; the signal payload must not carry
a binding that may already be stale after removal.

Emit the appropriate invalidation after:

- a usable primary or styled binding is registered while its layer is live;
- a usable primary or styled layer is removed from napari;
- adapter fallback removal unregisters a binding;
- relevant napari layer reordering occurs.

Do not emit membership invalidation for visibility, colormap, face-color, or
edge-color changes. Those are layer presentation events.

Viewer reconciliation should receive a complete, ordered snapshot of live
bindings for one element. It validates the snapshot before changing the card:

- no more than one primary binding;
- no duplicate styled binding for the same style specification;
- every binding belongs to the current SpatialData object, element, and
  coordinate system;
- every bound layer is still present in napari;
- every Shapes binding has a supported rendering mode.

On invalid membership, show a non-mutating card-level error. Do not silently
choose one duplicate.

### Visibility

Each live row reads its eye state from `layer.visible` and subscribes directly
to `layer.events.visible`.

The interaction flow is:

```text
Harpy eye intent
    -> ViewerWidget validates the row's binding identity
    -> layer.visible is assigned
    -> napari emits layer.events.visible
    -> the row refreshes its eye from layer.visible
```

When reflecting napari state into the Qt eye button, use `QSignalBlocker` so a
programmatic `setChecked(...)` does not create another mutation request. Skip
an assignment when `layer.visible` already equals the requested value.

### Removal

The row emits a removal intent containing or capturing its construction-time
binding. `ViewerWidget` must confirm that this is still the binding owned by
the current row and that the layer remains live before asking the adapter to
remove it.

The adapter should remove that exact layer or binding. It must not re-resolve
only by element name because several styled variants can coexist.

If the user removes a layer directly in napari, the adapter unregisters the
binding, emits membership invalidation, and reconciliation disposes the Harpy
row. A delayed signal from that disposed row must not be allowed to mutate a
replacement layer.

### Presentation

Rows should subscribe directly to the native events that drive their preview:

- Labels: the applicable colormap event;
- Shapes: applicable current face-color, current edge-color, face-color, and
  edge-color events for the concrete napari layer type;
- point-backed Shapes: the corresponding Points color events.

The exact subscription set should be kept inside the row's presentation
adapter rather than spread through `ViewerWidget`.

Presentation must remain semantically honest:

- show a solid swatch only when the live presentation is genuinely solid;
- show a palette or gradient preview for categorical or continuous mappings;
- do not let a single-color control overwrite a multi-value table-derived
  palette accidentally;
- keep the binding's color-source specification as provenance even when the
  user changes native colors in napari.

When Harpy provides an appropriate presentation editor, it should assign the
native layer property and let napari's event refresh all bound UI peers. Harpy
must not manually emit napari property events.

## Widget and row ownership

Use the same ownership boundaries for both element types:

- a live row owns native presentation subscriptions and rendering;
- the row emits user intent only;
- `ViewerWidget` owns current-context checks, binding identity validation, and
  mutations;
- `ViewerAdapter` owns viewer membership operations and the binding registry;
- a binding is fixed for the lifetime of a row;
- replace and dispose a row when its binding changes rather than rebinding the
  existing widget;
- row disposal disconnects every native layer event subscription.

A small shared presentation-neutral row shell can provide the eye, semantic
label, preview slot, and remove action. Labels- and Shapes-specific
presentation adapters should own their different palette and color semantics.
The existing image row should only be generalized if that produces a simpler
API than keeping these rows separate.

## Concrete implementation work

### Adapter membership foundation

- Add complete Labels and Shapes membership invalidation signals.
- Emit them from registration, insertion, removal, fallback cleanup, and
  reorder paths.
- Preserve `primary_labels_layers_changed` and
  `primary_shapes_layer_registered` semantics for their existing consumers.
- Add complete ordered live-binding queries for one element and coordinate
  system.
- Add exact binding/layer removal operations for both primary and styled
  variants.
- Validate duplicate primary and duplicate styled identities.

### Live row infrastructure

- Add a presentation-neutral live-layer row shell or narrowly shared helper.
- Implement native visibility subscription, blocked Qt reflection, intent
  signals, and deterministic disposal.
- Add Labels palette-preview extraction and event subscriptions.
- Add Shapes/point-backed-Shapes presentation-preview extraction and event
  subscriptions.
- Keep row construction-time binding identity available for stale-intent
  validation.

### Labels card integration

- Split primary membership from the styled source composer.
- Replace the primary Add/Update workflow with pending-load and live-row
  presentations.
- Load styled variants when a valid source is accepted.
- Render every live styled Labels binding.
- Reconcile external visibility, removal, ordering, and colormap changes.
- Remove `LabelsLoadRequest`, `add_update_requested`, and the broad
  `_add_or_update_labels_layer(...)` path after narrower intents cover all
  behavior.

### Shapes card integration

- Split primary membership from the styled source composer.
- Replace the primary Add/Update workflow with pending-load and live-row
  presentations.
- Load styled variants when a valid source is accepted.
- Render every live styled Shapes binding, including point-backed variants.
- Make Fill an immediate per-live-variant presentation change after loading.
- Reconcile external visibility, removal, ordering, and relevant Shapes or
  Points color events.
- Remove `ShapesLoadRequest`, `add_update_requested`, and the broad
  `_add_or_update_shapes_layer(...)` path after narrower intents cover all
  behavior.

### Cleanup and compatibility

- Remove obsolete action hints that describe Add/Update behavior.
- Retain actionable load, ambiguity, and mutation feedback.
- Confirm that primary Labels and Shapes remain discoverable by annotation and
  analysis widgets.
- Update the continuous colorbar roadmap implementation to target live styled
  rows rather than extending the soon-to-be-removed Add/Update request
  dataclasses.

## Test plan

### Adapter tests

- Primary and styled registration emit the general membership invalidation.
- Removal through napari unregisters the binding and emits invalidation.
- Exact removal deletes only the requested styled variant.
- Multiple different styled variants are returned in napari order.
- Duplicate primary or duplicate styled identities are reported.
- Existing primary-specific signal behavior remains unchanged.

### Row tests

- Initial eye state comes from `layer.visible`.
- A Harpy eye request changes napari visibility once.
- A napari visibility event refreshes the eye without a feedback request.
- Labels and Shapes presentation events refresh their previews.
- Disposal disconnects native event callbacks.
- A stale row intent cannot target a replacement binding.

### Viewer tests

- Existing live primary and styled layers hydrate when cards are built.
- An absent primary shows **Load in viewer** and a loaded primary shows its
  live row.
- Accepting a valid source loads a styled variant without another button.
- Multiple styled variants appear and remain independent.
- Harpy removal and napari removal reconcile in both directions.
- Point-backed Shapes variants receive the same membership and visibility UX.
- Invalid and ambiguous membership disables mutation without selecting an
  arbitrary layer.
- No Labels or Shapes card exposes **Add / Update in viewer**.

## Completion criteria

This work is complete when Labels and Shapes cards accurately represent live
napari membership, visibility, removal, and supported presentation state;
styled variants can be composed without a staged Add/Update action; primary
annotation and analysis workflows remain intact; and focused adapter, row, and
Viewer tests cover both Harpy-to-napari and napari-to-Harpy changes.
