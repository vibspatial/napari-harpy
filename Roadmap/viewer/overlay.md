# Viewer Overlay Composer and Napari Synchronization

Date: 2026-07-29
Updated: 2026-07-30

## Decision

Replace the overlay mode's full checkbox-and-color list with a **searchable,
live overlay composer**.

The composer is a synchronized control surface for Harpy-managed napari image
layers:

| Viewer widget state | Napari state |
| --- | --- |
| Channel appears in the overlay composer | Corresponding overlay layer exists |
| Open or closed eye | `layer.visible` |
| Color/colormap swatch | `layer.colormap` |
| Remove channel | Corresponding layer is deleted |
| Layer deleted in napari | Channel disappears from the composer |

Once a channel is loaded, the live napari layer is the source of truth for
existence, visibility, and colormap. The viewer widget must reflect changes made
from either location.

The image `Add / Update in viewer` action will be removed after live
synchronization is complete. Overlay changes will no longer have a separate
pending state.

## Problems being solved

### Channel discovery does not scale

The current overlay UI creates one checkbox and one color button for every
channel. Only five rows are visible, so a user with many channels must
repeatedly scroll and select channels one by one.

Color controls are also shown for channels that are not part of the overlay,
giving selected and unselected channels the same visual weight.

### Viewer state can diverge

Image layers can be controlled both from the Harpy viewer widget and napari's
native layer list. At present:

- visibility changes in napari are not reflected in the Harpy controls;
- Harpy color choices and napari colormap changes can diverge;
- deleting a layer in napari does not update the channel-selection controls;
- `Add / Update in viewer` suggests a staged form even though napari remains an
  independently editable viewer.

This creates two apparent sources of truth.

## UX model

Three concepts must remain distinct:

1. **Membership**: whether an overlay channel layer exists in napari.
2. **Visibility**: whether an existing layer is currently shown.
3. **Removal**: whether the layer should be deleted.

An eye controls visibility only. Closing an eye must not remove the layer or
discard its color.

Removing a selected row deletes the corresponding napari layer. Deleting the
layer from napari removes the selected row. The source image data is never
deleted.

## Target UI

```text
Display mode     (•) Stack    ( ) Overlay

Channels in viewer                                  3 channels
[ Search or add channels...                                  ]

[eye] [cyan]       DAPI                                  [×]
[eye] [magenta]    CD3                                   [×]
[eye] [yellow]     PanCK                                 [×]

[Remove all]
```

Detailed behavior:

- `Stack` and `Overlay` are presented as mutually exclusive choices.
- Entering overlay mode reveals the overlay composer.
- Clicking or focusing `Search or add channels...` opens a browseable popup.
- Typing filters channel names case-insensitively using substring matching.
- Activating a result immediately creates and shows its napari overlay layer.
- The input clears after a successful add and remains ready for another
  selection.
- A failed load leaves the composer unchanged and shows actionable feedback.
- Already loaded channels are omitted from the search results.
- Only channels with live overlay layers are shown as persistent rows.
- Clicking an eye immediately updates `layer.visible`.
- Toggling the native napari eye immediately updates the Harpy eye.
- Clicking a color swatch opens the color picker and immediately updates
  `layer.colormap`.
- Changing the colormap in napari immediately updates the Harpy swatch.
- Clicking `×` immediately removes that layer from napari.
- Deleting a layer in napari immediately removes its composer row.
- `Remove all` removes all overlay layers for that image and coordinate system.
- The selected-channel area shows up to five rows before scrolling.

## State ownership

### Napari owns applied layer state

For a loaded channel, derive the UI from the live layer and its
`ImageLayerBinding`:

- channel identity from `channel_index` and `channel_name`;
- membership from the binding and live layer list;
- visibility from `layer.visible`;
- color or colormap from `layer.colormap`.

Do not maintain a second applied-state copy inside the image card.

### The image card owns discovery and preferences

The image card may retain:

- available channel names and indices;
- a small last-used color mapping keyed by channel index.

The last-used color mapping is a preference, not applied state. It allows a
removed channel to recover its previous custom solid color if it is added again
while the card remains alive.

### Synchronization and signal ownership

Keep these three event types deliberately separate:

| Event | Meaning | Emitter |
| --- | --- | --- |
| `ViewerAdapter.image_overlay_layers_changed` | Overlay membership may have changed | `ViewerAdapter`, after layer insertion or removal |
| `color_selected` or a row-level change request | The user expressed an intent in Harpy | The Harpy control or channel row |
| `layer.events.colormap` / `layer.events.visible` | The authoritative live layer property changed | napari, after its layer property is assigned |

The lifecycle signal tells Histogram and Viewer widgets to re-query matching
bindings. It does not report visibility or colormap changes.

A Harpy user-intent signal is an internal UI request, not a state-change
notification. `ViewerWidget` handles the request, resolves or validates the
current live layer, and assigns `layer.colormap` or `layer.visible`.

Harpy must never manually call `layer.events.colormap.emit()` or
`layer.events.visible.emit()`. Assigning the napari property causes napari to
emit its native event. Consumers then read the accepted value back from the
live layer.

The Histogram and Viewer widgets are peer consumers. Neither widget emits
presentation events for the other, and the adapter does not mirror napari
property events through a second signal.

Multiple callbacks listening to one napari layer event are expected observer
behavior. For example, a colormap change can update both a Histogram card and a
Viewer channel row from the same `layer.events.colormap` emission.

Each consumer owns and disconnects its callbacks when its row/card target
changes or disappears. When updating Qt controls in response to napari, block
their signals so reflection does not trigger the same mutation again.

## Scope

In scope:

- Viewer image cards in overlay mode.
- Searchable channel discovery.
- Immediate add and removal of overlay channel layers.
- Bidirectional per-channel visibility synchronization.
- Bidirectional per-channel color/colormap synchronization.
- Live Histogram-to-napari color editing when a matching overlay is loaded.
- Selected-channel count, empty state, and remove-all action.
- Display-mode semantics and removal of the ambiguous image
  `Add / Update in viewer` action.
- Focused adapter, widget, and integration tests.

Out of scope:

- Synchronizing opacity, contrast limits, gamma, blending, or layer order.
- A single aggregate eye for the complete overlay.
- Presets, favorites, drag-to-reorder, or channel grouping.
- Persisting overlay configurations between application sessions.
- A hard maximum on selected channels.
- Reworking Feature Extraction channel selection.
- Synchronizing external image layers that do not have a Harpy
  `ImageLayerBinding`.
- A general rewrite of viewer-layer state management.
- A central adapter event hub that mirrors arbitrary napari layer properties.
- Widget-to-widget presentation events.

## Implementation constraints

Keep the design small and reuse existing infrastructure:

- Use the existing `LayerBindingRegistry` and `ImageLayerBinding`.
- Use `ViewerAdapter.get_loaded_image_layers(...)` to re-query live state.
- Keep `image_overlay_layers_changed` focused on overlay-layer lifecycle.
- Extend the adapter only with focused one-channel removal.
- Subscribe directly to napari layer property events from the widget row/card
  that presents those properties.
- Use the existing `CompleterPopupLineEdit`.
- Use `QStringListModel` and `QCompleter` for channel search.
- Reuse and modestly extend `OverlayColorButton`.
- Keep live visibility and colormap mutations in `ViewerWidget`; use
  `ViewerAdapter` for binding/lifecycle, load, and removal operations.
- Image-card controls emit user intent and render synchronized state.

Do not introduce a new overlay controller, custom `QAbstractItemModel`, delegate
framework, or application-wide state abstraction. Revisit model/view
virtualization only if profiling demonstrates a real channel-count problem.

Do not extract a generic layer-event subscription framework before the Viewer
row implementation demonstrates meaningful duplication with the Histogram.
The Histogram's existing direct subscriptions are the reference pattern.

## Slice 1: Adapter lifecycle and focused removal foundation

Status: completed and reconciled on 2026-07-30.

### Goal

Keep image-layer identity and lifecycle in `ViewerAdapter`, and add a focused
operation for removing one overlay channel.

This is an adapter-only slice. It must not redesign the image card or add eye
and colormap controls. It must not make the adapter a mirror for napari layer
property events.

### Target files

- `src/napari_harpy/viewer/adapter.py`
- `tests/test_viewer_adapter.py`

### Existing contracts to preserve

- `LayerBindingRegistry` remains the source of Harpy image and channel
  identity.
- `ImageLayerBinding` remains the binding type for stack and overlay image
  layers.
- `image_overlay_layers_changed` continues to mean that the set/order of
  histogram-usable overlay bindings changed.
- Consumers of `image_overlay_layers_changed` continue to re-query bindings.
- `get_loaded_image_layers(...)` remains the public read path for live image
  layers belonging to one SpatialData image.
- Layer-list insertion and removal continue to be observed through napari's
  layer-list events.
- `_remove_layer_from_viewer_and_registry(...)` remains the central
  Harpy-initiated removal path and retains its fallback for viewer-like objects
  that do not emit a removal event.

### Event boundary

- `image_overlay_layers_changed` remains the structural notification for
  histogram-usable overlay bindings appearing or disappearing.
- The adapter does not subscribe to `layer.events.visible`,
  `layer.events.colormap`, or other presentation properties on behalf of
  widgets.
- The adapter does not define or emit `image_layer_presentation_changed`.
- Histogram and Viewer consumers subscribe directly to the relevant napari
  layer properties after resolving a binding.
- `unregister_layer(...)` remains responsible only for binding-registry
  cleanup.

The initially implemented adapter presentation signal, callback registry, and
associated tests have been removed. The focused channel-removal operation and
its lifecycle coverage remain in place.

### Focused removal operation

The adapter exposes:

```python
def remove_image_overlay_channel(
    self,
    sdata: SpatialData,
    image_name: str,
    coordinate_system: str,
    *,
    channel_index: int,
) -> Image | None:
    ...
```

Method contract:

- Match by all four identity components:
  - SpatialData object identity;
  - image element name;
  - coordinate system;
  - overlay channel index.
- Match only `ImageLayerBinding` instances whose
  `image_display_mode == "overlay"`.
- Never remove a stack layer.
- Never remove a sibling overlay channel.
- Use `_remove_layer_from_viewer_and_registry(...)` for the actual removal.
- Return the removed `Image`.
- Return `None` when no matching live layer exists.
- Treat an absent layer as a normal idempotent no-op; do not log it as an error.
- Reject a negative `channel_index` with `ValueError`.
- The binding registry is expected to contain at most one live match for this
  identity. If that invariant is violated, raise `ValueError` rather than
  silently choosing or deleting multiple layers.

On a normal napari viewer, successful removal should cause exactly one
`image_overlay_layers_changed` emission through the existing layer-list removal
handler. The fallback path may emit it when the viewer does not emit removal,
but the two paths must not double-emit.

### Failure and logging behavior

- Existing warnings for malformed napari layer-list event payloads remain
  unchanged.
- Removing a missing overlay channel is not exceptional and does not produce a
  warning.
- Invalid or duplicate focused-removal matches fail before any layer is
  removed.

### Acceptance criteria

- Focused channel removal matches the full image/channel identity and preserves
  all non-matching layers.
- Missing-channel removal returns `None` without changing state.
- Existing `image_overlay_layers_changed` consumers retain their lifecycle
  behavior.
- The adapter has no presentation-property subscriptions or mirrored
  presentation signal.
- No viewer-widget or image-card behavior changes in this slice.

### Focused tests

Focused tests cover:

#### Focused channel removal

- Removes the requested overlay channel.
- Preserves sibling overlay channels.
- Preserves a stack layer with the same image identity.
- Preserves layers for another image, coordinate system, or SpatialData object.
- Returns the removed layer.
- Returns `None` for a missing channel.
- Rejects a negative channel index.
- Rejects duplicate live matches for the same overlay identity.
- Emits the existing overlay lifecycle signal exactly once on normal removal.
- Keeps registry state correct for both normal and fallback viewer behavior.

#### Regression coverage

- Existing image registration tests.
- Existing layer-list insertion/removal tests.
- Existing `image_overlay_layers_changed` tests.
- Existing focused Histogram lifecycle tests continue to pass without a
  presentation signal.

### Slice 1 completion criteria

Slice 1 is complete:

- the adapter presentation signal and property subscriptions are removed;
- one overlay channel can be removed safely by full identity;
- existing overlay lifecycle and histogram behavior remain unchanged;
- focused adapter tests pass;
- no viewer UI files changed.

## Slice 2: Searchable composer and live layer membership

### Goal

Replace the all-channel checkbox list with a searchable, selected-only
composer. Make channel membership bidirectional and live.

### Target files

- `src/napari_harpy/widgets/viewer/image_widget.py`
- `src/napari_harpy/widgets/viewer/widget.py`
- `tests/test_viewer_widget.py`
- `tests/test_feature_extraction_widget.py` where tests currently inspect
  viewer overlay checkboxes

### Search behavior

- Add a `CompleterPopupLineEdit` with placeholder text
  `Search or add channels`.
- Configure `QCompleter` with:
  - popup completion;
  - at most 10 visible items;
  - case-insensitive matching;
  - `Qt.MatchContains`.
- Preserve original dataset order in available search results.
- Open the popup on focus/click with an empty prefix.
- Add a channel through completer activation or Return on an exact valid name.
- Reject unknown text without changing napari or composer state.
- Exclude channels that already have a live overlay layer.
- Clear the input only after a successful add.

### Membership behavior

- Add a channel with the existing focused
  `ensure_image_overlay_channel_loaded(...)` adapter operation.
- Show the selected row only after the layer is successfully loaded and bound.
- Render selected rows from a fresh adapter query.
- Remove a channel using the focused one-channel adapter removal operation.
- Connect `ViewerWidget` once to the existing
  `image_overlay_layers_changed` lifecycle signal.
- On that signal, re-query the active SpatialData image and coordinate system,
  then reconcile selected-row membership against the live bindings.
- Dispose obsolete rows before destroying them. Property-event callbacks are
  added in Slice 3a and must be disconnected as part of row disposal.
- Do not use presentation-property events to infer membership. A hidden layer
  remains loaded; only binding/layer insertion and removal change membership.
- Implement `Remove all` using the adapter, scoped to the active SpatialData
  image and coordinate system.
- Keep the visible selected rows in live napari layer order unless a later
  usability test establishes a strong reason to maintain a separate order.
- Show `No channels in viewer` when no overlay layers exist.
- Show a live count such as `3 channels`.
- Limit the selected-row viewport to five rows before scrolling.

### Default color behavior

- For a newly added channel, choose the first unused color from
  `DEFAULT_OVERLAY_COLORS`.
- Choose defaults from currently loaded sibling colors rather than absolute
  channel index, so a small overlay starts with distinct colors.
- Cycle through the palette only after every default color is in use.
- Prefer the card's cached last-used solid color when re-adding a channel.

### Acceptance criteria

- A user can find and load a channel without navigating a long persistent list.
- Only channels represented by live Harpy overlay layers occupy selected rows.
- Search results never offer an already loaded channel.
- Add from Harpy creates a layer and napari-side delete removes the row.
- Remove from Harpy deletes the layer and restores the channel to search.
- Remove all affects only the matching image and coordinate system.
- A failed load does not create a misleading selected row.
- Many available channels do not create many persistent row widgets.
- One lifecycle notification performs one membership reconciliation; it does
  not create duplicate signal connections.

### Focused tests

- Empty composer state and live count.
- Popup configuration and substring filtering.
- Add by completer activation and exact Return.
- Unknown and duplicate input handling.
- Add success and load failure.
- Napari-side removal updates the composer.
- Harpy-side remove preserves sibling layers.
- Remove all scoping.
- Search results refresh after add and removal.
- Default color selection and cached color reuse.
- Many available channels do not produce persistent rows.
- Repeated lifecycle refreshes do not duplicate the Viewer callback.

## Slice 3a: Viewer bidirectional visibility and colormap controls

### Goal

Add per-channel eye and colormap controls that remain synchronized with native
napari controls.

### Target files

- `src/napari_harpy/widgets/viewer/image_widget.py`
- `src/napari_harpy/widgets/viewer/widget.py`
- `src/napari_harpy/widgets/overlay_color_button.py`
- `tests/test_viewer_widget.py`
- focused color-button tests if a separate test module is clearer

### Direct layer-event subscriptions

Each selected channel row owns its presentation subscriptions for its current
live napari `Image`:

- connect directly to `layer.events.visible`;
- connect directly to `layer.events.colormap`;
- update only the row bound to that layer when either event fires;
- verify that the callback still belongs to the row's current layer before
  applying state.

The Viewer must not receive these events through the Histogram or through an
adapter presentation signal. The Histogram keeps its own direct subscriptions.
A single napari colormap event therefore fans out naturally to both peer
widgets when both are open.

Connection ownership must be explicit and idempotent:

- connect each event at most once for a row/layer pair;
- disconnect both callbacks before a row is removed or rebound;
- disconnect before rebuilding card content or changing SpatialData, image, or
  coordinate-system context;
- disconnect during widget teardown where applicable;
- tolerate an already-disconnected or already-destroyed event source using the
  same narrow exception handling pattern as the Histogram.

Start with small Viewer-owned connection and disconnection methods. Do not
extract a shared Histogram/Viewer subscription abstraction in this slice.
Consider that only later if the completed implementations contain meaningful,
stable duplication.

### User intent and mutation ownership

Modestly extend `OverlayColorButton` with a user-only signal:

```python
color_selected = Signal(str)
```

- Emit `color_selected` once when the user accepts a valid color in the picker.
- Keep `set_color(...)` silent. It is the rendering API used for
  napari-originated updates and must not emit user intent.
- Slice 3a leaves existing consumers such as Histogram unchanged. Slice 3b
  explicitly connects Histogram to the new signal.

The selected channel row translates control interaction into an intent that
contains the channel identity and requested value. `ViewerWidget` owns the
presentation mutation:

1. Resolve or revalidate the current live `ImageLayerBinding`.
2. Return safely if the layer disappeared during the interaction.
3. Compare the requested value with the current live property.
4. Assign `layer.colormap` or `layer.visible` only when it differs.
5. Let napari emit `layer.events.colormap` or `layer.events.visible`.
6. Let the row's direct callback read the accepted property and render it.

Do not add generic adapter methods or signals for these presentation
mutations. The adapter remains responsible for binding identity, lifecycle,
loading, and focused removal.

The expected colormap flows are:

```text
Harpy color selection
  -> Harpy user-intent signal
  -> ViewerWidget assigns layer.colormap
  -> napari emits layer.events.colormap
  -> Viewer row and Histogram card independently refresh from the layer

napari color selection
  -> napari assigns layer.colormap
  -> napari emits layer.events.colormap
  -> Viewer row and Histogram card independently refresh from the layer
```

Visibility follows the same pattern with the eye control,
`layer.visible`, and `layer.events.visible`.

### Visibility behavior

- Add one eye control per selected channel row.
- Initialize it from `layer.visible`.
- User interaction updates `layer.visible` immediately.
- Napari `visible` events update the eye without recreating unrelated rows.
- Closing the eye leaves the row, layer, and color intact.
- Do not add an aggregate image-level eye in this slice.

### Colormap behavior

- Initialize each swatch from the live `layer.colormap`.
- Choosing a solid color in Harpy updates the layer immediately.
- A napari colormap change updates the Harpy swatch immediately.
- Normalize solid colors to a consistent hex representation.
- Preserve the existing accessible color name and hex value for solid colors.

Napari can assign a non-solid colormap such as Viridis, while the current Harpy
control assumes a solid tint. Handle this honestly:

- show a compact gradient preview for a non-solid colormap;
- expose its colormap name in the tooltip and accessible name;
- clicking the preview may still open the existing solid-color picker;
- accepting a solid color replaces the non-solid napari colormap.

Do not build a second full colormap picker in Harpy.

### Feedback-loop protection

- Keep programmatic `OverlayColorButton.set_color(...)` silent.
- Block the eye control's Qt signals while applying napari-originated state.
- Avoid writing a layer property when the requested value is already current.
- After a Harpy-originated mutation, read the accepted property back from the
  live layer rather than assuming the setter succeeded.
- Keep lifecycle reconciliation separate from property reflection: visibility
  and colormap events update their row, not the complete card membership.
- Never manually emit a napari layer property event.

### Acceptance criteria

- Harpy and napari eyes always represent the same visibility state.
- Hidden channels remain members of the overlay.
- Solid-color changes round-trip in both directions.
- Non-solid napari colormaps are represented without pretending they are a
  solid color.
- Repeated synchronization does not recurse or duplicate mutations.
- Changes to one layer do not disturb sibling rows.
- Viewer and Histogram can observe the same layer property independently
  without forwarding events through one another or the adapter.
- Removing or rebuilding a row leaves no callback targeting the stale row.
- The adapter remains free of visibility and colormap subscriptions.
- Harpy never manually emits napari's colormap or visibility events.
- User-intent signals and napari property-change events remain semantically
  distinct.

### Focused tests

- Initial eye state from the layer.
- Harpy-to-napari and napari-to-Harpy visibility changes.
- Hidden layer remains selected.
- Initial solid swatch from the layer.
- Harpy-to-napari and napari-to-Harpy solid-color changes.
- `color_selected` emits once for an accepted user choice.
- Programmatic `set_color(...)` emits no user-intent signal.
- One Harpy color choice produces one layer mutation and then reflects through
  napari's native colormap event.
- Non-solid colormap preview, tooltip, and accessible name.
- Signal blocking/re-entrancy protection.
- Direct callback connection is idempotent for a row/layer pair.
- Property-event callback cleanup after removal, rebinding, and card refresh.
- A stale callback cannot update a row after its target layer changes.
- One layer colormap change can update both open peer widgets without an
  adapter presentation signal.

## Slice 3b: Histogram live colormap editing for loaded overlays

### Dependency

Slice 3a is complete, including the user-only
`OverlayColorButton.color_selected` signal and silent `set_color(...)`
contract.

### Goal

Make the Histogram color picker update an already loaded, uniquely matched
napari overlay immediately. Preserve `Load overlay` as the explicit creation
path when no matching layer exists.

This is a focused Histogram integration slice. It must not add adapter
presentation signals or turn a color choice into implicit layer creation.

### Target files

- `src/napari_harpy/widgets/histogram/widget.py`
- `tests/test_histogram_widget.py`
- `src/napari_harpy/widgets/overlay_color_button.py` and its focused tests only
  if the Slice 3a signal contract needs a small correction

### User-intent connection

- Connect each Histogram card's `overlay_color_button.color_selected` signal
  exactly once when the card is created.
- Pass the stable card id and requested solid color to a Histogram-owned
  handler.
- Keep existing napari-to-Histogram synchronization connected directly to
  `layer.events.colormap`.
- Keep programmatic `set_color(...)` silent so a napari-originated refresh does
  not re-enter the user-intent handler.

### Loaded and unloaded behavior

On an accepted Histogram color choice:

1. Resolve the card's current image, channel, coordinate system, and matching
   overlay binding using the existing unique-binding path.
2. If exactly one live matching overlay exists:
   - verify that it is still the card's current target;
   - compare the normalized requested solid color with the live colormap;
   - assign `layer.colormap` once only when the value differs;
   - let napari emit `layer.events.colormap`;
   - let the existing direct callback read the accepted value and call the
     silent `set_color(...)` rendering method.
3. If no matching overlay exists:
   - do not create a layer;
   - keep the selected color in the button as the pending load color;
   - let the existing `Load overlay` action create the channel later using that
     color.
4. If matching is ambiguous, the target changed, or the layer disappeared:
   - do not mutate any layer;
   - do not guess which layer the user intended.

If a bound overlay is later removed, disconnect through the existing Histogram
cleanup path and retain the last displayed solid color as the pending color for
a possible later load.

### Event flow

```text
Histogram user chooses a color
  -> OverlayColorButton emits color_selected
  -> Histogram resolves one live matching overlay
  -> Histogram assigns layer.colormap
  -> napari emits layer.events.colormap
  -> Histogram and Viewer independently refresh from the live layer
```

No Histogram code manually emits a napari property event. No adapter method is
added for this presentation mutation.

### Failure and non-solid behavior

- If the napari assignment fails, restore the swatch from the current live
  layer and surface concise feedback through the existing Histogram status
  area.
- A solid Histogram choice may replace a live non-solid colormap.
- Reuse the shared colormap preview capability delivered by Slice 3a where it
  applies. Do not expand Slice 3b into a second colormap-picker redesign.

### Acceptance criteria

- A Histogram color choice immediately updates one uniquely matched loaded
  overlay.
- A Histogram color choice never creates an overlay implicitly.
- Without a loaded overlay, the chosen color remains available to the explicit
  `Load overlay` action.
- Napari remains the authoritative presentation source after assignment.
- Viewer and Histogram can both observe the resulting native napari event.
- Missing, stale, and ambiguous targets produce no layer mutation.
- Repeated synchronization neither recurses nor duplicates assignments.
- No adapter presentation signal or property setter is introduced.

### Focused tests

- A loaded unique overlay is updated immediately from a Histogram color choice.
- Choosing the already active solid color produces no layer assignment.
- The resulting native napari event refreshes the Histogram without a second
  user-intent emission.
- No loaded overlay preserves the pending color and creates nothing.
- The pending color is passed to the existing explicit load operation.
- Ambiguous matching overlays are not mutated.
- Removal or target change prevents a stale layer mutation.
- Assignment failure restores the live color and reports feedback.
- Existing napari-to-Histogram colormap synchronization and callback-cleanup
  tests continue to pass.

## Slice 4: Display-mode lifecycle and removal of staged apply

### Goal

Complete the live mental model and remove `Add / Update in viewer` without
automatically loading every image card.

### Target files

- `src/napari_harpy/widgets/viewer/image_widget.py`
- `src/napari_harpy/widgets/viewer/widget.py`
- `tests/test_viewer_widget.py`

### Mode behavior

- Replace the two mutually managed checkboxes with `QRadioButton` controls in a
  `QButtonGroup`, or an existing simple segmented style if one is already
  available.
- Exactly one editor mode is selected.
- Do not load an image merely because its card was created with `Stack` as the
  default editor mode.
- In overlay mode, adding the first channel performs the existing stack-to-
  overlay replacement.
- When live overlay layers exist, the card opens in overlay mode.
- When a live stack layer exists, the card opens in stack mode.
- If all overlay layers are deleted, keep the empty overlay composer visible;
  do not silently load or switch to stack.

### Initial stack loading

Stack mode still needs one explicit initial load action. Replace the ambiguous
generic action with a contextual `Load stack` action shown only when:

- stack mode is selected; and
- no matching stack layer is currently loaded.

After the stack layer exists, its state is live and there is no Update action.
If stack visibility/removal controls are later desired, they should follow the
same existence/eye/remove semantics, but that extension is not required for the
overlay roadmap.

### Remove staged overlay apply

- Remove `Add / Update in viewer` from overlay mode.
- Remove overlay request-building code that exists only to apply the complete
  checkbox list.
- Keep adapter validation as a defensive boundary.
- Do not remove existing layers in response to an invalid empty request.

### Acceptance criteria

- Display modes have proper mutually exclusive semantics.
- Creating or refreshing image cards does not load images automatically.
- Overlay additions, removals, visibility, and color are all live.
- There is no overlay Apply or Update action.
- The only initial-load action that remains is clearly labelled `Load stack`
  and appears only when needed.
- Card mode is hydrated correctly from existing registered layers.

### Focused tests

- Default mode without automatic loading.
- Mode hydration from existing stack and overlay bindings.
- Mutually exclusive mode controls.
- First overlay add replaces a loaded stack.
- Stack load replaces loaded overlays using existing adapter semantics.
- Empty overlay remains empty after all layers are deleted.
- Overlay mode has no Add/Update action.
- Contextual `Load stack` visibility and behavior.

## Slice 5: Professional UI polish and regression coverage

### Goal

Ensure the finished interaction is compact, accessible, and consistent with
the rest of the viewer widget.

### Target files

- `src/napari_harpy/widgets/viewer/image_widget.py`
- shared viewer styles only if a small reusable style is genuinely useful
- focused viewer tests

### Implementation

- Keep spacing, control heights, borders, hover states, and typography
  consistent with existing viewer cards.
- Elide long channel names and expose the full name in a tooltip.
- Give search, eye, colormap, remove, remove-all, mode, and count controls
  appropriate accessible names.
- Ensure every action is keyboard reachable.
- Keep visible focus indication for every interactive control.
- Use an established open-eye/closed-eye icon pair available in the project or
  Qt theme; do not use color alone for visibility.
- Use icon-plus-tooltip or a compact `×` for removal.
- Show a non-blocking note when more than eight channels are loaded:
  `Many overlaid channels may be difficult to distinguish.`
- Hide the note when the count returns to eight or fewer.
- Disable the relevant row briefly while a mutation is in progress if the
  operation can re-enter through napari events.
- Surface load/update failures through the existing viewer feedback area.
- Verify layout at narrow napari dock widths and with long channel names.

### Acceptance criteria

- The workflow can be completed using the keyboard.
- Long names do not widen the dock or require horizontal scrolling.
- Empty, populated, hidden, loading, error, non-solid-colormap, many-channel,
  no-channel-axis, and duplicate-channel states are understandable.
- Visibility and removal have distinct icons, tooltips, and behavior.
- The many-channel note informs but never blocks.
- The interface remains responsive and visually stable during napari-originated
  updates.

### Focused tests

- Long-name elision tooltip.
- Accessible names on interactive controls.
- Eye and remove controls remain semantically distinct.
- Warning threshold at 8/9 channels.
- Five-row viewport height and scrolling.
- Mutation failure feedback.
- Existing no-channel-axis and duplicate-channel error states.

## Manual verification

After focused automated tests pass, verify in napari with:

- an image with 3 channels;
- an image with approximately 30 channels;
- long and similar channel names;
- more than 8 loaded overlay channels;
- a customized color followed by remove and re-add;
- solid color changes from the Viewer, Histogram, and napari;
- a Histogram color chosen before loading, followed by explicit overlay load;
- a Histogram color changed after its matching overlay is loaded;
- a non-solid colormap selected from napari;
- eyes toggled from both Harpy and napari;
- layers deleted individually and in bulk from both locations;
- repeated stack/overlay transitions;
- coordinate-system changes while image layers are loaded;
- widget/card refresh while registered layers already exist.

Check mouse and keyboard workflows, narrow dock sizing, dark-theme contrast,
focus indication, color-picker behavior, layer identity, sibling preservation,
and callback cleanup after removal.

## Completion criteria

This roadmap is complete when:

- channels are discovered through a searchable popup;
- selected rows correspond exactly to live Harpy overlay layers;
- add, remove, and remove-all operations update napari immediately;
- eyes and colormaps synchronize in both directions;
- a Histogram picker edits an existing matching overlay live but never creates
  one implicitly;
- presentation synchronization comes directly from live napari layer events;
- external napari layer deletion updates the composer;
- membership, visibility, and removal remain distinct;
- the adapter remains responsible for binding/lifecycle operations rather than
  mirroring napari presentation properties;
- the overlay `Add / Update in viewer` action is gone;
- initial stack loading remains explicit and clearly labelled;
- focused automated tests and manual verification pass.
