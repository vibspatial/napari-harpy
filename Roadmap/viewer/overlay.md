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

[eye] DAPI                                  [cyan]       [×]
[eye] CD3                                   [magenta]    [×]
[eye] PanCK                                 [yellow]     [×]

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

The Histogram card uses a related but intentionally contextual Viewer area:

```text
Channel
[ Search or select channel: DAPI                         ]

Viewer — no matching overlay
[ Load in viewer ]                            [cyan]

Viewer — unique matching overlay
[eye] DAPI                                    [cyan]     [×]
```

The accepted Histogram channel remains visible in the search field because it
is persistent analysis-target state. Selecting another Histogram channel
replaces only that card's contextual Viewer row; it does not create or remove a
napari layer.

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
- Searchable Histogram channel targeting with explicit overlay creation and a
  contextual live overlay row for the current target.
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
- Keep live visibility and colormap mutations in the owning Viewer or Histogram
  widget; use `ViewerAdapter` for binding/lifecycle, load, and removal
  operations.
- Viewer image-card and Histogram contextual controls emit user intent and
  render synchronized state.

Do not introduce a new overlay controller, custom `QAbstractItemModel`, delegate
framework, or application-wide state abstraction. Revisit model/view
virtualization only if profiling demonstrates a real channel-count problem.

Do not extract a generic layer-event subscription framework. Slice 3b may
extract the now-demonstrated shared overlay-row and colormap-presentation
primitives, while Viewer and Histogram retain ownership of their own target
resolution and mutations.

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

Status: implemented on 2026-07-30.

### Goal

Replace the all-channel checkbox list with a searchable, selected-only
composer. Make channel membership bidirectional and live.

This slice changes overlay membership only. Visibility and live colormap
editing are added in Slice 3a.

### Target files

- `src/napari_harpy/widgets/viewer/image_widget.py`
- `src/napari_harpy/widgets/viewer/widget.py`
- `tests/test_viewer_widget.py`
- `tests/test_feature_extraction_widget.py` where tests currently inspect
  viewer overlay checkboxes

No adapter change is expected. Slice 2 uses the focused add and removal
operations completed in Slice 1.

### Ownership boundary

`ViewerWidget` owns:

- querying the adapter and binding registry;
- filtering live bindings to the active SpatialData image and coordinate
  system;
- adding and removing napari layers;
- handling mutation failures through the existing Viewer feedback area;
- responding once to overlay lifecycle notifications.

`_ImageCardWidget` owns:

- available channel discovery for its image;
- the searchable completer model;
- selected-row creation, reuse, ordering, and disposal;
- selected count, empty state, and remove-all controls;
- a card-lifetime cache of last-used solid colors keyed by channel index;
- emitting user intent without mutating napari.

Do not introduce an overlay controller or custom item model. Keep the existing
`ViewerWidget` plus image-card division.

### Card intent signals

Replace the aggregate overlay checkbox request with focused image-card intent
signals:

```python
overlay_channel_add_requested = Signal(str, int, str)
overlay_channel_remove_requested = Signal(str, int)
overlay_channels_remove_all_requested = Signal(str)
```

Signal values are:

- image name;
- channel index where applicable;
- requested initial solid color for addition.

Use channel index as the mutation identity. Channel name remains display
metadata and search input, not the removal key.

`ViewerWidget` connects each signal once when it creates the card. The card
does not call `ViewerAdapter` directly.

Expose one narrow card completion method for the synchronous add request:

```python
def finish_overlay_channel_add(
    self,
    channel_index: int,
    *,
    succeeded: bool,
) -> None:
    ...
```

On success, clear the input only if it still resolves to that channel. On
failure, preserve the input. This keeps mutation ownership in `ViewerWidget`
without making the card guess whether the adapter call succeeded.

### Live binding resolution

Add one private `ViewerWidget` helper that returns ordered live overlay
bindings for one current image.

Resolution contract:

1. Call `ViewerAdapter.get_loaded_image_layers(sdata, image_name)` so candidate
   layers retain current napari layer order.
2. Resolve each candidate through `LayerBindingRegistry.get_binding(...)`.
3. Retain only `ImageLayerBinding` instances matching:
   - the active SpatialData object identity;
   - the card's image name;
   - the active coordinate system;
   - `image_display_mode == "overlay"`;
   - a non-negative channel index;
   - a non-empty channel name.
4. Return bindings in candidate layer order.

Do not use the registry's insertion order as a substitute for napari layer
order. Do not add another public adapter query for this slice.

Duplicate live bindings for one channel index violate the adapter invariant.
Do not silently choose one. Put that card into a non-mutating membership-error
state, disable its composer actions, and surface concise feedback. A later
lifecycle reconciliation may recover after the external duplicate is removed.

Layer reordering itself remains outside this roadmap's synchronization scope.
The composer adopts current napari order whenever a membership reconciliation
occurs; it does not add a new reorder listener.

### Card membership rendering

Add an image-card rendering entry point:

```python
def set_loaded_overlay_bindings(
    self,
    bindings: Sequence[ImageLayerBinding],
) -> None:
    ...
```

Rendering requirements:

- Treat the supplied bindings as the complete selected membership state.
- Key selected rows by channel index.
- Reuse a row when the same channel remains loaded.
- Create rows only for newly loaded channels.
- Before removing an obsolete row, retain its current solid layer color in the
  card cache and run its disposal hook.
- Reorder retained and new rows to match the supplied binding order.
- Never create persistent row widgets for unselected channels.
- Derive the count and empty state from the rendered live rows.

Slice 2 selected rows contain only:

- the channel name;
- a remove action;
- the stable channel index and current binding as internal state.

Eye and editable colormap controls are deliberately deferred to Slice 3a. The
row disposal hook introduced here is the cleanup boundary Slice 3a will extend
with property-event disconnection.

### Search behavior

- Add a `CompleterPopupLineEdit` with placeholder text
  `Search or add channels`.
- Configure `QCompleter` with:
  - popup completion;
  - at most 10 visible items;
  - case-insensitive matching;
  - `Qt.MatchContains`.
- Preserve original dataset order in available search results.
- Enable the existing popup-on-entry behavior and open the popup on focus or
  click, including when the input is empty.
- Populate the `QStringListModel` only with channels that are not represented
  by a current selected row.
- Rebuild the model after every membership reconciliation.
- Add a channel through completer activation or Return.
- Completer activation uses the exact selected model item.
- Return first accepts an exact channel name. It may accept a case-insensitive
  match only when that match is unique; case-insensitive ambiguity is rejected.
- Trim surrounding whitespace before Return-key resolution.
- Reject empty, unknown, ambiguous, and already loaded input without changing
  napari or membership.
- Clear the input only after a successful add.
- Preserve the input after failure so the user can correct or retry it.

### Add behavior

On `overlay_channel_add_requested`:

1. Revalidate the active SpatialData object, coordinate system, image name, and
   channel index.
2. Re-query membership and return without mutation if the channel became
   loaded before the request was handled.
3. Call:

   ```python
   ensure_image_overlay_channel_loaded(
       sdata,
       image_name,
       coordinate_system,
       channel=channel_index,
       channel_color=requested_color,
   )
   ```

4. Do not add a row optimistically.
5. Let registration emit `image_overlay_layers_changed`.
6. Let lifecycle reconciliation create the row from the registered binding.
7. Clear the search input only after the adapter call succeeds.
8. On failure, leave membership and input unchanged and show Viewer feedback.

The first overlay addition may replace a loaded stack through the adapter's
existing stack-to-overlay behavior.

### Remove behavior

On `overlay_channel_remove_requested`:

1. Revalidate the active context and requested channel identity.
2. Cache the row's current solid color before mutation.
3. Call `remove_image_overlay_channel(...)`.
4. Do not remove the row optimistically.
5. Let napari removal and the adapter lifecycle signal drive reconciliation.
6. If the adapter returns `None`, explicitly reconcile once because no
   lifecycle signal is expected for an already absent layer.
7. On failure, keep the row and show Viewer feedback.

A napari-side deletion follows the same reconciliation path and must remove the
matching row without a Harpy request.

### Remove-all behavior

`Remove all` means all overlay channels for the card's active SpatialData
image and coordinate system. It must not remove a stack layer or layers for
another context.

Implementation contract:

1. Take a fresh ordered snapshot of matching live overlay bindings.
2. Cache each row's current solid color.
3. Call `remove_image_overlay_channel(...)` for each unique channel index in
   the snapshot.
4. Do not call `remove_image_layers(...)`, because that broader operation also
   removes stack layers.
5. Allow lifecycle reconciliation to update membership after each removal.
6. If a partial failure occurs, keep the remaining live rows and report concise
   feedback; never pretend the complete removal succeeded.

Do not add a bulk adapter method solely to reduce the number of lifecycle
notifications in this slice.

### Lifecycle connection and hydration

Connect `ViewerWidget` once during construction:

```python
self._app_state.viewer_adapter.image_overlay_layers_changed.connect(
    self._on_image_overlay_layers_changed
)
```

The handler:

- returns safely when SpatialData or coordinate system is absent;
- re-queries each current image card in the active context;
- calls `set_loaded_overlay_bindings(...)` on each card;
- does not rebuild the complete Viewer or recreate unrelated cards;
- does not connect another lifecycle callback.

When SpatialData or coordinate system changes, the existing card rebuild path
continues to run. Hydrate every newly created image card immediately from its
current live overlay bindings.

Property events do not determine membership. In particular,
`layer.visible = False` leaves the row loaded and selected.

### Composer presentation

- Show `No channels in viewer` when no overlay bindings are rendered.
- Show a live count such as `1 channel` or `3 channels`.
- Show `Remove all` only when at least one selected row exists.
- Limit the selected-row viewport to five rows before scrolling.
- Keep search available when the selected list is empty.
- Disable search when overlay is unavailable because channel discovery failed,
  including duplicate dataset channel names.

### Default color behavior

- For a newly added channel, choose the first unused color from
  `DEFAULT_OVERLAY_COLORS`.
- Choose defaults from currently loaded sibling colors rather than absolute
  channel index, so a small overlay starts with distinct colors.
- Cycle through the palette only after every default color is in use.
- Prefer the card's cached last-used solid color when re-adding a channel.
- Keep the cache keyed by channel index and scoped to the card lifetime.
- When an obsolete row is reconciled away, read its layer's current solid color
  before disposal so napari-side deletion also preserves a useful re-add
  preference.
- Do not overwrite a cached solid preference with a non-solid colormap.
- The cache is a preference only; selected-row membership and applied color
  continue to come from the live layer and binding.

### Transitional aggregate-action behavior

Slice 2 removes the aggregate overlay checkbox request path. The existing image
`Add / Update in viewer` action:

- remains temporarily available for stack mode only;
- is hidden or otherwise unavailable while overlay mode is active;
- never applies the complete overlay membership;
- retains only the stack request behavior needed until Slice 4.

Do not complete the radio-button and contextual `Load stack` redesign here.
Slice 4 owns that final mode cleanup and label change.

Existing `ImageLoadRequest` code may remain only where the temporary stack path
still needs it. Remove overlay-only request-building helpers that depend on the
old checkbox list.

### Failure behavior

- A failed add leaves the input and selected rows unchanged.
- A failed remove leaves the live row present.
- A successful remove clears the input when it still identifies the removed
  channel; a failed remove preserves that input.
- A missing remove target is an idempotent reconciliation case, not an error.
- Ambiguous live channel identity produces feedback and no guessed mutation.
- Invalid search input causes no adapter call.
- A duplicate live-binding invariant violation disables mutations for the
  affected card until a later valid reconciliation.
- Mutation feedback uses the existing Viewer feedback area; do not add modal
  dialogs for normal add/remove failures.

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
- Existing live overlays hydrate into the correct cards on initial render and
  coordinate-system change.
- Stack layers and hidden overlay layers are never mistaken for selected
  membership changes.
- Overlay mode no longer exposes an aggregate Apply or Update action.

### Focused tests

- Empty composer state, singular/plural live count, and conditional remove-all.
- Popup configuration and substring filtering.
- Popup-on-entry with an empty prefix.
- Add by completer activation, exact Return, and unique case-insensitive Return.
- Unknown, empty, case-insensitively ambiguous, and already loaded input.
- Add success, duplicate-race no-op, and load failure.
- Successful add clears input; failure preserves it.
- No optimistic row appears before a live binding exists.
- Napari-side removal updates the composer.
- Harpy-side remove preserves sibling layers.
- Missing focused removal explicitly reconciles without error.
- Remove all preserves stack layers and other images, coordinate systems, and
  SpatialData objects.
- Partial remove-all failure leaves truthful remaining rows.
- Search results refresh after add and removal.
- Default color selection and cached color reuse.
- Napari-side deletion caches the last live solid color for re-add.
- Non-solid colormap does not replace the cached solid preference.
- Many available channels do not produce persistent rows.
- Repeated lifecycle refreshes do not duplicate the Viewer callback.
- Lifecycle refresh updates image cards without rebuilding unrelated Viewer
  sections.
- Initial and coordinate-system-change hydration from existing overlay
  bindings.
- Hidden layer remains a selected row.
- Duplicate live channel bindings disable affected-card mutations without
  choosing or removing one.
- Aggregate image action is unavailable in overlay mode and the temporary stack
  path still works.
- Feature Extraction channel selection remains independent from Viewer overlay
  membership.

### Implementation result

Slice 2 is complete:

- overlay mode now uses a searchable completer and creates persistent rows only
  for live selected channels;
- channel activation loads immediately through the focused adapter operation;
- focused remove and remove-all actions are lifecycle-driven and preserve
  unrelated layers;
- cards hydrate and reconcile from live `ImageLayerBinding` state, including
  napari-side removal;
- the aggregate image action is hidden in overlay mode and remains a temporary
  stack-only action;
- focused Viewer and Feature Extraction regression tests pass.

## Slice 3a: Viewer bidirectional visibility and colormap controls

Status: implemented on 2026-07-30.

### Goal

Add per-channel eye and colormap controls that remain synchronized with native
napari controls.

### Target files

- `src/napari_harpy/widgets/viewer/image_widget.py`
- `src/napari_harpy/widgets/viewer/widget.py`
- `src/napari_harpy/widgets/overlay_color_button.py`
- `src/napari_harpy/widgets/shared_styles.py`
- `tests/test_viewer_widget.py`
- focused color-button tests if a separate test module is clearer

### Composer and initial-color boundary

Retain the immediate-add interaction completed in Slice 2. Activating a
channel through the search completer or Return loads it immediately; Slice 3a
must not introduce a pending channel, a second Add action, or a staged
pre-load editor.

- Available search results have no eye or colormap controls.
- Before requesting the layer, the card chooses the cached last-used solid
  color or the first unused entry from `DEFAULT_OVERLAY_COLORS`.
- The requested initial color is passed into layer creation, so the channel
  appears in napari already using that color. There is no intermediate
  uncolored layer.
- Eye and colormap controls exist only on selected rows backed by live overlay
  bindings.
- Create the selected row only after adapter lifecycle reconciliation confirms
  the live binding.
- Initialize the row's colormap preview from the accepted live
  `layer.colormap`, not from an assumed request value.
- A failed load leaves no selected row or presentation controls.
- Removing and re-adding a channel prefers the card's cached last-used solid
  color, as established in Slice 2.

The selected-row colormap control is a live layer-property editor, not a
pre-load parameter editor. Pre-load color customization in the Viewer is out
of scope for Slice 3a. If it later becomes a product requirement, design one
explicit pending-candidate row with a swatch and Add action; do not place
interactive color controls in every completer result.

Histogram intentionally retains a different, staged interaction: its card
holds a pending color beside the explicit `Load overlay` action. Do not change
that workflow in Slice 3a. Slice 3b replaces only the uniquely loaded state with
the shared contextual live row while preserving explicit creation. Slice 3c
then replaces the Histogram channel dropdown with persistent search.

### Direct layer-event subscriptions

Each selected channel row owns its presentation subscriptions for its current
live napari `Image`:

- connect directly to `layer.events.visible`;
- connect directly to `layer.events.colormap`;
- connect the row's bound presentation methods without intermediate capturing
  lambdas;
- read presentation state from `row.binding.layer` when either event fires.

The Viewer must not receive these events through the Histogram or through an
adapter presentation signal. The Histogram keeps its own direct subscriptions.
A single napari colormap event therefore fans out naturally to both peer
widgets when both are open.

Connection ownership must be explicit:

- let construction connect the two bound methods exactly once;
- disconnect both callbacks before a row is removed or replaced;
- disconnect before rebuilding card content or changing SpatialData, image, or
  coordinate-system context;
- disconnect during widget teardown where applicable;
- tolerate an already-disconnected or already-destroyed event source using the
  same narrow exception handling pattern as the Histogram.

Implement the connection, refresh, and disconnection methods on
`_OverlayChannelRow`. One row owns its construction-time binding for its entire
lifetime:

- construction connects the row to the initial layer and renders both
  properties from that layer;
- `dispose()` disconnects both layer-event callbacks and is idempotent.

Use `row.binding.layer` as the single layer reference for both connection and
disconnection. Napari's event system supports disconnecting the same bound
method through a fresh attribute lookup, so the row does not need
`_subscribed_layer`, capturing lambdas, or stored callback references.

`_ImageCardWidget` compares each existing row's binding with the latest binding
returned by membership reconciliation. If it is the same binding object, retain
the row and refresh its presentation. If the binding changed while the channel
index remained present, dispose and delete only that channel row, then construct
a replacement row from the new binding. Other channel rows remain untouched.
This keeps row ownership fixed and avoids a QWidget changing which napari layer
it represents.

Add `_ImageCardWidget.dispose()` to dispose all selected rows. `ViewerWidget`
must call it for every existing image card before clearing or rebuilding the
image-card layout; relying on `deleteLater()` alone leaves a window in which a
napari event can still target a stale Python callback.

Do not extract a shared Histogram/Viewer subscription abstraction in this
slice. The ownership differs: Viewer subscriptions belong to selected rows,
whereas Histogram subscriptions belong to Histogram cards. Consider a shared
abstraction only later if the completed implementations contain meaningful,
stable duplication.

### Selected-row UI contract

Render each confirmed live overlay row in this order:

1. a compact checkable eye `QToolButton`;
2. the elided channel name, taking the remaining horizontal space;
3. the compact colormap preview button;
4. the existing remove action.

The eye uses a small generated open-eye/closed-eye vector icon consistent with
the existing widget icon treatment; do not depend on napari-private icon
paths. Checked means visible. Its tooltip and accessible name describe the
next action using the channel name, for example `Hide channel DAPI` while the
layer is visible and `Show channel DAPI` while hidden.

Keep the controls compact enough that adding them does not make the selected
row materially taller. Available completer results remain plain text and do
not reuse this selected-row layout.

### User intent and mutation ownership

Modestly extend `OverlayColorButton` with a user-only signal:

```python
color_selected = Signal(str)
```

- Emit `color_selected` once when the user accepts a valid color in the picker.
- Keep `set_color(...)` silent. It is the rendering API used for
  napari-originated updates and must not emit user intent.
- Slice 3a leaves existing consumers such as Histogram unchanged. Slice 3b
  extracts the shared row/presentation contract and adopts it in both Viewer
  and Histogram. Slice 3c changes only Histogram target selection.

Add row-local intent signals:

```python
visibility_change_requested = Signal(int, bool)
color_change_requested = Signal(int, str)
```

The channel index is captured when the row is created and remains stable for
that row. The row must never mutate its bound layer directly.

The eye's user-toggle handler emits intent only. Do not call the row's
visibility rendering method from that handler: Qt has already changed the
checked state, and the successful napari `visible` event owns the
authoritative icon, tooltip, and accessibility refresh. The shared color
button continues to show an accepted picker choice locally before emitting
`color_selected`, because Histogram also uses that control for staged state;
the subsequent napari event confirms or replaces that preview in the Viewer.

`_ImageCardWidget` translates the row-local signals into public card signals
that add the image identity:

```python
overlay_channel_visibility_requested = Signal(str, int, bool)
overlay_channel_color_requested = Signal(str, int, str)
```

Connect these card signals once when `ViewerWidget` builds each image card,
beside the existing focused add and remove connections. Document them in the
card docstring as user intent consumed by `ViewerWidget`.

`ViewerWidget` owns the presentation mutation:

1. Resolve or revalidate the current live `ImageLayerBinding`.
2. Return safely if the layer disappeared during the interaction.
3. Compare the requested value with the current live property.
4. Assign `layer.colormap` or `layer.visible` only when it differs.
5. Let napari emit `layer.events.colormap` or `layer.events.visible`.
6. Let the row's direct callback read the accepted property and render it.

Add one focused live-binding resolver:

```python
def _resolve_live_overlay_binding(
    self,
    image_name: str,
    channel_index: int,
) -> ImageLayerBinding | None:
    ...
```

It may build on `_get_live_overlay_bindings(...)`, but it must validate the
current SpatialData and coordinate-system context. Return `None` when the
target is no longer live and preserve the existing invariant error for
duplicate or malformed matching bindings. Do not trust the binding object
originally held by the row for mutation, because the layer may have been
removed or replaced between the click and the handler.

Add a card method that performs a targeted read-only refresh for one channel:

```python
def refresh_overlay_channel_presentation(
    self,
    channel_index: int,
    binding: ImageLayerBinding,
) -> None:
    ...
```

Refresh only when the selected row still exists and owns that exact binding
object. A missing row or a row awaiting reconstruction for a replacement
binding is a safe no-op; normal lifecycle reconciliation owns those cases.

When assignment raises a supported property-validation error, render the
current live property back into the affected row and show concise feedback
through the Viewer's existing action-feedback area. Also perform a targeted
read-back when the requested value is already the live value, because no
setter—and therefore no new napari event—is needed. After a successful
assignment, do not perform an additional read-back: napari's native property
event is the single normal presentation path.

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

- Add one checkable eye control per selected channel row.
- Checked/open eye means `layer.visible is True`; unchecked/closed eye means
  `False`.
- Initialize it from `layer.visible` and update its icon, tooltip, and
  accessible name together.
- A user toggle emits row intent and `ViewerWidget` updates `layer.visible`
  immediately.
- Napari `visible` events update the eye without recreating unrelated rows.
- Apply napari-originated eye state under `QSignalBlocker` so reflection cannot
  become new user intent.
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

- Treat a single color, or napari's normal two-stop black-to-color tint, as a
  solid overlay color. Use the final color stop as the normalized solid hex
  value.
- Treat a colormap with additional meaningful stops as non-solid.
- Add a silent `set_colormap_preview(...)` rendering API to
  `OverlayColorButton`, accepting the colormap name and normalized preview
  stops without importing Viewer or adapter state into the button.
- Show a compact linear gradient for a non-solid colormap. Evenly sample a
  small bounded number of stops for the preview rather than producing
  unbounded stylesheet data from maps such as Viridis.
- Expose its colormap name in the tooltip and accessible name.
- Keep the button's last valid solid color as the color-dialog seed while a
  gradient is displayed.
- Clicking the preview opens the existing solid-color picker.
- Accepting a solid color switches the button to its solid preview, emits
  `color_selected` once, and requests replacement of the non-solid napari
  colormap.
- The subsequent napari event remains authoritative and may replace the local
  preview. A rejected assignment explicitly restores the current live
  presentation.

Do not build a second full colormap picker in Harpy.

### Feedback-loop protection

- Keep programmatic `OverlayColorButton.set_color(...)` silent.
- Block the eye control's Qt signals while applying napari-originated state.
- Avoid writing a layer property when the requested value is already current.
- Let the native napari event read and render the accepted property after a
  successful Harpy-originated mutation; explicitly read back only for a no-op
  or rejected assignment.
- Keep lifecycle reconciliation separate from property reflection: visibility
  and colormap events update their row, not the complete card membership.
- Never manually emit a napari layer property event.

### Implementation sequence

Keep the implementation in four focused parts:

1. Extend `OverlayColorButton` with its user-only signal and silent
   solid/gradient rendering APIs, and add the two-state eye icon helper.
2. Add the controls, row-local intent signals, direct property subscriptions,
   targeted rendering, fixed-binding ownership, and disposal to
   `_OverlayChannelRow`.
3. Forward row intent through `_ImageCardWidget`, add explicit card disposal,
   and implement the two validated `ViewerWidget` mutation handlers.
4. Add focused component, synchronization, stale-callback, teardown, and
   feedback-loop tests.

Do not combine this slice with Histogram live editing, adapter presentation
events, a general subscription framework, or a full colormap picker.

### Acceptance criteria

- Search activation still loads a channel immediately using its computed
  initial color, without a staged Add action.
- Available search results expose no pre-load eye or colormap controls.
- Presentation controls appear only after a live overlay binding exists.
- A newly reconciled row renders the accepted live layer colormap.
- Harpy and napari eyes always represent the same visibility state.
- Hidden channels remain members of the overlay.
- Solid-color changes round-trip in both directions.
- Non-solid napari colormaps are represented without pretending they are a
  solid color.
- Repeated synchronization does not recurse or duplicate mutations.
- Changes to one layer do not disturb sibling rows.
- Rebinding a stable channel row to a replacement layer disconnects the old
  layer and reflects the replacement immediately.
- Viewer and Histogram can observe the same layer property independently
  without forwarding events through one another or the adapter.
- Removing or rebuilding a row leaves no callback targeting the stale row.
- The adapter remains free of visibility and colormap subscriptions.
- Harpy never manually emits napari's colormap or visibility events.
- User-intent signals and napari property-change events remain semantically
  distinct.

### Focused tests

- Immediate search activation still creates the overlay with its computed
  default or cached initial color.
- No selected row or presentation control appears before a live binding exists.
- The first rendered swatch is read from the accepted live layer colormap.
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
- Solid tint detection for napari's two-stop black-to-color representation.
- Signal blocking/re-entrancy protection.
- The row lifecycle does not duplicate direct bound-method connections for a
  row/layer pair.
- Property-event callback cleanup after removal, row replacement, and card
  refresh.
- Card disposal before Viewer image-card rebuild and teardown.
- Replacing a binding reconstructs only that channel row, and events from the
  old layer do not update the replacement.
- A stale user intent cannot mutate a removed or replaced layer.
- Assignment failure restores the live property presentation and reports
  concise Viewer feedback.
- One layer colormap change can update both open peer widgets without an
  adapter presentation signal.

## Slice 3b: Shared overlay row and Histogram live-overlay alignment

Status: implemented on 2026-07-30.

Implementation result:

- `_OverlayChannelRow` and colormap interpretation now live in a focused shared
  widget module used by Viewer and Histogram.
- Viewer retains its Slice 3a behavior and fixed-binding row reconstruction.
- Each Histogram card reconciles invalid, missing, unique, and ambiguous overlay
  matches and owns at most one shared live row.
- Histogram live-row visibility, color, and removal intent is re-resolved
  against the exact current binding before mutation.
- Native napari visibility and colormap events update every bound peer row
  independently; adapter lifecycle events remain membership-only.
- Histogram pending load color and contrast-limit synchronization remain
  separate from live-row presentation.

### Dependency

Slice 3a is complete. The Viewer has a fixed-binding selected-channel row,
direct native napari property subscriptions, a user-only
`OverlayColorButton.color_selected` signal, and silent solid/gradient rendering
APIs.

### Goal

Extract the Viewer's overlay row and colormap interpretation into one focused
shared component, adopt that component in both Viewer and Histogram, and make a
uniquely loaded Histogram overlay live for visibility, color, and removal.

Keep the current Histogram channel dropdown and explicit `Load overlay` action
in this slice. Slice 3c changes channel selection only.

### Target files

- a focused shared overlay-row/presentation module under
  `src/napari_harpy/widgets/`;
- `src/napari_harpy/widgets/overlay_color_button.py`;
- `src/napari_harpy/widgets/viewer/image_widget.py`;
- `src/napari_harpy/widgets/viewer/widget.py`;
- `src/napari_harpy/widgets/histogram/widget.py`;
- focused shared-component, Viewer, and Histogram tests.

Do not add a central overlay controller, adapter presentation signal, generic
event bus, or generic layer-subscription framework.

### Shared fixed-binding row

Move the Viewer's selected-channel row into a shared widget component:

```text
[eye]  channel_name  [colormap]  [×]
```

It retains these intent signals:

```python
remove_requested = Signal(int)
visibility_change_requested = Signal(int, bool)
color_change_requested = Signal(int, str)
```

One row represents exactly one valid overlay `ImageLayerBinding` for its entire
lifetime. It:

- derives channel index and name from its construction-time binding;
- connects bound methods directly to `layer.events.visible` and
  `layer.events.colormap`;
- reads accepted presentation from the live layer in those callbacks;
- uses `QSignalBlocker` while reflecting napari visibility into the eye;
- renders solid and gradient colormaps silently;
- translates its internal color button's `color_selected` signal into
  `color_change_requested`;
- emits user intent but never assigns a layer property or invokes the adapter;
- disconnects both native callbacks in an idempotent `dispose()`;
- is disposed and reconstructed rather than rebound to another binding.

The owning Viewer or Histogram widget resolves identity and performs mutations.

### Shared colormap presentation

Move the existing Viewer-only colormap interpretation into the same focused
shared boundary. Provide one pure conversion from a live layer colormap to:

- display name;
- normalized preview colors;
- normalized solid color when it represents one solid tint.

Preserve the Slice 3a rules:

- one color stop is solid;
- black followed by one color is napari's solid-tint representation;
- two or more meaningful stops are a gradient;
- malformed or unsupported data produces no invented color.

Viewer, Histogram, and the shared row use this conversion. Remove parallel
Viewer and Histogram swatch-conversion implementations.

`OverlayColorButton.current_color` remains the last valid solid picker seed.
Rendering a gradient must not overwrite it.

### Signal and event ownership

The shared row makes presentation ownership identical in Viewer and Histogram,
while each owning widget retains its own target-resolution handlers:

| Signal or event | Listener | Responsibility |
| --- | --- | --- |
| `ViewerAdapter.image_overlay_layers_changed` | `ViewerWidget` and `HistogramWidget` | Re-query membership and construct, retain, or dispose rows |
| `layer.events.visible` | Every shared row bound to that layer | Read `layer.visible` and render that row's eye |
| `layer.events.colormap` | Every shared row bound to that layer | Read `layer.colormap` and render that row's swatch |
| live-row `OverlayColorButton.color_selected` | The shared row containing that button | Translate accepted picker intent into `row.color_change_requested` |
| pending `OverlayColorButton.color_selected` | Histogram card-local pending state | Retain the preferred load color only; do not mutate napari |
| `row.visibility_change_requested` | The Viewer or Histogram owner of that row | Re-resolve identity and assign `layer.visible` |
| `row.color_change_requested` | The Viewer or Histogram owner of that row | Re-resolve identity and assign `layer.colormap` |
| `row.remove_requested` | The Viewer or Histogram owner of that row | Re-resolve identity and call the adapter removal operation |
| `layer.events.contrast_limits` | Histogram contrast-sync code only | Synchronize the Histogram plot; this is not a shared-row concern |

The shared row does not listen to `image_overlay_layers_changed`. That signal
means the binding set may have changed; only owning widgets have enough context
to resolve membership.

Viewer and Histogram never listen to presentation signals from each other.
`ViewerAdapter` never mirrors `visible` or `colormap` changes.

### Viewer adoption

Replace the private Viewer row with the shared row without changing Viewer
behavior:

- selected rows remain keyed by channel index;
- unchanged exact bindings retain their row;
- replacement bindings reconstruct only the affected row;
- `_ImageCardWidget` continues to add image identity to row intent;
- `ViewerWidget` continues to re-resolve and assign live properties;
- native napari events remain the only successful presentation-return path.

Viewer intent still follows:

```text
shared row emits (channel_index, requested value)
  -> _ImageCardWidget adds image_name
  -> ViewerWidget re-resolves the exact live binding
  -> ViewerWidget assigns the layer property
  -> napari emits the native property event
  -> every shared row bound to that layer renders independently
```

No Viewer search, layout, membership, loading, removal, or feedback behavior
changes in this slice.

### Histogram overlay matching

The current Histogram channel dropdown remains the target selector. Replace the
existing resolver's ambiguous `None` result with explicit outcomes:

1. no matching live overlay;
2. exactly one matching live overlay binding;
3. multiple matching live overlay bindings.

Match by current:

- SpatialData object;
- coordinate system;
- image name;
- `image_display_mode == "overlay"`;
- channel name.

Invalid targets and ambiguous matches retain a concise reason for card-local
feedback. Never guess among duplicate bindings.

### Histogram contextual Viewer area

For the current dropdown target, render one state:

| Target and overlay state | Histogram Viewer area in Slice 3b |
| --- | --- |
| Invalid target | Disabled controls |
| No matching overlay | Existing `Load overlay` plus pending solid-color button |
| Exactly one live overlay | Shared `[eye] channel [colormap] [×]` row |
| Multiple live overlays | Warning state with no mutation controls |

Do not show pending-load controls and a live row simultaneously.

Each Histogram card owns at most one shared row:

- construct it only for one unique matching binding;
- retain it while the exact binding remains current;
- dispose and reconstruct it when the dropdown target or binding changes;
- dispose it when the match disappears, becomes ambiguous, or the card is
  removed;
- disconnect before `deleteLater()`;
- never reuse it for another binding.

Remove the separate Histogram colormap-subscription state after the shared row
owns colormap reflection. Histogram contrast-limit subscription state remains
separate and unchanged.

### Histogram row-intent connections

When Histogram constructs a shared row, connect its three intent signals exactly
once. Add the stable Histogram card id at that ownership boundary.

For example:

```text
row.visibility_change_requested(channel_index, visible)
  -> Histogram handler receives card_id
  -> Histogram re-resolves the card's dropdown target
  -> Histogram requires the exact current row binding
  -> Histogram assigns layer.visible
  -> napari emits layer.events.visible
  -> Histogram and Viewer shared rows update independently
```

Color follows the same flow through `layer.colormap`. A solid Histogram choice
may replace a live gradient. A napari gradient is rendered through the shared
preview without overwriting the row's solid picker seed.

For eye or color intent:

1. Re-resolve the current card target and unique live binding.
2. Require exact binding identity with the card's current row.
3. Compare the requested property with the live property.
4. Assign only when different.
5. Let napari emit its native property event.
6. Let all peer rows render independently.

For a no-op, silently refresh the row from its live binding. For a rejected
assignment, restore authoritative presentation and report concise card feedback.
Do not optimistically render after successful assignment and never manually emit
napari property events.

### Pending load and explicit removal

When no overlay exists, the existing pending color button remains local staged
state. Choosing a color does not create a layer. `Load overlay` continues to:

- pass the pending button's `current_color` to
  `ensure_image_overlay_channel_loaded(...)`;
- preserve sibling overlays;
- avoid Histogram calculation;
- activate the returned layer;
- let `image_overlay_layers_changed` create the shared live row.

The live row's `×` explicitly removes only its current overlay:

1. Re-resolve the current target and exact live binding.
2. Cache the row's last valid solid picker seed.
3. Call `ViewerAdapter.remove_image_overlay_channel(...)`.
4. Let lifecycle invalidation dispose the row.
5. Show pending-load controls for the still-selected dropdown target.

Removing the row never removes the Histogram card or clears its channel target.
If napari removes the layer, the same lifecycle path returns the card to pending
load. Never remove a sibling overlay.

### Shared native-event fan-out

If Viewer and one or more Histogram cards show the same binding:

```text
one owner assigns layer.colormap
  -> napari emits one layer.events.colormap event
      -> Viewer shared row renders
      -> Histogram card A shared row renders
      -> Histogram card B shared row renders
```

Multiple callbacks are expected peer-observer behavior. No callback may emit a
new user-intent signal during programmatic rendering.

### Failure and stale-state behavior

- Missing or stale rows reconcile and return without mutation.
- Ambiguous matches show a warning and expose no live mutation controls.
- Failed visibility or color assignment restores live presentation.
- Failed load or removal uses the existing Histogram status area.
- Rename `overlay_load_message` to a neutral Viewer-action name if it now
  carries load, visibility, color, ambiguity, and removal feedback.
- Do not show success messages for every eye or color change; the row and canvas
  already confirm success.

### Implementation sequence

1. Extract the shared row and colormap presentation conversion.
2. Migrate Viewer with no behavioral change.
3. Add explicit Histogram match outcomes and contextual Viewer-area states.
4. Replace Histogram's standalone colormap subscription with the shared row.
5. Connect validated Histogram visibility, color, and removal handlers.
6. Add focused shared, Viewer, Histogram, and regression tests.

### Acceptance criteria

- Viewer and Histogram use the same fixed-binding row for a unique live overlay.
- The signal ownership table above is reflected directly in connections.
- Viewer behavior remains unchanged after migration.
- Histogram retains its current dropdown and explicit `Load overlay`.
- Missing overlays keep pending color without implicit creation.
- Unique Histogram overlays expose live eye, colormap, and remove controls.
- Eye and colormap changes synchronize bidirectionally through native napari
  events.
- `×` removes only the uniquely matched overlay.
- Napari-side removal returns Histogram to pending-load state.
- Multiple Histogram cards and Viewer may observe the same layer independently.
- Histogram contrast-limit synchronization remains separate.
- Missing, stale, and ambiguous targets produce no unintended mutation.
- No adapter presentation signal, widget-to-widget event, or duplicated
  colormap conversion remains.

### Focused tests

- Shared row solid, gradient, visibility, intent, and disposal behavior.
- Viewer migration preserves current bidirectional behavior.
- Viewer replacement binding reconstructs only the affected row.
- Histogram missing, unique, and ambiguous contextual states.
- Existing unique overlay hydrates into one shared Histogram row.
- Histogram visibility and solid/gradient colormap synchronization in both
  directions.
- Same-value intent produces no property assignment.
- Rejected assignment restores authoritative presentation.
- `×` and napari-side removal both return to pending controls.
- Target, binding, layer, and card changes disconnect row callbacks.
- Repeated lifecycle invalidation does not duplicate callbacks.
- Two Histogram cards and Viewer observe one native property event independently.
- Existing Histogram calculation, contrast-sync, and explicit-load tests remain
  green.

## Slice 3c: Searchable Histogram target selection

Status: specified on 2026-07-30.

### Dependency

Slice 3b is complete. The current Histogram dropdown already drives pending,
unique-live-row, and ambiguous Viewer-area states through the shared row.

### Goal

Replace only the Histogram channel dropdown with a scalable persistent
searchable selector. Preserve all Slice 3b overlay behavior and keep analysis
target selection separate from overlay membership.

### Target files

- `src/napari_harpy/widgets/histogram/widget.py`;
- `src/napari_harpy/widgets/shared_styles.py` only if the existing completer
  contract needs a focused correction;
- `tests/test_histogram_widget.py`;
- `tests/test_shared_styles.py` only for a shared completer correction.

No adapter, shared-row, Histogram calculation, or napari property-event contract
change is expected.

### Channel selection is not overlay membership

```text
Accept Histogram channel
  -> change this card's analysis target
  -> dispose only this card's previous contextual row
  -> reconcile Slice 3b state for the new target
  -> do not add or remove any napari layer
```

Changing from DAPI to CD3 removes the DAPI row from that Histogram card only.
The DAPI napari layer remains loaded for Viewer or other Histogram cards.

### Persistent searchable selector

Replace the `CompactComboBox` channel control with
`CompleterPopupLineEdit(QLineEdit)` configured with:

- `QStringListModel`;
- popup completion;
- case-insensitive substring filtering;
- at most ten visible results;
- existing shared popup styling;
- popup-on-entry behavior.

Maintain accepted channel selection separately from transient edit text. Typing
filters candidates but does not change the target until the user:

- activates a completion; or
- presses Enter with an exact or uniquely case-insensitive match.

On acceptance:

- store the selected channel name as explicit card state;
- keep that name visible in the line edit;
- do **not** call `clear_after_accepted_completion(...)`;
- dispose the old contextual row before switching target;
- refresh controller binding, settings, status, cyclic pending-color default,
  and Slice 3b Viewer-area reconciliation exactly once.

Invalid free text does not replace the accepted target. Show concise validation
feedback or restore accepted text when editing finishes. If image or SpatialData
changes invalidate the accepted channel, clear accepted state and rebuild the
completion model.

### Reusing Slice 3b states

After each accepted target change, render the existing Slice 3b result:

| New target state | Viewer area |
| --- | --- |
| No matching overlay | Explicit load plus pending solid color |
| Exactly one live overlay | Shared `[eye] channel [colormap] [×]` row |
| Multiple live overlays | Warning with no mutation controls |

The final explicit-load label becomes `Load in viewer` to distinguish it from
Histogram calculation. Accepting a completion never invokes that button's
operation.

Changing coordinate system, image, SpatialData, or removing the Histogram card
uses the same ordering: dispose the old row, update accepted target state, then
reconcile.

### Event-flow preservation

Slice 3c changes only how the accepted channel is chosen. Once accepted:

- membership still comes from `image_overlay_layers_changed`;
- native visibility and colormap events still go directly to the shared row;
- row intent still goes to Histogram's validated handlers;
- contrast-limit events still go to Histogram contrast-sync code;
- explicit load and remove still use `ViewerAdapter`.

Do not add a second set of selector-specific property signals.

### Implementation sequence

1. Add explicit accepted-channel state to `_HistogramCard`.
2. Replace combo population with completer-model refresh.
3. Implement exact/unique completion acceptance and persistent text.
4. Route accepted target changes through existing cleanup and reconciliation.
5. Rename the explicit action to `Load in viewer`.
6. Add focused selector and target-switch regression tests.

### Acceptance criteria

- Channel discovery scales through a searchable popup.
- Accepted channel text remains visible.
- Typing alone does not change the Histogram target.
- Invalid free text preserves the previous accepted target.
- `clear_after_accepted_completion(...)` is never used.
- Accepting a channel never creates or removes a napari layer.
- The previous target's contextual row is disposed without removing its layer.
- The new target immediately shows its correct Slice 3b state.
- Explicit loading remains independent from Histogram calculation.
- Slice 3b visibility, colormap, removal, and callback ownership are unchanged.

### Focused tests

- Popup-on-entry and case-insensitive substring filtering.
- Accepted completion remains visible without deferred clearing.
- Enter accepts exact and uniquely case-insensitive names.
- Invalid and ambiguous free text preserves the previous target.
- Selecting another channel changes controller target but creates/removes no
  layer.
- Switching from one live target to another disposes only the old card row.
- Switching to an unloaded target shows pending controls.
- Switching to an already loaded target constructs the matching shared row.
- Image, SpatialData, and coordinate-system changes invalidate selection safely.
- Pending color is passed to `Load in viewer` without calculating a Histogram.
- All Slice 3b synchronization and removal tests remain green.

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

### Finish removal of staged apply

- Delete the transitional generic `Add / Update in viewer` action retained for
  stack mode by Slice 2 and replace it with the contextual `Load stack` action.
- Confirm that no aggregate overlay request-building or checkbox-list code
  remains after Slice 2.
- Remove obsolete `ImageLoadRequest` fields or helpers once the stack path no
  longer needs the aggregate request shape.
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
the rest of the Viewer and Histogram widgets.

### Target files

- `src/napari_harpy/widgets/viewer/image_widget.py`
- `src/napari_harpy/widgets/histogram/widget.py`
- the shared overlay-row and widget styles only when a small reusable correction
  is genuinely useful
- focused Viewer and Histogram tests

### Implementation

- Keep spacing, control heights, borders, hover states, and typography
  consistent with existing viewer cards.
- Elide long channel names and expose the full name in a tooltip.
- Give search, eye, colormap, remove, remove-all, mode, and count controls
  appropriate accessible names.
- Make pending-load controls visually distinct from a loaded live row.
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
- Surface load/update failures through the owning widget's existing feedback
  area.
- Verify Viewer and Histogram layouts at narrow napari dock widths and with long
  channel names.

### Acceptance criteria

- The workflow can be completed using the keyboard.
- Long names do not widen the dock or require horizontal scrolling.
- Empty, populated, hidden, loading, error, non-solid-colormap, many-channel,
  no-channel-axis, and duplicate-channel states are understandable.
- Histogram pending-load, unique-live-row, and ambiguous-match states are
  visually distinct.
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
- Histogram pending versus live Viewer-area presentation.

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
- Histogram channel search with long and similar names;
- changing a Histogram target while its previous overlay remains loaded;
- a Histogram live-row `×` removal followed by explicit reload;
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

- Viewer and Histogram channels are discovered through searchable popups;
- Viewer selected rows correspond exactly to live Harpy overlay layers;
- each Histogram card shows at most one contextual row for its accepted target
  and only while a unique matching overlay is live;
- add, remove, and remove-all operations update napari immediately;
- eyes and colormaps synchronize in both directions;
- Histogram channel selection never creates or removes an overlay implicitly;
- Histogram overlay creation remains explicit through `Load in viewer`;
- a Histogram live-row picker edits its uniquely matching overlay immediately;
- changing a Histogram target preserves the previous target's napari layer;
- presentation synchronization comes directly from live napari layer events;
- external napari layer deletion updates the composer;
- membership, visibility, and removal remain distinct;
- the adapter remains responsible for binding/lifecycle operations rather than
  mirroring napari presentation properties;
- the overlay `Add / Update in viewer` action is gone;
- initial stack loading remains explicit and clearly labelled;
- focused automated tests and manual verification pass.
