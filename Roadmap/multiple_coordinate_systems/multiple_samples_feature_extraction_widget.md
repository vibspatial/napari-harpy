# Multiple-Sample Feature Extraction Widget

## Purpose

This document breaks the batch feature-extraction widget refactor into smaller,
lower-risk implementation slices.

It is a companion to:

- `Roadmap/multiple_coordinate_systems/cross_sample_tables.md`

The goal is to avoid one large UI rewrite that changes rendering, selection
model, controller binding, channel handling, and validation all at once.

## Guiding Rule

Implement the widget in slices that each preserve a stable, testable
intermediate state.

In practice, that means:

- each slice should leave the widget usable;
- each slice should have a focused test update;
- controller and widget changes should be introduced incrementally rather than
  all at once.

## Slices

### 1. Extract a Single Triplet Card

Status: [x] Completed

Goal:

- keep the existing UX and behavior;
- refactor the current single-triplet widget internals into an explicit
  reusable "triplet card" structure.

Scope:

- keep one selected coordinate system;
- keep one segmentation selector;
- keep one image selector;
- remember the user’s last explicit `coordinate system -> segmentation ->
  image` selection per coordinate system while the current `sdata` remains
  loaded;
- when the user leaves a coordinate system and later returns to it, restore
  that remembered triplet if it is still valid;
- preserve the previously selected image only when it remains valid after a
  segmentation or coordinate-system refresh;
- otherwise fall back to `No image`;
- do not auto-select the only remaining valid image just because there is
  exactly one match;
- keep the current shared controls (`Channels`, `Table`, `Feature matrix key`,
  `Intensity Features`, `Morphology Features`) in their current positions;
- do not introduce batch selection yet.

Expected outcome:

- the widget still behaves like today's single-triplet widget;
- explicit per-coordinate-system triplet choices are sticky across
  coordinate-system switching;
- image selection is treated as explicit user choice rather than inferred from
  a single remaining match;
- the rendering code is prepared for multiple cards later.

### 2. Add Coordinate-System Checkbox Selection and Multi-Card Rendering

Status: [x] Completed

Goal:

- move from one visible coordinate-system selection flow to one or more
  selected coordinate systems rendered as cards.

Scope:

- replace the coordinate-system combo box with a checkbox list;
- keep the checkbox list sorted in the same order as the rendered cards;
- on startup or `sdata_changed`, leave all coordinate systems unchecked by
  default;
- render one triplet card per selected coordinate system;
- render cards in sorted coordinate-system order;
- keep one selected coordinate system equal to one triplet card;
- when a coordinate system is unchecked, hide its card but keep its remembered
  triplet state while the current `sdata` remains loaded;
- when a previously unchecked coordinate system is checked again, restore its
  remembered triplet if it is still valid;
- keep shared controls below the cards.
- treat this slice as a rendering/state step only;
- keep calculation disabled while multi-card execution semantics are still
  incomplete.
- when no coordinate systems are checked, render no triplet cards and show a
  status prompt such as `Choose one or more coordinate systems to start
  building extraction targets`.

Expected outcome:

- users can select multiple coordinate systems;
- the widget starts from an explicitly empty batch-selection state;
- the widget renders one card per selected coordinate system;
- cards still use the existing segmentation/image filtering logic per
  coordinate system;
- the widget can show and preserve multiple remembered triplet states without
  yet exposing a fully working batch calculation flow;
- cross-card batch-validity checks are still intentionally out of scope here:
  duplicate segmentation prevention, unavailable-item reasoning, and other
  multi-card consistency rules are deferred to slice 3.

### 3. Add Batch Validity Rules

Status: [x] Completed

Goal:

- make multi-card selection safe before introducing richer shared batch logic.

Scope:

- forbid selecting the same segmentation in more than one card;
- allow valid image reuse across cards;
- in this slice, treat segmentation choice as explicit rather than eager:
  when a card is first shown, its segmentation combo starts at `Choose a
  segmentation mask` instead of auto-selecting the first available
  segmentation for that coordinate system;
- keep selected cards visible even when they are temporarily incomplete or
  invalid;
- when the user explicitly selects a segmentation in one card, immediately
  exclude that segmentation from the selectable choices offered in the other
  selected cards;
- in those other cards, surface a short inline note that the segmentation is
  unavailable because it is already selected in another coordinate system card;
- keep image choice explicit in batch mode as well:
  even when exactly one matching image remains valid for a selected
  segmentation, keep the card at `No image` until the user explicitly chooses
  that image;
- if a card is restored with a previously explicit segmentation selection that
  has since become blocked by another selected card, clear the card back to
  `Choose a segmentation mask`, show a short explanatory note, and drop that
  remembered selection instead of silently restoring it later; clear the
  remembered image identity for that card at the same time so stale image
  memory cannot survive the failed restore;
- show only selectable segmentations and images in the card combo boxes;
- surface unavailable segmentation/image counts with short inline reasons;
- block submission while any selected card is invalid.
- keep calculation disabled in this slice; this slice introduces widget-local
  batch validity state and inline invalidity UI, while fully meaningful batch
  submission gating remains part of slices 4 and 5.

Clarification:

- all cross-card validity checks belong to this slice, not slice 2;
- cards stop auto-defaulting to the first available segmentation in this
  slice; each checked card now starts from an explicit `Choose a segmentation
  mask` state;
- each card distinguishes between its current staged selection and its
  currently selectable segmentation options; the current staged selection is
  either empty or one valid segmentation, while the selectable options exclude
  segmentations already chosen in other selected cards;
- slice 3 is limited to widget-local batch validity and per-card inline
  feedback;
- in this slice, the shared controls below the cards may still use the current
  active-card compatibility bridge rather than a fully batch-aware binding
  model;
- that means table selection, image-channel UI, top-level status text, and
  controller binding may still reflect the active card only during slice 3;
- slice 3 should therefore not be treated as the point where those shared
  controls become batch-authoritative;
- true batch-wide table validation, shared channel derivation, batch status
  messaging, and controller `bind_batch(...)` integration are deferred to
  slices 4 and 5;
- in other words, slice 3 is where the widget stops treating each card as an
  isolated local selector and starts enforcing whole-batch selection safety.

Implementation detail:

- before implementing slice 3 behavior, update the triplet-card widget/state
  data model so it can represent:
  - one current staged selection per card;
  - the currently selectable options for that card;
  - short inline note text for segmentation and image availability feedback.
- the current widget still projects one selected card into older
  single-triplet fields used by the lower half of the widget; slice 3 may keep
  that compatibility bridge in place while refactoring card-level selection
  and validity state.
- do not expand slice 3 to make the table selector, shared channel selector,
  primary status card, or controller binding fully batch-aware; use slice 3 to
  make card rendering, selectable-option derivation, duplicate prevention, and
  inline invalidity messaging correct first.
- the recommended shape is:

```python
@dataclass(frozen=True)
class _FeatureExtractionTripletCardWidgets:
    coordinate_system: str
    container: QGroupBox
    segmentation_combo: CompactComboBox
    segmentation_note_label: QLabel
    image_combo: CompactComboBox
    image_note_label: QLabel


@dataclass(frozen=True)
class _FeatureExtractionTripletCardState:
    coordinate_system: str | None
    selectable_label_options: list[SpatialDataLabelsOption]
    selected_label_option: SpatialDataLabelsOption | None
    segmentation_note_text: str | None
    selectable_image_options: list[SpatialDataImageOption]
    selected_image_option: SpatialDataImageOption | None
    image_note_text: str | None
```

- `selected_label_option` and `selected_image_option` represent the current
  staged selection and may therefore be only:
  - `None` / placeholder;
  - one valid currently staged option.
- `selectable_label_options` and `selectable_image_options` represent the
  options the user may actively choose from at that moment.
- `selectable_label_options` contains only real selectable label elements;
  `Choose a segmentation mask` remains a UI placeholder rather than a real
  option, represented in state by `selected_label_option=None`.
- `selectable_image_options` contains only real selectable image elements;
  `No image` remains a UI sentinel rendered by the combo box, represented in
  state by `selected_image_option=None`.
- keep remembered user choices in a separate structure rather than embedding
  them into `_FeatureExtractionTripletCardState`; remembered selection is
  restoration input, while triplet-card state is the current resolved UI
  state for this rebuild.
- the recommended remembered-selection shape is:

```python
@dataclass(frozen=True)
class _FeatureExtractionCardSelection:
    label_identity: ElementIdentity | None
    image_identity: ElementIdentity | None


self._remembered_card_selection_by_coordinate_system: dict[
    str, _FeatureExtractionCardSelection
]
```

- do not store `remembered_label_option` or `remembered_image_option` inside
  `_FeatureExtractionTripletCardState`.
- when a card is rebuilt, use the remembered-card-selection map only as a
  candidate source for restoration; the rebuilt card state should still expose
  only the current staged selections, selectable options, and inline note
  text.
- `selectable_label_options` should be treated as derived state rather than as
  a locally mutated list on one card.
- for one card, `selectable_label_options` depends on:
  - the card's coordinate system;
  - the currently loaded `sdata`;
  - the segmentation selections currently staged in the other checked cards.
- because of that cross-card dependency, changing the selected segmentation in
  one visible card must trigger recomputation of `selectable_label_options`
  for all visible cards, not only for the card that changed.
- keep that recomputation on an in-place state-refresh path rather than on a
  destructive widget-recreation path.
- to make that distinction obvious in code, prefer naming that uses
  `rebuild` for widget-structure changes and `recompute` for in-place state
  derivation.
- in other words:
  - rename `_refresh_triplet_cards()` to
    `_rebuild_visible_triplet_cards()` and use that structural path only when
    the set of visible cards changes, such as after `sdata` changes or after
    coordinate systems are checked/unchecked;
  - do not use that structural rebuild path for ordinary segmentation changes;
  - instead, recompute and reapply state onto the already-rendered card
    widgets.
- similarly, rename `_refresh_triplet_card_for_coordinate_system(...)` to
  `_recompute_triplet_card_state_for_coordinate_system(...)`, because that
  helper resolves one card state rather than structurally rebuilding cards.
- the helper shape should make that boundary explicit, for example:

```python
def _snapshot_visible_card_selections(
    self,
) -> dict[str, _FeatureExtractionCardSelection]:
    """Capture current staged card selections before cross-card recomputation."""


def _recompute_visible_triplet_card_states(
    self,
    *,
    preferred_selection_by_coordinate_system: Mapping[
        str, _FeatureExtractionCardSelection
    ]
    | None = None,
) -> None:
    """Recompute and reapply state for visible cards without recreating widgets."""
```

- `_snapshot_visible_card_selections()` is optional but recommended for
  maintainability: it gives the recomputation step one stable snapshot of the
  current staged selections before any card state is rebuilt.
- `_recompute_visible_triplet_card_states()` should:
  - iterate over the currently visible coordinate systems in display order;
  - call `_recompute_triplet_card_state_for_coordinate_system(...)` or an
    equivalent single-card recomputation path for each visible card, using the
    current `sdata`, the current checked-card set, and the preferred
    remembered/current selections for that card;
  - update `_triplet_card_states_by_coordinate_system`;
  - call `_apply_triplet_card_state(...)` on the existing widgets instead of
    clearing and recreating them;
  - finish by syncing the active-card compatibility bridge.
- `_on_triplet_card_segmentation_changed(...)` should therefore update the
  remembered selection for the changed card and then call
  `_recompute_visible_triplet_card_states(...)`, not the structural
  card-recreation path.
- remove the older partial-refresh helpers such as `_refresh_label_options()`
  and `_refresh_image_options()` during the slice 3 refactor rather than
  keeping multiple overlapping refresh entry points; `_recompute_visible_triplet_card_states(...)`
  should become the one authoritative visible-card recomputation flow.
- in practice, recompute visible card states when:
  - `sdata` changes;
  - the set of checked coordinate systems changes;
  - any visible card changes its selected segmentation.
- image changes, channel changes, feature-checkbox changes, and table changes
  do not by themselves change `selectable_label_options`.
- for slice 3 note text, the widget should not reimplement feature-extraction
  discovery and matching logic locally. Instead, `_spatialdata.py` should
  become the source of truth for feature-extraction discovery summaries.
- the recommended discovery API shape is:

```python
@dataclass(frozen=True)
class SpatialDataFeatureExtractionLabelDiscovery:
    coordinate_system: str
    eligible_label_options: list[SpatialDataLabelsOption]
    coordinate_system_label_count: int
    unavailable_label_count: int


@dataclass(frozen=True)
class SpatialDataFeatureExtractionImageDiscovery:
    coordinate_system: str
    label_name: str
    eligible_image_options: list[SpatialDataImageOption]
    coordinate_system_image_count: int
    unavailable_image_count: int


def get_spatialdata_feature_extraction_label_discovery_for_coordinate_system_from_sdata(
    *,
    sdata: SpatialData,
    coordinate_system: str,
) -> SpatialDataFeatureExtractionLabelDiscovery:
    ...


def get_spatialdata_feature_extraction_image_discovery_for_coordinate_system_and_label_from_sdata(
    *,
    sdata: SpatialData,
    coordinate_system: str,
    label_name: str,
) -> SpatialDataFeatureExtractionImageDiscovery:
    ...
```

- `eligible_label_options` means labels in the coordinate system that satisfy
  feature-extraction eligibility rules.
- `eligible_image_options` means images that are eligible for the specific
  `(coordinate_system, label_name)` pair; although the field name is
  symmetric with labels, its semantics remain label-dependent.
- the widget then derives:
  - `selectable_label_options` from `eligible_label_options`, minus
    segmentations already selected in the other checked cards;
  - `selectable_image_options` directly from `eligible_image_options`, because
    valid image reuse across cards is allowed.
- `unavailable_label_count` and `unavailable_image_count` support the coarse
  slice 3 notes such as `2 segmentations unavailable` and
  `3 images unavailable because they do not satisfy matching requirements`.
- because the slice 3 card state carries only one `segmentation_note_text`
  and one `image_note_text`, combine cross-card duplicate warnings and coarse
  unavailable-count feedback into one short sentence when both apply, rather
  than treating them as separate competing note slots.
- the older option-only feature-extraction helpers are removed; the discovery
  helpers are now the source of truth and the API the widget consumes directly.
- a remembered segmentation that becomes blocked by another selected card is
  not kept as a blocked current selection in state; instead:
  - the card resets to `Choose a segmentation mask`;
  - `selected_label_option` becomes `None`;
  - the remembered segmentation identity is dropped;
  - the remembered image identity for that card is dropped at the same time;
  - `segmentation_note_text` explains why the prior selection could not be
    restored.
- for maintainability, prefer using combo-box placeholder behavior for
  `Choose a segmentation mask` rather than storing that placeholder as a fake
  selectable option in state.

Expected outcome:

- the UI no longer allows ambiguous duplicate segmentation selection;
- normal interactive card selection no longer creates duplicate segmentation
  choices across cards, while blocked remembered selections reset clearly back
  to placeholder state with an explanatory note;
- the batch selection model is safe before calculation.

### 4. Add Shared Batch Channel Selection

Status: [x] Completed

Goal:

- move from per-image channel handling to one shared batch channel selector for
  intensity extraction.

Scope:

- derive the shared channel selector from the first explicitly selected image
  in sorted card order;
- block intensity batches when later selected images expose incompatible
  ordered channel-name schemas;
- treat selected images without explicit channel names as invalid for feature
  extraction rather than folding them into batch compatibility rules;
- keep morphology-only execution possible without images.

Clarification:

- slice 4 changes the meaning of the shared `Channels` control: it no longer
  reflects the active card's current image, but instead reflects the current
  batch's shared channel schema;
- that shared schema should be derived from the first explicitly selected
  image in sorted visible-card order;
- slice 4 assumes image elements used for feature extraction expose explicit
  channel names; if a selected image does not, that is invalid state and
  should raise rather than being silently skipped or treated as compatible;
- later selected images do not get independent channel controls; they either
  match that shared ordered channel-name schema or make the intensity batch
  incompatible;
- when selected images are channel-incompatible, keep the shared selector
  visible for the current reference schema rather than hiding it, show one
  shared incompatibility message below the selector, and preserve the current
  shared channel selection;
- when intensity-derived features are selected, any shared batch-channel
  error, including incompatible ordered channel schemas or duplicate channel
  names in a selected image, makes the batch invalid for calculation; when
  only morphology features are selected, those channel errors do not by
  themselves block the batch;
- morphology-only extraction must still remain possible when no selected image
  exists, so the absence of a shared channel selector is not itself an error;
- calculation may remain disabled in slice 4 while the widget adopts the
  shared channel-selection model; the goal of this slice is to make shared
  channel state visible and internally consistent before slice 5 finalizes
  batch binding and submission gating.

Implementation detail:

- replace the current image-local channel memory model with one shared current
  batch channel selection plus remembered selections keyed by ordered channel
  schema;
- in particular, do not keep slice 4 tied to
  `_selected_channel_names_by_image_identity`, because one shared channel
  selector no longer maps cleanly to one remembered selection per image;
- the recommended shared state shape is:

```python
self._batch_channel_names: list[str] = []
self._selected_batch_channel_names: tuple[str, ...] | None = None
self._batch_channel_error: str | None = None
self._selected_channel_names_by_schema: dict[tuple[str, ...], tuple[str, ...]] = {}
```

- slice 4 should also change the meaning of the existing
  `selected_extraction_channel_names` and
  `selected_extraction_channel_indices` accessors:
  - before slice 4, they reflect channel selection for the active image;
  - in slice 4, they should instead reflect the shared current batch channel
    selection resolved against the current reference schema;
- implement that semantic shift by moving those accessors off the
  active-image-local channel state and onto the shared batch channel state;
- in practice:
  - `selected_extraction_channel_names` should return the checked names from
    the shared batch channel checkbox list;
  - `selected_extraction_channel_indices` should return the indices of those
    checked names within `self._batch_channel_names`;
  - shared channel checkbox changes should update both
    `self._selected_batch_channel_names` and the schema-keyed remembered map;
- derive the ordered shared channel schema from the first explicitly selected
  image in sorted visible-card order;
- if no selected image exists, hide the shared channel selector entirely;
- when a shared channel schema is available:
  - render one shared checkbox list from that ordered schema;
  - first check whether `self._selected_batch_channel_names` is still a valid
    subset of the current schema and preserve it when possible;
  - otherwise try restoring the selection remembered for that exact ordered
    schema from `self._selected_channel_names_by_schema`;
  - if neither the current shared selection nor schema-keyed remembered state
    is valid, default to all channels selected;
- when the user changes the shared channel selection, store that selection
  back into both:
  - `self._selected_batch_channel_names` as the current batch state;
  - `self._selected_channel_names_by_schema[current_schema]` as the
    remembered selection for that ordered schema;
- key remembered channel state by the full ordered channel schema, not by a
  set of names, because ordered channel-name compatibility is the actual batch
  rule;
- for every other selected image in the batch, compare its ordered channel
  names against the reference schema;
- if any selected image exposes duplicate channel names, treat that as a
  shared batch-channel error rather than an active-image-local error; the
  shared batch channel helper should surface that problem through one shared
  `error_text` and prevent intensity execution semantics for the batch;
- if a later image exposes a different ordered schema, do not create a second
  channel selector; instead, surface a short batch-level incompatibility
  message below the shared selector, preserve the current shared channel
  selection, and block intensity execution semantics for that batch;
- the widget should therefore introduce a helper that resolves:
  - the reference selected image for the current batch;
  - the shared ordered channel schema, if any;
  - whether the selected images are channel-compatible for intensity
    extraction;
- a recommended helper shape is:

```python
@dataclass(frozen=True)
class _FeatureExtractionBatchChannelState:
    reference_coordinate_system: str | None
    reference_image_option: SpatialDataImageOption | None
    channel_names: tuple[str, ...]
    incompatible_coordinate_systems: tuple[str, ...]
    incompatible_image_names: tuple[str, ...]
    error_text: str | None

    @property
    def is_compatible(self) -> bool:
        return self.error_text is None


def _resolve_batch_channel_state(self) -> _FeatureExtractionBatchChannelState:
    ...
```

- `_resolve_batch_channel_state()` should:
  - walk visible cards in sorted display order and pick the first explicitly
    selected image as the reference image for the batch;
  - return an empty `channel_names` tuple when no selected image exists, so
    morphology-only batches can proceed without channel UI;
  - read ordered channel names from the reference image and treat that ordered
    tuple as the shared schema for the batch;
  - treat a selected image without channel names as invalid state and raise,
    rather than skipping it or inventing an implicit schema;
  - treat duplicate channel names in any selected image as a shared batch
    error, rather than leaving duplicate-channel validation tied only to the
    active card;
  - compare every other selected image's full ordered channel-name tuple
    against that schema;
  - collect incompatible coordinate systems and image names into the returned
    state rather than hiding those images from their per-card image selectors;
  - surface one shared `error_text` when selected images do not expose the
    same ordered channel-name schema.
- use that shared batch helper as the source of truth for both the visible
  channel selector and the intensity-features hint;
- use that shared batch helper as the source of truth for the shared channel
  selector refresh path too, rather than deriving channel UI from the active
  card's selected image;
- slice 4 should introduce one authoritative recomputation path for shared
  batch channel state, analogous to slice 3's authoritative recomputation path
  for visible triplet-card state;
- in practice, recompute shared batch channel state when:
  - `sdata_changed`;
  - a coordinate system is checked or unchecked;
  - a segmentation change changes or clears a selected image;
  - an image selection changes;
  - a shared channel checkbox changes;
- this schema-keyed memory is important for flows where one selected image
  disappears from the batch and a later image with the same ordered channel
  schema becomes the new reference image; in that case, Harpy should restore
  the prior shared channel selection instead of falling back to the full set
  of channels.
- keep per-card image notes focused on card-local image matching; channel
  schema compatibility is a shared batch concern and belongs below the cards;
- slice 4 does not need to fully switch the widget to `bind_batch(...)` yet,
  but it should stop deriving channel UI from `self._selected_image_option`
  alone.

Expected outcome:

- the batch channel rule is visible in the UI;
- channel compatibility is validated before backend submission.

### 5. Finalize Batch Binding and Submission Flow

Status: [x] Completed

Goal:

- make staged batch assembly, table flow, and controller binding clean and
  submission-ready.

Scope:

- ensure the calculate button is gated by batch validity, table validation, and
  controller readiness;
- ensure the widget builds one authoritative staged batch request from the
  currently checked cards;
- ensure the table selector is batch-aware and shows only tables that annotate
  every currently staged segmentation region;
- ensure the widget binds one explicit staged batch request into the controller;
- if any checked card is invalid, do not bind a partial valid subset into the
  controller; instead, keep the invalid card visible in widget-local staged
  state, block submission, and bind no batch request at all until the full
  checked batch is valid.

Implementation detail:

- slice 5 should introduce one authoritative helper that resolves the staged
  batch request from the currently checked cards, analogous to slice 4's
  `_resolve_batch_channel_state()`;
- that helper should become the source of truth for:
  - the decision to call `bind_batch(...)` versus binding no batch;
  - the calculate-button enabled state;
  - widget-local batch validity decisions that feed later UI messaging;
- a recommended shape is:

```python
@dataclass(frozen=True)
class _FeatureExtractionStagedBatchState:
    checked_coordinate_systems: tuple[str, ...]
    label_names: tuple[str, ...]
    triplets: tuple[FeatureExtractionTriplet, ...]
    invalid_coordinate_systems: tuple[str, ...]
    error_text: str | None

    @property
    def is_bindable(self) -> bool:
        return self.error_text is None and bool(self.triplets)


def _resolve_staged_batch_state(self) -> _FeatureExtractionStagedBatchState:
    ...
```

- `_resolve_staged_batch_state()` should:
  - walk checked cards in sorted visible-card order;
  - collect the ordered set of checked-card segmentation label names
    separately from the bindable triplet tuple, so batch table discovery can
    still operate even when the full checked batch is not yet bindable;
  - stage one explicit `FeatureExtractionTriplet` per checked card only when
    that card has a valid segmentation selection and, when needed, a valid
    image selection;
  - preserve invalid checked cards in returned widget-local state rather than
    silently dropping them from the staged batch model;
  - return `triplets=()` and one shared batch `error_text` whenever any
    checked card is invalid, so the widget binds no batch request at all
    rather than binding a partial valid subset;
  - keep `label_names` populated for the checked cards whose segmentation
    choice is already known, even when `triplets=()` because the full batch is
    still invalid;
  - return the full ordered triplet tuple only when every checked card is
    currently valid;
- use that staged-batch helper before table discovery:
  - while not every checked card has a valid segmentation selection, keep the
    table combo disabled because the required batch label set is not yet known;
  - once the staged batch is valid enough to expose `label_names`, show only
    tables that annotate every staged batch label name rather than showing a
    broader list plus a later incompatibility error;
  - for batch table discovery, derive selectable tables from the staged batch
    label set rather than from the active card's single selected segmentation;
- after table discovery, validate the selected output table against all staged
  triplet label names as a later batch-level step:
  - use `validate_table_annotation_coverage(...)` to confirm full region
    coverage for the batch;
  - use `validate_table_region_instance_ids(...)` to confirm per-region
    `instance_key` uniqueness before submission;
  - do not keep slice 5 tied to the old single-label
    `validate_table_binding(...)` path for batch validation;
- once the staged batch and selected table are both valid, call
  `bind_batch(...)` with the full explicit triplet list; otherwise bind no
  batch and let the widget own the blocking feedback.
- the calculate button should be enabled only when all of the following are
  true, in this order:
  - a `SpatialData` object is loaded;
  - at least one coordinate system is checked;
  - the staged batch state is bindable;
  - a batch-eligible output table is selected;
  - batch table validation passes for all staged triplet label names;
  - at least one feature is selected;
  - the output feature key is non-empty;
  - the controller is bound to that same full staged batch;
  - `controller.can_calculate` is `True`;
- implementation note for that enablement list:
  - items 1, 4, 6, 7, and 9 already have natural hooks in the current widget
    or controller state;
  - items 3 and 5 should be implemented through the new staged-batch and
    batch-table helpers introduced in slice 5;
  - item 8 should be supported by one small read-only controller exposure so
    the widget can verify cleanly that the controller is bound to the same full
    staged batch it is currently presenting in the UI.
- treat that ordered enablement list as the disabled-tooltip precedence too:
  - when the button is disabled, show the first unmet prerequisite from that
    same list rather than combining several blocking reasons at once;
  - keep the tooltip widget-owned until a full bindable batch exists and has
    been bound;
  - only fall through to controller-owned messaging once widget-side batch and
    table validation have passed and the remaining blocking reason is truly
    controller-owned;
  - when the button is enabled, leave the tooltip empty.
- for slice 5, status-card behavior may remain intentionally coarse while the
  richer status-surface ownership work is deferred to slice 6:
  - do not use placeholder strings such as `to be implemented` in the widget;
  - `selection_status` should own the interim coarse batch-assembly and
    table-validity messages, while `controller_feedback` should stay hidden
    unless a bindable batch exists and the controller has real post-bind state
    to report;
  - prefer short but accurate batch-level messages such as
    `Choose Coordinate Systems`, `Batch Incomplete`,
    `Selected table does not annotate all staged segmentations.`,
    `N extraction targets staged.`, or `Batch ready to bind.`;
  - if no bindable batch is currently staged, it is acceptable for
    `controller_feedback` to stay hidden rather than attempting to preview the
    later slice-6 messaging contract.

Expected outcome:

- the widget assembles one authoritative staged batch from the checked cards;
- the table selector reflects only batch-eligible output tables;
- the controller never reflects a silently reduced subset of the checked cards;
- submission stays blocked until the full checked batch and selected table are
  valid;
- controller binding reflects the same batch request the widget is prepared to
  submit;
- slice 5 can ship with minimal but accurate interim status messages while
  detailed staged-batch summaries remain deferred to slice 6.

### 6. Finalize Status-Card Ownership and Messaging

Goal:

- make the batch UI communicate staged-batch state clearly without mixing
  widget-local validation with controller lifecycle feedback.

Scope:

- ensure the widget reports which triplets will be written;
- keep the current two status-card surfaces, but give them strict ownership:
  - `selection_status` should become the widget-owned staged-batch summary and
    batch-validity card;
  - `controller_feedback` should become the controller-owned bound-request and
    execution-lifecycle card;
- `selection_status` should be driven from the staged batch state plus
  batch-level table validation, not from active-card-only selection state and
  not from controller readiness alone;
- `selection_status` should:
  - report one line per checked card, because one checked card corresponds to
    one intended triplet slot in the staged batch;
  - show valid checked cards as concrete triplet summaries such as
    `coordinate_system: segmentation -> image` or
    `coordinate_system: segmentation -> no image` for morphology-only targets;
  - keep invalid checked cards visible in that summary with a short blocking
    reason rather than omitting them from the card;
  - summarize overall staged-batch state with titles such as
    `Choose Coordinate Systems`, `Batch Incomplete`, `Table Not Ready`, or
    `Batch Ready`;
  - use tooltip text for the full unshortened staged triplet summary when
    names are long;
- `controller_feedback` should:
  - report only controller-owned state after binding, such as
    `ready to calculate`, `calculating N extraction targets`, successful write
    completion, or backend failure;
  - it is acceptable for `controller_feedback` to show `ready to calculate`
    while `selection_status` simultaneously shows `Batch Ready`; two green
    cards are acceptable here because they communicate different ownership:
    staged-batch readiness versus controller-bound execution readiness;
  - not own widget-local validation messages such as missing segmentation,
    invalid checked cards, batch table coverage issues, or shared channel
    compatibility errors;
  - stay hidden when no bindable batch is currently staged, rather than
    duplicating widget-owned blocking feedback;
- the calculate button's blocking tooltip/message should follow the same
  top-level staged-batch validity that `selection_status` reports, so the
  batch summary, button state, and controller binding all describe the same
  request;
- tighten status-card and tooltip messaging for the new batch flow.

Implementation detail:

- to keep `_feature_extraction_widget.py` from accumulating too much status
  formatting logic, slice 6 may introduce a helper module such as
  `_feature_extraction_status_card.py`;
- that helper module should remain presentation-focused:
  - it may derive normalized status-card content from widget/controller state;
  - it should not own widget event wiring, controller binding, or Qt widget
    mutation directly;
  - prefer returning plain dataclasses or other immutable payloads that the
    widget can pass into `set_status_card(...)`;
- a recommended helper-owned per-card summary shape is:

```python
@dataclass(frozen=True)
class _FeatureExtractionStatusCardEntry:
    coordinate_system: str
    label_name: str | None
    image_name: str | None
    blocking_reason: str | None = None

    @property
    def is_valid(self) -> bool:
        return self.blocking_reason is None
```

- this row model should represent semantic staged-batch summary data, not
  preformatted visible strings:
  - one row per checked card, in checked-card order;
  - valid rows carry the concrete triplet pieces that will be written;
  - invalid rows keep the same card visible but add a short blocking reason;
  - batch-level table problems remain batch-level card state, not per-row
    blocking reasons;
- a recommended helper-owned status-card payload shape is:

```python
@dataclass(frozen=True)
class _FeatureExtractionStatusCardSpec:
    title: str
    lines: tuple[str, ...]
    kind: StatusCardKind
    tooltip_message: str | None = None
```

- `_feature_extraction_status_card.py` should then own:
  - building `_FeatureExtractionStatusCardEntry` values from the staged batch
    and per-card widget state;
  - visible-line formatting for `selection_status`;
  - tooltip-line formatting for the full unshortened staged batch summary;
  - title/kind resolution for `selection_status`;
  - controller-feedback title/body formatting for `controller_feedback`;
- visible-line formatting should:
  - shorten long coordinate-system, segmentation, and image names with
    `format_feedback_identifier(...)`;
  - produce lines such as
    `global: blobs_labels -> blobs_image`,
    `global: blobs_labels -> no image`, or
    `aligned: choose an image`;
  - keep the visible wording compact and stable enough for focused tests;
- tooltip-line formatting should:
  - use full unshortened names;
  - preserve one line per checked card;
  - add table-specific detail only when the batch-level table state needs it;
- title/kind resolution for `selection_status` should stay centralized in the
  helper so the widget does not reimplement card-state wording inline:
  - `Choose Coordinate Systems`, `Batch Incomplete`, `Table Not Ready`, and
    `Batch Ready` remain the primary staged-batch titles;
  - the helper should choose the title/kind from staged-batch validity plus
    batch-level table state, not from controller readiness;
- controller-feedback formatting should also stay centralized in the helper:
  - strip the leading `Feature extraction:` prefix from controller messages for
    display;
  - map controller status kinds to stable titles such as
    `Feature Extraction`, `Feature Extraction Ready`,
    `Feature Extraction Warning`, or `Feature Extraction Error`;
  - return no card spec when `controller_feedback` should stay hidden;
- the calculate-button blocking tooltip may continue to be wired from widget
  logic, but slice 6 should keep its top-level wording aligned with the same
  staged-batch state titles and blocking reasons used by
  `selection_status`.

Expected outcome:

- users can review the staged triplets before launching work;
- `selection_status` and `controller_feedback` no longer compete to explain
  widget-local validation conditions, even if both are simultaneously positive
  for a fully ready bound batch;
- controller lifecycle feedback appears only when a real bindable batch exists;
- UI messaging reflects the same staged batch request the widget is preparing
  to submit;
- the widget is aligned with the step-3 acceptance criteria in
  `cross_sample_tables.md`.

## Suggested Execution Order

Recommended order:

1. extract a single triplet card;
2. add checkbox selection and multi-card rendering;
3. add batch validity rules;
4. add shared batch channel selection;
5. finalize batch binding and submission flow;
6. finalize status-card ownership and messaging.

## Testing Strategy

For each slice:

- update only the tests that correspond to the newly introduced behavior;
- keep the rest of the widget contract stable where possible;
- run focused widget and controller tests before moving to the next slice.
- for slice 3 specifically, expect a deliberate rewrite of the most
  slice-2-shaped widget tests rather than a minimal assertion tweak:
  - tests that currently expect a segmentation to be auto-selected as soon as
    a coordinate-system card is shown should be replaced with assertions that
    the card starts at `Choose a segmentation mask`;
  - tests that currently expect remembered restoration to immediately yield a
    selected segmentation should be updated to cover both successful restore
    and blocked-restore reset-to-placeholder behavior;
  - tests should explicitly cover cross-card duplicate-segmentation exclusion
    and in-place visible-card state recomputation after segmentation changes.
- plan that targeted test replacement up front so the first slice-3 patch does
  not look unexpectedly noisy just because old slice-2 assumptions are being
  removed.

Recommended focused test progression:

1. `tests/test_feature_extraction_widget.py`
2. `tests/test_feature_extraction.py`
3. broader integration checks only after the slice is stable
