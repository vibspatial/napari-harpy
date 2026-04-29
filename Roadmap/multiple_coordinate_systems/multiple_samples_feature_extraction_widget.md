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
- during the refactor, the older option-only feature-extraction helpers may
  remain temporarily as thin wrappers over these discovery helpers, but the
  discovery helpers are the intended source of truth and the end-state API the
  widget should consume directly.
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

Goal:

- move from per-image channel handling to one shared batch channel selector for
  intensity extraction.

Scope:

- derive the shared channel selector from the first explicitly selected image
  in sorted card order;
- hide channel selection when no selected image exposes channels;
- block intensity batches when later selected images expose incompatible
  ordered channel-name schemas;
- keep morphology-only execution possible without images.

Expected outcome:

- the batch channel rule is visible in the UI;
- channel compatibility is validated before backend submission.

### 5. Finalize Batch Status, Binding, and Submission Flow

Goal:

- make the batch UI communicate clearly and bind cleanly to the batch-aware
  controller.

Scope:

- ensure the widget reports which triplets will be written;
- ensure the calculate button is gated by batch validity, table validation, and
  controller readiness;
- ensure the widget binds one explicit staged batch request into the controller;
- tighten status-card and tooltip messaging for the new batch flow.

Expected outcome:

- users can review the staged triplets before launching work;
- controller binding and UI messaging reflect the same batch request;
- the widget is aligned with the step-3 acceptance criteria in
  `cross_sample_tables.md`.

## Suggested Execution Order

Recommended order:

1. extract a single triplet card;
2. add checkbox selection and multi-card rendering;
3. add batch validity rules;
4. add shared batch channel selection;
5. finalize status, binding, and submission flow.

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
