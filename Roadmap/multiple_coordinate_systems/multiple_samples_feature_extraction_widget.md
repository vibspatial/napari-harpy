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

Goal:

- move from one visible coordinate-system selection flow to one or more
  selected coordinate systems rendered as cards.

Scope:

- replace the coordinate-system combo box with a checkbox list;
- render one triplet card per selected coordinate system;
- render cards in sorted coordinate-system order;
- keep one selected coordinate system equal to one triplet card;
- keep shared controls below the cards.

Expected outcome:

- users can select multiple coordinate systems;
- the widget renders one card per selected coordinate system;
- cards still use the existing segmentation/image filtering logic per
  coordinate system.

### 3. Add Batch Validity Rules

Goal:

- make multi-card selection safe before introducing richer shared batch logic.

Scope:

- forbid selecting the same segmentation in more than one card;
- allow valid image reuse across cards;
- keep selected cards visible even when they are temporarily invalid;
- show only selectable segmentations and images in the card combo boxes;
- surface unavailable segmentation/image counts with short inline reasons;
- block submission while any selected card is invalid.

Expected outcome:

- the UI no longer allows ambiguous duplicate segmentation selection;
- invalid cards remain visible and understandable;
- the batch selection model is safe before calculation.

### 4. Add Shared Batch Channel Selection

Goal:

- move from per-image channel handling to one shared batch channel selector for
  intensity extraction.

Scope:

- derive the shared channel selector from the first selected image in sorted
  card order;
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

Recommended focused test progression:

1. `tests/test_feature_extraction_widget.py`
2. `tests/test_feature_extraction.py`
3. broader integration checks only after the slice is stable
