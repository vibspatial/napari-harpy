# Cross-Sample Tables

## Purpose

This document covers the follow-on design work that starts after Harpy has a
shared active coordinate system per viewer.

That first phase is described in
`Roadmap/multiple_coordinate_systems/multiple_coordinate_systems.md` and should
stay focused on:

- one active coordinate system per `HarpyAppState`;
- synchronized coordinate-system selectors across widgets;
- pruning Harpy-managed layers from inactive coordinate systems;
- keeping viewer state, widget state, and controller bindings aligned.

This document covers the next problem:

- one AnnData table may legitimately annotate labels regions from multiple
  coordinate systems or samples;
- feature extraction and object classification must then operate on a table that
  spans more than the currently visible sample.

## Core Invariant

Cross-sample tables do not change the viewer invariant.

Harpy should still show and interact with one active coordinate system at a
time. The viewer remains sample-local even when the selected table is
table-global.

In short:

```text
viewer context: one coordinate system
table context: one or more coordinate systems / labels regions
```

## Problem

Valid workflows exist where:

- coordinate system `sample_1` contains image and labels for sample 1;
- coordinate system `sample_2` contains image and labels for sample 2;
- one AnnData table contains rows for both labels regions.

This is not a viewer problem. It is a table semantics and workflow problem.

The viewer should still operate on one sample at a time, but:

- feature extraction may need to populate one feature matrix from multiple
  labels regions;
- classifier training may need to use labeled rows from multiple regions;
- classifier prediction may need to update either the active region only or the
  entire selected table.

## Current Codebase Baseline

The current codebase already provides part of the needed behavior.

### Feature Extraction

Today, `FeatureExtractionWidget` and `FeatureExtractionController` are
single-region and coordinate-system-first:

- one selected coordinate system;
- one selected segmentation;
- zero or one selected image;
- one selected table;
- one selected output key.

The current UI flow is already ordered roughly as:

```text
coordinate system -> segmentation -> image -> table -> output key
```

but it is still limited in important ways:

- it only supports one selection at each step;
- labels and images are filtered only by coordinate-system membership;
- segmentation options are not yet surfaced with explicit "valid /
  invalid-for-feature-extraction" eligibility states;
- the widget does not yet auto-derive one matching image per selected
  segmentation;
- output table selection is still a single-region binding step rather than a
  cross-sample request validation step.

The controller already treats coordinate-system rebinding as context change and
cancels stale work. That behavior should stay.

### Object Classification

Today, `ObjectClassificationWidget` is also sample-local:

- annotation binds to one selected segmentation;
- styling binds to one loaded labels layer;
- classifier preparation and prediction operate on rows whose `region_key`
  matches the active segmentation.

This is a good default for interactive work, but it is narrower than the
cross-sample table use case.

### Table Validation

Current table binding already has an important cross-sample property:

- duplicate `instance_key` values are rejected within a selected `region_key`
  region, not globally across the entire table.

That should remain the rule.

### Backend Capability

The current Harpy backend already supports multi-region feature extraction by
accepting list-valued `labels_layer`, `img_layer`, and `to_coordinate_system`
inputs and aligning results by `region_key` plus `instance_key`.

That means the main missing work in napari-harpy is UI, selection modeling,
eligibility validation, and controller orchestration rather than inventing a
new backend feature format from scratch.

## Data Model

The selected AnnData table should remain the authoritative cross-sample object
table.

Its row identity is:

```text
region_key + instance_key
```

This means:

- the same `instance_key` value may appear in different regions;
- table-level operations must always be explicit about which regions they read
  and which regions they write;
- feature matrices written into `.obsm[feature_key]` are table-level assets even
  when calculated region by region.

## Non-Goals

This document does not propose:

- showing multiple coordinate systems in the same viewer at once;
- replacing the phase-1 shared-coordinate-system design;
- making every workflow table-wide by default;
- making hidden samples silently update without clear UI communication.

## Feature Extraction Design

### Why This Needs Its Own Model

Cross-sample feature extraction should not be modeled as two independent
multi-select lists:

- multiple segmentation masks;
- multiple images.

That would allow ambiguous or invalid pairings.

The correct unit is an explicit per-sample triplet:

```text
coordinate system / sample
labels region / segmentation mask
matching image
```

The full feature-extraction request is then:

```text
one or more triplets
one shared channel selection for batch intensity extraction
one output AnnData table
one output feature key
```

`matching image` still remains optional at execution time when the chosen
feature set is morphology-only, but the UI model should stay segmentation-led:

- first choose a coordinate system and segmentation;
- then derive matching images for that segmentation;
- then validate the output table against the selected segmentation regions.

For channels:

- batch feature extraction uses one shared channel selection across all
  selected triplets when intensity-derived features are requested;
- cards are rendered in sorted coordinate-system order;
- that shared selector is read from the first card in that stable order that
  currently has an image selected;
- every later triplet must expose the same ordered channel-name list to remain
  intensity-compatible with the batch;
- if a later triplet's image exposes a different channel schema, that triplet
  must be blocked from joining the batch for intensity extraction with a short
  explanatory reason;
- controller validation should continue to reject mixed selected-channel states
  across triplets before backend submission.

Selection uniqueness rule:

- a segmentation / labels element may appear at most once across the staged
  batch request, even if that same element is available in more than one
  coordinate system;
- each checked coordinate-system card starts with no segmentation selected and
  a placeholder such as `Choose a segmentation mask`; Harpy does not
  auto-select the first available segmentation in that card;
- once the user explicitly selects a segmentation in one card, Harpy removes
  that segmentation from the selectable choices in the other selected cards
  and may show a short inline note explaining where it is already selected;
- if a card is restored from remembered widget state with a segmentation that
  has since become blocked by another selected card, Harpy clears the card
  back to `Choose a segmentation mask`, shows a short explanatory note, and
  drops that remembered selection rather than silently restoring it later;
  Harpy also clears the remembered image identity for that card at the same
  time so stale image memory cannot survive the failed restore;
- each card therefore distinguishes between its current staged selection and
  its currently selectable segmentation options: the current staged selection
  is either empty or one valid segmentation, while the selectable options may
  exclude segmentations already chosen in other selected cards;
- this is deliberate, because duplicate segmentation selection would create
  ambiguous repeated writes against the same table region identity;
- an image element may be reused across multiple triplets when it is a valid
  match for each selected segmentation;
- this asymmetry is deliberate: duplicate image reuse does not by itself imply
  duplicate row writes, while duplicate segmentation reuse does.
- this deliberate asymmetry should also be documented in a code docstring near
  the batch triplet validation / normalization path so future refactors do not
  “simplify” it away accidentally.

### Viewer Context vs Extraction Request

The viewer invariant still holds:

- the napari viewer shows one active coordinate system at a time.

But the feature-extraction request model may be broader than the viewer:

- the widget may stage one or more triplets from one or more coordinate
  systems;
- those triplets are widget-local request state;
- they should not be forced into one shared `HarpyAppState.coordinate_system`
  value.

This means cross-sample feature extraction is allowed to be broader than the
currently visible sample, even though viewer interaction stays sample-local.

Lifecycle rule:

- `FeatureExtractionWidget` does not subscribe to
  `HarpyAppState.coordinate_system_changed`;
- `FeatureExtractionWidget` does not publish its coordinate-system choices
  through `HarpyAppState.coordinate_system`;
- on `sdata_changed`, discard any staged triplets and rebuild the available
  triplets from a fresh local widget state.

### Intended Workflow

1. The user selects one coordinate system or several coordinate systems via a
   checkbox list.
2. Harpy renders one triplet card per selected coordinate system.
3. One selected coordinate system maps to one triplet card and therefore to
   one `coordinate_system -> segmentation -> image` triplet.
4. The same segmentation element cannot be selected in two different cards,
   even if it appears in more than one selected coordinate system.
5. Inside each card, Harpy shows a segmentation combo filtered to that
   coordinate system.
6. Each newly shown card starts with an explicit placeholder such as
   `Choose a segmentation mask`; Harpy does not auto-select the first
   available segmentation in that coordinate system.
7. Only transform-eligible segmentation masks should appear as selectable
   options in that combo.
8. If the user explicitly selects one segmentation in another card, Harpy
   immediately removes that segmentation from this card's selectable choices
   and may show a short inline note such as
   “`segmentation_1` is unavailable because it is already selected in
   `global_1`.”
9. If additional segmentation masks exist in the same coordinate system but
   are not feature-extraction-eligible, Harpy should communicate that with a
   short inline note such as “2 segmentations unavailable due to unsupported
   transform” rather than cluttering the combo with disabled items.
10. If a selected coordinate system has no eligible segmentation masks at all,
   the card should still be rendered, but the segmentation selector should be
   empty or disabled and the card should show a short inline message such as
   “No selectable segmentations in this coordinate system.”
11. That card cannot contribute a staged triplet until a valid segmentation
   becomes selectable.
12. If a card is restored from remembered widget state with a segmentation
    that has since become blocked by another selected card, Harpy clears the
    card back to `Choose a segmentation mask`, shows a short explanatory note,
    and drops that remembered selection rather than silently restoring it
    later. Harpy also clears the remembered image identity for that card at
    the same time.
13. For the selected segmentation in each card, Harpy derives matching image
   candidates in the same coordinate system.
14. Only selectable matching images should appear in the per-card image combo.
15. If additional images exist in the same coordinate system but are not valid
   matches because of shape mismatch, transform mismatch, or unsupported
   transform semantics, Harpy should communicate that with a short inline note
   rather than cluttering the combo with disabled items.
16. The same image element may still be reused across several cards when it is
    a valid match for each card's selected segmentation.
17. Harpy resolves a concrete `coordinate_system -> segmentation -> image`
    triplet for each selected coordinate system:
   - if exactly one matching image exists, show it as the only selectable
     image option but keep the card at `No image` until the user explicitly
     chooses it;
   - if multiple matching images exist, require an explicit user choice;
   - if no matching image exists, allow only morphology-only extraction.
18. Below the cards, Harpy shows one shared channel-selection area, one output
   table selector, one feature-matrix-key field, and the shared feature
   groups.
19. The user selects the output AnnData table.
20. Harpy validates that the selected table annotates every selected
   segmentation region.
21. Harpy blocks submission while any selected coordinate-system card remains
    invalid rather than silently dropping that selected card from the staged
    batch request.
22. Harpy submits one explicit multi-target feature-extraction request.
23. Harpy writes feature rows back into the same AnnData table, aligned by
    `region_key` plus `instance_key`.

### Segmentation Eligibility

A segmentation mask is feature-extraction-selectable only when:

- it is available in the chosen coordinate system;
- its transform in that coordinate system is supported by Harpy feature
  extraction;
- the transform resolves to pure `x` / `y` translation semantics.

At the time of writing, Harpy feature extraction supports transforms that
resolve to pure `x` / `y` translation, including:

- identity;
- translation;
- a sequence of translations;
- an affine transform whose affine matrix is equivalent to pure `x` / `y`
  translation.

Transforms that imply rotation, shear, or scale should make the segmentation
visibly unavailable for feature extraction in the UI rather than silently
selectable.

### Image Matching Rules

Harpy should never build a Cartesian product of selected labels and selected
images.

For one selected segmentation mask, an image is a valid match only when:

- the image is available in the same coordinate system as the segmentation;
- the image has the same spatial shape as the segmentation, ignoring the image
  channel axis;
- the image and segmentation resolve to the same effective transform in that
  coordinate system;
- the transform is supported by Harpy feature extraction.

This is the operational meaning of a `matching image` in the UI.

### Table Eligibility

The output table is selected after the triplets are assembled.

That table is valid only when:

- the selected table annotates every selected segmentation region;
- duplicate `instance_key` values are rejected within each selected region;
- the request does not mix segmentation-image pairs that violate the matching
  rules above.

### UI Direction

The current single-region flow should remain recognizable, but it should grow
into a triplet-based model rather than a table-first model.

Suggested direction:

- keep the selector order centered on `coordinate system -> segmentation ->
  image -> table`;
- for one-sample extraction, the widget operates on one triplet;
- for cross-sample extraction, the widget operates on several explicit triplets;
- group triplets by coordinate system or sample;
- show segmentation masks in the selected coordinate systems even when they are
  not currently eligible, but make only eligible ones selectable;
- show only matching images for a chosen segmentation rather than an
  independent image universe;
- auto-fill the image when exactly one valid match exists;
- require explicit image choice when several valid matches exist;
- validate the selected output table against all chosen segmentation regions.

Examples of unavailable reasons:

- the segmentation transform is not supported for feature extraction;
- no aligned image is available for intensity features;
- the image transform is not supported;
- the table does not annotate that labels region.

### Write Semantics

Cross-sample feature extraction should be presented as one explicit table-level
operation, even if the backend computes it as a batch of per-target jobs.

Important behavior:

- write features back only to rows matching each triplet’s `region_key` and
  `instance_key`;
- allow one table to accumulate features for multiple triplets / regions;
- do not present repeated single-region writes as the intended UX for building a
  shared cross-sample feature matrix.

The current implementation can already fill a shared table incrementally. Treat
that as implementation capability, not as the preferred workflow the UI should
encourage.

## Object Classification Design

Object classification is easier than feature extraction because interactive
annotation is already naturally sample-by-sample.

The intended first behavior should be:

- choose the active coordinate system;
- choose a segmentation in that coordinate system;
- annotate objects in the visible sample;
- write annotations into the shared table rows for that segmentation region;
- switch coordinate system and continue on another sample if needed.

This keeps interaction local while allowing one table to accumulate labels
across regions.

### Training Scope

Recommended default:

- training uses all eligible labeled rows in the selected table.

This makes classifier training table-level by default, which is usually what the
user wants once multiple samples contribute annotations to the same table.

Optional non-default mode:

- training uses only the selected region.

So the intended UX should expose training scope explicitly, for example:

- selected region only;
- all eligible labeled regions in the selected table.

Here, `selected region only` means:

- the `region_key` for the segmentation currently selected in
  `ObjectClassificationWidget`.

Classifier-eligible training rows should be defined explicitly:

- row belongs to the selected table;
- row belongs to the chosen training region scope;
- row has valid `region_key` and `instance_key` values;
- row has finite, non-missing values in the selected feature matrix;
- row has a user class other than the unlabeled sentinel.

### Prediction Scope

Recommended default:

- prediction updates only the selected region.

Optional non-default mode:

- prediction updates all eligible rows in the selected table.

So the intended UX should expose prediction scope explicitly, for example:

- selected region only;
- all eligible regions in the selected table.

For prediction, `selected region only` has the same meaning:

- the `region_key` for the segmentation currently selected in
  `ObjectClassificationWidget`.

This split keeps interactive retraining responsive and avoids unexpectedly
writing predictions into hidden rows.

The UI should make the scope visible before running prediction. For example:

```text
Training: 182 labeled rows across 4 regions
Prediction: 12,440 rows in selected region
```

When complete-table prediction is selected, the UI should explicitly warn that
rows outside the visible coordinate system may be updated.

### Metadata

Classifier metadata stored in `table.uns` should become more explicit once table
scope and prediction scope diverge.

Recommended additions:

- `training_scope`
- `training_regions`
- `n_training_rows`
- `prediction_scope`
- `prediction_regions`
- `n_predicted_rows`

Encoding rule:

- `training_scope` and `prediction_scope` store the user-facing mode, e.g.
  `all` or `selected_region_only`;
- `training_regions` and `prediction_regions` store the concrete resolved list
  of `region_key` values actually used for that run;
- when scope is `all`, resolve that mode at execution time to the full set of
  eligible `region_key` values and persist that expanded list;
- when scope is `selected_region_only`, resolve it at execution time to the
  one `region_key` for the segmentation currently selected in
  `ObjectClassificationWidget`, and persist that one-item list.

This helps future reload, debugging, and user-facing status text.

## Proposed Phase-2 Implementation Plan

### 1. Add Cross-Sample Table Helpers

Files:

- `src/napari_harpy/_spatialdata.py`
- `tests/test_spatialdata.py`

Work:

- [x] add helpers that derive table-annotated labels regions from
  `SpatialDataTableMetadata.regions`;
- [x] add helpers that derive feature-extraction-eligible segmentations per
  coordinate system;
- [x] add helpers that derive matching images for a
  `(coordinate system, segmentation)` pair;
- [x] add explicit validation helpers for per-region duplicate instance ids and
  table annotation coverage.

Acceptance:

- [x] one table can expose multiple annotated labels regions;
- [x] helper output is deterministic and sorted where appropriate;
- [x] invalid regions or duplicate ids are rejected with region-specific messages.

### 2. Add Batch Feature-Extraction Selection Model

Files:

- `src/napari_harpy/widgets/_feature_extraction_widget.py`
- `src/napari_harpy/_feature_extraction.py`
- `tests/test_feature_extraction_widget.py`
- `tests/test_feature_extraction.py`

Work:

- [x] model a batch request as explicit
  `coordinate_system -> segmentation -> image` triplets plus one output table,
  not as independent labels and image selections;
- [x] extend the controller to submit multi-target requests to Harpy;
- [x] preserve current single-triplet mode as the simple path;
- [x] keep feature extraction decoupled from `HarpyAppState.coordinate_system`;
- [x] on `sdata_changed`, clear any staged triplets and rebuild the available
  triplet choices from fresh local state;
- [x] make stale-work cancellation still happen when the widget-local extraction
  request changes.

Acceptance:

- [x] multi-region extraction is submitted as one explicit multi-target request;
- [x] no triplet pairs a segmentation with an image from a different coordinate
  system or with a non-matching transform / shape;
- [x] one shared feature matrix can be populated for multiple regions.

### 3. Add Batch Feature-Extraction UI

Files:

- `src/napari_harpy/widgets/_feature_extraction_widget.py`
- `src/napari_harpy/_spatialdata.py`
- `tests/test_feature_extraction_widget.py`
- `tests/test_spatialdata.py`

Work:

- extend the widget from one triplet to several triplet cards;
- let users select one or more coordinate systems through a checkbox list, and
  render one triplet card per selected coordinate system;
- render those cards in sorted coordinate-system order;
- treat one selected coordinate system as one triplet;
- forbid selecting the same segmentation element in more than one card, even
  when it appears in multiple coordinate systems;
- allow reusing the same image element across cards when it remains a valid
  match for each card's selected segmentation;
- document that segmentation/image asymmetry in a nearby code docstring when
  implementing the batch selection validation;
- move feature-extraction discovery in `_spatialdata.py` to structured
  discovery helpers that expose both eligible options and coarse unavailable
  counts, so the widget can show short inline notes without duplicating
  matching logic;
- use discovery helpers of the following shape as the source of truth for the
  widget:
  - `SpatialDataFeatureExtractionLabelDiscovery` with
    `eligible_label_options`, `coordinate_system_label_count`, and
    `unavailable_label_count`;
  - `SpatialDataFeatureExtractionImageDiscovery` with
    `eligible_image_options`, `coordinate_system_image_count`, and
    `unavailable_image_count`;
  - `get_spatialdata_feature_extraction_label_discovery_for_coordinate_system_from_sdata(...)`;
  - `get_spatialdata_feature_extraction_image_discovery_for_coordinate_system_and_label_from_sdata(...)`;
- in each card, keep the existing segmentation and image selectors filtered to
  that coordinate system;
- show only selectable segmentations in the per-card combo box, and surface
  unavailable segmentation counts with short inline reasons;
- still render a selected coordinate-system card when it has no eligible
  segmentation, but keep that card non-contributing until it becomes valid;
- show only selectable images in the per-card combo box, and surface
  unavailable image counts with short inline reasons;
- when exactly one matching image exists for a selected segmentation, show it
  as the only selectable image option but keep the card at `No image` until
  the user explicitly chooses it;
- when multiple matching images exist, require an explicit user choice;
- when no matching image exists, keep the card morphology-only-capable rather
  than silently fabricating an image selection;
- surface missing-image states with short reasons;
- for intensity features, show one shared batch channel selector derived from
  the first explicitly selected image rather than independent per-triplet
  channel controls;
- block triplets with incompatible ordered channel-name schemas from joining an
  intensity batch, with a short reason;
- keep the shared `Channels`, `Table`, `Feature matrix key`, `Intensity
  Features`, and `Morphology Features` controls below the triplet cards;
- block submission while any selected coordinate-system card remains invalid,
  rather than silently omitting that card from the request;
- keep output-table selection as a later validation step over the chosen
  triplets.

Acceptance:

- [x] users can review all selected triplets before launching work;
- [x] unavailable segmentations or invalid image matches are blocked before backend
  submission;
- [x] image selection remains explicit in batch mode: a single valid matching
  image may be shown as the only selectable option, but it is not silently
  auto-selected for the user;
- [x] for intensity extraction, the UI makes it clear that one shared channel
  selection will be used for the whole batch and blocks incompatible image
  channel schemas before submission;
- [x] one selected coordinate system produces one visible triplet card and one
  explicit triplet in the staged batch request;
- [x] the UI prevents duplicate segmentation selection across cards while allowing
  valid image reuse across cards;
- [x] submission stays blocked until every selected coordinate-system card is in a
  valid state;
- [x] the UI communicates which regions / triplets will be written.

### 4. Add Classifier Training and Prediction Scopes

Files:

- `src/napari_harpy/widgets/_object_classification_widget.py`
- `src/napari_harpy/_classifier.py`
- `tests/test_widget.py`
- `tests/test_classifier.py`

Work:

- add training-scope and prediction-scope controls;
- default training to all eligible labeled rows in the selected table;
- support explicit training on the selected region only;
- default prediction to the selected region only;
- support explicit complete-table prediction;
- record scope metadata and row counts.

Acceptance:

- training can use labeled rows from multiple regions by default;
- training can be restricted to the selected region only;
- selected-region-only prediction updates only selected-region rows;
- complete-table prediction updates all eligible rows only when explicitly
  requested;
- UI text makes the write scope clear before work starts.

### 5. Add Integration Tests

Files:

- `tests/test_feature_extraction_widget.py`
- `tests/test_feature_extraction.py`
- `tests/test_widget.py`
- `tests/test_classifier.py`

Recommended scenarios:

- one table annotates multiple labels regions across coordinate systems;
- feature extraction fills one `.obsm[feature_key]` matrix for several regions;
- same `instance_key` value is allowed in different regions;
- duplicate `instance_key` values within one region are rejected;
- classifier training uses labeled rows across regions by default;
- classifier training can be restricted to the selected region only;
- selected-region-only prediction leaves hidden-region prediction rows unchanged;
- complete-table prediction updates eligible rows across all regions only when
  explicitly selected.

## Open Questions

1. Should cross-sample feature extraction launch from the existing widget, or
   from a dedicated dialog?

   Recommendation: start inside the existing widget by evolving the current
   single-region selectors into an explicit triplet list, so the current flow
   remains recognizable.

2. Should batch feature extraction allow targets with no image?

   Recommendation: yes, but only for morphology-only feature sets. Intensity
   features must still require an eligible image.

3. Should classifier retraining automatically use all labeled rows in the table?

   Recommendation: yes by default, but the UI should also allow explicit
   restriction to the selected region only and should show row counts so the
   chosen scope is obvious.

4. Should complete-table prediction overwrite hidden-region predictions every
   time?

   Recommendation: only when explicitly requested. The UI should expose only
   two prediction modes, `selected region only` and `all`, and should make the
   chosen write scope visible before the run starts.

## Summary

Cross-sample tables are a valid and important extension, but they should remain
separate from the shared active coordinate-system refactor.

Phase 1 should make the viewer context unambiguous.

Phase 2 should then make table-level workflows explicit:

- feature extraction becomes a batch of explicit
  `coordinate_system -> segmentation -> image` triplets plus one output table;
- classifier training becomes table-level by default, with optional
  selected-region-only training;
- classifier prediction remains selected-region-only by default, with optional
  complete-table prediction as the only non-default option.

This keeps the viewer simple, the table model explicit, and the roadmap easier
to implement against the current codebase.
