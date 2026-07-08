# Fast Categorical Labels Coloring

Status: investigation; recommended implementation path specified.

## Goal

Make labels-layer coloring by categorical `.obs` columns fast enough for large
segmentation tables.

The first target is the current styled-labels path where users color a labels
element by a table observation column, for example:

- labels: `cell_labels_global_ROI1`
- table: `table_global_ROI1`
- `.obs` categorical column: `leiden`
- rows/cells: roughly `406,611`

The fix should improve categorical `.obs` coloring generally.

## Current Bottleneck

For labels colored by an instance id column, Harpy uses napari's procedural
`label_colormap(...)`. That path is fast because it does not need one explicit
color entry per label id.

For labels colored by a categorical `.obs` column, Harpy currently builds a
napari `DirectLabelColormap` with one entry per visible label id:

```text
label_id -> RGBA
```

This is necessary because the labels image stores object ids, not category
codes. For a column such as `leiden`, the image values are still cell ids, so
napari needs to know which cell id maps to which category color.

Observed timing on the Xenium full-data example:

```text
Color by cell_ID:
  value_kind: instance
  colormap: CyclicLabelColormap / label_colormap(...)
  apply_table_color_source_to_labels_layer: ~0.17 s
  ensure_styled_labels_loaded: ~1.16 s

Color by leiden:
  value_kind: categorical
  colormap: DirectLabelColormap
  apply_table_color_source_to_labels_layer: ~2.35 s
  ensure_styled_labels_loaded: ~2.98 s
```

Isolated styling path:

```text
cell_ID:
  build style:    ~0.0002 s
  apply colormap: ~0.002 s
  set features:   ~0.026 s

leiden:
  build color dict: ~0.059 s
  apply colormap:   ~2.22 s
  set features:     ~0.026 s
```

The main bottleneck is applying/building napari's `DirectLabelColormap` for a
large explicit label-id mapping.

## Napari Constraint

The ideal compact categorical representation would be:

```text
label_id -> category_code
category_code -> RGBA
```

That would store roughly one mapping per label id plus one color per category,
instead of repeating the same RGBA arrays for all labels in the same category.

Napari's current public `DirectLabelColormap` API does not expose that as a
first-class model. It accepts:

```text
label_id -> RGBA
```

Internally, napari does compress labels that share the same RGBA into shared
texture values, but creating the `DirectLabelColormap` through the public
constructor still pays per-entry pydantic validation and color transformation.

So the practical Harpy-side fix is:

- keep the public napari rendering semantics;
- keep one explicit label-id mapping for categorical `.obs` columns;
- make construction of that mapping much faster when Harpy already owns
  validated RGBA arrays.

## Recommended Fix

Add a shared Harpy helper for constructing `DirectLabelColormap` from
pre-normalized RGBA values.

Do not use `DirectLabelColormap.model_construct(...)` directly. It is very fast,
but it bypasses normal `EventedModel` setup, including field event emitters.

The safer fast path is:

1. Build a tiny normal `DirectLabelColormap` with only default/background
   colors. This lets napari/pydantic initialize the model and event emitters.
2. Install Harpy's prevalidated large `color_dict` via low-level attribute
   assignment.
3. Clear napari colormap caches with `_clear_cache()`.
4. Return the resulting normal `DirectLabelColormap` object.

Synthetic check with roughly `406k` labels:

```text
normal DirectLabelColormap(...):          ~1.23 s
tiny normal init + install RGBA dict:     ~0.0003 s
```

This keeps event emitters intact, unlike direct `model_construct(...)`.

## Safety Contract

The fast helper should only accept already-normalized RGBA values.

Expected input contract:

- keys are `int` label ids plus optional `None`;
- background label `0` is present and transparent;
- `None` default color is present;
- values are numeric RGBA arrays with shape `(4,)`;
- values are finite floats in the normal napari color range;
- no string colors are passed into the fast path.

If a caller has string colors or untrusted values, it should either normalize
them first or use the normal `DirectLabelColormap(...)` constructor.

This keeps the pydantic bypass narrow and explicit: Harpy may skip repeated
napari color transformation only after Harpy has already produced numeric RGBA
arrays.

## Implementation Slices

### Slice 1: Shared Fast Colormap Helper

Status: implemented.

Add a helper in a shared labels-viewer module, for example:

```text
src/napari_harpy/viewer/labels_colormap.py
```

Suggested API:

```python
def direct_label_colormap_from_rgba(
    color_dict: Mapping[int | None, np.ndarray],
    *,
    background_value: int = 0,
) -> DirectLabelColormap:
    ...
```

The helper should:

- validate the minimal RGBA contract above;
- construct a tiny normal `DirectLabelColormap`;
- install the full normalized mapping without re-transforming every color;
- clear napari colormap caches;
- return a normal `DirectLabelColormap` with working events;
- not assign the colormap to any layer;
- not call `layer.refresh()` or otherwise trigger viewer refresh work.

Layer assignment belongs to the calling code in later slices. Slice 1 should be
pure construction: input RGBA mapping in, napari `DirectLabelColormap` out.

The intended implementation shape is:

```python
small = {
    None: color_dict[None],
    background_value: color_dict[background_value],
}

cmap = DirectLabelColormap(
    color_dict=small,
    background_value=background_value,
)

object.__setattr__(cmap, "color_dict", color_dict)
cmap._clear_cache()

return cmap
```

The important distinction from the current slow path is that the large
`color_dict` is not passed to `DirectLabelColormap(...)`. The constructor only
sees the tiny default/background mapping, while the already-normalized full
mapping is installed afterwards.

Tests should verify:

- `map(...)` matches normal `DirectLabelColormap(...)` for representative label
  ids, missing ids, and background;
- event emitters such as `events.color_dict` exist;
- `_clear_cache()` can be called after construction;
- invalid keys or non-RGBA values fail loudly;
- the helper does not accept string colors.

Do not add timing assertions to unit tests. A small local benchmark note is fine,
but correctness tests should be deterministic.

### Slice 2: Use Helper In Styled Labels Direct Coloring

Status: implemented.

Update `src/napari_harpy/viewer/labels_styling.py`.

The main path is:

```text
apply_table_color_source_to_labels_layer(...)
  -> _build_obs_column_colormap(...)
  -> _build_categorical_color_dict(...)
  -> _apply_labels_colormap(...)
```

Use the fast helper for all direct styled-labels color maps built by this
module, not only categorical `.obs` columns. This includes:

- categorical `.obs` columns;
- continuous `.obs` columns;
- `.X` / `.var` feature coloring.

`_build_categorical_color_dict(...)` and `_build_continuous_color_dict(...)`
already receive vectorized numeric RGBA rows from
`categorical_rgba_for_values(...)` / `continuous_rgba_for_values(...)` for the
real label ids. The remaining cleanup is to replace the current string
default/background entries:

```python
{None: "transparent", 0: "transparent"}
```

with numeric transparent RGBA arrays, so the entire mapping satisfies
`direct_label_colormap_from_rgba(...)`'s trusted Harpy-generated RGBA contract.

`_apply_labels_colormap(...)` should assign the helper result to
`layer.colormap`:

```python
layer.colormap = direct_label_colormap_from_rgba(layer_color_dict, background_value=0)
```

Do not call an additional explicit `layer.refresh()` after that assignment
unless manual QA proves it is required. In current napari, assigning
`Labels.colormap` already emits the colormap event and calls
`layer.refresh(extent=False)`, so the extra refresh would be redundant work on
large labels layers.

Acceptance criteria:

- coloring by categorical and continuous direct labels sources remains visually
  equivalent;
- background label `0` stays transparent;
- unknown/missing labels use the `None` default color;
- categorical color application triggers only the napari refresh caused by
  `layer.colormap` assignment, not an additional explicit refresh call;
- no table or labels data are rewritten;
- categorical bool, binary integer, categorical dtype, and string-like
  categorical coercion tests keep passing.

### Slice 3: Use Helper In Object-Classification Labels Styling

Status: implemented.

Update `src/napari_harpy/widgets/object_classification/viewer_styling.py` for
all direct labels coloring paths:

- `user_class`
- `pred_class`
- `pred_confidence`

These paths share the same final `DirectLabelColormap(...)` assignment, so they
should share the same fast helper once their color dictionaries follow the
trusted Harpy-generated RGBA contract.

Implementation details:

- import `direct_label_colormap_from_rgba(...)`;
- replace direct `DirectLabelColormap(color_dict=..., background_value=0)`
  construction at the shared object-classification labels assignment point;
- ensure default/background entries use numeric transparent RGBA arrays instead
  of string colors;
- ensure all assigned colors are numeric RGBA arrays before they reach
  `direct_label_colormap_from_rgba(...)`. This includes:
  - class colors from `user_class` and `pred_class`;
  - missing prediction-confidence color, currently represented as
    `MISSING_CONTINUOUS_COLOR`;
  - matplotlib `pred_confidence` colormap outputs;
- build `pred_confidence` colors through one vectorized confidence-specific
  path:
  `confidence_array -> np.clip(..., 0, 1) -> viridis(...) -> label_id -> RGBA`;
  this avoids scalar pandas lookups and scalar matplotlib colormap calls for
  every label;
- keep sparse `user_class` behavior unchanged: unlabeled/default rows should
  still be represented by the default/background colors, and only nonzero
  class labels should need explicit per-label entries;
- keep row-scoped `user_class` updates sparse and avoid rebuilding full feature
  rows when the existing code already avoids that;
- remove the explicit `layer.refresh()` calls that currently follow
  `layer.colormap = ...` in both the full color refresh path and the
  row-scoped `refresh_user_class_colormap_and_feature(...)` path. Napari's
  colormap assignment already refreshes the layer, so these are redundant.
- keep `refresh_user_class_feature(...)` unchanged; that feature-only path
  already avoids a color repaint and explicit refresh.

This slice is only about faster categorical color application.

Acceptance criteria:

- `user_class` and `pred_class` color dictionaries use numeric transparent RGBA
  defaults for `None` and `0`;
- `pred_confidence` color dictionaries use numeric RGBA arrays for missing and
  non-missing confidence colors;
- `pred_confidence` applies the matplotlib colormap once to the full clipped
  confidence array, while still assigning the missing-confidence color to
  missing values;
- existing sparse `user_class` behavior remains intact;
- `pred_class` coloring remains visually equivalent;
- `pred_confidence` coloring remains visually equivalent;
- row-scoped user-class annotation refresh still works;
- fallback full refresh behavior remains available.
- full color refresh and row-scoped user-class color refresh do not call a
  second explicit `refresh()` after colormap assignment.

### Slice 4: Make Styled Labels Restyle Semantics Explicit

Status: implemented.

Current behavior:

- `ViewerAdapter.ensure_styled_labels_loaded(...)` correctly finds and reuses
  an already-loaded styled labels layer when `sdata`, `labels_name`,
  coordinate system, and `style_spec` match.
- However, even for that already-loaded matching layer, it still calls
  `apply_table_color_source_to_labels_layer(...)`.
- That means repeated calls rebuild the table-aligned feature rows, rebuild the
  `label_id -> RGBA` colormap, assign `layer.colormap`, and assign
  `layer.features`.

This should be isolated before benchmarking so we do not mix two different
questions:

- cold load: create the styled labels layer and apply the style for the first
  time;
- explicit update: reuse the existing styled labels layer object, but restyle
  it from the current table state.

Given the current UI label ("Add / Update in viewer"), the repeated-call
restyle is not necessarily a bug. It is the update path. If the user changes a
table column, palette, or `X` values and clicks the button again, Harpy should
refresh the existing overlay instead of silently returning stale colors.

Goal:

- first load still creates and styles the overlay;
- a repeated call for the same already-loaded styled labels layer should reuse
  the layer object but intentionally restyle it from the current table state;
- this behavior should be documented and covered by tests so benchmarking can
  measure cold-load and explicit-update costs separately.

Implementation direction:

1. Do not introduce a styled-labels cache or invalidation mechanism in this
   slice.
2. Keep `ensure_styled_labels_loaded(...)` as the explicit "load or update"
   API:
   - if the styled overlay does not exist, create it and style it;
   - if the styled overlay already exists, reuse the same layer object and
     restyle it.
3. Add or adjust tests to make the repeated-call behavior intentional:
   - first call returns `created=True`;
   - second call with the same style returns the same layer and
     `created=False`;
   - the second call updates the existing layer's colormap/features from the
     current table state.
4. Keep `get_loaded_styled_labels_layer(...)` as the read-only API for callers
   that only want to know whether a matching styled overlay already exists.
   That method must not style or restyle.
5. Update comments/docstrings where helpful so future readers do not interpret
   the repeated restyle as accidental.

Acceptance criteria:

- a repeated `ensure_styled_labels_loaded(...)` call for the same styled labels
  overlay returns the same layer with `created=False`;
- the repeated call restyles the existing layer rather than creating a second
  layer;
- if the backing table values or palette changed between calls, the second call
  reflects those changes in `layer.colormap` and/or `layer.features`;
- first load and distinct style specs still style normally;
- status-card behavior remains unchanged from the user's point of view;
- existing tests that assert no legacy style metadata keys are written to
  `layer.metadata` keep passing.

### Slice 5: Benchmark Cold Load And Explicit Restyle Costs

Status: implemented.

After Slices 1-4, re-measure:

```text
apply_table_color_source_to_labels_layer(...)
ensure_styled_labels_loaded(...)
```

on the Xenium full-data categorical `.obs` case.

Measure cold-load and explicit-restyle paths separately:

- cold load: no styled labels layer exists yet;
- explicit restyle: matching styled labels layer already exists, and
  `ensure_styled_labels_loaded(...)` updates it in place.

If either path is still too slow, investigate the remaining costs separately:

- building the Python `label_id -> RGBA` dictionary;
- napari's first internal label-id-to-texture mapping;
- reslicing/redrawing the current labels view;
- explicit restyling of an already-loaded styled labels layer.

Do not add a cache/invalidation mechanism here. If repeated explicit restyle is
expensive, that is useful benchmark evidence for Slice 6 and/or a separate UX
decision about whether the viewer needs a distinct "Show existing overlay"
action versus an "Update overlay" action.

Benchmark run:

- date: 2026-07-08;
- store:
  `/Users/arne.defauw/VIB/DATA/test_data/sdata_xenium_full_data_core.zarr`;
- labels: `cell_labels_global_ROI1`;
- table: `table_global_ROI1`;
- coordinate system: `global_ROI1`;
- rows: `406,611`;
- repeats: `3`, reported as medians.

End-to-end timings:

```text
instance_cell_ID:
  apply_table_color_source_to_labels_layer: 0.1750 s
  ensure_styled_labels_loaded cold:        0.1794 s
  ensure_styled_labels_loaded restyle:     0.1788 s

categorical_leiden:
  apply_table_color_source_to_labels_layer: 1.1544 s
  ensure_styled_labels_loaded cold:        1.1560 s
  ensure_styled_labels_loaded restyle:     1.1684 s

continuous_total_counts:
  apply_table_color_source_to_labels_layer: 1.2179 s
  ensure_styled_labels_loaded cold:        1.2244 s
  ensure_styled_labels_loaded restyle:     1.2416 s
```

Breakdown:

```text
instance_cell_ID:
  region rows:    0.1097 s
  build style:    0.0002 s
  apply colormap: 0.0031 s
  set features:   0.0265 s
  total:          0.1415 s

categorical_leiden:
  region rows:    0.1099 s
  build style:    0.0599 s
  apply colormap: 0.9205 s
  set features:   0.0270 s
  total:          1.1230 s

continuous_total_counts:
  region rows:    0.1113 s
  build style:    0.1309 s
  apply colormap: 0.9369 s
  set features:   0.0272 s
  total:          1.2173 s
```

Findings:

- The Slice 1 fast helper removed the old multi-second
  `DirectLabelColormap(...)` constructor bottleneck, but large direct labels
  colormaps still spend roughly `0.92-0.94 s` in colormap assignment /
  napari's direct-label mapping path for `406k` explicit labels.
- Cold load and explicit restyle have almost identical timings. The remaining
  cost is not layer creation; it is applying the requested direct labels
  colormap and rebuilding current table-derived features/colors.
- Continuous `.obs` coloring is now in the same performance class as
  categorical `.obs` coloring because both use the direct RGBA helper and both
  still need one explicit label-id entry per object.
- Continuous `.obs` coloring should use the same compact-colormap direction in
  a follow-up slice, but it needs an explicit 256-bin quantization contract
  rather than categorical category-code semantics.
- Instance-key coloring remains much faster because it uses napari's procedural
  `label_colormap(...)` and does not need a `label_id -> RGBA` mapping.
- These measurements support moving to Slice 6 if we want to reduce the
  remaining `label_id -> RGBA` / napari direct-mapping cost.

### Slice 6: Compact Categorical Labels Colormap Prototype

Investigate whether Harpy can support the more compact categorical model:

```text
label_id -> category_code
category_code -> RGBA
```

This is a follow-up investigation/prototype after the safer fast
`DirectLabelColormap` helper. It touches napari internals, so the product bar is
higher: if Harpy introduces a compact subclass, Harpy owns that implementation
as the labels-coloring path. We should not maintain two production ways to build
the same styled-labels colormap.

Current napari 0.7.1 findings:

- `Labels.colormap` normalizes accepted colormaps through napari's labels
  colormap machinery. A custom object must be a `DirectLabelColormap` or
  `CyclicLabelColormap` instance, otherwise the public normalizer rejects it.
- For high-bit labels, napari's direct rendering path already converts raw
  label ids to compact texture ids, then uploads a separate texture-id-to-RGBA
  table. For example, if the labels image contains object ids
  `[1001, 1002, 1003]` and labels `1001` and `1003` share the same category
  color, napari can render through a compact mapping like
  `1001 -> 1`, `1002 -> 2`, `1003 -> 1`, plus a small color table like
  `1 -> red`, `2 -> blue`.
- The expensive part is that Harpy must currently first provide a repeated
  `label_id -> RGBA` dictionary, and napari then groups identical RGBA values
  back into the compact texture representation.
- A Harpy compact colormap could avoid repeated RGBA storage and avoid napari's
  grouping step by directly providing the already-compact
  `label_id -> texture_code` and `texture_code -> RGBA` mappings.

Important constraints:

- Slice 6 is for categorical `.obs` coloring only.
- Continuous `.obs` coloring is handled separately in Slice 7 via 256-bin
  quantization.
- Do not rewrite the labels image to category codes. The image must keep
  per-cell label identity.
- Do not create a second labels layer just to display categorical colors.
- Do not patch installed napari source code.
- Do not monkeypatch napari globals at runtime.
- Do not implement fallback behavior that keeps both the Slice 1 direct RGBA
  helper and a compact colormap implementation alive for the same categorical
  styled-labels use case. If the compact colormap is productionized, it becomes
  Harpy's owned implementation path.
- During prototyping, the current Slice 1 helper may exist side-by-side with
  the compact subclass as a parity/benchmark reference. After the compact
  subclass is accepted, remove the old categorical styled-labels production
  route instead of maintaining both.

#### Slice 6.1: Napari Direct Colormap Contract Audit

Status: proposed.

Before writing production code, inspect the installed napari version's labels
colormap/rendering path and write down the exact private/public contract a
compact colormap would need to satisfy.

Questions to answer:

- Which `DirectLabelColormap` methods/properties are consumed by
  `Labels._set_colormap(...)`, `Labels.get_color(...)`, and the labels visual?
- Which parts are public enough to rely on, and which are private internals?
- Are these methods/properties the same for:
  - high-bit labels, where napari builds a compact texture mapping;
  - small integer labels, where napari may use a small lookup path;
  - selected-label rendering / `show_selected_label`;
  - multiscale labels?
- What napari behavior does Harpy need to own explicitly in tests if it
  subclasses `DirectLabelColormap`?

Expected output:

- an updated roadmap section listing the required contract;
- a clear recommendation on whether Slice 6.2 is still worth pursuing;
- no production behavior change.

#### Slice 6.2: Compact Mapping Builder

Status: proposed.

Build a Harpy-owned pure data helper for categorical labels, without touching
napari internals yet.

The helper should turn Harpy's existing table-aligned categorical colors into
compact state:

```text
positive label ids
label_id -> category_code
category_code -> RGBA
default RGBA for unmapped labels
transparent RGBA for background label 0
```

This helper should be independent of napari `DirectLabelColormap` subclassing.
It should be easy to test with ordinary NumPy arrays and dictionaries.

Acceptance criteria:

- repeated RGBA colors are stored once per category code;
- background label `0` remains transparent;
- missing/unmapped labels resolve to the same default color as the current
  direct RGBA helper;
- helper output preserves per-cell label ids and never rewrites the labels
  image;
- tests cover representative label ids, missing labels, repeated categories,
  and non-contiguous category codes.

#### Slice 6.3: Prototype Compact DirectLabelColormap

Status: proposed.

Create an isolated prototype, for example
`CompactCategoricalLabelColormap`, that can be assigned where napari expects a
`DirectLabelColormap`.

The prototype should:

- subclass or otherwise satisfy napari's accepted labels colormap type checks;
- store the compact state from Slice 6.2;
- override only the narrow methods/properties identified in Slice 6.1, likely:
  - `_values_mapping_to_minimum_values_set(...)`;
  - `_label_mapping_and_color_dict`;
  - `_num_unique_colors`;
  - `map(...)`;
  - `_clear_cache(...)`;
- preserve enough `color_dict` behavior for `Labels._set_colormap(...)` and
  `Labels.get_color(...)` to keep working;
- avoid looking like a default-only direct colormap, because napari may switch
  the layer back to auto color mode in that case;
- be designed as the single Harpy-owned categorical styled-labels colormap path
  if the prototype is accepted.

This slice is still prototype-level. Do not route production styled labels
coloring through the compact colormap yet.

#### Slice 6.4: Compact Colormap Parity Tests

Status: proposed.

Prove that the compact prototype is behaviorally equivalent to the existing
Slice 1 direct RGBA helper before benchmarking or integration.

Test parity for:

- `map(...)` on representative labels;
- missing/unmapped labels;
- background label `0`;
- repeated colors/categories;
- selected-label behavior if napari routes it through the colormap;
- `Labels.get_color(...)`;
- small integer label ids;
- large/high-bit label ids;
- multiscale labels where feasible in headless tests.

Acceptance criteria:

- visual parity with the Slice 1 helper for categorical `.obs` colors;
- `map(...)` parity for representative labels, missing labels, background, and
  selection mode;
- hover/status feature lookup remains label-id based;
- no loss of per-cell label identity;
- faster or materially lower-memory construction than the Slice 1 helper on the
  `table_global_ROI1.obs["leiden"]` benchmark;
- no regression for 2D, 3D, and multiscale Labels layers that Harpy supports.

#### Slice 6.5: Compact Colormap Benchmark

Status: proposed.

Benchmark the compact prototype against the Slice 1 direct RGBA helper on the
same Xenium full-data `leiden` case used in Slice 5.

Measure:

- compact mapping construction;
- compact colormap construction;
- `layer.colormap` assignment;
- end-to-end `apply_table_color_source_to_labels_layer(...)` equivalent;
- memory-ish size of the resulting Python-side mapping objects, if practical;
- Slice 1 helper timings in the same run as a baseline only, not as a planned
  production fallback.

Acceptance criteria:

- materially faster construction and/or assignment than the Slice 1 helper on
  `table_global_ROI1.obs["leiden"]`;
- or materially lower memory use with no performance regression;
- no slower path is allowed to replace the current helper.

#### Slice 6.6: Integration Decision

Status: proposed.

If the prototype is successful, decide whether to:

- productionize the Harpy subclass as the categorical styled-labels colormap
  implementation; or
- open an upstream napari proposal/PR for a public compact categorical labels
  colormap API.

Productionization requirements:

- focused tests for the Harpy-owned subclass behavior;
- benchmark evidence from Slice 6.5 in the roadmap;
- no product path depending silently on unverified napari internals;
- remove the old categorical direct-RGBA styled-labels production path;
- no duplicate long-term categorical styled-labels colormap implementations.

### Slice 7: Compact Continuous Labels Colormap Via 256 Bins

Status: proposed.

After Slice 6 settles the Harpy-owned compact colormap shape for categorical
`.obs`, extend the same idea to continuous `.obs` values by quantizing colors
into a fixed 256-bin color table:

```text
label_id -> color_bin
color_bin -> RGBA
```

Rationale:

- Continuous `.obs` styling now has similar timings to categorical styling
  because it also builds one explicit `label_id -> RGBA` entry per object.
- A 256-bin color table should be visually close to the current continuous
  colormap while avoiding repeated RGBA storage for hundreds of thousands of
  labels.
- This is not what Harpy does today. The current path maps each normalized
  continuous float value through the colormap and stores the resulting
  per-label RGBA.
- Napari does compact identical RGBA values after receiving a direct labels
  colormap. In `DirectLabelColormap._label_mapping_and_color_dict`, labels that
  share the exact same RGBA tuple are assigned the same compact texture id, and
  the vispy labels layer uploads that compact texture-id-to-RGBA table.
- However, Harpy currently still builds the full `label_id -> RGBA` dictionary
  first, and napari then scans that dictionary to rediscover the compact
  structure.
- Napari does not choose or expose a Harpy-owned 256-bin continuous
  source-value quantization before the direct colormap is built. Slice 7 is
  about moving that compact bin mapping into Harpy's own construction path.
- Therefore, 256-bin continuous coloring is an intentional approximation of the
  current per-value RGBA behavior, even if the current matplotlib colormap
  already emits repeated exact RGBA values for many datasets. It should be
  specified and tested separately from categorical coloring, with an explicit
  visual/parity tolerance.

Implementation direction:

1. Define the continuous binning contract:
   - use 256 bins;
   - normalize finite values over the same range currently used by
     `continuous_rgba_for_values(...)`;
   - map missing/non-finite values to the current missing/default color;
   - clamp values outside the chosen range.
2. Build a pure helper that produces compact state:
   - positive label ids;
   - `label_id -> color_bin`;
   - `color_bin -> RGBA`;
   - missing/default color;
   - transparent background label `0`.
3. Compare the proposed bin count with the current number of unique RGBA values
   produced by `continuous_rgba_for_values(...)` on benchmark columns. For
   example, the Slice 5 `total_counts` column produced `182` unique RGBA colors
   for `406,611` rows with the current path, because matplotlib/napari already
   yield repeated exact RGBA values.
4. Reuse the Harpy-owned compact colormap implementation shape from Slice 6
   where possible.
5. Add visual/parity-tolerance tests against the current continuous direct RGBA
   helper:
   - representative values at min, midpoint, max;
   - repeated values;
   - missing values;
   - out-of-range/clamped values if the helper exposes an explicit range;
   - large label ids.
6. Benchmark on the Slice 5 `continuous_total_counts` case.

Acceptance criteria:

- continuous labels coloring remains visually equivalent within an explicit
  accepted 256-bin tolerance; it is not expected to be bit-for-bit identical to
  the current per-value RGBA path;
- missing values keep the current missing/default color;
- background label `0` remains transparent;
- hover/status feature lookup remains label-id based;
- benchmark improves the `continuous_total_counts` direct colormap path or
  materially reduces memory use without making assignment slower;
- no duplicate long-term continuous styled-labels colormap implementations.

## Non-Goals

- Do not rewrite the labels image to category codes.
- Do not lose per-cell label identity.
- Do not change hover/status feature semantics.
- Do not depend on direct `DirectLabelColormap.model_construct(...)`.
- Do not patch installed napari source code.
- Do not introduce a private napari vispy visual for this roadmap.

## Longer-Term Upstream Direction

The real long-term solution would be native napari support for compact labels
colormaps. For categorical values:

```text
label_id -> category_code
category_code -> RGBA
```

For continuous values, the analogous model is a quantized color table:

```text
label_id -> color_bin
color_bin -> RGBA
```

Both avoid storing repeated RGBA arrays for hundreds of thousands of labels.

For Slices 1-5, Harpy should keep the public `DirectLabelColormap` contract and
remove the avoidable pydantic/color-transformation overhead that dominates the
current direct `.obs` path. Slice 6 can then use benchmarked prototype evidence
to decide whether a compact Harpy categorical colormap is worth carrying
locally or whether the right product move is an upstream napari API. Slice 7
handles continuous `.obs` coloring separately through explicit 256-bin
quantization.

## Suggested Verification

Focused tests:

```text
.venv/bin/pytest tests/test_viewer_adapter.py tests/test_viewer_styling.py
```

Lint:

```text
.venv/bin/ruff check src/napari_harpy/viewer src/napari_harpy/widgets/object_classification tests/test_viewer_adapter.py tests/test_viewer_styling.py
```

Manual benchmark target:

- compare `apply_table_color_source_to_labels_layer(...)` before/after for
  `table_global_ROI1.obs["leiden"]`;
- confirm the time previously attributed to `DirectLabelColormap` construction
  drops substantially;
- confirm visual categorical colors match the previous implementation.
