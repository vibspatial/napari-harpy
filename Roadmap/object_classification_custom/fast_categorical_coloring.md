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

Status: proposed.

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

### Slice 6: Investigate Compact Categorical Labels Colormap

Investigate whether Harpy can support the more compact categorical model:

```text
label_id -> category_code
category_code -> RGBA
```

This is a follow-up investigation/prototype after the safer fast
`DirectLabelColormap` helper. It touches napari internals, so the product bar is
higher: no production path should depend on a private napari contract unless it
is version-gated, tested, benchmarked, and has a reliable fallback to the Slice
1 helper.

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

Prototype route:

1. Build a Harpy-owned subclass of `DirectLabelColormap`, for example
   `CompactCategoricalLabelColormap`.
2. Store compact categorical state directly:
   - positive label ids;
   - a label-id-to-category-code mapping;
   - category-code-to-RGBA colors;
   - default color for unmapped labels;
   - transparent background label `0`.
3. Override only the narrow direct-colormap methods/properties that napari's
   renderer consumes:
   - `_values_mapping_to_minimum_values_set(...)`;
   - `_label_mapping_and_color_dict`;
   - `_num_unique_colors`;
   - `map(...)`;
   - `_clear_cache(...)`.
4. Preserve enough `color_dict` behavior for `Labels._set_colormap(...)` and
   `Labels.get_color(...)` to keep working. In particular, the colormap must
   not look like a default-only direct colormap, otherwise napari switches the
   layer back to auto color mode.
5. Test both paths in napari's renderer:
   - large integer labels, where the direct compact texture path is used;
   - small integer labels, where napari may use the small-dtype lookup path
     even for direct colormaps.

Important constraints:

- Do not rewrite the labels image to category codes. The image must keep
  per-cell label identity.
- Do not create a second labels layer just to display categorical colors.
- Do not patch installed napari source code.
- Do not monkeypatch napari globals at runtime.
- Keep the Slice 1 fast `DirectLabelColormap` helper as the fallback for
  unsupported napari versions, unsupported dtypes, or failed parity tests.

Prototype acceptance criteria:

- visual parity with the Slice 1 helper for categorical `.obs` colors;
- `map(...)` parity for representative labels, missing labels, background, and
  selection mode;
- hover/status feature lookup remains label-id based;
- no loss of per-cell label identity;
- faster or materially lower-memory construction than the Slice 1 helper on the
  `table_global_ROI1.obs["leiden"]` benchmark;
- no regression for 2D, 3D, and multiscale Labels layers that Harpy supports;
- explicit napari-version guard with a loud fallback when internals change.

If the prototype is successful, decide whether to:

- productionize the Harpy subclass behind a small adapter and fallback; or
- open an upstream napari proposal/PR for a public compact categorical labels
  colormap API.

## Non-Goals

- Do not rewrite the labels image to category codes.
- Do not lose per-cell label identity.
- Do not change hover/status feature semantics.
- Do not depend on direct `DirectLabelColormap.model_construct(...)`.
- Do not patch installed napari source code.
- Do not introduce a private napari vispy visual for Slices 1-5.

## Longer-Term Upstream Direction

The real long-term solution would be native napari support for a compact
categorical labels colormap:

```text
label_id -> category_code
category_code -> RGBA
```

That would avoid storing repeated RGBA arrays for hundreds of thousands of
labels when only a few categories are present.

For Slices 1-5, Harpy should keep the public `DirectLabelColormap` contract and
remove the avoidable pydantic/color-transformation overhead that dominates the
current categorical `.obs` path. Slice 6 can then use benchmarked prototype
evidence to decide whether a compact Harpy adapter is worth carrying locally or
whether the right product move is an upstream napari API.

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
