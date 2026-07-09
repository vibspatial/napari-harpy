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

Status: implemented.

Audit target:

- napari version: `0.7.1`;
- inspected paths:
  - `napari.utils.colormaps.colormap.DirectLabelColormap`;
  - `napari.layers.labels.labels.Labels`;
  - `napari._vispy.layers.labels.VispyLabelsLayer`.

Layer-side contract:

- `Labels.colormap = ...` calls `Labels._set_colormap(...)`, which first calls
  napari's `_normalize_label_colormap(...)`.
- `_normalize_label_colormap(...)` accepts an existing
  `DirectLabelColormap`/`CyclicLabelColormap` instance, a sequence/array for
  cyclic colors, or a mapping that it converts to `DirectLabelColormap`.
- A Harpy compact object therefore needs to be a `DirectLabelColormap`
  subclass, not an unrelated object.
- `Labels._set_colormap(...)` stores direct colormaps on
  `layer._direct_colormap`, but only uses direct color mode if
  `layer._is_default_colors(colormap.color_dict)` is false.
- A compact subclass must therefore expose a `color_dict` that does not look
  like the default-only `{None, background}` mapping, otherwise napari will
  switch back to auto color mode and ignore the direct colors.
- `Labels._set_colormap(...)` also invalidates cached labels, sets
  `layer._selected_color = layer.get_color(layer.selected_label)`, emits
  `events.colormap` and `events.selected_label`, and calls
  `refresh(extent=False)`.
- `Labels.get_color(label)` calls `colormap.map(...)`, except for background
  and selected-label edge cases.
- `Labels._slice_dtype(...)`, `_raw_to_displayed(...)`, and partial label
  updates call `colormap._data_to_texture(...)`.

`DirectLabelColormap` contract:

- `color_dict` is the public-looking mapping napari uses for direct-mode
  detection, scalar `map(...)`, default color lookup, and small-label fallback
  internals.
- `default_color` reads `color_dict[None]`, falling back to transparent.
- `_num_unique_colors` is a cached property currently computed by scanning
  unique RGBA tuples in `color_dict`.
- `_label_mapping_and_color_dict` is a cached property that groups labels with
  identical RGBA tuples into compact texture ids:

  ```text
  raw label id -> compact texture id
  compact texture id -> RGBA
  ```

- `_values_mapping_to_minimum_values_set(apply_selection=True)` returns either
  the cached compact mapping above, or a special selected-label-only mapping
  when `use_selection` is true.
- `_data_to_texture(...)` delegates to napari's direct-label cast helper.
  Higher-bit integer labels are mapped through `_label_mapping_and_color_dict`
  / `_get_typed_dict_mapping(...)`; small integer labels are returned as raw
  unsigned texture values.
- `map(...)` supports scalar labels and integer arrays. Array mapping uses
  `_get_mapping_from_cache(...)` for small integer dtypes when possible, and
  the accelerated direct mapping otherwise.
- `_clear_cache(...)` clears the base mapping caches and the cached direct
  properties `_num_unique_colors`, `_label_mapping_and_color_dict`, and
  `_array_map`.
- `use_selection`, `selection`, and `background_value` are model fields that
  napari mutates directly.

Vispy/rendering contract:

- `VispyLabelsLayer._on_colormap_change(...)` receives napari colormap events
  and branches on `isinstance(colormap, CyclicLabelColormap)`.
- For direct colormaps with high-bit raw labels (`raw_dtype.itemsize > 2`),
  vispy calls `colormap._values_mapping_to_minimum_values_set()[1]`, uploads
  that compact texture-id-to-RGBA table, and chooses a texture dtype from
  `layer._direct_colormap._num_unique_colors + 2`.
- For auto mode, or for direct colormaps with small raw labels
  (`raw_dtype.itemsize <= 2`), vispy uses `_select_colormap_texture(...)`,
  which calls `colormap._get_mapping_from_cache(...)` and uploads the resulting
  lookup texture.
- The small-label path means a compact subclass cannot only implement the
  high-bit direct path. It must also keep `_get_mapping_from_cache(...)`,
  `_map_without_cache(...)`, `_data_to_texture(...)`, and `map(...)` working
  for `uint8`/`uint16` label data.
- The current benchmark dataset likely exercises the high-bit path because
  object ids exceed 16-bit range, but Harpy tests should cover both high-bit
  and small integer labels.

Selection and multiscale contract:

- `Labels.selected_label` assigns `colormap.selection = selected_label`.
- `Labels.show_selected_label` assigns `colormap.use_selection = show_selected`
  and `colormap.selection = selected_label`.
- The vispy layer listens to `selected_label` and `show_selected_label` events
  and rebuilds the colormap texture/shader state.
- A compact subclass must preserve selected-label behavior for both
  `map(...)` and `_values_mapping_to_minimum_values_set(...)`.
- Multiscale labels still route through the same layer colormap object and
  current slice image. There is no separate multiscale colormap API, but tests
  should include a multiscale Labels layer if feasible because slicing changes
  the raw/view dtype path that vispy sees.

Probe result:

```text
color_dict:
  1001 -> red
  1002 -> blue
  1003 -> red

napari compact mapping:
  None -> 0
  0 -> 1
  1001 -> 2
  1002 -> 3
  1003 -> 2

high-bit texture values for [0, 1001, 1002, 1003, 9999]:
  [1, 2, 3, 2, 0]
```

This confirms that napari already compacts repeated RGBA colors after it has
received the full `label_id -> RGBA` dictionary. The remaining Harpy
opportunity is to build and expose that compact mapping directly, avoiding the
large repeated RGBA mapping and napari's grouping scan.

Harpy-owned test contract if subclassing:

- assignment to a real `Labels` layer keeps `layer.color_mode` in direct mode;
- `Labels.get_color(...)` matches the current Slice 1 helper for background,
  mapped labels, repeated categories, and missing labels;
- `map(...)` matches the current helper for scalar labels and integer arrays;
- `_data_to_texture(...)` returns expected texture ids for high-bit labels;
- small `uint8`/`uint16` labels render through the lookup-texture path;
- selected-label rendering matches napari's current direct colormap behavior;
- `_clear_cache(...)` invalidates all compact mappings owned by the subclass;
- multiscale labels keep the same behavior where headless tests can exercise
  it.

Recommendation:

- Slice 6.2 is still worth pursuing.
- Start with a pure Harpy compact mapping builder that produces explicit
  `label_id -> texture_code` and `texture_code -> RGBA` state.
- Slice 6.3 should subclass `DirectLabelColormap` and override the narrow
  compact-mapping hooks identified above, while still exposing enough
  `color_dict` for napari's direct-mode detection and scalar/default behavior.
- Harpy should explicitly set napari's labels colormap backend to
  `ColormapBackend.numba` for this path. Do not rely on napari's
  `Fastest available` setting, because that can choose different internal
  mapping contracts depending on optional installed packages.
- Because this relies on private napari internals, Harpy should own the
  behavior through focused tests rather than adding runtime fallback/version
  guards.

#### Slice 6.2: Compact Mapping Builder

Status: implemented.

Build a Harpy-owned pure data helper for categorical labels, without touching
napari internals yet.

Backend policy for the compact-labels work:

- target napari's `ColormapBackend.numba` direct-label path explicitly;
- do not let `Fastest available` decide between numba, PartSegCore, and pure
  Python for Harpy's compact labels coloring;
- the implementation should set the colormap backend to
  `ColormapBackend.numba` before constructing/assigning Harpy direct labels
  colormaps;
- if napari no longer exposes the numba colormap backend, fail loudly rather
  than silently falling back to a different rendering contract.

This matters because the audited high-bit labels path with numba consumes
`DirectLabelColormap._get_typed_dict_mapping(data.dtype)`, i.e. a per-dtype
numba typed mapping from raw label ids to texture codes. The pure helper in
Slice 6.2 should not build that typed dict yet, but it must produce compact
state that Slice 6.3 can turn into that typed mapping without rebuilding a full
`label_id -> RGBA` dictionary.

Small label ids use a different napari path. For `uint8`/`uint16` labels,
napari can keep raw label values as texture values and upload a dense lookup
texture. Harpy should preserve that path, but derive the dense lookup from
compact state:

```text
raw small label value -> RGBA
```

This bounded lookup is acceptable because it is at most `256` entries for
`uint8` or `65,536` entries for `uint16`, and should be built/cached per dtype
on the compact colormap instance. It should not require a full Python
`label_id -> RGBA` dictionary.

The path split should follow napari's dtype branch:

```text
raw labels dtype itemsize <= 2  -> small-label dense lookup path
raw labels dtype itemsize > 2   -> high-bit numba typed remapping path
```

Signed small integer labels should first follow napari's existing unsigned-view
conversion semantics (`int8 -> uint8`, `int16 -> uint16`) before using the
dense lookup path.

The helper should turn Harpy's existing table-aligned categorical colors into
napari-shaped compact state:

```text
positive label ids
label_id -> texture_code
texture_code -> RGBA
texture code for unmapped labels / default color
texture code for background label 0
optional texture code for missing/unknown categorical values
```

Use `texture_code` terminology here rather than generic `category_code`
terminology. Slice 6.1 showed that napari's direct labels rendering already
uses a compact texture-id-to-RGBA model internally. Harpy's builder should
prepare exactly the state the subclass will need later.

Implementation details:

- reserve texture code `0` for unmapped labels and the `None` default color;
- assign an explicit texture code for background label `0`, even if it is
  transparent like the default color, to stay close to napari's current direct
  colormap semantics;
- assign category texture codes after the default/background codes;
- treat missing or palette-unknown categorical table values as mapped rows with
  their own missing-category color, not as unmapped labels. In the current
  Harpy semantics, a known table row with a missing/unknown categorical value
  receives `MISSING_CATEGORICAL_COLOR`, while a label id with no table row
  falls through to the transparent `None` default;
- keep texture-code keys sequential from `0`, because napari's
  `build_textures_from_dict(...)` assumes sequential color-table keys;
- store repeated category RGBA values once in `texture_code -> RGBA`;
- keep `label_id -> texture_code` as the per-object mapping;
- keep the `label_id` and `texture_code` arrays suitable for later construction
  of a per-dtype numba typed dict in the compact `DirectLabelColormap`
  subclass;
- keep enough compact state to later build bounded dense small-label lookup
  textures for `uint8`/`uint16` labels;
- derive the small-label versus high-bit path from the actual labels dtype
  itemsize, matching napari's `raw_dtype.itemsize <= 2` branch;
- never rewrite the labels image to texture codes.

This helper should be independent of napari `DirectLabelColormap` subclassing.
It should be easy to test with ordinary NumPy arrays and dictionaries.

Performance contract:

- the helper must never build a full `label_id -> RGBA` dictionary;
- `label_id -> texture_code` is the logical mapping contract, not a requirement
  to store the mapping as a large Python `dict`;
- prefer compact array-backed state, for example positive `label_ids`,
  matching `texture_codes`, and a small `texture_code -> RGBA` table, unless
  benchmarking proves a different representation is better;
- repeated category colors must stay represented as compact texture codes;
- any conversion from categorical table values to compact state should be
  linear in the number of labels plus the number of categories, without a
  second per-label RGBA materialization step;
- the pure helper should not construct napari's numba typed dict itself; that
  belongs at the compact colormap/backend boundary, where it can be built once
  per dtype and reused;
- the pure helper should also not construct dense small-label lookup arrays
  itself; those belong at the compact colormap/backend boundary and should be
  built from compact state only for small dtypes that need them.

Acceptance criteria:

- repeated RGBA colors are stored once per texture code;
- background label `0` remains transparent;
- unmapped labels resolve to the same transparent `None` default color as the
  current direct RGBA helper;
- known table rows with missing or palette-unknown categorical values resolve
  to the missing-category color, not to the transparent unmapped-label color;
- texture-code keys are sequential from `0`;
- helper output can be used to build a numba typed
  `raw label_id -> texture_code` mapping without materializing
  `label_id -> RGBA`;
- helper output can be used to build bounded dense `uint8`/`uint16`
  `raw label value -> RGBA` lookup textures without materializing a full
  Python `label_id -> RGBA` dictionary;
- tests cover the dtype split: small integer dtypes use the dense lookup path,
  while `int32`/`uint32`/`int64`/`uint64` use the high-bit numba remapping path;
- helper output preserves per-cell label ids and never rewrites the labels
  image;
- tests cover representative label ids, missing labels, repeated categories,
  non-contiguous source categories, and sequential texture-code output.

#### Slice 6.3: Prototype Compact DirectLabelColormap

Status: implemented.

Create an isolated prototype, for example
`CompactCategoricalLabelColormap`, that can be assigned where napari expects a
`DirectLabelColormap`.

Scope:

- subclass `DirectLabelColormap` so napari's `_normalize_label_colormap(...)`
  accepts it as a labels colormap instance;
- accept one `CompactCategoricalLabelsMapping` from Slice 6.2;
- stay isolated from production styled-labels routing in this slice;
- include focused prototype tests, but do not replace the current categorical
  direct-RGBA production path yet.

Constructor / state:

- use the compact mapping as the source of truth:

  ```text
  label_ids
  texture_codes
  texture_rgba
  default_texture_code
  background_texture_code
  missing_texture_code
  ```

- initialize the base `DirectLabelColormap` with a small direct color mapping
  that is sufficient for napari/pydantic event setup and direct-mode detection;
- expose enough `color_dict` behavior for napari's direct-mode detection,
  default color lookup, and scalar compatibility;
- do not expose or lazily build a full `label_id -> RGBA` dictionary.

Backend policy:

- explicitly require/set napari's labels colormap backend to
  `ColormapBackend.numba` for this prototype path;
- fail loudly if the numba colormap backend is unavailable;
- do not add fallback behavior for PartSegCore or pure-python backends in this
  slice.

Implementation details:

- make the compact state, not `color_dict`, the source of truth for full
  per-label coloring;
- never lazily materialize a full `label_id -> RGBA` dictionary from
  `color_dict`, `map(...)`, `_values_mapping_to_minimum_values_set(...)`,
  `_label_mapping_and_color_dict`, `_num_unique_colors`, `_data_to_texture(...)`,
  or cache rebuilding;
- `_values_mapping_to_minimum_values_set(...)` must not build a real
  `label_id -> texture_code` Python dictionary just because napari/vispy asks
  for the compact color table at index `[1]`. Instead, it should return:

  ```text
  (
      cheap label-id-to-texture-code Mapping view,
      small texture-code-to-RGBA dictionary,
  )
  ```

  The first item should be an array-backed `Mapping` view over
  `CompactCategoricalLabelsMapping.label_ids` and `texture_codes`, not an
  eager `dict`. Constructing or returning this mapping view must be `O(1)`.
  Iterating `.items()` may still be `O(n)`, but only callers that truly need
  all `label_id -> texture_code` pairs should pay that cost.
- the small texture-code-to-RGBA dictionary returned as
  `_values_mapping_to_minimum_values_set()[1]` should be derived from
  `texture_rgba` and should contain only sequential texture-code keys:

  ```text
  0 -> transparent default
  1 -> transparent background
  2 -> first category/missing color
  ...
  ```

  This is the object vispy needs for `build_textures_from_dict(...)`, so it
  should stay proportional to the number of unique colors, not the number of
  labels.
- the label-id-to-texture-code mapping view should support the scalar lookup
  behavior napari expects without materializing a large dict:

  ```python
  class _CompactLabelToTextureMapping(Mapping):
      def __getitem__(self, key):
          if key is None:
              return compact.default_texture_code
          if int(key) == compact.background_value:
              return compact.background_texture_code
          pos = np.searchsorted(compact.label_ids, int(key))
          if pos < len(compact.label_ids) and compact.label_ids[pos] == key:
              return int(compact.texture_codes[pos])
          raise KeyError(key)
  ```

  The real implementation can add `__iter__`, `__len__`, and inherited
  `.get(...)` support, but it should keep construction cheap.
- `_label_mapping_and_color_dict` should return the same cheap mapping view and
  small texture-code color dictionary, for compatibility with napari internals.
  It should not be a cached property that first builds a full Python dict.
- `_get_typed_dict_mapping(data_dtype)` is the only high-bit path that should
  build a large mapping. It should build napari's numba typed
  `raw label_id -> texture_code` dictionary directly from compact arrays,
  cache it per dtype on the colormap instance, and reuse it on later calls.
  This large representation is acceptable there because high-bit labels need
  it for pixel remapping.
- `_get_typed_dict_mapping(data_dtype)` must fill the numba typed dict through
  a module-level jitted helper, not through a Python loop. The expensive line
  in napari's current implementation is repeated Python-side typed-dict
  insertion:

  ```python
  typed_mapping[data_dtype.type(label_id)] = target_dtype.type(texture_code)
  ```

  The compact subclass should avoid that bottleneck with this implementation
  shape:

  ```python
  from numba import njit


  @njit(cache=True)
  def _fill_typed_label_texture_mapping(
      typed_mapping,
      label_ids,
      texture_codes,
      background_value,
      background_texture_code,
  ):
      typed_mapping[background_value] = background_texture_code
      for i in range(label_ids.shape[0]):
          typed_mapping[label_ids[i]] = texture_codes[i]
      return typed_mapping
  ```

- the intended `_get_typed_dict_mapping(data_dtype)` shape is then:

  ```python
  def _get_typed_dict_mapping(self, data_dtype):
      data_dtype = np.dtype(data_dtype)
      cache_key = f"_compact_{data_dtype.name}_typed_dict"
      if cache_key in self._cache_other:
          return self._cache_other[cache_key]

      from numba import typed, types
      from napari.utils.colormaps import _accelerated_cmap as _accel_cmap

      compact = self._compact_mapping
      target_dtype = _accel_cmap.minimum_dtype_for_labels(
          len(compact.texture_rgba)
      )

      iinfo = np.iinfo(data_dtype)
      representable = (
          (compact.label_ids >= iinfo.min)
          & (compact.label_ids <= iinfo.max)
      )
      label_ids = compact.label_ids[representable].astype(
          data_dtype,
          copy=False,
      )
      texture_codes = compact.texture_codes[representable].astype(
          target_dtype,
          copy=False,
      )

      typed_mapping = typed.Dict.empty(
          key_type=getattr(types, data_dtype.name),
          value_type=getattr(types, target_dtype.name),
      )
      typed_mapping = _fill_typed_label_texture_mapping(
          typed_mapping,
          label_ids,
          texture_codes,
          data_dtype.type(compact.background_value),
          target_dtype.type(compact.background_texture_code),
      )

      self._cache_other[cache_key] = typed_mapping
      return typed_mapping
  ```

  The typed mapping should not include the `None` default entry, because the
  numba direct-label path maps integer label image values only. Unmapped labels
  fall through to napari's unknown/default texture code because the output
  texture array is initialized with that default before the typed mapping is
  applied.
- `_num_unique_colors` should stay consistent with this dtype choice. Because
  napari's high-bit path asks for `minimum_dtype_for_labels(_num_unique_colors
  + 2)`, the compact subclass should report the number of non-default,
  non-background texture rows, for example `max(0, len(texture_rgba) - 2)`.
  Then napari's existing `+ 2` lands back on `len(texture_rgba)`, the number of
  texture codes the compact colormap can emit.
- keep scalar/default compatibility small and explicit; if a method needs
  per-label information, it should use `label_id -> texture_code` plus
  `texture_code -> RGBA` directly;
- avoid looking like a default-only direct colormap, because napari may switch
  the layer back to auto color mode in that case;
- implement/support the narrow methods/properties identified in Slice 6.1:
  - `_values_mapping_to_minimum_values_set(...)`;
  - `_label_mapping_and_color_dict`;
  - `_num_unique_colors`;
  - `_data_to_texture(...)`;
  - `map(...)`;
  - `_map_without_cache(...)` / `_get_mapping_from_cache(...)` behavior for
    small `uint8`/`uint16` labels;
  - `_get_typed_dict_mapping(...)` behavior for the high-bit numba path;
  - `_clear_cache(...)`;
- use this exact dtype split:

  ```text
  uint8 / uint16 labels
    -> bounded dense lookup, because napari's small-label path uses that

  int32 / uint32 / int64 / uint64 labels
    -> numba typed dict, filled with the jitted helper above
  ```

  Signed `int8`/`int16` labels should follow napari's existing unsigned-view
  conversion semantics before the small-label lookup path.
- for small `uint8`/`uint16` labels, build bounded dense lookup arrays from
  compact state and cache them per dtype, preserving napari's small-label path
  without constructing a full Python `label_id -> RGBA` dictionary;
- large derived representations should be computed at most once per relevant
  dtype/compact state and reused on repeated access. This is local per-instance
  reuse, not a separate invalidation system;
- preserve `use_selection`, `selection`, and `background_value` behavior that
  napari mutates directly;
- keep `_values_mapping_to_minimum_values_set(apply_selection=True)` compatible
  with napari's selected-label-only path;
- keep `_values_mapping_to_minimum_values_set()[1]` compatible with vispy's
  `build_textures_from_dict(...)`, including sequential texture-code keys;
- be designed as the single Harpy-owned categorical styled-labels colormap path
  if the prototype is accepted.

Prototype acceptance criteria:

- the class is a `DirectLabelColormap` subclass;
- assigning it to a real `Labels` layer keeps the layer in direct color mode;
- scalar `map(...)` returns the expected RGBA for mapped labels, background,
  unmapped labels, and missing-category labels;
- array `map(...)` matches current direct-RGBA behavior for representative
  small and high-bit integer arrays;
- `_data_to_texture(...)` returns expected raw values for small dtypes and
  compact texture codes for high-bit dtypes;
- high-bit mapping uses the numba typed-dict path and does not materialize
  `label_id -> RGBA`;
- `_values_mapping_to_minimum_values_set(...)` returns a cheap mapping view and
  small texture-code color dictionary; it does not build a large
  `label_id -> texture_code` dict as a side effect;
- the large high-bit `label_id -> texture_code` representation is built only
  by `_get_typed_dict_mapping(data_dtype)` and is cached per dtype;
- small-label lookup arrays are bounded by dtype size and derived from compact
  state;
- selected-label mode matches current napari direct-colormap behavior;
- `_clear_cache(...)` clears derived typed-dict / lookup-array state without
  mutating the compact source state.

This slice is prototype-level. Production styled labels coloring is not routed
through the compact colormap yet.

#### Slice 6.4: Shared Compact Categorical Colormap Helper

Status: implemented.

Add the public Harpy helper that turns table-aligned categorical values into a
ready `CompactCategoricalLabelColormap`, without integrating it into viewer
styling yet.

Implementation shape:

- add a small helper in `src/napari_harpy/viewer/labels_colormap.py`, for
  example:

  ```python
  def compact_categorical_label_colormap_from_values(
      values: pd.Series,
      *,
      categories: Sequence[object],
      palette: Sequence[Any],
      missing_color: Any = MISSING_CATEGORICAL_COLOR,
      background_value: int = 0,
  ) -> CompactCategoricalLabelColormap:
      mapping = compact_categorical_labels_mapping_from_values(...)
      return CompactCategoricalLabelColormap(mapping)
  ```

- widen the existing compact builder palette typing from `Sequence[str]` to
  `Sequence[Any]`, because object-classification palettes already use numeric
  RGBA arrays in some paths;
- preserve all existing `compact_categorical_labels_mapping_from_values(...)`
  behavior, including missing/palette-unknown values and repeated RGBA
  compaction;
- keep `direct_label_colormap_from_rgba(...)` unchanged for direct RGBA use
  cases that are not categorical yet.

Tests:

- verify the helper returns `CompactCategoricalLabelColormap`;
- verify the helper preserves current compact mapping behavior for:
  - mapped categories;
  - missing/palette-unknown values;
  - repeated category colors;
  - numeric RGBA palette entries;
- verify it remains visually equivalent to the expanded direct RGBA baseline
  for representative labels.

Acceptance criteria:

- no production viewer styling code is routed through the helper in this slice;
- no new `label_id -> RGBA` categorical dictionary is built by the helper;
- existing compact-colormap tests still pass.

#### Slice 6.5: Styled Labels Integration

Status: implemented; full-data Xenium benchmark gate passed.

Make `CompactCategoricalLabelColormap` the categorical styled-labels path in
`src/napari_harpy/viewer/labels_styling.py`. We are working on a feature
branch, so do not keep the compact colormap as a long-lived optional
alternative next to the old categorical `label_id -> RGBA` route.

Scope:

- add an internal return type alias in `labels_styling.py`:

  ```python
  LabelsColormap = DirectLabelColormap | CompactCategoricalLabelColormap
  ```

  This is the shape returned by non-instance styling builders. It should refer
  to ready napari colormap objects, not intermediate color dictionaries.
- change `_build_obs_column_colormap(...)` so categorical-like branches return
  a concrete labels colormap object instead of a full categorical
  `label_id -> RGBA` dictionary;
- categorical-like branches include:
  - pandas categorical dtype;
  - bool dtype;
  - exact binary `0/1` numeric categories;
  - string-like columns coerced to categorical values;
- replace `_build_categorical_color_dict(...)` usage with
  `compact_categorical_label_colormap_from_values(...)`;
- remove `_build_categorical_color_dict(...)` if no categorical caller remains;
- continuous branches should build a ready direct colormap with
  `direct_label_colormap_from_rgba(...)` after creating the direct RGBA
  dictionary;
- `_build_x_var_colormap(...)` should also return a ready direct colormap until
  Slice 7;
- keep instance-id coloring on napari's procedural `label_colormap(...)`;
- update `_apply_labels_colormap(...)` so it only assigns a ready colormap:

  ```python
  def _apply_labels_colormap(layer: Labels, colormap: LabelsColormap) -> None:
      layer.colormap = colormap
  ```

  The helper should not build `DirectLabelColormap` itself anymore. Categorical
  branches build `CompactCategoricalLabelColormap`; continuous branches build
  `DirectLabelColormap`.

Tests:

- update styled-labels tests so categorical `.obs` styling assigns
  `CompactCategoricalLabelColormap`;
- verify categorical bool, binary numeric, pandas categorical, and string-like
  coercion paths still color representative labels like the expanded direct
  RGBA baseline;
- verify continuous `.obs` / `.X` styling still uses the direct RGBA helper
  until Slice 7;
- keep tests proving hover/status features remain label-id based and
  `layer.features` behavior is unchanged.

Benchmark acceptance:

- benchmark run date: 2026-07-08;
- store:
  `/Users/arne.defauw/VIB/DATA/test_data/sdata_xenium_full_data_core.zarr`;
- labels: `cell_labels_global_ROI1`;
- table: `table_global_ROI1`;
- coordinate system: `global_ROI1`;
- rows: `406,611`;
- repeats: `3`, reported as medians;
- the benchmark was run through Harpy's real
  `ViewerAdapter.ensure_styled_labels_loaded(...)` and
  `apply_table_color_source_to_labels_layer(...)` paths, using a lightweight
  non-GUI viewer object to avoid opening Qt/Vispy.

```text
instance_cell_ID:
  apply_table_color_source_to_labels_layer: 0.1792 s
  ensure_styled_labels_loaded cold:        0.1860 s
  ensure_styled_labels_loaded restyle:     0.1860 s
  colormap: CyclicLabelColormap

categorical_leiden:
  apply_table_color_source_to_labels_layer: 0.2259 s
  ensure_styled_labels_loaded cold:        0.2234 s
  ensure_styled_labels_loaded restyle:     0.2389 s
  colormap: CompactCategoricalLabelColormap
  color_dict entries: 3
  label ids: 406,611
  texture RGBA rows: 10
  unique texture codes used by labels: 8
  _values_mapping_to_minimum_values_set: ~0.000003 s

continuous_total_counts:
  apply_table_color_source_to_labels_layer: 1.2426 s
  ensure_styled_labels_loaded cold:        1.2387 s
  ensure_styled_labels_loaded restyle:     1.2695 s
  colormap: DirectLabelColormap
  color_dict entries: 406,613
```

Findings:

- The Slice 6.5 styled categorical path no longer spends roughly one second in
  direct-colormap grouping / typed-dict construction.
- `categorical_leiden` is now close to instance coloring for this benchmark
  (`~0.23 s` versus `~0.18 s`) and much faster than the earlier
  post-Slice-1 direct-colormap baseline (`~1.15 s`).
- The categorical colormap keeps only a tiny `color_dict` and uses compact
  `label_id -> texture_code` / `texture_code -> RGBA` state instead of a full
  `label_id -> RGBA` dictionary.
- Continuous `.obs` coloring remains slow and still builds a full
  `label_id -> RGBA` dictionary. That is expected before Slice 7.

#### Slice 6.6a: Compact Default-Color Support

Status: implemented.

Prepare the compact categorical helper for object-classification semantics
without touching the object-classification widget/controller yet.

Current compact styled-labels behavior:

```text
unmapped label id -> transparent default
background label 0 -> transparent background
```

Current object-classification categorical behavior:

```text
unmapped label id -> UNLABELED_COLOR
background label 0 -> transparent background
```

Implementation direction:

1. Extend `compact_categorical_labels_mapping_from_values(...)` and
   `compact_categorical_label_colormap_from_values(...)` with an optional
   `default_color` argument.
2. Keep the default `default_color` transparent, so Slice 6.5 styled-labels
   behavior remains unchanged.
3. Store `default_color` as texture code `0`, preserving the existing compact
   convention that code `0` is the `None` / unmapped-label default.
4. Keep `background_value=0` mapped to transparent background texture code `1`.
5. Do not change object-classification coloring in this slice.

Tests:

- styled-labels categorical coloring still uses transparent default/unmapped
  labels when no `default_color` override is passed;
- passing `default_color=UNLABELED_COLOR` produces a compact colormap whose
  unmapped labels map to the unlabeled gray color;
- background label `0` remains transparent even when `default_color` is
  non-transparent;
- existing compact colormap tests still pass.

#### Slice 6.6b: Object-Classification Full Categorical Repaint

Status: implemented.

Use `CompactCategoricalLabelColormap` for full object-classification
categorical labels repainting, while keeping the row-scoped sparse update as a
separate Slice 6.7.

Scope:

- in `ViewerStylingController.refresh_layer_colors(...)`, route
  `COLOR_BY_USER_CLASS` and `COLOR_BY_PRED_CLASS` through
  `CompactCategoricalLabelColormap`;
- build a class-value series indexed by instance id and pass sorted class ids
  plus the matching RGBA palette to the compact helper;
- pass `default_color=UNLABELED_COLOR` for object-classification categorical
  coloring, matching the current direct-colormap behavior where unmapped labels
  fall through to the unlabeled gray color;
- keep background label `0` transparent via `background_value=0`;
- preserve current missing/unlabeled behavior:
  - `pred_class` keeps explicit unlabeled/missing class coloring according to
    the existing class-color lookup;
  - `user_class` keeps unlabeled rows non-explicit per-label entries and lets
    them fall through to the unlabeled/default color, not to transparent;
- keep `COLOR_BY_PRED_CONFIDENCE` on the direct RGBA helper until Slice 7;
- do not keep a second full categorical `label_id -> RGBA` implementation for
  full `user_class` / `pred_class` recoloring.

Row-scoped annotation safety guard:

- `refresh_user_class_colormap_and_feature(...)` currently updates
  `DirectLabelColormap.color_dict` sparsely. That is not compatible with
  compact colormaps, where `color_dict` is intentionally tiny and not the
  source of truth.
- Add only the minimal guard needed for correctness in this slice:
  if the current labels colormap is `CompactCategoricalLabelColormap`, do not
  mutate `color_dict`; return `False` so the existing caller can use the
  correctness fallback.
- This fallback is temporary and should not be treated as the final design.
  Slice 6.7 replaces it with the real sparse compact update.
- The feature-only path for prediction color modes can remain unchanged,
  because it does not repaint label colors.

Tests:

- object-classification `user_class` and `pred_class` full refresh use
  `CompactCategoricalLabelColormap`;
- object-classification categorical coloring uses `UNLABELED_COLOR` as the
  compact default/unmapped color while keeping background label `0`
  transparent;
- `pred_confidence` remains visually equivalent on the direct RGBA path;
- row-scoped user-class updates do not mutate compact `color_dict` and return
  `False` when compact user-class coloring is active;
- hover/status features remain label-id based and `layer.features` behavior is
  unchanged.

Benchmark acceptance:

- run a representative object-classification categorical labels-coloring
  benchmark after full repaint integration;
- the `user_class` / `pred_class` full refresh path should no longer build a
  full `label_id -> RGBA` dictionary;
- confirm `_values_mapping_to_minimum_values_set(...)` stays cheap and no
  full `label_id -> RGBA` dictionary is built for categorical
  object-classification labels coloring;
- keep benchmark output in this roadmap before moving to Slice 7.

Benchmark check after implementation:

Synthetic object-classification repaint benchmark with 406,611 instance rows,
8 integer classes, and precomputed `feature_rows`:

```text
user_class: median=0.1492 s, min=0.1478 s, max=0.1539 s
pred_class: median=0.1513 s, min=0.1511 s, max=0.1545 s
```

Both paths produced `CompactCategoricalLabelColormap` with tiny bootstrap
`color_dict` length `<= 3`.

Follow-up cleanup in the same slice generalized the valid-categorical palette
lookup from only `user_class` to any valid class column, including
`pred_class`. With a bound table containing categorical `user_class` and
`pred_class` columns plus matching stored palettes, the same 406,611-row
synthetic check produced:

```text
user_class lookup median: 0.0623 s; full repaint median: 0.0910 s
pred_class lookup median: 0.0631 s; full repaint median: 0.0922 s
```

#### Slice 6.7: Compact Sparse User-Class Annotation Update

Status: proposed; phased implementation.

Replace the temporary row-scoped user-class fallback from Slice 6.6b with a
real compact sparse update.

Current direct-colormap sparse update:

```text
label_id -> RGBA
```

For `CompactCategoricalLabelColormap`, the source of truth is instead:

```text
label_id -> texture_code
texture_code -> RGBA
```

That means a single user-class annotation should update one label's texture code
without rebuilding the full colormap. We should implement this in phases:

1. First implement sparse compact state mutation plus an explicit
   `layer.refresh(extent=False)`. This avoids direct `layer._slice` mutation
   while proving the compact update semantics are correct.
2. Notify napari/vispy when a sparse update appends a new
   `texture_code -> RGBA` row, so the GPU lookup texture is rebuilt for
   brand-new classes.
3. Fix the registered-feature-matrix responsiveness regression so row-scoped
   visual annotation updates are not delayed by classifier preparation/status
   bookkeeping.
4. Finally remove the legacy direct-colormap sparse fallback and fail loudly if
   the compact sparse contract is broken.
5. Keep the explicit `layer.refresh(extent=False)` in the sparse happy path for
   now. Local benchmarks and manual QA show that it is responsive enough, and it
   avoids mutating private napari slice internals.

Shared compact-state contract:

1. Add enough class-to-texture-code state when building the object-classification
   user-class colormap so row-scoped updates can resolve class ids without
   rebuilding the full table-aligned mapping:

   ```text
   class_id -> texture_code
   ```

   This lookup should live on the compact colormap or compact mapping used for
   object-classification user-class coloring. It is separate from
   `label_id -> texture_code`: many labels can share one class texture code.
2. When a user annotation introduces a brand-new class,
   `set_user_class_for_rows(...)` has already synced the table palette, but
   the currently assigned compact colormap was built before that class existed.
   The sparse update must therefore be able to append a new
   `texture_code -> RGBA` row and record the new `class_id -> texture_code`
   entry before updating any label ids.
   - If the class was already present in the table categories/palette when the
     compact colormap was built, the colormap can already remember its
     `class_id -> texture_code` entry, even if no currently visible label uses
     that class.
   - If the class truly did not exist when the colormap was built, the sparse
     update cannot know it in advance. The annotation flow is:

     ```text
     set_user_class_for_rows(...)
       -> table.obs["user_class"] categories now include the new class
       -> table.uns["user_class_colors"] now includes its color

     sparse viewer update
       -> read the newly synced class color from the table palette
       -> append one `texture_code -> RGBA` row
       -> record `class_id -> texture_code`
       -> update the edited `label_id -> texture_code` entry
     ```

     After that first sparse update, the compact colormap remembers the new
     class for later annotations. The important distinction is that the table
     palette is consulted only to create a missing class mapping, not to
     rediscover every existing class mapping from RGBA on each annotation.
3. Preserve the current compact user-class semantics for unlabeled class `0`.
   In full repaint, `user_class == 0` rows are excluded from
   `label_id -> texture_code` and fall through to the default/unlabeled color.
   Sparse updates should keep the same representation.
4. Implement the three user-class transition cases explicitly:

   ```text
   0 -> class_id
     The label was previously absent from `label_ids`; insert it with the
     class texture code.

   class_id A -> class_id B
     The label already exists; update only its texture code.

   class_id -> 0
     The label should return to default/unlabeled behavior; remove it from
     `label_ids` and `texture_codes`.
   ```

   For the removal case, prefer removing the label entry rather than keeping an
   explicit default texture code. Removal matches the full-repaint compact
   state and keeps the mapping smaller.
5. Maintain `compact_mapping.label_ids` in sorted order after insertion or
   removal, because scalar lookup and array mapping use `np.searchsorted(...)`.
6. Do not update an existing `texture_code -> RGBA` row for a class change.
   That would recolor every label in the class. The sparse annotation operation
   changes one label's class membership, so it must update that label's
   `label_id -> texture_code` entry.
7. Appending a new `texture_code -> RGBA` row is different from updating an
   existing texture code. Appending is required only when the user introduces a
   class that the current compact colormap did not know about yet.

##### Phase 6.7a: Sparse State Update With Layer Refresh

Status: implemented.

Goal:

- mutate the compact colormap state sparsely;
- keep the existing row-scoped `layer.features` update;
- call `layer.refresh(extent=False)` so napari recomputes the displayed
  texture-code image from the updated compact mapping;
- avoid full feature-row rebuild, full compact colormap rebuild, and
  `layer.colormap = ...`.

Implementation direction:

1. Add a Harpy-owned helper/API on the compact colormap or compact mapping that
   applies one user-class annotation:
   - resolve or create the target class texture code through
     `class_id -> texture_code`;
   - insert, update, or remove the edited `label_id -> texture_code` entry;
   - keep `label_ids` sorted;
   - invalidate derived compact caches, including high-bit typed dicts and
     small-label dense lookup arrays;
   - return the target texture code when the label should be explicit, and the
     default/unlabeled texture code when the label was removed.
2. Update `refresh_user_class_colormap_and_feature(...)` so that when the
   current colormap is `CompactCategoricalLabelColormap` and
   `COLOR_BY_USER_CLASS` is active:
   - apply the compact sparse state mutation;
   - update only the edited row in `layer.features`;
   - call `layer.refresh(extent=False)`;
   - return `True`.
3. If the compact state, palette, features, or layer refresh hook is
   inconsistent, return `False` so the caller can keep the existing correctness
   fallback.
4. Do not access `layer._slice`, `layer._slice.image.raw`, or
   `layer._slice.image.view` in this phase.
5. Do not emit `layer.events.labels_update(...)` in this phase. Let napari's
   normal refresh path recode the displayed labels texture.

Why this works:

- Updating the compact `label_id -> texture_code` mapping handles future
  slices and future redraws.
- `Labels._raw_to_displayed(...)` maps raw label ids through
  `self.colormap._data_to_texture(...)`.
- `layer.refresh(data_displayed=True, extent=False)` asks napari to recode the
  current displayed labels data from the updated compact colormap state.
- This is less surgical than patching the current slice, but it avoids private
  slice mutation while we prove the compact state transition logic.

Phase 6.7a tests:

- row-scoped user-class annotation with compact colormap updates exactly the
  edited label's texture code;
- annotating an unlabeled/default label as a nonzero class inserts that label
  into `label_ids` / `texture_codes`;
- changing one label from class A to class B does not change texture codes for
  other labels in class A or class B;
- changing one label to unlabeled class `0` removes that label from
  `label_ids` / `texture_codes` and restores default/unlabeled behavior;
- annotating a label with a newly introduced class appends a new texture RGBA
  row, records the corresponding `class_id -> texture_code`, and then updates
  only the edited label;
- `label_ids` remains sorted after insertions and removals;
- the compact colormap `color_dict` remains tiny and is not treated as the
  full `label_id -> RGBA` mapping;
- `layer.features` is still updated row-scoped;
- the labels layer is refreshed with `extent=False` in the compact sparse happy
  path;
- the happy path does not assign a new full colormap object;
- the happy path does not call `_get_region_feature_rows(...)` or rebuild all
  features;
- rare inconsistent compact state still returns `False` so the caller can keep
  the existing correctness fallback.

Phase 6.7a benchmark acceptance:

- benchmark one user-class annotation update while `COLOR_BY_USER_CLASS` is
  active on a large labels layer;
- confirm the happy path avoids full feature-row rebuild, full compact colormap
  rebuild, and `layer.colormap = ...`;
- confirm the remaining cost is dominated by napari's layer refresh rather than
  table/colormap reconstruction.

##### Phase 6.7b: Notify Colormap Texture Changes For New Classes

Status: implemented.

Goal:

- keep the Phase 6.7a sparse state update and explicit
  `layer.refresh(extent=False)`;
- fix brand-new user classes whose sparse update appends a new
  `texture_code -> RGBA` row;
- notify napari/vispy to rebuild the colormap lookup texture when that texture
  table grows.

Observed bug:

When coloring by `user_class`, annotating with a class that already existed in
the compact colormap works because the visible label can reuse an already
uploaded texture code. Annotating with a brand-new class, for example class
`21`, appends a new `texture_code -> RGBA` row. The Python compact colormap and
table palette are correct, but vispy may still hold the previous
texture-code-to-RGBA lookup texture. The object can therefore display with the
wrong color until a full colormap rebuild happens, for example after switching
to `pred_class` and back.

Implementation direction:

1. Change the compact sparse mutation API so it reports whether the texture
   color table changed. One possible shape:

   ```python
   texture_code, texture_table_changed = colormap.set_label_category(...)
   ```

   `texture_table_changed` should be `True` only when the update appended a new
   `texture_code -> RGBA` row. Reusing an existing class texture code or
   removing a label should not report a texture-table change.
2. In `_refresh_compact_user_class_colormap_and_feature(...)`, after a
   successful sparse state mutation:
   - if `texture_table_changed` is `True`, emit the labels-layer colormap event
     before repainting the layer;
   - keep the Phase 6.7a `layer.features` row-scoped update;
   - keep `layer.refresh(extent=False)` for now.
3. Prefer the public layer event if available:

   ```python
   layer.events.colormap()
   ```

   This is the event napari's vispy labels layer listens to when it rebuilds
   the `texture_code -> RGBA` lookup texture.
4. Do not assign a new colormap object and do not rebuild the full compact
   colormap in this phase.

Why this works:

- `layer.refresh(extent=False)` asks napari to recode the visible label image
  from raw label ids to texture codes.
- `layer.events.colormap()` asks vispy to rebuild/upload the lookup texture
  that maps those texture codes to RGBA.
- Existing-class sparse updates usually need only the first step. Brand-new
  classes need both because they introduce a new texture code and a new RGBA
  row.
- Keep the explicit layer refresh for now. It updates the visible texture-code
  image without mutating private napari slice internals. Brand-new classes
  still need the colormap event because that updates the separate
  `texture_code -> RGBA` lookup texture.

Phase 6.7b tests:

- annotating with an already-known class mutates compact state and refreshes the
  layer without emitting the colormap event;
- annotating with a brand-new class appends one texture RGBA row and emits the
  colormap event exactly once;
- the appended texture row matches the table-synced user-class palette color;
- the labels layer still receives the Phase 6.7a `refresh(extent=False)` call;
- switching color modes is no longer required for the newly annotated class to
  have a matching compact texture row and colormap event.

Acceptance:

- sparse annotation with a brand-new user class shows the correct class color
  immediately in manual napari QA;
- no full compact colormap rebuild is needed for this fix;
- existing-class sparse annotations do not pay the extra colormap-event cost.

##### Phase 6.7c: Registered Feature-Matrix Annotation Responsiveness

Status: implemented.

Observed behavior:

- annotating objects in the object-classification widget is visually snappy
  when the selected feature matrix is unregistered;
- the same row-scoped annotation can feel noticeably slower after the feature
  matrix is registered;
- the compact sparse labels recoloring path itself does not depend on feature
  metadata, so this difference points to classifier/status work around the
  visual update rather than to the colormap mutation.

Investigation findings:

- `_on_annotation_changed(...)` currently calls
  `self._classifier_controller.mark_dirty(...)` before
  `_refresh_after_user_class_annotation(change)`.
- `mark_dirty(...)` updates classifier status, which emits the controller
  state-changed callback. The widget then rebuilds classifier controls and
  calls `describe_current_preparation()`.
- `_on_annotation_changed(...)` later calls `_update_selection_status()`, which
  can call `describe_current_preparation()` again through
  `_update_classifier_controls()`.
- For an unregistered feature matrix, `_prepare_classifier_summary(...)`
  exits early with a metadata blocker after metadata inspection.
- For a registered-valid feature matrix, `_prepare_classifier_summary(...)`
  continues into feature-validity checks, scope resolution, and user-class
  summary calculation before returning.
- The largest measured registered-only cost in a 400k-row synthetic table was
  `_get_user_class_values(...)`, because it always converts the whole
  `user_class` column through:

  ```python
  obs[USER_CLASS_COLUMN].astype("string")
  pd.to_numeric(...)
  ```

  This is especially wasteful for the normal widget state where `user_class` is
  already categorical with integer categories.
- Synthetic timing from the investigation:

  ```text
  unregistered per-click controller/status sequence: ~0.042 s
  registered per-click controller/status sequence:   ~0.320 s
  registered preparation summary:                    ~0.160 s
  _get_user_class_values(...) on 400k categorical:   ~0.113 s
  ```

- A prototype integer/categorical fast path for `_get_user_class_values(...)`
  reduced the full-column read to under 1 ms and reduced one registered
  preparation summary from ~160 ms to ~48 ms on the same synthetic table.
- Auto-train can amplify the effect because registered matrices can proceed to
  `schedule_retrain(...)`, while unregistered matrices stop at the metadata
  blocker. Auto-train is not required to reproduce the basic lag, because the
  status/control preparation path already explains it.
- After reordering the visual refresh first and adding the
  `_get_user_class_values(...)` fast path, annotation is snappy again for both
  registered and unregistered feature matrices in manual QA.
- After the visual-refresh reorder and fast user-class conversion, before the
  deferral cleanup, duplicate preparation-summary work was still measurable:

  ```text
  auto-train off: 2 preparation summaries per annotation
  auto-train on:  4 preparation summaries per annotation
  ```

  With the fast path in place, a 400k-row synthetic registered summary measured
  about `0.047 s`. This duplicate work no longer blocked the immediate visual
  update, but it was still worth batching with the widget-level deferral
  implemented in this phase.

Implementation direction:

1. Reorder `_on_annotation_changed(...)` so the row-scoped visual update runs
   before classifier dirty/status bookkeeping:

   ```python
   self._refresh_after_user_class_annotation(change)
   self._classifier_controller.mark_dirty(reason="the annotations changed")
   ```

   Persistence dirty marking and the first-time `user_class` color-source event
   can stay before the visual update unless manual QA shows they also delay the
   visible edit.
2. Add a fast path to `_get_user_class_values(...)`:
   - if `user_class` is an integer dtype, return/convert it with NumPy without
     string conversion;
   - if `user_class` is categorical with integer categories, map
     `cat.codes -> cat.categories`, treating code `-1` as unlabeled class `0`;
   - keep the current `astype("string")` / `pd.to_numeric(...)` fallback for
     non-normalized legacy states.
3. Keep the classifier controller as the owner of training eligibility. This
   slice should not move metadata gates back into the widget.
4. Reduce duplicate preparation-summary work with widget-level deferral rather
   than a controller-level cache:
   - keep `_on_classifier_state_changed(...)` updating classifier feedback
     immediately from the controller's cached `status_message` / `status_kind`;
   - allow `_on_annotation_changed(...)` to temporarily defer the expensive
     `_update_classifier_controls()` call while it runs classifier dirtying and
     auto-train scheduling:

     ```python
     with self._defer_classifier_control_updates():
         self._classifier_controller.mark_dirty(reason="the annotations changed")
         if self._auto_train_enabled:
             self._classifier_controller.schedule_retrain()
     ```

     The helper should follow the existing widget guard style used elsewhere in
     napari-harpy: store the previous boolean guard state, set deferral to
     `True`, then restore the previous value in `finally`.

     ```python
     @contextmanager
     def _defer_classifier_control_updates(self) -> Iterator[None]:
         previous = self._is_deferring_classifier_control_updates
         self._is_deferring_classifier_control_updates = True
         try:
             yield
         finally:
             self._is_deferring_classifier_control_updates = previous
     ```

   - let the final `_update_selection_status()` rebuild classifier controls and
     preparation status once for the completed annotation flow;
   - do not cache `ClassifierPreparationSummary` in the controller yet, because
     invalidating it correctly across table edits, feature-matrix changes,
     scope changes, async training, and reloads is riskier than batching one UI
     refresh.

Expected call-count improvement for the deferral cleanup:

```text
auto-train off: 2 preparation summaries -> 1 preparation summary
auto-train on:  4 preparation summaries -> about 2 preparation summaries
```

Completed substeps:

- `_on_annotation_changed(...)` now refreshes the row-scoped visual state
  before marking the classifier dirty.
- `_get_user_class_values(...)` now has fast paths for integer and
  categorical-integer `user_class` columns, with the legacy string/object
  fallback retained.
- `_on_annotation_changed(...)` now defers classifier-control refreshes while
  classifier dirtying and auto-train scheduling emit status callbacks, then
  lets the final `_update_selection_status()` rebuild controls once for that
  annotation callback.

Tests:

- unit-test `_get_user_class_values(...)` for:
  - missing `user_class` column;
  - plain integer series;
  - categorical integer series;
  - categorical integer series with missing codes;
  - legacy string/object values that still require the fallback.
- widget/controller test that an annotation in `COLOR_BY_USER_CLASS` refreshes
  the row-scoped viewer update before classifier dirty/status callbacks block
  the UI path.
- regression test or benchmark-style test showing registered categorical
  `user_class` does not call the slow string-normalization path in the normal
  widget state.
- deferral cleanup tests:
  - `mark_dirty(...)` during annotation still updates classifier feedback;
  - one annotation with auto-train disabled calls
    `describe_current_preparation()` only once from the final
    `_update_selection_status()` path;
  - one annotation with auto-train enabled does not run the extra
    state-callback classifier-controls rebuilds while still scheduling
    training;
  - selection changes, training-scope changes, prediction-scope changes, manual
    train clicks, worker completion, and reload paths still update classifier
    controls immediately outside the deferred annotation block.

Acceptance:

- annotation while coloring by `user_class` remains visually immediate for both
  unregistered and registered feature matrices;
- registered feature matrices still correctly enable/disable classifier
  training based on the controller preparation summary;
- auto-train behavior is unchanged apart from the visual annotation update
  happening before scheduled training/status work;
- duplicate preparation-summary work is batched during annotation without
  hiding classifier feedback/status changes;
- no full labels colormap rebuild is introduced.

##### Phase 6.7d: Remove Legacy Direct-Colormap Fallback

Status: implemented.

Goal:

- make compact user-class sparse updates the only supported happy path for
  row-scoped user-class annotation while `COLOR_BY_USER_CLASS` is active;
- remove the legacy `label_id -> RGBA` sparse fallback from
  `refresh_user_class_colormap_and_feature(...)`;
- fail loudly if compact sparse state mutation unexpectedly cannot be applied.

Current transitional flow:

```python
if self._refresh_compact_user_class_colormap_and_feature(change, feature_rows):
    return True

color_dict = self._build_user_class_annotation_color_dict(change)
if color_dict is None:
    return False
```

After Phase 6.7d, the compact colormap path should no longer silently fall
through to `_build_user_class_annotation_color_dict(...)` for user-class
annotation. If the current layer is expected to be compact user-class colored,
then failure of `_refresh_compact_user_class_colormap_and_feature(...)` should
raise a clear error that identifies the broken contract.

Implementation direction:

1. Keep non-user-class color modes unchanged. For example, prediction color
   modes can still use `refresh_user_class_feature(...)` because the visible
   color source is not `user_class`.
2. In the `COLOR_BY_USER_CLASS` row-scoped annotation path, require the current
   labels colormap to be `CompactCategoricalLabelColormap`.
3. Replace the boolean fallback around
   `_refresh_compact_user_class_colormap_and_feature(...)` with a loud failure
   if the compact sparse update returns `False`.
4. Remove `_build_user_class_annotation_color_dict(...)` if it becomes unused,
   together with tests that only exist to cover the legacy direct-colormap
   sparse fallback.
5. Keep tests that verify the compact sparse happy path:
   - no full feature-row rebuild;
   - no new full colormap assignment;
   - layer features are updated row-scoped;
   - `layer.refresh(extent=False)` is still called in the compact sparse happy
     path.
6. Add a test that simulates a broken compact sparse contract and asserts that
   the code raises a clear exception instead of silently rebuilding a legacy
   direct colormap.

Acceptance:

- `refresh_user_class_colormap_and_feature(...)` has one clear compact
  user-class sparse path;
- `_build_user_class_annotation_color_dict(...)` is removed if unused;
- unexpected compact sparse failures are visible during development and QA;
- no behavior changes for annotation while coloring by `pred_class` or
  `pred_confidence`, where only `layer.features` should be updated.

### Slice 7: Compact Continuous Labels Colormap Via 256 Bins

Status: proposed.

After Slice 6 settles the Harpy-owned compact colormap shape for categorical
`.obs`, extend the same idea to continuous `.obs` values by quantizing colors
into a fixed 256-bin color table:

```text
label_id -> color_bin
color_bin -> RGBA
```

This should be a separate continuous compact builder, not a call to
`compact_categorical_labels_mapping_from_values(...)`. The categorical helper
maps discrete category values to texture codes; continuous coloring must map
numeric values through normalization and quantization before assigning bins.

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
   This helper should be continuous-specific, for example
   `compact_continuous_labels_mapping_from_values(...)`, even if it reuses the
   same lower-level compact state dataclass or compact colormap subclass shape
   introduced for categorical labels.
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

### Slice 8: Async Slicing Labels Colormap Synchronization

Status: proposed.

When `Interactive(..., async_slicing=True)` is used, coloring labels by a
categorical or continuous table column can briefly or persistently render with
incorrect colors. The observed screenshots show patterns consistent with a
labels texture-code image being interpreted by the wrong labels colormap table.
This is not specific to the compact categorical colormap work: the same class
of issue can happen with napari's normal `DirectLabelColormap` path.

Investigation notes:

- Harpy only toggles napari's global async slicing setting in
  `Interactive(...)` through `get_settings().experimental.async_`.
- Harpy builds styled labels layers the same way for async and sync:
  `_build_labels_layer(...)` creates a `napari.layers.Labels` layer, and
  `apply_table_color_source_to_labels_layer(...)` later assigns the direct
  labels colormap.
- Napari labels rendering is a two-stage process:
  - `Labels._raw_to_displayed(...)` converts raw label ids to small texture
    codes with `layer.colormap._data_to_texture(...)`;
  - `VispyLabelsLayer._on_colormap_change(...)` uploads the texture-code-to-RGBA
    table used by the shader.
- With async slicing enabled, `Layer.refresh(data_displayed=True)` does not
  immediately recode the displayed labels texture. It emits a reload event and
  the recode happens when the async slice response is applied.
- The colormap event still updates the vispy shader/color table immediately.
  That means the viewer can temporarily display old texture codes from the
  previous labels colormap against the new texture-code-to-RGBA table.
- Headless `ViewerModel` checks for
  `table_transcriptomics_preprocessed.obs["leiden"]` and
  `table_transcriptomics_preprocessed.obs["total_counts"]` on
  `nucleus_segmentation_mask` showed that the final layer-side
  `layer._slice.image.raw` and `layer._slice.image.view` are identical for
  async and sync once the async slice response is applied. That points away
  from table alignment, label-id conversion, and compact colormap construction,
  and toward the real Qt/vispy display window between colormap assignment and
  async slice completion.
- On large dask-backed labels, that window can be long enough to be visible in
  normal use. If the user navigates while async slicing is pending, multiple
  requests can also make the mismatch feel persistent.

Recommended fix:

- After Harpy assigns a direct labels colormap for table-driven labels coloring,
  force the current labels slice to be recoded synchronously before returning
  control to the user.
- Implement this in the viewer-facing code path, not inside the pure colormap
  builders:
  - `ViewerAdapter.ensure_styled_labels_loaded(...)` has access to both the
    viewer and the styled labels layer, so it can force-refresh the just-styled
    layer after `apply_table_color_source_to_labels_layer(...)`.
  - Object-classification labels coloring should use the same helper once
    Slice 6.6b routes `user_class` / `pred_class` through compact categorical
    colormaps and still uses direct labels colormaps for `pred_confidence`.
- Prefer a small helper with a no-op fallback for test/dummy viewers:

  ```python
  def _force_sync_labels_slice_after_colormap_change(viewer: object, layer: Labels) -> None:
      layer_slicer = getattr(viewer, "_layer_slicer", None)
      dims = getattr(viewer, "dims", None)
      if layer_slicer is None or dims is None:
          return
      with layer_slicer.force_sync():
          layer_slicer.submit(layers=[layer], dims=dims, force=True)
  ```

- This deliberately blocks only after an explicit Harpy recoloring action. It
  does not disable async slicing globally for normal navigation.
- If forcing a sync slice is too expensive in practice, a more refined follow-up
  is to recode the already-loaded current raw slice in place and emit
  `layer.events.set_data()`. That avoids dask I/O for the current view, but it
  touches more napari internals and should only be chosen if the simple
  force-sync helper is too slow.

Tests:

- Add a `ViewerModel`-based unit test for the helper:
  - create a `Labels` layer with a previous labels colormap and loaded raw
    slice;
  - assign a direct labels colormap;
  - call the helper;
  - assert the displayed texture-code view equals
    `layer.colormap._data_to_texture(layer._slice.image.raw)`.
- Add a test that the helper is a no-op for `DummyViewer` / non-napari viewer
  objects so existing adapter tests stay lightweight.
- Keep styled-label semantic tests focused on final mapped colors; do not assert
  on transient async intermediate states.

Manual QA:

- Launch

  ```python
  Interactive(sdata, async_slicing=True)
  ```

  using `/Users/arne.defauw/VIB/DATA/test_data/sdata_xenium_3_6_26.zarr`.
- Color `nucleus_segmentation_mask` by
  `table_transcriptomics_preprocessed.obs["leiden"]` and by a continuous column
  such as `total_counts`.
- Confirm the styled labels layer never shows the wrong full-field/speckled
  color mismatch seen before the fix.
- Repeat with `async_slicing=False` to confirm no behavior change.

### Slice 9: Optional Sorted-Index Fast Path For Compact Mapping

Status: proposed.

The remaining compact categorical construction cost after Slice 6.6b is mostly
defensive index validation and sorting in
`compact_categorical_labels_mapping_from_values(...)`.

Observed on a synthetic `406,611`-row object-classification repaint:

```text
_read_class_values_without_normalizing(...) before vectorized fast path: ~0.064 s
_read_class_values_without_normalizing(...) after vectorized fast path:  ~0.001 s

compact_categorical_labels_mapping_from_values(...):                 ~0.028 s
  _positive_label_ids_from_index(...):                               ~0.023 s
  _categorical_texture_codes(...):                                   ~0.001 s
  sort/apply already-sorted label ids:                               ~0.005-0.008 s
```

The expensive part is not the categorical value mapping anymore. It is mostly
`np.unique(...)`-based uniqueness validation and unconditional sorting of the
label-id index.

Important safety decision:

- do not assume the table/index is sorted;
- do not remove validation globally;
- only take the faster path when the index itself proves it is already
  strictly increasing positive integer label ids.

Implementation direction:

1. Add an internal helper that detects and returns a trusted sorted-positive
   label-id array only when all of these are true:
   - the index converts losslessly to integer label ids;
   - every label id is positive;
   - label ids are strictly increasing:

     ```python
     len(label_ids) == 0 or (
         label_ids[0] > 0
         and np.all(label_ids[1:] > label_ids[:-1])
     )
     ```

   This single monotonic check proves both sortedness and uniqueness for the
   fast path.
2. If the monotonic check succeeds:
   - skip the `np.unique(...)` uniqueness check;
   - skip `np.argsort(...)`;
   - keep `label_ids` and `texture_codes` in their existing order.
3. If the monotonic check fails:
   - fall back to the current conservative path;
   - validate uniqueness with `np.unique(...)`;
   - sort label ids and texture codes before constructing
     `CompactCategoricalLabelsMapping`.
4. Keep this optimization local to compact label-coloring construction. It
   should not change table alignment, feature-row construction, hover features,
   or labels image data.
5. Benchmark both paths:
   - sorted positive integer index, expected common Harpy path;
   - unsorted positive integer index, fallback path;
   - duplicate index, still rejected;
   - non-integer index, still rejected;
   - non-positive index, still rejected.

Acceptance criteria:

- sorted positive integer indexes produce the same colors as the current
  implementation;
- unsorted valid indexes still work and are sorted internally as before;
- duplicate, non-integer, and non-positive indexes still fail loudly;
- no caller relies on sortedness unless the helper has explicitly detected it;
- the compact categorical construction benchmark shows the expected reduction
  in `_positive_label_ids_from_index(...)` and sorting overhead;
- object-classification `user_class` and `pred_class` full repaint tests keep
  passing.

### Slice 10: Strict Object-Classification Class Palette Contract

Status: proposed.

`ViewerStylingController._get_class_color_lookup(...)` currently mixes two
responsibilities:

- read the table-backed class palette for `user_class` / `pred_class`;
- recover from invalid or incomplete class-column state by re-normalizing
  values and backfilling palettes.

That makes the styling path harder to reason about and keeps fallback branches
alive in a performance-sensitive color-repaint path.

Preferred direction:

- make object-classification styling a strict reader;
- keep mutation/repair at mutation boundaries:
  - `set_user_class_for_rows(...)`;
  - `_ensure_prediction_columns(...)`;
  - `set_class_annotation_state(...)`.

Current guarantees and nuance:

- For `pred_class`, the normal widget flow is strict already:
  `ClassifierController.bind(...)` calls `_ensure_prediction_columns(...)`,
  which calls `set_class_annotation_state(...)`. Therefore `pred_class` should
  exist, be categorical, have integer categories, and have a synced
  `pred_class_colors` palette before styling.
- For `user_class`, the initial no-annotation state is intentionally lazier:
  `user_class` may be absent until the first call to
  `set_user_class_for_rows(...)`. In that state,
  `_get_region_feature_rows(...)` exposes all observed class values as `0`,
  and the viewer should render everything with the unlabeled/default color.

Implementation direction:

1. Replace `_get_class_color_lookup_from_normalized_values(...)` with strict
   validation helpers.
2. Treat an existing table-backed class column as a contract:
   - the column must be pandas categorical;
   - categories must be integer class ids;
   - categories must be sorted, unique, and include `0`;
   - categorical codes must not contain missing values;
   - `table.uns[colors_key]` must exist;
   - stored palette length must match the category count.
3. Treat `observed_class_values` as already prepared by
   `feature_rows[self._color_by]`:
   - values must be integer dtype;
   - values must be non-negative;
   - missing/float/object values should fail loudly instead of triggering a
     hidden normalization fallback.
4. Preserve one explicit special case:
   - if `category_column == USER_CLASS_COLUMN` and `user_class` is absent,
     allow styling only when observed class values are all `0`;
   - return a lookup containing only the unlabeled class color;
   - if observed values contain nonzero classes while `user_class` is absent,
     raise a clear `ValueError`.
5. For `pred_class`, absence of the table-backed column should fail loudly in
   widget-backed styling, because the classifier controller should have called
   `set_class_annotation_state(...)` via `_ensure_prediction_columns(...)`
   before styling.
6. Keep deterministic color backfilling only for class ids that are observed
   but not present in the stored palette categories, if that situation remains
   a deliberate product behavior. Otherwise fail loudly there too and require
   palette sync at the mutation boundary.

Acceptance criteria:

- valid `user_class` and `pred_class` categorical columns still produce the
  same compact categorical colors;
- `pred_class` missing from a bound widget table raises a clear error instead
  of silently backfilling;
- missing pre-annotation `user_class` with all observed values equal to `0`
  still renders as unlabeled/default gray;
- missing `user_class` with observed nonzero class values raises a clear error;
- invalid categorical contracts fail loudly with actionable messages:
  non-categorical column, non-integer categories, missing category codes,
  missing palette, palette-length mismatch;
- tests that currently expect fallback normalization for invalid table state
  are updated to expect strict failure or are removed if they only covered the
  old recovery behavior;
- `_get_class_color_lookup(...)` is shorter and has no hidden
  `normalize_class_values(...)` recovery branch.

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
