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

Status: proposed.

Create an isolated prototype, for example
`CompactCategoricalLabelColormap`, that can be assigned where napari expects a
`DirectLabelColormap`.

The prototype should:

- subclass `DirectLabelColormap` so napari's `_normalize_label_colormap(...)`
  accepts it as a labels colormap instance;
- store the compact state from Slice 6.2;
- make the compact state, not `color_dict`, the source of truth for full
  per-label coloring;
- expose enough `color_dict` behavior for napari's direct-mode detection,
  default color lookup, and scalar compatibility, without rebuilding the full
  large `label_id -> RGBA` dictionary;
- never lazily materialize a full `label_id -> RGBA` dictionary from
  `color_dict`, `map(...)`, `_values_mapping_to_minimum_values_set(...)`,
  `_label_mapping_and_color_dict`, `_num_unique_colors`, `_data_to_texture(...)`,
  or cache rebuilding;
- avoid making a large Python `label_id -> texture_code` dict the default
  source of truth if an array-backed mapping can satisfy the hot paths more
  cheaply;
- if a dict or typed-dict representation is needed for a napari/numba path,
  build it at most once per compact state and make repeated access use the
  cache;
- keep scalar/default compatibility small and explicit; if a method needs
  per-label information, it should use `label_id -> texture_code` plus
  `texture_code -> RGBA` directly;
- avoid looking like a default-only direct colormap, because napari may switch
  the layer back to auto color mode in that case;
- override/support the narrow methods/properties identified in Slice 6.1:
  - `_values_mapping_to_minimum_values_set(...)`;
  - `_label_mapping_and_color_dict`;
  - `_num_unique_colors`;
  - `_data_to_texture(...)`;
  - `map(...)`;
- `_map_without_cache(...)` / `_get_mapping_from_cache(...)` behavior for
    small `uint8`/`uint16` labels;
  - `_clear_cache(...)`;
- for small `uint8`/`uint16` labels, build bounded dense lookup arrays from
  compact state and cache them per dtype, preserving napari's small-label path
  without constructing a full Python `label_id -> RGBA` dictionary;
- preserve `use_selection`, `selection`, and `background_value` behavior that
  napari mutates directly;
- keep `_values_mapping_to_minimum_values_set(apply_selection=True)` compatible
  with napari's selected-label-only path;
- keep `_values_mapping_to_minimum_values_set()[1]` compatible with vispy's
  `build_textures_from_dict(...)`, including sequential texture-code keys;
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
- selected-label behavior with `use_selection=True`;
- `_values_mapping_to_minimum_values_set(apply_selection=True/False)`;
- `_data_to_texture(...)` for high-bit labels;
- lookup-texture behavior for small `uint8`/`uint16` label ids;
- `Labels.get_color(...)`;
- assignment to a real `Labels` layer, including direct color mode detection;
- `Labels.show_selected_label` / `Labels.selected_label` mutation of the
  colormap;
- multiscale labels where feasible in headless tests.

Acceptance criteria:

- visual parity with the Slice 1 helper for categorical `.obs` colors;
- `map(...)` parity for representative labels, missing labels, background, and
  selection mode;
- `layer.colormap = compact_colormap` keeps the layer in direct color mode;
- small integer labels and large/high-bit labels both map correctly;
- compact texture-code keys are sequential and suitable for vispy texture
  upload;
- none of the tested compact-colormap methods materialize a full
  `label_id -> RGBA` dictionary as a side effect;
- the compact implementation documents whether `label_id -> texture_code` is
  stored as arrays, a dict, or both, and tests the repeated-access cache
  behavior for whichever representation is used;
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
- first `_values_mapping_to_minimum_values_set(...)` call, because vispy uses
  it to upload the high-bit direct color table;
- `_data_to_texture(...)` on representative current-slice data, because labels
  slicing uses it before rendering;
- repeated access to `color_dict`, `map(...)`, `_num_unique_colors`,
  `_label_mapping_and_color_dict`, and
  `_values_mapping_to_minimum_values_set(...)`, to catch accidental lazy
  full-dict materialization;
- `_data_to_texture(...)` remapping cost for representative current-slice data,
  because the `label_id -> texture_code` lookup can become the new bottleneck
  if implemented with per-pixel Python work;
- end-to-end `apply_table_color_source_to_labels_layer(...)` equivalent;
- memory-ish size of the resulting Python-side mapping objects, if practical;
- Slice 1 helper timings in the same run as a baseline only, not as a planned
  production fallback.

Acceptance criteria:

- materially faster construction and/or assignment than the Slice 1 helper on
  `table_global_ROI1.obs["leiden"]`;
- or materially lower memory use with no performance regression;
- no worse high-bit texture mapping behavior than the Slice 1 helper;
- no worse small-label lookup behavior on a synthetic `uint16` labels case;
- repeated compact-colormap method access does not allocate or cache a full
  large `label_id -> RGBA` mapping;
- `label_id -> texture_code` lookup/remapping is not slower than napari's
  current direct-label mapping on representative high-bit label slices;
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
- keep the Slice 1 helper only for direct RGBA use cases that still need it,
  not as a second categorical styled-labels implementation;
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
