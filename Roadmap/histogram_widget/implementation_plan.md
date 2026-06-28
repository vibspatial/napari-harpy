# Histogram Widget Implementation Plan

## Purpose

Build a production-quality histogram widget around explicit image targets:

```text
coordinate_system, image_name, channel_name
```

Each target is represented by a card. The user can add as many cards as needed,
select the target explicitly, press Calculate for that card, and inspect the
histogram inside the card. Contrast-limit controls synchronize with matching
napari image layers when those layers are loaded.

Every histogram card owns its own calculation settings. Values such as `scale`,
`bins`, `value_range`, `density`, `exclude_nan`, `exclude_zeros`, `log_y`,
`percentile_min`, and `percentile_max` are per-card inputs, not shared
widget-level settings. Changing one card's settings must not affect any other
card.

The widget is a visualization and quality-control surface. It must not mutate
the underlying `SpatialData` image data.

## Recommended Direction

Use explicit histogram target cards as the source of truth.

Do not make the active napari layer the canonical selection. Active-layer
following is convenient, but it becomes ambiguous once coordinate systems,
stack/overlay image modes, unloaded images, and multiple histograms are in play.
The active layer can still be used later as a convenience action such as
"Add card from active image layer".

Compute histogram values with a small napari-harpy core calculator that mirrors
Harpy's filtering and Dask semantics, rather than using
`hp.qc.image_histogram(...)` as the widget engine. Harpy's QC function is useful
as a reference and already supports percentile guide lines, but it returns
Matplotlib axes, not the structured data needed for stale-job handling,
contrast synchronization, and tests.

Use `pyqtgraph` for plotting. This is an explicit product dependency decision:
the histogram widget is an interactive Qt control surface, so the plot backend
should be optimized for responsive redraws, marker interaction, and a native
Qt feel. Add `pyqtgraph` as a direct napari-harpy dependency and do not build a
Matplotlib fallback path for the widget.

## Evaluated Options

### Selection Model

Active napari layer:

- simple when the user only wants the current layer;
- weak fit for coordinate-system-specific workflows;
- only naturally represents one histogram at a time;
- risky because manual layer selection can silently change the widget target.

Loaded-layer selector:

- keeps contrast sync straightforward;
- cannot compute histograms for images that are not loaded;
- still makes coordinate system and channel identity indirect.

Explicit target cards:

- matches the feature extraction widget's triplet-card pattern;
- supports many histograms in one widget;
- keeps coordinate system, image, and channel visible and auditable;
- lets histogram calculation work from `SpatialData` even before a layer is
  loaded;
- requires a separate live-layer resolution step for contrast sync.

Recommendation: explicit target cards.

### Histogram Calculation

Call `hp.qc.image_histogram(...)`:

- minimizes duplicated logic;
- supports `scale`, `log_y`, `exclude_zeros`, `exclude_nan`, and
  `percentile_lines`;
- supports `range`, which restricts the histogram value range;
- is plot-oriented and does not return counts, bin edges, percentile values, or
  calculation metadata.

Implement `napari_harpy.core.histogram`:

- returns `HistogramResult` data for plotting and tests;
- keeps worker payloads immutable and stale-result checks simple;
- allows contrast and percentile markers to be redrawn without rerunning the
  full plot function;
- supports histogram counts only for the first implementation; ECDF/cumulative
  distribution calculation is intentionally out of scope;
- duplicates a small amount of Harpy QC logic, so tests should lock behavior to
  Harpy-compatible semantics.

Recommendation: implement a small local calculator, with Harpy's function as
the semantic reference.

### Plot Backend

Matplotlib:

- already available through the current environment;
- easy to test at the data/state level;
- useful as Harpy's static QC plotting reference;
- weaker fit for an interactive Qt widget with many plots and marker updates.

Pyqtgraph:

- native Qt plotting library;
- better fit for many lightweight histogram views;
- better fit for responsive contrast/percentile marker updates;
- keeps direct marker dragging possible without changing plotting backends;
- accepted as a new direct napari-harpy dependency.

Recommendation: pyqtgraph only. Do not implement a Matplotlib fallback for the
widget.

## Pyqtgraph Plot Architecture

Each histogram card should own one small plot widget, for example a
`_HistogramPlotWidget`, that wraps pyqtgraph and exposes data-oriented methods:

```python
class _HistogramPlotWidget(QWidget):
    def set_histogram(self, result: HistogramResult) -> None: ...
    def set_contrast_limits(self, low: float, high: float) -> None: ...
    def set_percentile_markers(self, markers: Mapping[float, float]) -> None: ...
    def clear_histogram(self) -> None: ...
```

The controller and calculator should never manipulate pyqtgraph objects
directly. They pass `HistogramResult` and contrast/percentile values into this
plot wrapper.

The plot wrapper should own the draggable contrast region and expose one Qt
signal such as:

```python
contrast_limits_dragged = Signal(float, float)
```

This signal is emitted only when the user moves the histogram contrast region.
Programmatic updates from napari layer events should move the region without
re-emitting `contrast_limits_dragged`.

Recommended pyqtgraph items:

- `pyqtgraph.PlotWidget` or `GraphicsLayoutWidget` containing one `PlotItem`;
- `pyqtgraph.BarGraphItem` for histogram bars, using the calculated bin edges
  and counts/densities;
- `pyqtgraph.LinearRegionItem` for contrast limits, configured as a vertical
  movable region with two boundary lines and subtle fill; if the UX needs fully
  independent line behavior, use two movable `pyqtgraph.InfiniteLine` items
  instead;
- non-movable dashed `pyqtgraph.InfiniteLine` items for percentile markers;
- `pyqtgraph.TextItem` labels only for percentile markers and compact marker
  readouts where they do not clutter the plot.

The contrast region is preferred over two unrelated line objects because it
keeps the lower/upper pair visually connected, makes dragging predictable, and
helps enforce valid ordering. It must still read visually as two vertical
contrast-limit lines.

Suggested contrast-region implementation:

1. Create one `pyqtgraph.LinearRegionItem` with vertical orientation and
   `movable=True`.
2. Style the two boundary lines with the contrast-limit line pen and the region
   brush with a low-alpha fill.
3. On `sigRegionChangeFinished`, read `low, high = region.getRegion()`.
4. Normalize and validate the pair, preserving `low < high`.
5. Emit `contrast_limits_dragged(low, high)` from `_HistogramPlotWidget`.
6. In the card/widget layer-binding code, handle that signal by assigning
   `layer.contrast_limits = (low, high)`.
7. Listen to `layer.events.contrast_limits`; when napari's native contrast
   controls change, call `_HistogramPlotWidget.set_contrast_limits(low, high)`.
8. Use an internal update guard such as `_syncing_contrast_limits` so
   programmatic region moves do not recursively assign `layer.contrast_limits`.

This makes the pyqtgraph region and napari's native contrast controls two UI
views over the same layer state, rather than independent state machines.

Plot behavior:

- the x-axis is intensity and the y-axis is frequency or density;
- plot x-range should fit the histogram bin edges by default;
- contrast-limit changes should update marker items without replacing the bar
  item;
- percentile changes should update percentile marker items without replacing
  the bar item when the histogram result is otherwise unchanged;
- log y-axis uses pyqtgraph's y-axis log mode and should handle zero-count bins
  without distorting the underlying histogram data;
- the plot should show a quiet empty/stale/running state instead of a blank
  black rectangle.

## Plot Palette And Styling

Use a restrained product palette derived from existing napari-harpy widget
tokens. Do not use ad hoc random colors or a Matplotlib-style color cycle.

Suggested constants should live near the histogram widget, for example in
`src/napari_harpy/widgets/histogram/styles.py`, and should reuse shared style
tokens from `widgets/shared_styles.py`.

Recommended plot styling:

| Element | Color/style |
| --- | --- |
| Plot background | `WIDGET_PANEL_SUBTLE_COLOR` or transparent over the card surface |
| Axis text | `WIDGET_TEXT_MUTED_COLOR` |
| Axis/grid lines | `WIDGET_BORDER_COLOR` with low alpha |
| Histogram bars | stable muted data blue such as `#7EA7FF` at medium alpha |
| Histogram bar edge | no edge, or `WIDGET_BORDER_STRONG_COLOR` at low alpha |
| Contrast-limit lines | `WIDGET_SUCCESS_COLOR`, solid, 2 px |
| Contrast selected region fill | `WIDGET_SUCCESS_COLOR` at very low alpha |
| Percentile lines | `WIDGET_WARNING_TEXT_COLOR`, dashed, 1 to 1.5 px |
| Percentile labels | `WIDGET_WARNING_TEXT_COLOR` |
| Empty/stale text | `WIDGET_TEXT_MUTED_COLOR` |

Rationale:

- histogram bars should be calm and data-focused;
- contrast-limit markers are interactive display controls, so they get the
  strongest affordance;
- percentile markers are secondary analytical guides and should be visually
  distinct from contrast controls;
- the palette should work on the existing dark widget surface and remain
  readable in dense cards.

If a matching live overlay layer has an obvious channel color, a later polish
can show that color as a small swatch in the card header. Do not make bar colors
depend on random per-card color cycling.

## Data Model

Add focused dataclasses in `src/napari_harpy/core/histogram.py`:

```python
@dataclass(frozen=True)
class HistogramTarget:
    coordinate_system: str
    image_name: str
    channel_name: str | None


@dataclass(frozen=True)
class HistogramSettings:
    bins: int = 256
    value_range: tuple[float, float] | None = None
    density: bool = False
    exclude_nan: bool = True
    exclude_zeros: bool = False
    log_y: bool = False
    scale: str | None = None
    percentile_min: float | None = None
    percentile_max: float | None = None


@dataclass(frozen=True)
class HistogramResult:
    target: HistogramTarget
    settings: HistogramSettings
    counts: np.ndarray
    bin_edges: np.ndarray
    data_range: tuple[float, float]
    percentile_values: Mapping[float, float]
    resolved_scale: str | None
```

`HistogramSettings` is stored per card and copied into each immutable
`HistogramJob`. There should be no shared global `bins`, `scale`,
`value_range`, filtering, or percentile state that implicitly changes all cards
at once.

For scalar images without a channel axis, keep `channel_name=None` internally
and display a read-only scalar channel placeholder in the UI. Multi-channel
images should require an explicit channel selection.

## Defaults And Optional Settings UI

Users should be able to calculate a histogram after selecting only:

```text
coordinate_system, image_name, channel_name
```

All calculation settings are optional and have sensible defaults.

Recommended defaults:

| Setting | Default | UI behavior |
| --- | --- | --- |
| `scale` | exact full-resolution data, displayed as `scale0` for multiscale images | show available scales only for multiscale images |
| `bins` | `256` | numeric stepper/spin box |
| `value_range` | empty/auto data range after filtering | optional histogram min/max numeric fields |
| `density` | `False` | checkbox/toggle |
| `exclude_nan` | `True` | checkbox/toggle |
| `exclude_zeros` | `False` | checkbox/toggle |
| `log_y` | `False` | checkbox/toggle |
| `percentile_min` | empty/off | optional numeric field |
| `percentile_max` | empty/off | optional numeric field |

The nicest UI is a collapsed per-card disclosure section named
`Histogram settings`. It should be collapsed by default so the normal workflow
stays target-first:

1. choose target
2. Calculate
3. inspect histogram and contrast limits

Do not use a generic `Advanced` label by itself. These settings are optional,
but not all are advanced; `Histogram settings` is clearer and more durable.

The collapsed settings header should summarize the active settings, for
example:

```text
Histogram settings: scale0, 256 bins
Histogram settings: scale2, range 0-4095, 512 bins, log y, p0.1-p99.9
```

This prevents hidden state from becoming surprising when users have several
histogram cards open. The section should also expose a `Reset settings` action
that restores that card's defaults without changing any other card.

Implementation note: the existing viewer disclosure widgets in
`src/napari_harpy/widgets/viewer/disclosure.py` are a good visual reference, but
they are private and viewer-specific. For the histogram widget, either extract a
small reusable shared collapsible section or add a histogram-local section with
the same styling and accessibility behavior.

## Contrast Sync Policy

After a card has a calculated histogram, resolve the matching live napari layer
through `ViewerAdapter.layer_bindings`.

For overlay layers:

- match `sdata`, `coordinate_system`, `image_name`, and `channel_name` or
  `channel_index`;
- sync contrast limits only to that channel layer.

For stack layers:

- match `sdata`, `coordinate_system`, and `image_name`;
- use the selected channel as the histogram source;
- sync the stack layer's scalar `contrast_limits`;
- communicate that independent per-channel contrast requires overlay mode.

For RGB(A) layers:

- histogram display can remain available when a scalar source can be resolved;
- contrast synchronization should be disabled because napari does not expose normal
  scalar contrast behavior for RGB layers.

The histogram value range and the napari contrast limits are separate concepts.
`HistogramSettings.value_range` corresponds to Harpy's
`hp.qc.image_histogram(..., range=(low, high))` parameter and controls which
values are included in the histogram bins. When `value_range=None`, compute the
histogram range automatically from the filtered data using the same
NaN/zero-filtered array used for the histogram. When `value_range=(low, high)`
is set, values outside that range are ignored by the histogram calculation.

Current napari contrast limits are drawn as two movable vertical lines in the
pyqtgraph histogram and synchronized with napari's native image contrast
controls.

Contrast-limit synchronization is bi-directional:

- dragging either vertical histogram line updates `layer.contrast_limits`;
- changing napari's native layer contrast controls emits
  `layer.events.contrast_limits`, which updates the vertical lines, contrast
  region, and any marker readout text;
- updates must guard against feedback loops while preserving one authoritative
  contrast-limit pair.

Use distinct styling for these two contrast-limit lines so they are visually
different from percentile guide lines.

## Percentile Policy

`percentile_min` and `percentile_max` are visual guide settings.

When provided, compute the requested percentile values after the same NaN/zero
filtering used by the histogram. Match Harpy's semantics: percentile values are
computed with `dask.array.percentile(..., internal_method="tdigest")` and are
not clipped to `HistogramSettings.value_range`. Draw the resulting values as
labeled vertical lines. Do not automatically change napari contrast limits when
percentile values are typed or recalculated.

Applying percentiles to contrast limits should be an explicit action, for
example an "Apply percentiles" button. This avoids surprising display changes
and leaves room to decide whether applying percentiles should set both limits,
only one limit, or operate independently per histogram card.

## Implementation Slices

### 1. Core Histogram Calculator

Status: [ ] Planned

Goal:

- create a pure, testable histogram calculation layer independent from Qt.

Scope:

- add `src/napari_harpy/core/histogram.py`;
- define the Slice 1 target and settings dataclasses exactly as:

  ```python
  @dataclass(frozen=True)
  class HistogramTarget:
      coordinate_system: str
      image_name: str
      channel_name: str | None


  @dataclass(frozen=True)
  class HistogramSettings:
      bins: int = 256
      value_range: tuple[float, float] | None = None
      density: bool = False
      exclude_nan: bool = True
      exclude_zeros: bool = False
      log_y: bool = False
      scale: str | None = None
      percentile_min: float | None = None
      percentile_max: float | None = None
  ```

- expose a small public calculation function:

  ```python
  def calculate_histogram(
      sdata: SpatialData,
      target: HistogramTarget,
      settings: HistogramSettings,
  ) -> HistogramResult:
      ...
  ```

- keep the module free of Qt, napari viewer, and pyqtgraph dependencies;
- resolve `HistogramTarget` against `SpatialData`;
- validate that the image exists in `sdata.images`;
- validate that the requested coordinate system exists for the selected image;
- validate that the requested channel exists for multi-channel images;
- support scalar images without a channel axis by requiring
  `target.channel_name is None`;
- accept one explicit `HistogramSettings` object per calculation;
- validate `settings.bins` as a positive integer;
- validate `settings.value_range` as finite `(low, high)` values with
  `low < high`;
- validate requested percentile values as finite values in `[0, 100]`;
- select the requested multiscale `scale`, defaulting to exact `scale0`;
- record the resolved scale used for the result;
- select the requested channel, or scalar image data when no channel axis is
  present;
- flatten values;
- apply `exclude_nan` and `exclude_zeros`;
- call `compute_chunk_sizes()` when filtering creates unknown Dask chunks;
- compute requested percentile values after NaN/zero filtering and before
  applying `value_range`, matching Harpy's behavior;
- use Dask's tdigest percentile path:

  ```python
  percentile_values = dask.array.percentile(
      filtered_values,
      q=list(requested_percentiles),
      internal_method="tdigest",
  ).compute()
  ```

- compute automatic histogram range from the filtered data when
  `value_range=None`;
- pass explicit `value_range` through as the Dask histogram `range` parameter
  when the user provides it;
- compute histogram counts and bin edges with `dask.array.histogram(...)`;
- return a structured `HistogramResult`.

Calculation order:

1. resolve image, scale, and channel;
2. flatten selected values;
3. apply NaN and zero filtering;
4. compute requested percentiles from the filtered values;
5. choose histogram range from either `value_range` or the filtered data;
6. compute counts and bin edges.

Non-goals:

- no Qt widgets;
- no pyqtgraph objects;
- no napari layer lookup;
- no contrast synchronization;
- no ECDF or cumulative distribution calculation;
- no histogram result caching.

Tests:

- default settings compute a valid histogram without requiring optional inputs;
- counts and bin edges match expected values for small Dask-backed images;
- NaN and zero filtering match the documented semantics;
- unknown chunk sizes after filtering are handled;
- channel names and channel indices resolve consistently;
- scalar images without a channel axis are supported;
- missing image, coordinate system, or channel selections raise clear
  `ValueError` exceptions;
- invalid `bins` values are rejected;
- invalid percentile values outside `[0, 100]` are rejected;
- percentile values are computed after NaN/zero filtering and are not clipped by
  `value_range`;
- percentile values use `internal_method="tdigest"` and are asserted with an
  appropriate tolerance instead of exact equality where Dask's tdigest
  approximation matters;
- explicit `value_range` restricts histogram counts and bin edges as expected;
- invalid `value_range` values, including `low >= high`, are rejected.

### 2. Widget Shell And Target Cards

Status: [ ] Planned

Goal:

- add the `Image Histogram` widget with explicit target-card selection.

Scope:

- add `pyqtgraph` to `pyproject.toml` as a direct runtime dependency;
- add `src/napari_harpy/widgets/histogram/`;
- register the widget in `src/napari_harpy/napari.yaml`;
- attach to shared `HarpyAppState` through `get_or_create_app_state(...)`;
- render an add-card action and a scrollable list of histogram cards;
- each card exposes coordinate system, image, channel, and Calculate as the
  primary card controls;
- each card exposes `scale`, `bins`, `value_range`, `density`, `exclude_nan`,
  `exclude_zeros`, `log_y`, and percentile fields inside a collapsed
  `Histogram settings` disclosure section;
- each card owns and remembers its own `HistogramSettings` while it remains
  visible;
- the settings section is collapsed by default and displays a concise summary
  of defaults or overrides;
- the settings section exposes a per-card `Reset settings` action;
- coordinate systems and images come from shared `SpatialData` discovery
  helpers;
- channel names come from `get_image_channel_names_from_sdata(...)`;
- changing target/settings marks that card's result stale until Calculate is
  pressed again.

Tests:

- widget instantiates without a viewer;
- widget seeds from shared `sdata`;
- cards can be added and removed;
- coordinate-system/image/channel selectors refresh on `sdata_changed`;
- Calculate is available with default settings once the target is valid;
- optional settings are hidden in a collapsed per-card settings section by
  default;
- the collapsed settings header summarizes active defaults/overrides;
- resetting settings affects only the current card;
- `scale`, `bins`, histogram value range, filtering, log-axis, density, and
  percentile controls are independent per card;
- stale card state is visible after settings change.

### 3. Background Controller

Status: [ ] Planned

Goal:

- run histogram jobs without blocking the Qt main thread.

Scope:

- add `widgets/histogram/controller.py`;
- define immutable `HistogramJob` with `card_id`, `job_id`, target, settings,
  and `sdata`;
- use the same `thread_worker` resolution pattern as feature extraction;
- track the latest requested job per card;
- ignore stale results when target/settings change during calculation;
- support one active worker per card;
- expose status text and status kind for card-level feedback;
- cancel or ignore jobs when a card is removed.

Tests:

- Calculate starts a worker for a valid card;
- worker jobs receive the settings from the card that launched them;
- stale worker results are ignored;
- worker errors surface as card errors;
- removing a card disconnects or ignores its worker result.

### 4. Histogram Plot Rendering

Status: [ ] Planned

Goal:

- draw each calculated histogram inside its card with pyqtgraph.

Scope:

- add a histogram plot wrapper such as `_HistogramPlotWidget`;
- embed a `pyqtgraph.PlotWidget` or equivalent pyqtgraph graphics view per
  card;
- draw bars from `HistogramResult.counts` and `bin_edges` with
  `pyqtgraph.BarGraphItem` or an equivalent pyqtgraph item;
- add histogram plot palette/style constants derived from
  `widgets/shared_styles.py`;
- configure plot background, axes, grid, bar fill, contrast region, percentile
  lines, and empty/stale text from those constants;
- support linear and log y-axis;
- draw a stale/empty state before calculation;
- redraw from existing result when only non-histogram markers change;
- expose hooks for adding/updating/removing the movable contrast-limit region
  and non-movable percentile guide lines;
- keep plot setup isolated in a small widget/helper so controller and
  calculator code never depend on pyqtgraph objects;
- keep plot labels compact and card-local.

Tests:

- pyqtgraph plot updates when a result arrives;
- log y-axis setting is applied;
- stale state is shown after target/settings changes;
- repeated calculations replace the previous plot instead of accumulating
  duplicate plot items;
- plot colors/styles come from the histogram palette constants;
- histogram bars, contrast controls, percentile markers, axes, and empty state
  have distinct visual roles.

### 5. Contrast-Limit Synchronization

Status: [ ] Planned

Goal:

- synchronize the movable histogram contrast region with matching napari image
  layers and napari's native image contrast controls.

Scope:

- resolve matching `ImageLayerBinding` for each card after calculation and on
  viewer layer lifecycle changes;
- initialize the histogram contrast region from `layer.contrast_limits`;
- draw the contrast limits with `pyqtgraph.LinearRegionItem` by default,
  styled to read as two movable vertical contrast-limit lines with a subtle
  selected-region fill;
- do not add a histogram-widget contrast slider or contrast low/high numeric
  fields; napari's native image contrast controls remain the non-plot control
  surface;
- `_HistogramPlotWidget` emits a card-level contrast-change signal when the
  user finishes moving either boundary of the contrast region;
- handling that signal updates
  `layer.contrast_limits`;
- `layer.events.contrast_limits` updates the histogram contrast region and any
  marker readout text when napari's own contrast controls change;
- prevent event feedback loops and keep lower/upper ordering valid while users
  drag markers;
- layer removal disables contrast synchronization without clearing the
  histogram;
- RGB(A) layers disable contrast synchronization with a clear status.

Tests:

- overlay channel target resolves the correct overlay layer;
- stack target resolves the stack layer;
- moving the lower/upper histogram lines assigns `layer.contrast_limits`;
- external napari contrast changes update the widget;
- external napari contrast changes move the histogram contrast region;
- programmatic contrast-region updates do not recursively reassign
  `layer.contrast_limits`;
- contrast-limit line styling remains distinct from percentile styling;
- layer removal clears the sync binding;
- RGB layers disable contrast synchronization.

### 6. Percentile Guide Lines

Status: [ ] Planned

Goal:

- visualize user-specified percentile markers on the histogram.

Scope:

- add `percentile_min` and `percentile_max` numeric fields;
- validate values in `[0, 100]`;
- compute percentile values in the histogram job;
- draw labeled percentile guide lines distinct from contrast-limit lines;
- show the computed percentile intensity values in the card;
- do not automatically update contrast limits from percentile values.

Tests:

- percentile values are computed from the filtered data;
- percentile lines are drawn only when specified;
- clearing a percentile field removes the corresponding marker after the next
  calculation;
- changing percentile settings marks the card stale.

### 7. Explicit Percentile-To-Contrast Action

Status: [ ] Planned

Goal:

- decide and implement a deliberate action for using percentiles as contrast
  limits.

Scope:

- add an "Apply percentiles" button only when both percentile values and a live
  contrast-sync layer are available;
- applying percentiles sets `layer.contrast_limits` once;
- keep ongoing two-way contrast sync after the assignment;
- do not add a persistent auto-apply toggle unless the workflow proves it is
  necessary.

Tests:

- button is disabled without a live layer or missing percentile values;
- applying percentiles sets contrast limits to the computed values;
- the normal contrast-limit event path updates the histogram contrast region.

### 8. Product Hardening

Status: [ ] Planned

Goal:

- make the widget reliable under large data, layer churn, and repeated use.

Scope:

- add card-level empty, running, success, warning, and error states;
- ensure repeated add/remove/calculate cycles do not leak layer event handlers;
- keep lower-resolution multiscale choices explicit and label the scale used;
- add an optional "Calculate all" action only after per-card behavior is solid;
- document that histogram results are computed from `SpatialData`, while
  contrast sync requires a matching live napari layer.

Tests:

- no duplicate event callbacks after recalculation or layer replacement;
- multiscale scale selection is passed to the calculator;
- per-card settings remain independent when multiple histograms are visible;
- `napari.yaml` contribution loads the widget.

## Out Of Scope

- no automatic active-layer retargeting;
- no automatic contrast changes from percentile fields;
- no Matplotlib fallback for the widget;
- no ECDF or cumulative distribution calculation;
- no histogram result caching layer;
- no CSV export;
- no overlaying multiple target histograms into one shared plot.
