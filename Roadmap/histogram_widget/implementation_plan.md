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
`bins`, `density`, `exclude_nan`, `exclude_zeros`, `log_y`, `percentile_min`,
and `percentile_max` are per-card inputs, not shared widget-level settings.
Changing one card's settings must not affect any other card.

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
- is plot-oriented and does not return counts, bin edges, percentile values, or
  calculation metadata.

Implement `napari_harpy.core.histogram`:

- returns `HistogramResult` data for plotting and tests;
- keeps worker payloads immutable and stale-result checks simple;
- allows contrast and percentile markers to be redrawn without rerunning the
  full plot function;
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
- contrast-limit controls are interactive display controls, so they get the
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
`HistogramJob`. There should be no shared global `bins`, `scale`, filtering, or
percentile state that implicitly changes all cards at once.

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
Histogram settings: scale2, 512 bins, log y, p0.1-p99.9
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
- contrast controls should be disabled because napari does not expose normal
  scalar contrast behavior for RGB layers.

The histogram bin range should remain the calculated data range by default.
Current napari contrast limits are drawn as two movable vertical lines in the
pyqtgraph histogram and controlled through the same state as the range slider
and numeric fields.

Contrast-limit synchronization is bi-directional:

- dragging either vertical histogram line updates `layer.contrast_limits`;
- changing the range slider or numeric fields updates `layer.contrast_limits`
  and moves the vertical lines;
- changing napari's own layer contrast controls emits
  `layer.events.contrast_limits`, which updates the vertical lines, range
  slider, and numeric fields;
- updates must guard against feedback loops while preserving one authoritative
  contrast-limit pair.

Use distinct styling for these two contrast-limit lines so they are visually
different from percentile guide lines.

## Percentile Policy

`percentile_min` and `percentile_max` are visual guide settings.

When provided, compute the requested percentile values from the same filtered
Dask array used for the histogram and draw them as labeled vertical lines. Do
not automatically change napari contrast limits when percentile values are
typed or recalculated.

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
- resolve a `HistogramTarget` against `SpatialData`;
- accept one explicit `HistogramSettings` object per calculation;
- select the requested multiscale `scale`, defaulting to exact `scale0`;
- select the requested channel, or scalar image data when no channel axis is
  present;
- flatten values;
- apply `exclude_nan` and `exclude_zeros`;
- call `compute_chunk_sizes()` when filtering creates unknown Dask chunks;
- compute auto range, counts, bin edges, and requested percentile values;
- record the resolved scale used for the result;
- return a structured `HistogramResult`.

Tests:

- default settings compute a valid histogram without requiring optional inputs;
- counts and bin edges match expected values for small Dask-backed images;
- NaN and zero filtering match the documented semantics;
- unknown chunk sizes after filtering are handled;
- channel names and channel indices resolve consistently;
- scalar images without a channel axis are supported;
- invalid percentile values outside `[0, 100]` are rejected.

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
- each card exposes `scale`, `bins`, `density`, `exclude_nan`,
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
- `scale`, `bins`, filtering, log-axis, density, and percentile controls are
  independent per card;
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

- synchronize histogram lower/upper contrast controls with matching napari
  image layers and the movable histogram contrast region.

Scope:

- resolve matching `ImageLayerBinding` for each card after calculation and on
  viewer layer lifecycle changes;
- add a `QLabeledDoubleRangeSlider` plus numeric lower/upper fields;
- initialize controls from `layer.contrast_limits`;
- draw the contrast limits with `pyqtgraph.LinearRegionItem` by default,
  styled to read as two movable vertical contrast-limit lines with a subtle
  selected-region fill;
- moving either boundary of the contrast region updates
  `layer.contrast_limits`;
- slider or numeric-field edits update `layer.contrast_limits` and move the
  contrast region;
- `layer.events.contrast_limits` updates the controls and contrast region when
  napari's own contrast controls change;
- prevent event feedback loops and keep lower/upper ordering valid while users
  drag markers;
- layer removal disables contrast controls without clearing the histogram;
- RGB(A) layers disable contrast controls with a clear status.

Tests:

- overlay channel target resolves the correct overlay layer;
- stack target resolves the stack layer;
- slider/numeric edits assign `layer.contrast_limits`;
- moving the lower/upper histogram lines assigns `layer.contrast_limits`;
- external napari contrast changes update the widget;
- external napari contrast changes move the histogram contrast region;
- contrast-limit line styling remains distinct from percentile styling;
- layer removal clears the sync binding;
- RGB layers disable contrast controls.

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
- the normal contrast-limit event path updates the slider and plot lines.

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
- no histogram result caching layer;
- no CSV export;
- no overlaying multiple target histograms into one shared plot.
