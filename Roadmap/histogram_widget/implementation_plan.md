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
- the plot should not duplicate empty/stale/running text inside the plotting
  area; card state text belongs in the status card.

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
| Empty/stale/running plot messages | Do not render in the plot; use the card status card |

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
    channel_name: str

    def __post_init__(self) -> None:
        if not isinstance(self.channel_name, str) or not self.channel_name.strip():
            raise ValueError("Histogram target requires an explicit channel name.")


@dataclass(frozen=True)
class HistogramSettings:
    bins: int = 256
    value_range: tuple[float, float] | None = None
    density: bool = False
    exclude_nan: bool = True
    exclude_zeros: bool = False
    log_y: bool = False
    scale: str | None = None
    percentiles: tuple[float, ...] = ()


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

The UI can still expose `percentile_min` and `percentile_max` fields for the
common lower/upper guide-line workflow, but the core calculator receives them
as `HistogramSettings.percentiles`.

Histogram targets require an explicit channel name. Images without a channel
axis are not supported by the first histogram widget implementation.

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

Status: [x] Implemented

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
      channel_name: str

      def __post_init__(self) -> None:
          if not isinstance(self.channel_name, str) or not self.channel_name.strip():
              raise ValueError("Histogram target requires an explicit channel name.")


  @dataclass(frozen=True)
  class HistogramSettings:
      bins: int = 256
      value_range: tuple[float, float] | None = None
      density: bool = False
      exclude_nan: bool = True
      exclude_zeros: bool = False
      log_y: bool = False
      scale: str | None = None
      percentiles: tuple[float, ...] = ()

      def __post_init__(self) -> None:
          if isinstance(self.bins, bool) or not isinstance(self.bins, int) or self.bins <= 0:
              raise ValueError("Histogram settings require `bins` to be a positive integer.")

          if self.value_range is not None:
              low, high = self.value_range
              if not np.isfinite(low) or not np.isfinite(high):
                  raise ValueError("Histogram settings require `value_range` bounds to be finite.")
              if low >= high:
                  raise ValueError("Histogram settings require `value_range` to satisfy low < high.")

          percentiles: list[float] = []
          for percentile in self.percentiles:
              percentile_value = float(percentile)
              if not np.isfinite(percentile_value) or percentile_value < 0 or percentile_value > 100:
                  raise ValueError("Histogram percentile values must be finite values in [0, 100].")
              if percentile_value not in percentiles:
                  percentiles.append(percentile_value)
          object.__setattr__(self, "percentiles", tuple(percentiles))
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
- validate that the selected image has a channel axis;
- validate that the requested channel exists for the selected image;
- accept one explicit `HistogramSettings` object per calculation;
- validate `settings.bins`, `settings.value_range`, and requested percentile
  values in `HistogramSettings.__post_init__`;
- select the requested multiscale `scale`, defaulting to exact `scale0`;
- record the resolved scale used for the result;
- select the requested channel with `array.sel(c=channel_value)`;
- flatten values;
- apply `exclude_nan` and `exclude_zeros`;
- call `compute_chunk_sizes()` when filtering creates unknown Dask chunks;
- compute requested percentile values after NaN/zero filtering and before
  applying `value_range`, matching Harpy's behavior;
- use Dask's tdigest percentile path:

  ```python
  percentile_values = dask.array.percentile(
      array,
      q=list(settings.percentiles),
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

- calculate a histogram from the existing `sdata_blobs` fixture and compare the
  result with direct Dask histogram output;
- verify that `value_range` restricts histogram counts/bin edges while
  percentiles remain computed from the filtered data before range clipping;
- verify that a requested multiscale `scale` is used and recorded;
- keep only focused dataclass validation tests for basic invalid inputs.

### 2. Widget Shell And Target Cards

Status: [x] Implemented

Goal:

- add the `Image Histogram` widget shell with explicit target-card selection,
  per-card settings, and a request boundary for later calculation slices.

Scope:

- add `src/napari_harpy/widgets/histogram/`;
- add `src/napari_harpy/widgets/histogram/widget.py`;
- add `src/napari_harpy/widgets/histogram/__init__.py`;
- register the widget in `src/napari_harpy/napari.yaml`;
- expose the widget as `Image Histogram`;
- expose the widget through lazy widget exports and the programmatic
  `Interactive(..., widgets="all")` launcher;
- attach to shared `HarpyAppState` through `get_or_create_app_state(...)`;
- follow the existing widget surface conventions from shared styles:
  `apply_widget_surface(...)`, `apply_scroll_content_surface(...)`,
  `CompactComboBox`, `create_form_label(...)`, and the shared button/input
  styles;
- render a scrollable widget body with a short header, an Add histogram action,
  and an empty state when no cards exist;
- allow users to add and remove as many target cards as they need;
- each card header exposes a compact trash/delete icon button whose tooltip and
  accessible name are `Remove histogram`;
- the remove action deletes only the histogram card and card-local staged UI
  state; it must never mutate `SpatialData` image data;
- removing the last card restores the widget empty state;
- each card owns an internal `card_id` that remains stable until the card is
  removed;
- each card stages incomplete UI state separately from core dataclasses because
  `HistogramTarget` requires a complete `coordinate_system`, `image_name`, and
  `channel_name`;
- use a local widget/request value object such as:

  ```python
  @dataclass(frozen=True)
  class HistogramCalculationRequest:
      card_id: str
      target: HistogramTarget
      settings: HistogramSettings
  ```

- expose a Qt signal such as `calculation_requested` carrying
  `HistogramCalculationRequest`;
- clicking Calculate builds and emits `HistogramCalculationRequest` for that
  card when the staged target and settings are valid. This is a Slice 2 scaffold
  only; Slice 3 replaces it with a controller-owned calculation path;
- do not call `calculate_histogram(...)` in Slice 2;
- do not start a worker in Slice 2;
- do not create `HistogramResult` objects in Slice 2.

Target-card controls:

- coordinate system selector;
- image selector;
- channel selector;
- Calculate button;
- trash/delete icon button in the card header for `Remove histogram`;
- card-local status text for empty, incomplete, valid, and invalid states.

Selector behavior:

- coordinate systems come from `get_coordinate_system_names_from_sdata(...)`;
- for a selected coordinate system, image options come from
  `get_spatialdata_image_options_for_coordinate_system_from_sdata(...)`;
- for a selected image, channel names come from
  `get_image_channel_names_from_sdata(...)`;
- images without channel names are unsupported for the first histogram widget
  implementation and should keep the card invalid with a clear status;
- new cards should seed the coordinate-system selector from
  `HarpyAppState.coordinate_system` when it is available, otherwise leave the
  target incomplete;
- image and channel selection must remain explicit and visible;
- on `sdata_changed`, refresh selector options, preserve existing selections
  when they are still valid, and otherwise clear only the invalid downstream
  selections for that card;
- changing coordinate system clears invalid image/channel selections;
- changing image clears invalid channel selection;
- changing target or settings marks that card dirty relative to the last emitted
  request in the Slice 2 scaffold; Slice 3 replaces this with controller binding
  invalidation.

Settings UI:

- each card exposes `scale`, `bins`, `value_range`, `density`, `exclude_nan`,
  `exclude_zeros`, `log_y`, and percentile guide fields inside a collapsed
  `Histogram settings` disclosure section;
- the disclosure is collapsed by default;
- use the title `Histogram settings`, not a generic `Advanced` label;
- the collapsed header summarizes the active settings, for example
  `scale0, 256 bins` or `scale1, range 0-4095, 512 bins, p0.1-p99.9`;
- the settings section exposes a per-card `Reset settings` action;
- settings are independent per card;
- `scale` defaults to `scale0`;
- `bins` defaults to `256`;
- `value_range` is optional and maps to `HistogramSettings.value_range` only
  when both low/high fields are provided;
- percentile UI can expose `percentile_min` and `percentile_max` fields for the
  common lower/upper workflow, but the card must translate non-empty fields to
  `HistogramSettings.percentiles`;
- invalid optional settings should keep Calculate disabled and show a
  card-local warning instead of constructing invalid core dataclasses.

Non-goals:

- no `calculate_histogram(...)` calls;
- no Dask computation;
- no background worker/controller;
- no pyqtgraph plot widget;
- no histogram result rendering;
- no contrast-limit synchronization;
- no percentile-to-contrast action;
- no automatic active-layer retargeting;
- no confirmation dialog for removing a histogram card, because Slice 2 removal
  is UI-local and non-destructive.

Tests:

- widget instantiates without a viewer;
- widget attaches to shared `HarpyAppState`;
- widget shows an empty state and Add histogram action when no cards exist;
- cards can be added and removed;
- each card exposes a trash/delete icon button with tooltip/accessibility text
  `Remove histogram`;
- removing a card does not mutate the selected `SpatialData`;
- removing the last card restores the empty state;
- coordinate-system/image/channel selectors refresh on `sdata_changed`;
- coordinate-system/image/channel selectors preserve valid selections across
  refreshes and clear invalid downstream selections;
- images without channel names keep the card invalid;
- Calculate is enabled only when the staged target and settings can build
  `HistogramTarget` and `HistogramSettings`;
- clicking Calculate emits `HistogramCalculationRequest` in the Slice 2 scaffold;
  Slice 3 replaces this with `controller.bind(...)` plus
  `controller.calculate(card_id)`;
- clicking Calculate does not call `calculate_histogram(...)`;
- optional settings are hidden in a collapsed per-card settings section by
  default;
- the collapsed settings header summarizes active defaults/overrides;
- resetting settings affects only the current card;
- `scale`, `bins`, histogram value range, filtering, log-axis, density, and
  percentile controls are independent per card;
- percentile min/max UI fields are translated to
  `HistogramSettings.percentiles`;
- changing target/settings marks the card dirty relative to the last emitted
  request.

### 3. Background Controller

Status: [x] Implemented

Goal:

- run histogram jobs without blocking the Qt main thread.

Scope:

- add `widgets/histogram/controller.py`;
- keep the responsibility split explicit:
  - the widget resolves staged card UI into structured card state
    (`HistogramTarget`, `HistogramSettings`) or a validation error;
  - the controller receives structured card state, creates the final calculation
    request, and owns job creation, worker lifecycle, stale-result handling, and
    job status;
- define immutable `HistogramJob` exactly as:

  ```python
  @dataclass(frozen=True)
  class HistogramJob:
      card_id: str
      job_id: str
      sdata: SpatialData
      target: HistogramTarget
      settings: HistogramSettings
  ```

- define a small immutable result envelope for worker completion, for example:

  ```python
  @dataclass(frozen=True)
  class HistogramJobResult:
      card_id: str
      job_id: str
      target: HistogramTarget
      settings: HistogramSettings
      result: HistogramResult
  ```

- use the same `thread_worker` resolution pattern as feature extraction;
- replace the Slice 2 widget-level `calculation_requested` scaffold with a
  controller-owned calculation path: clicking a card's Calculate button should
  bind the resolved card state and call the histogram controller, rather than
  emitting a public request signal as the long-term calculation mechanism;
- introduce an explicit card state-resolution step, e.g.
  `_resolve_card_binding(card_id)`, that parses the current card controls and
  returns `HistogramTarget | None`, `HistogramSettings | None`, and a validation
  error string;
- card state resolution is allowed to construct `HistogramTarget` and
  `HistogramSettings`, because those dataclasses are the validation boundary
  between raw Qt controls and structured domain state;
- the selected `SpatialData` object should be passed to the controller as
  execution context during binding;
- use a bind-then-calculate controller API, matching the feature extraction
  controller while adapting it to histogram's per-card model:

  ```python
  controller.bind(
      card_id: str,
      sdata: SpatialData | None,
      target: HistogramTarget | None,
      settings: HistogramSettings | None,
      validation_error: str | None = None,
  ) -> bool
  controller.calculate(card_id: str) -> bool
  controller.invalidate_card(card_id: str) -> None
  controller.remove_card(card_id: str) -> None
  ```

- `controller.bind(...)` stores the latest resolved `SpatialData` and
  structured card state for that card; if `sdata`, `target`, or `settings` is
  `None`, or if `validation_error` is set, the card is known to the controller but
  cannot calculate;
- `controller.bind(...)` is not responsible for reading Qt controls. It
  synchronizes the controller with the latest structured card state and
  invalidates stale work;
- `controller.bind(...)` should invalidate any in-flight job for the card when
  the bound `SpatialData` object, target, settings, or validation error changes;
- `controller.calculate(card_id)` uses the currently bound state for that card,
  creates a `HistogramJob`, launches a worker, and calls
  `calculate_histogram(job.sdata, job.target, job.settings)` inside that worker;
- do not build calculation request payloads directly inside the widget, especially
  inside
  `_update_card_state(...)`; target/settings change handlers should refresh a
  resolved card state, and status rendering should consume that state instead of
  performing request construction itself;
- after the controller is introduced, widget-level notifications should be for
  completed results/status updates only if another widget-level consumer needs
  them; calculation launch itself should live in the controller, as in the
  feature extraction widget;
- track the latest requested job per card;
- when target/settings change after a card has launched a job, the widget should
  mark that card's resolved state as changed and call `controller.bind(...)` with
  the latest structured card state; `invalidate_card(card_id)` remains available
  for explicit non-binding invalidation events;
- ignore stale results when a newer job was requested, when the card was
  invalidated, or when the card was removed;
- support one active worker per card and do not introduce a per-card job queue in
  this slice;
- allow different cards to calculate concurrently;
- expose per-card controller status text and status kind for card-level feedback;
- cancel or ignore jobs when a card is removed.

Widget card-resolution behavior:

- `_resolve_card_binding(card_id)` should be the only place where the widget
  translates combo-box/text-field state into `HistogramTarget` and
  `HistogramSettings`;
- missing `SpatialData`, coordinate system, image, channel, or invalid optional
  settings should return a validation error with user-facing status text;
- `_update_card_state(...)` should not call `calculate_histogram(...)`, start a
  worker, emit a calculation signal, or create a `HistogramJob`;
- the Calculate button should be enabled only when the resolved state is valid
  and the card is not already running;
- clicking Calculate should re-read the latest resolved target/settings for that
  card, bind the selected `SpatialData` plus that state into the controller, and
  call `controller.calculate(card_id)`.

Controller behavior:

- the controller must not import Qt widgets or pyqtgraph;
- the controller stores per-card bound state, latest requested job id, active
  worker, status, and last successful `HistogramResult`;
- the worker must call the Slice 1 calculator and return `HistogramJobResult`;
- worker errors should become card-local error status and should not crash the
  widget;
- stale results should not update the card result state or status as completed;
- a running card should show a running status until the current job returns,
  errors, is invalidated, or is removed.

Non-goals:

- no pyqtgraph plot widget;
- no histogram bar rendering;
- no contrast-limit synchronization;
- no percentile-to-contrast action;
- no automatic active-layer calculation action.

Tests:

- Calculate starts a worker for a valid card;
- clicking Calculate calls the histogram controller for the selected card rather
  than emitting the Slice 2 `calculation_requested` signal;
- widget binds the selected `SpatialData`, target, settings, and validation error
  before calling `controller.calculate(card_id)`;
- controller creates `HistogramJob` directly from the currently bound card state;
- controller `calculate(card_id)` uses the currently bound state for that card;
- widget card resolution returns validation errors for incomplete or invalid
  card UI without calling `controller.calculate(card_id)`;
- card status updates do not construct jobs and do not start workers;
- invalid card UI resolves to a validation error that disables Calculate and
  renders card feedback;
- worker calls `calculate_histogram(...)` with the exact `sdata`, target, and
  settings from the selected widget state and resolved card state;
- worker jobs receive the settings from the card that launched them;
- stale worker results are ignored;
- worker errors surface as card errors;
- changing target/settings while a job is running invalidates the in-flight job
  for that card through `controller.bind(...)`;
- removing a card disconnects or ignores its worker result;
- different cards can calculate independently without sharing settings or job
  state.

### 4. Pyqtgraph Dependency

Status: [x] Implemented

Goal:

- add `pyqtgraph` as the single supported plotting backend dependency for the
  histogram widget.

Scope:

- add `pyqtgraph` to `pyproject.toml` as a direct runtime dependency;
- keep this slice limited to dependency declaration and packaging metadata;
- do not introduce a Matplotlib fallback or optional plotting backend;
- do not add plot widget code in this slice;
- do not change histogram widget runtime behavior in this slice.

Tests:

- project metadata includes `pyqtgraph` as a runtime dependency;
- the package metadata/build check still succeeds after the dependency change.

### 5. Histogram Plot Rendering

Status: [x] Implemented

Goal:

- draw each calculated histogram inside its card with pyqtgraph, using a small
  plot wrapper that keeps rendering concerns separate from the widget controller
  and the core calculator.

Scope:

- use the `pyqtgraph` runtime dependency introduced in Slice 4;
- add `widgets/histogram/plot_widget.py` containing a private
  `_HistogramPlotWidget`;
- embed one `_HistogramPlotWidget` per histogram card, positioned below the
  Calculate action and above the card status feedback;
- keep the plot area horizontally expanding and vertically stable, with a
  compact product-appropriate height so multiple histogram cards remain
  scannable;
- keep all pyqtgraph imports inside the plot wrapper module; `core.histogram`,
  `widgets/histogram/controller.py`, and the Slice 1 calculator must remain free
  of pyqtgraph dependencies;
- add histogram plot palette/style constants in `widgets/histogram/styles.py`,
  derived from `widgets/shared_styles.py`;
- use the palette decisions from "Plot Palette And Styling"; do not introduce
  ad hoc plot colors or a Matplotlib-style color cycle;
- configure plot background, axes, grid, bar fill, and future marker styling
  from those constants;
- use the following initial palette contract:

  ```python
  HISTOGRAM_PLOT_BACKGROUND_COLOR = WIDGET_PANEL_SUBTLE_COLOR
  HISTOGRAM_AXIS_TEXT_COLOR = WIDGET_TEXT_MUTED_COLOR
  HISTOGRAM_AXIS_GRID_COLOR = WIDGET_BORDER_COLOR
  HISTOGRAM_BAR_FILL_COLOR = "#7EA7FF"
  HISTOGRAM_BAR_FILL_ALPHA = 150
  HISTOGRAM_BAR_EDGE_COLOR = None

  # Defined now so later contrast/percentile slices do not invent new colors.
  HISTOGRAM_CONTRAST_LINE_COLOR = WIDGET_SUCCESS_COLOR
  HISTOGRAM_CONTRAST_REGION_ALPHA = 32
  HISTOGRAM_PERCENTILE_LINE_COLOR = WIDGET_WARNING_TEXT_COLOR
  ```

- use the stable muted data blue for all histogram bars in the first rendering
  implementation; do not vary bar color per card or per channel in Slice 5;
- use `pyqtgraph.PlotWidget` with one `PlotItem` by default;
- render histogram bars from `HistogramResult.counts` and
  `HistogramResult.bin_edges` with `pyqtgraph.BarGraphItem`;
- derive bar centers and widths from the bin edges:

  ```python
  centers = (bin_edges[:-1] + bin_edges[1:]) / 2
  widths = np.diff(bin_edges)
  heights = counts
  ```

- keep `value_range` behavior purely data-driven in this slice: the plot renders
  the result returned by the calculator and does not reinterpret the requested
  range;
- label the x-axis as intensity and the y-axis as count or density based on
  `HistogramResult.settings.density`;
- support `HistogramSettings.log_y` by using pyqtgraph's y-axis log mode; do not
  mutate counts, add epsilons, or change the calculator result to make log
  rendering work;
- clear and replace the previous bar item when a new result is rendered, rather
  than accumulating plot items across repeated calculations;
- keep the plot area free of card-status text before calculation, while running,
  and after target/settings changes; the status card remains the only textual
  state surface;
- preserve existing bars while recalculating an unchanged target and clear bars
  when target/settings changes invalidate the previous result;
- when the controller reports a successful result, the widget reads
  `controller.result_for_card(card_id)` and calls the card plot widget's
  `set_histogram(...)`;
- when the controller reports no current result, the widget updates the plot
  state from the controller status instead of leaving stale bars visible;
- redraw from the current `HistogramResult` without re-running the calculator
  when only plot-local display state changes.

The plot wrapper API should stay data-oriented. A suitable first implementation
shape is:

```python
class _HistogramPlotWidget(QWidget):
    def set_histogram(self, result: HistogramResult) -> None: ...
    def clear_histogram(self, message: str = "No histogram calculated.") -> None: ...
    def show_running(self, message: str = "Calculating histogram.") -> None: ...
    def show_stale(self, message: str = "Target or settings changed.") -> None: ...
    def set_log_y(self, enabled: bool) -> None: ...
    def set_contrast_limits(self, limits: tuple[float, float] | None) -> None: ...
    def set_percentile_markers(self, markers: Mapping[float, float]) -> None: ...
```

For Slice 5, `set_contrast_limits(...)` and `set_percentile_markers(...)` may be
implemented as no-op placeholders or simple item-management hooks if that keeps
the plot wrapper API stable. The actual user-facing contrast synchronization and
percentile guide-line behavior belongs to later slices.

Non-goals:

- no napari image-layer matching;
- no synchronization with `layer.contrast_limits`;
- no draggable contrast region behavior;
- no percentile-to-contrast action;
- no automatic active-layer histogram calculation;
- no Matplotlib or fallback plotting backend.

Tests:

- `_HistogramPlotWidget.set_histogram(...)` creates histogram bars from
  `counts` and `bin_edges`;
- bar centers and widths are derived from the bin edges rather than assumed from
  the number of bins alone;
- repeated `set_histogram(...)` calls replace the previous bars instead of
  accumulating duplicate plot items;
- log y-axis setting is applied from `HistogramSettings.log_y`;
- y-axis label switches between count and density;
- empty, running, and stale plot states do not render duplicate plot messages;
- stale target/settings changes do not leave stale bars behind;
- widget/controller integration renders `controller.result_for_card(card_id)`
  into the matching card when a job succeeds;
- target/settings changes clear or stale-mark the card plot after the controller
  invalidates the previous result;
- plot colors/styles come from histogram palette constants;
- `core.histogram` and `widgets/histogram/controller.py` do not import
  pyqtgraph.

### 6. Smooth Distribution Overlay

Status: [ ] Planned

Goal:

- draw a smooth distribution approximation over the histogram bars without
  asking users to tune smoothing parameters.

Scope:

- reuse a library implementation rather than writing a custom smoothing
  algorithm;
- use `scipy.stats.gaussian_kde` with its automatic bandwidth selection by
  default;
- add `scipy` as a direct runtime dependency if it is not already direct by the
  time this slice is implemented, because the histogram widget will import it
  directly;
- derive the smooth line from the already calculated histogram result, not by
  re-reading or resampling the source image data;
- use bin centers as the KDE sample positions and histogram counts as weights;
- evaluate the KDE on a compact regular grid spanning the histogram bin edges;
- scale the KDE line so it overlays the current y-axis semantics:
  - for count histograms, scale the density by total count and bin width;
  - for density histograms, keep the KDE in density units;
- keep smoothing enabled by default if visual QA shows it improves readability;
- expose only a simple card-local display toggle such as `Smooth line` if users
  need to hide it; do not expose bandwidth, sigma, kernel, or sample-count
  controls in the first implementation;
- style the smooth line with a stable histogram palette constant, for example
  `HISTOGRAM_SMOOTH_LINE_COLOR`, visually related to but distinct from the bar
  fill color;
- handle small, empty, flat, or otherwise invalid histogram results gracefully by
  hiding the smooth line rather than showing an error.

Non-goals:

- no user-facing bandwidth or smoothing-strength setting;
- no additional Dask computation;
- no KDE calculation from raw image pixels;
- no ECDF/cumulative distribution calculation.

Tests:

- smooth line is derived from `HistogramResult.counts` and `bin_edges`;
- smooth line uses `scipy.stats.gaussian_kde` with weighted bin centers;
- count-mode and density-mode overlays use the correct y-axis scaling;
- empty, flat, or too-small histograms do not crash and hide the smooth line;
- repeated histogram rendering replaces the previous smooth line instead of
  accumulating duplicate plot items;
- disabling the display toggle hides the smooth line without recalculating the
  histogram.

### 7. Contrast-Limit Synchronization

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

### 8. Percentile Guide Lines

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

### 9. Explicit Percentile-To-Contrast Action

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

### 10. Add Histogram From Active Image

Status: [ ] Planned

Goal:

- provide a fast, explicit path from the currently selected napari image layer
  to a histogram card without making active-layer following the canonical
  selection model.

Scope:

- add a header action next to `Add histogram`, labeled `Add from active image`
  or `Add histogram from active image`;
- keep explicit cards as the source of truth; the active layer only pre-fills a
  new card or focuses an equivalent existing card;
- do not silently retarget existing histogram cards when the napari active layer
  changes;
- resolve the active layer through `HarpyAppState.viewer_adapter` and
  `LayerBindingRegistry`, not through layer names or legacy metadata;
- enable the action only when the active layer is a Harpy-managed napari
  `Image` layer for the currently loaded `SpatialData`;
- for overlay layers, use `ImageLayerBinding.coordinate_system`,
  `element_name`, and `channel_name` to create a complete histogram target;
- for stack layers, use `ImageLayerBinding.coordinate_system` and
  `element_name`, then resolve the active channel from napari dims only if that
  mapping is reliable; otherwise create the card with image/coordinate system
  prefilled and require the user to choose the channel explicitly;
- preserve the card's default histogram settings unless a later product decision
  adds a separate "copy settings from selected card" action;
- after creating the card, scroll/focus it and leave calculation explicit via
  that card's `Calculate` button;
- if no resolvable active image layer exists, keep the action disabled or show a
  clear widget-level status such as `Select a Harpy image layer in the viewer.`;
- after Slice 3 is implemented, consider a separate `Add and calculate from
  active image` action only if the two-click flow proves too slow in practice.

Tests:

- action is disabled or reports a clear status when no viewer, no active layer,
  or a non-image active layer is present;
- unregistered external napari image layers are ignored;
- active overlay image layer creates a card with coordinate system, image, and
  channel selected from `ImageLayerBinding`;
- active stack image layer creates a card with coordinate system and image
  selected, and either resolves the active channel safely or leaves channel
  selection explicit;
- clicking the action does not call `calculate_histogram(...)`;
- clicking the action does not mutate existing histogram cards;
- repeated clicks either add separate cards intentionally or focus an existing
  equivalent card, whichever behavior is chosen for implementation.

### 11. Automatic Bin Suggestion

Status: [ ] Planned

Goal:

- offer a sensible per-histogram bin-count suggestion without relying on
  unsupported `dask.array.histogram(..., bins="auto")` behavior.

Scope:

- decide the exact UX before implementation, including whether the control is a
  button, menu option, inline `Auto` action, or some other explicit per-card
  affordance;
- keep `HistogramSettings.bins` as an explicit positive integer passed to
  `dask.array.histogram(...)`;
- do not pass string estimators such as `"auto"`, `"fd"`, or `"sturges"` to
  Dask, because Dask's histogram API supports integer bin counts or explicit
  bin edges, not NumPy-style string bin estimators;
- add an optional UI action such as `Suggest bins` or `Auto` beside the bins
  spin box;
- calculate a suggested integer bin count from the selected target and current
  filtering/range settings;
- make the suggestion explicit by writing the resulting integer into the card's
  bins control, so the calculation remains reproducible and visible;
- prefer a bounded, robust estimator suitable for large image arrays, for
  example an approximate Freedman-Diaconis-style estimate based on Dask
  percentiles, optionally with sampling if full percentile calculation proves
  too expensive;
- clamp the suggested value to a product-defined range, for example
  `32 <= bins <= 2048`, to avoid unreadable plots and unexpectedly expensive
  calculations;
- treat `HistogramSettings.value_range` as the range being inspected when it is
  provided; otherwise derive the range from the filtered data as in the core
  calculator;
- keep the existing default of `256` bins until the user explicitly asks for a
  suggestion;
- show a clear warning if a robust suggestion cannot be computed, for example
  because the filtered data has too few finite values or near-zero spread.

Tests:

- default histogram cards still start with `256` bins;
- clicking the suggestion action writes a positive integer into the bins control;
- suggested bins are clamped to the configured min/max bounds;
- the suggestion respects NaN/zero filtering and explicit histogram
  `value_range`;
- flat or nearly flat data falls back to a safe bin count with a clear status;
- the suggestion action does not calculate or render the histogram by itself.

### 12. Product Hardening

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
