# Histogram Widget Roadmap

## Goal

Add a dedicated napari-harpy widget for visualizing image intensity histograms
and controlling the matching napari image layer's contrast limits from the same
place.

The intended workflow is:

1. add one or more explicit histogram targets
2. for each target choose a coordinate system, image element, and channel
3. choose histogram settings and optional percentile guide values
4. compute the histogram with Dask through an explicit Calculate action
5. inspect the intensity distribution and percentile markers
6. update matching napari contrast limits directly from the histogram widget

This should be a visualization and quality-control tool. It should not mutate
the underlying `SpatialData` image data.

## Feasibility

This is doable with the current napari-harpy architecture.

The repo already has the pieces needed for this:

- widgets attach to a shared per-viewer `HarpyAppState` through
  `get_or_create_app_state(...)`
- `HarpyAppState` owns the shared `ViewerAdapter`
- `ViewerAdapter` owns a `LayerBindingRegistry`
- image layers are registered as `ImageLayerBinding`
- image bindings already store:
  - `sdata_id`
  - image element name
  - coordinate system
  - image display mode, either `stack` or `overlay`
  - overlay channel index and channel name when applicable

The relevant files are:

- `src/napari_harpy/_app_state.py`
- `src/napari_harpy/viewer/adapter.py`
- `src/napari_harpy/viewer/image_styling.py`
- `src/napari_harpy/widgets/viewer/widget.py`
- `src/napari_harpy/widgets/viewer/image_widget.py`

This means a histogram widget can discover Harpy-loaded image layers without
relying on fragile napari layer names or non-authoritative layer metadata.

## Harpy Histogram API

The installed Harpy version in the project environment is `harpy-analysis`
`0.4.1`.

The public QC function is:

```python
hp.qc.image_histogram(...)
```

It is not exported as `harpy.qc.histogram` in this environment.

Internally, Harpy's image histogram implementation lives in:

```text
.venv/lib/python3.13/site-packages/harpy/qc/_qc_image_histogram.py
```

The core histogram calculation follows the expected Dask pattern:

```python
hist, bin_edges = da.histogram(
    array,
    bins=bins,
    range=range,
    density=density,
    **kwargs,
)
hist, bin_edges = dask.compute(hist, bin_edges)
```

Before histogramming, Harpy:

- selects a channel from the image `DataArray`
- flattens the values
- optionally excludes NaN values
- optionally excludes zeros
- computes the range with `da.nanmin(...)` and `da.nanmax(...)` when no range
  is provided

One important implementation detail: filtering with `da.compress(...)` can
produce unknown chunk sizes. Harpy handles this by calling
`compute_chunk_sizes()` when needed. A napari-harpy widget implementation should
preserve that behavior if it computes histogram data directly.

## Contrast Limit Control

Napari image layers expose the APIs needed for two-way contrast-limit sync:

```python
layer.contrast_limits
layer.contrast_limits_range
layer.events.contrast_limits
layer.events.contrast_limits_range
```

Updating the layer can be as direct as:

```python
layer.contrast_limits = (low, high)
```

Napari emits `layer.events.contrast_limits` after changes, so the histogram
widget can stay synchronized when contrast limits are changed elsewhere in
napari.

This is a strong fit for:

- a two-handle range slider
- two numeric fields
- two movable vertical contrast-limit lines over the histogram
- an optional "Reset from layer" or "Auto" action

`superqt` is already available through napari and exposes useful controls:

```python
QLabeledDoubleRangeSlider
QDoubleRangeSlider
```

Those are good candidates for the contrast-limit UI.

## Layer And Selection Strategy

There are three viable selection models.

### Option A: Follow The Active Image Layer

The widget follows the active napari image layer.

This is convenient because the user can click a layer in napari and immediately
see its histogram and contrast controls.

Napari exposes:

```python
viewer.layers.selection.events.active
```

This event exists in the current environment and can be used to follow manual
layer-list selection.

The widget should also listen to `ViewerAdapter.active_layer_changed`, but that
signal only fires when Harpy activates layers through the adapter. It does not
cover every manual layer selection by itself.

Trade-off: active-layer following is convenient but too implicit for a product
workflow that needs coordinate-system awareness, multiple simultaneous
histograms, and repeatable target selection.

### Option B: Explicit Image And Channel Selectors

The widget provides its own image and channel selectors based on the shared
`SpatialData` object and current coordinate system.

This is more explicit and works even when the image is not loaded yet, but
contrast-limit control requires a live napari `Image` layer. If the selected
image is not loaded, the widget can compute a histogram from `SpatialData`, but
it cannot update a layer until one exists.

### Option C: Explicit Histogram Target Cards

The widget owns a list of histogram target cards. Each card represents one
explicit target:

```text
coordinate_system, image_name, channel_name
```

The target card pattern matches the batch-aware feature extraction widget, where
the UI stages explicit triplets before calculation. It also supports multiple
histograms naturally because each card owns one plot, one Calculate action, and
one contrast-limit sync state.

For images without a channel axis, the internal target can store
`channel_name=None` while the UI shows a scalar-image channel placeholder.
Multi-channel images should require an explicit channel selection.

### Recommended Product Choice

Use explicit histogram target cards as the source of truth.

The active napari layer may be used only as a convenience for pre-filling a new
card or focusing an existing card. It should not silently change the target of
an existing histogram card.

This makes the coordinate system explicit, supports as many plotted histograms
as the user adds, and keeps contrast-limit synchronization unambiguous:

- histogram calculation reads from `SpatialData`;
- contrast-limit sync resolves the matching live napari `Image` layer through
  `ViewerAdapter.layer_bindings`;
- if no matching live layer exists, the histogram remains valid but the
  contrast controls are disabled with a clear status message.

## Stack And Overlay Images

The current Viewer widget supports two image display modes:

- `stack`: one napari `Image` layer containing the full image
- `overlay`: one napari `Image` layer per selected channel, using additive
  blending and per-channel color

Histogram behavior should respect these modes.

For stack layers:

- if the layer data has a channel axis, the widget should offer a channel
  selector
- contrast-limit changes apply to the selected stack layer
- napari's stack layer has one contrast-limit pair for the active image layer

For overlay layers:

- each overlay layer already represents one channel
- the histogram can compute from that layer's data directly
- contrast-limit changes apply only to that channel layer
- the image binding gives the source `channel_index` and `channel_name`

For RGB(A) images:

- napari treats contrast limits differently or ignores them
- histogram display may still be useful, but contrast-limit controls should be
  disabled or clearly marked unsupported for RGB layers

## Multiscale Images

Multiscale images need an explicit speed and accuracy policy.

Harpy's `hp.qc.image_histogram(...)` supports a `scale` parameter for
multiscale images:

- `scale=None` computes from `scale0`
- passing a lower-resolution scale gives a faster approximate histogram

For a product widget, the recommended behavior is:

- compute `scale0` by default so the result is exact unless the user chooses a
  different scale
- expose lower-resolution scales as explicit faster options for multiscale
  images
- show which scale was used for the current histogram

This avoids silently presenting approximate histograms as exact results while
still giving users a clear speed/accuracy choice for very large images.

## Async Computation

Histogram computation can trigger Dask work over large arrays, so it should not
run on the Qt main thread.

Existing napari-harpy controllers already use `thread_worker` patterns:

- `src/napari_harpy/widgets/viewer/points_controller.py`
- `src/napari_harpy/widgets/feature_extraction/controller.py`
- `src/napari_harpy/widgets/object_classification/controller.py`

The histogram widget should follow the same pattern:

- create an immutable histogram job
- launch it in a worker thread
- include a monotonically increasing `job_id`
- ignore stale results when the user changes layer, channel, bins, or scale
- keep the UI responsive while Dask computes

## Plotting Backend

Use `pyqtgraph` as the histogram widget plotting backend.

This is an explicit product dependency decision. The histogram widget is an
interactive Qt control surface, not a static report plot, so the implementation
should optimize for responsiveness, native Qt interaction, and long-term
interactive controls.

`pyqtgraph` should be added as a direct napari-harpy dependency in
`pyproject.toml`. Do not implement a Matplotlib fallback path for the widget.

Matplotlib remains relevant as Harpy's QC plotting reference, but it should not
be the widget plot surface.

The implementation should:

- render histogram bars with `pyqtgraph.BarGraphItem` or an equivalent
  pyqtgraph item
- draw lower and upper contrast-limit markers with movable pyqtgraph vertical
  line items
- draw optional percentile guide lines
- use a `QLabeledDoubleRangeSlider` and numeric fields for interaction
- keep the plot markers, range slider, numeric fields, and napari layer
  contrast controls synchronized in both directions

## Proposed Widget Structure

Suggested module layout:

```text
src/napari_harpy/widgets/histogram/
    __init__.py
    widget.py
    controller.py
```

Suggested new napari contribution:

```yaml
- id: napari-harpy.histogram
  title: Open image histogram widget
  python_name: napari_harpy.widgets.histogram.widget:HistogramWidget
```

Suggested display name:

```text
Image Histogram
```

The controller should own:

- current histogram card identities
- current target triplets
- current per-card channel selection
- bins
- density setting
- exclude-zero setting
- exclude-NaN setting
- selected multiscale level
- percentile settings
- worker lifecycle and stale-result handling

The widget should own:

- Qt controls
- plot canvas
- status text
- contrast-limit slider
- event connections to napari layer changes

## Core Product Scope

Core behavior:

- add a separate `Image Histogram` widget
- allow users to add and remove histogram target cards
- for each card, explicitly select coordinate system, image, and channel
- compute one histogram per card through its Calculate button
- compute with sensible defaults as soon as the target is valid
- expose optional per-card `Histogram settings` in a collapsed disclosure
  section
- support per-card `scale`, `bins`, `density`, zero/NaN filtering, log y-axis,
  and percentile settings
- compute range automatically from the filtered Dask array
- draw the histogram inside the card
- draw optional percentile guide lines when percentile settings are provided
- show current contrast limits as two movable vertical lines in the histogram
- update `layer.contrast_limits` when the user moves the histogram lines, the
  two-handle slider, or the numeric fields
- listen to `layer.events.contrast_limits` and update the widget when napari
  changes the limits elsewhere
- avoid blocking the UI during histogram computation

Recommended defaults:

- `scale = scale0` / exact full-resolution data unless the user selects a
  lower-resolution multiscale level
- `bins = 256`
- `density = False`
- `exclude_nan = True`
- `exclude_zeros = False` by default for general image display
- log y-axis optional, off by default
- use current layer contrast limits as synchronized vertical markers, not as
  the histogram range
- percentile guide lines are visual annotations by default; applying
  percentile values to contrast limits should be an explicit user action, not a
  side effect of typing percentile settings

The `Histogram settings` section should be collapsed by default and summarize
the active settings in its header, for example `scale0, 256 bins`. Changing one
card's settings must not affect any other histogram card.

## Planned Follow-Up Decisions

Potential follow-up decisions:

- explicit "apply percentiles to contrast limits" action
- optional contrast presets from percentiles
- cumulative distribution view
- histogram caching per image, channel, scale, and settings
- background recomputation debounce
- optional full-resolution recompute for multiscale images
- overlay multiple channels in one histogram plot
- export histogram values as CSV

## Risks And Edge Cases

### Large Images

Full-resolution histogram computation can still be expensive even with Dask.
The widget should keep work off the main thread, expose explicit lower-resolution
scale options for multiscale data, and show clear running/cancelled/error
states for long calculations.

### Unknown Dask Chunk Sizes

Filtering zeros or NaNs can produce unknown Dask chunk sizes. Call
`compute_chunk_sizes()` when needed before reductions or histogram calculation.

### RGB Images

RGB(A) layers may not support contrast-limit control in the same way as scalar
image layers. The histogram widget should detect `layer.rgb` and disable or
limit contrast controls.

### External Image Layers

The core implementation should target Harpy-managed image layers with
`ImageLayerBinding`. External napari image layers can be considered later.

### Layer Removal

If a selected layer is removed, the widget should disconnect its event handlers,
clear the histogram, and move to an empty or disabled state.

### Contrast Limit Range

Napari maintains `contrast_limits_range` separately from `contrast_limits`.
The widget should use `contrast_limits_range` to configure the slider bounds,
but it should be prepared for the range to expand when setting limits outside
the previous range.

## Testing Strategy

Core tests:

- histogram calculation returns expected counts and bin edges for a small Dask
  image
- zero exclusion works
- NaN exclusion works
- unknown chunks after filtering are handled
- stale worker results are ignored
- selecting a Harpy-managed stack layer binds the correct source
- selecting a Harpy-managed overlay layer binds the correct channel
- contrast slider changes assign `layer.contrast_limits`
- dragging histogram contrast-limit lines assigns `layer.contrast_limits`
- `layer.events.contrast_limits` updates the widget state
- removing the selected layer clears the widget state
- RGB layers disable contrast controls

Widget tests should use the existing dummy viewer patterns from:

- `tests/test_viewer_adapter.py`
- `tests/test_viewer_widget.py`
- `tests/test_points_controller.py`

## Conclusion

The feature is technically well aligned with napari-harpy.

The best implementation is a dedicated `Image Histogram` widget built around
explicit `coordinate_system, image_name, channel_name` target cards. Histogram
data should be computed from `SpatialData` with Dask in background workers, and
contrast-limit controls should synchronize with matching Harpy-managed napari
`Image` layers through `ImageLayerBinding`.
