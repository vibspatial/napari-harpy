# Histogram Widget Roadmap

## Goal

Add a dedicated napari-harpy widget for quickly visualizing image intensity
histograms and controlling the selected image layer's contrast limits from the
same place.

The intended workflow is:

1. select an image layer or image element
2. choose the channel and histogram settings
3. compute the histogram with Dask
4. inspect the intensity distribution
5. update napari contrast limits directly from the histogram widget

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
- two vertical guide lines over the histogram
- an optional "Reset from layer" or "Auto" action

`superqt` is already available through napari and exposes useful controls:

```python
QLabeledDoubleRangeSlider
QDoubleRangeSlider
```

Those are good candidates for the contrast-limit UI.

## Layer And Selection Strategy

There are two viable selection models.

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

### Option B: Explicit Image And Channel Selectors

The widget provides its own image and channel selectors based on the shared
`SpatialData` object and current coordinate system.

This is more explicit and works even when the image is not loaded yet, but
contrast-limit control requires a live napari `Image` layer. If the selected
image is not loaded, the widget can compute a histogram from `SpatialData`, but
it cannot update a layer until one exists.

### Recommended First Version

Use a hybrid:

- default to the active Harpy-managed image layer when one is selected
- expose a compact selector listing currently loaded Harpy-managed image layers
- show a clear disabled state when there is no compatible loaded image layer
- optionally add SpatialData element selection later

This keeps the first version focused on loaded images and makes contrast-limit
control unambiguous.

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

For an interactive widget, the recommended default is:

- compute a preview histogram from the coarsest or user-selected lower
  resolution scale
- allow full-resolution `scale0` recomputation as an explicit action
- show which scale was used for the current histogram

This avoids surprising UI freezes on large images while preserving access to an
accurate full-resolution histogram.

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

## Plotting Options

Two plotting backends are practical.

### Matplotlib

Pros:

- already available in the environment
- aligns with Harpy's QC plotting code
- easy to draw bars and vertical contrast-limit guide lines

Cons:

- less fluid for draggable interactive markers

### Pyqtgraph

Pros:

- available in the current environment
- better suited to interactive plots and draggable markers
- likely smoother for live contrast-limit manipulation

Cons:

- should be treated as an explicit dependency decision if napari-harpy wants to
  rely on it directly

### Recommended First Version

Use Matplotlib for the first version unless a highly interactive histogram is a
hard requirement.

The lower-risk first implementation is:

- render histogram bars with Matplotlib
- draw two vertical contrast-limit lines
- use a `QLabeledDoubleRangeSlider` and numeric values for interaction
- redraw lines when the slider or layer changes

Pyqtgraph can be revisited if dragging the markers directly on the plot becomes
important.

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

- current selected layer identity
- current channel selection
- bins
- density setting
- exclude-zero setting
- exclude-NaN setting
- selected multiscale level
- worker lifecycle and stale-result handling

The widget should own:

- Qt controls
- plot canvas
- status text
- contrast-limit slider
- event connections to napari layer changes

## Suggested MVP

MVP behavior:

- add a separate `Image Histogram` widget
- list compatible loaded Harpy-managed image layers
- optionally follow the active napari image layer
- compute one histogram for the selected layer or channel
- support `bins`
- support excluding zeros
- support excluding NaNs
- compute range automatically from the filtered Dask array
- draw the histogram
- show current contrast limits
- update `layer.contrast_limits` from a two-handle slider
- listen to `layer.events.contrast_limits` and update the widget when napari
  changes the limits elsewhere
- avoid blocking the UI during histogram computation

Recommended MVP defaults:

- `bins = 256`
- `density = False`
- `exclude_nan = True`
- `exclude_zeros = False` by default for general image display
- log y-axis optional, off by default
- use current layer contrast limits as guide lines, not as the histogram range

## Later Enhancements

Potential follow-ups:

- direct dragging of contrast-limit markers on the histogram
- percentile guide lines, for example p0.1 and p99.9
- one-click contrast presets from percentiles
- log-scaled y-axis
- cumulative distribution view
- histogram caching per image, channel, scale, and settings
- background recomputation debounce
- support for unloaded `SpatialData` image elements
- optional full-resolution recompute for multiscale images
- overlay multiple channels in one histogram plot
- export histogram values as CSV

## Risks And Edge Cases

### Large Images

Full-resolution histogram computation can still be expensive even with Dask.
The widget should default to a responsive scale for multiscale data and keep
work off the main thread.

### Unknown Dask Chunk Sizes

Filtering zeros or NaNs can produce unknown Dask chunk sizes. Call
`compute_chunk_sizes()` when needed before reductions or histogram calculation.

### RGB Images

RGB(A) layers may not support contrast-limit control in the same way as scalar
image layers. The histogram widget should detect `layer.rgb` and disable or
limit contrast controls.

### External Image Layers

The safest first version should target Harpy-managed image layers with
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
- `layer.events.contrast_limits` updates the widget state
- removing the selected layer clears the widget state
- RGB layers disable contrast controls

Widget tests should use the existing dummy viewer patterns from:

- `tests/test_viewer_adapter.py`
- `tests/test_viewer_widget.py`
- `tests/test_points_controller.py`

## Conclusion

The feature is technically well aligned with napari-harpy.

The best first implementation is a dedicated `Image Histogram` widget that works
with loaded Harpy-managed image layers, computes histogram data with Dask in a
background worker, and controls napari `Image.contrast_limits` through a
two-handle range slider synchronized with napari layer events.
