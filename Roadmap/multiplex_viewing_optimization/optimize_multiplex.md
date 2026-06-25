I investigated without changing repo code.

**Findings**

`napari-harpy` renders multiplex images in two very different ways:

Stack mode creates one napari `Image` layer for the full multiscale image. For `c, y, x` multiplex data, the channel axis stays as a napari dimension slider, so normally only one channel slice is rendered at a time. See [adapter.py](/Users/arne.defauw/VIB/napari_harpy/src/napari_harpy/viewer/adapter.py:1280).

Overlay mode removes the stack layer, then creates one separate napari `Image` layer per selected channel, using `isel(c=channel_index)` for every scale and additive blending. See [adapter.py](/Users/arne.defauw/VIB/napari_harpy/src/napari_harpy/viewer/adapter.py:1335) and [adapter.py](/Users/arne.defauw/VIB/napari_harpy/src/napari_harpy/viewer/adapter.py:1886).

For your image, `morphology_focus_global_ROI1_rechunked` is a dask-backed multiscale `DataTree`:

```text
scale0: (4, 37631, 54089), chunks=(1, 1024, 1024)
scale6: (4, 587, 845), chunks=(1, 587, 845)
channels: DAPI, ATP1A1/CD45/E-Cadherin, 18S, AlphaSMA/Vimentin
```

So selecting all channels in overlay mode means four image layers, four channel reads per visible tile/scale, four textures, and additive blending across all of them. Stack mode is snappier largely because it is one layer and one displayed channel slice.

**Important Dask Finding**

Napari has a dask cache path, and image layers default to `cache=True`. But Harpy currently passes xarray `DataArray` objects, or lists of xarray `DataArray`s, into `napari.layers.Image`. Napari’s dask detector only recognizes raw `dask.array.Array` or lists of those.

I verified this locally:

```text
_is_dask_data(stack xarray list): False
_is_dask_data(overlay xarray list): False
_is_dask_data(stack raw dask list): True
```

Creating a napari `Image` from the current xarray list left napari’s dask context as `nullcontext`; creating it from raw `.data` dask arrays activated napari’s dask cache. That is probably the biggest actionable code-side performance opportunity.

**Tuning Recommendations**

Best next tweaks, in order:

1. Enable napari async slicing before launch:
   ```bash
   NAPARI_ASYNC=1 .venv/bin/python your_script.py
   ```
   This should improve UI responsiveness during pan/zoom/load. It will not reduce total IO, but it moves slicing work off the main UI path.

2. When code changes are allowed, unwrap xarray arrays to raw dask arrays at the final napari layer boundary:
   ```python
   [scale.data for scale in multiscale_xarray_arrays]
   ```
   Keep xarray/SpatialData for metadata and transforms, but hand napari raw dask arrays so napari’s dask cache/fusion optimization actually engages.

3. Keep using `morphology_focus_global_ROI1_rechunked` over the local `_512` and `_4096` variants for this use case. The 1024 spatial chunks plus extra coarse scale look like the better balance for overview and viewport reads.

4. For many-channel multiplex images, avoid “select all channels” overlay as the default. Overlay cost scales roughly with selected channel count. A practical UI improvement would be presets or a small selected subset, not all channels.

A dask cache can help, but in the current Harpy image path napari’s built-in dask cache is effectively bypassed because the data arrive as xarray objects. Async slicing is worth trying immediately; the raw-dask handoff is the more structural fix.