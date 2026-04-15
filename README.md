<p align="center">
  <img src="docs/_static/logo.png" alt="Harpy logo" width="200">
</p>

`napari-harpy` is a napari plugin for interactive object classification on `SpatialData` datasets, with planned integration with [Harpy](https://github.com/saeyslab/harpy) for shallow and deep feature extraction.

The current repository contains a working annotation and classification workflow for `SpatialData` tables loaded through `napari-spatialdata`.

Today the plugin supports:

- interactive manual annotation of segmentation instances
- background `RandomForestClassifier` retraining
- live prediction updates and labels recoloring
- explicit write-back of in-memory table state to zarr
- explicit reload of on-disk table state back into memory

## Local development

Create the development environment:

```bash
./create_env.sh
```

Then launch napari:

```bash
source .venv/bin/activate
napari
```

Open the widget from the napari plugin menu:

`Plugins -> napari-harpy -> Object Classification`

## Debug script

A small local debug script is available at [`scripts/debug_widget.py`](scripts/debug_widget.py).

It loads a `SpatialData` zarr, opens `napari-spatialdata`, and docks the `Object Classification` widget automatically.
This is useful for quickly reproducing widget behavior during development.

Run it with:

```bash
source .venv/bin/activate
python scripts/debug_widget.py
```

The script currently contains a hard-coded `SDATA_PATH`, so update that path as needed before running it.

## Status

Main remaining roadmap work:

- ROI-aware annotation and prediction workflows
- Further integration with [Harpy](https://github.com/saeyslab/harpy) for shallow and deep feature extraction.
