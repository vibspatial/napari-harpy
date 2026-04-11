<p align="center">
  <img src="docs/_static/logo.png" alt="Harpy logo" width="200">
</p>

`napari-harpy` is a napari plugin for interactive object classification on `SpatialData` datasets, with planned integration with [Harpy](https://github.com/saeyslab/harpy) for shallow and deep feature extraction.

The current repository contains a working non-ROI annotation and classification workflow for `SpatialData` tables loaded through `napari-spatialdata`.

Today the plugin supports:

- interactive manual annotation of segmentation instances
- background `RandomForestClassifier` retraining
- live prediction updates and labels recoloring
- explicit write-back of in-memory table state to zarr
- explicit reload of on-disk table state back into memory

Current capabilities include:

- an installable Python package under `src/napari_harpy`
- an npe2 `napari.yaml` manifest
- a dock widget discoverable from napari
- `SpatialData`-aware segmentation and table discovery from `napari-spatialdata`
- validation of linked annotation tables and available feature matrices from `adata.obsm`
- manual object annotation through the widget
- recoloring of segmentation instances from `adata.obs["user_class"]`
- debounced background `RandomForestClassifier` retraining from labeled rows
- live prediction storage in:
  - `adata.obs["pred_class"]`
  - `adata.obs["pred_confidence"]`
  - `adata.uns["classifier_config"]`
- viewer-side `Color by` modes for:
  - `user_class`
  - `pred_class`
  - `pred_confidence`
- explicit `Write Table to zarr` of table state back to a zarr-backed `SpatialData` store
- explicit `Reload Table from zarr` of on-disk table state into the current in-memory table
- dirty-state tracking for in-memory table changes, with a reload decision flow:
  - `Write local edits and reload`
  - `Reload and discard local edits`
  - `Cancel`

Current limitation:

- ROI selection and ROI-restricted annotation/prediction behavior are not implemented yet and remain roadmap work for a later phase.

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

`Plugins -> napari-harpy -> Object Classifier`

## Debug script

A small local debug script is available at [`scripts/debug_widget.py`](scripts/debug_widget.py).

It loads a `SpatialData` zarr, opens `napari-spatialdata`, and docks the `Object Classifier` widget automatically.
This is useful for quickly reproducing widget behavior during development.

Run it with:

```bash
source .venv/bin/activate
python scripts/debug_widget.py
```

The script currently contains a hard-coded `SDATA_PATH`, so update that path as needed before running it.

## Status

The non-ROI workflow is implemented. You can load a compatible `SpatialData` object through `napari-spatialdata`, choose a segmentation/table pair and feature matrix, annotate objects, retrain the classifier in the background, inspect predictions through `Color by`, write table state to zarr, and reload table state back from zarr into memory.

Main remaining roadmap work:

- classifier hardening around reload
- ROI-aware annotation and prediction workflows
- Further integration with [Harpy](https://github.com/saeyslab/harpy) for shallow and deep feature extraction.
