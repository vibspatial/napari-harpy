# napari-harpy

`napari-harpy` is a napari plugin for interactive object classification on `SpatialData` datasets.

The current repository contains a working MVP for interactive annotation, background retraining, live prediction updates, and explicit sync back to a backed `SpatialData` store.

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
- explicit sync of table state back to a zarr-backed `SpatialData` store

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

The plugin has completed the current roadmap work through Phase 5 for the non-ROI workflow:

- Phase 1: `SpatialData` discovery and validation
- Phase 2: basic manual annotation workflow
- Phase 3: manual sync of annotation table state back to zarr
- Phase 4: background random forest training
- Phase 5: live prediction updates and viewer recoloring

At this point, you can load a compatible `SpatialData` object through `napari-spatialdata`, choose a segmentation/table pair and feature matrix, annotate objects with user classes, let the plugin retrain in the background, inspect predictions through the `Color by` menu, and persist the current table state with `Sync to zarr`.

The main missing piece from the original MVP plan is ROI support. Future roadmap work focuses on ROI-aware workflows plus reload/persistence hardening.
