# napari-harpy

`napari-harpy` is a napari plugin for interactive object classification on `SpatialData` datasets.

The current repository contains a working early MVP for manual annotation:

- an installable Python package under `src/napari_harpy`
- an npe2 `napari.yaml` manifest
- a dock widget discoverable from napari
- `SpatialData`-aware segmentation and table discovery from `napari-spatialdata`
- manual object annotation through the widget
- recoloring of segmentation instances from `adata.obs["user_class"]`
- explicit sync of annotation table state back to a zarr-backed `SpatialData` store

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

The plugin has completed the current roadmap work through Phase 3:

- Phase 1: `SpatialData` discovery and validation
- Phase 2: basic manual annotation workflow
- Phase 3: manual sync of annotation table state back to zarr

At this point, you can load a compatible `SpatialData` object through `napari-spatialdata`, choose a segmentation/table pair, annotate objects with user classes, and persist those edits with `Sync to zarr`.

Next roadmap steps focus on reload-from-zarr behavior, background model training, and prediction-driven recoloring.
