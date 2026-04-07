# napari-harpy

`napari-harpy` is a napari plugin for interactive object classification on `SpatialData` datasets.

The current repository contains the Phase 0 plugin skeleton:

- an installable Python package under `src/napari_harpy`
- an npe2 `napari.yaml` manifest
- a minimal dock widget that napari can discover and open

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

## Status

This is only the initial plugin scaffold. The next implementation step is wiring the widget to `napari-spatialdata` and discovering compatible segmentation layers, tables, ROI layers, and `adata.obsm` feature keys.
