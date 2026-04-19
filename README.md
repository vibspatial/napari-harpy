<p align="center">
  <img src="docs/_static/logo.png" alt="Harpy logo" width="200">
</p>

`napari-harpy` is a napari plugin for feature extraction and interactive object
classification on `SpatialData` datasets. It currently builds on
[`napari-spatialdata`](https://github.com/scverse/napari-spatialdata) for
viewer-linked dataset discovery and uses
[Harpy](https://github.com/saeyslab/harpy) for feature calculation.

The current repository contains two working widgets for `SpatialData` tables
loaded through `napari-spatialdata`:

- `Feature Extraction`
- `Object Classification`

Today the plugin supports:

- selecting a segmentation, optional image, compatible coordinate system, and
  linked table from viewer-linked `SpatialData`
- calculating shallow intensity and morphology features through Harpy
- writing feature matrices into the selected `AnnData` table linked to the
  segmentation mask, as `.obsm[feature_key]`, with companion metadata in
  `.uns["feature_matrices"][feature_key]`
- interactive manual annotation of segmentation instances
- background `RandomForestClassifier` retraining on the selected feature
  matrix stored in `.obsm[feature_key]` of the `AnnData` table linked to the
  segmentation mask
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

Open the widgets from the napari plugin menu:

- `Plugins -> napari-harpy -> Feature Extraction`
- `Plugins -> napari-harpy -> Object Classification`

## Debug script

A small local debug script is available at
[`scripts/debug_widget.py`](scripts/debug_widget.py).

It loads a `SpatialData` zarr, opens `napari-spatialdata`, and docks both the
`Feature Extraction` and `Object Classification` widgets automatically.
This is useful for quickly reproducing widget behavior during development.

Run it with:

```bash
source .venv/bin/activate
python scripts/debug_widget.py
```

The script currently contains a hard-coded `SDATA_PATH`, so update that path as needed before running it.

## Status

Current implementation status:

- the `Feature Extraction` widget is available in napari and can write feature
  matrices into existing linked tables
- the `Object Classification` widget remains the working annotation and
  classifier workflow for those tables (and matrices)

Main remaining roadmap work:

- tighter same-session integration between the two widgets
  - shared table-refresh behavior so newly written feature keys become
    available to object classification without any manual rescan/reload step
- the `"no table linked"` branch in the feature-extraction widget
- channel-selection and other advanced feature-extraction controls
- ROI-aware annotation and prediction workflows
- longer-term work to reduce dependence on `napari-spatialdata` for loading and
  session state
