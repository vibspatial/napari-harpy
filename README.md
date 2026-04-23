<p align="center">
  <img src="docs/_static/logo.png" alt="Harpy logo" width="200">
</p>

`napari-harpy` is a napari plugin for viewing, exploring, and analyzing
`SpatialData` datasets. It includes its own viewer for loading and
browsing data inside napari, alongside feature extraction and interactive
object classification workflows.

The current repository contains three working widgets:

- `Viewer`
- `Feature Extraction`
- `Object Classification`

Today the plugin supports:

- loading and viewing `SpatialData` through the Harpy viewer widget
- selecting a segmentation, optional image, compatible coordinate system, and
  linked table from the shared loaded `SpatialData`
- calculating intensity and morphology features through Harpy
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

- `Plugins -> napari-harpy -> Viewer`
- `Plugins -> napari-harpy -> Feature Extraction`
- `Plugins -> napari-harpy -> Object Classification`

## Debug script

A small local debug script is available at
[`scripts/debug_widget.py`](scripts/debug_widget.py).

It loads a `SpatialData` zarr into napari and docks the Harpy widgets
automatically.
This is useful for quickly reproducing widget behavior during development.

Run it with:

```bash
source .venv/bin/activate
python scripts/debug_widget.py
```

The script currently contains a hard-coded `SDATA_PATH`, so update that path as needed before running it.
