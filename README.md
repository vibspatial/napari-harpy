<p align="center">
  <img src="docs/_static/logo.png" alt="Harpy logo" width="200">
</p>

`napari-harpy` is a napari plugin for viewing, exploring, and analyzing
`SpatialData` datasets. It includes its own viewer for loading and
browsing data inside napari, alongside feature extraction and interactive
object classification workflows.

## Quickstart

The quickest way to try the plugin is to create a small example `SpatialData`
object, write it to a temporary zarr store, read it back as an on-disk dataset,
and launch the Harpy napari interface with `Interactive`.

```python
import tempfile
from pathlib import Path

from spatialdata import read_zarr

from napari_harpy import Interactive
from napari_harpy.datasets import blobs_multi_region

zarr_path = Path(tempfile.mkdtemp()) / "blobs_multi_region.zarr"

sdata = blobs_multi_region()
sdata.write(zarr_path)

sdata = read_zarr(zarr_path)
Interactive(sdata)
```

This opens napari with the Harpy widgets docked and the
`blobs_multi_region` dataset available in the shared viewer state.

The current repository contains three working widgets:

- `Viewer`
- `Feature Extraction`
- `Object Classification`

Today the plugin supports:

- loading and viewing `SpatialData` through the Harpy viewer widget
- selecting a labels element, optional image, compatible coordinate system, and
  linked table from the shared loaded `SpatialData`
- calculating intensity and morphology features through Harpy
- writing feature matrices into the selected `AnnData` table linked to the
  labels element, as `.obsm[feature_key]`, with companion metadata in
  `.uns["feature_matrices"][feature_key]`
- interactive manual annotation of instances in labels elements
- background `RandomForestClassifier` retraining on the selected feature
  matrix stored in `.obsm[feature_key]` of the `AnnData` table linked to the
  labels element
- live prediction updates and labels recoloring
- explicit write-back of in-memory table state to zarr
- explicit reload of on-disk table state back into memory
- multi-sample workflows through multi-region tables and explicit
  labels/image/coordinate-system matching
- headless feature extraction and classifier application for scripted or batch
  processing

Example napari-harpy session:

<p align="center">
  <img src="docs/_static/Screenshot%202026-05-09%20at%2021.31.53.png" alt="napari-harpy example session screenshot" width="900">
</p>

## Headless and Multi-Sample Workflows

For scripted or batch processing, use the public `napari_harpy.headless` module.
It can apply an exported classifier to an existing feature matrix, or compute
the required features before applying the classifier.

The headless APIs accept one labels element or a sequence of labels elements.
For multi-sample data, pass matching labels, image, and coordinate-system
sequences so Harpy can build or apply a shared table-level feature matrix across
the selected samples.

```python
from spatialdata import read_zarr

from napari_harpy import headless

sdata = read_zarr("experiment.zarr")

result = headless.apply_classifier_with_feature_extraction_from_path(
    sdata,
    "classifier.harpy-classifier.joblib",
    table_name="table_multi",
    labels_name=["sample_1_labels", "sample_2_labels"],
    coordinate_system=["sample_1", "sample_2"],
    image_name=["sample_1_image", "sample_2_image"],
)
```

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

It creates a temporary `blobs_multi_region` zarr store, loads it into napari,
and docks the Harpy widgets automatically.
This is useful for quickly reproducing widget behavior during development.

Run it with:

```bash
source .venv/bin/activate
python scripts/debug_widget.py
```
