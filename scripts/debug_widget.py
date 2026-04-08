from __future__ import annotations

from pathlib import Path

import napari
from napari_spatialdata import Interactive
from spatialdata import read_zarr

SDATA_PATH = Path("/Users/arne.defauw/VIB/DATA/test_data/sdata_blobs.zarr")
# SDATA_PATH = Path("/Users/arne.defauw/VIB/DATA/test_data/sdata_xenium.zarr")


def main() -> None:
    """Main debug script"""
    sdata = read_zarr(SDATA_PATH)
    interactive = Interactive(sdata, headless=True)
    viewer = interactive._viewer
    viewer.window.add_plugin_dock_widget(plugin_name="napari-harpy", widget_name="Object Classifier")
    napari.run()


if __name__ == "__main__":
    main()
