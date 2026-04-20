from pathlib import Path

import napari
from spatialdata import read_zarr

from napari_harpy import Interactive

SDATA_PATH = Path("/Users/arne.defauw/VIB/DATA/test_data/sdata_blobs.zarr")


def _main() -> None:
    sdata = read_zarr(SDATA_PATH)

    # headless=True so we can load the layer before entering napari.run()
    interactive = Interactive(sdata, headless=True)

    layer = interactive.app_state.viewer_adapter.ensure_labels_loaded(
        sdata,
        label_name="blobs_labels_rotated",
        coordinate_system="global",
    )
    interactive.app_state.viewer_adapter.activate_layer(layer)

    napari.run()


if __name__ == "__main__":
    _main()
