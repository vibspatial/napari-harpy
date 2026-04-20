from pathlib import Path

import napari
from spatialdata import read_zarr
from spatialdata.models import Image2DModel
from spatialdata.transformations import get_transformation

from napari_harpy import Interactive

SDATA_PATH = Path("/Users/arne.defauw/VIB/DATA/test_data/sdata_blobs.zarr")
COORDINATE_SYSTEM = "global"
LABEL_NAME = "blobs_labels_rotated"
IMAGE_NAME = "blobs_image"
MULTISCALE_IMAGE_NAME = "blobs_multiscale_image"


def _rename_image_channels_to_rgb(sdata, image_name: str) -> None:
    image = sdata.images[image_name]
    transformations = get_transformation(image, get_all=True)
    sdata.images[image_name] = Image2DModel.parse(
        image.data,
        c_coords=["r", "g", "b"],
        transformations=transformations,
    )


def _main() -> None:
    sdata = read_zarr(SDATA_PATH)
    _rename_image_channels_to_rgb(sdata, IMAGE_NAME)

    # headless=True so we can load layers before entering napari.run()
    interactive = Interactive(sdata, headless=True)
    viewer_adapter = interactive.app_state.viewer_adapter

    labels_layer = viewer_adapter.ensure_labels_loaded(
        sdata,
        label_name=LABEL_NAME,
        coordinate_system=COORDINATE_SYSTEM,
    )

    image_layer = viewer_adapter.ensure_image_loaded(
        sdata,
        image_name=IMAGE_NAME,
        coordinate_system=COORDINATE_SYSTEM,
        mode="stack",
    )

    # multiscale_image_layer = viewer_adapter.ensure_image_loaded(
    #    sdata,
    #    image_name=MULTISCALE_IMAGE_NAME,
    #    coordinate_system=COORDINATE_SYSTEM,
    #    mode="stack",
    # )

    viewer_adapter.activate_layer(image_layer)

    print(f"Loaded labels layer: {labels_layer.name}")
    print(f"Loaded image layer: {image_layer.name}, rgb={image_layer.rgb}, multiscale={image_layer.multiscale}")

    napari.run()


if __name__ == "__main__":
    _main()
