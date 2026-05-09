from __future__ import annotations

from pathlib import Path
from tempfile import mkdtemp

from spatialdata import read_zarr

from napari_harpy import Interactive
from napari_harpy.datasets import blobs_multi_region

DATASET_NAME = "blobs_multi_region"


def _write_debug_dataset_to_temp_zarr() -> Path:
    temp_dir = Path(mkdtemp(prefix="napari_harpy_debug_"))
    zarr_path = temp_dir / f"{DATASET_NAME}.zarr"
    sdata = blobs_multi_region()
    sdata.write(zarr_path)
    return zarr_path


def main() -> None:
    """Main debug script"""
    zarr_path = _write_debug_dataset_to_temp_zarr()
    print(f"Wrote debug SpatialData store to: {zarr_path}")

    sdata = read_zarr(zarr_path)
    Interactive(sdata)


if __name__ == "__main__":
    main()
