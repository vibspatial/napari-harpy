from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import harpy as hp
import numpy as np
import pandas as pd
from shapely.geometry import Point
from spatialdata import read_zarr
from spatialdata.transformations import Identity

from napari_harpy import Interactive

DEFAULT_SDATA_PATH = Path("/Users/arne.defauw/VIB/DATA/test_data/sdata_xenium_3_6_26.zarr")
SOURCE_TABLE_NAME = "table_transcriptomics_preprocessed"
OUTPUT_TABLE_NAME = "table_transcriptomics_preprocessed_shapes"
SHAPES_NAME = "cell_centroids_shapes"
INSTANCE_KEY = "cell_ID"
REGION_KEY = "fov_labels"
RADIUS = 20.0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create or open the Xenium course SpatialData zarr with point-radius "
            "centroid shapes, then launch the napari-harpy Viewer widget."
        )
    )
    parser.add_argument(
        "sdata_path",
        nargs="?",
        type=Path,
        default=DEFAULT_SDATA_PATH,
        help=f"SpatialData zarr path. Defaults to {DEFAULT_SDATA_PATH}.",
    )
    return parser.parse_args()


def _create_xenium_point_radius_shapes_zarr(sdata_path: Path) -> None:
    sdata_path.parent.mkdir(parents=True, exist_ok=True)

    sdata = hp.datasets.xenium_human_ovarian_cancer_course(checkpoint="checkpoint_2")
    sdata.write(sdata_path, overwrite=True)
    sdata = read_zarr(sdata_path)

    table = sdata[SOURCE_TABLE_NAME]
    spatial = np.asarray(table.obsm["spatial"])
    obs_names = table.obs_names.astype(str)

    gdf = gpd.GeoDataFrame(
        {
            # Redundant but allowed: must match the index exactly.
            INSTANCE_KEY: obs_names,
            "radius": np.full(len(obs_names), RADIUS, dtype=float),
        },
        geometry=[Point(float(x), float(y)) for x, y in spatial[:, :2]],
        index=pd.Index(obs_names, name=INSTANCE_KEY),
    )

    sdata = hp.sh.add_shapes(
        sdata,
        input=gdf,
        output_shapes_name=SHAPES_NAME,
        transformations={"global": Identity()},
        instance_key=INSTANCE_KEY,
        overwrite=True,
    )

    adata_shapes = table.copy()
    adata_shapes.obs[INSTANCE_KEY] = obs_names
    adata_shapes.obs[REGION_KEY] = pd.Categorical([SHAPES_NAME] * adata_shapes.n_obs)

    hp.tb.add_table(
        sdata,
        adata=adata_shapes,
        output_table_name=OUTPUT_TABLE_NAME,
        region=[SHAPES_NAME],
        instance_key=INSTANCE_KEY,
        region_key=REGION_KEY,
        overwrite=True,
    )


def main() -> None:
    """Create or open the debug dataset and launch the napari-harpy Viewer."""
    args = _parse_args()
    sdata_path = args.sdata_path.expanduser()

    if not sdata_path.exists():
        print(f"Creating SpatialData zarr at: {sdata_path}")
        _create_xenium_point_radius_shapes_zarr(sdata_path)
    else:
        print(f"Using existing SpatialData zarr: {sdata_path}")

    sdata = read_zarr(sdata_path)
    Interactive(sdata, widgets="viewer")


if __name__ == "__main__":
    main()
