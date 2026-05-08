from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from spatialdata import SpatialData

__all__ = ["blobs_multi_region", "blobs_points_repartitioned"]


def _attach_dummy_feature_matrices(sdata: SpatialData) -> SpatialData:
    table = sdata["table"]
    n_obs = table.n_obs
    rng = np.random.default_rng(seed=0)

    table.obsm["features_1"] = rng.normal(size=(n_obs, 4))
    table.obsm["features_2"] = rng.normal(size=(n_obs, 2))
    return sdata


def blobs_multi_region() -> SpatialData:
    """Return a SpatialData blobs dataset with a duplicated labels region and multi-region table."""
    import anndata as ad
    import harpy as hp
    import pandas as pd
    from spatialdata.datasets import blobs
    from spatialdata.models import TableModel
    from spatialdata.transformations import Identity

    sdata = _attach_dummy_feature_matrices(blobs())

    sdata = hp.im.add_image(
        sdata,
        arr=sdata["blobs_image"].data,
        output_image_name="blobs_image_2",
        transformations={"global_1": Identity()},
        overwrite=True,
    )
    sdata = hp.im.add_labels(
        sdata,
        arr=sdata["blobs_labels"].data,
        output_labels_name="blobs_labels_2",
        transformations={"global_1": Identity()},
        overwrite=True,
    )

    base_table = sdata["table"].copy()
    table_for_labels_2 = base_table.copy()
    table_for_labels_2.obs["region"] = table_for_labels_2.obs["region"].cat.add_categories(["blobs_labels_2"])
    table_for_labels_2.obs["region"] = "blobs_labels_2"
    table_for_labels_2.obs_names = [f"{idx}_blobs_labels_2" for idx in table_for_labels_2.obs_names]

    multi_region_table = ad.concat([base_table, table_for_labels_2], axis=0, merge="same")
    multi_region_table.obs["region"] = pd.Categorical(
        multi_region_table.obs["region"],
        categories=["blobs_labels", "blobs_labels_2"],
    )
    multi_region_table = TableModel.parse(
        multi_region_table,
        region=["blobs_labels", "blobs_labels_2"],
        region_key="region",
        instance_key="instance_id",
        overwrite_metadata=True,
    )

    sdata.tables["table_multi"] = multi_region_table
    return sdata


def blobs_points_repartitioned(
    *,
    n_points: int = 1_000,
    npartitions: int = 4,
    points_key: str = "blobs_points_repartitioned",
) -> SpatialData:
    """Return a blobs dataset with an extra repartitioned points element.

    The added points element is useful for transcript-cache development because it keeps the standard
    blobs point columns (`x`, `y`, `genes`, `instance_id`) while making partition-local tile shards easy to test.
    """
    from spatialdata.datasets import blobs

    if npartitions < 1:
        raise ValueError("`npartitions` must be at least 1.")

    sdata = blobs(n_points=n_points)
    source_points = sdata["blobs_points"]
    repartitioned_points = source_points.repartition(npartitions=npartitions)
    repartitioned_points.attrs.update(source_points.attrs)

    sdata[points_key] = repartitioned_points
    return sdata
