from __future__ import annotations

import pytest
from spatialdata import SpatialData, read_zarr

from napari_harpy._persistence import PersistenceController
from napari_harpy._spatialdata import SpatialDataAdapter


def test_persistence_controller_requires_backed_spatialdata(sdata_blobs: SpatialData) -> None:
    controller = PersistenceController(SpatialDataAdapter())
    controller.bind(sdata_blobs, "table")

    with pytest.raises(ValueError, match="not backed by zarr"):
        controller.sync_table_obs()


def test_persistence_controller_syncs_table_obs_to_backed_store(backed_sdata_blobs: SpatialData) -> None:
    controller = PersistenceController(SpatialDataAdapter())
    controller.bind(backed_sdata_blobs, "table")

    backed_sdata_blobs["table"].obs["test"] = 7

    table_path = controller.sync_table_obs()
    reread = read_zarr(backed_sdata_blobs.path)

    assert table_path == "tables/table"
    assert "test" in reread["table"].obs.columns
    assert reread["table"].obs["test"].tolist() == [7] * reread["table"].n_obs
    assert sorted(reread["table"].obsm.keys()) == ["features_1", "features_2"]
