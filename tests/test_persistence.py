from __future__ import annotations

import pandas as pd
import pytest
from spatialdata import SpatialData, read_zarr

from napari_harpy._annotation import USER_CLASS_COLORS_KEY, USER_CLASS_COLUMN
from napari_harpy._classifier import CLASSIFIER_CONFIG_KEY, PRED_CLASS_COLUMN, PRED_CONFIDENCE_COLUMN
from napari_harpy._persistence import PersistenceController
from napari_harpy._spatialdata import SpatialDataAdapter


def test_persistence_controller_requires_backed_spatialdata(sdata_blobs: SpatialData) -> None:
    controller = PersistenceController(SpatialDataAdapter())
    controller.bind(sdata_blobs, "table")

    with pytest.raises(ValueError, match="not backed by zarr"):
        controller.sync_table_state()


def test_persistence_controller_syncs_table_obs_and_colors_to_backed_store(backed_sdata_blobs: SpatialData) -> None:
    controller = PersistenceController(SpatialDataAdapter())
    controller.bind(backed_sdata_blobs, "table")
    index = backed_sdata_blobs["table"].obs.index

    backed_sdata_blobs["table"].obs[USER_CLASS_COLUMN] = pd.Categorical(
        [0] * (backed_sdata_blobs["table"].n_obs - 1) + [2],
        categories=[0, 2],
    )
    backed_sdata_blobs["table"].obs[PRED_CLASS_COLUMN] = pd.Series(
        [0] * (backed_sdata_blobs["table"].n_obs - 1) + [2],
        index=index,
        dtype="int64",
    )
    backed_sdata_blobs["table"].obs[PRED_CONFIDENCE_COLUMN] = pd.Series(
        [0.0] * (backed_sdata_blobs["table"].n_obs - 1) + [0.95],
        index=index,
        dtype="float64",
    )
    backed_sdata_blobs["table"].uns[USER_CLASS_COLORS_KEY] = ["#111111", "#222222"]
    backed_sdata_blobs["table"].uns[CLASSIFIER_CONFIG_KEY] = {
        "model_type": "RandomForestClassifier",
        "feature_key": "features_1",
        "trained": True,
    }

    table_path = controller.sync_table_state()
    reread = read_zarr(backed_sdata_blobs.path)

    assert table_path == "tables/table"
    assert USER_CLASS_COLUMN in reread["table"].obs
    assert isinstance(reread["table"].obs[USER_CLASS_COLUMN].dtype, pd.CategoricalDtype)
    assert list(reread["table"].obs[USER_CLASS_COLUMN].cat.categories) == [0, 2]
    assert reread["table"].obs[USER_CLASS_COLUMN].tolist().count(2) == 1
    assert reread["table"].obs[PRED_CLASS_COLUMN].tolist().count(2) == 1
    assert reread["table"].obs[PRED_CONFIDENCE_COLUMN].max() == 0.95
    assert list(reread["table"].uns[USER_CLASS_COLORS_KEY]) == ["#111111", "#222222"]
    assert reread["table"].uns[CLASSIFIER_CONFIG_KEY]["feature_key"] == "features_1"
    assert sorted(reread["table"].obsm.keys()) == ["features_1", "features_2"]
