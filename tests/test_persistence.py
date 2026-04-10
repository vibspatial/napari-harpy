from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import zarr
from spatialdata import SpatialData, read_zarr

from napari_harpy._annotation import USER_CLASS_COLORS_KEY, USER_CLASS_COLUMN
from napari_harpy._classifier import (
    CLASSIFIER_CONFIG_KEY,
    PRED_CLASS_COLORS_KEY,
    PRED_CLASS_COLUMN,
    PRED_CONFIDENCE_COLUMN,
)
from napari_harpy._persistence import PersistenceController
from napari_harpy._spatialdata import SpatialDataAdapter


def test_persistence_controller_requires_backed_spatialdata(sdata_blobs: SpatialData) -> None:
    controller = PersistenceController(SpatialDataAdapter())
    controller.bind(sdata_blobs, "table")

    with pytest.raises(ValueError, match="not backed by zarr"):
        controller.sync_table_state()


def test_persistence_controller_tracks_dirty_state_per_selected_table(
    sdata_blobs: SpatialData,
    backed_sdata_blobs: SpatialData,
) -> None:
    controller = PersistenceController(SpatialDataAdapter())
    controller.bind(sdata_blobs, "table")

    assert controller.is_dirty is False

    controller.mark_dirty()

    assert controller.is_dirty is True

    controller.bind(backed_sdata_blobs, "table")

    assert controller.is_dirty is False

    controller.bind(sdata_blobs, "table")

    assert controller.is_dirty is True


def test_persistence_controller_syncs_table_obs_and_colors_to_backed_store(backed_sdata_blobs: SpatialData) -> None:
    controller = PersistenceController(SpatialDataAdapter())
    controller.bind(backed_sdata_blobs, "table")
    index = backed_sdata_blobs["table"].obs.index

    backed_sdata_blobs["table"].obs[USER_CLASS_COLUMN] = pd.Categorical(
        [0] * (backed_sdata_blobs["table"].n_obs - 1) + [2],
        categories=[0, 2],
    )
    backed_sdata_blobs["table"].obs[PRED_CLASS_COLUMN] = pd.Categorical(
        [0] * (backed_sdata_blobs["table"].n_obs - 1) + [2],
        categories=[0, 2],
    )
    backed_sdata_blobs["table"].obs[PRED_CONFIDENCE_COLUMN] = pd.Series(
        [0.0] * (backed_sdata_blobs["table"].n_obs - 1) + [0.95],
        index=index,
        dtype="float64",
    )
    backed_sdata_blobs["table"].uns[USER_CLASS_COLORS_KEY] = ["#111111", "#222222"]
    backed_sdata_blobs["table"].uns[PRED_CLASS_COLORS_KEY] = ["#111111", "#222222"]
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
    assert isinstance(reread["table"].obs[PRED_CLASS_COLUMN].dtype, pd.CategoricalDtype)
    assert list(reread["table"].obs[PRED_CLASS_COLUMN].cat.categories) == [0, 2]
    assert reread["table"].obs[PRED_CLASS_COLUMN].tolist().count(2) == 1
    assert reread["table"].obs[PRED_CONFIDENCE_COLUMN].max() == 0.95
    assert list(reread["table"].uns[USER_CLASS_COLORS_KEY]) == ["#111111", "#222222"]
    assert list(reread["table"].uns[PRED_CLASS_COLORS_KEY]) == ["#111111", "#222222"]
    assert reread["table"].uns[CLASSIFIER_CONFIG_KEY]["feature_key"] == "features_1"
    assert sorted(reread["table"].obsm.keys()) == ["features_1", "features_2"]


def _write_disk_snapshot_payload(
    backed_sdata_blobs: SpatialData,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, object]]:
    table = backed_sdata_blobs["table"]
    obs = table.obs.copy()
    obs["disk_user_class"] = pd.Categorical([0] * (table.n_obs - 1) + [7], categories=[0, 7])

    obsm = dict(table.obsm)
    obsm["disk_features"] = np.arange(table.n_obs, dtype=np.float64).reshape(table.n_obs, 1)

    uns = dict(table.uns)
    uns["disk_flag"] = {"source": "disk"}

    _write_disk_snapshot_state(backed_sdata_blobs, obs=obs, obsm=obsm, uns=uns)
    return obs, obsm, uns


def _write_disk_snapshot_state(
    backed_sdata_blobs: SpatialData,
    *,
    obs: pd.DataFrame,
    obsm: dict[str, object],
    uns: dict[str, object],
) -> None:
    root = zarr.open_group(backed_sdata_blobs.path, mode="a", use_consolidated=False)
    table_group = root["tables/table"]
    ad.io.write_elem(table_group, "obs", obs)
    ad.io.write_elem(table_group, "obsm", obsm)
    ad.io.write_elem(table_group, "uns", uns)


def test_persistence_controller_reads_selected_table_snapshot_from_backed_store(
    backed_sdata_blobs: SpatialData,
) -> None:
    controller = PersistenceController(SpatialDataAdapter())
    controller.bind(backed_sdata_blobs, "table")
    disk_obs, disk_obsm, disk_uns = _write_disk_snapshot_payload(backed_sdata_blobs)

    snapshot = controller.read_table_snapshot_from_disk()

    assert snapshot.table_name == "table"
    assert snapshot.table_path == "tables/table"
    assert snapshot.obs.equals(disk_obs)
    assert sorted(snapshot.obsm.keys()) == sorted(disk_obsm.keys())
    assert np.array_equal(snapshot.obsm["disk_features"], disk_obsm["disk_features"])
    assert snapshot.uns["disk_flag"] == disk_uns["disk_flag"]
    assert "disk_user_class" not in backed_sdata_blobs["table"].obs
    assert "disk_features" not in backed_sdata_blobs["table"].obsm
    assert "disk_flag" not in backed_sdata_blobs["table"].uns


def test_persistence_controller_replaces_selected_table_with_reloaded_snapshot(
    backed_sdata_blobs: SpatialData,
) -> None:
    controller = PersistenceController(SpatialDataAdapter())
    controller.bind(backed_sdata_blobs, "table")
    _write_disk_snapshot_payload(backed_sdata_blobs)
    original_table = backed_sdata_blobs["table"]

    table_path = controller.reload_table_state()
    reloaded_table = backed_sdata_blobs["table"]

    assert table_path == "tables/table"
    assert reloaded_table is original_table
    assert "disk_user_class" in reloaded_table.obs
    assert "disk_features" in reloaded_table.obsm
    assert reloaded_table.uns["disk_flag"] == {"source": "disk"}
    assert list(reloaded_table.var_names) == list(original_table.var_names)
    assert backed_sdata_blobs.locate_element(reloaded_table) == ["tables/table"]


def test_persistence_controller_clears_dirty_state_after_sync(backed_sdata_blobs: SpatialData) -> None:
    controller = PersistenceController(SpatialDataAdapter())
    controller.bind(backed_sdata_blobs, "table")
    controller.mark_dirty()

    assert controller.is_dirty is True

    controller.sync_table_state()

    assert controller.is_dirty is False


def test_persistence_controller_clears_dirty_state_after_reload(backed_sdata_blobs: SpatialData) -> None:
    controller = PersistenceController(SpatialDataAdapter())
    controller.bind(backed_sdata_blobs, "table")
    _write_disk_snapshot_payload(backed_sdata_blobs)
    controller.mark_dirty()

    assert controller.is_dirty is True

    controller.reload_table_state()

    assert controller.is_dirty is False


def test_persistence_controller_reloads_obsm_key_written_directly_to_disk_group(
    backed_sdata_blobs: SpatialData,
) -> None:
    """Cover direct `tables/<table>/obsm/<key>` writes that future extensions will rely on.

    This is an important regression test because plugin extensions may add new
    feature matrices straight to the backed zarr group without rewriting the
    full `obsm` mapping. Reload must continue to discover those incrementally
    written keys and surface them on the in-memory table.
    """
    controller = PersistenceController(SpatialDataAdapter())
    controller.bind(backed_sdata_blobs, "table")
    table = backed_sdata_blobs["table"]
    features_3 = np.arange(table.n_obs * 3, dtype=np.float32).reshape(table.n_obs, 3)

    root = zarr.open_group(backed_sdata_blobs.path, mode="a", use_consolidated=False)
    table_group = root["tables/table"]
    ad.io.write_elem(table_group["obsm"], "features_3", features_3)

    assert "features_3" not in table.obsm

    table_path = controller.reload_table_state()

    assert table_path == "tables/table"
    assert sorted(table.obsm.keys()) == ["features_1", "features_2", "features_3"]
    assert np.array_equal(table.obsm["features_3"], features_3)


def test_persistence_controller_rejects_reload_when_row_count_changed(
    backed_sdata_blobs: SpatialData,
) -> None:
    controller = PersistenceController(SpatialDataAdapter())
    controller.bind(backed_sdata_blobs, "table", "blobs_labels")
    table = backed_sdata_blobs["table"]

    obs = table.obs.iloc[:-1].copy()
    obsm = {key: value[:-1] for key, value in table.obsm.items()}
    uns = dict(table.uns)
    _write_disk_snapshot_state(backed_sdata_blobs, obs=obs, obsm=obsm, uns=uns)

    with pytest.raises(ValueError, match="requires unchanged row identity and order"):
        controller.reload_table_state()

    assert backed_sdata_blobs["table"].n_obs == table.n_obs
    assert backed_sdata_blobs["table"].obs.index.equals(table.obs.index)


def test_persistence_controller_allows_reload_when_obs_names_change_but_rowwise_identity_matches(
    backed_sdata_blobs: SpatialData,
) -> None:
    controller = PersistenceController(SpatialDataAdapter())
    controller.bind(backed_sdata_blobs, "table", "blobs_labels")
    table = backed_sdata_blobs["table"]

    obs = table.obs.copy()
    obs.index = pd.Index([f"disk_row_{index}" for index in range(table.n_obs)], name=obs.index.name)
    obsm = dict(table.obsm)
    uns = dict(table.uns)
    _write_disk_snapshot_state(backed_sdata_blobs, obs=obs, obsm=obsm, uns=uns)

    table_path = controller.reload_table_state()

    assert table_path == "tables/table"
    assert backed_sdata_blobs["table"].obs.index.equals(obs.index)
    assert backed_sdata_blobs["table"].obs["instance_id"].equals(obs["instance_id"])


def test_persistence_controller_rejects_reload_when_region_key_values_changed_rowwise(
    backed_sdata_blobs: SpatialData,
) -> None:
    controller = PersistenceController(SpatialDataAdapter())
    controller.bind(backed_sdata_blobs, "table", "blobs_labels")
    table = backed_sdata_blobs["table"]

    obs = table.obs.copy()
    obs["region"] = obs["region"].astype(object)
    obs.loc[obs.index[0], "region"] = "different_labels"
    obsm = dict(table.obsm)
    uns = dict(table.uns)
    _write_disk_snapshot_state(backed_sdata_blobs, obs=obs, obsm=obsm, uns=uns)

    with pytest.raises(ValueError, match="`region` values do not exactly match"):
        controller.reload_table_state()

    assert backed_sdata_blobs["table"].obs["region"].equals(table.obs["region"])


def test_persistence_controller_rejects_reload_when_row_order_changed(
    backed_sdata_blobs: SpatialData,
) -> None:
    controller = PersistenceController(SpatialDataAdapter())
    controller.bind(backed_sdata_blobs, "table", "blobs_labels")
    table = backed_sdata_blobs["table"]
    reversed_positions = np.arange(table.n_obs - 1, -1, -1)

    obs = table.obs.iloc[reversed_positions].copy()
    obsm = {key: value[reversed_positions] for key, value in table.obsm.items()}
    uns = dict(table.uns)
    _write_disk_snapshot_state(backed_sdata_blobs, obs=obs, obsm=obsm, uns=uns)

    with pytest.raises(ValueError, match="instance_id` values do not exactly match"):
        controller.reload_table_state()

    assert backed_sdata_blobs["table"].obs.index.equals(table.obs.index)


def test_persistence_controller_rejects_reload_when_instance_key_values_changed_rowwise(
    backed_sdata_blobs: SpatialData,
) -> None:
    controller = PersistenceController(SpatialDataAdapter())
    controller.bind(backed_sdata_blobs, "table", "blobs_labels")
    table = backed_sdata_blobs["table"]

    obs = table.obs.copy()
    swapped = obs["instance_id"].to_numpy(copy=True)
    swapped[[0, 1]] = swapped[[1, 0]]
    obs["instance_id"] = swapped
    obsm = dict(table.obsm)
    uns = dict(table.uns)
    _write_disk_snapshot_state(backed_sdata_blobs, obs=obs, obsm=obsm, uns=uns)

    with pytest.raises(ValueError, match="instance_id` values do not exactly match"):
        controller.reload_table_state()

    assert backed_sdata_blobs["table"].obs["instance_id"].equals(table.obs["instance_id"])


def test_persistence_controller_rejects_reload_when_spatialdata_attrs_missing(
    backed_sdata_blobs: SpatialData,
) -> None:
    controller = PersistenceController(SpatialDataAdapter())
    controller.bind(backed_sdata_blobs, "table", "blobs_labels")
    table = backed_sdata_blobs["table"]

    obs = table.obs.copy()
    obsm = dict(table.obsm)
    uns = {"disk_flag": {"source": "disk"}}
    _write_disk_snapshot_state(backed_sdata_blobs, obs=obs, obsm=obsm, uns=uns)

    with pytest.raises(ValueError, match="missing `spatialdata_attrs` metadata"):
        controller.reload_table_state()


def test_persistence_controller_rejects_reload_when_selected_segmentation_is_no_longer_annotated(
    backed_sdata_blobs: SpatialData,
) -> None:
    controller = PersistenceController(SpatialDataAdapter())
    controller.bind(backed_sdata_blobs, "table", "blobs_labels")
    table = backed_sdata_blobs["table"]

    obs = table.obs.copy()
    obsm = dict(table.obsm)
    uns = dict(table.uns)
    uns["spatialdata_attrs"] = {
        **uns["spatialdata_attrs"],
        "region": ["different_labels"],
    }
    _write_disk_snapshot_state(backed_sdata_blobs, obs=obs, obsm=obsm, uns=uns)

    with pytest.raises(ValueError, match="no longer annotates the selected segmentation"):
        controller.reload_table_state()
