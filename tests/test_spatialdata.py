from __future__ import annotations

from spatialdata import SpatialData

from napari_harpy._spatialdata import (
    SpatialDataAdapter,
    get_annotating_table_names,
    get_table_metadata,
    get_table_obsm_keys,
)


def test_get_annotating_table_names_returns_tables_for_annotated_label(sdata_blobs: SpatialData) -> None:
    table_names = get_annotating_table_names(sdata_blobs, "blobs_labels")

    assert table_names == ["table"]


def test_get_annotating_table_names_returns_empty_list_for_unannotated_label(sdata_blobs: SpatialData) -> None:
    table_names = get_annotating_table_names(sdata_blobs, "blobs_multiscale_labels")

    assert table_names == []


def test_sdata_blobs_fixture_adds_dummy_feature_matrices(sdata_blobs: SpatialData) -> None:
    table = sdata_blobs["table"]

    assert "features_1" in table.obsm
    assert "features_2" in table.obsm
    assert table.obsm["features_1"].shape == (table.n_obs, 4)
    assert table.obsm["features_2"].shape == (table.n_obs, 2)


def test_get_table_obsm_keys_returns_sorted_feature_matrix_keys(sdata_blobs: SpatialData) -> None:
    obsm_keys = get_table_obsm_keys(sdata_blobs, "table")

    assert obsm_keys == ["features_1", "features_2"]


def test_spatialdata_adapter_resolves_table_metadata(sdata_blobs: SpatialData) -> None:
    adapter = SpatialDataAdapter()

    metadata = adapter.get_table_metadata(sdata_blobs, "table")

    assert metadata.table_name == "table"
    assert metadata.region_key == "region"
    assert metadata.instance_key == "instance_id"
    assert metadata.regions == ("blobs_labels",)
    assert metadata.annotates("blobs_labels")
    assert not metadata.annotates("blobs_multiscale_labels")


def test_get_table_metadata_returns_table_linkage(sdata_blobs: SpatialData) -> None:
    metadata = get_table_metadata(sdata_blobs, "table")

    assert metadata.region_key == "region"
    assert metadata.instance_key == "instance_id"
