from __future__ import annotations

from spatialdata import SpatialData

from napari_harpy._spatialdata import get_annotating_table_names


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
