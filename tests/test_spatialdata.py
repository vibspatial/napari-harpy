from __future__ import annotations

from spatialdata.datasets import blobs

from napari_harpy._spatialdata import get_annotating_table_names


def test_get_annotating_table_names_returns_tables_for_annotated_label() -> None:
    sdata = blobs()

    table_names = get_annotating_table_names(sdata, "blobs_labels")

    assert table_names == ["table"]


def test_get_annotating_table_names_returns_empty_list_for_unannotated_label() -> None:
    sdata = blobs()

    table_names = get_annotating_table_names(sdata, "blobs_multiscale_labels")

    assert table_names == []
