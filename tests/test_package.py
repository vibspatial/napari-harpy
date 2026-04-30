from __future__ import annotations

from importlib.resources import files

import napari_harpy
from napari_harpy.datasets import blobs_multi_region


def test_package_exposes_a_version() -> None:
    assert napari_harpy.__version__


def test_manifest_is_packaged_with_the_plugin() -> None:
    manifest = files("napari_harpy").joinpath("napari.yaml")
    assert manifest.is_file()


def test_blobs_multi_region_builds_a_multi_region_table() -> None:
    sdata = blobs_multi_region()
    table = sdata["table_multi"]

    assert "table_multi" in sdata.tables
    assert table.n_obs == sdata["table"].n_obs * 2
    assert tuple(table.obs["region"].cat.categories) == ("blobs_labels", "blobs_labels_2")
    assert table.obsm["features_1"].shape == (table.n_obs, 4)
