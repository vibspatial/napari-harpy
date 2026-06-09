from __future__ import annotations

import subprocess
import sys
import textwrap
from importlib.resources import files

import yaml
from spatialdata import read_zarr

import napari_harpy
from napari_harpy.datasets import blobs_multi_region, blobs_points_repartitioned


def _run_import_smoke_test(code: str) -> None:
    subprocess.run([sys.executable, "-c", textwrap.dedent(code)], check=True)


def test_package_exposes_a_version() -> None:
    assert napari_harpy.__version__


def test_package_import_is_lazy() -> None:
    _run_import_smoke_test(
        """
        import sys

        import napari_harpy

        forbidden_roots = ("napari", "qtpy")
        loaded = sorted(
            name
            for name in sys.modules
            if any(name == root or name.startswith(root + ".") for root in forbidden_roots)
        )
        if loaded:
            raise AssertionError(loaded)
        assert napari_harpy.__version__
        """
    )


def test_headless_import_is_lazy() -> None:
    _run_import_smoke_test(
        """
        import sys

        from napari_harpy import headless

        forbidden_roots = ("napari", "napari_harpy.widgets", "qtpy")
        loaded = sorted(
            name
            for name in sys.modules
            if any(name == root or name.startswith(root + ".") for root in forbidden_roots)
        )
        if loaded:
            raise AssertionError(loaded)
        assert headless.__name__ == "napari_harpy.headless"
        """
    )


def test_representative_lazy_attributes_resolve() -> None:
    _run_import_smoke_test(
        """
        import sys

        import napari_harpy.widgets
        from napari_harpy import Interactive
        from napari_harpy.widgets import ViewerWidget

        assert Interactive.__name__ == "Interactive"
        assert ViewerWidget.__name__ == "ViewerWidget"
        assert "napari_harpy.widgets.viewer.widget" in sys.modules
        """
    )


def test_manifest_is_packaged_with_the_plugin() -> None:
    manifest = files("napari_harpy").joinpath("napari.yaml")
    assert manifest.is_file()


def test_manifest_contributes_shapes_annotation_widget() -> None:
    manifest = files("napari_harpy").joinpath("napari.yaml")
    data = yaml.safe_load(manifest.read_text())

    commands = {command["id"]: command for command in data["contributions"]["commands"]}
    widgets = {widget["display_name"]: widget for widget in data["contributions"]["widgets"]}

    assert commands["napari-harpy.shapes_annotation"]["python_name"] == (
        "napari_harpy.widgets.shapes_annotation.widget:ShapesAnnotation"
    )
    assert widgets["Annotation"]["command"] == "napari-harpy.shapes_annotation"


def test_blobs_multi_region_builds_a_multi_region_table() -> None:
    sdata = blobs_multi_region()
    table = sdata["table_multi"]

    assert "table_multi" in sdata.tables
    assert table.n_obs == sdata["table"].n_obs * 2
    assert tuple(table.obs["region"].cat.categories) == ("blobs_labels", "blobs_labels_2")
    assert table.obsm["features_1"].shape == (table.n_obs, 4)


def test_blobs_points_repartitioned_adds_repartitioned_points_element() -> None:
    sdata = blobs_points_repartitioned(n_points=20, npartitions=4)
    points = sdata["blobs_points_repartitioned"]

    assert "blobs_points_repartitioned" in sdata.points
    assert points.npartitions == 4
    assert {"x", "y", "genes", "instance_id"}.issubset(points.columns)
    assert points.attrs["spatialdata_attrs"]["feature_key"] == "genes"
    assert points.attrs["spatialdata_attrs"]["instance_key"] == "instance_id"
    assert sdata.locate_element(points) == ["points/blobs_points_repartitioned"]


def test_blobs_points_repartitioned_can_be_written_and_read(tmp_path) -> None:
    path = tmp_path / "blobs_points_repartitioned.zarr"
    sdata = blobs_points_repartitioned(n_points=20, npartitions=4)

    sdata.write(path)
    reread = read_zarr(path)

    assert "blobs_points_repartitioned" in reread.points
    assert reread.points["blobs_points_repartitioned"].npartitions == 4
    assert reread.locate_element(reread.points["blobs_points_repartitioned"]) == [
        "points/blobs_points_repartitioned"
    ]
