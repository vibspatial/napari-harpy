from __future__ import annotations

import subprocess
import sys
import textwrap
from importlib.resources import files

import napari_harpy
from napari_harpy.datasets import blobs_multi_region


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


def test_blobs_multi_region_builds_a_multi_region_table() -> None:
    sdata = blobs_multi_region()
    table = sdata["table_multi"]

    assert "table_multi" in sdata.tables
    assert table.n_obs == sdata["table"].n_obs * 2
    assert tuple(table.obs["region"].cat.categories) == ("blobs_labels", "blobs_labels_2")
    assert table.obsm["features_1"].shape == (table.n_obs, 4)
