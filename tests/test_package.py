from __future__ import annotations

from importlib.resources import files

import napari_harpy


def test_package_exposes_a_version() -> None:
    assert napari_harpy.__version__


def test_manifest_is_packaged_with_the_plugin() -> None:
    manifest = files("napari_harpy").joinpath("napari.yaml")
    assert manifest.is_file()
