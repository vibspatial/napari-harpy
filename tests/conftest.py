from __future__ import annotations

import atexit
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
from spatialdata import SpatialData, read_zarr
from spatialdata.datasets import blobs

TEST_HOME = Path(tempfile.mkdtemp(prefix="napari-harpy-test-home-"))
TEST_CACHE = TEST_HOME / ".cache"
TEST_CONFIG = TEST_HOME / ".config"
TEST_NAPARI_CONFIG = TEST_CONFIG / "napari" / "settings.yaml"

TEST_CACHE.mkdir(parents=True, exist_ok=True)
TEST_CONFIG.mkdir(parents=True, exist_ok=True)
TEST_NAPARI_CONFIG.parent.mkdir(parents=True, exist_ok=True)

os.environ["HOME"] = str(TEST_HOME)
os.environ["XDG_CACHE_HOME"] = str(TEST_CACHE)
os.environ["XDG_CONFIG_HOME"] = str(TEST_CONFIG)
os.environ["NAPARI_CONFIG"] = str(TEST_NAPARI_CONFIG)
os.environ["NAPARI_CACHE_DIR"] = str(TEST_CACHE / "napari")

atexit.register(shutil.rmtree, TEST_HOME, ignore_errors=True)

SRC = Path(__file__).resolve().parents[1] / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _make_sdata_blobs() -> SpatialData:
    """Return a small SpatialData test object with dummy feature matrices in `table.obsm`."""
    sdata = blobs()
    table = sdata["table"]
    n_obs = table.n_obs
    rng = np.random.default_rng(seed=0)

    table.obsm["features_1"] = rng.normal(size=(n_obs, 4))
    table.obsm["features_2"] = rng.normal(size=(n_obs, 2))

    return sdata


@pytest.fixture
def sdata_blobs() -> SpatialData:
    """Return an in-memory SpatialData test object with dummy feature matrices."""
    return _make_sdata_blobs()


@pytest.fixture
def backed_sdata_blobs(tmp_path) -> SpatialData:
    """Return a backed SpatialData test object stored in a temporary zarr directory."""
    path = tmp_path / "blobs.zarr"
    sdata = _make_sdata_blobs()
    sdata.write(path)
    return read_zarr(path)
