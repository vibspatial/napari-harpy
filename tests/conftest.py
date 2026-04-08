from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from spatialdata import SpatialData, read_zarr
from spatialdata.datasets import blobs

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
