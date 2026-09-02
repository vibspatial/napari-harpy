"""Define where the points cache lives inside a backed SpatialData store."""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Final

POINTS_CACHE_DIRECTORY_NAME: Final = "transcripts_vis_zarr"
_SPATIALDATA_POINTS_GROUP: Final = "points"


def points_element_path(points_name: str) -> str:
    """Return the canonical SpatialData-relative path for a points element."""
    if not isinstance(points_name, str) or not points_name or "/" in points_name:
        raise ValueError("`points_name` must be a nonempty path-segment string.")
    return str(PurePosixPath(_SPATIALDATA_POINTS_GROUP, points_name))


def points_cache_path(spatialdata_path: Path, points_name: str) -> Path:
    """Return the conventional cache path nested below a points element."""
    if not isinstance(spatialdata_path, Path):
        raise ValueError("`spatialdata_path` must be pathlib.Path.")
    return spatialdata_path / points_element_path(points_name) / POINTS_CACHE_DIRECTORY_NAME
