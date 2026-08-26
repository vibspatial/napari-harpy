"""Napari model, controls, and registration for tiled-points layers."""

from napari_harpy.viewer.tiled_points.napari.layer import TiledPointsLayerModel
from napari_harpy.viewer.tiled_points.napari.registration import (
    TiledPointsLayerCompatibilityError,
    register_tiled_points_layer,
)

__all__ = [
    "TiledPointsLayerCompatibilityError",
    "TiledPointsLayerModel",
    "register_tiled_points_layer",
]
