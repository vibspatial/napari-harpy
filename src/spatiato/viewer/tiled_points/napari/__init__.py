"""Napari model, controls, and registration for tiled-points layers."""

from spatiato.viewer.tiled_points.napari.layer import TiledPointsLayerModel
from spatiato.viewer.tiled_points.napari.registration import (
    TiledPointsLayerCompatibilityError,
    register_tiled_points_layer,
)

__all__ = [
    "TiledPointsLayerCompatibilityError",
    "TiledPointsLayerModel",
    "register_tiled_points_layer",
]
