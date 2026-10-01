"""Cache-backed tiled-points visualization for napari."""

from napari_harpy.viewer.tiled_points.contracts import (
    TiledPointsDatasetReference,
    TiledPointsLayerStatus,
    TiledPointsRenderBatch,
    TiledPointsRenderResult,
    TiledPointsRenderSnapshot,
    TiledPointsRenderTile,
    TiledPointsViewportState,
    TileResidencyKey,
)
from napari_harpy.viewer.tiled_points.napari.layer import TiledPointsLayerModel
from napari_harpy.viewer.tiled_points.napari.registration import (
    TiledPointsLayerCompatibilityError,
    register_tiled_points_layer,
)

__all__ = [
    "TiledPointsDatasetReference",
    "TiledPointsLayerCompatibilityError",
    "TiledPointsLayerModel",
    "TiledPointsLayerStatus",
    "TiledPointsRenderBatch",
    "TiledPointsRenderResult",
    "TiledPointsRenderSnapshot",
    "TiledPointsRenderTile",
    "TiledPointsViewportState",
    "TileResidencyKey",
    "register_tiled_points_layer",
]
