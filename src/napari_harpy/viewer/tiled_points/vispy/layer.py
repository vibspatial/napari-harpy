"""Lifecycle-safe empty VisPy layer for logical tiled points."""

from __future__ import annotations

from typing import TYPE_CHECKING

from napari._vispy.layers.base import VispyBaseLayer
from vispy.scene.visuals import Compound

from napari_harpy.viewer.tiled_points.napari.layer import TiledPointsLayerModel

if TYPE_CHECKING:
    from napari._vispy.utils.qt_font import FontInfo


class VispyTiledPointsLayer(VispyBaseLayer[TiledPointsLayerModel]):
    """Provide napari layer lifecycle behavior before tile rendering exists."""

    def __init__(self, layer: TiledPointsLayerModel, font_info: FontInfo) -> None:
        # I6 will populate this stable compound root with retained tile visuals.
        super().__init__(layer, Compound([]), font_info=font_info)
        self.reset()

    def _on_data_change(self) -> None:
        """Refresh the empty root after replacement of the logical reference."""
        self.node.update()
        self._on_matrix_change()
