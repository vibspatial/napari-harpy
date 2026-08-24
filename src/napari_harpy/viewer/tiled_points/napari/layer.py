"""Logical napari layer model for cache-backed tiled points."""

from __future__ import annotations

import math
from numbers import Real
from typing import Any

import numpy as np
import numpy.typing as npt
from napari.layers import Layer
from napari.layers.base import _LayerSlicingState
from napari.types import LayerDataType
from napari.utils.events import Event

from napari_harpy.viewer.tiled_points.contracts import (
    TiledPointsDatasetReference,
    TiledPointsLayerStatus,
)

_DEFAULT_POINT_DIAMETER = 3.0
_SERIALIZATION_ERROR = (
    "TiledPointsLayerModel is a logical cache-backed layer and cannot be serialized through napari layer-data tuples."
)


class TiledPointsLayerModel(Layer):
    """Represent a complete tiled-points cache without storing point coordinates.

    ``data`` identifies the logical cache generation and its complete intrinsic
    bounds. Visible point payloads remain runtime and renderer state; they are
    never assigned to this model. The layer consequently keeps a stable extent
    while viewport tiles, value selections, and levels of detail change.
    """

    def __init__(
        self,
        data: TiledPointsDatasetReference,
        *,
        affine: Any | None = None,
        blending: str = "translucent",
        metadata: dict[str, Any] | None = None,
        name: str | None = None,
        opacity: float = 0.8,
        point_diameter: float = _DEFAULT_POINT_DIAMETER,
        rotate: Any | None = None,
        scale: Any | None = None,
        shear: Any | None = None,
        translate: Any | None = None,
        units: Any | None = None,
        visible: bool = True,
    ) -> None:
        if not isinstance(data, TiledPointsDatasetReference):
            raise ValueError("`data` must be TiledPointsDatasetReference.")
        self._data = data
        self._point_diameter = _require_point_diameter(point_diameter)
        self._display_status = TiledPointsLayerStatus()
        super().__init__(
            data=data,
            ndim=2,
            affine=affine,
            axis_labels=("y", "x"),
            blending=blending,
            cache=False,
            metadata=metadata,
            mode="pan_zoom",
            multiscale=False,
            name=name or data.points_name,
            opacity=opacity,
            rotate=rotate,
            scale=scale,
            shear=shear,
            translate=translate,
            units=units,
            visible=visible,
        )
        self.events.add(
            display_status=Event,
            point_diameter=Event,
            render_snapshot=Event,
            viewport=Event,
        )
        self.editable = False
        self._update_thumbnail()

    @property
    def data(self) -> TiledPointsDatasetReference:
        """Return the logical dataset reference, never resident point rows."""
        return self._data

    @data.setter
    def data(self, data: TiledPointsDatasetReference) -> None:
        if not isinstance(data, TiledPointsDatasetReference):
            raise ValueError("`data` must be TiledPointsDatasetReference.")
        if data == self._data:
            return
        self._data = data
        self._clear_extent()
        self.events.data(value=data)
        # Notify the VisPy boundary directly. A generic ``refresh()`` would
        # emit ``set_data`` again and also repeat no-op slicing, placeholder
        # thumbnail, and highlighting work for this logical 2D layer.
        self.events.set_data(value=data)

    @property
    def point_diameter(self) -> float:
        """Return the requested marker diameter in canvas pixels."""
        return self._point_diameter

    @point_diameter.setter
    def point_diameter(self, value: float) -> None:
        diameter = _require_point_diameter(value)
        if diameter == self._point_diameter:
            return
        self._point_diameter = diameter
        self.events.point_diameter(value=diameter)

    @property
    def display_status(self) -> TiledPointsLayerStatus:
        """Return the latest immutable display-status summary."""
        return self._display_status

    @display_status.setter
    def display_status(self, status: TiledPointsLayerStatus) -> None:
        if not isinstance(status, TiledPointsLayerStatus):
            raise ValueError("`display_status` must be TiledPointsLayerStatus.")
        if status == self._display_status:
            return
        self._display_status = status
        self.events.display_status(value=status)

    @property
    def _extent_data(self) -> npt.NDArray[np.float64]:
        """Return complete cache bounds in napari data-axis ``(y, x)`` order."""
        return np.array(
            (
                (self.data.y_min, self.data.x_min),
                (self.data.y_max, self.data.x_max),
            ),
            dtype=np.float64,
        )

    def _get_ndim(self) -> int:
        return 2

    def _get_state(self) -> dict[str, Any]:
        raise NotImplementedError(_SERIALIZATION_ERROR)

    def as_layer_data_tuple(self) -> None:
        """Reject napari's standard array-oriented layer serialization."""
        raise NotImplementedError(_SERIALIZATION_ERROR)

    def _set_view_slice(self) -> None:
        # A tiled-points cache is intrinsically 2D; there is no array slice to install.
        return None

    def _get_value(self, position: Any) -> None:
        # Point picking is deliberately outside the initial read-only layer contract.
        del position
        return None

    def _update_thumbnail(self) -> None:
        """Install a deterministic placeholder independent of resident tiles."""
        thumbnail = np.zeros(self._thumbnail_shape, dtype=np.uint8)
        thumbnail[..., 3] = 255
        for y, x in ((8, 8), (8, 23), (16, 16), (23, 8), (23, 23)):
            thumbnail[y - 1 : y + 2, x - 1 : x + 2, :] = (115, 205, 230, 255)
        self.thumbnail = thumbnail

    def _get_layer_slicing_state(self, data: LayerDataType, cache: bool) -> _TiledPointsLayerSlicingState:
        return _TiledPointsLayerSlicingState(layer=self, data=data, cache=cache)


class _TiledPointsLayerSlicingState(_LayerSlicingState):
    """Provide napari's required slicing contract for a fixed 2D layer."""

    def _set_view_slice(self) -> None:
        # The logical layer has no non-displayed axes and therefore no slice work.
        return None


def _require_point_diameter(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
        raise ValueError("`point_diameter` must be a positive finite number.")
    return float(value)
