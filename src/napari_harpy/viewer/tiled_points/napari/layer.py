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
    DEFAULT_HARD_RENDER_POINT_BUDGET,
    DEFAULT_TARGET_PIXELS_PER_POINT,
    TiledPointsDatasetReference,
    TiledPointsLayerStatus,
    TiledPointsViewportState,
)
from napari_harpy.viewer.tiled_points.napari.viewport import (
    _viewport_state_from_draw,
    _viewport_state_with_budget,
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

    Before the first instance is added to napari, integration code calls
    ``napari_harpy.viewer.tiled_points.napari.register_tiled_points_layer()``.
    That function installs napari's private
    ``TiledPointsLayerModel -> VispyTiledPointsLayer`` mapping. Constructing this
    model alone does not construct a renderer. When the instance is later
    inserted into a GUI viewer's layer list, napari's layer-insertion callback
    consults the mapping and constructs
    ``VispyTiledPointsLayer(model, font_info)`` on the GUI thread.

    This model owns persistent logical and presentation state: dataset
    identity, extent, transforms, palette, point style, and renderer input
    events. The corresponding VisPy layer owns scene nodes and GPU resources.
    The model itself performs no cache reads.
    """

    def __init__(
        self,
        data: TiledPointsDatasetReference,
        *,
        value_palette: npt.NDArray[np.uint8],
        max_gpu_tile_bytes: int,
        affine: Any | None = None,
        blending: str = "translucent",
        metadata: dict[str, Any] | None = None,
        name: str | None = None,
        opacity: float = 0.8,
        point_diameter: float = _DEFAULT_POINT_DIAMETER,
        hard_render_point_budget: int = DEFAULT_HARD_RENDER_POINT_BUDGET,
        target_pixels_per_point: float = DEFAULT_TARGET_PIXELS_PER_POINT,
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
        self._value_palette = _validated_value_palette(value_palette, value_count=data.value_count)
        self._max_gpu_tile_bytes = _require_positive_integer(max_gpu_tile_bytes, "max_gpu_tile_bytes")
        self._point_diameter = _require_point_diameter(point_diameter)
        self._hard_render_point_budget = _require_positive_integer(
            hard_render_point_budget,
            "hard_render_point_budget",
        )
        self._target_pixels_per_point = _require_positive_finite_float(
            target_pixels_per_point,
            "target_pixels_per_point",
        )
        self._viewport_state: TiledPointsViewportState | None = None
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
        # ``viewport`` carries an outbound normalized viewport request;
        # ``render_snapshot`` carries renderer input; ``render_snapshot_result``
        # acknowledges whether that candidate was atomically applied. The model
        # coordinates these events but performs no cache IO.
        self.events.add(
            display_status=Event,
            point_diameter=Event,
            render_error=Event,
            render_snapshot=Event,
            render_snapshot_result=Event,
            value_palette=Event,
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
        if data.value_count != self._data.value_count:
            raise ValueError(
                "Replacement data must preserve `value_count`; construct a new layer for a new vocabulary."
            )
        self._data = data
        self._clear_extent()
        self.events.data(value=data)
        # Notify the VisPy boundary directly. A generic ``refresh()`` would
        # emit ``set_data`` again and also repeat no-op slicing, placeholder
        # thumbnail, and highlighting work for this logical 2D layer.
        self.events.set_data(value=data)

    @property
    def value_palette(self) -> npt.NDArray[np.uint8]:
        """Return the complete immutable value-ID-aligned RGBA palette."""
        return self._value_palette

    @value_palette.setter
    def value_palette(self, value: npt.NDArray[np.uint8]) -> None:
        palette = _validated_value_palette(value, value_count=self.data.value_count)
        if np.array_equal(palette, self._value_palette):
            return
        self._value_palette = palette
        self.events.value_palette(value=palette)

    @property
    def max_gpu_tile_bytes(self) -> int:
        """Return the byte limit for one complete packed vertex payload.

        The name is retained temporarily for application-setting compatibility;
        the constant-resource renderer no longer retains per-tile GPU objects.
        """
        return self._max_gpu_tile_bytes

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
    def hard_render_point_budget(self) -> int:
        """Return the viewer's absolute upper bound for visible point rows."""
        return self._hard_render_point_budget

    @hard_render_point_budget.setter
    def hard_render_point_budget(self, value: int) -> None:
        budget = _require_positive_integer(value, "hard_render_point_budget")
        if budget == self._hard_render_point_budget:
            return
        self._hard_render_point_budget = budget
        self._refresh_viewport_budget()

    @property
    def target_pixels_per_point(self) -> float:
        """Return the target logical canvas-pixel area per rendered point.

        A value of ``9.0`` aims for approximately one point per ``3 x 3``
        canvas-pixel region::

            +---+---+---+
            |   |   |   |
            +---+---+---+
            |   | ● |   |  approximately one rendered point
            +---+---+---+
            |   |   |   |
            +---+---+---+

        This is a display-density heuristic, not the marker diameter or a
        guarantee that rendered points will be spatially separated.
        """
        return self._target_pixels_per_point

    @target_pixels_per_point.setter
    def target_pixels_per_point(self, value: float) -> None:
        target = _require_positive_finite_float(value, "target_pixels_per_point")
        if target == self._target_pixels_per_point:
            return
        self._target_pixels_per_point = target
        self._refresh_viewport_budget()

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

    def _update_draw(self, scale_factor: Any, corner_pixels_displayed: Any, shape_threshold: Any) -> None:
        """Emit one normalized intrinsic viewport after napari draw bookkeeping."""
        super()._update_draw(scale_factor, corner_pixels_displayed, shape_threshold)
        state = _viewport_state_from_draw(
            displayed_axes=tuple(self._slice_input.displayed),
            corner_pixels_displayed=corner_pixels_displayed,
            shape_threshold=shape_threshold,
            world_to_data=self.world_to_data,
            hard_render_point_budget=self._hard_render_point_budget,
            target_pixels_per_point=self._target_pixels_per_point,
        )
        if state is not None:
            self._emit_viewport(state)

    def _refresh_viewport_budget(self) -> None:
        state = self._viewport_state
        if state is None:
            return
        self._emit_viewport(
            _viewport_state_with_budget(
                state,
                hard_render_point_budget=self._hard_render_point_budget,
                target_pixels_per_point=self._target_pixels_per_point,
            )
        )

    def _emit_viewport(self, state: TiledPointsViewportState) -> None:
        if state == self._viewport_state:
            return
        # Store before emitting so a synchronous redraw caused by a listener is
        # recognized as the same request instead of recursively scheduling it.
        self._viewport_state = state
        self.events.viewport(value=state)

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


def _require_positive_integer(value: int, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"`{name}` must be a positive integer.")
    return value


def _require_positive_finite_float(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"`{name}` must be a positive finite number.")
    return float(value)


def _validated_value_palette(
    value: npt.NDArray[np.uint8],
    *,
    value_count: int,
) -> npt.NDArray[np.uint8]:
    """Return an owned read-only copy of one complete RGBA value palette."""
    if (
        not isinstance(value, np.ndarray)
        or value.dtype != np.dtype(np.uint8)
        or value.ndim != 2
        or value.shape != (value_count, 4)
    ):
        raise ValueError(f"`value_palette` must be a uint8 array with shape ({value_count}, 4).")
    palette = np.array(value, dtype=np.uint8, order="C", copy=True)
    palette.flags.writeable = False
    return palette
