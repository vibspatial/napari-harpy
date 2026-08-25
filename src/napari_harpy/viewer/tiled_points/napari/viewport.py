"""Pure viewport conversion and budgeting for the tiled-points layer.

The layer owns viewport-state creation; the cache runtime only consumes the
normalized result. The normal camera-driven path is::

    napari camera pan, zoom, resize, or redraw
            |
            v
    TiledPointsLayerModel._update_draw()
            |
            v
    _viewport_state_from_draw()
            |
            v
    TiledPointsViewportState
            |
            v
    layer.events.viewport
            |
            v
    integration listener -> coordinator.submit_viewport()

Changing a render-budget setting while the camera is stationary instead calls
``_viewport_state_with_budget()``. It creates a replacement state with retained
intrinsic geometry and recalculated budget fields, then follows the same layer
event path.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import replace
from itertools import product
from typing import Any

import numpy as np
import numpy.typing as npt

from napari_harpy.viewer.tiled_points.contracts import TiledPointsViewportState


def _viewport_state_from_draw(
    *,
    displayed_axes: tuple[int, ...],
    corner_pixels_displayed: Any,
    shape_threshold: Any,
    world_to_data: Callable[[npt.NDArray[np.float64]], Any],
    hard_render_point_budget: int,
    target_pixels_per_point: float,
) -> TiledPointsViewportState | None:
    """Convert one napari draw callback into an immutable viewport state.

    Napari supplies two opposite world-coordinate corners in layer-axis
    ``(y, x)`` order. All four combinations must be inverse-transformed because
    rotation or shear can move either remaining corner to a data-axis extremum.
    Degenerate transient viewboxes produce no geometry and therefore no request.

    ``world_to_data`` is the caller's bound ``Layer.world_to_data`` method. It
    inverse-transforms each world-space viewport corner into this layer's
    intrinsic data coordinates while keeping this conversion helper independent
    of the complete napari layer object.
    """
    axes = tuple(int(axis) for axis in displayed_axes)
    if len(axes) != 2 or len(set(axes)) != 2 or any(axis < 0 for axis in axes):
        raise ValueError("A tiled-points draw must contain two unique displayed axes.")

    world_bounds = np.asarray(corner_pixels_displayed, dtype=np.float64)
    if world_bounds.shape != (2, 2) or not bool(np.isfinite(world_bounds).all()):
        raise ValueError("`corner_pixels_displayed` must contain two finite two-dimensional world corners.")

    canvas_shape = np.asarray(shape_threshold, dtype=np.float64)
    if canvas_shape.shape != (2,) or not bool(np.isfinite(canvas_shape).all()):
        raise ValueError("`shape_threshold` must contain two finite canvas dimensions.")
    if bool((canvas_shape <= 0).any()):
        return None
    canvas_height = math.ceil(float(canvas_shape[0]))
    canvas_width = math.ceil(float(canvas_shape[1]))

    world_corners = np.asarray(tuple(product(*world_bounds.T)), dtype=np.float64)
    # napari converts one world position at a time; passing an (N, 2)
    # array would be interpreted as one longer position rather than a batch.
    data_corners = np.asarray(tuple(world_to_data(corner) for corner in world_corners), dtype=np.float64)
    if data_corners.shape != (4, 2) or not bool(np.isfinite(data_corners).all()):
        raise ValueError("The inverse layer transform must return four finite two-dimensional data coordinates.")

    data_min = data_corners.min(axis=0)
    data_max = data_corners.max(axis=0)
    if bool((data_max <= data_min).any()):
        return None

    # Napari data axes are (y, x); the cache reader consumes (x, y).
    screen_density_budget = _screen_density_budget(
        canvas_width,
        canvas_height,
        target_pixels_per_point,
    )
    return TiledPointsViewportState(
        displayed_axes=(axes[0], axes[1]),
        x_min=float(data_min[1]),
        y_min=float(data_min[0]),
        x_max=float(data_max[1]),
        y_max=float(data_max[0]),
        canvas_width=canvas_width,
        canvas_height=canvas_height,
        hard_render_point_budget=hard_render_point_budget,
        screen_density_budget=screen_density_budget,
    )


def _viewport_state_with_budget(
    state: TiledPointsViewportState,
    *,
    hard_render_point_budget: int,
    target_pixels_per_point: float,
) -> TiledPointsViewportState:
    """Recalculate the policy fields of a retained viewport state."""
    screen_density_budget = _screen_density_budget(
        state.canvas_width,
        state.canvas_height,
        target_pixels_per_point,
    )
    return replace(
        state,
        hard_render_point_budget=hard_render_point_budget,
        screen_density_budget=screen_density_budget,
    )


def _screen_density_budget(canvas_width: int, canvas_height: int, target_pixels_per_point: float) -> int:
    return max(1, math.floor(canvas_width * canvas_height / target_pixels_per_point))
