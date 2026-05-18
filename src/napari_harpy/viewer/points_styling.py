from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from napari.layers import Points

from napari_harpy.core.class_palette import default_categorical_colors

if TYPE_CHECKING:
    from napari_harpy._points_value_index import PointsValueSelection

POINTS_SELECTION_SOLID_COLOR = "#00FFFF"
POINTS_SELECTION_MAX_CATEGORICAL_COLORS = 102
POINTS_SELECTION_DEFAULT_SIZE = 5.0
_POINTS_SIZE_SYNC_CALLBACK_ATTR = "_harpy_points_size_sync_callback"


@dataclass(frozen=True)
class PointsStyleResult:
    """Describe how one points value selection was styled."""

    color_mode: Literal["solid", "categorical"]
    categorical_coloring_disabled: bool
    selected_value_count: int
    categorical_limit: int


@dataclass(frozen=True)
class PointsLayerResult(PointsStyleResult):
    """Describe applying an already loaded points selection to a viewer layer.

    This is named ``PointsLayerResult`` rather than ``PointsLoadResult`` because
    Dask-backed point loading is owned by the controller. This result only
    describes the napari layer that was created or updated from that loaded
    selection.
    """

    layer: Points
    created: bool


def build_points_selection_layer_name(
    points_name: str,
    index_column: str,
    selection: PointsValueSelection,
) -> str:
    """Return the user-facing layer name for one points value selection."""
    if len(selection.selected_values) == 0:
        return f"{points_name}: no {index_column} values"
    if selection.selection_mode == "all":
        return f"{points_name}: all {index_column} values"
    if len(selection.selected_values) == 1:
        return f"{points_name}: {index_column}={selection.selected_values[0]}"
    return f"{points_name}: {len(selection.selected_values)} {index_column} values"


def apply_points_selection_style(
    layer: Points,
    selection: PointsValueSelection,
    *,
    point_size: Any | None = None,
) -> PointsStyleResult:
    """Apply points value-selection styling."""
    resolved_point_size = _coerce_point_size(point_size, default=POINTS_SELECTION_DEFAULT_SIZE)
    layer.current_size = resolved_point_size
    layer.size = resolved_point_size
    layer.opacity = 0.8
    layer.symbol = "disc"
    layer.border_width = 0
    _connect_current_size_to_global_point_size(layer)

    selected_value_count = len(selection.selected_values)
    if selected_value_count < 2 or selected_value_count > POINTS_SELECTION_MAX_CATEGORICAL_COLORS:
        layer.face_color = POINTS_SELECTION_SOLID_COLOR
        return PointsStyleResult(
            color_mode="solid",
            categorical_coloring_disabled=selected_value_count > POINTS_SELECTION_MAX_CATEGORICAL_COLORS,
            selected_value_count=selected_value_count,
            categorical_limit=POINTS_SELECTION_MAX_CATEGORICAL_COLORS,
        )
    else:
        layer.face_color_cycle = default_categorical_colors(selected_value_count)
        layer.face_color = selection.index_column

    return PointsStyleResult(
        color_mode="categorical",
        categorical_coloring_disabled=False,
        selected_value_count=selected_value_count,
        categorical_limit=POINTS_SELECTION_MAX_CATEGORICAL_COLORS,
    )


def _connect_current_size_to_global_point_size(layer: Points) -> None:
    if getattr(layer, _POINTS_SIZE_SYNC_CALLBACK_ATTR, None) is not None:
        return

    def _sync_current_size_to_all_points(_event: Any | None = None) -> None:
        point_size = _coerce_point_size(layer.current_size, default=POINTS_SELECTION_DEFAULT_SIZE)
        layer.size = point_size

    layer.events.current_size.connect(_sync_current_size_to_all_points)
    setattr(layer, _POINTS_SIZE_SYNC_CALLBACK_ATTR, _sync_current_size_to_all_points)


def _coerce_point_size(value: Any | None, *, default: float) -> float:
    if value is None:
        return default

    if isinstance(value, np.ndarray):
        if value.size == 0:
            return default
        value = value.reshape(-1)[-1]
    elif isinstance(value, list | tuple):
        if not value:
            return default
        value = value[-1]

    try:
        point_size = float(value)
    except (TypeError, ValueError):
        return default
    return point_size if point_size > 0 else default
