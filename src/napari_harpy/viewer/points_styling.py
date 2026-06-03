from __future__ import annotations

from collections.abc import Sequence
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
POINTS_SELECTION_DEFAULT_SYMBOL = "disc"
_POINTS_SIZE_SYNC_CALLBACK_ATTR = "_harpy_points_size_sync_callback"
_POINTS_SYMBOL_SYNC_CALLBACK_ATTR = "_harpy_points_symbol_sync_callback"
_POINTS_FACE_COLOR_SYNC_CALLBACK_ATTR = "_harpy_points_face_color_sync_callback"
_POINTS_FACE_COLOR_OVERRIDE_ATTR = "_harpy_points_face_color_override"


@dataclass(frozen=True)
class PointsStyleResult:
    """Describe how one points value selection was styled."""

    color_mode: Literal["solid", "categorical"]
    categorical_coloring_disabled: bool
    selected_value_count: int
    categorical_limit: int


@dataclass(frozen=True)
class PointsLoadResult(PointsStyleResult):
    """Describe applying an already loaded points selection to a viewer layer.

    Dask-backed point selection loading is owned by the controller. This result
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
    point_symbol: Any | None = None,
    point_face_color: Any | None = None,
    categorical_colors: Sequence[str] | None = None,
) -> PointsStyleResult:
    """Apply points value-selection styling.

    ``point_face_color`` is a preserved napari UI color from the previous
    layer. It is used for solid fallback layers and single-category
    categorical layers, where one color cannot erase a multi-value palette.
    """
    resolved_point_size = _coerce_point_size(point_size, default=POINTS_SELECTION_DEFAULT_SIZE)
    resolved_point_symbol = POINTS_SELECTION_DEFAULT_SYMBOL if point_symbol is None else point_symbol
    layer.current_size = resolved_point_size
    layer.size = resolved_point_size
    layer.opacity = 0.8
    layer.current_symbol = resolved_point_symbol
    layer.symbol = layer.current_symbol
    layer.border_width = 0
    _connect_current_size_to_global_point_size(layer)
    connect_current_symbol_to_global_point_symbol(layer)

    selected_value_count = len(selection.selected_values)
    if selected_value_count == 0 or selected_value_count > POINTS_SELECTION_MAX_CATEGORICAL_COLORS:
        layer.current_face_color = POINTS_SELECTION_SOLID_COLOR if point_face_color is None else point_face_color
        layer.face_color = layer.current_face_color
        layer.current_border_color = layer.current_face_color
        layer.border_color = layer.face_color
        connect_current_face_color_to_global_point_face_color(layer)
        return PointsStyleResult(
            color_mode="solid",
            categorical_coloring_disabled=selected_value_count > POINTS_SELECTION_MAX_CATEGORICAL_COLORS,
            selected_value_count=selected_value_count,
            categorical_limit=POINTS_SELECTION_MAX_CATEGORICAL_COLORS,
        )
    else:
        layer.face_color_cycle = _resolve_points_categorical_color_mapping(
            selection,
            selected_value_count=selected_value_count,
            categorical_colors=categorical_colors,
        )
        layer.face_color = selection.index_column
        layer.border_color = layer.face_color
        if selected_value_count == 1:
            connect_current_face_color_to_global_point_face_color(layer)
            if point_face_color is not None:
                layer.current_face_color = point_face_color

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


def connect_current_symbol_to_global_point_symbol(layer: Points) -> None:
    """Sync napari's current point symbol control to all points in a layer."""
    if getattr(layer, _POINTS_SYMBOL_SYNC_CALLBACK_ATTR, None) is not None:
        return

    def _sync_current_symbol_to_all_points(_event: Any | None = None) -> None:
        layer.symbol = layer.current_symbol

    layer.events.current_symbol.connect(_sync_current_symbol_to_all_points)
    setattr(layer, _POINTS_SYMBOL_SYNC_CALLBACK_ATTR, _sync_current_symbol_to_all_points)


def connect_current_face_color_to_global_point_face_color(layer: Points) -> None:
    """Sync napari's current point face color control to all point colors."""
    if getattr(layer, _POINTS_FACE_COLOR_SYNC_CALLBACK_ATTR, None) is not None:
        return

    def _sync_current_face_color_to_all_points(_event: Any | None = None) -> None:
        setattr(layer, _POINTS_FACE_COLOR_OVERRIDE_ATTR, True)
        layer.face_color = layer.current_face_color
        layer.current_border_color = layer.current_face_color
        layer.border_color = layer.face_color

    layer.events.current_face_color.connect(_sync_current_face_color_to_all_points)
    setattr(layer, _POINTS_FACE_COLOR_SYNC_CALLBACK_ATTR, _sync_current_face_color_to_all_points)


def _resolve_points_categorical_color_mapping(
    selection: PointsValueSelection,
    *,
    selected_value_count: int,
    categorical_colors: Sequence[str] | None,
) -> list[str] | dict[str, str]:
    if categorical_colors is None:
        return default_categorical_colors(selected_value_count)

    colors = list(categorical_colors)
    if len(colors) != selected_value_count:
        raise ValueError("`categorical_colors` must match the number of selected values.")
    return dict(zip(selection.selected_values, colors, strict=True))


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
