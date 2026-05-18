from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from napari.layers import Points

from napari_harpy.core.class_palette import default_categorical_colors

if TYPE_CHECKING:
    from napari_harpy._points_value_index import PointsValueSelection

POINTS_SELECTION_SOLID_COLOR = "#00FFFF"
POINTS_SELECTION_MAX_CATEGORICAL_COLORS = 102


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
) -> PointsStyleResult:
    """Apply points value-selection styling."""
    layer.size = 5.0
    layer.opacity = 0.8
    layer.symbol = "disc"
    layer.border_width = 0

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
