"""Immutable viewer-side tiled-points layer values."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal
from uuid import UUID

DEFAULT_HARD_RENDER_POINT_BUDGET = 100_000
DEFAULT_TARGET_PIXELS_PER_POINT = 9.0


@dataclass(frozen=True)
class TiledPointsDatasetReference:
    """Identify one logical tiled-points dataset without storing point rows.

    Parameters
    ----------
    cache_generation_id
        UUID of the completed cache generation represented by the layer.
    points_name
        Name of the source SpatialData points element.
    value_column
        Source column represented by cache ``value_id`` rows.
    x_min, x_max, y_min, y_max
        Complete observed intrinsic-coordinate bounds of the cache.
    """

    cache_generation_id: str
    points_name: str
    value_column: str
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    def __post_init__(self) -> None:
        if not isinstance(self.cache_generation_id, str):
            raise ValueError("`cache_generation_id` must be a UUID string.")
        try:
            parsed = UUID(self.cache_generation_id)
        except ValueError as error:
            raise ValueError("`cache_generation_id` must be a UUID string.") from error
        if str(parsed) != self.cache_generation_id:
            raise ValueError("`cache_generation_id` must use canonical UUID spelling.")
        if not isinstance(self.points_name, str) or not self.points_name:
            raise ValueError("`points_name` must be a nonempty string.")
        if not isinstance(self.value_column, str) or not self.value_column:
            raise ValueError("`value_column` must be a nonempty string.")
        for name in ("x_min", "x_max", "y_min", "y_max"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
                raise ValueError(f"`{name}` must be a finite number.")
        if self.x_min > self.x_max or self.y_min > self.y_max:
            raise ValueError("Dataset minima must not exceed maxima.")


@dataclass(frozen=True)
class TiledPointsLayerStatus:
    """Describe the current cache-backed tiled-points display state."""

    level: int | None = None
    level_kind: Literal["exact", "bridge", "spatial"] | None = None
    rendered_point_count: int = 0
    rendered_tile_count: int = 0
    message: str = "Idle"
    sampled: bool = False
    omitted_value_ids: tuple[int, ...] = ()

    @property
    def level_label(self) -> str:
        """Return the canonical presentation label for the active cache level."""
        if self.level is None:
            return "—"
        if self.level_kind == "exact":
            return "Exact"
        if self.level_kind == "bridge":
            return "Bridge"
        return f"Spatial L{self.level}"

    def __post_init__(self) -> None:
        if (self.level is None) != (self.level_kind is None):
            raise ValueError("`level` and `level_kind` must either both be present or both be absent.")
        if self.level is not None:
            if not isinstance(self.level, int) or isinstance(self.level, bool) or self.level < 0:
                raise ValueError("`level` must be a nonnegative integer or None.")
            expected_kind = "exact" if self.level == 0 else "bridge" if self.level == 1 else "spatial"
            if self.level_kind != expected_kind:
                raise ValueError("`level_kind` does not match the serialized cache level.")
        if not isinstance(self.message, str) or not self.message:
            raise ValueError("`message` must be a nonempty string.")
        for name in ("rendered_point_count", "rendered_tile_count"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"`{name}` must be a nonnegative integer.")
        if not isinstance(self.sampled, bool):
            raise ValueError("`sampled` must be bool.")
        if (
            not isinstance(self.omitted_value_ids, tuple)
            or any(
                not isinstance(value_id, int) or isinstance(value_id, bool) or value_id < 0
                for value_id in self.omitted_value_ids
            )
            or tuple(sorted(set(self.omitted_value_ids))) != self.omitted_value_ids
        ):
            raise ValueError("`omitted_value_ids` must contain sorted unique nonnegative integers.")


@dataclass(frozen=True)
class TiledPointsViewportState:
    """Describe one normalized napari viewport in intrinsic cache coordinates.

    The hard and screen-density budgets are retained as viewer-side diagnostic
    evidence. ``effective_point_budget`` is their derived minimum. Later cache
    planning consumes only the intrinsic bounds and that effective budget.
    """

    displayed_axes: tuple[int, int]
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    canvas_width: int
    canvas_height: int
    hard_render_point_budget: int
    screen_density_budget: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.displayed_axes, tuple)
            or len(self.displayed_axes) != 2
            or any(not isinstance(axis, int) or isinstance(axis, bool) or axis < 0 for axis in self.displayed_axes)
            or len(set(self.displayed_axes)) != 2
        ):
            raise ValueError("`displayed_axes` must contain two unique nonnegative integers.")
        for name in ("x_min", "y_min", "x_max", "y_max"):
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value):
                raise ValueError(f"`{name}` must be a finite float.")
        if self.x_min >= self.x_max or self.y_min >= self.y_max:
            raise ValueError("Viewport bounds must have positive width and height.")
        for name in (
            "canvas_width",
            "canvas_height",
            "hard_render_point_budget",
            "screen_density_budget",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"`{name}` must be a positive integer.")

    @property
    def effective_point_budget(self) -> int:
        """Return the stricter of the hard and screen-density budgets."""
        return min(self.hard_render_point_budget, self.screen_density_budget)
