"""Plan the logical level hierarchy of a Zarr-backed points cache.

A typical multi-level plan is:

.. code-block:: text

    cache level 0: Exact       tile size S     uncapped
          |
          | sample within the same logical tiles
          v
    cache level 1: Bridge      tile size S     4,096 points/tile
          |
          | combine each 2 x 2 group of finer tiles and sample
          v
    cache level 2: Spatial 1   tile size 2S    8,192 points/tile
          |
          v
    cache level 3: Spatial 2   tile size 4S   16,384 points/tile
          |
         ...
          v
    terminal overview level    complete upper bound <= overview budget

``Spatial 1`` is the first spatial level but serialized cache level 2; cache
level 1 is reserved for Bridge. Exact and Bridge share tile geometry. Every
subsequent Spatial level doubles the preceding tile edge and scheduled
capacity, while each grid dimension becomes ``ceil(finer_dimension / 2)``.
The final one-tile level may clamp its capacity to the overview budget.

This hierarchy is logical and independent of physical Zarr arrays and chunks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from napari_harpy.core.multi_scale_cache_points_zarr.models import _INT16_MAX, _INT64_MAX, _require_integer_in_range
from napari_harpy.core.multi_scale_cache_points_zarr.source.models import PointsBounds, ValidatedPointsSource
from napari_harpy.core.multi_scale_cache_points_zarr.storage._paths import tile_major_level_path

BRIDGE_MAX_POINTS_PER_TILE = 4_096
_MAX_GRID_DIMENSION = 2**32


class _LevelKind(Enum):
    """Logical role of one independently planned Zarr-cache level."""

    EXACT = "exact"
    BRIDGE = "bridge"
    SPATIAL = "spatial"


@dataclass(frozen=True)
class _LevelBuildPlan:
    """Immutable logical construction plan for one Zarr-cache level.

    Parameters
    ----------
    level
        Non-negative serialized level number. Levels are numbered in
        construction order from the exact finest level ``0`` toward the
        sampled coarsest level.
    kind
        Logical role of the level: uncapped Exact membership, the sampled
        same-geometry Bridge, or a sampled Spatial level.
    tile_size
        Edge length of each square logical tile in intrinsic source-coordinate
        units. It is not a screen-pixel size.
    grid_width
        Number of logical tile columns required to cover the source x bounds
        when tiles of this level's ``tile_size`` are laid out starting at
        ``_PointsCacheBuildPlan.x_origin``. Valid ``tile_x`` indices are ``0``
        through ``grid_width - 1``; the grid may include empty tiles.
    grid_height
        Number of logical tile rows required to cover the source y bounds when
        tiles of this level's ``tile_size`` are laid out starting at
        ``_PointsCacheBuildPlan.y_origin``. Valid ``tile_y`` indices are ``0``
        through ``grid_height - 1``; the grid may include empty tiles.
    max_points_per_tile
        Maximum representatives stored in one logical tile. ``None`` means the
        Exact level is uncapped. Sampled levels require a positive integer.
        This logical sampling limit is independent of physical Zarr chunking.
    point_count_upper_bound
        Maximum possible representatives across the complete level. For Exact,
        this equals the validated source row count. For sampled levels, it is a
        conservative bound derived from the finer-level bound and the sum of
        per-tile limits; it is not the actual stored count.
    """

    level: int
    kind: _LevelKind
    tile_size: int
    grid_width: int
    grid_height: int
    max_points_per_tile: int | None
    point_count_upper_bound: int

    def __post_init__(self) -> None:
        _require_integer_in_range(self.level, "level", maximum=_INT16_MAX)
        if not isinstance(self.kind, _LevelKind):
            raise ValueError("`kind` must be a _LevelKind.")
        _require_integer_in_range(self.tile_size, "tile_size", minimum=1, maximum=_INT64_MAX)
        _require_integer_in_range(self.grid_width, "grid_width", minimum=1, maximum=_MAX_GRID_DIMENSION)
        _require_integer_in_range(self.grid_height, "grid_height", minimum=1, maximum=_MAX_GRID_DIMENSION)
        _require_integer_in_range(
            self.point_count_upper_bound,
            "point_count_upper_bound",
            minimum=1,
            maximum=_INT64_MAX,
        )
        if self.kind is _LevelKind.EXACT:
            if self.max_points_per_tile is not None:
                raise ValueError("An Exact level must not have a per-tile capacity.")
        else:
            _require_integer_in_range(
                self.max_points_per_tile,
                "max_points_per_tile",
                minimum=1,
                maximum=_INT64_MAX,
            )

    @property
    def relative_directory(self) -> str:
        """Return this level's cache-root-relative directory."""
        return tile_major_level_path(self.level)


@dataclass(frozen=True)
class _PointsCacheBuildPlan:
    """Immutable logical construction plan for a complete Zarr-backed cache.

    Parameters
    ----------
    x_origin
        X-coordinate anchor shared by every level. It is the greatest multiple
        of ``leaf_tile_size`` less than or equal to the validated source
        ``x_min``. Tile x indices are measured from this anchor, which keeps
        them non-negative and keeps adjacent finer and coarser grids aligned.
        It does not transform the source coordinates. For example,
        ``leaf_tile_size=512`` and ``x_min=600`` produce ``x_origin=512``.
    y_origin
        Y-axis counterpart of ``x_origin``, derived identically from the
        validated source ``y_min`` and used to calculate tile y indices.
    leaf_tile_size
        Edge length, in intrinsic source-coordinate units, of Exact and Bridge
        logical tiles. Every Spatial level doubles the preceding tile edge.
    overview_point_budget
        Maximum permitted representative count for the complete coarsest level.
        It is a whole-dataset construction limit, not the runtime viewport
        render budget.
    levels
        Immutable level plans ordered by ascending serialized level, from the
        exact finest level toward the terminal sampled coarsest level.
    """

    x_origin: float
    y_origin: float
    leaf_tile_size: int
    overview_point_budget: int
    levels: tuple[_LevelBuildPlan, ...]

    def __post_init__(self) -> None:
        _require_finite_float(self.x_origin, "x_origin")
        _require_finite_float(self.y_origin, "y_origin")
        _require_integer_in_range(self.leaf_tile_size, "leaf_tile_size", minimum=1, maximum=_INT64_MAX)
        _require_integer_in_range(
            self.overview_point_budget,
            "overview_point_budget",
            minimum=1,
            maximum=_INT64_MAX,
        )
        if not isinstance(self.levels, tuple) or not self.levels:
            raise ValueError("A cache build plan must contain Exact level 0.")
        if not all(isinstance(level, _LevelBuildPlan) for level in self.levels):
            raise ValueError("`levels` must be a tuple of _LevelBuildPlan values.")
        if self.levels[0].level != 0 or self.levels[0].kind is not _LevelKind.EXACT:
            raise ValueError("The first cache level must be Exact level 0.")
        if self.levels[0].tile_size != self.leaf_tile_size:
            raise ValueError("Exact tile size must equal `leaf_tile_size`.")
        if any(level.level != expected for expected, level in enumerate(self.levels)):
            raise ValueError("Cache levels must be consecutively numbered.")

        if len(self.levels) > 1:
            bridge = self.levels[1]
            exact = self.levels[0]
            if bridge.kind is not _LevelKind.BRIDGE:
                raise ValueError("Level 1 must be the Bridge level.")
            if (bridge.tile_size, bridge.grid_width, bridge.grid_height) != (
                exact.tile_size,
                exact.grid_width,
                exact.grid_height,
            ):
                raise ValueError("Exact and Bridge must have identical tile geometry.")

        for finer, coarser in zip(self.levels[1:], self.levels[2:], strict=False):
            if coarser.kind is not _LevelKind.SPATIAL:
                raise ValueError("Every level after Bridge must be spatial.")
            if coarser.tile_size != 2 * finer.tile_size:
                raise ValueError("Every spatial level must double the finer tile size.")
            if coarser.grid_width != math.ceil(finer.grid_width / 2) or coarser.grid_height != math.ceil(
                finer.grid_height / 2
            ):
                raise ValueError("Every spatial grid must be the halved immediate-finer grid.")

        if any(
            coarser.point_count_upper_bound > finer.point_count_upper_bound
            for finer, coarser in zip(self.levels, self.levels[1:], strict=False)
        ):
            raise ValueError("Level point-count upper bounds must not increase.")
        if self.levels[-1].point_count_upper_bound > self.overview_point_budget:
            raise ValueError("The terminal level must fit the overview point budget.")


def _plan_points_cache(
    validated: ValidatedPointsSource,
    *,
    leaf_tile_size: int,
    overview_point_budget: int,
) -> _PointsCacheBuildPlan:
    """Plan aligned Zarr-cache levels without reading source rows or writing files.

    The plan depends only on validated row count and coordinate bounds. It
    describes logical tile geometry and sampling limits; later construction
    slices are responsible for selecting points and encoding the Zarr payload.
    """
    _require_integer_in_range(leaf_tile_size, "leaf_tile_size", minimum=1, maximum=_INT64_MAX)
    _require_integer_in_range(overview_point_budget, "overview_point_budget", minimum=1, maximum=_INT64_MAX)
    row_count = _require_integer_in_range(validated.row_count, "row_count", minimum=1, maximum=_INT64_MAX)
    bounds = validated.bounds
    if not isinstance(bounds, PointsBounds):
        raise ValueError("`validated.bounds` must be PointsBounds.")

    x_origin = _aligned_origin(bounds.x_min, leaf_tile_size)
    y_origin = _aligned_origin(bounds.y_min, leaf_tile_size)
    grid_width, grid_height = _grid_shape(
        bounds,
        x_origin=x_origin,
        y_origin=y_origin,
        tile_size=leaf_tile_size,
    )
    exact = _LevelBuildPlan(
        level=0,
        kind=_LevelKind.EXACT,
        tile_size=leaf_tile_size,
        grid_width=grid_width,
        grid_height=grid_height,
        max_points_per_tile=None,
        point_count_upper_bound=row_count,
    )
    if row_count <= overview_point_budget:
        return _PointsCacheBuildPlan(
            x_origin=x_origin,
            y_origin=y_origin,
            leaf_tile_size=leaf_tile_size,
            overview_point_budget=overview_point_budget,
            levels=(exact,),
        )

    levels = [exact]
    kind = _LevelKind.BRIDGE
    tile_size = leaf_tile_size
    scheduled_capacity = BRIDGE_MAX_POINTS_PER_TILE
    while True:
        level = len(levels)
        if level > _INT16_MAX:
            raise ValueError("The planned level exceeds the supported int16 range.")
        grid_width, grid_height = _grid_shape(
            bounds,
            x_origin=x_origin,
            y_origin=y_origin,
            tile_size=tile_size,
        )
        capacity = scheduled_capacity

        # This level cannot retain more candidates than its finer input or more
        # than the sum of the per-tile limits across its complete logical grid.
        # For example, a 3-by-2 grid capped at 4,096 points per tile can retain
        # at most 3 * 2 * 4,096 = 24,576 points across the complete level.
        # The grid may include empty tiles, so this is a conservative upper bound.
        upper_bound = min(levels[-1].point_count_upper_bound, grid_width * grid_height * capacity)

        # Once one logical tile covers the complete dataset, increasing its size
        # cannot reduce the grid any further. If the normally scheduled capacity
        # still exceeds the whole-dataset overview budget, use that budget as the
        # tile's effective sampling limit. Planning does not sample points; later
        # cache construction enforces this limit. For a one-tile grid, the
        # per-tile capacity is also the upper bound for the complete level.
        if upper_bound > overview_point_budget and grid_width == 1 and grid_height == 1:
            capacity = overview_point_budget
            upper_bound = overview_point_budget

        levels.append(
            _LevelBuildPlan(
                level=level,
                kind=kind,
                tile_size=tile_size,
                grid_width=grid_width,
                grid_height=grid_height,
                max_points_per_tile=capacity,
                point_count_upper_bound=upper_bound,
            )
        )
        if upper_bound <= overview_point_budget:
            return _PointsCacheBuildPlan(
                x_origin=x_origin,
                y_origin=y_origin,
                leaf_tile_size=leaf_tile_size,
                overview_point_budget=overview_point_budget,
                levels=tuple(levels),
            )

        kind = _LevelKind.SPATIAL
        if tile_size > _INT64_MAX // 2 or scheduled_capacity > _INT64_MAX // 2:
            raise ValueError("Spatial tile size or capacity exceeds the supported int64 range.")
        tile_size *= 2
        scheduled_capacity *= 2


def _aligned_origin(minimum: float, leaf_tile_size: int) -> float:
    return float(math.floor(minimum / leaf_tile_size) * leaf_tile_size)


def _grid_shape(
    bounds: PointsBounds,
    *,
    x_origin: float,
    y_origin: float,
    tile_size: int,
) -> tuple[int, int]:
    grid_width = math.floor((bounds.x_max - x_origin) / tile_size) + 1
    grid_height = math.floor((bounds.y_max - y_origin) / tile_size) + 1
    if grid_width > _MAX_GRID_DIMENSION or grid_height > _MAX_GRID_DIMENSION:
        raise ValueError("The planned grid exceeds the supported uint32 tile-coordinate space.")
    return grid_width, grid_height


def _require_finite_float(value: object, name: str) -> float:
    if not isinstance(value, float) or not math.isfinite(value):
        raise ValueError(f"`{name}` must be a finite float.")
    return value
