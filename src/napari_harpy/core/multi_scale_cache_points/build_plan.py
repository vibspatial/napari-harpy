from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from napari_harpy.core.multi_scale_cache_points.models import PointsBounds, ValidatedPointsSource

_BRIDGE_MAX_POINTS_PER_TILE = 4_096
_MAX_GRID_DIMENSION = 2**32
_MAX_SERIALIZED_LEVEL = 2**15 - 1
_MAX_CACHE_COUNT = 2**63 - 1


class _LevelKind(Enum):
    """Logical role of one planned cache level."""

    EXACT = "exact"
    BRIDGE = "bridge"
    SPATIAL = "spatial"


@dataclass(frozen=True)
class _LevelBuildPlan:
    """Immutable logical construction plan for one cache level.

    Parameters
    ----------
    level
        Non-negative serialized level number. Levels are numbered in
        construction order from the exact finest level ``0`` toward the
        sampled coarsest level.
    kind
        Logical role of the level: uncapped exact membership, the sampled
        same-geometry bridge, or a sampled spatial level.
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
        exact level is uncapped. Sampled levels require a positive integer.
        This logical sampling limit is independent of physical Parquet row-group
        sharding.
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
        if not isinstance(self.kind, _LevelKind):
            raise ValueError("`kind` must be a _LevelKind.")
        if self.kind is _LevelKind.EXACT:
            if self.max_points_per_tile is not None:
                raise ValueError("An exact level must not have a per-tile capacity.")
        else:
            _require_positive_integer(self.max_points_per_tile, "max_points_per_tile")

    @property
    def relative_directory(self) -> str:
        """Return the cache-root-relative directory for this level."""
        return f"levels/level_{self.level}"


@dataclass(frozen=True)
class _PointsCacheBuildPlan:
    """Immutable logical construction plan for a complete points cache.

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
        logical tiles. Every spatial level doubles the preceding tile edge.
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
        if not self.levels:
            raise ValueError("A cache build plan must contain Exact level 0.")
        if self.levels[0].level != 0 or self.levels[0].kind is not _LevelKind.EXACT:
            raise ValueError("The first cache level must be serialized Exact level 0.")
        if any(level.level != expected for expected, level in enumerate(self.levels)):
            raise ValueError("Cache levels must be consecutively numbered in tuple order.")


def _plan_points_cache(
    validated: ValidatedPointsSource,
    *,
    leaf_tile_size: int,
    overview_point_budget: int,
) -> _PointsCacheBuildPlan:
    """Plan aligned cache levels without reading source rows or writing files."""
    _require_positive_integer(leaf_tile_size, "leaf_tile_size")
    _require_positive_integer(overview_point_budget, "overview_point_budget")
    if validated.row_count > _MAX_CACHE_COUNT:
        raise ValueError(f"Source row count exceeds the supported cache-count maximum of {_MAX_CACHE_COUNT}.")

    bounds = validated.bounds
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
        point_count_upper_bound=validated.row_count,
    )
    if validated.row_count <= overview_point_budget:
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
    scheduled_max_points_per_tile = _BRIDGE_MAX_POINTS_PER_TILE

    while True:
        level = len(levels)
        if level > _MAX_SERIALIZED_LEVEL:
            raise ValueError(f"Serialized level exceeds the supported int16 maximum of {_MAX_SERIALIZED_LEVEL}.")

        grid_width, grid_height = _grid_shape(
            bounds,
            x_origin=x_origin,
            y_origin=y_origin,
            tile_size=tile_size,
        )
        finer_upper_bound = levels[-1].point_count_upper_bound
        max_points_per_tile = scheduled_max_points_per_tile

        # This level cannot retain more candidates than its finer input or more
        # than the sum of the per-tile limits across its complete logical grid.
        # For example, a 3-by-2 grid capped at 4,096 points per tile can retain
        # at most 3 * 2 * 4,096 = 24,576 points across the complete level.
        # The grid may include empty tiles, so this is a conservative upper bound.
        point_count_upper_bound = min(
            finer_upper_bound,
            grid_width * grid_height * max_points_per_tile,
        )

        # Once one logical tile covers the complete dataset, increasing the tile
        # size cannot reduce the grid any further. If the normally scheduled
        # capacity still exceeds the whole-dataset overview budget, record that
        # budget as this planned tile's effective sampling limit. This planner
        # does not sample points; later cache construction enforces the limit.
        # With one tile, its effective per-tile limit (`max_points_per_tile`) is
        # also the `point_count_upper_bound` for the complete level.
        if point_count_upper_bound > overview_point_budget and grid_width == 1 and grid_height == 1:
            max_points_per_tile = overview_point_budget
            point_count_upper_bound = max_points_per_tile

        levels.append(
            _LevelBuildPlan(
                level=level,
                kind=kind,
                tile_size=tile_size,
                grid_width=grid_width,
                grid_height=grid_height,
                max_points_per_tile=max_points_per_tile,
                point_count_upper_bound=point_count_upper_bound,
            )
        )
        if point_count_upper_bound <= overview_point_budget:
            return _PointsCacheBuildPlan(
                x_origin=x_origin,
                y_origin=y_origin,
                leaf_tile_size=leaf_tile_size,
                overview_point_budget=overview_point_budget,
                levels=tuple(levels),
            )

        kind = _LevelKind.SPATIAL
        tile_size *= 2
        scheduled_max_points_per_tile *= 2


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
    if grid_width > _MAX_GRID_DIMENSION:
        raise ValueError(f"Grid width exceeds the supported uint32 tile-coordinate range of {_MAX_GRID_DIMENSION}.")
    if grid_height > _MAX_GRID_DIMENSION:
        raise ValueError(f"Grid height exceeds the supported uint32 tile-coordinate range of {_MAX_GRID_DIMENSION}.")
    return grid_width, grid_height


def _require_positive_integer(value: object, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"`{name}` must be a positive integer.")
