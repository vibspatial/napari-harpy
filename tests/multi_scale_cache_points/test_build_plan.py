from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest

from napari_harpy.core.multi_scale_cache_points.build_plan import (
    _LevelBuildPlan,
    _LevelKind,
    _plan_points_cache,
)
from napari_harpy.core.multi_scale_cache_points.models import PointsBounds, ValidatedPointsSource


def _validated(
    *,
    row_count: int,
    x_min: float = 0.0,
    x_max: float = 511.0,
    y_min: float = 0.0,
    y_max: float = 511.0,
) -> ValidatedPointsSource:
    return cast(
        ValidatedPointsSource,
        SimpleNamespace(
            row_count=row_count,
            bounds=PointsBounds(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max),
        ),
    )


@pytest.mark.parametrize(
    ("kind", "capacity"),
    [
        (_LevelKind.EXACT, 4_096),
        (_LevelKind.BRIDGE, None),
        ("bridge", 4_096),
    ],
)
def test_level_plan_requires_kind_appropriate_capacity(kind: object, capacity: int | None) -> None:
    with pytest.raises(ValueError, match="kind|capacity|max_points_per_tile"):
        _LevelBuildPlan(
            level=0,
            kind=kind,  # type: ignore[arg-type]
            tile_size=512,
            grid_width=1,
            grid_height=1,
            max_points_per_tile=capacity,
            point_count_upper_bound=1,
        )


@pytest.mark.parametrize(
    ("argument", "value"),
    [
        ("leaf_tile_size", 0),
        ("leaf_tile_size", True),
        ("overview_point_budget", -1),
        ("overview_point_budget", 1.5),
    ],
)
def test_plan_rejects_invalid_arguments(argument: str, value: object) -> None:
    arguments: dict[str, object] = {
        "leaf_tile_size": 512,
        "overview_point_budget": 100_000,
    }
    arguments[argument] = value

    with pytest.raises(ValueError, match=argument):
        _plan_points_cache(_validated(row_count=1), **arguments)  # type: ignore[arg-type]


def test_plan_returns_deterministic_exact_only_level() -> None:
    validated = _validated(
        row_count=80_000,
        x_min=-1.0,
        x_max=512.0,
        y_min=50.0,
        y_max=550.0,
    )

    plan = _plan_points_cache(validated, leaf_tile_size=512, overview_point_budget=100_000)

    assert plan == _plan_points_cache(validated, leaf_tile_size=512, overview_point_budget=100_000)
    assert (plan.x_origin, plan.y_origin) == (-512.0, 0.0)
    assert plan.leaf_tile_size == 512
    assert plan.overview_point_budget == 100_000
    assert len(plan.levels) == 1
    exact = plan.levels[0]
    assert (
        exact.level,
        exact.kind,
        exact.tile_size,
        exact.grid_width,
        exact.grid_height,
        exact.max_points_per_tile,
        exact.point_count_upper_bound,
        exact.relative_directory,
    ) == (0, _LevelKind.EXACT, 512, 3, 2, None, 80_000, "levels/level_0")


def test_bridge_is_terminal_when_its_complete_upper_bound_fits() -> None:
    validated = _validated(row_count=200_000, x_max=1_535.0, y_max=1_535.0)

    plan = _plan_points_cache(validated, leaf_tile_size=512, overview_point_budget=100_000)

    assert tuple(level.kind for level in plan.levels) == (_LevelKind.EXACT, _LevelKind.BRIDGE)
    assert tuple(level.level for level in plan.levels) == (0, 1)
    bridge = plan.levels[1]
    assert (bridge.tile_size, bridge.grid_width, bridge.grid_height) == (512, 3, 3)
    assert bridge.max_points_per_tile == 4_096
    assert bridge.point_count_upper_bound == 9 * 4_096


def test_plan_builds_spatial_levels_until_a_regular_level_fits() -> None:
    validated = _validated(row_count=1_000_000, x_max=4_095.0, y_max=4_095.0)

    plan = _plan_points_cache(validated, leaf_tile_size=512, overview_point_budget=100_000)

    assert tuple(level.kind for level in plan.levels) == (
        _LevelKind.EXACT,
        _LevelKind.BRIDGE,
        _LevelKind.SPATIAL,
        _LevelKind.SPATIAL,
    )
    assert tuple(level.tile_size for level in plan.levels) == (512, 512, 1_024, 2_048)
    assert tuple(level.max_points_per_tile for level in plan.levels) == (None, 4_096, 8_192, 16_384)
    assert tuple(level.point_count_upper_bound for level in plan.levels) == (
        1_000_000,
        262_144,
        131_072,
        65_536,
    )
    assert (plan.levels[-1].grid_width, plan.levels[-1].grid_height) == (2, 2)


def test_plan_clamps_only_the_terminal_one_tile_level() -> None:
    validated = _validated(row_count=1_000_000, x_max=32_767.0)

    plan = _plan_points_cache(validated, leaf_tile_size=512, overview_point_budget=100_000)

    assert tuple(level.level for level in plan.levels) == tuple(range(8))
    assert tuple(level.grid_width for level in plan.levels) == (64, 64, 32, 16, 8, 4, 2, 1)
    assert all(level.grid_height == 1 for level in plan.levels)
    assert all(level.point_count_upper_bound == 262_144 for level in plan.levels[1:-1])
    terminal = plan.levels[-1]
    assert terminal.kind is _LevelKind.SPATIAL
    assert terminal.tile_size == 32_768
    assert terminal.max_points_per_tile == 100_000
    assert terminal.point_count_upper_bound == 100_000


@pytest.mark.parametrize(
    ("bounds", "expected_origin", "expected_grid"),
    [
        (PointsBounds(0.0, 511.0, 0.0, 511.0), (0.0, 0.0), (1, 1)),
        (PointsBounds(100.0, 700.0, 130.0, 1_200.0), (0.0, 0.0), (2, 3)),
        (PointsBounds(0.0, 512.0, 0.0, 1_024.0), (0.0, 0.0), (2, 3)),
        (PointsBounds(-600.0, -1.0, -1_025.0, -1.0), (-1_024.0, -1_536.0), (2, 3)),
    ],
)
def test_plan_uses_aligned_half_open_grid_geometry(
    bounds: PointsBounds,
    expected_origin: tuple[float, float],
    expected_grid: tuple[int, int],
) -> None:
    validated = cast(ValidatedPointsSource, SimpleNamespace(row_count=1, bounds=bounds))

    plan = _plan_points_cache(validated, leaf_tile_size=512, overview_point_budget=1)

    assert (plan.x_origin, plan.y_origin) == expected_origin
    assert (plan.levels[0].grid_width, plan.levels[0].grid_height) == expected_grid


def test_plan_rejects_grid_outside_uint32_tile_coordinates() -> None:
    validated = _validated(row_count=1, x_max=float(2**32), y_max=0.0)

    with pytest.raises(ValueError, match="Grid width"):
        _plan_points_cache(validated, leaf_tile_size=1, overview_point_budget=1)
