from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest

from napari_harpy.core.multi_scale_cache_points.models import PointsBounds, ValidatedPointsSource
from napari_harpy.core.multi_scale_cache_points_zarr.build_plan import (
    _LevelBuildPlan,
    _LevelKind,
    _plan_points_cache,
    _PointsCacheBuildPlan,
)


class _ForbiddenSource:
    @property
    def parquet_path(self) -> object:
        pytest.fail("The pure planner accessed the canonical source path.")


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
            source=_ForbiddenSource(),
        ),
    )


def _level(**overrides: object) -> _LevelBuildPlan:
    values: dict[str, object] = {
        "level": 0,
        "kind": _LevelKind.EXACT,
        "tile_size": 512,
        "grid_width": 1,
        "grid_height": 1,
        "max_points_per_tile": None,
        "point_count_upper_bound": 10,
    }
    values.update(overrides)
    return _LevelBuildPlan(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("level", True),
        ("level", 2**15),
        ("tile_size", 0),
        ("grid_width", 2**32 + 1),
        ("grid_height", 0),
        ("point_count_upper_bound", 2**63),
    ],
)
def test_level_plan_rejects_out_of_range_fields(field: str, value: object) -> None:
    with pytest.raises(ValueError, match=field):
        _level(**{field: value})


@pytest.mark.parametrize(
    ("kind", "capacity"),
    [
        (_LevelKind.EXACT, 4_096),
        (_LevelKind.BRIDGE, None),
        (_LevelKind.SPATIAL, 0),
        ("bridge", 4_096),
    ],
)
def test_level_plan_requires_kind_appropriate_capacity(kind: object, capacity: object) -> None:
    with pytest.raises(ValueError, match="kind|capacity|max_points_per_tile"):
        _level(kind=kind, max_points_per_tile=capacity)


def test_exact_only_plan_uses_aligned_origins_without_source_access() -> None:
    validated = _validated(row_count=80_000, x_min=-1.0, x_max=512.0, y_min=50.0, y_max=550.0)

    plan = _plan_points_cache(validated, leaf_tile_size=512, overview_point_budget=100_000)

    assert (plan.x_origin, plan.y_origin) == (-512.0, 0.0)
    assert len(plan.levels) == 1
    assert plan.levels[0] == _level(grid_width=3, grid_height=2, point_count_upper_bound=80_000)


def test_bridge_is_terminal_when_its_upper_bound_fits() -> None:
    plan = _plan_points_cache(
        _validated(row_count=200_000, x_max=1_535.0, y_max=1_535.0),
        leaf_tile_size=512,
        overview_point_budget=100_000,
    )

    assert tuple(level.kind for level in plan.levels) == (_LevelKind.EXACT, _LevelKind.BRIDGE)
    bridge = plan.levels[1]
    assert (bridge.tile_size, bridge.grid_width, bridge.grid_height) == (512, 3, 3)
    assert bridge.max_points_per_tile == 4_096
    assert bridge.point_count_upper_bound == 9 * 4_096


def test_plan_builds_spatial_levels_with_doubled_tiles_and_capacities() -> None:
    plan = _plan_points_cache(
        _validated(row_count=1_000_000, x_max=4_095.0, y_max=4_095.0),
        leaf_tile_size=512,
        overview_point_budget=100_000,
    )

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


def test_plan_clamps_a_terminal_single_tile_to_the_overview_budget() -> None:
    plan = _plan_points_cache(
        _validated(row_count=1_000_000, x_max=32_767.0),
        leaf_tile_size=512,
        overview_point_budget=100_000,
    )

    assert tuple(level.grid_width for level in plan.levels) == (64, 64, 32, 16, 8, 4, 2, 1)
    terminal = plan.levels[-1]
    assert terminal.max_points_per_tile == 100_000
    assert terminal.point_count_upper_bound == 100_000


@pytest.mark.parametrize(
    ("bounds", "origin", "grid"),
    [
        (PointsBounds(0.0, 511.0, 0.0, 511.0), (0.0, 0.0), (1, 1)),
        (PointsBounds(0.0, 512.0, 0.0, 1_024.0), (0.0, 0.0), (2, 3)),
        (PointsBounds(-600.0, -1.0, -1_025.0, -1.0), (-1_024.0, -1_536.0), (2, 3)),
    ],
)
def test_plan_grid_geometry(bounds: PointsBounds, origin: tuple[float, float], grid: tuple[int, int]) -> None:
    validated = cast(ValidatedPointsSource, SimpleNamespace(row_count=1, bounds=bounds))

    plan = _plan_points_cache(validated, leaf_tile_size=512, overview_point_budget=1)

    assert (plan.x_origin, plan.y_origin) == origin
    assert (plan.levels[0].grid_width, plan.levels[0].grid_height) == grid


@pytest.mark.parametrize(
    ("argument", "value"),
    [
        ("leaf_tile_size", 0),
        ("leaf_tile_size", True),
        ("overview_point_budget", -1),
        ("overview_point_budget", 1.5),
    ],
)
def test_planner_rejects_invalid_arguments(argument: str, value: object) -> None:
    arguments: dict[str, object] = {"leaf_tile_size": 512, "overview_point_budget": 100}
    arguments[argument] = value

    with pytest.raises(ValueError, match=argument):
        _plan_points_cache(_validated(row_count=1), **arguments)  # type: ignore[arg-type]


def test_planner_rejects_row_count_and_grid_overflow() -> None:
    with pytest.raises(ValueError, match="row_count"):
        _plan_points_cache(_validated(row_count=2**63), leaf_tile_size=512, overview_point_budget=100)
    with pytest.raises(ValueError, match="grid"):
        _plan_points_cache(
            _validated(row_count=1, x_max=float(2**32)),
            leaf_tile_size=1,
            overview_point_budget=1,
        )


def test_complete_plan_rejects_invalid_level_relationships() -> None:
    exact = _level(point_count_upper_bound=100)
    wrong_bridge = _level(
        level=1,
        kind=_LevelKind.BRIDGE,
        max_points_per_tile=4_096,
        tile_size=1_024,
        point_count_upper_bound=50,
    )
    with pytest.raises(ValueError, match="identical"):
        _PointsCacheBuildPlan(0.0, 0.0, 512, 50, (exact, wrong_bridge))

    bridge = _level(
        level=1,
        kind=_LevelKind.BRIDGE,
        max_points_per_tile=4_096,
        point_count_upper_bound=50,
    )
    wrong_spatial = _level(
        level=2,
        kind=_LevelKind.SPATIAL,
        max_points_per_tile=8_192,
        tile_size=1_024,
        grid_width=2,
        point_count_upper_bound=40,
    )
    with pytest.raises(ValueError, match="halved"):
        _PointsCacheBuildPlan(0.0, 0.0, 512, 40, (exact, bridge, wrong_spatial))
