from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyarrow as pa

from napari_harpy.core.multi_scale_cache_points.build_plan import _LevelBuildPlan, _LevelKind
from napari_harpy.core.multi_scale_cache_points.sampling import _select_sampled_tile_indices
from napari_harpy.core.multi_scale_cache_points.writer.support import _POINT_PAYLOAD_SCHEMA


@dataclass(frozen=True)
class _FinerLevelTile:
    """Hold one finer tile's logical coordinates and in-memory point payload."""

    tile_x: int
    tile_y: int
    points: pa.Table

    def __post_init__(self) -> None:
        if not isinstance(self.tile_x, int) or isinstance(self.tile_x, bool) or self.tile_x < 0:
            raise ValueError("`tile_x` must be a non-negative integer.")
        if not isinstance(self.tile_y, int) or isinstance(self.tile_y, bool) or self.tile_y < 0:
            raise ValueError("`tile_y` must be a non-negative integer.")
        if not isinstance(self.points, pa.Table):
            raise ValueError("`points` must be a PyArrow table.")
        if self.points.num_rows == 0:
            raise ValueError("A finer-level tile must contain at least one point.")
        if not self.points.schema.equals(_POINT_PAYLOAD_SCHEMA, check_metadata=False):
            raise ValueError("A finer-level tile has an incompatible point-payload schema.")


def _assemble_and_sample_coarser_tile(
    finer_tiles: tuple[_FinerLevelTile, ...],
    *,
    finer_level: _LevelBuildPlan,
    coarser_level: _LevelBuildPlan,
    coarser_tile_x: int,
    coarser_tile_y: int,
) -> pa.Table:
    """Rebase immediate-finer tiles and sample one coarser spatial tile.

    One coarser tile receives candidates from one through four nonempty tiles
    at the immediately finer level. Their relative coordinates are rebased
    into the coarser tile, after which the shared value-neutral sampler selects
    at most the coarser level's planned capacity. The returned four-column
    payload is ordered by ``point_id`` and remains a subset of the supplied
    candidates.
    """
    capacity = _validate_spatial_level_pair(
        finer_level,
        coarser_level,
        coarser_tile_x=coarser_tile_x,
        coarser_tile_y=coarser_tile_y,
    )
    ordered_finer_tiles = _validate_and_order_finer_tiles(
        finer_tiles,
        finer_level=finer_level,
        coarser_tile_x=coarser_tile_x,
        coarser_tile_y=coarser_tile_y,
    )

    # For a fully occupied coarser tile this is conceptually
    # (table_with_offset_0_0, table_with_offset_1_0,
    #  table_with_offset_0_1, table_with_offset_1_1). Sparse or edge tiles may
    # contribute fewer tables.
    rebased_tables = tuple(
        _rebase_finer_tile(
            finer_tile,
            finer_tile_size=finer_level.tile_size,
            coarser_tile_x=coarser_tile_x,
            coarser_tile_y=coarser_tile_y,
        )
        for finer_tile in ordered_finer_tiles
    )
    candidates = pa.concat_tables(rebased_tables)
    selected_indices = _select_sampled_tile_indices(
        candidates["x_rel"].combine_chunks().to_numpy(zero_copy_only=False),
        candidates["y_rel"].combine_chunks().to_numpy(zero_copy_only=False),
        candidates["point_id"].combine_chunks().to_numpy(zero_copy_only=False),
        level=coarser_level.level,
        tile_x=coarser_tile_x,
        tile_y=coarser_tile_y,
        tile_size=coarser_level.tile_size,
        target=capacity,
    )
    sampled = candidates.take(pa.array(selected_indices, type=pa.int64()))
    expected_rows = min(candidates.num_rows, capacity)
    if sampled.num_rows != expected_rows:
        raise ValueError("The sampled spatial tile does not match its planned capacity.")
    return sampled


def _validate_spatial_level_pair(
    finer_level: _LevelBuildPlan,
    coarser_level: _LevelBuildPlan,
    *,
    coarser_tile_x: int,
    coarser_tile_y: int,
) -> int:
    if finer_level.kind not in {_LevelKind.BRIDGE, _LevelKind.SPATIAL}:
        raise ValueError("The finer level must be a sampled Bridge or spatial level.")
    if coarser_level.kind is not _LevelKind.SPATIAL:
        raise ValueError("The coarser level must be a spatial level.")
    if coarser_level.level != finer_level.level + 1:
        raise ValueError("The coarser level must immediately follow the finer level.")
    if coarser_level.tile_size != 2 * finer_level.tile_size:
        raise ValueError("The coarser tile size must be twice the finer tile size.")
    _require_grid_coordinate(coarser_tile_x, "coarser_tile_x", grid_size=coarser_level.grid_width)
    _require_grid_coordinate(coarser_tile_y, "coarser_tile_y", grid_size=coarser_level.grid_height)
    capacity = coarser_level.max_points_per_tile
    if capacity is None:  # guarded by the level-plan contract
        raise ValueError("The coarser spatial level must have a per-tile capacity.")
    return capacity


def _validate_and_order_finer_tiles(
    finer_tiles: tuple[_FinerLevelTile, ...],
    *,
    finer_level: _LevelBuildPlan,
    coarser_tile_x: int,
    coarser_tile_y: int,
) -> tuple[_FinerLevelTile, ...]:
    """Validate complete contributing finer tiles and return them in tile order.

    ``finer_tiles`` contains one through four reconstructed logical tiles from
    ``finer_level``. Their coordinates must be unique, lie inside the finer
    grid, and map to the requested coarser tile. The returned tuple is ordered
    deterministically by ``(tile_y, tile_x)`` for subsequent concatenation.
    """
    if not isinstance(finer_tiles, tuple) or not 1 <= len(finer_tiles) <= 4:
        raise ValueError("`finer_tiles` must contain one through four tiles.")

    coordinates: set[tuple[int, int]] = set()
    for finer_tile in finer_tiles:
        if not isinstance(finer_tile, _FinerLevelTile):
            raise ValueError("Every finer tile must be a _FinerLevelTile.")
        _require_grid_coordinate(finer_tile.tile_x, "finer tile_x", grid_size=finer_level.grid_width)
        _require_grid_coordinate(finer_tile.tile_y, "finer tile_y", grid_size=finer_level.grid_height)
        coordinates.add((finer_tile.tile_y, finer_tile.tile_x))
        if finer_tile.tile_x // 2 != coarser_tile_x or finer_tile.tile_y // 2 != coarser_tile_y:
            raise ValueError("A finer tile does not contribute to the requested coarser tile.")
    if len(coordinates) != len(finer_tiles):
        raise ValueError("Finer tile coordinates must be unique.")
    return tuple(sorted(finer_tiles, key=lambda tile: (tile.tile_y, tile.tile_x)))


def _rebase_finer_tile(
    finer_tile: _FinerLevelTile,
    *,
    finer_tile_size: int,
    coarser_tile_x: int,
    coarser_tile_y: int,
) -> pa.Table:
    """Express one finer tile's points in the containing coarser tile's frame.

    A coarser tile has twice the finer tile edge, so each contributing finer
    tile occupies one quadrant identified by an x/y offset of zero or one::

        coarser tile coordinates

        x: 0 ---------------- tile_size
           +--------+--------+
           | offset | offset |
           | (0, 0) | (1, 0) |
           +--------+--------+
           | offset | offset |
           | (0, 1) | (1, 1) |
           +--------+--------+

    Adding that quadrant offset to each finer-relative coordinate places all
    contributing points in one shared coarser-relative coordinate frame. This
    is required before the combined candidates can be sampled as one tile.
    """
    x_rel = finer_tile.points["x_rel"].combine_chunks().to_numpy(zero_copy_only=False).astype(np.float64)
    y_rel = finer_tile.points["y_rel"].combine_chunks().to_numpy(zero_copy_only=False).astype(np.float64)
    if (
        not bool(np.isfinite(x_rel).all())
        or not bool(np.isfinite(y_rel).all())
        or bool((x_rel < 0).any())
        or bool((x_rel > finer_tile_size).any())
        or bool((y_rel < 0).any())
        or bool((y_rel > finer_tile_size).any())
    ):
        raise ValueError("Finer-tile coordinates must be finite and lie within the finer tile.")

    tile_offset_x = finer_tile.tile_x - 2 * coarser_tile_x
    tile_offset_y = finer_tile.tile_y - 2 * coarser_tile_y
    coarser_x_rel = (tile_offset_x * finer_tile_size + x_rel).astype(np.float32)
    coarser_y_rel = (tile_offset_y * finer_tile_size + y_rel).astype(np.float32)
    return pa.Table.from_arrays(
        [
            pa.array(coarser_x_rel, type=pa.float32()),
            pa.array(coarser_y_rel, type=pa.float32()),
            finer_tile.points["value_id"].combine_chunks(),
            finer_tile.points["point_id"].combine_chunks(),
        ],
        schema=_POINT_PAYLOAD_SCHEMA,
    )


def _require_grid_coordinate(value: object, name: str, *, grid_size: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value < grid_size:
        raise ValueError(f"`{name}` must be an integer inside the planned grid.")
