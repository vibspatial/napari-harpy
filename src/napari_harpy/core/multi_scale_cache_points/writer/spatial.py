from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from napari_harpy.core.multi_scale_cache_points.build_plan import (
    _LevelBuildPlan,
    _LevelKind,
    _PointsCacheBuildPlan,
)
from napari_harpy.core.multi_scale_cache_points.sampling import _select_sampled_tile_indices
from napari_harpy.core.multi_scale_cache_points.writer.models import (
    _BucketWriteResult,
    _IntermediateTileValueCountFile,
    _LevelWriteResult,
    _ManifestRow,
)
from napari_harpy.core.multi_scale_cache_points.writer.support import (
    _INTERMEDIATE_TILE_VALUE_COUNTS_DIRECTORY,
    _POINT_PAYLOAD_SCHEMA,
    DEFAULT_MAX_ROWS_PER_ROW_GROUP,
    _bucket_count_for_level,
    _IntermediateTileValueCountWriter,
    _read_logical_tile,
    _reconcile_level_results,
    _tile_bucket_ids,
    _validate_bucket_files,
)


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


@dataclass(frozen=True)
class _FinerTileDescriptor:
    """Describe the physical row groups of one complete logical finer tile."""

    shard_descriptors: tuple[_ManifestRow, ...]

    @property
    def tile_x(self) -> int:
        """Return the logical x index shared by every physical shard."""
        return self.shard_descriptors[0].tile_x

    @property
    def tile_y(self) -> int:
        """Return the logical y index shared by every physical shard."""
        return self.shard_descriptors[0].tile_y

    @property
    def n_points(self) -> int:
        """Return the total points represented by all physical shards."""
        return sum(descriptor.n_points for descriptor in self.shard_descriptors)


@dataclass(frozen=True)
class _CoarserTileInput:
    """Group the complete finer tiles contributing to one coarser tile."""

    tile_x: int
    tile_y: int
    finer_tiles: tuple[_FinerTileDescriptor, ...]

    @property
    def candidate_count(self) -> int:
        """Return the number of immediate-finer candidates for this tile."""
        return sum(tile.n_points for tile in self.finer_tiles)


def _write_spatial_levels(
    bridge_result: _LevelWriteResult,
    plan: _PointsCacheBuildPlan,
    *,
    staging_directory: Path,
) -> tuple[_LevelWriteResult, ...]:
    """Build every planned spatial level from the completed Bridge result.

    Levels are written in ascending order. Each completed spatial level becomes
    the sole point input for the next, preserving nested representative
    membership without revisiting the original points source.
    """
    if not staging_directory.is_dir():
        raise ValueError("`staging_directory` must be an existing directory.")
    if len(plan.levels) < 2 or plan.levels[1].kind is not _LevelKind.BRIDGE:
        raise ValueError("The cache build plan has no Bridge level for spatial construction.")

    spatial_levels = plan.levels[2:]
    if not spatial_levels:
        return ()

    results: list[_LevelWriteResult] = []
    finer_result = bridge_result
    finer_level = plan.levels[1]
    for coarser_level in spatial_levels:
        result = _write_spatial_level(
            finer_result,
            finer_level=finer_level,
            coarser_level=coarser_level,
            staging_directory=staging_directory,
        )
        results.append(result)
        finer_result = result
        finer_level = coarser_level

    coarsest_count = sum(row.n_points for row in results[-1].manifest_rows)
    if coarsest_count > plan.overview_point_budget:
        raise ValueError("The completed coarsest level exceeds the overview point budget.")
    return tuple(results)


def _write_spatial_level(
    finer_result: _LevelWriteResult,
    *,
    finer_level: _LevelBuildPlan,
    coarser_level: _LevelBuildPlan,
    staging_directory: Path,
) -> _LevelWriteResult:
    """Write one complete spatial level from its staged immediate-finer level."""
    if not staging_directory.is_dir():
        raise ValueError("`staging_directory` must be an existing directory.")
    _require_spatial_level_transition(finer_level, coarser_level)
    finer_tiles = _group_finer_manifest_rows(finer_result, finer_level=finer_level)
    coarser_tiles = _group_finer_tiles_by_coarser_tile(
        finer_tiles,
        coarser_level=coarser_level,
    )

    level_directory = staging_directory / coarser_level.relative_directory
    intermediate_count_directory = (
        staging_directory / _INTERMEDIATE_TILE_VALUE_COUNTS_DIRECTORY / f"level_{coarser_level.level}"
    )
    for path in (level_directory, intermediate_count_directory):
        if path.exists():
            raise FileExistsError(f"Spatial-level output path already exists: `{path}`.")
    level_directory.mkdir(parents=True)
    intermediate_count_directory.mkdir(parents=True)

    bucket_count = _bucket_count_for_level(coarser_level)
    tiles_by_bucket = _assign_coarser_tiles_to_buckets(coarser_tiles, bucket_count=bucket_count)
    capacity = coarser_level.max_points_per_tile
    if capacity is None:  # guarded by the level-plan contract
        raise ValueError("A spatial level must have a per-tile capacity.")
    expected_tile_rows = {(tile.tile_y, tile.tile_x): min(tile.candidate_count, capacity) for tile in coarser_tiles}
    filename_width = max(3, len(str(bucket_count - 1)))
    parquet_files: dict[str, pq.ParquetFile] = {}
    try:
        bucket_results = tuple(
            _write_spatial_bucket(
                bucket_id=bucket_id,
                coarser_tiles=tiles_by_bucket[bucket_id],
                finer_level=finer_level,
                coarser_level=coarser_level,
                staging_directory=staging_directory,
                level_directory=level_directory,
                intermediate_count_directory=intermediate_count_directory,
                filename_width=filename_width,
                parquet_files=parquet_files,
            )
            for bucket_id in sorted(tiles_by_bucket)
        )
    finally:
        for parquet_file in parquet_files.values():
            parquet_file.close()

    result = _reconcile_level_results(
        bucket_results,
        expected_point_count=sum(expected_tile_rows.values()),
    )
    _validate_spatial_result(
        result,
        coarser_level=coarser_level,
        expected_tile_rows=expected_tile_rows,
    )
    return result


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
    capacity = _require_spatial_level_transition(finer_level, coarser_level)
    _require_grid_coordinate(coarser_tile_x, "coarser_tile_x", grid_size=coarser_level.grid_width)
    _require_grid_coordinate(coarser_tile_y, "coarser_tile_y", grid_size=coarser_level.grid_height)
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


def _require_spatial_level_transition(
    finer_level: _LevelBuildPlan,
    coarser_level: _LevelBuildPlan,
) -> int:
    """Require two plans to describe one valid sampled spatial transition."""
    if finer_level.kind not in {_LevelKind.BRIDGE, _LevelKind.SPATIAL}:
        raise ValueError("The finer level must be a sampled Bridge or spatial level.")
    if coarser_level.kind is not _LevelKind.SPATIAL:
        raise ValueError("The coarser level must be a spatial level.")
    if coarser_level.level != finer_level.level + 1:
        raise ValueError("The coarser level must immediately follow the finer level.")
    if coarser_level.tile_size != 2 * finer_level.tile_size:
        raise ValueError("The coarser tile size must be twice the finer tile size.")
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


def _group_finer_manifest_rows(
    finer_result: _LevelWriteResult,
    *,
    finer_level: _LevelBuildPlan,
) -> tuple[_FinerTileDescriptor, ...]:
    """Normalize a sampled finer-level manifest into logical-tile descriptors.

    This is the spatial-construction counterpart of Bridge construction's
    ``_group_exact_manifest_rows``. Both helpers group flat physical manifest
    rows by logical tile coordinates, order their shards, and require contiguous
    shard numbering. This variant accepts the immediately finer Bridge or
    spatial level and returns ``_FinerTileDescriptor`` records.

    A Bridge tile currently has exactly one physical shard because its 4,096
    representative capacity is below the Parquet row-group limit. This helper
    also consumes previously written spatial levels, however, and a later
    level's per-tile capacity may exceed that physical limit. It therefore
    reconstructs every logical finer tile from one or more consecutive shards.
    """
    if not finer_result.manifest_rows:
        raise ValueError("The finer level result contains no manifest rows.")

    grouped: dict[tuple[int, int], list[_ManifestRow]] = defaultdict(list)
    for row in finer_result.manifest_rows:
        if row.level != finer_level.level:
            raise ValueError("Every input manifest row must belong to the planned finer level.")
        if row.tile_x >= finer_level.grid_width or row.tile_y >= finer_level.grid_height:
            raise ValueError("A finer manifest tile lies outside the planned finer grid.")
        grouped[(row.tile_y, row.tile_x)].append(row)

    finer_tiles: list[_FinerTileDescriptor] = []
    for (tile_y, tile_x), rows in sorted(grouped.items()):
        ordered_rows = tuple(sorted(rows, key=lambda row: row.tile_shard))
        if tuple(row.tile_shard for row in ordered_rows) != tuple(range(len(ordered_rows))):
            raise ValueError(f"Finer tile (tile_y={tile_y}, tile_x={tile_x}) has non-contiguous shards.")
        finer_tiles.append(_FinerTileDescriptor(shard_descriptors=ordered_rows))
    return tuple(finer_tiles)


def _group_finer_tiles_by_coarser_tile(
    finer_tiles: tuple[_FinerTileDescriptor, ...],
    *,
    coarser_level: _LevelBuildPlan,
) -> tuple[_CoarserTileInput, ...]:
    """Group complete finer-tile descriptors by their coarser tile.

    Each spatial level doubles the preceding tile edge, so one coarser tile
    covers a two-by-two block of immediately finer tiles::

        finer tiles             coarser tile

        (0, 0)  (1, 0)    ┐
        (0, 1)  (1, 1)    ┴──→ (0, 0)

        (2, 0)  (3, 0)    ┐
        (2, 1)  (3, 1)    ┴──→ (1, 0)

    The returned tuple groups those complete logical finer-tile descriptors
    into deterministic coarser-tile inputs. The example above becomes::

        (
            _CoarserTileInput(
                tile_x=0,
                tile_y=0,
                finer_tiles=(
                    finer_0_0,
                    finer_1_0,
                    finer_0_1,
                    finer_1_1,
                ),
            ),
            _CoarserTileInput(
                tile_x=1,
                tile_y=0,
                finer_tiles=(
                    finer_2_0,
                    finer_3_0,
                    finer_2_1,
                    finer_3_1,
                ),
            ),
        )

    Sparse and edge regions may contribute fewer than four nonempty finer
    tiles. This function organizes descriptors only; it does not read point
    rows, rebase coordinates, sample candidates, or write output.
    """
    grouped: dict[tuple[int, int], list[_FinerTileDescriptor]] = defaultdict(list)
    for finer_tile in finer_tiles:
        coarser_tile_x = finer_tile.tile_x // 2
        coarser_tile_y = finer_tile.tile_y // 2
        _require_grid_coordinate(coarser_tile_x, "coarser_tile_x", grid_size=coarser_level.grid_width)
        _require_grid_coordinate(coarser_tile_y, "coarser_tile_y", grid_size=coarser_level.grid_height)
        grouped[(coarser_tile_y, coarser_tile_x)].append(finer_tile)

    return tuple(
        _CoarserTileInput(
            tile_x=tile_x,
            tile_y=tile_y,
            finer_tiles=tuple(sorted(tiles, key=lambda tile: (tile.tile_y, tile.tile_x))),
        )
        for (tile_y, tile_x), tiles in sorted(grouped.items())
    )


def _assign_coarser_tiles_to_buckets(
    coarser_tiles: tuple[_CoarserTileInput, ...],
    *,
    bucket_count: int,
) -> dict[int, tuple[_CoarserTileInput, ...]]:
    """Group coarser logical tiles by deterministic output bucket."""
    tile_x = np.fromiter((tile.tile_x for tile in coarser_tiles), dtype=np.uint32, count=len(coarser_tiles))
    tile_y = np.fromiter((tile.tile_y for tile in coarser_tiles), dtype=np.uint32, count=len(coarser_tiles))
    bucket_ids = _tile_bucket_ids(tile_x, tile_y, bucket_count=bucket_count)

    grouped: dict[int, list[_CoarserTileInput]] = defaultdict(list)
    for tile, bucket_id in zip(coarser_tiles, bucket_ids, strict=True):
        grouped[int(bucket_id)].append(tile)
    return {
        bucket_id: tuple(sorted(tiles, key=lambda tile: (tile.tile_y, tile.tile_x)))
        for bucket_id, tiles in grouped.items()
    }


def _write_spatial_bucket(
    *,
    bucket_id: int,
    coarser_tiles: tuple[_CoarserTileInput, ...],
    finer_level: _LevelBuildPlan,
    coarser_level: _LevelBuildPlan,
    staging_directory: Path,
    level_directory: Path,
    intermediate_count_directory: Path,
    filename_width: int,
    parquet_files: dict[str, pq.ParquetFile],
) -> _BucketWriteResult:
    """Reconstruct, sample, and write one nonempty spatial output bucket."""
    capacity = coarser_level.max_points_per_tile
    if capacity is None:  # guarded by the level-plan contract
        raise ValueError("A spatial level must have a per-tile capacity.")

    filename = f"bucket-{bucket_id:0{filename_width}d}.parquet"
    point_path = level_directory / filename
    intermediate_count_path = intermediate_count_directory / filename
    if point_path.exists() or intermediate_count_path.exists():
        raise FileExistsError(f"Spatial bucket output already exists for bucket {bucket_id}.")

    relative_point_path = point_path.relative_to(staging_directory).as_posix()
    relative_intermediate_count_path = intermediate_count_path.relative_to(staging_directory).as_posix()
    manifest_rows: list[_ManifestRow] = []
    physical_row_group = 0
    bucket_point_count = 0
    intermediate_count_writer = _IntermediateTileValueCountWriter(
        intermediate_count_path,
        level=coarser_level.level,
    )
    try:
        with pq.ParquetWriter(
            point_path,
            _POINT_PAYLOAD_SCHEMA,
            compression="snappy",
            use_dictionary=["value_id"],
        ) as point_writer:
            for coarser_tile in coarser_tiles:
                decoded_finer_tiles = tuple(
                    _FinerLevelTile(
                        tile_x=finer_tile.tile_x,
                        tile_y=finer_tile.tile_y,
                        points=_read_logical_tile(
                            finer_tile.shard_descriptors,
                            staging_directory=staging_directory,
                            parquet_files=parquet_files,
                        ),
                    )
                    for finer_tile in coarser_tile.finer_tiles
                )
                sampled_table = _assemble_and_sample_coarser_tile(
                    decoded_finer_tiles,
                    finer_level=finer_level,
                    coarser_level=coarser_level,
                    coarser_tile_x=coarser_tile.tile_x,
                    coarser_tile_y=coarser_tile.tile_y,
                )
                expected_rows = min(coarser_tile.candidate_count, capacity)
                if sampled_table.num_rows != expected_rows:
                    raise ValueError("The sampled spatial tile does not match its expected row count.")

                value_ids = sampled_table["value_id"].combine_chunks().to_numpy(zero_copy_only=False)
                unique_value_ids, value_counts = np.unique(value_ids, return_counts=True)
                # Count the complete logical tile before physical row-group
                # sharding so every nonzero tile/value key is emitted once.
                intermediate_count_writer.append(
                    tile_x=coarser_tile.tile_x,
                    tile_y=coarser_tile.tile_y,
                    value_ids=unique_value_ids,
                    counts=value_counts,
                )

                for tile_shard, start in enumerate(range(0, sampled_table.num_rows, DEFAULT_MAX_ROWS_PER_ROW_GROUP)):
                    shard = sampled_table.slice(start, DEFAULT_MAX_ROWS_PER_ROW_GROUP)
                    point_writer.write_table(shard, row_group_size=shard.num_rows)
                    manifest_rows.append(
                        _ManifestRow(
                            level=coarser_level.level,
                            level_file=relative_point_path,
                            tile_x=coarser_tile.tile_x,
                            tile_y=coarser_tile.tile_y,
                            n_points=shard.num_rows,
                            row_group=physical_row_group,
                            tile_shard=tile_shard,
                        )
                    )
                    physical_row_group += 1
                bucket_point_count += sampled_table.num_rows
    finally:
        intermediate_count_writer.close()

    intermediate_count_file = _IntermediateTileValueCountFile(
        level=coarser_level.level,
        relative_path=relative_intermediate_count_path,
        row_count=intermediate_count_writer.row_count,
    )
    _validate_bucket_files(
        point_path,
        intermediate_count_path,
        manifest_rows=manifest_rows,
        intermediate_count_file=intermediate_count_file,
    )
    return _BucketWriteResult(
        bucket_id=bucket_id,
        point_count=bucket_point_count,
        value_count_total=intermediate_count_writer.point_count,
        manifest_rows=tuple(manifest_rows),
        intermediate_value_count_file=intermediate_count_file,
    )


def _validate_spatial_result(
    result: _LevelWriteResult,
    *,
    coarser_level: _LevelBuildPlan,
    expected_tile_rows: dict[tuple[int, int], int],
) -> None:
    """Validate spatial-level tile counts, grid membership, and shard order."""
    grouped: dict[tuple[int, int], list[_ManifestRow]] = defaultdict(list)
    for row in result.manifest_rows:
        if row.level != coarser_level.level:
            raise ValueError("Every output manifest row must belong to the spatial level.")
        if row.tile_x >= coarser_level.grid_width or row.tile_y >= coarser_level.grid_height:
            raise ValueError("A spatial output tile lies outside the planned grid.")
        grouped[(row.tile_y, row.tile_x)].append(row)

    observed_tile_rows: dict[tuple[int, int], int] = {}
    capacity = coarser_level.max_points_per_tile
    if capacity is None:  # guarded by the level-plan contract
        raise ValueError("A spatial level must have a per-tile capacity.")
    for key, rows in grouped.items():
        ordered_rows = sorted(rows, key=lambda row: row.tile_shard)
        if tuple(row.tile_shard for row in ordered_rows) != tuple(range(len(ordered_rows))):
            raise ValueError("A spatial output tile has non-contiguous shards.")
        tile_rows = sum(row.n_points for row in ordered_rows)
        if tile_rows > capacity:
            raise ValueError("A spatial output tile exceeds its planned capacity.")
        observed_tile_rows[key] = tile_rows

    if observed_tile_rows != expected_tile_rows:
        raise ValueError("Spatial tile rows do not reconcile to their finer inputs and planned capacities.")
