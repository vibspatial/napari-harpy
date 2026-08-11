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
    _bucket_count_for_level,
    _IntermediateTileValueCountWriter,
    _reconcile_level_results,
    _tile_bucket_ids,
    _validate_bucket_files,
)


@dataclass(frozen=True)
class _ExactTile:
    """Helper record for physical shard descriptors of one complete Exact tile."""

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
        """Return the total number of Exact points across all physical shards."""
        return sum(descriptor.n_points for descriptor in self.shard_descriptors)


def _write_bridge_level(
    exact_result: _LevelWriteResult,
    plan: _PointsCacheBuildPlan,
    *,
    staging_directory: Path,
) -> _LevelWriteResult:
    """Construct the sampled same-geometry Bridge from staged Exact tiles.

    Exact manifest rows are first grouped into complete logical tiles. Each tile
    is independently assigned to a Bridge output bucket, reconstructed from its
    referenced Exact row groups, sampled through the value-neutral tile sampler,
    and appended to the current Bridge point and intermediate-count files.

    Output buckets and their logical tiles are processed sequentially in
    deterministic order. No original source rows are revisited and no point-level
    shuffle is performed. At most one complete candidate tile and one output
    writer pair are active at a time.
    """
    exact, bridge = _require_bridge_levels(plan)
    if not staging_directory.is_dir():
        raise ValueError("`staging_directory` must be an existing directory.")

    exact_tiles = _group_exact_manifest_rows(
        exact_result,
        exact=exact,
    )
    bridge_capacity = bridge.max_points_per_tile
    if bridge_capacity is None:  # guarded by the level-plan contract
        raise ValueError("The Bridge level must have a per-tile capacity.")

    level_directory = staging_directory / bridge.relative_directory
    intermediate_count_directory = (
        staging_directory / _INTERMEDIATE_TILE_VALUE_COUNTS_DIRECTORY / f"level_{bridge.level}"
    )
    for path in (level_directory, intermediate_count_directory):
        if path.exists():
            raise FileExistsError(f"Bridge-level output path already exists: `{path}`.")
    level_directory.mkdir(parents=True)
    intermediate_count_directory.mkdir(parents=True)

    bucket_count = _bucket_count_for_level(bridge)
    tiles_by_bucket = _assign_tiles_to_buckets(exact_tiles, bucket_count=bucket_count)
    expected_tile_rows = {(tile.tile_y, tile.tile_x): min(tile.n_points, bridge_capacity) for tile in exact_tiles}
    filename_width = max(3, len(str(bucket_count - 1)))
    parquet_files: dict[str, pq.ParquetFile] = {}
    try:
        bucket_results = tuple(
            _write_bridge_bucket(
                bucket_id=bucket_id,
                exact_tiles=tiles_by_bucket[bucket_id],
                bridge=bridge,
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
    _validate_bridge_result(
        result,
        bridge=bridge,
        expected_tile_rows=expected_tile_rows,
    )
    return result


def _require_bridge_levels(plan: _PointsCacheBuildPlan) -> tuple[_LevelBuildPlan, _LevelBuildPlan]:
    if len(plan.levels) < 2:
        raise ValueError("The cache build plan has no Bridge level to construct.")
    exact, bridge = plan.levels[:2]
    if exact.level != 0 or exact.kind is not _LevelKind.EXACT:
        raise ValueError("The first planned level must be serialized Exact level 0.")
    if bridge.level != 1 or bridge.kind is not _LevelKind.BRIDGE:
        raise ValueError("The second planned level must be serialized Bridge level 1.")
    if (exact.tile_size, exact.grid_width, exact.grid_height) != (
        bridge.tile_size,
        bridge.grid_width,
        bridge.grid_height,
    ):
        raise ValueError("Exact and Bridge levels must have identical logical tile geometry.")
    return exact, bridge


def _group_exact_manifest_rows(
    exact_result: _LevelWriteResult,
    *,
    exact: _LevelBuildPlan,
) -> tuple[_ExactTile, ...]:
    """Group physical Exact row groups into complete logical tiles.

    The Exact manifest is a flat sequence with one record per physical Parquet
    row group. A dense logical tile may therefore appear in several records,
    distinguished by consecutive ``tile_shard`` values. This function groups
    those records by ``(tile_y, tile_x)``, orders each group by ``tile_shard``,
    and requires its shard numbers to be exactly ``0, 1, ..., n - 1``.

    For example, manifest records for tile ``(tile_y=0, tile_x=0)`` with shards
    ``0`` and ``1``, followed by one record for tile ``(tile_y=0, tile_x=1)``,
    become::

        (
            _ExactTile(shard_descriptors=(tile_0_shard_0, tile_0_shard_1)),
            _ExactTile(shard_descriptors=(tile_1_shard_0,)),
        )

    Every record must belong to the planned Exact level and lie inside its
    logical grid. The returned records follow deterministic ``(tile_y, tile_x)``
    order. Downstream reconstruction reads each record as one complete candidate
    tile so the sampling capacity is applied once per logical tile, rather than
    independently to its physical shards.
    """
    if not exact_result.manifest_rows:
        raise ValueError("The Exact level result contains no manifest rows.")

    grouped: dict[tuple[int, int], list[_ManifestRow]] = defaultdict(list)
    for row in exact_result.manifest_rows:
        if row.level != exact.level:
            raise ValueError("Every input manifest row must belong to Exact level 0.")
        if row.tile_x >= exact.grid_width or row.tile_y >= exact.grid_height:
            raise ValueError("An Exact manifest tile lies outside the planned Exact grid.")
        grouped[(row.tile_y, row.tile_x)].append(row)

    exact_tiles: list[_ExactTile] = []
    for (tile_y, tile_x), rows in sorted(grouped.items()):
        ordered_rows = tuple(sorted(rows, key=lambda row: row.tile_shard))
        if tuple(row.tile_shard for row in ordered_rows) != tuple(range(len(ordered_rows))):
            raise ValueError(f"Exact tile (tile_y={tile_y}, tile_x={tile_x}) has non-contiguous shards.")
        exact_tiles.append(
            _ExactTile(
                shard_descriptors=ordered_rows,
            )
        )
    return tuple(exact_tiles)


def _assign_tiles_to_buckets(
    exact_tiles: tuple[_ExactTile, ...],
    *,
    bucket_count: int,
) -> dict[int, tuple[_ExactTile, ...]]:
    """Group Exact-tile descriptors by deterministic Bridge output bucket."""
    tile_y = np.fromiter((tile.tile_y for tile in exact_tiles), dtype=np.uint32, count=len(exact_tiles))
    tile_x = np.fromiter((tile.tile_x for tile in exact_tiles), dtype=np.uint32, count=len(exact_tiles))
    bucket_ids = _tile_bucket_ids(tile_x, tile_y, bucket_count=bucket_count)

    grouped: dict[int, list[_ExactTile]] = defaultdict(list)
    for tile, bucket_id in zip(exact_tiles, bucket_ids, strict=True):
        grouped[int(bucket_id)].append(tile)
    return {
        bucket_id: tuple(sorted(tiles, key=lambda tile: (tile.tile_y, tile.tile_x)))
        for bucket_id, tiles in grouped.items()
    }


def _write_bridge_bucket(
    *,
    bucket_id: int,
    exact_tiles: tuple[_ExactTile, ...],
    bridge: _LevelBuildPlan,
    staging_directory: Path,
    level_directory: Path,
    intermediate_count_directory: Path,
    filename_width: int,
    parquet_files: dict[str, pq.ParquetFile],
) -> _BucketWriteResult:
    """Write one nonempty physical Bridge output bucket.

    The supplied Exact tiles have already been assigned to this bucket and are
    ordered by logical ``(tile_y, tile_x)``. Each tile is reconstructed from its
    staged Exact row groups, retained completely when sparse or sampled down to
    the Bridge capacity when dense, written as one point row group, and counted
    by ``value_id`` in the bucket's companion intermediate file.

    After closing both Parquet writers, the files are validated against their
    descriptors. The returned result contains the bucket's manifest rows,
    intermediate-count descriptor, and reconciliable point totals.
    """
    bridge_capacity = bridge.max_points_per_tile
    if bridge_capacity is None:  # guarded by the level-plan contract
        raise ValueError("The Bridge level must have a per-tile capacity.")

    filename = f"bucket-{bucket_id:0{filename_width}d}.parquet"
    point_path = level_directory / filename
    intermediate_count_path = intermediate_count_directory / filename
    if point_path.exists() or intermediate_count_path.exists():
        raise FileExistsError(f"Bridge bucket output already exists for bucket {bucket_id}.")

    relative_point_path = point_path.relative_to(staging_directory).as_posix()
    relative_intermediate_count_path = intermediate_count_path.relative_to(staging_directory).as_posix()
    manifest_rows: list[_ManifestRow] = []
    bucket_point_count = 0
    intermediate_count_writer = _IntermediateTileValueCountWriter(
        intermediate_count_path,
        level=bridge.level,
    )
    try:
        with pq.ParquetWriter(
            point_path,
            _POINT_PAYLOAD_SCHEMA,
            compression="snappy",
            use_dictionary=["value_id"],
        ) as point_writer:
            # Every Bridge tile is capped to fit in one physical row group. Each
            # loop iteration therefore writes exactly one row group, making the
            # enumerate index its output row-group index and `tile_shard` zero.
            for physical_row_group, exact_tile in enumerate(exact_tiles):
                candidate_table = _read_exact_tile(
                    shard_descriptors=exact_tile.shard_descriptors,
                    staging_directory=staging_directory,
                    parquet_files=parquet_files,
                )
                selected_indices = _select_sampled_tile_indices(
                    candidate_table["x_rel"].combine_chunks().to_numpy(zero_copy_only=False),
                    candidate_table["y_rel"].combine_chunks().to_numpy(zero_copy_only=False),
                    candidate_table["point_id"].combine_chunks().to_numpy(zero_copy_only=False),
                    level=bridge.level,
                    tile_x=exact_tile.tile_x,
                    tile_y=exact_tile.tile_y,
                    tile_size=bridge.tile_size,
                    target=bridge_capacity,
                )
                sampled_table = candidate_table.take(pa.array(selected_indices, type=pa.int64()))
                expected_rows = min(candidate_table.num_rows, bridge_capacity)
                # Sparse tiles retain every candidate; dense tiles must fill the
                # Bridge capacity exactly.
                if sampled_table.num_rows != expected_rows:
                    raise ValueError("The sampled Bridge tile does not match its planned capacity.")

                value_ids = sampled_table["value_id"].combine_chunks().to_numpy(zero_copy_only=False)
                unique_value_ids, value_counts = np.unique(value_ids, return_counts=True)
                intermediate_count_writer.append(
                    tile_x=exact_tile.tile_x,
                    tile_y=exact_tile.tile_y,
                    value_ids=unique_value_ids,
                    counts=value_counts,
                )
                point_writer.write_table(sampled_table, row_group_size=sampled_table.num_rows)
                manifest_rows.append(
                    _ManifestRow(
                        level=bridge.level,
                        level_file=relative_point_path,
                        tile_x=exact_tile.tile_x,
                        tile_y=exact_tile.tile_y,
                        n_points=sampled_table.num_rows,
                        row_group=physical_row_group,
                        # A Bridge tile always fits in one physical row group;
                        # therefore its only tile-shard index is zero.
                        tile_shard=0,
                    )
                )
                bucket_point_count += sampled_table.num_rows
    finally:
        intermediate_count_writer.close()

    intermediate_count_file = _IntermediateTileValueCountFile(
        level=bridge.level,
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


def _read_exact_tile(
    shard_descriptors: tuple[_ManifestRow, ...],
    *,
    staging_directory: Path,
    parquet_files: dict[str, pq.ParquetFile],
) -> pa.Table:
    """Read and concatenate the Parquet row groups described by one Exact tile."""
    decoded_shard_tables: list[pa.Table] = []
    expected_rows = 0
    for descriptor in shard_descriptors:
        parquet_file = parquet_files.get(descriptor.level_file)
        if parquet_file is None:
            parquet_file = pq.ParquetFile(staging_directory / descriptor.level_file)
            if not parquet_file.schema_arrow.equals(_POINT_PAYLOAD_SCHEMA, check_metadata=False):
                parquet_file.close()
                raise ValueError(f"Exact point file `{descriptor.level_file}` has an incompatible payload schema.")
            parquet_files[descriptor.level_file] = parquet_file
        if descriptor.row_group >= parquet_file.num_row_groups:
            raise ValueError(
                f"Exact point file `{descriptor.level_file}` does not contain row group {descriptor.row_group}."
            )

        decoded_shard = parquet_file.read_row_group(
            descriptor.row_group,
            columns=_POINT_PAYLOAD_SCHEMA.names,
        )
        if decoded_shard.num_rows != descriptor.n_points:
            raise ValueError("A decoded Exact tile shard does not match its manifest row count.")
        decoded_shard_tables.append(decoded_shard)
        expected_rows += descriptor.n_points

    candidate_table = pa.concat_tables(decoded_shard_tables)
    if candidate_table.num_rows != expected_rows:
        raise ValueError("The reconstructed Exact tile does not match its manifest row count.")
    return candidate_table


def _validate_bridge_result(
    result: _LevelWriteResult,
    *,
    bridge: _LevelBuildPlan,
    expected_tile_rows: dict[tuple[int, int], int],
) -> None:
    """Validate the Bridge-specific invariants of a completed level result.

    Every manifest row must belong to the Bridge level, every tile must remain
    unsharded, and each tile's row count must match the count implied by its
    Exact candidates and the Bridge capacity.
    """
    observed_tile_rows: dict[tuple[int, int], int] = defaultdict(int)
    for row in result.manifest_rows:
        if row.level != bridge.level:
            raise ValueError("Every output manifest row must belong to the Bridge level.")
        if row.tile_shard != 0:
            raise ValueError("Every Bridge tile must fit in one physical row group.")
        observed_tile_rows[(row.tile_y, row.tile_x)] += row.n_points

    if observed_tile_rows != expected_tile_rows:
        raise ValueError("Bridge tile row counts do not reconcile to the Exact input tiles and planned capacity.")
