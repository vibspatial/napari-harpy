from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from napari_harpy.core.multi_scale_cache_points_zarr.build_plan import (
    _LevelBuildPlan,
    _LevelKind,
    _PointsCacheBuildPlan,
)
from napari_harpy.core.multi_scale_cache_points_zarr.hashing import (
    _bucket_count_for_level,
    _tile_bucket_ids,
)
from napari_harpy.core.multi_scale_cache_points_zarr.models import (
    _INT64_MAX,
    _require_integer_in_range,
    _TileDescriptor,
)
from napari_harpy.core.multi_scale_cache_points_zarr.sampling import _select_sampled_tile_indices
from napari_harpy.core.multi_scale_cache_points_zarr.storage.bucket_writer import _BucketWriter
from napari_harpy.core.multi_scale_cache_points_zarr.storage.models import (
    _BucketPlan,
    _BucketWriteResult,
    _LevelWriteResult,
    _PlannedTile,
    _ZarrWriteSettings,
)
from napari_harpy.core.multi_scale_cache_points_zarr.storage.reader_cache import _BucketReaderCache


@dataclass(frozen=True)
class _BridgeWriterConfig:
    """Configure Bridge storage and bounded Exact-reader lifetime.

    Parameters
    ----------
    zarr_settings
        Physical chunk, shard, and codec settings shared by every Bridge
        output bucket.
    max_open_exact_readers
        Positive maximum number of entered Exact bucket readers retained by
        Bridge construction. ``None`` retains all nonempty Exact bucket
        readers and is the default. Readers cache initialized metadata, not
        point chunks or point payloads. Bridge output order can revisit Exact
        buckets in an interleaved pattern, so retaining their readers avoids
        repeatedly reopening the same Zarr arrays and reloading their metadata.
    """

    zarr_settings: _ZarrWriteSettings
    max_open_exact_readers: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.zarr_settings, _ZarrWriteSettings):
            raise ValueError("`zarr_settings` must be _ZarrWriteSettings.")
        if self.max_open_exact_readers is not None:
            _require_integer_in_range(
                self.max_open_exact_readers,
                "max_open_exact_readers",
                minimum=1,
                maximum=_INT64_MAX,
            )


def _write_bridge_level(
    exact_result: _LevelWriteResult,
    plan: _PointsCacheBuildPlan,
    *,
    staging_root: Path,
    config: _BridgeWriterConfig,
) -> _LevelWriteResult:
    """Construct sampled same-geometry Bridge level one from Exact Zarr.

    Small Exact descriptors are routed to deterministic Bridge output buckets
    before any point data is read. Each output bucket is then written
    sequentially in ``(tile_y, tile_x)`` order::

        Exact _TileDescriptor
            -> bounded entered _BucketReader reuse
            -> read one complete Exact _PointPayload
            -> select value-neutral tile-local representatives
            -> take the same rows from all four aligned payload fields
            -> common _BucketWriter value-major ordering and sparse ranges
            -> finalized Bridge Zarr bucket

    No source Parquet, point-level shuffle, coordinate rebasing, or complete
    output-bucket point materialization is involved. At steady state the
    coordinator holds one complete Exact candidate payload, one retained
    payload, one active output writer, and bounded reader metadata.

    Parameters
    ----------
    exact_result
        Finalized nonempty Exact level-zero result. It provides exactly one
        descriptor per nonempty logical input tile.
    plan
        Complete logical cache plan containing matching Exact and Bridge
        levels at positions zero and one.
    staging_root
        Existing isolated generation root containing Exact. This function owns
        creation of the previously absent ``levels/level_1`` directory.
    config
        Bridge physical Zarr settings and Exact-reader bound.

    Returns
    -------
    _LevelWriteResult
        Nonempty finalized Bridge buckets ordered by numeric bucket ID.
    """
    if not isinstance(exact_result, _LevelWriteResult) or exact_result.level != 0:
        raise ValueError("`exact_result` must be a nonempty Exact level-zero result.")
    if not isinstance(plan, _PointsCacheBuildPlan):
        raise ValueError("`plan` must be a _PointsCacheBuildPlan.")
    if not isinstance(config, _BridgeWriterConfig):
        raise ValueError("`config` must be a _BridgeWriterConfig.")
    exact, bridge = _require_bridge_levels(plan)
    if not isinstance(staging_root, Path) or not staging_root.is_dir():
        raise ValueError("`staging_root` must be an existing pathlib.Path directory.")
    if not (staging_root / exact.relative_directory).is_dir():
        raise ValueError("The staged Exact level directory does not exist.")
    level_directory = staging_root / bridge.relative_directory
    if level_directory.exists():
        raise FileExistsError(f"Bridge-level output path already exists: {bridge.relative_directory}.")

    exact_descriptors = exact_result.tile_descriptors
    if any(
        descriptor.level != exact.level
        or descriptor.tile_x >= exact.grid_width
        or descriptor.tile_y >= exact.grid_height
        for descriptor in exact_descriptors
    ):
        raise ValueError("Every Exact descriptor must belong to level zero and lie inside its planned grid.")
    if exact_result.point_count != exact.point_count_upper_bound:
        raise ValueError("Exact rows do not reconcile to the uncapped planned source count.")

    bridge_capacity = bridge.max_points_per_tile
    if bridge_capacity is None:  # guarded by `_require_bridge_levels`
        raise ValueError("The Bridge level must have a per-tile capacity.")
    expected_bridge_point_count = sum(min(descriptor.n_points, bridge_capacity) for descriptor in exact_descriptors)
    if expected_bridge_point_count > bridge.point_count_upper_bound:
        raise ValueError("Expected Bridge rows exceed the planned point-count upper bound.")
    tiles_by_bucket = _assign_bridge_buckets(
        exact_descriptors,
        bucket_count=_bucket_count_for_level(bridge),
    )
    level_directory.mkdir(parents=True)
    bucket_results: list[_BucketWriteResult] = []
    reader_capacity = (
        exact_result.bucket_count
        if config.max_open_exact_readers is None
        else min(config.max_open_exact_readers, exact_result.bucket_count)
    )
    with _BucketReaderCache(
        staging_root,
        max_open_readers=reader_capacity,
    ) as reader_cache:
        for bucket_id in sorted(tiles_by_bucket):
            bucket_results.append(
                _write_bridge_bucket(
                    bucket_id=bucket_id,
                    exact_descriptors=tiles_by_bucket[bucket_id],
                    bridge=bridge,
                    staging_root=staging_root,
                    settings=config.zarr_settings,
                    reader_cache=reader_cache,
                )
            )

    return _reconcile_bridge_results(
        tuple(bucket_results),
        exact_descriptors=exact_descriptors,
        bridge=bridge,
    )


def _require_bridge_levels(plan: _PointsCacheBuildPlan) -> tuple[_LevelBuildPlan, _LevelBuildPlan]:
    """Return and validate the planned Exact-to-Bridge transition."""
    if len(plan.levels) < 2:
        raise ValueError("The cache build plan has no Bridge level to construct.")
    exact, bridge = plan.levels[:2]
    if exact.level != 0 or exact.kind is not _LevelKind.EXACT or exact.max_points_per_tile is not None:
        raise ValueError("The first planned level must be uncapped Exact level zero.")
    if bridge.level != 1 or bridge.kind is not _LevelKind.BRIDGE or bridge.max_points_per_tile is None:
        raise ValueError("The second planned level must be capped Bridge level one.")
    if (exact.tile_size, exact.grid_width, exact.grid_height) != (
        bridge.tile_size,
        bridge.grid_width,
        bridge.grid_height,
    ):
        raise ValueError("Exact and Bridge levels must have identical logical tile geometry.")
    return exact, bridge


def _assign_bridge_buckets(
    exact_descriptors: tuple[_TileDescriptor, ...],
    *,
    bucket_count: int,
) -> dict[int, tuple[_TileDescriptor, ...]]:
    """Group complete Exact tile descriptors by Bridge destination bucket."""
    tile_x = np.fromiter(
        (descriptor.tile_x for descriptor in exact_descriptors),
        dtype=np.uint32,
        count=len(exact_descriptors),
    )
    tile_y = np.fromiter(
        (descriptor.tile_y for descriptor in exact_descriptors),
        dtype=np.uint32,
        count=len(exact_descriptors),
    )
    bucket_id = _tile_bucket_ids(tile_x, tile_y, bucket_count=bucket_count)
    grouped: dict[int, list[_TileDescriptor]] = defaultdict(list)
    for descriptor, destination in zip(exact_descriptors, bucket_id, strict=True):
        grouped[int(destination)].append(descriptor)
    return {
        destination: tuple(sorted(descriptors, key=lambda descriptor: (descriptor.tile_y, descriptor.tile_x)))
        for destination, descriptors in grouped.items()
    }


def _write_bridge_bucket(
    *,
    bucket_id: int,
    exact_descriptors: tuple[_TileDescriptor, ...],
    bridge: _LevelBuildPlan,
    staging_root: Path,
    settings: _ZarrWriteSettings,
    reader_cache: _BucketReaderCache,
) -> _BucketWriteResult:
    """Read, sample, and write one nonempty Bridge destination bucket."""
    bridge_capacity = bridge.max_points_per_tile
    if bridge_capacity is None:
        raise ValueError("The Bridge level must have a per-tile capacity.")
    planned_tiles = tuple(
        _PlannedTile(
            tile_x=descriptor.tile_x,
            tile_y=descriptor.tile_y,
            n_points=min(descriptor.n_points, bridge_capacity),
        )
        for descriptor in exact_descriptors
    )
    bucket_plan = _BucketPlan(
        level=bridge.level,
        bucket_id=bucket_id,
        tiles=planned_tiles,
        settings=settings,
    )

    with _BucketWriter(staging_root, bucket_plan) as writer:
        for descriptor, planned_tile in zip(exact_descriptors, planned_tiles, strict=True):
            reader = reader_cache.get(level=descriptor.level, bucket_id=descriptor.bucket_id)
            candidate = reader.read_complete(descriptor)
            selected_indices = _select_sampled_tile_indices(
                candidate.x_rel,
                candidate.y_rel,
                candidate.point_id,
                level=bridge.level,
                tile_x=descriptor.tile_x,
                tile_y=descriptor.tile_y,
                tile_size=bridge.tile_size,
                target=bridge_capacity,
            )
            retained = candidate.take(selected_indices)
            if retained.n_points != planned_tile.n_points:
                raise ValueError("The sampled Bridge tile does not match its planned point count.")
            writer.write_tile(descriptor.tile_x, descriptor.tile_y, retained)
            # The cache alone owns reader lifetime; point arrays and the caller's
            # borrowed reader reference do not survive this tile operation.
            del reader, candidate, selected_indices, retained
        return writer.finalize()


def _reconcile_bridge_results(
    bucket_results: tuple[_BucketWriteResult, ...],
    *,
    exact_descriptors: tuple[_TileDescriptor, ...],
    bridge: _LevelBuildPlan,
) -> _LevelWriteResult:
    """Reconcile finalized Bridge descriptors with planned Exact tile facts."""
    result = _LevelWriteResult(buckets=bucket_results)
    if result.level != bridge.level:
        raise ValueError("Bridge construction produced the wrong serialized level.")
    capacity = bridge.max_points_per_tile
    if capacity is None:
        raise ValueError("The Bridge level must have a per-tile capacity.")
    expected = {
        (descriptor.tile_x, descriptor.tile_y): min(descriptor.n_points, capacity) for descriptor in exact_descriptors
    }
    observed = {(descriptor.tile_x, descriptor.tile_y): descriptor.n_points for descriptor in result.tile_descriptors}
    if observed != expected:
        raise ValueError("Bridge tile coordinates or point counts do not reconcile to Exact.")
    expected_point_count = sum(expected.values())
    if result.point_count != expected_point_count:
        raise ValueError("Bridge bucket rows do not reconcile to the expected sampled point count.")
    if result.point_count > bridge.point_count_upper_bound:
        raise ValueError("Bridge point count exceeds its planned upper bound.")
    if any(
        descriptor.tile_x >= bridge.grid_width or descriptor.tile_y >= bridge.grid_height
        for descriptor in result.tile_descriptors
    ):
        raise ValueError("Final Bridge descriptors fall outside the planned grid.")
    return result
