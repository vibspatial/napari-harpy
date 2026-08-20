from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

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
    _UINT32_MAX,
    _require_integer_in_range,
    _TileDescriptor,
)
from napari_harpy.core.multi_scale_cache_points_zarr.payload import _PointPayload
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
class _SpatialWriterConfig:
    """Configure Spatial storage and immediate-finer reader lifetime.

    Parameters
    ----------
    zarr_settings
        Physical chunk, shard, and codec settings shared by every Spatial
        output bucket.
    max_open_finer_readers
        Positive maximum number of entered immediate-finer bucket readers
        retained while constructing one Spatial level. ``None`` retains every
        nonempty input bucket and is the default. A fresh level-scoped cache is
        closed before the completed output becomes the next level's input.
    """

    zarr_settings: _ZarrWriteSettings
    max_open_finer_readers: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.zarr_settings, _ZarrWriteSettings):
            raise ValueError("`zarr_settings` must be _ZarrWriteSettings.")
        if self.max_open_finer_readers is not None:
            _require_integer_in_range(
                self.max_open_finer_readers,
                "max_open_finer_readers",
                minimum=1,
                maximum=_INT64_MAX,
            )


@dataclass(frozen=True)
class _CoarserTileInput:
    """Plan the complete immediate-finer input of one Spatial coarser tile.

    Parameters
    ----------
    tile_x, tile_y
        Coordinates of the nonempty coarser tile in its level grid.
    finer_descriptors
        One through four complete, nonempty immediate-finer tiles. They are
        ordered by ``(tile_y, tile_x)`` and all map to this coarser tile
        through integer division by two. Descriptors contain no decoded point
        arrays.

    Notes
    -----
    This is descriptor-only planning state. ``candidate_count`` determines the
    exact size of the later combined payload, while each descriptor retains the
    physical bucket address used to read one contributor at a time.
    """

    tile_x: int
    tile_y: int
    finer_descriptors: tuple[_TileDescriptor, ...]

    def __post_init__(self) -> None:
        _require_integer_in_range(self.tile_x, "tile_x", maximum=_UINT32_MAX)
        _require_integer_in_range(self.tile_y, "tile_y", maximum=_UINT32_MAX)
        if not isinstance(self.finer_descriptors, tuple) or not 1 <= len(self.finer_descriptors) <= 4:
            raise ValueError("`finer_descriptors` must contain one through four tiles.")
        if not all(isinstance(descriptor, _TileDescriptor) for descriptor in self.finer_descriptors):
            raise ValueError("Every finer descriptor must be a _TileDescriptor.")
        if self.finer_descriptors != tuple(
            sorted(self.finer_descriptors, key=lambda descriptor: (descriptor.tile_y, descriptor.tile_x))
        ):
            raise ValueError("Finer descriptors must follow (tile_y, tile_x) order.")
        coordinates = {(descriptor.tile_x, descriptor.tile_y) for descriptor in self.finer_descriptors}
        if len(coordinates) != len(self.finer_descriptors):
            raise ValueError("Finer descriptor coordinates must be unique.")
        if any(
            descriptor.tile_x // 2 != self.tile_x or descriptor.tile_y // 2 != self.tile_y
            for descriptor in self.finer_descriptors
        ):
            raise ValueError("Every finer descriptor must map to the stated coarser tile.")
        if len({descriptor.level for descriptor in self.finer_descriptors}) != 1:
            raise ValueError("Every finer descriptor must belong to the same level.")
        if self.candidate_count > _INT64_MAX:
            raise ValueError("Coarser-tile candidate count exceeds the supported int64 range.")

    @property
    def candidate_count(self) -> int:
        """Return the complete immediate-finer candidate count."""
        return sum(descriptor.n_points for descriptor in self.finer_descriptors)


def _write_spatial_levels(
    bridge_result: _LevelWriteResult,
    plan: _PointsCacheBuildPlan,
    *,
    staging_root: Path,
    config: _SpatialWriterConfig,
) -> tuple[_LevelWriteResult, ...]:
    """Construct every planned Spatial level from its immediate predecessor.

    Levels are built in ascending serialized order. Each transition groups one
    through four nonempty finer descriptors into a coarser tile, reads and
    rebases those complete payloads, samples the assembled coarser tile once,
    and persists it through the common Zarr bucket writer. A completed level is
    the sole point input to its successor; the canonical source is never
    accepted or revisited.

    Parameters
    ----------
    bridge_result
        Finalized nonempty level-one Bridge result.
    plan
        Complete logical plan containing Exact, Bridge, and zero or more
        Spatial levels.
    staging_root
        Existing isolated generation root containing the completed Bridge.
    config
        Spatial Zarr settings and per-level immediate-finer reader bound.

    Returns
    -------
    tuple of _LevelWriteResult
        Completed Spatial levels in ascending serialized order. The tuple is
        empty when Bridge is already the terminal overview level.
    """
    if not isinstance(bridge_result, _LevelWriteResult) or bridge_result.level != 1:
        raise ValueError("`bridge_result` must be a nonempty level-one result.")
    if not isinstance(plan, _PointsCacheBuildPlan):
        raise ValueError("`plan` must be a _PointsCacheBuildPlan.")
    if not isinstance(config, _SpatialWriterConfig):
        raise ValueError("`config` must be a _SpatialWriterConfig.")
    if not isinstance(staging_root, Path) or not staging_root.is_dir():
        raise ValueError("`staging_root` must be an existing pathlib.Path directory.")
    if len(plan.levels) < 2 or plan.levels[1].kind is not _LevelKind.BRIDGE:
        raise ValueError("The cache build plan has no Bridge level for Spatial construction.")

    bridge = plan.levels[1]
    _require_level_result(bridge_result, bridge, name="Bridge")
    if not (staging_root / bridge.relative_directory).is_dir():
        raise ValueError("The staged Bridge level directory does not exist.")

    spatial_levels = plan.levels[2:]
    if not spatial_levels:
        if bridge_result.point_count > plan.overview_point_budget:
            raise ValueError("The terminal Bridge exceeds the overview point budget.")
        return ()

    results: list[_LevelWriteResult] = []
    finer_result = bridge_result
    finer_level = bridge
    for coarser_level in spatial_levels:
        result = _write_spatial_level(
            finer_result,
            finer_level=finer_level,
            coarser_level=coarser_level,
            staging_root=staging_root,
            config=config,
        )
        results.append(result)
        finer_result = result
        finer_level = coarser_level

    if results[-1].point_count > plan.overview_point_budget:
        raise ValueError("The completed Spatial overview exceeds its point budget.")
    return tuple(results)


def _write_spatial_level(
    finer_result: _LevelWriteResult,
    *,
    finer_level: _LevelBuildPlan,
    coarser_level: _LevelBuildPlan,
    staging_root: Path,
    config: _SpatialWriterConfig,
) -> _LevelWriteResult:
    """Construct one complete Spatial level from one completed finer level."""
    if not isinstance(finer_result, _LevelWriteResult):
        raise ValueError("`finer_result` must be a _LevelWriteResult.")
    if not isinstance(config, _SpatialWriterConfig):
        raise ValueError("`config` must be a _SpatialWriterConfig.")
    if not isinstance(staging_root, Path) or not staging_root.is_dir():
        raise ValueError("`staging_root` must be an existing pathlib.Path directory.")
    _require_spatial_transition(finer_level, coarser_level)
    _require_level_result(finer_result, finer_level, name="Immediate-finer")
    if not (staging_root / finer_level.relative_directory).is_dir():
        raise ValueError("The staged immediate-finer level directory does not exist.")
    level_directory = staging_root / coarser_level.relative_directory
    if level_directory.exists():
        raise FileExistsError(f"Spatial-level output path already exists: {coarser_level.relative_directory}.")

    # Group the one through four nonempty descriptors in each 2-by-2
    # finer-tile block into one coarser logical tile. Empty and missing edge
    # tiles are absent rather than represented by placeholders.
    coarser_tiles = _group_finer_descriptors(
        finer_result.tile_descriptors,
        finer_level=finer_level,
        coarser_level=coarser_level,
    )
    tiles_by_bucket = _assign_spatial_buckets(
        coarser_tiles,
        bucket_count=_bucket_count_for_level(coarser_level),
    )

    level_directory.mkdir(parents=True)
    reader_capacity = (
        finer_result.bucket_count
        if config.max_open_finer_readers is None
        else min(config.max_open_finer_readers, finer_result.bucket_count)
    )
    bucket_results: list[_BucketWriteResult] = []
    with _BucketReaderCache(staging_root, max_open_readers=reader_capacity) as reader_cache:
        for bucket_id in sorted(tiles_by_bucket):
            bucket_results.append(
                _write_spatial_bucket(
                    bucket_id=bucket_id,
                    coarser_tiles=tiles_by_bucket[bucket_id],
                    finer_level=finer_level,
                    coarser_level=coarser_level,
                    staging_root=staging_root,
                    settings=config.zarr_settings,
                    reader_cache=reader_cache,
                )
            )

    return _reconcile_spatial_result(
        tuple(bucket_results),
        coarser_tiles=coarser_tiles,
        finer_result=finer_result,
        coarser_level=coarser_level,
    )


def _require_spatial_transition(finer_level: _LevelBuildPlan, coarser_level: _LevelBuildPlan) -> int:
    """Validate one immediate-finer to Spatial plan transition."""
    if not isinstance(finer_level, _LevelBuildPlan) or not isinstance(coarser_level, _LevelBuildPlan):
        raise ValueError("Spatial transitions require _LevelBuildPlan values.")
    if finer_level.kind not in {_LevelKind.BRIDGE, _LevelKind.SPATIAL}:
        raise ValueError("The immediate-finer level must be Bridge or Spatial.")
    if coarser_level.kind is not _LevelKind.SPATIAL:
        raise ValueError("The coarser level must be Spatial.")
    if coarser_level.level != finer_level.level + 1:
        raise ValueError("The coarser level must immediately follow the finer level.")
    if coarser_level.tile_size != 2 * finer_level.tile_size:
        raise ValueError("The coarser tile size must be twice the finer tile size.")
    if coarser_level.grid_width != math.ceil(finer_level.grid_width / 2) or coarser_level.grid_height != math.ceil(
        finer_level.grid_height / 2
    ):
        raise ValueError("The coarser grid must be the halved immediate-finer grid.")
    if coarser_level.point_count_upper_bound > finer_level.point_count_upper_bound:
        raise ValueError("The coarser point-count upper bound must not increase.")
    capacity = coarser_level.max_points_per_tile
    if capacity is None:
        raise ValueError("A Spatial level must have a per-tile capacity.")
    return capacity


def _require_level_result(result: _LevelWriteResult, level: _LevelBuildPlan, *, name: str) -> None:
    """Require a finalized level result to agree with its logical plan."""
    if result.level != level.level:
        raise ValueError(f"{name} result belongs to a different serialized level.")
    if result.point_count > level.point_count_upper_bound:
        raise ValueError(f"{name} result exceeds its planned point-count upper bound.")
    if any(
        descriptor.level != level.level
        or descriptor.tile_x >= level.grid_width
        or descriptor.tile_y >= level.grid_height
        for descriptor in result.tile_descriptors
    ):
        raise ValueError(f"{name} descriptors disagree with the planned level or grid.")


def _group_finer_descriptors(
    finer_descriptors: tuple[_TileDescriptor, ...],
    *,
    finer_level: _LevelBuildPlan,
    coarser_level: _LevelBuildPlan,
) -> tuple[_CoarserTileInput, ...]:
    """Group complete immediate-finer descriptors by their coarser tile.

    Each Spatial level doubles the preceding tile edge, so one coarser tile
    covers a 2-by-2 block of immediately finer tiles::

        finer tiles             coarser tile

        (0, 0)  (1, 0)    ─┐
        (0, 1)  (1, 1)    ─┴──→ (0, 0)

        (2, 0)  (3, 0)    ─┐
        (2, 1)  (3, 1)    ─┴──→ (1, 0)

    The first block therefore becomes one descriptor-only plan::

        _CoarserTileInput(
            tile_x=0,
            tile_y=0,
            finer_descriptors=(
                finer_0_0,
                finer_1_0,
                finer_0_1,
                finer_1_1,
            ),
        )

    Sparse and edge regions may contribute fewer than four nonempty finer
    descriptors. Empty tiles are absent rather than represented by
    placeholders. This function organizes descriptors only; it does not open
    Zarr, read or rebase point arrays, sample candidates, or write output.
    """
    _require_spatial_transition(finer_level, coarser_level)
    if not isinstance(finer_descriptors, tuple) or not finer_descriptors:
        raise ValueError("`finer_descriptors` must be a nonempty tuple.")
    if not all(isinstance(descriptor, _TileDescriptor) for descriptor in finer_descriptors):
        raise ValueError("Every finer descriptor must be a _TileDescriptor.")

    descriptors_by_coarser_tile: dict[tuple[int, int], list[_TileDescriptor]] = defaultdict(list)
    for descriptor in finer_descriptors:
        if descriptor.level != finer_level.level:
            raise ValueError("Every finer descriptor must belong to the immediate-finer level.")
        if descriptor.tile_x >= finer_level.grid_width or descriptor.tile_y >= finer_level.grid_height:
            raise ValueError("A finer descriptor lies outside its planned grid.")
        coarser_tile_x = descriptor.tile_x // 2
        coarser_tile_y = descriptor.tile_y // 2
        if coarser_tile_x >= coarser_level.grid_width or coarser_tile_y >= coarser_level.grid_height:
            raise ValueError("A derived coarser tile lies outside the coarser grid.")
        descriptors_by_coarser_tile[(coarser_tile_y, coarser_tile_x)].append(descriptor)

    return tuple(
        _CoarserTileInput(
            tile_x=coarser_tile_x,
            tile_y=coarser_tile_y,
            finer_descriptors=tuple(sorted(descriptors, key=lambda descriptor: (descriptor.tile_y, descriptor.tile_x))),
        )
        for (coarser_tile_y, coarser_tile_x), descriptors in sorted(descriptors_by_coarser_tile.items())
    )


def _assign_spatial_buckets(
    coarser_tiles: tuple[_CoarserTileInput, ...],
    *,
    bucket_count: int,
) -> dict[int, tuple[_CoarserTileInput, ...]]:
    """Group coarser logical tiles by deterministic destination bucket."""
    tile_x = np.fromiter((tile.tile_x for tile in coarser_tiles), dtype=np.uint32, count=len(coarser_tiles))
    tile_y = np.fromiter((tile.tile_y for tile in coarser_tiles), dtype=np.uint32, count=len(coarser_tiles))
    bucket_ids = _tile_bucket_ids(tile_x, tile_y, bucket_count=bucket_count)
    grouped: dict[int, list[_CoarserTileInput]] = defaultdict(list)
    for tile, destination in zip(coarser_tiles, bucket_ids, strict=True):
        grouped[int(destination)].append(tile)
    return {
        destination: tuple(sorted(tiles, key=lambda tile: (tile.tile_y, tile.tile_x)))
        for destination, tiles in grouped.items()
    }


def _rebase_finer_coordinates(
    payload: _PointPayload,
    *,
    finer_tile_x: int,
    finer_tile_y: int,
    coarser_tile_x: int,
    coarser_tile_y: int,
    finer_tile_size: int,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Express one finer payload in its containing coarser tile frame.

    A coarser tile has twice the finer tile edge, so it covers a 2-by-2
    block of finer tile positions. Write the containing coarser grid coordinate
    as ``(cx, cy)``, where ``cx = coarser_tile_x`` and
    ``cy = coarser_tile_y``. Its finer tile positions occupy four quadrants::

        finer tile coordinates relative to coarser tile (cx, cy)

        (2cx,   2cy)                         (2cx+1, 2cy)
        +----------------+----------------+
        | quadrant (0,0) | quadrant (1,0) |
        +----------------+----------------+
        | quadrant (0,1) | quadrant (1,1) |
        +----------------+----------------+
        (2cx, 2cy+1)                         (2cx+1, 2cy+1)

    Rebasing translates the coordinate origin; it does not scale or otherwise
    move a point in physical space. For one finer-relative point ``(x, y)``::

        quadrant_x = finer_tile_x - 2 * coarser_tile_x
        quadrant_y = finer_tile_y - 2 * coarser_tile_y

        coarser_x_rel = x + quadrant_x * finer_tile_size
        coarser_y_rel = y + quadrant_y * finer_tile_size

    Parameters
    ----------
    payload
        Complete finer-tile payload whose relative coordinates are finite and
        lie in the closed interval ``[0, finer_tile_size]``.
    finer_tile_x, finer_tile_y
        Grid coordinates of the contributing immediate-finer tile.
    coarser_tile_x, coarser_tile_y
        Grid coordinates of the containing coarser tile.
    finer_tile_size
        Positive finer-tile edge in intrinsic source-coordinate units. The
        containing coarser tile has edge ``2 * finer_tile_size``.

    Returns
    -------
    x_rel, y_rel
        C-contiguous ``float32`` coordinates in the containing coarser tile's
        frame and closed interval ``[0, 2 * finer_tile_size]``. ``value_id``
        and ``point_id`` are not transformed by this coordinate-only helper.
    """
    if not isinstance(payload, _PointPayload):
        raise ValueError("`payload` must be a _PointPayload.")
    _require_integer_in_range(finer_tile_x, "finer_tile_x", maximum=_UINT32_MAX)
    _require_integer_in_range(finer_tile_y, "finer_tile_y", maximum=_UINT32_MAX)
    _require_integer_in_range(coarser_tile_x, "coarser_tile_x", maximum=_UINT32_MAX)
    _require_integer_in_range(coarser_tile_y, "coarser_tile_y", maximum=_UINT32_MAX)
    _require_integer_in_range(finer_tile_size, "finer_tile_size", minimum=1, maximum=_INT64_MAX)
    quadrant_x = finer_tile_x - 2 * coarser_tile_x
    quadrant_y = finer_tile_y - 2 * coarser_tile_y
    if quadrant_x not in (0, 1) or quadrant_y not in (0, 1):
        raise ValueError("The finer tile does not occupy a valid coarser-tile quadrant.")
    if (
        bool((payload.x_rel < 0).any())
        or bool((payload.x_rel > finer_tile_size).any())
        or bool((payload.y_rel < 0).any())
        or bool((payload.y_rel > finer_tile_size).any())
    ):
        raise ValueError("Finer coordinates must lie inside the finer tile.")

    with np.errstate(over="ignore", invalid="ignore"):
        x_rel = np.ascontiguousarray(
            payload.x_rel + np.float32(quadrant_x * finer_tile_size),
            dtype=np.float32,
        )
        y_rel = np.ascontiguousarray(
            payload.y_rel + np.float32(quadrant_y * finer_tile_size),
            dtype=np.float32,
        )
    coarser_tile_size = 2 * finer_tile_size
    if (
        not bool(np.isfinite(x_rel).all())
        or not bool(np.isfinite(y_rel).all())
        or bool((x_rel > coarser_tile_size).any())
        or bool((y_rel > coarser_tile_size).any())
    ):
        raise ValueError("Rebased coordinates must lie inside the coarser tile.")
    return x_rel, y_rel


def _assemble_coarser_candidates(
    coarser_tile: _CoarserTileInput,
    *,
    finer_level: _LevelBuildPlan,
    reader_cache: _BucketReaderCache,
) -> _PointPayload:
    """Read, rebase, and concatenate one coarser tile with a checked cursor."""
    if not isinstance(coarser_tile, _CoarserTileInput):
        raise ValueError("`coarser_tile` must be a _CoarserTileInput.")
    if not isinstance(finer_level, _LevelBuildPlan):
        raise ValueError("`finer_level` must be a _LevelBuildPlan.")

    candidate_count = coarser_tile.candidate_count
    x_rel = np.empty(candidate_count, dtype=np.float32)
    y_rel = np.empty(candidate_count, dtype=np.float32)
    value_id = np.empty(candidate_count, dtype=np.uint32)
    point_id = np.empty(candidate_count, dtype=np.uint64)
    cursor = 0
    for descriptor in coarser_tile.finer_descriptors:
        reader = reader_cache.get(level=descriptor.level, bucket_id=descriptor.bucket_id)
        payload = reader.read_construction_payload(descriptor)
        rebased_x, rebased_y = _rebase_finer_coordinates(
            payload,
            finer_tile_x=descriptor.tile_x,
            finer_tile_y=descriptor.tile_y,
            coarser_tile_x=coarser_tile.tile_x,
            coarser_tile_y=coarser_tile.tile_y,
            finer_tile_size=finer_level.tile_size,
        )
        stop = cursor + descriptor.n_points
        x_rel[cursor:stop] = rebased_x
        y_rel[cursor:stop] = rebased_y
        value_id[cursor:stop] = payload.value_id
        point_id[cursor:stop] = payload.point_id
        cursor = stop
        # Combined arrays now own this contributor's rows. The cache alone owns
        # the entered reader; release tile-scoped decoded arrays immediately.
        del reader, payload, rebased_x, rebased_y

    if cursor != candidate_count:
        raise ValueError("Assembled rows do not match the descriptor-derived candidate count.")
    return _PointPayload(x_rel=x_rel, y_rel=y_rel, value_id=value_id, point_id=point_id)


def _write_spatial_bucket(
    *,
    bucket_id: int,
    coarser_tiles: tuple[_CoarserTileInput, ...],
    finer_level: _LevelBuildPlan,
    coarser_level: _LevelBuildPlan,
    staging_root: Path,
    settings: _ZarrWriteSettings,
    reader_cache: _BucketReaderCache,
) -> _BucketWriteResult:
    """Read, assemble, sample, and write one Spatial destination bucket."""
    capacity = _require_spatial_transition(finer_level, coarser_level)
    planned_tiles = tuple(
        _PlannedTile(
            tile_x=tile.tile_x,
            tile_y=tile.tile_y,
            n_points=min(tile.candidate_count, capacity),
        )
        for tile in coarser_tiles
    )
    bucket_plan = _BucketPlan(
        level=coarser_level.level,
        bucket_id=bucket_id,
        tiles=planned_tiles,
        settings=settings,
    )

    with _BucketWriter(staging_root, bucket_plan) as writer:
        for coarser_tile, planned_tile in zip(coarser_tiles, planned_tiles, strict=True):
            candidates = _assemble_coarser_candidates(
                coarser_tile,
                finer_level=finer_level,
                reader_cache=reader_cache,
            )
            selected_indices = _select_sampled_tile_indices(
                candidates.x_rel,
                candidates.y_rel,
                candidates.point_id,
                level=coarser_level.level,
                tile_x=coarser_tile.tile_x,
                tile_y=coarser_tile.tile_y,
                tile_size=coarser_level.tile_size,
                target=capacity,
            )
            retained = candidates.take(selected_indices)
            if retained.n_points != planned_tile.n_points:
                raise ValueError("The sampled Spatial tile does not match its planned point count.")
            writer.write_tile(coarser_tile.tile_x, coarser_tile.tile_y, retained)
            # `write_tile` copied retained rows into writer-owned buffers. Drop
            # coarser-tile-scoped arrays before the next tile or bucket finalization.
            del candidates, selected_indices, retained
        return writer.finalize()


def _reconcile_spatial_result(
    bucket_results: tuple[_BucketWriteResult, ...],
    *,
    coarser_tiles: tuple[_CoarserTileInput, ...],
    finer_result: _LevelWriteResult,
    coarser_level: _LevelBuildPlan,
) -> _LevelWriteResult:
    """Reconcile finalized Spatial descriptors with immediate-finer facts."""
    result = _LevelWriteResult(buckets=bucket_results)
    if result.level != coarser_level.level:
        raise ValueError("Spatial construction produced the wrong serialized level.")
    capacity = coarser_level.max_points_per_tile
    if capacity is None:
        raise ValueError("A Spatial level must have a per-tile capacity.")
    expected = {(tile.tile_x, tile.tile_y): min(tile.candidate_count, capacity) for tile in coarser_tiles}
    observed = {(descriptor.tile_x, descriptor.tile_y): descriptor.n_points for descriptor in result.tile_descriptors}
    if observed != expected:
        raise ValueError("Spatial tile coordinates or point counts do not reconcile to the finer level.")
    expected_point_count = sum(expected.values())
    if result.point_count != expected_point_count:
        raise ValueError("Spatial bucket rows do not reconcile to the expected sampled point count.")
    if result.point_count > coarser_level.point_count_upper_bound:
        raise ValueError("Spatial point count exceeds its planned upper bound.")
    if result.point_count > finer_result.point_count:
        raise ValueError("Spatial point count exceeds the immediate-finer point count.")
    if any(
        descriptor.tile_x >= coarser_level.grid_width or descriptor.tile_y >= coarser_level.grid_height
        for descriptor in result.tile_descriptors
    ):
        raise ValueError("Final Spatial descriptors fall outside the planned grid.")
    return result
