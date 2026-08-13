from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from napari_harpy.core.multi_scale_cache_points_zarr.models import (
    _INT16_MAX,
    _INT64_MAX,
    _UINT32_MAX,
    _bucket_path,
    _require_integer_in_range,
    _TileDescriptor,
)


@dataclass(frozen=True)
class _PlannedTile:
    """Describe one nonempty tile expected in a bucket write.

    This contains no point data. ``n_points`` is the expected row count that a
    later ``_PointPayload`` must supply for ``(tile_x, tile_y)``.
    """

    tile_x: int
    tile_y: int
    n_points: int

    def __post_init__(self) -> None:
        _require_integer_in_range(self.tile_x, "tile_x", maximum=_UINT32_MAX)
        _require_integer_in_range(self.tile_y, "tile_y", maximum=_UINT32_MAX)
        _require_integer_in_range(self.n_points, "n_points", minimum=1, maximum=_INT64_MAX)


@dataclass(frozen=True)
class _ZarrWriteSettings:
    """Describe provisional physical row grouping without importing Zarr."""

    point_chunk_rows: int
    point_shard_rows: int
    range_chunk_rows: int
    range_shard_rows: int
    codec_id: str

    def __post_init__(self) -> None:
        _require_integer_in_range(
            self.point_chunk_rows,
            "point_chunk_rows",
            minimum=1,
            maximum=_INT64_MAX,
        )
        _require_integer_in_range(
            self.point_shard_rows,
            "point_shard_rows",
            minimum=1,
            maximum=_INT64_MAX,
        )
        _require_integer_in_range(
            self.range_chunk_rows,
            "range_chunk_rows",
            minimum=1,
            maximum=_INT64_MAX,
        )
        _require_integer_in_range(
            self.range_shard_rows,
            "range_shard_rows",
            minimum=1,
            maximum=_INT64_MAX,
        )
        if self.point_shard_rows % self.point_chunk_rows:
            raise ValueError("`point_shard_rows` must be an integer multiple of `point_chunk_rows`.")
        if self.range_shard_rows % self.range_chunk_rows:
            raise ValueError("`range_shard_rows` must be an integer multiple of `range_chunk_rows`.")
        if not isinstance(self.codec_id, str) or self.codec_id == "":
            raise ValueError("`codec_id` must be a nonempty versioned string.")


@dataclass(frozen=True)
class _BucketPlan:
    """Plan one nonempty independent Zarr bucket before physical writing.

    This is the bucket-wide write contract, not a container for point data. It
    fixes the bucket identity, ordered nonempty tiles, expected count of each
    tile, derived path, total and offsets, and physical Zarr settings. It is
    therefore small relative to the points being written. ``bucket_path`` is a
    canonical property of ``level`` and ``bucket_id``, not independent state.

    The bucket writer retains this plan while callers provide one
    ``_PointPayload`` at a time. For each call, the writer matches the supplied
    tile coordinates and ``payload.n_points`` to the next ``_PlannedTile``
    before writing its arrays. The plan expresses expected input; payloads are
    the supplied point data; finalized Zarr shapes are the observed output.
    These three sources are reconciled rather than treated as interchangeable.

    Keeping the plan and payload separate also keeps sampling and rebasing
    independent of bucket paths, neighboring tiles, chunks, shards, and codecs.
    """

    level: int
    bucket_id: int
    tiles: tuple[_PlannedTile, ...]
    settings: _ZarrWriteSettings

    def __post_init__(self) -> None:
        _require_integer_in_range(self.level, "level", maximum=_INT16_MAX)
        _require_integer_in_range(self.bucket_id, "bucket_id", maximum=_UINT32_MAX)
        if not isinstance(self.tiles, tuple) or not self.tiles:
            raise ValueError("A bucket plan must contain at least one planned tile.")
        if not all(isinstance(tile, _PlannedTile) for tile in self.tiles):
            raise ValueError("`tiles` must be a tuple of _PlannedTile values.")
        if not isinstance(self.settings, _ZarrWriteSettings):
            raise ValueError("`settings` must be _ZarrWriteSettings.")
        if self.tiles != tuple(sorted(self.tiles, key=lambda tile: (tile.tile_y, tile.tile_x))):
            raise ValueError("Planned tiles must be strictly ordered by (tile_y, tile_x).")
        coordinates = {(tile.tile_x, tile.tile_y) for tile in self.tiles}
        if len(coordinates) != len(self.tiles):
            raise ValueError("Planned tile coordinates must be unique.")
        if self.point_count > _INT64_MAX:
            raise ValueError("Bucket point count exceeds the supported int64 range.")

    @property
    def bucket_path(self) -> str:
        """Return the canonical cache-relative Zarr path for this bucket."""
        return _bucket_path(level=self.level, bucket_id=self.bucket_id)

    @property
    def tile_count(self) -> int:
        """Return the number of nonempty planned tiles."""
        return len(self.tiles)

    @property
    def point_count(self) -> int:
        """Return the total planned point count."""
        return sum(tile.n_points for tile in self.tiles)

    @property
    def tile_offset(self) -> np.ndarray:
        """Return exact uint64 prefix sums for the planned tile intervals."""
        offsets = np.empty(self.tile_count + 1, dtype=np.uint64)
        offsets[0] = 0
        np.cumsum(
            np.fromiter((tile.n_points for tile in self.tiles), dtype=np.uint64, count=self.tile_count),
            out=offsets[1:],
        )
        offsets.flags.writeable = False
        return offsets


@dataclass(frozen=True)
class _BucketWriteResult:
    """Describe one finalized nonempty Zarr bucket.

    Parameters
    ----------
    tile_descriptors
        One descriptor for every nonempty logical tile in the finalized bucket,
        ordered by ``(tile_y, tile_x)``. All descriptors have the same level,
        bucket ID, and bucket path, and their bucket-local indexes are exactly
        ``0..K-1``. They are standalone tile addresses that later become
        manifest rows.
    point_count
        Observed number of rows in each finalized physical point array. The
        bucket finalizer obtains this value from the Zarr array shape rather
        than deriving it from the descriptors. Construction reconciles it with
        the writer cursor, terminal ``tile_offset``, bucket plan, and sum of
        descriptor ``n_points`` values.
    range_count
        Observed number of records in each finalized physical sparse-range
        array. One record represents one nonempty ``(logical tile, value_id)``
        combination whose points form a contiguous value run. The same value in
        three tiles therefore contributes three ranges. Finalization reconciles
        this count with the range cursor and terminal ``tile_indptr``. It is not
        the number of points, tiles, or globally distinct values.

    Notes
    -----
    Bucket identity is stored only by the standalone tile descriptors, which
    later become manifest rows. ``level``, ``bucket_id``, and ``bucket_path``
    are derived from their shared identity rather than duplicated here.
    ``point_count`` remains independent so finalization can reconcile the
    physical Zarr row count with the sum of the descriptor counts.
    """

    tile_descriptors: tuple[_TileDescriptor, ...]
    point_count: int
    range_count: int

    def __post_init__(self) -> None:
        _require_integer_in_range(self.point_count, "point_count", minimum=1, maximum=_INT64_MAX)
        _require_integer_in_range(self.range_count, "range_count", minimum=1, maximum=_INT64_MAX)
        if not isinstance(self.tile_descriptors, tuple) or not self.tile_descriptors:
            raise ValueError("A bucket result must contain at least one tile descriptor.")
        if not all(isinstance(tile, _TileDescriptor) for tile in self.tile_descriptors):
            raise ValueError("`tile_descriptors` must be a tuple of _TileDescriptor values.")

        identity = (self.tile_descriptors[0].level, self.tile_descriptors[0].bucket_id)
        if any(
            (tile.level, tile.bucket_id) != identity
            for tile in self.tile_descriptors
        ):
            raise ValueError("Every tile descriptor in a bucket result must have the same bucket identity.")
        if tuple(tile.bucket_tile_index for tile in self.tile_descriptors) != tuple(range(len(self.tile_descriptors))):
            raise ValueError("Bucket-local tile indexes must be contiguous from zero.")
        if self.tile_descriptors != tuple(sorted(self.tile_descriptors, key=lambda tile: (tile.tile_y, tile.tile_x))):
            raise ValueError("Bucket tile descriptors must follow (tile_y, tile_x) order.")
        if len({(tile.tile_x, tile.tile_y) for tile in self.tile_descriptors}) != len(self.tile_descriptors):
            raise ValueError("Bucket tile coordinates must be unique.")
        if sum(tile.n_points for tile in self.tile_descriptors) != self.point_count:
            raise ValueError("Bucket descriptor rows do not match `point_count`.")
        if not len(self.tile_descriptors) <= self.range_count <= self.point_count:
            raise ValueError("`range_count` must lie between tile count and point count.")

    @property
    def level(self) -> int:
        """Return the common serialized level of the bucket descriptors."""
        return self.tile_descriptors[0].level

    @property
    def bucket_id(self) -> int:
        """Return the common bucket identifier of the bucket descriptors."""
        return self.tile_descriptors[0].bucket_id

    @property
    def bucket_path(self) -> str:
        """Return the common Zarr path of the bucket descriptors."""
        return self.tile_descriptors[0].bucket_path


@dataclass(frozen=True)
class _LevelWriteResult:
    """Describe all finalized nonempty buckets of one constructed level.

    The serialized ``level`` is derived from the nonempty bucket results, which
    in turn derive it from their descriptors. This keeps descriptor identity as
    the single source of truth throughout the result hierarchy.
    """

    buckets: tuple[_BucketWriteResult, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.buckets, tuple) or not self.buckets:
            raise ValueError("A level result must contain at least one bucket result.")
        if not all(isinstance(bucket, _BucketWriteResult) for bucket in self.buckets):
            raise ValueError("`buckets` must be a tuple of _BucketWriteResult values.")
        if any(bucket.level != self.buckets[0].level for bucket in self.buckets):
            raise ValueError("Every bucket result in a level result must have the same level.")
        if tuple(bucket.bucket_id for bucket in self.buckets) != tuple(
            sorted(bucket.bucket_id for bucket in self.buckets)
        ):
            raise ValueError("Bucket results must be ordered by bucket_id.")
        if len({bucket.bucket_id for bucket in self.buckets}) != len(self.buckets):
            raise ValueError("Level bucket IDs must be unique.")
        tiles = self.tile_descriptors
        if len({(tile.tile_x, tile.tile_y) for tile in tiles}) != len(tiles):
            raise ValueError("Level tile coordinates must be unique.")
        if len({(tile.bucket_id, tile.bucket_tile_index) for tile in tiles}) != len(tiles):
            raise ValueError("Level bucket ID/index keys must be unique.")

    @property
    def level(self) -> int:
        """Return the common serialized level of all bucket results."""
        return self.buckets[0].level

    @property
    def tile_descriptors(self) -> tuple[_TileDescriptor, ...]:
        """Return all descriptors in global (tile_y, tile_x) order."""
        return tuple(
            sorted(
                (tile for bucket in self.buckets for tile in bucket.tile_descriptors),
                key=lambda tile: (tile.tile_y, tile.tile_x),
            )
        )

    @property
    def bucket_count(self) -> int:
        """Return the number of nonempty physical buckets."""
        return len(self.buckets)

    @property
    def tile_count(self) -> int:
        """Return the number of nonempty logical tiles."""
        return sum(len(bucket.tile_descriptors) for bucket in self.buckets)

    @property
    def point_count(self) -> int:
        """Return the stored level point total."""
        return sum(bucket.point_count for bucket in self.buckets)

    @property
    def range_count(self) -> int:
        """Return the sum of the buckets' observed physical range counts."""
        return sum(bucket.range_count for bucket in self.buckets)
