from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath

_INT16_MAX = 2**15 - 1
_INT32_MAX = 2**31 - 1
_INT64_MAX = 2**63 - 1
_UINT32_MAX = 2**32 - 1
_UINT64_MAX = 2**64 - 1


@dataclass(frozen=True)
class _ExactLevelWriterConfig:
    """Private physical execution settings for the Exact-level writer.

    Parameters
    ----------
    max_source_rows_per_partition
        Maximum selected source rows materialized in one Harpy-owned Dask input
        partition. A final partition may contain fewer rows. This bounds the
        decoded and annotated rows held by one source task; it does not change
        Exact membership, point identity, or final tile assignment.
    bucket_count
        Number of deterministic logical output buckets used by the local disk
        shuffle. Every logical tile maps to exactly one bucket, while one bucket
        may contain several tiles. Empty buckets need not create final files.
    max_rows_per_row_group
        Maximum points written to one physical Parquet row group. A denser tile
        is split into deterministic row-group shards of at most this size. This
        is physical sharding only; it never samples or removes Exact points.
    finalizer_concurrency
        Maximum number of complete shuffle buckets that may be computed,
        sorted, grouped by tile, and written concurrently.
    """

    max_source_rows_per_partition: int
    bucket_count: int
    max_rows_per_row_group: int
    finalizer_concurrency: int

    def __post_init__(self) -> None:
        _require_positive_integer(self.max_source_rows_per_partition, "max_source_rows_per_partition")
        _require_positive_integer(self.bucket_count, "bucket_count")
        _require_positive_integer(self.max_rows_per_row_group, "max_rows_per_row_group")
        _require_positive_integer(self.finalizer_concurrency, "finalizer_concurrency")


@dataclass(frozen=True)
class _ManifestRow:
    """Provisional description of one physical row group for one logical tile."""

    level: int
    level_file: str
    tile_x: int
    tile_y: int
    n_points: int
    row_group: int
    tile_shard: int

    def __post_init__(self) -> None:
        _require_integer_in_range(self.level, "level", maximum=_INT16_MAX)
        _require_cache_relative_path(self.level_file, level=self.level)
        _require_integer_in_range(self.tile_x, "tile_x", maximum=_UINT32_MAX)
        _require_integer_in_range(self.tile_y, "tile_y", maximum=_UINT32_MAX)
        _require_integer_in_range(self.n_points, "n_points", minimum=1, maximum=_INT64_MAX)
        _require_integer_in_range(self.row_group, "row_group", maximum=_INT32_MAX)
        _require_integer_in_range(self.tile_shard, "tile_shard", maximum=_INT32_MAX)


@dataclass(frozen=True)
class _TileValueCount:
    """Provisional nonzero count for one value in one logical cache tile."""

    level: int
    value_id: int
    tile_x: int
    tile_y: int
    n_points: int

    def __post_init__(self) -> None:
        _require_integer_in_range(self.level, "level", maximum=_INT16_MAX)
        _require_integer_in_range(self.value_id, "value_id", maximum=_UINT32_MAX)
        _require_integer_in_range(self.tile_x, "tile_x", maximum=_UINT32_MAX)
        _require_integer_in_range(self.tile_y, "tile_y", maximum=_UINT32_MAX)
        _require_integer_in_range(self.n_points, "n_points", minimum=1, maximum=_UINT64_MAX)


@dataclass(frozen=True)
class _LevelWriteResult:
    """Provisional manifest and value-count records emitted for one cache level."""

    manifest_rows: tuple[_ManifestRow, ...]
    tile_value_counts: tuple[_TileValueCount, ...]


def _require_positive_integer(value: object, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"`{name}` must be a positive integer.")


def _require_integer_in_range(
    value: object,
    name: str,
    *,
    minimum: int = 0,
    maximum: int,
) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or not minimum <= value <= maximum:
        raise ValueError(f"`{name}` must be an integer in the range [{minimum}, {maximum}].")


def _require_cache_relative_path(value: object, *, level: int) -> None:
    if not isinstance(value, str) or value == "":
        raise ValueError("`level_file` must be a non-empty cache-root-relative POSIX path.")

    relative_path = PurePosixPath(value)
    if (
        not relative_path.parts
        or relative_path.is_absolute()
        or ".." in relative_path.parts
        or relative_path.as_posix() != value
    ):
        raise ValueError(f"`level_file` `{value}` is not a normalized cache-root-relative POSIX path.")

    expected_directory = PurePosixPath(f"levels/level_{level}")
    if relative_path.parent != expected_directory:
        raise ValueError(f"`level_file` must be directly inside `{expected_directory.as_posix()}`.")
