from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath

_INT16_MAX = 2**15 - 1
_INT32_MAX = 2**31 - 1
_INT64_MAX = 2**63 - 1
_UINT32_MAX = 2**32 - 1


@dataclass(frozen=True)
class _ExactLevelWriterConfig:
    """Private physical execution settings for the Exact-level writer.

    Parameters
    ----------
    bucket_count
        Number of deterministic logical output buckets used by the local disk
        shuffle. Every logical tile maps to exactly one bucket, while one bucket
        may contain several tiles. Empty buckets need not create final files.
    max_rows_per_row_group
        Maximum points written to one physical Parquet row group. A denser tile
        is split into deterministic row-group shards of at most this size. This
        is physical sharding only; it never samples or removes Exact points.
    dask_worker_count
        Number of local threads available to the Dask scheduler for the complete
        read, annotation, disk-redistribution, sorting, and writing graph. These
        are local threaded-scheduler workers, not distributed processes.

    Notes
    -----
    Input partitioning is not configurable. The writer constructs one Dask
    input partition per validated physical Parquet file, so those source files
    determine the input-partition sizes.
    """

    bucket_count: int
    max_rows_per_row_group: int
    dask_worker_count: int

    def __post_init__(self) -> None:
        _require_positive_integer(self.bucket_count, "bucket_count")
        _require_positive_integer(self.max_rows_per_row_group, "max_rows_per_row_group")
        _require_positive_integer(self.dask_worker_count, "dask_worker_count")


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
        level_file = _require_cache_relative_path(self.level_file, name="level_file")
        expected_directory = PurePosixPath(f"levels/level_{self.level}")
        if level_file.parent != expected_directory:
            raise ValueError(f"`level_file` must be directly inside `{expected_directory.as_posix()}`.")
        _require_integer_in_range(self.tile_x, "tile_x", maximum=_UINT32_MAX)
        _require_integer_in_range(self.tile_y, "tile_y", maximum=_UINT32_MAX)
        _require_integer_in_range(self.n_points, "n_points", minimum=1, maximum=_INT64_MAX)
        _require_integer_in_range(self.row_group, "row_group", maximum=_INT32_MAX)
        _require_integer_in_range(self.tile_shard, "tile_shard", maximum=_INT32_MAX)


@dataclass(frozen=True)
class _IntermediateTileValueCountFile:
    """Describe one construction-only file of flat tile/value counts.

    Parameters
    ----------
    level
        Non-negative serialized level number shared by every count row in the
        intermediate file.
    relative_path
        Normalized cache-root-relative POSIX path to the staged intermediate
        file. The tile writer owns its directory and filename convention.
    row_count
        Number of aggregated nonzero tile/value rows in the intermediate file,
        not the number of original point rows.

    Notes
    -----
    The counts are exact for one finalized bucket. A later construction step
    consumes this file into the global ``tile_value_counts.parquet`` index and
    removes it before publishing the completed cache.
    """

    level: int
    relative_path: str
    row_count: int

    def __post_init__(self) -> None:
        _require_integer_in_range(self.level, "level", maximum=_INT16_MAX)
        _require_cache_relative_path(self.relative_path, name="relative_path")
        _require_integer_in_range(self.row_count, "row_count", minimum=1, maximum=_INT64_MAX)


@dataclass(frozen=True)
class _LevelWriteResult:
    """Manifest rows and intermediate count-file descriptors for one level."""

    manifest_rows: tuple[_ManifestRow, ...]
    intermediate_tile_value_count_files: tuple[_IntermediateTileValueCountFile, ...]


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


def _require_cache_relative_path(value: object, *, name: str) -> PurePosixPath:
    if not isinstance(value, str) or value == "":
        raise ValueError(f"`{name}` must be a non-empty cache-root-relative POSIX path.")

    relative_path = PurePosixPath(value)
    if (
        not relative_path.parts
        or relative_path.is_absolute()
        or ".." in relative_path.parts
        or relative_path.as_posix() != value
    ):
        raise ValueError(f"`{name}` `{value}` is not a normalized cache-root-relative POSIX path.")
    return relative_path
