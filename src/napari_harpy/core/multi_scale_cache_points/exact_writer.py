from __future__ import annotations

import math
import tempfile
from dataclasses import dataclass
from pathlib import Path

import dask
import dask.dataframe as dd
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from napari_harpy.core.multi_scale_cache_points.build_plan import (
    _LevelBuildPlan,
    _PointsCacheBuildPlan,
)
from napari_harpy.core.multi_scale_cache_points.models import ParquetSourceFile, ValidatedPointsSource
from napari_harpy.core.multi_scale_cache_points.value_normalization import (
    VALUE_NORMALIZATION_METHOD,
    _normalized_row_values,
)
from napari_harpy.core.multi_scale_cache_points.writer_models import (
    _ExactLevelWriterConfig,
    _IntermediateTileValueCountFile,
    _LevelWriteResult,
    _ManifestRow,
)

BUCKET_HASH_METHOD = "harpy-tile-splitmix64-v1"
TARGET_ROWS_PER_OUTPUT_BUCKET = 2_000_000
DEFAULT_MAX_ROWS_PER_ROW_GROUP = 1_000_000
DEFAULT_DASK_WORKER_COUNT = 1

_INTERMEDIATE_TILE_VALUE_COUNTS_DIRECTORY = "intermediate_tile_value_counts"
_INTERMEDIATE_COUNT_BUFFER_ROWS = 65_536
_UINT64_32 = np.uint64(32)
_UINT64_30 = np.uint64(30)
_UINT64_27 = np.uint64(27)
_UINT64_31 = np.uint64(31)
_SPLITMIX64_INCREMENT = np.uint64(0x9E3779B97F4A7C15)
_SPLITMIX64_MULTIPLIER_1 = np.uint64(0xBF58476D1CE4E5B9)
_SPLITMIX64_MULTIPLIER_2 = np.uint64(0x94D049BB133111EB)

_EXACT_PAYLOAD_SCHEMA = pa.schema(
    [
        pa.field("x_rel", pa.float32(), nullable=False),
        pa.field("y_rel", pa.float32(), nullable=False),
        pa.field("value_id", pa.uint32(), nullable=False),
        pa.field("point_id", pa.uint64(), nullable=False),
    ]
)
_TILE_VALUE_COUNT_SCHEMA = pa.schema(
    [
        pa.field("level", pa.int16(), nullable=False),
        pa.field("value_id", pa.uint32(), nullable=False),
        pa.field("tile_x", pa.uint32(), nullable=False),
        pa.field("tile_y", pa.uint32(), nullable=False),
        pa.field("n_points", pa.uint64(), nullable=False),
    ]
)


@dataclass(frozen=True)
class _BucketWriteResult:
    """Return the compact output of one bucket finalizer to the coordinator.

    Parameters
    ----------
    bucket_id
        Deterministic identifier used to order independently finalized buckets.
    point_count
        Number of point rows consumed and written by this bucket. The
        coordinator reconciles its level-wide sum with the validated source row
        count.
    value_count_total
        Number of points represented by this bucket's intermediate tile/value
        counts. Its level-wide sum is independently reconciled with the
        validated source row count.
    manifest_rows
        Descriptors for the physical point row groups written by this bucket.
    intermediate_value_count_file
        Descriptor for the bucket's intermediate tile/value-count file, or
        ``None`` when the bucket is empty and writes no files.
    """

    bucket_id: int
    point_count: int
    value_count_total: int
    manifest_rows: tuple[_ManifestRow, ...]
    intermediate_value_count_file: _IntermediateTileValueCountFile | None


class _IntermediateTileValueCountWriter:
    """Write one bucket-local intermediate tile/value-count Parquet file.

    Each instance exclusively writes the intermediate tile/value-count
    companion to one staged point-payload file. Both files cover the same cache
    level and logical tiles; this writer owns only the intermediate count file.
    For every complete logical tile, ``append`` receives the distinct value IDs
    present in that tile and their corresponding point counts. It emits one flat
    row per nonzero ``(level, value_id, tile_x, tile_y)`` key.

    Rows are buffered and periodically flushed using
    ``_TILE_VALUE_COUNT_SCHEMA``. ``row_count`` tracks the number of emitted
    tile/value rows, while ``point_count`` tracks the sum of their ``n_points``
    values.

    ``close`` flushes the remaining rows and closes the Parquet writer. The
    resulting file is a construction-only handoff artifact; a later step
    consolidates it with the other bucket-local files into
    ``tile_value_counts.parquet`` and removes it before cache publication.

    Parameters
    ----------
    path
        Filesystem path of the bucket-local intermediate Parquet file.
    level
        Cache level recorded on every intermediate count row.
    """

    def __init__(self, path: Path, *, level: int) -> None:
        self._path = path
        self._level = level
        self._writer = pq.ParquetWriter(
            path,
            _TILE_VALUE_COUNT_SCHEMA,
            compression="snappy",
            use_dictionary=False,
        )
        self._value_ids: list[np.ndarray] = []
        self._tile_x: list[np.ndarray] = []
        self._tile_y: list[np.ndarray] = []
        self._counts: list[np.ndarray] = []
        self._buffered_rows = 0
        self.row_count = 0
        self.point_count = 0

    def append(self, *, tile_x: int, tile_y: int, value_ids: np.ndarray, counts: np.ndarray) -> None:
        row_count = len(value_ids)
        if row_count == 0:
            return
        self._value_ids.append(value_ids.astype(np.uint32, copy=False))
        self._tile_x.append(np.full(row_count, tile_x, dtype=np.uint32))
        self._tile_y.append(np.full(row_count, tile_y, dtype=np.uint32))
        self._counts.append(counts.astype(np.uint64, copy=False))
        self._buffered_rows += row_count
        self.row_count += row_count
        self.point_count += int(counts.sum(dtype=np.uint64))
        if self._buffered_rows >= _INTERMEDIATE_COUNT_BUFFER_ROWS:
            self._flush()

    def close(self) -> None:
        self._flush()
        self._writer.close()

    def _flush(self) -> None:
        if self._buffered_rows == 0:
            return
        table = pa.Table.from_arrays(
            [
                pa.array(np.full(self._buffered_rows, self._level, dtype=np.int16), type=pa.int16()),
                pa.array(np.concatenate(self._value_ids), type=pa.uint32()),
                pa.array(np.concatenate(self._tile_x), type=pa.uint32()),
                pa.array(np.concatenate(self._tile_y), type=pa.uint32()),
                pa.array(np.concatenate(self._counts), type=pa.uint64()),
            ],
            schema=_TILE_VALUE_COUNT_SCHEMA,
        )
        self._writer.write_table(table, row_group_size=_INTERMEDIATE_COUNT_BUFFER_ROWS)
        self._value_ids.clear()
        self._tile_x.clear()
        self._tile_y.clear()
        self._counts.clear()
        self._buffered_rows = 0


def _bucket_count_for_level(level: _LevelBuildPlan) -> int:
    """Return the deterministic physical bucket count for a planned level."""
    return max(1, math.ceil(level.point_count_upper_bound / TARGET_ROWS_PER_OUTPUT_BUCKET))


def _tile_bucket_ids(tile_x: np.ndarray, tile_y: np.ndarray, *, bucket_count: int) -> np.ndarray:
    """Map uint32 tile coordinates through the versioned SplitMix64 policy."""
    if not isinstance(bucket_count, int) or isinstance(bucket_count, bool) or bucket_count <= 0:
        raise ValueError("`bucket_count` must be a positive integer.")
    x = np.asarray(tile_x, dtype=np.uint64)
    y = np.asarray(tile_y, dtype=np.uint64)
    if x.shape != y.shape:
        raise ValueError("`tile_x` and `tile_y` must have matching shapes.")

    tile_key = (y << _UINT64_32) | x
    mixed = tile_key + _SPLITMIX64_INCREMENT
    mixed = (mixed ^ (mixed >> _UINT64_30)) * _SPLITMIX64_MULTIPLIER_1
    mixed = (mixed ^ (mixed >> _UINT64_27)) * _SPLITMIX64_MULTIPLIER_2
    tile_hash = mixed ^ (mixed >> _UINT64_31)
    return tile_hash % np.uint64(bucket_count)


def _write_exact_level(
    validated: ValidatedPointsSource,
    plan: _PointsCacheBuildPlan,
    *,
    staging_directory: Path,
    temporary_directory_root: Path,
    config: _ExactLevelWriterConfig,
) -> _LevelWriteResult:
    """Write Exact points as tile-grouped Parquet files.

    Each point belongs to one logical spatial tile. To bring together points
    from the same tile, even when they originate in different source files, the
    writer deterministically assigns every tile to one of a fixed number of
    physical output groups. These output groups are called buckets. One bucket
    may contain several complete logical tiles, but one logical tile belongs to
    exactly one bucket.

    Construction follows this physical flow::

        file-aligned source partitions
            -> assign every point to logical (tile_x, tile_y)
            -> hash each logical tile to one bucket_id
            -> disk-redistribute points into partitions by bucket_id
            -> sort each complete bucket by (tile_y, tile_x, point_id)
            -> write contiguous tile runs as Parquet row groups

    For example, annotated source partitions may initially contain::

        source partition 0:
            (tile_y=0, tile_x=0, point_id=5, bucket_id=7)
            (tile_y=0, tile_x=2, point_id=1, bucket_id=3)

        source partition 1:
            (tile_y=0, tile_x=0, point_id=2, bucket_id=7)
            (tile_y=1, tile_x=1, point_id=9, bucket_id=7)

    The disk redistribution first co-locates rows by bucket without relying on
    their arrival order::

        bucket 3:
            (tile_y=0, tile_x=2, point_id=1)

        bucket 7:
            (tile_y=0, tile_x=0, point_id=5)
            (tile_y=0, tile_x=0, point_id=2)
            (tile_y=1, tile_x=1, point_id=9)

    Each complete bucket is then sorted by ``(tile_y, tile_x, point_id)``::

        bucket 7:
            (tile_y=0, tile_x=0, point_id=2)
            (tile_y=0, tile_x=0, point_id=5)
            (tile_y=1, tile_x=1, point_id=9)

    This makes every logical tile contiguous and gives its points a
    deterministic order before row-group writing.

    ``config.bucket_count`` controls the number of shuffle destinations and
    potential output files. It does not impose a strict row limit on a bucket:
    tile density and hash distribution determine the actual bucket sizes.

    Each physical row group contains points from exactly one logical tile. A
    tile larger than ``config.max_rows_per_row_group`` is split into multiple
    deterministically ordered row-group shards.

    The disk redistribution uses a uniquely named child directory under
    ``temporary_directory_root`` for Dask scratch data. This directory contains
    no cache artifacts and is removed when computation exits, including after
    an ordinary exception. ``temporary_directory_root`` itself remains
    caller-owned.

    Final point files and intermediate tile/value-count files are written
    separately under ``staging_directory``. The intermediate files remain until
    a later construction step creates ``tile_value_counts.parquet``. Neither is
    removed by the shuffle-directory context; the higher-level builder owns
    cleanup of an incomplete staging generation.
    """
    exact = plan.levels[0]
    expected_bucket_count = _bucket_count_for_level(exact)
    if config.bucket_count != expected_bucket_count:
        raise ValueError(
            f"`bucket_count` must equal {expected_bucket_count} for the Exact level's "
            f"{exact.point_count_upper_bound} planned points."
        )
    if validated.value_normalization_method != VALUE_NORMALIZATION_METHOD:
        raise ValueError("The validated source uses an unsupported value-normalization method.")
    if not staging_directory.is_dir():
        raise ValueError("`staging_directory` must be an existing directory.")
    temporary_directory_root.mkdir(parents=True, exist_ok=True)

    level_directory = staging_directory / exact.relative_directory
    intermediate_count_directory = (
        staging_directory / _INTERMEDIATE_TILE_VALUE_COUNTS_DIRECTORY / f"level_{exact.level}"
    )
    for path in (level_directory, intermediate_count_directory):
        if path.exists():
            raise FileExistsError(f"Exact-level output path already exists: `{path}`.")
        path.mkdir(parents=True)

    # Materialize the validated vocabulary in ID order; each tuple position is
    # the canonical uint32 value ID defined by `ValidatedPointsSource.value_table`.
    value_labels_by_id: tuple[str, ...] = tuple(validated.value_table["value"].to_pylist())
    annotated = _build_annotated_source(
        validated,
        exact=exact,
        x_origin=plan.x_origin,
        y_origin=plan.y_origin,
        bucket_count=config.bucket_count,
        value_labels_by_id=value_labels_by_id,
    )
    bucketed = annotated.set_index(
        "bucket_id",
        divisions=list(range(config.bucket_count + 1)),
        shuffle_method="disk",
        drop=False,
    )

    filename_width = max(3, len(str(config.bucket_count - 1)))
    # Isolate disposable Dask shuffle scratch from the staged cache output.
    with tempfile.TemporaryDirectory(
        prefix="napari-harpy-points-shuffle-",
        dir=temporary_directory_root,
    ) as shuffle_directory:
        finalizers = [
            # Each finalizer writes a point file and an intermediate count file.
            # Mark it impure so Dask does not treat this side-effecting call as
            # a reusable or deduplicatable computation.
            dask.delayed(_finalize_bucket, pure=False)(
                partition,
                bucket_id=bucket_id,
                filename_width=filename_width,
                level=exact.level,
                level_directory=level_directory,
                intermediate_count_directory=intermediate_count_directory,
                staging_directory=staging_directory,
                max_rows_per_row_group=config.max_rows_per_row_group,
            )
            for bucket_id, partition in enumerate(bucketed.to_delayed())
        ]
        with dask.config.set({"temporary-directory": shuffle_directory}):
            bucket_results = dask.compute(
                *finalizers,
                scheduler="threads",
                num_workers=config.dask_worker_count,
            )

    return _reconcile_level_results(bucket_results, expected_point_count=validated.row_count)


def _annotated_meta() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "tile_x": pd.Series(dtype="uint32"),
            "tile_y": pd.Series(dtype="uint32"),
            "x_rel": pd.Series(dtype="float32"),
            "y_rel": pd.Series(dtype="float32"),
            "value_id": pd.Series(dtype="uint32"),
            "point_id": pd.Series(dtype="uint64"),
            "bucket_id": pd.Series(dtype="uint64"),
        }
    )


def _build_annotated_source(
    validated: ValidatedPointsSource,
    *,
    exact: _LevelBuildPlan,
    x_origin: float,
    y_origin: float,
    bucket_count: int,
    value_labels_by_id: tuple[str, ...],
) -> dd.DataFrame:
    columns = validated.source.columns
    annotated_files: list[dd.DataFrame] = []
    for source_file in validated.files:
        path = validated.source.parquet_path / source_file.relative_path
        # Deliberately create one Dask input partition per validated physical
        # file so its row positions align with the validated point-ID offset.
        source_partition = dd.read_parquet(
            path,
            columns=[columns.x, columns.y, columns.value],
            split_row_groups=False,
        )
        if source_partition.npartitions != 1:
            raise ValueError(f"Source file `{source_file.relative_path}` did not produce exactly one Dask partition.")
        annotated_files.append(
            source_partition.map_partitions(
                _annotate_source_partition,
                source_file=source_file,
                x_column=columns.x,
                y_column=columns.y,
                value_column=columns.value,
                x_origin=x_origin,
                y_origin=y_origin,
                tile_size=exact.tile_size,
                grid_width=exact.grid_width,
                grid_height=exact.grid_height,
                bucket_count=bucket_count,
                value_labels_by_id=value_labels_by_id,
                meta=_annotated_meta(),
            )
        )
    return dd.concat(annotated_files)


def _annotate_source_partition(
    partition: pd.DataFrame,
    *,
    source_file: ParquetSourceFile,
    x_column: str,
    y_column: str,
    value_column: str,
    x_origin: float,
    y_origin: float,
    tile_size: int,
    grid_width: int,
    grid_height: int,
    bucket_count: int,
    value_labels_by_id: tuple[str, ...],
) -> pd.DataFrame:
    """Convert one file-aligned source partition into cache-routing columns.

    For example, with origin ``(0, 0)``, tile size ``512``, source-file row
    offset ``0``, and 69 output buckets, this conceptual input::

        x       y       value
        12.5    30.0    GAPDH
        520.0   42.0    ACTB

    can become::

        tile_x  tile_y  x_rel  y_rel  value_id  point_id  bucket_id
        0       0       12.5   30.0   127       0         16
        1       0       8.0    42.0   42        1         26

    Each normalized value receives the canonical ID equal to its position in
    ``value_labels_by_id``. Bucket IDs are determined by ``bucket_count`` and
    the deterministic tile hash.
    """
    row_count = len(partition)
    if row_count != source_file.row_count:
        raise ValueError(
            f"Decoded source file `{source_file.relative_path}` has {row_count} rows; "
            f"validation recorded {source_file.row_count}."
        )
    if row_count == 0:
        return _annotated_meta()

    x = partition[x_column].to_numpy(dtype=np.float64, na_value=np.nan)
    y = partition[y_column].to_numpy(dtype=np.float64, na_value=np.nan)
    if not bool(np.isfinite(x).all()) or not bool(np.isfinite(y).all()):
        raise ValueError(f"Source file `{source_file.relative_path}` contains invalid coordinates during construction.")

    tile_x_signed = np.floor((x - x_origin) / tile_size).astype(np.int64)
    tile_y_signed = np.floor((y - y_origin) / tile_size).astype(np.int64)
    if (
        bool((tile_x_signed < 0).any())
        or bool((tile_x_signed >= grid_width).any())
        or bool((tile_y_signed < 0).any())
        or bool((tile_y_signed >= grid_height).any())
    ):
        raise ValueError(f"Source file `{source_file.relative_path}` contains coordinates outside the Exact grid.")

    tile_x = tile_x_signed.astype(np.uint32)
    tile_y = tile_y_signed.astype(np.uint32)
    x_rel = (x - (x_origin + tile_x_signed * tile_size)).astype(np.float32)
    y_rel = (y - (y_origin + tile_y_signed * tile_size)).astype(np.float32)
    value_ids = _map_partition_value_ids(partition[value_column], value_labels_by_id, source_file.relative_path)
    point_ids = np.arange(row_count, dtype=np.uint64) + np.uint64(source_file.row_offset)
    bucket_ids = _tile_bucket_ids(tile_x, tile_y, bucket_count=bucket_count)

    return pd.DataFrame(
        {
            "tile_x": tile_x,
            "tile_y": tile_y,
            "x_rel": x_rel,
            "y_rel": y_rel,
            "value_id": value_ids,
            "point_id": point_ids,
            "bucket_id": bucket_ids,
        }
    )


def _map_partition_value_ids(
    values: pd.Series,
    value_labels_by_id: tuple[str, ...],
    relative_path: str,
) -> np.ndarray:
    arrow_values = pa.array(values, from_pandas=True)
    normalized = _normalized_row_values(arrow_values)
    if normalized.null_count:
        raise ValueError(f"Source file `{relative_path}` contains null normalized values during construction.")
    empty_count = int(pc.sum(pc.cast(pc.equal(normalized, ""), pa.int64())).as_py() or 0)
    if empty_count:
        raise ValueError(f"Source file `{relative_path}` contains empty normalized values during construction.")

    # `index_in` returns each label's position in `value_labels_by_id`.
    # `ValidatedPointsSource.value_table` guarantees that this position is the
    # label's canonical uint32 value ID.
    indices = pc.index_in(normalized, value_set=pa.array(value_labels_by_id, type=pa.string()))
    if indices.null_count:
        raise ValueError(f"Source file `{relative_path}` contains a normalized value absent from the validated table.")
    return pc.cast(indices, pa.uint32()).to_numpy(zero_copy_only=False)


def _finalize_bucket(
    partition: pd.DataFrame,
    *,
    bucket_id: int,
    filename_width: int,
    level: int,
    level_directory: Path,
    intermediate_count_directory: Path,
    staging_directory: Path,
    max_rows_per_row_group: int,
) -> _BucketWriteResult:
    if partition.empty:
        return _BucketWriteResult(
            bucket_id=bucket_id,
            point_count=0,
            value_count_total=0,
            manifest_rows=(),
            intermediate_value_count_file=None,
        )
    observed_bucket_ids = partition["bucket_id"].to_numpy(dtype=np.uint64, copy=False)
    if not bool((observed_bucket_ids == np.uint64(bucket_id)).all()):
        raise ValueError(f"Dask output partition {bucket_id} contains rows assigned to another bucket.")

    ordered = partition.reset_index(drop=True).sort_values(
        ["tile_y", "tile_x", "point_id"],
        kind="mergesort",
        ignore_index=True,
    )
    filename = f"bucket-{bucket_id:0{filename_width}d}.parquet"
    point_path = level_directory / filename
    intermediate_count_path = intermediate_count_directory / filename
    if point_path.exists() or intermediate_count_path.exists():
        raise FileExistsError(f"Bucket output already exists for bucket {bucket_id}.")

    relative_point_path = point_path.relative_to(staging_directory).as_posix()
    relative_intermediate_count_path = intermediate_count_path.relative_to(staging_directory).as_posix()
    manifest_rows: list[_ManifestRow] = []
    physical_row_group = 0
    intermediate_count_writer = _IntermediateTileValueCountWriter(intermediate_count_path, level=level)
    try:
        with pq.ParquetWriter(
            point_path,
            _EXACT_PAYLOAD_SCHEMA,
            compression="snappy",
            use_dictionary=["value_id"],
        ) as point_writer:
            for (tile_y, tile_x), tile_rows in ordered.groupby(
                ["tile_y", "tile_x"],
                sort=False,
                observed=True,
            ):
                tile_x_int = int(tile_x)
                tile_y_int = int(tile_y)
                x_rel = tile_rows["x_rel"].to_numpy(dtype=np.float32, copy=False)
                y_rel = tile_rows["y_rel"].to_numpy(dtype=np.float32, copy=False)
                value_ids = tile_rows["value_id"].to_numpy(dtype=np.uint32, copy=False)
                point_ids = tile_rows["point_id"].to_numpy(dtype=np.uint64, copy=False)
                # Count values over the complete logical tile before physical
                # row-group sharding. This emits each nonzero tile/value key
                # once even when an oversized tile spans several row groups.
                unique_value_ids, value_counts = np.unique(value_ids, return_counts=True)
                intermediate_count_writer.append(
                    tile_x=tile_x_int,
                    tile_y=tile_y_int,
                    value_ids=unique_value_ids,
                    counts=value_counts,
                )

                # Store one logical tile in one Parquet row group when it fits.
                # Split only an oversized tile into consecutive row-group shards;
                # a row group never contains points from different tiles.
                for tile_shard, start in enumerate(range(0, len(tile_rows), max_rows_per_row_group)):
                    stop = min(start + max_rows_per_row_group, len(tile_rows))
                    table = pa.Table.from_arrays(
                        [
                            pa.array(x_rel[start:stop], type=pa.float32()),
                            pa.array(y_rel[start:stop], type=pa.float32()),
                            pa.array(value_ids[start:stop], type=pa.uint32()),
                            pa.array(point_ids[start:stop], type=pa.uint64()),
                        ],
                        schema=_EXACT_PAYLOAD_SCHEMA,
                    )
                    point_writer.write_table(table, row_group_size=table.num_rows)
                    manifest_rows.append(
                        _ManifestRow(
                            level=level,
                            level_file=relative_point_path,
                            tile_x=tile_x_int,
                            tile_y=tile_y_int,
                            n_points=table.num_rows,
                            row_group=physical_row_group,
                            tile_shard=tile_shard,
                        )
                    )
                    physical_row_group += 1
    finally:
        intermediate_count_writer.close()

    intermediate_count_file = _IntermediateTileValueCountFile(
        level=level,
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
        point_count=len(ordered),
        value_count_total=intermediate_count_writer.point_count,
        manifest_rows=tuple(manifest_rows),
        intermediate_value_count_file=intermediate_count_file,
    )


def _validate_bucket_files(
    point_path: Path,
    intermediate_count_path: Path,
    *,
    manifest_rows: list[_ManifestRow],
    intermediate_count_file: _IntermediateTileValueCountFile,
) -> None:
    point_file = pq.ParquetFile(point_path)
    if not point_file.schema_arrow.equals(_EXACT_PAYLOAD_SCHEMA, check_metadata=False):
        raise ValueError(f"Exact bucket `{point_path}` has an incompatible payload schema.")
    if point_file.num_row_groups != len(manifest_rows):
        raise ValueError(f"Exact bucket `{point_path}` row-group count does not match its manifest rows.")
    for row_group_index, manifest_row in enumerate(manifest_rows):
        if point_file.metadata.row_group(row_group_index).num_rows != manifest_row.n_points:
            raise ValueError(f"Exact bucket `{point_path}` row-group rows do not match its manifest row.")

    intermediate_count_parquet = pq.ParquetFile(intermediate_count_path)
    if not intermediate_count_parquet.schema_arrow.equals(_TILE_VALUE_COUNT_SCHEMA, check_metadata=False):
        raise ValueError(f"Intermediate tile/value-count file `{intermediate_count_path}` has an incompatible schema.")
    if intermediate_count_parquet.metadata.num_rows != intermediate_count_file.row_count:
        raise ValueError(
            f"Intermediate tile/value-count file `{intermediate_count_path}` does not match its descriptor."
        )


def _reconcile_level_results(
    bucket_results: tuple[_BucketWriteResult, ...],
    *,
    expected_point_count: int,
) -> _LevelWriteResult:
    ordered_results = tuple(sorted(bucket_results, key=lambda result: result.bucket_id))
    if sum(result.point_count for result in ordered_results) != expected_point_count:
        raise ValueError("Exact bucket rows do not reconcile to the validated source row count.")
    if sum(result.value_count_total for result in ordered_results) != expected_point_count:
        raise ValueError("Exact tile/value counts do not reconcile to the validated source row count.")

    manifest_rows = tuple(
        sorted(
            (row for result in ordered_results for row in result.manifest_rows),
            key=lambda row: (row.level, row.tile_y, row.tile_x, row.tile_shard),
        )
    )
    physical_keys = {(row.level_file, row.row_group) for row in manifest_rows}
    if len(physical_keys) != len(manifest_rows):
        raise ValueError("Exact manifest contains duplicate physical row-group keys.")

    intermediate_files = tuple(
        result.intermediate_value_count_file
        for result in ordered_results
        if result.intermediate_value_count_file is not None
    )
    if len({file.relative_path for file in intermediate_files}) != len(intermediate_files):
        raise ValueError("Exact level contains duplicate intermediate tile/value-count file paths.")
    return _LevelWriteResult(
        manifest_rows=manifest_rows,
        intermediate_tile_value_count_files=intermediate_files,
    )
