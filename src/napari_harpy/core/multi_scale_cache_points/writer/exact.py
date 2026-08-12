from __future__ import annotations

import tempfile
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
from napari_harpy.core.multi_scale_cache_points.writer.models import (
    _BucketWriteResult,
    _ExactLevelWriterConfig,
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

DEFAULT_DASK_WORKER_COUNT = 1


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
            _POINT_PAYLOAD_SCHEMA,
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
                        schema=_POINT_PAYLOAD_SCHEMA,
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
