from __future__ import annotations

import tempfile
from dataclasses import dataclass
from functools import partial
from pathlib import Path, PurePosixPath

import dask
import dask.dataframe as dd
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from napari_harpy.core.multi_scale_cache_points.models import ValidatedPointsSource
from napari_harpy.core.multi_scale_cache_points.signature import POINT_ID_POLICY
from napari_harpy.core.multi_scale_cache_points.value_normalization import (
    VALUE_NORMALIZATION_METHOD,
    _normalized_row_values,
)
from napari_harpy.core.multi_scale_cache_points_zarr.build_plan import (
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
)
from napari_harpy.core.multi_scale_cache_points_zarr.payload import _PointPayload
from napari_harpy.core.multi_scale_cache_points_zarr.storage.bucket_writer import _BucketWriter
from napari_harpy.core.multi_scale_cache_points_zarr.storage.models import (
    _BucketPlan,
    _BucketWriteResult,
    _LevelWriteResult,
    _PlannedTile,
    _ZarrWriteSettings,
)


@dataclass(frozen=True)
class _ExactWriterConfig:
    """Configure Exact execution separately from its logical build plan.

    Parameters
    ----------
    zarr_settings
        Physical chunk, shard, and codec settings shared by every Exact bucket.
    dask_worker_count
        Positive number of local threaded-scheduler workers. This also bounds
        concurrently materialized shuffled buckets and active bucket writers.
    """

    zarr_settings: _ZarrWriteSettings
    dask_worker_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.zarr_settings, _ZarrWriteSettings):
            raise ValueError("`zarr_settings` must be _ZarrWriteSettings.")
        _require_integer_in_range(
            self.dask_worker_count,
            "dask_worker_count",
            minimum=1,
            maximum=_INT64_MAX,
        )


@dataclass(frozen=True)
class _SourceRowGroupReadSpec:
    """Describe one independently scheduled physical Parquet row-group read.

    Parameters
    ----------
    relative_path
        Normalized POSIX path of the source Parquet file relative to the
        validated dataset root.
    row_group_index
        Zero-based physical row-group index within the source file.
    expected_row_count
        Row count recorded during source validation. The read task compares
        this with the decoded row count and fails immediately if the physical
        source no longer matches the validated inventory.
    point_id_start
        First canonical internal cache point ID assigned to this row group.
        Harpy synthesizes this value from the source file's global row offset
        and the rows in preceding row groups within that file. It is not read
        from a Parquet column or Parquet metadata field.

    Notes
    -----
    This record contains no point data. It is a self-contained description of
    one Dask input task. Carrying the physical row-group identity and global
    point-ID start explicitly makes point identity independent of Dask
    partition and execution order. Parquet supplies physical row counts and
    ordering; Harpy assigns the resulting internal IDs.
    """

    relative_path: str
    row_group_index: int
    expected_row_count: int
    point_id_start: int

    def __post_init__(self) -> None:
        if not isinstance(self.relative_path, str) or self.relative_path == "":
            raise ValueError("`relative_path` must be a nonempty normalized relative POSIX path.")
        path = PurePosixPath(self.relative_path)
        if path.is_absolute() or ".." in path.parts or path.as_posix() != self.relative_path:
            raise ValueError("`relative_path` must be a nonempty normalized relative POSIX path.")
        _require_integer_in_range(self.row_group_index, "row_group_index", maximum=_INT64_MAX)
        _require_integer_in_range(self.expected_row_count, "expected_row_count", maximum=_INT64_MAX)
        _require_integer_in_range(self.point_id_start, "point_id_start", maximum=_INT64_MAX)
        if self.point_id_start > _INT64_MAX - self.expected_row_count:
            raise ValueError("Row-group point-ID interval exceeds the supported int64 range.")


@dataclass(frozen=True, eq=False)
class _ExactBucketOutcome:
    """Return one destination's bucket result and sparse value totals."""

    bucket_result: _BucketWriteResult | None
    value_id: np.ndarray
    value_count: np.ndarray

    def __post_init__(self) -> None:
        arrays = (
            ("value_id", self.value_id, np.dtype(np.uint32)),
            ("value_count", self.value_count, np.dtype(np.uint64)),
        )
        for name, array, dtype in arrays:
            if not isinstance(array, np.ndarray) or array.ndim != 1 or array.dtype != dtype:
                raise ValueError(f"`{name}` must be a one-dimensional {dtype.name} NumPy array.")
            if not array.flags.c_contiguous:
                raise ValueError(f"`{name}` must be C-contiguous.")
        if len(self.value_id) != len(self.value_count):
            raise ValueError("Exact bucket value IDs and counts must have equal lengths.")
        if self.bucket_result is None:
            if len(self.value_id):
                raise ValueError("An empty Exact bucket outcome must not contain value totals.")
        else:
            if not isinstance(self.bucket_result, _BucketWriteResult):
                raise ValueError("`bucket_result` must be _BucketWriteResult or None.")
            if len(self.value_id) == 0 or int(self.value_count.sum(dtype=np.uint64)) != self.bucket_result.point_count:
                raise ValueError("Exact bucket value totals must match its physical point count.")
            if bool((self.value_count == 0).any()) or bool((self.value_id[1:] <= self.value_id[:-1]).any()):
                raise ValueError("Exact bucket value IDs must be strictly increasing with positive counts.")

        # Outcomes cross the Dask task boundary. Install read-only views so the
        # reconciliation inputs cannot be mutated after finalization.
        for name, array, _ in arrays:
            read_only = array.view()
            read_only.flags.writeable = False
            object.__setattr__(self, name, read_only)


def _write_exact_level(
    validated: ValidatedPointsSource,
    plan: _PointsCacheBuildPlan,
    *,
    staging_root: Path,
    temporary_directory_root: Path,
    config: _ExactWriterConfig,
) -> _LevelWriteResult:
    """Construct uncapped Exact level zero directly from source Parquet to Zarr.

    Physical source row groups become explicit Dask input partitions. A disk
    shuffle co-locates complete logical tiles by deterministic bucket ID, after
    which each side-effecting finalizer exclusively owns one Zarr store. Dask
    scratch is disposable and separate from the caller-owned staging generation.

    Parameters
    ----------
    validated
        Canonical content-validated Parquet points source.
    plan
        Complete logical Zarr-cache plan whose first level is Exact level zero.
    staging_root
        Existing isolated generation root. This function owns creation of its
        previously absent ``levels/level_0`` directory.
    temporary_directory_root
        Existing caller-owned root for one disposable Dask scratch child.
    config
        Exact execution concurrency and physical Zarr settings.

    Returns
    -------
    _LevelWriteResult
        Nonempty finalized Exact buckets ordered by numeric bucket ID.
    """
    exact = plan.levels[0]
    if validated.value_normalization_method != VALUE_NORMALIZATION_METHOD:
        raise ValueError("The validated source uses an unsupported value-normalization method.")
    if validated.point_id_policy != POINT_ID_POLICY:
        raise ValueError("The validated source uses an unsupported point-ID policy.")
    if exact.level != 0 or exact.kind is not _LevelKind.EXACT or exact.max_points_per_tile is not None:
        raise ValueError("The first build-plan level must be uncapped Exact level zero.")
    if exact.point_count_upper_bound != validated.row_count:
        raise ValueError("Exact's planned point count must equal the validated source count.")

    if not staging_root.is_dir():
        raise ValueError("`staging_root` must be an existing pathlib.Path directory.")
    if not temporary_directory_root.is_dir():
        raise ValueError("`temporary_directory_root` must be an existing pathlib.Path directory.")
    staging_resolved = staging_root.resolve()
    temporary_resolved = temporary_directory_root.resolve()
    if (
        staging_resolved == temporary_resolved
        or staging_resolved in temporary_resolved.parents
        or temporary_resolved in staging_resolved.parents
    ):
        raise ValueError("Staging output and Dask temporary roots must be separate directory trees.")
    if (staging_root / exact.relative_directory).exists():
        raise FileExistsError(f"Exact-level output path already exists: {exact.relative_directory}.")

    read_specs = _source_row_group_read_specs(validated)
    if sum(spec.expected_row_count for spec in read_specs) != validated.row_count:
        raise ValueError("Row-group read specifications do not reconcile to the validated source count.")

    bucket_count = _bucket_count_for_level(exact)
    value_labels_by_id = tuple(validated.value_table["value"].to_pylist())
    columns = validated.source.columns
    read_partition = partial(
        _read_and_annotate_row_group,
        source_root=validated.source.parquet_path,
        x_column=columns.x,
        y_column=columns.y,
        value_column=columns.value,
        x_origin=plan.x_origin,
        y_origin=plan.y_origin,
        tile_size=exact.tile_size,
        grid_width=exact.grid_width,
        grid_height=exact.grid_height,
        bucket_count=bucket_count,
        value_labels_by_id=value_labels_by_id,
        validated_row_count=validated.row_count,
    )
    annotated = dd.from_map(
        read_partition,
        read_specs,
        meta=_annotated_meta(),
        enforce_metadata=True,
    )
    bucketed = annotated.set_index(
        "bucket_id",
        divisions=list(range(bucket_count + 1)),
        shuffle_method="disk",
        drop=False,
    )

    level_directory = staging_root / exact.relative_directory
    level_directory.mkdir(parents=True)
    finalizers = tuple(
        dask.delayed(_finalize_exact_bucket, pure=False)(
            partition,
            bucket_id=bucket_id,
            staging_root=staging_root,
            tile_size=exact.tile_size,
            grid_width=exact.grid_width,
            grid_height=exact.grid_height,
            validated_row_count=validated.row_count,
            settings=config.zarr_settings,
        )
        for bucket_id, partition in enumerate(bucketed.to_delayed())
    )
    with tempfile.TemporaryDirectory(
        prefix="napari-harpy-zarr-exact-shuffle-",
        dir=temporary_directory_root,
    ) as shuffle_directory:
        with dask.config.set({"temporary-directory": shuffle_directory}):
            outcomes = dask.compute(
                *finalizers,
                scheduler="threads",
                num_workers=config.dask_worker_count,
            )

    return _reconcile_exact_outcomes(
        outcomes,
        validated=validated,
        exact_grid_width=exact.grid_width,
        exact_grid_height=exact.grid_height,
        expected_point_count=exact.point_count_upper_bound,
    )


def _source_row_group_read_specs(validated: ValidatedPointsSource) -> tuple[_SourceRowGroupReadSpec, ...]:
    """Return canonical row-group reads with explicit global point-ID starts."""
    specs: list[_SourceRowGroupReadSpec] = []
    expected_file_offset = 0
    for source_file in validated.files:
        if source_file.row_offset != expected_file_offset:
            raise ValueError("Validated source-file row offsets are not contiguous.")
        row_group_offset_within_file = 0
        for row_group_index, row_group in enumerate(source_file.row_groups):
            specs.append(
                _SourceRowGroupReadSpec(
                    relative_path=source_file.relative_path,
                    row_group_index=row_group_index,
                    expected_row_count=row_group.row_count,
                    point_id_start=source_file.row_offset + row_group_offset_within_file,
                )
            )
            row_group_offset_within_file += row_group.row_count
        if row_group_offset_within_file != source_file.row_count:
            raise ValueError("Validated row-group counts do not match their source file.")
        expected_file_offset += source_file.row_count
    if expected_file_offset != validated.row_count or not specs:
        raise ValueError("Validated source-file rows do not match the source total.")
    return tuple(specs)


def _annotated_meta() -> pd.DataFrame:
    """Return the exact empty Pandas schema expected across the Dask graph."""
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


def _read_and_annotate_row_group(
    spec: _SourceRowGroupReadSpec,
    *,
    source_root: Path,
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
    validated_row_count: int,
) -> pd.DataFrame:
    """Read one explicit physical row group and annotate its cache routing."""
    if not isinstance(spec, _SourceRowGroupReadSpec):
        raise ValueError("`spec` must be _SourceRowGroupReadSpec.")
    path = source_root / spec.relative_path
    parquet_file = pq.ParquetFile(path)
    try:
        table = parquet_file.read_row_group(
            spec.row_group_index,
            columns=[x_column, y_column, value_column],
        )
    finally:
        parquet_file.close()
    if table.num_rows != spec.expected_row_count:
        raise ValueError(
            f"Decoded source row group {spec.relative_path}:{spec.row_group_index} has "
            f"{table.num_rows} rows; validation recorded {spec.expected_row_count}."
        )
    return _annotate_source_partition(
        table.to_pandas(),
        expected_row_count=spec.expected_row_count,
        point_id_start=spec.point_id_start,
        x_column=x_column,
        y_column=y_column,
        value_column=value_column,
        x_origin=x_origin,
        y_origin=y_origin,
        tile_size=tile_size,
        grid_width=grid_width,
        grid_height=grid_height,
        bucket_count=bucket_count,
        value_labels_by_id=value_labels_by_id,
        validated_row_count=validated_row_count,
        source_label=f"{spec.relative_path}:{spec.row_group_index}",
    )


def _annotate_source_partition(
    partition: pd.DataFrame,
    *,
    expected_row_count: int,
    point_id_start: int,
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
    validated_row_count: int,
    source_label: str,
) -> pd.DataFrame:
    """Convert one row-group DataFrame into the exact annotated Dask schema."""
    row_count = len(partition)
    if row_count != expected_row_count:
        raise ValueError(f"Decoded source row group `{source_label}` does not match its validated row count.")
    if row_count == 0:
        return _annotated_meta()
    if list(partition.columns) != [x_column, y_column, value_column]:
        raise ValueError(f"Decoded source row group `{source_label}` has unexpected columns or column order.")
    if point_id_start < 0 or point_id_start + row_count > validated_row_count:
        raise ValueError(f"Source row group `{source_label}` has an invalid point-ID interval.")

    x = np.ascontiguousarray(partition[x_column].to_numpy(dtype=np.float64, na_value=np.nan))
    y = np.ascontiguousarray(partition[y_column].to_numpy(dtype=np.float64, na_value=np.nan))
    if not bool(np.isfinite(x).all()) or not bool(np.isfinite(y).all()):
        raise ValueError(f"Source row group `{source_label}` contains nonfinite coordinates.")

    tile_x_float = np.floor((x - x_origin) / tile_size)
    tile_y_float = np.floor((y - y_origin) / tile_size)
    if (
        bool((tile_x_float < 0).any())
        or bool((tile_x_float >= grid_width).any())
        or bool((tile_y_float < 0).any())
        or bool((tile_y_float >= grid_height).any())
    ):
        raise ValueError(f"Source row group `{source_label}` contains coordinates outside the Exact grid.")
    tile_x_signed = tile_x_float.astype(np.int64)
    tile_y_signed = tile_y_float.astype(np.int64)
    tile_x = np.ascontiguousarray(tile_x_signed, dtype=np.uint32)
    tile_y = np.ascontiguousarray(tile_y_signed, dtype=np.uint32)

    x_rel = np.ascontiguousarray(x - (x_origin + tile_x_signed * tile_size), dtype=np.float32)
    y_rel = np.ascontiguousarray(y - (y_origin + tile_y_signed * tile_size), dtype=np.float32)
    if (
        not bool(np.isfinite(x_rel).all())
        or not bool(np.isfinite(y_rel).all())
        or bool((x_rel < 0).any())
        or bool((y_rel < 0).any())
        or bool((x_rel > tile_size).any())
        or bool((y_rel > tile_size).any())
    ):
        raise ValueError(f"Source row group `{source_label}` produced invalid tile-relative coordinates.")
    tolerance = float(np.spacing(np.float32(tile_size)))
    reconstructed_x = x_origin + tile_x_signed * tile_size + x_rel.astype(np.float64)
    reconstructed_y = y_origin + tile_y_signed * tile_size + y_rel.astype(np.float64)
    if not np.allclose(reconstructed_x, x, rtol=0.0, atol=tolerance) or not np.allclose(
        reconstructed_y,
        y,
        rtol=0.0,
        atol=tolerance,
    ):
        raise ValueError(f"Source row group `{source_label}` exceeds coordinate reconstruction tolerance.")

    value_id = _map_partition_value_ids(
        partition[value_column],
        value_labels_by_id=value_labels_by_id,
        source_label=source_label,
    )
    # The selected source columns contain coordinates and values only. Synthesize
    # Harpy's internal cache identity from canonical physical source-row order;
    # this is not a point-ID column read from Parquet.
    point_id = np.arange(
        point_id_start,
        point_id_start + row_count,
        dtype=np.uint64,
    )
    bucket_id = _tile_bucket_ids(tile_x, tile_y, bucket_count=bucket_count)
    return pd.DataFrame(
        {
            "tile_x": tile_x,
            "tile_y": tile_y,
            "x_rel": x_rel,
            "y_rel": y_rel,
            "value_id": value_id,
            "point_id": point_id,
            "bucket_id": bucket_id,
        }
    )


def _map_partition_value_ids(
    values: pd.Series,
    *,
    value_labels_by_id: tuple[str, ...],
    source_label: str,
) -> np.ndarray:
    # Supplying the canonical physical type keeps an all-null Pandas partition
    # from being inferred as Arrow's kernel-less ``null`` type. Nulls are then
    # rejected by the same explicit normalization check as mixed partitions.
    arrow_values = pa.array(values, type=pa.string(), from_pandas=True)
    normalized = _normalized_row_values(arrow_values)
    if normalized.null_count:
        raise ValueError(f"Source row group `{source_label}` contains null normalized values.")
    empty = pc.equal(normalized, "")
    if bool(pc.any(empty).as_py()):
        raise ValueError(f"Source row group `{source_label}` contains empty normalized values.")
    indices = pc.index_in(
        normalized,
        value_set=pa.array(value_labels_by_id, type=pa.string()),
    )
    if indices.null_count:
        raise ValueError(f"Source row group `{source_label}` contains a normalized value absent from validation.")
    return np.ascontiguousarray(
        pc.cast(indices, pa.uint32()).to_numpy(zero_copy_only=False),
        dtype=np.uint32,
    )


def _finalize_exact_bucket(
    partition: pd.DataFrame,
    *,
    bucket_id: int,
    staging_root: Path,
    tile_size: int,
    grid_width: int,
    grid_height: int,
    validated_row_count: int,
    settings: _ZarrWriteSettings,
) -> _ExactBucketOutcome:
    """Plan and write one complete nonempty shuffled destination bucket."""
    if partition.empty:
        return _empty_bucket_outcome()
    _require_annotated_partition(partition)
    frame = partition.reset_index(drop=True)
    observed_bucket_ids = frame["bucket_id"].to_numpy(dtype=np.uint64, copy=False)
    if not bool((observed_bucket_ids == np.uint64(bucket_id)).all()):
        raise ValueError(f"Dask output partition {bucket_id} contains rows assigned to another bucket.")

    point_ids = frame["point_id"].to_numpy(dtype=np.uint64, copy=False)
    if not frame["point_id"].is_unique or int(point_ids.max()) >= validated_row_count:
        raise ValueError(f"Dask output partition {bucket_id} contains duplicate or out-of-range point IDs.")
    value_ids, value_counts = np.unique(
        frame["value_id"].to_numpy(dtype=np.uint32, copy=False),
        return_counts=True,
    )
    value_ids = np.ascontiguousarray(value_ids, dtype=np.uint32)
    value_counts = np.ascontiguousarray(value_counts, dtype=np.uint64)

    ordered = frame.sort_values(
        ["tile_y", "tile_x"],
        kind="mergesort",
        ignore_index=True,
    )
    tile_x = np.ascontiguousarray(ordered["tile_x"].to_numpy(dtype=np.uint32, copy=False))
    tile_y = np.ascontiguousarray(ordered["tile_y"].to_numpy(dtype=np.uint32, copy=False))
    if int(tile_x.max()) >= grid_width or int(tile_y.max()) >= grid_height:
        raise ValueError(f"Dask output partition {bucket_id} contains tiles outside the Exact grid.")
    boundaries = np.flatnonzero(
        np.concatenate(
            (
                np.array([True], dtype=np.bool_),
                (tile_x[1:] != tile_x[:-1]) | (tile_y[1:] != tile_y[:-1]),
            )
        )
    )
    stops = np.concatenate((boundaries[1:], np.array([len(ordered)], dtype=np.int64)))
    tiles = tuple(
        _PlannedTile(
            tile_x=int(tile_x[start]),
            tile_y=int(tile_y[start]),
            n_points=int(stop - start),
        )
        for start, stop in zip(boundaries, stops, strict=True)
    )
    plan = _BucketPlan(level=0, bucket_id=bucket_id, tiles=tiles, settings=settings)
    with _BucketWriter(staging_root, plan) as writer:
        for start, stop, tile in zip(boundaries, stops, tiles, strict=True):
            tile_rows = ordered.iloc[int(start) : int(stop)]
            payload = _PointPayload(
                x_rel=np.ascontiguousarray(tile_rows["x_rel"].to_numpy(dtype=np.float32, copy=False)),
                y_rel=np.ascontiguousarray(tile_rows["y_rel"].to_numpy(dtype=np.float32, copy=False)),
                value_id=np.ascontiguousarray(tile_rows["value_id"].to_numpy(dtype=np.uint32, copy=False)),
                point_id=np.ascontiguousarray(tile_rows["point_id"].to_numpy(dtype=np.uint64, copy=False)),
            )
            if bool((payload.x_rel > tile_size).any()) or bool((payload.y_rel > tile_size).any()):
                raise ValueError("Exact tile-relative coordinates exceed the planned tile size.")
            writer.write_tile(tile.tile_x, tile.tile_y, payload)
        bucket_result = writer.finalize()
    return _ExactBucketOutcome(bucket_result, value_ids, value_counts)


def _require_annotated_partition(partition: pd.DataFrame) -> None:
    expected = _annotated_meta()
    if list(partition.columns) != list(expected.columns):
        raise ValueError("A shuffled Exact partition has unexpected columns or column order.")
    if any(partition[name].dtype != expected[name].dtype for name in expected.columns):
        raise ValueError("A shuffled Exact partition has incompatible column dtypes.")


def _empty_bucket_outcome() -> _ExactBucketOutcome:
    return _ExactBucketOutcome(
        None,
        np.empty(0, dtype=np.uint32),
        np.empty(0, dtype=np.uint64),
    )


def _reconcile_exact_outcomes(
    outcomes: tuple[_ExactBucketOutcome, ...],
    *,
    validated: ValidatedPointsSource,
    exact_grid_width: int,
    exact_grid_height: int,
    expected_point_count: int,
) -> _LevelWriteResult:
    """Reconcile independent bucket results with canonical Exact source facts."""
    if not isinstance(outcomes, tuple) or not outcomes:
        raise ValueError("Exact construction must produce one outcome per planned destination.")
    if not all(isinstance(outcome, _ExactBucketOutcome) for outcome in outcomes):
        raise ValueError("Exact finalizers returned an incompatible outcome.")
    buckets = tuple(
        sorted(
            (outcome.bucket_result for outcome in outcomes if outcome.bucket_result is not None),
            key=lambda result: result.bucket_id,
        )
    )
    result = _LevelWriteResult(buckets=buckets)
    if result.level != 0:
        raise ValueError("Exact construction produced a nonzero cache level.")
    if result.point_count != expected_point_count or result.point_count != validated.row_count:
        raise ValueError("Final Exact bucket rows do not reconcile to the validated source count.")
    if any(tile.tile_x >= exact_grid_width or tile.tile_y >= exact_grid_height for tile in result.tile_descriptors):
        raise ValueError("Final Exact descriptors fall outside the planned grid.")

    observed_value_counts = np.zeros(validated.value_table.num_rows, dtype=np.uint64)
    for outcome in outcomes:
        if len(outcome.value_id) == 0:
            continue
        if int(outcome.value_id[-1]) >= len(observed_value_counts):
            raise ValueError("An Exact finalizer returned an out-of-range value ID.")
        np.add.at(
            observed_value_counts,
            outcome.value_id.astype(np.intp, copy=False),
            outcome.value_count,
        )
    expected_value_counts = np.ascontiguousarray(
        validated.value_table["n_points"].combine_chunks().to_numpy(zero_copy_only=False),
        dtype=np.uint64,
    )
    if not np.array_equal(observed_value_counts, expected_value_counts):
        raise ValueError("Final Exact per-value totals do not reconcile to the validated source.")
    return result
