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
from napari_harpy.core.multi_scale_cache_points_zarr.source.models import ValidatedPointsSource
from napari_harpy.core.multi_scale_cache_points_zarr.source.signature import POINT_ID_POLICY
from napari_harpy.core.multi_scale_cache_points_zarr.source.value_normalization import (
    VALUE_NORMALIZATION_METHOD,
    _normalized_row_values,
)
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
    """Carry one Exact finalizer result across the Dask task boundary.

    Parameters
    ----------
    bucket_result
        Generic physical Zarr result shared by all cache levels, or ``None``
        when this planned destination received no points and created no bucket.
    value_id
        Strictly increasing ``uint32`` IDs present in this Exact destination.
        These are Exact-specific auxiliary data rather than part of the generic
        physical bucket result.
    value_count
        Positive ``uint64`` point counts aligned one-to-one with ``value_id``.
        Level reconciliation combines these counts across destinations and
        compares them with the validated source vocabulary totals.

    Notes
    -----
    ``bucket_result`` describes the finalized physical Zarr store. The aligned
    ``value_id`` and ``value_count`` arrays additionally let the Exact writer
    prove that construction preserved every validated source value without
    adding level-specific fields to the shared ``_BucketWriteResult`` contract.
    """

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

    Each point belongs to one logical spatial tile. The deterministic tile hash
    assigns that complete tile to one bucket, even when the tile's points occur
    in several source files or row groups. Different logical tiles may share a
    bucket, but one logical tile is never split across buckets.

    Construction follows this physical flow::

        row-group-aligned source partitions
            -> read only the selected x, y, and value columns
            -> annotate tile coordinates, tile-relative coordinates,
               canonical value IDs, and internal point IDs
            -> hash each logical (tile_x, tile_y) to one bucket_id
            -> disk-redistribute points into partitions by bucket_id
            -> stable-sort each complete bucket by (tile_y, tile_x)
            -> derive the ordered _BucketPlan and one _PointPayload per tile
            -> order each tile payload by (value_id, point_id)
            -> append aligned point arrays and sparse value ranges
            -> finalize one independent Zarr store per nonempty bucket

    The disk shuffle is what co-locates tile rows that arrived through different
    row-group partitions. Sorting the complete destination bucket then makes
    each tile a contiguous run. The Exact finalizer owns that bucket-level tile
    order; the shared bucket writer owns deterministic value-major ordering and
    sparse-range construction inside each tile.

    Every side-effecting finalizer exclusively owns one Zarr store, so concurrent
    tasks never write the same bucket. Dask scratch is disposable and separate
    from the caller-owned staging generation.

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
    # Materialize the validated vocabulary in ID order; each tuple position is
    # the canonical uint32 value ID defined by `ValidatedPointsSource.value_table`.
    value_labels_by_id: tuple[str, ...] = tuple(validated.value_table["value"].to_pylist())
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
    # Create one task from each explicit physical row-group specification rather
    # than deriving file/row-group identity from dd.read_parquet partition order.
    # This keeps point_id_start tied to validated source metadata and independent
    # of Dask partition planning.
    annotated = dd.from_map(
        read_partition,
        read_specs,
        meta=_annotated_meta(),
        enforce_metadata=True,
    )
    # Unit-width integer divisions create exactly one destination partition per
    # valid bucket ID: partition position i contains only bucket_id == i. This
    # makes enumerate(bucketed.to_delayed()) the canonical destination identity;
    # each finalizer still verifies that its retained routing column agrees.
    bucketed = annotated.set_index(
        "bucket_id",
        divisions=list(range(bucket_count + 1)),
        shuffle_method="disk",
        drop=False,
    )

    level_directory = staging_root / exact.relative_directory
    level_directory.mkdir(parents=True)
    finalizers = tuple(
        # Each finalizer owns one side-effecting Zarr bucket write. Mark it
        # impure so Dask does not reuse or deduplicate a call that must create
        # its own store.
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
        table,
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
    table: pa.Table,
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
    """Annotate one physical source row group for Exact cache routing.

    Parameters
    ----------
    table
        Decoded Arrow row-group table containing exactly the selected
        x-coordinate, y-coordinate, and value columns. Source columns remain in
        Arrow through annotation; only the final numeric result is materialized
        as a Pandas partition for Dask.
    expected_row_count
        Row count recorded for this physical row group during source
        validation. Annotation fails if the decoded partition disagrees.
    point_id_start
        First Harpy-internal cache point ID assigned to this row group. It is
        derived from canonical physical source-row order and is not read from
        a Parquet column.
    x_column
        Physical source column containing x coordinates.
    y_column
        Physical source column containing y coordinates.
    value_column
        Physical source column containing values that map to canonical
        ``value_id`` values.
    x_origin
        Shared x-coordinate origin from which Exact ``tile_x`` is calculated.
    y_origin
        Shared y-coordinate origin from which Exact ``tile_y`` is calculated.
    tile_size
        Exact logical tile edge in intrinsic source-coordinate units.
    grid_width
        Number of valid Exact tile columns.
    grid_height
        Number of valid Exact tile rows.
    bucket_count
        Number of deterministic tile-hash destinations.
    value_labels_by_id
        Canonical normalized vocabulary in ``value_id`` order. A label's tuple
        position is its serialized ``uint32`` ID.
    validated_row_count
        Complete validated source row count, used to bound the internal point-ID
        interval assigned to this row group.
    source_label
        Human-readable file and row-group identity used in validation errors.

    Returns
    -------
    pandas.DataFrame
        Row-aligned Exact annotations with the fixed ``tile_x``, ``tile_y``,
        ``x_rel``, ``y_rel``, ``value_id``, ``point_id``, and ``bucket_id``
        schema.

    Notes
    -----
    For example, with origin ``(0, 0)``, tile size ``512``,
    ``point_id_start=100``, and 69 buckets, assume ``ACTB`` and ``GAPDH`` have
    canonical value IDs 42 and 127. This conceptual row-group input::

        x       y       value
        12.5    30.0    GAPDH
        520.0   42.0    ACTB

    becomes::

        tile_x  tile_y  x_rel  y_rel  value_id  point_id  bucket_id
        0       0       12.5   30.0   127       100       16
        1       0       8.0    42.0   42        101       26

    Coordinates are assigned to logical tiles and stored relative to their tile
    origins. Values map to their positions in ``value_labels_by_id``. Point IDs
    are synthesized consecutively from ``point_id_start``. Bucket IDs come from
    the versioned deterministic hash of ``(tile_x, tile_y)``.
    """
    row_count = table.num_rows
    if row_count != expected_row_count:
        raise ValueError(f"Decoded source row group `{source_label}` does not match its validated row count.")
    if row_count == 0:
        return _annotated_meta()
    if table.column_names != [x_column, y_column, value_column]:
        raise ValueError(f"Decoded source row group `{source_label}` has unexpected columns or column order.")
    if point_id_start < 0 or point_id_start + row_count > validated_row_count:
        raise ValueError(f"Source row group `{source_label}` has an invalid point-ID interval.")

    x = np.ascontiguousarray(
        table[x_column].combine_chunks().to_numpy(zero_copy_only=False),
        dtype=np.float64,
    )
    y = np.ascontiguousarray(
        table[y_column].combine_chunks().to_numpy(zero_copy_only=False),
        dtype=np.float64,
    )
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
        table[value_column].combine_chunks(),
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
    values: pa.Array,
    *,
    value_labels_by_id: tuple[str, ...],
    source_label: str,
) -> np.ndarray:
    """Map one Arrow value array to canonical row-aligned ``uint32`` IDs.

    Dictionary inputs remain encoded while their distinct labels are normalized
    and mapped to the validated vocabulary. Taking those dictionary-level IDs by
    the original row indices avoids materializing a row-aligned string array.
    Plain UTF-8 inputs use the same normalization and vocabulary contract at row
    level. In either representation, any row whose normalized value is null,
    empty, or absent from ``value_labels_by_id`` fails construction.
    """
    value_set = pa.array(value_labels_by_id, type=pa.string())
    if pa.types.is_dictionary(values.type):
        if values.indices.null_count:
            raise ValueError(f"Source row group `{source_label}` contains null normalized values.")
        normalized_dictionary = _normalized_row_values(values.dictionary)
        used_indices = pc.unique(values.indices)
        used_values = pc.take(normalized_dictionary, used_indices)
        if used_values.null_count:
            raise ValueError(f"Source row group `{source_label}` contains null normalized values.")
        if bool(pc.any(pc.equal(used_values, "")).as_py()):
            raise ValueError(f"Source row group `{source_label}` contains empty normalized values.")
        dictionary_ids = pc.index_in(normalized_dictionary, value_set=value_set)
        indices = pc.take(dictionary_ids, values.indices)
    else:
        normalized = _normalized_row_values(values)
        if normalized.null_count:
            raise ValueError(f"Source row group `{source_label}` contains null normalized values.")
        if bool(pc.any(pc.equal(normalized, "")).as_py()):
            raise ValueError(f"Source row group `{source_label}` contains empty normalized values.")
        indices = pc.index_in(normalized, value_set=value_set)

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
    """Plan and write one complete nonempty shuffled destination bucket.

    Parameters
    ----------
    partition
        Complete materialized Dask destination partition. Row arrival order is
        arbitrary, but every row must belong to ``bucket_id``.
    bucket_id
        Deterministic destination identity assigned by the tile hash.
    staging_root
        Cache-generation root beneath which the bucket writer creates its
        canonical Zarr store.
    tile_size
        Exact logical tile edge, used to enforce relative-coordinate upper
        bounds before storage.
    grid_width
        Number of valid Exact tile columns.
    grid_height
        Number of valid Exact tile rows.
    validated_row_count
        Complete validated source count, used to bound internal point IDs.
    settings
        Physical Zarr chunk, shard, and codec settings for the bucket.

    Returns
    -------
    _ExactBucketOutcome
        Finalized nonempty bucket result and its sparse per-value totals, or an
        empty outcome when the planned destination received no rows.

    Notes
    -----
    Ordering ownership is intentionally split across construction layers. This
    finalizer stable-sorts the complete bucket only by ``(tile_y, tile_x)`` so
    every logical tile is contiguous and tiles follow `_BucketPlan` order. It
    then supplies one tile payload at a time to `_BucketWriter`. The bucket
    writer independently canonicalizes the rows inside each tile by
    ``(value_id, point_id)`` while constructing sparse value ranges. Together
    these responsibilities produce the final physical order::

        tile_y -> tile_x -> value_id -> point_id

    Do not add a bucket-wide value/point sort here: it would duplicate the
    shared writer's per-tile work and obscure which layer owns sparse-range
    ordering.
    """
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

    # This finalizer owns tile contiguity and row-major tile order only. The
    # shared bucket writer later owns canonical (value_id, point_id) ordering
    # within each resulting tile payload.
    ordered = frame.sort_values(
        ["tile_y", "tile_x"],
        kind="mergesort",
        ignore_index=True,
    )
    tile_x = np.ascontiguousarray(ordered["tile_x"].to_numpy(dtype=np.uint32, copy=False))
    tile_y = np.ascontiguousarray(ordered["tile_y"].to_numpy(dtype=np.uint32, copy=False))
    if int(tile_x.max()) >= grid_width or int(tile_y.max()) >= grid_height:
        raise ValueError(f"Dask output partition {bucket_id} contains tiles outside the Exact grid.")
    # Rows are ordered by (tile_y, tile_x), so a change in either adjacent
    # coordinate marks the start of a new contiguous logical-tile run. Prepend
    # True for the first run; these starts are reused both to build the complete
    # bucket plan and to slice one payload per tile.
    tile_starts = np.flatnonzero(
        np.concatenate(
            (
                np.array([True], dtype=np.bool_),
                (tile_x[1:] != tile_x[:-1]) | (tile_y[1:] != tile_y[:-1]),
            )
        )
    )
    tile_stops = np.concatenate((tile_starts[1:], np.array([len(ordered)], dtype=np.int64)))
    tiles = tuple(
        _PlannedTile(
            tile_x=int(tile_x[start]),
            tile_y=int(tile_y[start]),
            n_points=int(stop - start),
        )
        for start, stop in zip(tile_starts, tile_stops, strict=True)
    )
    plan = _BucketPlan(level=0, bucket_id=bucket_id, tiles=tiles, settings=settings)
    with _BucketWriter(staging_root, plan) as writer:
        for start, stop, tile in zip(tile_starts, tile_stops, tiles, strict=True):
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
