from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from napari_harpy.core.multi_scale_cache_points.build_plan import _LevelBuildPlan
from napari_harpy.core.multi_scale_cache_points.hashing import _splitmix64
from napari_harpy.core.multi_scale_cache_points.writer.models import (
    _BucketWriteResult,
    _IntermediateTileValueCountFile,
    _LevelWriteResult,
    _ManifestRow,
)

BUCKET_HASH_METHOD = "harpy-tile-splitmix64-v1"
TARGET_ROWS_PER_OUTPUT_BUCKET = 2_000_000
DEFAULT_MAX_ROWS_PER_ROW_GROUP = 1_000_000

_INTERMEDIATE_TILE_VALUE_COUNTS_DIRECTORY = "intermediate_tile_value_counts"
_INTERMEDIATE_COUNT_BUFFER_ROWS = 65_536
_UINT64_32 = np.uint64(32)

_POINT_PAYLOAD_SCHEMA = pa.schema(
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
    tile_hash = _splitmix64(tile_key)
    return tile_hash % np.uint64(bucket_count)


def _validate_bucket_files(
    point_path: Path,
    intermediate_count_path: Path,
    *,
    manifest_rows: list[_ManifestRow],
    intermediate_count_file: _IntermediateTileValueCountFile,
) -> None:
    point_file = pq.ParquetFile(point_path)
    if not point_file.schema_arrow.equals(_POINT_PAYLOAD_SCHEMA, check_metadata=False):
        raise ValueError(f"Point bucket `{point_path}` has an incompatible payload schema.")
    if point_file.num_row_groups != len(manifest_rows):
        raise ValueError(f"Point bucket `{point_path}` row-group count does not match its manifest rows.")
    for row_group_index, manifest_row in enumerate(manifest_rows):
        if point_file.metadata.row_group(row_group_index).num_rows != manifest_row.n_points:
            raise ValueError(f"Point bucket `{point_path}` row-group rows do not match its manifest row.")

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
    """Validate bucket outputs and assemble one deterministic level result.

    Written point rows and intermediate value-count totals must match the
    expected level count. Physical row-group keys and intermediate file paths
    must also be unique across the complete level.
    """
    ordered_results = tuple(sorted(bucket_results, key=lambda result: result.bucket_id))
    if sum(result.point_count for result in ordered_results) != expected_point_count:
        raise ValueError("Bucket rows do not reconcile to the expected level point count.")
    if sum(result.value_count_total for result in ordered_results) != expected_point_count:
        raise ValueError("Tile/value counts do not reconcile to the expected level point count.")

    manifest_rows = tuple(
        sorted(
            (row for result in ordered_results for row in result.manifest_rows),
            key=lambda row: (row.level, row.tile_y, row.tile_x, row.tile_shard),
        )
    )
    physical_keys = {(row.level_file, row.row_group) for row in manifest_rows}
    if len(physical_keys) != len(manifest_rows):
        raise ValueError("Level manifest contains duplicate physical row-group keys.")

    intermediate_files = tuple(
        result.intermediate_value_count_file
        for result in ordered_results
        if result.intermediate_value_count_file is not None
    )
    if len({file.relative_path for file in intermediate_files}) != len(intermediate_files):
        raise ValueError("Level contains duplicate intermediate tile/value-count file paths.")
    return _LevelWriteResult(
        manifest_rows=manifest_rows,
        intermediate_tile_value_count_files=intermediate_files,
    )
