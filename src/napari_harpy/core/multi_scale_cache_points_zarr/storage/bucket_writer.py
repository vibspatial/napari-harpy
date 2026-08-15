from __future__ import annotations

import shutil
from enum import Enum, auto
from pathlib import Path
from types import TracebackType

import numpy as np
import zarr
from zarr.codecs import BytesCodec
from zarr.storage import LocalStore

from napari_harpy.core.multi_scale_cache_points_zarr.models import (
    _UINT32_MAX,
    _require_integer_in_range,
    _TileDescriptor,
)
from napari_harpy.core.multi_scale_cache_points_zarr.payload import _PointPayload
from napari_harpy.core.multi_scale_cache_points_zarr.storage._schema import (
    _CHUNK_KEY_ENCODING,
    _COORDINATE_ENCODING,
    _PAYLOAD_SCHEMA_VERSION,
    _POINT_ORDER,
    _compressors,
)
from napari_harpy.core.multi_scale_cache_points_zarr.storage.models import (
    _BucketPlan,
    _BucketWriteResult,
)


class _WriterState(Enum):
    NEW = auto()
    OPEN = auto()
    CLOSED = auto()


class _BucketWriter:
    """Write and finalize one independent Zarr v3 bucket sequentially.

    Parameters
    ----------
    staging_root
        Root of the disposable cache generation. The writer derives the bucket
        target exclusively from ``plan.bucket_path``.
    plan
        Ordered tile/count contract and physical chunk, shard, and codec
        settings for the bucket.

    Notes
    -----
    Entering creates a new target and refuses overwrite. ``write_tile`` accepts
    every planned tile once and in order. ``finalize`` flushes the shared point
    and range shard buffers, reconciles independent counts, writes final root
    attributes, closes the store, and returns ``_BucketWriteResult``. Any
    ordinary construction failure removes only this partial target.
    """

    def __init__(self, staging_root: str | Path, plan: _BucketPlan) -> None:
        if not isinstance(plan, _BucketPlan):
            raise ValueError("`plan` must be a _BucketPlan.")
        self._staging_root = Path(staging_root)
        self._plan = plan
        self._target = self._staging_root / plan.bucket_path
        self._state = _WriterState.NEW
        self._failed = False
        self._store: LocalStore | None = None
        self._root: zarr.Group | None = None

        self._point_location: zarr.Array | None = None
        self._point_id: zarr.Array | None = None
        self._value_id: zarr.Array | None = None
        self._range_value_id: zarr.Array | None = None
        self._range_row_start: zarr.Array | None = None
        self._range_row_count: zarr.Array | None = None

        # These shard-sized NumPy arrays only stage and coalesce writes in
        # memory. The ``shards=`` arguments in ``_create_arrays`` define the
        # physical Zarr sharding; matching that size lets us normally submit
        # one contiguous write per shard instead of repeatedly updating it.
        point_rows = plan.settings.point_shard_rows
        self._point_location_buffer = np.empty((point_rows, 2), dtype=np.float32)
        self._point_value_buffer = np.empty(point_rows, dtype=np.uint32)
        self._point_id_buffer = np.empty(point_rows, dtype=np.uint64)
        self._point_buffer_count = 0
        self._point_input_cursor = 0
        self._point_write_cursor = 0

        range_rows = plan.settings.range_shard_rows
        self._range_value_buffer = np.empty(range_rows, dtype=np.uint32)
        self._range_start_buffer = np.empty(range_rows, dtype=np.uint64)
        self._range_count_buffer = np.empty(range_rows, dtype=np.uint64)
        self._range_buffer_count = 0
        self._range_input_cursor = 0
        self._range_write_cursor = 0
        self._range_capacity = range_rows

        self._tile_indptr = np.empty(plan.tile_count + 1, dtype=np.uint64)
        self._tile_indptr[0] = 0
        self._next_tile_index = 0

    @property
    def target(self) -> Path:
        """Return the canonical absolute target path for this writer."""
        return self._target

    def __enter__(self) -> _BucketWriter:
        if self._state is not _WriterState.NEW:
            raise RuntimeError("A bucket writer can be entered only once.")
        if self._target.exists():
            raise FileExistsError(f"Zarr bucket target already exists: {self._target}")

        self._target.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._store = LocalStore(self._target, read_only=False)
            self._root = zarr.open_group(
                store=self._store,
                mode="w-",
                zarr_format=3,
                use_consolidated=False,
            )
            self._create_arrays()
        except Exception:
            self._failed = True
            self._close_and_remove()
            raise

        self._state = _WriterState.OPEN
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        del exc_value, traceback
        if self._state is _WriterState.OPEN:
            self._failed = True
            self._close_and_remove()
        return False

    def write_tile(self, tile_x: int, tile_y: int, payload: _PointPayload) -> None:
        """Append the next planned logical tile to the bucket."""
        self._require_open()
        try:
            _require_integer_in_range(tile_x, "tile_x", maximum=_UINT32_MAX)
            _require_integer_in_range(tile_y, "tile_y", maximum=_UINT32_MAX)
            if not isinstance(payload, _PointPayload):
                raise ValueError("`payload` must be a _PointPayload.")
            if self._next_tile_index >= self._plan.tile_count:
                raise ValueError("Every planned tile has already been written.")

            planned = self._plan.tiles[self._next_tile_index]
            if (tile_x, tile_y) != (planned.tile_x, planned.tile_y):
                raise ValueError("Tile coordinates do not match the next planned tile.")
            if payload.n_points != planned.n_points:
                raise ValueError("Point payload count does not match the next planned tile.")
            if bool((payload.x_rel < 0).any()) or bool((payload.y_rel < 0).any()):
                raise ValueError("Tile-relative coordinates must be nonnegative.")

            ordered = payload.ordered_by_value_and_point_id()
            tile_point_start = self._point_input_cursor
            range_values, range_starts, range_counts = _ranges_for_payload(
                ordered.value_id,
                point_start=tile_point_start,
            )
            self._append_points(ordered)
            self._append_ranges(range_values, range_starts, range_counts)
            self._tile_indptr[self._next_tile_index + 1] = self._range_input_cursor
            self._next_tile_index += 1
        except Exception:
            self._failed = True
            self._close_and_remove()
            raise

    def finalize(self) -> _BucketWriteResult:
        """Flush, reconcile, attribute, close, and describe the completed bucket."""
        self._require_open()
        try:
            if self._next_tile_index != self._plan.tile_count:
                raise RuntimeError("Cannot finalize before every planned tile is written.")
            self._flush_point_buffer()
            self._flush_range_buffer()
            self._trim_range_arrays()
            self._write_tile_indptr()
            result = self._reconcile_result()
            self._write_root_attributes(result)
            self._close_store()
            self._state = _WriterState.CLOSED
            return result
        except Exception:
            self._failed = True
            self._close_and_remove()
            raise

    def _create_arrays(self) -> None:
        root = self._require_root()
        settings = self._plan.settings
        compressors = _compressors(settings.codec_id)
        common = {
            "compressors": compressors,
            "serializer": BytesCodec(endian="little"),
            "fill_value": 0,
            "chunk_key_encoding": _CHUNK_KEY_ENCODING,
            "config": {"write_empty_chunks": True},
        }

        # In Zarr v3, ``chunks`` are the inner compressed units while
        # ``shards`` group those chunks into the physical storage objects.
        self._point_location = root.create_array(
            "location",
            shape=(self._plan.point_count, 2),
            dtype=np.float32,
            chunks=(settings.point_chunk_rows, 2),
            shards=(settings.point_shard_rows, 2),
            **common,
        )
        self._point_id = root.create_array(
            "point_id",
            shape=(self._plan.point_count,),
            dtype=np.uint64,
            chunks=(settings.point_chunk_rows,),
            shards=(settings.point_shard_rows,),
            **common,
        )
        self._value_id = root.create_array(
            "value_id",
            shape=(self._plan.point_count,),
            dtype=np.uint32,
            chunks=(settings.point_chunk_rows,),
            shards=(settings.point_shard_rows,),
            **common,
        )

        tile_count = self._plan.tile_count
        tile_x = root.create_array(
            "tile_x",
            shape=(tile_count,),
            dtype=np.uint32,
            chunks=(tile_count,),
            **common,
        )
        tile_y = root.create_array(
            "tile_y",
            shape=(tile_count,),
            dtype=np.uint32,
            chunks=(tile_count,),
            **common,
        )
        tile_offset = root.create_array(
            "tile_offset",
            shape=(tile_count + 1,),
            dtype=np.uint64,
            chunks=(tile_count + 1,),
            **common,
        )
        tile_x[:] = np.fromiter((tile.tile_x for tile in self._plan.tiles), dtype=np.uint32, count=tile_count)
        tile_y[:] = np.fromiter((tile.tile_y for tile in self._plan.tiles), dtype=np.uint32, count=tile_count)
        tile_offset[:] = self._plan.tile_offset

        ranges = root.create_group("ranges")
        tile_indptr = ranges.create_array(
            "tile_indptr",
            shape=(tile_count + 1,),
            dtype=np.uint64,
            chunks=(tile_count + 1,),
            **common,
        )
        del tile_indptr
        self._range_value_id = ranges.create_array(
            "value_id",
            shape=(self._range_capacity,),
            dtype=np.uint32,
            chunks=(settings.range_chunk_rows,),
            shards=(settings.range_shard_rows,),
            **common,
        )
        self._range_row_start = ranges.create_array(
            "row_start",
            shape=(self._range_capacity,),
            dtype=np.uint64,
            chunks=(settings.range_chunk_rows,),
            shards=(settings.range_shard_rows,),
            **common,
        )
        self._range_row_count = ranges.create_array(
            "row_count",
            shape=(self._range_capacity,),
            dtype=np.uint64,
            chunks=(settings.range_chunk_rows,),
            shards=(settings.range_shard_rows,),
            **common,
        )

    def _append_points(self, payload: _PointPayload) -> None:
        source_cursor = 0
        buffer_rows = len(self._point_value_buffer)
        while source_cursor < payload.n_points:
            available = buffer_rows - self._point_buffer_count
            take = min(available, payload.n_points - source_cursor)
            source_slice = slice(source_cursor, source_cursor + take)
            target_slice = slice(self._point_buffer_count, self._point_buffer_count + take)
            self._point_location_buffer[target_slice, 0] = payload.x_rel[source_slice]
            self._point_location_buffer[target_slice, 1] = payload.y_rel[source_slice]
            self._point_value_buffer[target_slice] = payload.value_id[source_slice]
            self._point_id_buffer[target_slice] = payload.point_id[source_slice]
            source_cursor += take
            self._point_buffer_count += take
            self._point_input_cursor += take
            if self._point_buffer_count == buffer_rows:
                self._flush_point_buffer()

    def _flush_point_buffer(self) -> None:
        if self._point_buffer_count == 0:
            return
        start = self._point_write_cursor
        stop = start + self._point_buffer_count
        self._require_array(self._point_location, "location")[start:stop, :] = self._point_location_buffer[
            : self._point_buffer_count
        ]
        self._require_array(self._value_id, "value_id")[start:stop] = self._point_value_buffer[
            : self._point_buffer_count
        ]
        self._require_array(self._point_id, "point_id")[start:stop] = self._point_id_buffer[
            : self._point_buffer_count
        ]
        self._point_write_cursor = stop
        self._point_buffer_count = 0

    def _append_ranges(
        self,
        values: np.ndarray,
        starts: np.ndarray,
        counts: np.ndarray,
    ) -> None:
        """Stage sparse range records and flush each complete shard buffer.

        The three arrays are parallel: row ``i`` describes one contiguous
        ``value_id`` run using ``values[i]``, its bucket-global point-array
        start ``starts[i]``, and its length ``counts[i]``. They normally come
        directly from ``_ranges_for_payload`` for one sorted tile. Records may
        share a buffer with neighboring tiles; tile boundaries are tracked
        separately by ``tile_indptr``.

        Parameters
        ----------
        values
            Value identifier for each run.
        starts
            Bucket-global starting point row for each run.
        counts
            Number of consecutive point rows in each run.
        """
        source_cursor = 0
        total = len(values)
        buffer_rows = len(self._range_value_buffer)
        while source_cursor < total:
            available = buffer_rows - self._range_buffer_count
            take = min(available, total - source_cursor)
            source_slice = slice(source_cursor, source_cursor + take)
            target_slice = slice(self._range_buffer_count, self._range_buffer_count + take)
            self._range_value_buffer[target_slice] = values[source_slice]
            self._range_start_buffer[target_slice] = starts[source_slice]
            self._range_count_buffer[target_slice] = counts[source_slice]
            source_cursor += take
            self._range_buffer_count += take
            self._range_input_cursor += take
            if self._range_buffer_count == buffer_rows:
                self._flush_range_buffer()

    def _flush_range_buffer(self) -> None:
        if self._range_buffer_count == 0:
            return
        start = self._range_write_cursor
        stop = start + self._range_buffer_count
        self._ensure_range_capacity(stop)
        self._require_array(self._range_value_id, "ranges/value_id")[start:stop] = self._range_value_buffer[
            : self._range_buffer_count
        ]
        self._require_array(self._range_row_start, "ranges/row_start")[start:stop] = self._range_start_buffer[
            : self._range_buffer_count
        ]
        self._require_array(self._range_row_count, "ranges/row_count")[start:stop] = self._range_count_buffer[
            : self._range_buffer_count
        ]
        self._range_write_cursor = stop
        self._range_buffer_count = 0

    def _ensure_range_capacity(self, required: int) -> None:
        """Grow the parallel Zarr range arrays to cover an exclusive endpoint.

        ``required`` is the minimum array length needed for the next
        ``[start:required]`` write. Capacity grows geometrically and remains
        aligned to complete range shards, avoiding frequent Zarr resizes. This
        does not resize the fixed, one-shard NumPy staging buffers; excess Zarr
        capacity is removed during finalization.

        Parameters
        ----------
        required
            Exclusive endpoint of the pending bucket-global range write.
        """
        if required <= self._range_capacity:
            return
        shard_rows = self._plan.settings.range_shard_rows
        capacity = self._range_capacity
        while capacity < required:
            capacity *= 2
        capacity = ((capacity + shard_rows - 1) // shard_rows) * shard_rows
        for name, array in (
            ("ranges/value_id", self._range_value_id),
            ("ranges/row_start", self._range_row_start),
            ("ranges/row_count", self._range_row_count),
        ):
            self._require_array(array, name).resize((capacity,))
        self._range_capacity = capacity

    def _trim_range_arrays(self) -> None:
        for name, array in (
            ("ranges/value_id", self._range_value_id),
            ("ranges/row_start", self._range_row_start),
            ("ranges/row_count", self._range_row_count),
        ):
            self._require_array(array, name).resize((self._range_write_cursor,))
        self._range_capacity = self._range_write_cursor

    def _write_tile_indptr(self) -> None:
        root = self._require_root()
        array = root["ranges/tile_indptr"]
        if not isinstance(array, zarr.Array):
            raise RuntimeError("`ranges/tile_indptr` is not a Zarr array.")
        array[:] = self._tile_indptr

    def _reconcile_result(self) -> _BucketWriteResult:
        location = self._require_array(self._point_location, "location")
        point_id = self._require_array(self._point_id, "point_id")
        value_id = self._require_array(self._value_id, "value_id")
        physical_point_count = int(location.shape[0])
        if location.shape != (physical_point_count, 2):
            raise RuntimeError("Final location shape is inconsistent.")
        if point_id.shape != (physical_point_count,) or value_id.shape != (physical_point_count,):
            raise RuntimeError("Final point-array shapes are inconsistent.")

        descriptors = tuple(
            _TileDescriptor(
                level=self._plan.level,
                bucket_id=self._plan.bucket_id,
                bucket_tile_index=index,
                tile_x=tile.tile_x,
                tile_y=tile.tile_y,
                n_points=tile.n_points,
            )
            for index, tile in enumerate(self._plan.tiles)
        )
        expected_point_counts = (
            self._point_input_cursor,
            self._point_write_cursor,
            int(self._plan.tile_offset[-1]),
            sum(tile.n_points for tile in descriptors),
            self._plan.point_count,
        )
        if any(count != physical_point_count for count in expected_point_counts):
            raise RuntimeError("Final physical and logical point counts do not reconcile.")

        range_value = self._require_array(self._range_value_id, "ranges/value_id")
        range_start = self._require_array(self._range_row_start, "ranges/row_start")
        range_count = self._require_array(self._range_row_count, "ranges/row_count")
        physical_range_count = int(range_value.shape[0])
        if range_start.shape != (physical_range_count,) or range_count.shape != (physical_range_count,):
            raise RuntimeError("Final range-array shapes are inconsistent.")
        expected_range_counts = (
            self._range_input_cursor,
            self._range_write_cursor,
            int(self._tile_indptr[-1]),
        )
        if any(count != physical_range_count for count in expected_range_counts):
            raise RuntimeError("Final physical and logical range counts do not reconcile.")

        return _BucketWriteResult(
            tile_descriptors=descriptors,
            point_count=physical_point_count,
            range_count=physical_range_count,
        )

    def _write_root_attributes(self, result: _BucketWriteResult) -> None:
        self._require_root().attrs.update(
            {
                "payload_schema_version": _PAYLOAD_SCHEMA_VERSION,
                "level": self._plan.level,
                "bucket_id": self._plan.bucket_id,
                "tile_count": self._plan.tile_count,
                "point_count": result.point_count,
                "range_count": result.range_count,
                "point_order": list(_POINT_ORDER),
                "coordinate_encoding": _COORDINATE_ENCODING,
                "codec_id": self._plan.settings.codec_id,
            }
        )

    def _require_open(self) -> None:
        if self._state is not _WriterState.OPEN or self._failed:
            raise RuntimeError("Bucket writer is not open.")

    def _require_root(self) -> zarr.Group:
        if self._root is None:
            raise RuntimeError("Bucket Zarr group is not open.")
        return self._root

    @staticmethod
    def _require_array(array: zarr.Array | None, name: str) -> zarr.Array:
        if array is None:
            raise RuntimeError(f"Zarr array is not open: {name}.")
        return array

    def _close_store(self) -> None:
        if self._store is not None:
            self._store.close()
        self._store = None
        self._root = None

    def _close_and_remove(self) -> None:
        try:
            self._close_store()
        finally:
            if self._target.exists():
                shutil.rmtree(self._target)
            self._state = _WriterState.CLOSED


def _ranges_for_payload(
    value_id: np.ndarray,
    *,
    point_start: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Encode one sorted tile's values as bucket-global sparse ranges.

    ``value_id`` must be nonempty and ordered by value. Consequently, every
    distinct value occupies one contiguous run and needs exactly one range
    record. Change positions identify the local run starts; ``point_start``
    rebases those positions into the bucket's concatenated point arrays.

    Parameters
    ----------
    value_id
        Ordered value identifiers for all points in one nonempty tile.
    point_start
        Bucket-global row at which this tile begins.

    Returns
    -------
    values
        Value identifier for each contiguous run.
    starts
        Bucket-global starting point row for each run.
    counts
        Number of consecutive point rows in each run.

    Examples
    --------
    Values ``[2, 2, 5, 9, 9]`` at ``point_start=10`` produce values
    ``[2, 5, 9]``, starts ``[10, 12, 13]``, and counts ``[2, 1, 2]``.
    """
    changes = np.empty(len(value_id), dtype=np.bool_)
    changes[0] = True
    changes[1:] = value_id[1:] != value_id[:-1]
    local_starts = np.flatnonzero(changes).astype(np.uint64, copy=False)
    local_ends = np.empty_like(local_starts)
    local_ends[:-1] = local_starts[1:]
    local_ends[-1] = len(value_id)
    values = np.ascontiguousarray(value_id[local_starts])
    starts = np.ascontiguousarray(local_starts + np.uint64(point_start))
    counts = np.ascontiguousarray(local_ends - local_starts)
    return values, starts, counts
