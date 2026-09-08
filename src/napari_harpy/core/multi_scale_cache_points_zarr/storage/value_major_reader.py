"""Own validated array references and bounded reads for one value-major level."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from itertools import pairwise

import numpy as np
import numpy.typing as npt
import zarr

from napari_harpy.core.multi_scale_cache_points_zarr.cache_format import _ValueMajorMetadata
from napari_harpy.core.multi_scale_cache_points_zarr.models import (
    _INT64_MAX,
    _require_integer_in_range,
)
from napari_harpy.core.multi_scale_cache_points_zarr.storage._row_selection import (
    _build_exact_row_selection,
)
from napari_harpy.core.multi_scale_cache_points_zarr.storage._schema import (
    VALUE_MAJOR_LOCATION_DTYPE,
    VALUE_MAJOR_POINTER_DTYPE,
    value_major_location,
    value_major_point_indptr,
)
from napari_harpy.core.multi_scale_cache_points_zarr.storage.bucket_validation import (
    _strict_array,
    _validate_array_layout,
)


class _ValueMajorLevelReader:
    """Own the two Zarr array references for one value-major cache level.

    ``_CacheRootReader`` supplies its already-open root group and owns the
    shared store. This reader opens and validates
    ``value_major/level_N/location`` and
    ``value_major/level_N/value_point_indptr`` without decoding either array.
    It opens no additional store and retains no decoded payloads.

    ``load_point_indptr()`` returns a read-only NumPy vector owned by its caller;
    it is not cached here. ``read_intervals()`` accepts physical location-row
    intervals, not viewport plans or value selections. The root reader closes
    this reader before its store, invalidating even borrowed reader references.

    Parameters
    ----------
    root
        Already-open cache-root Zarr group; the caller owns its store.
    level
        Serialized level whose arrays to open.
    point_count
        Expected number of point rows at this level.
    value_count
        Number of canonical values across the cache, including values with
        zero points at this level.
    metadata
        Expected chunk and shard row counts for value-major point arrays.
    codec_id
        Expected cache-wide compression profile.
    """

    def __init__(
        self,
        root: zarr.Group,
        *,
        level: int,
        point_count: int,
        value_count: int,
        metadata: _ValueMajorMetadata,
        codec_id: str,
    ) -> None:
        # Install references only after both layouts pass validation. A failed
        # constructor neither owns a store to close nor publishes usable arrays.
        location = _strict_array(root, value_major_location(level))
        point_indptr = _strict_array(root, value_major_point_indptr(level))
        _validate_array_layout(
            location,
            name=value_major_location(level),
            dtype=VALUE_MAJOR_LOCATION_DTYPE,
            shape=(point_count, 2),
            chunks=(metadata.point_chunk_rows, 2),
            shards=(metadata.point_shard_rows, 2),
            codec_id=codec_id,
        )
        _validate_array_layout(
            point_indptr,
            name=value_major_point_indptr(level),
            dtype=VALUE_MAJOR_POINTER_DTYPE,
            shape=(value_count + 1,),
            chunks=(value_count + 1,),
            shards=None,
            codec_id=codec_id,
        )
        self._location: zarr.Array | None = location
        self._point_indptr: zarr.Array | None = point_indptr
        # Reuse the physical shard length only as a selected-row budget for one
        # Zarr operation. This bounds the integer selector and temporary result;
        # it does not imply that the selected intervals occupy one physical
        # shard, and a single operation may access several shards.
        self._max_batch_rows = metadata.point_shard_rows

    def load_point_indptr(self) -> npt.NDArray[np.uint64]:
        """Read the complete pointer vector without retaining a NumPy copy here.

        The caller controls residency and content validation: runtime startup
        checks pointer bounds/order, while publication validation additionally
        reconciles counts against the independent value-tile records.
        """
        if self._point_indptr is None:
            raise RuntimeError("Value-major level reader is closed.")
        pointer = np.ascontiguousarray(self._point_indptr[:], dtype=VALUE_MAJOR_POINTER_DTYPE)
        pointer.flags.writeable = False
        return pointer

    def close(self) -> None:
        """Release array references without closing the root reader's shared store."""
        self._location = None
        self._point_indptr = None

    def _location_or_raise(self) -> zarr.Array:
        if self._location is None:
            raise RuntimeError("Value-major level reader is closed.")
        return self._location

    def read_intervals(
        self,
        intervals: tuple[tuple[int, int], ...],
        *,
        expected_row_count: int,
        raise_if_cancelled: Callable[[], None] | None = None,
    ) -> npt.NDArray[np.float32]:
        """Return locations for ordered, nonoverlapping value-major intervals.

        Adjacent intervals become one basic slice. Disjoint intervals within a
        bounded batch become one exact orthogonal row selection. Larger inputs
        are split at row boundaries so cancellation can be observed between
        Zarr operations without changing output order.
        """
        location = self._location_or_raise()
        _require_integer_in_range(
            expected_row_count,
            "expected_row_count",
            maximum=_INT64_MAX,
        )
        if raise_if_cancelled is not None and not callable(raise_if_cancelled):
            raise ValueError("`raise_if_cancelled` must be callable or None.")
        intervals = _validate_intervals(intervals, point_count=int(location.shape[0]))
        if sum(stop - start for start, stop in intervals) != expected_row_count:
            raise ValueError("Value-major intervals do not match `expected_row_count`.")
        if expected_row_count == 0:
            return np.empty((0, 2), dtype=np.float32)

        batches = tuple(_split_intervals_by_rows(intervals, max_rows=self._max_batch_rows))
        if len(batches) == 1:
            return self._read_batch(batches[0], raise_if_cancelled=raise_if_cancelled)

        output = np.empty((expected_row_count, 2), dtype=np.float32)
        output_start = 0
        for batch in batches:
            locations = self._read_batch(batch, raise_if_cancelled=raise_if_cancelled)
            output_stop = output_start + len(locations)
            output[output_start:output_stop] = locations
            output_start = output_stop
        if output_start != expected_row_count:
            raise RuntimeError("Value-major batches did not fill the expected output rows.")
        return output

    def _read_batch(
        self,
        intervals: tuple[tuple[int, int], ...],
        *,
        raise_if_cancelled: Callable[[], None] | None,
    ) -> npt.NDArray[np.float32]:
        if raise_if_cancelled is not None:
            raise_if_cancelled()
        expected_row_count = sum(stop - start for start, stop in intervals)
        row_selection = _build_exact_row_selection(intervals)
        locations = np.ascontiguousarray(
            self._location_or_raise().get_orthogonal_selection((row_selection, slice(None))),
            dtype=np.float32,
        )
        expected_shape = (expected_row_count, 2)
        if locations.shape != expected_shape:
            raise RuntimeError("Value-major location selection returned an unexpected shape.")
        if raise_if_cancelled is not None:
            raise_if_cancelled()
        return locations


def _validate_intervals(
    intervals: tuple[tuple[int, int], ...],
    *,
    point_count: int,
) -> tuple[tuple[int, int], ...]:
    if not isinstance(intervals, tuple) or any(
        not isinstance(interval, tuple)
        or len(interval) != 2
        or any(not isinstance(value, int) or isinstance(value, bool) for value in interval)
        for interval in intervals
    ):
        raise ValueError("`intervals` must contain integer (start, stop) pairs.")
    if any(start < 0 or start >= stop or stop > point_count for start, stop in intervals):
        raise ValueError("Value-major intervals must be nonempty and lie inside the location array.")
    if any(start < previous_stop for (_, previous_stop), (start, _) in pairwise(intervals)):
        raise ValueError("Value-major intervals must be ordered and nonoverlapping.")
    return intervals


def _split_intervals_by_rows(
    intervals: tuple[tuple[int, int], ...],
    *,
    max_rows: int,
) -> Iterator[tuple[tuple[int, int], ...]]:
    """Yield ordered interval batches containing at most ``max_rows`` rows."""
    _require_integer_in_range(max_rows, "max_rows", minimum=1, maximum=_INT64_MAX)
    batch: list[tuple[int, int]] = []
    batch_rows = 0
    for start, stop in intervals:
        cursor = start
        while cursor < stop:
            available = max_rows - batch_rows
            fragment_stop = min(stop, cursor + available)
            batch.append((cursor, fragment_stop))
            batch_rows += fragment_stop - cursor
            cursor = fragment_stop
            if batch_rows == max_rows:
                yield tuple(batch)
                batch = []
                batch_rows = 0
    if batch:
        yield tuple(batch)
