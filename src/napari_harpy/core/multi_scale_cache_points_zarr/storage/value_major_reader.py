"""Read bounded location selections from one value-major cache level."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from itertools import pairwise

import numpy as np
import numpy.typing as npt
import zarr

from napari_harpy.core.multi_scale_cache_points_zarr.models import (
    _INT64_MAX,
    _require_integer_in_range,
)
from napari_harpy.core.multi_scale_cache_points_zarr.storage._row_selection import (
    _build_exact_row_selection,
)
from napari_harpy.core.multi_scale_cache_points_zarr.storage._schema import VALUE_MAJOR_LOCATION_DTYPE


class _ValueMajorLocationReader:
    """Read exact ordered row intervals from one validated sidecar array.

    The catalog reader owns and validates ``location``. This lightweight
    wrapper does not open another Zarr store; it only provides the bounded
    selection and cancellation contract needed by viewport reads.
    """

    def __init__(self, location: zarr.Array) -> None:
        if (
            not isinstance(location, zarr.Array)
            or location.ndim != 2
            or location.shape[1:] != (2,)
            or location.dtype != VALUE_MAJOR_LOCATION_DTYPE
        ):
            raise ValueError("`location` must be a two-dimensional float32 Zarr location array.")
        shards = location.shards
        if len(location.chunks) != 2 or shards is None or len(shards) != 2:
            raise ValueError("Value-major locations must use a two-dimensional sharded layout.")
        self._location = location
        # Reuse the physical shard length only as a selected-row budget for one
        # Zarr operation. This bounds the integer selector and temporary result;
        # it does not imply that the selected intervals occupy one physical
        # shard, and a single operation may access several shards.
        self._max_batch_rows = int(shards[0])

    def read_intervals(
        self,
        intervals: tuple[tuple[int, int], ...],
        *,
        expected_row_count: int,
        raise_if_cancelled: Callable[[], None] | None = None,
    ) -> npt.NDArray[np.float32]:
        """Return locations for ordered, nonoverlapping sidecar intervals.

        Adjacent intervals become one basic slice. Disjoint intervals within a
        bounded batch become one exact orthogonal row selection. Larger inputs
        are split at row boundaries so cancellation can be observed between
        Zarr operations without changing output order.
        """
        _require_integer_in_range(
            expected_row_count,
            "expected_row_count",
            maximum=_INT64_MAX,
        )
        if raise_if_cancelled is not None and not callable(raise_if_cancelled):
            raise ValueError("`raise_if_cancelled` must be callable or None.")
        intervals = _validate_intervals(intervals, point_count=int(self._location.shape[0]))
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
            self._location.get_orthogonal_selection((row_selection, slice(None))),
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
