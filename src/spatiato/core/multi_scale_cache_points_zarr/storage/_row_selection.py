"""Build exact Zarr row selectors from validated intervals."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt


def _build_exact_row_selection(
    intervals: Sequence[tuple[int, int]],
) -> slice | npt.NDArray[np.int64]:
    """Return the cheapest exact selector for validated row intervals.

    Callers retain their domain-specific validation and must provide nonempty,
    ordered, nonoverlapping half-open intervals. Adjacent intervals become one
    basic slice. Disjoint intervals become one C-contiguous ``int64`` selector
    without allocating a temporary array for every interval.
    """
    if not intervals:
        raise ValueError("`intervals` must be nonempty.")

    merged: list[tuple[int, int]] = []
    row_count = 0
    for start, stop in intervals:
        row_count += stop - start
        if merged and start == merged[-1][1]:
            merged[-1] = (merged[-1][0], stop)
        else:
            merged.append((start, stop))

    if len(merged) == 1:
        return slice(*merged[0])

    # Begin with output positions and shift each destination segment in place
    # to its source interval. This fills one selector allocation without
    # constructing one temporary array per disjoint interval.
    selected_rows = np.arange(row_count, dtype=np.int64)
    cursor = 0
    for start, stop in merged:
        count = stop - start
        selected_rows[cursor : cursor + count] += start - cursor
        cursor += count
    return selected_rows
