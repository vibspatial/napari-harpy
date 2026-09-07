"""Test bounded location-interval reads from value-major storage."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import zarr
from zarr.storage import LocalStore

from napari_harpy.core.multi_scale_cache_points_zarr.storage.value_major_reader import (
    _ValueMajorLocationReader,
)


@pytest.fixture
def value_major_location(tmp_path: Path):
    values = np.arange(20, dtype=np.float32).reshape(10, 2)
    with LocalStore(tmp_path, read_only=False) as store:
        root = zarr.open_group(store=store, mode="w", zarr_format=3)
        yield root.create_array(
            "location",
            data=values,
            chunks=(2, 2),
            shards=(4, 2),
        )


def test_value_major_reader_reads_adjacent_and_disjoint_intervals(value_major_location: zarr.Array) -> None:
    reader = _ValueMajorLocationReader(value_major_location)

    adjacent = reader.read_intervals(((1, 3), (3, 4)), expected_row_count=3)
    disjoint = reader.read_intervals(((0, 2), (5, 7)), expected_row_count=4)

    assert adjacent.tolist() == [[2.0, 3.0], [4.0, 5.0], [6.0, 7.0]]
    assert disjoint.tolist() == [[0.0, 1.0], [2.0, 3.0], [10.0, 11.0], [12.0, 13.0]]
    assert adjacent.flags.c_contiguous
    assert disjoint.flags.c_contiguous


def test_value_major_reader_splits_at_shard_row_bound_and_checks_cancellation(
    value_major_location: zarr.Array,
) -> None:
    reader = _ValueMajorLocationReader(value_major_location)
    cancellation_checks = 0

    def check() -> None:
        nonlocal cancellation_checks
        cancellation_checks += 1

    result = reader.read_intervals(
        ((0, 2), (4, 7)),
        expected_row_count=5,
        raise_if_cancelled=check,
    )

    assert result.tolist() == [[0.0, 1.0], [2.0, 3.0], [8.0, 9.0], [10.0, 11.0], [12.0, 13.0]]
    assert cancellation_checks == 4


def test_value_major_reader_propagates_cancellation_between_bounded_reads(
    value_major_location: zarr.Array,
) -> None:
    reader = _ValueMajorLocationReader(value_major_location)
    cancellation_checks = 0

    def raise_on_second_batch() -> None:
        nonlocal cancellation_checks
        cancellation_checks += 1
        if cancellation_checks == 3:
            raise RuntimeError("cancelled")

    with pytest.raises(RuntimeError, match="cancelled"):
        reader.read_intervals(
            ((0, 2), (4, 7)),
            expected_row_count=5,
            raise_if_cancelled=raise_on_second_batch,
        )

    assert cancellation_checks == 3


@pytest.mark.parametrize(
    ("intervals", "expected_row_count", "match"),
    [
        (((0, 2),), 1, "expected_row_count"),
        (((2, 4), (3, 5)), 4, "ordered and nonoverlapping"),
        (((-1, 1),), 2, "inside the location array"),
        (((9, 11),), 2, "inside the location array"),
    ],
)
def test_value_major_reader_rejects_invalid_interval_contracts(
    value_major_location: zarr.Array,
    intervals: tuple[tuple[int, int], ...],
    expected_row_count: int,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        _ValueMajorLocationReader(value_major_location).read_intervals(
            intervals,
            expected_row_count=expected_row_count,
        )
