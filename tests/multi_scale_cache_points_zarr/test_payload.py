from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from napari_harpy.core.multi_scale_cache_points_zarr.payload import _PointPayload


def _payload() -> _PointPayload:
    return _PointPayload(
        x_rel=np.array([3.0, 1.0, 2.0], dtype=np.float32),
        y_rel=np.array([30.0, 10.0, 20.0], dtype=np.float32),
        value_id=np.array([1, 0, 1], dtype=np.uint32),
        point_id=np.array([7, 9, 3], dtype=np.uint64),
    )


def test_payload_preserves_exact_aligned_arrays_as_read_only_views() -> None:
    x_rel = np.array([1.0, 2.0], dtype=np.float32)
    payload = _PointPayload(
        x_rel=x_rel,
        y_rel=np.array([3.0, 4.0], dtype=np.float32),
        value_id=np.array([5, 6], dtype=np.uint32),
        point_id=np.array([7, 8], dtype=np.uint64),
    )

    assert payload.n_points == 2
    assert np.shares_memory(payload.x_rel, x_rel)
    assert all(
        not array.flags.writeable for array in (payload.x_rel, payload.y_rel, payload.value_id, payload.point_id)
    )
    with pytest.raises(ValueError, match="read-only"):
        payload.x_rel[0] = 9.0
    with pytest.raises(FrozenInstanceError):
        payload.x_rel = np.array([1.0], dtype=np.float32)  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field_name", "value", "message"),
    [
        ("x_rel", [1.0], "NumPy"),
        ("x_rel", np.array([[1.0]], dtype=np.float32), "one-dimensional"),
        ("x_rel", np.array([1.0], dtype=np.float64), "float32"),
        ("value_id", np.array([1], dtype=np.int32), "uint32"),
        ("point_id", np.array([1], dtype=np.uint32), "uint64"),
        ("x_rel", np.arange(6, dtype=np.float32)[::2], "C-contiguous"),
    ],
)
def test_payload_rejects_wrong_array_contract(field_name: str, value: object, message: str) -> None:
    values: dict[str, object] = {
        "x_rel": np.array([1.0], dtype=np.float32),
        "y_rel": np.array([2.0], dtype=np.float32),
        "value_id": np.array([3], dtype=np.uint32),
        "point_id": np.array([4], dtype=np.uint64),
    }
    values[field_name] = value

    with pytest.raises(ValueError, match=message):
        _PointPayload(**values)  # type: ignore[arg-type]


def test_payload_rejects_empty_misaligned_and_nonfinite_arrays() -> None:
    with pytest.raises(ValueError, match="at least one"):
        _PointPayload(
            np.array([], dtype=np.float32),
            np.array([], dtype=np.float32),
            np.array([], dtype=np.uint32),
            np.array([], dtype=np.uint64),
        )
    with pytest.raises(ValueError, match="equal lengths"):
        _PointPayload(
            np.array([1.0, 2.0], dtype=np.float32),
            np.array([1.0], dtype=np.float32),
            np.array([1], dtype=np.uint32),
            np.array([1], dtype=np.uint64),
        )
    with pytest.raises(ValueError, match="finite"):
        _PointPayload(
            np.array([np.nan], dtype=np.float32),
            np.array([1.0], dtype=np.float32),
            np.array([1], dtype=np.uint32),
            np.array([1], dtype=np.uint64),
        )


def test_take_preserves_alignment_and_requested_order() -> None:
    selected = _payload().take(np.array([2, 0], dtype=np.int64))

    assert selected.x_rel.tolist() == [2.0, 3.0]
    assert selected.y_rel.tolist() == [20.0, 30.0]
    assert selected.value_id.tolist() == [1, 1]
    assert selected.point_id.tolist() == [3, 7]


@pytest.mark.parametrize(
    ("indices", "message"),
    [
        ([0], "NumPy"),
        (np.array([[0]], dtype=np.int64), "one-dimensional"),
        (np.array([0], dtype=np.int32), "int64"),
        (np.arange(4, dtype=np.int64)[::2], "C-contiguous"),
        (np.array([], dtype=np.int64), "at least one"),
        (np.array([-1], dtype=np.int64), "out-of-bounds"),
        (np.array([3], dtype=np.int64), "out-of-bounds"),
        (np.array([1, 1], dtype=np.int64), "duplicate"),
    ],
)
def test_take_rejects_invalid_indices(indices: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _payload().take(indices)  # type: ignore[arg-type]


def test_value_major_ordering_is_deterministic_and_membership_neutral() -> None:
    ordered = _payload().ordered_by_value_and_point_id()

    assert ordered.value_id.tolist() == [0, 1, 1]
    assert ordered.point_id.tolist() == [9, 3, 7]
    assert ordered.x_rel.tolist() == [1.0, 2.0, 3.0]
    assert set(ordered.point_id.tolist()) == set(_payload().point_id.tolist())
