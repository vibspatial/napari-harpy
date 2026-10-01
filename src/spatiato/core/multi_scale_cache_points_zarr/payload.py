from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True, eq=False)
class _PointPayload:
    """Hold the actual aligned point rows for one nonempty logical tile.

    This is the data passed between logical construction, sampling, rebasing,
    and physical storage. It contains only tile-local point fields and is
    deliberately unaware of cache level, tile coordinates, bucket identity,
    output path, other tiles, and Zarr write settings.

    A bucket writer receives the tile coordinates separately and checks
    ``n_points`` against the corresponding expected count in its
    ``_BucketPlan``. Keeping the payload separate from that small bucket-wide
    plan allows one tile's arrays to be materialized, written, and released at
    a time instead of retaining all bucket points in planning metadata.

    The arrays are borrowed rather than copied. Read-only views prevent
    mutation through this object, but callers must not mutate the original
    backing arrays while the payload is in use.
    """

    x_rel: npt.NDArray[np.float32]
    y_rel: npt.NDArray[np.float32]
    value_id: npt.NDArray[np.uint32]
    point_id: npt.NDArray[np.uint64]

    def __post_init__(self) -> None:
        arrays = (
            ("x_rel", self.x_rel, np.dtype(np.float32)),
            ("y_rel", self.y_rel, np.dtype(np.float32)),
            ("value_id", self.value_id, np.dtype(np.uint32)),
            ("point_id", self.point_id, np.dtype(np.uint64)),
        )
        row_count: int | None = None
        for name, array, expected_dtype in arrays:
            if not isinstance(array, np.ndarray):
                raise ValueError(f"`{name}` must be a NumPy array.")
            if array.ndim != 1:
                raise ValueError(f"`{name}` must be one-dimensional.")
            if array.dtype != expected_dtype:
                raise ValueError(f"`{name}` must have dtype {expected_dtype.name}.")
            if not array.flags.c_contiguous:
                raise ValueError(f"`{name}` must be C-contiguous.")
            if row_count is None:
                row_count = len(array)
            elif len(array) != row_count:
                raise ValueError("Point-payload arrays must have equal lengths.")

        if row_count == 0:
            raise ValueError("A point payload must contain at least one row.")
        if not bool(np.isfinite(self.x_rel).all()) or not bool(np.isfinite(self.y_rel).all()):
            raise ValueError("Point-payload coordinates must be finite.")

        # Frozen fields do not make NumPy contents immutable. Install read-only
        # views without copying data or changing the caller-owned array flags;
        # frozen post-init normalization requires `object.__setattr__`.
        for name, array, _ in arrays:
            read_only = array.view()
            read_only.flags.writeable = False
            object.__setattr__(self, name, read_only)

    @property
    def n_points(self) -> int:
        """Return the aligned payload row count."""
        return len(self.point_id)

    def take(self, indices: npt.NDArray[np.int64]) -> _PointPayload:
        """Take unique in-bounds rows while preserving four-field alignment."""
        if not isinstance(indices, np.ndarray):
            raise ValueError("`indices` must be a NumPy array.")
        if indices.ndim != 1:
            raise ValueError("`indices` must be one-dimensional.")
        if indices.dtype != np.dtype(np.int64):
            raise ValueError("`indices` must have dtype int64.")
        if not indices.flags.c_contiguous:
            raise ValueError("`indices` must be C-contiguous.")
        if len(indices) == 0:
            raise ValueError("`indices` must select at least one point.")
        if int(indices.min()) < 0 or int(indices.max()) >= self.n_points:
            raise ValueError("`indices` contains an out-of-bounds point index.")
        if len(np.unique(indices)) != len(indices):
            raise ValueError("`indices` must not contain duplicate point indexes.")
        return _PointPayload(
            x_rel=self.x_rel[indices],
            y_rel=self.y_rel[indices],
            value_id=self.value_id[indices],
            point_id=self.point_id[indices],
        )

    def ordered_by_value_and_point_id(self) -> _PointPayload:
        """Return the same rows in deterministic value-major point-ID order."""
        order = np.lexsort((self.point_id, self.value_id)).astype(np.int64, copy=False)
        # `lexsort` already returns a valid unique permutation, but `take`
        # deliberately rechecks uniqueness with `np.unique`. If profiling shows
        # that cost matters, add a private trusted indexing helper for this path
        # rather than weakening the general caller-facing `take` contract.
        return self.take(order)
