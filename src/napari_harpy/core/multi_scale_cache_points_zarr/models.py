from __future__ import annotations

from dataclasses import dataclass

_INT16_MAX = 2**15 - 1
_INT64_MAX = 2**63 - 1
_UINT32_MAX = 2**32 - 1


@dataclass(frozen=True)
class _TileDescriptor:
    """Identify one nonempty logical tile in one finalized Zarr bucket.

    Parameters
    ----------
    level
        Non-negative serialized cache-level number containing the tile.
    bucket_id
        Deterministic identifier of the Zarr bucket containing the tile. Together
        with ``level``, it determines the canonical ``bucket_path`` property.
    bucket_tile_index
        Zero-based ordinal of this tile among all nonempty tiles in its bucket,
        after ordering them by ``(tile_y, tile_x)``. For index ``i``, the bucket
        stores this tile's identity at ``tile_x[i]`` and ``tile_y[i]``, its
        complete point interval at ``tile_offset[i:i + 2]``, and its sparse
        value-range interval at ``tile_indptr[i:i + 2]``. It is not a point
        offset, chunk number, shard number, or Parquet row group.
    tile_x
        Logical x index of the tile in this cache level's aligned tile grid.
    tile_y
        Logical y index of the tile in this cache level's aligned tile grid.
    n_points
        Number of stored points in the complete logical tile. It must equal
        ``tile_offset[i + 1] - tile_offset[i]`` in the finalized bucket, where
        ``i`` is ``bucket_tile_index``.

    Notes
    -----
    ``bucket_path`` is derived canonically from ``level`` and ``bucket_id`` so
    those integer fields are the only stored source of bucket identity.

    The descriptor is a construction result and later becomes one manifest row.
    Keeping ``bucket_tile_index`` in that row gives the runtime reader direct
    access to both tile pointer arrays without searching the bucket's coordinate
    arrays. Construction reads still reconcile ``tile_x[i]`` and ``tile_y[i]``;
    visualization trusts the independently validated manifest coordinates and
    uses the resident bucket lookup index directly.

    Value membership is deliberately absent. A tile can contain a variable and
    potentially large number of distinct ``value_id`` values; duplicating them
    here would turn the compact descriptor into a second sparse index. Instead,
    ``bucket_tile_index`` locates the tile's records through
    ``tile_indptr[i:i + 2]``, and those records store each present value together
    with its point-row start and count.
    """

    level: int
    bucket_id: int
    bucket_tile_index: int
    tile_x: int
    tile_y: int
    n_points: int

    def __post_init__(self) -> None:
        _require_integer_in_range(self.level, "level", maximum=_INT16_MAX)
        _require_integer_in_range(self.bucket_id, "bucket_id", maximum=_UINT32_MAX)
        _require_integer_in_range(self.bucket_tile_index, "bucket_tile_index", maximum=_UINT32_MAX)
        _require_integer_in_range(self.tile_x, "tile_x", maximum=_UINT32_MAX)
        _require_integer_in_range(self.tile_y, "tile_y", maximum=_UINT32_MAX)
        _require_integer_in_range(self.n_points, "n_points", minimum=1, maximum=_INT64_MAX)

    @property
    def bucket_path(self) -> str:
        """Return the canonical cache-relative Zarr path for this bucket."""
        return _bucket_path(level=self.level, bucket_id=self.bucket_id)


def _bucket_path(*, level: int, bucket_id: int) -> str:
    """Return the canonical path derived from serialized bucket identity."""
    _require_integer_in_range(level, "level", maximum=_INT16_MAX)
    _require_integer_in_range(bucket_id, "bucket_id", maximum=_UINT32_MAX)
    return f"tile_major/level_{level}/bucket-{bucket_id:03d}.zarr"


def _require_integer_in_range(
    value: object,
    name: str,
    *,
    minimum: int = 0,
    maximum: int,
) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not minimum <= value <= maximum:
        raise ValueError(f"`{name}` must be an integer in the range [{minimum}, {maximum}].")
    return value
