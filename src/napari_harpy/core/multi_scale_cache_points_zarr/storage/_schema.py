from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
from zarr.codecs import ZstdCodec

from napari_harpy.core.multi_scale_cache_points_zarr.models import (
    _INT16_MAX,
    _INT64_MAX,
    _UINT32_MAX,
    _require_integer_in_range,
)
from napari_harpy.core.multi_scale_cache_points_zarr.storage.models import _ZarrWriteSettings

_PAYLOAD_SCHEMA_VERSION: Final = 1
_POINT_ORDER: Final = ("tile_y", "tile_x", "value_id", "point_id")
_COORDINATE_ENCODING: Final = "tile-relative-xy-float32-v1"
_ZSTD_CODEC_ID: Final = "zstd-v1"
_CHUNK_KEY_ENCODING: Final = {
    "name": "default",
    "configuration": {"separator": "/"},
}

_ROOT_ATTRIBUTE_KEYS: Final = frozenset(
    {
        "payload_schema_version",
        "level",
        "bucket_id",
        "tile_count",
        "point_count",
        "range_count",
        "point_order",
        "coordinate_encoding",
        "point_chunk_rows",
        "point_shard_rows",
        "range_chunk_rows",
        "range_shard_rows",
        "codec_id",
    }
)

_POINT_DTYPES: Final = {
    "location": np.dtype(np.float32),
    "point_id": np.dtype(np.uint64),
    "value_id": np.dtype(np.uint32),
}
_TILE_DTYPES: Final = {
    "tile_x": np.dtype(np.uint32),
    "tile_y": np.dtype(np.uint32),
    "tile_offset": np.dtype(np.uint64),
}
_RANGE_DTYPES: Final = {
    "tile_indptr": np.dtype(np.uint64),
    "value_id": np.dtype(np.uint32),
    "row_start": np.dtype(np.uint64),
    "row_count": np.dtype(np.uint64),
}


@dataclass(frozen=True)
class _BucketAttributes:
    level: int
    bucket_id: int
    tile_count: int
    point_count: int
    range_count: int
    settings: _ZarrWriteSettings


def _compressors(codec_id: str) -> tuple[ZstdCodec]:
    """Return the exact inner-chunk compressor for a supported codec ID."""
    if codec_id != _ZSTD_CODEC_ID:
        raise ValueError(f"Unsupported Zarr bucket codec ID: {codec_id!r}.")
    return (ZstdCodec(level=3, checksum=True),)


def _parse_root_attributes(
    attributes: Mapping[str, Any],
    *,
    expected_level: int,
    expected_bucket_id: int,
) -> _BucketAttributes:
    """Validate exact schema-v1 root attributes and return typed physical facts."""
    if set(attributes) != _ROOT_ATTRIBUTE_KEYS:
        raise ValueError("Zarr bucket root attributes do not match payload schema version 1.")
    if type(attributes["payload_schema_version"]) is not int or (
        attributes["payload_schema_version"] != _PAYLOAD_SCHEMA_VERSION
    ):
        raise ValueError("Unsupported Zarr bucket payload schema version.")
    if attributes["point_order"] != list(_POINT_ORDER):
        raise ValueError("Unsupported Zarr bucket point ordering.")
    if attributes["coordinate_encoding"] != _COORDINATE_ENCODING:
        raise ValueError("Unsupported Zarr bucket coordinate encoding.")

    level = _require_integer_in_range(attributes["level"], "level", maximum=_INT16_MAX)
    bucket_id = _require_integer_in_range(attributes["bucket_id"], "bucket_id", maximum=_UINT32_MAX)
    if (level, bucket_id) != (expected_level, expected_bucket_id):
        raise ValueError("Zarr bucket attributes do not match the requested bucket identity.")
    tile_count = _require_integer_in_range(
        attributes["tile_count"],
        "tile_count",
        minimum=1,
        maximum=_UINT32_MAX,
    )
    point_count = _require_integer_in_range(
        attributes["point_count"],
        "point_count",
        minimum=1,
        maximum=_INT64_MAX,
    )
    range_count = _require_integer_in_range(
        attributes["range_count"],
        "range_count",
        minimum=1,
        maximum=_INT64_MAX,
    )
    if not tile_count <= range_count <= point_count:
        raise ValueError("Zarr bucket tile, range, and point counts are inconsistent.")

    settings = _ZarrWriteSettings(
        point_chunk_rows=attributes["point_chunk_rows"],
        point_shard_rows=attributes["point_shard_rows"],
        range_chunk_rows=attributes["range_chunk_rows"],
        range_shard_rows=attributes["range_shard_rows"],
        codec_id=attributes["codec_id"],
    )
    _compressors(settings.codec_id)
    return _BucketAttributes(
        level=level,
        bucket_id=bucket_id,
        tile_count=tile_count,
        point_count=point_count,
        range_count=range_count,
        settings=settings,
    )
