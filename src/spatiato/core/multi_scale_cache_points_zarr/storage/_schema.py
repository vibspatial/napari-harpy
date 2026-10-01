"""Define canonical hierarchy, array, and bucket-payload storage contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
from zarr.codecs import BytesCodec, ZstdCodec

from napari_harpy.core.multi_scale_cache_points_zarr.models import (
    _INT16_MAX,
    _INT64_MAX,
    _UINT32_MAX,
    _require_integer_in_range,
)
from napari_harpy.core.multi_scale_cache_points_zarr.storage._paths import (
    MANIFEST_GROUP,
    VALUE_TILES_GROUP,
    VALUES_GROUP,
    value_major_level_path,
)

# Shared backend and encoding contract.
_PAYLOAD_SCHEMA_VERSION: Final = 1
_TILE_MAJOR_ROW_ORDER: Final = ("tile_y", "tile_x", "value_id", "point_id")
_COORDINATE_ENCODING: Final = "tile-relative-xy-float32-v1"
_ZSTD_CODEC_ID: Final = "zstd-v1"
ZARR_FORMAT_VERSION: Final = 3
ZARR_USE_CONSOLIDATED: Final = False
ZARR_READ_MISSING_CHUNKS: Final = False
_BYTE_ENDIAN: Final = "little"
_CHUNK_KEY_ENCODING_NAME: Final = "default"
_CHUNK_KEY_SEPARATOR: Final = "/"

# Persisted catalog and sidecar ordering contracts.
_MANIFEST_ROW_ORDER: Final = ("level", "tile_y", "tile_x")
_VALUE_TILE_ROW_ORDER: Final = ("level", "value_id", "manifest_index")
VALUE_MAJOR_ROW_ORDER: Final = ("value_id", "manifest_index", "point_id")

# Canonical cache-relative catalog array paths.
VALUES_N_POINTS: Final = f"{VALUES_GROUP}/n_points"
MANIFEST_LEVEL_INDPTR: Final = f"{MANIFEST_GROUP}/level_indptr"
MANIFEST_BUCKET_ID: Final = f"{MANIFEST_GROUP}/bucket_id"
MANIFEST_BUCKET_TILE_INDEX: Final = f"{MANIFEST_GROUP}/bucket_tile_index"
MANIFEST_TILE_X: Final = f"{MANIFEST_GROUP}/tile_x"
MANIFEST_TILE_Y: Final = f"{MANIFEST_GROUP}/tile_y"
MANIFEST_N_POINTS: Final = f"{MANIFEST_GROUP}/n_points"
VALUE_TILES_INDPTR: Final = f"{VALUE_TILES_GROUP}/indptr"
VALUE_TILES_MANIFEST_INDEX: Final = f"{VALUE_TILES_GROUP}/manifest_index"
VALUE_TILES_N_POINTS: Final = f"{VALUE_TILES_GROUP}/n_points"

# Canonical paths inside each standalone tile-major bucket store.
TILE_MAJOR_LOCATION: Final = "location"
TILE_MAJOR_POINT_ID: Final = "point_id"
TILE_MAJOR_VALUE_ID: Final = "value_id"
TILE_MAJOR_TILE_X: Final = "tile_x"
TILE_MAJOR_TILE_Y: Final = "tile_y"
TILE_MAJOR_TILE_OFFSET: Final = "tile_offset"
TILE_MAJOR_RANGES_GROUP: Final = "ranges"
TILE_MAJOR_RANGE_TILE_INDPTR: Final = f"{TILE_MAJOR_RANGES_GROUP}/tile_indptr"
TILE_MAJOR_RANGE_VALUE_ID: Final = f"{TILE_MAJOR_RANGES_GROUP}/value_id"
TILE_MAJOR_RANGE_ROW_START: Final = f"{TILE_MAJOR_RANGES_GROUP}/row_start"
TILE_MAJOR_RANGE_ROW_COUNT: Final = f"{TILE_MAJOR_RANGES_GROUP}/row_count"

# Canonical array names inside every value-major level group.
VALUE_MAJOR_LOCATION_ARRAY: Final = "location"
VALUE_MAJOR_POINT_INDPTR_ARRAY: Final = "value_point_indptr"
VALUE_MAJOR_LEVEL_ARRAYS: Final = frozenset(
    {
        VALUE_MAJOR_LOCATION_ARRAY,
        VALUE_MAJOR_POINT_INDPTR_ARRAY,
    }
)

# Cache-wide catalog and value-major array dtypes.
CATALOG_ARRAY_DTYPES: Final = {
    VALUES_N_POINTS: np.dtype(np.uint64),
    MANIFEST_LEVEL_INDPTR: np.dtype(np.uint64),
    MANIFEST_BUCKET_ID: np.dtype(np.uint32),
    MANIFEST_BUCKET_TILE_INDEX: np.dtype(np.uint32),
    MANIFEST_TILE_X: np.dtype(np.uint32),
    MANIFEST_TILE_Y: np.dtype(np.uint32),
    MANIFEST_N_POINTS: np.dtype(np.uint64),
    VALUE_TILES_INDPTR: np.dtype(np.uint64),
    VALUE_TILES_MANIFEST_INDEX: np.dtype(np.uint64),
    VALUE_TILES_N_POINTS: np.dtype(np.uint64),
}
CATALOG_ARRAY_PATHS: Final = tuple(CATALOG_ARRAY_DTYPES)
CATALOG_GROUP_ARRAYS: Final = {
    group: frozenset(path.removeprefix(f"{group}/") for path in CATALOG_ARRAY_PATHS if path.startswith(f"{group}/"))
    for group in (VALUES_GROUP, MANIFEST_GROUP, VALUE_TILES_GROUP)
}
VALUE_MAJOR_LOCATION_DTYPE: Final = np.dtype(np.float32)
VALUE_MAJOR_POINTER_DTYPE: Final = np.dtype(np.uint64)

# Standalone tile-major bucket root-attribute contract.
_TILE_MAJOR_BUCKET_ATTRIBUTE_KEYS: Final = frozenset(
    {
        "payload_schema_version",
        "level",
        "bucket_id",
        "tile_count",
        "point_count",
        "range_count",
        "point_row_order",
        "coordinate_encoding",
        "codec_id",
    }
)

# Standalone tile-major bucket array dtypes.
_POINT_DTYPES: Final = {
    TILE_MAJOR_LOCATION: np.dtype(np.float32),
    TILE_MAJOR_POINT_ID: np.dtype(np.uint64),
    TILE_MAJOR_VALUE_ID: np.dtype(np.uint32),
}
_TILE_DTYPES: Final = {
    TILE_MAJOR_TILE_X: np.dtype(np.uint32),
    TILE_MAJOR_TILE_Y: np.dtype(np.uint32),
    TILE_MAJOR_TILE_OFFSET: np.dtype(np.uint64),
}
_RANGE_DTYPES: Final = {
    TILE_MAJOR_RANGE_TILE_INDPTR: np.dtype(np.uint64),
    TILE_MAJOR_RANGE_VALUE_ID: np.dtype(np.uint32),
    TILE_MAJOR_RANGE_ROW_START: np.dtype(np.uint64),
    TILE_MAJOR_RANGE_ROW_COUNT: np.dtype(np.uint64),
}
TILE_MAJOR_BUCKET_ARRAY_PATHS: Final = (*_POINT_DTYPES, *_TILE_DTYPES, *_RANGE_DTYPES)
TILE_MAJOR_BUCKET_GROUP_PATHS: Final = frozenset({TILE_MAJOR_RANGES_GROUP})


def value_major_level_group(level: int) -> str:
    """Return the canonical cache-relative group for one value-major level."""
    _require_integer_in_range(level, "level", maximum=_INT16_MAX)
    return value_major_level_path(level)


def value_major_location(level: int) -> str:
    """Return the canonical cache-relative coordinate-array path for a level."""
    return f"{value_major_level_group(level)}/{VALUE_MAJOR_LOCATION_ARRAY}"


def value_major_point_indptr(level: int) -> str:
    """Return the canonical cache-relative value-pointer-array path for a level."""
    return f"{value_major_level_group(level)}/{VALUE_MAJOR_POINT_INDPTR_ARRAY}"


@dataclass(frozen=True)
class _BucketAttributes:
    level: int
    bucket_id: int
    tile_count: int
    point_count: int
    range_count: int
    codec_id: str

    def to_dict(self) -> dict[str, object]:
        """Return the exact standalone bucket root-attribute payload."""
        return {
            "payload_schema_version": _PAYLOAD_SCHEMA_VERSION,
            "level": self.level,
            "bucket_id": self.bucket_id,
            "tile_count": self.tile_count,
            "point_count": self.point_count,
            "range_count": self.range_count,
            "point_row_order": list(_TILE_MAJOR_ROW_ORDER),
            "coordinate_encoding": _COORDINATE_ENCODING,
            "codec_id": self.codec_id,
        }


def _compressors(codec_id: str) -> tuple[ZstdCodec]:
    """Return the exact inner-chunk compressor for a supported codec ID."""
    if codec_id != _ZSTD_CODEC_ID:
        raise ValueError(f"Unsupported Zarr bucket codec ID: {codec_id!r}.")
    return (ZstdCodec(level=3, checksum=True),)


def _array_creation_options(codec_id: str) -> dict[str, object]:
    """Return a fresh Zarr-v3 array profile shared by all cache writers."""
    return {
        "compressors": _compressors(codec_id),
        "serializer": BytesCodec(endian=_BYTE_ENDIAN),
        "fill_value": 0,
        "chunk_key_encoding": {
            "name": _CHUNK_KEY_ENCODING_NAME,
            "configuration": {"separator": _CHUNK_KEY_SEPARATOR},
        },
        "config": {"write_empty_chunks": True},
    }


def _parse_root_attributes(
    attributes: Mapping[str, Any],
    *,
    expected_level: int,
    expected_bucket_id: int,
) -> _BucketAttributes:
    """Validate exact schema-v1 root attributes and return typed physical facts."""
    if set(attributes) != _TILE_MAJOR_BUCKET_ATTRIBUTE_KEYS:
        raise ValueError("Zarr bucket root attributes do not match payload schema version 1.")
    if type(attributes["payload_schema_version"]) is not int or (
        attributes["payload_schema_version"] != _PAYLOAD_SCHEMA_VERSION
    ):
        raise ValueError("Unsupported Zarr bucket payload schema version.")
    if attributes["point_row_order"] != list(_TILE_MAJOR_ROW_ORDER):
        raise ValueError("Unsupported Zarr bucket tile-major row ordering.")
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

    codec_id = attributes["codec_id"]
    if not isinstance(codec_id, str) or codec_id == "":
        raise ValueError("Zarr bucket codec ID must be a nonempty string.")
    _compressors(codec_id)
    return _BucketAttributes(
        level=level,
        bucket_id=bucket_id,
        tile_count=tile_count,
        point_count=point_count,
        range_count=range_count,
        codec_id=codec_id,
    )
