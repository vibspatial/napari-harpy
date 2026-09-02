"""Define canonical hierarchy, array, and bucket-payload storage contracts."""

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

# Shared backend and encoding contract.
_PAYLOAD_SCHEMA_VERSION: Final = 1
_TILE_MAJOR_ROW_ORDER: Final = ("tile_y", "tile_x", "value_id", "point_id")
_COORDINATE_ENCODING: Final = "tile-relative-xy-float32-v1"
_ZSTD_CODEC_ID: Final = "zstd-v1"
_CHUNK_KEY_ENCODING: Final = {
    "name": "default",
    "configuration": {"separator": "/"},
}

# Cache-wide group names.
VALUES_GROUP: Final = "values"
MANIFEST_GROUP: Final = "manifest"
VALUE_TILES_GROUP: Final = "value_tiles"
TILE_MAJOR_GROUP: Final = "tile_major"
VALUE_MAJOR_GROUP: Final = "value_major"
ZARR_METADATA_FILENAME: Final = "zarr.json"
CACHE_ROOT_GROUPS: Final = frozenset(
    {
        VALUES_GROUP,
        MANIFEST_GROUP,
        VALUE_TILES_GROUP,
        TILE_MAJOR_GROUP,
        VALUE_MAJOR_GROUP,
    }
)

# Persisted catalog and sidecar ordering contracts.
_MANIFEST_ROW_ORDER: Final = ("level", "tile_y", "tile_x")
_VALUE_TILE_ROW_ORDER: Final = ("level", "value_id", "manifest_index")
VALUE_MAJOR_ROW_ORDER: Final = ("value_id", "manifest_index", "point_id")

# Canonical cache-relative catalog array paths.
VALUES_N_POINTS: Final = "values/n_points"
MANIFEST_LEVEL_INDPTR: Final = "manifest/level_indptr"
MANIFEST_BUCKET_ID: Final = "manifest/bucket_id"
MANIFEST_BUCKET_TILE_INDEX: Final = "manifest/bucket_tile_index"
MANIFEST_TILE_X: Final = "manifest/tile_x"
MANIFEST_TILE_Y: Final = "manifest/tile_y"
MANIFEST_N_POINTS: Final = "manifest/n_points"
VALUE_TILES_INDPTR: Final = "value_tiles/indptr"
VALUE_TILES_MANIFEST_INDEX: Final = "value_tiles/manifest_index"
VALUE_TILES_N_POINTS: Final = "value_tiles/n_points"

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
VALUE_MAJOR_LOCATION_DTYPE: Final = np.dtype(np.float32)
VALUE_MAJOR_POINTER_DTYPE: Final = np.dtype(np.uint64)

# Standalone tile-major bucket root-attribute contract.
_ROOT_ATTRIBUTE_KEYS: Final = frozenset(
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


def value_major_level_group(level: int) -> str:
    """Return the canonical cache-relative group for one value-major level."""
    _require_integer_in_range(level, "level", maximum=_INT16_MAX)
    return f"{VALUE_MAJOR_GROUP}/level_{level}"


def value_major_location(level: int) -> str:
    """Return the canonical cache-relative coordinate-array path for a level."""
    return f"{value_major_level_group(level)}/location"


def value_major_point_indptr(level: int) -> str:
    """Return the canonical cache-relative value-pointer-array path for a level."""
    return f"{value_major_level_group(level)}/value_point_indptr"


@dataclass(frozen=True)
class _BucketAttributes:
    level: int
    bucket_id: int
    tile_count: int
    point_count: int
    range_count: int
    codec_id: str


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
