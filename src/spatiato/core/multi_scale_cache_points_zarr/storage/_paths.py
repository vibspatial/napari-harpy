"""Define dependency-neutral cache hierarchy names and path builders."""

from __future__ import annotations

from typing import Final

# Cache-root group names.
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

# Canonical names used below the physical row-order groups.
LEVEL_NAME_PREFIX: Final = "level_"
TILE_MAJOR_BUCKET_NAME_PREFIX: Final = "bucket-"
TILE_MAJOR_BUCKET_NAME_SUFFIX: Final = ".zarr"
TILE_MAJOR_BUCKET_GLOB: Final = f"{TILE_MAJOR_BUCKET_NAME_PREFIX}*{TILE_MAJOR_BUCKET_NAME_SUFFIX}"


def level_name(level: int) -> str:
    """Return the serialized child-group name for a validated level."""
    return f"{LEVEL_NAME_PREFIX}{level}"


def tile_major_level_path(level: int) -> str:
    """Return the cache-relative tile-major group for a validated level."""
    return f"{TILE_MAJOR_GROUP}/{level_name(level)}"


def tile_major_bucket_name(bucket_id: int) -> str:
    """Return the serialized store name for a validated bucket ID."""
    return f"{TILE_MAJOR_BUCKET_NAME_PREFIX}{bucket_id:03d}{TILE_MAJOR_BUCKET_NAME_SUFFIX}"


def tile_major_bucket_path(*, level: int, bucket_id: int) -> str:
    """Return the cache-relative store path for validated bucket identity."""
    return f"{tile_major_level_path(level)}/{tile_major_bucket_name(bucket_id)}"


def value_major_level_path(level: int) -> str:
    """Return the cache-relative value-major group for a validated level."""
    return f"{VALUE_MAJOR_GROUP}/{level_name(level)}"
