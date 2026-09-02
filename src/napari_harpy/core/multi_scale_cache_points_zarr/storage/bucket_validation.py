from __future__ import annotations

from pathlib import Path

import numpy as np
import zarr
from zarr.codecs import BytesCodec, Crc32cCodec, ShardingCodec
from zarr.storage import LocalStore

from napari_harpy.core.multi_scale_cache_points_zarr.models import (
    _INT16_MAX,
    _UINT32_MAX,
    _bucket_path,
    _require_integer_in_range,
    _TileDescriptor,
)
from napari_harpy.core.multi_scale_cache_points_zarr.storage._schema import (
    _BYTE_ENDIAN,
    _CHUNK_KEY_ENCODING_NAME,
    _CHUNK_KEY_SEPARATOR,
    _POINT_DTYPES,
    _RANGE_DTYPES,
    _TILE_DTYPES,
    TILE_MAJOR_BUCKET_ARRAY_PATHS,
    TILE_MAJOR_BUCKET_GROUP_PATHS,
    TILE_MAJOR_LOCATION,
    TILE_MAJOR_POINT_ID,
    TILE_MAJOR_RANGE_ROW_COUNT,
    TILE_MAJOR_RANGE_ROW_START,
    TILE_MAJOR_RANGE_TILE_INDPTR,
    TILE_MAJOR_RANGE_VALUE_ID,
    TILE_MAJOR_RANGES_GROUP,
    TILE_MAJOR_TILE_OFFSET,
    TILE_MAJOR_TILE_X,
    TILE_MAJOR_TILE_Y,
    TILE_MAJOR_VALUE_ID,
    ZARR_FORMAT_VERSION,
    ZARR_READ_MISSING_CHUNKS,
    ZARR_USE_CONSOLIDATED,
    _BucketAttributes,
    _compressors,
    _parse_root_attributes,
)
from napari_harpy.core.multi_scale_cache_points_zarr.storage.models import _BucketWriteResult

_EXPECTED_NODES = {
    **dict.fromkeys(TILE_MAJOR_BUCKET_ARRAY_PATHS, zarr.Array),
    **dict.fromkeys(TILE_MAJOR_BUCKET_GROUP_PATHS, zarr.Group),
}


def _validate_bucket(
    cache_root: str | Path,
    *,
    level: int,
    bucket_id: int,
) -> _BucketWriteResult:
    """Reopen, independently validate, and describe one finalized Zarr bucket."""
    _require_integer_in_range(level, "level", maximum=_INT16_MAX)
    _require_integer_in_range(bucket_id, "bucket_id", maximum=_UINT32_MAX)
    target = Path(cache_root) / _bucket_path(level=level, bucket_id=bucket_id)
    if not target.exists():
        raise FileNotFoundError(f"Zarr bucket does not exist: {target}")

    store = LocalStore(target, read_only=True)
    try:
        root = zarr.open_group(
            store=store,
            mode="r",
            zarr_format=ZARR_FORMAT_VERSION,
            use_consolidated=ZARR_USE_CONSOLIDATED,
        )
        if root.metadata.zarr_format != ZARR_FORMAT_VERSION:
            raise ValueError("Bucket is not a Zarr v3 group.")
        _validate_hierarchy(root)
        attributes = _parse_root_attributes(
            dict(root.attrs),
            expected_level=level,
            expected_bucket_id=bucket_id,
        )
        arrays = {
            name: _strict_array(root, name) for name, node_type in _EXPECTED_NODES.items() if node_type is zarr.Array
        }
        _validate_array_layouts(arrays, attributes)
        return _validate_logical_contents(arrays, attributes)
    finally:
        store.close()


def _validate_hierarchy(root: zarr.Group) -> None:
    members = dict(root.members(max_depth=None, use_consolidated_for_children=False))
    if set(members) != set(_EXPECTED_NODES):
        raise ValueError("Zarr bucket hierarchy contains missing or unexpected logical nodes.")
    for name, expected_type in _EXPECTED_NODES.items():
        if not isinstance(members[name], expected_type):
            raise ValueError(f"Zarr bucket node has the wrong type: {name}.")
    if dict(root[TILE_MAJOR_RANGES_GROUP].attrs):
        raise ValueError("The ranges group must not contain attributes.")
    for name, node_type in _EXPECTED_NODES.items():
        if node_type is zarr.Array and dict(root[name].attrs):
            raise ValueError(f"Bucket arrays must not contain attributes: {name}.")


def _validate_array_layouts(
    arrays: dict[str, zarr.Array],
    attributes: _BucketAttributes,
) -> None:
    point_chunk_rows, point_shard_rows = _sharded_row_layout(
        arrays[TILE_MAJOR_VALUE_ID],
        name=TILE_MAJOR_VALUE_ID,
    )
    point_layouts = {
        TILE_MAJOR_LOCATION: (
            (attributes.point_count, 2),
            (point_chunk_rows, 2),
            (point_shard_rows, 2),
        ),
        TILE_MAJOR_POINT_ID: (
            (attributes.point_count,),
            (point_chunk_rows,),
            (point_shard_rows,),
        ),
        TILE_MAJOR_VALUE_ID: (
            (attributes.point_count,),
            (point_chunk_rows,),
            (point_shard_rows,),
        ),
    }
    for name, (shape, chunks, shards) in point_layouts.items():
        _validate_array_layout(
            arrays[name],
            name=name,
            dtype=_POINT_DTYPES[name],
            shape=shape,
            chunks=chunks,
            shards=shards,
            codec_id=attributes.codec_id,
        )

    tile_layouts = {
        TILE_MAJOR_TILE_X: ((attributes.tile_count,), (attributes.tile_count,)),
        TILE_MAJOR_TILE_Y: ((attributes.tile_count,), (attributes.tile_count,)),
        TILE_MAJOR_TILE_OFFSET: ((attributes.tile_count + 1,), (attributes.tile_count + 1,)),
    }
    for name, (shape, chunks) in tile_layouts.items():
        _validate_array_layout(
            arrays[name],
            name=name,
            dtype=_TILE_DTYPES[name],
            shape=shape,
            chunks=chunks,
            shards=None,
            codec_id=attributes.codec_id,
        )

    _validate_array_layout(
        arrays[TILE_MAJOR_RANGE_TILE_INDPTR],
        name=TILE_MAJOR_RANGE_TILE_INDPTR,
        dtype=_RANGE_DTYPES[TILE_MAJOR_RANGE_TILE_INDPTR],
        shape=(attributes.tile_count + 1,),
        chunks=(attributes.tile_count + 1,),
        shards=None,
        codec_id=attributes.codec_id,
    )
    range_chunk_rows, range_shard_rows = _sharded_row_layout(
        arrays[TILE_MAJOR_RANGE_VALUE_ID],
        name=TILE_MAJOR_RANGE_VALUE_ID,
    )
    for path in (
        TILE_MAJOR_RANGE_VALUE_ID,
        TILE_MAJOR_RANGE_ROW_START,
        TILE_MAJOR_RANGE_ROW_COUNT,
    ):
        _validate_array_layout(
            arrays[path],
            name=path,
            dtype=_RANGE_DTYPES[path],
            shape=(attributes.range_count,),
            chunks=(range_chunk_rows,),
            shards=(range_shard_rows,),
            codec_id=attributes.codec_id,
        )


def _sharded_row_layout(array: zarr.Array, *, name: str) -> tuple[int, int]:
    """Return a one-dimensional array's authoritative inner and outer row sizes."""
    shards = array.shards
    if len(array.chunks) != 1 or shards is None or len(shards) != 1:
        raise ValueError(f"Zarr bucket array must use a one-dimensional sharded row layout: {name}.")
    chunk_rows = array.chunks[0]
    shard_rows = shards[0]
    if shard_rows % chunk_rows:
        raise ValueError(f"Zarr bucket shard rows must be an integer multiple of chunk rows: {name}.")
    return chunk_rows, shard_rows


def _validate_array_layout(
    array: zarr.Array,
    *,
    name: str,
    dtype: np.dtype,
    shape: tuple[int, ...],
    chunks: tuple[int, ...],
    shards: tuple[int, ...] | None,
    codec_id: str,
) -> None:
    if np.dtype(array.dtype) != dtype:
        raise ValueError(f"Zarr bucket array has the wrong dtype: {name}.")
    if array.shape != shape or array.chunks != chunks or array.shards != shards:
        raise ValueError(f"Zarr bucket array has the wrong shape, chunks, or shards: {name}.")
    if bool(np.asarray(array.fill_value) != 0):
        raise ValueError(f"Zarr bucket array has the wrong fill value: {name}.")
    chunk_key = array.metadata.chunk_key_encoding
    if (
        getattr(chunk_key, "name", None) != _CHUNK_KEY_ENCODING_NAME
        or getattr(chunk_key, "separator", None) != _CHUNK_KEY_SEPARATOR
    ):
        raise ValueError(f"Zarr bucket array has the wrong chunk-key encoding: {name}.")

    compressors = _compressors(codec_id)
    inner_codecs = (BytesCodec(endian=_BYTE_ENDIAN), *compressors)
    if shards is None:
        # An unsharded array stores each logical chunk directly, so its
        # serializer and compressor are the complete top-level codec pipeline.
        expected_codecs = inner_codecs
    else:
        # With ``chunks=`` and ``shards=`` Zarr wraps those inner-chunk codecs
        # in one ShardingCodec. Its trailing index maps every inner chunk to a
        # byte offset and length; CRC32C detects corruption of that index.
        expected_codecs = (
            ShardingCodec(
                chunk_shape=chunks,
                codecs=inner_codecs,
                index_codecs=(BytesCodec(endian=_BYTE_ENDIAN), Crc32cCodec()),
                index_location="end",
            ),
        )
    if array.metadata.codecs != expected_codecs:
        raise ValueError(f"Zarr bucket array has the wrong codec pipeline: {name}.")


def _validate_logical_contents(
    arrays: dict[str, zarr.Array],
    attributes: _BucketAttributes,
) -> _BucketWriteResult:
    tile_x = np.asarray(arrays[TILE_MAJOR_TILE_X][:], dtype=np.uint32)
    tile_y = np.asarray(arrays[TILE_MAJOR_TILE_Y][:], dtype=np.uint32)
    tile_offset = np.asarray(arrays[TILE_MAJOR_TILE_OFFSET][:], dtype=np.uint64)
    tile_indptr = np.asarray(arrays[TILE_MAJOR_RANGE_TILE_INDPTR][:], dtype=np.uint64)

    coordinates = tuple(zip(tile_x.tolist(), tile_y.tolist(), strict=True))
    if coordinates != tuple(sorted(coordinates, key=lambda pair: (pair[1], pair[0]))):
        raise ValueError("Bucket tile coordinates are not ordered by (tile_y, tile_x).")
    if len(set(coordinates)) != attributes.tile_count:
        raise ValueError("Bucket tile coordinates are not unique.")
    _validate_pointer_array(
        tile_offset,
        expected_terminal=attributes.point_count,
        name="tile_offset",
    )
    _validate_pointer_array(
        tile_indptr,
        expected_terminal=attributes.range_count,
        name="ranges/tile_indptr",
    )

    descriptors: list[_TileDescriptor] = []
    for tile_index, (tile_coordinate_x, tile_coordinate_y) in enumerate(coordinates):
        point_start = int(tile_offset[tile_index])
        point_stop = int(tile_offset[tile_index + 1])
        range_start = int(tile_indptr[tile_index])
        range_stop = int(tile_indptr[tile_index + 1])
        location = np.asarray(arrays[TILE_MAJOR_LOCATION][point_start:point_stop, :], dtype=np.float32)
        value_id = np.asarray(arrays[TILE_MAJOR_VALUE_ID][point_start:point_stop], dtype=np.uint32)
        point_id = np.asarray(arrays[TILE_MAJOR_POINT_ID][point_start:point_stop], dtype=np.uint64)
        _validate_point_rows(location, value_id, point_id)

        range_values = np.asarray(
            arrays[TILE_MAJOR_RANGE_VALUE_ID][range_start:range_stop],
            dtype=np.uint32,
        )
        range_starts = np.asarray(
            arrays[TILE_MAJOR_RANGE_ROW_START][range_start:range_stop],
            dtype=np.uint64,
        )
        range_counts = np.asarray(
            arrays[TILE_MAJOR_RANGE_ROW_COUNT][range_start:range_stop],
            dtype=np.uint64,
        )
        _validate_tile_ranges(
            point_start=point_start,
            point_stop=point_stop,
            point_values=value_id,
            range_values=range_values,
            range_starts=range_starts,
            range_counts=range_counts,
        )
        descriptors.append(
            _TileDescriptor(
                level=attributes.level,
                bucket_id=attributes.bucket_id,
                bucket_tile_index=tile_index,
                tile_x=tile_coordinate_x,
                tile_y=tile_coordinate_y,
                n_points=point_stop - point_start,
            )
        )

    return _BucketWriteResult(
        tile_descriptors=tuple(descriptors),
        point_count=attributes.point_count,
        range_count=attributes.range_count,
    )


def _validate_pointer_array(
    values: np.ndarray,
    *,
    expected_terminal: int,
    name: str,
) -> None:
    if int(values[0]) != 0 or int(values[-1]) != expected_terminal:
        raise ValueError(f"Bucket pointer terminal is invalid: {name}.")
    if bool((values[1:] <= values[:-1]).any()):
        raise ValueError(f"Bucket pointers must be strictly increasing: {name}.")


def _validate_point_rows(
    location: np.ndarray,
    value_id: np.ndarray,
    point_id: np.ndarray,
) -> None:
    if location.shape != (len(value_id), 2) or point_id.shape != value_id.shape:
        raise ValueError("Tile point-array slices are not aligned.")
    if not bool(np.isfinite(location).all()) or bool((location < 0).any()):
        raise ValueError("Tile-relative coordinates must be finite and nonnegative.")
    if bool((value_id[1:] < value_id[:-1]).any()):
        raise ValueError("Tile values are not ordered.")
    same_value = value_id[1:] == value_id[:-1]
    if bool((point_id[1:][same_value] < point_id[:-1][same_value]).any()):
        raise ValueError("Tile point IDs are not ordered within value.")


def _validate_tile_ranges(
    *,
    point_start: int,
    point_stop: int,
    point_values: np.ndarray,
    range_values: np.ndarray,
    range_starts: np.ndarray,
    range_counts: np.ndarray,
) -> None:
    if not (len(range_values) == len(range_starts) == len(range_counts) >= 1):
        raise ValueError("Tile sparse-range arrays are not aligned and nonempty.")
    if bool((range_values[1:] <= range_values[:-1]).any()):
        raise ValueError("Tile sparse-range values must be strictly increasing.")
    if bool((range_counts == 0).any()):
        raise ValueError("Tile sparse ranges must have positive counts.")
    range_ends = range_starts + range_counts
    if int(range_starts[0]) != point_start or int(range_ends[-1]) != point_stop:
        raise ValueError("Tile sparse ranges do not cover the point interval.")
    if len(range_starts) > 1 and bool((range_starts[1:] != range_ends[:-1]).any()):
        raise ValueError("Tile sparse ranges contain a gap or overlap.")
    for value, start, stop in zip(range_values, range_starts, range_ends, strict=True):
        local_start = int(start) - point_start
        local_stop = int(stop) - point_start
        if not bool((point_values[local_start:local_stop] == value).all()):
            raise ValueError("Tile sparse range disagrees with point-level values.")


def _strict_array(root: zarr.Group, name: str) -> zarr.Array:
    node = root[name]
    if not isinstance(node, zarr.Array):
        raise ValueError(f"Required bucket node is not an array: {name}.")
    return node.with_config({"read_missing_chunks": ZARR_READ_MISSING_CHUNKS})
