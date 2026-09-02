"""Define the multiscale points Zarr cache contract.

See ``CACHE_FORMAT.md`` beside this module for a self-contained worked example
of the complete hierarchy and the relationships between its indexes.
"""

from __future__ import annotations

import copy
import math
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Final

from napari_harpy.core.multi_scale_cache_points_zarr.hashing import BUCKET_HASH_METHOD
from napari_harpy.core.multi_scale_cache_points_zarr.models import (
    _INT16_MAX,
    _INT64_MAX,
    _UINT32_MAX,
    _require_integer_in_range,
)
from napari_harpy.core.multi_scale_cache_points_zarr.sampling import (
    SAMPLED_TILE_MICROGRID_EDGE,
    SAMPLING_METHOD,
    SAMPLING_SEED,
)
from napari_harpy.core.multi_scale_cache_points_zarr.source.signature import POINT_ID_POLICY, SOURCE_SIGNATURE_METHOD
from napari_harpy.core.multi_scale_cache_points_zarr.source.value_normalization import VALUE_NORMALIZATION_METHOD
from napari_harpy.core.multi_scale_cache_points_zarr.storage._schema import (
    _COORDINATE_ENCODING,
    _MANIFEST_ROW_ORDER,
    _PAYLOAD_SCHEMA_VERSION,
    _TILE_MAJOR_ROW_ORDER,
    _VALUE_TILE_ROW_ORDER,
    _ZSTD_CODEC_ID,
    MANIFEST_GROUP,
    TILE_MAJOR_GROUP,
    VALUE_MAJOR_GROUP,
    VALUE_MAJOR_ROW_ORDER,
    VALUE_TILES_GROUP,
    VALUES_GROUP,
)
from napari_harpy.core.multi_scale_cache_points_zarr.storage.models import _ZarrWriteSettings

CACHE_SCHEMA_VERSION: Final = "harpy-multiscale-points-zarr-cache-0.2"
BACKEND_IDENTIFIER: Final = "harpy-zarr-v3-bucket-sparse-ranges-value-major-v2"
CREATED_BY_PACKAGE: Final = "napari-harpy"
PUBLICATION_STATE_STAGING: Final = "staging"
PUBLICATION_STATE_COMPLETE: Final = "complete"

_ROOT_ATTRIBUTE_KEYS: Final = frozenset(
    {
        "schema_version",
        "cache_generation_id",
        "publication_state",
        "created_by",
        "backend",
        "source",
        "geometry",
        "build",
        "levels",
        "value_names",
        "catalog",
        "value_major",
    }
)


@dataclass(frozen=True)
class _CatalogWriteSettings:
    """Configure physical catalog arrays without changing their logical schema."""

    manifest_chunk_rows: int = 65_536
    manifest_shard_rows: int = 262_144
    value_tile_chunk_rows: int = 65_536
    value_tile_shard_rows: int = 1_048_576

    def __post_init__(self) -> None:
        for name in (
            "manifest_chunk_rows",
            "manifest_shard_rows",
            "value_tile_chunk_rows",
            "value_tile_shard_rows",
        ):
            _require_integer_in_range(getattr(self, name), name, minimum=1, maximum=_INT64_MAX)
        if self.manifest_shard_rows % self.manifest_chunk_rows:
            raise ValueError("`manifest_shard_rows` must be a multiple of `manifest_chunk_rows`.")
        if self.value_tile_shard_rows % self.value_tile_chunk_rows:
            raise ValueError("`value_tile_shard_rows` must be a multiple of `value_tile_chunk_rows`.")


@dataclass(frozen=True)
class _ValueMajorWriteSettings:
    """Configure mandatory value-major storage and bounded construction.

    Parameters
    ----------
    point_chunk_rows
        Number of value-major point rows in one inner Zarr chunk. This layout
        applies to point-aligned arrays, currently ``location``, rather than
        pointer or index arrays.
    point_shard_rows
        Number of value-major point rows in one physical Zarr shard.
    construction_batch_points
        Maximum number of point rows copied into the sidecar in one bounded
        construction batch.
    """

    point_chunk_rows: int = 4_096
    point_shard_rows: int = 131_072
    construction_batch_points: int = 1_048_576

    def __post_init__(self) -> None:
        for name in (
            "point_chunk_rows",
            "point_shard_rows",
            "construction_batch_points",
        ):
            _require_integer_in_range(getattr(self, name), name, minimum=1, maximum=_INT64_MAX)
        if self.point_shard_rows % self.point_chunk_rows:
            raise ValueError("`point_shard_rows` must be a multiple of `point_chunk_rows`.")


@dataclass(frozen=True)
class _ValueMajorMetadata:
    """Describe the published physical profile of value-major point arrays.

    ``point_chunk_rows`` and ``point_shard_rows`` apply to every point-aligned
    sidecar array, currently ``location`` and potentially future arrays such as
    per-point quality values. Pointer and index arrays have independent
    layouts.
    """

    point_chunk_rows: int
    point_shard_rows: int

    def __post_init__(self) -> None:
        for name in ("point_chunk_rows", "point_shard_rows"):
            _require_integer_in_range(getattr(self, name), name, minimum=1, maximum=_INT64_MAX)
        if self.point_shard_rows % self.point_chunk_rows:
            raise ValueError("`point_shard_rows` must be a multiple of `point_chunk_rows`.")

    @classmethod
    def from_write_settings(cls, settings: _ValueMajorWriteSettings) -> _ValueMajorMetadata:
        """Project builder settings onto properties of the published cache."""
        if not isinstance(settings, _ValueMajorWriteSettings):
            raise ValueError("`settings` must be _ValueMajorWriteSettings.")
        return cls(
            point_chunk_rows=settings.point_chunk_rows,
            point_shard_rows=settings.point_shard_rows,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "group": VALUE_MAJOR_GROUP,
            "point_row_order": list(VALUE_MAJOR_ROW_ORDER),
            "point_chunk_rows": self.point_chunk_rows,
            "point_shard_rows": self.point_shard_rows,
        }


@dataclass(frozen=True)
class _SourceMetadata:
    points_name: str
    element_path: str
    row_count: int
    x_column: str
    y_column: str
    value_column: str
    selected_schema: tuple[dict[str, object], ...]
    signature_method: str
    signature: str
    value_normalization_method: str
    point_id_policy: str

    def __post_init__(self) -> None:
        _require_nonempty_string(self.points_name, "source.points_name")
        _require_relative_posix_path(self.element_path, "source.element_path")
        if self.element_path != f"points/{self.points_name}":
            raise ValueError("`source.element_path` does not match `source.points_name`.")
        _require_integer_in_range(self.row_count, "source.row_count", minimum=1, maximum=_INT64_MAX)
        for name, value in (
            ("source.columns.x", self.x_column),
            ("source.columns.y", self.y_column),
            ("source.columns.value", self.value_column),
            ("source.signature_method", self.signature_method),
            ("source.signature", self.signature),
            ("source.value_normalization_method", self.value_normalization_method),
            ("source.point_id_policy", self.point_id_policy),
        ):
            _require_nonempty_string(value, name)
        if len(self.signature) != 64 or any(character not in "0123456789abcdef" for character in self.signature):
            raise ValueError("`source.signature` must be a lowercase SHA-256 hexadecimal digest.")
        if (
            self.signature_method != SOURCE_SIGNATURE_METHOD
            or self.value_normalization_method != VALUE_NORMALIZATION_METHOD
            or self.point_id_policy != POINT_ID_POLICY
        ):
            raise ValueError("Cache source provenance methods are unsupported.")
        normalized = _normalize_selected_schema(self.selected_schema)
        object.__setattr__(self, "selected_schema", normalized)

    def to_dict(self) -> dict[str, object]:
        return {
            "points_name": self.points_name,
            "element_path": self.element_path,
            "row_count": self.row_count,
            "columns": {"x": self.x_column, "y": self.y_column, "value": self.value_column},
            "selected_schema": copy.deepcopy(list(self.selected_schema)),
            "signature_method": self.signature_method,
            "signature": self.signature,
            "value_normalization_method": self.value_normalization_method,
            "point_id_policy": self.point_id_policy,
        }


@dataclass(frozen=True)
class _GeometryMetadata:
    x_origin: float
    y_origin: float
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    def __post_init__(self) -> None:
        for name in ("x_origin", "y_origin", "x_min", "x_max", "y_min", "y_max"):
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value):
                raise ValueError(f"`geometry.{name}` must be a finite float.")
        if self.x_min > self.x_max or self.y_min > self.y_max:
            raise ValueError("Cache geometry bounds must not be inverted.")

    def to_dict(self) -> dict[str, object]:
        return {
            "x_origin": self.x_origin,
            "y_origin": self.y_origin,
            "x_min": self.x_min,
            "x_max": self.x_max,
            "y_min": self.y_min,
            "y_max": self.y_max,
            "coordinate_axes": ["x", "y"],
            "relative_coordinate_dtype": "float32",
        }


@dataclass(frozen=True)
class _BuildMetadata:
    leaf_tile_size: int
    overview_point_budget: int
    target_points_per_bucket: int
    bucket_hash_method: str
    sampling_method: str
    sampling_seed: int
    sampling_microgrid_edge: int

    def __post_init__(self) -> None:
        for name in (
            "leaf_tile_size",
            "overview_point_budget",
            "target_points_per_bucket",
            "sampling_microgrid_edge",
        ):
            _require_integer_in_range(getattr(self, name), f"build.{name}", minimum=1, maximum=_INT64_MAX)
        _require_integer_in_range(self.sampling_seed, "build.sampling_seed", maximum=_INT64_MAX)
        _require_nonempty_string(self.bucket_hash_method, "build.bucket_hash_method")
        _require_nonempty_string(self.sampling_method, "build.sampling_method")
        if (
            self.bucket_hash_method != BUCKET_HASH_METHOD
            or self.sampling_method != SAMPLING_METHOD
            or self.sampling_seed != SAMPLING_SEED
            or self.sampling_microgrid_edge != SAMPLED_TILE_MICROGRID_EDGE
        ):
            raise ValueError("Cache build method profile is unsupported.")

    def to_dict(self) -> dict[str, object]:
        return {
            "leaf_tile_size": self.leaf_tile_size,
            "overview_point_budget": self.overview_point_budget,
            "target_points_per_bucket": self.target_points_per_bucket,
            "bucket_hash_method": self.bucket_hash_method,
            "sampling_method": self.sampling_method,
            "sampling_seed": self.sampling_seed,
            "sampling_microgrid_edge": self.sampling_microgrid_edge,
        }


@dataclass(frozen=True)
class _LevelMetadata:
    level: int
    kind: str
    tile_size: int
    grid_width: int
    grid_height: int
    max_points_per_tile: int | None
    point_count_upper_bound: int
    bucket_count: int
    tile_count: int
    point_count: int
    range_count: int
    relative_directory: str

    def __post_init__(self) -> None:
        _require_integer_in_range(self.level, "levels.level", maximum=_INT16_MAX)
        if self.kind not in {"exact", "bridge", "spatial"}:
            raise ValueError("`levels.kind` is unsupported.")
        for name, minimum, maximum in (
            ("tile_size", 1, _INT64_MAX),
            ("grid_width", 1, _UINT32_MAX + 1),
            ("grid_height", 1, _UINT32_MAX + 1),
            ("point_count_upper_bound", 1, _INT64_MAX),
            ("bucket_count", 1, _UINT32_MAX + 1),
            ("tile_count", 1, _INT64_MAX),
            ("point_count", 1, _INT64_MAX),
            ("range_count", 1, _INT64_MAX),
        ):
            _require_integer_in_range(getattr(self, name), f"levels.{name}", minimum=minimum, maximum=maximum)
        if self.max_points_per_tile is None:
            if self.kind != "exact":
                raise ValueError("Only the Exact level may omit `max_points_per_tile`.")
        else:
            _require_integer_in_range(
                self.max_points_per_tile,
                "levels.max_points_per_tile",
                minimum=1,
                maximum=_INT64_MAX,
            )
            if self.kind == "exact":
                raise ValueError("The Exact level must omit `max_points_per_tile`.")
        if not self.tile_count <= self.range_count <= self.point_count <= self.point_count_upper_bound:
            raise ValueError("Level tile, range, point, and planned counts are inconsistent.")
        expected_directory = f"{TILE_MAJOR_GROUP}/level_{self.level}"
        if self.relative_directory != expected_directory:
            raise ValueError("Level relative directory is not canonical.")

    def to_dict(self) -> dict[str, object]:
        return {
            "level": self.level,
            "kind": self.kind,
            "tile_size": self.tile_size,
            "grid_width": self.grid_width,
            "grid_height": self.grid_height,
            "max_points_per_tile": self.max_points_per_tile,
            "point_count_upper_bound": self.point_count_upper_bound,
            "bucket_count": self.bucket_count,
            "tile_count": self.tile_count,
            "point_count": self.point_count,
            "range_count": self.range_count,
            "relative_directory": self.relative_directory,
        }


@dataclass(frozen=True)
class _CatalogMetadata:
    value_count: int
    level_count: int
    manifest_row_count: int
    value_tile_row_count: int
    settings: _CatalogWriteSettings

    def __post_init__(self) -> None:
        _require_integer_in_range(self.value_count, "catalog.value_count", minimum=1, maximum=_UINT32_MAX + 1)
        _require_integer_in_range(self.level_count, "catalog.level_count", minimum=1, maximum=_INT16_MAX + 1)
        _require_integer_in_range(
            self.manifest_row_count,
            "catalog.manifest_row_count",
            minimum=1,
            maximum=_INT64_MAX,
        )
        _require_integer_in_range(
            self.value_tile_row_count,
            "catalog.value_tile_row_count",
            minimum=1,
            maximum=_INT64_MAX,
        )
        if not isinstance(self.settings, _CatalogWriteSettings):
            raise ValueError("`catalog.settings` must be _CatalogWriteSettings.")

    def to_dict(self) -> dict[str, object]:
        return {
            "value_count": self.value_count,
            "level_count": self.level_count,
            "manifest_row_count": self.manifest_row_count,
            "value_tile_row_count": self.value_tile_row_count,
            "values_group": VALUES_GROUP,
            "manifest_group": MANIFEST_GROUP,
            "value_tiles_group": VALUE_TILES_GROUP,
            "manifest_row_order": list(_MANIFEST_ROW_ORDER),
            "value_tile_row_order": list(_VALUE_TILE_ROW_ORDER),
            "manifest_chunk_rows": self.settings.manifest_chunk_rows,
            "manifest_shard_rows": self.settings.manifest_shard_rows,
            "value_tile_chunk_rows": self.settings.value_tile_chunk_rows,
            "value_tile_shard_rows": self.settings.value_tile_shard_rows,
        }


@dataclass(frozen=True)
class _CacheAttributes:
    cache_generation_id: str
    publication_state: str
    created_by_version: str
    zarr_settings: _ZarrWriteSettings
    source: _SourceMetadata
    geometry: _GeometryMetadata
    build: _BuildMetadata
    levels: tuple[_LevelMetadata, ...]
    value_names: tuple[str, ...]
    catalog: _CatalogMetadata
    value_major: _ValueMajorMetadata

    def __post_init__(self) -> None:
        if not isinstance(self.cache_generation_id, str):
            raise ValueError("`cache_generation_id` must be a canonical UUID string.")
        try:
            parsed_uuid = uuid.UUID(self.cache_generation_id)
        except (ValueError, AttributeError) as error:
            raise ValueError("`cache_generation_id` must be a canonical UUID string.") from error
        if str(parsed_uuid) != self.cache_generation_id:
            raise ValueError("`cache_generation_id` must be a canonical lowercase UUID string.")
        if self.publication_state not in {PUBLICATION_STATE_STAGING, PUBLICATION_STATE_COMPLETE}:
            raise ValueError("`publication_state` must be 'staging' or 'complete'.")
        _require_nonempty_string(self.created_by_version, "created_by.version")
        if not isinstance(self.zarr_settings, _ZarrWriteSettings):
            raise ValueError("`zarr_settings` must be _ZarrWriteSettings.")
        if not isinstance(self.source, _SourceMetadata):
            raise ValueError("`source` must be _SourceMetadata.")
        if not isinstance(self.geometry, _GeometryMetadata):
            raise ValueError("`geometry` must be _GeometryMetadata.")
        if not isinstance(self.build, _BuildMetadata):
            raise ValueError("`build` must be _BuildMetadata.")
        if not isinstance(self.levels, tuple) or not self.levels:
            raise ValueError("`levels` must contain at least Exact level zero.")
        if not all(isinstance(level, _LevelMetadata) for level in self.levels):
            raise ValueError("Every cache level must be _LevelMetadata.")
        if tuple(level.level for level in self.levels) != tuple(range(len(self.levels))):
            raise ValueError("Cache levels must be consecutively numbered from zero.")
        if self.levels[0].kind != "exact" or any(level.kind != "spatial" for level in self.levels[2:]):
            raise ValueError("Cache level kinds do not follow Exact, optional Bridge, then Spatial.")
        if len(self.levels) > 1 and self.levels[1].kind != "bridge":
            raise ValueError("Cache level one must be Bridge.")
        if not isinstance(self.value_names, tuple) or not self.value_names:
            raise ValueError("`value_names` must be a nonempty tuple.")
        if any(not isinstance(value, str) or value == "" for value in self.value_names):
            raise ValueError("Every cache value name must be a nonempty string.")
        if len(set(self.value_names)) != len(self.value_names):
            raise ValueError("Cache value names must be unique.")
        if tuple(value.encode("utf-8") for value in self.value_names) != tuple(
            sorted(value.encode("utf-8") for value in self.value_names)
        ):
            raise ValueError("Cache value names must be ordered by UTF-8 bytes.")
        if not isinstance(self.catalog, _CatalogMetadata):
            raise ValueError("`catalog` must be _CatalogMetadata.")
        if self.catalog.value_count != len(self.value_names) or self.catalog.level_count != len(self.levels):
            raise ValueError("Catalog dimensions do not match cache values and levels.")
        if self.catalog.manifest_row_count != sum(level.tile_count for level in self.levels):
            raise ValueError("Catalog manifest rows do not match level tile totals.")
        if self.catalog.value_tile_row_count != sum(level.range_count for level in self.levels):
            raise ValueError("Catalog value-tile rows do not match level range totals.")
        if self.source.row_count != self.levels[0].point_count:
            raise ValueError("Source rows do not match Exact points.")
        if not isinstance(self.value_major, _ValueMajorMetadata):
            raise ValueError("`value_major` must be _ValueMajorMetadata.")

    def to_dict(self) -> dict[str, object]:
        settings = self.zarr_settings
        return {
            "schema_version": CACHE_SCHEMA_VERSION,
            "cache_generation_id": self.cache_generation_id,
            "publication_state": self.publication_state,
            "created_by": {"package": CREATED_BY_PACKAGE, "version": self.created_by_version},
            "backend": {
                "identifier": BACKEND_IDENTIFIER,
                "zarr_format": 3,
                "payload_schema_version": _PAYLOAD_SCHEMA_VERSION,
                "point_row_order": list(_TILE_MAJOR_ROW_ORDER),
                "coordinate_encoding": _COORDINATE_ENCODING,
                "codec_id": settings.codec_id,
                "point_chunk_rows": settings.point_chunk_rows,
                "point_shard_rows": settings.point_shard_rows,
                "range_chunk_rows": settings.range_chunk_rows,
                "range_shard_rows": settings.range_shard_rows,
            },
            "source": self.source.to_dict(),
            "geometry": self.geometry.to_dict(),
            "build": self.build.to_dict(),
            "levels": [level.to_dict() for level in self.levels],
            "value_names": list(self.value_names),
            "catalog": self.catalog.to_dict(),
            "value_major": self.value_major.to_dict(),
        }


def _parse_cache_attributes(attributes: Mapping[str, Any]) -> _CacheAttributes:
    """Parse exact cache-v0.2 root attributes into immutable typed metadata."""
    root = _require_mapping(attributes, "root attributes", keys=_ROOT_ATTRIBUTE_KEYS)
    if root["schema_version"] != CACHE_SCHEMA_VERSION:
        raise ValueError("Unsupported Zarr cache schema version.")

    created_by = _require_mapping(root["created_by"], "created_by", keys={"package", "version"})
    if created_by["package"] != CREATED_BY_PACKAGE:
        raise ValueError("Unsupported cache creator package.")

    backend = _require_mapping(
        root["backend"],
        "backend",
        keys={
            "identifier",
            "zarr_format",
            "payload_schema_version",
            "point_row_order",
            "coordinate_encoding",
            "codec_id",
            "point_chunk_rows",
            "point_shard_rows",
            "range_chunk_rows",
            "range_shard_rows",
        },
    )
    if (
        backend["identifier"] != BACKEND_IDENTIFIER
        or type(backend["zarr_format"]) is not int
        or backend["zarr_format"] != 3
        or type(backend["payload_schema_version"]) is not int
        or backend["payload_schema_version"] != _PAYLOAD_SCHEMA_VERSION
        or backend["point_row_order"] != list(_TILE_MAJOR_ROW_ORDER)
        or backend["coordinate_encoding"] != _COORDINATE_ENCODING
        or backend["codec_id"] != _ZSTD_CODEC_ID
    ):
        raise ValueError("Unsupported Zarr cache backend contract.")
    zarr_settings = _ZarrWriteSettings(
        point_chunk_rows=_require_exact_int(backend["point_chunk_rows"], "backend.point_chunk_rows"),
        point_shard_rows=_require_exact_int(backend["point_shard_rows"], "backend.point_shard_rows"),
        range_chunk_rows=_require_exact_int(backend["range_chunk_rows"], "backend.range_chunk_rows"),
        range_shard_rows=_require_exact_int(backend["range_shard_rows"], "backend.range_shard_rows"),
        codec_id=str(backend["codec_id"]),
    )

    source_payload = _require_mapping(
        root["source"],
        "source",
        keys={
            "points_name",
            "element_path",
            "row_count",
            "columns",
            "selected_schema",
            "signature_method",
            "signature",
            "value_normalization_method",
            "point_id_policy",
        },
    )
    columns = _require_mapping(source_payload["columns"], "source.columns", keys={"x", "y", "value"})
    selected_schema = source_payload["selected_schema"]
    if not isinstance(selected_schema, list):
        raise ValueError("`source.selected_schema` must be a JSON list.")
    source = _SourceMetadata(
        points_name=_require_string(source_payload["points_name"], "source.points_name"),
        element_path=_require_string(source_payload["element_path"], "source.element_path"),
        row_count=_require_exact_int(source_payload["row_count"], "source.row_count"),
        x_column=_require_string(columns["x"], "source.columns.x"),
        y_column=_require_string(columns["y"], "source.columns.y"),
        value_column=_require_string(columns["value"], "source.columns.value"),
        selected_schema=tuple(copy.deepcopy(selected_schema)),
        signature_method=_require_string(source_payload["signature_method"], "source.signature_method"),
        signature=_require_string(source_payload["signature"], "source.signature"),
        value_normalization_method=_require_string(
            source_payload["value_normalization_method"],
            "source.value_normalization_method",
        ),
        point_id_policy=_require_string(source_payload["point_id_policy"], "source.point_id_policy"),
    )

    geometry_payload = _require_mapping(
        root["geometry"],
        "geometry",
        keys={
            "x_origin",
            "y_origin",
            "x_min",
            "x_max",
            "y_min",
            "y_max",
            "coordinate_axes",
            "relative_coordinate_dtype",
        },
    )
    if geometry_payload["coordinate_axes"] != ["x", "y"] or geometry_payload["relative_coordinate_dtype"] != (
        "float32"
    ):
        raise ValueError("Unsupported cache coordinate contract.")
    geometry = _GeometryMetadata(
        **{
            name: _require_exact_float(geometry_payload[name], f"geometry.{name}")
            for name in ("x_origin", "y_origin", "x_min", "x_max", "y_min", "y_max")
        }
    )

    build_payload = _require_mapping(
        root["build"],
        "build",
        keys={
            "leaf_tile_size",
            "overview_point_budget",
            "target_points_per_bucket",
            "bucket_hash_method",
            "sampling_method",
            "sampling_seed",
            "sampling_microgrid_edge",
        },
    )
    build = _BuildMetadata(
        leaf_tile_size=_require_exact_int(build_payload["leaf_tile_size"], "build.leaf_tile_size"),
        overview_point_budget=_require_exact_int(
            build_payload["overview_point_budget"],
            "build.overview_point_budget",
        ),
        target_points_per_bucket=_require_exact_int(
            build_payload["target_points_per_bucket"],
            "build.target_points_per_bucket",
        ),
        bucket_hash_method=_require_string(build_payload["bucket_hash_method"], "build.bucket_hash_method"),
        sampling_method=_require_string(build_payload["sampling_method"], "build.sampling_method"),
        sampling_seed=_require_exact_int(build_payload["sampling_seed"], "build.sampling_seed"),
        sampling_microgrid_edge=_require_exact_int(
            build_payload["sampling_microgrid_edge"],
            "build.sampling_microgrid_edge",
        ),
    )

    levels_payload = root["levels"]
    if not isinstance(levels_payload, list) or not levels_payload:
        raise ValueError("`levels` must be a nonempty JSON list.")
    levels = tuple(_parse_level_metadata(value) for value in levels_payload)

    value_names_payload = root["value_names"]
    if not isinstance(value_names_payload, list):
        raise ValueError("`value_names` must be a JSON list.")
    value_names = tuple(_require_string(value, "value_names entry") for value in value_names_payload)

    catalog_payload = _require_mapping(
        root["catalog"],
        "catalog",
        keys={
            "value_count",
            "level_count",
            "manifest_row_count",
            "value_tile_row_count",
            "values_group",
            "manifest_group",
            "value_tiles_group",
            "manifest_row_order",
            "value_tile_row_order",
            "manifest_chunk_rows",
            "manifest_shard_rows",
            "value_tile_chunk_rows",
            "value_tile_shard_rows",
        },
    )
    if (
        catalog_payload["values_group"] != VALUES_GROUP
        or catalog_payload["manifest_group"] != MANIFEST_GROUP
        or catalog_payload["value_tiles_group"] != VALUE_TILES_GROUP
        or catalog_payload["manifest_row_order"] != list(_MANIFEST_ROW_ORDER)
        or catalog_payload["value_tile_row_order"] != list(_VALUE_TILE_ROW_ORDER)
    ):
        raise ValueError("Unsupported cache catalog identity or ordering.")
    catalog_settings = _CatalogWriteSettings(
        manifest_chunk_rows=_require_exact_int(
            catalog_payload["manifest_chunk_rows"],
            "catalog.manifest_chunk_rows",
        ),
        manifest_shard_rows=_require_exact_int(
            catalog_payload["manifest_shard_rows"],
            "catalog.manifest_shard_rows",
        ),
        value_tile_chunk_rows=_require_exact_int(
            catalog_payload["value_tile_chunk_rows"],
            "catalog.value_tile_chunk_rows",
        ),
        value_tile_shard_rows=_require_exact_int(
            catalog_payload["value_tile_shard_rows"],
            "catalog.value_tile_shard_rows",
        ),
    )
    catalog = _CatalogMetadata(
        value_count=_require_exact_int(catalog_payload["value_count"], "catalog.value_count"),
        level_count=_require_exact_int(catalog_payload["level_count"], "catalog.level_count"),
        manifest_row_count=_require_exact_int(
            catalog_payload["manifest_row_count"],
            "catalog.manifest_row_count",
        ),
        value_tile_row_count=_require_exact_int(
            catalog_payload["value_tile_row_count"],
            "catalog.value_tile_row_count",
        ),
        settings=catalog_settings,
    )

    value_major_payload = _require_mapping(
        root["value_major"],
        "value_major",
        keys={
            "group",
            "point_row_order",
            "point_chunk_rows",
            "point_shard_rows",
        },
    )
    if value_major_payload["group"] != VALUE_MAJOR_GROUP or value_major_payload["point_row_order"] != list(
        VALUE_MAJOR_ROW_ORDER
    ):
        raise ValueError("Unsupported value-major identity or ordering.")
    value_major = _ValueMajorMetadata(
        point_chunk_rows=_require_exact_int(
            value_major_payload["point_chunk_rows"],
            "value_major.point_chunk_rows",
        ),
        point_shard_rows=_require_exact_int(
            value_major_payload["point_shard_rows"],
            "value_major.point_shard_rows",
        ),
    )

    return _CacheAttributes(
        cache_generation_id=_require_string(root["cache_generation_id"], "cache_generation_id"),
        publication_state=_require_string(root["publication_state"], "publication_state"),
        created_by_version=_require_string(created_by["version"], "created_by.version"),
        zarr_settings=zarr_settings,
        source=source,
        geometry=geometry,
        build=build,
        levels=levels,
        value_names=value_names,
        catalog=catalog,
        value_major=value_major,
    )


def _parse_level_metadata(value: object) -> _LevelMetadata:
    payload = _require_mapping(
        value,
        "level",
        keys={
            "level",
            "kind",
            "tile_size",
            "grid_width",
            "grid_height",
            "max_points_per_tile",
            "point_count_upper_bound",
            "bucket_count",
            "tile_count",
            "point_count",
            "range_count",
            "relative_directory",
        },
    )
    maximum = payload["max_points_per_tile"]
    if maximum is not None:
        maximum = _require_exact_int(maximum, "levels.max_points_per_tile")
    return _LevelMetadata(
        level=_require_exact_int(payload["level"], "levels.level"),
        kind=_require_string(payload["kind"], "levels.kind"),
        tile_size=_require_exact_int(payload["tile_size"], "levels.tile_size"),
        grid_width=_require_exact_int(payload["grid_width"], "levels.grid_width"),
        grid_height=_require_exact_int(payload["grid_height"], "levels.grid_height"),
        max_points_per_tile=maximum,
        point_count_upper_bound=_require_exact_int(
            payload["point_count_upper_bound"],
            "levels.point_count_upper_bound",
        ),
        bucket_count=_require_exact_int(payload["bucket_count"], "levels.bucket_count"),
        tile_count=_require_exact_int(payload["tile_count"], "levels.tile_count"),
        point_count=_require_exact_int(payload["point_count"], "levels.point_count"),
        range_count=_require_exact_int(payload["range_count"], "levels.range_count"),
        relative_directory=_require_string(payload["relative_directory"], "levels.relative_directory"),
    )


def _normalize_selected_schema(value: object) -> tuple[dict[str, object], ...]:
    if not isinstance(value, tuple) or len(value) != 3:
        raise ValueError("`source.selected_schema` must contain x, y, and value fields.")
    normalized: list[dict[str, object]] = []
    for expected_role, item in zip(("x", "y", "value"), value, strict=True):
        field = _require_mapping(item, "selected schema field", keys={"role", "name", "nullable", "type"})
        if field["role"] != expected_role:
            raise ValueError("Selected schema fields must follow x, y, value role order.")
        name = _require_string(field["name"], "selected schema field name")
        nullable = field["nullable"]
        if type(nullable) is not bool:
            raise ValueError("Selected schema nullability must be boolean.")
        type_payload = _normalize_arrow_type(field["type"])
        normalized.append({"role": expected_role, "name": name, "nullable": nullable, "type": type_payload})
    return tuple(normalized)


def _normalize_arrow_type(value: object) -> dict[str, object]:
    payload = _require_mapping(value, "Arrow type")
    kind = payload.get("kind")
    if kind == "integer":
        _require_exact_keys(payload, {"kind", "signed", "bit_width"}, "Arrow integer type")
        if type(payload["signed"]) is not bool:
            raise ValueError("Arrow integer signedness must be boolean.")
        bit_width = _require_exact_int(payload["bit_width"], "Arrow integer bit width")
        if bit_width not in {8, 16, 32, 64}:
            raise ValueError("Arrow integer bit width is unsupported.")
        return {"kind": "integer", "signed": payload["signed"], "bit_width": bit_width}
    if kind == "float":
        _require_exact_keys(payload, {"kind", "bit_width"}, "Arrow float type")
        bit_width = _require_exact_int(payload["bit_width"], "Arrow float bit width")
        if bit_width not in {16, 32, 64}:
            raise ValueError("Arrow float bit width is unsupported.")
        return {"kind": "float", "bit_width": bit_width}
    if kind == "string":
        _require_exact_keys(payload, {"kind", "offset_width"}, "Arrow string type")
        offset_width = _require_exact_int(payload["offset_width"], "Arrow string offset width")
        if offset_width not in {32, 64}:
            raise ValueError("Arrow string offset width is unsupported.")
        return {"kind": "string", "offset_width": offset_width}
    if kind == "dictionary":
        _require_exact_keys(payload, {"kind", "index", "value", "ordered"}, "Arrow dictionary type")
        if type(payload["ordered"]) is not bool:
            raise ValueError("Arrow dictionary ordering must be boolean.")
        return {
            "kind": "dictionary",
            "index": _normalize_arrow_type(payload["index"]),
            "value": _normalize_arrow_type(payload["value"]),
            "ordered": payload["ordered"],
        }
    raise ValueError("Unsupported normalized Arrow type metadata.")


def _require_mapping(
    value: object,
    name: str,
    *,
    keys: set[str] | frozenset[str] | None = None,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"`{name}` must be a string-keyed mapping.")
    result = dict(value)
    if keys is not None:
        _require_exact_keys(result, keys, name)
    return result


def _require_exact_keys(value: Mapping[str, object], keys: set[str] | frozenset[str], name: str) -> None:
    if set(value) != set(keys):
        raise ValueError(f"`{name}` has missing or unexpected keys.")


def _require_exact_int(value: object, name: str) -> int:
    if type(value) is not int:
        raise ValueError(f"`{name}` must be a JSON integer.")
    return value


def _require_exact_float(value: object, name: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"`{name}` must be a finite JSON float.")
    return value


def _require_string(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"`{name}` must be a string.")
    return value


def _require_nonempty_string(value: object, name: str) -> str:
    result = _require_string(value, name)
    if result == "":
        raise ValueError(f"`{name}` must be nonempty.")
    return result


def _require_relative_posix_path(value: object, name: str) -> str:
    result = _require_nonempty_string(value, name)
    path = PurePosixPath(result)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != result:
        raise ValueError(f"`{name}` must be a normalized relative POSIX path.")
    return result
