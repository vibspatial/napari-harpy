from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import TYPE_CHECKING

import geopandas as gpd
import harpy as hp
import numpy as np
import pandas as pd
from napari.layers import Shapes
from shapely.geometry import Polygon
from spatialdata.transformations import Identity

from napari_harpy.core.spatialdata import get_coordinate_system_names_from_sdata
from napari_harpy.core.validation import normalize_spatialdata_dataframe_column_name, normalize_spatialdata_name

if TYPE_CHECKING:
    from spatialdata import SpatialData


DEFAULT_SHAPES_INDEX_NAME = "instance_id"
DEFAULT_SHAPES_INDEX_PREFIX = "shape"
DEFAULT_ELLIPSE_SEGMENTS = 64
MIN_ELLIPSE_SEGMENTS = 8


@dataclass(frozen=True)
class CreateShapesElementRequest:
    sdata: SpatialData
    shapes_name: str
    coordinate_system: str
    overwrite: bool = False
    index_name: str = DEFAULT_SHAPES_INDEX_NAME
    index_prefix: str = DEFAULT_SHAPES_INDEX_PREFIX


@dataclass(frozen=True)
class CreateShapesElementResult:
    shapes_name: str
    coordinate_system: str
    row_count: int


def create_shapes_element_from_napari_shapes_layer(
    request: CreateShapesElementRequest,
    layer: Shapes,
) -> CreateShapesElementResult:
    """Create or update one SpatialData shapes element from a napari Shapes layer."""
    sdata = request.sdata
    if sdata is None:
        raise ValueError("Create shapes element request requires a SpatialData object.")

    shapes_name = _normalize_spatialdata_name_field(request.shapes_name, field_name="`shapes_name`")
    coordinate_system = _normalize_string_field(request.coordinate_system, field_name="`coordinate_system`")
    if not isinstance(request.overwrite, bool):
        raise ValueError("`overwrite` must be a boolean.")
    index_name = _normalize_dataframe_column_name_field(request.index_name, field_name="`index_name`")
    index_prefix = _normalize_string_field(request.index_prefix, field_name="`index_prefix`")

    available_coordinate_systems = get_coordinate_system_names_from_sdata(sdata)
    if coordinate_system not in available_coordinate_systems:
        available = ", ".join(f"`{name}`" for name in available_coordinate_systems) or "none"
        raise ValueError(
            f"Coordinate system `{coordinate_system}` is not available in the selected SpatialData object. "
            f"Available coordinate systems: {available}."
        )

    if not request.overwrite and shapes_name in sdata.shapes:
        raise ValueError(f"Shapes element `{shapes_name}` already exists. Set `overwrite=True` to replace it.")

    geodataframe = napari_shapes_layer_to_geodataframe(
        layer,
        index_name=index_name,
        index_prefix=index_prefix,
    )

    _ = hp.sh.add_shapes(
        sdata,
        input=geodataframe,
        output_shapes_name=shapes_name,
        transformations={coordinate_system: Identity()},
        instance_key=index_name,
        overwrite=request.overwrite,
    )

    return CreateShapesElementResult(
        shapes_name=shapes_name,
        coordinate_system=coordinate_system,
        row_count=len(geodataframe),
    )


def napari_shapes_layer_to_geodataframe(
    layer: Shapes,
    *,
    index_name: str = DEFAULT_SHAPES_INDEX_NAME,
    index_prefix: str = DEFAULT_SHAPES_INDEX_PREFIX,
    ellipse_segments: int = DEFAULT_ELLIPSE_SEGMENTS,
) -> gpd.GeoDataFrame:
    """Convert one editable napari Shapes layer into a polygon GeoDataFrame."""
    _validate_ellipse_segments(ellipse_segments)

    data = list(layer.data)
    shape_types = [str(shape_type).lower() for shape_type in layer.shape_type]
    if not data:
        raise ValueError("Draw at least one supported shape before saving.")
    if len(shape_types) != len(data):
        raise ValueError("Napari shapes layer has inconsistent data and shape-type lengths.")

    features = _build_features_with_instance_ids(
        layer.features,
        row_count=len(data),
        index_name=index_name,
        index_prefix=index_prefix,
    )

    geometries: list[Polygon] = []
    for row_index, (vertices, shape_type) in enumerate(zip(data, shape_types, strict=True)):
        if shape_type in {"line", "path"}:
            raise ValueError("Lines and paths cannot be saved as SpatialData shapes yet.")
        if shape_type in {"polygon", "rectangle"}:
            geometries.append(_polygon_shape_to_polygon(vertices, row_index=row_index))
            continue
        if shape_type == "ellipse":
            geometries.append(
                _ellipse_shape_to_polygon(
                    vertices,
                    row_index=row_index,
                    ellipse_segments=ellipse_segments,
                )
            )
            continue
        raise ValueError(f"Shape row `{row_index}` has unsupported napari shape type `{shape_type}`.")

    # Only write generated IDs back after every geometry row validates, so
    # failed conversions leave the napari layer metadata untouched.
    layer.features = features

    geodataframe_data = {
        column: features[column].tolist()
        for column in features.columns
        if column != index_name
    }
    # Napari keeps row identity in features; SpatialData shapes keep it in the
    # GeoDataFrame index.
    instance_ids = features[index_name]
    geodataframe = gpd.GeoDataFrame(
        geodataframe_data,
        geometry=geometries,
        index=pd.Index(instance_ids.to_list(), name=index_name),
    )
    return geodataframe


def _normalize_spatialdata_name_field(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a non-empty string.")
    return normalize_spatialdata_name(value, field_name)


def _normalize_dataframe_column_name_field(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a non-empty string.")
    return normalize_spatialdata_dataframe_column_name(value, field_name)


def _normalize_string_field(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a non-empty string.")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be a non-empty string.")
    return normalized


def _validate_ellipse_segments(ellipse_segments: int) -> None:
    if (
        isinstance(ellipse_segments, bool)
        or not isinstance(ellipse_segments, Integral)
        or int(ellipse_segments) < MIN_ELLIPSE_SEGMENTS
    ):
        raise ValueError(f"`ellipse_segments` must be an integer greater than or equal to {MIN_ELLIPSE_SEGMENTS}.")


def _build_features_with_instance_ids(
    features: pd.DataFrame,
    *,
    row_count: int,
    index_name: str,
    index_prefix: str,
) -> pd.DataFrame:
    """Return row-aligned features with stable IDs for every napari shape row.

    Napari stores per-shape metadata in a features DataFrame rather than in a
    semantic row index. Harpy keeps the future GeoDataFrame index values in a
    named features column, usually ``instance_id``. Existing values are
    preserved for repeated saves, missing values from newly drawn rows are
    filled with unique ``shape_N`` IDs, and duplicate generated IDs copied by
    napari into new rows are treated as missing. The completed values can then
    be written back to ``layer.features`` before becoming the saved
    GeoDataFrame index.
    """
    features = features.copy()
    features = features.reindex(range(row_count)).reset_index(drop=True)
    if index_name not in features.columns:
        features[index_name] = pd.NA

    raw_values = features[index_name].tolist()
    existing_values = [_normalize_instance_id(value) for value in raw_values]
    normalized_values: list[str | None] = []
    used_values: set[str] = set()
    duplicate_values: set[str] = set()
    for value in existing_values:
        if value is None:
            normalized_values.append(None)
            continue
        if value in used_values:
            # Napari can copy current feature defaults into newly drawn rows.
            # Keep the first generated ID untouched, then treat later duplicate
            # generated IDs as missing so they get fresh IDs below.
            # Non-generated duplicates are collected in `duplicate_values` and
            # rejected after this loop.
            if _is_generated_instance_id(value, index_prefix=index_prefix):
                normalized_values.append(None)
                continue
            duplicate_values.add(value)
        used_values.add(value)
        normalized_values.append(value)

    if duplicate_values:
        preview = ", ".join(f"`{value}`" for value in sorted(duplicate_values)[:5])
        raise ValueError(f"`{index_name}` values must be unique before saving shapes: {preview}.")

    next_suffix = _next_generated_suffix(used_values, index_prefix=index_prefix)
    instance_ids: list[str] = []
    for value in normalized_values:
        if value is None:
            value = f"{index_prefix}_{next_suffix}"
            while value in used_values:
                next_suffix += 1
                value = f"{index_prefix}_{next_suffix}"
            used_values.add(value)
            next_suffix += 1
        instance_ids.append(value)

    features[index_name] = instance_ids
    return features


def _normalize_instance_id(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    normalized = str(value)
    return normalized if normalized else None


def _is_generated_instance_id(value: str, *, index_prefix: str) -> bool:
    prefix = f"{index_prefix}_"
    suffix_text = value.removeprefix(prefix)
    return value.startswith(prefix) and suffix_text.isdecimal()


def _next_generated_suffix(values: set[str], *, index_prefix: str) -> int:
    prefix = f"{index_prefix}_"
    suffixes: list[int] = []
    for value in values:
        if not value.startswith(prefix):
            continue
        suffix_text = value.removeprefix(prefix)
        if suffix_text.isdecimal():
            suffixes.append(int(suffix_text))
    if not suffixes:
        return 0
    return max(suffixes) + 1


def _polygon_shape_to_polygon(vertices: object, *, row_index: int) -> Polygon:
    coordinates_yx = _coerce_vertices(vertices, row_index=row_index)
    if len(coordinates_yx) < 3:
        raise ValueError(f"Shape row `{row_index}` has too few vertices for a valid polygon.")
    return _make_valid_polygon(coordinates_yx[:, [1, 0]], row_index=row_index)


def _ellipse_shape_to_polygon(
    vertices: object,
    *,
    row_index: int,
    ellipse_segments: int,
) -> Polygon:
    coordinates_yx = _coerce_vertices(vertices, row_index=row_index)
    if coordinates_yx.shape != (4, 2):
        raise ValueError(f"Ellipse row `{row_index}` cannot be converted to a valid polygon.")

    origin = coordinates_yx[0]
    axis_a = (coordinates_yx[1] - origin) / 2.0
    axis_b = (coordinates_yx[3] - origin) / 2.0
    if not _has_positive_finite_length(axis_a) or not _has_positive_finite_length(axis_b):
        raise ValueError(f"Ellipse row `{row_index}` cannot be converted to a valid polygon.")

    center = coordinates_yx.mean(axis=0)
    angles = np.linspace(0.0, 2.0 * np.pi, num=int(ellipse_segments), endpoint=False)
    boundary_yx = center + np.cos(angles)[:, np.newaxis] * axis_a + np.sin(angles)[:, np.newaxis] * axis_b
    return _make_valid_polygon(boundary_yx[:, [1, 0]], row_index=row_index, shape_name="Ellipse")


def _coerce_vertices(vertices: object, *, row_index: int) -> np.ndarray:
    coordinates = np.asarray(vertices, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError(f"Shape row `{row_index}` must contain 2D coordinates.")
    if coordinates.size == 0:
        raise ValueError(f"Shape row `{row_index}` is empty.")
    if not np.isfinite(coordinates).all():
        raise ValueError(f"Shape row `{row_index}` contains non-finite coordinates.")
    return coordinates


def _make_valid_polygon(coordinates_xy: np.ndarray, *, row_index: int, shape_name: str = "Shape") -> Polygon:
    polygon = Polygon(coordinates_xy)
    if polygon.is_empty or not polygon.is_valid or polygon.area <= 0:
        raise ValueError(f"{shape_name} row `{row_index}` cannot be converted to a valid polygon.")
    return polygon


def _has_positive_finite_length(vector: np.ndarray) -> bool:
    length = float(np.linalg.norm(vector))
    return np.isfinite(length) and length > 0
