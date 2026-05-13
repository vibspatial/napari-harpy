from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from loguru import logger
from matplotlib.colors import to_rgba
from napari.layers import Shapes

from napari_harpy.core._color_source import ShapeColorSourceSpec, ShapeColorValueKind
from napari_harpy.viewer._styling import (
    StyledPaletteSource,
    build_string_categorical_values,
    categorical_colors_for_values,
    continuous_colors_for_values,
    default_categorical_palette_for_categories,
    is_string_like_series,
    normalize_category_value,
)

if TYPE_CHECKING:
    import geopandas as gpd

StyledShapesPaletteSource = StyledPaletteSource
SHAPES_MISSING_BASE_COLOR = "#808080"
SHAPES_FACE_ALPHA = 0.35
SHAPES_EDGE_ALPHA = 1.0


@dataclass(frozen=True)
class StyledShapesStyleResult:
    """Describe how one styled shapes layer was colored."""

    value_kind: ShapeColorValueKind
    palette_source: StyledShapesPaletteSource | None
    coercion_applied: bool


def apply_shape_color_source_to_shapes_layer(
    layer: Shapes,
    *,
    shapes_element: gpd.GeoDataFrame,
    style_spec: ShapeColorSourceSpec,
    source_shapes_index_by_row: tuple[Any, ...],
    source_shapes_index_feature_name: str,
) -> StyledShapesStyleResult:
    """Apply one direct shapes-column color source to a napari ``Shapes`` layer."""
    if style_spec.source_kind != "shape_column":
        raise ValueError(f"Shape color sources must use `source_kind='shape_column'`, got `{style_spec.source_kind}`.")
    if style_spec.value_key not in shapes_element.columns:
        raise ValueError(f"Shape column `{style_spec.value_key}` is not available in the selected shapes element.")
    if not shapes_element.index.is_unique:
        raise ValueError(
            "Styled shape coloring requires unique source GeoDataFrame index labels; "
            "duplicate index labels make rendered-row to source-row style lookup ambiguous."
        )
    if len(source_shapes_index_by_row) != len(layer.data):
        raise ValueError(
            "`source_shapes_index_by_row` must contain one source index for each rendered napari shape row."
        )

    # Start with one style value per source GeoDataFrame row.
    full_values = shapes_element[style_spec.value_key]
    # Napari rows can outnumber source rows when a MultiPolygon expands into
    # multiple rendered polygons, so this index repeats source labels in
    # rendered-row order, e.g. ("cell_7", "cell_7", "cell_8").
    source_index = pd.Index(source_shapes_index_by_row)
    missing_source_indices = source_index[~source_index.isin(full_values.index)]
    if len(missing_source_indices) > 0:
        preview = ", ".join(repr(value) for value in pd.unique(missing_source_indices)[:5])
        raise ValueError(f"Could not align rendered shapes back to source index label(s): {preview}.")

    # Reindex repeats/reorders source values so there is exactly one style
    # value per rendered napari shape row, then switch to napari row numbering.
    row_values = full_values.reindex(source_index)
    row_values.index = pd.RangeIndex(len(row_values))

    if style_spec.value_kind == "categorical":
        style_result, base_colors, feature_values = _build_categorical_shape_style(
            shapes_element=shapes_element,
            column_name=style_spec.value_key,
            full_values=full_values,
            row_values=row_values,
        )
    else:
        style_result, base_colors, feature_values = _build_continuous_shape_style(row_values)

    _apply_base_colors_to_shapes_layer(layer, base_colors)
    # Unlike styled labels, styled shapes already have a useful feature table
    # from the geometry-loading path: the source GeoDataFrame index column used
    # for status-bar display. Preserve it and add only the selected style source
    # column for inspection/coloring.
    _set_shape_style_feature(
        layer,
        feature_values,
        style_column_name=style_spec.value_key,
        source_index_feature_name=source_shapes_index_feature_name,
    )
    return style_result


def build_styled_shapes_layer_name(shapes_name: str, style_spec: ShapeColorSourceSpec) -> str:
    """Return the user-facing layer name for one styled shapes variant."""
    return f"{shapes_name}[shape:{style_spec.value_key}]"


def disambiguate_shape_style_feature_name(style_column_name: str, source_index_feature_name: str) -> str:
    """Return the layer.features column name for a selected shape style value.

    GeoPandas and SpatialData allow a shapes GeoDataFrame to have both an
    index named, for example, ``cell_id`` and a normal column named
    ``cell_id``. Harpy stores the source GeoDataFrame index in
    ``layer.features`` under the index name for status display, while styled
    shapes also store the selected style column in ``layer.features`` for
    inspection. Coloring by the normal ``cell_id`` column would otherwise
    collide with the source-index feature column, so the selected style value
    is stored as ``cell_id__value``.
    """
    if style_column_name == source_index_feature_name:
        return f"{style_column_name}__value"
    return style_column_name


def _build_categorical_shape_style(
    *,
    shapes_element: gpd.GeoDataFrame,
    column_name: str,
    full_values: pd.Series,
    row_values: pd.Series,
) -> tuple[StyledShapesStyleResult, pd.Series, pd.Series]:
    """Build categorical colors from source-level values and rendered-row values.

    ``full_values`` is the source-level truth: it is one value per source
    GeoDataFrame row and is used for category discovery, companion palette
    validation, and string/object coercion warnings. ``row_values`` is already
    aligned to rendered napari shape rows, so it can repeat one source value
    for multiple rendered polygon parts and is used for actual coloring and
    ``layer.features``. Some source rows can be skipped before rendering
    because their geometries are empty, invalid, or unsupported, so
    ``row_values`` may be only a rendered subset of the source values. Category
    and companion-palette resolution therefore use ``full_values`` instead of
    the rendered subset.
    """
    companion_palette_allowed = True
    coercion_applied = False

    if _is_categorical_dtype(full_values):
        normalized_full_values = _normalize_category_series(full_values)
        normalized_row_values = _normalize_category_series(row_values)
        categories = _present_categories(normalized_full_values, full_values.cat.categories)
    elif _is_bool_series(full_values):
        normalized_full_values = _normalize_category_series(full_values)
        normalized_row_values = _normalize_category_series(row_values)
        categories = _present_categories(normalized_full_values, [False, True])
    elif _is_exact_binary_integer_series(full_values):
        numeric_full_values = pd.to_numeric(full_values, errors="coerce").astype("Int64")
        numeric_row_values = pd.to_numeric(row_values, errors="coerce").astype("Int64")
        normalized_full_values = _normalize_category_series(numeric_full_values)
        normalized_row_values = _normalize_category_series(numeric_row_values)
        categories = _present_categories(normalized_full_values, [0, 1])
    elif is_string_like_series(full_values):
        normalized_row_values, categories = build_string_categorical_values(
            full_values=full_values,
            row_values=row_values,
            column_name=column_name,
        )
        normalized_full_values = pd.Series(
            [pd.NA if pd.isna(value) else str(value) for value in full_values],
            index=full_values.index,
            name=column_name,
            dtype="object",
        )
        companion_palette_allowed = False
        coercion_applied = True
    else:
        normalized_full_values = _normalize_category_series(full_values)
        normalized_row_values = _normalize_category_series(row_values)
        categories = list(pd.unique(normalized_full_values.dropna()))
        companion_palette_allowed = False

    if companion_palette_allowed:
        palette_source, palette = _resolve_shape_categorical_palette(
            shapes_element=shapes_element,
            column_name=column_name,
            full_values=normalized_full_values,
            categories=categories,
        )
    else:
        palette_source = "default_missing"
        palette = default_categorical_palette_for_categories(categories)

    base_colors = categorical_colors_for_values(
        normalized_row_values,
        categories=categories,
        palette=palette,
        missing_color=SHAPES_MISSING_BASE_COLOR,
    )
    return (
        StyledShapesStyleResult(
            value_kind="categorical",
            palette_source=palette_source,
            coercion_applied=coercion_applied,
        ),
        base_colors,
        normalized_row_values,
    )


def _build_continuous_shape_style(row_values: pd.Series) -> tuple[StyledShapesStyleResult, pd.Series, pd.Series]:
    numeric_row_values = pd.to_numeric(row_values, errors="coerce").astype("float64")
    base_colors = continuous_colors_for_values(numeric_row_values, missing_color=SHAPES_MISSING_BASE_COLOR)
    return (
        StyledShapesStyleResult(
            value_kind="continuous",
            palette_source=None,
            coercion_applied=False,
        ),
        base_colors,
        numeric_row_values,
    )


def _resolve_shape_categorical_palette(
    *,
    shapes_element: gpd.GeoDataFrame,
    column_name: str,
    full_values: pd.Series,
    categories: Sequence[object],
) -> tuple[StyledShapesPaletteSource, list[Any]]:
    colors_column_name = f"{column_name}_colors"
    if colors_column_name not in shapes_element.columns:
        logger.info(
            f"No `{colors_column_name}` companion color column found; using the default categorical palette."
        )
        return "default_missing", default_categorical_palette_for_categories(categories)

    companion_colors = shapes_element[colors_column_name]
    color_by_category: dict[object, Any] = {}
    for category in categories:
        category_mask = full_values == category
        category_colors = companion_colors.loc[category_mask.fillna(False)].dropna()
        if category_colors.empty:
            logger.warning(
                f"Companion color column `{colors_column_name}` is missing a color for category `{category}`; "
                "using the default categorical palette."
            )
            return "default_invalid", default_categorical_palette_for_categories(categories)

        normalized_colors: set[tuple[float, float, float, float]] = set()
        for color in category_colors:
            normalized_color = _normalize_color(color)
            if normalized_color is None:
                logger.warning(
                    f"Companion color column `{colors_column_name}` contains invalid color values; "
                    "using the default categorical palette."
                )
                return "default_invalid", default_categorical_palette_for_categories(categories)
            normalized_colors.add(normalized_color)

        if len(normalized_colors) != 1:
            logger.warning(
                f"Companion color column `{colors_column_name}` maps category `{category}` to multiple colors; "
                "using the default categorical palette."
            )
            return "default_invalid", default_categorical_palette_for_categories(categories)

        color_by_category[normalize_category_value(category)] = next(iter(normalized_colors))

    logger.info(f"Using companion color column `{colors_column_name}` for shapes viewer coloring.")
    return "stored", [color_by_category[normalize_category_value(category)] for category in categories]


def _apply_base_colors_to_shapes_layer(layer: Shapes, base_colors: pd.Series) -> None:
    layer.face_color = _with_alpha(base_colors, SHAPES_FACE_ALPHA)
    layer.edge_color = _with_alpha(base_colors, SHAPES_EDGE_ALPHA)
    refresh = getattr(layer, "refresh", None)
    if callable(refresh):
        refresh()


def _set_shape_style_feature(
    layer: Shapes,
    values: pd.Series,
    *,
    style_column_name: str,
    source_index_feature_name: str,
) -> None:
    feature_name = disambiguate_shape_style_feature_name(style_column_name, source_index_feature_name)
    features = layer.features.copy()
    if len(features) == 0 and len(values) > 0:
        features = pd.DataFrame(index=pd.RangeIndex(len(values)))
    if len(features) != len(values):
        raise ValueError("Shapes layer features must contain one row for each rendered napari shape row.")

    features[feature_name] = values.to_numpy()
    layer.features = features


def _with_alpha(colors: pd.Series, alpha: float) -> np.ndarray:
    rgba = np.zeros((len(colors), 4), dtype=float)
    for row_index, color in enumerate(colors):
        base_rgba = to_rgba(color)
        rgba[row_index] = (base_rgba[0], base_rgba[1], base_rgba[2], alpha)
    return rgba


def _normalize_color(color: object) -> tuple[float, float, float, float] | None:
    try:
        return tuple(float(component) for component in to_rgba(color))
    except (TypeError, ValueError):
        return None


def _normalize_category_series(values: pd.Series) -> pd.Series:
    return pd.Series(
        [pd.NA if pd.isna(value) else normalize_category_value(value) for value in values],
        index=values.index,
        name=values.name,
        dtype="object",
    )


def _present_categories(values: pd.Series, candidate_categories: Sequence[object]) -> list[object]:
    present_values = {normalize_category_value(value) for value in values.dropna().tolist()}
    return [
        normalized_category
        for category in candidate_categories
        if (normalized_category := normalize_category_value(category)) in present_values
    ]


def _is_categorical_dtype(values: pd.Series) -> bool:
    return isinstance(values.dtype, pd.CategoricalDtype)


def _is_bool_series(values: pd.Series) -> bool:
    non_null = values.dropna()
    if non_null.empty:
        return False
    if pd.api.types.is_bool_dtype(values.dtype):
        return True
    return all(isinstance(value, bool | np.bool_) for value in non_null.tolist())


def _is_exact_binary_integer_series(values: pd.Series) -> bool:
    non_null = values.dropna()
    if non_null.empty:
        return False
    if not pd.api.types.is_integer_dtype(values.dtype) and not all(
        isinstance(value, int | np.integer) and not isinstance(value, bool | np.bool_)
        for value in non_null.tolist()
    ):
        return False

    normalized_values = {int(value) for value in pd.to_numeric(non_null, errors="coerce").dropna().tolist()}
    return normalized_values == {0, 1}
