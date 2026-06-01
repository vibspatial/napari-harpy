from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from loguru import logger
from matplotlib.colors import to_rgba
from napari.layers import Shapes

from napari_harpy.core._color_source import (
    ShapeColorValueKind,
    ShapeColumnColorSourceSpec,
    TableColorSourceSpec,
    validate_shape_color_value_kind,
)
from napari_harpy.viewer._styling import (
    StyledPaletteSource,
    build_string_categorical_values,
    categorical_colors_for_values,
    continuous_colors_for_values,
    default_categorical_palette_for_categories,
    is_string_like_series,
    normalize_category_value,
    validate_styled_palette_source,
)

if TYPE_CHECKING:
    import geopandas as gpd

SHAPES_MISSING_BASE_COLOR = "#808080"
SHAPES_FACE_ALPHA = 0.35
SHAPES_EDGE_ALPHA = 1.0
_SHAPES_EDGE_WIDTH_SYNC_CALLBACK_ATTR = "_harpy_shapes_edge_width_sync_callback"
_SHAPES_EDGE_COLOR_SYNC_CALLBACK_ATTR = "_harpy_shapes_edge_color_sync_callback"


@dataclass(frozen=True)
class ShapesStyleResult:
    """Describe shapes styling metadata, if styling was applied."""

    value_kind: ShapeColorValueKind | None
    palette_source: StyledPaletteSource | None
    coercion_applied: bool

    def __post_init__(self) -> None:
        if self.value_kind is not None:
            validate_shape_color_value_kind(self.value_kind)
        if self.palette_source is not None:
            validate_styled_palette_source(self.palette_source)


@dataclass(frozen=True)
class ShapesLoadResult(ShapesStyleResult):
    """Describe a primary or styled shapes layer load/update result."""

    layer: Shapes
    created: bool
    skipped_geometry_count: int = 0


def apply_shape_color_source_to_shapes_layer(
    layer: Shapes,
    *,
    shapes_element: gpd.GeoDataFrame,
    style_spec: ShapeColumnColorSourceSpec,
    source_shapes_index_by_row: tuple[Any, ...],
    source_shapes_index_feature_name: str,
    fill: bool = False,
) -> ShapesStyleResult:
    """Apply one direct shapes-column color source to a napari ``Shapes`` layer.

    Parameters
    ----------
    source_shapes_index_by_row
        Source GeoDataFrame index label for each rendered napari shape row.
        This can be longer than the source GeoDataFrame row count when one
        source row, such as a ``MultiPolygon``, expands into multiple rendered
        napari shapes. For example, if source row ``"cell_7"`` expands into
        three polygons and ``"cell_8"`` expands into one polygon, this mapping
        is ``("cell_7", "cell_7", "cell_7", "cell_8")``. Styled shapes use
        it to repeat the source row's style value for every rendered part.
    source_shapes_index_feature_name
        Name of the ``layer.features`` column that stores the source
        GeoDataFrame index for napari-visible inspection and status-bar text.
        This follows the GeoDataFrame index name, falling back to ``"index"``
        for unnamed indexes.
    """
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
    source_values = shapes_element[style_spec.value_key]
    # Napari rows can outnumber source rows when a MultiPolygon expands into
    # multiple rendered polygons, so this index repeats source labels in
    # rendered-row order, e.g. ("cell_7", "cell_7", "cell_8").
    source_index_by_rendered_row = pd.Index(source_shapes_index_by_row)
    missing_source_indices = source_index_by_rendered_row[~source_index_by_rendered_row.isin(source_values.index)]
    if len(missing_source_indices) > 0:
        preview = ", ".join(repr(value) for value in pd.unique(missing_source_indices)[:5])
        raise ValueError(f"Could not align rendered shapes back to source index label(s): {preview}.")

    # Reindex repeats/reorders source values so there is exactly one style
    # value per rendered napari shape row, then switch to napari row numbering.
    rendered_row_values = source_values.reindex(source_index_by_rendered_row)
    rendered_row_values.index = pd.RangeIndex(len(rendered_row_values))

    if style_spec.value_kind == "categorical":
        style_result, rendered_row_colors, feature_values = _build_categorical_shape_style(
            shapes_element=shapes_element,
            column_name=style_spec.value_key,
            source_values=source_values,
            rendered_row_values=rendered_row_values,
        )
    else:
        style_result, rendered_row_colors, feature_values = _build_continuous_shape_style(rendered_row_values)

    _apply_rendered_row_colors_to_shapes_layer(layer, rendered_row_colors, fill=fill)
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


def build_styled_shapes_layer_name(
    shapes_name: str,
    style_spec: ShapeColumnColorSourceSpec | TableColorSourceSpec,
) -> str:
    """Return the user-facing layer name for one styled shapes variant."""
    if style_spec.source_kind == "obs_column":
        return f"{shapes_name}[obs:{style_spec.value_key}]"
    if style_spec.source_kind == "x_var":
        return f"{shapes_name}[X:{style_spec.value_key}]"
    return f"{shapes_name}[shapes_column:{style_spec.value_key}]"


def _connect_current_edge_width_to_global_edge_width(layer: Shapes) -> None:
    if getattr(layer, _SHAPES_EDGE_WIDTH_SYNC_CALLBACK_ATTR, None) is not None:
        return

    def _sync_current_edge_width_to_all_shapes(_event: Any | None = None) -> None:
        layer.edge_width = layer.current_edge_width

    # Napari exposes `current_edge_width`, but emits its changes through
    # the `edge_width` event.
    layer.events.edge_width.connect(_sync_current_edge_width_to_all_shapes)
    setattr(layer, _SHAPES_EDGE_WIDTH_SYNC_CALLBACK_ATTR, _sync_current_edge_width_to_all_shapes)


def _connect_current_edge_color_to_global_edge_color(layer: Shapes) -> None:
    if getattr(layer, _SHAPES_EDGE_COLOR_SYNC_CALLBACK_ATTR, None) is not None:
        return

    def _sync_current_edge_color_to_all_shapes(_event: Any | None = None) -> None:
        layer.edge_color = layer.current_edge_color

    layer.events.current_edge_color.connect(_sync_current_edge_color_to_all_shapes)
    setattr(layer, _SHAPES_EDGE_COLOR_SYNC_CALLBACK_ATTR, _sync_current_edge_color_to_all_shapes)


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
    source_values: pd.Series,
    rendered_row_values: pd.Series,
) -> tuple[ShapesStyleResult, pd.Series, pd.Series]:
    """Build categorical colors from source-level values and rendered-row values.

    Parameters
    ----------
    shapes_element
        Source GeoDataFrame used to resolve optional companion palette columns.
    column_name
        Source column name used in warning messages and companion palette
        lookup.
    source_values
        Complete source column with one value per source GeoDataFrame row.
        Category discovery, companion palette validation, and string/object
        coercion warnings use this complete series so styling remains stable
        even when only a subset of source rows is rendered.
    rendered_row_values
        Values already aligned to rendered napari shape rows. This series can
        repeat one source value for multiple rendered polygon parts, and can be
        a subset of ``source_values`` when empty, invalid, or unsupported source
        geometries were skipped before rendering.
    """
    companion_palette_allowed = True
    coercion_applied = False

    if _is_categorical_dtype(source_values):
        normalized_source_values = _normalize_category_series(source_values)
        normalized_rendered_row_values = _normalize_category_series(rendered_row_values)
        categories = _present_categories(normalized_source_values, source_values.cat.categories)
    elif _is_bool_series(source_values):
        normalized_source_values = _normalize_category_series(source_values)
        normalized_rendered_row_values = _normalize_category_series(rendered_row_values)
        categories = _present_categories(normalized_source_values, [False, True])
    elif _is_exact_binary_integer_series(source_values):
        numeric_source_values = pd.to_numeric(source_values, errors="coerce").astype("Int64")
        numeric_rendered_row_values = pd.to_numeric(rendered_row_values, errors="coerce").astype("Int64")
        normalized_source_values = _normalize_category_series(numeric_source_values)
        normalized_rendered_row_values = _normalize_category_series(numeric_rendered_row_values)
        categories = _present_categories(normalized_source_values, [0, 1])
    elif is_string_like_series(source_values):
        normalized_rendered_row_values, categories = build_string_categorical_values(
            full_values=source_values,
            row_values=rendered_row_values,
            column_name=column_name,
        )
        normalized_source_values = pd.Series(
            [pd.NA if pd.isna(value) else str(value) for value in source_values],
            index=source_values.index,
            name=column_name,
            dtype="object",
        )
        companion_palette_allowed = False
        coercion_applied = True
    else:
        normalized_source_values = _normalize_category_series(source_values)
        normalized_rendered_row_values = _normalize_category_series(rendered_row_values)
        categories = list(pd.unique(normalized_source_values.dropna()))
        companion_palette_allowed = False

    if companion_palette_allowed:
        palette_source, palette = _resolve_shape_categorical_palette(
            shapes_element=shapes_element,
            column_name=column_name,
            source_values=normalized_source_values,
            categories=categories,
        )
    else:
        palette_source = "default_missing"
        palette = default_categorical_palette_for_categories(categories)

    rendered_row_colors = categorical_colors_for_values(
        normalized_rendered_row_values,
        categories=categories,
        palette=palette,
        missing_color=SHAPES_MISSING_BASE_COLOR,
    )
    return (
        ShapesStyleResult(
            value_kind="categorical",
            palette_source=palette_source,
            coercion_applied=coercion_applied,
        ),
        rendered_row_colors,
        normalized_rendered_row_values,
    )


def _build_continuous_shape_style(
    rendered_row_values: pd.Series,
) -> tuple[ShapesStyleResult, pd.Series, pd.Series]:
    numeric_rendered_row_values = pd.to_numeric(rendered_row_values, errors="coerce").astype("float64")
    rendered_row_colors = continuous_colors_for_values(
        numeric_rendered_row_values,
        missing_color=SHAPES_MISSING_BASE_COLOR,
    )
    return (
        ShapesStyleResult(
            value_kind="continuous",
            palette_source=None,
            coercion_applied=False,
        ),
        rendered_row_colors,
        numeric_rendered_row_values,
    )


def _resolve_shape_categorical_palette(
    *,
    shapes_element: gpd.GeoDataFrame,
    column_name: str,
    source_values: pd.Series,
    categories: Sequence[object],
) -> tuple[StyledPaletteSource, list[Any]]:
    colors_column_name = f"{column_name}_colors"
    if colors_column_name not in shapes_element.columns:
        logger.info(f"No `{colors_column_name}` companion color column found; using the default categorical palette.")
        return "default_missing", default_categorical_palette_for_categories(categories)

    companion_colors = shapes_element[colors_column_name]
    color_by_category: dict[object, Any] = {}
    for category in categories:
        category_mask = source_values == category
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


def _apply_rendered_row_colors_to_shapes_layer(
    layer: Shapes,
    rendered_row_colors: pd.Series,
    *,
    fill: bool = False,
) -> None:
    if len(rendered_row_colors) != len(layer.data):
        raise ValueError("Rendered-row colors must contain one color for each rendered napari shape row.")
    layer.face_color = _with_alpha(rendered_row_colors, _styled_shapes_face_alpha(fill))
    layer.edge_color = _with_alpha(rendered_row_colors, SHAPES_EDGE_ALPHA)
    refresh = getattr(layer, "refresh", None)
    if callable(refresh):
        refresh()


def _styled_shapes_face_alpha(fill: bool) -> float:
    return SHAPES_FACE_ALPHA if fill else 0.0


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
        isinstance(value, int | np.integer) and not isinstance(value, bool | np.bool_) for value in non_null.tolist()
    ):
        return False

    normalized_values = {int(value) for value in pd.to_numeric(non_null, errors="coerce").dropna().tolist()}
    return normalized_values == {0, 1}
