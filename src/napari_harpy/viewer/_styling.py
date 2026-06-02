from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import pandas as pd
from loguru import logger
from matplotlib import colormaps
from matplotlib.colors import to_rgba

from napari_harpy.core.class_palette import default_categorical_colors, normalize_color_sequence

if TYPE_CHECKING:
    from anndata import AnnData

StyledPaletteSource = Literal["stored", "default_missing", "default_invalid"]
STYLED_PALETTE_SOURCES: tuple[StyledPaletteSource, ...] = ("stored", "default_missing", "default_invalid")

MISSING_CATEGORICAL_COLOR = "#80808099"
MISSING_CONTINUOUS_COLOR = "#80808099"
OVERLAY_CONTINUOUS_COLORMAP = "viridis"
STRING_CATEGORICAL_WARNING_MIN_UNIQUE_COUNT = 20
STRING_CATEGORICAL_WARNING_ROW_COUNT_DIVISOR = 100


def validate_styled_palette_source(source: str) -> StyledPaletteSource:
    """Return a validated styled palette source."""
    if source not in STYLED_PALETTE_SOURCES:
        allowed = ", ".join(repr(allowed_source) for allowed_source in STYLED_PALETTE_SOURCES)
        raise ValueError(f"Invalid styled palette source {source!r}. Expected one of: {allowed}.")

    return cast(StyledPaletteSource, source)


def build_string_categorical_values(
    *,
    full_values: pd.Series,
    row_values: pd.Series,
    column_name: str,
) -> tuple[pd.Series, list[object]]:
    """Return viewer-only categorical values for a plain string/object column.

    Parameters
    ----------
    full_values
        Complete source column used for category discovery and cardinality
        warnings. Deriving categories from this complete series keeps the
        temporary categorical palette stable even when the currently rendered
        rows contain only a subset of the source values.
    row_values
        Already-aligned subset or repetition used by the viewer layer, for
        example linked labels table rows for one region or rendered shape rows
        after a ``MultiPolygon`` expands into multiple napari rows.
    column_name
        Source column name used in viewer-facing warning messages.
    """
    normalized_full_values = pd.Series(
        [normalize_string_value(value) for value in full_values],
        index=full_values.index,
        name=column_name,
        dtype="object",
    )
    normalized_row_values = pd.Series(
        [normalize_string_value(value) for value in row_values],
        index=row_values.index,
        name=column_name,
        dtype="object",
    )

    non_missing_values = normalized_full_values.dropna().tolist()
    if has_high_cardinality_string_values(non_missing_values, row_count=len(full_values)):
        unique_count = len({str(value) for value in non_missing_values})
        threshold = string_categorical_warning_threshold(len(full_values))
        logger.warning(
            f"Column `{column_name}` has {unique_count} unique string values across "
            f"{len(full_values)} rows, which exceeds the categorical viewer-coloring threshold of {threshold}. "
            "Harpy will render it with the default categorical palette anyway; "
            "convert the column to pandas categorical dtype to mark this as intentional."
        )
    else:
        logger.warning(
            f"Coercing plain string/object column `{column_name}` to temporary categorical values for viewer coloring."
        )

    categories = list(pd.unique(normalized_full_values.dropna()))
    return normalized_row_values, categories


def string_categorical_warning_threshold(row_count: int) -> int:
    return max(STRING_CATEGORICAL_WARNING_MIN_UNIQUE_COUNT, row_count // STRING_CATEGORICAL_WARNING_ROW_COUNT_DIVISOR)


def has_high_cardinality_string_values(values: Sequence[Any], *, row_count: int) -> bool:
    return len({str(value) for value in values}) > string_categorical_warning_threshold(row_count)


def is_string_like_series(values: pd.Series) -> bool:
    non_null = values.dropna()
    if non_null.empty:
        return False
    return all(_is_string_scalar(value) for value in non_null.tolist())


def normalize_string_value(value: object) -> object:
    if pd.isna(value):
        return pd.NA
    return str(value)


def normalize_category_value(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    return value


def default_categorical_palette_for_categories(categories: Sequence[object]) -> list[str]:
    return default_categorical_colors(len(categories))


def resolve_table_categorical_palette(
    *,
    table: AnnData,
    column_name: str,
    categories: Sequence[object],
) -> tuple[StyledPaletteSource, list[str]]:
    """Resolve a categorical palette stored in ``table.uns`` for viewer coloring."""
    colors_key = f"{column_name}_colors"
    stored_colors = normalize_color_sequence(table.uns.get(colors_key))
    if stored_colors is None:
        logger.info(
            f"No stored `{colors_key}` palette found in `table.uns`; using the default categorical palette for viewer coloring."
        )
        return "default_missing", default_categorical_palette_for_categories(categories)

    if len(stored_colors) != len(categories):
        logger.warning(
            f"Stored `{colors_key}` palette has {len(stored_colors)} colors for {len(categories)} categories; "
            "using the default categorical palette."
        )
        return "default_invalid", default_categorical_palette_for_categories(categories)

    if not all(is_valid_color(color) for color in stored_colors):
        logger.warning(
            f"Stored `{colors_key}` palette contains invalid color values; using the default categorical palette."
        )
        return "default_invalid", default_categorical_palette_for_categories(categories)

    logger.info(f"Using stored `{colors_key}` palette from `table.uns` for viewer coloring.")
    return "stored", list(stored_colors)


def categorical_colors_for_values(
    values: pd.Series,
    *,
    categories: Sequence[object],
    palette: Sequence[str],
    missing_color: Any = MISSING_CATEGORICAL_COLOR,
) -> pd.Series:
    lookup = {normalize_category_value(category): color for category, color in zip(categories, palette, strict=False)}
    colors = {}
    for index, value in values.items():
        if pd.isna(value):
            colors[index] = missing_color
            continue
        colors[index] = lookup.get(normalize_category_value(value), missing_color)
    return pd.Series(colors, index=values.index, dtype="object")


def continuous_colors_for_values(
    values: pd.Series,
    *,
    missing_color: Any = MISSING_CONTINUOUS_COLOR,
    colormap_name: str = OVERLAY_CONTINUOUS_COLORMAP,
) -> pd.Series:
    colors = {}
    non_missing = values.dropna()
    if non_missing.empty:
        for index in values.index:
            colors[index] = missing_color
        return pd.Series(colors, index=values.index, dtype="object")

    cmap = colormaps[colormap_name]
    min_value = float(non_missing.min())
    max_value = float(non_missing.max())
    if max_value == min_value:
        normalized_values = dict.fromkeys(non_missing.index, 0.5)
    else:
        normalized_values = {
            index: float((value - min_value) / (max_value - min_value)) for index, value in non_missing.items()
        }

    for index in values.index:
        value = values.at[index]
        if pd.isna(value):
            colors[index] = missing_color
        else:
            colors[index] = cmap(float(np.clip(normalized_values[index], 0.0, 1.0)))
    return pd.Series(colors, index=values.index, dtype="object")


def is_valid_color(value: str) -> bool:
    try:
        to_rgba(value)
    except (TypeError, ValueError):
        return False
    return True


def _is_string_scalar(value: object) -> bool:
    return isinstance(value, (str, bytes, np.str_, np.bytes_))
