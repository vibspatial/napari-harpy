from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
import pandas as pd
from loguru import logger
from matplotlib import rcParams
from matplotlib.colors import to_rgba
from scanpy.plotting.palettes import default_20, default_28, default_102

if TYPE_CHECKING:
    from anndata import AnnData

DEFAULT_NEUTRAL_COLOR = "#DCE8F2CC"
CategoricalPaletteSource = Literal["stored", "default_missing", "default_invalid"]
CATEGORICAL_PALETTE_SOURCES: tuple[CategoricalPaletteSource, ...] = (
    "stored",
    "default_missing",
    "default_invalid",
)


def validate_categorical_palette_source(source: str) -> CategoricalPaletteSource:
    """Return a validated categorical-palette source."""
    if source not in CATEGORICAL_PALETTE_SOURCES:
        allowed = ", ".join(repr(allowed_source) for allowed_source in CATEGORICAL_PALETTE_SOURCES)
        raise ValueError(f"Invalid categorical palette source {source!r}. Expected one of: {allowed}.")

    return cast(CategoricalPaletteSource, source)


def normalize_class_values(
    values: pd.Series,
    *,
    column_name: str,
) -> pd.Series:
    """Normalize positive integer class ids while preserving missing values."""
    if isinstance(values.dtype, pd.CategoricalDtype):
        _validate_positive_class_categories(values.cat.categories, column_name=column_name)
    elif pd.api.types.is_bool_dtype(values.dtype) or not pd.api.types.is_integer_dtype(values.dtype):
        raise ValueError(f"`{column_name}` must contain positive integer class ids or missing values.")

    normalized_values = pd.Series(
        pd.array(values, dtype="Int64"),
        index=values.index,
        name=column_name,
    )
    if bool((normalized_values.dropna() <= 0).any()):
        raise ValueError(f"`{column_name}` must contain only positive integer class ids.")
    return normalized_values


def compute_canonical_class_categories(
    values: pd.Series,
    *,
    column_name: str | None = None,
) -> list[int]:
    """Return the canonical category order for a class-valued series.

    This helper is for write-time normalization. It always returns the sorted class ids
    implied by the non-missing values, regardless of whether the input series already
    has a categorical dtype or category ordering.
    """
    normalized_values = normalize_class_values(
        values,
        column_name=column_name or values.name or "class",
    )
    return sorted(int(value) for value in normalized_values.dropna().unique())


def read_series_class_categories(
    values: pd.Series,
    *,
    column_name: str | None = None,
) -> list[int]:
    """Return category order as it is currently represented on the series.

    This helper is for read-time interpretation of existing table state. If the series is
    categorical, its stored category order is preserved so `uns` palettes can be read in
    the same order. If the series is not categorical yet, this falls back to the canonical
    sorted category order.
    """
    if isinstance(values.dtype, pd.CategoricalDtype):
        return _validate_positive_class_categories(
            values.cat.categories,
            column_name=column_name or values.name or "class",
        )

    return compute_canonical_class_categories(
        values,
        column_name=column_name or values.name or "class",
    )


def default_class_colors(categories: Sequence[int]) -> list[str]:
    """Return the default palette list aligned to the given ordered categories."""
    return [default_labeled_class_color(int(class_id)) for class_id in categories]


def default_categorical_colors(length: int) -> list[str]:
    """Return append-stable default colors for ordered categorical values."""
    if length <= 0:
        return []

    color_cycle = rcParams["axes.prop_cycle"].by_key()["color"]
    colors = list(color_cycle[:length])
    for base_palette in (default_20, default_28, default_102):
        stop = min(length, len(base_palette))
        if len(colors) < stop:
            colors.extend(base_palette[len(colors) : stop])
    if len(colors) < length:
        colors.extend(["grey"] * (length - len(colors)))
    return colors


def default_labeled_class_color(class_id: int) -> str:
    """Return the deterministic default color for one positive class id."""
    palette_index = _class_palette_index(class_id)
    palette = _default_labeled_class_palette(palette_index + 1)
    return "grey" if palette is None else palette[palette_index]


def normalize_color_sequence(value: object) -> list[str] | None:
    """Normalize palette-like inputs into a plain list of color strings."""
    if value is None:
        return None

    if isinstance(value, np.ndarray):
        return [str(item) for item in value.tolist()]

    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]

    return [str(value)]


def resolve_table_categorical_palette(
    *,
    table: AnnData,
    column_name: str,
    categories: Sequence[object],
) -> tuple[CategoricalPaletteSource, list[str]]:
    """Resolve one standard AnnData categorical palette without mutation."""
    stored_colors = normalize_color_sequence(table.uns.get(f"{column_name}_colors"))
    if stored_colors is None:
        return "default_missing", default_categorical_colors(len(categories))
    if len(stored_colors) != len(categories) or not all(_is_valid_color(color) for color in stored_colors):
        return "default_invalid", default_categorical_colors(len(categories))
    return "stored", list(stored_colors)


def extend_categorical_palette(
    palette: Sequence[str],
    *,
    current_categories: Sequence[object],
    next_categories: Sequence[object],
) -> list[str]:
    """Append stable colors for an append-only categorical transition."""
    current = tuple(current_categories)
    next_ = tuple(next_categories)
    colors = list(palette)

    if len(colors) != len(current):
        raise ValueError("Palette length must match the current category count.")
    if len(next_) < len(current):
        raise ValueError("Categorical palette extension cannot remove categories.")
    if next_[: len(current)] != current:
        raise ValueError("Next categories must preserve the complete current category prefix.")
    if any(not isinstance(color, str) or not _is_valid_color(color) for color in colors):
        raise ValueError("Categorical palette colors must be valid color strings.")

    colors.extend(default_labeled_class_color(position + 1) for position in range(len(current), len(next_)))
    return colors


def stored_palette_to_lookup(
    categories: Sequence[int],
    stored_colors: Sequence[str] | None,
) -> dict[int, str]:
    """Convert an ordered stored palette into a class-id -> color mapping."""
    lookup: dict[int, str] = {}
    if stored_colors is None:
        return lookup

    for class_id, color in zip(categories, stored_colors[: len(categories)], strict=False):
        lookup[int(class_id)] = str(color)

    return lookup


def backfill_missing_class_colors(
    lookup: dict[int, str],
    categories: Sequence[int],
) -> dict[int, str]:
    """Fill missing class ids with deterministic defaults without overwriting existing colors."""
    filled_lookup = dict(lookup)
    for class_id in sorted(int(value) for value in categories):
        filled_lookup.setdefault(class_id, default_labeled_class_color(class_id))

    return filled_lookup


def set_class_annotation_state(
    table: AnnData,
    values: pd.Series,
    *,
    column_name: str,
    colors_key: str | None = None,
    keep_colors: bool = True,
    warn_on_palette_overwrite: bool = True,
) -> None:
    """Normalize a class column in `table.obs` and explicitly sync its palette in `table.uns`.

    This is the high-level mutating entry point for generic class annotation state. It first
    canonicalizes the categorical values stored in `table.obs[column_name]`, then, when
    `keep_colors` is enabled, it explicitly regenerates and writes the corresponding
    `table.uns[colors_key]` palette via `sync_class_palette_state(...)`.
    """
    categories = set_class_obs_state(
        table,
        values,
        column_name=column_name,
    )

    if not keep_colors or colors_key is None:
        if colors_key is not None:
            drop_class_palette_state(table, colors_key=colors_key)
        return

    sync_class_palette_state(
        table,
        categories=categories,
        column_name=column_name,
        colors_key=colors_key,
        warn_on_palette_overwrite=warn_on_palette_overwrite,
    )


def set_class_obs_state(
    table: AnnData,
    values: pd.Series,
    *,
    column_name: str,
) -> list[int]:
    """Canonicalize the class column stored in `table.obs` and return its categories."""
    normalized_values = normalize_class_values(
        values,
        column_name=column_name,
    )
    categories = compute_canonical_class_categories(
        normalized_values,
        column_name=column_name,
    )
    table.obs[column_name] = pd.Series(
        pd.Categorical(normalized_values, categories=categories),
        index=normalized_values.index,
        name=column_name,
    )
    return categories


def drop_class_palette_state(table: AnnData, *, colors_key: str) -> None:
    """Remove the stored palette for one class column without mutating other `uns` entries."""
    if colors_key not in table.uns:
        return

    table.uns = {key: value for key, value in table.uns.items() if key != colors_key}


def sync_class_palette_state(
    table: AnnData,
    *,
    categories: list[int],
    column_name: str,
    colors_key: str,
    warn_on_palette_overwrite: bool,
) -> None:
    """Regenerate and store the palette that corresponds to the canonical class categories."""
    generated_colors = default_class_colors(categories)
    existing_colors = normalize_color_sequence(table.uns.get(colors_key))
    if warn_on_palette_overwrite and existing_colors is not None and existing_colors != generated_colors:
        logger.warning(
            f"Overwriting existing `{colors_key}` palette in `table.uns`. "
            f"Current napari-harpy behavior regenerates this palette from `{column_name}` categories."
        )
    table.uns[colors_key] = generated_colors


def _class_palette_index(class_id: int) -> int:
    if class_id <= 0:
        raise ValueError("Class ids must be positive integers.")

    return class_id - 1


def _validate_positive_class_categories(categories: pd.Index, *, column_name: str) -> list[int]:
    validated: list[int] = []
    for category in categories:
        if isinstance(category, (bool, np.bool_)) or not isinstance(category, (int, np.integer)):
            raise ValueError(f"`{column_name}` must contain positive integer class categories.")
        class_id = int(category)
        if class_id <= 0:
            raise ValueError(f"`{column_name}` must contain only positive integer class ids.")
        validated.append(class_id)
    return validated


def _is_valid_color(value: str) -> bool:
    try:
        to_rgba(value)
    except (TypeError, ValueError):
        return False
    return True


def _default_labeled_class_palette(length: int) -> Sequence[str] | None:
    """Return the base palette containing the requested one-based position."""
    color_cycle = rcParams["axes.prop_cycle"].by_key()["color"]
    if len(color_cycle) >= length:
        return color_cycle
    if length <= 20:
        return default_20
    if length <= 28:
        return default_28
    if length <= len(default_102):
        return default_102
    return None
