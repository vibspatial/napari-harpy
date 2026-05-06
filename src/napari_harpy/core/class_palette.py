from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from loguru import logger
from matplotlib import rcParams
from scanpy.plotting.palettes import default_20, default_28, default_102

if TYPE_CHECKING:
    from anndata import AnnData

DEFAULT_UNLABELED_CLASS = 0
DEFAULT_UNLABELED_COLOR = "#80808099"


def normalize_class_values(
    values: pd.Series,
    *,
    column_name: str,
    unlabeled_class: int = DEFAULT_UNLABELED_CLASS,
) -> pd.Series:
    """Normalize a class column to integer labels with a reserved unlabeled value."""
    numeric_values = pd.to_numeric(values.astype("string"), errors="coerce").fillna(unlabeled_class).astype("int64")
    numeric_values.name = column_name
    return numeric_values


def compute_canonical_class_categories(
    values: pd.Series,
    *,
    column_name: str | None = None,
    unlabeled_class: int = DEFAULT_UNLABELED_CLASS,
) -> list[int]:
    """Return the canonical category order for a class-valued series.

    This helper is for write-time normalization. It always returns the sorted class ids
    implied by the values, with the unlabeled class included, regardless of whether the
    input series already has a categorical dtype or category ordering.
    """
    normalized_values = normalize_class_values(
        values,
        column_name=column_name or values.name or "class",
        unlabeled_class=unlabeled_class,
    )
    return sorted({unlabeled_class, *normalized_values.tolist()})


def read_series_class_categories(
    values: pd.Series,
    *,
    column_name: str | None = None,
    unlabeled_class: int = DEFAULT_UNLABELED_CLASS,
) -> list[int]:
    """Return category order as it is currently represented on the series.

    This helper is for read-time interpretation of existing table state. If the series is
    categorical, its stored category order is preserved so `uns` palettes can be read in
    the same order. If the series is not categorical yet, this falls back to the canonical
    sorted category order.
    """
    if isinstance(values.dtype, pd.CategoricalDtype):
        return [int(value) for value in values.cat.categories]

    return compute_canonical_class_categories(
        values,
        column_name=column_name or values.name or "class",
        unlabeled_class=unlabeled_class,
    )


def default_class_colors(
    categories: Sequence[int],
    *,
    unlabeled_class: int = DEFAULT_UNLABELED_CLASS,
    unlabeled_color: str = DEFAULT_UNLABELED_COLOR,
) -> list[str]:
    """Return the default palette list aligned to the given ordered categories."""
    return [
        unlabeled_color if int(class_id) == unlabeled_class else default_labeled_class_color(int(class_id))
        for class_id in categories
    ]


def default_categorical_colors(length: int) -> list[str]:
    """Return the default generic categorical palette used for viewer overlays."""
    if length <= 0:
        return []
    return _default_labeled_class_colors(length)


def default_labeled_class_color(class_id: int) -> str:
    """Return the deterministic default color for one positive class id."""
    palette_index = _class_palette_index(class_id)
    palette = default_categorical_colors(palette_index + 1)
    return palette[palette_index]


def normalize_color_sequence(value: object) -> list[str] | None:
    """Normalize palette-like inputs into a plain list of color strings."""
    if value is None:
        return None

    if isinstance(value, np.ndarray):
        return [str(item) for item in value.tolist()]

    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]

    return [str(value)]


def stored_palette_to_lookup(
    categories: Sequence[int],
    stored_colors: Sequence[str] | None,
    *,
    unlabeled_class: int = DEFAULT_UNLABELED_CLASS,
    unlabeled_color: str = DEFAULT_UNLABELED_COLOR,
) -> dict[int, str]:
    """Convert an ordered stored palette into a class-id -> color mapping."""
    lookup = {unlabeled_class: unlabeled_color}
    if stored_colors is None:
        return lookup

    for class_id, color in zip(categories, stored_colors[: len(categories)], strict=False):
        lookup[int(class_id)] = str(color)

    return lookup


def backfill_missing_class_colors(
    lookup: dict[int, str],
    categories: Sequence[int],
    *,
    unlabeled_class: int = DEFAULT_UNLABELED_CLASS,
    unlabeled_color: str = DEFAULT_UNLABELED_COLOR,
) -> dict[int, str]:
    """Fill missing class ids with deterministic defaults without overwriting existing colors."""
    filled_lookup = dict(lookup)
    filled_lookup.setdefault(unlabeled_class, unlabeled_color)
    for class_id in sorted(int(value) for value in categories):
        if class_id == unlabeled_class:
            continue
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
    unlabeled_class: int = DEFAULT_UNLABELED_CLASS,
    unlabeled_color: str = DEFAULT_UNLABELED_COLOR,
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
        unlabeled_class=unlabeled_class,
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
        unlabeled_class=unlabeled_class,
        unlabeled_color=unlabeled_color,
    )


def set_class_obs_state(
    table: AnnData,
    values: pd.Series,
    *,
    column_name: str,
    unlabeled_class: int = DEFAULT_UNLABELED_CLASS,
) -> list[int]:
    """Canonicalize the class column stored in `table.obs` and return its categories."""
    normalized_values = normalize_class_values(
        values,
        column_name=column_name,
        unlabeled_class=unlabeled_class,
    )
    categories = compute_canonical_class_categories(
        normalized_values,
        column_name=column_name,
        unlabeled_class=unlabeled_class,
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
    unlabeled_class: int = DEFAULT_UNLABELED_CLASS,
    unlabeled_color: str = DEFAULT_UNLABELED_COLOR,
) -> None:
    """Regenerate and store the palette that corresponds to the canonical class categories."""
    generated_colors = default_class_colors(
        categories,
        unlabeled_class=unlabeled_class,
        unlabeled_color=unlabeled_color,
    )
    existing_colors = normalize_color_sequence(table.uns.get(colors_key))
    if warn_on_palette_overwrite and existing_colors is not None and existing_colors != generated_colors:
        logger.warning(
            f"Overwriting existing `{colors_key}` palette in `table.uns`. "
            f"Current napari-harpy behavior regenerates this palette from `{column_name}` categories."
        )
    table.uns[colors_key] = generated_colors


def _class_palette_index(class_id: int) -> int:
    if class_id == DEFAULT_UNLABELED_CLASS:
        raise ValueError("The unlabeled class does not map to the labeled-class palette.")
    if class_id < DEFAULT_UNLABELED_CLASS:
        raise ValueError("Class ids must be zero or positive integers.")

    return class_id - 1


def _default_labeled_class_colors(length: int) -> list[str]:
    """Return the default categorical palette used by spatialdata-plot/scanpy."""
    if len(rcParams["axes.prop_cycle"].by_key()["color"]) >= length:
        color_cycle = rcParams["axes.prop_cycle"]()
        palette = [next(color_cycle)["color"] for _ in range(length)]
    elif length <= 20:
        palette = list(default_20)
    elif length <= 28:
        palette = list(default_28)
    elif length <= len(default_102):
        palette = list(default_102)
    else:
        palette = ["grey" for _ in range(length)]

    return palette[:length]
