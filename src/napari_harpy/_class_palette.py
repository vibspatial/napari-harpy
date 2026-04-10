from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from matplotlib import rcParams
from scanpy.plotting.palettes import default_20, default_28, default_102

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


def extract_class_categories(
    values: pd.Series,
    *,
    column_name: str | None = None,
    unlabeled_class: int = DEFAULT_UNLABELED_CLASS,
) -> list[int]:
    """Return the canonical sorted class ids for a class-valued series."""
    normalized_values = normalize_class_values(
        values,
        column_name=column_name or values.name or "class",
        unlabeled_class=unlabeled_class,
    )
    return sorted({unlabeled_class, *normalized_values.tolist()})


def extract_stored_class_categories(
    values: pd.Series,
    *,
    column_name: str | None = None,
    unlabeled_class: int = DEFAULT_UNLABELED_CLASS,
) -> list[int]:
    """Return category order as stored on the series, normalizing if needed."""
    if isinstance(values.dtype, pd.CategoricalDtype):
        return [int(value) for value in values.cat.categories]

    return extract_class_categories(
        values,
        column_name=column_name or values.name or "class",
        unlabeled_class=unlabeled_class,
    )


def build_class_categorical_series(
    values: pd.Series,
    *,
    column_name: str,
    unlabeled_class: int = DEFAULT_UNLABELED_CLASS,
) -> tuple[pd.Series, list[int]]:
    """Build the canonical categorical series stored in table.obs for a class column."""
    normalized_values = normalize_class_values(values, column_name=column_name, unlabeled_class=unlabeled_class)
    categories = extract_class_categories(
        normalized_values,
        column_name=column_name,
        unlabeled_class=unlabeled_class,
    )
    categorical_series = pd.Series(
        pd.Categorical(normalized_values, categories=categories),
        index=normalized_values.index,
        name=column_name,
    )
    return categorical_series, categories


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


def default_labeled_class_color(class_id: int) -> str:
    """Return the deterministic default color for one positive class id."""
    palette_index = _class_palette_index(class_id)
    palette = _default_labeled_class_colors(palette_index + 1)
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
