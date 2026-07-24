from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba

if TYPE_CHECKING:
    from anndata import AnnData

DEFAULT_NEUTRAL_COLOR = "#DCE8F2CC"

# Maximally distinct palette adapted from:
# https://godsnotwheregodsnot.blogspot.com/2012/09/color-distribution-methodology.html
#
# Black is intentionally excluded because black is commonly used for
# annotation outlines and backgrounds.
GODSNOT_102: tuple[str, ...] = (
    "#FFFF00",
    "#1CE6FF",
    "#FF34FF",
    "#FF4A46",
    "#008941",
    "#006FA6",
    "#A30059",
    "#FFDBE5",
    "#7A4900",
    "#0000A6",
    "#63FFAC",
    "#B79762",
    "#004D43",
    "#8FB0FF",
    "#997D87",
    "#5A0007",
    "#809693",
    "#6A3A4C",
    "#1B4400",
    "#4FC601",
    "#3B5DFF",
    "#4A3B53",
    "#FF2F80",
    "#61615A",
    "#BA0900",
    "#6B7900",
    "#00C2A0",
    "#FFAA92",
    "#FF90C9",
    "#B903AA",
    "#D16100",
    "#DDEFFF",
    "#000035",
    "#7B4F4B",
    "#A1C299",
    "#300018",
    "#0AA6D8",
    "#013349",
    "#00846F",
    "#372101",
    "#FFB500",
    "#C2FFED",
    "#A079BF",
    "#CC0744",
    "#C0B9B2",
    "#C2FF99",
    "#001E09",
    "#00489C",
    "#6F0062",
    "#0CBD66",
    "#EEC3FF",
    "#456D75",
    "#B77B68",
    "#7A87A1",
    "#788D66",
    "#885578",
    "#FAD09F",
    "#FF8A9A",
    "#D157A0",
    "#BEC459",
    "#456648",
    "#0086ED",
    "#886F4C",
    "#34362D",
    "#B4A8BD",
    "#00A6AA",
    "#452C2C",
    "#636375",
    "#A3C8C9",
    "#FF913F",
    "#938A81",
    "#575329",
    "#00FECF",
    "#B05B6F",
    "#8CD0FF",
    "#3B9700",
    "#04F757",
    "#C8A1A1",
    "#1E6E00",
    "#7900D7",
    "#A77500",
    "#6367A9",
    "#A05837",
    "#6B002C",
    "#772600",
    "#D790FF",
    "#9B9700",
    "#549E79",
    "#FFF69F",
    "#201625",
    "#72418F",
    "#BC23FF",
    "#99ADC0",
    "#3A2465",
    "#922329",
    "#5B4534",
    "#FDE8DC",
    "#404E55",
    "#0089A3",
    "#CB7E98",
    "#A4E804",
    "#324E72",
)

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
        for category in values.cat.categories:
            if isinstance(category, (bool, np.bool_)) or not isinstance(category, (int, np.integer)):
                raise ValueError(f"`{column_name}` must contain positive integer class categories.")
            if int(category) <= 0:
                raise ValueError(f"`{column_name}` must contain only positive integer class ids.")
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


def default_class_colors(categories: Sequence[int]) -> list[str]:
    """Return the default palette list aligned to the given ordered categories."""
    return [default_labeled_class_color(int(class_id)) for class_id in categories]


def default_categorical_colors(length: int) -> list[str]:
    """Return cyclic append-stable colors for ordered categorical values."""
    return [default_labeled_class_color(position) for position in range(1, length + 1)]


def default_labeled_class_color(class_id: int) -> str:
    """Return the cyclic deterministic default color for one positive class id."""
    if class_id <= 0:
        raise ValueError("Class ids must be positive integers.")

    return GODSNOT_102[(class_id - 1) % len(GODSNOT_102)]


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


def _is_valid_color(value: str) -> bool:
    try:
        to_rgba(value)
    except (TypeError, ValueError):
        return False
    return True
