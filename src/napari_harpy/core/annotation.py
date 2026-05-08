from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from napari_harpy.core.class_palette import (
    DEFAULT_UNLABELED_CLASS,
    DEFAULT_UNLABELED_COLOR,
    default_class_colors,
    normalize_class_values,
    normalize_color_sequence,
    set_class_annotation_state,
    sync_class_palette_state,
)

if TYPE_CHECKING:
    from anndata import AnnData

USER_CLASS_COLUMN = "user_class"
USER_CLASS_COLORS_KEY = f"{USER_CLASS_COLUMN}_colors"
UNLABELED_CLASS = DEFAULT_UNLABELED_CLASS
UNLABELED_COLOR = DEFAULT_UNLABELED_COLOR


def _to_user_class_values(values: pd.Series) -> pd.Series:
    return normalize_class_values(values, column_name=USER_CLASS_COLUMN, unlabeled_class=UNLABELED_CLASS)


def _set_user_class_annotation_state(table: AnnData, values: pd.Series) -> None:
    set_class_annotation_state(
        table,
        values,
        column_name=USER_CLASS_COLUMN,
        colors_key=USER_CLASS_COLORS_KEY,
        warn_on_palette_overwrite=False,
        unlabeled_class=UNLABELED_CLASS,
        unlabeled_color=UNLABELED_COLOR,
    )


def set_user_class_for_rows(table: AnnData, rows: pd.Series, class_id: int) -> None:
    """Assign one user class to selected table rows without rewriting valid columns."""
    class_id = int(class_id)
    if class_id < UNLABELED_CLASS:
        raise ValueError("Class ids must be zero or positive integers.")

    row_mask = _coerce_row_mask(rows, table.obs.index)
    if not bool(row_mask.any()):
        return

    if USER_CLASS_COLUMN not in table.obs:
        _initialize_user_class_column(table, initial_class_id=class_id)
    else:
        categories = _valid_user_class_categories(table.obs[USER_CLASS_COLUMN])
        if categories is None:
            _normalize_user_class_column(table)

    user_class = table.obs[USER_CLASS_COLUMN]
    categories = _valid_user_class_categories(user_class)
    if categories is None:  # pragma: no cover - defensive: normalization above should recover.
        _normalize_user_class_column(table)
        user_class = table.obs[USER_CLASS_COLUMN]
        categories = _valid_user_class_categories(user_class)
        if categories is None:
            raise RuntimeError("Unable to normalize `user_class` into a categorical class column.")

    selected_values = user_class.loc[row_mask]
    previous_class_ids = {int(value) for value in selected_values.dropna().unique()}

    categories_changed = False
    if class_id not in categories:
        categories = sorted({*categories, class_id})
        table.obs[USER_CLASS_COLUMN] = user_class.cat.set_categories(categories)
        user_class = table.obs[USER_CLASS_COLUMN]
        categories_changed = True

    table.obs.loc[row_mask, USER_CLASS_COLUMN] = class_id

    old_classes_may_be_unused = any(
        previous_class_id not in {UNLABELED_CLASS, class_id} for previous_class_id in previous_class_ids
    )
    if old_classes_may_be_unused:
        user_class = table.obs[USER_CLASS_COLUMN]
        used_categories = _used_user_class_categories(user_class)
        if used_categories != categories:
            categories = used_categories
            table.obs[USER_CLASS_COLUMN] = user_class.cat.set_categories(categories)
            categories_changed = True

    if categories_changed or not _user_class_palette_matches(table, categories):
        _sync_user_class_palette_state(table, categories)


def _coerce_row_mask(rows: pd.Series, index: pd.Index) -> pd.Series:
    if not isinstance(rows, pd.Series):
        raise TypeError("`rows` must be a boolean pandas Series aligned to table.obs.")

    row_mask = rows.reindex(index, fill_value=False)
    if row_mask.isna().any():
        raise ValueError("`rows` contains missing mask values.")

    if row_mask.dtype != bool and str(row_mask.dtype) != "boolean":
        raise TypeError("`rows` must be a boolean mask.")

    return row_mask.astype(bool)


def _normalize_user_class_column(table: AnnData) -> None:
    values = _to_user_class_values(table.obs[USER_CLASS_COLUMN])
    _set_user_class_annotation_state(table, values)


def _initialize_user_class_column(table: AnnData, *, initial_class_id: int) -> None:
    categories = sorted({UNLABELED_CLASS, int(initial_class_id)})
    values = pd.Series(
        pd.Categorical([UNLABELED_CLASS] * len(table.obs), categories=categories),
        index=table.obs.index,
        name=USER_CLASS_COLUMN,
    )
    table.obs[USER_CLASS_COLUMN] = values
    _sync_user_class_palette_state(table, categories)


def _valid_user_class_categories(values: pd.Series) -> list[int] | None:
    if not isinstance(values.dtype, pd.CategoricalDtype):
        return None

    categories: list[int] = []
    for category in values.cat.categories:
        try:
            class_id = int(category)
        except (TypeError, ValueError):
            return None
        if class_id < UNLABELED_CLASS or category != class_id:
            return None
        categories.append(class_id)

    if categories != sorted(categories):
        return None
    if len(categories) != len(set(categories)):
        return None
    if UNLABELED_CLASS not in categories:
        return None
    if bool((values.cat.codes.to_numpy(copy=False) < 0).any()):
        return None

    return categories


def _used_user_class_categories(values: pd.Series) -> list[int]:
    categories = [int(category) for category in values.cat.categories]
    codes = values.cat.codes.to_numpy(copy=False)
    used_codes = {int(code) for code in codes if int(code) >= 0}
    used_categories = {categories[code] for code in used_codes}
    return sorted({UNLABELED_CLASS, *used_categories})


def _user_class_palette_matches(table: AnnData, categories: list[int]) -> bool:
    expected_colors = default_class_colors(
        categories,
        unlabeled_class=UNLABELED_CLASS,
        unlabeled_color=UNLABELED_COLOR,
    )
    return normalize_color_sequence(table.uns.get(USER_CLASS_COLORS_KEY)) == expected_colors


def _sync_user_class_palette_state(table: AnnData, categories: list[int]) -> None:
    sync_class_palette_state(
        table,
        categories=categories,
        column_name=USER_CLASS_COLUMN,
        colors_key=USER_CLASS_COLORS_KEY,
        warn_on_palette_overwrite=False,
        unlabeled_class=UNLABELED_CLASS,
        unlabeled_color=UNLABELED_COLOR,
    )
