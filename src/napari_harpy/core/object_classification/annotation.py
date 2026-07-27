"""Object Classification user-annotation state."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from napari_harpy.core.class_palette import (
    default_categorical_colors,
    extend_categorical_palette,
    normalize_class_values,
    normalize_color_sequence,
    resolve_table_categorical_palette,
)

if TYPE_CHECKING:
    from anndata import AnnData

USER_CLASS_COLUMN = "user_class"
USER_CLASS_COLORS_KEY = f"{USER_CLASS_COLUMN}_colors"


@dataclass(frozen=True)
class UserClassStateChange:
    """Describe the effective user-class column and palette mutations."""

    user_class_changed: bool
    palette_changed: bool

    @property
    def changed(self) -> bool:
        return self.user_class_changed or self.palette_changed


def _to_user_class_values(values: pd.Series) -> pd.Series:
    return normalize_class_values(values, column_name=USER_CLASS_COLUMN)


def set_user_class_for_rows(
    table: AnnData,
    rows: pd.Series,
    class_id: int | None,
) -> UserClassStateChange:
    """Assign or clear a user class for selected observation rows.

    Parameters
    ----------
    table
        AnnData table whose in-memory ``user_class`` column and companion
        ``user_class_colors`` palette may be mutated.
    rows
        Boolean Series selecting rows by ``table.obs`` index. The mask is
        aligned to the current observation index before assignment.
    class_id
        Positive non-boolean integer to assign, or ``None`` to remove the
        annotation by assigning ``pd.NA``.

    Returns
    -------
    UserClassStateChange
        Flags identifying whether the ``user_class`` observation column and
        its stored palette changed.

    Raises
    ------
    TypeError
        If ``rows`` is not a Boolean pandas Series.
    ValueError
        If ``class_id`` is not a positive integer, the row mask contains
        missing values, or an existing ``user_class`` column violates the
        positive-integer categorical contract.

    Notes
    -----
    This function mutates only the in-memory AnnData table. Its caller owns
    dirty-state publication and persistence. An empty selection, clearing an
    absent column, or assigning values already present is a no-op.
    """
    if class_id is not None:
        if isinstance(class_id, (bool, np.bool_)) or not isinstance(class_id, Integral) or class_id <= 0:
            raise ValueError("Class ids must be positive integers.")
        class_id = int(class_id)

    row_mask = _coerce_row_mask(rows, table.obs.index)
    if not bool(row_mask.any()):
        return UserClassStateChange(
            user_class_changed=False,
            palette_changed=False,
        )

    if USER_CLASS_COLUMN not in table.obs:
        if class_id is None:
            return UserClassStateChange(
                user_class_changed=False,
                palette_changed=False,
            )

        replacement = pd.Series(
            pd.Categorical([pd.NA] * table.n_obs, categories=[class_id]),
            index=table.obs.index,
            name=USER_CLASS_COLUMN,
        )
        replacement.loc[row_mask] = class_id
        table.obs[USER_CLASS_COLUMN] = replacement
        table.uns[USER_CLASS_COLORS_KEY] = default_categorical_colors(1)
        return UserClassStateChange(
            user_class_changed=True,
            palette_changed=True,
        )

    user_class = table.obs[USER_CLASS_COLUMN]
    categories = _read_user_class_categories(user_class)

    selected_values = user_class.loc[row_mask]
    if class_id is None and bool(selected_values.isna().all()):
        return UserClassStateChange(
            user_class_changed=False,
            palette_changed=False,
        )
    if class_id is not None and bool(selected_values.eq(class_id).all()):
        return UserClassStateChange(
            user_class_changed=False,
            palette_changed=False,
        )

    _, resolved_palette = resolve_table_categorical_palette(
        table=table,
        column_name=USER_CLASS_COLUMN,
        categories=categories,
    )
    next_categories = list(categories)
    replacement = user_class.copy()
    if class_id is not None and class_id not in categories:
        next_categories.append(class_id)
        replacement = replacement.cat.add_categories([class_id])

    replacement.loc[row_mask] = pd.NA if class_id is None else class_id
    next_palette = extend_categorical_palette(
        resolved_palette,
        current_categories=categories,
        next_categories=next_categories,
    )
    stored_palette = normalize_color_sequence(table.uns.get(USER_CLASS_COLORS_KEY))
    palette_changed = stored_palette != next_palette

    table.obs[USER_CLASS_COLUMN] = replacement
    if palette_changed:
        table.uns[USER_CLASS_COLORS_KEY] = next_palette
    return UserClassStateChange(
        user_class_changed=True,
        palette_changed=palette_changed,
    )


def _coerce_row_mask(rows: pd.Series, index: pd.Index) -> pd.Series:
    if not isinstance(rows, pd.Series):
        raise TypeError("`rows` must be a boolean pandas Series aligned to table.obs.")

    row_mask = rows.reindex(index, fill_value=False)
    if row_mask.isna().any():
        raise ValueError("`rows` contains missing mask values.")

    if row_mask.dtype != bool and str(row_mask.dtype) != "boolean":
        raise TypeError("`rows` must be a boolean mask.")

    return row_mask.astype(bool)


def _read_user_class_categories(values: pd.Series) -> list[int]:
    if not isinstance(values.dtype, pd.CategoricalDtype):
        raise ValueError(
            f"`{USER_CLASS_COLUMN}` must use a categorical dtype with positive integer categories before annotation."
        )

    categories: list[int] = []
    for category in values.cat.categories:
        if (
            isinstance(category, (bool, np.bool_))
            or not isinstance(category, (int, np.integer))
            or int(category) <= 0
        ):
            raise ValueError(
                f"`{USER_CLASS_COLUMN}` must use a categorical dtype with positive integer categories before "
                "annotation."
            )
        categories.append(int(category))

    return categories
