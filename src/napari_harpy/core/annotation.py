from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from napari_harpy.core.class_palette import (
    DEFAULT_UNLABELED_CLASS,
    DEFAULT_UNLABELED_COLOR,
    normalize_class_values,
    set_class_annotation_state,
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
