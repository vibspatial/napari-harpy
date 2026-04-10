from __future__ import annotations

import pandas as pd

from napari_harpy._class_palette import (
    backfill_missing_class_colors,
    build_class_categorical_series,
    default_class_colors,
    normalize_color_sequence,
    stored_palette_to_lookup,
)


def test_build_class_categorical_series_normalizes_values_and_categories() -> None:
    values = pd.Series(["7", None, "3", "7"], index=["a", "b", "c", "d"], name="pred_class")

    categorical_series, categories = build_class_categorical_series(values, column_name="pred_class")

    assert list(categorical_series.astype("int64")) == [7, 0, 3, 7]
    assert categories == [0, 3, 7]
    assert list(categorical_series.cat.categories) == [0, 3, 7]


def test_default_class_colors_are_stable_for_shared_class_ids() -> None:
    user_palette = default_class_colors([0, 1, 3, 7, 9, 21, 24])
    pred_palette = default_class_colors([0, 3, 7])

    assert user_palette[2] == pred_palette[1]
    assert user_palette[3] == pred_palette[2]


def test_stored_palette_lookup_backfills_missing_class_ids_without_overwriting_existing_colors() -> None:
    stored_colors = normalize_color_sequence(["#80808099", "#ff0000"])

    lookup = stored_palette_to_lookup([0, 3], stored_colors)
    filled_lookup = backfill_missing_class_colors(lookup, [0, 3, 7])

    assert filled_lookup[0] == "#80808099"
    assert filled_lookup[3] == "#ff0000"
    assert 7 in filled_lookup
    assert filled_lookup[7] != "#ff0000"
