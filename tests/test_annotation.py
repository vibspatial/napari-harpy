from __future__ import annotations

from collections.abc import Sequence

import anndata as ad
import pandas as pd
import pytest

from napari_harpy.core.class_palette import default_categorical_colors
from napari_harpy.core.object_classification.annotation import (
    USER_CLASS_COLORS_KEY,
    USER_CLASS_COLUMN,
    UserClassStateChange,
    set_user_class_for_rows,
)


def _make_table(
    values: Sequence[int] | Sequence[str | None] | None = None,
    *,
    categories: Sequence[int] | None = None,
    colors: Sequence[str] | None = None,
) -> ad.AnnData:
    index = [f"cell_{index}" for index in range(3)]
    obs = pd.DataFrame(index=index)
    if values is not None:
        if categories is None:
            data = values.array if isinstance(values, pd.Series) else values
            obs[USER_CLASS_COLUMN] = pd.Series(data, index=index, name=USER_CLASS_COLUMN)
        else:
            obs[USER_CLASS_COLUMN] = pd.Categorical(values, categories=categories)

    table = ad.AnnData(obs=obs)
    if colors is not None:
        table.uns[USER_CLASS_COLORS_KEY] = list(colors)
    return table


def _row_mask(table: ad.AnnData, *labels: str) -> pd.Series:
    return pd.Series(table.obs.index.isin(labels), index=table.obs.index)


def _assert_user_classes(table: ad.AnnData, expected: list[int | None], categories: list[int]) -> None:
    values = table.obs[USER_CLASS_COLUMN]
    assert values.isna().tolist() == [value is None for value in expected]
    assert [None if pd.isna(value) else int(value) for value in values] == expected
    assert list(values.cat.categories) == categories


def test_set_user_class_for_rows_existing_class_preserves_valid_palette() -> None:
    colors = ["#123456"]
    table = _make_table([pd.NA, 1, pd.NA], categories=[1], colors=colors)

    state_change = set_user_class_for_rows(table, _row_mask(table, "cell_2"), 1)

    _assert_user_classes(table, [None, 1, 1], [1])
    assert table.uns[USER_CLASS_COLORS_KEY] == colors
    assert state_change == UserClassStateChange(
        user_class_changed=True,
        palette_changed=False,
    )


def test_set_user_class_for_rows_appends_new_category_and_preserves_existing_color() -> None:
    table = _make_table([pd.NA, 1, pd.NA], categories=[1], colors=["#123456"])

    state_change = set_user_class_for_rows(table, _row_mask(table, "cell_0"), 3)

    _assert_user_classes(table, [3, 1, None], [1, 3])
    assert table.uns[USER_CLASS_COLORS_KEY] == ["#123456", default_categorical_colors(2)[1]]
    assert state_change == UserClassStateChange(
        user_class_changed=True,
        palette_changed=True,
    )


def test_set_user_class_for_rows_preserves_nonascending_category_order() -> None:
    colors = ["#123456", "#654321"]
    table = _make_table([3, 1, pd.NA], categories=[3, 1], colors=colors)

    set_user_class_for_rows(table, _row_mask(table, "cell_2"), 2)

    _assert_user_classes(table, [3, 1, 2], [3, 1, 2])
    assert table.uns[USER_CLASS_COLORS_KEY] == [*colors, default_categorical_colors(3)[2]]


def test_set_user_class_for_rows_same_value_is_a_no_op() -> None:
    colors = ["#123456"]
    table = _make_table([1, pd.NA, pd.NA], categories=[1], colors=colors)
    previous = table.obs[USER_CLASS_COLUMN]

    state_change = set_user_class_for_rows(table, _row_mask(table, "cell_0"), 1)

    assert table.obs[USER_CLASS_COLUMN] is previous
    assert table.uns[USER_CLASS_COLORS_KEY] == colors
    assert state_change == UserClassStateChange(
        user_class_changed=False,
        palette_changed=False,
    )


def test_set_user_class_for_rows_initializes_missing_column() -> None:
    table = _make_table()

    state_change = set_user_class_for_rows(table, _row_mask(table, "cell_1"), 3)

    _assert_user_classes(table, [None, 3, None], [3])
    assert table.uns[USER_CLASS_COLORS_KEY] == default_categorical_colors(1)
    assert state_change == UserClassStateChange(
        user_class_changed=True,
        palette_changed=True,
    )


def test_set_user_class_for_rows_clearing_only_labeled_object_retains_vocabulary() -> None:
    colors = ["#123456"]
    table = _make_table([pd.NA, 3, pd.NA], categories=[3], colors=colors)

    state_change = set_user_class_for_rows(table, _row_mask(table, "cell_1"), None)

    _assert_user_classes(table, [None, None, None], [3])
    assert table.uns[USER_CLASS_COLORS_KEY] == colors
    assert state_change == UserClassStateChange(
        user_class_changed=True,
        palette_changed=False,
    )


def test_set_user_class_for_rows_retains_unused_category_when_replacing_class() -> None:
    table = _make_table([pd.NA, 3, pd.NA], categories=[3], colors=["#123456"])

    set_user_class_for_rows(table, _row_mask(table, "cell_1"), 4)

    _assert_user_classes(table, [None, 4, None], [3, 4])
    assert table.uns[USER_CLASS_COLORS_KEY] == ["#123456", default_categorical_colors(2)[1]]


def test_set_user_class_for_rows_rejects_non_categorical_state() -> None:
    table = _make_table(pd.Series([2, pd.NA, pd.NA], dtype="Int64"))

    with pytest.raises(ValueError, match="categorical dtype"):
        set_user_class_for_rows(table, _row_mask(table, "cell_1"), 4)


def test_set_user_class_for_rows_rejects_legacy_class_zero() -> None:
    table = _make_table([0, 1, 0], categories=[0, 1])

    with pytest.raises(ValueError, match="positive integer"):
        set_user_class_for_rows(table, _row_mask(table, "cell_2"), 1)


def test_set_user_class_for_rows_resyncs_misaligned_palette() -> None:
    table = _make_table([pd.NA, 1, pd.NA], categories=[1], colors=["#ffffffff", "#000000ff"])

    state_change = set_user_class_for_rows(table, _row_mask(table, "cell_2"), 1)

    _assert_user_classes(table, [None, 1, 1], [1])
    assert table.uns[USER_CLASS_COLORS_KEY] == default_categorical_colors(1)
    assert state_change == UserClassStateChange(
        user_class_changed=True,
        palette_changed=True,
    )


def test_set_user_class_for_rows_rejects_non_series_row_mask() -> None:
    table = _make_table([pd.NA, 1, pd.NA], categories=[1], colors=default_categorical_colors(1))

    try:
        set_user_class_for_rows(table, [True, False, False], 2)
    except TypeError as error:
        assert "`rows` must be a boolean pandas Series" in str(error)
    else:  # pragma: no cover
        raise AssertionError("Expected non-Series row mask to be rejected.")


def test_set_user_class_for_rows_rejects_missing_row_mask_values() -> None:
    table = _make_table([pd.NA, 1, pd.NA], categories=[1], colors=default_categorical_colors(1))
    rows = pd.Series([True, pd.NA, False], index=table.obs.index, dtype="boolean")

    try:
        set_user_class_for_rows(table, rows, 2)
    except ValueError as error:
        assert "`rows` contains missing mask values" in str(error)
    else:  # pragma: no cover
        raise AssertionError("Expected missing row mask value to be rejected.")


def test_set_user_class_for_rows_rejects_non_boolean_row_mask() -> None:
    table = _make_table([pd.NA, 1, pd.NA], categories=[1], colors=default_categorical_colors(1))
    rows = pd.Series([1, 0, 0], index=table.obs.index)

    try:
        set_user_class_for_rows(table, rows, 2)
    except TypeError as error:
        assert "`rows` must be a boolean mask" in str(error)
    else:  # pragma: no cover
        raise AssertionError("Expected non-boolean row mask to be rejected.")
