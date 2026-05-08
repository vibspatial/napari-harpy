from __future__ import annotations

from collections.abc import Sequence

import anndata as ad
import pandas as pd

import napari_harpy.core.annotation as annotation_module
from napari_harpy.core.annotation import USER_CLASS_COLORS_KEY, USER_CLASS_COLUMN, set_user_class_for_rows
from napari_harpy.core.class_palette import default_class_colors


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
            obs[USER_CLASS_COLUMN] = pd.Series(values, index=index, name=USER_CLASS_COLUMN)
        else:
            obs[USER_CLASS_COLUMN] = pd.Categorical(values, categories=categories)

    table = ad.AnnData(obs=obs)
    if colors is not None:
        table.uns[USER_CLASS_COLORS_KEY] = list(colors)
    return table


def _row_mask(table: ad.AnnData, *labels: str) -> pd.Series:
    return pd.Series(table.obs.index.isin(labels), index=table.obs.index)


def test_set_user_class_for_rows_existing_class_updates_only_rows_without_full_normalization(monkeypatch) -> None:
    table = _make_table([0, 1, 0], categories=[0, 1], colors=default_class_colors([0, 1]))

    def fail_full_normalization(*args, **kwargs) -> None:
        del args, kwargs
        raise AssertionError("existing-class row edit should not use full normalization")

    def fail_palette_sync(*args, **kwargs) -> None:
        del args, kwargs
        raise AssertionError("existing-class row edit should not resync an already valid palette")

    monkeypatch.setattr(annotation_module, "_set_user_class_annotation_state", fail_full_normalization)
    monkeypatch.setattr(annotation_module, "sync_class_palette_state", fail_palette_sync)

    set_user_class_for_rows(table, _row_mask(table, "cell_2"), 1)

    assert table.obs[USER_CLASS_COLUMN].tolist() == [0, 1, 1]
    assert list(table.obs[USER_CLASS_COLUMN].cat.categories) == [0, 1]
    assert table.uns[USER_CLASS_COLORS_KEY] == default_class_colors([0, 1])


def test_set_user_class_for_rows_adds_new_category_and_updates_selected_rows_only() -> None:
    table = _make_table([0, 1, 0], categories=[0, 1], colors=default_class_colors([0, 1]))

    set_user_class_for_rows(table, _row_mask(table, "cell_0"), 3)

    assert table.obs[USER_CLASS_COLUMN].tolist() == [3, 1, 0]
    assert list(table.obs[USER_CLASS_COLUMN].cat.categories) == [0, 1, 3]
    assert table.uns[USER_CLASS_COLORS_KEY] == default_class_colors([0, 1, 3])


def test_set_user_class_for_rows_initializes_missing_column() -> None:
    table = _make_table()

    set_user_class_for_rows(table, _row_mask(table, "cell_1"), 3)

    assert table.obs[USER_CLASS_COLUMN].tolist() == [0, 3, 0]
    assert list(table.obs[USER_CLASS_COLUMN].cat.categories) == [0, 3]
    assert table.uns[USER_CLASS_COLORS_KEY] == default_class_colors([0, 3])


def test_set_user_class_for_rows_clearing_only_labeled_object_removes_unused_category() -> None:
    table = _make_table([0, 3, 0], categories=[0, 3], colors=default_class_colors([0, 3]))

    set_user_class_for_rows(table, _row_mask(table, "cell_1"), 0)

    assert table.obs[USER_CLASS_COLUMN].tolist() == [0, 0, 0]
    assert list(table.obs[USER_CLASS_COLUMN].cat.categories) == [0]
    assert table.uns[USER_CLASS_COLORS_KEY] == default_class_colors([0])


def test_set_user_class_for_rows_replaces_category_when_previous_class_becomes_unused() -> None:
    table = _make_table([0, 3, 0], categories=[0, 3], colors=default_class_colors([0, 3]))

    set_user_class_for_rows(table, _row_mask(table, "cell_1"), 4)

    assert table.obs[USER_CLASS_COLUMN].tolist() == [0, 4, 0]
    assert list(table.obs[USER_CLASS_COLUMN].cat.categories) == [0, 4]
    assert table.uns[USER_CLASS_COLORS_KEY] == default_class_colors([0, 4])


def test_set_user_class_for_rows_recovers_non_categorical_state() -> None:
    table = _make_table(["2", "bad", None])

    set_user_class_for_rows(table, _row_mask(table, "cell_1"), 4)

    assert table.obs[USER_CLASS_COLUMN].tolist() == [2, 4, 0]
    assert list(table.obs[USER_CLASS_COLUMN].cat.categories) == [0, 2, 4]
    assert table.uns[USER_CLASS_COLORS_KEY] == default_class_colors([0, 2, 4])


def test_set_user_class_for_rows_resyncs_misaligned_palette() -> None:
    table = _make_table([0, 1, 0], categories=[0, 1], colors=["#ffffffff"])

    set_user_class_for_rows(table, _row_mask(table, "cell_2"), 1)

    assert table.obs[USER_CLASS_COLUMN].tolist() == [0, 1, 1]
    assert list(table.obs[USER_CLASS_COLUMN].cat.categories) == [0, 1]
    assert table.uns[USER_CLASS_COLORS_KEY] == default_class_colors([0, 1])


def test_set_user_class_for_rows_rejects_non_series_row_mask() -> None:
    table = _make_table([0, 1, 0], categories=[0, 1], colors=default_class_colors([0, 1]))

    try:
        set_user_class_for_rows(table, [True, False, False], 2)
    except TypeError as error:
        assert "`rows` must be a boolean pandas Series" in str(error)
    else:  # pragma: no cover
        raise AssertionError("Expected non-Series row mask to be rejected.")


def test_set_user_class_for_rows_rejects_missing_row_mask_values() -> None:
    table = _make_table([0, 1, 0], categories=[0, 1], colors=default_class_colors([0, 1]))
    rows = pd.Series([True, pd.NA, False], index=table.obs.index, dtype="boolean")

    try:
        set_user_class_for_rows(table, rows, 2)
    except ValueError as error:
        assert "`rows` contains missing mask values" in str(error)
    else:  # pragma: no cover
        raise AssertionError("Expected missing row mask value to be rejected.")


def test_set_user_class_for_rows_rejects_non_boolean_row_mask() -> None:
    table = _make_table([0, 1, 0], categories=[0, 1], colors=default_class_colors([0, 1]))
    rows = pd.Series([1, 0, 0], index=table.obs.index)

    try:
        set_user_class_for_rows(table, rows, 2)
    except TypeError as error:
        assert "`rows` must be a boolean mask" in str(error)
    else:  # pragma: no cover
        raise AssertionError("Expected non-boolean row mask to be rejected.")
