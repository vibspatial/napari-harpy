from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from napari_harpy.core.class_palette import (
    backfill_missing_class_colors,
    default_categorical_colors,
    default_class_colors,
    default_labeled_class_color,
    extend_categorical_palette,
    normalize_color_sequence,
    resolve_table_categorical_palette,
    set_class_obs_state,
    stored_palette_to_lookup,
    validate_categorical_palette_source,
)


def test_set_class_obs_state_normalizes_values_and_categories() -> None:
    values = pd.Series([7, pd.NA, 3, 7], index=["a", "b", "c", "d"], dtype="Int64", name="pred_class")
    table = type("DummyTable", (), {"obs": pd.DataFrame(index=values.index)})()

    categories = set_class_obs_state(table, values, column_name="pred_class")
    categorical_series = table.obs["pred_class"]

    assert categorical_series.isna().tolist() == [False, True, False, False]
    assert categorical_series.dropna().astype("int64").tolist() == [7, 3, 7]
    assert categories == [3, 7]
    assert list(categorical_series.cat.categories) == [3, 7]


@pytest.mark.parametrize(
    "values",
    [
        pd.Series([0, 1], dtype="int64"),
        pd.Series(["1", "2"], dtype="string"),
        pd.Series([True, False], dtype="bool"),
    ],
)
def test_set_class_obs_state_rejects_noncanonical_class_values(values: pd.Series) -> None:
    table = type("DummyTable", (), {"obs": pd.DataFrame(index=values.index)})()

    with pytest.raises(ValueError, match="positive integer"):
        set_class_obs_state(table, values, column_name="pred_class")


def test_default_class_colors_are_stable_for_shared_class_ids() -> None:
    user_palette = default_class_colors([1, 3, 7, 9, 21, 24])
    pred_palette = default_class_colors([3, 7])

    assert user_palette[1] == pred_palette[0]
    assert user_palette[2] == pred_palette[1]


@pytest.mark.parametrize("current_length", [10, 20, 28, 102])
def test_default_categorical_colors_are_append_stable_across_palette_thresholds(current_length: int) -> None:
    current = default_categorical_colors(current_length)
    extended = default_categorical_colors(current_length + 1)

    assert extended[:current_length] == current
    assert extended[-1] == default_labeled_class_color(current_length + 1)


def test_default_categorical_colors_match_stable_colors_for_every_palette_family() -> None:
    assert default_categorical_colors(103) == [default_labeled_class_color(position) for position in range(1, 104)]


def test_resolve_table_categorical_palette_preserves_valid_stored_colors_without_mutation() -> None:
    stored_colors = ["#ff0000", "#00ff00"]
    table = SimpleNamespace(uns={"cell_type_colors": stored_colors})

    source, colors = resolve_table_categorical_palette(
        table=table,
        column_name="cell_type",
        categories=("T", "B"),
    )

    assert source == "stored"
    assert colors == stored_colors
    assert table.uns["cell_type_colors"] is stored_colors


@pytest.mark.parametrize(
    ("stored_value", "expected_source"),
    [
        (None, "default_missing"),
        (["#ff0000"], "default_invalid"),
        (["#ff0000", "not-a-color"], "default_invalid"),
    ],
)
def test_resolve_table_categorical_palette_uses_defaults_without_mutation(
    stored_value: object,
    expected_source: str,
) -> None:
    uns = {} if stored_value is None else {"cell_type_colors": stored_value}
    table = SimpleNamespace(uns=uns)
    before = dict(uns)

    source, colors = resolve_table_categorical_palette(
        table=table,
        column_name="cell_type",
        categories=("T", "B"),
    )

    assert source == expected_source
    assert colors == default_categorical_colors(2)
    assert table.uns == before


def test_extend_categorical_palette_preserves_existing_colors_and_appends_stable_defaults() -> None:
    palette = ["#ff0000", "#00ff00"]

    extended = extend_categorical_palette(
        palette,
        current_categories=("T", "B"),
        next_categories=("T", "B", "NK", "myeloid"),
    )

    assert extended == [
        *palette,
        default_labeled_class_color(3),
        default_labeled_class_color(4),
    ]
    assert palette == ["#ff0000", "#00ff00"]


@pytest.mark.parametrize(
    ("palette", "current_categories", "next_categories", "message"),
    [
        (["#ff0000"], ("T", "B"), ("T", "B", "NK"), "length"),
        (["#ff0000", "#00ff00"], ("T", "B"), ("T",), "remove"),
        (["#ff0000", "#00ff00"], ("T", "B"), ("B", "T", "NK"), "prefix"),
        (["#ff0000", "not-a-color"], ("T", "B"), ("T", "B", "NK"), "valid color strings"),
    ],
)
def test_extend_categorical_palette_rejects_invalid_transitions(
    palette: list[str],
    current_categories: tuple[str, ...],
    next_categories: tuple[str, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        extend_categorical_palette(
            palette,
            current_categories=current_categories,
            next_categories=next_categories,
        )


def test_validate_categorical_palette_source_rejects_unknown_value() -> None:
    assert validate_categorical_palette_source("stored") == "stored"
    with pytest.raises(ValueError, match="Invalid categorical palette source"):
        validate_categorical_palette_source("generated")


def test_stored_palette_lookup_backfills_missing_class_ids_without_overwriting_existing_colors() -> None:
    stored_colors = normalize_color_sequence(["#ff0000"])

    lookup = stored_palette_to_lookup([3], stored_colors)
    filled_lookup = backfill_missing_class_colors(lookup, [3, 7])

    assert filled_lookup[3] == "#ff0000"
    assert filled_lookup[7] == default_labeled_class_color(7)
    assert 7 in filled_lookup
    assert filled_lookup[7] != "#ff0000"
