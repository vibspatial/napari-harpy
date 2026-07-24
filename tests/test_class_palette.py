from __future__ import annotations

from types import SimpleNamespace

import pytest

from napari_harpy.core.class_palette import (
    GODSNOT_102,
    default_categorical_colors,
    default_class_colors,
    default_labeled_class_color,
    extend_categorical_palette,
    resolve_table_categorical_palette,
    validate_categorical_palette_source,
)


def test_godsnot_palette_is_explicit_and_excludes_black() -> None:
    assert len(GODSNOT_102) == 102
    assert GODSNOT_102[0] == "#FFFF00"
    assert "#000000" not in GODSNOT_102


def test_default_class_colors_are_stable_for_shared_class_ids() -> None:
    user_palette = default_class_colors([1, 3, 7, 9, 21, 24])
    pred_palette = default_class_colors([3, 7])

    assert user_palette[1] == pred_palette[0]
    assert user_palette[2] == pred_palette[1]


@pytest.mark.parametrize("current_length", [1, 10, 101, 102, 103])
def test_default_categorical_colors_are_append_stable(current_length: int) -> None:
    current = default_categorical_colors(current_length)
    extended = default_categorical_colors(current_length + 1)

    assert extended[:current_length] == current
    assert extended[-1] == default_labeled_class_color(current_length + 1)


def test_default_colors_cycle_after_the_complete_palette() -> None:
    assert default_categorical_colors(103) == [*GODSNOT_102, GODSNOT_102[0]]
    assert default_labeled_class_color(103) == GODSNOT_102[0]


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
