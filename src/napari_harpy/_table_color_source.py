from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

ColorSourceKind = Literal["obs_column", "x_var"]
ColorValueKind = Literal["categorical", "continuous", "instance"]
STRING_CATEGORICAL_WARNING_MIN_UNIQUE_COUNT = 20
STRING_CATEGORICAL_WARNING_ROW_COUNT_DIVISOR = 100


def string_categorical_warning_threshold(row_count: int) -> int:
    return max(STRING_CATEGORICAL_WARNING_MIN_UNIQUE_COUNT, row_count // STRING_CATEGORICAL_WARNING_ROW_COUNT_DIVISOR)


def has_high_cardinality_string_values(values: Sequence[Any], *, row_count: int) -> bool:
    return len({str(value) for value in values}) > string_categorical_warning_threshold(row_count)


@dataclass(frozen=True)
class TableColorSourceSpec:
    """Semantic description of one table-backed source used for labels coloring."""

    table_name: str
    source_kind: ColorSourceKind
    value_key: str
    value_kind: ColorValueKind

    @property
    def identity(self) -> tuple[str, ColorSourceKind, str]:
        """Return a stable identity for preserving selection across refreshes."""
        return (self.table_name, self.source_kind, self.value_key)

    @property
    def display_name(self) -> str:
        """Return the default user-facing name for this source."""
        return self.value_key
