from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

TableColorSourceKind = Literal["obs_column", "x_var"]
TableColorValueKind = Literal["categorical", "continuous", "instance"]
ShapeColorSourceKind = Literal["shape_column"]
ShapeColorValueKind = Literal["categorical", "continuous"]
TABLE_COLOR_SOURCE_KINDS: tuple[TableColorSourceKind, ...] = ("obs_column", "x_var")
TABLE_COLOR_VALUE_KINDS: tuple[TableColorValueKind, ...] = ("categorical", "continuous", "instance")
SHAPE_COLOR_SOURCE_KINDS: tuple[ShapeColorSourceKind, ...] = ("shape_column",)
SHAPE_COLOR_VALUE_KINDS: tuple[ShapeColorValueKind, ...] = ("categorical", "continuous")


def validate_table_color_source_kind(kind: str) -> TableColorSourceKind:
    """Return a validated table color source kind."""
    if kind not in TABLE_COLOR_SOURCE_KINDS:
        raise ValueError(_format_invalid_kind_error("table color source kind", kind, TABLE_COLOR_SOURCE_KINDS))

    return cast(TableColorSourceKind, kind)


def validate_table_color_value_kind(kind: str) -> TableColorValueKind:
    """Return a validated table color value kind."""
    if kind not in TABLE_COLOR_VALUE_KINDS:
        raise ValueError(_format_invalid_kind_error("table color value kind", kind, TABLE_COLOR_VALUE_KINDS))

    return cast(TableColorValueKind, kind)


def validate_shape_color_source_kind(kind: str) -> ShapeColorSourceKind:
    """Return a validated shape color source kind."""
    if kind not in SHAPE_COLOR_SOURCE_KINDS:
        raise ValueError(_format_invalid_kind_error("shape color source kind", kind, SHAPE_COLOR_SOURCE_KINDS))

    return cast(ShapeColorSourceKind, kind)


def validate_shape_color_value_kind(kind: str) -> ShapeColorValueKind:
    """Return a validated shape color value kind."""
    if kind not in SHAPE_COLOR_VALUE_KINDS:
        raise ValueError(_format_invalid_kind_error("shape color value kind", kind, SHAPE_COLOR_VALUE_KINDS))

    return cast(ShapeColorValueKind, kind)


@dataclass(frozen=True)
class TableColorSourceSpec:
    """Semantic description of one table-backed source used for viewer coloring."""

    table_name: str
    source_kind: TableColorSourceKind
    value_key: str
    value_kind: TableColorValueKind

    def __post_init__(self) -> None:
        validate_table_color_source_kind(self.source_kind)
        validate_table_color_value_kind(self.value_kind)

    @property
    def identity(self) -> tuple[str, TableColorSourceKind, str]:
        """Return a stable identity for preserving selection across refreshes."""
        return (self.table_name, self.source_kind, self.value_key)

    @property
    def display_name(self) -> str:
        """Return the default user-facing name for this source."""
        return self.value_key


@dataclass(frozen=True)
class ShapeColumnColorSourceSpec:
    """Semantic description of one direct shapes column used for coloring."""

    source_kind: ShapeColorSourceKind
    value_key: str
    value_kind: ShapeColorValueKind

    def __post_init__(self) -> None:
        validate_shape_color_source_kind(self.source_kind)
        validate_shape_color_value_kind(self.value_kind)

    @property
    def identity(self) -> tuple[ShapeColorSourceKind, str]:
        """Return a stable identity for preserving selection across refreshes."""
        return (self.source_kind, self.value_key)

    @property
    def display_name(self) -> str:
        """Return the default user-facing name for this source."""
        return self.value_key


def _format_invalid_kind_error(kind_label: str, kind: str, allowed_kinds: tuple[str, ...]) -> str:
    allowed = ", ".join(repr(allowed_kind) for allowed_kind in allowed_kinds)
    return f"Invalid {kind_label} {kind!r}. Expected one of: {allowed}."
