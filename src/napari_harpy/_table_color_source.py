from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ColorSourceKind = Literal["obs_column", "x_var"]
ColorValueKind = Literal["categorical", "continuous"]


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
