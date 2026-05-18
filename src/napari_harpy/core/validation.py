from __future__ import annotations

from typing import Any

from spatialdata._core.validation import check_valid_dataframe_column_name, check_valid_name


def normalize_spatialdata_name(value: Any, name: str) -> str:
    """Return a trimmed value that is valid as a SpatialData key."""
    normalized = _normalize_nonempty_str(value, name)
    try:
        check_valid_name(normalized)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a valid SpatialData name. {error}") from error
    return normalized


def normalize_spatialdata_dataframe_column_name(value: Any, name: str) -> str:
    """Return a trimmed value that is valid as a SpatialData table column."""
    normalized = _normalize_nonempty_str(value, name)
    try:
        check_valid_dataframe_column_name(normalized)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a valid SpatialData dataframe column name. {error}") from error
    return normalized


def _normalize_nonempty_str(value: Any, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must not be empty.")
    return normalized
