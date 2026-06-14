from __future__ import annotations

from typing import Any

from spatialdata._core.validation import check_valid_dataframe_column_name, check_valid_name

_SPATIALDATA_ELEMENT_CONTAINERS = ("images", "labels", "points", "shapes", "tables")


def normalize_spatialdata_name(value: Any, name: str) -> str:
    """Return a trimmed value that is valid as a SpatialData key."""
    normalized = _normalize_nonempty_str(value, name)
    try:
        check_valid_name(normalized)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a valid SpatialData name. {error}") from error
    return normalized


def validate_new_spatialdata_element_name(sdata: Any, name: Any, element_type_label: str) -> str:
    """Return a valid new SpatialData key or raise a user-facing collision error."""
    normalized = normalize_spatialdata_name(name, f"{element_type_label} name")
    if spatialdata_element_name_exists(sdata, normalized):
        target_name = "table name" if element_type_label.lower() == "table" else "name"
        raise ValueError(f'{element_type_label} "{normalized}" already exists. Choose a different {target_name}.')
    return normalized


def spatialdata_element_name_exists(sdata: Any, name: str) -> bool:
    """Return whether a SpatialData key exists across element types, ignoring case."""
    normalized_name = name.lower()
    return any(element_name.lower() == normalized_name for element_name in _iter_spatialdata_element_names(sdata))


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


def _iter_spatialdata_element_names(sdata: Any) -> list[str]:
    element_names: list[str] = []
    for container_name in _SPATIALDATA_ELEMENT_CONTAINERS:
        container = getattr(sdata, container_name, None)
        keys = getattr(container, "keys", None)
        if callable(keys):
            element_names.extend(keys())
    return element_names
