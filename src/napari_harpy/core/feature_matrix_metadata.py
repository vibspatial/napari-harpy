from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from harpy.utils._keys import _FEATURE_MATRICES_KEY

try:
    from scipy.sparse import issparse
except ImportError:  # pragma: no cover - scipy is expected in the plugin env

    def issparse(value: object) -> bool:
        return False


if TYPE_CHECKING:
    from anndata import AnnData


CUSTOM_OBSM_FEATURE_NAME = "custom_obsm"
CUSTOM_OBSM_SOURCE_KIND = "custom_obsm"
FEATURE_MATRIX_METADATA_SCHEMA_VERSION = 1


def normalize_feature_matrix(feature_matrix: Any, n_obs: int, *, copy: bool = True) -> Any:
    # `copy=True` is the eager-array snapshot path for worker payloads.
    # If `.obsm` later accepts lazy arrays, callers should explicitly
    # materialize them before relying on `.copy()` for isolation.
    if issparse(feature_matrix):
        if feature_matrix.ndim != 2:
            raise ValueError("Feature matrices stored in `.obsm` must be 2-dimensional.")
        if feature_matrix.shape[0] != n_obs:
            raise ValueError(
                f"Feature matrix has {feature_matrix.shape[0]} rows but the table has {n_obs} observations."
            )
        return feature_matrix.copy() if copy else feature_matrix

    array = np.asarray(feature_matrix, dtype=np.float64)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2:
        raise ValueError("Feature matrices stored in `.obsm` must be 2-dimensional.")
    if array.shape[0] != n_obs:
        raise ValueError(f"Feature matrix has {array.shape[0]} rows but the table has {n_obs} observations.")
    return array.copy() if copy else array


def register_feature_matrix_metadata(
    table: AnnData,
    feature_key: str,
    *,
    feature_columns: Sequence[str] | str | None = None,
    features: Sequence[str] | str | None = None,
    overwrite: bool = False,
) -> dict[str, object]:
    """Register classifier schema metadata for an existing `.obsm` matrix."""
    resolved_feature_key = _normalize_nonempty_str(feature_key, "feature_key")
    if resolved_feature_key not in table.obsm:
        raise ValueError(f'Feature matrix "{resolved_feature_key}" is not available in ".obsm".')

    raw_feature_matrix = table.obsm[resolved_feature_key]
    feature_matrix = normalize_feature_matrix(raw_feature_matrix, table.n_obs, copy=False)
    n_features = int(feature_matrix.shape[1])
    resolved_feature_columns = _normalize_feature_columns(
        feature_columns,
        feature_key=resolved_feature_key,
        n_features=n_features,
    )
    resolved_features = _normalize_features(features)

    feature_matrices_value = table.uns.get(_FEATURE_MATRICES_KEY)
    if feature_matrices_value is None:
        feature_matrices = {}
    elif isinstance(feature_matrices_value, Mapping):
        feature_matrices = dict(feature_matrices_value)
    else:
        raise ValueError(f"Feature matrix metadata `.uns[{_FEATURE_MATRICES_KEY!r}]` must be a mapping.")

    if resolved_feature_key in feature_matrices and not overwrite:
        raise ValueError(
            f'Feature matrix "{resolved_feature_key}" already has metadata in '
            f'`.uns[{_FEATURE_MATRICES_KEY!r}][{resolved_feature_key!r}]`. Pass `overwrite=True` to replace it.'
        )

    metadata: dict[str, object] = {
        "feature_columns": list(resolved_feature_columns),
        "schema_version": FEATURE_MATRIX_METADATA_SCHEMA_VERSION,
        "backend": "sparse" if issparse(raw_feature_matrix) else "numpy",
        "dtype": _feature_matrix_dtype(raw_feature_matrix),
        "features": list(resolved_features),
        "source_kind": CUSTOM_OBSM_SOURCE_KIND,
    }
    feature_matrices[resolved_feature_key] = metadata
    table.uns[_FEATURE_MATRICES_KEY] = feature_matrices
    return dict(metadata)


def is_custom_obsm_feature_metadata(feature_metadata: Mapping[str, object]) -> bool:
    """Return whether feature metadata came from custom `.obsm` registration."""
    return feature_metadata.get("source_kind") == CUSTOM_OBSM_SOURCE_KIND


def _feature_matrix_dtype(feature_matrix: Any) -> str:
    dtype = getattr(feature_matrix, "dtype", None)
    if dtype is not None:
        return str(dtype)
    return str(np.asarray(feature_matrix).dtype)


def _normalize_feature_columns(
    feature_columns: Sequence[str] | str | None,
    *,
    feature_key: str,
    n_features: int,
) -> tuple[str, ...]:
    if feature_columns is None:
        return tuple(f"{feature_key}_{index}" for index in range(n_features))

    columns = _normalize_str_sequence(feature_columns, name="feature_columns")
    if len(columns) != n_features:
        raise ValueError(
            f"`feature_columns` describes {len(columns)} feature column(s), "
            f"but `.obsm[{feature_key!r}]` has {n_features} column(s)."
        )
    if len(set(columns)) != len(columns):
        raise ValueError("`feature_columns` must not contain duplicate names.")
    return columns


def _normalize_features(features: Sequence[str] | str | None) -> tuple[str, ...]:
    if features is None:
        return (CUSTOM_OBSM_FEATURE_NAME,)

    values = _normalize_str_sequence(features, name="features")
    if len(set(values)) != len(values):
        raise ValueError("`features` must not contain duplicate names.")
    return values


def _normalize_str_sequence(value: Sequence[str] | str, *, name: str) -> tuple[str, ...]:
    values = [value] if isinstance(value, str) else list(value)
    if not values:
        raise ValueError(f"`{name}` must contain at least one value.")

    normalized = tuple(str(item).strip() for item in values)
    if any(not item for item in normalized):
        raise ValueError(f"`{name}` must not contain empty values.")
    return normalized


def _normalize_nonempty_str(value: str, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must not be empty.")
    return normalized
