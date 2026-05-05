from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from harpy.utils._keys import _FEATURE_MATRICES_KEY
from numpy.typing import NDArray

from napari_harpy._annotation import UNLABELED_CLASS
from napari_harpy._class_palette import set_class_annotation_state
from napari_harpy._classifier_export import ClassifierExportBundle, normalize_feature_columns
from napari_harpy._spatialdata import get_table, get_table_metadata

try:
    from scipy.sparse import issparse
except ImportError:  # pragma: no cover - scipy is expected in the plugin env

    def issparse(value: object) -> bool:
        return False


if TYPE_CHECKING:
    from anndata import AnnData
    from spatialdata import SpatialData


BoolArray = NDArray[np.bool_]
TableRowPositions = NDArray[np.int64]

PRED_CLASS_COLUMN = "pred_class"
PRED_CLASS_COLORS_KEY = f"{PRED_CLASS_COLUMN}_colors"
PRED_CONFIDENCE_COLUMN = "pred_confidence"
CLASSIFIER_CONFIG_KEY = "classifier_config"
CLASSIFIER_APPLY_CONFIG_KEY = "classifier_apply_config"


@dataclass(frozen=True)
class ClassifierApplyResult:
    """Summary of predictions applied to a table."""

    table_name: str
    feature_key: str
    prediction_regions: tuple[str, ...]
    n_predicted_rows: int
    n_skipped_feature_invalid_rows: int
    pred_class_column: str
    pred_confidence_column: str
    applied_at: str


def apply_classifier(
    sdata: SpatialData,
    bundle: ClassifierExportBundle,
    *,
    table_name: str,
    feature_key: str | None = None,
    prediction_regions: Sequence[str] | None = None,
    pred_class_column: str = PRED_CLASS_COLUMN,
    pred_confidence_column: str = PRED_CONFIDENCE_COLUMN,
    classifier_path: str | Path | None = None,
) -> ClassifierApplyResult:
    """Apply an exported classifier bundle to an existing compatible feature matrix."""
    resolved_table_name = _normalize_nonempty_str(table_name, "table_name")
    resolved_feature_key = (
        bundle.feature_key if feature_key is None else _normalize_nonempty_str(feature_key, "feature_key")
    )
    resolved_pred_class_column = _normalize_nonempty_str(pred_class_column, "pred_class_column")
    resolved_pred_confidence_column = _normalize_nonempty_str(pred_confidence_column, "pred_confidence_column")
    if resolved_pred_class_column == resolved_pred_confidence_column:
        raise ValueError("pred_class_column and pred_confidence_column must be different.")

    table = get_table(sdata, resolved_table_name)
    metadata = get_table_metadata(sdata, resolved_table_name)
    _validate_feature_matrix_compatible_with_bundle(table, resolved_feature_key, bundle)
    feature_matrix = _normalize_feature_matrix(table.obsm[resolved_feature_key], table.n_obs, copy=False)
    _validate_estimator_matches_feature_matrix(bundle, feature_matrix, resolved_feature_key)

    # Resolve the requested regions against the table metadata so typos fail instead of being skipped silently.
    resolved_prediction_regions = _resolve_prediction_regions(prediction_regions, metadata.regions)
    raw_prediction_positions = _resolve_region_row_positions(
        table.obs,
        metadata.region_key,
        resolved_prediction_regions,
    )
    feature_valid_row_mask = _get_finite_feature_row_mask(feature_matrix)
    valid_in_scope_mask = feature_valid_row_mask[raw_prediction_positions]
    prediction_positions = np.asarray(raw_prediction_positions[valid_in_scope_mask], dtype=np.int64)

    if prediction_positions.size:
        predict_features = feature_matrix[prediction_positions]
        pred_classes, pred_confidences = _predict_classifier(bundle, predict_features)
    else:
        pred_classes = np.array([], dtype=np.int64)
        pred_confidences = np.array([], dtype=np.float64)

    result = _write_classifier_predictions(
        table,
        table_name=resolved_table_name,
        feature_key=resolved_feature_key,
        prediction_regions=resolved_prediction_regions,
        raw_prediction_table_row_positions=raw_prediction_positions,
        prediction_table_row_positions=prediction_positions,
        pred_classes=pred_classes,
        pred_confidences=pred_confidences,
        pred_class_column=resolved_pred_class_column,
        pred_confidence_column=resolved_pred_confidence_column,
    )
    table.uns[CLASSIFIER_APPLY_CONFIG_KEY] = _build_classifier_apply_config(
        bundle,
        result,
        classifier_path=classifier_path,
    )
    return result


def _write_classifier_predictions(
    table: AnnData,
    *,
    table_name: str,
    feature_key: str,
    prediction_regions: tuple[str, ...],
    raw_prediction_table_row_positions: TableRowPositions,
    prediction_table_row_positions: TableRowPositions,
    pred_classes: np.ndarray,
    pred_confidences: np.ndarray,
    pred_class_column: str = PRED_CLASS_COLUMN,
    pred_confidence_column: str = PRED_CONFIDENCE_COLUMN,
) -> ClassifierApplyResult:
    _clear_predictions_for_row_positions(
        table,
        raw_prediction_table_row_positions,
        pred_class_column=pred_class_column,
        pred_confidence_column=pred_confidence_column,
    )
    if prediction_table_row_positions.size:
        _set_predictions_for_prediction_rows(
            table,
            prediction_table_row_positions,
            pred_classes,
            pred_confidences,
            pred_class_column=pred_class_column,
            pred_confidence_column=pred_confidence_column,
        )

    return ClassifierApplyResult(
        table_name=table_name,
        feature_key=feature_key,
        prediction_regions=prediction_regions,
        n_predicted_rows=int(prediction_table_row_positions.size),
        n_skipped_feature_invalid_rows=int(
            raw_prediction_table_row_positions.size - prediction_table_row_positions.size
        ),
        pred_class_column=pred_class_column,
        pred_confidence_column=pred_confidence_column,
        applied_at=datetime.now(UTC).isoformat(),
    )


def _validate_feature_matrix_compatible_with_bundle(
    table: AnnData,
    feature_key: str,
    bundle: ClassifierExportBundle,
) -> None:
    if feature_key not in table.obsm:
        raise ValueError(f"Feature matrix `{feature_key}` is not available in `.obsm`.")
    feature_metadata = _get_feature_metadata(table, feature_key)
    target_feature_columns = normalize_feature_columns(feature_metadata)
    if target_feature_columns != bundle.feature_columns:
        raise ValueError(
            f"Feature matrix `{feature_key}` columns do not match the classifier bundle feature schema."
        )
    _validate_current_feature_matrix_matches_columns(table, feature_key, target_feature_columns)


def _validate_current_feature_matrix_matches_columns(
    table: AnnData,
    feature_key: str,
    feature_columns: tuple[str, ...],
) -> None:
    try:
        feature_matrix = _normalize_feature_matrix(table.obsm[feature_key], table.n_obs, copy=False)
    except KeyError as error:
        raise ValueError(f"Feature matrix `{feature_key}` is not available in `.obsm`.") from error

    if int(feature_matrix.shape[1]) != len(feature_columns):
        raise ValueError(
            f"Feature matrix `{feature_key}` has {int(feature_matrix.shape[1])} column(s), but its metadata "
            f"describes {len(feature_columns)} feature column(s)."
        )


def _validate_estimator_matches_feature_matrix(
    bundle: ClassifierExportBundle,
    feature_matrix: Any,
    feature_key: str,
) -> None:
    estimator_feature_count = getattr(bundle.estimator, "n_features_in_", None)
    if estimator_feature_count is not None and int(estimator_feature_count) != int(feature_matrix.shape[1]):
        raise ValueError(
            f"Feature matrix `{feature_key}` has {int(feature_matrix.shape[1])} column(s), but the classifier "
            f"estimator expects {int(estimator_feature_count)} feature column(s)."
        )


def _resolve_prediction_regions(
    prediction_regions: Sequence[str] | None,
    table_regions: tuple[str, ...],
) -> tuple[str, ...]:
    if prediction_regions is None:
        return table_regions
    if isinstance(prediction_regions, str):
        prediction_regions = (prediction_regions,)

    normalized_regions = tuple(dict.fromkeys(str(region) for region in prediction_regions))
    if not normalized_regions:
        raise ValueError("prediction_regions must contain at least one region when provided.")
    if any(not region for region in normalized_regions):
        raise ValueError("prediction_regions must not contain empty region names.")

    missing_regions = sorted(set(normalized_regions).difference(table_regions))
    if missing_regions:
        raise ValueError(f"Table does not contain prediction region(s): {missing_regions}.")
    return normalized_regions


def _predict_classifier(bundle: ClassifierExportBundle, predict_features: Any) -> tuple[np.ndarray, np.ndarray]:
    pred_classes = np.asarray(bundle.estimator.predict(predict_features), dtype=np.int64)
    pred_proba = np.asarray(bundle.estimator.predict_proba(predict_features), dtype=np.float64)
    pred_confidences = pred_proba.max(axis=1)
    return pred_classes, pred_confidences


def _build_classifier_apply_config(
    bundle: ClassifierExportBundle,
    result: ClassifierApplyResult,
    *,
    classifier_path: str | Path | None,
) -> dict[str, object]:
    return {
        "applied": True,
        "apply_timestamp": result.applied_at,
        "classifier_path": None if classifier_path is None else str(classifier_path),
        "source_classifier_config": deepcopy(bundle.source_classifier_config),
        "table_name": result.table_name,
        "feature_key": result.feature_key,
        "prediction_regions": list(result.prediction_regions),
        "n_predicted_rows": result.n_predicted_rows,
        "n_skipped_feature_invalid_rows": result.n_skipped_feature_invalid_rows,
        "pred_class_column": result.pred_class_column,
        "pred_confidence_column": result.pred_confidence_column,
    }


def _normalize_nonempty_str(value: str, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must not be empty.")
    return normalized


def _normalize_feature_matrix(feature_matrix: Any, n_obs: int, *, copy: bool = True) -> Any:
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


def _get_finite_feature_row_mask(feature_matrix: Any) -> BoolArray:
    if issparse(feature_matrix):
        finite_data_mask = np.isfinite(feature_matrix.data)
        if bool(finite_data_mask.all()):
            return np.ones(feature_matrix.shape[0], dtype=bool)

        invalid_rows = np.unique(feature_matrix.tocoo().row[~finite_data_mask])
        valid_row_mask = np.ones(feature_matrix.shape[0], dtype=bool)
        valid_row_mask[invalid_rows] = False
        return valid_row_mask

    finite_feature_mask = np.isfinite(np.asarray(feature_matrix, dtype=np.float64))
    return np.asarray(finite_feature_mask.all(axis=1), dtype=bool)


def _resolve_region_row_positions(obs: pd.DataFrame, region_key: str, regions: tuple[str, ...]) -> TableRowPositions:
    if not regions:
        return np.array([], dtype=np.int64)
    if len(regions) == 1:
        region_mask = (obs[region_key] == regions[0]).to_numpy(dtype=bool, copy=False)
    else:
        region_mask = obs[region_key].isin(regions).to_numpy(dtype=bool, copy=False)
    return np.asarray(np.flatnonzero(region_mask), dtype=np.int64)


def _normalize_prediction_regions(value: object) -> tuple[str, ...]:
    if isinstance(value, np.ndarray):
        if value.ndim != 1:
            return ()
        value = value.tolist()
    if isinstance(value, (list, tuple)) and all(isinstance(region, str) for region in value):
        return tuple(value)
    return ()


def _get_feature_metadata(table: AnnData, feature_key: str) -> dict[str, object]:
    feature_matrices = table.uns.get(_FEATURE_MATRICES_KEY)
    if not isinstance(feature_matrices, Mapping):
        raise ValueError(
            f"Feature matrix `{feature_key}` is missing Harpy metadata in `.uns[{_FEATURE_MATRICES_KEY!r}]`."
        )

    feature_metadata = feature_matrices.get(feature_key)
    if not isinstance(feature_metadata, Mapping):
        raise ValueError(
            f"Feature matrix `{feature_key}` is missing Harpy metadata in "
            f"`.uns[{_FEATURE_MATRICES_KEY!r}][{feature_key!r}]`."
        )
    return dict(feature_metadata)


def _ensure_prediction_columns(
    table: AnnData,
    *,
    pred_class_column: str = PRED_CLASS_COLUMN,
    pred_confidence_column: str = PRED_CONFIDENCE_COLUMN,
) -> None:
    pred_class_values = _get_pred_class_values(table.obs, column_name=pred_class_column)
    pred_confidence_values = _get_pred_confidence_values(
        table.obs,
        column_name=pred_confidence_column,
    )
    _set_pred_class_annotation_state(table, pred_class_values, column_name=pred_class_column)
    table.obs[pred_confidence_column] = pred_confidence_values


def _set_predictions_for_prediction_rows(
    table: AnnData,
    prediction_table_row_positions: TableRowPositions,
    pred_classes: np.ndarray,
    pred_confidences: np.ndarray,
    *,
    pred_class_column: str = PRED_CLASS_COLUMN,
    pred_confidence_column: str = PRED_CONFIDENCE_COLUMN,
) -> None:
    pred_class_values = _get_pred_class_values(table.obs, column_name=pred_class_column)
    pred_confidence_values = _get_pred_confidence_values(
        table.obs,
        column_name=pred_confidence_column,
    )
    pred_class_values.iloc[prediction_table_row_positions] = np.asarray(pred_classes, dtype=np.int64)
    pred_confidence_values.iloc[prediction_table_row_positions] = np.asarray(pred_confidences, dtype=np.float64)
    _set_pred_class_annotation_state(table, pred_class_values, column_name=pred_class_column)
    table.obs[pred_confidence_column] = pred_confidence_values


def _clear_predictions_for_row_positions(
    table: AnnData,
    prediction_table_row_positions: TableRowPositions,
    *,
    pred_class_column: str = PRED_CLASS_COLUMN,
    pred_confidence_column: str = PRED_CONFIDENCE_COLUMN,
) -> None:
    _set_predictions_for_prediction_rows(
        table,
        prediction_table_row_positions,
        np.full(prediction_table_row_positions.shape, UNLABELED_CLASS, dtype=np.int64),
        np.full(prediction_table_row_positions.shape, np.nan, dtype=np.float64),
        pred_class_column=pred_class_column,
        pred_confidence_column=pred_confidence_column,
    )


def _get_pred_class_values(
    obs: pd.DataFrame,
    *,
    column_name: str = PRED_CLASS_COLUMN,
) -> pd.Series:
    if column_name not in obs:
        return pd.Series(UNLABELED_CLASS, index=obs.index, dtype="int64", name=column_name)

    values = pd.to_numeric(obs[column_name].astype("string"), errors="coerce").fillna(UNLABELED_CLASS)
    return pd.Series(np.asarray(values, dtype=np.int64), index=obs.index, dtype="int64", name=column_name)


def _set_pred_class_annotation_state(
    table: AnnData,
    values: pd.Series,
    *,
    column_name: str = PRED_CLASS_COLUMN,
) -> None:
    set_class_annotation_state(
        table,
        values,
        column_name=column_name,
        colors_key=_pred_class_colors_key(column_name),
        warn_on_palette_overwrite=False,
    )


def _pred_class_colors_key(column_name: str) -> str:
    return f"{column_name}_colors"


def _get_pred_confidence_values(
    obs: pd.DataFrame,
    *,
    column_name: str = PRED_CONFIDENCE_COLUMN,
) -> pd.Series:
    if column_name not in obs:
        return pd.Series(np.full(len(obs), np.nan, dtype=np.float64), index=obs.index, name=column_name)

    values = pd.to_numeric(obs[column_name], errors="coerce").astype("float64")
    return pd.Series(np.asarray(values, dtype=np.float64), index=obs.index, name=column_name)
