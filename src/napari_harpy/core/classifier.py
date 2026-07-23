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

from napari_harpy.core.annotation import USER_CLASS_COLUMN
from napari_harpy.core.class_palette import normalize_class_values, set_class_annotation_state
from napari_harpy.core.classifier_export import ClassifierExportBundle, normalize_feature_columns
from napari_harpy.core.feature_matrix_metadata import (
    CUSTOM_OBSM_SOURCE_KIND,
    normalize_feature_matrix,
    normalize_feature_matrix_source_kind,
)
from napari_harpy.core.spatialdata import get_table, get_table_metadata
from napari_harpy.core.validation import normalize_spatialdata_dataframe_column_name

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


class ObjectClassificationStateError(ValueError):
    """Raised when existing Object Classification columns are not canonical."""


def validate_object_classification_table_state(table: AnnData) -> None:
    """Validate existing Object Classification columns without mutation."""
    if USER_CLASS_COLUMN in table.obs:
        _validate_categorical_class_column(table.obs[USER_CLASS_COLUMN], column_name=USER_CLASS_COLUMN)

    has_pred_class = PRED_CLASS_COLUMN in table.obs
    has_pred_confidence = PRED_CONFIDENCE_COLUMN in table.obs
    if not has_pred_class and not has_pred_confidence:
        return
    if has_pred_class != has_pred_confidence:
        missing_column = PRED_CONFIDENCE_COLUMN if has_pred_class else PRED_CLASS_COLUMN
        raise ObjectClassificationStateError(
            "Object Classification state is invalid: "
            f"`{PRED_CLASS_COLUMN}` and `{PRED_CONFIDENCE_COLUMN}` must either both exist or both be absent. "
            f"Missing column: `{missing_column}`."
        )

    pred_class = table.obs[PRED_CLASS_COLUMN]
    pred_confidence = table.obs[PRED_CONFIDENCE_COLUMN]
    _validate_categorical_class_column(pred_class, column_name=PRED_CLASS_COLUMN)
    if not pd.api.types.is_float_dtype(pred_confidence.dtype):
        raise ObjectClassificationStateError(
            "Object Classification state is invalid: "
            f"`{PRED_CONFIDENCE_COLUMN}` must use a floating-point dtype with values between 0 and 1. "
            f"Observed dtype: `{pred_confidence.dtype}`."
        )

    pred_class_missing = pred_class.isna().to_numpy(dtype=bool, copy=False)
    pred_confidence_missing = pred_confidence.isna().to_numpy(dtype=bool, copy=False)
    mismatched_missingness = pred_class_missing != pred_confidence_missing
    if bool(mismatched_missingness.any()):
        raise ObjectClassificationStateError(
            "Object Classification state is invalid: "
            f"`{PRED_CONFIDENCE_COLUMN}` must be missing exactly when `{PRED_CLASS_COLUMN}` is missing. "
            f"Found {int(mismatched_missingness.sum())} mismatched row(s)."
        )

    confidence_values = pred_confidence.loc[~pred_confidence_missing].to_numpy(dtype=np.float64, copy=False)
    if confidence_values.size and (
        not bool(np.isfinite(confidence_values).all())
        or bool(((confidence_values < 0.0) | (confidence_values > 1.0)).any())
    ):
        raise ObjectClassificationStateError(
            "Object Classification state is invalid: "
            f"non-missing `{PRED_CONFIDENCE_COLUMN}` values must be finite floating-point values between 0 and 1."
        )


def _validate_categorical_class_column(values: pd.Series, *, column_name: str) -> None:
    if not isinstance(values.dtype, pd.CategoricalDtype):
        raise ObjectClassificationStateError(
            "Object Classification state is invalid: "
            f"`{column_name}` must use a categorical dtype with positive integer categories. "
            f"Observed dtype: `{values.dtype}`."
        )

    for category in values.cat.categories:
        if isinstance(category, (bool, np.bool_)) or not isinstance(category, (int, np.integer)) or int(category) <= 0:
            raise ObjectClassificationStateError(
                "Object Classification state is invalid: "
                f"`{column_name}` must use a categorical dtype with positive integer categories. "
                f"Observed invalid category: `{category!r}`. Rows without a class must be stored as missing values."
            )


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
    resolved_pred_class_column = normalize_spatialdata_dataframe_column_name(pred_class_column, "pred_class_column")
    resolved_pred_confidence_column = normalize_spatialdata_dataframe_column_name(
        pred_confidence_column,
        "pred_confidence_column",
    )
    if resolved_pred_class_column == resolved_pred_confidence_column:
        raise ValueError("pred_class_column and pred_confidence_column must be different.")

    table = get_table(sdata, resolved_table_name)
    metadata = get_table_metadata(sdata, resolved_table_name)
    _validate_feature_matrix_compatible_with_bundle(table, resolved_feature_key, bundle)
    feature_matrix = normalize_feature_matrix(table.obsm[resolved_feature_key], table.n_obs, copy=False)
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
    """Validate that a target feature matrix can be used with a classifier bundle.

    The target matrix source kind does not need to match the bundle source
    kind. For example, a classifier trained from Harpy-generated metadata can
    be applied to a custom `.obsm` matrix when both declare the exact same
    ordered `feature_columns` schema and the live matrix width matches that
    schema.

    Matching dimensions alone are not sufficient. A bundle expecting
    `("area", "mean")` is compatible with a target custom matrix declaring
    `("area", "mean")`, but not with one declaring `("mean", "area")`, even
    though both matrices have two columns. The estimator must receive the same
    feature meaning in each column position that it saw during training.
    """
    if feature_key not in table.obsm:
        raise ValueError(f"Feature matrix `{feature_key}` is not available in `.obsm`.")
    feature_metadata = _get_feature_metadata(table, feature_key)
    target_source_kind = normalize_feature_matrix_source_kind(feature_metadata)
    custom_obsm_involved = (
        bundle.source_kind == CUSTOM_OBSM_SOURCE_KIND or target_source_kind == CUSTOM_OBSM_SOURCE_KIND
    )
    target_feature_columns = normalize_feature_columns(feature_metadata)
    if target_feature_columns != bundle.feature_columns:
        if custom_obsm_involved:
            raise ValueError(
                f"Custom feature matrix `{feature_key}` columns do not match the classifier bundle feature schema. "
                "Custom matrices require the same `feature_columns` in the same order. "
                "Register/select a matching matrix or retrain the classifier."
            )
        raise ValueError(f"Feature matrix `{feature_key}` columns do not match the classifier bundle feature schema.")

    _validate_current_feature_matrix_matches_columns(
        table,
        feature_key,
        target_feature_columns,
        custom_obsm=custom_obsm_involved,
    )


def _validate_current_feature_matrix_matches_columns(
    table: AnnData,
    feature_key: str,
    feature_columns: tuple[str, ...],
    *,
    custom_obsm: bool = False,
) -> None:
    try:
        feature_matrix = normalize_feature_matrix(table.obsm[feature_key], table.n_obs, copy=False)
    except KeyError as error:
        raise ValueError(f"Feature matrix `{feature_key}` is not available in `.obsm`.") from error

    if int(feature_matrix.shape[1]) != len(feature_columns):
        raise ValueError(
            _format_feature_matrix_width_mismatch_error(
                feature_key,
                n_features=int(feature_matrix.shape[1]),
                metadata_n_features=len(feature_columns),
                custom_obsm=custom_obsm,
            )
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


def _format_feature_matrix_width_mismatch_error(
    feature_key: str,
    *,
    n_features: int,
    metadata_n_features: int,
    custom_obsm: bool,
) -> str:
    if custom_obsm:
        return (
            f"Custom feature matrix `{feature_key}` has {n_features} column(s), but its metadata describes "
            f"{metadata_n_features} feature column(s). Custom matrices require the live `.obsm` width to match "
            "the registered `feature_columns`. Register/select a matching matrix or retrain the classifier."
        )
    return (
        f"Feature matrix `{feature_key}` has {n_features} column(s), but its metadata describes "
        f"{metadata_n_features} feature column(s)."
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
    normalize_feature_matrix_source_kind(feature_metadata)
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
    pred_confidence_values.loc[pred_class_values.isna()] = np.nan
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
    pred_class_values = _get_pred_class_values(table.obs, column_name=pred_class_column)
    pred_confidence_values = _get_pred_confidence_values(
        table.obs,
        column_name=pred_confidence_column,
    )
    pred_class_values.iloc[prediction_table_row_positions] = pd.NA
    pred_confidence_values.iloc[prediction_table_row_positions] = np.nan
    _set_pred_class_annotation_state(table, pred_class_values, column_name=pred_class_column)
    table.obs[pred_confidence_column] = pred_confidence_values


def _get_pred_class_values(
    obs: pd.DataFrame,
    *,
    column_name: str = PRED_CLASS_COLUMN,
) -> pd.Series:
    if column_name not in obs:
        return pd.Series(pd.NA, index=obs.index, dtype="Int64", name=column_name)

    return normalize_class_values(obs[column_name], column_name=column_name)


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
