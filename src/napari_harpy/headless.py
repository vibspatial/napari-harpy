from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from harpy.utils._keys import _FEATURE_MATRICES_KEY

from napari_harpy._annotation import USER_CLASS_COLORS_KEY
from napari_harpy._classifier_core import (
    CLASSIFIER_APPLY_CONFIG_KEY,
    CLASSIFIER_CONFIG_KEY,
    ClassifierApplyResult,
    _validate_feature_matrix_compatible_with_bundle,
)
from napari_harpy._classifier_core import apply_classifier as _apply_classifier
from napari_harpy._classifier_export import (
    ClassifierExportBundle,
    read_classifier_export_bundle,
)
from napari_harpy._feature_extraction_core import (
    FeatureExtractionTriplet,
    _get_triplet_channel_selection_error,
    _normalize_triplets,
    _resolve_harpy_channel_parameter,
    _resolve_harpy_coordinate_system_parameter,
    _resolve_harpy_image_name_parameter,
    _resolve_harpy_labels_name_parameter,
)
from napari_harpy._persistence_core import write_table_prediction_state
from napari_harpy._spatialdata import get_table

if TYPE_CHECKING:
    from spatialdata import SpatialData


@dataclass(frozen=True)
class HeadlessFeatureTarget:
    """Target table and element mapping for headless feature calculation."""

    table_name: str
    feature_key: str
    triplets: tuple[FeatureExtractionTriplet, ...]
    overwrite_feature_key: bool = False


def load_classifier(path: str | Path) -> ClassifierExportBundle:
    """Load a trusted classifier export bundle from disk."""
    return read_classifier_export_bundle(path)


def compute_features_for_classifier(
    sdata: SpatialData,
    bundle: ClassifierExportBundle,
    *,
    target: HeadlessFeatureTarget,
) -> HeadlessFeatureTarget:
    """Compute the feature matrix required by an exported classifier bundle."""
    resolved_target = _normalize_headless_feature_target(target)
    feature_names = bundle.feature_names
    channel_selection_error = _get_triplet_channel_selection_error(resolved_target.triplets, feature_names)
    if channel_selection_error is not None:
        raise ValueError(channel_selection_error)

    import harpy as hp

    hp.tb.add_feature_matrix(
        sdata=sdata,
        labels_name=_resolve_harpy_labels_name_parameter(resolved_target.triplets),
        image_name=_resolve_harpy_image_name_parameter(resolved_target.triplets, feature_names),
        table_name=resolved_target.table_name,
        feature_key=resolved_target.feature_key,
        features=list(feature_names),
        channels=_resolve_harpy_channel_parameter(resolved_target.triplets, feature_names),
        feature_matrices_key=_FEATURE_MATRICES_KEY,
        overwrite_feature_key=resolved_target.overwrite_feature_key,
        to_coordinate_system=_resolve_harpy_coordinate_system_parameter(resolved_target.triplets),
    )

    table = get_table(sdata, resolved_target.table_name)
    _validate_feature_matrix_compatible_with_bundle(table, resolved_target.feature_key, bundle)
    return resolved_target


def apply_classifier(
    sdata: SpatialData,
    bundle: ClassifierExportBundle,
    *,
    table_name: str,
    feature_key: str | None = None,
    prediction_regions: Sequence[str] | None = None,
    pred_class_column: str = "pred_class",
    pred_confidence_column: str = "pred_confidence",
    classifier_path: str | Path | None = None,
) -> ClassifierApplyResult:
    """Apply an exported classifier bundle to an existing feature matrix.

    The target table must already contain a compatible feature matrix in
    `.obsm`. When `feature_key` is omitted, the source feature key stored in the
    bundle is used. Predictions are written in place to `table.obs` using
    `pred_class_column` and `pred_confidence_column`, and apply provenance is
    recorded in `table.uns["classifier_apply_config"]`. If `sdata` is backed by
    zarr, the updated prediction state is written back to disk automatically.

    Parameters
    ----------
    sdata
        SpatialData object containing the target table.
    bundle
        Classifier export bundle returned by `load_classifier(...)` or
        `ClassifierController.export_classifier(...)`.
    table_name
        Name of the target annotation table in `sdata`.
    feature_key
        Key in `table.obsm` for the target feature matrix. If omitted, the
        bundle's source feature key is used.
    prediction_regions
        Optional table regions to classify. If omitted, all regions declared in
        the table metadata are classified.
    pred_class_column
        Column in `table.obs` that will receive predicted class labels.
    pred_confidence_column
        Column in `table.obs` that will receive prediction confidences.
    classifier_path
        Optional artifact path to record in `classifier_apply_config`; this
        does not affect prediction.

    The returned `ClassifierApplyResult` summarizes the rows that were written
    and any in-scope rows skipped because of invalid feature values.
    """
    result = _apply_classifier(
        sdata,
        bundle,
        table_name=table_name,
        feature_key=feature_key,
        prediction_regions=prediction_regions,
        pred_class_column=pred_class_column,
        pred_confidence_column=pred_confidence_column,
        classifier_path=classifier_path,
    )
    _write_backed_prediction_state_if_needed(sdata, result)
    return result


def apply_classifier_from_path(
    sdata: SpatialData,
    path: str | Path,
    *,
    table_name: str,
    feature_key: str | None = None,
    prediction_regions: Sequence[str] | None = None,
    pred_class_column: str = "pred_class",
    pred_confidence_column: str = "pred_confidence",
) -> ClassifierApplyResult:
    """Load a classifier artifact from disk and apply it to an existing feature matrix.

    This is a convenience wrapper around `load_classifier(...)` and
    `apply_classifier(...)`. The artifact path is passed through as
    `classifier_path`, so it is recorded in
    `table.uns["classifier_apply_config"]`.
    """
    bundle = load_classifier(path)
    return apply_classifier(
        sdata,
        bundle,
        table_name=table_name,
        feature_key=feature_key,
        prediction_regions=prediction_regions,
        pred_class_column=pred_class_column,
        pred_confidence_column=pred_confidence_column,
        classifier_path=path,
    )


def apply_classifier_with_feature_extraction(
    sdata: SpatialData,
    bundle: ClassifierExportBundle,
    *,
    target: HeadlessFeatureTarget,
    prediction_regions: Sequence[str] | None = None,
    pred_class_column: str = "pred_class",
    pred_confidence_column: str = "pred_confidence",
    classifier_path: str | Path | None = None,
) -> ClassifierApplyResult:
    """Compute the classifier feature matrix on a target dataset, then apply it.

    This is the headless equivalent of first running Harpy feature extraction
    for `target` and then calling `apply_classifier(...)` with the computed
    feature key. The generated feature matrix is written to
    `table.obsm[target.feature_key]`, and Harpy feature metadata is written to
    `table.uns["feature_matrices"][target.feature_key]`. The computed feature
    columns must exactly match the feature schema stored in `bundle`.

    If `sdata` is backed by zarr, Harpy persists the feature matrix and feature
    metadata during feature extraction, and `apply_classifier(...)` persists
    the prediction columns and classifier apply metadata.

    Parameters
    ----------
    sdata
        SpatialData object containing the target table, labels, and optional
        image elements.
    bundle
        Classifier export bundle returned by `load_classifier(...)` or
        `ClassifierController.export_classifier(...)`.
    target
        Target table, output feature key, and feature-extraction triplets to
        use when computing the feature matrix.
    prediction_regions
        Optional table regions to classify after feature extraction. If
        omitted, all regions declared in the table metadata are classified.
    pred_class_column
        Column in `table.obs` that will receive predicted class labels.
    pred_confidence_column
        Column in `table.obs` that will receive prediction confidences.
    classifier_path
        Optional artifact path to record in `classifier_apply_config`; this
        does not affect feature extraction or prediction.

    The returned `ClassifierApplyResult` summarizes the rows that were written
    and any in-scope rows skipped because of invalid feature values.
    """
    resolved_target = compute_features_for_classifier(sdata, bundle, target=target)
    return apply_classifier(
        sdata,
        bundle,
        table_name=resolved_target.table_name,
        feature_key=resolved_target.feature_key,
        prediction_regions=prediction_regions,
        pred_class_column=pred_class_column,
        pred_confidence_column=pred_confidence_column,
        classifier_path=classifier_path,
    )


def apply_classifier_with_feature_extraction_from_path(
    sdata: SpatialData,
    path: str | Path,
    *,
    target: HeadlessFeatureTarget,
    prediction_regions: Sequence[str] | None = None,
    pred_class_column: str = "pred_class",
    pred_confidence_column: str = "pred_confidence",
) -> ClassifierApplyResult:
    """Load a classifier artifact, compute target features, and apply it.

    This is a convenience wrapper around `load_classifier(...)` and
    `apply_classifier_with_feature_extraction(...)`. The artifact path is
    passed through as `classifier_path`, so it is recorded in
    `table.uns["classifier_apply_config"]`.

    Parameters
    ----------
    sdata
        SpatialData object containing the target table, labels, and optional
        image elements.
    path
        Path to a trusted `.harpy-classifier.joblib` classifier artifact.
    target
        Target table, output feature key, and feature-extraction triplets to
        use when computing the feature matrix.
    prediction_regions
        Optional table regions to classify after feature extraction. If
        omitted, all regions declared in the table metadata are classified.
    pred_class_column
        Column in `table.obs` that will receive predicted class labels.
    pred_confidence_column
        Column in `table.obs` that will receive prediction confidences.

    The returned `ClassifierApplyResult` summarizes the rows that were written
    and any in-scope rows skipped because of invalid feature values.
    """
    bundle = load_classifier(path)
    return apply_classifier_with_feature_extraction(
        sdata,
        bundle,
        target=target,
        prediction_regions=prediction_regions,
        pred_class_column=pred_class_column,
        pred_confidence_column=pred_confidence_column,
        classifier_path=path,
    )


def _normalize_headless_feature_target(target: HeadlessFeatureTarget) -> HeadlessFeatureTarget:
    normalized_table_name = _normalize_nonempty_str(target.table_name, "target.table_name")
    normalized_feature_key = _normalize_nonempty_str(target.feature_key, "target.feature_key")
    normalized_triplets = _normalize_triplets(target.triplets)
    if not normalized_triplets:
        raise ValueError("target.triplets must contain at least one feature-extraction triplet.")

    return HeadlessFeatureTarget(
        table_name=normalized_table_name,
        feature_key=normalized_feature_key,
        triplets=normalized_triplets,
        overwrite_feature_key=bool(target.overwrite_feature_key),
    )


def _normalize_nonempty_str(value: str, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must not be empty.")
    return normalized


def _write_backed_prediction_state_if_needed(sdata: SpatialData, result: ClassifierApplyResult) -> None:
    if not sdata.is_backed() or sdata.path is None:
        return

    write_table_prediction_state(
        sdata,
        table_name=result.table_name,
        uns_keys=(
            USER_CLASS_COLORS_KEY,
            f"{result.pred_class_column}_colors",
            CLASSIFIER_CONFIG_KEY,
            CLASSIFIER_APPLY_CONFIG_KEY,
        ),
    )
