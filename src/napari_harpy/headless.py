from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

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
    FeatureExtractionChannel,
    FeatureExtractionTriplet,
    _get_triplet_channel_selection_error,
    _normalize_channels,
    _normalize_triplets,
    _resolve_harpy_channel_parameter,
    _resolve_harpy_coordinate_system_parameter,
    _resolve_harpy_image_name_parameter,
    _resolve_harpy_labels_name_parameter,
)
from napari_harpy._persistence_core import write_table_prediction_state
from napari_harpy._spatialdata import _get_element_coordinate_systems, get_table

if TYPE_CHECKING:
    from spatialdata import SpatialData

T = TypeVar("T")


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
    table_name: str,
    labels_name: str | Sequence[str],
    feature_key: str | None = None,
    coordinate_system: str | Sequence[str] | None = None,
    image_name: str | Sequence[str] | None = None,
    channels: Sequence[FeatureExtractionChannel] | FeatureExtractionChannel | None = None,
    overwrite_feature_key: bool = False,
    pred_class_column: str = "pred_class",
    pred_confidence_column: str = "pred_confidence",
    classifier_path: str | Path | None = None,
) -> ClassifierApplyResult:
    """Compute the classifier feature matrix on a target dataset, then apply it.

    This is the headless equivalent of first running Harpy feature extraction
    for `labels_name` and then calling `apply_classifier(...)` with the
    computed feature key. The generated feature matrix is written to
    `table.obsm[feature_key]`, and Harpy feature metadata is written to
    `table.uns["feature_matrices"][feature_key]`. The computed feature columns
    must exactly match the feature schema stored in `bundle`.

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
    table_name
        Name of the target annotation table in `sdata`.
    labels_name
        Labels element or elements to use for feature extraction and prediction.
        These labels become the prediction regions after feature extraction.
    feature_key
        Key in `table.obsm` for the computed feature matrix. If omitted, the
        source feature key stored in the classifier bundle is used.
    coordinate_system
        Coordinate system or systems to use for the selected labels. If omitted,
        the coordinate system is inferred only when unambiguous.
    image_name
        Image element or elements to use for intensity-derived features. This is
        required when the classifier bundle was trained on intensity features
        and ignored for morphology-only classifiers.
    channels
        Optional image channel selection for intensity-derived features. The
        same channel selection is used for every selected labels element.
    overwrite_feature_key
        Whether to replace an existing `table.obsm[feature_key]`.
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
    target = _build_headless_feature_target(
        sdata,
        bundle,
        table_name=table_name,
        labels_name=labels_name,
        feature_key=feature_key,
        coordinate_system=coordinate_system,
        image_name=image_name,
        channels=channels,
        overwrite_feature_key=overwrite_feature_key,
    )
    resolved_target = compute_features_for_classifier(sdata, bundle, target=target)
    return apply_classifier(
        sdata,
        bundle,
        table_name=resolved_target.table_name,
        feature_key=resolved_target.feature_key,
        prediction_regions=tuple(triplet.label_name for triplet in resolved_target.triplets),
        pred_class_column=pred_class_column,
        pred_confidence_column=pred_confidence_column,
        classifier_path=classifier_path,
    )


def apply_classifier_with_feature_extraction_from_path(
    sdata: SpatialData,
    path: str | Path,
    *,
    table_name: str,
    labels_name: str | Sequence[str],
    feature_key: str | None = None,
    coordinate_system: str | Sequence[str] | None = None,
    image_name: str | Sequence[str] | None = None,
    channels: Sequence[FeatureExtractionChannel] | FeatureExtractionChannel | None = None,
    overwrite_feature_key: bool = False,
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
    table_name
        Name of the target annotation table in `sdata`.
    labels_name
        Labels element or elements to use for feature extraction and prediction.
        These labels become the prediction regions after feature extraction.
    feature_key
        Key in `table.obsm` for the computed feature matrix. If omitted, the
        source feature key stored in the classifier bundle is used.
    coordinate_system
        Coordinate system or systems to use for the selected labels. If omitted,
        the coordinate system is inferred only when unambiguous.
    image_name
        Image element or elements to use for intensity-derived features. This is
        required when the classifier bundle was trained on intensity features
        and ignored for morphology-only classifiers.
    channels
        Optional image channel selection for intensity-derived features. The
        same channel selection is used for every selected labels element.
    overwrite_feature_key
        Whether to replace an existing `table.obsm[feature_key]`.
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
        table_name=table_name,
        labels_name=labels_name,
        feature_key=feature_key,
        coordinate_system=coordinate_system,
        image_name=image_name,
        channels=channels,
        overwrite_feature_key=overwrite_feature_key,
        pred_class_column=pred_class_column,
        pred_confidence_column=pred_confidence_column,
        classifier_path=path,
    )


def _build_headless_feature_target(
    sdata: SpatialData,
    bundle: ClassifierExportBundle,
    *,
    table_name: str,
    labels_name: str | Sequence[str],
    feature_key: str | None,
    coordinate_system: str | Sequence[str] | None,
    image_name: str | Sequence[str] | None,
    channels: Sequence[FeatureExtractionChannel] | FeatureExtractionChannel | None,
    overwrite_feature_key: bool,
) -> HeadlessFeatureTarget:
    normalized_labels = _normalize_name_sequence(labels_name, "labels_name")
    coordinate_systems = _resolve_coordinate_systems_for_labels(sdata, normalized_labels, coordinate_system)
    image_names = _normalize_optional_parallel_names(image_name, len(normalized_labels), "image_name")
    normalized_channels = _normalize_channels(channels)
    resolved_feature_key = bundle.feature_key if feature_key is None else _normalize_nonempty_str(feature_key, "feature_key")

    triplets = tuple(
        FeatureExtractionTriplet(
            coordinate_system=resolved_coordinate_system,
            label_name=resolved_label_name,
            image_name=resolved_image_name,
            channels=normalized_channels,
        )
        for resolved_label_name, resolved_coordinate_system, resolved_image_name in zip(
            normalized_labels,
            coordinate_systems,
            image_names,
            strict=True,
        )
    )

    return HeadlessFeatureTarget(
        table_name=_normalize_nonempty_str(table_name, "table_name"),
        feature_key=resolved_feature_key,
        triplets=triplets,
        overwrite_feature_key=overwrite_feature_key,
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


def _normalize_name_sequence(value: str | Sequence[str], name: str) -> tuple[str, ...]:
    values = (value,) if isinstance(value, str) else tuple(value)
    if not values:
        raise ValueError(f"{name} must contain at least one name.")

    normalized = tuple(_normalize_nonempty_str(item, name) for item in values)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} must not contain duplicate names.")
    return normalized


def _normalize_parallel_names(value: str | Sequence[str], expected_count: int, name: str) -> tuple[str, ...]:
    values = (value,) if isinstance(value, str) else tuple(value)
    if not values:
        raise ValueError(f"{name} must contain at least one name.")
    normalized = tuple(_normalize_nonempty_str(item, name) for item in values)
    return _broadcast_parallel_values(normalized, expected_count, name)


def _normalize_optional_parallel_names(
    value: str | Sequence[str] | None,
    expected_count: int,
    name: str,
) -> tuple[str | None, ...]:
    if value is None:
        return (None,) * expected_count

    values = (value,) if isinstance(value, str) else tuple(value)
    if not values:
        raise ValueError(f"{name} must contain at least one name when provided.")

    normalized = tuple(str(item).strip() or None for item in values)
    return _broadcast_parallel_values(normalized, expected_count, name)


def _broadcast_parallel_values(values: tuple[T, ...], expected_count: int, name: str) -> tuple[T, ...]:
    if len(values) == 1:
        return values * expected_count
    if len(values) != expected_count:
        raise ValueError(f"{name} must be a single value or contain exactly {expected_count} entries.")
    return values


def _resolve_coordinate_systems_for_labels(
    sdata: SpatialData,
    labels_name: tuple[str, ...],
    coordinate_system: str | Sequence[str] | None,
) -> tuple[str, ...]:
    """Return one validated coordinate system per labels element.

    Explicit coordinate-system input is broadcast or matched one-to-one against
    `labels_name`, then checked against each labels element. If no coordinate
    system is provided, inference succeeds only when every labels element
    exposes exactly one coordinate system.
    """
    if coordinate_system is not None:
        coordinate_systems = _normalize_parallel_names(coordinate_system, len(labels_name), "coordinate_system")
        for label_name, resolved_coordinate_system in zip(labels_name, coordinate_systems, strict=True):
            available_coordinate_systems = _get_label_coordinate_systems(sdata, label_name)
            if resolved_coordinate_system not in available_coordinate_systems:
                available = ", ".join(f"`{name}`" for name in available_coordinate_systems)
                raise ValueError(
                    f"Labels element `{label_name}` is not available in coordinate system "
                    f"`{resolved_coordinate_system}`. Available coordinate systems: {available}."
                )
        return coordinate_systems

    return _infer_coordinate_systems_for_labels(sdata, labels_name)


def _infer_coordinate_systems_for_labels(sdata: SpatialData, labels_name: tuple[str, ...]) -> tuple[str, ...]:
    coordinate_systems_by_label = {
        label_name: _get_label_coordinate_systems(sdata, label_name) for label_name in labels_name
    }
    if all(len(coordinate_systems) == 1 for coordinate_systems in coordinate_systems_by_label.values()):
        return tuple(coordinate_systems[0] for coordinate_systems in coordinate_systems_by_label.values())

    options = "; ".join(
        f"{label_name}: {', '.join(coordinate_systems)}"
        for label_name, coordinate_systems in coordinate_systems_by_label.items()
    )
    raise ValueError(
        "Could not infer a unique coordinate system for feature extraction. "
        f"Pass `coordinate_system` explicitly. Available coordinate systems by labels element: {options}."
    )


def _get_label_coordinate_systems(sdata: SpatialData, label_name: str) -> tuple[str, ...]:
    if label_name not in sdata.labels:
        raise ValueError(f"Labels element `{label_name}` is not available in the target SpatialData object.")

    coordinate_systems = _get_element_coordinate_systems(sdata.labels[label_name])
    if not coordinate_systems:
        raise ValueError(f"Labels element `{label_name}` does not expose any coordinate systems.")
    return coordinate_systems


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
