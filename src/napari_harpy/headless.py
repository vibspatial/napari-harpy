from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from napari_harpy._annotation import USER_CLASS_COLORS_KEY
from napari_harpy._classifier_core import CLASSIFIER_APPLY_CONFIG_KEY, CLASSIFIER_CONFIG_KEY, ClassifierApplyResult
from napari_harpy._classifier_core import apply_classifier as _apply_classifier
from napari_harpy._classifier_export import ClassifierExportBundle, read_classifier_export_bundle
from napari_harpy._persistence_core import write_table_prediction_state

if TYPE_CHECKING:
    from spatialdata import SpatialData


def load_classifier(path: str | Path) -> ClassifierExportBundle:
    """Load a trusted classifier export bundle from disk."""
    return read_classifier_export_bundle(path)


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
