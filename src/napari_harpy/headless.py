from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from napari_harpy._classifier_core import ClassifierApplyResult
from napari_harpy._classifier_core import apply_classifier as _apply_classifier
from napari_harpy._classifier_export import ClassifierExportBundle, read_classifier_export_bundle

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
    """Apply an exported classifier bundle to an existing compatible feature matrix."""
    return _apply_classifier(
        sdata,
        bundle,
        table_name=table_name,
        feature_key=feature_key,
        prediction_regions=prediction_regions,
        pred_class_column=pred_class_column,
        pred_confidence_column=pred_confidence_column,
        classifier_path=classifier_path,
    )


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
    """Load a classifier artifact from disk and apply it to an existing feature matrix."""
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
