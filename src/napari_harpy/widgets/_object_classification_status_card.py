from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from napari_harpy._annotation import UNLABELED_CLASS
from napari_harpy._classifier import ClassifierPreparationSummary
from napari_harpy.widgets._shared_styles import StatusCardKind

_LabelsLayerPreparationKind = Literal["none", "loaded", "activated", "error"]


@dataclass(frozen=True)
class _LabelsLayerPreparationResult:
    kind: _LabelsLayerPreparationKind
    label_name: str | None = None
    coordinate_system: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class _ObjectClassificationStatusCardSpec:
    title: str
    lines: tuple[str, ...]
    kind: StatusCardKind
    tooltip_message: str | None = None


def build_object_classification_selection_status_card_spec(
    *,
    has_spatialdata: bool,
    selected_coordinate_system: str | None,
    selected_segmentation_name: str | None,
    labels_layer_loaded: bool,
    labels_layer_preparation_result: _LabelsLayerPreparationResult,
    selected_table_name: str | None,
    table_binding_error: str | None,
    missing_table_row_message: str | None,
    selected_instance_id: int | None,
    instance_key_name: str,
    current_user_class: int | None,
) -> _ObjectClassificationStatusCardSpec:
    layer_preparation_lines = _labels_layer_preparation_lines(labels_layer_preparation_result)

    if not has_spatialdata:
        return _ObjectClassificationStatusCardSpec(
            title="No SpatialData Loaded",
            lines=(
                "Load a SpatialData object through the Harpy Viewer widget, reader, or `Interactive(sdata)`.",
                "This form updates automatically from the shared Harpy state.",
            ),
            kind="warning",
        )

    if selected_coordinate_system is None:
        return _ObjectClassificationStatusCardSpec(
            title="Selection",
            lines=("Choose a coordinate system to continue configuring object classification.",),
            kind="info",
        )

    if selected_segmentation_name is None:
        return _ObjectClassificationStatusCardSpec(
            title="Selection",
            lines=("Choose a segmentation mask in the selected coordinate system to enable object picking.",),
            kind="info",
        )

    if labels_layer_preparation_result.kind == "error":
        return _ObjectClassificationStatusCardSpec(
            title="Layer Load Issue",
            lines=(labels_layer_preparation_result.error or "Could not load the selected segmentation layer.",),
            kind="warning",
        )

    if not labels_layer_loaded:
        return _ObjectClassificationStatusCardSpec(
            title="Selection",
            lines=(
                "The chosen segmentation is known in SpatialData for the selected coordinate system, "
                "but is not currently loaded as a napari Labels layer.",
            ),
            kind="warning",
        )

    if selected_table_name is None:
        return _ObjectClassificationStatusCardSpec(
            title="Selection Warning",
            lines=(
                *layer_preparation_lines,
                f"Bound to {selected_segmentation_name}.",
                "This segmentation is loaded, but no annotation table is linked to it.",
            ),
            kind="warning",
        )

    if table_binding_error is not None:
        return _ObjectClassificationStatusCardSpec(
            title="Selection Warning",
            lines=(
                *layer_preparation_lines,
                f"Bound to {selected_segmentation_name}.",
                table_binding_error,
            ),
            kind="warning",
        )

    if selected_instance_id is None:
        return _ObjectClassificationStatusCardSpec(
            title="Selection",
            lines=(
                *layer_preparation_lines,
                f"Bound to {selected_segmentation_name}.",
                "Click an object in the viewer.",
            ),
            kind="info",
        )

    if missing_table_row_message is not None:
        return _ObjectClassificationStatusCardSpec(
            title="Selection Warning",
            lines=(
                f"Bound to {selected_segmentation_name}.",
                missing_table_row_message,
            ),
            kind="warning",
        )

    current_class_label = "unlabeled" if current_user_class in (None, UNLABELED_CLASS) else str(current_user_class)
    return _ObjectClassificationStatusCardSpec(
        title="Selection Ready",
        lines=(
            f"Bound to {selected_segmentation_name}.",
            f"Current {instance_key_name}: {selected_instance_id}.",
            f"Current class: {current_class_label}.",
        ),
        kind="success",
    )


def build_object_classification_classifier_preparation_card_spec(
    *,
    selected_segmentation_name: str | None,
    selected_table_name: str | None,
    selected_feature_key: str | None,
    table_binding_error: str | None,
    summary: ClassifierPreparationSummary | None,
) -> _ObjectClassificationStatusCardSpec | None:
    if (
        selected_segmentation_name is None
        or selected_table_name is None
        or selected_feature_key is None
        or table_binding_error is not None
        or summary is None
    ):
        return None

    hidden_prediction_regions = tuple(
        region for region in summary.prediction_scope.regions if region != selected_segmentation_name
    )
    kind: StatusCardKind = "warning" if not summary.eligible else "success"
    lines = [
        _format_training_line(summary),
        _format_prediction_line(summary),
        _format_feature_matrix_line(selected_feature_key, summary.n_features),
    ]
    if not summary.eligible and summary.reason:
        lines.append(summary.reason)
    if hidden_prediction_regions:
        lines.append("Some prediction updates may not be visible in the current selection.")

    return _ObjectClassificationStatusCardSpec(
        title="Classifier Preparation",
        lines=tuple(lines),
        kind=kind,
    )


def build_object_classification_classifier_feedback_card_spec(
    *,
    is_visible: bool,
    message: str,
    kind: StatusCardKind,
) -> _ObjectClassificationStatusCardSpec | None:
    if not is_visible or not message:
        return None

    title_by_kind = {
        "error": "Classifier Error",
        "warning": "Classifier Warning",
        "success": "Classifier Ready",
        "info": "Classifier",
    }
    return _ObjectClassificationStatusCardSpec(
        title=title_by_kind.get(kind, "Classifier"),
        lines=(message.removeprefix("Classifier: ").strip(),),
        kind=kind,
    )


def _labels_layer_preparation_lines(result: _LabelsLayerPreparationResult) -> tuple[str, ...]:
    line = _format_labels_layer_preparation_line(result)
    return () if line is None else (line,)


def _format_labels_layer_preparation_line(result: _LabelsLayerPreparationResult) -> str | None:
    if result.kind not in ("loaded", "activated") or result.label_name is None or result.coordinate_system is None:
        return None

    verb = "Loaded" if result.kind == "loaded" else "Activated"
    return f"{verb} segmentation `{result.label_name}` in coordinate system `{result.coordinate_system}`."


def _format_training_line(summary: ClassifierPreparationSummary) -> str:
    return (
        f"Training: {_format_count(summary.labeled_count, 'labeled row')} "
        f"across {_format_count(summary.training_region_count, 'region')}."
    )


def _format_prediction_line(summary: ClassifierPreparationSummary) -> str:
    prediction_count = _format_count(summary.resolved_prediction_row_count, "eligible row")
    region_count = len(summary.prediction_scope.regions)
    if summary.prediction_scope.mode == "selected_segmentation_only" and region_count <= 1:
        scope_text = "in selected region"
    else:
        scope_text = f"across {_format_count(region_count, 'region')}"
    return f"Prediction: {prediction_count} {scope_text}."


def _format_feature_matrix_line(feature_key: str, n_features: int | None) -> str:
    if n_features is None:
        return f"Feature matrix: `{feature_key}`."
    return f"Feature matrix: `{feature_key}`, {_format_count(n_features, 'feature')}."


def _format_count(count: int, singular: str, plural: str | None = None) -> str:
    word = singular if count == 1 else (plural or f"{singular}s")
    return f"{count:,} {word}"
