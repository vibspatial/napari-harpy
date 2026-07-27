"""Pure status-card specifications for the Spatial Query child."""

from __future__ import annotations

from dataclasses import dataclass

from napari_harpy.core.spatial_query import (
    CanonicalCacheReport,
    CanonicalCacheState,
    SpatialAnnotationSummary,
)
from napari_harpy.widgets.shared_styles import StatusCardKind, validate_status_card_kind

_FIRST_CALCULATION_TOOLTIP = (
    "Centers will first be calculated for the selected labels element before the spatial query runs."
)
_CENTER_CALCULATION_TOOLTIP = (
    "Centers will be calculated for the selected labels element before the spatial query runs."
)


@dataclass(frozen=True)
class _SpatialQueryStatusCardSpec:
    title: str
    lines: tuple[str, ...]
    kind: StatusCardKind
    tooltip_message: str | None = None

    def __post_init__(self) -> None:
        validate_status_card_kind(self.kind)


def build_spatial_query_execution_status_card_spec(
    *,
    message: str,
    kind: StatusCardKind,
    is_running: bool,
) -> _SpatialQueryStatusCardSpec:
    """Build the temporary execution outcome shown by the unified status card."""
    if not isinstance(message, str) or not message:
        raise ValueError("Spatial Query execution status requires a non-empty message.")
    if is_running:
        title = "Spatial Query Running"
    elif kind == "error":
        title = "Spatial Query Failed"
    else:
        title = "Spatial Query Complete"
    return _SpatialQueryStatusCardSpec(
        title=title,
        lines=(message,),
        kind=kind,
    )


def build_spatial_annotation_outcome_status_card_spec(
    summary: SpatialAnnotationSummary,
    *,
    layer_styling_error: str | None = None,
) -> _SpatialQueryStatusCardSpec:
    """Build the final status for one accepted annotation Apply."""
    if not isinstance(summary, SpatialAnnotationSummary):
        raise TypeError("Spatial annotation outcome status requires a SpatialAnnotationSummary.")

    if summary.changed_count == 0:
        if summary.is_removal:
            lines = (
                f"No annotations needed removal across {summary.matched_count} matched labeled objects.",
                f"Already missing: {summary.unchanged_count}.",
            )
        else:
            lines = (
                f"All {summary.matched_count} matched labeled objects already have "
                f"{_format_annotation_value(summary.annotation_value)}.",
                f"Already equal: {summary.unchanged_count}.",
            )
        return _SpatialQueryStatusCardSpec(
            title="No Annotation Changes",
            lines=lines,
            kind="info",
        )

    if summary.is_removal:
        lines = (
            f"Removed annotations from {summary.removal_count} matched labeled objects.",
            f"Already missing: {summary.unchanged_count}.",
        )
    else:
        lines = (
            f"Applied {_format_annotation_value(summary.annotation_value)} "
            f"to {summary.changed_count} matched labeled objects.",
            f"Overwritten: {summary.overwrite_count}. Already equal: {summary.unchanged_count}.",
        )

    if layer_styling_error is not None:
        lines = (*lines, f"Labels styling could not be refreshed: {layer_styling_error}")
    return _SpatialQueryStatusCardSpec(
        title="Annotation Applied",
        lines=lines,
        kind="warning" if layer_styling_error is not None else "success",
    )


def build_spatial_annotation_failure_status_card_spec(
    error: str,
) -> _SpatialQueryStatusCardSpec:
    """Build the final status for a rejected annotation Apply."""
    if not isinstance(error, str) or not error:
        raise ValueError("Spatial annotation failure status requires a non-empty error.")
    return _SpatialQueryStatusCardSpec(
        title="Annotation Failed",
        lines=(error, "Review the current annotation inputs and apply again."),
        kind="error",
    )


def build_spatial_query_status_card_spec(
    *,
    has_spatialdata: bool,
    coordinate_system: str | None,
    saved_shapes_name: str | None,
    has_unsaved_shapes_changes: bool,
    labels_name: str | None,
    table_name: str | None,
    cache_report: CanonicalCacheReport | None,
    canonical_input_inspection_error: str | None,
    annotation_column_error: str | None,
    annotation_column_description: str | None,
    annotation_mutation_error: str | None,
    annotation_mutation_description: str | None,
    layer_styling_error: str | None,
) -> _SpatialQueryStatusCardSpec:
    """Build the unified Spatial Query status from already-derived child state."""
    if not has_spatialdata:
        return _SpatialQueryStatusCardSpec(
            title="No SpatialData Loaded",
            lines=("Load a SpatialData object before configuring Spatial Query.",),
            kind="warning",
        )
    if coordinate_system is None:
        return _SpatialQueryStatusCardSpec(
            title="Coordinate System Required",
            lines=("Choose a coordinate system in the Annotation widget.",),
            kind="warning",
        )
    if saved_shapes_name is None:
        return _SpatialQueryStatusCardSpec(
            title="Saved Shapes Required",
            lines=("Select an existing Shapes element or save the new Shapes annotation first.",),
            kind="warning",
        )
    if has_unsaved_shapes_changes:
        return _SpatialQueryStatusCardSpec(
            title="Save or Discard Shapes Changes",
            lines=("Spatial Query uses saved in-memory geometry and cannot run while the selected Shapes is dirty.",),
            kind="warning",
        )
    if labels_name is None:
        return _SpatialQueryStatusCardSpec(
            title="Labels Required",
            lines=("Choose a supported 2D labels element.",),
            kind="warning",
        )
    if table_name is None:
        return _SpatialQueryStatusCardSpec(
            title="Linked Table Required",
            lines=(f'No linked table is selected for "{labels_name}".',),
            kind="warning",
        )
    if cache_report is None:
        if canonical_input_inspection_error is None:
            raise ValueError("A missing canonical cache report requires a canonical input inspection error.")
        return _SpatialQueryStatusCardSpec(
            title="Labels or Table Validation Failed",
            lines=(
                canonical_input_inspection_error,
                "Spatial Query cannot calculate centers until this issue is resolved.",
            ),
            kind="error",
        )
    if canonical_input_inspection_error is not None:
        raise ValueError("A canonical cache report and canonical input inspection error cannot be supplied together.")
    if annotation_column_error is not None:
        return _SpatialQueryStatusCardSpec(
            title="Annotation Column Not Ready",
            lines=(annotation_column_error,),
            kind="warning",
        )
    if annotation_mutation_error is not None:
        return _SpatialQueryStatusCardSpec(
            title="Annotation Value Required",
            lines=(annotation_mutation_error,),
            kind="warning",
        )
    if layer_styling_error is not None:
        return _SpatialQueryStatusCardSpec(
            title="Layer Styling Warning",
            lines=(layer_styling_error, "Spatial Query can still run."),
            kind="warning",
        )

    cache_line, kind, tooltip_message = _build_ready_cache_status(
        cache_report,
        labels_name,
    )
    return _SpatialQueryStatusCardSpec(
        title="Spatial Query Ready",
        lines=(
            f'Shapes "{saved_shapes_name}" will query labels "{labels_name}".',
            f"Target: {annotation_column_description or 'unknown annotation column'}.",
            annotation_mutation_description or "Annotation action is not configured.",
            cache_line,
        ),
        kind=kind,
        tooltip_message=tooltip_message,
    )


def _build_ready_cache_status(
    report: CanonicalCacheReport,
    labels_name: str,
) -> tuple[str, StatusCardKind, str | None]:
    if report.state is CanonicalCacheState.VALID:
        return f'Cached centers for "{labels_name}" will be reused.', "success", None
    if report.state is CanonicalCacheState.ABSENT:
        return (
            f'Centers for labels element "{labels_name}" will be calculated before querying.',
            "info",
            _FIRST_CALCULATION_TOOLTIP,
        )
    if report.state is CanonicalCacheState.PARTIAL:
        return (
            f'Centers for labels element "{labels_name}" will be calculated before querying.',
            "info",
            _CENTER_CALCULATION_TOOLTIP,
        )
    if report.state is CanonicalCacheState.STALE:
        return (
            f'Centers for labels element "{labels_name}" will be refreshed before querying.',
            "warning",
            _CENTER_CALCULATION_TOOLTIP,
        )
    if report.state is CanonicalCacheState.INVALID:
        mismatch = report.mismatches[0]
        return (
            f'Centers for labels element "{labels_name}" will be recalculated before querying.',
            "info",
            mismatch.detail or _CENTER_CALCULATION_TOOLTIP,
        )

    raise ValueError(f"Unsupported canonical cache state `{report.state}`.")


def _format_annotation_value(value: object) -> str:
    if isinstance(value, str):
        return f'"{value}"'
    return str(value)
