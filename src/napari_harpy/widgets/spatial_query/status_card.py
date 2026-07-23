"""Pure status-card specifications for the Spatial Query child."""

from __future__ import annotations

from dataclasses import dataclass

from napari_harpy.core.spatial_query import CanonicalCacheReport, CanonicalCacheState
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
    target_error: str | None,
    target_description: str | None,
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
    if target_error is not None:
        return _SpatialQueryStatusCardSpec(
            title="Annotation Target Not Ready",
            lines=(target_error,),
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
            f"Target: {target_description or 'unknown annotation column'}.",
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
