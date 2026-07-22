"""Pure status-card specifications for the Spatial Query child."""

from __future__ import annotations

from dataclasses import dataclass

from napari_harpy.core.spatial_query import CanonicalCacheReport, CanonicalCacheState
from napari_harpy.widgets.shared_styles import StatusCardKind, validate_status_card_kind

_FIRST_CALCULATION_TOOLTIP = (
    "Centers will first be calculated for the selected labels element before the spatial query runs."
)


@dataclass(frozen=True)
class _SpatialQueryStatusCardSpec:
    title: str
    lines: tuple[str, ...]
    kind: StatusCardKind
    tooltip_message: str | None = None

    def __post_init__(self) -> None:
        validate_status_card_kind(self.kind)


def build_spatial_query_cache_status_card_spec(
    report: CanonicalCacheReport | None,
) -> _SpatialQueryStatusCardSpec:
    """Build the current canonical-cache status presentation."""
    if report is None:
        return _SpatialQueryStatusCardSpec(
            title="Centroids Unavailable",
            lines=("Choose a supported labels element and linked table.",),
            kind="warning",
        )

    labels_name = report.labels_name
    if report.state is CanonicalCacheState.VALID:
        return _SpatialQueryStatusCardSpec(
            title="Centroids Ready",
            lines=(f'Cached centers for "{labels_name}" will be reused.',),
            kind="success",
        )
    if report.state is CanonicalCacheState.ABSENT:
        return _SpatialQueryStatusCardSpec(
            title="Centroids Not Calculated",
            lines=(f'Run will calculate centers for "{labels_name}" first.',),
            kind="info",
            tooltip_message=_FIRST_CALCULATION_TOOLTIP,
        )
    if report.state is CanonicalCacheState.PARTIAL:
        return _SpatialQueryStatusCardSpec(
            title="Centroids Partial",
            lines=(f'Run will calculate centers for labels region "{labels_name}".',),
            kind="info",
            tooltip_message=_FIRST_CALCULATION_TOOLTIP,
        )
    if report.state is CanonicalCacheState.STALE:
        return _SpatialQueryStatusCardSpec(
            title="Centroids Stale",
            lines=(f'Run will refresh centers for labels region "{labels_name}".',),
            kind="warning",
            tooltip_message=_FIRST_CALCULATION_TOOLTIP,
        )

    if report.state is CanonicalCacheState.INVALID:
        mismatch = report.mismatches[0]
        mismatch_name = mismatch.code.value.replace("_", " ")
        return _SpatialQueryStatusCardSpec(
            title="Centroids Invalid",
            lines=(
                f'Run will rebuild the managed centroid cache before querying "{labels_name}".',
                f"Detected: {mismatch_name}.",
            ),
            kind="warning",
            tooltip_message=mismatch.detail or _FIRST_CALCULATION_TOOLTIP,
        )

    raise ValueError(f"Unsupported canonical cache state `{report.state}`.")


def build_spatial_query_readiness_status_card_spec(
    *,
    has_spatialdata: bool,
    coordinate_system: str | None,
    saved_shapes_name: str | None,
    has_unsaved_shapes_changes: bool,
    labels_name: str | None,
    table_name: str | None,
    has_cache_report: bool,
    target_error: str | None,
    target_description: str | None,
    layer_styling_error: str | None,
) -> _SpatialQueryStatusCardSpec:
    """Build Run readiness from already-derived child state."""
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
    if not has_cache_report:
        return _SpatialQueryStatusCardSpec(
            title="Centroids Unavailable",
            lines=("Resolve the centroid-cache error before running Spatial Query.",),
            kind="error",
        )
    if target_error is not None:
        return _SpatialQueryStatusCardSpec(
            title="Annotation Target Not Ready",
            lines=(target_error,),
            kind="warning",
        )
    if layer_styling_error is not None:
        return _SpatialQueryStatusCardSpec(
            title="Layer Styling Warning",
            lines=(layer_styling_error,),
            kind="warning",
        )

    return _SpatialQueryStatusCardSpec(
        title="Spatial Query Ready",
        lines=(
            f'Shapes "{saved_shapes_name}" will query labels "{labels_name}".',
            f"Target: {target_description or 'unknown annotation column'}.",
        ),
        kind="success",
    )
