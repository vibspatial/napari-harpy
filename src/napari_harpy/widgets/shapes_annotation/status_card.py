from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from napari_harpy.widgets.shared_styles import StatusCardKind, format_feedback_identifier, validate_status_card_kind

_STATUS_IDENTIFIER_MAX_LENGTH = 32

if TYPE_CHECKING:
    from napari_harpy.core.shapes_annotation import AnnotateShapesElementResult


@dataclass(frozen=True)
class _ShapesAnnotationStatusCardSpec:
    title: str
    lines: tuple[str, ...]
    kind: StatusCardKind
    tooltip_message: str | None = None

    def __post_init__(self) -> None:
        validate_status_card_kind(self.kind)


def build_annotation_save_unavailable_card_spec(message: str) -> _ShapesAnnotationStatusCardSpec:
    return _ShapesAnnotationStatusCardSpec(
        title="Cannot Save Shapes",
        lines=(message,),
        kind="warning",
    )


# Create/open target readiness


def build_annotation_no_spatialdata_card_spec() -> _ShapesAnnotationStatusCardSpec:
    return _ShapesAnnotationStatusCardSpec(
        title="No SpatialData Loaded",
        lines=("Load a SpatialData object before creating shapes.",),
        kind="warning",
    )


def build_annotation_no_coordinate_systems_card_spec() -> _ShapesAnnotationStatusCardSpec:
    return _ShapesAnnotationStatusCardSpec(
        title="No Coordinate Systems",
        lines=("The loaded SpatialData object does not expose any coordinate systems.",),
        kind="warning",
    )


def build_annotation_coordinate_system_missing_card_spec() -> _ShapesAnnotationStatusCardSpec:
    return _ShapesAnnotationStatusCardSpec(
        title="Choose Coordinate System",
        lines=("Select a coordinate system before creating shapes.",),
        kind="warning",
    )


def build_annotation_target_missing_card_spec() -> _ShapesAnnotationStatusCardSpec:
    return _ShapesAnnotationStatusCardSpec(
        title="Choose Shapes Target",
        lines=("Select whether to create a new shapes element or edit an existing one.",),
        kind="warning",
    )


def build_annotation_shapes_unavailable_card_spec() -> _ShapesAnnotationStatusCardSpec:
    return _ShapesAnnotationStatusCardSpec(
        title="Shapes Unavailable",
        lines=("The selected shapes element is no longer available in this coordinate system.",),
        kind="warning",
    )


def build_annotation_existing_target_ready_card_spec(
    *,
    shapes_name: str,
    coordinate_system: str,
) -> _ShapesAnnotationStatusCardSpec:
    visible_shapes_name, shapes_name_shortened = format_feedback_identifier(
        shapes_name,
        max_length=_STATUS_IDENTIFIER_MAX_LENGTH,
    )
    visible_coordinate_system, coordinate_system_shortened = format_feedback_identifier(
        coordinate_system,
        max_length=_STATUS_IDENTIFIER_MAX_LENGTH,
    )
    tooltip_message = (
        f'Shapes element "{shapes_name}" is available in coordinate system "{coordinate_system}".'
        if shapes_name_shortened or coordinate_system_shortened
        else None
    )
    return _ShapesAnnotationStatusCardSpec(
        title="Ready",
        lines=(f'Selected shapes layer "{visible_shapes_name}" in coordinate system "{visible_coordinate_system}".',),
        kind="info",
        tooltip_message=tooltip_message,
    )


def build_annotation_invalid_shapes_name_card_spec(message: str) -> _ShapesAnnotationStatusCardSpec:
    return _ShapesAnnotationStatusCardSpec(
        title="Invalid Shapes Name",
        lines=(message,),
        kind="warning",
    )


def build_annotation_shapes_name_exists_card_spec(shapes_name: str) -> _ShapesAnnotationStatusCardSpec:
    visible_shapes_name, shortened = format_feedback_identifier(
        shapes_name,
        max_length=_STATUS_IDENTIFIER_MAX_LENGTH,
    )
    tooltip_message = f'Shapes element "{shapes_name}" already exists. Choose a different name.' if shortened else None
    return _ShapesAnnotationStatusCardSpec(
        title="Name Already Exists",
        lines=(f'Shapes element "{visible_shapes_name}" already exists. Choose a different name.',),
        kind="warning",
        tooltip_message=tooltip_message,
    )


def build_annotation_create_target_ready_card_spec(
    *,
    shapes_name: str,
    coordinate_system: str,
) -> _ShapesAnnotationStatusCardSpec:
    visible_shapes_name, shapes_name_shortened = format_feedback_identifier(
        shapes_name,
        max_length=_STATUS_IDENTIFIER_MAX_LENGTH,
    )
    visible_coordinate_system, coordinate_system_shortened = format_feedback_identifier(
        coordinate_system,
        max_length=_STATUS_IDENTIFIER_MAX_LENGTH,
    )
    tooltip_message = (
        f'Create shapes layer "{shapes_name}" in coordinate system "{coordinate_system}".'
        if shapes_name_shortened or coordinate_system_shortened
        else None
    )
    return _ShapesAnnotationStatusCardSpec(
        title="Ready",
        lines=(f'Create shapes layer "{visible_shapes_name}" in coordinate system "{visible_coordinate_system}".',),
        kind="info",
        tooltip_message=tooltip_message,
    )


# Save/action readiness


def build_annotation_existing_shapes_opened_card_spec(
    *,
    shapes_name: str,
    coordinate_system: str,
    table_linked: bool,
) -> _ShapesAnnotationStatusCardSpec:
    visible_shapes_name, shapes_name_shortened = format_feedback_identifier(
        shapes_name,
        max_length=_STATUS_IDENTIFIER_MAX_LENGTH,
    )
    visible_coordinate_system, coordinate_system_shortened = format_feedback_identifier(
        coordinate_system,
        max_length=_STATUS_IDENTIFIER_MAX_LENGTH,
    )
    lines = [f'Edit shapes layer "{visible_shapes_name}" in coordinate system "{visible_coordinate_system}".']
    if table_linked:
        lines.append("Linked tables are not updated by Annotation and may go out of sync if rows are added or removed.")

    tooltip_message = (
        f'Edit shapes layer "{shapes_name}" in coordinate system "{coordinate_system}".'
        if shapes_name_shortened or coordinate_system_shortened
        else None
    )
    return _ShapesAnnotationStatusCardSpec(
        title="Existing Shapes Opened",
        lines=tuple(lines),
        kind="info",
        tooltip_message=tooltip_message,
    )


def build_annotation_layer_ready_card_spec() -> _ShapesAnnotationStatusCardSpec:
    return _ShapesAnnotationStatusCardSpec(
        title="Annotation Layer Ready",
        lines=("Draw shapes in the viewer, then click Save shapes.",),
        kind="info",
    )


# Operation results


def build_annotation_save_success_card_spec(
    *,
    result: AnnotateShapesElementResult,
    table_linked: bool,
) -> _ShapesAnnotationStatusCardSpec:
    visible_shapes_name, shapes_name_shortened = format_feedback_identifier(
        result.shapes_name,
        max_length=_STATUS_IDENTIFIER_MAX_LENGTH,
    )
    visible_coordinate_system, coordinate_system_shortened = format_feedback_identifier(
        result.coordinate_system,
        max_length=_STATUS_IDENTIFIER_MAX_LENGTH,
    )
    lines = [
        (
            f'Saved "{visible_shapes_name}" with {result.row_count} shape(s) '
            f'in coordinate system "{visible_coordinate_system}".'
        )
    ]
    if table_linked:
        lines.append("Linked tables are not updated by Annotation and may go out of sync if rows are added or removed.")

    tooltip_message = (
        f'Saved "{result.shapes_name}" with {result.row_count} shape(s) '
        f'in coordinate system "{result.coordinate_system}".'
        if shapes_name_shortened or coordinate_system_shortened
        else None
    )
    return _ShapesAnnotationStatusCardSpec(
        title="Shapes Saved",
        lines=tuple(lines),
        kind="success",
        tooltip_message=tooltip_message,
    )


def build_annotation_save_error_card_spec(message: str) -> _ShapesAnnotationStatusCardSpec:
    return _ShapesAnnotationStatusCardSpec(
        title="Could Not Save Shapes",
        lines=(message,),
        kind="warning",
    )


def build_annotation_create_layer_error_card_spec(message: str) -> _ShapesAnnotationStatusCardSpec:
    return _ShapesAnnotationStatusCardSpec(
        title="Could Not Create Layer",
        lines=(message,),
        kind="warning",
    )


def build_annotation_open_shapes_error_card_spec(message: str) -> _ShapesAnnotationStatusCardSpec:
    return _ShapesAnnotationStatusCardSpec(
        title="Could Not Open Shapes",
        lines=(message,),
        kind="warning",
    )


def build_annotation_reload_shapes_error_card_spec(message: str) -> _ShapesAnnotationStatusCardSpec:
    return _ShapesAnnotationStatusCardSpec(
        title="Could Not Reload Shapes",
        lines=(message,),
        kind="warning",
    )


# Create holes


def build_create_holes_success_card_spec(
    *,
    hole_count: int,
    table_linked: bool,
) -> _ShapesAnnotationStatusCardSpec:
    lines = [f"Converted {hole_count} selected polygon(s) into hole(s) and removed their shape row(s)."]
    if table_linked:
        lines.append(
            "Linked tables are not updated automatically; after saving, table annotations may no longer match the "
            "shapes rows."
        )
    return _ShapesAnnotationStatusCardSpec(
        title="Created Holes",
        lines=tuple(lines),
        kind="success",
    )


def build_create_holes_error_card_spec(message: str) -> _ShapesAnnotationStatusCardSpec:
    return _ShapesAnnotationStatusCardSpec(
        title="Could Not Create Holes",
        lines=(message,),
        kind="warning",
    )


# Interactive edit guard


def build_annotation_edit_warning_card_spec(message: str) -> _ShapesAnnotationStatusCardSpec:
    return _ShapesAnnotationStatusCardSpec(
        title="Could Not Delete Vertex",
        lines=(message,),
        kind="warning",
    )
