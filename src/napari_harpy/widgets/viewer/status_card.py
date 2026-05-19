from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from napari_harpy.widgets.shared_styles import StatusCardKind, format_feedback_identifier, validate_status_card_kind

if TYPE_CHECKING:
    from napari_harpy.core._color_source import TableColorSourceSpec
    from napari_harpy.viewer.image_styling import ImageLoadResult
    from napari_harpy.viewer.labels_styling import LabelsLoadResult
    from napari_harpy.viewer.points_styling import PointsLayerResult
    from napari_harpy.viewer.shapes_styling import ShapesLoadResult
    from napari_harpy.widgets.viewer.image_widget import ImageLoadRequest
    from napari_harpy.widgets.viewer.labels_widget import LabelsLoadRequest
    from napari_harpy.widgets.viewer.points_controller import PointsLoadResult
    from napari_harpy.widgets.viewer.shapes_widget import ShapesLoadRequest


@dataclass(frozen=True)
class _ViewerStatusCardSpec:
    title: str
    lines: tuple[str, ...]
    kind: StatusCardKind
    tooltip_message: str | None = None

    def __post_init__(self) -> None:
        validate_status_card_kind(self.kind)


def build_viewer_error_card_spec(title: str, lines: Sequence[str]) -> _ViewerStatusCardSpec:
    return _ViewerStatusCardSpec(
        title=title,
        lines=tuple(lines),
        kind="error",
    )


def build_points_layer_card_spec(
    load_result: PointsLoadResult,
    layer_result: PointsLayerResult,
) -> _ViewerStatusCardSpec:
    selection = load_result.selection
    action = "Created" if layer_result.created else "Updated"
    lines = [
        (
            f"{action} points layer for `{load_result.identity.points_name}` "
            f"by `{selection.index_column}` with {selection.loaded_count:,} point(s)."
        )
    ]
    kind: StatusCardKind = "success"
    title = f"Points Layer {action}"
    if selection.warning:
        kind = "warning"
        title = _with_warning_suffix(title)
        lines.append(selection.warning)
    if layer_result.categorical_coloring_disabled:
        kind = "warning"
        title = _with_warning_suffix(title)
        lines.append(
            f"Categorical coloring is disabled for {layer_result.selected_value_count:,} selected values; "
            f"using one solid color because the categorical limit is {layer_result.categorical_limit:,}."
        )

    return _ViewerStatusCardSpec(
        title=title,
        lines=tuple(lines),
        kind=kind,
    )


def build_primary_labels_loaded_card_spec(
    request: LabelsLoadRequest,
    result: LabelsLoadResult,
    coordinate_system: str,
) -> _ViewerStatusCardSpec:
    del result
    labels_name = request.labels_name
    display_name, was_shortened = format_feedback_identifier(labels_name)
    return _ViewerStatusCardSpec(
        title="Labels Loaded",
        lines=(f"Loaded labels `{display_name}` in coordinate system `{coordinate_system}`.",),
        kind="success",
        tooltip_message=f"Loaded labels `{labels_name}` in coordinate system `{coordinate_system}`."
        if was_shortened
        else None,
    )


def build_styled_labels_card_spec(
    request: LabelsLoadRequest,
    result: LabelsLoadResult,
    coordinate_system: str,
) -> _ViewerStatusCardSpec:
    selected_color_source = request.selected_color_source
    if selected_color_source is None:
        return build_viewer_error_card_spec(
            "Colored Overlay Error",
            (f"Select a color source to create a colored overlay for `{request.labels_name}`.",),
        )

    action = "Created" if result.created else "Updated"
    source_text = _format_table_color_source(selected_color_source)
    lines = [
        (
            f"{action} colored overlay for {source_text} on labels element `{request.labels_name}` "
            f"in coordinate system `{coordinate_system}`."
        )
    ]
    title = f"Colored Overlay {action}"
    kind = _append_palette_status_lines(
        lines,
        result=result,
        include_instance_message=True,
    )
    return _ViewerStatusCardSpec(
        title=_palette_status_title(title, result, include_instance_message=True),
        lines=tuple(lines),
        kind=kind,
    )


def build_primary_shapes_loaded_card_spec(
    request: ShapesLoadRequest,
    result: ShapesLoadResult,
    coordinate_system: str,
) -> _ViewerStatusCardSpec:
    shapes_name = request.shapes_name
    display_name, was_shortened = format_feedback_identifier(shapes_name)
    title = "Shapes Loaded"
    kind: StatusCardKind = "success"
    lines = [f"Loaded shapes `{display_name}` in coordinate system `{coordinate_system}`."]
    if result.skipped_geometry_count:
        title = "Shapes Loaded With Warning"
        kind = "warning"
        lines.append(_format_skipped_geometry_line(result.skipped_geometry_count))

    return _ViewerStatusCardSpec(
        title=title,
        lines=tuple(lines),
        kind=kind,
        tooltip_message=f"Loaded shapes `{shapes_name}` in coordinate system `{coordinate_system}`."
        if was_shortened
        else None,
    )


def build_styled_shapes_card_spec(
    request: ShapesLoadRequest,
    result: ShapesLoadResult,
    coordinate_system: str,
) -> _ViewerStatusCardSpec:
    selected_color_source = request.selected_color_source
    if selected_color_source is None:
        return build_viewer_error_card_spec(
            "Styled Shapes Error",
            (f"Select a shapes column to create a styled shapes layer for `{request.shapes_name}`.",),
        )

    action = "Created" if result.created else "Updated"
    lines = [
        (
            f'{action} styled shapes layer for column "{selected_color_source.value_key}" '
            f"on shapes element `{request.shapes_name}` in coordinate system `{coordinate_system}`."
        )
    ]
    title = f"Styled Shapes {action}"
    kind = _append_palette_status_lines(
        lines,
        result=result,
        include_instance_message=False,
    )
    title = _palette_status_title(title, result, include_instance_message=False)
    if result.skipped_geometry_count:
        kind = "warning"
        title = _with_warning_suffix(title)
        lines.append(_format_skipped_geometry_line(result.skipped_geometry_count))

    return _ViewerStatusCardSpec(
        title=title,
        lines=tuple(lines),
        kind=kind,
    )


def build_image_loaded_card_spec(
    request: ImageLoadRequest,
    result: ImageLoadResult,
    coordinate_system: str,
) -> _ViewerStatusCardSpec:
    image_name = request.image_name
    display_name, was_shortened = format_feedback_identifier(image_name)
    if result.mode == "stack":
        line = f"Loaded image `{display_name}` in stack mode for coordinate system `{coordinate_system}`."
        tooltip_line = f"Loaded image `{image_name}` in stack mode for coordinate system `{coordinate_system}`."
    else:
        line = (
            f"Loaded image `{display_name}` in overlay mode for channels {list(result.channels)} "
            f"in coordinate system `{coordinate_system}`."
        )
        tooltip_line = (
            f"Loaded image `{image_name}` in overlay mode for channels {list(result.channels)} "
            f"in coordinate system `{coordinate_system}`."
        )

    return _ViewerStatusCardSpec(
        title="Image Loaded",
        lines=(line,),
        kind="success",
        tooltip_message=tooltip_line if was_shortened else None,
    )


def _format_table_color_source(color_source: TableColorSourceSpec) -> str:
    if color_source.source_kind == "obs_column":
        return f'obs["{color_source.value_key}"]'
    return f'X[:, "{color_source.value_key}"]'


def _append_palette_status_lines(
    lines: list[str],
    *,
    result: LabelsLoadResult | ShapesLoadResult,
    include_instance_message: bool,
) -> StatusCardKind:
    if include_instance_message and result.value_kind == "instance":
        lines.append("Used instance label colors.")
        return "success"
    if result.coercion_applied:
        lines.append("Coerced string values to categorical and used the default categorical palette.")
        return "warning"
    if result.palette_source == "stored":
        lines.append("Used the stored categorical palette.")
        return "success"
    if result.palette_source == "default_invalid":
        lines.append("The stored categorical palette was invalid, so Harpy used the default categorical palette.")
        return "warning"
    if result.palette_source == "default_missing":
        lines.append("Used the default categorical palette because no stored palette was present.")
        return "info"
    return "success"


def _palette_status_title(
    title: str,
    result: LabelsLoadResult | ShapesLoadResult,
    *,
    include_instance_message: bool,
) -> str:
    if include_instance_message and result.value_kind == "instance":
        return title
    if result.coercion_applied or result.palette_source == "default_invalid":
        return _with_warning_suffix(title)
    return title


def _format_skipped_geometry_line(skipped_geometry_count: int) -> str:
    return (
        f"Skipped {skipped_geometry_count} empty, invalid, or unsupported geometries while loading renderable shapes."
    )


def _with_warning_suffix(title: str) -> str:
    if title.endswith(" With Warning"):
        return title
    return f"{title} With Warning"
