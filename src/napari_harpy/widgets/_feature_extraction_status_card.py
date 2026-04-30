from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from napari_harpy.widgets._shared_styles import StatusCardKind, format_feedback_identifier

_FeatureExtractionTableBlocker = Literal["choose_table", "no_eligible", "invalid"] | None


@dataclass(frozen=True)
class _FeatureExtractionStatusCardEntry:
    coordinate_system: str
    label_name: str | None
    image_name: str | None
    blocking_reason: str | None = None

    @property
    def is_valid(self) -> bool:
        return self.blocking_reason is None


@dataclass(frozen=True)
class _FeatureExtractionStatusCardSpec:
    title: str
    lines: tuple[str, ...]
    kind: StatusCardKind
    tooltip_message: str | None = None


def build_feature_extraction_status_card_entries(
    checked_coordinate_systems: Sequence[str],
    *,
    label_names_by_coordinate_system: Mapping[str, str | None],
    image_names_by_coordinate_system: Mapping[str, str | None],
    blocking_reasons_by_coordinate_system: Mapping[str, str | None],
) -> tuple[_FeatureExtractionStatusCardEntry, ...]:
    return tuple(
        _FeatureExtractionStatusCardEntry(
            coordinate_system=coordinate_system,
            label_name=label_names_by_coordinate_system.get(coordinate_system),
            image_name=image_names_by_coordinate_system.get(coordinate_system),
            blocking_reason=blocking_reasons_by_coordinate_system.get(coordinate_system),
        )
        for coordinate_system in checked_coordinate_systems
    )


def build_feature_extraction_selection_status_card_spec(
    *,
    has_spatialdata: bool,
    checked_coordinate_systems: Sequence[str],
    entries: Sequence[_FeatureExtractionStatusCardEntry],
    table_blocker: _FeatureExtractionTableBlocker = None,
    table_tooltip_message: str | None = None,
) -> _FeatureExtractionStatusCardSpec:
    if not has_spatialdata:
        return _FeatureExtractionStatusCardSpec(
            title="No SpatialData Loaded",
            lines=(
                "Load a SpatialData object through the Harpy Viewer widget, reader, or `Interactive(sdata)`.",
                "This form updates automatically from the shared Harpy state.",
            ),
            kind="warning",
        )

    if not checked_coordinate_systems:
        return _FeatureExtractionStatusCardSpec(
            title="Choose Coordinate Systems",
            lines=("Choose one or more coordinate systems to start building extraction targets.",),
            kind="warning",
        )

    if any(not entry.is_valid for entry in entries):
        lines, tooltip_message = _build_entry_card_lines(entries)
        return _FeatureExtractionStatusCardSpec(
            title="Batch Incomplete",
            lines=lines,
            kind="warning",
            tooltip_message=tooltip_message,
        )

    if table_blocker is not None:
        return _build_table_blocker_card_spec(
            table_blocker=table_blocker,
            entries=entries,
            table_tooltip_message=table_tooltip_message,
        )

    lines, tooltip_message = _build_entry_card_lines(entries)
    return _FeatureExtractionStatusCardSpec(
        title="Batch Ready",
        lines=lines,
        kind="success",
        tooltip_message=tooltip_message,
    )


def build_feature_extraction_controller_feedback_card_spec(
    *,
    is_visible: bool,
    message: str,
    kind: StatusCardKind,
) -> _FeatureExtractionStatusCardSpec | None:
    if not is_visible or not message:
        return None

    title_by_kind = {
        "error": "Feature Extraction Error",
        "warning": "Feature Extraction Warning",
        "success": "Feature Extraction Ready",
        "info": "Feature Extraction",
    }
    return _FeatureExtractionStatusCardSpec(
        title=title_by_kind.get(kind, "Feature Extraction"),
        lines=(message.removeprefix("Feature extraction: ").strip(),),
        kind=kind,
    )


def _build_table_blocker_card_spec(
    *,
    table_blocker: _FeatureExtractionTableBlocker,
    entries: Sequence[_FeatureExtractionStatusCardEntry],
    table_tooltip_message: str | None,
) -> _FeatureExtractionStatusCardSpec:
    if table_blocker == "choose_table":
        title = "Batch Incomplete"
        lines = ("Choose a table that annotates all staged segmentations.",)
    elif table_blocker == "no_eligible":
        title = "Batch Incomplete"
        lines = ("No table annotates all currently staged segmentations.",)
    else:
        title = "Table Not Ready"
        lines = ("Selected table cannot currently be used for all staged segmentations.",)

    tooltip_lines = _build_entry_tooltip_lines(entries)
    if table_tooltip_message:
        tooltip_lines.append(table_tooltip_message)
    tooltip_message = "\n".join(tooltip_lines) if tooltip_lines else None
    return _FeatureExtractionStatusCardSpec(
        title=title,
        lines=lines,
        kind="warning",
        tooltip_message=tooltip_message,
    )


def _build_entry_card_lines(
    entries: Sequence[_FeatureExtractionStatusCardEntry],
) -> tuple[tuple[str, ...], str | None]:
    visible_lines: list[str] = []
    tooltip_lines: list[str] = []
    any_shortened = False
    for entry in entries:
        visible_line, shortened = _format_entry_line(entry, shorten=True)
        visible_lines.append(visible_line)
        tooltip_lines.append(_format_entry_line(entry, shorten=False)[0])
        any_shortened = any_shortened or shortened

    return tuple(visible_lines), "\n".join(tooltip_lines) if any_shortened else None


def _build_entry_tooltip_lines(entries: Sequence[_FeatureExtractionStatusCardEntry]) -> list[str]:
    return [_format_entry_line(entry, shorten=False)[0] for entry in entries]


def _format_entry_line(
    entry: _FeatureExtractionStatusCardEntry,
    *,
    shorten: bool,
) -> tuple[str, bool]:
    coordinate_system, coordinate_shortened = _format_identifier(entry.coordinate_system, shorten=shorten)
    if entry.blocking_reason is not None:
        if shorten or entry.label_name is None:
            return f"{coordinate_system}: {entry.blocking_reason}", coordinate_shortened

        label_name, label_shortened = _format_identifier(entry.label_name, shorten=shorten)
        return f"{coordinate_system}: {label_name} -> {entry.blocking_reason}", coordinate_shortened or label_shortened

    label_name, label_shortened = _format_identifier(entry.label_name or "unknown segmentation", shorten=shorten)
    if entry.image_name is None:
        image_text = "no image"
        image_shortened = False
    else:
        image_text, image_shortened = _format_identifier(entry.image_name, shorten=shorten)

    return (
        f"{coordinate_system}: {label_name} -> {image_text}",
        coordinate_shortened or label_shortened or image_shortened,
    )


def _format_identifier(name: str, *, shorten: bool) -> tuple[str, bool]:
    if not shorten:
        return name, False
    return format_feedback_identifier(name)
