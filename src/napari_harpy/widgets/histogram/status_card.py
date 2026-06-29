from __future__ import annotations

from dataclasses import dataclass

from napari_harpy.widgets.shared_styles import StatusCardKind, validate_status_card_kind


@dataclass(frozen=True)
class _HistogramStatusCardSpec:
    title: str
    lines: tuple[str, ...]
    kind: StatusCardKind
    tooltip_message: str | None = None

    def __post_init__(self) -> None:
        validate_status_card_kind(self.kind)


def build_histogram_incomplete_card_spec(message: str) -> _HistogramStatusCardSpec:
    return _HistogramStatusCardSpec(
        title="Histogram Incomplete",
        lines=(message,),
        kind="warning",
    )


def build_histogram_ready_card_spec(message: str = "Ready to calculate.") -> _HistogramStatusCardSpec:
    return _HistogramStatusCardSpec(
        title="Histogram Ready",
        lines=(message,),
        kind="info",
    )


def build_histogram_running_card_spec(message: str = "Calculating histogram.") -> _HistogramStatusCardSpec:
    return _HistogramStatusCardSpec(
        title="Histogram Running",
        lines=(message,),
        kind="info",
    )


def build_histogram_calculated_card_spec(message: str = "Histogram calculated.") -> _HistogramStatusCardSpec:
    return _HistogramStatusCardSpec(
        title="Histogram Calculated",
        lines=(message,),
        kind="success",
    )


def build_histogram_error_card_spec(message: str) -> _HistogramStatusCardSpec:
    return _HistogramStatusCardSpec(
        title="Histogram Error",
        lines=(message,),
        kind="error",
    )


def build_histogram_controller_status_card_spec(
    *,
    message: str,
    kind: StatusCardKind,
    is_running: bool = False,
) -> _HistogramStatusCardSpec:
    if is_running:
        return build_histogram_running_card_spec(message)
    if kind == "warning":
        return build_histogram_incomplete_card_spec(message)
    if kind == "error":
        return build_histogram_error_card_spec(message)
    if kind == "success":
        return build_histogram_calculated_card_spec(message)
    return build_histogram_ready_card_spec(message)


def build_histogram_request_emitted_card_spec() -> _HistogramStatusCardSpec:
    return _HistogramStatusCardSpec(
        title="Histogram Ready",
        lines=("Calculation request emitted.",),
        kind="success",
    )


def build_histogram_stale_request_card_spec() -> _HistogramStatusCardSpec:
    return _HistogramStatusCardSpec(
        title="Histogram Ready",
        lines=("Target or settings changed. Calculate again.",),
        kind="info",
    )
