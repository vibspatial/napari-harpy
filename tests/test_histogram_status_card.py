from __future__ import annotations

import pytest

from napari_harpy.widgets.histogram.status_card import (
    _HistogramStatusCardSpec,
    build_histogram_calculated_card_spec,
    build_histogram_controller_status_card_spec,
    build_histogram_error_card_spec,
    build_histogram_incomplete_card_spec,
    build_histogram_ready_card_spec,
    build_histogram_request_emitted_card_spec,
    build_histogram_running_card_spec,
    build_histogram_stale_request_card_spec,
)


def test_histogram_status_card_spec_rejects_invalid_kind() -> None:
    with pytest.raises(ValueError, match="Invalid status card kind"):
        _HistogramStatusCardSpec(
            title="Debug",
            lines=("Unexpected card kind.",),
            kind="debug",
        )


def test_build_histogram_incomplete_card_spec() -> None:
    spec = build_histogram_incomplete_card_spec("Choose an image.")

    assert spec.title == "Histogram Incomplete"
    assert spec.lines == ("Choose an image.",)
    assert spec.kind == "warning"
    assert spec.tooltip_message is None


def test_build_histogram_ready_card_spec_before_first_calculation() -> None:
    spec = build_histogram_ready_card_spec()

    assert spec.title == "Histogram Ready"
    assert spec.lines == ("Ready to calculate.",)
    assert spec.kind == "info"


def test_build_histogram_running_card_spec() -> None:
    spec = build_histogram_running_card_spec()

    assert spec.title == "Histogram Running"
    assert spec.lines == ("Calculating histogram.",)
    assert spec.kind == "info"


def test_build_histogram_calculated_card_spec() -> None:
    spec = build_histogram_calculated_card_spec()

    assert spec.title == "Histogram Calculated"
    assert spec.lines == ("Histogram calculated.",)
    assert spec.kind == "success"


def test_build_histogram_error_card_spec() -> None:
    spec = build_histogram_error_card_spec("Histogram calculation failed: boom")

    assert spec.title == "Histogram Error"
    assert spec.lines == ("Histogram calculation failed: boom",)
    assert spec.kind == "error"


def test_build_histogram_controller_status_card_spec_prefers_running_state() -> None:
    spec = build_histogram_controller_status_card_spec(
        message="Calculating histogram.",
        kind="success",
        is_running=True,
    )

    assert spec.title == "Histogram Running"
    assert spec.lines == ("Calculating histogram.",)
    assert spec.kind == "info"


def test_build_histogram_ready_card_spec_after_current_request_was_emitted() -> None:
    spec = build_histogram_request_emitted_card_spec()

    assert spec.title == "Histogram Ready"
    assert spec.lines == ("Calculation request emitted.",)
    assert spec.kind == "success"


def test_build_histogram_ready_card_spec_after_card_changed() -> None:
    spec = build_histogram_stale_request_card_spec()

    assert spec.title == "Histogram Ready"
    assert spec.lines == ("Target or settings changed. Calculate again.",)
    assert spec.kind == "info"
