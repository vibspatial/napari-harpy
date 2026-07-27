from types import SimpleNamespace

import pytest

from napari_harpy.core.spatial_query import CanonicalCacheState, SpatialAnnotationSummary
from napari_harpy.widgets.spatial_query.status_card import (
    build_spatial_annotation_failure_status_card_spec,
    build_spatial_annotation_outcome_status_card_spec,
    build_spatial_query_controller_status_card_spec,
    build_spatial_query_status_card_spec,
)


@pytest.mark.parametrize(
    ("state", "expected_kind", "expected_cache_text"),
    [
        (CanonicalCacheState.VALID, "success", "will be reused"),
        (CanonicalCacheState.ABSENT, "info", "will be calculated"),
        (CanonicalCacheState.PARTIAL, "info", "will be calculated"),
        (CanonicalCacheState.STALE, "warning", "will be refreshed"),
        (CanonicalCacheState.INVALID, "info", "will be recalculated"),
    ],
)
def test_spatial_query_status_describes_ready_cache_states(
    state: CanonicalCacheState,
    expected_kind: str,
    expected_cache_text: str,
) -> None:
    report = SimpleNamespace(
        labels_name="cells",
        state=state,
        mismatches=(SimpleNamespace(detail="technical mismatch detail"),),
    )

    spec = build_spatial_query_status_card_spec(
        has_spatialdata=True,
        coordinate_system="global",
        saved_shapes_name="regions",
        has_unsaved_shapes_changes=False,
        labels_name="cells",
        table_name="table",
        cache_report=report,
        canonical_input_inspection_error=None,
        annotation_column_error=None,
        annotation_column_description='New column "spatial_annotation"',
        annotation_mutation_error=None,
        annotation_mutation_description='Action: Set annotation to "tumor".',
        layer_styling_error=None,
    )

    visible_text = "\n".join(spec.lines)
    assert spec.source == "configuration"
    assert spec.title == "Spatial Query Ready"
    assert spec.kind == expected_kind
    assert 'Shapes "regions" will query labels "cells".' in visible_text
    assert 'Target: New column "spatial_annotation".' in visible_text
    assert 'Action: Set annotation to "tumor".' in visible_text
    assert expected_cache_text in visible_text
    assert "technical mismatch detail" not in visible_text
    if state is CanonicalCacheState.INVALID:
        assert spec.tooltip_message == "technical mismatch detail"


@pytest.mark.parametrize(
    ("is_running", "kind", "expected_title"),
    [
        (True, "info", "Spatial Query Running"),
        (False, "success", "Spatial Query Complete"),
        (False, "error", "Spatial Query Failed"),
    ],
)
def test_spatial_query_controller_status_uses_the_unified_card(
    is_running: bool,
    kind: str,
    expected_title: str,
) -> None:
    spec = build_spatial_query_controller_status_card_spec(
        message="current operation state",
        kind=kind,
        is_running=is_running,
    )

    assert spec.source == "controller"
    assert spec.title == expected_title
    assert spec.lines == ("current operation state",)
    assert spec.kind == kind


@pytest.mark.parametrize(
    ("summary", "expected_title", "expected_text"),
    [
        (
            SpatialAnnotationSummary(
                annotation_value="tumor",
                matched_count=4,
                current_missing_count=1,
                current_equal_count=1,
                current_other_count=2,
            ),
            "Annotation Applied",
            'Applied "tumor" to 3 matched labeled objects.',
        ),
        (
            SpatialAnnotationSummary(
                annotation_value=None,
                matched_count=4,
                current_missing_count=4,
                current_equal_count=0,
                current_other_count=0,
            ),
            "No Annotation Changes",
            "Already missing: 4.",
        ),
    ],
)
def test_spatial_annotation_outcome_status_reports_exact_summary_counts(
    summary: SpatialAnnotationSummary,
    expected_title: str,
    expected_text: str,
) -> None:
    spec = build_spatial_annotation_outcome_status_card_spec(summary)

    assert spec.source == "annotation_outcome"
    assert spec.title == expected_title
    assert expected_text in "\n".join(spec.lines)


def test_spatial_annotation_failure_status_is_actionable() -> None:
    spec = build_spatial_annotation_failure_status_card_spec("The prepared values changed.")

    assert spec.source == "annotation_outcome"
    assert spec.title == "Annotation Failed"
    assert spec.kind == "error"
    assert "apply again" in spec.lines[1]
