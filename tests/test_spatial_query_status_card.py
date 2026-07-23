from types import SimpleNamespace

import pytest

from napari_harpy.core.spatial_query import CanonicalCacheState
from napari_harpy.widgets.spatial_query.status_card import build_spatial_query_status_card_spec


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
        target_error=None,
        target_description='New column "spatial_annotation"',
        layer_styling_error=None,
    )

    visible_text = "\n".join(spec.lines)
    assert spec.title == "Spatial Query Ready"
    assert spec.kind == expected_kind
    assert 'Shapes "regions" will query labels "cells".' in visible_text
    assert 'Target: New column "spatial_annotation".' in visible_text
    assert expected_cache_text in visible_text
    assert "technical mismatch detail" not in visible_text
    if state is CanonicalCacheState.INVALID:
        assert spec.tooltip_message == "technical mismatch detail"
