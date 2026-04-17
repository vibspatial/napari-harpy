from __future__ import annotations

from spatialdata import SpatialData

from napari_harpy._feature_extraction import (
    FEATURE_EXTRACTION_IDLE_STATUS,
    FeatureExtractionController,
)


def test_feature_extraction_controller_starts_idle() -> None:
    controller = FeatureExtractionController()

    assert controller.status_message == FEATURE_EXTRACTION_IDLE_STATUS
    assert controller.status_kind == "warning"
    assert controller.is_running is False
    assert controller.can_calculate is False
    assert controller.table_binding_error is None


def test_feature_extraction_controller_bind_is_passive_and_ready_for_existing_table(
    sdata_blobs: SpatialData,
) -> None:
    controller = FeatureExtractionController()

    context_changed = controller.bind(
        sdata_blobs,
        "blobs_labels",
        None,
        "table",
        "global",
        ["area", "perimeter"],
        "feature_matrix_1",
    )

    assert context_changed is True
    assert controller.status_message == "Feature extraction: ready to calculate."
    assert controller.status_kind == "success"
    assert controller.is_running is False
    assert controller.can_calculate is True
    assert controller.table_binding_error is None


def test_feature_extraction_controller_blocks_when_no_table_is_selected(sdata_blobs: SpatialData) -> None:
    controller = FeatureExtractionController()

    controller.bind(
        sdata_blobs,
        "blobs_labels",
        None,
        None,
        "global",
        ["area"],
        "feature_matrix_1",
    )

    assert controller.can_calculate is False
    assert controller.status_kind == "warning"
    assert controller.status_message == "Feature extraction: choose an annotation table linked to the selected segmentation."


def test_feature_extraction_controller_requires_image_for_intensity_features(sdata_blobs: SpatialData) -> None:
    controller = FeatureExtractionController()

    controller.bind(
        sdata_blobs,
        "blobs_labels",
        None,
        "table",
        "global",
        ["mean", "area"],
        "feature_matrix_1",
    )

    assert controller.can_calculate is False
    assert controller.status_kind == "warning"
    assert controller.status_message == "Feature extraction: choose an image before calculating intensity features."


def test_feature_extraction_controller_reports_invalid_table_binding(sdata_blobs: SpatialData) -> None:
    controller = FeatureExtractionController()

    controller.bind(
        sdata_blobs,
        "blobs_multiscale_labels",
        None,
        "table",
        "global",
        ["area"],
        "feature_matrix_1",
    )

    assert controller.can_calculate is False
    assert controller.status_kind == "warning"
    assert controller.table_binding_error == "Table `table` does not annotate segmentation `blobs_multiscale_labels`."
    assert controller.status_message == (
        "Feature extraction: Table `table` does not annotate segmentation `blobs_multiscale_labels`."
    )


def test_feature_extraction_controller_bind_returns_false_for_unchanged_context(sdata_blobs: SpatialData) -> None:
    controller = FeatureExtractionController()

    first_changed = controller.bind(
        sdata_blobs,
        "blobs_labels",
        None,
        "table",
        "global",
        ["area"],
        "feature_matrix_1",
    )
    second_changed = controller.bind(
        sdata_blobs,
        "blobs_labels",
        None,
        "table",
        "global",
        ["area"],
        "feature_matrix_1",
    )

    assert first_changed is True
    assert second_changed is False
