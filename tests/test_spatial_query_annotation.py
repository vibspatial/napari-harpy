from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from spatialdata import SpatialData

from napari_harpy.core.class_palette import default_categorical_colors, default_labeled_class_color
from napari_harpy.core.spatial_query import (
    CanonicalCenterQueryResult,
    CanonicalCentersResult,
    SpatialAnnotationColumnChangedError,
    SpatialAnnotationQueryOutdatedError,
    apply_canonical_cache_update,
    apply_spatial_annotation,
    build_canonical_cache_update_payload,
    inspect_canonical_cache,
    prepare_spatial_annotation,
    summarize_spatial_annotation,
)


def test_existing_annotation_set_extends_category_and_valid_palette_stably(
    sdata_blobs: SpatialData,
) -> None:
    table = sdata_blobs.tables["table"]
    query_result = _query_result(sdata_blobs, matched_binding_positions=(0, 1, 2))
    values = [pd.NA] * table.n_obs
    values[1:4] = ["A", "B", "B"]
    table.obs["annotation"] = pd.Categorical(values, categories=["A", "B"], ordered=True)
    table.uns["annotation_colors"] = ["#112233", "#445566"]

    preparation = prepare_spatial_annotation(
        sdata_blobs,
        query_result=query_result,
        column_name="annotation",
        column_mode="existing",
    )
    summary = summarize_spatial_annotation(preparation, "  C  ")
    result = apply_spatial_annotation(sdata_blobs, preparation, summary)

    assert summary.annotation_value == "C"
    assert (summary.current_missing_count, summary.current_equal_count, summary.current_other_count) == (1, 0, 2)
    assert summary.changed_count == 3
    assert summary.overwrite_count == 2
    assert result.annotation_changed
    assert result.palette_changed
    assert table.obs["annotation"].iloc[:4].tolist() == ["C", "C", "C", "B"]
    assert list(table.obs["annotation"].cat.categories) == ["A", "B", "C"]
    assert table.obs["annotation"].cat.ordered
    assert table.uns["annotation_colors"] == ["#112233", "#445566", default_labeled_class_color(3)]


@pytest.mark.parametrize(
    ("stored_palette", "expected_palette_changed", "expected_palette"),
    [
        (None, True, default_categorical_colors(2)),
        (["#ff0000"], True, default_categorical_colors(2)),
        (["#112233", "#445566"], False, ["#112233", "#445566"]),
    ],
    ids=["missing", "invalid", "valid"],
)
def test_remove_annotation_preserves_categories_and_resolves_palette_only_on_effective_apply(
    sdata_blobs: SpatialData,
    stored_palette: list[str] | None,
    expected_palette_changed: bool,
    expected_palette: list[str],
) -> None:
    table = sdata_blobs.tables["table"]
    query_result = _query_result(sdata_blobs, matched_binding_positions=(0, 1, 2))
    values = [pd.NA] * table.n_obs
    values[:3] = ["A", pd.NA, "B"]
    table.obs["annotation"] = pd.Categorical(values, categories=["A", "B"])
    if stored_palette is not None:
        table.uns["annotation_colors"] = stored_palette

    preparation = prepare_spatial_annotation(
        sdata_blobs,
        query_result=query_result,
        column_name="annotation",
        column_mode="existing",
    )
    summary = summarize_spatial_annotation(preparation, None)
    result = apply_spatial_annotation(sdata_blobs, preparation, summary)

    assert (summary.current_missing_count, summary.current_equal_count, summary.current_other_count) == (1, 0, 2)
    assert summary.removal_count == 2
    assert table.obs["annotation"].iloc[:3].isna().all()
    assert list(table.obs["annotation"].cat.categories) == ["A", "B"]
    assert table.uns["annotation_colors"] == expected_palette
    assert result.annotation_changed
    assert result.palette_changed is expected_palette_changed


def test_new_annotation_column_is_categorical_and_missing_outside_matching_rows(
    sdata_blobs: SpatialData,
) -> None:
    table = sdata_blobs.tables["table"]
    query_result = _query_result(sdata_blobs, matched_binding_positions=(0, 2))

    preparation = prepare_spatial_annotation(
        sdata_blobs,
        query_result=query_result,
        column_name="  spatial_annotation  ",
        column_mode="new",
    )
    summary = summarize_spatial_annotation(preparation, "  tumor  ")
    result = apply_spatial_annotation(sdata_blobs, preparation, summary)

    annotation = table.obs["spatial_annotation"]
    assert preparation.column_name == "spatial_annotation"
    assert summary.annotation_value == "tumor"
    assert annotation.iloc[[0, 2]].tolist() == ["tumor", "tumor"]
    assert annotation.drop(annotation.index[[0, 2]]).isna().all()
    assert list(annotation.cat.categories) == ["tumor"]
    assert table.uns["spatial_annotation_colors"] == default_categorical_colors(1)
    assert result.annotation_changed and result.palette_changed


def test_noop_annotation_does_not_repair_a_missing_palette(
    sdata_blobs: SpatialData,
) -> None:
    table = sdata_blobs.tables["table"]
    query_result = _query_result(sdata_blobs, matched_binding_positions=(0, 1))
    table.obs["annotation"] = pd.Categorical(["A"] * table.n_obs, categories=["A"])
    before = table.obs["annotation"].copy(deep=True)

    preparation = prepare_spatial_annotation(
        sdata_blobs,
        query_result=query_result,
        column_name="annotation",
        column_mode="existing",
    )
    summary = summarize_spatial_annotation(preparation, "A")
    result = apply_spatial_annotation(sdata_blobs, preparation, summary)

    assert summary.changed_count == 0
    assert result.annotation_changed is False
    assert result.palette_changed is False
    assert table.obs["annotation"].equals(before)
    assert "annotation_colors" not in table.uns


def test_apply_rejects_changed_reviewed_values_without_mutation(
    sdata_blobs: SpatialData,
) -> None:
    table = sdata_blobs.tables["table"]
    query_result = _query_result(sdata_blobs, matched_binding_positions=(0, 1))
    table.obs["annotation"] = pd.Categorical(["A"] * table.n_obs, categories=["A", "B"])
    preparation = prepare_spatial_annotation(
        sdata_blobs,
        query_result=query_result,
        column_name="annotation",
        column_mode="existing",
    )
    summary = summarize_spatial_annotation(preparation, "B")
    table.obs.loc[table.obs.index[0], "annotation"] = "B"
    before = table.obs["annotation"].copy(deep=True)

    with pytest.raises(SpatialAnnotationColumnChangedError, match="values changed"):
        apply_spatial_annotation(sdata_blobs, preparation, summary)

    assert table.obs["annotation"].equals(before)
    assert "annotation_colors" not in table.uns


def test_apply_rejects_changed_canonical_centers_without_mutation(
    sdata_blobs: SpatialData,
) -> None:
    table = sdata_blobs.tables["table"]
    query_result = _query_result(sdata_blobs, matched_binding_positions=(0, 1))
    table.obs["annotation"] = pd.Categorical([pd.NA] * table.n_obs, categories=["A"])
    preparation = prepare_spatial_annotation(
        sdata_blobs,
        query_result=query_result,
        column_name="annotation",
        column_mode="existing",
    )
    summary = summarize_spatial_annotation(preparation, "A")
    table.obsm["spatial_canonical"][0, 1] += 1.0

    with pytest.raises(SpatialAnnotationQueryOutdatedError, match="centers changed"):
        apply_spatial_annotation(sdata_blobs, preparation, summary)

    assert table.obs["annotation"].isna().all()
    assert "annotation_colors" not in table.uns


def test_annotation_assignment_removes_new_column_when_palette_assignment_fails(
    sdata_blobs: SpatialData,
) -> None:
    table = sdata_blobs.tables["table"]
    query_result = _query_result(sdata_blobs, matched_binding_positions=(0, 1))
    preparation = prepare_spatial_annotation(
        sdata_blobs,
        query_result=query_result,
        column_name="annotation",
        column_mode="new",
    )
    summary = summarize_spatial_annotation(preparation, "B")

    class FailOncePaletteAssignment(dict):
        should_fail = True

        def __setitem__(self, key, value) -> None:
            if key == "annotation_colors" and self.should_fail:
                self.should_fail = False
                raise RuntimeError("injected palette assignment failure")
            super().__setitem__(key, value)

    table.uns = FailOncePaletteAssignment(table.uns)

    with pytest.raises(RuntimeError, match="injected palette assignment failure"):
        apply_spatial_annotation(sdata_blobs, preparation, summary)

    assert "annotation" not in table.obs.columns
    assert "annotation_colors" not in table.uns


def test_prepare_rejects_linkage_and_non_categorical_existing_columns(
    sdata_blobs: SpatialData,
) -> None:
    table = sdata_blobs.tables["table"]
    query_result = _query_result(sdata_blobs, matched_binding_positions=(0,))
    table.obs["text_annotation"] = "A"

    with pytest.raises(ValueError, match="linkage column"):
        prepare_spatial_annotation(
            sdata_blobs,
            query_result=query_result,
            column_name="instance_id",
            column_mode="existing",
        )
    with pytest.raises(ValueError, match="must be categorical"):
        prepare_spatial_annotation(
            sdata_blobs,
            query_result=query_result,
            column_name="text_annotation",
            column_mode="existing",
        )


def _query_result(
    sdata: SpatialData,
    *,
    matched_binding_positions: tuple[int, ...],
) -> CanonicalCenterQueryResult:
    report = inspect_canonical_cache(sdata, table_name="table", labels_name="blobs_labels")
    centers = np.zeros((report.binding.n_obs, 3), dtype=np.float64)
    centers[:, 1] = np.arange(report.binding.n_obs, dtype=np.float64)
    centers[:, 2] = np.arange(report.binding.n_obs, dtype=np.float64) + 0.5
    cache_update = apply_canonical_cache_update(
        sdata,
        build_canonical_cache_update_payload(
            binding=report.binding,
            centers=centers,
            source_signature=report.source_signature,
        ),
    )
    canonical_centers = CanonicalCentersResult(
        source_signature=report.source_signature,
        binding=report.binding,
        centers=centers,
        cache_update=cache_update,
    )
    matching_ids = np.sort(report.binding.instance_ids[list(matched_binding_positions)])
    return CanonicalCenterQueryResult(
        canonical_centers=canonical_centers,
        matched_instance_ids=matching_ids,
    )
