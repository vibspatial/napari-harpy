from __future__ import annotations

import dask.array as da
import numpy as np
import pandas as pd
import pytest
from dask.callbacks import Callback
from spatialdata import SpatialData

import napari_harpy.core.spatial_query.centroids as centroids_module
from napari_harpy.core.spatial_query import (
    CANONICAL_OBSM_KEY,
    SPATIAL_COORDINATES_KEY,
    CanonicalCacheUpdateAction,
    calculate_canonical_centers,
    ensure_canonical_centers,
    inspect_canonical_cache,
)


def test_calculate_canonical_centers_uses_requested_ids_and_aggregator_output(
    sdata_blobs: SpatialData,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = inspect_canonical_cache(sdata_blobs, table_name="table", labels_name="blobs_labels")
    captured: dict[str, object] = {}

    class FakeRasterAggregator:
        def __init__(self, mask_dask_array, image_dask_array, instance_key, run_on_gpu) -> None:
            captured.update(
                mask=mask_dask_array,
                image=image_dask_array,
                instance_key=instance_key,
                run_on_gpu=run_on_gpu,
            )
            self.instance_key = instance_key

        def center_of_mass(self, index):
            captured["index"] = index
            instance_ids = np.asarray(index)
            return pd.DataFrame(
                {
                    0: np.zeros(len(instance_ids)),
                    1: instance_ids.astype(np.float64) + 0.25,
                    2: instance_ids.astype(np.float64) + 0.75,
                    self.instance_key: instance_ids,
                }
            )

    monkeypatch.setattr(centroids_module, "RasterAggregator", FakeRasterAggregator)

    payload = calculate_canonical_centers(sdata_blobs, report)

    assert report.table_name == "table"
    assert report.labels_name == "blobs_labels"
    assert payload.table_name == "table"
    assert payload.labels_name == "blobs_labels"
    mask = captured["mask"]
    assert isinstance(mask, da.Array)
    assert mask.shape == (1, 512, 512)
    assert mask.chunks[0] == (1,)
    assert captured["image"] is None
    assert captured["instance_key"] == "instance_id"
    assert captured["run_on_gpu"] is False
    np.testing.assert_array_equal(captured["index"], report.binding.instance_ids)
    np.testing.assert_array_equal(payload.centers[:, 0], 0.0)
    np.testing.assert_allclose(payload.centers[:, 1], report.binding.instance_ids + 0.25)
    np.testing.assert_allclose(payload.centers[:, 2], report.binding.instance_ids + 0.75)


def test_calculate_canonical_centers_matches_multichunk_reference(
    backed_sdata_blobs: SpatialData,
) -> None:
    labels = backed_sdata_blobs.labels["blobs_labels"]
    labels.data = labels.data.rechunk((128, 128))
    mask = np.asarray(labels.data.compute())
    report = inspect_canonical_cache(backed_sdata_blobs, table_name="table", labels_name="blobs_labels")

    payload = calculate_canonical_centers(backed_sdata_blobs, report)

    expected = np.zeros((report.binding.n_obs, 3), dtype=np.float64)
    for row, instance_id in enumerate(report.binding.instance_ids):
        expected[row, 1:] = np.argwhere(mask == instance_id).mean(axis=0)
    np.testing.assert_allclose(payload.centers, expected)


def test_ensure_canonical_centers_creates_then_reuses_without_labels_tasks(
    backed_sdata_blobs: SpatialData,
) -> None:
    created = ensure_canonical_centers(
        backed_sdata_blobs,
        table_name="table",
        labels_name="blobs_labels",
    )
    executed_tasks: list[object] = []

    with Callback(pretask=lambda key, _dask, _state: executed_tasks.append(key)):
        reused = ensure_canonical_centers(
            backed_sdata_blobs,
            table_name="table",
            labels_name="blobs_labels",
        )

    assert created.cache_update is not None
    assert created.cache_update.action is CanonicalCacheUpdateAction.CREATE
    assert not created.reused
    assert reused.reused
    assert reused.cache_update is None
    assert reused.table_name == "table"
    assert not reused.centers.flags.writeable
    assert executed_tasks == []
    np.testing.assert_array_equal(reused.binding.instance_ids, created.binding.instance_ids)
    np.testing.assert_array_equal(reused.centers, created.centers)


def test_ensure_canonical_centers_refreshes_stale_and_forced_valid_cache(sdata_blobs: SpatialData) -> None:
    ensure_canonical_centers(sdata_blobs, table_name="table", labels_name="blobs_labels")
    table = sdata_blobs.tables["table"]
    table.obsm[CANONICAL_OBSM_KEY][0, 1] = np.nan

    stale_refresh = ensure_canonical_centers(
        sdata_blobs,
        table_name="table",
        labels_name="blobs_labels",
    )

    assert stale_refresh.cache_update is not None
    assert stale_refresh.cache_update.action is CanonicalCacheUpdateAction.REFRESH

    refreshed = ensure_canonical_centers(
        sdata_blobs,
        table_name="table",
        labels_name="blobs_labels",
        force_recalculation=True,
    )

    assert refreshed.cache_update is not None
    assert refreshed.cache_update.action is CanonicalCacheUpdateAction.REFRESH


def test_ensure_canonical_centers_rejects_instance_missing_from_raster_without_mutation(
    sdata_blobs: SpatialData,
) -> None:
    table = sdata_blobs.tables["table"]
    table.obs.iloc[0, table.obs.columns.get_loc("instance_id")] = np.iinfo(np.int16).max

    with pytest.raises(ValueError, match="no finite center for 1 requested instance ID"):
        ensure_canonical_centers(sdata_blobs, table_name="table", labels_name="blobs_labels")

    assert CANONICAL_OBSM_KEY not in table.obsm
    assert SPATIAL_COORDINATES_KEY not in table.uns


def test_calculate_canonical_centers_rejects_ids_outside_labels_dtype_before_aggregation(
    sdata_blobs: SpatialData,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table = sdata_blobs.tables["table"]
    table.obs.iloc[0, table.obs.columns.get_loc("instance_id")] = int(np.iinfo(np.int16).max) + 1
    report = inspect_canonical_cache(sdata_blobs, table_name="table", labels_name="blobs_labels")

    class UnexpectedRasterAggregator:
        def __init__(self, *args, **kwargs) -> None:
            raise AssertionError("RasterAggregator must not start before dtype validation.")

    monkeypatch.setattr(centroids_module, "RasterAggregator", UnexpectedRasterAggregator)

    with pytest.raises(ValueError, match="cannot be represented by labels dtype `int16`"):
        calculate_canonical_centers(sdata_blobs, report)
