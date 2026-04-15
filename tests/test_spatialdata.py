from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from napari.layers import Image, Labels
from spatialdata import SpatialData
from spatialdata.datasets import blobs
from spatialdata.models import TableModel

import napari_harpy._spatialdata as spatialdata_module
from napari_harpy._spatialdata import (
    SpatialDataViewerBinding,
    get_annotating_table_names,
    get_spatialdata_image_options,
    get_table,
    get_table_metadata,
    get_table_obsm_keys,
    normalize_table_metadata,
    validate_table_binding,
)


def make_blobs_labels_layer(sdata: SpatialData, label_name: str = "blobs_labels") -> Labels:
    return Labels(
        sdata.labels[label_name],
        name=label_name,
        metadata={"sdata": sdata, "name": label_name},
    )


def make_blobs_image_layer(sdata: SpatialData, image_name: str = "blobs_image") -> Image:
    return Image(
        np.asarray(sdata.images[image_name]),
        name=image_name,
        metadata={"sdata": sdata, "name": image_name},
    )


def test_get_annotating_table_names_returns_tables_for_annotated_label(sdata_blobs: SpatialData) -> None:
    table_names = get_annotating_table_names(sdata_blobs, "blobs_labels")

    assert table_names == ["table"]


def test_get_annotating_table_names_returns_empty_list_for_unannotated_label(sdata_blobs: SpatialData) -> None:
    table_names = get_annotating_table_names(sdata_blobs, "blobs_multiscale_labels")

    assert table_names == []


def test_sdata_blobs_fixture_adds_dummy_feature_matrices(sdata_blobs: SpatialData) -> None:
    table = sdata_blobs["table"]

    assert "features_1" in table.obsm
    assert "features_2" in table.obsm
    assert table.obsm["features_1"].shape == (table.n_obs, 4)
    assert table.obsm["features_2"].shape == (table.n_obs, 2)


def test_get_table_obsm_keys_returns_sorted_feature_matrix_keys(sdata_blobs: SpatialData) -> None:
    obsm_keys = get_table_obsm_keys(sdata_blobs, "table")

    assert obsm_keys == ["features_1", "features_2"]


def test_get_table_metadata_resolves_table_metadata(sdata_blobs: SpatialData) -> None:
    metadata = get_table_metadata(sdata_blobs, "table")

    assert metadata.table_name == "table"
    assert metadata.region_key == "region"
    assert metadata.instance_key == "instance_id"
    assert metadata.regions == ("blobs_labels",)
    assert metadata.annotates("blobs_labels")
    assert not metadata.annotates("blobs_multiscale_labels")


def test_get_table_metadata_returns_table_linkage(sdata_blobs: SpatialData) -> None:
    metadata = get_table_metadata(sdata_blobs, "table")

    assert metadata.region_key == "region"
    assert metadata.instance_key == "instance_id"


def test_validate_table_binding_rejects_duplicate_instance_ids_within_selected_region(sdata_blobs: SpatialData) -> None:
    table = sdata_blobs["table"]
    first_index, second_index = table.obs.index[:2]
    table.obs.loc[second_index, "instance_id"] = table.obs.loc[first_index, "instance_id"]

    with pytest.raises(ValueError, match="contains duplicate values within that region"):
        validate_table_binding(sdata_blobs, "blobs_labels", "table")


def test_normalize_table_metadata_normalizes_numpy_array_region_attrs_in_place(sdata_blobs: SpatialData) -> None:
    table = sdata_blobs["table"]
    table.uns[TableModel.ATTRS_KEY] = {
        **table.uns[TableModel.ATTRS_KEY],
        TableModel.REGION_KEY: np.array(["blobs_labels"]),
    }

    normalize_table_metadata(table)
    validated = get_table(sdata_blobs, "table")

    assert validated is table
    assert table.uns[TableModel.ATTRS_KEY][TableModel.REGION_KEY] == ["blobs_labels"]


def test_get_spatialdata_image_options_return_compatible_images_for_selected_label(sdata_blobs: SpatialData) -> None:
    label_layer = make_blobs_labels_layer(sdata_blobs)
    image_layer = make_blobs_image_layer(sdata_blobs)
    viewer = SimpleNamespace(layers=[label_layer, image_layer])

    options = get_spatialdata_image_options(
        viewer,
        sdata=sdata_blobs,
        label_name="blobs_labels",
    )

    assert [option.image_name for option in options] == [
        "blobs_image",
        "blobs_multiscale_image",
    ]
    assert [option.display_name for option in options] == [
        "blobs_image",
        "blobs_multiscale_image",
    ]
    assert [option.coordinate_systems for option in options] == [
        ("global",),
        ("global",),
    ]
    assert all(option.sdata is sdata_blobs for option in options)


def test_get_spatialdata_image_options_do_not_require_loaded_image_layers(sdata_blobs: SpatialData) -> None:
    label_layer = make_blobs_labels_layer(sdata_blobs)
    viewer = SimpleNamespace(layers=[label_layer])

    options = get_spatialdata_image_options(
        viewer,
        sdata=sdata_blobs,
        label_name="blobs_labels",
    )

    assert [option.image_name for option in options] == [
        "blobs_image",
        "blobs_multiscale_image",
    ]
    assert [option.display_name for option in options] == [
        "blobs_image",
        "blobs_multiscale_image",
    ]
    assert [option.coordinate_systems for option in options] == [
        ("global",),
        ("global",),
    ]
    assert all(option.sdata is sdata_blobs for option in options)


def test_get_spatialdata_image_options_include_dataset_names_when_multiple_datasets_are_present() -> None:
    first_sdata = blobs()
    second_sdata = blobs()
    viewer = SimpleNamespace(layers=[make_blobs_labels_layer(first_sdata), make_blobs_labels_layer(second_sdata)])

    options = get_spatialdata_image_options(
        viewer,
        sdata=first_sdata,
        label_name="blobs_labels",
    )

    assert [option.display_name for option in options] == [
        "blobs_image (SpatialData 1)",
        "blobs_multiscale_image (SpatialData 1)",
    ]


def test_get_spatialdata_image_options_filter_to_selected_dataset_and_shared_coordinate_systems(
    monkeypatch,
    sdata_blobs: SpatialData,
) -> None:
    other_sdata = blobs()
    viewer = SimpleNamespace(layers=[make_blobs_labels_layer(sdata_blobs), make_blobs_labels_layer(other_sdata)])

    transformation_by_id = {
        id(sdata_blobs.labels["blobs_labels"]): {"global": object(), "aligned": object()},
        id(sdata_blobs.images["blobs_image"]): {"aligned": object(), "global": object()},
        id(sdata_blobs.images["blobs_multiscale_image"]): {"other": object()},
    }

    def _fake_get_transformation(element, get_all: bool = False):
        del get_all
        return transformation_by_id[id(element)]

    monkeypatch.setattr(spatialdata_module, "get_transformation", _fake_get_transformation)

    options = get_spatialdata_image_options(
        viewer,
        sdata=sdata_blobs,
        label_name="blobs_labels",
    )

    assert [option.image_name for option in options] == ["blobs_image"]
    assert options[0].sdata is sdata_blobs
    assert options[0].coordinate_systems == ("aligned", "global")


def test_spatialdata_viewer_binding_builds_layer_metadata_adata_from_selected_table(sdata_blobs: SpatialData) -> None:
    viewer_binding = SpatialDataViewerBinding()

    metadata_adata = viewer_binding.build_layer_metadata_adata(sdata_blobs, "blobs_labels", "table")

    assert metadata_adata is not None
    assert metadata_adata is not sdata_blobs["table"]
    assert metadata_adata.is_view
    assert list(metadata_adata.obs.index) == list(sdata_blobs["table"].obs.index)
    assert "features_1" in metadata_adata.obsm
    assert "features_2" in metadata_adata.obsm


def test_spatialdata_viewer_binding_prefers_region_view_when_layer_indices_are_aligned(
    monkeypatch,
    sdata_blobs: SpatialData,
) -> None:
    layer = make_blobs_labels_layer(sdata_blobs)
    layer.metadata["indices"] = list(reversed(sdata_blobs["table"].obs["instance_id"].to_list()))
    viewer_binding = SpatialDataViewerBinding(SimpleNamespace(layers=[layer]))

    # check that we do not call join_spatialelement_table
    def _unexpected_join(*args, **kwargs):
        raise AssertionError("join_spatialelement_table should not be called for aligned layer indices.")

    monkeypatch.setattr(spatialdata_module, "join_spatialelement_table", _unexpected_join)

    metadata_adata = viewer_binding.build_layer_metadata_adata(sdata_blobs, "blobs_labels", "table")

    assert metadata_adata is not None
    assert metadata_adata.is_view
    assert set(metadata_adata.obs["instance_id"]) == set(layer.metadata["indices"])


def test_spatialdata_viewer_binding_falls_back_to_join_when_layer_indices_are_misaligned(
    monkeypatch,
    sdata_blobs: SpatialData,
) -> None:
    layer = make_blobs_labels_layer(sdata_blobs)
    layer.metadata["indices"] = sdata_blobs["table"].obs["instance_id"].to_list()[:-1]
    viewer_binding = SpatialDataViewerBinding(SimpleNamespace(layers=[layer]))
    sentinel = sdata_blobs["table"].copy()
    sentinel.obs["from_join"] = range(sentinel.n_obs)
    join_called = False

    def _fake_join(*args, **kwargs):
        nonlocal join_called
        join_called = True
        return None, sentinel

    monkeypatch.setattr(spatialdata_module, "join_spatialelement_table", _fake_join)

    metadata_adata = viewer_binding.build_layer_metadata_adata(sdata_blobs, "blobs_labels", "table")

    assert join_called is True
    assert metadata_adata is sentinel
    assert "from_join" in metadata_adata.obs


def test_spatialdata_viewer_binding_refreshes_only_table_derived_layer_metadata(sdata_blobs: SpatialData) -> None:
    layer = make_blobs_labels_layer(sdata_blobs)
    layer.metadata["adata"] = "stale"
    layer.metadata["region_key"] = "old_region"
    layer.metadata["instance_key"] = "old_instance"
    layer.metadata["table_names"] = ["old_table"]
    layer.metadata["indices"] = [1, 2, 3]
    layer.metadata["custom_flag"] = "keep-me"
    viewer = SimpleNamespace(layers=[layer])
    viewer_binding = SpatialDataViewerBinding(viewer)

    sdata_blobs["table"].obs["reloaded_obs"] = range(sdata_blobs["table"].n_obs)
    sdata_blobs["table"].obsm["reloaded_features"] = sdata_blobs["table"].obsm["features_1"][:, :1]

    refreshed = viewer_binding.refresh_layer_table_metadata(sdata_blobs, "blobs_labels", "table")

    assert refreshed is True
    assert layer.metadata["adata"] is not None
    assert "reloaded_obs" in layer.metadata["adata"].obs
    assert "reloaded_features" in layer.metadata["adata"].obsm
    assert layer.metadata["region_key"] == "region"
    assert layer.metadata["instance_key"] == "instance_id"
    assert layer.metadata["table_names"] == ["table"]
    assert layer.metadata["indices"] == [1, 2, 3]
    assert layer.metadata["custom_flag"] == "keep-me"


def test_spatialdata_viewer_binding_refresh_layer_table_metadata_returns_false_without_loaded_layer(
    sdata_blobs: SpatialData,
) -> None:
    viewer_binding = SpatialDataViewerBinding(SimpleNamespace(layers=[]))

    refreshed = viewer_binding.refresh_layer_table_metadata(sdata_blobs, "blobs_labels", "table")

    assert refreshed is False


def test_spatialdata_viewer_binding_get_image_layer_returns_loaded_image_layer(sdata_blobs: SpatialData) -> None:
    image_layer = make_blobs_image_layer(sdata_blobs)
    viewer_binding = SpatialDataViewerBinding(SimpleNamespace(layers=[image_layer]))

    loaded_layer = viewer_binding.get_image_layer(sdata_blobs, "blobs_image")

    assert loaded_layer is image_layer
    assert viewer_binding.get_image_layer(sdata_blobs, "blobs_multiscale_image") is None


def test_spatialdata_viewer_binding_get_image_layer_rejects_non_image_layers(sdata_blobs: SpatialData) -> None:
    fake_layer = SimpleNamespace(
        metadata={"sdata": sdata_blobs, "name": "blobs_image"},
    )
    viewer_binding = SpatialDataViewerBinding(SimpleNamespace(layers=[fake_layer]))

    loaded_layer = viewer_binding.get_image_layer(sdata_blobs, "blobs_image")

    assert loaded_layer is None


def test_spatialdata_viewer_binding_get_labels_layer_rejects_non_labels_layers(sdata_blobs: SpatialData) -> None:
    fake_layer = SimpleNamespace(
        metadata={"sdata": sdata_blobs, "name": "blobs_labels"},
        selected_label=1,
        events=SimpleNamespace(selected_label=object()),
    )
    viewer_binding = SpatialDataViewerBinding(SimpleNamespace(layers=[fake_layer]))

    loaded_layer = viewer_binding.get_labels_layer(sdata_blobs, "blobs_labels")

    assert loaded_layer is None
