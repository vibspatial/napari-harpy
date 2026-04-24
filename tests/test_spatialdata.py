from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from napari.layers import Image, Labels
from spatialdata import SpatialData
from spatialdata.models import TableModel
from xarray import DataArray

import napari_harpy._spatialdata as spatialdata_module
from napari_harpy._spatialdata import (
    build_layer_metadata_adata,
    get_annotating_table_names,
    get_coordinate_system_names_from_sdata,
    get_image_channel_names_from_sdata,
    get_spatialdata_image_options_for_coordinate_system_from_sdata,
    get_spatialdata_label_options_for_coordinate_system_from_sdata,
    get_spatialdata_label_options_from_sdata,
    get_table,
    get_table_color_source_options,
    get_table_metadata,
    get_table_obs_color_source_options,
    get_table_obsm_keys,
    get_table_x_var_color_source_options,
    normalize_table_metadata,
    refresh_layer_table_metadata,
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


def test_get_table_obs_color_source_options_classifies_supported_columns(sdata_blobs: SpatialData) -> None:
    table = sdata_blobs["table"]
    n_obs = table.n_obs
    repeated_values = ["a", "b"] * (n_obs // 2) + (["a"] if n_obs % 2 else [])

    table.obs["cat_obs"] = pd.Categorical(repeated_values)
    table.obs["bool_obs"] = pd.Series([index % 2 == 0 for index in range(n_obs)], index=table.obs.index)
    table.obs["binary_int_obs"] = pd.Series([index % 2 for index in range(n_obs)], index=table.obs.index, dtype="int64")
    table.obs["int_obs"] = pd.Series(np.arange(n_obs), index=table.obs.index, dtype="int64")
    table.obs["float_obs"] = pd.Series(np.linspace(0.0, 1.0, n_obs), index=table.obs.index, dtype="float64")
    table.obs["string_obs"] = pd.Series(repeated_values, index=table.obs.index, dtype="object")
    table.obs["string_id_obs"] = pd.Series(
        [f"cell-{index:04d}" for index in range(n_obs)],
        index=table.obs.index,
        dtype="object",
    )
    table.obs["cat_id_obs"] = pd.Categorical([f"cell-{index:04d}" for index in range(n_obs)])
    table.obs["datetime_obs"] = pd.date_range("2024-01-01", periods=n_obs)
    table.obs["object_obs"] = pd.Series([{"index": index} for index in range(n_obs)], index=table.obs.index, dtype="object")

    options = get_table_obs_color_source_options(sdata_blobs, "table")

    option_by_key = {option.value_key: option for option in options}

    assert "region" not in option_by_key
    assert option_by_key["instance_id"].value_kind == "instance"
    assert option_by_key["cat_obs"].value_kind == "categorical"
    assert option_by_key["bool_obs"].value_kind == "categorical"
    assert option_by_key["binary_int_obs"].value_kind == "categorical"
    assert option_by_key["int_obs"].value_kind == "continuous"
    assert option_by_key["float_obs"].value_kind == "continuous"
    assert option_by_key["string_obs"].value_kind == "categorical"
    assert option_by_key["string_id_obs"].value_kind == "categorical"
    assert option_by_key["cat_id_obs"].value_kind == "categorical"
    assert "datetime_obs" not in option_by_key
    assert "object_obs" not in option_by_key
    assert option_by_key["cat_obs"].source_kind == "obs_column"
    assert option_by_key["instance_id"].source_kind == "obs_column"
    assert option_by_key["cat_obs"].display_name == "cat_obs"
    assert option_by_key["instance_id"].display_name == "instance_id"
    assert option_by_key["cat_obs"].identity == ("table", "obs_column", "cat_obs")
    assert option_by_key["instance_id"].identity == ("table", "obs_column", "instance_id")


def test_get_table_x_var_color_source_options_exposes_var_names_as_continuous_sources(
    sdata_blobs: SpatialData,
) -> None:
    options = get_table_x_var_color_source_options(sdata_blobs, "table")

    assert [option.value_key for option in options] == ["channel_0_sum", "channel_1_sum", "channel_2_sum"]
    assert all(option.source_kind == "x_var" for option in options)
    assert all(option.value_kind == "continuous" for option in options)
    assert [option.display_name for option in options] == ["channel_0_sum", "channel_1_sum", "channel_2_sum"]


def test_get_table_color_source_options_combines_obs_and_x_var_sources(sdata_blobs: SpatialData) -> None:
    table = sdata_blobs["table"]
    table.obs["cat_obs"] = pd.Categorical(["a"] * table.n_obs)

    options = get_table_color_source_options(sdata_blobs, "table")

    assert [option.identity for option in options] == [
        ("table", "obs_column", "instance_id"),
        ("table", "obs_column", "cat_obs"),
        ("table", "x_var", "channel_0_sum"),
        ("table", "x_var", "channel_1_sum"),
        ("table", "x_var", "channel_2_sum"),
    ]


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

def test_get_spatialdata_label_options_from_sdata_returns_all_labels(sdata_blobs: SpatialData) -> None:
    options = get_spatialdata_label_options_from_sdata(sdata_blobs)

    assert [option.label_name for option in options] == [
        "blobs_labels",
        "blobs_multiscale_labels",
    ]
    assert [option.display_name for option in options] == [
        "blobs_labels",
        "blobs_multiscale_labels",
    ]
    assert [option.coordinate_systems for option in options] == [
        ("global",),
        ("global",),
    ]
    assert all(option.sdata is sdata_blobs for option in options)


def test_get_coordinate_system_names_from_sdata_returns_sorted_union(
    monkeypatch,
    sdata_blobs: SpatialData,
) -> None:
    transformation_by_id = {
        id(sdata_blobs.labels["blobs_labels"]): {"global": object(), "aligned": object()},
        id(sdata_blobs.labels["blobs_multiscale_labels"]): {"global": object()},
        id(sdata_blobs.images["blobs_image"]): {"global": object()},
        id(sdata_blobs.images["blobs_multiscale_image"]): {"local": object()},
    }

    def _fake_get_transformation(element, get_all: bool = False):
        del get_all
        return transformation_by_id[id(element)]

    monkeypatch.setattr(spatialdata_module, "get_transformation", _fake_get_transformation)

    assert get_coordinate_system_names_from_sdata(sdata_blobs) == ["aligned", "global", "local"]


def test_get_image_channel_names_from_sdata_returns_channel_axis_names(sdata_blobs: SpatialData) -> None:
    assert get_image_channel_names_from_sdata(sdata_blobs, "blobs_image") == ["0", "1", "2"]


def test_get_image_channel_names_from_sdata_rejects_duplicate_channel_names(monkeypatch) -> None:
    fake_sdata = SimpleNamespace(
        images={
            "image_with_duplicates": DataArray(
                np.zeros((3, 2, 2)),
                dims=("c", "y", "x"),
                coords={"c": ["dup", "dup", "other"]},
            )
        }
    )
    monkeypatch.setattr(spatialdata_module, "_get_image_names", lambda sdata: ["image_with_duplicates"])

    with pytest.raises(ValueError, match="duplicate channel names"):
        get_image_channel_names_from_sdata(fake_sdata, "image_with_duplicates")


def test_get_spatialdata_label_options_for_coordinate_system_from_sdata_filters_labels(
    monkeypatch,
    sdata_blobs: SpatialData,
) -> None:
    transformation_by_id = {
        id(sdata_blobs.labels["blobs_labels"]): {"global": object(), "aligned": object()},
        id(sdata_blobs.labels["blobs_multiscale_labels"]): {"global": object()},
        id(sdata_blobs.images["blobs_image"]): {"global": object()},
        id(sdata_blobs.images["blobs_multiscale_image"]): {"global": object()},
    }

    def _fake_get_transformation(element, get_all: bool = False):
        del get_all
        return transformation_by_id[id(element)]

    monkeypatch.setattr(spatialdata_module, "get_transformation", _fake_get_transformation)

    options = get_spatialdata_label_options_for_coordinate_system_from_sdata(
        sdata=sdata_blobs,
        coordinate_system="aligned",
    )

    assert [option.label_name for option in options] == ["blobs_labels"]
    assert options[0].coordinate_systems == ("aligned", "global")


def test_get_spatialdata_image_options_for_coordinate_system_from_sdata_filters_images(
    monkeypatch,
    sdata_blobs: SpatialData,
) -> None:
    transformation_by_id = {
        id(sdata_blobs.labels["blobs_labels"]): {"global": object()},
        id(sdata_blobs.labels["blobs_multiscale_labels"]): {"global": object()},
        id(sdata_blobs.images["blobs_image"]): {"global": object(), "aligned": object()},
        id(sdata_blobs.images["blobs_multiscale_image"]): {"local": object()},
    }

    def _fake_get_transformation(element, get_all: bool = False):
        del get_all
        return transformation_by_id[id(element)]

    monkeypatch.setattr(spatialdata_module, "get_transformation", _fake_get_transformation)

    options = get_spatialdata_image_options_for_coordinate_system_from_sdata(
        sdata=sdata_blobs,
        coordinate_system="aligned",
    )

    assert [option.image_name for option in options] == ["blobs_image"]
    assert options[0].coordinate_systems == ("aligned", "global")

def test_build_layer_metadata_adata_builds_from_selected_table(sdata_blobs: SpatialData) -> None:
    metadata_adata = build_layer_metadata_adata(None, sdata_blobs, "blobs_labels", "table")

    assert metadata_adata is not None
    assert metadata_adata is not sdata_blobs["table"]
    assert metadata_adata.is_view
    assert list(metadata_adata.obs.index) == list(sdata_blobs["table"].obs.index)
    assert "features_1" in metadata_adata.obsm
    assert "features_2" in metadata_adata.obsm


def test_build_layer_metadata_adata_prefers_region_view_when_layer_indices_are_aligned(
    monkeypatch,
    sdata_blobs: SpatialData,
) -> None:
    layer = make_blobs_labels_layer(sdata_blobs)
    layer.metadata["indices"] = list(reversed(sdata_blobs["table"].obs["instance_id"].to_list()))

    # check that we do not call join_spatialelement_table
    def _unexpected_join(*args, **kwargs):
        raise AssertionError("join_spatialelement_table should not be called for aligned layer indices.")

    monkeypatch.setattr(spatialdata_module, "join_spatialelement_table", _unexpected_join)

    metadata_adata = build_layer_metadata_adata(SimpleNamespace(layers=[layer]), sdata_blobs, "blobs_labels", "table")

    assert metadata_adata is not None
    assert metadata_adata.is_view
    assert set(metadata_adata.obs["instance_id"]) == set(layer.metadata["indices"])


def test_build_layer_metadata_adata_falls_back_to_join_when_layer_indices_are_misaligned(
    monkeypatch,
    sdata_blobs: SpatialData,
) -> None:
    layer = make_blobs_labels_layer(sdata_blobs)
    layer.metadata["indices"] = sdata_blobs["table"].obs["instance_id"].to_list()[:-1]
    sentinel = sdata_blobs["table"].copy()
    sentinel.obs["from_join"] = range(sentinel.n_obs)
    join_called = False

    def _fake_join(*args, **kwargs):
        nonlocal join_called
        join_called = True
        return None, sentinel

    monkeypatch.setattr(spatialdata_module, "join_spatialelement_table", _fake_join)

    metadata_adata = build_layer_metadata_adata(SimpleNamespace(layers=[layer]), sdata_blobs, "blobs_labels", "table")

    assert join_called is True
    assert metadata_adata is sentinel
    assert "from_join" in metadata_adata.obs


def test_refresh_layer_table_metadata_refreshes_only_table_derived_layer_metadata(sdata_blobs: SpatialData) -> None:
    layer = make_blobs_labels_layer(sdata_blobs)
    layer.metadata["adata"] = "stale"
    layer.metadata["region_key"] = "old_region"
    layer.metadata["instance_key"] = "old_instance"
    layer.metadata["table_names"] = ["old_table"]
    layer.metadata["indices"] = [1, 2, 3]
    layer.metadata["custom_flag"] = "keep-me"
    viewer = SimpleNamespace(layers=[layer])

    sdata_blobs["table"].obs["reloaded_obs"] = range(sdata_blobs["table"].n_obs)
    sdata_blobs["table"].obsm["reloaded_features"] = sdata_blobs["table"].obsm["features_1"][:, :1]

    refreshed = refresh_layer_table_metadata(viewer, sdata_blobs, "blobs_labels", "table")

    assert refreshed is True
    assert layer.metadata["adata"] is not None
    assert "reloaded_obs" in layer.metadata["adata"].obs
    assert "reloaded_features" in layer.metadata["adata"].obsm
    assert layer.metadata["region_key"] == "region"
    assert layer.metadata["instance_key"] == "instance_id"
    assert layer.metadata["table_names"] == ["table"]
    assert layer.metadata["indices"] == [1, 2, 3]
    assert layer.metadata["custom_flag"] == "keep-me"


def test_refresh_layer_table_metadata_returns_false_without_loaded_layer(
    sdata_blobs: SpatialData,
) -> None:
    refreshed = refresh_layer_table_metadata(SimpleNamespace(layers=[]), sdata_blobs, "blobs_labels", "table")

    assert refreshed is False


def test_get_loaded_spatialdata_layer_returns_loaded_image_layer(sdata_blobs: SpatialData) -> None:
    image_layer = make_blobs_image_layer(sdata_blobs)
    loaded_layer = spatialdata_module._get_loaded_spatialdata_layer(
        SimpleNamespace(layers=[image_layer]),
        sdata=sdata_blobs,
        element_name="blobs_image",
        layer_filter=lambda layer: isinstance(layer, Image),
    )

    assert loaded_layer is image_layer
    assert (
        spatialdata_module._get_loaded_spatialdata_layer(
            SimpleNamespace(layers=[image_layer]),
            sdata=sdata_blobs,
            element_name="blobs_multiscale_image",
            layer_filter=lambda layer: isinstance(layer, Image),
        )
        is None
    )


def test_get_loaded_spatialdata_layer_rejects_non_image_layers(sdata_blobs: SpatialData) -> None:
    fake_layer = SimpleNamespace(
        metadata={"sdata": sdata_blobs, "name": "blobs_image"},
    )
    loaded_layer = spatialdata_module._get_loaded_spatialdata_layer(
        SimpleNamespace(layers=[fake_layer]),
        sdata=sdata_blobs,
        element_name="blobs_image",
        layer_filter=lambda layer: isinstance(layer, Image),
    )

    assert loaded_layer is None


def test_get_loaded_spatialdata_layer_rejects_non_labels_layers(sdata_blobs: SpatialData) -> None:
    fake_layer = SimpleNamespace(
        metadata={"sdata": sdata_blobs, "name": "blobs_labels"},
        selected_label=1,
        events=SimpleNamespace(selected_label=object()),
    )
    loaded_layer = spatialdata_module._get_loaded_spatialdata_layer(
        SimpleNamespace(layers=[fake_layer]),
        sdata=sdata_blobs,
        element_name="blobs_labels",
        layer_filter=spatialdata_module._is_pickable_labels_layer,
    )

    assert loaded_layer is None


def test_get_loaded_spatialdata_layer_filters_by_coordinate_system(
    sdata_blobs: SpatialData,
) -> None:
    global_layer = make_blobs_labels_layer(sdata_blobs)
    global_layer.metadata["_current_cs"] = "global"
    local_layer = make_blobs_labels_layer(sdata_blobs)
    local_layer.metadata["_current_cs"] = "local"
    loaded_layer = spatialdata_module._get_loaded_spatialdata_layer(
        SimpleNamespace(layers=[local_layer, global_layer]),
        sdata=sdata_blobs,
        element_name="blobs_labels",
        layer_filter=spatialdata_module._is_pickable_labels_layer,
        coordinate_system="global",
    )

    assert loaded_layer is global_layer
