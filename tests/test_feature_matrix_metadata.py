from __future__ import annotations

import anndata as ad
import numpy as np
import pytest
from scipy import sparse

from napari_harpy.core.classifier_export import normalize_feature_columns
from napari_harpy.core.feature_matrix_metadata import (
    CUSTOM_OBSM_FEATURE_NAME,
    CUSTOM_OBSM_SOURCE_KIND,
    is_custom_obsm_feature_metadata,
    register_feature_matrix_metadata,
)

_FEATURE_MATRICES_KEY = "feature_matrices"


def _make_table(n_obs: int = 3) -> ad.AnnData:
    return ad.AnnData(np.empty((n_obs, 0), dtype=np.float64))


def test_register_feature_matrix_metadata_rejects_missing_obsm_key() -> None:
    table = _make_table()

    with pytest.raises(ValueError, match='Feature matrix "missing" is not available in ".obsm"'):
        register_feature_matrix_metadata(table, "missing")


def test_register_feature_matrix_metadata_registers_one_dimensional_matrix() -> None:
    table = _make_table()
    table.obsm["scores"] = np.array([0.1, 0.2, 0.3], dtype=np.float32)

    metadata = register_feature_matrix_metadata(table, "scores")

    assert metadata["feature_columns"] == ["scores_0"]
    assert metadata["features"] == [CUSTOM_OBSM_FEATURE_NAME]
    assert metadata["source_kind"] == CUSTOM_OBSM_SOURCE_KIND
    assert metadata["schema_version"] == 1
    assert metadata["backend"] == "numpy"
    assert metadata["dtype"] == "float32"
    assert table.uns[_FEATURE_MATRICES_KEY]["scores"] == metadata
    assert normalize_feature_columns(metadata) == ("scores_0",)


def test_register_feature_matrix_metadata_registers_dense_matrix_with_default_columns() -> None:
    table = _make_table()
    table.obsm["custom"] = np.arange(6, dtype=np.float64).reshape(3, 2)

    metadata = register_feature_matrix_metadata(table, "custom")

    assert metadata["feature_columns"] == ["custom_0", "custom_1"]
    assert metadata["features"] == [CUSTOM_OBSM_FEATURE_NAME]
    assert metadata["source_kind"] == CUSTOM_OBSM_SOURCE_KIND
    assert set(metadata).isdisjoint({"source_image", "source_channels", "source_label", "coordinate_system"})


def test_register_feature_matrix_metadata_registers_sparse_matrix_backend_and_dtype() -> None:
    table = _make_table()
    table.obsm["sparse_features"] = sparse.csr_matrix(
        np.arange(6, dtype=np.float32).reshape(3, 2),
    )

    metadata = register_feature_matrix_metadata(table, "sparse_features")

    assert metadata["feature_columns"] == ["sparse_features_0", "sparse_features_1"]
    assert metadata["backend"] == "sparse"
    assert metadata["dtype"] == "float32"


def test_register_feature_matrix_metadata_preserves_supplied_columns_and_features() -> None:
    table = _make_table()
    table.obsm["custom"] = np.arange(6, dtype=np.float64).reshape(3, 2)

    metadata = register_feature_matrix_metadata(
        table,
        "custom",
        feature_columns=("area", "texture"),
        features=("custom_features",),
    )

    assert metadata["feature_columns"] == ["area", "texture"]
    assert metadata["features"] == ["custom_features"]


@pytest.mark.parametrize(
    ("feature_columns", "match"),
    [
        (("area",), "describes 1 feature column"),
        (("area", "area"), "duplicate"),
        (("area", ""), "empty"),
    ],
)
def test_register_feature_matrix_metadata_rejects_invalid_feature_columns(
    feature_columns: tuple[str, ...],
    match: str,
) -> None:
    table = _make_table()
    table.obsm["custom"] = np.arange(6, dtype=np.float64).reshape(3, 2)

    with pytest.raises(ValueError, match=match):
        register_feature_matrix_metadata(table, "custom", feature_columns=feature_columns)


@pytest.mark.parametrize(
    ("features", "match"),
    [
        ((), "at least one"),
        (("custom", ""), "empty"),
        (("custom", "custom"), "duplicate"),
    ],
)
def test_register_feature_matrix_metadata_rejects_invalid_features(
    features: tuple[str, ...],
    match: str,
) -> None:
    table = _make_table()
    table.obsm["custom"] = np.arange(6, dtype=np.float64).reshape(3, 2)

    with pytest.raises(ValueError, match=match):
        register_feature_matrix_metadata(table, "custom", features=features)


def test_register_feature_matrix_metadata_does_not_overwrite_existing_metadata_by_default() -> None:
    table = _make_table()
    table.obsm["custom"] = np.arange(6, dtype=np.float64).reshape(3, 2)
    table.uns[_FEATURE_MATRICES_KEY] = {
        "custom": {"feature_columns": ["old_0", "old_1"], "features": ["old"]},
        "other": {"feature_columns": ["other_0"], "features": ["other"]},
    }

    with pytest.raises(ValueError, match="already has metadata"):
        register_feature_matrix_metadata(table, "custom")

    metadata = register_feature_matrix_metadata(table, "custom", overwrite=True)

    assert metadata["source_kind"] == CUSTOM_OBSM_SOURCE_KIND
    assert table.uns[_FEATURE_MATRICES_KEY]["custom"] == metadata
    assert table.uns[_FEATURE_MATRICES_KEY]["other"] == {"feature_columns": ["other_0"], "features": ["other"]}


def test_register_feature_matrix_metadata_rejects_non_mapping_metadata_namespace() -> None:
    table = _make_table()
    table.obsm["custom"] = np.arange(6, dtype=np.float64).reshape(3, 2)
    table.uns[_FEATURE_MATRICES_KEY] = "not-a-mapping"

    with pytest.raises(ValueError, match="must be a mapping"):
        register_feature_matrix_metadata(table, "custom")


def test_is_custom_obsm_feature_metadata_requires_explicit_source_kind() -> None:
    custom_metadata = {"feature_columns": ["a"], "features": [CUSTOM_OBSM_FEATURE_NAME], "source_kind": "custom_obsm"}
    legacy_harpy_metadata = {"feature_columns": ["area"], "features": ["area"]}

    assert is_custom_obsm_feature_metadata(custom_metadata)
    assert not is_custom_obsm_feature_metadata(legacy_harpy_metadata)
