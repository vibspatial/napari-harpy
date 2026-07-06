from __future__ import annotations

import anndata as ad
import numpy as np
import pytest
from scipy import sparse

from napari_harpy.core.classifier_export import normalize_feature_columns
from napari_harpy.core.feature_matrix_metadata import (
    CUSTOM_OBSM_FEATURE_NAME,
    CUSTOM_OBSM_SOURCE_KIND,
    HARPY_ADD_FEATURE_MATRIX_SOURCE_KIND,
    FeatureMatrixMetadataState,
    inspect_feature_matrix_metadata,
    is_custom_obsm_feature_metadata,
    normalize_feature_matrix_source_kind,
    register_feature_matrix_metadata,
)

_FEATURE_MATRICES_KEY = "feature_matrices"


def _make_table(n_obs: int = 3) -> ad.AnnData:
    return ad.AnnData(np.empty((n_obs, 0), dtype=np.float64))


class _FeatureMatrixTable:
    def __init__(self, *, obsm: dict[str, object], n_obs: int = 3) -> None:
        self.obsm = obsm
        self.uns: dict[str, object] = {}
        self.n_obs = n_obs


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
    harpy_metadata = {
        "feature_columns": ["area"],
        "features": ["area"],
        "source_kind": HARPY_ADD_FEATURE_MATRIX_SOURCE_KIND,
    }

    assert is_custom_obsm_feature_metadata(custom_metadata)
    assert not is_custom_obsm_feature_metadata(harpy_metadata)


def test_normalize_feature_matrix_source_kind_rejects_missing_or_unknown_source_kind() -> None:
    with pytest.raises(ValueError, match="source_kind"):
        normalize_feature_matrix_source_kind({"feature_columns": ["area"], "features": ["area"]})

    with pytest.raises(ValueError, match="unknown"):
        normalize_feature_matrix_source_kind(
            {"feature_columns": ["area"], "features": ["area"], "source_kind": "unknown"},
        )


def test_feature_matrix_metadata_state_rejects_unknown_status() -> None:
    with pytest.raises(ValueError, match="Unsupported feature matrix metadata status"):
        FeatureMatrixMetadataState(feature_key="custom", status="registered")


def test_feature_matrix_metadata_state_rejects_unknown_source_kind() -> None:
    with pytest.raises(ValueError, match="Unsupported feature matrix metadata source kind"):
        FeatureMatrixMetadataState(feature_key="custom", status="registered_valid", source_kind="unknown")


def test_inspect_feature_matrix_metadata_reports_missing_matrix() -> None:
    table = _make_table()

    state = inspect_feature_matrix_metadata(table, "missing")

    assert state.status == "missing_matrix"
    assert state.n_features is None
    assert state.metadata_n_features is None
    assert state.source_kind is None
    assert not state.is_custom_obsm
    assert state.error is not None


def test_inspect_feature_matrix_metadata_reports_invalid_matrix() -> None:
    table = _make_table()
    table.obsm["cube"] = np.ones((3, 2, 2), dtype=np.float64)

    state = inspect_feature_matrix_metadata(table, "cube")

    assert state.status == "invalid_matrix"
    assert state.n_features is None
    assert state.metadata_n_features is None
    assert state.source_kind is None
    assert state.error == "Feature matrices stored in `.obsm` must be 2-dimensional."


def test_inspect_feature_matrix_metadata_reports_wrong_row_count_as_invalid_matrix() -> None:
    table = _FeatureMatrixTable(obsm={"wrong_rows": np.ones((2, 1), dtype=np.float64)})

    state = inspect_feature_matrix_metadata(table, "wrong_rows")

    assert state.status == "invalid_matrix"
    assert state.source_kind is None
    assert state.error == "Feature matrix has 2 rows but the table has 3 observations."


def test_inspect_feature_matrix_metadata_reports_unregistered_matrix() -> None:
    table = _make_table()
    table.obsm["custom"] = np.arange(6, dtype=np.float64).reshape(3, 2)

    state = inspect_feature_matrix_metadata(table, "custom")

    assert state.status == "unregistered"
    assert state.n_features == 2
    assert state.metadata_n_features is None
    assert state.source_kind is None
    assert not state.is_custom_obsm
    assert state.error is None
    assert _FEATURE_MATRICES_KEY not in table.uns


def test_inspect_feature_matrix_metadata_reports_valid_custom_metadata() -> None:
    table = _make_table()
    table.obsm["custom"] = np.arange(6, dtype=np.float64).reshape(3, 2)
    register_feature_matrix_metadata(table, "custom", feature_columns=("area", "texture"))

    state = inspect_feature_matrix_metadata(table, "custom")

    assert state.status == "registered_valid"
    assert state.n_features == 2
    assert state.metadata_n_features == 2
    assert state.source_kind == CUSTOM_OBSM_SOURCE_KIND
    assert state.is_custom_obsm
    assert state.error is None


def test_inspect_feature_matrix_metadata_reports_valid_harpy_metadata() -> None:
    table = _make_table()
    table.obsm["harpy"] = np.arange(6, dtype=np.float64).reshape(3, 2)
    table.uns[_FEATURE_MATRICES_KEY] = {
        "harpy": {
            "feature_columns": ["area", "perimeter"],
            "features": ["area", "perimeter"],
            "source_kind": HARPY_ADD_FEATURE_MATRIX_SOURCE_KIND,
        },
    }

    state = inspect_feature_matrix_metadata(table, "harpy")

    assert state.status == "registered_valid"
    assert state.n_features == 2
    assert state.metadata_n_features == 2
    assert state.source_kind == HARPY_ADD_FEATURE_MATRIX_SOURCE_KIND
    assert not state.is_custom_obsm


def test_inspect_feature_matrix_metadata_reports_missing_source_kind_as_mismatched() -> None:
    table = _make_table()
    table.obsm["harpy"] = np.arange(6, dtype=np.float64).reshape(3, 2)
    table.uns[_FEATURE_MATRICES_KEY] = {
        "harpy": {"feature_columns": ["area", "perimeter"], "features": ["area", "perimeter"]},
    }

    state = inspect_feature_matrix_metadata(table, "harpy")

    assert state.status == "registered_mismatched"
    assert state.n_features == 2
    assert state.metadata_n_features is None
    assert state.source_kind is None
    assert state.error is not None
    assert "source_kind" in state.error


def test_inspect_feature_matrix_metadata_reports_mismatched_metadata_column_count() -> None:
    table = _make_table()
    table.obsm["custom"] = np.arange(6, dtype=np.float64).reshape(3, 2)
    table.uns[_FEATURE_MATRICES_KEY] = {
        "custom": {
            "feature_columns": ["area"],
            "features": ["area"],
            "source_kind": HARPY_ADD_FEATURE_MATRIX_SOURCE_KIND,
        },
    }

    state = inspect_feature_matrix_metadata(table, "custom")

    assert state.status == "registered_mismatched"
    assert state.n_features == 2
    assert state.metadata_n_features == 1
    assert state.source_kind == HARPY_ADD_FEATURE_MATRIX_SOURCE_KIND
    assert not state.is_custom_obsm
    assert state.error == 'Feature matrix "custom" has 2 column(s), but its metadata describes 1 feature column(s).'
