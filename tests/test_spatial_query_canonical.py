from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from spatialdata import SpatialData
from spatialdata.models import TableModel
from xarray import DataArray

from napari_harpy.core.spatial_query import (
    CANONICAL_CHANGED_PATHS,
    CANONICAL_OBSM_KEY,
    SPATIAL_COORDINATES_KEY,
    CanonicalCacheMismatch,
    CanonicalCacheState,
    CanonicalInstallAction,
    CanonicalMismatchCode,
    CanonicalRegionMetadata,
    CanonicalSourceSignature,
    build_canonical_installation_payload,
    build_canonical_metadata,
    build_canonical_region_bindings,
    build_canonical_source_signature,
    build_instance_set_digest,
    canonical_metadata_to_storage,
    inspect_canonical_cache,
    install_canonical_cache,
    parse_canonical_metadata,
)
from napari_harpy.core.spatialdata import SpatialDataTableMetadata


def test_mismatch_scope_is_derived_from_code() -> None:
    pair_mismatch = CanonicalCacheMismatch(code=CanonicalMismatchCode.MATRIX_INVALID)
    region_mismatch = CanonicalCacheMismatch(
        code=CanonicalMismatchCode.TABLE_SIGNATURE_MISMATCH,
        region="nuclei",
    )

    assert pair_mismatch.scope == "pair"
    assert region_mismatch.scope == "region"

    with pytest.raises(ValueError, match="must not name a region"):
        CanonicalCacheMismatch(
            code=CanonicalMismatchCode.MATRIX_INVALID,
            region="nuclei",
        )

    with pytest.raises(ValueError, match="must name a region"):
        CanonicalCacheMismatch(code=CanonicalMismatchCode.TABLE_SIGNATURE_MISMATCH)


def test_instance_set_digest_has_pinned_order_independent_encoding() -> None:
    expected = "sha256:1020a68ff134a26d0139cd20507546c0278f2c308da95133089a5a7c9c8a4718"

    assert build_instance_set_digest("nuclei", [3, 1, 2]) == expected
    assert build_instance_set_digest("nuclei", np.array([2, 3, 1], dtype=np.uint64)) == expected
    assert build_instance_set_digest("Nuclei", [1, 2, 3]) != expected
    assert build_instance_set_digest("nuclei", [1, 2, 4]) != expected
    assert build_instance_set_digest("nuclei", [255]) != build_instance_set_digest("nuclei", [256])
    assert build_instance_set_digest("nuclei", [2**64 - 1]).startswith("sha256:")


@pytest.mark.parametrize("invalid_id", [True, "1", None, np.nan, np.inf, 0, -1, 1.5, 2**64])
def test_instance_set_digest_rejects_invalid_ids(invalid_id: object) -> None:
    with pytest.raises((TypeError, ValueError), match="Instance IDs"):
        build_instance_set_digest("nuclei", [invalid_id])  # type: ignore[list-item]


def test_instance_set_digest_rejects_integer_like_floats() -> None:
    with pytest.raises(TypeError, match="integer NumPy dtype"):
        build_instance_set_digest("nuclei", [1.0, 2.0, 3.0])


def test_source_signature_is_dimension_independent_but_schema_v1_builder_is_2d() -> None:
    signature = CanonicalSourceSignature(
        labels_name="volume",
        source_scale="scale0",
        dims=("z", "y", "x"),
        shape=(128, 50_000, 70_000),
        dtype="uint32",
    )

    assert signature.ndim == 3
    with pytest.raises(ValueError, match="schema version 1"):
        build_canonical_metadata(
            region_key="region",
            instance_key="instance_id",
            regions={
                "volume": CanonicalRegionMetadata(
                    source_signature=signature,
                    table_signature=_table_signature_for("volume", [1]),
                    algorithm_version=1,
                )
            },
        )


def test_source_signature_reads_dataarray_metadata_only() -> None:
    labels = DataArray(np.zeros((4, 5), dtype=np.uint16), dims=("y", "x"))
    sdata = SimpleNamespace(labels={"nuclei": labels})

    assert build_canonical_source_signature(sdata, "nuclei") == CanonicalSourceSignature(  # type: ignore[arg-type]
        labels_name="nuclei",
        source_scale="scale0",
        dims=("y", "x"),
        shape=(4, 5),
        dtype="uint16",
    )


def test_region_bindings_are_strict_and_read_only() -> None:
    table = _simple_table(instance_ids=[2, 1])
    metadata = _table_metadata()

    bindings = build_canonical_region_bindings(table, metadata, "nuclei")

    assert bindings.instance_ids.tolist() == [2, 1]
    assert bindings.signature.instance_set_digest == build_instance_set_digest("nuclei", [1, 2])
    assert not bindings.instance_ids.flags.writeable
    assert not bindings.row_positions.flags.writeable


def test_table_signature_ignores_obs_names_row_order_and_same_set_reassignment() -> None:
    table = _simple_table(instance_ids=[1, 2])
    metadata = _table_metadata()
    original = build_canonical_region_bindings(table, metadata, "nuclei").signature

    table.obs_names = ["renamed-a", "renamed-b"]
    renamed = build_canonical_region_bindings(table, metadata, "nuclei").signature
    reordered = build_canonical_region_bindings(table[[1, 0]].copy(), metadata, "nuclei").signature
    table.obs["instance_id"] = [2, 1]
    reassigned = build_canonical_region_bindings(table, metadata, "nuclei").signature

    assert renamed == reordered == reassigned == original


@pytest.mark.parametrize("instance_ids", [[1, 1], [True, 2], ["1", 2], [1.5, 2], [0, 2], [np.nan, 2]])
def test_region_bindings_reject_invalid_selected_ids(instance_ids: list[object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        build_canonical_region_bindings(_simple_table(instance_ids=instance_ids), _table_metadata(), "nuclei")


def test_metadata_round_trip_is_strict_and_regions_are_read_only() -> None:
    metadata = _metadata_for("nuclei", [1, 2])
    storage = canonical_metadata_to_storage(metadata)

    parsed = parse_canonical_metadata(storage)

    assert parsed == metadata
    with pytest.raises(TypeError):
        parsed.regions["other"] = parsed.regions["nuclei"]  # type: ignore[index]
    malformed = deepcopy(storage)
    malformed["axes"] = ["y", "x"]
    with pytest.raises(ValueError, match="axes"):
        parse_canonical_metadata(malformed)


def test_metadata_storage_round_trips_through_anndata_zarr(tmp_path) -> None:
    table = _simple_table(instance_ids=[1, 2])
    table.obsm[CANONICAL_OBSM_KEY] = np.ones((2, 2), dtype=np.float64)
    table.uns[SPATIAL_COORDINATES_KEY] = {
        CANONICAL_OBSM_KEY: canonical_metadata_to_storage(_metadata_for("nuclei", [1, 2]))
    }
    path = tmp_path / "table.zarr"

    table.write_zarr(path)
    restored = ad.read_zarr(path)

    parsed = parse_canonical_metadata(restored.uns[SPATIAL_COORDINATES_KEY][CANONICAL_OBSM_KEY])
    assert parsed == _metadata_for("nuclei", [1, 2])


def test_inspector_classifies_absent_and_pair_presence_mismatches(sdata_blobs: SpatialData) -> None:
    absent = inspect_canonical_cache(sdata_blobs, table_name="table", labels_name="blobs_labels")
    assert absent.state is CanonicalCacheState.ABSENT

    table = sdata_blobs.tables["table"]
    table.obsm[CANONICAL_OBSM_KEY] = np.full((table.n_obs, 2), np.nan)
    matrix_only = inspect_canonical_cache(sdata_blobs, table_name="table", labels_name="blobs_labels")
    assert _codes(matrix_only) == [CanonicalMismatchCode.MATRIX_WITHOUT_METADATA]

    del table.obsm[CANONICAL_OBSM_KEY]
    table.uns[SPATIAL_COORDINATES_KEY] = {CANONICAL_OBSM_KEY: {}}
    metadata_only = inspect_canonical_cache(sdata_blobs, table_name="table", labels_name="blobs_labels")
    assert _codes(metadata_only) == [CanonicalMismatchCode.METADATA_WITHOUT_MATRIX]


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    [
        (
            lambda table, storage: table.obsm.__setitem__(
                CANONICAL_OBSM_KEY, np.ones((table.n_obs, 2), dtype=np.int64)
            ),
            CanonicalMismatchCode.MATRIX_INVALID,
        ),
        (
            lambda table, storage: storage.__setitem__("schema_version", 2),
            CanonicalMismatchCode.SCHEMA_VERSION_UNSUPPORTED,
        ),
        (lambda table, storage: storage.__setitem__("schema_version", True), CanonicalMismatchCode.METADATA_INVALID),
        (
            lambda table, storage: storage.__setitem__("axes", ["y", "x"]),
            CanonicalMismatchCode.TOP_LEVEL_CONTRACT_MISMATCH,
        ),
        (
            lambda table, storage: storage.__setitem__("region_key", "other_region"),
            CanonicalMismatchCode.TOP_LEVEL_CONTRACT_MISMATCH,
        ),
        (
            lambda table, storage: storage["regions"]["blobs_labels"].pop("coverage"),
            CanonicalMismatchCode.REGION_METADATA_INVALID,
        ),
        (lambda table, storage: storage.pop("dtype"), CanonicalMismatchCode.TOP_LEVEL_CONTRACT_MISMATCH),
    ],
)
def test_inspector_classifies_pair_wide_invalid_states(
    sdata_blobs: SpatialData,
    mutation,
    expected_code: CanonicalMismatchCode,
) -> None:
    _install_selected(sdata_blobs, "blobs_labels")
    table = sdata_blobs.tables["table"]
    storage = table.uns[SPATIAL_COORDINATES_KEY][CANONICAL_OBSM_KEY]
    mutation(table, storage)

    report = inspect_canonical_cache(sdata_blobs, table_name="table", labels_name="blobs_labels")

    assert report.state is CanonicalCacheState.INVALID
    assert _codes(report) == [expected_code]
    assert report.mismatches[0].scope == "pair"


def test_inspector_classifies_partial_valid_and_region_staleness(sdata_blobs: SpatialData) -> None:
    _install_selected(sdata_blobs, "blobs_labels")
    table = sdata_blobs.tables["table"]
    baseline_storage = deepcopy(table.uns[SPATIAL_COORDINATES_KEY][CANONICAL_OBSM_KEY])

    valid = inspect_canonical_cache(sdata_blobs, table_name="table", labels_name="blobs_labels")
    assert valid.state is CanonicalCacheState.VALID

    cases = [
        (
            lambda storage: storage["regions"].pop("blobs_labels"),
            CanonicalCacheState.PARTIAL,
            CanonicalMismatchCode.REGION_NOT_REGISTERED,
        ),
        (
            lambda storage: storage["regions"]["blobs_labels"]["source"]["shape"].__setitem__(0, 999),
            CanonicalCacheState.STALE,
            CanonicalMismatchCode.SOURCE_SIGNATURE_MISMATCH,
        ),
        (
            lambda storage: storage["regions"]["blobs_labels"]["coverage"].__setitem__(
                "instance_set_digest", "sha256:" + "0" * 64
            ),
            CanonicalCacheState.STALE,
            CanonicalMismatchCode.TABLE_SIGNATURE_MISMATCH,
        ),
        (
            lambda storage: storage["regions"]["blobs_labels"]["calculation"].__setitem__("algorithm_version", 2),
            CanonicalCacheState.STALE,
            CanonicalMismatchCode.ALGORITHM_VERSION_MISMATCH,
        ),
    ]
    for mutate, expected_state, expected_code in cases:
        table.uns[SPATIAL_COORDINATES_KEY][CANONICAL_OBSM_KEY] = deepcopy(baseline_storage)
        mutate(table.uns[SPATIAL_COORDINATES_KEY][CANONICAL_OBSM_KEY])
        report = inspect_canonical_cache(sdata_blobs, table_name="table", labels_name="blobs_labels")
        assert report.state is expected_state
        assert expected_code in _codes(report)

    table.uns[SPATIAL_COORDINATES_KEY][CANONICAL_OBSM_KEY] = deepcopy(baseline_storage)
    table.obsm[CANONICAL_OBSM_KEY][0, 0] = np.nan
    coordinates_stale = inspect_canonical_cache(sdata_blobs, table_name="table", labels_name="blobs_labels")
    assert coordinates_stale.state is CanonicalCacheState.STALE
    assert _codes(coordinates_stale) == [CanonicalMismatchCode.REGION_COORDINATES_INVALID]


def test_installer_creates_then_refreshes_and_reports_changed_paths(sdata_blobs: SpatialData) -> None:
    first = _install_selected(sdata_blobs, "blobs_labels", offset=0.0)
    second = _install_selected(sdata_blobs, "blobs_labels", offset=100.0)

    assert first.action is CanonicalInstallAction.CREATE
    assert second.action is CanonicalInstallAction.REFRESH
    assert first.changed_paths == CANONICAL_CHANGED_PATHS
    assert (
        inspect_canonical_cache(sdata_blobs, table_name="table", labels_name="blobs_labels").state
        is CanonicalCacheState.VALID
    )


def test_installer_extends_and_preserves_other_region_byte_for_byte(sdata_blobs: SpatialData) -> None:
    _make_two_region_table(sdata_blobs)
    first = _install_selected(sdata_blobs, "blobs_labels", offset=10.0)
    table = sdata_blobs.tables["table"]
    first_rows = np.flatnonzero(np.asarray(table.obs["region"] == "blobs_labels"))
    first_values = table.obsm[CANONICAL_OBSM_KEY][first_rows].copy()
    first_metadata = deepcopy(table.uns[SPATIAL_COORDINATES_KEY][CANONICAL_OBSM_KEY]["regions"]["blobs_labels"])

    second = _install_selected(sdata_blobs, "blobs_multiscale_labels", offset=100.0)

    assert first.action is CanonicalInstallAction.CREATE
    assert second.action is CanonicalInstallAction.EXTEND
    np.testing.assert_array_equal(table.obsm[CANONICAL_OBSM_KEY][first_rows], first_values)
    assert table.uns[SPATIAL_COORDINATES_KEY][CANONICAL_OBSM_KEY]["regions"]["blobs_labels"] == first_metadata


def test_pair_wide_invalid_install_rebuilds_selected_region_only(sdata_blobs: SpatialData) -> None:
    _make_two_region_table(sdata_blobs)
    _install_selected(sdata_blobs, "blobs_labels")
    _install_selected(sdata_blobs, "blobs_multiscale_labels")
    table = sdata_blobs.tables["table"]
    table.uns[SPATIAL_COORDINATES_KEY][CANONICAL_OBSM_KEY]["axes"] = ["y", "x"]

    result = _install_selected(sdata_blobs, "blobs_labels", offset=200.0)

    regions = table.uns[SPATIAL_COORDINATES_KEY][CANONICAL_OBSM_KEY]["regions"]
    other_rows = np.flatnonzero(np.asarray(table.obs["region"] == "blobs_multiscale_labels"))
    assert result.action is CanonicalInstallAction.REBUILD
    assert set(regions) == {"blobs_labels"}
    assert np.isnan(table.obsm[CANONICAL_OBSM_KEY][other_rows]).all()


def test_installer_rejects_changed_table_without_mutation(sdata_blobs: SpatialData) -> None:
    report = inspect_canonical_cache(sdata_blobs, table_name="table", labels_name="blobs_labels")
    payload = _payload_from_report(report, "table")
    table = sdata_blobs.tables["table"]
    table.obs.iloc[0, table.obs.columns.get_loc("instance_id")] = 999

    with pytest.raises(ValueError, match="changed"):
        install_canonical_cache(sdata_blobs, payload)

    assert CANONICAL_OBSM_KEY not in table.obsm
    assert SPATIAL_COORDINATES_KEY not in table.uns


def test_installer_rolls_back_both_paths_when_second_assignment_fails(sdata_blobs: SpatialData) -> None:
    report = inspect_canonical_cache(sdata_blobs, table_name="table", labels_name="blobs_labels")
    payload = _payload_from_report(report, "table")
    table = sdata_blobs.tables["table"]
    table.uns = _FailOnceSpatialCoordinatesDict(table.uns)

    with pytest.raises(RuntimeError, match="injected assignment failure"):
        install_canonical_cache(sdata_blobs, payload)

    assert CANONICAL_OBSM_KEY not in table.obsm
    assert SPATIAL_COORDINATES_KEY not in table.uns


def test_inspector_does_not_normalize_table_attrs_in_place(sdata_blobs: SpatialData) -> None:
    table = sdata_blobs.tables["table"]
    attrs = table.uns[TableModel.ATTRS_KEY]
    attrs[TableModel.REGION_KEY] = np.array(["blobs_labels"])
    before = attrs[TableModel.REGION_KEY]

    inspect_canonical_cache(sdata_blobs, table_name="table", labels_name="blobs_labels")

    assert attrs[TableModel.REGION_KEY] is before


def _simple_table(*, instance_ids: list[object]) -> AnnData:
    table = AnnData(
        X=np.zeros((len(instance_ids), 1)),
        obs=pd.DataFrame(
            {"region": ["nuclei"] * len(instance_ids), "instance_id": instance_ids},
            index=[f"row-{index}" for index in range(len(instance_ids))],
        ),
    )
    table.uns[TableModel.ATTRS_KEY] = {
        TableModel.REGION_KEY: "nuclei",
        TableModel.REGION_KEY_KEY: "region",
        TableModel.INSTANCE_KEY: "instance_id",
    }
    return table


def _table_metadata() -> SpatialDataTableMetadata:
    return SpatialDataTableMetadata(
        table_name="table",
        region_key="region",
        instance_key="instance_id",
        regions=("nuclei",),
    )


def _table_signature_for(labels_name: str, instance_ids: list[int]):
    from napari_harpy.core.spatial_query import CanonicalTableSignature

    return CanonicalTableSignature(
        labels_name=labels_name,
        region_key="region",
        instance_key="instance_id",
        n_obs=len(instance_ids),
        instance_set_digest=build_instance_set_digest(labels_name, instance_ids),
    )


def _metadata_for(labels_name: str, instance_ids: list[int]):
    source = CanonicalSourceSignature(
        labels_name=labels_name,
        source_scale="scale0",
        dims=("y", "x"),
        shape=(4, 5),
        dtype="uint16",
    )
    return build_canonical_metadata(
        region_key="region",
        instance_key="instance_id",
        regions={
            labels_name: CanonicalRegionMetadata(
                source_signature=source,
                table_signature=_table_signature_for(labels_name, instance_ids),
                algorithm_version=1,
                generated_by_package="napari-harpy",
                generated_by_version="0.1.1",
                generated_at="2026-07-14T00:00:00Z",
            )
        },
    )


def _payload_from_report(report, table_name: str, *, offset: float = 0.0):
    centers = np.arange(report.bindings.signature.n_obs * 2, dtype=np.float64).reshape(-1, 2) + offset
    return build_canonical_installation_payload(
        table_name=table_name,
        bindings=report.bindings,
        centers_xy=centers,
        source_signature=report.source_signature,
    )


def _install_selected(sdata: SpatialData, labels_name: str, *, offset: float = 0.0):
    report = inspect_canonical_cache(sdata, table_name="table", labels_name=labels_name)
    return install_canonical_cache(sdata, _payload_from_report(report, "table", offset=offset))


def _make_two_region_table(sdata: SpatialData) -> None:
    table = sdata.tables["table"]
    table.obs["region"] = table.obs["region"].astype(str)
    midpoint = table.n_obs // 2
    table.obs.iloc[midpoint:, table.obs.columns.get_loc("region")] = "blobs_multiscale_labels"
    table.uns[TableModel.ATTRS_KEY][TableModel.REGION_KEY] = [
        "blobs_labels",
        "blobs_multiscale_labels",
    ]


def _codes(report) -> list[CanonicalMismatchCode]:
    return [mismatch.code for mismatch in report.mismatches]


class _FailOnceSpatialCoordinatesDict(dict):
    def __init__(self, values) -> None:
        super().__init__(values)
        self._should_fail = True

    def __setitem__(self, key, value) -> None:
        if key == SPATIAL_COORDINATES_KEY and self._should_fail:
            self._should_fail = False
            raise RuntimeError("injected assignment failure")
        super().__setitem__(key, value)
