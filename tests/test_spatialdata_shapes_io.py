from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Polygon
from spatialdata import SpatialData, read_zarr
from spatialdata.models import ShapesModel
from spatialdata.transformations import Identity, get_transformation

import napari_harpy.core.spatialdata_io.shapes as shapes_io_module
from napari_harpy.core.spatialdata_io import (
    load_shapes_element_from_store,
    shapes_element_exists_in_store,
    write_shapes_element,
)


def _shapes_element(*, offset: float = 0.0, index_value: str = "region") -> gpd.GeoDataFrame:
    geodataframe = gpd.GeoDataFrame(
        {"class_id": [1]},
        geometry=[
            Polygon(
                [
                    (offset, 0),
                    (offset + 2, 0),
                    (offset + 2, 2),
                    (offset, 2),
                ]
            )
        ],
        index=pd.Index([index_value], name="instance_id"),
    )
    return ShapesModel.parse(geodataframe, transformations={"global": Identity()})


def _backed_shapes_sdata(tmp_path: Path) -> SpatialData:
    path = tmp_path / "shapes.zarr"
    SpatialData(
        shapes={
            "regions": _shapes_element(),
            "unrelated": _shapes_element(offset=100, index_value="unrelated"),
        }
    ).write(path)
    return read_zarr(path)


def test_load_shapes_element_from_store_installs_only_the_exact_element(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdata = _backed_shapes_sdata(tmp_path)
    sdata.shapes["regions"] = _shapes_element(offset=10)
    unrelated = sdata.shapes["unrelated"]
    read_paths: list[Path] = []
    real_read = shapes_io_module._spatialdata_read_shapes

    def record_exact_read(path: str | Path) -> gpd.GeoDataFrame:
        read_paths.append(Path(path))
        return real_read(path)

    monkeypatch.setattr(shapes_io_module, "_spatialdata_read_shapes", record_exact_read)

    assert shapes_element_exists_in_store(sdata, "regions") is True
    assert shapes_element_exists_in_store(sdata, "missing") is False

    loaded = load_shapes_element_from_store(sdata, "regions")

    assert read_paths == [sdata.path / "shapes" / "regions"]
    assert loaded is sdata.shapes["regions"]
    assert loaded.index.tolist() == ["region"]
    assert loaded.index.name == "instance_id"
    assert loaded["class_id"].tolist() == [1]
    assert loaded.geometry.iloc[0].bounds == (0.0, 0.0, 2.0, 2.0)
    assert isinstance(get_transformation(loaded, get_all=True)["global"], Identity)
    assert sdata.shapes["unrelated"] is unrelated


def test_load_shapes_element_validation_failure_preserves_live_element(tmp_path: Path) -> None:
    sdata = _backed_shapes_sdata(tmp_path)
    current = _shapes_element(offset=10)
    sdata.shapes["regions"] = current

    def reject_candidate(candidate: gpd.GeoDataFrame) -> None:
        assert candidate.geometry.iloc[0].bounds == (0.0, 0.0, 2.0, 2.0)
        raise ValueError("candidate rejected")

    with pytest.raises(ValueError, match="candidate rejected"):
        load_shapes_element_from_store(
            sdata,
            "regions",
            validate_before_install=reject_candidate,
        )

    assert sdata.shapes["regions"] is current


def test_write_shapes_element_preserves_unbacked_overwrite_semantics() -> None:
    sdata = SpatialData()
    created = _shapes_element()

    assert write_shapes_element(sdata, "regions", created, overwrite=True) is created

    with pytest.raises(ValueError, match="overwrite=True"):
        write_shapes_element(
            sdata,
            "regions",
            _shapes_element(offset=5),
            overwrite=False,
        )

    replacement = _shapes_element(offset=10)
    assert write_shapes_element(sdata, "regions", replacement, overwrite=True) is replacement
    assert sdata.shapes["regions"].geometry.iloc[0].bounds == (10.0, 0.0, 12.0, 2.0)


def test_write_shapes_element_creates_first_backed_shapes_collection(tmp_path: Path) -> None:
    path = tmp_path / "empty.zarr"
    SpatialData().write(path)
    sdata = read_zarr(path)

    committed = write_shapes_element(
        sdata,
        "regions",
        _shapes_element(offset=10),
        overwrite=True,
    )

    reread = read_zarr(path)
    assert committed is sdata.shapes["regions"]
    assert reread.shapes["regions"].geometry.iloc[0].bounds == (10.0, 0.0, 12.0, 2.0)
    assert (path / "shapes" / "zarr.json").is_file()
    assert not (path / ".harpy_recovery").exists()


def test_write_shapes_element_backed_create_and_overwrite_leave_no_staging_element(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdata = _backed_shapes_sdata(tmp_path)
    staging_writes: list[str] = []
    consolidation_count = 0
    consolidation_with_recovery_count = 0
    real_staging_write = shapes_io_module._write_staging_shapes_element
    real_consolidate = SpatialData.write_consolidated_metadata

    def record_staging_write(
        current_sdata: SpatialData,
        *,
        store_path: Path,
        staging_name: str,
        element: gpd.GeoDataFrame,
    ) -> None:
        staging_writes.append(staging_name)
        root_metadata = (store_path / "zarr.json").read_bytes()
        shapes_metadata = (store_path / "shapes" / "zarr.json").read_bytes()
        real_staging_write(
            current_sdata,
            store_path=store_path,
            staging_name=staging_name,
            element=element,
        )
        assert (store_path / "zarr.json").read_bytes() == root_metadata
        assert (store_path / "shapes" / "zarr.json").read_bytes() == shapes_metadata

    def record_consolidation(current_sdata: SpatialData) -> None:
        nonlocal consolidation_count, consolidation_with_recovery_count
        consolidation_count += 1
        recovery_root = Path(current_sdata.path) / ".harpy_recovery"
        has_recovery_payload = recovery_root.exists() and bool(list(recovery_root.iterdir()))
        real_consolidate(current_sdata)
        if has_recovery_payload:
            consolidation_with_recovery_count += 1
            assert set(read_zarr(current_sdata.path).shapes) == {"new_regions", "regions", "unrelated"}

    def reject_generic_element_io(*args, **kwargs) -> None:
        del args, kwargs
        raise AssertionError("The exact Shapes commit must not use generic SpatialData element I/O.")

    monkeypatch.setattr(shapes_io_module, "_write_staging_shapes_element", record_staging_write)
    monkeypatch.setattr(SpatialData, "write_consolidated_metadata", record_consolidation)
    monkeypatch.setattr(SpatialData, "write_element", reject_generic_element_io)
    monkeypatch.setattr(SpatialData, "delete_element_from_disk", reject_generic_element_io)

    write_shapes_element(
        sdata,
        "new_regions",
        _shapes_element(offset=10, index_value="new"),
        overwrite=True,
    )
    committed = write_shapes_element(
        sdata,
        "regions",
        _shapes_element(offset=20, index_value="replacement"),
        overwrite=True,
    )

    reread = read_zarr(sdata.path)
    assert committed is sdata.shapes["regions"]
    assert reread.shapes["new_regions"].geometry.iloc[0].bounds == (10.0, 0.0, 12.0, 2.0)
    assert reread.shapes["regions"].geometry.iloc[0].bounds == (20.0, 0.0, 22.0, 2.0)
    assert reread.shapes["unrelated"].geometry.iloc[0].bounds == (100.0, 0.0, 102.0, 2.0)
    shapes_path = Path(sdata.path) / "shapes"
    assert not list(shapes_path.glob("*__napari_harpy_stage_*"))
    assert not (Path(sdata.path) / ".harpy_recovery").exists()
    assert len(staging_writes) == 2
    assert consolidation_count == 2
    assert consolidation_with_recovery_count == 1


@pytest.mark.parametrize(
    "failure_stage",
    [
        "staging_serialization",
        "persisted_to_recovery_rename",
        "staging_to_requested_rename",
        "exact_read",
    ],
)
def test_write_shapes_element_precommit_failure_restores_previous_disk_and_live_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
) -> None:
    sdata = _backed_shapes_sdata(tmp_path)
    previous_live = sdata.shapes["regions"]
    real_staging_write = shapes_io_module._write_staging_shapes_element
    real_replace = shapes_io_module.os.replace
    real_read = shapes_io_module._spatialdata_read_shapes

    def staging_write_with_selected_failure(
        current_sdata: SpatialData,
        *,
        store_path: Path,
        staging_name: str,
        element: gpd.GeoDataFrame,
    ) -> None:
        if failure_stage == "staging_serialization":
            raise OSError("injected staging serialization failure")
        real_staging_write(
            current_sdata,
            store_path=store_path,
            staging_name=staging_name,
            element=element,
        )

    def rename_with_selected_failure(source: str | Path, destination: str | Path) -> None:
        source_path = Path(source)
        if failure_stage == "persisted_to_recovery_rename" and source_path.name == "regions":
            raise OSError("injected persisted-to-recovery rename failure")
        if failure_stage == "staging_to_requested_rename" and "__napari_harpy_stage_" in source_path.name:
            raise OSError("injected staging-to-requested rename failure")
        real_replace(source, destination)

    def exact_read_with_selected_failure(path: str | Path) -> gpd.GeoDataFrame:
        if failure_stage == "exact_read" and Path(path).name == "regions":
            raise OSError("injected exact read failure")
        return real_read(path)

    monkeypatch.setattr(shapes_io_module, "_write_staging_shapes_element", staging_write_with_selected_failure)
    monkeypatch.setattr(shapes_io_module.os, "replace", rename_with_selected_failure)
    monkeypatch.setattr(shapes_io_module, "_spatialdata_read_shapes", exact_read_with_selected_failure)

    with pytest.raises(OSError, match="injected"):
        write_shapes_element(
            sdata,
            "regions",
            _shapes_element(offset=20),
            overwrite=True,
        )

    reread = read_zarr(sdata.path)
    assert sdata.shapes["regions"] is previous_live
    assert reread.shapes["regions"].geometry.iloc[0].bounds == (0.0, 0.0, 2.0, 2.0)
    shapes_path = Path(sdata.path) / "shapes"
    assert not list(shapes_path.glob("*__napari_harpy_stage_*"))
    assert not (Path(sdata.path) / ".harpy_recovery").exists()


def test_write_shapes_element_consolidation_failure_restores_previous_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdata = _backed_shapes_sdata(tmp_path)
    previous_live = sdata.shapes["regions"]
    real_consolidate = SpatialData.write_consolidated_metadata
    consolidation_count = 0

    def fail_first_consolidation(current_sdata: SpatialData) -> None:
        nonlocal consolidation_count
        consolidation_count += 1
        if consolidation_count == 1:
            raise OSError("injected metadata consolidation failure")
        real_consolidate(current_sdata)

    monkeypatch.setattr(SpatialData, "write_consolidated_metadata", fail_first_consolidation)

    with pytest.raises(OSError, match="injected metadata consolidation failure"):
        write_shapes_element(
            sdata,
            "regions",
            _shapes_element(offset=20),
            overwrite=True,
        )

    reread = read_zarr(sdata.path)
    assert sdata.shapes["regions"] is previous_live
    assert reread.shapes["regions"].geometry.iloc[0].bounds == (0.0, 0.0, 2.0, 2.0)
    assert consolidation_count == 2
    assert not (Path(sdata.path) / ".harpy_recovery").exists()


def test_write_shapes_element_second_consolidation_failure_reports_recovery_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdata = _backed_shapes_sdata(tmp_path)
    previous_live = sdata.shapes["regions"]
    consolidation_count = 0

    def fail_consolidation(current_sdata: SpatialData) -> None:
        nonlocal consolidation_count
        del current_sdata
        consolidation_count += 1
        raise OSError(f"injected metadata consolidation failure {consolidation_count}")

    monkeypatch.setattr(SpatialData, "write_consolidated_metadata", fail_consolidation)

    with pytest.raises(RuntimeError, match="could not consolidate the restored SpatialData state"):
        write_shapes_element(
            sdata,
            "regions",
            _shapes_element(offset=20),
            overwrite=True,
        )

    assert consolidation_count == 2
    assert sdata.shapes["regions"] is previous_live
    assert read_zarr(sdata.path).shapes["regions"].geometry.iloc[0].bounds == (0.0, 0.0, 2.0, 2.0)
    assert not (Path(sdata.path) / ".harpy_recovery").exists()


def test_write_shapes_element_recovery_cleanup_failure_preserves_valid_committed_store(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdata = _backed_shapes_sdata(tmp_path)
    real_remove = shapes_io_module._remove_directory_if_present
    consolidation_count = 0
    real_consolidate = SpatialData.write_consolidated_metadata

    def fail_recovery_cleanup(path: Path) -> None:
        if path.parent.name == ".harpy_recovery":
            raise OSError("injected recovery cleanup failure")
        real_remove(path)

    def record_consolidation(current_sdata: SpatialData) -> None:
        nonlocal consolidation_count
        consolidation_count += 1
        real_consolidate(current_sdata)

    monkeypatch.setattr(shapes_io_module, "_remove_directory_if_present", fail_recovery_cleanup)
    monkeypatch.setattr(SpatialData, "write_consolidated_metadata", record_consolidation)

    with pytest.raises(RuntimeError, match="consolidated store remains valid"):
        write_shapes_element(
            sdata,
            "regions",
            _shapes_element(offset=20),
            overwrite=True,
        )

    reread = read_zarr(sdata.path)
    assert sdata.shapes["regions"].geometry.iloc[0].bounds == (20.0, 0.0, 22.0, 2.0)
    assert reread.shapes["regions"].geometry.iloc[0].bounds == (20.0, 0.0, 22.0, 2.0)
    assert set(reread.shapes) == {"regions", "unrelated"}
    recovery_payloads = list((Path(sdata.path) / ".harpy_recovery").iterdir())
    assert len(recovery_payloads) == 1
    assert recovery_payloads[0].name.startswith("shapes__regions__")
    assert consolidation_count == 1


@pytest.mark.parametrize("failure_stage", ["staging_serialization", "metadata_consolidation"])
def test_write_shapes_element_failed_first_shapes_create_removes_collection_group(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
) -> None:
    path = tmp_path / "empty.zarr"
    SpatialData().write(path)
    sdata = read_zarr(path)
    real_staging_write = shapes_io_module._write_staging_shapes_element
    real_consolidate = SpatialData.write_consolidated_metadata
    consolidation_count = 0

    def write_staging_then_maybe_fail(
        current_sdata: SpatialData,
        *,
        store_path: Path,
        staging_name: str,
        element: gpd.GeoDataFrame,
    ) -> None:
        real_staging_write(
            current_sdata,
            store_path=store_path,
            staging_name=staging_name,
            element=element,
        )
        if failure_stage == "staging_serialization":
            raise OSError("injected staging serialization failure")

    def fail_first_consolidation(current_sdata: SpatialData) -> None:
        nonlocal consolidation_count
        consolidation_count += 1
        if failure_stage == "metadata_consolidation" and consolidation_count == 1:
            raise OSError("injected metadata consolidation failure")
        real_consolidate(current_sdata)

    monkeypatch.setattr(shapes_io_module, "_write_staging_shapes_element", write_staging_then_maybe_fail)
    monkeypatch.setattr(SpatialData, "write_consolidated_metadata", fail_first_consolidation)

    with pytest.raises(OSError, match="injected"):
        write_shapes_element(
            sdata,
            "regions",
            _shapes_element(),
            overwrite=True,
        )

    assert "regions" not in sdata.shapes
    assert not (path / "shapes").exists()
    assert not read_zarr(path).shapes
    expected_consolidation_count = 0 if failure_stage == "staging_serialization" else 2
    assert consolidation_count == expected_consolidation_count


@pytest.mark.parametrize("recovery_root_kind", ["file", "zarr_group"])
def test_write_shapes_element_rejects_invalid_recovery_root_before_requested_rename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    recovery_root_kind: str,
) -> None:
    sdata = _backed_shapes_sdata(tmp_path)
    previous_live = sdata.shapes["regions"]
    recovery_root = Path(sdata.path) / ".harpy_recovery"
    if recovery_root_kind == "file":
        recovery_root.write_text("reserved path conflict")
    else:
        recovery_root.mkdir()
        (recovery_root / "zarr.json").write_text("{}")

    real_replace = shapes_io_module.os.replace
    requested_was_moved = False

    def record_requested_rename(source: str | Path, destination: str | Path) -> None:
        nonlocal requested_was_moved
        if Path(source).name == "regions":
            requested_was_moved = True
        real_replace(source, destination)

    monkeypatch.setattr(shapes_io_module.os, "replace", record_requested_rename)

    with pytest.raises(ValueError, match="Shapes recovery path"):
        write_shapes_element(
            sdata,
            "regions",
            _shapes_element(offset=20),
            overwrite=True,
        )

    assert not requested_was_moved
    assert sdata.shapes["regions"] is previous_live
    assert read_zarr(sdata.path).shapes["regions"].geometry.iloc[0].bounds == (0.0, 0.0, 2.0, 2.0)
    assert not list((Path(sdata.path) / "shapes").glob("*__napari_harpy_stage_*"))
