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


def test_write_shapes_element_backed_create_and_overwrite_leave_no_staging_element(
    tmp_path: Path,
) -> None:
    sdata = _backed_shapes_sdata(tmp_path)

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
    assert not any("__napari_harpy_stage_" in path for path in sdata.elements_paths_on_disk())


def test_write_shapes_element_target_failure_restores_previous_disk_and_live_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdata = _backed_shapes_sdata(tmp_path)
    previous_live = sdata.shapes["regions"]
    real_write_element = SpatialData.write_element
    fail_target_once = True

    def fail_requested_target_once(
        current_sdata: SpatialData,
        element_name: str | list[str],
        *args,
        **kwargs,
    ) -> None:
        nonlocal fail_target_once
        if element_name == "regions" and fail_target_once:
            fail_target_once = False
            raise OSError("injected target write failure")
        real_write_element(current_sdata, element_name, *args, **kwargs)

    monkeypatch.setattr(SpatialData, "write_element", fail_requested_target_once)

    with pytest.raises(OSError, match="injected target write failure"):
        write_shapes_element(
            sdata,
            "regions",
            _shapes_element(offset=20),
            overwrite=True,
        )

    reread = read_zarr(sdata.path)
    assert sdata.shapes["regions"] is previous_live
    assert reread.shapes["regions"].geometry.iloc[0].bounds == (0.0, 0.0, 2.0, 2.0)
    assert not any("__napari_harpy_stage_" in path for path in sdata.elements_paths_on_disk())


def test_write_shapes_element_reports_retained_staging_after_cleanup_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdata = _backed_shapes_sdata(tmp_path)
    real_delete_element = SpatialData.delete_element_from_disk

    def fail_staging_cleanup(
        current_sdata: SpatialData,
        element_name: str | list[str],
    ) -> None:
        if isinstance(element_name, str) and "__napari_harpy_stage_" in element_name:
            raise OSError("injected staging cleanup failure")
        real_delete_element(current_sdata, element_name)

    monkeypatch.setattr(SpatialData, "delete_element_from_disk", fail_staging_cleanup)

    with pytest.raises(RuntimeError, match="temporary Shapes element"):
        write_shapes_element(
            sdata,
            "new_regions",
            _shapes_element(offset=10),
            overwrite=False,
        )

    disk_paths = sdata.elements_paths_on_disk()
    assert "shapes/new_regions" in disk_paths
    assert sum("__napari_harpy_stage_" in path for path in disk_paths) == 1
