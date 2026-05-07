from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pandas as pd
import pytest
from spatialdata import SpatialData, read_zarr
from spatialdata.datasets import blobs
from spatialdata.models import PointsModel
from spatialdata.transformations import Identity

from napari_harpy._transcript_tiles import (
    TRANSCRIPT_TILE_CACHE_SCHEMA_VERSION,
    TranscriptTileCache,
    TranscriptTileLevel,
    _finalize_cache_with_staged_replacement,
    _prepare_cache_output_directory,
    _validate_cache_build_parameters,
    _validate_points_element,
)


def _example_levels() -> tuple[TranscriptTileLevel, ...]:
    return (
        TranscriptTileLevel(level=0, tile_size=4096.0, is_exact=False),
        TranscriptTileLevel(level=1, tile_size=2048.0, is_exact=False),
        TranscriptTileLevel(level=2, tile_size=1024.0, is_exact=True),
    )


def _example_cache(*, levels: tuple[TranscriptTileLevel, ...] | None = None, **overrides: object) -> TranscriptTileCache:
    values = {
        "path": Path("/tmp/example.zarr/points/blobs_points/transcripts_vis"),
        "schema_version": TRANSCRIPT_TILE_CACHE_SCHEMA_VERSION,
        "levels": _example_levels() if levels is None else levels,
        "x_origin": 10.0,
        "y_origin": 20.0,
        "x_min": 10.0,
        "x_max": 2058.0,
        "y_min": 20.0,
        "y_max": 2068.0,
    }
    values.update(overrides)
    return TranscriptTileCache(**values)


def _backed_blobs_with_points(tmp_path: Path, data: dict[str, object]) -> SpatialData:
    sdata = blobs(length=16, n_points=1, n_shapes=1)
    points = PointsModel.parse(
        pd.DataFrame(data),
        coordinates={"x": "x", "y": "y"},
        transformations={"global": Identity()},
    )
    sdata.points["blobs_points"] = points

    path = tmp_path / "blobs.zarr"
    sdata.write(path)
    return read_zarr(path)


def _write_cache_marker_files(path: Path, *, marker: str = "cache") -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "metadata.json").write_text(f'{{"marker": "{marker}"}}\n')
    (path / "manifest.parquet").write_text(f"{marker}\n")


def test_transcript_tile_level_records_metadata_and_derived_file() -> None:
    level = TranscriptTileLevel(level=2, tile_size=1024.0, is_exact=True)

    assert level.level == 2
    assert level.tile_size == 1024.0
    assert level.is_exact is True
    assert level.level_file == "levels/level_2.parquet"


@pytest.mark.parametrize("kwargs", [{"level": -1}, {"tile_size": 0.0}, {"tile_size": -1.0}, {"tile_size": float("inf")}])
def test_transcript_tile_level_rejects_invalid_values(kwargs: dict[str, object]) -> None:
    values = {"level": 0, "tile_size": 1024.0, "is_exact": True}
    values.update(kwargs)

    with pytest.raises(ValueError):
        TranscriptTileLevel(**values)


def test_transcript_tile_cache_records_metadata_and_derived_properties() -> None:
    cache_path = Path("/tmp/example.zarr/points/blobs_points/transcripts_vis")

    cache = _example_cache(path=cache_path)

    assert cache.path == cache_path
    assert cache.metadata_path == cache_path / "metadata.json"
    assert cache.manifest_path == cache_path / "manifest.parquet"
    assert cache.genes_path == cache_path / "genes.parquet"
    assert cache.levels_path == cache_path / "levels"
    assert cache.schema_version == "harpy-transcripts-vis-0.1"
    assert cache.levels == _example_levels()
    assert cache.n_levels == 3
    assert cache.finest_level == 2
    assert cache.leaf_tile_size == 1024.0
    assert cache.x_origin == 10.0
    assert cache.y_origin == 20.0
    assert cache.x_min == 10.0
    assert cache.x_max == 2058.0
    assert cache.y_min == 20.0
    assert cache.y_max == 2068.0


def test_transcript_tile_cache_is_immutable() -> None:
    cache = _example_cache()

    with pytest.raises(FrozenInstanceError):
        cache.x_min = 1.0


@pytest.mark.parametrize(
    ("levels", "match"),
    [
        ((), "at least one level"),
        (
            (
                TranscriptTileLevel(level=1, tile_size=2048.0, is_exact=False),
                TranscriptTileLevel(level=0, tile_size=4096.0, is_exact=True),
            ),
            "sorted",
        ),
        (
            (
                TranscriptTileLevel(level=0, tile_size=4096.0, is_exact=False),
                TranscriptTileLevel(level=0, tile_size=2048.0, is_exact=True),
            ),
            "duplicate",
        ),
        (
            (
                TranscriptTileLevel(level=0, tile_size=4096.0, is_exact=False),
                TranscriptTileLevel(level=2, tile_size=1024.0, is_exact=True),
            ),
            "contiguous",
        ),
        (
            (
                TranscriptTileLevel(level=0, tile_size=4096.0, is_exact=False),
                TranscriptTileLevel(level=1, tile_size=2048.0, is_exact=False),
            ),
            "exactly one exact",
        ),
        (
            (
                TranscriptTileLevel(level=0, tile_size=4096.0, is_exact=True),
                TranscriptTileLevel(level=1, tile_size=2048.0, is_exact=True),
            ),
            "exactly one exact",
        ),
        (
            (
                TranscriptTileLevel(level=0, tile_size=4096.0, is_exact=True),
                TranscriptTileLevel(level=1, tile_size=2048.0, is_exact=False),
            ),
            "finest",
        ),
    ],
)
def test_transcript_tile_cache_rejects_invalid_levels(
    levels: tuple[TranscriptTileLevel, ...], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        _example_cache(levels=levels)


def test_transcript_tile_cache_rejects_unsupported_schema_version() -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        _example_cache(schema_version="harpy-transcripts-vis-unknown")


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"x_origin": float("nan")}, "finite"),
        ({"y_origin": float("inf")}, "finite"),
        ({"x_min": 2.0, "x_max": 1.0}, "x_min <= x_max"),
        ({"y_min": 2.0, "y_max": 1.0}, "y_min <= y_max"),
    ],
)
def test_transcript_tile_cache_rejects_invalid_bounds(overrides: dict[str, float], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _example_cache(**overrides)


def test_validate_points_element_accepts_valid_dataframe_and_resolves_metadata(tmp_path: Path) -> None:
    output_path = tmp_path / "example.zarr" / "points" / "transcripts_vis"
    sdata = _backed_blobs_with_points(
        tmp_path,
        {
            "x": [0.0, 1.0, 2.0],
            "y": [3.0, 4.0, 5.0],
            "gene": ["Actb", "Gapdh", "Malat1"],
            "transcript_id": ["t0", "t1", "t2"],
        },
    )

    validated = _validate_points_element(
        sdata,
        "blobs_points",
        output_path=output_path,
        transcript_id="transcript_id",
    )

    assert validated.points is sdata.points["blobs_points"]
    assert validated.output_path == output_path
    assert validated.element_path == "points/blobs_points"
    assert validated.x == "x"
    assert validated.y == "y"
    assert validated.gene == "gene"
    assert validated.transcript_id == "transcript_id"
    assert validated.uses_internal_row_id is False


def test_validate_cache_build_parameters_normalizes_values() -> None:
    validated = _validate_cache_build_parameters(
        leaf_tile_size=2048,
        n_levels=3,
        max_rows_per_row_group=10,
        coarse_tile_budget=20,
    )

    assert validated.leaf_tile_size == 2048.0
    assert validated.n_levels == 3
    assert validated.max_rows_per_row_group == 10
    assert validated.coarse_tile_budget == 20


def test_prepare_cache_output_directory_creates_missing_output_with_parents(tmp_path: Path) -> None:
    output_path = tmp_path / "nested" / "points" / "transcripts_vis"

    build_path = _prepare_cache_output_directory(output_path)

    assert build_path == output_path
    assert output_path.is_dir()


def test_prepare_cache_output_directory_uses_sibling_temp_for_existing_cache(tmp_path: Path) -> None:
    output_path = tmp_path / "transcripts_vis"
    _write_cache_marker_files(output_path, marker="old")

    build_path = _prepare_cache_output_directory(output_path)

    assert build_path != output_path
    assert build_path.parent == output_path.parent
    assert build_path.name.startswith("transcripts_vis.tmp-")
    assert build_path.is_dir()
    assert (output_path / "metadata.json").read_text() == '{"marker": "old"}\n'


def test_prepare_cache_output_directory_rejects_existing_file(tmp_path: Path) -> None:
    output_path = tmp_path / "transcripts_vis"
    output_path.write_text("not a directory\n")

    with pytest.raises(ValueError, match="not a directory"):
        _prepare_cache_output_directory(output_path)


@pytest.mark.parametrize("files", [(), ("metadata.json",), ("manifest.parquet",)])
def test_prepare_cache_output_directory_rejects_incomplete_existing_directory(
    tmp_path: Path, files: tuple[str, ...]
) -> None:
    output_path = tmp_path / "transcripts_vis"
    output_path.mkdir()
    for file in files:
        (output_path / file).write_text("partial\n")

    with pytest.raises(ValueError, match="metadata.json.*manifest.parquet"):
        _prepare_cache_output_directory(output_path)


def test_finalize_cache_with_staged_replacement_accepts_direct_completed_output(tmp_path: Path) -> None:
    output_path = tmp_path / "transcripts_vis"
    output_path.mkdir()
    (output_path / "manifest.parquet").write_text("done\n")

    finalized_path = _finalize_cache_with_staged_replacement(output_path, output_path)

    assert finalized_path == output_path
    assert (output_path / "manifest.parquet").read_text() == "done\n"


def test_finalize_cache_with_staged_replacement_rejects_missing_build_manifest(tmp_path: Path) -> None:
    output_path = tmp_path / "transcripts_vis"
    build_path = tmp_path / "transcripts_vis.tmp-build"
    _write_cache_marker_files(output_path, marker="old")
    build_path.mkdir()

    with pytest.raises(ValueError, match="manifest.parquet"):
        _finalize_cache_with_staged_replacement(build_path, output_path)

    assert (output_path / "metadata.json").read_text() == '{"marker": "old"}\n'
    assert build_path.exists()


def test_finalize_cache_with_staged_replacement_swaps_temp_cache_and_removes_backup(tmp_path: Path) -> None:
    output_path = tmp_path / "transcripts_vis"
    build_path = tmp_path / "transcripts_vis.tmp-build"
    _write_cache_marker_files(output_path, marker="old")
    _write_cache_marker_files(build_path, marker="new")

    finalized_path = _finalize_cache_with_staged_replacement(build_path, output_path)

    assert finalized_path == output_path
    assert not build_path.exists()
    assert (output_path / "metadata.json").read_text() == '{"marker": "new"}\n'
    assert not list(tmp_path.glob("transcripts_vis.backup-*"))


def test_finalize_cache_with_staged_replacement_restores_old_cache_on_install_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_path = tmp_path / "transcripts_vis"
    build_path = tmp_path / "transcripts_vis.tmp-build"
    _write_cache_marker_files(output_path, marker="old")
    _write_cache_marker_files(build_path, marker="new")
    original_rename = Path.rename

    def fail_build_install(self: Path, target: str | Path) -> Path:
        target_path = Path(target)
        if self == build_path and target_path == output_path:
            raise OSError("simulated install failure")
        return original_rename(self, target)

    monkeypatch.setattr(Path, "rename", fail_build_install)

    with pytest.raises(OSError, match="simulated install failure"):
        _finalize_cache_with_staged_replacement(build_path, output_path)

    assert build_path.exists()
    assert (output_path / "metadata.json").read_text() == '{"marker": "old"}\n'
    assert not list(tmp_path.glob("transcripts_vis.backup-*"))


def test_validate_points_element_uses_default_output_path(backed_sdata_blobs: SpatialData) -> None:
    validated = _validate_points_element(backed_sdata_blobs, "blobs_points", gene="genes")

    assert validated.output_path == Path(backed_sdata_blobs.path) / "points/blobs_points/transcripts_vis"


def test_validate_points_element_records_internal_row_id_fallback(backed_sdata_blobs: SpatialData) -> None:
    validated = _validate_points_element(backed_sdata_blobs, "blobs_points", gene="genes")

    assert validated.transcript_id is None
    assert validated.uses_internal_row_id is True


def test_validate_points_element_rejects_non_pathlike_output_path(backed_sdata_blobs: SpatialData) -> None:
    with pytest.raises(ValueError, match="output_path"):
        _validate_points_element(backed_sdata_blobs, "blobs_points", output_path=1)


@pytest.mark.parametrize("parameter", ["x", "y", "gene", "transcript_id"])
def test_validate_points_element_rejects_non_string_column_names(
    backed_sdata_blobs: SpatialData, parameter: str
) -> None:
    kwargs = {"gene": "genes", parameter: 1}

    with pytest.raises(ValueError, match=parameter):
        _validate_points_element(backed_sdata_blobs, "blobs_points", **kwargs)


@pytest.mark.parametrize("column", ["x", "y", "gene", "transcript_id"])
def test_validate_points_element_rejects_missing_columns(backed_sdata_blobs: SpatialData, column: str) -> None:
    kwargs = {"gene": "genes"}
    if column == "x":
        kwargs["x"] = "missing_x"
    elif column == "y":
        kwargs["y"] = "missing_y"
    elif column == "gene":
        kwargs["gene"] = "missing_gene"
    else:
        kwargs["transcript_id"] = "missing_transcript_id"

    with pytest.raises(ValueError, match=column):
        _validate_points_element(backed_sdata_blobs, "blobs_points", **kwargs)


@pytest.mark.parametrize("column", ["x", "y"])
def test_validate_points_element_rejects_non_numeric_coordinate_metadata(
    backed_sdata_blobs: SpatialData, column: str
) -> None:
    kwargs = {"gene": "genes", column: "genes"}

    with pytest.raises(ValueError, match="numeric"):
        _validate_points_element(backed_sdata_blobs, "blobs_points", **kwargs)


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        ("leaf_tile_size", 0.0),
        ("leaf_tile_size", float("inf")),
        ("leaf_tile_size", True),
        ("max_rows_per_row_group", 0),
        ("max_rows_per_row_group", 1.0),
        ("max_rows_per_row_group", True),
        ("coarse_tile_budget", 0),
        ("coarse_tile_budget", 1.0),
        ("coarse_tile_budget", True),
        ("n_levels", 0),
        ("n_levels", 1.0),
        ("n_levels", True),
    ],
)
def test_validate_cache_build_parameters_rejects_invalid_parameters(parameter: str, value: object) -> None:
    with pytest.raises(ValueError, match=parameter):
        _validate_cache_build_parameters(**{parameter: value})


def test_validate_points_element_rejects_empty_dataframes(tmp_path: Path) -> None:
    sdata = _backed_blobs_with_points(tmp_path, {"x": [], "y": [], "gene": []})

    with pytest.raises(ValueError, match="empty"):
        _validate_points_element(sdata, "blobs_points")


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("x", float("nan")),
        ("x", float("inf")),
        ("y", float("nan")),
        ("y", float("-inf")),
    ],
)
def test_validate_points_element_rejects_invalid_coordinates(tmp_path: Path, column: str, value: float) -> None:
    data = {
        "x": [0.0, 1.0],
        "y": [2.0, 3.0],
        "gene": ["Actb", "Gapdh"],
    }
    data[column][1] = value
    sdata = _backed_blobs_with_points(tmp_path, data)

    with pytest.raises(ValueError, match=column):
        _validate_points_element(sdata, "blobs_points")


@pytest.mark.parametrize("gene_value", [None, "", "   "])
def test_validate_points_element_rejects_invalid_gene_values(tmp_path: Path, gene_value: object) -> None:
    sdata = _backed_blobs_with_points(
        tmp_path,
        {"x": [0.0, 1.0], "y": [2.0, 3.0], "gene": ["Actb", gene_value]},
    )

    with pytest.raises(ValueError, match="gene"):
        _validate_points_element(sdata, "blobs_points")


@pytest.mark.parametrize(
    ("transcript_values", "match"),
    [
        (["t0", None], "missing transcript_id"),
        (["t0", "t0"], "unique transcript_id"),
    ],
)
def test_validate_points_element_rejects_invalid_transcript_ids(
    tmp_path: Path, transcript_values: list[object], match: str
) -> None:
    sdata = _backed_blobs_with_points(
        tmp_path,
        {
            "x": [0.0, 1.0],
            "y": [2.0, 3.0],
            "gene": ["Actb", "Gapdh"],
            "transcript_id": transcript_values,
        },
    )

    with pytest.raises(ValueError, match=match):
        _validate_points_element(sdata, "blobs_points", transcript_id="transcript_id")


def test_validate_points_element_rejects_unbacked_spatialdata(sdata_blobs: SpatialData) -> None:
    with pytest.raises(ValueError, match="backed"):
        _validate_points_element(sdata_blobs, "blobs_points", gene="genes")


def test_validate_points_element_rejects_non_string_points_key(backed_sdata_blobs: SpatialData) -> None:
    with pytest.raises(ValueError, match="points_key"):
        _validate_points_element(backed_sdata_blobs, 1, gene="genes")


def test_validate_points_element_rejects_missing_points_key(backed_sdata_blobs: SpatialData) -> None:
    with pytest.raises(ValueError, match="not available"):
        _validate_points_element(backed_sdata_blobs, "missing_points", gene="genes")


def test_validate_points_element_rejects_unlocated_points_element(
    backed_sdata_blobs: SpatialData, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(backed_sdata_blobs, "locate_element", lambda element: [])

    with pytest.raises(ValueError, match="Could not locate"):
        _validate_points_element(backed_sdata_blobs, "blobs_points", gene="genes")


def test_validate_points_element_rejects_ambiguous_points_element(
    backed_sdata_blobs: SpatialData, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(backed_sdata_blobs, "locate_element", lambda element: ["points/a", "points/b"])

    with pytest.raises(ValueError, match="multiple"):
        _validate_points_element(backed_sdata_blobs, "blobs_points", gene="genes")
