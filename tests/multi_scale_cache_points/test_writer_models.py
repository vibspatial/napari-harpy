from __future__ import annotations

import pytest

from napari_harpy.core.multi_scale_cache_points.writer.models import (
    _ExactLevelWriterConfig,
    _IntermediateTileValueCountFile,
    _ManifestRow,
)


def _manifest_row(**overrides: object) -> _ManifestRow:
    values: dict[str, object] = {
        "level": 0,
        "level_file": "levels/level_0/bucket-000.parquet",
        "tile_x": 1,
        "tile_y": 2,
        "n_points": 3,
        "row_group": 4,
        "tile_shard": 0,
    }
    values.update(overrides)
    return _ManifestRow(**values)  # type: ignore[arg-type]


def _intermediate_tile_value_count_file(**overrides: object) -> _IntermediateTileValueCountFile:
    values: dict[str, object] = {
        "level": 0,
        "relative_path": "intermediate_tile_value_counts/level_0/bucket-000.parquet",
        "row_count": 4,
    }
    values.update(overrides)
    return _IntermediateTileValueCountFile(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("bucket_count", True),
        ("max_rows_per_row_group", -1),
        ("dask_worker_count", 1.5),
    ],
)
def test_exact_writer_config_requires_positive_integer_values(field_name: str, value: object) -> None:
    values: dict[str, object] = {
        "bucket_count": 128,
        "max_rows_per_row_group": 1_000_000,
        "dask_worker_count": 2,
    }
    values[field_name] = value

    with pytest.raises(ValueError, match=field_name):
        _ExactLevelWriterConfig(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "level_file",
    ["", "/levels/level_0/bucket.parquet", "../bucket.parquet", "levels/../bucket.parquet"],
)
def test_manifest_row_requires_cache_relative_normalized_path(level_file: str) -> None:
    with pytest.raises(ValueError, match="level_file"):
        _manifest_row(level_file=level_file)


def test_manifest_row_requires_file_inside_its_level_directory() -> None:
    with pytest.raises(ValueError, match="levels/level_2"):
        _manifest_row(level=2, level_file="levels/level_1/bucket-000.parquet")


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("level", 2**15),
        ("tile_x", 2**32),
        ("n_points", 0),
        ("row_group", -1),
        ("tile_shard", 2**31),
    ],
)
def test_manifest_row_enforces_serialized_integer_ranges(field_name: str, value: int) -> None:
    with pytest.raises(ValueError, match=field_name):
        _manifest_row(**{field_name: value})


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("level", -1),
        ("level", 2**15),
        ("row_count", True),
        ("row_count", 0),
        ("row_count", 2**63),
    ],
)
def test_intermediate_tile_value_count_file_enforces_serialized_integer_ranges(
    field_name: str,
    value: object,
) -> None:
    with pytest.raises(ValueError, match=field_name):
        _intermediate_tile_value_count_file(**{field_name: value})


@pytest.mark.parametrize(
    "relative_path",
    [
        "",
        "/intermediate_tile_value_counts/counts.parquet",
        "../counts.parquet",
        "intermediate_tile_value_counts/../counts.parquet",
    ],
)
def test_intermediate_tile_value_count_file_requires_cache_relative_normalized_path(relative_path: str) -> None:
    with pytest.raises(ValueError, match="relative_path"):
        _intermediate_tile_value_count_file(relative_path=relative_path)
