from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from napari_harpy._transcript_tiles import (
    TRANSCRIPT_TILE_CACHE_SCHEMA_VERSION,
    TranscriptTileCache,
    TranscriptTileLevel,
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
