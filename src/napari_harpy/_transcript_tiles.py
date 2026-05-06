from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

TRANSCRIPT_TILE_CACHE_SCHEMA_VERSION = "harpy-transcripts-vis-0.1"


@dataclass(frozen=True)
class TranscriptTileLevel:
    """Metadata for one transcript visualization cache level."""

    level: int
    tile_size: float
    is_exact: bool

    def __post_init__(self) -> None:
        if self.level < 0:
            raise ValueError("Transcript tile level must be non-negative.")
        if not math.isfinite(self.tile_size) or self.tile_size <= 0:
            raise ValueError("Transcript tile level tile_size must be finite and positive.")

    @property
    def level_file(self) -> str:
        return f"levels/level_{self.level}.parquet"


@dataclass(frozen=True)
class TranscriptTileCache:
    """Metadata and root path for a built transcript visualization cache."""

    path: Path
    schema_version: str
    levels: tuple[TranscriptTileLevel, ...]
    x_origin: float
    y_origin: float
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    def __post_init__(self) -> None:
        if self.schema_version != TRANSCRIPT_TILE_CACHE_SCHEMA_VERSION:
            raise ValueError("Unsupported transcript tile cache schema version.")
        if not self.levels:
            raise ValueError("Transcript tile cache must contain at least one level.")

        level_ids = [level.level for level in self.levels]
        if level_ids != sorted(level_ids):
            raise ValueError("Transcript tile cache levels must be sorted by ascending level.")
        if len(set(level_ids)) != len(level_ids):
            raise ValueError("Transcript tile cache levels must not contain duplicate level ids.")
        if level_ids != list(range(level_ids[-1] + 1)):
            raise ValueError("Transcript tile cache levels must be contiguous from 0.")

        exact_levels = [level for level in self.levels if level.is_exact]
        if len(exact_levels) != 1:
            raise ValueError("Expected exactly one exact transcript tile level.")
        if exact_levels[0].level != level_ids[-1]:
            raise ValueError("The exact transcript tile level must be the finest level.")

        bounds_and_origins = [self.x_origin, self.y_origin, self.x_min, self.x_max, self.y_min, self.y_max]
        if not all(math.isfinite(value) for value in bounds_and_origins):
            raise ValueError("Transcript tile cache bounds and origins must be finite.")
        if self.x_min > self.x_max:
            raise ValueError("Transcript tile cache requires x_min <= x_max.")
        if self.y_min > self.y_max:
            raise ValueError("Transcript tile cache requires y_min <= y_max.")

    @property
    def metadata_path(self) -> Path:
        return self.path / "metadata.json"

    @property
    def manifest_path(self) -> Path:
        return self.path / "manifest.parquet"

    @property
    def genes_path(self) -> Path:
        return self.path / "genes.parquet"

    @property
    def levels_path(self) -> Path:
        return self.path / "levels"

    @property
    def n_levels(self) -> int:
        return len(self.levels)

    @property
    def finest_level(self) -> int:
        exact_levels = [level for level in self.levels if level.is_exact]
        if len(exact_levels) != 1:
            raise ValueError("Expected exactly one exact transcript tile level.")
        return exact_levels[0].level

    @property
    def leaf_tile_size(self) -> float:
        exact_levels = [level for level in self.levels if level.is_exact]
        if len(exact_levels) != 1:
            raise ValueError("Expected exactly one exact transcript tile level.")
        return exact_levels[0].tile_size


__all__ = [
    "TRANSCRIPT_TILE_CACHE_SCHEMA_VERSION",
    "TranscriptTileCache",
    "TranscriptTileLevel",
]
