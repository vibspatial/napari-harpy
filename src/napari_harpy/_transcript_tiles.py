from __future__ import annotations

import math
import numbers
import shutil
import uuid
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any

import dask
import dask.dataframe as dd
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

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


@dataclass(frozen=True)
class _ValidatedPointsElement:
    points: dd.DataFrame
    element_path: str
    output_path: Path
    x: str
    y: str
    gene: str
    transcript_id: str | None
    uses_internal_row_id: bool


@dataclass(frozen=True)
class _ValidatedCacheBuildParameters:
    leaf_tile_size: float
    n_levels: int | None
    max_rows_per_row_group: int
    coarse_tile_budget: int


def _validate_points_element(
    sdata: Any,
    points_key: str,
    *,
    output_path: str | PathLike[str] | None = None,
    x: str = "x",
    y: str = "y",
    gene: str = "gene",
    transcript_id: str | None = None,
) -> _ValidatedPointsElement:
    if not hasattr(sdata, "is_backed") or not sdata.is_backed() or getattr(sdata, "path", None) is None:
        raise ValueError("SpatialData must be backed by a zarr store before building a transcript tile cache.")

    if not isinstance(points_key, str):
        raise ValueError("`points_key` must be a string.")

    points_collection = getattr(sdata, "points", None)
    if points_collection is None or points_key not in points_collection:
        raise ValueError(f"Points element `{points_key}` is not available in the SpatialData object.")

    points = points_collection[points_key]
    element_paths = sdata.locate_element(points)
    if not element_paths:
        raise ValueError(f"Could not locate points element `{points_key}` inside the backed SpatialData store.")
    if len(element_paths) > 1:
        raise ValueError(
            f"Points element `{points_key}` resolved to multiple zarr paths: {element_paths}. "
            "A unique points element path is required."
        )
    element_path = element_paths[0]

    if not isinstance(points, dd.DataFrame):
        raise ValueError(f"Points element `{points_key}` must resolve to a dask.dataframe.DataFrame.")

    normalized_output_path = (
        Path(sdata.path) / element_path / "transcripts_vis" if output_path is None else _normalize_output_path(output_path)
    )

    x = _validate_column_name(x, "x")
    y = _validate_column_name(y, "y")
    gene = _validate_column_name(gene, "gene")
    if transcript_id is not None:
        transcript_id = _validate_column_name(transcript_id, "transcript_id")

    _validate_required_columns(points, [x, y, gene])
    if transcript_id is not None:
        _validate_required_columns(points, [transcript_id])

    _validate_numeric_column(points, x)
    _validate_numeric_column(points, y)

    row_count, invalid_x, invalid_y, missing_gene, empty_gene, *transcript_checks = dask.compute(
        points.map_partitions(len, meta=("row_count", "int64")).sum(),
        points[x].map_partitions(_count_nonfinite_values, meta=("invalid_x", "int64")).sum(),
        points[y].map_partitions(_count_nonfinite_values, meta=("invalid_y", "int64")).sum(),
        points[gene].map_partitions(_count_missing_values, meta=("missing_gene", "int64")).sum(),
        points[gene].map_partitions(_count_stripped_empty_values, meta=("empty_gene", "int64")).sum(),
        *(
            (
                points[transcript_id]
                .map_partitions(_count_missing_values, meta=("missing_transcript_id", "int64"))
                .sum(),
                points[transcript_id].nunique(dropna=True),
            )
            if transcript_id is not None
            else ()
        ),
    )

    row_count = int(row_count)
    if row_count == 0:
        raise ValueError("`points` must not be empty.")
    if int(invalid_x) > 0:
        raise ValueError(f"Column `{x}` contains missing, NaN, or infinite coordinate values.")
    if int(invalid_y) > 0:
        raise ValueError(f"Column `{y}` contains missing, NaN, or infinite coordinate values.")
    if int(missing_gene) > 0:
        raise ValueError(f"Column `{gene}` contains missing gene values.")
    if int(empty_gene) > 0:
        raise ValueError(f"Column `{gene}` contains empty gene labels after stripping whitespace.")

    if transcript_id is not None:
        missing_transcript_id, unique_transcript_id_count = transcript_checks
        if int(missing_transcript_id) > 0:
            raise ValueError(f"Column `{transcript_id}` contains missing transcript_id values.")
        if int(unique_transcript_id_count) != row_count:
            raise ValueError(f"Column `{transcript_id}` must contain unique transcript_id values.")

    return _ValidatedPointsElement(
        points=points,
        element_path=element_path,
        output_path=normalized_output_path,
        x=x,
        y=y,
        gene=gene,
        transcript_id=transcript_id,
        uses_internal_row_id=transcript_id is None,
    )


def _validate_cache_build_parameters(
    *,
    leaf_tile_size: float = 1024.0,
    n_levels: int | None = None,
    max_rows_per_row_group: int = 50_000,
    coarse_tile_budget: int = 50_000,
) -> _ValidatedCacheBuildParameters:
    normalized_leaf_tile_size = _validate_positive_finite_number(leaf_tile_size, "leaf_tile_size")
    normalized_max_rows_per_row_group = _validate_positive_integer(
        max_rows_per_row_group, "max_rows_per_row_group"
    )
    normalized_coarse_tile_budget = _validate_positive_integer(coarse_tile_budget, "coarse_tile_budget")
    normalized_n_levels = None if n_levels is None else _validate_positive_integer(n_levels, "n_levels")

    return _ValidatedCacheBuildParameters(
        leaf_tile_size=normalized_leaf_tile_size,
        n_levels=normalized_n_levels,
        max_rows_per_row_group=normalized_max_rows_per_row_group,
        coarse_tile_budget=normalized_coarse_tile_budget,
    )


def _prepare_cache_output_directory(output_path: str | PathLike[str]) -> Path:
    output_path = Path(output_path)
    if not output_path.exists():
        output_path.mkdir(parents=True)
        return output_path

    if not output_path.is_dir():
        raise ValueError("Transcript tile cache output path exists but is not a directory.")

    metadata_path = output_path / "metadata.json"
    manifest_path = output_path / "manifest.parquet"
    if not metadata_path.is_file() or not manifest_path.is_file():
        raise ValueError(
            "Transcript tile cache output path already exists but does not contain both "
            "`metadata.json` and `manifest.parquet`."
        )

    build_path = _unique_sibling_path(output_path, "tmp")
    build_path.mkdir()
    return build_path


def _finalize_cache_with_staged_replacement(build_path: str | PathLike[str], output_path: str | PathLike[str]) -> Path:
    build_path = Path(build_path)
    output_path = Path(output_path)

    if not (build_path / "manifest.parquet").is_file():
        raise ValueError("Built transcript tile cache is missing `manifest.parquet`.")

    if build_path == output_path:
        return output_path

    if output_path.exists():
        if not output_path.is_dir():
            raise ValueError("Transcript tile cache output path exists but is not a directory.")
        if not (output_path / "metadata.json").is_file() or not (output_path / "manifest.parquet").is_file():
            raise ValueError(
                "Transcript tile cache output path already exists but does not contain both "
                "`metadata.json` and `manifest.parquet`."
            )

    backup_path: Path | None = None
    try:
        if output_path.exists():
            backup_path = _unique_sibling_path(output_path, "backup")
            output_path.rename(backup_path)
        build_path.rename(output_path)
    except Exception:
        if backup_path is not None and backup_path.exists() and not output_path.exists():
            try:
                backup_path.rename(output_path)
            except OSError:
                pass
        raise

    if backup_path is not None and backup_path.exists():
        shutil.rmtree(backup_path)

    return output_path


def _unique_sibling_path(path: Path, label: str) -> Path:
    while True:
        candidate = path.with_name(f"{path.name}.{label}-{uuid.uuid4().hex}")
        if not candidate.exists():
            return candidate


def _normalize_output_path(output_path: str | PathLike[str]) -> Path:
    try:
        return Path(output_path)
    except TypeError as exc:
        raise ValueError("`output_path` must be path-like.") from exc


def _validate_column_name(column: Any, parameter_name: str) -> str:
    if not isinstance(column, str):
        raise ValueError(f"`{parameter_name}` must be a string.")
    return column


def _validate_required_columns(points: dd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in points.columns]
    if missing:
        missing_columns = ", ".join(f"`{column}`" for column in missing)
        raise ValueError(f"`points` is missing required column(s): {missing_columns}.")


def _validate_numeric_column(points: dd.DataFrame, column: str) -> None:
    if not is_numeric_dtype(points._meta[column].dtype):
        raise ValueError(f"Column `{column}` must be numeric.")


def _validate_positive_finite_number(value: Any, parameter_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"`{parameter_name}` must be a finite positive number.")
    normalized_value = float(value)
    if not math.isfinite(normalized_value) or normalized_value <= 0:
        raise ValueError(f"`{parameter_name}` must be a finite positive number.")
    return normalized_value


def _validate_positive_integer(value: Any, parameter_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError(f"`{parameter_name}` must be a positive integer and not a boolean.")
    normalized_value = int(value)
    if normalized_value <= 0:
        raise ValueError(f"`{parameter_name}` must be a positive integer.")
    return normalized_value


def _count_nonfinite_values(series: pd.Series) -> int:
    values = series.to_numpy(dtype="float64", na_value=np.nan)
    return int((~np.isfinite(values)).sum())


def _count_missing_values(series: pd.Series) -> int:
    return int(series.isna().sum())


def _count_stripped_empty_values(series: pd.Series) -> int:
    non_missing = series.dropna()
    return int(non_missing.astype(str).str.strip().eq("").sum())


__all__ = [
    "TRANSCRIPT_TILE_CACHE_SCHEMA_VERSION",
    "TranscriptTileCache",
    "TranscriptTileLevel",
]
