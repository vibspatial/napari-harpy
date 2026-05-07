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
import pyarrow as pa
import pyarrow.parquet as pq
from pandas.api.types import is_numeric_dtype

TRANSCRIPT_TILE_CACHE_SCHEMA_VERSION = "harpy-transcripts-vis-0.1"
_INTERNAL_X_COLUMN = "x"
_INTERNAL_Y_COLUMN = "y"
_GENE_ID_COLUMN = "gene_id"
_TRANSCRIPT_ID_COLUMN = "transcript_id"
_RESERVED_SOURCE_COLUMNS = frozenset({_GENE_ID_COLUMN})
_GENE_ID_DTYPE = np.dtype("uint32")
_N_TRANSCRIPTS_DTYPE = np.dtype("uint64")
_TILE_INDEX_DTYPE = np.dtype("uint32")
_RELATIVE_COORDINATE_DTYPE = np.dtype("float32")
_MAX_N_UINT32_GENE_IDS = int(np.iinfo(_GENE_ID_DTYPE).max) + 1
_MAX_UINT32_TILE_INDEX = int(np.iinfo(_TILE_INDEX_DTYPE).max)
_GENE_TABLE_SCHEMA = pa.schema(
    [
        (_GENE_ID_COLUMN, pa.uint32()),
        ("gene", pa.string()),
        ("n_transcripts", pa.uint64()),
    ]
)


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
class TranscriptTileCacheBuildParameters:
    """Build-time provenance for one transcript visualization cache."""

    max_rows_per_row_group: int
    coarse_tile_budget: int
    x: str
    y: str
    gene: str
    transcript_id: str | None

    def __post_init__(self) -> None:
        for name, value in [
            ("max_rows_per_row_group", self.max_rows_per_row_group),
            ("coarse_tile_budget", self.coarse_tile_budget),
        ]:
            if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value <= 0:
                raise ValueError(f"Transcript tile cache build parameter `{name}` must be a positive integer.")

        for name, value in [("x", self.x), ("y", self.y), ("gene", self.gene)]:
            if not isinstance(value, str):
                raise ValueError(f"Transcript tile cache build parameter `{name}` must be a string.")

        if self.transcript_id is not None and not isinstance(self.transcript_id, str):
            raise ValueError("Transcript tile cache build parameter `transcript_id` must be a string or None.")


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
    build_parameters: TranscriptTileCacheBuildParameters | None = None

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
        if self.build_parameters is not None and not isinstance(
            self.build_parameters, TranscriptTileCacheBuildParameters
        ):
            raise ValueError("Transcript tile cache build_parameters must be a TranscriptTileCacheBuildParameters.")

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

    @property
    def uses_internal_row_id(self) -> bool:
        return self.transcript_id is None


def _validate_points_element(
    sdata: Any,
    points_name: str,
    *,
    output_path: str | PathLike[str] | None = None,
    x: str = "x",
    y: str = "y",
    gene: str = "gene",
    transcript_id: str | None = None,
) -> _ValidatedPointsElement:
    if not hasattr(sdata, "is_backed") or not sdata.is_backed() or getattr(sdata, "path", None) is None:
        raise ValueError("SpatialData must be backed by a zarr store before building a transcript tile cache.")

    if not isinstance(points_name, str):
        raise ValueError("`points_name` must be a string.")

    points_collection = getattr(sdata, "points", None)
    if points_collection is None or points_name not in points_collection:
        raise ValueError(f"Points element `{points_name}` is not available in the SpatialData object.")

    points = points_collection[points_name]
    element_paths = sdata.locate_element(points)
    if not element_paths:
        raise ValueError(f"Could not locate points element `{points_name}` inside the backed SpatialData store.")
    if len(element_paths) > 1:
        raise ValueError(
            f"Points element `{points_name}` resolved to multiple zarr paths: {element_paths}. "
            "A unique points element path is required."
        )
    element_path = element_paths[0]

    if not isinstance(points, dd.DataFrame):
        raise ValueError(f"Points element `{points_name}` must resolve to a dask.dataframe.DataFrame.")

    normalized_output_path = (
        Path(sdata.path) / element_path / "transcripts_vis"
        if output_path is None
        else _normalize_output_path(output_path)
    )

    x = _validate_column_name(x, "x")
    y = _validate_column_name(y, "y")
    gene = _validate_column_name(gene, "gene")
    if transcript_id is not None:
        transcript_id = _validate_column_name(transcript_id, "transcript_id")

    _validate_required_columns(points, [x, y, gene])
    if transcript_id is not None:
        _validate_required_columns(points, [transcript_id])
    _validate_no_reserved_source_columns(points)

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


def _compute_transcript_tile_cache_metadata(
    points_element: _ValidatedPointsElement,
    *,
    leaf_tile_size: float = 1024.0,
    n_levels: int | None = None,
    max_rows_per_row_group: int = 50_000,
    coarse_tile_budget: int = 50_000,
) -> TranscriptTileCache:
    leaf_tile_size = _validate_positive_finite_number(leaf_tile_size, "leaf_tile_size")
    max_rows_per_row_group = _validate_positive_integer(max_rows_per_row_group, "max_rows_per_row_group")
    coarse_tile_budget = _validate_positive_integer(coarse_tile_budget, "coarse_tile_budget")
    n_levels = None if n_levels is None else _validate_positive_integer(n_levels, "n_levels")

    points = points_element.points
    x = points_element.x
    y = points_element.y

    x_min, x_max, y_min, y_max = dask.compute(points[x].min(), points[x].max(), points[y].min(), points[y].max())
    x_min = float(x_min)
    x_max = float(x_max)
    y_min = float(y_min)
    y_max = float(y_max)
    x_origin = x_min
    y_origin = y_min

    bounds_and_origins = [x_origin, y_origin, x_min, x_max, y_min, y_max]
    if not all(math.isfinite(value) for value in bounds_and_origins):
        raise ValueError("Transcript tile cache bounds and origins must be finite.")
    if x_min > x_max:
        raise ValueError("Transcript tile cache requires x_min <= x_max.")
    if y_min > y_max:
        raise ValueError("Transcript tile cache requires y_min <= y_max.")

    if n_levels is None:
        extent = max(x_max - x_min, y_max - y_min)
        n_levels = 1 if extent <= leaf_tile_size else math.ceil(math.log2(extent / leaf_tile_size)) + 1

    finest_level = n_levels - 1
    levels = tuple(
        TranscriptTileLevel(
            level=level,
            tile_size=leaf_tile_size * 2 ** (finest_level - level),
            is_exact=level == finest_level,
        )
        for level in range(n_levels)
    )

    return TranscriptTileCache(
        path=points_element.output_path,
        schema_version=TRANSCRIPT_TILE_CACHE_SCHEMA_VERSION,
        levels=levels,
        x_origin=x_origin,
        y_origin=y_origin,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        build_parameters=TranscriptTileCacheBuildParameters(
            max_rows_per_row_group=max_rows_per_row_group,
            coarse_tile_budget=coarse_tile_budget,
            x=points_element.x,
            y=points_element.y,
            gene=points_element.gene,
            transcript_id=points_element.transcript_id,
        ),
    )


def _build_gene_table(points_element: _ValidatedPointsElement) -> pd.DataFrame:
    normalized_genes = points_element.points[points_element.gene].map_partitions(
        _normalize_gene_values,
        meta=(points_element.gene, "string"),
    )
    gene_counts = normalized_genes.value_counts(sort=False).compute()
    gene_counts = gene_counts.groupby(level=0).sum().sort_index()

    n_genes = len(gene_counts)
    if n_genes > _MAX_N_UINT32_GENE_IDS:
        raise ValueError("Number of unique genes exceeds the maximum representable uint32 gene_id value.")

    return pd.DataFrame(
        {
            _GENE_ID_COLUMN: np.arange(n_genes, dtype=_GENE_ID_DTYPE),
            "gene": pd.Series(gene_counts.index.to_numpy(dtype=object), dtype="string"),
            "n_transcripts": gene_counts.to_numpy(dtype=_N_TRANSCRIPTS_DTYPE),
        }
    )


def _write_genes_parquet(gene_table: pd.DataFrame, genes_path: str | PathLike[str]) -> None:
    genes_path = Path(genes_path)
    genes_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(gene_table, schema=_GENE_TABLE_SCHEMA, preserve_index=False)
    pq.write_table(table, genes_path)


def _prepare_gene_encoded_points(
    points_element: _ValidatedPointsElement,
    gene_table: pd.DataFrame,
) -> dd.DataFrame:
    """Return the internal working dataframe with normalized coordinates and `gene_id`.

    The returned dataframe drops the original gene string, renames the selected
    coordinate columns to the internal `x`/`y` names, adds `gene_id`, and carries
    through the optional validated transcript id for later sampling/writing.
    """
    _validate_no_reserved_source_columns(points_element.points)
    gene_to_id = {
        str(gene): int(gene_id) for gene, gene_id in zip(gene_table["gene"], gene_table[_GENE_ID_COLUMN], strict=True)
    }
    meta = _gene_encoded_points_meta(points_element)

    return points_element.points.map_partitions(
        _encode_gene_partition,
        x=points_element.x,
        y=points_element.y,
        gene=points_element.gene,
        transcript_id=points_element.transcript_id,
        gene_to_id=gene_to_id,
        meta=meta,
    )


def _encode_gene_partition(
    partition: pd.DataFrame,
    *,
    x: str,
    y: str,
    gene: str,
    transcript_id: str | None,
    gene_to_id: dict[str, int],
) -> pd.DataFrame:
    gene_ids = _normalize_gene_values(partition[gene]).map(gene_to_id)
    if gene_ids.isna().any():
        raise ValueError("Encountered a gene label that is missing from the gene dictionary.")

    encoded = pd.DataFrame(
        {
            _INTERNAL_X_COLUMN: partition[x].to_numpy(),
            _INTERNAL_Y_COLUMN: partition[y].to_numpy(),
            _GENE_ID_COLUMN: gene_ids.to_numpy(dtype=_GENE_ID_DTYPE),
        },
        index=partition.index,
    )
    if transcript_id is not None:
        encoded[_TRANSCRIPT_ID_COLUMN] = partition[transcript_id].to_numpy()

    return encoded


def _gene_encoded_points_meta(points_element: _ValidatedPointsElement) -> pd.DataFrame:
    meta = pd.DataFrame(
        {
            _INTERNAL_X_COLUMN: pd.Series(dtype=points_element.points._meta[points_element.x].dtype),
            _INTERNAL_Y_COLUMN: pd.Series(dtype=points_element.points._meta[points_element.y].dtype),
            _GENE_ID_COLUMN: pd.Series(dtype=_GENE_ID_DTYPE),
        }
    )
    if points_element.transcript_id is not None:
        meta[_TRANSCRIPT_ID_COLUMN] = pd.Series(dtype=points_element.points._meta[points_element.transcript_id].dtype)
    return meta


def _annotate_tiles_for_level(points: dd.DataFrame, cache: TranscriptTileCache, level: int) -> dd.DataFrame:
    """Annotate working rows with tile membership and tile-local coordinates.

    `points` must already use the internal `x`/`y` coordinate columns and contain
    `gene_id`. Row identity columns are preserved for later sampling/writing.
    """
    _validate_required_columns(points, [_INTERNAL_X_COLUMN, _INTERNAL_Y_COLUMN, _GENE_ID_COLUMN])
    level_record = _get_transcript_tile_level(cache, level)
    meta = pd.DataFrame(
        {
            "tile_id": pd.Series(dtype="string"),
            "tile_x": pd.Series(dtype=_TILE_INDEX_DTYPE),
            "tile_y": pd.Series(dtype=_TILE_INDEX_DTYPE),
            "x_rel": pd.Series(dtype=_RELATIVE_COORDINATE_DTYPE),
            "y_rel": pd.Series(dtype=_RELATIVE_COORDINATE_DTYPE),
            _GENE_ID_COLUMN: pd.Series(dtype=_GENE_ID_DTYPE),
        }
    )
    for column in _row_identity_columns(points._meta):
        meta[column] = pd.Series(dtype=points._meta[column].dtype)

    return points.map_partitions(
        _annotate_tile_partition,
        level=level_record.level,
        tile_size=level_record.tile_size,
        x_origin=cache.x_origin,
        y_origin=cache.y_origin,
        meta=meta,
    )


def _annotate_tile_partition(
    partition: pd.DataFrame,
    *,
    level: int,
    tile_size: float,
    x_origin: float,
    y_origin: float,
) -> pd.DataFrame:
    x_values = partition[_INTERNAL_X_COLUMN].to_numpy(dtype="float64", na_value=np.nan)
    y_values = partition[_INTERNAL_Y_COLUMN].to_numpy(dtype="float64", na_value=np.nan)
    tile_x_float = np.floor((x_values - x_origin) / tile_size)
    tile_y_float = np.floor((y_values - y_origin) / tile_size)
    _validate_tile_indices(tile_x_float, "tile_x")
    _validate_tile_indices(tile_y_float, "tile_y")

    tile_x = tile_x_float.astype(_TILE_INDEX_DTYPE)
    tile_y = tile_y_float.astype(_TILE_INDEX_DTYPE)
    x_rel = x_values - x_origin - tile_x.astype("float64") * tile_size
    y_rel = y_values - y_origin - tile_y.astype("float64") * tile_size
    tile_id = pd.Series(
        [f"{level}/{int(x_index)}/{int(y_index)}" for x_index, y_index in zip(tile_x, tile_y, strict=True)],
        dtype="string",
        index=partition.index,
    )

    annotated = pd.DataFrame(
        {
            "tile_id": tile_id,
            "tile_x": tile_x,
            "tile_y": tile_y,
            "x_rel": x_rel.astype(_RELATIVE_COORDINATE_DTYPE),
            "y_rel": y_rel.astype(_RELATIVE_COORDINATE_DTYPE),
            _GENE_ID_COLUMN: partition[_GENE_ID_COLUMN].to_numpy(dtype=_GENE_ID_DTYPE),
        },
        index=partition.index,
    )

    for column in _row_identity_columns(partition):
        annotated[column] = partition[column].to_numpy()

    return annotated


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


def _normalize_gene_values(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().astype("string")


def _validate_no_reserved_source_columns(points: dd.DataFrame) -> None:
    reserved_columns = sorted(column for column in _RESERVED_SOURCE_COLUMNS if column in points.columns)
    if reserved_columns:
        formatted_columns = ", ".join(f"`{column}`" for column in reserved_columns)
        raise ValueError(
            "`points` contains reserved transcript tile cache internal column(s): "
            f"{formatted_columns}. Rename these source columns before building the cache."
        )


def _get_transcript_tile_level(cache: TranscriptTileCache, level: int) -> TranscriptTileLevel:
    if isinstance(level, bool) or not isinstance(level, numbers.Integral):
        raise ValueError("Transcript tile annotation level must be an integer.")
    normalized_level = int(level)
    for level_record in cache.levels:
        if level_record.level == normalized_level:
            return level_record
    raise ValueError(f"Transcript tile annotation level `{normalized_level}` is not available in the cache.")


def _validate_tile_indices(tile_indices: np.ndarray, column: str) -> None:
    if not np.isfinite(tile_indices).all():
        raise ValueError(f"Computed `{column}` values must be finite.")
    if (tile_indices < 0).any():
        raise ValueError(f"Computed `{column}` values must be non-negative.")
    if (tile_indices > _MAX_UINT32_TILE_INDEX).any():
        raise ValueError(f"Computed `{column}` values exceed the maximum representable uint32 value.")


def _row_identity_columns(points: pd.DataFrame) -> list[str]:
    return [column for column in points.columns if column not in {_INTERNAL_X_COLUMN, _INTERNAL_Y_COLUMN, _GENE_ID_COLUMN}]


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
    "TranscriptTileCacheBuildParameters",
    "TranscriptTileLevel",
]
