from __future__ import annotations

import numbers
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import dask
import dask.dataframe as dd
import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_integer_dtype, is_numeric_dtype, is_object_dtype, is_string_dtype

TRANSCRIPT_VALUE_INDEX_SCHEMA_VERSION = "harpy-transcripts-value-index-0.1"
DEFAULT_X = "x"
DEFAULT_Y = "y"
DEFAULT_INDEX_COLUMN = "gene"
DEFAULT_RENDER_POINT_BUDGET = 100_000
DEFAULT_RANDOM_STATE = 42

VALUE_ID_COLUMN = "value_id"
VALUE_COLUMN = "value"
N_POINTS_COLUMN = "n_points"
VALUE_VOCABULARY_COLUMNS = (VALUE_ID_COLUMN, VALUE_COLUMN, N_POINTS_COLUMN)
VALUE_ID_DTYPE = np.dtype("uint32")
N_POINTS_DTYPE = np.dtype("uint64")
COORDINATE_DTYPE = np.dtype("float32")


@dataclass(frozen=True)
class _ValidatedPointsElement:
    points: dd.DataFrame
    points_name: str
    source_path: Path | None
    source_n_points: int
    x: str
    y: str
    index_column: str
    transcript_id: str | None

    @property
    def is_backed(self) -> bool:
        return self.source_path is not None

    @property
    def element_path(self) -> str | None:
        if self.source_path is None:
            return None
        return f"points/{self.points_name}"


@dataclass(frozen=True, kw_only=True)
class TranscriptValueVocabulary:
    """In-memory value vocabulary for a selected transcript points column.

    Parameters
    ----------
    values
        DataFrame with exactly `value_id`, `value`, and `n_points` columns.
        `value_id` must be `uint32`, `value` must contain unique normalized
        strings, and `n_points` must be `uint64`.
    index_column
        Name of the source points column used to produce the value vocabulary,
        for example `gene`, `target`, or `probe`.
    total_count
        Total number of source points represented by the vocabulary. Must equal
        `values["n_points"].sum()`.
    """

    values: pd.DataFrame
    index_column: str
    total_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.values, pd.DataFrame):
            raise ValueError("Transcript value vocabulary `values` must be a pandas DataFrame.")
        if tuple(self.values.columns) != VALUE_VOCABULARY_COLUMNS:
            raise ValueError(
                "Transcript value vocabulary `values` must contain exactly `value_id`, `value`, and `n_points` columns."
            )
        if not isinstance(self.index_column, str) or not self.index_column:
            raise ValueError("Transcript value vocabulary `index_column` must be a non-empty string.")
        _validate_non_negative_integer("total_count", self.total_count)

        if self.values[VALUE_ID_COLUMN].dtype != VALUE_ID_DTYPE:
            raise ValueError("Transcript value vocabulary `value_id` column must have dtype uint32.")
        if self.values[N_POINTS_COLUMN].dtype != N_POINTS_DTYPE:
            raise ValueError("Transcript value vocabulary `n_points` column must have dtype uint64.")
        if self.values[VALUE_ID_COLUMN].isna().any():
            raise ValueError("Transcript value vocabulary `value_id` column must not contain missing values.")
        if self.values[VALUE_COLUMN].isna().any():
            raise ValueError("Transcript value vocabulary `value` column must not contain missing values.")
        if self.values[N_POINTS_COLUMN].isna().any():
            raise ValueError("Transcript value vocabulary `n_points` column must not contain missing values.")
        if self.values[VALUE_ID_COLUMN].duplicated().any():
            raise ValueError("Transcript value vocabulary `value_id` values must be unique.")
        if self.values[VALUE_COLUMN].duplicated().any():
            raise ValueError("Transcript value vocabulary `value` values must be unique.")
        if not self.values[VALUE_COLUMN].map(lambda value: isinstance(value, str)).all():
            raise ValueError("Transcript value vocabulary `value` column must contain strings.")

        observed_total = int(self.values[N_POINTS_COLUMN].sum())
        if observed_total != self.total_count:
            raise ValueError("Transcript value vocabulary `total_count` must equal the sum of `n_points`.")


@dataclass(frozen=True, kw_only=True)
class TranscriptValueSelection:
    """Selected transcript points ready to be displayed as one napari Points layer.

    Parameters
    ----------
    coordinates
        `Nx2` `float32` coordinate array in napari display order `y, x`.
    features
        DataFrame with one row per coordinate. It must contain exactly the
        configured `index_column` as a pandas categorical column and `value_id`
        as an integer column.
    index_column
        Name of the source points column used for value selection and stored in
        `features`.
    selected_values
        Normalized selected values represented by this selection.
    selected_value_ids
        Integer value ids corresponding one-to-one with `selected_values`.
    total_count
        Number of selected source points before render-budget sampling.
    render_point_budget
        Maximum number of points allowed in the returned selection.
    is_sampled
        Whether the result is a sampled preview rather than an exact selection.
    warning
        User-visible warning for sampled selections, or `None` for exact
        selections.
    """

    coordinates: np.ndarray
    features: pd.DataFrame
    index_column: str
    selected_values: tuple[str, ...]
    selected_value_ids: tuple[int, ...]
    total_count: int
    render_point_budget: int
    is_sampled: bool
    warning: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.coordinates, np.ndarray):
            raise ValueError("Transcript value selection `coordinates` must be a numpy array.")
        if self.coordinates.ndim != 2 or self.coordinates.shape[1] != 2:
            raise ValueError("Transcript value selection `coordinates` must be an Nx2 array.")
        if self.coordinates.dtype != COORDINATE_DTYPE:
            raise ValueError("Transcript value selection `coordinates` must have dtype float32.")
        if not isinstance(self.features, pd.DataFrame):
            raise ValueError("Transcript value selection `features` must be a pandas DataFrame.")
        if not isinstance(self.index_column, str) or not self.index_column:
            raise ValueError("Transcript value selection `index_column` must be a non-empty string.")
        if self.index_column == VALUE_ID_COLUMN:
            raise ValueError("Transcript value selection `index_column` must not be `value_id`.")
        if set(self.features.columns) != {self.index_column, VALUE_ID_COLUMN}:
            raise ValueError(
                "Transcript value selection `features` must contain exactly the configured index column and `value_id`."
            )
        if not isinstance(self.features[self.index_column].dtype, pd.CategoricalDtype):
            raise ValueError("Transcript value selection index feature must be categorical.")
        if is_bool_dtype(self.features[VALUE_ID_COLUMN].dtype) or not is_integer_dtype(
            self.features[VALUE_ID_COLUMN].dtype
        ):
            raise ValueError("Transcript value selection `value_id` feature must be integer typed.")
        if not isinstance(self.selected_values, tuple) or not all(
            isinstance(value, str) for value in self.selected_values
        ):
            raise ValueError("Transcript value selection `selected_values` must be a tuple of strings.")
        if not isinstance(self.selected_value_ids, tuple) or not all(
            _is_non_negative_integral(value) for value in self.selected_value_ids
        ):
            raise ValueError(
                "Transcript value selection `selected_value_ids` must be a tuple of non-negative integers."
            )
        if len(self.selected_values) != len(self.selected_value_ids):
            raise ValueError("Transcript value selection selected values and ids must have the same length.")
        if len(set(self.selected_values)) != len(self.selected_values):
            raise ValueError("Transcript value selection `selected_values` must not contain duplicates.")
        if len({int(value_id) for value_id in self.selected_value_ids}) != len(self.selected_value_ids):
            raise ValueError("Transcript value selection `selected_value_ids` must not contain duplicates.")

        _validate_non_negative_integer("total_count", self.total_count)
        _validate_positive_integer("render_point_budget", self.render_point_budget)
        if not isinstance(self.is_sampled, bool):
            raise ValueError("Transcript value selection `is_sampled` must be a boolean.")
        if self.warning is not None and not isinstance(self.warning, str):
            raise ValueError("Transcript value selection `warning` must be a string or None.")

        if len(self.features) != self.loaded_count:
            raise ValueError("Transcript value selection loaded rows must match coordinates and features.")
        if self.loaded_count > self.total_count:
            raise ValueError("Transcript value selection `loaded_count` must be <= `total_count`.")
        if self.loaded_count > self.render_point_budget:
            raise ValueError("Transcript value selection `loaded_count` must be <= `render_point_budget`.")
        if not self.is_sampled and self.loaded_count != self.total_count:
            raise ValueError("Exact transcript value selections must load all selected points.")
        if self.is_sampled and (self.warning is None or not self.warning):
            raise ValueError("Sampled transcript value selections must include a warning.")
        if not self.is_sampled and self.warning is not None:
            raise ValueError("Exact transcript value selections must not include a warning.")

    @property
    def loaded_count(self) -> int:
        """Number of points loaded into `coordinates` and `features`."""
        return int(len(self.coordinates))


def _is_non_negative_integral(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, numbers.Integral) and value >= 0


def _validate_non_negative_integer(name: str, value: object) -> None:
    if not _is_non_negative_integral(value):
        raise ValueError(f"`{name}` must be a non-negative integer.")


def _validate_positive_integer(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value <= 0:
        raise ValueError(f"`{name}` must be a positive integer.")


def validate_points_element_for_value_selection(
    sdata: Any,
    points_name: str,
    *,
    x: str = DEFAULT_X,
    y: str = DEFAULT_Y,
    index_column: str = DEFAULT_INDEX_COLUMN,
    transcript_id: str | None = None,
) -> _ValidatedPointsElement:
    """Validate a SpatialData points element for direct value selection."""
    points_name = _validate_column_name(points_name, "points_name")

    points_collection = getattr(sdata, "points", None)
    if points_collection is None or points_name not in points_collection:
        raise ValueError(f"Points element `{points_name}` is not available in the SpatialData object.")

    points = points_collection[points_name]
    if not isinstance(points, dd.DataFrame):
        raise ValueError(f"Points element `{points_name}` must resolve to a dask.dataframe.DataFrame.")

    x = _validate_column_name(x, "x")
    y = _validate_column_name(y, "y")
    index_column = _validate_column_name(index_column, "index_column")
    if transcript_id is not None:
        transcript_id = _validate_column_name(transcript_id, "transcript_id")

    if index_column in {x, y}:
        raise ValueError("`index_column` must be different from the configured coordinate columns.")

    _validate_required_columns(points, [x, y, index_column])
    if transcript_id is not None:
        _validate_required_columns(points, [transcript_id])

    _validate_numeric_column(points, x)
    _validate_numeric_column(points, y)
    _validate_index_column_dtype(points, index_column)

    row_count, invalid_x, invalid_y, missing_index, empty_index, invalid_index, *transcript_checks = dask.compute(
        points.map_partitions(len, meta=("row_count", "int64")).sum(),
        points[x].map_partitions(_count_nonfinite_values, meta=("invalid_x", "int64")).sum(),
        points[y].map_partitions(_count_nonfinite_values, meta=("invalid_y", "int64")).sum(),
        points[index_column].map_partitions(_count_missing_values, meta=("missing_index", "int64")).sum(),
        points[index_column].map_partitions(_count_empty_index_values, meta=("empty_index", "int64")).sum(),
        points[index_column].map_partitions(_count_invalid_index_values, meta=("invalid_index", "int64")).sum(),
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

    source_n_points = int(row_count)
    if source_n_points == 0:
        raise ValueError("`points` must not be empty.")
    if int(invalid_x) > 0:
        raise ValueError(f"Column `{x}` contains missing, NaN, or infinite coordinate values.")
    if int(invalid_y) > 0:
        raise ValueError(f"Column `{y}` contains missing, NaN, or infinite coordinate values.")
    if int(missing_index) > 0:
        raise ValueError(f"Column `{index_column}` contains missing index values.")
    if int(empty_index) > 0:
        raise ValueError(f"Column `{index_column}` contains empty index values after stripping whitespace.")
    if int(invalid_index) > 0:
        raise ValueError(f"Column `{index_column}` contains unsupported index values.")

    if transcript_id is not None:
        missing_transcript_id, unique_transcript_id_count = transcript_checks
        if int(missing_transcript_id) > 0:
            raise ValueError(f"Column `{transcript_id}` contains missing transcript_id values.")
        if int(unique_transcript_id_count) != source_n_points:
            raise ValueError(f"Column `{transcript_id}` must contain unique transcript_id values.")

    return _ValidatedPointsElement(
        points=points,
        points_name=points_name,
        source_path=_source_path_from_sdata(sdata),
        source_n_points=source_n_points,
        x=x,
        y=y,
        index_column=index_column,
        transcript_id=transcript_id,
    )


def normalize_index_value(value: object) -> str:
    """Normalize one index-column value for direct value selection."""
    if _is_missing_scalar(value):
        raise ValueError("Index values must not be missing.")
    if _is_unsupported_index_value(value):
        raise ValueError("Index values must be strings.")

    normalized = str(value).strip()
    if not normalized:
        raise ValueError("Index values must not be empty after stripping whitespace.")
    return normalized


def normalize_index_values(values: pd.Series) -> pd.Series:
    """Normalize a pandas Series of index-column values."""
    if not isinstance(values, pd.Series):
        raise ValueError("`values` must be a pandas Series.")
    return values.map(normalize_index_value)


def _source_path_from_sdata(sdata: Any) -> Path | None:
    is_backed = getattr(sdata, "is_backed", None)
    path = getattr(sdata, "path", None)
    if callable(is_backed) and is_backed() and path is not None:
        return Path(path)
    return None


def _validate_column_name(column: Any, parameter_name: str) -> str:
    if not isinstance(column, str) or not column:
        raise ValueError(f"`{parameter_name}` must be a non-empty string.")
    return column


def _validate_required_columns(points: dd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in points.columns]
    if missing:
        missing_columns = ", ".join(f"`{column}`" for column in missing)
        raise ValueError(f"`points` is missing required column(s): {missing_columns}.")


def _validate_numeric_column(points: dd.DataFrame, column: str) -> None:
    if not is_numeric_dtype(points._meta[column].dtype):
        raise ValueError(f"Column `{column}` must be numeric.")


def _validate_index_column_dtype(points: dd.DataFrame, column: str) -> None:
    dtype = points._meta[column].dtype
    if is_bool_dtype(dtype) or is_numeric_dtype(dtype):
        raise ValueError(f"Column `{column}` must contain string-like or categorical values.")
    if isinstance(dtype, pd.CategoricalDtype) or is_string_dtype(dtype) or is_object_dtype(dtype):
        return
    raise ValueError(f"Column `{column}` must contain string-like or categorical values.")


def _count_nonfinite_values(values: pd.Series) -> int:
    numeric_values = pd.to_numeric(values, errors="coerce").to_numpy(dtype="float64", na_value=np.nan)
    return int((~np.isfinite(numeric_values)).sum())


def _count_missing_values(values: pd.Series) -> int:
    return int(values.isna().sum())


def _count_empty_index_values(values: pd.Series) -> int:
    count = 0
    for value in values:
        if _is_missing_scalar(value) or _is_unsupported_index_value(value):
            continue
        if not str(value).strip():
            count += 1
    return count


def _count_invalid_index_values(values: pd.Series) -> int:
    count = 0
    for value in values:
        if _is_missing_scalar(value):
            continue
        if _is_unsupported_index_value(value):
            count += 1
    return count


def _is_missing_scalar(value: object) -> bool:
    if value is None:
        return True
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    if isinstance(missing, (bool, np.bool_)):
        return bool(missing)
    return False


def _is_unsupported_index_value(value: object) -> bool:
    if isinstance(value, bytes | bytearray | memoryview):
        return True
    if isinstance(value, bool | np.bool_):
        return True
    if isinstance(value, numbers.Number):
        return True
    if isinstance(value, list | tuple | dict | set | frozenset):
        return True
    return not isinstance(value, str)
