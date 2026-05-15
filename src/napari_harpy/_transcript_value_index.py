from __future__ import annotations

import numbers
from dataclasses import dataclass

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_integer_dtype

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
                "Transcript value vocabulary `values` must contain exactly "
                "`value_id`, `value`, and `n_points` columns."
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
                "Transcript value selection `features` must contain exactly the configured index column "
                "and `value_id`."
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
            raise ValueError("Transcript value selection `selected_value_ids` must be a tuple of non-negative integers.")
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
