from __future__ import annotations

import numbers
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import dask
import dask.dataframe as dd
import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_integer_dtype, is_numeric_dtype, is_object_dtype, is_string_dtype

POINTS_VALUE_INDEX_SCHEMA_VERSION = "harpy-points-value-index-0.1"
DEFAULT_X = "x"
DEFAULT_Y = "y"
DEFAULT_INDEX_COLUMN = "gene"
DEFAULT_RENDER_POINT_BUDGET = 100_000
DEFAULT_RANDOM_STATE = 42

VALUE_ID_COLUMN = "value_id"
VALUE_COLUMN = "value"
N_POINTS_COLUMN = "n_points"
VALUE_TABLE_COLUMNS = (VALUE_ID_COLUMN, VALUE_COLUMN, N_POINTS_COLUMN)
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
class PointsValueTable:
    """In-memory value table for a selected points column.

    Parameters
    ----------
    values
        DataFrame with exactly `value_id`, `value`, and `n_points` columns.
        `value_id` must be `uint32`, `value` must contain unique normalized
        strings, and `n_points` must be `uint64`.
    index_column
        Name of the source points column used to produce the value table,
        for example `gene`, `target`, or `probe`.
    total_count
        Total number of source points represented by the value table. Must equal
        `values["n_points"].sum()`.
    """

    values: pd.DataFrame
    index_column: str
    total_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.values, pd.DataFrame):
            raise ValueError("Points value table `values` must be a pandas DataFrame.")
        if tuple(self.values.columns) != VALUE_TABLE_COLUMNS:
            raise ValueError(
                "Points value table `values` must contain exactly `value_id`, `value`, and `n_points` columns."
            )
        if not isinstance(self.index_column, str) or not self.index_column:
            raise ValueError("Points value table `index_column` must be a non-empty string.")
        _validate_non_negative_integer("total_count", self.total_count)

        if self.values[VALUE_ID_COLUMN].dtype != VALUE_ID_DTYPE:
            raise ValueError("Points value table `value_id` column must have dtype uint32.")
        if self.values[N_POINTS_COLUMN].dtype != N_POINTS_DTYPE:
            raise ValueError("Points value table `n_points` column must have dtype uint64.")
        if self.values[VALUE_ID_COLUMN].isna().any():
            raise ValueError("Points value table `value_id` column must not contain missing values.")
        if self.values[VALUE_COLUMN].isna().any():
            raise ValueError("Points value table `value` column must not contain missing values.")
        if self.values[N_POINTS_COLUMN].isna().any():
            raise ValueError("Points value table `n_points` column must not contain missing values.")
        if self.values[VALUE_ID_COLUMN].duplicated().any():
            raise ValueError("Points value table `value_id` values must be unique.")
        if self.values[VALUE_COLUMN].duplicated().any():
            raise ValueError("Points value table `value` values must be unique.")
        if not self.values[VALUE_COLUMN].map(lambda value: isinstance(value, str)).all():
            raise ValueError("Points value table `value` column must contain strings.")

        observed_total = int(self.values[N_POINTS_COLUMN].sum())
        if observed_total != self.total_count:
            raise ValueError("Points value table `total_count` must equal the sum of `n_points`.")


@dataclass(frozen=True, kw_only=True)
class PointsValueSelection:
    """Selected points ready to be displayed as one napari Points layer.

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
            raise ValueError("Points value selection `coordinates` must be a numpy array.")
        if self.coordinates.ndim != 2 or self.coordinates.shape[1] != 2:
            raise ValueError("Points value selection `coordinates` must be an Nx2 array.")
        if self.coordinates.dtype != COORDINATE_DTYPE:
            raise ValueError("Points value selection `coordinates` must have dtype float32.")
        if not isinstance(self.features, pd.DataFrame):
            raise ValueError("Points value selection `features` must be a pandas DataFrame.")
        if not isinstance(self.index_column, str) or not self.index_column:
            raise ValueError("Points value selection `index_column` must be a non-empty string.")
        if self.index_column == VALUE_ID_COLUMN:
            raise ValueError("Points value selection `index_column` must not be `value_id`.")
        if set(self.features.columns) != {self.index_column, VALUE_ID_COLUMN}:
            raise ValueError(
                "Points value selection `features` must contain exactly the configured index column and `value_id`."
            )
        if not isinstance(self.features[self.index_column].dtype, pd.CategoricalDtype):
            raise ValueError("Points value selection index feature must be categorical.")
        if is_bool_dtype(self.features[VALUE_ID_COLUMN].dtype) or not is_integer_dtype(
            self.features[VALUE_ID_COLUMN].dtype
        ):
            raise ValueError("Points value selection `value_id` feature must be integer typed.")
        if not isinstance(self.selected_values, tuple) or not all(
            isinstance(value, str) for value in self.selected_values
        ):
            raise ValueError("Points value selection `selected_values` must be a tuple of strings.")
        if not isinstance(self.selected_value_ids, tuple) or not all(
            _is_non_negative_integral(value) for value in self.selected_value_ids
        ):
            raise ValueError("Points value selection `selected_value_ids` must be a tuple of non-negative integers.")
        if len(self.selected_values) != len(self.selected_value_ids):
            raise ValueError("Points value selection selected values and ids must have the same length.")
        if len(set(self.selected_values)) != len(self.selected_values):
            raise ValueError("Points value selection `selected_values` must not contain duplicates.")
        if len({int(value_id) for value_id in self.selected_value_ids}) != len(self.selected_value_ids):
            raise ValueError("Points value selection `selected_value_ids` must not contain duplicates.")

        _validate_non_negative_integer("total_count", self.total_count)
        _validate_positive_integer("render_point_budget", self.render_point_budget)
        if not isinstance(self.is_sampled, bool):
            raise ValueError("Points value selection `is_sampled` must be a boolean.")
        if self.warning is not None and not isinstance(self.warning, str):
            raise ValueError("Points value selection `warning` must be a string or None.")

        if len(self.features) != self.loaded_count:
            raise ValueError("Points value selection loaded rows must match coordinates and features.")
        if self.loaded_count > self.total_count:
            raise ValueError("Points value selection `loaded_count` must be <= `total_count`.")
        if self.loaded_count > self.render_point_budget:
            raise ValueError("Points value selection `loaded_count` must be <= `render_point_budget`.")
        if not self.is_sampled and self.loaded_count != self.total_count:
            raise ValueError("Exact points value selections must load all selected points.")
        if self.is_sampled and (self.warning is None or not self.warning):
            raise ValueError("Sampled points value selections must include a warning.")
        if not self.is_sampled and self.warning is not None:
            raise ValueError("Exact points value selections must not include a warning.")

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

    row_count, invalid_x, invalid_y, index_value_errors, *transcript_checks = dask.compute(
        points.map_partitions(len, meta=("row_count", "int64")).sum(),
        points[x].map_partitions(_count_nonfinite_values, meta=("invalid_x", "int64")).sum(),
        points[y].map_partitions(_count_nonfinite_values, meta=("invalid_y", "int64")).sum(),
        points[index_column].map_partitions(_count_index_value_errors, meta=_index_value_error_meta()).sum(),
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
    if int(index_value_errors["missing_index"]) > 0:
        raise ValueError(f"Column `{index_column}` contains missing index values.")
    if int(index_value_errors["invalid_index"]) > 0:
        raise ValueError(f"Column `{index_column}` contains invalid index values.")

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


def build_points_value_table(validated: _ValidatedPointsElement) -> PointsValueTable:
    """Build an in-memory value table from a validated Dask points element."""
    if not isinstance(validated, _ValidatedPointsElement):
        raise ValueError("`validated` must be a _ValidatedPointsElement.")

    normalized_values = validated.points[validated.index_column].map_partitions(
        _normalize_index_value_partition,
        meta=(VALUE_COLUMN, "object"),
    )
    value_counts = normalized_values.value_counts(sort=False).compute()
    value_counts = value_counts.groupby(level=0, observed=True).sum()
    value_counts = value_counts[value_counts > 0].sort_index()

    if len(value_counts) > np.iinfo(VALUE_ID_DTYPE).max:
        raise ValueError("Too many unique point values to represent with uint32 value ids.")

    total_count = int(value_counts.sum())
    if total_count != validated.source_n_points:
        raise ValueError("Direct value table total count does not match the validated source point count.")

    values = pd.DataFrame(
        {
            VALUE_ID_COLUMN: pd.Series(np.arange(len(value_counts), dtype=VALUE_ID_DTYPE), dtype=VALUE_ID_DTYPE),
            VALUE_COLUMN: value_counts.index.to_list(),
            N_POINTS_COLUMN: pd.Series(value_counts.to_numpy(dtype=N_POINTS_DTYPE), dtype=N_POINTS_DTYPE),
        }
    )
    return PointsValueTable(
        values=values,
        index_column=validated.index_column,
        total_count=total_count,
    )


def load_points(
    validated: _ValidatedPointsElement,
    value_table: PointsValueTable,
    values: Sequence[str] | Literal["all"],
    *,
    render_point_budget: int = DEFAULT_RENDER_POINT_BUDGET,
    random_state: int | None = DEFAULT_RANDOM_STATE,
) -> PointsValueSelection:
    """Load points for selected values using the direct no-cache path.

    Parameters
    ----------
    validated
        Validated source points element returned by
        :func:`validate_points_element_for_value_selection`.
    value_table
        In-memory value/count table returned by :func:`build_points_value_table`.
    values
        Values to load. Source-form values are normalized before lookup. Use
        ``"all"`` to select every value in ``value_table``.
    render_point_budget
        Maximum number of points to return for napari rendering. If the selected
        total count exceeds this budget, rows are sampled before compute and
        trimmed after compute.
    random_state
        Random seed forwarded to Dask sampling when sampling is required.
    """
    if not isinstance(validated, _ValidatedPointsElement):
        raise ValueError("`validated` must be a _ValidatedPointsElement.")
    if not isinstance(value_table, PointsValueTable):
        raise ValueError("`value_table` must be a PointsValueTable.")
    if validated.index_column != value_table.index_column:
        raise ValueError(
            "`validated.index_column` and `value_table.index_column` must match. "
            f"Got {validated.index_column!r} and {value_table.index_column!r}."
        )
    _validate_positive_integer("render_point_budget", render_point_budget)
    render_point_budget = int(render_point_budget)

    selected_values_table = _resolve_selected_values(value_table, values)
    selected_values = tuple(str(value) for value in selected_values_table[VALUE_COLUMN].to_numpy())
    selected_value_ids = tuple(int(value_id) for value_id in selected_values_table[VALUE_ID_COLUMN].to_numpy())
    total_count = int(selected_values_table[N_POINTS_COLUMN].sum())

    if total_count == 0:
        return _selection_from_points_frame(
            _empty_selected_points_frame(validated),
            validated=validated,
            selected_values=selected_values,
            selected_value_ids=selected_value_ids,
            total_count=total_count,
            render_point_budget=render_point_budget,
            is_sampled=False,
        )

    value_id_by_value = {
        str(row[VALUE_COLUMN]): int(row[VALUE_ID_COLUMN]) for _, row in selected_values_table.iterrows()
    }
    selected_points = validated.points.map_partitions(
        _filter_points_partition,
        x=validated.x,
        y=validated.y,
        index_column=validated.index_column,
        selected_values=selected_values,
        value_id_by_value=value_id_by_value,
        meta=_empty_selected_points_frame(validated),
    )
    is_sampled = total_count > render_point_budget
    if is_sampled:
        selected_points = selected_points.sample(
            frac=render_point_budget / total_count,
            random_state=random_state,
        )

    loaded = selected_points.compute()
    if is_sampled and len(loaded) > render_point_budget:
        loaded = loaded.iloc[:render_point_budget]

    return _selection_from_points_frame(
        loaded,
        validated=validated,
        selected_values=selected_values,
        selected_value_ids=selected_value_ids,
        total_count=total_count,
        render_point_budget=render_point_budget,
        is_sampled=is_sampled,
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


def _normalize_index_value_partition(values: pd.Series) -> pd.Series:
    normalized = normalize_index_values(values)
    normalized.name = VALUE_COLUMN
    return normalized


def _resolve_selected_values(
    value_table: PointsValueTable,
    values: Sequence[str] | Literal["all"],
) -> pd.DataFrame:
    sorted_values = value_table.values.sort_values(VALUE_ID_COLUMN).reset_index(drop=True)
    if isinstance(values, str):
        if values == "all":
            return sorted_values
        raise ValueError("`values` must be a sequence of values or the literal 'all'.")
    if not isinstance(values, Sequence):
        raise ValueError("`values` must be a sequence of values or the literal 'all'.")

    requested_values = tuple(dict.fromkeys(normalize_index_value(value) for value in values))
    if not requested_values:
        return sorted_values.iloc[0:0].copy()

    requested_value_set = set(requested_values)
    selected_values = sorted_values[sorted_values[VALUE_COLUMN].isin(requested_value_set)].copy()
    known_values = set(selected_values[VALUE_COLUMN])
    unknown_values = [value for value in requested_values if value not in known_values]
    if unknown_values:
        unknown = ", ".join(repr(value) for value in unknown_values)
        raise ValueError(f"Unknown selected point value(s): {unknown}.")
    return selected_values


def _filter_points_partition(
    points: pd.DataFrame,
    *,
    x: str,
    y: str,
    index_column: str,
    selected_values: tuple[str, ...],
    value_id_by_value: dict[str, int],
) -> pd.DataFrame:
    normalized_values = normalize_index_values(points[index_column])
    selected_mask = normalized_values.isin(selected_values)
    if not bool(selected_mask.any()):
        return _empty_points_frame_for_columns(points, x=x, y=y, index_column=index_column)

    selected_normalized = normalized_values.loc[selected_mask].astype("object")
    filtered = points.loc[selected_mask, [y, x]].copy()
    filtered[index_column] = selected_normalized.to_numpy()
    filtered[VALUE_ID_COLUMN] = selected_normalized.map(value_id_by_value).astype(VALUE_ID_DTYPE).to_numpy()
    return filtered[[y, x, index_column, VALUE_ID_COLUMN]]


def _selection_from_points_frame(
    points: pd.DataFrame,
    *,
    validated: _ValidatedPointsElement,
    selected_values: tuple[str, ...],
    selected_value_ids: tuple[int, ...],
    total_count: int,
    render_point_budget: int,
    is_sampled: bool,
) -> PointsValueSelection:
    coordinates = points[[validated.y, validated.x]].to_numpy(dtype=COORDINATE_DTYPE, copy=True)
    features = pd.DataFrame(
        {
            validated.index_column: pd.Categorical(
                points[validated.index_column].to_numpy(),
                categories=list(selected_values),
            ),
            VALUE_ID_COLUMN: pd.Series(points[VALUE_ID_COLUMN].to_numpy(dtype=VALUE_ID_DTYPE), dtype=VALUE_ID_DTYPE),
        }
    )
    warning = (
        f"Showing {len(points):,} of {total_count:,} selected points "
        f"because the render point budget is {render_point_budget:,}."
        if is_sampled
        else None
    )
    return PointsValueSelection(
        coordinates=coordinates,
        features=features,
        index_column=validated.index_column,
        selected_values=selected_values,
        selected_value_ids=selected_value_ids,
        total_count=total_count,
        render_point_budget=render_point_budget,
        is_sampled=is_sampled,
        warning=warning,
    )


def _empty_selected_points_frame(validated: _ValidatedPointsElement) -> pd.DataFrame:
    return pd.DataFrame(
        {
            validated.y: pd.Series(dtype=validated.points._meta[validated.y].dtype),
            validated.x: pd.Series(dtype=validated.points._meta[validated.x].dtype),
            validated.index_column: pd.Series(dtype="object"),
            VALUE_ID_COLUMN: pd.Series(dtype=VALUE_ID_DTYPE),
        }
    )[[validated.y, validated.x, validated.index_column, VALUE_ID_COLUMN]]


def _empty_points_frame_for_columns(points: pd.DataFrame, *, x: str, y: str, index_column: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            y: pd.Series(dtype=points[y].dtype),
            x: pd.Series(dtype=points[x].dtype),
            index_column: pd.Series(dtype="object"),
            VALUE_ID_COLUMN: pd.Series(dtype=VALUE_ID_DTYPE),
        }
    )[[y, x, index_column, VALUE_ID_COLUMN]]


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


def _index_value_error_meta() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "missing_index": pd.Series(dtype="int64"),
            "invalid_index": pd.Series(dtype="int64"),
        }
    )


def _count_index_value_errors(values: pd.Series) -> pd.DataFrame:
    missing_count = int(values.isna().sum())
    invalid_count = _count_invalid_index_values(values)
    return pd.DataFrame(
        {
            "missing_index": pd.Series([missing_count], dtype="int64"),
            "invalid_index": pd.Series([invalid_count], dtype="int64"),
        }
    )


def _count_invalid_index_values(values: pd.Series) -> int:
    if isinstance(values.dtype, pd.CategoricalDtype):
        categories = values.cat.remove_unused_categories().cat.categories
        invalid_categories = [value for value in categories if _is_invalid_index_value(value)]
        if not invalid_categories:
            return 0
        return int(values.isin(invalid_categories).sum())
    if is_string_dtype(values.dtype) and not is_object_dtype(values.dtype):
        return int(values.dropna().str.strip().eq("").sum())

    count = 0
    for value in values:
        if _is_invalid_index_value(value):
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


def _is_invalid_index_value(value: object) -> bool:
    if _is_missing_scalar(value):
        return False
    if _is_unsupported_index_value(value):
        return True
    return not str(value).strip()
