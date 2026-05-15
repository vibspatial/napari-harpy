from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import dask
import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest

from napari_harpy._transcript_value_index import (
    DEFAULT_INDEX_COLUMN,
    DEFAULT_RANDOM_STATE,
    DEFAULT_RENDER_POINT_BUDGET,
    DEFAULT_X,
    DEFAULT_Y,
    TRANSCRIPT_VALUE_INDEX_SCHEMA_VERSION,
    TranscriptValueSelection,
    TranscriptValueTable,
    _ValidatedPointsElement,
    normalize_index_value,
    normalize_index_values,
    validate_points_element_for_value_selection,
)


class _DummySpatialData:
    def __init__(self, points: dict[str, object], *, path: Path | None = None, backed: bool = False) -> None:
        self.points = points
        self.path = path
        self._backed = backed

    def is_backed(self) -> bool:
        return self._backed


def _example_value_table(**overrides: object) -> TranscriptValueTable:
    values = {
        "values": pd.DataFrame(
            {
                "value_id": pd.Series([0, 1], dtype="uint32"),
                "value": ["AAMP", "AXL"],
                "n_points": pd.Series([3, 2], dtype="uint64"),
            }
        ),
        "index_column": "gene",
        "total_count": 5,
    }
    values.update(overrides)
    return TranscriptValueTable(**values)


def _example_features(values: list[str] | None = None, value_ids: list[int] | None = None) -> pd.DataFrame:
    gene_values = ["AAMP", "AAMP", "AXL"] if values is None else values
    ids = [0, 0, 1] if value_ids is None else value_ids
    return pd.DataFrame(
        {
            "gene": pd.Categorical(gene_values, categories=["AAMP", "AXL"]),
            "value_id": pd.Series(ids, dtype="uint32"),
        }
    )


def _example_selection(**overrides: object) -> TranscriptValueSelection:
    values = {
        "coordinates": np.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype="float32"),
        "features": _example_features(),
        "index_column": "gene",
        "selected_values": ("AAMP", "AXL"),
        "selected_value_ids": (0, 1),
        "total_count": 3,
        "render_point_budget": 100_000,
        "is_sampled": False,
        "warning": None,
    }
    values.update(overrides)
    return TranscriptValueSelection(**values)


def _points_dataframe(data: dict[str, object], *, npartitions: int = 1) -> dd.DataFrame:
    with dask.config.set({"dataframe.convert-string": False}):
        return dd.from_pandas(pd.DataFrame(data), npartitions=npartitions)


def _sdata_with_points(
    data: dict[str, object],
    *,
    points_name: str = "transcripts",
    backed: bool = False,
    path: Path | None = None,
    npartitions: int = 1,
) -> _DummySpatialData:
    return _DummySpatialData(
        {points_name: _points_dataframe(data, npartitions=npartitions)},
        path=path,
        backed=backed,
    )


def _valid_points_data(**overrides: object) -> dict[str, object]:
    data: dict[str, object] = {
        "x": [0.0, 1.0, 2.0],
        "y": [3.0, 4.0, 5.0],
        "gene": [" AAMP ", "AXL", "MALAT1"],
        "transcript_id": ["tx1", "tx2", "tx3"],
    }
    data.update(overrides)
    return data


def test_transcript_value_index_constants() -> None:
    assert DEFAULT_X == "x"
    assert DEFAULT_Y == "y"
    assert DEFAULT_INDEX_COLUMN == "gene"
    assert DEFAULT_RENDER_POINT_BUDGET == 100_000
    assert DEFAULT_RANDOM_STATE == 42
    assert TRANSCRIPT_VALUE_INDEX_SCHEMA_VERSION == "harpy-transcripts-value-index-0.1"


def test_transcript_value_table_records_value_table() -> None:
    value_table = _example_value_table()

    assert tuple(value_table.values.columns) == ("value_id", "value", "n_points")
    assert value_table.values["value_id"].dtype == np.dtype("uint32")
    assert value_table.values["n_points"].dtype == np.dtype("uint64")
    assert value_table.index_column == "gene"
    assert value_table.total_count == 5


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"values": pd.DataFrame({"value": ["AAMP"], "value_id": pd.Series([0], dtype="uint32")})}, "exactly"),
        (
            {
                "values": pd.DataFrame(
                    {
                        "value_id": pd.Series([0], dtype="uint64"),
                        "value": ["AAMP"],
                        "n_points": pd.Series([1], dtype="uint64"),
                    }
                )
            },
            "value_id.*uint32",
        ),
        (
            {
                "values": pd.DataFrame(
                    {
                        "value_id": pd.Series([0], dtype="uint32"),
                        "value": ["AAMP"],
                        "n_points": pd.Series([1], dtype="uint32"),
                    }
                )
            },
            "n_points.*uint64",
        ),
        (
            {
                "values": pd.DataFrame(
                    {
                        "value_id": pd.Series([0, 0], dtype="uint32"),
                        "value": ["AAMP", "AXL"],
                        "n_points": pd.Series([1, 1], dtype="uint64"),
                    }
                )
            },
            "value_id.*unique",
        ),
        (
            {
                "values": pd.DataFrame(
                    {
                        "value_id": pd.Series([0, 1], dtype="uint32"),
                        "value": ["AAMP", "AAMP"],
                        "n_points": pd.Series([1, 1], dtype="uint64"),
                    }
                )
            },
            "value.*unique",
        ),
        (
            {
                "values": pd.DataFrame(
                    {
                        "value_id": pd.Series([0], dtype="uint32"),
                        "value": ["AAMP"],
                        "n_points": pd.Series([1], dtype="uint64"),
                    }
                ),
                "total_count": 2,
            },
            "total_count",
        ),
        ({"index_column": ""}, "index_column"),
    ],
)
def test_transcript_value_table_rejects_invalid_values(overrides: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _example_value_table(**overrides)


def test_transcript_value_selection_records_display_payload() -> None:
    selection = _example_selection()

    assert selection.coordinates.dtype == np.dtype("float32")
    assert selection.coordinates.shape == (3, 2)
    assert selection.index_column == "gene"
    assert list(selection.features.columns) == ["gene", "value_id"]
    assert isinstance(selection.features["gene"].dtype, pd.CategoricalDtype)
    assert selection.selected_values == ("AAMP", "AXL")
    assert selection.selected_value_ids == (0, 1)
    assert selection.total_count == 3
    assert selection.loaded_count == 3
    assert selection.render_point_budget == 100_000
    assert selection.is_sampled is False
    assert selection.warning is None


def test_transcript_value_selection_accepts_sampled_result() -> None:
    selection = _example_selection(
        coordinates=np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype="float32"),
        features=_example_features(["AAMP", "AXL"], [0, 1]),
        total_count=10,
        render_point_budget=2,
        is_sampled=True,
        warning="Showing 2 of 10 selected points.",
    )

    assert selection.is_sampled is True
    assert selection.loaded_count == 2
    assert selection.warning == "Showing 2 of 10 selected points."


def test_transcript_value_selection_accepts_empty_exact_result() -> None:
    selection = _example_selection(
        coordinates=np.empty((0, 2), dtype="float32"),
        features=_example_features([], []),
        selected_values=(),
        selected_value_ids=(),
        total_count=0,
    )

    assert selection.coordinates.shape == (0, 2)
    assert len(selection.features) == 0
    assert selection.total_count == 0
    assert selection.loaded_count == 0


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"coordinates": np.asarray([1.0, 2.0], dtype="float32")}, "Nx2"),
        ({"coordinates": np.asarray([[1.0, 2.0]], dtype="float64")}, "float32"),
        ({"features": pd.DataFrame({"gene": pd.Categorical(["AAMP"])})}, "exactly"),
        (
            {"features": pd.DataFrame({"gene": ["AAMP"], "value_id": pd.Series([0], dtype="uint32")})},
            "categorical",
        ),
        ({"features": pd.DataFrame({"gene": pd.Categorical(["AAMP"]), "value_id": [True]})}, "integer"),
        ({"index_column": ""}, "index_column"),
        ({"index_column": "value_id"}, "value_id"),
        ({"selected_values": ["AAMP"]}, "selected_values"),
        ({"selected_values": ("AAMP", "AAMP"), "selected_value_ids": (0, 1)}, "duplicates"),
        ({"selected_value_ids": (0, -1)}, "selected_value_ids"),
        ({"selected_value_ids": (0,)}, "same length"),
        ({"total_count": -1}, "total_count"),
        ({"features": _example_features(["AAMP"], [0])}, "loaded rows"),
        ({"render_point_budget": 0}, "render_point_budget"),
        ({"is_sampled": "no"}, "is_sampled"),
        ({"warning": 1}, "warning"),
        ({"total_count": 4}, "Exact"),
        (
            {
                "coordinates": np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype="float32"),
                "features": _example_features(["AAMP", "AXL"], [0, 1]),
                "total_count": 1,
            },
            "loaded_count.*total_count",
        ),
        ({"render_point_budget": 2}, "loaded_count.*render_point_budget"),
        ({"is_sampled": True, "total_count": 4}, "warning"),
        ({"warning": "not needed"}, "must not include a warning"),
    ],
)
def test_transcript_value_selection_rejects_invalid_values(overrides: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _example_selection(**overrides)


def test_transcript_value_selection_is_immutable() -> None:
    selection = _example_selection()

    with pytest.raises(FrozenInstanceError):
        selection.loaded_count = 1


def test_normalize_index_value_strips_edges_and_preserves_case_and_internal_whitespace() -> None:
    assert normalize_index_value(" Act b ") == "Act b"
    assert normalize_index_value("ACTB") == "ACTB"
    assert normalize_index_value("actb") == "actb"


@pytest.mark.parametrize(
    "value",
    [None, pd.NA, b"AAMP", bytearray(b"AAMP"), 1, 1.2, True, ["AAMP"], {"gene": "AAMP"}, ("AAMP",)],
)
def test_normalize_index_value_rejects_invalid_values(value: object) -> None:
    with pytest.raises(ValueError):
        normalize_index_value(value)


def test_normalize_index_values_normalizes_series() -> None:
    values = pd.Series([" AAMP ", "AXL"], name="gene")

    normalized = normalize_index_values(values)

    assert normalized.tolist() == ["AAMP", "AXL"]
    assert normalized.name == "gene"


def test_normalize_index_values_rejects_non_series() -> None:
    with pytest.raises(ValueError, match="pandas Series"):
        normalize_index_values(["AAMP"])  # type: ignore[arg-type]


def test_validate_points_element_for_value_selection_accepts_unbacked_points() -> None:
    sdata = _sdata_with_points(_valid_points_data(), backed=False)

    validated = validate_points_element_for_value_selection(sdata, "transcripts")

    assert isinstance(validated, _ValidatedPointsElement)
    assert validated.points is sdata.points["transcripts"]
    assert validated.points_name == "transcripts"
    assert validated.source_path is None
    assert validated.is_backed is False
    assert validated.element_path is None
    assert validated.source_n_points == 3
    assert validated.x == "x"
    assert validated.y == "y"
    assert validated.index_column == "gene"
    assert validated.transcript_id is None


def test_validate_points_element_for_value_selection_accepts_backed_points(tmp_path: Path) -> None:
    path = tmp_path / "example.zarr"
    sdata = _sdata_with_points(_valid_points_data(), backed=True, path=path)

    validated = validate_points_element_for_value_selection(sdata, "transcripts", transcript_id="transcript_id")

    assert validated.source_path == path
    assert validated.is_backed is True
    assert validated.element_path == "points/transcripts"
    assert validated.transcript_id == "transcript_id"


def test_validate_points_element_for_value_selection_accepts_configured_index_column() -> None:
    sdata = _sdata_with_points(_valid_points_data(target=pd.Series([" A ", "B", "B"], dtype="string")))

    validated = validate_points_element_for_value_selection(sdata, "transcripts", index_column="target")

    assert validated.index_column == "target"


def test_validate_points_element_for_value_selection_accepts_categorical_index_column() -> None:
    sdata = _sdata_with_points(_valid_points_data(gene=pd.Categorical([" AAMP ", "AXL", "MALAT1"])))

    validated = validate_points_element_for_value_selection(sdata, "transcripts")

    assert validated.index_column == "gene"


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"points_name": ""}, "points_name"),
        ({"points_name": "missing"}, "not available"),
        ({"x": ""}, "`x`"),
        ({"y": ""}, "`y`"),
        ({"index_column": ""}, "`index_column`"),
        ({"transcript_id": ""}, "`transcript_id`"),
        ({"x": "missing"}, "missing"),
        ({"y": "missing"}, "missing"),
        ({"index_column": "missing"}, "missing"),
        ({"index_column": "x"}, "index_column"),
        ({"index_column": "y"}, "index_column"),
        ({"transcript_id": "missing"}, "missing"),
    ],
)
def test_validate_points_element_for_value_selection_rejects_invalid_arguments(
    kwargs: dict[str, object], match: str
) -> None:
    sdata = _sdata_with_points(_valid_points_data())
    points_name = str(kwargs.pop("points_name", "transcripts"))

    with pytest.raises(ValueError, match=match):
        validate_points_element_for_value_selection(sdata, points_name, **kwargs)


def test_validate_points_element_for_value_selection_rejects_non_dask_points() -> None:
    sdata = _DummySpatialData({"transcripts": pd.DataFrame(_valid_points_data())})

    with pytest.raises(ValueError, match="dask.dataframe.DataFrame"):
        validate_points_element_for_value_selection(sdata, "transcripts")


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"x": ["0", "1", "2"]}, "numeric"),
        ({"y": ["0", "1", "2"]}, "numeric"),
        ({"x": [0.0, np.nan, 2.0]}, "x.*finite"),
        ({"y": [0.0, np.inf, 2.0]}, "y.*finite"),
        ({"gene": ["AAMP", None, "AXL"]}, "missing index"),
        ({"gene": ["AAMP", "  ", "AXL"]}, "empty index"),
        ({"gene": [1, 2, 3]}, "string-like"),
        ({"gene": [True, False, True]}, "string-like"),
        ({"gene": ["AAMP", b"AXL", "MALAT1"]}, "unsupported"),
        ({"gene": ["AAMP", 1, "MALAT1"]}, "unsupported"),
        ({"gene": ["AAMP", True, "MALAT1"]}, "unsupported"),
        ({"gene": ["AAMP", ["AXL"], "MALAT1"]}, "unsupported"),
        ({"gene": ["AAMP", {"gene": "AXL"}, "MALAT1"]}, "unsupported"),
        ({"gene": ["AAMP", ("AXL",), "MALAT1"]}, "unsupported"),
    ],
)
def test_validate_points_element_for_value_selection_rejects_invalid_source_values(
    overrides: dict[str, object], match: str
) -> None:
    sdata = _sdata_with_points(_valid_points_data(**overrides))

    with pytest.raises(ValueError, match=match):
        validate_points_element_for_value_selection(sdata, "transcripts")


def test_validate_points_element_for_value_selection_rejects_empty_points() -> None:
    data = {
        "x": pd.Series([], dtype="float64"),
        "y": pd.Series([], dtype="float64"),
        "gene": pd.Series([], dtype="object"),
    }
    sdata = _sdata_with_points(data)

    with pytest.raises(ValueError, match="empty"):
        validate_points_element_for_value_selection(sdata, "transcripts")


@pytest.mark.parametrize(
    ("transcript_id_values", "match"),
    [
        (["tx1", None, "tx3"], "missing transcript_id"),
        (["tx1", "tx1", "tx3"], "unique transcript_id"),
    ],
)
def test_validate_points_element_for_value_selection_rejects_invalid_transcript_id_values(
    transcript_id_values: list[str | None], match: str
) -> None:
    sdata = _sdata_with_points(_valid_points_data(transcript_id=transcript_id_values))

    with pytest.raises(ValueError, match=match):
        validate_points_element_for_value_selection(sdata, "transcripts", transcript_id="transcript_id")
