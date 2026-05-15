from __future__ import annotations

from dataclasses import FrozenInstanceError

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
    TranscriptValueVocabulary,
)


def _example_vocabulary(**overrides: object) -> TranscriptValueVocabulary:
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
    return TranscriptValueVocabulary(**values)


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


def test_transcript_value_index_constants() -> None:
    assert DEFAULT_X == "x"
    assert DEFAULT_Y == "y"
    assert DEFAULT_INDEX_COLUMN == "gene"
    assert DEFAULT_RENDER_POINT_BUDGET == 100_000
    assert DEFAULT_RANDOM_STATE == 42
    assert TRANSCRIPT_VALUE_INDEX_SCHEMA_VERSION == "harpy-transcripts-value-index-0.1"


def test_transcript_value_vocabulary_records_value_table() -> None:
    vocabulary = _example_vocabulary()

    assert tuple(vocabulary.values.columns) == ("value_id", "value", "n_points")
    assert vocabulary.values["value_id"].dtype == np.dtype("uint32")
    assert vocabulary.values["n_points"].dtype == np.dtype("uint64")
    assert vocabulary.index_column == "gene"
    assert vocabulary.total_count == 5


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
def test_transcript_value_vocabulary_rejects_invalid_values(overrides: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _example_vocabulary(**overrides)


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
