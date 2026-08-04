from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import napari_harpy.core.multi_scale_cache_points.validation as validation_module
from napari_harpy.core.multi_scale_cache_points import ParquetPointsSource, PointColumnSelection
from napari_harpy.core.multi_scale_cache_points.errors import PointContentValidationError
from napari_harpy.core.multi_scale_cache_points.validation import (
    VALUE_NORMALIZATION_METHOD,
    _normalized_value_counts,
    _read_parquet_source_inventory,
    _scan_points_content,
)


def _source(tmp_path: Path) -> ParquetPointsSource:
    source = ParquetPointsSource(
        spatialdata_path=tmp_path / "example.zarr",
        points_name="transcripts",
        columns=PointColumnSelection(x="x", y="y", value="gene"),
    )
    source.parquet_path.mkdir(parents=True)
    return source


def _plain_table(x: list[int], y: list[float], gene: list[str | None]) -> pa.Table:
    return pa.table(
        {
            "x": pa.array(x, type=pa.int64()),
            "y": pa.array(y, type=pa.float64()),
            "gene": pa.array(gene, type=pa.string()),
        }
    )


def _dictionary_array(dictionary: list[str | None], indices: list[int | None]) -> pa.DictionaryArray:
    return pa.DictionaryArray.from_arrays(
        pa.array(indices, type=pa.int8()),
        pa.array(dictionary, type=pa.string()),
    )


def _dictionary_table(indices: list[int], dictionary: list[str]) -> pa.Table:
    row_count = len(indices)
    return pa.table(
        {
            "x": pa.array(range(row_count), type=pa.float64()),
            "y": pa.array(range(row_count), type=pa.float64()),
            "gene": _dictionary_array(dictionary, indices),
        }
    )


def test_scan_builds_exact_summary_across_files_row_groups_and_batches(tmp_path: Path) -> None:
    source = _source(tmp_path)
    pq.write_table(
        _plain_table(
            [-5, 0, 10, 20, 100_000],
            [4.5, -2.0, 8.0, 100.0, 3.0],
            [" B ", "A", "B", "\u00a0A\u00a0", "a"],
        ),
        source.parquet_path / "part.0.parquet",
        row_group_size=3,
    )
    pq.write_table(
        _plain_table(
            [7, 8, 9, 10],
            [-3.0, 200.0, 0.0, 1.0],
            ["\u2003B\u2003", "A", "Å", "A\u030a"],
        ),
        source.parquet_path / "part.1.parquet",
        row_group_size=2,
    )

    content = _scan_points_content(_read_parquet_source_inventory(source), max_batch_rows=2)

    assert VALUE_NORMALIZATION_METHOD == "harpy-string-trim-unicode-white-space-case-sensitive-v1"
    assert content.row_count == 9
    assert content.bounds.x_min == -5.0
    assert content.bounds.x_max == 100_000.0
    assert content.bounds.y_min == -3.0
    assert content.bounds.y_max == 200.0
    assert content.value_table.schema == pa.schema(
        [
            pa.field("value_id", pa.uint32(), nullable=False),
            pa.field("value", pa.string(), nullable=False),
            pa.field("n_points", pa.uint64(), nullable=False),
        ]
    )
    assert content.value_table.to_pylist() == [
        {"value_id": 0, "value": "A", "n_points": 3},
        {"value_id": 1, "value": "A\u030a", "n_points": 1},
        {"value_id": 2, "value": "B", "n_points": 3},
        {"value_id": 3, "value": "a", "n_points": 1},
        {"value_id": 4, "value": "Å", "n_points": 1},
    ]


def test_scan_merges_local_dictionary_encodings(tmp_path: Path) -> None:
    source = _source(tmp_path)
    pq.write_table(
        _dictionary_table([0, 1, 2, 0], [" B ", "A", "B"]),
        source.parquet_path / "part.0.parquet",
    )
    pq.write_table(
        _dictionary_table([0, 1, 0], [" A ", "B"]),
        source.parquet_path / "part.1.parquet",
    )

    content = _scan_points_content(_read_parquet_source_inventory(source), max_batch_rows=2)

    assert content.value_table.to_pylist() == [
        {"value_id": 0, "value": "A", "n_points": 3},
        {"value_id": 1, "value": "B", "n_points": 4},
    ]

    counts = _normalized_value_counts(
        _dictionary_array(["A", None, " "], [0, 0]),
        column_name="gene",
        relative_path="part.0.parquet",
        row_group_index=0,
    )
    assert counts == {"A": 2}


@pytest.mark.parametrize("invalid_kind", ["coordinate", "empty_value", "dictionary_null"])
def test_scan_fails_without_requesting_another_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    invalid_kind: str,
) -> None:
    source = _source(tmp_path)
    if invalid_kind == "dictionary_null":
        pq.write_table(
            _dictionary_table([0, 0], ["A"]),
            source.parquet_path / "part.0.parquet",
        )
        invalid_value = _dictionary_array([None], [0])
        valid_value = _dictionary_array(["A"], [0])
    else:
        pq.write_table(
            _plain_table([0, 1], [0.0, 1.0], ["A", "B"]),
            source.parquet_path / "part.0.parquet",
        )
        invalid_value = pa.array([" " if invalid_kind == "empty_value" else "A"])
        valid_value = pa.array(["B"])

    invalid_x = pa.array([np.nan if invalid_kind == "coordinate" else 0.0])
    invalid_batch = pa.record_batch(
        [invalid_x, pa.array([0.0]), invalid_value],
        names=["x", "y", "gene"],
    )
    valid_batch = pa.record_batch(
        [pa.array([1.0]), pa.array([1.0]), valid_value],
        names=["x", "y", "gene"],
    )
    reader = _FailAfterFirstBatchReader(invalid_batch, valid_batch)
    monkeypatch.setattr(validation_module, "_open_parquet_content_file", lambda _path, _relative_path: reader)

    with pytest.raises(PointContentValidationError) as error:
        _scan_points_content(_read_parquet_source_inventory(source), max_batch_rows=1)

    expected_code = "invalid_coordinate_content" if invalid_kind == "coordinate" else "invalid_value_content"
    assert error.value.code == expected_code
    assert not reader.requested_after_first_batch


def test_scan_rejects_decoded_row_group_count_disagreement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source(tmp_path)
    pq.write_table(
        _plain_table([0, 1, 2], [0.0, 1.0, 2.0], ["A", "B", "C"]),
        source.parquet_path / "part.0.parquet",
    )
    truncated_batch = pa.record_batch(
        [pa.array([0.0, 1.0]), pa.array([0.0, 1.0]), pa.array(["A", "B"])],
        names=["x", "y", "gene"],
    )
    reader = _RecordingParquetFile([truncated_batch])
    monkeypatch.setattr(validation_module, "_open_parquet_content_file", lambda _path, _relative_path: reader)

    with pytest.raises(PointContentValidationError) as error:
        _scan_points_content(_read_parquet_source_inventory(source), max_batch_rows=2)

    assert error.value.code == "row_group_row_count_mismatch"


def test_scan_requests_one_bounded_selected_column_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source(tmp_path)
    pq.write_table(
        _plain_table([0, 1, 2, 3, 4], [0.0, 1.0, 2.0, 3.0, 4.0], ["A", "B", "A", "B", "A"]),
        source.parquet_path / "part.0.parquet",
        row_group_size=2,
    )
    readers: list[_RecordingParquetFile] = []

    def _open(path: Path, _relative_path: str) -> _RecordingParquetFile:
        reader = _RecordingParquetFile(parquet_file=pq.ParquetFile(path))
        readers.append(reader)
        return reader

    monkeypatch.setattr(validation_module, "_open_parquet_content_file", _open)

    _scan_points_content(_read_parquet_source_inventory(source), max_batch_rows=2)

    assert len(readers) == 1
    assert readers[0].calls == [
        {"row_groups": [0], "batch_size": 2, "columns": ["x", "y", "gene"]},
        {"row_groups": [1], "batch_size": 2, "columns": ["x", "y", "gene"]},
        {"row_groups": [2], "batch_size": 2, "columns": ["x", "y", "gene"]},
    ]


class _RecordingParquetFile:
    def __init__(
        self,
        batches: list[pa.RecordBatch] | None = None,
        *,
        parquet_file: pq.ParquetFile | None = None,
    ) -> None:
        self._batches = [] if batches is None else batches
        self._parquet_file = parquet_file
        self.calls: list[dict[str, object]] = []

    def iter_batches(self, **kwargs: object) -> Iterator[pa.RecordBatch]:
        self.calls.append(kwargs)
        if self._parquet_file is not None:
            return self._parquet_file.iter_batches(**kwargs)
        return iter(self._batches)


class _FailAfterFirstBatchReader:
    def __init__(self, first: pa.RecordBatch, second: pa.RecordBatch) -> None:
        self._first = first
        self._second = second
        self.requested_after_first_batch = False

    def iter_batches(self, **_kwargs: object) -> Iterator[pa.RecordBatch]:
        yield self._first
        self.requested_after_first_batch = True
        yield self._second
