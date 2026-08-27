from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import napari_harpy.core.multi_scale_cache_points_zarr.writer.exact as exact_module
from napari_harpy.core.multi_scale_cache_points_zarr.build_plan import (
    _plan_points_cache,
    _PointsCacheBuildPlan,
)
from napari_harpy.core.multi_scale_cache_points_zarr.source import (
    ParquetPointsSource,
    PointColumnSelection,
    validate_parquet_points_source,
)
from napari_harpy.core.multi_scale_cache_points_zarr.source.models import ValidatedPointsSource
from napari_harpy.core.multi_scale_cache_points_zarr.storage.bucket_reader import _BucketReader
from napari_harpy.core.multi_scale_cache_points_zarr.storage.bucket_validation import _validate_bucket
from napari_harpy.core.multi_scale_cache_points_zarr.storage.models import (
    _LevelWriteResult,
    _ZarrWriteSettings,
)
from napari_harpy.core.multi_scale_cache_points_zarr.writer.exact import (
    _annotate_source_partition,
    _ExactWriterConfig,
    _finalize_exact_bucket,
    _map_partition_value_ids,
    _read_and_annotate_row_group,
    _source_row_group_read_specs,
    _write_exact_level,
)


def _dictionary_array(dictionary: list[str], indices: list[int | None]) -> pa.DictionaryArray:
    return pa.DictionaryArray.from_arrays(
        pa.array(indices, type=pa.int8()),
        pa.array(dictionary, type=pa.string()),
    )


def _source(tmp_path: Path) -> ParquetPointsSource:
    source = ParquetPointsSource(
        spatialdata_path=tmp_path / "example.zarr",
        points_name="transcripts",
        columns=PointColumnSelection(x="x", y="y", value="gene"),
    )
    source.parquet_path.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "x": pa.array([1.0, 11.0, 3.0]),
                "y": pa.array([1.0, 1.0, 2.0]),
                "gene": _dictionary_array([" B ", "A"], [1, 0, 0]),
            }
        ),
        source.parquet_path / "part.0.parquet",
        row_group_size=2,
    )
    pq.write_table(
        pa.table(
            {
                "x": pa.array([2.0, 12.0, 4.0]),
                "y": pa.array([3.0, 2.0, 4.0]),
                "gene": _dictionary_array(["A", " B "], [0, 1, 0]),
            }
        ),
        source.parquet_path / "part.1.parquet",
        row_group_size=2,
    )
    return source


def _config(*, workers: int = 2, target_points_per_bucket: int = 2_000_000) -> _ExactWriterConfig:
    return _ExactWriterConfig(
        zarr_settings=_ZarrWriteSettings(
            point_chunk_rows=2,
            point_shard_rows=4,
            range_chunk_rows=2,
            range_shard_rows=4,
            codec_id="zstd-v1",
        ),
        dask_worker_count=workers,
        target_points_per_bucket=target_points_per_bucket,
    )


def _build_exact(
    source: ParquetPointsSource,
    tmp_path: Path,
    *,
    name: str,
    workers: int = 2,
    target_points_per_bucket: int = 2_000_000,
) -> tuple[_LevelWriteResult, ValidatedPointsSource, _PointsCacheBuildPlan, Path, Path]:
    validated = validate_parquet_points_source(source, max_batch_rows=2)
    plan = _plan_points_cache(validated, leaf_tile_size=10, overview_point_budget=10)
    staging = tmp_path / name
    temporary = tmp_path / f"{name}-temporary"
    staging.mkdir()
    temporary.mkdir()
    result = _write_exact_level(
        validated,
        plan,
        staging_root=staging,
        temporary_directory_root=temporary,
        config=_config(workers=workers, target_points_per_bucket=target_points_per_bucket),
    )
    return result, validated, plan, staging, temporary


def _decoded_points(
    result: _LevelWriteResult,
    plan: _PointsCacheBuildPlan,
    staging: Path,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    x_origin = plan.x_origin
    y_origin = plan.y_origin
    tile_size = plan.levels[0].tile_size
    for bucket in result.buckets:
        with _BucketReader(staging, level=0, bucket_id=bucket.bucket_id) as reader:
            for descriptor in bucket.tile_descriptors:
                payload = reader.read_construction_payload(descriptor)
                for index in range(payload.n_points):
                    rows.append(
                        {
                            "point_id": int(payload.point_id[index]),
                            "value_id": int(payload.value_id[index]),
                            "x": x_origin + descriptor.tile_x * tile_size + float(payload.x_rel[index]),
                            "y": y_origin + descriptor.tile_y * tile_size + float(payload.y_rel[index]),
                            "tile_x": descriptor.tile_x,
                            "tile_y": descriptor.tile_y,
                        }
                    )
    return sorted(rows, key=lambda row: row["point_id"])


def test_row_group_specs_preserve_file_and_within_file_offsets(tmp_path: Path) -> None:
    validated = validate_parquet_points_source(_source(tmp_path), max_batch_rows=2)

    specs = _source_row_group_read_specs(validated)

    assert [
        (spec.relative_path, spec.row_group_index, spec.expected_row_count, spec.point_id_start) for spec in specs
    ] == [
        ("part.0.parquet", 0, 2, 0),
        ("part.0.parquet", 1, 1, 2),
        ("part.1.parquet", 0, 2, 3),
        ("part.1.parquet", 1, 1, 5),
    ]


def test_annotation_reconstructs_fractional_coordinates_and_canonical_values() -> None:
    x = np.array([-511.99997, -0.00000001, 0.0, 511.99997], dtype=np.float64)
    y = np.array([-1023.75, -512.00000001, -512.0, -0.00000001], dtype=np.float64)
    table = pa.table({"x": x, "y": y, "gene": [" A "] * len(x)})

    annotated = _annotate_source_partition(
        table,
        expected_row_count=table.num_rows,
        point_id_start=10,
        x_column="x",
        y_column="y",
        value_column="gene",
        x_origin=-512.0,
        y_origin=-1024.0,
        tile_size=512,
        grid_width=2,
        grid_height=2,
        bucket_count=3,
        value_labels_by_id=("A",),
        validated_row_count=20,
        source_label="part.0.parquet:0",
    )

    reconstructed_x = -512.0 + annotated["tile_x"].to_numpy(dtype=np.float64) * 512 + annotated["x_rel"]
    reconstructed_y = -1024.0 + annotated["tile_y"].to_numpy(dtype=np.float64) * 512 + annotated["y_rel"]
    tolerance = float(np.spacing(np.float32(512)))
    np.testing.assert_allclose(reconstructed_x, x, rtol=0.0, atol=tolerance)
    np.testing.assert_allclose(reconstructed_y, y, rtol=0.0, atol=tolerance)
    assert 512.0 in annotated["x_rel"].to_numpy()
    assert 512.0 in annotated["y_rel"].to_numpy()
    assert annotated["point_id"].tolist() == [10, 11, 12, 13]
    assert annotated["value_id"].tolist() == [0, 0, 0, 0]


def test_value_mapping_preserves_dictionary_encoding_until_row_id_take() -> None:
    values = _dictionary_array([" B ", "A", "unused"], [1, 0, 0, 1])

    value_ids = _map_partition_value_ids(
        values,
        value_labels_by_id=("A", "B"),
        source_label="part.0.parquet:0",
    )

    np.testing.assert_array_equal(value_ids, np.array([0, 1, 1, 0], dtype=np.uint32))


def test_exact_writer_builds_deterministic_validated_zarr_from_row_groups(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source(tmp_path)
    first, validated, first_plan, first_staging, first_temporary = _build_exact(
        source,
        tmp_path,
        name="first",
        target_points_per_bucket=2,
    )
    original_specs = exact_module._source_row_group_read_specs
    monkeypatch.setattr(
        exact_module,
        "_source_row_group_read_specs",
        lambda value: tuple(reversed(original_specs(value))),
    )
    second, _, second_plan, second_staging, second_temporary = _build_exact(
        source,
        tmp_path,
        name="second",
        target_points_per_bucket=2,
    )

    assert first == second
    assert first.point_count == validated.row_count == 6
    assert tuple(bucket.bucket_id for bucket in first.buckets) == (1, 2)
    assert not (first_staging / "levels" / "level_0" / "bucket-000000.zarr").exists()
    assert list(first_temporary.iterdir()) == []
    assert list(second_temporary.iterdir()) == []
    assert not list(first_staging.rglob("*.parquet"))
    assert not list(second_staging.rglob("*.parquet"))
    expected_values = {row["value"]: row["value_id"] for row in validated.value_table.to_pylist()}
    for bucket in first.buckets:
        assert _validate_bucket(first_staging, level=0, bucket_id=bucket.bucket_id) == bucket
        with _BucketReader(first_staging, level=0, bucket_id=bucket.bucket_id) as reader:
            reader.load_lookup_index()
            for descriptor in bucket.tile_descriptors:
                complete = reader.read_construction_payload(descriptor)
                order = np.lexsort((complete.point_id, complete.value_id))
                np.testing.assert_array_equal(order, np.arange(complete.n_points))
                selected = reader.read_display_payload(
                    descriptor,
                    np.array([expected_values["A"]], dtype=np.uint32),
                )
                if selected is not None:
                    assert bool((selected.value_id == expected_values["A"]).all())

    expected_points = [
        {"point_id": 0, "value_id": expected_values["A"], "x": 1.0, "y": 1.0, "tile_x": 0, "tile_y": 0},
        {"point_id": 1, "value_id": expected_values["B"], "x": 11.0, "y": 1.0, "tile_x": 1, "tile_y": 0},
        {"point_id": 2, "value_id": expected_values["B"], "x": 3.0, "y": 2.0, "tile_x": 0, "tile_y": 0},
        {"point_id": 3, "value_id": expected_values["A"], "x": 2.0, "y": 3.0, "tile_x": 0, "tile_y": 0},
        {"point_id": 4, "value_id": expected_values["B"], "x": 12.0, "y": 2.0, "tile_x": 1, "tile_y": 0},
        {"point_id": 5, "value_id": expected_values["A"], "x": 4.0, "y": 4.0, "tile_x": 0, "tile_y": 0},
    ]
    assert _decoded_points(first, first_plan, first_staging) == expected_points
    assert _decoded_points(second, second_plan, second_staging) == expected_points


def test_exact_writer_reads_empty_row_group_and_plain_utf8_values(tmp_path: Path) -> None:
    source = ParquetPointsSource(
        spatialdata_path=tmp_path / "empty-row-group.zarr",
        points_name="transcripts",
        columns=PointColumnSelection(x="x", y="y", value="gene"),
    )
    source.parquet_path.mkdir(parents=True)
    schema = pa.schema((pa.field("x", pa.float64()), pa.field("y", pa.float64()), pa.field("gene", pa.string())))
    with pq.ParquetWriter(source.parquet_path / "part.0.parquet", schema) as writer:
        writer.write_table(pa.Table.from_pylist([], schema=schema))
        writer.write_table(pa.table({"x": [1.5, 2.5], "y": [3.5, 4.5], "gene": ["A", "B"]}))

    result, validated, plan, staging, _ = _build_exact(source, tmp_path, name="empty-row-group")

    assert [spec.expected_row_count for spec in _source_row_group_read_specs(validated)] == [0, 2]
    assert [row["point_id"] for row in _decoded_points(result, plan, staging)] == [0, 1]


@pytest.mark.parametrize("value", [None, "   ", "changed"])
def test_value_mapping_fails_closed_for_noncanonical_values(value: object) -> None:
    with pytest.raises((TypeError, ValueError), match="null|empty|absent"):
        _map_partition_value_ids(
            pa.array([value], type=pa.string()),
            value_labels_by_id=("A",),
            source_label="part.0.parquet:0",
        )


@pytest.mark.parametrize(
    ("values", "message"),
    [
        (_dictionary_array(["A"], [None]), "null"),
        (_dictionary_array(["   "], [0]), "empty"),
        (_dictionary_array(["changed"], [0]), "absent"),
    ],
)
def test_dictionary_value_mapping_fails_closed_for_noncanonical_values(
    values: pa.DictionaryArray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _map_partition_value_ids(
            values,
            value_labels_by_id=("A",),
            source_label="part.0.parquet:0",
        )


def test_row_group_read_rejects_changed_physical_row_count(tmp_path: Path) -> None:
    validated = validate_parquet_points_source(_source(tmp_path), max_batch_rows=2)
    spec = replace(_source_row_group_read_specs(validated)[0], expected_row_count=1)
    exact = _plan_points_cache(validated, leaf_tile_size=10, overview_point_budget=10).levels[0]

    with pytest.raises(ValueError, match="validation recorded 1"):
        _read_and_annotate_row_group(
            spec,
            source_root=validated.source.parquet_path,
            x_column="x",
            y_column="y",
            value_column="gene",
            x_origin=0.0,
            y_origin=0.0,
            tile_size=exact.tile_size,
            grid_width=exact.grid_width,
            grid_height=exact.grid_height,
            bucket_count=1,
            value_labels_by_id=tuple(validated.value_table["value"].to_pylist()),
            validated_row_count=validated.row_count,
        )


def test_exact_finalizer_rejects_rows_for_another_destination(tmp_path: Path) -> None:
    partition = exact_module._annotated_meta()
    partition.loc[0] = {
        "tile_x": 0,
        "tile_y": 0,
        "x_rel": 1.0,
        "y_rel": 2.0,
        "value_id": 0,
        "point_id": 0,
        "bucket_id": 1,
    }
    partition = partition.astype(exact_module._annotated_meta().dtypes.to_dict())

    with pytest.raises(ValueError, match="another bucket"):
        _finalize_exact_bucket(
            partition,
            bucket_id=0,
            staging_root=tmp_path,
            tile_size=10,
            grid_width=1,
            grid_height=1,
            validated_row_count=1,
            settings=_config().zarr_settings,
        )
    assert not (tmp_path / "levels").exists()


def test_exact_writer_rejects_unsupported_validated_policy(tmp_path: Path) -> None:
    source = _source(tmp_path)
    validated = validate_parquet_points_source(source, max_batch_rows=2)
    plan = _plan_points_cache(validated, leaf_tile_size=10, overview_point_budget=10)
    staging = tmp_path / "staging"
    temporary = tmp_path / "temporary"
    staging.mkdir()
    temporary.mkdir()

    with pytest.raises(ValueError, match="point-ID policy"):
        _write_exact_level(
            replace(validated, point_id_policy="unsupported"),
            plan,
            staging_root=staging,
            temporary_directory_root=temporary,
            config=_config(),
        )
    assert list(staging.iterdir()) == []


def test_exact_writer_propagates_finalizer_failure_and_cleans_shuffle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source(tmp_path)
    validated = validate_parquet_points_source(source, max_batch_rows=2)
    plan = _plan_points_cache(validated, leaf_tile_size=10, overview_point_budget=10)
    staging = tmp_path / "staging"
    temporary = tmp_path / "temporary"
    staging.mkdir()
    temporary.mkdir()

    def fail_finalizer(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("injected Exact finalizer failure")

    monkeypatch.setattr(exact_module, "_finalize_exact_bucket", fail_finalizer)
    with pytest.raises(RuntimeError, match="injected Exact finalizer failure"):
        _write_exact_level(
            validated,
            plan,
            staging_root=staging,
            temporary_directory_root=temporary,
            config=_config(workers=1),
        )

    assert list(temporary.iterdir()) == []


def test_exact_writer_rejects_existing_level_directory_before_reading(tmp_path: Path) -> None:
    source = _source(tmp_path)
    validated = validate_parquet_points_source(source, max_batch_rows=2)
    plan = _plan_points_cache(validated, leaf_tile_size=10, overview_point_budget=10)
    staging = tmp_path / "staging"
    temporary = tmp_path / "temporary"
    (staging / "levels" / "level_0").mkdir(parents=True)
    temporary.mkdir()

    with pytest.raises(FileExistsError, match="Exact-level output"):
        _write_exact_level(
            validated,
            plan,
            staging_root=staging,
            temporary_directory_root=temporary,
            config=_config(),
        )
