from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from filelock import FileLock

import napari_harpy.core.multi_scale_cache_points_zarr.builder as builder_module
from napari_harpy.core.multi_scale_cache_points_zarr.builder import (
    _acquire_output_build_lock,
    _build_points_cache_zarr,
    _PointsCacheBuilderConfig,
)
from napari_harpy.core.multi_scale_cache_points_zarr.cache_format import (
    PUBLICATION_STATE_COMPLETE,
    _CatalogWriteSettings,
)
from napari_harpy.core.multi_scale_cache_points_zarr.source import (
    ParquetPointsSource,
    PointColumnSelection,
    PointsSourceValidationError,
    validate_parquet_points_source,
)
from napari_harpy.core.multi_scale_cache_points_zarr.source.models import ValidatedPointsSource
from napari_harpy.core.multi_scale_cache_points_zarr.storage.catalog_reader import _CatalogReader
from napari_harpy.core.multi_scale_cache_points_zarr.storage.models import _ZarrWriteSettings


def _validated_source(tmp_path: Path) -> ValidatedPointsSource:
    source = ParquetPointsSource(
        spatialdata_path=tmp_path / "source.zarr",
        points_name="transcripts",
        columns=PointColumnSelection(x="x", y="y", value="gene"),
    )
    source.parquet_path.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "x": pa.array([1.0, 3.0, 2.0, 4.0, 11.0, 12.0], type=pa.float64()),
                "y": pa.array([1.0, 2.0, 3.0, 4.0, 1.0, 2.0], type=pa.float64()),
                "gene": pa.array(["A", "B", "A", "A", "B", "B"]),
            }
        ),
        source.parquet_path / "part.0.parquet",
        row_group_size=2,
    )
    return validate_parquet_points_source(source, max_batch_rows=2)


def _config(*, overview_point_budget: int) -> _PointsCacheBuilderConfig:
    return _PointsCacheBuilderConfig(
        leaf_tile_size=10,
        overview_point_budget=overview_point_budget,
        dask_worker_count=2,
        zarr_settings=_ZarrWriteSettings(
            point_chunk_rows=2,
            point_shard_rows=4,
            range_chunk_rows=2,
            range_shard_rows=4,
            codec_id="zstd-v1",
        ),
        catalog_settings=_CatalogWriteSettings(
            manifest_chunk_rows=2,
            manifest_shard_rows=4,
            value_tile_chunk_rows=2,
            value_tile_shard_rows=4,
        ),
    )


def _build(
    validated: ValidatedPointsSource,
    tmp_path: Path,
    *,
    output_name: str = "transcripts_vis_zarr",
    overview_point_budget: int = 10,
) -> Path:
    temporary_root = tmp_path / "temporary"
    temporary_root.mkdir(exist_ok=True)
    return _build_points_cache_zarr(
        validated,
        output_path=tmp_path / output_name,
        temporary_directory_root=temporary_root,
        config=_config(overview_point_budget=overview_point_budget),
    )


def _generation_id(cache_root: Path) -> str:
    with _CatalogReader(cache_root) as reader:
        return reader.attributes.cache_generation_id


def _assert_publication_released(tmp_path: Path, output_name: str = "transcripts_vis_zarr") -> None:
    # Native lock pathnames may remain after release; successful non-blocking
    # acquisition, rather than path absence, proves builder ownership ended.
    with FileLock(tmp_path / f"{output_name}.build-lock", timeout=0):
        pass
    assert not list(tmp_path.glob(f"{output_name}.staging-*"))
    assert not list(tmp_path.glob(f"{output_name}.backup-*"))


def test_builder_publishes_complete_exact_only_generation(tmp_path: Path) -> None:
    validated = _validated_source(tmp_path)

    output = _build(validated, tmp_path)

    generation_id = _generation_id(output)
    assert output == tmp_path / "transcripts_vis_zarr"
    with _CatalogReader(output) as reader:
        reader.validate_contents()
        assert reader.attributes.cache_generation_id == generation_id
        assert reader.attributes.publication_state == PUBLICATION_STATE_COMPLETE
        assert tuple(level.kind for level in reader.attributes.levels) == ("exact",)
    assert not (output / "COMPLETED").exists()
    assert (tmp_path / "temporary").is_dir()
    assert not any((tmp_path / "temporary").iterdir())
    _assert_publication_released(tmp_path)


def test_builder_publishes_complete_multilevel_generation(tmp_path: Path) -> None:
    validated = _validated_source(tmp_path)

    output = _build(validated, tmp_path, overview_point_budget=2)

    with _CatalogReader(output) as reader:
        reader.validate_contents()
        levels = reader.attributes.levels
        assert tuple(level.kind for level in levels[:2]) == ("exact", "bridge")
        assert all(level.kind == "spatial" for level in levels[2:])
        assert levels[-1].point_count <= 2
    _assert_publication_released(tmp_path)


def test_builder_replaces_one_completed_generation(tmp_path: Path) -> None:
    validated = _validated_source(tmp_path)
    output = _build(validated, tmp_path)
    first_generation_id = _generation_id(output)

    repeated_output = _build(validated, tmp_path)

    assert repeated_output == output
    assert _generation_id(output) != first_generation_id
    _assert_publication_released(tmp_path)


def test_unlocked_coordination_path_does_not_report_contention(tmp_path: Path) -> None:
    lock_path = tmp_path / "transcripts_vis_zarr.build-lock"
    lock_path.write_text("left by an earlier process\n", encoding="utf-8")

    with _acquire_output_build_lock(lock_path):
        pass


def test_builder_rejects_held_lock_before_creating_staging(tmp_path: Path) -> None:
    validated = _validated_source(tmp_path)
    lock_path = tmp_path / "transcripts_vis_zarr.build-lock"

    with FileLock(lock_path):
        with pytest.raises(FileExistsError, match="currently owns the output lock"):
            _build(validated, tmp_path)

    assert not list(tmp_path.glob("transcripts_vis_zarr.staging-*"))
    assert not (tmp_path / "transcripts_vis_zarr").exists()


def test_builder_rejects_and_preserves_incomplete_existing_output(tmp_path: Path) -> None:
    validated = _validated_source(tmp_path)
    output = tmp_path / "transcripts_vis_zarr"
    output.mkdir()
    sentinel = output / "user-data.txt"
    sentinel.write_text("preserve me\n", encoding="utf-8")

    with pytest.raises(ValueError, match="group|Group|Zarr|zarr"):
        _build(validated, tmp_path)

    assert sentinel.read_text(encoding="utf-8") == "preserve me\n"
    _assert_publication_released(tmp_path)


def test_builder_final_source_guard_failure_removes_unpublished_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validated = _validated_source(tmp_path)
    original_guard = builder_module._require_parquet_source_unchanged
    guard_calls = 0

    def fail_second_guard(current: ValidatedPointsSource) -> None:
        nonlocal guard_calls
        guard_calls += 1
        if guard_calls == 2:
            raise PointsSourceValidationError("simulated source change", code="source_changed_after_validation")
        original_guard(current)

    monkeypatch.setattr(builder_module, "_require_parquet_source_unchanged", fail_second_guard)

    with pytest.raises(PointsSourceValidationError, match="simulated source change"):
        _build(validated, tmp_path)

    assert guard_calls == 2
    assert not (tmp_path / "transcripts_vis_zarr").exists()
    assert (tmp_path / "temporary").is_dir()
    assert not any((tmp_path / "temporary").iterdir())
    _assert_publication_released(tmp_path)


def test_builder_restores_completed_output_when_staging_install_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validated = _validated_source(tmp_path)
    output = _build(validated, tmp_path)
    original_generation_id = _generation_id(output)
    original_rename = Path.rename

    def fail_staging_install(self: Path, target: str | Path) -> Path:
        target_path = Path(target)
        if self.name.startswith("transcripts_vis_zarr.staging-") and target_path == output:
            raise OSError("simulated staging install failure")
        return original_rename(self, target)

    monkeypatch.setattr(Path, "rename", fail_staging_install)

    with pytest.raises(OSError, match="simulated staging install failure"):
        _build(validated, tmp_path)

    assert _generation_id(output) == original_generation_id
    _assert_publication_released(tmp_path)
