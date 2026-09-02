from __future__ import annotations

import shutil
from dataclasses import dataclass
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import zarr
from zarr.errors import ChunkNotFoundError
from zarr.storage import LocalStore

from napari_harpy.core.multi_scale_cache_points_zarr.builder import (
    _build_points_cache_zarr,
    _PointsCacheBuilderConfig,
)
from napari_harpy.core.multi_scale_cache_points_zarr.cache_format import (
    PUBLICATION_STATE_STAGING,
    _CatalogWriteSettings,
    _ValueMajorWriteSettings,
)
from napari_harpy.core.multi_scale_cache_points_zarr.source import (
    ParquetPointsSource,
    PointColumnSelection,
    validate_parquet_points_source,
)
from napari_harpy.core.multi_scale_cache_points_zarr.storage._schema import value_major_point_indptr
from napari_harpy.core.multi_scale_cache_points_zarr.storage.catalog_reader import _CatalogReader
from napari_harpy.core.multi_scale_cache_points_zarr.storage.models import _ZarrWriteSettings
from napari_harpy.core.multi_scale_cache_points_zarr.storage.reader_cache import _BucketReaderCache
from napari_harpy.core.multi_scale_cache_points_zarr.writer.staging_validation import (
    _validate_complete_cache,
    _validate_staged_cache,
)
from napari_harpy.core.multi_scale_cache_points_zarr.writer.value_major import _RangeFragmentBatch


def _load_exhaustive_module() -> ModuleType:
    script_path = Path(__file__).parents[2] / "scripts/validate_multi_scale_cache_points_zarr_exhaustive.py"
    spec = spec_from_file_location("napari_harpy_exhaustive_validation_script", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load the exhaustive-validation developer script.")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


exhaustive_module = _load_exhaustive_module()


@dataclass(frozen=True)
class _ExhaustiveFixture:
    cache_root: Path
    temporary_root: Path


def test_global_coordinate_reconstruction_promotes_relative_coordinates_before_offset() -> None:
    finer_relative = np.asarray([0.0030720001], dtype=np.float32)
    coarser_relative = np.asarray(finer_relative + np.float32(512), dtype=np.float32)

    finer = exhaustive_module._reconstruct_global_coordinates(
        finer_relative,
        origin=0.0,
        tile_index=5,
        tile_size=512,
    )
    coarser = exhaustive_module._reconstruct_global_coordinates(
        coarser_relative,
        origin=0.0,
        tile_index=2,
        tile_size=1024,
    )

    assert finer.dtype == np.float64
    assert coarser.dtype == np.float64
    assert abs(float(finer[0] - coarser[0])) <= exhaustive_module._coordinate_tolerance(1024)


@pytest.fixture
def exhaustive_fixture(tmp_path: Path) -> _ExhaustiveFixture:
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
    validated = validate_parquet_points_source(source, max_batch_rows=2)
    builder_temporary_root = tmp_path / "builder-temporary"
    builder_temporary_root.mkdir()
    cache_root = _build_points_cache_zarr(
        validated,
        output_path=tmp_path / "transcripts_vis_zarr",
        temporary_directory_root=builder_temporary_root,
        config=_PointsCacheBuilderConfig(
            leaf_tile_size=10,
            overview_point_budget=2,
            dask_worker_count=2,
            target_points_per_bucket=2,
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
            value_major_settings=_ValueMajorWriteSettings(
                point_chunk_rows=2,
                point_shard_rows=4,
                construction_batch_points=4,
            ),
        ),
    )
    temporary_root = tmp_path / "exhaustive-temporary"
    temporary_root.mkdir()
    return _ExhaustiveFixture(cache_root=cache_root, temporary_root=temporary_root)


def test_publication_state_wrappers_are_strict(exhaustive_fixture: _ExhaustiveFixture) -> None:
    cache_root = exhaustive_fixture.cache_root
    _validate_complete_cache(cache_root)
    with pytest.raises(ValueError, match="publication_state='staging'"):
        _validate_staged_cache(cache_root)

    staging_root = cache_root.with_name("copied-staging")
    shutil.copytree(cache_root, staging_root)
    with LocalStore(staging_root, read_only=False) as store:
        root = zarr.open_group(store=store, mode="r+", zarr_format=3, use_consolidated=False)
        root.update_attributes({"publication_state": PUBLICATION_STATE_STAGING})

    _validate_staged_cache(staging_root)
    with pytest.raises(ValueError, match="publication_state='complete'"):
        _validate_complete_cache(staging_root)


def test_exhaustive_comparison_covers_all_levels_in_bounded_cross_chunk_batches(
    exhaustive_fixture: _ExhaustiveFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compared_levels: set[int] = set()
    compared_batch_sizes: list[int] = []
    original = exhaustive_module._read_fragment_locations

    def tracked_read(
        fragments: _RangeFragmentBatch,
        *,
        level: int,
        manifest_bucket_id: np.ndarray,
        readers: _BucketReaderCache,
    ) -> np.ndarray:
        compared_levels.add(level)
        compared_batch_sizes.append(fragments.point_count)
        return original(
            fragments,
            level=level,
            manifest_bucket_id=manifest_bucket_id,
            readers=readers,
        )

    monkeypatch.setattr(exhaustive_module, "_read_fragment_locations", tracked_read)

    exhaustive_module._validate_cache_exhaustive(
        exhaustive_fixture.cache_root,
        source=None,
        temporary_directory_root=exhaustive_fixture.temporary_root,
        value_major_comparison_batch_points=3,
    )

    with _CatalogReader(exhaustive_fixture.cache_root) as reader:
        expected_levels = set(range(len(reader.attributes.levels)))
        assert any(
            bool((np.diff(reader.array(value_major_point_indptr(level))[:]) == 0).any())
            for level in range(1, len(reader.attributes.levels))
        )
    assert compared_levels == expected_levels
    assert compared_batch_sizes
    assert max(compared_batch_sizes) == 3
    assert all(size <= 3 for size in compared_batch_sizes)
    assert not any(exhaustive_fixture.temporary_root.iterdir())


@pytest.mark.parametrize(
    "corruption",
    ["one_location", "equal_sized_blocks", "block_boundary"],
)
def test_exhaustive_comparison_rejects_semantic_location_corruption_ignored_by_normal_validation(
    exhaustive_fixture: _ExhaustiveFixture,
    corruption: str,
) -> None:
    with LocalStore(exhaustive_fixture.cache_root, read_only=False) as store:
        root = zarr.open_group(store=store, mode="r+", zarr_format=3, use_consolidated=False)
        location = root["value_major/level_0/location"]
        values = np.asarray(location[:], dtype=np.float32)
        pointer = np.asarray(root["value_major/level_0/value_point_indptr"][:], dtype=np.uint64)
        boundary = int(pointer[1])
        if corruption == "one_location":
            values[0, 0] += np.float32(0.5)
        elif corruption == "equal_sized_blocks":
            first = values[int(pointer[0]) : boundary].copy()
            second = values[boundary : int(pointer[2])].copy()
            assert len(first) == len(second) > 0
            values[int(pointer[0]) : boundary] = second
            values[boundary : int(pointer[2])] = first
        else:
            assert 0 < boundary < len(values)
            values[[boundary - 1, boundary]] = values[[boundary, boundary - 1]]
        location[:] = values

    # The mandatory validator intentionally checks sidecar layout, pointers,
    # and counts without decoding coordinate payloads.
    _validate_complete_cache(exhaustive_fixture.cache_root)
    with pytest.raises(ValueError, match="locations disagree with their tile-major sources"):
        exhaustive_module._validate_cache_exhaustive(
            exhaustive_fixture.cache_root,
            source=None,
            temporary_directory_root=exhaustive_fixture.temporary_root,
            value_major_comparison_batch_points=3,
        )
    assert not any(exhaustive_fixture.temporary_root.iterdir())


def test_exhaustive_comparison_rejects_missing_location_shard_ignored_by_normal_validation(
    exhaustive_fixture: _ExhaustiveFixture,
) -> None:
    location_chunks = exhaustive_fixture.cache_root / "value_major/level_0/location/c"
    shard_files = [path for path in location_chunks.rglob("*") if path.is_file()]
    assert shard_files
    shard_files[0].unlink()

    _validate_complete_cache(exhaustive_fixture.cache_root)
    with pytest.raises(ChunkNotFoundError, match="not found"):
        exhaustive_module._validate_cache_exhaustive(
            exhaustive_fixture.cache_root,
            source=None,
            temporary_directory_root=exhaustive_fixture.temporary_root,
            value_major_comparison_batch_points=3,
        )
    assert not any(exhaustive_fixture.temporary_root.iterdir())
