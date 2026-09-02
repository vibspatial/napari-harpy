from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import zarr
from zarr.storage import LocalStore

from napari_harpy.core.multi_scale_cache_points_zarr.build_plan import _plan_points_cache
from napari_harpy.core.multi_scale_cache_points_zarr.cache_format import (
    PUBLICATION_STATE_COMPLETE,
    _CatalogWriteSettings,
    _ValueMajorWriteSettings,
)
from napari_harpy.core.multi_scale_cache_points_zarr.models import _TileDescriptor
from napari_harpy.core.multi_scale_cache_points_zarr.storage.catalog_reader import _CatalogReader
from napari_harpy.core.multi_scale_cache_points_zarr.writer.bridge import (
    _BridgeWriterConfig,
    _write_bridge_level,
)
from napari_harpy.core.multi_scale_cache_points_zarr.writer.catalog import _write_staged_cache_catalog
from napari_harpy.core.multi_scale_cache_points_zarr.writer.spatial import (
    _SpatialWriterConfig,
    _write_spatial_levels,
)
from napari_harpy.core.multi_scale_cache_points_zarr.writer.staging_validation import (
    _ManifestBucket,
    _validate_staged_cache,
)

CatalogExactFixture = Any
_GENERATION_ID = "12345678-1234-5678-9234-567812345678"


def test_manifest_bucket_rejects_descriptor_from_another_bucket() -> None:
    descriptor = _TileDescriptor(
        level=0,
        bucket_id=1,
        bucket_tile_index=0,
        tile_x=0,
        tile_y=0,
        n_points=1,
    )

    with pytest.raises(ValueError, match="descriptor must belong to the stated bucket"):
        _ManifestBucket(
            level=0,
            bucket_id=0,
            descriptors=(descriptor,),
            manifest_indexes=np.array([0], dtype=np.uint64),
        )


def _write_catalog(fixture: CatalogExactFixture) -> None:
    _write_staged_cache_catalog(
        fixture.validated,
        fixture.plan,
        (fixture.result,),
        staging_root=fixture.staging_root,
        cache_generation_id=_GENERATION_ID,
        settings=_CatalogWriteSettings(
            manifest_chunk_rows=2,
            manifest_shard_rows=4,
            value_tile_chunk_rows=2,
            value_tile_shard_rows=4,
        ),
        value_major_settings=_ValueMajorWriteSettings(2, 4, 4),
        temporary_directory_root=fixture.temporary_root,
    )


def test_normal_staged_validation_accepts_complete_exact_generation(
    catalog_exact_fixture: CatalogExactFixture,
) -> None:
    _write_catalog(catalog_exact_fixture)

    _validate_staged_cache(catalog_exact_fixture.staging_root)


def test_normal_staged_validation_accepts_complete_multilevel_generation(
    catalog_exact_fixture: CatalogExactFixture,
) -> None:
    plan = _plan_points_cache(
        catalog_exact_fixture.validated,
        leaf_tile_size=10,
        overview_point_budget=2,
    )
    bridge = _write_bridge_level(
        catalog_exact_fixture.result,
        plan,
        staging_root=catalog_exact_fixture.staging_root,
        config=_BridgeWriterConfig(catalog_exact_fixture.zarr_settings),
    )
    spatial = _write_spatial_levels(
        bridge,
        plan,
        staging_root=catalog_exact_fixture.staging_root,
        config=_SpatialWriterConfig(catalog_exact_fixture.zarr_settings),
    )
    _write_staged_cache_catalog(
        catalog_exact_fixture.validated,
        plan,
        (catalog_exact_fixture.result, bridge, *spatial),
        staging_root=catalog_exact_fixture.staging_root,
        cache_generation_id=_GENERATION_ID,
        settings=_CatalogWriteSettings(2, 4, 2, 4),
        value_major_settings=_ValueMajorWriteSettings(2, 4, 4),
        temporary_directory_root=catalog_exact_fixture.temporary_root,
    )

    _validate_staged_cache(catalog_exact_fixture.staging_root)
    with _CatalogReader(catalog_exact_fixture.staging_root) as reader:
        for level, metadata in enumerate(reader.attributes.levels):
            assert reader.array(f"value_major/level_{level}/location").shape == (metadata.point_count, 2)
            pointer = reader.array(f"value_major/level_{level}/value_point_indptr")[:]
            assert int(pointer[0]) == 0
            assert int(pointer[-1]) == metadata.point_count


def test_normal_staged_validation_rejects_complete_publication_state(
    catalog_exact_fixture: CatalogExactFixture,
) -> None:
    _write_catalog(catalog_exact_fixture)
    with LocalStore(catalog_exact_fixture.staging_root, read_only=False) as store:
        root = zarr.open_group(store=store, mode="r+", zarr_format=3, use_consolidated=False)
        root.update_attributes({"publication_state": PUBLICATION_STATE_COMPLETE})

    with pytest.raises(ValueError, match="publication_state='staging'"):
        _validate_staged_cache(catalog_exact_fixture.staging_root)


def test_normal_staged_validation_compares_bucket_ranges_with_value_tiles(
    catalog_exact_fixture: CatalogExactFixture,
) -> None:
    _write_catalog(catalog_exact_fixture)
    bucket_path = catalog_exact_fixture.staging_root / catalog_exact_fixture.result.buckets[0].bucket_path
    with LocalStore(bucket_path, read_only=False) as store:
        root = zarr.open_group(store=store, mode="a", zarr_format=3, use_consolidated=False)
        root["ranges/row_count"][0] = int(root["ranges/row_count"][0]) + 1

    with pytest.raises(ValueError, match="ranges|coverage|boundary"):
        _validate_staged_cache(catalog_exact_fixture.staging_root)


def test_normal_staged_validation_checks_value_major_layout_without_decoding_locations(
    catalog_exact_fixture: CatalogExactFixture,
) -> None:
    _write_catalog(catalog_exact_fixture)
    location_objects = [
        path
        for path in (catalog_exact_fixture.staging_root / "value_major/level_0/location/c").rglob("*")
        if path.is_file()
    ]
    assert location_objects
    location_objects[0].unlink()

    # Normal publication validation deliberately verifies only the declared
    # location layout and compact pointer/count relationships. It does not
    # decode location shards; detecting missing or corrupt payload bytes
    # requires a separately invoked exhaustive payload validator.
    _validate_staged_cache(catalog_exact_fixture.staging_root)


def test_normal_staged_validation_does_not_decode_point_payload(
    catalog_exact_fixture: CatalogExactFixture,
) -> None:
    _write_catalog(catalog_exact_fixture)
    bucket_path = catalog_exact_fixture.staging_root / catalog_exact_fixture.result.buckets[0].bucket_path
    point_shards = [path for path in (bucket_path / "location/c").rglob("*") if path.is_file()]
    assert point_shards
    point_shards[0].unlink()

    _validate_staged_cache(catalog_exact_fixture.staging_root)


def test_normal_staged_validation_does_not_open_canonical_parquet(
    catalog_exact_fixture: CatalogExactFixture,
) -> None:
    _write_catalog(catalog_exact_fixture)
    source_files = list(catalog_exact_fixture.validated.source.parquet_path.glob("*.parquet"))
    assert source_files
    for source_file in source_files:
        source_file.unlink()

    _validate_staged_cache(catalog_exact_fixture.staging_root)
