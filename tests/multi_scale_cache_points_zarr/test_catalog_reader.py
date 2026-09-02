from __future__ import annotations

from typing import Any

import pytest
import zarr
from zarr.storage import LocalStore

from napari_harpy.core.multi_scale_cache_points_zarr.cache_format import (
    _CatalogWriteSettings,
    _ValueMajorWriteSettings,
)
from napari_harpy.core.multi_scale_cache_points_zarr.storage.catalog_reader import _CatalogReader
from napari_harpy.core.multi_scale_cache_points_zarr.writer.catalog import _write_staged_cache_catalog

CatalogExactFixture = Any
_GENERATION_ID = "12345678-1234-5678-9234-567812345678"


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


def test_catalog_reader_streams_complete_logical_reconciliation(
    catalog_exact_fixture: CatalogExactFixture,
) -> None:
    _write_catalog(catalog_exact_fixture)

    with _CatalogReader(catalog_exact_fixture.staging_root) as reader:
        reader.validate_contents()


def test_catalog_reader_rejects_corrupt_value_tile_count(
    catalog_exact_fixture: CatalogExactFixture,
) -> None:
    _write_catalog(catalog_exact_fixture)
    with LocalStore(catalog_exact_fixture.staging_root, read_only=False) as store:
        root = zarr.open_group(store=store, mode="a", zarr_format=3, use_consolidated=False)
        root["value_tiles/n_points"][0] = 2

    with _CatalogReader(catalog_exact_fixture.staging_root) as reader:
        with pytest.raises(ValueError, match="manifest tile totals"):
            reader.validate_contents()


def test_catalog_reader_rejects_unknown_root_group(catalog_exact_fixture: CatalogExactFixture) -> None:
    _write_catalog(catalog_exact_fixture)
    with LocalStore(catalog_exact_fixture.staging_root, read_only=False) as store:
        root = zarr.open_group(store=store, mode="a", zarr_format=3, use_consolidated=False)
        root.create_group("unexpected")

    with pytest.raises(ValueError, match="unexpected"):
        with _CatalogReader(catalog_exact_fixture.staging_root):
            pass


def test_catalog_reader_missing_value_tile_shard_fails_strict_read(
    catalog_exact_fixture: CatalogExactFixture,
) -> None:
    _write_catalog(catalog_exact_fixture)
    shard_objects = [
        path for path in (catalog_exact_fixture.staging_root / "value_tiles/n_points/c").rglob("*") if path.is_file()
    ]
    assert shard_objects
    shard_objects[0].unlink()

    with _CatalogReader(catalog_exact_fixture.staging_root) as reader:
        with pytest.raises(Exception, match="chunk|Chunk|shard|Shard"):
            reader.validate_contents()


def test_catalog_reader_rejects_missing_value_major_level_array(
    catalog_exact_fixture: CatalogExactFixture,
) -> None:
    _write_catalog(catalog_exact_fixture)
    with LocalStore(catalog_exact_fixture.staging_root, read_only=False) as store:
        root = zarr.open_group(store=store, mode="a", zarr_format=3, use_consolidated=False)
        del root["value_major/level_0/location"]

    with pytest.raises(ValueError, match="Value-major level"):
        with _CatalogReader(catalog_exact_fixture.staging_root):
            pass


def test_catalog_reader_rejects_value_major_pointer_count_disagreement(
    catalog_exact_fixture: CatalogExactFixture,
) -> None:
    _write_catalog(catalog_exact_fixture)
    with LocalStore(catalog_exact_fixture.staging_root, read_only=False) as store:
        root = zarr.open_group(store=store, mode="a", zarr_format=3, use_consolidated=False)
        root["value_major/level_0/value_point_indptr"][1] = 2

    with _CatalogReader(catalog_exact_fixture.staging_root) as reader:
        with pytest.raises(ValueError, match="Value-major pointers"):
            reader.validate_contents()
