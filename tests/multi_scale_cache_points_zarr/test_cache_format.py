from __future__ import annotations

import copy
from typing import Any

import pytest

from napari_harpy.core.multi_scale_cache_points_zarr.cache_format import (
    BACKEND_IDENTIFIER,
    CACHE_SCHEMA_VERSION,
    _CatalogMetadata,
    _CatalogWriteSettings,
    _parse_cache_attributes,
)
from napari_harpy.core.multi_scale_cache_points_zarr.writer.catalog import _build_cache_attributes

_GENERATION_ID = "12345678-1234-5678-9234-567812345678"
CatalogExactFixture = Any


def _attributes(fixture: CatalogExactFixture, *, settings: _CatalogWriteSettings | None = None) -> dict[str, object]:
    result = fixture.result
    value_names = tuple(fixture.validated.value_table["value"].to_pylist())
    catalog_settings = settings or _CatalogWriteSettings()
    metadata = _CatalogMetadata(
        value_count=len(value_names),
        level_count=1,
        manifest_row_count=result.tile_count,
        value_tile_row_count=result.range_count,
        settings=catalog_settings,
    )
    return _build_cache_attributes(
        fixture.validated,
        fixture.plan,
        (result,),
        cache_generation_id=_GENERATION_ID,
        zarr_settings=fixture.zarr_settings,
        value_names=value_names,
        catalog_metadata=metadata,
    ).to_dict()


def test_cache_attributes_round_trip_exact_frozen_contract(catalog_exact_fixture: CatalogExactFixture) -> None:
    payload = _attributes(catalog_exact_fixture)

    parsed = _parse_cache_attributes(payload)

    assert parsed.to_dict() == payload
    assert payload["schema_version"] == CACHE_SCHEMA_VERSION
    assert payload["backend"]["identifier"] == BACKEND_IDENTIFIER  # type: ignore[index]
    assert parsed.catalog.value_count == 2
    assert parsed.catalog.manifest_row_count == 2
    assert parsed.catalog.value_tile_row_count == 3
    assert parsed.value_names == ("A", "B")


@pytest.mark.parametrize(
    ("corrupt", "message"),
    [
        (lambda value: value.update({"unexpected": 1}), "missing or unexpected"),
        (lambda value: value.update({"schema_version": "unknown"}), "schema version"),
        (lambda value: value.update({"cache_generation_id": "NOT-A-UUID"}), "UUID"),
        (lambda value: value["backend"].update({"identifier": "unknown"}), "backend"),
        (lambda value: value["catalog"].update({"manifest_row_order": ["tile_x"]}), "ordering"),
    ],
)
def test_cache_attribute_parser_fails_closed(
    catalog_exact_fixture: CatalogExactFixture,
    corrupt: object,
    message: str,
) -> None:
    payload = copy.deepcopy(_attributes(catalog_exact_fixture))
    corrupt(payload)  # type: ignore[operator]

    with pytest.raises(ValueError, match=message):
        _parse_cache_attributes(payload)


def test_catalog_settings_require_positive_aligned_bounded_layouts() -> None:
    with pytest.raises(ValueError, match="multiple"):
        _CatalogWriteSettings(manifest_chunk_rows=3, manifest_shard_rows=4)
    with pytest.raises(ValueError, match="value_tile_chunk_rows"):
        _CatalogWriteSettings(value_tile_chunk_rows=0)
