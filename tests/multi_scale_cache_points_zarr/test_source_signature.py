from dataclasses import replace
from pathlib import Path

import pyarrow as pa

from napari_harpy.core.multi_scale_cache_points_zarr.source.models import (
    ParquetPointsSource,
    ParquetSourceFile,
    ParquetSourceRowGroup,
    PointColumnSelection,
)
from napari_harpy.core.multi_scale_cache_points_zarr.source.signature import (
    POINT_ID_POLICY,
    SOURCE_SIGNATURE_METHOD,
    _canonical_source_signature_bytes,
    build_source_signature,
)
from napari_harpy.core.multi_scale_cache_points_zarr.source.validation import _ParquetSourceInventory

_EXPECTED_CANONICAL_BYTES = (
    b'{"columns":[{"name":"x","nullable":false,"role":"x","type":{"bit_width":64,"kind":"float"}},'
    b'{"name":"y","nullable":true,"role":"y","type":{"bit_width":32,"kind":"integer","signed":false}},'
    b'{"name":"gene","nullable":true,"role":"value","type":{"index":{"bit_width":16,"kind":"integer",'
    b'"signed":true},"kind":"dictionary","ordered":false,"value":{"kind":"string","offset_width":32}}}],'
    b'"element_path":"points/transcripts","files":[{"modified_time_ns":null,"path":"nested/part.0.parquet",'
    b'"row_count":3,"row_groups":[{"compressed_size_bytes":100,"row_count":2},{"compressed_size_bytes":50,'
    b'"row_count":1}],"size_bytes":123}],"method":"harpy-parquet-source-inventory-sha256-v1","row_count":3}'
)
_EXPECTED_DIGEST = "eda799d9710e3bf18c37dfb2b543b6a1e19cf00d0a0bd62567495b6cee433117"


def _inventory(*, spatialdata_path: Path = Path("/source/example.zarr")) -> _ParquetSourceInventory:
    source = ParquetPointsSource(
        spatialdata_path=spatialdata_path,
        points_name="transcripts",
        columns=PointColumnSelection(x="x", y="y", value="gene"),
    )
    selected_schema = pa.schema(
        [
            pa.field("x", pa.float64(), nullable=False, metadata={b"excluded": b"field"}),
            pa.field("y", pa.uint32()),
            pa.field("gene", pa.dictionary(pa.int16(), pa.string(), ordered=False)),
        ],
        metadata={b"excluded": b"schema"},
    )
    source_file = ParquetSourceFile(
        relative_path="nested/part.0.parquet",
        size_bytes=123,
        modified_time_ns=None,
        row_count=3,
        row_offset=0,
        row_groups=(
            ParquetSourceRowGroup(row_count=2, compressed_size_bytes=100),
            ParquetSourceRowGroup(row_count=1, compressed_size_bytes=50),
        ),
    )
    return _ParquetSourceInventory(
        source=source,
        files=(source_file,),
        selected_schema=selected_schema,
        row_count=3,
    )


def test_source_signature_has_frozen_canonical_bytes_digest_and_methods() -> None:
    inventory = _inventory()

    assert _canonical_source_signature_bytes(inventory) == _EXPECTED_CANONICAL_BYTES
    assert build_source_signature(inventory) == _EXPECTED_DIGEST
    assert SOURCE_SIGNATURE_METHOD == "harpy-parquet-source-inventory-sha256-v1"
    assert POINT_ID_POLICY == "harpy-source-file-row-offset-uint64-v1"


def test_source_signature_excludes_absolute_host_path() -> None:
    inventory = _inventory()
    relocated = _inventory(spatialdata_path=Path("/another/host/copied.zarr"))

    assert build_source_signature(relocated) == build_source_signature(inventory)


def test_source_signature_changes_when_included_metadata_changes() -> None:
    inventory = _inventory()
    source_file = inventory.files[0]
    changed_row_groups = (
        ParquetSourceRowGroup(row_count=1, compressed_size_bytes=100),
        ParquetSourceRowGroup(row_count=2, compressed_size_bytes=50),
    )
    changed_schema = pa.schema(
        [
            pa.field("x", pa.float32(), nullable=False),
            inventory.selected_schema.field("y"),
            inventory.selected_schema.field("gene"),
        ]
    )
    variants = (
        replace(inventory, files=(replace(source_file, relative_path="renamed.parquet"),)),
        replace(inventory, files=(replace(source_file, size_bytes=124),)),
        replace(inventory, files=(replace(source_file, modified_time_ns=1),)),
        replace(inventory, files=(replace(source_file, row_groups=changed_row_groups),)),
        replace(inventory, selected_schema=changed_schema),
    )

    expected = build_source_signature(inventory)
    assert all(build_source_signature(variant) != expected for variant in variants)
