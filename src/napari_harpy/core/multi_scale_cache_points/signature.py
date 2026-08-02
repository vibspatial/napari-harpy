from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

import pyarrow as pa

from napari_harpy.core.multi_scale_cache_points.errors import ParquetMetadataValidationError

if TYPE_CHECKING:
    from napari_harpy.core.multi_scale_cache_points.validation import _ParquetSourceInventory


SOURCE_SIGNATURE_METHOD = "harpy-parquet-source-inventory-sha256-v1"
POINT_ID_POLICY = "harpy-source-file-row-offset-uint64-v1"


def build_source_signature(inventory: _ParquetSourceInventory) -> str:
    """Hash a canonical metadata snapshot, without reading Parquet data pages.

    This digest detects changes to the versioned source-inventory fields. It is
    not a content hash of the Parquet point data.
    """
    return hashlib.sha256(_canonical_source_signature_bytes(inventory)).hexdigest()


def _canonical_source_signature_bytes(inventory: _ParquetSourceInventory) -> bytes:
    payload = _source_signature_payload(inventory)
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _source_signature_payload(inventory: _ParquetSourceInventory) -> dict[str, object]:
    columns = []
    for role, name in (
        ("x", inventory.source.columns.x),
        ("y", inventory.source.columns.y),
        ("value", inventory.source.columns.value),
    ):
        field = inventory.selected_schema.field(name)
        columns.append(
            {
                "role": role,
                "name": field.name,
                "nullable": field.nullable,
                "type": _normalized_arrow_type(field.type),
            }
        )

    files = [
        {
            "path": source_file.relative_path,
            "size_bytes": source_file.size_bytes,
            "modified_time_ns": source_file.modified_time_ns,
            "row_count": source_file.row_count,
            "row_groups": [
                {
                    "row_count": row_group.row_count,
                    "compressed_size_bytes": row_group.compressed_size_bytes,
                }
                for row_group in source_file.row_groups
            ],
        }
        for source_file in inventory.files
    ]

    return {
        "method": SOURCE_SIGNATURE_METHOD,
        "element_path": inventory.source.element_path,
        "columns": columns,
        "files": files,
        "row_count": inventory.row_count,
    }


def _normalized_arrow_type(data_type: pa.DataType) -> dict[str, object]:
    if pa.types.is_signed_integer(data_type):
        return {
            "kind": "integer",
            "signed": True,
            "bit_width": data_type.bit_width,
        }
    if pa.types.is_unsigned_integer(data_type):
        return {
            "kind": "integer",
            "signed": False,
            "bit_width": data_type.bit_width,
        }
    if pa.types.is_floating(data_type):
        return {
            "kind": "float",
            "bit_width": data_type.bit_width,
        }
    if pa.types.is_string(data_type):
        return {
            "kind": "string",
            "offset_width": 32,
        }
    if pa.types.is_large_string(data_type):
        return {
            "kind": "string",
            "offset_width": 64,
        }
    if pa.types.is_dictionary(data_type):
        return {
            "kind": "dictionary",
            "index": _normalized_arrow_type(data_type.index_type),
            "value": _normalized_arrow_type(data_type.value_type),
            "ordered": data_type.ordered,
        }

    raise ParquetMetadataValidationError(
        f"Arrow type `{data_type}` cannot be represented by source-signature method "
        f"`{SOURCE_SIGNATURE_METHOD}`.",
        code="unsupported_source_signature_type",
    )
