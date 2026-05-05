from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import anndata as ad
import zarr

from napari_harpy._spatialdata import get_table

if TYPE_CHECKING:
    from anndata import AnnData
    from spatialdata import SpatialData


def write_table_prediction_state(
    sdata: SpatialData,
    *,
    table_name: str,
    uns_keys: Sequence[str] = (),
) -> str:
    """Write table `obs` and selected `uns` entries for a backed table."""
    if not sdata.is_backed() or sdata.path is None:
        raise ValueError("SpatialData must be backed by a zarr store before writing prediction state.")

    table = get_table(sdata, table_name)
    table_path = resolve_table_path(sdata, table, table_name)

    root = zarr.open_group(sdata.path, mode="r+", use_consolidated=False)
    table_group = root[table_path]
    ad.io.write_elem(table_group, "obs", table.obs)

    for key in uns_keys:
        if key in table.uns:
            ad.io.write_elem(table_group["uns"], key, table.uns[key])

    return table_path


def resolve_table_path(sdata: SpatialData, table: AnnData, table_name: str) -> str:
    """Return the unique zarr path for `table` inside a backed SpatialData object."""
    table_paths = sdata.locate_element(table)
    if not table_paths:
        raise ValueError(f"Could not locate table `{table_name}` inside the backed SpatialData store.")

    if len(table_paths) > 1:
        raise ValueError(
            f"Table `{table_name}` resolved to multiple zarr paths: {table_paths}. A unique table path is required."
        )

    return table_paths[0]
