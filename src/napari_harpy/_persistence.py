from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import anndata as ad
import zarr

from napari_harpy._annotation import USER_CLASS_COLORS_KEY
from napari_harpy._classifier import CLASSIFIER_CONFIG_KEY
from napari_harpy._spatialdata import SpatialDataAdapter

if TYPE_CHECKING:
    from anndata import AnnData
    from spatialdata import SpatialData


class PersistenceController:
    """Persist the selected SpatialData table back to its backed zarr store."""

    def __init__(self, spatialdata_adapter: SpatialDataAdapter | None = None) -> None:
        self._spatialdata_adapter = spatialdata_adapter or SpatialDataAdapter()
        self._selected_spatialdata: SpatialData | None = None
        self._selected_table_name: str | None = None

    @property
    def can_sync(self) -> bool:
        """Return whether the currently selected table can be synced to zarr."""
        return (
            self._selected_spatialdata is not None
            and self._selected_table_name is not None
            and self._selected_spatialdata.is_backed()
        )

    @property
    def selected_store_path(self) -> Path | None:
        """Return the backed zarr path for the current SpatialData selection."""
        if self._selected_spatialdata is None or self._selected_spatialdata.path is None:
            return None

        return Path(self._selected_spatialdata.path)

    @property
    def selected_table_store_path(self) -> Path | None:
        """Return the full on-disk zarr path for the current table selection."""
        if not self.can_sync:
            return None

        sdata = self._require_selected_spatialdata()
        table_name = self._require_selected_table_name()
        table = self._spatialdata_adapter.get_table(sdata, table_name)
        table_path = self._resolve_table_path(sdata, table, table_name)
        store_path = self.selected_store_path
        if store_path is None:
            return None

        return store_path / table_path

    def bind(self, sdata: SpatialData | None, table_name: str | None) -> None:
        """Bind persistence to the selected SpatialData table."""
        self._selected_spatialdata = sdata
        self._selected_table_name = table_name

    def sync_table_state(self) -> str:
        """Write the current table annotation state back to the backed zarr store."""
        sdata = self._require_selected_spatialdata()
        table_name = self._require_selected_table_name()
        table = self._spatialdata_adapter.get_table(sdata, table_name)
        table_path = self._resolve_table_path(sdata, table, table_name)

        root = zarr.open_group(self.selected_store_path, mode="a", use_consolidated=False)
        table_group = root[table_path]
        ad.io.write_elem(table_group, "obs", table.obs)
        if USER_CLASS_COLORS_KEY in table.uns:
            ad.io.write_elem(table_group["uns"], USER_CLASS_COLORS_KEY, table.uns[USER_CLASS_COLORS_KEY])
        if CLASSIFIER_CONFIG_KEY in table.uns:
            ad.io.write_elem(table_group["uns"], CLASSIFIER_CONFIG_KEY, table.uns[CLASSIFIER_CONFIG_KEY])
        return table_path

    def _require_selected_spatialdata(self) -> SpatialData:
        if self._selected_spatialdata is None:
            raise ValueError("Choose a SpatialData dataset before syncing to zarr.")

        if not self._selected_spatialdata.is_backed() or self._selected_spatialdata.path is None:
            raise ValueError("The selected SpatialData dataset is not backed by zarr.")

        return self._selected_spatialdata

    def _require_selected_table_name(self) -> str:
        if self._selected_table_name is None:
            raise ValueError("Choose an annotation table before syncing to zarr.")

        return self._selected_table_name

    def _resolve_table_path(self, sdata: SpatialData, table: AnnData, table_name: str) -> str:
        table_paths = sdata.locate_element(table)
        if not table_paths:
            raise ValueError(f"Could not locate table `{table_name}` inside the backed SpatialData store.")

        if len(table_paths) > 1:
            raise ValueError(
                f"Table `{table_name}` resolved to multiple zarr paths: {table_paths}. A unique table path is required."
            )

        return table_paths[0]
