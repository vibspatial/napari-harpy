from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import anndata as ad
import pandas as pd
import zarr
from spatialdata.models import TableModel

from napari_harpy._annotation import USER_CLASS_COLORS_KEY
from napari_harpy._classifier import CLASSIFIER_CONFIG_KEY
from napari_harpy._spatialdata import SpatialDataAdapter

if TYPE_CHECKING:
    from anndata import AnnData
    from spatialdata import SpatialData


@dataclass(frozen=True)
class TableDiskSnapshot:
    """Partial table state reloaded directly from the backed zarr store."""

    table_name: str
    table_path: str
    obs: pd.DataFrame
    obsm: dict[str, Any]
    uns: dict[str, Any]


class PersistenceController:
    """Persist the selected SpatialData table back to its backed zarr store."""

    def __init__(self, spatialdata_adapter: SpatialDataAdapter | None = None) -> None:
        self._spatialdata_adapter = spatialdata_adapter or SpatialDataAdapter()
        self._selected_spatialdata: SpatialData | None = None
        self._selected_label_name: str | None = None
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
    def can_reload(self) -> bool:
        """Return whether the currently selected table can be reloaded from zarr."""
        return self.can_sync

    @property
    def selected_store_path(self) -> Path | None:
        """Return the backed zarr path for the current SpatialData selection."""
        if self._selected_spatialdata is None or self._selected_spatialdata.path is None:
            return None

        return Path(self._selected_spatialdata.path)

    @property
    def selected_table_store_path(self) -> Path | None:
        """Return the full on-disk zarr path for the current table selection."""
        if not self.can_reload:
            return None

        sdata = self._require_selected_spatialdata()
        table_name = self._require_selected_table_name()
        table = self._spatialdata_adapter.get_table(sdata, table_name)
        table_path = self._resolve_table_path(sdata, table, table_name)
        store_path = self.selected_store_path
        if store_path is None:
            return None

        return store_path / table_path

    def bind(self, sdata: SpatialData | None, table_name: str | None, label_name: str | None = None) -> None:
        """Bind persistence to the selected SpatialData table."""
        self._selected_spatialdata = sdata
        self._selected_label_name = label_name
        self._selected_table_name = table_name

    def read_table_snapshot_from_disk(self) -> TableDiskSnapshot:
        """Read the selected table's `obs`, `obsm`, and `uns` directly from zarr."""
        sdata = self._require_selected_spatialdata(action="reloading from zarr")
        table_name = self._require_selected_table_name(action="reloading from zarr")
        table = self._spatialdata_adapter.get_table(sdata, table_name)
        table_path = self._resolve_table_path(sdata, table, table_name)

        root = zarr.open_group(self.selected_store_path, mode="r", use_consolidated=False)
        table_group = root[table_path]
        obs = ad.io.read_elem(table_group["obs"])
        obsm = ad.io.read_elem(table_group["obsm"])
        uns = ad.io.read_elem(table_group["uns"])

        return TableDiskSnapshot(
            table_name=table_name,
            table_path=table_path,
            obs=obs,
            obsm=obsm,
            uns=uns,
        )

    def replace_selected_table(self, snapshot: TableDiskSnapshot) -> str:
        """Replace the selected in-memory table with a reloaded snapshot."""
        sdata = self._require_selected_spatialdata(action="reloading from zarr")
        table_name = self._require_selected_table_name(action="reloading from zarr")
        current_table = self._spatialdata_adapter.get_table(sdata, table_name)
        table_path = self._resolve_table_path(sdata, current_table, table_name)
        if snapshot.table_name != table_name or snapshot.table_path != table_path:
            raise ValueError(
                f"Reload snapshot targets `{snapshot.table_name}` at `{snapshot.table_path}`, "
                f"but the current selection is `{table_name}` at `{table_path}`."
            )

        current_table.obs = snapshot.obs
        # Shallow-copy the mapping to avoid aliasing the transient snapshot.
        # The arrays stored in `obsm` are not copied by `dict(...)`.
        current_table.obsm = dict(snapshot.obsm)
        current_table.uns = dict(snapshot.uns)
        TableModel.validate(current_table)
        return table_path

    def validate_reload_snapshot(self, snapshot: TableDiskSnapshot) -> None:
        """Validate that a disk snapshot can safely replace the selected in-memory table state."""
        sdata = self._require_selected_spatialdata(action="validating reload state")
        table_name = self._require_selected_table_name(action="validating reload state")
        current_table = self._spatialdata_adapter.get_table(sdata, table_name)
        table_path = self._resolve_table_path(sdata, current_table, table_name)

        if snapshot.table_name != table_name or snapshot.table_path != table_path:
            raise ValueError(
                f"Reload snapshot targets `{snapshot.table_name}` at `{snapshot.table_path}`, "
                f"but the current selection is `{table_name}` at `{table_path}`."
            )

        if not isinstance(snapshot.obs, pd.DataFrame):
            raise ValueError(f"Reload snapshot for table `{table_name}` has invalid `obs`; expected a DataFrame.")

        if not isinstance(snapshot.obsm, Mapping):
            raise ValueError(f"Reload snapshot for table `{table_name}` has invalid `obsm`; expected a mapping.")

        if not isinstance(snapshot.uns, Mapping):
            raise ValueError(f"Reload snapshot for table `{table_name}` has invalid `uns`; expected a mapping.")

        attrs = snapshot.uns.get(TableModel.ATTRS_KEY)
        if not isinstance(attrs, Mapping):
            raise ValueError(
                f"Reload snapshot for table `{table_name}` is missing `{TableModel.ATTRS_KEY}` metadata in `uns`."
            )

        current_attrs = current_table.uns.get(TableModel.ATTRS_KEY)
        if not isinstance(current_attrs, Mapping):
            raise ValueError(
                f"The current in-memory table `{table_name}` is missing `{TableModel.ATTRS_KEY}` metadata in `uns`."
            )

        region_key = attrs.get(TableModel.REGION_KEY_KEY)
        instance_key = attrs.get(TableModel.INSTANCE_KEY)
        if not region_key or not instance_key:
            raise ValueError(
                f"Reload snapshot for table `{table_name}` is missing required SpatialData table linkage metadata."
            )

        current_region_key = current_attrs.get(TableModel.REGION_KEY_KEY)
        current_instance_key = current_attrs.get(TableModel.INSTANCE_KEY)
        if region_key != current_region_key:
            raise ValueError(
                f"Cannot reload table `{table_name}`: disk snapshot uses region key `{region_key}` but the current "
                f"in-memory table uses `{current_region_key}`."
            )

        if instance_key != current_instance_key:
            raise ValueError(
                f"Cannot reload table `{table_name}`: disk snapshot uses instance key `{instance_key}` but the "
                f"current in-memory table uses `{current_instance_key}`."
            )

        if region_key not in snapshot.obs.columns:
            raise ValueError(
                f"Reload snapshot for table `{table_name}` is missing required obs column `{region_key}`."
            )

        if instance_key not in snapshot.obs.columns:
            raise ValueError(
                f"Reload snapshot for table `{table_name}` is missing required obs column `{instance_key}`."
            )

        if len(snapshot.obs) != current_table.n_obs:
            raise ValueError(
                f"Cannot reload table `{table_name}`: disk snapshot has {len(snapshot.obs)} rows but the in-memory "
                f"table has {current_table.n_obs}. Partial reload requires unchanged row identity and order."
            )

        if not snapshot.obs.index.equals(current_table.obs_names):
            raise ValueError(
                f"Cannot reload table `{table_name}`: disk snapshot obs_names do not exactly match the in-memory "
                "table. Partial reload requires unchanged row identity and order."
            )

        current_instance_values = current_table.obs[current_instance_key].astype("string")
        snapshot_instance_values = snapshot.obs[instance_key].astype("string")
        if not snapshot_instance_values.equals(current_instance_values):
            raise ValueError(
                f"Cannot reload table `{table_name}`: disk snapshot `{instance_key}` values do not exactly match "
                "the current in-memory table row by row."
            )

        for key, value in snapshot.obsm.items():
            shape = getattr(value, "shape", None)
            if shape is None or len(shape) == 0:
                raise ValueError(
                    f"Reload snapshot for table `{table_name}` has invalid `obsm[{key!r}]`; expected an array-like "
                    "value with a leading observation dimension."
                )
            if int(shape[0]) != len(snapshot.obs):
                raise ValueError(
                    f"Reload snapshot for table `{table_name}` has invalid `obsm[{key!r}]` with leading dimension "
                    f"{shape[0]}; expected {len(snapshot.obs)} rows."
                )

        if self._selected_label_name is not None:
            regions = _normalize_regions(attrs.get(TableModel.REGION_KEY))
            if self._selected_label_name not in regions:
                raise ValueError(
                    f"Cannot reload table `{table_name}` for segmentation `{self._selected_label_name}`: "
                    "the disk snapshot no longer annotates the selected segmentation."
                )

    def reload_table_state(self) -> str:
        """Reload the selected table's partial state from disk into the current SpatialData object."""
        snapshot = self.read_table_snapshot_from_disk()
        self.validate_reload_snapshot(snapshot)
        return self.replace_selected_table(snapshot)

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

    def _require_selected_spatialdata(self, *, action: str = "syncing to zarr") -> SpatialData:
        if self._selected_spatialdata is None:
            raise ValueError(f"Choose a SpatialData dataset before {action}.")

        if not self._selected_spatialdata.is_backed() or self._selected_spatialdata.path is None:
            raise ValueError("The selected SpatialData dataset is not backed by zarr.")

        return self._selected_spatialdata

    def _require_selected_table_name(self, *, action: str = "syncing to zarr") -> str:
        if self._selected_table_name is None:
            raise ValueError(f"Choose an annotation table before {action}.")

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


def _normalize_regions(region: str | list[str] | None) -> tuple[str, ...]:
    if region is None:
        return ()

    if isinstance(region, str):
        return (region,)

    return tuple(str(label_name) for label_name in region)
