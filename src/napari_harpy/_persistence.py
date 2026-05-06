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
from napari_harpy._app_state import HarpyAppState
from napari_harpy._classifier_core import CLASSIFIER_APPLY_CONFIG_KEY, CLASSIFIER_CONFIG_KEY, PRED_CLASS_COLORS_KEY
from napari_harpy._persistence_core import resolve_table_path, write_table_prediction_state
from napari_harpy._spatialdata import get_table, normalize_table_metadata

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

    def __init__(self, app_state: HarpyAppState | None = None) -> None:
        self._app_state = HarpyAppState() if app_state is None else app_state
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
    def is_dirty(self) -> bool:
        """Return whether the current selected table has unsynced local changes."""
        return self._app_state.is_table_dirty(self._selected_spatialdata, self._selected_table_name)

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
        table = get_table(sdata, table_name)
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

    def mark_dirty(self) -> None:
        """Mark the current selected table as having unsynced local changes."""
        self._app_state.mark_table_dirty(self._selected_spatialdata, self._selected_table_name)

    def clear_dirty(self) -> None:
        """Clear the unsynced-local-changes marker for the current selected table."""
        self._app_state.clear_table_dirty(self._selected_spatialdata, self._selected_table_name)

    def read_table_snapshot_from_disk(self) -> TableDiskSnapshot:
        """Read the selected table's `obs`, `obsm`, and `uns` directly from zarr."""
        sdata = self._require_selected_spatialdata(action="reloading from zarr")
        table_name = self._require_selected_table_name(action="reloading from zarr")
        table = get_table(sdata, table_name)
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

    def replace_selected_table_state(self, snapshot: TableDiskSnapshot) -> str:
        """Replace the selected table's in-memory state from a reloaded snapshot.

        This is an in-place partial reload of ``obs``, ``obsm``, and ``uns`` on
        the currently selected table object. We intentionally keep the existing
        ``AnnData`` instance instead of swapping in a second full table object.
        """
        sdata = self._require_selected_spatialdata(action="reloading from zarr")
        table_name = self._require_selected_table_name(action="reloading from zarr")
        current_table = get_table(sdata, table_name)
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
        normalize_table_metadata(current_table)
        TableModel.validate(current_table)
        return table_path

    def validate_reload_snapshot(self, snapshot: TableDiskSnapshot) -> None:
        """Validate that a disk snapshot can safely replace the selected in-memory table state."""
        sdata = self._require_selected_spatialdata(action="validating reload state")
        current_table_name = self._require_selected_table_name(action="validating reload state")
        current_table = get_table(sdata, current_table_name)
        current_table_path = self._resolve_table_path(sdata, current_table, current_table_name)

        if snapshot.table_name != current_table_name or snapshot.table_path != current_table_path:
            raise ValueError(
                f"Reload snapshot targets `{snapshot.table_name}` at `{snapshot.table_path}`, "
                f"but the current selection is `{current_table_name}` at `{current_table_path}`."
            )

        if not isinstance(snapshot.obs, pd.DataFrame):
            raise ValueError(
                f"Reload snapshot for table `{current_table_name}` has invalid `obs`; expected a DataFrame."
            )

        if not isinstance(snapshot.obsm, Mapping):
            raise ValueError(
                f"Reload snapshot for table `{current_table_name}` has invalid `obsm`; expected a mapping."
            )

        if not isinstance(snapshot.uns, Mapping):
            raise ValueError(f"Reload snapshot for table `{current_table_name}` has invalid `uns`; expected a mapping.")

        snapshot_attrs = snapshot.uns.get(TableModel.ATTRS_KEY)
        if not isinstance(snapshot_attrs, Mapping):
            raise ValueError(
                f"Reload snapshot for table `{current_table_name}` is missing `{TableModel.ATTRS_KEY}` metadata in `uns`."
            )

        current_attrs = current_table.uns.get(TableModel.ATTRS_KEY)
        if not isinstance(current_attrs, Mapping):
            raise ValueError(
                f"The current in-memory table `{current_table_name}` is missing `{TableModel.ATTRS_KEY}` metadata in `uns`."
            )

        snapshot_region_key = snapshot_attrs.get(TableModel.REGION_KEY_KEY)
        snapshot_instance_key = snapshot_attrs.get(TableModel.INSTANCE_KEY)
        if not snapshot_region_key or not snapshot_instance_key:
            raise ValueError(
                f"Reload snapshot for table `{current_table_name}` is missing required SpatialData table linkage metadata."
            )

        current_region_key = current_attrs.get(TableModel.REGION_KEY_KEY)
        current_instance_key = current_attrs.get(TableModel.INSTANCE_KEY)
        if snapshot_region_key != current_region_key:
            raise ValueError(
                f"Cannot reload table `{current_table_name}`: disk snapshot uses region key `{snapshot_region_key}` but the "
                f"current in-memory table uses `{current_region_key}`."
            )

        if snapshot_instance_key != current_instance_key:
            raise ValueError(
                f"Cannot reload table `{current_table_name}`: disk snapshot uses instance key `{snapshot_instance_key}` but the "
                f"current in-memory table uses `{current_instance_key}`."
            )

        if snapshot_region_key not in snapshot.obs.columns:
            raise ValueError(
                f"Reload snapshot for table `{current_table_name}` is missing required obs column `{snapshot_region_key}`."
            )

        if snapshot_instance_key not in snapshot.obs.columns:
            raise ValueError(
                f"Reload snapshot for table `{current_table_name}` is missing required obs column `{snapshot_instance_key}`."
            )

        if len(snapshot.obs) != current_table.n_obs:
            raise ValueError(
                f"Cannot reload table `{current_table_name}`: disk snapshot has {len(snapshot.obs)} rows but the in-memory "
                f"table has {current_table.n_obs}. Partial reload requires unchanged row identity and order."
            )

        current_region_values = current_table.obs[current_region_key].astype("string").reset_index(drop=True)
        snapshot_region_values = snapshot.obs[snapshot_region_key].astype("string").reset_index(drop=True)
        if not snapshot_region_values.equals(current_region_values):
            raise ValueError(
                f"Cannot reload table `{current_table_name}`: disk snapshot `{snapshot_region_key}` values do not "
                "exactly match the current in-memory table row by row."
            )

        current_instance_values = current_table.obs[current_instance_key].astype("string").reset_index(drop=True)
        snapshot_instance_values = snapshot.obs[snapshot_instance_key].astype("string").reset_index(drop=True)
        if not snapshot_instance_values.equals(current_instance_values):
            raise ValueError(
                f"Cannot reload table `{current_table_name}`: disk snapshot `{snapshot_instance_key}` values do not "
                "exactly match the current in-memory table row by row."
            )

        for key, value in snapshot.obsm.items():
            shape = getattr(value, "shape", None)
            if shape is None or len(shape) == 0:
                raise ValueError(
                    f"Reload snapshot for table `{current_table_name}` has invalid `obsm[{key!r}]`; expected an array-like "
                    "value with a leading observation dimension."
                )
            if int(shape[0]) != len(snapshot.obs):
                raise ValueError(
                    f"Reload snapshot for table `{current_table_name}` has invalid `obsm[{key!r}]` with leading dimension "
                    f"{shape[0]}; expected {len(snapshot.obs)} rows."
                )

        if self._selected_label_name is not None:
            regions = _normalize_regions(snapshot_attrs.get(TableModel.REGION_KEY))
            if self._selected_label_name not in regions:
                raise ValueError(
                    f"Cannot reload table `{current_table_name}` for segmentation `{self._selected_label_name}`: "
                    "the disk snapshot no longer annotates the selected segmentation."
                )

    def reload_table_state(self) -> str:
        """Reload the selected table's partial state from disk into the current SpatialData object."""
        snapshot = self.read_table_snapshot_from_disk()
        # Reload uses a validated in-place update of `obs`, `obsm`, and `uns`.
        # We intentionally do not build a second full AnnData object for a
        # strictly atomic swap because this code path is meant to avoid that cost.
        self.validate_reload_snapshot(snapshot)
        table_path = self.replace_selected_table_state(snapshot)
        self.clear_dirty()
        return table_path

    def write_table_state(self) -> str:
        """Write the current table annotation state back to the backed zarr store."""
        sdata = self._require_selected_spatialdata()
        table_name = self._require_selected_table_name()
        table_path = write_table_prediction_state(
            sdata,
            table_name=table_name,
            uns_keys=(
                USER_CLASS_COLORS_KEY,
                PRED_CLASS_COLORS_KEY,
                CLASSIFIER_CONFIG_KEY,
                CLASSIFIER_APPLY_CONFIG_KEY,
            ),
        )
        self.clear_dirty()
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
        return resolve_table_path(sdata, table, table_name)

def _normalize_regions(region: str | list[str] | None) -> tuple[str, ...]:
    if region is None:
        return ()

    if isinstance(region, str):
        return (region,)

    return tuple(str(label_name) for label_name in region)
