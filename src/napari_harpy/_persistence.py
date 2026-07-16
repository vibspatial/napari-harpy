from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from napari_harpy._app_state import HarpyAppState, TableStateChangedEvent
from napari_harpy.core.persistence import (
    TableComponentPath,
    TableComponentReloadResult,
    build_full_table_reload_paths,
    reload_table_components,
    resolve_table_path,
    write_table_components,
)
from napari_harpy.core.spatialdata import get_table, get_table_metadata

if TYPE_CHECKING:
    from anndata import AnnData
    from spatialdata import SpatialData


class PersistenceController:
    """Persist and reload explicit components of one selected SpatialData table."""

    def __init__(self, app_state: HarpyAppState | None = None) -> None:
        self._app_state = HarpyAppState() if app_state is None else app_state
        self._selected_spatialdata: SpatialData | None = None
        self._selected_labels_name: str | None = None
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
    def can_write_table_state(self) -> bool:
        """Return whether the selected table has local changes that can be written."""
        return self.can_sync and self.is_dirty

    @property
    def can_reload(self) -> bool:
        """Return whether the currently selected table can be reloaded from zarr."""
        return self.can_sync

    @property
    def is_dirty(self) -> bool:
        """Return whether the selected table has unsynced component mutations."""
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
        return None if store_path is None else store_path / table_path

    def bind(self, sdata: SpatialData | None, table_name: str | None, labels_name: str | None = None) -> None:
        """Bind persistence to the selected SpatialData table."""
        self._selected_spatialdata = sdata
        self._selected_labels_name = labels_name
        self._selected_table_name = table_name

    def write_table_state(self) -> str:
        """Persist the selected table's captured dirty components."""
        sdata = self._require_selected_spatialdata()
        table_name = self._require_selected_table_name()
        snapshot = self._app_state.snapshot_table_dirty_state(sdata, table_name)
        if not snapshot.paths:
            raise ValueError(f"Table `{table_name}` has no unsynced components to write.")
        result = write_table_components(
            sdata,
            table_name=table_name,
            paths=snapshot.paths,
        )
        self._app_state.acknowledge_table_write(
            snapshot,
            persisted_paths=result.persisted_paths,
        )
        return result.table_path

    def reload_table_components(
        self,
        paths: frozenset[TableComponentPath],
    ) -> TableComponentReloadResult:
        """Reload explicit, already-expanded component paths from disk."""
        sdata = self._require_selected_spatialdata(action="reloading from zarr")
        table_name = self._require_selected_table_name(action="reloading from zarr")
        result = reload_table_components(
            sdata,
            table_name=table_name,
            paths=paths,
            labels_name=self._selected_labels_name,
        )
        regions = (
            get_table_metadata(sdata, table_name).regions
            if any(path.component in ("obs", "obsm") for path in result.reloaded_paths)
            else ()
        )
        self._app_state.record_table_reload(
            TableStateChangedEvent(
                sdata=sdata,
                table_name=table_name,
                paths=result.reloaded_paths,
                regions=regions,
                change_kind="reloaded",
                source="persistence_controller",
            )
        )
        return result

    def reload_table_state(self) -> str:
        """Reload all supported table components through the selective API."""
        sdata = self._require_selected_spatialdata(action="reloading from zarr")
        table_name = self._require_selected_table_name(action="reloading from zarr")
        dirty_snapshot = self._app_state.snapshot_table_dirty_state(sdata, table_name)
        paths = build_full_table_reload_paths(
            sdata,
            table_name=table_name,
            extra_paths=dirty_snapshot.paths,
        )
        return self.reload_table_components(paths).table_path

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
