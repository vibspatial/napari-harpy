from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import anndata as ad
import pandas as pd
import zarr
from spatialdata.models import TableModel

from napari_harpy.core.spatialdata import get_table, normalize_table_metadata

if TYPE_CHECKING:
    from anndata import AnnData
    from spatialdata import SpatialData


TableComponent = Literal["obs", "obsm", "uns"]
_MISSING = object()


@dataclass(frozen=True, order=True)
class TableComponentPath:
    """Identify one logical, independently tracked AnnData table component."""

    component: TableComponent
    keys: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.component not in ("obs", "obsm", "uns"):
            raise ValueError(f"Unsupported table component: {self.component!r}.")
        keys = tuple(self.keys)
        if not keys or any(not isinstance(key, str) or not key for key in keys):
            raise ValueError("Table component path keys must be non-empty strings.")
        if self.component in ("obs", "obsm") and len(keys) != 1:
            raise ValueError("obs and obsm paths must identify exactly one key.")
        object.__setattr__(self, "keys", keys)


@dataclass(frozen=True)
class TableComponentWriteResult:
    """Report the table store and logical paths persisted by one write."""

    table_path: str
    persisted_paths: frozenset[TableComponentPath]


@dataclass(frozen=True)
class TableComponentReloadResult:
    """Report the table store and logical path coverage restored from disk."""

    table_path: str
    reloaded_paths: frozenset[TableComponentPath]


def write_table_components(
    sdata: SpatialData,
    *,
    table_name: str,
    paths: frozenset[TableComponentPath],
) -> TableComponentWriteResult:
    """Persist selected logical table components through AnnData zarr encodings."""
    table, table_path, table_group = _resolve_backed_table_group(sdata, table_name, mode="r+")
    normalized_paths = _remove_redundant_uns_descendants(frozenset(paths))
    if not normalized_paths:
        raise ValueError("At least one table component path is required for persistence.")

    uns_paths = tuple(sorted(path for path in normalized_paths if path.component == "uns"))
    for path in uns_paths:
        _read_mapping_path(table.uns, path.keys)
        _validate_disk_parent_chain(table_group["uns"], path.keys[:-1], table_name=table_name)

    if any(path.component == "obs" for path in normalized_paths):
        ad.io.write_elem(table_group, "obs", table.obs)

    for path in sorted(path for path in normalized_paths if path.component == "obsm"):
        key = path.keys[0]
        if key in table.obsm:
            ad.io.write_elem(table_group["obsm"], key, table.obsm[key])
        elif key in table_group["obsm"]:
            del table_group["obsm"][key]

    for path in uns_paths:
        exists, value = _read_mapping_path(table.uns, path.keys)
        key = path.keys[-1]
        if exists:
            parent = _resolve_disk_parent_for_write(table_group["uns"], path.keys[:-1])
            ad.io.write_elem(parent, key, value)
            continue
        parent = _resolve_existing_disk_parent(table_group["uns"], path.keys[:-1])
        if parent is not None and key in parent:
            del parent[key]

    zarr.consolidate_metadata(sdata.path)
    return TableComponentWriteResult(table_path=table_path, persisted_paths=normalized_paths)


def build_full_table_reload_paths(
    sdata: SpatialData,
    *,
    table_name: str,
    extra_paths: frozenset[TableComponentPath] = frozenset(),
) -> frozenset[TableComponentPath]:
    """Expand a full supported table reload into explicit logical paths."""
    table, _table_path, table_group = _resolve_backed_table_group(sdata, table_name, mode="r")
    disk_obs = ad.io.read_elem(table_group["obs"])
    if not isinstance(disk_obs, pd.DataFrame):
        raise ValueError(f"Reloaded obs for table `{table_name}` must be a DataFrame.")

    paths = set(extra_paths)
    paths.update(TableComponentPath("obs", (str(column),)) for column in {*table.obs.columns, *disk_obs.columns})
    paths.update(TableComponentPath("obsm", (str(key),)) for key in {*table.obsm.keys(), *table_group["obsm"].keys()})
    paths.update(TableComponentPath("uns", (str(key),)) for key in {*table.uns.keys(), *table_group["uns"].keys()})
    if not paths:
        raise ValueError(f"Table `{table_name}` has no supported components to reload.")
    return _remove_redundant_uns_descendants(frozenset(paths))


def reload_table_components(
    sdata: SpatialData,
    *,
    table_name: str,
    paths: frozenset[TableComponentPath],
    labels_name: str | None = None,
) -> TableComponentReloadResult:
    """Reload selected table components in place from a backed zarr store.

    Named ``obsm`` keys and ``uns`` paths are reloaded individually, preserving
    unrelated entries. Requesting any ``obs`` path reloads the complete ``obs``
    dataframe so its columns and shared row index are restored consistently.
    """
    table, table_path, table_group = _resolve_backed_table_group(sdata, table_name, mode="r")
    normalized_paths = _remove_redundant_uns_descendants(frozenset(paths))
    if not normalized_paths:
        raise ValueError("At least one table component path is required for reload.")

    disk_obs = ad.io.read_elem(table_group["obs"])
    disk_attrs_exists, disk_attrs = _read_disk_path(table_group["uns"], (TableModel.ATTRS_KEY,))
    if not disk_attrs_exists:
        disk_attrs = _MISSING
    _validate_reload_binding(
        table,
        table_name=table_name,
        disk_obs=disk_obs,
        disk_attrs=disk_attrs,
        labels_name=labels_name,
    )

    reload_obs = any(path.component == "obs" for path in normalized_paths)
    obsm_values: dict[str, object] = {}
    for path in sorted(path for path in normalized_paths if path.component == "obsm"):
        key = path.keys[0]
        exists, value = _read_disk_path(table_group["obsm"], (key,))
        if exists:
            shape = getattr(value, "shape", None)
            if shape is None or len(shape) == 0 or int(shape[0]) != table.n_obs:
                raise ValueError(f"Reloaded obsm entry `{key}` for table `{table_name}` must have {table.n_obs} rows.")
            obsm_values[key] = value
        else:
            obsm_values[key] = _MISSING

    uns_values: dict[tuple[str, ...], object] = {}
    for path in sorted(path for path in normalized_paths if path.component == "uns"):
        exists, value = _read_disk_path(table_group["uns"], path.keys)
        uns_values[path.keys] = value if exists else _MISSING

    previous_obs = table.obs
    previous_obsm = dict(table.obsm)
    previous_uns = dict(table.uns)
    try:
        if reload_obs:
            table.obs = disk_obs
        if obsm_values:
            next_obsm = dict(table.obsm)
            for key, value in obsm_values.items():
                if value is _MISSING:
                    next_obsm.pop(key, None)
                else:
                    next_obsm[key] = value
            table.obsm = next_obsm
        if uns_values:
            next_uns = dict(table.uns)
            for keys, value in uns_values.items():
                next_uns = _replace_mapping_path(next_uns, keys, value)
            table.uns = next_uns
            if any(keys[0] == TableModel.ATTRS_KEY for keys in uns_values):
                normalize_table_metadata(table)
        TableModel.validate(table)
    except Exception:
        table.obs = previous_obs
        table.obsm = previous_obsm
        table.uns = previous_uns
        raise

    reloaded_paths = set(normalized_paths)
    if reload_obs:
        reloaded_paths.difference_update(path for path in tuple(reloaded_paths) if path.component == "obs")
        reloaded_paths.update(
            TableComponentPath("obs", (str(column),)) for column in {*previous_obs.columns, *disk_obs.columns}
        )
    return TableComponentReloadResult(
        table_path=table_path,
        reloaded_paths=frozenset(reloaded_paths),
    )


def write_table_prediction_state(
    sdata: SpatialData,
    *,
    table_name: str,
    uns_keys: Sequence[str] = (),
) -> str:
    """Compatibility wrapper for writing table obs and selected live uns entries."""
    table = get_table(sdata, table_name)
    paths = {TableComponentPath("obs", (str(column),)) for column in table.obs.columns}
    paths.update(TableComponentPath("uns", (key,)) for key in uns_keys if key in table.uns)
    return write_table_components(
        sdata,
        table_name=table_name,
        paths=frozenset(paths),
    ).table_path


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


def _resolve_backed_table_group(
    sdata: SpatialData,
    table_name: str,
    *,
    mode: Literal["r", "r+"],
) -> tuple[AnnData, str, Any]:
    if not sdata.is_backed() or sdata.path is None:
        raise ValueError("SpatialData must be backed by a zarr store for table persistence.")
    table = get_table(sdata, table_name)
    table_path = resolve_table_path(sdata, table, table_name)
    root = zarr.open_group(sdata.path, mode=mode, use_consolidated=False)
    return table, table_path, root[table_path]


def _read_mapping_path(mapping: Mapping[str, object], keys: tuple[str, ...]) -> tuple[bool, object]:
    current: object = mapping
    for index, key in enumerate(keys):
        if not isinstance(current, Mapping):
            parent = ".".join(keys[:index])
            raise ValueError(f"Table uns path `{parent}` must contain a mapping.")
        if key not in current:
            return False, _MISSING
        current = current[key]
    return True, current


def _validate_disk_parent_chain(group: Any, keys: tuple[str, ...], *, table_name: str) -> None:
    current = group
    for key in keys:
        if key not in current:
            return
        current = current[key]
        if not hasattr(current, "keys"):
            raise ValueError(f"Table `{table_name}` has a non-mapping encoded uns parent at `{key}`.")


def _resolve_disk_parent_for_write(group: Any, keys: tuple[str, ...]) -> Any:
    current = group
    for key in keys:
        if key not in current:
            ad.io.write_elem(current, key, {})
        current = current[key]
    return current


def _resolve_existing_disk_parent(group: Any, keys: tuple[str, ...]) -> Any | None:
    current = group
    for key in keys:
        if key not in current:
            return None
        current = current[key]
    return current


def _read_disk_path(group: Any, keys: tuple[str, ...]) -> tuple[bool, object]:
    current = group
    for key in keys:
        if not hasattr(current, "keys") or key not in current:
            return False, _MISSING
        current = current[key]
    return True, ad.io.read_elem(current)


def _replace_mapping_path(
    mapping: dict[str, object],
    keys: tuple[str, ...],
    value: object,
) -> dict[str, object]:
    result = dict(mapping)
    key = keys[0]
    if len(keys) == 1:
        if value is _MISSING:
            result.pop(key, None)
        else:
            result[key] = value
        return result

    child = result.get(key)
    if child is None:
        child_mapping: dict[str, object] = {}
    elif isinstance(child, Mapping):
        child_mapping = dict(child)
    else:
        raise ValueError(f"Table uns path `{key}` must contain a mapping.")
    result[key] = _replace_mapping_path(child_mapping, keys[1:], value)
    return result


def _remove_redundant_uns_descendants(
    paths: frozenset[TableComponentPath],
) -> frozenset[TableComponentPath]:
    result = set(paths)
    uns_paths = tuple(path for path in paths if path.component == "uns")
    for path in uns_paths:
        if any(
            other != path and len(other.keys) < len(path.keys) and path.keys[: len(other.keys)] == other.keys
            for other in uns_paths
        ):
            result.discard(path)
    return frozenset(result)


def _validate_reload_binding(
    table: AnnData,
    *,
    table_name: str,
    disk_obs: object,
    disk_attrs: object,
    labels_name: str | None,
) -> None:
    if not isinstance(disk_obs, pd.DataFrame):
        raise ValueError(f"Reloaded obs for table `{table_name}` must be a DataFrame.")
    if not isinstance(disk_attrs, Mapping):
        raise ValueError(f"Reloaded table `{table_name}` is missing `{TableModel.ATTRS_KEY}` metadata in uns.")
    current_attrs = table.uns.get(TableModel.ATTRS_KEY)
    if not isinstance(current_attrs, Mapping):
        raise ValueError(
            f"The current in-memory table `{table_name}` is missing `{TableModel.ATTRS_KEY}` metadata in uns."
        )

    disk_region_key = disk_attrs.get(TableModel.REGION_KEY_KEY)
    disk_instance_key = disk_attrs.get(TableModel.INSTANCE_KEY)
    current_region_key = current_attrs.get(TableModel.REGION_KEY_KEY)
    current_instance_key = current_attrs.get(TableModel.INSTANCE_KEY)
    if not disk_region_key or not disk_instance_key:
        raise ValueError(f"Reloaded table `{table_name}` is missing required SpatialData table linkage metadata.")
    if disk_region_key != current_region_key:
        raise ValueError(
            f"Cannot reload table `{table_name}`: disk uses region key `{disk_region_key}` but memory uses "
            f"`{current_region_key}`."
        )
    if disk_instance_key != current_instance_key:
        raise ValueError(
            f"Cannot reload table `{table_name}`: disk uses instance key `{disk_instance_key}` but memory uses "
            f"`{current_instance_key}`."
        )
    if disk_region_key not in disk_obs or disk_instance_key not in disk_obs:
        raise ValueError(f"Reloaded table `{table_name}` is missing required linkage columns in obs.")
    if len(disk_obs) != table.n_obs:
        raise ValueError(
            f"Cannot reload table `{table_name}`: disk has {len(disk_obs)} rows but memory has {table.n_obs}. "
            "Reload requires unchanged row identity and order."
        )

    current_regions = table.obs[current_region_key].astype("string").reset_index(drop=True)
    disk_regions = disk_obs[disk_region_key].astype("string").reset_index(drop=True)
    if not disk_regions.equals(current_regions):
        raise ValueError(
            f"Cannot reload table `{table_name}`: `{disk_region_key}` values do not exactly match row by row."
        )
    current_instances = table.obs[current_instance_key].astype("string").reset_index(drop=True)
    disk_instances = disk_obs[disk_instance_key].astype("string").reset_index(drop=True)
    if not disk_instances.equals(current_instances):
        raise ValueError(
            f"Cannot reload table `{table_name}`: `{disk_instance_key}` values do not exactly match row by row."
        )

    if labels_name is not None:
        regions = _normalize_regions(disk_attrs.get(TableModel.REGION_KEY))
        if labels_name not in regions:
            raise ValueError(
                f"Cannot reload table `{table_name}` for labels element `{labels_name}`: "
                "the disk table no longer annotates the selected labels element."
            )


def _normalize_regions(region: str | Sequence[str] | None) -> tuple[str, ...]:
    if region is None:
        return ()
    if isinstance(region, str):
        return (region,)
    return tuple(str(labels_name) for labels_name in region)
